"""
Resilience module for the Unified AI Abstraction Layer.

Provides retry logic with exponential backoff for handling
transient failures from AI providers.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union
import structlog

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    before_sleep_log,
)

from inflow_unified_ai.providers.base import (
    ProviderError,
    RateLimitError,
    AuthenticationError,
)

logger = structlog.get_logger(__name__)

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_WAIT = 1  # seconds
DEFAULT_MAX_WAIT = 60  # seconds


def _should_retry(exception: BaseException) -> bool:
    """
    Determine if an exception should trigger a retry.
    
    We retry on:
    - Rate limit errors (429)
    - Server errors (5xx)
    - Timeout errors
    - Connection errors
    
    We do NOT retry on:
    - Authentication errors
    - Invalid request errors
    - Model not found errors
    
    Args:
        exception: The exception to check
        
    Returns:
        True if should retry
    """
    # Don't retry authentication errors
    if isinstance(exception, AuthenticationError):
        return False
    
    # Retry rate limit errors
    if isinstance(exception, RateLimitError):
        return True
    
    # Check for transient errors
    if isinstance(exception, ProviderError):
        status_code = exception.status_code
        if status_code:
            # Retry server errors (5xx) and rate limits (429)
            return status_code >= 500 or status_code == 429
    
    # Retry generic connection/timeout errors
    error_message = str(exception).lower()
    transient_keywords = ["timeout", "connection", "temporary", "unavailable"]
    return any(kw in error_message for kw in transient_keywords)


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying after failure",
        attempt=retry_state.attempt_number,
        exception=str(exception) if exception else None,
        wait_time=retry_state.next_action.sleep if retry_state.next_action else None,
    )


class RetryConfig:
    """
    Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to wait times
    """
    
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_wait: float = DEFAULT_MIN_WAIT,
        max_wait: float = DEFAULT_MAX_WAIT,
        exponential_base: float = 2,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_wait_strategy(self):
        """Get the tenacity wait strategy."""
        if self.jitter:
            return wait_random_exponential(
                multiplier=self.min_wait,
                max=self.max_wait,
            )
        return wait_exponential(
            multiplier=self.min_wait,
            max=self.max_wait,
            exp_base=self.exponential_base,
        )


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


def with_retry(
    config: Optional[RetryConfig] = None,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """
    Decorator to add retry logic to a function.
    
    Args:
        config: Retry configuration (uses default if None)
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(RetryConfig(max_retries=5))
        def call_api():
            ...
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    
    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            retryer = Retrying(
                stop=stop_after_attempt(cfg.max_retries + 1),
                wait=cfg.get_wait_strategy(),
                retry=retry_if_exception_type(retry_on) if retry_on else _should_retry,
                before_sleep=_log_retry,
                reraise=True,
            )
            return retryer(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(cfg.max_retries + 1),
                wait=cfg.get_wait_strategy(),
                retry=retry_if_exception_type(retry_on) if retry_on else _should_retry,
                before_sleep=_log_retry,
                reraise=True,
            )
            async for attempt in retryer:
                with attempt:
                    return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def retry_with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    min_wait: float = DEFAULT_MIN_WAIT,
    max_wait: float = DEFAULT_MAX_WAIT,
) -> Callable[[F], F]:
    """
    Simplified decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        
    Returns:
        Decorated function
        
    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def call_api():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        min_wait=min_wait,
        max_wait=max_wait,
        jitter=True,
    )
    return with_retry(config)


class CircuitBreaker:
    """
    Simple circuit breaker implementation.
    
    When failures exceed the threshold, the circuit "opens" and
    subsequent calls fail fast without attempting the operation.
    After a timeout, the circuit becomes "half-open" and allows
    a test call through.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
    
    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state
    
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        if self._state == self.CLOSED:
            return True
        
        if self._state == self.OPEN:
            # Check if recovery timeout has passed
            import time
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    return True
            return False
        
        # HALF_OPEN state
        return self._half_open_calls < self.half_open_max_calls
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == self.HALF_OPEN:
            self._state = self.CLOSED
            self._failure_count = 0
            logger.info("Circuit breaker closed after successful recovery")
    
    def record_failure(self) -> None:
        """Record a failed call."""
        import time
        
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == self.HALF_OPEN:
            self._state = self.OPEN
            logger.warning("Circuit breaker reopened after failure in half-open state")
        elif self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                "Circuit breaker opened",
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
            )
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
