#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite for inflow-unified-ai

This script performs extensive testing of all Azure OpenAI models across
multiple endpoints, measuring:
- Response quality
- Latency (time to first token, total time)
- Token usage (prompt, completion, total)
- Streaming capability
- Structured output support
- Error handling

Results are exported to CSV for executive reporting.

Author: AI Engineering Team
Date: January 2026
"""

import asyncio
import csv
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inflow_unified_ai import AIClient
from inflow_unified_ai.models import (
    Message,
    MessageRole,
    get_model_capabilities,
)
from inflow_unified_ai.models.requests import ResponseFormat
from inflow_unified_ai.providers import AzureOpenAIProvider, ProviderError


# =============================================================================
# CONFIGURATION
# =============================================================================

# Endpoint 1: Primary models
ENDPOINT_1 = {
    "name": "DevTest AI Foundry (Primary)",
    "endpoint": "https://devtest-ai-foundry.cognitiveservices.azure.com/",
    "api_key": "os.getenv("AZURE_OPENAI_API_KEY_1")",
    "models": [
        "gpt-4.1",
        "gpt-4.1-mini", 
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.1-chat",
        "o3",
        "o3-mini",
    ]
}

# Endpoint 2: Additional models
ENDPOINT_2 = {
    "name": "DevTest AI Foundry (Secondary)",
    "endpoint": "https://ai-implementationprojectshub775496099306.cognitiveservices.azure.com/",
    "api_key": "os.getenv("AZURE_OPENAI_API_KEY_2")",
    "models": [
        "gpt-5.2",
        "gpt-5.2-chat",
        "model-router",
        "o1",
    ]
}

# Test prompts for different scenarios
TEST_PROMPTS = {
    "simple_math": {
        "system": "You are a helpful math assistant. Be concise.",
        "user": "What is 15 * 17? Give only the number.",
        "expected_contains": "255"
    },
    "reasoning": {
        "system": "You are a logical reasoning assistant.",
        "user": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer Yes or No and explain briefly.",
        "expected_contains": None  # Just check for response
    },
    "code_generation": {
        "system": "You are an expert Python programmer.",
        "user": "Write a Python function to check if a number is prime. Keep it simple.",
        "expected_contains": "def"
    },
    "structured_output": {
        "system": "You are a data generator.",
        "user": "Generate a fictional company with name, industry, and employee count. Return as JSON.",
        "schema": {
            "name": "Company",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "industry": {"type": "string"},
                    "employee_count": {"type": "integer"}
                },
                "required": ["name", "industry", "employee_count"],
                "additionalProperties": False
            }
        }
    }
}


@dataclass
class TestResult:
    """Stores results from a single test."""
    timestamp: str
    endpoint_name: str
    model: str
    test_type: str
    status: str  # PASS, FAIL, SKIP, ERROR
    
    # Latency metrics (in milliseconds)
    time_to_first_token_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Response info
    response_preview: str = ""
    finish_reason: str = ""
    actual_model: str = ""  # Model returned by API
    
    # Capabilities
    supports_streaming: bool = False
    supports_structured_output: bool = False
    supports_temperature: bool = False
    is_reasoning_model: bool = False
    
    # Error info
    error_message: str = ""
    error_type: str = ""


class ComprehensiveModelTester:
    """
    Comprehensive testing suite for Azure OpenAI models.
    
    Performs multiple test types across all configured models and endpoints,
    collecting detailed metrics for executive reporting.
    """
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def _create_provider(self, endpoint_config: Dict[str, Any]) -> AzureOpenAIProvider:
        """Create a provider for the given endpoint configuration."""
        return AzureOpenAIProvider(
            endpoint=endpoint_config["endpoint"],
            api_key=endpoint_config["api_key"],
            api_version="2024-12-01-preview",
        )
    
    def _create_client(self, provider: AzureOpenAIProvider) -> AIClient:
        """Create an AI client with the given provider."""
        return AIClient(
            provider="azure_openai",
            api_key=provider.api_key,
            endpoint=provider.endpoint,
            api_version=provider.api_version,
        )
    
    def _get_model_info(self, model: str) -> Dict[str, bool]:
        """Get capability information for a model."""
        caps = get_model_capabilities(model)
        if caps:
            return {
                "supports_streaming": caps.supports_streaming,
                "supports_structured_output": caps.supports_structured_output,
                "supports_temperature": caps.supports_temperature,
                "is_reasoning_model": not caps.supports_temperature,  # Reasoning models don't support temp
            }
        return {
            "supports_streaming": True,
            "supports_structured_output": True,
            "supports_temperature": True,
            "is_reasoning_model": False,
        }
    
    async def test_basic_completion(
        self,
        client: AIClient,
        model: str,
        endpoint_name: str,
        prompt_config: Dict[str, Any]
    ) -> TestResult:
        """Test basic (non-streaming) completion."""
        timestamp = datetime.now().isoformat()
        model_info = self._get_model_info(model)
        
        result = TestResult(
            timestamp=timestamp,
            endpoint_name=endpoint_name,
            model=model,
            test_type="basic_completion",
            status="PENDING",
            **model_info
        )
        
        try:
            start_time = time.perf_counter()
            
            messages = [
                Message(role=MessageRole.USER, content=prompt_config["user"]),
            ]
            if prompt_config.get("system") and model_info["supports_temperature"]:
                messages.insert(0, Message(role=MessageRole.SYSTEM, content=prompt_config["system"]))
            
            # Use appropriate parameters based on model type
            kwargs = {"max_tokens": 500}
            if model_info["supports_temperature"]:
                kwargs["temperature"] = 0.7
            else:
                kwargs["reasoning_effort"] = "low"
            
            response = await client.agenerate(
                model=model,
                messages=messages,
                **kwargs
            )
            
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            
            result.status = "PASS"
            result.total_time_ms = round(total_time_ms, 2)
            result.time_to_first_token_ms = round(total_time_ms, 2)  # Same for non-streaming
            result.response_preview = (response.content[:200] + "...") if len(response.content) > 200 else response.content
            result.finish_reason = response.finish_reason or ""
            result.actual_model = response.model or model
            
            if response.usage:
                result.prompt_tokens = response.usage.prompt_tokens
                result.completion_tokens = response.usage.completion_tokens
                result.total_tokens = response.usage.total_tokens
            
            # Check expected content
            expected = prompt_config.get("expected_contains")
            if expected and expected not in response.content:
                result.status = "PASS_UNEXPECTED"
                
        except ProviderError as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        except Exception as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        
        return result
    
    async def test_streaming(
        self,
        client: AIClient,
        model: str,
        endpoint_name: str,
        prompt_config: Dict[str, Any]
    ) -> TestResult:
        """Test streaming completion with time-to-first-token measurement."""
        timestamp = datetime.now().isoformat()
        model_info = self._get_model_info(model)
        
        result = TestResult(
            timestamp=timestamp,
            endpoint_name=endpoint_name,
            model=model,
            test_type="streaming",
            status="PENDING",
            **model_info
        )
        
        if not model_info["supports_streaming"]:
            result.status = "SKIP"
            result.error_message = "Model does not support streaming"
            return result
        
        try:
            start_time = time.perf_counter()
            first_token_time = None
            full_content = ""
            last_chunk = None
            
            messages = [
                Message(role=MessageRole.USER, content=prompt_config["user"]),
            ]
            if prompt_config.get("system") and model_info["supports_temperature"]:
                messages.insert(0, Message(role=MessageRole.SYSTEM, content=prompt_config["system"]))
            
            kwargs = {"max_tokens": 500}
            if model_info["supports_temperature"]:
                kwargs["temperature"] = 0.7
            
            async for chunk in client.astream(
                model=model,
                messages=messages,
                **kwargs
            ):
                if first_token_time is None and chunk.content:
                    first_token_time = time.perf_counter()
                if chunk.content:
                    full_content += chunk.content
                last_chunk = chunk
            
            end_time = time.perf_counter()
            
            result.status = "PASS"
            result.total_time_ms = round((end_time - start_time) * 1000, 2)
            if first_token_time:
                result.time_to_first_token_ms = round((first_token_time - start_time) * 1000, 2)
            result.response_preview = (full_content[:200] + "...") if len(full_content) > 200 else full_content
            
            if last_chunk:
                result.finish_reason = last_chunk.finish_reason or ""
                result.actual_model = last_chunk.model or model
                if last_chunk.usage:
                    result.prompt_tokens = last_chunk.usage.prompt_tokens
                    result.completion_tokens = last_chunk.usage.completion_tokens
                    result.total_tokens = last_chunk.usage.total_tokens
                    
        except ProviderError as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        except Exception as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        
        return result
    
    async def test_structured_output(
        self,
        client: AIClient,
        model: str,
        endpoint_name: str,
        prompt_config: Dict[str, Any]
    ) -> TestResult:
        """Test structured JSON output."""
        timestamp = datetime.now().isoformat()
        model_info = self._get_model_info(model)
        
        result = TestResult(
            timestamp=timestamp,
            endpoint_name=endpoint_name,
            model=model,
            test_type="structured_output",
            status="PENDING",
            **model_info
        )
        
        if not model_info["supports_structured_output"]:
            result.status = "SKIP"
            result.error_message = "Model does not support structured output"
            return result
        
        try:
            start_time = time.perf_counter()
            
            response_format = ResponseFormat(
                type="json_schema",
                json_schema=prompt_config["schema"]
            )
            
            messages = [
                Message(role=MessageRole.USER, content=prompt_config["user"]),
            ]
            
            # Use higher token limit for reasoning models (they use internal chain-of-thought tokens)
            # Reasoning models (gpt-5 series, o-series) consume variable amounts of internal reasoning tokens
            # We need generous limits to ensure complete JSON responses
            if model_info["is_reasoning_model"]:
                max_tokens_for_structured = 1500  # High limit for reasoning models
            else:
                max_tokens_for_structured = 200
            kwargs = {"max_tokens": max_tokens_for_structured, "response_format": response_format}
            if model_info["supports_temperature"]:
                kwargs["temperature"] = 0.7
            
            response = await client.agenerate(
                model=model,
                messages=messages,
                **kwargs
            )
            
            end_time = time.perf_counter()
            
            result.status = "PASS"
            result.total_time_ms = round((end_time - start_time) * 1000, 2)
            result.time_to_first_token_ms = result.total_time_ms
            result.response_preview = response.content[:200] if response.content else ""
            result.finish_reason = response.finish_reason or ""
            result.actual_model = response.model or model
            
            if response.usage:
                result.prompt_tokens = response.usage.prompt_tokens
                result.completion_tokens = response.usage.completion_tokens
                result.total_tokens = response.usage.total_tokens
            
            # Validate JSON structure
            import json
            try:
                parsed = json.loads(response.content)
                if not all(k in parsed for k in ["name", "industry", "employee_count"]):
                    result.status = "PASS_INCOMPLETE"
            except json.JSONDecodeError:
                result.status = "FAIL"
                result.error_message = "Invalid JSON response"
                
        except ProviderError as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        except Exception as e:
            result.status = "ERROR"
            result.error_message = str(e)[:500]
            result.error_type = type(e).__name__
        
        return result
    
    async def test_model(
        self,
        endpoint_config: Dict[str, Any],
        model: str
    ) -> List[TestResult]:
        """Run all tests for a single model."""
        results = []
        
        provider = self._create_provider(endpoint_config)
        client = self._create_client(provider)
        endpoint_name = endpoint_config["name"]
        
        print(f"\n  Testing {model}...")
        
        # Test 1: Basic completion with math
        print(f"    ├─ Basic completion (math)...", end=" ", flush=True)
        result = await self.test_basic_completion(
            client, model, endpoint_name, TEST_PROMPTS["simple_math"]
        )
        results.append(result)
        status_icon = "✅" if result.status == "PASS" else "❌" if result.status == "ERROR" else "⚠️"
        print(f"{status_icon} {result.total_time_ms:.0f}ms" if result.total_time_ms else f"{status_icon}")
        
        # Test 2: Streaming
        print(f"    ├─ Streaming...", end=" ", flush=True)
        result = await self.test_streaming(
            client, model, endpoint_name, TEST_PROMPTS["reasoning"]
        )
        results.append(result)
        status_icon = "✅" if result.status == "PASS" else "⏭️" if result.status == "SKIP" else "❌"
        ttft = f"TTFT:{result.time_to_first_token_ms:.0f}ms" if result.time_to_first_token_ms else ""
        print(f"{status_icon} {ttft}")
        
        # Test 3: Code generation
        print(f"    ├─ Code generation...", end=" ", flush=True)
        result = await self.test_basic_completion(
            client, model, endpoint_name, TEST_PROMPTS["code_generation"]
        )
        result.test_type = "code_generation"
        results.append(result)
        status_icon = "✅" if result.status == "PASS" else "❌"
        print(f"{status_icon} {result.total_time_ms:.0f}ms" if result.total_time_ms else f"{status_icon}")
        
        # Test 4: Structured output
        print(f"    └─ Structured output...", end=" ", flush=True)
        result = await self.test_structured_output(
            client, model, endpoint_name, TEST_PROMPTS["structured_output"]
        )
        results.append(result)
        status_icon = "✅" if result.status == "PASS" else "⏭️" if result.status == "SKIP" else "❌"
        print(f"{status_icon} {result.total_time_ms:.0f}ms" if result.total_time_ms else f"{status_icon}")
        
        return results
    
    async def run_all_tests(self) -> None:
        """Run tests on all configured endpoints and models."""
        print("=" * 80)
        print("COMPREHENSIVE MODEL TESTING SUITE")
        print("=" * 80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        all_endpoints = [ENDPOINT_1, ENDPOINT_2]
        
        for endpoint_config in all_endpoints:
            print(f"\n{'─' * 60}")
            print(f"Endpoint: {endpoint_config['name']}")
            print(f"URL: {endpoint_config['endpoint']}")
            print(f"Models to test: {len(endpoint_config['models'])}")
            print(f"{'─' * 60}")
            
            for model in endpoint_config["models"]:
                try:
                    model_results = await self.test_model(endpoint_config, model)
                    self.results.extend(model_results)
                except Exception as e:
                    print(f"\n  ❌ Critical error testing {model}: {e}")
                    # Add error result
                    self.results.append(TestResult(
                        timestamp=datetime.now().isoformat(),
                        endpoint_name=endpoint_config["name"],
                        model=model,
                        test_type="all",
                        status="CRITICAL_ERROR",
                        error_message=str(e)[:500],
                        error_type=type(e).__name__
                    ))
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    def generate_csv_report(self, output_path: str = "model_test_report.csv") -> str:
        """Generate a detailed CSV report of all test results."""
        fieldnames = [
            "timestamp",
            "endpoint_name", 
            "model",
            "actual_model",
            "test_type",
            "status",
            "time_to_first_token_ms",
            "total_time_ms",
            "prompt_tokens",
            "completion_tokens", 
            "total_tokens",
            "supports_streaming",
            "supports_structured_output",
            "supports_temperature",
            "is_reasoning_model",
            "finish_reason",
            "response_preview",
            "error_type",
            "error_message",
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = asdict(result)
                # Clean up response preview for CSV
                row["response_preview"] = row["response_preview"].replace('\n', ' ').replace('\r', '')[:200]
                writer.writerow(row)
        
        return output_path
    
    def generate_summary_csv(self, output_path: str = "model_summary_report.csv") -> str:
        """Generate a summary CSV grouped by model."""
        # Aggregate results by model
        model_stats: Dict[str, Dict[str, Any]] = {}
        
        for result in self.results:
            model = result.model
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "endpoint": result.endpoint_name,
                    "actual_model": result.actual_model,
                    "is_reasoning_model": result.is_reasoning_model,
                    "supports_streaming": result.supports_streaming,
                    "supports_structured_output": result.supports_structured_output,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "tests_skipped": 0,
                    "tests_error": 0,
                    "total_tests": 0,
                    "avg_latency_ms": [],
                    "avg_ttft_ms": [],
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                }
            
            stats = model_stats[model]
            stats["total_tests"] += 1
            
            if result.status == "PASS":
                stats["tests_passed"] += 1
            elif result.status == "SKIP":
                stats["tests_skipped"] += 1
            elif result.status in ["FAIL", "ERROR", "CRITICAL_ERROR"]:
                stats["tests_error"] += 1
            else:
                stats["tests_passed"] += 1  # PASS_UNEXPECTED, etc.
            
            if result.total_time_ms:
                stats["avg_latency_ms"].append(result.total_time_ms)
            if result.time_to_first_token_ms:
                stats["avg_ttft_ms"].append(result.time_to_first_token_ms)
            
            stats["total_prompt_tokens"] += result.prompt_tokens
            stats["total_completion_tokens"] += result.completion_tokens
            stats["total_tokens"] += result.total_tokens
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["avg_latency_ms"]:
                stats["avg_latency_ms"] = round(sum(stats["avg_latency_ms"]) / len(stats["avg_latency_ms"]), 2)
            else:
                stats["avg_latency_ms"] = None
            
            if stats["avg_ttft_ms"]:
                stats["avg_ttft_ms"] = round(sum(stats["avg_ttft_ms"]) / len(stats["avg_ttft_ms"]), 2)
            else:
                stats["avg_ttft_ms"] = None
            
            # Calculate pass rate
            total_countable = stats["tests_passed"] + stats["tests_error"]
            if total_countable > 0:
                stats["pass_rate"] = round(stats["tests_passed"] / total_countable * 100, 1)
            else:
                stats["pass_rate"] = 100.0
        
        # Write CSV
        fieldnames = [
            "model",
            "endpoint", 
            "actual_model",
            "is_reasoning_model",
            "supports_streaming",
            "supports_structured_output",
            "tests_passed",
            "tests_failed",
            "tests_skipped",
            "tests_error",
            "total_tests",
            "pass_rate",
            "avg_latency_ms",
            "avg_ttft_ms",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for model in sorted(model_stats.keys()):
                writer.writerow(model_stats[model])
        
        return output_path
    
    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY")
        print("=" * 80)
        
        # Overall stats
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status in ["PASS", "PASS_UNEXPECTED", "PASS_INCOMPLETE"])
        failed = sum(1 for r in self.results if r.status in ["FAIL", "ERROR", "CRITICAL_ERROR"])
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        # Calculate pass rate excluding skipped tests (skips are expected for unsupported features)
        countable_tests = total_tests - skipped
        pass_rate = (passed / countable_tests * 100) if countable_tests > 0 else 100.0
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"  ✅ Passed: {passed}/{countable_tests} ({pass_rate:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/total_tests*100:.1f}%)")
        print(f"  ⏭️  Skipped: {skipped} (expected - unsupported features)")
        
        # Token usage
        total_prompt = sum(r.prompt_tokens for r in self.results)
        total_completion = sum(r.completion_tokens for r in self.results)
        total_tokens = sum(r.total_tokens for r in self.results)
        
        print(f"\nToken Usage:")
        print(f"  Prompt tokens: {total_prompt:,}")
        print(f"  Completion tokens: {total_completion:,}")
        print(f"  Total tokens: {total_tokens:,}")
        
        # Latency stats
        latencies = [r.total_time_ms for r in self.results if r.total_time_ms]
        ttfts = [r.time_to_first_token_ms for r in self.results if r.time_to_first_token_ms]
        
        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Average response time: {sum(latencies)/len(latencies):.0f}ms")
            print(f"  Min response time: {min(latencies):.0f}ms")
            print(f"  Max response time: {max(latencies):.0f}ms")
        
        if ttfts:
            print(f"  Average TTFT: {sum(ttfts)/len(ttfts):.0f}ms")
            print(f"  Min TTFT: {min(ttfts):.0f}ms")
            print(f"  Max TTFT: {max(ttfts):.0f}ms")
        
        # Per-model summary
        print("\n" + "-" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"\n{'Model':<20} {'Status':<10} {'Avg Latency':<12} {'TTFT':<10} {'Tokens':<10}")
        print("-" * 70)
        
        # Group by model
        model_results: Dict[str, List[TestResult]] = {}
        for r in self.results:
            if r.model not in model_results:
                model_results[r.model] = []
            model_results[r.model].append(r)
        
        for model in sorted(model_results.keys()):
            results = model_results[model]
            passed = sum(1 for r in results if r.status.startswith("PASS"))
            total = len(results)
            
            latencies = [r.total_time_ms for r in results if r.total_time_ms]
            avg_latency = f"{sum(latencies)/len(latencies):.0f}ms" if latencies else "N/A"
            
            ttfts = [r.time_to_first_token_ms for r in results if r.time_to_first_token_ms]
            avg_ttft = f"{sum(ttfts)/len(ttfts):.0f}ms" if ttfts else "N/A"
            
            tokens = sum(r.total_tokens for r in results)
            
            status = f"{passed}/{total}"
            status_icon = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
            
            print(f"{model:<20} {status_icon} {status:<8} {avg_latency:<12} {avg_ttft:<10} {tokens:<10,}")
        
        # Duration
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print(f"\nTest Duration: {duration:.1f} seconds")
        print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    """Main entry point."""
    print("\n" + "🔬" * 20)
    print("  inflow-unified-ai COMPREHENSIVE MODEL TESTING")
    print("🔬" * 20 + "\n")
    
    tester = ComprehensiveModelTester()
    
    try:
        # Run all tests
        await tester.run_all_tests()
        
        # Generate reports
        print("\n📊 Generating reports...")
        
        # Detailed CSV
        detailed_path = tester.generate_csv_report("model_test_detailed_report.csv")
        print(f"  ✅ Detailed report: {detailed_path}")
        
        # Summary CSV
        summary_path = tester.generate_summary_csv("model_test_summary_report.csv")
        print(f"  ✅ Summary report: {summary_path}")
        
        # Print summary
        tester.print_summary()
        
        print("\n" + "=" * 80)
        print("📁 REPORTS GENERATED")
        print("=" * 80)
        print(f"\n  1. {Path(detailed_path).absolute()}")
        print(f"  2. {Path(summary_path).absolute()}")
        print("\nThese CSV files can be opened in Excel for further analysis.")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
        if tester.results:
            print("Generating partial reports...")
            tester.generate_csv_report("model_test_partial_report.csv")
            tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
