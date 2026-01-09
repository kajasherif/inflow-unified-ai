#!/usr/bin/env python3
"""
Demo script for inflow-unified-ai library.

This script demonstrates the Unified AI Abstraction Layer with Azure OpenAI.
It tests various models including GPT-4o, GPT-4.1, and reasoning models.

Usage:
    Option 1: Set environment variables:
        AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        AZURE_OPENAI_API_KEY=your-api-key
    
    Option 2: Create a .env file with the variables above
    
    Run:
        python demo.py
"""

import asyncio
import os
import sys

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inflow_unified_ai import AIClient
from inflow_unified_ai.models import (
    CompletionRequest,
    Message,
    MessageRole,
    MODEL_CAPABILITIES_REGISTRY,
    ModelFamily,
    get_model_capabilities,
)
from inflow_unified_ai.providers import (
    AzureOpenAIProvider,
    ProviderError,
)


async def test_basic_completion(client: AIClient, model: str):
    """Test basic text completion."""
    print(f"\n{'='*60}")
    print(f"Testing basic completion with: {model}")
    print('='*60)
    
    try:
        response = await client.agenerate(
            model=model,
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                Message(role=MessageRole.USER, content="What is 2 + 2? Answer in one word."),
            ],
            temperature=0.7,  # Will be ignored for reasoning models
            max_tokens=50,
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Finish reason: {response.finish_reason}")
        if response.usage:
            print(f"Tokens - Prompt: {response.usage.prompt_tokens}, "
                  f"Completion: {response.usage.completion_tokens}, "
                  f"Total: {response.usage.total_tokens}")
        print("\u2705 Success!")
        return True, response.usage
        
    except ProviderError as e:
        print(f"\u274c Provider error: {e}")
        return False, None
    except Exception as e:
        print(f"\u274c Unexpected error: {type(e).__name__}: {e}")
        return False, None


async def test_streaming(client: AIClient, model: str):
    """Test streaming completion."""
    print(f"\n{'='*60}")
    print(f"Testing streaming with: {model}")
    print('='*60)
    
    try:
        print("Response: ", end="", flush=True)
        total_tokens = 0
        last_chunk = None
        async for chunk in client.astream(
            model=model,
            messages=[
                Message(role=MessageRole.USER, content="Count from 1 to 5."),
            ],
            max_tokens=100,
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            last_chunk = chunk
            if chunk.usage:
                total_tokens = chunk.usage.total_tokens
        print()
        if last_chunk and last_chunk.usage:
            print(f"Tokens - Prompt: {last_chunk.usage.prompt_tokens}, "
                  f"Completion: {last_chunk.usage.completion_tokens}, "
                  f"Total: {last_chunk.usage.total_tokens}")
        print("\u2705 Streaming success!")
        return True, last_chunk.usage if last_chunk else None
        
    except ProviderError as e:
        print(f"\n\u274c Provider error: {e}")
        return False, None
    except Exception as e:
        print(f"\n\u274c Unexpected error: {type(e).__name__}: {e}")
        return False, None


async def test_reasoning_model(client: AIClient, model: str):
    """Test reasoning model with reasoning_effort parameter."""
    print(f"\n{'='*60}")
    print(f"Testing reasoning model: {model}")
    print('='*60)
    
    # Check if model supports reasoning
    caps = get_model_capabilities(model)
    if not caps or caps.family not in [
        ModelFamily.O1, ModelFamily.O1_MINI, ModelFamily.O1_PREVIEW,
        ModelFamily.O3, ModelFamily.O3_MINI, ModelFamily.O3_PRO, ModelFamily.O4_MINI,
        ModelFamily.GPT_5, ModelFamily.GPT_5_MINI, ModelFamily.GPT_5_NANO,
        ModelFamily.GPT_5_PRO, ModelFamily.GPT_51, ModelFamily.GPT_52,
    ]:
        print(f"Skipping - {model} is not a reasoning model")
        return True
    
    try:
        response = await client.agenerate(
            model=model,
            messages=[
                Message(role=MessageRole.USER, content="What is 15 * 17? Think step by step."),
            ],
            reasoning_effort="low",  # low, medium, or high
            max_tokens=500,
        )
        
        print(f"Model: {response.model}")
        print(f"Response: {response.content[:500]}...")
        print(f"Finish reason: {response.finish_reason}")
        if response.usage:
            print(f"Tokens used: {response.usage.total_tokens}")
        print("✅ Reasoning model success!")
        return True
        
    except ProviderError as e:
        print(f"❌ Provider error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False


async def test_structured_output(client: AIClient, model: str):
    """Test structured JSON output."""
    print(f"\n{'='*60}")
    print(f"Testing structured output with: {model}")
    print('='*60)
    
    # Check if model supports structured output
    caps = get_model_capabilities(model)
    if caps and not caps.supports_structured_output:
        print(f"Skipping - {model} doesn't support structured output")
        return True
    
    try:
        # Define JSON response format for structured output
        from inflow_unified_ai.models.requests import ResponseFormat
        
        response_format = ResponseFormat(
            type="json_schema",
            json_schema={
                "name": "Person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "occupation": {"type": "string"}
                    },
                    "required": ["name", "age", "occupation"],
                    "additionalProperties": False
                }
            }
        )
        
        result = await client.agenerate(
            model=model,
            messages=[
                Message(
                    role=MessageRole.USER, 
                    content="Generate a fictional person with name, age, and occupation. Return as JSON."
                ),
            ],
            response_format=response_format,
            max_tokens=200,
        )
        
        print(f"Structured result: {result.content}")
        if result.usage:
            print(f"Tokens - Prompt: {result.usage.prompt_tokens}, "
                  f"Completion: {result.usage.completion_tokens}, "
                  f"Total: {result.usage.total_tokens}")
        print("\u2705 Structured output success!")
        return True, result.usage
        
    except ProviderError as e:
        print(f"\u274c Provider error: {e}")
        return False, None
    except Exception as e:
        print(f"\u274c Unexpected error: {type(e).__name__}: {e}")
        return False, None


async def show_model_capabilities():
    """Display all supported models and their capabilities."""
    print("\n" + "="*80)
    print("SUPPORTED MODELS AND CAPABILITIES")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Family':<18} {'Temp':<6} {'Struct':<7} {'Stream':<7} {'Vision':<7}")
    print("-"*80)
    
    for model_name, caps in sorted(MODEL_CAPABILITIES_REGISTRY.items()):
        family = caps.family.value[:16]
        temp = "✓" if caps.supports_temperature else "✗"
        struct = "✓" if caps.supports_structured_output else "✗"
        stream = "✓" if caps.supports_streaming else "✗"
        vision = "✓" if caps.supports_vision else "✗"
        
        print(f"{model_name:<30} {family:<18} {temp:<6} {struct:<7} {stream:<7} {vision:<7}")


async def main():
    """Main demo function."""
    print("="*80)
    print("inflow-unified-ai UNIFIED AI ABSTRACTION LAYER DEMO")
    print("="*80)
    
    # Check for Azure credentials
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        print("\n⚠️  Azure OpenAI credentials not found!")
        print("Please set environment variables:")
        print("  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
        print("  AZURE_OPENAI_API_KEY=your-api-key")
        print("\nShowing model capabilities instead...\n")
        await show_model_capabilities()
        return
    
    print(f"\n✅ Azure OpenAI endpoint: {endpoint}")
    
    # Create the AI client with Azure OpenAI provider
    provider = AzureOpenAIProvider(
        endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview",  # Latest stable version
    )
    
    client = AIClient(default_provider=provider)
    
    # Show supported models
    await show_model_capabilities()
    
    # Test with available models - adjust these based on your deployment names
    # The deployment name should match your Azure OpenAI deployment
    test_models = [
        "gpt-4o",           # Standard model
        "gpt-4o-mini",      # Fast and cost-effective
        "gpt-4.1",          # Latest GPT-4.1
        "gpt-4.1-mini",     # GPT-4.1 mini variant
        # "o4-mini",        # Reasoning model - uncomment if deployed
    ]
    
    results = {}
    token_usage = {}  # Track token usage per model
    
    for model in test_models:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING MODEL: {model}")
        print(f"{'#'*80}")
        
        token_usage[model] = {"prompt": 0, "completion": 0, "total": 0}
        
        # Run tests
        success, usage = await test_basic_completion(client, model)
        results[f"{model}_basic"] = success
        if usage:
            token_usage[model]["prompt"] += usage.prompt_tokens
            token_usage[model]["completion"] += usage.completion_tokens
            token_usage[model]["total"] += usage.total_tokens
        
        success, usage = await test_streaming(client, model)
        results[f"{model}_stream"] = success
        if usage:
            token_usage[model]["prompt"] += usage.prompt_tokens
            token_usage[model]["completion"] += usage.completion_tokens
            token_usage[model]["total"] += usage.total_tokens
        
        success, usage = await test_structured_output(client, model)
        results[f"{model}_structured"] = success
        if usage:
            token_usage[model]["prompt"] += usage.prompt_tokens
            token_usage[model]["completion"] += usage.completion_tokens
            token_usage[model]["total"] += usage.total_tokens
        
        # Test reasoning if applicable
        caps = get_model_capabilities(model)
        if caps and not caps.supports_temperature:  # Reasoning models don't support temperature
            results[f"{model}_reasoning"] = await test_reasoning_model(client, model)
    
    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Token Usage Summary
    print("\n\n" + "="*80)
    print("TOKEN USAGE SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Prompt':>12} {'Completion':>12} {'Total':>12}")
    print("-"*60)
    
    grand_total_prompt = 0
    grand_total_completion = 0
    grand_total = 0
    
    for model, usage in token_usage.items():
        print(f"{model:<20} {usage['prompt']:>12,} {usage['completion']:>12,} {usage['total']:>12,}")
        grand_total_prompt += usage['prompt']
        grand_total_completion += usage['completion']
        grand_total += usage['total']
    
    print("-"*60)
    print(f"{'TOTAL':<20} {grand_total_prompt:>12,} {grand_total_completion:>12,} {grand_total:>12,}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")


if __name__ == "__main__":
    asyncio.run(main())
