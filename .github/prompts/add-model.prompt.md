---
mode: agent
description: Add a new AI model to the ModelRegistry
tools: ['codebase', 'editFiles', 'search', 'runCommands', 'usages']
---

# Add New Model to Registry

## Purpose
Add a new AI model to the ModelRegistry with proper configuration.

## Required Information
- Model ID (e.g., "gpt-4.5-turbo", "claude-4-opus")
- Provider (azure, openai, anthropic, google)
- Model family (existing or new)
- Capabilities (vision, streaming, function calling, etc.)
- Context window size
- Max output tokens

## Implementation Steps

### Step 1: Check if ModelFamily Exists

Look in `src/inflow_unified_ai/models/ai_model.py` for existing families:

```python
class ModelFamily(Enum):
    # OpenAI families
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41 = "gpt-4.1"
    # ... etc
```

### Step 2: Add New ModelFamily (if needed)

If the model is from a new family, add to `ModelFamily` enum:

```python
class ModelFamily(Enum):
    # ... existing families
    ${input:family_name} = "${input:family_value}"
```

### Step 3: Add Model to Registry

In `src/inflow_unified_ai/models/registry.py`, add to `_models` dict:

```python
class ModelRegistry:
    _models = {
        # ... existing models
        
        "${input:model_id}": AIModel(
            model_id="${input:model_id}",
            provider="${input:provider}",
            family=ModelFamily.${input:family_enum},
            capabilities=ModelCapabilities(
                supports_vision=${input:vision},
                supports_streaming=${input:streaming},
                supports_function_calling=${input:function_calling},
                supports_json_mode=${input:json_mode},
                supports_system_message=${input:system_message},
                supports_tools=${input:tools},
            ),
            context_window=${input:context_window},
            max_output_tokens=${input:max_output_tokens},
        ),
    }
```

### Step 4: Add Tests

Add test in `tests/test_package.py` or `tests/test_models.py`:

```python
def test_model_${input:model_id_snake}_exists():
    """Test that ${input:model_id} is in registry."""
    model = ModelRegistry.get_model("${input:model_id}")
    assert model is not None
    assert model.model_id == "${input:model_id}"
    assert model.provider == "${input:provider}"
    assert model.family == ModelFamily.${input:family_enum}


def test_model_${input:model_id_snake}_capabilities():
    """Test ${input:model_id} capabilities."""
    model = ModelRegistry.get_model("${input:model_id}")
    assert model.capabilities.supports_vision == ${input:vision}
    assert model.capabilities.supports_streaming == ${input:streaming}
    assert model.context_window == ${input:context_window}
```

### Step 5: Update Documentation

Add to model tables in `docs/07-API-REFERENCE.md`:

```markdown
| `${input:model_id}` | ${input:provider} | ${input:context_window} | ✅/❌ Vision | ✅/❌ Streaming |
```

## Example: Adding GPT-4.5-Turbo

```python
# In registry.py
"gpt-4.5-turbo": AIModel(
    model_id="gpt-4.5-turbo",
    provider="openai",
    family=ModelFamily.GPT_45,  # New family if needed
    capabilities=ModelCapabilities(
        supports_vision=True,
        supports_streaming=True,
        supports_function_calling=True,
        supports_json_mode=True,
        supports_system_message=True,
        supports_tools=True,
    ),
    context_window=256000,
    max_output_tokens=16384,
),
```

## Common Model Configurations

### OpenAI Models
```python
capabilities=ModelCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_system_message=True,
    supports_tools=True,
)
```

### Anthropic Claude Models
```python
capabilities=ModelCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=False,  # Claude uses XML
    supports_system_message=True,
    supports_tools=True,
)
```

### Google Gemini Models
```python
capabilities=ModelCapabilities(
    supports_vision=True,
    supports_streaming=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_system_message=True,
    supports_tools=True,
)
```

## Checklist

- [ ] ModelFamily added (if new family)
- [ ] Model added to `_models` in registry.py
- [ ] Capabilities correctly configured
- [ ] Context window and max tokens set
- [ ] Unit tests added
- [ ] Documentation updated
- [ ] Tests pass: `pytest tests/ -v`
