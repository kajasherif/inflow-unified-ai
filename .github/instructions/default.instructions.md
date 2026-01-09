---
applyTo: '**'
---
# Default Instructions for inflow-unified-ai

## Before Any Work
1. Create a todo list with the planned approach
2. Share the approach with the user for validation
3. Wait for confirmation before proceeding
4. Do NOT create summary markdown files unless explicitly requested

## Code Quality Standards
- All functions must have type hints
- All public methods must have docstrings
- Use async/await for all provider methods
- Never hardcode API keys or secrets
- Export all public APIs from `__init__.py`

## Testing Requirements
- Write tests for all new code
- Use pytest with pytest-asyncio
- Mock external API calls
- Maintain coverage targets (90%+ for core)

## Documentation
- Update API reference when adding public APIs
- Include code examples in docstrings
- Keep README and docs/ in sync
