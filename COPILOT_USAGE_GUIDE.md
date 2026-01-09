# GitHub Copilot Usage Guide

This guide explains how to use GitHub Copilot's custom agents, chat modes, and prompts configured for the **inflow-unified-ai** project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Reference](#quick-reference)
- [Using Agents](#using-agents)
- [Using Chat Modes](#using-chat-modes)
- [Using Prompts](#using-prompts)
- [Copilot Instructions](#copilot-instructions)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **VS Code** with GitHub Copilot extension installed
2. **GitHub Copilot Chat** extension installed
3. Active GitHub Copilot subscription
4. This repository cloned locally

### Verify Extensions

In VS Code, ensure you have:
- `GitHub.copilot` - GitHub Copilot
- `GitHub.copilot-chat` - GitHub Copilot Chat

---

## Quick Reference

| Type | Syntax | Example |
|------|--------|---------|
| Agent | `@agent-name` | `@python-package-dev help me add a feature` |
| Chat Mode | Type in chat | Select from dropdown or type mode name |
| Prompt | `/prompt-name` | `/add-provider Mistral` |
| Instructions | Automatic | Applied automatically to all interactions |

---

## Using Agents

Agents are specialized AI assistants with domain expertise. Invoke them using `@agent-name` in Copilot Chat.

### Available Agents

#### @python-package-dev

**Purpose**: Expert Python package developer for source code work.

**When to use**:
- Adding new features to the package
- Fixing bugs in source code
- Refactoring existing code
- Understanding package architecture

**How to use**:
```
@python-package-dev How do I add streaming support to a provider?
```

```
@python-package-dev Explain the AIClient class architecture
```

```
@python-package-dev Help me implement retry logic for API calls
```

**What it knows**:
- Package structure (`src/inflow_unified_ai/`)
- Async patterns for providers
- Type hints and dataclasses
- Export conventions

---

#### @test-writer

**Purpose**: Expert test writer using pytest.

**When to use**:
- Writing new unit tests
- Adding test coverage
- Understanding test patterns
- Mocking external dependencies

**How to use**:
```
@test-writer Write tests for the AzureOpenAIProvider class
```

```
@test-writer How do I mock async API calls?
```

```
@test-writer Add parametrized tests for model capabilities
```

**What it knows**:
- pytest and pytest-asyncio
- Mocking patterns with unittest.mock
- Coverage targets (90% core, 80% providers)
- Test file conventions

---

#### @docs-writer

**Purpose**: Technical documentation specialist.

**When to use**:
- Writing API documentation
- Creating usage examples
- Updating README files
- Documenting new features

**How to use**:
```
@docs-writer Document the new MistralProvider class
```

```
@docs-writer Write a getting started guide for this package
```

```
@docs-writer Add examples for streaming responses
```

**What it knows**:
- Documentation structure in `docs/`
- Markdown formatting conventions
- API documentation patterns
- Code example requirements

---

### Agent Usage Tips

1. **Be specific**: Include context about what you're working on
   ```
   ❌ @python-package-dev add a provider
   ✅ @python-package-dev add a new MistralProvider that supports their chat API
   ```

2. **Reference files**: Mention specific files when relevant
   ```
   @python-package-dev In client.py, how should I handle connection errors?
   ```

3. **Chain agents**: Use different agents for different tasks
   ```
   1. @python-package-dev implement the feature
   2. @test-writer write tests for it
   3. @docs-writer document it
   ```

---

## Using Chat Modes

Chat modes change how Copilot responds. Select them from the chat input dropdown or type the mode name.

### Available Chat Modes

#### Plan Mode

**Purpose**: Plan changes without generating code.

**When to use**:
- Before implementing complex features
- When you need to understand impact
- For architecture decisions
- To create implementation roadmaps

**How to access**:
1. Open Copilot Chat
2. Click the mode selector (dropdown)
3. Select "Plan"

**Example interaction**:
```
User: Add support for Cohere AI provider

Copilot (Plan mode):
## Change Plan: Add Cohere Provider

### Affected Files
| File | Change Type | Description |
|------|-------------|-------------|
| `src/.../providers/cohere.py` | Create | New provider |
| `src/.../providers/__init__.py` | Modify | Export |
| `src/.../__init__.py` | Modify | Export |

### Implementation Steps
1. Create CohereProvider class
2. Implement generate() and stream()
...
```

---

#### Review Mode

**Purpose**: Review code for quality and best practices.

**When to use**:
- Before committing changes
- During code review
- To check for security issues
- To verify best practices

**How to access**:
1. Open Copilot Chat
2. Click the mode selector
3. Select "Review"

**Example interaction**:
```
User: Review my changes to azure_openai.py

Copilot (Review mode):
## Code Review Summary

### Overall Assessment
⚠️ Needs Changes

### Issues Found
#### 🔴 Critical
- Hardcoded API key on line 45
  - Suggestion: Use os.getenv()
...
```

---

## Using Prompts

Prompts are guided workflows for common tasks. Invoke them with `/prompt-name`.

### Available Prompts

#### /add-provider

**Purpose**: Step-by-step guide to add a new AI provider.

**How to use**:
```
/add-provider
```

Or with context:
```
/add-provider Mistral AI
```

**What it does**:
1. Creates provider file template
2. Shows export steps
3. Provides test template
4. Lists documentation updates

**Example**:
```
User: /add-provider Cohere

Copilot: # Add New AI Provider

## Required Information
- Provider name: Cohere
- API client library: cohere
- Environment variable: COHERE_API_KEY

## Step 1: Create Provider File
Create `src/inflow_unified_ai/providers/cohere.py`:
...
```

---

#### /add-model

**Purpose**: Add a new model to the ModelRegistry.

**How to use**:
```
/add-model
```

Or with context:
```
/add-model gpt-4.5-turbo with 256k context
```

**What it does**:
1. Checks if ModelFamily exists
2. Shows how to add to registry
3. Provides test template
4. Lists documentation updates

**Example**:
```
User: /add-model claude-4-sonnet

Copilot: # Add New Model to Registry

## Step 1: Check ModelFamily
Looking for CLAUDE_4_SONNET in ModelFamily enum...

## Step 2: Add to Registry
Add to `src/inflow_unified_ai/models/registry.py`:
...
```

---

#### /release

**Purpose**: Complete release checklist for publishing.

**How to use**:
```
/release
```

Or with version:
```
/release 0.2.0
```

**What it does**:
1. Pre-release checks (tests, lint)
2. Version update steps
3. Publishing workflow
4. Verification steps

**Example**:
```
User: /release 0.2.0

Copilot: # Release Checklist

## Pre-Release Checks
- [ ] Run: pytest tests/ -v
- [ ] Run: ruff check src/
...

## Step 1: Update Version
Edit pyproject.toml:
version = "0.2.0"
...
```

---

## Copilot Instructions

Copilot Instructions are automatically applied to all interactions. You don't need to do anything special.

### What's Included

#### Main Instructions (`.github/copilot-instructions.md`)
- Project overview and structure
- Development patterns and workflows
- Agent modes documentation
- Chat modes documentation
- Prompt documentation
- CI/CD reference

#### Default Instructions (`.github/instructions/default.instructions.md`)
- Applies to all file types (`**`)
- Requires todo list before work
- Code quality standards
- Testing requirements

### How It Works

When you open this project in VS Code:
1. Copilot automatically loads `.github/copilot-instructions.md`
2. Default instructions apply to all files
3. Copilot understands project context
4. Responses follow project conventions

---

## Best Practices

### 1. Start with Planning

Before implementing features:
```
1. Use Plan mode to outline changes
2. Review the plan
3. Then implement with @python-package-dev
```

### 2. Use Agents for Their Expertise

| Task | Best Agent |
|------|------------|
| Writing source code | @python-package-dev |
| Writing tests | @test-writer |
| Writing docs | @docs-writer |

### 3. Follow the Prompts

For common tasks, use prompts:
- Adding provider? Use `/add-provider`
- Adding model? Use `/add-model`
- Releasing? Use `/release`

### 4. Review Before Committing

Use Review mode to check your changes:
```
1. Make your changes
2. Switch to Review mode
3. Ask: "Review my changes"
4. Address feedback
5. Commit
```

### 5. Be Specific with Context

```
❌ "Add a feature"
✅ "Add retry logic to AzureOpenAIProvider.generate() with exponential backoff"
```

### 6. Reference Files

```
❌ "Fix the bug"
✅ "Fix the async context manager in src/inflow_unified_ai/providers/base.py"
```

---

## Troubleshooting

### Agents Not Appearing

**Problem**: Can't see `@python-package-dev` in suggestions.

**Solutions**:
1. Ensure `.github/agents/` folder exists
2. Check file naming: `{name}.agent.md`
3. Reload VS Code window: `Ctrl+Shift+P` → "Reload Window"
4. Verify Copilot Chat extension is updated

### Chat Modes Not Working

**Problem**: Chat modes don't appear in dropdown.

**Solutions**:
1. Ensure `.github/chatmodes/` folder exists
2. Check file naming: `{name}.chatmode.md`
3. Reload VS Code window
4. Check Copilot Chat version (needs latest)

### Prompts Not Recognized

**Problem**: `/add-provider` doesn't work.

**Solutions**:
1. Ensure `.github/prompts/` folder exists
2. Check file naming: `{name}.prompt.md`
3. Try typing full path: `/add-provider`
4. Reload VS Code window

### Instructions Not Applied

**Problem**: Copilot doesn't follow project conventions.

**Solutions**:
1. Verify `.github/copilot-instructions.md` exists
2. Check file isn't empty
3. Open a file in the workspace first
4. Reload VS Code window

### General Fixes

1. **Reload Window**: `Ctrl+Shift+P` → "Developer: Reload Window"
2. **Update Extensions**: Check for Copilot updates
3. **Clear Cache**: Restart VS Code completely
4. **Check Logs**: `Ctrl+Shift+P` → "GitHub Copilot: Open Logs"

---

## File Reference

| File | Purpose |
|------|---------|
| `.github/copilot-instructions.md` | Main project instructions |
| `.github/instructions/default.instructions.md` | Default rules for all files |
| `.github/agents/python-package-dev.agent.md` | Package dev agent |
| `.github/agents/test-writer.agent.md` | Test writing agent |
| `.github/agents/docs-writer.agent.md` | Documentation agent |
| `.github/chatmodes/Plan.chatmode.md` | Planning mode |
| `.github/chatmodes/Review.chatmode.md` | Review mode |
| `.github/prompts/add-provider.prompt.md` | Add provider workflow |
| `.github/prompts/add-model.prompt.md` | Add model workflow |
| `.github/prompts/release.prompt.md` | Release checklist |

---

## Learn More

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [VS Code Copilot Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
- [Copilot Chat Extension](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)
