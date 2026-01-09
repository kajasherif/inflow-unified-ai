---
mode: agent
description: Complete release checklist for publishing a new version
tools: ['codebase', 'editFiles', 'runCommands', 'problems', 'terminalLastCommand']
---

# Release Checklist

## Purpose
Step-by-step guide to release a new version of inflow-unified-ai.

## Version Information
- Current version: Check `pyproject.toml`
- New version: ${input:new_version}
- Release type: ${input:release_type} (major/minor/patch)

---

## Pre-Release Checks

### 1. Code Quality

```bash
# Run all tests
pytest tests/ -v

# Check lint
ruff check src/

# Check formatting
ruff format --check src/

# Type check
mypy src/inflow_unified_ai --ignore-missing-imports
```

**Checklist**:
- [ ] All tests pass (currently 16 tests)
- [ ] No lint errors
- [ ] Code is formatted
- [ ] Type checking passes

### 2. Review Changes

```bash
# See what's changed since last release
git log --oneline $(git describe --tags --abbrev=0)..HEAD

# Check for uncommitted changes
git status
```

**Checklist**:
- [ ] All changes are committed
- [ ] Changes are appropriate for this release
- [ ] No debug code left in
- [ ] No hardcoded secrets

### 3. Documentation

**Checklist**:
- [ ] API changes documented in `docs/07-API-REFERENCE.md`
- [ ] README.md updated if needed
- [ ] New features have examples

---

## Release Process

### Step 1: Update Version

Edit `pyproject.toml`:

```toml
[project]
name = "inflow-unified-ai"
version = "${input:new_version}"  # Update this
```

### Step 2: Commit Version Bump

```bash
git add pyproject.toml
git commit -m "Bump version to ${input:new_version}"
```

### Step 3: Push to Main (Triggers TestPyPI)

```bash
git push origin main
```

**Wait for CI/CD**:
1. Go to: https://github.com/kajasherif/inflow-unified-ai/actions
2. Wait for "CI/CD Pipeline" to complete
3. Verify all jobs pass (lint, test, build, publish-testpypi)

### Step 4: Verify TestPyPI

```bash
# Create fresh environment
python -m venv test-release
test-release\Scripts\activate  # Windows
# source test-release/bin/activate  # Linux/Mac

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai==${input:new_version}

# Test import
python -c "from inflow_unified_ai import AIClient, __version__; print(f'Version: {__version__}')"

# Deactivate and cleanup
deactivate
Remove-Item -Recurse -Force test-release  # Windows
```

**Checklist**:
- [ ] Package installs from TestPyPI
- [ ] Import works correctly
- [ ] Version is correct

### Step 5: Create and Push Tag (Triggers PyPI)

```bash
git tag v${input:new_version}
git push origin v${input:new_version}
```

**Wait for CI/CD**:
1. Go to: https://github.com/kajasherif/inflow-unified-ai/actions
2. Wait for "CI/CD Pipeline" to complete
3. Verify publish-pypi and release jobs pass

### Step 6: Verify PyPI

```bash
# Create fresh environment
python -m venv test-prod
test-prod\Scripts\activate

# Install from PyPI
pip install inflow-unified-ai==${input:new_version}

# Test import
python -c "from inflow_unified_ai import AIClient; print('Success!')"

# Cleanup
deactivate
Remove-Item -Recurse -Force test-prod
```

**Checklist**:
- [ ] Package installs from PyPI
- [ ] Import works correctly

### Step 7: Verify GitHub Release

1. Go to: https://github.com/kajasherif/inflow-unified-ai/releases
2. Verify release v${input:new_version} was created
3. Verify wheel and tarball are attached

---

## Post-Release

### 1. Announce Release (Optional)

- Update team on new release
- Post to relevant channels

### 2. Prepare for Next Development

If needed, update version to next dev version:

```toml
version = "${input:next_dev_version}"  # e.g., "0.2.1.dev0"
```

---

## Rollback (If Needed)

If issues are found after release:

### Option 1: Yank from PyPI (Prevents new installs)

```bash
# Not automated - must be done manually on PyPI
# Go to: https://pypi.org/manage/project/inflow-unified-ai/releases/
# Click on the problematic version and yank it
```

### Option 2: Release Patch

```bash
# Fix the issue
git add .
git commit -m "Fix: [description]"

# Bump to patch version
# Edit pyproject.toml: version = "${input:patch_version}"
git add pyproject.toml
git commit -m "Bump version to ${input:patch_version}"
git push origin main

# Tag and release
git tag v${input:patch_version}
git push origin v${input:patch_version}
```

---

## Quick Reference

| Step | Command |
|------|---------|
| Run tests | `pytest tests/ -v` |
| Check lint | `ruff check src/` |
| Push to main | `git push origin main` |
| Create tag | `git tag v${input:new_version}` |
| Push tag | `git push origin v${input:new_version}` |

| Link | URL |
|------|-----|
| Actions | https://github.com/kajasherif/inflow-unified-ai/actions |
| TestPyPI | https://test.pypi.org/project/inflow-unified-ai/ |
| PyPI | https://pypi.org/project/inflow-unified-ai/ |
| Releases | https://github.com/kajasherif/inflow-unified-ai/releases |
