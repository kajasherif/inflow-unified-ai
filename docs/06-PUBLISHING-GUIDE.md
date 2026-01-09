# Publishing Guide

## Overview

This guide covers how to publish the inflow-unified-ai package to TestPyPI (for testing) and PyPI (for production).

## Publishing Methods

| Method | Use Case |
|--------|----------|
| CI/CD (Recommended) | Automated publishing via GitHub Actions |
| Manual | Local publishing for debugging |

## Automated Publishing (CI/CD)

### Publishing to TestPyPI

TestPyPI publishing happens automatically when you push to the `main` branch.

```bash
# 1. Make your changes
git add .
git commit -m "Your changes"

# 2. Push to main
git push origin main
```

The pipeline will:
1. Run lint and tests
2. Build the package
3. Publish to TestPyPI automatically

**View progress**: https://github.com/kajasherif/inflow-unified-ai/actions

**View package**: https://test.pypi.org/project/inflow-unified-ai/

### Publishing to PyPI (Production)

PyPI publishing happens when you create a version tag.

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Commit the version bump
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin main

# 3. Create and push a version tag
git tag v0.2.0
git push origin v0.2.0
```

The pipeline will:
1. Run all checks
2. Build the package
3. Publish to PyPI
4. Create a GitHub Release

## Setting Up Trusted Publishing

### What is Trusted Publishing?

Trusted Publishing uses OpenID Connect (OIDC) to authenticate GitHub Actions with PyPI. This is more secure than using API tokens because:

- No secrets stored in GitHub
- Permissions are scoped to specific workflows
- Automatic credential rotation

### TestPyPI Setup

1. **Create TestPyPI Account**
   - Go to: https://test.pypi.org/account/register/
   - Verify your email

2. **Add Trusted Publisher**
   - Go to: https://test.pypi.org/manage/account/publishing/
   - Scroll to "Add a new pending publisher"
   - Fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `inflow-unified-ai` |
   | Owner | `kajasherif` |
   | Repository name | `inflow-unified-ai` |
   | Workflow name | `ci-cd.yml` |
   | Environment name | `testpypi` |

   - Click "Add"

### PyPI Setup (Production)

1. **Create PyPI Account**
   - Go to: https://pypi.org/account/register/
   - Verify your email
   - Enable 2FA (required for trusted publishing)

2. **Add Trusted Publisher**
   - Go to: https://pypi.org/manage/account/publishing/
   - Fill in:

   | Field | Value |
   |-------|-------|
   | PyPI Project Name | `inflow-unified-ai` |
   | Owner | `kajasherif` |
   | Repository name | `inflow-unified-ai` |
   | Workflow name | `ci-cd.yml` |
   | Environment name | `pypi` |

   - Click "Add"

### GitHub Environments Setup

1. **Navigate to Settings**
   - Go to: https://github.com/kajasherif/inflow-unified-ai/settings/environments

2. **Create `testpypi` Environment**
   - Click "New environment"
   - Name: `testpypi`
   - Click "Configure environment"
   - Optionally add protection rules

3. **Create `pypi` Environment**
   - Click "New environment"
   - Name: `pypi`
   - Click "Configure environment"
   - **Recommended**: Add "Required reviewers" for production safety

## Manual Publishing

### Prerequisites

```bash
# Install build tools
pip install build twine
```

### Building the Package

```bash
# Navigate to project root
cd d:\workspace\rnd\inflow-unified-ai

# Clean previous builds
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

# Build the package
python -m build
```

This creates:
- `dist/inflow_unified_ai-0.1.0-py3-none-any.whl` (wheel)
- `dist/inflow_unified_ai-0.1.0.tar.gz` (source distribution)

### Validating the Package

```bash
# Check package metadata
twine check dist/*
```

### Publishing to TestPyPI (Manual)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your TestPyPI API token

### Publishing to PyPI (Manual)

```bash
# Upload to PyPI
twine upload dist/*
```

### Using .pypirc for Authentication

Create `~/.pypirc` (or `$HOME/.pypirc` on Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN
```

Then upload without prompts:

```bash
twine upload --repository testpypi dist/*
twine upload --repository pypi dist/*
```

## Version Management

### Semantic Versioning

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH
  │     │     └── Bug fixes, backwards compatible
  │     └──────── New features, backwards compatible
  └────────────── Breaking changes
```

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.1` → `0.2.0`: New feature
- `0.2.0` → `1.0.0`: Breaking change or stable release

### Updating Version

1. **Edit pyproject.toml**
   ```toml
   [project]
   version = "0.2.0"  # Update this
   ```

2. **Commit and Tag**
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git tag v0.2.0
   git push origin main --tags
   ```

## Verifying Installation

### From TestPyPI

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ inflow-unified-ai

# Test import
python -c "from inflow_unified_ai import AIClient; print('Success!')"
```

### From PyPI

```bash
# Install from PyPI
pip install inflow-unified-ai

# Test import
python -c "from inflow_unified_ai import AIClient; print('Success!')"
```

## Troubleshooting

### "File already exists" Error

**Cause**: Version already published to PyPI

**Fix**: 
- Bump version number in `pyproject.toml`
- You cannot overwrite existing versions on PyPI

### "Invalid credentials" Error

**For CI/CD**:
1. Verify trusted publisher is configured
2. Check environment name matches exactly
3. Verify workflow file name is correct

**For Manual Upload**:
1. Check `.pypirc` token is correct
2. Ensure token has upload permissions
3. Token format should be `pypi-...`

### "Package not found" After Publishing

**Cause**: PyPI index takes time to update

**Fix**: Wait 1-2 minutes and try again

### Build Fails

**Check**:
```bash
# Verify pyproject.toml is valid
pip install build
python -m build --dry-run

# Check for syntax errors
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

## Release Checklist

Before every release:

- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG.md (if you have one)
- [ ] Run tests locally: `pytest tests/ -v`
- [ ] Run linter: `ruff check src/`
- [ ] Commit all changes
- [ ] Push to main (triggers TestPyPI)
- [ ] Verify TestPyPI installation works
- [ ] Create version tag (triggers PyPI)
- [ ] Verify PyPI installation works
- [ ] Update documentation if needed

## Links

| Resource | URL |
|----------|-----|
| TestPyPI Package | https://test.pypi.org/project/inflow-unified-ai/ |
| PyPI Package | https://pypi.org/project/inflow-unified-ai/ |
| GitHub Actions | https://github.com/kajasherif/inflow-unified-ai/actions |
| GitHub Releases | https://github.com/kajasherif/inflow-unified-ai/releases |
