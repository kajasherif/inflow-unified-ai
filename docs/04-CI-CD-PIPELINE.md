# CI/CD Pipeline Documentation

## Overview

The inflow-unified-ai package uses GitHub Actions for Continuous Integration and Continuous Deployment. The pipeline automatically:

1. **Lints and type-checks** code on every push
2. **Runs tests** on Python 3.10, 3.11, and 3.12
3. **Builds** the package
4. **Publishes to TestPyPI** on every push to main
5. **Publishes to PyPI** when a version tag is created

## Pipeline File Location

```
.github/workflows/ci-cd.yml
```

## Pipeline Jobs

### 1. Lint & Type Check

**Trigger**: Every push and pull request

**What it does**:
- Runs Ruff linter to check code quality
- Runs Ruff formatter to check code style
- Runs MyPy for type checking

```yaml
lint:
  name: Lint & Type Check
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install ruff mypy
    - run: ruff check src/
    - run: ruff format --check src/
    - run: mypy src/inflow_unified_ai --ignore-missing-imports
```

### 2. Test

**Trigger**: Every push and pull request

**What it does**:
- Runs unit tests on Python 3.10, 3.11, and 3.12
- Generates coverage report
- Verifies package imports work correctly

```yaml
test:
  name: Test Python ${{ matrix.python-version }}
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ['3.10', '3.11', '3.12']
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install -e ".[dev]"
    - run: pytest tests/ -v --cov=inflow_unified_ai
    - run: python -c "from inflow_unified_ai import AIClient"
```

### 3. Build Package

**Trigger**: After lint and test pass

**What it does**:
- Builds wheel (.whl) and source distribution (.tar.gz)
- Validates package with twine
- Uploads artifacts for downstream jobs

```yaml
build:
  name: Build Package
  runs-on: ubuntu-latest
  needs: [lint, test]
  steps:
    - uses: actions/checkout@v4
    - run: pip install build twine
    - run: python -m build
    - run: twine check dist/*
    - uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/
```

### 4. Publish to TestPyPI

**Trigger**: Push to main branch (after build passes)

**What it does**:
- Downloads built artifacts
- Publishes to TestPyPI using trusted publishing (OIDC)

```yaml
publish-testpypi:
  name: Publish to TestPyPI
  runs-on: ubuntu-latest
  needs: build
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  environment:
    name: testpypi
    url: https://test.pypi.org/project/inflow-unified-ai/
  permissions:
    id-token: write
  steps:
    - uses: actions/download-artifact@v4
    - uses: pypa/gh-action-pypi-publish@release/v1
      with:
        repository-url: https://test.pypi.org/legacy/
        skip-existing: true
```

### 5. Publish to PyPI

**Trigger**: Version tag (e.g., v0.1.0)

**What it does**:
- Downloads built artifacts
- Publishes to production PyPI using trusted publishing

```yaml
publish-pypi:
  name: Publish to PyPI
  runs-on: ubuntu-latest
  needs: build
  if: startsWith(github.ref, 'refs/tags/v')
  environment:
    name: pypi
    url: https://pypi.org/project/inflow-unified-ai/
  permissions:
    id-token: write
  steps:
    - uses: actions/download-artifact@v4
    - uses: pypa/gh-action-pypi-publish@release/v1
```

### 6. Create GitHub Release

**Trigger**: After PyPI publish succeeds

**What it does**:
- Creates GitHub release with auto-generated notes
- Attaches wheel and tarball as release assets

```yaml
release:
  name: Create GitHub Release
  runs-on: ubuntu-latest
  needs: publish-pypi
  if: startsWith(github.ref, 'refs/tags/v')
  permissions:
    contents: write
  steps:
    - uses: actions/checkout@v4
    - uses: actions/download-artifact@v4
    - uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
```

## Pipeline Flow Diagram

```
┌─────────────────┐
│   Push/PR       │
└────────┬────────┘
         │
    ┌────▼────┐     ┌────────────┐
    │  Lint   │────►│   Test     │
    └────┬────┘     │ (3.10/11/12)│
         │          └─────┬──────┘
         │                │
         └───────┬────────┘
                 │
          ┌──────▼──────┐
          │   Build     │
          └──────┬──────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            │
┌──────────┐ ┌──────────┐     │
│ TestPyPI │ │  PyPI    │◄────┘
│(on main) │ │(on tag)  │
└──────────┘ └────┬─────┘
                  │
            ┌─────▼─────┐
            │  Release  │
            └───────────┘
```

## Setting Up Trusted Publishing

### TestPyPI Setup

1. Go to: https://test.pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `inflow-unified-ai`
   - **Owner**: `kajasherif` (your GitHub username)
   - **Repository name**: `inflow-unified-ai`
   - **Workflow name**: `ci-cd.yml`
   - **Environment name**: `testpypi`
4. Click "Add"

### PyPI Setup (Production)

1. Go to: https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `inflow-unified-ai`
   - **Owner**: `kajasherif`
   - **Repository name**: `inflow-unified-ai`
   - **Workflow name**: `ci-cd.yml`
   - **Environment name**: `pypi`
4. Click "Add"

### GitHub Environments Setup

1. Go to: https://github.com/kajasherif/inflow-unified-ai/settings/environments
2. Create two environments:
   - `testpypi` - for TestPyPI publishing
   - `pypi` - for PyPI publishing
3. Optionally add protection rules (require approval, etc.)

## Triggering the Pipeline

### Automatic Triggers

| Event | What Runs |
|-------|-----------|
| Push to any branch | Lint + Test |
| Pull request to main | Lint + Test |
| Push to main | Lint + Test + Build + TestPyPI |
| Tag starting with `v` | Lint + Test + Build + PyPI + Release |

### Manual Release Process

```bash
# 1. Ensure all changes are committed
git status

# 2. Update version in pyproject.toml
# version = "0.2.0"

# 3. Commit the version bump
git add pyproject.toml
git commit -m "Bump version to 0.2.0"

# 4. Push to main (triggers TestPyPI)
git push origin main

# 5. Create and push version tag (triggers PyPI)
git tag v0.2.0
git push origin v0.2.0
```

## Viewing Pipeline Status

1. Go to: https://github.com/kajasherif/inflow-unified-ai/actions
2. Click on a workflow run to see details
3. Click on individual jobs to see logs

## Troubleshooting

### Lint Failures

**Error**: Ruff found linting errors

**Fix**:
```bash
# Auto-fix lint errors
ruff check src/ --fix
ruff format src/

# Commit and push
git add .
git commit -m "Fix lint errors"
git push
```

### Test Failures

**Error**: Tests failed on CI but pass locally

**Fix**:
1. Check if tests depend on environment variables
2. Ensure all dependencies are in `pyproject.toml`
3. Run tests in clean environment:
   ```bash
   python -m venv test-env
   test-env/Scripts/activate
   pip install -e ".[dev]"
   pytest tests/ -v
   ```

### Publish Failures

**Error**: Trusted publishing exchange failure

**Fix**:
1. Verify trusted publisher is configured on PyPI/TestPyPI
2. Verify GitHub environment exists with correct name
3. Check workflow file has correct environment name

**Error**: Version already exists

**Fix**:
- Bump version number in `pyproject.toml`
- Or use `skip-existing: true` in workflow (already configured)

## Security Notes

1. **Never commit API keys** - Use environment variables
2. **Use trusted publishing** - No need for PyPI tokens in secrets
3. **Protect main branch** - Require PR reviews
4. **Use environments** - Adds deployment protection
