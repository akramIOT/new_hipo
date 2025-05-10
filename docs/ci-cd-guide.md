# CI/CD and Release Process Guide

This document provides an overview of the CI/CD pipeline and release process for the HIPO project. 

## Table of Contents

- [GitHub Actions Setup](#github-actions-setup)
- [CI Pipeline](#ci-pipeline)
- [Release Process](#release-process)
- [Docker Images](#docker-images)
- [Environment Deployments](#environment-deployments)
- [Troubleshooting](#troubleshooting)

## GitHub Actions Setup

HIPO uses GitHub Actions for continuous integration and delivery. All workflow files are located in the `.github/workflows/` directory.

### Available Workflows

| Workflow | File | Description |
|----------|------|-------------|
| CI | `ci.yml` | Main CI pipeline that runs on push and pull requests |
| Documentation | `docs.yml` | Builds and deploys documentation |
| Nightly Tests | `nightly-tests.yml` | Runs integration and performance tests nightly |
| Package Publishing | `publish.yml` | Publishes Python package to PyPI |
| Release | `release.yml` | Handles GitHub release creation |
| Security Scans | `security.yml` | Runs detailed security scans |

### Required Secrets

The following secrets need to be configured in your GitHub repository settings:

- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: Used for AWS integrations and deployments
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`: Used for Docker image publishing
- `CODECOV_TOKEN`: Used for uploading test coverage reports
- `PYPI_API_TOKEN`: Used for publishing to PyPI

## CI Pipeline

The main CI pipeline is defined in `ci.yml` and includes the following stages:

### 1. Lint

Checks code quality using:
- flake8 for PEP 8 style enforcement
- black for code formatting
- isort for import sorting 
- mypy for static type checking

```bash
# Run locally
pip install -r requirements-dev.txt
flake8 src/ tests/
black --check --line-length=127 src/ tests/
isort --check-only --profile black --line-length=127 src/ tests/
mypy src/
```

### 2. Security Scan

Scans code for security issues using:
- bandit for identifying common security issues
- safety for checking dependencies against vulnerability databases

```bash
# Run locally
pip install bandit safety
bandit -r src/ -c pyproject.toml
safety check -r requirements.txt -r requirements-dev.txt
```

### 3. Unit Tests

Runs unit tests with pytest across multiple Python versions:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

Tests are configured to generate code coverage reports.

```bash
# Run locally
pip install pytest pytest-cov
pytest tests/unit --cov=src
```

### 4. Integration Tests

Runs integration tests using LocalStack to emulate AWS services.

```bash
# Run locally with LocalStack
docker run -d -p 4566:4566 -e "SERVICES=s3,secretsmanager,lambda" localstack/localstack:latest
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:4566
pytest tests/integration
```

### 5. Build Package

Builds the Python package using the Python build system:

```bash
# Run locally
pip install build wheel twine
python -m build
twine check dist/*
```

### 6. Build Docker

Builds and optionally pushes Docker images:
- Uses Docker buildx for multi-platform support
- Caches layers for faster builds
- Tags images based on branch name and version

```bash
# Run locally
docker build -t hipodeploy/hipo:latest .
```

## Release Process

The package release process is handled automatically by the `publish.yml` workflow, which is triggered in two ways:

1. **Automatically** when a new GitHub release is created
2. **Manually** via workflow_dispatch with a version parameter

### Manual Release Steps

1. From the GitHub repository, go to Actions > Publish Python Package
2. Click "Run workflow"
3. Enter the new version (e.g., `0.2.0`)
4. Click "Run workflow" to start the process

The workflow will:
1. Update version in `pyproject.toml` and `src/__init__.py`
2. Commit and tag the version change
3. Build the package
4. Publish to PyPI
5. Create a GitHub release with release notes

### Version Scheme

We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes (X.0.0)
- MINOR version for backwards-compatible functionality (0.X.0)
- PATCH version for backwards-compatible bug fixes (0.0.X)

## Docker Images

Docker images are automatically built and pushed to Docker Hub as part of the CI pipeline:

- `akramsheriff/hipo:latest` - Latest version from the main branch
- `akramsheriff/hipo:<branch>` - Latest version from a specific branch
- `akramsheriff/hipo:<version>` - Specific version (e.g., 0.2.0)
- `akramsheriff/hipo:<major>.<minor>` - Latest patch version (e.g., 0.2)

To run the Docker image locally:

```bash
# Run API server
docker run -p 5000:5000 akramsheriff/hipo:latest

# Run Streamlit UI
docker run -p 8501:8501 akramsheriff/hipo:latest ui

# Run specific commands
docker run akramsheriff/hipo:latest train /path/to/data model_name
```

## Environment Deployments

The CI pipeline includes automatic deployments to development and production environments:

### Development Environment

- Triggered on pushes to the `develop` branch
- Deploys to EKS development cluster
- Uses configuration from `k8s/dev/`

### Production Environment

- Triggered on pushes to the `main` branch
- Deploys to EKS production cluster
- Uses configuration from `k8s/prod/`

Deployment status and URLs are available in the GitHub Actions interface.

## Troubleshooting

### Common Issues and Solutions

#### CI Pipeline Failure

1. Check the specific job that failed
2. Look at the logs for detailed error messages
3. Run the equivalent command locally to reproduce the issue
4. Fix the issue and push your changes

#### Failed Publication

1. Check that the version doesn't already exist on PyPI
2. Verify that all tests are passing
3. Ensure the package builds locally with `python -m build`
4. Check the GitHub secrets for PyPI authentication

#### Docker Build Issues

1. Verify that Dockerfile is valid with `docker build -t test .`
2. Check if Docker Hub credentials are correctly set
3. Make sure you're not exceeding resource limits

#### Local Testing

To run the full CI pipeline locally before pushing:

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run all checks
pre-commit run --all-files

# Run unit tests
pytest tests/unit

# Run integration tests (requires LocalStack)
pytest tests/integration
```

### Getting Help

If you're still having issues:

1. Check the GitHub Actions documentation
2. Review our internal documentation
3. Open an issue in the repository
4. Contact Akram Sheriff at [sheriff.akram.usa@gmail.com](mailto:sheriff.akram.usa@gmail.com)