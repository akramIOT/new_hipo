# GitHub Secrets Management for HIPO

This document describes how to set up and manage the required GitHub Secrets for the HIPO CI/CD pipeline.

## Overview

GitHub Secrets provide a secure way to store sensitive information like API tokens, credentials, and other secrets that your GitHub Actions workflows need to access. The HIPO project requires several secrets to be configured for its CI/CD pipeline to function properly.

## Required Secrets

| Secret Name | Description | Used In | Example Value |
|-------------|-------------|---------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS access key for deployment and testing | ci.yml, deploy workflows | `AKIA1234567890ABCDEF` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | ci.yml, deploy workflows | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `DOCKERHUB_USERNAME` | Docker Hub username for image pushing | ci.yml | `hipodeploy` |
| `DOCKERHUB_TOKEN` | Docker Hub API token or password | ci.yml | `dckr_pat_1234567890abcdefghijklmnopqrstuvwxyz` |
| `CODECOV_TOKEN` | Codecov.io token for coverage reporting | ci.yml | `12345678-abcd-1234-5678-abcdefghijkl` |
| `PYPI_API_TOKEN` | PyPI token for package publishing | publish.yml | `pypi-AgEIcHlwaS5vcmcCJDEyMzQ1Njc4LWFiY2...` |
| `GCP_SERVICE_ACCOUNT_KEY` | GCP service account JSON key (base64 encoded) | ci.yml | `ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgIn...` |
| `AZURE_CREDENTIALS` | Azure service principal credentials | ci.yml | `{"clientId": "...", "clientSecret": "...", ...}` |

## Setting Up Secrets

### Repository Secrets

To add secrets to your repository:

1. Navigate to your repository on GitHub
2. Go to "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret"
4. Enter the secret name (from the table above) and its value
5. Click "Add secret"

### Environment Secrets

The HIPO CI/CD pipeline uses specific GitHub Environments for deployment:

- `development` - For the development environment
- `production` - For the production environment

Each environment can have its own secrets:

1. Navigate to your repository on GitHub
2. Go to "Settings" > "Environments"
3. Click on the environment (create it if it doesn't exist)
4. Under "Environment secrets", click "Add secret"
5. Enter the secret name and value
6. Click "Add secret"

Environment-specific secrets:

| Environment | Secret Name | Description |
|-------------|-------------|-------------|
| `development` | `AWS_ACCESS_KEY_ID` | AWS credentials for dev environment |
| `development` | `AWS_SECRET_ACCESS_KEY` | AWS credentials for dev environment |
| `development` | `EKS_CLUSTER_NAME` | Name of the dev EKS cluster |
| `production` | `AWS_ACCESS_KEY_ID` | AWS credentials for prod environment |
| `production` | `AWS_SECRET_ACCESS_KEY` | AWS credentials for prod environment |
| `production` | `EKS_CLUSTER_NAME` | Name of the prod EKS cluster |

## Secret Rotation

For security best practices, rotate your secrets regularly:

1. Generate new credentials in the respective service (AWS, DockerHub, etc.)
2. Update the GitHub secret with the new value
3. Revoke the old credentials after confirming the new ones work

Recommended rotation schedule:
- Access keys / API tokens: Every 90 days
- Service account keys: Every 180 days
- DockerHub tokens: Every 90 days

## Using Secrets in Workflows

Secrets are already configured in the workflow files. Here's how they're referenced:

```yaml
# Example from CI workflow
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-west-2
```

```yaml
# Example from Docker publishing
- name: Login to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

## Access Control

GitHub Secrets can only be accessed by:
- Repository administrators
- Users with the "write" permission on the repository
- GitHub Actions workflows running in the repository

Environment secrets have additional protection:
- They can only be accessed by workflows running for the designated environment
- Required reviewers can be added for environment deployments

## Secret Values Best Practices

1. **Never commit secrets** directly to code or config files
2. **Never log secrets** in workflow outputs
3. **Use environment variables** in application code, not hardcoded values
4. **Generate strong credentials** with high entropy
5. **Limit permissions** of service accounts and access keys to only what's needed
6. **Use shorter expiry times** for tokens where possible
7. **Monitor usage** of your access keys and tokens

## Troubleshooting

If a workflow fails with credential or authentication errors:

1. Check that all required secrets are defined
2. Verify the secret name matches exactly what's used in the workflow
3. Confirm the secret value is valid and not expired
4. Check that the service account or user has the required permissions

For security concerns or to report a vulnerability, contact [sheriff.akram.usa@gmail.com](mailto:sheriff.akram.usa@gmail.com).