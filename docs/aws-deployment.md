# AWS Deployment Guide

This document explains how to deploy HIPO to AWS using the CI/CD pipeline.

## AWS Infrastructure Setup

Before using the deployment workflow, you'll need to set up the following AWS resources:

### 1. ECR Repository

Create an ECR repository to store Docker images:

```bash
aws ecr create-repository --repository-name hipo --region us-west-2
```

### 2. ECS Cluster and Service

Create an ECS cluster and service to run containers:

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name hipo-cluster --region us-west-2

# Create service (after task definition is created)
aws ecs create-service \
  --cluster hipo-cluster \
  --service-name hipo-api \
  --task-definition hipo-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-zzz],assignPublicIp=ENABLED}"
```

### 3. IAM Roles

Ensure the following IAM roles are available:

- `ecsTaskExecutionRole`: Used by ECS to execute tasks
- `ecsTaskRole`: Used by the task itself for AWS API calls

### 4. Secrets Manager

Create secrets for AWS credentials:

```bash
aws secretsmanager create-secret \
  --name hipo/aws/credentials \
  --secret-string '{"access_key_id":"YOUR_ACCESS_KEY","secret_access_key":"YOUR_SECRET_KEY"}' \
  --region us-west-2
```

## CI/CD Configuration

The deployment is handled by the `.github/workflows/aws.yml` workflow, which:

1. Builds and pushes a Docker image to ECR
2. Updates the task definition with the new image
3. Deploys the updated task definition to ECS

### Required GitHub Secrets

The following secrets need to be configured in your GitHub repository:

- `AWS_ACCESS_KEY_ID`: AWS access key with permissions for ECR, ECS, and Secrets Manager
- `AWS_SECRET_ACCESS_KEY`: Corresponding secret access key

### Workflow Configuration

The workflow is configured with environment variables in the `aws.yml` file:

```yaml
env:
  AWS_REGION: us-west-2                      # AWS region
  ECR_REPOSITORY: hipo                       # ECR repository name
  ECS_SERVICE: hipo-api                      # ECS service name
  ECS_CLUSTER: hipo-cluster                  # ECS cluster name
  ECS_TASK_DEFINITION: .aws/task-definition.json # Task definition file
  CONTAINER_NAME: hipo-api                   # Container name
```

You may need to adjust these values to match your AWS environment.

## Task Definition

The task definition in `.aws/task-definition.json` defines the container configuration for ECS. It includes:

- Container image
- Port mappings
- Environment variables
- Secrets
- CPU and memory requirements
- Health check configuration

## Troubleshooting

If deployments fail, check the following:

1. **Image Build Failures**:
   - Ensure Docker builds locally
   - Check ECR permissions

2. **Task Definition Errors**:
   - Validate task definition format
   - Ensure container name matches

3. **Deployment Failures**:
   - Check ECS service logs
   - Verify network configuration
   - Ensure target group health checks pass

## Monitoring Deployments

To monitor deployments:

```bash
# Check service status
aws ecs describe-services --cluster hipo-cluster --services hipo-api

# Check task status
aws ecs list-tasks --cluster hipo-cluster --service-name hipo-api
aws ecs describe-tasks --cluster hipo-cluster --tasks <task-id>

# Check CloudWatch logs
aws logs get-log-events --log-group-name /ecs/hipo --log-stream-name <log-stream-name>
```