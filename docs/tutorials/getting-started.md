# Getting Started

This guide will help you get started with HIPO for multi-cloud ML model deployment.

## Installation

```bash
pip install hipo
```

## Configuration

Create a configuration file `config.yaml`:

```yaml
cloud_providers:
  aws:
    enabled: true
    region: us-west-2
  gcp:
    enabled: true
    project_id: my-project
    region: us-central1

kubernetes:
  cluster_name: ml-cluster
  namespace: ml-models

models:
  storage:
    primary: s3
    s3_bucket: ml-model-weights
    encryption_enabled: true
```

## Basic Usage

```python
from hipo import MultiCloudOrchestrator

# Initialize the orchestrator
orchestrator = MultiCloudOrchestrator(config="config.yaml")

# Deploy a model
orchestrator.deploy_model(
    name="llama2-7b",
    version="v1.0",
    weights_path="/path/to/weights",
    replicas=3,
    gpu_type="nvidia-a100"
)
```