# Secure Model Deployment Tutorial

This tutorial demonstrates how to securely deploy an LLM model using HIPO's secure model weights storage and multi-cloud Kubernetes infrastructure.

## Prerequisites

- HIPO installed and configured
- Access to at least one cloud provider (AWS, GCP, or Azure)
- A trained model to deploy

## Step 1: Configure Secure Storage

First, ensure your configuration includes secure storage settings. Edit `config/default_config.yaml`:

```yaml
security:
  encryption:
    key_directory: 'secrets'
    load_keys_from_env: false
    
  secure_weights:
    enabled: true
    storage:
      primary: 's3'  # Choose your primary storage provider
      replicate_to: ['gcs']  # Optional redundancy
      s3_bucket: 'my-llm-models'
      gcs_bucket: 'my-llm-models'
      versioning_enabled: true
      checksum_algorithm: 'sha256'
      access_control_enabled: true
      encryption_enabled: true
    
    cache:
      enabled: true
      directory: 'weights_cache'
      max_size_gb: 10

secrets:
  model_weights:
    storage_type: 's3'
    s3_bucket: 'my-llm-models'
    sync_enabled: true
    versioning_enabled: true
```

## Step 2: Initialize Secure Storage Components

Create a Python script `secure_deploy.py`:

```python
import os
from pathlib import Path

from src.config.config import load_config
from src.security.encryption import EncryptionService
from src.models.secure_weights import create_secure_weights_manager
from src.cloud.factory import CloudProviderFactory
from src.secrets.secret_manager import SecretManager
from src.kubernetes.orchestrator import KubernetesOrchestrator

# Load configuration
config = load_config("config/default_config.yaml")

# Initialize cloud providers
cloud_factory = CloudProviderFactory(config)
cloud_providers = {
    "aws": cloud_factory.create_provider("aws"),
    "gcp": cloud_factory.create_provider("gcp")
}

# Initialize encryption service
encryption_service = EncryptionService(config.get("security", {}).get("encryption", {}))

# Initialize secret manager
secret_manager = SecretManager(config, cloud_providers)

# Initialize secure weights manager
secure_weights_config = secret_manager.manage_model_weights()
secure_weights = create_secure_weights_manager(
    config={"secure_weights": secure_weights_config},
    encryption_service=encryption_service,
    secret_manager=secret_manager
)

# Initialize Kubernetes orchestrator
orchestrator = KubernetesOrchestrator(config, cloud_providers)
```

## Step 3: Upload Model Weights Securely

Add the following to your script:

```python
def upload_model_weights(model_name, weights_path):
    """Upload model weights securely to multi-cloud storage."""
    # Generate a version based on timestamp
    from datetime import datetime
    version = f"v_{int(datetime.now().timestamp())}"
    
    print(f"Uploading {model_name} weights version {version}...")
    
    # Add metadata for the model
    metadata = {
        "model_type": "llm",
        "framework": "pytorch",
        "parameters": "7B",
        "trained_on": datetime.now().isoformat(),
        "description": "Llama 2 fine-tuned model"
    }
    
    # Store weights securely
    result = secure_weights.store_weights(
        model_name=model_name,
        weights_file=weights_path,
        version=version,
        metadata=metadata,
        encrypt=True  # Enable encryption
    )
    
    print(f"Upload complete. Stored in {len(result['storage_locations'])} locations:")
    for location in result['storage_locations']:
        print(f"- {location['provider']}")
    
    return result

# Path to your model weights
model_weights_path = "/path/to/llama2-7b-weights.bin"
model_name = "llama2-7b-finetuned"

# Upload the model
upload_result = upload_model_weights(model_name, model_weights_path)
```

## Step 4: Deploy the Model to Kubernetes

Add the deployment code to your script:

```python
def deploy_model_to_kubernetes(model_name, version, provider="aws"):
    """Deploy the model to Kubernetes."""
    print(f"Deploying {model_name} version {version} to Kubernetes...")
    
    # Create a deployment configuration
    deployment_config = {
        "name": model_name,
        "version": version,
        "replicas": 2,
        "resources": {
            "requests": {
                "cpu": "4",
                "memory": "16Gi",
                "nvidia.com/gpu": "1"
            },
            "limits": {
                "cpu": "8",
                "memory": "32Gi",
                "nvidia.com/gpu": "1"
            }
        },
        "env": {
            "MAX_BATCH_SIZE": "8",
            "MAX_SEQUENCE_LENGTH": "2048",
            "TEMPERATURE": "0.7"
        },
        "secure_weights": True,  # Enable secure weights loading
        "provider": provider
    }
    
    # Deploy the model
    deployment = orchestrator.deploy_model(
        deployment_config=deployment_config,
        weights_manager=secure_weights
    )
    
    print(f"Deployment complete: {deployment['status']}")
    print(f"API endpoint: {deployment['api_endpoint']}")
    
    return deployment

# Deploy the model to Kubernetes
deployment = deploy_model_to_kubernetes(
    model_name=model_name,
    version=upload_result['version'],
    provider="aws"  # Primary provider
)
```

## Step 5: Verify the Deployment

Add verification code:

```python
def verify_deployment(deployment):
    """Verify the deployment is healthy."""
    import time
    import requests
    
    # Wait for deployment to be ready
    print("Waiting for deployment to be ready...")
    for i in range(12):  # Wait up to 2 minutes
        status = orchestrator.get_deployment_status(deployment['id'])
        if status['status'] == 'Running':
            print("Deployment is ready!")
            break
        print(f"Status: {status['status']} ({i+1}/12)")
        time.sleep(10)
    
    # Check that the API endpoint is responding
    try:
        response = requests.get(
            f"{deployment['api_endpoint']}/health",
            timeout=5
        )
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print(f"❌ API health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API health check failed: {e}")
    
    # Verify model weights integrity
    integrity_results = secure_weights.verify_weights_integrity(
        model_name=model_name,
        version=upload_result['version']
    )
    
    all_verified = all(integrity_results.values())
    if all_verified:
        print("✅ Model weights integrity verified")
    else:
        print("❌ Model weights integrity check failed")
        for location, verified in integrity_results.items():
            if not verified:
                print(f"  - Failed: {location}")

# Verify the deployment
verify_deployment(deployment)
```

## Step 6: Run the Script

Save and run the script:

```bash
python secure_deploy.py
```

## Step 7: Monitor the Deployment

Use the Streamlit UI to monitor your deployment:

```bash
python src/ui/run_ui.py
```

Navigate to the Monitoring page to see:
- Deployment status
- Resource usage
- API request metrics
- Cost metrics

## Conclusion

You've successfully:
1. Configured secure model weights storage
2. Uploaded a model with encryption and redundancy
3. Deployed the model to Kubernetes
4. Verified the deployment's health and integrity
5. Set up monitoring

This approach ensures your model weights are securely stored, with encryption, integrity verification, and multi-cloud redundancy. The deployment is managed by Kubernetes for scalability and resilience.