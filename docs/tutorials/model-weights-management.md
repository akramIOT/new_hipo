# Model Weights Management Tutorial

This tutorial demonstrates how to use HIPO's model weights management system to securely store, track, and distribute machine learning model weights across multiple cloud providers.

## Overview

Large Language Models (LLMs) and other modern ML models often have weights files that are several gigabytes in size. Managing these large files across development and production environments presents several challenges:

1. **Storage**: Where and how to store these large files efficiently
2. **Versioning**: Tracking different versions of model weights 
3. **Distribution**: Getting the right weights to the right environment
4. **Security**: Ensuring weights for proprietary models remain secure
5. **Cross-cloud availability**: Making weights available across cloud providers

HIPO's model weights management system addresses these challenges with a unified API for model weights operations across cloud providers.

## Prerequisites

Before you begin, make sure you have:

1. HIPO installed and configured
2. AWS and/or GCP credentials configured
3. Appropriate S3 buckets or GCS buckets created
4. Python 3.8 or higher

## Configuration

First, configure the model weights management in your configuration file:

```yaml
# config/default_config.yaml
aws:
  enabled: true
  region: "us-west-2"
  model_weights:
    s3_bucket: "my-llm-models"  # Replace with your S3 bucket name

gcp:
  enabled: true
  project_id: "my-llm-project"
  region: "us-central1"
  model_weights:
    gcs_bucket: "my-llm-models"  # Replace with your GCS bucket name

model_weights:
  storage_type: "aws"  # Primary storage provider: aws, gcp, azure, or local
  sync_enabled: true   # Enable cross-cloud synchronization
  versioning_enabled: true
  access_control_enabled: true
  encryption_enabled: true
  checksum_algorithm: "sha256"
  cache_enabled: true
  cache_directory: "weights_cache"
  cache_max_size_gb: 20
```

## Initializing the Secret Manager

The `SecretManager` class is responsible for model weights management. Initialize it with your configuration:

```python
import yaml
from src.secrets.secret_manager import SecretManager
from src.cloud.factory import CloudProviderFactory

# Load configuration
with open("config/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create cloud providers
factory = CloudProviderFactory()
cloud_providers = {}

if config.get("aws", {}).get("enabled", False):
    cloud_providers["aws"] = factory.create_provider("aws", config["aws"])
    
if config.get("gcp", {}).get("enabled", False):
    cloud_providers["gcp"] = factory.create_provider("gcp", config["gcp"])

# Create and initialize the secret manager
secret_manager = SecretManager(config, cloud_providers)
secret_manager.start()
```

## Uploading Model Weights

To upload model weights to cloud storage:

```python
# Upload model weights
model_name = "llama-7b"
local_path = "/path/to/llama-7b-weights"

# Upload with automatic versioning (timestamp-based)
result = secret_manager.upload_model_weights(model_name, local_path)
if result:
    print(f"Successfully uploaded {model_name} weights")
else:
    print(f"Failed to upload {model_name} weights")

# Upload with specific version
version = "v1.0.0"
result = secret_manager.upload_model_weights(model_name, local_path, version=version)
```

Under the hood, the system performs these actions:

1. Calculates a checksum of the weights for integrity verification
2. Uploads the weights to the primary storage provider (AWS S3 in this example)
3. Records metadata about the weights (model name, version, checksum, upload time)
4. If cross-cloud sync is enabled, replicates the weights to secondary providers (GCP in this example)

## Listing Available Models

To list available models and their versions:

```python
# Get all available models and versions
models = secret_manager.list_available_models()

# Print the results
print("Available models:")
for model_name, versions in models.items():
    print(f"  {model_name}:")
    for version in versions:
        print(f"    - {version}")
```

Example output:

```
Available models:
  llama-7b:
    - 20231220123045
    - 20231105093012
    - v1.0.0
  gpt-j-6b:
    - 20231118145723
    - 20230922104532
  bert-base:
    - 20231115083045
    - 20230901062211
    - 20230612041839
    - 20230405121502
```

## Downloading Model Weights

To download model weights from cloud storage:

```python
model_name = "llama-7b"
output_path = "/output/path"

# Download the latest version
result = secret_manager.download_model_weights(model_name, output_path)
if result:
    print(f"Successfully downloaded latest {model_name} weights")
else:
    print(f"Failed to download {model_name} weights")

# Download a specific version
version = "v1.0.0"
result = secret_manager.download_model_weights(model_name, output_path, version=version)

# Download from a specific provider (useful for region-specific deployments)
result = secret_manager.download_model_weights(
    model_name, 
    output_path, 
    version=version,
    preferred_provider="gcp"
)
```

During the download process, the system:

1. Tries to download from the preferred provider first (if specified)
2. Falls back to the primary provider if the preferred provider fails
3. If both fail, tries secondary providers
4. After download, verifies the checksum to ensure integrity
5. Creates any necessary directories in the output path

## Deleting Model Weights

To delete model weights from cloud storage:

```python
model_name = "llama-7b"
version = "20231105093012"

# Delete from all providers
result = secret_manager.delete_model_weights(model_name, version)
if result:
    print(f"Successfully deleted {model_name} version {version}")
else:
    print(f"Failed to delete {model_name} version {version}")

# Delete only from primary provider
result = secret_manager.delete_model_weights(
    model_name, 
    version, 
    delete_all_providers=False
)
```

## Security Features

The model weights management system includes several security features:

### Encryption

Model weights are encrypted at rest and in transit:

```yaml
model_weights:
  encryption_enabled: true  # Enable encryption
```

To rotate encryption keys periodically:

```python
# Rotate keys for all models
secret_manager.rotate_model_weights_encryption()

# Rotate keys for a specific model
secret_manager.rotate_model_weights_encryption("llama-7b")
```

### Access Control

Access to model weights can be controlled through cloud provider-specific IAM policies:

```yaml
model_weights:
  access_control_enabled: true  # Enable access control
```

### Integrity Verification

The system automatically calculates checksums of model weights and verifies them during download:

```yaml
model_weights:
  checksum_algorithm: "sha256"  # Checksum algorithm for integrity verification
```

## Example: Complete Model Weights Workflow

Here's a complete example of a model weights workflow:

```python
import os
import yaml
import logging
from src.secrets.secret_manager import SecretManager
from src.cloud.factory import CloudProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create cloud providers
factory = CloudProviderFactory()
cloud_providers = {}

if config.get("aws", {}).get("enabled", False):
    cloud_providers["aws"] = factory.create_provider("aws", config["aws"])
    
if config.get("gcp", {}).get("enabled", False):
    cloud_providers["gcp"] = factory.create_provider("gcp", config["gcp"])

# Create and initialize the secret manager
secret_manager = SecretManager(config, cloud_providers)
secret_manager.start()

try:
    # 1. Upload model weights
    model_name = "llama-7b"
    local_path = "./models/llama-7b"
    
    logger.info(f"Uploading {model_name} weights from {local_path}")
    result = secret_manager.upload_model_weights(model_name, local_path)
    if result:
        logger.info(f"Successfully uploaded {model_name} weights")
    else:
        logger.error(f"Failed to upload {model_name} weights")
        exit(1)
    
    # 2. List available models
    logger.info("Listing available models")
    models = secret_manager.list_available_models()
    for model_name, versions in models.items():
        logger.info(f"  {model_name}: {versions}")
    
    # 3. Download the model weights to a different location
    output_path = "./deployed_models/llama-7b"
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Downloading {model_name} weights to {output_path}")
    result = secret_manager.download_model_weights(model_name, output_path)
    if result:
        logger.info(f"Successfully downloaded {model_name} weights")
    else:
        logger.error(f"Failed to download {model_name} weights")
        exit(1)
    
    # 4. Use the model for inference
    logger.info(f"Model weights available at {output_path}")
    logger.info("Ready for inference")
    
finally:
    # Always stop the secret manager when done
    secret_manager.stop()
```

## Command-Line Example

HIPO also provides a command-line example for managing model weights. Run the following command to see available options:

```bash
python examples/manage_model_weights_example.py --help
```

To upload model weights:

```bash
python examples/manage_model_weights_example.py --action upload --model-name llama-7b --model-path ./models/llama-7b
```

To list available models:

```bash
python examples/manage_model_weights_example.py --action list
```

To download model weights:

```bash
python examples/manage_model_weights_example.py --action download --model-name llama-7b --output-path ./downloaded_models
```

To download a specific version:

```bash
python examples/manage_model_weights_example.py --action download --model-name llama-7b --output-path ./downloaded_models --version 20231220123045
```

To delete model weights:

```bash
python examples/manage_model_weights_example.py --action delete --model-name llama-7b --version 20231220123045
```

## Best Practices

1. **Structured Model Directories**: Organize your model weights in a structured way (e.g., model name/version)
2. **Version Naming**: Use semantic versioning or timestamp-based versioning consistently
3. **Cache Management**: Clean up the cache periodically to avoid disk space issues
4. **Regular Rotation**: Rotate encryption keys regularly for sensitive models
5. **Cross-Region Availability**: Configure multiple regions for critical models to ensure high availability
6. **Checksums**: Always verify checksums after downloading to ensure integrity
7. **Cleanup**: Delete old versions that are no longer needed to save storage costs

## Conclusion

The HIPO model weights management system provides a secure, efficient, and cloud-agnostic way to manage large model weight files. It integrates seamlessly with the rest of the HIPO infrastructure and provides both programmatic and command-line interfaces for ease of use.

By centralizing model weights management across cloud providers, it simplifies MLOps workflows and ensures consistency across development and production environments.

For more details, see the [Secret Management API documentation](../api/secrets.md).