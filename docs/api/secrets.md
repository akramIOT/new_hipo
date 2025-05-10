# Secret Management API

The Secret Management API provides secure storage, retrieval, and management of secrets and model weights across multiple cloud providers.

## Overview

The Secret Manager is a critical component of the HIPO infrastructure that handles:

1. **Secret Management**: Securely store and retrieve sensitive information like API keys, credentials, and certificates
2. **Model Weights Management**: Efficiently upload, download, and sync large model weights files across cloud providers
3. **Rotation**: Automatically rotate secrets based on configurable schedules
4. **Cross-Cloud Synchronization**: Ensure secrets and model weights are consistently available across all cloud providers

## SecretManager Class

The `SecretManager` class is the central component for secret and model weights management:

```python
from src.secrets.secret_manager import SecretManager

# Create and initialize a secret manager
secret_manager = SecretManager(config, cloud_providers)
secret_manager.start()
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `Dict[str, Any]` | Configuration dictionary for the secret manager |
| `cloud_providers` | `Dict[str, CloudProvider]` | Dictionary of cloud provider implementations |

### Configuration Options

```yaml
# Secret manager configuration
secrets:
  # HashiCorp Vault configuration (optional)
  vault:
    enabled: false
    address: "https://vault.example.com:8200"
    auth_method: "kubernetes"
    
  # Model weights configuration
  model_weights:
    storage_type: "aws"  # Primary storage provider (aws, gcp, azure, or local)
    s3_bucket: "llm-models"  # AWS S3 bucket name
    gcs_bucket: "llm-models"  # GCP Cloud Storage bucket name
    azure_container: "llm-models"  # Azure Blob Storage container name
    sync_enabled: true  # Enable cross-cloud synchronization
    versioning_enabled: true  # Enable model versioning
    access_control_enabled: true  # Enable access control
    encryption_enabled: true  # Enable encryption
    checksum_algorithm: "sha256"  # Checksum algorithm for integrity verification
    cache_enabled: true  # Enable local caching
    cache_directory: "weights_cache"  # Local cache directory
    cache_max_size_gb: 10  # Maximum cache size in GB
    
  # Secret rotation configuration
  rotation:
    enabled: false  # Enable automatic secret rotation
    schedule: "0 0 * * 0"  # Cron-style schedule (weekly on Sunday at midnight)
```

## Secret Management API

### get_secret

Retrieve a secret from the primary or a specific cloud provider.

```python
secret_data = secret_manager.get_secret("api-keys")
# Retrieve from a specific provider
aws_secret = secret_manager.get_secret("aws-credentials", provider_name="aws")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `secret_name` | `str` | The name of the secret to retrieve |
| `provider_name` | `Optional[str]` | Specific cloud provider to retrieve from. If None, use primary provider |

**Returns:**

Dictionary containing the secret data.

### create_secret

Create a new secret in the primary or all cloud providers.

```python
# Create a secret
secret_manager.create_secret("api-keys", {
    "openai": "sk-1234567890",
    "huggingface": "hf_1234567890"
})

# Create in a specific provider
secret_manager.create_secret("aws-credentials", {
    "access_key_id": "AKIAXXXXXXXX",
    "secret_access_key": "xxxxxxxxxxxxxxxx",
    "region": "us-west-2"
}, provider_name="aws")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `secret_name` | `str` | The name of the secret to create |
| `secret_data` | `Dict[str, Any]` | The secret data to store |
| `provider_name` | `Optional[str]` | Specific cloud provider to create in. If None, create in all providers |

**Returns:**

Boolean indicating success or failure.

### update_secret

Update an existing secret in the primary or all cloud providers.

```python
# Update a secret
secret_manager.update_secret("api-keys", {
    "openai": "sk-0987654321",
    "huggingface": "hf_0987654321",
    "anthropic": "sk-ant-1234567890"
})
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `secret_name` | `str` | The name of the secret to update |
| `secret_data` | `Dict[str, Any]` | The new secret data |
| `provider_name` | `Optional[str]` | Specific cloud provider to update in. If None, update in all providers |

**Returns:**

Boolean indicating success or failure.

### get_rotation_status

Get the status of secret rotation.

```python
rotation_status = secret_manager.get_rotation_status()
```

**Returns:**

Dictionary containing rotation status information.

## Model Weights Management API

### upload_model_weights

Upload model weights to cloud storage across multiple providers.

```python
# Upload model weights with automatic versioning
secret_manager.upload_model_weights("llama-7b", "/path/to/model/weights")

# Upload with a specific version
secret_manager.upload_model_weights("llama-7b", "/path/to/model/weights", version="v1.0")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the model (e.g., "llama-7b", "gpt-j-6b") |
| `local_path` | `str` | Local path to model weights directory or file |
| `version` | `Optional[str]` | Optional version string. If not provided, a timestamp will be used |

**Returns:**

Boolean indicating success or failure.

### download_model_weights

Download model weights from cloud storage.

```python
# Download the latest version
secret_manager.download_model_weights("llama-7b", "/output/path")

# Download a specific version
secret_manager.download_model_weights("llama-7b", "/output/path", version="20230815")

# Download from a specific provider
secret_manager.download_model_weights("llama-7b", "/output/path", 
                                     preferred_provider="gcp")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the model (e.g., "llama-7b", "gpt-j-6b") |
| `local_path` | `str` | Local path to download the model weights to |
| `version` | `Optional[str]` | Specific version to download. If None, the latest version will be used |
| `preferred_provider` | `Optional[str]` | Preferred cloud provider to download from. If None, use primary |

**Returns:**

Boolean indicating success or failure.

### list_available_models

List available models and their versions across all cloud providers.

```python
models = secret_manager.list_available_models()
for model_name, versions in models.items():
    print(f"{model_name}: {versions}")
```

**Returns:**

Dictionary mapping model names to lists of available versions.

### delete_model_weights

Delete model weights from cloud storage.

```python
# Delete from primary provider only
secret_manager.delete_model_weights("llama-7b", "20230815", delete_all_providers=False)

# Delete from all providers
secret_manager.delete_model_weights("llama-7b", "20230815")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Name of the model to delete |
| `version` | `str` | Version of the model to delete |
| `delete_all_providers` | `bool` | If True, delete from all providers. If False, delete only from primary |

**Returns:**

Boolean indicating success or failure.

### rotate_model_weights_encryption

Rotate encryption keys for model weights.

```python
# Rotate keys for a specific model
secret_manager.rotate_model_weights_encryption("llama-7b")

# Rotate keys for all models
secret_manager.rotate_model_weights_encryption()
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `Optional[str]` | Specific model to rotate keys for. If None, rotate for all models |

**Returns:**

Boolean indicating success or failure.

## Example Usage

```python
from src.secrets.secret_manager import SecretManager
from src.cloud.factory import CloudProviderFactory

# Create cloud providers
factory = CloudProviderFactory()
cloud_providers = {
    "aws": factory.create_provider("aws", aws_config),
    "gcp": factory.create_provider("gcp", gcp_config)
}

# Create and initialize the secret manager
secret_manager = SecretManager(config, cloud_providers)
secret_manager.start()

try:
    # Store API keys as a secret
    api_keys = {
        "openai": "sk-1234567890",
        "huggingface": "hf_1234567890"
    }
    secret_manager.create_secret("api-keys", api_keys)
    
    # Upload model weights
    result = secret_manager.upload_model_weights("llama-7b", "./models/llama-7b")
    print(f"Upload successful: {result}")
    
    # List available models
    models = secret_manager.list_available_models()
    print("Available models:")
    for model_name, versions in models.items():
        print(f"  {model_name}: {versions}")
    
    # Download model weights
    result = secret_manager.download_model_weights("llama-7b", "./downloaded_models")
    print(f"Download successful: {result}")
    
    # Retrieve API keys
    retrieved_keys = secret_manager.get_secret("api-keys")
    print(f"Retrieved API keys: {retrieved_keys}")
    
finally:
    # Stop the secret manager
    secret_manager.stop()
```

## Best Practices

1. **Rotation**: Enable automatic secret rotation for sensitive credentials.
2. **Cross-Cloud Replication**: Enable replication to ensure availability in case of provider outages.
3. **Versioning**: Use versioning for model weights to maintain a history of changes.
4. **Encryption**: Always enable encryption for sensitive model weights.
5. **Checksumming**: Use checksumming to verify integrity of model weights during transfers.
6. **Access Control**: Implement strict access control policies for sensitive models.
7. **Caching**: Enable caching to improve performance for frequently used models.