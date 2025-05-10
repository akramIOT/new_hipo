# Security Guide

This guide covers the security features and best practices for using HIPO, with a focus on securing ML model weights and sensitive configurations.

## Table of Contents

- [Overview](#overview)
- [Secure Model Weights Storage](#secure-model-weights-storage)
  - [Encryption Features](#encryption-features)
  - [Multi-Cloud Storage](#multi-cloud-storage)
  - [Access Control](#access-control)
  - [Integrity Verification](#integrity-verification)
- [Secret Management](#secret-management)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Overview

HIPO implements a comprehensive security model to protect your ML/LLM assets, with a focus on:

- End-to-end encryption of model weights
- Multi-cloud redundancy and secure storage
- Access control and audit logging
- Secret management across environments
- Secure key rotation

## Secure Model Weights Storage

The `SecureModelWeights` module provides a secure framework for storing, managing, and accessing ML model weights across multiple cloud providers.

### Encryption Features

Model weights are encrypted before storage using strong cryptographic algorithms:

```python
from src.security.encryption import EncryptionService
from src.models.secure_weights import SecureModelWeights, create_secure_weights_manager

# Create encryption service
encryption_service = EncryptionService()

# Create secure weights manager
secure_weights = create_secure_weights_manager(
    config={'secure_weights': {'storage': {...}, 'cache': {...}}},
    encryption_service=encryption_service
)

# Store weights with encryption
metadata = secure_weights.store_weights(
    model_name='llama2-7b',
    weights_file='/path/to/weights.bin',
    encrypt=True  # Enable encryption
)
```

Encryption uses AES-256 for symmetric encryption and RSA-2048 for key protection. Keys can be stored securely in the cloud provider's key management service or in a local secure directory.

### Multi-Cloud Storage

Model weights can be stored redundantly across multiple cloud providers:

```python
# Configuration for multi-cloud storage
config = {
    'storage': {
        'primary': 's3',  # Primary storage provider
        'replicate_to': ['gcs', 'azure'],  # Secondary providers for replication
        's3_bucket': 'llm-models',
        'gcs_bucket': 'llm-models',
        'azure_container': 'llm-models',
        'versioning_enabled': True
    }
}

# Create secure weights manager with multi-cloud config
secure_weights = create_secure_weights_manager(config={'secure_weights': config})

# Store weights (will be replicated automatically)
metadata = secure_weights.store_weights(
    model_name='llama2-7b',
    weights_file='/path/to/weights.bin',
    encrypt=True
)

# Verify storage locations
for location in metadata['storage_locations']:
    print(f"Stored in {location['provider']}")
```

### Access Control

Access control can be enabled to restrict who can access and modify model weights:

```python
# Enable access control in configuration
config = {
    'storage': {
        'primary': 's3',
        'access_control_enabled': True,
    }
}

# Create secure weights manager
secure_weights = create_secure_weights_manager(config={'secure_weights': config})
```

When access control is enabled, the system will:
- Track all operations in an audit log
- Require authentication for all operations
- Enforce role-based permissions
- Validate signatures for integrity

### Integrity Verification

The system automatically calculates checksums for model weights and verifies them on access:

```python
# Verify integrity of model weights
results = secure_weights.verify_weights_integrity(
    model_name='llama2-7b',
    version='v1.0'
)

for location, verified in results.items():
    if verified:
        print(f"✅ Integrity verified for {location}")
    else:
        print(f"❌ Integrity check failed for {location}")
```

## Secret Management

The `SecretManager` provides a unified interface for managing secrets across cloud providers:

```python
from src.secrets.secret_manager import SecretManager

# Create secret manager
secret_manager = SecretManager(config, cloud_providers)

# Store a secret across all providers
secret_manager.create_secret(
    secret_name="api-keys",
    secret_data={"openai": "sk-...", "anthropic": "sk-..."}
)

# Get a secret
api_keys = secret_manager.get_secret("api-keys")

# Automatic secret rotation
secret_manager.start()  # Start rotation thread
```

## Configuration

Configure security settings in your `config.yaml`:

```yaml
security:
  encryption:
    key_directory: 'secrets'  # Directory to store encryption keys
    load_keys_from_env: false # Load keys from environment variables

  secure_weights:
    enabled: true
    storage:
      primary: 's3'
      replicate_to: ['gcs']
      s3_bucket: 'llm-models'
      versioning_enabled: true
      checksum_algorithm: 'sha256'
      access_control_enabled: true
      encryption_enabled: true

secrets:
  vault:
    enabled: false
    address: 'http://vault:8200'
    auth_method: 'kubernetes'
    
  model_weights:
    storage_type: 's3'
    s3_bucket: 'llm-models'
    sync_enabled: true
    versioning_enabled: true
    
  rotation:
    enabled: true
    schedule: '0 0 * * 0'  # Weekly on Sunday
```

## Best Practices

1. **Always Enable Encryption**: Use `encrypt=True` when storing model weights.
2. **Use Multi-Cloud Replication**: Configure replication to at least one additional provider.
3. **Regular Key Rotation**: Enable automatic key rotation in the configuration.
4. **Integrity Verification**: Periodically run `verify_weights_integrity()` to check model integrity.
5. **Secure Local Cache**: Configure cache cleanup to prevent accumulation of unneeded weights.
6. **Access Control**: Enable access control and audit logging for all operations.
7. **Backup Keys**: Ensure encryption keys are backed up securely.
8. **Environment Isolation**: Use different buckets and configurations for dev/staging/prod.