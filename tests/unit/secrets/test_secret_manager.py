"""
Unit tests for the secret manager functionality.
"""
import os
import tempfile
import unittest
from unittest import mock
import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from src.secrets.secret_manager import SecretManager


class MockCloudProvider:
    """Mock cloud provider for testing."""

    def __init__(self, name, enabled=True):
        """Initialize mock cloud provider."""
        self.name = name
        self._enabled = enabled
        self.secrets = {}
        self.model_weights = {}

    def is_enabled(self):
        """Check if provider is enabled."""
        return self._enabled

    def get_secret_manager(self):
        """Get secret manager."""
        return {"type": "mock_secret_manager", "provider": self.name}

    def get_secret(self, secret_name):
        """Get a secret."""
        return self.secrets.get(secret_name)

    def create_secret(self, secret_name, secret_data):
        """Create a secret."""
        if secret_name in self.secrets:
            return False
        self.secrets[secret_name] = secret_data
        return True

    def update_secret(self, secret_name, secret_data):
        """Update a secret."""
        self.secrets[secret_name] = secret_data
        return True

    def sync_model_weights(self, local_path, remote_path, bucket_name=None):
        """Synchronize model weights between local storage and cloud."""
        self.model_weights[remote_path] = {
            "local_path": local_path,
            "bucket": bucket_name or f"{self.name}-bucket",
            "last_sync": "mock_time"
        }
        return True

    def download_model_weights(self, remote_path, local_path, bucket_name=None):
        """Download model weights from cloud to local storage."""
        # For testing, just record the download attempt
        if remote_path in self.model_weights:
            self.model_weights[remote_path]["downloaded_to"] = local_path
            return True
        return False

    def check_model_weights_exists(self, remote_path, bucket_name=None):
        """Check if model weights exist in cloud storage."""
        return remote_path in self.model_weights


class TestSecretManager(unittest.TestCase):
    """Test case for SecretManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock cloud providers
        self.aws_provider = MockCloudProvider("aws")
        self.gcp_provider = MockCloudProvider("gcp")
        self.azure_provider = MockCloudProvider("azure")
        self.disabled_provider = MockCloudProvider("disabled", enabled=False)
        
        # Create cloud providers dictionary
        self.cloud_providers = {
            "aws": self.aws_provider,
            "gcp": self.gcp_provider,
            "azure": self.azure_provider,
            "disabled": self.disabled_provider
        }
        
        # Create config
        self.config = {
            'vault': {
                'enabled': False,
                'address': 'http://vault:8200',
                'auth_method': 'kubernetes'
            },
            'model_weights': {
                'storage_type': 's3',
                's3_bucket': 'test-llm-models',
                'gcs_bucket': 'test-llm-models',
                'azure_container': 'test-llm-models',
                'sync_enabled': True,
                'versioning_enabled': True,
                'cache_enabled': True,
                'cache_directory': 'test_weights_cache',
                'cache_max_size_gb': 5,
                'access_control_enabled': True,
                'encryption_enabled': True,
                'checksum_algorithm': 'sha256'
            },
            'rotation': {
                'enabled': True,
                'schedule': '0 0 * * 0'  # Weekly on Sunday at midnight
            }
        }
        
        # Create secret manager
        self.secret_manager = SecretManager(self.config, self.cloud_providers)
        
        # Add some mock secrets to providers
        self.aws_provider.secrets["test-secret"] = {"key": "aws-value"}
        self.gcp_provider.secrets["test-secret"] = {"key": "gcp-value"}
        self.aws_provider.secrets["aws-only-secret"] = {"key": "aws-only-value"}
        self.gcp_provider.secrets["gcp-only-secret"] = {"key": "gcp-only-value"}
        
    def test_initialization(self):
        """Test secret manager initialization."""
        # Check that secret manager was initialized correctly
        self.assertEqual(self.secret_manager.cloud_providers, self.cloud_providers)
        self.assertEqual(self.secret_manager.config, self.config)
        self.assertEqual(self.secret_manager.vault_config, self.config['vault'])
        self.assertEqual(self.secret_manager.model_weights_config, self.config['model_weights'])
        self.assertEqual(self.secret_manager.rotation_config, self.config['rotation'])
        
    def test_get_secret(self):
        """Test getting secrets."""
        # Get secret from specific provider
        aws_secret = self.secret_manager.get_secret("test-secret", "aws")
        self.assertEqual(aws_secret, {"key": "aws-value"})
        
        gcp_secret = self.secret_manager.get_secret("test-secret", "gcp")
        self.assertEqual(gcp_secret, {"key": "gcp-value"})
        
        # Get secret from any provider (should try all providers)
        any_secret = self.secret_manager.get_secret("test-secret")
        self.assertIn(any_secret, [{"key": "aws-value"}, {"key": "gcp-value"}])
        
        # Get provider-specific secret
        aws_only = self.secret_manager.get_secret("aws-only-secret")
        self.assertEqual(aws_only, {"key": "aws-only-value"})
        
        gcp_only = self.secret_manager.get_secret("gcp-only-secret")
        self.assertEqual(gcp_only, {"key": "gcp-only-value"})
        
        # Try to get nonexistent secret
        nonexistent = self.secret_manager.get_secret("nonexistent-secret")
        self.assertEqual(nonexistent, {})
        
        # Try to get from disabled provider
        disabled_secret = self.secret_manager.get_secret("test-secret", "disabled")
        self.assertEqual(disabled_secret, {})
        
        # Try to get from nonexistent provider
        nonexistent_provider = self.secret_manager.get_secret("test-secret", "nonexistent")
        self.assertEqual(nonexistent_provider, {})
        
    def test_create_secret(self):
        """Test creating secrets."""
        # Create secret in specific provider
        result = self.secret_manager.create_secret("new-secret", {"key": "new-value"}, "aws")
        self.assertTrue(result)
        self.assertEqual(self.aws_provider.secrets["new-secret"], {"key": "new-value"})
        
        # Create secret in all providers
        result = self.secret_manager.create_secret("all-providers-secret", {"key": "all-value"})
        self.assertTrue(result)
        self.assertEqual(self.aws_provider.secrets["all-providers-secret"], {"key": "all-value"})
        self.assertEqual(self.gcp_provider.secrets["all-providers-secret"], {"key": "all-value"})
        self.assertEqual(self.azure_provider.secrets["all-providers-secret"], {"key": "all-value"})
        self.assertNotIn("all-providers-secret", self.disabled_provider.secrets)
        
        # Try to create in disabled provider
        result = self.secret_manager.create_secret("disabled-secret", {"key": "value"}, "disabled")
        self.assertFalse(result)
        
        # Try to create in nonexistent provider
        result = self.secret_manager.create_secret("nonexistent-secret", {"key": "value"}, "nonexistent")
        self.assertFalse(result)
        
    def test_update_secret(self):
        """Test updating secrets."""
        # Update secret in specific provider
        result = self.secret_manager.update_secret("test-secret", {"key": "updated-value"}, "aws")
        self.assertTrue(result)
        self.assertEqual(self.aws_provider.secrets["test-secret"], {"key": "updated-value"})
        
        # Update secret in all providers
        result = self.secret_manager.update_secret("test-secret", {"key": "all-updated-value"})
        self.assertTrue(result)
        self.assertEqual(self.aws_provider.secrets["test-secret"], {"key": "all-updated-value"})
        self.assertEqual(self.gcp_provider.secrets["test-secret"], {"key": "all-updated-value"})
        
        # Try to update in disabled provider
        result = self.secret_manager.update_secret("test-secret", {"key": "value"}, "disabled")
        self.assertFalse(result)
        
        # Try to update in nonexistent provider
        result = self.secret_manager.update_secret("test-secret", {"key": "value"}, "nonexistent")
        self.assertFalse(result)
        
    def test_manage_model_weights(self):
        """Test model weights management."""
        # Test model weights management
        storage_config = self.secret_manager.manage_model_weights()
        
        # Check that storage configuration was created correctly
        self.assertEqual(storage_config['primary'], 's3')
        self.assertIn('replicate_to', storage_config)
        self.assertIn('gcp', storage_config['replicate_to'])
        self.assertIn('azure', storage_config['replicate_to'])
        self.assertEqual(storage_config['s3_bucket'], 'test-llm-models')
        self.assertEqual(storage_config['gcs_bucket'], 'test-llm-models')
        self.assertEqual(storage_config['versioning_enabled'], True)
        self.assertEqual(storage_config['access_control_enabled'], True)
        self.assertEqual(storage_config['encryption_enabled'], True)
        self.assertEqual(storage_config['checksum_algorithm'], 'sha256')
        
        # Cache configuration
        self.assertIn('cache', storage_config)
        self.assertEqual(storage_config['cache']['enabled'], True)
        self.assertEqual(storage_config['cache']['directory'], 'test_weights_cache')
        self.assertEqual(storage_config['cache']['max_size_gb'], 5)
        
    def test_get_environment_config(self):
        """Test getting environment configuration."""
        # Get dev environment config
        dev_config = self.secret_manager.get_environment_config('dev')
        self.assertEqual(dev_config['log_level'], 'DEBUG')
        self.assertEqual(dev_config['replicas'], 1)
        self.assertEqual(dev_config['enable_debug'], True)
        
        # Get staging environment config
        staging_config = self.secret_manager.get_environment_config('staging')
        self.assertEqual(staging_config['log_level'], 'INFO')
        self.assertEqual(staging_config['replicas'], 2)
        self.assertEqual(staging_config['enable_debug'], False)
        
        # Get prod environment config
        prod_config = self.secret_manager.get_environment_config('prod')
        self.assertEqual(prod_config['log_level'], 'WARNING')
        self.assertEqual(prod_config['replicas'], 3)
        self.assertEqual(prod_config['enable_debug'], False)
        
        # Get nonexistent environment config
        nonexistent_config = self.secret_manager.get_environment_config('nonexistent')
        self.assertEqual(nonexistent_config, {})
        
    def test_get_rotation_status(self):
        """Test getting rotation status."""
        # Get rotation status
        rotation_status = self.secret_manager.get_rotation_status()
        
        # Check rotation status
        self.assertEqual(rotation_status['enabled'], True)
        self.assertEqual(rotation_status['schedule'], '0 0 * * 0')
        self.assertEqual(rotation_status['last_rotation_time'], {})
        self.assertEqual(rotation_status['secret_versions'], {})
        
    def test_start_stop(self):
        """Test starting and stopping the secret manager."""
        # Test starting
        self.secret_manager.start()
        self.assertTrue(self.secret_manager.running)
        self.assertIsNotNone(self.secret_manager.rotation_thread)
        
        # Test stopping
        self.secret_manager.stop()
        self.assertFalse(self.secret_manager.running)
        self.assertIsNone(self.secret_manager.rotation_thread)
        

@pytest.fixture
def mock_providers():
    """Fixture for mock cloud providers."""
    aws = MockCloudProvider("aws")
    gcp = MockCloudProvider("gcp")
    return {"aws": aws, "gcp": gcp}


def test_rotation_loop(mock_providers):
    """Test the secret rotation loop."""
    # Create config with rotation enabled
    config = {
        'rotation': {
            'enabled': True,
            'schedule': '0 0 * * 0'  # Weekly on Sunday at midnight
        }
    }
    
    # Create secret manager
    secret_manager = SecretManager(config, mock_providers)
    
    # Mock _rotate_secrets method
    with mock.patch.object(secret_manager, '_rotate_secrets') as mock_rotate:
        # Start secret manager
        secret_manager.start()
        
        try:
            # Directly call _rotation_loop to test it
            # This is normally run in a separate thread
            secret_manager._rotation_loop()
            
            # Check that _rotate_secrets was called for each provider
            assert mock_rotate.call_count == 2
            mock_rotate.assert_any_call("aws", mock_providers["aws"])
            mock_rotate.assert_any_call("gcp", mock_providers["gcp"])
            
            # Check that last_rotation_time was updated
            assert "aws" in secret_manager.last_rotation_time
            assert "gcp" in secret_manager.last_rotation_time
            assert isinstance(secret_manager.last_rotation_time["aws"], datetime)
            assert isinstance(secret_manager.last_rotation_time["gcp"], datetime)
            
        finally:
            # Stop secret manager
            secret_manager.stop()


def test_rotate_secrets():
    """Test rotating secrets."""
    # Create mock provider with secrets
    provider = MockCloudProvider("test")
    provider.secrets = {
        "api-keys": {"key": "old-value"},
        "certificates": {"cert": "old-cert"},
        "database-credentials": {"password": "old-password"}
    }

    # Create secret manager
    secret_manager = SecretManager({}, {"test": provider})

    # Call _rotate_secrets directly
    secret_manager._rotate_secrets("test", provider)

    # Check that secrets were updated
    assert "rotated_at" in provider.secrets["api-keys"]
    assert "version" in provider.secrets["api-keys"]
    assert "rotated_at" in provider.secrets["certificates"]
    assert "version" in provider.secrets["certificates"]
    assert "rotated_at" in provider.secrets["database-credentials"]
    assert "version" in provider.secrets["database-credentials"]

    # Check that secret_versions was updated
    assert "test" in secret_manager.secret_versions
    assert "api-keys" in secret_manager.secret_versions["test"]
    assert "certificates" in secret_manager.secret_versions["test"]
    assert "database-credentials" in secret_manager.secret_versions["test"]
    assert secret_manager.secret_versions["test"]["api-keys"] == 1
    assert secret_manager.secret_versions["test"]["certificates"] == 1
    assert secret_manager.secret_versions["test"]["database-credentials"] == 1


def test_upload_model_weights():
    """Test uploading model weights."""
    # Create mock providers
    aws_provider = MockCloudProvider("aws")
    gcp_provider = MockCloudProvider("gcp")

    # Create cloud providers dictionary
    cloud_providers = {
        "aws": aws_provider,
        "gcp": gcp_provider
    }

    # Create config with AWS as primary
    config = {
        'model_weights': {
            'storage_type': 'aws',
            's3_bucket': 'test-bucket',
            'gcs_bucket': 'test-bucket',
            'sync_enabled': True
        }
    }

    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)

    # Create a temporary file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file
        test_file = os.path.join(temp_dir, "test.bin")
        with open(test_file, "wb") as f:
            f.write(b"test data")

        # Mock the _calculate_checksum method to avoid file operations
        with mock.patch.object(secret_manager, '_calculate_checksum', return_value="mock_checksum"):
            # Upload model weights
            result = secret_manager.upload_model_weights("test-model", temp_dir, "v1.0")

            # Check result
            assert result is True

            # Check that upload was called for primary provider
            expected_path = "test-model/v1.0"
            assert expected_path in aws_provider.model_weights
            assert aws_provider.model_weights[expected_path]["local_path"] == temp_dir
            assert aws_provider.model_weights[expected_path]["bucket"] == "test-bucket"

            # Check that it was replicated to secondary provider
            assert expected_path in gcp_provider.model_weights
            assert gcp_provider.model_weights[expected_path]["local_path"] == temp_dir
            assert gcp_provider.model_weights[expected_path]["bucket"] == "test-bucket"

            # Check that metadata was stored as a secret
            secret_name = f"model-weights-test-model-v1.0"
            assert secret_name in aws_provider.secrets
            assert aws_provider.secrets[secret_name]["model_name"] == "test-model"
            assert aws_provider.secrets[secret_name]["version"] == "v1.0"
            assert aws_provider.secrets[secret_name]["primary_provider"] == "aws"
            assert aws_provider.secrets[secret_name]["checksum"] == "mock_checksum"


def test_download_model_weights():
    """Test downloading model weights."""
    # Create mock providers
    aws_provider = MockCloudProvider("aws")
    gcp_provider = MockCloudProvider("gcp")

    # Create cloud providers dictionary
    cloud_providers = {
        "aws": aws_provider,
        "gcp": gcp_provider
    }

    # Create config with AWS as primary
    config = {
        'model_weights': {
            'storage_type': 'aws',
            's3_bucket': 'test-bucket',
            'gcs_bucket': 'test-bucket',
            'sync_enabled': True
        }
    }

    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)

    # Set up test data
    model_path = "test-model/v1.0"
    aws_provider.model_weights[model_path] = {
        "local_path": "/tmp/model",
        "bucket": "test-bucket",
        "last_sync": "mock_time"
    }

    # Create metadata secret
    secret_name = f"model-weights-test-model-v1.0"
    aws_provider.secrets[secret_name] = {
        "model_name": "test-model",
        "version": "v1.0",
        "primary_provider": "aws",
        "path": model_path,
        "checksum": "mock_checksum"
    }

    # Mock the _calculate_checksum method to avoid file operations
    with mock.patch.object(secret_manager, '_calculate_checksum', return_value="mock_checksum"):
        # Download model weights
        result = secret_manager.download_model_weights("test-model", "/tmp/download", "v1.0")

        # Check result
        assert result is True

        # Check that download was called
        assert "downloaded_to" in aws_provider.model_weights[model_path]
        assert aws_provider.model_weights[model_path]["downloaded_to"] == "/tmp/download"


def test_download_model_weights_fallback():
    """Test downloading model weights with fallback to secondary provider."""
    # Create mock providers
    aws_provider = MockCloudProvider("aws")
    gcp_provider = MockCloudProvider("gcp")

    # Create cloud providers dictionary
    cloud_providers = {
        "aws": aws_provider,
        "gcp": gcp_provider
    }

    # Create config with AWS as primary
    config = {
        'model_weights': {
            'storage_type': 'aws',
            's3_bucket': 'test-bucket',
            'gcs_bucket': 'test-bucket',
            'sync_enabled': True
        }
    }

    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)

    # Set up test data - only in secondary provider
    model_path = "test-model/v1.0"
    gcp_provider.model_weights[model_path] = {
        "local_path": "/tmp/model",
        "bucket": "test-bucket",
        "last_sync": "mock_time"
    }

    # Create metadata secret
    secret_name = f"model-weights-test-model-v1.0"
    gcp_provider.secrets[secret_name] = {
        "model_name": "test-model",
        "version": "v1.0",
        "primary_provider": "gcp",
        "path": model_path,
        "checksum": "mock_checksum"
    }

    # Mock the _calculate_checksum method to avoid file operations
    with mock.patch.object(secret_manager, '_calculate_checksum', return_value="mock_checksum"):
        # Download model weights - should fall back to GCP
        result = secret_manager.download_model_weights("test-model", "/tmp/download", "v1.0")

        # Check result
        assert result is True

        # Check that download was called from GCP
        assert "downloaded_to" in gcp_provider.model_weights[model_path]
        assert gcp_provider.model_weights[model_path]["downloaded_to"] == "/tmp/download"


def test_list_available_models():
    """Test listing available models."""
    # Create mock providers
    aws_provider = MockCloudProvider("aws")
    gcp_provider = MockCloudProvider("gcp")

    # Create cloud providers dictionary
    cloud_providers = {
        "aws": aws_provider,
        "gcp": gcp_provider
    }

    # Create config
    config = {
        'model_weights': {
            'storage_type': 'aws',
            's3_bucket': 'test-bucket',
            'gcs_bucket': 'test-bucket',
            'sync_enabled': True
        }
    }

    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)

    # Mock the _collect_models_from_secrets method to provide test data
    with mock.patch.object(secret_manager, '_collect_models_from_secrets', side_effect=lambda models: models.update({
        "model1": ["v1.0", "v1.1"],
        "model2": ["v2.0"]
    })):
        # List available models
        models = secret_manager.list_available_models()

        # Check result
        assert "model1" in models
        assert "model2" in models
        assert "v1.0" in models["model1"]
        assert "v1.1" in models["model1"]
        assert "v2.0" in models["model2"]


def test_delete_model_weights():
    """Test deleting model weights."""
    # Since we haven't implemented the deletion in the mock cloud provider,
    # we'll just test that the method exists and returns successfully

    # Create mock providers
    aws_provider = MockCloudProvider("aws")
    gcp_provider = MockCloudProvider("gcp")

    # Create cloud providers dictionary
    cloud_providers = {
        "aws": aws_provider,
        "gcp": gcp_provider
    }

    # Create config
    config = {
        'model_weights': {
            'storage_type': 'aws',
            's3_bucket': 'test-bucket',
            'gcs_bucket': 'test-bucket',
            'sync_enabled': True
        }
    }

    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)

    # Set up test data
    model_path = "test-model/v1.0"
    aws_provider.model_weights[model_path] = {
        "local_path": "/tmp/model",
        "bucket": "test-bucket",
        "last_sync": "mock_time"
    }

    # Delete model weights
    result = secret_manager.delete_model_weights("test-model", "v1.0")

    # Check result - the mock implementation always returns True
    assert result is True
