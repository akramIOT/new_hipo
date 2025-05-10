"""
Pytest configuration file for all tests.
"""
import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

from src.security.encryption import EncryptionService
from src.models.secure_weights import SecureModelWeights


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests."""
    dir_path = tempfile.TemporaryDirectory()
    yield dir_path.name
    dir_path.cleanup()


@pytest.fixture(scope="session")
def encryption_service(temp_dir):
    """Create an encryption service for tests."""
    config = {
        'key_directory': temp_dir,
        'load_keys_from_env': False
    }
    return EncryptionService(config)


@pytest.fixture(scope="session")
def secure_weights_config(temp_dir):
    """Create a configuration for secure weights."""
    return {
        'storage': {
            'primary': 'local',  # Use local storage for tests
            'replicate_to': [],  # No replication for tests
            'local_path': os.path.join(temp_dir, 'weights'),
            'versioning_enabled': True,
            'checksum_algorithm': 'sha256',
            'access_control_enabled': True,
            'encryption_enabled': True
        },
        'cache': {
            'enabled': True,
            'directory': os.path.join(temp_dir, 'cache'),
            'max_size_gb': 1
        }
    }


@pytest.fixture(scope="session")
def secure_weights_manager(encryption_service, secure_weights_config):
    """Create a secure weights manager for tests."""
    return SecureModelWeights(
        encryption_service=encryption_service,
        config=secure_weights_config
    )


@pytest.fixture(scope="session")
def test_model_data():
    """Create test model data."""
    # Create a small test model
    model = {
        'layers': [
            {
                'type': 'dense',
                'weights': np.random.rand(10, 5).tolist(),
                'bias': np.random.rand(5).tolist()
            },
            {
                'type': 'dense',
                'weights': np.random.rand(5, 2).tolist(),
                'bias': np.random.rand(2).tolist()
            }
        ]
    }
    
    # Create metadata
    metadata = {
        'name': 'test_model',
        'created_at': '2025-05-06 12:34:56',
        'params': {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100
        },
        'metrics': {
            'accuracy': 0.95,
            'loss': 0.12
        }
    }
    
    return {
        'model': model,
        'metadata': metadata
    }


@pytest.fixture(scope="session")
def mock_cloud_provider():
    """Create a mock cloud provider."""
    class MockProvider:
        def __init__(self, name):
            self.name = name
            self.enabled = True
            self.secrets = {}
            self.resources = {}
        
        def is_enabled(self):
            return self.enabled
        
        def get_secret(self, name):
            return self.secrets.get(name)
        
        def create_secret(self, name, data):
            self.secrets[name] = data
            return True
        
        def update_secret(self, name, data):
            self.secrets[name] = data
            return True
        
        def get_secret_manager(self):
            return {'provider': self.name}
        
        def upload_file(self, local_path, remote_path):
            with open(local_path, 'rb') as f:
                self.resources[remote_path] = f.read()
            return {'path': remote_path}
        
        def download_file(self, remote_path, local_path):
            if remote_path not in self.resources:
                return False
            
            with open(local_path, 'wb') as f:
                f.write(self.resources[remote_path])
            return True
        
        def delete_file(self, remote_path):
            if remote_path in self.resources:
                del self.resources[remote_path]
                return True
            return False
    
    return MockProvider
