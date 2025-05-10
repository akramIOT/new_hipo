"""
Unit tests for the secure model weights functionality.
"""
import os
import tempfile
import unittest
from unittest import mock
import json
import hashlib
import time
from pathlib import Path

import pytest

from src.models.secure_weights import SecureModelWeights, create_secure_weights_manager
from src.security.encryption import EncryptionService


class TestSecureModelWeights(unittest.TestCase):
    """Test case for SecureModelWeights."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Set up an encryption service for testing
        self.encryption_config = {
            'key_directory': self.temp_dir.name,
            'load_keys_from_env': False
        }
        self.encryption_service = EncryptionService(self.encryption_config)
        
        # Set up a configuration for secure weights
        self.config = {
            'storage': {
                'primary': 'local',  # Use local storage for tests
                'replicate_to': [],  # No replication for tests
                'local_path': os.path.join(self.temp_dir.name, 'weights'),
                'versioning_enabled': True,
                'checksum_algorithm': 'sha256',
                'access_control_enabled': True,
                'encryption_enabled': True
            },
            'cache': {
                'enabled': True,
                'directory': os.path.join(self.temp_dir.name, 'cache'),
                'max_size_gb': 1
            }
        }
        
        # Create the secure weights manager
        self.secure_weights = SecureModelWeights(
            encryption_service=self.encryption_service,
            config=self.config
        )
        
        # Create some test data
        self.test_model_name = 'test_model'
        self.test_version = f'v_{int(time.time())}'
        self.test_weights_data = b'dummy model weights data for testing'
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_create_secure_weights_manager(self):
        """Test creation of secure weights manager."""
        manager = create_secure_weights_manager(
            config={'secure_weights': self.config},
            encryption_service=self.encryption_service
        )
        
        self.assertIsInstance(manager, SecureModelWeights)
        self.assertEqual(manager.primary_storage, 'local')
        self.assertEqual(manager.versioning_enabled, True)
        self.assertEqual(manager.checksum_algorithm, 'sha256')
        
    def test_calculate_checksum(self):
        """Test checksum calculation."""
        expected_checksum = hashlib.sha256(self.test_weights_data).hexdigest()
        calculated_checksum = self.secure_weights._calculate_checksum(self.test_weights_data)
        
        self.assertEqual(calculated_checksum, expected_checksum)
        
    def test_store_and_load_weights(self):
        """Test storing and loading weights."""
        # Create a temporary file with test weights
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(self.test_weights_data)
            temp_path = temp_file.name
            
        try:
            # Store the weights
            metadata = self.secure_weights.store_weights(
                model_name=self.test_model_name,
                weights_file=temp_path,
                version=self.test_version,
                metadata={'test_key': 'test_value'},
                encrypt=True
            )
            
            # Check metadata
            self.assertEqual(metadata['model_name'], self.test_model_name)
            self.assertEqual(metadata['version'], self.test_version)
            self.assertEqual(metadata['checksum_algorithm'], 'sha256')
            self.assertEqual(metadata['encrypted'], True)
            self.assertIn('storage_locations', metadata)
            self.assertGreater(len(metadata['storage_locations']), 0)
            
            # Load the weights
            loaded_data, loaded_metadata = self.secure_weights.load_weights(
                model_name=self.test_model_name,
                version=self.test_version,
                decrypt=True
            )
            
            # Check loaded data
            self.assertEqual(loaded_data, self.test_weights_data)
            self.assertEqual(loaded_metadata['version'], self.test_version)
            self.assertEqual(loaded_metadata['test_key'], 'test_value')
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_store_weights_with_file_object(self):
        """Test storing weights with a file-like object."""
        # Create a file-like object with test weights
        from io import BytesIO
        file_obj = BytesIO(self.test_weights_data)
        
        # Store the weights
        metadata = self.secure_weights.store_weights(
            model_name=self.test_model_name,
            weights_file=file_obj,
            version=self.test_version,
            metadata={'test_key': 'test_value'},
            encrypt=True
        )
        
        # Check metadata
        self.assertEqual(metadata['model_name'], self.test_model_name)
        self.assertEqual(metadata['version'], self.test_version)
        
        # Load the weights
        loaded_data, loaded_metadata = self.secure_weights.load_weights(
            model_name=self.test_model_name,
            version=self.test_version,
            decrypt=True
        )
        
        # Check loaded data
        self.assertEqual(loaded_data, self.test_weights_data)
        
    def test_list_models_and_versions(self):
        """Test listing models and versions."""
        # Create two test models with different versions
        model1_name = 'test_model_1'
        model2_name = 'test_model_2'
        version1 = 'v_1000000000'
        version2 = 'v_2000000000'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store model 1, version 1
            self.secure_weights.store_weights(
                model_name=model1_name,
                weights_file=temp_file.name,
                version=version1,
                encrypt=True
            )
            
            # Store model 1, version 2
            self.secure_weights.store_weights(
                model_name=model1_name,
                weights_file=temp_file.name,
                version=version2,
                encrypt=True
            )
            
            # Store model 2, version 1
            self.secure_weights.store_weights(
                model_name=model2_name,
                weights_file=temp_file.name,
                version=version1,
                encrypt=True
            )
        
        # List models
        models = self.secure_weights.list_models()
        self.assertIn(model1_name, models)
        self.assertIn(model2_name, models)
        
        # List versions for model 1
        versions = self.secure_weights.list_versions(model1_name)
        self.assertEqual(len(versions), 2)
        version_ids = [v['version'] for v in versions]
        self.assertIn(version1, version_ids)
        self.assertIn(version2, version_ids)
        
        # List versions for model 2
        versions = self.secure_weights.list_versions(model2_name)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]['version'], version1)
        
    def test_get_weights_info(self):
        """Test getting weights info."""
        # Create test model with metadata
        model_name = 'test_info_model'
        version = 'v_test_info'
        metadata = {
            'test_key': 'test_value',
            'params': {'param1': 1, 'param2': 2}
        }
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store the model
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version,
                metadata=metadata,
                encrypt=True
            )
        
        # Get info for all versions
        all_versions_info = self.secure_weights.get_weights_info(model_name)
        self.assertEqual(all_versions_info['model_name'], model_name)
        self.assertEqual(len(all_versions_info['versions']), 1)
        
        # Get info for specific version
        version_info = self.secure_weights.get_weights_info(model_name, version)
        self.assertEqual(version_info['model_name'], model_name)
        self.assertEqual(version_info['version'], version)
        self.assertEqual(version_info['test_key'], 'test_value')
        self.assertEqual(version_info['params'], {'param1': 1, 'param2': 2})
        
    def test_delete_weights(self):
        """Test deleting weights."""
        # Create test model
        model_name = 'test_delete_model'
        version = 'v_test_delete'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store the model
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version,
                encrypt=True
            )
        
        # Verify model exists
        models = self.secure_weights.list_models()
        self.assertIn(model_name, models)
        
        # Delete the model
        result = self.secure_weights.delete_weights(model_name, version)
        self.assertTrue(result)
        
        # Verify model is deleted
        models = self.secure_weights.list_models()
        if model_name in models:  # It might still be in the list if there are other versions
            versions = self.secure_weights.list_versions(model_name)
            version_ids = [v['version'] for v in versions]
            self.assertNotIn(version, version_ids)
            
    def test_verify_weights_integrity(self):
        """Test verifying weights integrity."""
        # Create test model
        model_name = 'test_integrity_model'
        version = 'v_test_integrity'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store the model
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version,
                encrypt=True
            )
        
        # Verify integrity
        results = self.secure_weights.verify_weights_integrity(model_name, version)
        self.assertIn(f"{model_name}/{version}/local", results)
        self.assertTrue(results[f"{model_name}/{version}/local"])
        
    @mock.patch('src.models.secure_weights.SecureModelWeights._store_to_provider')
    def test_replication(self, mock_store_to_provider):
        """Test replication to multiple providers."""
        # Configure replication
        self.secure_weights.replicate_to = ['s3', 'gcs']
        
        # Mock the store_to_provider method to return fake location info
        mock_store_to_provider.side_effect = lambda provider, *args, **kwargs: {
            'provider': provider,
            'path': f'/fake/path/{provider}/test'
        }
        
        # Create test model
        model_name = 'test_replication_model'
        version = 'v_test_replication'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store the model
            metadata = self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version,
                encrypt=True
            )
        
        # Check that store_to_provider was called for all providers
        expected_calls = [
            mock.call('local', model_name, version, mock.ANY, mock.ANY),  # Primary storage
            mock.call('s3', model_name, version, mock.ANY, mock.ANY),     # Replication
            mock.call('gcs', model_name, version, mock.ANY, mock.ANY)     # Replication
        ]
        self.assertEqual(mock_store_to_provider.call_count, 3)
        mock_store_to_provider.assert_has_calls(expected_calls, any_order=True)
        
        # Check that storage locations in metadata are correct
        storage_providers = [loc['provider'] for loc in metadata['storage_locations']]
        self.assertIn('local', storage_providers)
        self.assertIn('s3', storage_providers)
        self.assertIn('gcs', storage_providers)
        
    def test_cache_functionality(self):
        """Test cache functionality."""
        # Create test model
        model_name = 'test_cache_model'
        version = 'v_test_cache'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store the model
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version,
                encrypt=True
            )
        
        # Check if weights are in cache
        cache_path = Path(self.config['cache']['directory']) / model_name / f"{version}.weights"
        self.assertTrue(cache_path.exists())
        
        # Load weights and verify they're loaded from cache
        with mock.patch('src.models.secure_weights.SecureModelWeights._load_from_provider') as mock_load:
            loaded_data, _ = self.secure_weights.load_weights(
                model_name=model_name,
                version=version,
                decrypt=True
            )
            
            # Load from provider should not be called as data is in cache
            mock_load.assert_not_called()
            
            # Verify loaded data is correct
            self.assertEqual(loaded_data, self.test_weights_data)
            
    def test_get_latest_version(self):
        """Test getting the latest version."""
        # Create test model with multiple versions
        model_name = 'test_latest_model'
        version1 = 'v_1000000000'
        version2 = 'v_2000000000'
        
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(self.test_weights_data)
            temp_file.flush()
            
            # Store version 1 (older)
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version1,
                metadata={'timestamp': 1000000000},
                encrypt=True
            )
            
            # Store version 2 (newer)
            self.secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_file.name,
                version=version2,
                metadata={'timestamp': 2000000000},
                encrypt=True
            )
        
        # Get latest version
        latest_version = self.secure_weights._get_latest_version(model_name)
        self.assertEqual(latest_version, version2)
        
        # Load using default version (should be latest)
        loaded_data, loaded_metadata = self.secure_weights.load_weights(
            model_name=model_name,
            decrypt=True
        )
        
        self.assertEqual(loaded_metadata['version'], version2)


# Pytest style tests for edge cases
def test_nonexistent_model():
    """Test behavior with nonexistent model."""
    # Set up secure weights manager with minimal config
    temp_dir = tempfile.TemporaryDirectory()
    config = {
        'storage': {
            'primary': 'local',
            'local_path': os.path.join(temp_dir.name, 'weights'),
        },
        'cache': {
            'enabled': False
        }
    }
    
    encryption_service = EncryptionService({'key_directory': temp_dir.name})
    secure_weights = SecureModelWeights(encryption_service, config=config)
    
    # Try to load nonexistent model
    with pytest.raises(ValueError, match="No versions found for model"):
        secure_weights.load_weights("nonexistent_model")
    
    # Try to get info for nonexistent model
    info = secure_weights.get_weights_info("nonexistent_model")
    assert info['version_count'] == 0
    
    # Clean up
    temp_dir.cleanup()

def test_store_weights_without_version():
    """Test storing weights without specifying a version."""
    # Set up secure weights manager with minimal config
    temp_dir = tempfile.TemporaryDirectory()
    config = {
        'storage': {
            'primary': 'local',
            'local_path': os.path.join(temp_dir.name, 'weights'),
        },
        'cache': {
            'enabled': False
        }
    }
    
    encryption_service = EncryptionService({'key_directory': temp_dir.name})
    secure_weights = SecureModelWeights(encryption_service, config=config)
    
    # Create a temporary file with test weights
    test_data = b'test data without version'
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(test_data)
        temp_path = temp_file.name
    
    try:
        # Store without specifying version
        metadata = secure_weights.store_weights(
            model_name="auto_version_model",
            weights_file=temp_path,
            encrypt=False
        )
        
        # Check that a version was generated
        assert 'version' in metadata
        assert metadata['version'].startswith('v_')
        
        # Load using the generated version
        loaded_data, _ = secure_weights.load_weights(
            model_name="auto_version_model",
            version=metadata['version'],
            decrypt=False
        )
        
        # Check loaded data
        assert loaded_data == test_data
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        temp_dir.cleanup()
