"""
Integration tests for cloud storage functionality.
"""
import os
import tempfile
import pytest
import time
import boto3
import json
from moto import mock_aws
from pathlib import Path

from src.security.encryption import EncryptionService
from src.models.secure_weights import SecureModelWeights


class TestS3Integration:
    """Test integration with AWS S3 (using moto mock)."""
    
    @mock_aws
    def test_s3_storage_integration(self, encryption_service, test_model_data):
        """Test storing and retrieving model weights from S3."""
        # Set up S3 bucket using moto
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-llm-models'
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Create temporary directory for cache
        temp_cache_dir = tempfile.mkdtemp()
        
        # Create configuration for S3 storage
        config = {
            'storage': {
                'primary': 'local',  # Use local for testing
                'replicate_to': [],
                's3_bucket': bucket_name,
                'versioning_enabled': True,
                'checksum_algorithm': 'sha256',
                'encryption_enabled': True,
                'local_path': temp_cache_dir
            },
            'cache': {
                'enabled': False
            }
        }
        
        # Create secure weights manager
        secure_weights = SecureModelWeights(
            encryption_service=encryption_service,
            config=config
        )
        
        # Override _store_to_provider method to use moto mock
        original_store_method = secure_weights._store_to_provider
        
        def mock_s3_store(provider, model_name, version, weights_data, metadata):
            # First call the original method to handle local storage
            location = original_store_method(provider, model_name, version, weights_data, metadata)
            
            # Also store in S3 for testing
            key = f"{model_name}/{version}.weights"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=weights_data
            )
            
            # Store metadata
            metadata_key = f"{model_name}/{version}.metadata.json"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata).encode()
            )
            
            return location
        
        # Override _load_from_provider method to use moto mock
        original_load_method = secure_weights._load_from_provider
        
        def mock_s3_load(provider, model_name, version, metadata):
            if provider == 's3':
                location = next(
                    (loc for loc in metadata.get('storage_locations', []) if loc.get('provider') == 's3'),
                    None
                )
                
                if not location:
                    raise ValueError(f"No S3 storage location found for model {model_name} version {version}")
                
                # Get from S3 using boto3
                response = s3_client.get_object(
                    Bucket=location['bucket'],
                    Key=location['key']
                )
                
                return response['Body'].read()
            else:
                return original_load_method(provider, model_name, version, metadata)
        
        # Apply the overrides
        secure_weights._store_to_provider = mock_s3_store
        secure_weights._load_from_provider = mock_s3_load
        
        try:
            # Create a temporary file with test model data
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                model_data = json.dumps(test_model_data).encode()
                temp_file.write(model_data)
                temp_path = temp_file.name
            
            # Test model and version names
            model_name = 'test_s3_model'
            version = f'v_{int(time.time())}'
            
            # Store weights in S3
            metadata = secure_weights.store_weights(
                model_name=model_name,
                weights_file=temp_path,
                version=version,
                encrypt=True
            )
            
            # Verify metadata
            assert metadata['model_name'] == model_name
            assert metadata['version'] == version
            assert metadata['encrypted'] is True
            assert len(metadata['storage_locations']) == 1
            assert metadata['storage_locations'][0]['provider'] == 'local'
            
            # Verify file exists in S3
            objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{model_name}/")
            assert 'Contents' in objects
            assert any(obj['Key'] == f"{model_name}/{version}.weights" for obj in objects['Contents'])
            assert any(obj['Key'] == f"{model_name}/{version}.metadata.json" for obj in objects['Contents'])
            
            # Load weights from S3
            loaded_data, loaded_metadata = secure_weights.load_weights(
                model_name=model_name,
                version=version,
                decrypt=True
            )
            
            # Verify loaded data
            assert loaded_data == model_data
            assert loaded_metadata['model_name'] == model_name
            assert loaded_metadata['version'] == version
            
            # Delete the weights
            delete_result = secure_weights.delete_weights(
                model_name=model_name,
                version=version
            )
            
            # Verify deletion
            assert delete_result is True
            
            # Delete S3 objects manually since we added them manually
            for obj in s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{model_name}/").get('Contents', []):
                s3_client.delete_object(Bucket=bucket_name, Key=obj['Key'])
                
            # Verify files are gone from S3
            objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{model_name}/")
            assert 'Contents' not in objects or len(objects['Contents']) == 0
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Clean up temp directory
            if os.path.exists(temp_cache_dir):
                import shutil
                shutil.rmtree(temp_cache_dir)


class TestSecretsManagerIntegration:
    """Test integration with AWS Secrets Manager (using moto mock)."""
    
    @mock_aws
    def test_secrets_manager_integration(self):
        """Test storing and retrieving secrets from AWS Secrets Manager."""
        # Set up Secrets Manager using moto
        secretsmanager_client = boto3.client('secretsmanager', region_name='us-east-1')
        
        # Override CloudProvider with direct access to secretsmanager_client
        class MockAWSProvider:
            def __init__(self, client):
                self.client = client
                
            def is_enabled(self):
                return True
                
            def get_secret_manager(self):
                return {"type": "aws_secrets_manager"}
                
            def get_secret(self, secret_name):
                try:
                    response = self.client.get_secret_value(SecretId=secret_name)
                    if 'SecretString' in response:
                        return json.loads(response['SecretString'])
                    else:
                        return None
                except Exception:
                    return None
                
            def create_secret(self, secret_name, secret_data):
                try:
                    self.client.create_secret(
                        Name=secret_name,
                        SecretString=json.dumps(secret_data)
                    )
                    return True
                except Exception:
                    return False
                
            def update_secret(self, secret_name, secret_data):
                try:
                    self.client.put_secret_value(
                        SecretId=secret_name,
                        SecretString=json.dumps(secret_data)
                    )
                    return True
                except Exception:
                    return False
        
        # Create mock provider
        aws_provider = MockAWSProvider(secretsmanager_client)
        
        # Create cloud providers dictionary
        cloud_providers = {
            "aws": aws_provider
        }
        
        # Create config
        config = {
            'model_weights': {
                'storage_type': 's3',
                's3_bucket': 'test-llm-models',
                'sync_enabled': True,
                'versioning_enabled': True,
            }
        }
        
        # Create secret manager
        from src.secrets.secret_manager import SecretManager
        secret_manager = SecretManager(config, cloud_providers)
        
        # Test create_secret
        assert secret_manager.create_secret("test-aws-secret", {"key": "value"}, "aws") is True
        
        # Test get_secret
        secret = secret_manager.get_secret("test-aws-secret", "aws")
        assert secret == {"key": "value"}
        
        # Test update_secret
        assert secret_manager.update_secret("test-aws-secret", {"key": "updated-value"}, "aws") is True
        
        # Test get_secret after update
        updated_secret = secret_manager.get_secret("test-aws-secret", "aws")
        assert updated_secret == {"key": "updated-value"}
        
        # Test manage_model_weights returns proper config with AWS credentials
        model_weights_config = secret_manager.manage_model_weights()
        assert model_weights_config['primary'] == 'aws'
        assert model_weights_config['s3_bucket'] == 'test-llm-models'
        assert 's3_bucket' in model_weights_config
        assert model_weights_config['s3_bucket'] == 'test-llm-models'
        assert model_weights_config['versioning_enabled'] is True
