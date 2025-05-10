"""
Tests for Kubernetes Orchestrator.
"""
import unittest
from unittest.mock import MagicMock, patch
import yaml
import json
import os
import tempfile
from pathlib import Path

from src.kubernetes.orchestrator import KubernetesOrchestrator
from src.cloud.provider import CloudProvider


class MockCloudProvider(CloudProvider):
    """Mock cloud provider for testing."""
    
    def __init__(self, config):
        """Initialize mock cloud provider."""
        super().__init__(config)
        self.enabled = config.get('enabled', True)
        self.is_enabled_called = False
        self.scale_node_group_called = False
        self.get_api_gateway_called = False
        self.get_kubernetes_client_called = False
        
    def get_kubernetes_client(self):
        """Get Kubernetes client."""
        self.get_kubernetes_client_called = True
        return MagicMock()
        
    def get_kubernetes_config(self):
        """Get Kubernetes configuration."""
        return {'version': '1.24'}
        
    def create_kubernetes_cluster(self):
        """Create a Kubernetes cluster."""
        return 'cluster-1'
        
    def delete_kubernetes_cluster(self, cluster_id):
        """Delete a Kubernetes cluster."""
        return True
        
    def get_node_groups(self):
        """Get node groups."""
        return [
            {
                'name': 'gpu-nodes',
                'instance_type': 'g4dn.xlarge',
                'desiredCapacity': 2,
                'labels': {'node-type': 'gpu'}
            }
        ]
        
    def scale_node_group(self, node_group_id, desired_size):
        """Scale a node group."""
        self.scale_node_group_called = True
        return True
        
    def get_gpu_metrics(self, node_ids=None):
        """Get GPU metrics."""
        return {
            'gpu_utilization': {'node-1': 75.0, 'node-2': 85.0},
            'gpu_memory_used': {'node-1': 8.0, 'node-2': 12.0}
        }
        
    def get_api_gateway(self):
        """Get API gateway."""
        self.get_api_gateway_called = True
        return MagicMock()
        
    def create_api_gateway(self, name, description):
        """Create an API gateway."""
        return 'api-gw-1'
        
    def get_secret_manager(self):
        """Get secret manager."""
        return MagicMock()
        
    def get_secret(self, secret_name):
        """Get a secret."""
        return {'value': 'secret-value'}
        
    def create_secret(self, secret_name, secret_data):
        """Create a secret."""
        return 'secret-1'
        
    def update_secret(self, secret_name, secret_data):
        """Update a secret."""
        return True
        
    def get_cost_metrics(self, timeframe='daily'):
        """Get cost metrics."""
        return {
            'total_cost': 100.0,
            'compute_cost': 70.0,
            'storage_cost': 20.0,
            'network_cost': 10.0
        }
        
    def is_enabled(self):
        """Check if this cloud provider is enabled."""
        self.is_enabled_called = True
        return self.enabled


class TestKubernetesOrchestrator(unittest.TestCase):
    """Tests for KubernetesOrchestrator."""
    
    def setUp(self):
        """Set up tests."""
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, 'test_config.yaml')
        
        self.config = {
            'cloud_providers': {
                'aws': {
                    'enabled': True,
                    'region': 'us-west-2'
                },
                'gcp': {
                    'enabled': False,
                    'region': 'us-central1'
                }
            },
            'api_gateway': {
                'global_lb': {
                    'type': 'cloudflare',
                    'routing_policy': 'latency'
                }
            },
            'autoscaling': {
                'metrics': {
                    'collection_interval': 15
                },
                'scaling_config': {
                    'min_replicas': 1,
                    'max_replicas': 10
                }
            },
            'secrets': {
                'vault': {
                    'enabled': True,
                    'address': 'https://vault.example.com:8200'
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)
            
        # Mock cloud provider factory
        self.mock_providers = {
            'aws': MockCloudProvider(self.config['cloud_providers']['aws']),
            'gcp': MockCloudProvider(self.config['cloud_providers']['gcp'])
        }
        
    def tearDown(self):
        """Tear down tests."""
        self.temp_dir.cleanup()
        
    @patch('src.cloud.factory.CloudProviderFactory.create_providers')
    def test_initialization(self, mock_create_providers):
        """Test initialization."""
        mock_create_providers.return_value = self.mock_providers
        
        orchestrator = KubernetesOrchestrator(self.config_path)
        
        # Check if cloud providers are initialized
        self.assertEqual(orchestrator.cloud_providers, self.mock_providers)
        
        # Initialize the orchestrator
        success = orchestrator.initialize()
        self.assertTrue(success)
        
        # Check if components are initialized
        self.assertIsNotNone(orchestrator.api_gateway)
        self.assertIsNotNone(orchestrator.gpu_autoscaler)
        self.assertIsNotNone(orchestrator.secret_manager)
        
        # Check status
        status = orchestrator.get_status()
        self.assertIn('cloud_providers', status)
        self.assertIn('aws', status['cloud_providers'])
        self.assertIn('gcp', status['cloud_providers'])
        
    @patch('src.cloud.factory.CloudProviderFactory.create_providers')
    def test_start_stop(self, mock_create_providers):
        """Test start and stop."""
        mock_create_providers.return_value = self.mock_providers
        
        orchestrator = KubernetesOrchestrator(self.config_path)
        orchestrator.initialize()
        
        # Mock components
        orchestrator.api_gateway = MagicMock()
        orchestrator.gpu_autoscaler = MagicMock()
        orchestrator.secret_manager = MagicMock()
        
        # Start orchestrator
        success = orchestrator.start()
        self.assertTrue(success)
        self.assertTrue(orchestrator.running)
        
        # Check if components are started
        orchestrator.api_gateway.setup_global_load_balancer.assert_called_once()
        orchestrator.api_gateway.configure_security.assert_called_once()
        orchestrator.api_gateway.deploy.assert_called_once()
        orchestrator.gpu_autoscaler.start.assert_called_once()
        orchestrator.secret_manager.start.assert_called_once()
        
        # Stop orchestrator
        success = orchestrator.stop()
        self.assertTrue(success)
        self.assertFalse(orchestrator.running)
        
        # Check if components are stopped
        orchestrator.gpu_autoscaler.stop.assert_called_once()
        orchestrator.secret_manager.stop.assert_called_once()
        
    @patch('src.cloud.factory.CloudProviderFactory.create_providers')
    def test_deploy_llm_model(self, mock_create_providers):
        """Test deploy LLM model."""
        mock_create_providers.return_value = self.mock_providers
        
        orchestrator = KubernetesOrchestrator(self.config_path)
        orchestrator.initialize()
        
        # Mock API gateway
        orchestrator.api_gateway = MagicMock()
        
        # Deploy model
        model_config = {
            'name': 'llama2-7b',
            'version': '1.0',
            'resource_requirements': {
                'cpu': 8,
                'memory': '32Gi',
                'gpu': 1
            },
            'container': {
                'image': 'llm-service/llama2:1.0',
                'port': 8000,
                'environment': {
                    'MODEL_PATH': '/models/llama2-7b'
                },
                'volume_mounts': [
                    {
                        'name': 'models-volume',
                        'mount_path': '/models'
                    }
                ]
            }
        }
        
        success = orchestrator.deploy_llm_model(model_config)
        self.assertTrue(success)
        
        # Check if API gateway route is registered
        orchestrator.api_gateway.register_route.assert_called_once_with(
            path='/api/models/llama2-7b',
            method='POST',
            service_name='llama2-7b-service',
            service_port=8000
        )
        
    @patch('src.cloud.factory.CloudProviderFactory.create_providers')
    def test_handle_cloud_failure(self, mock_create_providers):
        """Test handle cloud failure."""
        mock_create_providers.return_value = self.mock_providers
        
        orchestrator = KubernetesOrchestrator(self.config_path)
        orchestrator.initialize()
        
        # Mock API gateway
        orchestrator.api_gateway = MagicMock()
        
        # Handle failure
        success = orchestrator.handle_cloud_failure('gcp')
        self.assertTrue(success)
        
        # Check if API gateway is updated
        orchestrator.api_gateway.assert_any_call()
        
    @patch('src.cloud.factory.CloudProviderFactory.create_providers')
    def test_monitor_costs(self, mock_create_providers):
        """Test monitor costs."""
        mock_create_providers.return_value = self.mock_providers
        
        orchestrator = KubernetesOrchestrator(self.config_path)
        orchestrator.initialize()
        
        # Monitor costs
        costs = orchestrator.monitor_costs()
        
        # Check costs
        self.assertIn('aws', costs)
        self.assertEqual(costs['aws'], 100.0)
        

if __name__ == '__main__':
    unittest.main()
