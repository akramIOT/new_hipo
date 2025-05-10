"""
Tests for GCP Cloud Provider.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.cloud.gcp_provider import GCPProvider


@pytest.fixture
def mock_storage_client():
    """Mock GCP storage client for testing."""
    with patch("src.cloud.gcp_provider.storage") as mock:
        client_mock = MagicMock()
        mock.Client.return_value = client_mock
        yield mock


def test_gcp_provider_init():
    """Test GCPProvider initialization."""
    provider = GCPProvider(
        {
            "project_id": "test-project",
            "region": "us-central1",
            "credentials": {
                "type": "service_account",
                "project_id": "test-project",
            },
        }
    )
    
    assert provider.project_id == "test-project"
    assert provider.name == "GCPProvider"


def test_get_kubernetes_client():
    """Test getting Kubernetes client."""
    # Arrange
    provider = GCPProvider(
        {
            "project_id": "test-project",
            "region": "us-central1",
            "credentials": {
                "type": "service_account",
                "project_id": "test-project",
            },
        }
    )
    
    # Act
    result = provider.get_kubernetes_client()
    
    # Assert
    assert result is None


def test_get_kubernetes_config():
    """Test getting Kubernetes config."""
    # Arrange
    provider = GCPProvider(
        {
            "project_id": "test-project",
            "region": "us-central1",
            "gke": {"cluster_name": "test-cluster", "version": "1.24"},
            "credentials": {
                "type": "service_account",
                "project_id": "test-project",
            },
        }
    )
    
    # Act
    result = provider.get_kubernetes_config()
    
    # Assert
    assert result == {
        "name": "test-cluster",
        "location": "us-central1",
        "status": "RUNNING",
        "version": "1.24"
    }
