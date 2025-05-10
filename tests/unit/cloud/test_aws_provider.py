"""
Tests for AWS Cloud Provider.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.cloud.aws_provider import AWSProvider


@pytest.fixture
def mock_boto3():
    """Mock boto3 for testing."""
    with patch("src.cloud.aws_provider.boto3") as mock:
        # Set up necessary mocks
        mock.client.return_value = MagicMock()
        mock.resource.return_value = MagicMock()
        yield mock


def test_aws_provider_init():
    """Test AWSProvider initialization."""
    provider = AWSProvider(
        {
            "region": "us-west-2",
            "credentials": {
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
        }
    )
    
    assert provider.region == "us-west-2"
    assert provider.name == "AWSProvider"


@patch("src.cloud.aws_provider.boto3")
def test_get_kubernetes_client(mock_boto3):
    """Test getting Kubernetes client."""
    # Arrange
    provider = AWSProvider(
        {
            "region": "us-west-2",
            "credentials": {
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
        }
    )
    
    # Act
    result = provider.get_kubernetes_client()
    
    # Assert
    assert result is None


@patch("src.cloud.aws_provider.boto3")
def test_get_kubernetes_config(mock_boto3):
    """Test getting Kubernetes config."""
    # Arrange
    provider = AWSProvider(
        {
            "region": "us-west-2",
            "eks": {"cluster_name": "test-cluster"},
            "credentials": {
                "aws_access_key_id": "test_key",
                "aws_secret_access_key": "test_secret",
            },
        }
    )
    
    # Setup mock EKS client
    mock_eks_client = MagicMock()
    mock_eks_client.describe_cluster.return_value = {
        "cluster": {"name": "test-cluster", "endpoint": "https://test-endpoint.eks.amazonaws.com"}
    }
    
    # Inject mock EKS client
    provider.eks_client = mock_eks_client
    
    # Act
    result = provider.get_kubernetes_config()
    
    # Assert
    assert result == {"name": "test-cluster", "endpoint": "https://test-endpoint.eks.amazonaws.com"}
    mock_eks_client.describe_cluster.assert_called_once_with(name="test-cluster")
