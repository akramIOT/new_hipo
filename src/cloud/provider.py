"""
Cloud provider interface for multi-cloud Kubernetes infrastructure.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CloudProvider(ABC):
    """Abstract base class for cloud providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize cloud provider.

        Args:
            config: Cloud provider configuration.
        """
        self.config = config
        self.name = self.__class__.__name__
        self.region = config.get("region")
        self.enabled = config.get("enabled", True)
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    def get_kubernetes_client(self):
        """Get Kubernetes client for this cloud provider.

        Returns:
            Kubernetes client.
        """
        pass

    @abstractmethod
    def get_kubernetes_config(self) -> Dict[str, Any]:
        """Get Kubernetes configuration for this cloud provider.

        Returns:
            Kubernetes configuration.
        """
        pass

    @abstractmethod
    def create_kubernetes_cluster(self) -> str:
        """Create a Kubernetes cluster in this cloud provider.

        Returns:
            Cluster ID.
        """
        pass

    @abstractmethod
    def delete_kubernetes_cluster(self, cluster_id: str) -> bool:
        """Delete a Kubernetes cluster in this cloud provider.

        Args:
            cluster_id: Cluster ID.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_node_groups(self) -> List[Dict[str, Any]]:
        """Get node groups for this cloud provider.

        Returns:
            List of node groups.
        """
        pass

    @abstractmethod
    def scale_node_group(self, node_group_id: str, desired_size: int) -> bool:
        """Scale a node group to the desired size.

        Args:
            node_group_id: Node group ID.
            desired_size: Desired size of the node group.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_gpu_metrics(self, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get GPU metrics for the specified nodes.

        Args:
            node_ids: List of node IDs to get metrics for. If None, get metrics for all nodes.

        Returns:
            Dictionary of GPU metrics.
        """
        pass

    @abstractmethod
    def get_api_gateway(self):
        """Get API gateway for this cloud provider.

        Returns:
            API gateway client.
        """
        pass

    @abstractmethod
    def create_api_gateway(self, name: str, description: str) -> str:
        """Create an API gateway in this cloud provider.

        Args:
            name: API gateway name.
            description: API gateway description.

        Returns:
            API gateway ID.
        """
        pass

    @abstractmethod
    def get_secret_manager(self):
        """Get secret manager for this cloud provider.

        Returns:
            Secret manager client.
        """
        pass

    @abstractmethod
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Get a secret from this cloud provider.

        Args:
            secret_name: Secret name.

        Returns:
            Secret data.
        """
        pass

    @abstractmethod
    def create_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> str:
        """Create a secret in this cloud provider.

        Args:
            secret_name: Secret name.
            secret_data: Secret data.

        Returns:
            Secret ID.
        """
        pass

    @abstractmethod
    def update_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> bool:
        """Update a secret in this cloud provider.

        Args:
            secret_name: Secret name.
            secret_data: Secret data.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_cost_metrics(self, timeframe: str = "daily") -> Dict[str, float]:
        """Get cost metrics for this cloud provider.

        Args:
            timeframe: Timeframe for cost metrics. Options: hourly, daily, weekly, monthly.

        Returns:
            Dictionary of cost metrics.
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this cloud provider is enabled.

        Returns:
            True if enabled, False otherwise.
        """
        return self.enabled

    def __str__(self) -> str:
        """String representation of this cloud provider.

        Returns:
            String representation.
        """
        return f"{self.name} ({self.region})"
