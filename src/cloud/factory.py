"""
Cloud provider factory for multi-cloud Kubernetes infrastructure.
"""
import logging
from typing import Dict, List, Any, Optional

from src.cloud.provider import CloudProvider
from src.cloud.aws_provider import AWSProvider
from src.cloud.gcp_provider import GCPProvider

logger = logging.getLogger(__name__)


class CloudProviderFactory:
    """Factory for creating cloud providers."""

    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> Optional[CloudProvider]:
        """Create a cloud provider of the specified type.

        Args:
            provider_type: Cloud provider type.
            config: Cloud provider configuration.

        Returns:
            Cloud provider instance.
        """
        if provider_type == "aws":
            return AWSProvider(config)
        elif provider_type == "gcp":
            return GCPProvider(config)
        else:
            logger.error(f"Unsupported cloud provider type: {provider_type}")
            return None

    @staticmethod
    def create_providers(config: Dict[str, Any]) -> Dict[str, CloudProvider]:
        """Create cloud providers based on configuration.

        Args:
            config: Configuration containing cloud provider settings.

        Returns:
            Dictionary of cloud providers.
        """
        providers = {}

        cloud_providers_config = config.get("cloud_providers", {})

        for provider_type, provider_config in cloud_providers_config.items():
            if not provider_config.get("enabled", True):
                logger.info(f"Cloud provider {provider_type} is disabled, skipping")
                continue

            provider = CloudProviderFactory.create_provider(provider_type, provider_config)
            if provider:
                providers[provider_type] = provider
                logger.info(f"Created cloud provider: {provider}")

        return providers
