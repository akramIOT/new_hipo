#!/usr/bin/env python
"""
Example script for managing model weights with the SecretManager.

This example demonstrates how to:
1. Upload model weights to cloud storage
2. List available model versions
3. Download model weights from cloud storage
4. Delete model weights from cloud storage
"""
import os
import logging
import argparse
import json
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path to allow importing from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.secrets.secret_manager import SecretManager
from src.cloud.factory import CloudProviderFactory


def setup_secret_manager(config: Dict[str, Any]) -> SecretManager:
    """Set up and initialize the SecretManager with cloud providers.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized SecretManager instance.
    """
    # Create cloud providers
    factory = CloudProviderFactory()
    cloud_providers = {}
    
    # Create AWS provider if configured
    if config.get("aws", {}).get("enabled", False):
        cloud_providers["aws"] = factory.create_provider("aws", config["aws"])
        
    # Create GCP provider if configured
    if config.get("gcp", {}).get("enabled", False):
        cloud_providers["gcp"] = factory.create_provider("gcp", config["gcp"])
    
    # Create secret manager
    secret_manager = SecretManager(config, cloud_providers)
    secret_manager.start()
    
    return secret_manager


def upload_model_weights(secret_manager: SecretManager, model_name: str, model_path: str) -> None:
    """Upload model weights to cloud storage.
    
    Args:
        secret_manager: Initialized SecretManager instance.
        model_name: Name of the model (e.g., "llama-7b").
        model_path: Path to the model weights directory or file.
    """
    logger.info(f"Uploading model weights for {model_name} from {model_path}")
    
    # Validate the model path
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist")
        return
    
    # Upload the model weights
    result = secret_manager.upload_model_weights(model_name, model_path)
    
    if result:
        logger.info(f"Successfully uploaded model weights for {model_name}")
    else:
        logger.error(f"Failed to upload model weights for {model_name}")


def list_models(secret_manager: SecretManager) -> None:
    """List available models and their versions.
    
    Args:
        secret_manager: Initialized SecretManager instance.
    """
    logger.info("Listing available models and versions")
    
    # Get available models
    models = secret_manager.list_available_models()
    
    if not models:
        logger.info("No models found")
        return
    
    # Print the models and versions
    logger.info("Available models:")
    for model_name, versions in models.items():
        logger.info(f"  {model_name}:")
        for version in versions:
            logger.info(f"    - {version}")


def download_model_weights(secret_manager: SecretManager, model_name: str, 
                           output_path: str, version: str = None) -> None:
    """Download model weights from cloud storage.
    
    Args:
        secret_manager: Initialized SecretManager instance.
        model_name: Name of the model (e.g., "llama-7b").
        output_path: Path to save the downloaded model weights.
        version: Specific version to download. If None, the latest version will be used.
    """
    logger.info(f"Downloading model weights for {model_name}{' (version: ' + version + ')' if version else ''}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Download the model weights
    result = secret_manager.download_model_weights(model_name, output_path, version)
    
    if result:
        logger.info(f"Successfully downloaded model weights for {model_name} to {output_path}")
    else:
        logger.error(f"Failed to download model weights for {model_name}")


def delete_model_weights(secret_manager: SecretManager, model_name: str, version: str) -> None:
    """Delete model weights from cloud storage.
    
    Args:
        secret_manager: Initialized SecretManager instance.
        model_name: Name of the model to delete.
        version: Version of the model to delete.
    """
    logger.info(f"Deleting model weights for {model_name} (version: {version})")
    
    # Delete the model weights
    result = secret_manager.delete_model_weights(model_name, version)
    
    if result:
        logger.info(f"Successfully deleted model weights for {model_name} (version: {version})")
    else:
        logger.error(f"Failed to delete model weights for {model_name} (version: {version})")


def main():
    """Main function to demonstrate SecretManager model weights management."""
    parser = argparse.ArgumentParser(description="Model weights management example.")
    parser.add_argument("--action", choices=["upload", "list", "download", "delete"], 
                        required=True, help="Action to perform")
    parser.add_argument("--model-name", help="Name of the model (e.g., 'llama-7b')")
    parser.add_argument("--model-path", help="Path to the model weights (for upload)")
    parser.add_argument("--output-path", help="Path to save downloaded model weights")
    parser.add_argument("--version", help="Specific model version")
    parser.add_argument("--config", default="config/default_config.yaml", 
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration (simplified for example)
    try:
        # This would normally load from a YAML file
        # For this example, we'll use a hardcoded configuration
        config = {
            "aws": {
                "enabled": True,
                "region": "us-west-2",
                "model_weights": {
                    "s3_bucket": "llm-models",
                }
            },
            "gcp": {
                "enabled": True,
                "project_id": "my-llm-project",
                "region": "us-central1",
                "model_weights": {
                    "gcs_bucket": "llm-models",
                }
            },
            "model_weights": {
                "storage_type": "aws",  # Primary storage provider
                "sync_enabled": True,  # Replicate to secondary providers
                "versioning_enabled": True,
                "access_control_enabled": True,
                "encryption_enabled": True,
                "checksum_algorithm": "sha256",
                "cache_enabled": True,
                "cache_directory": "weights_cache",
                "cache_max_size_gb": 10
            },
            "vault": {
                "enabled": False
            },
            "rotation": {
                "enabled": False
            }
        }
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return
    
    # Set up the secret manager
    secret_manager = setup_secret_manager(config)
    
    try:
        # Perform the requested action
        if args.action == "upload":
            if not args.model_name or not args.model_path:
                logger.error("--model-name and --model-path are required for upload action")
                return
            upload_model_weights(secret_manager, args.model_name, args.model_path)
            
        elif args.action == "list":
            list_models(secret_manager)
            
        elif args.action == "download":
            if not args.model_name or not args.output_path:
                logger.error("--model-name and --output-path are required for download action")
                return
            download_model_weights(secret_manager, args.model_name, args.output_path, args.version)
            
        elif args.action == "delete":
            if not args.model_name or not args.version:
                logger.error("--model-name and --version are required for delete action")
                return
            delete_model_weights(secret_manager, args.model_name, args.version)
    finally:
        # Stop the secret manager
        secret_manager.stop()


if __name__ == "__main__":
    main()