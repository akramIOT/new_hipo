"""
Secure model weights storage and management for ML/LLM models.

This module provides functionality for:
1. Encrypting and decrypting model weights
2. Securely storing model weights across cloud providers
3. Versioning and access control for model weights
4. Integrity verification of model weights
"""
import os
import io
import time
import json
import hashlib
import logging
import tempfile
import threading
from typing import Dict, Any, Optional, List, Tuple, Union, BinaryIO
from pathlib import Path

from src.security.encryption import EncryptionService
from src.secrets.secret_manager import SecretManager

logger = logging.getLogger(__name__)


class SecureModelWeights:
    """Secure storage and management of model weights."""

    def __init__(
        self,
        encryption_service: EncryptionService,
        secret_manager: Optional[SecretManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize secure model weights manager.

        Args:
            encryption_service: Encryption service for encrypting/decrypting model weights.
            secret_manager: Secret manager for managing access credentials.
            config: Configuration for secure model weights.
        """
        self.encryption_service = encryption_service
        self.secret_manager = secret_manager
        self.config = config or {}

        # Storage configuration
        self.storage_config = self.config.get("storage", {})
        self.primary_storage = self.storage_config.get("primary", "s3")
        self.replicate_to = self.storage_config.get("replicate_to", [])
        self.versioning_enabled = self.storage_config.get("versioning_enabled", True)
        self.checksum_algorithm = self.storage_config.get("checksum_algorithm", "sha256")

        # Cache configuration
        self.cache_config = self.config.get("cache", {})
        self.cache_enabled = self.cache_config.get("enabled", True)
        self.cache_dir = Path(self.cache_config.get("directory", "weights_cache"))
        self.cache_max_size_gb = self.cache_config.get("max_size_gb", 10)

        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Access control
        self.access_control_enabled = self.config.get("access_control_enabled", True)

        # Metadata storage
        self.metadata = {}
        self.metadata_lock = threading.Lock()

        logger.info(
            f"Initialized SecureModelWeights with primary storage: {self.primary_storage}, "
            f"replication: {self.replicate_to}, versioning: {self.versioning_enabled}"
        )

    def store_weights(
        self,
        model_name: str,
        weights_file: Union[str, Path, BinaryIO],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encrypt: bool = True,
    ) -> Dict[str, Any]:
        """Store model weights securely.

        Args:
            model_name: Name of the model.
            weights_file: Path to weights file or file-like object.
            version: Version of the weights. If None, a timestamp-based version is used.
            metadata: Additional metadata for the weights.
            encrypt: Whether to encrypt the weights.

        Returns:
            Dictionary containing details of the stored weights.
        """
        # Generate version if not provided
        if version is None:
            version = f"v_{int(time.time())}"

        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        # Get file data and calculate checksum
        if isinstance(weights_file, (str, Path)):
            weights_path = Path(weights_file)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file {weights_path} not found")

            with open(weights_path, "rb") as f:
                weights_data = f.read()
        else:
            # File-like object
            weights_data = weights_file.read()
            if hasattr(weights_file, "seek"):
                weights_file.seek(0)

        # Calculate checksum
        checksum = self._calculate_checksum(weights_data)

        # Prepare metadata
        weights_metadata = {
            "model_name": model_name,
            "version": version,
            "timestamp": int(time.time()),
            "checksum": checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "size_bytes": len(weights_data),
            "encrypted": encrypt,
            "storage_locations": [],
        }

        # Add any additional metadata
        if metadata:
            weights_metadata.update(metadata)

        # Encrypt weights if required
        if encrypt:
            weights_data = self.encryption_service.encrypt_data(weights_data)
            weights_metadata["encrypted"] = True

        # Store in primary storage
        primary_location = self._store_to_provider(
            self.primary_storage, model_name, version, weights_data, weights_metadata
        )
        weights_metadata["storage_locations"].append(primary_location)

        # Replicate to other providers if configured
        for provider in self.replicate_to:
            try:
                location = self._store_to_provider(provider, model_name, version, weights_data, weights_metadata)
                weights_metadata["storage_locations"].append(location)
            except Exception as e:
                logger.error(f"Error replicating weights to {provider}: {e}")

        # Store in local cache if enabled
        if self.cache_enabled:
            self._store_in_cache(model_name, version, weights_data, weights_metadata)

        # Update metadata registry
        with self.metadata_lock:
            if model_name not in self.metadata:
                self.metadata[model_name] = {}
            self.metadata[model_name][version] = weights_metadata

        logger.info(
            f"Stored weights for {model_name} version {version}, "
            f"size: {len(weights_data)/1024/1024:.2f} MB, "
            f"checksum: {checksum[:8]}..., encrypted: {encrypt}"
        )

        return weights_metadata

    def load_weights(
        self, model_name: str, version: Optional[str] = None, decrypt: bool = True, provider: Optional[str] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Load model weights securely.

        Args:
            model_name: Name of the model.
            version: Version of the weights. If None, the latest version is used.
            decrypt: Whether to decrypt the weights.
            provider: Storage provider to load from. If None, try all available locations.

        Returns:
            Tuple of (weights_data, metadata).
        """
        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        # Get version if not specified
        if version is None:
            version = self._get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")

        # Check if we have metadata for this model and version
        weights_metadata = self._get_weights_metadata(model_name, version)
        if not weights_metadata:
            raise ValueError(f"No metadata found for model {model_name} version {version}")

        # Check if weights are in cache
        if self.cache_enabled and not provider:
            cache_data = self._load_from_cache(model_name, version)
            if cache_data is not None:
                logger.info(f"Loaded weights for {model_name} version {version} from cache")
                weights_data = cache_data

                # Verify checksum
                calculated_checksum = self._calculate_checksum(
                    weights_data
                    if not decrypt or not weights_metadata.get("encrypted", False)
                    else self.encryption_service.decrypt_data(weights_data)
                )

                if calculated_checksum != weights_metadata["checksum"]:
                    logger.warning(
                        f"Checksum mismatch for cached weights of {model_name} version {version}. "
                        f"Expected {weights_metadata['checksum']}, got {calculated_checksum}. "
                        f"Loading from storage instead."
                    )
                else:
                    # Decrypt if required
                    if decrypt and weights_metadata.get("encrypted", False):
                        weights_data = self.encryption_service.decrypt_data(weights_data)

                    return weights_data, weights_metadata

        # Load from specified provider or try all available locations
        if provider:
            providers_to_try = [provider]
        else:
            # Get providers from metadata
            providers_to_try = [location["provider"] for location in weights_metadata.get("storage_locations", [])]

        # Try to load from each provider
        last_error = None
        for provider_name in providers_to_try:
            try:
                weights_data = self._load_from_provider(provider_name, model_name, version, weights_metadata)

                # Verify checksum
                calculated_checksum = self._calculate_checksum(
                    weights_data
                    if not decrypt or not weights_metadata.get("encrypted", False)
                    else self.encryption_service.decrypt_data(weights_data)
                )

                if calculated_checksum != weights_metadata["checksum"]:
                    logger.warning(
                        f"Checksum mismatch for weights of {model_name} version {version} "
                        f"from provider {provider_name}. Expected {weights_metadata['checksum']}, "
                        f"got {calculated_checksum}. Trying another provider."
                    )
                    continue

                # Store in cache if enabled
                if self.cache_enabled:
                    self._store_in_cache(model_name, version, weights_data, weights_metadata)

                # Decrypt if required
                if decrypt and weights_metadata.get("encrypted", False):
                    weights_data = self.encryption_service.decrypt_data(weights_data)

                logger.info(f"Loaded weights for {model_name} version {version} " f"from provider {provider_name}")

                return weights_data, weights_metadata

            except Exception as e:
                logger.error(
                    f"Error loading weights for {model_name} version {version} " f"from provider {provider_name}: {e}"
                )
                last_error = e

        # If we get here, we failed to load from any provider
        raise RuntimeError(
            f"Failed to load weights for {model_name} version {version} " f"from any provider: {last_error}"
        )

    def load_weights_to_file(
        self,
        model_name: str,
        output_path: Union[str, Path],
        version: Optional[str] = None,
        decrypt: bool = True,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load model weights to a file.

        Args:
            model_name: Name of the model.
            output_path: Path to save the weights.
            version: Version of the weights. If None, the latest version is used.
            decrypt: Whether to decrypt the weights.
            provider: Storage provider to load from. If None, try all available locations.

        Returns:
            Metadata for the loaded weights.
        """
        # Load weights
        weights_data, weights_metadata = self.load_weights(model_name, version, decrypt, provider)

        # Save to file
        output_path = Path(output_path)
        with open(output_path, "wb") as f:
            f.write(weights_data)

        logger.info(
            f"Saved weights for {model_name} version {version} to {output_path}, "
            f"size: {len(weights_data)/1024/1024:.2f} MB"
        )

        return weights_metadata

    def delete_weights(self, model_name: str, version: Optional[str] = None, provider: Optional[str] = None) -> bool:
        """Delete model weights.

        Args:
            model_name: Name of the model.
            version: Version of the weights. If None, delete all versions.
            provider: Storage provider to delete from. If None, delete from all providers.

        Returns:
            True if successful, False otherwise.
        """
        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        # Get versions to delete
        versions_to_delete = []
        if version is None:
            # Delete all versions
            with self.metadata_lock:
                if model_name in self.metadata:
                    versions_to_delete = list(self.metadata[model_name].keys())
        else:
            versions_to_delete = [version]

        if not versions_to_delete:
            logger.warning(f"No versions found for model {model_name}")
            return False

        # Delete each version
        success = True
        for ver in versions_to_delete:
            # Get metadata
            weights_metadata = self._get_weights_metadata(model_name, ver)
            if not weights_metadata:
                logger.warning(f"No metadata found for model {model_name} version {ver}")
                success = False
                continue

            # Get providers to delete from
            if provider:
                providers_to_delete = [provider]
            else:
                # Get providers from metadata
                providers_to_delete = [
                    location["provider"] for location in weights_metadata.get("storage_locations", [])
                ]

            # Delete from each provider
            for provider_name in providers_to_delete:
                try:
                    self._delete_from_provider(provider_name, model_name, ver, weights_metadata)
                    logger.info(f"Deleted weights for {model_name} version {ver} " f"from provider {provider_name}")
                except Exception as e:
                    logger.error(
                        f"Error deleting weights for {model_name} version {ver} " f"from provider {provider_name}: {e}"
                    )
                    success = False

            # Delete from cache
            if self.cache_enabled:
                self._delete_from_cache(model_name, ver)

            # Update metadata
            with self.metadata_lock:
                if model_name in self.metadata and ver in self.metadata[model_name]:
                    del self.metadata[model_name][ver]

                    # Remove model if no versions left
                    if not self.metadata[model_name]:
                        del self.metadata[model_name]

        return success

    def list_models(self) -> List[str]:
        """List all models with stored weights.

        Returns:
            List of model names.
        """
        with self.metadata_lock:
            return list(self.metadata.keys())

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions for a model.

        Args:
            model_name: Name of the model.

        Returns:
            List of version metadata.
        """
        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        with self.metadata_lock:
            if model_name not in self.metadata:
                return []

            return [
                {
                    "version": version,
                    "timestamp": metadata.get("timestamp"),
                    "size_bytes": metadata.get("size_bytes"),
                    "storage_locations": [loc.get("provider") for loc in metadata.get("storage_locations", [])],
                }
                for version, metadata in self.metadata[model_name].items()
            ]

    def get_weights_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about model weights.

        Args:
            model_name: Name of the model.
            version: Version of the weights. If None, get info for all versions.

        Returns:
            Dictionary containing weights information.
        """
        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        if version is None:
            # Get info for all versions
            versions = self.list_versions(model_name)
            return {"model_name": model_name, "versions": versions, "version_count": len(versions)}
        else:
            # Get info for specific version
            weights_metadata = self._get_weights_metadata(model_name, version)
            if not weights_metadata:
                return {}

            return weights_metadata

    def verify_weights_integrity(
        self, model_name: str, version: Optional[str] = None, provider: Optional[str] = None
    ) -> Dict[str, bool]:
        """Verify the integrity of model weights.

        Args:
            model_name: Name of the model.
            version: Version of the weights. If None, verify all versions.
            provider: Storage provider to verify. If None, verify all providers.

        Returns:
            Dictionary mapping provider to verification result.
        """
        # Normalize model name
        model_name = model_name.replace(" ", "_").lower()

        # Get versions to verify
        versions_to_verify = []
        if version is None:
            # Verify all versions
            with self.metadata_lock:
                if model_name in self.metadata:
                    versions_to_verify = list(self.metadata[model_name].keys())
        else:
            versions_to_verify = [version]

        if not versions_to_verify:
            logger.warning(f"No versions found for model {model_name}")
            return {}

        # Verify each version
        results = {}
        for ver in versions_to_verify:
            # Get metadata
            weights_metadata = self._get_weights_metadata(model_name, ver)
            if not weights_metadata:
                logger.warning(f"No metadata found for model {model_name} version {ver}")
                continue

            # Get providers to verify
            if provider:
                providers_to_verify = [provider]
            else:
                # Get providers from metadata
                providers_to_verify = [
                    location["provider"] for location in weights_metadata.get("storage_locations", [])
                ]

            # Verify each provider
            for provider_name in providers_to_verify:
                try:
                    # Load weights
                    weights_data = self._load_from_provider(provider_name, model_name, ver, weights_metadata)

                    # Calculate checksum
                    if weights_metadata.get("encrypted", False):
                        # Decrypt first to get the original checksum
                        decrypted_data = self.encryption_service.decrypt_data(weights_data)
                        calculated_checksum = self._calculate_checksum(decrypted_data)
                    else:
                        calculated_checksum = self._calculate_checksum(weights_data)

                    # Verify checksum
                    expected_checksum = weights_metadata.get("checksum")
                    if calculated_checksum == expected_checksum:
                        results[f"{model_name}/{ver}/{provider_name}"] = True
                        logger.info(
                            f"Verified integrity of weights for {model_name} version {ver} "
                            f"from provider {provider_name}: OK"
                        )
                    else:
                        results[f"{model_name}/{ver}/{provider_name}"] = False
                        logger.warning(
                            f"Integrity check failed for weights of {model_name} version {ver} "
                            f"from provider {provider_name}. Expected {expected_checksum}, "
                            f"got {calculated_checksum}."
                        )
                except Exception as e:
                    results[f"{model_name}/{ver}/{provider_name}"] = False
                    logger.error(
                        f"Error verifying weights for {model_name} version {ver} " f"from provider {provider_name}: {e}"
                    )

        return results

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data.

        Args:
            data: Data to calculate checksum for.

        Returns:
            Checksum as hex string.
        """
        if self.checksum_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.checksum_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {self.checksum_algorithm}")

    def _store_to_provider(
        self, provider: str, model_name: str, version: str, weights_data: bytes, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store weights to a specific storage provider.

        Args:
            provider: Storage provider name.
            model_name: Name of the model.
            version: Version of the weights.
            weights_data: Model weights data.
            metadata: Weights metadata.

        Returns:
            Storage location metadata.
        """
        # This would be implemented to store weights in different cloud providers
        # For now, we'll implement a simplified local filesystem storage

        if provider == "local":
            storage_dir = Path(self.storage_config.get("local_path", "secure_weights"))
            os.makedirs(storage_dir, exist_ok=True)

            # Create model directory
            model_dir = storage_dir / model_name
            os.makedirs(model_dir, exist_ok=True)

            # Save weights file
            weights_path = model_dir / f"{version}.weights"
            with open(weights_path, "wb") as f:
                f.write(weights_data)

            # Save metadata
            metadata_path = model_dir / f"{version}.metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Return location info
            return {"provider": "local", "path": str(weights_path), "metadata_path": str(metadata_path)}

        elif provider == "s3":
            # This would use boto3 to store in S3
            # For simplicity, we'll just log and return a mock response
            logger.info(
                f"Would store weights for {model_name} version {version} in S3 "
                f"bucket {self.storage_config.get('s3_bucket', 'llm-models')}"
            )

            return {
                "provider": "s3",
                "bucket": self.storage_config.get("s3_bucket", "llm-models"),
                "key": f"{model_name}/{version}.weights",
            }

        elif provider == "gcs":
            # This would use Google Cloud Storage client to store in GCS
            # For simplicity, we'll just log and return a mock response
            logger.info(
                f"Would store weights for {model_name} version {version} in GCS "
                f"bucket {self.storage_config.get('gcs_bucket', 'llm-models')}"
            )

            return {
                "provider": "gcs",
                "bucket": self.storage_config.get("gcs_bucket", "llm-models"),
                "path": f"{model_name}/{version}.weights",
            }

        elif provider == "azure":
            # This would use Azure Blob Storage client
            # For simplicity, we'll just log and return a mock response
            logger.info(
                f"Would store weights for {model_name} version {version} in Azure "
                f"container {self.storage_config.get('azure_container', 'llm-models')}"
            )

            return {
                "provider": "azure",
                "container": self.storage_config.get("azure_container", "llm-models"),
                "blob": f"{model_name}/{version}.weights",
            }

        else:
            raise ValueError(f"Unsupported storage provider: {provider}")

    def _load_from_provider(self, provider: str, model_name: str, version: str, metadata: Dict[str, Any]) -> bytes:
        """Load weights from a specific storage provider.

        Args:
            provider: Storage provider name.
            model_name: Name of the model.
            version: Version of the weights.
            metadata: Weights metadata.

        Returns:
            Model weights data.
        """
        # This would be implemented to load weights from different cloud providers
        # For now, we'll implement a simplified local filesystem storage

        if provider == "local":
            # Find location info
            location = next(
                (loc for loc in metadata.get("storage_locations", []) if loc.get("provider") == "local"), None
            )

            if not location:
                raise ValueError(f"No local storage location found for model {model_name} version {version}")

            # Load weights
            weights_path = location.get("path")
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Weights file {weights_path} not found for model {model_name} version {version}"
                )

            with open(weights_path, "rb") as f:
                return f.read()

        # For other providers, we'd implement the actual loading logic
        # For now, we'll raise an error
        raise NotImplementedError(f"Loading from provider {provider} not implemented")

    def _delete_from_provider(self, provider: str, model_name: str, version: str, metadata: Dict[str, Any]) -> bool:
        """Delete weights from a specific storage provider.

        Args:
            provider: Storage provider name.
            model_name: Name of the model.
            version: Version of the weights.
            metadata: Weights metadata.

        Returns:
            True if successful, False otherwise.
        """
        # This would be implemented to delete weights from different cloud providers
        # For now, we'll implement a simplified local filesystem storage

        if provider == "local":
            # Find location info
            location = next(
                (loc for loc in metadata.get("storage_locations", []) if loc.get("provider") == "local"), None
            )

            if not location:
                raise ValueError(f"No local storage location found for model {model_name} version {version}")

            # Delete weights
            weights_path = location.get("path")
            if weights_path and os.path.exists(weights_path):
                os.remove(weights_path)

            # Delete metadata
            metadata_path = location.get("metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)

            return True

        # For other providers, we'd implement the actual deletion logic
        # For now, we'll just log and return success
        logger.info(f"Would delete weights for {model_name} version {version} from provider {provider}")

        return True

    def _store_in_cache(self, model_name: str, version: str, weights_data: bytes, metadata: Dict[str, Any]) -> None:
        """Store weights in local cache.

        Args:
            model_name: Name of the model.
            version: Version of the weights.
            weights_data: Model weights data.
            metadata: Weights metadata.
        """
        # Create model directory in cache
        model_cache_dir = self.cache_dir / model_name
        os.makedirs(model_cache_dir, exist_ok=True)

        # Save weights file
        weights_path = model_cache_dir / f"{version}.weights"
        with open(weights_path, "wb") as f:
            f.write(weights_data)

        # Save metadata
        metadata_path = model_cache_dir / f"{version}.metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Check cache size and cleanup if necessary
        self._cleanup_cache()

    def _load_from_cache(self, model_name: str, version: str) -> Optional[bytes]:
        """Load weights from local cache.

        Args:
            model_name: Name of the model.
            version: Version of the weights.

        Returns:
            Model weights data or None if not in cache.
        """
        weights_path = self.cache_dir / model_name / f"{version}.weights"
        if not weights_path.exists():
            return None

        with open(weights_path, "rb") as f:
            return f.read()

    def _delete_from_cache(self, model_name: str, version: str) -> None:
        """Delete weights from local cache.

        Args:
            model_name: Name of the model.
            version: Version of the weights.
        """
        weights_path = self.cache_dir / model_name / f"{version}.weights"
        if weights_path.exists():
            os.remove(weights_path)

        metadata_path = self.cache_dir / model_name / f"{version}.metadata.json"
        if metadata_path.exists():
            os.remove(metadata_path)

    def _cleanup_cache(self) -> None:
        """Clean up cache if it exceeds the maximum size."""
        # Get cache size
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("**/*") if f.is_file())
        cache_size_gb = cache_size / 1024 / 1024 / 1024

        # Check if cache size exceeds maximum
        if cache_size_gb <= self.cache_max_size_gb:
            return

        logger.info(
            f"Cache size {cache_size_gb:.2f} GB exceeds maximum {self.cache_max_size_gb} GB, " f"cleaning up..."
        )

        # Get all weights files with their last access time
        weight_files = []
        for weights_path in self.cache_dir.glob("**/*.weights"):
            try:
                # Use last access time as a proxy for LRU
                atime = os.path.getatime(weights_path)
                weight_files.append((weights_path, atime))
            except Exception:
                continue

        # Sort by access time (oldest first)
        weight_files.sort(key=lambda x: x[1])

        # Delete files until cache size is below maximum
        for weights_path, _ in weight_files:
            if cache_size_gb <= self.cache_max_size_gb * 0.8:  # Leave some buffer
                break

            try:
                # Get file size
                file_size = weights_path.stat().st_size

                # Delete weights file
                os.remove(weights_path)

                # Delete metadata file
                metadata_path = weights_path.with_name(weights_path.name.replace(".weights", ".metadata.json"))
                if metadata_path.exists():
                    os.remove(metadata_path)

                # Update cache size
                cache_size -= file_size
                cache_size_gb = cache_size / 1024 / 1024 / 1024

                logger.info(f"Removed {weights_path} from cache")
            except Exception as e:
                logger.error(f"Error removing {weights_path} from cache: {e}")

    def _get_weights_metadata(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get metadata for model weights.

        Args:
            model_name: Name of the model.
            version: Version of the weights.

        Returns:
            Weights metadata or None if not found.
        """
        with self.metadata_lock:
            if model_name not in self.metadata:
                return None

            return self.metadata[model_name].get(version)

    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Latest version or None if no versions found.
        """
        with self.metadata_lock:
            if model_name not in self.metadata:
                return None

            # Get all versions with timestamps
            versions = [
                (version, metadata.get("timestamp", 0)) for version, metadata in self.metadata[model_name].items()
            ]

            # Sort by timestamp (newest first)
            versions.sort(key=lambda x: x[1], reverse=True)

            # Return latest version
            return versions[0][0] if versions else None


# Factory function to create SecureModelWeights instance
def create_secure_weights_manager(
    config: Dict[str, Any],
    encryption_service: Optional[EncryptionService] = None,
    secret_manager: Optional[SecretManager] = None,
) -> SecureModelWeights:
    """Create a SecureModelWeights instance.

    Args:
        config: Configuration dictionary.
        encryption_service: Encryption service instance. If None, a new one is created.
        secret_manager: Secret manager instance.

    Returns:
        SecureModelWeights instance.
    """
    # Create encryption service if not provided
    if encryption_service is None:
        encryption_config = config.get("encryption", {})
        encryption_service = EncryptionService(encryption_config)

    # Create secure weights manager
    return SecureModelWeights(
        encryption_service=encryption_service, secret_manager=secret_manager, config=config.get("secure_weights", {})
    )
