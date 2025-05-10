"""
Base model module for ML infrastructure.
Provides a base class for all ML models.
"""
import os
import pickle
import logging
import tempfile
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelBase:
    """Base class for all ML models."""

    def __init__(self, model_name: str, models_dir: Optional[str] = None):
        """Initialize model.

        Args:
            model_name: Name of the model.
            models_dir: Directory to save/load models. If None, use current directory.
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir) if models_dir else Path.cwd() / "models"
        self.model = None
        self.metadata = {
            "name": model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": {},
            "metrics": {},
        }
        os.makedirs(self.models_dir, exist_ok=True)

    def train(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], **kwargs) -> Dict[str, Any]:
        """Train the model.

        Args:
            X: Training features.
            y: Training labels.
            **kwargs: Additional training parameters.

        Returns:
            Dictionary containing training metrics.
        """
        raise NotImplementedError("Subclasses must implement train()")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the model.

        Args:
            X: Features to predict.

        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            X: Evaluation features.
            y: Evaluation labels.

        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        raise NotImplementedError("Subclasses must implement evaluate()")

    def save(
        self,
        filename: Optional[str] = None,
        secure: bool = False,
        secure_weights_manager=None,
        version: Optional[str] = None,
    ) -> str:
        """Save model to disk.

        Args:
            filename: Filename to save model. If None, use model_name.
            secure: Whether to save with secure weights storage.
            secure_weights_manager: SecureModelWeights instance for secure storage.
            version: Version of the model weights. If None, a timestamp-based version is used.

        Returns:
            Path to saved model or a dictionary with secure storage details.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        # If secure storage is requested but no manager is provided
        if secure and secure_weights_manager is None:
            logger.warning(
                "Secure storage requested but no secure_weights_manager provided. Falling back to standard storage."
            )
            secure = False

        # Standard storage option
        if not secure:
            if filename is None:
                filename = f"{self.model_name}.pkl"

            file_path = self.models_dir / filename
            logger.info(f"Saving model to {file_path}")

            try:
                with open(file_path, "wb") as f:
                    pickle.dump({"model": self.model, "metadata": self.metadata}, f)
                return str(file_path)
            except Exception as e:
                logger.error(f"Error saving model to {file_path}: {e}")
                raise

        # Secure storage option
        else:
            try:
                logger.info(f"Saving model {self.model_name} with secure storage")

                # Create a temporary file for the model
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
                    temp_path = temp_file.name

                    # Save model to temporary file
                    pickle.dump({"model": self.model, "metadata": self.metadata}, temp_file)

                # Store in secure weights manager
                additional_metadata = {
                    "model_type": self.__class__.__name__,
                    "params": self.metadata.get("params", {}),
                    "metrics": self.metadata.get("metrics", {}),
                }

                result = secure_weights_manager.store_weights(
                    model_name=self.model_name,
                    weights_file=temp_path,
                    version=version,
                    metadata=additional_metadata,
                    encrypt=True,
                )

                # Delete the temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

                # Return information about the stored model
                return result

            except Exception as e:
                logger.error(f"Error saving model {self.model_name} to secure storage: {e}")
                raise

    def load(
        self,
        filename: Optional[str] = None,
        secure: bool = False,
        secure_weights_manager=None,
        version: Optional[str] = None,
    ) -> None:
        """Load model from disk.

        Args:
            filename: Filename to load model from. If None, use model_name.
            secure: Whether to load from secure weights storage.
            secure_weights_manager: SecureModelWeights instance for secure storage.
            version: Version of the model weights to load. If None, the latest version is used.
        """
        # If secure storage is requested but no manager is provided
        if secure and secure_weights_manager is None:
            logger.warning(
                "Secure storage requested but no secure_weights_manager provided. Falling back to standard storage."
            )
            secure = False

        # Standard storage option
        if not secure:
            if filename is None:
                filename = f"{self.model_name}.pkl"

            file_path = self.models_dir / filename
            logger.info(f"Loading model from {file_path}")

            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.metadata = data["metadata"]
            except Exception as e:
                logger.error(f"Error loading model from {file_path}: {e}")
                raise

        # Secure storage option
        else:
            try:
                logger.info(f"Loading model {self.model_name} from secure storage (version: {version or 'latest'})")

                # Create a temporary file to store the model
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
                    temp_path = temp_file.name

                # Load weights to temporary file
                weights_metadata = secure_weights_manager.load_weights_to_file(
                    model_name=self.model_name, output_path=temp_path, version=version, decrypt=True
                )

                # Load model from temporary file
                with open(temp_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.metadata = data["metadata"]

                # Update metadata with any additional info from secure storage
                if "params" in weights_metadata:
                    self.metadata["params"].update(weights_metadata["params"])

                if "metrics" in weights_metadata:
                    self.metadata["metrics"].update(weights_metadata["metrics"])

                # Delete the temporary file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

                logger.info(
                    f"Successfully loaded model {self.model_name} version {weights_metadata.get('version', 'unknown')}"
                )

            except Exception as e:
                logger.error(f"Error loading model {self.model_name} from secure storage: {e}")
                raise

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary containing model parameters.
        """
        return self.metadata["params"]

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters.

        Args:
            params: Dictionary containing model parameters.
        """
        self.metadata["params"].update(params)

    def log_metric(self, name: str, value: float) -> None:
        """Log a metric.

        Args:
            name: Metric name.
            value: Metric value.
        """
        self.metadata["metrics"][name] = value
        logger.info(f"Metric {name}: {value}")

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics.

        Returns:
            Dictionary containing all metrics.
        """
        return self.metadata["metrics"]
