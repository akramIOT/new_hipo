"""
Neural network model implementation for ML infrastructure.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import json

from .model_base import ModelBase

logger = logging.getLogger(__name__)


class NeuralNetworkModel(ModelBase):
    """Neural network model implementation."""

    def __init__(
        self,
        model_name: str,
        models_dir: Optional[str] = None,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
    ):
        """Initialize neural network model.

        Args:
            model_name: Name of the model.
            models_dir: Directory to save/load models. If None, use current directory.
            hidden_layers: List of hidden layer sizes. If None, use [64, 32].
            activation: Activation function ('relu', 'tanh', or 'sigmoid').
            learning_rate: Learning rate.
            batch_size: Batch size.
            epochs: Number of epochs.
        """
        super().__init__(model_name, models_dir)

        self.hidden_layers = hidden_layers or [64, 32]
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Update metadata
        self.metadata["type"] = "neural_network"
        self.metadata["params"] = {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

        # Initialize model
        self.model = None
        self.input_shape = None
        self.output_shape = None
        self.is_classifier = False

    def _build_model(self) -> None:
        """Build neural network model.

        This method is framework-agnostic and can be implemented using
        TensorFlow/Keras, PyTorch, or other frameworks.
        """
        try:
            # Try to import TensorFlow/Keras
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            # Clear previous session
            tf.keras.backend.clear_session()

            # Create model
            model = Sequential()

            # Add input layer
            model.add(Dense(self.hidden_layers[0], activation=self.activation, input_shape=(self.input_shape,)))
            model.add(Dropout(0.2))

            # Add hidden layers
            for units in self.hidden_layers[1:]:
                model.add(Dense(units, activation=self.activation))
                model.add(Dropout(0.2))

            # Add output layer
            if self.is_classifier:
                if self.output_shape == 1:
                    # Binary classification
                    model.add(Dense(1, activation="sigmoid"))
                    loss = "binary_crossentropy"
                    metrics = ["accuracy"]
                else:
                    # Multi-class classification
                    model.add(Dense(self.output_shape, activation="softmax"))
                    loss = "categorical_crossentropy"
                    metrics = ["accuracy"]
            else:
                # Regression
                model.add(Dense(self.output_shape, activation="linear"))
                loss = "mse"
                metrics = ["mae"]

            # Compile model
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=loss, metrics=metrics)

            self.model = model
            logger.info(f"Built neural network model with {len(self.hidden_layers)} hidden layers")

        except ImportError:
            try:
                # Try to use PyTorch if TensorFlow is not available
                import torch
                import torch.nn as nn
                import torch.optim as optim

                class PyTorchNN(nn.Module):
                    def __init__(self, input_shape, hidden_layers, output_shape, activation):
                        super(PyTorchNN, self).__init__()

                        # Define activation function
                        if activation == "relu":
                            act_fn = nn.ReLU()
                        elif activation == "tanh":
                            act_fn = nn.Tanh()
                        elif activation == "sigmoid":
                            act_fn = nn.Sigmoid()
                        else:
                            act_fn = nn.ReLU()

                        # Create layer list
                        layers = []

                        # Input layer
                        layers.append(nn.Linear(input_shape, hidden_layers[0]))
                        layers.append(act_fn)
                        layers.append(nn.Dropout(0.2))

                        # Hidden layers
                        for i in range(len(hidden_layers) - 1):
                            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                            layers.append(act_fn)
                            layers.append(nn.Dropout(0.2))

                        # Output layer
                        layers.append(nn.Linear(hidden_layers[-1], output_shape))

                        # Add output activation for classification
                        if output_shape > 1:
                            layers.append(nn.Softmax(dim=1))

                        self.model = nn.Sequential(*layers)

                    def forward(self, x):
                        return self.model(x)

                # Create PyTorch model
                model = PyTorchNN(
                    input_shape=self.input_shape,
                    hidden_layers=self.hidden_layers,
                    output_shape=self.output_shape,
                    activation=self.activation,
                )

                # Define optimizer
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

                # Define loss function
                if self.is_classifier:
                    if self.output_shape == 1:
                        loss_fn = nn.BCELoss()
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.MSELoss()

                self.model = {"model": model, "optimizer": optimizer, "loss_fn": loss_fn, "framework": "pytorch"}

                logger.info(f"Built PyTorch neural network model with {len(self.hidden_layers)} hidden layers")

            except ImportError:
                raise ImportError("Either TensorFlow or PyTorch is required for neural network models")

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_data: Optional[Tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X: Training features.
            y: Training labels.
            validation_data: Optional tuple of (X_val, y_val).
            **kwargs: Additional training parameters.

        Returns:
            Dictionary containing training metrics.
        """
        # Convert inputs to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Determine input and output shapes
        self.input_shape = X.shape[1]

        # Determine if classification or regression
        unique_y = np.unique(y)
        if len(unique_y) < 10 and all(isinstance(val, (int, np.integer)) for val in unique_y):
            # Classification
            self.is_classifier = True

            # One-hot encode if needed
            if len(unique_y) > 2:
                from sklearn.preprocessing import OneHotEncoder

                encoder = OneHotEncoder(sparse_output=False)
                y = encoder.fit_transform(y.reshape(-1, 1))
                self.output_shape = len(unique_y)
                self.metadata["classes"] = encoder.categories_[0].tolist()
            else:
                self.output_shape = 1
                self.metadata["classes"] = unique_y.tolist()
        else:
            # Regression
            self.is_classifier = False
            if len(y.shape) > 1:
                self.output_shape = y.shape[1]
            else:
                self.output_shape = 1

        # Update parameters from kwargs
        for param_name in ["hidden_layers", "activation", "learning_rate", "batch_size", "epochs"]:
            if param_name in kwargs:
                setattr(self, param_name, kwargs[param_name])
                self.metadata["params"][param_name] = kwargs[param_name]

        # Build model
        if self.model is None:
            self._build_model()

        # Train model
        history = {}

        try:
            # Check if model is TensorFlow
            import tensorflow as tf

            if isinstance(self.model, tf.keras.Model):
                # Prepare validation data
                val_data = None
                if validation_data:
                    X_val, y_val = validation_data
                    if isinstance(X_val, pd.DataFrame):
                        X_val = X_val.values
                    if isinstance(y_val, pd.Series):
                        y_val = y_val.values

                    if self.is_classifier and len(unique_y) > 2:
                        y_val = OneHotEncoder(sparse_output=False).fit_transform(y_val.reshape(-1, 1))

                    val_data = (X_val, y_val)

                # Train model
                history = self.model.fit(
                    X, y, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_data, verbose=1
                ).history

                # Log metrics
                for metric_name, values in history.items():
                    self.log_metric(metric_name, values[-1])
        except (ImportError, NameError, AttributeError):
            try:
                # Check if model is PyTorch
                import torch

                if isinstance(self.model, dict) and self.model.get("framework") == "pytorch":
                    # Convert to torch tensors
                    X_tensor = torch.FloatTensor(X)
                    y_tensor = torch.FloatTensor(y)

                    # Create dataset
                    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                    # Training loop
                    model = self.model["model"]
                    optimizer = self.model["optimizer"]
                    loss_fn = self.model["loss_fn"]

                    model.train()
                    for epoch in range(self.epochs):
                        epoch_loss = 0.0

                        for batch_X, batch_y in dataloader:
                            # Zero gradients
                            optimizer.zero_grad()

                            # Forward pass
                            outputs = model(batch_X)

                            # Calculate loss
                            loss = loss_fn(outputs, batch_y)

                            # Backward pass and optimize
                            loss.backward()
                            optimizer.step()

                            epoch_loss += loss.item()

                        # Log epoch metrics
                        avg_loss = epoch_loss / len(dataloader)
                        history[f"epoch_{epoch}_loss"] = avg_loss

                        if (epoch + 1) % 10 == 0:
                            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

                    # Log final metrics
                    self.log_metric("train_loss", avg_loss)

                    # Evaluate on validation data if provided
                    if validation_data:
                        X_val, y_val = validation_data
                        if isinstance(X_val, pd.DataFrame):
                            X_val = X_val.values
                        if isinstance(y_val, pd.Series):
                            y_val = y_val.values

                        X_val_tensor = torch.FloatTensor(X_val)
                        y_val_tensor = torch.FloatTensor(y_val)

                        model.eval()
                        with torch.no_grad():
                            val_outputs = model(X_val_tensor)
                            val_loss = loss_fn(val_outputs, y_val_tensor).item()

                        self.log_metric("val_loss", val_loss)
            except (ImportError, NameError, AttributeError):
                raise RuntimeError("Failed to train neural network model")

        return self.get_metrics()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the model.

        Args:
            X: Features to predict.

        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        try:
            # Try TensorFlow prediction
            import tensorflow as tf

            if isinstance(self.model, tf.keras.Model):
                return self.model.predict(X)
        except (ImportError, NameError, AttributeError):
            try:
                # Try PyTorch prediction
                import torch

                if isinstance(self.model, dict) and self.model.get("framework") == "pytorch":
                    X_tensor = torch.FloatTensor(X)

                    model = self.model["model"]
                    model.eval()

                    with torch.no_grad():
                        outputs = model(X_tensor)

                    return outputs.numpy()
            except (ImportError, NameError, AttributeError):
                raise RuntimeError("Failed to make predictions with neural network model")

    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            X: Evaluation features.
            y: Evaluation labels.

        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Convert inputs to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = {}

        if self.is_classifier:
            # Classification metrics
            if self.output_shape == 1:
                # Binary classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                # Convert probabilities to binary predictions
                y_pred_binary = (y_pred > 0.5).astype(int)

                metrics["accuracy"] = accuracy_score(y, y_pred_binary)
                metrics["precision"] = precision_score(y, y_pred_binary)
                metrics["recall"] = recall_score(y, y_pred_binary)
                metrics["f1"] = f1_score(y, y_pred_binary)
            else:
                # Multi-class classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                # Convert one-hot encoded predictions to class indices
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y, axis=1) if len(y.shape) > 1 else y

                metrics["accuracy"] = accuracy_score(y_true_classes, y_pred_classes)
                metrics["precision"] = precision_score(y_true_classes, y_pred_classes, average="weighted")
                metrics["recall"] = recall_score(y_true_classes, y_pred_classes, average="weighted")
                metrics["f1"] = f1_score(y_true_classes, y_pred_classes, average="weighted")
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            metrics["mse"] = mean_squared_error(y, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y, y_pred)
            metrics["r2"] = r2_score(y, y_pred)

        # Log metrics
        for name, value in metrics.items():
            self.log_metric(f"test_{name}", value)

        return metrics

    def save(self, filename: Optional[str] = None) -> str:
        """Save model to disk.

        Args:
            filename: Filename to save model. If None, use model_name.

        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise ValueError("No model to save")

        if filename is None:
            filename = f"{self.model_name}.pkl"

        file_path = os.path.join(self.models_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        logger.info(f"Saving model to {file_path}")

        try:
            # Try TensorFlow save
            import tensorflow as tf

            if isinstance(self.model, tf.keras.Model):
                # Save Keras model in H5 format
                h5_path = os.path.splitext(file_path)[0] + ".h5"
                self.model.save(h5_path)

                # Save metadata
                metadata_path = os.path.splitext(file_path)[0] + "_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(
                        {
                            "input_shape": self.input_shape,
                            "output_shape": self.output_shape,
                            "is_classifier": self.is_classifier,
                            "metadata": self.metadata,
                        },
                        f,
                    )

                return file_path
        except (ImportError, NameError, AttributeError):
            try:
                # Try PyTorch save
                import torch

                if isinstance(self.model, dict) and self.model.get("framework") == "pytorch":
                    # Save PyTorch model
                    pt_path = os.path.splitext(file_path)[0] + ".pt"
                    torch.save(self.model["model"].state_dict(), pt_path)

                    # Save metadata
                    metadata_path = os.path.splitext(file_path)[0] + "_metadata.json"
                    with open(metadata_path, "w") as f:
                        json.dump(
                            {
                                "input_shape": self.input_shape,
                                "output_shape": self.output_shape,
                                "is_classifier": self.is_classifier,
                                "hidden_layers": self.hidden_layers,
                                "activation": self.activation,
                                "metadata": self.metadata,
                            },
                            f,
                        )

                    return file_path
            except (ImportError, NameError, AttributeError):
                pass

        # Generic pickle save as fallback
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "input_shape": self.input_shape,
                    "output_shape": self.output_shape,
                    "is_classifier": self.is_classifier,
                    "metadata": self.metadata,
                },
                f,
            )

        return file_path

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load model from.
        """
        logger.info(f"Loading model from {filepath}")

        # Check for TensorFlow H5 file
        h5_path = os.path.splitext(filepath)[0] + ".h5"
        if os.path.exists(h5_path):
            try:
                import tensorflow as tf

                # Load Keras model
                self.model = tf.keras.models.load_model(h5_path)

                # Load metadata
                metadata_path = os.path.splitext(filepath)[0] + "_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        self.input_shape = data["input_shape"]
                        self.output_shape = data["output_shape"]
                        self.is_classifier = data["is_classifier"]
                        self.metadata = data["metadata"]

                return
            except (ImportError, NameError, AttributeError):
                pass

        # Check for PyTorch PT file
        pt_path = os.path.splitext(filepath)[0] + ".pt"
        if os.path.exists(pt_path):
            try:
                import torch
                import torch.nn as nn

                # Load metadata
                metadata_path = os.path.splitext(filepath)[0] + "_metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        data = json.load(f)
                        self.input_shape = data["input_shape"]
                        self.output_shape = data["output_shape"]
                        self.is_classifier = data["is_classifier"]
                        self.hidden_layers = data["hidden_layers"]
                        self.activation = data["activation"]
                        self.metadata = data["metadata"]

                    # Create PyTorch model with same architecture
                    class PyTorchNN(nn.Module):
                        def __init__(self, input_shape, hidden_layers, output_shape, activation):
                            super(PyTorchNN, self).__init__()

                            # Define activation function
                            if activation == "relu":
                                act_fn = nn.ReLU()
                            elif activation == "tanh":
                                act_fn = nn.Tanh()
                            elif activation == "sigmoid":
                                act_fn = nn.Sigmoid()
                            else:
                                act_fn = nn.ReLU()

                            # Create layer list
                            layers = []

                            # Input layer
                            layers.append(nn.Linear(input_shape, hidden_layers[0]))
                            layers.append(act_fn)
                            layers.append(nn.Dropout(0.2))

                            # Hidden layers
                            for i in range(len(hidden_layers) - 1):
                                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                                layers.append(act_fn)
                                layers.append(nn.Dropout(0.2))

                            # Output layer
                            layers.append(nn.Linear(hidden_layers[-1], output_shape))

                            # Add output activation for classification
                            if output_shape > 1:
                                layers.append(nn.Softmax(dim=1))

                            self.model = nn.Sequential(*layers)

                        def forward(self, x):
                            return self.model(x)

                    # Create model
                    model = PyTorchNN(
                        input_shape=self.input_shape,
                        hidden_layers=self.hidden_layers,
                        output_shape=self.output_shape,
                        activation=self.activation,
                    )

                    # Load state dict
                    model.load_state_dict(torch.load(pt_path))
                    model.eval()

                    # Define optimizer and loss function
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.metadata["params"]["learning_rate"])

                    if self.is_classifier:
                        if self.output_shape == 1:
                            loss_fn = nn.BCELoss()
                        else:
                            loss_fn = nn.CrossEntropyLoss()
                    else:
                        loss_fn = nn.MSELoss()

                    self.model = {"model": model, "optimizer": optimizer, "loss_fn": loss_fn, "framework": "pytorch"}

                    return
            except (ImportError, NameError, AttributeError):
                pass

        # Fallback to pickle load
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.input_shape = data["input_shape"]
                self.output_shape = data["output_shape"]
                self.is_classifier = data["is_classifier"]
                self.metadata = data["metadata"]
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {e}")
            raise
