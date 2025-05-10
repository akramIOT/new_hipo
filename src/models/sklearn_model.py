"""
Scikit-learn model implementation for ML infrastructure.
"""
import logging
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from .model_base import ModelBase

logger = logging.getLogger(__name__)


class SklearnModel(ModelBase):
    """Wrapper for scikit-learn models."""

    def __init__(self, model_name: str, estimator: BaseEstimator, models_dir: Optional[str] = None):
        """Initialize scikit-learn model.

        Args:
            model_name: Name of the model.
            estimator: Scikit-learn estimator.
            models_dir: Directory to save/load models. If None, use current directory.
        """
        super().__init__(model_name, models_dir)
        self.estimator = estimator
        self.model = estimator
        self.metadata["params"] = estimator.get_params()
        self.metadata["type"] = "sklearn"
        self.metadata["sklearn_class"] = estimator.__class__.__name__

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        eval_data: Optional[tuple] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X: Training features.
            y: Training labels.
            eval_data: Optional tuple of (X_eval, y_eval) for evaluation after training.
            **kwargs: Additional training parameters.

        Returns:
            Dictionary containing training metrics.
        """
        logger.info(f"Training {self.model_name} with {len(X)} samples")
        start_time = pd.Timestamp.now()

        # Update model parameters if provided
        if "params" in kwargs:
            self.model.set_params(**kwargs["params"])
            self.metadata["params"] = self.model.get_params()

        # Fit the model
        self.model.fit(X, y)

        # Calculate training time
        train_time = (pd.Timestamp.now() - start_time).total_seconds()
        self.log_metric("train_time_seconds", train_time)

        # Evaluate on training data
        y_pred = self.predict(X)
        train_metrics = self._calculate_metrics(y, y_pred)

        for name, value in train_metrics.items():
            self.log_metric(f"train_{name}", value)

        # Evaluate on validation data if provided
        if eval_data is not None:
            X_eval, y_eval = eval_data
            y_eval_pred = self.predict(X_eval)
            eval_metrics = self._calculate_metrics(y_eval, y_eval_pred)

            for name, value in eval_metrics.items():
                self.log_metric(f"val_{name}", value)

        # Perform cross-validation if requested
        if kwargs.get("cv", 0) > 0:
            cv = kwargs.get("cv")
            cv_scoring = kwargs.get("cv_scoring", "accuracy")
            logger.info(f"Performing {cv}-fold cross-validation with {cv_scoring} scoring")

            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=cv_scoring)
            self.log_metric(f"cv_{cv_scoring}_mean", np.mean(cv_scores))
            self.log_metric(f"cv_{cv_scoring}_std", np.std(cv_scores))

        return self.get_metrics()

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the model.

        Args:
            X: Features to predict.

        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get probability estimates.

        Args:
            X: Features to predict.

        Returns:
            Probability estimates.

        Raises:
            AttributeError: If model doesn't support predict_proba.
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba")

        return self.model.predict_proba(X)

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

        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y, y_pred)

        for name, value in metrics.items():
            self.log_metric(f"test_{name}", value)

        return metrics

    def _calculate_metrics(self, y_true: Union[np.ndarray, pd.Series], y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics based on prediction task.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary containing metrics.
        """
        metrics = {}

        # Determine if classification or regression task
        if self._is_classifier():
            # Classification metrics
            try:
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
                metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
                metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
            except Exception as e:
                logger.warning(f"Could not calculate all classification metrics: {e}")
        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        return metrics

    def _is_classifier(self) -> bool:
        """Determine if model is a classifier.

        Returns:
            True if model is a classifier, False otherwise.
        """
        return hasattr(self.model, "classes_") or "Classifier" in self.model.__class__.__name__
