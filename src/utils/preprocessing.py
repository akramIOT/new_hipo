"""
Preprocessing utilities for ML infrastructure.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocessing utilities for ML data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.transformers = {}

    def fit_transform(
        self,
        data: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        scaling: str = "standard",
        handle_missing: bool = True,
    ) -> pd.DataFrame:
        """Fit and transform the data.

        Args:
            data: DataFrame to preprocess.
            numeric_columns: List of numeric columns. If None, infer from data.
            categorical_columns: List of categorical columns. If None, infer from data.
            scaling: Scaling method ('standard', 'minmax', or None).
            handle_missing: Whether to handle missing values.

        Returns:
            Preprocessed DataFrame.
        """
        # Make a copy of the input data
        result = data.copy()

        # Infer column types if not provided
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(
            f"Preprocessing {len(numeric_columns)} numeric columns and {len(categorical_columns)} categorical columns"
        )

        # Handle missing values
        if handle_missing:
            result = self._handle_missing_values(result, numeric_columns, categorical_columns)

        # Scale numeric features
        if scaling and numeric_columns:
            result = self._scale_numeric_features(result, numeric_columns, scaling)

        # Encode categorical features
        if categorical_columns:
            result = self._encode_categorical_features(result, categorical_columns)

        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers.

        Args:
            data: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        # Make a copy of the input data
        result = data.copy()

        # Transform using fitted transformers
        for name, transformer_info in self.transformers.items():
            transformer = transformer_info["transformer"]
            columns = transformer_info["columns"]
            transform_type = transformer_info["type"]

            if transform_type == "imputer":
                result[columns] = transformer.transform(result[columns])
            elif transform_type == "scaler":
                result[columns] = transformer.transform(result[columns])
            elif transform_type == "encoder":
                # Get encoded feature names
                encoded_features = transformer_info["encoded_features"]

                # Encode categorical features
                encoded = transformer.transform(result[columns])

                # If sparse matrix, convert to dense
                if hasattr(encoded, "toarray"):
                    encoded = encoded.toarray()

                # Create a DataFrame with encoded values
                encoded_df = pd.DataFrame(encoded, index=result.index, columns=encoded_features)

                # Drop original columns and join encoded features
                result = result.drop(columns=columns).join(encoded_df)

        return result

    def _handle_missing_values(
        self, data: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]
    ) -> pd.DataFrame:
        """Handle missing values in the data.

        Args:
            data: DataFrame to process.
            numeric_columns: List of numeric columns.
            categorical_columns: List of categorical columns.

        Returns:
            DataFrame with handled missing values.
        """
        result = data.copy()

        # Handle missing values in numeric columns
        if numeric_columns:
            numeric_imputer = SimpleImputer(strategy="mean")
            result[numeric_columns] = numeric_imputer.fit_transform(result[numeric_columns])

            self.transformers["numeric_imputer"] = {
                "transformer": numeric_imputer,
                "columns": numeric_columns,
                "type": "imputer",
            }

        # Handle missing values in categorical columns
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy="most_frequent")
            result[categorical_columns] = categorical_imputer.fit_transform(result[categorical_columns])

            self.transformers["categorical_imputer"] = {
                "transformer": categorical_imputer,
                "columns": categorical_columns,
                "type": "imputer",
            }

        return result

    def _scale_numeric_features(self, data: pd.DataFrame, numeric_columns: List[str], scaling: str) -> pd.DataFrame:
        """Scale numeric features.

        Args:
            data: DataFrame to process.
            numeric_columns: List of numeric columns.
            scaling: Scaling method ('standard' or 'minmax').

        Returns:
            DataFrame with scaled features.
        """
        result = data.copy()

        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling}")

        # Fit and transform the data
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])

        self.transformers["scaler"] = {"transformer": scaler, "columns": numeric_columns, "type": "scaler"}

        return result

    def _encode_categorical_features(self, data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            data: DataFrame to process.
            categorical_columns: List of categorical columns.

        Returns:
            DataFrame with encoded features.
        """
        result = data.copy()

        # Create and fit the encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(result[categorical_columns])

        # Get encoded feature names
        encoded_features = []
        for i, col in enumerate(categorical_columns):
            for category in encoder.categories_[i]:
                encoded_features.append(f"{col}_{category}")

        # Transform the data
        encoded = encoder.transform(result[categorical_columns])

        # Create a DataFrame with encoded values
        encoded_df = pd.DataFrame(encoded, index=result.index, columns=encoded_features)

        # Store the encoder
        self.transformers["encoder"] = {
            "transformer": encoder,
            "columns": categorical_columns,
            "type": "encoder",
            "encoded_features": encoded_features,
        }

        # Drop original columns and join encoded features
        result = result.drop(columns=categorical_columns).join(encoded_df)

        return result

    def save(self, filepath: str) -> None:
        """Save preprocessor to disk.

        Args:
            filepath: Path to save preprocessor.
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self.transformers, f)

    def load(self, filepath: str) -> None:
        """Load preprocessor from disk.

        Args:
            filepath: Path to load preprocessor from.
        """
        import pickle

        with open(filepath, "rb") as f:
            self.transformers = pickle.load(f)
