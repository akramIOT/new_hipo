"""
Data loading module for ML infrastructure.
Provides utilities for loading data from various sources.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Data loader class for handling various data sources."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize data loader.

        Args:
            data_dir: Directory containing data files. If None, use current directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            filename: CSV file name.
            **kwargs: Additional arguments to pass to pandas.read_csv.

        Returns:
            DataFrame containing the data.
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading CSV data from {file_path}")
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {e}")
            raise

    def load_json(self, filename: str, **kwargs) -> Union[pd.DataFrame, Dict]:
        """Load data from a JSON file.

        Args:
            filename: JSON file name.
            **kwargs: Additional arguments to pass to pandas.read_json.

        Returns:
            DataFrame or dict containing the data.
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading JSON data from {file_path}")
        try:
            return pd.read_json(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading JSON data from {file_path}: {e}")
            raise

    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load data from a Parquet file.

        Args:
            filename: Parquet file name.
            **kwargs: Additional arguments to pass to pandas.read_parquet.

        Returns:
            DataFrame containing the data.
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading Parquet data from {file_path}")
        try:
            return pd.read_parquet(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading Parquet data from {file_path}: {e}")
            raise

    def load_numpy(self, filename: str) -> np.ndarray:
        """Load data from a NumPy file.

        Args:
            filename: NumPy file name.

        Returns:
            NumPy array containing the data.
        """
        file_path = self.data_dir / filename
        logger.info(f"Loading NumPy data from {file_path}")
        try:
            return np.load(file_path)
        except Exception as e:
            logger.error(f"Error loading NumPy data from {file_path}: {e}")
            raise

    def load_multiple_csv(self, pattern: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load multiple CSV files matching a pattern.

        Args:
            pattern: Glob pattern to match files.
            **kwargs: Additional arguments to pass to pandas.read_csv.

        Returns:
            Dictionary mapping file names to DataFrames.
        """
        files = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern {pattern}")

        data_dict = {}
        for file_path in files:
            try:
                data_dict[file_path.name] = pd.read_csv(file_path, **kwargs)
            except Exception as e:
                logger.error(f"Error loading CSV data from {file_path}: {e}")

        return data_dict

    def save_csv(self, data: pd.DataFrame, filename: str, **kwargs) -> None:
        """Save data to a CSV file.

        Args:
            data: DataFrame to save.
            filename: CSV file name.
            **kwargs: Additional arguments to pass to DataFrame.to_csv.
        """
        file_path = self.data_dir / filename
        logger.info(f"Saving CSV data to {file_path}")
        try:
            os.makedirs(file_path.parent, exist_ok=True)
            data.to_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error saving CSV data to {file_path}: {e}")
            raise

    def save_parquet(self, data: pd.DataFrame, filename: str, **kwargs) -> None:
        """Save data to a Parquet file.

        Args:
            data: DataFrame to save.
            filename: Parquet file name.
            **kwargs: Additional arguments to pass to DataFrame.to_parquet.
        """
        file_path = self.data_dir / filename
        logger.info(f"Saving Parquet data to {file_path}")
        try:
            os.makedirs(file_path.parent, exist_ok=True)
            data.to_parquet(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error saving Parquet data to {file_path}: {e}")
            raise
