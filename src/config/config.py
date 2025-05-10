"""
Configuration module for ML infrastructure.
Handles all configuration settings, paths, and hyperparameters.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration class for ML infrastructure."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, use default.
        """
        self.config_dir = Path(__file__).parent.absolute()
        self.project_root = self.config_dir.parent.parent

        # Default paths
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Load config from file if provided
        self.config_data = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to configuration file.
        """
        with open(config_path, "r") as f:
            self.config_data = yaml.safe_load(f)

        # Update paths if specified in config
        if "paths" in self.config_data:
            paths = self.config_data["paths"]
            if "data_dir" in paths:
                self.data_dir = Path(paths["data_dir"])
            if "models_dir" in paths:
                self.models_dir = Path(paths["models_dir"])
            if "logs_dir" in paths:
                self.logs_dir = Path(paths["logs_dir"])

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        return self.config_data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dictionary-like access.

        Args:
            key: Configuration key.

        Returns:
            Configuration value.

        Raises:
            KeyError: If key not found.
        """
        return self.config_data[key]

    def save_config(self, config_path: str) -> None:
        """Save current configuration to a YAML file.

        Args:
            config_path: Path to save configuration file.
        """
        with open(config_path, "w") as f:
            yaml.dump(self.config_data, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Create a Config instance from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            Config instance.
        """
        config = cls()
        config.load_config(yaml_path)
        return config
