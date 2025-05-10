"""
Logging utilities for ML infrastructure.
"""
import logging
import logging.handlers
import os
from typing import Optional
from pathlib import Path


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_filename: str = "ml_infra.log",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to store log files. If None, use 'logs' in current directory.
        level: Logging level.
        log_to_console: Whether to log to console.
        log_to_file: Whether to log to file.
        log_filename: Log file name.
        max_bytes: Maximum log file size before rotating.
        backup_count: Number of backup log files to keep.

    Returns:
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler
    if log_to_file:
        if log_dir is None:
            log_dir = "logs"

        log_path = Path(log_dir)
        os.makedirs(log_path, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_filename, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class MLLogger:
    """ML-specific logger with additional features."""

    def __init__(self, name: str, log_dir: Optional[str] = None):
        """Initialize ML logger.

        Args:
            name: Logger name.
            log_dir: Directory to store log files. If None, use 'logs' in current directory.
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)

        # Create metrics logger
        self.metrics_logger = logging.getLogger(f"{name}.metrics")
        self.setup_metrics_logger()

    def setup_metrics_logger(self) -> None:
        """Set up metrics logger."""
        # Remove existing handlers
        for handler in self.metrics_logger.handlers[:]:
            self.metrics_logger.removeHandler(handler)

        # Create metrics file handler
        metrics_file = self.log_dir / f"{self.name}_metrics.log"
        metrics_handler = logging.FileHandler(metrics_file)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        metrics_handler.setFormatter(formatter)

        # Add handler
        self.metrics_logger.addHandler(metrics_handler)
        self.metrics_logger.setLevel(logging.INFO)

        # Make sure metrics logger doesn't propagate to parent
        self.metrics_logger.propagate = False

    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric.

        Args:
            metric_name: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if step is not None:
            message = f"{metric_name}={value:.6f} (step={step})"
        else:
            message = f"{metric_name}={value:.6f}"

        self.metrics_logger.info(message)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics.
            step: Optional step number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_params(self, params: dict) -> None:
        """Log parameters.

        Args:
            params: Dictionary of parameters.
        """
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.metrics_logger.info(f"PARAMETERS: {params_str}")

    def log_model(self, model_name: str, model_type: str, metrics: dict) -> None:
        """Log model information.

        Args:
            model_name: Model name.
            model_type: Model type.
            metrics: Model metrics.
        """
        self.metrics_logger.info(f"MODEL: {model_name} ({model_type})")
        self.log_metrics(metrics)
