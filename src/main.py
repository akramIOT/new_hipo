"""
Main entry point for ML infrastructure.
"""
import os
import argparse
import logging
from pathlib import Path

from src.config.config import Config
from src.utils.logging_utils import setup_logging
from src.data.data_loader import DataLoader
from src.models.model_base import ModelBase
from src.pipelines.pipeline import Pipeline
from src.api.app import init_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ML infrastructure")

    # Mode selection
    parser.add_argument(
        "--mode", type=str, choices=["train", "predict", "serve"], default="train", help="Operation mode"
    )

    # Configuration
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")

    # Data options
    parser.add_argument("--data", type=str, default=None, help="Path to data file or directory")
    parser.add_argument("--output", type=str, default=None, help="Path to output directory")

    # Model options
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory to save/load models")

    # API options
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=5000, help="Port for API server")

    return parser.parse_args()


def run_train_mode(config, args):
    """Run in training mode.

    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    logger = logging.getLogger("train")
    logger.info("Running in training mode")

    # Set up data directory
    data_dir = args.data or config.get("paths", {}).get("data_dir", "data")
    logger.info(f"Using data directory: {data_dir}")

    # Set up output directory
    output_dir = args.output or config.get("paths", {}).get("models_dir", "models")
    logger.info(f"Using output directory: {output_dir}")

    # Load data
    loader = DataLoader(data_dir)

    # Create pipeline
    pipeline = Pipeline("training_pipeline")

    # TODO: Implement training pipeline
    logger.info("Training pipeline not implemented yet")


def run_predict_mode(config, args):
    """Run in prediction mode.

    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    logger = logging.getLogger("predict")
    logger.info("Running in prediction mode")

    # Set up data directory
    data_dir = args.data or config.get("paths", {}).get("data_dir", "data")
    logger.info(f"Using data directory: {data_dir}")

    # Set up output directory
    output_dir = args.output or config.get("paths", {}).get("output_dir", "output")
    logger.info(f"Using output directory: {output_dir}")

    # Set up model directory
    model_dir = args.model_dir or config.get("paths", {}).get("models_dir", "models")
    logger.info(f"Using model directory: {model_dir}")

    # Load model
    model_path = args.model
    if model_path is None:
        logger.error("No model specified")
        return

    logger.info(f"Loading model from: {model_path}")
    model = ModelBase("model")
    model.load(os.path.join(model_dir, model_path))

    # Load data
    loader = DataLoader(data_dir)

    # TODO: Implement prediction pipeline
    logger.info("Prediction pipeline not implemented yet")


def run_serve_mode(config, args):
    """Run in API server mode.

    Args:
        config: Configuration object.
        args: Command line arguments.
    """
    logger = logging.getLogger("serve")
    logger.info("Running in API server mode")

    # Set up model directory
    model_dir = args.model_dir or config.get("paths", {}).get("models_dir", "models")
    logger.info(f"Using model directory: {model_dir}")

    # Set up host and port
    host = args.host or config.get("api", {}).get("host", "0.0.0.0")
    port = args.port or config.get("api", {}).get("port", 5000)
    logger.info(f"Starting API server on {host}:{port}")

    # Load models
    model_paths = {}
    if args.model:
        model_name = Path(args.model).stem
        model_paths[model_name] = os.path.join(model_dir, args.model)
    else:
        # Load all models in model directory
        if os.path.exists(model_dir):
            for filename in os.listdir(model_dir):
                if filename.endswith(".pkl"):
                    model_name = Path(filename).stem
                    model_paths[model_name] = os.path.join(model_dir, filename)

    # Initialize app
    app = init_app(model_paths=model_paths)

    # Run app
    app.run(host=host, port=port)


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = args.config
    config = Config(config_path)

    # Set up logging
    log_dir = config.get("paths", {}).get("logs_dir", "logs")
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))

    setup_logging(
        log_dir=log_dir,
        level=log_level,
        log_to_console=config.get("logging", {}).get("log_to_console", True),
        log_to_file=config.get("logging", {}).get("log_to_file", True),
        log_filename=config.get("logging", {}).get("log_filename", "ml_infra.log"),
    )

    logger = logging.getLogger("main")
    logger.info(f"Starting ML infrastructure in {args.mode} mode")

    # Run in selected mode
    if args.mode == "train":
        run_train_mode(config, args)
    elif args.mode == "predict":
        run_predict_mode(config, args)
    elif args.mode == "serve":
        run_serve_mode(config, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
