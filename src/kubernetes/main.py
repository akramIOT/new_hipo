"""
Main application for multi-cloud Kubernetes infrastructure.
"""
import argparse
import logging
import sys
import yaml
import os
from pathlib import Path

from src.kubernetes.orchestrator import KubernetesOrchestrator


def setup_logger():
    """Set up logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point."""
    # Set up logger
    setup_logger()
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-cloud Kubernetes infrastructure for LLM models")
    parser.add_argument(
        "--config", type=str, default="config/kubernetes_config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--action", type=str, choices=["start", "stop", "deploy", "status"], default="status", help="Action to perform"
    )
    parser.add_argument("--model", type=str, help="Model to deploy (required for deploy action)")
    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file {config_path} not found")
        return 1

    # Create and initialize orchestrator
    try:
        orchestrator = KubernetesOrchestrator(str(config_path))
        orchestrator.initialize()
    except Exception as e:
        logger.error(f"Error initializing orchestrator: {e}")
        return 1

    # Perform action
    if args.action == "start":
        if orchestrator.start():
            logger.info("Orchestrator started successfully")
        else:
            logger.error("Failed to start orchestrator")
            return 1
    elif args.action == "stop":
        if orchestrator.stop():
            logger.info("Orchestrator stopped successfully")
        else:
            logger.error("Failed to stop orchestrator")
            return 1
    elif args.action == "deploy":
        if not args.model:
            logger.error("Model is required for deploy action")
            return 1

        # Load model configuration
        model_config_path = Path(f"config/models/{args.model}.yaml")
        if not model_config_path.exists():
            logger.error(f"Model configuration file {model_config_path} not found")
            return 1

        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Start orchestrator if not running
        if not orchestrator.running:
            logger.info("Starting orchestrator before deployment")
            if not orchestrator.start():
                logger.error("Failed to start orchestrator")
                return 1

        # Deploy model
        if orchestrator.deploy_llm_model(model_config):
            logger.info(f"Model {args.model} deployed successfully")
        else:
            logger.error(f"Failed to deploy model {args.model}")
            return 1
    elif args.action == "status":
        # Get status
        status = orchestrator.get_status()

        # Print status
        logger.info("Orchestrator Status:")
        for category, category_status in status.items():
            logger.info(f"  {category}:")
            for key, value in category_status.items():
                logger.info(f"    {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
