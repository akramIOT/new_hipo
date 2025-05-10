"""
Example script for serving a model with the ML infrastructure API.
"""
import os
import sys
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import Config
from src.utils.logging_utils import setup_logging
from src.api.app import init_app


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Serve a model')
    
    parser.add_argument('--model', type=str, required=True,
                      help='Model file name')
    parser.add_argument('--models-dir', type=str, default='models',
                      help='Directory containing models')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host for the API server')
    parser.add_argument('--port', type=int, default=5000,
                      help='Port for the API server')
    parser.add_argument('--uploads-dir', type=str, default='/tmp/ml_uploads',
                      help='Directory for file uploads')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(level='INFO')
    
    # Load configuration
    config = Config()
    
    # Set up uploads directory
    uploads_dir = os.path.abspath(args.uploads_dir)
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Set up model path
    models_dir = os.path.abspath(args.models_dir)
    model_path = os.path.join(models_dir, args.model)
    
    if not os.path.exists(model_path) and not (
        os.path.exists(model_path.replace('.pkl', '.h5')) or
        os.path.exists(model_path.replace('.pkl', '.pt'))
    ):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Initialize Flask app
    model_paths = {
        os.path.splitext(os.path.basename(args.model))[0]: model_path
    }
    
    app = init_app(
        model_paths=model_paths,
        upload_folder=uploads_dir
    )
    
    # Start server
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
