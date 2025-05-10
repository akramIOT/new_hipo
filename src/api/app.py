"""
API module for ML infrastructure.
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from src.config.config import Config
from src.models.model_base import ModelBase

logger = logging.getLogger(__name__)

# Initialize app
app = Flask(__name__)

# Load configuration
config = Config()

# Dictionary to hold loaded models
models = {}


def load_model(model_path: str, model_name: Optional[str] = None) -> str:
    """Load a model from disk.

    Args:
        model_path: Path to model file.
        model_name: Name to use for the model. If None, use filename.

    Returns:
        Model name.
    """
    if model_name is None:
        model_name = Path(model_path).stem

    logger.info(f"Loading model {model_name} from {model_path}")
    model = ModelBase(model_name)
    model.load(model_path)

    models[model_name] = model
    return model_name


@app.route("/api/models", methods=["GET"])
def get_models() -> Dict[str, Any]:
    """Get list of available models.

    Returns:
        JSON response with models.
    """
    model_list = []
    for name, model in models.items():
        model_info = {"name": name, "metadata": model.metadata}
        model_list.append(model_info)

    return jsonify({"status": "success", "models": model_list})


@app.route("/api/models/<model_name>", methods=["GET"])
def get_model(model_name: str) -> Dict[str, Any]:
    """Get model information.

    Args:
        model_name: Name of the model.

    Returns:
        JSON response with model information.
    """
    if model_name not in models:
        return jsonify({"status": "error", "message": f"Model {model_name} not found"}), 404

    model = models[model_name]
    model_info = {"name": model_name, "metadata": model.metadata}

    return jsonify({"status": "success", "model": model_info})


def _parse_json_data(data) -> pd.DataFrame:
    """Parse JSON data into a DataFrame.
    
    Args:
        data: JSON data from request.
        
    Returns:
        DataFrame or error response.
    """
    if data is None:
        return None, ({"status": "error", "message": "No data provided"}, 400)
        
    # Convert to DataFrame
    try:
        if isinstance(data, list) or isinstance(data, dict):
            # List of records or dictionary of arrays
            return pd.DataFrame(data), None
        else:
            return None, ({"status": "error", "message": "Invalid data format"}, 400)
    except Exception as e:
        return None, ({"status": "error", "message": f"Error parsing data: {str(e)}"}, 400)


def _parse_file_data(file) -> pd.DataFrame:
    """Parse file data into a DataFrame.
    
    Args:
        file: File from request.
        
    Returns:
        DataFrame or error response.
    """
    if not file:
        return None, ({"status": "error", "message": "No file provided"}, 400)
        
    # Save file
    filename = secure_filename(file.filename)
    file_path = Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(file_path)
    
    # Read file
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif filename.endswith(".json"):
            df = pd.read_json(file_path)
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            os.remove(file_path)
            return None, ({"status": "error", "message": f"Unsupported file format: {filename}"}, 400)
            
        # Clean up
        os.remove(file_path)
        return df, None
    except Exception as e:
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, ({"status": "error", "message": f"Error reading file: {str(e)}"}, 400)


@app.route("/api/models/<model_name>/predict", methods=["POST"])
def predict(model_name: str) -> Dict[str, Any]:
    """Make predictions using a model.

    Args:
        model_name: Name of the model.

    Returns:
        JSON response with predictions.
    """
    if model_name not in models:
        return jsonify({"status": "error", "message": f"Model {model_name} not found"}), 404

    # Get data from request
    if request.json:
        # Parse JSON data
        X, error = _parse_json_data(request.json.get("data"))
        if error:
            return jsonify(error[0]), error[1]
    elif request.files:
        # Parse file data
        X, error = _parse_file_data(request.files.get("file"))
        if error:
            return jsonify(error[0]), error[1]
    else:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    # Make predictions
    try:
        model = models[model_name]
        predictions = model.predict(X)

        # Convert to JSON-serializable format
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        return jsonify({"status": "success", "predictions": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error making predictions: {str(e)}"}), 500


def _parse_evaluation_json_data(data, target):
    """Parse JSON evaluation data.
    
    Args:
        data: Input data in JSON format.
        target: Target data in JSON format.
        
    Returns:
        Tuple of (X DataFrame, y array) or (None, None, error_response) if error.
    """
    if data is None or target is None:
        return None, None, ({"status": "error", "message": "Both data and target must be provided"}, 400)
        
    # Convert to DataFrame
    try:
        if isinstance(data, list) or isinstance(data, dict):
            # List of records or dictionary of arrays
            X = pd.DataFrame(data)
        else:
            return None, None, ({"status": "error", "message": "Invalid data format"}, 400)
            
        # Convert target to array
        y = np.array(target)
        return X, y, None
    except Exception as e:
        return None, None, ({"status": "error", "message": f"Error parsing data: {str(e)}"}, 400)


def _parse_evaluation_file_data(data_file, target_file, upload_folder):
    """Parse file data for evaluation.
    
    Args:
        data_file: Data file from request.
        target_file: Target file from request.
        upload_folder: Folder to temporarily store uploaded files.
        
    Returns:
        Tuple of (X DataFrame, y array) or (None, None, error_response) if error.
    """
    if not data_file or not target_file:
        return None, None, ({"status": "error", "message": "Both data_file and target_file must be provided"}, 400)
        
    # Save files
    data_filename = secure_filename(data_file.filename)
    target_filename = secure_filename(target_file.filename)
    
    data_path = Path(upload_folder) / data_filename
    target_path = Path(upload_folder) / target_filename
    
    data_file.save(data_path)
    target_file.save(target_path)
    
    try:
        # Read data file
        X = _read_data_file(data_filename, data_path)
        if X is None:
            os.remove(data_path)
            os.remove(target_path)
            return None, None, ({"status": "error", "message": f"Unsupported data file format: {data_filename}"}, 400)
            
        # Read target file
        y = _read_target_file(target_filename, target_path)
        if y is None:
            os.remove(data_path)
            os.remove(target_path)
            return None, None, ({"status": "error", "message": f"Unsupported target file format: {target_filename}"}, 400)
            
        # Clean up
        os.remove(data_path)
        os.remove(target_path)
        return X, y, None
    except Exception as e:
        # Clean up
        if os.path.exists(data_path):
            os.remove(data_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        return None, None, ({"status": "error", "message": f"Error reading files: {str(e)}"}, 400)


def _read_data_file(filename, file_path):
    """Read data file into DataFrame.
    
    Args:
        filename: Name of the file.
        file_path: Path to the file.
        
    Returns:
        DataFrame or None if unsupported format.
    """
    if filename.endswith(".csv"):
        return pd.read_csv(file_path)
    elif filename.endswith(".json"):
        return pd.read_json(file_path)
    elif filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        return None


def _read_target_file(filename, file_path):
    """Read target file into array.
    
    Args:
        filename: Name of the file.
        file_path: Path to the file.
        
    Returns:
        Array or None if unsupported format.
    """
    if filename.endswith(".csv"):
        y_df = pd.read_csv(file_path)
        return y_df.values.ravel()
    elif filename.endswith(".json"):
        y_df = pd.read_json(file_path)
        return y_df.values.ravel()
    elif filename.endswith(".npy"):
        return np.load(file_path)
    else:
        return None


@app.route("/api/models/<model_name>/evaluate", methods=["POST"])
def evaluate(model_name: str) -> Dict[str, Any]:
    """Evaluate a model.

    Args:
        model_name: Name of the model.

    Returns:
        JSON response with evaluation metrics.
    """
    if model_name not in models:
        return jsonify({"status": "error", "message": f"Model {model_name} not found"}), 404

    # Get data from request
    if request.json:
        # Parse JSON data
        X, y, error = _parse_evaluation_json_data(request.json.get("data"), request.json.get("target"))
        if error:
            return jsonify(error[0]), error[1]
    elif request.files:
        # Parse file data
        X, y, error = _parse_evaluation_file_data(
            request.files.get("data_file"), 
            request.files.get("target_file"),
            app.config["UPLOAD_FOLDER"]
        )
        if error:
            return jsonify(error[0]), error[1]
    else:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    # Evaluate model
    try:
        model = models[model_name]
        metrics = model.evaluate(X, y)
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error evaluating model: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health() -> Dict[str, Any]:
    """Health check endpoint.

    Returns:
        JSON response with health status.
    """
    return jsonify({"status": "success", "message": "API is running"})


def init_app(model_paths: Optional[Dict[str, str]] = None, upload_folder: Optional[str] = None) -> Flask:
    """Initialize Flask app.

    Args:
        model_paths: Dictionary mapping model names to paths.
        upload_folder: Folder to store uploaded files.

    Returns:
        Flask app.
    """
    # Configure upload folder
    if upload_folder is None:
        upload_folder = "/tmp/ml_uploads"

    os.makedirs(upload_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_folder

    # Load models
    if model_paths:
        for name, path in model_paths.items():
            try:
                load_model(path, name)
            except Exception as e:
                logger.error(f"Error loading model {name} from {path}: {e}")

    return app


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Initialize app with models from config
    model_paths = config.get("model_paths", {})
    app = init_app(model_paths)

    # Run app
    port = config.get("api_port", 5000)
    app.run(host="0.0.0.0", port=port)
