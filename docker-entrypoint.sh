#!/bin/bash
set -e

# Initialize environment
echo "Initializing HIPO environment..."

# Check if config directory exists
if [ ! -d "/app/config" ]; then
  echo "Error: Config directory not found!"
  exit 1
fi

# Function to start the API server
start_api() {
  echo "Starting API server..."
  python -m src.api.app --config config/default_config.yaml
}

# Function to start the Streamlit UI
start_ui() {
  echo "Starting Streamlit UI..."
  python src/ui/run_ui.py
}

# Function to run a training job
run_training() {
  echo "Starting training job..."
  python -m src.main --mode train --config config/default_config.yaml --data "$2" --model "$3"
}

# Function to run a prediction job
run_prediction() {
  echo "Starting prediction job..."
  python -m src.main --mode predict --config config/default_config.yaml --data "$2" --model "$3" --output "$4"
}

# Function to deploy a model to kubernetes
deploy_model() {
  echo "Deploying model to kubernetes..."
  python -m src.kubernetes.main --config config/default_config.yaml --model "$2"
}

# Determine what to execute based on the first argument
case "$1" in
  serve)
    start_api
    ;;
  ui)
    start_ui
    ;;
  train)
    if [ "$#" -lt 3 ]; then
      echo "Usage: $0 train <data_path> <model_name>"
      exit 1
    fi
    run_training "$@"
    ;;
  predict)
    if [ "$#" -lt 4 ]; then
      echo "Usage: $0 predict <data_path> <model_path> <output_path>"
      exit 1
    fi
    run_prediction "$@"
    ;;
  deploy)
    if [ "$#" -lt 2 ]; then
      echo "Usage: $0 deploy <model_name>"
      exit 1
    fi
    deploy_model "$@"
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    # If the first argument is not one of our commands, assume it's a program to execute
    echo "Running custom command: $@"
    exec "$@"
    ;;
esac