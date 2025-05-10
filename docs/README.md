# HIPO - Multi-Cloud Kubernetes ML Platform

A modular and scalable infrastructure for deploying machine learning and LLM models across multiple cloud providers.

## Overview

This project provides a comprehensive infrastructure for ML/LLM workflows, including:

- Multi-cloud Kubernetes orchestration (AWS EKS and GCP GKE)
- GPU-optimized autoscaling for ML/LLM workloads
- Global API gateway with intelligent routing
- Fault tolerance with cross-cloud failover
- Cost optimization across cloud providers
- Model training, evaluation, and serving
- Streamlit-based UI for platform management
- Comprehensive observability with metrics and logging

## Project Structure

```
├── config/             # Kubernetes and model configurations 
├── data/               # Data files
├── logs/               # Log files
├── models/             # Saved models
├── src/                # Source code
│   ├── api/            # API server
│   ├── autoscaling/    # GPU autoscaling components
│   ├── cloud/          # Cloud provider implementations
│   ├── config/         # Configuration management
│   ├── data/           # Data loading and preprocessing
│   ├── gateway/        # API gateway and routing
│   ├── kubernetes/     # Kubernetes orchestration
│   ├── models/         # Model implementations
│   ├── observability/  # Metrics and tracing
│   ├── pipelines/      # Pipeline orchestration
│   ├── secrets/        # Secret management 
│   ├── security/       # Encryption and security
│   ├── ui/             # Streamlit web interface
│   └── utils/          # Utility functions 
└── tests/              # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-infra.git
cd ml-infra
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

The platform uses YAML files for configuration of Kubernetes resources, cloud providers, and models:

```yaml
# kubernetes_config.yaml example
apiVersion: v1
kind: ConfigMap
metadata:
  name: hipo-config
  namespace: ml-models
data:
  log_level: "INFO"
  monitoring_enabled: "true"
  auto_scaling_enabled: "true"
  default_replicas: "2"
  gpu_resource_limit: "1"
```

### Training a Model

```bash
python -m src.main --mode train --config config/default_config.yaml --data data/train.csv --model my_model
```

### Making Predictions

```bash
python -m src.main --mode predict --config config/default_config.yaml --data data/test.csv --model my_model.pkl --output predictions.csv
```

### Running the API Server

```bash
python -m src.main --mode serve --config config/default_config.yaml --model my_model.pkl --port 5000
```

### Running the Streamlit UI

```bash
# Make sure streamlit is installed
pip install streamlit

# Run the UI
python src/ui/run_ui.py
# or use streamlit directly
streamlit run src/ui/app.py
```

## API Gateway Endpoints

- `GET /api/v1/models`: List available models
- `GET /api/v1/models/<model_name>`: Get model information
- `POST /api/v1/models/<model_name>/predict`: Make predictions using the model
- `POST /api/v1/models/<model_name>/generate`: Generate text with LLM models
- `POST /api/v1/models/<model_name>/embed`: Get embeddings for input text
- `POST /api/v1/models/<model_name>/evaluate`: Evaluate model performance
- `GET /api/v1/health`: Health check endpoint
- `GET /api/v1/metrics`: Platform metrics endpoint

## Streamlit UI Features

The platform includes a comprehensive Streamlit-based UI with the following features:

- **Dashboard**: Overall platform status, resource usage, and cost metrics
- **Model Deployment**: Interface for deploying ML/LLM models across cloud providers
- **Model Inference**: Test interface for running inference with deployed models
- **Configuration**: Management of cloud providers, Kubernetes, and model configurations
- **Monitoring**: Real-time metrics, logs, and alerting dashboard
- **Logs**: Searchable log viewer with filtering capabilities

## Development

### Adding a New Model

To add a new model, create a new class that inherits from `ModelBase`:

```python
from src.models.model_base import ModelBase

class MyModel(ModelBase):
    def __init__(self, model_name, **kwargs):
        super().__init__(model_name, **kwargs)
        # Initialize your model
        
    def train(self, X, y, **kwargs):
        # Implement training logic
        
    def predict(self, X):
        # Implement prediction logic
        
    def evaluate(self, X, y):
        # Implement evaluation logic
```

### Creating a Pipeline

```python
from src.pipelines.pipeline import Pipeline

# Create a pipeline
pipeline = Pipeline('my_pipeline')

# Add steps
pipeline.add_step('load_data', load_data_function, data_path='data/train.csv')
pipeline.add_step('preprocess', preprocess_function)
pipeline.add_step('train_model', train_model_function, model_name='my_model')

# Run the pipeline
results = pipeline.run()
```

## License

MIT
