# HIPO (High Performance Multi-Cloud K8's Cluster Orchestration) - Multi-Cloud Kubernetes ML Platform

A modular and scalable infrastructure for deploying machine learning and LLM models across multiple cloud providers.

## Overview

This project provides a comprehensive infrastructure for ML/LLM workflows, including:

- Multi-cloud Kubernetes orchestration (AWS EKS and GCP GKE)
- GPU-optimized autoscaling for ML/LLM workloads based on GPU Metrics monitoring
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

## System Design and Architecture: 

![New Note](https://github.com/user-attachments/assets/989ae643-4f45-476a-9d9d-0e9177019aaf)


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
- `GET /api/v1/model-weights`: List available model weights
- `POST /api/v1/model-weights/<model_name>`: Upload model weights
- `GET /api/v1/model-weights/<model_name>/<version>`: Download model weights

## Streamlit UI Features

The platform includes a comprehensive Streamlit-based UI with the following features:

- **Dashboard**: Overall platform status, resource usage, and cost metrics
- **Model Deployment**: Interface for deploying ML/LLM models across cloud providers
- **Model Inference**: Test interface for running inference with deployed models
- **Configuration**: Management of cloud providers, Kubernetes, and model configurations
- **Monitoring**: Real-time metrics, logs, and alerting dashboard
- **Logs**: Searchable log viewer with filtering capabilities

![New Note](https://github.com/user-attachments/assets/53aa90b6-7288-4c50-9bbb-7aa2d5492a2d)

![New Note](https://github.com/user-attachments/assets/5b59f620-a71c-4ffb-8ab2-a4eaf28f6efd)

![New Note](https://github.com/user-attachments/assets/d81bc6e9-cf60-4d3c-ab0a-bfdd6f0ab6b3)


## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment. The CI/CD pipeline includes:

- Automated linting and code quality checks
- Unit and integration testing across multiple Python versions
- Security scanning with Bandit and Safety
- Python package building and publishing
- Docker image building and publishing
- Automated deployment to development and production environments

For details about the CI/CD setup and release process, see [CI/CD Guide](docs/ci-cd-guide.md).

### GitHub Secrets Management

This project requires several GitHub Secrets to be configured for the CI/CD pipeline to function properly. These include:

- AWS credentials for deployment and testing
- Docker Hub credentials for image publishing
- PyPI credentials for package publishing
- Codecov token for coverage reporting

For details on setting up the required secrets, see [GitHub Secrets Setup](docs/github-secrets-setup.md).

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

### Managing Model Weights

The platform includes a secure model weights management system that works across multiple cloud providers:

```python
from src.secrets.secret_manager import SecretManager
from src.cloud.factory import CloudProviderFactory

# Set up cloud providers
factory = CloudProviderFactory()
cloud_providers = {
    "aws": factory.create_provider("aws", aws_config),
    "gcp": factory.create_provider("gcp", gcp_config)
}

# Create secret manager
secret_manager = SecretManager(config, cloud_providers)
secret_manager.start()

# Upload model weights
secret_manager.upload_model_weights("llama-7b", "/path/to/model/weights")

# List available models
models = secret_manager.list_available_models()
for model_name, versions in models.items():
    print(f"{model_name}: {versions}")

# Download latest model weights
secret_manager.download_model_weights("llama-7b", "/output/path")

# Download specific version
secret_manager.download_model_weights("llama-7b", "/output/path", version="20230815")

# Clean up
secret_manager.stop()
```

Key features of the model weights management system:

- **Multi-cloud storage**: Transparently store and sync weights across AWS S3, GCP Cloud Storage, and other providers
- **Versioning**: Maintain multiple versions of model weights with automatic versioning
- **Secure access**: Fully integrated with the secret management system for secure credentials
- **Checksumming**: Automatic validation of weights integrity during transfers
- **Cross-cloud replication**: Replicate weights across clouds for reliability and high availability
- **Encryption**: End-to-end encryption for model weights

## License

MIT License

Copyright (c) 2025 Akram Sheriff (sheriff.akram.usa@gmail.com)

For questions, suggestions, or contributions, please contact: sheriff.akram.usa@gmail.com
