"""
Example script for training a model with the ML infrastructure.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import Config
from src.utils.logging_utils import setup_logging
from src.models.sklearn_model import SklearnModel
from src.utils.preprocessing import Preprocessor
from src.pipelines.pipeline import Pipeline


def load_data():
    """Load example data."""
    # Load Iris dataset
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    
    return {
        'data': data,
        'target_column': 'target',
        'feature_columns': iris.feature_names
    }


def preprocess_data(data, target_column, feature_columns):
    """Preprocess the data."""
    # Split data
    X = data[feature_columns]
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess data
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor
    }


def train_model(X_train, y_train, models_dir):
    """Train a model."""
    # Create scikit-learn model
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Create model wrapper
    model = SklearnModel('iris_classifier', rf_classifier, models_dir)
    
    # Train model
    training_params = {
        'cv': 5,
        'cv_scoring': 'accuracy'
    }
    
    metrics = model.train(X_train, y_train, **training_params)
    
    return {
        'model': model,
        'metrics': metrics
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return {
        'metrics': metrics
    }


def save_model(model, output_dir):
    """Save the model."""
    # Save model
    model_path = model.save()
    
    return {
        'model_path': model_path
    }


def main():
    """Main function."""
    # Setup logging
    logger = setup_logging(level='INFO')
    
    # Load configuration
    config = Config()
    
    # Set up directories
    models_dir = os.path.abspath('models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = Pipeline('iris_classification_pipeline')
    
    # Add steps
    pipeline.add_step('load_data', load_data)
    pipeline.add_step('preprocess_data', preprocess_data, 
                    data=lambda data: data['data'],
                    target_column=lambda data: data['target_column'],
                    feature_columns=lambda data: data['feature_columns'])
    pipeline.add_step('train_model', train_model, 
                    X_train=lambda data: data['X_train'],
                    y_train=lambda data: data['y_train'],
                    models_dir=models_dir)
    pipeline.add_step('evaluate_model', evaluate_model,
                    model=lambda data: data['model'],
                    X_test=lambda data: data['X_test'],
                    y_test=lambda data: data['y_test'])
    pipeline.add_step('save_model', save_model, 
                    model=lambda data: data['model'],
                    output_dir=models_dir)
    
    # Run pipeline
    results = pipeline.run()
    
    # Print results
    print("\nTraining metrics:")
    for name, value in results['metrics'].items():
        if name.startswith('train_'):
            print(f"  {name}: {value:.4f}")
            
    print("\nTest metrics:")
    for name, value in results['metrics']['metrics'].items():
        print(f"  {name}: {value:.4f}")
        
    print(f"\nModel saved to: {results['model_path']}")
    

if __name__ == "__main__":
    main()
