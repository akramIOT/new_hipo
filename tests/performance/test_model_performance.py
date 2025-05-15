"""
Performance tests for model training and inference.
Used to benchmark and detect performance regressions.
"""

import os
import tempfile
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pytest


def test_data_loading_performance(benchmark):
    """Test performance of data loading operations."""
    
    def create_and_load_data():
        # Generate synthetic data
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_X:
            X_path = f_X.name
            np.save(f_X, X_train)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_y:
            y_path = f_y.name
            np.save(f_y, y_train)
            
        # Load data back
        X_loaded = np.load(X_path)
        y_loaded = np.load(y_path)
        
        # Clean up
        os.unlink(X_path)
        os.unlink(y_path)
        
        return X_loaded, y_loaded
    
    # Benchmark the function
    result = benchmark(create_and_load_data)
    assert len(result[0]) == 8000  # Verify we got the expected data back


def test_model_training_performance(benchmark):
    """Test performance of model training."""
    from sklearn.ensemble import RandomForestClassifier
    
    def train_model():
        # Generate synthetic data
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Make predictions to verify
        preds = model.predict(X[:10])
        
        return model, preds
    
    # Benchmark the function
    model, preds = benchmark(train_model)
    assert len(preds) == 10  # Verify we got predictions


def test_model_inference_performance(benchmark):
    """Test performance of model inference."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Setup - train model first (not benchmarked)
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Benchmark only the inference
    def model_inference():
        # Create test data - same shape but different values
        X_test, _ = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=43  # Different seed
        )
        
        # Run inference
        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)
        
        return predictions, proba
    
    # Benchmark the function
    predictions, probabilities = benchmark(model_inference)
    assert len(predictions) == 1000  # Verify we got the expected number of predictions