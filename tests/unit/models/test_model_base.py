"""
Unit tests for the model base functionality.
"""
import os
import tempfile
import unittest
from unittest import mock
import json
import pickle
import time
from pathlib import Path

import pytest
import numpy as np

from src.models.model_base import ModelBase
from src.models.secure_weights import SecureModelWeights
from src.security.encryption import EncryptionService


class TestModelBase(unittest.TestCase):
    """Test case for ModelBase."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for models
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)
        
        # Create a simple model
        self.model_name = "test_model"
        self.model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Set some dummy model data
        self.model.model = {"type": "dummy", "data": [1, 2, 3, 4, 5]}
        self.model.metadata['params'] = {"param1": 10, "param2": "test"}
        self.model.metadata['metrics'] = {"accuracy": 0.95, "loss": 0.1}
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_init(self):
        """Test model initialization."""
        # Test initialization with default parameters
        model = ModelBase("default_test")
        self.assertEqual(model.model_name, "default_test")
        self.assertIsNone(model.model)
        self.assertEqual(model.metadata['name'], "default_test")
        
        # Test initialization with custom models_dir
        custom_dir = "custom_models_dir"
        model = ModelBase("custom_test", models_dir=custom_dir)
        self.assertEqual(model.models_dir, Path(custom_dir))
        
    def test_standard_save_load(self):
        """Test standard save and load functionality."""
        # Save the model
        file_path = self.model.save()
        
        # Check that the file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Create a new model instance
        new_model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Load the model
        new_model.load()
        
        # Check that the loaded model data matches the original
        self.assertEqual(new_model.model, self.model.model)
        self.assertEqual(new_model.metadata['params'], self.model.metadata['params'])
        self.assertEqual(new_model.metadata['metrics'], self.model.metadata['metrics'])
        
    def test_save_load_with_filename(self):
        """Test save and load with custom filename."""
        # Save the model with custom filename
        custom_filename = "custom_model.pkl"
        file_path = self.model.save(filename=custom_filename)
        
        # Check that the file exists with the custom name
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(os.path.basename(file_path) == custom_filename)
        
        # Create a new model instance
        new_model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Load the model with custom filename
        new_model.load(filename=custom_filename)
        
        # Check that the loaded model data matches the original
        self.assertEqual(new_model.model, self.model.model)
        
    def test_secure_save_load(self):
        """Test secure save and load functionality."""
        # Create a mock SecureModelWeights instance
        mock_secure_weights = mock.MagicMock()
        
        # Set up the mock to return fake metadata and data when store_weights is called
        fake_metadata = {
            'model_name': self.model_name,
            'version': 'v_test',
            'timestamp': int(time.time())
        }
        mock_secure_weights.store_weights.return_value = fake_metadata
        
        # Create pickle data that would be returned when load_weights is called
        pickle_data = pickle.dumps({
            'model': self.model.model,
            'metadata': self.model.metadata
        })
        mock_secure_weights.load_weights_to_file.side_effect = lambda **kwargs: fake_metadata
        mock_secure_weights.load_weights.return_value = (pickle_data, fake_metadata)
        
        # Save the model with secure storage
        result = self.model.save(secure=True, secure_weights_manager=mock_secure_weights, version='v_test')
        
        # Check that store_weights was called with the right parameters
        mock_secure_weights.store_weights.assert_called_once()
        call_args = mock_secure_weights.store_weights.call_args[1]
        self.assertEqual(call_args['model_name'], self.model_name)
        self.assertEqual(call_args['version'], 'v_test')
        self.assertTrue(call_args['encrypt'])
        
        # Check that the result is the metadata from store_weights
        self.assertEqual(result, fake_metadata)
        
        # Create a new model instance
        new_model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Load the model with secure storage
        new_model.load(secure=True, secure_weights_manager=mock_secure_weights, version='v_test')
        
        # Check that load_weights_to_file was called with the right parameters
        mock_secure_weights.load_weights_to_file.assert_called_once()
        call_args = mock_secure_weights.load_weights_to_file.call_args[1]
        self.assertEqual(call_args['model_name'], self.model_name)
        self.assertEqual(call_args['version'], 'v_test')
        self.assertTrue(call_args['decrypt'])
        
    def test_secure_save_without_manager(self):
        """Test secure save without a manager (should fall back to standard)."""
        # Save the model with secure=True but no manager
        file_path = self.model.save(secure=True)
        
        # Check that the file exists (should fall back to standard save)
        self.assertTrue(os.path.exists(file_path))
        
        # Create a new model instance
        new_model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Load the model
        new_model.load()
        
        # Check that the loaded model data matches the original
        self.assertEqual(new_model.model, self.model.model)
        
    def test_secure_load_without_manager(self):
        """Test secure load without a manager (should fall back to standard)."""
        # Save the model normally
        file_path = self.model.save()
        
        # Create a new model instance
        new_model = ModelBase(self.model_name, models_dir=self.models_dir)
        
        # Load the model with secure=True but no manager
        new_model.load(secure=True)
        
        # Check that the loaded model data matches the original (should fall back to standard load)
        self.assertEqual(new_model.model, self.model.model)
        
    def test_metadata_methods(self):
        """Test metadata manipulation methods."""
        # Test get_params
        params = self.model.get_params()
        self.assertEqual(params, {"param1": 10, "param2": "test"})
        
        # Test set_params
        new_params = {"param3": 20, "param4": "new_test"}
        self.model.set_params(new_params)
        params = self.model.get_params()
        self.assertEqual(params, {"param1": 10, "param2": "test", "param3": 20, "param4": "new_test"})
        
        # Test log_metric
        self.model.log_metric("precision", 0.88)
        metrics = self.model.get_metrics()
        self.assertEqual(metrics, {"accuracy": 0.95, "loss": 0.1, "precision": 0.88})
        
    def test_error_cases(self):
        """Test error cases."""
        # Test save without model
        empty_model = ModelBase("empty_model", models_dir=self.models_dir)
        with self.assertRaises(ValueError):
            empty_model.save()
            
        # Test predict without model
        with self.assertRaises(ValueError):
            empty_model.predict(None)
            
        # Test evaluate without model
        with self.assertRaises(ValueError):
            empty_model.evaluate(None, None)
            
        # Test load with nonexistent file
        nonexistent_model = ModelBase("nonexistent", models_dir=self.models_dir)
        with self.assertRaises(Exception):
            nonexistent_model.load()


class DummyModel(ModelBase):
    """Dummy model implementation for testing abstract methods."""
    
    def train(self, X, y, **kwargs):
        """Implement train method."""
        self.model = {"type": "trained_dummy", "shape": X.shape}
        self.metadata['params'] = kwargs
        return {"loss": 0.1}
        
    def predict(self, X):
        """Implement predict method."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return np.zeros(len(X))
        
    def evaluate(self, X, y):
        """Implement evaluate method."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return {"accuracy": 0.9, "loss": 0.2}


def test_abstract_methods():
    """Test implementation of abstract methods."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy model
        model = DummyModel("dummy", models_dir=temp_dir)
        
        # Test train method
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        result = model.train(X, y, learning_rate=0.01)
        
        assert result == {"loss": 0.1}
        assert model.model["type"] == "trained_dummy"
        assert model.model["shape"] == X.shape
        assert model.metadata['params'] == {"learning_rate": 0.01}
        
        # Test predict method
        predictions = model.predict(X)
        assert predictions.shape == (2,)
        
        # Test evaluate method
        metrics = model.evaluate(X, y)
        assert metrics["accuracy"] == 0.9
        assert metrics["loss"] == 0.2
        
        # Test save and load
        file_path = model.save()
        assert os.path.exists(file_path)
        
        # Create new model and load
        new_model = DummyModel("dummy", models_dir=temp_dir)
        new_model.load()
        
        # Test methods with loaded model
        predictions = new_model.predict(X)
        assert predictions.shape == (2,)
        
        metrics = new_model.evaluate(X, y)
        assert metrics["accuracy"] == 0.9
