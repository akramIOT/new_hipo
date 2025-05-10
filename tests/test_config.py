"""
Tests for configuration module.
"""
import os
import pytest
import yaml
from pathlib import Path

from src.config.config import Config


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config_data = {
        'paths': {
            'data_dir': 'test_data',
            'models_dir': 'test_models',
            'logs_dir': 'test_logs'
        },
        'test_key': 'test_value'
    }
    
    config_file = tmp_path / 'test_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
        
    return config_file


def test_config_init():
    """Test Config initialization."""
    config = Config()
    assert config.config_data == {}
    assert isinstance(config.project_root, Path)
    
    # Test default paths
    assert config.data_dir.name == 'data'
    assert config.models_dir.name == 'models'
    assert config.logs_dir.name == 'logs'


def test_config_load(temp_config_file):
    """Test loading configuration from file."""
    config = Config(temp_config_file)
    
    # Test loaded data
    assert config.get('test_key') == 'test_value'
    
    # Test paths
    assert config.data_dir.name == 'test_data'
    assert config.models_dir.name == 'test_models'
    assert config.logs_dir.name == 'test_logs'


def test_config_get():
    """Test getting configuration values."""
    config = Config()
    config.config_data = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        }
    }
    
    # Test direct key
    assert config.get('key1') == 'value1'
    
    # Test nested key
    assert config.get('key2')['nested_key'] == 'nested_value'
    
    # Test default value
    assert config.get('non_existent_key', 'default') == 'default'


def test_config_getitem():
    """Test dictionary-like access."""
    config = Config()
    config.config_data = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        }
    }
    
    # Test direct key
    assert config['key1'] == 'value1'
    
    # Test nested key
    assert config['key2']['nested_key'] == 'nested_value'
    
    # Test KeyError
    with pytest.raises(KeyError):
        _ = config['non_existent_key']


def test_config_save(tmp_path):
    """Test saving configuration to file."""
    config = Config()
    config.config_data = {
        'key1': 'value1',
        'key2': {
            'nested_key': 'nested_value'
        }
    }
    
    # Save configuration
    config_file = tmp_path / 'config.yaml'
    config.save_config(config_file)
    
    # Load saved configuration
    with open(config_file, 'r') as f:
        loaded_data = yaml.safe_load(f)
        
    # Verify loaded data matches original
    assert loaded_data == config.config_data


def test_config_from_yaml(temp_config_file):
    """Test creating Config from YAML file."""
    config = Config.from_yaml(temp_config_file)
    
    # Test loaded data
    assert config.get('test_key') == 'test_value'
    
    # Test paths
    assert config.data_dir.name == 'test_data'
    assert config.models_dir.name == 'test_models'
    assert config.logs_dir.name == 'test_logs'
