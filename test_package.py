"""Test script for the HIPO package"""
import os
import sys


def test_import():
    """Test importing the package modules"""
    try:
        import src
        print(f"Successfully imported src package, version: {src.__version__}")
        
        # Test importing some key modules
        from src.cloud import provider
        from src.models import model_base
        from src.config import config
        from src.api import app
        
        print("Successfully imported key modules")
        return True
    except ImportError as e:
        print(f"Error importing package: {str(e)}")
        return False


def test_config():
    """Test basic configuration loading"""
    try:
        from src.config.config import Config
        
        # Create a simple config object
        cfg = Config()
        print("Successfully created Config object")
        return True
    except Exception as e:
        print(f"Error creating Config object: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing HIPO package...")
    
    import_success = test_import()
    config_success = test_config()
    
    if import_success and config_success:
        print("\n✅ All tests passed! The package is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
