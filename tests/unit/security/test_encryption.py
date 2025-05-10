"""
Unit tests for the encryption service.
"""
import os
import tempfile
import unittest
from unittest import mock
import json
import time
from pathlib import Path

import pytest

from src.security.encryption import EncryptionService


class TestEncryptionService(unittest.TestCase):
    """Test case for EncryptionService."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test keys
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Set up an encryption service for testing
        self.encryption_config = {
            'key_directory': self.temp_dir.name,
            'load_keys_from_env': False
        }
        self.encryption_service = EncryptionService(self.encryption_config)
        
        # Test data
        self.test_data = b'test data for encryption'
        self.test_string = 'test string for encryption'
        self.test_dict = {'key1': 'value1', 'key2': 42}
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
        
    def test_initialization(self):
        """Test encryption service initialization."""
        # Check that keys were generated
        self.assertIsNotNone(self.encryption_service.symmetric_key)
        self.assertIsNotNone(self.encryption_service.fernet)
        self.assertIsNotNone(self.encryption_service.private_key)
        self.assertIsNotNone(self.encryption_service.public_key)
        
        # Check that key files were created
        symmetric_key_path = os.path.join(self.temp_dir.name, 'symmetric_key.key')
        private_key_path = os.path.join(self.temp_dir.name, 'private_key.pem')
        public_key_path = os.path.join(self.temp_dir.name, 'public_key.pem')
        
        self.assertTrue(os.path.exists(symmetric_key_path))
        self.assertTrue(os.path.exists(private_key_path))
        self.assertTrue(os.path.exists(public_key_path))
        
    def test_symmetric_encryption(self):
        """Test symmetric encryption and decryption."""
        # Encrypt bytes
        encrypted_bytes = self.encryption_service.encrypt_data(self.test_data)
        decrypted_bytes = self.encryption_service.decrypt_data(encrypted_bytes)
        self.assertEqual(decrypted_bytes, self.test_data)
        
        # Encrypt string
        encrypted_string = self.encryption_service.encrypt_data(self.test_string)
        decrypted_string = self.encryption_service.decrypt_data(encrypted_string)
        self.assertEqual(decrypted_string.decode(), self.test_string)
        
        # Encrypt dictionary
        encrypted_dict = self.encryption_service.encrypt_data(self.test_dict)
        decrypted_dict = self.encryption_service.decrypt_data(encrypted_dict)
        self.assertEqual(json.loads(decrypted_dict), self.test_dict)
        
    def test_rsa_encryption(self):
        """Test RSA encryption and decryption."""
        # RSA can only encrypt small amounts of data
        small_data = b'small test data'
        
        # Encrypt with public key
        encrypted_data = self.encryption_service.rsa_encrypt(small_data)
        
        # Decrypt with private key
        decrypted_data = self.encryption_service.rsa_decrypt(encrypted_data)
        
        self.assertEqual(decrypted_data, small_data)
        
    def test_file_encryption(self):
        """Test file encryption and decryption."""
        # Create a test file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(self.test_data)
            temp_path = temp_file.name
            
        try:
            # Encrypt the file
            encrypted_path = self.encryption_service.encrypt_file(temp_path)
            
            # Check that the encrypted file exists
            self.assertTrue(os.path.exists(encrypted_path))
            
            # Decrypt the file
            decrypted_path = self.encryption_service.decrypt_file(encrypted_path)
            
            # Check that the decrypted file exists
            self.assertTrue(os.path.exists(decrypted_path))
            
            # Check that the decrypted content matches the original
            with open(decrypted_path, 'rb') as f:
                decrypted_data = f.read()
                
            self.assertEqual(decrypted_data, self.test_data)
            
        finally:
            # Clean up test files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(encrypted_path):
                os.unlink(encrypted_path)
            if os.path.exists(decrypted_path):
                os.unlink(decrypted_path)
                
    def test_password_based_key(self):
        """Test key generation from password."""
        password = "test_password"
        
        # Generate key with random salt
        key1, salt = self.encryption_service.generate_key_from_password(password)
        
        # Generate key with the same salt
        key2, _ = self.encryption_service.generate_key_from_password(password, salt)
        
        # Check that keys are the same when using the same salt
        self.assertEqual(key1, key2)
        
        # Generate key with different salt
        different_salt = os.urandom(16)
        key3, _ = self.encryption_service.generate_key_from_password(password, different_salt)
        
        # Check that keys are different when using different salt
        self.assertNotEqual(key1, key3)
        
    def test_password_hash_verify(self):
        """Test password hashing and verification."""
        password = "test_password"
        
        # Hash the password
        password_hash, salt = self.encryption_service.hash_password(password)
        
        # Verify correct password
        self.assertTrue(self.encryption_service.verify_password(password, password_hash, salt))
        
        # Verify incorrect password
        self.assertFalse(self.encryption_service.verify_password("wrong_password", password_hash, salt))
        
    def test_hmac(self):
        """Test HMAC generation and verification."""
        # Generate HMAC
        hmac = self.encryption_service.generate_hmac(self.test_data)
        
        # Verify HMAC with correct data
        self.assertTrue(self.encryption_service.verify_hmac(self.test_data, hmac))
        
        # Verify HMAC with incorrect data
        self.assertFalse(self.encryption_service.verify_hmac(b'wrong data', hmac))
        
    def test_api_key_generation(self):
        """Test API key generation."""
        # Generate API key
        api_key = self.encryption_service.generate_api_key()
        
        # Check that API key is a string
        self.assertIsInstance(api_key, str)
        
        # Check that API key is URL-safe base64
        try:
            import base64
            decoded = base64.urlsafe_b64decode(api_key)
            # API key should be 32 bytes decoded
            self.assertEqual(len(decoded), 32)
        except Exception as e:
            self.fail(f"API key is not valid URL-safe base64: {e}")
            
    def test_self_signed_cert(self):
        """Test self-signed certificate generation."""
        # Generate certificate
        common_name = "test.example.com"
        cert_pem, key_pem = self.encryption_service.generate_self_signed_cert(common_name)
        
        # Check that certificate and key are bytes
        self.assertIsInstance(cert_pem, bytes)
        self.assertIsInstance(key_pem, bytes)
        
        # Check that they look like PEM format (begin with -----)
        self.assertTrue(cert_pem.startswith(b'-----'))
        self.assertTrue(key_pem.startswith(b'-----'))


def test_loading_keys_from_env():
    """Test loading keys from environment variables."""
    # Create a test environment
    test_symmetric_key = 'ZmFrZSBzeW1tZXRyaWMga2V5IGZvciB0ZXN0aW5n'  # Base64 encoded
    test_private_key = '-----BEGIN PRIVATE KEY-----\nDummy PEM key for testing\n-----END PRIVATE KEY-----'
    with mock.patch.dict(os.environ, {
        'ENCRYPTION_SYMMETRIC_KEY': test_symmetric_key,
        'ENCRYPTION_PRIVATE_KEY': test_private_key
    }):
        # Create encryption service with load_keys_from_env=True
        with pytest.raises(Exception):
            # This will fail because the private key is not valid, but it will try to load it
            EncryptionService({'load_keys_from_env': True})


def test_load_or_generate_keys():
    """Test loading or generating keys."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an encryption service (should generate keys)
        encryption_service1 = EncryptionService({'key_directory': temp_dir})
        
        # Get the keys
        symmetric_key1 = encryption_service1.symmetric_key
        
        # Create another encryption service (should load the same keys)
        encryption_service2 = EncryptionService({'key_directory': temp_dir})
        
        # Check that the keys are the same
        symmetric_key2 = encryption_service2.symmetric_key
        assert symmetric_key1 == symmetric_key2


def test_key_generation_without_directory():
    """Test key generation without a directory (not saved)."""
    # Create an encryption service without a key directory
    encryption_service = EncryptionService({'key_directory': None})
    
    # Check that keys were generated but not saved
    assert encryption_service.symmetric_key is not None
    assert encryption_service.private_key is not None
