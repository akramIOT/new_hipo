"""
Security module for encryption and decryption of sensitive data.
"""
import os
import base64
import logging
import json
from typing import Dict, Any, Optional, Union
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key,
    load_pem_public_key,
    PrivateFormat,
    PublicFormat,
    Encoding,
    NoEncryption,
)
from cryptography.x509.oid import NameOID
from cryptography.x509 import CertificateBuilder
from cryptography import x509
import datetime

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encryption and decryption of sensitive data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize encryption service.

        Args:
            config: Encryption configuration.
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.EncryptionService")

        # Initialize encryption keys
        self._init_encryption_keys()

    def _init_encryption_keys(self) -> None:
        """Initialize encryption keys."""
        # Check if keys should be loaded from environment
        if self.config.get("load_keys_from_env", False):
            self._load_keys_from_env()
        else:
            self._generate_or_load_keys()

    def _load_keys_from_env(self) -> None:
        """Load encryption keys from environment variables."""
        # Load symmetric key
        symmetric_key_b64 = os.environ.get("ENCRYPTION_SYMMETRIC_KEY")
        if symmetric_key_b64:
            self.symmetric_key = base64.urlsafe_b64decode(symmetric_key_b64)
            self.fernet = Fernet(symmetric_key_b64)
        else:
            self.logger.warning("Symmetric encryption key not found in environment, generating new key")
            self._generate_symmetric_key()

        # Load RSA keys
        private_key_pem = os.environ.get("ENCRYPTION_PRIVATE_KEY")
        if private_key_pem:
            try:
                self.private_key = load_pem_private_key(private_key_pem.encode(), password=None)
                self.public_key = self.private_key.public_key()
            except Exception as e:
                self.logger.error(f"Error loading RSA keys from environment: {e}")
                self._generate_rsa_keys()
        else:
            self.logger.warning("RSA keys not found in environment, generating new keys")
            self._generate_rsa_keys()

    def _generate_or_load_keys(self) -> None:
        """Generate new keys or load from files."""
        # Check if key directory exists
        key_dir = self.config.get("key_directory", "secrets")
        os.makedirs(key_dir, exist_ok=True)

        # Check if symmetric key exists
        symmetric_key_path = os.path.join(key_dir, "symmetric_key.key")
        if os.path.exists(symmetric_key_path):
            try:
                with open(symmetric_key_path, "rb") as f:
                    symmetric_key_b64 = f.read()
                self.symmetric_key = base64.urlsafe_b64decode(symmetric_key_b64)
                self.fernet = Fernet(symmetric_key_b64)
                self.logger.info("Loaded symmetric key from file")
            except Exception as e:
                self.logger.error(f"Error loading symmetric key: {e}")
                self._generate_symmetric_key(key_dir)
        else:
            self._generate_symmetric_key(key_dir)

        # Check if RSA keys exist
        private_key_path = os.path.join(key_dir, "private_key.pem")
        public_key_path = os.path.join(key_dir, "public_key.pem")

        if os.path.exists(private_key_path) and os.path.exists(public_key_path):
            try:
                with open(private_key_path, "rb") as f:
                    private_key_data = f.read()
                with open(public_key_path, "rb") as f:
                    public_key_data = f.read()

                self.private_key = load_pem_private_key(private_key_data, password=None)
                self.public_key = load_pem_public_key(public_key_data)
                self.logger.info("Loaded RSA keys from files")
            except Exception as e:
                self.logger.error(f"Error loading RSA keys: {e}")
                self._generate_rsa_keys(key_dir)
        else:
            self._generate_rsa_keys(key_dir)

    def _generate_symmetric_key(self, key_dir: Optional[str] = None) -> None:
        """Generate a new symmetric encryption key.

        Args:
            key_dir: Directory to save the key.
        """
        # Generate a new key
        self.symmetric_key = Fernet.generate_key()
        self.fernet = Fernet(self.symmetric_key)

        # Save the key if directory is provided
        if key_dir:
            key_path = os.path.join(key_dir, "symmetric_key.key")
            with open(key_path, "wb") as f:
                f.write(self.symmetric_key)
            self.logger.info(f"Generated and saved new symmetric key to {key_path}")
        else:
            self.logger.info("Generated new symmetric key (not saved)")

    def _generate_rsa_keys(self, key_dir: Optional[str] = None) -> None:
        """Generate new RSA key pair.

        Args:
            key_dir: Directory to save the keys.
        """
        # Generate a new key pair
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()

        # Save the keys if directory is provided
        if key_dir:
            private_key_path = os.path.join(key_dir, "private_key.pem")
            public_key_path = os.path.join(key_dir, "public_key.pem")

            private_key_pem = self.private_key.private_bytes(
                encoding=Encoding.PEM, format=PrivateFormat.PKCS8, encryption_algorithm=NoEncryption()
            )

            public_key_pem = self.public_key.public_bytes(
                encoding=Encoding.PEM, format=PublicFormat.SubjectPublicKeyInfo
            )

            with open(private_key_path, "wb") as f:
                f.write(private_key_pem)
            with open(public_key_path, "wb") as f:
                f.write(public_key_pem)

            self.logger.info(f"Generated and saved new RSA keys to {key_dir}")
        else:
            self.logger.info("Generated new RSA keys (not saved)")

    def generate_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate a key from a password using PBKDF2.

        Args:
            password: Password to derive key from.
            salt: Salt for PBKDF2. If None, a random salt is generated.

        Returns:
            Generated key.
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)

        key = kdf.derive(password.encode())

        return key, salt

    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """Encrypt data using symmetric encryption.

        Args:
            data: Data to encrypt (string, bytes, or JSON-serializable object).

        Returns:
            Encrypted data as bytes.
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode()

        try:
            return self.fernet.encrypt(data)
        except Exception as e:
            self.logger.error(f"Error encrypting data: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption.

        Args:
            encrypted_data: Encrypted data as bytes.

        Returns:
            Decrypted data as bytes.
        """
        try:
            return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            self.logger.error(f"Error decrypting data: {e}")
            raise

    def rsa_encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data using RSA public key.

        Args:
            data: Data to encrypt (string or bytes).

        Returns:
            Encrypted data as bytes.
        """
        if isinstance(data, str):
            data = data.encode()

        try:
            return self.public_key.encrypt(
                data, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
            )
        except Exception as e:
            self.logger.error(f"Error encrypting data with RSA: {e}")
            raise

    def rsa_decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA private key.

        Args:
            encrypted_data: Encrypted data as bytes.

        Returns:
            Decrypted data as bytes.
        """
        try:
            return self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
            )
        except Exception as e:
            self.logger.error(f"Error decrypting data with RSA: {e}")
            raise

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file using symmetric encryption.

        Args:
            file_path: Path to file to encrypt.
            output_path: Path to save encrypted file. If None, appends '.enc' to input path.

        Returns:
            Path to encrypted file.
        """
        if output_path is None:
            output_path = f"{file_path}.enc"

        try:
            with open(file_path, "rb") as f:
                data = f.read()

            encrypted_data = self.encrypt_data(data)

            with open(output_path, "wb") as f:
                f.write(encrypted_data)

            return output_path
        except Exception as e:
            self.logger.error(f"Error encrypting file {file_path}: {e}")
            raise

    def decrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file using symmetric encryption.

        Args:
            file_path: Path to encrypted file.
            output_path: Path to save decrypted file. If None, removes '.enc' from input path.

        Returns:
            Path to decrypted file.
        """
        if output_path is None:
            if file_path.endswith(".enc"):
                output_path = file_path[:-4]
            else:
                output_path = f"{file_path}.dec"

        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.decrypt_data(encrypted_data)

            with open(output_path, "wb") as f:
                f.write(decrypted_data)

            return output_path
        except Exception as e:
            self.logger.error(f"Error decrypting file {file_path}: {e}")
            raise

    def generate_self_signed_cert(self, common_name: str, validity_days: int = 365) -> tuple:
        """Generate a self-signed TLS certificate.

        Args:
            common_name: Common name for the certificate.
            validity_days: Number of days the certificate is valid for.

        Returns:
            Tuple of (certificate, private key) in PEM format.
        """
        # Generate a new private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Create a self-signed certificate
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ML Infrastructure"),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ]
        )

        # Certificate validity period
        now = datetime.datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=validity_days))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(common_name)]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Convert to PEM format
        cert_pem = cert.public_bytes(encoding=Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=Encoding.PEM, format=PrivateFormat.PKCS8, encryption_algorithm=NoEncryption()
        )

        return cert_pem, key_pem

    def generate_hmac(self, data: Union[str, bytes], key: Optional[bytes] = None) -> bytes:
        """Generate HMAC for data.

        Args:
            data: Data to generate HMAC for (string or bytes).
            key: Key to use for HMAC. If None, uses symmetric key.

        Returns:
            HMAC as bytes.
        """
        if isinstance(data, str):
            data = data.encode()

        if key is None:
            key = self.symmetric_key

        from cryptography.hazmat.primitives import hmac as crypto_hmac

        h = crypto_hmac.HMAC(key, hashes.SHA256())
        h.update(data)
        return h.finalize()

    def verify_hmac(self, data: Union[str, bytes], signature: bytes, key: Optional[bytes] = None) -> bool:
        """Verify HMAC for data.

        Args:
            data: Data to verify HMAC for (string or bytes).
            signature: Expected HMAC.
            key: Key to use for HMAC. If None, uses symmetric key.

        Returns:
            True if HMAC is valid, False otherwise.
        """
        if isinstance(data, str):
            data = data.encode()

        if key is None:
            key = self.symmetric_key

        from cryptography.hazmat.primitives import hmac as crypto_hmac

        h = crypto_hmac.HMAC(key, hashes.SHA256())
        h.update(data)

        try:
            h.verify(signature)
            return True
        except Exception:
            return False

    def generate_api_key(self) -> str:
        """Generate a secure API key.

        Returns:
            API key.
        """
        # Generate 32 random bytes and encode in URL-safe base64
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def hash_password(self, password: str) -> tuple:
        """Hash a password securely.

        Args:
            password: Password to hash.

        Returns:
            Tuple of (hash, salt) in base64 encoding.
        """
        salt = os.urandom(16)
        key, _ = self.generate_key_from_password(password, salt)

        return base64.b64encode(key).decode(), base64.b64encode(salt).decode()

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against a hash.

        Args:
            password: Password to verify.
            password_hash: Expected password hash in base64 encoding.
            salt: Salt used for hashing in base64 encoding.

        Returns:
            True if password is valid, False otherwise.
        """
        salt_bytes = base64.b64decode(salt)
        key, _ = self.generate_key_from_password(password, salt_bytes)

        expected_hash = base64.b64decode(password_hash)

        return secrets.compare_digest(key, expected_hash)
