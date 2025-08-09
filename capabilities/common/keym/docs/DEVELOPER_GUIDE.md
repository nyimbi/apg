# APG Key Management - Developer Guide

## Overview

This guide provides comprehensive information for developers working with the APG Key Management system, including architecture, APIs, extension points, and best practices.

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    APG Platform Layer                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Event Bus   │  │ Service     │  │ Configuration       │  │
│  │             │  │ Mesh        │  │ Management          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Key Management Layer                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ API Layer   │  │ Service     │  │ Security            │  │
│  │             │  │ Layer       │  │ Intelligence        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Multi-Cloud │  │ HSM         │  │ Quantum-Safe        │  │
│  │ Federation  │  │ Integration │  │ Cryptography        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Data Layer                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PostgreSQL  │  │ Redis       │  │ HSM Storage         │  │
│  │             │  │ Cache       │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Core Data Models

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from uuid_extensions import uuid7str

class KeyAlgorithm(str, Enum):
    """Supported cryptographic algorithms"""
    AES_128 = "AES_128"
    AES_256 = "AES_256"
    RSA_2048 = "RSA_2048"
    RSA_4096 = "RSA_4096"
    ECDSA_P256 = "ECDSA_P256"
    ECDSA_P384 = "ECDSA_P384"
    KYBER_512 = "KYBER_512"    # Post-quantum
    KYBER_768 = "KYBER_768"    # Post-quantum
    KYBER_1024 = "KYBER_1024"  # Post-quantum

class KeyUsage(str, Enum):
    """Key usage patterns"""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    KEY_EXCHANGE = "key_exchange"
    DERIVE_KEY = "derive_key"

class KeySpecification(BaseModel):
    """Key specification model"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    id: str = Field(default_factory=uuid7str)
    tenant_id: str
    algorithm: KeyAlgorithm
    usage: List[KeyUsage]
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    rotation_policy: Optional[str] = None
```

## API Development

### Service Layer Architecture

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import asyncio
from contextlib import asynccontextmanager

class KeyManagementServiceInterface(ABC):
    """Abstract interface for key management services"""
    
    @abstractmethod
    async def initialize(self, tenant_id: str, config: Dict[str, Any] = None) -> None:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def create_key(self, spec: KeySpecification, user_id: str) -> KeyInfo:
        """Create a new cryptographic key"""
        pass
    
    @abstractmethod
    async def retrieve_key(self, key_id: str, user_id: str) -> KeyInfo:
        """Retrieve key information"""
        pass
    
    @abstractmethod
    async def encrypt_data(self, key_id: str, data: bytes, user_id: str) -> bytes:
        """Encrypt data using specified key"""
        pass
    
    @abstractmethod
    async def decrypt_data(self, key_id: str, encrypted_data: bytes, user_id: str) -> bytes:
        """Decrypt data using specified key"""
        pass

class KeyManagementService(KeyManagementServiceInterface):
    """Concrete implementation of key management service"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self._db_pool = None
        self._cache = None
        self._event_publisher = None
    
    async def initialize(self, tenant_id: str, config: Dict[str, Any] = None) -> None:
        """Initialize service with database connection, cache, and event publisher"""
        if self.is_initialized:
            return
        
        # Initialize database connection pool
        self._db_pool = await self._create_database_pool()
        
        # Initialize cache
        self._cache = await self._create_cache_client()
        
        # Initialize event publisher
        self._event_publisher = await self._create_event_publisher()
        
        # Initialize security components
        await self._initialize_security_components()
        
        self.tenant_id = tenant_id
        self.is_initialized = True
    
    async def _create_database_pool(self):
        """Create database connection pool"""
        import asyncpg
        return await asyncpg.create_pool(
            dsn=self.config.get('database_url'),
            min_size=10,
            max_size=100,
            command_timeout=60
        )
    
    @asynccontextmanager
    async def database_transaction(self):
        """Database transaction context manager"""
        async with self._db_pool.acquire() as connection:
            async with connection.transaction():
                yield connection
```

### REST API Implementation

```python
from flask import Blueprint, request, jsonify, current_app
from flask.views import MethodView
from marshmallow import Schema, fields, ValidationError
from keym.service import KeyManagementService
from keym.models import KeySpecification, KeyAlgorithm, KeyUsage
from keym.auth import require_authentication, require_permission

keym_api = Blueprint('keym_api', __name__, url_prefix='/api/v1')

class KeySpecificationSchema(Schema):
    """Schema for key specification validation"""
    algorithm = fields.Enum(KeyAlgorithm, required=True)
    usage = fields.List(fields.Enum(KeyUsage), required=True)
    name = fields.String(required=True, validate=lambda x: len(x) > 0)
    description = fields.String(missing=None)
    metadata = fields.Dict(missing=dict)
    expires_at = fields.DateTime(missing=None)
    rotation_policy = fields.String(missing=None)

class KeyAPI(MethodView):
    """RESTful API for key management"""
    
    def __init__(self):
        self.service = current_app.extensions['keym_service']
        self.schema = KeySpecificationSchema()
    
    @require_authentication
    @require_permission('keym:create_key')
    async def post(self):
        """Create a new key"""
        try:
            # Validate request data
            spec_data = self.schema.load(request.json)
            
            # Create key specification
            spec = KeySpecification(
                tenant_id=request.user.tenant_id,
                created_by=request.user.id,
                **spec_data
            )
            
            # Create key
            key = await self.service.create_key(spec, request.user.id)
            
            return jsonify({
                'status': 'success',
                'key_id': key.spec.id,
                'algorithm': key.spec.algorithm.value,
                'usage': [u.value for u in key.spec.usage],
                'created_at': key.metadata.created_at.isoformat()
            }), 201
            
        except ValidationError as e:
            return jsonify({'error': 'Validation failed', 'details': e.messages}), 400
        except Exception as e:
            current_app.logger.error(f"Key creation failed: {e}")
            return jsonify({'error': 'Key creation failed'}), 500
    
    @require_authentication
    @require_permission('keym:list_keys')
    async def get(self, key_id=None):
        """Get key information or list keys"""
        if key_id:
            # Get specific key
            try:
                key = await self.service.retrieve_key(key_id, request.user.id)
                return jsonify({
                    'key_id': key.spec.id,
                    'name': key.spec.name,
                    'algorithm': key.spec.algorithm.value,
                    'usage': [u.value for u in key.spec.usage],
                    'status': key.metadata.status,
                    'created_at': key.metadata.created_at.isoformat()
                })
            except KeyNotFoundError:
                return jsonify({'error': 'Key not found'}), 404
        else:
            # List keys
            limit = request.args.get('limit', 50, type=int)
            offset = request.args.get('offset', 0, type=int)
            
            keys = await self.service.list_keys(
                user_id=request.user.id,
                limit=limit,
                offset=offset
            )
            
            return jsonify({
                'keys': [
                    {
                        'key_id': key.spec.id,
                        'name': key.spec.name,
                        'algorithm': key.spec.algorithm.value,
                        'status': key.metadata.status,
                        'created_at': key.metadata.created_at.isoformat()
                    }
                    for key in keys
                ],
                'pagination': {
                    'limit': limit,
                    'offset': offset,
                    'total': await self.service.count_keys(request.user.id)
                }
            })

# Register API endpoints
keym_api.add_url_rule('/keys', view_func=KeyAPI.as_view('keys'))
keym_api.add_url_rule('/keys/<key_id>', view_func=KeyAPI.as_view('key_detail'))
```

### Cryptographic Operations API

```python
class CryptoAPI(MethodView):
    """API for cryptographic operations"""
    
    @require_authentication
    @require_permission('keym:encrypt')
    async def post(self, operation):
        """Perform cryptographic operations"""
        key_id = request.json.get('key_id')
        data = request.json.get('data')
        
        if not key_id or not data:
            return jsonify({'error': 'key_id and data are required'}), 400
        
        try:
            if operation == 'encrypt':
                # Decode base64 data
                import base64
                plain_data = base64.b64decode(data)
                
                # Encrypt
                encrypted_data = await self.service.encrypt_data(
                    key_id, plain_data, request.user.id
                )
                
                return jsonify({
                    'status': 'success',
                    'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                    'key_id': key_id
                })
            
            elif operation == 'decrypt':
                # Decode encrypted data
                encrypted_data = base64.b64decode(data)
                
                # Decrypt
                plain_data = await self.service.decrypt_data(
                    key_id, encrypted_data, request.user.id
                )
                
                return jsonify({
                    'status': 'success',
                    'decrypted_data': base64.b64encode(plain_data).decode('utf-8'),
                    'key_id': key_id
                })
            
            else:
                return jsonify({'error': f'Unknown operation: {operation}'}), 400
                
        except Exception as e:
            current_app.logger.error(f"Crypto operation {operation} failed: {e}")
            return jsonify({'error': f'Operation {operation} failed'}), 500

keym_api.add_url_rule('/crypto/<operation>', 
                     view_func=CryptoAPI.as_view('crypto_operations'),
                     methods=['POST'])
```

## Extension Points

### Plugin Architecture

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import importlib

class KeyManagementPlugin(ABC):
    """Base class for key management plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    async def initialize(self, service: KeyManagementService, config: Dict[str, Any]) -> None:
        """Initialize plugin"""
        pass
    
    @abstractmethod
    async def on_key_created(self, key_info: KeyInfo) -> None:
        """Called when a key is created"""
        pass
    
    @abstractmethod
    async def on_key_accessed(self, key_id: str, operation: str, user_id: str) -> None:
        """Called when a key is accessed"""
        pass

class PluginManager:
    """Manages key management plugins"""
    
    def __init__(self, service: KeyManagementService):
        self.service = service
        self.plugins: Dict[str, KeyManagementPlugin] = {}
    
    async def load_plugin(self, plugin_module: str, config: Dict[str, Any] = None) -> None:
        """Load and initialize a plugin"""
        try:
            module = importlib.import_module(plugin_module)
            plugin_class = getattr(module, 'Plugin')
            plugin = plugin_class()
            
            await plugin.initialize(self.service, config or {})
            self.plugins[plugin.name] = plugin
            
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin {plugin_module}: {e}")
    
    async def notify_key_created(self, key_info: KeyInfo) -> None:
        """Notify all plugins of key creation"""
        for plugin in self.plugins.values():
            try:
                await plugin.on_key_created(key_info)
            except Exception as e:
                self.service.logger.error(f"Plugin {plugin.name} error: {e}")
```

### Custom Algorithm Support

```python
class CustomCryptoProvider(ABC):
    """Abstract base for custom cryptographic providers"""
    
    @property
    @abstractmethod
    def supported_algorithms(self) -> List[str]:
        """List of supported algorithms"""
        pass
    
    @abstractmethod
    async def create_key(self, algorithm: str, key_size: int) -> bytes:
        """Create key material"""
        pass
    
    @abstractmethod
    async def encrypt(self, key_material: bytes, plaintext: bytes) -> bytes:
        """Encrypt data"""
        pass
    
    @abstractmethod
    async def decrypt(self, key_material: bytes, ciphertext: bytes) -> bytes:
        """Decrypt data"""
        pass

class HSMCryptoProvider(CustomCryptoProvider):
    """HSM-based cryptographic provider"""
    
    def __init__(self, hsm_config: Dict[str, Any]):
        self.hsm_config = hsm_config
        self.hsm_client = None
    
    @property
    def supported_algorithms(self) -> List[str]:
        return ["HSM_AES_256", "HSM_RSA_2048", "HSM_ECDSA_P256"]
    
    async def create_key(self, algorithm: str, key_size: int) -> bytes:
        """Create key in HSM"""
        if not self.hsm_client:
            await self._initialize_hsm()
        
        return await self.hsm_client.create_key(algorithm, key_size)
    
    async def _initialize_hsm(self):
        """Initialize HSM connection"""
        # Implementation specific to HSM vendor
        pass
```

## Testing Framework

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from keym.service import KeyManagementService
from keym.models import KeySpecification, KeyAlgorithm, KeyUsage

@pytest.fixture
async def service():
    """Create service instance for testing"""
    service = KeyManagementService({
        'database_url': 'postgresql://test:test@localhost/test_keym',
        'cache_url': 'redis://localhost:6379/0'
    })
    await service.initialize("test_tenant")
    yield service
    # Cleanup
    await service.cleanup()

@pytest.fixture
def sample_key_spec():
    """Sample key specification"""
    return KeySpecification(
        tenant_id="test_tenant",
        algorithm=KeyAlgorithm.AES_256,
        usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
        name="Test Key",
        created_by="test@example.com"
    )

class TestKeyManagementService:
    """Test suite for key management service"""
    
    @pytest.mark.asyncio
    async def test_create_key(self, service, sample_key_spec):
        """Test key creation"""
        key = await service.create_key(sample_key_spec, "test@example.com")
        
        assert key.spec.id is not None
        assert key.spec.algorithm == KeyAlgorithm.AES_256
        assert key.metadata.status == "active"
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt_cycle(self, service, sample_key_spec):
        """Test encryption and decryption"""
        # Create key
        key = await service.create_key(sample_key_spec, "test@example.com")
        
        # Test data
        original_data = b"This is test data for encryption"
        
        # Encrypt
        encrypted_data = await service.encrypt_data(
            key.spec.id, original_data, "test@example.com"
        )
        
        assert encrypted_data != original_data
        
        # Decrypt
        decrypted_data = await service.decrypt_data(
            key.spec.id, encrypted_data, "test@example.com"
        )
        
        assert decrypted_data == original_data
    
    @pytest.mark.asyncio
    async def test_key_rotation(self, service, sample_key_spec):
        """Test key rotation"""
        # Create original key
        original_key = await service.create_key(sample_key_spec, "test@example.com")
        original_key_id = original_key.spec.id
        
        # Rotate key
        rotated_key = await service.rotate_key(original_key_id, "test@example.com")
        
        assert rotated_key.spec.id != original_key_id
        assert rotated_key.spec.name == original_key.spec.name
        assert rotated_key.metadata.status == "active"
```

### Integration Testing

```python
@pytest.mark.integration
class TestKeyManagementIntegration:
    """Integration tests for key management"""
    
    @pytest.mark.asyncio
    async def test_hsm_integration(self, service):
        """Test HSM integration"""
        if not service.hsm_manager.is_available():
            pytest.skip("HSM not available")
        
        # Create HSM-backed key
        hsm_spec = KeySpecification(
            tenant_id="test_tenant",
            algorithm=KeyAlgorithm.AES_256,
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            name="HSM Test Key",
            created_by="test@example.com",
            metadata={"storage": "hsm", "hsm_id": "test_hsm"}
        )
        
        key = await service.create_key(hsm_spec, "test@example.com")
        assert key.metadata.storage_location == "hsm"
    
    @pytest.mark.asyncio
    async def test_multi_cloud_federation(self, service):
        """Test multi-cloud key federation"""
        from keym.multi_cloud_federation import CloudKeyFederation
        
        federation = CloudKeyFederation(service)
        
        # Create key
        spec = KeySpecification(
            tenant_id="test_tenant",
            algorithm=KeyAlgorithm.AES_256,
            usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            name="Federated Test Key",
            created_by="test@example.com"
        )
        
        key = await service.create_key(spec, "test@example.com")
        
        # Federate to multiple clouds (mock)
        await federation.federate_key(
            key.spec.id, 
            ["aws", "azure"], 
            "test@example.com"
        )
        
        # Verify federation
        federation_status = await federation.get_federation_status(key.spec.id)
        assert "aws" in federation_status
        assert "azure" in federation_status
```

### Performance Testing

```python
@pytest.mark.performance
class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_key_creation(self, service):
        """Test concurrent key creation performance"""
        import time
        
        concurrent_requests = 100
        
        async def create_test_key(index):
            spec = KeySpecification(
                tenant_id="test_tenant",
                algorithm=KeyAlgorithm.AES_256,
                usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
                name=f"Perf Test Key {index}",
                created_by="test@example.com"
            )
            return await service.create_key(spec, "test@example.com")
        
        start_time = time.time()
        
        tasks = [create_test_key(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert len(results) == concurrent_requests
        assert duration < 10.0  # Should complete within 10 seconds
        
        ops_per_second = concurrent_requests / duration
        print(f"Key creation rate: {ops_per_second:.2f} ops/sec")
```

## Custom Development

### Creating Custom Algorithms

```python
from keym.crypto.base import CryptographicAlgorithm
from keym.crypto.registry import register_algorithm

class CustomAESGCM(CryptographicAlgorithm):
    """Custom AES-GCM implementation with additional features"""
    
    algorithm_name = "CUSTOM_AES_GCM_256"
    key_size = 256
    
    async def generate_key(self) -> bytes:
        """Generate key material"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM.generate_key(bit_length=256)
    
    async def encrypt(self, key_material: bytes, plaintext: bytes, 
                     additional_data: bytes = None) -> bytes:
        """Encrypt with AES-GCM"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        import os
        
        aesgcm = AESGCM(key_material)
        nonce = os.urandom(12)  # 96-bit nonce
        
        ciphertext = aesgcm.encrypt(nonce, plaintext, additional_data)
        
        # Return nonce + ciphertext
        return nonce + ciphertext
    
    async def decrypt(self, key_material: bytes, ciphertext: bytes, 
                     additional_data: bytes = None) -> bytes:
        """Decrypt with AES-GCM"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        aesgcm = AESGCM(key_material)
        
        # Extract nonce and ciphertext
        nonce = ciphertext[:12]
        actual_ciphertext = ciphertext[12:]
        
        return aesgcm.decrypt(nonce, actual_ciphertext, additional_data)

# Register the custom algorithm
register_algorithm(CustomAESGCM)
```

### Event Handlers

```python
from keym.events import EventHandler, KeyEvent

class CustomEventHandler(EventHandler):
    """Custom event handler for key management events"""
    
    async def handle_key_created(self, event: KeyEvent) -> None:
        """Handle key creation events"""
        key_info = event.data
        
        # Log to custom audit system
        await self.log_to_audit_system({
            'event': 'key_created',
            'key_id': key_info.spec.id,
            'user_id': event.user_id,
            'timestamp': event.timestamp
        })
        
        # Send notification to security team for high-value keys
        if self.is_high_value_key(key_info):
            await self.notify_security_team(key_info)
    
    async def handle_key_accessed(self, event: KeyEvent) -> None:
        """Handle key access events"""
        # Implement access logging
        await self.log_key_access(event.data['key_id'], event.user_id)
        
        # Check for anomalous access patterns
        if await self.detect_anomaly(event.data['key_id'], event.user_id):
            await self.trigger_security_alert(event)
    
    def is_high_value_key(self, key_info) -> bool:
        """Determine if key is high-value"""
        return key_info.spec.algorithm in [
            KeyAlgorithm.RSA_4096, 
            KeyAlgorithm.ECDSA_P384
        ]
```

## Deployment and Configuration

### Environment Configuration

```python
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class KeyManagementConfig:
    """Configuration for key management service"""
    
    # Database configuration
    database_url: str = os.getenv('KEYM_DATABASE_URL', 'postgresql://localhost/keym')
    database_pool_size: int = int(os.getenv('KEYM_DB_POOL_SIZE', '20'))
    
    # Cache configuration
    cache_url: str = os.getenv('KEYM_CACHE_URL', 'redis://localhost:6379/0')
    cache_ttl: int = int(os.getenv('KEYM_CACHE_TTL', '3600'))
    
    # Security configuration
    encryption_key: str = os.getenv('KEYM_ENCRYPTION_KEY', '')
    hsm_enabled: bool = os.getenv('KEYM_HSM_ENABLED', 'false').lower() == 'true'
    hsm_library_path: str = os.getenv('KEYM_HSM_LIBRARY_PATH', '')
    
    # Performance configuration
    max_concurrent_operations: int = int(os.getenv('KEYM_MAX_CONCURRENT_OPS', '100'))
    operation_timeout: int = int(os.getenv('KEYM_OPERATION_TIMEOUT', '30'))
    
    # Monitoring configuration
    metrics_enabled: bool = os.getenv('KEYM_METRICS_ENABLED', 'true').lower() == 'true'
    log_level: str = os.getenv('KEYM_LOG_LEVEL', 'INFO')

def load_config() -> KeyManagementConfig:
    """Load configuration from environment"""
    return KeyManagementConfig()
```

### Docker Configuration

```dockerfile
# Dockerfile for Key Management Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 keym && chown -R keym:keym /app
USER keym

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Start application
CMD ["python", "-m", "keym.app"]
```

## Best Practices

### Error Handling

```python
from typing import Optional
import logging

class KeyManagementError(Exception):
    """Base exception for key management errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class KeyNotFoundError(KeyManagementError):
    """Key not found exception"""
    
    def __init__(self, key_id: str):
        super().__init__(
            f"Key not found: {key_id}",
            error_code="KEY_NOT_FOUND",
            details={"key_id": key_id}
        )

class CryptographicError(KeyManagementError):
    """Cryptographic operation error"""
    
    def __init__(self, operation: str, underlying_error: Exception = None):
        message = f"Cryptographic operation failed: {operation}"
        if underlying_error:
            message += f" ({str(underlying_error)})"
        
        super().__init__(
            message,
            error_code="CRYPTO_ERROR",
            details={"operation": operation}
        )

# Error handling in service methods
async def encrypt_data(self, key_id: str, data: bytes, user_id: str) -> bytes:
    """Encrypt data with proper error handling"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not key_id or not data:
            raise ValueError("key_id and data are required")
        
        # Retrieve key
        key_info = await self.retrieve_key(key_id, user_id)
        if not key_info:
            raise KeyNotFoundError(key_id)
        
        # Perform encryption
        encrypted_data = await self._perform_encryption(key_info, data)
        
        # Log successful operation
        logger.info(f"Data encrypted successfully with key {key_id}")
        
        return encrypted_data
        
    except KeyNotFoundError:
        logger.warning(f"Encryption failed - key not found: {key_id}")
        raise
    except Exception as e:
        logger.error(f"Encryption failed for key {key_id}: {e}")
        raise CryptographicError("encryption", e)
```

### Logging and Monitoring

```python
import logging
import time
from functools import wraps
from contextlib import asynccontextmanager

def log_performance(func):
    """Decorator to log performance metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger = logging.getLogger(func.__module__)
        
        try:
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper

@asynccontextmanager
async def audit_context(operation: str, user_id: str, resource_id: str = None):
    """Audit context manager"""
    audit_logger = logging.getLogger('keym.audit')
    start_time = time.time()
    
    audit_data = {
        'operation': operation,
        'user_id': user_id,
        'resource_id': resource_id,
        'start_time': start_time
    }
    
    try:
        yield audit_data
        audit_data['status'] = 'success'
    except Exception as e:
        audit_data['status'] = 'failed'
        audit_data['error'] = str(e)
        raise
    finally:
        audit_data['duration'] = time.time() - start_time
        audit_logger.info(f"AUDIT: {audit_data}")
```

---

This developer guide provides comprehensive information for working with the APG Key Management system. For additional support and examples, refer to the source code and test suites.

**Contact Information**
- Website: www.datacraft.co.ke
- Email: nyimbi@gmail.com
- Copyright: © 2025 Datacraft