# APG Key Management - User Guide

## Overview

The APG Key Management capability provides enterprise-grade cryptographic key lifecycle management with intelligent automation, multi-cloud federation, and quantum-safe cryptography. This guide covers how to use the key management system effectively within the APG platform.

## Quick Start

### 1. Initial Setup

```python
from keym.service import KeyManagementService
from keym.models import create_key_spec_async, KeyAlgorithm, KeyUsage

# Initialize service
service = KeyManagementService()
await service.initialize("your-tenant-id")
```

### 2. Create Your First Key

```python
# Create key specification
spec = await create_key_spec_async(
    tenant_id="your-tenant",
    algorithm=KeyAlgorithm.AES_256,
    usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
    name="My First Key",
    created_by="user@company.com"
)

# Create the key
key = await service.create_key(spec, "user@company.com")
print(f"Created key: {key.spec.id}")
```

### 3. Encrypt and Decrypt Data

```python
# Encrypt sensitive data
data = b"Confidential information"
encrypted_data = await service.encrypt_data(
    key.spec.id, 
    data, 
    "user@company.com"
)

# Decrypt data
decrypted_data = await service.decrypt_data(
    key.spec.id, 
    encrypted_data, 
    "user@company.com"
)
```

## Key Management Operations

### Creating Keys

The system supports multiple cryptographic algorithms:

```python
# Symmetric encryption keys
aes_spec = await create_key_spec_async(
    tenant_id="tenant",
    algorithm=KeyAlgorithm.AES_256,
    usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
    name="Data Encryption Key"
)

# Asymmetric signing keys
rsa_spec = await create_key_spec_async(
    tenant_id="tenant",
    algorithm=KeyAlgorithm.RSA_2048,
    usage=[KeyUsage.SIGN, KeyUsage.VERIFY],
    name="Document Signing Key"
)

# Elliptic curve keys
ecdsa_spec = await create_key_spec_async(
    tenant_id="tenant",
    algorithm=KeyAlgorithm.ECDSA_P256,
    usage=[KeyUsage.SIGN, KeyUsage.VERIFY],
    name="ECDSA Signing Key"
)
```

### Key Lifecycle Management

```python
# Retrieve existing key
key = await service.retrieve_key(key_id, "user@company.com")

# Rotate key for enhanced security
new_key = await service.rotate_key(key_id, "user@company.com")

# Archive key when no longer needed
await service.archive_key(key_id, "user@company.com")

# Secure deletion
await service.delete_key(key_id, "user@company.com", secure_delete=True)
```

### Batch Operations

```python
# Create multiple keys efficiently
specs = [
    await create_key_spec_async(
        tenant_id="tenant",
        algorithm=KeyAlgorithm.AES_256,
        usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
        name=f"Batch Key {i}"
    )
    for i in range(10)
]

keys = await service.create_keys_batch(specs, "user@company.com")
```

## Web Dashboard Usage

Access the APG Key Management dashboard at `/keym` in your APG installation.

### Dashboard Features

1. **Key Overview**: View all keys with status, algorithm, and usage
2. **Key Creation Wizard**: Step-by-step key creation with templates
3. **Security Analytics**: Real-time security metrics and anomaly detection
4. **Policy Management**: Configure automated key policies
5. **Audit Logs**: Comprehensive key operation audit trail

### Navigation

- **Keys**: Main key listing and management
- **Policies**: Automated policy configuration
- **Analytics**: Security analytics and reporting
- **Settings**: Tenant configuration and preferences

## Advanced Features

### Multi-Cloud Key Federation

Seamlessly manage keys across multiple cloud providers:

```python
from keym.multi_cloud_federation import CloudKeyFederation

federation = CloudKeyFederation(service)

# Federate key across AWS, Azure, and GCP
await federation.federate_key(
    key_id="key-123",
    target_clouds=["aws", "azure", "gcp"],
    user_id="user@company.com"
)

# Retrieve federated key from any cloud
key_data = await federation.retrieve_federated_key(
    key_id="key-123",
    cloud_provider="aws",
    user_id="user@company.com"
)
```

### Hardware Security Module (HSM) Integration

Leverage HSM for enhanced security:

```python
from keym.hsm_integration import HSMManager

hsm_manager = HSMManager(service)

# Create key in HSM
hsm_key = await hsm_manager.create_hsm_key(
    hsm_id="primary-hsm",
    key_spec=spec,
    user_id="user@company.com"
)

# Perform hardware-backed operations
signature = await hsm_manager.hsm_sign_data(
    hsm_id="primary-hsm",
    key_id=hsm_key.key_id,
    data=b"Document to sign",
    user_id="user@company.com"
)
```

### Intelligent Security Features

#### Anomaly Detection

The system automatically detects unusual key usage patterns:

```python
from keym.security_intelligence import SecurityIntelligence

security = SecurityIntelligence(service)

# Get security analysis for a key
analysis = await security.analyze_key_usage(
    key_id="key-123",
    time_range_days=30
)

print(f"Risk Score: {analysis['risk_score']}")
print(f"Anomalies: {len(analysis['anomalies'])}")
```

#### Automated Policies

Configure intelligent policies for key management:

```python
from keym.policy_automation import PolicyEngine

policy_engine = PolicyEngine(service)

# Create automatic key rotation policy
policy = await policy_engine.create_policy(
    name="Monthly Key Rotation",
    type="automatic_rotation",
    schedule="monthly",
    target_keys={"algorithm": "AES_256"},
    user_id="admin@company.com"
)
```

### Quantum-Safe Cryptography

Prepare for the quantum computing era:

```python
from keym.quantum_safe import QuantumSafeCrypto

quantum_crypto = QuantumSafeCrypto(service)

# Create quantum-safe key
pqc_spec = await create_key_spec_async(
    tenant_id="tenant",
    algorithm=KeyAlgorithm.KYBER_1024,  # Post-quantum algorithm
    usage=[KeyUsage.KEY_EXCHANGE],
    name="Quantum-Safe Key Exchange"
)

pqc_key = await quantum_crypto.create_quantum_safe_key(
    spec=pqc_spec,
    user_id="user@company.com"
)
```

## Integration Patterns

### APG Capability Composition

Use Key Management with other APG capabilities:

```python
# Integration with APG Auth/RBAC
from capabilities.auth_rbac.service import AuthRBACService

auth_service = AuthRBACService()

# Create key with RBAC authorization
if await auth_service.check_permission(user_id, "keym:create_key"):
    key = await service.create_key(spec, user_id)
```

### Event-Driven Integration

Subscribe to key management events:

```python
from keym.events import KeyEventSubscriber

subscriber = KeyEventSubscriber()

@subscriber.on_key_created
async def handle_key_creation(event):
    print(f"Key created: {event.key_id}")
    # Trigger downstream processes

@subscriber.on_security_anomaly
async def handle_security_alert(event):
    print(f"Security anomaly detected: {event.description}")
    # Send alert to security team
```

## Performance Optimization

### Connection Pooling

Configure optimal connection settings:

```python
# Initialize with custom pool settings
service = KeyManagementService({
    'db_pool_min_size': 10,
    'db_pool_max_size': 100,
    'connection_timeout': 30
})
```

### Caching Configuration

Optimize performance with intelligent caching:

```python
# Enable key metadata caching
service.configure_caching({
    'enable_key_cache': True,
    'cache_size': 10000,
    'cache_ttl': 3600
})
```

## Security Best Practices

### Access Control

1. **Least Privilege**: Grant minimal required permissions
2. **Regular Audits**: Review key access patterns regularly
3. **Strong Authentication**: Use multi-factor authentication

### Key Hygiene

1. **Regular Rotation**: Rotate keys according to policy
2. **Secure Storage**: Use HSM for high-value keys
3. **Proper Deletion**: Always use secure deletion

### Monitoring

1. **Enable Logging**: Capture all key operations
2. **Set Alerts**: Monitor for anomalous behavior
3. **Regular Reviews**: Conduct periodic security reviews

## Troubleshooting

### Common Issues

#### Key Creation Fails

```python
try:
    key = await service.create_key(spec, user_id)
except ValidationError as e:
    print(f"Validation error: {e}")
    # Check key specification parameters
except AuthorizationError as e:
    print(f"Authorization error: {e}")
    # Verify user permissions
```

#### Performance Issues

```python
# Check system performance
from keym.performance import create_performance_tester

tester = await create_performance_tester(service)
metrics = await tester.run_load_test(load_config)

print(f"Operations per second: {metrics.operations_per_second}")
print(f"Average latency: {metrics.latency_stats['mean']:.3f}s")
```

#### HSM Connectivity

```python
# Test HSM connection
status = await hsm_manager.get_hsm_status("primary-hsm")
if status['status'] != 'online':
    print(f"HSM issue: {status['error_message']}")
```

### Support Resources

1. **Documentation**: Complete API reference at `/docs`
2. **Logs**: Check application logs for detailed errors
3. **Monitoring**: Use built-in performance dashboard
4. **Support**: Contact Datacraft support team

## API Reference Quick Links

- **Service API**: Complete service method documentation
- **Models**: Data model schemas and validation rules
- **Events**: Event types and subscription patterns
- **Errors**: Error codes and handling strategies

---

**Contact Information**
- Website: www.datacraft.co.ke
- Email: nyimbi@gmail.com
- Copyright: © 2025 Datacraft