# Quantum-Resistant Encryption Guide

## Overview

The APG Central Configuration capability implements state-of-the-art quantum-resistant encryption to protect sensitive configuration data against both classical and quantum computing attacks. This guide covers the encryption mechanisms, key management, and security best practices.

## Encryption Algorithms

### Post-Quantum Cryptography (PQC)

#### Kyber768 (Key Encapsulation)
- **Purpose**: Secure key exchange and encryption
- **Security Level**: NIST Level 3 (192-bit classical security)
- **Key Sizes**: Public key (1184 bytes), Private key (2400 bytes)
- **Ciphertext Size**: 1088 bytes
- **Quantum Resistance**: Secure against Shor's algorithm

#### Dilithium3 (Digital Signatures)
- **Purpose**: Message authentication and integrity
- **Security Level**: NIST Level 3 (192-bit classical security)  
- **Key Sizes**: Public key (1952 bytes), Private key (4000 bytes)
- **Signature Size**: ~3300 bytes (variable)
- **Quantum Resistance**: Secure against Grover's algorithm

### Symmetric Encryption

#### ChaCha20-Poly1305
- **Purpose**: Fast symmetric encryption for configuration data
- **Key Size**: 256 bits
- **Nonce Size**: 96 bits
- **Authentication**: Integrated AEAD (Authenticated Encryption with Associated Data)
- **Performance**: Optimized for modern processors

## Architecture

### Hybrid Encryption Scheme

```
Configuration Data → ChaCha20-Poly1305 → Encrypted Data
                           ↑
                    Symmetric Key
                           ↑
                    Kyber768 KEM → Encapsulated Key
                           ↑
                    Recipient Public Key
```

### Key Management Hierarchy

```
Master Key (Dilithium3)
├── Tenant Keys (Per-Tenant Isolation)
│   ├── Configuration Keys (Per-Config Type)
│   │   ├── Data Encryption Keys (ChaCha20)
│   │   └── Key Encapsulation (Kyber768)
│   └── Audit Keys (Tamper Protection)
└── System Keys (Infrastructure)
```

## Implementation Details

### Key Generation

```python
from capabilities.composition.central_configuration.security_engine import SecurityEngine

# Initialize security engine
security = SecurityEngine(tenant_id="mycompany")

# Generate new key pair
keypair = await security.generate_keypair("kyber768")
print(f"Public key: {keypair.public_key}")
print(f"Private key: {keypair.private_key}")

# Generate signing keys
signing_keys = await security.generate_signing_keys("dilithium3")
```

### Configuration Encryption

```python
# Encrypt sensitive configuration
encrypted_value = await security.encrypt_config_value(
    "database_password",
    "super_secret_password",
    tenant_id="mycompany"
)

# Decrypt configuration
decrypted_value = await security.decrypt_config_value(
    encrypted_value,
    tenant_id="mycompany"
)
```

### Digital Signatures

```python
# Sign configuration change
signature = await security.sign_config_change(
    config_key="app.database.host",
    old_value="localhost",
    new_value="prod-db.internal",
    user_id="admin",
    timestamp=datetime.utcnow()
)

# Verify signature
is_valid = await security.verify_config_signature(
    signature,
    config_data,
    public_key
)
```

## Key Management

### Key Storage

#### Hardware Security Modules (HSM)
```python
# Configure HSM integration
hsm_config = {
    "provider": "aws-cloudhsm",  # or "azure-keyvault", "pkcs11"
    "cluster_id": "cluster-abc123",
    "partition": "production",
    "credentials": {
        "username": "crypto_user",
        "password": "hsm_password"
    }
}
await security.configure_hsm(hsm_config)
```

#### File-Based Storage (Development)
```python
# Configure file-based key storage
file_storage_config = {
    "key_directory": "/etc/apg/keys",
    "key_format": "pem",
    "permissions": 0o600,
    "backup_enabled": True,
    "backup_directory": "/etc/apg/keys/backup"
}
await security.configure_file_storage(file_storage_config)
```

#### Cloud Key Management
```python
# AWS KMS integration
kms_config = {
    "provider": "aws-kms",
    "region": "us-east-1",
    "key_id": "arn:aws:kms:us-east-1:123456789:key/abc-def-123",
    "encryption_context": {
        "service": "apg-central-config",
        "environment": "production"
    }
}
await security.configure_cloud_kms(kms_config)
```

### Key Rotation

#### Automatic Rotation
```python
# Configure automatic key rotation
rotation_config = {
    "rotation_interval": 2592000,  # 30 days in seconds
    "overlap_period": 86400,       # 24 hours for key overlap
    "notification_before": 604800, # 7 days advance notice
    "auto_rotation": True,
    "backup_old_keys": True,
    "max_key_age": 7776000         # 90 days maximum
}
await security.configure_key_rotation(rotation_config)
```

#### Manual Rotation
```python
# Manually rotate keys
rotation_result = await security.rotate_keys(
    key_type="kyber768",
    tenant_id="mycompany",
    reason="scheduled_rotation",
    user_id="admin"
)

print(f"Rotation completed: {rotation_result.success}")
print(f"New key ID: {rotation_result.new_key_id}")
print(f"Old key archived: {rotation_result.old_key_archived}")
```

### Key Recovery

#### Key Escrow
```python
# Configure key escrow for recovery
escrow_config = {
    "enabled": True,
    "threshold": 3,  # Number of shares needed for recovery
    "total_shares": 5,  # Total number of shares created
    "trustees": [
        "admin1@company.com",
        "admin2@company.com", 
        "admin3@company.com",
        "security@company.com",
        "compliance@company.com"
    ],
    "recovery_policy": {
        "min_approval_time": 86400,  # 24 hours
        "approvers_required": 2,
        "audit_log": True
    }
}
await security.configure_key_escrow(escrow_config)
```

#### Disaster Recovery
```python
# Create disaster recovery backup
backup_info = await security.create_disaster_recovery_backup(
    tenant_id="mycompany",
    backup_location="s3://company-dr-backup/crypto-keys/",
    encryption_key="recovery_master_key",
    include_metadata=True
)

# Restore from disaster recovery backup
restore_result = await security.restore_from_disaster_backup(
    backup_location="s3://company-dr-backup/crypto-keys/backup-20250101.tar.enc",
    recovery_key="recovery_master_key",
    target_tenant="mycompany"
)
```

## Security Features

### Zero-Knowledge Architecture

#### Client-Side Encryption
```python
# Configuration encrypted before transmission
class ClientSideEncryption:
    def __init__(self, tenant_public_key):
        self.public_key = tenant_public_key
    
    async def encrypt_before_send(self, config_data):
        # Generate ephemeral key
        ephemeral_key = await generate_random_key()
        
        # Encrypt data with ephemeral key
        encrypted_data = await chacha20_encrypt(config_data, ephemeral_key)
        
        # Encapsulate ephemeral key with tenant's public key
        encapsulated_key = await kyber768_encapsulate(ephemeral_key, self.public_key)
        
        return {
            "encrypted_data": encrypted_data,
            "encapsulated_key": encapsulated_key
        }
```

#### Server-Side Key Isolation
```python
# Server never sees plaintext sensitive data
class ServerSideDecryption:
    def __init__(self, tenant_private_key):
        self.private_key = tenant_private_key
    
    async def decrypt_on_server(self, encrypted_payload):
        # Decapsulate the ephemeral key
        ephemeral_key = await kyber768_decapsulate(
            encrypted_payload["encapsulated_key"],
            self.private_key
        )
        
        # Decrypt the actual data
        plaintext = await chacha20_decrypt(
            encrypted_payload["encrypted_data"],
            ephemeral_key
        )
        
        # Immediately clear ephemeral key from memory
        secure_zero(ephemeral_key)
        
        return plaintext
```

### Perfect Forward Secrecy

```python
# Each session uses unique ephemeral keys
class ForwardSecureSession:
    def __init__(self):
        self.session_keys = {}
        self.key_lifetime = 3600  # 1 hour
    
    async def create_session(self, tenant_id):
        # Generate ephemeral session key
        session_key = await generate_random_key()
        session_id = await generate_session_id()
        
        # Store with expiration
        self.session_keys[session_id] = {
            "key": session_key,
            "tenant_id": tenant_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.key_lifetime
        }
        
        return session_id
    
    async def cleanup_expired_sessions(self):
        current_time = time.time()
        expired_sessions = [
            sid for sid, info in self.session_keys.items()
            if info["expires_at"] < current_time
        ]
        
        for session_id in expired_sessions:
            # Securely clear key from memory
            secure_zero(self.session_keys[session_id]["key"])
            del self.session_keys[session_id]
```

### Memory Protection

```python
# Secure memory handling for sensitive data
import mlock
import ctypes

class SecureMemory:
    def __init__(self, size):
        self.size = size
        self.buffer = mlock.mlocked_memory(size)
    
    def write_secret(self, data):
        """Write sensitive data to locked memory"""
        if len(data) > self.size:
            raise ValueError("Data too large for secure buffer")
        
        # Copy data to locked memory
        ctypes.memmove(self.buffer, data, len(data))
        
        # Clear source data
        if hasattr(data, 'encode'):
            data = '\x00' * len(data)
    
    def read_secret(self):
        """Read data from secure memory"""
        return bytes(self.buffer)
    
    def clear(self):
        """Securely clear memory"""
        ctypes.memset(self.buffer, 0, self.size)
    
    def __del__(self):
        """Ensure memory is cleared on destruction"""
        self.clear()
```

## Performance Optimization

### Caching Strategies

#### Encrypted Cache
```python
# Cache encrypted values to avoid re-encryption
class EncryptedCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    async def get_cached_encrypted(self, config_key, tenant_id):
        cache_key = f"encrypted:{tenant_id}:{config_key}"
        encrypted_data = await self.redis.get(cache_key)
        
        if encrypted_data:
            return json.loads(encrypted_data)
        return None
    
    async def cache_encrypted(self, config_key, tenant_id, encrypted_data):
        cache_key = f"encrypted:{tenant_id}:{config_key}"
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(encrypted_data)
        )
```

#### Key Caching
```python
# Cache decryption keys with secure expiration
class KeyCache:
    def __init__(self):
        self.cache = {}
        self.max_age = 300  # 5 minutes
    
    async def get_cached_key(self, key_id):
        if key_id in self.cache:
            key_info = self.cache[key_id]
            if time.time() - key_info["cached_at"] < self.max_age:
                return key_info["key"]
            else:
                # Key expired, remove from cache
                secure_zero(key_info["key"])
                del self.cache[key_id]
        return None
    
    async def cache_key(self, key_id, key_data):
        self.cache[key_id] = {
            "key": key_data,
            "cached_at": time.time()
        }
```

### Batch Operations

```python
# Batch encrypt/decrypt for improved performance
class BatchCrypto:
    async def batch_encrypt(self, config_items, tenant_id):
        """Encrypt multiple configurations efficiently"""
        # Generate single ephemeral key for the batch
        ephemeral_key = await generate_random_key()
        
        encrypted_items = []
        for item in config_items:
            encrypted_data = await chacha20_encrypt(
                item["value"], 
                ephemeral_key
            )
            encrypted_items.append({
                "key": item["key"],
                "encrypted_value": encrypted_data
            })
        
        # Encapsulate ephemeral key once for all items
        tenant_public_key = await get_tenant_public_key(tenant_id)
        encapsulated_key = await kyber768_encapsulate(
            ephemeral_key, 
            tenant_public_key
        )
        
        return {
            "items": encrypted_items,
            "encapsulated_key": encapsulated_key
        }
```

## Compliance and Auditing

### Cryptographic Audit Trail

```python
# Log all cryptographic operations
class CryptoAuditLogger:
    def __init__(self, audit_service):
        self.audit = audit_service
    
    async def log_encryption(self, config_key, tenant_id, user_id, algorithm):
        await self.audit.log_event({
            "event_type": "encryption",
            "config_key": config_key,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "algorithm": algorithm,
            "timestamp": datetime.utcnow().isoformat(),
            "key_id": await self.get_current_key_id(tenant_id),
            "success": True
        })
    
    async def log_decryption(self, config_key, tenant_id, user_id, success):
        await self.audit.log_event({
            "event_type": "decryption",
            "config_key": config_key,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "access_pattern": await self.analyze_access_pattern(user_id)
        })
```

### Compliance Reporting

```python
# Generate compliance reports
class ComplianceReporter:
    async def generate_crypto_compliance_report(self, tenant_id, period_days=30):
        """Generate cryptographic compliance report"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        report = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "encryption_stats": await self.get_encryption_stats(tenant_id, start_date, end_date),
            "key_rotation_events": await self.get_key_rotation_events(tenant_id, start_date, end_date),
            "access_patterns": await self.get_access_patterns(tenant_id, start_date, end_date),
            "security_incidents": await self.get_security_incidents(tenant_id, start_date, end_date),
            "compliance_status": await self.check_compliance_status(tenant_id)
        }
        
        return report
```

## Migration from Classical Cryptography

### Migration Strategy

```python
# Gradual migration from classical to quantum-resistant crypto
class CryptoMigration:
    def __init__(self, security_engine):
        self.security = security_engine
        self.migration_status = {}
    
    async def start_migration(self, tenant_id):
        """Start migration to quantum-resistant cryptography"""
        # 1. Generate new quantum-resistant keys
        pqc_keys = await self.security.generate_pqc_keypair(tenant_id)
        
        # 2. Mark tenant for migration
        self.migration_status[tenant_id] = {
            "status": "in_progress",
            "started_at": datetime.utcnow(),
            "classical_key_id": await self.get_current_key_id(tenant_id),
            "pqc_key_id": pqc_keys.key_id,
            "migrated_configs": []
        }
        
        # 3. Begin incremental migration
        await self.migrate_configurations_incrementally(tenant_id)
    
    async def migrate_configurations_incrementally(self, tenant_id):
        """Migrate configurations in batches"""
        batch_size = 100
        offset = 0
        
        while True:
            # Get batch of configurations
            configs = await self.get_tenant_configs(
                tenant_id, 
                limit=batch_size, 
                offset=offset
            )
            
            if not configs:
                break  # No more configurations to migrate
            
            # Re-encrypt each configuration with new keys
            for config in configs:
                try:
                    # Decrypt with old key
                    plaintext = await self.security.decrypt_with_classical_key(
                        config.encrypted_value,
                        self.migration_status[tenant_id]["classical_key_id"]
                    )
                    
                    # Encrypt with new quantum-resistant key
                    new_encrypted = await self.security.encrypt_with_pqc_key(
                        plaintext,
                        self.migration_status[tenant_id]["pqc_key_id"]
                    )
                    
                    # Update configuration
                    await self.update_config_encryption(
                        config.key,
                        new_encrypted,
                        tenant_id
                    )
                    
                    # Track migration progress
                    self.migration_status[tenant_id]["migrated_configs"].append(config.key)
                    
                except Exception as e:
                    logger.error(f"Migration failed for {config.key}: {e}")
                    # Continue with other configurations
            
            offset += batch_size
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        # Mark migration as complete
        self.migration_status[tenant_id]["status"] = "completed"
        self.migration_status[tenant_id]["completed_at"] = datetime.utcnow()
```

### Dual-Mode Operation

```python
# Support both classical and quantum-resistant encryption during transition
class DualModeCrypto:
    def __init__(self, security_engine):
        self.security = security_engine
    
    async def encrypt_dual_mode(self, data, tenant_id):
        """Encrypt data with both classical and quantum-resistant algorithms"""
        # Check tenant's migration status
        migration_status = await self.get_migration_status(tenant_id)
        
        if migration_status == "not_started":
            # Use classical encryption only
            return await self.security.encrypt_classical(data, tenant_id)
        
        elif migration_status == "in_progress":
            # Use both algorithms for compatibility
            classical_encrypted = await self.security.encrypt_classical(data, tenant_id)
            pqc_encrypted = await self.security.encrypt_pqc(data, tenant_id)
            
            return {
                "classical": classical_encrypted,
                "pqc": pqc_encrypted,
                "dual_mode": True
            }
        
        else:  # migration_status == "completed"
            # Use quantum-resistant encryption only
            return await self.security.encrypt_pqc(data, tenant_id)
    
    async def decrypt_dual_mode(self, encrypted_data, tenant_id):
        """Decrypt data handling both formats"""
        if isinstance(encrypted_data, dict) and encrypted_data.get("dual_mode"):
            # Try PQC first, fall back to classical if needed
            try:
                return await self.security.decrypt_pqc(
                    encrypted_data["pqc"], 
                    tenant_id
                )
            except DecryptionError:
                return await self.security.decrypt_classical(
                    encrypted_data["classical"], 
                    tenant_id
                )
        else:
            # Determine encryption type and decrypt accordingly
            crypto_type = await self.detect_encryption_type(encrypted_data)
            
            if crypto_type == "pqc":
                return await self.security.decrypt_pqc(encrypted_data, tenant_id)
            else:
                return await self.security.decrypt_classical(encrypted_data, tenant_id)
```

## Best Practices

### Key Management
1. **Use HSMs for production environments**
2. **Implement regular key rotation (30-90 days)**
3. **Maintain secure key escrow for disaster recovery**
4. **Monitor key usage and access patterns**
5. **Use strong random number generation**

### Performance
1. **Cache encrypted values when possible**
2. **Use batch operations for multiple configurations**
3. **Implement connection pooling for HSM access**
4. **Monitor encryption/decryption latency**
5. **Use async operations to avoid blocking**

### Security
1. **Never log plaintext sensitive data**
2. **Implement secure memory handling**
3. **Use constant-time operations to prevent timing attacks**
4. **Validate all inputs before cryptographic operations**
5. **Implement proper session management**

### Monitoring
1. **Track key rotation events**
2. **Monitor failed decryption attempts**
3. **Alert on unusual access patterns**
4. **Log all cryptographic operations**
5. **Generate regular compliance reports**

## Next Steps

- Learn about [Authentication and Authorization](authentication.md)
- Set up [Audit Logging](audit-logging.md)
- Review [Security Best Practices](security-best-practices.md)
- Configure [Real-time Monitoring](../troubleshooting/monitoring.md)