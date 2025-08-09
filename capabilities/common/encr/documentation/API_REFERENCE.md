# APG Encryption Services - API Reference

Complete API documentation for APG Encryption Services with quantum-safe cryptography.

## Base URL
```
https://api.datacraft.co.ke/api/v1
```

## Authentication

All API requests require authentication using Bearer tokens and tenant identification:

```http
Authorization: Bearer <your-api-key>
X-Tenant-ID: <your-tenant-id>
Content-Type: application/json
User-Agent: <your-client/version>
```

### API Key Management
- API keys are tenant-specific and should be kept secure
- Keys can be rotated via the management console
- Invalid keys return `401 Unauthorized`
- Rate limits are applied per API key

## Rate Limits

| Endpoint Category | Rate Limit | Burst Limit |
|-------------------|------------|-------------|
| Encryption Operations | 1000 req/min | 100 req/10s |
| Key Management | 100 req/min | 20 req/10s |
| Metadata Operations | 500 req/min | 50 req/10s |
| Administrative | 50 req/min | 10 req/10s |

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1673875200
X-RateLimit-Retry-After: 60
```

## Standard Response Format

### Success Response
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "request_id": "req_01234567-89ab-cdef-0123-456789abcdef",
    "timestamp": "2025-01-15T10:30:00Z",
    "processing_time_ms": 150,
    "api_version": "v1"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ENCRYPTION_FAILED",
    "message": "Encryption operation failed",
    "details": {
      "reason": "Invalid algorithm specified",
      "supported_algorithms": [
        "CRYSTALS-Kyber-512",
        "CRYSTALS-Kyber-768", 
        "CRYSTALS-Kyber-1024"
      ]
    },
    "request_id": "req_01234567-89ab-cdef-0123-456789abcdef",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

## Encryption Operations

### POST /encrypt

Encrypt data using quantum-safe post-quantum cryptography algorithms.

#### Request Body
```json
{
  "data": "48656c6c6f2c20576f726c6421",
  "algorithm": "CRYSTALS-Kyber-1024",
  "metadata": {
    "source": "api_client",
    "data_type": "user_content",
    "classification": "sensitive",
    "tags": ["user-data", "encrypted"]
  },
  "encryption_context": {
    "compliance_requirements": ["GDPR", "HIPAA"],
    "geographic_location": "EU",
    "retention_period": "7_years",
    "access_control": {
      "required_roles": ["data_processor"],
      "require_mfa": true
    }
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | string | Yes | Hex-encoded data to encrypt |
| `algorithm` | string | No | Encryption algorithm (default: CRYSTALS-Kyber-1024) |
| `metadata` | object | No | Additional metadata to store with encrypted data |
| `encryption_context` | object | No | Context for policy-based encryption |

#### Supported Algorithms

| Algorithm | Security Level | Key Size | Public Key | Ciphertext Overhead |
|-----------|----------------|----------|------------|-------------------|
| `CRYSTALS-Kyber-512` | NIST Level 1 | 1632 bytes | 800 bytes | ~800 bytes |
| `CRYSTALS-Kyber-768` | NIST Level 3 | 2400 bytes | 1184 bytes | ~1088 bytes |
| `CRYSTALS-Kyber-1024` | NIST Level 5 | 3168 bytes | 1568 bytes | ~1568 bytes |

#### Response
```json
{
  "success": true,
  "data": {
    "encrypted_data": "a1b2c3d4e5f6789...",
    "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "algorithm": "CRYSTALS-Kyber-1024",
    "security_level": "NIST_LEVEL_5",
    "data_size_bytes": 1024,
    "encrypted_size_bytes": 2592,
    "compression_applied": false,
    "integrity_hash": "sha256:abcd1234...",
    "encryption_timestamp": "2025-01-15T10:30:00Z",
    "expiry_timestamp": null,
    "compliance_validated": true,
    "compliance_frameworks": ["GDPR", "HIPAA"],
    "policy_applied": {
      "policy_id": "policy_healthcare_eu",
      "algorithm_selected_reason": "high_sensitivity_data",
      "retention_enforcement": true
    }
  },
  "metadata": {
    "request_id": "req_encrypt_001",
    "processing_time_ms": 125,
    "performance_metrics": {
      "key_generation_ms": 45,
      "encryption_ms": 80,
      "total_ms": 125
    }
  }
}
```

### POST /decrypt

Decrypt previously encrypted data using the associated key.

#### Request Body
```json
{
  "encrypted_data": "a1b2c3d4e5f6789...",
  "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
  "decryption_context": {
    "user_id": "user_12345",
    "purpose": "data_access",
    "access_justification": "User requested their personal data"
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `encrypted_data` | string | Yes | Encrypted data to decrypt |
| `key_id` | string | Yes | Key identifier used for encryption |
| `decryption_context` | object | No | Context for access control and auditing |

#### Response
```json
{
  "success": true,
  "data": {
    "decrypted_data": "48656c6c6f2c20576f726c6421",
    "original_size_bytes": 1024,
    "algorithm_used": "CRYSTALS-Kyber-1024",
    "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "integrity_verified": true,
    "decryption_timestamp": "2025-01-15T10:35:00Z",
    "access_granted": true,
    "audit_logged": true,
    "compliance_check_passed": true
  },
  "metadata": {
    "request_id": "req_decrypt_001",
    "processing_time_ms": 95,
    "performance_metrics": {
      "key_retrieval_ms": 15,
      "decryption_ms": 80,
      "total_ms": 95
    }
  }
}
```

### POST /encrypt/batch

Encrypt multiple data items in a single request for improved efficiency.

#### Request Body
```json
{
  "items": [
    {
      "id": "item_1",
      "data": "48656c6c6f2c20576f726c6421",
      "algorithm": "CRYSTALS-Kyber-1024"
    },
    {
      "id": "item_2", 
      "data": "476f6f646279652c20576f726c6421",
      "algorithm": "CRYSTALS-Kyber-768"
    }
  ],
  "batch_options": {
    "parallel_processing": true,
    "max_concurrent": 10,
    "fail_on_error": false
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "id": "item_1",
        "success": true,
        "encrypted_data": "a1b2c3d4...",
        "key_id": "key_001"
      },
      {
        "id": "item_2",
        "success": true,
        "encrypted_data": "e5f6g7h8...",
        "key_id": "key_002"
      }
    ],
    "summary": {
      "total_items": 2,
      "successful": 2,
      "failed": 0,
      "processing_time_ms": 180
    }
  }
}
```

## Key Management

### POST /keys/generate

Generate a new quantum-safe cryptographic key pair.

#### Request Body
```json
{
  "algorithm": "CRYSTALS-Kyber-1024",
  "metadata": {
    "purpose": "data_encryption",
    "department": "healthcare",
    "project": "patient_records_system"
  },
  "key_policy": {
    "rotation_interval_days": 90,
    "auto_rotation": true,
    "expiry_date": "2026-01-15T10:30:00Z",
    "usage_limits": {
      "max_encryptions": 1000000,
      "max_decryptions": 1000000
    }
  },
  "access_control": {
    "allowed_users": ["user_123", "user_456"],
    "allowed_roles": ["data_processor", "admin"],
    "require_mfa": true
  }
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `algorithm` | string | No | Key generation algorithm (default: CRYSTALS-Kyber-1024) |
| `metadata` | object | No | Key metadata for organization and tracking |
| `key_policy` | object | No | Automated key management policies |
| `access_control` | object | No | Access control settings for the key |

#### Response
```json
{
  "success": true,
  "data": {
    "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "algorithm": "CRYSTALS-Kyber-1024",
    "security_level": "NIST_LEVEL_5",
    "key_size_bytes": 3168,
    "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
    "key_fingerprint": "SHA256:abc123def456...",
    "created_at": "2025-01-15T10:30:00Z",
    "expires_at": "2026-01-15T10:30:00Z",
    "status": "active",
    "usage_count": 0,
    "rotation_policy": {
      "next_rotation": "2025-04-15T10:30:00Z",
      "auto_rotation_enabled": true
    }
  }
}
```

### GET /keys

List all keys for the tenant with filtering and pagination support.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | `active` | Filter by key status (active, inactive, expired) |
| `algorithm` | string | all | Filter by algorithm |
| `limit` | integer | 50 | Maximum number of keys to return |
| `offset` | integer | 0 | Number of keys to skip |
| `sort_by` | string | `created_at` | Sort field (created_at, expires_at, algorithm) |
| `sort_order` | string | `desc` | Sort order (asc, desc) |

#### Response
```json
{
  "success": true,
  "data": {
    "keys": [
      {
        "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
        "algorithm": "CRYSTALS-Kyber-1024",
        "security_level": "NIST_LEVEL_5",
        "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
        "status": "active",
        "created_at": "2025-01-15T10:30:00Z",
        "expires_at": "2026-01-15T10:30:00Z",
        "usage_count": 1523,
        "last_used": "2025-01-15T14:22:00Z"
      }
    ],
    "pagination": {
      "total": 125,
      "limit": 50,
      "offset": 0,
      "has_more": true
    }
  }
}
```

### GET /keys/{key_id}

Get detailed information about a specific key.

#### Response
```json
{
  "success": true,
  "data": {
    "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "algorithm": "CRYSTALS-Kyber-1024",
    "security_level": "NIST_LEVEL_5",
    "key_size_bytes": 3168,
    "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----",
    "key_fingerprint": "SHA256:abc123def456...",
    "status": "active",
    "created_at": "2025-01-15T10:30:00Z",
    "expires_at": "2026-01-15T10:30:00Z",
    "last_rotation": "2024-10-15T10:30:00Z",
    "next_rotation": "2025-04-15T10:30:00Z",
    "usage_statistics": {
      "total_encryptions": 1523,
      "total_decryptions": 1498,
      "last_used": "2025-01-15T14:22:00Z",
      "bytes_encrypted": 15680000,
      "average_operation_time_ms": 125
    },
    "compliance": {
      "frameworks": ["GDPR", "HIPAA"],
      "last_audit": "2024-12-01T00:00:00Z",
      "next_audit": "2025-03-01T00:00:00Z"
    }
  }
}
```

### POST /keys/{key_id}/rotate

Manually rotate a key to generate a new version while maintaining backward compatibility.

#### Request Body
```json
{
  "rotation_reason": "scheduled_rotation",
  "migration_policy": {
    "auto_migrate_existing": true,
    "deprecation_period_days": 30,
    "notification_recipients": ["admin@company.com"]
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "old_key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "new_key_id": "key_76543210-cdef-89ab-4567-0123456789ab",
    "rotation_timestamp": "2025-01-15T15:00:00Z",
    "migration_status": "in_progress",
    "migration_progress": {
      "total_items": 1523,
      "migrated_items": 0,
      "estimated_completion": "2025-01-15T16:30:00Z"
    }
  }
}
```

### DELETE /keys/{key_id}

Deactivate or permanently delete a key (depending on compliance requirements).

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `permanent` | boolean | false | Permanently delete key (requires compliance check) |
| `force` | boolean | false | Force deletion even with active encryptions |

#### Response
```json
{
  "success": true,
  "data": {
    "key_id": "key_01234567-89ab-cdef-0123-456789abcdef",
    "status": "deactivated",
    "deactivated_at": "2025-01-15T15:30:00Z",
    "permanent_deletion_date": "2025-04-15T15:30:00Z",
    "compliance_retention": {
      "required_retention_days": 90,
      "can_permanently_delete": false,
      "reason": "GDPR retention requirements"
    }
  }
}
```

## Advanced Operations

### POST /homomorphic/encrypt

Encrypt data for homomorphic computation (allows computation on encrypted data).

#### Request Body
```json
{
  "data": [42, 27, 15],
  "scheme": "BGV",
  "parameters": {
    "polynomial_degree": 8192,
    "coefficient_modulus": [60, 40, 40, 60],
    "plain_modulus": 1024
  },
  "computation_context": {
    "expected_operations": ["addition", "multiplication"],
    "depth_budget": 5
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "ciphertext_id": "ct_01234567-89ab-cdef-0123-456789abcdef",
    "scheme": "BGV",
    "encrypted_values": ["enc_42", "enc_27", "enc_15"],
    "public_key_id": "key_homomorphic_001",
    "context_id": "ctx_computation_001",
    "noise_budget": 45,
    "max_operations_remaining": 12
  }
}
```

### POST /homomorphic/compute

Perform computation on homomorphically encrypted data.

#### Request Body
```json
{
  "operation": "add",
  "operands": [
    "ct_01234567-89ab-cdef-0123-456789abcdef",
    "ct_76543210-cdef-89ab-4567-0123456789ab"
  ],
  "context_id": "ctx_computation_001"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "result_ciphertext_id": "ct_result_001",
    "operation_performed": "addition",
    "noise_budget_remaining": 42,
    "computation_time_ms": 85,
    "can_decrypt": true
  }
}
```

### POST /mpc/setup

Set up a secure multi-party computation session.

#### Request Body
```json
{
  "computation_id": "mpc_private_sum_001",
  "participants": [
    {
      "party_id": "company_a",
      "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
    },
    {
      "party_id": "company_b", 
      "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
    }
  ],
  "protocol": "BGW",
  "security_threshold": 1,
  "computation_type": "private_sum"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "computation_id": "mpc_private_sum_001",
    "session_id": "session_01234567-89ab-cdef",
    "status": "setup_complete",
    "participants": ["company_a", "company_b"],
    "protocol": "BGW", 
    "security_parameters": {
      "threshold": 1,
      "modulus": "prime_256bit",
      "field_size": 256
    },
    "ready_for_input": true
  }
}
```

## Policy and Compliance

### POST /policies

Create an automated cryptographic policy.

#### Request Body
```json
{
  "policy_name": "Healthcare_EU_Patient_Data",
  "description": "Encryption policy for EU healthcare patient data",
  "scope": {
    "data_types": ["pii", "phi"],
    "geographic_regions": ["EU"],
    "departments": ["healthcare", "medical_research"]
  },
  "encryption_rules": {
    "minimum_algorithm": "CRYSTALS-Kyber-768",
    "prefer_algorithm": "CRYSTALS-Kyber-1024",
    "key_rotation_interval": "quarterly",
    "require_hsm": true
  },
  "compliance_requirements": {
    "frameworks": ["GDPR", "HIPAA"],
    "audit_frequency": "monthly",
    "retention_period": "7_years",
    "data_residency": "EU"
  },
  "access_controls": {
    "require_mfa": true,
    "allowed_roles": ["physician", "nurse", "admin"],
    "ip_whitelist": ["10.0.0.0/8"],
    "time_restrictions": {
      "allowed_hours": "09:00-17:00",
      "timezone": "Europe/Berlin"
    }
  }
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "policy_id": "policy_01234567-89ab-cdef-0123-456789abcdef",
    "policy_name": "Healthcare_EU_Patient_Data",
    "status": "active",
    "created_at": "2025-01-15T10:30:00Z",
    "last_updated": "2025-01-15T10:30:00Z",
    "compliance_validated": true,
    "affected_keys": 45,
    "affected_users": 123,
    "automatic_enforcement": true
  }
}
```

### GET /compliance/audit

Generate compliance audit report for specified time period.

#### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `framework` | string | Yes | Compliance framework (GDPR, HIPAA, PCI_DSS, SOX) |
| `start_date` | string | Yes | Start date (ISO 8601 format) |
| `end_date` | string | Yes | End date (ISO 8601 format) |
| `format` | string | No | Report format (json, pdf, csv) |

#### Response
```json
{
  "success": true,
  "data": {
    "audit_report": {
      "framework": "GDPR",
      "period": {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-31T23:59:59Z"
      },
      "summary": {
        "total_operations": 15420,
        "compliant_operations": 15420,
        "violations": 0,
        "compliance_rate": 100.0
      },
      "operations_breakdown": {
        "encryptions": 7850,
        "decryptions": 7570,
        "key_generations": 25,
        "key_rotations": 5
      },
      "data_subject_requests": {
        "access_requests": 12,
        "erasure_requests": 3,
        "portability_requests": 5,
        "average_response_time_hours": 18
      },
      "policy_enforcement": {
        "policies_applied": 8,
        "automatic_enforcement": 100.0,
        "manual_overrides": 0
      }
    },
    "download_url": "https://api.datacraft.co.ke/reports/audit_report_20250115.pdf",
    "expires_at": "2025-01-22T10:30:00Z"
  }
}
```

## Analytics and Monitoring

### GET /analytics/usage

Get detailed usage analytics for the tenant.

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `last_30_days` | Time period (last_24_hours, last_7_days, last_30_days, custom) |
| `granularity` | string | `daily` | Data granularity (hourly, daily, weekly) |
| `metrics` | array | all | Specific metrics to include |
| `start_date` | string | - | Custom period start date |
| `end_date` | string | - | Custom period end date |

#### Response
```json
{
  "success": true,
  "data": {
    "period": {
      "start": "2024-12-16T00:00:00Z",
      "end": "2025-01-15T23:59:59Z",
      "granularity": "daily"
    },
    "summary": {
      "total_operations": 45678,
      "total_data_encrypted_bytes": 987654321,
      "average_response_time_ms": 125,
      "success_rate": 99.95,
      "active_keys": 23,
      "unique_users": 156
    },
    "daily_metrics": [
      {
        "date": "2025-01-15",
        "operations": 1523,
        "data_bytes": 32145678,
        "avg_response_time_ms": 118,
        "success_rate": 99.87,
        "errors": 2
      }
    ],
    "algorithm_usage": {
      "CRYSTALS-Kyber-1024": 72.3,
      "CRYSTALS-Kyber-768": 23.1,
      "CRYSTALS-Kyber-512": 4.6
    },
    "geographic_distribution": {
      "EU": 45.2,
      "US": 32.1,
      "APAC": 22.7
    }
  }
}
```

### GET /health

Health check endpoint for monitoring system status.

#### Response
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2025-01-15T15:30:00Z",
    "version": "1.0.0",
    "components": {
      "encryption_service": {
        "status": "healthy",
        "response_time_ms": 45,
        "last_check": "2025-01-15T15:29:00Z"
      },
      "key_management": {
        "status": "healthy",
        "active_keys": 1523,
        "key_generation_time_ms": 89
      },
      "database": {
        "status": "healthy",
        "connection_pool": "85% utilized",
        "query_time_ms": 12
      },
      "quantum_entropy": {
        "status": "healthy",
        "entropy_quality": 0.997,
        "sources_active": 4
      }
    },
    "performance": {
      "operations_per_second": 167,
      "average_latency_ms": 125,
      "cpu_usage": "45%",
      "memory_usage": "67%"
    }
  }
}
```

## Error Codes Reference

### Authentication & Authorization (4xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `AUTH_INVALID_API_KEY` | 401 | Invalid or expired API key | Check API key configuration |
| `AUTH_INVALID_TENANT` | 401 | Invalid tenant ID | Verify tenant ID |
| `AUTH_INSUFFICIENT_PERMISSIONS` | 403 | Insufficient permissions | Check role assignments |
| `AUTH_MFA_REQUIRED` | 403 | Multi-factor authentication required | Complete MFA challenge |

### Request Validation (4xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `VALIDATION_INVALID_DATA` | 400 | Invalid request data | Check request format |
| `VALIDATION_MISSING_FIELD` | 400 | Required field missing | Include all required fields |
| `VALIDATION_INVALID_ALGORITHM` | 400 | Unsupported algorithm | Use supported algorithm |
| `VALIDATION_DATA_TOO_LARGE` | 413 | Data exceeds size limit | Reduce data size or use batch API |

### Rate Limiting (4xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded | Implement exponential backoff |
| `QUOTA_EXCEEDED` | 429 | Monthly quota exceeded | Upgrade plan or wait for reset |

### Resource Errors (4xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `RESOURCE_KEY_NOT_FOUND` | 404 | Encryption key not found | Verify key ID exists |
| `RESOURCE_POLICY_NOT_FOUND` | 404 | Policy not found | Check policy ID |
| `RESOURCE_KEY_EXPIRED` | 410 | Key has expired | Generate new key |
| `RESOURCE_KEY_DEACTIVATED` | 410 | Key is deactivated | Use active key |

### Encryption/Decryption Errors (4xx/5xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `ENCRYPT_OPERATION_FAILED` | 400 | Encryption operation failed | Check data format and algorithm |
| `DECRYPT_OPERATION_FAILED` | 400 | Decryption operation failed | Verify key and encrypted data |
| `DECRYPT_INTEGRITY_CHECK_FAILED` | 400 | Data integrity check failed | Data may be corrupted |
| `ENCRYPT_KEY_GENERATION_FAILED` | 500 | Key generation failed | Retry or contact support |

### System Errors (5xx)
| Code | HTTP Status | Description | Solution |
|------|-------------|-------------|----------|
| `SYSTEM_INTERNAL_ERROR` | 500 | Internal system error | Retry request or contact support |
| `SYSTEM_SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable | Check status page and retry |
| `SYSTEM_TIMEOUT` | 504 | Request timeout | Increase timeout or retry |
| `SYSTEM_MAINTENANCE` | 503 | System under maintenance | Check maintenance schedule |

## SDK Integration Examples

### Python SDK
```python
import asyncio
from apg_encryption import APGEncryptionClient, APGEncryptionError

async def main():
    try:
        async with APGEncryptionClient(
            tenant_id="your-tenant-id",
            api_key="your-api-key",
            base_url="https://api.datacraft.co.ke"
        ) as client:
            
            # Encrypt data
            result = await client.encrypt_quantum_safe(
                "Sensitive information",
                algorithm="CRYSTALS-Kyber-1024"
            )
            
            print(f"Encrypted data: {result.encrypted_data}")
            print(f"Key ID: {result.key_id}")
            
            # Decrypt data
            decrypted = await client.decrypt_quantum_safe(
                result.encrypted_data,
                key_id=result.key_id
            )
            
            print(f"Decrypted data: {decrypted.decrypted_data}")
            
    except APGEncryptionError as e:
        print(f"Encryption error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript SDK
```javascript
const { APGEncryptionClient } = require('apg-encryption-js');

async function main() {
    const client = new APGEncryptionClient({
        tenantId: 'your-tenant-id',
        apiKey: 'your-api-key',
        baseUrl: 'https://api.datacraft.co.ke'
    });
    
    try {
        // Encrypt data
        const result = await client.encryptQuantumSafe(
            'Sensitive information',
            'CRYSTALS-Kyber-1024'
        );
        
        console.log('Encrypted data:', result.encryptedData);
        console.log('Key ID:', result.keyId);
        
        // Decrypt data
        const decrypted = await client.decryptQuantumSafe(
            result.encryptedData,
            result.keyId
        );
        
        console.log('Decrypted data:', decrypted.decryptedData);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

### cURL Examples
```bash
# Encrypt data
curl -X POST https://api.datacraft.co.ke/api/v1/encrypt \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Tenant-ID: your-tenant-id" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "48656c6c6f2c20576f726c6421",
    "algorithm": "CRYSTALS-Kyber-1024"
  }'

# Generate key pair
curl -X POST https://api.datacraft.co.ke/api/v1/keys/generate \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Tenant-ID: your-tenant-id" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "CRYSTALS-Kyber-1024",
    "metadata": {
      "purpose": "data_encryption",
      "department": "engineering"
    }
  }'

# List keys
curl -X GET "https://api.datacraft.co.ke/api/v1/keys?limit=10&status=active" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Tenant-ID: your-tenant-id"
```

## Webhooks

APG Encryption Services supports webhooks for real-time notifications about key events.

### Webhook Events
| Event | Description |
|-------|-------------|
| `key.generated` | New key pair generated |
| `key.rotated` | Key rotation completed |
| `key.expired` | Key has expired |
| `encryption.failed` | Encryption operation failed |
| `compliance.violation` | Compliance policy violation detected |
| `audit.required` | Audit action required |

### Webhook Configuration
```bash
curl -X POST https://api.datacraft.co.ke/api/v1/webhooks \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Tenant-ID: your-tenant-id" \
  -d '{
    "url": "https://your-server.com/webhooks/apg",
    "events": ["key.generated", "key.rotated"],
    "secret": "your-webhook-secret"
  }'
```

### Webhook Payload Example
```json
{
  "event": "key.rotated",
  "timestamp": "2025-01-15T10:30:00Z",
  "tenant_id": "your-tenant-id",
  "data": {
    "old_key_id": "key_old_123",
    "new_key_id": "key_new_456",
    "algorithm": "CRYSTALS-Kyber-1024",
    "rotation_reason": "scheduled"
  },
  "signature": "sha256=abc123def456..."
}
```

## Support and Resources

- **API Status**: [status.datacraft.co.ke](https://status.datacraft.co.ke)
- **Documentation**: [docs.datacraft.co.ke](https://docs.datacraft.co.ke)
- **Support**: [support@datacraft.co.ke](mailto:support@datacraft.co.ke)
- **GitHub**: [github.com/datacraft/apg-encryption](https://github.com/datacraft/apg-encryption)

---

© 2025 Datacraft - [www.datacraft.co.ke](https://www.datacraft.co.ke)