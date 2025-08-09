# APG Key Management - API Reference

## Overview

The APG Key Management API provides comprehensive cryptographic key lifecycle management through RESTful endpoints. This reference covers all available endpoints, data models, authentication, and usage examples.

## Base URL

```
https://your-apg-domain.com/keym/api/v1
```

## Authentication

All API endpoints require authentication using APG's standard authentication mechanisms.

### API Key Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://your-apg-domain.com/keym/api/v1/keys
```

### JWT Token Authentication

```bash
curl -H "Authorization: JWT YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     https://your-apg-domain.com/keym/api/v1/keys
```

## Data Models

### KeySpecification

```json
{
  "id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "tenant_id": "enterprise-tenant",
  "algorithm": "AES_256",
  "usage": ["encrypt", "decrypt"],
  "name": "Production Encryption Key",
  "description": "Key for encrypting customer data",
  "metadata": {
    "department": "security",
    "environment": "production"
  },
  "created_by": "admin@company.com",
  "created_at": "2025-01-09T10:30:00Z",
  "expires_at": "2026-01-09T10:30:00Z",
  "rotation_policy": "monthly"
}
```

### KeyInfo

```json
{
  "spec": {
    "id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
    "tenant_id": "enterprise-tenant",
    "algorithm": "AES_256",
    "usage": ["encrypt", "decrypt"],
    "name": "Production Encryption Key",
    "created_by": "admin@company.com",
    "created_at": "2025-01-09T10:30:00Z"
  },
  "metadata": {
    "status": "active",
    "version": 1,
    "last_used": "2025-01-09T15:45:00Z",
    "usage_count": 1250,
    "storage_location": "database",
    "federation_status": ["aws", "azure"]
  }
}
```

### Error Response

```json
{
  "error": {
    "code": "KEY_NOT_FOUND",
    "message": "Key not found: 01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
    "details": {
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "timestamp": "2025-01-09T16:00:00Z"
    }
  }
}
```

## Key Management Endpoints

### Create Key

Create a new cryptographic key.

**Endpoint:** `POST /keys`

**Request Body:**

```json
{
  "algorithm": "AES_256",
  "usage": ["encrypt", "decrypt"],
  "name": "My Encryption Key",
  "description": "Key for encrypting sensitive data",
  "metadata": {
    "project": "alpha",
    "cost_center": "engineering"
  },
  "expires_at": "2026-01-09T10:30:00Z",
  "rotation_policy": "quarterly"
}
```

**Response:** `201 Created`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "algorithm": "AES_256",
  "usage": ["encrypt", "decrypt"],
  "created_at": "2025-01-09T10:30:00Z",
  "metadata": {
    "status": "active",
    "version": 1
  }
}
```

**Example:**

```bash
curl -X POST https://your-apg-domain.com/keym/api/v1/keys \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "AES_256",
    "usage": ["encrypt", "decrypt"],
    "name": "Customer Data Encryption Key",
    "description": "Used for encrypting customer PII"
  }'
```

### List Keys

Retrieve a list of keys for the authenticated user.

**Endpoint:** `GET /keys`

**Query Parameters:**
- `limit` (integer, optional): Maximum number of keys to return (default: 50, max: 1000)
- `offset` (integer, optional): Number of keys to skip (default: 0)
- `status` (string, optional): Filter by key status (`active`, `archived`, `expired`)
- `algorithm` (string, optional): Filter by algorithm
- `search` (string, optional): Search in key names and descriptions

**Response:** `200 OK`

```json
{
  "keys": [
    {
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "name": "Production Encryption Key",
      "algorithm": "AES_256",
      "usage": ["encrypt", "decrypt"],
      "status": "active",
      "created_at": "2025-01-09T10:30:00Z",
      "last_used": "2025-01-09T15:45:00Z"
    }
  ],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 1,
    "has_next": false,
    "has_previous": false
  }
}
```

**Example:**

```bash
curl "https://your-apg-domain.com/keym/api/v1/keys?limit=10&status=active" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Get Key Details

Retrieve detailed information about a specific key.

**Endpoint:** `GET /keys/{key_id}`

**Response:** `200 OK`

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "name": "Production Encryption Key",
  "algorithm": "AES_256",
  "usage": ["encrypt", "decrypt"],
  "status": "active",
  "created_at": "2025-01-09T10:30:00Z",
  "created_by": "admin@company.com",
  "description": "Key for encrypting customer data",
  "metadata": {
    "version": 1,
    "last_used": "2025-01-09T15:45:00Z",
    "usage_count": 1250,
    "storage_location": "database"
  },
  "rotation_policy": "monthly",
  "expires_at": "2026-01-09T10:30:00Z"
}
```

**Example:**

```bash
curl https://your-apg-domain.com/keym/api/v1/keys/01HZ9Q2K3M4N5P6Q7R8S9T0U1V \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Update Key

Update key metadata and configuration.

**Endpoint:** `PUT /keys/{key_id}`

**Request Body:**

```json
{
  "name": "Updated Key Name",
  "description": "Updated description",
  "metadata": {
    "project": "beta",
    "updated": "2025-01-09"
  },
  "rotation_policy": "monthly"
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "message": "Key updated successfully"
}
```

### Archive Key

Archive a key (makes it inactive but preserves for decryption).

**Endpoint:** `POST /keys/{key_id}/archive`

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "previous_status": "active",
  "new_status": "archived",
  "archived_at": "2025-01-09T16:00:00Z"
}
```

### Delete Key

Permanently delete a key (irreversible).

**Endpoint:** `DELETE /keys/{key_id}`

**Query Parameters:**
- `secure_delete` (boolean, optional): Perform secure deletion (default: true)
- `force` (boolean, optional): Force deletion even if key is still in use (default: false)

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "message": "Key deleted successfully",
  "secure_delete": true,
  "deleted_at": "2025-01-09T16:00:00Z"
}
```

**Example:**

```bash
curl -X DELETE "https://your-apg-domain.com/keym/api/v1/keys/01HZ9Q2K3M4N5P6Q7R8S9T0U1V?secure_delete=true" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Cryptographic Operations

### Encrypt Data

Encrypt data using a specified key.

**Endpoint:** `POST /crypto/encrypt`

**Request Body:**

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "data": "SGVsbG8gV29ybGQh",
  "additional_data": "Y29udGV4dA=="
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "encrypted_data": "gAAAAABhZ2M...",
  "algorithm": "AES_256",
  "operation_id": "op_01HZ9Q2K3M4N5P6Q7R8S9T0U1V"
}
```

**Example:**

```bash
# Encrypt text data (base64 encoded)
echo -n "Hello World!" | base64  # SGVsbG8gV29ybGQh

curl -X POST https://your-apg-domain.com/keym/api/v1/crypto/encrypt \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
    "data": "SGVsbG8gV29ybGQh"
  }'
```

### Decrypt Data

Decrypt data using a specified key.

**Endpoint:** `POST /crypto/decrypt`

**Request Body:**

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "encrypted_data": "gAAAAABhZ2M...",
  "additional_data": "Y29udGV4dA=="
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "decrypted_data": "SGVsbG8gV29ybGQh",
  "algorithm": "AES_256",
  "operation_id": "op_01HZ9Q2K3M4N5P6Q7R8S9T0U1V"
}
```

### Sign Data

Create digital signature for data.

**Endpoint:** `POST /crypto/sign`

**Request Body:**

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "data": "SGVsbG8gV29ybGQh",
  "hash_algorithm": "SHA256"
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "signature": "MEUCIQDv...",
  "hash_algorithm": "SHA256",
  "signature_algorithm": "ECDSA_P256"
}
```

### Verify Signature

Verify digital signature.

**Endpoint:** `POST /crypto/verify`

**Request Body:**

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "data": "SGVsbG8gV29ybGQh",
  "signature": "MEUCIQDv...",
  "hash_algorithm": "SHA256"
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "valid": true,
  "signature_algorithm": "ECDSA_P256",
  "verified_at": "2025-01-09T16:00:00Z"
}
```

## Key Lifecycle Management

### Rotate Key

Create a new version of an existing key.

**Endpoint:** `POST /keys/{key_id}/rotate`

**Request Body (optional):**

```json
{
  "reason": "Scheduled rotation",
  "preserve_old_version": true,
  "immediate_activation": true
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "old_key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "new_key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1W",
  "rotation_id": "rot_01HZ9Q2K3M4N5P6Q7R8S9T0U1X",
  "rotated_at": "2025-01-09T16:00:00Z",
  "reason": "Scheduled rotation"
}
```

### Get Key Versions

List all versions of a key.

**Endpoint:** `GET /keys/{key_id}/versions`

**Response:** `200 OK`

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "versions": [
    {
      "version": 2,
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1W",
      "status": "active",
      "created_at": "2025-01-09T16:00:00Z"
    },
    {
      "version": 1,
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "status": "archived",
      "created_at": "2025-01-09T10:30:00Z"
    }
  ],
  "current_version": 2
}
```

## Batch Operations

### Create Multiple Keys

Create multiple keys in a single request.

**Endpoint:** `POST /keys/batch`

**Request Body:**

```json
{
  "keys": [
    {
      "algorithm": "AES_256",
      "usage": ["encrypt", "decrypt"],
      "name": "Batch Key 1"
    },
    {
      "algorithm": "RSA_2048",
      "usage": ["sign", "verify"],
      "name": "Batch Key 2"
    }
  ]
}
```

**Response:** `201 Created`

```json
{
  "status": "success",
  "created_count": 2,
  "failed_count": 0,
  "keys": [
    {
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "status": "created",
      "name": "Batch Key 1"
    },
    {
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1W",
      "status": "created",
      "name": "Batch Key 2"
    }
  ]
}
```

### Batch Encrypt

Encrypt multiple data items.

**Endpoint:** `POST /crypto/batch/encrypt`

**Request Body:**

```json
{
  "operations": [
    {
      "operation_id": "op1",
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "data": "SGVsbG8gV29ybGQh"
    },
    {
      "operation_id": "op2",
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1W",
      "data": "Rm9vIEJhcg=="
    }
  ]
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "successful_operations": 2,
  "failed_operations": 0,
  "results": [
    {
      "operation_id": "op1",
      "status": "success",
      "encrypted_data": "gAAAAABhZ2M..."
    },
    {
      "operation_id": "op2",
      "status": "success",
      "encrypted_data": "gAAAAABhZ2N..."
    }
  ]
}
```

## Advanced Features

### Multi-Cloud Key Federation

#### Federate Key

Replicate a key across multiple cloud providers.

**Endpoint:** `POST /keys/{key_id}/federate`

**Request Body:**

```json
{
  "target_clouds": ["aws", "azure", "gcp"],
  "sync_policy": "immediate",
  "encryption_in_transit": true
}
```

**Response:** `200 OK`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "federation_id": "fed_01HZ9Q2K3M4N5P6Q7R8S9T0U1X",
  "target_clouds": ["aws", "azure", "gcp"],
  "federation_status": {
    "aws": "completed",
    "azure": "in_progress",
    "gcp": "pending"
  },
  "started_at": "2025-01-09T16:00:00Z"
}
```

#### Get Federation Status

Check the status of key federation.

**Endpoint:** `GET /keys/{key_id}/federation`

**Response:** `200 OK`

```json
{
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "federation_enabled": true,
  "clouds": {
    "aws": {
      "status": "active",
      "key_arn": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
      "last_sync": "2025-01-09T16:00:00Z"
    },
    "azure": {
      "status": "active",
      "key_vault_url": "https://vault.vault.azure.net/keys/key-name/version",
      "last_sync": "2025-01-09T16:00:00Z"
    },
    "gcp": {
      "status": "active",
      "key_name": "projects/project-id/locations/global/keyRings/ring/cryptoKeys/key",
      "last_sync": "2025-01-09T16:00:00Z"
    }
  }
}
```

### HSM Integration

#### Create HSM Key

Create a key in Hardware Security Module.

**Endpoint:** `POST /hsm/keys`

**Request Body:**

```json
{
  "hsm_id": "primary-hsm",
  "algorithm": "AES_256",
  "usage": ["encrypt", "decrypt"],
  "name": "HSM Protected Key",
  "extractable": false,
  "backup_enabled": true
}
```

**Response:** `201 Created`

```json
{
  "status": "success",
  "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "hsm_id": "primary-hsm",
  "hsm_key_handle": "hsm_handle_123",
  "extractable": false,
  "created_at": "2025-01-09T16:00:00Z"
}
```

#### Get HSM Status

Check HSM connectivity and status.

**Endpoint:** `GET /hsm/{hsm_id}/status`

**Response:** `200 OK`

```json
{
  "hsm_id": "primary-hsm",
  "status": "online",
  "firmware_version": "7.4.0",
  "authentication_status": "authenticated",
  "available_slots": 8,
  "used_slots": 3,
  "performance": {
    "operations_per_second": 10000,
    "average_latency_ms": 2.5
  },
  "last_health_check": "2025-01-09T16:00:00Z"
}
```

### Security Intelligence

#### Get Security Analytics

Retrieve security analytics for keys.

**Endpoint:** `GET /security/analytics`

**Query Parameters:**
- `time_range` (string): Time range for analytics (e.g., "7d", "30d", "90d")
- `key_id` (string, optional): Specific key to analyze
- `include_anomalies` (boolean): Include anomaly detection results

**Response:** `200 OK`

```json
{
  "time_range": "30d",
  "total_keys": 150,
  "total_operations": 50000,
  "security_score": 0.85,
  "risk_distribution": {
    "low": 130,
    "medium": 15,
    "high": 4,
    "critical": 1
  },
  "anomalies_detected": 3,
  "anomalies": [
    {
      "key_id": "01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "anomaly_type": "unusual_access_pattern",
      "severity": "medium",
      "description": "Key accessed from unusual location",
      "detected_at": "2025-01-09T15:30:00Z"
    }
  ],
  "recommendations": [
    "Enable multi-factor authentication for high-risk keys",
    "Review access patterns for keys with medium risk score"
  ]
}
```

### Policy Management

#### Create Policy

Create automated key management policy.

**Endpoint:** `POST /policies`

**Request Body:**

```json
{
  "name": "Monthly Rotation Policy",
  "type": "automatic_rotation",
  "schedule": "0 0 1 * *",
  "conditions": {
    "key_age_days": 30,
    "usage_threshold": 10000
  },
  "actions": [
    {
      "type": "rotate_key",
      "parameters": {
        "preserve_old_version": true,
        "notification_required": true
      }
    }
  ],
  "target_keys": {
    "algorithm": ["AES_256"],
    "metadata.environment": ["production"]
  }
}
```

**Response:** `201 Created`

```json
{
  "status": "success",
  "policy_id": "pol_01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "name": "Monthly Rotation Policy",
  "created_at": "2025-01-09T16:00:00Z",
  "next_execution": "2025-02-01T00:00:00Z"
}
```

#### List Policies

**Endpoint:** `GET /policies`

**Response:** `200 OK`

```json
{
  "policies": [
    {
      "policy_id": "pol_01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
      "name": "Monthly Rotation Policy",
      "type": "automatic_rotation",
      "status": "active",
      "next_execution": "2025-02-01T00:00:00Z",
      "affected_keys_count": 25
    }
  ],
  "total": 1
}
```

## Monitoring and Reporting

### Get Usage Statistics

**Endpoint:** `GET /statistics/usage`

**Query Parameters:**
- `start_date` (string): Start date (ISO 8601)
- `end_date` (string): End date (ISO 8601)
- `granularity` (string): Data granularity ("hour", "day", "month")

**Response:** `200 OK`

```json
{
  "period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-09T23:59:59Z"
  },
  "summary": {
    "total_operations": 150000,
    "unique_keys_used": 85,
    "average_operations_per_day": 18750
  },
  "operations_by_type": {
    "encrypt": 75000,
    "decrypt": 70000,
    "sign": 3000,
    "verify": 2000
  },
  "algorithms_usage": {
    "AES_256": 145000,
    "RSA_2048": 3000,
    "ECDSA_P256": 2000
  },
  "timeline": [
    {
      "date": "2025-01-09",
      "operations": 20000,
      "unique_keys": 45
    }
  ]
}
```

### Generate Audit Report

**Endpoint:** `POST /reports/audit`

**Request Body:**

```json
{
  "report_type": "compliance",
  "period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-09T23:59:59Z"
  },
  "include_sections": [
    "key_inventory",
    "access_logs",
    "policy_compliance",
    "security_events"
  ],
  "format": "json"
}
```

**Response:** `202 Accepted`

```json
{
  "status": "accepted",
  "report_id": "rep_01HZ9Q2K3M4N5P6Q7R8S9T0U1V",
  "estimated_completion": "2025-01-09T16:05:00Z",
  "download_url": "/reports/rep_01HZ9Q2K3M4N5P6Q7R8S9T0U1V/download"
}
```

### Download Report

**Endpoint:** `GET /reports/{report_id}/download`

**Response:** `200 OK` (File download)

## Error Handling

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (async operation) |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field_name": "Additional error details",
      "timestamp": "2025-01-09T16:00:00Z"
    },
    "request_id": "req_01HZ9Q2K3M4N5P6Q7R8S9T0U1V"
  }
}
```

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `KEY_NOT_FOUND` | Specified key does not exist |
| `ALGORITHM_NOT_SUPPORTED` | Unsupported cryptographic algorithm |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `KEY_EXPIRED` | Key has expired and cannot be used |
| `HSM_UNAVAILABLE` | HSM is not available |
| `QUOTA_EXCEEDED` | API quota or rate limit exceeded |
| `CRYPTO_ERROR` | Cryptographic operation failed |

## Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|--------|
| Key Management | 1000 requests | 1 hour |
| Crypto Operations | 10000 requests | 1 hour |
| Batch Operations | 100 requests | 1 hour |
| Reports | 10 requests | 1 hour |

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641750000
```

## SDKs and Client Libraries

### Python SDK Example

```python
from keym_sdk import KeyManagementClient

client = KeyManagementClient(
    api_key="your-api-key",
    base_url="https://your-apg-domain.com/keym/api/v1"
)

# Create key
key = await client.create_key(
    algorithm="AES_256",
    usage=["encrypt", "decrypt"],
    name="My Key"
)

# Encrypt data
encrypted = await client.encrypt(key.key_id, b"Hello World!")

# Decrypt data
decrypted = await client.decrypt(key.key_id, encrypted)
```

### JavaScript SDK Example

```javascript
import { KeyManagementClient } from '@datacraft/keym-sdk';

const client = new KeyManagementClient({
  apiKey: 'your-api-key',
  baseURL: 'https://your-apg-domain.com/keym/api/v1'
});

// Create key
const key = await client.createKey({
  algorithm: 'AES_256',
  usage: ['encrypt', 'decrypt'],
  name: 'My Key'
});

// Encrypt data
const encrypted = await client.encrypt(key.keyId, 'Hello World!');

// Decrypt data
const decrypted = await client.decrypt(key.keyId, encrypted);
```

---

For additional information and examples, refer to the [User Guide](./USER_GUIDE.md) and [Developer Guide](./DEVELOPER_GUIDE.md).

**Contact Information**
- Website: www.datacraft.co.ke
- Email: nyimbi@gmail.com
- Copyright: © 2025 Datacraft