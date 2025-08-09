# APG API Reference

Complete API documentation for the APG platform. All APIs are production-ready with comprehensive error handling, authentication, and rate limiting.

## 🚀 API Overview

### Base URLs
- **Development**: `http://localhost:5000/api`
- **Production**: `https://api.yourdomain.com/api`
- **API Version**: `v1` (current)

### Authentication
All API endpoints require authentication unless otherwise specified.

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

### Standard Response Format
```json
{
  "success": true,
  "data": {},
  "message": "Operation completed successfully",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "reason": "Invalid email format"
    }
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

## 🔐 Authentication API

### POST /api/auth/login
Authenticate user and obtain JWT token.

**Request Body**:
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "mfa_code": "123456"  // Optional: MFA code if enabled
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "def50200...",
    "expires_in": 3600,
    "user": {
      "id": "user_123",
      "email": "user@example.com",
      "roles": ["user", "workflow_manager"]
    }
  }
}
```

### POST /api/auth/refresh
Refresh JWT token using refresh token.

**Request Body**:
```json
{
  "refresh_token": "def50200..."
}
```

### POST /api/auth/logout
Invalidate current session.

### GET /api/auth/me
Get current user information.

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "roles": ["user", "workflow_manager"],
    "permissions": ["workflow:create", "workflow:execute"],
    "last_login": "2025-01-15T09:00:00Z",
    "mfa_enabled": true
  }
}
```

## 🔄 Workflow Orchestration API

### GET /api/workflows/
List workflows with filtering and pagination.

**Query Parameters**:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 50, max: 100)
- `status`: Filter by status (pending, running, completed, failed)
- `created_by`: Filter by creator user ID
- `search`: Search in workflow name and description

**Response**:
```json
{
  "success": true,
  "data": {
    "workflows": [
      {
        "id": "wf_abc123",
        "name": "Data Processing Pipeline",
        "description": "Process customer data files",
        "status": "completed",
        "created_by": "user_123",
        "created_at": "2025-01-15T08:00:00Z",
        "updated_at": "2025-01-15T10:00:00Z",
        "execution_count": 15,
        "success_rate": 93.3
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 50,
      "total": 125,
      "pages": 3
    }
  }
}
```

### POST /api/workflows/
Create a new workflow.

**Request Body**:
```json
{
  "name": "Data Processing Pipeline",
  "description": "Process customer data files",
  "engine": "prefect",  // prefect, airflow, celery, native
  "tasks": [
    {
      "id": "extract_data",
      "type": "python",
      "config": {
        "function": "extract_customer_data",
        "parameters": {
          "source_path": "/data/input"
        }
      },
      "dependencies": []
    },
    {
      "id": "transform_data",
      "type": "python",
      "config": {
        "function": "transform_data",
        "parameters": {
          "transformation_rules": "standard"
        }
      },
      "dependencies": ["extract_data"]
    }
  ],
  "schedule": {
    "type": "cron",
    "expression": "0 9 * * *"  // Daily at 9 AM
  },
  "retry_config": {
    "max_retries": 3,
    "retry_delay": 300,
    "fail_fast": false
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "wf_abc123",
    "name": "Data Processing Pipeline",
    "status": "created",
    "deployment_id": "dep_xyz789",
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

### GET /api/workflows/{id}
Get workflow details.

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "wf_abc123",
    "name": "Data Processing Pipeline",
    "description": "Process customer data files",
    "status": "running",
    "engine": "prefect",
    "tasks": [...],
    "schedule": {...},
    "retry_config": {...},
    "created_by": "user_123",
    "created_at": "2025-01-15T08:00:00Z",
    "updated_at": "2025-01-15T10:00:00Z",
    "executions": {
      "total": 15,
      "successful": 14,
      "failed": 1,
      "success_rate": 93.3
    },
    "current_execution": {
      "id": "exec_def456",
      "status": "running",
      "started_at": "2025-01-15T10:25:00Z",
      "progress": 65.0,
      "current_tasks": ["transform_data"]
    }
  }
}
```

### POST /api/workflows/{id}/execute
Execute workflow.

**Request Body**:
```json
{
  "parameters": {
    "input_file": "/data/customers_2025.csv",
    "output_format": "json"
  },
  "priority": "normal",  // low, normal, high, urgent
  "timeout": 3600  // Execution timeout in seconds
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "execution_id": "exec_def456",
    "workflow_id": "wf_abc123",
    "status": "queued",
    "estimated_duration": 1800,
    "started_at": "2025-01-15T10:30:00Z"
  }
}
```

### GET /api/workflows/{id}/executions
Get workflow execution history.

**Query Parameters**:
- `page`: Page number
- `limit`: Items per page
- `status`: Filter by execution status
- `from_date`: Start date filter
- `to_date`: End date filter

### GET /api/workflows/executions/{execution_id}
Get execution details and status.

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "exec_def456",
    "workflow_id": "wf_abc123",
    "status": "running",
    "progress": 65.0,
    "started_at": "2025-01-15T10:25:00Z",
    "estimated_completion": "2025-01-15T10:55:00Z",
    "parameters": {...},
    "tasks": [
      {
        "id": "extract_data",
        "status": "completed",
        "started_at": "2025-01-15T10:25:00Z",
        "completed_at": "2025-01-15T10:30:00Z",
        "duration": 300,
        "result": {
          "records_processed": 10000,
          "output_file": "/tmp/extracted_data.json"
        }
      },
      {
        "id": "transform_data",
        "status": "running",
        "started_at": "2025-01-15T10:30:00Z",
        "progress": 45.0,
        "current_operation": "applying_transformations"
      }
    ],
    "logs": [
      {
        "timestamp": "2025-01-15T10:25:00Z",
        "level": "INFO",
        "task": "extract_data",
        "message": "Starting data extraction"
      }
    ]
  }
}
```

### POST /api/workflows/executions/{execution_id}/cancel
Cancel running execution.

### GET /api/workflows/executions/{execution_id}/logs
Get execution logs with streaming support.

**Query Parameters**:
- `follow`: Stream logs in real-time (true/false)
- `tail`: Number of recent log entries
- `level`: Filter by log level (DEBUG, INFO, WARNING, ERROR)

## 🤖 AI/ML API

### POST /api/ml/federated/train
Start federated learning training session.

**Request Body**:
```json
{
  "model_id": "model_abc123",
  "participants": [
    {
      "id": "participant_1",
      "endpoint": "https://client1.example.com",
      "data_size": 10000
    },
    {
      "id": "participant_2", 
      "endpoint": "https://client2.example.com",
      "data_size": 15000
    }
  ],
  "aggregation_method": "fedavg",  // fedavg, weighted, secure
  "privacy_config": {
    "differential_privacy": true,
    "epsilon": 1.0,
    "delta": 1e-5
  },
  "training_config": {
    "rounds": 10,
    "min_participants": 2,
    "timeout_per_round": 600
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "training_session_id": "fs_def456",
    "status": "initializing",
    "participants_count": 2,
    "estimated_duration": 3600,
    "started_at": "2025-01-15T10:30:00Z"
  }
}
```

### GET /api/ml/federated/sessions/{session_id}
Get federated training session status.

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "fs_def456",
    "model_id": "model_abc123",
    "status": "training",
    "current_round": 3,
    "total_rounds": 10,
    "progress": 30.0,
    "participants": [
      {
        "id": "participant_1",
        "status": "training",
        "contribution_weight": 0.4,
        "last_update": "2025-01-15T10:35:00Z"
      }
    ],
    "metrics": {
      "accuracy": 0.87,
      "loss": 0.23,
      "privacy_budget_used": 0.3
    },
    "started_at": "2025-01-15T10:30:00Z",
    "estimated_completion": "2025-01-15T11:30:00Z"
  }
}
```

### GET /api/ml/models
List available ML models.

### POST /api/ml/models/{model_id}/predict
Make predictions using trained model.

**Request Body**:
```json
{
  "inputs": [
    {
      "feature1": 1.2,
      "feature2": "category_a",
      "feature3": [0.1, 0.2, 0.3]
    }
  ],
  "return_confidence": true,
  "return_explanations": false
}
```

## 🔗 Blockchain API

### POST /api/blockchain/wallets/connect
Connect blockchain wallet.

**Request Body**:
```json
{
  "network": "ethereum",  // ethereum, polygon, bsc, arbitrum
  "wallet_type": "metamask",  // metamask, walletconnect, hardware
  "address": "0x742d35Cc6634C0532925a3b8D35fC8B8BD02be21",
  "signature": "0x...",  // Signature for verification
  "message": "Connect to APG Platform"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "wallet_id": "wallet_abc123",
    "address": "0x742d35Cc6634C0532925a3b8D35fC8B8BD02be21",
    "network": "ethereum",
    "balance": {
      "ETH": "1.2345",
      "USDC": "1000.50",
      "DAI": "500.00"
    },
    "connected_at": "2025-01-15T10:30:00Z"
  }
}
```

### POST /api/blockchain/contracts/deploy
Deploy smart contract.

**Request Body**:
```json
{
  "wallet_id": "wallet_abc123",
  "contract_type": "ERC20",  // ERC20, ERC721, ERC1155, custom
  "source_code": "pragma solidity ^0.8.0; contract MyToken { ... }",
  "compiler_version": "0.8.19",
  "constructor_args": ["MyToken", "MTK", 18, 1000000],
  "gas_limit": 2000000
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "deployment_id": "deploy_def456",
    "transaction_hash": "0x1234567890abcdef...",
    "contract_address": "0xabcdef1234567890...",
    "gas_used": 1500000,
    "deployment_cost": "0.045",
    "status": "pending"
  }
}
```

### POST /api/blockchain/defi/lend
Create DeFi lending position.

**Request Body**:
```json
{
  "wallet_id": "wallet_abc123",
  "protocol": "aave",  // aave, compound, maker
  "token_address": "0xA0b86a33E6b0b5c0F7E68e7d3c5C3E2C4C7E6C7E",
  "amount": "100.0",
  "duration": 30  // days
}
```

## 📱 Mobile API

### GET /api/mobile/sync
Synchronize mobile app data.

**Query Parameters**:
- `last_sync`: Last sync timestamp
- `device_id`: Unique device identifier

**Response**:
```json
{
  "success": true,
  "data": {
    "workflows": [...],
    "notifications": [...],
    "user_data": {...},
    "sync_timestamp": "2025-01-15T10:30:00Z",
    "pending_operations": []
  }
}
```

### POST /api/mobile/sync/conflict
Resolve synchronization conflict.

**Request Body**:
```json
{
  "conflict_id": "conflict_123",
  "resolution": "server_wins",  // server_wins, client_wins, merge
  "merged_data": {}  // Required if resolution is 'merge'
}
```

## 🔒 Biometric API

### POST /api/biometric/enroll
Enroll biometric template.

**Request Body**:
```json
{
  "user_id": "user_123",
  "biometric_type": "fingerprint",  // fingerprint, face, voice
  "template_data": "base64_encoded_template",
  "quality_score": 0.95,
  "device_info": {
    "device_id": "device_abc",
    "sensor_type": "optical",
    "firmware_version": "1.2.3"
  }
}
```

### POST /api/biometric/verify
Verify biometric authentication.

**Request Body**:
```json
{
  "user_id": "user_123",
  "biometric_type": "fingerprint",
  "template_data": "base64_encoded_template",
  "challenge_id": "challenge_def456"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "verified": true,
    "confidence_score": 0.98,
    "match_quality": "high",
    "verification_time": 150,  // milliseconds
    "liveness_detected": true
  }
}
```

## 📄 Document API

### POST /api/documents/generate/pdf
Generate PDF document.

**Request Body**:
```json
{
  "template": "invoice",
  "data": {
    "invoice_number": "INV-001",
    "customer_name": "John Doe",
    "items": [
      {
        "description": "Service A",
        "quantity": 2,
        "price": 100.00
      }
    ]
  },
  "options": {
    "format": "A4",
    "orientation": "portrait",
    "margins": {
      "top": 20,
      "bottom": 20,
      "left": 20,
      "right": 20
    }
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "document_id": "doc_abc123",
    "download_url": "/api/documents/doc_abc123/download",
    "file_size": 245760,
    "page_count": 3,
    "generated_at": "2025-01-15T10:30:00Z"
  }
}
```

### POST /api/documents/generate/excel
Generate Excel spreadsheet.

**Request Body**:
```json
{
  "template": "financial_report",
  "data": {
    "sheets": [
      {
        "name": "Summary",
        "data": [
          ["Metric", "Value"],
          ["Revenue", 100000],
          ["Expenses", 75000]
        ]
      }
    ]
  },
  "options": {
    "include_charts": true,
    "format_numbers": true
  }
}
```

## 🔔 Notification API

### POST /api/notifications/send
Send notification.

**Request Body**:
```json
{
  "recipients": ["user_123", "user_456"],
  "channels": ["email", "push", "in_app"],
  "title": "Workflow Completed",
  "message": "Your data processing workflow has completed successfully.",
  "data": {
    "workflow_id": "wf_abc123",
    "execution_id": "exec_def456"
  },
  "priority": "normal",  // low, normal, high, urgent
  "schedule_at": "2025-01-15T15:00:00Z"  // Optional: scheduled delivery
}
```

### GET /api/notifications/
Get user notifications.

**Query Parameters**:
- `page`: Page number
- `limit`: Items per page
- `status`: Filter by status (unread, read, all)
- `channel`: Filter by channel

### POST /api/notifications/{id}/mark-read
Mark notification as read.

## 📊 Analytics API

### GET /api/analytics/workflows
Get workflow analytics.

**Query Parameters**:
- `from_date`: Start date
- `to_date`: End date
- `group_by`: Grouping (day, week, month)

**Response**:
```json
{
  "success": true,
  "data": {
    "total_executions": 1250,
    "success_rate": 94.2,
    "average_duration": 1800,
    "popular_workflows": [
      {
        "id": "wf_abc123",
        "name": "Data Processing",
        "execution_count": 450
      }
    ],
    "execution_trends": [
      {
        "date": "2025-01-15",
        "executions": 45,
        "success_rate": 95.6
      }
    ]
  }
}
```

### GET /api/analytics/performance
Get system performance metrics.

**Response**:
```json
{
  "success": true,
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "disk_usage": 34.1,
    "active_workflows": 12,
    "queue_length": 3,
    "average_response_time": 150,
    "requests_per_minute": 1200
  }
}
```

## 🚨 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Authentication required |
| `INVALID_CREDENTIALS` | 401 | Invalid login credentials |
| `ACCESS_DENIED` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Input validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | API rate limit exceeded |
| `WORKFLOW_EXECUTION_ERROR` | 422 | Workflow execution failed |
| `BLOCKCHAIN_ERROR` | 422 | Blockchain operation failed |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected server error |

## 📡 WebSocket API

### Connection
```javascript
const ws = new WebSocket('wss://api.yourdomain.com/ws');
ws.send(JSON.stringify({
  type: 'authenticate',
  token: 'your_jwt_token'
}));
```

### Events
- `workflow_status_changed`: Real-time workflow execution updates
- `notification_received`: New notifications
- `system_alert`: System alerts and warnings
- `user_activity`: User activity in collaborative features

### Example Messages
```javascript
// Workflow status update
{
  "type": "workflow_status_changed",
  "data": {
    "workflow_id": "wf_abc123",
    "execution_id": "exec_def456",
    "status": "running",
    "progress": 75.0
  }
}

// New notification
{
  "type": "notification_received",
  "data": {
    "id": "notif_123",
    "title": "Task Completed",
    "message": "Your document generation task has completed",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

## 📞 Support

For API support and questions:
- **Email**: nyimbi@gmail.com
- **Documentation Issues**: Create GitHub issue
- **API Status**: Check system status page

---

*Last Updated: January 2025*