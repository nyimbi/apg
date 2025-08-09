# APG Accounts Payable - API Documentation

**Version**: 1.0  
**API Version**: v1  
**Last Updated**: January 2025  
**© 2025 Datacraft. All rights reserved.**

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Error Handling](#error-handling)
4. [Rate Limiting](#rate-limiting)
5. [Vendor Management APIs](#vendor-management-apis)
6. [Invoice Processing APIs](#invoice-processing-apis)
7. [Payment Processing APIs](#payment-processing-apis)
8. [Workflow Management APIs](#workflow-management-apis)
9. [Analytics APIs](#analytics-apis)
10. [WebSocket APIs](#websocket-apis)
11. [Webhook Integration](#webhook-integration)
12. [SDK Integration](#sdk-integration)
13. [Testing Guide](#testing-guide)

---

## API Overview

### Base URL
```
https://your-apg-instance.com/api/v1/core_financials/accounts_payable
```

### API Design Principles
- **RESTful Architecture**: Standard HTTP methods and status codes
- **JSON Format**: All request/response bodies use JSON
- **Async Operations**: Support for long-running operations
- **Pagination**: Consistent pagination across list endpoints
- **Versioning**: URL-based versioning with backward compatibility
- **Security**: APG authentication and authorization integration

### Response Format
All API responses follow this standard format:

```json
{
  "success": true,
  "data": {
    // Response data here
  },
  "metadata": {
    "timestamp": "2025-01-26T10:30:00Z",
    "request_id": "req_abc123",
    "api_version": "v1",
    "processing_time_ms": 150
  },
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 150,
    "has_more": true
  }
}
```

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Vendor code is required",
    "details": {
      "field": "vendor_code",
      "value": "",
      "constraint": "minLength:3"
    }
  },
  "metadata": {
    "timestamp": "2025-01-26T10:30:00Z",
    "request_id": "req_abc123",
    "api_version": "v1"
  }
}
```

---

## Authentication

### APG Authentication Integration

All API endpoints require valid APG authentication. The system supports multiple authentication methods:

#### Bearer Token Authentication
```http
Authorization: Bearer apg_token_abc123xyz789
```

#### API Key Authentication
```http
X-APG-API-Key: your-api-key-here
X-APG-Tenant-ID: your-tenant-id
```

#### Session-Based Authentication
```http
Cookie: apg_session=session_token_here
```

### Required Headers
```http
Content-Type: application/json
X-APG-Tenant-ID: your-tenant-id
X-APG-User-ID: your-user-id
Authorization: Bearer your-token
```

### Permission Requirements

Each endpoint requires specific permissions:

| Endpoint | Required Permission | Description |
|----------|-------------------|-------------|
| `GET /vendors` | `ap.read` | View vendor information |
| `POST /vendors` | `ap.vendor_admin` | Create new vendors |
| `GET /invoices` | `ap.read` | View invoice information |
| `POST /invoices` | `ap.write` | Create invoices |
| `POST /invoices/{id}/approve` | `ap.approve_invoice` | Approve invoices |
| `POST /payments` | `ap.process_payment` | Create payments |
| `GET /analytics/*` | `ap.analytics` | Access analytics |

---

## Error Handling

### HTTP Status Codes

| Status Code | Description | Usage |
|-------------|-------------|-------|
| `200` | OK | Successful GET, PUT, PATCH |
| `201` | Created | Successful POST |
| `202` | Accepted | Async operation started |
| `400` | Bad Request | Invalid request data |
| `401` | Unauthorized | Authentication required |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource not found |
| `409` | Conflict | Duplicate resource |
| `422` | Unprocessable Entity | Validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | Temporary unavailable |

### Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `VALIDATION_ERROR` | Request validation failed | Check request format |
| `AUTHENTICATION_ERROR` | Invalid credentials | Refresh authentication |
| `PERMISSION_DENIED` | Insufficient permissions | Check user roles |
| `RESOURCE_NOT_FOUND` | Requested resource not found | Verify resource ID |
| `DUPLICATE_RESOURCE` | Resource already exists | Check for duplicates |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Implement backoff |
| `EXTERNAL_SERVICE_ERROR` | Third-party service error | Retry after delay |
| `BUSINESS_RULE_VIOLATION` | Business logic constraint | Review business rules |

---

## Rate Limiting

### Rate Limits

| Endpoint Type | Requests per Minute | Burst Limit |
|---------------|-------------------|-------------|
| Read Operations | 1000 | 100 |
| Write Operations | 300 | 30 |
| Analytics | 100 | 10 |
| File Upload | 50 | 5 |
| Batch Operations | 20 | 2 |

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
Retry-After: 60
```

### Rate Limit Handling
```javascript
if (response.status === 429) {
  const retryAfter = response.headers['Retry-After'];
  await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
  // Retry request
}
```

---

## Vendor Management APIs

### Create Vendor

**Endpoint**: `POST /vendors`  
**Permission**: `ap.vendor_admin`

```http
POST /api/v1/core_financials/accounts_payable/vendors
Content-Type: application/json
Authorization: Bearer your-token

{
  "vendor_code": "ACME001",
  "legal_name": "ACME Corporation",
  "trade_name": "ACME Co.",
  "vendor_type": "supplier",
  "status": "pending_approval",
  "primary_contact": {
    "name": "John Smith",
    "email": "john.smith@acme.com",
    "phone": "+1-555-123-4567",
    "title": "Accounts Manager"
  },
  "addresses": [
    {
      "address_type": "billing",
      "line1": "123 Business Ave",
      "city": "Business City",
      "state_province": "BC",
      "postal_code": "12345",
      "country_code": "US",
      "is_primary": true
    }
  ],
  "payment_terms": {
    "code": "NET_30",
    "name": "Net 30",
    "net_days": 30,
    "discount_days": 10,
    "discount_percent": 2.00
  },
  "tax_information": {
    "tax_id": "12-3456789",
    "tax_id_type": "ein",
    "is_1099_vendor": false
  },
  "banking_details": [
    {
      "account_type": "checking",
      "bank_name": "First National Bank",
      "routing_number": "123456789",
      "account_number": "987654321",
      "account_holder_name": "ACME Corporation",
      "is_primary": true,
      "is_active": true
    }
  ]
}
```

**Response**: `201 Created`
```json
{
  "success": true,
  "data": {
    "id": "vendor_abc123",
    "vendor_code": "ACME001",
    "legal_name": "ACME Corporation",
    "status": "pending_approval",
    "created_at": "2025-01-26T10:30:00Z",
    "created_by": "user_xyz789"
  }
}
```

### Get Vendor

**Endpoint**: `GET /vendors/{vendor_id}`  
**Permission**: `ap.read`

```http
GET /api/v1/core_financials/accounts_payable/vendors/vendor_abc123
Authorization: Bearer your-token
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "id": "vendor_abc123",
    "vendor_code": "ACME001",
    "legal_name": "ACME Corporation",
    "trade_name": "ACME Co.",
    "vendor_type": "supplier",
    "status": "active",
    "primary_contact": {
      "name": "John Smith",
      "email": "john.smith@acme.com",
      "phone": "+1-555-123-4567"
    },
    "performance_metrics": {
      "on_time_payment_rate": 0.95,
      "invoice_accuracy_rate": 0.92,
      "overall_rating": 4.2
    },
    "created_at": "2025-01-26T10:30:00Z",
    "updated_at": "2025-01-26T10:30:00Z"
  }
}
```

### List Vendors

**Endpoint**: `GET /vendors`  
**Permission**: `ap.read`

```http
GET /api/v1/core_financials/accounts_payable/vendors?page=1&limit=50&status=active&vendor_type=supplier
```

**Query Parameters**:
- `page` (integer): Page number (default: 1)
- `limit` (integer): Results per page (default: 50, max: 200)
- `status` (string): Filter by status (active, inactive, pending_approval)
- `vendor_type` (string): Filter by type (supplier, contractor, service_provider)
- `search` (string): Search in name and code fields
- `sort` (string): Sort field (name, code, created_at)
- `order` (string): Sort order (asc, desc)

**Response**: `200 OK`
```json
{
  "success": true,
  "data": [
    {
      "id": "vendor_abc123",
      "vendor_code": "ACME001",
      "legal_name": "ACME Corporation",
      "status": "active",
      "total_spend": 150000.00,
      "last_payment_date": "2025-01-20T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 150,
    "has_more": true
  }
}
```

---

## Invoice Processing APIs

### Create Invoice

**Endpoint**: `POST /invoices`  
**Permission**: `ap.write`

```http
POST /api/v1/core_financials/accounts_payable/invoices
Content-Type: application/json

{
  "invoice_number": "INV-2025-001",
  "vendor_id": "vendor_abc123",
  "vendor_invoice_number": "ACME-12345",
  "invoice_date": "2025-01-26",
  "due_date": "2025-02-25",
  "purchase_order_number": "PO-2025-001",
  "currency_code": "USD",
  "exchange_rate": 1.00,
  "subtotal_amount": 1000.00,
  "tax_amount": 85.00,
  "total_amount": 1085.00,
  "payment_terms": {
    "code": "NET_30",
    "net_days": 30
  },
  "line_items": [
    {
      "line_number": 1,
      "description": "Professional Services",
      "quantity": 10.0000,
      "unit_price": 100.0000,
      "line_amount": 1000.00,
      "gl_account_code": "5000",
      "cost_center": "CC001",
      "department": "IT",
      "tax_code": "STANDARD"
    }
  ]
}
```

**Response**: `201 Created`
```json
{
  "success": true,
  "data": {
    "id": "invoice_xyz789",
    "invoice_number": "INV-2025-001",
    "status": "pending",
    "matching_status": "not_matched",
    "workflow_id": "workflow_def456",
    "created_at": "2025-01-26T10:30:00Z"
  }
}
```

### AI Invoice Processing

**Endpoint**: `POST /invoices/process-with-ai`  
**Permission**: `ap.write`

```http
POST /api/v1/core_financials/accounts_payable/invoices/process-with-ai
Content-Type: multipart/form-data

file: [PDF/Image file]
vendor_id: vendor_abc123
processing_options: {
  "confidence_threshold": 0.95,
  "auto_approve_high_confidence": false,
  "suggest_gl_codes": true
}
```

**Response**: `202 Accepted`
```json
{
  "success": true,
  "data": {
    "processing_id": "proc_ghi123",
    "status": "processing",
    "estimated_completion": "2025-01-26T10:32:00Z"
  }
}
```

**Get AI Processing Result**:
```http
GET /api/v1/core_financials/accounts_payable/invoices/ai-processing/proc_ghi123
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "processing_id": "proc_ghi123",
    "status": "completed",
    "confidence_score": 0.98,
    "processing_time_ms": 1200,
    "extracted_data": {
      "vendor_name": "ACME Corporation",
      "invoice_number": "ACME-12345",
      "invoice_date": "2025-01-26",
      "total_amount": "1085.00",
      "line_items": [
        {
          "description": "Professional Services",
          "amount": "1000.00",
          "gl_code_suggestion": "5000",
          "confidence": 0.96
        }
      ]
    },
    "suggested_gl_codes": [
      {
        "account_code": "5000",
        "account_name": "Professional Services",
        "confidence": 0.96
      }
    ],
    "validation_warnings": []
  }
}
```

### Approve Invoice

**Endpoint**: `POST /invoices/{invoice_id}/approve`  
**Permission**: `ap.approve_invoice`

```http
POST /api/v1/core_financials/accounts_payable/invoices/invoice_xyz789/approve
Content-Type: application/json

{
  "approval_comments": "Approved - documentation verified",
  "override_amount": null,
  "expedite_payment": false
}
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "invoice_id": "invoice_xyz789",
    "status": "approved",
    "approved_by": "user_xyz789",
    "approved_at": "2025-01-26T10:30:00Z",
    "next_workflow_step": "payment_processing"
  }
}
```

---

## Payment Processing APIs

### Create Payment

**Endpoint**: `POST /payments`  
**Permission**: `ap.process_payment`

```http
POST /api/v1/core_financials/accounts_payable/payments
Content-Type: application/json

{
  "payment_number": "PAY-2025-001",
  "vendor_id": "vendor_abc123",
  "payment_method": "ach",
  "payment_date": "2025-01-30",
  "currency_code": "USD",
  "bank_account_id": "bank_def456",
  "payment_lines": [
    {
      "invoice_id": "invoice_xyz789",
      "invoice_number": "INV-2025-001",
      "payment_amount": 1085.00,
      "discount_taken": 0.00,
      "discount_reason": null
    }
  ],
  "payment_memo": "Monthly payment run",
  "expedite": false
}
```

**Response**: `201 Created`
```json
{
  "success": true,
  "data": {
    "id": "payment_jkl012",
    "payment_number": "PAY-2025-001",
    "status": "pending",
    "payment_amount": 1085.00,
    "fraud_score": 0.12,
    "workflow_id": "workflow_mno345",
    "created_at": "2025-01-26T10:30:00Z"
  }
}
```

### Process Payment with Fraud Check

**Endpoint**: `POST /payments/{payment_id}/process-with-fraud-check`  
**Permission**: `ap.process_payment`

```http
POST /api/v1/core_financials/accounts_payable/payments/payment_jkl012/process-with-fraud-check
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "payment_id": "payment_jkl012",
    "fraud_check_result": {
      "risk_score": 0.12,
      "risk_level": "low",
      "risk_factors": [],
      "recommendation": "approve",
      "check_timestamp": "2025-01-26T10:30:00Z"
    },
    "processing_status": "approved_for_processing",
    "estimated_settlement": "2025-01-29T00:00:00Z"
  }
}
```

### Get Payment Status

**Endpoint**: `GET /payments/{payment_id}/status`  
**Permission**: `ap.read`

```http
GET /api/v1/core_financials/accounts_payable/payments/payment_jkl012/status
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "payment_id": "payment_jkl012",
    "status": "processing",
    "status_history": [
      {
        "status": "pending",
        "timestamp": "2025-01-26T10:30:00Z",
        "user": "user_xyz789"
      },
      {
        "status": "approved",
        "timestamp": "2025-01-26T11:00:00Z",
        "user": "user_abc123"
      },
      {
        "status": "processing",
        "timestamp": "2025-01-26T11:30:00Z",
        "user": "system"
      }
    ],
    "bank_reference": "ACH240126001",
    "settlement_date": "2025-01-29T00:00:00Z"
  }
}
```

---

## Workflow Management APIs

### Get Pending Approvals

**Endpoint**: `GET /workflows/pending-approvals`  
**Permission**: `ap.read`

```http
GET /api/v1/core_financials/accounts_payable/workflows/pending-approvals?approver_id=user_xyz789
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": [
    {
      "workflow_id": "workflow_def456",
      "entity_type": "invoice",
      "entity_id": "invoice_xyz789",
      "entity_number": "INV-2025-001",
      "vendor_name": "ACME Corporation",
      "amount": 1085.00,
      "priority": "normal",
      "time_remaining": "22h 30m",
      "created_at": "2025-01-26T10:30:00Z"
    }
  ]
}
```

### Process Approval Step

**Endpoint**: `POST /workflows/{workflow_id}/process-step`  
**Permission**: `ap.approve_invoice` or `ap.approve_payment`

```http
POST /api/v1/core_financials/accounts_payable/workflows/workflow_def456/process-step
Content-Type: application/json

{
  "step_index": 0,
  "action": "approve",
  "comments": "Approved after review",
  "delegate_to": null,
  "request_info": false
}
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "workflow_id": "workflow_def456",
    "step_processed": 0,
    "action_taken": "approve",
    "next_step": 1,
    "workflow_status": "in_progress",
    "processed_at": "2025-01-26T10:30:00Z"
  }
}
```

---

## Analytics APIs

### Cash Flow Forecast

**Endpoint**: `POST /analytics/cash-flow-forecast`  
**Permission**: `ap.analytics`

```http
POST /api/v1/core_financials/accounts_payable/analytics/cash-flow-forecast
Content-Type: application/json

{
  "forecast_horizon_days": 30,
  "confidence_level": 0.90,
  "include_pending_invoices": true,
  "include_planned_payments": true,
  "scenario_analysis": true
}
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "forecast_id": "forecast_pqr678",
    "horizon_days": 30,
    "confidence_score": 0.92,
    "model_version": "cash_flow_v2.1",
    "daily_projections": [
      {
        "date": "2025-01-27",
        "projected_amount": 25000.00,
        "confidence_interval": {
          "lower": 22000.00,
          "upper": 28000.00
        },
        "major_payments": [
          {
            "vendor": "ACME Corporation",
            "amount": 15000.00,
            "probability": 0.95
          }
        ]
      }
    ],
    "feature_importance": {
      "seasonal_patterns": 0.35,
      "vendor_payment_history": 0.28,
      "invoice_aging_distribution": 0.22,
      "economic_indicators": 0.15
    },
    "scenario_analysis": {
      "best_case": 850000.00,
      "most_likely": 750000.00,
      "worst_case": 650000.00
    }
  }
}
```

### Spending Analysis

**Endpoint**: `POST /analytics/spending-analysis`  
**Permission**: `ap.analytics`

```http
POST /api/v1/core_financials/accounts_payable/analytics/spending-analysis
Content-Type: application/json

{
  "analysis_period_days": 90,
  "group_by": ["category", "vendor"],
  "include_trends": true,
  "include_optimization": true,
  "currency": "USD"
}
```

**Response**: `200 OK`
```json
{
  "success": true,
  "data": {
    "analysis_period": {
      "start_date": "2024-10-27",
      "end_date": "2025-01-26",
      "total_spend": 2500000.00
    },
    "category_breakdown": [
      {
        "category": "professional_services",
        "total_amount": 750000.00,
        "percentage": 30.0,
        "vendor_count": 25,
        "trend": "increasing"
      }
    ],
    "vendor_rankings": [
      {
        "vendor_id": "vendor_abc123",
        "vendor_name": "ACME Corporation",
        "total_spend": 150000.00,
        "percentage": 6.0,
        "invoice_count": 45,
        "average_invoice": 3333.33
      }
    ],
    "cost_optimization_opportunities": [
      {
        "type": "vendor_consolidation",
        "description": "Consolidate office supplies vendors",
        "potential_savings": 25000.00,
        "confidence": 0.85
      }
    ]
  }
}
```

### Fraud Risk Analysis

**Endpoint**: `POST /analytics/fraud-risk-analysis`  
**Permission**: `ap.analytics`

```http
POST /api/v1/core_financials/accounts_payable/analytics/fraud-risk-analysis
Content-Type: application/json

{
  "analysis_period_days": 30,
  "include_vendors": true,
  "include_payments": true,
  "risk_threshold": 0.7
}
```

**Response**: `200 OK`  
```json
{
  "success": true,
  "data": {
    "analysis_summary": {
      "total_transactions": 1250,
      "high_risk_count": 15,
      "medium_risk_count": 85,
      "overall_risk_score": 0.23
    },
    "high_risk_transactions": [
      {
        "transaction_id": "payment_xyz123",
        "transaction_type": "payment",
        "risk_score": 0.87,
        "risk_factors": [
          "new_vendor",
          "high_amount",
          "unusual_time"
        ],
        "amount": 75000.00,
        "vendor_name": "Suspicious Vendor Inc"
      }
    ],
    "risk_patterns": [
      {
        "pattern": "off_hours_creation",
        "frequency": 12,
        "average_risk_increase": 0.25
      }
    ],
    "recommendations": [
      "Implement additional approval for payments > $50K to new vendors",
      "Monitor transactions created outside business hours"
    ]
  }
}
```

---

## WebSocket APIs

### Real-Time Notifications

**Connection**: `wss://your-apg-instance.com/ws/ap/notifications`

**Authentication**:
```javascript
const ws = new WebSocket('wss://your-apg-instance.com/ws/ap/notifications', [], {
  headers: {
    'Authorization': 'Bearer your-token',
    'X-APG-Tenant-ID': 'your-tenant-id'
  }
});
```

**Message Types**:

```javascript
// Invoice approval notification
{
  "type": "invoice_approval_required",
  "data": {
    "invoice_id": "invoice_xyz789",
    "invoice_number": "INV-2025-001",
    "vendor_name": "ACME Corporation",
    "amount": 1085.00,
    "approver_id": "user_xyz789",
    "time_limit": "24h"
  },
  "timestamp": "2025-01-26T10:30:00Z"
}

// Payment status update
{
  "type": "payment_status_update",
  "data": {
    "payment_id": "payment_jkl012",
    "old_status": "processing",
    "new_status": "completed",
    "settlement_reference": "ACH240126001"
  },
  "timestamp": "2025-01-26T10:30:00Z"
}

// Fraud alert
{
  "type": "fraud_alert",
  "data": {
    "transaction_id": "payment_xyz123",
    "risk_score": 0.87,
    "risk_factors": ["new_vendor", "high_amount"],
    "action_required": "review_immediately"
  },
  "timestamp": "2025-01-26T10:30:00Z"
}
```

### Real-Time Analytics

**Connection**: `wss://your-apg-instance.com/ws/ap/analytics`

**Subscribe to Cash Flow Updates**:
```javascript
ws.send(JSON.stringify({
  "action": "subscribe",
  "channel": "cash_flow_updates",
  "parameters": {
    "forecast_horizon": 30,
    "update_frequency": "hourly"
  }
}));
```

**Receive Updates**:
```javascript
{
  "channel": "cash_flow_updates",
  "data": {
    "updated_forecast": {
      "date": "2025-01-27",
      "projected_amount": 26500.00,
      "change_from_previous": 1500.00,
      "confidence": 0.91
    }
  },
  "timestamp": "2025-01-26T10:30:00Z"
}
```

---

## Webhook Integration

### Webhook Configuration

**Endpoint**: `POST /webhooks`  
**Permission**: `ap.admin`

```http
POST /api/v1/core_financials/accounts_payable/webhooks
Content-Type: application/json

{
  "url": "https://your-system.com/ap-webhooks",
  "events": [
    "invoice.created",
    "invoice.approved",
    "payment.completed",
    "fraud.alert"
  ],
  "secret": "your-webhook-secret",
  "active": true,
  "retry_config": {
    "max_retries": 3,
    "retry_delay_seconds": 60
  }
}
```

### Webhook Events

**Invoice Created**:
```json
{
  "event": "invoice.created",
  "timestamp": "2025-01-26T10:30:00Z",
  "data": {
    "invoice_id": "invoice_xyz789",
    "invoice_number": "INV-2025-001",
    "vendor_id": "vendor_abc123",
    "total_amount": 1085.00,
    "status": "pending"
  }
}
```

**Payment Completed**:
```json
{
  "event": "payment.completed",
  "timestamp": "2025-01-26T10:30:00Z",
  "data": {
    "payment_id": "payment_jkl012",
    "payment_number": "PAY-2025-001",
    "vendor_id": "vendor_abc123",
    "amount": 1085.00,
    "settlement_date": "2025-01-29",
    "bank_reference": "ACH240126001"
  }
}
```

### Webhook Security

**Signature Verification**:
```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payload, signature, secret) {
  const expectedSignature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return `sha256=${expectedSignature}` === signature;
}

// Express.js example
app.post('/ap-webhooks', (req, res) => {
  const signature = req.headers['x-apg-signature'];
  const payload = JSON.stringify(req.body);
  
  if (!verifyWebhookSignature(payload, signature, process.env.WEBHOOK_SECRET)) {
    return res.status(401).send('Invalid signature');
  }
  
  // Process webhook
  console.log('Webhook received:', req.body);
  res.status(200).send('OK');
});
```

---

## SDK Integration

### JavaScript/Node.js SDK

**Installation**:
```bash
npm install @apg/accounts-payable-sdk
```

**Basic Usage**:
```javascript
const { APAccountsPayableClient } = require('@apg/accounts-payable-sdk');

const client = new APAccountsPayableClient({
  baseUrl: 'https://your-apg-instance.com',
  apiKey: 'your-api-key',
  tenantId: 'your-tenant-id'
});

// Create vendor
const vendor = await client.vendors.create({
  vendor_code: 'ACME001',
  legal_name: 'ACME Corporation',
  // ... other fields
});

// Process invoice with AI
const aiResult = await client.invoices.processWithAI(
  invoiceFile,
  vendorId,
  { confidence_threshold: 0.95 }
);

// Create payment
const payment = await client.payments.create({
  vendor_id: vendorId,
  payment_method: 'ach',
  payment_lines: [
    { invoice_id: invoiceId, payment_amount: 1085.00 }
  ]
});
```

### Python SDK

**Installation**:
```bash
pip install apg-accounts-payable-sdk
```

**Basic Usage**:
```python
from apg_accounts_payable import APAccountsPayableClient

client = APAccountsPayableClient(
    base_url='https://your-apg-instance.com',
    api_key='your-api-key',
    tenant_id='your-tenant-id'
)

# Create vendor
vendor = client.vendors.create({
    'vendor_code': 'ACME001',
    'legal_name': 'ACME Corporation',
    # ... other fields
})

# Get analytics
forecast = client.analytics.cash_flow_forecast(
    horizon_days=30,
    confidence_level=0.90
)
```

---

## Testing Guide

### Test Environment

**Base URL**: `https://test.your-apg-instance.com/api/v1/core_financials/accounts_payable`

### Test Data

**Test Vendor IDs**:
- `test_vendor_001`: Active supplier
- `test_vendor_002`: Inactive contractor
- `test_vendor_999`: High-risk vendor (triggers fraud alerts)

**Test Invoice IDs**:
- `test_invoice_001`: Approved invoice ready for payment
- `test_invoice_002`: Pending approval invoice
- `test_invoice_999`: Invalid invoice (testing error scenarios)

### Example Test Scripts

**JavaScript/Jest**:
```javascript
describe('AP API Tests', () => {
  test('should create vendor successfully', async () => {
    const response = await fetch('/api/v1/core_financials/accounts_payable/vendors', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${testToken}`
      },
      body: JSON.stringify(testVendorData)
    });
    
    expect(response.status).toBe(201);
    const result = await response.json();
    expect(result.success).toBe(true);
    expect(result.data.vendor_code).toBe(testVendorData.vendor_code);
  });
  
  test('should handle AI invoice processing', async () => {
    const formData = new FormData();
    formData.append('file', testInvoiceFile);
    formData.append('vendor_id', 'test_vendor_001');
    
    const response = await fetch('/api/v1/core_financials/accounts_payable/invoices/process-with-ai', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testToken}`
      },
      body: formData
    });
    
    expect(response.status).toBe(202);
    const result = await response.json();
    expect(result.data.status).toBe('processing');
  });
});
```

**Python/pytest**:
```python
import pytest
import requests

class TestAPAPI:
    def test_create_vendor(self, api_client):
        response = api_client.post('/vendors', json={
            'vendor_code': 'TEST001',
            'legal_name': 'Test Vendor Corp',
            # ... other test data
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data['success'] is True
        assert data['data']['vendor_code'] == 'TEST001'
    
    def test_fraud_detection(self, api_client):
        # Create high-risk payment
        response = api_client.post('/payments', json={
            'vendor_id': 'test_vendor_999',  # High-risk test vendor
            'payment_amount': 100000.00,  # High amount
            'payment_method': 'wire'
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data['data']['fraud_score'] > 0.8
```

### Postman Collection

A comprehensive Postman collection is available for API testing:

**Import URL**: `https://your-apg-instance.com/api/postman/ap-collection.json`

**Collection Structure**:
- Authentication tests
- Vendor CRUD operations
- Invoice processing workflows
- Payment creation and tracking
- Analytics and reporting
- Error handling scenarios

---

**Support Information:**
- **API Documentation**: Updated with each release
- **SDK Documentation**: Available in respective package repositories
- **Developer Support**: developer-support@datacraft.co.ke
- **Sandbox Environment**: Available for testing and integration
- **Rate Limit Monitoring**: Available in APG dashboard

**© 2025 Datacraft. All rights reserved.**