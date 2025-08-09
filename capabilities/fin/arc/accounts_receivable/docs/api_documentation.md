# APG Accounts Receivable - API Documentation

**Comprehensive API Reference for AR Capability Integration**

Version 1.0 | © 2025 Datacraft | Author: Nyimbi Odero

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base Configuration](#base-configuration)
4. [Customer API](#customer-api)
5. [Invoice API](#invoice-api)
6. [Payment API](#payment-api)
7. [Collections API](#collections-api)
8. [Analytics API](#analytics-api)
9. [AI Services API](#ai-services-api)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [SDKs and Examples](#sdks-and-examples)

---

## Overview

The APG Accounts Receivable API provides comprehensive programmatic access to all AR functionality, enabling seamless integration with external systems, custom applications, and automated workflows.

### API Characteristics

- **Protocol**: REST over HTTPS
- **Data Format**: JSON
- **Authentication**: Bearer tokens with APG auth integration
- **Rate Limiting**: Configurable per tenant
- **Versioning**: URL-based versioning (`/api/v1/ar/`)
- **Multi-tenant**: Tenant context in headers

### Base URL

```
https://your-apg-instance.com/api/v1/ar/
```

### Common Response Format

```json
{
  "status": "success|error",
  "data": { ... },
  "message": "Optional message",
  "timestamp": "2025-01-20T10:30:00Z",
  "request_id": "req_abc123xyz"
}
```

---

## Authentication

### Bearer Token Authentication

All API requests require a valid Bearer token in the Authorization header:

```http
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### Obtaining Access Tokens

```http
POST /api/auth/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password",
  "tenant_id": "your_tenant_id"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here",
  "scope": ["ar:read", "ar:write", "ar:ai"]
}
```

### Required Permissions

- **ar:read**: Read access to AR data
- **ar:write**: Create and update AR records
- **ar:ai**: Access to AI-powered features
- **ar:admin**: Administrative functions

### Tenant Context

Include tenant information in headers:

```http
X-Tenant-ID: your_tenant_id
X-User-ID: your_user_id
```

---

## Base Configuration

### Request Headers

```http
Content-Type: application/json
Authorization: Bearer YOUR_ACCESS_TOKEN
X-Tenant-ID: your_tenant_id
X-User-ID: your_user_id
Accept: application/json
```

### Pagination Parameters

Most list endpoints support pagination:

```http
GET /api/v1/ar/customers?page=1&per_page=50&sort=created_at&order=desc
```

**Parameters:**
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)
- `sort`: Sort field
- `order`: Sort order (asc/desc)

### Filtering Parameters

Common filter parameters:

- `created_after`: ISO 8601 datetime
- `created_before`: ISO 8601 datetime
- `status`: Status filter
- `search`: Text search across relevant fields

---

## Customer API

### Create Customer

```http
POST /api/v1/ar/customers
```

**Request Body:**
```json
{
  "customer_code": "ACME001",
  "legal_name": "ACME Corporation",
  "display_name": "ACME Corp",
  "customer_type": "CORPORATION",
  "status": "ACTIVE",
  "credit_limit": 50000.00,
  "payment_terms_days": 30,
  "contact_email": "billing@acme.com",
  "contact_phone": "+1-555-123-4567",
  "billing_address": "123 Business St, City, ST 12345",
  "notes": "Preferred customer"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "cust_abc123xyz",
    "customer_code": "ACME001",
    "legal_name": "ACME Corporation",
    "customer_type": "CORPORATION",
    "status": "ACTIVE",
    "credit_limit": 50000.00,
    "total_outstanding": 0.00,
    "overdue_amount": 0.00,
    "created_at": "2025-01-20T10:30:00Z",
    "updated_at": "2025-01-20T10:30:00Z"
  }
}
```

### Get Customer

```http
GET /api/v1/ar/customers/{customer_id}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "cust_abc123xyz",
    "customer_code": "ACME001",
    "legal_name": "ACME Corporation",
    "customer_type": "CORPORATION",
    "status": "ACTIVE",
    "credit_limit": 50000.00,
    "total_outstanding": 15000.00,
    "overdue_amount": 3000.00,
    "payment_terms_days": 30,
    "contact_email": "billing@acme.com",
    "contact_phone": "+1-555-123-4567",
    "billing_address": "123 Business St, City, ST 12345",
    "last_payment_date": "2025-01-15T14:20:00Z",
    "last_payment_amount": 5000.00,
    "created_at": "2025-01-01T10:30:00Z",
    "updated_at": "2025-01-20T10:30:00Z"
  }
}
```

### List Customers

```http
GET /api/v1/ar/customers?page=1&per_page=20&customer_type=CORPORATION&status=ACTIVE
```

**Query Parameters:**
- `customer_type`: INDIVIDUAL|CORPORATION|PARTNERSHIP|GOVERNMENT
- `status`: ACTIVE|INACTIVE|SUSPENDED|CLOSED
- `min_outstanding`: Minimum outstanding amount
- `max_outstanding`: Maximum outstanding amount
- `has_overdue`: true|false
- `search`: Search in customer_code and legal_name

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": "cust_abc123xyz",
      "customer_code": "ACME001",
      "legal_name": "ACME Corporation",
      "total_outstanding": 15000.00,
      "overdue_amount": 3000.00,
      "status": "ACTIVE"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 156,
    "pages": 8
  }
}
```

### Update Customer

```http
PUT /api/v1/ar/customers/{customer_id}
```

**Request Body:**
```json
{
  "credit_limit": 75000.00,
  "contact_email": "new-billing@acme.com",
  "status": "ACTIVE"
}
```

### Get Customer Summary

```http
GET /api/v1/ar/customers/{customer_id}/summary
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "customer": { ... },
    "total_invoices": 25,
    "total_payments": 22,
    "average_payment_days": 28.5,
    "last_payment_date": "2025-01-15",
    "last_payment_amount": 5000.00,
    "credit_utilization": 0.30,
    "payment_history_rating": "GOOD",
    "aging_breakdown": {
      "current": 8000.00,
      "days_1_30": 4000.00,
      "days_31_60": 2000.00,
      "days_61_90": 1000.00,
      "days_90_plus": 0.00
    }
  }
}
```

---

## Invoice API

### Create Invoice

```http
POST /api/v1/ar/invoices
```

**Request Body:**
```json
{
  "customer_id": "cust_abc123xyz",
  "invoice_number": "INV-2025-001",
  "invoice_date": "2025-01-20",
  "due_date": "2025-02-19",
  "total_amount": 10000.00,
  "currency_code": "USD",
  "description": "Professional services for Q1 2025",
  "reference_number": "PO-12345",
  "line_items": [
    {
      "description": "Consulting services",
      "quantity": 40,
      "unit_price": 200.00,
      "amount": 8000.00
    },
    {
      "description": "Software license",
      "quantity": 1,
      "unit_price": 2000.00,
      "amount": 2000.00
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "inv_xyz789abc",
    "invoice_number": "INV-2025-001",
    "customer_id": "cust_abc123xyz",
    "invoice_date": "2025-01-20",
    "due_date": "2025-02-19",
    "total_amount": 10000.00,
    "paid_amount": 0.00,
    "outstanding_amount": 10000.00,
    "status": "DRAFT",
    "payment_status": "UNPAID",
    "currency_code": "USD",
    "created_at": "2025-01-20T10:30:00Z"
  }
}
```

### Get Invoice

```http
GET /api/v1/ar/invoices/{invoice_id}
```

### List Invoices

```http
GET /api/v1/ar/invoices?customer_id=cust_abc123xyz&status=SENT&overdue=true
```

**Query Parameters:**
- `customer_id`: Filter by customer
- `status`: DRAFT|SENT|PAID|OVERDUE|CANCELLED
- `payment_status`: UNPAID|PARTIAL|PAID
- `overdue`: true|false
- `due_before`: Date filter
- `due_after`: Date filter
- `amount_min`: Minimum amount
- `amount_max`: Maximum amount

### Update Invoice

```http
PUT /api/v1/ar/invoices/{invoice_id}
```

### Send Invoice

```http
POST /api/v1/ar/invoices/{invoice_id}/send
```

**Request Body:**
```json
{
  "send_method": "EMAIL",
  "email_addresses": ["billing@customer.com"],
  "message": "Please find attached invoice for review."
}
```

### Mark Invoice Overdue

```http
POST /api/v1/ar/invoices/{invoice_id}/mark-overdue
```

### AI Payment Prediction

```http
POST /api/v1/ar/invoices/{invoice_id}/predict-payment
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "predicted_payment_date": "2025-02-22",
    "confidence_level": 0.85,
    "factors": [
      "Customer has good payment history",
      "Invoice amount within normal range",
      "No seasonal payment delays expected"
    ],
    "risk_indicators": []
  }
}
```

---

## Payment API

### Create Payment

```http
POST /api/v1/ar/payments
```

**Request Body:**
```json
{
  "customer_id": "cust_abc123xyz",
  "payment_reference": "PAY-2025-001",
  "payment_date": "2025-01-20",
  "payment_amount": 8000.00,
  "payment_method": "WIRE_TRANSFER",
  "currency_code": "USD",
  "bank_reference": "WIRE123456789",
  "notes": "Payment for Q4 2024 invoices",
  "invoice_applications": [
    {
      "invoice_id": "inv_xyz789abc",
      "amount": 5000.00
    },
    {
      "invoice_id": "inv_def456ghi",
      "amount": 3000.00
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "id": "pay_mno345pqr",
    "payment_reference": "PAY-2025-001",
    "customer_id": "cust_abc123xyz",
    "payment_date": "2025-01-20",
    "payment_amount": 8000.00,
    "payment_method": "WIRE_TRANSFER",
    "status": "PENDING",
    "currency_code": "USD",
    "created_at": "2025-01-20T10:30:00Z"
  }
}
```

### Get Payment

```http
GET /api/v1/ar/payments/{payment_id}
```

### List Payments

```http
GET /api/v1/ar/payments?customer_id=cust_abc123xyz&status=PROCESSED
```

**Query Parameters:**
- `customer_id`: Filter by customer
- `status`: PENDING|PROCESSING|PROCESSED|FAILED|CANCELLED
- `payment_method`: Filter by payment method
- `date_from`: Start date filter
- `date_to`: End date filter
- `amount_min`: Minimum amount
- `amount_max`: Maximum amount

### Process Payment

```http
POST /api/v1/ar/payments/{payment_id}/process
```

### Apply Payment to Invoices

```http
POST /api/v1/ar/payments/{payment_id}/apply
```

**Request Body:**
```json
{
  "applications": [
    {
      "invoice_id": "inv_xyz789abc",
      "amount": 5000.00
    }
  ]
}
```

---

## Collections API

### Send Payment Reminder

```http
POST /api/v1/ar/collections/{invoice_id}/remind
```

**Request Body:**
```json
{
  "reminder_type": "EMAIL",
  "template": "standard_reminder",
  "custom_message": "Please remit payment at your earliest convenience."
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "success": true,
    "message": "Payment reminder sent successfully",
    "activity_id": "act_stu678vwx",
    "sent_at": "2025-01-20T10:30:00Z"
  }
}
```

### Create Collection Activity

```http
POST /api/v1/ar/collections/activities
```

**Request Body:**
```json
{
  "customer_id": "cust_abc123xyz",
  "invoice_id": "inv_xyz789abc",
  "activity_type": "PHONE_CALL",
  "activity_date": "2025-01-20",
  "priority": "HIGH",
  "contact_method": "PHONE",
  "outcome": "PROMISED_PAYMENT",
  "status": "COMPLETED",
  "notes": "Customer promised payment by Friday",
  "follow_up_date": "2025-01-25",
  "assigned_to": "collector_001"
}
```

### List Collection Activities

```http
GET /api/v1/ar/collections/activities?customer_id=cust_abc123xyz&status=PENDING
```

### Get Collections Metrics

```http
GET /api/v1/ar/collections/metrics
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_activities": 245,
    "successful_activities": 168,
    "success_rate": 0.686,
    "average_resolution_days": 14.2,
    "total_amount_collected": 285000.00,
    "collection_by_method": {
      "EMAIL": {
        "count": 98,
        "success_rate": 0.72,
        "amount": 95000.00
      },
      "PHONE": {
        "count": 85,
        "success_rate": 0.68,
        "amount": 125000.00
      }
    },
    "monthly_trend": [
      {
        "month": "Jan 2025",
        "collected": 45000.00,
        "activities": 52
      }
    ]
  }
}
```

### AI Collections Optimization

```http
POST /api/v1/ar/collections/optimize
```

**Request Body:**
```json
{
  "optimization_scope": "batch",
  "customer_ids": ["cust_abc123xyz", "cust_def456ghi"],
  "scenario_type": "realistic",
  "include_ai_recommendations": true,
  "generate_campaign_plan": true
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_customers": 2,
    "overall_success_probability": 0.72,
    "total_target_amount": 25000.00,
    "estimated_collection_days": 14,
    "strategy_breakdown": {
      "EMAIL_REMINDER": 1,
      "PHONE_CALL": 1
    },
    "customer_strategies": [
      {
        "customer_id": "cust_abc123xyz",
        "customer_name": "ACME Corporation",
        "customer_code": "ACME001",
        "overdue_amount": 15000.00,
        "recommended_strategy": "EMAIL_REMINDER",
        "contact_method": "EMAIL",
        "success_probability": 0.75,
        "priority": "MEDIUM"
      }
    ],
    "insights": [
      "Customers respond well to email communication"
    ],
    "recommendations": [
      "Send reminders during business hours"
    ]
  }
}
```

---

## Analytics API

### Dashboard Metrics

```http
GET /api/v1/ar/analytics/dashboard
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "total_ar_balance": 450000.00,
    "overdue_amount": 125000.00,
    "current_month_sales": 85000.00,
    "current_month_collections": 92000.00,
    "total_customers": 156,
    "active_customers": 142,
    "overdue_customers": 28,
    "average_days_to_pay": 31.5,
    "collection_effectiveness_index": 0.85,
    "days_sales_outstanding": 38.2,
    "ai_assessments_today": 8,
    "ai_collection_recommendations": 15
  }
}
```

### Aging Analysis

```http
GET /api/v1/ar/analytics/aging?as_of_date=2025-01-20
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "as_of_date": "2025-01-20",
    "current": 185000.00,
    "days_1_30": 95000.00,
    "days_31_60": 65000.00,
    "days_61_90": 45000.00,
    "days_90_plus": 60000.00,
    "total_outstanding": 450000.00,
    "aging_buckets": [
      {
        "bucket": "Current",
        "amount": 185000.00,
        "percentage": 41.1,
        "customer_count": 89
      }
    ]
  }
}
```

### Collection Performance

```http
GET /api/v1/ar/analytics/collection-performance
```

### Cash Flow Forecast

```http
POST /api/v1/ar/analytics/cashflow-forecast
```

**Request Body:**
```json
{
  "forecast_start_date": "2025-01-20",
  "forecast_end_date": "2025-04-20",
  "forecast_period": "weekly",
  "scenario_type": "realistic",
  "include_seasonal_trends": true,
  "include_external_factors": true,
  "confidence_level": 0.95
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "forecast_id": "forecast_123",
    "scenario_type": "realistic",
    "forecast_points": [
      {
        "forecast_date": "2025-01-27",
        "expected_collections": 25000.00,
        "invoice_receipts": 18000.00,
        "overdue_collections": 7000.00,
        "total_cash_flow": 50000.00,
        "confidence_interval_lower": 45000.00,
        "confidence_interval_upper": 55000.00
      }
    ],
    "overall_accuracy": 0.92,
    "model_confidence": 0.88,
    "seasonal_factors": [
      "Month-end payment patterns"
    ],
    "risk_factors": [
      "Economic uncertainty"
    ],
    "insights": [
      "Strong payment patterns expected"
    ]
  }
}
```

---

## AI Services API

### Credit Assessment

```http
POST /api/v1/ar/customers/{customer_id}/credit-assessment
```

**Request Body:**
```json
{
  "assessment_type": "comprehensive",
  "include_explanations": true,
  "generate_recommendations": true,
  "update_customer_record": false,
  "notes": "Annual credit review"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "customer_id": "cust_abc123xyz",
    "assessment_date": "2025-01-20",
    "credit_score": 720,
    "risk_level": "MEDIUM",
    "confidence_score": 0.85,
    "recommended_credit_limit": 75000.00,
    "current_credit_limit": 50000.00,
    "predicted_payment_days": 32,
    "payment_reliability": 0.88,
    "positive_factors": [
      "Strong payment history",
      "Stable business operations",
      "Good financial ratios"
    ],
    "risk_factors": [
      "Industry volatility",
      "Recent credit utilization increase"
    ],
    "explanations": [
      "Credit score based on 24-month payment history",
      "Risk level reflects industry and company factors"
    ],
    "recommendations": [
      "Consider increasing credit limit to $75,000",
      "Monitor monthly credit utilization",
      "Schedule quarterly reviews"
    ]
  }
}
```

### Batch Credit Assessment

```http
POST /api/v1/ar/analytics/batch-credit-assessment
```

**Request Body:**
```json
{
  "customer_ids": ["cust_abc123xyz", "cust_def456ghi"],
  "assessment_type": "standard",
  "priority": "normal",
  "email_results": true
}
```

### Credit Risk Monitoring

```http
GET /api/v1/ar/customers/{customer_id}/credit-risk-monitor
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "customer_id": "cust_abc123xyz",
    "score_change": 15,
    "risk_level_change": "IMPROVED",
    "confidence_change": 0.03,
    "monitoring_recommendation": "STANDARD",
    "change_factors": [
      "Improved payment timing",
      "Reduced outstanding balance"
    ],
    "next_assessment_date": "2025-04-20"
  }
}
```

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid customer data provided",
    "details": {
      "field": "credit_limit",
      "reason": "Credit limit must be non-negative"
    }
  },
  "timestamp": "2025-01-20T10:30:00Z",
  "request_id": "req_abc123xyz"
}
```

### HTTP Status Codes

- **200**: Success
- **201**: Created successfully
- **400**: Bad request / Validation error
- **401**: Unauthorized / Invalid token
- **403**: Forbidden / Insufficient permissions
- **404**: Not found
- **409**: Conflict / Duplicate resource
- **422**: Unprocessable entity
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `PERMISSION_DENIED` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `CREDIT_LIMIT_EXCEEDED` | Customer credit limit exceeded |
| `INVALID_CUSTOMER_STATUS` | Customer status invalid for operation |
| `PAYMENT_APPLICATION_FAILED` | Payment could not be applied |
| `AI_SERVICE_UNAVAILABLE` | AI service temporarily unavailable |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

### Error Handling Best Practices

```python
import requests

try:
    response = requests.post(
        "https://api.apg.platform/ar/customers",
        json=customer_data,
        headers=headers
    )
    response.raise_for_status()
    
    result = response.json()
    if result['status'] == 'success':
        customer = result['data']
        print(f"Customer created: {customer['id']}")
    else:
        print(f"Error: {result['error']['message']}")
        
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 422:
        error_detail = e.response.json()
        print(f"Validation error: {error_detail['error']['details']}")
    elif e.response.status_code == 429:
        print("Rate limit exceeded. Please retry after delay.")
    else:
        print(f"HTTP error: {e.response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## Rate Limiting

### Rate Limit Headers

Responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642694400
X-RateLimit-Retry-After: 60
```

### Default Limits

| Endpoint Category | Requests per Hour | Burst Limit |
|-------------------|-------------------|-------------|
| Standard CRUD | 1000 | 50 |
| Search/List | 500 | 25 |
| AI Services | 100 | 10 |
| Analytics | 200 | 15 |
| Bulk Operations | 50 | 5 |

### Rate Limit Best Practices

```python
import time
import requests

def api_request_with_retry(url, data, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 429:
                # Rate limited, check retry-after header
                retry_after = int(response.headers.get('X-RateLimit-Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("Max retries exceeded")
```

---

## SDKs and Examples

### Python SDK Example

```python
from apg_ar_client import ARClient

# Initialize client
client = ARClient(
    base_url="https://api.apg.platform",
    api_key="your_api_key",
    tenant_id="your_tenant_id"
)

# Create customer
customer = client.customers.create({
    "customer_code": "ACME001",
    "legal_name": "ACME Corporation",
    "credit_limit": 50000.00
})

# Create invoice
invoice = client.invoices.create({
    "customer_id": customer.id,
    "total_amount": 10000.00,
    "due_date": "2025-02-19"
})

# AI credit assessment
assessment = client.ai.assess_credit(customer.id)
print(f"Credit score: {assessment.credit_score}")

# Collections optimization
optimization = client.collections.optimize([customer.id])
for strategy in optimization.customer_strategies:
    print(f"Strategy: {strategy.recommended_strategy}")
```

### JavaScript SDK Example

```javascript
import { ARClient } from '@apg/ar-client';

const client = new ARClient({
  baseUrl: 'https://api.apg.platform',
  apiKey: 'your_api_key',
  tenantId: 'your_tenant_id'
});

// Create customer
const customer = await client.customers.create({
  customerCode: 'ACME001',
  legalName: 'ACME Corporation',
  creditLimit: 50000.00
});

// List overdue invoices
const overdueInvoices = await client.invoices.list({
  overdue: true,
  customerId: customer.id
});

// Send payment reminders
for (const invoice of overdueInvoices) {
  await client.collections.sendReminder(invoice.id);
}

// Generate cash flow forecast
const forecast = await client.analytics.cashFlowForecast({
  forecastStartDate: '2025-01-20',
  forecastEndDate: '2025-04-20',
  scenarioType: 'realistic'
});
```

### cURL Examples

```bash
# Create customer
curl -X POST https://api.apg.platform/ar/customers \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "customer_code": "ACME001",
    "legal_name": "ACME Corporation",
    "credit_limit": 50000.00
  }'

# Get dashboard metrics
curl -X GET https://api.apg.platform/ar/analytics/dashboard \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-Tenant-ID: your_tenant_id"

# AI credit assessment
curl -X POST https://api.apg.platform/ar/customers/cust_123/credit-assessment \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: your_tenant_id" \
  -d '{
    "assessment_type": "comprehensive",
    "include_explanations": true
  }'
```

### Webhook Integration

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your_webhook_secret"

@app.route('/ar-webhook', methods=['POST'])
def handle_ar_webhook():
    # Verify webhook signature
    signature = request.headers.get('X-APG-Signature')
    body = request.get_data()
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, f"sha256={expected_signature}"):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Process webhook event
    event = request.json
    event_type = event.get('type')
    
    if event_type == 'invoice.overdue':
        handle_invoice_overdue(event['data'])
    elif event_type == 'payment.processed':
        handle_payment_processed(event['data'])
    elif event_type == 'customer.credit_limit_exceeded':
        handle_credit_limit_exceeded(event['data'])
    
    return jsonify({"status": "processed"})

def handle_invoice_overdue(invoice_data):
    # Trigger automated collections workflow
    print(f"Invoice {invoice_data['invoice_number']} is overdue")

def handle_payment_processed(payment_data):
    # Update internal systems
    print(f"Payment {payment_data['payment_reference']} processed")

def handle_credit_limit_exceeded(customer_data):
    # Alert risk management team
    print(f"Customer {customer_data['customer_code']} exceeded credit limit")
```

---

## Versioning and Deprecation

### API Versioning

- Current version: `v1`
- Version in URL: `/api/v1/ar/`
- Backward compatibility maintained for one major version
- Deprecation notices provided 6 months before removal

### Migration Guides

When upgrading API versions, refer to migration guides:
- [v1 to v2 Migration Guide](migration_v1_to_v2.md)
- [Breaking Changes Log](breaking_changes.md)

---

*For additional API documentation, examples, and support, visit the APG Developer Portal or contact our API support team.*