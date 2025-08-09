# APG Billing API Reference

## Overview

The APG Billing API provides comprehensive REST endpoints for managing billing operations. The API follows RESTful conventions and returns JSON responses.

## Base URL

```
Production: https://api.yourdomain.com/api/v1/billing
Development: http://localhost:5000/api/v1/billing
```

## Authentication

### API Key Authentication
Include your API key in the request headers:

```bash
curl -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     https://api.yourdomain.com/api/v1/billing/customers
```

### JWT Authentication
For user-specific operations, use JWT tokens:

```bash
curl -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." \
     -H "Content-Type: application/json" \
     https://api.yourdomain.com/api/v1/billing/customers
```

## Response Format

### Success Response
```json
{
  "success": true,
  "data": {
    "id": "cust_12345",
    "name": "John Doe",
    "email": "john@example.com"
  },
  "meta": {
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "The email field is required",
    "details": {
      "field": "email",
      "reason": "missing_required_field"
    }
  },
  "meta": {
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

## Rate Limiting

API requests are rate limited:
- **100 requests per minute** per API key
- **1000 requests per hour** per API key
- **10000 requests per day** per API key

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Pagination

List endpoints support pagination:

### Request Parameters
```bash
GET /api/v1/billing/customers?page=2&limit=25&sort=created_at&order=desc
```

### Response Format
```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 25,
    "total": 150,
    "pages": 6,
    "has_next": true,
    "has_prev": true,
    "next_page": 3,
    "prev_page": 1
  }
}
```

## API Endpoints

### Customers

#### List Customers
```bash
GET /api/v1/billing/customers
```

**Parameters:**
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 25, max: 100)
- `search` (string): Search by name or email
- `status` (string): Filter by status (active, inactive)
- `created_after` (datetime): Filter by creation date
- `created_before` (datetime): Filter by creation date

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "cust_12345",
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "+1234567890",
      "company": "Acme Corp",
      "status": "active",
      "currency": "USD",
      "billing_address": {
        "street": "123 Main St",
        "city": "San Francisco",
        "state": "CA",
        "postal_code": "94105",
        "country": "US"
      },
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

#### Create Customer
```bash
POST /api/v1/billing/customers
```

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890",
  "company": "Acme Corp",
  "tax_id": "12-3456789",
  "currency": "USD",
  "language": "en",
  "timezone": "America/Los_Angeles",
  "billing_address": {
    "street": "123 Main St",
    "city": "San Francisco",
    "state": "CA",
    "postal_code": "94105",
    "country": "US"
  },
  "shipping_address": {
    "street": "456 Oak Ave",
    "city": "San Francisco", 
    "state": "CA",
    "postal_code": "94105",
    "country": "US"
  },
  "metadata": {
    "source": "website_signup",
    "campaign": "winter_2025"
  }
}
```

#### Get Customer
```bash
GET /api/v1/billing/customers/{customer_id}
```

#### Update Customer
```bash
PUT /api/v1/billing/customers/{customer_id}
```

#### Delete Customer
```bash
DELETE /api/v1/billing/customers/{customer_id}
```

### Plans

#### List Plans
```bash
GET /api/v1/billing/plans
```

**Parameters:**
- `active` (boolean): Filter by active status
- `currency` (string): Filter by currency
- `billing_period` (string): Filter by billing period

#### Create Plan
```bash
POST /api/v1/billing/plans
```

**Request Body:**
```json
{
  "name": "Professional Plan",
  "description": "Full-featured plan for growing businesses",
  "amount": 99.99,
  "currency": "USD",
  "billing_period": "monthly",
  "trial_period_days": 14,
  "setup_fee": 50.00,
  "features": ["api_access", "analytics", "priority_support"],
  "usage_based_billing": {
    "enabled": true,
    "billable_metrics": [
      {
        "metric_name": "api_calls",
        "unit_price": 0.01,
        "included_quantity": 10000
      }
    ]
  },
  "pricing_tiers": [
    {
      "up_to": 1000,
      "unit_price": 0.10
    },
    {
      "up_to": 10000,
      "unit_price": 0.08
    },
    {
      "up_to": null,
      "unit_price": 0.05
    }
  ],
  "tax_behavior": "exclusive",
  "active": true
}
```

### Subscriptions

#### List Subscriptions
```bash
GET /api/v1/billing/subscriptions
```

**Parameters:**
- `customer_id` (string): Filter by customer
- `plan_id` (string): Filter by plan
- `status` (string): Filter by status
- `current_period_start` (datetime): Filter by billing period

#### Create Subscription
```bash
POST /api/v1/billing/subscriptions
```

**Request Body:**
```json
{
  "customer_id": "cust_12345",
  "plan_id": "plan_67890",
  "start_date": "2025-01-15",
  "trial_end_date": "2025-01-29",
  "payment_method_id": "pm_12345",
  "billing_cycle_anchor": 1,
  "proration_behavior": "create_prorations",
  "collection_method": "charge_automatically",
  "default_tax_rates": ["txr_12345"],
  "metadata": {
    "source": "api",
    "sales_rep": "john_smith"
  }
}
```

#### Update Subscription
```bash
PUT /api/v1/billing/subscriptions/{subscription_id}
```

#### Cancel Subscription
```bash
POST /api/v1/billing/subscriptions/{subscription_id}/cancel
```

**Request Body:**
```json
{
  "cancellation_reason": "customer_request",
  "cancel_at_period_end": true,
  "prorate": false
}
```

#### Pause Subscription
```bash
POST /api/v1/billing/subscriptions/{subscription_id}/pause
```

**Request Body:**
```json
{
  "pause_behavior": "keep_as_draft",
  "resume_at": "2025-02-15"
}
```

#### Resume Subscription
```bash
POST /api/v1/billing/subscriptions/{subscription_id}/resume
```

### Invoices

#### List Invoices
```bash
GET /api/v1/billing/invoices
```

**Parameters:**
- `customer_id` (string): Filter by customer
- `subscription_id` (string): Filter by subscription
- `status` (string): Filter by status
- `due_date_before` (date): Filter by due date
- `due_date_after` (date): Filter by due date

#### Create Invoice
```bash
POST /api/v1/billing/invoices
```

**Request Body:**
```json
{
  "customer_id": "cust_12345",
  "subscription_id": "sub_67890",
  "description": "Monthly subscription fee",
  "due_date": "2025-02-15",
  "currency": "USD",
  "items": [
    {
      "description": "Professional Plan - Monthly",
      "amount": 99.99,
      "quantity": 1,
      "tax_rates": ["txr_12345"]
    },
    {
      "description": "API Usage Overage",
      "amount": 25.50,
      "quantity": 1
    }
  ],
  "tax_behavior": "exclusive",
  "auto_advance": true,
  "collection_method": "charge_automatically",
  "metadata": {
    "billing_period": "2025-01"
  }
}
```

#### Get Invoice
```bash
GET /api/v1/billing/invoices/{invoice_id}
```

#### Send Invoice
```bash
POST /api/v1/billing/invoices/{invoice_id}/send
```

**Request Body:**
```json
{
  "email_template": "standard_invoice",
  "custom_message": "Thank you for your business!",
  "send_copy_to": ["accounting@company.com"]
}
```

#### Pay Invoice
```bash
POST /api/v1/billing/invoices/{invoice_id}/pay
```

**Request Body:**
```json
{
  "payment_method_id": "pm_12345",
  "amount": 125.49
}
```

#### Void Invoice
```bash
POST /api/v1/billing/invoices/{invoice_id}/void
```

### Payments

#### List Payments
```bash
GET /api/v1/billing/payments
```

#### Create Payment
```bash
POST /api/v1/billing/payments
```

**Request Body:**
```json
{
  "customer_id": "cust_12345",
  "invoice_id": "inv_67890",
  "amount": 99.99,
  "currency": "USD",
  "payment_method": {
    "type": "card",
    "card": {
      "number": "4242424242424242",
      "exp_month": 12,
      "exp_year": 2025,
      "cvc": "123"
    }
  },
  "capture": true,
  "description": "Monthly subscription payment",
  "metadata": {
    "order_id": "order_12345"
  }
}
```

#### Get Payment
```bash
GET /api/v1/billing/payments/{payment_id}
```

#### Refund Payment
```bash
POST /api/v1/billing/payments/{payment_id}/refund
```

**Request Body:**
```json
{
  "amount": 99.99,
  "reason": "requested_by_customer",
  "refund_application_fee": false,
  "metadata": {
    "refund_reason": "service_cancellation"
  }
}
```

### Usage Tracking

#### Track Usage
```bash
POST /api/v1/billing/usage
```

**Request Body:**
```json
{
  "customer_id": "cust_12345",
  "subscription_id": "sub_67890",
  "metric_name": "api_calls",
  "quantity": 150,
  "timestamp": "2025-01-15T10:30:00Z",
  "properties": {
    "endpoint": "/api/v1/users",
    "method": "GET",
    "response_time": 245
  }
}
```

#### Bulk Usage Import
```bash
POST /api/v1/billing/usage/bulk
```

**Request Body:**
```json
{
  "usage_records": [
    {
      "customer_id": "cust_12345",
      "subscription_id": "sub_67890",
      "metric_name": "api_calls",
      "quantity": 100,
      "timestamp": "2025-01-15T09:00:00Z"
    },
    {
      "customer_id": "cust_12345",
      "subscription_id": "sub_67890",
      "metric_name": "storage_gb",
      "quantity": 5.5,
      "timestamp": "2025-01-15T09:00:00Z"
    }
  ]
}
```

#### Get Usage Summary
```bash
GET /api/v1/billing/usage/summary?subscription_id=sub_67890&period_start=2025-01-01&period_end=2025-01-31
```

### Payment Methods

#### List Payment Methods
```bash
GET /api/v1/billing/payment-methods?customer_id=cust_12345
```

#### Create Payment Method
```bash
POST /api/v1/billing/payment-methods
```

**Request Body:**
```json
{
  "customer_id": "cust_12345",
  "type": "card",
  "card": {
    "number": "4242424242424242",
    "exp_month": 12,
    "exp_year": 2025,
    "cvc": "123"
  },
  "billing_details": {
    "name": "John Doe",
    "email": "john@example.com",
    "address": {
      "line1": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "postal_code": "94105",
      "country": "US"
    }
  },
  "set_as_default": true
}
```

### Tax Rates

#### List Tax Rates
```bash
GET /api/v1/billing/tax-rates
```

#### Create Tax Rate
```bash
POST /api/v1/billing/tax-rates
```

**Request Body:**
```json
{
  "display_name": "CA Sales Tax",
  "percentage": 8.75,
  "inclusive": false,
  "jurisdiction": "CA",
  "country": "US",
  "state": "CA",
  "active": true
}
```

### Analytics

#### Revenue Analytics
```bash
GET /api/v1/billing/analytics/revenue?period_start=2025-01-01&period_end=2025-01-31
```

#### Customer Analytics
```bash
GET /api/v1/billing/analytics/customers?period_start=2025-01-01&period_end=2025-01-31
```

#### Subscription Analytics
```bash
GET /api/v1/billing/analytics/subscriptions?period_start=2025-01-01&period_end=2025-01-31
```

## Webhooks

The APG Billing system sends webhooks for important events:

### Event Types
- `customer.created`
- `customer.updated`
- `customer.deleted`
- `subscription.created`
- `subscription.updated`
- `subscription.cancelled`
- `invoice.created`
- `invoice.payment_succeeded`
- `invoice.payment_failed`
- `payment.succeeded`
- `payment.failed`
- `usage.recorded`

### Webhook Format
```json
{
  "id": "evt_12345",
  "type": "invoice.payment_succeeded",
  "created": 1642248000,
  "data": {
    "object": {
      "id": "inv_67890",
      "customer_id": "cust_12345",
      "amount_paid": 99.99,
      "currency": "USD",
      "status": "paid"
    }
  },
  "api_version": "v1"
}
```

### Webhook Security
Verify webhook signatures using the webhook secret:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)
```

## Error Codes

### Common Error Codes
- `AUTHENTICATION_FAILED` - Invalid API key or token
- `AUTHORIZATION_FAILED` - Insufficient permissions
- `INVALID_PARAMETER` - Invalid or missing parameter
- `RESOURCE_NOT_FOUND` - Requested resource not found
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `PAYMENT_FAILED` - Payment processing failed
- `SUBSCRIPTION_CANCELLED` - Operation not allowed on cancelled subscription
- `INVOICE_ALREADY_PAID` - Invoice is already paid
- `INSUFFICIENT_FUNDS` - Customer has insufficient funds
- `CARD_DECLINED` - Credit card was declined

### HTTP Status Codes
- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation errors
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## SDKs and Libraries

### Python SDK
```python
from apg_billing import BillingClient

client = BillingClient(api_key='your_api_key')

# Create customer
customer = client.customers.create({
    'name': 'John Doe',
    'email': 'john@example.com'
})

# Create subscription
subscription = client.subscriptions.create({
    'customer_id': customer.id,
    'plan_id': 'plan_12345'
})
```

### JavaScript SDK
```javascript
const { BillingClient } = require('@datacraft/apg-billing');

const client = new BillingClient('your_api_key');

// Create customer
const customer = await client.customers.create({
  name: 'John Doe',
  email: 'john@example.com'
});

// Create subscription
const subscription = await client.subscriptions.create({
  customer_id: customer.id,
  plan_id: 'plan_12345'
});
```

## Testing

### Test Environment
Use test API keys for development:
```bash
# Test API key format
API_KEY=test_sk_12345abcdef67890
```

### Test Data
The system provides test data for development:
- Test customers with predictable IDs
- Test plans with different billing scenarios
- Test payment methods that simulate various outcomes

### Webhook Testing
Use tools like ngrok for local webhook testing:
```bash
ngrok http 3000
# Use the ngrok URL for webhook endpoints
```

---

© 2025 Datacraft. All rights reserved.