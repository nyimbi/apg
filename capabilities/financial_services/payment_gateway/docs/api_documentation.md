# APG Payment Gateway API Documentation

## Overview

The APG Payment Gateway is a revolutionary payment processing system that provides seamless integration with multiple payment processors including MPESA, Stripe, PayPal, and Adyen. It features AI-powered fraud detection, instant settlement, and 10 revolutionary features that set it apart from industry leaders.

### Base URL
```
Production: https://payments.datacraft.co.ke/api/v1
Staging: https://staging-payments.datacraft.co.ke/api/v1
```

### Authentication

All API requests require authentication using API keys:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

## Core Payment APIs

### Process Payment

Process a payment transaction through the intelligent payment orchestration system.

**Endpoint:** `POST /payment/process`

**Request Body:**
```json
{
	"amount": 1000.00,
	"currency": "KES",
	"payment_method": {
		"type": "mpesa",
		"phone_number": "254712345678"
	},
	"merchant_id": "merchant_12345",
	"customer": {
		"name": "John Doe",
		"email": "john@example.com",
		"phone": "254712345678"
	},
	"description": "Payment for Order #12345",
	"callback_url": "https://merchant.com/payment/callback",
	"metadata": {
		"order_id": "12345",
		"customer_id": "cust_67890"
	}
}
```

**Response:**
```json
{
	"transaction_id": "txn_abcd1234",
	"status": "processing",
	"processor": "mpesa",
	"estimated_completion": "2025-01-15T10:30:00Z",
	"checkout_url": "https://payments.datacraft.co.ke/checkout/txn_abcd1234",
	"qr_code": "data:image/png;base64,iVBORw0KGgoAAAANSU...",
	"message": "Payment initiated successfully"
}
```

### Get Payment Status

Retrieve the current status of a payment transaction.

**Endpoint:** `GET /payment/{transaction_id}/status`

**Response:**
```json
{
	"transaction_id": "txn_abcd1234",
	"status": "completed",
	"processor": "mpesa",
	"amount": 1000.00,
	"currency": "KES",
	"completed_at": "2025-01-15T10:32:15Z",
	"processor_reference": "OEI2AK4Q16",
	"fees": {
		"processing_fee": 25.00,
		"currency": "KES"
	}
}
```

### Validate Payment Method

Validate payment method details before processing.

**Endpoint:** `POST /payment/validate`

**Request Body:**
```json
{
	"payment_method": {
		"type": "card",
		"card_number": "4242424242424242",
		"expiry_month": 12,
		"expiry_year": 2026,
		"cvv": "123"
	}
}
```

**Response:**
```json
{
	"valid": true,
	"payment_method_id": "pm_1234567890",
	"brand": "visa",
	"last4": "4242",
	"country": "US",
	"funding": "credit"
}
```

## Revolutionary Features APIs

### Zero-Code Integration

Generate integration code for your platform automatically.

**Endpoint:** `POST /integration/generate`

**Request Body:**
```json
{
	"platform": "wordpress",
	"language": "php",
	"features": ["payment_forms", "webhooks", "dashboard"],
	"customization": {
		"theme": "modern",
		"colors": {
			"primary": "#007cba",
			"secondary": "#666666"
		}
	}
}
```

### Predictive Payment Orchestration

Get AI-powered processor recommendations for optimal success rates.

**Endpoint:** `POST /orchestration/predict`

**Request Body:**
```json
{
	"amount": 500.00,
	"currency": "USD",
	"customer_profile": {
		"country": "US",
		"payment_history": "excellent",
		"risk_score": 0.1
	},
	"transaction_context": {
		"time_of_day": "afternoon",
		"day_of_week": "tuesday",
		"merchant_category": "ecommerce"
	}
}
```

**Response:**
```json
{
	"recommendations": [
		{
			"processor": "stripe",
			"success_probability": 0.94,
			"estimated_cost": 2.9,
			"processing_time": "instant"
		},
		{
			"processor": "adyen",
			"success_probability": 0.91,
			"estimated_cost": 2.7,
			"processing_time": "instant"
		}
	],
	"selected_processor": "stripe",
	"reasoning": "Highest success probability for this customer profile"
}
```

### Instant Settlement

Request instant settlement for completed transactions.

**Endpoint:** `POST /settlement/instant`

**Request Body:**
```json
{
	"merchant_id": "merchant_12345",
	"transactions": ["txn_abcd1234", "txn_efgh5678"],
	"settlement_account": {
		"type": "bank_account",
		"account_number": "1234567890",
		"routing_number": "021000021"
	}
}
```

### Real-Time Risk Analysis

Analyze fraud risk for transactions in real-time.

**Endpoint:** `POST /fraud/analyze`

**Request Body:**
```json
{
	"transaction_data": {
		"amount": 1000.00,
		"currency": "USD",
		"customer_ip": "192.168.1.1",
		"device_fingerprint": "fp_abcd1234",
		"payment_method": "card",
		"merchant_id": "merchant_12345"
	},
	"behavioral_data": {
		"session_duration": 300,
		"pages_visited": 5,
		"typing_patterns": "normal"
	}
}
```

**Response:**
```json
{
	"risk_score": 0.15,
	"risk_level": "low",
	"factors": [
		{
			"factor": "customer_history",
			"impact": "positive",
			"weight": 0.3
		},
		{
			"factor": "device_trust",
			"impact": "positive", 
			"weight": 0.2
		}
	],
	"recommendation": "approve",
	"confidence": 0.92
}
```

## Embedded Financial Services

### Cash Advance

Request instant cash advance based on transaction velocity.

**Endpoint:** `POST /financial/cash-advance`

**Request Body:**
```json
{
	"merchant_id": "merchant_12345",
	"requested_amount": 10000.00,
	"currency": "USD",
	"repayment_terms": {
		"method": "percentage_of_sales",
		"percentage": 10.0,
		"duration_days": 90
	}
}
```

### Working Capital Analysis

Get AI-powered working capital insights.

**Endpoint:** `GET /financial/working-capital/{merchant_id}`

**Response:**
```json
{
	"current_position": {
		"cash_on_hand": 25000.00,
		"pending_settlements": 5000.00,
		"projected_inflows_7d": 12000.00
	},
	"recommendations": [
		{
			"type": "optimize_settlement",
			"description": "Switch to instant settlement to improve cash flow by 15%",
			"impact": 1800.00
		}
	],
	"credit_score": 785,
	"max_advance_eligible": 15000.00
}
```

## Global Processing

### Process Multi-Currency Payment

Handle payments with automatic currency conversion.

**Endpoint:** `POST /payment/multi-currency`

**Request Body:**
```json
{
	"amount": 100.00,
	"source_currency": "USD",
	"target_currency": "KES",
	"payment_method": {
		"type": "card",
		"card_token": "tok_visa4242"
	},
	"fx_options": {
		"rate_lock_duration": 300,
		"preferred_rate": 130.50
	}
}
```

## Webhooks

### Webhook Events

The payment gateway sends webhooks for key events:

- `payment.initiated`
- `payment.completed` 
- `payment.failed`
- `fraud.detected`
- `settlement.completed`
- `advance.approved`

**Webhook Payload Example:**
```json
{
	"event": "payment.completed",
	"timestamp": "2025-01-15T10:32:15Z",
	"data": {
		"transaction_id": "txn_abcd1234",
		"amount": 1000.00,
		"currency": "KES",
		"status": "completed",
		"processor": "mpesa",
		"merchant_id": "merchant_12345"
	},
	"signature": "sha256=abc123def456..."
}
```

## Error Handling

### Error Response Format

```json
{
	"error": {
		"code": "INVALID_PAYMENT_METHOD",
		"message": "The provided payment method is invalid",
		"details": {
			"field": "payment_method.card_number",
			"issue": "Invalid card number format"
		},
		"request_id": "req_abcd1234"
	}
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_API_KEY` | API key is missing or invalid |
| `INSUFFICIENT_FUNDS` | Customer has insufficient funds |
| `INVALID_PAYMENT_METHOD` | Payment method details are invalid |
| `TRANSACTION_DECLINED` | Transaction was declined by processor |
| `FRAUD_DETECTED` | Transaction blocked due to fraud detection |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `PROCESSOR_ERROR` | Error from payment processor |
| `SETTLEMENT_FAILED` | Settlement could not be completed |

## Rate Limits

- **Standard:** 100 requests per minute
- **Premium:** 1000 requests per minute  
- **Enterprise:** 10000 requests per minute

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642176000
```

## SDKs and Libraries

### Supported Languages

- **Python:** `pip install apg-payment-gateway`
- **Node.js:** `npm install @datacraft/payment-gateway`
- **PHP:** `composer require datacraft/payment-gateway`
- **Java:** Maven/Gradle dependency available
- **Go:** `go get github.com/datacraft/payment-gateway-go`
- **Ruby:** `gem install apg_payment_gateway`

### Quick Start Example (Python)

```python
from apg_payment_gateway import PaymentGateway

gateway = PaymentGateway(api_key="your_api_key")

payment = gateway.process_payment(
    amount=1000.00,
    currency="KES",
    payment_method={
        "type": "mpesa",
        "phone_number": "254712345678"
    },
    merchant_id="merchant_12345"
)

print(f"Transaction ID: {payment.transaction_id}")
print(f"Status: {payment.status}")
```

## Testing

### Test Environment

Use the staging environment for testing:
```
Base URL: https://staging-payments.datacraft.co.ke/api/v1
```

### Test Cards

For card testing, use these test numbers:

| Card Number | Brand | Result |
|-------------|-------|--------|
| 4242424242424242 | Visa | Success |
| 4000000000000002 | Visa | Declined |
| 4000000000009995 | Visa | Insufficient funds |

### Test MPESA Numbers

- `254700000000` - Success
- `254700000001` - Insufficient funds
- `254700000002` - Invalid number

## Support

- **Documentation:** https://docs.datacraft.co.ke/payment-gateway
- **API Status:** https://status.datacraft.co.ke
- **Support Email:** support@datacraft.co.ke
- **Developer Slack:** https://datacraft-dev.slack.com