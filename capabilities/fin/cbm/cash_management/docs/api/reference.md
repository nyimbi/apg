# APG Cash Management - API Reference

**Comprehensive REST API Documentation**

© 2025 Datacraft. All rights reserved.
Version: 1.0.0

---

## 🚀 API Overview

The APG Cash Management API provides programmatic access to all cash management functionality through a RESTful interface built with FastAPI. The API supports JSON request/response formats and includes comprehensive error handling, authentication, and rate limiting.

### Base URL
```
Production: https://api.datacraft.co.ke/v1/cash-management
Staging: https://api-staging.datacraft.co.ke/v1/cash-management
```

### API Versioning
- Current Version: `v1`
- Version Header: `Accept: application/vnd.apg.v1+json`
- URL Versioning: `/v1/cash-management`

---

## 🔐 Authentication

### OAuth2 + JWT Authentication

All API endpoints require authentication using OAuth2 with JWT tokens.

#### 1. Obtain Access Token

```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password&grant_type=password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": "read write"
}
```

#### 2. Use Access Token

```http
GET /accounts
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### 3. Refresh Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### API Key Authentication (Alternative)

For server-to-server communication:

```http
GET /accounts
X-API-Key: your-api-key-here
```

---

## 📊 Rate Limiting

API requests are rate-limited to ensure fair usage and system stability.

### Rate Limits
- **Standard Plan**: 1,000 requests/hour
- **Professional Plan**: 10,000 requests/hour  
- **Enterprise Plan**: 100,000 requests/hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
Retry-After: 3600
```

---

## 💰 Cash Accounts API

### List Accounts

Get all cash accounts for the authenticated tenant.

```http
GET /accounts
```

**Query Parameters:**
- `page` (integer, optional): Page number (default: 1)
- `limit` (integer, optional): Items per page (default: 50, max: 100)
- `account_type` (string, optional): Filter by account type
- `bank_id` (string, optional): Filter by bank
- `is_active` (boolean, optional): Filter by active status

**Response:**
```json
{
  "accounts": [
    {
      "id": "acc_1234567890",
      "tenant_id": "tenant_123",
      "account_number": "****6789",
      "account_type": "checking",
      "bank_id": "bank_chase",
      "bank_name": "JPMorgan Chase Bank",
      "current_balance": 125000.50,
      "available_balance": 120000.50,
      "currency": "USD",
      "is_active": true,
      "created_at": "2025-01-15T10:30:00Z",
      "last_updated": "2025-01-27T14:25:30Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 12,
    "pages": 1
  }
}
```

### Get Account Details

```http
GET /accounts/{account_id}
```

**Response:**
```json
{
  "id": "acc_1234567890",
  "tenant_id": "tenant_123",
  "account_number": "1234567890",
  "account_type": "checking",
  "bank_id": "bank_chase",
  "bank_name": "JPMorgan Chase Bank",
  "current_balance": 125000.50,
  "available_balance": 120000.50,
  "pending_credits": 2500.00,
  "pending_debits": 5000.00,
  "currency": "USD",
  "interest_rate": 0.0125,
  "minimum_balance": 1000.00,
  "overdraft_limit": 50000.00,
  "is_active": true,
  "created_at": "2025-01-15T10:30:00Z",
  "last_updated": "2025-01-27T14:25:30Z",
  "metadata": {
    "routing_number": "021000021",
    "swift_code": "CHASUS33",
    "account_manager": "John Smith"
  }
}
```

### Create Account

```http
POST /accounts
Content-Type: application/json

{
  "account_number": "1234567890",
  "account_type": "checking",
  "bank_id": "bank_chase",
  "currency": "USD",
  "initial_balance": 100000.00,
  "minimum_balance": 1000.00,
  "interest_rate": 0.0125,
  "metadata": {
    "routing_number": "021000021",
    "account_manager": "John Smith"
  }
}
```

**Response:**
```json
{
  "id": "acc_1234567890",
  "message": "Account created successfully",
  "created_at": "2025-01-27T14:30:00Z"
}
```

### Update Account

```http
PUT /accounts/{account_id}
Content-Type: application/json

{
  "account_type": "savings",
  "interest_rate": 0.025,
  "minimum_balance": 5000.00,
  "is_active": true
}
```

### Delete Account

```http
DELETE /accounts/{account_id}
```

**Response:**
```json
{
  "message": "Account deleted successfully",
  "deleted_at": "2025-01-27T14:35:00Z"
}
```

---

## 💸 Cash Flows API

### List Cash Flows

```http
GET /cash-flows
```

**Query Parameters:**
- `account_id` (string, optional): Filter by account
- `start_date` (date, optional): Start date (YYYY-MM-DD)
- `end_date` (date, optional): End date (YYYY-MM-DD)
- `category` (string, optional): Filter by category
- `min_amount` (number, optional): Minimum amount filter
- `max_amount` (number, optional): Maximum amount filter
- `page` (integer, optional): Page number
- `limit` (integer, optional): Items per page

**Response:**
```json
{
  "cash_flows": [
    {
      "id": "flow_1234567890",
      "tenant_id": "tenant_123",
      "account_id": "acc_1234567890",
      "amount": 15000.00,
      "transaction_date": "2025-01-27T09:30:00Z",
      "value_date": "2025-01-27T09:30:00Z",
      "description": "Customer payment - Invoice #12345",
      "category": "operating_revenue",
      "counterparty": "ABC Corporation",
      "reference_number": "TXN789012",
      "is_recurring": false,
      "confidence_score": 0.95,
      "source_system": "accounts_receivable",
      "created_at": "2025-01-27T09:35:00Z"
    }
  ],
  "summary": {
    "total_inflows": 125000.00,
    "total_outflows": 89000.00,
    "net_flow": 36000.00,
    "transaction_count": 45
  },
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 127,
    "pages": 3
  }
}
```

### Create Cash Flow

```http
POST /cash-flows
Content-Type: application/json

{
  "account_id": "acc_1234567890",
  "amount": 15000.00,
  "transaction_date": "2025-01-27T09:30:00Z",
  "description": "Customer payment - Invoice #12345",
  "category": "operating_revenue",
  "counterparty": "ABC Corporation",
  "reference_number": "TXN789012",
  "is_recurring": false,
  "metadata": {
    "invoice_number": "INV-12345",
    "payment_method": "wire_transfer"
  }
}
```

### Bulk Import Cash Flows

```http
POST /cash-flows/bulk-import
Content-Type: application/json

{
  "cash_flows": [
    {
      "account_id": "acc_1234567890",
      "amount": 5000.00,
      "transaction_date": "2025-01-27T10:00:00Z",
      "description": "Payment 1"
    },
    {
      "account_id": "acc_1234567890", 
      "amount": -2500.00,
      "transaction_date": "2025-01-27T11:00:00Z",
      "description": "Payment 2"
    }
  ],
  "batch_id": "batch_789012",
  "validate_balances": true
}
```

**Response:**
```json
{
  "batch_id": "batch_789012",
  "total_processed": 2,
  "successful": 2,
  "failed": 0,
  "errors": [],
  "processing_time_ms": 150,
  "created_at": "2025-01-27T15:00:00Z"
}
```

---

## 🔮 Forecasting API

### Generate Forecast

```http
POST /forecasting/generate
Content-Type: application/json

{
  "account_id": "acc_1234567890",
  "forecast_horizon": 30,
  "confidence_level": 0.95,
  "include_scenarios": true,
  "model_type": "ensemble"
}
```

**Response:**
```json
{
  "forecast_id": "forecast_1234567890",
  "account_id": "acc_1234567890",
  "forecast_horizon": 30,
  "confidence_level": 0.95,
  "model_used": "ensemble_v2.1",
  "predictions": [
    {
      "date": "2025-01-28",
      "predicted_amount": 12500.00,
      "confidence_lower": 10200.00,
      "confidence_upper": 14800.00
    },
    {
      "date": "2025-01-29", 
      "predicted_amount": 13200.00,
      "confidence_lower": 10900.00,
      "confidence_upper": 15500.00
    }
  ],
  "scenarios": {
    "base_case": {
      "total_forecast": 385000.00,
      "probability": 0.6
    },
    "optimistic": {
      "total_forecast": 425000.00,
      "probability": 0.2
    },
    "pessimistic": {
      "total_forecast": 345000.00,
      "probability": 0.2
    }
  },
  "model_performance": {
    "historical_accuracy": 0.94,
    "mean_absolute_error": 1250.00,
    "r_squared": 0.89
  },
  "generated_at": "2025-01-27T15:30:00Z"
}
```

### Get Forecast History

```http
GET /forecasting/history
```

**Query Parameters:**
- `account_id` (string, optional): Filter by account
- `start_date` (date, optional): Start date filter
- `end_date` (date, optional): End date filter
- `model_type` (string, optional): Filter by model type

### Forecast Performance Analysis

```http
GET /forecasting/performance
```

**Response:**
```json
{
  "overall_accuracy": 0.94,
  "model_performance": {
    "ensemble_v2.1": {
      "accuracy": 0.96,
      "mean_absolute_error": 1150.00,
      "usage_count": 1250
    },
    "xgboost_v1.5": {
      "accuracy": 0.93,
      "mean_absolute_error": 1380.00,
      "usage_count": 850
    }
  },
  "accuracy_trend": [
    {"month": "2024-12", "accuracy": 0.92},
    {"month": "2025-01", "accuracy": 0.94}
  ]
}
```

---

## ⚖️ Optimization API

### Optimize Cash Allocation

```http
POST /optimization/allocate
Content-Type: application/json

{
  "accounts": [
    {
      "id": "acc_1234567890",
      "current_balance": 125000.00,
      "account_type": "checking",
      "constraints": {
        "min_balance": 10000.00,
        "max_balance": 500000.00
      }
    },
    {
      "id": "acc_0987654321",
      "current_balance": 250000.00,
      "account_type": "savings"
    }
  ],
  "objectives": [
    {
      "type": "maximize_yield",
      "weight": 0.6
    },
    {
      "type": "minimize_risk",
      "weight": 0.4
    }
  ],
  "constraints": [
    {
      "type": "balance_conservation",
      "target_value": 375000.00
    },
    {
      "type": "concentration_limit",
      "max_percentage": 0.6
    }
  ],
  "optimization_method": "multi_objective"
}
```

**Response:**
```json
{
  "optimization_id": "opt_1234567890",
  "success": true,
  "objective_value": 0.0347,
  "execution_time_ms": 2340,
  "optimal_allocation": {
    "acc_1234567890": 150000.00,
    "acc_0987654321": 225000.00
  },
  "allocation_changes": {
    "acc_1234567890": {
      "current": 125000.00,
      "optimal": 150000.00,
      "change": 25000.00,
      "action": "transfer_in"
    },
    "acc_0987654321": {
      "current": 250000.00,
      "optimal": 225000.00,
      "change": -25000.00,
      "action": "transfer_out"
    }
  },
  "recommendations": [
    {
      "priority": 1,
      "action": "Transfer $25,000 from savings to checking account",
      "rationale": "Optimize yield while maintaining liquidity requirements",
      "expected_benefit": "Additional $125/month in interest income"
    }
  ],
  "risk_metrics": {
    "portfolio_var_95": 0.0234,
    "concentration_hhi": 0.52,
    "liquidity_ratio": 0.87
  },
  "confidence_score": 0.89,
  "generated_at": "2025-01-27T16:00:00Z"
}
```

### Get Optimization History

```http
GET /optimization/history
```

### Optimization Performance

```http
GET /optimization/performance
```

---

## 🛡️ Risk Analytics API

### Calculate Risk Metrics

```http
POST /risk/calculate
Content-Type: application/json

{
  "portfolio": {
    "acc_1234567890": {
      "balance": 125000.00,
      "account_type": "checking"
    },
    "acc_0987654321": {
      "balance": 250000.00,
      "account_type": "savings"
    }
  },
  "risk_types": ["var", "expected_shortfall", "liquidity"],
  "confidence_levels": [0.95, 0.99],
  "time_horizons": [1, 10],
  "include_stress_tests": true
}
```

**Response:**
```json
{
  "calculation_id": "risk_calc_1234567890",
  "portfolio_value": 375000.00,
  "calculation_date": "2025-01-27T16:30:00Z",
  "value_at_risk": {
    "var_95_1d": {
      "parametric": 8750.00,
      "historical": 9200.00,
      "monte_carlo": 8950.00
    },
    "var_99_1d": {
      "parametric": 12500.00,
      "historical": 13100.00,
      "monte_carlo": 12800.00
    }
  },
  "expected_shortfall": {
    "es_95_1d": 11200.00,
    "es_99_1d": 15800.00
  },
  "liquidity_metrics": {
    "liquidity_coverage_ratio": 1.25,
    "net_stable_funding_ratio": 1.18,
    "cash_flow_gap_7d": 15000.00,
    "cash_flow_gap_30d": 45000.00
  },
  "performance_ratios": {
    "sharpe_ratio": 1.34,
    "sortino_ratio": 1.67,
    "max_drawdown": -0.085,
    "annual_volatility": 0.145
  },
  "stress_tests": {
    "2008_financial_crisis": {
      "loss_amount": 67500.00,
      "loss_percentage": 18.0,
      "recovery_days": 180
    },
    "covid_pandemic": {
      "loss_amount": 48750.00,
      "loss_percentage": 13.0,
      "recovery_days": 90
    }
  }
}
```

### Run Stress Tests

```http
POST /risk/stress-test
Content-Type: application/json

{
  "portfolio": {
    "acc_1234567890": {"balance": 125000.00},
    "acc_0987654321": {"balance": 250000.00}
  },
  "scenarios": [
    "2008_financial_crisis",
    "covid_pandemic",
    "custom_scenario"
  ],
  "custom_shocks": {
    "equity": -0.30,
    "bonds": -0.10,
    "cash": 0.0
  }
}
```

### Risk Dashboard

```http
GET /risk/dashboard
```

**Response:**
```json
{
  "risk_summary": {
    "overall_risk_score": 34.5,
    "risk_category": "moderate",
    "last_updated": "2025-01-27T16:45:00Z"
  },
  "key_metrics": {
    "portfolio_var_95": 0.0247,
    "liquidity_ratio": 1.25,
    "concentration_hhi": 0.52
  },
  "alerts": [
    {
      "id": "alert_1234567890",
      "type": "concentration_risk",
      "severity": "medium",
      "message": "Single account concentration exceeds 60%",
      "triggered_at": "2025-01-27T15:20:00Z"
    }
  ],
  "compliance_status": {
    "basel_iii": "compliant",
    "lcr_requirement": "compliant",
    "stress_test": "warning"
  }
}
```

---

## 📊 Analytics & Reporting API

### Generate Executive Dashboard

```http
GET /analytics/dashboard
```

**Query Parameters:**
- `date_range` (string): Date range (7d, 30d, 90d, 1y)
- `include_forecasts` (boolean): Include forecast data
- `include_risk_metrics` (boolean): Include risk analytics

**Response:**
```json
{
  "dashboard_id": "dash_1234567890",
  "generated_at": "2025-01-27T17:00:00Z",
  "date_range": "30d",
  "cash_position_summary": {
    "total_cash": 2450000.00,
    "available_cash": 2380000.00,
    "restricted_cash": 70000.00,
    "change_30d": 125000.00,
    "change_percentage": 5.4
  },
  "account_breakdown": [
    {
      "account_type": "checking",
      "total_balance": 450000.00,
      "percentage": 18.4,
      "accounts_count": 3
    },
    {
      "account_type": "savings",
      "total_balance": 1200000.00,
      "percentage": 49.0,
      "accounts_count": 2
    },
    {
      "account_type": "money_market",
      "total_balance": 600000.00,
      "percentage": 24.5,
      "accounts_count": 2
    },
    {
      "account_type": "investment",
      "total_balance": 200000.00,
      "percentage": 8.1,
      "accounts_count": 1
    }
  ],
  "cash_flow_summary": {
    "total_inflows_30d": 1250000.00,
    "total_outflows_30d": 1125000.00,
    "net_flow_30d": 125000.00,
    "average_daily_flow": 4166.67
  },
  "forecast_summary": {
    "next_30d_forecast": 135000.00,
    "confidence_level": 0.95,
    "forecast_accuracy": 0.94
  },
  "risk_summary": {
    "portfolio_var_95": 0.0247,
    "risk_score": 34.5,
    "compliance_status": "compliant"
  },
  "kpis": [
    {
      "name": "Days Cash on Hand",
      "value": 65.2,
      "unit": "days",
      "trend": "up",
      "change": 2.3
    },
    {
      "name": "Cash Utilization Rate",
      "value": 87.5,
      "unit": "percent",
      "trend": "stable",
      "change": 0.1
    }
  ]
}
```

### Custom Reports

```http
POST /analytics/reports
Content-Type: application/json

{
  "report_type": "cash_flow_analysis",
  "parameters": {
    "start_date": "2025-01-01",
    "end_date": "2025-01-31",
    "accounts": ["acc_1234567890", "acc_0987654321"],
    "grouping": "daily",
    "include_forecasts": true
  },
  "format": "json",
  "delivery": {
    "method": "email",
    "recipients": ["finance@company.com"]
  }
}
```

---

## 🔧 System API

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T17:30:00Z",
  "version": "1.0.0",
  "uptime": "72h 15m 30s",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connections": 15
    },
    "cache": {
      "status": "healthy",
      "response_time_ms": 2,
      "memory_usage": "45%"
    },
    "bank_apis": {
      "status": "healthy", 
      "available_banks": 47,
      "avg_response_time_ms": 850
    },
    "ml_models": {
      "status": "healthy",
      "loaded_models": 12,
      "prediction_latency_ms": 35
    }
  }
}
```

### System Metrics

```http
GET /metrics
```

**Response:** Prometheus-format metrics

### Configuration

```http
GET /config
Authorization: Bearer admin_token
```

---

## ❌ Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid account ID format",
    "details": {
      "field": "account_id",
      "received": "invalid-id",
      "expected": "acc_[a-zA-Z0-9]{10}"
    },
    "request_id": "req_1234567890",
    "timestamp": "2025-01-27T18:00:00Z"
  }
}
```

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., duplicate account) |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | System error |
| 503 | Service Unavailable | System maintenance |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `BANK_API_ERROR` | External bank API error |
| `ML_MODEL_ERROR` | Machine learning model error |
| `CALCULATION_ERROR` | Risk calculation error |
| `SYSTEM_ERROR` | Internal system error |

---

## 📚 SDK Examples

### Python SDK

```python
from apg_cash_management import CashManagementClient

# Initialize client
client = CashManagementClient(
    api_key="your-api-key",
    base_url="https://api.datacraft.co.ke/v1/cash-management"
)

# Get accounts
accounts = await client.accounts.list()

# Create cash flow
cash_flow = await client.cash_flows.create({
    "account_id": "acc_1234567890",
    "amount": 15000.00,
    "description": "Customer payment"
})

# Generate forecast
forecast = await client.forecasting.generate({
    "account_id": "acc_1234567890",
    "forecast_horizon": 30
})
```

### JavaScript SDK

```javascript
import { CashManagementClient } from '@datacraft/apg-cash-management';

const client = new CashManagementClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.datacraft.co.ke/v1/cash-management'
});

// Get accounts
const accounts = await client.accounts.list();

// Generate forecast
const forecast = await client.forecasting.generate({
  accountId: 'acc_1234567890',
  forecastHorizon: 30
});
```

### cURL Examples

```bash
# Get accounts
curl -X GET "https://api.datacraft.co.ke/v1/cash-management/accounts" \
  -H "Authorization: Bearer your-access-token" \
  -H "Accept: application/json"

# Create cash flow
curl -X POST "https://api.datacraft.co.ke/v1/cash-management/cash-flows" \
  -H "Authorization: Bearer your-access-token" \
  -H "Content-Type: application/json" \
  -d '{
    "account_id": "acc_1234567890",
    "amount": 15000.00,
    "description": "Customer payment"
  }'
```

---

## 🔗 Webhooks

### Webhook Configuration

```http
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-domain.com/webhooks/cash-management",
  "events": [
    "account.balance_updated",
    "cash_flow.created",
    "risk.alert_triggered"
  ],
  "secret": "your-webhook-secret",
  "retry_policy": {
    "max_attempts": 3,
    "backoff_factor": 2
  }
}
```

### Webhook Events

```json
{
  "event": "account.balance_updated",
  "data": {
    "account_id": "acc_1234567890",
    "old_balance": 120000.00,
    "new_balance": 125000.00,
    "change_amount": 5000.00,
    "updated_at": "2025-01-27T18:30:00Z"
  },
  "webhook_id": "wh_1234567890",
  "timestamp": "2025-01-27T18:30:05Z"
}
```

---

## 📖 Additional Resources

- [OpenAPI Specification](openapi.yaml)
- [Postman Collection](postman-collection.json)
- [SDK Documentation](../sdks/README.md)
- [Rate Limiting Guide](rate-limiting.md)
- [Webhook Reference](webhooks.md)

---

**API Version:** 1.0.0  
**Last Updated:** January 27, 2025  
**Next Review:** April 27, 2025

*© 2025 Datacraft. All rights reserved.*