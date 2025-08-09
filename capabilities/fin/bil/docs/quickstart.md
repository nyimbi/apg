# Quick Start Guide

Get your APG Billing System up and running in 15 minutes with this step-by-step guide.

## Prerequisites

- Python 3.11+
- PostgreSQL or SQLite
- Redis (optional for development)
- Git

## Step 1: Installation (5 minutes)

### Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd apg/capabilities/common/billing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with minimal configuration
nano .env
```

**Minimal .env configuration:**
```bash
# Basic settings
FLASK_ENV=development
APP_SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=sqlite:///quickstart.db
LOG_LEVEL=INFO

# Payment processor (for testing)
STRIPE_PUBLISHABLE_KEY=pk_test_51...
STRIPE_SECRET_KEY=sk_test_51...

# Email (optional)
SENDGRID_API_KEY=SG.your_key_here
```

## Step 2: Database Setup (2 minutes)

```bash
# Initialize the database
python -c "
from service import get_billing_service
service = get_billing_service()
print('Database initialized successfully!')
"
```

## Step 3: Start the Service (1 minute)

```bash
# Start the billing service
python service.py
```

You should see:
```
APG Billing Service starting...
Database connected: sqlite:///quickstart.db
Service running on http://localhost:5000
```

## Step 4: Verify Installation (2 minutes)

### Health Check
```bash
curl http://localhost:5000/billing/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "cache": "not_configured"
  }
}
```

### Access Web Dashboard
Open your browser to: http://localhost:5000/billing/dashboard

## Step 5: Create Your First Data (5 minutes)

### Create a Customer
```bash
curl -X POST http://localhost:5000/api/v1/billing/customers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "billing_address": {
      "street": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "postal_code": "94105",
      "country": "US"
    }
  }'
```

### Create a Billing Plan
```bash
curl -X POST http://localhost:5000/api/v1/billing/plans \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Starter Plan",
    "description": "Perfect for getting started",
    "amount": 29.99,
    "currency": "USD",
    "billing_period": "monthly",
    "trial_period_days": 14
  }'
```

### Create a Subscription
```bash
curl -X POST http://localhost:5000/api/v1/billing/subscriptions \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_...", 
    "plan_id": "plan_...",
    "start_date": "2025-01-15",
    "payment_method": {
      "type": "test_card",
      "card_number": "4242424242424242"
    }
  }'
```

## Step 6: Test Core Features

### Generate an Invoice
```bash
curl -X POST http://localhost:5000/api/v1/billing/invoices/generate \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_...",
    "items": [
      {
        "description": "Starter Plan - Monthly",
        "amount": 29.99,
        "quantity": 1
      }
    ]
  }'
```

### Process a Payment
```bash
curl -X POST http://localhost:5000/api/v1/billing/payments \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_...",
    "invoice_id": "inv_...",
    "amount": 29.99,
    "payment_method": "test_card"
  }'
```

### Track Usage (if applicable)
```bash
curl -X POST http://localhost:5000/api/v1/billing/usage \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "cust_...",
    "subscription_id": "sub_...",
    "metric_name": "api_calls",
    "quantity": 100,
    "timestamp": "2025-01-15T10:30:00Z"
  }'
```

## Common Quick Start Scenarios

### Scenario 1: SaaS Subscription Service

```python
# Create subscription-based billing
from service import get_billing_service

service = get_billing_service()

# Create customer
customer = service.create_customer({
    "name": "Acme Corp",
    "email": "billing@acme.com",
    "company": "Acme Corporation"
})

# Create plan
plan = service.create_plan({
    "name": "Professional Plan",
    "amount": 99.00,
    "currency": "USD",
    "billing_period": "monthly",
    "trial_period_days": 14
})

# Create subscription
subscription = service.create_subscription({
    "customer_id": customer.id,
    "plan_id": plan.id,
    "trial_end_date": "2025-02-01"
})
```

### Scenario 2: Usage-Based Billing

```python
# Create usage-based plan
usage_plan = service.create_plan({
    "name": "Pay-as-you-go",
    "amount": 0.00,  # Base price
    "currency": "USD",
    "billing_period": "monthly",
    "usage_based_billing": {
        "enabled": True,
        "billable_metrics": [
            {
                "metric_name": "api_calls",
                "unit_price": 0.01,
                "included_quantity": 1000
            }
        ]
    }
})

# Track usage
service.track_usage({
    "customer_id": customer.id,
    "subscription_id": subscription.id,
    "metric_name": "api_calls",
    "quantity": 150
})
```

### Scenario 3: E-commerce with Tax Calculation

```python
# Create invoice with tax calculation
invoice = service.create_invoice({
    "customer_id": customer.id,
    "items": [
        {
            "description": "Premium Widget",
            "amount": 299.99,
            "quantity": 2,
            "tax_code": "P0000000"  # Physical goods
        }
    ],
    "calculate_tax": True,
    "shipping_address": {
        "street": "456 Oak Ave",
        "city": "New York",
        "state": "NY",
        "postal_code": "10001",
        "country": "US"
    }
})
```

## Next Steps

### 1. Configure External Services
Set up your production integrations:
- [Payment Processors](configuration.md#payment-processor-configuration)
- [Email Services](configuration.md#email-configuration)
- [Tax Services](configuration.md#tax-service-configuration)

### 2. Explore Advanced Features
- [Dunning Management](modules/dunning-management.md) - Automated collections
- [Revenue Recognition](modules/revenue-recognition.md) - ASC 606 compliance
- [Analytics](modules/analytics.md) - Business intelligence
- [Fraud Detection](modules/fraud-detection.md) - Real-time fraud prevention

### 3. Set Up Monitoring
- [Health Monitoring](operations/monitoring.md)
- [Alerting Setup](operations/monitoring.md#alerting)
- [Performance Metrics](operations/performance.md)

### 4. Production Deployment
- [Deployment Guide](operations/deployment.md)
- [Security Checklist](development/security.md)
- [Scaling Considerations](operations/performance.md#scaling)

## Testing Your Setup

### Automated Test Suite
```bash
# Run basic functionality tests
python -m pytest tests/test_quickstart.py -v

# Run integration tests
python -m pytest tests/integration/ -v
```

### Manual Testing Checklist

- [ ] Service starts without errors
- [ ] Health endpoint returns 200
- [ ] Dashboard loads successfully
- [ ] Customer creation works
- [ ] Plan creation works
- [ ] Subscription creation works
- [ ] Invoice generation works
- [ ] Payment processing works
- [ ] Email notifications work (if configured)

## Troubleshooting

### Service Won't Start
```bash
# Check Python version
python --version  # Should be 3.11+

# Check dependencies
pip list | grep -E "(flask|sqlalchemy)"

# Check database connection
python -c "from service import get_billing_service; get_billing_service()"
```

### API Errors
```bash
# Check logs
tail -f logs/apg_billing.log

# Test API directly
curl -v http://localhost:5000/billing/health
```

### Database Issues
```bash
# Check database file (SQLite)
ls -la *.db

# Check connection string
echo $DATABASE_URL
```

## Getting Help

- **Documentation**: [docs/README.md](README.md)
- **API Reference**: [api/README.md](api/README.md)
- **Examples**: [examples/](../examples/)
- **Support**: nyimbi@gmail.com

## Sample Data

### Load Sample Data
```bash
# Load demo customers, plans, and subscriptions
python scripts/load_sample_data.py
```

This creates:
- 10 sample customers
- 5 billing plans (Free, Starter, Professional, Enterprise, Usage-based)
- 15 active subscriptions
- Sample invoices and payments
- Usage data for testing

### Demo Credentials
```bash
# Admin user
Username: admin@datacraft.co.ke
Password: admin123

# Test customer
Email: demo@example.com
Customer ID: cust_demo_001
```

---

🎉 **Congratulations!** Your APG Billing System is now running. Start exploring the features and customize it for your business needs.

© 2025 Datacraft. All rights reserved.