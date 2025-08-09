# APG Billing Capability

**Company:** Datacraft  
**Copyright:** © 2025  
**Author:** Nyimbi Odero <nyimbi@gmail.com>  
**Website:** www.datacraft.co.ke

## Overview

The APG Billing capability provides a comprehensive, enterprise-grade billing and subscription management system that integrates seamlessly with all APG platform capabilities. This billing system is designed to be 10x better than industry leaders like Stripe, Chargebee, and Zuora by providing AI-powered billing intelligence, real-time usage tracking, and autonomous revenue optimization.

## Features

### 🎯 Core Features
- **Multi-tenant billing architecture** with complete data isolation
- **Advanced subscription management** with flexible billing models
- **Real-time usage tracking** and aggregation
- **AI-powered billing intelligence** and revenue optimization
- **Comprehensive invoice generation** with customizable templates
- **Payment processing integration** with multiple gateways
- **Advanced analytics and reporting** with predictive insights
- **Tax calculation and compliance** for global operations

### 🤖 AI-Powered Intelligence
- Revenue optimization with machine learning
- Churn prediction and prevention
- Pricing optimization recommendations
- Billing anomaly detection
- Predictive analytics and forecasting

### 🔗 APG Platform Integration
- Seamless integration with all APG capabilities
- Usage tracking from AI orchestration
- Agent-based billing automation
- Real-time collaboration billing
- Federated learning usage metrics

## Quick Start

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Setup Database**
```bash
# Create PostgreSQL database and run schema
psql -d your_database -f schema.sql
```

3. **Initialize Service**
```python
from capabilities.common.billing import get_billing_service

billing_service = get_billing_service()
```

### Basic Usage

#### Create a Customer
```python
customer_data = {
    "name": "Acme Corporation",
    "email": "billing@acme.com",
    "company": "Acme Corp",
    "currency": "USD",
    "billing_address": {
        "street": "123 Business St",
        "city": "Enterprise City",
        "state": "CA",
        "postal_code": "90210",
        "country": "US"
    }
}

customer = await billing_service.create_customer("user-123", customer_data)
```

#### Create a Billing Plan
```python
plan_data = {
    "name": "Professional Plan",
    "description": "For growing businesses",
    "base_price": 99.99,
    "currency": "USD",
    "billing_period": "monthly",
    "features": ["Advanced features", "Priority support", "Analytics"],
    "trial_period_days": 14
}

plan = await billing_service.create_plan("user-123", plan_data)
```

#### Create a Subscription
```python
from capabilities.common.billing.models import CreateSubscriptionRequest

request = CreateSubscriptionRequest(
    customer_id=customer.id,
    plan_id=plan.id,
    trial_period_days=14
)

subscription = await billing_service.create_subscription("user-123", request)
```

#### Submit Usage Data
```python
from capabilities.common.billing.models import UsageSubmissionRequest

usage_request = UsageSubmissionRequest(
    subscription_id=subscription.id,
    metric_name="api_calls",
    quantity=5000
)

usage = await billing_service.submit_usage("user-123", usage_request)
```

#### Generate Invoice
```python
from capabilities.common.billing.models import InvoiceGenerationRequest

invoice_request = InvoiceGenerationRequest(
    subscription_id=subscription.id,
    billing_period_start=subscription.current_period_start,
    billing_period_end=subscription.current_period_end,
    include_usage=True
)

invoice = await billing_service.generate_invoice("user-123", invoice_request)
```

## API Reference

### REST API Endpoints

The billing capability provides a comprehensive REST API:

#### Customers
- `GET /api/v1/billing/customers` - List customers
- `POST /api/v1/billing/customers` - Create customer
- `GET /api/v1/billing/customers/{id}` - Get customer

#### Plans
- `GET /api/v1/billing/plans` - List plans
- `POST /api/v1/billing/plans` - Create plan
- `GET /api/v1/billing/plans/{id}` - Get plan

#### Subscriptions
- `GET /api/v1/billing/subscriptions` - List subscriptions
- `POST /api/v1/billing/subscriptions` - Create subscription
- `GET /api/v1/billing/subscriptions/{id}` - Get subscription
- `PUT /api/v1/billing/subscriptions/{id}` - Update subscription
- `POST /api/v1/billing/subscriptions/{id}/cancel` - Cancel subscription

#### Usage
- `POST /api/v1/billing/usage` - Submit usage
- `GET /api/v1/billing/usage/{subscription_id}/summary` - Get usage summary

#### Invoices
- `GET /api/v1/billing/invoices` - List invoices
- `POST /api/v1/billing/invoices` - Generate invoice
- `GET /api/v1/billing/invoices/{id}` - Get invoice

#### Payments
- `GET /api/v1/billing/payments` - List payments
- `POST /api/v1/billing/payments` - Process payment
- `GET /api/v1/billing/payments/{id}` - Get payment

#### Analytics
- `GET /api/v1/billing/analytics/billing` - Get billing analytics
- `GET /api/v1/billing/analytics/revenue` - Get revenue analytics
- `GET /api/v1/billing/analytics/customers` - Get customer analytics

### Service Methods

#### BillingService Class

```python
class BillingService:
    # Customer Management
    async def create_customer(user_id: str, customer_data: dict) -> BLCustomer
    async def get_customer(user_id: str, customer_id: str) -> BLCustomer
    async def list_customers(user_id: str, filters: dict = None) -> List[BLCustomer]
    
    # Plan Management
    async def create_plan(user_id: str, plan_data: dict) -> BLPlan
    async def get_plan(user_id: str, plan_id: str) -> BLPlan
    
    # Subscription Management
    async def create_subscription(user_id: str, request: CreateSubscriptionRequest) -> BLSubscription
    async def get_subscription(user_id: str, subscription_id: str) -> BLSubscription
    async def update_subscription(user_id: str, subscription_id: str, updates: dict) -> BLSubscription
    async def cancel_subscription(user_id: str, subscription_id: str, cancel_at_period_end: bool, reason: str) -> BLSubscription
    
    # Usage Tracking
    async def submit_usage(user_id: str, request: UsageSubmissionRequest) -> BLUsage
    async def get_usage_summary(user_id: str, subscription_id: str, period_start: datetime, period_end: datetime) -> dict
    
    # Invoice Management
    async def generate_invoice(user_id: str, request: InvoiceGenerationRequest) -> BLInvoice
    async def get_invoice(user_id: str, invoice_id: str) -> BLInvoice
    
    # Payment Processing
    async def process_payment(user_id: str, payment_data: dict) -> BLPayment
    
    # Analytics
    async def get_billing_analytics(user_id: str, tenant_id: str = None, period_start: datetime = None, period_end: datetime = None) -> dict
```

## Data Models

### Core Models

All models use the `BL` prefix following APG naming conventions:

- **BLCustomer** - Customer billing information
- **BLPlan** - Billing plan definitions
- **BLSubscription** - Customer subscriptions
- **BLUsage** - Usage tracking records
- **BLInvoice** - Invoice generation and management
- **BLPayment** - Payment processing records
- **BLPricingRule** - Dynamic pricing rules
- **BLTax** - Tax calculation records
- **BLDiscount** - Discount and promotion management
- **BLRevenue** - Revenue recognition tracking

### Model Relationships

```
BLCustomer 1:N BLSubscription
BLPlan 1:N BLSubscription
BLSubscription 1:N BLUsage
BLSubscription 1:N BLInvoice
BLInvoice 1:N BLPayment
BLInvoice 1:N BLTax
```

## Flask-AppBuilder Integration

The billing capability includes a comprehensive Flask-AppBuilder interface:

### Admin Views
- Customer management with CRUD operations
- Plan configuration and versioning
- Subscription lifecycle management
- Invoice generation and tracking
- Payment processing and monitoring
- Usage analytics and reporting

### Customer Portal
- Self-service billing management
- Usage dashboards and analytics
- Payment method management
- Invoice history and downloads
- Subscription management

### Analytics Dashboards
- Revenue analytics with trends and forecasting
- Customer analytics with segmentation
- Subscription metrics and health
- Usage patterns and optimization

## Advanced Features

### AI-Powered Analytics

```python
from capabilities.common.billing.analytics import get_billing_analytics_engine

analytics_engine = get_billing_analytics_engine()

# Revenue analytics
revenue_analytics = await analytics_engine.get_revenue_analytics("tenant-123")

# Churn prediction
churn_prediction = await analytics_engine.get_churn_prediction("tenant-123")

# Predictive analytics
predictions = await analytics_engine.get_predictive_analytics("tenant-123")
```

### Usage-Based Billing

```python
# Submit usage for API calls
api_usage = UsageSubmissionRequest(
    subscription_id="sub-123",
    metric_name="api_calls",
    quantity=10000,
    metadata={"endpoint": "/api/v1/data", "user_id": "user-456"}
)

# Submit usage for storage
storage_usage = UsageSubmissionRequest(
    subscription_id="sub-123",
    metric_name="storage_gb",
    quantity=250.5,
    metadata={"data_type": "documents"}
)
```

### Dynamic Pricing Rules

```python
# Create tiered pricing rule
pricing_rule_data = {
    "name": "API Calls Tiered Pricing",
    "metric_name": "api_calls",
    "pricing_tiers": [
        {"min": 0, "max": 10000, "price": 0.001},
        {"min": 10001, "max": 100000, "price": 0.0008},
        {"min": 100001, "max": None, "price": 0.0005}
    ],
    "active": True
}
```

### Revenue Recognition

```python
# Automatic revenue recognition for subscriptions
revenue_record = BLRevenue(
    subscription_id="sub-123",
    revenue_amount=Decimal("299.99"),
    recognition_date=datetime.utcnow(),
    revenue_type="subscription",
    accounting_period="2025-01"
)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest capabilities/common/billing/tests/

# Run specific test file
pytest capabilities/common/billing/tests/test_service.py -v

# Run with coverage
pytest capabilities/common/billing/tests/ --cov=capabilities.common.billing
```

### Test Structure

```
tests/
├── __init__.py
├── test_service.py          # Service layer tests
├── test_models.py           # Model validation tests
├── test_api.py              # API endpoint tests
├── test_analytics.py        # Analytics engine tests
├── test_integration.py      # Integration tests
└── fixtures/                # Test data fixtures
```

## Configuration

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/billing_db

# Redis for caching
REDIS_URL=redis://localhost:6379/0

# Payment gateway configuration
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Email configuration for invoices
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=billing@company.com
SMTP_PASSWORD=password
```

### Service Configuration

```python
# config.py
BILLING_CONFIG = {
    "default_currency": "USD",
    "trial_period_days": 14,
    "payment_retry_attempts": 3,
    "invoice_due_days": 30,
    "analytics_cache_ttl": 900,  # 15 minutes
    "enable_ai_features": True
}
```

## Security and Compliance

### Data Protection
- End-to-end encryption for sensitive data
- PCI DSS compliance for payment processing
- GDPR compliance for customer data
- SOX compliance for revenue recognition

### Access Control
- Role-based permissions with Flask-AppBuilder
- Multi-tenant data isolation
- API authentication and authorization
- Audit logging for all operations

### Financial Controls
- Multi-level approval workflows
- Segregation of duties
- Immutable audit trails
- Automated reconciliation

## Performance and Scalability

### Database Optimization
- Comprehensive indexing strategy
- Partitioning for large tables
- Query optimization
- Connection pooling

### Caching Strategy
- Redis caching for analytics
- Query result caching
- Session management
- Rate limiting

### Monitoring
- Real-time metrics collection
- Performance monitoring
- Error tracking and alerting
- Health checks and status reporting

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY capabilities/common/billing/ ./capabilities/common/billing/
EXPOSE 8000

CMD ["python", "-m", "capabilities.common.billing.app"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: billing-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: billing-service
  template:
    metadata:
      labels:
        app: billing-service
    spec:
      containers:
      - name: billing-service
        image: billing-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: billing-secrets
              key: database-url
```

## Monitoring and Alerting

### Health Checks

```python
# Health check endpoint
@app.route('/health')
async def health_check():
    billing_service = get_billing_service()
    status = await billing_service.get_service_status()
    return jsonify(status)
```

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

billing_requests = Counter('billing_requests_total', 'Total billing requests')
billing_duration = Histogram('billing_request_duration_seconds', 'Request duration')
active_subscriptions = Gauge('billing_active_subscriptions', 'Active subscriptions')
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   - Check DATABASE_URL configuration
   - Verify database server is running
   - Check connection pooling settings

2. **Payment Processing Failures**
   - Verify payment gateway credentials
   - Check webhook configurations
   - Review payment method validation

3. **Performance Issues**
   - Monitor database query performance
   - Check Redis cache hit rates
   - Review API response times

### Logging

```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Billing-specific logger
billing_logger = logging.getLogger('billing')
```

## Contributing

### Development Setup

1. **Clone Repository**
```bash
git clone https://github.com/company/apg-platform.git
cd apg-platform/capabilities/common/billing
```

2. **Install Dependencies**
```bash
pip install -r requirements-dev.txt
```

3. **Run Tests**
```bash
pytest tests/ -v
```

4. **Code Quality**
```bash
# Linting
flake8 .

# Type checking
mypy .

# Security scan
bandit -r .
```

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints throughout
- Write comprehensive docstrings
- Include unit tests for all functionality
- Use async/await for I/O operations

## License

© 2025 Datacraft. All rights reserved.

## Support

For support and questions:
- **Email:** nyimbi@gmail.com
- **Website:** www.datacraft.co.ke
- **Documentation:** [APG Billing Docs](https://docs.datacraft.co.ke/billing)

## Changelog

### Version 1.0.0 (2025-01-01)
- Initial release with comprehensive billing functionality
- Multi-tenant subscription management
- Real-time usage tracking
- AI-powered analytics and insights
- Flask-AppBuilder integration
- REST API with full CRUD operations
- Advanced reporting and analytics