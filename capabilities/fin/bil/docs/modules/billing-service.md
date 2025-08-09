# Billing Service Module

## Overview

The Billing Service (`service.py`) is the core engine of the APG Billing System. It orchestrates all billing operations, manages the data layer, and provides the primary interface for billing operations across the system.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                Billing Service                      │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐│
│  │   Data Layer    │  │     Business Logic          ││
│  │                 │  │                             ││
│  │ • Customers     │  │ • Subscription Management   ││
│  │ • Plans         │  │ • Invoice Generation        ││
│  │ • Subscriptions │  │ • Payment Processing        ││
│  │ • Invoices      │  │ • Usage Tracking           ││
│  │ • Payments      │  │ • Revenue Recognition      ││
│  │ • Usage Records │  │ • Dunning Management       ││
│  └─────────────────┘  └─────────────────────────────┘│
├─────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────┐│
│  │  Integration    │  │     Background Tasks       ││
│  │     Layer       │  │                             ││
│  │                 │  │ • Renewal Processing        ││
│  │ • Payment       │  │ • Invoice Generation        ││
│  │   Processors    │  │ • Overdue Processing        ││
│  │ • Tax Services  │  │ • Revenue Recognition       ││
│  │ • Email/SMS     │  │ • Health Monitoring         ││
│  │ • Analytics     │  │ • Usage Aggregation         ││
│  └─────────────────┘  └─────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Service Lifecycle

1. **Initialization**: Service setup and background task startup
2. **Data Loading**: Load existing data and establish connections
3. **Event Processing**: Handle billing events and background tasks
4. **Integration Management**: Coordinate with external services
5. **Health Monitoring**: Continuous system health checks

## Core Features

### Customer Management

#### Create Customer
```python
customer_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "company": "Acme Corp",
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
    "tax_id": "12-3456789",
    "currency": "USD",
    "language": "en",
    "timezone": "America/Los_Angeles"
}

customer = billing_service.create_customer(customer_data)
```

#### Update Customer
```python
updated_data = {
    "email": "newemail@example.com",
    "billing_address": {
        "street": "789 New St",
        "city": "Los Angeles",
        "state": "CA",
        "postal_code": "90210",
        "country": "US"
    }
}

customer = billing_service.update_customer(customer_id, updated_data)
```

### Plan Management

#### Create Billing Plan
```python
plan_data = {
    "name": "Professional Plan",
    "description": "Full-featured plan for growing businesses",
    "amount": 99.99,
    "currency": "USD",
    "billing_period": "monthly",  # monthly, quarterly, yearly
    "trial_period_days": 14,
    "setup_fee": 50.00,
    "features": ["api_access", "analytics", "priority_support"],
    "usage_based_billing": {
        "enabled": True,
        "billable_metrics": [
            {
                "metric_name": "api_calls",
                "unit_price": 0.01,
                "included_quantity": 10000,
                "overage_price": 0.015
            },
            {
                "metric_name": "storage_gb",
                "unit_price": 0.50,
                "included_quantity": 100
            }
        ]
    },
    "tax_behavior": "inclusive",  # inclusive, exclusive
    "active": True
}

plan = billing_service.create_plan(plan_data)
```

#### Plan Pricing Tiers
```python
tiered_plan_data = {
    "name": "Usage-Based Plan",
    "description": "Pay for what you use",
    "amount": 0.00,  # Base amount
    "currency": "USD",
    "billing_period": "monthly",
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
            "up_to": None,  # Unlimited
            "unit_price": 0.05
        }
    ]
}

tiered_plan = billing_service.create_plan(tiered_plan_data)
```

### Subscription Management

#### Create Subscription
```python
subscription_data = {
    "customer_id": customer.id,
    "plan_id": plan.id,
    "start_date": "2025-01-15",
    "trial_end_date": "2025-01-29",  # Optional
    "payment_method_id": "pm_12345",
    "billing_cycle_anchor": 1,  # Day of month for billing
    "proration_behavior": "create_prorations",
    "collection_method": "charge_automatically",
    "metadata": {
        "source": "website_signup",
        "campaign": "winter_2025"
    }
}

subscription = billing_service.create_subscription(subscription_data)
```

#### Subscription Lifecycle Operations
```python
# Pause subscription
billing_service.pause_subscription(subscription_id, {
    "pause_behavior": "keep_as_draft",
    "resume_at": "2025-02-15"
})

# Resume subscription
billing_service.resume_subscription(subscription_id)

# Cancel subscription
billing_service.cancel_subscription(subscription_id, {
    "cancellation_reason": "customer_request",
    "cancel_at_period_end": True
})

# Change subscription plan
billing_service.change_subscription_plan(subscription_id, {
    "new_plan_id": new_plan.id,
    "proration_behavior": "create_prorations",
    "effective_date": "2025-02-01"
})
```

### Invoice Management

#### Generate Invoice
```python
invoice_data = {
    "customer_id": customer.id,
    "subscription_id": subscription.id,  # Optional
    "description": "Monthly subscription fee",
    "due_date": "2025-02-15",
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
    "auto_advance": True,
    "collection_method": "charge_automatically"
}

invoice = billing_service.create_invoice(invoice_data)
```

#### Invoice Operations
```python
# Send invoice via email
billing_service.send_invoice(invoice_id, {
    "email_template": "standard_invoice",
    "custom_message": "Thank you for your business!"
})

# Mark invoice as paid (manual)
billing_service.mark_invoice_paid(invoice_id, {
    "payment_method": "bank_transfer",
    "payment_date": "2025-01-20",
    "notes": "Wire transfer received"
})

# Void invoice
billing_service.void_invoice(invoice_id, {
    "reason": "duplicate_invoice"
})

# Generate credit note
credit_note = billing_service.create_credit_note(invoice_id, {
    "amount": 50.00,
    "reason": "service_credit",
    "description": "Refund for service outage"
})
```

### Payment Processing

#### Process Payment
```python
payment_data = {
    "customer_id": customer.id,
    "invoice_id": invoice.id,
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
    "capture": True,
    "description": "Monthly subscription payment"
}

payment = billing_service.process_payment(payment_data)
```

#### Payment Method Management
```python
# Save payment method for future use
payment_method_data = {
    "customer_id": customer.id,
    "type": "card",
    "card": {
        "number": "4242424242424242",
        "exp_month": 12,
        "exp_year": 2025,
        "cvc": "123"
    },
    "set_as_default": True
}

payment_method = billing_service.create_payment_method(payment_method_data)

# Update payment method
billing_service.update_payment_method(payment_method_id, {
    "exp_month": 6,
    "exp_year": 2026
})

# Delete payment method
billing_service.delete_payment_method(payment_method_id)
```

### Usage Tracking

#### Track Usage Events
```python
usage_data = {
    "customer_id": customer.id,
    "subscription_id": subscription.id,
    "metric_name": "api_calls",
    "quantity": 150,
    "timestamp": "2025-01-15T10:30:00Z",
    "properties": {
        "endpoint": "/api/v1/users",
        "method": "GET",
        "response_time": 245
    }
}

usage = billing_service.track_usage(usage_data)
```

#### Bulk Usage Import
```python
bulk_usage = [
    {
        "customer_id": customer.id,
        "subscription_id": subscription.id,
        "metric_name": "api_calls",
        "quantity": 100,
        "timestamp": "2025-01-15T09:00:00Z"
    },
    {
        "customer_id": customer.id,
        "subscription_id": subscription.id,
        "metric_name": "storage_gb",
        "quantity": 5.5,
        "timestamp": "2025-01-15T09:00:00Z"
    }
]

results = billing_service.import_usage_batch(bulk_usage)
```

#### Usage Aggregation
```python
# Get usage summary for billing period
usage_summary = billing_service.get_usage_summary(
    subscription_id=subscription.id,
    period_start="2025-01-01",
    period_end="2025-01-31"
)

# Get real-time usage
current_usage = billing_service.get_current_usage(
    subscription_id=subscription.id,
    metric_name="api_calls"
)
```

## Background Tasks

### Automatic Renewals
The service automatically processes subscription renewals:

```python
async def _process_renewals(self):
    """Process subscription renewals"""
    # Find subscriptions due for renewal
    due_subscriptions = self._get_due_renewals()
    
    for subscription in due_subscriptions:
        try:
            # Generate renewal invoice
            invoice = self._generate_renewal_invoice(subscription)
            
            # Attempt payment
            payment_result = await self._process_subscription_payment(subscription, invoice)
            
            if payment_result.success:
                # Update subscription period
                self._update_subscription_period(subscription)
                
                # Send confirmation email
                await self._send_renewal_confirmation(subscription, invoice)
            else:
                # Handle failed renewal
                await self._handle_failed_renewal(subscription, payment_result)
                
        except Exception as e:
            self.logger.error(f"Renewal processing failed for {subscription.id}: {e}")
```

### Invoice Generation
Automated invoice generation for recurring charges:

```python
async def _generate_recurring_invoices(self):
    """Generate invoices for recurring charges"""
    # Get subscriptions ready for billing
    billing_subscriptions = self._get_subscriptions_for_billing()
    
    for subscription in billing_subscriptions:
        try:
            # Calculate charges for the period
            charges = await self._calculate_period_charges(subscription)
            
            # Create invoice
            invoice = self._create_subscription_invoice(subscription, charges)
            
            # Apply taxes
            await self._apply_taxes(invoice)
            
            # Send to customer
            await self._deliver_invoice(invoice)
            
        except Exception as e:
            self.logger.error(f"Invoice generation failed for {subscription.id}: {e}")
```

### Dunning Management
Automated handling of failed payments:

```python
async def _process_overdue_invoices(self):
    """Process overdue invoices for dunning"""
    overdue_invoices = self._get_overdue_invoices()
    
    for invoice in overdue_invoices:
        try:
            # Check if dunning is already in progress
            if not self._has_active_dunning_case(invoice.id):
                # Trigger dunning process
                await self._trigger_dunning(invoice)
            
        except Exception as e:
            self.logger.error(f"Dunning trigger failed for {invoice.id}: {e}")
```

### Revenue Recognition
Automated revenue recognition processing:

```python
async def _process_revenue_recognition(self):
    """Process revenue recognition according to ASC 606"""
    # Get active subscriptions for revenue recognition
    active_subscriptions = self._get_active_subscriptions()
    
    for subscription in active_subscriptions:
        try:
            # Calculate daily revenue recognition
            recognition_amount = self._calculate_daily_recognition(subscription)
            
            # Create revenue record
            await self._create_revenue_record(subscription, recognition_amount)
            
        except Exception as e:
            self.logger.error(f"Revenue recognition failed for {subscription.id}: {e}")
```

## Integration Points

### Payment Processors
```python
# Initialize payment processors
from .payment_processors import get_payment_manager

payment_manager = get_payment_manager()

# Process payment with automatic processor selection
result = await payment_manager.process_payment_with_fallback(
    payment_data, preferred_processor='stripe'
)
```

### Tax Services
```python
# Initialize tax services
from .tax_services import get_tax_service_manager

tax_manager = get_tax_service_manager()

# Calculate tax for invoice
tax_result = await tax_manager.calculate_tax_with_fallback(
    transaction_data, preferred_service='avalara'
)
```

### Communication Services
```python
# Initialize email services
from .email_services import get_email_service_manager

email_manager = get_email_service_manager()
billing_email = email_manager.get_billing_email_manager()

# Send invoice email
await billing_email.send_invoice_email(
    customer.email, invoice_data
)
```

## Configuration

### Service Configuration
```python
class BillingServiceConfig:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.redis_url = os.getenv('REDIS_URL')
        self.default_currency = os.getenv('DEFAULT_CURRENCY', 'USD')
        self.trial_period_days = int(os.getenv('DEFAULT_TRIAL_DAYS', '14'))
        self.grace_period_days = int(os.getenv('DEFAULT_GRACE_PERIOD_DAYS', '3'))
        self.enable_usage_tracking = os.getenv('USAGE_TRACKING_ENABLED', 'true').lower() == 'true'
        self.enable_tax_calculation = os.getenv('TAX_CALCULATION_ENABLED', 'true').lower() == 'true'
        self.enable_dunning = os.getenv('DUNNING_ENABLED', 'true').lower() == 'true'
        self.enable_revenue_recognition = os.getenv('REVENUE_RECOGNITION_ENABLED', 'true').lower() == 'true'
```

### Service Health Monitoring
```python
def get_service_status(self) -> Dict[str, Any]:
    """Get comprehensive service status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": self._get_uptime(),
        "services": {
            "database": self._check_database_health(),
            "cache": self._check_cache_health(),
            "payment_processors": self._check_payment_processors(),
            "tax_services": self._check_tax_services(),
            "email_services": self._check_email_services()
        },
        "background_tasks": {
            "renewals": self._check_renewal_task(),
            "invoicing": self._check_invoicing_task(),
            "dunning": self._check_dunning_task(),
            "revenue_recognition": self._check_revenue_task()
        },
        "metrics": {
            "active_customers": len(self.customers),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.status == SubscriptionStatus.ACTIVE]),
            "pending_invoices": len([i for i in self.invoices.values() if i.status == InvoiceStatus.OUTSTANDING]),
            "failed_payments": len([p for p in self.payments.values() if p.status == PaymentStatus.FAILED])
        }
    }
```

## Error Handling

### Exception Types
```python
class BillingError(Exception):
    """Base billing error"""
    pass

class SubscriptionError(BillingError):
    """Subscription-related errors"""
    pass

class PaymentError(BillingError):
    """Payment processing errors"""
    pass

class InvoiceError(BillingError):
    """Invoice processing errors"""
    pass

class UsageError(BillingError):
    """Usage tracking errors"""
    pass
```

### Error Recovery
```python
def handle_service_error(self, error: Exception, context: Dict[str, Any]):
    """Handle service errors with appropriate recovery"""
    if isinstance(error, PaymentError):
        # Retry payment with different processor
        return self._retry_payment_with_fallback(context)
    elif isinstance(error, InvoiceError):
        # Queue invoice for manual review
        return self._queue_invoice_for_review(context)
    elif isinstance(error, UsageError):
        # Buffer usage data for retry
        return self._buffer_usage_for_retry(context)
    else:
        # Log and alert for unknown errors
        self.logger.error(f"Unknown billing error: {error}", extra=context)
        return False
```

## Performance Optimization

### Caching Strategy
```python
def _get_cached_result(self, cache_key: str) -> Optional[Any]:
    """Get cached result with TTL check"""
    if cache_key in self.cache:
        cached_data = self.cache[cache_key]
        if self._is_cache_valid(cached_data):
            return cached_data["result"]
    return None

def _cache_result(self, cache_key: str, result: Any, ttl: int = 3600):
    """Cache result with TTL"""
    self.cache[cache_key] = {
        "result": result,
        "timestamp": datetime.utcnow(),
        "ttl": ttl
    }
```

### Batch Processing
```python
async def process_batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process operations in batches for efficiency"""
    results = []
    batch_size = 100
    
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        batch_results = await self._process_operation_batch(batch)
        results.extend(batch_results)
    
    return results
```

## Testing

### Unit Testing
```python
import pytest
from service import BillingService

@pytest.fixture
def billing_service():
    return BillingService(test_mode=True)

def test_create_customer(billing_service):
    customer_data = {
        "name": "Test Customer",
        "email": "test@example.com"
    }
    
    customer = billing_service.create_customer(customer_data)
    
    assert customer.id is not None
    assert customer.name == "Test Customer"
    assert customer.email == "test@example.com"

def test_subscription_lifecycle(billing_service):
    # Create customer and plan
    customer = billing_service.create_customer({"name": "Test", "email": "test@example.com"})
    plan = billing_service.create_plan({"name": "Test Plan", "amount": 10.00})
    
    # Create subscription
    subscription = billing_service.create_subscription({
        "customer_id": customer.id,
        "plan_id": plan.id
    })
    
    assert subscription.status == SubscriptionStatus.ACTIVE
    
    # Cancel subscription
    billing_service.cancel_subscription(subscription.id)
    
    updated_subscription = billing_service.get_subscription(subscription.id)
    assert updated_subscription.status == SubscriptionStatus.CANCELLED
```

---

© 2025 Datacraft. All rights reserved.