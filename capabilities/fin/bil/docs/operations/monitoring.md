# Monitoring and Alerting

## Overview

The APG Billing System includes comprehensive monitoring and alerting capabilities to ensure system health, performance, and reliability. This guide covers monitoring setup, alert configuration, and operational procedures.

## System Health Monitoring

### Health Check Endpoints

#### Main Health Check
```bash
GET /billing/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "uptime": "5d 14h 32m",
  "services": {
    "database": "connected",
    "cache": "connected", 
    "payment_processors": {
      "stripe": "configured",
      "paypal": "configured"
    },
    "tax_services": {
      "avalara": "configured",
      "taxjar": "configured"
    },
    "email_services": {
      "sendgrid": "configured"
    }
  },
  "background_tasks": {
    "renewals": "running",
    "invoicing": "running",
    "dunning": "running",
    "revenue_recognition": "running"
  },
  "metrics": {
    "active_customers": 15420,
    "active_subscriptions": 8750,
    "pending_invoices": 342,
    "failed_payments": 15
  }
}
```

#### Detailed Health Check
```bash
GET /billing/health/detailed
```

### Unified Financial Operations Center

The system includes a comprehensive financial operations center for real-time monitoring:

```python
from unified_financial_operations_center import UnifiedFinancialOperationsCenter

# Initialize monitoring
ufoc = UnifiedFinancialOperationsCenter()

# Get real-time dashboard
dashboard = await ufoc.get_operational_dashboard('admin')

# Monitor financial metrics
metrics = await ufoc.get_real_time_metrics([
    'revenue',
    'churn_rate', 
    'payment_failure_rate',
    'customer_acquisition_cost'
])
```

## Key Metrics to Monitor

### Business Metrics

#### Revenue Metrics
```python
# Monthly Recurring Revenue (MRR)
mrr = billing_service.get_mrr()

# Annual Recurring Revenue (ARR) 
arr = billing_service.get_arr()

# Revenue Growth Rate
growth_rate = billing_service.get_revenue_growth_rate()

# Average Revenue Per User (ARPU)
arpu = billing_service.get_arpu()
```

#### Customer Metrics
```python
# Customer Acquisition Cost (CAC)
cac = billing_service.get_cac()

# Customer Lifetime Value (LTV)
ltv = billing_service.get_ltv()

# Churn Rate
churn_rate = billing_service.get_churn_rate()

# Net Revenue Retention
nrr = billing_service.get_net_revenue_retention()
```

#### Operational Metrics
```python
# Payment Success Rate
payment_success_rate = billing_service.get_payment_success_rate()

# Invoice Collection Rate
collection_rate = billing_service.get_collection_rate()

# Dunning Effectiveness
dunning_effectiveness = billing_service.get_dunning_effectiveness()

# Failed Payment Recovery Rate
recovery_rate = billing_service.get_failed_payment_recovery_rate()
```

### Technical Metrics

#### System Performance
- **Response Time**: API endpoint response times
- **Throughput**: Requests per second
- **Error Rate**: 4xx and 5xx error rates
- **Database Performance**: Query execution times
- **Cache Hit Rate**: Redis cache effectiveness

#### Resource Utilization
- **CPU Usage**: Application and database CPU usage
- **Memory Usage**: RAM consumption and garbage collection
- **Disk Usage**: Storage utilization and growth
- **Network I/O**: Bandwidth utilization

#### Background Tasks
- **Queue Length**: Background job queue sizes
- **Processing Time**: Job execution times
- **Failure Rate**: Background job failure rates
- **Retry Attempts**: Number of job retries

## Alerting System

### Alert Channels

#### Email Alerts
```bash
# Configure email recipients
HEALTH_ALERT_EMAILS=ops@company.com,admin@company.com
CRITICAL_ALERT_EMAILS=cto@company.com,oncall@company.com

# Email alert example
Subject: [APG BILLING] Critical Alert - Payment Processor Down
Body: Stripe payment processor has been unavailable for 5 minutes.
      Failed payments: 23
      Revenue impact: $2,340
      
      Please investigate immediately.
```

#### Slack Integration
```bash
# Configure Slack webhook
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Slack alert format
{
  "text": "APG Billing System Alert",
  "attachments": [
    {
      "color": "danger",
      "title": "Payment Processor Failure",
      "fields": [
        {
          "title": "Processor",
          "value": "Stripe",
          "short": true
        },
        {
          "title": "Duration", 
          "value": "5 minutes",
          "short": true
        },
        {
          "title": "Impact",
          "value": "23 failed payments ($2,340)",
          "short": false
        }
      ]
    }
  ]
}
```

#### PagerDuty Integration
```bash
# Configure PagerDuty
PAGERDUTY_INTEGRATION_KEY=your_integration_key

# PagerDuty alert for critical issues
{
  "routing_key": "your_integration_key",
  "event_action": "trigger",
  "dedup_key": "billing_payment_processor_failure",
  "payload": {
    "summary": "APG Billing - Payment Processor Failure",
    "severity": "critical",
    "source": "APG Billing System",
    "component": "payment_processing",
    "custom_details": {
      "processor": "stripe",
      "duration": "5 minutes",
      "failed_payments": 23,
      "revenue_impact": 2340
    }
  }
}
```

### Alert Rules

#### Payment Processing Alerts
```python
# Payment failure rate alert
if payment_failure_rate > 0.05:  # 5% failure rate
    send_alert(
        severity="high",
        message=f"Payment failure rate is {payment_failure_rate:.2%}",
        channels=["email", "slack"]
    )

# Payment processor down
if not payment_processor.is_healthy():
    send_alert(
        severity="critical",
        message=f"Payment processor {processor.name} is down",
        channels=["email", "slack", "pagerduty"]
    )
```

#### Revenue Alerts
```python
# Significant revenue drop
daily_revenue = get_daily_revenue()
if daily_revenue < expected_revenue * 0.8:  # 20% drop
    send_alert(
        severity="high",
        message=f"Daily revenue dropped to ${daily_revenue}",
        channels=["email", "slack"]
    )

# High churn rate
if monthly_churn_rate > 0.1:  # 10% monthly churn
    send_alert(
        severity="medium",
        message=f"Monthly churn rate is {monthly_churn_rate:.2%}",
        channels=["email"]
    )
```

#### System Health Alerts
```python
# Database connection issues
if not database.is_connected():
    send_alert(
        severity="critical",
        message="Database connection lost",
        channels=["email", "slack", "pagerduty"]
    )

# High CPU usage
if cpu_usage > 0.9:  # 90% CPU usage
    send_alert(
        severity="medium",
        message=f"High CPU usage: {cpu_usage:.1%}",
        channels=["email", "slack"]
    )

# Memory usage
if memory_usage > 0.85:  # 85% memory usage
    send_alert(
        severity="medium", 
        message=f"High memory usage: {memory_usage:.1%}",
        channels=["email"]
    )
```

## Logging Strategy

### Log Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General operational information
- **WARNING**: Warning conditions that should be noted
- **ERROR**: Error conditions that need attention
- **CRITICAL**: Critical errors requiring immediate action

### Log Configuration
```python
import logging
import os

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler('logs/apg_billing.log'),
        logging.StreamHandler()
    ]
)

# Service-specific loggers
billing_logger = logging.getLogger('apg_billing.service')
payment_logger = logging.getLogger('apg_billing.payments')
email_logger = logging.getLogger('apg_billing.email')
```

### Structured Logging
```python
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'customer_id'):
            log_data['customer_id'] = record.customer_id
        if hasattr(record, 'transaction_id'):
            log_data['transaction_id'] = record.transaction_id
            
        return json.dumps(log_data)

# Usage
logger = logging.getLogger('apg_billing')
logger.info(
    "Payment processed successfully",
    extra={
        'customer_id': 'cust_12345',
        'payment_id': 'pay_67890',
        'amount': 99.99,
        'currency': 'USD'
    }
)
```

### Log Categories

#### Business Logic Logs
```python
# Customer operations
logger.info(f"Customer created: {customer.id}", extra={'customer_id': customer.id})
logger.info(f"Subscription activated: {subscription.id}", extra={'subscription_id': subscription.id})

# Payment operations
logger.info(f"Payment processed: {payment.id}", extra={'payment_id': payment.id, 'amount': payment.amount})
logger.warning(f"Payment failed: {payment.id}", extra={'payment_id': payment.id, 'error': payment.error})

# Invoice operations
logger.info(f"Invoice generated: {invoice.id}", extra={'invoice_id': invoice.id})
logger.info(f"Invoice sent: {invoice.id}", extra={'invoice_id': invoice.id, 'email': customer.email})
```

#### Integration Logs
```python
# Payment processor logs
logger.info(f"Stripe payment processed: {charge_id}")
logger.error(f"PayPal API error: {error_message}")

# Tax service logs
logger.info(f"Tax calculated via Avalara: {tax_amount}")
logger.warning(f"TaxJar rate limit exceeded")

# Email service logs
logger.info(f"Email sent via SendGrid: {message_id}")
logger.error(f"SMTP delivery failed: {error}")
```

#### Security Logs
```python
# Authentication logs
logger.info(f"API key authenticated: {api_key_prefix}")
logger.warning(f"Invalid API key used: {ip_address}")

# Data access logs
logger.info(f"Customer data accessed: {customer_id}", extra={'user_id': user_id})
logger.warning(f"Unauthorized access attempt: {resource}", extra={'ip_address': ip_address})
```

## Performance Monitoring

### Application Performance Monitoring (APM)

#### Response Time Monitoring
```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metric
            logger.info(
                f"Function executed: {func.__name__}",
                extra={
                    'execution_time': execution_time,
                    'function': func.__name__
                }
            )
            
            # Alert on slow operations
            if execution_time > 5.0:  # 5 seconds
                send_alert(
                    severity="medium",
                    message=f"Slow operation: {func.__name__} took {execution_time:.2f}s"
                )
                
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function failed: {func.__name__}",
                extra={
                    'execution_time': execution_time,
                    'error': str(e)
                }
            )
            raise
    return wrapper

# Usage
@monitor_performance
def process_payment(payment_data):
    # Payment processing logic
    pass
```

#### Database Query Monitoring
```python
import sqlalchemy
from sqlalchemy import event

# Monitor slow queries
@event.listens_for(sqlalchemy.engine.Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = time.time()

@event.listens_for(sqlalchemy.engine.Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - context._query_start_time
    
    if total > 1.0:  # Log queries taking more than 1 second
        logger.warning(
            f"Slow query detected: {total:.2f}s",
            extra={
                'query_time': total,
                'statement': statement[:200]  # First 200 chars
            }
        )
```

### Infrastructure Monitoring

#### System Metrics Collection
```python
import psutil
import time

def collect_system_metrics():
    """Collect system metrics for monitoring"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'network_io': psutil.net_io_counters()._asdict(),
        'process_count': len(psutil.pids()),
        'timestamp': time.time()
    }

# Send metrics to monitoring system
async def send_metrics():
    while True:
        metrics = collect_system_metrics()
        
        # Send to monitoring service (e.g., DataDog, New Relic)
        monitoring_client.send_metrics(metrics)
        
        # Check thresholds
        if metrics['cpu_percent'] > 90:
            send_alert("High CPU usage", severity="medium")
        if metrics['memory_percent'] > 85:
            send_alert("High memory usage", severity="medium")
            
        await asyncio.sleep(60)  # Collect every minute
```

## Monitoring Tools Integration

### DataDog Integration
```python
from datadog import initialize, statsd

# Initialize DataDog
initialize(
    api_key=os.getenv('DATADOG_API_KEY'),
    app_key=os.getenv('DATADOG_APP_KEY')
)

# Send custom metrics
def send_business_metrics():
    # Revenue metrics
    statsd.gauge('apg_billing.mrr', current_mrr)
    statsd.gauge('apg_billing.arr', current_arr)
    statsd.gauge('apg_billing.churn_rate', churn_rate)
    
    # Operational metrics
    statsd.gauge('apg_billing.active_subscriptions', active_subscriptions)
    statsd.gauge('apg_billing.failed_payments', failed_payments)
    statsd.gauge('apg_billing.payment_success_rate', payment_success_rate)
```

### Prometheus Integration
```python
from prometheus_client import Gauge, Counter, Histogram, start_http_server

# Define metrics
mrr_gauge = Gauge('apg_billing_mrr_dollars', 'Monthly Recurring Revenue')
payment_counter = Counter('apg_billing_payments_total', 'Total payments processed', ['status'])
response_time_histogram = Histogram('apg_billing_request_duration_seconds', 'Request duration')

# Update metrics
mrr_gauge.set(current_mrr)
payment_counter.labels(status='success').inc()
payment_counter.labels(status='failed').inc()

# Start Prometheus metrics server
start_http_server(8000)
```

### New Relic Integration
```python
import newrelic.agent

# Initialize New Relic
newrelic.agent.initialize('newrelic.ini')

# Custom metrics
@newrelic.agent.function_trace()
def process_payment(payment_data):
    # Payment processing with New Relic tracing
    with newrelic.agent.BackgroundTask(application, 'payment_processing'):
        # Process payment
        result = payment_processor.process(payment_data)
        
        # Record custom metrics
        newrelic.agent.record_custom_metric('Custom/PaymentAmount', payment_data['amount'])
        newrelic.agent.record_custom_metric('Custom/PaymentSuccess', 1 if result.success else 0)
        
        return result
```

## Operational Procedures

### Daily Operations Checklist
1. **Check System Health**
   - Review health dashboard
   - Check all services are running
   - Verify background tasks are processing

2. **Review Metrics**
   - Daily revenue and MRR
   - Payment failure rates
   - Customer acquisition and churn

3. **Check Alerts**
   - Review overnight alerts
   - Investigate any failures
   - Verify resolutions

4. **Monitor Queues**
   - Check background job queues
   - Look for stuck or failed jobs
   - Monitor processing times

### Weekly Operations Tasks
1. **Performance Review**
   - Analyze response times
   - Review slow queries
   - Check resource utilization

2. **Business Metrics Analysis**
   - Weekly revenue trends
   - Customer acquisition analysis
   - Churn analysis and trends

3. **System Maintenance**
   - Review logs for patterns
   - Update monitoring thresholds
   - Clean up old data

### Monthly Operations Tasks
1. **Capacity Planning**
   - Review growth trends
   - Plan infrastructure scaling
   - Update resource allocations

2. **Security Review**
   - Review access logs
   - Check for anomalies
   - Update security policies

3. **Business Intelligence**
   - Generate monthly reports
   - Analyze business trends
   - Review KPI performance

## Troubleshooting Workflows

### Payment Failure Investigation
1. **Identify the Issue**
   - Check payment processor status
   - Review error messages
   - Check customer payment methods

2. **Assess Impact**
   - Count affected payments
   - Calculate revenue impact
   - Identify affected customers

3. **Remediate**
   - Retry failed payments
   - Contact affected customers
   - Switch payment processors if needed

4. **Follow Up**
   - Monitor resolution
   - Update customers
   - Document lessons learned

### Performance Degradation Response
1. **Identify Bottleneck**
   - Check CPU and memory usage
   - Review database performance
   - Analyze network latency

2. **Immediate Actions**
   - Scale resources if possible
   - Restart services if needed
   - Enable circuit breakers

3. **Long-term Resolution**
   - Optimize slow queries
   - Improve caching
   - Plan capacity upgrades

---

© 2025 Datacraft. All rights reserved.