# Getting Started with APG Notification System

Get your APG Notification System up and running in minutes with this comprehensive quick start guide.

## 🎯 Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** with asyncio support
- **PostgreSQL 15+** for data storage
- **Redis 7+** for caching and real-time features
- **Docker** (optional, for containerized deployment)
- **Valid APG License** (contact sales@datacraft.co.ke)

## 🚀 5-Minute Quick Start

### Step 1: Installation

```bash
# Install the core package
pip install apg-notification

# Install with all optional dependencies
pip install apg-notification[all]

# Or install from source
git clone https://github.com/datacraft/apg-notification.git
cd apg-notification
pip install -e .
```

### Step 2: Environment Setup

Create a `.env` file with your configuration:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/apg_notifications
REDIS_URL=redis://localhost:6379/0

# APG Configuration
APG_TENANT_ID=your-tenant-id
APG_API_KEY=your-api-key
APG_SECRET_KEY=your-secret-key

# Optional: External Service APIs
SENDGRID_API_KEY=your-sendgrid-key
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
```

### Step 3: Initialize the Service

```python
import asyncio
from apg.notification import NotificationService, create_notification_service

# Method 1: Simple initialization
service = create_notification_service(
    tenant_id="your-tenant-id",
    config={
        'database_url': 'postgresql://user:pass@localhost/apg',
        'redis_url': 'redis://localhost:6379/0',
        'api_key': 'your-api-key'
    }
)

# Method 2: Advanced initialization with custom config
from apg.notification.config import NotificationConfig

config = NotificationConfig(
    tenant_id="your-tenant-id",
    database_url="postgresql://user:pass@localhost/apg",
    redis_url="redis://localhost:6379/0",
    
    # Performance settings
    max_concurrent_deliveries=1000,
    rate_limiting_enabled=True,
    cache_ttl=3600,
    
    # Feature flags
    personalization_enabled=True,
    analytics_enabled=True,
    security_enabled=True,
    geofencing_enabled=True
)

service = NotificationService(config)
```

### Step 4: Create Your First Template

```python
# Create a notification template
template_id = await service.create_template(
    name="Welcome Email",
    subject_template="Welcome to {{company_name}}, {{user_name}}! 🎉",
    text_template="""
    Hi {{user_name}},
    
    Welcome to {{company_name}}! We're excited to have you join our community.
    
    Here's what you can expect:
    • Personalized recommendations
    • Regular updates about {{user_interests}}
    • Exclusive offers just for you
    
    Get started: {{onboarding_url}}
    
    Best regards,
    The {{company_name}} Team
    """,
    html_template="""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h1 style="color: #007bff;">Welcome {{user_name}}! 🎉</h1>
        
        <p>We're excited to have you join <strong>{{company_name}}</strong>!</p>
        
        <h3>What's Next?</h3>
        <ul>
            <li>Personalized recommendations</li>
            <li>Regular updates about <em>{{user_interests}}</em></li>
            <li>Exclusive offers just for you</li>
        </ul>
        
        <div style="margin: 20px 0;">
            <a href="{{onboarding_url}}" 
               style="background: #007bff; color: white; padding: 12px 24px; 
                      text-decoration: none; border-radius: 5px;">
                Get Started Now
            </a>
        </div>
        
        <p>Best regards,<br>The {{company_name}} Team</p>
    </body>
    </html>
    """
)

print(f"Template created with ID: {template_id}")
```

### Step 5: Create User Profiles

```python
# Create user profiles for targeted notifications
users = [
    {
        'user_id': 'user_001',
        'email': 'alice@example.com',
        'phone': '+1-555-123-4567',
        'name': 'Alice Johnson',
        'preferences': {
            'email_enabled': True,
            'sms_enabled': True,
            'push_enabled': True,
            'quiet_hours': {'start': 22, 'end': 7}
        },
        'profile_data': {
            'interests': ['technology', 'ai', 'automation'],
            'segment': 'tech_enthusiast',
            'location': 'San Francisco, CA'
        }
    },
    {
        'user_id': 'user_002',
        'email': 'bob@example.com',
        'phone': '+1-555-987-6543',
        'name': 'Bob Smith',
        'preferences': {
            'email_enabled': True,
            'sms_enabled': False,
            'push_enabled': True
        },
        'profile_data': {
            'interests': ['business', 'finance', 'investing'],
            'segment': 'business_professional',
            'location': 'New York, NY'
        }
    }
]

# Create user profiles
for user_data in users:
    success = await service.create_user_profile(**user_data)
    print(f"Created profile for {user_data['name']}: {success}")
```

### Step 6: Send Your First Notifications

```python
from apg.notification.models import NotificationRequest, DeliveryChannel, NotificationPriority

# Send individual notification
result = await service.send_notification(
    NotificationRequest(
        template_id=template_id,
        user_id="user_001",
        channels=[DeliveryChannel.EMAIL, DeliveryChannel.SMS],
        context={
            'user_name': 'Alice Johnson',
            'company_name': 'TechCorp',
            'user_interests': 'AI and automation',
            'onboarding_url': 'https://techcorp.com/onboarding?user=alice'
        },
        priority=NotificationPriority.NORMAL
    )
)

print(f"Notification sent! ID: {result['notification_id']}")
print(f"Delivery results: {result['delivery_results']}")

# Send bulk notifications
bulk_result = await service.send_bulk_notifications(
    template_id=template_id,
    user_ids=["user_001", "user_002"],
    channels=[DeliveryChannel.EMAIL],
    context={
        'company_name': 'TechCorp',
        'onboarding_url': 'https://techcorp.com/onboarding'
    }
)

print(f"Bulk notifications sent! Batch ID: {bulk_result['batch_id']}")
print(f"Success rate: {bulk_result['summary']['success_rate']:.1%}")
```

### Step 7: Track and Analyze Results

```python
# Get notification status
notification_id = result['notification_id']
status = await service.get_notification_status(notification_id)
print(f"Notification status: {status['status']}")
print(f"Delivery details: {status['delivery_results']}")

# Get basic analytics
analytics = await service.get_basic_analytics(
    start_date="2025-01-01",
    end_date="2025-01-31"
)

print("Analytics Summary:")
print(f"  Total sent: {analytics['total_sent']:,}")
print(f"  Delivery rate: {analytics['delivery_rate']:.1%}")
print(f"  Open rate: {analytics['open_rate']:.1%}")
print(f"  Click rate: {analytics['click_rate']:.1%}")
```

## 🎨 Basic Examples

### Email with Personalization

```python
# Enable personalization for better engagement
result = await service.send_notification(
    NotificationRequest(
        template_id=template_id,
        user_id="user_001",
        channels=[DeliveryChannel.EMAIL],
        context={
            'user_name': 'Alice',
            'company_name': 'TechCorp'
        },
        enable_personalization=True,  # Enable AI personalization
        personalization_level="standard"  # or "premium", "enterprise"
    )
)

print(f"Personalized notification sent with quality score: {result.get('personalization_quality', 'N/A')}")
```

### SMS with Retry Logic

```python
# SMS with automatic retry on failure
result = await service.send_notification(
    NotificationRequest(
        template_id=template_id,
        user_id="user_002",
        channels=[DeliveryChannel.SMS],
        context={'user_name': 'Bob', 'company_name': 'TechCorp'},
        retry_config={
            'max_retries': 3,
            'retry_delays': [60, 300, 900]  # 1min, 5min, 15min
        }
    )
)
```

### Scheduled Notifications

```python
from datetime import datetime, timedelta

# Schedule notification for future delivery
scheduled_time = datetime.utcnow() + timedelta(hours=24)

result = await service.send_notification(
    NotificationRequest(
        template_id=template_id,
        user_id="user_001",
        channels=[DeliveryChannel.EMAIL],
        context={'user_name': 'Alice', 'company_name': 'TechCorp'},
        scheduled_at=scheduled_time
    )
)

print(f"Notification scheduled for: {scheduled_time}")
print(f"Status: {result['status']}")  # Should be 'scheduled'
```

### Multi-Channel with Fallbacks

```python
# Multi-channel delivery with intelligent fallbacks
result = await service.send_notification(
    NotificationRequest(
        template_id=template_id,
        user_id="user_001",
        channels=[
            DeliveryChannel.PUSH,      # Try push first
            DeliveryChannel.EMAIL,     # Fallback to email
            DeliveryChannel.SMS        # Final fallback to SMS
        ],
        context={'user_name': 'Alice', 'company_name': 'TechCorp'},
        fallback_strategy="cascade",  # Try channels in order
        max_fallback_attempts=2
    )
)

# Check which channels were used
successful_channels = [
    delivery['channel'] for delivery in result['delivery_results'] 
    if delivery['status'] == 'delivered'
]
print(f"Successfully delivered via: {successful_channels}")
```

## 🔧 Configuration Options

### Basic Configuration

```python
from apg.notification.config import NotificationConfig

config = NotificationConfig(
    tenant_id="your-tenant-id",
    
    # Database
    database_url="postgresql://user:pass@localhost/apg",
    redis_url="redis://localhost:6379/0",
    
    # API Settings
    api_key="your-api-key",
    api_rate_limit=1000,  # requests per minute
    
    # Performance
    max_concurrent_deliveries=500,
    delivery_timeout_seconds=30,
    cache_ttl=3600,
    
    # Features
    personalization_enabled=True,
    analytics_enabled=True,
    security_enabled=True
)
```

### Advanced Configuration

```python
config = NotificationConfig(
    tenant_id="enterprise-tenant",
    
    # High-performance settings
    max_concurrent_deliveries=2000,
    batch_size=100,
    worker_processes=4,
    
    # Security settings
    encryption_enabled=True,
    audit_logging=True,
    compliance_mode="gdpr",  # "gdpr", "ccpa", "hipaa"
    
    # Channel-specific settings
    channel_config={
        'email': {
            'provider': 'sendgrid',
            'rate_limit': 1000,
            'retry_attempts': 3
        },
        'sms': {
            'provider': 'twilio',
            'rate_limit': 100,
            'retry_attempts': 2
        }
    },
    
    # Personalization settings
    personalization_config={
        'service_level': 'enterprise',
        'enable_real_time': True,
        'enable_predictive': True,
        'min_quality_score': 0.8
    },
    
    # Analytics settings
    analytics_config={
        'enable_real_time': True,
        'retention_days': 365,
        'export_formats': ['json', 'csv', 'parquet']
    }
)
```

## 🌍 Environment Setup

### Development Environment

```bash
# .env.development
DATABASE_URL=postgresql://dev:dev@localhost:5432/apg_dev
REDIS_URL=redis://localhost:6379/1
LOG_LEVEL=DEBUG
ENABLE_DEBUG_TOOLBAR=true
MOCK_EXTERNAL_SERVICES=true
```

### Production Environment

```bash
# .env.production
DATABASE_URL=postgresql://prod_user:secure_pass@db.example.com:5432/apg_prod
REDIS_URL=redis://cache.example.com:6379/0
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  apg-notification:
    image: apg/notification:latest
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/apg
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: apg
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 🎯 Next Steps

### Enable Advanced Features

1. **AI Personalization**: [Setup Guide](../personalization/docs/getting-started.md)
   ```python
   # Enable advanced personalization
   service.enable_personalization("enterprise")
   ```

2. **Analytics Dashboard**: [Analytics Guide](analytics.md)
   ```python
   # Get comprehensive analytics
   analytics = await service.get_comprehensive_analytics()
   ```

3. **Security & Compliance**: [Security Guide](security.md)
   ```python
   # Enable GDPR compliance
   service.enable_compliance("gdpr")
   ```

4. **Geofencing**: [Location Services](geofencing.md)
   ```python
   # Enable location-based notifications
   service.enable_geofencing()
   ```

### Integration Guides

- [Flask-AppBuilder Integration](flask-integration.md) - Web admin interface
- [REST API Reference](api-reference.md) - Complete API documentation
- [WebSocket Integration](websocket-api.md) - Real-time features
- [Webhook Setup](webhook-api.md) - Event handling

### Examples & Tutorials

- [Advanced Workflows](examples/advanced-workflows.md) - Complex automation
- [Personalization Examples](examples/personalization.md) - AI-powered content
- [Analytics Dashboards](examples/analytics.md) - Business intelligence
- [Security Implementation](examples/security.md) - Compliance workflows

## 🆘 Troubleshooting

### Common Issues

**Issue**: `DatabaseConnectionError: Could not connect to database`
```bash
# Solution: Check database URL and credentials
psql $DATABASE_URL -c "SELECT 1;"
```

**Issue**: `RedisConnectionError: Redis server not available`
```bash
# Solution: Verify Redis is running
redis-cli ping
```

**Issue**: `TemplateNotFoundError: Template 'xyz' not found`
```python
# Solution: Verify template exists
templates = await service.list_templates()
print([t['id'] for t in templates])
```

**Issue**: `RateLimitExceededError: API rate limit exceeded`
```python
# Solution: Implement backoff or increase limits
config.api_rate_limit = 2000  # Increase limit
```

### Getting Help

1. **Documentation**: Check our [comprehensive docs](README.md)
2. **Examples**: Browse [example implementations](examples/)
3. **Community**: Join our [Discord server](https://discord.gg/datacraft)
4. **Support**: Email [support@datacraft.co.ke](mailto:support@datacraft.co.ke)

---

**Congratulations!** 🎉 You now have a fully functional APG Notification System. 

Ready to explore advanced features? Check out:
- [Personalization Engine](../personalization/docs/README.md) for AI-powered content
- [Analytics Guide](analytics.md) for business intelligence
- [API Reference](api-reference.md) for complete functionality

*Need help? We're here to support your success!*