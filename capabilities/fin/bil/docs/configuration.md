# Configuration Guide

## Environment Variables

The APG Billing System uses environment variables for configuration. All settings can be configured via a `.env` file or system environment variables.

### Core Configuration

#### Application Settings
```bash
# Application Environment
FLASK_ENV=production                    # development, testing, production
FLASK_DEBUG=false                      # Enable debug mode (development only)
APP_SECRET_KEY=your-super-secret-key   # Flask secret key for sessions
LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Server Configuration
HOST=0.0.0.0                          # Bind address
PORT=5000                             # Port number
WORKERS=4                             # Number of worker processes
```

#### Database Configuration
```bash
# Primary Database (PostgreSQL recommended)
DATABASE_URL=postgresql://user:password@localhost:5432/apg_billing
DB_POOL_SIZE=20                       # Connection pool size
DB_MAX_OVERFLOW=10                    # Max overflow connections
DB_POOL_TIMEOUT=30                    # Connection timeout in seconds
DB_POOL_RECYCLE=3600                  # Connection recycle time

# Database Features
DB_ECHO=false                         # Log SQL queries (development only)
DB_AUTOCOMMIT=false                   # Auto-commit transactions
DB_AUTOFLUSH=true                     # Auto-flush changes
```

#### Cache Configuration
```bash
# Redis Cache
REDIS_URL=redis://localhost:6379/0    # Redis connection string
REDIS_PASSWORD=your_redis_password    # Redis password (if required)
REDIS_SSL=false                       # Use SSL for Redis connection

# Cache Settings
CACHE_TTL=3600                        # Default cache TTL in seconds
ANALYTICS_CACHE_TTL=900               # Analytics cache TTL
SESSION_CACHE_TTL=86400               # Session cache TTL
```

### Payment Processor Configuration

#### Stripe
```bash
STRIPE_PUBLISHABLE_KEY=pk_live_...     # Stripe publishable key
STRIPE_SECRET_KEY=sk_live_...          # Stripe secret key
STRIPE_WEBHOOK_SECRET=whsec_...        # Webhook endpoint secret
STRIPE_API_VERSION=2023-10-16          # API version
STRIPE_TIMEOUT=30                      # Request timeout in seconds
```

#### PayPal
```bash
PAYPAL_CLIENT_ID=your_paypal_client_id         # PayPal client ID
PAYPAL_CLIENT_SECRET=your_paypal_secret        # PayPal client secret
PAYPAL_ENVIRONMENT=live                        # sandbox or live
PAYPAL_WEBHOOK_ID=your_webhook_id             # Webhook ID for verification
PAYPAL_TIMEOUT=30                             # Request timeout
```

#### Generic Card Processing
```bash
CARD_PROCESSOR_ENABLED=true           # Enable generic card processing
CARD_PROCESSOR_ENDPOINT=https://api.processor.com
CARD_PROCESSOR_API_KEY=your_api_key
CARD_PROCESSOR_MERCHANT_ID=your_merchant_id
```

### Tax Service Configuration

#### Avalara
```bash
AVALARA_USERNAME=your_avalara_username        # Avalara account username
AVALARA_PASSWORD=your_avalara_password        # Avalara account password
AVALARA_ENVIRONMENT=production                # sandbox or production
AVALARA_COMPANY_CODE=DEFAULT                  # Default company code
AVALARA_TIMEOUT=30                           # Request timeout
```

#### TaxJar
```bash
TAXJAR_API_TOKEN=your_taxjar_token           # TaxJar API token
TAXJAR_ENVIRONMENT=production                # sandbox or production
TAXJAR_TIMEOUT=30                           # Request timeout
```

### Communication Services

#### Email Configuration
```bash
# SendGrid
SENDGRID_API_KEY=SG.your_sendgrid_key       # SendGrid API key
SENDGRID_FROM_EMAIL=noreply@yourcompany.com # Default sender email
SENDGRID_FROM_NAME=APG Billing              # Default sender name

# Amazon SES
AWS_SES_REGION=us-east-1                    # SES region
AWS_SES_ACCESS_KEY=your_aws_key             # AWS access key
AWS_SES_SECRET_KEY=your_aws_secret          # AWS secret key

# SMTP Fallback
SMTP_HOST=smtp.gmail.com                    # SMTP server host
SMTP_PORT=587                               # SMTP server port
SMTP_USERNAME=your_email@gmail.com          # SMTP username
SMTP_PASSWORD=your_app_password             # SMTP password
SMTP_USE_TLS=true                          # Use TLS encryption
```

#### SMS Configuration
```bash
# Twilio
TWILIO_ACCOUNT_SID=your_twilio_sid          # Twilio account SID
TWILIO_AUTH_TOKEN=your_twilio_token         # Twilio auth token
TWILIO_PHONE_NUMBER=+1234567890             # Twilio phone number

# AWS SNS
AWS_SNS_REGION=us-east-1                    # SNS region
AWS_SNS_ACCESS_KEY=your_aws_key             # AWS access key
AWS_SNS_SECRET_KEY=your_aws_secret          # AWS secret key

# Africa's Talking
AFRICAISTALKING_API_KEY=your_at_key         # Africa's Talking API key
AFRICAISTALKING_USERNAME=sandbox            # Username (sandbox or live)
AFRICAISTALKING_SENDER_ID=APG               # Sender ID

# RapidPro
RAPIDPRO_API_TOKEN=your_rapidpro_token      # RapidPro API token
RAPIDPRO_BASE_URL=https://app.rapidpro.io  # RapidPro instance URL
```

#### Push Notification Configuration
```bash
# Firebase Cloud Messaging
FIREBASE_PROJECT_ID=your_firebase_project   # Firebase project ID
FIREBASE_PRIVATE_KEY_ID=your_key_id         # Private key ID
FIREBASE_PRIVATE_KEY=your_private_key       # Private key (base64 encoded)
FIREBASE_CLIENT_EMAIL=firebase@project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your_client_id           # Client ID

# AWS SNS Mobile
AWS_SNS_PLATFORM_APPLICATION_ARN=arn:aws:sns:...  # Platform application ARN

# Apple Push Notification Service
APNS_KEY_ID=your_apns_key_id               # APNs key ID
APNS_TEAM_ID=your_team_id                  # Apple team ID
APNS_BUNDLE_ID=com.yourapp.bundle          # App bundle ID
APNS_PRIVATE_KEY=your_apns_private_key     # APNs private key
APNS_ENVIRONMENT=production                # sandbox or production
```

### Analytics & Marketing

#### Google Services
```bash
# Google Ads API
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token    # Developer token
GOOGLE_ADS_CLIENT_ID=your_client_id                # OAuth client ID
GOOGLE_ADS_CLIENT_SECRET=your_client_secret        # OAuth client secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token        # OAuth refresh token
GOOGLE_ADS_CUSTOMER_ID=1234567890                  # Customer ID

# Google Analytics
GOOGLE_ANALYTICS_MEASUREMENT_ID=G-XXXXXXXXXX      # GA4 measurement ID
GOOGLE_ANALYTICS_API_SECRET=your_api_secret       # Measurement Protocol secret
```

#### Facebook Marketing
```bash
FACEBOOK_ACCESS_TOKEN=your_facebook_token          # Facebook access token
FACEBOOK_APP_ID=your_app_id                       # Facebook app ID
FACEBOOK_APP_SECRET=your_app_secret               # Facebook app secret
FACEBOOK_AD_ACCOUNT_ID=act_1234567890             # Ad account ID
```

### Monitoring & Alerting

#### Slack Integration
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # Slack webhook URL
SLACK_CHANNEL=#billing-alerts                           # Default channel
SLACK_USERNAME=APG Billing Bot                         # Bot username
```

#### PagerDuty Integration
```bash
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key           # Integration key
PAGERDUTY_SERVICE_ID=your_service_id                   # Service ID
```

#### Email Alerts
```bash
HEALTH_ALERT_EMAILS=ops@company.com,admin@company.com  # Alert recipients
CRITICAL_ALERT_EMAILS=cto@company.com                  # Critical alerts
```

### Security & Compliance

#### Encryption
```bash
ENCRYPTION_KEY=your_32_byte_encryption_key             # Fernet encryption key
DATA_ENCRYPTION_ENABLED=true                          # Enable data encryption
AUDIT_ENCRYPTION_ENABLED=true                         # Enable audit encryption
```

#### Authentication
```bash
JWT_SECRET_KEY=your_jwt_secret                         # JWT signing key
JWT_EXPIRATION_HOURS=24                               # JWT expiration time
AUTH_TIMEOUT_MINUTES=30                               # Auth session timeout

# OAuth Providers
GOOGLE_OAUTH_CLIENT_ID=your_google_client_id
GOOGLE_OAUTH_CLIENT_SECRET=your_google_secret
MICROSOFT_OAUTH_CLIENT_ID=your_microsoft_client_id
MICROSOFT_OAUTH_CLIENT_SECRET=your_microsoft_secret
```

#### Rate Limiting
```bash
RATE_LIMIT_ENABLED=true                               # Enable rate limiting
RATE_LIMIT_PER_MINUTE=100                            # Requests per minute
RATE_LIMIT_PER_HOUR=1000                             # Requests per hour
RATE_LIMIT_PER_DAY=10000                             # Requests per day
```

### Business Logic Configuration

#### Billing Settings
```bash
DEFAULT_CURRENCY=USD                                  # Default currency
SUPPORTED_CURRENCIES=USD,EUR,GBP,CAD,AUD             # Supported currencies
TAX_CALCULATION_ENABLED=true                         # Enable tax calculation
DUNNING_ENABLED=true                                 # Enable dunning management
REVENUE_RECOGNITION_ENABLED=true                     # Enable revenue recognition
```

#### Subscription Defaults
```bash
DEFAULT_TRIAL_DAYS=14                                # Default trial period
DEFAULT_GRACE_PERIOD_DAYS=3                         # Payment grace period
DEFAULT_BILLING_CYCLE=monthly                        # monthly, quarterly, annual
PRORATION_ENABLED=true                               # Enable proration
```

#### Usage Tracking
```bash
USAGE_TRACKING_ENABLED=true                          # Enable usage tracking
USAGE_AGGREGATION_INTERVAL=300                       # Aggregation interval (seconds)
USAGE_RETENTION_DAYS=90                              # Usage data retention
```

### Advanced Configuration

#### Machine Learning
```bash
ML_ENABLED=true                                       # Enable ML features
CHURN_PREDICTION_ENABLED=true                        # Enable churn prediction
FRAUD_DETECTION_ENABLED=true                         # Enable fraud detection
PERSONALIZATION_ENABLED=true                         # Enable personalization
```

#### Performance
```bash
ASYNC_WORKERS=4                                       # Async worker count
BATCH_SIZE=100                                        # Batch processing size
CACHE_PRELOAD=true                                   # Preload cache on startup
DATABASE_POOL_PRE_PING=true                         # Test connections before use
```

#### Backup & Recovery
```bash
BACKUP_ENABLED=true                                   # Enable automated backups
BACKUP_SCHEDULE=0 2 * * *                           # Backup schedule (cron)
BACKUP_RETENTION_DAYS=30                             # Backup retention period
BACKUP_S3_BUCKET=apg-billing-backups                # S3 bucket for backups
```

## Configuration Files

### .env.example
```bash
# Copy this file to .env and configure your values

# Core Application
FLASK_ENV=production
APP_SECRET_KEY=generate-a-secure-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/apg_billing
REDIS_URL=redis://localhost:6379/0

# Payment Processors
STRIPE_SECRET_KEY=sk_live_...
PAYPAL_CLIENT_ID=your_paypal_client_id

# Communication
SENDGRID_API_KEY=SG.your_sendgrid_key
TWILIO_ACCOUNT_SID=your_twilio_sid

# Add other required variables...
```

### Configuration Validation

The system validates all configuration on startup. Use the configuration checker:

```python
from service import validate_configuration

# Check configuration
validation_result = validate_configuration()
if not validation_result['valid']:
    print("Configuration errors:")
    for error in validation_result['errors']:
        print(f"- {error}")
```

### Environment-Specific Configurations

#### Development
```bash
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///apg_billing_dev.db
```

#### Testing
```bash
FLASK_ENV=testing
DATABASE_URL=sqlite:///apg_billing_test.db
TESTING=true
```

#### Production
```bash
FLASK_ENV=production
FLASK_DEBUG=false
LOG_LEVEL=INFO
# Use production database and Redis
```

## Security Best Practices

### Environment Variables
- Never commit `.env` files to version control
- Use strong, unique passwords and API keys
- Rotate secrets regularly
- Use environment-specific configurations

### API Keys
- Store API keys securely (e.g., AWS Secrets Manager, HashiCorp Vault)
- Use least-privilege access for all services
- Monitor API key usage and set up alerts

### Database Security
- Use strong database passwords
- Enable SSL/TLS for database connections
- Configure database firewalls
- Regular security updates

## Configuration Management

### Docker
Use environment variables in `docker-compose.yml`:
```yaml
services:
  apg-billing:
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
```

### Kubernetes
Use ConfigMaps and Secrets:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-billing-config
data:
  FLASK_ENV: "production"
  LOG_LEVEL: "INFO"
---
apiVersion: v1
kind: Secret
metadata:
  name: apg-billing-secrets
data:
  stripe-secret-key: <base64-encoded-key>
```

## Troubleshooting Configuration

### Common Issues
1. **Invalid API Keys**: Check key format and permissions
2. **Database Connection**: Verify connection string and network access
3. **Missing Environment Variables**: Check required variables are set
4. **SSL/TLS Issues**: Verify certificate configuration

### Configuration Debugging
```bash
# Check loaded configuration
python -c "
from service import get_billing_service
service = get_billing_service()
print(service.get_configuration_status())
"
```

---

© 2025 Datacraft. All rights reserved.