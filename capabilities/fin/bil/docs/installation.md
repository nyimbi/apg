# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended (2 cores minimum)
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 50GB+ available space
- **Network**: Reliable internet connection for external API integrations

### Software Requirements
- **Python**: 3.11 or higher
- **Database**: PostgreSQL 13+ (recommended) or SQLite for development
- **Cache**: Redis 6+ (recommended) or in-memory for development
- **Operating System**: Linux (recommended), macOS, or Windows

## Installation Methods

### Method 1: Standard Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd apg/capabilities/common/billing
```

#### 2. Create Virtual Environment
```bash
# Using Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n apg-billing python=3.11
conda activate apg-billing
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section)
nano .env
```

#### 5. Database Setup
```bash
# Initialize database
python -c "from service import get_billing_service; get_billing_service()"

# Run migrations (if using Flask-Migrate)
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

#### 6. Start the Service
```bash
python service.py
```

### Method 2: Docker Installation

#### 1. Using Docker Compose
```bash
# Clone repository
git clone <repository-url>
cd apg/capabilities/common/billing

# Start all services
docker-compose up -d
```

#### 2. Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  apg-billing:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/apg_billing
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=apg_billing
      - POSTGRES_USER=apg_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Method 3: Kubernetes Deployment

#### 1. Helm Chart Installation
```bash
# Add Datacraft Helm repository
helm repo add datacraft https://charts.datacraft.co.ke
helm repo update

# Install APG Billing
helm install apg-billing datacraft/apg-billing \
  --set database.host=your-postgres-host \
  --set redis.host=your-redis-host
```

#### 2. Manual Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-billing
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-billing
  template:
    metadata:
      labels:
        app: apg-billing
    spec:
      containers:
      - name: apg-billing
        image: datacraft/apg-billing:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-billing-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: apg-billing-secrets
              key: redis-url
```

## Database Setup

### PostgreSQL Setup (Recommended)

#### 1. Install PostgreSQL
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib

# macOS
brew install postgresql
```

#### 2. Create Database and User
```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE apg_billing;
CREATE USER apg_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE apg_billing TO apg_user;

-- Create extensions
\c apg_billing
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

#### 3. Configure Connection
```bash
# In .env file
DATABASE_URL=postgresql://apg_user:your_secure_password@localhost:5432/apg_billing
```

### SQLite Setup (Development Only)

```bash
# In .env file
DATABASE_URL=sqlite:///apg_billing.db
```

## Redis Setup

### Install Redis
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# CentOS/RHEL
sudo yum install redis

# macOS
brew install redis
```

### Configure Redis
```bash
# Start Redis
sudo systemctl start redis-server

# Configure in .env
REDIS_URL=redis://localhost:6379/0
```

## External Service Setup

### Required API Keys

#### Payment Processors
```bash
# Stripe
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# PayPal
PAYPAL_CLIENT_ID=your_paypal_client_id
PAYPAL_CLIENT_SECRET=your_paypal_client_secret
PAYPAL_ENVIRONMENT=sandbox  # or live
```

#### Tax Services
```bash
# Avalara
AVALARA_USERNAME=your_avalara_username
AVALARA_PASSWORD=your_avalara_password
AVALARA_ENVIRONMENT=sandbox  # or production

# TaxJar
TAXJAR_API_TOKEN=your_taxjar_token
TAXJAR_ENVIRONMENT=sandbox  # or production
```

#### Communication Services
```bash
# SendGrid
SENDGRID_API_KEY=SG.your_sendgrid_key

# Twilio
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_number

# AWS SNS
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-1
```

#### Analytics & Marketing
```bash
# Google APIs
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token

# Facebook Marketing
FACEBOOK_ACCESS_TOKEN=your_facebook_token
FACEBOOK_APP_ID=your_app_id
FACEBOOK_APP_SECRET=your_app_secret

# Google Analytics
GOOGLE_ANALYTICS_MEASUREMENT_ID=G-XXXXXXXXXX
```

### Optional Services

#### Monitoring & Alerting
```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# PagerDuty
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key

# Health Alert Recipients
HEALTH_ALERT_EMAILS=ops@yourcompany.com,admin@yourcompany.com
```

## Verification

### 1. Service Health Check
```bash
curl http://localhost:5000/billing/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00Z",
  "services": {
    "database": "connected",
    "cache": "connected",
    "payment_processors": "configured"
  }
}
```

### 2. Database Connection Test
```bash
python -c "
from service import get_billing_service
service = get_billing_service()
print('Database connection: OK')
print(f'Tables created: {len(service.get_service_status())}')
"
```

### 3. API Endpoints Test
```bash
# Test main dashboard
curl http://localhost:5000/billing/dashboard

# Test API endpoint
curl http://localhost:5000/api/v1/billing/plans
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection string
echo $DATABASE_URL

# Test manual connection
psql $DATABASE_URL
```

#### 2. Redis Connection Failed
```bash
# Check Redis status
sudo systemctl status redis-server

# Test Redis connection
redis-cli ping
```

#### 3. Import Errors
```bash
# Verify Python environment
which python
python --version

# Check installed packages
pip list | grep -E "(flask|sqlalchemy|redis)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 4. Permission Errors
```bash
# Check file permissions
ls -la

# Fix ownership
sudo chown -R $USER:$USER .

# Fix permissions
chmod +x scripts/*.sh
```

### Logs and Debugging

#### Enable Debug Mode
```bash
# In .env file
FLASK_ENV=development
FLASK_DEBUG=1
LOG_LEVEL=DEBUG
```

#### Check Logs
```bash
# Application logs
tail -f logs/apg_billing.log

# Error logs
tail -f logs/error.log

# Service logs
journalctl -u apg-billing -f
```

## Next Steps

1. **Configuration**: Review [Configuration Guide](configuration.md)
2. **Quick Start**: Follow [Quick Start Guide](quickstart.md)
3. **API Setup**: Configure [API Access](api/README.md)
4. **Monitoring**: Set up [Monitoring & Alerts](operations/monitoring.md)
5. **Production**: Review [Production Deployment](operations/deployment.md)

## Support

- **Documentation**: [docs/](README.md)
- **Issues**: Report installation issues via GitHub Issues
- **Email**: nyimbi@gmail.com
- **Website**: www.datacraft.co.ke

---

© 2025 Datacraft. All rights reserved.