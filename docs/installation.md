# APG Installation & Setup Guide

This guide covers the complete installation and setup process for the APG platform, from basic development setup to production deployment.

## 🚀 Quick Start Installation

### Prerequisites

1. **Python 3.9+** (Recommended: Python 3.11+)
```bash
python --version  # Should be 3.9 or higher
```

2. **PostgreSQL 12+** (Recommended: PostgreSQL 14+)
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Verify installation
psql --version
```

3. **Redis 6+** (Recommended: Redis 7+)
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Verify installation
redis-server --version
```

### Basic Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd apg
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Core Dependencies**
```bash
pip install -r requirements.txt
```

4. **Database Setup**
```bash
# Create PostgreSQL database
createdb apg_development

# Set environment variables
export DATABASE_URL="postgresql://username:password@localhost:5432/apg_development"
export REDIS_URL="redis://localhost:6379/0"

# Initialize database
python -c "from capabilities.composition.database import init_db; init_db()"
```

5. **Start Development Server**
```bash
python cli.py run --debug
```

The APG platform should now be running at `http://localhost:5000`

## 🛠️ Development Setup

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/apg_development
REDIS_URL=redis://localhost:6379/0

# Application Settings
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True

# APG Configuration
APG_DATA_DIR=./data
APG_LOGS_DIR=./logs
APG_TEMP_DIR=./tmp

# Optional: Blockchain Configuration
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
BLOCKCHAIN_NETWORK=ethereum

# Optional: AI/ML Configuration
PYTORCH_DEVICE=cpu  # or 'cuda' for GPU
FEDERATED_LEARNING_ENABLED=true

# Optional: Real-time Collaboration
WEBRTC_STUN_SERVER=stun:stun.l.google.com:19302
WEBSOCKET_REDIS_URL=redis://localhost:6379/1
```

### Advanced Development Dependencies

For full development capabilities, install additional dependencies:

```bash
# Blockchain development
pip install web3 py-solc-x eth-account

# Mobile development
pip install briefcase

# Advanced workflow orchestration
pip install prefect[all] apache-airflow

# AI/ML development
pip install torch torchvision transformers

# Testing and development tools
pip install pytest pytest-asyncio pytest-cov black isort mypy
```

## 📱 Mobile App Development Setup

### BeeWare Mobile Apps

1. **Install BeeWare Dependencies**
```bash
pip install briefcase
cd mobile_apps/beeware
briefcase dev
```

2. **Build Mobile Apps**
```bash
# Android
briefcase build android

# iOS (macOS only)
briefcase build iOS

# Cross-platform
briefcase package
```

### Platform-Specific Requirements

**Android Development:**
- Android SDK
- Java 8+
- Android Studio (recommended)

**iOS Development (macOS only):**
- Xcode 12+
- iOS Simulator
- Apple Developer Account (for device deployment)

## 🔗 Blockchain Integration Setup

### Web3 Dependencies

```bash
# Install Web3 dependencies
pip install web3 py-solc-x eth-account requests

# Install Solidity compiler
python -c "from solcx import install_solc; install_solc('0.8.19')"
```

### Network Configuration

Configure blockchain network endpoints in your environment:

```bash
# Ethereum Mainnet
ETHEREUM_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID

# Polygon Network
POLYGON_RPC_URL=https://polygon-rpc.com

# Local Development
GANACHE_RPC_URL=http://localhost:8545
```

## 🤖 AI/ML Setup

### Federated Learning

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (NVIDIA CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional ML dependencies
pip install scikit-learn numpy pandas transformers
```

### Configuration

```bash
# Environment variables for AI/ML
export PYTORCH_DEVICE=cuda  # or 'cpu'
export FEDERATED_LEARNING_PARTICIPANTS=5
export ML_MODEL_CACHE_DIR=./models
```

## 🐳 Docker Setup

### Development with Docker

```bash
# Build development image
docker build -t apg:dev -f docker/Dockerfile.dev .

# Run with Docker Compose
docker-compose -f docker-compose.dev.yml up
```

### Production Docker Setup

```bash
# Build production image
docker build -t apg:prod -f docker/Dockerfile.prod .

# Deploy with production compose
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Configuration Validation

### System Health Check

Run the built-in system health check:

```bash
python -c "
from capabilities.composition.workflow_orchestration.service import WorkflowOrchestrationService
from capabilities.common.real_time_collaboration.service import CollaborationService
print('✅ Core services available')

# Check database connection
from capabilities.composition.database import get_async_db_session
print('✅ Database connection working')

# Check optional dependencies
try:
    import web3
    print('✅ Blockchain dependencies available')
except ImportError:
    print('⚠️  Blockchain dependencies not installed')

try:
    import torch
    print('✅ AI/ML dependencies available')
except ImportError:
    print('⚠️  AI/ML dependencies not installed')
"
```

### Configuration Verification

Verify your configuration:

```bash
python cli.py config verify
```

## 🚀 Production Installation

### System Requirements

**Minimum Production Requirements:**
- 4 CPU cores
- 16GB RAM
- 100GB SSD storage
- PostgreSQL 14+
- Redis 7+
- Python 3.11+

**Recommended Production Requirements:**
- 8+ CPU cores
- 32GB RAM
- 500GB SSD storage
- Load balancer (nginx/HAProxy)
- CDN for static assets
- Monitoring (Prometheus/Grafana)

### Production Dependencies

```bash
# Production-specific packages
pip install gunicorn uvicorn[standard] nginx-python

# Production database
pip install psycopg2-binary

# Monitoring and logging
pip install prometheus-client structlog
```

### Production Environment Variables

```bash
# Production configuration
FLASK_ENV=production
DEBUG=False
SECRET_KEY=your-very-secure-secret-key

# Database with connection pooling
DATABASE_URL=postgresql://user:pass@db-host:5432/apg_prod?pool_size=20

# Redis cluster
REDIS_URL=redis://redis-cluster:6379/0

# Security
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True

# Performance
WEB_CONCURRENCY=8
MAX_WORKERS=16
```

### SSL/TLS Setup

```bash
# Generate SSL certificates (Let's Encrypt recommended)
certbot --nginx -d yourdomain.com

# Configure nginx for APG
sudo cp deploy/nginx/apg.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/apg.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## 🔍 Troubleshooting Installation

### Common Issues

**1. Database Connection Error**
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Verify connection
psql -h localhost -U username -d apg_development -c "SELECT 1;"
```

**2. Redis Connection Error**
```bash
# Check Redis service
sudo systemctl status redis

# Test connection
redis-cli ping  # Should return PONG
```

**3. Python Dependencies**
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install --no-cache-dir -r requirements.txt
```

**4. Web3 Dependencies**
```bash
# Install system dependencies for Web3
sudo apt-get install build-essential python3-dev

# Reinstall Web3 packages
pip install --no-cache-dir web3 py-solc-x
```

### Performance Optimization

**Database Optimization:**
```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

**Redis Optimization:**
```bash
# Redis configuration
echo "maxmemory 512mb" >> /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" >> /etc/redis/redis.conf
sudo systemctl restart redis
```

## 📋 Installation Checklist

- [ ] Python 3.9+ installed and verified
- [ ] PostgreSQL 12+ installed and running
- [ ] Redis 6+ installed and running
- [ ] Virtual environment created and activated
- [ ] Core dependencies installed
- [ ] Database initialized
- [ ] Environment variables configured
- [ ] Development server running
- [ ] System health check passed
- [ ] Optional dependencies installed (as needed)
- [ ] Configuration validated

## 📞 Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Review the [Configuration Guide](./configuration.md)
3. Contact support at nyimbi@gmail.com

---

*Next Steps: [Quick Start Guide](./quickstart.md) →*