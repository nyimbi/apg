# 🚀 APG Real-Time Collaboration - Complete Deployment Guide

**Status: ✅ PRODUCTION READY**  
**Version:** 1.0.0  
**Date:** January 30, 2025  
**Author:** Datacraft (nyimbi@gmail.com)

## 🎯 Deployment Overview

This guide provides complete deployment instructions for the APG Real-Time Collaboration capability, including all protocols, Flask-AppBuilder integration, and production configuration.

## 📦 Prerequisites

### System Requirements
- **Python:** 3.11+ with asyncio support
- **PostgreSQL:** 14+ for data persistence
- **Redis:** 7+ for WebSocket scaling and caching
- **Node.js:** 18+ for frontend asset building (optional)
- **Docker:** 20+ for containerized deployment (optional)

### APG Dependencies
- **APG Core Platform:** Latest version
- **APG Auth & RBAC:** For authentication and permissions
- **APG AI Orchestration:** For intelligent features
- **APG Notification Engine:** For smart notifications

## 🛠️ Installation Steps

### 1. Install Dependencies

```bash
# Install Python dependencies
cd capabilities/common/real_time_collaboration
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

### 2. Database Setup

```sql
-- Create collaboration database schema
CREATE DATABASE apg_collaboration;

-- Run migrations
python -m alembic upgrade head
```

### 3. Redis Configuration

```bash
# Install and start Redis
sudo apt install redis-server  # Ubuntu/Debian
brew install redis             # macOS

# Start Redis service
redis-server
```

### 4. Environment Configuration

Create `.env` file:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/apg_collaboration

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0

# APG Integration
APG_AUTH_SERVICE_URL=http://localhost:8001
APG_AI_SERVICE_URL=http://localhost:8002
APG_NOTIFICATION_SERVICE_URL=http://localhost:8003

# WebSocket Configuration
WEBSOCKET_MAX_CONNECTIONS=10000
WEBSOCKET_MESSAGE_RATE_LIMIT=1000

# Protocol Configuration
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
GRPC_SERVER_PORT=50051
SOCKETIO_SERVER_PORT=3000
SIP_SERVER_PORT=5060
RTMP_SERVER_PORT=1935

# Third-Party Integrations
TEAMS_CLIENT_ID=your_teams_client_id
TEAMS_CLIENT_SECRET=your_teams_client_secret
ZOOM_API_KEY=your_zoom_api_key
ZOOM_API_SECRET=your_zoom_api_secret
GOOGLE_MEET_CLIENT_ID=your_google_client_id
GOOGLE_MEET_CLIENT_SECRET=your_google_client_secret

# Security
SECRET_KEY=your_super_secret_key
JWT_SECRET=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key

# Performance
MAX_CONCURRENT_SESSIONS=1000
MAX_PARTICIPANTS_PER_SESSION=100
SESSION_TIMEOUT_MINUTES=480
```

## 🔧 Flask-AppBuilder Integration

### 1. Register the Capability

Add to your main Flask-AppBuilder application:

```python
# app.py
from flask import Flask
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.sqla import SecurityManager

# Import RTC capability
from capabilities.common.real_time_collaboration.blueprint import init_app as init_rtc
from capabilities.common.real_time_collaboration.flask_integration_middleware import init_collaboration_middleware

def create_app():
    app = Flask(__name__)
    app.config.from_object('config')
    
    # Initialize AppBuilder
    appbuilder = AppBuilder(app, db.session, security_manager_class=SecurityManager)
    
    # Initialize Real-Time Collaboration
    init_rtc(app, appbuilder)
    
    # Initialize collaboration middleware for automatic page integration
    init_collaboration_middleware(app, appbuilder)
    
    return app
```

### 2. Add Menu Integration

The capability automatically registers with Flask-AppBuilder menus:

```python
# Menus will be automatically available:
# - Real-Time Collaboration > Collaboration Sessions
# - Real-Time Collaboration > Video Calls  
# - Real-Time Collaboration > Page Collaboration
# - Real-Time Collaboration > Third-Party Integrations
# - Real-Time Collaboration > Collaboration Dashboard
```

### 3. Page-Level Collaboration

Collaboration is now automatically enabled on ALL Flask-AppBuilder pages with:

- **Presence indicators** showing who's viewing the page
- **Contextual chat** overlay for real-time communication  
- **Form delegation** for collaborative form filling
- **Assistance requests** with intelligent routing
- **Mobile-responsive** interface

## 🌐 Protocol Server Deployment

### 1. WebSocket Server

```python
# websocket_server.py
import asyncio
from capabilities.common.real_time_collaboration.websocket_manager import websocket_manager

async def main():
    await websocket_manager.start()
    print("WebSocket server running on ws://localhost:8765")
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Multi-Protocol Server

```python
# protocol_server.py  
import asyncio
from capabilities.common.real_time_collaboration.unified_protocol_manager import initialize_unified_protocols, ProtocolType

async def main():
    # Configure all protocols
    protocol_configs = {
        ProtocolType.MQTT: {"broker_host": "localhost", "broker_port": 1883},
        ProtocolType.GRPC: {"host": "localhost", "port": 50051}, 
        ProtocolType.SOCKETIO: {"host": "localhost", "port": 3000},
        ProtocolType.SIP: {"local_host": "0.0.0.0", "local_port": 5060},
        ProtocolType.RTMP: {"host": "0.0.0.0", "port": 1935}
    }
    
    # Start all protocol servers
    result = await initialize_unified_protocols(protocol_configs)
    print(f"All protocols initialized: {result}")
    
    # Keep running
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🐳 Docker Deployment

### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8765 50051 3000 5060 1935

# Start command
CMD ["python", "run_server.py"]
```

### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Real-Time Collaboration Service
  rtc-app:
    build: .
    ports:
      - "8000:8000"    # HTTP API
      - "8765:8765"    # WebSocket
      - "50051:50051"  # gRPC
      - "3000:3000"    # Socket.IO
      - "5060:5060"    # SIP
      - "1935:1935"    # RTMP
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/apg_collaboration
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  # PostgreSQL Database
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: apg_collaboration
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # MQTT Broker (Eclipse Mosquitto)
  mqtt:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - mosquitto_data:/mosquitto/data
      - mosquitto_logs:/mosquitto/log

volumes:
  postgres_data:
  redis_data:
  mosquitto_data:
  mosquitto_logs:
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rtc-app

# Scale WebSocket connections
docker-compose up --scale rtc-app=3
```

## ⚙️ Production Configuration

### 1. Nginx Load Balancer

```nginx
# /etc/nginx/sites-available/rtc-collaboration
upstream rtc_app {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

upstream rtc_websocket {
    server 127.0.0.1:8765;
    server 127.0.0.1:8766;
    server 127.0.0.1:8767;
}

server {
    listen 80;
    server_name collaboration.example.com;
    
    # HTTP API endpoints
    location /api/ {
        proxy_pass http://rtc_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # WebSocket connections
    location /ws/ {
        proxy_pass http://rtc_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # Static files
    location /static/ {
        alias /var/www/rtc-collaboration/static/;
        expires 30d;
    }
}
```

### 2. Systemd Service

```ini
# /etc/systemd/system/rtc-collaboration.service
[Unit]
Description=APG Real-Time Collaboration Service
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=rtc
Group=rtc
WorkingDirectory=/opt/rtc-collaboration
Environment=PATH=/opt/rtc-collaboration/venv/bin
ExecStart=/opt/rtc-collaboration/venv/bin/python run_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rtc-collaboration
sudo systemctl start rtc-collaboration
sudo systemctl status rtc-collaboration
```

## 📊 Monitoring & Health Checks

### 1. Health Check Endpoints

```bash
# Application health
curl http://localhost:8000/api/v1/rtc/health

# Protocol status
curl http://localhost:8000/api/v1/rtc/status

# WebSocket statistics  
curl http://localhost:8000/api/v1/rtc/analytics/presence
```

### 2. Prometheus Metrics

```python
# metrics.py - Prometheus integration
from prometheus_client import Counter, Histogram, Gauge

# WebSocket metrics
websocket_connections = Gauge('rtc_websocket_connections_total', 'Active WebSocket connections')
websocket_messages = Counter('rtc_websocket_messages_total', 'WebSocket messages sent')

# Protocol metrics
protocol_requests = Counter('rtc_protocol_requests_total', 'Protocol requests', ['protocol'])
protocol_latency = Histogram('rtc_protocol_latency_seconds', 'Protocol latency', ['protocol'])

# Collaboration metrics
active_sessions = Gauge('rtc_active_sessions_total', 'Active collaboration sessions')
page_collaborations = Gauge('rtc_page_collaborations_total', 'Pages with active collaboration')
```

### 3. Logging Configuration

```python
# logging_config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/rtc-collaboration/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        'rtc': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## 🔐 Security Configuration

### 1. Authentication Integration

```python
# security_config.py
from flask_appbuilder.security.sqla import SecurityManager

class RTCSecurityManager(SecurityManager):
    """Custom security manager with RTC permissions"""
    
    def __init__(self, appbuilder):
        super().__init__(appbuilder)
        
        # Add RTC-specific permissions
        self.add_permissions_view(
            ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete'],
            'RTCSession'
        )
        self.add_permissions_view(
            ['can_start', 'can_join', 'can_end'],
            'VideoCall'
        )
```

### 2. API Rate Limiting

```python
# rate_limiting.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Apply to collaboration endpoints
@app.route('/api/v1/rtc/sessions', methods=['POST'])
@limiter.limit("10 per minute")
def create_session():
    pass
```

## 🧪 Testing Deployment

### 1. Integration Tests

```bash
# Run comprehensive tests
uv run pytest -vxs tests/

# Run specific test categories
uv run pytest tests/test_api.py -v
uv run pytest tests/test_websocket.py -v
uv run pytest tests/test_protocols.py -v
```

### 2. Load Testing

```bash
# WebSocket load test
python test_websocket_load.py --connections 1000 --duration 300

# API load test  
ab -n 10000 -c 100 http://localhost:8000/api/v1/rtc/health

# Protocol stress test
python test_all_protocols.py --stress-test
```

### 3. End-to-End Testing

```python
# e2e_test.py
import asyncio
import pytest
from selenium import webdriver

@pytest.mark.asyncio
async def test_flask_appbuilder_integration():
    """Test collaboration on Flask-AppBuilder pages"""
    driver = webdriver.Chrome()
    
    try:
        # Navigate to Flask-AppBuilder page
        driver.get("http://localhost:8000/admin/user/list")
        
        # Verify collaboration widget is injected
        widget = driver.find_element_by_id("rtc-collaboration-widget")
        assert widget.is_displayed()
        
        # Test presence indicator
        presence = driver.find_element_by_class_name("rtc-presence-indicator")
        assert presence.is_displayed()
        
        # Test chat functionality
        chat_toggle = driver.find_element_by_class_name("rtc-chat-toggle")
        chat_toggle.click()
        
        # Verify chat panel opens
        chat_panel = driver.find_element_by_id("rtc-collaboration-panel")
        assert not chat_panel.get_attribute("class").contains("hidden")
        
    finally:
        driver.quit()
```

## 📈 Performance Tuning

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_rtc_session_tenant_active ON rtc_sessions(tenant_id, is_active);
CREATE INDEX idx_rtc_messages_session_timestamp ON rtc_messages(session_id, created_at);
CREATE INDEX idx_rtc_presence_page_user ON rtc_page_presence(page_url, user_id);

-- Optimize queries
ANALYZE rtc_sessions;
ANALYZE rtc_messages;
ANALYZE rtc_page_collaboration;
```

### 2. Redis Optimization

```conf
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 0
save 900 1
save 300 10
save 60 10000
```

### 3. WebSocket Tuning

```python
# websocket_config.py
WEBSOCKET_CONFIG = {
    'max_connections': 10000,
    'max_message_size': 64 * 1024,  # 64KB
    'ping_interval': 20,
    'ping_timeout': 60,
    'compression': 'deflate',
    'max_queue_size': 100
}
```

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] Database migrations applied
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring setup complete
- [ ] Backup procedures tested

### Deployment
- [ ] Code deployed to production servers
- [ ] Services started and verified
- [ ] Load balancer configured
- [ ] Health checks passing
- [ ] Logging functioning correctly

### Post-Deployment
- [ ] End-to-end tests passing
- [ ] Performance metrics within targets
- [ ] Security scans completed
- [ ] Documentation updated
- [ ] Team trained on new features

## 🎉 Success! 

Your APG Real-Time Collaboration capability is now fully deployed and ready for production use! 

**Key Features Now Available:**
- ✅ **Automatic page-level collaboration** on ALL Flask-AppBuilder pages
- ✅ **8 communication protocols** (WebRTC, MQTT, gRPC, Socket.IO, XMPP, SIP, RTMP, WebSocket)
- ✅ **Teams/Zoom/Google Meet feature parity** in video calls  
- ✅ **Real-time presence tracking** and contextual chat
- ✅ **Form delegation and assistance requests** with AI routing
- ✅ **Enterprise-grade security** and scalability
- ✅ **Comprehensive monitoring** and analytics

For support or questions, contact: **nyimbi@gmail.com**