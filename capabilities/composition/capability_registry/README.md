# APG Capability Registry

**Foundation Infrastructure for Intelligent Capability Discovery and Orchestration**

The APG Capability Registry is a foundational component of the APG (Application Programming Generation) platform that provides intelligent capability discovery, registration, and orchestration services. It enables APG's unique modular, composable architecture by maintaining a comprehensive registry of all available capabilities, their dependencies, and composition patterns.

## 🚀 Key Features

### Core Capabilities
- **Intelligent Discovery**: Advanced capability search and discovery with AI-enhanced recommendations
- **Composition Engine**: Automated validation and optimization of capability compositions
- **Dependency Management**: Comprehensive dependency resolution and conflict detection
- **Version Control**: Semantic versioning with compatibility analysis and migration support
- **Marketplace Integration**: Seamless integration with APG marketplace for capability distribution

### APG Platform Integration
- **Multi-tenant Architecture**: Complete tenant isolation and security
- **Real-time Collaboration**: Live updates and collaborative composition design
- **Analytics Dashboard**: Advanced usage analytics and performance monitoring
- **Mobile Support**: Offline-first mobile app with PWA capabilities
- **API-First Design**: Comprehensive REST API with WebSocket real-time updates

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🏁 Quick Start

### 1. Register a Capability

```python
from capability_registry import get_registry_service

# Get service instance
service = await get_registry_service("your-tenant-id")

# Register capability
capability_data = {
    "capability_code": "USER_MANAGEMENT",
    "capability_name": "User Management System",
    "description": "Comprehensive user management with RBAC",
    "category": "foundation_infrastructure",
    "version": "1.0.0",
    "multi_tenant": True,
    "audit_enabled": True,
    "api_endpoints": ["/api/users", "/api/roles"],
    "provides_services": ["user_crud", "role_management"]
}

result = await service.register_capability(capability_data)
print(f"Registered capability: {result['capability_id']}")
```

### 2. Create a Composition

```python
# Create capability composition
composition_data = {
    "name": "Enterprise User Portal",
    "description": "Complete user portal with authentication and management",
    "composition_type": "enterprise_portal",
    "capability_ids": ["cap_001", "cap_002", "cap_003"],
    "business_requirements": ["user_authentication", "role_management"],
    "target_users": ["end_users", "administrators"]
}

composition_result = await service.create_composition(composition_data)
print(f"Created composition: {composition_result['composition_id']}")
```

### 3. Search and Discover

```python
# Search capabilities
search_results = await service.search_capabilities({
    "query": "user management",
    "category": "foundation_infrastructure",
    "min_quality_score": 0.8,
    "page": 1,
    "per_page": 10
})

for capability in search_results["capabilities"]:
    print(f"Found: {capability.capability_name} (Score: {capability.quality_score})")
```

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    APG Capability Registry                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Discovery     │  │   Composition   │  │   Version   │  │
│  │    Service      │  │     Engine      │  │  Manager    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Registry      │  │   Analytics     │  │ Marketplace │  │
│  │    Core         │  │    Engine       │  │Integration  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    APG Platform Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Flask-AB UI   │  │   FastAPI       │  │  WebSocket  │  │
│  │    Dashboard    │  │   REST API      │  │   Gateway   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  Mobile PWA     │  │   Webhook       │  │   APG       │  │
│  │   Service       │  │  Integration    │  │Integration  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Capability Registration**: Capabilities are registered through the service layer
2. **Metadata Processing**: Rich metadata is extracted and indexed for discovery
3. **Dependency Analysis**: Dependencies are analyzed and mapped
4. **Composition Validation**: Compositions are validated for compatibility
5. **APG Integration**: Capabilities are registered with APG composition engine
6. **Real-time Updates**: Changes are broadcast via WebSocket to connected clients

## 🔧 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 6+ (for caching)
- Node.js 18+ (for UI development)

### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install UI dependencies
cd ui && npm install
```

### Database Setup

```bash
# Create database
createdb apg_capability_registry

# Run migrations
alembic upgrade head

# Load initial data (optional)
python scripts/load_sample_data.py
```

## ⚙️ Configuration

### Environment Variables

```bash
# Database Configuration
DATABASE_URL="postgresql://user:password@localhost/apg_capability_registry"
REDIS_URL="redis://localhost:6379"

# APG Platform Integration
APG_PLATFORM_URL="https://apg.platform.url"
APG_TENANT_ID="your-tenant-id"
APG_API_KEY="your-api-key"

# Security
JWT_SECRET_KEY="your-jwt-secret"
ENCRYPTION_KEY="your-encryption-key"

# Feature Flags
ENABLE_MARKETPLACE_INTEGRATION=true
ENABLE_REAL_TIME_COLLABORATION=true
ENABLE_ANALYTICS_DASHBOARD=true
```

### Configuration Files

```python
# config.py
class Config:
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # APG Integration
    APG_COMPOSITION_ENGINE_URL = os.environ.get('APG_COMPOSITION_ENGINE_URL')
    APG_DISCOVERY_SERVICE_URL = os.environ.get('APG_DISCOVERY_SERVICE_URL')
    
    # Performance
    REDIS_CACHE_TTL = 3600  # 1 hour
    SEARCH_INDEX_UPDATE_INTERVAL = 300  # 5 minutes
    
    # Security
    ENABLE_AUDIT_LOGGING = True
    ENABLE_ENCRYPTION_AT_REST = True
```

## 📖 Usage Guide

### Core Service Operations

#### Capability Management

```python
from capability_registry import CRService

service = CRService(tenant_id="your-tenant")

# Register capability
capability = await service.register_capability({
    "capability_code": "EMAIL_SERVICE",
    "capability_name": "Email Service",
    "description": "SMTP email service with templates",
    "category": "communication",
    "provides_services": ["email_send", "template_management"],
    "api_endpoints": ["/api/email/send", "/api/email/templates"]
})

# Update capability
await service.update_capability(capability_id, {
    "quality_score": 0.95,
    "performance_optimized": True
})

# Get capability details
capability = await service.get_capability(capability_id)

# Search capabilities
results = await service.search_capabilities({
    "query": "email communication",
    "category": "communication"
})
```

#### Composition Management

```python
from capability_registry import IntelligentCompositionEngine

engine = IntelligentCompositionEngine(tenant_id="your-tenant")

# Validate composition
validation = await engine.analyze_composition([
    "email_service_id",
    "user_management_id",
    "notification_engine_id"
])

# Generate recommendations
recommendations = await engine.generate_recommendations({
    "composition_type": "customer_portal",
    "target_users": ["customers"],
    "business_requirements": ["user_auth", "notifications"]
})

# Create composition
composition = await service.create_composition({
    "name": "Customer Portal",
    "capability_ids": ["cap_1", "cap_2", "cap_3"],
    "composition_type": "customer_portal"
})
```

#### Version Management

```python
from capability_registry import VersionManagerService

version_manager = VersionManagerService(tenant_id="your-tenant")

# Create new version
version = await version_manager.create_version(capability_id, {
    "version_number": "2.0.0",
    "release_notes": "Major update with breaking changes",
    "breaking_changes": ["API endpoint restructure"],
    "new_features": ["GraphQL support", "Enhanced security"],
    "backward_compatible": False
})

# Analyze compatibility
compatibility = await version_manager.analyze_compatibility(
    "email_service", "1.0.0", "2.0.0"
)

# Get migration path
migration = await version_manager.get_migration_path(
    capability_id, "1.5.0", "2.0.0"
)
```

### APG Platform Integration

#### Composition Engine Registration

```python
from capability_registry import APGIntegrationService

apg_service = APGIntegrationService(tenant_id="your-tenant")
await apg_service.set_registry_service(service)

# Register with APG composition engine
apg_metadata = await apg_service.register_with_composition_engine(capability_id)

# Create APG composition
apg_config = await apg_service.create_apg_composition(
    composition_id, capability_ids
)

# Sync with APG ecosystem
sync_results = await apg_service.sync_with_apg_ecosystem()
```

#### Discovery Service Integration

```python
# Register with APG discovery service
discovery_registration = await apg_service.register_with_discovery_service(
    capability_id
)

print(f"Registered with discovery: {discovery_registration.registration_id}")
print(f"Discovery tags: {discovery_registration.discovery_tags}")
```

### Mobile and Offline Support

```python
from capability_registry import MobileOfflineService

mobile_service = MobileOfflineService(
    tenant_id="your-tenant",
    offline_db_path="/path/to/mobile.db"
)

# Get mobile-optimized capabilities
capabilities = await mobile_service.get_mobile_capabilities(
    category="foundation_infrastructure",
    limit=50
)

# Sync offline data
sync_result = await mobile_service.sync_from_online(force_full_sync=True)

# Get mobile dashboard data
dashboard_data = await mobile_service.get_mobile_dashboard_data()
```

## 🔌 API Documentation

### REST API Endpoints

#### Capabilities

```http
GET    /api/capabilities                 # List capabilities
POST   /api/capabilities                 # Create capability
GET    /api/capabilities/{id}            # Get capability
PUT    /api/capabilities/{id}            # Update capability
DELETE /api/capabilities/{id}            # Delete capability
```

#### Compositions

```http
GET    /api/compositions                 # List compositions
POST   /api/compositions                 # Create composition
GET    /api/compositions/{id}            # Get composition
POST   /api/compositions/validate        # Validate composition
```

#### Analytics

```http
GET    /api/analytics/usage              # Usage analytics
GET    /api/analytics/performance        # Performance metrics
```

#### Mobile API

```http
GET    /api/mobile/capabilities          # Mobile capabilities
POST   /api/mobile/sync                  # Mobile sync
```

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/api/ws/your-tenant-id');

// Subscribe to events
ws.send(JSON.stringify({
    type: 'subscribe',
    data: {
        events: ['capability.created', 'composition.updated']
    }
}));

// Handle real-time updates
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'registry_update') {
        console.log('Registry update:', message.data);
    }
};
```

### Example API Usage

```python
import httpx

# Create capability via API
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/capabilities",
        json={
            "capability_code": "PAYMENT_GATEWAY",
            "capability_name": "Payment Gateway",
            "description": "Secure payment processing",
            "category": "financial_services",
            "version": "1.0.0"
        },
        headers={"Authorization": "Bearer your-token"}
    )
    
    capability = response.json()
    print(f"Created: {capability['data']['capability_id']}")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=capability_registry --cov-report=html

# Run specific test categories
pytest -m "not slow"                    # Skip slow tests
pytest -m "integration"                 # Integration tests only
pytest -m "performance"                 # Performance tests only
```

### Test Categories

- **Unit Tests**: Model validation, service logic, utilities
- **Integration Tests**: Service integration, database operations
- **API Tests**: REST API endpoints, WebSocket connections
- **Performance Tests**: Load testing, bulk operations
- **Security Tests**: Authentication, authorization, data protection

### Test Configuration

```python
# conftest.py
import pytest
from capability_registry.service import CRService

@pytest.fixture
async def service():
    service = CRService(tenant_id="test")
    # Setup test database
    yield service
    # Cleanup
```

## 🔄 Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/datacraft/apg-capability-registry.git
cd apg-capability-registry

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Start development services
docker-compose up -d postgres redis

# Run migrations
alembic upgrade head

# Start development server
uvicorn capability_registry.api:api_app --reload
```

### Code Quality

```bash
# Code formatting
black .
isort .

# Type checking
mypy capability_registry/

# Linting
flake8 capability_registry/
pylint capability_registry/

# Security scanning
bandit -r capability_registry/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Add new feature"

# Apply migration
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 📊 Performance and Monitoring

### Key Metrics

- **Capability Registration Time**: Target < 500ms
- **Search Response Time**: Target < 200ms
- **Composition Validation Time**: Target < 1s
- **API Response Time**: Target < 100ms (95th percentile)
- **Database Query Performance**: All queries < 50ms

### Monitoring Setup

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

capability_registrations = Counter('capability_registrations_total')
search_duration = Histogram('search_duration_seconds')
active_compositions = Gauge('active_compositions_count')

# Usage in service
capability_registrations.inc()
with search_duration.time():
    results = await search_capabilities(query)
```

### Health Checks

```bash
# API health check
curl http://localhost:8000/api/health

# Database health
curl http://localhost:8000/api/registry/health

# Service dependencies
curl http://localhost:8000/api/health/dependencies
```

## 🚀 Deployment

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  capability-registry:
    image: datacraft/apg-capability-registry:latest
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres/registry
      - REDIS_URL=redis://redis:6379
      - APG_PLATFORM_URL=https://apg.platform.url
    depends_on:
      - postgres
      - redis
    ports:
      - "8000:8000"

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: registry
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: capability-registry
spec:
  replicas: 3
  selector:
    matchLabels:
      app: capability-registry
  template:
    metadata:
      labels:
        app: capability-registry
    spec:
      containers:
      - name: capability-registry
        image: datacraft/apg-capability-registry:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔒 Security

### Authentication and Authorization

```python
# JWT token validation
from capability_registry.auth import validate_token

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    token = request.headers.get("Authorization")
    if token:
        user = await validate_token(token)
        request.state.user = user
    response = await call_next(request)
    return response
```

### Data Protection

- **Encryption at Rest**: All sensitive data encrypted with AES-256
- **Encryption in Transit**: TLS 1.3 for all communications
- **Access Control**: Role-based access control (RBAC) integration
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Anonymization**: PII data automatically anonymized

### Security Best Practices

- Regular security scans with Bandit and Safety
- Dependency vulnerability scanning
- Input validation and sanitization
- SQL injection prevention with parameterized queries
- XSS protection with content security policies

## 🤝 Contributing

### Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow code style** guidelines (Black, isort, flake8)
4. **Update documentation** for new features
5. **Submit a pull request** with detailed description

### Development Standards

- **Test Coverage**: Minimum 85% code coverage
- **Documentation**: All public APIs must be documented
- **Type Hints**: All functions must have proper type annotations
- **Error Handling**: Comprehensive error handling and logging
- **Performance**: New features must meet performance benchmarks

### Pull Request Process

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and add tests
3. Run quality checks: `make lint test`
4. Update documentation if needed
5. Submit PR with detailed description
6. Address review feedback
7. Merge after approval

## 📄 License

© 2025 Datacraft. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or modification is strictly prohibited.

For licensing inquiries, contact: nyimbi@gmail.com

---

## 📞 Support

- **Email**: nyimbi@gmail.com
- **Website**: https://www.datacraft.co.ke
- **Documentation**: https://docs.datacraft.co.ke/apg/capability-registry
- **Issues**: https://github.com/datacraft/apg-capability-registry/issues

## 📈 Roadmap

### Version 1.1 (Q2 2025)
- Enhanced AI recommendations
- Advanced analytics dashboard
- Multi-region deployment support
- GraphQL API

### Version 1.2 (Q3 2025)
- Blockchain capability verification
- Advanced composition templates
- Real-time collaboration features
- Enhanced mobile capabilities

### Version 2.0 (Q4 2025)
- Distributed registry architecture
- Advanced security features
- Machine learning optimization
- Enterprise marketplace integration

---

**Built with ❤️ by the Datacraft Team**