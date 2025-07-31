# APG API Service Mesh

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/datacraft/apg-service-mesh)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

**Intelligent API orchestration and service mesh networking for the APG platform ecosystem.**

The APG API Service Mesh provides comprehensive service discovery, load balancing, traffic management, and observability for distributed microservices architectures. It seamlessly integrates with the APG platform to enable intelligent service composition and automated scaling.

## 🚀 Features

### **Core Service Mesh**
- ✅ **Automatic Service Discovery** - Zero-configuration service registration and discovery
- ✅ **Intelligent Load Balancing** - Multiple algorithms with health-aware routing
- ✅ **Advanced Traffic Management** - Sophisticated routing, splitting, and policies
- ✅ **Circuit Breaker Patterns** - Automatic failure detection and recovery
- ✅ **Health Monitoring** - Continuous health checks with configurable thresholds
- ✅ **Security Policies** - Rate limiting, authentication, and access control

### **Observability & Monitoring**
- 📊 **Real-time Metrics** - Comprehensive performance and business metrics
- 🔍 **Distributed Tracing** - End-to-end request tracing across services
- 📈 **Interactive Dashboards** - Real-time monitoring with live charts
- 🗺️ **Service Topology** - Visual service dependency mapping
- 🚨 **Intelligent Alerting** - Proactive monitoring with smart notifications

### **APG Platform Integration**
- 🔗 **Capability Registry** - Seamless integration with APG capability discovery
- 📡 **Event Streaming** - Real-time event publishing and subscription
- 🎯 **Composition Engine** - Dynamic service composition support
- ⚡ **Auto-scaling** - Intelligent scaling recommendations
- 🌐 **Cross-platform Discovery** - Service discovery across APG ecosystem

### **Developer Experience**
- 📚 **OpenAPI Documentation** - Comprehensive REST API documentation
- 🖥️ **Modern Web Interface** - Intuitive Flask-AppBuilder management UI
- 📱 **Mobile-responsive** - Full functionality on all devices
- 🔄 **WebSocket Support** - Real-time updates and live monitoring
- 🛠️ **CLI Tools** - Command-line interface for automation

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Web Interface](#-web-interface)
- [Monitoring](#-monitoring)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 6+
- Docker (optional)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/datacraft/apg-service-mesh.git
cd apg-service-mesh

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### 2. Configuration

```bash
# Copy configuration template
cp config/config.example.yaml config/config.yaml

# Edit configuration
vim config/config.yaml
```

### 3. Database Setup

```bash
# Run database migrations
alembic upgrade head

# Initialize with sample data (optional)
python scripts/init_sample_data.py
```

### 4. Start the Service

```bash
# Start the service mesh
python -m api_service_mesh

# Or with Docker
docker-compose up -d
```

### 5. Verify Installation

```bash
# Check health
curl http://localhost:8000/api/health

# Access web interface
open http://localhost:8000/service-mesh/dashboard

# View API documentation
open http://localhost:8000/api/docs
```

## 🔧 Installation

### Production Installation

```bash
# Install from PyPI (when available)
pip install apg-service-mesh

# Or install from source
git clone https://github.com/datacraft/apg-service-mesh.git
cd apg-service-mesh
pip install -e .
```

### Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/datacraft/apg-service-mesh.git
cd apg-service-mesh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Pull the official image
docker pull datacraft/apg-service-mesh:latest

# Or build from source
docker build -t apg-service-mesh .

# Run with Docker Compose
docker-compose up -d
```

## ⚙️ Configuration

### Environment Variables

```bash
# Database configuration
DATABASE_URL=postgresql://user:password@localhost:5432/apg_service_mesh
REDIS_URL=redis://localhost:6379/0

# Service mesh settings
MESH_NAME=apg-production
MESH_NAMESPACE=default
MESH_ENVIRONMENT=production

# Security settings
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
CORS_ORIGINS=["http://localhost:3000", "https://your-domain.com"]

# Monitoring settings
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=30

# APG platform integration
APG_REGISTRY_URL=http://localhost:8001
APG_EVENT_BUS_URL=redis://localhost:6379/1
```

### Configuration File

```yaml
# config/config.yaml
service_mesh:
  name: "apg-production"
  namespace: "default"
  environment: "production"
  
database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  echo: false
  
redis:
  url: "${REDIS_URL}"
  connection_pool_size: 20
  
load_balancing:
  default_algorithm: "round_robin"
  health_check_interval: 30
  circuit_breaker_enabled: true
  
traffic_management:
  default_timeout: 30000
  max_retries: 3
  rate_limiting_enabled: true
  
monitoring:
  metrics_enabled: true
  tracing_enabled: true
  retention_days: 30
  
security:
  authentication_required: true
  rate_limiting:
    requests_per_minute: 1000
    burst_size: 100
```

## 💡 Usage Examples

### Register a Service

```python
import httpx
import asyncio

async def register_service():
    async with httpx.AsyncClient() as client:
        service_data = {
            "service_config": {
                "service_name": "user-service",
                "service_version": "v1.2.0",
                "namespace": "auth",
                "description": "User authentication and management service"
            },
            "endpoints": [
                {
                    "host": "user-service-1.auth.svc.cluster.local",
                    "port": 8080,
                    "protocol": "http",
                    "path": "/api/v1",
                    "weight": 100,
                    "health_check_path": "/health"
                },
                {
                    "host": "user-service-2.auth.svc.cluster.local", 
                    "port": 8080,
                    "protocol": "http",
                    "path": "/api/v1",
                    "weight": 100,
                    "health_check_path": "/health"
                }
            ]
        }
        
        response = await client.post(
            "http://localhost:8000/api/services",
            json=service_data,
            headers={"Authorization": "Bearer your-token"}
        )
        
        result = response.json()
        print(f"Service registered: {result['data']['service_id']}")

# Run the example
asyncio.run(register_service())
```

### Create Traffic Routes

```python
async def create_route():
    async with httpx.AsyncClient() as client:
        route_data = {
            "route_config": {
                "route_name": "user-api-route",
                "match_type": "prefix",
                "match_value": "/api/users",
                "destination_services": [
                    {"service_id": "svc_user_service", "weight": 90},
                    {"service_id": "svc_user_service_v2", "weight": 10}
                ],
                "timeout_ms": 30000,
                "retry_attempts": 3,
                "priority": 1000
            }
        }
        
        response = await client.post(
            "http://localhost:8000/api/routes",
            json=route_data,
            headers={"Authorization": "Bearer your-token"}
        )
        
        result = response.json()
        print(f"Route created: {result['data']['route_id']}")

asyncio.run(create_route())
```

### Configure Load Balancing

```python
async def setup_load_balancer():
    async with httpx.AsyncClient() as client:
        lb_data = {
            "load_balancer_name": "user-service-lb",
            "algorithm": "weighted_round_robin",
            "session_affinity": True,
            "session_affinity_cookie": "SESSIONID",
            "health_check_enabled": True,
            "health_check_interval": 30,
            "circuit_breaker_enabled": True,
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "max_connections": 200
        }
        
        response = await client.post(
            "http://localhost:8000/api/load-balancers",
            json=lb_data,
            headers={"Authorization": "Bearer your-token"}
        )
        
        result = response.json()
        print(f"Load balancer created: {result['data']['load_balancer_id']}")

asyncio.run(setup_load_balancer())
```

### Real-time Monitoring

```python
import websockets
import json

async def monitor_services():
    uri = "ws://localhost:8000/ws/monitoring?tenant_id=your-tenant"
    
    async with websockets.connect(uri) as websocket:
        # Start monitoring
        await websocket.send(json.dumps({
            "action": "start_monitoring",
            "interval": 5000
        }))
        
        # Listen for real-time updates
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "monitoring_data":
                metrics = data["data"]
                print(f"RPS: {metrics['traffic']['requests_per_second']}")
                print(f"Response Time: {metrics['traffic']['avg_response_time']}ms")
                print(f"Error Rate: {metrics['traffic']['error_rate']}%")

asyncio.run(monitor_services())
```

### Query Metrics

```python
async def get_metrics():
    async with httpx.AsyncClient() as client:
        query_data = {
            "service_ids": ["svc_user_service", "svc_payment_service"],
            "metric_names": ["request_total", "response_time"],
            "start_time": "2025-01-26T00:00:00Z",
            "end_time": "2025-01-26T23:59:59Z",
            "aggregation": "avg"
        }
        
        response = await client.post(
            "http://localhost:8000/api/metrics/query",
            json=query_data,
            headers={"Authorization": "Bearer your-token"}
        )
        
        metrics = response.json()["data"]["metrics"]
        for service_id, data in metrics.items():
            print(f"Service {service_id}:")
            print(f"  Requests: {data.request_count}")
            print(f"  Avg Response Time: {data.avg_response_time}ms")
            print(f"  Error Rate: {data.error_count / data.request_count * 100:.2f}%")

asyncio.run(get_metrics())
```

## 📚 API Reference

### Service Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/services` | POST | Register a new service |
| `/api/services` | GET | List all services |
| `/api/services/{id}` | GET | Get service details |
| `/api/services/{id}` | PUT | Update service configuration |
| `/api/services/{id}` | DELETE | Deregister service |
| `/api/services/{id}/status` | PUT | Update service status |

### Traffic Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/routes` | POST | Create traffic route |
| `/api/routes` | GET | List all routes |
| `/api/routes/{id}` | GET | Get route details |
| `/api/routes/{id}` | PUT | Update route configuration |
| `/api/routes/{id}` | DELETE | Delete route |
| `/api/routes/{id}/traffic-split` | POST | Update traffic splitting |

### Load Balancing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/load-balancers` | POST | Create load balancer |
| `/api/load-balancers` | GET | List load balancers |
| `/api/load-balancers/{id}` | GET | Get load balancer details |
| `/api/load-balancers/{id}` | PUT | Update load balancer |
| `/api/load-balancers/{id}` | DELETE | Delete load balancer |

### Monitoring & Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Get mesh health status |
| `/api/health-check` | POST | Trigger health checks |
| `/api/metrics/query` | POST | Query service metrics |
| `/api/topology` | GET | Get service topology |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/monitoring` | Real-time monitoring updates |
| `/ws/alerts` | Real-time alert notifications |

For complete API documentation, visit `/api/docs` when the service is running.

## 🖥️ Web Interface

The service mesh includes a comprehensive web interface built with Flask-AppBuilder.

### Dashboard Features

- **📊 Overview Dashboard** - Key metrics and system health
- **🗺️ Service Topology** - Interactive service dependency graph
- **📈 Real-time Monitoring** - Live charts and metrics
- **⚙️ Service Management** - CRUD operations for services
- **🛣️ Route Configuration** - Traffic routing management
- **⚖️ Load Balancer Setup** - Load balancing configuration
- **🛡️ Policy Management** - Security and traffic policies

### Accessing the Interface

```bash
# Start the service
python -m api_service_mesh

# Open web interface
open http://localhost:8000/service-mesh/dashboard
```

### Interface Screenshots

```
┌─────────────────────────────────────────────────────────┐
│                  Service Mesh Dashboard                 │
├─────────────────────────────────────────────────────────┤
│  Total Services: 12    Healthy: 11    RPS: 1,247      │
│  Response Time: 124ms  Error Rate: 0.3%  Alerts: 2    │
├─────────────────────────────────────────────────────────┤
│  [Traffic Chart]  [Service Health]  [Quick Actions]    │
│  [Service List]   [Recent Alerts]   [Topology View]    │
└─────────────────────────────────────────────────────────┘
```

## 📊 Monitoring

### Metrics Collection

The service mesh automatically collects comprehensive metrics:

- **Request Metrics**: Rate, latency, error rate (RED metrics)
- **Resource Metrics**: CPU, memory, network utilization  
- **Business Metrics**: User sessions, transaction volume
- **Custom Metrics**: Application-specific measurements

### Health Monitoring

- **Endpoint Health Checks**: HTTP/TCP/gRPC health verification
- **Service Health**: Aggregate health based on endpoint status
- **Circuit Breaker**: Automatic failure detection and recovery
- **Alert Rules**: Configurable alerting with smart notifications

### Observability Stack

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    scrape_interval: 15s
    
  grafana:
    enabled: true
    port: 3000
    dashboards_enabled: true
    
  jaeger:
    enabled: true
    port: 16686
    sampling_rate: 0.1
    
  alertmanager:
    enabled: true
    port: 9093
    webhook_url: "https://your-webhook.com"
```

### Custom Dashboards

Import pre-built Grafana dashboards:

```bash
# Import service mesh dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/service-mesh-overview.json

# Import performance dashboard  
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/performance-analysis.json
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m performance            # Performance tests only

# Run with coverage
pytest --cov=api_service_mesh --cov-report=html

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000
```

### Test Configuration

```bash
# Set test environment variables
export TESTING=true
export TEST_DATABASE_URL=sqlite:///test.db
export TEST_REDIS_URL=redis://localhost:6379/15

# Run tests with custom configuration
pytest --env test --config config/test.yaml
```

### Performance Testing

```bash
# Run performance test suite
python tests/performance_tests.py

# Run load tests with specific parameters
locust -f tests/load_tests.py \
  --host http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 300s
```

### Test Coverage

Current test coverage:

- **Unit Tests**: 95%+ coverage
- **Integration Tests**: End-to-end scenarios
- **Performance Tests**: Load and stress testing
- **API Tests**: Complete REST API validation
- **WebSocket Tests**: Real-time functionality

## 🚀 Deployment

### Production Deployment

#### Docker Deployment

```bash
# Build production image
docker build -f Dockerfile.prod -t apg-service-mesh:latest .

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale app=3
```

#### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n apg-service-mesh
kubectl get services -n apg-service-mesh
```

#### Helm Deployment

```bash
# Install with Helm
helm repo add datacraft https://charts.datacraft.co.ke
helm repo update

# Install chart
helm install apg-service-mesh datacraft/api-service-mesh \
  --namespace apg-service-mesh \
  --create-namespace \
  --values values.prod.yaml

# Upgrade deployment
helm upgrade apg-service-mesh datacraft/api-service-mesh \
  --values values.prod.yaml
```

### Configuration Management

#### Environment-specific Configurations

```bash
# Development
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export METRICS_ENABLED=true

# Staging
export ENVIRONMENT=staging
export LOG_LEVEL=INFO
export RATE_LIMITING_ENABLED=true

# Production
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export HIGH_AVAILABILITY=true
export BACKUP_ENABLED=true
```

#### Secrets Management

```bash
# Using Kubernetes secrets
kubectl create secret generic apg-service-mesh-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=jwt-secret="..." \
  --namespace apg-service-mesh

# Using HashiCorp Vault
vault kv put secret/apg-service-mesh \
  database_url="postgresql://..." \
  redis_url="redis://..." \
  jwt_secret="..."
```

### High Availability Setup

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-service-mesh
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: service-mesh
        image: datacraft/apg-service-mesh:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Monitoring in Production

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml
kubectl apply -f monitoring/alertmanager.yaml

# Configure service monitors
kubectl apply -f monitoring/service-monitor.yaml

# Setup alerts
kubectl apply -f monitoring/alert-rules.yaml
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/apg-service-mesh.git
cd apg-service-mesh

# Create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Make your changes and run tests
pytest

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### Code Standards

- **Python**: Use async throughout, modern typing, follow PEP 8
- **Testing**: Maintain >90% test coverage
- **Documentation**: Update docs for all public APIs
- **Security**: Follow secure coding practices

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Documentation**: [https://docs.datacraft.co.ke/apg-service-mesh](https://docs.datacraft.co.ke/apg-service-mesh)
- **Issues**: [GitHub Issues](https://github.com/datacraft/apg-service-mesh/issues)
- **Discussions**: [GitHub Discussions](https://github.com/datacraft/apg-service-mesh/discussions)
- **Email**: support@datacraft.co.ke

## 🗺️ Roadmap

### v1.1.0 (Q2 2025)
- [ ] GraphQL federation support
- [ ] Advanced ML-based routing
- [ ] Multi-cloud deployment
- [ ] Enhanced security policies

### v1.2.0 (Q3 2025)
- [ ] Service mesh federation
- [ ] Edge computing integration
- [ ] Serverless function routing
- [ ] AI-powered anomaly detection

### v2.0.0 (Q4 2025)
- [ ] Complete rewrite in Rust for performance
- [ ] Native Kubernetes CRD support
- [ ] Advanced cost optimization
- [ ] Enterprise federation

---

© 2025 Datacraft. All rights reserved.