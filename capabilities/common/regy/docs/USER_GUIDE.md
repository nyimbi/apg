# Registry (regy) - User Guide

**APG Registry Capability - Intelligent Service Discovery & Management**

Version: 1.0.0  
Author: APG Platform Team  
Copyright: © 2025 Datacraft  
Website: www.datacraft.co.ke

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Service Registration](#service-registration)
4. [Service Discovery](#service-discovery)
5. [Health Monitoring](#health-monitoring)
6. [Analytics & Metrics](#analytics--metrics)
7. [Advanced Features](#advanced-features)
8. [API Reference](#api-reference)
9. [Integration Examples](#integration-examples)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The APG Registry capability provides intelligent service discovery and management with ML-powered features that surpass industry leaders like Consul, Eureka, and Zookeeper. It offers:

- **🔍 Intelligent Service Discovery** - AI-powered ranking and filtering
- **❤️ ML Health Monitoring** - Predictive failure detection and anomaly monitoring
- **⚡ Circuit Breaking** - Adaptive thresholds with pattern recognition
- **📊 Real-time Analytics** - Performance metrics and business insights
- **🔒 Enterprise Security** - Multi-tenant isolation with RBAC
- **🚀 High Availability** - Distributed architecture with fault tolerance

### Key Benefits

- **10x Better Performance** - Sub-millisecond discovery with intelligent caching
- **Predictive Scaling** - AI-driven capacity planning and auto-scaling
- **Zero Downtime** - Rolling updates with dependency management
- **Developer Experience** - Intuitive APIs with comprehensive tooling

---

## Quick Start

### Prerequisites

- APG Platform with auth, conf, moni, audl capabilities
- Python 3.9+ with async support
- PostgreSQL database
- Redis for caching (optional)

### Installation

1. **Enable the Registry Capability:**
```bash
apg capability enable regy
```

2. **Verify Installation:**
```bash
curl http://localhost:5000/api/regy/v1/status
```

3. **Access UI:**
Navigate to `http://localhost:5000/serviceregistryview/list/` in your APG dashboard.

### First Service Registration

```python
import requests

# Register your first service
service_data = {
    "name": "my-api-service",
    "display_name": "My API Service",
    "description": "Production API service",
    "service_type": "rest_api",
    "namespace": "production",
    "environment": "prod",
    "base_path": "/api/v1",
    "instances": [{
        "instance_name": "primary",
        "host": "api.mycompany.com",
        "port": 443,
        "base_url": "https://api.mycompany.com",
        "weight": 100
    }],
    "tags": ["production", "api", "critical"]
}

response = requests.post(
    "http://localhost:5000/api/regy/v1/services",
    json=service_data,
    headers={"X-Tenant-ID": "your-tenant-id"}
)

print(f"Service registered: {response.json()['id']}")
```

---

## Service Registration

### Basic Registration

Services are registered with comprehensive metadata for intelligent discovery:

```python
service_config = {
    # Core Identity
    "name": "user-service",           # Unique service name
    "display_name": "User Service",   # Human-readable name
    "description": "User management service",
    
    # Service Classification
    "service_type": "microservice",   # rest_api, microservice, web_service, etc.
    "namespace": "user-mgmt",         # Logical grouping
    "environment": "production",      # deployment environment
    
    # Network Configuration
    "base_path": "/api/v2",
    "instances": [/* instance configs */],
    
    # Discovery & Routing
    "discovery_enabled": True,
    "load_balance_strategy": "weighted_response_time",
    
    # Intelligence Features
    "predictive_scaling": True,       # AI-powered scaling
    "intelligent_routing": True,      # ML-optimized routing
    "anomaly_detection": True,        # Behavior monitoring
    
    # Security
    "authentication_required": True,
    "authorization_policies": ["user-read", "user-write"],
    "rate_limiting_enabled": True,
    
    # Metadata
    "tags": ["users", "authentication", "critical"],
    "labels": {"tier": "critical", "team": "platform"},
    "dependencies": ["auth-service", "database-service"]
}
```

### Service Instances

Each service can have multiple instances for high availability:

```python
instance_config = {
    "instance_name": "user-service-01",
    "host": "user-svc-01.internal.com",
    "port": 8080,
    "base_url": "http://user-svc-01.internal.com:8080",
    
    # Load Balancing
    "weight": 100,                    # Routing weight (0-1000)
    "max_connections": 1000,          # Connection limit
    
    # Health Monitoring
    "health_checks": [{
        "name": "HTTP Health Check",
        "type": "http",
        "url": "http://user-svc-01.internal.com:8080/health",
        "interval_seconds": 30,
        "timeout_seconds": 10,
        "adaptive_intervals": True,   # AI-powered intervals
        "anomaly_detection": True     # ML anomaly detection
    }],
    
    # Circuit Breaking
    "circuit_breakers": [{
        "name": "Main Circuit Breaker",
        "failure_threshold": 5,
        "success_threshold": 3,
        "timeout_seconds": 60,
        "adaptive_thresholds": True,  # ML-optimized thresholds
        "pattern_recognition": True,  # Failure pattern analysis
        "intelligent_recovery": True  # Smart recovery strategies
    }],
    
    # Deployment Info
    "deployment_version": "2.1.0",
    "environment": "production",
    "container_id": "container-abc123",
    "node_id": "k8s-node-01",
    "tags": ["primary", "high-memory"]
}
```

### Using the Web UI

1. Navigate to **Integration → Service Registry**
2. Click **Add Service**
3. Fill in the service details:
   - Basic info (name, type, environment)
   - Instance configuration
   - Health check settings
   - Advanced features (ML, security)
4. Click **Save**

---

## Service Discovery

### Basic Discovery

Discover services using various filters and intelligent ranking:

```python
# Simple name-based discovery
discovery_query = {
    "service_name": "user-service",
    "environment": "production",
    "healthy_only": True,
    "include_instances": True,
    "include_health": True
}

response = requests.post(
    "http://localhost:5000/api/regy/v1/discovery/search",
    json=discovery_query
)

services = response.json()['services']
```

### Intelligent Discovery

Enable AI-powered features for optimal service selection:

```python
intelligent_query = {
    "service_type": "rest_api",
    "namespace": "production",
    
    # AI Features
    "intelligent_ranking": True,      # ML-powered ranking
    "predictive_filtering": True,     # Predict availability
    "similarity_search": True,       # Find similar services
    
    # Performance Criteria
    "min_health_score": 0.8,
    "max_response_time": 100.0,      # milliseconds
    "min_availability": 99.0,        # percentage
    
    # Advanced Filtering
    "tags": ["critical", "production"],
    "labels": {"tier": "critical"},
    "version_constraints": ">=2.0.0",
    "preferred_regions": ["us-east-1", "us-west-2"],
    
    # Load Balancing
    "load_balance_strategy": "least_connections",
    
    # Result Options
    "limit": 50,
    "include_metrics": True,
    "include_predictions": True      # AI predictions
}
```

### Discovery Results

The discovery service returns comprehensive results:

```json
{
    "services": [/* service objects */],
    "total_count": 25,
    "returned_count": 10,
    "query_time_ms": 12.5,
    
    "ranking_algorithm": "ml_health_score_v2",
    "confidence_scores": {
        "health_prediction": 0.92,
        "load_prediction": 0.87
    },
    
    "recommendations": [
        "Consider scaling user-service-cluster-1",
        "Monitor payment-service for anomalies"
    ],
    
    "performance_insights": {
        "average_response_time": 45.2,
        "average_health_score": 0.94,
        "geographic_distribution": {
            "us-east-1": 15,
            "us-west-2": 10
        }
    }
}
```

### Discovery by URL Patterns

```bash
# Discover by service name
GET /api/regy/v1/discovery/by-name/user-service

# Discover by service type
GET /api/regy/v1/discovery/by-type/microservice?environment=prod

# Parameters
?healthy_only=true
&intelligent_ranking=true
&limit=25
&environment=production
&namespace=core
```

---

## Health Monitoring

### Overview Dashboard

Access comprehensive health monitoring at **Monitoring → Service Health**:

- **System Overview**: Total services, health distribution, uptime
- **Real-time Metrics**: CPU, memory, response times, error rates
- **ML Insights**: Anomaly detection, failure predictions, recommendations
- **Geographic View**: Service distribution across regions

### Individual Service Health

Each service provides detailed health information:

```python
# Get service health status
response = requests.get(
    f"http://localhost:5000/api/regy/v1/health/services/{service_id}"
)

health_status = response.json()
```

**Health Status Response:**
```json
{
    "service_id": "user-service-123",
    "instance_id": "instance-456",
    "overall_status": "healthy",
    "health_score": 0.95,
    "status_message": "All systems operational",
    
    "performance_metrics": {
        "response_time_ms": 45.2,
        "cpu_usage_percent": 35.7,
        "memory_usage_percent": 52.3,
        "active_connections": 150,
        "requests_per_second": 500.0
    },
    
    "circuit_breaker": {
        "state": "closed",
        "failure_count": 0,
        "success_rate": 99.8
    },
    
    "ml_insights": {
        "anomaly_detected": false,
        "predicted_failure_probability": 0.05,
        "recommended_actions": [
            "Monitor CPU usage trends",
            "Consider connection pool optimization"
        ]
    },
    
    "last_updated": "2025-01-15T10:30:00Z"
}
```

### Health Check Configuration

Configure comprehensive health monitoring:

```python
health_check_config = {
    "name": "Comprehensive Health Check",
    "type": "http",
    "enabled": True,
    
    # Basic Configuration
    "url": "https://service.com/health",
    "interval_seconds": 30,
    "timeout_seconds": 10,
    "expected_response_codes": [200, 201],
    
    # Thresholds
    "healthy_threshold": 2,           # Consecutive successes
    "unhealthy_threshold": 3,         # Consecutive failures
    
    # AI-Powered Features
    "adaptive_intervals": True,       # Dynamic interval adjustment
    "anomaly_detection": True,        # ML anomaly detection
    "predictive_analysis": True,      # Failure prediction
    
    # Custom Validation
    "response_validation": {
        "content_contains": ["status", "ok"],
        "max_response_time_ms": 1000,
        "min_success_rate": 0.95
    }
}
```

### Manual Health Checks

Trigger manual health checks for immediate status updates:

```python
# Trigger health check
response = requests.post(
    f"http://localhost:5000/api/regy/v1/health/check/{service_id}"
)

result = response.json()
print(f"Health check result: {result['health_status']['health_score']}")
```

---

## Analytics & Metrics

### Performance Dashboard

Access detailed analytics at **Analytics → Service Analytics**:

- **Performance Trends**: Response times, throughput, error rates
- **Business Metrics**: User activity, transaction volumes, revenue impact
- **Capacity Planning**: Resource utilization, scaling recommendations
- **Comparative Analysis**: Service-to-service performance comparison

### Service Metrics

Retrieve detailed metrics for analysis:

```python
# Get service metrics (last 24 hours)
response = requests.get(
    f"http://localhost:5000/api/regy/v1/metrics/services/{service_id}?hours=24"
)

metrics = response.json()['metrics']
```

**Metrics Response:**
```json
{
    "service_id": "user-service-123",
    "time_range_hours": 24,
    "metrics_count": 24,
    "metrics": [{
        "timestamp": "2025-01-15T10:00:00Z",
        "metric_type": "performance",
        "time_window_seconds": 3600,
        
        "request_metrics": {
            "request_count": 50000,
            "error_count": 150,
            "success_rate": 99.7,
            "response_time_p50": 45.2,
            "response_time_p95": 120.5,
            "response_time_p99": 250.8
        },
        
        "resource_metrics": {
            "cpu_usage_avg": 45.8,
            "memory_usage_avg": 62.1,
            "disk_usage_avg": 78.9,
            "network_bytes_in": 1048576,
            "network_bytes_out": 2097152
        },
        
        "business_metrics": {
            "active_users": 1250,
            "successful_transactions": 49850,
            "revenue_impact": 125000.75,
            "conversion_rate": 12.5
        },
        
        "availability_metrics": {
            "uptime_seconds": 3540,
            "downtime_seconds": 60,
            "availability_percentage": 98.33
        },
        
        "custom_metrics": {
            "cache_hit_rate": 0.92,
            "database_query_time": 15.3,
            "external_api_calls": 5000
        }
    }]
}
```

### Registry Statistics

Get comprehensive registry-wide statistics:

```python
response = requests.get(
    "http://localhost:5000/api/regy/v1/metrics/registry/statistics"
)

stats = response.json()
```

---

## Advanced Features

### ML-Powered Intelligence

Enable advanced AI capabilities for optimal service management:

#### Predictive Scaling
```python
service_config = {
    "predictive_scaling": True,
    "scaling_policies": {
        "prediction_window_hours": 24,
        "confidence_threshold": 0.85,
        "max_scale_factor": 3.0,
        "min_instances": 2,
        "max_instances": 20
    }
}
```

#### Intelligent Routing
```python
service_config = {
    "intelligent_routing": True,
    "routing_algorithm": "ml_weighted_round_robin",
    "routing_features": [
        "response_time",
        "cpu_usage",
        "error_rate",
        "geographic_proximity"
    ]
}
```

#### Anomaly Detection
```python
service_config = {
    "anomaly_detection": True,
    "anomaly_config": {
        "sensitivity": "medium",      # low, medium, high
        "detection_window_minutes": 15,
        "alert_threshold": 0.8,
        "auto_remediation": True
    }
}
```

### Circuit Breaking

Advanced circuit breaker with ML optimization:

```python
circuit_breaker_config = {
    "name": "ML Circuit Breaker",
    
    # Traditional Settings
    "failure_threshold": 5,
    "success_threshold": 3,
    "timeout_seconds": 60,
    "failure_rate_threshold": 50.0,
    
    # ML-Enhanced Features
    "adaptive_thresholds": True,      # Dynamic threshold adjustment
    "pattern_recognition": True,      # Failure pattern analysis
    "intelligent_recovery": True,     # Smart recovery strategies
    
    # Advanced Configuration
    "rolling_window_seconds": 300,
    "minimum_request_threshold": 20,
    "half_open_max_calls": 5,
    
    # Monitoring
    "metrics_enabled": True,
    "alert_on_state_change": True
}
```

### API Versioning

Comprehensive API version management:

```python
version_config = {
    "version": "2.1.0",
    "is_current": True,
    "backward_compatible": True,
    
    # Documentation
    "api_schema_url": "https://api.example.com/schema/v2.1.0",
    "documentation_url": "https://docs.example.com/api/v2.1.0",
    
    # Migration Support
    "breaking_changes": [
        "Removed deprecated /v1/legacy endpoint",
        "Changed response format for /users endpoint"
    ],
    "migration_guide_url": "https://docs.example.com/migration/v2.0-to-v2.1",
    
    # Lifecycle Management
    "deprecation_date": null,
    "end_of_life_date": null,
    "sunset_plan": null
}
```

---

## API Reference

### Authentication

All API requests require authentication:

```bash
# Header-based authentication
curl -H "Authorization: Bearer your-token" \
     -H "X-Tenant-ID: your-tenant" \
     -H "X-User-ID: your-user-id" \
     http://localhost:5000/api/regy/v1/services
```

### Service Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regy/v1/services` | List all services |
| `POST` | `/api/regy/v1/services` | Register new service |
| `GET` | `/api/regy/v1/services/{id}` | Get service details |
| `PUT` | `/api/regy/v1/services/{id}` | Update service |
| `DELETE` | `/api/regy/v1/services/{id}` | Deregister service |

### Discovery Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/regy/v1/discovery/search` | Intelligent search |
| `GET` | `/api/regy/v1/discovery/by-name/{name}` | Find by name |
| `GET` | `/api/regy/v1/discovery/by-type/{type}` | Find by type |

### Health Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regy/v1/health` | Registry health overview |
| `GET` | `/api/regy/v1/health/services/{id}` | Service health |
| `PUT` | `/api/regy/v1/health/services/{id}` | Update health |
| `POST` | `/api/regy/v1/health/check/{id}` | Trigger health check |

### Metrics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regy/v1/metrics/services/{id}` | Service metrics |
| `GET` | `/api/regy/v1/metrics/registry/statistics` | Registry stats |

### Events Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/regy/v1/events` | List registry events |
| `GET` | `/api/regy/v1/events/{id}` | Get event details |

---

## Integration Examples

### Kubernetes Integration

Deploy services with automatic registry integration:

```yaml
# k8s-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  annotations:
    apg.registry/enabled: "true"
    apg.registry/service-type: "microservice"
    apg.registry/namespace: "user-management"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
      annotations:
        apg.registry/health-check: "/health"
        apg.registry/metrics-endpoint: "/metrics"
    spec:
      containers:
      - name: user-service
        image: user-service:v2.1.0
        ports:
        - containerPort: 8080
        env:
        - name: APG_REGISTRY_ENDPOINT
          value: "http://apg-registry:5000"
        - name: APG_TENANT_ID
          valueFrom:
            configMapKeyRef:
              name: apg-config
              key: tenant-id
```

### Service Mesh Integration

Integrate with Istio service mesh:

```yaml
# istio-integration.yaml
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: user-service-registry
spec:
  hosts:
  - user-service.internal
  ports:
  - number: 8080
    name: http
    protocol: HTTP
  resolution: DNS
  endpoints:
  - address: user-service.user-mgmt.svc.cluster.local
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: user-service-circuit-breaker
spec:
  host: user-service.internal
  trafficPolicy:
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
```

### Spring Boot Integration

Automatic service registration with Spring Boot:

```java
// Application.java
@SpringBootApplication
@EnableAPGRegistry
public class UserServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }
}

// Configuration
@Component
public class RegistryConfig {
    
    @Bean
    @ConditionalOnProperty(value = "apg.registry.enabled", havingValue = "true")
    public APGRegistryClient registryClient() {
        return APGRegistryClient.builder()
            .endpoint("http://apg-registry:5000")
            .tenantId("${apg.tenant.id}")
            .serviceName("${spring.application.name}")
            .serviceType(ServiceType.MICROSERVICE)
            .namespace("${apg.namespace:default}")
            .environment("${spring.profiles.active}")
            .healthCheckPath("/actuator/health")
            .metricsPath("/actuator/metrics")
            .intelligentRouting(true)
            .predictiveScaling(true)
            .build();
    }
}
```

### Node.js Integration

Express.js middleware for automatic registration:

```javascript
// registry-middleware.js
const APGRegistry = require('@apg/registry-client');

const registryMiddleware = (config) => {
    const registry = new APGRegistry({
        endpoint: process.env.APG_REGISTRY_ENDPOINT,
        tenantId: process.env.APG_TENANT_ID,
        serviceName: config.serviceName,
        serviceType: 'rest_api',
        namespace: config.namespace,
        environment: process.env.NODE_ENV,
        healthCheckPath: '/health',
        predictiveScaling: true,
        intelligentRouting: true
    });

    return async (req, res, next) => {
        if (!registry.isRegistered()) {
            await registry.register();
        }
        next();
    };
};

// app.js
const express = require('express');
const registryMiddleware = require('./registry-middleware');

const app = express();

app.use(registryMiddleware({
    serviceName: 'user-api',
    namespace: 'user-management'
}));

app.listen(3000);
```

---

## Troubleshooting

### Common Issues

#### Service Registration Fails

**Problem**: Service registration returns 400 Bad Request

**Solutions**:
1. Verify required fields are provided
2. Check service name format (alphanumeric with hyphens/underscores)
3. Ensure tenant ID is valid
4. Validate port numbers (1-65535)
5. Check URL format for endpoints

```python
# Debug registration
try:
    response = requests.post(
        "http://localhost:5000/api/regy/v1/services",
        json=service_data,
        headers={"X-Tenant-ID": tenant_id}
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"Registration failed: {e.response.text}")
```

#### Discovery Returns Empty Results

**Problem**: Service discovery finds no services

**Solutions**:
1. Check tenant isolation - ensure correct tenant ID
2. Verify service status (healthy_only=false to include unhealthy)
3. Check namespace and environment filters
4. Ensure services are actually registered
5. Verify network connectivity

```python
# Debug discovery
discovery_query = {
    "tenant_id": tenant_id,
    "healthy_only": False,  # Include all services
    "limit": 1000,          # Increase limit
    "include_instances": True,
    "include_health": True
}
```

#### Health Checks Failing

**Problem**: Services showing as unhealthy

**Solutions**:
1. Verify health check URL accessibility
2. Check expected response codes
3. Ensure timeout values are appropriate
4. Validate network connectivity from registry
5. Review health check logs

```bash
# Test health endpoint manually
curl -v http://service-host:port/health
```

#### Performance Issues

**Problem**: Slow discovery or high latency

**Solutions**:
1. Enable caching with appropriate TTL
2. Optimize discovery query filters
3. Use intelligent ranking sparingly
4. Check database performance
5. Monitor resource usage

```python
# Optimize discovery
optimized_query = {
    "service_type": "microservice",  # Specific type
    "environment": "production",     # Specific environment
    "limit": 25,                    # Reasonable limit
    "intelligent_ranking": False,   # Disable if not needed
    "include_metrics": False        # Reduce payload
}
```

### Debugging Tools

#### Registry Health Check

```bash
# Check registry status
curl http://localhost:5000/api/regy/v1/status

# Check registry readiness
curl http://localhost:5000/api/regy/v1/ready

# Get comprehensive statistics
curl http://localhost:5000/api/regy/v1/metrics/registry/statistics
```

#### Service Health Validation

```bash
# Trigger manual health check
curl -X POST http://localhost:5000/api/regy/v1/health/check/{service_id}

# Get detailed health status
curl http://localhost:5000/api/regy/v1/health/services/{service_id}
```

#### Event Monitoring

```bash
# Monitor registry events
curl "http://localhost:5000/api/regy/v1/events?limit=50&event_type=service_failure"

# Get specific event details
curl http://localhost:5000/api/regy/v1/events/{event_id}
```

### Log Analysis

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.getLogger('apg.registry').setLevel(logging.DEBUG)

# Key log sources to monitor
loggers = [
    'apg.registry.service',      # Service operations
    'apg.registry.discovery',    # Discovery operations
    'apg.registry.health',       # Health monitoring
    'apg.registry.api',          # API requests
    'apg.registry.ml'            # ML operations
]
```

### Support Resources

- **Documentation**: https://docs.datacraft.co.ke/apg/regy
- **API Reference**: http://localhost:5000/api/regy/v1/docs/
- **GitHub Issues**: https://github.com/datacraft/apg/issues
- **Community Support**: https://community.datacraft.co.ke
- **Professional Support**: support@datacraft.co.ke

---

*This user guide covers the comprehensive functionality of the APG Registry capability. For advanced configuration and enterprise features, consult the Administrator Guide and API documentation.*