# Getting Started with AICR

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Steps](#first-steps)
5. [Basic Operations](#basic-operations)
6. [Advanced Features](#advanced-features)
7. [Testing and Validation](#testing-and-validation)
8. [Next Steps](#next-steps)

## Prerequisites

### System Requirements

- **Python**: 3.12 or higher
- **Database**: PostgreSQL 13+ (for metadata storage)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: 100GB+ available space for model artifacts
- **Network**: Reliable internet connection for model downloads

### Required Software

```bash
# Python 3.12+
python --version  # Should be 3.12 or higher

# PostgreSQL
psql --version    # Should be 13 or higher

# Docker (optional, for containerized deployment)
docker --version

# Kubernetes (optional, for production deployment)
kubectl version
```

### APG Platform

The AICR capability requires the APG (Advanced Processing Gateway) platform core components:

- APG Composition Engine
- APG Security Framework
- APG Monitoring Infrastructure
- APG Configuration Management

## Installation

### 1. Clone the Repository

```bash
git clone <apg-repository-url>
cd apg/capabilities/common/aicr
```

### 2. Install Dependencies

#### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Using uv (recommended)

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run basic import test
python -c "from aicr import AICoreService; print('AICR imported successfully')"

# Run comprehensive tests
uv run pytest tests/ci/ -v
```

## Configuration

### 1. Database Setup

Create a PostgreSQL database for AICR:

```sql
-- Connect to PostgreSQL as admin user
CREATE DATABASE aicr_db;
CREATE USER aicr_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE aicr_db TO aicr_user;
```

### 2. Environment Configuration

Create a `.env` file in the AICR directory:

```bash
# Database Configuration
DATABASE_URL=postgresql://aicr_user:secure_password@localhost:5432/aicr_db

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
ENCRYPTION_KEY=your_32_byte_encryption_key_here

# Service Configuration
AICR_HOST=0.0.0.0
AICR_PORT=8080
AICR_WORKERS=4

# Storage Configuration
MODEL_STORAGE_PATH=/opt/aicr/models
TEMP_STORAGE_PATH=/tmp/aicr

# Monitoring Configuration
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Security Settings
ENABLE_AUTHENTICATION=true
ENABLE_AUTHORIZATION=true
ENABLE_AUDIT_LOGGING=true

# Performance Settings
MAX_CONCURRENT_REQUESTS=100
INFERENCE_TIMEOUT_SECONDS=300
BATCH_SIZE_LIMIT=1000
```

### 3. APG Integration Configuration

Update your APG configuration to include the AICR capability:

```yaml
# apg_config.yaml
capabilities:
  aicr:
    enabled: true
    priority: high
    interfaces:
      - ai.inference
      - ai.training
      - ai.management
    dependencies:
      - security
      - monitoring
      - storage
    configuration:
      auto_start: true
      health_check_interval: 30
```

## First Steps

### 1. Initialize the Service

```python
import asyncio
from aicr.service import AICoreService

async def initialize_aicr():
    # Create service instance
    service = AICoreService()

    # Initialize with configuration
    config = {
        "database_url": "postgresql://aicr_user:password@localhost:5432/aicr_db",
        "model_storage_path": "/opt/aicr/models",
        "enable_monitoring": True,
        "security_enabled": True
    }

    # Initialize the service
    await service.initialize(config)

    print(f"AICR Service initialized: {service.service_id}")
    return service

# Run initialization
service = asyncio.run(initialize_aicr())
```

### 2. Health Check

Verify that the service is running properly:

```python
async def health_check():
    # Get service health
    health = await service.get_health_status()

    print(f"Service Status: {health['status']}")
    print(f"Database: {health['database']['status']}")
    print(f"Storage: {health['storage']['status']}")
    print(f"Security: {health['security']['status']}")

    return health['status'] == 'healthy'

# Check health
healthy = asyncio.run(health_check())
print(f"Service is healthy: {healthy}")
```

### 3. Register Your First Model

```python
from aicr.models import AICRModel

async def register_first_model():
    # Define model metadata
    model_data = {
        "name": "image_classifier_v1",
        "description": "A convolutional neural network for image classification",
        "model_type": "classification",
        "framework": "pytorch",
        "version": "1.0.0",
        "input_schema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Base64 encoded image data"
                }
            },
            "required": ["image"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence_scores": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            }
        },
        "configuration": {
            "batch_size": 32,
            "device": "cpu",
            "num_classes": 1000
        },
        "tags": ["computer_vision", "imagenet", "resnet"]
    }

    # Register the model
    model = await service.register_model(model_data)

    print(f"Model registered: {model.model_id}")
    print(f"Model name: {model.name}")
    print(f"Model status: {model.status}")

    return model

# Register model
model = asyncio.run(register_first_model())
```

## Basic Operations

### 1. Model Management

#### List Models

```python
async def list_models():
    models = await service.list_models(
        limit=10,
        filters={"model_type": "classification"}
    )

    for model in models:
        print(f"ID: {model.model_id}")
        print(f"Name: {model.name}")
        print(f"Type: {model.model_type}")
        print(f"Status: {model.status}")
        print("---")

asyncio.run(list_models())
```

#### Get Model Details

```python
async def get_model_details(model_id: str):
    model = await service.get_model(model_id)

    if model:
        print(f"Model: {model.name}")
        print(f"Description: {model.description}")
        print(f"Framework: {model.framework}")
        print(f"Version: {model.version}")
        print(f"Performance: {model.performance_metrics}")
    else:
        print("Model not found")

# Get details for your model
asyncio.run(get_model_details(model.model_id))
```

#### Update Model

```python
async def update_model(model_id: str):
    updates = {
        "description": "Updated image classification model with improved accuracy",
        "version": "1.1.0",
        "performance_metrics": {
            "accuracy": 0.92,
            "precision": 0.90,
            "recall": 0.89
        }
    }

    updated_model = await service.update_model(model_id, updates)
    print(f"Model updated: {updated_model.version}")

asyncio.run(update_model(model.model_id))
```

### 2. Model Deployment

#### Deploy Model

```python
async def deploy_model(model_id: str):
    deployment_config = {
        "instance_type": "cpu",
        "min_instances": 1,
        "max_instances": 5,
        "auto_scaling": True,
        "health_check_path": "/health"
    }

    deployment = await service.deploy_model(
        model_id=model_id,
        deployment_config=deployment_config
    )

    print(f"Model deployed: {deployment['deployment_id']}")
    print(f"Endpoint: {deployment['endpoint']}")
    print(f"Status: {deployment['status']}")

    return deployment

# Deploy your model
deployment = asyncio.run(deploy_model(model.model_id))
```

#### Check Deployment Status

```python
async def check_deployment_status(deployment_id: str):
    status = await service.get_deployment_status(deployment_id)

    print(f"Deployment ID: {deployment_id}")
    print(f"Status: {status['status']}")
    print(f"Health: {status['health']}")
    print(f"Instances: {status['instances']}")

asyncio.run(check_deployment_status(deployment['deployment_id']))
```

### 3. Running Inference

#### Single Inference

```python
from aicr.models import AICRInferenceRequest
import base64

async def run_single_inference():
    # Prepare input data (example with base64 encoded image)
    with open("test_image.jpg", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode()

    # Create inference request
    request = AICRInferenceRequest(
        model_id=model.model_id,
        input_data={
            "image": image_data
        },
        parameters={
            "confidence_threshold": 0.8,
            "top_k": 5
        },
        output_format="json",
        priority="normal"
    )

    # Run inference
    response = await service.run_inference(request)

    print(f"Request ID: {response.request_id}")
    print(f"Status: {response.status}")
    print(f"Predictions: {response.predictions}")
    print(f"Confidence: {response.confidence_scores}")
    print(f"Processing time: {response.processing_time_ms}ms")

    return response

# Run inference
response = asyncio.run(run_single_inference())
```

#### Batch Inference

```python
async def run_batch_inference():
    # Prepare multiple requests
    requests = []
    for i in range(5):
        with open(f"test_image_{i}.jpg", "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()

        request = AICRInferenceRequest(
            model_id=model.model_id,
            input_data={"image": image_data},
            parameters={"confidence_threshold": 0.8}
        )
        requests.append(request)

    # Run batch inference
    responses = await service.run_batch_inference(requests)

    for i, response in enumerate(responses):
        print(f"Image {i}: {response.predictions}")

asyncio.run(run_batch_inference())
```

## Advanced Features

### 1. Model Marketplace

```python
from aicr.model_marketplace import ModelMarketplace

async def explore_marketplace():
    marketplace = ModelMarketplace()
    await marketplace.initialize()

    # Discover models
    recommendations = await marketplace.get_model_recommendations(
        user_id="user_123",
        task_type="image_classification",
        requirements={"accuracy": ">0.9", "latency": "<100ms"}
    )

    for model in recommendations:
        print(f"Recommended: {model.name}")
        print(f"Rating: {model.average_rating}")
        print(f"Downloads: {model.download_count}")

asyncio.run(explore_marketplace())
```

### 2. Distributed Computing

```python
from aicr.distributed_computing import DistributedComputingCluster

async def setup_distributed_cluster():
    cluster = DistributedComputingCluster()
    await cluster.initialize()

    # Configure auto-scaling
    scaling_config = {
        "min_nodes": 2,
        "max_nodes": 10,
        "target_cpu_utilization": 70,
        "scale_up_threshold": 80,
        "scale_down_threshold": 30
    }

    await cluster.configure_auto_scaling(scaling_config)

    # Deploy model to cluster
    await cluster.deploy_model_distributed(
        model_id=model.model_id,
        replication_factor=3
    )

    print("Distributed deployment completed")

asyncio.run(setup_distributed_cluster())
```

### 3. Monitoring and Analytics

```python
from aicr.monitoring import AIMonitoringSystem

async def setup_monitoring():
    monitoring = AIMonitoringSystem()
    await monitoring.initialize()

    # Get system health
    health = await monitoring.get_system_health()
    print(f"Overall health: {health['status']}")

    # Get performance summary
    performance = await monitoring.get_performance_summary(
        time_range="1h",
        component="inference_engine"
    )

    print(f"Average latency: {performance['avg_latency']}ms")
    print(f"Throughput: {performance['requests_per_second']} req/s")
    print(f"Error rate: {performance['error_rate']}%")

asyncio.run(setup_monitoring())
```

## Testing and Validation

### 1. Run Comprehensive Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/ci/test_models.py -v          # Unit tests
uv run pytest tests/ci/test_integration.py -v    # Integration tests
uv run pytest tests/ci/test_performance.py -v    # Performance tests
uv run pytest tests/ci/test_security.py -v       # Security tests
```

### 2. Validate Configuration

```python
async def validate_configuration():
    # Test database connection
    db_status = await service.test_database_connection()
    print(f"Database connection: {'✓' if db_status else '✗'}")

    # Test storage access
    storage_status = await service.test_storage_access()
    print(f"Storage access: {'✓' if storage_status else '✗'}")

    # Test security components
    security_status = await service.test_security_components()
    print(f"Security components: {'✓' if security_status else '✗'}")

asyncio.run(validate_configuration())
```

### 3. Performance Benchmarking

```python
import time

async def benchmark_inference():
    # Prepare test data
    test_requests = []
    for i in range(100):
        request = AICRInferenceRequest(
            model_id=model.model_id,
            input_data={"features": [1.0, 2.0, 3.0, 4.0, 5.0]},
            parameters={"batch_size": 1}
        )
        test_requests.append(request)

    # Benchmark single requests
    start_time = time.time()
    for request in test_requests:
        await service.run_inference(request)
    single_time = time.time() - start_time

    # Benchmark batch requests
    start_time = time.time()
    await service.run_batch_inference(test_requests)
    batch_time = time.time() - start_time

    print(f"Single requests: {single_time:.2f}s ({100/single_time:.1f} req/s)")
    print(f"Batch requests: {batch_time:.2f}s ({100/batch_time:.1f} req/s)")
    print(f"Batch speedup: {single_time/batch_time:.1f}x")

asyncio.run(benchmark_inference())
```

## Common Issues and Solutions

### 1. Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U aicr_user -d aicr_db -c "SELECT 1;"
```

### 2. Permission Issues

```bash
# Check file permissions
ls -la /opt/aicr/models/

# Fix permissions if needed
sudo chown -R $USER:$USER /opt/aicr/
chmod 755 /opt/aicr/models/
```

### 3. Port Conflicts

```bash
# Check if port is in use
netstat -tulpn | grep :8080

# Kill process using port
sudo kill -9 $(lsof -t -i:8080)
```

## Next Steps

Congratulations! You now have a working AICR installation. Here's what to explore next:

### 1. **Core Functionality**
- [Model Management Guide](guides/model_management.md)
- [Inference Guide](guides/inference_guide.md)
- [Security Configuration](guides/security_guide.md)

### 2. **Advanced Features**
- [Distributed AI Setup](guides/distributed_ai.md)
- [Federated Learning](guides/federated_learning.md)
- [Edge AI Deployment](guides/edge_ai.md)

### 3. **Production Deployment**
- [Docker Deployment](deployment/docker_deployment.md)
- [Kubernetes Deployment](deployment/kubernetes_deployment.md)
- [Production Configuration](deployment/production_setup.md)

### 4. **API Integration**
- [REST API Reference](api/rest_api.md)
- [WebSocket API](api/websocket_api.md)
- [Python SDK](api/python_api.md)

### 5. **Monitoring and Operations**
- [Monitoring Guide](guides/monitoring_guide.md)
- [Troubleshooting](troubleshooting.md)
- [Performance Tuning](guides/performance_tuning.md)

## Support

If you encounter issues or need help:

1. **Documentation**: Check the comprehensive documentation in the `docs/` directory
2. **Examples**: Review code examples in the `examples/` directory
3. **Tests**: Examine test cases in the `tests/` directory for usage patterns
4. **Community**: Join the APG community forums
5. **Support**: Contact support@datacraft.co.ke

---

**Happy AI building with AICR!** 🚀