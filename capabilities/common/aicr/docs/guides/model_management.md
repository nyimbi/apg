# Model Management Guide

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [Overview](#overview)
2. [Model Lifecycle](#model-lifecycle)
3. [Registering Models](#registering-models)
4. [Model Validation](#model-validation)
5. [Model Deployment](#model-deployment)
6. [Model Versioning](#model-versioning)
7. [Performance Monitoring](#performance-monitoring)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

Model management in AICR provides a comprehensive system for handling AI models throughout their entire lifecycle. This includes registration, validation, deployment, monitoring, and retirement of models in a production environment.

### Key Features

- **Multi-Framework Support**: PyTorch, TensorFlow, ONNX, and custom frameworks
- **Automated Validation**: Schema validation, performance testing, and security checks
- **Version Control**: Complete model versioning with rollback capabilities
- **Deployment Management**: Automated deployment with scaling and health monitoring
- **Performance Tracking**: Real-time monitoring of model performance and metrics

### Supported Model Types

| Type | Description | Frameworks | Use Cases |
|------|-------------|------------|-----------|
| `classification` | Classification models | PyTorch, TensorFlow, ONNX | Image classification, sentiment analysis |
| `regression` | Regression models | PyTorch, TensorFlow, ONNX | Price prediction, forecasting |
| `clustering` | Clustering algorithms | Scikit-learn, PyTorch | Customer segmentation, anomaly detection |
| `nlp` | Natural language processing | Transformers, BERT, GPT | Text analysis, translation |
| `computer_vision` | Image/video processing | CNN, ResNet, YOLO | Object detection, image recognition |
| `time_series` | Time series analysis | LSTM, ARIMA, Prophet | Forecasting, trend analysis |
| `recommendation` | Recommendation systems | Collaborative filtering, Matrix factorization | Product recommendations, content filtering |

## Model Lifecycle

The AICR model lifecycle consists of several stages:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Develop    │ -> │  Register   │ -> │  Validate   │ -> │   Deploy    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ^                                                         │
       │                                                         v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Retire    │ <- │   Monitor   │ <- │   Manage    │ <- │   Serve     │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Stage Descriptions

1. **Develop**: Create and train your model using your preferred framework
2. **Register**: Register the model with AICR, providing metadata and artifacts
3. **Validate**: Automated validation of model format, schema, and performance
4. **Deploy**: Deploy the model to production with scaling and monitoring
5. **Serve**: Handle inference requests and provide predictions
6. **Manage**: Update configurations, scale resources, and maintain the model
7. **Monitor**: Track performance, usage, and health metrics
8. **Retire**: Gracefully retire models when they're no longer needed

## Registering Models

### Step 1: Prepare Model Metadata

Create a comprehensive model specification:

```python
from aicr import AICRClient
from aicr.models import ModelCreate, ModelType

model_metadata = ModelCreate(
    name="sentiment_analyzer_v2",
    description="BERT-based sentiment analysis model with enhanced accuracy for social media text",
    model_type=ModelType.CLASSIFICATION,
    framework="pytorch",
    version="2.0.0",

    # Input/Output schemas define the expected data format
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Input text for sentiment analysis",
                "minLength": 1,
                "maxLength": 512
            },
            "language": {
                "type": "string",
                "description": "Language code (optional)",
                "enum": ["en", "es", "fr", "de"],
                "default": "en"
            }
        },
        "required": ["text"]
    },

    output_schema={
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Predicted sentiment"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence score for the prediction"
            },
            "probabilities": {
                "type": "object",
                "properties": {
                    "positive": {"type": "number"},
                    "negative": {"type": "number"},
                    "neutral": {"type": "number"}
                },
                "description": "Probability distribution across all classes"
            }
        },
        "required": ["sentiment", "confidence"]
    },

    # Model configuration parameters
    configuration={
        "max_sequence_length": 512,
        "device": "auto",  # auto, cpu, cuda, mps
        "precision": "fp16",  # fp32, fp16, int8
        "batch_size": 32,
        "num_classes": 3,
        "model_architecture": "bert-base-uncased"
    },

    # Performance metrics from training/validation
    performance_metrics={
        "accuracy": 0.94,
        "precision": 0.92,
        "recall": 0.93,
        "f1_score": 0.925,
        "auc": 0.97,
        "training_loss": 0.15,
        "validation_loss": 0.18
    },

    # Tags for organization and discovery
    tags=["nlp", "sentiment", "bert", "social_media", "production"],

    # Additional metadata
    metadata={
        "license": "Apache-2.0",
        "training_dataset": "social_media_sentiment_v2",
        "training_duration_hours": 6.5,
        "data_size": "2.5M samples",
        "base_model": "bert-base-uncased",
        "fine_tuning_epochs": 3,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "author": "ML Team",
        "contact": "ml-team@example.com",
        "documentation_url": "https://docs.company.com/models/sentiment-v2"
    }
)
```

### Step 2: Register the Model

```python
client = AICRClient()

# Register model metadata
model = client.models.create(model_metadata)

print(f"Model registered successfully!")
print(f"Model ID: {model.model_id}")
print(f"Upload URL: {model.upload_info.upload_url}")
print(f"Upload expires: {model.upload_info.expires_at}")
```

### Step 3: Upload Model Artifacts

```python
from pathlib import Path

# Prepare model files
model_file = Path("./sentiment_model_v2.pth")
config_file = Path("./model_config.json")
tokenizer_file = Path("./tokenizer.json")
requirements_file = Path("./requirements.txt")

# Upload with progress tracking
def upload_progress(uploaded_bytes: int, total_bytes: int):
    percent = (uploaded_bytes / total_bytes) * 100
    print(f"Upload progress: {percent:.1f}% ({uploaded_bytes:,}/{total_bytes:,} bytes)")

upload_result = client.models.upload_artifacts(
    model_id=model.model_id,
    model_file=model_file,
    config_file=config_file,
    additional_files={
        "tokenizer.json": tokenizer_file,
        "requirements.txt": requirements_file
    },
    progress_callback=upload_progress
)

print(f"Upload completed: {upload_result.status}")
print(f"Files uploaded: {len(upload_result.files)}")

# Wait for processing to complete
processing_result = client.models.wait_for_processing(
    model_id=model.model_id,
    timeout=600  # 10 minutes
)

if processing_result.is_successful:
    print("Model processing completed successfully!")
    print(f"Model status: {processing_result.model.status}")
else:
    print(f"Processing failed: {processing_result.error_message}")
```

### Step 4: Verify Registration

```python
# Get updated model information
updated_model = client.models.get(model.model_id)

print(f"Model: {updated_model.name}")
print(f"Status: {updated_model.status}")
print(f"Size: {updated_model.size_mb:.1f} MB")
print(f"Checksum: {updated_model.checksum}")

# Verify model can be loaded
validation_result = client.models.validate(model.model_id)

if validation_result.is_valid:
    print("✅ Model validation passed")
    print(f"Validation score: {validation_result.score}")
else:
    print("❌ Model validation failed")
    for issue in validation_result.issues:
        print(f"  - {issue.severity}: {issue.message}")
```

## Model Validation

AICR performs comprehensive validation of all registered models:

### Automated Validation Checks

#### 1. Format Validation
```python
# Check if model file format is supported
format_check = client.models.validate_format(model.model_id)

print(f"Format: {format_check.detected_format}")
print(f"Framework: {format_check.framework}")
print(f"Compatible: {format_check.is_compatible}")
```

#### 2. Schema Validation
```python
# Validate input/output schemas with sample data
schema_validation = client.models.validate_schema(
    model_id=model.model_id,
    sample_inputs=[
        {"text": "I love this product!"},
        {"text": "This is terrible.", "language": "en"}
    ]
)

print(f"Schema validation: {schema_validation.is_valid}")
if not schema_validation.is_valid:
    for error in schema_validation.errors:
        print(f"  - {error.field}: {error.message}")
```

#### 3. Performance Validation
```python
# Run performance benchmarks
performance_test = client.models.run_performance_test(
    model_id=model.model_id,
    test_cases=[
        {"text": "Great product!"},
        {"text": "Not satisfied."},
        {"text": "It's okay."}
    ],
    metrics=["latency", "throughput", "accuracy"]
)

print(f"Performance Results:")
print(f"  Average Latency: {performance_test.avg_latency_ms:.1f}ms")
print(f"  Throughput: {performance_test.throughput_rps:.1f} req/s")
print(f"  Memory Usage: {performance_test.memory_usage_mb:.1f} MB")
```

#### 4. Security Validation
```python
# Security and safety checks
security_scan = client.models.run_security_scan(model.model_id)

print(f"Security Scan Results:")
print(f"  Status: {security_scan.status}")
print(f"  Risk Level: {security_scan.risk_level}")

if security_scan.vulnerabilities:
    print("  Vulnerabilities found:")
    for vuln in security_scan.vulnerabilities:
        print(f"    - {vuln.severity}: {vuln.description}")
```

### Custom Validation

Add custom validation rules:

```python
from aicr.validation import ValidationRule, ValidationSeverity

class CustomAccuracyRule(ValidationRule):
    def __init__(self, min_accuracy: float = 0.8):
        self.min_accuracy = min_accuracy

    def validate(self, model) -> ValidationResult:
        if model.performance_metrics.get("accuracy", 0) < self.min_accuracy:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Model accuracy {model.performance_metrics['accuracy']:.3f} below minimum {self.min_accuracy}"
            )
        return ValidationResult(is_valid=True)

# Register custom validation rule
client.models.add_validation_rule(CustomAccuracyRule(min_accuracy=0.9))

# Run validation with custom rules
validation_result = client.models.validate(model.model_id, include_custom_rules=True)
```

## Model Deployment

### Basic Deployment

```python
from aicr.deployment import DeploymentConfig

# Configure deployment
deployment_config = DeploymentConfig(
    instance_type="gpu_medium",  # cpu_small, cpu_medium, gpu_small, gpu_medium, gpu_large
    min_instances=2,
    max_instances=10,
    auto_scaling=True,
    health_check_enabled=True,
    environment="production"
)

# Deploy the model
deployment = client.models.deploy(
    model_id=model.model_id,
    config=deployment_config
)

print(f"Deployment initiated: {deployment.deployment_id}")
print(f"Status: {deployment.status}")
print(f"Estimated ready time: {deployment.estimated_ready_time}")
```

### Advanced Deployment Configuration

```python
from aicr.deployment import DeploymentConfig, AutoScalingConfig, HealthCheckConfig

# Advanced auto-scaling configuration
auto_scaling = AutoScalingConfig(
    min_instances=2,
    max_instances=20,
    target_cpu_utilization=70,
    target_memory_utilization=80,
    scale_up_threshold=85,
    scale_down_threshold=30,
    scale_up_cooldown=300,  # 5 minutes
    scale_down_cooldown=600,  # 10 minutes
    custom_metrics=[
        {
            "name": "inference_queue_length",
            "target_value": 10,
            "scale_up_threshold": 20
        },
        {
            "name": "response_time_p95",
            "target_value": 100,  # ms
            "scale_up_threshold": 200
        }
    ]
)

# Health check configuration
health_check = HealthCheckConfig(
    path="/health",
    interval_seconds=30,
    timeout_seconds=10,
    healthy_threshold=2,
    unhealthy_threshold=3,
    grace_period_seconds=300
)

# Complete deployment configuration
deployment_config = DeploymentConfig(
    instance_type="gpu_medium",
    auto_scaling=auto_scaling,
    health_check=health_check,
    environment="production",
    tags={
        "team": "ml-platform",
        "cost_center": "research",
        "project": "sentiment-analysis"
    },
    environment_variables={
        "LOG_LEVEL": "INFO",
        "CACHE_SIZE": "1GB",
        "MAX_BATCH_SIZE": "32"
    },
    resource_limits={
        "cpu": "4",
        "memory": "8Gi",
        "gpu": "1"
    }
)

# Deploy with advanced configuration
deployment = client.models.deploy(
    model_id=model.model_id,
    config=deployment_config,
    wait_for_ready=True,
    timeout=600  # 10 minutes
)

if deployment.is_ready:
    print(f"✅ Model deployed successfully!")
    print(f"Endpoint: {deployment.endpoint}")
    print(f"Running instances: {deployment.instance_count}")
    print(f"Health status: {deployment.health_status}")
else:
    print(f"❌ Deployment failed: {deployment.error_message}")
```

### Blue-Green Deployment

```python
# Blue-green deployment for zero-downtime updates
blue_green_deployment = client.models.deploy_blue_green(
    current_model_id="mdl_v1_123",  # Currently serving model
    new_model_id=model.model_id,    # New model to deploy
    traffic_shift_strategy="gradual",  # gradual, immediate, canary
    validation_tests=[
        {
            "name": "accuracy_test",
            "sample_size": 1000,
            "min_accuracy": 0.9
        },
        {
            "name": "latency_test",
            "duration_minutes": 5,
            "max_p95_latency": 100
        }
    ]
)

# Monitor deployment progress
for update in client.deployments.stream_progress(blue_green_deployment.deployment_id):
    print(f"Stage: {update.stage}")
    print(f"Progress: {update.progress_percent}%")
    print(f"Traffic split: {update.traffic_split}")

    if update.validation_results:
        for test_name, result in update.validation_results.items():
            status = "✅" if result.passed else "❌"
            print(f"  {status} {test_name}: {result.message}")

print(f"Blue-green deployment completed: {blue_green_deployment.status}")
```

### Canary Deployment

```python
# Canary deployment with gradual traffic shift
canary_deployment = client.models.deploy_canary(
    stable_model_id="mdl_v1_123",
    canary_model_id=model.model_id,
    canary_traffic_percent=10,  # Start with 10% traffic
    success_criteria={
        "error_rate_threshold": 0.01,  # Max 1% error rate
        "latency_p95_threshold": 150,  # Max 150ms p95 latency
        "min_sample_size": 500         # Minimum requests for evaluation
    },
    traffic_shift_schedule=[
        {"traffic_percent": 10, "duration_minutes": 30},
        {"traffic_percent": 25, "duration_minutes": 30},
        {"traffic_percent": 50, "duration_minutes": 30},
        {"traffic_percent": 100, "duration_minutes": 0}  # Full traffic
    ]
)

print(f"Canary deployment started: {canary_deployment.deployment_id}")
```

## Model Versioning

### Version Management

```python
# List all versions of a model
versions = client.models.list_versions("sentiment_analyzer")

for version in versions:
    print(f"Version {version.version}: {version.status}")
    print(f"  Created: {version.created_at}")
    print(f"  Accuracy: {version.performance_metrics.get('accuracy', 'N/A')}")
    print(f"  Deployed: {'Yes' if version.is_deployed else 'No'}")

# Get specific version
model_v1 = client.models.get("sentiment_analyzer", version="1.0.0")
model_v2 = client.models.get("sentiment_analyzer", version="2.0.0")

# Compare versions
comparison = client.models.compare_versions(
    model_name="sentiment_analyzer",
    version_a="1.0.0",
    version_b="2.0.0",
    metrics=["accuracy", "latency", "memory_usage"]
)

print("Version Comparison:")
for metric, values in comparison.metrics.items():
    v1_value = values["1.0.0"]
    v2_value = values["2.0.0"]
    improvement = ((v2_value - v1_value) / v1_value) * 100
    print(f"  {metric}: v1={v1_value:.3f}, v2={v2_value:.3f} ({improvement:+.1f}%)")
```

### Model Promotion

```python
# Promote model from staging to production
promotion = client.models.promote(
    model_id=model.model_id,
    from_environment="staging",
    to_environment="production",
    approval_required=True,
    rollback_plan={
        "auto_rollback": True,
        "error_rate_threshold": 0.05,
        "latency_threshold": 200
    }
)

print(f"Promotion request: {promotion.promotion_id}")
print(f"Status: {promotion.status}")

if promotion.approval_required:
    print(f"Approval URL: {promotion.approval_url}")

    # Wait for approval and monitor promotion
    final_status = client.models.wait_for_promotion(
        promotion.promotion_id,
        timeout=1800  # 30 minutes
    )

    print(f"Final status: {final_status.status}")
```

### Rollback

```python
# Rollback to previous version
rollback = client.models.rollback(
    model_name="sentiment_analyzer",
    target_version="1.0.0",
    reason="Performance degradation in v2.0.0"
)

print(f"Rollback initiated: {rollback.rollback_id}")

# Monitor rollback progress
rollback_status = client.models.wait_for_rollback(rollback.rollback_id)

if rollback_status.is_successful:
    print("✅ Rollback completed successfully")
    print(f"Active version: {rollback_status.active_version}")
else:
    print(f"❌ Rollback failed: {rollback_status.error_message}")
```

## Performance Monitoring

### Real-time Metrics

```python
from aicr.monitoring import MetricQuery, TimeRange

# Get real-time performance metrics
metrics = client.monitoring.get_model_metrics(
    model_id=model.model_id,
    metrics=["latency", "throughput", "error_rate", "accuracy"],
    time_range=TimeRange.LAST_1_HOUR
)

print("Performance Metrics (Last 1 Hour):")
for metric_name, data in metrics.items():
    print(f"  {metric_name}:")
    print(f"    Current: {data.current_value}")
    print(f"    Average: {data.statistics.avg:.2f}")
    print(f"    P95: {data.statistics.p95:.2f}")
    print(f"    P99: {data.statistics.p99:.2f}")
```

### Performance Alerts

```python
from aicr.monitoring import AlertRule, AlertSeverity, AlertChannel

# Create performance alert rules
latency_alert = AlertRule(
    name="high_latency_alert",
    metric="inference_latency_p95",
    condition="greater_than",
    threshold=200,  # 200ms
    duration_minutes=5,
    severity=AlertSeverity.WARNING,
    channels=[
        AlertChannel.EMAIL,
        AlertChannel.SLACK
    ]
)

error_rate_alert = AlertRule(
    name="high_error_rate_alert",
    metric="error_rate",
    condition="greater_than",
    threshold=0.05,  # 5%
    duration_minutes=2,
    severity=AlertSeverity.CRITICAL,
    channels=[
        AlertChannel.EMAIL,
        AlertChannel.PAGERDUTY
    ]
)

# Register alert rules
client.monitoring.create_alert_rule(model.model_id, latency_alert)
client.monitoring.create_alert_rule(model.model_id, error_rate_alert)

print("Alert rules created successfully")
```

### Performance Dashboard

```python
# Create custom performance dashboard
dashboard = client.monitoring.create_dashboard(
    name=f"Model Performance - {model.name}",
    model_id=model.model_id,
    widgets=[
        {
            "type": "time_series",
            "title": "Inference Latency",
            "metrics": ["inference_latency_p50", "inference_latency_p95"],
            "time_range": "24h"
        },
        {
            "type": "gauge",
            "title": "Current Throughput",
            "metric": "requests_per_second",
            "max_value": 1000
        },
        {
            "type": "bar_chart",
            "title": "Error Rate by Type",
            "metric": "error_rate",
            "group_by": "error_type"
        },
        {
            "type": "heatmap",
            "title": "Response Time Distribution",
            "metric": "response_time",
            "time_range": "7d"
        }
    ]
)

print(f"Dashboard created: {dashboard.url}")
```

## Best Practices

### 1. Model Organization

```python
# Use consistent naming conventions
naming_convention = {
    "format": "{model_type}_{use_case}_{version}",
    "examples": [
        "classification_sentiment_v1",
        "regression_price_prediction_v2",
        "nlp_text_summarization_v1"
    ]
}

# Organize models with tags
model_tags = {
    "environment": ["development", "staging", "production"],
    "team": ["data_science", "ml_engineering", "research"],
    "use_case": ["customer_facing", "internal", "research"],
    "framework": ["pytorch", "tensorflow", "onnx"],
    "model_type": ["classification", "regression", "clustering"]
}
```

### 2. Schema Design

```python
# Design robust input/output schemas
input_schema = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512,
            "pattern": "^[a-zA-Z0-9\\s\\.,!?]+$",  # Basic text validation
            "description": "Input text for analysis"
        },
        "metadata": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["en", "es", "fr"]},
                "source": {"type": "string", "enum": ["web", "mobile", "api"]},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "additionalProperties": False
        }
    },
    "required": ["text"],
    "additionalProperties": False
}

output_schema = {
    "type": "object",
    "properties": {
        "prediction": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "probabilities": {
            "type": "object",
            "patternProperties": {
                "^(positive|negative|neutral)$": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "additionalProperties": False
        },
        "metadata": {
            "type": "object",
            "properties": {
                "model_version": {"type": "string"},
                "processing_time_ms": {"type": "number"},
                "confidence_band": {"type": "string", "enum": ["high", "medium", "low"]}
            }
        }
    },
    "required": ["prediction", "confidence"]
}
```

### 3. Performance Optimization

```python
# Model optimization best practices

# 1. Use appropriate precision
optimization_config = {
    "precision": "fp16",  # Use fp16 for inference to save memory
    "optimization_level": "O2",  # TensorRT optimization level
    "batch_size": 32,  # Optimal batch size for your hardware
    "max_sequence_length": 512,  # Truncate to reasonable length
}

# 2. Enable model caching
caching_config = {
    "enable_model_cache": True,
    "cache_size": "2GB",
    "cache_ttl_hours": 24,
    "warm_up_requests": 100  # Pre-warm the cache
}

# 3. Configure appropriate timeouts
timeout_config = {
    "inference_timeout": 30,  # seconds
    "model_load_timeout": 300,  # seconds
    "health_check_timeout": 10  # seconds
}

# Apply optimizations
client.models.update_configuration(
    model_id=model.model_id,
    optimization=optimization_config,
    caching=caching_config,
    timeouts=timeout_config
)
```

### 4. Security Best Practices

```python
# Security configuration
security_config = {
    "enable_input_validation": True,
    "sanitize_inputs": True,
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 1000,
        "burst_size": 100
    },
    "access_control": {
        "require_authentication": True,
        "allowed_roles": ["ml_engineer", "data_scientist"],
        "ip_whitelist": ["10.0.0.0/8", "192.168.0.0/16"]
    },
    "audit_logging": {
        "enabled": True,
        "log_inputs": False,  # Don't log sensitive input data
        "log_outputs": True,
        "log_performance": True
    }
}

client.models.update_security_config(model.model_id, security_config)
```

## Troubleshooting

### Common Issues

#### 1. Model Upload Failures

```python
# Debug upload issues
try:
    upload_result = client.models.upload_artifacts(
        model_id=model.model_id,
        model_file=model_file
    )
except Exception as e:
    print(f"Upload failed: {e}")

    # Check upload prerequisites
    upload_info = client.models.get_upload_info(model.model_id)
    print(f"Upload URL valid: {upload_info.is_valid}")
    print(f"Upload expires: {upload_info.expires_at}")
    print(f"Max file size: {upload_info.max_file_size_mb} MB")

    # Check file size
    file_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"Your file size: {file_size_mb:.1f} MB")

    if file_size_mb > upload_info.max_file_size_mb:
        print("❌ File too large. Consider model compression.")

    # Retry upload with chunked transfer
    upload_result = client.models.upload_artifacts(
        model_id=model.model_id,
        model_file=model_file,
        chunk_size=1024*1024*10,  # 10MB chunks
        max_retries=3
    )
```

#### 2. Deployment Issues

```python
# Debug deployment problems
deployment = client.models.get_deployment(deployment_id)

if deployment.status == "failed":
    print(f"Deployment failed: {deployment.error_message}")

    # Check deployment logs
    logs = client.deployments.get_logs(deployment_id, lines=100)
    for log_entry in logs:
        if log_entry.level in ["ERROR", "FATAL"]:
            print(f"❌ {log_entry.timestamp}: {log_entry.message}")

    # Check resource availability
    resource_check = client.system.check_resources(
        instance_type=deployment.config.instance_type,
        region=deployment.config.region
    )

    if not resource_check.available:
        print(f"❌ Insufficient resources: {resource_check.message}")
        print(f"Available alternatives: {resource_check.alternatives}")

# Retry deployment with different configuration
if deployment.status == "failed":
    new_config = deployment.config.copy()
    new_config.instance_type = "cpu_large"  # Fallback to CPU

    retry_deployment = client.models.deploy(
        model_id=model.model_id,
        config=new_config
    )
```

#### 3. Performance Issues

```python
# Diagnose performance problems
performance_report = client.models.diagnose_performance(model.model_id)

print("Performance Diagnosis:")
print(f"  Current latency P95: {performance_report.latency_p95}ms")
print(f"  Target latency P95: {performance_report.target_latency}ms")

if performance_report.latency_p95 > performance_report.target_latency:
    print("\n🔍 Performance Issues Detected:")
    for issue in performance_report.issues:
        print(f"  - {issue.category}: {issue.description}")
        print(f"    Recommended action: {issue.recommendation}")

# Apply performance recommendations
if performance_report.recommendations:
    for rec in performance_report.recommendations:
        if rec.type == "batch_size_optimization":
            client.models.update_configuration(
                model.model_id,
                {"batch_size": rec.suggested_value}
            )
        elif rec.type == "instance_type_upgrade":
            client.deployments.scale_up(
                deployment_id,
                instance_type=rec.suggested_instance_type
            )
```

### Getting Help

```python
# Get model status and diagnostics
model_status = client.models.get_detailed_status(model.model_id)

print("Model Status Report:")
print(f"  Status: {model_status.status}")
print(f"  Health Score: {model_status.health_score}/100")
print(f"  Last Updated: {model_status.last_updated}")

# Export model information for support
support_bundle = client.models.export_support_bundle(
    model.model_id,
    include_logs=True,
    include_metrics=True,
    time_range="24h"
)

print(f"Support bundle created: {support_bundle.download_url}")
print(f"Bundle expires: {support_bundle.expires_at}")
print(f"Reference ID: {support_bundle.bundle_id}")
```

---

**Next Steps:**
- [Inference Guide](inference_guide.md) - Learn how to run inference with your deployed models
- [Security Guide](security_guide.md) - Secure your models and deployments
- [Monitoring Guide](monitoring_guide.md) - Monitor model performance and health
- [API Reference](../api/rest_api.md) - Detailed API documentation