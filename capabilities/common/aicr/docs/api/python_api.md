# AICR Python SDK Reference

**Version:** 1.0.0
**Author:** Nyimbi Odero <nyimbi@gmail.com>
**Copyright:** © 2025 Datacraft

## Table of Contents

1. [SDK Overview](#sdk-overview)
2. [Installation](#installation)
3. [Client Configuration](#client-configuration)
4. [Authentication](#authentication)
5. [Model Management](#model-management)
6. [Inference Operations](#inference-operations)
7. [Pipeline Management](#pipeline-management)
8. [Monitoring & Analytics](#monitoring--analytics)
9. [Async Operations](#async-operations)
10. [Error Handling](#error-handling)
11. [Advanced Features](#advanced-features)

## SDK Overview

The AICR Python SDK provides a comprehensive, type-safe interface for interacting with the AI Core Framework. Built with modern Python practices, it offers both synchronous and asynchronous APIs for maximum flexibility.

### Key Features

- **Type Safety**: Full type hints and Pydantic model validation
- **Async Support**: Complete async/await API for high-performance applications
- **Auto-retry**: Intelligent retry logic with exponential backoff
- **Caching**: Built-in response caching for improved performance
- **Streaming**: Support for real-time inference and file streaming
- **Error Handling**: Comprehensive error types with detailed context

### Python Version Support

- **Python 3.8+**: Minimum supported version
- **Python 3.12+**: Recommended for best performance
- **Type Checking**: Full support for mypy, pyright, and pylint

## Installation

### Using pip

```bash
pip install aicr-python-sdk
```

### Using Poetry

```bash
poetry add aicr-python-sdk
```

### Development Installation

```bash
git clone https://github.com/datacraft/aicr-python-sdk
cd aicr-python-sdk
pip install -e ".[dev]"
```

### Optional Dependencies

```bash
# For async HTTP support
pip install aicr-python-sdk[async]

# For advanced features
pip install aicr-python-sdk[advanced]

# For development tools
pip install aicr-python-sdk[dev]

# Install all extras
pip install aicr-python-sdk[all]
```

## Client Configuration

### Basic Client Setup

```python
from aicr import AICRClient

# Basic client with API key
client = AICRClient(
    api_key="your_api_key_here",
    base_url="https://api.datacraft.co.ke/aicr/v1"
)

# Client with JWT token
client = AICRClient(
    token="your_jwt_token_here",
    base_url="https://api.datacraft.co.ke/aicr/v1"
)

# Client with custom configuration
client = AICRClient(
    api_key="your_api_key_here",
    base_url="https://api.datacraft.co.ke/aicr/v1",
    timeout=30.0,
    max_retries=3,
    retry_delay=1.0,
    verify_ssl=True,
    user_agent="MyApp/1.0.0"
)
```

### Environment Configuration

Set environment variables for automatic configuration:

```bash
export AICR_API_KEY="your_api_key_here"
export AICR_BASE_URL="https://api.datacraft.co.ke/aicr/v1"
export AICR_TIMEOUT=30
export AICR_MAX_RETRIES=3
```

```python
from aicr import AICRClient

# Automatically uses environment variables
client = AICRClient()
```

### Configuration File

Create `~/.aicr/config.yaml`:

```yaml
api_key: "your_api_key_here"
base_url: "https://api.datacraft.co.ke/aicr/v1"
timeout: 30.0
max_retries: 3
retry_delay: 1.0
verify_ssl: true
user_agent: "AICR-Python-SDK/1.0.0"

# Profile-specific configuration
profiles:
  development:
    base_url: "http://localhost:8080/api/v1"
    verify_ssl: false

  staging:
    base_url: "https://staging-api.datacraft.co.ke/aicr/v1"
    timeout: 60.0

  production:
    base_url: "https://api.datacraft.co.ke/aicr/v1"
    timeout: 30.0
    max_retries: 5
```

```python
from aicr import AICRClient

# Use default profile
client = AICRClient.from_config()

# Use specific profile
client = AICRClient.from_config(profile="development")
```

## Authentication

### API Key Authentication

```python
from aicr import AICRClient

client = AICRClient(api_key="aicr_api_key_abc123xyz789")

# Verify authentication
user_info = client.auth.get_user_info()
print(f"Authenticated as: {user_info.username}")
```

### JWT Token Authentication

```python
from aicr import AICRClient
from aicr.auth import AuthCredentials

# Login with username/password
auth = AuthCredentials(
    username="user@example.com",
    password="secure_password"
)

client = AICRClient()
token_info = client.auth.login(auth)

print(f"Access token: {token_info.access_token}")
print(f"Expires in: {token_info.expires_in} seconds")

# Use the token for subsequent requests
client = AICRClient(token=token_info.access_token)
```

### OAuth2 Authentication

```python
from aicr.auth import OAuth2Flow

# OAuth2 authorization code flow
oauth = OAuth2Flow(
    client_id="your_client_id",
    client_secret="your_client_secret",
    redirect_uri="https://your-app.com/callback"
)

# Get authorization URL
auth_url = oauth.get_authorization_url(
    scopes=["model:read", "inference:execute"],
    state="random_state_string"
)

print(f"Visit: {auth_url}")

# Exchange authorization code for token
token_info = oauth.exchange_code("authorization_code_from_callback")

client = AICRClient(token=token_info.access_token)
```

### Token Refresh

```python
from aicr import AICRClient

client = AICRClient(token="your_jwt_token")

# Automatic token refresh (if refresh token available)
client.auth.enable_auto_refresh(refresh_token="your_refresh_token")

# Manual token refresh
new_token_info = client.auth.refresh_token("your_refresh_token")
client.set_token(new_token_info.access_token)
```

## Model Management

### Model Data Types

```python
from aicr.models import (
    Model,
    ModelCreate,
    ModelUpdate,
    ModelFilter,
    ModelType,
    ModelStatus
)
from typing import Optional, List, Dict, Any

# Model creation data
model_data = ModelCreate(
    name="sentiment_analyzer_v2",
    description="BERT-based sentiment analysis model",
    model_type=ModelType.CLASSIFICATION,
    framework="pytorch",
    version="2.0.0",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 512}
        },
        "required": ["text"]
    },
    output_schema={
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
    },
    tags=["nlp", "sentiment", "bert"],
    metadata={
        "license": "Apache-2.0",
        "training_dataset": "sentiment140_extended"
    }
)
```

### List Models

```python
from aicr import AICRClient
from aicr.models import ModelFilter, ModelSort

client = AICRClient()

# List all models
models = client.models.list()
print(f"Found {len(models)} models")

# List with pagination
models_page = client.models.list(limit=10, cursor="next_cursor")

# List with filtering
models = client.models.list(
    filter=ModelFilter(
        model_type=ModelType.CLASSIFICATION,
        framework="pytorch",
        status=ModelStatus.ACTIVE,
        tags=["nlp"]
    ),
    sort=ModelSort.CREATED_DESC,
    limit=20
)

# List with field selection
models = client.models.list(
    fields=["model_id", "name", "status", "created_at"],
    limit=50
)

for model in models:
    print(f"{model.name} ({model.model_id}) - {model.status}")
```

### Get Model Details

```python
from aicr import AICRClient
from aicr.exceptions import ModelNotFoundError

client = AICRClient()

try:
    model = client.models.get("mdl_abc123")

    print(f"Model: {model.name}")
    print(f"Type: {model.model_type}")
    print(f"Framework: {model.framework}")
    print(f"Status: {model.status}")
    print(f"Performance: {model.performance_metrics}")

    if model.is_deployed:
        print(f"Endpoint: {model.deployment_info.endpoint}")
        print(f"Instances: {model.deployment_info.instances}")

except ModelNotFoundError as e:
    print(f"Model not found: {e}")
```

### Register New Model

```python
from aicr import AICRClient
from aicr.models import ModelCreate
from pathlib import Path

client = AICRClient()

# Register model metadata
model_data = ModelCreate(
    name="image_classifier_v3",
    description="ResNet-50 based image classifier",
    model_type=ModelType.CLASSIFICATION,
    framework="pytorch",
    version="3.0.0",
    tags=["computer_vision", "resnet", "imagenet"]
)

model = client.models.create(model_data)
print(f"Model registered: {model.model_id}")

# Upload model artifacts
model_file = Path("./model.pth")
config_file = Path("./config.json")

upload_result = client.models.upload_artifacts(
    model_id=model.model_id,
    model_file=model_file,
    config_file=config_file,
    progress_callback=lambda uploaded, total: print(f"Upload: {uploaded}/{total}")
)

print(f"Upload completed: {upload_result.status}")
```

### Update Model

```python
from aicr import AICRClient
from aicr.models import ModelUpdate

client = AICRClient()

# Update model metadata
updates = ModelUpdate(
    description="Updated image classifier with improved accuracy",
    version="3.1.0",
    performance_metrics={
        "accuracy": 0.94,
        "precision": 0.92,
        "recall": 0.93,
        "f1_score": 0.925
    },
    tags=["computer_vision", "resnet", "imagenet", "production"]
)

updated_model = client.models.update("mdl_abc123", updates)
print(f"Model updated to version: {updated_model.version}")
```

### Delete Model

```python
from aicr import AICRClient

client = AICRClient()

# Delete model
deletion_result = client.models.delete("mdl_abc123")
print(f"Deletion initiated: {deletion_result.operation_id}")

# Wait for deletion to complete
operation = client.operations.wait_for_completion(
    deletion_result.operation_id,
    timeout=300  # 5 minutes
)

if operation.is_successful:
    print("Model deleted successfully")
else:
    print(f"Deletion failed: {operation.error_message}")
```

## Inference Operations

### Inference Data Types

```python
from aicr.inference import (
    InferenceRequest,
    InferenceResponse,
    BatchInferenceRequest,
    AsyncInferenceRequest,
    InferenceStatus,
    Priority
)
from typing import Dict, Any, List

# Single inference request
request = InferenceRequest(
    model_id="mdl_abc123",
    input_data={"text": "I love this product!"},
    parameters={
        "confidence_threshold": 0.8,
        "return_probabilities": True,
        "temperature": 0.7
    },
    output_format="json",
    priority=Priority.NORMAL,
    timeout_seconds=30
)
```

### Single Inference

```python
from aicr import AICRClient
from aicr.inference import InferenceRequest

client = AICRClient()

# Simple inference
response = client.inference.predict(
    model_id="mdl_sentiment_123",
    input_data={"text": "I love this product! It's amazing."}
)

print(f"Sentiment: {response.predictions['sentiment']}")
print(f"Confidence: {response.predictions['confidence']}")
print(f"Processing time: {response.processing_time_ms}ms")

# Inference with parameters
response = client.inference.predict(
    model_id="mdl_sentiment_123",
    input_data={"text": "This is a great product!"},
    parameters={
        "confidence_threshold": 0.8,
        "return_probabilities": True,
        "explain_prediction": True
    }
)

if response.is_successful:
    print(f"Prediction: {response.predictions}")
    print(f"Probabilities: {response.predictions['probabilities']}")
    if 'explanation' in response.metadata:
        print(f"Explanation: {response.metadata['explanation']}")
```

### Batch Inference

```python
from aicr import AICRClient
from aicr.inference import BatchInferenceRequest

client = AICRClient()

# Prepare batch inputs
inputs = [
    {"input_id": "1", "data": {"text": "Great product!"}},
    {"input_id": "2", "data": {"text": "Terrible service."}},
    {"input_id": "3", "data": {"text": "It's okay, nothing special."}},
    {"input_id": "4", "data": {"text": "Absolutely love it!"}},
    {"input_id": "5", "data": {"text": "Waste of money."}}
]

# Execute batch inference
batch_response = client.inference.batch_predict(
    model_id="mdl_sentiment_123",
    inputs=inputs,
    parameters={"batch_size": 16},
    progress_callback=lambda completed, total: print(f"Progress: {completed}/{total}")
)

# Process results
for result in batch_response.results:
    if result.is_successful:
        sentiment = result.predictions["sentiment"]
        confidence = result.predictions["confidence"]
        print(f"Input {result.input_id}: {sentiment} ({confidence:.2f})")
    else:
        print(f"Input {result.input_id}: Error - {result.error_message}")

print(f"Batch summary: {batch_response.summary}")
```

### Async Inference

```python
from aicr import AICRClient
from aicr.inference import AsyncInferenceRequest
import time

client = AICRClient()

# Start async inference
async_request = AsyncInferenceRequest(
    model_id="mdl_large_model_456",
    input_data={"large_dataset": "..."},
    parameters={"processing_mode": "comprehensive"},
    callback_url="https://your-app.com/inference_callback"
)

job = client.inference.async_predict(async_request)
print(f"Job started: {job.job_id}")
print(f"Estimated completion: {job.estimated_completion}")

# Poll for completion
while not job.is_completed:
    time.sleep(10)
    job = client.inference.get_job_status(job.job_id)
    print(f"Status: {job.status} - {job.progress_percent}%")

if job.is_successful:
    result = client.inference.get_job_result(job.job_id)
    print(f"Result: {result.predictions}")
else:
    print(f"Job failed: {job.error_message}")
```

### Streaming Inference

```python
from aicr import AICRClient
from aicr.inference import StreamingRequest

client = AICRClient()

# Stream inference results in real-time
streaming_request = StreamingRequest(
    model_id="mdl_llm_789",
    input_data={"prompt": "Write a story about AI"},
    parameters={"max_tokens": 500, "stream": True}
)

for chunk in client.inference.stream_predict(streaming_request):
    if chunk.is_final:
        print(f"\nFinal result: {chunk.predictions}")
        print(f"Total tokens: {chunk.metadata['total_tokens']}")
    else:
        print(chunk.predictions.get("text", ""), end="", flush=True)
```

## Pipeline Management

### Pipeline Data Types

```python
from aicr.pipelines import (
    Pipeline,
    PipelineCreate,
    PipelineExecution,
    PipelineStage,
    PipelineStatus,
    ExecutionStatus
)
from datetime import datetime
from typing import List, Dict, Any

# Pipeline creation data
pipeline_data = PipelineCreate(
    name="sentiment_training_pipeline",
    description="End-to-end sentiment analysis training pipeline",
    pipeline_type="training",
    stages=[
        PipelineStage(
            name="data_loading",
            description="Load and validate training data",
            configuration={"dataset_version": "v2.1"},
            dependencies=[]
        ),
        PipelineStage(
            name="preprocessing",
            description="Data preprocessing and feature engineering",
            configuration={"max_length": 512, "vocab_size": 30000},
            dependencies=["data_loading"]
        ),
        PipelineStage(
            name="training",
            description="Model training with hyperparameter optimization",
            configuration={"epochs": 50, "batch_size": 32},
            dependencies=["preprocessing"]
        )
    ],
    schedule="0 2 * * 0",  # Weekly at 2 AM
    configuration={
        "notification_emails": ["ml-team@example.com"],
        "max_execution_time_hours": 6
    }
)
```

### List Pipelines

```python
from aicr import AICRClient
from aicr.pipelines import PipelineFilter

client = AICRClient()

# List all pipelines
pipelines = client.pipelines.list()

# List with filtering
active_pipelines = client.pipelines.list(
    filter=PipelineFilter(
        pipeline_type="training",
        status=PipelineStatus.ACTIVE
    )
)

for pipeline in active_pipelines:
    print(f"{pipeline.name}: {pipeline.status}")
    if pipeline.last_execution:
        print(f"  Last run: {pipeline.last_execution.completed_at}")
        print(f"  Status: {pipeline.last_execution.status}")
```

### Execute Pipeline

```python
from aicr import AICRClient
from aicr.pipelines import ExecutionParameters

client = AICRClient()

# Execute pipeline with custom parameters
execution_params = ExecutionParameters(
    dataset_version="v2.2",
    epochs=75,
    learning_rate=0.0005,
    batch_size=64,
    use_early_stopping=True
)

execution = client.pipelines.execute(
    pipeline_id="pip_training_123",
    parameters=execution_params,
    priority="high"
)

print(f"Execution started: {execution.execution_id}")
print(f"Estimated completion: {execution.estimated_completion}")

# Monitor execution progress
for update in client.pipelines.stream_execution_progress(execution.execution_id):
    if update.stage_completed:
        print(f"✓ {update.stage_name} completed in {update.duration_minutes}min")
    elif update.stage_started:
        print(f"▶ {update.stage_name} started")
    elif update.progress_update:
        print(f"  {update.stage_name}: {update.progress_percent}%")

# Get final results
final_execution = client.pipelines.get_execution(execution.execution_id)
if final_execution.is_successful:
    print(f"Pipeline completed successfully!")
    print(f"Model ID: {final_execution.results['model_id']}")
    print(f"Accuracy: {final_execution.results['accuracy']}")
else:
    print(f"Pipeline failed: {final_execution.error_message}")
```

### Pipeline Templates

```python
from aicr import AICRClient

client = AICRClient()

# List available pipeline templates
templates = client.pipelines.list_templates()

for template in templates:
    print(f"{template.name}: {template.description}")
    print(f"  Stages: {', '.join(template.stages)}")

# Create pipeline from template
new_pipeline = client.pipelines.create_from_template(
    template_id="tmpl_nlp_training",
    name="my_custom_nlp_pipeline",
    parameters={
        "model_architecture": "bert-base",
        "dataset_type": "classification",
        "evaluation_metrics": ["accuracy", "f1_score"]
    }
)

print(f"Pipeline created: {new_pipeline.pipeline_id}")
```

## Monitoring & Analytics

### System Health

```python
from aicr import AICRClient

client = AICRClient()

# Get overall system health
health = client.monitoring.get_health()

print(f"System status: {health.status}")
print(f"Overall health score: {health.health_score}")

# Check component health
for component, status in health.components.items():
    print(f"{component}: {status.status}")
    if status.status != "healthy":
        print(f"  Issues: {status.issues}")

# Get resource usage
resource_usage = client.monitoring.get_resource_usage()
print(f"CPU: {resource_usage.cpu_percent}%")
print(f"Memory: {resource_usage.memory_percent}%")
print(f"GPU: {resource_usage.gpu_percent}%")
```

### Metrics and Analytics

```python
from aicr import AICRClient
from aicr.monitoring import MetricQuery, TimeRange
from datetime import datetime, timedelta

client = AICRClient()

# Query specific metrics
metric_query = MetricQuery(
    metric_name="inference_latency",
    component="inference_engine",
    time_range=TimeRange.LAST_24_HOURS,
    aggregation="avg",
    labels={"model_type": "classification"}
)

metrics = client.monitoring.get_metrics(metric_query)

print(f"Average latency (24h): {metrics.statistics.avg}ms")
print(f"P95 latency: {metrics.statistics.p95}ms")

# Plot metrics (requires matplotlib)
import matplotlib.pyplot as plt

timestamps = [point.timestamp for point in metrics.data_points]
values = [point.value for point in metrics.data_points]

plt.figure(figsize=(12, 6))
plt.plot(timestamps, values)
plt.title("Inference Latency (24h)")
plt.xlabel("Time")
plt.ylabel("Latency (ms)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Performance Analytics

```python
from aicr import AICRClient
from aicr.analytics import PerformanceAnalysis

client = AICRClient()

# Get performance summary
performance = client.analytics.get_performance_summary(
    time_range=TimeRange.LAST_7_DAYS
)

print(f"Total requests: {performance.total_requests:,}")
print(f"Success rate: {performance.success_rate:.2%}")
print(f"Average latency: {performance.avg_latency_ms:.1f}ms")
print(f"Throughput: {performance.throughput_rps:.1f} req/s")

# Model performance comparison
model_comparison = client.analytics.compare_model_performance(
    model_ids=["mdl_v1", "mdl_v2", "mdl_v3"],
    metrics=["accuracy", "latency", "throughput"],
    time_range=TimeRange.LAST_30_DAYS
)

for model_id, metrics in model_comparison.items():
    print(f"\n{model_id}:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Latency: {metrics['latency']:.1f}ms")
    print(f"  Throughput: {metrics['throughput']:.1f} req/s")
```

## Async Operations

### Async Client

```python
import asyncio
from aicr import AsyncAICRClient

async def main():
    async with AsyncAICRClient() as client:
        # List models asynchronously
        models = await client.models.list()
        print(f"Found {len(models)} models")

        # Run inference asynchronously
        response = await client.inference.predict(
            model_id="mdl_sentiment_123",
            input_data={"text": "Async inference is great!"}
        )

        print(f"Prediction: {response.predictions}")

# Run async operations
asyncio.run(main())
```

### Concurrent Operations

```python
import asyncio
from aicr import AsyncAICRClient

async def run_multiple_inferences():
    async with AsyncAICRClient() as client:

        # Prepare multiple inference tasks
        tasks = []
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Absolutely amazing!",
            "Complete waste of money."
        ]

        for i, text in enumerate(texts):
            task = client.inference.predict(
                model_id="mdl_sentiment_123",
                input_data={"text": text}
            )
            tasks.append(task)

        # Run all inferences concurrently
        responses = await asyncio.gather(*tasks)

        # Process results
        for i, response in enumerate(responses):
            sentiment = response.predictions["sentiment"]
            confidence = response.predictions["confidence"]
            print(f"Text {i+1}: {sentiment} ({confidence:.2f})")

asyncio.run(run_multiple_inferences())
```

### Async Streaming

```python
import asyncio
from aicr import AsyncAICRClient

async def stream_inference():
    async with AsyncAICRClient() as client:

        async for chunk in client.inference.stream_predict(
            model_id="mdl_llm_789",
            input_data={"prompt": "Tell me about AI"},
            parameters={"max_tokens": 200, "stream": True}
        ):
            if chunk.is_final:
                print(f"\n\nFinal result received.")
                print(f"Total tokens: {chunk.metadata['total_tokens']}")
            else:
                print(chunk.predictions.get("text", ""), end="", flush=True)

asyncio.run(stream_inference())
```

## Error Handling

### Exception Types

```python
from aicr.exceptions import (
    AICRError,                    # Base exception
    AuthenticationError,          # Authentication failed
    AuthorizationError,           # Insufficient permissions
    ModelNotFoundError,           # Model doesn't exist
    InferenceError,              # Inference operation failed
    ValidationError,             # Input validation failed
    RateLimitError,              # Rate limit exceeded
    ServiceUnavailableError,     # Service temporarily unavailable
    TimeoutError,                # Operation timed out
    NetworkError                 # Network connectivity issue
)
```

### Basic Error Handling

```python
from aicr import AICRClient
from aicr.exceptions import (
    ModelNotFoundError,
    InferenceError,
    RateLimitError,
    ValidationError
)

client = AICRClient()

try:
    response = client.inference.predict(
        model_id="mdl_invalid",
        input_data={"text": "Test input"}
    )
    print(f"Prediction: {response.predictions}")

except ModelNotFoundError as e:
    print(f"Model not found: {e.model_id}")
    print("Available models:")
    for model in client.models.list():
        print(f"  - {model.name} ({model.model_id})")

except ValidationError as e:
    print(f"Validation error: {e.message}")
    for error in e.validation_errors:
        print(f"  - {error.field}: {error.message}")

except InferenceError as e:
    print(f"Inference failed: {e.message}")
    if e.retry_possible:
        print("Retrying is possible")

except RateLimitError as e:
    print(f"Rate limit exceeded: {e.message}")
    print(f"Retry after: {e.retry_after} seconds")
    print(f"Current limit: {e.limit} requests per {e.window}")

except AICRError as e:
    print(f"AICR error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Request ID: {e.request_id}")
```

### Advanced Error Handling

```python
from aicr import AICRClient
from aicr.exceptions import AICRError
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_aicr_error(error: AICRError) -> bool:
    """
    Handle AICR errors with appropriate retry logic.

    Returns:
        bool: True if operation should be retried, False otherwise
    """
    logger.error(f"AICR Error: {error.error_code} - {error.message}")

    # Log error details
    if error.request_id:
        logger.error(f"Request ID: {error.request_id}")

    if error.details:
        logger.error(f"Details: {error.details}")

    # Determine if retry is appropriate
    if isinstance(error, RateLimitError):
        logger.info(f"Rate limited. Waiting {error.retry_after} seconds...")
        time.sleep(error.retry_after)
        return True

    elif isinstance(error, ServiceUnavailableError):
        logger.info("Service unavailable. Retrying in 30 seconds...")
        time.sleep(30)
        return True

    elif isinstance(error, TimeoutError):
        logger.info("Request timed out. Retrying with longer timeout...")
        return True

    elif isinstance(error, NetworkError):
        logger.info("Network error. Retrying in 10 seconds...")
        time.sleep(10)
        return True

    else:
        # Don't retry for other error types
        return False

def robust_inference(client: AICRClient, model_id: str, input_data: dict, max_retries: int = 3):
    """Run inference with robust error handling and retry logic."""

    for attempt in range(max_retries + 1):
        try:
            response = client.inference.predict(
                model_id=model_id,
                input_data=input_data,
                timeout_seconds=30 + (attempt * 10)  # Increase timeout on retries
            )
            return response

        except AICRError as e:
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded")
                raise

            should_retry = handle_aicr_error(e)
            if not should_retry:
                logger.error("Error is not recoverable")
                raise

            logger.info(f"Retrying attempt {attempt + 2}/{max_retries + 1}")

# Usage
client = AICRClient()

try:
    response = robust_inference(
        client=client,
        model_id="mdl_sentiment_123",
        input_data={"text": "Test input"},
        max_retries=3
    )
    print(f"Success: {response.predictions}")

except AICRError as e:
    logger.error(f"Final error: {e}")
```

## Advanced Features

### Custom HTTP Client

```python
from aicr import AICRClient
import httpx

# Create custom HTTP client with specific configuration
http_client = httpx.Client(
    timeout=60.0,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
    headers={"User-Agent": "MyApp/2.0.0"},
    proxies="http://proxy.company.com:8080"
)

client = AICRClient(
    api_key="your_api_key",
    http_client=http_client
)
```

### Request Middleware

```python
from aicr import AICRClient
from aicr.middleware import RequestMiddleware
import time

class LoggingMiddleware(RequestMiddleware):
    """Log all requests and responses."""

    def before_request(self, request):
        print(f"Sending request: {request.method} {request.url}")
        request.start_time = time.time()
        return request

    def after_response(self, request, response):
        duration = time.time() - request.start_time
        print(f"Response: {response.status_code} in {duration:.2f}s")
        return response

class RetryMiddleware(RequestMiddleware):
    """Add custom retry logic."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def after_response(self, request, response):
        if response.status_code >= 500 and request.retry_count < self.max_retries:
            time.sleep(2 ** request.retry_count)  # Exponential backoff
            request.retry_count += 1
            return self.retry_request(request)
        return response

# Add middleware to client
client = AICRClient()
client.add_middleware(LoggingMiddleware())
client.add_middleware(RetryMiddleware(max_retries=5))
```

### Response Caching

```python
from aicr import AICRClient
from aicr.cache import MemoryCache, RedisCache
import redis

# In-memory caching
memory_cache = MemoryCache(max_size=1000, ttl=300)  # 5 minutes
client = AICRClient(cache=memory_cache)

# Redis caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_cache = RedisCache(redis_client, ttl=600)  # 10 minutes
client = AICRClient(cache=redis_cache)

# Cache-aware operations
response1 = client.models.get("mdl_abc123")  # Cache miss - API call
response2 = client.models.get("mdl_abc123")  # Cache hit - no API call

# Disable caching for specific requests
response = client.models.get("mdl_abc123", use_cache=False)
```

### Batch Operations

```python
from aicr import AICRClient
from aicr.batch import BatchProcessor

client = AICRClient()

# Batch model operations
batch_processor = BatchProcessor(client, batch_size=10, delay=0.1)

# Add multiple operations to batch
model_ids = ["mdl_001", "mdl_002", "mdl_003", "mdl_004", "mdl_005"]

batch_processor.add_operations([
    ("get_model", {"model_id": model_id}) for model_id in model_ids
])

# Execute all operations in batches
results = batch_processor.execute()

for result in results:
    if result.is_successful:
        model = result.data
        print(f"Model: {model.name}")
    else:
        print(f"Error: {result.error}")
```

### Custom Serialization

```python
from aicr import AICRClient
from aicr.serialization import CustomSerializer
import numpy as np

class NumpySerializer(CustomSerializer):
    """Custom serializer for NumPy arrays."""

    def serialize(self, data):
        if isinstance(data, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": data.tolist(),
                "dtype": str(data.dtype),
                "shape": data.shape
            }
        return data

    def deserialize(self, data):
        if isinstance(data, dict) and data.get("__numpy_array__"):
            return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])
        return data

# Use custom serializer
client = AICRClient(serializer=NumpySerializer())

# NumPy arrays will be automatically serialized/deserialized
image_array = np.random.rand(224, 224, 3)
response = client.inference.predict(
    model_id="mdl_image_classifier",
    input_data={"image": image_array}
)
```

---

**API Reference:**
- [Model Management API](../api/rest_api.md#model-management)
- [Inference API](../api/rest_api.md#inference-operations)
- [WebSocket API](websocket_api.md)

**Examples:**
- [Basic Usage Examples](../examples/basic_usage.py)
- [Advanced Examples](../examples/advanced_features.py)
- [Integration Examples](../examples/integration_examples/)

**Next Steps:**
- [User Guides](../guides/)
- [Code Examples](../examples/)
- [Troubleshooting](../troubleshooting.md)