# APG Import/Export (IMEX) - Developer Guide

**Version**: 1.0.0
**Date**: 2025-08-13
**Audience**: Developers, System Integrators, Technical Teams

## Architecture Overview

The APG Import/Export (IMEX) capability is built using modern async Python architecture with seamless APG platform integration. It follows APG's composition-first design principles and integrates with the broader APG ecosystem.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APG Platform Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Auth/RBAC  │  Audit  │  AI Orchestration  │  Notifications │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  IMEX Capability Layer                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Flask Views   │   REST API      │    WebSocket Events     │
│                 │                 │                         │
│ • Job Mgmt      │ • CRUD Ops      │ • Real-time Updates     │
│ • Workflows     │ • Execution     │ • Progress Monitoring   │
│ • Monitoring    │ • Metrics       │ • Notifications         │
│ • Analytics     │ • Health Check  │ • Collaboration         │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Import Service  │ Export Service  │  Workflow Service       │
│                 │                 │                         │
│ • Schema Detect │ • Format Conv   │ • Orchestration         │
│ • Data Valid    │ • Parallel Exp  │ • Dependencies          │
│ • Transform     │ • Compression   │ • Error Handling        │
│ • Quality Check │ • Incremental   │ • Performance           │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
├─────────────────┬─────────────────┬─────────────────────────┤
│  PostgreSQL     │  Redis Cache    │   External Systems      │
│                 │                 │                         │
│ • Job Metadata  │ • Performance   │ • Source Systems        │
│ • Execution Log │ • Sessions      │ • Target Systems        │
│ • Audit Trails  │ • Temp Data     │ • Cloud Storage         │
│ • Metrics       │ • Real-time St  │ • APIs                  │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Core Components

1. **Models Layer** (`models.py`): Pydantic v2 data models with APG standards
2. **Service Layer** (`service.py`): Async business logic with APG integration
3. **Views Layer** (`views.py`): Flask-AppBuilder views and Pydantic models
4. **API Layer** (`api.py`): REST API endpoints with OpenAPI documentation
5. **Blueprint Layer** (`blueprint.py`): APG composition engine integration

## Development Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6.0+
- APG Platform (development environment)

### Installation

```bash
# Clone APG platform repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform

# Navigate to IMEX capability
cd capabilities/common/imex

# Install dependencies
uv install

# Setup development database
createdb apg_imex_dev
psql apg_imex_dev < schema.sql

# Setup Redis
redis-server --port 6379

# Run tests
uv run pytest -vxs tests/

# Type checking
uv run pyright
```

### Environment Configuration

```bash
# .env file
APG_DATABASE_URL=postgresql://user:password@localhost/apg_imex_dev
APG_REDIS_URL=redis://localhost:6379/0
APG_COMPOSITION_ENABLED=true
APG_AI_ENABLED=true
APG_LOG_LEVEL=DEBUG
```

## Data Models

### Core Model Structure

All models follow APG standards:

- **Async-first**: Designed for async operations
- **Tabs for indentation**: Following CLAUDE.md standards
- **Modern typing**: Using `str | None`, `list[str]`, etc.
- **UUID7 IDs**: Using `uuid7str` for all entity IDs
- **Multi-tenancy**: All models include `tenant_id`
- **Audit trails**: Automatic created/updated timestamps

### Key Models

```python
# Primary job entity
class ImportExportJob(BaseModel):
    id: str = Field(default_factory=uuid7str)
    tenant_id: str
    name: str
    job_type: JobType
    source_config: SourceConfig
    target_config: TargetConfig
    status: JobStatus = JobStatus.DRAFT
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

# Configuration models
class SourceConfig(BaseModel):
    source_type: SourceType
    format: DataFormat
    chunk_size: int = 10000
    # ... additional fields

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

# Execution tracking
class JobExecution(BaseModel):
    id: str = Field(default_factory=uuid7str)
    job_id: str
    status: JobStatus
    metrics: ProcessingMetrics
    # ... additional fields
```

### Model Validation

Models include comprehensive validation using Pydantic v2:

```python
from pydantic import BaseModel, Field, AfterValidator
from typing import Annotated

def validate_positive_int(value: int) -> int:
    if value <= 0:
        raise ValueError("Value must be positive")
    return value

class SourceConfig(BaseModel):
    chunk_size: Annotated[int, AfterValidator(validate_positive_int)] = 10000
```

## Service Layer

### ImportExportService

The core service class manages all business logic:

```python
class ImportExportService:
    """Core import/export service with APG platform integration"""

    def __init__(self):
        self.health_status = "healthy"
        self.active_jobs = {}
        self.performance_metrics = {}
        self.ai_client = None
        self.etlp_client = None
        # ... other clients

    async def initialize(self) -> bool:
        """Initialize service with APG capability dependencies"""
        await self._initialize_apg_clients()
        await self._setup_monitoring()
        await self._initialize_ai_capabilities()
        self.health_status = "ready"
        return True

    async def create_job(self, job_config: dict[str, Any], created_by: str) -> ImportExportJob:
        """Create new import/export job with validation"""
        job = ImportExportJob(**job_config, created_by=created_by)
        await self._validate_job_configuration(job)
        await self._optimize_job_configuration(job)
        self.active_jobs[job.id] = job
        return job
```

### Async Patterns

All service methods are async and follow APG standards:

```python
async def execute_job(self, job_id: str, execution_config: dict[str, Any] | None = None) -> JobExecution:
    """Execute import/export job with real-time monitoring"""
    job = self.active_jobs.get(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    execution = JobExecution(job_id=job_id, execution_number=len(job.execution_history) + 1)

    try:
        # Update job status
        job.status = JobStatus.RUNNING
        execution.status = JobStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)

        # Execute based on job type
        if job.job_type == JobType.IMPORT:
            await self._execute_import_job(job, execution)
        elif job.job_type == JobType.EXPORT:
            await self._execute_export_job(job, execution)
        # ... other job types

        execution.status = JobStatus.COMPLETED
        job.status = JobStatus.COMPLETED
        return execution

    except Exception as e:
        execution.status = JobStatus.FAILED
        job.status = JobStatus.FAILED
        raise RuntimeError(f"Job execution failed: {e}")
```

### Error Handling

Comprehensive error handling with logging:

```python
def _log_service_error(self, message: str):
    """Log service error message"""
    logger.error(f"[IMEX Service] {message}")

async def execute_job(self, job_id: str) -> JobExecution:
    try:
        # Job execution logic
        pass
    except ValidationError as e:
        self._log_validation_error(f"Job validation failed: {e}")
        raise ValueError(f"Invalid job configuration: {e}")
    except Exception as e:
        self._log_service_error(f"Failed to execute job: {e}")
        raise RuntimeError(f"Job execution failed: {e}")
```

## API Layer

### REST API Design

The API follows RESTful principles with async endpoints:

```python
from flask_restx import Api, Resource, fields
from flask import request, jsonify

# API namespace
jobs_ns = Namespace('jobs', description='Import/Export job operations')

@jobs_ns.route('/')
class JobListAPI(Resource):
    @jobs_ns.doc('list_jobs')
    def get(self):
        """List import/export jobs with filtering"""
        try:
            tenant_id = request.args.get('tenant_id')
            status = request.args.get('status')
            # ... filtering logic

            jobs = []
            for job_id, job in imex_service.active_jobs.items():
                if tenant_id and job.tenant_id != tenant_id:
                    continue
                jobs.append(job.dict())

            return {'jobs': jobs}, 200

        except Exception as e:
            return {'error': str(e)}, 500

    @jobs_ns.expect(job_create_model)
    def post(self):
        """Create new import/export job"""
        try:
            job_request = JobCreateRequest(**request.json)
            job = await imex_service.create_job(job_request.dict(), "current_user")
            return job.dict(), 201
        except Exception as e:
            return {'error': str(e)}, 400
```

### API Documentation

Full OpenAPI/Swagger documentation is automatically generated:

```python
# Request/Response models for documentation
job_create_model = api.model('JobCreateRequest', {
    'name': fields.String(required=True, description='Job name'),
    'job_type': fields.String(required=True, enum=['import', 'export', 'migration']),
    'source_config': fields.Raw(required=True, description='Source configuration'),
    'target_config': fields.Raw(required=True, description='Target configuration')
})

job_response_model = api.model('JobResponse', {
    'id': fields.String(description='Job ID'),
    'name': fields.String(description='Job name'),
    'status': fields.String(description='Job status'),
    'created_at': fields.DateTime(description='Creation timestamp')
})
```

### Authentication and Authorization

Integration with APG auth/RBAC:

```python
def _get_current_user_id() -> str:
    """Get current user ID from request context"""
    return request.headers.get('X-User-ID', 'anonymous')

def _validate_tenant_access(tenant_id: str) -> bool:
    """Validate tenant access permissions"""
    # Integration with APG auth/RBAC capability
    return True

@jobs_ns.route('/<string:job_id>')
class JobAPI(Resource):
    def get(self, job_id):
        """Get job details"""
        user_id = _get_current_user_id()
        job = imex_service.active_jobs.get(job_id)

        if not job:
            return {'error': 'Job not found'}, 404

        # Check permissions
        if not _validate_tenant_access(job.tenant_id):
            return {'error': 'Access denied'}, 403

        return job.dict(), 200
```

### WebSocket Integration

Real-time monitoring with WebSocket:

```python
from flask_socketio import emit, join_room, leave_room

def setup_websocket_events(socketio):
    @socketio.on('join_job_monitor')
    def on_join_job_monitor(data):
        job_id = data.get('job_id')
        if job_id:
            join_room(f'job_{job_id}')
            emit('joined', {'job_id': job_id})

    @socketio.on('get_job_metrics')
    def on_get_job_metrics(data):
        job_id = data.get('job_id')
        if job_id:
            metrics = await imex_service.get_job_metrics(job_id)
            emit('job_metrics', {
                'job_id': job_id,
                'metrics': metrics.dict(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, room=f'job_{job_id}')
```

## UI Layer

### Flask-AppBuilder Views

Views integrate with APG's Flask-AppBuilder infrastructure:

```python
from flask_appbuilder import ModelView, BaseView, expose, has_access

class ImportExportJobView(ModelView):
    datamodel = SQLAInterface(ImportExportJob)

    list_columns = ['name', 'job_type', 'status', 'created_at', 'created_by']
    search_columns = ['name', 'description', 'job_type', 'status']

    @action("execute_job", "Execute Job", "Execute selected jobs", "fa-play")
    def execute_job_action(self, items):
        for item in items:
            try:
                imex_service.execute_job(item.id)
                flash(f"Job '{item.name}' execution started", "success")
            except Exception as e:
                flash(f"Failed to execute job '{item.name}': {str(e)}", "error")
        return redirect(self.get_redirect())

    @expose('/monitor/<job_id>')
    @has_access
    def monitor_job(self, job_id):
        job = self.datamodel.get(job_id)
        metrics = imex_service.get_job_metrics(job_id)
        return self.render_template('imex/job_monitor.html', job=job, metrics=metrics)
```

### Custom Widgets

Custom widgets for enhanced UI experience:

```python
class JobMonitoringWidget(ShowWidget):
    """Custom widget for job monitoring view"""
    template = 'imex/widgets/job_monitoring.html'

class WorkflowDesignerWidget(EditWidget):
    """Custom widget for workflow designer"""
    template = 'imex/widgets/workflow_designer.html'
```

### Template System

Templates follow APG UI patterns:

```html
<!-- imex/job_monitor.html -->
{% extends "appbuilder/general/widgets/base_list.html" %}

{% block content %}
<div class="container-fluid">
    <div class="row">
        <div class="col-md-8">
            <div class="panel panel-primary">
                <div class="panel-heading">
                    <h3>Job: {{ job.name }}</h3>
                </div>
                <div class="panel-body">
                    <!-- Job details -->
                    <p><strong>Status:</strong> {{ job.status.value }}</p>
                    <p><strong>Type:</strong> {{ job.job_type.value }}</p>

                    <!-- Real-time metrics -->
                    <div id="metrics-container">
                        <div class="progress">
                            <div class="progress-bar" role="progressbar"
                                 style="width: {{ (metrics.records_successful / metrics.records_processed * 100) if metrics.records_processed > 0 else 0 }}%">
                            </div>
                        </div>
                        <p>Records: {{ metrics.records_processed }} processed, {{ metrics.records_successful }} successful</p>
                        <p>Throughput: {{ metrics.throughput_records_per_second }} records/second</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://{{ request.host }}/ws/v1/imex');
ws.onopen = function() {
    ws.send(JSON.stringify({event: 'join_job_monitor', job_id: '{{ job.id }}'}));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.event === 'job_metrics') {
        updateMetrics(data.metrics);
    }
};

function updateMetrics(metrics) {
    // Update UI with new metrics
    document.getElementById('metrics-container').innerHTML = `
        <div class="progress">
            <div class="progress-bar" style="width: ${metrics.records_successful / metrics.records_processed * 100}%"></div>
        </div>
        <p>Records: ${metrics.records_processed} processed, ${metrics.records_successful} successful</p>
        <p>Throughput: ${metrics.throughput_records_per_second} records/second</p>
    `;
}
</script>
{% endblock %}
```

## APG Integration

### Composition Engine Registration

Register with APG's composition engine:

```python
from .__init__ import imex_capability, capability_metadata

class ImportExportBlueprint:
    async def _register_apg_composition(self):
        """Register with APG composition engine"""
        await imex_capability.initialize()
        await self._register_composition_patterns()
        await self._register_marketplace()

    async def _register_composition_patterns(self):
        """Register composition patterns with APG orchestration"""
        patterns = capability_metadata["composition_patterns"]
        for pattern in patterns:
            # await apg.composition.register_pattern(pattern, imex_capability)
            pass
```

### Dependency Integration

Integration with APG capabilities:

```python
# Service layer integration
async def _initialize_apg_clients(self):
    """Initialize connections to APG capabilities"""
    # self.ai_client = await apg.get_capability("ai_orchestration")
    # self.etlp_client = await apg.get_capability("etlp")
    # self.conn_client = await apg.get_capability("conn")
    # self.audit_client = await apg.get_capability("audit_compliance")
    # self.notification_client = await apg.get_capability("notification_engine")

    # Mock clients for development
    self.ai_client = MockAIClient()
    self.etlp_client = MockETLPClient()
    # ... other mock clients
```

### Health Check Integration

Comprehensive health checks for APG monitoring:

```python
async def health_check(self) -> dict[str, Any]:
    """Comprehensive service health check"""
    components_health = {
        "ai_client": "healthy" if self.ai_client else "unavailable",
        "etlp_client": "healthy" if self.etlp_client else "unavailable",
        "conn_client": "healthy" if self.conn_client else "unavailable",
        "audit_client": "healthy" if self.audit_client else "unavailable"
    }

    overall_health = "healthy" if all(
        status == "healthy" for status in components_health.values()
    ) else "degraded"

    return {
        "service": "imex",
        "status": overall_health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components_health,
        "active_jobs": len(self.active_jobs),
        "performance_metrics": await self.get_system_performance_metrics()
    }
```

## Testing

### Test Structure

Tests follow APG testing standards:

```python
# No @pytest.mark.asyncio decorators needed
# Use real objects with pytest fixtures
# Use pytest-httpserver for API testing

class TestImportExportService:
    async def test_service_initialization(self):
        service = ImportExportService()
        await service.initialize()
        assert service.health_status == "ready"

    async def test_create_job_valid(self):
        service = ImportExportService()
        await service.initialize()

        job_config = generate_test_job_config()
        job = await service.create_job(job_config, "test_user")

        assert isinstance(job, ImportExportJob)
        assert job.created_by == "test_user"
```

### Test Fixtures

Comprehensive test fixtures:

```python
@pytest.fixture
def test_config():
    return TEST_CONFIG.copy()

@pytest.fixture
def test_job_config():
    return generate_test_job_config()

@pytest.fixture
def mock_ai_service():
    return MockAIService()

@pytest.fixture
async def initialized_service():
    service = ImportExportService()
    await service.initialize()
    return service
```

### Performance Testing

Performance tests with benchmarks:

```python
async def test_concurrent_job_creation(self):
    service = ImportExportService()
    await service.initialize()

    async def create_job(index):
        job_config = generate_test_job_config()
        job_config["name"] = f"Concurrent Job {index}"
        return await service.create_job(job_config, f"user_{index}")

    # Create 10 jobs concurrently
    tasks = [create_job(i) for i in range(10)]
    jobs = await asyncio.gather(*tasks)

    assert len(jobs) == 10
    assert len(service.active_jobs) == 10
```

## Performance Optimization

### Async Processing

Optimized async patterns for high performance:

```python
async def _stream_data_batches(self, data_source, chunk_size: int) -> AsyncIterator[list[dict[str, Any]]]:
    """Stream data in batches for memory efficiency"""
    async for batch in data_source.stream_batches(chunk_size):
        yield batch

async def _execute_import_job(self, job: ImportExportJob, execution: JobExecution):
    """Execute import with parallel processing"""
    async for batch in self._stream_data_batches(data_source, job.source_config.chunk_size):
        # Process batch in parallel
        tasks = [
            self._process_record(record, job.validation_rules)
            for record in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update metrics
        execution.metrics.records_processed += len(batch)
        execution.metrics.records_successful += sum(1 for r in results if not isinstance(r, Exception))
```

### Caching Strategies

Redis integration for performance:

```python
import aioredis

class ImportExportService:
    async def _initialize_cache(self):
        self.redis = aioredis.from_url("redis://localhost:6379")

    async def _cache_schema(self, source_config: SourceConfig, schema: dict):
        """Cache detected schema for reuse"""
        cache_key = f"schema:{hash(str(source_config.dict()))}"
        await self.redis.setex(cache_key, 3600, json.dumps(schema))

    async def _get_cached_schema(self, source_config: SourceConfig) -> dict | None:
        """Get cached schema if available"""
        cache_key = f"schema:{hash(str(source_config.dict()))}"
        cached = await self.redis.get(cache_key)
        return json.loads(cached) if cached else None
```

### Database Optimization

Efficient database operations:

```python
# Bulk operations for better performance
async def bulk_create_executions(self, executions: list[JobExecution]):
    """Bulk create execution records"""
    # Use SQLAlchemy bulk operations
    pass

# Proper indexing for common queries
# Index on (tenant_id, status) for job filtering
# Index on (job_id, execution_number) for execution lookup
# Index on (created_at) for time-based queries
```

## Security

### Data Protection

Encryption and security measures:

```python
from cryptography.fernet import Fernet

class SecureConfigManager:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)

    def encrypt_sensitive_config(self, config: dict) -> dict:
        """Encrypt sensitive configuration fields"""
        sensitive_fields = ['password', 'api_key', 'secret']

        for field in sensitive_fields:
            if field in config:
                config[field] = self.cipher.encrypt(config[field].encode()).decode()

        return config

    def decrypt_sensitive_config(self, config: dict) -> dict:
        """Decrypt sensitive configuration fields"""
        sensitive_fields = ['password', 'api_key', 'secret']

        for field in sensitive_fields:
            if field in config:
                config[field] = self.cipher.decrypt(config[field].encode()).decode()

        return config
```

### Access Control

Integration with APG auth/RBAC:

```python
class PermissionManager:
    def __init__(self, auth_client):
        self.auth_client = auth_client

    async def check_job_permission(self, user_id: str, job_id: str, action: str) -> bool:
        """Check if user has permission for job action"""
        job = await self.get_job(job_id)

        # Check tenant access
        if not await self.auth_client.check_tenant_access(user_id, job.tenant_id):
            return False

        # Check specific permission
        permission = f"imex.{action}"
        return await self.auth_client.check_permission(user_id, permission)
```

### Audit Logging

Complete audit trails:

```python
async def _create_audit_trail(self, job: ImportExportJob):
    """Create audit trail for job creation"""
    audit_entry = {
        "event_type": "job_created",
        "entity_type": "ImportExportJob",
        "entity_id": job.id,
        "user_id": job.created_by,
        "tenant_id": job.tenant_id,
        "metadata": {
            "job_name": job.name,
            "job_type": job.job_type.value,
            "source_type": job.source_config.source_type.value,
            "target_type": job.target_config.target_type.value
        },
        "timestamp": datetime.now(timezone.utc)
    }

    await self.audit_client.log_event(audit_entry)
```

## Deployment

### Docker Configuration

Multi-stage Docker build:

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

ENV PYTHONPATH=/app
ENV APG_COMPOSITION_ENABLED=true

EXPOSE 8080

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
```

### Kubernetes Deployment

Production-ready Kubernetes configuration:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-imex
  namespace: apg-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-imex
  template:
    metadata:
      labels:
        app: apg-imex
    spec:
      containers:
      - name: imex-service
        image: apg/imex:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: APG_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: apg-db-secret
              key: database-url
        - name: APG_REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: apg-config
              key: redis-url
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /api/v1/imex/monitoring/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/imex/monitoring/health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: apg-imex-service
spec:
  selector:
    app: apg-imex
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

### Monitoring and Observability

Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics
job_counter = Counter('imex_jobs_total', 'Total number of jobs', ['job_type', 'status'])
job_duration = Histogram('imex_job_duration_seconds', 'Job execution duration')
active_jobs = Gauge('imex_active_jobs', 'Number of active jobs')

class MetricsCollector:
    def record_job_completion(self, job: ImportExportJob, execution: JobExecution):
        job_counter.labels(job_type=job.job_type.value, status=execution.status.value).inc()
        job_duration.observe(execution.metrics.processing_time_seconds)

    def update_active_jobs(self, count: int):
        active_jobs.set(count)

    def get_metrics(self):
        return generate_latest()
```

## Extension Points

### Custom Transformations

Extend with custom transformation functions:

```python
class CustomTransformationRegistry:
    def __init__(self):
        self.transformations = {}

    def register(self, name: str, func: callable):
        """Register custom transformation function"""
        self.transformations[name] = func

    async def apply_transformation(self, data: Any, transform_name: str, **kwargs) -> Any:
        """Apply registered transformation"""
        if transform_name in self.transformations:
            return await self.transformations[transform_name](data, **kwargs)
        raise ValueError(f"Unknown transformation: {transform_name}")

# Usage
registry = CustomTransformationRegistry()

@registry.register("clean_phone_number")
async def clean_phone_number(phone: str) -> str:
    """Custom phone number cleaning"""
    import re
    return re.sub(r'[^\d]', '', phone)
```

### Custom Validation Rules

Add custom validation rules:

```python
class CustomValidationRegistry:
    def __init__(self):
        self.validators = {}

    def register(self, rule_type: str, validator: callable):
        """Register custom validation rule"""
        self.validators[rule_type] = validator

    async def validate(self, value: Any, rule: ValidationRule) -> bool:
        """Apply custom validation rule"""
        if rule.rule_type in self.validators:
            return await self.validators[rule.rule_type](value, rule.parameters)
        return True

# Usage
validation_registry = CustomValidationRegistry()

@validation_registry.register("business_email")
async def validate_business_email(email: str, params: dict) -> bool:
    """Validate business email domains"""
    allowed_domains = params.get('allowed_domains', [])
    domain = email.split('@')[1] if '@' in email else ''
    return domain in allowed_domains
```

### Plugin System

Extensible plugin architecture:

```python
class IMEXPlugin:
    """Base class for IMEX plugins"""

    def __init__(self, name: str):
        self.name = name

    async def initialize(self, service: ImportExportService):
        """Initialize plugin with service instance"""
        pass

    async def before_job_execution(self, job: ImportExportJob):
        """Hook called before job execution"""
        pass

    async def after_job_execution(self, job: ImportExportJob, execution: JobExecution):
        """Hook called after job execution"""
        pass

class PluginManager:
    def __init__(self):
        self.plugins = []

    def register_plugin(self, plugin: IMEXPlugin):
        """Register a plugin"""
        self.plugins.append(plugin)

    async def call_hook(self, hook_name: str, *args, **kwargs):
        """Call hook on all registered plugins"""
        for plugin in self.plugins:
            if hasattr(plugin, hook_name):
                await getattr(plugin, hook_name)(*args, **kwargs)

# Example plugin
class SlackNotificationPlugin(IMEXPlugin):
    async def after_job_execution(self, job: ImportExportJob, execution: JobExecution):
        if execution.status == JobStatus.COMPLETED:
            await self.send_slack_notification(f"Job {job.name} completed successfully")
```

## Troubleshooting

### Common Development Issues

**Import Errors**:
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/apg-platform:$PYTHONPATH

# Check module structure
python -c "from capabilities.common.imex import imex_service; print('Import successful')"
```

**Database Connection Issues**:
```python
# Test database connection
async def test_db_connection():
    import asyncpg
    try:
        conn = await asyncpg.connect("postgresql://user:password@localhost/apg_imex_dev")
        await conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
```

**Redis Connection Issues**:
```python
# Test Redis connection
async def test_redis_connection():
    import aioredis
    try:
        redis = aioredis.from_url("redis://localhost:6379")
        await redis.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis connection failed: {e}")
```

### Debugging

Enable detailed logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable SQL logging (if using SQLAlchemy)
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### Performance Profiling

Profile async operations:

```python
import cProfile
import asyncio

async def profile_job_execution():
    service = ImportExportService()
    await service.initialize()

    job_config = generate_test_job_config()
    job = await service.create_job(job_config, "test_user")

    profiler = cProfile.Profile()
    profiler.enable()

    execution = await service.execute_job(job.id)

    profiler.disable()
    profiler.print_stats(sort='cumulative')

# Run profiling
asyncio.run(profile_job_execution())
```

---

**Next Steps**:
- Review the [API Reference](api_reference.md) for complete API documentation
- Check the [Installation Guide](installation_guide.md) for deployment instructions
- Explore the [User Guide](user_guide.md) for end-user documentation

**Support**: For development support, contact the APG development team at dev@datacraft.co.ke