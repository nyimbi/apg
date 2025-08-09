# APG Key Management - APG Platform Integration Guide

## Overview

This guide explains how the Key Management capability integrates with the APG (Application Programming Generation) platform, including composition patterns, event handling, and service orchestration.

## APG Architecture Integration

### Capability Registration

The Key Management capability registers with the APG platform's capability registry:

```python
from capabilities.composition.capability_registry import CapabilityRegistry

registry = CapabilityRegistry()

# Register Key Management capability
await registry.register_capability({
    'name': 'keym',
    'version': '1.0.0',
    'description': 'Enterprise Key Management System',
    'endpoints': ['/keym/api/v1'],
    'dependencies': ['auth_rbac', 'notification'],
    'capabilities': [
        'key_lifecycle_management',
        'multi_cloud_federation',
        'hsm_integration',
        'quantum_safe_cryptography'
    ]
})
```

### Service Composition

Key Management integrates with other APG capabilities:

```python
from capabilities.composition.api_service_mesh import ServiceMesh

# Register with service mesh
mesh = ServiceMesh()
await mesh.register_service('keym', {
    'url': 'http://localhost:8080/keym',
    'health_check': '/keym/health',
    'metrics': '/keym/metrics'
})
```

## APG Blueprint Integration

### Flask Blueprint Registration

```python
from flask import Flask
from keym.blueprint import keym_blueprint
from capabilities.composition.capability_registry import CapabilityRegistry

app = Flask(__name__)
registry = CapabilityRegistry(app)

# Register blueprint with APG patterns
app.register_blueprint(keym_blueprint, url_prefix='/keym')

# Enable APG composition features
registry.enable_composition_features(keym_blueprint)
```

### Route Composition

APG enables intelligent route composition:

```python
from capabilities.composition.api_service_mesh import RouteComposer

composer = RouteComposer()

# Compose multi-capability endpoints
@composer.compose_endpoint('/secure-document')
async def secure_document_endpoint(request):
    # 1. Authenticate with Auth/RBAC
    user = await auth_service.authenticate(request)
    
    # 2. Create encryption key
    key = await keym_service.create_key(encryption_spec, user.id)
    
    # 3. Encrypt document
    encrypted_doc = await keym_service.encrypt_data(
        key.spec.id, 
        request.files['document'].read(), 
        user.id
    )
    
    # 4. Send notification
    await notification_service.send_notification(
        user.id, 
        f"Document encrypted with key {key.spec.id}"
    )
    
    return {'status': 'encrypted', 'key_id': key.spec.id}
```

## Event-Driven Architecture

### APG Event Bus Integration

```python
from capabilities.composition.event_streaming_bus import EventBus

event_bus = EventBus()

# Publish key management events
@service.on_key_created
async def publish_key_created(key_info):
    await event_bus.publish('keym.key.created', {
        'key_id': key_info.spec.id,
        'algorithm': key_info.spec.algorithm.value,
        'tenant_id': key_info.spec.tenant_id,
        'timestamp': datetime.utcnow().isoformat()
    })

# Subscribe to other capability events
@event_bus.subscribe('auth.user.created')
async def setup_user_keys(event_data):
    """Automatically create default keys for new users"""
    user_id = event_data['user_id']
    tenant_id = event_data['tenant_id']
    
    # Create default encryption key for user
    spec = await create_key_spec_async(
        tenant_id=tenant_id,
        algorithm=KeyAlgorithm.AES_256,
        usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
        name=f"Default Key - {user_id}",
        created_by=user_id
    )
    
    await service.create_key(spec, user_id)
```

### Event Patterns

Common event patterns in APG integration:

```python
# Security Events
@event_bus.subscribe('security.threat.detected')
async def handle_security_threat(event_data):
    threat_level = event_data.get('level')
    affected_resources = event_data.get('resources', [])
    
    if threat_level == 'critical':
        # Rotate all affected keys
        for resource in affected_resources:
            if resource['type'] == 'encryption_key':
                await service.rotate_key(resource['id'], 'system@security')

# Compliance Events
@event_bus.subscribe('compliance.audit.required')
async def generate_compliance_report(event_data):
    audit_type = event_data['audit_type']
    time_range = event_data['time_range']
    
    report = await service.generate_compliance_report(
        audit_type=audit_type,
        start_date=time_range['start'],
        end_date=time_range['end']
    )
    
    await event_bus.publish('keym.compliance.report.generated', {
        'report_id': report['id'],
        'audit_type': audit_type,
        'findings': report['findings']
    })
```

## APG Workflow Integration

### Workflow Orchestration

```python
from capabilities.composition.workflow_orchestration import WorkflowEngine

workflow_engine = WorkflowEngine()

# Define key lifecycle workflow
key_lifecycle_workflow = {
    'name': 'enterprise_key_lifecycle',
    'steps': [
        {
            'name': 'create_key',
            'service': 'keym',
            'action': 'create_key',
            'parameters': {'spec': '${input.key_spec}', 'user_id': '${input.user_id}'}
        },
        {
            'name': 'setup_policies',
            'service': 'keym',
            'action': 'create_policy',
            'parameters': {
                'policy_type': 'automatic_rotation',
                'schedule': 'monthly',
                'key_id': '${steps.create_key.output.key_id}'
            }
        },
        {
            'name': 'enable_monitoring',
            'service': 'keym',
            'action': 'enable_monitoring',
            'parameters': {'key_id': '${steps.create_key.output.key_id}'}
        },
        {
            'name': 'send_notification',
            'service': 'notification',
            'action': 'send_notification',
            'parameters': {
                'user_id': '${input.user_id}',
                'message': 'Enterprise key ${steps.create_key.output.key_id} created successfully'
            }
        }
    ]
}

await workflow_engine.register_workflow(key_lifecycle_workflow)
```

### Workflow Execution

```python
# Execute key lifecycle workflow
workflow_result = await workflow_engine.execute_workflow(
    'enterprise_key_lifecycle',
    {
        'key_spec': {
            'tenant_id': 'enterprise-tenant',
            'algorithm': 'AES_256',
            'name': 'Production Encryption Key'
        },
        'user_id': 'admin@enterprise.com'
    }
)
```

## APG Configuration Management

### Central Configuration Integration

```python
from capabilities.composition.central_configuration import ConfigurationManager

config_manager = ConfigurationManager()

# Register Key Management configuration schema
await config_manager.register_schema('keym', {
    'type': 'object',
    'properties': {
        'default_algorithm': {'type': 'string', 'enum': ['AES_128', 'AES_256']},
        'key_rotation_interval': {'type': 'string', 'pattern': r'^\d+[dhmw]$'},
        'hsm_enabled': {'type': 'boolean'},
        'multi_cloud_federation': {'type': 'boolean'},
        'quantum_safe_mode': {'type': 'boolean'}
    }
})

# Get configuration
config = await config_manager.get_configuration('keym')
service.configure(config)

# Listen for configuration changes
@config_manager.on_configuration_changed('keym')
async def handle_config_change(new_config):
    await service.reconfigure(new_config)
```

## APG Authentication Integration

### RBAC Integration

```python
from capabilities.auth_rbac.service import AuthRBACService
from keym.security import require_permission

auth_service = AuthRBACService()

# Secure endpoint with RBAC
@keym_blueprint.route('/api/v1/keys', methods=['POST'])
@require_permission('keym:create_key')
async def create_key_endpoint():
    # User is already authenticated and authorized
    user_id = request.user.id
    key_spec_data = request.json
    
    spec = await create_key_spec_from_dict(key_spec_data, user_id)
    key = await service.create_key(spec, user_id)
    
    return jsonify({'key_id': key.spec.id, 'status': 'created'})

# Custom permission checks
async def check_key_access(key_id: str, user_id: str, action: str) -> bool:
    """Check if user can perform action on specific key"""
    # Check if user owns the key
    key = await service.retrieve_key(key_id, user_id)
    if key.spec.created_by == user_id:
        return True
    
    # Check delegated permissions
    return await auth_service.check_permission(
        user_id, 
        f'keym:{action}:key:{key_id}'
    )
```

## APG Notification Integration

### Automated Notifications

```python
from capabilities.common.notification.service import NotificationService

notification_service = NotificationService()

# Send key-related notifications
async def send_key_notifications(event_type: str, key_info: dict, user_id: str):
    templates = {
        'key_created': 'Key {key_name} has been created successfully',
        'key_rotated': 'Key {key_name} has been rotated for security',
        'key_expires_soon': 'Key {key_name} will expire in {days_remaining} days',
        'security_anomaly': 'Unusual activity detected on key {key_name}'
    }
    
    message = templates[event_type].format(**key_info)
    
    await notification_service.send_notification({
        'user_id': user_id,
        'title': f'Key Management: {event_type.replace("_", " ").title()}',
        'message': message,
        'priority': 'high' if 'security' in event_type else 'normal',
        'channels': ['email', 'dashboard'],
        'metadata': {
            'capability': 'keym',
            'event_type': event_type,
            'key_id': key_info.get('key_id')
        }
    })
```

## APG Analytics Integration

### Key Management Analytics

```python
from capabilities.composition.analytics import AnalyticsEngine

analytics = AnalyticsEngine()

# Track key usage metrics
async def track_key_metrics(operation: str, key_info: dict):
    await analytics.track_event('keym.operation', {
        'operation': operation,
        'algorithm': key_info['algorithm'],
        'key_size': key_info.get('key_size'),
        'tenant_id': key_info['tenant_id'],
        'timestamp': datetime.utcnow().isoformat()
    })

# Generate analytics dashboards
key_analytics_dashboard = {
    'name': 'Key Management Analytics',
    'widgets': [
        {
            'type': 'metric',
            'title': 'Total Keys',
            'query': 'count(keym.key.created)'
        },
        {
            'type': 'chart',
            'title': 'Key Creation Trend',
            'query': 'count(keym.key.created) group by date(timestamp)'
        },
        {
            'type': 'table',
            'title': 'Algorithm Distribution',
            'query': 'count(keym.operation) group by algorithm'
        }
    ]
}

await analytics.create_dashboard(key_analytics_dashboard)
```

## APG Mobile Integration

### Mobile-Friendly APIs

```python
from capabilities.composition.mobile import MobileAPIAdapter

mobile_adapter = MobileAPIAdapter()

# Create mobile-optimized endpoints
@mobile_adapter.mobile_endpoint('/keym/mobile/keys')
async def mobile_keys_list(request):
    """Mobile-optimized key listing"""
    user_id = await mobile_adapter.authenticate_mobile_request(request)
    
    keys = await service.list_user_keys(
        user_id=user_id,
        limit=20,  # Mobile pagination
        include_metadata=False  # Reduced payload
    )
    
    # Format for mobile consumption
    return {
        'keys': [
            {
                'id': key.spec.id,
                'name': key.spec.name,
                'algorithm': key.spec.algorithm.value,
                'status': key.metadata.status,
                'created_date': key.metadata.created_at.isoformat()
            }
            for key in keys
        ]
    }
```

## Deployment with APG

### Docker Compose Integration

```yaml
# APG Key Management in docker-compose.yml
version: '3.8'
services:
  apg-keym:
    build:
      context: ./capabilities/common/keym
      dockerfile: Dockerfile
    environment:
      - APG_TENANT_ID=${APG_TENANT_ID}
      - DATABASE_URL=${KEYM_DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - HSM_CONFIG=${HSM_CONFIG}
    depends_on:
      - apg-registry
      - apg-event-bus
      - postgres-keym
    networks:
      - apg-network
    volumes:
      - keym-data:/app/data

  postgres-keym:
    image: postgres:15
    environment:
      POSTGRES_DB: apg_keym
      POSTGRES_USER: keym_user
      POSTGRES_PASSWORD: ${KEYM_DB_PASSWORD}
    volumes:
      - postgres-keym-data:/var/lib/postgresql/data
    networks:
      - apg-network

volumes:
  keym-data:
  postgres-keym-data:

networks:
  apg-network:
    external: true
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-keym
  namespace: apg-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-keym
  template:
    metadata:
      labels:
        app: apg-keym
        capability: keym
    spec:
      containers:
      - name: keym
        image: datacraft/apg-keym:latest
        env:
        - name: APG_CAPABILITY_NAME
          value: "keym"
        - name: APG_REGISTRY_URL
          value: "http://apg-registry:8080"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: keym-secrets
              key: database-url
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: apg-keym-service
  namespace: apg-system
spec:
  selector:
    app: apg-keym
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

## Monitoring and Observability

### APG Metrics Integration

```python
from capabilities.composition.monitoring import MetricsCollector

metrics = MetricsCollector()

# Define custom metrics
key_operations_counter = metrics.counter(
    'keym_operations_total',
    'Total key management operations',
    ['operation', 'algorithm', 'status']
)

key_rotation_histogram = metrics.histogram(
    'keym_key_rotation_duration_seconds',
    'Key rotation operation duration'
)

# Instrument service methods
async def create_key_with_metrics(spec, user_id):
    with key_rotation_histogram.time():
        try:
            result = await service.create_key(spec, user_id)
            key_operations_counter.labels(
                operation='create',
                algorithm=spec.algorithm.value,
                status='success'
            ).inc()
            return result
        except Exception as e:
            key_operations_counter.labels(
                operation='create',
                algorithm=spec.algorithm.value,
                status='error'
            ).inc()
            raise
```

---

This integration guide demonstrates how Key Management seamlessly integrates with the APG platform, providing enterprise-grade capabilities while maintaining the flexibility and composability that APG offers.

**Contact Information**
- Website: www.datacraft.co.ke
- Email: nyimbi@gmail.com
- Copyright: © 2025 Datacraft