# User Guide

## Getting Started

The APG Central Configuration capability provides a powerful, secure way to manage configuration across your entire organization. This guide will help you understand the core concepts and learn how to use the system effectively.

## Core Concepts

### Configuration Keys

Configuration keys use hierarchical naming with dot notation:

```
app.database.host
app.database.port
app.features.experimental.featureA
environment.production.scaling.maxInstances
```

### Tenants

Multi-tenant architecture allows complete isolation between organizations:
- Each tenant has separate configuration namespace
- Tenant-specific encryption keys
- Independent access controls and audit logs

### Schemas

JSON schemas provide validation and documentation:
- Prevent invalid configuration values
- Auto-complete in UI
- Documentation generation

## Basic Operations

### Setting Configuration Values

#### Simple Values

```python
from capabilities.composition.central_configuration.service import CentralConfigurationService

# Initialize service
service = await CentralConfigurationService.create(tenant_id="mycompany")

# Set simple values
await service.set_config("app.name", "My Application")
await service.set_config("app.version", "1.2.3")
await service.set_config("app.debug", True)
await service.set_config("app.max_connections", 100)
```

#### Complex Objects

```python
# Database configuration
database_config = {
    "host": "db.example.com",
    "port": 5432,
    "database": "myapp",
    "ssl": True,
    "pool_size": 20
}
await service.set_config("app.database", database_config)

# Feature flags
features = {
    "new_ui": {"enabled": True, "rollout_percentage": 50},
    "beta_api": {"enabled": False},
    "analytics": {"enabled": True, "providers": ["google", "mixpanel"]}
}
await service.set_config("app.features", features)
```

#### With Validation

```python
# Set configuration with schema validation
await service.set_config(
    "app.database.port",
    5432,
    validate_schema=True,
    metadata={"environment": "production", "approved_by": "admin"}
)
```

### Getting Configuration Values

#### Basic Retrieval

```python
# Get simple values
app_name = await service.get_config("app.name")
debug_mode = await service.get_config("app.debug", False)  # with default

# Get complex objects
database_config = await service.get_config("app.database")
host = database_config["host"]
```

#### Pattern Matching

```python
# Get all database-related configs
db_configs = await service.list_configs("app.database.*")

# Get all feature flags
features = await service.list_configs("app.features.*")
```

#### Batch Operations

```python
# Get multiple configs at once
keys = ["app.name", "app.version", "app.debug"]
configs = await service.batch_get_configs(keys)

# Set multiple configs in transaction
new_configs = {
    "app.timeout": 30,
    "app.retries": 3,
    "app.cache_ttl": 3600
}
results = await service.batch_set_configs(new_configs)
```

## Advanced Features

### Schema Management

#### Defining Schemas

```python
# Define schema for database configuration
database_schema = {
    "type": "object",
    "properties": {
        "host": {
            "type": "string",
            "format": "hostname",
            "description": "Database hostname"
        },
        "port": {
            "type": "integer",
            "minimum": 1,
            "maximum": 65535,
            "description": "Database port"
        },
        "ssl": {
            "type": "boolean",
            "default": True,
            "description": "Enable SSL connection"
        }
    },
    "required": ["host", "port"]
}

# Register schema
await service.register_schema("app.database.*", database_schema)
```

#### Schema Validation

```python
# Validate before setting
is_valid, errors = await service.validate_config(
    "app.database.port", 
    "invalid_port"
)
if not is_valid:
    print(f"Validation errors: {errors}")
```

### Configuration History and Versioning

#### View History

```python
# Get change history for a key
history = await service.get_config_history("app.database.host", limit=10)

for change in history:
    print(f"Version {change['version']}: {change['old_value']} -> {change['new_value']}")
    print(f"Changed by {change['user_id']} at {change['timestamp']}")
```

#### Rollback Changes

```python
# Rollback to previous version
success = await service.rollback_config(
    "app.database.host", 
    version=5, 
    user_id="admin"
)
```

### Search and Discovery

#### Text Search

```python
# Search in configuration keys and values
results = await service.search_configs("database", limit=20)

# Search only in values
results = await service.search_configs(
    "localhost", 
    search_values=True, 
    search_metadata=False
)
```

#### Advanced Queries

```python
# Complex filtering
results = await service.query_configs({
    "key_pattern": "*.database.*",
    "metadata.environment": "production",
    "value_type": "object"
}, sort_by="updated_at", limit=50)
```

## Real-time Updates

### WebSocket Subscriptions

```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8080/ws/config');

ws.onopen = function() {
    // Subscribe to specific patterns
    ws.send(JSON.stringify({
        action: 'subscribe',
        patterns: ['app.database.*', 'app.features.*']
    }));
};

ws.onmessage = function(event) {
    const update = JSON.parse(event.data);
    console.log(`Config updated: ${update.config_key} = ${update.new_value}`);
    
    // Update application configuration
    updateAppConfig(update.config_key, update.new_value);
};
```

### Python Event Handling

```python
async def config_change_handler(event):
    """Handle configuration change events"""
    print(f"Configuration changed: {event.config_key}")
    print(f"Old value: {event.old_value}")
    print(f"New value: {event.new_value}")
    
    # Update application state
    if event.config_key.startswith("app.database"):
        await reconnect_database()
    elif event.config_key.startswith("app.features"):
        await update_feature_flags()

# Register event handler
service.realtime_sync.add_event_handler(
    SyncEventType.CONFIG_CHANGED,
    config_change_handler
)
```

## Security and Access Control

### Authentication

```python
# Using JWT tokens
service = await CentralConfigurationService.create(
    tenant_id="mycompany",
    auth_token="eyJhbGciOiJIUzI1NiIs..."
)

# Using API keys
service = await CentralConfigurationService.create(
    tenant_id="mycompany",
    api_key="apg_sk_live_..."
)
```

### Role-Based Access

```python
# Different operations require different permissions:

# READ permission
value = await service.get_config("app.database.host")

# WRITE permission  
await service.set_config("app.new_feature", True)

# ADMIN permission
await service.register_schema("app.*", schema)
await service.delete_config("deprecated.setting")
```

### Encryption

```python
# Sensitive values are automatically encrypted
await service.set_config("app.database.password", "secret123")
await service.set_config("app.api_keys.stripe", "sk_live_...")

# Values are decrypted when retrieved
password = await service.get_config("app.database.password")
# Returns decrypted value: "secret123"
```

## Configuration Patterns

### Environment-Specific Configuration

```python
# Structure configuration by environment
await service.set_config("environments.development.database.host", "localhost")
await service.set_config("environments.staging.database.host", "staging-db.internal")
await service.set_config("environments.production.database.host", "prod-db.internal")

# Get environment-specific config
current_env = "production"
db_host = await service.get_config(f"environments.{current_env}.database.host")
```

### Feature Flags

```python
# Feature flag configuration
feature_flags = {
    "new_checkout_flow": {
        "enabled": True,
        "rollout_percentage": 25,
        "target_users": ["beta_testers"],
        "start_date": "2025-01-01",
        "end_date": "2025-02-01"
    },
    "enhanced_search": {
        "enabled": False,
        "reason": "Performance testing needed"
    }
}
await service.set_config("features", feature_flags)

# Check feature flags in application
def is_feature_enabled(feature_name: str, user_id: str = None) -> bool:
    features = await service.get_config("features", {})
    feature = features.get(feature_name, {})
    
    if not feature.get("enabled", False):
        return False
    
    # Check rollout percentage
    rollout = feature.get("rollout_percentage", 100)
    if rollout < 100:
        user_hash = hash(user_id or "anonymous") % 100
        if user_hash >= rollout:
            return False
    
    return True
```

### Application Configuration

```python
# Complete application configuration example
app_config = {
    "metadata": {
        "name": "My Application",
        "version": "2.1.0",
        "description": "Enterprise application"
    },
    "database": {
        "primary": {
            "host": "primary-db.internal",
            "port": 5432,
            "database": "myapp",
            "ssl": True,
            "pool_size": 20,
            "timeout": 30
        },
        "readonly": {
            "host": "readonly-db.internal",
            "port": 5432,
            "database": "myapp",
            "ssl": True,
            "pool_size": 10
        }
    },
    "cache": {
        "redis": {
            "host": "redis.internal",
            "port": 6379,
            "db": 0,
            "ttl": 3600
        }
    },
    "external_apis": {
        "payment_processor": {
            "base_url": "https://api.stripe.com",
            "timeout": 10,
            "retries": 3
        },
        "email_service": {
            "provider": "sendgrid",
            "timeout": 5
        }
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "outputs": ["stdout", "file"],
        "file_path": "/var/log/myapp.log"
    }
}

await service.set_config("app", app_config)
```

## Best Practices

### Naming Conventions

```python
# Good naming patterns
"app.database.primary.host"           # Clear hierarchy
"features.checkout.v2.enabled"        # Version in key
"integrations.stripe.webhook_secret"  # Service-specific
"monitoring.alerts.cpu_threshold"     # Functional grouping

# Avoid these patterns
"dbhost"                             # Too abbreviated
"app.config.database.config.host"    # Redundant "config"
"STRIPE_SECRET_KEY"                  # Environment variable style
```

### Configuration Organization

```python
# Organize by functional areas
await service.set_config("database.*", database_configs)
await service.set_config("features.*", feature_flags)
await service.set_config("integrations.*", third_party_configs)
await service.set_config("monitoring.*", monitoring_configs)

# Use metadata for additional context
await service.set_config(
    "app.critical_setting",
    value,
    metadata={
        "owner": "platform-team",
        "environment": "production",
        "last_reviewed": "2025-01-01",
        "documentation": "https://wiki.company.com/critical-setting"
    }
)
```

### Error Handling

```python
from capabilities.composition.central_configuration.service import (
    ConfigurationError, ValidationError, EncryptionError
)

try:
    await service.set_config("app.database.port", "invalid_port")
except ValidationError as e:
    print(f"Validation failed: {e.details}")
except EncryptionError as e:
    print(f"Encryption failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Performance Optimization

```python
# Use batch operations for multiple configs
configs = {
    "app.timeout": 30,
    "app.retries": 3,
    "app.cache_ttl": 3600
}
await service.batch_set_configs(configs)

# Cache frequently accessed configs locally
from functools import lru_cache

@lru_cache(maxsize=100)
async def get_cached_config(key: str):
    return await service.get_config(key)

# Use patterns to get related configs efficiently
db_configs = await service.list_configs("app.database.*")
```

## Integration Examples

### Flask Application

```python
from flask import Flask
from capabilities.composition.central_configuration.service import CentralConfigurationService

app = Flask(__name__)

# Initialize configuration service
config_service = None

@app.before_first_request
async def init_config():
    global config_service
    config_service = await CentralConfigurationService.create(
        tenant_id="mycompany"
    )

@app.route('/config/<key>')
async def get_config_endpoint(key):
    value = await config_service.get_config(key)
    return {"key": key, "value": value}

# Use configuration in routes
@app.route('/database-info')
async def database_info():
    db_config = await config_service.get_config("app.database")
    return {
        "host": db_config["host"],
        "port": db_config["port"],
        "ssl": db_config["ssl"]
    }
```

### FastAPI Application

```python
from fastapi import FastAPI, Depends
from capabilities.composition.central_configuration.service import CentralConfigurationService

app = FastAPI()

async def get_config_service():
    return await CentralConfigurationService.create(tenant_id="mycompany")

@app.get("/config/{key}")
async def get_config(
    key: str, 
    service: CentralConfigurationService = Depends(get_config_service)
):
    value = await service.get_config(key)
    return {"key": key, "value": value}

@app.post("/config/{key}")
async def set_config(
    key: str, 
    value: dict,
    service: CentralConfigurationService = Depends(get_config_service)
):
    success = await service.set_config(key, value)
    return {"success": success}
```

## Next Steps

- Learn about [Enterprise Integrations](../integrations/enterprise-connectors.md)
- Set up [Real-time Synchronization](../advanced/realtime-sync.md)
- Configure [Security and Encryption](../security/encryption.md)
- Explore [Machine Learning Features](../advanced/ml-models.md)