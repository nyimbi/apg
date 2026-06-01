# APG Configuration Management Capability

**System-wide configuration store providing centralized, hierarchical configuration management with environment-specific overrides, validation, and real-time updates.**

## Overview

The Configuration Management capability (`conf`) is the foundational capability required by all other APG capabilities. It provides:

- **Centralized Configuration**: Single source of truth for all system configuration
- **Hierarchical Scoping**: System, tenant, user, and session-level configurations
- **Environment Support**: Development, testing, staging, and production environments
- **Real-time Updates**: Configuration changes without service restarts
- **Validation**: Built-in validation for configuration values
- **Security**: Sensitive configuration handling and protection
- **Multi-tenancy**: Tenant-isolated configuration management
- **Agent Composition**: Codex, Claude Code, OpenCode, and Pi review agents can
  inspect, prepare, and recommend configuration changes under human approval
  guardrails.
- **Lifecycle Streaming**: APG composition metadata publishes configuration
  lifecycle events on Bytewax stream metadata.
- **Durable Review Evidence**: Production changes, drift remediation, privileged
  configuration agents, lifecycle batches, and audit events retain policy
  decisions, matched rules, review reasons, and required actions.

## Features

### ✅ Core Features
- **Hierarchical Configuration Scopes**: System → Tenant → User → Session
- **Multiple Configuration Sources**: Files, environment variables, runtime
- **Environment-Specific Overrides**: Development, testing, production configurations
- **Real-time Configuration Watching**: Automatic updates on configuration changes
- **Configuration Validation**: Built-in validators for common configuration types
- **Sensitive Data Handling**: Secure storage and retrieval of secrets
- **Backup and Restore**: Configuration backup and disaster recovery
- **REST API**: Full REST API for configuration management

### 🔄 Configuration Precedence
1. **Runtime** - Programmatically set values (highest priority)
2. **Database** - Database-stored configurations
3. **Environment Variables** - OS environment variables
4. **Files** - YAML/JSON configuration files
5. **Default** - Built-in default values (lowest priority)

### 🏗️ Architecture Components
- **ConfigurationManager** - Core configuration management engine
- **ConfigurationService** - Business logic and validation layer
- **REST API** - HTTP endpoints for configuration operations
- **Data Models** - Pydantic models for validation and serialization

### Review Evidence

Generated APG applications can build operator queues directly from the
dependency-light `ConfService`:

- `request_change()` stores production changes as `review_required` with
  `policy_decision`, `matched_rules`, `review_reasons`, and `audit_evidence`.
- `request_drift_remediation()` stores drift remediation requests as
  `review_required` when a remediation plan is present.
- `register_conf_agent()` stores privileged deployment and policy reviewer
  agents as `pending_review` when human approval evidence is missing.
- `validate_batch()` persists denied non-Bytewax batch evidence before raising
  `PermissionError`, so generated consoles can show the rejected stream,
  matched rule, and remediation action.
- `list_pending_reviews()` returns reviewable changes, drift remediations,
  agents, and batches without replaying rules.

## Installation & Setup

### Prerequisites
- Python 3.9+
- AsyncIO support
- PyYAML for YAML configuration files
- Pydantic for data validation

### Basic Setup

```python
from apg.capabilities.common.conf import init_config, get_config, set_config

# Initialize configuration system
config_manager = init_config()

# Get configuration values
database_host = await get_config("apg.database.host")
database_port = await get_config("apg.database.port", default=5432)

# Set configuration values
await set_config("app.debug.enabled", True)
await set_config("api.secret.key", "secret-value", is_sensitive=True)
```

### Advanced Setup with Scoping

```python
from apg.capabilities.common.conf import ConfigurationManager, ConfigScope

# Create configuration manager
config_manager = ConfigurationManager()

# Set tenant context
config_manager.set_context(tenant_id="company-123", user_id="user-456")

# Set tenant-specific configuration
await config_manager.set("app.theme", "dark", scope=ConfigScope.TENANT)
await config_manager.set("user.language", "es", scope=ConfigScope.USER)

# Get scoped configuration
theme = await config_manager.get("app.theme", scope=ConfigScope.TENANT)
language = await config_manager.get("user.language", scope=ConfigScope.USER)
```

## Configuration Structure

### Default APG Configuration

The system comes with built-in default configurations:

```yaml
apg:
  system:
    name: "APG Platform"
    version: "2.0.0"
  database:
    host: "localhost"
    port: 5432
    name: "apg_platform"
  security:
    secret_key: "change-me-in-production"  # Sensitive
  logging:
    level: "INFO"
  cache:
    redis:
      host: "localhost"
      port: 6379
  messaging:
    rabbitmq:
      host: "localhost"
      port: 5672
  auth:
    jwt:
      secret: "jwt-secret-change-me"  # Sensitive
      expiration: 3600
  monitoring:
    enabled: true
  multitenancy:
    enabled: true
```

### Environment-Specific Configurations

Create environment-specific configuration files:

**config/development.yaml:**
```yaml
apg:
  database:
    host: "localhost"
    name: "apg_platform_dev"
  logging:
    level: "DEBUG"
  cache:
    enabled: false
```

**config/production.yaml:**
```yaml
apg:
  database:
    host: "${DB_HOST}"
    name: "${DB_NAME}"
  logging:
    level: "WARNING"
  cache:
    enabled: true
  monitoring:
    enabled: true
```

## Usage Examples

### Basic Configuration Operations

```python
import asyncio
from apg.capabilities.common.conf import get_config_manager

async def config_examples():
    config_manager = get_config_manager()
    
    # Get configuration
    db_host = await config_manager.get("apg.database.host")
    print(f"Database host: {db_host}")
    
    # Set configuration with validation
    await config_manager.set("apg.logging.level", "DEBUG")
    
    # Set sensitive configuration
    await config_manager.set(
        "app.api.secret", 
        "super-secret-key", 
        is_sensitive=True
    )
    
    # List configurations
    keys = config_manager.list_keys(prefix="apg.database")
    print(f"Database configs: {keys}")

# Run example
asyncio.run(config_examples())
```

### Configuration Watching

```python
async def config_watcher():
    config_manager = get_config_manager()
    
    # Register configuration change watcher
    async def on_config_change(key, value):
        print(f"Configuration changed: {key} = {value}")
    
    config_manager.watch(on_config_change, "app.*")
    
    # Changes to keys starting with "app." will trigger the watcher
    await config_manager.set("app.debug", True)  # Will trigger watcher
    await config_manager.set("system.name", "Test")  # Won't trigger watcher
```

### Multi-tenant Configuration

```python
async def multitenant_config():
    config_manager = get_config_manager()
    
    # Configure for tenant A
    config_manager.set_context(tenant_id="tenant-a")
    await config_manager.set("app.theme", "dark", scope=ConfigScope.TENANT)
    await config_manager.set("app.currency", "USD", scope=ConfigScope.TENANT)
    
    # Configure for tenant B
    config_manager.set_context(tenant_id="tenant-b")
    await config_manager.set("app.theme", "light", scope=ConfigScope.TENANT)
    await config_manager.set("app.currency", "EUR", scope=ConfigScope.TENANT)
    
    # Get tenant-specific values
    config_manager.set_context(tenant_id="tenant-a")
    theme_a = await config_manager.get("app.theme", scope=ConfigScope.TENANT)
    print(f"Tenant A theme: {theme_a}")  # "dark"
    
    config_manager.set_context(tenant_id="tenant-b")
    theme_b = await config_manager.get("app.theme", scope=ConfigScope.TENANT)
    print(f"Tenant B theme: {theme_b}")  # "light"
```

### File Operations

```python
from pathlib import Path

async def file_operations():
    config_manager = get_config_manager()
    
    # Load configuration from file
    config_file = Path("./my_config.yaml")
    loaded_configs = await config_manager.load_from_file(config_file)
    print(f"Loaded {len(loaded_configs)} configurations")
    
    # Save configuration to file
    await config_manager.set("export.test.key", "test_value")
    output_file = Path("./exported_config.yaml")
    await config_manager.save_to_file(output_file, keys=["export.test.key"])
    print(f"Configuration saved to {output_file}")
```

## REST API

### Endpoints

The configuration management capability provides a full REST API:

#### Get Configuration
```bash
GET /api/config/apg.database.host
GET /api/config/app.theme?scope=tenant&tenant_id=company-123
```

#### Set Configuration
```bash
PUT /api/config/app.debug.enabled
Content-Type: application/json

{
  "value": true,
  "description": "Enable debug mode"
}
```

#### List Configurations
```bash
GET /api/config?prefix=apg.database&include_sensitive=false
```

#### Delete Configuration
```bash
DELETE /api/config/app.temp.setting?scope=user&user_id=user-456
```

#### Backup Configuration
```bash
POST /api/config/backup
Content-Type: application/json

{
  "backup_path": "/tmp/config_backup.yaml",
  "scope": "system"
}
```

#### Restore Configuration
```bash
POST /api/config/restore
Content-Type: application/json

{
  "backup_path": "/tmp/config_backup.yaml"
}
```

#### Validate Configuration
```bash
POST /api/config/validate/apg.database.port
Content-Type: application/json

{
  "value": 5432
}
```

### API Response Format

```json
{
  "success": true,
  "key": "apg.database.host",
  "value": "localhost",
  "scope": "system",
  "source": "default",
  "last_updated": "2025-01-08T10:30:00Z",
  "description": "Database host",
  "is_sensitive": false
}
```

## Configuration Templates

Built-in configuration templates for different environments:

### Development Template
```python
from apg.capabilities.common.conf.service import get_config_service

config_service = get_config_service()
await config_service.load_configuration_template("development")
```

### Production Template
```python
await config_service.load_configuration_template("production")
```

### Custom Templates
```python
custom_template = {
    "app.feature.x": {
        "value": True,
        "scope": "system",
        "description": "Enable feature X"
    },
    "app.timeout": {
        "value": 30,
        "scope": "system", 
        "description": "Request timeout in seconds"
    }
}

config_service.register_template("custom", custom_template)
await config_service.load_configuration_template("custom")
```

## Validation

### Built-in Validators

The system includes validators for common configuration types:

- **Database ports**: Must be valid port numbers (1-65535)
- **Log levels**: Must be valid logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **JWT expiration**: Must be positive integers

### Custom Validators

```python
from apg.capabilities.common.conf.service import get_config_service

def validate_email(value):
    """Custom email validator"""
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(email_pattern, value):
        return True, None
    return False, "Invalid email format"

# Register custom validator
config_service = get_config_service()
config_service.register_validator("app.admin.email", validate_email)

# Now validation will be applied when setting this configuration
await config_service.set_configuration(
    "app.admin.email", 
    "admin@company.com", 
    validate=True
)
```

## Security Considerations

### Sensitive Configuration
- Mark sensitive configurations with `is_sensitive=True`
- Sensitive configurations are excluded from backups and logs
- Use environment variables for production secrets

### Environment Variables
- Override any configuration with environment variables
- Format: `APG_DATABASE_HOST` for key `apg.database.host`
- Automatic type parsing (boolean, integer, JSON)

### Multi-tenant Isolation
- Tenant-scoped configurations are completely isolated
- No cross-tenant configuration access
- Secure context switching

## Integration with Other Capabilities

This capability is used by all other APG capabilities:

```python
# Other capabilities use configuration
from apg.capabilities.common.conf import get_config

# Database capability
db_host = await get_config("apg.database.host")
db_port = await get_config("apg.database.port") 

# Cache capability  
redis_host = await get_config("apg.cache.redis.host")

# Auth capability
jwt_secret = await get_config("apg.auth.jwt.secret")
```

## Development & Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest apg/capabilities/common/conf/tests/

# Run with coverage
pytest --cov=apg.capabilities.common.conf
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Set development environment
export APG_ENVIRONMENT=development
export APG_CONFIG_DIR=./config
```

## Performance Considerations

- **Caching**: All configurations are cached in memory for fast access
- **Async Operations**: All operations are async for better concurrency
- **File Loading**: Configuration files are loaded once and cached
- **Watchers**: Minimal overhead for configuration change notifications

## Troubleshooting

### Common Issues

1. **Configuration not found**
   - Check key spelling and case
   - Verify configuration scope and context
   - Check if environment variable override exists

2. **Validation errors**
   - Review validator requirements
   - Check value type and format
   - Disable validation temporarily for debugging

3. **File loading issues**
   - Verify file path and permissions
   - Check YAML/JSON syntax
   - Ensure configuration directory exists

### Debug Configuration

```python
# Enable debug mode
await set_config("apg.logging.level", "DEBUG")

# List all configurations for debugging
config_manager = get_config_manager()
all_keys = config_manager.list_keys()
print(f"All configuration keys: {all_keys}")
```

## License

© 2025 Datacraft. All rights reserved.

---

**Next Steps**: After implementing the Configuration Management capability, proceed with [Audit Logging (`audl`)](../audl/README.md) as defined in the development order plan.
