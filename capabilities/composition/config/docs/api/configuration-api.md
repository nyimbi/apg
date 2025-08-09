# Configuration API Reference

## Core Configuration Service

The `CentralConfigurationService` provides the primary interface for configuration management.

### Class: CentralConfigurationService

#### Constructor Parameters

```python
CentralConfigurationService(
    tenant_id: str,
    encryption_enabled: bool = True,
    realtime_sync: bool = True,
    database_config: Optional[Dict[str, Any]] = None,
    security_config: Optional[Dict[str, Any]] = None
)
```

- `tenant_id`: Unique identifier for the tenant
- `encryption_enabled`: Enable quantum-resistant encryption
- `realtime_sync`: Enable real-time synchronization
- `database_config`: Database connection configuration
- `security_config`: Security engine configuration

#### Factory Method

```python
@classmethod
async def create(
    cls,
    tenant_id: str,
    encryption_enabled: bool = True,
    realtime_sync: bool = True,
    **kwargs
) -> "CentralConfigurationService"
```

Creates and initializes a new service instance.

### Configuration Management

#### Set Configuration

```python
async def set_config(
    self,
    key: str,
    value: Any,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    validate_schema: bool = True
) -> bool
```

Sets a configuration value with validation and encryption.

**Parameters:**
- `key`: Configuration key (supports hierarchical dot notation)
- `value`: Configuration value (any JSON-serializable type)
- `user_id`: User making the change (for audit)
- `metadata`: Additional metadata for the configuration
- `validate_schema`: Whether to validate against registered schemas

**Returns:** `bool` - Success status

**Example:**
```python
# Simple configuration
await service.set_config("app.database.host", "localhost")

# Complex configuration with metadata
await service.set_config(
    "app.features.experimental",
    {"feature_a": True, "feature_b": False},
    user_id="user123",
    metadata={"environment": "staging", "reason": "testing"}
)
```

#### Get Configuration

```python
async def get_config(
    self,
    key: str,
    default: Any = None,
    user_id: Optional[str] = None,
    decrypt: bool = True
) -> Any
```

Retrieves a configuration value with decryption.

**Parameters:**
- `key`: Configuration key to retrieve
- `default`: Default value if key doesn't exist
- `user_id`: User requesting the value (for audit)
- `decrypt`: Whether to decrypt encrypted values

**Returns:** Configuration value or default

**Example:**
```python
# Get simple value
host = await service.get_config("app.database.host", "localhost")

# Get complex object
features = await service.get_config("app.features.experimental", {})
```

#### Delete Configuration

```python
async def delete_config(
    self,
    key: str,
    user_id: Optional[str] = None
) -> bool
```

Deletes a configuration key.

**Parameters:**
- `key`: Configuration key to delete
- `user_id`: User performing the deletion

**Returns:** `bool` - Success status

#### List Configurations

```python
async def list_configs(
    self,
    pattern: Optional[str] = None,
    include_metadata: bool = False,
    user_id: Optional[str] = None
) -> Dict[str, Any]
```

Lists all configurations matching a pattern.

**Parameters:**
- `pattern`: Glob pattern to filter keys (e.g., "app.database.*")
- `include_metadata`: Include metadata in results
- `user_id`: User requesting the list

**Returns:** Dictionary of matching configurations

### Batch Operations

#### Batch Set

```python
async def batch_set_configs(
    self,
    configs: Dict[str, Any],
    user_id: Optional[str] = None,
    validate_all: bool = True
) -> Dict[str, bool]
```

Sets multiple configurations in a single transaction.

**Parameters:**
- `configs`: Dictionary of key-value pairs to set
- `user_id`: User performing the batch operation
- `validate_all`: Validate all configs before setting any

**Returns:** Dictionary of key -> success status

**Example:**
```python
configs = {
    "app.database.host": "localhost",
    "app.database.port": 5432,
    "app.database.name": "myapp"
}
results = await service.batch_set_configs(configs, user_id="admin")
```

#### Batch Get

```python
async def batch_get_configs(
    self,
    keys: List[str],
    user_id: Optional[str] = None
) -> Dict[str, Any]
```

Retrieves multiple configurations efficiently.

### Schema Management

#### Register Schema

```python
async def register_schema(
    self,
    key_pattern: str,
    schema: Dict[str, Any],
    version: str = "1.0.0"
) -> bool
```

Registers a JSON schema for configuration validation.

**Parameters:**
- `key_pattern`: Pattern for keys this schema applies to
- `schema`: JSON Schema definition
- `version`: Schema version

**Example:**
```python
schema = {
    "type": "object",
    "properties": {
        "host": {"type": "string"},
        "port": {"type": "integer", "minimum": 1, "maximum": 65535}
    },
    "required": ["host", "port"]
}
await service.register_schema("app.database.*", schema)
```

#### Validate Configuration

```python
async def validate_config(
    self,
    key: str,
    value: Any
) -> Tuple[bool, List[str]]
```

Validates a configuration against its registered schema.

**Returns:** Tuple of (is_valid, error_messages)

### Versioning and History

#### Get Configuration History

```python
async def get_config_history(
    self,
    key: str,
    limit: int = 50,
    user_id: Optional[str] = None
) -> List[Dict[str, Any]]
```

Retrieves the change history for a configuration key.

**Returns:** List of historical changes with timestamps, users, and values

#### Rollback Configuration

```python
async def rollback_config(
    self,
    key: str,
    version: int,
    user_id: Optional[str] = None
) -> bool
```

Rolls back a configuration to a previous version.

### Search and Query

#### Search Configurations

```python
async def search_configs(
    self,
    query: str,
    search_values: bool = True,
    search_metadata: bool = False,
    limit: int = 100
) -> List[Dict[str, Any]]
```

Searches configurations using text queries.

**Parameters:**
- `query`: Search text
- `search_values`: Search in configuration values
- `search_metadata`: Search in metadata
- `limit`: Maximum results to return

#### Advanced Query

```python
async def query_configs(
    self,
    filters: Dict[str, Any],
    sort_by: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]
```

Performs advanced queries with filtering and sorting.

**Example:**
```python
# Find all database configurations
results = await service.query_configs({
    "key_pattern": "*.database.*",
    "metadata.environment": "production"
})
```

### Health and Status

#### Health Check

```python
async def health_check(self) -> Dict[str, Any]
```

Returns the health status of the configuration service.

**Returns:** Health status including database, encryption, and sync status

#### Get Statistics

```python
async def get_statistics(self) -> Dict[str, Any]
```

Returns usage statistics and metrics.

## Error Handling

All API methods use structured error handling with the following exception types:

- `ConfigurationError`: Configuration-related errors
- `ValidationError`: Schema validation failures
- `EncryptionError`: Encryption/decryption failures
- `SynchronizationError`: Real-time sync issues
- `AuthenticationError`: Authentication failures
- `AuthorizationError`: Permission denied errors

## Response Formats

### Standard Response

```json
{
    "success": true,
    "data": "configuration_value",
    "metadata": {
        "key": "app.database.host",
        "version": 1,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "created_by": "user123"
    }
}
```

### Error Response

```json
{
    "success": false,
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Configuration value does not match schema",
        "details": ["port must be an integer between 1 and 65535"]
    }
}
```

### Batch Response

```json
{
    "success": true,
    "results": {
        "app.database.host": true,
        "app.database.port": true,
        "app.invalid.config": false
    },
    "errors": {
        "app.invalid.config": "Schema validation failed"
    }
}
```

## Authentication and Authorization

All API endpoints require authentication via:
- JWT tokens (preferred)
- API keys
- OAuth2 (for enterprise integrations)

Authorization is role-based with the following levels:
- `read`: Can read configurations
- `write`: Can create/update configurations
- `admin`: Full access including schema management
- `audit`: Can access audit logs and history