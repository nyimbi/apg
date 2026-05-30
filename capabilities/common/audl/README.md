# APG Audit Logging Capability

**Comprehensive audit logging system providing secure, scalable, and queryable audit trail capabilities for the APG platform.**

## Overview

The Audit Logging capability (`audl`) is a foundational security and compliance capability that provides comprehensive audit trail functionality for the APG platform. It captures, stores, and manages audit events across all system components with integrity verification, multi-tenant isolation, and advanced querying capabilities.

## Current Executable Package

The package-backed AUDL slice is dependency-light and can be composed by
generated APG applications without starting Elasticsearch, Flask/FastAPI,
external SIEM systems, ML providers, or a running Bytewax worker. The local
runtime is `audit_runtime.AudlService`; generated apps should use
`api_helpers.py` and `view_models.py` when they need a stable package surface.

It currently provides:

- tenant-scoped audit event append with checksum enforcement;
- legal hold, regulated export, dual-control purge, and investigation
  lifecycles;
- first-class audit agent registration for `codex`, `claude_code`,
  `opencode`, and `pi`;
- role and approval guardrails for audit agents;
- Bytewax lifecycle-stream metadata and batch validation guardrails;
- UI/view-model surfaces for dashboards, evidence timelines, review queues,
  compliance, rules, settings, and audit-agent rosters.

### Dependency-Light Usage

```python
from capabilities.common.audl.audit_runtime import AudlService

service = AudlService()

event = service.append_event(
    event_id="evt-001",
    tenant_id="tenant-audl",
    actor="security-analyst",
    action="review_access",
    resource_type="account",
    resource_id="acct-001",
    severity="critical",
    escalation_configured=True,
)

agent = service.register_audit_agent(
    agent_id="agent-001",
    tenant_id="tenant-audl",
    name="Evidence Reviewer",
    runtime="codex",
    role="evidence_reviewer",
    purpose="Review chain-of-custody evidence before release.",
    owner="security-lead",
)

batch = service.validate_batch(
    tenant_id="tenant-audl",
    record_count=12000,
    event_stream="bytewax",
)
```

Use `service.describe("tenant-audl")` to inspect the configuration schema,
rules, UI routes, theme tokens, agent contract, and Bytewax stream metadata.

## Features

### ✅ Core Features
- **Comprehensive Event Tracking**: 19 built-in audit event types covering authentication, data access, security, and system events
- **Hierarchical Audit Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL logging levels
- **Multi-tenant Isolation**: Complete tenant-level audit separation and filtering
- **Integrity Verification**: SHA-256 checksums for tamper detection and log integrity
- **Dual Storage**: Database + file-based logging for redundancy and performance
- **Real-time Logging**: Async logging with configurable handlers
- **Advanced Querying**: Rich filtering, pagination, and search capabilities
- **Export Functionality**: JSON and CSV export formats
- **Automatic Cleanup**: Configurable retention policies and log rotation
- **REST API**: Complete REST API for audit operations

### 🔐 Security Features
- **Tamper Detection**: Cryptographic checksums for each audit entry
- **Non-repudiation**: Immutable audit trails with integrity verification
- **Multi-tenant Security**: Strict tenant isolation and access controls
- **Sensitive Data Handling**: Special handling for security-sensitive events
- **IP and User Agent Tracking**: Complete client attribution

### 📊 Analytics & Compliance
- **Audit Summaries**: Statistical analysis and reporting
- **Component Analysis**: Per-component event tracking and statistics
- **Success/Failure Metrics**: Detailed success rate analytics
- **Timeline Analysis**: Time-based event pattern analysis
- **Compliance Reporting**: Export and retention for regulatory compliance

## Architecture Components

- **AuditLogger** - Core logging engine with async event handling
- **AuditService** - Business logic layer with advanced query capabilities  
- **DatabaseHandler** - SQLite-based persistent storage with indexing
- **FileHandler** - Daily log file rotation and archival
- **REST API** - HTTP endpoints for audit operations
- **Data Models** - Pydantic models for validation and serialization

## Installation & Setup

### Prerequisites
- Python 3.9+
- AsyncIO support
- SQLite3 (included with Python)
- Pydantic for data validation
- aiosqlite for async database operations

### Basic Setup

```python
from apg.capabilities.common.audl import init_audit_logging, audit_user_login

# Initialize audit logging system
audit_logger = init_audit_logging()

# Log user authentication events
await audit_user_login(user_id="user123", success=True, ip_address="192.168.1.100")
await audit_user_login(user_id="user456", success=False, ip_address="192.168.1.200")

# Log data access events  
await audit_data_access(resource="customer_data", action="read", user_id="user123")

# Log security events
await audit_security_event(action="brute_force_attempt", details={"attempts": 5})
```

### Advanced Setup with Service Layer

```python
from apg.capabilities.common.audl.service import init_audit_service, AuditQueryFilter

# Initialize audit service
audit_service = await init_audit_service()

# Set audit context for multi-tenant logging
audit_service.logger.set_context(tenant_id="company-123", user_id="user-456")

# Log events with context automatically applied
await audit_service.logger.log_data_access("user_profiles", resource_id="profile-789")

# Query audit logs with advanced filtering
filter_criteria = AuditQueryFilter(
    tenant_id="company-123",
    event_type=AuditEventType.DATA_ACCESS,
    start_time=datetime.utcnow() - timedelta(hours=24),
    limit=100
)
result = await audit_service.query_audit_logs(filter_criteria)
```

## Audit Event Types

The system supports 19 built-in event types:

### Authentication Events
- `USER_LOGIN` - Successful user authentication
- `USER_LOGOUT` - User session termination
- `USER_FAILED_LOGIN` - Failed authentication attempts

### User Management Events
- `USER_CREATED` - New user account creation
- `USER_UPDATED` - User account modifications
- `USER_DELETED` - User account deletion
- `PERMISSION_GRANTED` - Permission assignments
- `PERMISSION_REVOKED` - Permission removals

### Data Events
- `DATA_ACCESS` - Data read operations
- `DATA_CREATE` - Data creation operations
- `DATA_UPDATE` - Data modification operations
- `DATA_DELETE` - Data deletion operations

### System Events
- `CONFIG_CHANGE` - Configuration modifications
- `SYSTEM_START` - System startup events
- `SYSTEM_STOP` - System shutdown events
- `API_CALL` - API endpoint access

### Security Events
- `SECURITY_EVENT` - Security-related incidents
- `COMPLIANCE_EVENT` - Compliance-related events
- `CUSTOM_EVENT` - Application-specific events

## Usage Examples

### Basic Audit Logging

```python
import asyncio
from apg.capabilities.common.audl import (
    get_audit_logger, AuditLevel, AuditEventType
)

async def audit_examples():
    logger = get_audit_logger()
    
    # Set multi-tenant context
    logger.set_context(tenant_id="acme-corp", user_id="john.doe")
    
    # Log various events
    await logger.log_user_login("john.doe", success=True, ip_address="10.0.1.100")
    
    await logger.log_data_access(
        resource="customer_database",
        resource_id="customer_12345",
        action="read"
    )
    
    await logger.log_security_event(
        action="suspicious_login_pattern",
        level=AuditLevel.WARNING,
        details={"pattern": "multiple_failed_attempts", "count": 3}
    )
    
    await logger.log_api_call(
        endpoint="/api/v1/customers",
        method="GET",
        status_code=200,
        duration_ms=145
    )

asyncio.run(audit_examples())
```

### Advanced Querying

```python
from apg.capabilities.common.audl.service import get_audit_service, AuditQueryFilter
from datetime import datetime, timedelta

async def query_examples():
    service = get_audit_service()
    
    # Query failed login attempts in the last 24 hours
    failed_logins = AuditQueryFilter(
        event_type=AuditEventType.USER_FAILED_LOGIN,
        start_time=datetime.utcnow() - timedelta(hours=24),
        success=False
    )
    result = await service.query_audit_logs(failed_logins)
    print(f"Failed login attempts: {result.total_count}")
    
    # Query data access by specific user
    user_activity = AuditQueryFilter(
        user_id="john.doe",
        event_type=AuditEventType.DATA_ACCESS,
        start_time=datetime.utcnow() - timedelta(days=7)
    )
    result = await service.query_audit_logs(user_activity)
    
    # Query security events by tenant
    security_events = AuditQueryFilter(
        tenant_id="acme-corp",
        event_type=AuditEventType.SECURITY_EVENT,
        level=AuditLevel.WARNING
    )
    result = await service.query_audit_logs(security_events)
```

### Audit Analytics

```python
async def analytics_examples():
    service = get_audit_service()
    
    # Get comprehensive audit summary
    summary = await service.get_audit_summary(tenant_id="acme-corp", days=30)
    print(f"Total events: {summary['total_events']}")
    print(f"Success rate: {summary['success_rate']}%")
    print(f"Event types: {summary['event_types']}")
    print(f"Components: {summary['components']}")
    
    # Verify audit log integrity
    integrity = await service.verify_log_integrity(tenant_id="acme-corp")
    print(f"Integrity: {integrity['integrity_percentage']}%")
    print(f"Corrupted entries: {integrity['corrupted_count']}")
    
    # Export audit logs for compliance
    export_filter = AuditQueryFilter(
        tenant_id="acme-corp",
        start_time=datetime.utcnow() - timedelta(days=90)
    )
    csv_data = await service.export_audit_logs(export_filter, format="csv")
    with open("compliance_audit.csv", "w") as f:
        f.write(csv_data)
```

### Custom Event Handlers

```python
async def custom_handler_example():
    logger = get_audit_logger()
    
    # Define custom handlers
    async def security_alert_handler(entry):
        if entry.event_type == AuditEventType.SECURITY_EVENT:
            # Send alert to security team
            print(f"SECURITY ALERT: {entry.action} by {entry.user_id}")
    
    def compliance_log_handler(entry):
        if entry.component in ["payment", "financial"]:
            # Log to compliance system
            print(f"COMPLIANCE LOG: {entry.action} on {entry.resource}")
    
    # Register handlers
    logger.add_handler(security_alert_handler)
    logger.add_handler(compliance_log_handler)
    
    # Events will now trigger custom handlers
    await logger.log_security_event("unauthorized_access_attempt")
```

## REST API

### Endpoints

The audit logging capability provides a comprehensive REST API:

#### Query Audit Logs
```bash
GET /api/audit/logs?tenant_id=acme-corp&event_type=user_login&limit=100&offset=0
```

#### Get Audit Summary
```bash
GET /api/audit/logs/summary?tenant_id=acme-corp&days=30
```

#### Export Audit Logs
```bash
POST /api/audit/logs/export
Content-Type: application/json

{
  "format": "csv",
  "tenant_id": "acme-corp",
  "start_time": "2025-01-01T00:00:00Z",
  "end_time": "2025-01-31T23:59:59Z",
  "event_type": "data_access"
}
```

#### Verify Log Integrity
```bash
GET /api/audit/logs/integrity?tenant_id=acme-corp
```

#### Cleanup Old Logs
```bash
POST /api/audit/logs/cleanup
Content-Type: application/json

{
  "retention_days": 365
}
```

#### Get Available Event Types
```bash
GET /api/audit/logs/event-types
```

### API Response Format

```json
{
  "success": true,
  "total_count": 1523,
  "count": 100,
  "has_more": true,
  "entries": [
    {
      "id": "01HKQR9Z8XVJ2K3L4M5N6P7Q8R",
      "timestamp": "2025-01-08T10:30:00.123Z",
      "level": "INFO",
      "event_type": "user_login",
      "tenant_id": "acme-corp",
      "user_id": "john.doe",
      "component": "authentication",
      "action": "login",
      "success": true,
      "ip_address": "10.0.1.100",
      "duration_ms": 234,
      "checksum": "7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"
    }
  ]
}
```

## Configuration

### Environment Variables
- `APG_AUDIT_ENABLED` - Enable/disable audit logging (default: true)
- `APG_AUDIT_LEVEL` - Minimum audit level (default: INFO)
- `APG_AUDIT_RETENTION_DAYS` - Log retention period (default: 90)
- `APG_AUDIT_DIR` - Audit storage directory (default: ./audit)

### Configuration Templates

```python
from apg.capabilities.common.audl.models import AUDIT_CONFIGURATION_TEMPLATES

# Load development configuration
development_config = AUDIT_CONFIGURATION_TEMPLATES["development"]

# Load production configuration  
production_config = AUDIT_CONFIGURATION_TEMPLATES["production"]
```

## Security Considerations

### Integrity Protection
- All audit entries include SHA-256 checksums for tamper detection
- Integrity verification can detect corrupted or modified entries
- Database and file storage provide redundant integrity checking

### Multi-tenant Isolation
- Complete separation of audit data by tenant
- No cross-tenant access or data leakage
- Tenant context automatically applied to all audit events

### Data Protection
- Sensitive audit events are specially marked and handled
- IP addresses and user agents tracked for attribution
- Configurable data retention and automatic cleanup

### Access Control
- REST API requires appropriate authentication and authorization
- Admin-level access required for integrity verification and cleanup
- Tenant-scoped access controls for audit data

## Performance Considerations

- **Async Operations**: All logging operations are non-blocking
- **Dual Storage**: Database for querying, files for archival
- **Indexed Queries**: Optimized database indexes for common query patterns
- **Batch Processing**: Efficient handling of high-volume audit events
- **Configurable Retention**: Automatic cleanup prevents storage bloat

## Integration with Other Capabilities

The audit logging capability integrates seamlessly with other APG capabilities:

```python
# Authentication capability integration
from apg.capabilities.common.auth import authenticate_user
from apg.capabilities.common.audl import audit_user_login

async def secure_login(username, password, ip_address):
    result = await authenticate_user(username, password)
    await audit_user_login(
        user_id=username,
        success=result.success,
        ip_address=ip_address,
        details={"method": "password"}
    )
    return result

# Data access capability integration
from apg.capabilities.common.audl import audit_data_access

async def secure_data_access(resource, action, user_id):
    # Perform data operation
    result = await perform_data_operation(resource, action)
    
    # Log the access
    await audit_data_access(
        resource=resource,
        action=action,
        user_id=user_id,
        success=result.success,
        details=result.metadata
    )
    return result
```

## Development & Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run audit logging tests
pytest apg/capabilities/common/audl/tests/ -v

# Run with coverage
pytest --cov=apg.capabilities.common.audl apg/capabilities/common/audl/tests/
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Set development environment
export APG_ENVIRONMENT=development
export APG_AUDIT_LEVEL=DEBUG
export APG_AUDIT_DIR=./dev_audit_logs
```

## Compliance & Regulatory Support

The audit logging capability supports various compliance requirements:

### SOX Compliance
- Immutable audit trails for financial data access
- User attribution and change tracking
- Automated retention and archival

### GDPR Compliance  
- Data access logging for privacy audits
- User consent tracking and management
- Right to deletion audit trails

### HIPAA Compliance
- Healthcare data access logging
- User authentication tracking
- Security incident documentation

### SOC 2 Compliance
- Security control monitoring
- Access control verification
- System availability tracking

## Troubleshooting

### Common Issues

1. **High Storage Usage**
   - Configure appropriate retention policies
   - Enable automatic cleanup
   - Monitor audit volume and adjust logging levels

2. **Performance Issues**
   - Use async logging patterns
   - Implement appropriate query filters
   - Consider batch processing for high-volume scenarios

3. **Integrity Verification Failures**
   - Check for database corruption
   - Verify storage system integrity
   - Review audit entry modification patterns

### Debug Configuration

```python
# Enable debug logging
from apg.capabilities.common.audl import get_audit_logger, AuditLevel

logger = get_audit_logger()
await logger.log(
    level=AuditLevel.DEBUG,
    event_type=AuditEventType.CUSTOM_EVENT,
    component="debug",
    action="debug_info",
    details={"debug_data": "troubleshooting"}
)

# Query debug events
from apg.capabilities.common.audl.service import get_audit_service

service = get_audit_service()
debug_filter = AuditQueryFilter(level=AuditLevel.DEBUG, component="debug")
debug_events = await service.query_audit_logs(debug_filter)
```

## License

© 2025 Datacraft. All rights reserved.

---

**Next Steps**: After implementing the Audit Logging capability, proceed with [Multi-Tenancy (`mten`)](../mten/README.md) as defined in the development order plan.
