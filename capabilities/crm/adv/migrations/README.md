# CRM Database Migrations

Revolutionary database migration system for the APG Customer Relationship Management capability, providing comprehensive schema management with dependency resolution, rollback capabilities, and multi-tenant support.

## Features

- **Dependency Resolution**: Automatic migration ordering based on dependencies
- **Rollback Support**: Safe rollback capabilities with validation
- **Multi-tenant Isolation**: Full tenant isolation and data protection
- **Schema Validation**: Comprehensive schema state validation
- **Health Monitoring**: Built-in health checks and monitoring
- **Transaction Safety**: All migrations run within database transactions
- **Audit Trail**: Complete migration history and audit logging

## Quick Start

1. **Configure Database Connection**:
   ```bash
   export CRM_DB_HOST=localhost
   export CRM_DB_PORT=5432
   export CRM_DB_NAME=crm_database
   export CRM_DB_USER=crm_user
   export CRM_DB_PASSWORD=your_password
   ```

2. **Run Migrations**:
   ```bash
   python migrate.py migrate
   ```

3. **Check Status**:
   ```bash
   python migrate.py status
   ```

## Usage

### Migration Commands

- **Migrate to Latest**: Apply all pending migrations
  ```bash
  python migrate.py migrate
  ```

- **Migrate Specific Version**: Apply specific migration
  ```bash
  python migrate.py migrate 001_initial_schema
  ```

- **Rollback Last Migration**: Rollback the most recent migration
  ```bash
  python migrate.py rollback
  ```

- **Rollback to Specific Version**: Rollback to a specific migration
  ```bash
  python migrate.py rollback 001_initial_schema
  ```

- **Show Status**: Display migration status and history
  ```bash
  python migrate.py status
  ```

- **Validate Schema**: Validate current database schema
  ```bash
  python migrate.py validate
  ```

- **Health Check**: Check migration system health
  ```bash
  python migrate.py health
  ```

## Available Migrations

### 001_initial_schema
- **Description**: Create initial CRM database schema with core tables
- **Tables Created**:
  - `crm_contacts` - Contact management
  - `crm_accounts` - Account/company management
  - `crm_leads` - Lead tracking and management
  - `crm_opportunities` - Sales opportunity tracking
  - `crm_activities` - Activity and task management
  - `crm_campaigns` - Marketing campaign management
- **Features**:
  - Multi-tenant isolation
  - Comprehensive indexing
  - Full-text search capabilities
  - Audit trail columns
  - Advanced constraints and validation

### 002_advanced_features
- **Description**: Add advanced CRM features: AI insights, relationships, communication history
- **Dependencies**: `001_initial_schema`
- **Tables Created**:
  - `crm_ai_insights` - AI-powered insights and recommendations
  - `crm_communications` - Communication history tracking
  - `crm_contact_relationships` - Contact relationship mapping
  - `crm_pipeline_stages` - Customizable sales pipeline stages
  - `crm_campaign_members` - Campaign membership tracking
- **Enhanced Columns**:
  - AI engagement scores for contacts
  - Predictive analytics for opportunities
  - Communication preferences
  - Performance metrics for campaigns

## Migration Development

### Creating New Migrations

1. **Create Migration File**:
   ```python
   # migrations/migration_003_your_feature.py
   from .base_migration import BaseMigration
   
   class YourFeatureMigration(BaseMigration):
       def _get_migration_id(self) -> str:
           return "003_your_feature"
       
       def _get_version(self) -> str:
           return "003"
       
       def _get_description(self) -> str:
           return "Add your feature description"
       
       def _get_dependencies(self) -> list:
           return ["002_advanced_features"]
       
       async def up(self, connection):
           # Forward migration logic
           pass
       
       async def down(self, connection):
           # Rollback migration logic
           pass
   ```

2. **Migration Best Practices**:
   - Always use transactions
   - Include comprehensive validation
   - Add appropriate indexes
   - Follow naming conventions
   - Include rollback logic
   - Test thoroughly

### Migration Guidelines

- **Naming**: Use format `migration_XXX_descriptive_name.py`
- **Versioning**: Use sequential 3-digit versions (001, 002, 003...)
- **Dependencies**: Explicitly declare migration dependencies
- **Reversibility**: Make migrations reversible when possible
- **Validation**: Include pre/post-condition validation
- **Testing**: Test both forward and rollback directions

## Database Schema

### Core Tables

- **crm_contacts**: Contact information and management
- **crm_accounts**: Company/account information
- **crm_leads**: Lead tracking and qualification
- **crm_opportunities**: Sales pipeline and opportunity management
- **crm_activities**: Tasks, calls, meetings, and activities
- **crm_campaigns**: Marketing campaigns and tracking

### Advanced Tables

- **crm_ai_insights**: AI-powered insights and recommendations
- **crm_communications**: Email, call, and communication history
- **crm_contact_relationships**: Relationship mapping between contacts
- **crm_pipeline_stages**: Customizable sales pipeline configuration
- **crm_campaign_members**: Campaign membership and tracking

### System Tables

- **crm_schema_migrations**: Migration history and tracking

## Configuration

### Environment Variables

- `CRM_DB_HOST`: Database host (default: localhost)
- `CRM_DB_PORT`: Database port (default: 5432)
- `CRM_DB_NAME`: Database name (default: crm_db)
- `CRM_DB_USER`: Database user (default: crm_user)
- `CRM_DB_PASSWORD`: Database password (default: crm_password)

### Configuration File

Copy `config.example.json` to `config.json` and customize:

```json
{
  "database": {
    "host": "your-db-host",
    "port": 5432,
    "database": "your-crm-db",
    "user": "your-db-user",
    "password": "your-db-password"
  }
}
```

## Monitoring and Maintenance

### Health Checks

Regular health checks ensure the migration system is functioning properly:

```bash
python migrate.py health
```

### Schema Validation

Validate that the database schema matches expected state:

```bash
python migrate.py validate
```

### Migration Status

Monitor migration status and history:

```bash
python migrate.py status
```

## Troubleshooting

### Common Issues

1. **Migration Fails**:
   - Check database connectivity
   - Verify migration dependencies
   - Review migration logs
   - Validate schema state

2. **Rollback Issues**:
   - Ensure migration is reversible
   - Check for data dependencies
   - Verify rollback logic

3. **Performance Issues**:
   - Review migration complexity
   - Check index creation
   - Monitor transaction size

### Recovery

If migrations fail or leave the database in an inconsistent state:

1. **Check Migration Status**:
   ```bash
   python migrate.py status
   ```

2. **Validate Schema**:
   ```bash
   python migrate.py validate
   ```

3. **Manual Recovery**:
   - Review migration logs
   - Fix data inconsistencies
   - Update migration history if needed

## Security Considerations

- **Database Credentials**: Use environment variables or secure configuration
- **Multi-tenant Isolation**: All queries include tenant_id filtering
- **Input Validation**: All migrations include data validation
- **Audit Trail**: Complete migration history is maintained
- **Transaction Safety**: All changes are atomic and reversible

## Support

For issues, questions, or contributions:
- **Email**: nyimbi@gmail.com
- **Company**: Datacraft (www.datacraft.co.ke)
- **Documentation**: See capability documentation for detailed API reference