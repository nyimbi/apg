# Getting Started with APG MDM

This guide will help you get up and running with APG Master Data Management in minutes.

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- PostgreSQL 14 or higher
- Redis 6.0 or higher (optional, for caching)
- APG Core Framework installed

## Installation

### Option 1: Via APG Framework (Recommended)

If you have APG installed, MDM is available as a built-in capability:

```bash
# APG MDM is included in the APG framework
apg capability enable mdm
apg capability init mdm
```

### Option 2: Standalone Installation

For development or standalone deployment:

```bash
# Clone the repository
git clone https://github.com/datacraft/apg
cd apg/capabilities/common/mdm

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Database Setup

### PostgreSQL Configuration

1. **Create Database and User**

```sql
-- Connect as postgres superuser
CREATE DATABASE apg_mdm;
CREATE USER mdm_user WITH ENCRYPTED PASSWORD 'your_secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE apg_mdm TO mdm_user;
GRANT CREATE ON SCHEMA public TO mdm_user;

-- Connect to apg_mdm database
\c apg_mdm

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

2. **Environment Configuration**

Create a `.env` file in your project root:

```env
# Database Configuration
DATABASE_URL=postgresql://mdm_user:your_secure_password@localhost:5432/apg_mdm

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# APG Configuration
APG_TENANT_ID=your-tenant-id
APG_SECRET_KEY=your-secret-key-here

# MDM Specific Configuration
MDM_ENABLE_AI=true
MDM_ENABLE_CACHING=true
MDM_LOG_LEVEL=INFO
```

### Database Schema Creation

Initialize the database schema:

```python
from apg.capabilities.common.mdm.database import MDMDatabaseManager

# Initialize database
db_manager = MDMDatabaseManager()
await db_manager.initialize()
await db_manager.create_schema()
```

Or using the CLI:

```bash
# Using APG CLI
apg mdm init-db

# Or using Python script
python -m apg.capabilities.common.mdm.database --init
```

## Basic Configuration

### Minimal Configuration

```python
# config.py
MDM_CONFIG = {
    "database_url": "postgresql://mdm_user:password@localhost:5432/apg_mdm",
    "enable_ai": True,
    "enable_caching": True,
    "quality_thresholds": {
        "excellent": 95.0,
        "good": 80.0,
        "fair": 60.0,
        "poor": 40.0
    }
}
```

### APG Integration Configuration

```python
# APG ecosystem integration
APG_INTEGRATION_CONFIG = {
    "mqeb_url": "http://localhost:8080",  # Message Queue Event Bus
    "cach_url": "redis://localhost:6379/0",  # Caching
    "audl_url": "http://localhost:8081",  # Audit Logging
    "conf_url": "http://localhost:8082",  # Configuration Management
}
```

## First Steps

### 1. Initialize the MDM Service

```python
import asyncio
from apg.capabilities.common.mdm import MDMService

async def init_mdm():
    # Initialize MDM service
    mdm_service = MDMService()
    await mdm_service.initialize()
    
    # Check health
    health = await mdm_service.health_check()
    print(f"MDM Status: {health['status']}")
    
    return mdm_service

# Run initialization
mdm = asyncio.run(init_mdm())
```

### 2. Create Your First Entity

```python
from apg.capabilities.common.mdm.models import MdEntityCreate, EntityType, EntityStatus

async def create_first_entity(mdm_service):
    # Define entity data
    entity_data = MdEntityCreate(
        tenant_id="demo-tenant",
        entity_type=EntityType.PERSON,
        entity_name="Alice Johnson",
        entity_description="Demo customer record",
        business_key="CUST-001",
        source_system="demo_system",
        status=EntityStatus.ACTIVE,
        attributes={
            "first_name": "Alice",
            "last_name": "Johnson",
            "email": "alice.johnson@example.com",
            "phone": "+1-555-123-4567",
            "department": "Engineering",
            "hire_date": "2023-01-15",
            "employee_id": "EMP-1001"
        },
        tags=["employee", "engineering", "active"],
        data_classification="internal"
    )
    
    # Create operation context
    context = mdm_service.create_operation_context(
        tenant_id="demo-tenant",
        user_id="demo-user",
        operation_type="create_entity",
        source_system="getting_started_guide"
    )
    
    # Create the entity
    result = await mdm_service.entity_service.create_entity(entity_data, context)
    
    if result["status"] == "success":
        print(f"✅ Entity created successfully!")
        print(f"   Entity ID: {result['entity_id']}")
        return result["entity_id"]
    else:
        print(f"❌ Error creating entity: {result['message']}")
        return None

# Create your first entity
entity_id = asyncio.run(create_first_entity(mdm))
```

### 3. Retrieve and Display the Entity

```python
async def get_entity_details(mdm_service, entity_id, tenant_id):
    # Get entity with full details
    result = await mdm_service.entity_service.get_entity(
        entity_id, 
        tenant_id,
        include_versions=True,
        include_quality=True,
        include_cross_refs=True
    )
    
    if result["status"] == "success":
        entity = result["entity"]
        print(f"\n📋 Entity Details:")
        print(f"   ID: {entity['entity_id']}")
        print(f"   Name: {entity['entity_name']}")
        print(f"   Type: {entity['entity_type']}")
        print(f"   Business Key: {entity['business_key']}")
        print(f"   Quality Score: {entity['quality_score']:.1f}%")
        print(f"   Status: {entity['status']}")
        print(f"   Created: {entity['created_at']}")
        
        # Display attributes
        print(f"\n📊 Attributes:")
        for key, value in entity['attributes'].items():
            print(f"   {key}: {value}")
            
        # Display tags
        if entity['tags']:
            print(f"\n🏷️  Tags: {', '.join(entity['tags'])}")
            
        return entity
    else:
        print(f"❌ Error retrieving entity: {result['message']}")
        return None

# Get entity details
if entity_id:
    entity = asyncio.run(get_entity_details(mdm, entity_id, "demo-tenant"))
```

### 4. Run Quality Assessment

```python
async def assess_entity_quality(mdm_service, entity_id, tenant_id):
    # Get entity data first
    entity_result = await mdm_service.entity_service.get_entity(entity_id, tenant_id)
    if entity_result["status"] != "success":
        print("❌ Could not retrieve entity for quality assessment")
        return
    
    entity = entity_result["entity"]
    
    # Run quality assessment
    quality_result = await mdm_service.quality_service.assess_quality(
        entity_id,
        tenant_id,
        entity["attributes"],
        entity["entity_type"]
    )
    
    if quality_result["status"] == "success":
        print(f"\n🔍 Quality Assessment Results:")
        print(f"   Overall Score: {quality_result['overall_score']:.1f}%")
        print(f"   Quality Status: {quality_result['quality_status'].title()}")
        print(f"   Assessment Time: {quality_result.get('assessment_duration_ms', 0):.1f}ms")
        
        # Dimension scores
        dimensions = [
            ("Completeness", quality_result['completeness_score']),
            ("Accuracy", quality_result['accuracy_score']),
            ("Consistency", quality_result['consistency_score']),
            ("Validity", quality_result['validity_score']),
            ("Uniqueness", quality_result['uniqueness_score']),
            ("Timeliness", quality_result['timeliness_score'])
        ]
        
        print(f"\n📈 Quality Dimensions:")
        for dimension, score in dimensions:
            print(f"   {dimension}: {score:.1f}%")
        
        # Show recommendations if any
        if quality_result.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in quality_result['recommendations']:
                print(f"   • {rec}")
                
    else:
        print(f"❌ Quality assessment failed: {quality_result['message']}")

# Run quality assessment
if entity_id:
    asyncio.run(assess_entity_quality(mdm, entity_id, "demo-tenant"))
```

### 5. Search for Entities

```python
async def search_entities_example(mdm_service, tenant_id):
    # Search for entities
    search_criteria = {
        "entity_type": EntityType.PERSON,
        "limit": 10,
        "offset": 0,
        "sort_by": "created_at",
        "sort_order": "desc"
    }
    
    result = await mdm_service.entity_service.search_entities(tenant_id, search_criteria)
    
    if result["status"] == "success":
        entities = result["entities"]
        pagination = result["pagination"]
        
        print(f"\n🔎 Search Results ({pagination['total_count']} total):")
        
        for entity in entities:
            print(f"   • {entity['entity_name']} ({entity['business_key']})")
            print(f"     Type: {entity['entity_type']} | Quality: {entity['quality_score']:.1f}%")
            print(f"     Created: {entity['created_at']}")
            print()
    else:
        print(f"❌ Search failed: {result['message']}")

# Search for entities
asyncio.run(search_entities_example(mdm, "demo-tenant"))
```

## Running the Web Interface

### Using Flask Blueprint

```python
from flask import Flask
from flask_appbuilder import AppBuilder, SQLA
from apg.capabilities.common.mdm.blueprint import register_mdm_views

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://...'

# Initialize Flask-AppBuilder
db = SQLA(app)
appbuilder = AppBuilder(app, db.session)

# Register MDM views
register_mdm_views(appbuilder, mdm_service)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

Access the web interface at: `http://localhost:5000/mdm`

### Using FastAPI

```python
from fastapi import FastAPI
from apg.capabilities.common.mdm.api import create_mdm_app

# Create FastAPI app with MDM
app = create_mdm_app(mdm_service)

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Access the API at: `http://localhost:8000/docs`

## Next Steps

Now that you have APG MDM running, explore these advanced features:

1. **[API Reference](api_reference.md)** - Complete API documentation
2. **[Quality Management](user_guide.md#quality-management)** - Advanced quality assessment
3. **[Duplicate Detection](user_guide.md#duplicate-detection)** - Entity matching and deduplication
4. **[Golden Records](user_guide.md#golden-records)** - Master record management
5. **[Batch Operations](user_guide.md#batch-operations)** - High-volume data processing
6. **[APG Integration](developer_guide.md#apg-integration)** - Ecosystem integration

## Troubleshooting

### Common Issues

**Database Connection Error**
```
psycopg2.OperationalError: could not connect to server
```
- Verify PostgreSQL is running
- Check database URL and credentials
- Ensure database exists

**Permission Denied**
```
psycopg2.errors.InsufficientPrivilege: permission denied for schema public
```
- Grant proper permissions to MDM user
- Ensure user can create tables and indexes

**Import Errors**
```
ModuleNotFoundError: No module named 'apg.capabilities.common.mdm'
```
- Ensure APG MDM is properly installed
- Check Python path and virtual environment

### Getting Help

- Check the [FAQ](faq.md)
- Review [examples](../examples/)
- Open an issue on [GitHub](https://github.com/datacraft/apg/issues)
- Contact support: nyimbi@gmail.com

## Performance Tips

1. **Enable Caching** - Use Redis for better performance
2. **Index Strategy** - Ensure proper database indexes
3. **Batch Operations** - Use bulk operations for large datasets
4. **Connection Pooling** - Configure appropriate pool sizes
5. **Monitoring** - Enable metrics and monitoring

Congratulations! You're now ready to use APG MDM for world-class master data management. 🎉