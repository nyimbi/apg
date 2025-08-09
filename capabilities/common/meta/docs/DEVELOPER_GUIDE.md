# APG Metadata Management Developer Guide

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Core Components](#core-components)
- [API Development](#api-development)
- [Connector Development](#connector-development)
- [AI Classification Extensions](#ai-classification-extensions)
- [Database Schema](#database-schema)
- [Testing Framework](#testing-framework)
- [Deployment Guide](#deployment-guide)
- [Performance Optimization](#performance-optimization)
- [Contributing Guidelines](#contributing-guidelines)

---

## Architecture Overview

APG Metadata Management follows a modular, microservices-inspired architecture with clear separation of concerns.

### System Architecture

```mermaid
graph TB
    UI[Web Interface] --> API[REST API Layer]
    API --> Service[Metadata Service]
    Service --> DB[(PostgreSQL)]
    Service --> Graph[(Neo4j)]
    Service --> Cache[(Redis)]
    Service --> Search[Search Engine]
    Service --> AI[AI Classifier]
    Service --> Discovery[Discovery Service]
    Discovery --> Connectors[Data Connectors]
    Connectors --> Sources[(Data Sources)]
```

### Key Architectural Principles

**1. Async-First Design**
- All I/O operations use asyncio
- Non-blocking database operations
- Concurrent discovery processing

**2. Multi-Database Strategy**
- PostgreSQL: Primary metadata storage
- Neo4j: Graph-based lineage relationships
- Redis: Caching and session management

**3. Plugin Architecture**
- Extensible connector framework
- Custom classification rules
- Configurable search engines

**4. Event-Driven Updates**
- Real-time metadata synchronization
- Webhook-based notifications
- Message queue integration

### Technology Stack

**Backend:**
- **Python 3.9+** with asyncio
- **FastAPI** for REST API
- **SQLAlchemy** for database ORM
- **Pydantic** for data validation
- **Neo4j** for graph operations
- **Redis** for caching

**Frontend:**
- **Flask-AppBuilder** for admin interface
- **React** components for interactive features
- **D3.js** for lineage visualization
- **Bootstrap** for responsive design

**Infrastructure:**
- **Docker** containers
- **PostgreSQL 12+** database
- **Redis 6+** cache
- **Neo4j 4+** graph database

---

## Development Setup

### Prerequisites

```bash
# Required software
python >= 3.9
postgresql >= 12
redis >= 6
neo4j >= 4.0

# Optional for advanced features
docker >= 20.10
elasticsearch >= 7.0
```

### Installation

**1. Clone and Setup Virtual Environment:**
```bash
git clone https://github.com/your-org/apg.git
cd apg/capabilities/common/meta

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -e .
pip install -r requirements-dev.txt
```

**2. Database Setup:**
```bash
# PostgreSQL
createdb apg_metadata_dev
psql apg_metadata_dev < schema/postgresql_schema.sql

# Redis (default configuration usually works)
redis-server

# Neo4j
# Download from neo4j.com/download
# Start Neo4j Desktop or server
```

**3. Environment Configuration:**
```bash
# Create .env file
cat > .env << EOF
POSTGRES_URL=postgresql://localhost:5432/apg_metadata_dev
REDIS_URL=redis://localhost:6379/0
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
ENVIRONMENT=development
DEBUG=True
EOF
```

**4. Initialize the Service:**
```python
# test_setup.py
import asyncio
from capabilities.common.meta import initialize_capability

async def test_setup():
    service = await initialize_capability()
    health = await service.get_health_status()
    print(f"Service Status: {health}")

if __name__ == "__main__":
    asyncio.run(test_setup())
```

### Development Workflow

**1. Code Style:**
```bash
# Install pre-commit hooks
pre-commit install

# Format code
black capabilities/
isort capabilities/

# Lint code  
flake8 capabilities/
mypy capabilities/
```

**2. Testing:**
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# With coverage
pytest --cov=capabilities.common.meta --cov-report=html
```

**3. Local Development Server:**
```bash
# Start development server
python -m capabilities.common.meta.server --debug --reload

# Or using uvicorn directly
uvicorn capabilities.common.meta.api:app --reload --host 0.0.0.0 --port 8000
```

---

## Core Components

### Service Layer Architecture

The service layer follows a clean architecture pattern:

```python
# capabilities/common/meta/service.py
class APGMetadataService:
    """Main service orchestrator"""
    
    def __init__(self, 
                 db_manager: MetaDatabaseManager,
                 search_engine: SearchEngine,
                 ai_classifier: AIClassificationEngine,
                 discovery_service: DiscoveryService):
        self.db_manager = db_manager
        self.search_engine = search_engine  
        self.ai_classifier = ai_classifier
        self.discovery_service = discovery_service
    
    async def create_asset(self, asset_data: Dict[str, Any]) -> str:
        """Create new metadata asset"""
        # Validation
        asset = AssetMetadata(**asset_data)
        
        # Store in database
        asset_id = await self.db_manager.create_asset(asset)
        
        # Index for search
        await self.search_engine.index_asset(asset)
        
        # Classify data
        if asset.columns:
            await self._classify_asset_columns(asset)
            
        return asset_id
```

### Database Manager

The database manager handles multi-database operations:

```python
# capabilities/common/meta/database_manager.py
class MetaDatabaseManager:
    """Handles all database operations across PostgreSQL, Neo4j, and Redis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.pg_pool: asyncpg.Pool = None
        self.neo4j_driver: neo4j.AsyncDriver = None  
        self.redis_client: aioredis.Redis = None
        self.config = config
    
    async def initialize(self):
        """Initialize all database connections"""
        # PostgreSQL connection pool
        self.pg_pool = await asyncpg.create_pool(
            dsn=self.config['postgresql_url'],
            min_size=5,
            max_size=20,
            command_timeout=30
        )
        
        # Neo4j connection
        self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
            self.config['neo4j_url'],
            auth=(self.config['neo4j_user'], self.config['neo4j_password'])
        )
        
        # Redis connection
        self.redis_client = aioredis.from_url(
            self.config['redis_url'],
            decode_responses=True
        )
    
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute PostgreSQL query"""
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *params if params else ())
            return [dict(row) for row in rows]
    
    async def execute_graph_query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Execute Neo4j Cypher query"""
        async with self.neo4j_driver.session() as session:
            result = await session.run(cypher, params or {})
            return [record.data() for record in await result.data()]
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        return await self.redis_client.get(key)
    
    async def cache_set(self, key: str, value: str, ttl: int = 3600):
        """Set value in Redis cache with TTL"""
        await self.redis_client.set(key, value, ex=ttl)
```

### Data Models

All data models use Pydantic for validation:

```python
# capabilities/common/meta/models.py
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

class AssetType(str, Enum):
    TABLE = "table"
    VIEW = "view"
    FILE = "file"
    API = "api"
    MODEL = "model"
    PIPELINE = "pipeline"

class DataClassification(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    PII = "PII"
    SENSITIVE_PII = "SENSITIVE_PII"

class ColumnMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Column name")
    display_name: Optional[str] = None
    data_type: str = Field(..., description="Data type")
    is_nullable: bool = False
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_reference: Optional[str] = None
    classification: Optional[DataClassification] = None
    description: Optional[str] = None
    business_name: Optional[str] = None
    data_quality: Optional[Dict[str, float]] = None
    sample_values: Optional[List[Any]] = None

class AssetMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Asset name")
    display_name: Optional[str] = None
    asset_type: AssetType = Field(..., description="Type of asset")
    source_system: str = Field(..., description="Source system name")
    database: Optional[str] = None
    schema: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    steward: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    classification: Optional[DataClassification] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    columns: List[ColumnMetadata] = Field(default_factory=list)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)
```

---

## API Development

### FastAPI Application Structure

```python
# capabilities/common/meta/api.py
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from typing import List, Optional
import asyncio

app = FastAPI(
    title="APG Metadata Management API",
    description="Enterprise metadata management and lineage tracking",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency injection
async def get_metadata_service() -> APGMetadataService:
    """Get metadata service instance"""
    return await get_service_instance()

async def get_current_tenant(token = Depends(security)) -> str:
    """Extract tenant ID from JWT token"""
    # Implementation depends on your auth system
    return "default_tenant"

# Routers
from .routers import assets, discovery, search, lineage, classification

app.include_router(assets.router, prefix="/api/v1/metadata")
app.include_router(discovery.router, prefix="/api/v1/metadata")
app.include_router(search.router, prefix="/api/v1/metadata")
app.include_router(lineage.router, prefix="/api/v1/metadata")
app.include_router(classification.router, prefix="/api/v1/metadata")
```

### Router Implementation Example

```python
# capabilities/common/meta/routers/assets.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from ..service import APGMetadataService
from ..models import AssetMetadata, AssetSearchRequest, PaginatedResponse

router = APIRouter(prefix="/assets", tags=["assets"])

@router.get("/", response_model=PaginatedResponse[AssetMetadata])
async def list_assets(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    asset_type: Optional[str] = None,
    source_system: Optional[str] = None,
    owner: Optional[str] = None,
    service: APGMetadataService = Depends(get_metadata_service),
    tenant_id: str = Depends(get_current_tenant)
):
    """List metadata assets with filtering and pagination"""
    
    filters = {}
    if asset_type:
        filters['asset_type'] = asset_type
    if source_system:
        filters['source_system'] = source_system
    if owner:
        filters['owner'] = owner
    
    try:
        result = await service.list_assets(
            tenant_id=tenant_id,
            filters=filters,
            limit=limit,
            offset=offset
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{asset_id}", response_model=AssetMetadata)
async def get_asset(
    asset_id: str,
    service: APGMetadataService = Depends(get_metadata_service),
    tenant_id: str = Depends(get_current_tenant)
):
    """Get detailed asset metadata"""
    
    try:
        asset = await service.get_asset(asset_id, tenant_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        return asset
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=dict)
async def create_asset(
    asset_data: AssetMetadata,
    service: APGMetadataService = Depends(get_metadata_service),
    tenant_id: str = Depends(get_current_tenant)
):
    """Create new metadata asset"""
    
    try:
        asset_id = await service.create_asset(
            asset_data.dict(),
            tenant_id=tenant_id
        )
        return {"asset_id": asset_id, "status": "created"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### API Error Handling

```python
# capabilities/common/meta/api/exceptions.py
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import logging
import traceback

logger = logging.getLogger(__name__)

class MetadataException(Exception):
    """Base exception for metadata operations"""
    def __init__(self, message: str, code: str = "METADATA_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)

class AssetNotFoundError(MetadataException):
    def __init__(self, asset_id: str):
        super().__init__(f"Asset {asset_id} not found", "ASSET_NOT_FOUND")

class ConnectorError(MetadataException):
    def __init__(self, message: str):
        super().__init__(message, "CONNECTOR_ERROR")

# Exception handlers
async def metadata_exception_handler(request: Request, exc: MetadataException):
    """Handle custom metadata exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

# Register exception handlers
app.add_exception_handler(MetadataException, metadata_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
```

---

## Connector Development

### Base Connector Interface

All connectors inherit from the base connector class:

```python
# capabilities/common/meta/connectors/base_connector.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum

class ConnectorType(Enum):
    DATABASE = "database"
    FILE = "file"
    API = "api"
    ML_PLATFORM = "ml_platform"
    CLOUD = "cloud"

class BaseConnector(ABC):
    """Abstract base class for all data connectors"""
    
    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.connector_type: ConnectorType = None
        self.source_system: str = None
        self.connected: bool = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
        
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection and return status"""
        pass
        
    @abstractmethod
    async def discover_assets(self) -> DiscoveryResult:
        """Discover all assets from the data source"""
        pass
        
    @abstractmethod
    async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
        """Get detailed schema for a specific asset"""
        pass
        
    @abstractmethod
    async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from the asset"""
        pass
```

### Custom Connector Implementation

Here's how to implement a custom connector:

```python
# capabilities/common/meta/connectors/custom_connector.py
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from .base_connector import BaseConnector, ConnectorType
from ..models import AssetMetadata, ColumnMetadata, DataType

class CustomAPIConnector(BaseConnector):
    """Custom connector for a specific API"""
    
    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self.connector_type = ConnectorType.API
        self.source_system = "custom_api"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Connect to the API"""
        try:
            # Create HTTP session with authentication
            headers = {
                "Authorization": f"Bearer {self.config.password}",
                "Content-Type": "application/json"
            }
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test the connection
            test_result = await self.test_connection()
            self.connected = test_result["status"] == "success"
            return self.connected
            
        except Exception as e:
            self.log_error(f"Connection failed: {str(e)}")
            return False
    
    async def disconnect(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        try:
            async with self.session.get(f"{self.config.connection_string}/health") as response:
                if response.status == 200:
                    return {"status": "success", "message": "API connection successful"}
                else:
                    return {"status": "error", "message": f"API returned {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def discover_assets(self) -> DiscoveryResult:
        """Discover API endpoints as assets"""
        result = DiscoveryResult(self.connector_type, self.source_system)
        
        try:
            # Get API schema/specification
            async with self.session.get(f"{self.config.connection_string}/schema") as response:
                if response.status == 200:
                    schema_data = await response.json()
                    
                    # Process each endpoint
                    for endpoint in schema_data.get("endpoints", []):
                        asset = AssetMetadata(
                            name=f"api_endpoint_{endpoint['name']}",
                            display_name=endpoint.get("display_name", endpoint["name"]),
                            asset_type="api",
                            source_system=self.source_system,
                            description=endpoint.get("description", ""),
                            custom_attributes={
                                "method": endpoint.get("method", "GET"),
                                "path": endpoint.get("path", ""),
                                "parameters": endpoint.get("parameters", [])
                            }
                        )
                        
                        # Add column metadata for parameters
                        for param in endpoint.get("parameters", []):
                            column = ColumnMetadata(
                                name=param["name"],
                                data_type=param.get("type", "string"),
                                is_nullable=not param.get("required", False),
                                description=param.get("description", "")
                            )
                            asset.columns.append(column)
                        
                        result.add_asset(asset)
                        
            result.complete_discovery()
            return result
            
        except Exception as e:
            result.add_error(f"Discovery failed: {str(e)}")
            result.complete_discovery()
            return result
    
    async def get_asset_schema(self, asset_name: str) -> Optional[AssetMetadata]:
        """Get detailed asset schema"""
        # Implementation specific to your API
        # Return AssetMetadata with detailed schema information
        pass
    
    async def sample_asset_data(self, asset_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from API endpoint"""
        # Implementation specific to your API
        # Return sample response data
        pass
```

### Connector Registration

Register your custom connector with the system:

```python
# capabilities/common/meta/connectors/__init__.py
from .database_connectors import PostgreSQLConnector, MySQLConnector
from .file_connectors import CSVConnector, JSONConnector
from .api_connectors import RestAPIConnector, GraphQLConnector
from .custom_connector import CustomAPIConnector

# Connector registry
CONNECTOR_REGISTRY = {
    "postgresql": PostgreSQLConnector,
    "mysql": MySQLConnector,
    "csv": CSVConnector,
    "json": JSONConnector,
    "rest_api": RestAPIConnector,
    "graphql": GraphQLConnector,
    "custom_api": CustomAPIConnector,  # Your custom connector
}

def get_connector_class(connector_type: str):
    """Get connector class by type"""
    return CONNECTOR_REGISTRY.get(connector_type)

def create_connector(connector_type: str, config: ConnectorConfig) -> BaseConnector:
    """Factory function to create connector instances"""
    connector_class = get_connector_class(connector_type)
    if not connector_class:
        raise ValueError(f"Unknown connector type: {connector_type}")
    return connector_class(config)
```

---

## AI Classification Extensions

### Custom Classification Rules

Extend the AI classification system with custom rules:

```python
# capabilities/common/meta/classification/custom_rules.py
from typing import List, Dict, Any, Optional
from ..models import ClassificationResult, ClassificationRule

class CustomClassificationRule:
    """Custom classification rule implementation"""
    
    def __init__(self, rule_config: Dict[str, Any]):
        self.rule_id = rule_config["rule_id"]
        self.name = rule_config["name"]
        self.classification = rule_config["classification"]
        self.confidence_score = rule_config["confidence_score"]
        self.conditions = rule_config["conditions"]
    
    async def apply(self, 
                   column_name: str, 
                   data_type: str, 
                   sample_data: List[Any],
                   context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Apply custom classification logic"""
        
        # Example: Classify internal IDs
        if self._is_internal_id(column_name, data_type, sample_data):
            return ClassificationResult(
                classification="INTERNAL",
                confidence_score=0.95,
                reasoning="Detected internal identifier pattern",
                method_used="custom_rule",
                tags=["identifier", "internal"]
            )
        
        # Example: Classify financial amounts
        if self._is_financial_amount(column_name, data_type, sample_data):
            return ClassificationResult(
                classification="CONFIDENTIAL",
                confidence_score=0.90,
                reasoning="Detected financial amount pattern",
                method_used="custom_rule",
                tags=["financial", "amount"]
            )
        
        return None
    
    def _is_internal_id(self, column_name: str, data_type: str, sample_data: List[Any]) -> bool:
        """Check if column represents internal ID"""
        # Column name patterns
        id_patterns = ["_id", "id_", "identifier", "key"]
        if any(pattern in column_name.lower() for pattern in id_patterns):
            # Check data type
            if data_type.lower() in ["integer", "bigint", "uuid"]:
                # Check sample data patterns
                if sample_data:
                    # All values should be unique and non-null
                    non_null_values = [v for v in sample_data if v is not None]
                    if len(set(non_null_values)) == len(non_null_values):
                        return True
        return False
    
    def _is_financial_amount(self, column_name: str, data_type: str, sample_data: List[Any]) -> bool:
        """Check if column represents financial amount"""
        # Column name patterns
        money_patterns = ["amount", "price", "cost", "fee", "charge", "balance", "salary"]
        if any(pattern in column_name.lower() for pattern in money_patterns):
            # Check data type  
            if data_type.lower() in ["decimal", "numeric", "float", "double"]:
                # Check sample data ranges (reasonable monetary values)
                if sample_data:
                    numeric_values = []
                    for value in sample_data:
                        try:
                            numeric_values.append(float(value))
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_values:
                        avg_value = sum(numeric_values) / len(numeric_values)
                        # Reasonable range for monetary values
                        return 0.01 <= avg_value <= 1000000
        return False

# Register custom rule
async def register_custom_rules(ai_classifier):
    """Register custom classification rules"""
    
    # Internal ID rule
    id_rule = ClassificationRule(
        rule_id="custom_internal_id",
        name="Internal ID Detection",
        description="Detects internal identifier columns",
        classification="INTERNAL",
        confidence_score=0.95,
        conditions={
            "column_patterns": ["*_id", "id_*", "*identifier*", "*key*"],
            "data_types": ["integer", "bigint", "uuid"],
            "uniqueness_threshold": 0.95
        }
    )
    
    await ai_classifier.add_classification_rule(id_rule)
    
    # Financial amount rule
    financial_rule = ClassificationRule(
        rule_id="custom_financial_amount",
        name="Financial Amount Detection", 
        description="Detects financial amount columns",
        classification="CONFIDENTIAL",
        confidence_score=0.90,
        conditions={
            "column_patterns": ["*amount*", "*price*", "*cost*", "*fee*", "*salary*"],
            "data_types": ["decimal", "numeric", "float", "double"],
            "value_range": [0.01, 1000000]
        }
    )
    
    await ai_classifier.add_classification_rule(financial_rule)
```

### Custom ML Models

Integrate custom machine learning models:

```python
# capabilities/common/meta/classification/custom_ml_models.py
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom feature extractor for data classification"""
    
    def __init__(self):
        self.feature_names = [
            'column_name_length',
            'has_email_pattern',
            'has_phone_pattern', 
            'has_date_pattern',
            'uniqueness_ratio',
            'null_ratio',
            'numeric_ratio',
            'string_length_avg',
            'contains_id_keyword',
            'contains_name_keyword'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from column metadata"""
        features = []
        
        for item in X:
            column_name = item.get('column_name', '').lower()
            data_type = item.get('data_type', '').lower()
            sample_data = item.get('sample_data', [])
            
            feature_vector = [
                len(column_name),                           # column_name_length
                self._has_email_pattern(sample_data),       # has_email_pattern
                self._has_phone_pattern(sample_data),       # has_phone_pattern
                self._has_date_pattern(sample_data),        # has_date_pattern
                self._calculate_uniqueness(sample_data),    # uniqueness_ratio
                self._calculate_null_ratio(sample_data),    # null_ratio
                self._calculate_numeric_ratio(sample_data), # numeric_ratio
                self._calculate_avg_length(sample_data),    # string_length_avg
                int('id' in column_name),                   # contains_id_keyword
                int(any(kw in column_name for kw in ['name', 'title', 'label']))  # contains_name_keyword
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _has_email_pattern(self, sample_data: List[Any]) -> float:
        """Check if sample data contains email patterns"""
        if not sample_data:
            return 0.0
        
        import re
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        matches = sum(1 for item in sample_data if isinstance(item, str) and email_pattern.search(item))
        return matches / len(sample_data)
    
    def _has_phone_pattern(self, sample_data: List[Any]) -> float:
        """Check if sample data contains phone number patterns"""
        if not sample_data:
            return 0.0
            
        import re
        phone_pattern = re.compile(r'[\+]?[1-9]?[\d\s\-\(\)]{10,}')
        matches = sum(1 for item in sample_data if isinstance(item, str) and phone_pattern.search(item))
        return matches / len(sample_data)
    
    # Additional helper methods...

class CustomMLClassifier:
    """Custom ML-based data classifier"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = CustomFeatureExtractor()
        self.model = None
        self.class_labels = ['PUBLIC', 'INTERNAL', 'CONFIDENTIAL', 'PII', 'SENSITIVE_PII']
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    
    async def classify(self, 
                      column_name: str, 
                      data_type: str, 
                      sample_data: List[Any],
                      context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Classify using custom ML model"""
        
        if not self.model:
            return None
        
        try:
            # Prepare input data
            input_data = [{
                'column_name': column_name,
                'data_type': data_type,
                'sample_data': sample_data[:100],  # Limit sample size
                'context': context
            }]
            
            # Extract features
            features = self.feature_extractor.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            confidence = float(probability.max())
            
            # Map prediction to classification
            classification = self.class_labels[prediction]
            
            return ClassificationResult(
                classification=classification,
                confidence_score=confidence,
                reasoning=f"Custom ML model prediction with {confidence:.2f} confidence",
                method_used="custom_ml_model",
                tags=["ml_classified"]
            )
            
        except Exception as e:
            # Log error and return None for fallback
            return None
    
    def train_model(self, training_data: List[Dict[str, Any]], labels: List[str]):
        """Train the custom model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        
        # Extract features
        features = self.feature_extractor.transform(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        return self.model
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model:
            joblib.dump(self.model, model_path)
```

---

## Database Schema

### PostgreSQL Schema

The primary metadata storage uses PostgreSQL:

```sql
-- Core metadata tables
-- capabilities/common/meta/schema/postgresql_schema.sql

-- Assets table
CREATE TABLE meta_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    asset_type VARCHAR(50) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    database_name VARCHAR(100),
    schema_name VARCHAR(100),
    description TEXT,
    owner VARCHAR(255),
    steward VARCHAR(255),
    tags TEXT[],
    classification VARCHAR(50),
    quality_score DECIMAL(3,2),
    custom_attributes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL,
    
    CONSTRAINT unique_asset_per_tenant UNIQUE (name, source_system, tenant_id)
);

-- Columns table  
CREATE TABLE meta_columns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES meta_assets(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    data_type VARCHAR(100) NOT NULL,
    is_nullable BOOLEAN DEFAULT TRUE,
    is_primary_key BOOLEAN DEFAULT FALSE,
    is_foreign_key BOOLEAN DEFAULT FALSE,
    foreign_key_reference VARCHAR(255),
    classification VARCHAR(50),
    description TEXT,
    business_name VARCHAR(255),
    data_quality JSONB,
    sample_values JSONB,
    position INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL,
    
    CONSTRAINT unique_column_per_asset UNIQUE (asset_id, name)
);

-- Discovery schedules
CREATE TABLE meta_discovery_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    connector_config JSONB NOT NULL,
    schedule_type VARCHAR(20) NOT NULL CHECK (schedule_type IN ('one_time', 'recurring')),
    cron_expression VARCHAR(100),
    interval_minutes INTEGER,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    is_enabled BOOLEAN DEFAULT TRUE,
    is_one_time BOOLEAN DEFAULT FALSE,
    last_run TIMESTAMP WITH TIME ZONE,
    next_run TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL,
    
    CONSTRAINT unique_schedule_per_tenant UNIQUE (name, tenant_id)
);

-- Discovery jobs
CREATE TABLE meta_discovery_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID REFERENCES meta_discovery_schedules(schedule_id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    progress_percentage INTEGER DEFAULT 0,
    current_step VARCHAR(255),
    current_connector VARCHAR(255),
    results JSONB,
    errors JSONB,
    tenant_id VARCHAR(50) NOT NULL
);

-- Classification rules
CREATE TABLE meta_classification_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    conditions JSONB NOT NULL,
    classification VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    is_enabled BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 100,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL,
    
    CONSTRAINT unique_rule_per_tenant UNIQUE (rule_id, tenant_id)
);

-- Indexes for performance
CREATE INDEX idx_assets_tenant_type ON meta_assets(tenant_id, asset_type);
CREATE INDEX idx_assets_source_system ON meta_assets(source_system);
CREATE INDEX idx_assets_classification ON meta_assets(classification);
CREATE INDEX idx_assets_quality_score ON meta_assets(quality_score);
CREATE INDEX idx_assets_updated_at ON meta_assets(updated_at);
CREATE INDEX idx_assets_tags ON meta_assets USING GIN(tags);
CREATE INDEX idx_assets_custom_attributes ON meta_assets USING GIN(custom_attributes);

CREATE INDEX idx_columns_asset_id ON meta_columns(asset_id);
CREATE INDEX idx_columns_data_type ON meta_columns(data_type);
CREATE INDEX idx_columns_classification ON meta_columns(classification);

CREATE INDEX idx_discovery_jobs_status ON meta_discovery_jobs(status);
CREATE INDEX idx_discovery_jobs_started_at ON meta_discovery_jobs(started_at);

-- Row Level Security (RLS) for multi-tenancy
ALTER TABLE meta_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE meta_columns ENABLE ROW LEVEL SECURITY;
ALTER TABLE meta_discovery_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE meta_discovery_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE meta_classification_rules ENABLE ROW LEVEL SECURITY;

-- RLS policies (example - adjust based on your auth system)
CREATE POLICY tenant_isolation_assets ON meta_assets
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant'));

CREATE POLICY tenant_isolation_columns ON meta_columns  
    FOR ALL TO authenticated_users
    USING (tenant_id = current_setting('app.current_tenant'));
```

### Neo4j Schema

Lineage relationships are stored in Neo4j:

```cypher
// Neo4j schema for lineage relationships
// capabilities/common/meta/schema/neo4j_schema.cypher

// Asset nodes
CREATE CONSTRAINT asset_id_unique IF NOT EXISTS 
FOR (a:Asset) REQUIRE a.id IS UNIQUE;

CREATE INDEX asset_tenant_type IF NOT EXISTS
FOR (a:Asset) ON (a.tenant_id, a.asset_type);

// Lineage relationship types
CREATE (:RelationshipType {
    name: 'TRANSFORMS',
    description: 'Data transformation relationship'
});

CREATE (:RelationshipType {
    name: 'DERIVES_FROM', 
    description: 'Data derivation relationship'
});

CREATE (:RelationshipType {
    name: 'COPIES_TO',
    description: 'Direct data copy relationship'
});

CREATE (:RelationshipType {
    name: 'JOINS_WITH',
    description: 'Data join relationship'
});

// Example relationships
// Asset A transforms to Asset B
MATCH (a:Asset {id: 'asset_a_id'}), (b:Asset {id: 'asset_b_id'})
CREATE (a)-[:TRANSFORMS {
    transformation_logic: 'SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id',
    created_at: datetime(),
    tenant_id: 'tenant1'
}]->(b);

// Lineage query examples
// Get downstream assets (what this asset feeds into)
MATCH (a:Asset {id: $asset_id})-[r:TRANSFORMS|DERIVES_FROM|COPIES_TO*1..5]->(downstream)
WHERE a.tenant_id = $tenant_id
RETURN downstream, r;

// Get upstream assets (what feeds into this asset)  
MATCH (upstream)-[r:TRANSFORMS|DERIVES_FROM|COPIES_TO*1..5]->(a:Asset {id: $asset_id})
WHERE a.tenant_id = $tenant_id
RETURN upstream, r;

// Impact analysis - find all assets affected by a change
MATCH (a:Asset {id: $asset_id})-[r:TRANSFORMS|DERIVES_FROM|COPIES_TO*0..10]->(affected)
WHERE a.tenant_id = $tenant_id
RETURN affected.id, affected.name, length(r) as distance
ORDER BY distance;
```

---

## Testing Framework

### Test Structure

```bash
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_models.py
│   ├── test_database_manager.py
│   ├── test_search_engine.py
│   └── test_ai_classifier.py
├── integration/             # Integration tests  
│   ├── test_api_endpoints.py
│   ├── test_discovery_flow.py
│   └── test_lineage_tracking.py
├── performance/             # Performance tests
│   ├── test_search_performance.py
│   └── test_discovery_performance.py
├── fixtures/               # Test fixtures and sample data
│   ├── sample_assets.json
│   └── test_databases.sql
└── conftest.py            # Pytest configuration
```

### Test Configuration

```python
# tests/conftest.py
import asyncio
import pytest
import asyncpg
from typing import AsyncGenerator
from capabilities.common.meta import APGMetadataService
from capabilities.common.meta.database_manager import MetaDatabaseManager

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_database():
    """Setup test database"""
    # Create test database
    conn = await asyncpg.connect("postgresql://localhost/template1")
    await conn.execute("DROP DATABASE IF EXISTS test_apg_metadata")
    await conn.execute("CREATE DATABASE test_apg_metadata")
    await conn.close()
    
    # Setup schema
    test_conn = await asyncpg.connect("postgresql://localhost/test_apg_metadata")
    with open("schema/postgresql_schema.sql") as f:
        await test_conn.execute(f.read())
    await test_conn.close()
    
    yield "postgresql://localhost/test_apg_metadata"
    
    # Cleanup
    conn = await asyncpg.connect("postgresql://localhost/template1")
    await conn.execute("DROP DATABASE test_apg_metadata")
    await conn.close()

@pytest.fixture
async def metadata_service(test_database) -> AsyncGenerator[APGMetadataService, None]:
    """Create metadata service instance for testing"""
    config = {
        "postgresql_url": test_database,
        "redis_url": "redis://localhost:6379/1",  # Use different DB for tests
        "neo4j_url": "bolt://localhost:7687",
        "tenant_id": "test_tenant"
    }
    
    service = APGMetadataService(config)
    await service.initialize()
    
    yield service
    
    await service.cleanup()

@pytest.fixture
def sample_asset_data():
    """Sample asset data for testing"""
    return {
        "name": "test_table",
        "display_name": "Test Table",
        "asset_type": "table",
        "source_system": "test_system",
        "database": "test_db",
        "schema": "public",
        "description": "Test table for unit testing",
        "columns": [
            {
                "name": "id",
                "data_type": "INTEGER",
                "is_nullable": False,
                "is_primary_key": True
            },
            {
                "name": "email",
                "data_type": "VARCHAR",
                "is_nullable": True,
                "classification": "PII"
            }
        ]
    }
```

### Unit Tests

```python
# tests/unit/test_models.py
import pytest
from pydantic import ValidationError
from capabilities.common.meta.models import AssetMetadata, ColumnMetadata

def test_asset_metadata_validation():
    """Test asset metadata model validation"""
    
    # Valid asset
    asset_data = {
        "name": "test_table",
        "asset_type": "table", 
        "source_system": "postgresql"
    }
    asset = AssetMetadata(**asset_data)
    assert asset.name == "test_table"
    assert asset.asset_type == "table"
    
    # Invalid asset type
    with pytest.raises(ValidationError):
        AssetMetadata(
            name="test",
            asset_type="invalid_type",
            source_system="test"
        )
    
    # Missing required fields
    with pytest.raises(ValidationError):
        AssetMetadata(name="test")

def test_column_metadata_validation():
    """Test column metadata validation"""
    
    # Valid column
    column_data = {
        "name": "email",
        "data_type": "VARCHAR"
    }
    column = ColumnMetadata(**column_data)
    assert column.name == "email"
    assert column.is_nullable == False  # Default value
    
    # Test classification validation
    column_with_classification = ColumnMetadata(
        name="ssn",
        data_type="VARCHAR", 
        classification="SENSITIVE_PII"
    )
    assert column_with_classification.classification == "SENSITIVE_PII"

# tests/unit/test_database_manager.py
import pytest
from capabilities.common.meta.database_manager import MetaDatabaseManager

@pytest.mark.asyncio
async def test_database_connection(test_database):
    """Test database connection and basic operations"""
    
    config = {
        "postgresql_url": test_database,
        "redis_url": "redis://localhost:6379/1"
    }
    
    db_manager = MetaDatabaseManager(config)
    await db_manager.initialize()
    
    # Test PostgreSQL connection
    result = await db_manager.execute_query("SELECT 1 as test")
    assert len(result) == 1
    assert result[0]["test"] == 1
    
    # Test Redis connection
    await db_manager.cache_set("test_key", "test_value")
    value = await db_manager.cache_get("test_key")
    assert value == "test_value"
    
    await db_manager.cleanup()

# tests/unit/test_search_engine.py
import pytest
from capabilities.common.meta.search_engine import SearchEngine

@pytest.mark.asyncio
async def test_search_functionality(metadata_service, sample_asset_data):
    """Test search engine functionality"""
    
    # Create test asset
    asset_id = await metadata_service.create_asset(sample_asset_data)
    
    # Wait for indexing
    await asyncio.sleep(0.5)
    
    # Test text search
    results = await metadata_service.search_assets("test table")
    assert len(results["results"]) > 0
    assert results["results"][0]["name"] == "test_table"
    
    # Test filtered search
    results = await metadata_service.search_assets(
        query="table",
        filters={"asset_type": "table"}
    )
    assert len(results["results"]) > 0
    
    # Test empty results
    results = await metadata_service.search_assets("nonexistent_asset")
    assert len(results["results"]) == 0
```

### Integration Tests

```python
# tests/integration/test_discovery_flow.py
import pytest
from capabilities.common.meta.connectors import create_connector, ConnectorConfig

@pytest.mark.asyncio
async def test_discovery_end_to_end(metadata_service):
    """Test complete discovery workflow"""
    
    # Setup test data source (mock)
    connector_config = ConnectorConfig(
        connector_type="postgresql",
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_pass"
    )
    
    # Create discovery schedule
    schedule_data = {
        "name": "Test Discovery",
        "description": "Integration test discovery",
        "connector_config": connector_config,
        "schedule_type": "one_time",
        "is_enabled": True
    }
    
    schedule_id = await metadata_service.create_discovery_schedule(schedule_data)
    assert schedule_id is not None
    
    # Run discovery job  
    job_id = await metadata_service.run_discovery_job(schedule_id)
    assert job_id is not None
    
    # Wait for completion
    import time
    for _ in range(30):  # Wait up to 30 seconds
        job_status = await metadata_service.get_discovery_job_status(job_id)
        if job_status["status"] in ["completed", "failed"]:
            break
        time.sleep(1)
    
    # Verify results
    assert job_status["status"] == "completed"
    assert job_status["results"]["assets_discovered"] > 0

# tests/integration/test_api_endpoints.py  
import pytest
from fastapi.testclient import TestClient
from capabilities.common.meta.api import app

@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/metadata/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_assets_crud(client):
    """Test asset CRUD operations"""
    
    # Create asset
    asset_data = {
        "name": "api_test_table",
        "asset_type": "table",
        "source_system": "test_api",
        "description": "Test asset via API"
    }
    
    response = client.post("/api/v1/metadata/assets/", json=asset_data)
    assert response.status_code == 200
    asset_id = response.json()["asset_id"]
    
    # Get asset
    response = client.get(f"/api/v1/metadata/assets/{asset_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "api_test_table"
    
    # Update asset
    update_data = {"description": "Updated description"}
    response = client.put(f"/api/v1/metadata/assets/{asset_id}", json=update_data)
    assert response.status_code == 200
    
    # List assets
    response = client.get("/api/v1/metadata/assets/")
    assert response.status_code == 200
    assert len(response.json()["assets"]) > 0
    
    # Delete asset
    response = client.delete(f"/api/v1/metadata/assets/{asset_id}")
    assert response.status_code == 204
```

### Performance Tests

```python
# tests/performance/test_search_performance.py
import pytest
import time
import asyncio
from capabilities.common.meta.models import AssetMetadata

@pytest.mark.asyncio
async def test_search_performance(metadata_service):
    """Test search performance with large dataset"""
    
    # Create test dataset
    assets_to_create = 1000
    batch_size = 100
    
    start_time = time.time()
    
    for batch in range(0, assets_to_create, batch_size):
        tasks = []
        for i in range(batch, min(batch + batch_size, assets_to_create)):
            asset_data = {
                "name": f"performance_test_table_{i}",
                "asset_type": "table",
                "source_system": "performance_test",
                "description": f"Performance test table number {i}",
                "tags": [f"batch_{batch//batch_size}", "performance", "test"]
            }
            tasks.append(metadata_service.create_asset(asset_data))
        
        await asyncio.gather(*tasks)
    
    creation_time = time.time() - start_time
    print(f"Created {assets_to_create} assets in {creation_time:.2f} seconds")
    
    # Wait for indexing
    await asyncio.sleep(2)
    
    # Test search performance
    search_times = []
    queries = [
        "performance test",
        "table",
        "batch_5",
        "nonexistent query"
    ]
    
    for query in queries:
        start_time = time.time()
        results = await metadata_service.search_assets(query, limit=50)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        print(f"Query '{query}': {len(results['results'])} results in {search_time:.3f}s")
    
    # Performance assertions
    avg_search_time = sum(search_times) / len(search_times)
    assert avg_search_time < 0.5  # Average search time should be under 500ms
    assert max(search_times) < 1.0  # No search should take more than 1 second

@pytest.mark.asyncio
async def test_discovery_performance():
    """Test discovery job performance"""
    
    # This would test discovery performance with real data sources
    # Implementation depends on your test environment setup
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=capabilities.common.meta --cov-report=html --cov-report=term

# Run performance tests only
pytest tests/performance/ -v

# Run tests in parallel
pytest -n auto

# Run with verbose output  
pytest -v -s
```

---

## Deployment Guide

### Docker Deployment

**Dockerfile:**
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY capabilities/ capabilities/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/metadata/health || exit 1

# Start command
CMD ["uvicorn", "capabilities.common.meta.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Database services
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: apg_metadata
      POSTGRES_USER: apg_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./schema/postgresql_schema.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U apg_user -d apg_metadata"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/secure_password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - ./schema/neo4j_schema.cypher:/var/lib/neo4j/import/schema.cypher
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "secure_password", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Application services  
  apg-metadata:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://apg_user:secure_password@postgres:5432/apg_metadata
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=secure_password
      - ENVIRONMENT=production
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      neo4j:
        condition: service_healthy
    volumes:
      - app_logs:/app/logs
    restart: unless-stopped

  # Optional: Web interface
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/private
    depends_on:
      - apg-metadata
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  app_logs:
```

### Kubernetes Deployment

**Namespace and ConfigMap:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: apg-metadata

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: apg-metadata-config
  namespace: apg-metadata
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"
  NEO4J_URL: "bolt://neo4j-service:7687"
```

**Database Deployments:**
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: apg-metadata
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: "apg_metadata"
        - name: POSTGRES_USER
          value: "apg_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: init-script
          mountPath: /docker-entrypoint-initdb.d
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
      - name: init-script
        configMap:
          name: postgres-init

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: apg-metadata
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

**Application Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apg-metadata
  namespace: apg-metadata
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apg-metadata
  template:
    metadata:
      labels:
        app: apg-metadata
    spec:
      containers:
      - name: apg-metadata
        image: your-registry/apg-metadata:latest
        ports:
        - containerPort: 8000
        env:
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-url
        envFrom:
        - configMapRef:
            name: apg-metadata-config
        livenessProbe:
          httpGet:
            path: /api/v1/metadata/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/metadata/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

---
apiVersion: v1
kind: Service
metadata:
  name: apg-metadata-service
  namespace: apg-metadata
spec:
  selector:
    app: apg-metadata
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Configuration

**Production Settings:**
```yaml
# config/production.yaml
database:
  postgresql:
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    pool_recycle: 3600
  
  redis:
    connection_pool_size: 50
    socket_timeout: 5
    
  neo4j:
    max_connection_lifetime: 300
    max_connection_pool_size: 100

search:
  enable_caching: true
  cache_ttl: 3600
  max_results: 10000

discovery:
  max_concurrent_jobs: 10
  job_timeout_minutes: 120
  retry_attempts: 3

classification:
  enable_ml_models: true
  confidence_threshold: 0.7
  batch_size: 1000

security:
  enable_jwt_auth: true
  jwt_secret_key: ${JWT_SECRET_KEY}
  cors_origins: 
    - "https://metadata.company.com"
    - "https://app.company.com"

logging:
  level: INFO
  format: json
  handlers:
    - console
    - file
  file_path: "/app/logs/apg-metadata.log"
  
monitoring:
  enable_metrics: true
  metrics_endpoint: "/metrics"
  health_check_interval: 30
```

### Monitoring and Observability

**Prometheus Configuration:**
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'apg-metadata'
    static_configs:
      - targets: ['apg-metadata-service:80']
    metrics_path: '/api/v1/metadata/metrics'
    scrape_interval: 30s
```

**Grafana Dashboard Configuration:**
```json
{
  "dashboard": {
    "title": "APG Metadata Management",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Discovery Job Status", 
        "type": "stat",
        "targets": [
          {
            "expr": "discovery_jobs_total",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Classification Accuracy",
        "type": "gauge", 
        "targets": [
          {
            "expr": "classification_accuracy",
            "legendFormat": "Accuracy"
          }
        ]
      }
    ]
  }
}
```

---

*This completes the comprehensive Developer Guide for APG Metadata Management. The system is now fully documented with complete implementation details, testing frameworks, and deployment procedures.*