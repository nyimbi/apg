# Enterprise Asset Management - Developer Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Code Organization](#code-organization)
4. [Data Models and Database](#data-models-and-database)
5. [Service Layer Development](#service-layer-development)
6. [API Development](#api-development)
7. [UI Development](#ui-development)
8. [Testing Framework](#testing-framework)
9. [APG Platform Integration](#apg-platform-integration)
10. [Deployment and DevOps](#deployment-and-devops)
11. [Contributing Guidelines](#contributing-guidelines)
12. [Advanced Topics](#advanced-topics)

## Development Environment Setup

### Prerequisites

#### System Requirements
- **Python**: 3.12+ with asyncio support
- **Database**: PostgreSQL 14+ with asyncio drivers
- **Cache**: Redis 6+ for session management
- **Node.js**: 18+ for frontend tooling
- **Docker**: 20+ for containerized development

#### Development Tools
- **IDE**: VS Code with Python extension
- **Database Tool**: pgAdmin or DataGrip
- **API Testing**: Postman or HTTPie
- **Version Control**: Git with pre-commit hooks

### Local Development Setup

#### 1. Clone Repository
```bash
git clone https://github.com/datacraft/apg.git
cd apg/capabilities/general_cross_functional/enterprise_asset_management
```

#### 2. Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### 3. Database Setup
```bash
# Start PostgreSQL with Docker
docker run --name eam-postgres \
  -e POSTGRES_DB=eam_dev \
  -e POSTGRES_USER=eam_dev \
  -e POSTGRES_PASSWORD=dev_password \
  -p 5432:5432 -d postgres:15

# Run migrations
alembic upgrade head
```

#### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
export DATABASE_URL="postgresql://eam_dev:dev_password@localhost:5432/eam_dev"
export REDIS_URL="redis://localhost:6379/0"
export APG_AUTH_SERVICE_URL="http://localhost:8081"
export DEBUG=true
```

#### 5. Start Development Server
```bash
# Start API server
uvicorn api:app --reload --port 8000

# Start background workers (separate terminal)
python -m worker.scheduler
```

### Development Tools Configuration

#### VS Code Settings
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.insertSpaces": false,
    "editor.tabSize": 4
}
```

#### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        args: [--line-length=120]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=120]
```

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        APG Platform                        │
├─────────────────────────────────────────────────────────────┤
│  Auth RBAC  │  Audit     │  Notification  │  Composition   │
│  Service    │  Service   │  Engine        │  Engine        │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    EAM Capability                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   UI Layer      │   API Layer     │    Service Layer       │
│                 │                 │                        │
│ Flask-AppBuilder│  FastAPI        │  Business Logic        │
│ Pydantic Models │  WebSocket      │  Cross-Capability      │
│ Responsive UI   │  Authentication │  Integration           │
└─────────────────┴─────────────────┴─────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                               │
├─────────────────┬─────────────────┬─────────────────────────┤
│  PostgreSQL     │     Redis       │    File Storage        │
│  Multi-tenant   │     Cache       │    Documents           │
│  ACID Compliance│     Sessions    │    Images              │
└─────────────────┴─────────────────┴─────────────────────────┘
```

### Design Patterns

#### 1. Repository Pattern
```python
# Abstract repository interface
class BaseRepository:
	async def create(self, entity: Model) -> Model:
		pass
	
	async def get_by_id(self, entity_id: str) -> Model | None:
		pass
	
	async def update(self, entity: Model) -> Model:
		pass
	
	async def delete(self, entity_id: str) -> bool:
		pass

# Concrete implementation
class AssetRepository(BaseRepository):
	def __init__(self, session: AsyncSession):
		self.session = session
	
	async def create(self, asset: EAAsset) -> EAAsset:
		self.session.add(asset)
		await self.session.commit()
		await self.session.refresh(asset)
		return asset
```

#### 2. Service Layer Pattern
```python
class EAMAssetService:
	def __init__(self, 
		asset_repo: AssetRepository,
		auth_service: AuthService,
		audit_service: AuditService
	):
		self.asset_repo = asset_repo
		self.auth_service = auth_service
		self.audit_service = audit_service
	
	async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
		# Validate permissions
		await self.auth_service.check_permission("eam.asset.create")
		
		# Business logic
		asset = EAAsset(**asset_data)
		asset = await self.asset_repo.create(asset)
		
		# Audit logging
		await self.audit_service.log_action("asset_created", asset.asset_id)
		
		return asset
```

#### 3. Dependency Injection
```python
# Dependency container
class Container:
	def __init__(self):
		self._services = {}
	
	def register(self, interface: Type, implementation: Type):
		self._services[interface] = implementation
	
	def get(self, interface: Type):
		return self._services[interface]()

# Service registration
container = Container()
container.register(AssetRepository, AssetRepository)
container.register(EAMAssetService, EAMAssetService)
```

### Multi-Tenant Architecture

#### Tenant Isolation Strategy
```python
# Tenant context middleware
@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
	tenant_id = request.headers.get("X-Tenant-ID")
	if not tenant_id:
		tenant_id = await extract_tenant_from_jwt(request)
	
	# Set tenant context
	set_tenant_context(tenant_id)
	
	response = await call_next(request)
	return response

# Row-level security
class TenantMixin:
	tenant_id = Column(String(36), nullable=False, index=True)
	
	@classmethod
	def query_for_tenant(cls, tenant_id: str):
		return select(cls).where(cls.tenant_id == tenant_id)
```

## Code Organization

### Directory Structure

```
enterprise_asset_management/
├── __init__.py                 # Capability metadata
├── api.py                      # REST API endpoints
├── blueprint.py                # Flask integration
├── models.py                   # Data models
├── service.py                  # Business logic
├── views.py                    # UI views
├── cap_spec.md                 # Capability specification
├── todo.md                     # Development tasks
├── docs/                       # Documentation
│   ├── README.md               # Main documentation
│   ├── user_guide.md           # User documentation
│   ├── developer_guide.md      # This file
│   ├── API_REFERENCE.md        # API documentation
│   └── DEPLOYMENT.md           # Deployment guide
├── tests/                      # Test suite
│   ├── test_models.py          # Model tests
│   ├── test_services.py        # Service tests
│   ├── test_api.py             # API tests
│   └── test_integration.py     # Integration tests
└── migrations/                 # Database migrations
    ├── 001_initial_schema.py
    ├── 002_add_performance.py
    └── versions/
```

### Code Organization Principles

#### 1. Separation of Concerns
- **Models**: Data structure and relationships only
- **Services**: Business logic and validation
- **API**: HTTP handling and serialization
- **Views**: UI presentation logic

#### 2. Dependency Direction
```
UI/API → Services → Repositories → Models
```

#### 3. APG Integration Points
- **Authentication**: Via auth_rbac service
- **Auditing**: Via audit_compliance service
- **Notifications**: Via notification_engine
- **Composition**: Via composition_engine

### Coding Standards

#### Python Standards (CLAUDE.md)
```python
# Use async throughout
async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
	pass

# Use tabs for indentation (not spaces)
class EAAsset(Model):
	def __init__(self):
		if condition:
			do_something()

# Modern typing
def process_assets(assets: list[EAAsset]) -> dict[str, Any]:
	return {"count": len(assets)}

# UUID generation
from uuid_extensions import uuid7str
asset_id: str = Field(default_factory=uuid7str)
```

#### Naming Conventions
- **Classes**: PascalCase (`EAAsset`, `WorkOrderService`)
- **Functions/Variables**: snake_case (`create_asset`, `work_order_id`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ASSETS_PER_TENANT`)
- **Files**: snake_case (`enterprise_asset_management`)

#### Documentation Standards
```python
class EAMAssetService:
	"""
	Service for managing enterprise assets with APG integration.
	
	Provides comprehensive asset lifecycle management including:
	- Asset creation and validation
	- Health monitoring and scoring
	- Maintenance scheduling integration
	- Performance analytics
	
	APG Integration:
	- auth_rbac: Permission validation
	- audit_compliance: Action logging
	- predictive_maintenance: Health analysis
	"""
	
	async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
		"""
		Create new asset with validation and audit logging.
		
		Args:
			asset_data: Asset information including name, type, location
		
		Returns:
			Created asset with generated ID and timestamps
		
		Raises:
			PermissionError: If user lacks asset creation permission
			ValidationError: If asset data is invalid
		
		Example:
			asset_data = {
				"asset_name": "CNC Machine #1",
				"asset_type": "equipment",
				"location_id": "loc_123"
			}
			asset = await service.create_asset(asset_data)
		"""
		assert asset_data is not None, "Asset data is required"
		assert "asset_name" in asset_data, "Asset name is required"
		
		# Implementation here
		pass
```

## Data Models and Database

### Model Design Principles

#### 1. Base Model Pattern
```python
from uuid_extensions import uuid7str
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()

class BaseMixin:
	"""Base mixin for all models with common fields"""
	id = Column(String(36), primary_key=True, default=uuid7str)
	created_on = Column(DateTime(timezone=True), default=datetime.utcnow)
	changed_on = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class AuditMixin:
	"""Audit trail mixin for change tracking"""
	created_by = Column(String(100))
	changed_by = Column(String(100))
	version = Column(Integer, default=1)

class TenantMixin:
	"""Multi-tenant isolation mixin"""
	tenant_id = Column(String(36), nullable=False, index=True)
```

#### 2. Model Relationships
```python
class EAAsset(Model, BaseMixin, AuditMixin, TenantMixin):
	# Primary fields
	asset_id = Column(String(36), unique=True, nullable=False, default=uuid7str)
	asset_name = Column(String(200), nullable=False)
	
	# Foreign key relationships
	location_id = Column(String(36), ForeignKey('ea_location.location_id'))
	parent_asset_id = Column(String(36), ForeignKey('ea_asset.asset_id'))
	
	# Relationships
	location = relationship("EALocation", back_populates="assets")
	parent_asset = relationship("EAAsset", remote_side=[asset_id])
	child_assets = relationship("EAAsset")
	work_orders = relationship("EAWorkOrder", back_populates="asset")
	maintenance_records = relationship("EAMaintenanceRecord", back_populates="asset")
```

#### 3. Database Constraints
```python
# Unique constraints
__table_args__ = (
	UniqueConstraint('tenant_id', 'asset_number', name='uq_asset_tenant_number'),
	CheckConstraint('health_score >= 0 AND health_score <= 100', name='ck_health_score'),
	Index('idx_asset_tenant_type', 'tenant_id', 'asset_type'),
	Index('idx_asset_health_score', 'health_score'),
)
```

### Database Migrations

#### Migration Strategy
```python
# migrations/env.py
from alembic import context
from sqlalchemy import engine_from_config, pool
from models import Base

# Multi-tenant migration support
def run_migrations_online():
	connectable = engine_from_config(
		config.get_section(config.config_ini_section),
		prefix="sqlalchemy.",
		poolclass=pool.NullPool,
	)
	
	with connectable.connect() as connection:
		context.configure(
			connection=connection, 
			target_metadata=Base.metadata,
			include_schemas=True  # Support for schema per tenant
		)
		
		with context.begin_transaction():
			context.run_migrations()
```

#### Sample Migration
```python
# migrations/versions/001_initial_schema.py
"""Initial EAM schema

Revision ID: 001
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
	# Create ea_location table
	op.create_table('ea_location',
		sa.Column('location_id', sa.String(36), nullable=False),
		sa.Column('tenant_id', sa.String(36), nullable=False),
		sa.Column('location_name', sa.String(200), nullable=False),
		sa.Column('location_type', sa.String(50), nullable=False),
		sa.Column('created_on', sa.DateTime(timezone=True), nullable=False),
		sa.PrimaryKeyConstraint('location_id')
	)
	
	# Create indexes
	op.create_index('idx_location_tenant', 'ea_location', ['tenant_id'])

def downgrade():
	op.drop_table('ea_location')
```

### Performance Optimization

#### Query Optimization
```python
# Efficient queries with proper joins
async def get_assets_with_location(tenant_id: str) -> List[EAAsset]:
	query = (
		select(EAAsset)
		.options(selectinload(EAAsset.location))  # Eager loading
		.where(EAAsset.tenant_id == tenant_id)
		.order_by(EAAsset.asset_number)
	)
	result = await session.execute(query)
	return result.scalars().all()

# Pagination for large datasets
async def get_assets_paginated(
	tenant_id: str, 
	page: int = 1, 
	limit: int = 25
) -> Tuple[List[EAAsset], int]:
	
	offset = (page - 1) * limit
	
	# Count query
	count_query = select(func.count(EAAsset.asset_id)).where(
		EAAsset.tenant_id == tenant_id
	)
	total = await session.scalar(count_query)
	
	# Data query
	data_query = (
		select(EAAsset)
		.where(EAAsset.tenant_id == tenant_id)
		.offset(offset)
		.limit(limit)
	)
	result = await session.execute(data_query)
	assets = result.scalars().all()
	
	return assets, total
```

#### Database Indexing Strategy
```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_ea_asset_tenant_type 
ON ea_asset(tenant_id, asset_type);

CREATE INDEX CONCURRENTLY idx_ea_asset_health_score 
ON ea_asset(health_score) 
WHERE health_score < 80;

CREATE INDEX CONCURRENTLY idx_ea_workorder_status_priority 
ON ea_work_order(status, priority);

-- Partial indexes for performance
CREATE INDEX CONCURRENTLY idx_ea_asset_active 
ON ea_asset(tenant_id, status) 
WHERE status = 'active';
```

## Service Layer Development

### Service Architecture

#### Service Base Class
```python
class BaseService:
	"""Base service with common functionality"""
	
	def __init__(self, session: AsyncSession):
		self.session = session
		self.auth_service = get_auth_service()
		self.audit_service = get_audit_service()
	
	async def _check_permission(self, permission: str, tenant_id: str):
		"""Check user permissions"""
		if not await self.auth_service.check_permission(permission, tenant_id):
			raise PermissionError(f"Permission denied: {permission}")
	
	async def _log_action(self, action: str, entity_id: str, details: Dict[str, Any] = None):
		"""Log audit trail"""
		await self.audit_service.log_action(
			action=action,
			entity_id=entity_id,
			details=details
		)
```

#### Service Implementation
```python
class EAMAssetService(BaseService):
	"""Asset management service with business logic"""
	
	async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
		"""Create asset with validation and audit"""
		
		# Runtime assertions
		assert asset_data is not None, "Asset data is required"
		assert "tenant_id" in asset_data, "Tenant ID is required"
		assert "asset_name" in asset_data, "Asset name is required"
		
		tenant_id = asset_data["tenant_id"]
		
		# Permission check
		await self._check_permission("eam.asset.create", tenant_id)
		
		# Business validation
		await self._validate_asset_data(asset_data)
		
		# Check for duplicates
		if "asset_number" in asset_data:
			existing = await self._get_asset_by_number(
				asset_data["asset_number"], 
				tenant_id
			)
			if existing:
				raise ValueError("Asset number already exists")
		
		# Create asset
		asset = EAAsset(**asset_data)
		if not asset.asset_number:
			asset.asset_number = await self._generate_asset_number(tenant_id)
		
		self.session.add(asset)
		await self.session.commit()
		await self.session.refresh(asset)
		
		# Audit logging
		await self._log_action("asset_created", asset.asset_id, {
			"asset_name": asset.asset_name,
			"asset_type": asset.asset_type
		})
		
		# Cross-capability integration
		await self._sync_with_fixed_asset_management(asset)
		await self._register_with_predictive_maintenance(asset)
		
		return asset
	
	async def _validate_asset_data(self, asset_data: Dict[str, Any]):
		"""Validate asset data with business rules"""
		
		# Required fields validation
		required_fields = ["asset_name", "asset_type", "tenant_id"]
		for field in required_fields:
			if not asset_data.get(field):
				raise ValueError(f"{field} is required")
		
		# Business rule validation
		if asset_data.get("purchase_cost") and asset_data["purchase_cost"] < 0:
			raise ValueError("Purchase cost must be positive")
		
		if asset_data.get("health_score"):
			if not 0 <= asset_data["health_score"] <= 100:
				raise ValueError("Health score must be between 0 and 100")
		
		# Location validation
		if asset_data.get("location_id"):
			location = await self._get_location(asset_data["location_id"])
			if not location:
				raise ValueError("Invalid location specified")
```

### Cross-Capability Integration

#### APG Service Integration
```python
class APGServiceIntegration:
	"""Integration with other APG capabilities"""
	
	def __init__(self):
		self.fixed_asset_service = get_service("fixed_asset_management")
		self.predictive_service = get_service("predictive_maintenance")
		self.notification_service = get_service("notification_engine")
	
	async def sync_with_fixed_asset_management(self, asset: EAAsset):
		"""Sync asset with financial management"""
		
		if asset.is_capitalized and asset.purchase_cost:
			fixed_asset_data = {
				"asset_reference_id": asset.asset_id,
				"description": asset.asset_name,
				"cost": asset.purchase_cost,
				"acquisition_date": asset.installation_date,
				"useful_life": asset.expected_useful_life,
				"depreciation_method": "straight_line"
			}
			
			await self.fixed_asset_service.create_fixed_asset(fixed_asset_data)
	
	async def register_with_predictive_maintenance(self, asset: EAAsset):
		"""Register asset for predictive analytics"""
		
		if asset.maintenance_strategy == "predictive":
			pm_registration = {
				"asset_id": asset.asset_id,
				"asset_type": asset.asset_type,
				"criticality": asset.criticality_level,
				"monitoring_parameters": asset.monitoring_parameters
			}
			
			await self.predictive_service.register_asset(pm_registration)
```

### Error Handling and Resilience

#### Service Error Handling
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class EAMAssetService(BaseService):
	
	@retry(
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=4, max=10)
	)
	async def update_asset_health(self, asset_id: str, health_data: Dict[str, Any]) -> EAAsset:
		"""Update asset health with retry logic"""
		
		try:
			asset = await self._get_asset(asset_id)
			if not asset:
				raise ValueError("Asset not found")
			
			# Update health metrics
			asset.health_score = health_data.get("health_score", asset.health_score)
			asset.condition_status = health_data.get("condition_status", asset.condition_status)
			asset.last_assessment_date = datetime.utcnow()
			
			await self.session.commit()
			
			# Trigger alerts if health is critical
			if asset.health_score < 60:
				await self._trigger_health_alert(asset)
			
			return asset
			
		except Exception as e:
			await self.session.rollback()
			logger.error(f"Failed to update asset health: {e}")
			raise
	
	async def _trigger_health_alert(self, asset: EAAsset):
		"""Trigger health alert through notification service"""
		
		alert_data = {
			"type": "asset_health_critical",
			"severity": "high",
			"asset_id": asset.asset_id,
			"asset_name": asset.asset_name,
			"health_score": asset.health_score,
			"message": f"Asset {asset.asset_name} health score is critical: {asset.health_score}%"
		}
		
		await self.notification_service.send_alert(alert_data)
```

## API Development

### FastAPI Application Structure

#### API Router Setup
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(
	title="Enterprise Asset Management API",
	description="Comprehensive EAM API with APG integration",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# Middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
from .routers import assets, work_orders, inventory, analytics

app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(work_orders.router, prefix="/api/v1/work-orders", tags=["work-orders"])
app.include_router(inventory.router, prefix="/api/v1/inventory", tags=["inventory"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
```

#### Request/Response Models
```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class AssetCreateRequest(BaseModel):
	"""Asset creation request model"""
	model_config = ConfigDict(extra='forbid')
	
	asset_name: str = Field(..., min_length=1, max_length=200)
	asset_type: str = Field(..., description="Type of asset")
	asset_category: str = Field(..., description="Asset category")
	description: Optional[str] = Field(None, max_length=1000)
	manufacturer: Optional[str] = Field(None, max_length=100)
	model_number: Optional[str] = Field(None, max_length=100)
	serial_number: Optional[str] = Field(None, max_length=100)
	purchase_cost: Optional[Decimal] = Field(None, ge=0)
	location_id: Optional[str] = Field(None)

class AssetResponse(BaseModel):
	"""Asset response model"""
	model_config = ConfigDict(from_attributes=True)
	
	asset_id: str
	asset_number: str
	asset_name: str
	asset_type: str
	status: str
	health_score: Optional[Decimal]
	created_on: datetime
	changed_on: datetime

class APIResponse(BaseModel):
	"""Standard API response wrapper"""
	message: str
	data: Optional[dict] = None
	errors: Optional[List[str]] = None
	metadata: Optional[dict] = None
```

#### Dependency Injection
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from .auth import get_current_user, verify_permissions

async def get_asset_service(
	session: AsyncSession = Depends(get_session)
) -> EAMAssetService:
	"""Get asset service instance"""
	return EAMAssetService(session)

async def check_permission(permission: str):
	"""Permission checking dependency"""
	async def _check(current_user = Depends(get_current_user)):
		if not await verify_permissions(current_user, permission):
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail=f"Permission required: {permission}"
			)
		return True
	return _check

# Usage in endpoints
@router.post("/", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_asset(
	asset_data: AssetCreateRequest,
	service: EAMAssetService = Depends(get_asset_service),
	_: bool = Depends(check_permission("eam.asset.create"))
) -> APIResponse:
	"""Create new asset"""
	pass
```

#### API Endpoint Implementation
```python
@router.post("/", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_asset(
	asset_data: AssetCreateRequest,
	service: EAMAssetService = Depends(get_asset_service),
	current_user = Depends(get_current_user),
	_: bool = Depends(check_permission("eam.asset.create"))
) -> APIResponse:
	"""
	Create new asset with validation and audit logging.
	
	Requires permission: eam.asset.create
	"""
	try:
		# Add tenant context
		asset_dict = asset_data.model_dump()
		asset_dict["tenant_id"] = current_user.tenant_id
		asset_dict["created_by"] = current_user.user_id
		
		# Create asset
		asset = await service.create_asset(asset_dict)
		
		return APIResponse(
			message="Asset created successfully",
			data={
				"asset_id": asset.asset_id,
				"asset_number": asset.asset_number,
				"status": asset.status
			}
		)
		
	except ValueError as e:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(e)
		)
	except PermissionError as e:
		raise HTTPException(
			status_code=status.HTTP_403_FORBIDDEN,
			detail=str(e)
		)
	except Exception as e:
		logger.error(f"Asset creation failed: {e}")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Internal server error"
		)

@router.get("/", response_model=List[AssetResponse])
async def list_assets(
	page: int = Query(1, ge=1),
	limit: int = Query(25, ge=1, le=100),
	asset_type: Optional[str] = Query(None),
	status: Optional[str] = Query(None),
	search: Optional[str] = Query(None),
	service: EAMAssetService = Depends(get_asset_service),
	current_user = Depends(get_current_user)
) -> List[AssetResponse]:
	"""
	List assets with filtering and pagination.
	
	Query Parameters:
	- page: Page number (default: 1)
	- limit: Items per page (default: 25, max: 100)
	- asset_type: Filter by asset type
	- status: Filter by status
	- search: Text search in name/description
	"""
	
	filters = {}
	if asset_type:
		filters["asset_type"] = asset_type
	if status:
		filters["status"] = status
	if search:
		filters["search"] = search
	
	assets = await service.search_assets(
		tenant_id=current_user.tenant_id,
		filters=filters,
		page=page,
		limit=limit
	)
	
	return [AssetResponse.model_validate(asset) for asset in assets]
```

### WebSocket Implementation

#### Real-time Updates
```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json

class ConnectionManager:
	"""Manage WebSocket connections"""
	
	def __init__(self):
		self.active_connections: Dict[str, Set[WebSocket]] = {}
	
	async def connect(self, websocket: WebSocket, tenant_id: str):
		await websocket.accept()
		if tenant_id not in self.active_connections:
			self.active_connections[tenant_id] = set()
		self.active_connections[tenant_id].add(websocket)
	
	def disconnect(self, websocket: WebSocket, tenant_id: str):
		if tenant_id in self.active_connections:
			self.active_connections[tenant_id].discard(websocket)
	
	async def broadcast_to_tenant(self, message: dict, tenant_id: str):
		if tenant_id in self.active_connections:
			for connection in self.active_connections[tenant_id].copy():
				try:
					await connection.send_text(json.dumps(message))
				except:
					self.active_connections[tenant_id].discard(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await manager.connect(websocket, tenant_id)
	try:
		while True:
			data = await websocket.receive_text()
			message = json.loads(data)
			
			if message.get("type") == "subscribe":
				# Handle subscriptions
				await handle_subscription(websocket, message)
			
	except WebSocketDisconnect:
		manager.disconnect(websocket, tenant_id)

async def notify_asset_update(asset: EAAsset):
	"""Notify clients of asset updates"""
	message = {
		"type": "asset.updated",
		"data": {
			"asset_id": asset.asset_id,
			"health_score": float(asset.health_score),
			"status": asset.status,
			"timestamp": datetime.utcnow().isoformat()
		}
	}
	await manager.broadcast_to_tenant(message, asset.tenant_id)
```

### API Security

#### JWT Authentication
```python
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

class AuthService:
	def __init__(self):
		self.secret_key = settings.JWT_SECRET_KEY
		self.algorithm = "HS256"
		self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
	
	async def verify_token(self, token: str) -> dict:
		"""Verify JWT token and return user data"""
		try:
			payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
			user_id = payload.get("sub")
			tenant_id = payload.get("tenant_id")
			
			if not user_id or not tenant_id:
				raise HTTPException(
					status_code=status.HTTP_401_UNAUTHORIZED,
					detail="Invalid token"
				)
			
			return {
				"user_id": user_id,
				"tenant_id": tenant_id,
				"permissions": payload.get("permissions", [])
			}
			
		except JWTError:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="Invalid token"
			)

async def get_current_user(
	token: str = Depends(oauth2_scheme),
	auth_service: AuthService = Depends(get_auth_service)
) -> dict:
	"""Get current user from JWT token"""
	return await auth_service.verify_token(token)
```

#### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/")
@limiter.limit("10/minute")
async def create_asset(
	request: Request,
	asset_data: AssetCreateRequest
):
	"""Rate limited asset creation"""
	pass
```

## UI Development

### Flask-AppBuilder Integration

#### View Configuration
```python
from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget

class EAAssetModelView(ModelView):
	"""Asset management view with APG integration"""
	
	datamodel = SQLAInterface(EAAsset)
	
	# List view configuration
	list_columns = [
		'asset_number', 'asset_name', 'asset_type', 'status',
		'health_score', 'location.location_name', 'last_maintenance_date'
	]
	
	search_columns = [
		'asset_number', 'asset_name', 'description', 
		'manufacturer', 'model_number', 'serial_number'
	]
	
	# Form configuration
	add_columns = [
		'asset_name', 'asset_type', 'asset_category', 'description',
		'manufacturer', 'model_number', 'serial_number',
		'location_id', 'purchase_cost', 'installation_date'
	]
	
	edit_columns = add_columns + ['status', 'health_score', 'condition_status']
	
	# Custom formatters
	formatters_columns = {
		'health_score': lambda x: f"{x:.1f}%" if x else "N/A",
		'purchase_cost': lambda x: f"${x:,.2f}" if x else "N/A",
	}
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Multi-tenant filtering
	base_filters = [['tenant_id', lambda: g.user.tenant_id]]
	
	@expose('/health_dashboard')
	@has_access
	def health_dashboard(self):
		"""Custom health dashboard view"""
		return self.render_template('eam/asset_health_dashboard.html')
```

#### Custom Widgets and Forms
```python
from wtforms import SelectField, DecimalField, DateField
from wtforms.validators import DataRequired, NumberRange
from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
from flask_appbuilder.forms import DynamicForm

class AssetHealthWidget(BS3TextFieldWidget):
	"""Custom widget for asset health display"""
	
	def __call__(self, field, **kwargs):
		kwargs.setdefault('id', field.id)
		kwargs.setdefault('type', 'number')
		kwargs.setdefault('min', 0)
		kwargs.setdefault('max', 100)
		kwargs.setdefault('step', 0.1)
		
		if field.data:
			# Add CSS class based on health score
			if field.data >= 90:
				kwargs.setdefault('class', 'form-control health-excellent')
			elif field.data >= 70:
				kwargs.setdefault('class', 'form-control health-good')
			elif field.data >= 50:
				kwargs.setdefault('class', 'form-control health-fair')
			else:
				kwargs.setdefault('class', 'form-control health-poor')
		
		return super().__call__(field, **kwargs)

class AssetForm(DynamicForm):
	"""Custom asset form with validation"""
	
	asset_name = StringField('Asset Name', 
		validators=[DataRequired()], 
		widget=BS3TextFieldWidget()
	)
	
	health_score = DecimalField('Health Score (%)',
		validators=[NumberRange(min=0, max=100)],
		widget=AssetHealthWidget()
	)
	
	criticality_level = SelectField('Criticality Level',
		choices=[
			('low', 'Low'),
			('medium', 'Medium'), 
			('high', 'High'),
			('critical', 'Critical')
		],
		default='medium'
	)
```

#### Responsive Templates
```html
<!-- templates/eam/asset_health_dashboard.html -->
{% extends "appbuilder/general/widgets/search.html" %}

{% block content %}
<div class="container-fluid">
	<div class="row">
		<div class="col-lg-3 col-md-6 col-sm-12">
			<div class="card health-card">
				<div class="card-header">
					<h5>Asset Health Overview</h5>
				</div>
				<div class="card-body">
					<canvas id="healthChart"></canvas>
				</div>
			</div>
		</div>
		
		<div class="col-lg-9 col-md-6 col-sm-12">
			<div class="card">
				<div class="card-header">
					<h5>Critical Assets Requiring Attention</h5>
				</div>
				<div class="card-body">
					<div class="table-responsive">
						<table class="table table-striped" id="criticalAssetsTable">
							<thead>
								<tr>
									<th>Asset Number</th>
									<th>Asset Name</th>
									<th>Health Score</th>
									<th>Last Maintenance</th>
									<th>Actions</th>
								</tr>
							</thead>
							<tbody id="criticalAssetsBody">
								<!-- Dynamic content loaded via JavaScript -->
							</tbody>
						</table>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Real-time updates via WebSocket -->
<script>
document.addEventListener('DOMContentLoaded', function() {
	// Initialize WebSocket connection
	const ws = new WebSocket('wss://{{ request.host }}/ws');
	
	ws.onopen = function() {
		// Subscribe to asset health updates
		ws.send(JSON.stringify({
			type: 'subscribe',
			channels: ['asset.health.*']
		}));
	};
	
	ws.onmessage = function(event) {
		const data = JSON.parse(event.data);
		if (data.type === 'asset.health.update') {
			updateAssetHealthDisplay(data.payload);
		}
	};
	
	// Load initial dashboard data
	loadHealthDashboard();
});

function updateAssetHealthDisplay(assetData) {
	// Update health chart and critical assets table
	const row = document.querySelector(`#asset-${assetData.asset_id}`);
	if (row) {
		row.querySelector('.health-score').textContent = assetData.health_score + '%';
		row.className = getHealthScoreClass(assetData.health_score);
	}
}
</script>
{% endblock %}
```

### Frontend JavaScript Integration

#### Modern JavaScript with Real-time Updates
```javascript
// static/js/eam-dashboard.js
class EAMDashboard {
	constructor() {
		this.ws = null;
		this.charts = {};
		this.filters = {};
		this.init();
	}
	
	init() {
		this.connectWebSocket();
		this.initializeCharts();
		this.setupEventListeners();
		this.loadDashboardData();
	}
	
	connectWebSocket() {
		const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
		
		this.ws.onopen = () => {
			console.log('WebSocket connected');
			this.subscribeToUpdates();
		};
		
		this.ws.onmessage = (event) => {
			const data = JSON.parse(event.data);
			this.handleRealtimeUpdate(data);
		};
		
		this.ws.onerror = (error) => {
			console.error('WebSocket error:', error);
		};
	}
	
	subscribeToUpdates() {
		const subscriptions = {
			type: 'subscribe',
			channels: [
				'asset.health.*',
				'workorder.status.*',
				'inventory.reorder.*'
			]
		};
		this.ws.send(JSON.stringify(subscriptions));
	}
	
	handleRealtimeUpdate(data) {
		switch (data.type) {
			case 'asset.health.update':
				this.updateAssetHealth(data.payload);
				break;
			case 'workorder.status.changed':
				this.updateWorkOrderStatus(data.payload);
				break;
			case 'inventory.reorder.alert':
				this.showInventoryAlert(data.payload);
				break;
		}
	}
	
	async loadDashboardData() {
		try {
			const response = await fetch('/api/v1/analytics/dashboard', {
				headers: {
					'Authorization': `Bearer ${this.getAuthToken()}`,
					'Content-Type': 'application/json'
				}
			});
			
			const data = await response.json();
			this.updateDashboard(data);
			
		} catch (error) {
			console.error('Failed to load dashboard data:', error);
			this.showError('Failed to load dashboard data');
		}
	}
	
	updateAssetHealth(assetData) {
		const healthElement = document.querySelector(`#asset-health-${assetData.asset_id}`);
		if (healthElement) {
			healthElement.textContent = `${assetData.health_score}%`;
			healthElement.className = this.getHealthScoreClass(assetData.health_score);
		}
		
		// Update charts if needed
		this.updateHealthChart(assetData);
	}
	
	getHealthScoreClass(score) {
		if (score >= 90) return 'badge badge-success';
		if (score >= 70) return 'badge badge-info';
		if (score >= 50) return 'badge badge-warning';
		return 'badge badge-danger';
	}
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
	window.eamDashboard = new EAMDashboard();
});
```

## Testing Framework

### Test Structure and Organization

#### Test Configuration
```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock

from models import Base
from service import EAMAssetService, EAMWorkOrderService

# Test database configuration
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/eam_test"

@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture(scope="session")
async def engine():
	"""Create test database engine"""
	engine = create_async_engine(TEST_DATABASE_URL)
	
	# Create tables
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	yield engine
	
	# Cleanup
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.drop_all)
	
	await engine.dispose()

@pytest.fixture
async def session(engine):
	"""Create test database session"""
	async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
	
	async with async_session() as session:
		yield session
		await session.rollback()

@pytest.fixture
def mock_auth_service():
	"""Mock authentication service"""
	mock = AsyncMock()
	mock.check_permission.return_value = True
	mock.get_current_user.return_value = {
		"user_id": "test_user",
		"tenant_id": "test_tenant"
	}
	return mock

@pytest.fixture
def mock_audit_service():
	"""Mock audit service"""
	mock = AsyncMock()
	mock.log_action.return_value = True
	return mock

@pytest.fixture
async def asset_service(session, mock_auth_service, mock_audit_service):
	"""Create asset service for testing"""
	service = EAMAssetService(session)
	service.auth_service = mock_auth_service
	service.audit_service = mock_audit_service
	return service
```

#### Model Testing
```python
# tests/test_models.py
import pytest
from decimal import Decimal
from datetime import date, datetime
from uuid_extensions import uuid7str

from models import EAAsset, EALocation, EAWorkOrder

class TestEAAssetModel:
	"""Test EAAsset model functionality"""
	
	def test_asset_creation(self):
		"""Test basic asset creation"""
		asset = EAAsset(
			tenant_id="test_tenant",
			asset_name="Test Asset",
			asset_type="equipment",
			asset_category="production"
		)
		
		assert asset.asset_name == "Test Asset"
		assert asset.asset_type == "equipment"
		assert asset.status == "active"  # Default value
		assert asset.health_score == Decimal("100.00")  # Default value
	
	def test_asset_validation(self):
		"""Test asset field validation"""
		# Test required fields
		with pytest.raises(ValueError):
			EAAsset(asset_name="")  # Empty name should fail
		
		# Test health score bounds
		asset = EAAsset(
			tenant_id="test_tenant",
			asset_name="Test Asset",
			health_score=Decimal("150.0")  # Should be clamped to 100
		)
		assert asset.health_score <= 100
	
	def test_asset_hierarchy(self):
		"""Test parent-child relationships"""
		parent = EAAsset(
			tenant_id="test_tenant",
			asset_name="Parent Asset",
			asset_type="system"
		)
		
		child = EAAsset(
			tenant_id="test_tenant",
			asset_name="Child Asset",
			asset_type="component",
			parent_asset_id=parent.asset_id
		)
		
		assert child.parent_asset_id == parent.asset_id
	
	async def test_asset_persistence(self, session):
		"""Test asset database operations"""
		asset = EAAsset(
			tenant_id="test_tenant",
			asset_name="Persistent Asset",
			asset_type="equipment",
			purchase_cost=Decimal("10000.00")
		)
		
		session.add(asset)
		await session.commit()
		await session.refresh(asset)
		
		assert asset.asset_id is not None
		assert asset.created_on is not None
		assert asset.purchase_cost == Decimal("10000.00")
```

#### Service Testing
```python
# tests/test_services.py
import pytest
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from service import EAMAssetService

class TestEAMAssetService:
	"""Test EAM Asset Service functionality"""
	
	@pytest.mark.asyncio
	async def test_create_asset_success(self, asset_service):
		"""Test successful asset creation"""
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "Test CNC Machine",
			"asset_type": "equipment",
			"asset_category": "production",
			"purchase_cost": Decimal("50000.00")
		}
		
		asset = await asset_service.create_asset(asset_data)
		
		assert asset.asset_name == "Test CNC Machine"
		assert asset.asset_type == "equipment"
		assert asset.purchase_cost == Decimal("50000.00")
		assert asset.asset_id is not None
	
	@pytest.mark.asyncio
	async def test_create_asset_validation_error(self, asset_service):
		"""Test asset creation with invalid data"""
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "",  # Invalid: empty name
			"asset_type": "equipment"
		}
		
		with pytest.raises(ValueError, match="asset_name is required"):
			await asset_service.create_asset(asset_data)
	
	@pytest.mark.asyncio
	async def test_create_asset_permission_denied(self, asset_service):
		"""Test asset creation without permission"""
		# Mock permission denial
		asset_service.auth_service.check_permission.return_value = False
		
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "Test Asset",
			"asset_type": "equipment"
		}
		
		with pytest.raises(PermissionError):
			await asset_service.create_asset(asset_data)
	
	@pytest.mark.asyncio
	async def test_update_asset_health(self, asset_service):
		"""Test asset health updates"""
		# Create asset first
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "Health Test Asset",
			"asset_type": "equipment"
		}
		asset = await asset_service.create_asset(asset_data)
		
		# Update health
		health_data = {
			"health_score": Decimal("85.5"),
			"condition_status": "good",
			"change_reason": "Routine inspection"
		}
		
		updated_asset = await asset_service.update_asset_health(
			asset.asset_id, 
			health_data
		)
		
		assert updated_asset.health_score == Decimal("85.5")
		assert updated_asset.condition_status == "good"
	
	@pytest.mark.asyncio
	async def test_cross_capability_integration(self, asset_service):
		"""Test integration with other APG capabilities"""
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "Integration Test Asset",
			"asset_type": "equipment",
			"is_capitalized": True,
			"purchase_cost": Decimal("100000.00"),
			"maintenance_strategy": "predictive"
		}
		
		with patch.object(asset_service, '_sync_with_fixed_asset_management') as mock_sync:
			with patch.object(asset_service, '_register_with_predictive_maintenance') as mock_register:
				asset = await asset_service.create_asset(asset_data)
				
				# Verify integration calls
				mock_sync.assert_called_once_with(asset)
				mock_register.assert_called_once_with(asset)
```

#### API Testing
```python
# tests/test_api.py
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch

from api import app

@pytest.fixture
def client():
	"""Create test client"""
	return TestClient(app)

@pytest.fixture
def auth_headers():
	"""Create authentication headers"""
	return {
		"Authorization": "Bearer test_token",
		"X-Tenant-ID": "test_tenant",
		"Content-Type": "application/json"
	}

class TestAssetAPI:
	"""Test Asset API endpoints"""
	
	def test_create_asset_success(self, client, auth_headers):
		"""Test successful asset creation via API"""
		asset_data = {
			"asset_name": "API Test Asset",
			"asset_type": "equipment",
			"asset_category": "production",
			"description": "Test asset created via API"
		}
		
		with patch('api.get_asset_service') as mock_service:
			# Mock service response
			mock_asset = AsyncMock()
			mock_asset.asset_id = "test_asset_id"
			mock_asset.asset_number = "ASSET-001"
			mock_asset.status = "active"
			
			mock_service.return_value.create_asset.return_value = mock_asset
			
			response = client.post(
				"/api/v1/assets/",
				json=asset_data,
				headers=auth_headers
			)
			
			assert response.status_code == 201
			data = response.json()
			assert data["message"] == "Asset created successfully"
			assert data["data"]["asset_id"] == "test_asset_id"
	
	def test_create_asset_validation_error(self, client, auth_headers):
		"""Test asset creation with invalid data"""
		asset_data = {
			"asset_name": "",  # Invalid: empty name
			"asset_type": "equipment"
		}
		
		response = client.post(
			"/api/v1/assets/",
			json=asset_data,
			headers=auth_headers
		)
		
		assert response.status_code == 422  # Validation error
	
	def test_list_assets_with_pagination(self, client, auth_headers):
		"""Test asset listing with pagination"""
		with patch('api.get_asset_service') as mock_service:
			# Mock service response
			mock_assets = [
				AsyncMock(asset_id=f"asset_{i}", asset_name=f"Asset {i}")
				for i in range(10)
			]
			
			mock_service.return_value.search_assets.return_value = mock_assets
			
			response = client.get(
				"/api/v1/assets/?page=1&limit=5",
				headers=auth_headers
			)
			
			assert response.status_code == 200
			data = response.json()
			assert len(data) <= 5  # Respects limit
	
	def test_unauthorized_access(self, client):
		"""Test API access without authentication"""
		response = client.get("/api/v1/assets/")
		assert response.status_code == 401  # Unauthorized
```

#### Integration Testing
```python
# tests/test_integration.py
import pytest
from decimal import Decimal

class TestEAMIntegration:
	"""Integration tests for complete EAM workflows"""
	
	@pytest.mark.asyncio
	async def test_complete_asset_lifecycle(self, asset_service, work_order_service):
		"""Test complete asset lifecycle from creation to retirement"""
		
		# 1. Create asset
		asset_data = {
			"tenant_id": "test_tenant",
			"asset_name": "Lifecycle Test Asset",
			"asset_type": "equipment",
			"asset_category": "production",
			"purchase_cost": Decimal("75000.00")
		}
		
		asset = await asset_service.create_asset(asset_data)
		assert asset.status == "active"
		
		# 2. Create maintenance work order
		work_order_data = {
			"tenant_id": "test_tenant",
			"title": "Preventive Maintenance",
			"description": "Scheduled PM for lifecycle test",
			"asset_id": asset.asset_id,
			"work_type": "maintenance",
			"priority": "medium"
		}
		
		work_order = await work_order_service.create_work_order(work_order_data)
		assert work_order.asset_id == asset.asset_id
		
		# 3. Update asset health based on maintenance
		health_data = {
			"health_score": Decimal("95.0"),
			"condition_status": "excellent",
			"change_reason": "Post-maintenance inspection"
		}
		
		updated_asset = await asset_service.update_asset_health(
			asset.asset_id,
			health_data
		)
		
		assert updated_asset.health_score == Decimal("95.0")
		
		# 4. Eventually retire asset
		retirement_data = {
			"status": "retired",
			"retirement_date": "2024-12-31",
			"retirement_reason": "End of useful life"
		}
		
		retired_asset = await asset_service.retire_asset(
			asset.asset_id,
			retirement_data
		)
		
		assert retired_asset.status == "retired"
```

### Performance Testing

#### Load Testing with pytest-benchmark
```python
# tests/test_performance.py
import pytest
from decimal import Decimal
import asyncio

class TestEAMPerformance:
	"""Performance tests for EAM operations"""
	
	@pytest.mark.benchmark(group="asset_creation")
	@pytest.mark.asyncio
	async def test_bulk_asset_creation_performance(self, asset_service, benchmark):
		"""Test performance of bulk asset creation"""
		
		async def create_assets():
			tasks = []
			for i in range(100):
				asset_data = {
					"tenant_id": "test_tenant",
					"asset_name": f"Bulk Asset {i}",
					"asset_type": "equipment",
					"asset_category": "test"
				}
				tasks.append(asset_service.create_asset(asset_data))
			
			return await asyncio.gather(*tasks)
		
		# Benchmark the operation
		assets = benchmark(asyncio.run, create_assets())
		assert len(assets) == 100
	
	@pytest.mark.benchmark(group="search_operations")
	@pytest.mark.asyncio
	async def test_asset_search_performance(self, asset_service, benchmark):
		"""Test performance of asset search operations"""
		
		# Create test data first
		for i in range(1000):
			asset_data = {
				"tenant_id": "test_tenant",
				"asset_name": f"Search Test Asset {i}",
				"asset_type": "equipment" if i % 2 == 0 else "vehicle",
				"asset_category": "test"
			}
			await asset_service.create_asset(asset_data)
		
		async def search_assets():
			return await asset_service.search_assets(
				tenant_id="test_tenant",
				filters={"asset_type": "equipment"},
				page=1,
				limit=50
			)
		
		# Benchmark the search
		results = benchmark(asyncio.run, search_assets())
		assert len(results) <= 50
```

## APG Platform Integration

### Composition Engine Integration

#### Capability Registration
```python
# __init__.py - APG capability registration
from typing import Dict, Any

def register_with_apg_composition_engine() -> Dict[str, Any]:
	"""Register EAM capability with APG composition engine"""
	
	registration_data = {
		"capability_metadata": get_capability_metadata(),
		"health_check_endpoint": "/health",
		"api_documentation": "/api/docs",
		"ui_routes": get_ui_views(),
		"permissions": get_permissions(),
		"dependencies": {
			"required": get_required_dependencies(),
			"optional": get_optional_dependencies()
		},
		"provided_services": get_provided_services(),
		"event_subscriptions": [
			"auth.user.created",
			"audit.compliance.required",
			"predictive.maintenance.alert"
		],
		"event_publications": [
			"eam.asset.created",
			"eam.asset.health.critical",
			"eam.workorder.completed"
		]
	}
	
	# Register with composition engine
	composition_engine = get_composition_engine()
	return composition_engine.register_capability(registration_data)

async def handle_apg_events(event_type: str, event_data: Dict[str, Any]):
	"""Handle events from other APG capabilities"""
	
	if event_type == "predictive.maintenance.alert":
		await handle_predictive_maintenance_alert(event_data)
	
	elif event_type == "auth.user.created":
		await setup_default_permissions(event_data)
	
	elif event_type == "audit.compliance.required":
		await generate_compliance_report(event_data)

async def publish_apg_event(event_type: str, event_data: Dict[str, Any]):
	"""Publish events to other APG capabilities"""
	
	event_publisher = get_event_publisher()
	await event_publisher.publish(
		event_type=event_type,
		event_data=event_data,
		source_capability="enterprise_asset_management"
	)
```

#### Cross-Capability Data Flow
```python
class APGDataSynchronizer:
	"""Synchronize data across APG capabilities"""
	
	def __init__(self):
		self.fixed_asset_service = get_apg_service("fixed_asset_management")
		self.predictive_service = get_apg_service("predictive_maintenance")
		self.notification_service = get_apg_service("notification_engine")
	
	async def sync_asset_with_fixed_assets(self, asset: EAAsset):
		"""Sync EAM asset with financial fixed assets"""
		
		if asset.is_capitalized and asset.purchase_cost:
			fixed_asset_data = {
				"reference_id": asset.asset_id,
				"asset_name": asset.asset_name,
				"cost": float(asset.purchase_cost),
				"acquisition_date": asset.installation_date.isoformat() if asset.installation_date else None,
				"useful_life_years": asset.expected_useful_life,
				"depreciation_method": "straight_line",
				"category": asset.asset_category,
				"tenant_id": asset.tenant_id
			}
			
			try:
				await self.fixed_asset_service.create_or_update_asset(fixed_asset_data)
				
				# Publish synchronization event
				await publish_apg_event("eam.asset.synced", {
					"asset_id": asset.asset_id,
					"sync_target": "fixed_asset_management",
					"sync_status": "success"
				})
				
			except Exception as e:
				logger.error(f"Failed to sync asset with fixed assets: {e}")
				
				await publish_apg_event("eam.asset.sync.failed", {
					"asset_id": asset.asset_id,
					"sync_target": "fixed_asset_management",
					"error": str(e)
				})
	
	async def register_predictive_monitoring(self, asset: EAAsset):
		"""Register asset for predictive maintenance monitoring"""
		
		if asset.maintenance_strategy == "predictive" and asset.iot_enabled:
			monitoring_config = {
				"asset_id": asset.asset_id,
				"asset_type": asset.asset_type,
				"criticality": asset.criticality_level,
				"monitoring_parameters": {
					"temperature": {"min": 0, "max": 80, "unit": "celsius"},
					"vibration": {"min": 0, "max": 10, "unit": "mm/s"},
					"pressure": {"min": 0, "max": 100, "unit": "bar"}
				},
				"alert_thresholds": {
					"warning": 70,
					"critical": 85
				},
				"tenant_id": asset.tenant_id
			}
			
			await self.predictive_service.register_asset_monitoring(monitoring_config)
```

### Authentication and Authorization

#### APG RBAC Integration
```python
from apg.auth_rbac import RBACService, Permission, Role

class EAMAuthIntegration:
	"""Integration with APG auth_rbac capability"""
	
	def __init__(self):
		self.rbac_service = RBACService()
	
	async def setup_eam_permissions(self):
		"""Set up EAM-specific permissions in APG RBAC"""
		
		permissions = [
			Permission(
				name="eam.admin",
				description="Full EAM system administration",
				resource_type="eam",
				actions=["*"]
			),
			Permission(
				name="eam.asset.create",
				description="Create and modify assets",
				resource_type="asset",
				actions=["create", "update"]
			),
			Permission(
				name="eam.asset.view",
				description="View asset information",
				resource_type="asset",
				actions=["read"]
			),
			Permission(
				name="eam.workorder.execute",
				description="Execute work orders",
				resource_type="workorder",
				actions=["update", "complete"]
			)
		]
		
		for permission in permissions:
			await self.rbac_service.create_permission(permission)
	
	async def setup_eam_roles(self):
		"""Set up default EAM roles"""
		
		roles = [
			Role(
				name="EAM_Administrator",
				description="EAM system administrator",
				permissions=["eam.admin"]
			),
			Role(
				name="Maintenance_Manager",
				description="Maintenance management role",
				permissions=[
					"eam.asset.view",
					"eam.workorder.create",
					"eam.workorder.assign",
					"eam.maintenance.plan"
				]
			),
			Role(
				name="Maintenance_Technician",
				description="Field maintenance technician",
				permissions=[
					"eam.asset.view",
					"eam.workorder.execute",
					"eam.inventory.issue"
				]
			)
		]
		
		for role in roles:
			await self.rbac_service.create_role(role)
	
	async def check_permission(self, user_id: str, permission: str, resource_id: str = None) -> bool:
		"""Check user permission through APG RBAC"""
		
		return await self.rbac_service.check_permission(
			user_id=user_id,
			permission=permission,
			resource_id=resource_id
		)
```

#### Multi-Tenant Security Implementation
```python
class TenantSecurityManager:
	"""Manage multi-tenant security for EAM"""
	
	def __init__(self):
		self.rbac_service = get_rbac_service()
	
	async def ensure_tenant_isolation(self, user_id: str, tenant_id: str, resource_id: str = None):
		"""Ensure user can only access their tenant's data"""
		
		user_tenant = await self.rbac_service.get_user_tenant(user_id)
		
		if user_tenant != tenant_id:
			raise PermissionError("Cross-tenant access denied")
		
		# Additional resource-level checks
		if resource_id:
			resource_tenant = await self.get_resource_tenant(resource_id)
			if resource_tenant != tenant_id:
				raise PermissionError("Resource belongs to different tenant")
	
	async def get_resource_tenant(self, resource_id: str) -> str:
		"""Get tenant ID for a resource"""
		
		# Check different resource types
		for model_class in [EAAsset, EAWorkOrder, EAInventory]:
			resource = await self.session.get(model_class, resource_id)
			if resource:
				return resource.tenant_id
		
		raise ValueError("Resource not found")
	
	def apply_tenant_filter(self, query, tenant_id: str):
		"""Apply tenant filtering to database queries"""
		
		# Dynamically add tenant filter to queries
		if hasattr(query.column_descriptions[0]['type'], 'tenant_id'):
			query = query.filter(
				query.column_descriptions[0]['type'].tenant_id == tenant_id
			)
		
		return query
```

### Audit and Compliance Integration

#### Comprehensive Audit Logging
```python
from apg.audit_compliance import AuditService, AuditEvent

class EAMAuditIntegration:
	"""Integration with APG audit_compliance capability"""
	
	def __init__(self):
		self.audit_service = AuditService()
	
	async def log_asset_action(self, action: str, asset: EAAsset, user_id: str, details: Dict[str, Any] = None):
		"""Log asset-related actions for audit compliance"""
		
		audit_event = AuditEvent(
			event_type="eam.asset.action",
			action=action,
			resource_type="asset",
			resource_id=asset.asset_id,
			user_id=user_id,
			tenant_id=asset.tenant_id,
			timestamp=datetime.utcnow(),
			details={
				"asset_number": asset.asset_number,
				"asset_name": asset.asset_name,
				"asset_type": asset.asset_type,
				"previous_values": details.get("previous_values"),
				"new_values": details.get("new_values"),
				"change_reason": details.get("change_reason")
			}
		)
		
		await self.audit_service.log_event(audit_event)
	
	async def generate_compliance_report(self, tenant_id: str, report_type: str, date_range: Dict[str, Any]):
		"""Generate compliance reports for regulatory requirements"""
		
		report_config = {
			"report_type": report_type,
			"tenant_id": tenant_id,
			"date_range": date_range,
			"data_sources": [
				"eam.assets",
				"eam.work_orders", 
				"eam.maintenance_records"
			],
			"compliance_standards": [
				"ISO_55000",
				"OSHA_1910",
				"SOX_404"
			]
		}
		
		return await self.audit_service.generate_compliance_report(report_config)
```

## Deployment and DevOps

### Container Configuration

#### Dockerfile Optimization
```dockerfile
# Multi-stage build for production
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Create application user
RUN useradd --create-home --shell /bin/bash eam_user

WORKDIR /app

# Copy application code
COPY --chown=eam_user:eam_user . .

# Switch to application user
USER eam_user

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  eam-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://eam_dev:dev_password@postgres:5432/eam_dev
      - REDIS_URL=redis://redis:6379/0
      - DEBUG=true
    volumes:
      - .:/app
      - ~/.aws:/home/eam_user/.aws:ro
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped

  eam-worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: python -m worker.main
    environment:
      - DATABASE_URL=postgresql://eam_dev:dev_password@postgres:5432/eam_dev
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: eam_dev
      POSTGRES_USER: eam_dev
      POSTGRES_PASSWORD: dev_password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/01-init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U eam_dev -d eam_dev"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_dev_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

volumes:
  postgres_dev_data:
  redis_dev_data:
```

### CI/CD Pipeline

#### GitHub Actions Workflow
```yaml
# .github/workflows/eam-ci-cd.yml
name: EAM Capability CI/CD

on:
  push:
    branches: [main, develop]
    paths:
      - 'capabilities/general_cross_functional/enterprise_asset_management/**'
  pull_request:
    branches: [main]
    paths:
      - 'capabilities/general_cross_functional/enterprise_asset_management/**'

env:
  PYTHON_VERSION: 3.12
  POETRY_VERSION: 1.7.1

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: eam_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install Poetry
      run: |
        curl -sSL https://install.python-poetry.org | python3 -
        echo "$HOME/.local/bin" >> $GITHUB_PATH
    
    - name: Install dependencies
      working-directory: capabilities/general_cross_functional/enterprise_asset_management
      run: |
        poetry install --with dev
    
    - name: Run linting
      working-directory: capabilities/general_cross_functional/enterprise_asset_management
      run: |
        poetry run black --check .
        poetry run isort --check-only .
        poetry run flake8 .
        poetry run mypy .
    
    - name: Run tests
      working-directory: capabilities/general_cross_functional/enterprise_asset_management
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/eam_test
        REDIS_URL: redis://localhost:6379/0
      run: |
        poetry run pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: capabilities/general_cross_functional/enterprise_asset_management/coverage.xml
        flags: eam-capability

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      working-directory: capabilities/general_cross_functional/enterprise_asset_management
      run: |
        pip install safety bandit
        safety check
        bandit -r . -f json -o bandit-report.json

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: capabilities/general_cross_functional/enterprise_asset_management
        push: true
        tags: |
          ghcr.io/${{ github.repository }}/eam-capability:latest
          ghcr.io/${{ github.repository }}/eam-capability:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment script here
    
    - name: Run integration tests
      run: |
        echo "Running integration tests"
        # Add integration test script here
    
    - name: Deploy to production
      if: success()
      run: |
        echo "Deploying to production environment"
        # Add production deployment script here
```

### Monitoring and Observability

#### Prometheus Metrics
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Define metrics
eam_requests_total = Counter(
    'eam_requests_total',
    'Total number of EAM API requests',
    ['method', 'endpoint', 'status_code']
)

eam_request_duration = Histogram(
    'eam_request_duration_seconds',
    'EAM request duration in seconds',
    ['method', 'endpoint']
)

eam_assets_total = Gauge(
    'eam_assets_total',
    'Total number of assets',
    ['tenant_id', 'asset_type']
)

eam_asset_health_score = Gauge(
    'eam_asset_health_score',
    'Asset health score',
    ['tenant_id', 'asset_id', 'asset_type']
)

eam_work_orders_active = Gauge(
    'eam_work_orders_active',
    'Number of active work orders',
    ['tenant_id', 'priority', 'status']
)

# Decorators for automatic metrics collection
def track_requests(func):
    """Decorator to track API request metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status_code = 200
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            raise
        finally:
            duration = time.time() - start_time
            
            # Extract request information
            request = kwargs.get('request')
            if request:
                eam_requests_total.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=status_code
                ).inc()
                
                eam_request_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
    
    return wrapper

# Metrics collection functions
async def update_asset_metrics():
    """Update asset-related metrics"""
    
    # This would run periodically to update Prometheus metrics
    session = get_async_session()
    
    # Update asset counts by type and tenant
    query = select(
        EAAsset.tenant_id,
        EAAsset.asset_type,
        func.count(EAAsset.asset_id)
    ).group_by(EAAsset.tenant_id, EAAsset.asset_type)
    
    result = await session.execute(query)
    
    for tenant_id, asset_type, count in result:
        eam_assets_total.labels(
            tenant_id=tenant_id,
            asset_type=asset_type
        ).set(count)
    
    # Update individual asset health scores
    health_query = select(EAAsset).where(EAAsset.health_score.isnot(None))
    health_result = await session.execute(health_query)
    
    for asset in health_result.scalars():
        eam_asset_health_score.labels(
            tenant_id=asset.tenant_id,
            asset_id=asset.asset_id,
            asset_type=asset.asset_type
        ).set(float(asset.health_score))
```

#### Application Logging
```python
# logging_config.py
import logging
import logging.config
from pythonjsonlogger import jsonlogger

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': jsonlogger.JsonFormatter,
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'json',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': '/var/log/eam/application.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'audit': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': '/var/log/eam/audit.log',
            'maxBytes': 10485760,
            'backupCount': 10
        }
    },
    'loggers': {
        'eam': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'eam.audit': {
            'level': 'INFO',
            'handlers': ['audit'],
            'propagate': False
        },
        'uvicorn': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}

def setup_logging():
    """Set up application logging"""
    logging.config.dictConfig(LOGGING_CONFIG)

# Usage in application
logger = logging.getLogger('eam')
audit_logger = logging.getLogger('eam.audit')

# Log business events
logger.info("Asset created", extra={
    "asset_id": asset.asset_id,
    "tenant_id": asset.tenant_id,
    "user_id": user_id
})

# Log audit events
audit_logger.info("Asset modified", extra={
    "event_type": "asset_update",
    "asset_id": asset.asset_id,
    "user_id": user_id,
    "changes": changes_dict
})
```

## Contributing Guidelines

### Development Workflow

#### Git Workflow
```bash
# 1. Create feature branch
git checkout -b feature/eam-enhancement-xyz

# 2. Make changes following coding standards
# - Use tabs for indentation
# - Follow async patterns
# - Add comprehensive tests
# - Update documentation

# 3. Run tests and linting
poetry run pytest tests/ -v
poetry run black .
poetry run isort .
poetry run flake8 .

# 4. Commit with descriptive messages
git add .
git commit -m "feat(eam): add predictive maintenance alerts

- Add health score monitoring with configurable thresholds
- Integrate with APG notification engine for alerts
- Add WebSocket support for real-time updates
- Include comprehensive test coverage"

# 5. Push and create pull request
git push origin feature/eam-enhancement-xyz
```

#### Code Review Checklist

**Functionality:**
- [ ] Feature works as specified
- [ ] Edge cases are handled
- [ ] Error handling is comprehensive
- [ ] Performance is acceptable

**Code Quality:**
- [ ] Follows CLAUDE.md standards (async, tabs, modern typing)
- [ ] Code is well-documented with docstrings
- [ ] Variable and function names are descriptive
- [ ] No code duplication

**Testing:**
- [ ] Unit tests cover new functionality
- [ ] Integration tests verify cross-capability interaction
- [ ] Performance tests validate scalability
- [ ] All tests pass in CI/CD pipeline

**APG Integration:**
- [ ] Proper multi-tenant isolation
- [ ] Audit logging implemented
- [ ] Permission checks in place
- [ ] Cross-capability events handled

**Documentation:**
- [ ] API documentation updated
- [ ] User guide reflects changes
- [ ] Developer guide includes new patterns
- [ ] Deployment notes updated if needed

### Contribution Types

#### Bug Fixes
```python
# Example bug fix with comprehensive testing
async def fix_asset_health_calculation(self, asset: EAAsset) -> Decimal:
    """
    Fix health score calculation to handle edge cases.
    
    Bug: Health score calculation failed when maintenance_records was empty
    Fix: Add null check and default calculation
    
    Related issue: #EAM-123
    """
    
    # Previous implementation failed here
    if not asset.maintenance_records:
        # Default health score based on age and usage
        age_factor = self._calculate_age_factor(asset)
        usage_factor = self._calculate_usage_factor(asset)
        return Decimal("100.0") * age_factor * usage_factor
    
    # Existing calculation for assets with maintenance history
    return self._calculate_health_from_history(asset)

# Test for the bug fix
@pytest.mark.asyncio
async def test_health_calculation_no_maintenance_records(asset_service):
    """Test health calculation for assets without maintenance records"""
    
    asset = EAAsset(
        tenant_id="test_tenant",
        asset_name="New Asset",
        asset_type="equipment",
        installation_date=date.today() - timedelta(days=30)
    )
    
    # Should not raise exception
    health_score = await asset_service.calculate_asset_health(asset)
    
    assert health_score is not None
    assert 0 <= health_score <= 100
```

#### Feature Enhancements
```python
# Example feature enhancement with full APG integration
class AssetDigitalTwinIntegration:
    """
    Enhancement: Integrate EAM assets with digital twin capability
    
    Features:
    - Automatic digital twin creation for IoT-enabled assets
    - Real-time sensor data synchronization
    - Predictive analytics integration
    - 3D visualization support
    """
    
    async def create_digital_twin(self, asset: EAAsset) -> str:
        """Create digital twin for asset with APG integration"""
        
        assert asset.iot_enabled, "Asset must be IoT-enabled for digital twin"
        
        digital_twin_service = get_apg_service("digital_twin_marketplace")
        
        twin_config = {
            "asset_reference_id": asset.asset_id,
            "twin_type": "physical_asset",
            "model_template": self._get_twin_template(asset.asset_type),
            "sensor_mappings": self._get_sensor_config(asset),
            "update_frequency": "real_time",
            "tenant_id": asset.tenant_id
        }
        
        twin_id = await digital_twin_service.create_twin(twin_config)
        
        # Update asset with digital twin reference
        asset.digital_twin_id = twin_id
        asset.has_digital_twin = True
        
        # Publish integration event
        await publish_apg_event("eam.digital_twin.created", {
            "asset_id": asset.asset_id,
            "twin_id": twin_id,
            "integration_status": "active"
        })
        
        return twin_id
```

#### Performance Improvements
```python
# Example performance optimization with benchmarking
class OptimizedAssetSearch:
    """
    Performance improvement: Optimize asset search with advanced indexing
    
    Improvements:
    - Add full-text search capabilities
    - Implement query result caching
    - Optimize database indexes
    - Add search result pagination
    """
    
    @track_requests
    async def search_assets_optimized(
        self,
        tenant_id: str,
        search_query: str,
        filters: Dict[str, Any],
        page: int = 1,
        limit: int = 25
    ) -> Tuple[List[EAAsset], int]:
        """Optimized asset search with caching and full-text search"""
        
        # Check cache first
        cache_key = f"asset_search:{tenant_id}:{hash(str(filters))}:{search_query}:{page}:{limit}"
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Build optimized query with proper indexes
        query = select(EAAsset).where(EAAsset.tenant_id == tenant_id)
        
        # Full-text search on indexed columns
        if search_query:
            search_vector = func.to_tsvector('english', 
                EAAsset.asset_name + ' ' + 
                func.coalesce(EAAsset.description, '') + ' ' +
                func.coalesce(EAAsset.manufacturer, '')
            )
            query = query.where(search_vector.match(search_query))
        
        # Apply filters with index hints
        for field, value in filters.items():
            if hasattr(EAAsset, field):
                query = query.where(getattr(EAAsset, field) == value)
        
        # Count total with same filters (but cached separately)
        count_query = select(func.count(EAAsset.asset_id)).select_from(query.subquery())
        total = await self.session.scalar(count_query)
        
        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        # Execute query with read replica if available
        result = await self.session.execute(query)
        assets = result.scalars().all()
        
        # Cache results for 5 minutes
        search_result = (assets, total)
        await self.cache.set(cache_key, search_result, ttl=300)
        
        return search_result

# Performance test for the optimization
@pytest.mark.benchmark(group="search_optimization")
@pytest.mark.asyncio
async def test_optimized_search_performance(asset_service, benchmark):
    """Benchmark optimized search performance"""
    
    # Create test data
    for i in range(10000):
        await asset_service.create_asset({
            "tenant_id": "perf_test",
            "asset_name": f"Performance Test Asset {i}",
            "asset_type": "equipment" if i % 3 == 0 else "vehicle",
            "description": f"Test asset for performance benchmark {i}"
        })
    
    async def search_operation():
        return await asset_service.search_assets_optimized(
            tenant_id="perf_test",
            search_query="Performance Test",
            filters={"asset_type": "equipment"},
            page=1,
            limit=50
        )
    
    # Benchmark should show improvement over previous implementation
    results = benchmark(asyncio.run, search_operation())
    assert len(results[0]) <= 50
    assert results[1] > 3000  # Should find ~3333 equipment assets
```

## Advanced Topics

### Custom Validators and Business Rules

#### Domain-Specific Validation
```python
# validators.py
from typing import Any, Dict
from decimal import Decimal
from datetime import date, datetime, timedelta

class AssetValidationRules:
    """Custom validation rules for asset management"""
    
    @staticmethod
    def validate_asset_hierarchy(parent_id: str, child_data: Dict[str, Any]) -> bool:
        """Validate asset hierarchy rules"""
        
        # Rule: Child asset cannot be same type as parent
        if parent_id and child_data.get('asset_type'):
            parent = get_asset_by_id(parent_id)
            if parent and parent.asset_type == child_data['asset_type']:
                raise ValueError("Child asset cannot be same type as parent")
        
        # Rule: Maximum hierarchy depth of 5 levels
        if parent_id:
            depth = calculate_hierarchy_depth(parent_id)
            if depth >= 5:
                raise ValueError("Maximum hierarchy depth exceeded")
        
        return True
    
    @staticmethod
    def validate_maintenance_schedule(asset: EAAsset, schedule_data: Dict[str, Any]) -> bool:
        """Validate maintenance scheduling rules"""
        
        # Rule: Preventive maintenance frequency based on criticality
        if asset.criticality_level == "critical":
            max_days = 30
        elif asset.criticality_level == "high":
            max_days = 60
        elif asset.criticality_level == "medium":
            max_days = 90
        else:
            max_days = 180
        
        frequency = schedule_data.get('maintenance_frequency_days')
        if frequency and frequency > max_days:
            raise ValueError(f"Maintenance frequency too long for {asset.criticality_level} asset")
        
        return True
    
    @staticmethod
    def validate_financial_data(asset_data: Dict[str, Any]) -> bool:
        """Validate financial data consistency"""
        
        purchase_cost = asset_data.get('purchase_cost')
        replacement_cost = asset_data.get('replacement_cost')
        
        # Rule: Replacement cost should be >= purchase cost (inflation)
        if purchase_cost and replacement_cost:
            if replacement_cost < purchase_cost:
                raise ValueError("Replacement cost should not be less than purchase cost")
        
        # Rule: Capitalization threshold
        capitalization_threshold = Decimal("5000.00")
        if purchase_cost and purchase_cost >= capitalization_threshold:
            asset_data['is_capitalized'] = True
        
        return True

# Usage in service layer
class EAMAssetService(BaseService):
    
    async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
        """Create asset with comprehensive validation"""
        
        # Apply custom validation rules
        AssetValidationRules.validate_financial_data(asset_data)
        
        if asset_data.get('parent_asset_id'):
            AssetValidationRules.validate_asset_hierarchy(
                asset_data['parent_asset_id'], 
                asset_data
            )
        
        # Continue with creation...
        return await super().create_asset(asset_data)
```

### Event-Driven Architecture

#### Event Publishing and Subscription
```python
# events.py
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class EAMEvent:
    """EAM domain event"""
    event_type: str
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    tenant_id: str
    user_id: str
    version: int = 1

class EventBus:
    """Event bus for EAM domain events"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_store = []
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event: EAMEvent):
        """Publish event to subscribers"""
        
        # Store event for audit and replay
        self.event_store.append(event)
        
        # Notify subscribers
        if event.event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event.event_type]:
                tasks.append(handler(event))
            
            # Execute handlers concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Publish to APG event system
        await self._publish_to_apg(event)
    
    async def _publish_to_apg(self, event: EAMEvent):
        """Publish event to APG platform"""
        apg_event = {
            "source": "enterprise_asset_management",
            "type": event.event_type,
            "data": event.event_data,
            "timestamp": event.timestamp.isoformat(),
            "tenant_id": event.tenant_id
        }
        
        await publish_apg_event(event.event_type, apg_event)

# Event handlers
class AssetEventHandlers:
    """Event handlers for asset-related events"""
    
    def __init__(self, notification_service, predictive_service):
        self.notification_service = notification_service
        self.predictive_service = predictive_service
    
    async def handle_asset_created(self, event: EAMEvent):
        """Handle asset creation event"""
        asset_data = event.event_data
        
        # Send welcome notification
        await self.notification_service.send_notification({
            "type": "asset_created",
            "recipient": asset_data.get("custodian_employee_id"),
            "message": f"Asset {asset_data['asset_name']} has been assigned to you"
        })
        
        # Register for predictive maintenance if applicable
        if asset_data.get("maintenance_strategy") == "predictive":
            await self.predictive_service.register_asset({
                "asset_id": event.aggregate_id,
                "asset_type": asset_data["asset_type"],
                "criticality": asset_data["criticality_level"]
            })
    
    async def handle_asset_health_critical(self, event: EAMEvent):
        """Handle critical asset health event"""
        
        health_data = event.event_data
        
        # Create emergency work order
        work_order_data = {
            "title": f"URGENT: Asset Health Critical - {health_data['asset_name']}",
            "description": f"Asset health score dropped to {health_data['health_score']}%",
            "asset_id": event.aggregate_id,
            "work_type": "emergency",
            "priority": "critical",
            "tenant_id": event.tenant_id
        }
        
        # This would integrate with work order service
        await self._create_emergency_work_order(work_order_data)
        
        # Send immediate alerts
        await self.notification_service.send_alert({
            "type": "asset_health_critical",
            "severity": "high",
            "asset_id": event.aggregate_id,
            "message": f"Asset {health_data['asset_name']} requires immediate attention"
        })

# Integration in service layer
class EAMAssetService(BaseService):
    
    def __init__(self, session, event_bus: EventBus):
        super().__init__(session)
        self.event_bus = event_bus
    
    async def create_asset(self, asset_data: Dict[str, Any]) -> EAAsset:
        """Create asset and publish event"""
        
        asset = await super().create_asset(asset_data)
        
        # Publish domain event
        event = EAMEvent(
            event_type="asset.created",
            aggregate_id=asset.asset_id,
            aggregate_type="asset",
            event_data={
                "asset_name": asset.asset_name,
                "asset_type": asset.asset_type,
                "criticality_level": asset.criticality_level,
                "custodian_employee_id": asset.custodian_employee_id
            },
            timestamp=datetime.utcnow(),
            tenant_id=asset.tenant_id,
            user_id=self.current_user.user_id
        )
        
        await self.event_bus.publish(event)
        
        return asset
```

### Plugin and Extension System

#### Plugin Architecture
```python
# plugins/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class EAMPlugin(ABC):
    """Base class for EAM plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    async def process_asset_event(self, event_type: str, asset_data: Dict[str, Any]):
        """Process asset-related events"""
        pass
    
    @abstractmethod
    def get_additional_fields(self) -> List[Dict[str, Any]]:
        """Return additional fields for asset model"""
        pass

# Example plugin implementation
class MaintenanceOptimizationPlugin(EAMPlugin):
    """Plugin for advanced maintenance optimization"""
    
    @property
    def name(self) -> str:
        return "maintenance_optimization"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]):
        """Initialize optimization algorithms"""
        self.ml_model = await self._load_ml_model(config.get("model_path"))
        self.optimization_params = config.get("optimization_params", {})
    
    async def process_asset_event(self, event_type: str, asset_data: Dict[str, Any]):
        """Process events for maintenance optimization"""
        
        if event_type == "asset.health.updated":
            # Run optimization algorithm
            optimization_result = await self._optimize_maintenance_schedule(asset_data)
            
            if optimization_result.get("schedule_change_recommended"):
                # Publish optimization recommendation
                await publish_apg_event("maintenance.optimization.recommendation", {
                    "asset_id": asset_data["asset_id"],
                    "current_schedule": asset_data["maintenance_frequency_days"],
                    "recommended_schedule": optimization_result["recommended_frequency"],
                    "expected_savings": optimization_result["cost_savings"],
                    "confidence": optimization_result["confidence_score"]
                })
    
    def get_additional_fields(self) -> List[Dict[str, Any]]:
        """Add optimization-specific fields"""
        return [
            {
                "name": "optimization_score",
                "type": "decimal",
                "description": "Maintenance optimization score (0-100)"
            },
            {
                "name": "last_optimization_date",
                "type": "datetime",
                "description": "Last optimization analysis date"
            }
        ]
    
    async def _optimize_maintenance_schedule(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML-based maintenance optimization"""
        
        # Feature extraction
        features = self._extract_features(asset_data)
        
        # ML prediction
        prediction = await self.ml_model.predict(features)
        
        return {
            "recommended_frequency": prediction.get("optimal_frequency"),
            "cost_savings": prediction.get("estimated_savings"),
            "confidence_score": prediction.get("confidence"),
            "schedule_change_recommended": prediction.get("change_recommended", False)
        }

# Plugin manager
class PluginManager:
    """Manage EAM plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, EAMPlugin] = {}
        self.hooks: Dict[str, List[EAMPlugin]] = {}
    
    async def register_plugin(self, plugin: EAMPlugin, config: Dict[str, Any]):
        """Register and initialize plugin"""
        
        await plugin.initialize(config)
        self.plugins[plugin.name] = plugin
        
        # Register for event hooks
        self.hooks.setdefault("asset_events", []).append(plugin)
    
    async def execute_hooks(self, hook_type: str, event_type: str, data: Dict[str, Any]):
        """Execute plugin hooks for events"""
        
        if hook_type in self.hooks:
            tasks = []
            for plugin in self.hooks[hook_type]:
                tasks.append(plugin.process_asset_event(event_type, data))
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_additional_model_fields(self) -> List[Dict[str, Any]]:
        """Get additional fields from all plugins"""
        
        fields = []
        for plugin in self.plugins.values():
            fields.extend(plugin.get_additional_fields())
        
        return fields

# Integration with service layer
class EAMAssetService(BaseService):
    
    def __init__(self, session, plugin_manager: PluginManager):
        super().__init__(session)
        self.plugin_manager = plugin_manager
    
    async def update_asset_health(self, asset_id: str, health_data: Dict[str, Any]) -> EAAsset:
        """Update asset health and trigger plugin hooks"""
        
        asset = await super().update_asset_health(asset_id, health_data)
        
        # Trigger plugin hooks
        await self.plugin_manager.execute_hooks(
            "asset_events",
            "asset.health.updated",
            {
                "asset_id": asset.asset_id,
                "health_score": float(asset.health_score),
                "maintenance_frequency_days": asset.maintenance_frequency_days,
                "asset_type": asset.asset_type
            }
        )
        
        return asset
```

---

*Developer Guide Version 1.0 - Last Updated: 2024-01-01*

This guide covers the essential aspects of developing and extending the Enterprise Asset Management capability. For specific implementation questions or advanced use cases, please refer to the [API Reference](API_REFERENCE.md) or contact the development team.

**Contributing**: Follow the guidelines in this document and submit pull requests with comprehensive tests and documentation.

**Support**: For development support, contact nyimbi@gmail.com or visit the project repository.