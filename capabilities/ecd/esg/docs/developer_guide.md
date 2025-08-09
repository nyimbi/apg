# APG Sustainability & ESG Management - Developer Guide

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Target Audience:** Developers, DevOps Engineers, System Integrators

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Environment](#development-environment)
3. [APG Integration Patterns](#apg-integration-patterns)
4. [Database Schema](#database-schema)
5. [API Reference](#api-reference)
6. [Service Layer](#service-layer)
7. [Frontend Development](#frontend-development)
8. [Testing Framework](#testing-framework)
9. [Deployment Guide](#deployment-guide)
10. [Performance Optimization](#performance-optimization)
11. [Security Guidelines](#security-guidelines)
12. [Monitoring & Observability](#monitoring--observability)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APG Platform Layer                       │
├─────────────────────────────────────────────────────────────┤
│  auth_rbac  │ audit_compliance │ ai_orchestration │ etc...  │
├─────────────────────────────────────────────────────────────┤
│                ESG Management Capability                    │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Web UI    │   REST API   │  WebSocket  │   Mobile    │  │
│  │ (Flask-AB)  │  (FastAPI)   │   (Real-time)│    (PWA)   │  │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤  │
│  │          Service Layer (Async Python)                │  │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐  │  │
│  │  │ Metrics │ Targets │ Stakeh. │ Supply  │   AI    │  │  │
│  │  │ Service │ Service │ Service │ Chain   │ Service │  │  │
│  ├─────────────────────────────────────────────────────────┤  │
│  │               Data Layer (PostgreSQL)                  │  │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐  │  │
│  │  │ Tenants │ Metrics │ Targets │ Stakeh. │  ESG    │  │  │
│  │  │         │         │         │         │ Reports │  │  │
└──────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Models Layer** (`models.py`)
- SQLAlchemy ORM models with APG patterns
- Multi-tenant data architecture
- Pydantic validation models
- Async-compatible database operations

#### 2. **Service Layer** (`service.py`)
- Business logic implementation
- APG ecosystem integration
- AI/ML model integration
- Real-time data processing

#### 3. **API Layer** (`api.py`)
- FastAPI REST endpoints
- WebSocket real-time features
- APG authentication integration
- Comprehensive error handling

#### 4. **Views Layer** (`views.py`)
- Flask-AppBuilder UI components
- Executive dashboards
- Management interfaces
- Public stakeholder portals

#### 5. **Blueprint Integration** (`blueprint.py`)
- APG composition engine registration
- Menu structure and routing
- Security integration
- Health monitoring

### Technology Stack

**Backend**
- **Python 3.12+**: Modern async/await patterns
- **FastAPI**: High-performance API framework
- **SQLAlchemy**: Advanced ORM with async support
- **Pydantic v2**: Data validation and serialization
- **PostgreSQL**: Primary data store

**Frontend**
- **Flask-AppBuilder**: APG-compatible UI framework
- **Jinja2**: Template engine
- **Bootstrap**: Responsive design framework
- **Chart.js**: Data visualization
- **Progressive Web App**: Mobile experience

**Integration**
- **APG Ecosystem**: Full platform integration
- **Redis**: Caching and real-time features
- **Celery**: Background task processing
- **WebSockets**: Real-time communication
- **Docker**: Containerized deployment

---

## Development Environment

### Prerequisites

```bash
# Python 3.12+ with uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# APG Platform development environment
export APG_ENV=development
export APG_DATABASE_URL=postgresql://user:pass@localhost/apg_dev
export APG_REDIS_URL=redis://localhost:6379
```

### Project Setup

```bash
# Clone APG platform repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform

# Navigate to ESG capability
cd capabilities/general_cross_functional/sustainability_esg_management

# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head

# Start development server
uv run python -m uvicorn api:app --reload --port 8000
```

### Development Tools

```bash
# Type checking
uv run pyright

# Testing
uv run pytest -vxs tests/

# Code formatting
uv run black . --line-length 100

# Linting
uv run ruff check .

# Security scanning
uv run bandit -r .
```

### Code Standards

#### Python Style (per CLAUDE.md)
```python
# Use tabs for indentation (not spaces)
def async_function_example():
	"""Example function following APG patterns"""
	user_id: str | None = None
	items: list[str] = []
	config: dict[str, Any] = {}
	
	# Use _log_ prefix for logging methods
	log_message = _log_user_action(user_id, "example_action")
	
	# Runtime assertions at function start/end
	assert user_id is not None, "User ID is required"
	
	return {"status": "success"}

# Pydantic models with APG patterns
class ESGMetricRequest(BaseModel):
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True
	)
	
	name: str = Field(..., min_length=1, max_length=255)
	value: Decimal = Field(..., ge=0)
	metric_type: ESGMetricType
	
	@AfterValidator
	def validate_metric_type(cls, v):
		"""Custom validation logic"""
		return v
```

#### Async Patterns
```python
# Service layer example
class ESGManagementService:
	def __init__(self, db_session: Session, tenant_id: str):
		self.db_session = db_session
		self.tenant_id = tenant_id
	
	async def create_metric(
		self,
		user_id: str,
		metric_data: dict[str, Any]
	) -> ESGMetric:
		"""Create ESG metric with APG integration"""
		
		# Permission check via APG auth_rbac
		await self.auth_service.check_permission(
			user_id=user_id,
			tenant_id=self.tenant_id,
			resource="esg_metrics",
			action="create"
		)
		
		# Create metric with validation
		metric = ESGMetric(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			**metric_data
		)
		
		# Database operations
		self.db_session.add(metric)
		self.db_session.commit()
		self.db_session.refresh(metric)
		
		# APG audit logging
		await self.audit_service.log_activity(
			user_id=user_id,
			action="create_metric",
			resource_id=metric.id
		)
		
		# Real-time collaboration update
		await self.collaboration_service.broadcast_update(
			tenant_id=self.tenant_id,
			channel="esg_metrics_updates",
			message={"type": "metric_created", "metric_id": metric.id}
		)
		
		return metric
```

---

## APG Integration Patterns

### Capability Registration

```python
# __init__.py - APG capability metadata
CAPABILITY_METADATA = {
	"name": "sustainability_esg_management",
	"version": "1.0.0",
	"description": "Revolutionary ESG management with AI intelligence",
	"category": "general_cross_functional",
	"requires": [
		"auth_rbac",
		"audit_compliance", 
		"ai_orchestration",
		"real_time_collaboration"
	],
	"provides": [
		"esg_metrics_management",
		"sustainability_intelligence",
		"stakeholder_engagement",
		"supply_chain_esg"
	],
	"api_endpoints": [
		{"path": "/api/v1/esg/metrics", "methods": ["GET", "POST"]},
		{"path": "/api/v1/esg/targets", "methods": ["GET", "POST"]},
		{"path": "/api/v1/esg/stakeholders", "methods": ["GET", "POST"]}
	],
	"ai_capabilities": [
		{
			"name": "sustainability_prediction",
			"models": ["lstm_environmental", "transformer_carbon"],
			"input_schema": "ESGMetricData",
			"output_schema": "SustainabilityPrediction"
		}
	]
}
```

### Authentication Integration

```python
# APG auth_rbac service integration
from ...auth_rbac.service import AuthRBACService

class ESGAuthIntegration:
	def __init__(self, auth_service: AuthRBACService):
		self.auth_service = auth_service
	
	async def check_esg_permission(
		self,
		user_id: str,
		tenant_id: str,
		resource: str,
		action: str
	) -> bool:
		"""Check ESG-specific permissions"""
		return await self.auth_service.check_permission(
			user_id=user_id,
			tenant_id=tenant_id,
			resource=f"esg_{resource}",
			action=action
		)
	
	async def get_user_esg_roles(self, user_id: str) -> list[str]:
		"""Get user's ESG roles"""
		roles = await self.auth_service.get_user_roles(user_id)
		return [role for role in roles if role.startswith("esg_")]
```

### Audit Compliance Integration

```python
# APG audit_compliance service integration
from ...audit_compliance.service import AuditComplianceService

class ESGAuditIntegration:
	def __init__(self, audit_service: AuditComplianceService):
		self.audit_service = audit_service
	
	async def log_esg_activity(
		self,
		user_id: str,
		tenant_id: str,
		action: str,
		resource_type: str,
		resource_id: str,
		details: dict[str, Any] | None = None
	) -> str:
		"""Log ESG-specific activities"""
		return await self.audit_service.log_activity(
			user_id=user_id,
			tenant_id=tenant_id,
			action=action,
			resource_type=resource_type,
			resource_id=resource_id,
			details=details or {},
			category="esg_management"
		)
```

### AI Orchestration Integration

```python
# APG ai_orchestration service integration
from ...ai_orchestration.service import AIOrchestrationService

class ESGAIIntegration:
	def __init__(self, ai_service: AIOrchestrationService):
		self.ai_service = ai_service
	
	async def predict_sustainability_metrics(
		self,
		tenant_id: str,
		metric_data: dict[str, Any],
		historical_data: list[dict[str, Any]]
	) -> dict[str, Any]:
		"""Predict sustainability metrics using AI"""
		return await self.ai_service.predict(
			model_name="sustainability_prediction",
			input_data={
				"tenant_id": tenant_id,
				"metric_data": metric_data,
				"historical_data": historical_data
			},
			model_config={
				"prediction_horizon": "12_months",
				"confidence_level": 0.9,
				"include_scenarios": True
			}
		)
```

### Real-Time Collaboration Integration

```python
# APG real_time_collaboration service integration
from ...real_time_collaboration.service import RealTimeCollaborationService

class ESGCollaborationIntegration:
	def __init__(self, collab_service: RealTimeCollaborationService):
		self.collab_service = collab_service
	
	async def broadcast_esg_update(
		self,
		tenant_id: str,
		channel: str,
		message: dict[str, Any]
	) -> None:
		"""Broadcast ESG updates to stakeholders"""
		await self.collab_service.broadcast_update(
			tenant_id=tenant_id,
			channel=f"esg_{channel}",
			message={
				**message,
				"timestamp": datetime.utcnow().isoformat(),
				"source": "esg_management"
			}
		)
	
	async def notify_stakeholders(
		self,
		tenant_id: str,
		stakeholder_ids: list[str],
		notification: dict[str, Any]
	) -> None:
		"""Send targeted notifications to stakeholders"""
		await self.collab_service.send_targeted_notification(
			tenant_id=tenant_id,
			user_ids=stakeholder_ids,
			notification={
				**notification,
				"category": "esg_update",
				"priority": "normal"
			}
		)
```

---

## Database Schema

### Core Tables

#### ESG Tenants
```sql
CREATE TABLE esg_tenants (
	id VARCHAR(36) PRIMARY KEY,
	name VARCHAR(255) NOT NULL,
	slug VARCHAR(128) UNIQUE NOT NULL,
	description TEXT,
	industry VARCHAR(128),
	headquarters_country VARCHAR(3),
	employee_count INTEGER,
	annual_revenue DECIMAL(20,2),
	esg_frameworks JSONB DEFAULT '[]',
	ai_enabled BOOLEAN DEFAULT TRUE,
	ai_configuration JSONB DEFAULT '{}',
	settings JSONB DEFAULT '{}',
	subscription_tier VARCHAR(32) DEFAULT 'standard',
	is_active BOOLEAN DEFAULT TRUE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(36) NOT NULL,
	updated_by VARCHAR(36) NOT NULL
);

CREATE INDEX idx_esg_tenants_active ON esg_tenants(is_active);
CREATE INDEX idx_esg_tenants_industry ON esg_tenants(industry);
```

#### ESG Metrics
```sql
CREATE TABLE esg_metrics (
	id VARCHAR(36) PRIMARY KEY,
	tenant_id VARCHAR(36) NOT NULL REFERENCES esg_tenants(id),
	framework_id VARCHAR(36) REFERENCES esg_frameworks(id),
	name VARCHAR(255) NOT NULL,
	code VARCHAR(64) NOT NULL,
	metric_type VARCHAR(32) NOT NULL,
	category VARCHAR(128) NOT NULL,
	subcategory VARCHAR(128),
	description TEXT,
	calculation_method TEXT,
	data_sources JSONB DEFAULT '[]',
	unit VARCHAR(32) NOT NULL,
	current_value DECIMAL(20,6),
	target_value DECIMAL(20,6),
	baseline_value DECIMAL(20,6),
	measurement_period VARCHAR(32),
	is_kpi BOOLEAN DEFAULT FALSE,
	is_public BOOLEAN DEFAULT FALSE,
	is_automated BOOLEAN DEFAULT FALSE,
	automation_config JSONB DEFAULT '{}',
	ai_predictions JSONB DEFAULT '{}',
	trend_analysis JSONB DEFAULT '{}',
	data_quality_score DECIMAL(5,2),
	validation_rules JSONB DEFAULT '[]',
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(36) NOT NULL,
	updated_by VARCHAR(36) NOT NULL,
	
	CONSTRAINT uk_esg_metrics_tenant_code UNIQUE(tenant_id, code)
);

CREATE INDEX idx_esg_metrics_tenant ON esg_metrics(tenant_id);
CREATE INDEX idx_esg_metrics_type ON esg_metrics(metric_type);
CREATE INDEX idx_esg_metrics_category ON esg_metrics(category);
CREATE INDEX idx_esg_metrics_kpi ON esg_metrics(is_kpi) WHERE is_kpi = TRUE;
```

#### ESG Measurements
```sql
CREATE TABLE esg_measurements (
	id VARCHAR(36) PRIMARY KEY,
	metric_id VARCHAR(36) NOT NULL REFERENCES esg_metrics(id),
	value DECIMAL(20,6) NOT NULL,
	measurement_date TIMESTAMP WITH TIME ZONE NOT NULL,
	period_start TIMESTAMP WITH TIME ZONE,
	period_end TIMESTAMP WITH TIME ZONE,
	data_source VARCHAR(128),
	collection_method VARCHAR(64),
	metadata JSONB DEFAULT '{}',
	validation_score DECIMAL(5,2),
	anomaly_score DECIMAL(5,2),
	quality_flags JSONB DEFAULT '[]',
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
	created_by VARCHAR(36) NOT NULL,
	updated_by VARCHAR(36) NOT NULL
);

CREATE INDEX idx_esg_measurements_metric ON esg_measurements(metric_id);
CREATE INDEX idx_esg_measurements_date ON esg_measurements(measurement_date);
CREATE INDEX idx_esg_measurements_period ON esg_measurements(period_start, period_end);
```

### Multi-Tenant Patterns

#### Row-Level Security (RLS)
```sql
-- Enable RLS on all ESG tables
ALTER TABLE esg_tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE esg_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE esg_measurements ENABLE ROW LEVEL SECURITY;

-- Create tenant isolation policies
CREATE POLICY esg_tenant_isolation ON esg_metrics
	FOR ALL TO application_role
	USING (tenant_id = current_setting('app.current_tenant_id'));

CREATE POLICY esg_measurement_isolation ON esg_measurements
	FOR ALL TO application_role
	USING (
		metric_id IN (
			SELECT id FROM esg_metrics 
			WHERE tenant_id = current_setting('app.current_tenant_id')
		)
	);
```

#### Tenant Context Management
```python
class TenantContextManager:
	def __init__(self, db_session: Session, tenant_id: str):
		self.db_session = db_session
		self.tenant_id = tenant_id
	
	def __enter__(self):
		"""Set tenant context for database session"""
		self.db_session.execute(
			text("SET app.current_tenant_id = :tenant_id"),
			{"tenant_id": self.tenant_id}
		)
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""Clear tenant context"""
		self.db_session.execute(
			text("RESET app.current_tenant_id")
		)

# Usage in service layer
async def get_tenant_metrics(self, tenant_id: str) -> list[ESGMetric]:
	with TenantContextManager(self.db_session, tenant_id):
		return self.db_session.query(ESGMetric).all()
```

### Performance Optimization

#### Indexing Strategy
```sql
-- Composite indexes for common query patterns
CREATE INDEX idx_esg_metrics_tenant_type_category 
ON esg_metrics(tenant_id, metric_type, category);

CREATE INDEX idx_esg_measurements_metric_date_desc 
ON esg_measurements(metric_id, measurement_date DESC);

-- Partial indexes for filtered queries
CREATE INDEX idx_esg_metrics_active_kpi 
ON esg_metrics(tenant_id, updated_at) 
WHERE is_kpi = TRUE AND is_active = TRUE;

-- JSON indexes for metadata queries
CREATE INDEX idx_esg_metrics_ai_predictions_gin 
ON esg_metrics USING GIN (ai_predictions);
```

#### Partitioning for Scale
```sql
-- Partition measurements by date for better performance
CREATE TABLE esg_measurements_y2024 PARTITION OF esg_measurements
	FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE esg_measurements_y2025 PARTITION OF esg_measurements
	FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

---

## API Reference

### Authentication

All API endpoints require APG authentication via JWT tokens:

```bash
# Get authentication token from APG platform
curl -X POST https://api.apg.platform/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user@example.com","password":"password"}'

# Use token in ESG API requests
curl -X GET https://api.apg.platform/api/v1/esg/metrics \
  -H "Authorization: Bearer <jwt_token>"
```

### Core Endpoints

#### ESG Metrics API

**Create Metric**
```bash
POST /api/v1/esg/metrics
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Carbon Emissions Scope 1",
  "code": "CARBON_SCOPE1",
  "metric_type": "environmental",
  "category": "emissions",
  "subcategory": "direct_emissions",
  "description": "Direct GHG emissions from owned sources",
  "unit": "tonnes_co2",
  "target_value": 10000.0,
  "baseline_value": 15000.0,
  "is_kpi": true,
  "is_automated": true,
  "enable_ai_predictions": true
}
```

**Get Metrics**
```bash
GET /api/v1/esg/metrics?metric_type=environmental&is_kpi=true&limit=50
Authorization: Bearer <token>

Response:
[
  {
    "id": "metric_123",
    "name": "Carbon Emissions Scope 1", 
    "code": "CARBON_SCOPE1",
    "metric_type": "environmental",
    "current_value": 12500.75,
    "target_value": 10000.0,
    "data_quality_score": 94.5,
    "ai_predictions": {
      "predicted_6_month": 11800.50,
      "confidence": 0.89
    },
    "created_at": "2024-01-15T10:30:00Z"
  }
]
```

**Record Measurement**
```bash
POST /api/v1/esg/metrics/{metric_id}/measurements
Content-Type: application/json
Authorization: Bearer <token>

{
  "value": 1250.75,
  "measurement_date": "2024-01-15T10:30:00Z",
  "data_source": "iot_sensors",
  "collection_method": "automated",
  "metadata": {
    "sensor_id": "ENV_001",
    "location": "facility_main"
  }
}
```

#### ESG Targets API

**Create Target**
```bash
POST /api/v1/esg/targets
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "Carbon Reduction 2025",
  "metric_id": "metric_123",
  "description": "Reduce carbon emissions by 33%",
  "target_value": 10000.0,
  "baseline_value": 15000.0,
  "start_date": "2024-01-01T00:00:00Z",
  "target_date": "2025-12-31T23:59:59Z",
  "priority": "high",
  "owner_id": "user_456",
  "create_milestones": true
}
```

**Get Target Prediction**
```bash
GET /api/v1/esg/targets/{target_id}/prediction
Authorization: Bearer <token>

Response:
{
  "status": "success",
  "target_id": "target_789",
  "prediction": {
    "probability": 82.5,
    "predicted_completion_date": "2025-10-15",
    "risk_factors": ["energy_price_volatility", "supply_chain_disruption"],
    "recommendations": [
      {
        "action": "increase_renewable_energy",
        "impact": 0.35,
        "cost": 2500000
      }
    ]
  }
}
```

#### Real-Time WebSocket API

**Connect to WebSocket**
```javascript
// JavaScript client example
const ws = new WebSocket('wss://api.apg.platform/api/v1/esg/ws/tenant_123');

ws.onopen = function() {
  // Subscribe to metric updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['esg_metrics_updates', 'esg_alerts']
  }));
};

ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  if (message.type === 'metric_update') {
    updateDashboard(message.data);
  }
};
```

**Server-Sent Events**
```bash
GET /api/v1/esg/stream/metrics/{metric_id}?interval_seconds=5
Authorization: Bearer <token>
Accept: text/event-stream

Response:
data: {"timestamp":"2024-01-15T10:30:00Z","metric_id":"metric_123","current_value":1250.75}

data: {"timestamp":"2024-01-15T10:30:05Z","metric_id":"metric_123","current_value":1251.20}
```

### Error Handling

**Standard Error Response**
```json
{
  "status": "error",
  "message": "Validation error",
  "error_code": 422,
  "details": [
    {
      "field": "target_value",
      "message": "Must be a positive number"
    }
  ],
  "request_id": "req_123456"
}
```

**Common HTTP Status Codes**
- `200` - Success
- `201` - Created
- `400` - Bad Request (validation error)
- `401` - Unauthorized (invalid/missing token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `422` - Unprocessable Entity (validation error)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

---

## Service Layer

### Service Architecture

```python
# Base service class with APG integration
class BaseESGService:
	def __init__(
		self,
		db_session: Session,
		tenant_id: str,
		config: ESGServiceConfig
	):
		self.db_session = db_session
		self.tenant_id = tenant_id
		self.config = config
		
		# APG service integrations
		self.auth_service = self._get_auth_service()
		self.audit_service = self._get_audit_service()
		self.ai_service = self._get_ai_service()
		self.collaboration_service = self._get_collaboration_service()
	
	async def _check_permission(
		self,
		user_id: str,
		resource: str,
		action: str
	) -> None:
		"""Check user permissions via APG auth_rbac"""
		has_permission = await self.auth_service.check_permission(
			user_id=user_id,
			tenant_id=self.tenant_id,
			resource=resource,
			action=action
		)
		
		if not has_permission:
			raise PermissionError(
				f"User {user_id} lacks {action} permission for {resource}"
			)
	
	async def _log_activity(
		self,
		user_id: str,
		action: str,
		resource_type: str,
		resource_id: str | None = None,
		details: dict[str, Any] | None = None
	) -> str:
		"""Log activity via APG audit_compliance"""
		return await self.audit_service.log_activity(
			user_id=user_id,
			tenant_id=self.tenant_id,
			action=action,
			resource_type=resource_type,
			resource_id=resource_id,
			details=details or {}
		)
```

### ESG Metrics Service

```python
class ESGMetricsService(BaseESGService):
	async def create_metric(
		self,
		user_id: str,
		metric_data: dict[str, Any]
	) -> ESGMetric:
		"""Create ESG metric with validation and AI integration"""
		
		# Permission check
		await self._check_permission(user_id, "esg_metrics", "create")
		
		# Validate input data
		validation_errors = self._validate_metric_data(metric_data)
		if validation_errors:
			raise ValueError(f"Validation errors: {validation_errors}")
		
		# Create metric instance
		metric = ESGMetric(
			id=uuid7str(),
			tenant_id=self.tenant_id,
			name=metric_data["name"],
			code=metric_data["code"],
			metric_type=ESGMetricType(metric_data["metric_type"]),
			category=metric_data["category"],
			unit=ESGMetricUnit(metric_data["unit"]),
			created_by=user_id,
			updated_by=user_id,
			**{k: v for k, v in metric_data.items() 
			   if k not in ["name", "code", "metric_type", "category", "unit"]}
		)
		
		# Database transaction
		try:
			self.db_session.add(metric)
			self.db_session.commit()
			self.db_session.refresh(metric)
		except IntegrityError as e:
			self.db_session.rollback()
			raise ValueError(f"Database constraint violation: {e}")
		
		# Initialize AI predictions if enabled
		if metric_data.get("enable_ai_predictions", False) and self.config.ai_enabled:
			try:
				await self._initialize_metric_ai_predictions(metric.id, user_id)
			except Exception as e:
				# Log error but don't fail metric creation
				await self._log_activity(
					user_id=user_id,
					action="ai_prediction_error",
					resource_type="esg_metric",
					resource_id=metric.id,
					details={"error": str(e)}
				)
		
		# Audit logging
		await self._log_activity(
			user_id=user_id,
			action="create_metric",
			resource_type="esg_metric",
			resource_id=metric.id,
			details={
				"metric_name": metric.name,
				"metric_code": metric.code,
				"metric_type": metric.metric_type.value
			}
		)
		
		# Real-time collaboration update
		if self.config.real_time_processing:
			await self.collaboration_service.broadcast_update(
				tenant_id=self.tenant_id,
				channel="esg_metrics_updates",
				message={
					"type": "metric_created",
					"metric_id": metric.id,
					"user_id": user_id,
					"timestamp": datetime.utcnow()
				}
			)
		
		return metric
	
	async def _initialize_metric_ai_predictions(
		self,
		metric_id: str,
		user_id: str
	) -> dict[str, Any]:
		"""Initialize AI predictions for metric"""
		
		# Get metric and historical data
		metric = self.db_session.query(ESGMetric).filter_by(id=metric_id).first()
		if not metric:
			raise ValueError(f"Metric {metric_id} not found")
		
		historical_data = self._get_metric_historical_data(metric_id)
		
		# Call AI service for predictions
		predictions = await self.ai_service.predict_metric_trends(
			metric_data=self._serialize_metric_for_ai(metric),
			historical_data=historical_data
		)
		
		# Update metric with predictions
		metric.ai_predictions = predictions
		self.db_session.commit()
		
		return predictions
```

### Background Task Integration

```python
# Celery task example for background processing
from celery import Celery

celery_app = Celery('esg_management')

@celery_app.task
async def process_bulk_measurements(
	tenant_id: str,
	metric_id: str,
	measurements_data: list[dict[str, Any]]
) -> dict[str, Any]:
	"""Process bulk measurements in background"""
	
	service = ESGMetricsService(
		db_session=get_database_session(),
		tenant_id=tenant_id,
		config=ESGServiceConfig()
	)
	
	results = {
		"processed": 0,
		"errors": [],
		"total": len(measurements_data)
	}
	
	for measurement_data in measurements_data:
		try:
			await service.record_measurement(
				user_id=measurement_data["user_id"],
				measurement_data=measurement_data
			)
			results["processed"] += 1
		except Exception as e:
			results["errors"].append({
				"measurement": measurement_data,
				"error": str(e)
			})
	
	return results
```

---

## Frontend Development

### Flask-AppBuilder Integration

```python
# views.py - Executive Dashboard View
class ESGExecutiveDashboardView(BaseView):
	route_base = "/esg/executive"
	default_view = "dashboard"
	
	@expose("/")
	@expose("/dashboard")
	@has_access
	async def dashboard(self):
		"""Executive ESG dashboard with AI insights"""
		
		tenant_id = self.get_user_tenant_id()
		
		# Initialize ESG service
		esg_service = ESGManagementService(
			db_session=self.appbuilder.get_session,
			tenant_id=tenant_id,
			config=ESGServiceConfig(ai_enabled=True)
		)
		
		# Get dashboard data
		dashboard_data = await self._get_executive_dashboard_data(
			esg_service, 
			self.get_user_id()
		)
		
		return self.render_template(
			'esg/executive_dashboard.html',
			dashboard_data=dashboard_data,
			tenant_id=tenant_id,
			real_time_enabled=True
		)
	
	async def _get_executive_dashboard_data(
		self, 
		esg_service: ESGManagementService, 
		user_id: str
	) -> dict[str, Any]:
		"""Gather comprehensive dashboard data"""
		
		# Parallel data gathering for performance
		key_metrics_task = esg_service.get_metrics(
			user_id=user_id,
			filters={"is_kpi": True, "limit": 10}
		)
		
		active_targets_task = self._get_active_targets(esg_service, user_id)
		stakeholder_summary_task = self._get_stakeholder_summary(esg_service, user_id)
		ai_insights_task = self._get_ai_insights(esg_service, user_id)
		
		# Await all tasks
		key_metrics, active_targets, stakeholder_summary, ai_insights = await asyncio.gather(
			key_metrics_task,
			active_targets_task,
			stakeholder_summary_task,
			ai_insights_task
		)
		
		return {
			"key_metrics": [self._serialize_metric(m) for m in key_metrics],
			"active_targets": active_targets,
			"stakeholder_summary": stakeholder_summary,
			"ai_insights": ai_insights,
			"last_updated": datetime.utcnow().isoformat()
		}
```

### Custom Widgets

```python
# Custom dashboard widget
class ESGDashboardWidget(ListWidget):
	template = 'esg/widgets/dashboard_widget.html'
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.include_charts = kwargs.get('include_charts', True)
		self.include_kpis = kwargs.get('include_kpis', True)
		self.real_time_updates = kwargs.get('real_time_updates', True)

# Widget template (dashboard_widget.html)
{% extends "appbuilder/general/widgets/list.html" %}

{% block content %}
<div class="esg-dashboard-widget" 
     data-real-time="{{ widget.real_time_updates|lower }}"
     data-tenant-id="{{ g.user.tenant_id }}">
  
  {% if widget.include_kpis %}
  <div class="kpi-section">
    <h4>Key Performance Indicators</h4>
    <div class="row">
      {% for metric in data.key_metrics %}
      <div class="col-md-3">
        <div class="kpi-card" data-metric-id="{{ metric.id }}">
          <h5>{{ metric.name }}</h5>
          <div class="kpi-value">
            {{ metric.current_value }}
            <span class="unit">{{ metric.unit }}</span>
          </div>
          <div class="kpi-trend trend-{{ metric.trend }}">
            <i class="fa fa-arrow-{{ 'up' if metric.trend == 'improving' else 'down' }}"></i>
            {{ metric.trend|title }}
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
  
  {% if widget.include_charts %}
  <div class="charts-section">
    <div class="row">
      <div class="col-md-6">
        <canvas id="esg-trend-chart"></canvas>
      </div>
      <div class="col-md-6">
        <canvas id="esg-target-progress"></canvas>
      </div>
    </div>
  </div>
  {% endif %}
  
</div>

<script>
// Real-time updates via WebSocket
if (document.querySelector('.esg-dashboard-widget').dataset.realTime === 'true') {
  const tenantId = document.querySelector('.esg-dashboard-widget').dataset.tenantId;
  const ws = new WebSocket(`wss://${window.location.host}/api/v1/esg/ws/${tenantId}`);
  
  ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    if (message.type === 'metric_update') {
      updateMetricDisplay(message.data);
    }
  };
}
</script>
{% endblock %}
```

### Progressive Web App (PWA)

```javascript
// Service Worker for offline functionality
// sw.js
const CACHE_NAME = 'esg-management-v1.0.0';
const urlsToCache = [
  '/static/esg/css/dashboard.css',
  '/static/esg/js/dashboard.js',
  '/static/esg/js/charts.js',
  '/api/v1/esg/metrics',
  '/esg/executive/dashboard'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        // Cache hit - return response
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});

// PWA Manifest
// manifest.json
{
  "name": "APG ESG Management",
  "short_name": "ESG Manager",
  "description": "Revolutionary ESG management platform",
  "start_url": "/esg/executive/dashboard",
  "display": "standalone",
  "background_color": "#2E7D32",
  "theme_color": "#4CAF50",
  "icons": [
    {
      "src": "/static/icons/esg-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/icons/esg-512.png", 
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

---

## Testing Framework

### Test Structure

```python
# tests/conftest.py - Test configuration and fixtures
import pytest
import asyncio
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..models import Base, ESGTenant, ESGMetric
from ..service import ESGManagementService, ESGServiceConfig

# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
	"""Create fresh database session for each test"""
	Base.metadata.create_all(bind=engine)
	session = TestSessionLocal()
	try:
		yield session
	finally:
		session.rollback()
		session.close()
		Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def sample_tenant(db_session):
	"""Create sample tenant for testing"""
	tenant = ESGTenant(
		id="test_tenant",
		name="Test Corporation",
		slug="test-corp",
		industry="technology",
		ai_enabled=True,
		created_by="test_user",
		updated_by="test_user"
	)
	db_session.add(tenant)
	db_session.commit()
	db_session.refresh(tenant)
	return tenant

@pytest.fixture(scope="function")
def mock_esg_service(db_session):
	"""Create ESG service with mocked dependencies"""
	service = ESGManagementService(
		db_session=db_session,
		tenant_id="test_tenant",
		config=ESGServiceConfig()
	)
	
	# Mock APG service integrations
	service.auth_service = Mock()
	service.audit_service = Mock()
	service.ai_service = Mock()
	
	# Configure mock behaviors
	service.auth_service.check_permission = AsyncMock(return_value=True)
	service.audit_service.log_activity = AsyncMock(return_value="audit_id")
	service.ai_service.predict_metric_trends = AsyncMock(return_value={
		"predictions": {"6_month": 95.5},
		"confidence": 0.89
	})
	
	return service
```

### Unit Tests

```python
# tests/test_service.py - Service layer tests
class TestESGManagementService:
	@pytest.mark.asyncio
	async def test_create_metric_success(self, mock_esg_service):
		"""Test successful metric creation"""
		metric_data = {
			"name": "Test Metric",
			"code": "TEST_METRIC",
			"metric_type": "environmental",
			"category": "energy",
			"unit": "kwh",
			"enable_ai_predictions": True
		}
		
		with patch('uuid_extensions.uuid7str', return_value="test_metric_id"):
			metric = await mock_esg_service.create_metric(
				user_id="test_user",
				metric_data=metric_data
			)
		
		# Verify metric creation
		assert metric.name == "Test Metric"
		assert metric.code == "TEST_METRIC"
		
		# Verify APG integrations were called
		mock_esg_service.auth_service.check_permission.assert_called()
		mock_esg_service.audit_service.log_activity.assert_called()
		mock_esg_service.ai_service.predict_metric_trends.assert_called()
	
	@pytest.mark.asyncio
	async def test_permission_denied(self, mock_esg_service):
		"""Test permission denied scenario"""
		mock_esg_service.auth_service.check_permission.return_value = False
		
		with pytest.raises(PermissionError):
			await mock_esg_service.create_metric(
				user_id="unauthorized_user",
				metric_data={"name": "Test"}
			)
```

### Integration Tests

```python
# tests/test_integration.py - End-to-end integration tests
@pytest.mark.integration
class TestESGWorkflowIntegration:
	@pytest.mark.asyncio
	async def test_complete_metric_lifecycle(self, db_session, sample_tenant):
		"""Test complete metric lifecycle from creation to reporting"""
		
		# Create service with real database
		service = ESGManagementService(
			db_session=db_session,
			tenant_id=sample_tenant.id,
			config=ESGServiceConfig()
		)
		
		# Mock only external dependencies
		service.auth_service = Mock()
		service.auth_service.check_permission = AsyncMock(return_value=True)
		
		# Step 1: Create metric
		metric_data = {
			"name": "Integration Test Metric",
			"code": "INT_TEST",
			"metric_type": "environmental",
			"category": "energy",
			"unit": "kwh"
		}
		
		metric = await service.create_metric("test_user", metric_data)
		assert metric.id is not None
		
		# Step 2: Record measurements
		measurement_data = {
			"metric_id": metric.id,
			"value": "1250.75",
			"measurement_date": datetime.utcnow()
		}
		
		measurement = await service.record_measurement("test_user", measurement_data)
		assert measurement.value == Decimal("1250.75")
		
		# Step 3: Verify database state
		saved_metric = db_session.query(ESGMetric).filter_by(id=metric.id).first()
		assert saved_metric is not None
		assert saved_metric.name == "Integration Test Metric"
```

### Performance Tests

```python
# tests/test_performance.py - Performance and load testing
@pytest.mark.performance
class TestESGPerformance:
	def test_bulk_metric_creation_performance(self, mock_esg_service, performance_timer):
		"""Test bulk metric creation performance"""
		
		performance_timer.start()
		
		# Create 100 metrics
		tasks = []
		for i in range(100):
			metric_data = {
				"name": f"Performance Test Metric {i}",
				"code": f"PERF_TEST_{i:03d}",
				"metric_type": "environmental",
				"category": "performance",
				"unit": "count"
			}
			
			task = mock_esg_service.create_metric("test_user", metric_data)
			tasks.append(task)
		
		# Execute all tasks
		asyncio.run(asyncio.gather(*tasks))
		
		performance_timer.stop()
		
		# Should complete in under 2 seconds
		performance_timer.assert_max_time(2.0, "Bulk creation too slow")
```

### API Tests

```python
# tests/test_api.py - API endpoint testing
from fastapi.testclient import TestClient
from ..api import app

class TestESGAPI:
	@pytest.fixture
	def api_client(self):
		return TestClient(app)
	
	def test_metrics_endpoint(self, api_client, mock_auth_headers):
		"""Test metrics API endpoint"""
		response = api_client.get(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers
		)
		
		assert response.status_code == 200
		assert isinstance(response.json(), list)
	
	def test_create_metric_validation(self, api_client, mock_auth_headers):
		"""Test metric creation validation"""
		invalid_data = {
			"name": "",  # Invalid: empty name
			"code": "invalid code",  # Invalid: spaces
			"metric_type": "invalid_type"  # Invalid: unknown type
		}
		
		response = api_client.post(
			"/api/v1/esg/metrics",
			headers=mock_auth_headers,
			json=invalid_data
		)
		
		assert response.status_code == 422
		assert "details" in response.json()
```

### Test Execution

```bash
# Run all tests
uv run pytest -vxs tests/

# Run specific test categories
uv run pytest -vxs tests/ -m "not integration"
uv run pytest -vxs tests/ -m integration
uv run pytest -vxs tests/ -m performance

# Run with coverage
uv run pytest --cov=. --cov-report=html tests/

# Run tests in parallel
uv run pytest -n auto tests/
```

---

## Deployment Guide

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/esg/health || exit 1

# Start application
CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  esg-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APG_DATABASE_URL=postgresql://postgres:password@postgres:5432/apg
      - APG_REDIS_URL=redis://redis:6379
      - APG_ENV=production
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/esg/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=apg
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - esg-api

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: esg-management
  labels:
    app: esg-management
spec:
  replicas: 3
  selector:
    matchLabels:
      app: esg-management
  template:
    metadata:
      labels:
        app: esg-management
    spec:
      containers:
      - name: esg-api
        image: apg/esg-management:latest
        ports:
        - containerPort: 8000
        env:
        - name: APG_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: esg-secrets
              key: database-url
        - name: APG_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: esg-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/esg/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/esg/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: esg-management-service
spec:
  selector:
    app: esg-management
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Configuration

```bash
# .env.production
APG_ENV=production
APG_DATABASE_URL=postgresql://user:pass@db-host:5432/apg_prod
APG_REDIS_URL=redis://redis-host:6379/0

# ESG-specific configuration
ESG_AI_ENABLED=true
ESG_REAL_TIME_PROCESSING=true
ESG_MAX_CONCURRENT_USERS=10000
ESG_DATA_RETENTION_DAYS=2555  # 7 years

# External service configuration
WEATHER_API_KEY=your_weather_api_key
EMISSION_FACTORS_API_URL=https://api.emissionfactors.com
IOT_PLATFORM_URL=https://iot.company.com

# Monitoring and observability
SENTRY_DSN=https://your-sentry-dsn@sentry.io
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
```

### Database Migrations

```python
# alembic/versions/001_initial_esg_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    # Create ESG tenants table
    op.create_table(
        'esg_tenants',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(128), nullable=False, unique=True),
        sa.Column('description', sa.Text),
        sa.Column('industry', sa.String(128)),
        sa.Column('headquarters_country', sa.String(3)),
        sa.Column('employee_count', sa.Integer),
        sa.Column('annual_revenue', sa.Numeric(20, 2)),
        sa.Column('esg_frameworks', postgresql.JSONB(), default=[]),
        sa.Column('ai_enabled', sa.Boolean, default=True),
        sa.Column('ai_configuration', postgresql.JSONB(), default={}),
        sa.Column('settings', postgresql.JSONB(), default={}),
        sa.Column('subscription_tier', sa.String(32), default='standard'),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('created_by', sa.String(36), nullable=False),
        sa.Column('updated_by', sa.String(36), nullable=False)
    )
    
    # Create indexes
    op.create_index('idx_esg_tenants_active', 'esg_tenants', ['is_active'])
    op.create_index('idx_esg_tenants_industry', 'esg_tenants', ['industry'])

def downgrade():
    op.drop_table('esg_tenants')
```

---

## Performance Optimization

### Database Optimization

```python
# Database connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Query optimization with proper indexing
class OptimizedESGService:
    async def get_metrics_optimized(
        self,
        tenant_id: str,
        filters: dict[str, Any]
    ) -> list[ESGMetric]:
        """Optimized metric retrieval with efficient queries"""
        
        query = self.db_session.query(ESGMetric)\
            .filter(ESGMetric.tenant_id == tenant_id)
        
        # Use indexes efficiently
        if filters.get("metric_type"):
            query = query.filter(ESGMetric.metric_type == filters["metric_type"])
        
        if filters.get("is_kpi"):
            query = query.filter(ESGMetric.is_kpi == True)
        
        # Limit results and use pagination
        limit = min(filters.get("limit", 50), 1000)
        offset = filters.get("offset", 0)
        
        return query.order_by(ESGMetric.updated_at.desc())\
                   .limit(limit)\
                   .offset(offset)\
                   .all()
```

### Caching Strategy

```python
import redis
from functools import wraps

# Redis configuration
redis_client = redis.Redis(
    host='redis-host',
    port=6379,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

def cache_result(expiration: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"esg:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# Usage in service methods
class ESGMetricsService:
    @cache_result(expiration=600)  # Cache for 10 minutes
    async def get_dashboard_metrics(self, tenant_id: str) -> dict[str, Any]:
        """Get dashboard metrics with caching"""
        return await self._fetch_dashboard_data(tenant_id)
```

### Async Performance

```python
# Parallel data processing
import asyncio
from concurrent.futures import ThreadPoolExecutor

class HighPerformanceESGService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def process_bulk_data_parallel(
        self,
        data_batches: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Process multiple data batches in parallel"""
        
        # Create tasks for parallel processing
        tasks = [
            self._process_batch_async(batch) 
            for batch in data_batches
        ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
            else:
                processed_results.extend(result)
        
        return processed_results
    
    async def _process_batch_async(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process single batch asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Use thread executor for CPU-intensive work
        return await loop.run_in_executor(
            self.executor,
            self._process_batch_sync,
            batch
        )
```

### Memory Optimization

```python
# Streaming data processing for large datasets
async def stream_large_dataset(
    self,
    query: Query,
    batch_size: int = 1000
) -> AsyncGenerator[list[ESGMetric], None]:
    """Stream large datasets to avoid memory issues"""
    
    offset = 0
    while True:
        batch = query.offset(offset).limit(batch_size).all()
        if not batch:
            break
        
        yield batch
        offset += batch_size
        
        # Explicitly clear session to free memory
        self.db_session.expunge_all()

# Usage
async def process_all_metrics(self):
    query = self.db_session.query(ESGMetric)
    
    async for metric_batch in self.stream_large_dataset(query):
        # Process batch
        await self._process_metric_batch(metric_batch)
```

---

## Security Guidelines

### Authentication & Authorization

```python
# Secure authentication with APG integration
from jose import JWTError, jwt
from passlib.context import CryptContext

class ESGSecurityManager:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = os.getenv("APG_SECRET_KEY")
        self.algorithm = "HS256"
    
    async def verify_token(self, token: str) -> dict[str, Any]:
        """Verify JWT token with APG integration"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            tenant_id = payload.get("tenant_id")
            
            if not user_id or not tenant_id:
                raise JWTError("Invalid token payload")
            
            return {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "permissions": payload.get("permissions", [])
            }
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
    
    async def check_esg_permission(
        self,
        user_data: dict[str, Any],
        resource: str,
        action: str
    ) -> bool:
        """Check ESG-specific permissions"""
        required_permission = f"esg:{resource}:{action}"
        return required_permission in user_data.get("permissions", [])
```

### Data Encryption

```python
from cryptography.fernet import Fernet

class ESGDataEncryption:
    def __init__(self):
        self.encryption_key = os.getenv("ESG_ENCRYPTION_KEY").encode()
        self.fernet = Fernet(self.encryption_key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive ESG data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive ESG data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# Model with encrypted fields
class ESGStakeholder(Base):
    __tablename__ = "esg_stakeholders"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    _encrypted_email = Column("email", String(500))  # Encrypted field
    
    @hybrid_property
    def email(self):
        if self._encrypted_email:
            return ESGDataEncryption().decrypt_sensitive_data(self._encrypted_email)
        return None
    
    @email.setter
    def email(self, value):
        if value:
            self._encrypted_email = ESGDataEncryption().encrypt_sensitive_data(value)
        else:
            self._encrypted_email = None
```

### Input Validation & Sanitization

```python
from bleach import clean
from pydantic import validator

class SecureESGMetricRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    
    @validator('name', 'description')
    def sanitize_text_fields(cls, v):
        """Sanitize text input to prevent XSS"""
        if v is None:
            return v
        
        # Remove HTML tags and dangerous content
        cleaned = clean(
            v,
            tags=[],  # No HTML tags allowed
            attributes={},
            protocols=[],
            strip=True
        )
        
        return cleaned.strip()
    
    @validator('name')
    def validate_name_content(cls, v):
        """Additional validation for metric names"""
        # Prevent SQL injection patterns
        dangerous_patterns = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError(f"Name contains prohibited content: {pattern}")
        
        return v
```

### Audit Logging

```python
class ESGSecurityAuditLogger:
    def __init__(self, audit_service):
        self.audit_service = audit_service
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: str,
        tenant_id: str,
        details: dict[str, Any],
        risk_level: str = "low"
    ) -> None:
        """Log security-related events"""
        
        security_event = {
            "event_type": event_type,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "risk_level": risk_level,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": self._get_client_ip(),
            "user_agent": self._get_user_agent(),
            **details
        }
        
        await self.audit_service.log_activity(
            user_id=user_id,
            tenant_id=tenant_id,
            action="security_event",
            resource_type="security",
            details=security_event,
            category="security"
        )
        
        # Send alert for high-risk events
        if risk_level in ["high", "critical"]:
            await self._send_security_alert(security_event)
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
esg_requests_total = Counter(
    'esg_requests_total',
    'Total ESG API requests',
    ['method', 'endpoint', 'status']
)

esg_request_duration = Histogram(
    'esg_request_duration_seconds',
    'ESG API request duration',
    ['method', 'endpoint']
)

esg_active_metrics = Gauge(
    'esg_active_metrics_total',
    'Total number of active ESG metrics',
    ['tenant_id', 'metric_type']
)

# Middleware for automatic metrics collection
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    method = request.method
    endpoint = request.url.path
    status = str(response.status_code)
    
    esg_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    esg_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response

# Custom business metrics
class ESGMetricsCollector:
    async def collect_business_metrics(self, tenant_id: str):
        """Collect ESG-specific business metrics"""
        
        # Count active metrics by type
        metric_counts = self.db_session.query(
            ESGMetric.metric_type,
            func.count(ESGMetric.id)
        ).filter(
            ESGMetric.tenant_id == tenant_id,
            ESGMetric.is_active == True
        ).group_by(ESGMetric.metric_type).all()
        
        for metric_type, count in metric_counts:
            esg_active_metrics.labels(
                tenant_id=tenant_id,
                metric_type=metric_type.value
            ).set(count)
```

### Structured Logging

```python
import structlog
from pythonjsonlogger import jsonlogger

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ESGStructuredLogger:
    def __init__(self, service_name: str = "esg_management"):
        self.logger = logger.bind(service=service_name)
    
    def _log_esg_operation(
        self,
        operation: str,
        user_id: str,
        tenant_id: str,
        resource_id: str | None = None,
        duration_ms: float | None = None,
        **kwargs
    ):
        """Log ESG operations with structured data"""
        log_data = {
            "operation": operation,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "resource_id": resource_id,
            "duration_ms": duration_ms,
            **kwargs
        }
        
        self.logger.info("ESG operation completed", **log_data)
    
    async def log_metric_creation(
        self,
        user_id: str,
        tenant_id: str,
        metric_id: str,
        metric_name: str,
        duration_ms: float
    ):
        """Log metric creation with performance data"""
        self._log_esg_operation(
            operation="create_metric",
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=metric_id,
            duration_ms=duration_ms,
            metric_name=metric_name
        )
```

### Health Checks

```python
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = None

class ESGHealthChecker:
    def __init__(self, db_session: Session):
        self.db_session = db_session
    
    async def check_health(self) -> dict[str, Any]:
        """Comprehensive health check for ESG capability"""
        
        checks = [
            await self._check_database(),
            await self._check_redis(),
            await self._check_external_apis(),
            await self._check_ai_services()
        ]
        
        # Determine overall health
        overall_status = HealthStatus.HEALTHY
        if any(check.status == HealthStatus.UNHEALTHY for check in checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.DEGRADED for check in checks):
            overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "checks": {check.name: {
                "status": check.status.value,
                "message": check.message,
                "details": check.details or {}
            } for check in checks},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_database(self) -> HealthCheck:
        """Check database connectivity and performance"""
        try:
            start_time = time.time()
            result = self.db_session.execute("SELECT 1").scalar()
            duration_ms = (time.time() - start_time) * 1000
            
            if result == 1:
                if duration_ms < 100:
                    return HealthCheck(
                        name="database",
                        status=HealthStatus.HEALTHY,
                        message="Database connection healthy",
                        details={"response_time_ms": duration_ms}
                    )
                else:
                    return HealthCheck(
                        name="database", 
                        status=HealthStatus.DEGRADED,
                        message="Database responding slowly",
                        details={"response_time_ms": duration_ms}
                    )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database query returned unexpected result"
                )
        
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )
```

### Error Tracking

```python
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Initialize Sentry for error tracking
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FastApiIntegration(auto_enabling_integrations=False),
        SqlalchemyIntegration(),
    ],
    traces_sample_rate=0.1,
    environment=os.getenv("APG_ENV", "development")
)

class ESGErrorHandler:
    @staticmethod
    def capture_esg_exception(
        exception: Exception,
        context: dict[str, Any] = None
    ):
        """Capture ESG-specific exceptions with context"""
        
        with sentry_sdk.configure_scope() as scope:
            # Add ESG-specific context
            scope.set_tag("capability", "esg_management")
            scope.set_context("esg", context or {})
            
            # Capture exception
            sentry_sdk.capture_exception(exception)
    
    @staticmethod
    def capture_performance_issue(
        operation: str,
        duration_ms: float,
        threshold_ms: float = 1000
    ):
        """Capture performance issues"""
        
        if duration_ms > threshold_ms:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("performance_issue", True)
                scope.set_context("performance", {
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "threshold_ms": threshold_ms
                })
                
                sentry_sdk.capture_message(
                    f"Performance threshold exceeded: {operation}",
                    level="warning"
                )
```

---

## Conclusion

This developer guide provides comprehensive technical documentation for the APG Sustainability & ESG Management capability. The platform leverages modern async Python patterns, comprehensive APG ecosystem integration, and enterprise-grade security to deliver revolutionary ESG management capabilities.

For additional technical support and advanced integration scenarios, please consult the APG platform documentation or contact the development team.

---

**Copyright © 2025 Datacraft - All rights reserved.**  
**Author: Nyimbi Odero <nyimbi@gmail.com>**  
**Website: www.datacraft.co.ke**