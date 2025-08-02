# APG Workflow Orchestration - Developer Guide

**Comprehensive guide for developers building and extending the workflow orchestration platform**

© 2025 Datacraft. All rights reserved.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Database Design](#database-design)
5. [API Development](#api-development)
6. [Component Development](#component-development)
7. [Connector Development](#connector-development)
8. [Template Development](#template-development)
9. [Testing Strategies](#testing-strategies)
10. [Performance Optimization](#performance-optimization)
11. [Security Implementation](#security-implementation)
12. [Deployment & CI/CD](#deployment--cicd)
13. [Troubleshooting & Debugging](#troubleshooting--debugging)
14. [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

### Prerequisites

**System Requirements:**
- Python 3.9+ with asyncio support
- Node.js 16+ for UI components
- PostgreSQL 12+ with JSON support
- Redis 6+ for caching and queues
- Git for version control

**Development Tools:**
- Code editor with Python support (VS Code, PyCharm)
- Docker and Docker Compose
- Postman or curl for API testing
- Browser developer tools

### Local Development Setup

1. **Clone Repository**
```bash
git clone https://github.com/datacraft/apg-workflow-orchestration.git
cd apg-workflow-orchestration
```

2. **Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Node.js Environment**
```bash
cd static/
npm install
npm run build:dev
```

4. **Database Setup**
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Create database schema
python -m alembic upgrade head

# Load sample data
python scripts/load_sample_data.py
```

5. **Configuration**
```bash
# Copy development configuration
cp config/config.dev.yaml config/config.yaml

# Set environment variables
export APG_ENV=development
export DEBUG=true
export LOG_LEVEL=DEBUG
```

6. **Start Development Server**
```bash
# Start with auto-reload
python service.py --development --reload

# Or use the development script
./scripts/dev_server.sh
```

### Development Workflow

**Code Style and Standards:**
- Follow PEP 8 for Python code
- Use async/await throughout for I/O operations
- Tabs for indentation (not spaces)
- Modern type hints: `str | None`, `list[str]`, `dict[str, Any]`
- Pydantic v2 models with strict validation

**Git Workflow:**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
# Commit with descriptive messages
git commit -m "feat: add workflow validation service"

# Push and create pull request
git push origin feature/your-feature-name
```

**Testing Before Commit:**
```bash
# Run all tests
uv run pytest -vxs tests/ci

# Type checking
uv run pyright

# Code formatting
black .
isort .
```

## Architecture Overview

### System Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        UI[React UI Components]
        API[Flask-AppBuilder Views]
    end
    
    subgraph "API Layer"
        REST[REST APIs]
        GraphQL[GraphQL API]
        WS[WebSocket API]
    end
    
    subgraph "Service Layer"
        WS_SVC[Workflow Service]
        EXEC_SVC[Execution Service]
        TMPL_SVC[Template Service]
        CONN_SVC[Connector Service]
    end
    
    subgraph "Engine Layer"
        WF_ENGINE[Workflow Engine]
        SCHED[Task Scheduler]
        STATE[State Manager]
        OPT[Optimization Engine]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL)]
        REDIS[(Redis)]
        APG[APG Framework]
    end
    
    UI --> API
    API --> REST
    REST --> WS_SVC
    WS_SVC --> WF_ENGINE
    WF_ENGINE --> PG
    SCHED --> REDIS
```

### Core Design Patterns

**Async/Await Pattern:**
```python
class WorkflowService(APGBaseService):
	"""Async service following APG patterns."""
	
	async def create_workflow(self, workflow_data: WorkflowCreate) -> Workflow:
		"""Create new workflow with validation."""
		# Validate input
		await self._validate_workflow_data(workflow_data)
		
		# Create in database
		async with self.db_session() as session:
			workflow = Workflow(**workflow_data.model_dump())
			session.add(workflow)
			await session.commit()
			await session.refresh(workflow)
			
		# Post-creation actions
		await self._setup_workflow_environment(workflow)
		await self._audit_workflow_creation(workflow)
		
		return workflow
```

**Service Pattern:**
```python
class BaseWorkflowService(APGBaseService):
	"""Base service for workflow operations."""
	
	def __init__(self):
		super().__init__()
		self.db = get_database()
		self.cache = get_redis()
		self.event_bus = get_event_bus()
	
	async def _validate_permissions(self, user_id: str, resource_id: str, action: str) -> bool:
		"""Validate user permissions for resource action."""
		return await self.auth_service.check_permission(user_id, resource_id, action)
	
	async def _audit_action(self, user_id: str, action: str, resource_id: str, details: dict):
		"""Log action for audit trail."""
		await self.audit_service.log_action(user_id, action, resource_id, details)
```

**Factory Pattern for Components:**
```python
class ComponentFactory:
	"""Factory for creating workflow components."""
	
	_components: dict[str, type[BaseWorkflowComponent]] = {}
	
	@classmethod
	def register(cls, component_type: str, component_class: type[BaseWorkflowComponent]):
		"""Register component type."""
		cls._components[component_type] = component_class
	
	@classmethod
	def create(cls, component_type: str, config: dict) -> BaseWorkflowComponent:
		"""Create component instance."""
		if component_type not in cls._components:
			raise ValueError(f"Unknown component type: {component_type}")
		
		component_class = cls._components[component_type]
		return component_class(config)
```

## Core Components

### Workflow Engine

The workflow engine is the heart of the orchestration system:

```python
class WorkflowEngine(APGBaseService):
	"""Core workflow execution engine."""
	
	def __init__(self):
		super().__init__()
		self.scheduler = TaskScheduler()
		self.state_manager = StateManager()
		self.executor = WorkflowExecutor()
	
	async def execute_workflow(self, workflow_id: str, parameters: dict) -> str:
		"""Execute workflow and return execution ID."""
		# Load workflow definition
		workflow = await self._load_workflow(workflow_id)
		
		# Create execution context
		execution = await self._create_execution(workflow, parameters)
		
		# Schedule for execution
		await self.scheduler.schedule_workflow(execution)
		
		return execution.id
	
	async def _execute_component(self, component: WorkflowComponent, context: ExecutionContext) -> ExecutionResult:
		"""Execute single workflow component."""
		try:
			# Pre-execution hooks
			await self._pre_execution_hook(component, context)
			
			# Execute component
			result = await component.execute(context.input_data, context.variables)
			
			# Post-execution hooks
			await self._post_execution_hook(component, context, result)
			
			return result
			
		except Exception as e:
			# Handle component failure
			return await self._handle_component_failure(component, context, e)
```

### State Management

Persistent state management with recovery capabilities:

```python
class StateManager(APGBaseService):
	"""Manages workflow execution state."""
	
	async def save_checkpoint(self, execution_id: str, component_id: str, state_data: dict):
		"""Save execution checkpoint."""
		checkpoint = ExecutionCheckpoint(
			execution_id=execution_id,
			component_id=component_id,
			state_data=state_data,
			created_at=datetime.utcnow()
		)
		
		async with self.db_session() as session:
			session.add(checkpoint)
			await session.commit()
		
		# Cache for quick access
		await self.cache.setex(
			f"checkpoint:{execution_id}:{component_id}",
			3600,
			json.dumps(state_data)
		)
	
	async def restore_from_checkpoint(self, execution_id: str, component_id: str) -> dict | None:
		"""Restore execution state from checkpoint."""
		# Try cache first
		cached = await self.cache.get(f"checkpoint:{execution_id}:{component_id}")
		if cached:
			return json.loads(cached)
		
		# Fallback to database
		async with self.db_session() as session:
			checkpoint = await session.get(ExecutionCheckpoint, {
				'execution_id': execution_id,
				'component_id': component_id
			})
			return checkpoint.state_data if checkpoint else None
```

### Task Scheduler

Intelligent task scheduling with priority and resource management:

```python
class TaskScheduler(APGBaseService):
	"""Intelligent task scheduler with priority queuing."""
	
	def __init__(self):
		super().__init__()
		self.high_priority_queue = "workflow:queue:high"
		self.normal_priority_queue = "workflow:queue:normal"
		self.low_priority_queue = "workflow:queue:low"
		self.workers = []
	
	async def schedule_workflow(self, execution: WorkflowExecution):
		"""Schedule workflow for execution."""
		# Determine priority queue
		queue = self._get_priority_queue(execution.priority)
		
		# Create task
		task = {
			'execution_id': execution.id,
			'workflow_id': execution.workflow_id,
			'priority': execution.priority,
			'scheduled_at': datetime.utcnow().isoformat(),
			'parameters': execution.parameters
		}
		
		# Add to queue
		await self.cache.lpush(queue, json.dumps(task))
		
		# Update execution status
		await self._update_execution_status(execution.id, ExecutionStatus.QUEUED)
	
	async def start_workers(self, worker_count: int = 5):
		"""Start background workers."""
		for i in range(worker_count):
			worker = WorkflowWorker(f"worker-{i}", self)
			self.workers.append(worker)
			asyncio.create_task(worker.run())
```

## Database Design

### Schema Design

**Core Tables:**
```sql
-- Workflows table
CREATE TABLE wo_workflows (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	name VARCHAR(255) NOT NULL,
	description TEXT,
	definition JSONB NOT NULL,
	status VARCHAR(50) NOT NULL DEFAULT 'draft',
	category VARCHAR(100),
	tags TEXT[],
	version INTEGER DEFAULT 1,
	created_by UUID NOT NULL,
	created_at TIMESTAMP DEFAULT NOW(),
	updated_at TIMESTAMP DEFAULT NOW(),
	
	-- Indexes
	CONSTRAINT wo_workflows_name_key UNIQUE (name, created_by)
);

CREATE INDEX idx_wo_workflows_status ON wo_workflows(status);
CREATE INDEX idx_wo_workflows_category ON wo_workflows(category);
CREATE INDEX idx_wo_workflows_created_by ON wo_workflows(created_by);
CREATE INDEX idx_wo_workflows_tags ON wo_workflows USING GIN(tags);

-- Workflow executions table
CREATE TABLE wo_executions (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	workflow_id UUID REFERENCES wo_workflows(id) ON DELETE CASCADE,
	status VARCHAR(50) NOT NULL DEFAULT 'pending',
	priority INTEGER DEFAULT 5,
	parameters JSONB,
	results JSONB,
	error_message TEXT,
	started_at TIMESTAMP,
	completed_at TIMESTAMP,
	created_at TIMESTAMP DEFAULT NOW(),
	
	-- Partitioning by month
	PARTITION BY RANGE (created_at)
);

-- Component executions for detailed tracking
CREATE TABLE wo_component_executions (
	id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
	execution_id UUID REFERENCES wo_executions(id) ON DELETE CASCADE,
	component_id VARCHAR(255) NOT NULL,
	component_type VARCHAR(100) NOT NULL,
	status VARCHAR(50) NOT NULL,
	input_data JSONB,
	output_data JSONB,
	error_message TEXT,
	duration_ms INTEGER,
	started_at TIMESTAMP,
	completed_at TIMESTAMP
);
```

**Pydantic Models:**
```python
class WorkflowCreate(BaseModel):
	"""Pydantic model for workflow creation."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., min_length=1, max_length=255)
	description: str | None = Field(None, max_length=1000)
	definition: dict[str, Any] = Field(..., description="Workflow definition JSON")
	category: str | None = Field(None, max_length=100)
	tags: list[str] = Field(default_factory=list)
	priority: int = Field(default=5, ge=1, le=10)
	
	@field_validator('name')
	@classmethod
	def validate_name(cls, v: str) -> str:
		if not v.strip():
			raise ValueError('Name cannot be empty')
		return v.strip()

class Workflow(BaseModel):
	"""Workflow model with full details."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str
	description: str | None = None
	definition: dict[str, Any]
	status: WorkflowStatus = WorkflowStatus.DRAFT
	category: str | None = None
	tags: list[str] = Field(default_factory=list)
	version: int = 1
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	
	# Computed properties
	@computed_field
	@property
	def component_count(self) -> int:
		return len(self.definition.get('components', []))
```

### Migration Management

```python
# migrations/versions/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
	"""Create initial workflow orchestration schema."""
	# Create workflows table
	op.create_table('wo_workflows',
		sa.Column('id', postgresql.UUID(), nullable=False),
		sa.Column('name', sa.String(255), nullable=False),
		sa.Column('description', sa.Text()),
		sa.Column('definition', postgresql.JSONB(), nullable=False),
		sa.Column('status', sa.String(50), nullable=False),
		sa.Column('category', sa.String(100)),
		sa.Column('tags', postgresql.ARRAY(sa.Text())),
		sa.Column('version', sa.Integer(), nullable=False),
		sa.Column('created_by', postgresql.UUID(), nullable=False),
		sa.Column('created_at', sa.DateTime(), nullable=False),
		sa.Column('updated_at', sa.DateTime(), nullable=False),
		sa.PrimaryKeyConstraint('id')
	)
	
	# Create indexes
	op.create_index('idx_wo_workflows_status', 'wo_workflows', ['status'])
	op.create_index('idx_wo_workflows_category', 'wo_workflows', ['category'])
	op.create_index('idx_wo_workflows_created_by', 'wo_workflows', ['created_by'])
	op.create_index('idx_wo_workflows_tags', 'wo_workflows', ['tags'], postgresql_using='gin')

def downgrade():
	"""Drop workflow orchestration schema."""
	op.drop_table('wo_workflows')
```

## API Development

### REST API Design

**Base API Controller:**
```python
class BaseWorkflowAPI(APGBaseView):
	"""Base API controller for workflow endpoints."""
	
	def __init__(self):
		super().__init__()
		self.workflow_service = WorkflowService()
		self.execution_service = ExecutionService()
	
	async def _validate_request(self, request_data: dict, model_class: type[BaseModel]) -> BaseModel:
		"""Validate request data using Pydantic model."""
		try:
			return model_class(**request_data)
		except ValidationError as e:
			raise ValidationException(f"Invalid request data: {e}")
	
	async def _check_permissions(self, resource_id: str, action: str):
		"""Check user permissions for resource action."""
		user_id = self.get_current_user_id()
		if not await self.auth_service.check_permission(user_id, resource_id, action):
			raise PermissionDeniedException(f"Insufficient permissions for {action}")

@workflow_bp.route('/workflows', methods=['POST'])
async def create_workflow():
	"""Create new workflow."""
	try:
		# Validate request
		request_data = await request.get_json()
		workflow_data = await api._validate_request(request_data, WorkflowCreate)
		
		# Check permissions
		await api._check_permissions('workflows', 'create')
		
		# Create workflow
		workflow = await api.workflow_service.create_workflow(workflow_data)
		
		# Return response
		return jsonify({
			'success': True,
			'data': workflow.model_dump(),
			'message': 'Workflow created successfully'
		}), 201
		
	except ValidationException as e:
		return jsonify({'success': False, 'error': str(e)}), 400
	except PermissionDeniedException as e:
		return jsonify({'success': False, 'error': str(e)}), 403
	except Exception as e:
		logger.error(f"Error creating workflow: {e}")
		return jsonify({'success': False, 'error': 'Internal server error'}), 500
```

### GraphQL API

**GraphQL Schema:**
```python
import strawberry
from typing import Optional

@strawberry.type
class Workflow:
	id: str
	name: str
	description: Optional[str]
	status: str
	category: Optional[str]
	tags: list[str]
	created_at: datetime
	updated_at: datetime

@strawberry.type
class WorkflowExecution:
	id: str
	workflow_id: str
	status: str
	priority: int
	started_at: Optional[datetime]
	completed_at: Optional[datetime]
	duration: Optional[int]

@strawberry.type
class Query:
	@strawberry.field
	async def workflows(
		self,
		limit: Optional[int] = 50,
		offset: Optional[int] = 0,
		category: Optional[str] = None
	) -> list[Workflow]:
		"""Get workflows with optional filtering."""
		workflow_service = WorkflowService()
		return await workflow_service.list_workflows(
			limit=limit,
			offset=offset,
			filters={'category': category} if category else {}
		)
	
	@strawberry.field
	async def workflow(self, id: str) -> Optional[Workflow]:
		"""Get workflow by ID."""
		workflow_service = WorkflowService()
		return await workflow_service.get_workflow(id)

@strawberry.type
class Mutation:
	@strawberry.mutation
	async def create_workflow(
		self,
		name: str,
		description: Optional[str] = None,
		definition: str = "{}",
		category: Optional[str] = None
	) -> Workflow:
		"""Create new workflow."""
		workflow_service = WorkflowService()
		
		workflow_data = WorkflowCreate(
			name=name,
			description=description,
			definition=json.loads(definition),
			category=category
		)
		
		return await workflow_service.create_workflow(workflow_data)

# Create schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
```

### WebSocket API

**Real-time Updates:**
```python
class WorkflowWebSocketManager:
	"""Manages WebSocket connections for real-time updates."""
	
	def __init__(self):
		self.connections: dict[str, set[WebSocket]] = {}
		self.user_connections: dict[str, set[WebSocket]] = {}
	
	async def connect(self, websocket: WebSocket, user_id: str):
		"""Connect user to WebSocket."""
		await websocket.accept()
		
		if user_id not in self.user_connections:
			self.user_connections[user_id] = set()
		self.user_connections[user_id].add(websocket)
	
	async def disconnect(self, websocket: WebSocket, user_id: str):
		"""Disconnect user from WebSocket."""
		if user_id in self.user_connections:
			self.user_connections[user_id].discard(websocket)
			if not self.user_connections[user_id]:
				del self.user_connections[user_id]
	
	async def subscribe_to_workflow(self, websocket: WebSocket, workflow_id: str):
		"""Subscribe to workflow updates."""
		if workflow_id not in self.connections:
			self.connections[workflow_id] = set()
		self.connections[workflow_id].add(websocket)
	
	async def broadcast_workflow_update(self, workflow_id: str, update_data: dict):
		"""Broadcast workflow update to subscribers."""
		if workflow_id in self.connections:
			message = json.dumps({
				'type': 'workflow_update',
				'workflow_id': workflow_id,
				'data': update_data,
				'timestamp': datetime.utcnow().isoformat()
			})
			
			for websocket in self.connections[workflow_id].copy():
				try:
					await websocket.send_text(message)
				except ConnectionClosedError:
					self.connections[workflow_id].discard(websocket)
```

## Component Development

### Base Component Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, Field

class ComponentConfig(BaseModel):
	"""Base configuration for workflow components."""
	model_config = ConfigDict(extra='forbid')
	
	name: str = Field(..., min_length=1)
	description: str | None = None
	timeout_seconds: int = Field(default=300, ge=1, le=3600)
	retry_attempts: int = Field(default=3, ge=0, le=10)
	retry_delay_seconds: int = Field(default=60, ge=1, le=300)

class ExecutionResult(BaseModel):
	"""Result of component execution."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool
	output_data: Any = None
	error_message: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	duration_ms: int | None = None

class BaseWorkflowComponent(ABC):
	"""Base class for all workflow components."""
	
	def __init__(self, config: ComponentConfig):
		self.config = config
		self.logger = logging.getLogger(self.__class__.__name__)
	
	@abstractmethod
	async def execute(self, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute the component logic."""
		pass
	
	@abstractmethod
	def validate_config(self) -> bool:
		"""Validate component configuration."""
		pass
	
	async def _handle_retry(self, func, *args, **kwargs) -> ExecutionResult:
		"""Handle retries for component execution."""
		last_error = None
		
		for attempt in range(self.config.retry_attempts + 1):
			try:
				result = await func(*args, **kwargs)
				if result.success:
					return result
				last_error = result.error_message
				
			except Exception as e:
				last_error = str(e)
				self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
			
			if attempt < self.config.retry_attempts:
				await asyncio.sleep(self.config.retry_delay_seconds)
		
		return ExecutionResult(
			success=False,
			error_message=f"Component failed after {self.config.retry_attempts + 1} attempts: {last_error}"
		)
```

### Custom Component Example

```python
class DataTransformComponent(BaseWorkflowComponent):
	"""Component for data transformation operations."""
	
	class Config(ComponentConfig):
		transformation_type: str = Field(..., regex=r'^(map|filter|aggregate|sort)$')
		transformation_rules: dict[str, Any]
		output_format: str = Field(default='json')
	
	def __init__(self, config: Config):
		super().__init__(config)
		self.config: DataTransformComponent.Config = config
	
	def validate_config(self) -> bool:
		"""Validate transformation configuration."""
		if not self.config.transformation_rules:
			raise ValueError("Transformation rules cannot be empty")
		
		if self.config.transformation_type == 'map' and 'mapping' not in self.config.transformation_rules:
			raise ValueError("Map transformation requires 'mapping' in rules")
		
		return True
	
	async def execute(self, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute data transformation."""
		start_time = time.time()
		
		try:
			# Validate input data
			if not isinstance(input_data, (list, dict)):
				raise ValueError("Input data must be list or dict")
			
			# Apply transformation
			result_data = await self._apply_transformation(input_data)
			
			# Format output
			formatted_output = await self._format_output(result_data)
			
			duration_ms = int((time.time() - start_time) * 1000)
			
			return ExecutionResult(
				success=True,
				output_data=formatted_output,
				metadata={
					'transformation_type': self.config.transformation_type,
					'input_count': len(input_data) if isinstance(input_data, list) else 1,
					'output_count': len(formatted_output) if isinstance(formatted_output, list) else 1
				},
				duration_ms=duration_ms
			)
			
		except Exception as e:
			duration_ms = int((time.time() - start_time) * 1000)
			return ExecutionResult(
				success=False,
				error_message=str(e),
				duration_ms=duration_ms
			)
	
	async def _apply_transformation(self, data: Any) -> Any:
		"""Apply the configured transformation."""
		if self.config.transformation_type == 'map':
			return await self._apply_mapping(data)
		elif self.config.transformation_type == 'filter':
			return await self._apply_filter(data)
		elif self.config.transformation_type == 'aggregate':
			return await self._apply_aggregation(data)
		elif self.config.transformation_type == 'sort':
			return await self._apply_sort(data)
		else:
			raise ValueError(f"Unsupported transformation type: {self.config.transformation_type}")
	
	async def _apply_mapping(self, data: Any) -> Any:
		"""Apply field mapping transformation."""
		mapping = self.config.transformation_rules['mapping']
		
		if isinstance(data, list):
			return [self._map_item(item, mapping) for item in data]
		else:
			return self._map_item(data, mapping)
	
	def _map_item(self, item: dict, mapping: dict) -> dict:
		"""Map individual item fields."""
		result = {}
		for source_field, target_field in mapping.items():
			if source_field in item:
				result[target_field] = item[source_field]
		return result
```

### Component Registration

```python
class ComponentRegistry:
	"""Registry for workflow components."""
	
	_components: dict[str, type[BaseWorkflowComponent]] = {}
	_validators: dict[str, callable] = {}
	
	@classmethod
	def register(
		cls,
		component_type: str,
		component_class: type[BaseWorkflowComponent],
		validator: callable | None = None
	):
		"""Register component type."""
		cls._components[component_type] = component_class
		if validator:
			cls._validators[component_type] = validator
	
	@classmethod
	def create(cls, component_type: str, config: dict) -> BaseWorkflowComponent:
		"""Create component instance."""
		if component_type not in cls._components:
			raise ValueError(f"Unknown component type: {component_type}")
		
		component_class = cls._components[component_type]
		config_class = getattr(component_class, 'Config', ComponentConfig)
		
		# Validate configuration
		validated_config = config_class(**config)
		
		# Create component
		component = component_class(validated_config)
		component.validate_config()
		
		return component
	
	@classmethod
	def list_components(cls) -> list[str]:
		"""List available component types."""
		return list(cls._components.keys())

# Register built-in components
ComponentRegistry.register('transform', DataTransformComponent)
ComponentRegistry.register('http_request', HTTPRequestComponent)
ComponentRegistry.register('database_query', DatabaseQueryComponent)
ComponentRegistry.register('email', EmailComponent)
ComponentRegistry.register('file_operation', FileOperationComponent)
```

## Connector Development

### APG Connector Framework

```python
class APGConnectorBase(BaseWorkflowComponent):
	"""Base class for APG capability connectors."""
	
	class Config(ComponentConfig):
		capability_name: str = Field(..., min_length=1)
		operation: str = Field(..., min_length=1)
		parameters: dict[str, Any] = Field(default_factory=dict)
		use_cache: bool = Field(default=True)
		cache_ttl_seconds: int = Field(default=300, ge=0)
	
	def __init__(self, config: Config):
		super().__init__(config)
		self.config: APGConnectorBase.Config = config
		self.apg_client = self._get_apg_client()
	
	def _get_apg_client(self):
		"""Get APG client instance."""
		from apg.client import APGClient
		return APGClient()
	
	async def execute(self, input_data: Any, context: dict[str, Any]) -> ExecutionResult:
		"""Execute APG capability operation."""
		start_time = time.time()
		
		try:
			# Prepare parameters
			operation_params = self._prepare_parameters(input_data, context)
			
			# Check cache if enabled
			if self.config.use_cache:
				cached_result = await self._get_cached_result(operation_params)
				if cached_result:
					return cached_result
			
			# Execute APG operation
			result = await self._execute_apg_operation(operation_params)
			
			# Cache result if successful
			if self.config.use_cache and result.success:
				await self._cache_result(operation_params, result)
			
			duration_ms = int((time.time() - start_time) * 1000)
			result.duration_ms = duration_ms
			
			return result
			
		except Exception as e:
			duration_ms = int((time.time() - start_time) * 1000)
			return ExecutionResult(
				success=False,
				error_message=str(e),
				duration_ms=duration_ms
			)
	
	async def _execute_apg_operation(self, parameters: dict) -> ExecutionResult:
		"""Execute the APG capability operation."""
		response = await self.apg_client.call_capability(
			capability=self.config.capability_name,
			operation=self.config.operation,
			parameters=parameters
		)
		
		if response.success:
			return ExecutionResult(
				success=True,
				output_data=response.data,
				metadata={'apg_response_id': response.id}
			)
		else:
			return ExecutionResult(
				success=False,
				error_message=response.error_message
			)
	
	def _prepare_parameters(self, input_data: Any, context: dict[str, Any]) -> dict:
		"""Prepare parameters for APG operation."""
		params = self.config.parameters.copy()
		
		# Merge input data if it's a dict
		if isinstance(input_data, dict):
			params.update(input_data)
		else:
			params['input'] = input_data
		
		# Add context variables
		params['context'] = context
		
		return params
```

### Specific APG Connectors

```python
class UserManagementConnector(APGConnectorBase):
	"""Connector for APG User Management capability."""
	
	class Config(APGConnectorBase.Config):
		capability_name: str = Field(default='user_management')
		operation: str = Field(..., regex=r'^(create|update|delete|get|list|authenticate)$')
		user_fields: list[str] = Field(default_factory=lambda: ['id', 'username', 'email'])
	
	async def _execute_apg_operation(self, parameters: dict) -> ExecutionResult:
		"""Execute user management operation."""
		if self.config.operation == 'create':
			return await self._create_user(parameters)
		elif self.config.operation == 'update':
			return await self._update_user(parameters)
		elif self.config.operation == 'delete':
			return await self._delete_user(parameters)
		elif self.config.operation == 'get':
			return await self._get_user(parameters)
		elif self.config.operation == 'list':
			return await self._list_users(parameters)
		elif self.config.operation == 'authenticate':
			return await self._authenticate_user(parameters)
		else:
			return ExecutionResult(
				success=False,
				error_message=f"Unsupported operation: {self.config.operation}"
			)
	
	async def _create_user(self, parameters: dict) -> ExecutionResult:
		"""Create new user."""
		required_fields = ['username', 'email', 'password']
		for field in required_fields:
			if field not in parameters:
				return ExecutionResult(
					success=False,
					error_message=f"Missing required field: {field}"
				)
		
		response = await self.apg_client.user_management.create_user(
			username=parameters['username'],
			email=parameters['email'],
			password=parameters['password'],
			additional_data=parameters.get('additional_data', {})
		)
		
		return ExecutionResult(
			success=response.success,
			output_data=response.data if response.success else None,
			error_message=response.error_message if not response.success else None
		)

# Register APG connectors
ComponentRegistry.register('apg_user_management', UserManagementConnector)
ComponentRegistry.register('apg_notification', NotificationConnector)
ComponentRegistry.register('apg_file_management', FileManagementConnector)
ComponentRegistry.register('apg_audit', AuditConnector)
```

## Template Development

### Template Structure

```python
class WorkflowTemplate(BaseModel):
	"""Workflow template model."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=255)
	description: str = Field(..., min_length=1, max_length=1000)
	category: str = Field(..., min_length=1)
	tags: list[str] = Field(default_factory=list)
	
	# Template definition
	workflow_definition: dict[str, Any]
	parameters: list[TemplateParameter] = Field(default_factory=list)
	
	# Metadata
	author: str
	version: str = Field(default='1.0.0')
	min_apg_version: str = Field(default='1.0.0')
	complexity_level: str = Field(default='beginner', regex=r'^(beginner|intermediate|advanced|expert)$')
	estimated_setup_time_minutes: int = Field(default=30, ge=1)
	
	# Usage
	usage_count: int = Field(default=0, ge=0)
	rating: float = Field(default=0.0, ge=0.0, le=5.0)
	
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

class TemplateParameter(BaseModel):
	"""Template parameter definition."""
	model_config = ConfigDict(extra='forbid')
	
	name: str = Field(..., min_length=1)
	display_name: str = Field(..., min_length=1)
	description: str
	parameter_type: str = Field(..., regex=r'^(string|integer|float|boolean|list|dict)$')
	required: bool = Field(default=True)
	default_value: Any = None
	validation_rules: dict[str, Any] = Field(default_factory=dict)
	
	# UI hints
	input_type: str = Field(default='text')  # text, password, select, multiselect, textarea
	options: list[str] = Field(default_factory=list)  # For select inputs
	placeholder: str | None = None
	help_text: str | None = None
```

### Template Creation

```python
class TemplateBuilder:
	"""Builder for creating workflow templates."""
	
	def __init__(self):
		self.template_data = {}
		self.parameters = []
		self.workflow_components = []
	
	def set_basic_info(self, name: str, description: str, category: str, author: str) -> 'TemplateBuilder':
		"""Set basic template information."""
		self.template_data.update({
			'name': name,
			'description': description,
			'category': category,
			'author': author
		})
		return self
	
	def add_parameter(
		self,
		name: str,
		display_name: str,
		description: str,
		parameter_type: str,
		required: bool = True,
		default_value: Any = None,
		**kwargs
	) -> 'TemplateBuilder':
		"""Add parameter to template."""
		parameter = TemplateParameter(
			name=name,
			display_name=display_name,
			description=description,
			parameter_type=parameter_type,
			required=required,
			default_value=default_value,
			**kwargs
		)
		self.parameters.append(parameter)
		return self
	
	def add_component(
		self,
		component_id: str,
		component_type: str,
		config: dict,
		position: dict[str, int] | None = None
	) -> 'TemplateBuilder':
		"""Add component to workflow."""
		component = {
			'id': component_id,
			'type': component_type,
			'config': config,
			'position': position or {'x': 0, 'y': 0}
		}
		self.workflow_components.append(component)
		return self
	
	def add_connection(self, source_id: str, target_id: str, connection_type: str = 'success') -> 'TemplateBuilder':
		"""Add connection between components."""
		if 'connections' not in self.template_data:
			self.template_data['connections'] = []
		
		connection = {
			'source': source_id,
			'target': target_id,
			'type': connection_type
		}
		self.template_data['connections'].append(connection)
		return self
	
	def build(self) -> WorkflowTemplate:
		"""Build the template."""
		workflow_definition = {
			'components': self.workflow_components,
			'connections': self.template_data.get('connections', []),
			'metadata': {
				'created_by_template': True,
				'template_version': self.template_data.get('version', '1.0.0')
			}
		}
		
		template = WorkflowTemplate(
			**self.template_data,
			workflow_definition=workflow_definition,
			parameters=self.parameters
		)
		
		return template

# Example: Data Processing Pipeline Template
def create_data_processing_template() -> WorkflowTemplate:
	"""Create a data processing pipeline template."""
	builder = TemplateBuilder()
	
	template = (builder
		.set_basic_info(
			name="Data Processing Pipeline",
			description="Extract, transform, and load data from various sources",
			category="data_processing",
			author="APG System"
		)
		.add_parameter(
			name="source_url",
			display_name="Data Source URL",
			description="URL or path to the data source",
			parameter_type="string",
			required=True,
			input_type="url",
			placeholder="https://api.example.com/data"
		)
		.add_parameter(
			name="output_format",
			display_name="Output Format",
			description="Format for processed data",
			parameter_type="string",
			required=True,
			input_type="select",
			options=["json", "csv", "parquet"],
			default_value="json"
		)
		.add_parameter(
			name="batch_size",
			display_name="Batch Size",
			description="Number of records to process in each batch",
			parameter_type="integer",
			required=False,
			default_value=1000,
			validation_rules={"min": 1, "max": 10000}
		)
		.add_component(
			component_id="start",
			component_type="start",
			config={"trigger_type": "manual"},
			position={"x": 100, "y": 100}
		)
		.add_component(
			component_id="extract",
			component_type="http_request",
			config={
				"url": "{{source_url}}",
				"method": "GET",
				"timeout": 30
			},
			position={"x": 300, "y": 100}
		)
		.add_component(
			component_id="transform",
			component_type="transform",
			config={
				"transformation_type": "map",
				"transformation_rules": {
					"mapping": {
						"id": "record_id",
						"name": "record_name",
						"value": "record_value"
					}
				},
				"batch_size": "{{batch_size}}"
			},
			position={"x": 500, "y": 100}
		)
		.add_component(
			component_id="load",
			component_type="file_operation",
			config={
				"operation": "write",
				"file_path": "/tmp/processed_data.{{output_format}}",
				"format": "{{output_format}}"
			},
			position={"x": 700, "y": 100}
		)
		.add_component(
			component_id="end",
			component_type="end",
			config={"status": "success"},
			position={"x": 900, "y": 100}
		)
		.add_connection("start", "extract")
		.add_connection("extract", "transform")
		.add_connection("transform", "load")
		.add_connection("load", "end")
		.build()
	)
	
	return template
```

## Testing Strategies

### Unit Testing

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from workflow_orchestration.models import WorkflowCreate, Workflow
from workflow_orchestration.services import WorkflowService

class TestWorkflowService:
	"""Test suite for WorkflowService."""
	
	@pytest.fixture
	def workflow_service(self):
		"""Create WorkflowService instance for testing."""
		return WorkflowService()
	
	@pytest.fixture
	def sample_workflow_data(self):
		"""Sample workflow data for testing."""
		return WorkflowCreate(
			name="Test Workflow",
			description="Test workflow description",
			definition={
				"components": [
					{"id": "start", "type": "start", "config": {}},
					{"id": "end", "type": "end", "config": {}}
				],
				"connections": [
					{"source": "start", "target": "end", "type": "success"}
				]
			},
			category="test",
			tags=["test", "example"]
		)
	
	@pytest.mark.asyncio
	async def test_create_workflow_success(self, workflow_service, sample_workflow_data):
		"""Test successful workflow creation."""
		# Mock database session
		mock_session = AsyncMock()
		mock_workflow = Workflow(**sample_workflow_data.model_dump(), id="test-id", created_by="user-123")
		
		with patch.object(workflow_service, 'db_session', return_value=mock_session):
			with patch.object(workflow_service, '_validate_workflow_data', return_value=None):
				with patch.object(workflow_service, '_setup_workflow_environment', return_value=None):
					with patch.object(workflow_service, '_audit_workflow_creation', return_value=None):
						mock_session.__aenter__.return_value = mock_session
						mock_session.add = MagicMock()
						mock_session.commit = AsyncMock()
						mock_session.refresh = AsyncMock()
						
						# Execute
						result = await workflow_service.create_workflow(sample_workflow_data)
						
						# Assertions
						assert isinstance(result, Workflow)
						assert result.name == sample_workflow_data.name
						mock_session.add.assert_called_once()
						mock_session.commit.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_create_workflow_validation_error(self, workflow_service):
		"""Test workflow creation with validation error."""
		invalid_data = WorkflowCreate(
			name="",  # Invalid: empty name
			description="Test",
			definition={}
		)
		
		with pytest.raises(ValueError, match="Name cannot be empty"):
			await workflow_service.create_workflow(invalid_data)
	
	@pytest.mark.asyncio
	async def test_execute_workflow(self, workflow_service):
		"""Test workflow execution."""
		workflow_id = "test-workflow-id"
		parameters = {"param1": "value1"}
		
		mock_execution = MagicMock()
		mock_execution.id = "execution-id"
		
		with patch.object(workflow_service.execution_service, 'create_execution', return_value=mock_execution):
			with patch.object(workflow_service.scheduler, 'schedule_workflow', return_value=None):
				result = await workflow_service.execute_workflow(workflow_id, parameters)
				
				assert result == "execution-id"
```

### Integration Testing

```python
@pytest.mark.integration
class TestWorkflowIntegration:
	"""Integration tests for workflow orchestration."""
	
	@pytest.fixture(scope="class")
	async def test_app(self):
		"""Create test application instance."""
		from workflow_orchestration.service import create_app
		
		app = create_app(testing=True)
		
		# Setup test database
		async with app.app_context():
			from workflow_orchestration.database import create_tables
			await create_tables()
		
		yield app
		
		# Cleanup
		async with app.app_context():
			from workflow_orchestration.database import drop_tables
			await drop_tables()
	
	@pytest.fixture
	async def client(self, test_app):
		"""Create test client."""
		return test_app.test_client()
	
	@pytest.mark.asyncio
	async def test_full_workflow_lifecycle(self, client):
		"""Test complete workflow lifecycle."""
		# 1. Create workflow
		workflow_data = {
			"name": "Integration Test Workflow",
			"description": "Test workflow for integration testing",
			"definition": {
				"components": [
					{"id": "start", "type": "start", "config": {"trigger_type": "manual"}},
					{"id": "task", "type": "task", "config": {"task_type": "echo", "message": "Hello World"}},
					{"id": "end", "type": "end", "config": {"status": "success"}}
				],
				"connections": [
					{"source": "start", "target": "task", "type": "success"},
					{"source": "task", "target": "end", "type": "success"}
				]
			},
			"category": "test"
		}
		
		# Create workflow
		response = await client.post('/api/v1/workflows', json=workflow_data)
		assert response.status_code == 201
		workflow = response.get_json()['data']
		workflow_id = workflow['id']
		
		# 2. Execute workflow
		response = await client.post(f'/api/v1/workflows/{workflow_id}/execute', json={})
		assert response.status_code == 200
		execution = response.get_json()['data']
		execution_id = execution['id']
		
		# 3. Wait for completion
		timeout = 30  # seconds
		start_time = time.time()
		
		while time.time() - start_time < timeout:
			response = await client.get(f'/api/v1/executions/{execution_id}')
			assert response.status_code == 200
			execution_status = response.get_json()['data']['status']
			
			if execution_status in ['completed', 'failed', 'cancelled']:
				break
			
			await asyncio.sleep(1)
		
		# 4. Verify completion
		assert execution_status == 'completed'
		
		# 5. Get execution logs
		response = await client.get(f'/api/v1/executions/{execution_id}/logs')
		assert response.status_code == 200
		logs = response.get_json()['data']
		assert len(logs) > 0
		
		# 6. Delete workflow
		response = await client.delete(f'/api/v1/workflows/{workflow_id}')
		assert response.status_code == 200
```

### Performance Testing

```python
@pytest.mark.performance
class TestWorkflowPerformance:
	"""Performance tests for workflow orchestration."""
	
	@pytest.mark.asyncio
	async def test_concurrent_workflow_execution(self):
		"""Test concurrent execution of multiple workflows."""
		from workflow_orchestration.services import WorkflowService, ExecutionService
		
		workflow_service = WorkflowService()
		execution_service = ExecutionService()
		
		# Create test workflow
		workflow_data = WorkflowCreate(
			name="Performance Test Workflow",
			description="Simple workflow for performance testing",
			definition={
				"components": [
					{"id": "start", "type": "start", "config": {}},
					{"id": "delay", "type": "delay", "config": {"seconds": 1}},
					{"id": "end", "type": "end", "config": {}}
				],
				"connections": [
					{"source": "start", "target": "delay"},
					{"source": "delay", "target": "end"}
				]
			}
		)
		
		workflow = await workflow_service.create_workflow(workflow_data)
		
		# Execute multiple workflows concurrently
		concurrent_executions = 50
		start_time = time.time()
		
		tasks = []
		for i in range(concurrent_executions):
			task = asyncio.create_task(
				workflow_service.execute_workflow(workflow.id, {"iteration": i})
			)
			tasks.append(task)
		
		# Wait for all executions to start
		execution_ids = await asyncio.gather(*tasks)
		
		# Wait for all executions to complete
		completion_tasks = []
		for execution_id in execution_ids:
			task = asyncio.create_task(
				execution_service.wait_for_completion(execution_id, timeout=60)
			)
			completion_tasks.append(task)
		
		results = await asyncio.gather(*completion_tasks)
		end_time = time.time()
		
		# Verify results
		successful_executions = sum(1 for result in results if result.status == 'completed')
		total_time = end_time - start_time
		
		print(f"Executed {concurrent_executions} workflows in {total_time:.2f} seconds")
		print(f"Success rate: {successful_executions}/{concurrent_executions} ({successful_executions/concurrent_executions*100:.1f}%)")
		print(f"Average time per execution: {total_time/concurrent_executions:.2f} seconds")
		
		# Performance assertions
		assert successful_executions >= concurrent_executions * 0.95  # 95% success rate
		assert total_time < concurrent_executions * 0.5  # Should be faster than sequential execution
	
	@pytest.mark.asyncio
	async def test_large_workflow_performance(self):
		"""Test performance with large workflows."""
		from workflow_orchestration.services import WorkflowService
		
		# Create workflow with many components
		components = [{"id": "start", "type": "start", "config": {}}]
		connections = []
		
		# Add 1000 task components
		for i in range(1000):
			component_id = f"task_{i}"
			components.append({
				"id": component_id,
				"type": "task",
				"config": {"task_type": "noop"}
			})
			
			# Connect to previous component
			if i == 0:
				connections.append({"source": "start", "target": component_id})
			else:
				connections.append({"source": f"task_{i-1}", "target": component_id})
		
		# Add end component
		components.append({"id": "end", "type": "end", "config": {}})
		connections.append({"source": "task_999", "target": "end"})
		
		workflow_data = WorkflowCreate(
			name="Large Workflow Performance Test",
			description="Workflow with 1000+ components",
			definition={
				"components": components,
				"connections": connections
			}
		)
		
		workflow_service = WorkflowService()
		
		# Test workflow creation performance
		start_time = time.time()
		workflow = await workflow_service.create_workflow(workflow_data)
		creation_time = time.time() - start_time
		
		print(f"Created workflow with {len(components)} components in {creation_time:.2f} seconds")
		
		# Test workflow validation performance
		start_time = time.time()
		validation_result = await workflow_service.validate_workflow(workflow.id)
		validation_time = time.time() - start_time
		
		print(f"Validated workflow in {validation_time:.2f} seconds")
		
		# Performance assertions
		assert creation_time < 5.0  # Should create within 5 seconds
		assert validation_time < 2.0  # Should validate within 2 seconds
		assert validation_result.is_valid
```

## Performance Optimization

### Database Optimization

```python
# Database query optimization
class OptimizedWorkflowService(WorkflowService):
	"""Optimized workflow service with database performance improvements."""
	
	async def list_workflows(
		self,
		limit: int = 50,
		offset: int = 0,
		filters: dict | None = None,
		user_id: str | None = None
	) -> list[Workflow]:
		"""List workflows with optimized query."""
		query = """
		SELECT w.*, 
			   COUNT(e.id) as execution_count,
			   AVG(CASE WHEN e.status = 'completed' THEN 1.0 ELSE 0.0 END) as success_rate,
			   MAX(e.created_at) as last_execution
		FROM wo_workflows w
		LEFT JOIN wo_executions e ON w.id = e.workflow_id
		WHERE 1=1
		"""
		
		params = []
		
		# Add filters
		if filters:
			if 'status' in filters:
				query += " AND w.status = %s"
				params.append(filters['status'])
			
			if 'category' in filters:
				query += " AND w.category = %s"
				params.append(filters['category'])
			
			if 'search' in filters:
				query += " AND (w.name ILIKE %s OR w.description ILIKE %s)"
				search_term = f"%{filters['search']}%"
				params.extend([search_term, search_term])
		
		# Add user filter if provided
		if user_id:
			query += " AND w.created_by = %s"
			params.append(user_id)
		
		# Group by and order
		query += """
		GROUP BY w.id, w.name, w.description, w.definition, w.status, 
				 w.category, w.tags, w.version, w.created_by, w.created_at, w.updated_at
		ORDER BY w.updated_at DESC
		LIMIT %s OFFSET %s
		"""
		params.extend([limit, offset])
		
		async with self.db_session() as session:
			result = await session.execute(text(query), params)
			rows = result.fetchall()
			
			workflows = []
			for row in rows:
				workflow_dict = dict(row._mapping)
				workflow = Workflow(**workflow_dict)
				workflows.append(workflow)
			
			return workflows
	
	async def get_workflow_with_stats(self, workflow_id: str) -> dict:
		"""Get workflow with execution statistics."""
		query = """
		SELECT w.*,
			   COUNT(e.id) as total_executions,
			   COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as successful_executions,
			   COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_executions,
			   AVG(EXTRACT(EPOCH FROM (e.completed_at - e.started_at))) as avg_duration_seconds,
			   MAX(e.completed_at) as last_execution_at
		FROM wo_workflows w
		LEFT JOIN wo_executions e ON w.id = e.workflow_id
		WHERE w.id = %s
		GROUP BY w.id
		"""
		
		async with self.db_session() as session:
			result = await session.execute(text(query), [workflow_id])
			row = result.fetchone()
			
			if not row:
				raise WorkflowNotFoundException(f"Workflow {workflow_id} not found")
			
			return dict(row._mapping)
```

### Caching Strategy

```python
class CachedWorkflowService(WorkflowService):
	"""Workflow service with caching layer."""
	
	def __init__(self):
		super().__init__()
		self.cache_ttl = {
			'workflow': 3600,  # 1 hour
			'execution': 300,  # 5 minutes
			'template': 7200,  # 2 hours
			'user_permissions': 1800  # 30 minutes
		}
	
	async def get_workflow(self, workflow_id: str) -> Workflow | None:
		"""Get workflow with caching."""
		cache_key = f"workflow:{workflow_id}"
		
		# Try cache first
		cached_data = await self.cache.get(cache_key)
		if cached_data:
			return Workflow.model_validate_json(cached_data)
		
		# Fallback to database
		workflow = await super().get_workflow(workflow_id)
		if workflow:
			# Cache the result
			await self.cache.setex(
				cache_key,
				self.cache_ttl['workflow'],
				workflow.model_dump_json()
			)
		
		return workflow
	
	async def invalidate_workflow_cache(self, workflow_id: str):
		"""Invalidate workflow cache."""
		cache_keys = [
			f"workflow:{workflow_id}",
			f"workflow:stats:{workflow_id}",
			"workflows:list:*"  # Invalidate list caches
		]
		
		for key in cache_keys:
			if "*" in key:
				# Handle pattern-based invalidation
				pattern_keys = await self.cache.keys(key)
				if pattern_keys:
					await self.cache.delete(*pattern_keys)
			else:
				await self.cache.delete(key)
	
	async def cache_execution_results(self, execution_id: str, results: dict):
		"""Cache execution results for quick access."""
		cache_key = f"execution:results:{execution_id}"
		await self.cache.setex(
			cache_key,
			self.cache_ttl['execution'],
			json.dumps(results)
		)
```

### Async Optimization

```python
class AsyncOptimizedExecutor:
	"""Optimized async executor for workflow components."""
	
	def __init__(self, max_concurrent_tasks: int = 100):
		self.max_concurrent_tasks = max_concurrent_tasks
		self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
		self.task_pool = set()
	
	async def execute_components_parallel(
		self,
		components: list[WorkflowComponent],
		context: ExecutionContext
	) -> list[ExecutionResult]:
		"""Execute components in parallel with concurrency control."""
		async def execute_with_semaphore(component: WorkflowComponent) -> ExecutionResult:
			async with self.semaphore:
				return await self._execute_component_safe(component, context)
		
		# Create tasks for all components
		tasks = []
		for component in components:
			task = asyncio.create_task(execute_with_semaphore(component))
			tasks.append(task)
			self.task_pool.add(task)
		
		try:
			# Wait for all tasks to complete
			results = await asyncio.gather(*tasks, return_exceptions=True)
			
			# Process results
			execution_results = []
			for i, result in enumerate(results):
				if isinstance(result, Exception):
					execution_results.append(ExecutionResult(
						success=False,
						error_message=str(result)
					))
				else:
					execution_results.append(result)
			
			return execution_results
			
		finally:
			# Clean up task pool
			for task in tasks:
				self.task_pool.discard(task)
	
	async def _execute_component_safe(
		self,
		component: WorkflowComponent,
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute component with error handling and timeout."""
		try:
			# Apply timeout
			return await asyncio.wait_for(
				component.execute(context.input_data, context.variables),
				timeout=component.config.timeout_seconds
			)
		except asyncio.TimeoutError:
			return ExecutionResult(
				success=False,
				error_message=f"Component execution timed out after {component.config.timeout_seconds} seconds"
			)
		except Exception as e:
			return ExecutionResult(
				success=False,
				error_message=f"Component execution failed: {str(e)}"
			)
```

## Security Implementation

### Authentication & Authorization

```python
class WorkflowSecurityManager:
	"""Security manager for workflow operations."""
	
	def __init__(self):
		self.auth_service = get_auth_service()
		self.rbac_service = get_rbac_service()
	
	async def check_workflow_permission(
		self,
		user_id: str,
		workflow_id: str,
		action: str
	) -> bool:
		"""Check if user has permission for workflow action."""
		# Get user roles and permissions
		user_roles = await self.rbac_service.get_user_roles(user_id)
		user_permissions = await self.rbac_service.get_user_permissions(user_id)
		
		# Check direct permission
		required_permission = f"workflow.{action}"
		if required_permission in user_permissions:
			return True
		
		# Check workflow ownership
		workflow = await self._get_workflow(workflow_id)
		if workflow and workflow.created_by == user_id:
			# Owner can perform most actions
			if action in ['read', 'update', 'execute', 'delete']:
				return True
		
		# Check shared permissions
		shared_permissions = await self._get_workflow_shared_permissions(workflow_id)
		if user_id in shared_permissions:
			user_workflow_permissions = shared_permissions[user_id]
			if action in user_workflow_permissions:
				return True
		
		# Check role-based permissions
		for role in user_roles:
			role_permissions = await self.rbac_service.get_role_permissions(role)
			if required_permission in role_permissions:
				return True
		
		return False
	
	async def validate_workflow_security(self, workflow_definition: dict) -> list[str]:
		"""Validate workflow for security issues."""
		security_warnings = []
		
		# Check for sensitive data exposure
		components = workflow_definition.get('components', [])
		for component in components:
			config = component.get('config', {})
			
			# Check for hardcoded secrets
			for key, value in config.items():
				if isinstance(value, str):
					if self._contains_potential_secret(key, value):
						security_warnings.append(
							f"Component {component['id']} may contain hardcoded secret in '{key}'"
						)
			
			# Check for unsafe operations
			if component.get('type') == 'script_execution':
				script_content = config.get('script', '')
				if self._contains_unsafe_operations(script_content):
					security_warnings.append(
						f"Component {component['id']} contains potentially unsafe script operations"
					)
		
		return security_warnings
	
	def _contains_potential_secret(self, key: str, value: str) -> bool:
		"""Check if a value might be a secret."""
		sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
		key_lower = key.lower()
		
		# Check if key name suggests a secret
		if any(sensitive in key_lower for sensitive in sensitive_keys):
			# Check if value looks like a secret (not a template variable)
			if len(value) > 8 and not value.startswith('{{') and not value.startswith('${'):
				return True
		
		return False
	
	def _contains_unsafe_operations(self, script: str) -> bool:
		"""Check if script contains potentially unsafe operations."""
		unsafe_patterns = [
			'eval(',
			'exec(',
			'import os',
			'import subprocess',
			'import sys',
			'__import__',
			'open(',
			'file(',
			'input(',
			'raw_input('
		]
		
		script_lower = script.lower()
		return any(pattern in script_lower for pattern in unsafe_patterns)
```

### Data Encryption

```python
class DataEncryptionService:
	"""Service for encrypting sensitive data in workflows."""
	
	def __init__(self):
		self.encryption_key = self._get_encryption_key()
		self.cipher_suite = Fernet(self.encryption_key)
	
	def _get_encryption_key(self) -> bytes:
		"""Get encryption key from secure storage."""
		key = os.environ.get('WORKFLOW_ENCRYPTION_KEY')
		if not key:
			raise ValueError("WORKFLOW_ENCRYPTION_KEY environment variable not set")
		return key.encode()
	
	def encrypt_sensitive_data(self, data: dict) -> dict:
		"""Encrypt sensitive fields in workflow data."""
		encrypted_data = data.copy()
		
		# Define sensitive fields
		sensitive_fields = ['password', 'secret', 'token', 'api_key', 'private_key']
		
		def encrypt_recursive(obj):
			if isinstance(obj, dict):
				for key, value in obj.items():
					if key.lower() in sensitive_fields and isinstance(value, str):
						# Encrypt the value
						encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
						obj[key] = f"encrypted:{encrypted_value}"
					elif isinstance(value, (dict, list)):
						encrypt_recursive(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						encrypt_recursive(item)
		
		encrypt_recursive(encrypted_data)
		return encrypted_data
	
	def decrypt_sensitive_data(self, data: dict) -> dict:
		"""Decrypt sensitive fields in workflow data."""
		decrypted_data = data.copy()
		
		def decrypt_recursive(obj):
			if isinstance(obj, dict):
				for key, value in obj.items():
					if isinstance(value, str) and value.startswith('encrypted:'):
						# Decrypt the value
						encrypted_value = value[10:]  # Remove 'encrypted:' prefix
						decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
						obj[key] = decrypted_value
					elif isinstance(value, (dict, list)):
						decrypt_recursive(value)
			elif isinstance(obj, list):
				for item in obj:
					if isinstance(item, (dict, list)):
						decrypt_recursive(item)
		
		decrypt_recursive(decrypted_data)
		return decrypted_data
```

## Deployment & CI/CD

### Docker Configuration

```dockerfile
# Multi-stage Dockerfile for production
FROM node:16-alpine AS frontend-builder

WORKDIR /app/static
COPY static/package*.json ./
RUN npm ci --only=production

COPY static/ ./
RUN npm run build:prod

FROM python:3.9-slim AS backend-builder

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim AS production

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies
COPY --from=backend-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy frontend assets
COPY --from=frontend-builder /app/static/dist ./static/dist

# Copy application code
COPY . .

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

CMD ["python", "service.py"]
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: Workflow Orchestration CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: datacraft/workflow-orchestration

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_workflow_orchestration
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:6-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '16'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          node_modules
        key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements.txt', '**/package-lock.json') }}
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Install Node.js dependencies
      run: |
        cd static
        npm ci
    
    - name: Lint Python code
      run: |
        black --check .
        isort --check-only .
        flake8 .
    
    - name: Type checking
      run: pyright
    
    - name: Build frontend
      run: |
        cd static
        npm run build:prod
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/test_workflow_orchestration
        REDIS_URL: redis://localhost:6379/0
        APG_ENV: testing
      run: |
        uv run pytest -vxs tests/ci --cov=workflow_orchestration --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD

  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Add deployment logic here
        echo "Deploying to production..."
```

## Troubleshooting & Debugging

### Logging Configuration

```python
import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
	"""Custom JSON formatter for structured logging."""
	
	def format(self, record):
		log_entry = {
			'timestamp': datetime.utcnow().isoformat(),
			'level': record.levelname,
			'logger': record.name,
			'message': record.getMessage(),
			'module': record.module,
			'function': record.funcName,
			'line': record.lineno
		}
		
		# Add extra fields
		for key, value in record.__dict__.items():
			if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
						   'filename', 'module', 'lineno', 'funcName', 'created', 
						   'msecs', 'relativeCreated', 'thread', 'threadName', 
						   'processName', 'process', 'getMessage']:
				log_entry[key] = value
		
		# Add exception info if present
		if record.exc_info:
			log_entry['exception'] = self.formatException(record.exc_info)
		
		return json.dumps(log_entry)

def setup_logging(log_level: str = 'INFO', log_file: str = 'workflow_orchestration.log'):
	"""Setup logging configuration."""
	logger = logging.getLogger('workflow_orchestration')
	logger.setLevel(getattr(logging, log_level.upper()))
	
	# Console handler
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(logging.INFO)
	console_formatter = logging.Formatter(
		'%(asctime)s [%(levelname)s] %(name)s: %(message)s'
	)
	console_handler.setFormatter(console_formatter)
	logger.addHandler(console_handler)
	
	# File handler with rotation
	file_handler = RotatingFileHandler(
		log_file,
		maxBytes=10*1024*1024,  # 10MB
		backupCount=5
	)
	file_handler.setLevel(logging.DEBUG)
	file_formatter = JSONFormatter()
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)
	
	# Error log handler
	error_handler = TimedRotatingFileHandler(
		'errors.log',
		when='midnight',
		interval=1,
		backupCount=7
	)
	error_handler.setLevel(logging.ERROR)
	error_handler.setFormatter(file_formatter)
	logger.addHandler(error_handler)
	
	return logger
```

### Debug Mode

```python
class WorkflowDebugger:
	"""Debugging utilities for workflow execution."""
	
	def __init__(self):
		self.debug_enabled = os.environ.get('DEBUG', 'false').lower() == 'true'
		self.breakpoints = set()
		self.execution_trace = []
	
	def set_breakpoint(self, component_id: str):
		"""Set breakpoint for component."""
		self.breakpoints.add(component_id)
	
	def clear_breakpoint(self, component_id: str):
		"""Clear breakpoint for component."""
		self.breakpoints.discard(component_id)
	
	async def debug_component_execution(
		self, 
		component: WorkflowComponent, 
		context: ExecutionContext
	) -> ExecutionResult:
		"""Execute component with debugging support."""
		if not self.debug_enabled:
			return await component.execute(context.input_data, context.variables)
		
		# Record execution trace
		trace_entry = {
			'component_id': component.config.name,
			'component_type': component.__class__.__name__,
			'timestamp': datetime.utcnow().isoformat(),
			'input_data': context.input_data,
			'variables': context.variables.copy()
		}
		
		# Check for breakpoint
		if component.config.name in self.breakpoints:
			print(f"🔴 Breakpoint hit: {component.config.name}")
			print(f"Input data: {json.dumps(context.input_data, indent=2)}")
			print(f"Variables: {json.dumps(context.variables, indent=2)}")
			
			# Wait for user input
			while True:
				command = input("Debug> ").strip().lower()
				if command in ['continue', 'c']:
					break
				elif command in ['step', 's']:
					break
				elif command.startswith('print '):
					var_name = command[6:]
					if var_name in context.variables:
						print(f"{var_name}: {context.variables[var_name]}")
					else:
						print(f"Variable '{var_name}' not found")
				elif command in ['help', 'h']:
					print("Available commands:")
					print("  continue, c - Continue execution")
					print("  step, s     - Step to next component")
					print("  print <var> - Print variable value")
					print("  help, h     - Show this help")
				else:
					print(f"Unknown command: {command}")
		
		# Execute component
		try:
			result = await component.execute(context.input_data, context.variables)
			trace_entry['result'] = {
				'success': result.success,
				'output_data': result.output_data,
				'error_message': result.error_message,
				'duration_ms': result.duration_ms
			}
		except Exception as e:
			trace_entry['exception'] = str(e)
			raise
		finally:
			self.execution_trace.append(trace_entry)
		
		return result
	
	def get_execution_trace(self) -> list[dict]:
		"""Get complete execution trace."""
		return self.execution_trace.copy()
	
	def save_trace_to_file(self, filename: str):
		"""Save execution trace to file."""
		with open(filename, 'w') as f:
			json.dump(self.execution_trace, f, indent=2)
```

### Error Handling

```python
class WorkflowErrorHandler:
	"""Centralized error handling for workflow operations."""
	
	def __init__(self):
		self.logger = logging.getLogger(self.__class__.__name__)
		self.error_handlers = {}
	
	def register_error_handler(self, error_type: type, handler: callable):
		"""Register custom error handler."""
		self.error_handlers[error_type] = handler
	
	async def handle_error(
		self, 
		error: Exception, 
		context: dict | None = None
	) -> dict:
		"""Handle error with appropriate response."""
		error_info = {
			'error_type': error.__class__.__name__,
			'error_message': str(error),
			'timestamp': datetime.utcnow().isoformat(),
			'context': context or {}
		}
		
		# Check for custom handler
		error_type = type(error)
		if error_type in self.error_handlers:
			try:
				custom_response = await self.error_handlers[error_type](error, context)
				error_info.update(custom_response)
			except Exception as handler_error:
				self.logger.error(f"Error in custom error handler: {handler_error}")
		
		# Handle specific error types
		if isinstance(error, ValidationError):
			error_info.update(await self._handle_validation_error(error))
		elif isinstance(error, PermissionError):
			error_info.update(await self._handle_permission_error(error))
		elif isinstance(error, TimeoutError):
			error_info.update(await self._handle_timeout_error(error))
		elif isinstance(error, ConnectionError):
			error_info.update(await self._handle_connection_error(error))
		else:
			error_info.update(await self._handle_generic_error(error))
		
		# Log error
		self.logger.error(
			f"Error handled: {error_info['error_type']}",
			extra=error_info
		)
		
		return error_info
	
	async def _handle_validation_error(self, error: ValidationError) -> dict:
		"""Handle validation errors."""
		return {
			'category': 'validation',
			'severity': 'warning',
			'user_message': 'The provided data is invalid. Please check your input.',
			'details': error.errors() if hasattr(error, 'errors') else None,
			'recoverable': True
		}
	
	async def _handle_permission_error(self, error: PermissionError) -> dict:
		"""Handle permission errors."""
		return {
			'category': 'security',
			'severity': 'error',
			'user_message': 'You do not have permission to perform this action.',
			'recoverable': False
		}
	
	async def _handle_timeout_error(self, error: TimeoutError) -> dict:
		"""Handle timeout errors."""
		return {
			'category': 'performance',
			'severity': 'error',
			'user_message': 'The operation timed out. Please try again.',
			'recoverable': True,
			'retry_suggested': True
		}
	
	async def _handle_connection_error(self, error: ConnectionError) -> dict:
		"""Handle connection errors."""
		return {
			'category': 'network',
			'severity': 'error',
			'user_message': 'Connection failed. Please check your network and try again.',
			'recoverable': True,
			'retry_suggested': True
		}
	
	async def _handle_generic_error(self, error: Exception) -> dict:
		"""Handle generic errors."""
		return {
			'category': 'system',
			'severity': 'error',
			'user_message': 'An unexpected error occurred. Please contact support if the problem persists.',
			'recoverable': False
		}
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use async/await throughout for I/O operations  
- Tabs for indentation (not spaces)
- Modern type hints: `str | None`, `list[str]`, `dict[str, Any]`
- Pydantic v2 models with strict validation
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Git Workflow

1. **Fork the repository** and create your feature branch
2. **Make your changes** following the coding standards
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run the test suite** and ensure all tests pass
6. **Submit a pull request** with a clear description

### Pull Request Process

1. **PR Title**: Use conventional commit format (feat:, fix:, docs:, etc.)
2. **Description**: Provide detailed description of changes
3. **Testing**: Include test results and coverage information
4. **Documentation**: Update relevant documentation
5. **Review**: Address all review comments before merging

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/apg-workflow-orchestration.git
cd apg-workflow-orchestration

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest -vxs tests/ci

# Start development server
python service.py --development
```

---

This developer guide provides comprehensive information for building and extending the APG Workflow Orchestration platform. For additional support, please refer to the API documentation or contact the development team.

**© 2025 Datacraft. All rights reserved.**