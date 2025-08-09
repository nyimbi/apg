# APG Workflow & Business Process Management - Development Guide

**Comprehensive Developer Guide for WBPM Capability**

© 2025 Datacraft | Author: Nyimbi Odero | Version 1.0

---

## 📖 **Table of Contents**

1. [Development Environment Setup](#development-environment-setup)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [API Development](#api-development)
5. [Database Design](#database-design)
6. [Testing Strategy](#testing-strategy)
7. [Integration Patterns](#integration-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Security Implementation](#security-implementation)
10. [Deployment Guidelines](#deployment-guidelines)
11. [Monitoring & Observability](#monitoring--observability)
12. [Contributing Guidelines](#contributing-guidelines)

---

## 🛠️ **Development Environment Setup**

### **Prerequisites**

- **Python 3.11+** with async/await support
- **PostgreSQL 15+** for database storage
- **Redis 6+** for caching and session management
- **Node.js 18+** for frontend development
- **Docker & Docker Compose** for containerized development

### **Local Development Setup**

```bash
# Clone the repository
git clone https://github.com/datacraft/apg-platform.git
cd apg-platform/capabilities/general_cross_functional/workflow_business_process_mgmt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Development Configuration**

```python
# config/development.py
from standalone_integration import WBPMConfigurationFactory

# Create development configuration
config = WBPMConfigurationFactory.create_development_config()

# Database configuration
DATABASE_URL = "postgresql://localhost:5432/wbpm_dev"
REDIS_URL = "redis://localhost:6379/0"

# Feature flags for development
ENABLE_DEBUG_LOGGING = True
ENABLE_API_DOCS = True
ENABLE_CORS = True
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

# Performance settings for development
MAX_CONCURRENT_WORKFLOWS = 100
CACHE_TTL_SECONDS = 300
```

### **Docker Development Environment**

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  wbpm-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/wbpm_dev
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: wbmp_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules

volumes:
  postgres_data:
```

---

## 🏗️ **Architecture Overview**

### **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    APG WBMP Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │   React     │ │   Mobile    │ │    Visual Designer      │ │
│  │    Web      │ │    Apps     │ │      Studio             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  API Gateway Layer                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │  FastAPI    │ │  GraphQL    │ │      WebSocket          │ │
│  │   REST      │ │   Query     │ │   Real-time API         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Business Logic Layer                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │  Workflow   │ │    Task     │ │      Visual             │ │
│  │   Engine    │ │ Management  │ │     Designer            │ │
│  │             │ │             │ │                         │ │
│  ├─────────────┤ ├─────────────┤ ├─────────────────────────┤ │
│  │  Scheduler  │ │ Integration │ │     Analytics           │ │
│  │   Engine    │ │     Hub     │ │      Engine             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ PostgreSQL  │ │    Redis    │ │      File Storage       │ │
│  │  Database   │ │    Cache    │ │       (S3/Local)        │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Integration Layer                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ APG Platform│ │   External  │ │      Message            │ │
│  │ Integration │ │   Systems   │ │       Queue             │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **Component Interaction Flow**

```python
# Example: Process Execution Flow
async def execute_process_flow():
    """Demonstrates component interaction during process execution."""
    
    # 1. Visual Designer creates process definition
    canvas_result = await visual_designer.create_canvas(name="Order Process")
    canvas_id = canvas_result.data['canvas_id']
    
    # 2. Parser converts definition to executable format
    definition = await apg_parser.parse_workflow_definition(
        definition_text, APGWorkflowFormat.APG_DSL, context
    )
    
    # 3. Workflow engine executes process instance
    instance = await workflow_engine.start_process_instance(
        process_definition_id=canvas_id,
        variables={"order_id": "12345"},
        context=context
    )
    
    # 4. Task management assigns and routes tasks
    tasks = await task_manager.get_active_tasks(instance.instance_id)
    for task in tasks:
        await task_manager.assign_task(task.task_id, context)
    
    # 5. Scheduler monitors timing and creates alerts
    timer = create_process_timer(
        process_instance_id=instance.instance_id,
        duration_minutes=120,
        tenant_context=context
    )
    await scheduler.create_process_timer(timer)
    
    # 6. Analytics engine tracks performance
    await analytics_engine.record_process_metrics(instance)
```

---

## 🔧 **Core Components**

### **1. Workflow Engine (`workflow_engine.py`)**

The heart of the WBPM system, providing BPMN 2.0 compliant process execution.

```python
# Core workflow engine implementation
class WorkflowExecutionEngine:
    """High-performance BPMN 2.0 compliant workflow execution engine."""
    
    def __init__(self, max_concurrent_instances: int = 1000):
        self.max_concurrent_instances = max_concurrent_instances
        self.active_executions: Dict[str, ProcessExecution] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_instances)
    
    async def start_process_instance(
        self,
        process_definition: WBPMProcessDefinition,
        initial_variables: Dict[str, Any],
        context: APGTenantContext
    ) -> ProcessExecution:
        """Start new process instance with variables."""
        async with self.execution_semaphore:
            execution = ProcessExecution(
                instance_id=uuid7str(),
                process_definition=process_definition,
                variables=initial_variables,
                context=context
            )
            
            self.active_executions[execution.instance_id] = execution
            await self._execute_start_event(execution)
            
            return execution

# Usage example
engine = WorkflowExecutionEngine()
process_def = await load_process_definition("order_process_v1")
execution = await engine.start_process_instance(
    process_def, {"customer_id": "12345"}, context
)
```

### **2. Enhanced Visual Designer (`enhanced_visual_designer.py`)**

Advanced graphical workflow designer with real-time collaboration.

```python
# Visual designer implementation
class EnhancedVisualDesignerService:
    """Enhanced visual designer with advanced graphical capabilities."""
    
    async def create_canvas(
        self,
        name: str,
        context: APGTenantContext,
        template_id: Optional[str] = None
    ) -> WBPMServiceResponse:
        """Create new process diagram canvas."""
        canvas = ProcessDiagramCanvas(
            tenant_id=context.tenant_id,
            created_by=context.user_id,
            name=name
        )
        
        # Initialize with default start event
        await self._create_default_start_event(canvas, context)
        
        self.active_canvases[canvas.canvas_id] = canvas
        return WBPMServiceResponse(
            success=True,
            data={"canvas_id": canvas.canvas_id}
        )

# Usage example
designer = EnhancedVisualDesignerService()
canvas_result = await designer.create_canvas("My Process", context)
canvas_id = canvas_result.data['canvas_id']

# Add elements with timing configuration
await designer.add_element(
    canvas_id=canvas_id,
    element_type="userTask",
    position=VisualPosition(x=200, y=150),
    context=context
)
```

### **3. Workflow Scheduler (`workflow_scheduler.py`)**

Comprehensive scheduling and timing system with deep instrumentation.

```python
# Scheduler implementation
class WorkflowScheduler:
    """Comprehensive workflow scheduling and timing engine."""
    
    async def create_schedule(self, schedule: WorkflowSchedule) -> WBPMServiceResponse:
        """Create new workflow schedule."""
        # Validate schedule configuration
        validation_result = await self._validate_schedule(schedule)
        if not validation_result.success:
            return validation_result
        
        # Calculate next execution time
        next_execution = await self._calculate_next_execution(schedule)
        schedule.next_execution = next_execution
        
        # Store and activate schedule
        self.active_schedules[schedule.schedule_id] = schedule
        
        return WBPMServiceResponse(
            success=True,
            data={"schedule_id": schedule.schedule_id}
        )

# Usage example - Create recurring schedule
schedule = create_recurring_schedule(
    name="Daily Report Generation",
    process_definition_id="report_process",
    interval_minutes=1440,  # Daily
    tenant_context=context
)
await scheduler.create_schedule(schedule)
```

### **4. APG Workflow Parser (`apg_workflow_parser.py`)**

Multi-format workflow definition parser and implementer.

```python
# Parser implementation
class APGWorkflowParser:
    """Parser for APG workflow definitions in various formats."""
    
    async def parse_workflow_definition(
        self,
        definition_text: str,
        format_type: APGWorkflowFormat,
        context: APGTenantContext
    ) -> WBPMServiceResponse:
        """Parse workflow definition from text."""
        # Detect format if needed
        if format_type == APGWorkflowFormat.NATURAL_LANGUAGE:
            detected_format = await self._detect_format(definition_text)
            format_type = detected_format
        
        # Parse using appropriate parser
        parser_func = self.supported_formats.get(format_type)
        workflow_definition = await parser_func(definition_text, context)
        
        # Validate parsed definition
        validation_result = await self._validate_workflow_definition(workflow_definition)
        
        return WBPMServiceResponse(
            success=validation_result.success,
            data={"workflow_definition": workflow_definition}
        )

# Usage example - Parse APG DSL
apg_dsl = '''
WORKFLOW "Employee Onboarding"
START "New Employee"
USER_TASK "Complete Forms"
  duration: "2h"
  sla: "4h"
END "Complete"
'''

parser = APGWorkflowParser()
result = await parser.parse_workflow_definition(
    apg_dsl, APGWorkflowFormat.APG_DSL, context
)
```

---

## 🚀 **API Development**

### **FastAPI Implementation**

```python
# api.py - Main API router
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="APG Workflow & Business Process Management API",
    description="Enterprise-grade workflow automation and process management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection for services
async def get_workflow_service() -> WorkflowBusinessProcessMgmtService:
    """Get workflow service instance."""
    return await WorkflowServiceFactory.get_service()

async def get_tenant_context(
    authorization: str = Header(...),
) -> APGTenantContext:
    """Extract tenant context from request."""
    # In production, integrate with APG auth service
    return APGTenantContext(
        tenant_id="default",
        user_id="api_user",
        user_roles=["user"],
        permissions=["workflow_execute"]
    )

# Process management endpoints
@app.post("/api/v1/processes", response_model=WBPMServiceResponse)
async def create_process(
    process_data: ProcessCreationRequest,
    context: APGTenantContext = Depends(get_tenant_context),
    service: WorkflowBusinessProcessMgmtService = Depends(get_workflow_service)
):
    """Create new process definition."""
    return await service.create_process_definition(process_data, context)

@app.post("/api/v1/processes/{process_id}/instances", response_model=WBPMServiceResponse)
async def start_process_instance(
    process_id: str,
    instance_data: ProcessInstanceRequest,
    context: APGTenantContext = Depends(get_tenant_context),
    service: WorkflowBusinessProcessMgmtService = Depends(get_workflow_service)
):
    """Start new process instance."""
    return await service.start_process_instance(process_id, instance_data, context)

# Task management endpoints
@app.get("/api/v1/tasks", response_model=WBPMServiceResponse)
async def get_user_tasks(
    context: APGTenantContext = Depends(get_tenant_context),
    service: WorkflowBusinessProcessMgmtService = Depends(get_workflow_service)
):
    """Get tasks for current user."""
    return await service.get_user_tasks(context.user_id, context)

@app.post("/api/v1/tasks/{task_id}/complete", response_model=WBPMServiceResponse)
async def complete_task(
    task_id: str,
    completion_data: TaskCompletionRequest,
    context: APGTenantContext = Depends(get_tenant_context),
    service: WorkflowBusinessProcessMgmtService = Depends(get_workflow_service)
):
    """Complete a task."""
    return await service.complete_task(task_id, completion_data, context)

# Visual designer endpoints
@app.post("/api/v1/designer/canvas", response_model=WBPMServiceResponse)
async def create_canvas(
    canvas_data: CanvasCreationRequest,
    context: APGTenantContext = Depends(get_tenant_context),
    designer: EnhancedVisualDesignerService = Depends(get_visual_designer)
):
    """Create new design canvas."""
    return await designer.create_canvas(canvas_data.name, context)

# Scheduler endpoints
@app.post("/api/v1/schedules", response_model=WBPMServiceResponse)
async def create_schedule(
    schedule_data: ScheduleCreationRequest,
    context: APGTenantContext = Depends(get_tenant_context),
    scheduler: WorkflowScheduler = Depends(get_scheduler)
):
    """Create workflow schedule."""
    schedule = WorkflowSchedule(**schedule_data.dict(), tenant_id=context.tenant_id)
    return await scheduler.create_schedule(schedule)
```

### **GraphQL API**

```python
# graphql_api.py - GraphQL implementation
import strawberry
from typing import List, Optional

@strawberry.type
class Process:
    id: str
    name: str
    description: str
    version: str
    status: str
    created_at: str

@strawberry.type
class Task:
    id: str
    name: str
    description: str
    assignee: Optional[str]
    due_date: Optional[str]
    status: str

@strawberry.type
class Query:
    @strawberry.field
    async def processes(self, info) -> List[Process]:
        """Get all processes for current user."""
        context = get_context_from_info(info)
        service = await get_workflow_service()
        result = await service.get_processes(context)
        return [Process(**p) for p in result.data['processes']]
    
    @strawberry.field
    async def tasks(self, info) -> List[Task]:
        """Get tasks for current user."""
        context = get_context_from_info(info)
        service = await get_workflow_service()
        result = await service.get_user_tasks(context.user_id, context)
        return [Task(**t) for t in result.data['tasks']]

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def start_process(self, info, process_id: str, variables: str) -> Process:
        """Start new process instance."""
        context = get_context_from_info(info)
        service = await get_workflow_service()
        
        import json
        variables_dict = json.loads(variables) if variables else {}
        
        result = await service.start_process_instance(
            process_id, variables_dict, context
        )
        
        if result.success:
            return Process(**result.data['process'])
        else:
            raise Exception(result.message)

schema = strawberry.Schema(query=Query, mutation=Mutation)
```

### **WebSocket Real-time API**

```python
# websocket_api.py - Real-time updates
from fastapi import WebSocket, WebSocketDisconnect
import json

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_subscriptions[user_id] = set()
    
    def disconnect(self, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_subscriptions:
            del self.user_subscriptions[user_id]
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user."""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            await websocket.send_text(json.dumps(message))
    
    async def broadcast_process_update(self, process_id: str, update_data: dict):
        """Broadcast process update to subscribed users."""
        message = {
            "type": "process_update",
            "process_id": process_id,
            "data": update_data
        }
        
        for user_id, subscriptions in self.user_subscriptions.items():
            if process_id in subscriptions:
                await self.send_personal_message(message, user_id)

websocket_manager = WebSocketManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "subscribe_process":
                process_id = message["process_id"]
                websocket_manager.user_subscriptions[user_id].add(process_id)
            
            elif message["type"] == "unsubscribe_process":
                process_id = message["process_id"]
                websocket_manager.user_subscriptions[user_id].discard(process_id)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)
```

---

## 🗄️ **Database Design**

### **Migration Strategy**

```python
# migrations/env.py - Alembic configuration
from alembic import context
from sqlalchemy import engine_from_config, pool
from models import APGBaseModel

# Multi-tenant migration support
def run_migrations_online():
    """Run migrations in 'online' mode with multi-tenant support."""
    connectable = engine_from_config(
        context.config.get_section(context.config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Set search path for tenant isolation
        tenant_id = context.get_x_argument(as_dictionary=True).get('tenant_id')
        if tenant_id:
            connection.execute(f"SET search_path TO tenant_{tenant_id}, public")
        
        context.configure(
            connection=connection,
            target_metadata=APGBaseModel.metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()
```

### **Performance Optimization**

```sql
-- database_indexes.sql - Performance indexes
-- Process instance queries
CREATE INDEX CONCURRENTLY idx_process_instances_tenant_status 
ON wbpm_process_instances(tenant_id, status) 
WHERE status IN ('active', 'suspended');

-- Task assignment queries
CREATE INDEX CONCURRENTLY idx_tasks_assignee_status 
ON wbpm_tasks(assignee, status) 
WHERE status = 'active';

-- Process analytics queries
CREATE INDEX CONCURRENTLY idx_process_metrics_date_range 
ON wbpm_process_metrics(tenant_id, created_at) 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';

-- Timing and SLA monitoring
CREATE INDEX CONCURRENTLY idx_timers_expiration 
ON wbmp_process_timers(tenant_id, target_date) 
WHERE is_active = true AND target_date <= NOW() + INTERVAL '1 hour';

-- Partitioning for large tables
CREATE TABLE wbpm_audit_logs_y2025 PARTITION OF wbpm_audit_logs
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

### **Data Archival Strategy**

```python
# data_archival.py - Automated data archival
class DataArchivalService:
    """Automated archival of completed process data."""
    
    async def archive_completed_processes(self, older_than_days: int = 90):
        """Archive processes completed more than specified days ago."""
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Move completed processes to archive tables
        await self._archive_process_instances(cutoff_date)
        await self._archive_process_tasks(cutoff_date)
        await self._archive_audit_logs(cutoff_date)
        
        # Update statistics
        await self._update_archival_statistics()
    
    async def _archive_process_instances(self, cutoff_date: datetime):
        """Archive completed process instances."""
        query = """
        WITH archived_processes AS (
            DELETE FROM wbpm_process_instances 
            WHERE status = 'completed' 
            AND end_time < :cutoff_date
            RETURNING *
        )
        INSERT INTO wbpm_process_instances_archive 
        SELECT * FROM archived_processes;
        """
        
        await self.database.execute(query, {"cutoff_date": cutoff_date})
```

---

## 🧪 **Testing Strategy**

### **Unit Testing Framework**

```python
# tests/unit/test_workflow_engine.py
import pytest
import asyncio
from datetime import datetime
from models import APGTenantContext, WBPMProcessDefinition
from workflow_engine import WorkflowExecutionEngine

@pytest.fixture
def tenant_context():
    """Create test tenant context."""
    return APGTenantContext(
        tenant_id="test_tenant",
        user_id="test_user",
        user_roles=["user"],
        permissions=["workflow_execute"]
    )

@pytest.fixture
def workflow_engine():
    """Create workflow engine instance."""
    return WorkflowExecutionEngine(max_concurrent_instances=10)

@pytest.fixture
def simple_process_definition():
    """Create simple process definition for testing."""
    return WBPMProcessDefinition(
        process_id="test_process",
        name="Test Process",
        version="1.0",
        activities=[
            {
                "id": "start1",
                "type": "startEvent",
                "name": "Start"
            },
            {
                "id": "task1",
                "type": "userTask",
                "name": "Test Task"
            },
            {
                "id": "end1",
                "type": "endEvent",
                "name": "End"
            }
        ],
        flows=[
            {"from": "start1", "to": "task1"},
            {"from": "task1", "to": "end1"}
        ]
    )

class TestWorkflowEngine:
    """Test cases for workflow engine."""
    
    @pytest.mark.asyncio
    async def test_start_process_instance(
        self, 
        workflow_engine, 
        simple_process_definition, 
        tenant_context
    ):
        """Test starting a process instance."""
        execution = await workflow_engine.start_process_instance(
            simple_process_definition,
            {"test_var": "test_value"},
            tenant_context
        )
        
        assert execution.instance_id is not None
        assert execution.status == "active"
        assert execution.variables["test_var"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_concurrent_process_execution(
        self, 
        workflow_engine, 
        simple_process_definition, 
        tenant_context
    ):
        """Test concurrent process execution."""
        tasks = []
        for i in range(5):
            task = workflow_engine.start_process_instance(
                simple_process_definition,
                {"instance_number": i},
                tenant_context
            )
            tasks.append(task)
        
        executions = await asyncio.gather(*tasks)
        
        assert len(executions) == 5
        assert all(exec.status == "active" for exec in executions)
        assert len(set(exec.instance_id for exec in executions)) == 5
```

### **Integration Testing**

```python
# tests/integration/test_full_workflow.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

class TestWorkflowIntegration:
    """Integration tests for complete workflow functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self, client):
        """Test complete workflow from creation to completion."""
        
        # 1. Create process definition
        process_data = {
            "name": "Integration Test Process",
            "description": "Test process for integration testing",
            "activities": [
                {"id": "start1", "type": "startEvent", "name": "Start"},
                {"id": "task1", "type": "userTask", "name": "Test Task"},
                {"id": "end1", "type": "endEvent", "name": "End"}
            ],
            "flows": [
                {"from": "start1", "to": "task1"},
                {"from": "task1", "to": "end1"}
            ]
        }
        
        response = await client.post("/api/v1/processes", json=process_data)
        assert response.status_code == 200
        process_id = response.json()["data"]["process_id"]
        
        # 2. Start process instance
        instance_data = {"variables": {"test_input": "integration_test"}}
        response = await client.post(
            f"/api/v1/processes/{process_id}/instances", 
            json=instance_data
        )
        assert response.status_code == 200
        instance_id = response.json()["data"]["instance_id"]
        
        # 3. Get user tasks
        response = await client.get("/api/v1/tasks")
        assert response.status_code == 200
        tasks = response.json()["data"]["tasks"]
        assert len(tasks) > 0
        
        # 4. Complete task
        task_id = tasks[0]["id"]
        completion_data = {"variables": {"task_output": "completed"}}
        response = await client.post(
            f"/api/v1/tasks/{task_id}/complete", 
            json=completion_data
        )
        assert response.status_code == 200
        
        # 5. Verify process completion
        response = await client.get(f"/api/v1/processes/{process_id}/instances/{instance_id}")
        assert response.status_code == 200
        assert response.json()["data"]["status"] == "completed"
```

### **Performance Testing**

```python
# tests/performance/test_load.py
import asyncio
import time
from statistics import mean, median

class TestPerformance:
    """Performance tests for workflow system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_process_creation(self):
        """Test performance of concurrent process creation."""
        start_time = time.time()
        
        async def create_process():
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/api/v1/processes", json=test_process_data)
                return response.status_code == 200
        
        # Create 100 processes concurrently
        tasks = [create_process() for _ in range(100)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert all(results), "All process creations should succeed"
        assert duration < 10.0, f"Should complete within 10 seconds, took {duration:.2f}s"
        
        print(f"Created 100 processes in {duration:.2f} seconds")
        print(f"Average: {duration/100*1000:.2f} ms per process")
    
    @pytest.mark.asyncio
    async def test_task_completion_performance(self):
        """Test performance of task completion operations."""
        completion_times = []
        
        for i in range(50):
            start_time = time.time()
            
            # Complete a task
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    f"/api/v1/tasks/test_task_{i}/complete",
                    json={"variables": {"result": "success"}}
                )
            
            end_time = time.time()
            completion_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = mean(completion_times)
        median_time = median(completion_times)
        max_time = max(completion_times)
        
        assert avg_time < 100, f"Average completion time should be under 100ms, got {avg_time:.2f}ms"
        assert max_time < 500, f"Max completion time should be under 500ms, got {max_time:.2f}ms"
        
        print(f"Task completion - Avg: {avg_time:.2f}ms, Median: {median_time:.2f}ms, Max: {max_time:.2f}ms")
```

---

## 🔗 **Integration Patterns**

### **APG Platform Integration**

```python
# apg_integration.py - APG platform service integration
class APGPlatformIntegration:
    """Integration layer for APG platform services."""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.auth_client = APGAuthClient(config.auth_rbac_endpoint, config.apg_api_key)
        self.audit_client = APGAuditClient(config.audit_compliance_endpoint, config.apg_api_key)
        self.notification_client = APGNotificationClient(config.notification_engine_endpoint, config.apg_api_key)
    
    async def authenticate_user(self, token: str) -> APGTenantContext:
        """Authenticate user through APG auth service."""
        try:
            auth_response = await self.auth_client.validate_token(token)
            
            return APGTenantContext(
                tenant_id=auth_response["tenant_id"],
                user_id=auth_response["user_id"],
                user_roles=auth_response["roles"],
                permissions=auth_response["permissions"]
            )
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError("Invalid token")
    
    async def log_audit_event(self, event: AuditEvent) -> None:
        """Log audit event to APG audit service."""
        try:
            await self.audit_client.log_event({
                "event_type": event.event_type,
                "user_id": event.user_id,
                "tenant_id": event.tenant_id,
                "resource_type": "workflow",
                "resource_id": event.resource_id,
                "action": event.action,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details
            })
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
            # Don't fail the main operation due to audit logging issues
    
    async def send_notification(self, notification: NotificationRequest) -> None:
        """Send notification through APG notification service."""
        try:
            await self.notification_client.send_notification({
                "recipient": notification.recipient,
                "channel": notification.channel,
                "template": notification.template,
                "variables": notification.variables,
                "priority": notification.priority
            })
        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
```

### **External System Integration**

```python
# external_integrations.py - External system connectors
class ExternalSystemIntegrator:
    """Manages integrations with external systems."""
    
    def __init__(self):
        self.connectors: Dict[str, SystemConnector] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    async def register_connector(self, system_id: str, connector: SystemConnector):
        """Register external system connector."""
        self.connectors[system_id] = connector
        self.circuit_breakers[system_id] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=IntegrationError
        )
    
    async def execute_integration(
        self, 
        system_id: str, 
        operation: str, 
        data: Dict[str, Any]
    ) -> IntegrationResult:
        """Execute integration with circuit breaker protection."""
        if system_id not in self.connectors:
            raise IntegrationError(f"Unknown system: {system_id}")
        
        connector = self.connectors[system_id]
        circuit_breaker = self.circuit_breakers[system_id]
        
        try:
            async with circuit_breaker:
                result = await connector.execute(operation, data)
                return IntegrationResult(success=True, data=result)
        
        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker open for system {system_id}")
            return IntegrationResult(
                success=False, 
                error="System temporarily unavailable"
            )
        
        except Exception as e:
            logger.error(f"Integration failed for system {system_id}: {e}")
            return IntegrationResult(success=False, error=str(e))

# Example: ERP System Connector
class ERPSystemConnector(SystemConnector):
    """Connector for ERP system integration."""
    
    async def execute(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ERP operation."""
        if operation == "create_purchase_order":
            return await self._create_purchase_order(data)
        elif operation == "get_customer_data":
            return await self._get_customer_data(data)
        else:
            raise IntegrationError(f"Unknown operation: {operation}")
    
    async def _create_purchase_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create purchase order in ERP system."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/purchase_orders",
                json=data,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
```

---

## ⚡ **Performance Optimization**

### **Async Processing Optimization**

```python
# performance_optimizations.py
class PerformanceOptimizer:
    """Performance optimization utilities for workflow processing."""
    
    def __init__(self):
        self.connection_pool = None
        self.cache_client = None
        self.metrics_collector = MetricsCollector()
    
    async def optimize_database_queries(self):
        """Optimize database query performance."""
        # Connection pooling
        self.connection_pool = asyncpg.create_pool(
            DATABASE_URL,
            min_size=10,
            max_size=50,
            command_timeout=30,
            server_settings={
                'jit': 'off',  # Disable JIT for consistent performance
                'application_name': 'wbpm_api'
            }
        )
        
        # Query optimization with prepared statements
        await self._prepare_common_queries()
    
    async def _prepare_common_queries(self):
        """Prepare frequently used queries."""
        common_queries = {
            'get_user_tasks': """
                SELECT t.*, p.name as process_name 
                FROM wbpm_tasks t 
                JOIN wbpm_process_instances pi ON t.process_instance_id = pi.instance_id
                JOIN wbpm_process_definitions p ON pi.process_definition_id = p.process_id
                WHERE t.assignee = $1 AND t.status = 'active'
                ORDER BY t.priority DESC, t.created_at ASC
            """,
            'get_process_metrics': """
                SELECT 
                    COUNT(*) as total_instances,
                    AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) as avg_duration_minutes,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_count,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_count
                FROM wbpm_process_instances 
                WHERE process_definition_id = $1 
                AND created_at >= $2
            """
        }
        
        async with self.connection_pool.acquire() as conn:
            for query_name, query_sql in common_queries.items():
                await conn.prepare(query_sql)
    
    @asyncio_utils.timeout(30.0)
    async def execute_with_timeout(self, coro):
        """Execute coroutine with timeout protection."""
        try:
            return await coro
        except asyncio.TimeoutError:
            self.metrics_collector.increment('timeouts')
            raise WorkflowTimeoutError("Operation timed out")
```

### **Caching Strategy**

```python
# caching.py - Intelligent caching system
class WBPMCacheManager:
    """Intelligent caching for workflow data."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_strategies = {
            'process_definitions': CacheStrategy(ttl=3600, max_size=1000),
            'user_tasks': CacheStrategy(ttl=300, max_size=10000),
            'process_metrics': CacheStrategy(ttl=900, max_size=5000),
        }
    
    async def cache_process_definition(self, process_id: str, definition: dict):
        """Cache process definition with intelligent TTL."""
        cache_key = f"process_def:{process_id}"
        strategy = self.cache_strategies['process_definitions']
        
        # Use longer TTL for stable processes
        ttl = strategy.ttl * 2 if definition.get('status') == 'stable' else strategy.ttl
        
        await self.redis.setex(
            cache_key, 
            ttl, 
            json.dumps(definition, default=str)
        )
    
    async def get_cached_user_tasks(self, user_id: str) -> Optional[List[Dict]]:
        """Get cached user tasks with consistency check."""
        cache_key = f"user_tasks:{user_id}"
        
        cached_data = await self.redis.get(cache_key)
        if not cached_data:
            return None
        
        # Check cache consistency
        data = json.loads(cached_data)
        if await self._is_cache_stale(cache_key, data):
            await self.redis.delete(cache_key)
            return None
        
        return data['tasks']
    
    async def invalidate_related_caches(self, event_type: str, resource_id: str):
        """Invalidate caches based on events."""
        if event_type == 'task_completed':
            # Invalidate user task caches
            pattern = "user_tasks:*"
            await self._invalidate_pattern(pattern)
        
        elif event_type == 'process_updated':
            # Invalidate process definition cache
            cache_key = f"process_def:{resource_id}"
            await self.redis.delete(cache_key)
    
    async def _invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)
```

### **Memory Management**

```python
# memory_management.py - Memory optimization utilities
class MemoryManager:
    """Manages memory usage for large workflow operations."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.object_pools = {}
    
    @contextmanager
    def memory_limit(self, operation_name: str):
        """Context manager for memory-limited operations."""
        initial_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            final_memory = self._get_memory_usage()
            memory_used = final_memory - initial_memory
            
            if memory_used > self.max_memory_bytes:
                logger.warning(
                    f"Operation {operation_name} used {memory_used/1024/1024:.2f}MB "
                    f"(limit: {self.max_memory_bytes/1024/1024:.2f}MB)"
                )
                
                # Force garbage collection
                gc.collect()
    
    def create_object_pool(self, object_type: type, pool_size: int = 100):
        """Create object pool for frequently used objects."""
        pool_name = object_type.__name__
        self.object_pools[pool_name] = Queue(maxsize=pool_size)
        
        # Pre-populate pool
        for _ in range(pool_size):
            obj = object_type()
            self.object_pools[pool_name].put_nowait(obj)
    
    def get_pooled_object(self, object_type: type):
        """Get object from pool or create new one."""
        pool_name = object_type.__name__
        
        if pool_name in self.object_pools:
            try:
                return self.object_pools[pool_name].get_nowait()
            except Empty:
                pass
        
        return object_type()
    
    def return_to_pool(self, obj):
        """Return object to pool for reuse."""
        pool_name = obj.__class__.__name__
        
        if pool_name in self.object_pools:
            try:
                # Reset object state before returning to pool
                if hasattr(obj, 'reset'):
                    obj.reset()
                
                self.object_pools[pool_name].put_nowait(obj)
            except Full:
                pass  # Pool is full, let object be garbage collected
```

---

## 🔒 **Security Implementation**

### **Authentication & Authorization**

```python
# security.py - Security implementation
class WBPMSecurityManager:
    """Comprehensive security manager for WBPM."""
    
    def __init__(self, jwt_secret: str, encryption_key: str):
        self.jwt_secret = jwt_secret
        self.encryption_key = encryption_key.encode()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Security audit logger
        self.security_logger = logging.getLogger('wbpm.security')
    
    async def authenticate_request(self, request: Request) -> APGTenantContext:
        """Authenticate API request."""
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise AuthenticationError("Missing or invalid authorization header")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Verify JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check token expiration
            if payload.get('exp', 0) < time.time():
                raise AuthenticationError("Token expired")
            
            # Extract user context
            context = APGTenantContext(
                tenant_id=payload['tenant_id'],
                user_id=payload['user_id'],
                user_roles=payload['roles'],
                permissions=payload['permissions']
            )
            
            # Log successful authentication
            self.security_logger.info(f"User authenticated: {context.user_id}")
            
            return context
            
        except jwt.InvalidTokenError as e:
            self.security_logger.warning(f"Invalid token: {e}")
            raise AuthenticationError("Invalid token")
    
    async def authorize_action(
        self, 
        context: APGTenantContext, 
        resource: str, 
        action: str
    ) -> bool:
        """Authorize user action on resource."""
        # Check permissions
        required_permission = f"{resource}_{action}"
        if required_permission not in context.permissions:
            self.security_logger.warning(
                f"Access denied: {context.user_id} attempted {action} on {resource}"
            )
            return False
        
        # Check rate limits
        if not await self.rate_limiter.check_limit(context.user_id, action):
            self.security_logger.warning(f"Rate limit exceeded: {context.user_id}")
            return False
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data from storage."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    async def audit_security_event(
        self, 
        event_type: str, 
        user_id: str, 
        details: Dict[str, Any]
    ):
        """Audit security-related events."""
        audit_event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'ip_address': details.get('ip_address'),
            'user_agent': details.get('user_agent'),
            'resource': details.get('resource'),
            'action': details.get('action'),
            'success': details.get('success', True)
        }
        
        # Log to security audit file
        self.security_logger.info(json.dumps(audit_event))
        
        # Send to APG audit service if integrated
        if hasattr(self, 'apg_integration'):
            await self.apg_integration.log_audit_event(audit_event)
```

### **Data Protection**

```python
# data_protection.py - Data privacy and protection
class DataProtectionManager:
    """Manages data privacy and protection compliance."""
    
    def __init__(self):
        self.gdpr_handler = GDPRComplianceHandler()
        self.data_classifier = DataClassifier()
        self.retention_policies = RetentionPolicyManager()
    
    async def classify_workflow_data(self, workflow_data: Dict[str, Any]) -> DataClassification:
        """Classify workflow data for privacy compliance."""
        classification = DataClassification()
        
        for field_name, field_value in workflow_data.items():
            sensitivity = await self.data_classifier.classify_field(field_name, field_value)
            classification.add_field(field_name, sensitivity)
        
        return classification
    
    async def apply_data_masking(
        self, 
        data: Dict[str, Any], 
        user_permissions: List[str]
    ) -> Dict[str, Any]:
        """Apply data masking based on user permissions."""
        masked_data = data.copy()
        classification = await self.classify_workflow_data(data)
        
        for field_name, sensitivity in classification.fields.items():
            if sensitivity == DataSensitivity.HIGH:
                if 'view_sensitive_data' not in user_permissions:
                    masked_data[field_name] = self._mask_sensitive_value(
                        masked_data[field_name]
                    )
            elif sensitivity == DataSensitivity.PII:
                if 'view_pii_data' not in user_permissions:
                    masked_data[field_name] = self._mask_pii_value(
                        masked_data[field_name]
                    )
        
        return masked_data
    
    def _mask_sensitive_value(self, value: Any) -> str:
        """Mask sensitive data value."""
        if isinstance(value, str):
            if len(value) <= 4:
                return '*' * len(value)
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
        return '***'
    
    def _mask_pii_value(self, value: Any) -> str:
        """Mask PII data value."""
        if isinstance(value, str):
            # Email masking
            if '@' in value:
                local, domain = value.split('@', 1)
                return f"{local[0]}***@{domain}"
            # General PII masking
            return value[0] + '*' * (len(value) - 1) if value else ''
        return '***'
    
    async def handle_data_deletion_request(
        self, 
        user_id: str, 
        deletion_scope: str = 'all'
    ) -> DeletionResult:
        """Handle GDPR data deletion request."""
        deletion_result = DeletionResult(user_id=user_id)
        
        try:
            # Delete user data from workflow instances
            await self._delete_user_workflow_data(user_id, deletion_result)
            
            # Delete user data from tasks
            await self._delete_user_task_data(user_id, deletion_result)
            
            # Delete user data from audit logs (if allowed by retention policy)
            if deletion_scope == 'all':
                await self._delete_user_audit_data(user_id, deletion_result)
            
            deletion_result.success = True
            deletion_result.completed_at = datetime.utcnow()
            
        except Exception as e:
            deletion_result.success = False
            deletion_result.error = str(e)
        
        return deletion_result
```

---

## 🚀 **Deployment Guidelines**

### **Container Configuration**

```dockerfile
# Dockerfile - Production container
FROM python:3.11-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash wbpm

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-deps -r requirements-prod.txt

# Copy application code
COPY --chown=wbpm:wbpm . .

# Switch to application user
USER wbpm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### **Kubernetes Deployment**

```yaml
# k8s/deployment.yaml - Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wbpm-api
  labels:
    app: wbpm-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: wbpm-api
  template:
    metadata:
      labels:
        app: wbpm-api
        version: v1.0.0
    spec:
      containers:
      - name: wbpm-api
        image: datacraft/wbpm-api:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: wbpm-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: wbpm-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: wbpm-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: config-volume
        configMap:
          name: wbpm-config
      - name: logs-volume
        emptyDir: {}
      imagePullSecrets:
      - name: registry-secret
---
apiVersion: v1
kind: Service
metadata:
  name: wbpm-api-service
spec:
  selector:
    app: wbpm-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wbpm-api-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - wbpm-api.company.com
    secretName: wbpm-api-tls
  rules:
  - host: wbmp-api.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: wbpm-api-service
            port:
              number: 80
```

### **Environment Configuration**

```python
# config/production.py - Production configuration
from standalone_integration import WBPMConfigurationFactory
import os

# Production configuration
config = WBPMConfigurationFactory.create_production_integrated_config(
    apg_platform_url=os.getenv('APG_PLATFORM_URL'),
    apg_api_key=os.getenv('APG_API_KEY')
)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
DATABASE_POOL_SIZE = int(os.getenv('DATABASE_POOL_SIZE', '20'))
DATABASE_MAX_OVERFLOW = int(os.getenv('DATABASE_MAX_OVERFLOW', '30'))

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL')
REDIS_POOL_SIZE = int(os.getenv('REDIS_POOL_SIZE', '10'))

# Security configuration
JWT_SECRET = os.getenv('JWT_SECRET')
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')

# Performance settings
MAX_CONCURRENT_WORKFLOWS = int(os.getenv('MAX_CONCURRENT_WORKFLOWS', '1000'))
TASK_EXECUTION_TIMEOUT = int(os.getenv('TASK_EXECUTION_TIMEOUT', '3600'))

# Monitoring configuration
ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
METRICS_ENDPOINT = os.getenv('METRICS_ENDPOINT', '/metrics')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Feature flags
ENABLE_API_DOCS = os.getenv('ENABLE_API_DOCS', 'false').lower() == 'true'
ENABLE_DEBUG_LOGGING = os.getenv('ENABLE_DEBUG_LOGGING', 'false').lower() == 'true'
```

---

## 📊 **Monitoring & Observability**

### **Metrics Collection**

```python
# monitoring.py - Comprehensive monitoring system
from prometheus_client import Counter, Histogram, Gauge, generate_latest

class WBPMMetricsCollector:
    """Collects and exposes metrics for WBPM system."""
    
    def __init__(self):
        # Process metrics
        self.processes_created = Counter(
            'wbpm_processes_created_total', 
            'Total number of processes created',
            ['tenant_id', 'process_type']
        )
        
        self.process_duration = Histogram(
            'wbpm_process_duration_seconds',
            'Process execution duration',
            ['tenant_id', 'process_id', 'status']
        )
        
        self.active_processes = Gauge(
            'wbpm_active_processes',
            'Number of currently active processes',
            ['tenant_id']
        )
        
        # Task metrics
        self.tasks_created = Counter(
            'wbpm_tasks_created_total',
            'Total number of tasks created',
            ['tenant_id', 'task_type']
        )
        
        self.task_completion_time = Histogram(
            'wbmp_task_completion_seconds',
            'Task completion time',
            ['tenant_id', 'task_type', 'assignee_type']
        )
        
        # System metrics
        self.api_requests = Counter(
            'wbmp_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.api_duration = Histogram(
            'wbmp_api_duration_seconds',
            'API request duration',
            ['method', 'endpoint']
        )
    
    def record_process_created(self, tenant_id: str, process_type: str):
        """Record process creation."""
        self.processes_created.labels(
            tenant_id=tenant_id, 
            process_type=process_type
        ).inc()
    
    def record_process_completed(
        self, 
        tenant_id: str, 
        process_id: str, 
        duration: float, 
        status: str
    ):
        """Record process completion."""
        self.process_duration.labels(
            tenant_id=tenant_id,
            process_id=process_id,
            status=status
        ).observe(duration)
    
    def update_active_processes(self, tenant_id: str, count: int):
        """Update active process count."""
        self.active_processes.labels(tenant_id=tenant_id).set(count)
    
    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        # Database connection pool metrics
        if hasattr(self, 'db_pool'):
            pool_size = self.db_pool.get_size()
            pool_available = self.db_pool.get_available_size()
            
            db_pool_size = Gauge('wbpm_db_pool_size', 'Database pool size')
            db_pool_available = Gauge('wbpm_db_pool_available', 'Available database connections')
            
            db_pool_size.set(pool_size)
            db_pool_available.set(pool_available)
        
        # Memory usage
        import psutil
        process = psutil.Process()
        memory_usage = Gauge('wbpm_memory_usage_bytes', 'Memory usage in bytes')
        memory_usage.set(process.memory_info().rss)
        
        # CPU usage
        cpu_usage = Gauge('wbpm_cpu_usage_percent', 'CPU usage percentage')
        cpu_usage.set(process.cpu_percent())
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest()
```

### **Structured Logging**

```python
# logging_config.py - Structured logging configuration
import structlog
from pythonjsonlogger import jsonlogger

def configure_logging():
    """Configure structured logging for WBPM."""
    
    # Configure structlog
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
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/app/logs/wbpm.log')
        ]
    )
    
    # Add JSON formatter
    json_handler = logging.StreamHandler()
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    json_handler.setFormatter(json_formatter)
    
    # Configure specific loggers
    workflow_logger = logging.getLogger('wbpm.workflow')
    workflow_logger.addHandler(json_handler)
    
    security_logger = logging.getLogger('wbpm.security')
    security_logger.addHandler(json_handler)
    
    performance_logger = logging.getLogger('wbpm.performance')
    performance_logger.addHandler(json_handler)

# Usage in application code
logger = structlog.get_logger("wbpm.api")

async def process_request(request_id: str, user_id: str):
    """Example of structured logging in request processing."""
    logger = logger.bind(
        request_id=request_id,
        user_id=user_id,
        operation="process_request"
    )
    
    logger.info("Processing request started")
    
    try:
        result = await perform_operation()
        logger.info("Request processed successfully", result_count=len(result))
        return result
    
    except Exception as e:
        logger.error("Request processing failed", error=str(e), exc_info=True)
        raise
```

### **Health Checks**

```python
# health_checks.py - Comprehensive health monitoring
class HealthCheckManager:
    """Manages health checks for WBPM system."""
    
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'workflow_engine': self.check_workflow_engine,
            'scheduler': self.check_scheduler,
            'apg_integration': self.check_apg_integration
        }
    
    async def run_health_checks(self) -> HealthCheckResult:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                check_result = await check_func()
                results[check_name] = {
                    'status': 'healthy' if check_result else 'unhealthy',
                    'timestamp': datetime.utcnow().isoformat(),
                    'details': check_result.get('details', {}) if isinstance(check_result, dict) else {}
                }
                
                if not check_result:
                    overall_healthy = False
                    
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                }
                overall_healthy = False
        
        return HealthCheckResult(
            status='healthy' if overall_healthy else 'unhealthy',
            timestamp=datetime.utcnow().isoformat(),
            checks=results
        )
    
    async def check_database(self) -> bool:
        """Check database connectivity and performance."""
        try:
            async with self.db_pool.acquire() as conn:
                # Test basic connectivity
                result = await conn.fetchval('SELECT 1')
                if result != 1:
                    return False
                
                # Check response time
                start_time = time.time()
                await conn.fetchval('SELECT COUNT(*) FROM wbpm_process_definitions')
                response_time = time.time() - start_time
                
                # Healthy if response time < 1 second
                return response_time < 1.0
        
        except Exception:
            return False
    
    async def check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def check_workflow_engine(self) -> bool:
        """Check workflow engine health."""
        try:
            engine = await get_workflow_engine()
            # Check if engine can accept new processes
            return len(engine.active_executions) < engine.max_concurrent_instances
        except Exception:
            return False
    
    async def check_scheduler(self) -> dict:
        """Check scheduler health."""
        try:
            scheduler = await get_scheduler()
            return {
                'healthy': scheduler.is_running,
                'details': {
                    'active_schedules': len(scheduler.active_schedules),
                    'active_timers': len(scheduler.active_timers)
                }
            }
        except Exception:
            return False
    
    async def check_apg_integration(self) -> bool:
        """Check APG platform integration health."""
        try:
            integration = await get_apg_integration()
            health_result = await integration.check_integration_health()
            return health_result.success
        except Exception:
            return False
```

---

## 🤝 **Contributing Guidelines**

### **Development Workflow**

1. **Branch Strategy**
   - `main` - Production-ready code
   - `develop` - Integration branch for features
   - `feature/*` - Feature development branches
   - `hotfix/*` - Critical bug fixes

2. **Code Standards**
   - Follow PEP 8 style guide
   - Use type hints for all functions
   - Maintain 90%+ test coverage
   - Document all public APIs

3. **Pull Request Process**
   - Create feature branch from `develop`
   - Implement feature with tests
   - Update documentation
   - Submit PR with detailed description
   - Address review feedback
   - Merge after approval

### **Code Review Checklist**

```markdown
## Code Review Checklist

### Functionality
- [ ] Feature works as described
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Performance impact is acceptable

### Code Quality
- [ ] Code follows style guidelines
- [ ] Functions are well-documented
- [ ] Type hints are present
- [ ] No security vulnerabilities

### Testing
- [ ] Unit tests cover new code
- [ ] Integration tests pass
- [ ] Test coverage meets requirements
- [ ] Performance tests added if needed

### Documentation
- [ ] API documentation updated
- [ ] User guide updated if needed
- [ ] Code comments are clear
- [ ] CHANGELOG.md updated
```

### **Release Process**

```bash
# Release preparation script
#!/bin/bash

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

# Run tests
echo "Running test suite..."
pytest tests/

# Check code quality
echo "Checking code quality..."
flake8 .
mypy .

# Update version
echo "Updating version to $VERSION..."
sed -i "s/version = .*/version = \"$VERSION\"/" pyproject.toml

# Build documentation
echo "Building documentation..."
mkdocs build

# Create release branch
git checkout -b "release/$VERSION"
git add .
git commit -m "Prepare release $VERSION"

# Tag release
git tag -a "v$VERSION" -m "Release $VERSION"

echo "Release $VERSION prepared. Push to origin and create PR to main."
```

---

**© 2025 Datacraft. All rights reserved.**  
**Contact: www.datacraft.co.ke | nyimbi@gmail.com**

*This development guide provides comprehensive guidance for building, extending, and maintaining the APG Workflow & Business Process Management capability.*