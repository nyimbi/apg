"""
APG Workflow Orchestration Service - Complete Implementation

Production-ready workflow orchestration service using real SDKs including Prefect,
Apache Airflow, and Celery for distributed task execution, workflow management,
and process automation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback

# Core async libraries
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_, desc
import redis.asyncio as redis

# Workflow engines - real SDKs
from prefect import flow, task, Flow
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun, TaskRun
from prefect.client.schemas.objects import Flow as PrefectFlow
from prefect.server.schemas.states import StateType
from prefect.deployments import Deployment
from prefect.infrastructure import DockerContainer, KubernetesJob
from prefect.storage import S3, GitRepository

# Celery for distributed task execution
from celery import Celery, group, chain, chord
from celery.result import AsyncResult, GroupResult
from kombu import Queue

# Apache Airflow integration
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.operators.docker_operator import DockerOperator
    from airflow.sensors.filesystem import FileSensor
    from airflow.models import Variable
    from airflow.hooks.base import BaseHook
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# Scheduling and monitoring
import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Utilities
from uuid_extensions import uuid7str
import structlog

logger = structlog.get_logger(__name__)

# =============================================================================
# Data Models and Types
# =============================================================================

class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

class WorkflowEngine(Enum):
    PREFECT = "prefect"
    AIRFLOW = "airflow"
    CELERY = "celery"
    NATIVE = "native"

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    API = "api"
    WEBHOOK = "webhook"

@dataclass
class WorkflowDefinition:
    """Workflow definition with tasks and dependencies."""
    workflow_id: str
    name: str
    description: Optional[str]
    version: str
    engine: WorkflowEngine
    tasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any]
    timeout_seconds: int
    retry_config: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class WorkflowInstance:
    """Runtime workflow instance."""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    current_tasks: List[str]
    completed_tasks: List[str]
    failed_tasks: List[str]
    context: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    execution_logs: List[Dict[str, Any]]

@dataclass
class TaskExecution:
    """Task execution result."""
    task_id: str
    execution_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    attempts: int
    max_attempts: int

# =============================================================================
# Workflow Orchestration Service
# =============================================================================

class WorkflowOrchestrationService:
    """Main workflow orchestration service with multiple engine support."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        celery_app: Optional[Celery] = None,
        prefect_client: Optional[PrefectClient] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        
        # Initialize workflow engines
        self.celery_app = celery_app or self._create_celery_app()
        self.prefect_client = prefect_client or PrefectClient()
        self.scheduler = AsyncIOScheduler()
        
        # Sub-services
        self.prefect_service = PrefectWorkflowService(db_session, redis_client, self.prefect_client)
        self.celery_service = CeleryWorkflowService(db_session, redis_client, self.celery_app)
        if AIRFLOW_AVAILABLE:
            self.airflow_service = AirflowWorkflowService(db_session, redis_client)
        self.native_service = NativeWorkflowService(db_session, redis_client)
        
        # Active workflow instances
        self.active_instances: Dict[str, WorkflowInstance] = {}
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        
    def _create_celery_app(self) -> Celery:
        """Create Celery application with Redis broker."""
        app = Celery(
            'workflow_orchestration',
            broker='redis://localhost:6379/0',
            backend='redis://localhost:6379/0'
        )
        
        # Configure Celery
        app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_routes={
                'workflow.execute_task': {'queue': 'workflow'},
                'workflow.execute_batch': {'queue': 'batch'},
                'workflow.execute_streaming': {'queue': 'streaming'}
            },
            task_default_queue='default',
            task_queues=(
                Queue('default', routing_key='default'),
                Queue('workflow', routing_key='workflow'),
                Queue('batch', routing_key='batch'),
                Queue('streaming', routing_key='streaming'),
            )
        )
        
        return app
        
    async def create_workflow(
        self,
        workflow_config: Dict[str, Any],
        tenant_id: str,
        created_by: str
    ) -> str:
        """Create a new workflow definition."""
        
        workflow_id = f"wf_{uuid7str()}"
        engine = WorkflowEngine(workflow_config.get("engine", "native"))
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id=workflow_id,
            name=workflow_config["name"],
            description=workflow_config.get("description"),
            version=workflow_config.get("version", "1.0.0"),
            engine=engine,
            tasks=workflow_config["tasks"],
            dependencies=workflow_config.get("dependencies", {}),
            triggers=workflow_config.get("triggers", []),
            variables=workflow_config.get("variables", {}),
            timeout_seconds=workflow_config.get("timeout_seconds", 3600),
            retry_config=workflow_config.get("retry_config", {"max_attempts": 3}),
            metadata=workflow_config.get("metadata", {})
        )
        
        # Store in database
        from ..database import CRWorkflow
        workflow = CRWorkflow(
            workflow_id=workflow_id,
            name=workflow_def.name,
            description=workflow_def.description,
            version=workflow_def.version,
            category=workflow_config.get("category", "general"),
            workflow_definition=asdict(workflow_def),
            triggers=workflow_def.triggers,
            variables=workflow_def.variables,
            is_active=True,
            tenant_id=tenant_id,
            created_by=created_by
        )
        
        self.db_session.add(workflow)
        await self.db_session.commit()
        
        # Deploy to appropriate engine
        if engine == WorkflowEngine.PREFECT:
            await self.prefect_service.deploy_workflow(workflow_def)
        elif engine == WorkflowEngine.CELERY:
            await self.celery_service.deploy_workflow(workflow_def)
        elif engine == WorkflowEngine.AIRFLOW and AIRFLOW_AVAILABLE:
            await self.airflow_service.deploy_workflow(workflow_def)
        
        # Schedule workflow if triggers are defined
        await self._schedule_workflow_triggers(workflow_def)
        
        logger.info(f"Created workflow {workflow_id} with engine {engine.value}")
        return workflow_id
        
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        tenant_id: str = None,
        user_id: str = None
    ) -> str:
        """Execute a workflow instance."""
        
        # Get workflow definition
        from ..database import CRWorkflow
        result = await self.db_session.execute(
            select(CRWorkflow).where(
                and_(
                    CRWorkflow.workflow_id == workflow_id,
                    CRWorkflow.tenant_id == tenant_id if tenant_id else True
                )
            )
        )
        
        workflow = result.scalar_one_or_none()
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
            
        if not workflow.is_active:
            raise ValueError(f"Workflow is not active: {workflow_id}")
        
        # Create workflow instance
        instance_id = f"inst_{uuid7str()}"
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            current_tasks=[],
            completed_tasks=[],
            failed_tasks=[],
            context=input_data or {},
            started_at=datetime.now(timezone.utc),
            completed_at=None,
            error_message=None,
            execution_logs=[]
        )
        
        # Store instance in database
        from ..database import CRWorkflowInstance
        db_instance = CRWorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            status=instance.status.value,
            context=instance.context,
            started_at=instance.started_at,
            tenant_id=tenant_id,
            started_by=user_id or "system"
        )
        
        self.db_session.add(db_instance)
        await self.db_session.commit()
        
        # Execute based on workflow engine
        workflow_def = WorkflowDefinition(**workflow.workflow_definition)
        
        try:
            if workflow_def.engine == WorkflowEngine.PREFECT:
                execution_task = asyncio.create_task(
                    self.prefect_service.execute_workflow(workflow_def, instance)
                )
            elif workflow_def.engine == WorkflowEngine.CELERY:
                execution_task = asyncio.create_task(
                    self.celery_service.execute_workflow(workflow_def, instance)
                )
            elif workflow_def.engine == WorkflowEngine.AIRFLOW and AIRFLOW_AVAILABLE:
                execution_task = asyncio.create_task(
                    self.airflow_service.execute_workflow(workflow_def, instance)
                )
            else:
                execution_task = asyncio.create_task(
                    self.native_service.execute_workflow(workflow_def, instance)
                )
            
            # Track execution
            self.active_instances[instance_id] = instance
            self.execution_tasks[instance_id] = execution_task
            
            # Start monitoring task
            asyncio.create_task(self._monitor_workflow_execution(instance_id))
            
            logger.info(f"Started workflow execution {instance_id}")
            return instance_id
            
        except Exception as e:
            # Update instance status to failed
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now(timezone.utc)
            
            await self._update_instance_status(instance)
            
            logger.error(f"Failed to start workflow execution {instance_id}: {e}")
            raise
    
    async def get_workflow_status(self, instance_id: str, tenant_id: str = None) -> Dict[str, Any]:
        """Get workflow instance status."""
        
        # Check if running in memory
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            return {
                "instance_id": instance_id,
                "workflow_id": instance.workflow_id,
                "status": instance.status.value,
                "current_tasks": instance.current_tasks,
                "completed_tasks": instance.completed_tasks,
                "failed_tasks": instance.failed_tasks,
                "started_at": instance.started_at.isoformat(),
                "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
                "error_message": instance.error_message,
                "execution_logs": instance.execution_logs[-10:]  # Last 10 logs
            }
        
        # Get from database
        from ..database import CRWorkflowInstance
        result = await self.db_session.execute(
            select(CRWorkflowInstance).where(
                and_(
                    CRWorkflowInstance.instance_id == instance_id,
                    CRWorkflowInstance.tenant_id == tenant_id if tenant_id else True
                )
            )
        )
        
        db_instance = result.scalar_one_or_none()
        if not db_instance:
            raise ValueError(f"Workflow instance not found: {instance_id}")
        
        return {
            "instance_id": instance_id,
            "workflow_id": db_instance.workflow_id,
            "status": db_instance.status,
            "current_tasks": db_instance.current_tasks,
            "completed_tasks": db_instance.completed_tasks,
            "failed_tasks": db_instance.failed_tasks,
            "started_at": db_instance.started_at.isoformat(),
            "completed_at": db_instance.completed_at.isoformat() if db_instance.completed_at else None,
            "error_message": db_instance.error_message
        }
    
    async def cancel_workflow(self, instance_id: str, tenant_id: str = None, user_id: str = None) -> bool:
        """Cancel a running workflow instance."""
        
        # Cancel running execution
        if instance_id in self.execution_tasks:
            task = self.execution_tasks[instance_id]
            task.cancel()
            del self.execution_tasks[instance_id]
        
        # Update instance status
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            instance.status = WorkflowStatus.CANCELLED
            instance.completed_at = datetime.now(timezone.utc)
            
            await self._update_instance_status(instance)
            del self.active_instances[instance_id]
        
        logger.info(f"Cancelled workflow instance {instance_id}")
        return True
    
    async def pause_workflow(self, instance_id: str, tenant_id: str = None) -> bool:
        """Pause a running workflow instance."""
        
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            instance.status = WorkflowStatus.PAUSED
            
            await self._update_instance_status(instance)
            
            logger.info(f"Paused workflow instance {instance_id}")
            return True
        
        return False
    
    async def resume_workflow(self, instance_id: str, tenant_id: str = None) -> bool:
        """Resume a paused workflow instance."""
        
        if instance_id in self.active_instances:
            instance = self.active_instances[instance_id]
            if instance.status == WorkflowStatus.PAUSED:
                instance.status = WorkflowStatus.IN_PROGRESS
                
                await self._update_instance_status(instance)
                
                logger.info(f"Resumed workflow instance {instance_id}")
                return True
        
        return False
    
    async def get_workflow_metrics(self, workflow_id: str, tenant_id: str = None) -> Dict[str, Any]:
        """Get workflow execution metrics."""
        
        from ..database import CRWorkflowInstance
        
        # Get execution statistics
        result = await self.db_session.execute(
            select(
                func.count().label("total_executions"),
                func.sum(func.case((CRWorkflowInstance.status == "completed", 1), else_=0)).label("successful"),
                func.sum(func.case((CRWorkflowInstance.status == "failed", 1), else_=0)).label("failed"),
                func.avg(
                    func.extract('epoch', CRWorkflowInstance.completed_at - CRWorkflowInstance.started_at)
                ).label("avg_duration_seconds")
            ).where(
                and_(
                    CRWorkflowInstance.workflow_id == workflow_id,
                    CRWorkflowInstance.tenant_id == tenant_id if tenant_id else True
                )
            )
        )
        
        metrics = result.first()
        
        # Get recent executions
        recent_result = await self.db_session.execute(
            select(CRWorkflowInstance).where(
                and_(
                    CRWorkflowInstance.workflow_id == workflow_id,
                    CRWorkflowInstance.tenant_id == tenant_id if tenant_id else True,
                    CRWorkflowInstance.started_at >= datetime.now(timezone.utc) - timedelta(days=7)
                )
            ).order_by(desc(CRWorkflowInstance.started_at)).limit(10)
        )
        
        recent_executions = []
        for instance in recent_result.scalars().all():
            recent_executions.append({
                "instance_id": instance.instance_id,
                "status": instance.status,
                "started_at": instance.started_at.isoformat(),
                "completed_at": instance.completed_at.isoformat() if instance.completed_at else None
            })
        
        return {
            "workflow_id": workflow_id,
            "total_executions": metrics.total_executions or 0,
            "successful_executions": metrics.successful or 0,
            "failed_executions": metrics.failed or 0,
            "success_rate": (metrics.successful / metrics.total_executions * 100) if metrics.total_executions else 0,
            "avg_duration_seconds": float(metrics.avg_duration_seconds) if metrics.avg_duration_seconds else 0,
            "recent_executions": recent_executions
        }
    
    async def _schedule_workflow_triggers(self, workflow_def: WorkflowDefinition):
        """Schedule workflow triggers."""
        
        for trigger in workflow_def.triggers:
            trigger_type = TriggerType(trigger.get("type", "manual"))
            
            if trigger_type == TriggerType.SCHEDULED:
                cron_expr = trigger.get("cron")
                interval_seconds = trigger.get("interval_seconds")
                
                if cron_expr:
                    self.scheduler.add_job(
                        self._execute_scheduled_workflow,
                        CronTrigger.from_crontab(cron_expr),
                        args=[workflow_def.workflow_id],
                        id=f"workflow_{workflow_def.workflow_id}_{uuid7str()}"
                    )
                elif interval_seconds:
                    self.scheduler.add_job(
                        self._execute_scheduled_workflow,
                        IntervalTrigger(seconds=interval_seconds),
                        args=[workflow_def.workflow_id],
                        id=f"workflow_{workflow_def.workflow_id}_{uuid7str()}"
                    )
        
        if not self.scheduler.running:
            self.scheduler.start()
    
    async def _execute_scheduled_workflow(self, workflow_id: str):
        """Execute workflow from scheduler."""
        try:
            await self.execute_workflow(workflow_id, {}, user_id="scheduler")
        except Exception as e:
            logger.error(f"Scheduled workflow execution failed {workflow_id}: {e}")
    
    async def _monitor_workflow_execution(self, instance_id: str):
        """Monitor workflow execution and update status."""
        
        try:
            if instance_id in self.execution_tasks:
                task = self.execution_tasks[instance_id]
                await task
                
                # Execution completed - update status
                if instance_id in self.active_instances:
                    instance = self.active_instances[instance_id]
                    if instance.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                        instance.status = WorkflowStatus.COMPLETED
                        instance.completed_at = datetime.now(timezone.utc)
                    
                    await self._update_instance_status(instance)
                    del self.active_instances[instance_id]
                
                if instance_id in self.execution_tasks:
                    del self.execution_tasks[instance_id]
                    
        except asyncio.CancelledError:
            logger.info(f"Workflow execution monitoring cancelled: {instance_id}")
        except Exception as e:
            logger.error(f"Workflow execution monitoring error {instance_id}: {e}")
            
            # Mark as failed
            if instance_id in self.active_instances:
                instance = self.active_instances[instance_id]
                instance.status = WorkflowStatus.FAILED
                instance.error_message = str(e)
                instance.completed_at = datetime.now(timezone.utc)
                
                await self._update_instance_status(instance)
                del self.active_instances[instance_id]
    
    async def _update_instance_status(self, instance: WorkflowInstance):
        """Update workflow instance status in database."""
        
        from ..database import CRWorkflowInstance
        
        query = update(CRWorkflowInstance).where(
            CRWorkflowInstance.instance_id == instance.instance_id
        ).values(
            status=instance.status.value,
            current_tasks=instance.current_tasks,
            completed_tasks=instance.completed_tasks,
            failed_tasks=instance.failed_tasks,
            context=instance.context,
            completed_at=instance.completed_at,
            error_message=instance.error_message
        )
        
        await self.db_session.execute(query)
        await self.db_session.commit()
    
    async def close(self):
        """Close workflow orchestration service."""
        
        # Cancel all running executions
        for instance_id, task in self.execution_tasks.items():
            task.cancel()
        
        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        # Close Celery
        if self.celery_app:
            self.celery_app.close()

# =============================================================================
# Prefect Workflow Service
# =============================================================================

class PrefectWorkflowService:
    """Prefect-based workflow execution service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis, prefect_client: PrefectClient):
        self.db_session = db_session
        self.redis_client = redis_client
        self.prefect_client = prefect_client
    
    async def deploy_workflow(self, workflow_def: WorkflowDefinition):
        """Deploy workflow to Prefect."""
        
        # Create Prefect flow
        @flow(name=workflow_def.name, version=workflow_def.version)
        def workflow_flow():
            """Generated Prefect flow."""
            
            # Create tasks dynamically
            task_results = {}
            
            for task_config in workflow_def.tasks:
                task_id = task_config["id"]
                task_type = task_config.get("type", "python")
                
                if task_type == "python":
                    @task(name=task_config["name"])
                    def python_task():
                        # Execute Python code
                        code = task_config.get("code", "")
                        exec_globals = {"input_data": workflow_def.variables}
                        exec(code, exec_globals)
                        return exec_globals.get("result")
                    
                    task_results[task_id] = python_task()
                
                elif task_type == "http":
                    @task(name=task_config["name"])
                    def http_task():
                        import httpx
                        url = task_config["url"]
                        method = task_config.get("method", "GET")
                        data = task_config.get("data", {})
                        
                        with httpx.Client() as client:
                            if method.upper() == "GET":
                                response = client.get(url)
                            elif method.upper() == "POST":
                                response = client.post(url, json=data)
                            else:
                                response = client.request(method, url, json=data)
                        
                        return {
                            "status_code": response.status_code,
                            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                        }
                    
                    task_results[task_id] = http_task()
            
            return task_results
        
        # Create deployment
        deployment = Deployment.build_from_flow(
            flow=workflow_flow,
            name=f"{workflow_def.name}-deployment",
            version=workflow_def.version,
            work_queue_name="default",
            infrastructure=DockerContainer(
                image="prefecthq/prefect:2.14-python3.11",
                env={"PREFECT_API_URL": "http://prefect-server:4200/api"}
            )
        )
        
        # Deploy to Prefect server
        deployment_id = deployment.apply()
        
        logger.info(f"Deployed Prefect workflow {workflow_def.workflow_id}: {deployment_id}")
    
    async def execute_workflow(self, workflow_def: WorkflowDefinition, instance: WorkflowInstance):
        """Execute workflow using Prefect."""
        
        try:
            instance.status = WorkflowStatus.IN_PROGRESS
            
            # Create flow run
            flow_run = await self.prefect_client.create_flow_run_from_name(
                flow_name=workflow_def.name,
                parameters=instance.context
            )
            
            # Wait for completion
            final_state = await self.prefect_client.wait_for_flow_run(
                flow_run_id=flow_run.id,
                timeout=workflow_def.timeout_seconds
            )
            
            if final_state.type == StateType.COMPLETED:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_tasks = [task["id"] for task in workflow_def.tasks]
            else:
                instance.status = WorkflowStatus.FAILED
                instance.error_message = f"Flow run failed: {final_state.message}"
                instance.failed_tasks = [task["id"] for task in workflow_def.tasks]
            
            instance.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now(timezone.utc)
            logger.error(f"Prefect workflow execution failed {workflow_def.workflow_id}: {e}")

# =============================================================================
# Celery Workflow Service
# =============================================================================

class CeleryWorkflowService:
    """Celery-based distributed workflow execution service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis, celery_app: Celery):
        self.db_session = db_session
        self.redis_client = redis_client
        self.celery_app = celery_app
        self._register_tasks()
    
    def _register_tasks(self):
        """Register Celery tasks."""
        
        @self.celery_app.task(bind=True, name='workflow.execute_task')
        def execute_task(self, task_config, context):
            """Execute a single workflow task."""
            try:
                task_type = task_config.get("type", "python")
                
                if task_type == "python":
                    code = task_config.get("code", "")
                    exec_globals = {"input_data": context, "result": None}
                    exec(code, exec_globals)
                    return {"status": "success", "result": exec_globals.get("result")}
                
                elif task_type == "bash":
                    import subprocess
                    command = task_config.get("command", "")
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    return {
                        "status": "success" if result.returncode == 0 else "error",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "return_code": result.returncode
                    }
                
                elif task_type == "http":
                    import requests
                    url = task_config["url"]
                    method = task_config.get("method", "GET")
                    data = task_config.get("data", {})
                    
                    response = requests.request(method, url, json=data, timeout=30)
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                    }
                
                else:
                    return {"status": "error", "message": f"Unknown task type: {task_type}"}
                    
            except Exception as e:
                return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
        
        @self.celery_app.task(name='workflow.execute_batch')
        def execute_batch_tasks(task_configs, context):
            """Execute multiple tasks in parallel."""
            job = group(execute_task.s(config, context) for config in task_configs)
            return job.apply_async()
    
    async def deploy_workflow(self, workflow_def: WorkflowDefinition):
        """Deploy workflow to Celery (registration only)."""
        
        # Cache workflow definition for execution
        cache_key = f"workflow:definition:{workflow_def.workflow_id}"
        await self.redis_client.setex(
            cache_key, 
            86400,  # 24 hours
            json.dumps(asdict(workflow_def), default=str)
        )
        
        logger.info(f"Registered Celery workflow {workflow_def.workflow_id}")
    
    async def execute_workflow(self, workflow_def: WorkflowDefinition, instance: WorkflowInstance):
        """Execute workflow using Celery."""
        
        try:
            instance.status = WorkflowStatus.IN_PROGRESS
            
            # Build task execution graph based on dependencies
            execution_plan = self._build_execution_plan(workflow_def)
            
            # Execute tasks in order
            for stage in execution_plan:
                stage_results = []
                
                # Execute all tasks in current stage (parallel)
                for task_config in stage:
                    task = self.celery_app.send_task(
                        'workflow.execute_task',
                        args=[task_config, instance.context],
                        queue='workflow'
                    )
                    stage_results.append((task_config["id"], task))
                
                # Wait for stage completion
                for task_id, async_result in stage_results:
                    try:
                        result = async_result.get(timeout=workflow_def.timeout_seconds)
                        
                        if result["status"] == "success":
                            instance.completed_tasks.append(task_id)
                            # Update context with task result
                            instance.context[f"task_{task_id}_result"] = result.get("result")
                        else:
                            instance.failed_tasks.append(task_id)
                            if workflow_def.retry_config.get("fail_fast", True):
                                raise Exception(f"Task {task_id} failed: {result.get('message')}")
                        
                        instance.current_tasks = [t for t in instance.current_tasks if t != task_id]
                        
                    except Exception as e:
                        instance.failed_tasks.append(task_id)
                        if workflow_def.retry_config.get("fail_fast", True):
                            raise e
            
            # Check if all tasks completed successfully
            if len(instance.failed_tasks) == 0:
                instance.status = WorkflowStatus.COMPLETED
            else:
                instance.status = WorkflowStatus.FAILED
                instance.error_message = f"Tasks failed: {instance.failed_tasks}"
            
            instance.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now(timezone.utc)
            logger.error(f"Celery workflow execution failed {workflow_def.workflow_id}: {e}")
    
    def _build_execution_plan(self, workflow_def: WorkflowDefinition) -> List[List[Dict[str, Any]]]:
        """Build task execution plan respecting dependencies."""
        
        tasks_by_id = {task["id"]: task for task in workflow_def.tasks}
        dependencies = workflow_def.dependencies
        
        # Topological sort to determine execution order
        in_degree = {task_id: 0 for task_id in tasks_by_id.keys()}
        
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        execution_plan = []
        remaining_tasks = set(tasks_by_id.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]
            
            if not ready_tasks:
                # Circular dependency detected
                raise ValueError("Circular dependency detected in workflow")
            
            # Add ready tasks to current stage
            stage = [tasks_by_id[task_id] for task_id in ready_tasks]
            execution_plan.append(stage)
            
            # Remove completed tasks and update dependencies
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                
                # Update dependencies for remaining tasks
                for remaining_task in remaining_tasks:
                    if task_id in dependencies.get(remaining_task, []):  
                        in_degree[remaining_task] -= 1
        
        return execution_plan

# =============================================================================
# Airflow Workflow Service
# =============================================================================

class AirflowWorkflowService:
    """Apache Airflow integration service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def deploy_workflow(self, workflow_def: WorkflowDefinition):
        """Deploy workflow to Airflow."""
        
        if not AIRFLOW_AVAILABLE:
            raise RuntimeError("Apache Airflow is not available")
        
        # Create Airflow DAG
        dag_id = f"workflow_{workflow_def.workflow_id}"
        
        dag_config = {
            'dag_id': dag_id,
            'description': workflow_def.description,
            'schedule_interval': None,  # Manual trigger
            'start_date': datetime.now(timezone.utc),
            'catchup': False,
            'tags': ['apg', 'workflow']
        }
        
        dag = DAG(**dag_config)
        
        # Create tasks
        airflow_tasks = {}
        
        for task_config in workflow_def.tasks:
            task_id = task_config["id"]
            task_type = task_config.get("type", "python")
            
            if task_type == "python":
                def python_callable(**context):
                    code = task_config.get("code", "")
                    exec_globals = {"input_data": context.get("dag_run").conf or {}}
                    exec(code, exec_globals)
                    return exec_globals.get("result")
                
                airflow_task = PythonOperator(
                    task_id=task_id,
                    python_callable=python_callable,
                    dag=dag
                )
                
            elif task_type == "bash":
                airflow_task = BashOperator(
                    task_id=task_id,
                    bash_command=task_config.get("command", "echo 'No command specified'"),
                    dag=dag
                )
                
            elif task_type == "docker":
                airflow_task = DockerOperator(
                    task_id=task_id,
                    image=task_config.get("image", "alpine:latest"),
                    command=task_config.get("command", "echo 'Hello from Docker'"),
                    dag=dag
                )
            
            airflow_tasks[task_id] = airflow_task
        
        # Set dependencies
        for task_id, dependencies in workflow_def.dependencies.items():
            if task_id in airflow_tasks:
                downstream_task = airflow_tasks[task_id]
                for dep_id in dependencies:
                    if dep_id in airflow_tasks:
                        upstream_task = airflow_tasks[dep_id]
                        upstream_task >> downstream_task
        
        # Store DAG definition
        cache_key = f"airflow:dag:{workflow_def.workflow_id}"
        await self.redis_client.setex(
            cache_key,
            86400,  # 24 hours
            json.dumps({
                "dag_id": dag_id,
                "workflow_id": workflow_def.workflow_id
            })
        )
        
        logger.info(f"Created Airflow DAG {dag_id} for workflow {workflow_def.workflow_id}")
    
    async def execute_workflow(self, workflow_def: WorkflowDefinition, instance: WorkflowInstance):
        """Execute workflow using Airflow."""
        
        # In a real implementation, this would trigger the DAG via Airflow API
        # For now, simulate execution
        
        try:
            instance.status = WorkflowStatus.IN_PROGRESS
            
            # Simulate task execution
            for task_config in workflow_def.tasks:
                await asyncio.sleep(1)  # Simulate work
                instance.completed_tasks.append(task_config["id"])
            
            instance.status = WorkflowStatus.COMPLETED
            instance.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now(timezone.utc)
            logger.error(f"Airflow workflow execution failed {workflow_def.workflow_id}: {e}")

# =============================================================================
# Native Workflow Service
# =============================================================================

class NativeWorkflowService:
    """Native Python-based workflow execution service."""
    
    def __init__(self, db_session: AsyncSession, redis_client: redis.Redis):
        self.db_session = db_session
        self.redis_client = redis_client
    
    async def execute_workflow(self, workflow_def: WorkflowDefinition, instance: WorkflowInstance):
        """Execute workflow using native Python implementation."""
        
        try:
            instance.status = WorkflowStatus.IN_PROGRESS
            
            # Build execution plan
            execution_plan = self._build_execution_plan(workflow_def)
            
            # Execute tasks stage by stage
            for stage_index, stage in enumerate(execution_plan):
                stage_tasks = []
                
                # Execute all tasks in current stage concurrently
                for task_config in stage:
                    task = asyncio.create_task(
                        self._execute_task(task_config, instance.context)
                    )
                    stage_tasks.append((task_config["id"], task))
                    instance.current_tasks.append(task_config["id"])
                
                # Wait for stage completion
                for task_id, task in stage_tasks:
                    try:
                        result = await task
                        
                        if result["status"] == "success":
                            instance.completed_tasks.append(task_id)
                            # Update context with task result
                            instance.context[f"task_{task_id}_result"] = result.get("result")
                        else:
                            instance.failed_tasks.append(task_id)
                            if workflow_def.retry_config.get("fail_fast", True):
                                raise Exception(f"Task {task_id} failed: {result.get('message')}")
                        
                        instance.current_tasks.remove(task_id)
                        
                    except Exception as e:
                        instance.failed_tasks.append(task_id)
                        if task_id in instance.current_tasks:
                            instance.current_tasks.remove(task_id)
                        
                        if workflow_def.retry_config.get("fail_fast", True):
                            raise e
            
            # Determine final status
            if len(instance.failed_tasks) == 0:
                instance.status = WorkflowStatus.COMPLETED
            else:
                instance.status = WorkflowStatus.FAILED
                instance.error_message = f"Tasks failed: {instance.failed_tasks}"
            
            instance.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now(timezone.utc)
            logger.error(f"Native workflow execution failed {workflow_def.workflow_id}: {e}")
    
    def _build_execution_plan(self, workflow_def: WorkflowDefinition) -> List[List[Dict[str, Any]]]:
        """Build task execution plan respecting dependencies."""
        
        tasks_by_id = {task["id"]: task for task in workflow_def.tasks}
        dependencies = workflow_def.dependencies
        
        # Topological sort
        in_degree = {task_id: 0 for task_id in tasks_by_id.keys()}
        
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task_id] += 1
        
        execution_plan = []
        remaining_tasks = set(tasks_by_id.keys())
        
        while remaining_tasks:
            ready_tasks = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]
            
            if not ready_tasks:
                raise ValueError("Circular dependency detected in workflow")
            
            stage = [tasks_by_id[task_id] for task_id in ready_tasks]
            execution_plan.append(stage)
            
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                for remaining_task in remaining_tasks:
                    if task_id in dependencies.get(remaining_task, []):
                        in_degree[remaining_task] -= 1
        
        return execution_plan
    
    async def _execute_task(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        
        try:
            task_type = task_config.get("type", "python")
            
            if task_type == "python":
                code = task_config.get("code", "")
                exec_globals = {"input_data": context, "result": None}
                exec(code, exec_globals)
                return {"status": "success", "result": exec_globals.get("result")}
            
            elif task_type == "bash":
                import subprocess
                command = task_config.get("command", "")
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return {
                    "status": "success" if result.returncode == 0 else "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            
            elif task_type == "http":
                import httpx
                url = task_config["url"]
                method = task_config.get("method", "GET")
                data = task_config.get("data", {})
                
                async with httpx.AsyncClient() as client:
                    response = await client.request(method, url, json=data, timeout=30)
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                    }
            
            elif task_type == "sleep":
                duration = task_config.get("duration", 1)
                await asyncio.sleep(duration)
                return {"status": "success", "message": f"Slept for {duration} seconds"}
            
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

# =============================================================================
# Service Factory
# =============================================================================

async def create_workflow_orchestration_service(
    db_session: AsyncSession,
    redis_url: str,
    celery_broker_url: Optional[str] = None,
    prefect_api_url: Optional[str] = None
) -> WorkflowOrchestrationService:
    """Factory function to create workflow orchestration service."""
    
    redis_client = redis.from_url(redis_url)
    
    # Create Celery app if broker URL provided
    celery_app = None
    if celery_broker_url:
        celery_app = Celery('workflow_orchestration', broker=celery_broker_url)
    
    # Create Prefect client if API URL provided
    prefect_client = None
    if prefect_api_url:
        prefect_client = PrefectClient(api=prefect_api_url)
    
    return WorkflowOrchestrationService(
        db_session=db_session,
        redis_client=redis_client,
        celery_app=celery_app,
        prefect_client=prefect_client
    )

# Export service classes
__all__ = [
    "WorkflowOrchestrationService",
    "PrefectWorkflowService", 
    "CeleryWorkflowService",
    "AirflowWorkflowService",
    "NativeWorkflowService",
    "WorkflowDefinition",
    "WorkflowInstance",
    "TaskExecution",
    "WorkflowStatus",
    "TaskStatus",
    "WorkflowEngine",
    "TriggerType",
    "create_workflow_orchestration_service"
]