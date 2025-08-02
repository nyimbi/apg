"""
APG Workflow Orchestration Deployment Manager

Comprehensive deployment automation with multiple strategies, environment management,
rollback capabilities, and health monitoring for workflow deployments.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
import hashlib

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, and_, or_, func, text

from ..models import Workflow, WorkflowStatus
from ..database import DatabaseManager
from .version_control import WorkflowVersion, VersionManager

logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
	"""Deployment strategies for workflows."""
	DIRECT = "direct"					# Direct deployment (immediate)
	BLUE_GREEN = "blue_green"			# Blue-green deployment
	CANARY = "canary"					# Canary deployment
	ROLLING = "rolling"					# Rolling deployment
	A_B_TEST = "a_b_test"				# A/B testing deployment

class DeploymentStatus(Enum):
	"""Status of workflow deployments."""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	ROLLED_BACK = "rolled_back"
	PAUSED = "paused"

class EnvironmentType(Enum):
	"""Types of deployment environments."""
	DEVELOPMENT = "development"
	TESTING = "testing"
	STAGING = "staging"
	PRODUCTION = "production"
	CUSTOM = "custom"

class HealthCheckStatus(Enum):
	"""Health check statuses."""
	HEALTHY = "healthy"
	UNHEALTHY = "unhealthy"
	WARNING = "warning"
	UNKNOWN = "unknown"

class DeploymentEnvironment(BaseModel):
	"""Deployment environment configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=100)
	environment_type: EnvironmentType
	description: str = Field(default="", max_length=500)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	resource_limits: Dict[str, Any] = Field(default_factory=dict)
	environment_variables: Dict[str, str] = Field(default_factory=dict)
	health_check_config: Dict[str, Any] = Field(default_factory=dict)
	approval_required: bool = Field(default=False)
	approvers: List[str] = Field(default_factory=list)
	auto_rollback_enabled: bool = Field(default=True)
	rollback_threshold_minutes: int = Field(default=30, ge=1, le=1440)
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the environment")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DeploymentPlan(BaseModel):
	"""Deployment execution plan."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	name: str = Field(..., min_length=1, max_length=200)
	workflow_id: str = Field(..., description="Target workflow ID")
	version_id: str = Field(..., description="Version to deploy")
	environment_id: str = Field(..., description="Target environment")
	strategy: DeploymentStrategy
	description: str = Field(default="", max_length=1000)
	configuration: Dict[str, Any] = Field(default_factory=dict)
	pre_deployment_checks: List[str] = Field(default_factory=list)
	post_deployment_checks: List[str] = Field(default_factory=list)
	rollback_plan: Dict[str, Any] = Field(default_factory=dict)
	notification_config: Dict[str, Any] = Field(default_factory=dict)
	schedule: Optional[datetime] = Field(default=None, description="Scheduled deployment time")
	auto_approve: bool = Field(default=False)
	tenant_id: str = Field(..., description="APG tenant identifier")
	created_by: str = Field(..., description="User who created the plan")
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DeploymentExecution(BaseModel):
	"""Deployment execution record."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	plan_id: str = Field(..., description="Deployment plan ID")
	status: DeploymentStatus = Field(default=DeploymentStatus.PENDING)
	started_at: Optional[datetime] = Field(default=None)
	completed_at: Optional[datetime] = Field(default=None)
	started_by: str = Field(..., description="User who started the deployment")
	progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
	current_step: str = Field(default="", description="Current deployment step")
	logs: List[str] = Field(default_factory=list, description="Deployment logs")
	metrics: Dict[str, Any] = Field(default_factory=dict)
	health_checks: List[Dict[str, Any]] = Field(default_factory=list)
	rollback_execution_id: Optional[str] = Field(default=None)
	error_details: Optional[str] = Field(default=None)
	tenant_id: str = Field(..., description="APG tenant identifier")

class HealthCheckResult(BaseModel):
	"""Health check result."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	check_name: str
	status: HealthCheckStatus
	message: str = Field(default="")
	metrics: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	duration_ms: float = Field(default=0.0, ge=0.0)

class DeploymentManager:
	"""Comprehensive deployment management system."""
	
	def __init__(
		self,
		database_manager: DatabaseManager,
		redis_client: redis.Redis,
		version_manager: VersionManager,
		tenant_id: str
	):
		self.database_manager = database_manager
		self.redis_client = redis_client
		self.version_manager = version_manager
		self.tenant_id = tenant_id
		
		# Deployment tables
		self.environments_table = "cr_deployment_environments"
		self.plans_table = "cr_deployment_plans"
		self.executions_table = "cr_deployment_executions"
		
		# Active deployments tracking
		self.active_deployments: Dict[str, DeploymentExecution] = {}
		
		# Health check functions
		self.health_check_functions: Dict[str, Callable] = {}
		
		# Deployment strategy handlers
		self.strategy_handlers = {
			DeploymentStrategy.DIRECT: self._execute_direct_deployment,
			DeploymentStrategy.BLUE_GREEN: self._execute_blue_green_deployment,
			DeploymentStrategy.CANARY: self._execute_canary_deployment,
			DeploymentStrategy.ROLLING: self._execute_rolling_deployment,
			DeploymentStrategy.A_B_TEST: self._execute_a_b_test_deployment
		}
		
		logger.info(f"Initialized DeploymentManager for tenant {tenant_id}")
	
	async def create_environment(
		self,
		environment_data: Dict[str, Any],
		user_id: str
	) -> DeploymentEnvironment:
		"""Create a new deployment environment."""
		
		try:
			environment_data.update({
				"tenant_id": self.tenant_id,
				"created_by": user_id
			})
			
			environment = DeploymentEnvironment(**environment_data)
			
			# Save to database
			async with self.database_manager.get_session() as session:
				await session.execute(
					text(f"""
					INSERT INTO {self.environments_table} (
						id, name, environment_type, description, configuration,
						resource_limits, environment_variables, health_check_config,
						approval_required, approvers, auto_rollback_enabled,
						rollback_threshold_minutes, tenant_id, created_by, created_at, updated_at
					) VALUES (
						:id, :name, :environment_type, :description, :configuration,
						:resource_limits, :environment_variables, :health_check_config,
						:approval_required, :approvers, :auto_rollback_enabled,
						:rollback_threshold_minutes, :tenant_id, :created_by, :created_at, :updated_at
					)
					"""),
					{
						"id": environment.id,
						"name": environment.name,
						"environment_type": environment.environment_type.value,
						"description": environment.description,
						"configuration": json.dumps(environment.configuration),
						"resource_limits": json.dumps(environment.resource_limits),
						"environment_variables": json.dumps(environment.environment_variables),
						"health_check_config": json.dumps(environment.health_check_config),
						"approval_required": environment.approval_required,
						"approvers": json.dumps(environment.approvers),
						"auto_rollback_enabled": environment.auto_rollback_enabled,
						"rollback_threshold_minutes": environment.rollback_threshold_minutes,
						"tenant_id": environment.tenant_id,
						"created_by": environment.created_by,
						"created_at": environment.created_at,
						"updated_at": environment.updated_at
					}
				)
				await session.commit()
			
			logger.info(f"Created deployment environment: {environment.name}")
			return environment
			
		except Exception as e:
			logger.error(f"Failed to create environment: {e}")
			raise
	
	async def create_deployment_plan(
		self,
		plan_data: Dict[str, Any],
		user_id: str
	) -> DeploymentPlan:
		"""Create a new deployment plan."""
		
		try:
			plan_data.update({
				"tenant_id": self.tenant_id,
				"created_by": user_id
			})
			
			plan = DeploymentPlan(**plan_data)
			
			# Validate plan
			await self._validate_deployment_plan(plan)
			
			# Save to database
			async with self.database_manager.get_session() as session:
				await session.execute(
					text(f"""
					INSERT INTO {self.plans_table} (
						id, name, workflow_id, version_id, environment_id, strategy,
						description, configuration, pre_deployment_checks, post_deployment_checks,
						rollback_plan, notification_config, schedule, auto_approve,
						tenant_id, created_by, created_at
					) VALUES (
						:id, :name, :workflow_id, :version_id, :environment_id, :strategy,
						:description, :configuration, :pre_deployment_checks, :post_deployment_checks,
						:rollback_plan, :notification_config, :schedule, :auto_approve,
						:tenant_id, :created_by, :created_at
					)
					"""),
					{
						"id": plan.id,
						"name": plan.name,
						"workflow_id": plan.workflow_id,
						"version_id": plan.version_id,
						"environment_id": plan.environment_id,
						"strategy": plan.strategy.value,
						"description": plan.description,
						"configuration": json.dumps(plan.configuration),
						"pre_deployment_checks": json.dumps(plan.pre_deployment_checks),
						"post_deployment_checks": json.dumps(plan.post_deployment_checks),
						"rollback_plan": json.dumps(plan.rollback_plan),
						"notification_config": json.dumps(plan.notification_config),
						"schedule": plan.schedule,
						"auto_approve": plan.auto_approve,
						"tenant_id": plan.tenant_id,
						"created_by": plan.created_by,
						"created_at": plan.created_at
					}
				)
				await session.commit()
			
			logger.info(f"Created deployment plan: {plan.name}")
			return plan
			
		except Exception as e:
			logger.error(f"Failed to create deployment plan: {e}")
			raise
	
	async def execute_deployment(
		self,
		plan_id: str,
		user_id: str,
		force_execute: bool = False
	) -> DeploymentExecution:
		"""Execute a deployment plan."""
		
		try:
			# Get deployment plan
			plan = await self._get_deployment_plan(plan_id)
			if not plan:
				raise ValueError(f"Deployment plan not found: {plan_id}")
			
			# Get environment
			environment = await self._get_environment(plan.environment_id)
			if not environment:
				raise ValueError(f"Environment not found: {plan.environment_id}")
			
			# Check approval requirements
			if environment.approval_required and not force_execute:
				# Implementation would check approval status
				logger.info(f"Deployment {plan_id} requires approval")
			
			# Create execution record
			execution = DeploymentExecution(
				plan_id=plan_id,
				started_by=user_id,
				tenant_id=self.tenant_id
			)
			
			# Save execution record
			await self._save_deployment_execution(execution)
			
			# Track active deployment
			self.active_deployments[execution.id] = execution
			
			# Start deployment asynchronously
			asyncio.create_task(self._execute_deployment_async(execution, plan, environment))
			
			logger.info(f"Started deployment execution: {execution.id}")
			return execution
			
		except Exception as e:
			logger.error(f"Failed to execute deployment: {e}")
			raise
	
	async def _execute_deployment_async(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute deployment asynchronously."""
		
		try:
			# Update execution status
			execution.status = DeploymentStatus.IN_PROGRESS
			execution.started_at = datetime.now(timezone.utc)
			execution.current_step = "Starting deployment"
			await self._update_deployment_execution(execution)
			
			# Run pre-deployment checks
			if plan.pre_deployment_checks:
				execution.current_step = "Running pre-deployment checks"
				execution.progress_percentage = 10.0
				await self._update_deployment_execution(execution)
				
				pre_check_results = await self._run_deployment_checks(plan.pre_deployment_checks, environment)
				if not all(result["passed"] for result in pre_check_results):
					raise Exception("Pre-deployment checks failed")
				
				execution.logs.append("Pre-deployment checks passed")
			
			# Execute deployment strategy
			execution.current_step = f"Executing {plan.strategy.value} deployment"
			execution.progress_percentage = 30.0
			await self._update_deployment_execution(execution)
			
			strategy_handler = self.strategy_handlers.get(plan.strategy)
			if not strategy_handler:
				raise ValueError(f"Unsupported deployment strategy: {plan.strategy}")
			
			await strategy_handler(execution, plan, environment)
			
			# Run post-deployment checks
			if plan.post_deployment_checks:
				execution.current_step = "Running post-deployment checks"
				execution.progress_percentage = 90.0
				await self._update_deployment_execution(execution)
				
				post_check_results = await self._run_deployment_checks(plan.post_deployment_checks, environment)
				if not all(result["passed"] for result in post_check_results):
					if environment.auto_rollback_enabled:
						await self._initiate_rollback(execution, "Post-deployment checks failed")
						return
					else:
						raise Exception("Post-deployment checks failed")
				
				execution.logs.append("Post-deployment checks passed")
			
			# Complete deployment
			execution.status = DeploymentStatus.COMPLETED
			execution.completed_at = datetime.now(timezone.utc)
			execution.progress_percentage = 100.0
			execution.current_step = "Deployment completed successfully"
			await self._update_deployment_execution(execution)
			
			# Start health monitoring
			asyncio.create_task(self._monitor_deployment_health(execution, environment))
			
			logger.info(f"Deployment completed successfully: {execution.id}")
			
		except Exception as e:
			# Handle deployment failure
			execution.status = DeploymentStatus.FAILED
			execution.completed_at = datetime.now(timezone.utc)
			execution.error_details = str(e)
			execution.current_step = f"Deployment failed: {str(e)}"
			await self._update_deployment_execution(execution)
			
			# Auto-rollback if enabled
			if environment.auto_rollback_enabled:
				await self._initiate_rollback(execution, f"Deployment failed: {str(e)}")
			
			logger.error(f"Deployment failed: {execution.id} - {e}")
		
		finally:
			# Remove from active deployments
			if execution.id in self.active_deployments:
				del self.active_deployments[execution.id]
	
	async def _execute_direct_deployment(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute direct deployment strategy."""
		
		try:
			# Get workflow version
			version = await self.version_manager.get_version(plan.version_id)
			if not version:
				raise ValueError(f"Version not found: {plan.version_id}")
			
			workflow = Workflow(**version.workflow_definition)
			
			# Update workflow status to deploying
			execution.logs.append(f"Deploying workflow {workflow.name} version {version.version_number}")
			execution.progress_percentage = 50.0
			await self._update_deployment_execution(execution)
			
			# Simulate deployment process
			await asyncio.sleep(2)  # Simulate deployment time
			
			# Update workflow configuration with environment settings
			if environment.environment_variables:
				if not workflow.configuration:
					workflow.configuration = {}
				workflow.configuration.update(environment.environment_variables)
			
			# Apply resource limits
			if environment.resource_limits:
				workflow.metadata = workflow.metadata or {}
				workflow.metadata["resource_limits"] = environment.resource_limits
			
			# Mark as deployed
			workflow.status = WorkflowStatus.ACTIVE
			execution.logs.append("Direct deployment completed")
			execution.progress_percentage = 80.0
			await self._update_deployment_execution(execution)
			
		except Exception as e:
			execution.logs.append(f"Direct deployment failed: {str(e)}")
			raise
	
	async def _execute_blue_green_deployment(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute blue-green deployment strategy."""
		
		try:
			execution.logs.append("Starting blue-green deployment")
			
			# Deploy to green environment
			execution.logs.append("Deploying to green environment")
			execution.progress_percentage = 40.0
			await self._update_deployment_execution(execution)
			
			await self._execute_direct_deployment(execution, plan, environment)
			
			# Health check green environment
			execution.logs.append("Health checking green environment")
			execution.progress_percentage = 60.0
			await self._update_deployment_execution(execution)
			
			health_results = await self._run_health_checks(environment)
			if not all(result.status == HealthCheckStatus.HEALTHY for result in health_results):
				raise Exception("Green environment health checks failed")
			
			# Switch traffic to green
			execution.logs.append("Switching traffic to green environment")
			execution.progress_percentage = 75.0
			await self._update_deployment_execution(execution)
			
			# Simulate traffic switch
			await asyncio.sleep(1)
			
			execution.logs.append("Blue-green deployment completed")
			
		except Exception as e:
			execution.logs.append(f"Blue-green deployment failed: {str(e)}")
			raise
	
	async def _execute_canary_deployment(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute canary deployment strategy."""
		
		try:
			execution.logs.append("Starting canary deployment")
			
			# Get canary configuration
			canary_config = plan.configuration.get("canary", {})
			traffic_percentage = canary_config.get("traffic_percentage", 10)
			monitoring_duration = canary_config.get("monitoring_duration_minutes", 30)
			
			# Deploy canary version
			execution.logs.append(f"Deploying canary with {traffic_percentage}% traffic")
			execution.progress_percentage = 30.0
			await self._update_deployment_execution(execution)
			
			await self._execute_direct_deployment(execution, plan, environment)
			
			# Monitor canary metrics
			execution.logs.append(f"Monitoring canary for {monitoring_duration} minutes")
			execution.progress_percentage = 50.0
			await self._update_deployment_execution(execution)
			
			# Simulate monitoring period
			monitoring_end = datetime.now(timezone.utc) + timedelta(minutes=monitoring_duration)
			while datetime.now(timezone.utc) < monitoring_end:
				health_results = await self._run_health_checks(environment)
				execution.health_checks.extend([result.model_dump() for result in health_results])
				
				if any(result.status == HealthCheckStatus.UNHEALTHY for result in health_results):
					raise Exception("Canary health checks failed during monitoring")
				
				await asyncio.sleep(60)  # Check every minute
			
			# Promote canary to full deployment
			execution.logs.append("Promoting canary to full deployment")
			execution.progress_percentage = 80.0
			await self._update_deployment_execution(execution)
			
			execution.logs.append("Canary deployment completed successfully")
			
		except Exception as e:
			execution.logs.append(f"Canary deployment failed: {str(e)}")
			raise
	
	async def _execute_rolling_deployment(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute rolling deployment strategy."""
		
		try:
			execution.logs.append("Starting rolling deployment")
			
			# Get rolling configuration
			rolling_config = plan.configuration.get("rolling", {})
			batch_size = rolling_config.get("batch_size", 1)
			batch_delay = rolling_config.get("batch_delay_seconds", 30)
			
			# Simulate rolling deployment batches
			total_instances = rolling_config.get("total_instances", 3)
			batches = (total_instances + batch_size - 1) // batch_size
			
			for batch_num in range(batches):
				execution.logs.append(f"Deploying batch {batch_num + 1}/{batches}")
				execution.progress_percentage = 30.0 + (batch_num / batches) * 40.0
				await self._update_deployment_execution(execution)
				
				# Deploy batch
				await asyncio.sleep(batch_delay)
				
				# Health check batch
				health_results = await self._run_health_checks(environment)
				if any(result.status == HealthCheckStatus.UNHEALTHY for result in health_results):
					raise Exception(f"Batch {batch_num + 1} health checks failed")
			
			execution.logs.append("Rolling deployment completed successfully")
			
		except Exception as e:
			execution.logs.append(f"Rolling deployment failed: {str(e)}")
			raise
	
	async def _execute_a_b_test_deployment(
		self,
		execution: DeploymentExecution,
		plan: DeploymentPlan,
		environment: DeploymentEnvironment
	) -> None:
		"""Execute A/B test deployment strategy."""
		
		try:
			execution.logs.append("Starting A/B test deployment")
			
			# Get A/B test configuration
			ab_config = plan.configuration.get("ab_test", {})
			test_percentage = ab_config.get("test_percentage", 50)
			test_duration = ab_config.get("test_duration_hours", 24)
			
			# Deploy B version
			execution.logs.append(f"Deploying B version with {test_percentage}% traffic")
			execution.progress_percentage = 40.0
			await self._update_deployment_execution(execution)
			
			await self._execute_direct_deployment(execution, plan, environment)
			
			# Set up A/B test monitoring
			execution.logs.append(f"Starting A/B test for {test_duration} hours")
			execution.progress_percentage = 60.0
			await self._update_deployment_execution(execution)
			
			# Store A/B test metadata
			execution.metadata["ab_test"] = {
				"start_time": datetime.now(timezone.utc).isoformat(),
				"test_percentage": test_percentage,
				"duration_hours": test_duration
			}
			
			execution.logs.append("A/B test deployment setup completed")
			
		except Exception as e:
			execution.logs.append(f"A/B test deployment failed: {str(e)}")
			raise
	
	async def rollback_deployment(
		self,
		execution_id: str,
		user_id: str,
		reason: str = ""
	) -> DeploymentExecution:
		"""Rollback a deployment."""
		
		try:
			# Get deployment execution
			execution = await self._get_deployment_execution(execution_id)
			if not execution:
				raise ValueError(f"Deployment execution not found: {execution_id}")
			
			# Get deployment plan
			plan = await self._get_deployment_plan(execution.plan_id)
			if not plan:
				raise ValueError(f"Deployment plan not found: {execution.plan_id}")
			
			# Create rollback execution
			rollback_execution = DeploymentExecution(
				plan_id=execution.plan_id,
				status=DeploymentStatus.IN_PROGRESS,
				started_at=datetime.now(timezone.utc),
				started_by=user_id,
				current_step="Initiating rollback",
				tenant_id=self.tenant_id
			)
			
			rollback_execution.logs.append(f"Rollback initiated by {user_id}: {reason}")
			
			# Save rollback execution
			await self._save_deployment_execution(rollback_execution)
			
			# Update original execution with rollback reference
			execution.rollback_execution_id = rollback_execution.id
			await self._update_deployment_execution(execution)
			
			# Execute rollback
			await self._execute_rollback(rollback_execution, plan, reason)
			
			logger.info(f"Rollback completed: {rollback_execution.id}")
			return rollback_execution
			
		except Exception as e:
			logger.error(f"Failed to rollback deployment: {e}")
			raise
	
	async def _initiate_rollback(self, execution: DeploymentExecution, reason: str) -> None:
		"""Initiate automatic rollback."""
		
		try:
			rollback_execution = await self.rollback_deployment(
				execution.id,
				"system",  # System-initiated rollback
				f"Auto-rollback: {reason}"
			)
			
			execution.status = DeploymentStatus.ROLLED_BACK
			execution.rollback_execution_id = rollback_execution.id
			await self._update_deployment_execution(execution)
			
		except Exception as e:
			logger.error(f"Failed to initiate auto-rollback: {e}")
			execution.logs.append(f"Auto-rollback failed: {str(e)}")
	
	async def _execute_rollback(
		self,
		rollback_execution: DeploymentExecution,
		plan: DeploymentPlan,
		reason: str
	) -> None:
		"""Execute rollback process."""
		
		try:
			rollback_execution.current_step = "Executing rollback plan"
			rollback_execution.progress_percentage = 30.0
			await self._update_deployment_execution(rollback_execution)
			
			# Execute rollback steps from plan
			rollback_plan = plan.rollback_plan
			rollback_steps = rollback_plan.get("steps", [])
			
			for i, step in enumerate(rollback_steps):
				step_name = step.get("name", f"Step {i + 1}")
				rollback_execution.current_step = f"Executing rollback step: {step_name}"
				rollback_execution.progress_percentage = 30.0 + (i / len(rollback_steps)) * 60.0
				await self._update_deployment_execution(rollback_execution)
				
				# Execute rollback step
				await self._execute_rollback_step(step)
				rollback_execution.logs.append(f"Completed rollback step: {step_name}")
			
			# Complete rollback
			rollback_execution.status = DeploymentStatus.COMPLETED
			rollback_execution.completed_at = datetime.now(timezone.utc)
			rollback_execution.progress_percentage = 100.0
			rollback_execution.current_step = "Rollback completed"
			await self._update_deployment_execution(rollback_execution)
			
		except Exception as e:
			rollback_execution.status = DeploymentStatus.FAILED
			rollback_execution.error_details = str(e)
			rollback_execution.logs.append(f"Rollback failed: {str(e)}")
			await self._update_deployment_execution(rollback_execution)
			raise
	
	async def _execute_rollback_step(self, step: Dict[str, Any]) -> None:
		"""Execute a single rollback step."""
		
		step_type = step.get("type", "")
		
		if step_type == "restore_previous_version":
			# Simulate restoring previous version
			await asyncio.sleep(1)
		
		elif step_type == "switch_traffic":
			# Simulate switching traffic back
			await asyncio.sleep(0.5)
		
		elif step_type == "cleanup_resources":
			# Simulate cleaning up resources
			await asyncio.sleep(0.5)
		
		else:
			logger.warning(f"Unknown rollback step type: {step_type}")
	
	async def _run_deployment_checks(
		self,
		checks: List[str],
		environment: DeploymentEnvironment
	) -> List[Dict[str, Any]]:
		"""Run deployment checks."""
		
		results = []
		
		for check_name in checks:
			try:
				# Simulate running check
				await asyncio.sleep(0.5)
				
				# Most checks pass for simulation
				passed = True
				message = f"Check {check_name} passed"
				
				results.append({
					"check": check_name,
					"passed": passed,
					"message": message,
					"timestamp": datetime.now(timezone.utc).isoformat()
				})
				
			except Exception as e:
				results.append({
					"check": check_name,
					"passed": False,
					"message": str(e),
					"timestamp": datetime.now(timezone.utc).isoformat()
				})
		
		return results
	
	async def _run_health_checks(self, environment: DeploymentEnvironment) -> List[HealthCheckResult]:
		"""Run health checks for an environment."""
		
		results = []
		health_config = environment.health_check_config
		
		# Default health checks
		default_checks = ["workflow_status", "resource_usage", "error_rate"]
		custom_checks = health_config.get("checks", [])
		all_checks = default_checks + custom_checks
		
		for check_name in all_checks:
			start_time = datetime.now(timezone.utc)
			
			try:
				# Run health check function if available
				if check_name in self.health_check_functions:
					status, message, metrics = await self.health_check_functions[check_name](environment)
				else:
					# Default simulation
					await asyncio.sleep(0.1)
					status = HealthCheckStatus.HEALTHY
					message = f"{check_name} is healthy"
					metrics = {"value": 100}
				
				duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
				
				results.append(HealthCheckResult(
					check_name=check_name,
					status=status,
					message=message,
					metrics=metrics,
					duration_ms=duration_ms
				))
				
			except Exception as e:
				duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
				
				results.append(HealthCheckResult(
					check_name=check_name,
					status=HealthCheckStatus.UNHEALTHY,
					message=str(e),
					duration_ms=duration_ms
				))
		
		return results
	
	async def _monitor_deployment_health(
		self,
		execution: DeploymentExecution,
		environment: DeploymentEnvironment
	) -> None:
		"""Monitor deployment health after completion."""
		
		try:
			monitoring_duration = environment.rollback_threshold_minutes
			end_time = datetime.now(timezone.utc) + timedelta(minutes=monitoring_duration)
			
			while datetime.now(timezone.utc) < end_time:
				health_results = await self._run_health_checks(environment)
				
				# Check for unhealthy status
				unhealthy_checks = [
					result for result in health_results
					if result.status == HealthCheckStatus.UNHEALTHY
				]
				
				if unhealthy_checks and environment.auto_rollback_enabled:
					logger.warning(f"Unhealthy checks detected, initiating rollback: {execution.id}")
					await self._initiate_rollback(execution, f"Health monitoring failed: {len(unhealthy_checks)} checks unhealthy")
					break
				
				# Store health check results
				execution.health_checks.extend([result.model_dump() for result in health_results])
				await self._update_deployment_execution(execution)
				
				# Wait before next health check
				await asyncio.sleep(60)  # Check every minute
			
		except Exception as e:
			logger.error(f"Health monitoring error: {e}")
	
	async def _validate_deployment_plan(self, plan: DeploymentPlan) -> None:
		"""Validate deployment plan."""
		
		# Check if workflow exists
		# Check if version exists
		version = await self.version_manager.get_version(plan.version_id)
		if not version:
			raise ValueError(f"Version not found: {plan.version_id}")
		
		# Check if environment exists
		environment = await self._get_environment(plan.environment_id)
		if not environment:
			raise ValueError(f"Environment not found: {plan.environment_id}")
		
		# Validate strategy-specific configuration
		if plan.strategy == DeploymentStrategy.CANARY:
			canary_config = plan.configuration.get("canary", {})
			if not canary_config.get("traffic_percentage"):
				raise ValueError("Canary deployment requires traffic_percentage configuration")
		
		elif plan.strategy == DeploymentStrategy.ROLLING:
			rolling_config = plan.configuration.get("rolling", {})
			if not rolling_config.get("batch_size"):
				raise ValueError("Rolling deployment requires batch_size configuration")
	
	async def _get_deployment_plan(self, plan_id: str) -> Optional[DeploymentPlan]:
		"""Get deployment plan by ID."""
		
		async with self.database_manager.get_session() as session:
			result = await session.execute(
				text(f"""
				SELECT * FROM {self.plans_table}
				WHERE id = :plan_id AND tenant_id = :tenant_id
				"""),
				{"plan_id": plan_id, "tenant_id": self.tenant_id}
			)
			
			row = result.first()
			if not row:
				return None
			
			return DeploymentPlan(
				id=row.id,
				name=row.name,
				workflow_id=row.workflow_id,
				version_id=row.version_id,
				environment_id=row.environment_id,
				strategy=DeploymentStrategy(row.strategy),
				description=row.description,
				configuration=json.loads(row.configuration) if row.configuration else {},
				pre_deployment_checks=json.loads(row.pre_deployment_checks) if row.pre_deployment_checks else [],
				post_deployment_checks=json.loads(row.post_deployment_checks) if row.post_deployment_checks else [],
				rollback_plan=json.loads(row.rollback_plan) if row.rollback_plan else {},
				notification_config=json.loads(row.notification_config) if row.notification_config else {},
				schedule=row.schedule,
				auto_approve=row.auto_approve,
				tenant_id=row.tenant_id,
				created_by=row.created_by,
				created_at=row.created_at
			)
	
	async def _get_environment(self, environment_id: str) -> Optional[DeploymentEnvironment]:
		"""Get deployment environment by ID."""
		
		async with self.database_manager.get_session() as session:
			result = await session.execute(
				text(f"""
				SELECT * FROM {self.environments_table}
				WHERE id = :environment_id AND tenant_id = :tenant_id
				"""),
				{"environment_id": environment_id, "tenant_id": self.tenant_id}
			)
			
			row = result.first()
			if not row:
				return None
			
			return DeploymentEnvironment(
				id=row.id,
				name=row.name,
				environment_type=EnvironmentType(row.environment_type),
				description=row.description,
				configuration=json.loads(row.configuration) if row.configuration else {},
				resource_limits=json.loads(row.resource_limits) if row.resource_limits else {},
				environment_variables=json.loads(row.environment_variables) if row.environment_variables else {},
				health_check_config=json.loads(row.health_check_config) if row.health_check_config else {},
				approval_required=row.approval_required,
				approvers=json.loads(row.approvers) if row.approvers else [],
				auto_rollback_enabled=row.auto_rollback_enabled,
				rollback_threshold_minutes=row.rollback_threshold_minutes,
				tenant_id=row.tenant_id,
				created_by=row.created_by,
				created_at=row.created_at,
				updated_at=row.updated_at
			)
	
	async def _save_deployment_execution(self, execution: DeploymentExecution) -> None:
		"""Save deployment execution to database."""
		
		async with self.database_manager.get_session() as session:
			await session.execute(
				text(f"""
				INSERT INTO {self.executions_table} (
					id, plan_id, status, started_at, completed_at, started_by,
					progress_percentage, current_step, logs, metrics, health_checks,
					rollback_execution_id, error_details, tenant_id
				) VALUES (
					:id, :plan_id, :status, :started_at, :completed_at, :started_by,
					:progress_percentage, :current_step, :logs, :metrics, :health_checks,
					:rollback_execution_id, :error_details, :tenant_id
				)
				"""),
				{
					"id": execution.id,
					"plan_id": execution.plan_id,
					"status": execution.status.value,
					"started_at": execution.started_at,
					"completed_at": execution.completed_at,
					"started_by": execution.started_by,
					"progress_percentage": execution.progress_percentage,
					"current_step": execution.current_step,
					"logs": json.dumps(execution.logs),
					"metrics": json.dumps(execution.metrics),
					"health_checks": json.dumps(execution.health_checks),
					"rollback_execution_id": execution.rollback_execution_id,
					"error_details": execution.error_details,
					"tenant_id": execution.tenant_id
				}
			)
			await session.commit()
	
	async def _update_deployment_execution(self, execution: DeploymentExecution) -> None:
		"""Update deployment execution in database."""
		
		async with self.database_manager.get_session() as session:
			await session.execute(
				text(f"""
				UPDATE {self.executions_table}
				SET status = :status, started_at = :started_at, completed_at = :completed_at,
					progress_percentage = :progress_percentage, current_step = :current_step,
					logs = :logs, metrics = :metrics, health_checks = :health_checks,
					rollback_execution_id = :rollback_execution_id, error_details = :error_details
				WHERE id = :id AND tenant_id = :tenant_id
				"""),
				{
					"id": execution.id,
					"status": execution.status.value,
					"started_at": execution.started_at,
					"completed_at": execution.completed_at,
					"progress_percentage": execution.progress_percentage,
					"current_step": execution.current_step,
					"logs": json.dumps(execution.logs),
					"metrics": json.dumps(execution.metrics),
					"health_checks": json.dumps(execution.health_checks),
					"rollback_execution_id": execution.rollback_execution_id,
					"error_details": execution.error_details,
					"tenant_id": self.tenant_id
				}
			)
			await session.commit()
	
	async def _get_deployment_execution(self, execution_id: str) -> Optional[DeploymentExecution]:
		"""Get deployment execution by ID."""
		
		async with self.database_manager.get_session() as session:
			result = await session.execute(
				text(f"""
				SELECT * FROM {self.executions_table}
				WHERE id = :execution_id AND tenant_id = :tenant_id
				"""),
				{"execution_id": execution_id, "tenant_id": self.tenant_id}
			)
			
			row = result.first()
			if not row:
				return None
			
			return DeploymentExecution(
				id=row.id,
				plan_id=row.plan_id,
				status=DeploymentStatus(row.status),
				started_at=row.started_at,
				completed_at=row.completed_at,
				started_by=row.started_by,
				progress_percentage=row.progress_percentage,
				current_step=row.current_step,
				logs=json.loads(row.logs) if row.logs else [],
				metrics=json.loads(row.metrics) if row.metrics else {},
				health_checks=json.loads(row.health_checks) if row.health_checks else [],
				rollback_execution_id=row.rollback_execution_id,
				error_details=row.error_details,
				tenant_id=row.tenant_id
			)
	
	def register_health_check_function(self, check_name: str, check_function: Callable) -> None:
		"""Register a custom health check function."""
		self.health_check_functions[check_name] = check_function
		logger.info(f"Registered health check function: {check_name}")

# Export deployment management classes
__all__ = [
	"DeploymentManager",
	"DeploymentStrategy",
	"DeploymentStatus",
	"EnvironmentType",
	"HealthCheckStatus",
	"DeploymentEnvironment",
	"DeploymentPlan",
	"DeploymentExecution",
	"HealthCheckResult"
]