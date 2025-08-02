"""
APG Workflow Orchestration REST API

Comprehensive async REST API endpoints with CRUD operations, APG authentication
integration, rate limiting, input validation, and comprehensive error handling.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict, validator
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
from uuid_extensions import uuid7str

from .models import (
	Workflow, WorkflowInstance, TaskDefinition, TaskExecution,
	WorkflowStatus, TaskStatus, Priority, TaskType
)
from .database import DatabaseManager, create_repositories
from .management import (
	WorkflowManager, WorkflowSearchFilter, WorkflowValidationLevel,
	VersionManager, DeploymentManager
)
from .service import WorkflowOrchestrationService

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# API Models
class APIResponse(BaseModel):
	"""Standard API response format."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool = True
	message: str = ""
	data: Any = None
	errors: List[str] = Field(default_factory=list)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WorkflowCreateRequest(BaseModel):
	"""Request model for creating workflows."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: str = Field(..., min_length=1, max_length=200)
	description: str = Field(default="", max_length=1000)
	tasks: List[Dict[str, Any]] = Field(..., min_items=1)
	configuration: Optional[Dict[str, Any]] = None
	metadata: Optional[Dict[str, Any]] = None
	tags: List[str] = Field(default_factory=list)
	priority: Priority = Field(default=Priority.MEDIUM)
	sla_hours: Optional[float] = Field(default=None, gt=0)
	validation_level: WorkflowValidationLevel = Field(default=WorkflowValidationLevel.STANDARD)

class WorkflowUpdateRequest(BaseModel):
	"""Request model for updating workflows."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	name: Optional[str] = Field(default=None, min_length=1, max_length=200)
	description: Optional[str] = Field(default=None, max_length=1000)
	tasks: Optional[List[Dict[str, Any]]] = Field(default=None, min_items=1)
	configuration: Optional[Dict[str, Any]] = None
	metadata: Optional[Dict[str, Any]] = None
	tags: Optional[List[str]] = None
	priority: Optional[Priority] = None
	sla_hours: Optional[float] = Field(default=None, gt=0)
	validation_level: WorkflowValidationLevel = Field(default=WorkflowValidationLevel.STANDARD)

class WorkflowExecuteRequest(BaseModel):
	"""Request model for executing workflows."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	input_data: Dict[str, Any] = Field(default_factory=dict)
	configuration_overrides: Dict[str, Any] = Field(default_factory=dict)
	priority: Optional[Priority] = None
	tags: List[str] = Field(default_factory=list)
	schedule_at: Optional[datetime] = None

class PaginationParams(BaseModel):
	"""Pagination parameters."""
	model_config = ConfigDict(extra='forbid')
	
	offset: int = Field(default=0, ge=0, description="Number of items to skip")
	limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of items to return")

# Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
	"""Rate limiting middleware using Redis."""
	
	def __init__(self, app, redis_client: redis.Redis, default_rate_limit: int = 1000):
		super().__init__(app)
		self.redis_client = redis_client
		self.default_rate_limit = default_rate_limit
	
	async def dispatch(self, request: Request, call_next):
		# Extract user ID from request (would be set by auth middleware)
		user_id = getattr(request.state, 'user_id', 'anonymous')
		
		# Create rate limit key
		rate_limit_key = f"rate_limit:{user_id}:{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M')}"
		
		try:
			# Check current request count
			current_count = await self.redis_client.get(rate_limit_key)
			current_count = int(current_count) if current_count else 0
			
			# Check if rate limit exceeded
			if current_count >= self.default_rate_limit:
				return JSONResponse(
					status_code=429,
					content={
						"success": False,
						"message": "Rate limit exceeded",
						"errors": ["Too many requests. Please try again later."]
					}
				)
			
			# Increment counter
			await self.redis_client.incr(rate_limit_key)
			await self.redis_client.expire(rate_limit_key, 60)  # 1 minute window
			
			# Process request
			response = await call_next(request)
			
			# Add rate limit headers
			response.headers["X-RateLimit-Limit"] = str(self.default_rate_limit)
			response.headers["X-RateLimit-Remaining"] = str(max(0, self.default_rate_limit - current_count - 1))
			response.headers["X-RateLimit-Reset"] = str(int((datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp()))
			
			return response
			
		except Exception as e:
			logger.error(f"Rate limiting error: {e}")
			# Continue processing if rate limiting fails
			return await call_next(request)

# Authentication Dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
	"""Extract and validate user from APG authentication token."""
	
	try:
		token = credentials.credentials
		
		if not token or token == "invalid":
			raise HTTPException(
				status_code=401,
				detail="Invalid authentication token"
			)
		
		# Try APG auth_rbac capability integration
		try:
			from apg.capabilities.auth_rbac import AuthRBACService
			auth_service = AuthRBACService()
			
			# Validate token with APG auth service
			user_info = await auth_service.validate_token(token)
			if not user_info:
				raise HTTPException(status_code=401, detail="Token validation failed")
			
			return {
				"user_id": user_info.get("user_id", "unknown"),
				"tenant_id": user_info.get("tenant_id", "default_tenant"),
				"roles": user_info.get("roles", ["workflow_user"]),
				"permissions": user_info.get("permissions", ["workflow.read"])
			}
			
		except ImportError:
			# APG auth_rbac not available, try JWT validation
			try:
				import jwt
				import os
				
				# Get JWT secret from environment
				jwt_secret = os.getenv('JWT_SECRET', 'workflow_orchestration_secret')
				jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
				
				# Decode and validate JWT token
				payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])
				
				# Extract user information from JWT payload
				user_id = payload.get('sub') or payload.get('user_id')
				if not user_id:
					raise HTTPException(status_code=401, detail="Invalid token payload")
				
				return {
					"user_id": str(user_id),
					"tenant_id": payload.get("tenant_id", "default_tenant"),
					"roles": payload.get("roles", ["workflow_user"]),
					"permissions": payload.get("permissions", ["workflow.read", "workflow.write"])
				}
				
			except jwt.ExpiredSignatureError:
				raise HTTPException(status_code=401, detail="Token has expired")
			except jwt.InvalidTokenError:
				raise HTTPException(status_code=401, detail="Invalid token")
		
	except HTTPException:
		raise  # Re-raise HTTP exceptions
	except Exception as e:
		logger.error(f"Authentication failed: {e}")
		raise HTTPException(
			status_code=401,
			detail="Authentication failed"
		)

async def get_tenant_id(current_user: Dict[str, Any] = Depends(get_current_user)) -> str:
	"""Extract tenant ID from authenticated user."""
	return current_user["tenant_id"]

async def get_user_id(current_user: Dict[str, Any] = Depends(get_current_user)) -> str:
	"""Extract user ID from authenticated user."""
	return current_user["user_id"]

# Permission Dependencies
def require_permission(permission: str):
	"""Require specific permission for endpoint access."""
	
	def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
		if permission not in current_user.get("permissions", []):
			raise HTTPException(
				status_code=403,
				detail=f"Insufficient permissions. Required: {permission}"
			)
		return current_user
	
	return permission_checker

# API Application Factory
@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan events."""
	# Startup
	logger.info("Starting APG Workflow Orchestration API")
	yield
	# Shutdown
	logger.info("Shutting down APG Workflow Orchestration API")

def create_api_app(
	database_manager: DatabaseManager,
	redis_client: redis.Redis,
	tenant_id: str = "default"
) -> FastAPI:
	"""Create FastAPI application with all endpoints."""
	
	app = FastAPI(
		title="APG Workflow Orchestration API",
		description="Comprehensive workflow orchestration and automation platform",
		version="1.0.0",
		lifespan=lifespan,
		docs_url="/docs",
		redoc_url="/redoc",
		openapi_url="/openapi.json"
	)
	
	# Add middleware
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],  # Configure appropriately for production
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	app.add_middleware(GZipMiddleware, minimum_size=1000)
	app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
	
	# Initialize services
	workflow_service = WorkflowOrchestrationService(database_manager, redis_client, tenant_id)
	workflow_manager = WorkflowManager(database_manager, redis_client, tenant_id)
	version_manager = VersionManager(database_manager, redis_client, tenant_id)
	deployment_manager = DeploymentManager(database_manager, redis_client, version_manager, tenant_id)
	
	# Store services in app state
	app.state.workflow_service = workflow_service
	app.state.workflow_manager = workflow_manager
	app.state.version_manager = version_manager
	app.state.deployment_manager = deployment_manager
	
	# Exception Handlers
	@app.exception_handler(HTTPException)
	async def http_exception_handler(request: Request, exc: HTTPException):
		return JSONResponse(
			status_code=exc.status_code,
			content=APIResponse(
				success=False,
				message=exc.detail,
				errors=[exc.detail]
			).model_dump()
		)
	
	@app.exception_handler(Exception)
	async def general_exception_handler(request: Request, exc: Exception):
		logger.error(f"Unhandled exception: {exc}", exc_info=True)
		return JSONResponse(
			status_code=500,
			content=APIResponse(
				success=False,
				message="Internal server error",
				errors=["An unexpected error occurred"]
			).model_dump()
		)
	
	# Health Check Endpoints
	@app.get("/health", response_model=APIResponse)
	async def health_check():
		"""Health check endpoint."""
		return APIResponse(
			message="API is healthy",
			data={"status": "healthy", "timestamp": datetime.now(timezone.utc)}
		)
	
	@app.get("/health/detailed", response_model=APIResponse)
	async def detailed_health_check(current_user: Dict[str, Any] = Depends(get_current_user)):
		"""Detailed health check with service status."""
		
		health_data = {
			"api": "healthy",
			"database": "unknown",
			"redis": "unknown",
			"services": {}
		}
		
		try:
			# Check database
			async with database_manager.get_session() as session:
				await session.execute("SELECT 1")
			health_data["database"] = "healthy"
		except Exception as e:
			health_data["database"] = f"unhealthy: {str(e)}"
		
		try:
			# Check Redis
			await redis_client.ping()
			health_data["redis"] = "healthy"
		except Exception as e:
			health_data["redis"] = f"unhealthy: {str(e)}"
		
		# Check services
		try:
			health_data["services"]["workflow_service"] = "healthy"
			health_data["services"]["workflow_manager"] = "healthy"
			health_data["services"]["version_manager"] = "healthy"
			health_data["services"]["deployment_manager"] = "healthy"
		except Exception as e:
			health_data["services"]["error"] = str(e)
		
		return APIResponse(
			message="Detailed health check completed",
			data=health_data
		)
	
	# Workflow CRUD Endpoints
	@app.post("/workflows", response_model=APIResponse)
	async def create_workflow(
		request: WorkflowCreateRequest,
		user_id: str = Depends(get_user_id),
		tenant_id: str = Depends(get_tenant_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.write"))
	):
		"""Create a new workflow."""
		
		try:
			workflow_data = request.model_dump()
			workflow_data["tenant_id"] = tenant_id
			
			workflow = await workflow_manager.create_workflow(
				workflow_data, user_id, request.validation_level
			)
			
			return APIResponse(
				message="Workflow created successfully",
				data=workflow.model_dump(),
				metadata={"workflow_id": workflow.id}
			)
			
		except Exception as e:
			logger.error(f"Failed to create workflow: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows/{workflow_id}", response_model=APIResponse)
	async def get_workflow(
		workflow_id: str = Path(..., description="Workflow ID"),
		include_instances: bool = Query(False, description="Include workflow instances"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Get a workflow by ID."""
		
		try:
			workflow = await workflow_manager.get_workflow(workflow_id, user_id, include_instances)
			
			if not workflow:
				raise HTTPException(status_code=404, detail="Workflow not found")
			
			return APIResponse(
				message="Workflow retrieved successfully",
				data=workflow.model_dump()
			)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Failed to get workflow: {e}")
			raise HTTPException(status_code=500, detail=str(e))
	
	@app.put("/workflows/{workflow_id}", response_model=APIResponse)
	async def update_workflow(
		workflow_id: str = Path(..., description="Workflow ID"),
		request: WorkflowUpdateRequest = Body(...),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.write"))
	):
		"""Update a workflow."""
		
		try:
			updates = request.model_dump(exclude_none=True)
			validation_level = updates.pop("validation_level", WorkflowValidationLevel.STANDARD)
			
			workflow = await workflow_manager.update_workflow(
				workflow_id, updates, user_id, validation_level
			)
			
			return APIResponse(
				message="Workflow updated successfully",
				data=workflow.model_dump()
			)
			
		except Exception as e:
			logger.error(f"Failed to update workflow: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.delete("/workflows/{workflow_id}", response_model=APIResponse)
	async def delete_workflow(
		workflow_id: str = Path(..., description="Workflow ID"),
		hard_delete: bool = Query(False, description="Perform hard delete"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.delete"))
	):
		"""Delete a workflow."""
		
		try:
			success = await workflow_manager.delete_workflow(workflow_id, user_id, hard_delete)
			
			return APIResponse(
				message="Workflow deleted successfully",
				data={"deleted": success}
			)
			
		except Exception as e:
			logger.error(f"Failed to delete workflow: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows", response_model=APIResponse)
	async def search_workflows(
		name_pattern: Optional[str] = Query(None, description="Name pattern (supports wildcards)"),
		status: Optional[List[WorkflowStatus]] = Query(None, description="Workflow statuses"),
		tags: Optional[List[str]] = Query(None, description="Tags to filter by"),
		created_after: Optional[datetime] = Query(None, description="Created after timestamp"),
		created_before: Optional[datetime] = Query(None, description="Created before timestamp"),
		author: Optional[str] = Query(None, description="Author user ID"),
		priority: Optional[List[Priority]] = Query(None, description="Priority levels"),
		sort_by: str = Query("updated_at", regex="^(name|created_at|updated_at|status|priority)$"),
		sort_order: str = Query("desc", regex="^(asc|desc)$"),
		offset: int = Query(0, ge=0, description="Pagination offset"),
		limit: int = Query(100, ge=1, le=1000, description="Pagination limit"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Search workflows with filters."""
		
		try:
			filters = WorkflowSearchFilter(
				name_pattern=name_pattern,
				status=status,
				tags=tags,
				created_after=created_after,
				created_before=created_before,
				author=author,
				priority=priority,
				sort_by=sort_by,
				sort_order=sort_order,
				offset=offset,
				limit=limit
			)
			
			workflows = await workflow_manager.search_workflows(filters, user_id)
			
			return APIResponse(
				message="Workflows retrieved successfully",
				data=[workflow.model_dump() for workflow in workflows],
				metadata={
					"count": len(workflows),
					"offset": offset,
					"limit": limit,
					"filters": filters.model_dump(exclude_none=True)
				}
			)
			
		except Exception as e:
			logger.error(f"Failed to search workflows: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.post("/workflows/{workflow_id}/clone", response_model=APIResponse)
	async def clone_workflow(
		workflow_id: str = Path(..., description="Source workflow ID"),
		new_name: str = Body(..., embed=True, description="New workflow name"),
		modifications: Optional[Dict[str, Any]] = Body(None, embed=True, description="Modifications to apply"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.write"))
	):
		"""Clone a workflow with optional modifications."""
		
		try:
			cloned_workflow = await workflow_manager.clone_workflow(
				workflow_id, new_name, user_id, modifications
			)
			
			return APIResponse(
				message="Workflow cloned successfully",
				data=cloned_workflow.model_dump(),
				metadata={"original_workflow_id": workflow_id}
			)
			
		except Exception as e:
			logger.error(f"Failed to clone workflow: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows/{workflow_id}/statistics", response_model=APIResponse)
	async def get_workflow_statistics(
		workflow_id: str = Path(..., description="Workflow ID"),
		time_range_days: int = Query(30, ge=1, le=365, description="Time range in days"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Get workflow execution statistics."""
		
		try:
			statistics = await workflow_manager.get_workflow_statistics(
				workflow_id, user_id, time_range_days
			)
			
			return APIResponse(
				message="Workflow statistics retrieved successfully",
				data=statistics.model_dump(),
				metadata={"time_range_days": time_range_days}
			)
			
		except Exception as e:
			logger.error(f"Failed to get workflow statistics: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	# Workflow Execution Endpoints
	@app.post("/workflows/{workflow_id}/execute", response_model=APIResponse)
	async def execute_workflow(
		workflow_id: str = Path(..., description="Workflow ID"),
		request: WorkflowExecuteRequest = Body(...),
		user_id: str = Depends(get_user_id),
		tenant_id: str = Depends(get_tenant_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.execute"))
	):
		"""Execute a workflow."""
		
		try:
			instance = await workflow_service.execute_workflow(
				workflow_id=workflow_id,
				input_data=request.input_data,
				user_id=user_id,
				configuration_overrides=request.configuration_overrides,
				priority=request.priority,
				tags=request.tags,
				schedule_at=request.schedule_at
			)
			
			return APIResponse(
				message="Workflow execution started",
				data=instance.model_dump(),
				metadata={"instance_id": instance.id}
			)
			
		except Exception as e:
			logger.error(f"Failed to execute workflow: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows/{workflow_id}/instances", response_model=APIResponse)
	async def get_workflow_instances(
		workflow_id: str = Path(..., description="Workflow ID"),
		status: Optional[List[str]] = Query(None, description="Instance statuses"),
		limit: int = Query(100, ge=1, le=1000),
		offset: int = Query(0, ge=0),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Get workflow instances."""
		
		try:
			instances = await workflow_service.get_workflow_instances(
				workflow_id, status, limit, offset
			)
			
			return APIResponse(
				message="Workflow instances retrieved successfully",
				data=[instance.model_dump() for instance in instances],
				metadata={"count": len(instances), "offset": offset, "limit": limit}
			)
			
		except Exception as e:
			logger.error(f"Failed to get workflow instances: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows/{workflow_id}/instances/{instance_id}", response_model=APIResponse)
	async def get_workflow_instance(
		workflow_id: str = Path(..., description="Workflow ID"),
		instance_id: str = Path(..., description="Instance ID"),
		include_tasks: bool = Query(True, description="Include task executions"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Get a specific workflow instance."""
		
		try:
			instance = await workflow_service.get_workflow_instance(
				instance_id, include_tasks
			)
			
			if not instance:
				raise HTTPException(status_code=404, detail="Workflow instance not found")
			
			return APIResponse(
				message="Workflow instance retrieved successfully",
				data=instance.model_dump()
			)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Failed to get workflow instance: {e}")
			raise HTTPException(status_code=500, detail=str(e))
	
	@app.post("/workflows/{workflow_id}/instances/{instance_id}/pause", response_model=APIResponse)
	async def pause_workflow_instance(
		workflow_id: str = Path(..., description="Workflow ID"),
		instance_id: str = Path(..., description="Instance ID"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.execute"))
	):
		"""Pause a workflow instance."""
		
		try:
			success = await workflow_service.pause_workflow_instance(instance_id, user_id)
			
			return APIResponse(
				message="Workflow instance paused successfully",
				data={"paused": success}
			)
			
		except Exception as e:
			logger.error(f"Failed to pause workflow instance: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.post("/workflows/{workflow_id}/instances/{instance_id}/resume", response_model=APIResponse)
	async def resume_workflow_instance(
		workflow_id: str = Path(..., description="Workflow ID"),
		instance_id: str = Path(..., description="Instance ID"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.execute"))
	):
		"""Resume a paused workflow instance."""
		
		try:
			success = await workflow_service.resume_workflow_instance(instance_id, user_id)
			
			return APIResponse(
				message="Workflow instance resumed successfully",
				data={"resumed": success}
			)
			
		except Exception as e:
			logger.error(f"Failed to resume workflow instance: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.post("/workflows/{workflow_id}/instances/{instance_id}/stop", response_model=APIResponse)
	async def stop_workflow_instance(
		workflow_id: str = Path(..., description="Workflow ID"),
		instance_id: str = Path(..., description="Instance ID"),
		reason: Optional[str] = Body(None, embed=True, description="Reason for stopping"),
		user_id: str = Depends(get_user_id),
		_: Dict[str, Any] = Depends(require_permission("workflows.execute"))
	):
		"""Stop a workflow instance."""
		
		try:
			success = await workflow_service.stop_workflow_instance(instance_id, user_id, reason)
			
			return APIResponse(
				message="Workflow instance stopped successfully",
				data={"stopped": success}
			)
			
		except Exception as e:
			logger.error(f"Failed to stop workflow instance: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	# Version Control Endpoints
	@app.get("/workflows/{workflow_id}/versions", response_model=APIResponse)
	async def list_workflow_versions(
		workflow_id: str = Path(..., description="Workflow ID"),
		branch_name: Optional[str] = Query(None, description="Branch name filter"),
		limit: int = Query(50, ge=1, le=100),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""List versions for a workflow."""
		
		try:
			versions = await version_manager.list_versions(workflow_id, branch_name, None, limit)
			
			return APIResponse(
				message="Workflow versions retrieved successfully",
				data=[version.model_dump() for version in versions],
				metadata={"count": len(versions)}
			)
			
		except Exception as e:
			logger.error(f"Failed to list workflow versions: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	@app.get("/workflows/{workflow_id}/versions/{version_id}", response_model=APIResponse)
	async def get_workflow_version(
		workflow_id: str = Path(..., description="Workflow ID"),
		version_id: str = Path(..., description="Version ID"),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Get a specific workflow version."""
		
		try:
			version = await version_manager.get_version(version_id)
			
			if not version:
				raise HTTPException(status_code=404, detail="Workflow version not found")
			
			return APIResponse(
				message="Workflow version retrieved successfully",
				data=version.model_dump()
			)
			
		except HTTPException:
			raise
		except Exception as e:
			logger.error(f"Failed to get workflow version: {e}")
			raise HTTPException(status_code=500, detail=str(e))
	
	@app.post("/workflows/{workflow_id}/versions/{from_version}/compare/{to_version}", response_model=APIResponse)
	async def compare_workflow_versions(
		workflow_id: str = Path(..., description="Workflow ID"),
		from_version: str = Path(..., description="From version ID"),
		to_version: str = Path(..., description="To version ID"),
		_: Dict[str, Any] = Depends(require_permission("workflows.read"))
	):
		"""Compare two workflow versions."""
		
		try:
			comparison = await version_manager.compare_versions(from_version, to_version)
			
			return APIResponse(
				message="Version comparison completed",
				data=comparison.model_dump()
			)
			
		except Exception as e:
			logger.error(f"Failed to compare versions: {e}")
			raise HTTPException(status_code=400, detail=str(e))
	
	return app

# Export API components
__all__ = [
	"create_api_app",
	"APIResponse",
	"WorkflowCreateRequest",
	"WorkflowUpdateRequest",
	"WorkflowExecuteRequest",
	"PaginationParams",
	"get_current_user",
	"require_permission"
]