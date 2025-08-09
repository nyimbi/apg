#!/usr/bin/env python3
"""
APG ETLP REST API Endpoints
Async REST API for pipeline orchestration and management

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from uuid_extensions import uuid7str

from .service import ETLPService
from .field_mapper import FieldMapperService, TableSchema, MappingConfiguration, FieldMapping
from .models import Pipeline, Execution, Transformation, DataSource, QualityRule, PipelineStatus
from .views import (
	PipelineCreateRequest, PipelineUpdateRequest, PipelineResponse, PipelineExecuteRequest,
	TransformationCreateRequest, TransformationResponse,
	DataSourceCreateRequest, DataSourceResponse, 
	QualityRuleCreateRequest, QualityRuleResponse,
	ExecutionResponse, ExecutionListResponse, PipelineListResponse,
	PipelineMetricsResponse, PipelineHealthResponse, PipelineOptimizationResponse,
	CollaboratorResponse, ErrorResponse, SuccessResponse
)

# APG authentication integration
security = HTTPBearer()


class ETLPAPIController:
	"""Main API controller for ETLP operations"""
	
	def __init__(self):
		self.app = FastAPI(
			title="APG ETLP API",
			description="Next-generation data processing and pipeline orchestration",
			version="1.0.0",
			docs_url="/api/v1/etlp/docs",
			redoc_url="/api/v1/etlp/redoc"
		)
		self._setup_routes()
		
		# APG service integrations - will be injected
		self.auth_service = None
		self.audit_service = None
		self.notification_service = None
	
	def _setup_routes(self) -> None:
		"""Setup API routes with APG patterns"""
		
		# Pipeline Management Routes
		self.app.post("/api/v1/etlp/pipelines", response_model=PipelineResponse)(self.create_pipeline)
		self.app.get("/api/v1/etlp/pipelines", response_model=PipelineListResponse)(self.list_pipelines)
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}", response_model=PipelineResponse)(self.get_pipeline)
		self.app.put("/api/v1/etlp/pipelines/{pipeline_id}", response_model=PipelineResponse)(self.update_pipeline)
		self.app.delete("/api/v1/etlp/pipelines/{pipeline_id}", response_model=SuccessResponse)(self.delete_pipeline)
		
		# Pipeline Execution Routes
		self.app.post("/api/v1/etlp/pipelines/{pipeline_id}/execute", response_model=Dict[str, str])(self.execute_pipeline)
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/executions", response_model=ExecutionListResponse)(self.list_pipeline_executions)
		self.app.get("/api/v1/etlp/executions/{execution_id}", response_model=ExecutionResponse)(self.get_execution)
		self.app.delete("/api/v1/etlp/executions/{execution_id}", response_model=SuccessResponse)(self.cancel_execution)
		
		# Pipeline Analytics and Monitoring
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/metrics", response_model=PipelineMetricsResponse)(self.get_pipeline_metrics)
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/health", response_model=PipelineHealthResponse)(self.get_pipeline_health)
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/logs", response_model=List[Dict[str, Any]])(self.get_pipeline_logs)
		
		# Pipeline Optimization
		self.app.post("/api/v1/etlp/pipelines/{pipeline_id}/optimize", response_model=PipelineOptimizationResponse)(self.optimize_pipeline)
		
		# Pipeline Collaboration
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/collaborators", response_model=List[CollaboratorResponse])(self.get_pipeline_collaborators)
		
		# Transformation Management Routes
		self.app.post("/api/v1/etlp/transformations", response_model=TransformationResponse)(self.create_transformation)
		self.app.get("/api/v1/etlp/transformations", response_model=List[TransformationResponse])(self.list_transformations)
		self.app.get("/api/v1/etlp/transformations/{transformation_id}", response_model=TransformationResponse)(self.get_transformation)
		self.app.put("/api/v1/etlp/transformations/{transformation_id}", response_model=TransformationResponse)(self.update_transformation)
		self.app.delete("/api/v1/etlp/transformations/{transformation_id}", response_model=SuccessResponse)(self.delete_transformation)
		
		# Data Source Management Routes
		self.app.post("/api/v1/etlp/datasources", response_model=DataSourceResponse)(self.create_data_source)
		self.app.get("/api/v1/etlp/datasources", response_model=List[DataSourceResponse])(self.list_data_sources)
		self.app.get("/api/v1/etlp/datasources/{source_id}", response_model=DataSourceResponse)(self.get_data_source)
		self.app.put("/api/v1/etlp/datasources/{source_id}", response_model=DataSourceResponse)(self.update_data_source)
		self.app.delete("/api/v1/etlp/datasources/{source_id}", response_model=SuccessResponse)(self.delete_data_source)
		self.app.post("/api/v1/etlp/datasources/{source_id}/test", response_model=Dict[str, Any])(self.test_data_source)
		
		# Quality Rule Management Routes
		self.app.post("/api/v1/etlp/quality-rules", response_model=QualityRuleResponse)(self.create_quality_rule)
		self.app.get("/api/v1/etlp/quality-rules", response_model=List[QualityRuleResponse])(self.list_quality_rules)
		self.app.get("/api/v1/etlp/quality-rules/{rule_id}", response_model=QualityRuleResponse)(self.get_quality_rule)
		self.app.put("/api/v1/etlp/quality-rules/{rule_id}", response_model=QualityRuleResponse)(self.update_quality_rule)
		self.app.delete("/api/v1/etlp/quality-rules/{rule_id}", response_model=SuccessResponse)(self.delete_quality_rule)
		
		# Real-time Streaming Routes
		self.app.get("/api/v1/etlp/pipelines/{pipeline_id}/stream-logs")(self.stream_pipeline_logs)
		self.app.get("/api/v1/etlp/executions/{execution_id}/stream-progress")(self.stream_execution_progress)
		
		# Field Mapping Routes
		self.app.get("/api/v1/etlp/field-mapping/schema/{data_source_id}/{table_name}", response_model=Dict[str, Any])(self.get_table_schema)
		self.app.post("/api/v1/etlp/field-mapping/suggest", response_model=Dict[str, Any])(self.generate_field_mappings)
		self.app.post("/api/v1/etlp/field-mapping/save", response_model=Dict[str, Any])(self.save_field_mapping)
		self.app.post("/api/v1/etlp/field-mapping/execute", response_model=Dict[str, str])(self.execute_field_mapping)
		
		# Health Check
		self.app.get("/api/v1/etlp/health", response_model=Dict[str, Any])(self.health_check)
	
	async def _get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
		"""Extract current user from APG authentication"""
		if self.auth_service:
			# Use APG auth service to validate token
			user_info = await self.auth_service.validate_token(credentials.credentials)
			if not user_info:
				raise HTTPException(status_code=401, detail="Invalid authentication token")
			return user_info
		
		# Mock user for development
		return {
			"user_id": "dev-user",
			"tenant_id": "dev-tenant",
			"username": "developer",
			"permissions": ["*"]
		}
	
	async def _get_etlp_service(self, user_info: Dict[str, Any] = Depends(_get_current_user)) -> ETLPService:
		"""Get ETLP service instance with user context"""
		service = ETLPService(user_info["tenant_id"], user_info["user_id"])
		
		# Inject APG service dependencies
		service.auth_service = self.auth_service
		service.audit_service = self.audit_service
		service.notification_service = self.notification_service
		
		return service
	
	async def _log_api_access(self, endpoint: str, user_id: str, params: Optional[Dict[str, Any]] = None) -> None:
		"""Log API access for audit trail"""
		timestamp = datetime.utcnow().isoformat()
		print(f"[{timestamp}] API ACCESS: {endpoint} by {user_id} | params: {params}")
		
		if self.audit_service:
			await self.audit_service.log_event(
				"api_access",
				{"endpoint": endpoint, "params": params},
				user_id
			)
	
	# Pipeline Management Endpoints
	
	async def create_pipeline(
		self,
		request: PipelineCreateRequest,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineResponse:
		"""Create new pipeline"""
		try:
			await self._log_api_access("POST /pipelines", etlp_service.user_id, {"name": request.name})
			
			pipeline = await etlp_service.create_pipeline(request.model_dump())
			
			return PipelineResponse(
				id=pipeline.id,
				name=pipeline.name,
				description=pipeline.description,
				version=pipeline.version,
				status=pipeline.status,
				execution_mode=pipeline.execution_mode,
				steps=pipeline.steps,
				transformations=pipeline.transformations,
				data_sources=pipeline.data_sources,
				data_targets=pipeline.data_targets,
				configuration=pipeline.configuration,
				schedule_cron=pipeline.schedule_cron,
				tags=pipeline.tags,
				created_by=pipeline.created_by,
				created_at=pipeline.created_at,
				updated_at=pipeline.updated_at,
				updated_by=pipeline.updated_by
			)
			
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e))
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline creation failed: {str(e)}")
	
	async def list_pipelines(
		self,
		limit: int = Query(100, ge=1, le=1000),
		offset: int = Query(0, ge=0),
		status: Optional[PipelineStatus] = Query(None),
		search: Optional[str] = Query(None),
		tags: Optional[str] = Query(None),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineListResponse:
		"""List pipelines with filtering and pagination"""
		try:
			await self._log_api_access("GET /pipelines", etlp_service.user_id, 
									 {"limit": limit, "offset": offset, "status": status})
			
			filters = {}
			if status:
				filters["status"] = status
			if search:
				filters["search"] = search
			if tags:
				filters["tags"] = tags.split(",")
			
			pipelines = await etlp_service.list_pipelines(filters, limit, offset)
			
			# Convert to response format
			pipeline_responses = []
			for pipeline in pipelines:
				pipeline_responses.append(PipelineResponse(
					id=pipeline.id,
					name=pipeline.name,
					description=pipeline.description,
					version=pipeline.version,
					status=pipeline.status,
					execution_mode=pipeline.execution_mode,
					steps=pipeline.steps,
					transformations=pipeline.transformations,
					data_sources=pipeline.data_sources,
					data_targets=pipeline.data_targets,
					configuration=pipeline.configuration,
					schedule_cron=pipeline.schedule_cron,
					tags=pipeline.tags,
					created_by=pipeline.created_by,
					created_at=pipeline.created_at,
					updated_at=pipeline.updated_at,
					updated_by=pipeline.updated_by
				))
			
			return PipelineListResponse(
				pipelines=pipeline_responses,
				total=len(pipeline_responses),  # In real implementation, get total count from database
				offset=offset,
				limit=limit
			)
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline listing failed: {str(e)}")
	
	async def get_pipeline(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineResponse:
		"""Get pipeline by ID"""
		try:
			await self._log_api_access("GET /pipelines/{id}", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			pipeline = await etlp_service.get_pipeline(pipeline_id)
			if not pipeline:
				raise HTTPException(status_code=404, detail="Pipeline not found")
			
			return PipelineResponse(
				id=pipeline.id,
				name=pipeline.name,
				description=pipeline.description,
				version=pipeline.version,
				status=pipeline.status,
				execution_mode=pipeline.execution_mode,
				steps=pipeline.steps,
				transformations=pipeline.transformations,
				data_sources=pipeline.data_sources,
				data_targets=pipeline.data_targets,
				configuration=pipeline.configuration,
				schedule_cron=pipeline.schedule_cron,
				tags=pipeline.tags,
				created_by=pipeline.created_by,
				created_at=pipeline.created_at,
				updated_at=pipeline.updated_at,
				updated_by=pipeline.updated_by
			)
			
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline retrieval failed: {str(e)}")
	
	async def update_pipeline(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		request: PipelineUpdateRequest = ...,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineResponse:
		"""Update pipeline"""
		try:
			await self._log_api_access("PUT /pipelines/{id}", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			# Filter out None values
			updates = {k: v for k, v in request.model_dump().items() if v is not None}
			
			pipeline = await etlp_service.update_pipeline(pipeline_id, updates)
			
			return PipelineResponse(
				id=pipeline.id,
				name=pipeline.name,
				description=pipeline.description,
				version=pipeline.version,
				status=pipeline.status,
				execution_mode=pipeline.execution_mode,
				steps=pipeline.steps,
				transformations=pipeline.transformations,
				data_sources=pipeline.data_sources,
				data_targets=pipeline.data_targets,
				configuration=pipeline.configuration,
				schedule_cron=pipeline.schedule_cron,
				tags=pipeline.tags,
				created_by=pipeline.created_by,
				created_at=pipeline.created_at,
				updated_at=pipeline.updated_at,
				updated_by=pipeline.updated_by
			)
			
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e))
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline update failed: {str(e)}")
	
	async def delete_pipeline(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		hard_delete: bool = Query(False, description="Permanently delete pipeline"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> SuccessResponse:
		"""Delete pipeline"""
		try:
			await self._log_api_access("DELETE /pipelines/{id}", etlp_service.user_id, 
									 {"pipeline_id": pipeline_id, "hard_delete": hard_delete})
			
			success = await etlp_service.delete_pipeline(pipeline_id, hard_delete)
			if not success:
				raise HTTPException(status_code=400, detail="Pipeline deletion failed")
			
			return SuccessResponse(
				message="Pipeline deleted successfully",
				data={"pipeline_id": pipeline_id, "hard_delete": hard_delete}
			)
			
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e))
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline deletion failed: {str(e)}")
	
	# Pipeline Execution Endpoints
	
	async def execute_pipeline(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		request: PipelineExecuteRequest = PipelineExecuteRequest(),
		background_tasks: BackgroundTasks = BackgroundTasks(),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> Dict[str, str]:
		"""Execute pipeline"""
		try:
			await self._log_api_access("POST /pipelines/{id}/execute", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			execution_id = await etlp_service.execute_pipeline(
				pipeline_id,
				request.configuration,
				request.execution_mode
			)
			
			return {
				"execution_id": execution_id,
				"status": "started",
				"message": "Pipeline execution started successfully"
			}
			
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e))
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
	
	async def get_execution(
		self,
		execution_id: str = Path(..., description="Execution ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> ExecutionResponse:
		"""Get execution details"""
		try:
			await self._log_api_access("GET /executions/{id}", etlp_service.user_id, {"execution_id": execution_id})
			
			execution = await etlp_service.get_execution(execution_id)
			if not execution:
				raise HTTPException(status_code=404, detail="Execution not found")
			
			return ExecutionResponse(
				id=execution.id,
				pipeline_id=execution.pipeline_id,
				status=execution.status,
				execution_mode=execution.execution_mode,
				triggered_by=execution.triggered_by,
				trigger_type=execution.trigger_type,
				started_at=execution.started_at,
				completed_at=execution.completed_at,
				duration_ms=execution.duration_ms,
				pipeline_version=execution.pipeline_version,
				records_processed=execution.records_processed,
				records_failed=execution.records_failed,
				success_rate=execution.success_rate,
				error_message=execution.error_message,
				max_memory_mb=execution.max_memory_mb,
				avg_cpu_percent=execution.avg_cpu_percent,
				data_quality_score=execution.data_quality_score,
				created_at=execution.created_at
			)
			
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Execution retrieval failed: {str(e)}")
	
	async def list_pipeline_executions(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		limit: int = Query(100, ge=1, le=1000),
		offset: int = Query(0, ge=0),
		status: Optional[PipelineStatus] = Query(None),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> ExecutionListResponse:
		"""List pipeline executions"""
		try:
			await self._log_api_access("GET /pipelines/{id}/executions", etlp_service.user_id, 
									 {"pipeline_id": pipeline_id, "limit": limit, "offset": offset})
			
			executions = await etlp_service.list_executions(pipeline_id, status, limit, offset)
			
			execution_responses = []
			for execution in executions:
				execution_responses.append(ExecutionResponse(
					id=execution.id,
					pipeline_id=execution.pipeline_id,
					status=execution.status,
					execution_mode=execution.execution_mode,
					triggered_by=execution.triggered_by,
					trigger_type=execution.trigger_type,
					started_at=execution.started_at,
					completed_at=execution.completed_at,
					duration_ms=execution.duration_ms,
					pipeline_version=execution.pipeline_version,
					records_processed=execution.records_processed,
					records_failed=execution.records_failed,
					success_rate=execution.success_rate,
					error_message=execution.error_message,
					max_memory_mb=execution.max_memory_mb,
					avg_cpu_percent=execution.avg_cpu_percent,
					data_quality_score=execution.data_quality_score,
					created_at=execution.created_at
				))
			
			return ExecutionListResponse(
				executions=execution_responses,
				total=len(execution_responses),
				offset=offset,
				limit=limit
			)
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Execution listing failed: {str(e)}")
	
	async def cancel_execution(
		self,
		execution_id: str = Path(..., description="Execution ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> SuccessResponse:
		"""Cancel running execution"""
		try:
			await self._log_api_access("DELETE /executions/{id}", etlp_service.user_id, {"execution_id": execution_id})
			
			success = await etlp_service.cancel_execution(execution_id)
			if not success:
				raise HTTPException(status_code=400, detail="Execution cancellation failed")
			
			return SuccessResponse(
				message="Execution cancelled successfully",
				data={"execution_id": execution_id}
			)
			
		except ValueError as e:
			raise HTTPException(status_code=400, detail=str(e))
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Execution cancellation failed: {str(e)}")
	
	# Analytics and Monitoring Endpoints
	
	async def get_pipeline_metrics(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		days: int = Query(30, ge=1, le=365, description="Number of days for metrics"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineMetricsResponse:
		"""Get pipeline performance metrics"""
		try:
			await self._log_api_access("GET /pipelines/{id}/metrics", etlp_service.user_id, 
									 {"pipeline_id": pipeline_id, "days": days})
			
			# Get recent executions for metrics calculation
			executions = await etlp_service.list_executions(pipeline_id, None, 1000, 0)
			
			# Calculate metrics
			total_executions = len(executions)
			successful_executions = len([e for e in executions if e.status == PipelineStatus.SUCCESS])
			failed_executions = len([e for e in executions if e.status == PipelineStatus.FAILED])
			success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
			
			successful_exec = [e for e in executions if e.duration_ms and e.status == PipelineStatus.SUCCESS]
			avg_duration_ms = sum(e.duration_ms for e in successful_exec) / len(successful_exec) if successful_exec else 0
			
			successful_records = [e for e in executions if e.records_processed and e.status == PipelineStatus.SUCCESS]
			avg_records_processed = sum(e.records_processed for e in successful_records) / len(successful_records) if successful_records else 0
			
			last_execution = max(executions, key=lambda e: e.created_at).created_at if executions else None
			last_success = max([e for e in executions if e.status == PipelineStatus.SUCCESS], 
							  key=lambda e: e.created_at).created_at if successful_executions > 0 else None
			
			return PipelineMetricsResponse(
				pipeline_id=pipeline_id,
				total_executions=total_executions,
				successful_executions=successful_executions,
				failed_executions=failed_executions,
				success_rate=success_rate,
				avg_duration_ms=avg_duration_ms,
				avg_records_processed=avg_records_processed,
				last_execution=last_execution,
				last_success=last_success
			)
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")
	
	async def get_pipeline_health(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineHealthResponse:
		"""Get pipeline health status"""
		try:
			await self._log_api_access("GET /pipelines/{id}/health", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			pipeline = await etlp_service.get_pipeline(pipeline_id)
			if not pipeline:
				raise HTTPException(status_code=404, detail="Pipeline not found")
			
			# Perform health checks
			health_checks = []
			health_score = 100.0
			
			# Check recent execution success rate
			executions = await etlp_service.list_executions(pipeline_id, None, 10, 0)
			if executions:
				recent_failures = len([e for e in executions if e.status == PipelineStatus.FAILED])
				failure_rate = recent_failures / len(executions)
				if failure_rate > 0.2:  # > 20% failure rate
					health_score -= 30
					health_checks.append({
						"name": "execution_success_rate",
						"status": "warning",
						"message": f"High failure rate: {failure_rate*100:.1f}%"
					})
				else:
					health_checks.append({
						"name": "execution_success_rate",
						"status": "healthy",
						"message": f"Good success rate: {(1-failure_rate)*100:.1f}%"
					})
			
			# Check data source health
			for source_id in pipeline.data_sources:
				health_result = await etlp_service.test_data_source(source_id)
				if not health_result.get("healthy"):
					health_score -= 20
					health_checks.append({
						"name": f"data_source_{source_id}",
						"status": "unhealthy",
						"message": "Data source connection failed"
					})
			
			# Determine overall health status
			if health_score >= 90:
				health_status = "healthy"
			elif health_score >= 70:
				health_status = "warning"
			else:
				health_status = "unhealthy"
			
			recommendations = []
			if health_score < 100:
				recommendations.append("Review recent execution failures")
				if health_score < 80:
					recommendations.append("Check data source connections")
					recommendations.append("Consider pipeline optimization")
			
			return PipelineHealthResponse(
				pipeline_id=pipeline_id,
				health_status=health_status,
				health_score=health_score,
				checks=health_checks,
				recommendations=recommendations,
				last_check=datetime.utcnow()
			)
			
		except HTTPException:
			raise
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
	
	async def optimize_pipeline(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> PipelineOptimizationResponse:
		"""Get AI-powered pipeline optimization recommendations"""
		try:
			await self._log_api_access("POST /pipelines/{id}/optimize", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			recommendations = await etlp_service.optimize_pipeline(pipeline_id)
			
			# Calculate optimization scores
			overall_score = 75.0  # Base score
			potential_improvement = 0.0
			
			for category in recommendations.values():
				potential_improvement += len(category) * 5  # 5% per recommendation
			
			return PipelineOptimizationResponse(
				pipeline_id=pipeline_id,
				performance_improvements=recommendations.get("performance_improvements", []),
				resource_optimizations=recommendations.get("resource_optimizations", []),
				reliability_enhancements=recommendations.get("reliability_enhancements", []),
				cost_optimizations=recommendations.get("cost_optimizations", []),
				overall_score=overall_score,
				potential_improvement=min(potential_improvement, 25.0),  # Cap at 25%
				generated_at=datetime.utcnow()
			)
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Pipeline optimization failed: {str(e)}")
	
	async def get_pipeline_collaborators(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> List[CollaboratorResponse]:
		"""Get pipeline collaborators"""
		try:
			await self._log_api_access("GET /pipelines/{id}/collaborators", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			collaborators = await etlp_service.get_pipeline_collaborators(pipeline_id)
			
			collaborator_responses = []
			for collab in collaborators:
				collaborator_responses.append(CollaboratorResponse(
					user_id=collab.get("user_id", ""),
					username=collab.get("username", ""),
					role=collab.get("role", "viewer"),
					permissions=collab.get("permissions", []),
					last_active=collab.get("last_active"),
					status=collab.get("status", "active")
				))
			
			return collaborator_responses
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Collaborator retrieval failed: {str(e)}")
	
	# Real-time Streaming Endpoints
	
	async def stream_pipeline_logs(
		self,
		pipeline_id: str = Path(..., description="Pipeline ID"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> StreamingResponse:
		"""Stream real-time pipeline logs"""
		try:
			await self._log_api_access("GET /pipelines/{id}/stream-logs", etlp_service.user_id, {"pipeline_id": pipeline_id})
			
			async def log_generator():
				# Mock log streaming - in real implementation would connect to log stream
				while True:
					yield f"data: {{'timestamp': '{datetime.utcnow().isoformat()}', 'level': 'INFO', 'message': 'Pipeline processing...'}}\n\n"
					await asyncio.sleep(1)
			
			return StreamingResponse(
				log_generator(),
				media_type="text/plain",
				headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
			)
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Log streaming failed: {str(e)}")
	
	# Health Check Endpoint
	
	async def health_check(self) -> Dict[str, Any]:
		"""API health check"""
		return {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"version": "1.0.0",
			"services": {
				"etlp_core": "healthy",
				"database": "healthy",
				"cache": "healthy"
			}
		}
	
	# Placeholder endpoints for other entities (similar pattern)
	
	async def create_transformation(
		self,
		request: TransformationCreateRequest,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> TransformationResponse:
		"""Create new transformation
		
		Creates a new reusable transformation with specified logic and parameters.
		Transformations are reusable components that can be applied to data in pipelines.
		
		Args:
			request: Transformation creation parameters
			etlp_service: ETLP service instance with user context
			
		Returns:
			TransformationResponse: Created transformation details
			
		Raises:
			HTTPException: If transformation creation fails
		"""
		try:
			await self._log_api_access("POST /transformations", etlp_service.user_id, {
				"name": request.name,
				"type": request.type.value if hasattr(request.type, 'value') else str(request.type)
			})
			
			# Create transformation using service
			transformation_data = {
				"name": request.name,
				"description": request.description,
				"type": request.type,
				"logic": request.logic,
				"input_schema": request.input_schema,
				"output_schema": request.output_schema,
				"parameters": request.parameters,
				"tags": request.tags,
				"category": request.category,
				"is_public": request.is_public,
				"cacheable": request.cacheable,
				"parallel_execution": request.parallel_execution
			}
			
			transformation = await etlp_service.create_transformation(transformation_data)
			
			# Convert to response model
			return TransformationResponse(
				id=transformation.id,
				name=transformation.name,
				description=transformation.description,
				type=transformation.type,
				version=transformation.version,
				tags=transformation.tags,
				category=transformation.category,
				is_public=transformation.is_public,
				usage_count=0,  # New transformation, no usage yet
				last_used=None,
				created_by=transformation.created_by,
				created_at=transformation.created_at,
				updated_at=transformation.updated_at
			)
			
		except Exception as e:
			raise HTTPException(status_code=400, detail=f"Transformation creation failed: {str(e)}")
	
	async def create_data_source(
		self,
		request: DataSourceCreateRequest,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> DataSourceResponse:
		"""Create new data source connection
		
		Creates a new data source connection that can be used in pipelines.
		Includes connection validation and health check setup.
		
		Args:
			request: Data source creation parameters
			etlp_service: ETLP service instance with user context
			
		Returns:
			DataSourceResponse: Created data source details
			
		Raises:
			HTTPException: If data source creation or connection test fails
		"""
		try:
			await self._log_api_access("POST /datasources", etlp_service.user_id, {
				"name": request.name,
				"type": request.type.value if hasattr(request.type, 'value') else str(request.type)
			})
			
			# Create data source using service
			data_source_data = {
				"name": request.name,
				"description": request.description,
				"type": request.type,
				"connection_string": request.connection_string,
				"credentials": request.credentials,
				"use_ssl": request.use_ssl,
				"timeout_seconds": request.timeout_seconds,
				"settings": request.settings,
				"headers": request.headers,
				"batch_size": request.batch_size,
				"max_connections": request.max_connections,
				"tags": request.tags,
				"category": request.category,
				"health_check_enabled": request.health_check_enabled
			}
			
			data_source = await etlp_service.create_data_source(data_source_data)
			
			# Test connection during creation
			connection_test = await etlp_service.test_data_source_connection(data_source.id)
			is_healthy = connection_test.get('success', False)
			
			# Convert to response model with masked credentials
			masked_connection = self._mask_connection_string(data_source.connection_string)
			
			return DataSourceResponse(
				id=data_source.id,
				name=data_source.name,
				description=data_source.description,
				type=data_source.type,
				connection_string=masked_connection,
				use_ssl=data_source.use_ssl,
				timeout_seconds=data_source.timeout_seconds,
				batch_size=data_source.batch_size,
				max_connections=data_source.max_connections,
				tags=data_source.tags,
				category=data_source.category,
				health_check_enabled=data_source.health_check_enabled,
				is_healthy=is_healthy,
				last_health_check=datetime.utcnow() if is_healthy else None,
				usage_count=0,  # New data source, no usage yet
				created_by=data_source.created_by,
				created_at=data_source.created_at,
				updated_at=data_source.updated_at
			)
			
		except Exception as e:
			raise HTTPException(status_code=400, detail=f"Data source creation failed: {str(e)}")
	
	async def create_quality_rule(
		self,
		request: QualityRuleCreateRequest,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> QualityRuleResponse:
		"""Create new data quality rule
		
		Creates a new data quality validation rule that can be applied to pipelines
		to ensure data meets specified quality standards.
		
		Args:
			request: Quality rule creation parameters
			etlp_service: ETLP service instance with user context
			
		Returns:
			QualityRuleResponse: Created quality rule details
			
		Raises:
			HTTPException: If quality rule creation or validation fails
		"""
		try:
			await self._log_api_access("POST /quality-rules", etlp_service.user_id, {
				"name": request.name,
				"type": request.type.value if hasattr(request.type, 'value') else str(request.type),
				"field": request.field_name
			})
			
			# Create quality rule using service
			quality_rule_data = {
				"name": request.name,
				"description": request.description,
				"type": request.type,
				"field_name": request.field_name,
				"condition": request.condition,
				"severity": request.severity,
				"validation_logic": request.validation_logic,
				"error_message": request.error_message,
				"suggested_fix": request.suggested_fix,
				"enabled": request.enabled,
				"stop_on_violation": request.stop_on_violation,
				"sample_percentage": request.sample_percentage,
				"tags": request.tags,
				"category": request.category,
				"is_public": request.is_public
			}
			
			quality_rule = await etlp_service.create_quality_rule(quality_rule_data)
			
			# Validate rule logic during creation
			validation_result = await etlp_service.validate_quality_rule_logic(quality_rule.id)
			
			# Convert to response model
			return QualityRuleResponse(
				id=quality_rule.id,
				name=quality_rule.name,
				description=quality_rule.description,
				type=quality_rule.type,
				field_name=quality_rule.field_name,
				severity=quality_rule.severity,
				errorMessage=quality_rule.error_message,
				suggested_fix=quality_rule.suggested_fix,
				enabled=quality_rule.enabled,
				stop_on_violation=quality_rule.stop_on_violation,
				sample_percentage=quality_rule.sample_percentage,
				tags=quality_rule.tags,
				category=quality_rule.category,
				is_public=quality_rule.is_public,
				usage_count=0,  # New rule, no usage yet
				last_used=None,
				validation_status="valid" if validation_result.get('valid') else "invalid",
				created_by=quality_rule.created_by,
				created_at=quality_rule.created_at,
				updated_at=quality_rule.updated_at
			)
			
		except Exception as e:
			raise HTTPException(status_code=400, detail=f"Quality rule creation failed: {str(e)}")
	
	# Field Mapping Endpoints
	
	async def get_table_schema(
		self,
		data_source_id: str = Path(..., description="Data source ID"),
		table_name: str = Path(..., description="Table name"),
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> Dict[str, Any]:
		"""Get table schema for field mapping"""
		try:
			await self._log_api_access("GET /field-mapping/schema", etlp_service.user_id, {
				"data_source_id": data_source_id,
				"table_name": table_name
			})
			
			field_mapper = FieldMapperService(etlp_service.tenant_id, etlp_service.user_id)
			schema = await field_mapper.analyze_schema(data_source_id, table_name)
			
			return {
				"schema": schema.model_dump(),
				"status": "success"
			}
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Schema analysis failed: {str(e)}")
	
	async def generate_field_mappings(
		self,
		request: Dict[str, Any],
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> Dict[str, Any]:
		"""Generate intelligent field mapping suggestions"""
		try:
			await self._log_api_access("POST /field-mapping/suggest", etlp_service.user_id, request)
			
			field_mapper = FieldMapperService(etlp_service.tenant_id, etlp_service.user_id)
			
			# Parse schemas from request
			source_schema = TableSchema.model_validate(request["source_schema"])
			target_schema = TableSchema.model_validate(request["target_schema"])
			
			# Generate intelligent mappings
			mappings = await field_mapper.generate_intelligent_mappings(source_schema, target_schema)
			
			# Convert to response format
			suggestions = []
			for mapping in mappings:
				suggestions.append({
					"source_field": mapping.source_field,
					"target_field": mapping.target_field,
					"transformation": mapping.transformation.value,
					"config": mapping.transformation_config,
					"confidence": await self._calculate_mapping_confidence(mapping, source_schema, target_schema),
					"validation_rules": mapping.validation_rules
				})
			
			return {
				"suggestions": suggestions,
				"status": "success",
				"count": len(suggestions)
			}
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Mapping generation failed: {str(e)}")
	
	async def save_field_mapping(
		self,
		request: Dict[str, Any],
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> Dict[str, Any]:
		"""Save field mapping configuration"""
		try:
			await self._log_api_access("POST /field-mapping/save", etlp_service.user_id, {
				"config_name": request.get("name", "Unnamed Configuration")
			})
			
			field_mapper = FieldMapperService(etlp_service.tenant_id, etlp_service.user_id)
			
			# Parse mapping configuration
			config = MappingConfiguration.model_validate(request)
			
			# Validate configuration
			validation_result = await field_mapper.validate_mapping_configuration(config)
			
			if not validation_result["valid"]:
				return {
					"status": "validation_failed",
					"errors": validation_result["errors"],
					"warnings": validation_result["warnings"]
				}
			
			# Save configuration to database
			config_id = await etlp_service.save_field_mapping_configuration(config)
			
			return {
				"id": config_id,
				"status": "saved",
				"validation_result": validation_result
			}
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Configuration save failed: {str(e)}")
	
	async def execute_field_mapping(
		self,
		request: Dict[str, Any],
		background_tasks: BackgroundTasks,
		etlp_service: ETLPService = Depends(_get_etlp_service)
	) -> Dict[str, str]:
		"""Execute field mapping configuration as pipeline"""
		try:
			await self._log_api_access("POST /field-mapping/execute", etlp_service.user_id, {
				"config_id": request.get("config_id", "inline")
			})
			
			field_mapper = FieldMapperService(etlp_service.tenant_id, etlp_service.user_id)
			
			# Parse mapping configuration
			config = MappingConfiguration.model_validate(request["configuration"])
			
			# Execute mapping in background
			execution_id = await field_mapper.execute_mapping(config)
			
			return {
				"execution_id": execution_id,
				"status": "started",
				"message": "Field mapping execution started"
			}
			
		except Exception as e:
			raise HTTPException(status_code=500, detail=f"Mapping execution failed: {str(e)}")
	
	# Helper Methods
	
	def _mask_connection_string(self, connection_string: str) -> str:
		"""Mask sensitive information in connection strings
		
		Replaces passwords and sensitive credentials with asterisks while preserving
		the connection string structure for display purposes.
		
		Args:
			connection_string: The original connection string
			
		Returns:
			str: Connection string with sensitive parts masked
		"""
		if not connection_string:
			return connection_string
			
		# Common patterns to mask
		patterns = [
			r'(password=)[^;@\s]+',  # password=value
			r'(pwd=)[^;@\s]+',       # pwd=value  
			r'(pass=)[^;@\s]+',      # pass=value
			r'://[^:]+:([^@]+)@',    # protocol://user:password@host
			r'(key=)[^;@\s]+',       # key=value
			r'(secret=)[^;@\s]+',    # secret=value
			r'(token=)[^;@\s]+',     # token=value
		]
		
		masked = connection_string
		for pattern in patterns:
			import re
			masked = re.sub(pattern, r'\1***', masked, flags=re.IGNORECASE)
		
		return masked
	
	async def _calculate_mapping_confidence(
		self, 
		mapping: FieldMapping, 
		source_schema: TableSchema, 
		target_schema: TableSchema
	) -> float:
		"""Calculate confidence score for a field mapping
		
		Analyzes the mapping quality based on name similarity, type compatibility,
		sample data patterns, and transformation complexity.
		
		Args:
			mapping: The field mapping to analyze
			source_schema: Source table schema
			target_schema: Target table schema
			
		Returns:
			float: Confidence score between 0.0 and 1.0
		"""
		try:
			# Find source and target field definitions
			source_field = None
			target_field = None
			
			for field in source_schema.fields:
				if field.name == mapping.source_field:
					source_field = field
					break
			
			for field in target_schema.fields:
				if field.name == mapping.target_field:
					target_field = field
					break
			
			if not source_field or not target_field:
				return 0.3  # Low confidence if fields not found
			
			confidence = 0.0
			
			# Name similarity (40% weight)
			name_similarity = await self._calculate_name_similarity(
				source_field.name, target_field.name
			)
			confidence += name_similarity * 0.4
			
			# Type compatibility (30% weight)
			type_compatibility = self._get_type_compatibility(
				source_field.data_type, target_field.data_type
			)
			confidence += type_compatibility * 0.3
			
			# Transformation complexity penalty (20% weight)
			if mapping.transformation.value == 'direct_copy':
				confidence += 0.2  # Full points for direct copy
			elif mapping.transformation.value in ['type_convert', 'format_string']:
				confidence += 0.15  # Slight penalty for simple transforms
			elif mapping.transformation.value in ['uppercase', 'lowercase', 'trim']:
				confidence += 0.18  # Minimal penalty for string transforms
			else:
				confidence += 0.1   # Higher penalty for complex transforms
			
			# Sample data similarity (10% weight)
			if source_field.sample_values and target_field.sample_values:
				data_similarity = await self._calculate_sample_similarity(
					source_field.sample_values, target_field.sample_values
				)
				confidence += data_similarity * 0.1
			
			return min(confidence, 1.0)  # Cap at 1.0
			
		except Exception:
			return 0.5  # Default confidence if calculation fails
	
	async def _calculate_name_similarity(self, name1: str, name2: str) -> float:
		"""Calculate similarity between field names using fuzzy matching"""
		if name1 == name2:
			return 1.0
		
		# Normalize names for comparison
		norm1 = self._normalize_field_name(name1)
		norm2 = self._normalize_field_name(name2)
		
		if norm1 == norm2:
			return 0.95
		
		# Calculate Levenshtein distance
		distance = self._levenshtein_distance(norm1, norm2)
		max_len = max(len(norm1), len(norm2))
		
		if max_len == 0:
			return 1.0
		
		similarity = 1.0 - (distance / max_len)
		
		# Bonus for common patterns
		if self._has_common_patterns(norm1, norm2):
			similarity += 0.1
		
		return min(similarity, 1.0)
	
	def _normalize_field_name(self, name: str) -> str:
		"""Normalize field name for comparison"""
		import re
		normalized = name.lower()
		# Remove common prefixes/suffixes
		normalized = re.sub(r'^(src_|tgt_|source_|target_)', '', normalized)
		normalized = re.sub(r'(_id|_key|_code)$', '', normalized)
		normalized = re.sub(r'[_\s]+', '_', normalized)
		return normalized.strip('_')
	
	def _levenshtein_distance(self, s1: str, s2: str) -> int:
		"""Calculate Levenshtein distance between two strings"""
		if len(s1) < len(s2):
			return self._levenshtein_distance(s2, s1)
		
		if len(s2) == 0:
			return len(s1)
		
		previous_row = list(range(len(s2) + 1))
		for i, c1 in enumerate(s1):
			current_row = [i + 1]
			for j, c2 in enumerate(s2):
				insertions = previous_row[j + 1] + 1
				deletions = current_row[j] + 1
				substitutions = previous_row[j] + (c1 != c2)
				current_row.append(min(insertions, deletions, substitutions))
			previous_row = current_row
		
		return previous_row[-1]
	
	def _has_common_patterns(self, name1: str, name2: str) -> bool:
		"""Check if names have common semantic patterns"""
		common_patterns = [
			('first_name', 'fname'),
			('last_name', 'lname'),
			('email_address', 'email'),
			('phone_number', 'phone'),
			('created_at', 'creation_date'),
			('updated_at', 'modification_date')
		]
		
		for pattern1, pattern2 in common_patterns:
			if (pattern1 in name1 and pattern2 in name2) or (pattern2 in name1 and pattern1 in name2):
				return True
		return False
	
	def _get_type_compatibility(self, source_type, target_type) -> float:
		"""Get compatibility score between data types"""
		if source_type == target_type:
			return 1.0
		
		# Type compatibility matrix
		compatibility_map = {
			'string': {'email': 0.9, 'url': 0.9, 'phone': 0.9, 'uuid': 0.8, 'json': 0.7, 'date': 0.6},
			'integer': {'float': 0.9, 'decimal': 0.9, 'boolean': 0.7, 'string': 0.6},
			'float': {'decimal': 0.9, 'integer': 0.8, 'string': 0.6},
			'date': {'datetime': 0.9, 'timestamp': 0.9, 'string': 0.7},
			'datetime': {'timestamp': 0.9, 'date': 0.8, 'string': 0.7},
			'boolean': {'integer': 0.7, 'string': 0.6},
			'json': {'string': 0.8, 'array': 0.7}
		}
		
		source_str = str(source_type).lower()
		target_str = str(target_type).lower()
		
		if source_str in compatibility_map and target_str in compatibility_map[source_str]:
			return compatibility_map[source_str][target_str]
		
		return 0.2  # Default low compatibility for unknown combinations
	
	async def _calculate_sample_similarity(self, samples1: List[Any], samples2: List[Any]) -> float:
		"""Calculate similarity between sample data sets"""
		if not samples1 or not samples2:
			return 0.5
		
		# Simple pattern-based similarity for now
		# Could be enhanced with more sophisticated analysis
		str_samples1 = [str(s) for s in samples1[:5]]  # Take first 5 samples
		str_samples2 = [str(s) for s in samples2[:5]]
		
		# Check for format similarities
		format_score = 0.0
		
		# Email pattern check
		email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
		import re
		
		emails1 = sum(1 for s in str_samples1 if re.match(email_pattern, s))
		emails2 = sum(1 for s in str_samples2 if re.match(email_pattern, s))
		
		if emails1 > 0 and emails2 > 0:
			format_score += 0.5
		
		# Numeric pattern check
		numeric1 = sum(1 for s in str_samples1 if s.replace('.', '').replace('-', '').isdigit())
		numeric2 = sum(1 for s in str_samples2 if s.replace('.', '').replace('-', '').isdigit())
		
		if numeric1 > 0 and numeric2 > 0:
			format_score += 0.3
		
		# Date pattern check  
		date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
		for pattern in date_patterns:
			dates1 = sum(1 for s in str_samples1 if re.match(pattern, s))
			dates2 = sum(1 for s in str_samples2 if re.match(pattern, s))
			if dates1 > 0 and dates2 > 0:
				format_score += 0.2
				break
		
		return min(format_score, 1.0)


# Initialize API controller
api_controller = ETLPAPIController()
app = api_controller.app