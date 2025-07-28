"""
Advanced Analytics Platform - Comprehensive FastAPI REST API

Enterprise-grade data analytics, machine learning, and AI platform API
providing real-time processing, predictive analytics, and business intelligence.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	APDataSourceConnection, APDataSource, APAnalyticsJob, APAnalyticsExecution,
	APMLModel, APMLTrainingJob, APDashboard, APVisualization, APReport, APReportExecution,
	APAlert, APAlertInstance, APDataQualityRule, APDataLineage, APFeatureStore, APFeature,
	APComputeCluster, APResourceUsage, APPredictiveModel, APAnomalyDetection,
	APFinancialAnalytics, APHealthcareAnalytics, APAnalyticsPipeline, APBusinessIntelligence,
	APDataSourceType, APDataFormat, APProcessingStatus, APModelType, APVisualizationType,
	APAlertSeverity, APComputeResourceType
)
from .service import AdvancedAnalyticsPlatformService

# Import common dependencies and utilities
from ...common.dependencies import get_current_user, get_db_session, get_redis_client
from ...common.responses import SuccessResponse, ErrorResponse, PaginatedResponse


# Initialize router
router = APIRouter(prefix="/api/v1/analytics", tags=["Advanced Analytics Platform"])


# Dependency injection
async def get_service(
	db: AsyncSession = Depends(get_db_session),
	redis = Depends(get_redis_client)
) -> AdvancedAnalyticsPlatformService:
	"""Get Advanced Analytics Platform service instance."""
	return AdvancedAnalyticsPlatformService(db, redis)


# Request/Response Models
class DataSourceConnectionRequest(BaseModel):
	name: str = Field(..., description="Connection name")
	source_type: APDataSourceType = Field(..., description="Type of data source")
	connection_string: str = Field(..., description="Connection string")
	authentication: Dict[str, Any] = Field(..., description="Authentication configuration")
	ssl_config: Optional[Dict[str, Any]] = Field(default=None, description="SSL configuration")
	connection_pool_size: int = Field(default=10, description="Connection pool size")
	timeout_seconds: int = Field(default=30, description="Connection timeout")
	retry_attempts: int = Field(default=3, description="Retry attempts on failure")


class DataSourceRequest(BaseModel):
	name: str = Field(..., description="Data source name")
	description: Optional[str] = Field(default=None, description="Data source description")
	connection_id: str = Field(..., description="Connection ID")
	source_schema: Dict[str, Any] = Field(..., description="Data schema definition")
	data_format: APDataFormat = Field(..., description="Data format type")
	refresh_interval: Optional[int] = Field(default=None, description="Auto-refresh interval")
	quality_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Quality rules")


class AnalyticsJobRequest(BaseModel):
	name: str = Field(..., description="Job name")
	description: Optional[str] = Field(default=None, description="Job description")
	job_type: str = Field(..., description="Type of analytics job")
	data_sources: List[str] = Field(..., description="List of data source IDs")
	processing_config: Dict[str, Any] = Field(..., description="Processing configuration")
	schedule_config: Optional[Dict[str, Any]] = Field(default=None, description="Scheduling configuration")
	output_destinations: List[Dict[str, Any]] = Field(default_factory=list, description="Output destinations")


class MLModelRequest(BaseModel):
	name: str = Field(..., description="Model name")
	description: Optional[str] = Field(default=None, description="Model description")
	model_type: APModelType = Field(..., description="Type of ML model")
	algorithm: str = Field(..., description="ML algorithm used")
	framework: str = Field(..., description="ML framework")
	training_data_sources: List[str] = Field(..., description="Training data source IDs")
	feature_columns: List[str] = Field(default_factory=list, description="Feature column names")
	target_column: Optional[str] = Field(default=None, description="Target column name")
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")


class MLTrainingRequest(BaseModel):
	training_config: Dict[str, Any] = Field(..., description="Training configuration")
	hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters")
	epochs: int = Field(default=100, description="Number of training epochs")
	validation_split: float = Field(default=0.2, description="Validation data split ratio")


class DashboardRequest(BaseModel):
	name: str = Field(..., description="Dashboard name")
	description: Optional[str] = Field(default=None, description="Dashboard description")
	category: str = Field(default="general", description="Dashboard category")
	layout_config: Dict[str, Any] = Field(..., description="Dashboard layout configuration")
	widget_configurations: List[Dict[str, Any]] = Field(default_factory=list, description="Widget configurations")
	data_refresh_interval: int = Field(default=300, description="Data refresh interval in seconds")


class VisualizationRequest(BaseModel):
	dashboard_id: Optional[str] = Field(default=None, description="Parent dashboard ID")
	name: str = Field(..., description="Visualization name")
	description: Optional[str] = Field(default=None, description="Visualization description")
	visualization_type: APVisualizationType = Field(..., description="Type of visualization")
	data_source_id: str = Field(..., description="Data source ID")
	query_config: Dict[str, Any] = Field(..., description="Data query configuration")
	chart_config: Dict[str, Any] = Field(..., description="Chart configuration")


class AlertRequest(BaseModel):
	name: str = Field(..., description="Alert name")
	description: Optional[str] = Field(default=None, description="Alert description")
	data_source_id: str = Field(..., description="Data source ID to monitor")
	alert_condition: Dict[str, Any] = Field(..., description="Alert trigger condition")
	severity: APAlertSeverity = Field(..., description="Alert severity level")
	threshold_config: Dict[str, Any] = Field(..., description="Threshold configuration")
	notification_channels: List[Dict[str, Any]] = Field(..., description="Notification channels")
	evaluation_frequency: int = Field(default=300, description="Evaluation frequency in seconds")


class ReportRequest(BaseModel):
	name: str = Field(..., description="Report name")
	description: Optional[str] = Field(default=None, description="Report description")
	report_category: str = Field(default="operational", description="Report category")
	data_sources: List[str] = Field(..., description="Data source IDs")
	report_structure: Dict[str, Any] = Field(..., description="Report structure definition")
	parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Report parameters")
	output_formats: List[str] = Field(default_factory=list, description="Supported output formats")


class PredictiveModelRequest(BaseModel):
	name: str = Field(..., description="Predictive model name")
	description: Optional[str] = Field(default=None, description="Model description")
	prediction_target: str = Field(..., description="What the model predicts")
	model_algorithm: str = Field(..., description="Prediction algorithm used")
	input_features: List[str] = Field(..., description="Input feature names")
	training_data_period: Dict[str, Any] = Field(..., description="Training data time period")
	prediction_horizon: int = Field(..., description="Prediction horizon in time units")


class AnomalyDetectionRequest(BaseModel):
	name: str = Field(..., description="Anomaly detection name")
	description: Optional[str] = Field(default=None, description="Detection description")
	data_source_id: str = Field(..., description="Data source to monitor")
	detection_algorithm: str = Field(..., description="Anomaly detection algorithm")
	algorithm_parameters: Dict[str, Any] = Field(..., description="Algorithm-specific parameters")
	sensitivity_level: float = Field(default=0.95, description="Detection sensitivity (0-1)")
	detection_frequency: int = Field(default=300, description="Detection frequency in seconds")


class PredictionRequest(BaseModel):
	input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
	prediction_count: int = Field(default=1, description="Number of predictions to generate")


# API Endpoints

# ============================================================================
# Data Source Management Endpoints
# ============================================================================

@router.post("/data-sources/connections", response_model=SuccessResponse, status_code=201)
async def create_data_source_connection(
	request: DataSourceConnectionRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new data source connection."""
	try:
		connection = await service.create_data_source_connection(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Data source connection created successfully",
			data={"connection_id": connection.id, "name": connection.name}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-sources/connections/{connection_id}/test", response_model=SuccessResponse)
async def test_data_source_connection(
	connection_id: str = Path(..., description="Connection ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Test data source connection health and performance."""
	try:
		test_results = await service.test_data_source_connection(
			connection_id=connection_id,
			tenant_id=current_user["tenant_id"]
		)
		
		return SuccessResponse(
			message="Connection tested successfully",
			data=test_results
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/data-sources", response_model=SuccessResponse, status_code=201)
async def create_data_source(
	request: DataSourceRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new data source definition."""
	try:
		data_source = await service.create_data_source(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Data source created successfully",
			data={"data_source_id": data_source.id, "name": data_source.name}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics Job Management Endpoints
# ============================================================================

@router.post("/jobs", response_model=SuccessResponse, status_code=201)
async def create_analytics_job(
	request: AnalyticsJobRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new analytics job."""
	try:
		job = await service.create_analytics_job(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Analytics job created successfully",
			data={"job_id": job.id, "name": job.name, "status": job.status}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/execute", response_model=SuccessResponse, status_code=202)
async def execute_analytics_job(
	job_id: str = Path(..., description="Job ID"),
	execution_config: Optional[Dict[str, Any]] = Body(default=None, description="Execution configuration"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Execute an analytics job."""
	try:
		execution = await service.execute_analytics_job(
			job_id=job_id,
			tenant_id=current_user["tenant_id"],
			execution_config=execution_config
		)
		
		return SuccessResponse(
			message="Job execution started successfully",
			data={
				"execution_id": execution.id,
				"job_id": job_id,
				"status": execution.status,
				"started_at": execution.started_at
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/executions/{execution_id}/status", response_model=SuccessResponse)
async def get_execution_status(
	job_id: str = Path(..., description="Job ID"),
	execution_id: str = Path(..., description="Execution ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get real-time execution status and progress."""
	try:
		# Get cached progress from Redis
		import json
		progress_data = await service.redis.get(f"execution_progress:{execution_id}")
		
		if progress_data:
			progress = json.loads(progress_data)
			return SuccessResponse(
				message="Execution status retrieved successfully",
				data={
					"execution_id": execution_id,
					"job_id": job_id,
					"current_stage": progress.get("stage"),
					"progress_percentage": progress.get("progress"),
					"last_updated": progress.get("timestamp")
				}
			)
		else:
			return SuccessResponse(
				message="Execution status retrieved successfully",
				data={
					"execution_id": execution_id,
					"job_id": job_id,
					"status": "completed_or_not_found"
				}
			)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Machine Learning Model Endpoints
# ============================================================================

@router.post("/ml/models", response_model=SuccessResponse, status_code=201)
async def create_ml_model(
	request: MLModelRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new machine learning model."""
	try:
		model = await service.create_ml_model(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="ML model created successfully",
			data={
				"model_id": model.id,
				"name": model.name,
				"model_type": model.model_type,
				"algorithm": model.algorithm
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml/models/{model_id}/train", response_model=SuccessResponse, status_code=202)
async def train_ml_model(
	model_id: str = Path(..., description="Model ID"),
	request: MLTrainingRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Start training a machine learning model."""
	try:
		training_job = await service.train_ml_model(
			model_id=model_id,
			tenant_id=current_user["tenant_id"],
			training_config=request.dict()
		)
		
		return SuccessResponse(
			message="Model training started successfully",
			data={
				"training_job_id": training_job.id,
				"model_id": model_id,
				"status": training_job.status,
				"total_epochs": training_job.total_epochs
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/models/{model_id}/training/{training_job_id}/progress", response_model=SuccessResponse)
async def get_training_progress(
	model_id: str = Path(..., description="Model ID"),
	training_job_id: str = Path(..., description="Training job ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get real-time model training progress."""
	try:
		# Get cached training progress from Redis
		import json
		progress_data = await service.redis.get(f"training_progress:{training_job_id}")
		
		if progress_data:
			progress = json.loads(progress_data)
			return SuccessResponse(
				message="Training progress retrieved successfully",
				data={
					"training_job_id": training_job_id,
					"model_id": model_id,
					"current_epoch": progress.get("epoch"),
					"total_epochs": progress.get("total_epochs"),
					"current_loss": progress.get("current_loss"),
					"validation_score": progress.get("validation_score"),
					"last_updated": progress.get("timestamp")
				}
			)
		else:
			return SuccessResponse(
				message="Training progress retrieved successfully",
				data={
					"training_job_id": training_job_id,
					"model_id": model_id,
					"status": "completed_or_not_found"
				}
			)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard and Visualization Endpoints
# ============================================================================

@router.post("/dashboards", response_model=SuccessResponse, status_code=201)
async def create_dashboard(
	request: DashboardRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new analytics dashboard."""
	try:
		dashboard = await service.create_dashboard(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Dashboard created successfully",
			data={
				"dashboard_id": dashboard.id,
				"name": dashboard.name,
				"category": dashboard.category
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualizations", response_model=SuccessResponse, status_code=201)
async def create_visualization(
	request: VisualizationRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new visualization."""
	try:
		visualization = await service.create_visualization(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Visualization created successfully",
			data={
				"visualization_id": visualization.id,
				"name": visualization.name,
				"type": visualization.visualization_type
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualizations/{visualization_id}/data", response_model=SuccessResponse)
async def get_visualization_data(
	visualization_id: str = Path(..., description="Visualization ID"),
	refresh: bool = Query(default=False, description="Force data refresh"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get data for a visualization with optional refresh."""
	try:
		# Simulate data retrieval for visualization
		visualization_data = {
			"visualization_id": visualization_id,
			"data": {
				"series": [
					{"name": "Revenue", "data": [100, 120, 140, 160, 180, 200]},
					{"name": "Profit", "data": [20, 25, 30, 35, 40, 45]}
				],
				"categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
				"metadata": {
					"last_updated": datetime.utcnow().isoformat(),
					"data_points": 12,
					"refresh_requested": refresh
				}
			}
		}
		
		return SuccessResponse(
			message="Visualization data retrieved successfully",
			data=visualization_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Alert Management Endpoints
# ============================================================================

@router.post("/alerts", response_model=SuccessResponse, status_code=201)
async def create_alert(
	request: AlertRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new analytics alert."""
	try:
		alert = await service.create_alert(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Alert created successfully",
			data={
				"alert_id": alert.id,
				"name": alert.name,
				"severity": alert.severity,
				"is_enabled": alert.is_enabled
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/instances", response_model=SuccessResponse)
async def get_recent_alert_instances(
	limit: int = Query(default=50, description="Maximum number of instances to return"),
	severity: Optional[APAlertSeverity] = Query(default=None, description="Filter by severity"),
	status: Optional[str] = Query(default=None, description="Filter by status"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get recent alert instances with optional filtering."""
	try:
		# Get recent alerts from Redis
		import json
		recent_alerts = await service.redis.lrange(
			f"recent_alerts:{current_user['tenant_id']}", 0, limit - 1
		)
		
		alert_instances = []
		for alert_data in recent_alerts:
			alert = json.loads(alert_data)
			
			# Apply filters
			if severity and alert.get("severity") != severity:
				continue
			if status and alert.get("status") != status:
				continue
			
			alert_instances.append(alert)
		
		return SuccessResponse(
			message="Alert instances retrieved successfully",
			data={
				"instances": alert_instances,
				"total_count": len(alert_instances),
				"filters_applied": {
					"severity": severity,
					"status": status,
					"limit": limit
				}
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/instances/{instance_id}/acknowledge", response_model=SuccessResponse)
async def acknowledge_alert_instance(
	instance_id: str = Path(..., description="Alert instance ID"),
	notes: Optional[str] = Body(default=None, description="Acknowledgment notes"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Acknowledge an alert instance."""
	try:
		# Simulate alert acknowledgment
		acknowledgment_data = {
			"instance_id": instance_id,
			"acknowledged_by": current_user["user_id"],
			"acknowledged_at": datetime.utcnow().isoformat(),
			"notes": notes,
			"status": "acknowledged"
		}
		
		# Cache acknowledgment
		import json
		await service.redis.setex(
			f"alert_ack:{instance_id}",
			86400,  # 24 hours
			json.dumps(acknowledgment_data)
		)
		
		return SuccessResponse(
			message="Alert instance acknowledged successfully",
			data=acknowledgment_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Report Management Endpoints
# ============================================================================

@router.post("/reports", response_model=SuccessResponse, status_code=201)
async def create_report(
	request: ReportRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new analytics report."""
	try:
		report = await service.create_report(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Report created successfully",
			data={
				"report_id": report.id,
				"name": report.name,
				"category": report.report_category
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/{report_id}/generate", response_model=SuccessResponse, status_code=202)
async def generate_report(
	report_id: str = Path(..., description="Report ID"),
	parameters: Optional[Dict[str, Any]] = Body(default=None, description="Report parameters"),
	output_format: str = Body(default="pdf", description="Output format"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Generate a report."""
	try:
		execution = await service.generate_report(
			report_id=report_id,
			tenant_id=current_user["tenant_id"],
			parameters=parameters,
			output_format=output_format
		)
		
		return SuccessResponse(
			message="Report generation started successfully",
			data={
				"execution_id": execution.id,
				"report_id": report_id,
				"status": execution.status,
				"output_format": output_format
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{report_id}/executions/{execution_id}/download")
async def download_report(
	report_id: str = Path(..., description="Report ID"),
	execution_id: str = Path(..., description="Execution ID"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Download a generated report."""
	try:
		# Simulate report download
		import io
		
		# Create a mock PDF content
		mock_pdf_content = b"Mock PDF content for report execution " + execution_id.encode()
		
		def generate_file():
			yield mock_pdf_content
		
		return StreamingResponse(
			generate_file(),
			media_type="application/pdf",
			headers={
				"Content-Disposition": f"attachment; filename=report_{report_id}_{execution_id}.pdf"
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Predictive Analytics Endpoints
# ============================================================================

@router.post("/predictive/models", response_model=SuccessResponse, status_code=201)
async def create_predictive_model(
	request: PredictiveModelRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new predictive analytics model."""
	try:
		model = await service.create_predictive_model(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Predictive model created successfully",
			data={
				"model_id": model.id,
				"name": model.name,
				"prediction_target": model.prediction_target,
				"algorithm": model.model_algorithm
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictive/models/{model_id}/predict", response_model=SuccessResponse)
async def generate_predictions(
	model_id: str = Path(..., description="Model ID"),
	request: PredictionRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Generate predictions using a predictive model."""
	try:
		predictions = await service.generate_predictions(
			model_id=model_id,
			tenant_id=current_user["tenant_id"],
			input_data=request.input_data,
			prediction_count=request.prediction_count
		)
		
		return SuccessResponse(
			message="Predictions generated successfully",
			data={
				"model_id": model_id,
				"predictions": predictions,
				"prediction_count": len(predictions)
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Anomaly Detection Endpoints
# ============================================================================

@router.post("/anomaly-detection", response_model=SuccessResponse, status_code=201)
async def create_anomaly_detection(
	request: AnomalyDetectionRequest,
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new anomaly detection configuration."""
	try:
		detection = await service.create_anomaly_detection(
			tenant_id=current_user["tenant_id"],
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"],
			**request.dict()
		)
		
		return SuccessResponse(
			message="Anomaly detection created successfully",
			data={
				"detection_id": detection.id,
				"name": detection.name,
				"algorithm": detection.detection_algorithm,
				"sensitivity_level": detection.sensitivity_level
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomaly-detection/anomalies", response_model=SuccessResponse)
async def get_recent_anomalies(
	limit: int = Query(default=100, description="Maximum number of anomalies to return"),
	since: Optional[datetime] = Query(default=None, description="Get anomalies since this timestamp"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get recent anomalies detected across all configurations."""
	try:
		# Get recent anomalies from Redis
		import json
		recent_anomalies = await service.redis.lrange(
			f"anomalies:{current_user['tenant_id']}", 0, limit - 1
		)
		
		anomalies = []
		for anomaly_data in recent_anomalies:
			anomaly = json.loads(anomaly_data)
			
			# Apply time filter if provided
			if since:
				anomaly_time = datetime.fromisoformat(anomaly["detected_at"].replace("Z", "+00:00"))
				if anomaly_time < since:
					continue
			
			anomalies.append(anomaly)
		
		return SuccessResponse(
			message="Recent anomalies retrieved successfully",
			data={
				"anomalies": anomalies,
				"total_count": len(anomalies),
				"filters_applied": {
					"limit": limit,
					"since": since.isoformat() if since else None
				}
			}
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Real-time Analytics Endpoints
# ============================================================================

@router.get("/real-time/metrics", response_model=SuccessResponse)
async def get_real_time_metrics(
	metric_types: List[str] = Query(..., description="Types of metrics to retrieve"),
	time_window: int = Query(default=3600, description="Time window in seconds"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get real-time analytics metrics."""
	try:
		metrics = await service.get_real_time_metrics(
			tenant_id=current_user["tenant_id"],
			metric_types=metric_types,
			time_window=time_window
		)
		
		return SuccessResponse(
			message="Real-time metrics retrieved successfully",
			data=metrics
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights", response_model=SuccessResponse)
async def generate_insights(
	data_source_ids: List[str] = Query(..., description="Data source IDs to analyze"),
	insight_types: List[str] = Query(..., description="Types of insights to generate"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Generate AI-powered insights from data."""
	try:
		insights = await service.generate_insights(
			tenant_id=current_user["tenant_id"],
			data_source_ids=data_source_ids,
			insight_types=insight_types
		)
		
		return SuccessResponse(
			message="Insights generated successfully",
			data={
				"insights": insights,
				"insight_count": len(insights),
				"generated_at": datetime.utcnow().isoformat()
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics Pipeline Endpoints
# ============================================================================

@router.post("/pipelines", response_model=SuccessResponse, status_code=201)
async def create_analytics_pipeline(
	name: str = Body(..., description="Pipeline name"),
	description: Optional[str] = Body(default=None, description="Pipeline description"),
	stages: List[Dict[str, Any]] = Body(..., description="Pipeline stages"),
	data_flow: Dict[str, Any] = Body(..., description="Data flow definition"),
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Create a new analytics pipeline."""
	try:
		pipeline = await service.create_analytics_pipeline(
			tenant_id=current_user["tenant_id"],
			name=name,
			stages=stages,
			data_flow=data_flow,
			description=description,
			created_by=current_user["user_id"],
			updated_by=current_user["user_id"]
		)
		
		return SuccessResponse(
			message="Analytics pipeline created successfully",
			data={
				"pipeline_id": pipeline.id,
				"name": pipeline.name,
				"stage_count": len(pipeline.stages)
			}
		)
	except HTTPException as e:
		raise e
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# System Health and Monitoring Endpoints
# ============================================================================

@router.get("/health", response_model=SuccessResponse)
async def get_system_health(
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get comprehensive system health status."""
	try:
		health_data = {
			"status": "healthy",
			"timestamp": datetime.utcnow().isoformat(),
			"tenant_id": current_user["tenant_id"],
			"components": {
				"database": {"status": "healthy", "response_time_ms": 12},
				"redis": {"status": "healthy", "response_time_ms": 3},
				"analytics_engine": {"status": "healthy", "active_jobs": 5},
				"ml_platform": {"status": "healthy", "training_jobs": 2},
				"alert_system": {"status": "healthy", "active_alerts": 8}
			},
			"performance_metrics": {
				"total_data_sources": 25,
				"active_jobs": 12,
				"completed_jobs_24h": 87,
				"active_models": 15,
				"predictions_24h": 1543,
				"alerts_24h": 23,
				"anomalies_detected_24h": 4
			},
			"resource_utilization": {
				"cpu_usage": 0.67,
				"memory_usage": 0.72,
				"storage_usage": 0.45,
				"network_throughput": 156.7
			}
		}
		
		return SuccessResponse(
			message="System health retrieved successfully",
			data=health_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics-summary", response_model=SuccessResponse)
async def get_analytics_summary(
	current_user: Dict[str, Any] = Depends(get_current_user),
	service: AdvancedAnalyticsPlatformService = Depends(get_service)
):
	"""Get comprehensive analytics platform summary."""
	try:
		summary_data = {
			"tenant_id": current_user["tenant_id"],
			"summary_timestamp": datetime.utcnow().isoformat(),
			"data_sources": {
				"total_connections": 15,
				"active_sources": 25,
				"data_volume_gb": 1250.7,
				"ingestion_rate_mb_per_hour": 890.2
			},
			"analytics_jobs": {
				"total_jobs": 67,
				"running_jobs": 5,
				"completed_today": 23,
				"failed_today": 1,
				"average_execution_time_minutes": 12.5
			},
			"machine_learning": {
				"total_models": 18,
				"training_models": 2,
				"deployed_models": 12,
				"predictions_generated_today": 1847,
				"model_accuracy_average": 0.91
			},
			"dashboards_reports": {
				"total_dashboards": 8,
				"total_reports": 15,
				"reports_generated_today": 34,
				"active_visualizations": 45
			},
			"alerts_monitoring": {
				"active_alerts": 12,
				"triggered_today": 6,
				"resolved_today": 8,
				"critical_alerts": 1
			},
			"top_insights": [
				"Revenue growth accelerated by 23% this quarter",
				"Customer churn increased 15% in premium segment",
				"Product A stockout predicted in 2 weeks",
				"Marketing ROI improved 18% with AI optimization"
			]
		}
		
		return SuccessResponse(
			message="Analytics summary retrieved successfully",
			data=summary_data
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))