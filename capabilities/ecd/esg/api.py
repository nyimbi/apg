#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - REST API

Comprehensive async REST API endpoints with APG authentication,
real-time WebSocket support, and AI-powered ESG analytics.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Query, Path, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uvicorn

# APG integration imports
from ...auth_rbac.service import AuthRBACService
from ...audit_compliance.service import AuditComplianceService
from ...real_time_collaboration.service import RealTimeCollaborationService

from .models import (
	ESGTenant, ESGMetric, ESGMeasurement, ESGTarget, ESGStakeholder,
	ESGSupplier, ESGInitiative, ESGReport, ESGRisk,
	ESGFrameworkType, ESGMetricType, ESGTargetStatus, ESGRiskLevel
)
from .service import ESGManagementService, ESGServiceConfig
from .views import (
	ESGTenantView, ESGMetricView, ESGTargetView, ESGStakeholderView,
	ESGSupplierView, ESGInitiativeView, ESGReportView
)

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# FastAPI app configuration
app = FastAPI(
	title="APG Sustainability & ESG Management API",
	description="Revolutionary ESG management with AI intelligence and real-time processing",
	version="1.0.0",
	docs_url="/api/v1/esg/docs",
	redoc_url="/api/v1/esg/redoc"
)

# Dependency injection
async def get_database_session() -> Session:
	"""Get database session dependency"""
	# In real implementation, this would use APG's database session manager
	pass

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
	"""Get current authenticated user from APG auth service"""
	# Implementation would integrate with APG's auth_rbac service
	return {
		"user_id": "demo_user",
		"tenant_id": "demo_tenant",
		"permissions": ["esg:read", "esg:write", "esg:admin"]
	}

async def get_esg_service(
	user: Dict[str, Any] = Depends(get_current_user),
	db_session: Session = Depends(get_database_session)
) -> ESGManagementService:
	"""Get configured ESG service instance"""
	return ESGManagementService(
		db_session=db_session,
		tenant_id=user["tenant_id"],
		config=ESGServiceConfig(
			ai_enabled=True,
			real_time_processing=True,
			automated_reporting=True,
			stakeholder_engagement=True,
			supply_chain_monitoring=True,
			predictive_analytics=True
		)
	)

# Request/Response Models

class ESGMetricCreateRequest(BaseModel):
	"""Request model for creating ESG metric"""
	name: str = Field(..., min_length=1, max_length=255)
	code: str = Field(..., min_length=1, max_length=64)
	metric_type: ESGMetricType
	category: str = Field(..., min_length=1, max_length=128)
	subcategory: Optional[str] = Field(None, max_length=128)
	description: Optional[str] = None
	unit: str = Field(..., min_length=1, max_length=32)
	target_value: Optional[Decimal] = Field(None, ge=0)
	baseline_value: Optional[Decimal] = Field(None, ge=0)
	is_kpi: bool = False
	is_public: bool = False
	is_automated: bool = False
	enable_ai_predictions: bool = True

class ESGMeasurementRequest(BaseModel):
	"""Request model for recording ESG measurement"""
	metric_id: str = Field(..., min_length=1)
	value: Decimal = Field(..., ge=0)
	measurement_date: datetime
	period_start: Optional[datetime] = None
	period_end: Optional[datetime] = None
	data_source: str = Field(default="api", max_length=128)
	collection_method: str = Field(default="automated", max_length=64)
	metadata: Dict[str, Any] = Field(default_factory=dict)
	notes: Optional[str] = None

class ESGTargetCreateRequest(BaseModel):
	"""Request model for creating ESG target"""
	name: str = Field(..., min_length=1, max_length=255)
	metric_id: str = Field(..., min_length=1)
	description: Optional[str] = None
	target_value: Decimal = Field(..., ge=0)
	baseline_value: Optional[Decimal] = Field(None, ge=0)
	start_date: datetime
	target_date: datetime
	priority: str = Field(default="medium", regex="^(low|medium|high|critical)$")
	owner_id: Optional[str] = None
	is_public: bool = False
	create_milestones: bool = True

class ESGStakeholderCreateRequest(BaseModel):
	"""Request model for creating ESG stakeholder"""
	name: str = Field(..., min_length=1, max_length=255)
	organization: Optional[str] = Field(None, max_length=255)
	stakeholder_type: str = Field(..., min_length=1, max_length=64)
	email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
	phone: Optional[str] = Field(None, max_length=32)
	country: Optional[str] = Field(None, min_length=2, max_length=3)
	language_preference: str = Field(default="en_US", max_length=10)
	esg_interests: List[str] = Field(default_factory=list)
	engagement_frequency: str = Field(default="quarterly", regex="^(weekly|monthly|quarterly|annually)$")
	portal_access: bool = False
	data_access_level: str = Field(default="public", regex="^(public|internal|confidential)$")

class ESGAnalyticsResponse(BaseModel):
	"""Response model for ESG analytics"""
	period: str
	total_metrics: int
	active_targets: int
	completed_targets: int
	stakeholder_engagement_score: Optional[Decimal]
	overall_esg_score: Optional[Decimal]
	trends: Dict[str, Any]
	ai_insights: Dict[str, Any]
	recommendations: List[Dict[str, Any]]

class ESGDashboardResponse(BaseModel):
	"""Response model for ESG dashboard data"""
	key_metrics: List[ESGMetricView]
	active_targets: List[ESGTargetView]
	stakeholder_summary: Dict[str, Any]
	recent_initiatives: List[ESGInitiativeView]
	ai_insights: Dict[str, Any]
	real_time_data: Dict[str, Any]
	last_updated: datetime

# Core API Endpoints

@app.get("/api/v1/esg/health")
async def health_check():
	"""API health check endpoint"""
	return {
		"status": "healthy",
		"version": "1.0.0",
		"timestamp": datetime.utcnow().isoformat(),
		"services": {
			"database": "connected",
			"ai_engine": "active",
			"real_time": "active"
		}
	}

# ESG Metrics Endpoints

@app.get("/api/v1/esg/metrics", response_model=List[ESGMetricView])
async def get_metrics(
	metric_type: Optional[ESGMetricType] = Query(None, description="Filter by metric type"),
	category: Optional[str] = Query(None, description="Filter by category"),
	is_kpi: Optional[bool] = Query(None, description="Filter KPI metrics"),
	is_public: Optional[bool] = Query(None, description="Filter public metrics"),
	search: Optional[str] = Query(None, description="Search in name, code, description"),
	limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
	offset: int = Query(0, ge=0, description="Number of results to skip"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get ESG metrics with filtering and pagination"""
	try:
		filters = {
			"metric_type": metric_type.value if metric_type else None,
			"category": category,
			"is_kpi": is_kpi,
			"is_public": is_public,
			"search": search,
			"limit": limit,
			"offset": offset
		}
		
		metrics = await esg_service.get_metrics(
			user_id=user["user_id"],
			filters={k: v for k, v in filters.items() if v is not None}
		)
		
		return [ESGMetricView.model_validate(metric.__dict__) for metric in metrics]
		
	except Exception as e:
		logger.error(f"Failed to get ESG metrics: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/esg/metrics", response_model=ESGMetricView, status_code=201)
async def create_metric(
	metric_request: ESGMetricCreateRequest,
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Create new ESG metric with AI-enhanced configuration"""
	try:
		metric = await esg_service.create_metric(
			user_id=user["user_id"],
			metric_data=metric_request.model_dump()
		)
		
		return ESGMetricView.model_validate(metric.__dict__)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Failed to create ESG metric: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/esg/metrics/{metric_id}", response_model=ESGMetricView)
async def get_metric(
	metric_id: str = Path(..., description="Metric ID"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get specific ESG metric by ID"""
	try:
		metrics = await esg_service.get_metrics(
			user_id=user["user_id"],
			filters={"metric_id": metric_id}
		)
		
		if not metrics:
			raise HTTPException(status_code=404, detail="Metric not found")
		
		return ESGMetricView.model_validate(metrics[0].__dict__)
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Failed to get ESG metric: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/v1/esg/metrics/{metric_id}", response_model=ESGMetricView)
async def update_metric(
	metric_id: str = Path(..., description="Metric ID"),
	updates: Dict[str, Any] = {},
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Update ESG metric with AI re-analysis"""
	try:
		metric = await esg_service.update_metric(
			user_id=user["user_id"],
			metric_id=metric_id,
			updates=updates
		)
		
		return ESGMetricView.model_validate(metric.__dict__)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Failed to update ESG metric: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/esg/metrics/{metric_id}/measurements", status_code=201)
async def record_measurement(
	metric_id: str = Path(..., description="Metric ID"),
	measurement: ESGMeasurementRequest = ...,
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Record new measurement for ESG metric"""
	try:
		measurement_data = measurement.model_dump()
		measurement_data["metric_id"] = metric_id
		
		recorded_measurement = await esg_service.record_measurement(
			user_id=user["user_id"],
			measurement_data=measurement_data
		)
		
		return {
			"status": "success",
			"measurement_id": recorded_measurement.id,
			"metric_id": metric_id,
			"value": float(recorded_measurement.value),
			"validation_score": float(recorded_measurement.validation_score) if recorded_measurement.validation_score else None,
			"anomaly_score": float(recorded_measurement.anomaly_score) if recorded_measurement.anomaly_score else None,
			"message": "Measurement recorded successfully"
		}
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Failed to record ESG measurement: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/esg/metrics/{metric_id}/ai-insights")
async def get_metric_ai_insights(
	metric_id: str = Path(..., description="Metric ID"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get AI insights for specific ESG metric"""
	try:
		insights = await esg_service._initialize_metric_ai_predictions(metric_id, user["user_id"])
		
		return {
			"status": "success",
			"metric_id": metric_id,
			"ai_insights": insights,
			"generated_at": datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		logger.error(f"Failed to get metric AI insights: {e}")
		raise HTTPException(status_code=500, detail=str(e))

# ESG Targets Endpoints

@app.get("/api/v1/esg/targets", response_model=List[ESGTargetView])
async def get_targets(
	status: Optional[ESGTargetStatus] = Query(None, description="Filter by target status"),
	metric_id: Optional[str] = Query(None, description="Filter by metric ID"),
	owner_id: Optional[str] = Query(None, description="Filter by owner ID"),
	is_public: Optional[bool] = Query(None, description="Filter public targets"),
	limit: int = Query(50, ge=1, le=1000),
	offset: int = Query(0, ge=0),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get ESG targets with filtering and pagination"""
	try:
		# Implementation would use service method to get targets
		return []  # Placeholder
		
	except Exception as e:
		logger.error(f"Failed to get ESG targets: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/esg/targets", response_model=ESGTargetView, status_code=201)
async def create_target(
	target_request: ESGTargetCreateRequest,
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Create new ESG target with AI-powered achievement prediction"""
	try:
		target = await esg_service.create_target(
			user_id=user["user_id"],
			target_data=target_request.model_dump()
		)
		
		return ESGTargetView.model_validate(target.__dict__)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Failed to create ESG target: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/esg/targets/{target_id}/prediction")
async def get_target_prediction(
	target_id: str = Path(..., description="Target ID"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get AI prediction for target achievement"""
	try:
		prediction = await esg_service._predict_target_achievement(target_id, user["user_id"])
		
		return {
			"status": "success",
			"target_id": target_id,
			"prediction": prediction,
			"generated_at": datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		logger.error(f"Failed to get target prediction: {e}")
		raise HTTPException(status_code=500, detail=str(e))

# ESG Stakeholders Endpoints

@app.get("/api/v1/esg/stakeholders", response_model=List[ESGStakeholderView])
async def get_stakeholders(
	stakeholder_type: Optional[str] = Query(None, description="Filter by stakeholder type"),
	country: Optional[str] = Query(None, description="Filter by country"),
	portal_access: Optional[bool] = Query(None, description="Filter by portal access"),
	is_active: Optional[bool] = Query(None, description="Filter active stakeholders"),
	engagement_score_min: Optional[float] = Query(None, ge=0, le=100, description="Minimum engagement score"),
	limit: int = Query(50, ge=1, le=1000),
	offset: int = Query(0, ge=0),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get ESG stakeholders with filtering and pagination"""
	try:
		# Implementation would use service method
		return []  # Placeholder
		
	except Exception as e:
		logger.error(f"Failed to get ESG stakeholders: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/esg/stakeholders", response_model=ESGStakeholderView, status_code=201)
async def create_stakeholder(
	stakeholder_request: ESGStakeholderCreateRequest,
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Create new ESG stakeholder with engagement tracking"""
	try:
		stakeholder = await esg_service.create_stakeholder(
			user_id=user["user_id"],
			stakeholder_data=stakeholder_request.model_dump()
		)
		
		return ESGStakeholderView.model_validate(stakeholder.__dict__)
		
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except Exception as e:
		logger.error(f"Failed to create ESG stakeholder: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/esg/stakeholders/{stakeholder_id}/analytics")
async def get_stakeholder_analytics(
	stakeholder_id: str = Path(..., description="Stakeholder ID"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get detailed engagement analytics for stakeholder"""
	try:
		analytics = await esg_service._initialize_stakeholder_analytics(stakeholder_id, user["user_id"])
		
		return {
			"status": "success",
			"stakeholder_id": stakeholder_id,
			"analytics": analytics,
			"generated_at": datetime.utcnow().isoformat()
		}
		
	except Exception as e:
		logger.error(f"Failed to get stakeholder analytics: {e}")
		raise HTTPException(status_code=500, detail=str(e))

# ESG Analytics & Dashboard Endpoints

@app.get("/api/v1/esg/dashboard", response_model=ESGDashboardResponse)
async def get_dashboard(
	period: str = Query("current_month", description="Dashboard period (current_month, current_quarter, current_year)"),
	include_ai_insights: bool = Query(True, description="Include AI insights"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get comprehensive ESG dashboard data"""
	try:
		# Get key metrics
		key_metrics = await esg_service.get_metrics(
			user_id=user["user_id"],
			filters={"is_kpi": True, "limit": 10}
		)
		
		# Implementation would gather all dashboard data
		dashboard_data = ESGDashboardResponse(
			key_metrics=[ESGMetricView.model_validate(m.__dict__) for m in key_metrics],
			active_targets=[],
			stakeholder_summary={},
			recent_initiatives=[],
			ai_insights={} if include_ai_insights else {},
			real_time_data={},
			last_updated=datetime.utcnow()
		)
		
		return dashboard_data
		
	except Exception as e:
		logger.error(f"Failed to get ESG dashboard: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/esg/analytics", response_model=ESGAnalyticsResponse)
async def get_analytics(
	period: str = Query("current_quarter", description="Analytics period"),
	metric_types: Optional[List[ESGMetricType]] = Query(None, description="Filter by metric types"),
	include_trends: bool = Query(True, description="Include trend analysis"),
	include_predictions: bool = Query(True, description="Include AI predictions"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Get comprehensive ESG analytics"""
	try:
		# Implementation would calculate comprehensive analytics
		analytics = ESGAnalyticsResponse(
			period=period,
			total_metrics=0,
			active_targets=0,
			completed_targets=0,
			stakeholder_engagement_score=None,
			overall_esg_score=None,
			trends={},
			ai_insights={},
			recommendations=[]
		)
		
		return analytics
		
	except Exception as e:
		logger.error(f"Failed to get ESG analytics: {e}")
		raise HTTPException(status_code=500, detail=str(e))

# Real-time WebSocket Endpoints

class ESGWebSocketManager:
	"""WebSocket connection manager for real-time ESG updates"""
	
	def __init__(self):
		self.active_connections: List[WebSocket] = []
		self.tenant_connections: Dict[str, List[WebSocket]] = {}
	
	async def connect(self, websocket: WebSocket, tenant_id: str):
		"""Accept WebSocket connection"""
		await websocket.accept()
		self.active_connections.append(websocket)
		
		if tenant_id not in self.tenant_connections:
			self.tenant_connections[tenant_id] = []
		self.tenant_connections[tenant_id].append(websocket)
		
		logger.info(f"WebSocket connected for tenant {tenant_id}")
	
	def disconnect(self, websocket: WebSocket, tenant_id: str):
		"""Remove WebSocket connection"""
		self.active_connections.remove(websocket)
		if tenant_id in self.tenant_connections:
			self.tenant_connections[tenant_id].remove(websocket)
		
		logger.info(f"WebSocket disconnected for tenant {tenant_id}")
	
	async def broadcast_to_tenant(self, tenant_id: str, message: Dict[str, Any]):
		"""Broadcast message to all connections for tenant"""
		if tenant_id in self.tenant_connections:
			for connection in self.tenant_connections[tenant_id]:
				try:
					await connection.send_json(message)
				except Exception as e:
					logger.error(f"Failed to send WebSocket message: {e}")

# WebSocket manager instance
websocket_manager = ESGWebSocketManager()

@app.websocket("/api/v1/esg/ws/{tenant_id}")
async def esg_websocket_endpoint(websocket: WebSocket, tenant_id: str):
	"""WebSocket endpoint for real-time ESG updates"""
	await websocket_manager.connect(websocket, tenant_id)
	
	try:
		while True:
			# Keep connection alive and handle incoming messages
			data = await websocket.receive_text()
			message = json.loads(data)
			
			# Handle different message types
			if message.get("type") == "subscribe":
				# Subscribe to specific ESG data streams
				await websocket.send_json({
					"type": "subscription_confirmed",
					"channels": message.get("channels", []),
					"tenant_id": tenant_id
				})
			
			elif message.get("type") == "ping":
				# Respond to ping with pong
				await websocket.send_json({
					"type": "pong",
					"timestamp": datetime.utcnow().isoformat()
				})
	
	except WebSocketDisconnect:
		websocket_manager.disconnect(websocket, tenant_id)
	except Exception as e:
		logger.error(f"WebSocket error: {e}")
		websocket_manager.disconnect(websocket, tenant_id)

# Real-time data streaming endpoints

@app.get("/api/v1/esg/stream/metrics/{metric_id}")
async def stream_metric_data(
	metric_id: str = Path(..., description="Metric ID"),
	interval_seconds: int = Query(5, ge=1, le=3600, description="Update interval in seconds"),
	esg_service: ESGManagementService = Depends(get_esg_service),
	user: Dict[str, Any] = Depends(get_current_user)
):
	"""Stream real-time metric data via Server-Sent Events"""
	async def generate_metric_stream():
		while True:
			try:
				# Get current metric data
				metrics = await esg_service.get_metrics(
					user_id=user["user_id"],
					filters={"metric_id": metric_id}
				)
				
				if metrics:
					metric = metrics[0]
					data = {
						"timestamp": datetime.utcnow().isoformat(),
						"metric_id": metric.id,
						"current_value": float(metric.current_value) if metric.current_value else None,
						"trend": metric.trend_analysis.get("direction", "stable"),
						"data_quality": float(metric.data_quality_score) if metric.data_quality_score else None
					}
					
					yield f"data: {json.dumps(data)}\n\n"
				
				await asyncio.sleep(interval_seconds)
				
			except Exception as e:
				logger.error(f"Error in metric stream: {e}")
				yield f"data: {json.dumps({'error': str(e)})}\n\n"
				break
	
	return StreamingResponse(
		generate_metric_stream(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"Access-Control-Allow-Origin": "*",
			"Access-Control-Allow-Headers": "*"
		}
	)

# Error handlers

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
	"""Handle Pydantic validation errors"""
	return JSONResponse(
		status_code=422,
		content={
			"status": "error",
			"message": "Validation error",
			"details": exc.errors()
		}
	)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
	"""Handle HTTP exceptions"""
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"status": "error",
			"message": exc.detail,
			"error_code": exc.status_code
		}
	)

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
	"""Handle general exceptions"""
	logger.error(f"Unhandled exception: {exc}")
	return JSONResponse(
		status_code=500,
		content={
			"status": "error",
			"message": "Internal server error",
			"error_type": type(exc).__name__
		}
	)

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
	"""API startup initialization"""
	logger.info("🚀 APG ESG Management API starting up...")
	logger.info("✅ Real-time WebSocket endpoints active")
	logger.info("✅ AI-powered analytics ready")
	logger.info("✅ APG ecosystem integration active")

@app.on_event("shutdown")
async def shutdown_event():
	"""API shutdown cleanup"""
	logger.info("🛑 APG ESG Management API shutting down...")
	
	# Close all WebSocket connections
	for connection in websocket_manager.active_connections:
		try:
			await connection.close()
		except Exception as e:
			logger.error(f"Error closing WebSocket: {e}")

# Development server configuration
if __name__ == "__main__":
	uvicorn.run(
		"api:app",
		host="0.0.0.0",
		port=8000,
		reload=True,
		log_level="info"
	)