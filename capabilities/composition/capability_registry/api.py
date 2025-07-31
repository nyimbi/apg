"""
APG Capability Registry - Comprehensive API Layer

RESTful API endpoints with OpenAPI documentation, WebSocket support,
webhook integration, and APG ecosystem integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid_extensions import uuid7str

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from .models import CRCapability, CRComposition, CRRegistry, CRVersion
from .views import (
	CapabilityListView, CapabilityDetailView, CapabilityCreateForm,
	CompositionListView, CompositionDetailView, CompositionCreateForm,
	CapabilitySearchForm, CompositionSearchForm, RegistryDashboardData,
	UIResponse
)
from .service import get_registry_service, CRService

# =============================================================================
# API Models
# =============================================================================

class APIResponse(BaseModel):
	"""Standard API response model."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool = Field(..., description="Success status")
	message: str = Field(..., description="Response message")
	data: Optional[Dict[str, Any]] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class PaginatedResponse(BaseModel):
	"""Paginated response model."""
	model_config = ConfigDict(extra='forbid')
	
	items: List[Any] = Field(..., description="Response items")
	total_count: int = Field(..., ge=0, description="Total item count")
	page: int = Field(..., ge=1, description="Current page")
	per_page: int = Field(..., ge=1, le=100, description="Items per page")
	total_pages: int = Field(..., ge=0, description="Total pages")
	has_next: bool = Field(..., description="Has next page")
	has_prev: bool = Field(..., description="Has previous page")

class WebhookEvent(BaseModel):
	"""Webhook event model."""
	model_config = ConfigDict(extra='forbid')
	
	event_id: str = Field(default_factory=uuid7str, description="Event ID")
	event_type: str = Field(..., description="Event type")
	resource_type: str = Field(..., description="Resource type")
	resource_id: str = Field(..., description="Resource ID")
	action: str = Field(..., description="Action performed")
	tenant_id: str = Field(..., description="Tenant ID")
	user_id: Optional[str] = Field(None, description="User ID")
	payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")

class WebSocketMessage(BaseModel):
	"""WebSocket message model."""
	model_config = ConfigDict(extra='forbid')
	
	type: str = Field(..., description="Message type")
	data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
	session_id: Optional[str] = Field(None, description="Session ID")

# =============================================================================
# FastAPI Application Setup
# =============================================================================

api_app = FastAPI(
	title="APG Capability Registry API",
	description="Comprehensive API for APG capability registry and composition management",
	version="1.0.0",
	docs_url="/api/docs",
	redoc_url="/api/redoc",
	openapi_url="/api/openapi.json",
	contact={
		"name": "Datacraft",
		"url": "https://www.datacraft.co.ke",
		"email": "nyimbi@gmail.com"
	},
	license_info={
		"name": "Proprietary",
		"url": "https://www.datacraft.co.ke/license"
	}
)

# CORS middleware
api_app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Security
security = HTTPBearer()

# =============================================================================
# Dependencies
# =============================================================================

async def get_current_tenant(
	authorization: HTTPAuthorizationCredentials = Depends(security)
) -> str:
	"""Extract tenant ID from authorization header."""
	# In production, would validate JWT token and extract tenant
	return "default"

async def get_registry_service_dep(
	tenant_id: str = Depends(get_current_tenant)
) -> CRService:
	"""Get registry service instance."""
	return await get_registry_service(tenant_id)

async def validate_pagination(
	page: int = Query(1, ge=1, description="Page number"),
	per_page: int = Query(25, ge=1, le=100, description="Items per page")
) -> Dict[str, int]:
	"""Validate pagination parameters."""
	return {"page": page, "per_page": per_page}

# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
	"""WebSocket connection manager."""
	
	def __init__(self):
		self.active_connections: Dict[str, WebSocket] = {}
		self.tenant_connections: Dict[str, List[str]] = {}
	
	async def connect(self, websocket: WebSocket, connection_id: str, tenant_id: str):
		"""Accept WebSocket connection."""
		await websocket.accept()
		self.active_connections[connection_id] = websocket
		
		if tenant_id not in self.tenant_connections:
			self.tenant_connections[tenant_id] = []
		self.tenant_connections[tenant_id].append(connection_id)
	
	def disconnect(self, connection_id: str, tenant_id: str):
		"""Remove WebSocket connection."""
		if connection_id in self.active_connections:
			del self.active_connections[connection_id]
		
		if tenant_id in self.tenant_connections:
			if connection_id in self.tenant_connections[tenant_id]:
				self.tenant_connections[tenant_id].remove(connection_id)
	
	async def send_personal_message(self, message: str, connection_id: str):
		"""Send message to specific connection."""
		if connection_id in self.active_connections:
			websocket = self.active_connections[connection_id]
			await websocket.send_text(message)
	
	async def broadcast_to_tenant(self, message: str, tenant_id: str):
		"""Broadcast message to all connections for a tenant."""
		if tenant_id in self.tenant_connections:
			for connection_id in self.tenant_connections[tenant_id]:
				await self.send_personal_message(message, connection_id)

manager = ConnectionManager()

# =============================================================================
# Capability API Endpoints
# =============================================================================

@api_app.get(
	"/api/capabilities",
	response_model=PaginatedResponse,
	summary="List capabilities",
	description="Get paginated list of capabilities with filtering and search"
)
async def list_capabilities(
	search: Optional[str] = Query(None, description="Search query"),
	category: Optional[str] = Query(None, description="Category filter"),
	status: Optional[str] = Query(None, description="Status filter"),
	min_quality_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum quality score"),
	pagination: Dict[str, int] = Depends(validate_pagination),
	service: CRService = Depends(get_registry_service_dep)
):
	"""List capabilities with pagination and filtering."""
	try:
		# Build search criteria
		search_form = CapabilitySearchForm(
			query=search,
			category=category,
			status=status,
			min_quality_score=min_quality_score,
			page=pagination["page"],
			per_page=pagination["per_page"]
		)
		
		# Get capabilities
		capabilities = await service.search_capabilities(search_form.model_dump())
		
		# Convert to list view
		capability_views = [
			CapabilityListView(
				capability_id=cap.capability_id,
				capability_code=cap.capability_code,
				capability_name=cap.capability_name,
				description=cap.description,
				version=cap.version,
				category=cap.category,
				status=cap.status,
				quality_score=cap.quality_score,
				popularity_score=cap.popularity_score,
				usage_count=cap.usage_count,
				created_at=cap.created_at
			)
			for cap in capabilities.get('capabilities', [])
		]
		
		total_count = capabilities.get('total_count', 0)
		total_pages = (total_count + pagination["per_page"] - 1) // pagination["per_page"]
		
		return PaginatedResponse(
			items=[cap.model_dump() for cap in capability_views],
			total_count=total_count,
			page=pagination["page"],
			per_page=pagination["per_page"],
			total_pages=total_pages,
			has_next=pagination["page"] < total_pages,
			has_prev=pagination["page"] > 1
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.get(
	"/api/capabilities/{capability_id}",
	response_model=APIResponse,
	summary="Get capability details",
	description="Get detailed information about a specific capability"
)
async def get_capability(
	capability_id: str = Path(..., description="Capability ID"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get capability by ID."""
	try:
		capability = await service.get_capability(capability_id)
		
		if not capability:
			raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="Capability not found")
		
		# Convert to detail view
		capability_view = CapabilityDetailView(
			capability_id=capability.capability_id,
			capability_code=capability.capability_code,
			capability_name=capability.capability_name,
			description=capability.description,
			long_description=capability.long_description,
			version=capability.version,
			category=capability.category,
			subcategory=capability.subcategory,
			status=capability.status,
			multi_tenant=capability.multi_tenant,
			audit_enabled=capability.audit_enabled,
			security_integration=capability.security_integration,
			performance_optimized=capability.performance_optimized,
			ai_enhanced=capability.ai_enhanced,
			target_users=capability.target_users or [],
			business_value=capability.business_value,
			use_cases=capability.use_cases or [],
			industry_focus=capability.industry_focus or [],
			composition_keywords=capability.composition_keywords or [],
			provides_services=capability.provides_services or [],
			data_models=capability.data_models or [],
			api_endpoints=capability.api_endpoints or [],
			file_path=capability.file_path,
			module_path=capability.module_path,
			documentation_path=capability.documentation_path,
			repository_url=capability.repository_url,
			complexity_score=capability.complexity_score,
			quality_score=capability.quality_score,
			popularity_score=capability.popularity_score,
			usage_count=capability.usage_count,
			created_at=capability.created_at,
			updated_at=capability.updated_at,
			metadata=capability.metadata or {}
		)
		
		return APIResponse(
			success=True,
			message="Capability retrieved successfully",
			data=capability_view.model_dump()
		)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.post(
	"/api/capabilities",
	response_model=APIResponse,
	status_code=HTTP_201_CREATED,
	summary="Create capability",
	description="Create a new capability in the registry"
)
async def create_capability(
	capability_form: CapabilityCreateForm,
	service: CRService = Depends(get_registry_service_dep)
):
	"""Create new capability."""
	try:
		# Register capability
		result = await service.register_capability(capability_form.model_dump())
		
		# Emit webhook event
		await emit_webhook_event(
			"capability.created",
			"capability",
			result["capability_id"],
			{"capability": capability_form.model_dump()}
		)
		
		# Broadcast WebSocket update
		await broadcast_registry_update("capability_created", result)
		
		return APIResponse(
			success=True,
			message="Capability created successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.put(
	"/api/capabilities/{capability_id}",
	response_model=APIResponse,
	summary="Update capability",
	description="Update an existing capability"
)
async def update_capability(
	capability_id: str = Path(..., description="Capability ID"),
	updates: Dict[str, Any] = Body(..., description="Update data"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Update capability."""
	try:
		result = await service.update_capability(capability_id, updates)
		
		# Emit webhook event
		await emit_webhook_event(
			"capability.updated",
			"capability",
			capability_id,
			{"updates": updates, "result": result}
		)
		
		# Broadcast WebSocket update
		await broadcast_registry_update("capability_updated", {"capability_id": capability_id, **result})
		
		return APIResponse(
			success=True,
			message="Capability updated successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.delete(
	"/api/capabilities/{capability_id}",
	response_model=APIResponse,
	summary="Delete capability",
	description="Delete a capability from the registry"
)
async def delete_capability(
	capability_id: str = Path(..., description="Capability ID"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Delete capability."""
	try:
		result = await service.delete_capability(capability_id)
		
		# Emit webhook event
		await emit_webhook_event(
			"capability.deleted",
			"capability",
			capability_id,
			{"deleted": True}
		)
		
		# Broadcast WebSocket update
		await broadcast_registry_update("capability_deleted", {"capability_id": capability_id})
		
		return APIResponse(
			success=True,
			message="Capability deleted successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# Composition API Endpoints
# =============================================================================

@api_app.get(
	"/api/compositions",
	response_model=PaginatedResponse,
	summary="List compositions",
	description="Get paginated list of compositions with filtering"
)
async def list_compositions(
	search: Optional[str] = Query(None, description="Search query"),
	composition_type: Optional[str] = Query(None, description="Type filter"),
	validation_status: Optional[str] = Query(None, description="Validation status filter"),
	is_template: Optional[bool] = Query(None, description="Template filter"),
	pagination: Dict[str, int] = Depends(validate_pagination),
	service: CRService = Depends(get_registry_service_dep)
):
	"""List compositions with pagination and filtering."""
	try:
		search_form = CompositionSearchForm(
			query=search,
			composition_type=composition_type,
			validation_status=validation_status,
			is_template=is_template,
			page=pagination["page"],
			per_page=pagination["per_page"]
		)
		
		compositions = await service.search_compositions(search_form.model_dump())
		
		composition_views = [
			CompositionListView(
				composition_id=comp.composition_id,
				name=comp.name,
				description=comp.description,
				composition_type=comp.composition_type,
				version=comp.version,
				validation_status=comp.validation_status,
				validation_score=comp.validation_score,
				estimated_complexity=comp.estimated_complexity,
				estimated_cost=comp.estimated_cost,
				capability_count=len(comp.capability_ids) if comp.capability_ids else 0,
				is_template=comp.is_template,
				is_public=comp.is_public,
				created_at=comp.created_at
			)
			for comp in compositions.get('compositions', [])
		]
		
		total_count = compositions.get('total_count', 0)
		total_pages = (total_count + pagination["per_page"] - 1) // pagination["per_page"]
		
		return PaginatedResponse(
			items=[comp.model_dump() for comp in composition_views],
			total_count=total_count,
			page=pagination["page"],
			per_page=pagination["per_page"],
			total_pages=total_pages,
			has_next=pagination["page"] < total_pages,
			has_prev=pagination["page"] > 1
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.post(
	"/api/compositions",
	response_model=APIResponse,
	status_code=HTTP_201_CREATED,
	summary="Create composition",
	description="Create a new capability composition"
)
async def create_composition(
	composition_form: CompositionCreateForm,
	service: CRService = Depends(get_registry_service_dep)
):
	"""Create new composition."""
	try:
		result = await service.create_composition(composition_form.model_dump())
		
		# Emit webhook event
		await emit_webhook_event(
			"composition.created",
			"composition",
			result["composition_id"],
			{"composition": composition_form.model_dump()}
		)
		
		# Broadcast WebSocket update
		await broadcast_registry_update("composition_created", result)
		
		return APIResponse(
			success=True,
			message="Composition created successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.post(
	"/api/compositions/validate",
	response_model=APIResponse,
	summary="Validate composition",
	description="Validate a capability composition"
)
async def validate_composition(
	capability_ids: List[str] = Body(..., description="Capability IDs to validate"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Validate composition."""
	try:
		result = await service.validate_composition(capability_ids)
		
		return APIResponse(
			success=True,
			message="Composition validated successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# Registry Management API
# =============================================================================

@api_app.get(
	"/api/registry/health",
	response_model=APIResponse,
	summary="Registry health check",
	description="Get registry health status and metrics"
)
async def registry_health(
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get registry health status."""
	try:
		health_data = await service.get_registry_health()
		
		return APIResponse(
			success=True,
			message="Registry health retrieved successfully",
			data=health_data
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.get(
	"/api/registry/dashboard",
	response_model=APIResponse,
	summary="Registry dashboard data",
	description="Get dashboard data for registry overview"
)
async def registry_dashboard(
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get registry dashboard data."""
	try:
		dashboard_data = await service.get_dashboard_data()
		
		return APIResponse(
			success=True,
			message="Dashboard data retrieved successfully",
			data=dashboard_data
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.post(
	"/api/registry/sync",
	response_model=APIResponse,
	summary="Sync registry",
	description="Trigger registry synchronization"
)
async def sync_registry(
	force_full_sync: bool = Body(False, description="Force full synchronization"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Trigger registry synchronization."""
	try:
		result = await service.sync_registry(force_full_sync)
		
		# Broadcast sync completion
		await broadcast_registry_update("registry_synced", result)
		
		return APIResponse(
			success=True,
			message="Registry sync completed successfully",
			data=result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# Analytics API Endpoints
# =============================================================================

@api_app.get(
	"/api/analytics/usage",
	response_model=APIResponse,
	summary="Usage analytics",
	description="Get capability usage analytics"
)
async def get_usage_analytics(
	start_date: Optional[datetime] = Query(None, description="Start date"),
	end_date: Optional[datetime] = Query(None, description="End date"),
	capability_id: Optional[str] = Query(None, description="Specific capability ID"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get usage analytics."""
	try:
		analytics_data = await service.get_usage_analytics(
			start_date=start_date,
			end_date=end_date,
			capability_id=capability_id
		)
		
		return APIResponse(
			success=True,
			message="Usage analytics retrieved successfully",
			data=analytics_data
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.get(
	"/api/analytics/performance",
	response_model=APIResponse,
	summary="Performance analytics",
	description="Get performance metrics and analytics"
)
async def get_performance_analytics(
	metric_type: Optional[str] = Query(None, description="Metric type"),
	time_range: str = Query("7d", description="Time range (1d, 7d, 30d, 90d)"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get performance analytics."""
	try:
		analytics_data = await service.get_performance_analytics(
			metric_type=metric_type,
			time_range=time_range
		)
		
		return APIResponse(
			success=True,
			message="Performance analytics retrieved successfully",
			data=analytics_data
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# WebSocket API
# =============================================================================

@api_app.websocket("/api/ws/{tenant_id}")
async def websocket_endpoint(
	websocket: WebSocket,
	tenant_id: str = Path(..., description="Tenant ID")
):
	"""WebSocket endpoint for real-time updates."""
	connection_id = uuid7str()
	
	await manager.connect(websocket, connection_id, tenant_id)
	
	try:
		# Send welcome message
		welcome_message = WebSocketMessage(
			type="connection",
			data={
				"connection_id": connection_id,
				"tenant_id": tenant_id,
				"message": "Connected to APG Registry WebSocket"
			}
		)
		await websocket.send_text(welcome_message.model_dump_json())
		
		while True:
			# Receive and process messages
			data = await websocket.receive_text()
			
			try:
				message = WebSocketMessage.model_validate_json(data)
				await handle_websocket_message(message, connection_id, tenant_id)
			except ValidationError as e:
				error_message = WebSocketMessage(
					type="error",
					data={"message": "Invalid message format", "errors": str(e)}
				)
				await websocket.send_text(error_message.model_dump_json())
			
	except WebSocketDisconnect:
		manager.disconnect(connection_id, tenant_id)
		print(f"WebSocket disconnected: {connection_id}")

async def handle_websocket_message(
	message: WebSocketMessage,
	connection_id: str,
	tenant_id: str
):
	"""Handle incoming WebSocket messages."""
	if message.type == "ping":
		# Respond to ping
		pong_message = WebSocketMessage(
			type="pong",
			data={"message": "pong", "timestamp": datetime.utcnow().isoformat()}
		)
		await manager.send_personal_message(pong_message.model_dump_json(), connection_id)
	
	elif message.type == "subscribe":
		# Subscribe to specific events
		subscription_data = message.data
		# Handle subscription logic
		response = WebSocketMessage(
			type="subscription_confirmed",
			data={"subscribed_to": subscription_data}
		)
		await manager.send_personal_message(response.model_dump_json(), connection_id)

async def broadcast_registry_update(event_type: str, data: Dict[str, Any]):
	"""Broadcast registry updates to all connected clients."""
	message = WebSocketMessage(
		type="registry_update",
		data={
			"event_type": event_type,
			"data": data,
			"timestamp": datetime.utcnow().isoformat()
		}
	)
	
	# Broadcast to all tenants (in production, filter by tenant)
	for tenant_id in manager.tenant_connections.keys():
		await manager.broadcast_to_tenant(message.model_dump_json(), tenant_id)

# =============================================================================
# Webhook Integration
# =============================================================================

# Webhook storage (in production, use proper database)
webhook_events: List[WebhookEvent] = []

async def emit_webhook_event(
	event_type: str,
	resource_type: str,
	resource_id: str,
	payload: Dict[str, Any],
	tenant_id: str = "default",
	user_id: Optional[str] = None
):
	"""Emit webhook event."""
	event = WebhookEvent(
		event_type=event_type,
		resource_type=resource_type,
		resource_id=resource_id,
		action=event_type.split('.')[-1],
		tenant_id=tenant_id,
		user_id=user_id,
		payload=payload
	)
	
	webhook_events.append(event)
	
	# In production, would send to configured webhook endpoints
	print(f"Webhook event emitted: {event_type} for {resource_type}:{resource_id}")

@api_app.get(
	"/api/webhooks/events",
	response_model=PaginatedResponse,
	summary="List webhook events",
	description="Get paginated list of webhook events"
)
async def list_webhook_events(
	event_type: Optional[str] = Query(None, description="Event type filter"),
	resource_type: Optional[str] = Query(None, description="Resource type filter"),
	since: Optional[datetime] = Query(None, description="Events since timestamp"),
	pagination: Dict[str, int] = Depends(validate_pagination)
):
	"""List webhook events."""
	try:
		# Filter events
		filtered_events = webhook_events
		
		if event_type:
			filtered_events = [e for e in filtered_events if e.event_type == event_type]
		
		if resource_type:
			filtered_events = [e for e in filtered_events if e.resource_type == resource_type]
		
		if since:
			filtered_events = [e for e in filtered_events if e.timestamp >= since]
		
		# Sort by timestamp (newest first)
		filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
		
		# Paginate
		start_idx = (pagination["page"] - 1) * pagination["per_page"]
		end_idx = start_idx + pagination["per_page"]
		page_events = filtered_events[start_idx:end_idx]
		
		total_count = len(filtered_events)
		total_pages = (total_count + pagination["per_page"] - 1) // pagination["per_page"]
		
		return PaginatedResponse(
			items=[event.model_dump() for event in page_events],
			total_count=total_count,
			page=pagination["page"],
			per_page=pagination["per_page"],
			total_pages=total_pages,
			has_next=pagination["page"] < total_pages,
			has_prev=pagination["page"] > 1
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# Mobile API Endpoints
# =============================================================================

@api_app.get(
	"/api/mobile/capabilities",
	response_model=APIResponse,
	summary="Mobile capabilities",
	description="Get mobile-optimized capability list"
)
async def get_mobile_capabilities(
	category: Optional[str] = Query(None, description="Category filter"),
	limit: int = Query(50, ge=1, le=100, description="Limit"),
	offset: int = Query(0, ge=0, description="Offset"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Get mobile-optimized capabilities."""
	try:
		from .mobile_service import MobileOfflineService
		
		mobile_service = MobileOfflineService()
		await mobile_service.set_online_service(service)
		
		capabilities = await mobile_service.get_mobile_capabilities(
			category=category,
			limit=limit,
			offset=offset
		)
		
		return APIResponse(
			success=True,
			message="Mobile capabilities retrieved successfully",
			data={
				"capabilities": [cap.model_dump() for cap in capabilities],
				"total": len(capabilities),
				"has_more": len(capabilities) == limit
			}
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@api_app.post(
	"/api/mobile/sync",
	response_model=APIResponse,
	summary="Mobile sync",
	description="Sync offline actions for mobile app"
)
async def sync_mobile_data(
	force_full_sync: bool = Body(False, description="Force full sync"),
	service: CRService = Depends(get_registry_service_dep)
):
	"""Sync mobile offline data."""
	try:
		from .mobile_service import MobileOfflineService
		
		mobile_service = MobileOfflineService()
		await mobile_service.set_online_service(service)
		await mobile_service.set_connection_status(True)
		
		sync_result = await mobile_service.sync_from_online(force_full_sync)
		
		return APIResponse(
			success=True,
			message="Mobile sync completed successfully",
			data=sync_result
		)
		
	except Exception as e:
		raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# =============================================================================
# Error Handlers
# =============================================================================

@api_app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
	"""Handle Pydantic validation errors."""
	return JSONResponse(
		status_code=HTTP_400_BAD_REQUEST,
		content=APIResponse(
			success=False,
			message="Validation error",
			errors=[str(error) for error in exc.errors()]
		).model_dump()
	)

@api_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
	"""Handle HTTP exceptions."""
	return JSONResponse(
		status_code=exc.status_code,
		content=APIResponse(
			success=False,
			message=exc.detail,
			errors=[exc.detail]
		).model_dump()
	)

# =============================================================================
# Health Check
# =============================================================================

@api_app.get("/api/health")
async def health_check():
	"""API health check endpoint."""
	return APIResponse(
		success=True,
		message="APG Capability Registry API is healthy",
		data={
			"version": "1.0.0",
			"timestamp": datetime.utcnow().isoformat(),
			"status": "operational"
		}
	)

# Export FastAPI app
__all__ = ["api_app"]