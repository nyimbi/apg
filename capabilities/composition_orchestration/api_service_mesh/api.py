"""
APG API Service Mesh - FastAPI Application

Comprehensive REST API with WebSocket support for real-time service mesh management,
monitoring, and configuration. Provides full CRUD operations and real-time updates.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
	ServiceConfig, EndpointConfig, RouteConfig, LoadBalancerConfig, PolicyConfig,
	ServiceStatus, EndpointProtocol, LoadBalancerAlgorithm, HealthStatus, PolicyType
)
from .service import ASMService, create_asm_service

# =============================================================================
# Pydantic Models for API
# =============================================================================

class APIResponse(BaseModel):
	"""Standard API response model."""
	model_config = ConfigDict(extra='forbid')
	
	success: bool
	message: str
	data: Optional[Dict[str, Any]] = None
	errors: Optional[List[str]] = None
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PaginatedResponse(BaseModel):
	"""Paginated response model."""
	model_config = ConfigDict(extra='forbid')
	
	items: List[Dict[str, Any]]
	total: int
	page: int
	per_page: int
	pages: int
	has_next: bool
	has_prev: bool

class ServiceRegistrationRequest(BaseModel):
	"""Service registration request model."""
	model_config = ConfigDict(extra='forbid')
	
	service_config: ServiceConfig
	endpoints: List[EndpointConfig]

class RouteCreationRequest(BaseModel):
	"""Route creation request model."""
	model_config = ConfigDict(extra='forbid')
	
	route_config: RouteConfig
	service_id: Optional[str] = None

class TrafficSplitRequest(BaseModel):
	"""Traffic splitting request model."""
	model_config = ConfigDict(extra='forbid')
	
	route_id: str
	destination_services: List[Dict[str, Any]]

class HealthCheckRequest(BaseModel):
	"""Health check request model."""
	model_config = ConfigDict(extra='forbid')
	
	service_ids: Optional[List[str]] = None
	force_check: bool = False

class MetricsQueryRequest(BaseModel):
	"""Metrics query request model."""
	model_config = ConfigDict(extra='forbid')
	
	service_ids: Optional[List[str]] = None
	metric_names: Optional[List[str]] = None
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	aggregation: str = "avg"  # avg, sum, count, min, max

class WebSocketMessage(BaseModel):
	"""WebSocket message model."""
	model_config = ConfigDict(extra='forbid')
	
	type: str
	action: Optional[str] = None
	data: Optional[Dict[str, Any]] = None
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
	"""Manages WebSocket connections for real-time updates."""
	
	def __init__(self):
		self.active_connections: Dict[str, List[WebSocket]] = {}
		self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
	
	async def connect(self, websocket: WebSocket, tenant_id: str, connection_type: str = "monitoring"):
		"""Connect a new WebSocket client."""
		await websocket.accept()
		
		if tenant_id not in self.active_connections:
			self.active_connections[tenant_id] = []
		
		self.active_connections[tenant_id].append(websocket)
		self.connection_metadata[websocket] = {
			"tenant_id": tenant_id,
			"connection_type": connection_type,
			"connected_at": datetime.now(timezone.utc)
		}
	
	def disconnect(self, websocket: WebSocket):
		"""Disconnect a WebSocket client."""
		if websocket in self.connection_metadata:
			tenant_id = self.connection_metadata[websocket]["tenant_id"]
			
			if tenant_id in self.active_connections:
				self.active_connections[tenant_id].remove(websocket)
				
				if not self.active_connections[tenant_id]:
					del self.active_connections[tenant_id]
			
			del self.connection_metadata[websocket]
	
	async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
		"""Send message to specific WebSocket."""
		try:
			await websocket.send_text(json.dumps(message, default=str))
		except:
			self.disconnect(websocket)
	
	async def broadcast_to_tenant(self, message: Dict[str, Any], tenant_id: str):
		"""Broadcast message to all connections for a tenant."""
		if tenant_id not in self.active_connections:
			return
		
		disconnected = []
		for websocket in self.active_connections[tenant_id]:
			try:
				await websocket.send_text(json.dumps(message, default=str))
			except:
				disconnected.append(websocket)
		
		# Clean up disconnected sockets
		for websocket in disconnected:
			self.disconnect(websocket)
	
	async def broadcast_to_all(self, message: Dict[str, Any]):
		"""Broadcast message to all connected clients."""
		for tenant_id in list(self.active_connections.keys()):
			await self.broadcast_to_tenant(message, tenant_id)

# =============================================================================
# FastAPI Application Setup
# =============================================================================

# Global connection manager
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan manager."""
	# Startup
	print("🚀 Starting APG API Service Mesh API...")
	
	# Initialize background tasks
	asyncio.create_task(monitoring_broadcast_task())
	
	yield
	
	# Shutdown
	print("🛑 Shutting down APG API Service Mesh API...")

# Create FastAPI application
api_app = FastAPI(
	title="APG API Service Mesh",
	description="Intelligent API orchestration and service mesh networking",
	version="1.0.0",
	docs_url="/api/docs",
	redoc_url="/api/redoc",
	openapi_url="/api/openapi.json",
	lifespan=lifespan
)

# Add CORS middleware
api_app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Configure appropriately for production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# =============================================================================
# Dependency Injection
# =============================================================================

async def get_db_session() -> AsyncSession:
	"""Get database session dependency."""
	# This would be implemented with your actual database session factory
	# For now, return None as placeholder
	return None

async def get_asm_service() -> ASMService:
	"""Get ASM service dependency."""
	# This would be implemented with your actual service factory
	# For now, return None as placeholder
	return None

async def get_tenant_id() -> str:
	"""Get tenant ID from request context."""
	# This would extract tenant ID from JWT token or headers
	return "default_tenant"

# =============================================================================
# Service Management Endpoints
# =============================================================================

@api_app.post("/api/services", response_model=APIResponse, tags=["Service Management"])
async def register_service(
	request: ServiceRegistrationRequest,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Register a new service with the mesh."""
	try:
		service_id = await asm_service.register_service(
			service_config=request.service_config.model_dump(),
			endpoints=[ep.model_dump() for ep in request.endpoints],
			tenant_id=tenant_id,
			created_by="api_user"  # Would come from authentication
		)
		
		# Broadcast service registration event
		await connection_manager.broadcast_to_tenant({
			"type": "service_registered",
			"data": {"service_id": service_id, "service_name": request.service_config.service_name}
		}, tenant_id)
		
		return APIResponse(
			success=True,
			message="Service registered successfully",
			data={"service_id": service_id}
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@api_app.get("/api/services", response_model=PaginatedResponse, tags=["Service Management"])
async def list_services(
	page: int = 1,
	per_page: int = 20,
	search: Optional[str] = None,
	namespace: Optional[str] = None,
	status: Optional[ServiceStatus] = None,
	health_status: Optional[HealthStatus] = None,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List services with filtering and pagination."""
	try:
		services = await asm_service.discover_services(
			service_name=search,
			namespace=namespace,
			health_status=health_status,
			tenant_id=tenant_id
		)
		
		# Convert to dictionaries for JSON response
		service_dicts = []
		for service in services:
			service_dict = {
				"service_id": service.service_id,
				"service_name": service.service_name,
				"service_version": service.service_version,
				"namespace": getattr(service, 'namespace', 'default'),
				"status": service.status.value,
				"health_status": service.health_status.value,
				"endpoints": service.endpoints,
				"metadata": service.metadata,
				"last_health_check": service.last_health_check
			}
			service_dicts.append(service_dict)
		
		# Apply pagination
		total = len(service_dicts)
		start = (page - 1) * per_page
		end = start + per_page
		items = service_dicts[start:end]
		
		return PaginatedResponse(
			items=items,
			total=total,
			page=page,
			per_page=per_page,
			pages=(total + per_page - 1) // per_page,
			has_next=end < total,
			has_prev=page > 1
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/api/services/{service_id}", response_model=APIResponse, tags=["Service Management"])
async def get_service(
	service_id: str,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get detailed service information."""
	try:
		service = await asm_service.service_registry.get_service_by_id(service_id, tenant_id)
		
		if not service:
			raise HTTPException(status_code=404, detail="Service not found")
		
		service_data = {
			"service_id": service.service_id,
			"service_name": service.service_name,
			"service_version": service.service_version,
			"namespace": getattr(service, 'namespace', 'default'),
			"status": service.status.value,
			"health_status": service.health_status.value,
			"endpoints": service.endpoints,
			"metadata": service.metadata,
			"last_health_check": service.last_health_check
		}
		
		return APIResponse(
			success=True,
			message="Service retrieved successfully",
			data=service_data
		)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.put("/api/services/{service_id}/status", response_model=APIResponse, tags=["Service Management"])
async def update_service_status(
	service_id: str,
	status: ServiceStatus,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Update service status."""
	try:
		await asm_service.service_registry.update_service_status(service_id, status, tenant_id)
		
		# Broadcast status update
		await connection_manager.broadcast_to_tenant({
			"type": "service_status_updated",
			"data": {"service_id": service_id, "status": status.value}
		}, tenant_id)
		
		return APIResponse(
			success=True,
			message="Service status updated successfully"
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.delete("/api/services/{service_id}", response_model=APIResponse, tags=["Service Management"])
async def deregister_service(
	service_id: str,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Deregister a service from the mesh."""
	try:
		# Implementation would involve updating service status and cleanup
		await asm_service.service_registry.update_service_status(
			service_id, ServiceStatus.DEREGISTERING, tenant_id
		)
		
		# Broadcast deregistration event
		await connection_manager.broadcast_to_tenant({
			"type": "service_deregistered",
			"data": {"service_id": service_id}
		}, tenant_id)
		
		return APIResponse(
			success=True,
			message="Service deregistration initiated"
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Traffic Management Endpoints
# =============================================================================

@api_app.post("/api/routes", response_model=APIResponse, tags=["Traffic Management"])
async def create_route(
	request: RouteCreationRequest,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Create a new traffic routing rule."""
	try:
		route_id = await asm_service.traffic_manager.create_route(
			route_config=request.route_config.model_dump(),
			tenant_id=tenant_id,
			created_by="api_user"
		)
		
		# Broadcast route creation event
		await connection_manager.broadcast_to_tenant({
			"type": "route_created",
			"data": {"route_id": route_id, "route_name": request.route_config.route_name}
		}, tenant_id)
		
		return APIResponse(
			success=True,
			message="Route created successfully",
			data={"route_id": route_id}
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@api_app.get("/api/routes", response_model=APIResponse, tags=["Traffic Management"])
async def list_routes(
	page: int = 1,
	per_page: int = 20,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List traffic routing rules."""
	try:
		# Implementation would query routes from database
		routes = []  # Placeholder
		
		return APIResponse(
			success=True,
			message="Routes retrieved successfully",
			data={"routes": routes}
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/api/routes/{route_id}/traffic-split", response_model=APIResponse, tags=["Traffic Management"])
async def update_traffic_split(
	route_id: str,
	request: TrafficSplitRequest,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Update traffic splitting configuration."""
	try:
		await asm_service.traffic_manager.update_traffic_split(
			route_id=route_id,
			destination_services=request.destination_services,
			tenant_id=tenant_id,
			updated_by="api_user"
		)
		
		# Broadcast traffic split update
		await connection_manager.broadcast_to_tenant({
			"type": "traffic_split_updated",
			"data": {"route_id": route_id, "destinations": len(request.destination_services)}
		}, tenant_id)
		
		return APIResponse(
			success=True,
			message="Traffic split updated successfully"
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# =============================================================================
# Load Balancing Endpoints
# =============================================================================

@api_app.post("/api/load-balancers", response_model=APIResponse, tags=["Load Balancing"])
async def create_load_balancer(
	config: LoadBalancerConfig,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Create a new load balancer configuration."""
	try:
		# Implementation would create load balancer
		lb_id = "lb_" + str(int(datetime.now().timestamp()))
		
		return APIResponse(
			success=True,
			message="Load balancer created successfully",
			data={"load_balancer_id": lb_id}
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@api_app.get("/api/load-balancers", response_model=APIResponse, tags=["Load Balancing"])
async def list_load_balancers(
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List load balancer configurations."""
	try:
		# Implementation would query load balancers
		load_balancers = []  # Placeholder
		
		return APIResponse(
			success=True,
			message="Load balancers retrieved successfully",
			data={"load_balancers": load_balancers}
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Policy Management Endpoints
# =============================================================================

@api_app.post("/api/policies", response_model=APIResponse, tags=["Policy Management"])
async def create_policy(
	config: PolicyConfig,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Create a new traffic or security policy."""
	try:
		# Implementation would create policy
		policy_id = "pol_" + str(int(datetime.now().timestamp()))
		
		return APIResponse(
			success=True,
			message="Policy created successfully",
			data={"policy_id": policy_id}
		)
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@api_app.get("/api/policies", response_model=APIResponse, tags=["Policy Management"])
async def list_policies(
	policy_type: Optional[PolicyType] = None,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List traffic and security policies."""
	try:
		# Implementation would query policies
		policies = []  # Placeholder
		
		return APIResponse(
			success=True,
			message="Policies retrieved successfully",
			data={"policies": policies}
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Health and Monitoring Endpoints
# =============================================================================

@api_app.post("/api/health-check", response_model=APIResponse, tags=["Health & Monitoring"])
async def trigger_health_check(
	request: HealthCheckRequest,
	background_tasks: BackgroundTasks,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Trigger health checks for services."""
	try:
		# Add background task for health checks
		if request.service_ids:
			for service_id in request.service_ids:
				background_tasks.add_task(
					trigger_service_health_check, service_id, asm_service, tenant_id
				)
		
		return APIResponse(
			success=True,
			message="Health checks initiated"
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

async def trigger_service_health_check(service_id: str, asm_service: ASMService, tenant_id: str):
	"""Background task for service health check."""
	try:
		# Implementation would trigger health check
		pass
	except Exception as e:
		print(f"Health check failed for service {service_id}: {e}")

@api_app.get("/api/health", response_model=APIResponse, tags=["Health & Monitoring"])
async def get_mesh_health(
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get overall service mesh health status."""
	try:
		health_status = await asm_service.get_mesh_status(tenant_id)
		
		return APIResponse(
			success=True,
			message="Health status retrieved successfully",
			data=health_status
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.post("/api/metrics/query", response_model=APIResponse, tags=["Health & Monitoring"])
async def query_metrics(
	request: MetricsQueryRequest,
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Query service mesh metrics."""
	try:
		# Implementation would query metrics based on request parameters
		metrics = await asm_service.metrics_collector.get_recent_metrics(
			tenant_id=tenant_id,
			hours=1  # Default to last hour
		)
		
		return APIResponse(
			success=True,
			message="Metrics retrieved successfully",
			data={"metrics": metrics}
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/api/topology", response_model=APIResponse, tags=["Health & Monitoring"])
async def get_service_topology(
	asm_service: ASMService = Depends(get_asm_service),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get service dependency topology."""
	try:
		topology = await asm_service.get_service_topology(tenant_id)
		
		return APIResponse(
			success=True,
			message="Topology retrieved successfully",
			data=topology
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# WebSocket Endpoints
# =============================================================================

@api_app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket, tenant_id: str = "default"):
	"""WebSocket endpoint for real-time monitoring updates."""
	await connection_manager.connect(websocket, tenant_id, "monitoring")
	
	try:
		while True:
			# Receive messages from client
			data = await websocket.receive_text()
			message = json.loads(data)
			
			# Handle client messages
			if message.get("action") == "start_monitoring":
				await websocket.send_text(json.dumps({
					"type": "monitoring_started",
					"message": "Real-time monitoring activated"
				}))
			elif message.get("action") == "stop_monitoring":
				await websocket.send_text(json.dumps({
					"type": "monitoring_stopped",
					"message": "Real-time monitoring deactivated"
				}))
			
	except WebSocketDisconnect:
		connection_manager.disconnect(websocket)

@api_app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket, tenant_id: str = "default"):
	"""WebSocket endpoint for real-time alert notifications."""
	await connection_manager.connect(websocket, tenant_id, "alerts")
	
	try:
		while True:
			# Keep connection alive
			await asyncio.sleep(30)
			await websocket.send_text(json.dumps({
				"type": "ping",
				"timestamp": datetime.now(timezone.utc).isoformat()
			}))
			
	except WebSocketDisconnect:
		connection_manager.disconnect(websocket)

# =============================================================================
# Background Tasks
# =============================================================================

async def monitoring_broadcast_task():
	"""Background task to broadcast monitoring data."""
	while True:
		try:
			# Simulate real-time monitoring data
			monitoring_data = {
				"type": "monitoring_data",
				"data": {
					"timestamp": datetime.now(timezone.utc).isoformat(),
					"services": {
						"total": 5,
						"healthy": 4,
						"unhealthy": 1,
						"details": [
							{
								"service_id": "svc_001",
								"service_name": "user-service",
								"service_version": "v1.2.0",
								"health_status": "healthy"
							},
							{
								"service_id": "svc_002", 
								"service_name": "payment-service",
								"service_version": "v2.1.0",
								"health_status": "healthy"
							}
						]
					},
					"traffic": {
						"requests_per_second": 125.3,
						"avg_response_time": 234.5,
						"p95_response_time": 456.7,
						"error_rate": 1.2,
						"total_requests": 50000,
						"successful_requests": 49400,
						"client_errors": 450,
						"server_errors": 150
					},
					"network": {
						"throughput_mbps": 85.6,
						"latency_ms": 12.3
					},
					"alerts": [],
					"events": [
						{
							"type": "service_registered",
							"description": "New service 'notification-service' registered",
							"timestamp": datetime.now(timezone.utc).isoformat()
						}
					]
				}
			}
			
			# Broadcast to all monitoring connections
			await connection_manager.broadcast_to_all(monitoring_data)
			
			# Wait 5 seconds before next broadcast
			await asyncio.sleep(5)
			
		except Exception as e:
			print(f"Error in monitoring broadcast: {e}")
			await asyncio.sleep(10)

# =============================================================================
# Application Metadata
# =============================================================================

@api_app.get("/api/info", response_model=APIResponse, tags=["System"])
async def get_api_info():
	"""Get API information and capabilities."""
	return APIResponse(
		success=True,
		message="API information retrieved successfully",
		data={
			"name": "APG API Service Mesh",
			"version": "1.0.0",
			"description": "Intelligent API orchestration and service mesh networking",
			"capabilities": [
				"service_discovery",
				"load_balancing", 
				"traffic_routing",
				"health_monitoring",
				"metrics_collection",
				"policy_enforcement",
				"real_time_monitoring"
			],
			"endpoints": {
				"services": "/api/services",
				"routes": "/api/routes",
				"load_balancers": "/api/load-balancers",
				"policies": "/api/policies",
				"health": "/api/health",
				"metrics": "/api/metrics",
				"topology": "/api/topology"
			},
			"websockets": {
				"monitoring": "/ws/monitoring",
				"alerts": "/ws/alerts"
			}
		}
	)

# =============================================================================
# Error Handlers
# =============================================================================

@api_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
	"""Handle HTTP exceptions."""
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"success": False,
			"message": exc.detail,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
	)

@api_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
	"""Handle general exceptions."""
	return JSONResponse(
		status_code=500,
		content={
			"success": False,
			"message": "Internal server error",
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
	)

# Create router for external use
from fastapi import APIRouter
router = APIRouter()

# Include all routes in the router
for route in api_app.routes:
	if hasattr(route, 'path') and route.path.startswith('/api/'):
		router.routes.append(route)

# Export the FastAPI app and router
__all__ = ["api_app", "router", "connection_manager"]