"""
APG Connection Management - REST API Layer

Comprehensive REST API with OpenAPI specification for connection management,
flow execution, and AI-powered integration capabilities.

Author: APG Platform Team
Version: 1.0.0
License: Proprietary - Datacraft © 2025
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from .service import ConnectionManager, FlowExecutor, IntelligentConnector
from .conn_runtime import ConnService
from .models import ConnectionStatus, ConnectionType, SyncMode
from .visual_designer import VisualFlowDesigner
from .data_lineage import DataLineageTracker
from .security import AuthenticationError, SecurityContext, auth_manager

# FastAPI App Configuration
app = FastAPI(
	title="APG Connection Management API",
	description="Enterprise integration platform with Singer.io ecosystem and AI-powered automation",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # In production, specify allowed origins
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)

# Security
security = HTTPBearer()

# Global Instances (In production, use dependency injection)
connection_manager = ConnectionManager()
flow_executor = FlowExecutor(connection_manager=connection_manager)
intelligent_connector = IntelligentConnector()
lineage_tracker = DataLineageTracker()
generated_conn_service = ConnService()

# Pydantic Models for API
class CreateConnectionRequest(BaseModel):
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)
	connection_type: ConnectionType
	singer_tap: Optional[str] = None
	singer_target: Optional[str] = None
	tap_config: Dict[str, Any] = Field(default_factory=dict)
	target_config: Dict[str, Any] = Field(default_factory=dict)
	sync_mode: SyncMode = SyncMode.INCREMENTAL
	sync_frequency: Optional[str] = None
	batch_size: int = Field(default=1000, ge=1, le=100000)
	tags: List[str] = Field(default_factory=list)

class ConnectionResponse(BaseModel):
	id: str
	name: str
	description: Optional[str]
	connection_type: ConnectionType
	status: ConnectionStatus
	singer_tap: Optional[str]
	singer_target: Optional[str]
	sync_mode: SyncMode
	batch_size: int
	last_sync: Optional[datetime]
	last_success: Optional[datetime]
	last_error: Optional[str]
	error_count: int
	tags: List[str]
	created_at: datetime
	updated_at: datetime
	created_by: str

class CreateFlowRequest(BaseModel):
	name: str = Field(..., min_length=1, max_length=255)
	description: Optional[str] = Field(None, max_length=1000)
	source_connection_id: str
	target_connection_id: str
	selected_streams: List[str] = Field(default_factory=list)
	transformation_rules: List[str] = Field(default_factory=list)
	schedule_expression: Optional[str] = None
	enabled: bool = False
	tags: List[str] = Field(default_factory=list)

class FlowResponse(BaseModel):
	id: str
	name: str
	description: Optional[str]
	source_connection_id: str
	target_connection_id: str
	selected_streams: List[str]
	enabled: bool
	schedule_expression: Optional[str]
	last_run: Optional[datetime]
	last_success: Optional[datetime]
	run_count: int
	success_count: int
	error_count: int
	created_at: datetime
	created_by: str

class SchemaAnalysisRequest(BaseModel):
	sample_data: List[Dict[str, Any]]
	source_name: str = "unknown"

class MappingSuggestionRequest(BaseModel):
	source_schema: Dict[str, Any]
	target_schema: Dict[str, Any]
	source_sample_data: Optional[List[Dict[str, Any]]] = None
	context: Optional[Dict[str, Any]] = None

class CreateVisualFlowRequest(BaseModel):
	name: str = Field(..., min_length=1, max_length=255)
	description: str = ""
	template_name: Optional[str] = None

# Authentication helpers
def _normalize_bearer_token(token: Optional[str]) -> Optional[str]:
	if token is None:
		return None

	normalized = str(token).strip()
	if not normalized:
		return None

	if normalized.lower().startswith("bearer "):
		normalized = normalized[7:].strip()

	return normalized or None


def _user_from_security_context(context: SecurityContext, source: str) -> Dict[str, Any]:
	return {
		"user_id": context.user.user_id,
		"username": context.user.username,
		"tenant_id": context.tenant_id,
		"roles": list(context.user.roles),
		"is_admin": context.user.is_admin,
		"session_id": context.session_id,
		"auth_source": source
	}


def validate_api_credentials(
	token: Optional[str],
	authentication_manager=auth_manager
) -> Dict[str, Any]:
	"""Validate REST/WebSocket credentials against APG security primitives."""
	normalized_token = _normalize_bearer_token(token)
	if not normalized_token:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Authentication token required"
		)

	session_context = authentication_manager.validate_session(normalized_token)
	if session_context:
		return _user_from_security_context(session_context, "session")

	jwt_error: Optional[AuthenticationError] = None
	try:
		payload = authentication_manager.validate_jwt_token(normalized_token)
	except AuthenticationError as error:
		jwt_error = error
	else:
		user_id = payload.get("user_id")
		tenant_id = payload.get("tenant_id")
		if not user_id or not tenant_id:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="Authentication token missing required identity claims"
			)
		return {
			"user_id": user_id,
			"username": payload.get("username"),
			"tenant_id": tenant_id,
			"roles": payload.get("roles", []),
			"is_admin": payload.get("is_admin", False),
			"session_id": None,
			"auth_source": "jwt"
		}

	try:
		api_key_context = authentication_manager.authenticate_api_key(normalized_token)
	except AuthenticationError as api_key_error:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid authentication token"
		) from jwt_error or api_key_error

	return _user_from_security_context(api_key_context, "api_key")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
	"""Extract the authenticated APG user from an HTTP bearer token."""
	return validate_api_credentials(credentials.credentials)


def get_websocket_user(websocket: WebSocket, authentication_manager=auth_manager) -> Dict[str, Any]:
	"""Extract the authenticated APG user from WebSocket headers or query params."""
	auth_header = websocket.headers.get("authorization")
	token = _normalize_bearer_token(auth_header)
	if not token:
		token = websocket.query_params.get("token") or websocket.query_params.get("access_token")
	return validate_api_credentials(token, authentication_manager=authentication_manager)

# API Startup
@app.on_event("startup")
async def startup_event():
	"""Initialize services on startup."""
	await connection_manager.initialize()
	await intelligent_connector.schema_analyzer._initialize_field_patterns()

# Connection Management Endpoints
@app.post("/api/v1/connections", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
async def create_connection(
	request: CreateConnectionRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Create a new connection with Singer.io integration."""
	try:
		connection_data = request.dict()
		connection_data["tenant_id"] = current_user["tenant_id"]
		connection_data["created_by"] = current_user["user_id"]

		connection = await connection_manager.create_connection(connection_data)

		return ConnectionResponse(
			id=connection.id,
			name=connection.name,
			description=connection.description,
			connection_type=connection.connection_type,
			status=connection.status,
			singer_tap=connection.singer_tap,
			singer_target=connection.singer_target,
			sync_mode=connection.sync_mode,
			batch_size=connection.batch_size,
			last_sync=connection.last_sync,
			last_success=connection.last_success,
			last_error=connection.last_error,
			error_count=connection.error_count,
			tags=connection.tags,
			created_at=connection.created_at,
			updated_at=connection.updated_at,
			created_by=connection.created_by
		)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/connections", response_model=List[ConnectionResponse])
async def list_connections(
	status: Optional[ConnectionStatus] = None,
	connection_type: Optional[ConnectionType] = None,
	current_user: dict = Depends(get_current_user)
):
	"""List connections with optional filtering."""
	try:
		connections = await connection_manager.list_connections(
			tenant_id=current_user["tenant_id"],
			status=status,
			connection_type=connection_type
		)

		return [
			ConnectionResponse(
				id=conn.id,
				name=conn.name,
				description=conn.description,
				connection_type=conn.connection_type,
				status=conn.status,
				singer_tap=conn.singer_tap,
				singer_target=conn.singer_target,
				sync_mode=conn.sync_mode,
				batch_size=conn.batch_size,
				last_sync=conn.last_sync,
				last_success=conn.last_success,
				last_error=conn.last_error,
				error_count=conn.error_count,
				tags=conn.tags,
				created_at=conn.created_at,
				updated_at=conn.updated_at,
				created_by=conn.created_by
			) for conn in connections
		]
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/connections/{connection_id}", response_model=ConnectionResponse)
async def get_connection(
	connection_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Get connection by ID."""
	connection = await connection_manager.get_connection(connection_id)
	if not connection:
		raise HTTPException(status_code=404, detail="Connection not found")

	return ConnectionResponse(
		id=connection.id,
		name=connection.name,
		description=connection.description,
		connection_type=connection.connection_type,
		status=connection.status,
		singer_tap=connection.singer_tap,
		singer_target=connection.singer_target,
		sync_mode=connection.sync_mode,
		batch_size=connection.batch_size,
		last_sync=connection.last_sync,
		last_success=connection.last_success,
		last_error=connection.last_error,
		error_count=connection.error_count,
		tags=connection.tags,
		created_at=connection.created_at,
		updated_at=connection.updated_at,
		created_by=connection.created_by
	)

@app.put("/api/v1/connections/{connection_id}", response_model=ConnectionResponse)
async def update_connection(
	connection_id: str,
	updates: Dict[str, Any],
	current_user: dict = Depends(get_current_user)
):
	"""Update connection configuration."""
	try:
		connection = await connection_manager.update_connection(connection_id, updates)

		return ConnectionResponse(
			id=connection.id,
			name=connection.name,
			description=connection.description,
			connection_type=connection.connection_type,
			status=connection.status,
			singer_tap=connection.singer_tap,
			singer_target=connection.singer_target,
			sync_mode=connection.sync_mode,
			batch_size=connection.batch_size,
			last_sync=connection.last_sync,
			last_success=connection.last_success,
			last_error=connection.last_error,
			error_count=connection.error_count,
			tags=connection.tags,
			created_at=connection.created_at,
			updated_at=connection.updated_at,
			created_by=connection.created_by
		)
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/v1/connections/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
	connection_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Delete connection."""
	try:
		await connection_manager.delete_connection(connection_id)
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/connections/{connection_id}/test")
async def test_connection(
	connection_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Test connection with live sync."""
	try:
		result = await connection_manager.test_connection_sync(connection_id)
		return result
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Flow Management Endpoints
@app.post("/api/v1/flows", response_model=FlowResponse, status_code=status.HTTP_201_CREATED)
async def create_flow(
	request: CreateFlowRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Create a new data flow."""
	try:
		flow_data = request.dict()
		flow_data["tenant_id"] = current_user["tenant_id"]
		flow_data["created_by"] = current_user["user_id"]

		flow = await flow_executor.create_flow(flow_data)

		return FlowResponse(
			id=flow.id,
			name=flow.name,
			description=flow.description,
			source_connection_id=flow.source_connection_id,
			target_connection_id=flow.target_connection_id,
			selected_streams=flow.selected_streams,
			enabled=flow.enabled,
			schedule_expression=flow.schedule_expression,
			last_run=flow.last_run,
			last_success=flow.last_success,
			run_count=flow.run_count,
			success_count=flow.success_count,
			error_count=flow.error_count,
			created_at=flow.created_at,
			created_by=flow.created_by
		)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/flows/{flow_id}/start")
async def start_flow(
	flow_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Start flow execution."""
	try:
		result = await flow_executor.start_flow(flow_id)
		return {"status": "started" if result else "failed"}
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/flows/{flow_id}/stop")
async def stop_flow(
	flow_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Stop flow execution."""
	try:
		result = await flow_executor.stop_flow(flow_id)
		return {"status": "stopped" if result else "failed"}
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/flows/{flow_id}/execute")
async def execute_flow_once(
	flow_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Execute flow once and return results."""
	try:
		result = await flow_executor.execute_flow_once(flow_id)
		return result
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# AI Intelligence Endpoints
@app.post("/api/v1/ai/detect-schema")
async def detect_schema(
	request: SchemaAnalysisRequest,
	current_user: dict = Depends(get_current_user)
):
	"""AI-powered schema detection from sample data."""
	try:
		result = await intelligent_connector.detect_schema(
			request.sample_data,
			request.source_name
		)
		return result
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/ai/suggest-mappings")
async def suggest_field_mappings(
	request: MappingSuggestionRequest,
	current_user: dict = Depends(get_current_user)
):
	"""AI-powered field mapping suggestions."""
	try:
		suggestions = await intelligent_connector.suggest_field_mappings(
			request.source_schema,
			request.target_schema,
			request.source_sample_data,
			request.context
		)
		return {"suggestions": suggestions}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/ai/predict-performance")
async def predict_performance(
	connection_config: Dict[str, Any],
	current_user: dict = Depends(get_current_user)
):
	"""Predict connection performance."""
	try:
		prediction = await intelligent_connector.predict_performance(connection_config)
		return prediction
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Visual Flow Designer Endpoints
@app.post("/api/v1/visual/flows", status_code=status.HTTP_201_CREATED)
async def create_visual_flow(
	request: CreateVisualFlowRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Create visual flow canvas."""
	try:
		canvas_id = await intelligent_connector.create_visual_flow(
			request.name,
			current_user["user_id"],
			request.template_name
		)
		return {"canvas_id": canvas_id}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/visual/flows/{canvas_id}")
async def get_visual_flow(
	canvas_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Get visual flow canvas."""
	try:
		canvas = intelligent_connector.visual_designer.canvases.get(canvas_id)
		if not canvas:
			raise HTTPException(status_code=404, detail="Canvas not found")

		return intelligent_connector.visual_designer._serialize_canvas(canvas)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/visual/flows/{canvas_id}/validate")
async def validate_visual_flow(
	canvas_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Validate visual flow."""
	try:
		validation = await intelligent_connector.validate_visual_flow(canvas_id)
		return validation
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/visual/flows/{canvas_id}/export")
async def export_visual_flow(
	canvas_id: str,
	format: str = "apg",
	current_user: dict = Depends(get_current_user)
):
	"""Export visual flow definition."""
	try:
		flow_definition = await intelligent_connector.export_flow_definition(canvas_id)
		return flow_definition
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Singer.io Endpoints
@app.get("/api/v1/singer/taps")
async def list_singer_taps(current_user: dict = Depends(get_current_user)):
	"""List available Singer taps."""
	taps = []
	for tap_name, tap in connection_manager.singer_runtime.tap_registry.items():
		taps.append({
			"name": tap.name,
			"display_name": tap.display_name,
			"description": tap.description,
			"version": tap.version,
			"installation_status": tap.installation_status,
			"supported_connection_types": [t.value for t in tap.supported_connection_types],
			"supports_incremental": tap.supports_incremental,
			"is_custom": tap.is_custom
		})

	return {"taps": taps}

@app.get("/api/v1/singer/targets")
async def list_singer_targets(current_user: dict = Depends(get_current_user)):
	"""List available Singer targets."""
	targets = []
	for target_name, target in connection_manager.singer_runtime.target_registry.items():
		targets.append({
			"name": target.name,
			"display_name": target.display_name,
			"description": target.description,
			"version": target.version,
			"installation_status": target.installation_status,
			"supported_connection_types": [t.value for t in target.supported_connection_types],
			"supports_upsert": target.supports_upsert,
			"is_custom": target.is_custom
		})

	return {"targets": targets}

@app.post("/api/v1/singer/taps/{tap_name}/install")
async def install_singer_tap(
	tap_name: str,
	current_user: dict = Depends(get_current_user)
):
	"""Install Singer tap."""
	try:
		result = await connection_manager.singer_runtime.install_tap(tap_name)
		return {"status": "installed" if result else "failed"}
	except AssertionError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Monitoring & Health Endpoints
@app.get("/api/v1/health")
async def health_check():
	"""Health check endpoint."""
	return {
		"status": "healthy",
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"version": "1.0.0",
		"components": {
			"connection_manager": "healthy",
			"flow_executor": "healthy",
			"ai_intelligence": "healthy"
		}
	}

@app.get("/api/v1/metrics")
async def get_metrics(current_user: dict = Depends(get_current_user)):
	"""Get comprehensive system metrics."""
	try:
		performance_metrics = await connection_manager.get_performance_metrics()
		ai_insights = await intelligent_connector.get_ai_insights()

		return {
			"connection_metrics": performance_metrics,
			"ai_insights": ai_insights,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/connections/{connection_id}/health")
async def get_connection_health(
	connection_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Get connection health status."""
	try:
		health = await connection_manager.get_connection_health(connection_id)
		if not health:
			raise HTTPException(status_code=404, detail="Health data not found")

		diagnostics = await health.run_diagnostics()

		return {
			"connection_id": health.connection_id,
			"status": health.status.value,
			"latency_ms": health.latency_ms,
			"error_rate": health.error_rate,
			"is_healthy": health.is_healthy(),
			"diagnostics": diagnostics,
			"timestamp": health.timestamp.isoformat()
		}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint for real-time updates (simplified)
@app.websocket("/api/v1/ws/{canvas_id}")
async def websocket_endpoint(websocket: WebSocket, canvas_id: str):
	"""WebSocket endpoint for real-time collaboration."""
	await websocket.accept()

	try:
		current_user = get_websocket_user(websocket)
	except HTTPException as exc:
		await websocket.send_json({"type": "auth_failed", "detail": exc.detail})
		await websocket.close(code=1008)
		return

	user_id = current_user["user_id"]
	session_info = await intelligent_connector.visual_designer.join_collaborative_session(
		canvas_id,
		user_id
	)

	await websocket.send_json({
		"type": "session_joined",
		"data": session_info,
		"user": {
			"user_id": current_user["user_id"],
			"tenant_id": current_user["tenant_id"],
			"auth_source": current_user["auth_source"]
		}
	})

	try:
		while True:
			data = await websocket.receive_json()

			if data["type"] == "cursor_move":
				await intelligent_connector.visual_designer.update_user_cursor(
					canvas_id,
					user_id,
					tuple(data["position"])
				)

			elif data["type"] == "ping":
				await websocket.send_json({
					"type": "pong",
					"timestamp": datetime.now(timezone.utc).isoformat()
				})

	except Exception as e:
		print(f"WebSocket error: {e}")
	finally:
		# Clean up user session
		if canvas_id in intelligent_connector.visual_designer.active_sessions:
			intelligent_connector.visual_designer.active_sessions[canvas_id].discard(user_id)

# Data Lineage API Models
class LineageVisualizationRequest(BaseModel):
	node_id: Optional[str] = None
	visualization_type: str = Field(default="full", pattern="^(full|upstream|downstream|impact)$")

class LineageSearchRequest(BaseModel):
	query: str = Field(..., min_length=1, max_length=100)
	search_type: str = Field(default="all", pattern="^(all|entities|fields|flows)$")

class TrackConnectionRequest(BaseModel):
	connection_id: str
	connection_name: str
	connection_type: str
	schema_info: Dict[str, Any] = Field(default_factory=dict)

class TrackFlowExecutionRequest(BaseModel):
	flow_id: str
	flow_name: str
	source_connection_id: str
	target_connection_id: str
	transformations: List[Dict[str, Any]] = Field(default_factory=list)
	field_mappings: Dict[str, str] = Field(default_factory=dict)

# Data Lineage & Visualization Endpoints
@app.post("/api/v1/lineage/track-connection")
async def track_connection_lineage(
	request: TrackConnectionRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Track connection in data lineage graph."""
	try:
		node_ids = await lineage_tracker.track_connection(
			request.connection_id,
			request.connection_name,
			request.connection_type,
			request.schema_info
		)
		return {"message": "Connection tracked successfully", "node_ids": node_ids}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/lineage/track-flow")
async def track_flow_execution(
	request: TrackFlowExecutionRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Track data flow execution in lineage graph."""
	try:
		await lineage_tracker.track_flow_execution(
			request.flow_id,
			request.flow_name,
			request.source_connection_id,
			request.target_connection_id,
			request.transformations,
			request.field_mappings
		)
		return {"message": "Flow execution tracked successfully"}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/lineage/visualization")
async def generate_lineage_visualization(
	request: LineageVisualizationRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Generate data lineage visualization."""
	try:
		visualization = await lineage_tracker.generate_lineage_visualization(
			request.node_id,
			request.visualization_type
		)
		return visualization
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/upstream/{node_id}")
async def get_upstream_lineage(
	node_id: str,
	max_depth: int = 10,
	current_user: dict = Depends(get_current_user)
):
	"""Get upstream lineage for a specific node."""
	try:
		lineage = lineage_tracker.lineage_graph.get_upstream_lineage(node_id, max_depth)
		return lineage
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/downstream/{node_id}")
async def get_downstream_lineage(
	node_id: str,
	max_depth: int = 10,
	current_user: dict = Depends(get_current_user)
):
	"""Get downstream lineage for a specific node."""
	try:
		lineage = lineage_tracker.lineage_graph.get_downstream_lineage(node_id, max_depth)
		return lineage
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/impact/{node_id}")
async def analyze_impact(
	node_id: str,
	current_user: dict = Depends(get_current_user)
):
	"""Analyze impact of changes to a specific node."""
	try:
		impact_analysis = lineage_tracker.lineage_graph.analyze_impact(node_id)
		return impact_analysis
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/catalog")
async def get_data_catalog(current_user: dict = Depends(get_current_user)):
	"""Get comprehensive data catalog from lineage information."""
	try:
		catalog = await lineage_tracker.get_data_catalog()
		return catalog
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/lineage/search")
async def search_lineage(
	request: LineageSearchRequest,
	current_user: dict = Depends(get_current_user)
):
	"""Search through lineage graph."""
	try:
		results = await lineage_tracker.search_lineage(
			request.query,
			request.search_type
		)
		return {"results": results, "total": len(results)}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/cycles")
async def detect_lineage_cycles(current_user: dict = Depends(get_current_user)):
	"""Detect cycles in the data lineage graph."""
	try:
		cycles = lineage_tracker.lineage_graph.detect_cycles()
		return {"cycles": cycles, "cycle_count": len(cycles)}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/root-sources")
async def get_root_sources(current_user: dict = Depends(get_current_user)):
	"""Find root source nodes in the lineage graph."""
	try:
		root_sources = lineage_tracker.lineage_graph.find_root_sources()
		return {"root_sources": [
			{
				"id": node.id,
				"name": node.name,
				"type": node.type,
				"source_type": node.source_type,
				"description": node.description
			} for node in root_sources
		]}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/lineage/leaf-destinations")
async def get_leaf_destinations(current_user: dict = Depends(get_current_user)):
	"""Find leaf destination nodes in the lineage graph."""
	try:
		leaf_destinations = lineage_tracker.lineage_graph.find_leaf_destinations()
		return {"leaf_destinations": [
			{
				"id": node.id,
				"name": node.name,
				"type": node.type,
				"source_type": node.source_type,
				"description": node.description
			} for node in leaf_destinations
		]}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


def capability_status(tenant_id: str = "default") -> Dict[str, Any]:
	"""Return dependency-light CONN generated-app status."""
	contract = generated_conn_service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**generated_conn_service.dashboard_summary(tenant_id),
	}


def register_generated_connector(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Register a connector through the generated-app control plane."""
	return generated_conn_service.register_connector(
		connector_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or "singer"),
		source_ref=str(payload["source_ref"]),
		checksum=str(payload["checksum"]),
		owner=str(payload["owner"]),
		verified_source=_payload_bool(payload, "verified_source", True),
		marketplace_review_recorded=_payload_bool(payload, "marketplace_review_recorded", False),
		auth_policy_attached=_payload_bool(payload, "auth_policy_attached", True),
		metadata=dict(payload.get("metadata") or {}),
	)


def register_generated_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Register a connection through the generated-app control plane."""
	return generated_conn_service.register_connection(
		connection_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		connector_id=str(payload["connector_id"]),
		owner=str(payload["owner"]),
		environment=str(payload.get("environment") or "development"),
		contains_credentials=_payload_bool(payload, "contains_credentials", True),
		credential_vault_ref=str(payload.get("credential_vault_ref") or ""),
		credentials_encrypted=_payload_bool(payload, "credentials_encrypted", True),
		cross_tenant_connection=_payload_bool(payload, "cross_tenant_connection", False),
		metadata=dict(payload.get("metadata") or {}),
	)


def record_generated_connection_test(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Record generated-app connection test evidence."""
	return generated_conn_service.record_connection_test(
		tenant_id=str(payload.get("tenant_id") or "default"),
		connection_id=str(payload["id"]),
		passed=_payload_bool(payload, "passed", False),
		evidence=dict(payload.get("evidence") or {}),
	)


def activate_generated_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Activate a generated-app connection."""
	return generated_conn_service.activate_connection(
		tenant_id=str(payload.get("tenant_id") or "default"),
		connection_id=str(payload["id"]),
		secret_rotation_recorded=_payload_bool(payload, "secret_rotation_recorded", False),
		activation_review_recorded=_payload_bool(payload, "activation_review_recorded", False),
	)


def create_generated_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Create a governed generated-app data flow."""
	return generated_conn_service.create_flow(
		flow_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		source_connection_id=str(payload["source_connection_id"]),
		target_connection_id=str(payload["target_connection_id"]),
		owner=str(payload["owner"]),
		mapping_ref=str(payload["mapping_ref"]),
		lineage_enabled=_payload_bool(payload, "lineage_enabled", True),
		quality_gate_ref=str(payload.get("quality_gate_ref") or ""),
		pii_detected=_payload_bool(payload, "pii_detected", False),
		pii_policy_attached=_payload_bool(payload, "pii_policy_attached", True),
	)


def start_generated_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Start a generated-app sync run."""
	return generated_conn_service.start_sync(
		run_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		flow_id=str(payload["flow_id"]),
		mode=str(payload.get("mode") or "incremental"),
		batch_size=int(payload.get("batch_size") or 1000),
		monitoring_enabled=_payload_bool(payload, "monitoring_enabled", True),
		schema_change_detected=_payload_bool(payload, "schema_change_detected", False),
		schema_review_recorded=_payload_bool(payload, "schema_review_recorded", False),
	)


def schedule_generated_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Schedule a generated-app flow."""
	return generated_conn_service.schedule_flow(
		tenant_id=str(payload.get("tenant_id") or "default"),
		schedule_id=str(payload["id"]),
		flow_id=str(payload["flow_id"]),
		cron=str(payload["cron"]),
		timezone=str(payload.get("timezone") or ""),
	)


def replay_generated_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Replay a generated-app sync run."""
	return generated_conn_service.replay_sync(
		tenant_id=str(payload.get("tenant_id") or "default"),
		run_id=str(payload["run_id"]),
		replay_id=str(payload["id"]),
		idempotency_key=str(payload.get("idempotency_key") or ""),
	)


def retire_generated_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Retire a generated-app connection."""
	return generated_conn_service.retire_connection(
		tenant_id=str(payload.get("tenant_id") or "default"),
		connection_id=str(payload["id"]),
		actor=str(payload["actor"]),
		impact_review_recorded=_payload_bool(payload, "impact_review_recorded", False),
	)


def list_generated_connectors(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_connectors(tenant_id)


def list_generated_connections(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_connections(tenant_id)


def list_generated_flows(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_flows(tenant_id)


def list_generated_sync_runs(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_sync_runs(tenant_id)


def list_generated_schedules(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_schedules(tenant_id)


def list_generated_reviews(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_reviews(tenant_id)


def list_generated_audit_events(tenant_id: str | None = None) -> List[Dict[str, Any]]:
	return generated_conn_service.list_audit_events(tenant_id)


def _payload_bool(payload: Dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)
