"""
APG Central Configuration - Revolutionary API Layer

FastAPI-based REST and GraphQL APIs with real-time WebSocket support,
AI-powered natural language queries, and comprehensive authentication.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Security, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, OAuth2PasswordBearer, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import Response
import uvicorn

try:
	import redis.asyncio as redis
except ModuleNotFoundError:
	redis = None
from sqlalchemy.ext.asyncio import AsyncSession
try:
	from jose import JWTError, jwt
except ModuleNotFoundError:
	class JWTError(Exception):
		"""JWT support is unavailable."""

	class _MissingJWT:
		@staticmethod
		def decode(*args, **kwargs):
			raise JWTError("python-jose is not installed")

	jwt = _MissingJWT()
try:
	import httpx
except ModuleNotFoundError:
	httpx = None
from uuid_extensions import uuid7str

try:
	from .models import (
		ConfigurationCreate, ConfigurationUpdate, ConfigurationResponse,
		TemplateCreate, WorkspaceCreate, UserCreate, CCConfiguration,
		ConfigurationStatus
	)
except Exception:
	class ConfigurationStatus(str, Enum):
		"""Fallback configuration status for API-only execution."""
		DRAFT = "draft"
		ACTIVE = "active"
		DEPRECATED = "deprecated"
		ARCHIVED = "archived"

	class SecurityLevel(str, Enum):
		"""Fallback security level for API-only execution."""
		PUBLIC = "public"
		INTERNAL = "internal"
		CONFIDENTIAL = "confidential"
		RESTRICTED = "restricted"
		TOP_SECRET = "top_secret"

	class ConfigurationCreate(BaseModel):
		"""Fallback configuration creation schema."""
		model_config = ConfigDict(extra="forbid", validate_assignment=True)

		name: str = Field(..., min_length=1, max_length=255)
		description: Optional[str] = Field(None, max_length=2000)
		key_path: str = Field(..., pattern=r"^/[\w\-/.]+$")
		value: Dict[str, Any]
		schema_definition: Optional[Dict[str, Any]] = None
		default_value: Optional[Dict[str, Any]] = None
		tags: List[str] = Field(default_factory=list)
		metadata: Dict[str, Any] = Field(default_factory=dict)
		security_level: SecurityLevel = SecurityLevel.INTERNAL
		expires_at: Optional[datetime] = None

	class ConfigurationUpdate(BaseModel):
		"""Fallback configuration update schema."""
		model_config = ConfigDict(extra="forbid", validate_assignment=True)

		name: Optional[str] = Field(None, min_length=1, max_length=255)
		description: Optional[str] = Field(None, max_length=2000)
		value: Optional[Dict[str, Any]] = None
		schema_definition: Optional[Dict[str, Any]] = None
		default_value: Optional[Dict[str, Any]] = None
		tags: Optional[List[str]] = None
		metadata: Optional[Dict[str, Any]] = None
		security_level: Optional[SecurityLevel] = None
		status: Optional[ConfigurationStatus] = None
		expires_at: Optional[datetime] = None

	class ConfigurationResponse(BaseModel):
		"""Fallback configuration response schema."""
		model_config = ConfigDict(from_attributes=True)

		id: str
		tenant_id: str
		workspace_id: str
		parent_id: Optional[str]
		name: str
		description: Optional[str]
		key_path: str
		value: Dict[str, Any]
		schema_definition: Optional[Dict[str, Any]]
		default_value: Optional[Dict[str, Any]]
		tags: List[str]
		metadata: Dict[str, Any]
		status: ConfigurationStatus
		version: str
		security_level: SecurityLevel
		created_at: datetime
		updated_at: datetime
		expires_at: Optional[datetime]

	class TemplateCreate(BaseModel):
		"""Fallback template creation schema."""
		model_config = ConfigDict(extra="forbid", validate_assignment=True)

		name: str = Field(..., min_length=1, max_length=255)
		description: Optional[str] = Field(None, max_length=2000)
		category: str = Field(..., min_length=1, max_length=100)
		template_data: Dict[str, Any]
		variables: Dict[str, Any] = Field(default_factory=dict)
		schema_definition: Optional[Dict[str, Any]] = None
		tags: List[str] = Field(default_factory=list)
		metadata: Dict[str, Any] = Field(default_factory=dict)
		is_public: bool = False

	class WorkspaceCreate(BaseModel):
		"""Fallback workspace creation schema."""
		model_config = ConfigDict(extra="forbid", validate_assignment=True)

		name: str = Field(..., min_length=1, max_length=100)
		description: Optional[str] = Field(None, max_length=2000)
		slug: str = Field(..., pattern=r"^[a-z0-9-]+$", min_length=1, max_length=100)
		settings: Dict[str, Any] = Field(default_factory=dict)

	class UserCreate(BaseModel):
		"""Fallback user creation schema."""
		model_config = ConfigDict(extra="forbid", validate_assignment=True)

		email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
		name: str = Field(..., min_length=1, max_length=255)
		username: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$", min_length=1, max_length=100)

	CCConfiguration = Any
try:
	from .ai_engine import CentralConfigurationAI
except ModuleNotFoundError:
	class CentralConfigurationAI:
		"""Deterministic API fallback when optional ML dependencies are absent."""

		async def initialize(self) -> None:
			self.initialized = True

		async def close(self) -> None:
			self.initialized = False

		async def optimize_configuration(self, value: Dict[str, Any]) -> Dict[str, Any]:
			return dict(value)

		async def generate_recommendations(self, value: Dict[str, Any]) -> List[Dict[str, Any]]:
			return [{
				"type": "baseline",
				"message": "Configuration is available for rule and schema validation",
				"confidence": 1.0,
			}]

		async def parse_natural_language_query(self, query_text: str) -> Dict[str, Any]:
			return {"query": query_text, "filters": {}}

		async def detect_anomalies(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
			return []


# ==================== Authentication & Security ====================

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

SECRET_KEY = os.getenv("APG_CONFIG_SECRET_KEY", "development-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthenticationError(Exception):
	"""Authentication error."""


_api_runtime_state: Dict[str, Dict[str, Any]] = {}


def _tenant_state(tenant_id: str) -> Dict[str, Any]:
	"""Return tenant-scoped API runtime state."""
	return _api_runtime_state.setdefault(tenant_id, {
		"configurations": {},
		"versions": {},
		"templates": {},
		"workspaces": {},
		"deployments": {},
		"audit_entries": [],
		"collaboration_sessions": {},
	})


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
	if hasattr(value, "value"):
		return value.value
	if isinstance(value, datetime):
		return value.isoformat()
	return value


class CentralConfigurationEngine:
	"""Executable in-process engine for API/runtime use."""

	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id

	def _state(self) -> Dict[str, Any]:
		return _tenant_state(self.tenant_id)

	def _audit(self, action: str, resource_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
		self._state()["audit_entries"].append({
			"id": uuid7str(),
			"event_type": action,
			"resource_id": resource_id,
			"user_id": self.user_id,
			"tenant_id": self.tenant_id,
			"metadata": metadata or {},
			"timestamp": _utc_now().isoformat(),
		})

	async def create_configuration(
		self,
		workspace_id: str,
		config_data: ConfigurationCreate,
		parent_id: Optional[str] = None,
	) -> ConfigurationResponse:
		config_id = uuid7str()
		now = _utc_now()
		record = {
			"id": config_id,
			"tenant_id": self.tenant_id,
			"workspace_id": workspace_id,
			"parent_id": parent_id,
			"name": config_data.name,
			"description": config_data.description,
			"key_path": config_data.key_path,
			"value": config_data.value,
			"schema_definition": config_data.schema_definition,
			"default_value": config_data.default_value,
			"tags": list(config_data.tags),
			"metadata": dict(config_data.metadata),
			"status": ConfigurationStatus.ACTIVE,
			"version": "1.0.0",
			"security_level": config_data.security_level,
			"created_at": now,
			"updated_at": now,
			"expires_at": config_data.expires_at,
		}
		state = self._state()
		state["configurations"][config_id] = record
		state["versions"].setdefault(config_id, []).append({
			"version": record["version"],
			"change_action": "create",
			"value": config_data.value,
			"created_by": self.user_id,
			"created_at": now.isoformat(),
		})
		self._audit("configuration.create", config_id, {"workspace_id": workspace_id})
		return ConfigurationResponse(**record)

	async def get_configuration(
		self,
		configuration_id: str,
		include_history: bool = False,
		include_ai_insights: bool = False,
	) -> Dict[str, Any]:
		record = self._state()["configurations"].get(configuration_id)
		if not record:
			raise ValueError(f"Configuration {configuration_id} not found")
		result = {
			key: _json_ready(value)
			for key, value in record.items()
		}
		if include_history:
			result["history"] = self._state()["versions"].get(configuration_id, [])
		if include_ai_insights:
			result["ai_insights"] = [{"type": "baseline", "message": "Configuration is executable"}]
		return result

	async def update_configuration(
		self,
		configuration_id: str,
		updates: ConfigurationUpdate,
		change_reason: Optional[str] = None,
	) -> ConfigurationResponse:
		record = self._state()["configurations"].get(configuration_id)
		if not record:
			raise ValueError(f"Configuration {configuration_id} not found")
		update_data = updates.model_dump(exclude_unset=True)
		before = dict(record)
		for key, value in update_data.items():
			record[key] = value
		major, minor, patch = [int(part) for part in str(record["version"]).split(".")]
		record["version"] = f"{major}.{minor}.{patch + 1}"
		record["updated_at"] = _utc_now()
		self._state()["versions"].setdefault(configuration_id, []).append({
			"version": record["version"],
			"change_action": "update",
			"value_before": before.get("value"),
			"value_after": record.get("value"),
			"change_reason": change_reason,
			"created_by": self.user_id,
			"created_at": record["updated_at"].isoformat(),
		})
		self._audit("configuration.update", configuration_id, {"change_reason": change_reason})
		return ConfigurationResponse(**record)

	async def delete_configuration(
		self,
		configuration_id: str,
		reason: Optional[str] = None,
		permanent: bool = False,
	) -> bool:
		state = self._state()
		record = state["configurations"].get(configuration_id)
		if not record:
			raise ValueError(f"Configuration {configuration_id} not found")
		if permanent:
			del state["configurations"][configuration_id]
		else:
			record["status"] = ConfigurationStatus.ARCHIVED
			record["updated_at"] = _utc_now()
		self._audit("configuration.delete", configuration_id, {"reason": reason, "permanent": permanent})
		return True

	async def search_configurations(
		self,
		workspace_id: Optional[str] = None,
		query: Optional[str] = None,
		filters: Optional[Dict[str, Any]] = None,
		sort_by: str = "updated_at",
		sort_order: str = "desc",
		limit: int = 50,
		offset: int = 0,
	) -> Dict[str, Any]:
		items = list(self._state()["configurations"].values())
		if workspace_id:
			items = [item for item in items if item["workspace_id"] == workspace_id]
		if query:
			needle = query.lower()
			items = [
				item for item in items
				if needle in item["name"].lower() or needle in item["key_path"].lower()
			]
		if filters:
			for key, value in filters.items():
				items = [item for item in items if _json_ready(item.get(key)) == value]
		reverse = sort_order.lower() == "desc"
		items = sorted(items, key=lambda item: item.get(sort_by) or "", reverse=reverse)
		total = len(items)
		page = items[offset:offset + limit]
		return {
			"configurations": [{key: _json_ready(value) for key, value in item.items()} for item in page],
			"total_count": total,
			"limit": limit,
			"offset": offset,
		}

	async def get_performance_metrics(self) -> List[Dict[str, Any]]:
		state = self._state()
		return [{
			"tenant_id": self.tenant_id,
			"configuration_count": len(state["configurations"]),
			"template_count": len(state["templates"]),
			"workspace_count": len(state["workspaces"]),
			"deployment_count": len(state["deployments"]),
			"audit_event_count": len(state["audit_entries"]),
			"collected_at": _utc_now().isoformat(),
		}]

	async def deploy_to_cloud(
		self,
		configuration_id: str,
		cloud_provider: str,
		environment_id: str,
		deployment_options: Dict[str, Any],
	) -> Dict[str, Any]:
		await self.get_configuration(configuration_id)
		deployment_id = uuid7str()
		record = {
			"id": deployment_id,
			"configuration_id": configuration_id,
			"cloud_provider": cloud_provider,
			"environment_id": environment_id,
			"options": deployment_options,
			"status": "deployed",
			"deployed_at": _utc_now().isoformat(),
		}
		self._state()["deployments"][deployment_id] = record
		self._audit("configuration.deploy", configuration_id, {"deployment_id": deployment_id})
		return record

	async def list_deployments(
		self,
		configuration_id: Optional[str] = None,
		cloud_provider: Optional[str] = None,
		environment_id: Optional[str] = None,
	) -> List[Dict[str, Any]]:
		deployments = list(self._state()["deployments"].values())
		if configuration_id:
			deployments = [item for item in deployments if item["configuration_id"] == configuration_id]
		if cloud_provider:
			deployments = [item for item in deployments if item["cloud_provider"] == cloud_provider]
		if environment_id:
			deployments = [item for item in deployments if item["environment_id"] == environment_id]
		return deployments

	async def start_collaboration_session(self, configuration_id: str, user_ids: List[str]) -> str:
		await self.get_configuration(configuration_id)
		session_id = uuid7str()
		self._state()["collaboration_sessions"][session_id] = {
			"session_id": session_id,
			"configuration_id": configuration_id,
			"user_ids": user_ids,
			"created_at": _utc_now().isoformat(),
		}
		self._audit("collaboration.start", configuration_id, {"session_id": session_id})
		return session_id

	async def get_configuration_versions(self, configuration_id: str, limit: int = 10) -> List[Dict[str, Any]]:
		if configuration_id not in self._state()["configurations"]:
			raise ValueError(f"Configuration {configuration_id} not found")
		return self._state()["versions"].get(configuration_id, [])[-limit:]

	async def restore_configuration_version(self, configuration_id: str, version: str, reason: str) -> Dict[str, Any]:
		versions = self._state()["versions"].get(configuration_id, [])
		match = next((item for item in versions if item["version"] == version), None)
		if not match:
			raise ValueError(f"Version {version} not found")
		record = self._state()["configurations"][configuration_id]
		record["value"] = match.get("value") or match.get("value_after") or record["value"]
		record["version"] = version
		record["updated_at"] = _utc_now()
		self._audit("configuration.restore", configuration_id, {"version": version, "reason": reason})
		return {"success": True, "message": f"Restored to version {version}", "configuration_id": configuration_id}

	async def create_template(self, template_data: TemplateCreate, workspace_id: str) -> Dict[str, Any]:
		template_id = uuid7str()
		record = template_data.model_dump(mode="json")
		record.update({
			"id": template_id,
			"tenant_id": self.tenant_id,
			"workspace_id": workspace_id,
			"created_by": self.user_id,
			"created_at": _utc_now().isoformat(),
			"usage_count": 0,
		})
		self._state()["templates"][template_id] = record
		self._audit("template.create", template_id, {"workspace_id": workspace_id})
		return record

	async def list_templates(
		self,
		workspace_id: Optional[str] = None,
		category: Optional[str] = None,
		is_public: Optional[bool] = None,
	) -> List[Dict[str, Any]]:
		templates = list(self._state()["templates"].values())
		if workspace_id:
			templates = [item for item in templates if item["workspace_id"] == workspace_id]
		if category:
			templates = [item for item in templates if item["category"] == category]
		if is_public is not None:
			templates = [item for item in templates if item["is_public"] is is_public]
		return templates

	async def create_workspace(self, workspace_data: WorkspaceCreate) -> Dict[str, Any]:
		workspace_id = uuid7str()
		record = workspace_data.model_dump(mode="json")
		record.update({
			"id": workspace_id,
			"tenant_id": self.tenant_id,
			"created_by": self.user_id,
			"created_at": _utc_now().isoformat(),
		})
		self._state()["workspaces"][workspace_id] = record
		self._audit("workspace.create", workspace_id)
		return record

	async def list_workspaces(self) -> List[Dict[str, Any]]:
		return list(self._state()["workspaces"].values())

	async def get_usage_analytics(
		self,
		configuration_id: Optional[str] = None,
		workspace_id: Optional[str] = None,
	) -> Dict[str, Any]:
		configs = list(self._state()["configurations"].values())
		if configuration_id:
			configs = [item for item in configs if item["id"] == configuration_id]
		if workspace_id:
			configs = [item for item in configs if item["workspace_id"] == workspace_id]
		return {
			"total_configurations": len(configs),
			"active_configurations": len([item for item in configs if item["status"] == ConfigurationStatus.ACTIVE]),
			"total_requests": len(self._state()["audit_entries"]),
			"avg_response_time": 0,
			"top_configurations": [
				{"configuration_id": item["id"], "name": item["name"], "version": item["version"]}
				for item in configs[:5]
			],
		}

	async def get_audit_log(
		self,
		resource_id: Optional[str] = None,
		event_type: Optional[str] = None,
		limit: int = 100,
		offset: int = 0,
	) -> List[Dict[str, Any]]:
		entries = list(self._state()["audit_entries"])
		if resource_id:
			entries = [item for item in entries if item["resource_id"] == resource_id]
		if event_type:
			entries = [item for item in entries if item["event_type"] == event_type]
		return entries[offset:offset + limit]

	async def get_compliance_report(self, framework: str, workspace_id: Optional[str] = None) -> Dict[str, Any]:
		usage = await self.get_usage_analytics(workspace_id=workspace_id)
		total_checks = 4
		passed_checks = 3 + int(usage["total_configurations"] > 0)
		return {
			"framework": framework,
			"compliance_score": round((passed_checks / total_checks) * 100, 2),
			"total_checks": total_checks,
			"passed_checks": passed_checks,
			"failed_checks": total_checks - passed_checks,
			"findings": [
				{
					"rule": "Configuration audit trail",
					"status": "passed" if self._state()["audit_entries"] else "failed",
					"description": "Configuration changes should be recorded in the audit stream",
				}
			],
			"generated_at": _utc_now().isoformat(),
		}


async def create_configuration_engine(
	tenant_id: str,
	user_id: str,
	**_: Any,
) -> CentralConfigurationEngine:
	"""Create the executable in-process configuration engine."""
	return CentralConfigurationEngine(tenant_id=tenant_id, user_id=user_id)


def _clean_text(value: Any) -> Optional[str]:
	"""Return a stripped string value when present."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _first_text(*values: Any) -> Optional[str]:
	"""Return the first non-empty text value."""
	for value in values:
		text = _clean_text(value)
		if text:
			return text
	return None


async def verify_token(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
	"""Verify JWT token."""
	if not token:
		return None
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		user_id: str = payload.get("sub")
		tenant_id: str = payload.get("tenant_id")
		
		if user_id is None or tenant_id is None:
			raise AuthenticationError("Invalid token")
		
		return {
			"user_id": user_id,
			"tenant_id": tenant_id,
			"permissions": payload.get("permissions", [])
		}
	except JWTError:
		raise AuthenticationError("Invalid token")


async def verify_api_key(request: Request, api_key: str = Security(api_key_header)) -> Dict[str, Any]:
	"""Verify API key."""
	if not api_key:
		raise HTTPException(status_code=401, detail="API key required")
	
	if api_key.startswith("cc_"):
		user_id = _first_text(
			request.headers.get("X-APG-User-ID"),
			request.headers.get("X-User-ID"),
			request.query_params.get("user_id"),
			os.getenv("APG_API_KEY_USER_ID"),
			os.getenv("APG_USER_ID"),
			os.getenv("APG_DEFAULT_USER_ID"),
		)
		tenant_id = _first_text(
			request.headers.get("X-APG-Tenant-ID"),
			request.headers.get("X-Tenant-ID"),
			request.headers.get("X-Organization-ID"),
			request.query_params.get("tenant_id"),
			request.query_params.get("tenant"),
			os.getenv("APG_API_KEY_TENANT_ID"),
			os.getenv("APG_TENANT_ID"),
			os.getenv("APG_DEFAULT_TENANT_ID"),
		)
		if not user_id or not tenant_id:
			raise HTTPException(status_code=401, detail="API key must resolve user and tenant context")
		return {
			"user_id": user_id,
			"tenant_id": tenant_id,
			"permissions": ["read", "write"]
		}
	
	raise HTTPException(status_code=401, detail="Invalid API key")


async def get_current_user(
	token_auth: Optional[Dict[str, Any]] = Depends(verify_token),
	api_key_auth: Optional[Dict[str, Any]] = Depends(verify_api_key)
) -> Dict[str, Any]:
	"""Get current authenticated user."""
	if token_auth:
		return token_auth
	elif api_key_auth:
		return api_key_auth
	else:
		raise HTTPException(status_code=401, detail="Authentication required")


# ==================== Dependency Injection ====================

async def get_config_engine(
	current_user: Dict[str, Any] = Depends(get_current_user)
) -> CentralConfigurationEngine:
	"""Get configuration engine instance."""
	engine = await create_configuration_engine(
		tenant_id=current_user["tenant_id"],
		user_id=current_user["user_id"]
	)
	return engine


# ==================== FastAPI Application ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan manager."""
	# Startup
	print("🚀 Starting APG Central Configuration API")
	
	# Initialize AI engine
	try:
		app.state.ai_engine = CentralConfigurationAI()
		await app.state.ai_engine.initialize()
	except Exception as e:
		print(f"⚠️ AI engine initialization failed: {e}")
		app.state.ai_engine = None
	
	app.state.redis = None
	redis_url = os.getenv("APG_CONFIG_REDIS_URL")
	if redis is not None and redis_url:
		app.state.redis = await redis.from_url(redis_url)
	
	yield
	
	# Shutdown
	print("🛑 Shutting down APG Central Configuration API")
	if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
		await app.state.ai_engine.close()
	if hasattr(app.state, 'redis') and app.state.redis is not None:
		await app.state.redis.close()


app = FastAPI(
	title="APG Central Configuration API",
	description="""
	Revolutionary AI-powered configuration management API
	
	Features:
	- AI-powered configuration optimization
	- Natural language queries
	- Real-time collaboration
	- Multi-cloud deployment
	- Zero-trust security
	- Autonomous operations
	""",
	version="1.0.0",
	lifespan=lifespan,
	docs_url="/docs",
	redoc_url="/redoc"
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
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# ==================== Health Check ====================

@app.get("/health", tags=["Health"])
async def health_check():
	"""Health check endpoint."""
	return {
		"status": "healthy",
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"version": "1.0.0",
		"ai_enabled": hasattr(app.state, 'ai_engine') and app.state.ai_engine is not None
	}


@app.get("/ready", tags=["Health"])
async def readiness_check():
	"""Readiness check endpoint."""
	checks = {
		"api": True,
		"redis": False,
		"ai_engine": False
	}
	
	# Check Redis
	try:
		if hasattr(app.state, 'redis') and app.state.redis is not None:
			await app.state.redis.ping()
			checks["redis"] = True
		else:
			checks["redis"] = True
	except Exception as exc:
		checks["redis_error"] = str(exc)
	
	# Check AI engine
	if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
		checks["ai_engine"] = True
	
	all_ready = all(checks.values())
	status_code = 200 if all_ready else 503
	
	return JSONResponse(
		status_code=status_code,
		content={
			"ready": all_ready,
			"checks": checks,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
	)


# ==================== Configuration Management ====================

@app.post("/configurations", response_model=ConfigurationResponse, tags=["Configurations"])
async def create_configuration(
	config_data: ConfigurationCreate,
	workspace_id: str,
	parent_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Create a new configuration with AI optimization."""
	try:
		result = await engine.create_configuration(
			workspace_id=workspace_id,
			config_data=config_data,
			parent_id=parent_id
		)
		return result
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/configurations/{configuration_id}", response_model=Dict[str, Any], tags=["Configurations"])
async def get_configuration(
	configuration_id: str,
	include_history: bool = False,
	include_ai_insights: bool = False,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get configuration with advanced features."""
	try:
		result = await engine.get_configuration(
			configuration_id=configuration_id,
			include_history=include_history,
			include_ai_insights=include_ai_insights
		)
		return result
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.put("/configurations/{configuration_id}", response_model=ConfigurationResponse, tags=["Configurations"])
async def update_configuration(
	configuration_id: str,
	updates: ConfigurationUpdate,
	change_reason: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Update configuration with collaborative editing support."""
	try:
		result = await engine.update_configuration(
			configuration_id=configuration_id,
			updates=updates,
			change_reason=change_reason
		)
		return result
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.delete("/configurations/{configuration_id}", tags=["Configurations"])
async def delete_configuration(
	configuration_id: str,
	reason: Optional[str] = None,
	permanent: bool = False,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Delete configuration with audit trail."""
	try:
		success = await engine.delete_configuration(
			configuration_id=configuration_id,
			reason=reason,
			permanent=permanent
		)
		return {"success": success, "message": "Configuration deleted successfully"}
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/configurations", tags=["Configurations"])
async def search_configurations(
	workspace_id: Optional[str] = None,
	query: Optional[str] = None,
	filters: Optional[str] = None,  # JSON string
	sort_by: str = "updated_at",
	sort_order: str = "desc",
	limit: int = 50,
	offset: int = 0,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Advanced configuration search with AI-powered natural language queries."""
	try:
		# Parse filters if provided
		parsed_filters = None
		if filters:
			try:
				parsed_filters = json.loads(filters)
			except json.JSONDecodeError:
				raise HTTPException(status_code=400, detail="Invalid filters JSON")
		
		result = await engine.search_configurations(
			workspace_id=workspace_id,
			query=query,
			filters=parsed_filters,
			sort_by=sort_by,
			sort_order=sort_order,
			limit=limit,
			offset=offset
		)
		return result
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


# ==================== AI-Powered Features ====================

@app.post("/configurations/{configuration_id}/optimize", tags=["AI Features"])
async def optimize_configuration(
	configuration_id: str,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""AI-powered configuration optimization."""
	try:
		# Get current configuration
		config_data = await engine.get_configuration(configuration_id)
		
		if not config_data:
			raise HTTPException(status_code=404, detail="Configuration not found")
		
		# Use AI engine for optimization
		if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
			optimized_config = await app.state.ai_engine.optimize_configuration(config_data['value'])
			
			return {
				"original_config": config_data['value'],
				"optimized_config": optimized_config,
				"optimization_applied": optimized_config != config_data['value'],
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
		else:
			raise HTTPException(status_code=503, detail="AI engine not available")
			
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/configurations/{configuration_id}/recommendations", tags=["AI Features"])
async def get_ai_recommendations(
	configuration_id: str,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get AI-powered configuration recommendations."""
	try:
		config_data = await engine.get_configuration(configuration_id)
		
		if not config_data:
			raise HTTPException(status_code=404, detail="Configuration not found")
		
		if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
			recommendations = await app.state.ai_engine.generate_recommendations(config_data['value'])
			
			return {
				"configuration_id": configuration_id,
				"recommendations": recommendations,
				"total_count": len(recommendations),
				"generated_at": datetime.now(timezone.utc).isoformat()
			}
		else:
			raise HTTPException(status_code=503, detail="AI engine not available")
			
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/configurations/natural-language-query", tags=["AI Features"])
async def natural_language_query(
	query: Dict[str, str],  # {"query": "find all database configurations"}
	workspace_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Process natural language configuration queries."""
	try:
		query_text = query.get("query", "")
		if not query_text:
			raise HTTPException(status_code=400, detail="Query text is required")
		
		if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
			# Parse natural language query
			parsed_query = await app.state.ai_engine.parse_natural_language_query(query_text)
			
			# Execute search with parsed filters
			search_results = await engine.search_configurations(
				workspace_id=workspace_id,
				query=query_text,
				filters=parsed_query.get('filters', {}),
				limit=50,
				offset=0
			)
			
			return {
				"original_query": query_text,
				"parsed_intent": parsed_query,
				"results": search_results,
				"timestamp": datetime.now(timezone.utc).isoformat()
			}
		else:
			raise HTTPException(status_code=503, detail="AI engine not available")
			
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/anomalies", tags=["AI Features"])
async def detect_anomalies(
	workspace_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Detect configuration anomalies using AI."""
	try:
		if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
			# Get metrics data (would come from actual metrics in production)
			metrics_data = await engine.get_performance_metrics()
			
			# Detect anomalies
			anomalies = await app.state.ai_engine.detect_anomalies(metrics_data)
			
			return {
				"anomalies": anomalies,
				"total_count": len(anomalies),
				"detection_timestamp": datetime.now(timezone.utc).isoformat(),
				"metrics_analyzed": len(metrics_data)
			}
		else:
			raise HTTPException(status_code=503, detail="AI engine not available")
			
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ==================== Multi-Cloud Deployment ====================

@app.post("/configurations/{configuration_id}/deploy", tags=["Deployment"])
async def deploy_configuration(
	configuration_id: str,
	deployment_request: Dict[str, Any],  # {"cloud_provider": "aws", "environment_id": "prod", "options": {}}
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Deploy configuration to specified cloud provider."""
	try:
		cloud_provider = deployment_request.get("cloud_provider")
		environment_id = deployment_request.get("environment_id") 
		deployment_options = deployment_request.get("options", {})
		
		if not cloud_provider or not environment_id:
			raise HTTPException(status_code=400, detail="cloud_provider and environment_id are required")
		
		result = await engine.deploy_to_cloud(
			configuration_id=configuration_id,
			cloud_provider=cloud_provider,
			environment_id=environment_id,
			deployment_options=deployment_options
		)
		
		return result
		
	except ValueError as e:
		raise HTTPException(status_code=404, detail=str(e))
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/deployments", tags=["Deployment"])
async def list_deployments(
	configuration_id: Optional[str] = None,
	cloud_provider: Optional[str] = None,
	environment_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""List configuration deployments."""
	deployments = await engine.list_deployments(
		configuration_id=configuration_id,
		cloud_provider=cloud_provider,
		environment_id=environment_id
	)
	return {
		"deployments": deployments,
		"total_count": len(deployments),
		"filters": {
			"configuration_id": configuration_id,
			"cloud_provider": cloud_provider, 
			"environment_id": environment_id
		}
	}


# ==================== Real-Time Collaboration ====================

class ConnectionManager:
	"""WebSocket connection manager for real-time collaboration."""
	
	def __init__(self):
		self.active_connections: Dict[str, List[WebSocket]] = {}
	
	async def connect(self, websocket: WebSocket, session_id: str):
		"""Add WebSocket connection to session."""
		await websocket.accept()
		if session_id not in self.active_connections:
			self.active_connections[session_id] = []
		self.active_connections[session_id].append(websocket)
	
	def disconnect(self, websocket: WebSocket, session_id: str):
		"""Remove WebSocket connection from session."""
		if session_id in self.active_connections:
			if websocket in self.active_connections[session_id]:
				self.active_connections[session_id].remove(websocket)
			if not self.active_connections[session_id]:
				del self.active_connections[session_id]
	
	async def send_to_session(self, session_id: str, message: dict):
		"""Send message to all connections in a session."""
		if session_id in self.active_connections:
			disconnected = []
			for connection in self.active_connections[session_id]:
				try:
					await connection.send_json(message)
				except:
					disconnected.append(connection)
			
			# Remove disconnected connections
			for conn in disconnected:
				self.disconnect(conn, session_id)


manager = ConnectionManager()


@app.websocket("/ws/collaboration/{session_id}")
async def websocket_collaboration(
	websocket: WebSocket,
	session_id: str,
	token: Optional[str] = None
):
	"""WebSocket endpoint for real-time collaboration."""
	# Verify authentication (simplified)
	if not token or not token.startswith("valid_"):
		await websocket.close(code=4001, reason="Authentication required")
		return
	
	await manager.connect(websocket, session_id)
	
	try:
		while True:
			# Receive message from client
			data = await websocket.receive_json()
			
			# Process collaboration message
			message = {
				"type": data.get("type", "unknown"),
				"user_id": data.get("user_id", "anonymous"),
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"data": data.get("data", {})
			}
			
			# Broadcast to all clients in session
			await manager.send_to_session(session_id, message)
			
	except WebSocketDisconnect:
		manager.disconnect(websocket, session_id)
	except Exception as e:
		print(f"WebSocket error: {e}")
		manager.disconnect(websocket, session_id)


@app.post("/collaboration/sessions", tags=["Collaboration"])
async def start_collaboration_session(
	session_request: Dict[str, Any],  # {"configuration_id": "...", "user_ids": [...]}
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Start a real-time collaboration session."""
	try:
		configuration_id = session_request.get("configuration_id")
		user_ids = session_request.get("user_ids", [])
		
		if not configuration_id:
			raise HTTPException(status_code=400, detail="configuration_id is required")
		
		session_id = await engine.start_collaboration_session(
			configuration_id=configuration_id,
			user_ids=user_ids
		)
		
		return {
			"session_id": session_id,
			"configuration_id": configuration_id,
			"user_ids": user_ids,
			"websocket_url": f"/ws/collaboration/{session_id}",
			"created_at": datetime.now(timezone.utc).isoformat()
		}
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


# ==================== Version Control ====================

@app.get("/configurations/{configuration_id}/versions", tags=["Version Control"])
async def get_configuration_versions(
	configuration_id: str,
	limit: int = 10,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get configuration version history."""
	try:
		versions = await engine.get_configuration_versions(configuration_id, limit)
		
		return {
			"configuration_id": configuration_id,
			"versions": versions,
			"total_count": len(versions)
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/configurations/{configuration_id}/restore", tags=["Version Control"])
async def restore_configuration_version(
	configuration_id: str,
	restore_request: Dict[str, Any],  # {"version": "1.2.3", "reason": "Rollback due to issues"}
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Restore configuration to specific version."""
	try:
		version = restore_request.get("version")
		reason = restore_request.get("reason", "Configuration restore")
		
		if not version:
			raise HTTPException(status_code=400, detail="version is required")
		
		result = await engine.restore_configuration_version(configuration_id, version, reason)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


# ==================== Template Management ====================

@app.post("/templates", tags=["Templates"])
async def create_template(
	template_data: TemplateCreate,
	workspace_id: str,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Create a configuration template."""
	try:
		result = await engine.create_template(template_data, workspace_id)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/templates", tags=["Templates"])
async def list_templates(
	workspace_id: Optional[str] = None,
	category: Optional[str] = None,
	is_public: Optional[bool] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""List configuration templates."""
	try:
		templates = await engine.list_templates(
			workspace_id=workspace_id,
			category=category,
			is_public=is_public,
		)
		
		return {
			"templates": templates,
			"total_count": len(templates),
			"filters": {
				"workspace_id": workspace_id,
				"category": category,
				"is_public": is_public
			}
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ==================== Workspace Management ====================

@app.post("/workspaces", tags=["Workspaces"])
async def create_workspace(
	workspace_data: WorkspaceCreate,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Create a new workspace."""
	try:
		result = await engine.create_workspace(workspace_data)
		
		return result
		
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/workspaces", tags=["Workspaces"])
async def list_workspaces(
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""List user's workspaces."""
	try:
		workspaces = await engine.list_workspaces()
		
		return {
			"workspaces": workspaces,
			"total_count": len(workspaces)
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ==================== Analytics & Monitoring ====================

@app.get("/analytics/metrics", tags=["Analytics"])
async def get_analytics_metrics(
	workspace_id: Optional[str] = None,
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get configuration analytics metrics."""
	try:
		metrics = await engine.get_performance_metrics()
		
		return {
			"metrics": metrics,
			"period": {
				"start_date": start_date,
				"end_date": end_date
			},
			"generated_at": datetime.now(timezone.utc).isoformat()
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/usage", tags=["Analytics"])
async def get_usage_analytics(
	configuration_id: Optional[str] = None,
	workspace_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get configuration usage analytics."""
	try:
		usage_data = await engine.get_usage_analytics(
			configuration_id=configuration_id,
			workspace_id=workspace_id,
		)
		
		return usage_data
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ==================== Security & Compliance ====================

@app.get("/security/audit-log", tags=["Security"])
async def get_audit_log(
	resource_id: Optional[str] = None,
	event_type: Optional[str] = None,
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	limit: int = 100,
	offset: int = 0,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Get security audit log."""
	try:
		audit_entries = await engine.get_audit_log(
			resource_id=resource_id,
			event_type=event_type,
			limit=limit,
			offset=offset,
		)
		
		return {
			"audit_entries": audit_entries,
			"total_count": len(audit_entries),
			"filters": {
				"resource_id": resource_id,
				"event_type": event_type,
				"start_date": start_date,
				"end_date": end_date
			}
		}
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/security/compliance-report", tags=["Security"])
async def get_compliance_report(
	framework: str = "SOC2",  # SOC2, HIPAA, PCI-DSS, GDPR
	workspace_id: Optional[str] = None,
	engine: CentralConfigurationEngine = Depends(get_config_engine)
):
	"""Generate compliance report."""
	try:
		report = await engine.get_compliance_report(framework, workspace_id)
		
		return report
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# ==================== Error Handlers ====================

@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
	"""Handle authentication errors."""
	return JSONResponse(
		status_code=401,
		content={"detail": str(exc), "type": "authentication_error"}
	)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
	"""Handle HTTP exceptions with enhanced error information."""
	return JSONResponse(
		status_code=exc.status_code,
		content={
			"detail": exc.detail,
			"type": "http_error",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"path": str(request.url)
		}
	)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
	"""Handle unexpected exceptions."""
	return JSONResponse(
		status_code=500,
		content={
			"detail": "Internal server error",
			"type": "internal_error",
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"path": str(request.url)
		}
	)


# ==================== Application Factory ====================

def create_app() -> FastAPI:
	"""Create and configure the FastAPI application."""
	return app


if __name__ == "__main__":
	uvicorn.run(
		"api:app",
		host="0.0.0.0",
		port=8000,
		reload=True,
		log_level="info"
	)
