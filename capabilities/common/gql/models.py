"""GraphQL Federation Gateway — Pydantic v2 models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid4())


class SubgraphCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	url: str
	schema_sdl: str = ""
	health_check_path: str = "/health"
	timeout_ms: int = 5000
	enabled: bool = True

class SubgraphUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	url: str | None = None
	schema_sdl: str | None = None
	timeout_ms: int | None = None
	enabled: bool | None = None

class SubgraphResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	url: str
	schema_sdl: str
	health_check_path: str
	timeout_ms: int
	enabled: bool
	status: str = "active"
	created_at: str

class PersistedQueryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	query_id: str
	document: str
	name: str = ""
	tags: list[str] = Field(default_factory=list)

class PersistedQueryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	query_id: str
	document: str
	name: str
	tags: list[str]
	created_at: str

class QueryExecuteRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	query: str
	variables: dict[str, Any] = Field(default_factory=dict)
	operation_name: str | None = None
	extensions: dict[str, Any] = Field(default_factory=dict)

class QueryResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	data: dict[str, Any] | None = None
	errors: list[dict[str, Any]] | None = None
	extensions: dict[str, Any] = Field(default_factory=dict)

class GQLAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	operation: str = ""
	subgraph: str = ""
	duration_ms: float = 0.0
	payload: dict[str, Any] = Field(default_factory=dict)
	created_at: str

class GQLFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	enabled: bool | None = None
	subgraph: str | None = None
