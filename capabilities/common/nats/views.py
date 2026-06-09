"""NATS JetStream capability — Flask-AppBuilder views and Pydantic request/response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
	CreateStreamRequest,
	CreateConsumerRequest,
	PublishRequest,
	PublishBatchRequest,
	StreamInfo,
	ConsumerInfo,
	HealthStatus,
)


# ── Additional view-layer schemas ─────────────────────────────────────────────

class SubjectRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	capability_id: str
	event_type: str = ">"


class SubjectResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	subject: str
	capability_id: str
	event_type: str


class StreamListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	streams: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0


class ConsumerListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	consumers: list[dict[str, Any]] = Field(default_factory=list)
	total: int = 0


class PublishResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	published: bool
	subject: str
	event_id: str


class SetRetentionRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	stream_name: str
	max_age_seconds: int = Field(default=0, ge=0)
	max_bytes: int = Field(default=-1)


class ConnectionInfoResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	nats_url: str
	tenant_id: str
	connected: bool


# ── Flask-AppBuilder view integration ─────────────────────────────────────────
# Views are registered via blueprint.py; this module exposes the schema
# classes needed for form rendering and API serialization.

__all__ = [
	"CreateStreamRequest",
	"CreateConsumerRequest",
	"PublishRequest",
	"PublishBatchRequest",
	"StreamInfo",
	"ConsumerInfo",
	"HealthStatus",
	"SubjectRequest",
	"SubjectResponse",
	"StreamListResponse",
	"ConsumerListResponse",
	"PublishResponse",
	"SetRetentionRequest",
	"ConnectionInfoResponse",
]
