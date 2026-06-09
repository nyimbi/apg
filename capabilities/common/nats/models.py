"""NATS JetStream capability — Pydantic v2 models."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetentionPolicy(str, Enum):
	limits = "limits"
	interest = "interest"
	work_queue = "workqueue"


class DeliverPolicy(str, Enum):
	all = "all"
	last = "last"
	new = "new"
	by_start_sequence = "by_start_sequence"
	by_start_time = "by_start_time"
	last_per_subject = "last_per_subject"


class CreateStreamRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	stream_name: str = Field(..., description="JetStream stream name")
	subjects: list[str] = Field(default_factory=list)
	retention: RetentionPolicy = RetentionPolicy.limits
	max_age_seconds: int = Field(default=0, ge=0)
	max_bytes: int = Field(default=-1)
	replicas: int = Field(default=1, ge=1, le=5)


class CreateConsumerRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	stream_name: str
	consumer_name: str
	filter_subject: str = ""
	deliver_policy: DeliverPolicy = DeliverPolicy.all
	ack_wait_seconds: int = Field(default=30, ge=1)
	max_deliver: int = Field(default=3, ge=1)


class PublishRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	capability_id: str
	event_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	actor_id: str = "system"


class PublishBatchRequest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	capability_id: str
	events: list[dict[str, Any]] = Field(default_factory=list, description="List of {event_type, payload, actor_id} dicts")


class StreamInfo(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	subjects: list[str] = Field(default_factory=list)
	messages: int = 0
	bytes_stored: int = Field(default=0, alias="bytes")

	model_config = ConfigDict(extra="ignore", validate_by_name=True)


class ConsumerInfo(BaseModel):
	model_config = ConfigDict(extra="ignore", validate_by_name=True)

	name: str
	filter_subject: str | None = None
	ack_pending: int = 0
	pending: int = 0


class HealthStatus(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str
	nats_url: str
	error: str | None = None
