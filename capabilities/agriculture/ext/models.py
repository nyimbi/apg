"""Extension Services models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class AdvisoryChannel(str, Enum):
	SMS = "sms"
	VOICE = "voice"
	APP = "app"
	FIELD_VISIT = "field_visit"
	GROUP_MEETING = "group_meeting"
	RADIO = "radio"


class TrainingStatus(str, Enum):
	SCHEDULED = "scheduled"
	ONGOING = "ongoing"
	COMPLETED = "completed"
	CANCELLED = "cancelled"


class KnowledgeCategory(str, Enum):
	AGRONOMY = "agronomy"
	PEST_DISEASE = "pest_disease"
	SOIL_HEALTH = "soil_health"
	MARKET = "market"
	CLIMATE = "climate"
	FINANCE = "finance"
	NUTRITION = "nutrition"


class AdvisoryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farmer_id: str
	extension_worker_id: str
	channel: AdvisoryChannel
	topic: str
	message: str
	crop_type: str | None = None
	farm_parcel_id: str | None = None
	delivered_at: str | None = None
	follow_up_required: bool = False
	notes: str | None = None


class AdvisoryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farmer_id: str
	extension_worker_id: str
	channel: AdvisoryChannel
	topic: str
	message: str
	crop_type: str | None = None
	farm_parcel_id: str | None = None
	delivered_at: str | None = None
	follow_up_required: bool
	follow_up_done: bool = False
	notes: str | None = None
	created_at: str


class DemoPlotCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	farm_parcel_id: str
	extension_worker_id: str
	crop_type: str
	variety: str | None = None
	demonstration_topic: str
	start_date: str
	end_date: str | None = None
	target_farmers: list[str] = Field(default_factory=list)
	notes: str | None = None


class DemoPlotResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	farm_parcel_id: str
	extension_worker_id: str
	crop_type: str
	variety: str | None = None
	demonstration_topic: str
	start_date: str
	end_date: str | None = None
	target_farmers: list[str]
	farmer_visits: int = 0
	outcome: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class TrainingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	trainer_id: str
	topic: str
	scheduled_date: str
	location: str
	participant_ids: list[str] = Field(default_factory=list)
	max_participants: int = 50
	notes: str | None = None


class TrainingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	trainer_id: str
	topic: str
	scheduled_date: str
	location: str
	participant_ids: list[str]
	max_participants: int
	status: TrainingStatus = TrainingStatus.SCHEDULED
	actual_attendance: int = 0
	notes: str | None = None
	created_at: str
	updated_at: str


class KnowledgeArticleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	category: KnowledgeCategory
	content: str
	crop_types: list[str] = Field(default_factory=list)
	author_id: str | None = None
	tags: list[str] = Field(default_factory=list)
	language: str = "en"


class KnowledgeArticleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	category: KnowledgeCategory
	content: str
	crop_types: list[str]
	author_id: str | None = None
	tags: list[str]
	language: str
	views: int = 0
	created_at: str
	updated_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
