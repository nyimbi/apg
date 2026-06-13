"""Pydantic v2 models for APG MLOps Pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	from uuid6 import uuid7  # type: ignore[import]
	def uuid7str() -> str:
		return str(uuid7())


class RunStatus(str, Enum):
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	KILLED = "killed"


class ModelStage(str, Enum):
	NONE = "none"
	STAGING = "staging"
	PRODUCTION = "production"
	ARCHIVED = "archived"


class DriftStatus(str, Enum):
	OK = "ok"
	WARNING = "warning"
	CRITICAL = "critical"


class MlrExperiment(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str = ""
	tags: list[str] = Field(default_factory=list)
	artifact_location: str = ""
	lifecycle_stage: str = "active"
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	created_by: str = "system"


class MlrRun(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	experiment_id: str
	run_name: str = ""
	status: RunStatus = RunStatus.RUNNING
	start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	end_time: datetime | None = None
	metrics: dict[str, float] = Field(default_factory=dict)
	params: dict[str, str] = Field(default_factory=dict)
	tags: dict[str, str] = Field(default_factory=dict)
	artifact_uri: str = ""
	source_version: str = ""
	entry_point: str = ""


class MlrFeatureView(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str = ""
	entities: list[str] = Field(default_factory=list)
	features: list[dict[str, Any]] = Field(default_factory=list)
	source_table: str = ""
	ttl_minutes: int = 60
	tags: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MlrFeatureVector(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	entity_id: str
	feature_view: str
	features: dict[str, Any] = Field(default_factory=dict)
	event_timestamp: datetime
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MlrRegisteredModel(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str = ""
	tags: dict[str, str] = Field(default_factory=dict)
	latest_versions: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MlrModelVersion(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	model_name: str
	version: int = 1
	stage: ModelStage = ModelStage.NONE
	source_run_id: str
	artifact_path: str = ""
	description: str = ""
	approved_by: str | None = None
	approval_notes: str = ""
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	promoted_at: datetime | None = None


class MlrAbTest(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	control_model_version_id: str
	treatment_model_version_id: str
	traffic_split_pct: float = 20.0  # % of traffic to treatment
	metrics_to_compare: list[str] = Field(default_factory=lambda: ["accuracy", "latency_p99"])
	status: str = "running"  # running | completed | stopped
	winner: str | None = None
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	ended_at: datetime | None = None


class MlrDriftReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	model_version_id: str
	feature_view_name: str
	status: DriftStatus = DriftStatus.OK
	psi_scores: dict[str, float] = Field(default_factory=dict)
	js_distances: dict[str, float] = Field(default_factory=dict)
	drifted_features: list[str] = Field(default_factory=list)
	retraining_recommended: bool = False
	checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	samples_checked: int = 0
