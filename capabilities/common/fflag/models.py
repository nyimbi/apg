"""Feature Flags — Pydantic v2 models."""
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


class FeatureFlagCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	key: str
	name: str
	description: str = ""
	enabled: bool = False
	rollout_percentage: float = 0.0  # 0.0 – 100.0
	targeting_rules: list[dict[str, Any]] = Field(default_factory=list)
	variants: dict[str, Any] = Field(default_factory=dict)
	tags: list[str] = Field(default_factory=list)
	owner: str = ""

class FeatureFlagUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	description: str | None = None
	enabled: bool | None = None
	rollout_percentage: float | None = None
	targeting_rules: list[dict[str, Any]] | None = None
	variants: dict[str, Any] | None = None
	tags: list[str] | None = None

class FeatureFlagResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	key: str
	name: str
	description: str
	enabled: bool
	rollout_percentage: float
	targeting_rules: list[dict[str, Any]]
	variants: dict[str, Any]
	tags: list[str]
	owner: str
	created_at: str
	updated_at: str | None = None

class FeatureFlagListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[dict[str, Any]]
	total: int

class ExperimentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	flag_key: str
	name: str
	description: str = ""
	variants: list[dict[str, Any]] = Field(default_factory=list)  # [{"key": "control", "weight": 50}, ...]
	targeting_rule: dict[str, Any] = Field(default_factory=dict)
	owner: str = ""

class ExperimentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	flag_key: str
	name: str
	description: str
	variants: list[dict[str, Any]]
	targeting_rule: dict[str, Any]
	owner: str
	status: str = "draft"
	created_at: str

class EvaluationResult(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	flag_key: str
	enabled: bool
	variant: str | None = None
	reason: str = "default"
	targeting_matched: bool = False

class FlagAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	flag_key: str
	actor: str = "system"
	before: dict[str, Any] | None = None
	after: dict[str, Any] | None = None
	created_at: str

class FlagFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	enabled: bool | None = None
	tags: list[str] = Field(default_factory=list)
	owner: str | None = None
