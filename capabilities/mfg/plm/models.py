"""Pydantic v2 models for APG Product Lifecycle Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str
except ImportError:
	from uuid6 import uuid7

	def uuid7str() -> str:
		return str(uuid7())


class MfPlmProduct(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	product_code: str
	product_name: str
	product_type: str = "standard"
	lifecycle_stage: str = "concept"
	revision: str = "A"
	description: str = ""
	product_family: str | None = None
	owner_id: str | None = None
	released_at: str | None = None
	discontinued_at: str | None = None
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)


class MfPlmStageGate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	product_id: str
	gate_number: int
	gate_name: str
	from_stage: str
	to_stage: str
	decision: str | None = None  # pass | conditional_pass | hold | kill
	reviewer_id: str | None = None
	reviewed_at: str | None = None
	conditions: str = ""
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
	metadata: dict[str, Any] = Field(default_factory=dict)
