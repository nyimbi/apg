"""Pydantic v2 models for grc_rsa capability.

© 2025 Datacraft  |  Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field

uuid7str = lambda: str(uuid7())


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


# ── Enums ─────────────────────────────────────────────────────────────────────

class RiskStatus(str, Enum):
	identified = "identified"
	assessed = "assessed"
	treated = "treated"
	monitored = "monitored"
	closed = "closed"
	accepted = "accepted"


class RiskRating(str, Enum):
	critical = "critical"
	high = "high"
	medium = "medium"
	low = "low"
	negligible = "negligible"


class ControlEffectiveness(str, Enum):
	effective = "effective"
	partially_effective = "partially_effective"
	ineffective = "ineffective"
	not_tested = "not_tested"


class TreatmentType(str, Enum):
	accept = "accept"
	mitigate = "mitigate"
	transfer = "transfer"
	avoid = "avoid"
	monitor = "monitor"


class KRIStatus(str, Enum):
	green = "green"
	amber = "amber"
	red = "red"


class Velocity(str, Enum):
	low = "low"
	medium = "medium"
	high = "high"
	very_high = "very_high"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Risk(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str
	risk_name: str
	category: str
	description: str
	owner_id: str
	status: RiskStatus = RiskStatus.identified
	inherent_score: float | None = None
	residual_score: float | None = None
	inherent_rating: RiskRating | None = None
	residual_rating: RiskRating | None = None
	controls: list[str] = Field(default_factory=list)
	treatment_plan_id: str | None = None
	kris: list[str] = Field(default_factory=list)
	updated_at: str = Field(default_factory=_now)


class RiskAssessment(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str | None = None
	created_at: str = Field(default_factory=_now)
	risk_id: str
	likelihood: int
	impact: int
	velocity: Velocity
	inherent_score: float
	inherent_rating: RiskRating
	assessor_id: str
	assessed_at: str = Field(default_factory=_now)


class Control(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	control_name: str
	control_type: str = "preventive"
	description: str = ""
	owner_id: str | None = None
	effectiveness_rating: ControlEffectiveness | None = None
	effectiveness_pct: float | None = None
	last_assessed_by: str | None = None
	last_assessed_at: str | None = None
	risk_ids: list[str] = Field(default_factory=list)


class KeyRiskIndicator(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str | None = None
	kri_name: str
	threshold_amber: float
	threshold_red: float
	current_value: float
	unit: str = ""
	period: str
	status: KRIStatus = KRIStatus.green
	recorded_at: str = Field(default_factory=_now)


# ── Request / Response ────────────────────────────────────────────────────────

class RiskRegisterRequest(_Base):
	entity_id: str
	risk_name: str
	category: str
	description: str
	owner_id: str
	risk_id: str | None = None


class RiskAssessmentRequest(_Base):
	likelihood_1_5: int
	impact_1_5: int
	velocity: Velocity
	assessor_id: str


class ControlAssessmentRequest(_Base):
	effectiveness_rating: ControlEffectiveness
	evidence: str
	assessed_by: str


class TreatmentPlanRequest(_Base):
	treatment_type: TreatmentType
	actions: list[dict[str, Any]]
	owner_id: str
	deadline: str


class TreatmentUpdateRequest(_Base):
	progress_pct: float
	notes: str
	updated_by: str


class KRIRequest(_Base):
	kri_name: str
	threshold_amber: float
	threshold_red: float
	current_value: float
	period: str
	entity_id: str | None = None
	unit: str = ""


class RiskAppetiteRequest(_Base):
	entity_id: str
	risk_category: str
	tolerance_level: str
