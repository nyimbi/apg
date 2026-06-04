"""Pydantic v2 models for grc_icm capability.

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

class IncidentStatus(str, Enum):
	new = "new"
	triaged = "triaged"
	in_investigation = "in_investigation"
	pending_closure = "pending_closure"
	closed = "closed"
	false_positive = "false_positive"


class IncidentSeverity(str, Enum):
	low = "low"
	medium = "medium"
	high = "high"
	critical = "critical"


class IncidentType(str, Enum):
	security_breach = "security_breach"
	data_loss = "data_loss"
	system_outage = "system_outage"
	fraud = "fraud"
	compliance_violation = "compliance_violation"
	operational = "operational"
	third_party = "third_party"
	physical = "physical"


class ActionType(str, Enum):
	corrective = "corrective"
	preventive = "preventive"
	systemic = "systemic"
	immediate = "immediate"


class ActionStatus(str, Enum):
	open = "open"
	closed = "closed"
	overdue = "overdue"


class TestType(str, Enum):
	design = "design"
	operating_effectiveness = "operating_effectiveness"
	walkthrough = "walkthrough"
	inquiry = "inquiry"


class TestResult(str, Enum):
	pass_ = "pass"
	fail = "fail"
	partial = "partial"


class DeficiencyType(str, Enum):
	design_gap = "design_gap"
	operating_ineffectiveness = "operating_ineffectiveness"
	absent_control = "absent_control"


class DeficiencySeverity(str, Enum):
	observation = "observation"
	significant = "significant"
	material_weakness = "material_weakness"


# ── Core models ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class Incident(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str
	title: str
	incident_type: IncidentType
	description: str
	severity: IncidentSeverity
	affected_systems: list[str] = Field(default_factory=list)
	reported_by: str
	owner_id: str
	detection_time: str
	status: IncidentStatus = IncidentStatus.new
	timeline: list[dict[str, Any]] = Field(default_factory=list)
	root_cause: str | None = None
	lessons_learned: str | None = None
	regulatory_breach: bool = False
	updated_at: str = Field(default_factory=_now)


class CorrectiveAction(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	incident_id: str
	action_type: ActionType
	description: str
	owner_id: str
	deadline: str
	progress_pct: float = 0.0
	status: ActionStatus = ActionStatus.open
	updated_at: str = Field(default_factory=_now)


class ComplianceTest(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	entity_id: str
	control_id: str
	test_type: TestType
	test_date: str
	result: str
	tester_id: str
	status: str = "completed"


class ComplianceDeficiency(_Base):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	created_at: str = Field(default_factory=_now)
	control_id: str
	deficiency_type: DeficiencyType
	severity: DeficiencySeverity
	identified_by: str
	status: str = "open"
	remediation_plan_id: str | None = None


# ── Request / Response ────────────────────────────────────────────────────────

class ReportIncidentRequest(_Base):
	entity_id: str
	incident_type: IncidentType
	description: str
	severity: IncidentSeverity
	affected_systems: list[str]
	reported_by: str
	title: str | None = None
	detection_time: str | None = None


class TriageRequest(_Base):
	incident_commander_id: str
	priority: str
	initial_response: str


class InvestigationRequest(_Base):
	findings: str
	evidence: list[dict[str, Any]]
	investigator_id: str


class RCARequest(_Base):
	rca_method: str
	root_causes: list[str]
	contributing_factors: list[str]


class CorrectiveActionRequest(_Base):
	action_type: ActionType
	description: str
	owner_id: str
	deadline: str


class CloseIncidentRequest(_Base):
	resolution: str
	lessons_learned: str
	closed_by: str


class ComplianceTestRequest(_Base):
	entity_id: str
	control_id: str
	test_type: TestType
	test_date: str
	result: str
	tester_id: str


class DeficiencyRequest(_Base):
	control_id: str
	deficiency_type: DeficiencyType
	severity: DeficiencySeverity
	identified_by: str


class RegulatoryNotificationRequest(_Base):
	regulator: str
	notification_type: str
	deadline: str
