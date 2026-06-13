"""Pydantic v2 models for APG ITSM Change Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:  # pragma: no cover
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


_MODEL_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


class ItChange(BaseModel):
	"""Change record — standard, normal, or emergency change ticket."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	description: str = ""
	change_type: str										# standard, normal, emergency
	status: str = "draft"
	risk_level: str = "medium"
	impact_level: str = "medium"
	# What is being changed
	affected_ci_ids: list[str] = Field(default_factory=list)
	affected_services: list[str] = Field(default_factory=list)
	# Who
	requester_id: str = "system"
	implementer_id: str | None = None
	team_id: str | None = None
	# Planning
	implementation_plan: str = ""
	rollback_plan: str = ""
	test_plan: str = ""
	rollback_ready: bool = False
	# Schedule
	scheduled_start: str | None = None
	scheduled_end: str | None = None
	actual_start: str | None = None
	actual_end: str | None = None
	maintenance_window_id: str | None = None
	# CAB
	cab_meeting_id: str | None = None
	cab_approved_at: str | None = None
	cab_approved_by: str | None = None
	cab_rejected_at: str | None = None
	cab_rejection_reason: str | None = None
	# Outcome
	implementation_notes: str = ""
	failed_reason: str | None = None
	rolled_back: bool = False
	rollback_at: str | None = None
	# PIR
	pir_id: str | None = None
	pir_completed: bool = False
	# Linkages
	incident_id: str | None = None						# incident that triggered change
	problem_id: str | None = None						# problem driving permanent fix
	# Temporal workflow
	temporal_workflow_id: str | None = None			# CAB approval workflow handle
	# Timestamps
	created_at: str = Field(default_factory=_now_iso)
	submitted_at: str | None = None
	closed_at: str | None = None
	version: int = 1
	tags: list[str] = Field(default_factory=list)
	custom_fields: dict[str, Any] = Field(default_factory=dict)


class ItCabApproval(BaseModel):
	"""CAB (Change Advisory Board) meeting and vote record."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	change_id: str
	meeting_date: str
	chair_id: str
	members: list[str] = Field(default_factory=list)	# member user IDs
	quorum_required: int = 2
	votes: list[dict[str, Any]] = Field(default_factory=list)  # [{member_id, outcome, notes, voted_at}]
	outcome: str | None = None							# approve, reject, defer
	outcome_notes: str = ""
	decided_at: str | None = None
	agenda_items: list[str] = Field(default_factory=list)
	meeting_minutes: str = ""
	created_at: str = Field(default_factory=_now_iso)


class ItChangeSchedule(BaseModel):
	"""Change schedule window — used for conflict detection and freeze windows."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	schedule_type: str									# maintenance_window, freeze_window, blackout
	start_datetime: str
	end_datetime: str
	recurrence_rule: str | None = None				# iCal RRULE for repeating windows
	affected_services: list[str] = Field(default_factory=list)
	affected_environments: list[str] = Field(default_factory=list)
	description: str = ""
	created_by: str = "system"
	created_at: str = Field(default_factory=_now_iso)
	is_active: bool = True


class ItChangeReview(BaseModel):
	"""Post-Implementation Review (PIR) for a change."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	change_id: str
	reviewer_id: str
	outcome: str										# success, partial_success, failure, inconclusive
	implementation_notes: str = ""
	objectives_met: bool = True
	incidents_caused: list[str] = Field(default_factory=list)	# incident IDs created by this change
	rollback_required: bool = False
	lessons_learned: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	process_improvements: list[str] = Field(default_factory=list)
	created_at: str = Field(default_factory=_now_iso)
	completed_at: str | None = None
