"""Pydantic v2 models for APG ITSM Incident Management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ItIncident(BaseModel):
	"""ITIL v4 Incident record — full lifecycle from new to closed."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	# Core identification
	title: str
	description: str = ""
	category: str											# hardware, software, network, …
	subcategory: str | None = None
	# Priority matrix
	priority: str											# P1–P4
	impact: str = "medium"									# low, medium, high, critical
	urgency: str = "medium"								# low, medium, high, critical
	# Lifecycle
	status: str = "new"									# new → acknowledged → in_progress → resolved → closed
	# Assignment
	assigned_to: str | None = None
	assigned_team: str | None = None
	reported_by: str = "system"
	# CI context (from itsm_cmdb)
	affected_ci_id: str | None = None
	affected_ci_name: str | None = None
	affected_service: str | None = None
	# SLA tracking
	sla_due_at: str | None = None						# ISO-8601 deadline
	sla_breached: bool = False
	sla_breached_at: str | None = None
	response_sla_minutes: int | None = None			# time-to-acknowledge target
	resolve_sla_minutes: int | None = None				# time-to-resolve target
	# Resolution
	resolution_code: str | None = None
	resolution_notes: str = ""
	workaround: str | None = None
	root_cause_summary: str | None = None
	# Major incident
	is_major: bool = False
	major_declared_at: str | None = None
	incident_commander_id: str | None = None
	# Problem linkage
	problem_id: str | None = None						# FK to itsm_prb.ItProblem
	known_error_id: str | None = None					# FK to itsm_prb.ItKnownError
	# Change linkage
	caused_by_change_id: str | None = None				# FK to itsm_chg.ItChange
	# Alert linkage (from intel_alerts)
	source_alert_id: str | None = None
	# Timestamps
	created_at: str = Field(default_factory=_now_iso)
	acknowledged_at: str | None = None
	in_progress_at: str | None = None
	resolved_at: str | None = None
	closed_at: str | None = None
	# Metadata
	tags: list[str] = Field(default_factory=list)
	custom_fields: dict[str, Any] = Field(default_factory=dict)
	version: int = 1

	@field_validator("priority")
	@classmethod
	def _validate_priority(cls, v: str) -> str:
		assert v in ("P1", "P2", "P3", "P4"), f"priority must be P1–P4, got {v!r}"
		return v


class ItIncidentUpdate(BaseModel):
	"""Timestamped update (note, workaround, status change, escalation) on an incident."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_id: str
	update_type: str										# note, workaround, status_change, escalation, resolution
	author_id: str
	content: str
	previous_status: str | None = None
	new_status: str | None = None
	internal_only: bool = False
	created_at: str = Field(default_factory=_now_iso)
	attachments: list[str] = Field(default_factory=list)	# attachment references


class ItIncidentSLA(BaseModel):
	"""SLA snapshot for an incident — one record per SLA evaluation."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_id: str
	priority: str
	sla_type: str											# response, resolve
	target_minutes: int
	started_at: str
	due_at: str
	paused_duration_minutes: float = 0.0				# maintenance window exclusions
	elapsed_minutes: float = 0.0
	remaining_minutes: float = 0.0
	breached: bool = False
	breached_at: str | None = None
	met_at: str | None = None
	evaluated_at: str = Field(default_factory=_now_iso)

	@field_validator("target_minutes", "elapsed_minutes", mode="before")
	@classmethod
	def _non_negative(cls, v: float) -> float:
		assert float(v) >= 0, "minutes must be non-negative"
		return v


class ItMajorIncident(BaseModel):
	"""Major Incident declaration record — bridges multiple affected CIs/services."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_id: str										# primary incident
	linked_incident_ids: list[str] = Field(default_factory=list)
	incident_commander_id: str
	declared_by: str
	declared_at: str = Field(default_factory=_now_iso)
	communication_bridge_url: str | None = None
	status_page_url: str | None = None
	bridge_open: bool = True
	affected_services: list[str] = Field(default_factory=list)
	customer_impact_statement: str = ""
	internal_status_updates: list[dict[str, Any]] = Field(default_factory=list)
	external_communications: list[dict[str, Any]] = Field(default_factory=list)
	resolved_at: str | None = None
	pir_required: bool = True
	pir_due_date: str | None = None
	pir_completed_at: str | None = None
	pir_summary: str | None = None
	lessons_learned: list[str] = Field(default_factory=list)
	created_at: str = Field(default_factory=_now_iso)
