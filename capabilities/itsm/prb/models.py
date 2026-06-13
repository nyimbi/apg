"""Pydantic v2 models for APG ITSM Problem Management."""

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


class ItProblem(BaseModel):
	"""Problem record — a cause of one or more incidents."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	description: str = ""
	status: str = "new"									# new → under_investigation → root_cause_identified → known_error → resolved → closed
	priority: str = "P3"								# inherited from linked incidents
	category: str = "other"
	affected_service: str | None = None
	affected_ci_id: str | None = None
	# Incident linkages (1:many)
	linked_incident_ids: list[str] = Field(default_factory=list)
	# RCA
	rca_id: str | None = None
	root_cause_summary: str | None = None
	# Resolution
	resolution_notes: str = ""
	fix_type: str | None = None							# permanent, temporary, vendor_patch, …
	fix_applied_at: str | None = None
	change_ticket_id: str | None = None				# FK to itsm_chg triggering fix
	# Known error
	known_error_id: str | None = None
	# Timestamps
	created_at: str = Field(default_factory=_now_iso)
	resolved_at: str | None = None
	closed_at: str | None = None
	# Assignment
	owner_id: str | None = None
	team_id: str | None = None
	version: int = 1
	tags: list[str] = Field(default_factory=list)


class ItKnownError(BaseModel):
	"""Known Error Database (KEDB) entry — documented workaround for a recurring problem."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	problem_id: str
	title: str
	description: str
	workaround: str									# mandatory — the actionable workaround
	workaround_type: str							# manual, automated, bypass, …
	workaround_steps: list[str] = Field(default_factory=list)
	permanent_fix_available: bool = False
	permanent_fix_eta: str | None = None
	permanent_fix_description: str | None = None
	affected_services: list[str] = Field(default_factory=list)
	affected_ci_ids: list[str] = Field(default_factory=list)
	symptom_description: str = ""
	search_keywords: list[str] = Field(default_factory=list)
	review_date: str | None = None
	reviewed_by: str | None = None
	status: str = "active"							# active, retired
	created_at: str = Field(default_factory=_now_iso)
	created_by: str = "system"
	last_updated_at: str = Field(default_factory=_now_iso)
	usage_count: int = 0							# how many times applied to incidents


class ItRootCauseAnalysis(BaseModel):
	"""RCA record attached to a Problem — captures the investigation findings."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	problem_id: str
	method: str										# five_whys, fishbone, fault_tree, …
	findings: dict[str, Any] = Field(default_factory=dict)
	# Five-Whys structured capture
	why_chain: list[str] = Field(default_factory=list)
	# Fishbone categories (Ishikawa)
	fishbone_causes: dict[str, list[str]] = Field(default_factory=dict)
	root_cause: str = ""
	contributing_factors: list[str] = Field(default_factory=list)
	recommendations: list[str] = Field(default_factory=list)
	conducted_by: str = "system"
	reviewed_by: str | None = None
	status: str = "draft"							# draft, under_review, approved
	started_at: str = Field(default_factory=_now_iso)
	completed_at: str | None = None
	approved_at: str | None = None


class ItWorkaround(BaseModel):
	"""Instance of a workaround applied to a specific incident (from KEDB or ad-hoc)."""
	model_config = _MODEL_CFG

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_id: str
	problem_id: str | None = None
	known_error_id: str | None = None
	workaround_description: str
	workaround_type: str
	applied_by: str
	applied_at: str = Field(default_factory=_now_iso)
	effectiveness: str | None = None				# effective, partial, ineffective
	follow_up_required: bool = False
	notes: str = ""
