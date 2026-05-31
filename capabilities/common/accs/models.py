"""Domain models for APG Accessibility Services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AccessibilityStandard:
	"""Tenant-scoped accessibility standard profile."""

	id: str
	tenant_id: str
	name: str = "WCAG"
	version: str = "2.2"
	level: str = "AA"
	criteria: tuple[str, ...] = (
		"perceivable",
		"operable",
		"understandable",
		"robust",
	)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"version": self.version,
			"level": self.level,
			"criteria": list(self.criteria),
		}


@dataclass(frozen=True)
class AccessibilityTarget:
	"""UI, content, or media surface that can be audited."""

	id: str
	tenant_id: str
	surface: str
	route: str
	owner: str
	published_ui: bool = False
	contrast_ratio: float = 4.5
	semantic_labels_present: bool = True
	keyboard_navigation_present: bool = True
	media_content_present: bool = False
	captions_available: bool = True

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"surface": self.surface,
			"route": self.route,
			"owner": self.owner,
			"published_ui": self.published_ui,
			"contrast_ratio": self.contrast_ratio,
			"semantic_labels_present": self.semantic_labels_present,
			"keyboard_navigation_present": self.keyboard_navigation_present,
			"media_content_present": self.media_content_present,
			"captions_available": self.captions_available,
		}


@dataclass(frozen=True)
class AccessibilityFinding:
	"""Deterministic accessibility finding produced by an audit or manual review."""

	id: str
	tenant_id: str
	target_id: str
	rule: str
	severity: str
	description: str
	remediation_owner: str
	status: str = "open"
	review_required: bool = False
	review_recorded: bool = False
	resolution: str | None = None
	evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"target_id": self.target_id,
			"rule": self.rule,
			"severity": self.severity,
			"description": self.description,
			"remediation_owner": self.remediation_owner,
			"status": self.status,
			"review_required": self.review_required,
			"review_recorded": self.review_recorded,
			"resolution": self.resolution,
			"evidence": dict(self.evidence),
		}


@dataclass(frozen=True)
class RemediationTask:
	"""Tracked remediation work item for an accessibility finding."""

	id: str
	tenant_id: str
	finding_id: str
	owner: str
	status: str = "open"
	due_date: str | None = None
	review_recorded: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"finding_id": self.finding_id,
			"owner": self.owner,
			"status": self.status,
			"due_date": self.due_date,
			"review_recorded": self.review_recorded,
		}


@dataclass(frozen=True)
class AccessibilityAudit:
	"""Completed deterministic audit run for one or more targets."""

	id: str
	tenant_id: str
	standard_id: str
	target_ids: tuple[str, ...]
	finding_ids: tuple[str, ...]
	status: str = "completed"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"standard_id": self.standard_id,
			"target_ids": list(self.target_ids),
			"finding_ids": list(self.finding_ids),
			"status": self.status,
		}


@dataclass(frozen=True)
class AccessibilityReview:
	"""Formal review decision for a high-risk accessibility finding."""

	id: str
	tenant_id: str
	finding_id: str
	reviewer: str
	decision: str
	notes: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"finding_id": self.finding_id,
			"reviewer": self.reviewer,
			"decision": self.decision,
			"notes": self.notes,
		}


@dataclass(frozen=True)
class AccessibilityException:
	"""Approved temporary exception for an unresolved accessibility finding."""

	id: str
	tenant_id: str
	finding_id: str
	approver: str
	reason: str
	expires_on: str
	compensating_controls: tuple[str, ...]
	status: str = "approved"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"finding_id": self.finding_id,
			"approver": self.approver,
			"reason": self.reason,
			"expires_on": self.expires_on,
			"compensating_controls": list(self.compensating_controls),
			"status": self.status,
		}


@dataclass(frozen=True)
class AccessibilityAuditEvent:
	"""Tenant-scoped evidence event for ACCS lifecycle changes."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"evidence": dict(self.evidence),
		}


@dataclass(frozen=True)
class AccessibilityAgent:
	"""Governed AI accessibility agent registration."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool
	contribution_disclosed: bool
	policy_ref: str | None = None
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
		}
