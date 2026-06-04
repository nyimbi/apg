"""Domain events for Intelligence Fusion.

Factory functions return :class:`DomainEvent` instances that are serialised
and stored/emitted by the service via ``_emit_event()``.  Every state change
in the fusion lifecycle emits exactly one event.

Subscribe to stream ``apg.intel.fusion.lifecycle`` for downstream integration.

© 2025 Datacraft — Nyimbi Odero
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
	"""Base class for all Intelligence Fusion domain events."""

	event_type: str
	tenant_id: str
	actor_id: str
	resource_id: str
	resource_type: str
	timestamp: datetime = field(default_factory=datetime.utcnow)
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"event_type": self.event_type,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"resource_id": self.resource_id,
			"resource_type": self.resource_type,
			"timestamp": self.timestamp.isoformat(),
			"payload": self.payload,
			"capability_id": "intel_fusion",
			"stream": "apg.intel.fusion.lifecycle",
		}


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceItem events
# ─────────────────────────────────────────────────────────────────────────────

def intel_item_created(
	tenant_id: str,
	actor_id: str,
	item_id: str,
	source_type: str,
	workspace_id: str,
) -> DomainEvent:
	"""Emitted when a raw intelligence item is ingested."""
	return DomainEvent(
		event_type="intel_item.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=item_id,
		resource_type="IntelligenceItem",
		payload={"source_type": source_type, "workspace_id": workspace_id},
	)


def intel_item_status_changed(
	tenant_id: str,
	actor_id: str,
	item_id: str,
	old_status: str,
	new_status: str,
) -> DomainEvent:
	"""Emitted when an item transitions to a new status."""
	return DomainEvent(
		event_type="intel_item.status_changed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=item_id,
		resource_type="IntelligenceItem",
		payload={"old_status": old_status, "new_status": new_status},
	)


# ─────────────────────────────────────────────────────────────────────────────
# FusionWorkspace events
# ─────────────────────────────────────────────────────────────────────────────

def workspace_created(
	tenant_id: str,
	actor_id: str,
	workspace_id: str,
	workspace_type: str,
) -> DomainEvent:
	"""Emitted when a new fusion workspace is created."""
	return DomainEvent(
		event_type="workspace.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=workspace_id,
		resource_type="FusionWorkspace",
		payload={"workspace_type": workspace_type},
	)


def workspace_status_changed(
	tenant_id: str,
	actor_id: str,
	workspace_id: str,
	new_status: str,
) -> DomainEvent:
	"""Emitted when a workspace changes status (suspended, closed, etc.)."""
	return DomainEvent(
		event_type="workspace.status_changed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=workspace_id,
		resource_type="FusionWorkspace",
		payload={"new_status": new_status},
	)


# ─────────────────────────────────────────────────────────────────────────────
# CorrelationSet events
# ─────────────────────────────────────────────────────────────────────────────

def correlation_created(
	tenant_id: str,
	actor_id: str,
	correlation_id: str,
	correlation_type: str,
	item_count: int,
) -> DomainEvent:
	"""Emitted when a correlation set is created."""
	return DomainEvent(
		event_type="correlation.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=correlation_id,
		resource_type="CorrelationSet",
		payload={"correlation_type": correlation_type, "item_count": item_count},
	)


def correlation_status_changed(
	tenant_id: str,
	actor_id: str,
	correlation_id: str,
	new_status: str,
) -> DomainEvent:
	"""Emitted when a correlation set status changes (confirmed, disputed, closed)."""
	return DomainEvent(
		event_type="correlation.status_changed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=correlation_id,
		resource_type="CorrelationSet",
		payload={"new_status": new_status},
	)


# ─────────────────────────────────────────────────────────────────────────────
# AssessmentPicture events
# ─────────────────────────────────────────────────────────────────────────────

def assessment_created(
	tenant_id: str,
	actor_id: str,
	assessment_id: str,
	risk_level: str,
) -> DomainEvent:
	"""Emitted when a synthesised assessment picture is created."""
	return DomainEvent(
		event_type="assessment.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=assessment_id,
		resource_type="AssessmentPicture",
		payload={"risk_level": risk_level},
	)


def assessment_approved(
	tenant_id: str,
	approver_id: str,
	assessment_id: str,
) -> DomainEvent:
	"""Emitted when a senior analyst approves an assessment picture."""
	return DomainEvent(
		event_type="assessment.approved",
		tenant_id=tenant_id,
		actor_id=approver_id,
		resource_id=assessment_id,
		resource_type="AssessmentPicture",
		payload={"approver_id": approver_id},
	)


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceProduct events
# ─────────────────────────────────────────────────────────────────────────────

def product_created(
	tenant_id: str,
	actor_id: str,
	product_id: str,
	product_type: str,
	tlp: str,
) -> DomainEvent:
	"""Emitted when a finished intelligence product is created."""
	return DomainEvent(
		event_type="product.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=product_id,
		resource_type="IntelligenceProduct",
		payload={"product_type": product_type, "tlp": tlp},
	)


def product_status_changed(
	tenant_id: str,
	actor_id: str,
	product_id: str,
	new_status: str,
) -> DomainEvent:
	"""Emitted on every product status transition (review, approved, released, recalled)."""
	return DomainEvent(
		event_type="product.status_changed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=product_id,
		resource_type="IntelligenceProduct",
		payload={"new_status": new_status},
	)


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalJudgement events
# ─────────────────────────────────────────────────────────────────────────────

def judgement_created(
	tenant_id: str,
	actor_id: str,
	judgement_id: str,
	judgement_type: str,
	confidence_level: str,
) -> DomainEvent:
	"""Emitted when a calibrated analytical judgement is recorded."""
	return DomainEvent(
		event_type="judgement.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=judgement_id,
		resource_type="AnalyticalJudgement",
		payload={"judgement_type": judgement_type, "confidence_level": confidence_level},
	)


def judgement_challenged(
	tenant_id: str,
	actor_id: str,
	judgement_id: str,
	challenger_id: str,
) -> DomainEvent:
	"""Emitted when a red-team or devil's advocate challenge is registered."""
	return DomainEvent(
		event_type="judgement.challenged",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=judgement_id,
		resource_type="AnalyticalJudgement",
		payload={"challenger_id": challenger_id},
	)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence events
# ─────────────────────────────────────────────────────────────────────────────

def evidence_created(
	tenant_id: str,
	actor_id: str,
	evidence_id: str,
	evidence_type: str,
) -> DomainEvent:
	"""Emitted when a provenance-tracked evidence item is recorded."""
	return DomainEvent(
		event_type="evidence.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=evidence_id,
		resource_type="Evidence",
		payload={"evidence_type": evidence_type},
	)


def evidence_status_changed(
	tenant_id: str,
	actor_id: str,
	evidence_id: str,
	new_status: str,
) -> DomainEvent:
	"""Emitted when evidence status changes (verified, challenged, discredited)."""
	return DomainEvent(
		event_type="evidence.status_changed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=evidence_id,
		resource_type="Evidence",
		payload={"new_status": new_status},
	)


# ─────────────────────────────────────────────────────────────────────────────
# HypothesisTest events
# ─────────────────────────────────────────────────────────────────────────────

def hypothesis_created(
	tenant_id: str,
	actor_id: str,
	hypothesis_id: str,
	sat_method: str,
) -> DomainEvent:
	"""Emitted when a structured hypothesis test is opened."""
	return DomainEvent(
		event_type="hypothesis.created",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=hypothesis_id,
		resource_type="HypothesisTest",
		payload={"sat_method": sat_method},
	)


def hypothesis_concluded(
	tenant_id: str,
	actor_id: str,
	hypothesis_id: str,
	conclusion_status: str,
	final_confidence: float,
) -> DomainEvent:
	"""Emitted when a hypothesis test reaches a conclusion (supported/refuted/inconclusive)."""
	return DomainEvent(
		event_type="hypothesis.concluded",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=hypothesis_id,
		resource_type="HypothesisTest",
		payload={"conclusion_status": conclusion_status, "final_confidence": final_confidence},
	)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion-level events
# ─────────────────────────────────────────────────────────────────────────────

def fusion_completed(
	tenant_id: str,
	actor_id: str,
	workspace_id: str,
	item_count: int,
	source_types: list[str],
	quality_score: float,
) -> DomainEvent:
	"""Emitted when a full intelligence fusion run completes on a workspace."""
	return DomainEvent(
		event_type="fusion.completed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=workspace_id,
		resource_type="FusionWorkspace",
		payload={
			"item_count": item_count,
			"source_types": source_types,
			"quality_score": quality_score,
		},
	)


def ach_completed(
	tenant_id: str,
	actor_id: str,
	workspace_id: str,
	leading_hypothesis: str,
	confidence: float,
) -> DomainEvent:
	"""Emitted when an ACH analysis completes and a leading hypothesis is identified."""
	return DomainEvent(
		event_type="ach.completed",
		tenant_id=tenant_id,
		actor_id=actor_id,
		resource_id=workspace_id,
		resource_type="FusionWorkspace",
		payload={"leading_hypothesis": leading_hypothesis, "confidence": confidence},
	)
