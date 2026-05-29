"""Dependency-light AUDL evidence lifecycle runtime for package composition."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	AuditExportRequest,
	AuditGovernanceEvent,
	AuditInvestigationRecord,
	AuditLegalHoldRecord,
	AuditLifecycleEvent,
	AuditPurgeRequest,
)


class AudlService:
	"""Tenant-scoped audit governance facade for generated APG applications."""

	def __init__(self) -> None:
		self._events: dict[tuple[str, str], AuditLifecycleEvent] = {}
		self._holds: dict[tuple[str, str], AuditLegalHoldRecord] = {}
		self._exports: dict[tuple[str, str], AuditExportRequest] = {}
		self._purges: dict[tuple[str, str], AuditPurgeRequest] = {}
		self._investigations: dict[tuple[str, str], AuditInvestigationRecord] = {}
		self._governance_events: list[AuditGovernanceEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def append_event(
		self,
		event_id: str,
		tenant_id: str,
		actor: str,
		action: str,
		resource_type: str,
		resource_id: str,
		severity: str = "info",
		contains_pii: bool = False,
		immutable: bool = True,
		checksum: str | None = None,
		details: dict[str, Any] | None = None,
		escalation_configured: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, event_id) in self._events:
			raise ValueError(f"audit event already exists for tenant: {event_id}")
		if not actor:
			raise ValueError("audit event actor is required")
		if not action:
			raise ValueError("audit event action is required")
		if not resource_type or not resource_id:
			raise ValueError("audit event resource type and ID are required")
		expected_checksum = self.calculate_checksum(
			event_id=event_id,
			tenant_id=tenant_id,
			actor=actor,
			action=action,
			resource_type=resource_type,
			resource_id=resource_id,
			severity=severity,
			contains_pii=contains_pii,
			immutable=immutable,
			details=dict(details or {}),
		)
		checksum_verified = checksum in {None, expected_checksum}
		result = self.evaluate({
			"tenant_id_missing": not bool(tenant_id),
			"immutable_storage": immutable,
			"checksum_verified": checksum_verified,
			"event_severity": severity,
			"escalation_configured": escalation_configured,
		})
		_raise_if_blocked(result)
		record = AuditLifecycleEvent(
			id=event_id,
			tenant_id=tenant_id,
			actor=actor,
			action=action,
			resource_type=resource_type,
			resource_id=resource_id,
			severity=severity,
			contains_pii=contains_pii,
			immutable=immutable,
			checksum=expected_checksum,
			details=dict(details or {}),
		)
		self._events[self._tenant_key(tenant_id, event_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="audit_event_appended",
			subject_id=event_id,
			message=f"Appended audit event {event_id}.",
			evidence={"severity": severity, "resource_type": resource_type, "contains_pii": contains_pii},
		)
		return record.model_dump(mode="json")

	def apply_legal_hold(
		self,
		hold_id: str,
		tenant_id: str,
		scope: dict[str, Any],
		reason: str,
		approver: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, hold_id) in self._holds:
			raise ValueError(f"legal hold already exists for tenant: {hold_id}")
		if not scope:
			raise ValueError("legal hold scope is required")
		if not reason:
			raise ValueError("legal hold reason is required")
		if not approver:
			raise ValueError("legal hold approver is required")
		hold = AuditLegalHoldRecord(
			id=hold_id,
			tenant_id=tenant_id,
			scope=dict(scope),
			reason=reason,
			approver=approver,
		)
		self._holds[self._tenant_key(tenant_id, hold_id)] = hold
		self._record_governance(
			tenant_id=tenant_id,
			event_type="legal_hold_applied",
			subject_id=hold_id,
			message=f"Applied legal hold {hold_id}.",
			evidence={"scope": hold.scope, "approver": approver},
		)
		return hold.model_dump(mode="json")

	def release_legal_hold(
		self,
		hold_id: str,
		tenant_id: str,
		released_by: str,
		release_evidence: str,
	) -> dict[str, Any]:
		hold = self._get_hold(tenant_id, hold_id)
		if not released_by:
			raise ValueError("legal hold release actor is required")
		if not release_evidence:
			raise ValueError("legal hold release evidence is required")
		released = AuditLegalHoldRecord(
			id=hold.id,
			tenant_id=hold.tenant_id,
			scope=dict(hold.scope),
			reason=hold.reason,
			approver=hold.approver,
			status="released",
			applied_at=hold.applied_at,
			released_by=released_by,
			release_evidence=release_evidence,
			released_at=datetime.utcnow(),
		)
		self._holds[self._tenant_key(tenant_id, hold_id)] = released
		self._record_governance(
			tenant_id=tenant_id,
			event_type="legal_hold_released",
			subject_id=hold_id,
			message=f"Released legal hold {hold_id}.",
			evidence={"released_by": released_by},
		)
		return released.model_dump(mode="json")

	def request_export(
		self,
		export_id: str,
		tenant_id: str,
		requested_by: str,
		query: dict[str, Any],
		contains_pii: bool,
		masking_enabled: bool,
		reason: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, export_id) in self._exports:
			raise ValueError(f"export request already exists for tenant: {export_id}")
		if not requested_by:
			raise ValueError("export requester is required")
		if not reason:
			raise ValueError("export reason is required")
		result = self.evaluate({
			"tenant_id_missing": not bool(tenant_id),
			"requested_operation": "export",
			"contains_pii": contains_pii,
			"masking_enabled": masking_enabled,
		})
		_raise_if_blocked(result)
		record = AuditExportRequest(
			id=export_id,
			tenant_id=tenant_id,
			query=dict(query),
			requested_by=requested_by,
			contains_pii=contains_pii,
			masking_enabled=masking_enabled,
			reason=reason,
		)
		self._exports[self._tenant_key(tenant_id, export_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="export_requested",
			subject_id=export_id,
			message=f"Requested audit export {export_id}.",
			evidence={"contains_pii": contains_pii, "masking_enabled": masking_enabled},
		)
		return record.model_dump(mode="json")

	def decide_export(
		self,
		export_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		record = self._exports.get(self._tenant_key(tenant_id, export_id))
		if record is None:
			raise KeyError(f"unknown export request for tenant: {export_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("export decision must be approved or rejected")
		if not reviewer:
			raise ValueError("export reviewer is required")
		if not notes:
			raise ValueError("export reviewer notes are required")
		decided = AuditExportRequest(
			id=record.id,
			tenant_id=record.tenant_id,
			query=dict(record.query),
			requested_by=record.requested_by,
			contains_pii=record.contains_pii,
			masking_enabled=record.masking_enabled,
			reason=record.reason,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			requested_at=record.requested_at,
			decided_at=datetime.utcnow(),
		)
		self._exports[self._tenant_key(tenant_id, export_id)] = decided
		self._record_governance(
			tenant_id=tenant_id,
			event_type="export_decided",
			subject_id=export_id,
			message=f"Audit export {export_id} was {decision}.",
			evidence={"reviewer": reviewer, "decision": decision},
		)
		return decided.model_dump(mode="json")

	def request_purge(
		self,
		purge_id: str,
		tenant_id: str,
		requested_by: str,
		scope: dict[str, Any],
		reason: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, purge_id) in self._purges:
			raise ValueError(f"purge request already exists for tenant: {purge_id}")
		if not requested_by:
			raise ValueError("purge requester is required")
		if not scope:
			raise ValueError("purge scope is required")
		if not reason:
			raise ValueError("purge reason is required")
		legal_hold_active = self._legal_hold_active(tenant_id, scope)
		result = self.evaluate({
			"tenant_id_missing": not bool(tenant_id),
			"requested_operation": "purge",
			"legal_hold_active": legal_hold_active,
		})
		_raise_if_blocked(result)
		record = AuditPurgeRequest(
			id=purge_id,
			tenant_id=tenant_id,
			scope=dict(scope),
			requested_by=requested_by,
			reason=reason,
		)
		self._purges[self._tenant_key(tenant_id, purge_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="purge_requested",
			subject_id=purge_id,
			message=f"Requested audit purge {purge_id}.",
			evidence={"scope": record.scope, "requested_by": requested_by},
		)
		return record.model_dump(mode="json")

	def decide_purge(
		self,
		purge_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		record = self._purges.get(self._tenant_key(tenant_id, purge_id))
		if record is None:
			raise KeyError(f"unknown purge request for tenant: {purge_id}")
		if decision not in {"approved", "rejected"}:
			raise ValueError("purge decision must be approved or rejected")
		if not reviewer:
			raise ValueError("purge reviewer is required")
		if reviewer == record.requested_by:
			raise PermissionError("dual_control_reviewer_required")
		if not notes:
			raise ValueError("purge reviewer notes are required")
		if decision == "approved" and self._legal_hold_active(tenant_id, record.scope):
			raise PermissionError("legal_hold_active")
		decided = AuditPurgeRequest(
			id=record.id,
			tenant_id=record.tenant_id,
			scope=dict(record.scope),
			requested_by=record.requested_by,
			reason=record.reason,
			decision=decision,
			reviewer=reviewer,
			notes=notes,
			requested_at=record.requested_at,
			decided_at=datetime.utcnow(),
		)
		self._purges[self._tenant_key(tenant_id, purge_id)] = decided
		self._record_governance(
			tenant_id=tenant_id,
			event_type="purge_decided",
			subject_id=purge_id,
			message=f"Audit purge {purge_id} was {decision}.",
			evidence={"reviewer": reviewer, "decision": decision},
		)
		return decided.model_dump(mode="json")

	def open_investigation(
		self,
		investigation_id: str,
		tenant_id: str,
		event_ids: list[str] | tuple[str, ...],
		owner: str,
		priority: str = "high",
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if self._tenant_key(tenant_id, investigation_id) in self._investigations:
			raise ValueError(f"investigation already exists for tenant: {investigation_id}")
		if not owner:
			raise ValueError("investigation owner is required")
		if not event_ids:
			raise ValueError("investigation requires at least one event")
		for event_id in event_ids:
			self._get_event(tenant_id, event_id)
		record = AuditInvestigationRecord(
			id=investigation_id,
			tenant_id=tenant_id,
			event_ids=list(event_ids),
			owner=owner,
			priority=priority,
		)
		self._investigations[self._tenant_key(tenant_id, investigation_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			event_type="investigation_opened",
			subject_id=investigation_id,
			message=f"Opened investigation {investigation_id}.",
			evidence={"event_ids": list(event_ids), "owner": owner, "priority": priority},
		)
		return record.model_dump(mode="json")

	def close_investigation(
		self,
		investigation_id: str,
		tenant_id: str,
		closed_by: str,
		resolution: str,
		evidence: dict[str, Any],
	) -> dict[str, Any]:
		record = self._investigations.get(self._tenant_key(tenant_id, investigation_id))
		if record is None:
			raise KeyError(f"unknown investigation for tenant: {investigation_id}")
		if not closed_by:
			raise ValueError("investigation closure actor is required")
		if not resolution:
			raise ValueError("investigation resolution is required")
		if not evidence:
			raise ValueError("investigation closure evidence is required")
		closed = AuditInvestigationRecord(
			id=record.id,
			tenant_id=record.tenant_id,
			event_ids=list(record.event_ids),
			owner=record.owner,
			priority=record.priority,
			status="closed",
			opened_at=record.opened_at,
			closed_by=closed_by,
			resolution=resolution,
			evidence=dict(evidence),
			closed_at=datetime.utcnow(),
		)
		self._investigations[self._tenant_key(tenant_id, investigation_id)] = closed
		self._record_governance(
			tenant_id=tenant_id,
			event_type="investigation_closed",
			subject_id=investigation_id,
			message=f"Closed investigation {investigation_id}.",
			evidence={"closed_by": closed_by, "resolution": resolution},
		)
		return closed.model_dump(mode="json")

	def list_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._events, tenant_id)

	def list_legal_holds(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._holds, tenant_id)

	def list_exports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._exports, tenant_id)

	def list_purges(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._purges, tenant_id)

	def list_investigations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._dump_tenant_records(self._investigations, tenant_id)

	def list_governance_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._governance_events)
		if tenant_id is not None:
			events = [event for event in events if event.tenant_id == tenant_id]
		return [event.model_dump(mode="json") for event in events]

	def audit_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		events = self.list_events(tenant_id)
		holds = self.list_legal_holds(tenant_id)
		exports = self.list_exports(tenant_id)
		purges = self.list_purges(tenant_id)
		investigations = self.list_investigations(tenant_id)
		return {
			"tenant_id": tenant_id,
			"event_count": len(events),
			"critical_event_count": len([event for event in events if event["severity"] == "critical"]),
			"pii_event_count": len([event for event in events if event["contains_pii"]]),
			"active_legal_hold_count": len([hold for hold in holds if hold["status"] == "active"]),
			"pending_export_count": len([item for item in exports if item["decision"] == "pending"]),
			"pending_purge_count": len([item for item in purges if item["decision"] == "pending"]),
			"open_investigation_count": len([item for item in investigations if item["status"] == "open"]),
			"governance_event_count": len(self.list_governance_events(tenant_id)),
		}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_events(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.append_event(
			event_id=record_id,
			tenant_id=tenant_id,
			actor=str(metadata.get("actor") or metadata.get("user_id") or "system"),
			action=str(metadata.get("action") or "record"),
			resource_type=str(metadata.get("resource_type") or "manual"),
			resource_id=str(metadata.get("resource_id") or record_id),
			severity=str(metadata.get("severity") or "info"),
			contains_pii=bool(metadata.get("contains_pii", False)),
			immutable=bool(metadata.get("immutable", True)),
			checksum=metadata.get("checksum"),
			details=metadata,
			escalation_configured=bool(metadata.get("escalation_configured", True)),
		)
		record["status"] = status
		return record

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_id_missing": not bool(tenant_id)})
		_raise_if_blocked(result)

	def _get_event(self, tenant_id: str, event_id: str) -> AuditLifecycleEvent:
		event = self._events.get(self._tenant_key(tenant_id, event_id))
		if event is None:
			raise KeyError(f"unknown audit event for tenant: {event_id}")
		return event

	def _get_hold(self, tenant_id: str, hold_id: str) -> AuditLegalHoldRecord:
		hold = self._holds.get(self._tenant_key(tenant_id, hold_id))
		if hold is None:
			raise KeyError(f"unknown legal hold for tenant: {hold_id}")
		return hold

	def _legal_hold_active(self, tenant_id: str, scope: dict[str, Any]) -> bool:
		for hold in self._holds.values():
			if hold.tenant_id != tenant_id or hold.status != "active":
				continue
			if not hold.scope:
				return True
			shared_keys = set(hold.scope).intersection(scope)
			if shared_keys and all(scope[key] == hold.scope[key] for key in shared_keys):
				return True
		return False

	def _dump_tenant_records(self, records: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.model_dump(mode="json") for record in sorted(values, key=lambda item: item.id)]

	def _record_governance(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._governance_events.append(
			AuditGovernanceEvent(
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)

	@staticmethod
	def calculate_checksum(
		event_id: str,
		tenant_id: str,
		actor: str,
		action: str,
		resource_type: str,
		resource_id: str,
		severity: str,
		contains_pii: bool,
		immutable: bool,
		details: dict[str, Any] | None = None,
	) -> str:
		payload = {
			"id": event_id,
			"tenant_id": tenant_id,
			"actor": actor,
			"action": action,
			"resource_type": resource_type,
			"resource_id": resource_id,
			"severity": severity,
			"contains_pii": contains_pii,
			"immutable": immutable,
			"details": dict(details or {}),
		}
		return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "audit_policy_blocked") for action in result["actions"])
	raise PermissionError(reasons or "audit_policy_blocked")


__all__ = ["AudlService"]
