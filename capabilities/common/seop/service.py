"""Service layer for the Security Operations capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .ops_runtime import (
	DetectionRecord,
	IncidentRecord,
	OpsAuditEventRecord,
	PlaybookRecord,
	PostureControlRecord,
	ResponseActionRecord,
	normalize_confidence,
	normalize_severity,
	response_required_actions,
	stable_id,
	utc_now,
)


class SeopService:
	"""Deterministic Security Operations service for APG composition."""

	def __init__(self) -> None:
		self.detections: dict[str, DetectionRecord] = {}
		self.incidents: dict[str, IncidentRecord] = {}
		self.playbooks: dict[str, PlaybookRecord] = {}
		self.responses: dict[str, ResponseActionRecord] = {}
		self.posture_controls: dict[str, PostureControlRecord] = {}
		self.audit_events: dict[str, OpsAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_detection(
		self,
		tenant_id: str,
		title: str,
		alert_source: str,
		anomaly_confidence: float,
		severity: str = "medium",
		signal_refs: list[str] | None = None,
		triage_review_recorded: bool = False,
		owner: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(title or "").strip():
			raise ValueError("detection_title_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_detection",
			"alert_source_present": bool(str(alert_source or "").strip()),
			"anomaly_confidence": normalize_confidence(anomaly_confidence),
			"triage_review_recorded": bool(triage_review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "review_required" if result["decision"] == "require_review" else "new"
		record = DetectionRecord(
			id=stable_id("seop_detection", tenant_id, title, alert_source, len(self.detections)),
			tenant_id=tenant_id,
			title=title,
			alert_source=alert_source,
			severity=normalize_severity(severity),
			anomaly_confidence=context["anomaly_confidence"],
			status=status,
			signal_refs=sorted({str(ref) for ref in signal_refs or [] if str(ref).strip()}),
			owner=owner,
			matched_rules=list(result["matched_rules"]),
			required_actions=response_required_actions(result),
		)
		self.detections[record.id] = record
		self._record_event(tenant_id, "detection_created", record.id, f"Detection created: {title}", owner or alert_source, record.severity)
		return record.to_dict()

	def open_incident(
		self,
		tenant_id: str,
		title: str,
		owner: str,
		severity: str,
		detection_ids: list[str] | None = None,
		escalation_recorded: bool = False,
		evidence_refs: list[str] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(title or "").strip():
			raise ValueError("incident_title_required")
		normalized_severity = normalize_severity(severity)
		context = {
			"tenant_context_present": True,
			"operation": "open_incident",
			"incident_owner_assigned": bool(str(owner or "").strip()),
			"incident_severity": normalized_severity,
			"escalation_recorded": bool(escalation_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = IncidentRecord(
			id=stable_id("seop_incident", tenant_id, title, len(self.incidents)),
			tenant_id=tenant_id,
			title=title,
			owner=owner,
			severity=normalized_severity,
			status="escalated" if escalation_recorded else "open",
			detection_ids=sorted({str(item) for item in detection_ids or [] if str(item).strip()}),
			evidence_refs=sorted({str(item) for item in evidence_refs or [] if str(item).strip()}),
			escalation_recorded=bool(escalation_recorded),
		)
		self.incidents[record.id] = record
		for detection_id in record.detection_ids:
			if detection_id in self.detections:
				self.detections[detection_id].status = "linked"
		self._record_event(tenant_id, "incident_opened", record.id, f"Incident opened: {title}", owner, normalized_severity)
		return record.to_dict()

	def approve_playbook(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		steps: list[str],
		approved_by: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("playbook_name_required")
		if not str(owner or "").strip():
			raise ValueError("playbook_owner_required")
		if not steps:
			raise ValueError("playbook_steps_required")
		if not str(approved_by or "").strip():
			raise PermissionError("playbook_approval_required")
		record = PlaybookRecord(
			id=stable_id("seop_playbook", tenant_id, name),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			steps=[str(step) for step in steps],
			approved_by=approved_by,
		)
		self.playbooks[record.id] = record
		self._record_event(tenant_id, "playbook_approved", record.id, f"Playbook approved: {name}", approved_by)
		return record.to_dict()

	def execute_response(
		self,
		tenant_id: str,
		incident_id: str,
		playbook_id: str,
		action: str,
		actor: str,
		containment_reviewed: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		incident = self._get_incident(tenant_id, incident_id)
		playbook = self._get_playbook(tenant_id, playbook_id)
		context = {
			"tenant_context_present": True,
			"operation": "execute_response",
			"playbook_approved": bool(playbook.approved),
			"containment_review_recorded": bool(containment_reviewed),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = ResponseActionRecord(
			id=stable_id("seop_response", tenant_id, incident_id, playbook_id, action, len(self.responses)),
			tenant_id=tenant_id,
			incident_id=incident_id,
			playbook_id=playbook_id,
			action=action,
			actor=actor,
			status="executed",
			required_actions=response_required_actions(result),
		)
		self.responses[record.id] = record
		incident.status = "responding"
		self._record_event(tenant_id, "response_executed", record.id, f"Response executed: {action}", actor, incident.severity)
		return record.to_dict()

	def record_posture_control(
		self,
		tenant_id: str,
		control_id: str,
		domain: str,
		coverage: float,
		owner: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(control_id or "").strip():
			raise ValueError("control_id_required")
		value = float(coverage)
		if value < 0 or value > 1:
			raise ValueError("coverage_out_of_range")
		status = "covered" if value >= 0.9 else "gap" if value < 0.6 else "partial"
		record = PostureControlRecord(
			id=stable_id("seop_posture", tenant_id, control_id),
			tenant_id=tenant_id,
			control_id=control_id,
			domain=domain,
			coverage=round(value, 3),
			owner=owner,
			status=status,
		)
		self.posture_controls[record.id] = record
		return record.to_dict()

	def close_incident(
		self,
		tenant_id: str,
		incident_id: str,
		closure_evidence: str,
		actor: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		incident = self._get_incident(tenant_id, incident_id)
		if not str(closure_evidence or "").strip():
			raise PermissionError("closure_evidence_required")
		incident.status = "closed"
		incident.closed_at = utc_now()
		incident.evidence_refs.append(closure_evidence)
		self._record_event(tenant_id, "incident_closed", incident.id, f"Incident closed: {incident.title}", actor, incident.severity)
		return incident.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_detection(
			tenant_id=tenant_id,
			title=record_id,
			alert_source=str(metadata.get("alert_source") or "compatibility"),
			anomaly_confidence=float(metadata.get("anomaly_confidence", 0.1)),
			severity=str(metadata.get("severity") or "medium"),
			triage_review_recorded=status in {"reviewed", "closed"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_detections(tenant_id)

	def list_detections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.detections, tenant_id)

	def list_incidents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.incidents, tenant_id)

	def list_playbooks(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.playbooks, tenant_id)

	def list_responses(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.responses, tenant_id)

	def list_posture_controls(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.posture_controls, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		incidents = self.list_incidents(tenant_id)
		detections = self.list_detections(tenant_id)
		return {
			"tenant_id": tenant_id,
			"detection_count": len(detections),
			"review_required_count": sum(1 for item in detections if item["status"] == "review_required"),
			"incident_count": len(incidents),
			"open_incident_count": sum(1 for item in incidents if item["status"] != "closed"),
			"critical_incident_count": sum(1 for item in incidents if item["severity"] == "critical"),
			"approved_playbook_count": len(self.list_playbooks(tenant_id)),
			"response_count": len(self.list_responses(tenant_id)),
			"posture_gap_count": sum(1 for item in self.list_posture_controls(tenant_id) if item["status"] == "gap"),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "seop_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "seop_policy_blocked")

	def _get_incident(self, tenant_id: str, incident_id: str) -> IncidentRecord:
		incident = self.incidents.get(incident_id)
		if incident is None or incident.tenant_id != tenant_id:
			raise KeyError(f"incident_not_found:{incident_id}")
		return incident

	def _get_playbook(self, tenant_id: str, playbook_id: str) -> PlaybookRecord:
		playbook = self.playbooks.get(playbook_id)
		if playbook is None or playbook.tenant_id != tenant_id:
			raise KeyError(f"playbook_not_found:{playbook_id}")
		return playbook

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		record = OpsAuditEventRecord(
			id=stable_id("seop_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=normalize_severity(severity),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
