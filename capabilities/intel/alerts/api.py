"""Process-local API helpers for APG Alert Management."""

from __future__ import annotations

try:
	from .service import AlertManagementService
except ImportError:  # pragma: no cover
	from service import AlertManagementService  # type: ignore


_SERVICE = AlertManagementService()


def service() -> AlertManagementService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def record_rule(payload: dict):
	return _SERVICE.record_rule(payload["rule_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["rule_type"], payload["rule_reference"], payload["severity"], payload["owner_id"], payload["evidence_reference"])


def record_signal(payload: dict):
	return _SERVICE.record_signal(payload["signal_id"], payload.get("tenant_id", "default"), payload["rule_id"], payload["signal_type"], payload["signal_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_alert(payload: dict):
	return _SERVICE.record_alert(payload["alert_id"], payload.get("tenant_id", "default"), payload["signal_id"], payload["alert_type"], payload["severity"], payload["alert_reference"], payload["evidence_reference"])


def record_escalation(payload: dict):
	return _SERVICE.record_escalation(payload["escalation_id"], payload.get("tenant_id", "default"), payload["alert_id"], payload["escalation_type"], payload["target_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_notification(payload: dict):
	return _SERVICE.record_notification(payload["notification_id"], payload.get("tenant_id", "default"), payload["alert_id"], payload["notification_type"], payload["recipient_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_assignment(payload: dict):
	return _SERVICE.record_assignment(payload["assignment_id"], payload.get("tenant_id", "default"), payload["alert_id"], payload["assignment_type"], payload["assignee_id"], payload["evidence_reference"])


def record_resolution(payload: dict):
	return _SERVICE.record_resolution(payload["resolution_id"], payload.get("tenant_id", "default"), payload["alert_id"], payload["resolution_type"], payload["resolution_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_alert_agent(payload: dict):
	return _SERVICE.register_alert_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "alert management operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("unapproved_escalation_scope", False), payload.get("unapproved_notification_scope", False), payload.get("alert_suppression_scope", False), payload.get("evidence_fabrication_scope", False), payload.get("privacy_bypass_scope", False), payload.get("autonomous_closure_scope", False), payload.get("severity_downgrade_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))

