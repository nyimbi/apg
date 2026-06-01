"""Process-local API helpers for APG Intelligence Dashboard."""

from __future__ import annotations

try:
	from .service import IntelligenceDashboardService
except ImportError:  # pragma: no cover
	from service import IntelligenceDashboardService  # type: ignore


_SERVICE = IntelligenceDashboardService()


def service() -> IntelligenceDashboardService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def record_dashboard(payload: dict):
	return _SERVICE.record_dashboard(payload["dashboard_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["dashboard_type"], payload["title"], payload["owner_id"], payload["classification"], payload["evidence_reference"])


def record_source(payload: dict):
	return _SERVICE.record_source(payload["source_id"], payload.get("tenant_id", "default"), payload["dashboard_id"], payload["source_type"], payload["source_reference"], payload["custodian_id"], payload["evidence_reference"])


def record_metric(payload: dict):
	return _SERVICE.record_metric(payload["metric_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["metric_type"], payload["metric_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_widget(payload: dict):
	return _SERVICE.record_widget(payload["widget_id"], payload.get("tenant_id", "default"), payload["dashboard_id"], payload["widget_type"], payload["widget_reference"], payload["metric_id"], payload["evidence_reference"])


def record_filter(payload: dict):
	return _SERVICE.record_filter(payload["filter_id"], payload.get("tenant_id", "default"), payload["dashboard_id"], payload["filter_type"], payload["filter_reference"], payload["evidence_reference"])


def record_view(payload: dict):
	return _SERVICE.record_view(payload["view_id"], payload.get("tenant_id", "default"), payload["dashboard_id"], payload["view_type"], payload["view_reference"], payload["viewer_role"], payload["evidence_reference"])


def record_share(payload: dict):
	return _SERVICE.record_share(payload["share_id"], payload.get("tenant_id", "default"), payload["dashboard_id"], payload["share_type"], payload["recipient_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_dashboard_agent(payload: dict):
	return _SERVICE.register_dashboard_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "intelligence dashboard operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("uncited_metric_scope", False), payload.get("classification_leak_scope", False), payload.get("source_tampering_scope", False), payload.get("privacy_bypass_scope", False), payload.get("autonomous_share_scope", False), payload.get("unapproved_public_view_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))

