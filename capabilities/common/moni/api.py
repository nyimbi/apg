"""API helpers for the Monitoring and Observability capability."""

from __future__ import annotations

from typing import Any

from .service import (
	AlertRecord,
	IncidentRecord,
	MoniService,
	RemediationRequestRecord,
	SignalRecord,
	SignalSourceRecord,
	SloRecord,
)


SERVICE = MoniService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
	}


def register_source_record(**kwargs: Any) -> SignalSourceRecord:
	return SERVICE.register_source(**kwargs)


def ingest_signal_record(**kwargs: Any) -> SignalRecord:
	return SERVICE.ingest_signal(**kwargs)


def create_slo_record(**kwargs: Any) -> SloRecord:
	return SERVICE.create_slo(**kwargs)


def create_alert_record(**kwargs: Any) -> AlertRecord:
	return SERVICE.create_alert(**kwargs)


def create_incident_record(**kwargs: Any) -> IncidentRecord:
	return SERVICE.create_incident(**kwargs)


def request_remediation(**kwargs: Any) -> RemediationRequestRecord:
	return SERVICE.request_remediation(**kwargs)


def decide_remediation(**kwargs: Any) -> RemediationRequestRecord:
	return SERVICE.decide_remediation(**kwargs)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id, record_type)


def list_observability(tenant_id: str | None = None) -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"sources": SERVICE.list_records(tenant_id, "sources"),
		"signals": SERVICE.list_records(tenant_id, "signals"),
		"slos": SERVICE.list_records(tenant_id, "slos"),
		"alerts": SERVICE.list_records(tenant_id, "alerts"),
		"incidents": SERVICE.list_records(tenant_id, "incidents"),
		"remediation_requests": SERVICE.list_records(tenant_id, "remediation_requests"),
		"audit_events": SERVICE.list_records(tenant_id, "audit_events"),
	}
