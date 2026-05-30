"""Dependency-light API helpers for Sustainability and ESG Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import ESGManagementLifecycleService
except ImportError:  # pragma: no cover
	from service import ESGManagementLifecycleService  # type: ignore


_SERVICE = ESGManagementLifecycleService()


def service() -> ESGManagementLifecycleService:
	return _SERVICE


def create_esg_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_esg_profile(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("name", ""), payload.get("industry", ""), payload.get("country", ""), payload.get("reporting_year"), payload.get("owner_id", ""))


def add_framework(payload: dict[str, Any]) -> dict[str, Any]:
	return service().add_framework(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("code", "gri"), payload.get("version", ""), payload.get("mandatory", True), payload.get("owner_id", ""))


def define_metric(payload: dict[str, Any]) -> dict[str, Any]:
	return service().define_metric(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("pillar", "environmental"), payload.get("metric_type", "emissions"), payload.get("unit", "tco2e"), payload.get("name", ""), payload.get("owner_id", ""))


def record_measurement(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_measurement(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("metric_id", ""), payload.get("period", ""), payload.get("value"), payload.get("source", "manual"), payload.get("evidence_id", ""), payload.get("reviewed_by"))


def set_target(payload: dict[str, Any]) -> dict[str, Any]:
	return service().set_target(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("metric_id", ""), payload.get("target_type", "absolute"), payload.get("baseline_value"), payload.get("target_value"), payload.get("due_date", ""), payload.get("owner_id", ""))


def record_supplier_assessment(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_supplier_assessment(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("supplier_id", ""), payload.get("period", ""), payload.get("score", 0), payload.get("risk_tier", "low"), payload.get("evidence_id", ""), payload.get("owner_id"))


def record_initiative(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_initiative(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("name", ""), payload.get("pillar", "environmental"), payload.get("budget", 0), payload.get("owner_id", ""), payload.get("expected_impact", ""))


def record_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_risk(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("tier", "medium"), payload.get("category", "climate"), payload.get("description", ""), payload.get("owner_id"))


def create_report(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_report(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("report_type", "annual"), payload.get("period", ""), payload.get("framework_ids", []), payload.get("measurement_ids", []), payload.get("approved_by", ""))


def register_stakeholder(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_stakeholder(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("stakeholder_type", "investor"), payload.get("name", ""), payload.get("channel", ""), payload.get("consent_recorded", False))


def record_engagement(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_engagement(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("stakeholder_id", ""), payload.get("topic", ""), payload.get("channel", ""), payload.get("sentiment", "neutral"), payload.get("owner_id"))


def register_esg_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_esg_agent(payload.get("tenant_id", "default"), payload.get("name", "ESG Agent"), payload.get("runtime", "codex"), payload.get("role", "sustainability_reviewer"), payload.get("purpose", "review ESG records"), payload.get("owner_id"))


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return service().dashboard_summary(tenant_id)


def audit_events(tenant_id: str = "default") -> list[dict[str, Any]]:
	return service().audit_events(tenant_id)
