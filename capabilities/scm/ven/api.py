"""Dependency-light API helpers for SCM Vendor Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import VendorManagementLifecycleService
except ImportError:  # pragma: no cover
	from service import VendorManagementLifecycleService  # type: ignore


_SERVICE = VendorManagementLifecycleService()


def service() -> VendorManagementLifecycleService:
	return _SERVICE


def create_vendor(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_vendor(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("code", ""), payload.get("name", ""), payload.get("vendor_type", "service_provider"), payload.get("category", ""), payload.get("country", ""), payload.get("owner_id", ""))


def qualify_vendor(payload: dict[str, Any]) -> dict[str, Any]:
	return service().qualify_vendor(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("criteria", []), payload.get("qualified_by", ""), payload.get("score"), payload.get("reviewed_by"))


def onboard_vendor(payload: dict[str, Any]) -> dict[str, Any]:
	return service().onboard_vendor(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("checklist", []), payload.get("owner_id", ""))


def record_performance(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_performance(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("period", ""), payload.get("scores", {}), payload.get("reviewed_by"))


def record_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_risk(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("risk_type", "operational"), payload.get("tier", "medium"), payload.get("description", ""), payload.get("owner_id"))


def record_compliance(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_compliance(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("framework", ""), payload.get("status", "pending"), payload.get("evidence_id", ""), payload.get("reviewed_by"))


def create_contract(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_contract(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("value"), payload.get("currency", ""), payload.get("start_date", ""), payload.get("end_date", ""), payload.get("approved_by", ""))


def record_communication(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_communication(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("channel", ""), payload.get("subject", ""), payload.get("sentiment", "neutral"), payload.get("owner_id"))


def create_portal_user(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_portal_user(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("email", ""), payload.get("role", ""), payload.get("approved_by", ""))


def create_scorecard(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_scorecard(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("vendor_id", ""), payload.get("period", ""), payload.get("performance_id", ""), payload.get("risk_id", ""), payload.get("compliance_ids", []), payload.get("generated_by", ""))


def register_vendor_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_vendor_agent(payload.get("tenant_id", "default"), payload.get("name", "Vendor Agent"), payload.get("runtime", "codex"), payload.get("role", "vendor_onboarding_reviewer"), payload.get("purpose", "review vendor records"), payload.get("owner_id"))


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_record(payload)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return service().dashboard_summary(tenant_id)


def audit_events(tenant_id: str = "default") -> list[dict[str, Any]]:
	return service().audit_events(tenant_id)
