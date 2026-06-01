"""View models for generated FinTech Compliance Automation screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ComplianceAutomationService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ComplianceAutomationService  # type: ignore


def dashboard_model(service: ComplianceAutomationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "FinTech Compliance Automation", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def compliance_console_model(service: ComplianceAutomationService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "obligations": _tenant_items(service.obligations, tenant_id), "controls": _tenant_items(service.controls, tenant_id), "checks": _tenant_items(service.checks, tenant_id), "evidence": _tenant_items(service.evidence, tenant_id), "attestations": _tenant_items(service.attestations, tenant_id), "issues": _tenant_items(service.issues, tenant_id), "remediations": _tenant_items(service.remediations, tenant_id), "reports": _tenant_items(service.reports, tenant_id), "reviews": _tenant_items(service.reviews, tenant_id)}


def agent_workbench_model(service: ComplianceAutomationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id]}


def _tenant_items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda value: value.id) if item.tenant_id == tenant_id]
