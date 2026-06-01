"""View models for APG InsurTech."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import InsurTechService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import InsurTechService  # type: ignore


def dashboard_model(service: InsurTechService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {"title": "InsurTech", "tenant_id": tenant_id, "summary": summary, "cards": [{"label": "Policyholders", "value": summary["policyholder_count"], "icon": "users"}, {"label": "Products", "value": summary["product_count"], "icon": "package-check"}, {"label": "Policies", "value": summary["policy_count"], "icon": "shield-check"}, {"label": "Premiums", "value": summary["premium_count"], "icon": "receipt"}, {"label": "Claims", "value": summary["claim_count"], "icon": "file-warning"}, {"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"}], "routes": contract["ui"]["routes"], "theme": contract["theme"]}


def insurance_console_model(service: InsurTechService, tenant_id: str) -> dict[str, Any]:
	return {"tenant_id": tenant_id, "policyholders": _items(service.policyholders, tenant_id), "products": _items(service.products, tenant_id), "quotes": _items(service.quotes, tenant_id), "policies": _items(service.policies, tenant_id), "premiums": _items(service.premiums, tenant_id), "claims": _items(service.claims, tenant_id), "documents": _items(service.documents, tenant_id), "risk": _items(service.risk, tenant_id), "reinsurance": _items(service.reinsurance, tenant_id), "compliance": _items(service.compliance, tenant_id), "reviews": _items(service.reviews, tenant_id), "agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"]}


def route_models(service: InsurTechService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = insurance_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
