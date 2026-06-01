"""View models for APG Crowdfunding Platform."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CrowdfundingPlatformService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import CrowdfundingPlatformService  # type: ignore


def dashboard_model(service: CrowdfundingPlatformService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Crowdfunding Platform",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Issuers", "value": summary["issuer_count"], "icon": "building-2"},
			{"label": "Campaigns", "value": summary["campaign_count"], "icon": "megaphone"},
			{"label": "Commitments", "value": summary["commitment_count"], "icon": "hand-coins"},
			{"label": "Escrow", "value": summary["escrow_count"], "icon": "landmark"},
			{"label": "Payouts", "value": summary["payout_count"], "icon": "circle-dollar-sign"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def crowdfunding_console_model(service: CrowdfundingPlatformService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"issuers": _items(service.issuers, tenant_id),
		"campaigns": _items(service.campaigns, tenant_id),
		"disclosures": _items(service.disclosures, tenant_id),
		"commitments": _items(service.commitments, tenant_id),
		"escrow": _items(service.escrow, tenant_id),
		"milestones": _items(service.milestones, tenant_id),
		"payouts": _items(service.payouts, tenant_id),
		"updates": _items(service.updates, tenant_id),
		"compliance": _items(service.compliance, tenant_id),
		"reviews": _items(service.reviews, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: CrowdfundingPlatformService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = crowdfunding_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
