"""UI metadata helpers for the APG Intelligent Gateway capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .gateway_runtime import ApigService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.gateway_summary(tenant_id),
		"upstreams": service.list_upstreams(tenant_id),
		"records": service.list_records(tenant_id),
		"quota_reviews": service.list_quota_reviews(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def route_designer_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	return {
		"upstreams": service.list_upstreams(tenant_id),
		"routes": service.list_routes(tenant_id),
		"required_fields": ["id", "path", "methods", "upstream_id", "owner"],
		"exposure_options": ["internal", "partner", "public"],
	}


def traffic_console_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	summary = service.gateway_summary(tenant_id)
	return {
		"summary": summary,
		**summary,
		"quota_reviews": service.list_quota_reviews(tenant_id),
		"quota_review_required_above_rps": 100000,
	}


def security_policy_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	return {
		"routes": service.list_routes(tenant_id),
		"required_public_route_controls": ["auth_policy_attached"],
		"required_unsafe_method_controls": ["threat_policy_attached"],
	}


def edge_filter_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	return {
		"routes": [
			route for route in service.list_routes(tenant_id)
			if route["wasm_filter_attached"]
		],
		"required_fields": ["wasm_filter_attached", "filter_signature_verified"],
	}


def analytics_model(
	service: ApigService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ApigService()
	return {
		"summary": service.gateway_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}
