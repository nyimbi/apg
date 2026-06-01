"""Generated-application view models for the APIG capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .gateway_runtime import ApigService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.gateway_summary(tenant_id),
		"upstreams": service.list_upstreams(tenant_id),
		"consumers": service.list_consumers(tenant_id),
		"records": service.list_records(tenant_id),
		"quota_reviews": service.list_quota_reviews(tenant_id),
		"traffic_shifts": service.list_traffic_shifts(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"gateway_agents": service.list_gateway_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"review_evidence": contract["review_evidence"],
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def route_designer_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"upstreams": service.list_upstreams(tenant_id),
		"consumers": service.list_consumers(tenant_id),
		"routes": service.list_routes(tenant_id),
		"required_fields": ["id", "path", "methods", "upstream_id", "owner"],
		"exposure_options": ["internal", "partner", "public", "external"],
	}


def upstream_manager_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"rows": service.list_upstreams(tenant_id),
		"columns": ["id", "name", "base_url", "owner", "health", "labels"],
	}


def consumer_manager_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"rows": service.list_consumers(tenant_id),
		"columns": ["id", "name", "owner", "access_tier", "identity_provider", "status"],
	}


def traffic_console_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	summary = service.gateway_summary(tenant_id)
	traffic = get_capability_contract(tenant_id)["configuration"]["traffic"]
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		**summary,
		"quota_reviews": service.list_quota_reviews(tenant_id),
		"traffic_shifts": service.list_traffic_shifts(tenant_id),
		"quota_review_required_above_rps": traffic["max_rps_without_review"],
	}


def security_policy_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"routes": service.list_routes(tenant_id),
		"consumers": service.list_consumers(tenant_id),
		"policies": service.list_policies(tenant_id),
		"required_public_route_controls": ["auth_policy_attached"],
		"required_external_route_controls": ["mtls_enabled"],
		"required_unsafe_method_controls": ["threat_policy_attached"],
	}


def edge_filter_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"routes": [
			route for route in service.list_routes(tenant_id)
			if route["wasm_filter_attached"]
		],
		"required_fields": ["wasm_filter_attached", "filter_signature_verified"],
	}


def quota_review_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"rows": service.list_quota_reviews(tenant_id),
		"columns": ["id", "route_id", "requested_rps_limit", "requester", "decision", "reviewer", "notes"],
	}


def canary_release_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"rows": service.list_traffic_shifts(tenant_id),
		"columns": ["id", "route_id", "canary_percent", "actor", "status", "decision", "matched_rules"],
	}


def deployment_gate_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"rows": service.list_deployments(tenant_id),
		"columns": ["id", "environment", "region", "actor", "status", "decision", "matched_rules"],
	}


def analytics_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"summary": service.gateway_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def gateway_agent_roster_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_gateway_agents(tenant_id),
		"columns": [
			"id",
			"name",
			"runtime",
			"role",
			"scope",
			"owner",
			"purpose",
			"human_approval_required",
			"status",
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"pending_reviews": [
			review
			for review in service.list_pending_reviews(tenant_id)
			if "runtime" in review and "role" in review
		],
	}


def lifecycle_batch_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_lifecycle_batches(tenant_id),
		"columns": ["id", "event_stream", "mutation_count", "accepted", "status", "matched_rules"],
		"streaming": contract["streaming"],
	}


def audit_timeline_model(service: ApigService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ApigService()
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"columns": ["timestamp", "event_type", "subject_id", "message", "evidence"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
