"""Regression coverage for the APIG executable capability contract."""

import pytest

from capabilities.common.apig import api, register_capability, views
from capabilities.common.apig.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.apig.gateway_runtime import ApigService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-gateway", {"traffic": {"default_rps_limit": 500}})

	assert contract["capability"] == "apig"
	assert contract["configuration"]["tenant_id"] == "tenant-gateway"
	assert contract["configuration"]["traffic"]["default_rps_limit"] == 500
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"routing",
		"security",
		"traffic",
		"observability",
		"edge",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"routes",
		"traffic",
		"security",
		"upstreams",
		"edge",
		"analytics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/apig/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "gateway_topology_map" in contract["theme"]["components"]


def test_rule_engine_enforces_gateway_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_route",
		"service_registered": False,
		"route_exposure": "public",
		"auth_policy_attached": False,
		"unsafe_http_method_enabled": True,
		"threat_policy_attached": False,
		"wasm_filter_attached": True,
		"filter_signature_verified": False,
		"requested_rps_limit": 250000,
		"quota_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"route_requires_registered_service",
		"public_route_requires_auth_policy",
		"unsafe_method_requires_threat_policy",
		"wasm_filter_requires_signature",
		"high_quota_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "apig_gateway_console"
	assert registration["ui_components"]["routes"] == "/apig/routes"
	assert "quota_review_governance" in registration["capabilities"]
	assert "auth_rbac" in registration["dependencies"]


def test_service_runs_high_quota_route_review_activation_lifecycle():
	service = ApigService()
	upstream = service.register_upstream(
		upstream_id="orders-api",
		tenant_id="tenant-gateway",
		name="Orders API",
		base_url="https://orders.internal",
		owner="platform-owner",
		labels={"domain": "orders"},
	)
	request = service.request_route(
		route_id="orders-public",
		tenant_id="tenant-gateway",
		path="/orders",
		methods=["GET"],
		upstream_id=upstream["id"],
		owner="api-owner",
		route_exposure="public",
		auth_policy_attached=True,
		requested_rps_limit=250000,
		justification="Launch traffic forecast requires elevated quota.",
	)
	traffic = views.traffic_console_model(service, "tenant-gateway")

	assert request["route"]["status"] == "pending_quota_review"
	assert request["quota_review"]["decision"] == "pending"
	assert traffic["pending_quota_review_count"] == 1

	with pytest.raises(PermissionError, match="quota_review_required"):
		service.activate_route("orders-public", "tenant-gateway")

	decision = service.decide_quota_review(
		review_id=request["quota_review"]["id"],
		tenant_id="tenant-gateway",
		reviewer="gateway-reviewer",
		decision="approved",
		notes="Capacity and abuse protections reviewed.",
	)
	activated = service.activate_route("orders-public", "tenant-gateway")
	dashboard = views.dashboard_model(service, "tenant-gateway")

	assert decision["decision"] == "approved"
	assert activated["status"] == "active"
	assert dashboard["summary"]["active_route_count"] == 1
	assert {event["event_type"] for event in dashboard["audit_events"]} >= {
		"upstream_registered",
		"route_requested",
		"quota_review_requested",
		"quota_review_decided",
		"route_activated",
	}

	rejected = service.request_route(
		route_id="orders-rejected",
		tenant_id="tenant-gateway",
		path="/orders-rejected",
		methods=["GET"],
		upstream_id=upstream["id"],
		owner="api-owner",
		route_exposure="public",
		auth_policy_attached=True,
		requested_rps_limit=250000,
	)
	service.decide_quota_review(
		review_id=rejected["quota_review"]["id"],
		tenant_id="tenant-gateway",
		reviewer="gateway-reviewer",
		decision="rejected",
		notes="Quota exceeds launch risk budget.",
	)

	with pytest.raises(PermissionError, match="quota_review_required"):
		service.activate_route("orders-rejected", "tenant-gateway")


def test_service_blocks_route_policy_violations_and_tenant_mismatch():
	service = ApigService()
	service.register_upstream(
		upstream_id="billing-api",
		tenant_id="tenant-gateway",
		name="Billing API",
		base_url="https://billing.internal",
		owner="platform-owner",
	)

	with pytest.raises(PermissionError, match="auth_policy_required"):
		service.request_route(
			route_id="public-no-auth",
			tenant_id="tenant-gateway",
			path="/billing",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
			route_exposure="public",
			auth_policy_attached=False,
		)

	with pytest.raises(PermissionError, match="threat_policy_required"):
		service.request_route(
			route_id="unsafe-no-threat",
			tenant_id="tenant-gateway",
			path="/billing",
			methods=["POST"],
			upstream_id="billing-api",
			owner="api-owner",
			threat_policy_attached=False,
		)

	with pytest.raises(PermissionError, match="filter_signature_required"):
		service.request_route(
			route_id="unsigned-filter",
			tenant_id="tenant-gateway",
			path="/billing",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
			wasm_filter_attached=True,
			filter_signature_verified=False,
		)

	with pytest.raises(PermissionError, match="registered_service_required"):
		service.request_route(
			route_id="wrong-tenant-upstream",
			tenant_id="other-tenant",
			path="/billing",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
		)

	with pytest.raises(ValueError, match="upstream already exists"):
		service.register_upstream(
			upstream_id="billing-api",
			tenant_id="tenant-gateway",
			name="Billing API Duplicate",
			base_url="https://billing-duplicate.internal",
			owner="platform-owner",
		)

	service.request_route(
		route_id="billing-route",
		tenant_id="tenant-gateway",
		path="/billing",
		methods=["GET"],
		upstream_id="billing-api",
		owner="api-owner",
	)

	with pytest.raises(ValueError, match="route already exists"):
		service.request_route(
			route_id="billing-route",
			tenant_id="tenant-gateway",
			path="/billing-v2",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
		)


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = ApigService()
	for tenant_id, owner in [("tenant-a", "owner-a"), ("tenant-b", "owner-b")]:
		service.register_upstream(
			upstream_id="shared-upstream",
			tenant_id=tenant_id,
			name=f"Shared Upstream {tenant_id}",
			base_url=f"https://{tenant_id}.internal",
			owner=owner,
		)
		service.request_route(
			route_id="shared-route",
			tenant_id=tenant_id,
			path="/shared",
			methods=["GET"],
			upstream_id="shared-upstream",
			owner=owner,
			route_exposure="internal",
		)

	assert service.list_routes("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_routes("tenant-b")[0]["tenant_id"] == "tenant-b"
	assert service.list_routes("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_routes("tenant-b")[0]["owner"] == "owner-b"


def test_api_helpers_expose_governed_route_lifecycle():
	upstream = api.register_upstream({
		"id": "api-upstream",
		"tenant_id": "tenant-api-gateway",
		"name": "API Upstream",
		"base_url": "https://api-upstream.internal",
		"owner": "api-owner",
	})
	request = api.request_route({
		"id": "api-route",
		"tenant_id": upstream["tenant_id"],
		"path": "/api",
		"methods": ["GET"],
		"upstream_id": upstream["id"],
		"owner": "api-owner",
		"route_exposure": "public",
		"auth_policy_attached": "true",
		"requested_rps_limit": 250000,
	})
	decision = api.decide_quota_review({
		"id": request["quota_review"]["id"],
		"tenant_id": upstream["tenant_id"],
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Quota accepted.",
	})
	activated = api.activate_route({
		"id": request["route"]["id"],
		"tenant_id": upstream["tenant_id"],
	})

	assert decision["decision"] == "approved"
	assert activated["status"] == "active"
	assert api.list_routes(upstream["tenant_id"])[0]["id"] == "api-route"
	assert api.list_audit_events(upstream["tenant_id"])[-1]["event_type"] == "route_activated"
