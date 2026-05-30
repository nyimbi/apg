"""Regression coverage for the APIG executable capability contract."""

import pytest

from capabilities.common.apig import api, register_capability, view_models as views
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
		"upstreams",
		"consumers",
		"routes",
		"traffic",
		"security",
		"edge",
		"canary",
		"deployments",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 20
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"routes",
		"upstreams",
		"consumers",
		"traffic",
		"security",
		"edge",
		"quota_reviews",
		"canary",
		"deployments",
		"analytics",
		"audit",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/apig/api/v1"
	assert contract["ui"]["view_module"] == "view_models.py"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "gateway_topology_map" in contract["theme"]["components"]


def test_rule_engine_enforces_gateway_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_route",
		"route_owner_assigned": False,
		"absolute_path": False,
		"service_registered": False,
		"methods_present": False,
		"route_exposure": "public",
		"auth_policy_attached": False,
		"mtls_enabled": False,
		"unsafe_http_method_enabled": True,
		"threat_policy_attached": False,
		"rate_limit_configured": False,
		"wasm_filter_attached": True,
		"filter_signature_verified": False,
		"requested_rps_limit": 250000,
		"quota_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"route_requires_owner",
		"route_path_must_be_absolute",
		"route_requires_registered_service",
		"route_requires_methods",
		"public_route_requires_auth_policy",
		"unsafe_method_requires_threat_policy",
		"route_requires_rate_limit",
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
	assert "consumer_lifecycle" in registration["capabilities"]
	assert "deployment_gates" in registration["capabilities"]
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
	consumer = service.register_consumer(
		consumer_id="orders-client",
		tenant_id="tenant-gateway",
		name="Orders Client",
		owner="integration-owner",
		credential_rotation_recorded=True,
	)
	request = service.request_route(
		route_id="orders-public",
		tenant_id="tenant-gateway",
		path="/orders",
		methods=["GET"],
		upstream_id=upstream["id"],
		consumer_id=consumer["id"],
		owner="api-owner",
		route_exposure="public",
		auth_policy_attached=True,
		requested_rps_limit=250000,
		justification="Launch traffic forecast requires elevated quota.",
	)
	traffic = views.traffic_console_model(service, "tenant-gateway")

	assert request["route"]["status"] == "pending_quota_review"
	assert request["quota_review"]["decision"] == "pending"
	assert request["route"]["consumer_id"] == "orders-client"
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

	with pytest.raises(PermissionError, match="credential_rotation_required"):
		service.register_consumer(
			consumer_id="billing-client",
			tenant_id="tenant-gateway",
			name="Billing Client",
			owner="billing-owner",
			credential_rotation_recorded=False,
		)

	with pytest.raises(PermissionError, match="consumer_rbac_approval_required"):
		service.register_consumer(
			consumer_id="restricted-client",
			tenant_id="tenant-gateway",
			name="Restricted Client",
			owner="billing-owner",
			access_tier="restricted",
			credential_rotation_recorded=True,
			rbac_approval_recorded=False,
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

	with pytest.raises(PermissionError, match="mtls_required"):
		service.request_route(
			route_id="external-no-mtls",
			tenant_id="tenant-gateway",
			path="/billing",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
			route_exposure="external",
			mtls_enabled=False,
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

	with pytest.raises(PermissionError, match="rate_limit_required"):
		service.request_route(
			route_id="no-rate-limit",
			tenant_id="tenant-gateway",
			path="/billing",
			methods=["GET"],
			upstream_id="billing-api",
			owner="api-owner",
			rate_limit_configured=False,
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


def test_service_enforces_policy_canary_deployment_and_retirement_guardrails():
	service = ApigService()
	tenant_id = "tenant-lifecycle"
	service.register_upstream(
		upstream_id="orders-api",
		tenant_id=tenant_id,
		name="Orders API",
		base_url="https://orders.internal",
		owner="platform-owner",
	)
	route = service.request_route(
		route_id="orders-route",
		tenant_id=tenant_id,
		path="/orders",
		methods=["GET"],
		upstream_id="orders-api",
		owner="api-owner",
	)
	assert route["status"] == "active"

	policy = service.change_policy(
		policy_id="policy-auth",
		tenant_id=tenant_id,
		name="Auth policy",
		policy_type="security",
		actor="security",
		policy_review_recorded=False,
	)
	assert policy["status"] == "pending_review"
	assert policy["matched_rules"] == ["policy_change_requires_review"]

	no_rollback_shift = service.shift_traffic(
		shift_id="shift-no-rollback",
		tenant_id=tenant_id,
		route_id="orders-route",
		canary_percent=5,
		actor="release",
		rollback_plan_recorded=False,
		canary_review_recorded=False,
	)
	assert no_rollback_shift["status"] == "denied"
	assert no_rollback_shift["matched_rules"] == ["traffic_shift_requires_rollback_plan"]

	blocked_shift = service.shift_traffic(
		shift_id="shift-too-high",
		tenant_id=tenant_id,
		route_id="orders-route",
		canary_percent=75,
		actor="release",
		rollback_plan_recorded=True,
		canary_review_recorded=True,
		rollback_plan="Return all traffic to stable route.",
	)
	assert blocked_shift["status"] == "denied"
	assert blocked_shift["matched_rules"] == ["canary_percent_limit_enforced"]

	review_shift = service.shift_traffic(
		shift_id="shift-review",
		tenant_id=tenant_id,
		route_id="orders-route",
		canary_percent=20,
		actor="release",
		rollback_plan_recorded=True,
		canary_review_recorded=False,
		rollback_plan="Return all traffic to stable route.",
	)
	assert review_shift["status"] == "pending_review"
	assert review_shift["matched_rules"] == ["canary_requires_review"]

	allowed_shift = service.shift_traffic(
		shift_id="shift-allowed",
		tenant_id=tenant_id,
		route_id="orders-route",
		canary_percent=5,
		actor="release",
		rollback_plan_recorded=True,
		canary_review_recorded=False,
		rollback_plan="Return all traffic to stable route.",
	)
	assert allowed_shift["status"] == "active"

	bad_deployment = service.deploy_gateway(
		deployment_id="deploy-bad",
		tenant_id=tenant_id,
		environment="production",
		region="unknown",
		actor="platform",
		observability_configured=False,
		deployment_approval_recorded=False,
	)
	assert bad_deployment["status"] == "denied"
	assert set(bad_deployment["matched_rules"]) >= {
		"deployment_requires_region",
		"deployment_requires_observability",
		"production_deployment_requires_approval",
	}

	prod_deployment = service.deploy_gateway(
		deployment_id="deploy-prod",
		tenant_id=tenant_id,
		environment="production",
		region="edge-east",
		actor="platform",
		observability_configured=True,
		deployment_approval_recorded=False,
	)
	assert prod_deployment["status"] == "pending_review"
	assert prod_deployment["matched_rules"] == ["production_deployment_requires_approval"]

	with pytest.raises(PermissionError, match="impact_review_required"):
		service.retire_route(
			route_id="orders-route",
			tenant_id=tenant_id,
			actor="api-owner",
			impact_review_recorded=False,
		)

	retired = service.retire_route(
		route_id="orders-route",
		tenant_id=tenant_id,
		actor="api-owner",
		impact_review_recorded=True,
	)
	assert retired["status"] == "retired"
	assert service.gateway_summary(tenant_id)["audit_event_count"] >= 9


def test_view_models_expose_gateway_composition_surfaces():
	service = ApigService()
	tenant_id = "tenant-ui"
	service.register_upstream(
		upstream_id="ui-api",
		tenant_id=tenant_id,
		name="UI API",
		base_url="https://ui.internal",
		owner="platform",
	)
	service.register_consumer(
		consumer_id="ui-client",
		tenant_id=tenant_id,
		name="UI Client",
		owner="integrations",
		credential_rotation_recorded=True,
	)
	service.request_route(
		route_id="ui-route",
		tenant_id=tenant_id,
		path="/ui",
		methods=["GET"],
		upstream_id="ui-api",
		consumer_id="ui-client",
		owner="api",
		wasm_filter_attached=True,
		filter_signature_verified=True,
	)

	assert views.dashboard_model(service, tenant_id)["summary"]["route_count"] == 1
	assert views.upstream_manager_model(service, tenant_id)["rows"][0]["id"] == "ui-api"
	assert views.consumer_manager_model(service, tenant_id)["rows"][0]["id"] == "ui-client"
	assert views.route_designer_model(service, tenant_id)["routes"][0]["id"] == "ui-route"
	assert views.edge_filter_model(service, tenant_id)["routes"][0]["id"] == "ui-route"
	assert views.security_policy_model(service, tenant_id)["required_external_route_controls"] == ["mtls_enabled"]
	assert views.settings_model(tenant_id)["configuration"]["adapters"]["event_stream"] == "bytewax"


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

	consumer = api.register_consumer({
		"id": "api-consumer",
		"tenant_id": "tenant-api-gateway",
		"name": "API Consumer",
		"owner": "consumer-owner",
		"credential_rotation_recorded": "true",
	})
	assert consumer["id"] == "api-consumer"

	shift = api.shift_traffic({
		"id": "api-shift",
		"tenant_id": "tenant-api-gateway",
		"route_id": "api-route",
		"canary_percent": 5,
		"actor": "release",
		"rollback_plan_recorded": "true",
		"canary_review_recorded": "false",
	})
	assert shift["status"] == "active"
