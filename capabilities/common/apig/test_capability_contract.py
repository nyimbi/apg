"""Regression coverage for the APIG executable capability contract."""

from capabilities.common.apig import register_capability
from capabilities.common.apig.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


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
	assert "auth_rbac" in registration["dependencies"]
