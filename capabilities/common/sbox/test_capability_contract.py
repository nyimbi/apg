"""Regression coverage for the SBOX executable capability contract."""

from capabilities.common.sbox import register_capability
from capabilities.common.sbox.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sbox", {"sandboxes": {"ttl_hours": 12}})

	assert contract["capability"] == "sbox"
	assert contract["configuration"]["tenant_id"] == "tenant-sbox"
	assert contract["configuration"]["sandboxes"]["ttl_hours"] == 12
	assert contract["configuration_schema"]["required"] == ["tenant_id", "sandboxes", "isolation", "datasets", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sandboxes", "templates", "datasets", "runs", "policies", "logs", "settings"}
	assert contract["theme"]["name"] == "sbox_safe_testing"


def test_rule_engine_enforces_sbox_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_sandbox", "sandbox_owner_assigned": False, "isolation_profile_attached": False, "secret_access_requested": True, "secret_redaction_enabled": False, "ttl_hours": 72, "lifecycle_review_recorded": False})
	network_result = evaluate_capability_rules({"tenant_context_present": True, "isolation_profile_attached": True, "outbound_network_requested": True, "network_approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "sandbox_requires_owner", "sandbox_requires_isolation_profile", "secrets_require_redaction", "long_lived_sandbox_requires_review"}
	assert network_result["matched_rules"] == ["outbound_network_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "sbox"
	assert "plgn" in registration["dependencies"]
	assert registration["ui_components"]["runs"] == "/sbox/runs"
	assert "sbox:run_tests" in registration["permissions"]
