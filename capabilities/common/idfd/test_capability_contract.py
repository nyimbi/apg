"""Regression coverage for the IDFD executable capability contract."""

from capabilities.common.idfd import register_capability
from capabilities.common.idfd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sso", {"sessions": {"max_session_hours": 8}})

	assert contract["capability"] == "idfd"
	assert contract["configuration"]["tenant_id"] == "tenant-sso"
	assert contract["configuration"]["sessions"]["max_session_hours"] == 8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "providers", "protocols", "sessions", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "providers", "protocols", "mappings", "sessions", "certificates", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/idfd/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "certificate_timeline" in contract["theme"]["components"]


def test_rule_engine_enforces_federation_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_provider",
		"signing_key_present": False,
		"protocol": "saml",
		"assertion_encrypted": False,
		"redirect_allowlist_configured": False,
		"session_privilege": "privileged",
		"mfa_completed": False,
		"metadata_age_hours": 72,
		"metadata_refresh_completed": False
	})
	oidc_result = evaluate_capability_rules({"tenant_context_present": True, "protocol": "oidc", "redirect_allowlist_configured": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "provider_requires_signing_key", "saml_assertion_requires_encryption", "privileged_federation_requires_mfa", "stale_metadata_requires_refresh"}
	assert oidc_result["matched_rules"] == ["oidc_client_requires_redirect_allowlist"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "idfd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "idfd_federation_console"
	assert registration["ui_components"]["providers"] == "/idfd/providers"
	assert "mfau" in registration["dependencies"]
	assert "idfd:rotate_keys" in registration["permissions"]
