"""Tests for mob_map capability contract shape and rule evaluation."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	STREAMING,
	SUPPORTED_APP_CATEGORIES,
	SUPPORTED_AUTH_METHODS,
	SUPPORTED_NOTIFICATION_CHANNELS,
	SUPPORTED_PLATFORMS,
	SUPPORTED_SYNC_STRATEGIES,
	SUPPORTED_VERSION_CHANNELS,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "mob_map"


def test_contract_top_level_keys():
	c = get_capability_contract("acme")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required <= set(c.keys())


def test_contract_tenant_scoped():
	c = get_capability_contract("tenant_x")
	assert c["configuration"]["tenant_id"] == "tenant_x"


def test_configuration_schema_required_fields():
	c = get_capability_contract()
	schema = c["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_theme_tokens_complete():
	c = get_capability_contract()
	tokens = c["theme"]["tokens"]
	required_tokens = {"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
		"surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"}
	assert required_tokens <= set(tokens.keys())


def test_theme_components_present():
	c = get_capability_contract()
	comps = c["theme"]["components"]
	assert len(comps) >= 5
	for comp in comps.values():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_rule_engine_structure():
	c = get_capability_contract()
	re = c["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_all_rules_have_required_fields():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_structure():
	for route in UI_ROUTES:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_provides_and_requires_counts():
	assert len(PROVIDES) >= 5
	assert len(REQUIRES) >= 4
	assert "auth" in REQUIRES
	assert "audl" in REQUIRES
	assert "mten" in REQUIRES
	assert "conf" in REQUIRES


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert "stream" in STREAMING
	assert "key" in STREAMING
	assert "events" in STREAMING
	assert len(STREAMING["events"]) >= 5
	assert "guardrails" in STREAMING


def test_supported_constants_non_empty():
	assert len(SUPPORTED_PLATFORMS) >= 4
	assert len(SUPPORTED_AUTH_METHODS) >= 4
	assert len(SUPPORTED_SYNC_STRATEGIES) >= 3
	assert len(SUPPORTED_NOTIFICATION_CHANNELS) >= 3
	assert len(SUPPORTED_VERSION_CHANNELS) >= 3
	assert len(SUPPORTED_APP_CATEGORIES) >= 3


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def test_deny_missing_tenant_context():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "tenant_context_required"


def test_deny_write_without_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "write_requires_policy"


def test_deny_unsupported_platform():
	result = evaluate_capability_rules({"operation": "register_app", "platform_supported": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "platform_must_be_supported"


def test_deny_deployment_without_approval():
	result = evaluate_capability_rules({"operation": "deploy_version", "approval_present": False})
	assert result["decision"] == "deny"


def test_deny_sync_without_encryption():
	result = evaluate_capability_rules({"operation": "start_sync", "encryption_enabled": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "sync_encryption_mandatory"


def test_deny_notification_rate_limit():
	result = evaluate_capability_rules({"operation": "send_notification", "rate_limit_exceeded": True})
	assert result["decision"] == "deny"


def test_deny_cross_tenant_access():
	result = evaluate_capability_rules({"cross_tenant_access": True})
	assert result["decision"] == "deny"
	assert result["rule"] == "cross_tenant_access_denied"


def test_allow_clean_context():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_deny_retired_app_deployment():
	result = evaluate_capability_rules({"operation": "deploy_version", "app_state": "retired"})
	assert result["decision"] == "deny"


def test_deny_rollback_no_previous():
	result = evaluate_capability_rules({"operation": "rollback_version", "previous_version_exists": False})
	assert result["decision"] == "deny"
