"""Tests for mob_mdm capability contract shape and rule evaluation."""

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
	SUPPORTED_DEVICE_TYPES,
	SUPPORTED_ENROLMENT_METHODS,
	SUPPORTED_OS_PLATFORMS,
	SUPPORTED_POLICY_TYPES,
	SUPPORTED_WIPE_TYPES,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "mob_mdm"


def test_contract_top_level_keys():
	c = get_capability_contract("corp")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema",
		"rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required <= set(c.keys())


def test_contract_tenant_scoped():
	c = get_capability_contract("my_org")
	assert c["configuration"]["tenant_id"] == "my_org"


def test_configuration_schema_required_fields():
	c = get_capability_contract()
	schema = c["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_theme_tokens_complete():
	c = get_capability_contract()
	tokens = c["theme"]["tokens"]
	required = {"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
		"surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"}
	assert required <= set(tokens.keys())


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


def test_provides_and_requires():
	assert len(PROVIDES) >= 5
	assert len(REQUIRES) >= 4
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5
	assert "guardrails" in STREAMING


def test_supported_constants():
	assert len(SUPPORTED_DEVICE_TYPES) >= 4
	assert len(SUPPORTED_OS_PLATFORMS) >= 4
	assert len(SUPPORTED_ENROLMENT_METHODS) >= 4
	assert len(SUPPORTED_POLICY_TYPES) >= 4
	assert len(SUPPORTED_WIPE_TYPES) >= 3


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

def test_deny_missing_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "tenant_context_required"


def test_deny_write_no_policy():
	r = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert r["decision"] == "deny"


def test_deny_unsupported_device_type():
	r = evaluate_capability_rules({"operation": "enrol_device", "device_type_supported": False})
	assert r["decision"] == "deny"


def test_deny_enrolment_no_approval():
	r = evaluate_capability_rules({"operation": "enrol_device", "approval_present": False})
	assert r["decision"] == "deny"


def test_deny_policy_activation_no_approval():
	r = evaluate_capability_rules({"operation": "activate_policy", "approval_present": False})
	assert r["decision"] == "deny"


def test_deny_wipe_no_dual_approval():
	r = evaluate_capability_rules({"operation": "request_wipe", "dual_approval_present": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "wipe_requires_dual_approval"


def test_deny_app_distribution_unenrolled():
	r = evaluate_capability_rules({"operation": "distribute_app", "device_enrolled": False})
	assert r["decision"] == "deny"


def test_deny_non_compliant_blocks_access():
	r = evaluate_capability_rules({"operation": "grant_access", "device_compliance_state": "non_compliant"})
	assert r["decision"] == "deny"


def test_deny_suspended_device():
	r = evaluate_capability_rules({"device_state": "suspended"})
	assert r["decision"] == "deny"


def test_deny_wiped_device():
	r = evaluate_capability_rules({"device_state": "wiped"})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_allow_clean_context():
	r = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert r["decision"] == "allow"
