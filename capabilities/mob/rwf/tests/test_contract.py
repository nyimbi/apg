"""Tests for mob_rwf capability contract shape and rule evaluation."""

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
	SUPPORTED_COMPLIANCE_CHECK_TYPES,
	SUPPORTED_EQUIPMENT_TYPES,
	SUPPORTED_INCIDENT_TYPES,
	SUPPORTED_ONBOARDING_STEP_TYPES,
	SUPPORTED_PRODUCTIVITY_METRICS,
	SUPPORTED_VPN_PROTOCOLS,
	SUPPORTED_WORK_POLICY_TYPES,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "mob_rwf"


def test_contract_top_level_keys():
	c = get_capability_contract("acme")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema",
		"rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required <= set(c.keys())


def test_contract_tenant_scoped():
	c = get_capability_contract("remote_corp")
	assert c["configuration"]["tenant_id"] == "remote_corp"


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
		for field in ("name", "path", "component", "permission", "nav_group"):
			assert field in route


def test_provides_and_requires():
	assert len(PROVIDES) >= 5
	assert len(REQUIRES) >= 4
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES
	assert "ntfy" in REQUIRES
	assert "wflo" in REQUIRES


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5
	assert "guardrails" in STREAMING


def test_supported_constants():
	assert len(SUPPORTED_WORK_POLICY_TYPES) >= 4
	assert len(SUPPORTED_VPN_PROTOCOLS) >= 3
	assert len(SUPPORTED_PRODUCTIVITY_METRICS) >= 4
	assert len(SUPPORTED_EQUIPMENT_TYPES) >= 5
	assert len(SUPPORTED_ONBOARDING_STEP_TYPES) >= 5
	assert len(SUPPORTED_COMPLIANCE_CHECK_TYPES) >= 4
	assert len(SUPPORTED_INCIDENT_TYPES) >= 4


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


def test_deny_unsupported_work_policy_type():
	r = evaluate_capability_rules({"operation": "create_work_policy", "policy_type_supported": False})
	assert r["decision"] == "deny"


def test_deny_vpn_no_mfa():
	r = evaluate_capability_rules({"operation": "provision_vpn", "mfa_verified": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "vpn_requires_mfa"


def test_deny_vpn_split_tunneling():
	r = evaluate_capability_rules({"operation": "provision_vpn", "split_tunneling_requested": True})
	assert r["decision"] == "deny"
	assert r["rule"] == "vpn_split_tunneling_denied"


def test_deny_productivity_no_consent():
	r = evaluate_capability_rules({"operation": "record_productivity", "consent_given": False})
	assert r["decision"] == "deny"
	assert r["rule"] == "productivity_tracking_requires_consent"


def test_deny_equipment_limit_exceeded():
	r = evaluate_capability_rules({"operation": "request_equipment", "equipment_limit_exceeded": True})
	assert r["decision"] == "deny"


def test_deny_onboarding_no_manager_approval():
	r = evaluate_capability_rules({"operation": "start_onboarding", "manager_approval_present": False})
	assert r["decision"] == "deny"


def test_deny_policy_acknowledge_draft():
	r = evaluate_capability_rules({"operation": "acknowledge_policy", "policy_state": "draft"})
	assert r["decision"] == "deny"


def test_deny_revoked_vpn_session():
	r = evaluate_capability_rules({"vpn_state": "revoked"})
	assert r["decision"] == "deny"
	assert r["rule"] == "revoked_vpn_blocks_session"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_allow_clean_context():
	r = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert r["decision"] == "allow"
