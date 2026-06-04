"""Tests for MCY capability contract shape, rule evaluation, and UI routes."""

from __future__ import annotations

import sys
import os

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CAP_DIR)

from capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	SUPPORTED_CURRENCIES,
	SUPPORTED_FX_ACCOUNT_TYPES,
	SUPPORTED_RATE_SOURCES,
	SUPPORTED_RATE_TYPES,
	SUPPORTED_REVALUATION_METHODS,
	SUPPORTED_ROUNDING_MODES,
	SUPPORTED_TRANSLATION_METHODS,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "loc_mcy"


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required.issubset(set(contract.keys()))


def test_tenant_id_propagated():
	contract = get_capability_contract("mcy_tenant")
	assert contract["configuration"]["tenant_id"] == "mcy_tenant"


def test_configuration_schema_required():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_rule_engine_structure():
	contract = get_capability_contract()
	re = contract["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert len(re["rules"]) >= 20


def test_rules_have_required_fields():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_provides_minimum_count():
	assert len(PROVIDES) >= 5


def test_requires_includes_core():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_ui_routes_minimum_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_structure():
	for route in UI_ROUTES:
		for key in ("name", "path", "component", "permission", "nav_group"):
			assert key in route


def test_theme_tokens():
	tokens = get_capability_contract()["theme"]["tokens"]
	required = {"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
				"surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"}
	assert required.issubset(set(tokens.keys()))


def test_theme_components():
	components = get_capability_contract()["theme"]["components"]
	assert len(components) >= 5
	for comp in components.values():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_streaming_structure():
	s = get_capability_contract()["streaming"]
	assert s["processor"] == "bytewax"
	assert len(s["events"]) >= 5
	assert len(s["guardrails"]) >= 3


def test_evaluate_allow_clean():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_write_no_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False, "tenant_context_present": True})
	assert result["decision"] == "deny"


def test_evaluate_deny_unsupported_currency():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "configure_currency",
		"currency_supported": False,
		"currency_name_present": True,
		"precision_valid": True,
		"rounding_mode_supported": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "currency_code_supported" for a in result["actions"])


def test_evaluate_deny_manual_rate_no_approval():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "record_rate",
		"from_currency_supported": True,
		"to_currency_supported": True,
		"rate_type_supported": True,
		"rate_source_supported": True,
		"effective_date_present": True,
		"rate_positive": True,
		"rate_source": "manual",
		"approval_present": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "manual_rate_approval_required" for a in result["actions"])


def test_evaluate_deny_unapproved_revaluation_post():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "post_revaluation",
		"approval_present": False,
		"fx_account_bypass": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "unapproved_revaluation_posting_denied" for a in result["actions"])


def test_evaluate_deny_unapproved_translation_post():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "post_translation",
		"approval_present": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "unapproved_translation_posting_denied" for a in result["actions"])


def test_evaluate_deny_fx_account_bypass():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "post_revaluation",
		"approval_present": True,
		"fx_account_bypass": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "fx_gain_loss_account_bypass_denied" for a in result["actions"])


def test_evaluate_deny_backdating_no_override():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "record_rate",
		"from_currency_supported": True,
		"to_currency_supported": True,
		"rate_type_supported": True,
		"rate_source_supported": True,
		"effective_date_present": True,
		"rate_positive": True,
		"rate_source": "central_bank",
		"approval_present": True,
		"backdated": True,
		"backdating_override_present": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "rate_backdating_restricted" for a in result["actions"])


def test_supported_constants_nonempty():
	assert len(SUPPORTED_CURRENCIES) >= 10
	assert len(SUPPORTED_RATE_TYPES) >= 5
	assert len(SUPPORTED_RATE_SOURCES) >= 4
	assert len(SUPPORTED_REVALUATION_METHODS) >= 3
	assert len(SUPPORTED_TRANSLATION_METHODS) >= 3
	assert len(SUPPORTED_ROUNDING_MODES) >= 4
	assert len(SUPPORTED_FX_ACCOUNT_TYPES) >= 4


def test_contract_is_deepcopy():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
