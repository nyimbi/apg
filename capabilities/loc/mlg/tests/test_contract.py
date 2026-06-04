"""Tests for MLG capability contract shape, rule evaluation, and UI routes."""

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
	SUPPORTED_CONTENT_TYPES,
	SUPPORTED_DATE_FORMATS,
	SUPPORTED_LANGUAGES,
	SUPPORTED_LOCALES,
	SUPPORTED_NUMBER_FORMATS,
	SUPPORTED_RTL_LANGUAGES,
	SUPPORTED_SCRIPTS,
	SUPPORTED_TEXT_DIRECTIONS,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_capability_id():
	assert CAPABILITY_ID == "loc_mlg"


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required.issubset(set(contract.keys()))


def test_tenant_id_propagated():
	contract = get_capability_contract("mlg_tenant")
	assert contract["configuration"]["tenant_id"] == "mlg_tenant"


def test_configuration_schema_required():
	schema = get_capability_contract()["configuration_schema"]
	assert "tenant_id" in schema["required"]
	assert "ui" in schema["required"]
	assert "theme" in schema["required"]


def test_rule_engine_structure():
	re = get_capability_contract()["rule_engine"]
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


def test_requires_nlpc_for_text_processing():
	assert "nlpc" in REQUIRES


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


def test_evaluate_deny_unsupported_locale():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "configure_locale",
		"locale_supported": False,
		"language_supported": True,
		"script_supported": True,
		"direction_supported": True,
		"date_format_supported": True,
		"number_format_supported": True,
		"rtl_language": False,
		"rtl_direction_set": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "locale_code_supported" for a in result["actions"])


def test_evaluate_deny_rtl_bypass():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "configure_locale",
		"locale_supported": True,
		"language_supported": True,
		"script_supported": True,
		"direction_supported": True,
		"date_format_supported": True,
		"number_format_supported": True,
		"rtl_language": True,
		"rtl_direction_set": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "rtl_bypass_denied" for a in result["actions"])


def test_evaluate_deny_unapproved_publish():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_translation",
		"status_is_approved": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "unapproved_publish_denied" for a in result["actions"])


def test_evaluate_deny_self_review():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "approve_translation",
		"reviewer_present": True,
		"reviewer_is_translator": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "self_review_denied" for a in result["actions"])


def test_evaluate_deny_legal_text_untranslated():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_content",
		"content_type": "legal_text",
		"translation_approved": False,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "untranslated_legal_text_blocked" for a in result["actions"])


def test_evaluate_deny_missing_translator():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "create_translation",
		"source_language_present": True,
		"target_language_present": True,
		"content_type_supported": True,
		"translator_present": False,
		"translation_key_present": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "translation_translator_required" for a in result["actions"])


def test_supported_constants():
	assert len(SUPPORTED_LOCALES) >= 10
	assert len(SUPPORTED_LANGUAGES) >= 5
	assert len(SUPPORTED_SCRIPTS) >= 4
	assert len(SUPPORTED_TEXT_DIRECTIONS) >= 2
	assert len(SUPPORTED_DATE_FORMATS) >= 4
	assert len(SUPPORTED_NUMBER_FORMATS) >= 3
	assert len(SUPPORTED_CONTENT_TYPES) >= 5
	assert len(SUPPORTED_RTL_LANGUAGES) >= 3


def test_contract_is_deepcopy():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
