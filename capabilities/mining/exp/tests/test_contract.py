"""Tests for mining_exp capability contract."""

from __future__ import annotations

import pytest

from capabilities.mining.exp.capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	THEME,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required_keys = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required_keys.issubset(contract.keys())


def test_capability_id():
	assert CAPABILITY_ID == "mining_exp"


def test_tenant_id_propagated():
	contract = get_capability_contract("acme_mining")
	assert contract["configuration"]["tenant_id"] == "acme_mining"


def test_default_tenant():
	contract = get_capability_contract()
	assert contract["configuration"]["tenant_id"] == "default"


def test_configuration_schema_required_keys():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	assert "required" in schema
	for key in ("tenant_id", "ui", "theme"):
		assert key in schema["required"]


def test_rule_engine_structure():
	contract = get_capability_contract()
	re = contract["rule_engine"]
	assert re["type"] == "deterministic"
	assert re["default_decision"] == "allow"
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_rules_have_required_fields():
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
		assert route["path"].startswith("/mining-exp/")


def test_theme_required_tokens():
	tokens = THEME["tokens"]
	required = {"color.primary", "color.accent", "color.success", "color.warning", "color.danger",
				"surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"}
	assert required.issubset(tokens.keys())


def test_theme_components_not_empty():
	assert len(THEME["components"]) >= 5
	for comp_name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_contains_mandatory():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_structure():
	contract = get_capability_contract()
	streaming = contract["streaming"]
	assert streaming["processor"] == "bytewax"
	assert "stream" in streaming
	assert streaming["key"] == "tenant_id"
	assert len(streaming["events"]) >= 5
	assert len(streaming["guardrails"]) >= 3


def test_evaluate_rules_tenant_context_missing():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(r["rule"] == "tenant_context_required" for r in result["matched_denials"])
	assert "attach_tenant_context" in result["required_actions"]


def test_evaluate_rules_write_no_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"
	assert "attach_write_policy" in result["required_actions"]


def test_evaluate_rules_assay_requires_collar():
	result = evaluate_capability_rules({"operation": "import_assays", "collar_exists": False})
	assert result["decision"] == "deny"
	assert "create_collar_first" in result["required_actions"]


def test_evaluate_rules_resource_competent_person():
	result = evaluate_capability_rules({"operation": "submit_resource_estimate", "competent_person_present": False})
	assert result["decision"] == "deny"
	assert "assign_competent_person" in result["required_actions"]


def test_evaluate_rules_allow_benign_context():
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] == "allow"
	assert result["matched_denials"] == []


def test_evaluate_rules_qaqc_bypass_denied():
	result = evaluate_capability_rules({"operation": "bypass_qaqc_check", "has_override_authority": False})
	assert result["decision"] == "deny"


def test_evaluate_rules_report_sign_off():
	result = evaluate_capability_rules({"operation": "publish_compliance_report", "competent_person_signed": False})
	assert result["decision"] == "deny"
	assert "obtain_competent_person_signature" in result["required_actions"]


def test_evaluate_rules_interval_overlap():
	result = evaluate_capability_rules({"operation": "import_assays", "interval_overlap_detected": True})
	assert result["decision"] == "deny"
	assert "resolve_interval_overlap" in result["required_actions"]


def test_contract_immutable_between_calls():
	c1 = get_capability_contract("tenant_a")
	c2 = get_capability_contract("tenant_b")
	assert c1["configuration"]["tenant_id"] == "tenant_a"
	assert c2["configuration"]["tenant_id"] == "tenant_b"
