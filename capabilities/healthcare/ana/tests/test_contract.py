"""Tests for healthcare_ana capability contract."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ana.capability_contract import (
	CAPABILITY_ID, CAPABILITY_VERSION, PROVIDES, REQUIRES, RULES,
	THEME, UI_ROUTES, evaluate_capability_rules, get_capability_contract,
)


def test_contract_top_level_keys():
	contract = get_capability_contract("test_tenant")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "streaming", "provides", "requires"}
	assert required.issubset(contract.keys())


def test_contract_capability_id():
	contract = get_capability_contract()
	assert contract["capability"] == "healthcare_ana"


def test_contract_tenant_id_propagated():
	contract = get_capability_contract("hospital_abc")
	assert contract["configuration"]["tenant_id"] == "hospital_abc"


def test_configuration_schema_required_keys():
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
	assert isinstance(re["rules"], list)
	assert len(re["rules"]) >= 20


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8
	for route in UI_ROUTES:
		assert "name" in route
		assert "path" in route
		assert "component" in route
		assert "permission" in route
		assert "nav_group" in route


def test_theme_tokens():
	required_tokens = {
		"color.primary", "color.accent", "color.success", "color.warning",
		"color.danger", "surface.canvas", "surface.panel", "text.primary",
		"text.secondary", "border.radius", "density",
	}
	assert required_tokens.issubset(THEME["tokens"].keys())


def test_theme_components_present():
	assert len(THEME["components"]) >= 1
	for _name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_count():
	assert len(REQUIRES) >= 4
	assert "auth" in REQUIRES
	assert "audl" in REQUIRES


def test_streaming_structure():
	contract = get_capability_contract()
	streaming = contract["streaming"]
	assert streaming["processor"] == "bytewax"
	assert "stream" in streaming
	assert streaming["key"] == "tenant_id"
	assert len(streaming["events"]) >= 1
	assert len(streaming["guardrails"]) >= 1


def test_evaluate_tenant_context_denied():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "tenant_context_required"


def test_evaluate_write_requires_policy():
	result = evaluate_capability_rules({"operation_type": "write", "policy_attached": False})
	assert result["decision"] == "deny"
	assert result["rule"] == "write_requires_policy"


def test_evaluate_cross_tenant_denied():
	result = evaluate_capability_rules({"cross_tenant_access": True})
	assert result["decision"] == "deny"
	assert result["rule"] == "cross_tenant_data_denied"


def test_evaluate_model_deployment_no_approval():
	result = evaluate_capability_rules({
		"operation": "deploy_model",
		"model_type_supported": True,
		"approval_present": False,
		"auc_above_threshold": True,
	})
	assert result["decision"] == "deny"
	assert "approval" in result["rule"]


def test_evaluate_phi_export_denied():
	result = evaluate_capability_rules({"operation": "export_data", "phi_deidentified": False})
	assert result["decision"] == "deny"
	assert "phi" in result["rule"]


def test_evaluate_allow_by_default():
	result = evaluate_capability_rules({"operation": "view_dashboard", "tenant_context_present": True})
	assert result["decision"] == "allow"
	assert result["rule"] is None


def test_contract_is_deepcopy():
	c1 = get_capability_contract("tenant_a")
	c2 = get_capability_contract("tenant_b")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "tenant_b"
