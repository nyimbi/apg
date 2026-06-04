"""Contract tests for energy_grd capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="grd"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_grd_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

from capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, RULES, UI_ROUTES, THEME, STREAMING,
	SUPPORTED_STATE_ESTIMATOR_TYPES, SUPPORTED_CONTINGENCY_TYPES,
	SUPPORTED_VOLTAGE_CONTROL_METHODS, SUPPORTED_FREQUENCY_CONTROL_METHODS,
	SUPPORTED_MARKET_PRODUCTS, SUPPORTED_EMS_FUNCTIONS,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_grd"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for dep in ("auth", "audl", "mten", "conf"):
		assert dep in REQUIRES


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "energy_grd"
	assert c["configuration"]["tenant_id"] == "t1"
	for key in ("rule_engine", "ui", "theme", "streaming"):
		assert key in c


def test_schema_required():
	c = get_capability_contract()
	for f in ("tenant_id", "ui", "theme"):
		assert f in c["configuration_schema"]["required"]


def test_rules_count():
	assert len(RULES) >= 20


def test_rules_structure():
	for r in RULES:
		assert "name" in r and "condition" in r and "effect" in r


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_theme_tokens():
	for tok in ("color.primary", "border.radius", "density"):
		assert tok in THEME["tokens"]


def test_streaming_bytewax():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_evaluate_allow_empty():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_evaluate_deny_se_no_model():
	r = evaluate_capability_rules({
		"operation": "run_state_estimation",
		"se_type_supported": True,
		"network_model_present": False,
		"measurements_present": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "se_network_model_required" for a in r["actions"])


def test_evaluate_deny_contingency_no_base_case():
	r = evaluate_capability_rules({
		"operation": "run_contingency",
		"contingency_type_supported": True,
		"base_case_converged": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "contingency_base_case_required" for a in r["actions"])


def test_evaluate_deny_voltage_control_no_approval():
	r = evaluate_capability_rules({
		"operation": "apply_voltage_control",
		"control_method_supported": True,
		"approval_present": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "voltage_control_approval_required" for a in r["actions"])


def test_evaluate_deny_settlement_no_metered_data():
	r = evaluate_capability_rules({
		"operation": "settle_market_interval",
		"product_supported": True,
		"metered_data_present": False,
		"bid_offer_present": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "market_metered_data_required" for a in r["actions"])


def test_evaluate_deny_critical_alarm_clear_without_ack():
	r = evaluate_capability_rules({
		"operation": "clear_alarm",
		"alarm_severity": "critical",
		"acknowledged": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "critical_alarm_acknowledgement_required" for a in r["actions"])


def test_evaluate_deny_n1_bypass():
	r = evaluate_capability_rules({
		"operation": "skip_n1_contingency",
		"n1_bypass_allowed": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "n1_contingency_mandatory" for a in r["actions"])


def test_evaluate_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_supported_se_types():
	assert "weighted_least_squares" in SUPPORTED_STATE_ESTIMATOR_TYPES


def test_supported_market_products():
	assert "energy" in SUPPORTED_MARKET_PRODUCTS
	assert "spinning_reserve" in SUPPORTED_MARKET_PRODUCTS


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
