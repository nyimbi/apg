"""Contract tests for energy_dis capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="dis"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_dis_{name}"
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
	SUPPORTED_NETWORK_ELEMENT_TYPES, SUPPORTED_FAULT_TYPES, SUPPORTED_VOLTAGE_LEVELS,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_dis"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for dep in ("auth", "audl", "mten", "conf"):
		assert dep in REQUIRES


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "energy_dis"
	assert c["configuration"]["tenant_id"] == "t1"
	for key in ("rule_engine", "ui", "theme", "streaming"):
		assert key in c


def test_schema_required_fields():
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
	for tok in ("color.primary", "color.accent", "border.radius", "density"):
		assert tok in THEME["tokens"]


def test_streaming_bytewax():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_evaluate_allow_empty():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in r["actions"])


def test_evaluate_deny_unsupported_fault_type():
	r = evaluate_capability_rules({"operation": "report_fault", "fault_type_supported": False})
	assert r["decision"] == "deny"


def test_evaluate_deny_switching_without_approval():
	r = evaluate_capability_rules({
		"operation": "execute_switching",
		"approval_present": False,
		"switching_order_present": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "switching_approval_required" for a in r["actions"])


def test_evaluate_deny_live_network_switching():
	r = evaluate_capability_rules({
		"operation": "execute_switching",
		"network_live": True,
		"approval_present": False,
	})
	assert r["decision"] == "deny"


def test_evaluate_deny_voltage_outside_limits():
	r = evaluate_capability_rules({
		"operation": "load_balance_check",
		"voltage_within_limits": False,
	})
	assert r["decision"] == "deny"


def test_evaluate_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"


def test_supported_element_types():
	assert "feeder" in SUPPORTED_NETWORK_ELEMENT_TYPES
	assert "transformer" in SUPPORTED_NETWORK_ELEMENT_TYPES


def test_supported_fault_types():
	assert "phase_to_ground" in SUPPORTED_FAULT_TYPES
	assert "equipment_failure" in SUPPORTED_FAULT_TYPES
