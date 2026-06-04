"""Contract tests for energy_met capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="met"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_met_{name}"
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
	SUPPORTED_METER_TYPES, SUPPORTED_TAMPER_TYPES, SUPPORTED_COMMAND_TYPES,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_met"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for dep in ("auth", "audl", "mten", "conf"):
		assert dep in REQUIRES


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "energy_met"
	assert c["configuration"]["tenant_id"] == "t1"


def test_schema_required():
	c = get_capability_contract()
	for f in ("tenant_id", "ui", "theme"):
		assert f in c["configuration_schema"]["required"]


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_theme_tokens():
	for tok in ("color.primary", "border.radius", "density"):
		assert tok in THEME["tokens"]


def test_streaming():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_evaluate_allow_empty():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_evaluate_deny_unsupported_meter_type():
	r = evaluate_capability_rules({"operation": "register_meter", "meter_type_supported": False})
	assert r["decision"] == "deny"


def test_evaluate_deny_disconnect_without_approval():
	r = evaluate_capability_rules({
		"operation": "issue_command",
		"command_type_supported": True,
		"command_is_disconnect": True,
		"approval_present": False,
		"meter_active": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "disconnect_approval_required" for a in r["actions"])


def test_evaluate_deny_dr_opt_out():
	r = evaluate_capability_rules({
		"operation": "activate_dr_event",
		"customer_opted_out": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "dr_opt_out_respected" for a in r["actions"])


def test_evaluate_deny_tamper_no_evidence():
	r = evaluate_capability_rules({
		"operation": "report_tamper",
		"tamper_type_supported": True,
		"evidence_present": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "tamper_evidence_required" for a in r["actions"])


def test_evaluate_deny_firmware_without_approval():
	r = evaluate_capability_rules({
		"operation": "issue_command",
		"command_type_supported": True,
		"command_is_firmware": True,
		"approval_present": False,
		"meter_active": True,
		"command_is_disconnect": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "firmware_update_approval_required" for a in r["actions"])


def test_evaluate_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_supported_tamper_types():
	assert "magnetic_tamper" in SUPPORTED_TAMPER_TYPES
	assert "bypass_detected" in SUPPORTED_TAMPER_TYPES


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
