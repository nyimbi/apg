"""Contract tests for energy_ren capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="ren"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_ren_{name}"
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
	SUPPORTED_RENEWABLE_TYPES, SUPPORTED_REC_TYPES, SUPPORTED_CARBON_CREDIT_TYPES,
	SUPPORTED_CURTAILMENT_REASONS, SUPPORTED_FEED_IN_TARIFF_TYPES,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_ren"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for dep in ("auth", "audl", "mten", "conf"):
		assert dep in REQUIRES


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "energy_ren"
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


def test_streaming_bytewax():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_evaluate_allow_empty():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_evaluate_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_evaluate_deny_unsupported_renewable_type():
	r = evaluate_capability_rules({"operation": "register_asset", "renewable_type_supported": False})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "renewable_type_supported" for a in r["actions"])


def test_evaluate_deny_zero_capacity():
	r = evaluate_capability_rules({
		"operation": "register_asset",
		"renewable_type_supported": True,
		"capacity_positive": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "asset_capacity_positive" for a in r["actions"])


def test_evaluate_deny_rec_double_issuance():
	r = evaluate_capability_rules({
		"operation": "issue_rec",
		"rec_type_supported": True,
		"registry_present": True,
		"vintage_year_present": True,
		"rec_already_issued": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "rec_double_issuance_denied" for a in r["actions"])


def test_evaluate_deny_carbon_no_verification():
	r = evaluate_capability_rules({
		"operation": "issue_carbon_credit",
		"credit_type_supported": True,
		"verification_present": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "carbon_credit_verification_required" for a in r["actions"])


def test_evaluate_deny_curtailment_no_mwh():
	r = evaluate_capability_rules({
		"operation": "record_curtailment",
		"curtailment_reason_supported": True,
		"mwh_positive": False,
	})
	assert r["decision"] == "deny"


def test_evaluate_deny_retired_rec_cancel():
	r = evaluate_capability_rules({
		"operation": "cancel_rec",
		"rec_status": "retired",
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "rec_retirement_irreversible" for a in r["actions"])


def test_evaluate_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_supported_renewable_types():
	assert "solar_pv_utility" in SUPPORTED_RENEWABLE_TYPES
	assert "wind_offshore" in SUPPORTED_RENEWABLE_TYPES
	assert "large_hydro" in SUPPORTED_RENEWABLE_TYPES


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
