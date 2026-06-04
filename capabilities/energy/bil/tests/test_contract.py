"""Contract tests for energy_bil capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="bil"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_bil_{name}"
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
	SUPPORTED_TARIFF_TYPES, SUPPORTED_CUSTOMER_CLASSES, SUPPORTED_PAYMENT_METHODS,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_bil"


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for dep in ("auth", "audl", "mten", "conf"):
		assert dep in REQUIRES


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "energy_bil"
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


def test_evaluate_deny_unsupported_tariff_type():
	r = evaluate_capability_rules({"operation": "create_tariff", "tariff_type_supported": False})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "tariff_type_supported" for a in r["actions"])


def test_evaluate_deny_tariff_no_effective_date():
	r = evaluate_capability_rules({
		"operation": "create_tariff",
		"tariff_type_supported": True,
		"customer_class_supported": True,
		"effective_date_present": False,
		"rate_positive": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "tariff_effective_date_required" for a in r["actions"])


def test_evaluate_deny_bill_no_tariff():
	r = evaluate_capability_rules({
		"operation": "generate_bill",
		"billing_cycle_supported": True,
		"tariff_exists": False,
		"meter_reading_present": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "bill_tariff_exists" for a in r["actions"])


def test_evaluate_deny_write_off_without_approval():
	r = evaluate_capability_rules({
		"operation": "write_off_bill",
		"approval_present": False,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "write_off_approval_required" for a in r["actions"])


def test_evaluate_deny_credit_no_approval():
	r = evaluate_capability_rules({
		"operation": "issue_credit",
		"credit_type_supported": True,
		"approval_present": False,
		"expiry_present": True,
	})
	assert r["decision"] == "deny"
	assert any(a["rule"] == "credit_approval_required" for a in r["actions"])


def test_evaluate_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_supported_tariff_types():
	assert "time_of_use" in SUPPORTED_TARIFF_TYPES
	assert "net_metering" in SUPPORTED_TARIFF_TYPES


def test_supported_customer_classes():
	assert "residential" in SUPPORTED_CUSTOMER_CLASSES
	assert "industrial" in SUPPORTED_CUSTOMER_CLASSES


def test_contract_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
