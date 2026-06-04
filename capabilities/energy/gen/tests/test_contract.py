"""Contract tests for energy_gen capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="gen"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_gen_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

from capability_contract import (
	CAPABILITY_ID, CAPABILITY_VERSION, PROVIDES, REQUIRES, RULES,
	SUPPORTED_PLANT_TYPES, SUPPORTED_DISPATCH_MODES, SUPPORTED_OUTAGE_TYPES,
	SUPPORTED_KPI_TYPES, SUPPORTED_FUEL_TYPES, SUPPORTED_AGENT_RUNTIMES,
	UI_ROUTES, THEME, STREAMING,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "energy_gen"


def test_version_semver():
	parts = CAPABILITY_VERSION.split(".")
	assert len(parts) == 3
	assert all(p.isdigit() for p in parts)


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_contains_core():
	for core in ("auth", "audl", "mten", "conf"):
		assert core in REQUIRES, f"missing required dep: {core}"


def test_contract_shape():
	contract = get_capability_contract("test_org")
	assert contract["capability"] == "energy_gen"
	assert contract["configuration"]["tenant_id"] == "test_org"
	for key in ("display_name", "version", "provides", "requires", "configuration",
	            "configuration_schema", "rule_engine", "ui", "theme", "streaming"):
		assert key in contract, f"missing key: {key}"


def test_configuration_schema_required_fields():
	contract = get_capability_contract()
	schema = contract["configuration_schema"]
	for field in ("tenant_id", "ui", "theme"):
		assert field in schema["required"], f"missing required schema field: {field}"


def test_rule_engine_type():
	contract = get_capability_contract()
	assert contract["rule_engine"]["type"] == "deterministic"
	assert contract["rule_engine"]["default_decision"] == "allow"


def test_rules_count():
	assert len(RULES) >= 20


def test_rules_have_required_keys():
	for rule in RULES:
		assert "name" in rule
		assert "condition" in rule
		assert "effect" in rule
		assert "decision" in rule["effect"]
		assert "reason" in rule["effect"]
		assert "required_action" in rule["effect"]


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_have_required_keys():
	for route in UI_ROUTES:
		for key in ("name", "path", "component", "permission", "nav_group"):
			assert key in route, f"route missing key: {key}"


def test_theme_required_tokens():
	for token in ("color.primary", "color.accent", "color.success", "color.warning",
	              "color.danger", "surface.canvas", "surface.panel", "text.primary",
	              "text.secondary", "border.radius", "density"):
		assert token in THEME["tokens"], f"theme missing token: {token}"


def test_theme_components_present():
	assert len(THEME["components"]) >= 5
	for name, comp in THEME["components"].items():
		assert "icon" in comp
		assert "status_indicator" in comp


def test_streaming_shape():
	assert STREAMING["processor"] == "bytewax"
	assert "stream" in STREAMING
	assert "key" in STREAMING
	assert len(STREAMING["events"]) >= 5
	assert len(STREAMING["guardrails"]) >= 2


def test_supported_plant_types():
	assert len(SUPPORTED_PLANT_TYPES) >= 10


def test_supported_dispatch_modes():
	assert "baseload" in SUPPORTED_DISPATCH_MODES
	assert "economic_dispatch" in SUPPORTED_DISPATCH_MODES


def test_evaluate_allow_on_empty():
	result = evaluate_capability_rules({})
	assert result["decision"] == "allow"
	assert result["actions"] == []


def test_evaluate_deny_missing_tenant():
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "tenant_context_required" for a in result["actions"])


def test_evaluate_deny_write_no_policy():
	result = evaluate_capability_rules({
		"operation_type": "write",
		"policy_attached": False,
		"tenant_context_present": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "write_requires_policy" for a in result["actions"])


def test_evaluate_deny_unsupported_plant_type():
	result = evaluate_capability_rules({
		"operation": "register_plant",
		"plant_type_supported": False,
	})
	assert result["decision"] == "deny"


def test_evaluate_deny_dispatch_exceeds_capacity():
	result = evaluate_capability_rules({
		"operation": "create_dispatch_schedule",
		"mw_within_capacity": False,
		"dispatch_mode_supported": True,
		"plant_exists": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "dispatch_mw_within_capacity" for a in result["actions"])


def test_evaluate_deny_outage_overlap():
	result = evaluate_capability_rules({
		"operation": "schedule_outage",
		"outage_type_supported": True,
		"plant_exists": True,
		"sufficient_notice": True,
		"outage_overlap": True,
	})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "outage_overlap_check" for a in result["actions"])


def test_evaluate_deny_cross_tenant():
	result = evaluate_capability_rules({"cross_tenant_access": True})
	assert result["decision"] == "deny"
	assert any(a["rule"] == "cross_tenant_denied" for a in result["actions"])


def test_evaluate_deny_agent_runtime_unsupported():
	result = evaluate_capability_rules({
		"operation": "register_gen_agent",
		"agent_runtime_supported": False,
	})
	assert result["decision"] == "deny"


def test_contract_deepcopy_isolation():
	c1 = get_capability_contract("t1")
	c2 = get_capability_contract("t2")
	c1["configuration"]["tenant_id"] = "mutated"
	assert c2["configuration"]["tenant_id"] == "t2"
