"""Tests for mining_eqp capability contract."""

from __future__ import annotations

import pytest

from capabilities.mining.eqp.capability_contract import (
	CAPABILITY_ID,
	PROVIDES,
	REQUIRES,
	RULES,
	THEME,
	UI_ROUTES,
	evaluate_capability_rules,
	get_capability_contract,
)


def test_contract_keys():
	contract = get_capability_contract()
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required.issubset(contract.keys())


def test_capability_id():
	assert CAPABILITY_ID == "mining_eqp"


def test_tenant_propagated():
	assert get_capability_contract("fleet_co")["configuration"]["tenant_id"] == "fleet_co"


def test_min_rules():
	assert len(RULES) >= 20


def test_rules_structure():
	for rule in RULES:
		assert all(k in rule for k in ("name", "condition", "effect"))


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_prefix():
	for r in UI_ROUTES:
		assert r["path"].startswith("/mining-eqp/")


def test_theme_tokens_complete():
	for k in ("color.primary", "surface.canvas", "text.primary", "border.radius", "density"):
		assert k in THEME["tokens"]


def test_provides_min():
	assert len(PROVIDES) >= 5


def test_requires_mandatory():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_events():
	contract = get_capability_contract()
	events = contract["streaming"]["events"]
	assert "equipment_breakdown_recorded" in events
	assert "work_order_completed" in events


def test_evaluate_breakdown_dispatch_denied():
	r = evaluate_capability_rules({"operation": "dispatch_equipment", "equipment_status": "breakdown"})
	assert r["decision"] == "deny"
	assert "resolve_breakdown_first" in r["required_actions"]


def test_evaluate_unlicensed_operator():
	r = evaluate_capability_rules({"operation": "dispatch_equipment", "operator_licensed": False})
	assert r["decision"] == "deny"
	assert "verify_operator_licence" in r["required_actions"]


def test_evaluate_unapproved_work_order():
	r = evaluate_capability_rules({"operation": "execute_work_order", "work_order_approved": False})
	assert r["decision"] == "deny"


def test_evaluate_active_equipment_delete():
	r = evaluate_capability_rules({"operation": "delete", "equipment_lifecycle_status": "active"})
	assert r["decision"] == "deny"
	assert "decommission_first" in r["required_actions"]


def test_evaluate_critical_fault_work_order():
	r = evaluate_capability_rules({"operation": "record_critical_fault", "work_order_raised": False})
	assert r["decision"] == "deny"
	assert "raise_work_order" in r["required_actions"]


def test_evaluate_negative_fuel():
	r = evaluate_capability_rules({"operation": "record_fuel", "fuel_quantity_negative": True})
	assert r["decision"] == "deny"


def test_evaluate_pre_shift_inspection_required():
	r = evaluate_capability_rules({"operation": "dispatch_equipment", "pre_shift_inspection_complete": False})
	assert r["decision"] == "deny"


def test_evaluate_allow_clean():
	r = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert r["decision"] == "allow"
