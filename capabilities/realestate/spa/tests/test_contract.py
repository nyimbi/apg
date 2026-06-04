"""Tests for realestate_spa capability contract."""

from __future__ import annotations

from capabilities.realestate.spa.capability_contract import (
	get_capability_contract, evaluate_capability_rules,
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, RULES, THEME, STREAMING,
)


def test_contract_shape():
	c = get_capability_contract("t1")
	assert c["capability"] == "realestate_spa"
	assert c["display_name"] == "Space Planning & Management"


def test_tenant_isolation():
	assert get_capability_contract("a")["configuration"]["tenant_id"] == "a"


def test_rules_count():
	assert len(RULES) >= 20


def test_ui_routes():
	assert len(UI_ROUTES) >= 8
	for rt in UI_ROUTES:
		assert rt["permission"].startswith("realestate_spa:")


def test_theme_tokens():
	for t in ("color.primary", "surface.panel", "border.radius", "density"):
		assert t in THEME["tokens"]


def test_streaming():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_provides_requires():
	assert len(PROVIDES) >= 5
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_allow_default():
	assert evaluate_capability_rules({})["decision"] == "allow"


def test_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_deny_space_no_floor_plan():
	r = evaluate_capability_rules({"operation": "create_space", "floor_plan_linked": False})
	assert r["decision"] == "deny"


def test_deny_double_booking():
	r = evaluate_capability_rules({"operation": "book_space", "space_already_booked": True})
	assert r["decision"] == "deny"


def test_deny_booking_decommissioned_space():
	r = evaluate_capability_rules({"operation": "book_space", "space_status": "decommissioned"})
	assert r["decision"] == "deny"


def test_deny_large_move_no_approval():
	r = evaluate_capability_rules({"operation": "create_move", "headcount_above_threshold": True, "approved": False})
	assert r["decision"] == "deny"


def test_deny_sensor_data_not_anonymised():
	r = evaluate_capability_rules({"operation": "ingest_sensor_data", "data_anonymised": False})
	assert r["decision"] == "deny"


def test_deny_chargeback_unverified_data():
	r = evaluate_capability_rules({"operation": "calculate_chargeback", "occupancy_data_verified": False})
	assert r["decision"] == "deny"


def test_deny_booking_too_far_ahead():
	r = evaluate_capability_rules({"operation": "create_booking", "booking_too_far_in_advance": True})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"operation_type": "write", "cross_tenant": True})
	assert r["decision"] == "deny"
