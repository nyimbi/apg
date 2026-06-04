"""Tests for mining_ore capability contract."""

from __future__ import annotations

import pytest

from capabilities.mining.ore.capability_contract import (
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
	assert CAPABILITY_ID == "mining_ore"


def test_tenant_propagated():
	assert get_capability_contract("plant_a")["configuration"]["tenant_id"] == "plant_a"


def test_min_rules():
	assert len(RULES) >= 20


def test_rules_structure():
	for rule in RULES:
		assert all(k in rule for k in ("name", "condition", "effect"))


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_prefix():
	for r in UI_ROUTES:
		assert r["path"].startswith("/mining-ore/")


def test_theme_tokens():
	for k in ("color.primary", "surface.canvas", "text.primary", "border.radius"):
		assert k in THEME["tokens"]


def test_provides_min():
	assert len(PROVIDES) >= 5


def test_requires_mandatory():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_events():
	contract = get_capability_contract()
	events = contract["streaming"]["events"]
	assert "metallurgical_balance_approved" in events
	assert "off_spec_product_flagged" in events


def test_evaluate_tenant_missing():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_evaluate_negative_recovery():
	r = evaluate_capability_rules({"operation": "submit_met_balance", "recovery_negative": True})
	assert r["decision"] == "deny"
	assert "review_mass_balance_inputs" in r["required_actions"]


def test_evaluate_recovery_over_100():
	r = evaluate_capability_rules({"operation": "submit_met_balance", "recovery_over_100": True})
	assert r["decision"] == "deny"


def test_evaluate_off_spec_dispatch_denied():
	r = evaluate_capability_rules({"operation": "dispatch_product", "product_meets_spec": False})
	assert r["decision"] == "deny"
	assert "obtain_off_spec_dispatch_approval" in r["required_actions"]


def test_evaluate_unapproved_balance_publication():
	r = evaluate_capability_rules({"operation": "publish_met_balance", "balance_approved": False})
	assert r["decision"] == "deny"


def test_evaluate_cyanide_code_compliance():
	r = evaluate_capability_rules({"operation": "record_cyanide_usage", "cyanide_code_compliant": False})
	assert r["decision"] == "deny"
	assert "verify_cyanide_code_compliance" in r["required_actions"]


def test_evaluate_delete_approved_balance():
	r = evaluate_capability_rules({"operation": "delete", "balance_status": "approved"})
	assert r["decision"] == "deny"


def test_evaluate_allow_clean():
	r = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert r["decision"] == "allow"
