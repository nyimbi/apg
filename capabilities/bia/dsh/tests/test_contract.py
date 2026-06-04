"""Tests for bia_dsh capability contract."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capability_contract import (
	CAPABILITY_ID, PROVIDES, REQUIRES, UI_ROUTES, THEME, STREAMING, RULES,
	get_capability_contract, evaluate_capability_rules,
)


def test_capability_id():
	assert CAPABILITY_ID == "bia_dsh"


def test_contract_keys():
	c = get_capability_contract("acme")
	for k in ["capability", "display_name", "version", "configuration", "configuration_schema",
	          "rule_engine", "ui", "theme", "streaming", "provides", "requires"]:
		assert k in c


def test_tenant_id_in_config():
	c = get_capability_contract("org1")
	assert c["configuration"]["tenant_id"] == "org1"


def test_rule_count():
	assert len(RULES) >= 20


def test_ui_routes_count():
	assert len(UI_ROUTES) >= 8


def test_theme_tokens():
	for k in ["color.primary", "color.accent", "color.success", "color.warning", "color.danger",
	          "surface.canvas", "surface.panel", "text.primary", "text.secondary", "border.radius", "density"]:
		assert k in THEME["tokens"]


def test_provides_count():
	assert len(PROVIDES) >= 5


def test_requires_core():
	for cap in ["auth", "audl", "mten", "conf"]:
		assert cap in REQUIRES


def test_streaming_structure():
	assert STREAMING["processor"] == "bytewax"
	assert len(STREAMING["events"]) >= 5


def test_deny_no_tenant():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"


def test_deny_cross_tenant():
	r = evaluate_capability_rules({"cross_tenant_access": True})
	assert r["decision"] == "deny"


def test_allow_no_match():
	r = evaluate_capability_rules({"operation": "view_gallery", "tenant_context_present": True})
	assert r["decision"] == "allow"


def test_routes_have_fields():
	for route in UI_ROUTES:
		for f in ["name", "path", "component", "permission", "nav_group"]:
			assert f in route


def test_theme_components():
	assert len(THEME["components"]) >= 3
	for comp in THEME["components"].values():
		assert "icon" in comp and "status_indicator" in comp
