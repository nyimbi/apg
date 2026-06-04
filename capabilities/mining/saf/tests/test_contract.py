"""Tests for mining_saf capability contract."""

from __future__ import annotations

import pytest

from capabilities.mining.saf.capability_contract import (
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
	contract = get_capability_contract("test_tenant")
	required = {"capability", "display_name", "version", "configuration", "configuration_schema", "rule_engine", "ui", "theme", "provides", "requires", "streaming"}
	assert required.issubset(contract.keys())


def test_capability_id():
	assert CAPABILITY_ID == "mining_saf"


def test_tenant_propagated():
	assert get_capability_contract("mine_x")["configuration"]["tenant_id"] == "mine_x"


def test_min_rules():
	assert len(RULES) >= 20


def test_rules_structure():
	for rule in RULES:
		assert "name" in rule and "condition" in rule and "effect" in rule


def test_ui_routes_min():
	assert len(UI_ROUTES) >= 8


def test_ui_routes_prefix():
	for r in UI_ROUTES:
		assert r["path"].startswith("/mining-saf/")


def test_theme_tokens():
	tokens = THEME["tokens"]
	for k in ("color.primary", "border.radius", "surface.canvas", "text.primary"):
		assert k in tokens


def test_provides_min():
	assert len(PROVIDES) >= 5


def test_requires_mandatory():
	for cap in ("auth", "audl", "mten", "conf"):
		assert cap in REQUIRES


def test_streaming_events_include_incident():
	contract = get_capability_contract()
	events = contract["streaming"]["events"]
	assert "incident_reported" in events
	assert "permit_issued" in events


def test_evaluate_tenant_missing():
	r = evaluate_capability_rules({"tenant_context_present": False})
	assert r["decision"] == "deny"
	assert "attach_tenant_context" in r["required_actions"]


def test_evaluate_fatality_notification():
	r = evaluate_capability_rules({
		"operation": "report_incident",
		"incident_severity": "fatality",
		"notification_sent": False,
	})
	assert r["decision"] == "deny"
	assert "send_immediate_notification" in r["required_actions"]


def test_evaluate_lti_close_without_investigation():
	r = evaluate_capability_rules({
		"operation": "close_incident",
		"incident_type": "lost_time_injury",
		"investigation_complete": False,
	})
	assert r["decision"] == "deny"


def test_evaluate_extreme_risk_stop_work():
	r = evaluate_capability_rules({
		"operation": "submit_hazard",
		"risk_rating": "extreme",
		"stop_work_invoked": False,
	})
	assert r["decision"] == "deny"
	assert "invoke_stop_work_authority" in r["required_actions"]


def test_evaluate_expired_ptw():
	r = evaluate_capability_rules({"operation": "access_with_permit", "permit_expired": True})
	assert r["decision"] == "deny"
	assert "renew_or_reissue_permit" in r["required_actions"]


def test_evaluate_unqualified_ptw_issuer():
	r = evaluate_capability_rules({"operation": "issue_permit", "issuer_qualified": False})
	assert r["decision"] == "deny"


def test_evaluate_delete_closed_incident():
	r = evaluate_capability_rules({"operation": "delete", "incident_status": "closed"})
	assert r["decision"] == "deny"
	assert "archive_instead" in r["required_actions"]


def test_evaluate_allow_clean():
	r = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert r["decision"] == "allow"
