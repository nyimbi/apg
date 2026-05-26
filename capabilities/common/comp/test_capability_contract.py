"""Regression coverage for the COMP executable capability contract."""

from capabilities.common.comp import register_capability
from capabilities.common.comp.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-comp", {"evidence": {"evidence_freshness_days": 14}})

	assert contract["capability"] == "comp"
	assert contract["configuration"]["tenant_id"] == "tenant-comp"
	assert contract["configuration"]["evidence"]["evidence_freshness_days"] == 14
	assert contract["configuration_schema"]["required"] == ["tenant_id", "frameworks", "controls", "evidence", "reporting", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "frameworks", "controls", "evidence", "findings", "reports", "attestations", "settings"}
	assert contract["ui"]["api_prefix"] == "/comp/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "evidence_vault" in contract["theme"]["components"]


def test_rule_engine_enforces_compliance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_control",
		"control_owner_assigned": False,
		"evidence_age_days": 60,
		"evidence_refresh_completed": False,
		"regulated_data_scope": True,
		"dlp_policy_linked": False,
		"approval_recorded": False,
		"finding_age_days": 45,
		"escalation_recorded": False
	})
	report_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_report", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"control_requires_owner",
		"stale_evidence_requires_refresh",
		"regulated_data_requires_dlp",
		"overdue_finding_requires_escalation"
	}
	assert report_result["matched_rules"] == ["report_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "comp"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "comp_compliance_command_center"
	assert registration["ui_components"]["controls"] == "/comp/controls"
	assert "dlpd" in registration["dependencies"]
	assert "comp:approve_reports" in registration["permissions"]
