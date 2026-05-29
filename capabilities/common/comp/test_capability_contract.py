"""Regression coverage for the COMP executable capability contract."""

from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.comp import register_capability
from capabilities.common.comp.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.comp.service import CompService


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


def test_service_runs_compliance_lifecycle_with_attested_report():
	service = CompService()

	framework = service.register_framework(
		"fw-soc2",
		"tenant-comp",
		"SOC 2",
		"chief-risk-officer",
		["CC6.1", "CC7.2"],
		"2026.1",
	)
	control = service.create_control(
		"ctrl-access-review",
		"tenant-comp",
		"fw-soc2",
		"Quarterly access review",
		"identity-owner",
		regulated_data_scope=True,
		dlp_policy_linked=True,
	)
	evidence = service.record_evidence(
		"ev-access-review",
		"tenant-comp",
		"ctrl-access-review",
		"access-review-export",
		"auditor",
		encrypted=True,
		immutable_reference="sha256:access-review",
	)
	assessment = service.assess_control(
		"assess-access-review",
		"tenant-comp",
		"ctrl-access-review",
		"ev-access-review",
		"control-tester",
	)
	finding = service.open_finding(
		"finding-privileged-access",
		"tenant-comp",
		"ctrl-access-review",
		"medium",
		"Privileged account review evidence needs manager sign-off.",
		"identity-owner",
		remediation_plan="Collect manager sign-off before next attestation.",
	)
	report = service.prepare_report("report-soc2-q1", "tenant-comp", "fw-soc2", "2026-Q1", "compliance-lead")
	approved = service.approve_report("report-soc2-q1", "tenant-comp", "risk-committee")
	attestation = service.attest_report(
		"attest-soc2-q1",
		"report-soc2-q1",
		"tenant-comp",
		"chief-risk-officer",
		"Control evidence and known findings have been reviewed.",
	)
	published = service.publish_report("report-soc2-q1", "tenant-comp")
	summary = service.dashboard_summary("tenant-comp")

	assert framework["owner"] == "chief-risk-officer"
	assert control["dlp_policy_linked"] is True
	assert evidence["encrypted"] is True
	assert assessment["result"] == "effective"
	assert finding["status"] == "open"
	assert report["finding_count"] == 1
	assert approved["status"] == "approved"
	assert attestation["attested_by"] == "chief-risk-officer"
	assert published["status"] == "published"
	assert summary["framework_count"] == 1
	assert summary["control_count"] == 1
	assert summary["open_finding_count"] == 1
	assert summary["coverage"]["assurance"] == "findings_open"
	assert len(service.list_audit_events("tenant-comp")) >= 8


def test_service_enforces_control_evidence_and_report_guardrails():
	service = CompService()
	service.register_framework("fw-gdpr", "tenant-comp", "GDPR", "privacy-owner", ["Article 32"], "2026.1")

	with pytest.raises(PermissionError, match="control_owner_required"):
		service.create_control("ctrl-ownerless", "tenant-comp", "fw-gdpr", "Ownerless control", "")

	with pytest.raises(PermissionError, match="dlp_policy_required"):
		service.create_control(
			"ctrl-regulated",
			"tenant-comp",
			"fw-gdpr",
			"Regulated personal data export",
			"privacy-owner",
			regulated_data_scope=True,
			dlp_policy_linked=False,
		)

	service.create_control(
		"ctrl-regulated",
		"tenant-comp",
		"fw-gdpr",
		"Regulated personal data export",
		"privacy-owner",
		regulated_data_scope=True,
		dlp_policy_linked=True,
	)
	with pytest.raises(PermissionError, match="encrypted_evidence_required"):
		service.record_evidence("ev-clear", "tenant-comp", "ctrl-regulated", "export", "auditor", encrypted=False, immutable_reference="sha256:clear")
	with pytest.raises(PermissionError, match="immutable_evidence_reference_required"):
		service.record_evidence("ev-mutable", "tenant-comp", "ctrl-regulated", "export", "auditor", encrypted=True)

	stale_time = datetime.now(timezone.utc) - timedelta(days=45)
	service.record_evidence(
		"ev-stale",
		"tenant-comp",
		"ctrl-regulated",
		"export",
		"auditor",
		encrypted=True,
		immutable_reference="sha256:stale",
		collected_at=stale_time,
	)
	with pytest.raises(PermissionError, match="evidence_refresh_required"):
		service.assess_control("assess-stale", "tenant-comp", "ctrl-regulated", "ev-stale", "tester")

	service.prepare_report("report-gdpr", "tenant-comp", "fw-gdpr", "2026-Q1", "privacy-lead")
	with pytest.raises(PermissionError, match="report_approval_required"):
		service.publish_report("report-gdpr", "tenant-comp")


def test_service_escalates_overdue_findings_once():
	service = CompService()
	service.register_framework("fw-pci", "tenant-comp", "PCI DSS", "payments-owner", ["Req 10"], "4.0")
	service.create_control("ctrl-logging", "tenant-comp", "fw-pci", "Payment logging review", "payments-owner")
	service.open_finding(
		"finding-logging",
		"tenant-comp",
		"ctrl-logging",
		"high",
		"Logging review is incomplete.",
		"payments-owner",
		created_at=datetime.now(timezone.utc) - timedelta(days=45),
	)

	first = service.escalate_overdue_findings("tenant-comp")
	second = service.escalate_overdue_findings("tenant-comp")

	assert [item["id"] for item in first] == ["finding-logging"]
	assert second == []
	assert service.dashboard_summary("tenant-comp")["escalated_finding_count"] == 1
