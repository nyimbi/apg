"""Regression coverage for the COMP executable capability contract."""

from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.comp import register_capability
from capabilities.common.comp import api
from capabilities.common.comp import views
from capabilities.common.comp.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.comp.service import CompService


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-comp", {"evidence": {"evidence_freshness_days": 14}})

	assert contract["capability"] == "comp"
	assert contract["configuration"]["tenant_id"] == "tenant-comp"
	assert contract["configuration"]["evidence"]["evidence_freshness_days"] == 14
	assert set(contract["configuration_schema"]["required"]) >= {
		"tenant_id",
		"frameworks",
		"controls",
		"evidence",
		"assessments",
		"findings",
		"reporting",
		"exceptions",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	}
	assert len(contract["rule_engine"]["rules"]) >= 45
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "frameworks", "controls", "evidence", "assessments", "findings", "exceptions", "reports", "attestations", "exports", "audit", "agents", "lifecycle", "settings"}
	assert contract["ui"]["api_prefix"] == "/comp/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "assessment_workbench" in contract["theme"]["components"]
	assert "compliance_agent_roster" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["agent_adapter"] == "aicr_provider_neutral_compliance_agent_adapter"
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "compliance_agent_batch" in contract["streaming"]["required_operations"]


def test_rule_engine_enforces_compliance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_control",
		"framework_present": False,
		"control_name_present": False,
		"control_owner_assigned": False,
		"testing_frequency_days": 0,
		"regulated_data_scope": True,
		"dlp_policy_linked": False,
		"approval_recorded": False,
		"finding_age_days": 45,
		"escalation_recorded": False,
	})
	report_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_report", "approval_recorded": False, "attestation_recorded": False})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_compliance_mutation", "event_stream": "kafka"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_compliance_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
	})
	privileged_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_compliance_agent",
		"agent_runtime_supported": True,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "validate_comp_lifecycle_batch", "event_stream": "kafka", "mutation_count": 1})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"control_requires_framework",
		"control_requires_name",
		"control_requires_owner",
		"control_frequency_requires_positive_days",
		"regulated_data_requires_dlp",
		"overdue_finding_requires_escalation",
	}
	assert report_result["matched_rules"] == ["report_requires_approval", "report_requires_attestation"]
	assert stream_result["decision"] == "deny"
	assert "batch_compliance_mutation_requires_bytewax" in stream_result["matched_rules"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"compliance_agent_runtime_supported",
		"compliance_agent_role_supported",
		"compliance_agent_requires_scope",
		"compliance_agent_requires_owner",
		"compliance_agent_requires_purpose",
		"compliance_agent_requires_contribution_disclosure",
	}
	assert privileged_result["decision"] == "require_review"
	assert privileged_result["matched_rules"] == ["compliance_agent_privileged_role_requires_human_approval"]
	assert lifecycle_result["decision"] == "deny"
	assert lifecycle_result["matched_rules"] == ["bytewax_comp_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "comp"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "comp_compliance_command_center"
	assert registration["ui_components"]["controls"] == "/comp/controls"
	assert registration["ui_components"]["audit"] == "/comp/audit"
	assert registration["ui_components"]["agents"] == "/comp/agents"
	assert "dlpd" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["lifecycle_stream"] == "comp.lifecycle"
	assert "comp:approve_reports" in registration["permissions"]
	assert "comp:audit" in registration["permissions"]


def test_service_runs_compliance_lifecycle_with_attested_report_and_views():
	service = CompService()

	framework = service.register_framework("fw-soc2", "tenant-comp", "SOC 2", "chief-risk-officer", ["CC6.1", "CC7.2"], "2026.1")
	control = service.create_control(
		"ctrl-access-review",
		"tenant-comp",
		"fw-soc2",
		"Quarterly access review",
		"identity-owner",
		regulated_data_scope=True,
		dlp_policy_linked=True,
	)
	evidence = service.record_evidence("ev-access-review", "tenant-comp", "ctrl-access-review", "access-review-export", "auditor", encrypted=True, immutable_reference="sha256:access-review")
	assessment = service.assess_control("assess-access-review", "tenant-comp", "ctrl-access-review", "ev-access-review", "control-tester")
	finding = service.open_finding("finding-privileged-access", "tenant-comp", "ctrl-access-review", "medium", "Privileged account review evidence needs manager sign-off.", "identity-owner", remediation_plan="Collect manager sign-off before next attestation.")
	report = service.prepare_report("report-soc2-q1", "tenant-comp", "fw-soc2", "2026-Q1", "compliance-lead")
	approved = service.approve_report("report-soc2-q1", "tenant-comp", "risk-committee")
	attestation = service.attest_report("attest-soc2-q1", "report-soc2-q1", "tenant-comp", "chief-risk-officer", "Control evidence and known findings have been reviewed.")
	published = service.publish_report("report-soc2-q1", "tenant-comp")
	agent = service.register_compliance_agent("agent-steward", "tenant-comp", "Compliance Steward", "codex", "compliance_steward", "framework:fw-soc2", "chief-risk-officer", "review framework posture", human_approval_required=True)
	batch = service.validate_comp_lifecycle_batch("tenant-comp", "bytewax", 3, "compliance_agent_batch", "batch-agent")
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
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert summary["framework_count"] == 1
	assert summary["control_count"] == 1
	assert summary["open_finding_count"] == 1
	assert summary["coverage"]["assurance"] == "findings_open"
	assert summary["compliance_agent_count"] == 1
	assert summary["lifecycle_batch_count"] == 1
	assert views.framework_matrix_model(service, "tenant-comp")["frameworks"][0]["id"] == "fw-soc2"
	assert views.assessment_workbench_model(service, "tenant-comp")["assessments"][0]["id"] == "assess-access-review"
	assert views.report_builder_model(service, "tenant-comp")["published"][0]["id"] == "report-soc2-q1"
	assert views.compliance_agent_roster_model(service, "tenant-comp")["active"][0]["id"] == "agent-steward"
	assert views.lifecycle_batch_model(service, "tenant-comp")["accepted"][0]["id"] == "batch-agent"
	assert len(views.audit_model(service, "tenant-comp")["audit_events"]) >= 10


def test_service_enforces_control_evidence_and_report_guardrails():
	service = CompService()
	service.register_framework("fw-gdpr", "tenant-comp", "GDPR", "privacy-owner", ["Article 32"], "2026.1")

	with pytest.raises(PermissionError, match="control_owner_required"):
		service.create_control("ctrl-ownerless", "tenant-comp", "fw-gdpr", "Ownerless control", "")

	with pytest.raises(PermissionError, match="dlp_policy_required"):
		service.create_control("ctrl-regulated", "tenant-comp", "fw-gdpr", "Regulated personal data export", "privacy-owner", regulated_data_scope=True, dlp_policy_linked=False)

	service.create_control("ctrl-regulated", "tenant-comp", "fw-gdpr", "Regulated personal data export", "privacy-owner", regulated_data_scope=True, dlp_policy_linked=True)
	with pytest.raises(PermissionError, match="encrypted_evidence_required"):
		service.record_evidence("ev-clear", "tenant-comp", "ctrl-regulated", "export", "auditor", encrypted=False, immutable_reference="sha256:clear")
	with pytest.raises(PermissionError, match="immutable_evidence_reference_required"):
		service.record_evidence("ev-mutable", "tenant-comp", "ctrl-regulated", "export", "auditor", encrypted=True)

	stale_time = datetime.now(timezone.utc) - timedelta(days=45)
	service.record_evidence("ev-stale", "tenant-comp", "ctrl-regulated", "export", "auditor", encrypted=True, immutable_reference="sha256:stale", collected_at=stale_time)
	with pytest.raises(PermissionError, match="evidence_refresh_required"):
		service.assess_control("assess-stale", "tenant-comp", "ctrl-regulated", "ev-stale", "tester")

	service.prepare_report("report-gdpr", "tenant-comp", "fw-gdpr", "2026-Q1", "privacy-lead")
	with pytest.raises(PermissionError, match="report_approval_required"):
		service.publish_report("report-gdpr", "tenant-comp")


def test_service_escalates_and_resolves_findings_with_evidence():
	service = CompService()
	service.register_framework("fw-pci", "tenant-comp", "PCI DSS", "payments-owner", ["Req 10"], "4.0")
	service.create_control("ctrl-logging", "tenant-comp", "fw-pci", "Payment logging review", "payments-owner")
	service.record_evidence("ev-logging", "tenant-comp", "ctrl-logging", "logging-export", "auditor", encrypted=True, immutable_reference="sha256:logging")
	service.open_finding("finding-logging", "tenant-comp", "ctrl-logging", "high", "Logging review is incomplete.", "payments-owner", created_at=datetime.now(timezone.utc) - timedelta(days=45), remediation_plan="Complete logging review.")

	first = service.escalate_overdue_findings("tenant-comp")
	second = service.escalate_overdue_findings("tenant-comp")
	with pytest.raises(PermissionError, match="finding_resolution_evidence_required"):
		service.resolve_finding("finding-logging", "tenant-comp", "payments-owner", "Review completed.")
	resolved = service.resolve_finding("finding-logging", "tenant-comp", "payments-owner", "Review completed.", evidence_id="ev-logging")

	assert [item["id"] for item in first] == ["finding-logging"]
	assert second == []
	assert resolved["status"] == "resolved"
	assert service.dashboard_summary("tenant-comp")["escalated_finding_count"] == 0


def test_tenant_local_ids_do_not_collide_and_cross_tenant_lookups_fail():
	service = CompService()
	alpha = service.register_framework("shared", "tenant-alpha", "Shared", "alpha-owner", ["A1"], "1")
	beta = service.register_framework("shared", "tenant-beta", "Shared", "beta-owner", ["B1"], "1")
	service.create_control("shared-control", "tenant-alpha", "shared", "Alpha Control", "alpha-owner")
	service.create_control("shared-control", "tenant-beta", "shared", "Beta Control", "beta-owner")

	assert alpha["id"] == beta["id"] == "shared"
	assert service.list_frameworks("tenant-alpha")[0]["owner"] == "alpha-owner"
	assert service.list_controls("tenant-beta")[0]["owner"] == "beta-owner"
	with pytest.raises(KeyError, match="control_not_found"):
		service.record_evidence("ev-cross", "tenant-alpha", "missing-control", "source", "auditor", encrypted=True, immutable_reference="sha256:x")


def test_service_and_api_enforce_compliance_agent_and_lifecycle_guardrails():
	service = CompService()

	with pytest.raises(PermissionError, match="unsupported_compliance_agent_runtime"):
		service.register_compliance_agent("agent-unsupported", "tenant-comp", "Unsupported", "kafka_agent", "framework_reviewer", "framework:*", "owner", "review frameworks")

	with pytest.raises(PermissionError, match="compliance_agent_contribution_disclosure_required"):
		service.register_compliance_agent("agent-undisclosed", "tenant-comp", "Undisclosed", "codex", "framework_reviewer", "framework:*", "owner", "review frameworks", contribution_disclosed=False)

	pending = service.register_compliance_agent(
		"agent-report-reviewer",
		"tenant-comp",
		"Report Reviewer",
		"claude_code",
		"report_reviewer",
		"report:*",
		"risk-office",
		"review regulatory report drafts",
	)

	with pytest.raises(ValueError, match="comp_lifecycle_batch_empty"):
		service.validate_comp_lifecycle_batch("tenant-comp", "bytewax", 0, "report_batch")
	with pytest.raises(ValueError, match="unsupported_comp_lifecycle_operation"):
		service.validate_comp_lifecycle_batch("tenant-comp", "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_comp_lifecycle_batch("tenant-comp", "kafka", 1, "report_batch")

	api.SERVICE = CompService()
	api_agent = api.register_compliance_agent({
		"id": "agent-api",
		"tenant_id": "tenant-api",
		"name": "API Agent",
		"runtime": "opencode",
		"role": "framework_reviewer",
		"scope": "framework:*",
		"owner": "compliance-owner",
		"purpose": "review framework drift",
	})
	api_batch = api.validate_lifecycle_batch({"tenant_id": "tenant-api", "event_stream": "bytewax", "mutation_count": 2, "operation": "framework_batch", "batch_id": "batch-api"})
	status = api.capability_status("tenant-api")
	state = api.compliance_state("tenant-api")

	assert pending["status"] == "pending_review"
	assert pending["human_approval_required"] is False
	assert api_agent["runtime"] == "opencode"
	assert api_batch["accepted"] is True
	assert status["agents"]["first_class"] is True
	assert status["lifecycle_batch_count"] == 1
	assert state["compliance_agents"][0]["id"] == "agent-api"
	assert state["lifecycle_batches"][0]["id"] == "batch-api"
