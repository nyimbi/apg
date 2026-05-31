"""Compliance Management package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.comp import api, views
from capabilities.common.comp.service import CompService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_package_contract_shape_and_entrypoint_are_publishable():
	contract_module = _load_module("comp_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("comp_app_runtime", PACKAGE_DIR / "app.py")
	contract = contract_module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "comp"
	assert len(contract["ui"]["routes"]) >= 14
	assert len(contract["rule_engine"]["rules"]) >= 45
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["agents"]["first_class"] is True
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"]
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["comp"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["comp"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert model["capabilities"]["comp"]["runtime"]["service"] == "service.CompService"


def test_api_helpers_execute_compliance_lifecycle():
	api.SERVICE = CompService()
	framework = api.register_framework({"id": "fw-api", "tenant_id": "tenant-api", "name": "SOC 2", "owner": "risk-owner", "obligations": ["CC6.1"], "policy_version": "2026.1"})
	control = api.create_control({"id": "ctrl-api", "tenant_id": "tenant-api", "framework_id": framework["id"], "name": "Access Review", "owner": "identity-owner"})
	evidence = api.record_evidence({"id": "ev-api", "tenant_id": "tenant-api", "control_id": control["id"], "source": "review-export", "collected_by": "auditor", "encrypted": True, "immutable_reference": "sha256:api"})
	assessment = api.assess_control({"id": "assess-api", "tenant_id": "tenant-api", "control_id": control["id"], "evidence_id": evidence["id"], "tested_by": "tester"})
	finding = api.open_finding({"id": "finding-api", "tenant_id": "tenant-api", "control_id": control["id"], "severity": "medium", "description": "Needs follow-up", "owner": "identity-owner", "remediation_plan": "Follow up"})
	resolved = api.resolve_finding({"finding_id": finding["id"], "tenant_id": "tenant-api", "resolved_by": "identity-owner", "resolution": "Done", "evidence_id": evidence["id"]})
	report = api.prepare_report({"id": "report-api", "tenant_id": "tenant-api", "framework_id": framework["id"], "period": "2026-Q1", "prepared_by": "compliance-lead"})
	approved = api.approve_report({"report_id": report["id"], "tenant_id": "tenant-api", "approved_by": "risk-committee"})
	attestation = api.attest_report({"id": "attest-api", "tenant_id": "tenant-api", "report_id": report["id"], "attested_by": "cro", "statement": "Reviewed"})
	published = api.publish_report({"report_id": report["id"], "tenant_id": "tenant-api"})
	agent = api.register_compliance_agent({
		"id": "agent-api",
		"tenant_id": "tenant-api",
		"name": "API Compliance Reviewer",
		"runtime": "pi",
		"role": "control_reviewer",
		"scope": "control:*",
		"owner": "compliance-owner",
		"purpose": "review control drift",
	})
	batch = api.validate_lifecycle_batch({"tenant_id": "tenant-api", "event_stream": "bytewax", "mutation_count": 2, "operation": "control_batch", "batch_id": "batch-api"})
	state = api.compliance_state("tenant-api")

	assert assessment["result"] == "effective"
	assert resolved["status"] == "resolved"
	assert approved["status"] == "approved"
	assert attestation["statement"] == "Reviewed"
	assert published["status"] == "published"
	assert agent["runtime"] == "pi"
	assert batch["status"] == "accepted"
	assert state["summary"]["framework_count"] == 1
	assert state["summary"]["compliance_agent_count"] == 1
	assert state["lifecycle_batches"][0]["id"] == "batch-api"
	assert state["audit_events"]


def test_view_models_match_routes_and_runtime_state():
	service = CompService()
	service.register_framework("fw-view", "tenant-view", "SOC 2", "risk-owner", ["CC6.1"], "2026.1")
	service.create_control("ctrl-view", "tenant-view", "fw-view", "Access Review", "identity-owner")
	service.record_evidence("ev-view", "tenant-view", "ctrl-view", "review-export", "auditor", encrypted=True, immutable_reference="sha256:view")
	service.assess_control("assess-view", "tenant-view", "ctrl-view", "ev-view", "tester")
	service.open_finding("finding-view", "tenant-view", "ctrl-view", "medium", "Needs follow-up", "identity-owner", remediation_plan="Follow up")
	service.prepare_report("report-view", "tenant-view", "fw-view", "2026-Q1", "compliance-lead")

	dashboard = views.dashboard_model(service, "tenant-view")
	frameworks = views.framework_matrix_model(service, "tenant-view")
	controls = views.control_library_model(service, "tenant-view")
	evidence = views.evidence_vault_model(service, "tenant-view")
	assessments = views.assessment_workbench_model(service, "tenant-view")
	findings = views.finding_board_model(service, "tenant-view")
	reports = views.report_builder_model(service, "tenant-view")
	attestations = views.attestation_center_model(service, "tenant-view")
	audit = views.audit_model(service, "tenant-view")
	agents = views.compliance_agent_roster_model(service, "tenant-view")
	lifecycle = views.lifecycle_batch_model(service, "tenant-view")
	settings = views.settings_model("tenant-view")

	assert dashboard["summary"]["control_count"] == 1
	assert frameworks["frameworks"][0]["id"] == "fw-view"
	assert controls["controls"][0]["id"] == "ctrl-view"
	assert evidence["evidence"][0]["id"] == "ev-view"
	assert assessments["assessments"][0]["id"] == "assess-view"
	assert findings["open"][0]["id"] == "finding-view"
	assert reports["drafts"][0]["id"] == "report-view"
	assert attestations["attestations"] == []
	assert agents["agents"] == []
	assert agents["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert lifecycle["required_processor"] == "bytewax"
	assert audit["audit_events"]
	assert settings["theme"]["name"] == "comp_compliance_command_center"


def test_report_publication_blocks_critical_findings_and_self_approval():
	service = CompService()
	service.register_framework("fw-critical", "tenant-critical", "SOC 2", "risk-owner", ["CC6.1"], "2026.1")
	service.create_control("ctrl-critical", "tenant-critical", "fw-critical", "Critical Control", "risk-owner")
	service.record_evidence("ev-critical", "tenant-critical", "ctrl-critical", "critical-export", "auditor", encrypted=True, immutable_reference="sha256:critical")
	service.open_finding("finding-critical", "tenant-critical", "ctrl-critical", "critical", "Critical gap", "risk-owner", remediation_plan="Fix critical gap")
	report = service.prepare_report("report-critical", "tenant-critical", "fw-critical", "2026-Q1", "compliance-lead")

	with pytest.raises(PermissionError, match="independent_report_approval_required"):
		service.approve_report(report["id"], "tenant-critical", "compliance-lead")

	service.approve_report(report["id"], "tenant-critical", "risk-committee")
	service.attest_report("attest-critical", report["id"], "tenant-critical", "cro", "Reviewed")
	with pytest.raises(PermissionError, match="critical_findings_open"):
		service.publish_report(report["id"], "tenant-critical")


def test_agent_and_lifecycle_api_guardrails_are_publishable():
	api.SERVICE = CompService()

	with pytest.raises(PermissionError, match="unsupported_compliance_agent_runtime"):
		api.register_compliance_agent({
			"id": "agent-bad",
			"tenant_id": "tenant-agent",
			"name": "Bad Agent",
			"runtime": "kafka_agent",
			"role": "framework_reviewer",
			"scope": "framework:*",
			"owner": "compliance-owner",
			"purpose": "review frameworks",
		})

	pending = api.register_compliance_agent({
		"id": "agent-report",
		"tenant_id": "tenant-agent",
		"name": "Report Agent",
		"runtime": "codex",
		"role": "report_reviewer",
		"scope": "report:*",
		"owner": "risk-office",
		"purpose": "review report packages",
	})
	with pytest.raises(ValueError, match="comp_lifecycle_batch_empty"):
		api.validate_lifecycle_batch({"tenant_id": "tenant-agent", "event_stream": "bytewax", "mutation_count": 0, "operation": "report_batch"})
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		api.validate_lifecycle_batch({"tenant_id": "tenant-agent", "event_stream": "broker_core", "mutation_count": 1, "operation": "report_batch"})

	assert pending["status"] == "pending_review"
