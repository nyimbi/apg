"""Executable FinTech Compliance Automation capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_fintech_compliance", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_compliance"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "compliance_agent_workflow" in contract["provides"]
	assert "/fintech-compliance/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_compliance", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "compliance_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "compliance_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_compliance_lifecycle():
	service_module = _load_module("service_fintech_compliance", PACKAGE_DIR / "service.py")
	service = service_module.ComplianceAutomationService()

	obligation = service.register_obligation("obl-1", "tenant-test", "pci_dss", "control", "Protect card data", "owner-1", "policy-1", "2026-06-01")
	control = service.map_control("control-1", "tenant-test", obligation["id"], "preventive", "control-owner", "control-evidence", "monthly")
	check = service.record_check("check-1", "tenant-test", obligation["id"], control["id"], "transaction", "payment-1", "compliant")
	evidence = service.attach_evidence("evidence-1", "tenant-test", check["id"], "control_log", "system-1", 365)
	attestation = service.record_attestation("attest-1", "tenant-test", obligation["id"], "attestor-1", "compliant", evidence["id"])
	issue = service.open_issue("issue-1", "tenant-test", obligation["id"], "medium", "issue-owner", "issue-evidence", "2026-07-01")
	remediation = service.record_remediation("remediation-1", "tenant-test", issue["id"], "remediation-owner", "plan-1")
	report = service.publish_report("report-1", "tenant-test", "regulatory", "pci_dss", "2026-Q2", "report-evidence", "approver-1")
	review = service.record_review("review-1", "tenant-test", report["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_compliance_agent("agent-1", "tenant-test", "Compliance Agent", "codex", "evidence_reviewer", "review evidence")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert obligation["framework"] == "pci_dss"
	assert control["control_type"] == "preventive"
	assert attestation["status"] == "compliant"
	assert remediation["status"] == "active"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["obligation_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_compliance_actions():
	service_module = _load_module("guardrail_service_fintech_compliance", PACKAGE_DIR / "service.py")
	service = service_module.ComplianceAutomationService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_obligation("obl", "", "pci_dss", "control", "Title", "owner", "evidence", "2026-06-01")
	with pytest.raises(PermissionError, match="framework_not_supported"):
		service.register_obligation("obl", "tenant-test", "unknown", "control", "Title", "owner", "evidence", "2026-06-01")
	with pytest.raises(PermissionError, match="effective_date_required"):
		service.register_obligation("obl", "tenant-test", "pci_dss", "control", "Title", "owner", "evidence", "")
	obligation = service.register_obligation("obl-ok", "tenant-test", "pci_dss", "control", "Title", "owner", "evidence", "2026-06-01")
	with pytest.raises(PermissionError, match="control_type_not_supported"):
		service.map_control("control", "tenant-test", obligation["id"], "unknown", "owner", "evidence", "monthly")
	control = service.map_control("control-ok", "tenant-test", obligation["id"], "preventive", "owner", "evidence", "monthly")
	with pytest.raises(PermissionError, match="failed_check_evidence_required"):
		service.record_check("check", "tenant-test", obligation["id"], control["id"], "transaction", "subject", "failed")
	with pytest.raises(PermissionError, match="retention_period_required"):
		service.attach_evidence("evidence", "tenant-test", control["id"], "control_log", "source", 0)
	with pytest.raises(PermissionError, match="attestor_required"):
		service.record_attestation("attest", "tenant-test", obligation["id"], "", "compliant", "evidence")
	with pytest.raises(PermissionError, match="issue_due_date_required"):
		service.open_issue("issue", "tenant-test", obligation["id"], "medium", "owner", "evidence", "")
	issue = service.open_issue("issue-ok", "tenant-test", obligation["id"], "medium", "owner", "evidence", "2026-07-01")
	with pytest.raises(PermissionError, match="remediation_approval_required"):
		service.record_remediation("remediation", "tenant-test", issue["id"], "owner", "plan", high_impact=True)
	with pytest.raises(PermissionError, match="report_approver_required"):
		service.publish_report("report", "tenant-test", "regulatory", "pci_dss", "2026-Q2", "evidence", "")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", obligation["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="compliance_agent_runtime_not_supported"):
		service.register_compliance_agent("agent", "tenant-test", "Bad Agent", "unsupported", "evidence_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_compliance", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_compliance", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_compliance", PACKAGE_DIR / "app.py")

	obligation = api.register_obligation({"tenant_id": "tenant-api", "obligation_id": "api-obl", "framework": "pci_dss", "obligation_type": "control", "title": "Protect data", "owner_id": "owner", "evidence_reference": "evidence", "effective_date": "2026-06-01"})
	control = api.map_control({"tenant_id": "tenant-api", "control_id": "api-control", "obligation_id": obligation["id"], "control_type": "preventive", "owner_id": "owner", "evidence_reference": "evidence", "frequency": "monthly"})
	api.record_check({"tenant_id": "tenant-api", "check_id": "api-check", "obligation_id": obligation["id"], "control_id": control["id"], "check_type": "transaction", "subject_reference": "payment", "result": "compliant"})
	agent = api.register_compliance_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Compliance Agent", "runtime": "claude_code", "role": "obligation_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.compliance_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "obligation_reviewer"
	assert dashboard["summary"]["obligation_count"] == 1
	assert console["checks"][0]["id"] == "api-check"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_compliance"]["screens"]["agents"]["route"] == "/fintech-compliance/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_compliance", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_compliance"]["streaming"]["processor"] == "bytewax"
