"""Executable RCM capability package tests."""

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
	module = _load_module("contract_grc_rcm", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "grc_rcm"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "rcm_agents" in contract["provides"]
	assert "/grc-rcm/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"


def test_rule_engine_blocks_missing_context_non_bytewax_and_evidence_gap():
	module = _load_module("rules_grc_rcm", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "rcm_batch",
		"event_stream": "queue",
	})["decision"] == "deny"
	assert module.evaluate_capability_rules({
		"tenant_id": "tenant-test",
		"tenant_context_present": True,
		"operation": "assess_control",
		"failed_assessment": True,
		"evidence_present": False,
	})["matched_rules"] == ["failed_assessment_requires_evidence"]


def test_service_executes_rcm_lifecycle():
	service_module = _load_module("service_grc_rcm", PACKAGE_DIR / "service.py")
	service = service_module.GrcRcmService()

	risk = service.register_risk("risk-1", "tenant-test", "Payment outage", "technology", "owner-1", 0.8, 0.7, "reviewer-1")
	control = service.register_control("control-1", "tenant-test", "Provider failover test", "owner-2", "detective", [risk["id"]])
	obligation = service.register_obligation("obligation-1", "tenant-test", "PCI DSS", "Maintain payment controls", "owner-3", "global", "2026-12-31", [control["id"]])
	evidence = service.collect_evidence("evidence-1", "tenant-test", "control-test-log", "control", control["id"])
	assessment = service.assess_control("assessment-1", "tenant-test", control["id"], "assessor-1", "partially_effective", [evidence["id"]], ["failover delay"])
	issue = service.open_issue("issue-1", "tenant-test", "Reduce failover delay", "high", "owner-4", "Tune routing", assessment["id"], "reviewer-2")
	remediation_evidence = service.collect_evidence("evidence-2", "tenant-test", "remediation-log", "issue", issue["id"])
	remediated = service.remediate_issue(issue["id"], "tenant-test", remediation_evidence["id"])
	decision = service.record_governance_decision("decision-1", "tenant-test", "Accept residual delay", "approver-1", "Residual exposure reviewed", [risk["id"]], "reviewer-3")
	exception = service.register_exception("exception-1", "tenant-test", "risk_acceptance", risk["id"], "2026-09-30", "approver-2")
	agent = service.register_rcm_agent("tenant-test", "Control Review Agent", "codex", "control_reviewer", "review controls")

	summary = service.dashboard_summary("tenant-test")
	assert risk["risk_level"] == "high"
	assert obligation["framework"] == "PCI DSS"
	assert remediated["status"] == "remediated"
	assert decision["status"] == "approved"
	assert exception["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert summary["audit_event_count"] == 11
	assert summary["streaming"]["processor"] == "bytewax"


def test_service_guardrails_reject_invalid_actions():
	service_module = _load_module("guardrail_service_grc_rcm", PACKAGE_DIR / "service.py")
	service = service_module.GrcRcmService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_risk("risk", "", "Risk", "technology", "owner", 0.2, 0.2)
	with pytest.raises(PermissionError, match="risk_category_not_supported"):
		service.register_risk("risk", "tenant-test", "Risk", "unsupported", "owner", 0.2, 0.2)
	with pytest.raises(PermissionError, match="high_risk_review_required"):
		service.register_risk("risk", "tenant-test", "Risk", "technology", "owner", 0.9, 0.9)

	risk = service.register_risk("risk", "tenant-test", "Risk", "technology", "owner", 0.2, 0.2)
	with pytest.raises(PermissionError, match="mapped_risk_required"):
		service.register_control("control", "tenant-test", "Control", "owner", "preventive", [])
	control = service.register_control("control", "tenant-test", "Control", "owner", "preventive", [risk["id"]])
	with pytest.raises(PermissionError, match="failed_assessment_evidence_required"):
		service.assess_control("assessment", "tenant-test", control["id"], "assessor", "ineffective")
	with pytest.raises(PermissionError, match="obligation_jurisdiction_required"):
		service.register_obligation("obligation", "tenant-test", "ISO27001", "Access control", "owner", "", "2026-12-31", [control["id"]])
	with pytest.raises(PermissionError, match="evidence_encryption_required"):
		service.collect_evidence("evidence", "tenant-test", "source", "control", control["id"], encrypted=False)
	with pytest.raises(PermissionError, match="issue_review_required"):
		service.open_issue("issue", "tenant-test", "Issue", "critical", "owner", "plan")
	with pytest.raises(PermissionError, match="governance_title_required"):
		service.record_governance_decision("decision", "tenant-test", "", "approver", "rationale", [risk["id"]])


def test_agents_batch_api_views_and_app_are_executable():
	api = _load_module("api_grc_rcm", PACKAGE_DIR / "api.py")
	views = _load_module("views_grc_rcm", PACKAGE_DIR / "views.py")
	app = _load_module("app_grc_rcm", PACKAGE_DIR / "app.py")

	risk = api.create_record({"tenant_id": "tenant-api", "id": "api-risk"})
	agent = api.register_rcm_agent({
		"tenant_id": "tenant-api",
		"name": "Risk Review Agent",
		"runtime": "claude_code",
		"role": "risk_reviewer",
	})
	batch = api.service().validate_batch("tenant-api", 2)
	model = views.risk_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert risk["id"] == "api-risk"
	assert agent["role"] == "risk_reviewer"
	assert batch["processor"] == "bytewax"
	assert model["records"][0]["title"] == "API Risk"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["grc_rcm"]["screens"]["agents"]["route"] == "/grc-rcm/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_grc_rcm", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["grc_rcm"]["streaming"]["processor"] == "bytewax"
