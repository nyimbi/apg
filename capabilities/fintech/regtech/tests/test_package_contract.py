"""Executable Regulatory Technology capability package tests."""

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
	module = _load_module("contract_fintech_regtech", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_regtech"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "regulatory_obligation_mapping_workflow" in contract["provides"]
	assert "regulatory_agent_workflow" in contract["provides"]
	assert "/fintech-regtech/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_fintech_regtech", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "regtech_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "regtech_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_regtech_lifecycle():
	service_module = _load_module("service_fintech_regtech", PACKAGE_DIR / "service.py")
	service = service_module.RegTechService()

	source = service.register_source("source-1", "tenant-test", "central_bank", "KE", "gazette-1", "owner-1", "source-evidence")
	change = service.record_change("change-1", "tenant-test", source["id"], "psd2", "new_rule", "Digital credit rules", "2026-06-01", "high", "evidence-1")
	obligation = service.map_obligation("mapping-1", "tenant-test", change["id"], "obligation-1", "policy-1", "owner-1", "2026-07-01")
	impact = service.assess_impact("impact-1", "tenant-test", change["id"], "fintech_lending", "high", "impact-evidence", "reviewer-1")
	filing = service.prepare_filing("filing-1", "tenant-test", "psd2", "regulatory_return", "2026-Q2", "filing-evidence", "owner-1")
	submission = service.record_submission("submission-1", "tenant-test", filing["id"], "portal", "submitter-1", "2026-06-02T10:00:00Z", "ack-1")
	inquiry = service.open_inquiry("inquiry-1", "tenant-test", "central_bank", "inquiry-ref", "medium", "2026-07-01", "inquiry-evidence")
	response = service.record_response("response-1", "tenant-test", inquiry["id"], "responder-1", "response-doc", "approval-1")
	review = service.record_review("review-1", "tenant-test", submission["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_regtech_agent("agent-1", "tenant-test", "RegTech Agent", "codex", "filing_preparer", "prepare filings")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert source["regulator"] == "central_bank"
	assert change["framework"] == "psd2"
	assert obligation["policy_reference"] == "policy-1"
	assert impact["risk_rating"] == "high"
	assert submission["channel"] == "portal"
	assert response["approval_reference"] == "approval-1"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["obligation_count"] == 1
	assert summary["response_count"] == 1
	assert summary["audit_event_count"] == 10


def test_service_guardrails_reject_invalid_regtech_actions():
	service_module = _load_module("guardrail_service_fintech_regtech", PACKAGE_DIR / "service.py")
	service = service_module.RegTechService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_source("source", "", "central_bank", "KE", "source", "owner", "evidence")
	with pytest.raises(PermissionError, match="regulator_not_supported"):
		service.register_source("source", "tenant-test", "unsupported", "KE", "source", "owner", "evidence")
	with pytest.raises(PermissionError, match="jurisdiction_not_supported"):
		service.register_source("source", "tenant-test", "central_bank", "XX", "source", "owner", "evidence")
	source = service.register_source("source-ok", "tenant-test", "central_bank", "KE", "source", "owner", "evidence")
	with pytest.raises(PermissionError, match="framework_not_supported"):
		service.record_change("change", "tenant-test", source["id"], "unknown", "new_rule", "Title", "2026-06-01", "medium", "evidence")
	with pytest.raises(PermissionError, match="severity_not_supported"):
		service.record_change("change", "tenant-test", source["id"], "psd2", "new_rule", "Title", "2026-06-01", "severe", "evidence")
	change = service.record_change("change-ok", "tenant-test", source["id"], "psd2", "new_rule", "Title", "2026-06-01", "medium", "evidence")
	with pytest.raises(PermissionError, match="policy_reference_required"):
		service.map_obligation("mapping", "tenant-test", change["id"], "obligation", "", "owner", "2026-07-01")
	with pytest.raises(PermissionError, match="risk_rating_not_supported"):
		service.assess_impact("impact", "tenant-test", change["id"], "capability", "severe", "evidence", "reviewer")
	with pytest.raises(PermissionError, match="filing_owner_required"):
		service.prepare_filing("filing", "tenant-test", "psd2", "regulatory_return", "2026-Q2", "evidence", "")
	filing = service.prepare_filing("filing-ok", "tenant-test", "psd2", "regulatory_return", "2026-Q2", "evidence", "owner")
	with pytest.raises(PermissionError, match="submission_channel_not_supported"):
		service.record_submission("submission", "tenant-test", filing["id"], "fax", "submitter", "2026-06-02T10:00:00Z", "ack")
	with pytest.raises(PermissionError, match="inquiry_due_date_required"):
		service.open_inquiry("inquiry", "tenant-test", "central_bank", "ref", "medium", "", "evidence")
	inquiry = service.open_inquiry("inquiry-ok", "tenant-test", "central_bank", "ref", "medium", "2026-07-01", "evidence")
	with pytest.raises(PermissionError, match="response_approval_required"):
		service.record_response("response", "tenant-test", inquiry["id"], "responder", "response", "")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", filing["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="regtech_agent_runtime_not_supported"):
		service.register_regtech_agent("agent", "tenant-test", "Bad Agent", "unsupported", "filing_preparer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_regtech", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_regtech", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_regtech", PACKAGE_DIR / "app.py")

	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "regulator": "central_bank", "jurisdiction": "KE", "source_reference": "gazette", "owner_id": "owner", "evidence_reference": "evidence"})
	change = api.record_change({"tenant_id": "tenant-api", "change_id": "api-change", "source_id": source["id"], "framework": "psd2", "change_type": "new_rule", "title": "Rule", "effective_date": "2026-06-01", "severity": "medium", "evidence_reference": "evidence"})
	obligation = api.map_obligation({"tenant_id": "tenant-api", "mapping_id": "api-mapping", "change_id": change["id"], "obligation_reference": "obligation", "policy_reference": "policy", "owner_id": "owner", "due_date": "2026-07-01"})
	api.assess_impact({"tenant_id": "tenant-api", "assessment_id": "api-impact", "change_id": change["id"], "impacted_capability": "fintech_lending", "risk_rating": "medium", "evidence_reference": "evidence", "reviewer_id": "reviewer"})
	agent = api.register_regtech_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "RegTech Agent", "runtime": "claude_code", "role": "regulatory_change_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.regtech_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "regulatory_change_reviewer"
	assert dashboard["summary"]["source_count"] == 1
	assert console["obligations"][0]["id"] == obligation["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_regtech"]["screens"]["agents"]["route"] == "/fintech-regtech/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_regtech", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_regtech"]["streaming"]["processor"] == "bytewax"
