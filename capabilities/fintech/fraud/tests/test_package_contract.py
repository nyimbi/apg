"""Executable Fraud Detection capability package tests."""

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
	module = _load_module("contract_fintech_fraud", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_fraud"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "fraud_agent_workflow" in contract["provides"]
	assert "/fintech-fraud/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_high_impact_decisions():
	module = _load_module("rules_fintech_fraud", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fraud_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "score_signal", "high_risk_score": True, "review_recorded": False})["decision"] == "require_review"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_decision", "hold_or_block": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_fraud_lifecycle():
	service_module = _load_module("service_fintech_fraud", PACKAGE_DIR / "service.py")
	service = service_module.FraudDetectionService()

	signal = service.score_signal("sig-1", "tenant-test", "customer-a", "kyc-a", "payment", "api", "payment-1", 120, "KES", 52, review_id="review-1")
	decision = service.record_decision("decision-1", "tenant-test", signal["id"], "step_up", reviewer_id="analyst-1", challenge_reference="challenge-1")
	case = service.open_case("case-1", "tenant-test", signal["id"], "transaction_fraud", "investigator-1", ["sig-1", "decision-1"])
	resolved = service.resolve_case(case["id"], "tenant-test", "customer_verified", "fraud-manager")
	agent = service.register_fraud_agent("agent-1", "tenant-test", "Fraud Agent", "codex", "case_investigator", "investigate cases")
	batch = service.validate_batch("tenant-test", 4)
	summary = service.dashboard_summary("tenant-test")

	assert signal["risk_band"] == "medium"
	assert signal["recommended_decision"] == "review"
	assert decision["decision"] == "step_up"
	assert resolved["status"] == "resolved"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["signal_count"] == 1
	assert summary["decision_count"] == 1
	assert summary["case_count"] == 1
	assert summary["open_case_count"] == 0
	assert summary["audit_event_count"] == 5


def test_service_guardrails_reject_invalid_fraud_actions():
	service_module = _load_module("guardrail_service_fintech_fraud", PACKAGE_DIR / "service.py")
	service = service_module.FraudDetectionService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.score_signal("sig", "", "subject", "kyc", "payment", "api", "pay", 10, "KES", 20)
	with pytest.raises(PermissionError, match="kyc_link_required"):
		service.score_signal("sig", "tenant-test", "subject", "", "payment", "api", "pay", 10, "KES", 20)
	with pytest.raises(PermissionError, match="positive_amount_required"):
		service.score_signal("sig", "tenant-test", "subject", "kyc", "payment", "api", "pay", -1, "KES", 20)
	with pytest.raises(PermissionError, match="high_fraud_risk_review_required"):
		service.score_signal("sig", "tenant-test", "subject", "kyc", "payment", "api", "pay", 10, "KES", 91)
	with pytest.raises(PermissionError, match="account_takeover_review_required"):
		service.score_signal("sig", "tenant-test", "subject", "kyc", "account_login", "web", "login", risk_score=30, account_takeover_indicator=True)
	signal = service.score_signal("sig-ok", "tenant-test", "subject", "kyc", "payment", "api", "pay", 10, "KES", 91, review_id="review-1")
	with pytest.raises(PermissionError, match="challenge_reference_required"):
		service.record_decision("decision-step", "tenant-test", signal["id"], "step_up")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.record_decision("decision-block", "tenant-test", signal["id"], "block", reason="confirmed fraud")
	with pytest.raises(PermissionError, match="case_investigator_required"):
		service.open_case("case", "tenant-test", signal["id"], "transaction_fraud", "", [signal["id"]])
	case = service.open_case("case", "tenant-test", signal["id"], "transaction_fraud", "investigator", [signal["id"]])
	with pytest.raises(PermissionError, match="case_disposition_required"):
		service.resolve_case(case["id"], "tenant-test", "", "reviewer")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="fraud_agent_runtime_not_supported"):
		service.register_fraud_agent("agent", "tenant-test", "Bad Agent", "unsupported", "case_investigator", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_fraud", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_fraud", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_fraud", PACKAGE_DIR / "app.py")

	signal = api.score_signal({"tenant_id": "tenant-api", "signal_id": "api-sig", "subject_reference": "customer-api", "kyc_profile_id": "kyc-api", "signal_type": "payment", "channel": "mobile", "source_reference": "payment-api", "amount": 44, "currency": "KES", "risk_score": 52, "review_id": "review-api"})
	api.record_decision({"tenant_id": "tenant-api", "decision_id": "api-decision", "signal_id": signal["id"], "decision": "review"})
	api.open_case({"tenant_id": "tenant-api", "case_id": "api-case", "signal_id": signal["id"], "case_type": "transaction_fraud", "investigator_id": "investigator-api", "evidence_references": [signal["id"]]})
	agent = api.register_fraud_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Fraud Agent", "runtime": "claude_code", "role": "fraud_ops_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.signal_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "fraud_ops_reviewer"
	assert dashboard["summary"]["signal_count"] == 1
	assert console["signals"][0]["id"] == "api-sig"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_fraud"]["screens"]["agents"]["route"] == "/fintech-fraud/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_fraud", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_fraud"]["streaming"]["processor"] == "bytewax"
