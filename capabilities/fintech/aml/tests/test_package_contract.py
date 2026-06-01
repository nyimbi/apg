"""Executable Anti Money Laundering capability package tests."""

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
	module = _load_module("contract_fintech_aml", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_aml"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "aml_agent_workflow" in contract["provides"]
	assert "/fintech-aml/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_sar_without_approval():
	module = _load_module("rules_fintech_aml", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "aml_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "monitor_transaction", "large_transaction": True, "review_recorded": False})["decision"] == "require_review"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "draft_sar", "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_aml_lifecycle():
	service_module = _load_module("service_fintech_aml", PACKAGE_DIR / "service.py")
	service = service_module.AntiMoneyLaunderingService()

	transaction = service.monitor_transaction("txn-1", "tenant-test", "customer-a", "kyc-a", 120, "KES", "fintech_payments", "pay-1", 22)
	alert = service.create_alert_from_transaction("alert-1", "tenant-test", transaction["id"])
	triaged = service.triage_alert(alert["id"], "tenant-test", "escalate", reviewer_id="analyst-1")
	case = service.open_case("case-1", "tenant-test", triaged["id"], "transaction_monitoring", "investigator-1")
	sar = service.draft_sar("sar-1", "tenant-test", case["id"], "customer-a", "KE", "Suspicious activity narrative with linked evidence.", ["txn-1", "alert-1", "case-1"], "compliance-manager")
	agent = service.register_aml_agent("agent-1", "tenant-test", "AML Agent", "codex", "case_investigator", "investigate cases")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert transaction["status"] == "monitored"
	assert alert["alert_type"] == "agent_review"
	assert triaged["status"] == "escalated"
	assert service.cases[case["id"]].status == "confirmed_suspicious"
	assert sar["status"] == "approved_for_filing"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["transaction_count"] == 1
	assert summary["alert_count"] == 1
	assert summary["case_count"] == 1
	assert summary["sar_count"] == 1
	assert summary["audit_event_count"] == 6


def test_service_guardrails_reject_invalid_aml_actions():
	service_module = _load_module("guardrail_service_fintech_aml", PACKAGE_DIR / "service.py")
	service = service_module.AntiMoneyLaunderingService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.monitor_transaction("txn", "", "subject", "kyc", 10, "KES", "fintech_payments", "pay")
	with pytest.raises(PermissionError, match="positive_amount_required"):
		service.monitor_transaction("txn", "tenant-test", "subject", "kyc", -1, "KES", "fintech_payments", "pay")
	with pytest.raises(PermissionError, match="kyc_link_required"):
		service.monitor_transaction("txn", "tenant-test", "subject", "", 10, "KES", "fintech_payments", "pay")
	with pytest.raises(PermissionError, match="large_transaction_review_required"):
		service.monitor_transaction("txn", "tenant-test", "subject", "kyc", 10000, "KES", "fintech_payments", "pay")
	transaction = service.monitor_transaction("txn-ok", "tenant-test", "subject", "kyc", 10000, "KES", "fintech_payments", "pay", review_id="review-1")
	with pytest.raises(PermissionError, match="aml_alert_type_not_supported"):
		service.create_alert("alert-invalid", "tenant-test", "unknown", "medium", "subject", [transaction["id"]])
	alert = service.create_alert("alert", "tenant-test", "large_transaction", "medium", "subject", [transaction["id"]])
	with pytest.raises(PermissionError, match="alert_disposition_required"):
		service.triage_alert(alert["id"], "tenant-test", "close")
	with pytest.raises(PermissionError, match="case_investigator_required"):
		service.open_case("case", "tenant-test", alert["id"], "transaction_monitoring", "")
	case = service.open_case("case", "tenant-test", alert["id"], "transaction_monitoring", "investigator")
	with pytest.raises(PermissionError, match="sar_human_approval_required"):
		service.draft_sar("sar", "tenant-test", case["id"], "subject", "KE", "Narrative", [transaction["id"]], "")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="aml_agent_runtime_not_supported"):
		service.register_aml_agent("agent", "tenant-test", "Bad Agent", "unsupported", "case_investigator", "scope")


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_aml", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_aml", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_aml", PACKAGE_DIR / "app.py")

	transaction = api.monitor_transaction({"tenant_id": "tenant-api", "transaction_id": "api-txn", "subject_reference": "customer-api", "kyc_profile_id": "kyc-api", "amount": 40, "currency": "KES", "source_reference": "payment-api", "risk_score": 12})
	alert = api.create_alert_from_transaction({"tenant_id": "tenant-api", "alert_id": "api-alert", "transaction_id": transaction["id"]})
	api.triage_alert({"tenant_id": "tenant-api", "alert_id": alert["id"], "action": "escalate", "reviewer_id": "analyst-api"})
	api.open_case({"tenant_id": "tenant-api", "case_id": "api-case", "alert_id": alert["id"], "investigator_id": "investigator-api"})
	agent = api.register_aml_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "AML Agent", "runtime": "claude_code", "role": "aml_ops_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.alert_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "aml_ops_reviewer"
	assert dashboard["summary"]["alert_count"] == 1
	assert console["alerts"][0]["id"] == "api-alert"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_aml"]["screens"]["agents"]["route"] == "/fintech-aml/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_aml", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_aml"]["streaming"]["processor"] == "bytewax"
