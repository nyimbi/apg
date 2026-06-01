"""Executable Know Your Customer capability package tests."""

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
	module = _load_module("contract_fintech_kyc", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "fintech_kyc"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "kyc_agent_workflow" in contract["provides"]
	assert "/fintech-kyc/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_hits_without_review():
	module = _load_module("rules_fintech_kyc", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "kyc_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "record_screening", "screening_hit": True, "review_recorded": False})["decision"] == "require_review"


def test_service_executes_kyc_lifecycle():
	service_module = _load_module("service_fintech_kyc", PACKAGE_DIR / "service.py")
	service = service_module.KnowYourCustomerService()

	profile = service.open_profile("profile-a", "tenant-test", "customer-a", "Amina Njeri", "individual", "KE", "consent-1")
	identity = service.register_document("doc-id", "tenant-test", profile["id"], "national_id", "vault://id", "Amina Njeri", 0.94)
	address = service.register_document("doc-address", "tenant-test", profile["id"], "utility_bill", "vault://address", "Amina Njeri", 0.9)
	screening = service.record_screening("screen-1", "tenant-test", profile["id"])
	risk = service.score_risk("risk-1", "tenant-test", profile["id"], 22)
	decision = service.record_decision("decision-1", "tenant-test", profile["id"], "approve", 22)
	agent = service.register_kyc_agent("agent-1", "tenant-test", "KYC Agent", "codex", "document_reviewer", "review documents")
	batch = service.validate_batch("tenant-test", 7)
	summary = service.dashboard_summary("tenant-test")

	assert identity["status"] == "verified"
	assert address["document_type"] == "utility_bill"
	assert screening["status"] == "clear"
	assert risk["decision"] == "low"
	assert decision["status"] == "verified"
	assert agent["metadata"]["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["profile_count"] == 1
	assert summary["verified_count"] == 1
	assert summary["audit_event_count"] == 7


def test_service_guardrails_reject_invalid_kyc_actions():
	service_module = _load_module("guardrail_service_fintech_kyc", PACKAGE_DIR / "service.py")
	service = service_module.KnowYourCustomerService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.open_profile("profile", "", "subject", "Name", "individual", "KE", "consent")
	with pytest.raises(PermissionError, match="customer_type_not_supported"):
		service.open_profile("profile", "tenant-test", "subject", "Name", "alien", "KE", "consent")
	with pytest.raises(PermissionError, match="customer_consent_required"):
		service.open_profile("profile", "tenant-test", "subject", "Name", "individual", "KE", "")
	profile = service.open_profile("profile", "tenant-test", "subject", "Name", "individual", "KE", "consent")
	with pytest.raises(PermissionError, match="document_confidence_below_minimum"):
		service.register_document("doc-low", "tenant-test", profile["id"], "passport", "vault://doc", "Name", 0.5)
	with pytest.raises(PermissionError, match="screening_review_required"):
		service.record_screening("screen-hit", "tenant-test", profile["id"], sanctions_hit=True)
	with pytest.raises(PermissionError, match="enhanced_due_diligence_required"):
		service.score_risk("risk-high", "tenant-test", profile["id"], 91)
	service.record_screening("screen-clear", "tenant-test", profile["id"])
	service.score_risk("risk-low", "tenant-test", profile["id"], 20)
	with pytest.raises(PermissionError, match="identity_document_required"):
		service.record_decision("decision", "tenant-test", profile["id"], "approve", 20)


def test_api_views_and_app_are_executable():
	api = _load_module("api_fintech_kyc", PACKAGE_DIR / "api.py")
	views = _load_module("views_fintech_kyc", PACKAGE_DIR / "views.py")
	app = _load_module("app_fintech_kyc", PACKAGE_DIR / "app.py")

	profile = api.open_profile({"tenant_id": "tenant-api", "profile_id": "api-profile", "subject_reference": "customer-api", "legal_name": "API Customer", "customer_type": "individual", "country_code": "KE", "consent_reference": "consent-api"})
	api.register_document({"tenant_id": "tenant-api", "document_id": "api-id", "profile_id": profile["id"], "document_type": "national_id", "token_reference": "vault://api-id", "extracted_subject": "API Customer", "confidence": 0.91})
	api.register_document({"tenant_id": "tenant-api", "document_id": "api-address", "profile_id": profile["id"], "document_type": "utility_bill", "token_reference": "vault://api-address", "extracted_subject": "API Customer", "confidence": 0.9})
	api.record_screening({"tenant_id": "tenant-api", "screening_id": "api-screen", "profile_id": profile["id"]})
	api.score_risk({"tenant_id": "tenant-api", "decision_id": "api-risk", "profile_id": profile["id"], "risk_score": 21})
	agent = api.register_kyc_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "KYC Agent", "runtime": "claude_code", "role": "kyc_ops_reviewer"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.profile_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["metadata"]["role"] == "kyc_ops_reviewer"
	assert dashboard["summary"]["profile_count"] == 1
	assert console["profiles"][0]["id"] == "api-profile"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["fintech_kyc"]["screens"]["agents"]["route"] == "/fintech-kyc/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_fintech_kyc", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["fintech_kyc"]["streaming"]["processor"] == "bytewax"
