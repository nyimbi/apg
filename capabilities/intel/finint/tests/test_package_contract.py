"""Executable Financial Intelligence capability package tests."""

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
	module = _load_module("contract_intel_finint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_finint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "finint_agent_workflow" in contract["provides"]
	assert "/intel-finint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_finint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "finint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "finint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "finint_agent_action", "funds_movement_scope": True})["decision"] == "deny"


def test_service_executes_finint_lifecycle():
	service_module = _load_module("service_intel_finint", PACKAGE_DIR / "service.py")
	service = service_module.FinancialIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "regulatory_authority", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	source = service.register_source("source-1", "tenant-test", "bank_feed", "KE", "owner-1", authority["id"], "source-evidence")
	subject = service.record_subject("subject-1", "tenant-test", "account", "acct-ref", "high", authority["id"], "subject-evidence")
	transaction = service.record_transaction("txn-1", "tenant-test", source["id"], subject["id"], "txn-ref", 2500.0, "kes", "transfer", "2026-06-01T00:00:00Z", "txn-evidence")
	pattern = service.record_pattern("pattern-1", "tenant-test", transaction["id"], "structuring", 0.86, "analyst-1", "pattern-evidence")
	risk = service.record_risk("risk-1", "tenant-test", pattern["id"], "aml", "high", 0.88, "analyst-1", "risk-evidence")
	referral = service.record_referral("referral-1", "tenant-test", risk["id"], "sar", "fiu", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", risk["id"], "compliance-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_finint_agent("agent-1", "tenant-test", "FININT Agent", "codex", "transaction_analyst", "transaction analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "regulatory_authority"
	assert source["source_type"] == "bank_feed"
	assert subject["risk_tier"] == "high"
	assert transaction["currency"] == "KES"
	assert pattern["pattern_type"] == "structuring"
	assert risk["risk_type"] == "aml"
	assert referral["referral_type"] == "sar"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 10


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_finint", PACKAGE_DIR / "service.py")
	service = service_module.FinancialIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "regulatory_authority", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.register_source("shared-source", "tenant-a", "bank_feed", "KE", "owner-a", tenant_a["id"], "evidence-a")
	service.register_source("shared-source", "tenant-b", "public_filing", "US", "owner-b", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["source_count"] == 1
	assert dashboard_b["source_count"] == 1
	assert service._tenant_source_or_none("shared-source", "tenant-a").jurisdiction == "KE"
	assert service._tenant_source_or_none("shared-source", "tenant-b").jurisdiction == "US"


def test_service_guardrails_reject_invalid_finint_actions():
	service_module = _load_module("guardrail_service_intel_finint", PACKAGE_DIR / "service.py")
	service = service_module.FinancialIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "regulatory_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "regulatory_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.register_source("source", "tenant-test", "bank_feed", "KE", "owner", "missing-auth", "evidence")
	with pytest.raises(PermissionError, match="source_type_not_supported"):
		service.register_source("source", "tenant-test", "unknown", "KE", "owner", authority["id"], "evidence")
	source = service.register_source("source-ok", "tenant-test", "bank_feed", "KE", "owner", authority["id"], "evidence")
	subject = service.record_subject("subject-ok", "tenant-test", "account", "acct", "high", authority["id"], "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_subject = service.record_subject("subject-other", "tenant-test", "account", "acct2", "medium", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_transaction("txn", "tenant-test", source["id"], other_subject["id"], "txn", 10, "KES", "transfer", "2026-06-01", "evidence")
	with pytest.raises(PermissionError, match="amount_invalid"):
		service.record_transaction("txn", "tenant-test", source["id"], subject["id"], "txn", 0, "KES", "transfer", "2026-06-01", "evidence")
	transaction = service.record_transaction("txn-ok", "tenant-test", source["id"], subject["id"], "txn", 10, "KES", "transfer", "2026-06-01", "evidence")
	with pytest.raises(PermissionError, match="pattern_type_not_supported"):
		service.record_pattern("pattern", "tenant-test", transaction["id"], "unknown", 0.8, "analyst", "evidence")
	pattern = service.record_pattern("pattern-ok", "tenant-test", transaction["id"], "structuring", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="risk_type_not_supported"):
		service.record_risk("risk", "tenant-test", pattern["id"], "unknown", "high", 0.8, "analyst", "evidence")
	risk = service.record_risk("risk-ok", "tenant-test", pattern["id"], "aml", "high", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", risk["id"], "sar", "fiu", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", risk["id"], "audience", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", risk["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="finint_agent_runtime_not_supported"):
		service.register_finint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "transaction_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="funds_movement_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, funds_movement_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_finint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_finint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_finint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "public_filing", "jurisdiction": "US", "owner_id": "owner", "authority_id": authority["id"], "evidence_reference": "evidence"})
	api.record_subject({"tenant_id": "tenant-api", "subject_id": "api-subject", "subject_type": "organization", "subject_reference": "org-ref", "risk_tier": "medium", "authority_id": authority["id"], "evidence_reference": "evidence"})
	agent = api.register_finint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "FININT Agent", "runtime": "claude_code", "role": "transaction_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.finint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "transaction_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_finint"]["screens"]["agents"]["route"] == "/intel-finint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_finint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_finint"]["streaming"]["processor"] == "bytewax"
