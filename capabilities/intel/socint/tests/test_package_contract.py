"""Executable Social Media Intelligence capability package tests."""

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
	module = _load_module("contract_intel_socint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_socint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "socint_agent_workflow" in contract["provides"]
	assert "/intel-socint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_socint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "socint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "socint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "socint_agent_action", "platform_abuse_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "socint_agent_action", "doxxing_scope": True})["decision"] == "deny"


def test_service_executes_socint_lifecycle():
	service_module = _load_module("service_intel_socint", PACKAGE_DIR / "service.py")
	service = service_module.SocialMediaIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "legal_mandate", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	topic = service.record_topic("topic-1", "tenant-test", "disinformation", "Election rumor watch", "high", authority["id"], "topic-evidence")
	source = service.register_source("source-1", "tenant-test", "hashtag", "microblog", "#public-topic", "owner-1", authority["id"], "terms-review", "source-evidence")
	post = service.record_post("post-1", "tenant-test", topic["id"], source["id"], "post", "post-ref", "sha256:abc", "2026-06-01T00:00:00Z", 0.82, "post-evidence")
	signal = service.record_signal("signal-1", "tenant-test", post["id"], "misinformation", "high", 0.86, "analyst-1", "signal-evidence")
	influence = service.record_influence("influence-1", "tenant-test", signal["id"], "amplification", 0.84, "analyst-1", "influence-evidence")
	network = service.record_network("network-1", "tenant-test", signal["id"], "hashtag_graph", "medium", 0.79, "analyst-1", "network-evidence")
	referral = service.record_referral("referral-1", "tenant-test", influence["id"], "policy_review", "policy-team", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", network["id"], "public-safety-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_socint_agent("agent-1", "tenant-test", "SOCINT Agent", "codex", "signal_analyst", "signal analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "legal_mandate"
	assert topic["topic_type"] == "disinformation"
	assert source["platform_type"] == "microblog"
	assert post["post_type"] == "post"
	assert signal["signal_type"] == "misinformation"
	assert influence["influence_type"] == "amplification"
	assert network["network_type"] == "hashtag_graph"
	assert referral["referral_type"] == "policy_review"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_socint", PACKAGE_DIR / "service.py")
	service = service_module.SocialMediaIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "legal_mandate", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_topic("shared-topic", "tenant-a", "brand", "Brand A", "medium", tenant_a["id"], "evidence-a")
	service.record_topic("shared-topic", "tenant-b", "policy", "Policy B", "low", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["topic_count"] == 1
	assert dashboard_b["topic_count"] == 1
	assert service._tenant_topic_or_none("shared-topic", "tenant-a").name == "Brand A"
	assert service._tenant_topic_or_none("shared-topic", "tenant-b").name == "Policy B"


def test_service_guardrails_reject_invalid_socint_actions():
	service_module = _load_module("guardrail_service_intel_socint", PACKAGE_DIR / "service.py")
	service = service_module.SocialMediaIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "legal_mandate", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "legal_mandate", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_topic("topic", "tenant-test", "brand", "brand", "medium", "missing-auth", "evidence")
	topic = service.record_topic("topic-ok", "tenant-test", "brand", "brand", "medium", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_terms_review_required"):
		service.register_source("source", "tenant-test", "account", "social_network", "@public", "owner", authority["id"], "", "evidence")
	source = service.register_source("source-ok", "tenant-test", "account", "social_network", "@public", "owner", authority["id"], "terms", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_topic = service.record_topic("topic-other", "tenant-test", "policy", "policy", "low", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_post("post", "tenant-test", other_topic["id"], source["id"], "post", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_post("post", "tenant-test", topic["id"], source["id"], "post", "ref", "hash", "2026-06-01", 1.8, "evidence")
	post = service.record_post("post-ok", "tenant-test", topic["id"], source["id"], "post", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="signal_type_not_supported"):
		service.record_signal("signal", "tenant-test", post["id"], "unknown", "high", 0.8, "analyst", "evidence")
	signal = service.record_signal("signal-ok", "tenant-test", post["id"], "trend", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="influence_type_not_supported"):
		service.record_influence("influence", "tenant-test", signal["id"], "unknown", 0.8, "analyst", "evidence")
	influence = service.record_influence("influence-ok", "tenant-test", signal["id"], "reach", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="network_type_not_supported"):
		service.record_network("network", "tenant-test", signal["id"], "unknown", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", influence["id"], "policy_review", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", influence["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", influence["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="socint_agent_runtime_not_supported"):
		service.register_socint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "signal_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="platform_abuse_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, platform_abuse_scope=True)
	with pytest.raises(PermissionError, match="doxxing_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, doxxing_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_socint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_socint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_socint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	topic = api.record_topic({"tenant_id": "tenant-api", "topic_id": "api-topic", "topic_type": "brand", "name": "Brand", "priority": "medium", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "page", "platform_type": "social_network", "source_reference": "page-ref", "owner_id": "owner", "authority_id": authority["id"], "terms_review_reference": "terms", "evidence_reference": "evidence"})
	api.record_post({"tenant_id": "tenant-api", "post_id": "api-post", "topic_id": topic["id"], "source_id": source["id"], "post_type": "post", "post_reference": "post-ref", "content_fingerprint": "hash", "observed_at": "2026-06-01", "confidence_score": 0.8, "evidence_reference": "evidence"})
	agent = api.register_socint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "SOCINT Agent", "runtime": "claude_code", "role": "signal_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.socint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "signal_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_socint"]["screens"]["agents"]["route"] == "/intel-socint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_socint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_socint"]["streaming"]["processor"] == "bytewax"
