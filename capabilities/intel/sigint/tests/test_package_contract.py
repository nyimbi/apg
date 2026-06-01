"""Executable Signals Intelligence capability package tests."""

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
	module = _load_module("contract_intel_sigint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_sigint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "sigint_agent_workflow" in contract["provides"]
	assert "/intel-sigint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_intel_sigint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "sigint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "sigint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_sigint_lifecycle():
	service_module = _load_module("service_intel_sigint", PACKAGE_DIR / "service.py")
	service = service_module.SignalsIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "secret", "approver-1", "2026-12-31", "authority-evidence")
	source = service.register_source("source-1", "tenant-test", "radio", "vhf", "sensor-ref", "owner-1", authority["id"], "source-evidence")
	task = service.record_collection_task("task-1", "tenant-test", authority["id"], source["id"], "metadata_only", 30, "min-ref", "approval-ref", "task-evidence")
	observation = service.record_observation("obs-1", "tenant-test", task["id"], "observation-ref", "sha256:abc", 0.91, "observation-evidence")
	batch = service.record_processing_batch("batch-1", "tenant-test", observation["id"], "traffic_analysis", 0.87, "analyst-1", "batch-evidence")
	pattern = service.record_pattern("pattern-1", "tenant-test", batch["id"], "beacon", 0.82, "analyst-1", "pattern-evidence")
	assessment = service.record_assessment("assessment-1", "tenant-test", pattern["id"], "threat", "secret", "analyst-1", "assessment-evidence")
	review = service.record_review("review-1", "tenant-test", assessment["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_sigint_agent("agent-1", "tenant-test", "SIGINT Agent", "codex", "authority_reviewer", "review authorities")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert source["band"] == "vhf"
	assert task["retention_days"] == 30
	assert observation["confidence_score"] == 0.91
	assert batch["processing_type"] == "traffic_analysis"
	assert pattern["pattern_type"] == "beacon"
	assert assessment["classification"] == "secret"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 9


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_sigint", PACKAGE_DIR / "service.py")
	service = service_module.SignalsIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "secret", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "confidential", "approver-b", "2026-12-31", "evidence-b")
	service.register_source("shared-source", "tenant-a", "radio", "vhf", "sensor-a", "owner-a", tenant_a["id"], "evidence-a")
	service.register_source("shared-source", "tenant-b", "partner_feed", "metadata", "feed-b", "owner-b", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["source_count"] == 1
	assert dashboard_b["source_count"] == 1
	assert service._tenant_authority_or_none("shared-auth", "tenant-a").authority_type == "mission_order"
	assert service._tenant_authority_or_none("shared-auth", "tenant-b").authority_type == "consent"


def test_service_guardrails_reject_invalid_sigint_actions():
	service_module = _load_module("guardrail_service_intel_sigint", PACKAGE_DIR / "service.py")
	service = service_module.SignalsIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "secret", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="signal_band_not_supported"):
		service.register_source("source", "tenant-test", "radio", "bad_band", "ref", "owner", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.register_source("source-missing-auth", "tenant-test", "radio", "vhf", "ref", "owner", "missing-auth", "evidence")
	source = service.register_source("source-ok", "tenant-test", "radio", "vhf", "ref", "owner", authority["id"], "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="source_authority_mismatch"):
		service.record_collection_task("task", "tenant-test", other_authority["id"], source["id"], "metadata_only", 30, "min", "approval", "evidence")
	with pytest.raises(PermissionError, match="retention_days_invalid"):
		service.record_collection_task("task", "tenant-test", authority["id"], source["id"], "metadata_only", 0, "min", "approval", "evidence")
	with pytest.raises(PermissionError, match="minimization_reference_required"):
		service.record_collection_task("task", "tenant-test", authority["id"], source["id"], "metadata_only", 30, "", "approval", "evidence")
	task = service.record_collection_task("task-ok", "tenant-test", authority["id"], source["id"], "metadata_only", 30, "min", "approval", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_observation("obs", "tenant-test", task["id"], "obs-ref", "fingerprint", 1.2, "evidence")
	observation = service.record_observation("obs-ok", "tenant-test", task["id"], "obs-ref", "fingerprint", 0.9, "evidence")
	with pytest.raises(PermissionError, match="processing_type_not_supported"):
		service.record_processing_batch("batch", "tenant-test", observation["id"], "unknown", 0.9, "analyst", "evidence")
	batch = service.record_processing_batch("batch-ok", "tenant-test", observation["id"], "normalization", 0.9, "analyst", "evidence")
	with pytest.raises(PermissionError, match="pattern_type_not_supported"):
		service.record_pattern("pattern", "tenant-test", batch["id"], "unknown", 0.9, "analyst", "evidence")
	pattern = service.record_pattern("pattern-ok", "tenant-test", batch["id"], "beacon", 0.9, "analyst", "evidence")
	with pytest.raises(PermissionError, match="classification_not_supported"):
		service.record_assessment("assessment", "tenant-test", pattern["id"], "threat", "unknown", "analyst", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", pattern["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="sigint_agent_runtime_not_supported"):
		service.register_sigint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "authority_reviewer", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_sigint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_sigint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_sigint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "confidential", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "radio", "band": "vhf", "source_reference": "sensor", "owner_id": "owner", "authority_id": authority["id"], "evidence_reference": "evidence"})
	api.record_collection_task({"tenant_id": "tenant-api", "task_id": "api-task", "authority_id": authority["id"], "source_id": source["id"], "collection_mode": "metadata_only", "retention_days": 7, "minimization_reference": "min", "approval_reference": "approval", "evidence_reference": "evidence"})
	agent = api.register_sigint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "SIGINT Agent", "runtime": "claude_code", "role": "collection_planner"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.sigint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "collection_planner"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["collection_tasks"][0]["id"] == "api-task"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_sigint"]["screens"]["agents"]["route"] == "/intel-sigint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_sigint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_sigint"]["streaming"]["processor"] == "bytewax"
