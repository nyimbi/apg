"""Regression coverage for the NLPC executable capability contract."""

import pytest

from capabilities.common.nlpc import register_capability
from capabilities.common.nlpc.capability_contract import (
	SUPPORTED_LANGUAGES,
	evaluate_capability_rules,
	get_capability_contract
)
from capabilities.common.nlpc.nlpc_runtime import NlpcService
from capabilities.common.nlpc.view_models import (
	annotation_workbench_model,
	audit_timeline_model,
	batch_queue_model,
	dashboard_model,
	document_workbench_model,
	governance_model,
	language_coverage_model,
	lexicon_manager_model,
	lifecycle_batch_model,
	model_registry_model,
	nlp_agent_roster_model,
	pipeline_designer_model,
	processing_console_model,
	review_console_model,
	semantic_search_model,
)


AFRICAN_LANGUAGE_CODES = {
	"af", "aa", "ak", "am", "bm", "ee", "ff", "ha", "ig", "kr",
	"ki", "rw", "rn", "kg", "ln", "lg", "mg", "ny", "om", "sg",
	"sn", "so", "st", "sw", "ss", "ti", "ts", "tn", "tw", "ve",
	"wo", "xh", "yo", "zu", "kab", "kam", "luo", "mas", "mer",
	"mos", "nus", "suk", "tzm", "tig", "umb"
}


def test_contract_exposes_configuration_rules_ui_theme_adapters_and_languages():
	contract = get_capability_contract("tenant-text", {"processing": {"max_document_chars": 5000}})

	assert contract["capability"] == "nlpc"
	assert contract["configuration"]["tenant_id"] == "tenant-text"
	assert contract["configuration"]["processing"]["max_document_chars"] == 5000
	assert AFRICAN_LANGUAGE_CODES <= set(SUPPORTED_LANGUAGES)
	assert AFRICAN_LANGUAGE_CODES <= set(contract["configuration"]["processing"]["supported_languages"])
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"processing",
		"languages",
		"tasks",
		"pipelines",
		"annotation",
		"model_registry",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"process",
		"documents",
		"pipelines",
		"batches",
		"annotations",
		"review",
		"models",
		"languages",
		"lexicons",
		"search",
		"agents",
		"lifecycle",
		"governance",
		"audit",
		"settings"
	}
	assert contract["provides"] == ["text_intelligence", "multilingual_processing", "nlp_agent_composition"]
	assert contract["requires"] == ["aicr", "mlcm", "conf"]
	assert contract["agents"]["first_class"] is True
	assert {"codex", "claude_code", "opencode", "pi"} <= set(contract["agents"]["supported_runtimes"])
	assert "generation_safety_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "nlp_agent_batch" in contract["streaming"]["required_operations"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "nlpc_runtime.NlpcService"
	assert contract["ui"]["api_prefix"] == "/nlpc/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "pipeline_designer" in contract["theme"]["components"]
	assert "nlp_agent_roster" in contract["theme"]["components"]


def test_rule_engine_enforces_nlp_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_document",
		"language_known": False,
		"language_supported": True,
		"task": "text_generation",
		"task_enabled": True,
		"safety_policy_attached": False,
		"model_policy_attached": False,
		"confidence_score": 0.25,
		"human_review_recorded": False,
		"document_count": 100,
		"async_queue_enabled": False
	})
	pii_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"task": "pii_detection",
		"redaction_policy_attached": False
	})
	stream_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_batch_events",
		"event_stream": "rabbitmq",
	})
	bytewax_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "configure_batch_events",
		"event_stream": "bytewax",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"language_required_or_detected",
		"generation_requires_safety_policy",
		"generation_requires_model_policy",
		"low_confidence_requires_review",
		"large_batch_requires_async_queue"
	}
	assert pii_result["decision"] == "deny"
	assert pii_result["matched_rules"] == ["pii_requires_redaction_policy"]
	assert stream_result["matched_rules"] == ["batch_requires_bytewax_stream"]
	assert bytewax_result["decision"] == "allow"
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_nlp_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
	})
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"nlp_agent_runtime_supported",
		"nlp_agent_role_supported",
		"nlp_agent_requires_scope",
		"nlp_agent_requires_owner",
		"nlp_agent_requires_purpose",
		"nlp_agent_requires_contribution_disclosure",
	}
	review_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_nlp_agent",
		"agent_runtime_supported": True,
		"agent_role_supported": True,
		"scope_present": True,
		"owner_present": True,
		"purpose_present": True,
		"contribution_disclosed": True,
		"privileged_role": True,
		"human_approval_required": False,
	})
	assert review_result["decision"] == "require_review"
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_nlpc_lifecycle_batch",
		"event_stream": "redis",
	})
	assert lifecycle_result["matched_rules"] == ["bytewax_nlpc_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "nlpc"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "nlpc_text_intelligence"
	assert registration["ui_components"]["languages"] == "/nlpc/languages"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "nlp_agent_composition" in registration["capabilities"]
	assert "aicr" in registration["dependencies"]
	assert "moni" in registration["dependencies"]
	assert "nlpc:process" in registration["permissions"]
	assert "nlpc:search" in registration["permissions"]


def test_nlpc_lifecycle_is_executable():
	service = NlpcService()

	document = service.ingest_document(
		"doc-001",
		"tenant-text",
		"Habari Nairobi. This excellent report was prepared by Amina for ops@example.com.",
		"auto",
		"case://001",
	)
	model = service.register_model(
		"model-001",
		"tenant-text",
		"Entity and Sentiment Model",
		"mlcm://nlpc/entity-sentiment",
		"language-team",
		"policy://nlpc/safe",
	)
	pipeline = service.register_pipeline(
		"pipe-001",
		"tenant-text",
		"Customer Text Pipeline",
		"language-team",
		model["id"],
		"1.0.0",
		["sentiment_analysis", "entity_recognition", "pii_detection", "semantic_search"],
	)
	run = service.process_document(
		"run-001",
		"tenant-text",
		document["id"],
		pipeline["tasks"],
		redaction_policy_attached=True,
		search_index_attached=True,
	)
	project = service.create_annotation_project(
		"ann-project-001",
		"tenant-text",
		"Entity Review",
		"Label people and organizations only.",
		"entity_recognition",
	)
	annotation = service.submit_annotation(
		"ann-001",
		"tenant-text",
		project["id"],
		document["id"],
		"reviewer-1",
		["PERSON", "LOCATION"],
		0.95,
	)
	release = service.release_model(model["id"], "tenant-text", True, True)
	lexicon = service.register_lexicon(
		"lex-001",
		"tenant-text",
		"Service Terms",
		"sw",
		["habari", "asante"],
		"language-team",
	)

	assert document["language"] == "sw"
	assert run["results"]["sentiment_analysis"]["label"] == "positive"
	assert run["results"]["pii_detection"]["pii_detected"] is True
	assert annotation["status"] == "accepted"
	assert release["status"] == "released"
	assert lexicon["term_count"] == 2
	assert service.dashboard_summary("tenant-text") == {
		"tenant_id": "tenant-text",
		"document_count": 1,
		"processing_run_count": 1,
		"pipeline_count": 1,
		"model_count": 1,
		"released_model_count": 1,
		"annotation_project_count": 1,
		"annotation_count": 1,
		"lexicon_count": 1,
		"nlp_agent_count": 0,
		"pending_agent_review_count": 0,
		"lifecycle_batch_count": 0,
		"denied_lifecycle_batch_count": 0,
		"audit_event_count": 8,
		"supported_language_count": len(SUPPORTED_LANGUAGES),
		"african_language_count": len(AFRICAN_LANGUAGE_CODES),
	}
	agent = service.register_nlp_agent(
		"nlp-reviewer",
		"tenant-text",
		"NLPC Safety Reviewer",
		"codex",
		"generation_safety_reviewer",
		"pipe-001 generation outputs",
		"language-team",
		"Review generated summaries and safety policy drift",
		human_approval_required=True,
	)
	batch = service.validate_nlpc_lifecycle_batch(
		"tenant-text",
		"bytewax",
		4,
		"nlp_agent_batch",
		"nlpc-batch-001",
	)
	assert agent["runtime"] == "codex"
	assert agent["role"] == "generation_safety_reviewer"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert dashboard_model(service, "tenant-text")["summary"]["processing_run_count"] == 1
	assert dashboard_model(service, "tenant-text")["summary"]["nlp_agent_count"] == 1
	assert processing_console_model(service, "tenant-text")["enabled_tasks"]
	assert document_workbench_model(service, "tenant-text")["documents"][0]["id"] == "doc-001"
	assert pipeline_designer_model(service, "tenant-text")["pipelines"][0]["id"] == "pipe-001"
	assert batch_queue_model(service, "tenant-text")["engine"] == "bytewax"
	assert annotation_workbench_model(service, "tenant-text")["annotations"][0]["id"] == "ann-001"
	assert review_console_model(service, "tenant-text")["low_confidence_runs"] == []
	assert model_registry_model(service, "tenant-text")["models"][0]["status"] == "released"
	assert language_coverage_model("tenant-text")["african_language_count"] >= 40
	assert lexicon_manager_model(service, "tenant-text")["lexicons"][0]["id"] == "lex-001"
	assert semantic_search_model(service, "tenant-text")["search_runs"][0]["id"] == "run-001"
	assert nlp_agent_roster_model(service, "tenant-text")["agents"][0]["id"] == "nlp-reviewer"
	assert lifecycle_batch_model(service, "tenant-text")["batches"][0]["id"] == "nlpc-batch-001"
	assert governance_model(service, "tenant-text")["rules"]
	assert audit_timeline_model(service, "tenant-text")["audit_events"]


def test_nlpc_service_enforces_policy_guardrails():
	service = NlpcService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.ingest_document("doc", "", "No tenant")
	with pytest.raises(PermissionError, match="document_content_required"):
		service.ingest_document("doc", "tenant-text", "")

	service.ingest_document("doc", "tenant-text", "Hello from Nairobi", "en")
	task_calls: list[str] = []

	def forbidden_task_execution(content: str, task: str, language: str) -> tuple[float, dict[str, object]]:
		task_calls.append(task)
		return 0.99, {}

	service._run_task = forbidden_task_execution  # type: ignore[method-assign]
	with pytest.raises(PermissionError, match="unsupported_language"):
		service.ingest_document("doc-bad-lang", "tenant-text", "Hello", "zz")
		service.process_document("run-bad-lang", "tenant-text", "doc-bad-lang", "sentiment_analysis")
	assert task_calls == []
	with pytest.raises(PermissionError, match="pii_redaction_policy_required"):
		service.process_document("run-pii", "tenant-text", "doc", "pii_detection")
	assert task_calls == []
	with pytest.raises(PermissionError, match="generation_safety_policy_required"):
		service.process_document("run-gen", "tenant-text", "doc", "text_generation", model_policy_attached=True)
	assert task_calls == []
	with pytest.raises(PermissionError, match="model_policy_required"):
		service.process_document("run-gen-policy", "tenant-text", "doc", "text_generation", safety_policy_attached=True)
	assert task_calls == []
	with pytest.raises(PermissionError, match="search_index_required"):
		service.process_document("run-search", "tenant-text", "doc", "semantic_search")
	assert task_calls == []

	with pytest.raises(PermissionError, match="pipeline_owner_required"):
		service.register_pipeline("pipe", "tenant-text", "Pipeline", "", "model", "1.0.0", ["sentiment_analysis"])
	with pytest.raises(PermissionError, match="registered_model_required"):
		service.register_pipeline("pipe", "tenant-text", "Pipeline", "owner", "", "1.0.0", ["sentiment_analysis"])
	with pytest.raises(PermissionError, match="pipeline_version_required"):
		service.register_pipeline("pipe", "tenant-text", "Pipeline", "owner", "model", "", ["sentiment_analysis"])

	with pytest.raises(PermissionError, match="mlcm_model_ref_required"):
		service.register_model("model", "tenant-text", "Model", "", "owner")
	service.register_model("model", "tenant-text", "Model", "mlcm://model", "owner")
	with pytest.raises(PermissionError, match="model_evaluation_required"):
		service.release_model("model", "tenant-text", False, True)
	with pytest.raises(PermissionError, match="model_release_approval_required"):
		service.release_model("model", "tenant-text", True, False)

	with pytest.raises(PermissionError, match="annotation_guidelines_required"):
		service.create_annotation_project("project", "tenant-text", "Project", "", "entity_recognition")
	service.create_annotation_project("project", "tenant-text", "Project", "Guidelines", "entity_recognition")
	with pytest.raises(PermissionError, match="annotation_adjudication_required"):
		service.submit_annotation("ann", "tenant-text", "project", "doc", "reviewer", ["ORG"], 0.20, False)
	with pytest.raises(PermissionError, match="lexicon_language_required"):
		service.register_lexicon("lex", "tenant-text", "Lexicon", "", ["term"])

	with pytest.raises(PermissionError, match="unsupported_nlp_agent_runtime"):
		service.register_nlp_agent("agent-bad-runtime", "tenant-text", "Bad Runtime", "unknown", "language_steward", "doc", "owner", "purpose")
	with pytest.raises(PermissionError, match="nlp_agent_scope_required"):
		service.register_nlp_agent("agent-no-scope", "tenant-text", "No Scope", "codex", "language_steward", "", "owner", "purpose")
	agent = service.register_nlp_agent("agent-review", "tenant-text", "Review Agent", "claude-code", "pii reviewer", "doc", "owner", "purpose")
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "pii_reviewer"
	assert agent["status"] == "pending_review"
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_nlpc_lifecycle_batch("tenant-text", "legacy_queue", 1)
	batch = service.validate_nlpc_lifecycle_batch("tenant-text", "bytewax", 1, "language_registry_batch")
	assert batch["accepted"] is True
