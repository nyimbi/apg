"""Regression coverage for the RAGN executable capability contract."""

import pytest

from capabilities.common.ragn import register_capability
from capabilities.common.ragn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.ragn.rag_runtime import RagnService
from capabilities.common.ragn.views import (
	audit_timeline_model,
	citation_model,
	conversation_model,
	curation_model,
	dashboard_model,
	document_model,
	generation_model,
	governance_model,
	knowledge_base_model,
	retrieval_model,
	settings_model,
	studio_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-rag", {"retrieval": {"minimum_context_confidence": 0.8}})

	assert contract["capability"] == "ragn"
	assert contract["configuration"]["tenant_id"] == "tenant-rag"
	assert contract["configuration"]["retrieval"]["minimum_context_confidence"] == 0.8
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "knowledge_bases", "documents", "retrieval", "generation", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "studio", "knowledge_bases", "documents", "retrieval", "generation", "conversations", "citations", "curation", "governance", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/ragn/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "rag_runtime.RagnService"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "answer_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_rag_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "generate_answer",
		"query_present": False,
		"context_present": False,
		"answer_text_present": False,
		"citations_attached": False,
		"context_confidence": 0.2,
		"review_recorded": False,
		"model_location": "external",
		"model_policy_attached": False,
		"prompt_injection_detected": True,
		"unsafe_answer_detected": True,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "batch_rag_mutation",
		"event_stream": "kafka",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"generation_requires_query",
		"generation_requires_context",
		"generation_requires_answer_text",
		"generation_requires_citations",
		"low_context_confidence_requires_review",
		"external_model_requires_policy",
		"prompt_injection_requires_block",
		"unsafe_generation_requires_block",
	}
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_rag_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ragn"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ragn_answer_studio"
	assert registration["ui_components"]["studio"] == "/ragn/studio"
	assert "srch" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["capabilities"]["grounded_generation"]
	assert registration["endpoints"]["audit"] == "/ragn/api/v1/audit"
	assert "ragn:query" in registration["permissions"]
	assert "ragn:audit" in registration["permissions"]


def test_ragn_lifecycle_is_executable():
	service = RagnService()

	kb = service.create_knowledge_base(
		knowledge_base_id="kb-policy",
		tenant_id="tenant-rag",
		name="Policy knowledge base",
		owner="knowledge-steward",
		source_attribution="policy-library",
	)
	document = service.ingest_document(
		document_id="doc-travel",
		tenant_id="tenant-rag",
		knowledge_base_id=kb["id"],
		title="Travel policy",
		source_uri="meta://policies/travel",
		content_hash="sha256:travel",
		classification="internal",
	)
	retrieval = service.retrieve_context(
		retrieval_id="ret-travel",
		tenant_id="tenant-rag",
		knowledge_base_id=kb["id"],
		query="What approval is required for international travel?",
		document_ids=[document["id"]],
		context_confidence=0.92,
	)
	answer = service.generate_answer(
		answer_id="ans-travel",
		tenant_id="tenant-rag",
		retrieval_id=retrieval["id"],
		query="What approval is required for international travel?",
		answer_text="International travel requires manager and finance approval.",
		citations=[{"source_id": "policy-library", "document_id": document["id"], "chunk_id": "chunk-1"}],
	)
	turn = service.record_turn(
		turn_id="turn-travel",
		tenant_id="tenant-rag",
		conversation_id="conv-travel",
		user_id="user-1",
		query="What approval is required for international travel?",
		answer_id=answer["id"],
		turn_count=1,
	)
	curation = service.curate_answer(
		curation_id="curate-travel",
		tenant_id="tenant-rag",
		answer_id=answer["id"],
		curator="knowledge-steward",
		decision="approved",
		evidence="review:travel-answer",
	)

	assert kb["metadata"]["owner"] == "knowledge-steward"
	assert document["status"] == "indexed"
	assert retrieval["metadata"]["context_confidence"] == 0.92
	assert answer["metadata"]["citation_count"] == 1
	assert turn["status"] == "recorded"
	assert curation["status"] == "approved"
	assert service.dashboard_summary("tenant-rag")["citation_count"] == 1
	assert service.rag_package("tenant-rag")["summary"]["answer_count"] == 1
	assert dashboard_model(service, "tenant-rag")["summary"]["document_count"] == 1
	assert studio_model(service, "tenant-rag")["answers"][0]["id"] == "ans-travel"
	assert knowledge_base_model(service, "tenant-rag")["knowledge_bases"][0]["id"] == "kb-policy"
	assert document_model(service, "tenant-rag")["documents"][0]["id"] == "doc-travel"
	assert retrieval_model(service, "tenant-rag")["retrievals"][0]["id"] == "ret-travel"
	assert generation_model(service, "tenant-rag")["answers"][0]["id"] == "ans-travel"
	assert conversation_model(service, "tenant-rag")["conversation_turns"][0]["id"] == "turn-travel"
	assert citation_model(service, "tenant-rag")["citation_count"] == 1
	assert curation_model(service, "tenant-rag")["curations"][0]["id"] == "curate-travel"
	assert governance_model(service, "tenant-rag")["rules"]
	assert audit_timeline_model(service, "tenant-rag")["audit_events"]
	assert settings_model(service, "tenant-rag")["adapters"]["event_stream"] == "bytewax"


def test_ragn_service_enforces_policy_guardrails():
	service = RagnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_knowledge_base(
			knowledge_base_id="kb-missing-tenant",
			tenant_id="",
			name="Missing tenant",
			owner="owner",
			source_attribution="manual",
		)

	with pytest.raises(PermissionError, match="knowledge_base_owner_required"):
		service.create_knowledge_base(
			knowledge_base_id="kb-no-owner",
			tenant_id="tenant-rag",
			name="No owner",
			owner="",
			source_attribution="manual",
		)

	kb = service.create_knowledge_base(
		knowledge_base_id="kb-policy",
		tenant_id="tenant-rag",
		name="Policy knowledge base",
		owner="steward",
		source_attribution="manual",
	)

	with pytest.raises(PermissionError, match="content_hash_required"):
		service.ingest_document(
			document_id="doc-no-hash",
			tenant_id="tenant-rag",
			knowledge_base_id=kb["id"],
			title="No hash",
			source_uri="manual://no-hash",
			content_hash="",
		)

	document = service.ingest_document(
		document_id="doc-restricted",
		tenant_id="tenant-rag",
		knowledge_base_id=kb["id"],
		title="Restricted policy",
		source_uri="manual://restricted",
		content_hash="sha256:restricted",
		classification="restricted",
	)

	with pytest.raises(PermissionError, match="access_filter_required"):
		service.retrieve_context(
			retrieval_id="ret-no-filter",
			tenant_id="tenant-rag",
			knowledge_base_id=kb["id"],
			query="restricted?",
			document_ids=[document["id"]],
			context_confidence=0.9,
			source_classification="restricted",
			access_filter_applied=False,
		)

	with pytest.raises(PermissionError, match="low_context_confidence_review_required"):
		service.retrieve_context(
			retrieval_id="ret-low",
			tenant_id="tenant-rag",
			knowledge_base_id=kb["id"],
			query="low?",
			document_ids=[document["id"]],
			context_confidence=0.4,
			review_recorded=False,
		)

	retrieval = service.retrieve_context(
		retrieval_id="ret-reviewed",
		tenant_id="tenant-rag",
		knowledge_base_id=kb["id"],
		query="low?",
		document_ids=[document["id"]],
		context_confidence=0.4,
		review_recorded=True,
	)

	with pytest.raises(PermissionError, match="citations_required"):
		service.generate_answer(
			answer_id="ans-no-citations",
			tenant_id="tenant-rag",
			retrieval_id=retrieval["id"],
			query="low?",
			answer_text="No citations",
			citations=[],
		)

	with pytest.raises(PermissionError, match="model_policy_required"):
		service.generate_answer(
			answer_id="ans-external",
			tenant_id="tenant-rag",
			retrieval_id=retrieval["id"],
			query="low?",
			answer_text="External answer",
			citations=[{"source_id": "manual", "document_id": document["id"], "chunk_id": "chunk-1"}],
			model_location="external",
			model_policy_attached=False,
		)
