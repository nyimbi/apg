"""Regression coverage for the GRAG executable capability contract."""

import pytest

from capabilities.common.grag import register_capability
from capabilities.common.grag.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.grag.grag_runtime import GragService
from capabilities.common.grag.views import (
	audit_timeline_model,
	curation_model,
	dashboard_model,
	generation_model,
	governance_model,
	graph_source_model,
	hybrid_retrieval_model,
	provenance_model,
	query_model,
	reasoning_model,
	settings_model,
	vector_source_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-grag", {"reasoning": {"max_hops": 2}})

	assert contract["capability"] == "grag"
	assert contract["configuration"]["tenant_id"] == "tenant-grag"
	assert contract["configuration"]["reasoning"]["max_hops"] == 2
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "graph_sources", "vector_sources", "hybrid_retrieval", "reasoning", "generation", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "query", "graph_sources", "vector_sources", "hybrid_retrieval", "reasoning", "provenance", "generation", "curation", "governance", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/grag/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "grag_runtime.GragService"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "reasoning_path" in contract["theme"]["components"]


def test_rule_engine_enforces_graphrag_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "generate_answer",
		"query_present": False,
		"retrieval_context_present": False,
		"reasoning_path_present": False,
		"answer_text_present": False,
		"provenance_attached": False,
		"citations_attached": False,
		"model_location": "external",
		"model_policy_attached": False,
		"unsafe_answer_detected": True,
		"answer_confidence": 0.4,
		"review_recorded": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "batch_grag_mutation",
		"event_stream": "kafka",
	})
	reasoning_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "build_reasoning_path",
		"query_present": True,
		"start_node_present": False,
		"evidence_path_present": False,
		"hop_count": 0,
		"explanation_present": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"generation_requires_query",
		"generation_requires_retrieval_context",
		"generation_requires_reasoning_path",
		"generation_requires_answer_text",
		"answer_requires_provenance",
		"generation_requires_citations",
		"external_model_requires_policy",
		"unsafe_generation_requires_block",
		"low_answer_confidence_requires_review",
	}
	assert batch_result["decision"] == "deny"
	assert batch_result["matched_rules"] == ["batch_grag_mutation_requires_bytewax"]
	assert set(reasoning_result["matched_rules"]) >= {"reasoning_requires_start_node", "reasoning_requires_evidence_path", "reasoning_requires_positive_hops", "reasoning_requires_explanation"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "grag"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "grag_reasoning_console"
	assert registration["ui_components"]["reasoning"] == "/grag/reasoning"
	assert "ragn" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["capabilities"]["graph_grounded_generation"]
	assert registration["endpoints"]["audit"] == "/grag/api/v1/audit"
	assert "grag:reason" in registration["permissions"]
	assert "grag:audit" in registration["permissions"]


def test_grag_lifecycle_is_executable():
	service = GragService()

	graph = service.register_graph_source(
		source_id="graph-policy",
		tenant_id="tenant-grag",
		name="Policy graph",
		owner="knowledge-steward",
		graph_id="grph-policy",
		provenance_refs=["source:policy-library"],
	)
	vector = service.register_vector_source(
		source_id="vector-policy",
		tenant_id="tenant-grag",
		index_id="idx-policy",
		embedding_model="text-embedding-3-large",
		document_refs=["doc-travel"],
		owner="knowledge-steward",
	)
	retrieval = service.run_hybrid_query(
		query_id="query-travel",
		tenant_id="tenant-grag",
		query="What approval is required for international travel?",
		graph_source_id=graph["id"],
		vector_source_id=vector["id"],
		retrieval_confidence=0.92,
	)
	path = service.build_reasoning_path(
		path_id="path-travel",
		tenant_id="tenant-grag",
		query_id=retrieval["id"],
		start_node_id="policy:travel",
		evidence_path=["policy:travel", "approval:manager", "approval:finance"],
		hop_count=2,
		explanation="The travel policy points to manager and finance approval.",
	)
	answer = service.generate_answer(
		answer_id="answer-travel",
		tenant_id="tenant-grag",
		query_id=retrieval["id"],
		path_id=path["id"],
		query="What approval is required for international travel?",
		answer_text="International travel requires manager and finance approval.",
		provenance_refs=["source:policy-library", "path:path-travel"],
		citations=[{"source_id": "policy-library", "document_id": "doc-travel", "chunk_id": "chunk-1"}],
		confidence_score=0.93,
	)
	curation = service.curate_answer(
		curation_id="curation-travel",
		tenant_id="tenant-grag",
		answer_id=answer["id"],
		curator="knowledge-steward",
		decision="approved",
		evidence="review:travel-answer",
	)
	publication = service.publish_answer(
		publication_id="publication-travel",
		tenant_id="tenant-grag",
		answer_id=answer["id"],
		curation_id=curation["id"],
		publisher="knowledge-steward",
	)

	assert graph["metadata"]["owner"] == "knowledge-steward"
	assert vector["status"] == "indexed"
	assert retrieval["metadata"]["retrieval_confidence"] == 0.92
	assert path["metadata"]["hop_count"] == 2
	assert answer["metadata"]["citation_count"] == 1
	assert curation["status"] == "approved"
	assert publication["status"] == "published"
	assert service.dashboard_summary("tenant-grag")["publication_count"] == 1
	assert service.grag_package("tenant-grag")["summary"]["answer_count"] == 1
	assert dashboard_model(service, "tenant-grag")["summary"]["graph_source_count"] == 1
	assert query_model(service, "tenant-grag")["answers"][0]["id"] == "answer-travel"
	assert graph_source_model(service, "tenant-grag")["graph_sources"][0]["id"] == "graph-policy"
	assert vector_source_model(service, "tenant-grag")["vector_sources"][0]["id"] == "vector-policy"
	assert hybrid_retrieval_model(service, "tenant-grag")["hybrid_queries"][0]["id"] == "query-travel"
	assert reasoning_model(service, "tenant-grag")["reasoning_paths"][0]["id"] == "path-travel"
	assert provenance_model(service, "tenant-grag")["citation_count"] == 1
	assert generation_model(service, "tenant-grag")["answers"][0]["id"] == "answer-travel"
	assert curation_model(service, "tenant-grag")["curations"][0]["id"] == "curation-travel"
	assert governance_model(service, "tenant-grag")["rules"]
	assert audit_timeline_model(service, "tenant-grag")["audit_events"]
	assert settings_model(service, "tenant-grag")["adapters"]["event_stream"] == "bytewax"


def test_grag_service_enforces_policy_guardrails():
	service = GragService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_graph_source(
			source_id="graph-missing-tenant",
			tenant_id="",
			name="Missing tenant",
			owner="owner",
			graph_id="graph",
			provenance_refs=["manual"],
		)

	with pytest.raises(PermissionError, match="graph_source_owner_required"):
		service.register_graph_source(
			source_id="graph-no-owner",
			tenant_id="tenant-grag",
			name="No owner",
			owner="",
			graph_id="graph",
			provenance_refs=["manual"],
		)

	graph = service.register_graph_source(
		source_id="graph-policy",
		tenant_id="tenant-grag",
		name="Policy graph",
		owner="steward",
		graph_id="grph-policy",
		provenance_refs=["manual"],
		classification="restricted",
	)

	with pytest.raises(PermissionError, match="source_documents_required"):
		service.register_vector_source(
			source_id="vector-no-docs",
			tenant_id="tenant-grag",
			index_id="idx",
			embedding_model="model",
			document_refs=[],
			owner="steward",
		)

	vector = service.register_vector_source(
		source_id="vector-policy",
		tenant_id="tenant-grag",
		index_id="idx-policy",
		embedding_model="text-embedding-3-large",
		document_refs=["doc-restricted"],
		owner="steward",
	)

	with pytest.raises(PermissionError, match="access_filter_required"):
		service.run_hybrid_query(
			query_id="query-no-filter",
			tenant_id="tenant-grag",
			query="restricted?",
			graph_source_id=graph["id"],
			vector_source_id=vector["id"],
			source_classification="restricted",
			access_filter_applied=False,
		)

	with pytest.raises(PermissionError, match="low_retrieval_confidence_review_required"):
		service.run_hybrid_query(
			query_id="query-low",
			tenant_id="tenant-grag",
			query="low?",
			graph_source_id=graph["id"],
			vector_source_id=vector["id"],
			retrieval_confidence=0.4,
			review_recorded=False,
		)

	retrieval = service.run_hybrid_query(
		query_id="query-reviewed",
		tenant_id="tenant-grag",
		query="low?",
		graph_source_id=graph["id"],
		vector_source_id=vector["id"],
		retrieval_confidence=0.4,
		review_recorded=True,
	)

	with pytest.raises(PermissionError, match="evidence_path_required"):
		service.build_reasoning_path(
			path_id="path-no-evidence",
			tenant_id="tenant-grag",
			query_id=retrieval["id"],
			start_node_id="policy:travel",
			evidence_path=[],
			hop_count=1,
			explanation="No evidence.",
		)

	path = service.build_reasoning_path(
		path_id="path-reviewed",
		tenant_id="tenant-grag",
		query_id=retrieval["id"],
		start_node_id="policy:travel",
		evidence_path=["policy:travel", "approval:manager"],
		hop_count=5,
		explanation="Reviewed deep path.",
		review_recorded=True,
	)

	with pytest.raises(PermissionError, match="citations_required"):
		service.generate_answer(
			answer_id="answer-no-citations",
			tenant_id="tenant-grag",
			query_id=retrieval["id"],
			path_id=path["id"],
			query="low?",
			answer_text="No citations",
			provenance_refs=["manual"],
			citations=[],
		)

	with pytest.raises(PermissionError, match="model_policy_required"):
		service.generate_answer(
			answer_id="answer-external-no-policy",
			tenant_id="tenant-grag",
			query_id=retrieval["id"],
			path_id=path["id"],
			query="low?",
			answer_text="External model answer",
			provenance_refs=["manual"],
			citations=[{"source_id": "manual", "document_id": "doc", "chunk_id": "chunk"}],
			model_location="external",
			model_policy_attached=False,
		)

	answer = service.generate_answer(
		answer_id="answer-reviewed",
		tenant_id="tenant-grag",
		query_id=retrieval["id"],
		path_id=path["id"],
		query="low?",
		answer_text="Reviewed answer.",
		provenance_refs=["manual"],
		citations=[{"source_id": "manual", "document_id": "doc", "chunk_id": "chunk"}],
		confidence_score=0.4,
		review_recorded=True,
	)
	curation = service.curate_answer(
		curation_id="curation-reject",
		tenant_id="tenant-grag",
		answer_id=answer["id"],
		curator="steward",
		decision="rejected",
		evidence="not supported",
	)

	with pytest.raises(PermissionError, match="curated_answer_required"):
		service.publish_answer(
			publication_id="publication-rejected",
			tenant_id="tenant-grag",
			answer_id=answer["id"],
			curation_id=curation["id"],
			publisher="steward",
		)
