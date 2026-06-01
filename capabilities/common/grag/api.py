"""API helpers for the Graph-based RAG capability."""

from __future__ import annotations

from typing import Any

from .grag_runtime import GragService


SERVICE = GragService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**summary,
	}


def register_graph_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_graph_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		graph_id=str(payload["graph_id"]),
		provenance_refs=tuple(payload.get("provenance_refs") or ()),
		classification=str(payload.get("classification") or "internal"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def retire_graph_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_graph_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def register_vector_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_vector_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		index_id=str(payload["index_id"]),
		embedding_model=str(payload["embedding_model"]),
		document_refs=tuple(payload.get("document_refs") or ()),
		owner=str(payload["owner"]),
	)


def run_hybrid_query(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_hybrid_query(
		query_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query=str(payload["query"]),
		graph_source_id=str(payload["graph_source_id"]),
		vector_source_id=str(payload["vector_source_id"]),
		vector_index_ready=bool(payload.get("vector_index_ready", True)),
		graph_index_ready=bool(payload.get("graph_index_ready", True)),
		result_window=int(payload.get("result_window", 10)),
		source_classification=str(payload.get("source_classification") or "internal"),
		access_filter_applied=bool(payload.get("access_filter_applied", True)),
		retrieval_confidence=float(payload.get("retrieval_confidence", 1.0)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def build_reasoning_path(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.build_reasoning_path(
		path_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query_id=str(payload["query_id"]),
		start_node_id=str(payload["start_node_id"]),
		evidence_path=tuple(payload.get("evidence_path") or ()),
		hop_count=int(payload.get("hop_count", 1)),
		explanation=str(payload.get("explanation") or ""),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def generate_answer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.generate_answer(
		answer_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query_id=str(payload["query_id"]),
		path_id=str(payload["path_id"]),
		query=str(payload["query"]),
		answer_text=str(payload.get("answer_text") or ""),
		provenance_refs=tuple(payload.get("provenance_refs") or ()),
		citations=tuple(payload.get("citations") or ()),
		model_location=str(payload.get("model_location") or "local"),
		model_policy_attached=bool(payload.get("model_policy_attached", True)),
		unsafe_answer_detected=bool(payload.get("unsafe_answer_detected", False)),
		confidence_score=float(payload.get("confidence_score", 1.0)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def curate_answer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.curate_answer(
		curation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		answer_id=str(payload["answer_id"]),
		curator=str(payload["curator"]),
		decision=str(payload["decision"]),
		evidence=str(payload["evidence"]),
	)


def publish_answer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_answer(
		publication_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		answer_id=str(payload["answer_id"]),
		curation_id=str(payload["curation_id"]),
		publisher=str(payload["publisher"]),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def register_grag_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_grag_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_grag_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_grag_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count", 1)),
		operation=str(payload.get("operation") or "graphrag_agent_batch"),
		batch_id=payload.get("id") or payload.get("batch_id"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_graphrag_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_graphrag_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def list_pending_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_pending_reviews(tenant_id)


def grag_package(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.grag_package(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
