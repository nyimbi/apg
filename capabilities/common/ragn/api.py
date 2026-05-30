"""API helpers for the Retrieval-Augmented Generation capability."""

from __future__ import annotations

from typing import Any

from .rag_runtime import RagnService


SERVICE = RagnService()


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


def create_knowledge_base(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_knowledge_base(
		knowledge_base_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload["owner"]),
		source_attribution=str(payload["source_attribution"]),
		classification=str(payload.get("classification") or "internal"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def ingest_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.ingest_document(
		document_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		knowledge_base_id=str(payload["knowledge_base_id"]),
		title=str(payload["title"]),
		source_uri=str(payload["source_uri"]),
		content_hash=str(payload["content_hash"]),
		classification=str(payload.get("classification") or "internal"),
		document_count=int(payload.get("document_count", 1)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def retrieve_context(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retrieve_context(
		retrieval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		knowledge_base_id=str(payload["knowledge_base_id"]),
		query=str(payload["query"]),
		document_ids=tuple(payload.get("document_ids") or ()),
		context_confidence=float(payload.get("context_confidence", 1.0)),
		result_window=int(payload.get("result_window", 10)),
		source_classification=str(payload.get("source_classification") or "internal"),
		access_filter_applied=bool(payload.get("access_filter_applied", True)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def generate_answer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.generate_answer(
		answer_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		retrieval_id=str(payload["retrieval_id"]),
		query=str(payload["query"]),
		answer_text=str(payload.get("answer_text") or ""),
		citations=tuple(payload.get("citations") or ()),
		model_location=str(payload.get("model_location") or "local"),
		model_policy_attached=bool(payload.get("model_policy_attached", True)),
		prompt_injection_detected=bool(payload.get("prompt_injection_detected", False)),
		unsafe_answer_detected=bool(payload.get("unsafe_answer_detected", False)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def record_turn(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_turn(
		turn_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		conversation_id=str(payload["conversation_id"]),
		user_id=str(payload["user_id"]),
		query=str(payload["query"]),
		answer_id=str(payload["answer_id"]),
		turn_count=int(payload.get("turn_count", 1)),
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


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def rag_package(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.rag_package(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
