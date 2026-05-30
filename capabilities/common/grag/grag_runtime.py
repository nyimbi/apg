"""Dependency-light runtime for the APG GRAG generated-app surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class GragRecord:
	id: str
	tenant_id: str
	kind: str
	status: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"kind": self.kind,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


class GragService:
	"""Import-light GraphRAG lifecycle service for generated APG applications."""

	def __init__(self) -> None:
		self._graph_sources: dict[str, GragRecord] = {}
		self._vector_sources: dict[str, GragRecord] = {}
		self._hybrid_queries: dict[str, GragRecord] = {}
		self._reasoning_paths: dict[str, GragRecord] = {}
		self._answers: dict[str, GragRecord] = {}
		self._curations: dict[str, GragRecord] = {}
		self._publications: dict[str, GragRecord] = {}
		self._audit_events: dict[str, GragRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_graph_source(
		self,
		source_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		graph_id: str,
		provenance_refs: list[str] | tuple[str, ...],
		classification: str = "internal",
		review_recorded: bool = False,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_graph_source",
			"graph_source_id_present": bool(source_id),
			"graph_source_name_present": bool(name),
			"owner_assigned": bool(owner),
			"registered_graph_present": bool(graph_id),
			"provenance_attached": bool(provenance_refs),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = GragRecord(
			id=source_id,
			tenant_id=tenant_id,
			kind="graph_source",
			status="active",
			metadata={
				"name": name,
				"owner": owner,
				"graph_id": graph_id,
				"provenance_refs": list(provenance_refs),
				"classification": classification,
			},
		)
		self._graph_sources[source_id] = record
		self._audit(tenant_id, source_id, "graph_source_registered", owner, result)
		return record.to_dict()

	def retire_graph_source(self, source_id: str, tenant_id: str, reviewer: str, review_recorded: bool = False) -> dict[str, Any]:
		source = self._require_record(self._graph_sources, source_id, tenant_id, "graph_source_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_graph_source",
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = GragRecord(
			id=source.id,
			tenant_id=tenant_id,
			kind="graph_source",
			status="retired",
			metadata={**source.metadata, "retired_by": reviewer},
		)
		self._graph_sources[source_id] = record
		self._audit(tenant_id, source_id, "graph_source_retired", reviewer, result)
		return record.to_dict()

	def register_vector_source(
		self,
		source_id: str,
		tenant_id: str,
		index_id: str,
		embedding_model: str,
		document_refs: list[str] | tuple[str, ...],
		owner: str,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_vector_source",
			"vector_source_id_present": bool(source_id),
			"vector_index_present": bool(index_id),
			"embedding_model_present": bool(embedding_model),
			"source_documents_present": bool(document_refs),
			"owner_assigned": bool(owner),
		})
		self._raise_if_denied(result)
		record = GragRecord(
			id=source_id,
			tenant_id=tenant_id,
			kind="vector_source",
			status="indexed",
			metadata={
				"index_id": index_id,
				"embedding_model": embedding_model,
				"document_refs": list(document_refs),
				"owner": owner,
			},
		)
		self._vector_sources[source_id] = record
		self._audit(tenant_id, source_id, "vector_source_registered", owner, result)
		return record.to_dict()

	def run_hybrid_query(
		self,
		query_id: str,
		tenant_id: str,
		query: str,
		graph_source_id: str,
		vector_source_id: str,
		vector_index_ready: bool = True,
		graph_index_ready: bool = True,
		result_window: int = 10,
		source_classification: str = "internal",
		access_filter_applied: bool = True,
		retrieval_confidence: float = 1.0,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		graph_source = self._require_record(self._graph_sources, graph_source_id, tenant_id, "graph_source_not_found")
		vector_source = self._require_record(self._vector_sources, vector_source_id, tenant_id, "vector_source_not_found")
		confidence = max(0.0, min(1.0, round(float(retrieval_confidence), 4)))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "hybrid_query",
			"query_present": bool(query),
			"graph_source_present": bool(graph_source),
			"vector_source_present": bool(vector_source),
			"vector_index_ready": bool(vector_index_ready),
			"graph_index_ready": bool(graph_index_ready),
			"result_window": int(result_window),
			"source_classification": source_classification,
			"access_filter_applied": bool(access_filter_applied),
			"retrieval_confidence": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = GragRecord(
			id=query_id,
			tenant_id=tenant_id,
			kind="hybrid_query",
			status="reviewed" if review_recorded else "active",
			metadata={
				"query": query,
				"graph_source_id": graph_source_id,
				"vector_source_id": vector_source_id,
				"result_window": int(result_window),
				"source_classification": source_classification,
				"retrieval_confidence": confidence,
			},
		)
		self._hybrid_queries[query_id] = record
		self._audit(tenant_id, query_id, "hybrid_query_run", graph_source_id, result)
		return record.to_dict()

	def build_reasoning_path(
		self,
		path_id: str,
		tenant_id: str,
		query_id: str,
		start_node_id: str,
		evidence_path: list[str] | tuple[str, ...],
		hop_count: int,
		explanation: str,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_record(self._hybrid_queries, query_id, tenant_id, "hybrid_query_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "build_reasoning_path",
			"query_present": bool(query_id),
			"start_node_present": bool(start_node_id),
			"evidence_path_present": bool(evidence_path),
			"hop_count": int(hop_count),
			"explanation_present": bool(explanation),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = GragRecord(
			id=path_id,
			tenant_id=tenant_id,
			kind="reasoning_path",
			status="reviewed" if review_recorded else "active",
			metadata={
				"query_id": query_id,
				"start_node_id": start_node_id,
				"evidence_path": list(evidence_path),
				"hop_count": int(hop_count),
				"explanation": explanation,
			},
		)
		self._reasoning_paths[path_id] = record
		self._audit(tenant_id, path_id, "reasoning_path_built", start_node_id, result)
		return record.to_dict()

	def generate_answer(
		self,
		answer_id: str,
		tenant_id: str,
		query_id: str,
		path_id: str,
		query: str,
		answer_text: str,
		provenance_refs: list[str] | tuple[str, ...],
		citations: list[dict[str, str]] | tuple[dict[str, str], ...],
		model_location: str = "local",
		model_policy_attached: bool = True,
		unsafe_answer_detected: bool = False,
		confidence_score: float = 1.0,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_record(self._hybrid_queries, query_id, tenant_id, "hybrid_query_not_found")
		self._require_record(self._reasoning_paths, path_id, tenant_id, "reasoning_path_not_found")
		confidence = max(0.0, min(1.0, round(float(confidence_score), 4)))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "generate_answer",
			"query_present": bool(query),
			"retrieval_context_present": bool(query_id),
			"reasoning_path_present": bool(path_id),
			"answer_text_present": bool(answer_text),
			"provenance_attached": bool(provenance_refs),
			"citations_attached": bool(citations),
			"model_location": model_location,
			"model_policy_attached": bool(model_policy_attached),
			"unsafe_answer_detected": bool(unsafe_answer_detected),
			"answer_confidence": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = GragRecord(
			id=answer_id,
			tenant_id=tenant_id,
			kind="answer",
			status="generated",
			metadata={
				"query_id": query_id,
				"path_id": path_id,
				"query": query,
				"answer_text": answer_text,
				"provenance_refs": list(provenance_refs),
				"citation_count": len(citations),
				"citations": [dict(citation) for citation in citations],
				"model_location": model_location,
				"confidence_score": confidence,
			},
		)
		self._answers[answer_id] = record
		self._audit(tenant_id, answer_id, "answer_generated", path_id, result)
		return record.to_dict()

	def curate_answer(self, curation_id: str, tenant_id: str, answer_id: str, curator: str, decision: str, evidence: str) -> dict[str, Any]:
		self._require_record(self._answers, answer_id, tenant_id, "answer_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "curate_answer",
			"curator_present": bool(curator),
			"curation_decision_present": decision in {"approved", "rejected", "needs_revision"},
			"evidence_present": bool(evidence),
		})
		self._raise_if_denied(result)
		record = GragRecord(
			id=curation_id,
			tenant_id=tenant_id,
			kind="curation",
			status=decision,
			metadata={"answer_id": answer_id, "curator": curator, "decision": decision, "evidence": evidence},
		)
		self._curations[curation_id] = record
		self._audit(tenant_id, curation_id, "answer_curated", curator, result)
		return record.to_dict()

	def publish_answer(self, publication_id: str, tenant_id: str, answer_id: str, curation_id: str, publisher: str) -> dict[str, Any]:
		self._require_record(self._answers, answer_id, tenant_id, "answer_not_found")
		curation = self._require_record(self._curations, curation_id, tenant_id, "curation_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_answer",
			"curated_answer_present": bool(curation and curation.metadata.get("answer_id") == answer_id and curation.status == "approved"),
		})
		self._raise_if_denied(result)
		record = GragRecord(
			id=publication_id,
			tenant_id=tenant_id,
			kind="publication",
			status="published",
			metadata={"answer_id": answer_id, "curation_id": curation_id, "publisher": publisher},
		)
		self._publications[publication_id] = record
		self._audit(tenant_id, publication_id, "answer_published", publisher, result)
		return record.to_dict()

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		record = self.register_graph_source(
			source_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or metadata.get("title") or record_id),
			owner=str(metadata.get("owner") or "system"),
			graph_id=str(metadata.get("graph_id") or record_id),
			provenance_refs=tuple(metadata.get("provenance_refs") or ("manual",)),
			classification=str(metadata.get("classification") or "internal"),
		)
		return record | {"status": status}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._graph_sources, tenant_id)

	def list_graph_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._graph_sources, tenant_id)

	def list_vector_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._vector_sources, tenant_id)

	def list_hybrid_queries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._hybrid_queries, tenant_id)

	def list_reasoning_paths(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reasoning_paths, tenant_id)

	def list_answers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._answers, tenant_id)

	def list_curations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._curations, tenant_id)

	def list_publications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._publications, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str | None = "default") -> dict[str, Any]:
		answers = self.list_answers(tenant_id)
		queries = self.list_hybrid_queries(tenant_id)
		return {
			"tenant_id": tenant_id,
			"graph_source_count": len(self.list_graph_sources(tenant_id)),
			"vector_source_count": len(self.list_vector_sources(tenant_id)),
			"hybrid_query_count": len(queries),
			"reasoning_path_count": len(self.list_reasoning_paths(tenant_id)),
			"answer_count": len(answers),
			"curation_count": len(self.list_curations(tenant_id)),
			"publication_count": len(self.list_publications(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"citation_count": sum(int(answer["metadata"].get("citation_count", 0)) for answer in answers),
			"low_confidence_query_count": len([query for query in queries if query["metadata"].get("retrieval_confidence", 1.0) < 0.7]),
		}

	def grag_package(self, tenant_id: str | None = None) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"graph_sources": self.list_graph_sources(tenant_id),
			"vector_sources": self.list_vector_sources(tenant_id),
			"hybrid_queries": self.list_hybrid_queries(tenant_id),
			"reasoning_paths": self.list_reasoning_paths(tenant_id),
			"answers": self.list_answers(tenant_id),
			"curations": self.list_curations(tenant_id),
			"publications": self.list_publications(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
			"summary": self.dashboard_summary(tenant_id),
		}

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "grag_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any], review_recorded: bool) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review" and not review_recorded:
			raise PermissionError(", ".join(self._reasons(result)) or "grag_review_required")

	def _require_record(self, records: dict[str, GragRecord], record_id: str, tenant_id: str, reason: str) -> GragRecord:
		record = records.get(record_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(reason)
		return record

	def _audit(self, tenant_id: str, subject_id: str, event_type: str, actor: str, result: dict[str, Any]) -> None:
		event_id = f"audit-{len(self._audit_events):06d}"
		self._audit_events[event_id] = GragRecord(
			id=event_id,
			tenant_id=tenant_id,
			kind="audit_event",
			status=result["decision"],
			metadata={"subject_id": subject_id, "event_type": event_type, "actor": actor, "reasons": self._reasons(result)},
		)

	def _list(self, records: dict[str, GragRecord], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "grag_policy_blocked") for action in result.get("actions", ()))
