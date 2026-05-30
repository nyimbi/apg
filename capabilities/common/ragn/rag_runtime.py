"""Dependency-light runtime for the APG RAGN generated-app surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class RagnRecord:
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


class RagnService:
	"""Import-light RAG lifecycle service for generated APG applications."""

	def __init__(self) -> None:
		self._knowledge_bases: dict[str, RagnRecord] = {}
		self._documents: dict[str, RagnRecord] = {}
		self._retrievals: dict[str, RagnRecord] = {}
		self._answers: dict[str, RagnRecord] = {}
		self._conversations: dict[str, RagnRecord] = {}
		self._curations: dict[str, RagnRecord] = {}
		self._audit_events: dict[str, RagnRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_knowledge_base(
		self,
		knowledge_base_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		source_attribution: str,
		classification: str = "internal",
		review_recorded: bool = False,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_knowledge_base",
			"knowledge_base_id_present": bool(knowledge_base_id),
			"knowledge_base_name_present": bool(name),
			"owner_assigned": bool(owner),
			"source_attribution_present": bool(source_attribution),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = RagnRecord(
			id=knowledge_base_id,
			tenant_id=tenant_id,
			kind="knowledge_base",
			status="active",
			metadata={
				"name": name,
				"owner": owner,
				"source_attribution": source_attribution,
				"classification": classification,
			},
		)
		self._knowledge_bases[knowledge_base_id] = record
		self._audit(tenant_id, knowledge_base_id, "knowledge_base_created", owner, result)
		return record.to_dict()

	def ingest_document(
		self,
		document_id: str,
		tenant_id: str,
		knowledge_base_id: str,
		title: str,
		source_uri: str,
		content_hash: str,
		classification: str = "internal",
		document_count: int = 1,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_record(self._knowledge_bases, knowledge_base_id, tenant_id, "knowledge_base_not_found")
		allowed = classification in {"public", "internal", "confidential", "restricted"}
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "ingest_document",
			"knowledge_base_present": bool(knowledge_base_id),
			"document_title_present": bool(title),
			"content_hash_present": bool(content_hash),
			"source_uri_present": bool(source_uri),
			"classification_allowed": allowed,
			"document_count": document_count,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = RagnRecord(
			id=document_id,
			tenant_id=tenant_id,
			kind="document",
			status="indexed",
			metadata={
				"knowledge_base_id": knowledge_base_id,
				"title": title,
				"source_uri": source_uri,
				"content_hash": content_hash,
				"classification": classification,
			},
		)
		self._documents[document_id] = record
		self._audit(tenant_id, document_id, "document_ingested", knowledge_base_id, result)
		return record.to_dict()

	def retrieve_context(
		self,
		retrieval_id: str,
		tenant_id: str,
		knowledge_base_id: str,
		query: str,
		document_ids: list[str] | tuple[str, ...],
		context_confidence: float,
		result_window: int = 10,
		source_classification: str = "internal",
		access_filter_applied: bool = True,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_record(self._knowledge_bases, knowledge_base_id, tenant_id, "knowledge_base_not_found")
		for document_id in document_ids:
			self._require_record(self._documents, document_id, tenant_id, "document_not_found")
		confidence = max(0.0, min(1.0, round(float(context_confidence), 4)))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retrieve_context",
			"query_present": bool(query),
			"knowledge_base_present": bool(knowledge_base_id),
			"result_window": result_window,
			"source_classification": source_classification,
			"access_filter_applied": bool(access_filter_applied),
			"context_confidence": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = RagnRecord(
			id=retrieval_id,
			tenant_id=tenant_id,
			kind="retrieval",
			status="reviewed" if review_recorded else "active",
			metadata={
				"knowledge_base_id": knowledge_base_id,
				"query": query,
				"document_ids": list(document_ids),
				"context_confidence": confidence,
				"result_window": result_window,
				"source_classification": source_classification,
			},
		)
		self._retrievals[retrieval_id] = record
		self._audit(tenant_id, retrieval_id, "context_retrieved", knowledge_base_id, result)
		return record.to_dict()

	def generate_answer(
		self,
		answer_id: str,
		tenant_id: str,
		retrieval_id: str,
		query: str,
		answer_text: str,
		citations: list[dict[str, str]] | tuple[dict[str, str], ...],
		model_location: str = "local",
		model_policy_attached: bool = True,
		prompt_injection_detected: bool = False,
		unsafe_answer_detected: bool = False,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		retrieval = self._require_record(self._retrievals, retrieval_id, tenant_id, "retrieval_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "generate_answer",
			"query_present": bool(query),
			"context_present": bool(retrieval),
			"answer_text_present": bool(answer_text),
			"citations_attached": bool(citations),
			"model_location": model_location,
			"model_policy_attached": bool(model_policy_attached),
			"prompt_injection_detected": bool(prompt_injection_detected),
			"unsafe_answer_detected": bool(unsafe_answer_detected),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		for citation in citations:
			self.attach_citation(
				citation_id=f"{answer_id}:{citation.get('chunk_id', len(self._audit_events))}",
				tenant_id=tenant_id,
				source_id=str(citation.get("source_id") or ""),
				document_id=str(citation.get("document_id") or ""),
				chunk_id=str(citation.get("chunk_id") or ""),
			)
		record = RagnRecord(
			id=answer_id,
			tenant_id=tenant_id,
			kind="answer",
			status="generated",
			metadata={
				"retrieval_id": retrieval_id,
				"query": query,
				"answer_text": answer_text,
				"citation_count": len(citations),
				"model_location": model_location,
			},
		)
		self._answers[answer_id] = record
		self._audit(tenant_id, answer_id, "answer_generated", retrieval_id, result)
		return record.to_dict()

	def record_turn(
		self,
		turn_id: str,
		tenant_id: str,
		conversation_id: str,
		user_id: str,
		query: str,
		answer_id: str,
		turn_count: int,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_record(self._answers, answer_id, tenant_id, "answer_not_found")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_turn",
			"conversation_id_present": bool(conversation_id),
			"user_id_present": bool(user_id),
			"turn_count": turn_count,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_review_required(result, review_recorded)
		record = RagnRecord(
			id=turn_id,
			tenant_id=tenant_id,
			kind="conversation_turn",
			status="recorded",
			metadata={"conversation_id": conversation_id, "user_id": user_id, "query": query, "answer_id": answer_id, "turn_count": turn_count},
		)
		self._conversations[turn_id] = record
		self._audit(tenant_id, turn_id, "conversation_turn_recorded", user_id, result)
		return record.to_dict()

	def attach_citation(self, citation_id: str, tenant_id: str, source_id: str, document_id: str, chunk_id: str) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "attach_citation",
			"source_id_present": bool(source_id),
			"document_id_present": bool(document_id),
			"chunk_id_present": bool(chunk_id),
		})
		self._raise_if_denied(result)
		record = RagnRecord(
			id=citation_id,
			tenant_id=tenant_id,
			kind="citation",
			status="attached",
			metadata={"source_id": source_id, "document_id": document_id, "chunk_id": chunk_id},
		)
		self._audit(tenant_id, citation_id, "citation_attached", document_id, result)
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
		record = RagnRecord(
			id=curation_id,
			tenant_id=tenant_id,
			kind="curation",
			status=decision,
			metadata={"answer_id": answer_id, "curator": curator, "decision": decision, "evidence": evidence},
		)
		self._curations[curation_id] = record
		self._audit(tenant_id, curation_id, "answer_curated", curator, result)
		return record.to_dict()

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_knowledge_base(
			knowledge_base_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or metadata.get("title") or record_id),
			owner=str(metadata.get("owner") or "system"),
			source_attribution=str(metadata.get("source_attribution") or "manual"),
			classification=str(metadata.get("classification") or "internal"),
		) | {"status": status}

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._knowledge_bases, tenant_id)

	def list_knowledge_bases(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._knowledge_bases, tenant_id)

	def list_documents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._documents, tenant_id)

	def list_retrievals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._retrievals, tenant_id)

	def list_answers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._answers, tenant_id)

	def list_conversations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._conversations, tenant_id)

	def list_curations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._curations, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str | None = "default") -> dict[str, Any]:
		kbs = self.list_knowledge_bases(tenant_id)
		documents = self.list_documents(tenant_id)
		answers = self.list_answers(tenant_id)
		return {
			"tenant_id": tenant_id,
			"knowledge_base_count": len(kbs),
			"document_count": len(documents),
			"retrieval_count": len(self.list_retrievals(tenant_id)),
			"answer_count": len(answers),
			"conversation_turn_count": len(self.list_conversations(tenant_id)),
			"curation_count": len(self.list_curations(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"citation_count": sum(int(answer["metadata"].get("citation_count", 0)) for answer in answers),
			"restricted_document_count": len([doc for doc in documents if doc["metadata"].get("classification") == "restricted"]),
		}

	def rag_package(self, tenant_id: str | None = None) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"knowledge_bases": self.list_knowledge_bases(tenant_id),
			"documents": self.list_documents(tenant_id),
			"retrievals": self.list_retrievals(tenant_id),
			"answers": self.list_answers(tenant_id),
			"conversations": self.list_conversations(tenant_id),
			"curations": self.list_curations(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
			"summary": self.dashboard_summary(tenant_id),
		}

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "ragn_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any], review_recorded: bool) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review" and not review_recorded:
			raise PermissionError(", ".join(self._reasons(result)) or "ragn_review_required")

	def _require_record(self, records: dict[str, RagnRecord], record_id: str, tenant_id: str, reason: str) -> RagnRecord:
		record = records.get(record_id)
		if record is None or record.tenant_id != tenant_id:
			raise KeyError(reason)
		return record

	def _audit(self, tenant_id: str, subject_id: str, event_type: str, actor: str, result: dict[str, Any]) -> None:
		event_id = f"audit-{len(self._audit_events):06d}"
		self._audit_events[event_id] = RagnRecord(
			id=event_id,
			tenant_id=tenant_id,
			kind="audit_event",
			status=result["decision"],
			metadata={"subject_id": subject_id, "event_type": event_type, "actor": actor, "reasons": self._reasons(result)},
		)

	def _list(self, records: dict[str, RagnRecord], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "ragn_policy_blocked") for action in result.get("actions", ()))
