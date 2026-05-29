"""Service layer for the Search Engine capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .search_runtime import (
	QueryRecord,
	SearchAuditEventRecord,
	SearchDocumentRecord,
	SearchIndexRecord,
	normalize_classification,
	normalize_query_type,
	search_required_actions,
	stable_id,
	utc_now,
)


class SrchService:
	"""Deterministic enterprise-search service for APG composition."""

	def __init__(self) -> None:
		self.indices: dict[str, SearchIndexRecord] = {}
		self.documents: dict[str, SearchDocumentRecord] = {}
		self.queries: dict[str, QueryRecord] = {}
		self.audit_events: dict[str, SearchAuditEventRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_index(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		content_type: str = "document",
		classification: str = "internal",
		source_lineage_ref: str | None = None,
		embedding_index_ready: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(name or "").strip():
			raise ValueError("index_name_required")
		context = {
			"tenant_context_present": True,
			"operation": "create_index",
			"owner_assigned": bool(str(owner or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = SearchIndexRecord(
			id=stable_id("srch_index", tenant_id, name),
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			content_type=str(content_type or "document"),
			classification=normalize_classification(classification),
			source_lineage_ref=source_lineage_ref,
			embedding_index_ready=bool(embedding_index_ready),
			status="embedding_ready" if embedding_index_ready else "ready",
		)
		self.indices[record.id] = record
		self._record_event(tenant_id, "index_created", record.id, f"Search index created: {name}", owner)
		return record.to_dict()

	def mark_embedding_index_ready(
		self,
		tenant_id: str,
		index_id: str,
		actor: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, index_id)
		index.embedding_index_ready = True
		index.status = "embedding_ready"
		index.updated_at = utc_now()
		self._record_event(tenant_id, "embedding_index_ready", index.id, f"Embedding index ready: {index.name}", actor)
		return index.to_dict()

	def index_document(
		self,
		tenant_id: str,
		index_id: str,
		document_id: str,
		title: str,
		body: str,
		classification: str | None = None,
		facets: dict[str, str] | None = None,
		metadata: dict[str, Any] | None = None,
		source_lineage_ref: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, index_id)
		if not str(document_id or "").strip():
			raise ValueError("document_id_required")
		if not str(title or "").strip():
			raise ValueError("document_title_required")
		if not str(body or "").strip():
			raise ValueError("document_body_required")
		if not (source_lineage_ref or index.source_lineage_ref):
			raise PermissionError("source_lineage_required")
		record = SearchDocumentRecord(
			id=stable_id("srch_doc", tenant_id, index.id, document_id),
			tenant_id=tenant_id,
			index_id=index.id,
			document_id=document_id,
			title=title,
			body=body,
			classification=normalize_classification(classification or index.classification),
			facets={str(key): str(value) for key, value in dict(facets or {}).items()},
			metadata=dict(metadata or {}),
		)
		self.documents[record.id] = record
		index.document_count = len([item for item in self.documents.values() if item.index_id == index.id])
		index.updated_at = utc_now()
		self._record_event(tenant_id, "document_indexed", record.id, f"Document indexed: {title}", index.owner)
		return record.to_dict()

	def bulk_index_documents(
		self,
		tenant_id: str,
		index_id: str,
		documents: list[dict[str, Any]],
		source_lineage_ref: str | None,
	) -> list[dict[str, Any]]:
		self._require_tenant(tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "bulk_index",
			"source_lineage_present": bool(str(source_lineage_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not documents:
			raise ValueError("bulk_documents_required")
		if len(documents) > int(self.describe(tenant_id)["configuration"]["indexing"]["max_documents_per_batch"]):
			raise ValueError("bulk_document_batch_too_large")
		return [
			self.index_document(
				tenant_id=tenant_id,
				index_id=index_id,
				document_id=str(document["document_id"]),
				title=str(document.get("title") or document["document_id"]),
				body=str(document.get("body") or ""),
				classification=document.get("classification"),
				facets=dict(document.get("facets") or {}),
				metadata=dict(document.get("metadata") or {}),
				source_lineage_ref=source_lineage_ref,
			)
			for document in documents
		]

	def query(
		self,
		tenant_id: str,
		query_text: str,
		index_ids: list[str],
		query_type: str = "keyword",
		result_window: int = 10,
		rbac_filter_applied: bool = True,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(query_text or "").strip():
			raise ValueError("query_text_required")
		if result_window <= 0:
			raise ValueError("result_window_must_be_positive")
		indices = [self._get_index(tenant_id, index_id) for index_id in index_ids]
		normalized_query_type = normalize_query_type(query_type)
		restricted = any(index.classification == "restricted" for index in indices)
		embedding_ready = all(index.embedding_index_ready for index in indices)
		context = {
			"tenant_context_present": True,
			"content_classification": "restricted" if restricted else "internal",
			"rbac_filter_applied": bool(rbac_filter_applied),
			"query_type": "semantic" if normalized_query_type in {"semantic", "hybrid"} else "keyword",
			"embedding_index_ready": bool(embedding_ready),
			"result_window": int(result_window),
			"review_recorded": bool(review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._record_query(tenant_id, query_text, normalized_query_type, indices, result_window, rbac_filter_applied, review_recorded, "denied", 0, result)
			self._raise_policy(result)
		matches = self._search_documents(query_text, [index.id for index in indices], result_window, rbac_filter_applied)
		status = "review_required" if result["decision"] == "require_review" else "completed"
		record = self._record_query(
			tenant_id,
			query_text,
			normalized_query_type,
			indices,
			result_window,
			rbac_filter_applied,
			review_recorded,
			status,
			len(matches),
			result,
		)
		return {
			"query": record,
			"results": matches,
			"facets": self.facets(tenant_id, [index.id for index in indices]),
		}

	def facets(self, tenant_id: str, index_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
		self._require_tenant(tenant_id)
		selected = set(index_ids or [item.id for item in self.indices.values() if item.tenant_id == tenant_id])
		facet_counts: dict[str, dict[str, int]] = {}
		for document in self.documents.values():
			if document.tenant_id != tenant_id or document.index_id not in selected:
				continue
			for key, value in document.facets.items():
				facet_counts.setdefault(key, {})
				facet_counts[key][value] = facet_counts[key].get(value, 0) + 1
		return facet_counts

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_index(
			tenant_id=tenant_id,
			name=record_id,
			owner=str(metadata.get("owner") or "compatibility-owner"),
			content_type=str(metadata.get("content_type") or "document"),
			classification=str(metadata.get("classification") or "internal"),
			source_lineage_ref=metadata.get("source_lineage_ref") or status,
			embedding_index_ready=bool(metadata.get("embedding_index_ready", False)),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_indices(tenant_id)

	def list_indices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.indices, tenant_id)

	def list_documents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.documents, tenant_id)

	def list_queries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.queries, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		indices = self.list_indices(tenant_id)
		documents = self.list_documents(tenant_id)
		queries = self.list_queries(tenant_id)
		return {
			"tenant_id": tenant_id,
			"index_count": len(indices),
			"document_count": len(documents),
			"restricted_index_count": sum(1 for item in indices if item["classification"] == "restricted"),
			"embedding_ready_count": sum(1 for item in indices if item["embedding_index_ready"]),
			"query_count": len(queries),
			"review_required_query_count": sum(1 for item in queries if item["status"] == "review_required"),
			"denied_query_count": sum(1 for item in queries if item["status"] == "denied"),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "srch_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "srch_policy_blocked")

	def _get_index(self, tenant_id: str, index_id: str) -> SearchIndexRecord:
		index = self.indices.get(index_id)
		if index is None:
			index = next((item for item in self.indices.values() if item.tenant_id == tenant_id and item.name == index_id), None)
		if index is None or index.tenant_id != tenant_id:
			raise KeyError(f"search_index_not_found:{index_id}")
		return index

	def _search_documents(
		self,
		query_text: str,
		index_ids: list[str],
		result_window: int,
		rbac_filter_applied: bool,
	) -> list[dict[str, Any]]:
		terms = [term.lower() for term in query_text.split() if term.strip()]
		results: list[dict[str, Any]] = []
		for document in self.documents.values():
			if document.index_id not in index_ids:
				continue
			if document.classification == "restricted" and not rbac_filter_applied:
				continue
			haystack = f"{document.title} {document.body}".lower()
			score = sum(1 for term in terms if term in haystack)
			if score <= 0 and terms:
				continue
			results.append({
				"document_id": document.document_id,
				"title": document.title,
				"index_id": document.index_id,
				"classification": document.classification,
				"score": score or 1,
				"facets": dict(document.facets),
			})
		return sorted(results, key=lambda item: (-int(item["score"]), item["title"]))[:result_window]

	def _record_query(
		self,
		tenant_id: str,
		query_text: str,
		query_type: str,
		indices: list[SearchIndexRecord],
		result_window: int,
		rbac_filter_applied: bool,
		review_recorded: bool,
		status: str,
		result_count: int,
		rule_result: dict[str, Any],
	) -> dict[str, Any]:
		record = QueryRecord(
			id=stable_id("srch_query", tenant_id, query_text, len(self.queries)),
			tenant_id=tenant_id,
			query_text=query_text,
			query_type=query_type,
			index_ids=[index.id for index in indices],
			result_window=int(result_window),
			rbac_filter_applied=bool(rbac_filter_applied),
			review_recorded=bool(review_recorded),
			status=status,
			result_count=result_count,
			required_actions=search_required_actions(rule_result),
			matched_rules=list(rule_result["matched_rules"]),
		)
		self.queries[record.id] = record
		self._record_event(tenant_id, "query_recorded", record.id, f"Search query {status}: {query_text}", "query-api")
		return record.to_dict()

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
	) -> dict[str, Any]:
		record = SearchAuditEventRecord(
			id=stable_id("srch_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])
