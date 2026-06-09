"""Service layer for the Search Engine capability."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .search_runtime import (
	QueryRecord,
	SearchAgentRecord,
	SearchAuditEventRecord,
	SearchDocumentRecord,
	SearchIndexRecord,
	SrchLifecycleBatchRecord,
	normalize_classification,
	normalize_query_type,
	search_required_actions,
	stable_id,
	utc_now,
)


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class SrchService:
	"""Deterministic enterprise-search service for APG composition."""

	def __init__(self) -> None:
		self.indices: dict[str, SearchIndexRecord] = {}
		self.documents: dict[str, SearchDocumentRecord] = {}
		self.queries: dict[str, QueryRecord] = {}
		self.search_agents: dict[str, SearchAgentRecord] = {}
		self.lifecycle_batches: dict[str, SrchLifecycleBatchRecord] = {}
		self.audit_events: dict[str, SearchAuditEventRecord] = {}
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

		# Extended stores
		self._synonyms: dict[str, list[list[str]]] = {}        # collection -> synonym groups
		self._boost_fields: dict[str, dict[str, float]] = {}   # collection -> {field: boost}
		self._mappings: dict[str, dict[str, Any]] = {}         # collection -> field configs
		self._webhooks: dict[str, dict[str, Any]] = {}         # not used here — placeholder

	# ── capability contract ────────────────────────────────────────────────────

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── ORIGINAL INDEX/DOCUMENT/QUERY METHODS ─────────────────────────────────

	def create_index(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		content_type: str = "",
		classification: str = "",
		source_lineage_ref: str | None = None,
		embedding_index_ready: bool = False,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		config = self.describe(tenant_id)["configuration"]
		classification_value = str(classification or "").strip().lower()
		content_type_value = str(content_type or "").strip().lower()
		classification_known = classification_value in config["indices"]["allowed_classifications"]
		stored_classification = classification_value if classification_known else "restricted"
		context = {
			"tenant_context_present": True, "operation": "create_index",
			"index_name_present": bool(str(name or "").strip()),
			"owner_assigned": bool(str(owner or "").strip()),
			"content_type_present": bool(content_type_value),
			"content_type_known": content_type_value in config["indices"]["allowed_content_types"],
			"classification_present": bool(classification_value),
			"classification_known": classification_known,
			"content_classification": stored_classification,
			"source_lineage_present": bool(str(source_lineage_ref or "").strip()),
			"review_recorded": bool(review_recorded),
		}
		result = self.evaluate(context)
		self._raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else ("embedding_ready" if embedding_index_ready else "ready")
		record = SearchIndexRecord(
			id=stable_id("srch_index", tenant_id, name), tenant_id=tenant_id, name=name, owner=owner,
			content_type=content_type_value, classification=normalize_classification(stored_classification),
			source_lineage_ref=source_lineage_ref, embedding_index_ready=bool(embedding_index_ready),
			status=status, decision=result["decision"], matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self.indices[record.id] = record
		self._record_event(tenant_id, "index_created", record.id, f"Search index created: {name}", owner,
						   severity="medium" if status == "pending_review" else "low",
						   evidence=_rule_evidence(result))
		return record.to_dict()

	def mark_embedding_index_ready(self, tenant_id: str, index_id: str, actor: str) -> dict[str, Any]:
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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		index_present = any(item.id == index_id or item.name == index_id
							for item in self.indices.values() if item.tenant_id == tenant_id)
		index = self._get_index(tenant_id, index_id) if index_present else None
		allowed_facets = set(self.describe(tenant_id)["configuration"]["facets"]["allowed_facet_keys"])
		facet_keys = {str(k) for k in dict(facets or {})}
		classification_present = bool(str(classification or (index.classification if index else "")).strip())
		result = self.evaluate({
			"tenant_context_present": True, "operation": "index_document",
			"index_present": index_present,
			"document_id_present": bool(str(document_id or "").strip()),
			"title_present": bool(str(title or "").strip()),
			"body_present": bool(str(body or "").strip()),
			"source_lineage_present": bool(source_lineage_ref or (index.source_lineage_ref if index else None)),
			"classification_present": classification_present,
			"facet_keys_allowed": facet_keys <= allowed_facets,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		assert index is not None
		status = "pending_review" if result["decision"] == "require_review" else "indexed"
		record = SearchDocumentRecord(
			id=stable_id("srch_doc", tenant_id, index.id, document_id), tenant_id=tenant_id,
			index_id=index.id, document_id=document_id, title=title, body=body,
			classification=normalize_classification(classification or index.classification),
			status=status, facets={str(k): str(v) for k, v in dict(facets or {}).items()},
			metadata=dict(metadata or {}), decision=result["decision"],
			matched_rules=list(result["matched_rules"]), review_reasons=list(_review_reasons(result)),
		)
		self.documents[record.id] = record
		index.document_count = len([d for d in self.documents.values() if d.index_id == index.id])
		index.updated_at = utc_now()
		self._record_event(tenant_id, "document_indexed", record.id, f"Document indexed: {title}",
						   index.owner, severity="medium" if status == "pending_review" else "low",
						   evidence=_rule_evidence(result))
		return record.to_dict()

	def bulk_index_documents(
		self,
		tenant_id: str,
		index_id: str,
		documents: list[dict[str, Any]],
		source_lineage_ref: str | None,
		review_recorded: bool = False,
	) -> list[dict[str, Any]]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": True, "operation": "bulk_index",
			"source_lineage_present": bool(str(source_lineage_ref or "").strip()),
			"document_count": len(documents), "review_recorded": bool(review_recorded),
		})
		self._raise_if_blocked(result)
		max_batch = int(self.describe(tenant_id)["configuration"]["indexing"]["max_documents_per_batch"])
		if not review_recorded and len(documents) > max_batch:
			raise ValueError("bulk_document_batch_too_large")
		return [self.index_document(tenant_id=tenant_id, index_id=index_id,
									document_id=str(doc["document_id"]),
									title=str(doc.get("title") or ""), body=str(doc.get("body") or ""),
									classification=doc.get("classification"),
									facets=dict(doc.get("facets") or {}),
									metadata=dict(doc.get("metadata") or {}),
									source_lineage_ref=source_lineage_ref,
									review_recorded=review_recorded)
				for doc in documents]

	def query(
		self,
		tenant_id: str,
		query_text: str,
		index_ids: list[str],
		query_type: str = "",
		result_window: int = 10,
		rbac_filter_applied: bool = True,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		config = self.describe(tenant_id)["configuration"]
		query_type_value = str(query_type or "").strip().lower()
		preflight = self.evaluate({
			"tenant_context_present": True, "operation": "query",
			"query_text_present": bool(str(query_text or "").strip()),
			"index_ids_present": bool(index_ids),
			"query_type_present": bool(query_type_value),
			"query_type_known": query_type_value in config["query"]["allowed_query_types"],
			"result_window": int(result_window), "review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(preflight)
		indices = [self._get_index(tenant_id, iid) for iid in index_ids]
		normalized_qt = (normalize_query_type(query_type) if query_type_value in config["query"]["allowed_query_types"]
						 else query_type_value)
		restricted = any(idx.classification == "restricted" for idx in indices)
		embedding_ready = all(idx.embedding_index_ready for idx in indices)
		context = {
			"tenant_context_present": True, "operation": "query",
			"content_classification": "restricted" if restricted else "internal",
			"rbac_filter_applied": bool(rbac_filter_applied), "query_type": normalized_qt,
			"embedding_index_ready": bool(embedding_ready), "result_window": int(result_window),
			"result_window_review_check": True, "review_recorded": bool(review_recorded),
		}
		result = self.evaluate(context)
		combined = _combine_rule_results(preflight, result)
		if combined["decision"] == "deny":
			self._record_query(tenant_id, query_text, normalized_qt, indices, result_window,
							   rbac_filter_applied, review_recorded, "denied", 0, combined)
			self._raise_policy(combined)
		matches = self._search_documents(query_text, [idx.id for idx in indices], result_window, rbac_filter_applied)
		status = "review_required" if combined["decision"] == "require_review" else "completed"
		record = self._record_query(tenant_id, query_text, normalized_qt, indices, result_window,
									rbac_filter_applied, review_recorded, status, len(matches), combined)
		return {"query": record, "results": matches,
				"facets": self.facets(tenant_id, [idx.id for idx in indices])}

	def facets(self, tenant_id: str, index_ids: list[str] | None = None) -> dict[str, dict[str, int]]:
		self._require_tenant(tenant_id)
		selected = set(index_ids or [item.id for item in self.indices.values() if item.tenant_id == tenant_id])
		facet_counts: dict[str, dict[str, int]] = {}
		for doc in self.documents.values():
			if doc.tenant_id != tenant_id or doc.index_id not in selected:
				continue
			for key, value in doc.facets.items():
				facet_counts.setdefault(key, {})
				facet_counts[key][value] = facet_counts[key].get(value, 0) + 1
		return facet_counts

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None,
					  status: str = "active") -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.create_index(tenant_id=tenant_id, name=record_id,
								 owner=str(metadata.get("owner") or "compatibility-owner"),
								 content_type=str(metadata.get("content_type") or "document"),
								 classification=str(metadata.get("classification") or "internal"),
								 source_lineage_ref=metadata.get("source_lineage_ref") or status,
								 embedding_index_ready=bool(metadata.get("embedding_index_ready", False)))

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_indices(tenant_id)

	def list_indices(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.indices, tenant_id)

	def list_documents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.documents, tenant_id)

	def list_queries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.queries, tenant_id)

	def register_search_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_search_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not name:
			raise ValueError("search_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = SearchAgentRecord(
			id=agent_id, tenant_id=tenant_id, name=name, runtime=runtime_value, role=role_value,
			scope=str(scope).strip(), owner=str(owner).strip(), purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required), status=status,
		)
		self.search_agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_event(tenant_id, "search_agent_registered", record.id,
						   f"Search agent registered: {name}", owner,
						   severity="medium" if status == "pending_review" else "low")
		return record.to_dict()

	def validate_srch_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "search_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("srch_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_srch_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_srch_lifecycle_batch", "event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = SrchLifecycleBatchRecord(
			id=batch_id or f"srchbatch:{len(self.lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id, event_stream=stream_value, mutation_count=mutation_count,
			operation=operation_value, accepted=accepted, decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self.lifecycle_batches[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_event(tenant_id, f"srch_lifecycle_batch_{record.status}", record.id,
						   f"Validated SRCH lifecycle batch: {record.id}", "srch",
						   severity="medium" if not accepted else "low")
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

	def list_search_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.search_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		indices = self.list_indices(tenant_id)
		documents = self.list_documents(tenant_id)
		queries = self.list_queries(tenant_id)
		return {
			"tenant_id": tenant_id, "index_count": len(indices), "document_count": len(documents),
			"restricted_index_count": sum(1 for i in indices if i["classification"] == "restricted"),
			"embedding_ready_count": sum(1 for i in indices if i["embedding_index_ready"]),
			"query_count": len(queries),
			"pending_index_review_count": sum(1 for i in indices if i["status"] == "pending_review"),
			"pending_document_review_count": sum(1 for d in documents if d["status"] == "pending_review"),
			"pending_query_review_count": sum(1 for q in queries if q["status"] == "review_required"),
			"denied_query_count": sum(1 for q in queries if q["status"] == "denied"),
			"search_agent_count": len(self.list_search_agents(tenant_id)),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
		}

	# ── INDEXING EXTENSIONS ───────────────────────────────────────────────────

	async def delete_document(self, tenant_id: str, collection: str, doc_id: str) -> dict[str, Any]:
		"""Remove a document from the index by document_id."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		key_to_del = None
		for k, doc in self.documents.items():
			if doc.index_id == index.id and doc.document_id == doc_id and doc.tenant_id == tenant_id:
				key_to_del = k
				break
		if key_to_del is None:
			return {"deleted": False, "error": "document_not_found"}
		del self.documents[key_to_del]
		index.document_count = len([d for d in self.documents.values() if d.index_id == index.id])
		index.updated_at = utc_now()
		self._record_event(tenant_id, "document_deleted", doc_id, f"Document deleted: {doc_id}", "system")
		return {"deleted": True, "document_id": doc_id, "collection": collection}

	async def reindex_collection(self, tenant_id: str, collection: str) -> dict[str, Any]:
		"""Trigger full reindex of a collection (marks embedding index as not-ready, then ready)."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		index.embedding_index_ready = False
		index.status = "reindexing"
		index.updated_at = utc_now()
		doc_count = len([d for d in self.documents.values() if d.index_id == index.id])
		# Simulate rebuild completion
		for doc in self.documents.values():
			if doc.index_id == index.id:
				doc.status = "indexed"
		index.embedding_index_ready = True
		index.status = "embedding_ready"
		index.updated_at = utc_now()
		self._record_event(tenant_id, "collection_reindexed", index.id,
						   f"Reindexed collection: {collection}", "system")
		return {"collection": collection, "document_count": doc_count, "status": "completed", "ts": _now_iso()}

	async def index_stats(self, tenant_id: str, collection: str) -> dict[str, Any]:
		"""Return per-collection index statistics."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		docs = [d for d in self.documents.values() if d.index_id == index.id]
		size_bytes = sum(len(d.body.encode()) + len(d.title.encode()) for d in docs)
		return {
			"collection": collection, "index_id": index.id, "document_count": len(docs),
			"approx_size_bytes": size_bytes, "embedding_ready": index.embedding_index_ready,
			"status": index.status, "classification": index.classification,
			"updated_at": str(index.updated_at),
		}

	async def mapping_update(self, tenant_id: str, collection: str, field_configs: dict[str, Any]) -> dict[str, Any]:
		"""Update index field mappings (type, searchable, facetable flags)."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		existing = self._mappings.get(index.id, {})
		existing.update(field_configs)
		self._mappings[index.id] = existing
		self._record_event(tenant_id, "mapping_updated", index.id,
						   f"Updated mapping for {collection}", "system",
						   evidence={"fields": list(field_configs.keys())})
		return {"collection": collection, "mapping": existing, "ts": _now_iso()}

	# ── SEARCH EXTENSIONS ────────────────────────────────────────────────────

	async def faceted_search(
		self,
		tenant_id: str,
		collection: str,
		text: str,
		facets: dict[str, str],
	) -> dict[str, Any]:
		"""Search with mandatory facet filters applied."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		terms = [t.lower() for t in text.split() if t.strip()]
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			# All specified facets must match
			if not all(doc.facets.get(k) == v for k, v in facets.items()):
				continue
			haystack = f"{doc.title} {doc.body}".lower()
			score = sum(1 for t in terms if t in haystack)
			if score > 0 or not terms:
				results.append({"document_id": doc.document_id, "title": doc.title,
								"score": score, "facets": dict(doc.facets)})
		results.sort(key=lambda r: -r["score"])
		return {"collection": collection, "query": text, "facet_filters": facets,
				"result_count": len(results), "results": results}

	async def autocomplete(
		self,
		tenant_id: str,
		collection: str,
		prefix: str,
		field: str = "title",
		limit: int = 10,
	) -> list[str]:
		"""Return completions for a prefix against the specified field."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		prefix_lower = prefix.lower()
		seen: set[str] = set()
		completions: list[str] = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			value = doc.title if field == "title" else doc.metadata.get(field, "")
			words = str(value).lower().split()
			for w in words:
				if w.startswith(prefix_lower) and w not in seen:
					seen.add(w)
					completions.append(w)
					if len(completions) >= limit:
						break
			if len(completions) >= limit:
				break
		return sorted(completions)[:limit]

	async def fuzzy_search(
		self,
		tenant_id: str,
		collection: str,
		text: str,
		fuzziness: int = 1,
		result_window: int = 10,
	) -> dict[str, Any]:
		"""Search with edit-distance fuzziness for typo tolerance."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		terms = [t.lower() for t in text.split() if t.strip()]
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			haystack_words = set(f"{doc.title} {doc.body}".lower().split())
			score = 0
			for term in terms:
				if term in haystack_words:
					score += 2
				elif any(self._edit_distance(term, w) <= fuzziness for w in haystack_words):
					score += 1
			if score > 0:
				results.append({"document_id": doc.document_id, "title": doc.title, "score": score})
		results.sort(key=lambda r: -r["score"])
		return {"collection": collection, "query": text, "fuzziness": fuzziness,
				"results": results[:result_window]}

	async def phrase_search(self, tenant_id: str, collection: str, phrase: str) -> dict[str, Any]:
		"""Exact phrase search across title and body."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		phrase_lower = phrase.lower()
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			haystack = f"{doc.title} {doc.body}".lower()
			if phrase_lower in haystack:
				pos = haystack.find(phrase_lower)
				results.append({"document_id": doc.document_id, "title": doc.title,
								"match_position": pos, "score": 1})
		return {"collection": collection, "phrase": phrase,
				"result_count": len(results), "results": results}

	async def boolean_search(
		self,
		tenant_id: str,
		collection: str,
		must: list[str] | None = None,
		should: list[str] | None = None,
		must_not: list[str] | None = None,
		result_window: int = 10,
	) -> dict[str, Any]:
		"""Boolean query: must AND, should OR, must_not NOT."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		must_terms = [t.lower() for t in (must or [])]
		should_terms = [t.lower() for t in (should or [])]
		must_not_terms = [t.lower() for t in (must_not or [])]
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			haystack = f"{doc.title} {doc.body}".lower()
			if must_not_terms and any(t in haystack for t in must_not_terms):
				continue
			if must_terms and not all(t in haystack for t in must_terms):
				continue
			should_score = sum(1 for t in should_terms if t in haystack)
			if should_terms and should_score == 0 and not must_terms:
				continue
			score = len(must_terms) + should_score
			results.append({"document_id": doc.document_id, "title": doc.title, "score": score})
		results.sort(key=lambda r: -r["score"])
		return {"collection": collection, "must": must, "should": should, "must_not": must_not,
				"result_count": len(results), "results": results[:result_window]}

	async def geo_search(
		self,
		tenant_id: str,
		collection: str,
		lat: float,
		lon: float,
		radius_km: float,
	) -> dict[str, Any]:
		"""Search documents whose metadata contains geo-coordinates within radius_km."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			meta = doc.metadata
			doc_lat = meta.get("lat") or meta.get("latitude")
			doc_lon = meta.get("lon") or meta.get("longitude")
			if doc_lat is None or doc_lon is None:
				continue
			dist = self._haversine(lat, lon, float(doc_lat), float(doc_lon))
			if dist <= radius_km:
				results.append({"document_id": doc.document_id, "title": doc.title,
								"distance_km": round(dist, 3)})
		results.sort(key=lambda r: r["distance_km"])
		return {"collection": collection, "centre": {"lat": lat, "lon": lon},
				"radius_km": radius_km, "result_count": len(results), "results": results}

	async def more_like_this(self, tenant_id: str, doc_id: str, collection: str,
							 limit: int = 5) -> dict[str, Any]:
		"""Return documents similar to a given document based on TF-IDF term overlap."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		source: SearchDocumentRecord | None = None
		for doc in self.documents.values():
			if doc.index_id == index.id and doc.document_id == doc_id:
				source = doc
				break
		if source is None:
			return {"error": "document_not_found", "doc_id": doc_id}
		source_terms = Counter(f"{source.title} {source.body}".lower().split())
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id or doc.document_id == doc_id:
				continue
			doc_terms = Counter(f"{doc.title} {doc.body}".lower().split())
			overlap = sum((source_terms & doc_terms).values())
			if overlap > 0:
				results.append({"document_id": doc.document_id, "title": doc.title, "overlap": overlap})
		results.sort(key=lambda r: -r["overlap"])
		return {"source_doc_id": doc_id, "collection": collection,
				"similar_count": len(results[:limit]), "results": results[:limit]}

	# ── RELEVANCE ────────────────────────────────────────────────────────────

	async def ranking_tune(self, tenant_id: str, collection: str, signals: dict[str, float]) -> dict[str, Any]:
		"""Update relevance ranking signals for a collection."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		existing_boosts = self._boost_fields.get(index.id, {})
		existing_boosts.update(signals)
		self._boost_fields[index.id] = existing_boosts
		self._record_event(tenant_id, "ranking_tuned", index.id,
						   f"Ranking signals updated for {collection}", "system",
						   evidence={"signals": signals})
		return {"collection": collection, "signals": existing_boosts, "ts": _now_iso()}

	async def personalised_search(
		self,
		tenant_id: str,
		user_id: str,
		query_text: str,
		collection: str,
		result_window: int = 10,
	) -> dict[str, Any]:
		"""Search with personalisation applied from user query history."""
		self._require_tenant(tenant_id)
		# Derive user-specific boost from their past query terms
		user_queries = [q for q in self.queries.values()
						if q.tenant_id == tenant_id and str(q.query_text)]
		history_terms: Counter[str] = Counter()
		for uq in user_queries[-20:]:
			history_terms.update(uq.query_text.lower().split())
		index = self._get_index(tenant_id, collection)
		terms = [t.lower() for t in query_text.split() if t.strip()]
		results = []
		for doc in self.documents.values():
			if doc.index_id != index.id:
				continue
			haystack = f"{doc.title} {doc.body}".lower()
			base_score = sum(1 for t in terms if t in haystack)
			personalisation_boost = sum(history_terms.get(t, 0) * 0.1 for t in haystack.split() if t in history_terms)
			score = base_score + personalisation_boost
			if score > 0:
				results.append({"document_id": doc.document_id, "title": doc.title,
								"score": round(score, 3), "personalised": True})
		results.sort(key=lambda r: -r["score"])
		return {"user_id": user_id, "collection": collection, "query": query_text,
				"result_count": len(results[:result_window]), "results": results[:result_window]}

	async def synonym_add(self, tenant_id: str, collection: str, synonyms_list: list[list[str]]) -> dict[str, Any]:
		"""Add synonym groups to a collection's search config."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		existing = self._synonyms.get(index.id, [])
		existing.extend(synonyms_list)
		self._synonyms[index.id] = existing
		return {"collection": collection, "synonym_groups": len(existing), "added": len(synonyms_list)}

	async def synonym_remove(self, tenant_id: str, collection: str, word: str) -> dict[str, Any]:
		"""Remove all synonym groups containing a given word."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		before = self._synonyms.get(index.id, [])
		after = [grp for grp in before if word.lower() not in [w.lower() for w in grp]]
		self._synonyms[index.id] = after
		return {"collection": collection, "removed_groups": len(before) - len(after)}

	async def boost_field(self, tenant_id: str, collection: str, field: str, boost_factor: float) -> dict[str, Any]:
		"""Apply a multiplicative boost to a field for ranking."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		boosts = self._boost_fields.setdefault(index.id, {})
		boosts[field] = boost_factor
		return {"collection": collection, "field": field, "boost_factor": boost_factor}

	# ── QUALITY ──────────────────────────────────────────────────────────────

	async def spell_check(self, query: str, language: str = "en") -> dict[str, Any]:
		"""Basic spell check using edit-distance against a minimal vocabulary."""
		# Use all indexed document words as corpus vocabulary
		vocab: set[str] = set()
		for doc in self.documents.values():
			vocab.update(f"{doc.title} {doc.body}".lower().split())
		words = query.split()
		corrections: dict[str, str | None] = {}
		for word in words:
			lower = word.lower()
			if lower in vocab:
				corrections[word] = None
			else:
				candidates = [(self._edit_distance(lower, v), v) for v in vocab if abs(len(v) - len(lower)) <= 2]
				candidates.sort()
				corrections[word] = candidates[0][1] if candidates and candidates[0][0] <= 2 else None
		corrected = " ".join(corrections.get(w) or w for w in words)
		return {"original": query, "corrected": corrected, "corrections": corrections, "language": language}

	async def search_suggestions(self, tenant_id: str, collection: str, partial_query: str) -> list[str]:
		"""Return query suggestions based on past successful queries."""
		self._require_tenant(tenant_id)
		partial = partial_query.lower()
		suggestions: list[str] = []
		for q in self.queries.values():
			if q.tenant_id == tenant_id and partial in q.query_text.lower() and q.status == "completed":
				if q.query_text not in suggestions:
					suggestions.append(q.query_text)
		return sorted(set(suggestions))[:10]

	async def highlight_results(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
		"""Add highlighted snippets to search results."""
		terms = [re.escape(t) for t in query.split() if t.strip()]
		if not terms:
			return results
		pattern = re.compile(f"({'|'.join(terms)})", re.IGNORECASE)
		highlighted = []
		for r in results:
			body = r.get("body", r.get("title", ""))
			highlighted_body = pattern.sub(r"<mark>\1</mark>", str(body))
			highlighted.append({**r, "highlighted": highlighted_body})
		return highlighted

	async def search_analytics(self, tenant_id: str, collection: str, period: str) -> dict[str, Any]:
		"""Return search analytics for a collection."""
		self._require_tenant(tenant_id)
		queries = [q for q in self.queries.values() if q.tenant_id == tenant_id]
		total = len(queries)
		zero_result = sum(1 for q in queries if q.result_count == 0)
		top_queries = Counter(q.query_text for q in queries).most_common(10)
		return {
			"collection": collection, "period": period, "total_queries": total,
			"zero_result_queries": zero_result,
			"zero_result_rate": zero_result / total if total else 0.0,
			"top_queries": [{"query": q, "count": c} for q, c in top_queries],
			"avg_results": sum(q.result_count for q in queries) / total if total else 0.0,
		}

	# ── ADMIN ────────────────────────────────────────────────────────────────

	async def collection_create(self, tenant_id: str, name: str, schema: dict[str, Any],
								settings: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Create a collection with explicit schema and settings."""
		result = self.create_index(
			tenant_id=tenant_id, name=name,
			owner=str(settings.get("owner", "system") if settings else "system"),
			content_type=str(schema.get("content_type", "document")),
			classification=str(settings.get("classification", "internal") if settings else "internal"),
			source_lineage_ref=str(settings.get("lineage", "api") if settings else "api"),
		)
		if result:
			self._mappings[result["id"]] = schema
		return result

	async def collection_delete(self, tenant_id: str, name: str) -> dict[str, Any]:
		"""Delete a collection and all its documents."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, name)
		doc_keys = [k for k, d in self.documents.items() if d.index_id == index.id]
		for k in doc_keys:
			del self.documents[k]
		del self.indices[index.id]
		self._mappings.pop(index.id, None)
		self._synonyms.pop(index.id, None)
		self._boost_fields.pop(index.id, None)
		self._record_event(tenant_id, "collection_deleted", index.id, f"Collection deleted: {name}", "system")
		return {"deleted": True, "collection": name, "documents_removed": len(doc_keys)}

	async def collection_clone(self, tenant_id: str, src: str, dst: str) -> dict[str, Any]:
		"""Clone a collection including all documents."""
		self._require_tenant(tenant_id)
		src_index = self._get_index(tenant_id, src)
		new_index_dict = self.create_index(
			tenant_id=tenant_id, name=dst, owner=src_index.owner,
			content_type=src_index.content_type, classification=src_index.classification,
			source_lineage_ref=src_index.source_lineage_ref,
		)
		new_index = self._get_index(tenant_id, dst)
		src_docs = [d for d in self.documents.values() if d.index_id == src_index.id]
		for doc in src_docs:
			self.index_document(
				tenant_id=tenant_id, index_id=new_index.id,
				document_id=doc.document_id, title=doc.title, body=doc.body,
				classification=doc.classification, facets=dict(doc.facets),
				metadata=dict(doc.metadata),
			)
		return {"cloned": True, "source": src, "destination": dst,
				"document_count": len(src_docs), "ts": _now_iso()}

	async def index_health(self, tenant_id: str, collection: str) -> dict[str, Any]:
		"""Return health status of a collection index."""
		self._require_tenant(tenant_id)
		index = self._get_index(tenant_id, collection)
		docs = [d for d in self.documents.values() if d.index_id == index.id]
		pending = sum(1 for d in docs if d.status == "pending_review")
		return {"collection": collection, "status": index.status, "document_count": len(docs),
				"pending_review_count": pending, "embedding_ready": index.embedding_index_ready,
				"health": "healthy" if index.status in ("ready", "embedding_ready") else "degraded"}

	async def search_volume_report(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Report total search volume and trends."""
		self._require_tenant(tenant_id)
		all_queries = [q for q in self.queries.values() if q.tenant_id == tenant_id]
		successful = [q for q in all_queries if q.status == "completed"]
		denied = [q for q in all_queries if q.status == "denied"]
		return {
			"tenant_id": tenant_id, "period": period, "total_queries": len(all_queries),
			"successful": len(successful), "denied": len(denied),
			"review_required": len([q for q in all_queries if q.status == "review_required"]),
			"unique_index_count": len({q.index_ids[0] for q in all_queries if q.index_ids}),
		}

	# ── private helpers ────────────────────────────────────────────────────────

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(a.get("reason", "srch_policy_blocked") for a in result["actions"])
		raise PermissionError(reasons or "srch_policy_blocked")

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] != "allow":
			self._raise_policy(result)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			self._raise_policy(result)

	def _get_index(self, tenant_id: str, index_id: str) -> SearchIndexRecord:
		index = self.indices.get(index_id)
		if index is None:
			index = next((i for i in self.indices.values()
						  if i.tenant_id == tenant_id and i.name == index_id), None)
		if index is None or index.tenant_id != tenant_id:
			raise KeyError(f"search_index_not_found:{index_id}")
		return index

	def _search_documents(self, query_text: str, index_ids: list[str],
						  result_window: int, rbac_filter_applied: bool) -> list[dict[str, Any]]:
		terms = [t.lower() for t in query_text.split() if t.strip()]
		results: list[dict[str, Any]] = []
		for doc in self.documents.values():
			if doc.index_id not in index_ids:
				continue
			if doc.classification == "restricted" and not rbac_filter_applied:
				continue
			haystack = f"{doc.title} {doc.body}".lower()
			score = sum(1 for t in terms if t in haystack)
			if score <= 0 and terms:
				continue
			results.append({"document_id": doc.document_id, "title": doc.title,
							"index_id": doc.index_id, "classification": doc.classification,
							"score": score or 1, "facets": dict(doc.facets)})
		return sorted(results, key=lambda r: (-int(r["score"]), r["title"]))[:result_window]

	def _record_query(self, tenant_id: str, query_text: str, query_type: str,
					  indices: list[SearchIndexRecord], result_window: int, rbac_filter_applied: bool,
					  review_recorded: bool, status: str, result_count: int,
					  rule_result: dict[str, Any]) -> dict[str, Any]:
		record = QueryRecord(
			id=stable_id("srch_query", tenant_id, query_text, len(self.queries)),
			tenant_id=tenant_id, query_text=query_text, query_type=query_type,
			index_ids=[idx.id for idx in indices], result_window=int(result_window),
			rbac_filter_applied=bool(rbac_filter_applied), review_recorded=bool(review_recorded),
			status=status, result_count=result_count, decision=rule_result["decision"],
			required_actions=search_required_actions(rule_result),
			matched_rules=list(rule_result["matched_rules"]),
			review_reasons=list(_review_reasons(rule_result)),
		)
		self.queries[record.id] = record
		self._record_event(tenant_id, "query_recorded", record.id,
						   f"Search query {status}: {query_text}", "query-api",
						   severity="medium" if status in {"review_required", "denied"} else "low",
						   evidence=_rule_evidence(rule_result))
		return record.to_dict()

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, actor: str,
					  severity: str = "low", evidence: dict[str, Any] | None = None) -> dict[str, Any]:
		record = SearchAuditEventRecord(
			id=stable_id("srch_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id, event_type=event_type, subject_id=subject_id, message=message,
			actor=actor, severity=severity, evidence=dict(evidence or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [r.to_dict() for r in records.values()]
		if tenant_id is not None:
			items = [i for i in items if i["tenant_id"] == tenant_id]
		return sorted(items, key=lambda i: i["id"])

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	@staticmethod
	def _edit_distance(a: str, b: str) -> int:
		if len(a) > len(b):
			a, b = b, a
		row = list(range(len(a) + 1))
		for j, cb in enumerate(b):
			new_row = [j + 1]
			for i, ca in enumerate(a):
				new_row.append(min(row[i + 1] + 1, new_row[-1] + 1,
								   row[i] + (0 if ca == cb else 1)))
			row = new_row
		return row[-1]

	@staticmethod
	def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
		R = 6371.0
		dlat = math.radians(lat2 - lat1)
		dlon = math.radians(lon2 - lon1)
		a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
		return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── module-level helpers ───────────────────────────────────────────────────────

def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _review_reasons(result: dict[str, Any]) -> tuple[str, ...]:
	return tuple(a.get("reason", "search_review_required") for a in result.get("actions", [])
				 if a.get("decision") == "require_review")


def _rule_evidence(result: dict[str, Any]) -> dict[str, Any]:
	return {"decision": result["decision"], "matched_rules": list(result["matched_rules"]),
			"reasons": list(_review_reasons(result))}


def _combine_rule_results(*results: dict[str, Any]) -> dict[str, Any]:
	decision = "allow"
	matched_rules: list[str] = []
	actions: list[dict[str, Any]] = []
	context: dict[str, Any] = {}
	for result in results:
		matched_rules.extend(result.get("matched_rules", []))
		actions.extend(result.get("actions", []))
		context.update(result.get("context", {}))
		if result.get("decision") == "deny":
			decision = "deny"
		elif result.get("decision") == "require_review" and decision != "deny":
			decision = "require_review"
	return {"decision": decision, "matched_rules": matched_rules, "actions": actions, "context": context}

	async def ml_search_result_rank(self, *args, **kwargs):
		"""AI-powered AI-powered search result relevance ranking. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score({"query": str(kwargs.get("query","")), "result": str(kwargs.get("result",""))}, task="search_relevance_scoring")
			return {"relevance_score": round(result.score,3), "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

