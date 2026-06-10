"""Document & eDiscovery — async service layer."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

DOCUMENT_TYPES = {
	"pleading", "brief", "contract", "correspondence", "evidence",
	"internal", "court_order", "affidavit", "exhibit", "expert_report",
}
PRIVILEGE_TYPES = {"attorney_client", "work_product", "common_interest", "settlement", "deliberative"}
PRODUCTION_FORMATS = {"pdf", "tiff", "native", "searchable_pdf"}


class DocumentEDiscoveryService:
	"""In-memory async service for document repository and eDiscovery."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.documents: dict[str, dict[str, Any]] = {}
		self.privilege_log: dict[str, dict[str, Any]] = {}
		self.litigation_holds: dict[str, dict[str, Any]] = {}
		self.hold_documents: dict[str, set[str]] = {}  # hold_id -> set of doc_ids
		self.production_sets: dict[str, dict[str, Any]] = {}
		self.access_log: list[dict[str, Any]] = []
		self._audit_events: list[dict[str, Any]] = []

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}{uuid4().hex[:12]}"

	def _tenant(self, tenant_id: str | None = None) -> str:
		val = tenant_id or self.tenant_id
		guard_tenant_id(val)
		return val

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("evt-"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"details": details or {},
			"created_at": self._now(),
		})

	# ── Health ───────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "leg_dsc",
			"status": "healthy",
			"document_count": len(self.documents),
			"active_holds": sum(1 for h in self.litigation_holds.values() if h["status"] == "active"),
			"privileged_docs": sum(1 for d in self.documents.values() if d.get("is_privileged")),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "leg_dsc",
			"name": "Document & eDiscovery",
			"domain": "legal",
			"version": "1.0.0",
			"document_types": sorted(DOCUMENT_TYPES),
			"privilege_types": sorted(PRIVILEGE_TYPES),
			"production_formats": sorted(PRODUCTION_FORMATS),
		}

	# ── Documents ────────────────────────────────────────────────────────────

	async def create_document(
		self,
		tenant_id: str,
		title: str,
		document_type: str,
		owner_id: str,
		file_reference: str,
		matter_id: str | None = None,
		file_size_bytes: int = 0,
		mime_type: str = "application/pdf",
		description: str = "",
		tags: list[str] | None = None,
		is_privileged: bool = False,
		privilege_type: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Ingest a document into the repository."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(file_reference, "file_reference")
		if document_type not in DOCUMENT_TYPES:
			raise ValueError(f"document_type must be one of {DOCUMENT_TYPES}")
		if is_privileged and privilege_type and privilege_type not in PRIVILEGE_TYPES:
			raise ValueError(f"privilege_type must be one of {PRIVILEGE_TYPES}")
		record: dict[str, Any] = {
			"id": self._id("doc-"),
			"tenant_id": tenant,
			"title": title,
			"document_type": document_type,
			"matter_id": matter_id,
			"owner_id": owner_id,
			"file_reference": file_reference,
			"file_size_bytes": file_size_bytes,
			"mime_type": mime_type,
			"description": description,
			"tags": list(tags or []),
			"is_privileged": is_privileged,
			"privilege_type": privilege_type,
			"version": 1,
			"status": "active",
			"on_hold": False,
			"hold_ids": [],
			"metadata": dict(metadata or {}),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.documents[record["id"]] = record
		if is_privileged and privilege_type:
			await self._auto_log_privilege(tenant, record["id"], privilege_type, owner_id)
		self._emit(tenant, "document_created", record["id"], {"title": title, "type": document_type})
		_log.info("document created tenant=%s id=%s type=%s", tenant, record["id"], document_type)
		return deepcopy(record)

	async def _auto_log_privilege(self, tenant: str, doc_id: str, privilege_type: str, owner_id: str) -> None:
		"""Auto-create privilege log entry on privileged document creation."""
		entry: dict[str, Any] = {
			"id": self._id("prv-"),
			"tenant_id": tenant,
			"document_id": doc_id,
			"privilege_type": privilege_type,
			"basis": "auto_flagged_on_upload",
			"logged_by_id": owner_id,
			"notes": "Automatically logged at upload",
			"status": "logged",
			"created_at": self._now(),
		}
		self.privilege_log[entry["id"]] = entry

	async def get_document(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document {document_id} not found")
		self.access_log.append({"tenant_id": tenant, "document_id": document_id, "accessed_at": self._now()})
		return deepcopy(doc)

	async def list_documents(
		self,
		tenant_id: str,
		matter_id: str | None = None,
		document_type: str | None = None,
		is_privileged: bool | None = None,
		on_hold: bool | None = None,
		tags: list[str] | None = None,
	) -> list[dict[str, Any]]:
		"""List documents with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(d) for d in self.documents.values() if d["tenant_id"] == tenant]
		if matter_id is not None:
			items = [d for d in items if d["matter_id"] == matter_id]
		if document_type:
			items = [d for d in items if d["document_type"] == document_type]
		if is_privileged is not None:
			items = [d for d in items if d["is_privileged"] == is_privileged]
		if on_hold is not None:
			items = [d for d in items if d["on_hold"] == on_hold]
		if tags:
			items = [d for d in items if any(t in d["tags"] for t in tags)]
		return items

	async def update_document(self, tenant_id: str, document_id: str, **updates: Any) -> dict[str, Any]:
		"""Update document metadata."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document {document_id} not found")
		if doc.get("on_hold"):
			raise ValueError("document under litigation hold cannot be modified")
		allowed = {"title", "description", "tags", "is_privileged", "privilege_type", "metadata"}
		for k, v in updates.items():
			if k in allowed and v is not None:
				doc[k] = v
		doc["updated_at"] = self._now()
		doc["version"] = doc.get("version", 1) + 1
		self._emit(tenant, "document_updated", document_id, {"version": doc["version"]})
		return deepcopy(doc)

	async def delete_document(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Soft-delete (archive) a document."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document {document_id} not found")
		if doc.get("on_hold"):
			raise ValueError("cannot delete document under litigation hold")
		doc["status"] = "archived"
		doc["updated_at"] = self._now()
		self._emit(tenant, "document_archived", document_id)
		return deepcopy(doc)

	async def search_documents(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Full-text search across document titles and descriptions."""
		tenant = self._tenant(tenant_id)
		q = query.lower()
		return [
			deepcopy(d) for d in self.documents.values()
			if d["tenant_id"] == tenant and d["status"] == "active" and (
				q in d["title"].lower() or q in d.get("description", "").lower()
			)
		]

	# ── Privilege Log ────────────────────────────────────────────────────────

	async def log_privilege(
		self,
		tenant_id: str,
		document_id: str,
		privilege_type: str,
		basis: str,
		logged_by_id: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Log a privilege assertion for a document."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document {document_id} not found")
		entry: dict[str, Any] = {
			"id": self._id("prv-"),
			"tenant_id": tenant,
			"document_id": document_id,
			"privilege_type": privilege_type,
			"basis": basis,
			"logged_by_id": logged_by_id,
			"notes": notes,
			"status": "logged",
			"created_at": self._now(),
		}
		self.privilege_log[entry["id"]] = entry
		doc["is_privileged"] = True
		doc["privilege_type"] = privilege_type
		self._emit(tenant, "privilege_logged", entry["id"], {"document_id": document_id, "type": privilege_type})
		return deepcopy(entry)

	async def list_privilege_log(self, tenant_id: str, document_id: str | None = None) -> list[dict[str, Any]]:
		"""List privilege log entries."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.privilege_log.values() if e["tenant_id"] == tenant]
		if document_id:
			items = [e for e in items if e["document_id"] == document_id]
		return items

	# ── Litigation Holds ─────────────────────────────────────────────────────

	async def create_litigation_hold(
		self,
		tenant_id: str,
		matter_id: str,
		title: str,
		description: str,
		custodian_ids: list[str],
		issued_by_id: str,
		scope_query: str = "",
	) -> dict[str, Any]:
		"""Issue a litigation hold on a matter."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		# Auto-apply hold to matching documents
		matched_docs = [
			d["id"] for d in self.documents.values()
			if d["tenant_id"] == tenant
			and d["matter_id"] == matter_id
			and d["status"] == "active"
		]
		if scope_query:
			q = scope_query.lower()
			matched_docs = [
				d["id"] for d in self.documents.values()
				if d["tenant_id"] == tenant and d["status"] == "active"
				and (q in d["title"].lower() or any(q in tag for tag in d.get("tags", [])))
			]
		hold: dict[str, Any] = {
			"id": self._id("hld-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"title": title,
			"description": description,
			"custodian_ids": custodian_ids,
			"issued_by_id": issued_by_id,
			"scope_query": scope_query,
			"document_count": len(matched_docs),
			"status": "active",
			"released_at": None,
			"created_at": self._now(),
		}
		self.litigation_holds[hold["id"]] = hold
		self.hold_documents[hold["id"]] = set(matched_docs)
		for doc_id in matched_docs:
			doc = self.documents.get(doc_id)
			if doc:
				doc["on_hold"] = True
				doc["hold_ids"] = list(set(doc.get("hold_ids", []) + [hold["id"]]))
		self._emit(tenant, "litigation_hold_issued", hold["id"], {"matter_id": matter_id, "docs_held": len(matched_docs)})
		return deepcopy(hold)

	async def get_litigation_hold(self, tenant_id: str, hold_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		hold = self.litigation_holds.get(hold_id)
		if not hold or hold["tenant_id"] != tenant:
			raise KeyError(f"hold {hold_id} not found")
		return deepcopy(hold)

	async def list_litigation_holds(self, tenant_id: str, matter_id: str | None = None) -> list[dict[str, Any]]:
		"""List litigation holds."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(h) for h in self.litigation_holds.values() if h["tenant_id"] == tenant]
		if matter_id:
			items = [h for h in items if h["matter_id"] == matter_id]
		return items

	async def release_litigation_hold(self, tenant_id: str, hold_id: str, released_by: str) -> dict[str, Any]:
		"""Release a litigation hold."""
		tenant = self._tenant(tenant_id)
		hold = self.litigation_holds.get(hold_id)
		if not hold or hold["tenant_id"] != tenant:
			raise KeyError(f"hold {hold_id} not found")
		if hold["status"] != "active":
			raise ValueError("hold is not active")
		hold["status"] = "released"
		hold["released_by"] = released_by
		hold["released_at"] = self._now()
		# Remove hold from documents
		doc_ids = self.hold_documents.pop(hold_id, set())
		for doc_id in doc_ids:
			doc = self.documents.get(doc_id)
			if doc:
				doc["hold_ids"] = [h for h in doc.get("hold_ids", []) if h != hold_id]
				doc["on_hold"] = bool(doc["hold_ids"])
		self._emit(tenant, "litigation_hold_released", hold_id)
		return deepcopy(hold)

	async def delete_litigation_hold(self, tenant_id: str, hold_id: str) -> dict[str, Any]:
		"""Archive a released hold."""
		return await self.release_litigation_hold(tenant_id, hold_id, released_by="system")

	# ── Production Sets ──────────────────────────────────────────────────────

	async def create_production_set(
		self,
		tenant_id: str,
		matter_id: str,
		title: str,
		document_ids: list[str],
		production_format: str,
		requesting_party: str,
		prepared_by_id: str,
		bates_prefix: str = "",
	) -> dict[str, Any]:
		"""Create an eDiscovery production set."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		if production_format not in PRODUCTION_FORMATS:
			raise ValueError(f"production_format must be one of {PRODUCTION_FORMATS}")
		# Validate docs exist and none are privileged (unless explicitly included)
		for doc_id in document_ids:
			doc = self.documents.get(doc_id)
			if not doc or doc["tenant_id"] != tenant:
				raise KeyError(f"document {doc_id} not found")
			if doc.get("is_privileged"):
				raise ValueError(f"document {doc_id} is privileged and cannot be produced without review")
		bates_start = 1
		bates_end = bates_start + len(document_ids) - 1
		prod: dict[str, Any] = {
			"id": self._id("prd-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"title": title,
			"document_ids": list(document_ids),
			"production_format": production_format,
			"bates_prefix": bates_prefix,
			"bates_start": bates_start,
			"bates_end": bates_end,
			"requesting_party": requesting_party,
			"prepared_by_id": prepared_by_id,
			"status": "pending",
			"produced_at": None,
			"created_at": self._now(),
		}
		self.production_sets[prod["id"]] = prod
		self._emit(tenant, "production_set_created", prod["id"], {"doc_count": len(document_ids)})
		return deepcopy(prod)

	async def get_production_set(self, tenant_id: str, production_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		prod = self.production_sets.get(production_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"production set {production_id} not found")
		return deepcopy(prod)

	async def list_production_sets(self, tenant_id: str, matter_id: str | None = None) -> list[dict[str, Any]]:
		"""List production sets."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.production_sets.values() if p["tenant_id"] == tenant]
		if matter_id:
			items = [p for p in items if p["matter_id"] == matter_id]
		return items

	async def finalize_production(self, tenant_id: str, production_id: str, finalized_by: str) -> dict[str, Any]:
		"""Mark a production set as produced."""
		tenant = self._tenant(tenant_id)
		prod = self.production_sets.get(production_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"production set {production_id} not found")
		prod["status"] = "produced"
		prod["produced_at"] = self._now()
		prod["finalized_by"] = finalized_by
		self._emit(tenant, "production_finalized", production_id)
		return deepcopy(prod)

	async def delete_production_set(self, tenant_id: str, production_id: str) -> dict[str, Any]:
		"""Cancel a production set."""
		tenant = self._tenant(tenant_id)
		prod = self.production_sets.get(production_id)
		if not prod or prod["tenant_id"] != tenant:
			raise KeyError(f"production set {production_id} not found")
		if prod["status"] == "produced":
			raise ValueError("cannot cancel an already produced set")
		prod["status"] = "cancelled"
		self._emit(tenant, "production_cancelled", production_id)
		return deepcopy(prod)

	# ── Analytics ────────────────────────────────────────────────────────────

	async def repository_stats(self, tenant_id: str) -> dict[str, Any]:
		"""Return repository statistics."""
		tenant = self._tenant(tenant_id)
		docs = [d for d in self.documents.values() if d["tenant_id"] == tenant]
		by_type: dict[str, int] = {}
		total_bytes = 0
		for d in docs:
			by_type[d["document_type"]] = by_type.get(d["document_type"], 0) + 1
			total_bytes += d.get("file_size_bytes", 0)
		return {
			"tenant_id": tenant,
			"total_documents": len(docs),
			"total_size_bytes": total_bytes,
			"by_type": by_type,
			"privileged_count": sum(1 for d in docs if d.get("is_privileged")),
			"on_hold_count": sum(1 for d in docs if d.get("on_hold")),
			"active_holds": sum(1 for h in self.litigation_holds.values() if h["tenant_id"] == tenant and h["status"] == "active"),
			"generated_at": self._now(),
		}

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]
