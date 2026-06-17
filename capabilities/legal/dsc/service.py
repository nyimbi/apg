"""Document & eDiscovery — async service layer.

Implements I2 (hold acknowledgement), I4 (integrity verification), I5 (review coding),
I6 (redaction engine), I7 (FRCP deadline calendar), I8 (privilege challenge tracker),
I9 (rolling Bates numbering), I10 (document families), I11 (retention policy),
I12 (cost tracking) from WORLD_CLASS_IMPROVEMENTS.md.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import hmac
import logging
import secrets
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_non_empty_string, guard_tenant_id

_log = logging.getLogger(__name__)

DOCUMENT_TYPES = {
	"pleading", "brief", "contract", "correspondence", "evidence",
	"internal", "court_order", "affidavit", "exhibit", "expert_report",
}
PRIVILEGE_TYPES = {"attorney_client", "work_product", "common_interest", "settlement", "deliberative"}
PRODUCTION_FORMATS = {"pdf", "tiff", "native", "searchable_pdf"}
REVIEW_CODINGS = {"responsive", "non_responsive", "needs_review", "redact", "withhold"}
COST_TYPES = {"processing", "hosting", "review", "production", "collection", "other"}
DEADLINE_TYPES = {
	"rule26_initial", "rule26_supplemental", "meet_and_confer",
	"production_response", "deposition_notice", "expert_disclosure",
}
EDRM_STAGES = [
	"identification", "preservation", "collection", "processing",
	"review", "analysis", "production", "presentation",
]


class DocumentEDiscoveryService:
	"""In-memory async service for document repository and eDiscovery.

	Covers: document ingestion, version control, attorney-client privilege logging,
	auto-litigation hold, eDiscovery production (rolling Bates), redaction engine,
	review coding, document families, retention/destruction policy, hold
	acknowledgement workflow, FRCP deadline calendar, privilege challenge tracker,
	matter cost tracker.
	"""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.documents: dict[str, dict[str, Any]] = {}
		self.privilege_log: dict[str, dict[str, Any]] = {}
		self.privilege_challenges: dict[str, dict[str, Any]] = {}
		self.litigation_holds: dict[str, dict[str, Any]] = {}
		self.hold_documents: dict[str, set[str]] = {}  # hold_id -> set of doc_ids
		self.hold_acknowledgements: dict[str, dict[str, Any]] = {}  # ack_id -> record
		self.production_sets: dict[str, dict[str, Any]] = {}
		self.access_log: list[dict[str, Any]] = []
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# Rolling Bates counter keyed by matter_id (I9)
		self._matter_bates: dict[str, int] = {}
		# Document version history keyed by doc_id (I11-adjacent)
		self._version_history: dict[str, list[dict[str, Any]]] = {}
		# Review codings keyed by coding_id (I5)
		self._review_codings = WriteThruDict('review_codings', tenant_id, _store)
		# Redaction log keyed by redaction_id (I6)
		self._redaction_log = WriteThruDict('redaction_log', tenant_id, _store)
		# Discovery deadlines keyed by deadline_id (I7)
		self._deadlines = WriteThruDict('deadlines', tenant_id, _store)
		# Cost entries keyed by cost_id (I12)
		self._cost_entries = WriteThruDict('cost_entries', tenant_id, _store)
		# Secure share tokens keyed by token (I15)
		self._share_tokens = WriteThruDict('share_tokens', tenant_id, _store)
		self._share_secret: str = secrets.token_hex(32)

	# ── Internals ────────────────────────────────────────────────────────────

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

	def _get_doc(self, tenant: str, document_id: str) -> dict[str, Any]:
		"""Fetch doc with tenant guard; raises KeyError on miss."""
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise KeyError(f"document {document_id} not found")
		return doc

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
			"version": "2.0.0",
			"document_types": sorted(DOCUMENT_TYPES),
			"privilege_types": sorted(PRIVILEGE_TYPES),
			"production_formats": sorted(PRODUCTION_FORMATS),
			"review_codings": sorted(REVIEW_CODINGS),
			"cost_types": sorted(COST_TYPES),
			"edrm_stages": EDRM_STAGES,
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
		content_sha256: str | None = None,
		parent_document_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Ingest a document into the repository with forensic integrity hash (I4)."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		guard_non_empty_string(file_reference, "file_reference")
		if document_type not in DOCUMENT_TYPES:
			raise ValueError(f"document_type must be one of {DOCUMENT_TYPES}")
		if is_privileged and privilege_type and privilege_type not in PRIVILEGE_TYPES:
			raise ValueError(f"privilege_type must be one of {PRIVILEGE_TYPES}")

		# I10: resolve family_id from parent
		family_id: str | None = None
		if parent_document_id:
			parent = self.documents.get(parent_document_id)
			if not parent or parent["tenant_id"] != tenant:
				raise KeyError(f"parent document {parent_document_id} not found")
			family_id = parent.get("family_id") or parent["id"]

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
			# I4 — forensic integrity hash stored at ingest
			"content_sha256": content_sha256,
			# I10 — document family linkage
			"parent_document_id": parent_document_id,
			"family_id": family_id,
			# I3 — near-duplicate cluster placeholder (populated by dedup scan)
			"near_dup_cluster_id": None,
			"content_hash": None,
			# I5 — review state
			"review_coding": None,
			"review_confidence": None,
			# I11 — retention
			"retention_policy_id": None,
			"destroy_after_date": None,
			# I13 — entity extraction placeholder
			"entities": [],
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
		doc = self._get_doc(tenant, document_id)
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
		doc = self._get_doc(tenant, document_id)
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
		doc = self._get_doc(tenant, document_id)
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

	# ── I4: Forensic Integrity Verification ──────────────────────────────────

	async def verify_integrity(self, tenant_id: str, document_id: str, current_sha256: str) -> dict[str, Any]:
		"""Verify document has not been tampered with since ingest (I4 — ISO 27037 / FRCP Rule 34).

		Business value: provides court-admissible evidence of document authenticity.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(current_sha256, "current_sha256")
		doc = self._get_doc(tenant, document_id)
		stored = doc.get("content_sha256")
		if stored is None:
			result = {"status": "no_hash_stored", "verified": False}
		else:
			verified = hmac.compare_digest(stored.lower(), current_sha256.lower())
			result = {
				"status": "verified" if verified else "tampered",
				"verified": verified,
				"stored_sha256": stored,
				"provided_sha256": current_sha256,
			}
		result.update({
			"document_id": document_id,
			"checked_at": self._now(),
		})
		self._emit(tenant, "integrity_verified", document_id, {"verified": result["verified"]})
		_log.info("integrity_verified tenant=%s doc=%s verified=%s", tenant, document_id, result["verified"])
		return result

	# ── I5: Review Coding with Near-Dup Propagation ──────────────────────────

	async def code_document(
		self,
		tenant_id: str,
		document_id: str,
		coding: str,
		reviewer_id: str,
		note: str = "",
	) -> dict[str, Any]:
		"""Record a review coding decision on a document (I5 — cuts review cost 40%).

		Business value: first-pass coding decisions are the most expensive review action;
		recording them enables propagation to near-duplicates via propagate_coding().
		"""
		tenant = self._tenant(tenant_id)
		if coding not in REVIEW_CODINGS:
			raise ValueError(f"coding must be one of {REVIEW_CODINGS}")
		doc = self._get_doc(tenant, document_id)
		doc["review_coding"] = coding
		doc["review_confidence"] = 1.0
		doc["updated_at"] = self._now()
		entry: dict[str, Any] = {
			"id": self._id("cod-"),
			"tenant_id": tenant,
			"document_id": document_id,
			"coding": coding,
			"reviewer_id": reviewer_id,
			"note": note,
			"confidence": 1.0,
			"propagated": False,
			"created_at": self._now(),
		}
		self._review_codings[entry["id"]] = entry
		self._emit(tenant, "document_coded", document_id, {"coding": coding, "reviewer_id": reviewer_id})
		_log.info("document coded tenant=%s doc=%s coding=%s", tenant, document_id, coding)
		return deepcopy(entry)

	async def propagate_coding(self, tenant_id: str, document_id: str) -> dict[str, Any]:
		"""Auto-code near-duplicate cluster with same coding at reduced confidence (I5).

		Business value: eliminates re-review of duplicate documents across the cluster.
		"""
		tenant = self._tenant(tenant_id)
		doc = self._get_doc(tenant, document_id)
		cluster_id = doc.get("near_dup_cluster_id")
		coding = doc.get("review_coding")
		if not coding:
			raise ValueError("document has no review coding to propagate")
		propagated_ids: list[str] = []
		if cluster_id:
			for other in self.documents.values():
				if (
					other["tenant_id"] == tenant
					and other.get("near_dup_cluster_id") == cluster_id
					and other["id"] != document_id
					and not other.get("review_coding")
				):
					other["review_coding"] = coding
					other["review_confidence"] = 0.85  # lower confidence for propagated calls
					other["updated_at"] = self._now()
					propagated_ids.append(other["id"])
					self._emit(tenant, "coding_propagated", other["id"], {
						"source_doc": document_id,
						"coding": coding,
					})
		_log.info("coding propagated tenant=%s source=%s cluster=%s count=%d", tenant, document_id, cluster_id, len(propagated_ids))
		return {"source_document_id": document_id, "coding": coding, "propagated_to": propagated_ids}

	# ── I6: Redaction Engine ─────────────────────────────────────────────────

	async def add_redaction(
		self,
		tenant_id: str,
		document_id: str,
		page: int,
		bbox: list[float],
		reason: str,
		redacted_by: str,
	) -> dict[str, Any]:
		"""Record a redaction on a specific document page region (I6 — GDPR/CCPA required).

		Business value: provides court-admissible redaction log, preventing waiver of
		privilege through improper disclosure.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(reason, "reason")
		guard_non_empty_string(redacted_by, "redacted_by")
		if page < 1:
			raise ValueError("page must be >= 1")
		if len(bbox) != 4:
			raise ValueError("bbox must be [x1, y1, x2, y2]")
		doc = self._get_doc(tenant, document_id)
		entry: dict[str, Any] = {
			"id": self._id("rdx-"),
			"tenant_id": tenant,
			"document_id": document_id,
			"page": page,
			"bbox": list(bbox),
			"reason": reason,
			"redacted_by": redacted_by,
			"created_at": self._now(),
		}
		self._redaction_log[entry["id"]] = entry
		# Flag document as having redactions
		doc.setdefault("redaction_count", 0)
		doc["redaction_count"] += 1
		doc["updated_at"] = self._now()
		self._emit(tenant, "redaction_added", document_id, {"page": page, "reason": reason})
		_log.info("redaction added tenant=%s doc=%s page=%d", tenant, document_id, page)
		return deepcopy(entry)

	async def list_redactions(self, tenant_id: str, document_id: str) -> list[dict[str, Any]]:
		"""Return all redaction records for a document (I6 — privilege log export)."""
		tenant = self._tenant(tenant_id)
		self._get_doc(tenant, document_id)  # ownership guard
		return [
			deepcopy(r) for r in self._redaction_log.values()
			if r["tenant_id"] == tenant and r["document_id"] == document_id
		]

	# ── I7: FRCP Discovery Deadline Calendar ─────────────────────────────────

	async def create_discovery_deadline(
		self,
		tenant_id: str,
		matter_id: str,
		deadline_type: str,
		due_date: str,
		description: str,
		assigned_to_id: str,
	) -> dict[str, Any]:
		"""Create a tracked FRCP/CPR discovery deadline (I7 — prevents sanctions).

		Business value: auto-surfacing overdue deadlines eliminates the single most
		avoidable source of court sanctions.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(matter_id, "matter_id")
		guard_non_empty_string(description, "description")
		if deadline_type not in DEADLINE_TYPES:
			raise ValueError(f"deadline_type must be one of {DEADLINE_TYPES}")
		entry: dict[str, Any] = {
			"id": self._id("ddl-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"deadline_type": deadline_type,
			"due_date": due_date,
			"description": description,
			"assigned_to_id": assigned_to_id,
			"status": "pending",
			"completed_at": None,
			"created_at": self._now(),
		}
		self._deadlines[entry["id"]] = entry
		self._emit(tenant, "discovery_deadline_created", entry["id"], {"due_date": due_date, "type": deadline_type})
		_log.info("deadline created tenant=%s matter=%s type=%s due=%s", tenant, matter_id, deadline_type, due_date)
		return deepcopy(entry)

	async def list_overdue_deadlines(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all overdue discovery deadlines with days_overdue computed (I7).

		Business value: daily review of this list is a firm's primary sanctions-prevention mechanism.
		"""
		tenant = self._tenant(tenant_id)
		now = datetime.utcnow()
		overdue: list[dict[str, Any]] = []
		for ddl in self._deadlines.values():
			if ddl["tenant_id"] != tenant or ddl["status"] == "completed":
				continue
			try:
				due_dt = datetime.fromisoformat(ddl["due_date"].rstrip("Z"))
			except ValueError:
				continue
			if due_dt < now:
				item = deepcopy(ddl)
				item["days_overdue"] = (now - due_dt).days
				overdue.append(item)
		overdue.sort(key=lambda x: x["days_overdue"], reverse=True)
		return overdue

	async def complete_deadline(self, tenant_id: str, deadline_id: str, completed_by: str) -> dict[str, Any]:
		"""Mark a discovery deadline as completed."""
		tenant = self._tenant(tenant_id)
		ddl = self._deadlines.get(deadline_id)
		if not ddl or ddl["tenant_id"] != tenant:
			raise KeyError(f"deadline {deadline_id} not found")
		ddl["status"] = "completed"
		ddl["completed_at"] = self._now()
		ddl["completed_by"] = completed_by
		self._emit(tenant, "deadline_completed", deadline_id)
		return deepcopy(ddl)

	# ── I8: Privilege Challenge Tracker ──────────────────────────────────────

	async def raise_privilege_challenge(
		self,
		tenant_id: str,
		privilege_id: str,
		challenger_id: str,
		basis: str,
	) -> dict[str, Any]:
		"""Open a formal privilege challenge against a privilege log entry (I8 — waiver risk control).

		Business value: structured challenge workflow prevents waiver through
		inconsistent or undocumented responses to opposing counsel.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(basis, "basis")
		priv = self.privilege_log.get(privilege_id)
		if not priv or priv["tenant_id"] != tenant:
			raise KeyError(f"privilege log entry {privilege_id} not found")
		challenge: dict[str, Any] = {
			"id": self._id("chg-"),
			"tenant_id": tenant,
			"privilege_id": privilege_id,
			"document_id": priv["document_id"],
			"challenger_id": challenger_id,
			"basis": basis,
			"status": "pending",
			"response_text": None,
			"supporting_doc_ids": [],
			"ruled_at": None,
			"ruling": None,
			"created_at": self._now(),
		}
		self.privilege_challenges[challenge["id"]] = challenge
		self._emit(tenant, "privilege_challenge_raised", challenge["id"], {
			"privilege_id": privilege_id,
			"challenger_id": challenger_id,
		})
		_log.info("privilege challenge raised tenant=%s privilege=%s challenger=%s", tenant, privilege_id, challenger_id)
		return deepcopy(challenge)

	async def respond_to_challenge(
		self,
		tenant_id: str,
		challenge_id: str,
		response_text: str,
		supporting_doc_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Provide a formal response to a privilege challenge (I8).

		Business value: documented response is evidence against waiver in in-camera review.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(response_text, "response_text")
		chg = self.privilege_challenges.get(challenge_id)
		if not chg or chg["tenant_id"] != tenant:
			raise KeyError(f"challenge {challenge_id} not found")
		if chg["status"] != "pending":
			raise ValueError(f"challenge is in state '{chg['status']}', not pending")
		chg["response_text"] = response_text
		chg["supporting_doc_ids"] = list(supporting_doc_ids or [])
		chg["status"] = "responded"
		chg["responded_at"] = self._now()
		self._emit(tenant, "privilege_challenge_responded", challenge_id)
		return deepcopy(chg)

	async def rule_on_challenge(
		self,
		tenant_id: str,
		challenge_id: str,
		ruling: str,
		ruled_by: str,
	) -> dict[str, Any]:
		"""Record a court ruling on a privilege challenge (I8)."""
		tenant = self._tenant(tenant_id)
		if ruling not in ("upheld", "overruled"):
			raise ValueError("ruling must be 'upheld' or 'overruled'")
		chg = self.privilege_challenges.get(challenge_id)
		if not chg or chg["tenant_id"] != tenant:
			raise KeyError(f"challenge {challenge_id} not found")
		chg["ruling"] = ruling
		chg["ruled_by"] = ruled_by
		chg["ruled_at"] = self._now()
		chg["status"] = "ruled"
		if ruling == "overruled":
			# Privilege assertion is no longer valid — update document and log entry
			priv = self.privilege_log.get(chg["privilege_id"])
			if priv:
				priv["status"] = "overruled"
			doc = self.documents.get(chg["document_id"])
			if doc:
				doc["is_privileged"] = False
		self._emit(tenant, "privilege_challenge_ruled", challenge_id, {"ruling": ruling})
		return deepcopy(chg)

	# ── I9: Rolling Bates Numbering ───────────────────────────────────────────

	async def get_bates_range(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Return current high-water Bates number for a matter (I9 — prevents duplicate Bates).

		Business value: attorneys often run multiple productions per matter;
		this prevents duplicate or gap Bates numbers that trigger court sanctions.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(matter_id, "matter_id")
		counter = self._matter_bates.get(matter_id, 0)
		return {
			"tenant_id": tenant,
			"matter_id": matter_id,
			"current_high_water": counter,
			"next_start": counter + 1,
		}

	# ── I10: Document Families ────────────────────────────────────────────────

	async def attach_document(
		self,
		tenant_id: str,
		child_doc_id: str,
		parent_doc_id: str,
	) -> dict[str, Any]:
		"""Link a document as a child (attachment) of a parent (I10 — FRCP Rule 34(b)(2)(E)).

		Business value: ensures email attachments travel with parent emails in every
		production, preventing deficient productions.
		"""
		tenant = self._tenant(tenant_id)
		child = self._get_doc(tenant, child_doc_id)
		parent = self._get_doc(tenant, parent_doc_id)
		# Use parent's family_id or parent's own id as the family root
		family_id = parent.get("family_id") or parent["id"]
		parent["family_id"] = family_id
		child["parent_document_id"] = parent_doc_id
		child["family_id"] = family_id
		child["updated_at"] = self._now()
		self._emit(tenant, "document_attached", child_doc_id, {"parent_id": parent_doc_id, "family_id": family_id})
		_log.info("document attached tenant=%s child=%s parent=%s", tenant, child_doc_id, parent_doc_id)
		return {"child_doc_id": child_doc_id, "parent_doc_id": parent_doc_id, "family_id": family_id}

	async def get_document_family(self, tenant_id: str, document_id: str) -> list[dict[str, Any]]:
		"""Return all family members for a document in parent-first order (I10).

		Business value: provides complete family unit for production, satisfying
		FRCP family-unit production requirements.
		"""
		tenant = self._tenant(tenant_id)
		doc = self._get_doc(tenant, document_id)
		family_id = doc.get("family_id")
		if not family_id:
			return [deepcopy(doc)]
		members = [
			deepcopy(d) for d in self.documents.values()
			if d["tenant_id"] == tenant and d.get("family_id") == family_id
		]
		# parent first (no parent_document_id), then children
		members.sort(key=lambda d: (d.get("parent_document_id") is not None, d["created_at"]))
		return members

	# ── I11: Data Retention & Destruction Policy ──────────────────────────────

	async def set_retention_policy(
		self,
		tenant_id: str,
		document_id: str,
		policy_id: str,
		destroy_after_date: str,
	) -> dict[str, Any]:
		"""Attach a retention/destruction policy to a document (I11 — GDPR Art. 17 / CCPA).

		Business value: provable destruction at end-of-retention satisfies regulatory
		requirements; litigation hold override prevents accidental spoliation.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(policy_id, "policy_id")
		guard_non_empty_string(destroy_after_date, "destroy_after_date")
		doc = self._get_doc(tenant, document_id)
		doc["retention_policy_id"] = policy_id
		doc["destroy_after_date"] = destroy_after_date
		doc["updated_at"] = self._now()
		self._emit(tenant, "retention_policy_set", document_id, {
			"policy_id": policy_id,
			"destroy_after_date": destroy_after_date,
		})
		_log.info("retention policy set tenant=%s doc=%s policy=%s", tenant, document_id, policy_id)
		return deepcopy(doc)

	async def list_destruction_eligible(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return documents past their destroy_after_date and NOT on hold (I11).

		Business value: daily review enables compliant destruction schedules; hold
		check prevents accidental spoliation of litigation-relevant documents.
		"""
		tenant = self._tenant(tenant_id)
		now = datetime.utcnow()
		eligible: list[dict[str, Any]] = []
		for doc in self.documents.values():
			if doc["tenant_id"] != tenant or doc["status"] != "active":
				continue
			if doc.get("on_hold"):
				continue
			destroy_after = doc.get("destroy_after_date")
			if not destroy_after:
				continue
			try:
				destroy_dt = datetime.fromisoformat(destroy_after.rstrip("Z"))
			except ValueError:
				continue
			if destroy_dt < now:
				item = deepcopy(doc)
				item["days_past_retention"] = (now - destroy_dt).days
				eligible.append(item)
		eligible.sort(key=lambda d: d["days_past_retention"], reverse=True)
		return eligible

	# ── I12: Matter-Level Cost Tracking ──────────────────────────────────────

	async def record_cost(
		self,
		tenant_id: str,
		matter_id: str,
		cost_type: str,
		amount: Decimal,
		vendor: str,
		description: str,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""Record an eDiscovery cost entry against a matter (I12 — no float leakage).

		Business value: real-time cost visibility against matter budget is the top
		factor in retaining litigation mandates for repeat clients.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(matter_id, "matter_id")
		guard_non_empty_string(vendor, "vendor")
		if cost_type not in COST_TYPES:
			raise ValueError(f"cost_type must be one of {COST_TYPES}")
		if not isinstance(amount, Decimal):
			raise TypeError("amount must be Decimal, not float")
		if amount < Decimal("0"):
			raise ValueError("amount must be non-negative")
		entry: dict[str, Any] = {
			"id": self._id("cst-"),
			"tenant_id": tenant,
			"matter_id": matter_id,
			"cost_type": cost_type,
			"amount": str(amount),          # serialise Decimal as string — never float
			"currency": currency.upper(),
			"vendor": vendor,
			"description": description,
			"created_at": self._now(),
		}
		self._cost_entries[entry["id"]] = entry
		self._emit(tenant, "cost_recorded", entry["id"], {"matter_id": matter_id, "amount": str(amount)})
		_log.info("cost recorded tenant=%s matter=%s type=%s amount=%s", tenant, matter_id, cost_type, amount)
		return deepcopy(entry)

	async def matter_cost_summary(self, tenant_id: str, matter_id: str) -> dict[str, Any]:
		"""Return matter eDiscovery cost totals by type, all as Decimal strings (I12).

		Business value: partners can compare estimated vs actual at any point,
		enabling proactive budget conversations before client shock invoices.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(matter_id, "matter_id")
		entries = [
			e for e in self._cost_entries.values()
			if e["tenant_id"] == tenant and e["matter_id"] == matter_id
		]
		totals: dict[str, Decimal] = {}
		for e in entries:
			ct = e["cost_type"]
			totals[ct] = totals.get(ct, Decimal("0")) + Decimal(e["amount"])
		grand_total = sum(totals.values(), Decimal("0"))
		return {
			"tenant_id": tenant,
			"matter_id": matter_id,
			"by_type": {k: str(v) for k, v in totals.items()},
			"grand_total": str(grand_total),
			"currency": entries[0]["currency"] if entries else "USD",
			"entry_count": len(entries),
			"generated_at": self._now(),
		}

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
		doc = self._get_doc(tenant, document_id)
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
			"custodian_ids": list(custodian_ids),
			"issued_by_id": issued_by_id,
			"scope_query": scope_query,
			"document_count": len(matched_docs),
			"status": "active",
			"released_at": None,
			"acknowledgement_required": True,
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

	# ── I2: Hold Acknowledgement Workflow ────────────────────────────────────

	async def request_hold_acknowledgement(
		self,
		tenant_id: str,
		hold_id: str,
		custodian_id: str,
		due_in_days: int = 5,
	) -> dict[str, Any]:
		"""Send a formal acknowledgement request to a custodian (I2 — prevents spoliation sanctions).

		Business value: documented acknowledgement is the primary defence against
		court-imposed sanctions for improper litigation holds.
		"""
		tenant = self._tenant(tenant_id)
		hold = self.litigation_holds.get(hold_id)
		if not hold or hold["tenant_id"] != tenant:
			raise KeyError(f"hold {hold_id} not found")
		if custodian_id not in hold.get("custodian_ids", []):
			raise ValueError(f"custodian {custodian_id} is not on hold {hold_id}")
		due_date = (datetime.utcnow() + timedelta(days=due_in_days)).isoformat(timespec="seconds") + "Z"
		ack: dict[str, Any] = {
			"id": self._id("ack-"),
			"tenant_id": tenant,
			"hold_id": hold_id,
			"custodian_id": custodian_id,
			"status": "pending",
			"requested_at": self._now(),
			"due_date": due_date,
			"acknowledged_at": None,
			"signature_reference": None,
			"escalated": False,
		}
		self.hold_acknowledgements[ack["id"]] = ack
		self._emit(tenant, "hold_acknowledgement_requested", ack["id"], {
			"hold_id": hold_id, "custodian_id": custodian_id,
		})
		_log.info("hold ack requested tenant=%s hold=%s custodian=%s", tenant, hold_id, custodian_id)
		return deepcopy(ack)

	async def record_acknowledgement(
		self,
		tenant_id: str,
		hold_id: str,
		custodian_id: str,
		signature_reference: str,
	) -> dict[str, Any]:
		"""Record that a custodian has acknowledged a litigation hold (I2).

		Business value: creates the court-admissible record of receipt that defeats
		spoliation motions based on alleged lack of notice.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(signature_reference, "signature_reference")
		# Find the pending ack record
		ack = next(
			(a for a in self.hold_acknowledgements.values()
			 if a["tenant_id"] == tenant and a["hold_id"] == hold_id and a["custodian_id"] == custodian_id),
			None,
		)
		if ack is None:
			raise KeyError(f"no pending acknowledgement for hold={hold_id} custodian={custodian_id}")
		ack["status"] = "acknowledged"
		ack["acknowledged_at"] = self._now()
		ack["signature_reference"] = signature_reference
		self._emit(tenant, "hold_acknowledged", ack["id"], {
			"hold_id": hold_id, "custodian_id": custodian_id,
		})
		_log.info("hold acknowledged tenant=%s hold=%s custodian=%s", tenant, hold_id, custodian_id)
		return deepcopy(ack)

	async def list_unacknowledged_holds(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all overdue acknowledgement requests (I2 — escalation input).

		Business value: daily triage prevents a custodian's silence from becoming
		evidence of spoliation.
		"""
		tenant = self._tenant(tenant_id)
		now = datetime.utcnow()
		overdue: list[dict[str, Any]] = []
		for ack in self.hold_acknowledgements.values():
			if ack["tenant_id"] != tenant or ack["status"] != "pending":
				continue
			try:
				due_dt = datetime.fromisoformat(ack["due_date"].rstrip("Z"))
			except ValueError:
				continue
			if due_dt < now:
				item = deepcopy(ack)
				item["days_overdue"] = (now - due_dt).days
				overdue.append(item)
		return overdue

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
		"""Create an eDiscovery production set with rolling Bates numbering (I9).

		Business value: incremental counter guarantees no Bates gaps or duplicates
		across multiple productions in a single matter.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(title, "title")
		if production_format not in PRODUCTION_FORMATS:
			raise ValueError(f"production_format must be one of {PRODUCTION_FORMATS}")
		# Validate docs exist and none are privileged (unless explicitly reviewed)
		for doc_id in document_ids:
			doc = self.documents.get(doc_id)
			if not doc or doc["tenant_id"] != tenant:
				raise KeyError(f"document {doc_id} not found")
			if doc.get("is_privileged"):
				raise ValueError(f"document {doc_id} is privileged and cannot be produced without review")
		# I9 — rolling Bates: start from prior high-water mark
		bates_start = self._matter_bates.get(matter_id, 0) + 1
		bates_end = bates_start + len(document_ids) - 1
		self._matter_bates[matter_id] = bates_end  # persist high-water mark
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
			"destruction_eligible": sum(1 for d in docs if d.get("destroy_after_date") and not d.get("on_hold")),
			"generated_at": self._now(),
		}

	async def get_audit_events(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_review_codings', '_redaction_log', '_deadlines', '_cost_entries', '_share_tokens', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

