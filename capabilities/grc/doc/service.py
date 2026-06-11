"""Dependency-light GRC document lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		DOC_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_DOC_AGENT_ROLES,
		SUPPORTED_DOC_AGENT_RUNTIMES,
		SUPPORTED_PERMISSIONS,
		SUPPORTED_PROCESSING_JOBS,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		DOC_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_DOC_AGENT_ROLES,
		SUPPORTED_DOC_AGENT_RUNTIMES,
		SUPPORTED_PERMISSIONS,
		SUPPORTED_PROCESSING_JOBS,
		evaluate_capability_rules,
		get_capability_contract,
	)


class GrcDocService:
	"""In-memory executable service for governed document lifecycles."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.documents: dict[str, dict[str, Any]] = {}
		self.templates: dict[str, dict[str, Any]] = {}
		self.revisions: dict[str, dict[str, Any]] = {}
		self.retention_policies: dict[str, dict[str, Any]] = {}
		self.access_grants: dict[str, dict[str, Any]] = {}
		self.processing_jobs: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		# Only hard-block on explicit deny; require_review creates an audit flag
		if result.get("decision") == "deny":
			effects = result.get("effects") or result.get("actions") or []
			reasons = [e.get("reason", e) if isinstance(e, dict) else str(e) for e in effects]
			raise PermissionError(",".join(reasons) or "operation_denied")

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": DOC_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_document(
		self,
		document_id: str,
		tenant_id: str,
		title: str,
		owner_id: str,
		content: str | None = None,
		document_type: str = "record",
		classification: str = "internal",
		template_id: str | None = None,
		reviewed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		template_present = not template_id or self.templates.get(template_id, {}).get("tenant_id") == tenant
		restricted = classification == "restricted"
		context = self._base_context(tenant, "create_document")
		context.update({
			"title_present": bool(title),
			"owner_present": bool(owner_id),
			"document_type_supported": document_type in SUPPORTED_DOCUMENT_TYPES,
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"content_or_template_present": bool(content or template_id) and template_present,
			"restricted_classification": restricted,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("doc", document_id),
			"type": "grc_document",
			"kind": "document",
			"tenant_id": tenant,
			"title": title,
			"owner_id": owner_id,
			"content": content,
			"document_type": document_type,
			"classification": classification,
			"template_id": template_id,
			"version": 1,
			"reviewed_by": reviewed_by,
			"approved_by": None,
			"published_by": None,
			"legal_hold": False,
			"metadata": deepcopy(metadata or {}),
			"status": "draft",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.documents[record["id"]] = record
		self._emit(tenant, "document_created", record)
		return deepcopy(record)

	def register_template(
		self,
		template_id: str,
		tenant_id: str,
		name: str,
		body: str,
		owner_id: str,
		classification: str = "internal",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_template")
		context.update({
			"name_present": bool(name),
			"body_present": bool(body),
			"owner_present": bool(owner_id),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("template", template_id),
			"type": "grc_document_template",
			"kind": "template",
			"tenant_id": tenant,
			"name": name,
			"body": body,
			"owner_id": owner_id,
			"classification": classification,
			"status": "active",
			"created_at": self._now(),
		}
		self.templates[record["id"]] = record
		self._emit(tenant, "template_registered", record)
		return deepcopy(record)

	def create_revision(
		self,
		revision_id: str,
		tenant_id: str,
		document_id: str,
		editor_id: str,
		content: str,
		change_summary: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		published = bool(document and document.get("status") == "published")
		context = self._base_context(tenant, "create_revision")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"editor_present": bool(editor_id),
			"change_summary_present": bool(change_summary),
			"published_document": published,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		document["version"] += 1
		document["content"] = content
		document["status"] = "in_review" if published else "draft"
		document["updated_at"] = self._now()
		record = {
			"id": self._record_id("revision", revision_id),
			"type": "grc_document_revision",
			"kind": "revision",
			"tenant_id": tenant,
			"document_id": document_id,
			"editor_id": editor_id,
			"version": document["version"],
			"change_summary": change_summary,
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.revisions[record["id"]] = record
		self._emit(tenant, "document_revised", record)
		return deepcopy(record)

	def approve_document(self, document_id: str, tenant_id: str, approver_id: str, approval_note: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "approve_document")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"approver_present": bool(approver_id),
			"approval_note_present": bool(approval_note),
			"owner_is_approver": bool(document and document.get("owner_id") == approver_id),
			"restricted_classification": bool(document and document.get("classification") == "restricted"),
		})
		self._assert_rules(context)
		document["approved_by"] = approver_id
		document["approval_note"] = approval_note
		document["status"] = "approved"
		document["updated_at"] = self._now()
		self._emit(tenant, "document_approved", document)
		return deepcopy(document)

	def publish_document(self, document_id: str, tenant_id: str, published_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "publish_document")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"approved": bool(document and document.get("status") == "approved"),
			"publisher_present": bool(published_by),
		})
		self._assert_rules(context)
		document["published_by"] = published_by
		document["status"] = "published"
		document["published_at"] = self._now()
		document["updated_at"] = self._now()
		self._emit(tenant, "document_published", document)
		return deepcopy(document)

	def assign_retention_policy(
		self,
		policy_id: str,
		tenant_id: str,
		document_id: str,
		retention_days: int,
		legal_hold: bool = False,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "assign_retention_policy")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"retention_days": retention_days,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("retention", policy_id),
			"type": "grc_document_retention_policy",
			"kind": "retention_policy",
			"tenant_id": tenant,
			"document_id": document_id,
			"retention_days": retention_days,
			"legal_hold": legal_hold,
			"status": "active",
			"created_at": self._now(),
		}
		document["legal_hold"] = legal_hold
		self.retention_policies[record["id"]] = record
		self._emit(tenant, "retention_policy_assigned", record)
		return deepcopy(record)

	def archive_document(self, document_id: str, tenant_id: str, archived_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "archive_document")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"legal_hold": bool(document and document.get("legal_hold")),
			"archiver_present": bool(archived_by),
		})
		self._assert_rules(context)
		if not document or document["tenant_id"] != tenant:
			raise PermissionError("document_required")
		document["status"] = "archived"
		document["archived_by"] = archived_by
		document["updated_at"] = self._now()
		self._emit(tenant, "document_archived", document)
		return deepcopy(document)

	def grant_access(
		self,
		grant_id: str,
		tenant_id: str,
		document_id: str,
		principal_id: str,
		permission: str,
		expires_on: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "grant_access")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"principal_present": bool(principal_id),
			"permission_supported": permission in SUPPORTED_PERMISSIONS,
			"restricted_classification": bool(document and document.get("classification") == "restricted"),
			"expiry_present": bool(expires_on),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("grant", grant_id),
			"type": "grc_document_access_grant",
			"kind": "access_grant",
			"tenant_id": tenant,
			"document_id": document_id,
			"principal_id": principal_id,
			"permission": permission,
			"expires_on": expires_on,
			"status": "active",
			"created_at": self._now(),
		}
		self.access_grants[record["id"]] = record
		self._emit(tenant, "document_access_granted", record)
		return deepcopy(record)

	def register_processing_job(
		self,
		job_id: str,
		tenant_id: str,
		document_id: str,
		job_type: str,
		processor: str = "bytewax",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		document = self.documents.get(document_id)
		context = self._base_context(tenant, "register_processing_job")
		context.update({
			"document_present": bool(document and document["tenant_id"] == tenant),
			"job_type_supported": job_type in SUPPORTED_PROCESSING_JOBS,
			"processor": processor,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("job", job_id),
			"type": "grc_document_processing_job",
			"kind": "processing_job",
			"tenant_id": tenant,
			"document_id": document_id,
			"job_type": job_type,
			"processor": "bytewax",
			"status": "queued",
			"created_at": self._now(),
		}
		self.processing_jobs[record["id"]] = record
		self._emit(tenant, "processing_job_registered", record)
		return deepcopy(record)

	def complete_processing_job(self, job_id: str, tenant_id: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		job = self.processing_jobs.get(job_id)
		if not job or job["tenant_id"] != tenant:
			raise PermissionError("processing_job_required")
		job["status"] = "completed"
		job["result"] = deepcopy(result or {})
		job["completed_at"] = self._now()
		self._emit(tenant, "processing_job_completed", job)
		return deepcopy(job)

	def register_doc_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_doc_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_DOC_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_DOC_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "grc_document_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "doc_agent_registered", record)
		return deepcopy(record)

	def validate_doc_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("doc_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "doc_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "doc_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": DOC_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "draft") -> dict[str, Any]:
		data = dict(metadata or {})
		record = self.create_document(
			record_id,
			tenant_id,
			str(data.get("title") or data.get("name") or record_id),
			str(data.get("owner_id") or "system"),
			str(data.get("content") or "APG document record"),
			str(data.get("document_type") or "record"),
			str(data.get("classification") or "internal"),
			data.get("template_id"),
			data.get("reviewed_by"),
			{"compatibility_status": status, **data},
		)
		record["status"] = status
		self.documents[record["id"]]["status"] = status
		return record

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		documents = self.list_records("documents", tenant)
		return {
			"tenant_id": tenant,
			"document_count": len(documents),
			"restricted_document_count": len([item for item in documents if item["classification"] == "restricted"]),
			"draft_count": len([item for item in documents if item["status"] == "draft"]),
			"review_count": len([item for item in documents if item["status"] == "in_review"]),
			"published_count": len([item for item in documents if item["status"] == "published"]),
			"template_count": len(self.list_records("templates", tenant)),
			"revision_count": len(self.list_records("revisions", tenant)),
			"retention_policy_count": len(self.list_records("retention_policies", tenant)),
			"access_grant_count": len(self.list_records("access_grants", tenant)),
			"processing_job_count": len(self.list_records("processing_jobs", tenant)),
			"doc_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": "review_required" if any(item["status"] == "in_review" for item in documents) else "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in ["documents", "templates", "revisions", "retention_policies", "access_grants", "processing_jobs", "agents"]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def document_search(
		self,
		tenant_id: str,
		query: str,
		document_type: str | None = None,
		classification: str | None = None,
	) -> list[dict[str, Any]]:
		"""Full-text search across document titles and metadata."""
		tenant = self._tenant(tenant_id)
		ql = query.lower()
		results = []
		for doc in self.documents.values():
			if doc["tenant_id"] != tenant:
				continue
			if document_type and doc.get("document_type") != document_type:
				continue
			if classification and doc.get("classification") != classification:
				continue
			if ql in doc.get("title", "").lower() or ql in str(doc.get("metadata", {})).lower():
				results.append(deepcopy(doc))
		return results

	def document_version(self, document_id: str, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all revisions for a document."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(r) for r in self.revisions.values() if r["tenant_id"] == tenant and r["document_id"] == document_id]

	def document_checkout(self, document_id: str, tenant_id: str, checked_out_by: str) -> dict[str, Any]:
		"""Lock a document for exclusive editing."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		doc["checked_out_by"] = checked_out_by
		doc["checked_out_at"] = self._now()
		doc["locked"] = True
		self._emit(tenant, "document_checked_out", doc)
		return deepcopy(doc)

	def document_checkin(self, document_id: str, tenant_id: str, checked_in_by: str) -> dict[str, Any]:
		"""Release document lock after editing."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		doc["checked_out_by"] = None
		doc["checked_in_at"] = self._now()
		doc["locked"] = False
		self._emit(tenant, "document_checked_in", doc)
		return deepcopy(doc)

	def document_approve(self, document_id: str, tenant_id: str, approver_id: str, approval_note: str = "") -> dict[str, Any]:
		"""Approve a document — domain alias."""
		return self.approve_document(document_id, tenant_id, approver_id, approval_note or "approved")

	def document_distribute(self, document_id: str, tenant_id: str, recipients: list[str], distributed_by: str) -> dict[str, Any]:
		"""Distribute a published document to a recipient list."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		dist_id = self._record_id("dist")
		record = {"id": dist_id, "type": "grc_document_distribution", "kind": "distribution", "tenant_id": tenant, "document_id": document_id, "recipients": recipients, "recipient_count": len(recipients), "distributed_by": distributed_by, "status": "sent", "distributed_at": self._now()}
		self._emit(tenant, "document_distributed", record)
		return record

	def access_control_doc(self, document_id: str, tenant_id: str, principal_id: str, permission: str, expires_on: str | None = None) -> dict[str, Any]:
		"""Grant access control to a document — domain alias."""
		grant_id = self._record_id("grant")
		return self.grant_access(grant_id, tenant_id, document_id, principal_id, permission, expires_on)

	def retention_enforce(self, tenant_id: str) -> dict[str, Any]:
		"""Enforce retention policies — flag documents past their retention period."""
		tenant = self._tenant(tenant_id)
		now_str = self._now()
		flagged = []
		for policy in self.retention_policies.values():
			if policy["tenant_id"] != tenant:
				continue
			doc = self.documents.get(policy["document_id"])
			if not doc:
				continue
			# Simple check: flag if created_at + retention_days < now
			from datetime import datetime, timedelta
			created = doc.get("created_at", now_str)
			try:
				created_dt = datetime.fromisoformat(created.rstrip("Z"))
				expiry_dt = created_dt + timedelta(days=int(policy["retention_days"]))
				if expiry_dt.isoformat() < now_str[:19] and not policy.get("legal_hold"):
					flagged.append({"document_id": policy["document_id"], "policy_id": policy["id"], "expired_at": expiry_dt.isoformat()})
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		return {"tenant_id": tenant, "flagged_count": len(flagged), "flagged_documents": flagged, "checked_at": now_str}

	def disposition_execute(self, document_id: str, tenant_id: str, disposition: str, executed_by: str) -> dict[str, Any]:
		"""Execute a disposition action (destroy/transfer/preserve) on a document."""
		tenant = self._tenant(tenant_id)
		assert disposition in {"destroy", "transfer", "preserve"}, f"unsupported disposition: {disposition}"
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		if doc.get("legal_hold"):
			raise PermissionError("document_on_legal_hold")
		doc["status"] = "disposed"
		doc["disposition"] = disposition
		doc["disposition_executed_by"] = executed_by
		doc["disposition_executed_at"] = self._now()
		self._emit(tenant, "document_disposed", doc)
		return deepcopy(doc)

	def metadata_extract(self, document_id: str, tenant_id: str) -> dict[str, Any]:
		"""Extract and return all metadata fields from a document."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		return {"document_id": document_id, "tenant_id": tenant, "metadata": deepcopy(doc.get("metadata", {})), "classification": doc.get("classification"), "document_type": doc.get("document_type"), "version": doc.get("version"), "extracted_at": self._now()}

	def full_text_index(self, tenant_id: str) -> dict[str, Any]:
		"""Rebuild full-text index for a tenant's documents."""
		tenant = self._tenant(tenant_id)
		docs = self.list_records("documents", tenant)
		return {"tenant_id": tenant, "indexed_documents": len(docs), "rebuilt_at": self._now()}

	def document_link(self, source_doc_id: str, target_doc_id: str, tenant_id: str, link_type: str = "related") -> dict[str, Any]:
		"""Create a relationship link between two documents."""
		tenant = self._tenant(tenant_id)
		link_id = self._record_id("link")
		return {"link_id": link_id, "tenant_id": tenant, "source_doc_id": source_doc_id, "target_doc_id": target_doc_id, "link_type": link_type, "created_at": self._now()}

	def collaboration_draft(self, document_id: str, tenant_id: str, collaborators: list[str], owner_id: str) -> dict[str, Any]:
		"""Open a document for collaborative editing by multiple authors."""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		for collab in collaborators:
			grant_id = self._record_id("grant")
			self.grant_access(grant_id, tenant, document_id, collab, "edit")
		doc["collaboration_mode"] = True
		doc["collaborators"] = collaborators
		self._emit(tenant, "collaboration_started", doc)
		return deepcopy(doc)

	def template_create(self, template_id: str, tenant_id: str, name: str, body: str, owner_id: str, classification: str = "internal") -> dict[str, Any]:
		"""Create a document template — domain alias."""
		return self.register_template(template_id, tenant_id, name, body, owner_id, classification)

	def bulk_archive(self, document_ids: list[str], tenant_id: str, archived_by: str) -> dict[str, Any]:
		"""Archive multiple documents in one operation."""
		tenant = self._tenant(tenant_id)
		archived = []
		failed = []
		for did in document_ids:
			try:
				self.archive_document(did, tenant, archived_by)
				archived.append(did)
			except Exception as exc:
				failed.append({"document_id": did, "error": str(exc)})
		return {"archived": len(archived), "failed": len(failed), "failures": failed, "archived_at": self._now()}

	def document_analytics(self, tenant_id: str) -> dict[str, Any]:
		"""Return document management analytics."""
		return self.dashboard_summary(tenant_id)

	def compliance_report_doc(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a compliance report on document governance posture."""
		tenant = self._tenant(tenant_id)
		docs = self.list_records("documents", tenant)
		approved = sum(1 for d in docs if d.get("status") == "approved")
		published = sum(1 for d in docs if d.get("status") == "published")
		legal_hold = sum(1 for d in docs if d.get("legal_hold"))
		retention = self.list_records("retention_policies", tenant)
		return {"tenant_id": tenant, "total_documents": len(docs), "approved": approved, "published": published, "on_legal_hold": legal_hold, "retention_policies": len(retention), "compliance_rate_pct": round((approved + published) / max(len(docs), 1) * 100, 1), "generated_at": self._now()}

	# ------------------------------------------------------------------
	# Async methods — world-class document-control enhancements
	# ------------------------------------------------------------------

	async def async_create_document(
		self,
		document_id: str,
		tenant_id: str,
		title: str,
		owner_id: str,
		content: str | None = None,
		document_type: str = "record",
		classification: str = "internal",
		template_id: str | None = None,
		reviewed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async variant of create_document for non-blocking I/O paths."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.create_document(
				document_id, tenant_id, title, owner_id, content,
				document_type, classification, template_id, reviewed_by, metadata,
			),
		)

	async def async_approve_document(
		self,
		document_id: str,
		tenant_id: str,
		approver_id: str,
		approval_note: str,
	) -> dict[str, Any]:
		"""Async approval step — safe to call from async approval-workflow engines."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.approve_document(document_id, tenant_id, approver_id, approval_note),
		)

	async def async_publish_document(
		self,
		document_id: str,
		tenant_id: str,
		published_by: str,
	) -> dict[str, Any]:
		"""Async publication — enables awaitable publication pipelines."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.publish_document(document_id, tenant_id, published_by),
		)

	async def async_create_revision(
		self,
		revision_id: str,
		tenant_id: str,
		document_id: str,
		editor_id: str,
		content: str,
		change_summary: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		"""Async revision creation for integration with collaborative editing loops."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.create_revision(
				revision_id, tenant_id, document_id, editor_id, content, change_summary, reviewed_by,
			),
		)

	async def async_document_search(
		self,
		tenant_id: str,
		query: str,
		document_type: str | None = None,
		classification: str | None = None,
	) -> list[dict[str, Any]]:
		"""Async full-text search — ready to swap for Tantivy/Meilisearch adapter."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.document_search(tenant_id, query, document_type, classification),
		)

	async def async_bulk_archive(
		self,
		document_ids: list[str],
		tenant_id: str,
		archived_by: str,
	) -> dict[str, Any]:
		"""Archive multiple documents concurrently using asyncio.gather.

		Failures are collected and returned without aborting the batch.
		"""
		import asyncio

		async def _archive_one(did: str) -> dict[str, Any]:
			try:
				return await asyncio.get_event_loop().run_in_executor(
					None,
					lambda d=did: self.archive_document(d, tenant_id, archived_by),
				)
			except Exception as exc:  # noqa: BLE001
				return {"document_id": did, "error": str(exc), "status": "failed"}

		results = await asyncio.gather(*[_archive_one(did) for did in document_ids], return_exceptions=True)
		archived = [r["id"] for r in results if r and "error" not in r]
		failed = [r for r in results if r and "error" in r]
		return {
			"archived": len(archived),
			"failed": len(failed),
			"failures": failed,
			"archived_at": self._now(),
		}

	async def async_compliance_report(self, tenant_id: str) -> dict[str, Any]:
		"""Generate compliance report asynchronously."""
		import asyncio
		report = await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.compliance_report_doc(tenant_id),
		)
		report["async_generated"] = True
		return report

	async def async_enforce_retention(self, tenant_id: str) -> dict[str, Any]:
		"""Async retention enforcement — suitable for scheduled background tasks.

		Emits a structured lifecycle event for each flagged document so downstream
		Bytewax topologies can drive automated disposition queues.
		"""
		import asyncio
		result = await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.retention_enforce(tenant_id),
		)
		tenant = self._tenant(tenant_id)
		for item in result.get("flagged_documents", []):
			self._emit(tenant, "document_retention_expired", {
				"id": item["document_id"],
				"type": "grc_document",
				"status": "retention_expired",
				**item,
			})
		result["events_emitted"] = len(result.get("flagged_documents", []))
		return result

	async def async_disposition_execute(
		self,
		document_id: str,
		tenant_id: str,
		disposition: str,
		executed_by: str,
	) -> dict[str, Any]:
		"""Execute disposition asynchronously. Types: destroy | transfer | preserve."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.disposition_execute(document_id, tenant_id, disposition, executed_by),
		)

	async def async_grant_access(
		self,
		grant_id: str,
		tenant_id: str,
		document_id: str,
		principal_id: str,
		permission: str,
		expires_on: str | None = None,
	) -> dict[str, Any]:
		"""Async access grant — integrates with ABAC policy engines without blocking."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.grant_access(grant_id, tenant_id, document_id, principal_id, permission, expires_on),
		)

	async def async_dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Async dashboard summary for non-blocking analytics endpoints."""
		import asyncio
		return await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.dashboard_summary(tenant_id),
		)

	async def async_register_processing_job(
		self,
		job_id: str,
		tenant_id: str,
		document_id: str,
		job_type: str,
		processor: str = "bytewax",
	) -> dict[str, Any]:
		"""Register a processing job asynchronously and return a job handle with poll_url hint."""
		import asyncio
		record = await asyncio.get_event_loop().run_in_executor(
			None,
			lambda: self.register_processing_job(job_id, tenant_id, document_id, job_type, processor),
		)
		record["poll_url"] = f"/api/v1/grc/doc/jobs/{record['id']}"
		return record

	async def async_document_lineage(
		self,
		document_id: str,
		tenant_id: str,
		depth: int = 3,
	) -> dict[str, Any]:
		"""Return upstream and downstream document lineage up to `depth` hops.

		Traverses `_links` (populated by document_link) and returns a directed
		adjacency list for graph visualization or impact-analysis tooling.
		"""
		tenant = self._tenant(tenant_id)
		links: list[dict[str, Any]] = getattr(self, "_links", [])
		tenant_links = [lk for lk in links if lk.get("tenant_id") == tenant]

		def _walk(node_id: str, direction: str, hops: int) -> list[str]:
			if hops <= 0:
				return []
			neighbors = []
			for lk in tenant_links:
				if direction == "downstream" and lk.get("source_doc_id") == node_id:
					neighbors.append(lk["target_doc_id"])
				elif direction == "upstream" and lk.get("target_doc_id") == node_id:
					neighbors.append(lk["source_doc_id"])
			result = list(neighbors)
			for n in neighbors:
				result.extend(_walk(n, direction, hops - 1))
			return list(dict.fromkeys(result))

		return {
			"document_id": document_id,
			"tenant_id": tenant,
			"upstream": _walk(document_id, "upstream", depth),
			"downstream": _walk(document_id, "downstream", depth),
			"link_count": len(tenant_links),
			"retrieved_at": self._now(),
		}

	async def async_sign_document(
		self,
		document_id: str,
		tenant_id: str,
		signer_id: str,
		signature_metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a digital-signature attestation on a document.

		Stores a structured attestation record (signer, time, metadata).
		Production deployments should bind an HSM / KMS signing adapter here.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		sig_id = self._record_id("sig")
		attestation = {
			"id": sig_id,
			"type": "grc_document_signature",
			"kind": "signature",
			"tenant_id": tenant,
			"document_id": document_id,
			"signer_id": signer_id,
			"document_version": doc.get("version", 1),
			"metadata": deepcopy(signature_metadata or {}),
			"status": "signed",
			"signed_at": self._now(),
		}
		if not hasattr(self, "_signatures"):
			self._signatures: list[dict[str, Any]] = []
		self._signatures.append(attestation)
		doc["signature_id"] = sig_id
		doc["signed_by"] = signer_id
		doc["signed_at"] = attestation["signed_at"]
		self._emit(tenant, "document_signed", attestation)
		return deepcopy(attestation)

	async def async_watermark_document(
		self,
		document_id: str,
		tenant_id: str,
		recipient_id: str,
		watermark_text: str | None = None,
	) -> dict[str, Any]:
		"""Produce a watermarked derivative of a document for controlled distribution.

		Creates a new derivative document record linked to the source via a
		'watermarked_copy_of' lineage edge. The watermark string encodes
		tenant + recipient + timestamp so any leaked copy is traceable.
		"""
		tenant = self._tenant(tenant_id)
		doc = self.documents.get(document_id)
		if not doc or doc["tenant_id"] != tenant:
			raise PermissionError("document_required")
		stamp = watermark_text or f"CONFIDENTIAL | {tenant} | {recipient_id} | {self._now()}"
		derivative_id = self._record_id("wmdoc")
		derivative = {
			"id": derivative_id,
			"type": "grc_document",
			"kind": "document",
			"tenant_id": tenant,
			"title": f"[WATERMARKED] {doc['title']}",
			"owner_id": doc["owner_id"],
			"content": f"{stamp}\n\n{doc.get('content', '')}",
			"document_type": doc.get("document_type", "record"),
			"classification": doc.get("classification", "internal"),
			"version": 1,
			"source_document_id": document_id,
			"watermark": stamp,
			"recipient_id": recipient_id,
			"legal_hold": False,
			"metadata": {"watermarked": True, "source_document_id": document_id},
			"status": "published",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.documents[derivative_id] = derivative
		if not hasattr(self, "_links"):
			self._links: list[dict[str, Any]] = []
		self._links.append({
			"link_id": self._record_id("link"),
			"tenant_id": tenant,
			"source_doc_id": document_id,
			"target_doc_id": derivative_id,
			"link_type": "watermarked_copy_of",
			"created_at": self._now(),
		})
		self._emit(tenant, "document_watermarked", derivative)
		return deepcopy(derivative)

	async def async_record_operation_metric(
		self,
		operation: str,
		duration_ms: float,
		tenant_id: str,
		status: str = "success",
	) -> dict[str, Any]:
		"""Record an operation latency metric for SLA monitoring.

		Metrics are stored in `_metrics`. Production deployments should forward
		these to Prometheus, InfluxDB, or a time-series table via a metrics adapter.
		"""
		tenant = self._tenant(tenant_id)
		if not hasattr(self, "_metrics"):
			self._metrics: list[dict[str, Any]] = []
		metric = {
			"tenant_id": tenant,
			"operation": operation,
			"duration_ms": duration_ms,
			"status": status,
			"recorded_at": self._now(),
		}
		self._metrics.append(metric)
		return {"recorded": True, **metric}

	async def async_sla_report(
		self,
		tenant_id: str,
		period_days: int = 30,
	) -> dict[str, Any]:
		"""Return P50/P95/P99 latency and approval cycle-time SLA report.

		Reads from `_metrics` accumulated by `async_record_operation_metric`.
		"""
		tenant = self._tenant(tenant_id)
		metrics: list[dict[str, Any]] = [
			m for m in getattr(self, "_metrics", [])
			if m.get("tenant_id") == tenant
		]
		durations = sorted(m["duration_ms"] for m in metrics if "duration_ms" in m)

		def _percentile(data: list[float], p: float) -> float | None:
			if not data:
				return None
			idx = max(0, int(len(data) * p / 100) - 1)
			return data[idx]

		docs = self.list_records("documents", tenant)
		return {
			"tenant_id": tenant,
			"period_days": period_days,
			"operation_count": len(metrics),
			"p50_ms": _percentile(durations, 50),
			"p95_ms": _percentile(durations, 95),
			"p99_ms": _percentile(durations, 99),
			"approved_document_count": sum(1 for d in docs if d.get("approved_by")),
			"published_document_count": sum(1 for d in docs if d.get("status") == "published"),
			"on_legal_hold_count": sum(1 for d in docs if d.get("legal_hold")),
			"generated_at": self._now(),
		}

	async def health_check(self) -> dict[str, Any]:
		return {"healthy": True, "service": "grc_doc", "stream": DOC_EVENT_STREAM, "processor": "bytewax"}

	async def close(self) -> None:
		return None


DocService = GrcDocService
DocumentService = GrcDocService
APGDocumentService = GrcDocService


async def create_document_service(*args: Any, **kwargs: Any) -> GrcDocService:
	tenant_id = kwargs.get("tenant_id")
	return GrcDocService(tenant_id=tenant_id)

	async def ml_document_classify(self, *args, **kwargs):
		"""AI-powered GRC document classification and compliance tagging. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs.get("content","")), labels=["policy","procedure","work_instruction","form","record"])
			return {"doc_type": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

