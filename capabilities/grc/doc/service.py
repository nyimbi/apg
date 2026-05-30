"""Dependency-light GRC document lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

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
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

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
