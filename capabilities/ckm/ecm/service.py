"""Async service layer for APG ECM / Records Management.

All public methods are async to integrate cleanly with FastAPI / ASGI hosts.
In-memory stores (dicts keyed by (tenant_id, id)) are used so the service can
run as a pure-Python capability without a database dependency — swap in a
SQLAlchemy or motor backend by subclassing and overriding the _store_* helpers.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_DISPOSAL_METHODS,
		SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_REGULATORY_FRAMEWORKS,
		SUPPORTED_RETENTION_CATEGORIES,
		SUPPORTED_RETENTION_TRIGGERS,
		SUPPORTED_SENSITIVITY_LEVELS,
		SUPPORTED_WORKFLOW_DECISIONS,
		SUPPORTED_WORKFLOW_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		EcDisposalRecord,
		EcDocument,
		EcDocumentVersion,
		EcRecordClassification,
		EcRetentionPolicy,
		EcWorkflowInstance,
		EcWorkflowStep,
		uuid7str,
	)
except ImportError:  # pragma: no cover — standalone execution
	from capability_contract import (  # type: ignore[no-redef]
		SUPPORTED_DISPOSAL_METHODS,
		SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_REGULATORY_FRAMEWORKS,
		SUPPORTED_RETENTION_CATEGORIES,
		SUPPORTED_RETENTION_TRIGGERS,
		SUPPORTED_SENSITIVITY_LEVELS,
		SUPPORTED_WORKFLOW_DECISIONS,
		SUPPORTED_WORKFLOW_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore[no-redef]
		EcDisposalRecord,
		EcDocument,
		EcDocumentVersion,
		EcRecordClassification,
		EcRetentionPolicy,
		EcWorkflowInstance,
		EcWorkflowStep,
		uuid7str,
	)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
	return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
	if not ts:
		return None
	try:
		dt = datetime.fromisoformat(ts)
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt
	except (ValueError, TypeError):
		return None


def _normalize(value: str) -> str:
	"""Lowercase, strip, replace spaces/hyphens with underscores."""
	return re.sub(r"[\s\-]+", "_", value.strip().lower())


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


def _retention_due_date(policy: EcRetentionPolicy, doc: EcDocument) -> str:
	"""Compute the ISO-8601 disposal due date from a retention policy and document."""
	anchor: datetime | None = None
	if policy.trigger == "creation":
		anchor = _parse_iso(doc.created_at)
	elif policy.trigger in ("last_access", "last_modified"):
		anchor = _parse_iso(doc.updated_at)
	# 'event' trigger requires external signal — fall back to created_at
	if anchor is None:
		anchor = _now_dt()
	due = anchor + timedelta(days=policy.retention_years * 365)
	return due.isoformat()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class EcmService:
	"""Tenant-scoped ECM / Records Management service."""

	def __init__(self) -> None:
		# Primary stores: (tenant_id, id) -> model
		self._documents: dict[tuple[str, str], EcDocument] = {}
		self._versions: dict[tuple[str, str], EcDocumentVersion] = {}
		self._retention_policies: dict[tuple[str, str], EcRetentionPolicy] = {}
		self._classifications: dict[tuple[str, str], EcRecordClassification] = {}
		self._workflows: dict[tuple[str, str], EcWorkflowInstance] = {}
		self._disposals: dict[tuple[str, str], EcDisposalRecord] = {}
		# Secondary index: document_id -> list[version_id]  (per tenant)
		self._doc_version_index: dict[tuple[str, str], list[str]] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------ #
	# Describe / evaluate                                                   #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Document lifecycle                                                    #
	# ------------------------------------------------------------------ #

	async def create_document(
		self,
		title: str,
		document_type: str,
		content_hash: str,
		retention_category: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
	) -> EcDocument:
		"""Create a new managed document and open its version history."""
		doc_type = _normalize(document_type)
		ret_cat = _normalize(retention_category)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_document",
			"title_present": _present(title),
			"document_type_supported": doc_type in SUPPORTED_DOCUMENT_TYPES,
			"content_hash_present": _present(content_hash),
			"retention_category_present": _present(retention_category),
			"retention_category_supported": ret_cat in SUPPORTED_RETENTION_CATEGORIES,
		})
		doc = EcDocument(
			id=uuid7str(),
			tenant_id=tenant_id,
			title=title.strip(),
			document_type=doc_type,
			content_hash=content_hash.strip(),
			version=1,
			status="draft",
			retention_category=ret_cat,
			metadata=metadata or {},
		)
		self._documents[(tenant_id, doc.id)] = doc
		# seed version history with v1
		v1 = EcDocumentVersion(
			id=uuid7str(),
			tenant_id=tenant_id,
			document_id=doc.id,
			version_number=1,
			author="system",
			change_summary="Initial version",
			content_hash=content_hash.strip(),
		)
		self._versions[(tenant_id, v1.id)] = v1
		self._doc_version_index.setdefault((tenant_id, doc.id), []).append(v1.id)
		self._audit(tenant_id, "document.created", doc.id)
		return doc

	async def add_version(
		self,
		document_id: str,
		content_hash: str,
		change_summary: str,
		author: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
	) -> EcDocumentVersion:
		"""Append an immutable version to an existing document."""
		doc = self._doc_or_raise(document_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "add_version",
			"document_present": True,
			"author_present": _present(author),
			"change_summary_present": _present(change_summary),
			"content_hash_present": _present(content_hash),
		})
		new_version_number = doc.version + 1
		ver = EcDocumentVersion(
			id=uuid7str(),
			tenant_id=tenant_id,
			document_id=document_id,
			version_number=new_version_number,
			author=author.strip(),
			change_summary=change_summary.strip(),
			content_hash=content_hash.strip(),
			metadata=metadata or {},
		)
		self._versions[(tenant_id, ver.id)] = ver
		self._doc_version_index.setdefault((tenant_id, document_id), []).append(ver.id)
		# bump document envelope
		doc.version = new_version_number
		doc.content_hash = content_hash.strip()
		doc.updated_at = _now()
		self._audit(tenant_id, "document.versioned", document_id)
		return ver

	async def get_document(
		self,
		document_id: str,
		version: int | None = None,
		tenant_id: str = "default",
	) -> EcDocument:
		"""Retrieve a document, optionally pinned to a specific version.

		When version is specified the returned EcDocument has content_hash
		patched to match that version's hash (all other fields reflect current
		document state).
		"""
		doc = self._doc_or_raise(document_id, tenant_id)
		if version is not None:
			# locate the version record matching the requested number
			ver = self._find_version_by_number(document_id, version, tenant_id)
			if ver is None:
				raise KeyError(f"version {version} not found for document {document_id!r}")
			# return a copy with the historical content_hash
			patched = doc.model_copy(update={"version": ver.version_number, "content_hash": ver.content_hash})
			return patched
		return doc

	async def classify_document(
		self,
		document_id: str,
		category: str,
		sensitivity: str,
		regulatory_framework: str,
		tenant_id: str,
		classified_by: str = "",
		notes: str = "",
	) -> EcRecordClassification:
		"""Assign a sensitivity + regulatory classification to a document."""
		doc = self._doc_or_raise(document_id, tenant_id)
		sens = _normalize(sensitivity)
		framework = _normalize(regulatory_framework)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "classify_document",
			"sensitivity_supported": sens in SUPPORTED_SENSITIVITY_LEVELS,
			"regulatory_framework_supported": framework in SUPPORTED_REGULATORY_FRAMEWORKS,
		})
		clf = EcRecordClassification(
			id=uuid7str(),
			tenant_id=tenant_id,
			document_id=document_id,
			category=category.strip(),
			sensitivity=sens,
			regulatory_framework=framework,
			classified_by=classified_by,
			notes=notes,
		)
		self._classifications[(tenant_id, clf.id)] = clf
		# update document envelope
		doc.sensitivity = sens
		doc.regulatory_framework = framework
		doc.updated_at = _now()
		self._audit(tenant_id, "document.classified", document_id)
		return clf

	# ------------------------------------------------------------------ #
	# Retention management                                                  #
	# ------------------------------------------------------------------ #

	async def create_retention_policy(
		self,
		category: str,
		years: int,
		trigger: str,
		disposal_method: str,
		tenant_id: str,
		description: str = "",
		regulatory_basis: str = "",
	) -> EcRetentionPolicy:
		"""Define a new retention rule for a document category."""
		cat = _normalize(category)
		trg = _normalize(trigger)
		method = _normalize(disposal_method)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_retention_policy",
			"retention_category_supported": cat in SUPPORTED_RETENTION_CATEGORIES,
			"trigger_supported": trg in SUPPORTED_RETENTION_TRIGGERS,
			"disposal_method_supported": method in SUPPORTED_DISPOSAL_METHODS,
			"years_positive": years > 0,
		})
		policy = EcRetentionPolicy(
			id=uuid7str(),
			tenant_id=tenant_id,
			category=cat,
			retention_years=years,
			trigger=trg,
			disposal_method=method,
			description=description,
			regulatory_basis=regulatory_basis,
		)
		self._retention_policies[(tenant_id, policy.id)] = policy
		self._audit(tenant_id, "retention_policy.created", policy.id)
		return policy

	async def apply_retention_policy(
		self,
		policy_id: str,
		document_ids: list[str],
		tenant_id: str,
	) -> list[EcDocument]:
		"""Attach a retention policy to a batch of documents and compute disposal dates."""
		policy = self._policy_or_raise(policy_id, tenant_id)
		updated: list[EcDocument] = []
		for doc_id in document_ids:
			doc = self._doc_or_raise(doc_id, tenant_id)
			assert doc.retention_category == policy.category, (
				f"document {doc_id!r} has category {doc.retention_category!r}; "
				f"policy targets {policy.category!r}"
			)
			doc.retention_policy_id = policy_id
			doc.disposal_due_date = _retention_due_date(policy, doc)
			doc.updated_at = _now()
			updated.append(doc)
		self._audit(tenant_id, "retention_policy.applied", policy_id)
		return updated

	async def find_due_for_disposal(
		self,
		tenant_id: str,
		as_of_date: str | None = None,
	) -> list[EcDocument]:
		"""Return documents whose disposal_due_date has passed and status is 'active' or 'archived'."""
		cutoff = _parse_iso(as_of_date) if as_of_date else _now_dt()
		eligible_statuses = {"active", "archived"}
		results: list[EcDocument] = []
		for (tid, _), doc in self._documents.items():
			if tid != tenant_id:
				continue
			if doc.status not in eligible_statuses:
				continue
			if not doc.disposal_due_date:
				continue
			due_dt = _parse_iso(doc.disposal_due_date)
			if due_dt is None:
				continue
			if due_dt <= cutoff:
				results.append(doc)
		return results

	# ------------------------------------------------------------------ #
	# Disposal                                                              #
	# ------------------------------------------------------------------ #

	async def dispose_documents(
		self,
		document_ids: list[str],
		method: str,
		authorized_by: str,
		tenant_id: str,
		notes: str = "",
		disposal_approved: bool = True,
	) -> list[EcDisposalRecord]:
		"""Execute disposal of a list of documents; creates immutable disposal records."""
		m = _normalize(method)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "dispose_documents",
			"document_present": bool(document_ids),
			"disposal_method_supported": m in SUPPORTED_DISPOSAL_METHODS,
			"authorized_by_present": _present(authorized_by),
			"disposal_approved": disposal_approved,
		})
		records: list[EcDisposalRecord] = []
		for doc_id in document_ids:
			doc = self._doc_or_raise(doc_id, tenant_id)
			record = EcDisposalRecord(
				id=uuid7str(),
				tenant_id=tenant_id,
				document_id=doc_id,
				document_title=doc.title,
				method=m,
				authorized_by=authorized_by.strip(),
				retention_policy_id=doc.retention_policy_id,
				notes=notes,
			)
			self._disposals[(tenant_id, record.id)] = record
			doc.status = "disposed"
			doc.updated_at = _now()
			records.append(record)
		self._audit(tenant_id, "disposal.executed", f"count={len(records)}")
		return records

	# ------------------------------------------------------------------ #
	# Workflow                                                              #
	# ------------------------------------------------------------------ #

	async def start_review_workflow(
		self,
		document_id: str,
		approvers: list[str],
		workflow_type: str,
		tenant_id: str,
		initiated_by: str = "",
	) -> EcWorkflowInstance:
		"""Launch a multi-step review/approval workflow for a document."""
		doc = self._doc_or_raise(document_id, tenant_id)
		wf_type = _normalize(workflow_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "start_review_workflow",
			"document_present": True,
			"workflow_type_supported": wf_type in SUPPORTED_WORKFLOW_TYPES,
			"approvers_present": bool(approvers),
		})
		steps = [
			EcWorkflowStep(
				step_number=i + 1,
				approver_id=approver.strip(),
			)
			for i, approver in enumerate(approvers)
		]
		wf = EcWorkflowInstance(
			id=uuid7str(),
			tenant_id=tenant_id,
			document_id=document_id,
			workflow_type=wf_type,
			steps=steps,
			current_step=1,
			status="in_progress",
			initiated_by=initiated_by,
		)
		self._workflows[(tenant_id, wf.id)] = wf
		doc.current_workflow_id = wf.id
		doc.updated_at = _now()
		self._audit(tenant_id, "workflow.started", wf.id)
		return wf

	async def approve_workflow_step(
		self,
		workflow_id: str,
		approver_id: str,
		decision: str,
		comments: str,
		tenant_id: str,
	) -> EcWorkflowInstance:
		"""Record a decision on the current step; advance or complete the workflow."""
		wf = self._workflow_or_raise(workflow_id, tenant_id)
		dec = _normalize(decision)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_workflow_step",
			"decision_supported": dec in SUPPORTED_WORKFLOW_DECISIONS,
		})
		assert wf.status == "in_progress", f"workflow {workflow_id!r} is not in_progress (status={wf.status!r})"
		# find current step
		step_idx = wf.current_step - 1  # 0-based
		assert 0 <= step_idx < len(wf.steps), "current_step out of range"
		step = wf.steps[step_idx]
		assert step.approver_id == approver_id, (
			f"step {wf.current_step} expects approver {step.approver_id!r}; got {approver_id!r}"
		)
		step.status = dec
		step.decision = dec
		step.comments = comments
		step.decided_at = _now()

		if dec in ("rejected", "returned_for_revision"):
			wf.status = "rejected" if dec == "rejected" else "in_progress"
			wf.outcome = dec
			wf.completed_at = _now() if dec == "rejected" else None
		elif dec == "escalated":
			# keep current step, mark escalated
			step.status = "escalated"
		else:  # approved
			if wf.current_step >= len(wf.steps):
				# all steps done
				wf.status = "completed"
				wf.outcome = "approved"
				wf.completed_at = _now()
				# update document status to 'approved'
				doc = self._documents.get((tenant_id, wf.document_id))
				if doc:
					doc.status = "approved"
					doc.current_workflow_id = None
					doc.updated_at = _now()
			else:
				wf.current_step += 1

		wf_key = (tenant_id, workflow_id)
		# update in store (object is mutable but keep reference consistent)
		self._workflows[wf_key] = wf
		self._audit(tenant_id, "workflow.step_completed", workflow_id)
		return wf

	# ------------------------------------------------------------------ #
	# Search                                                                #
	# ------------------------------------------------------------------ #

	async def search_documents(
		self,
		query: str,
		filters: dict[str, Any],
		tenant_id: str,
	) -> list[EcDocument]:
		"""Simple in-memory full-text + field-filter search.

		Production deployments should delegate to srch (Elasticsearch/OpenSearch).
		"""
		q = query.strip().lower() if query else ""
		results: list[EcDocument] = []
		for (tid, _), doc in self._documents.items():
			if tid != tenant_id:
				continue
			if doc.status == "disposed":
				continue
			# field filters
			if "status" in filters and doc.status != filters["status"]:
				continue
			if "document_type" in filters and doc.document_type != _normalize(filters["document_type"]):
				continue
			if "retention_category" in filters and doc.retention_category != _normalize(filters["retention_category"]):
				continue
			if "sensitivity" in filters and doc.sensitivity != _normalize(filters["sensitivity"]):
				continue
			if "regulatory_framework" in filters and doc.regulatory_framework != _normalize(filters["regulatory_framework"]):
				continue
			# full-text match on title
			if q and q not in doc.title.lower():
				continue
			results.append(doc)
		return results

	# ------------------------------------------------------------------ #
	# Convenience / analytics                                               #
	# ------------------------------------------------------------------ #

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Count of documents by status and retention category for a tenant."""
		by_status: dict[str, int] = {}
		by_category: dict[str, int] = {}
		for (tid, _), doc in self._documents.items():
			if tid != tenant_id:
				continue
			by_status[doc.status] = by_status.get(doc.status, 0) + 1
			by_category[doc.retention_category] = by_category.get(doc.retention_category, 0) + 1
		return {
			"tenant_id": tenant_id,
			"total_documents": sum(by_status.values()),
			"by_status": by_status,
			"by_retention_category": by_category,
			"total_versions": self._count(self._versions, tenant_id),
			"total_retention_policies": self._count(self._retention_policies, tenant_id),
			"total_workflows": self._count(self._workflows, tenant_id),
			"total_disposals": self._count(self._disposals, tenant_id),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"as_of": _now(),
		}

	def get_version_history(self, document_id: str, tenant_id: str) -> list[EcDocumentVersion]:
		"""Return all versions for a document, newest first."""
		_ = self._doc_or_raise(document_id, tenant_id)
		version_ids = self._doc_version_index.get((tenant_id, document_id), [])
		versions = [
			self._versions[(tenant_id, vid)]
			for vid in version_ids
			if (tenant_id, vid) in self._versions
		]
		return sorted(versions, key=lambda v: v.version_number, reverse=True)

	def list_retention_policies(self, tenant_id: str, active_only: bool = True) -> list[EcRetentionPolicy]:
		"""Return retention policies for a tenant."""
		return [
			p for (tid, _), p in self._retention_policies.items()
			if tid == tenant_id and (not active_only or p.active)
		]

	def list_disposal_records(self, tenant_id: str) -> list[EcDisposalRecord]:
		"""Return all disposal records for a tenant."""
		return [r for (tid, _), r in self._disposals.items() if tid == tenant_id]

	def list_active_workflows(self, tenant_id: str) -> list[EcWorkflowInstance]:
		"""Return in-progress workflows for a tenant."""
		return [
			wf for (tid, _), wf in self._workflows.items()
			if tid == tenant_id and wf.status == "in_progress"
		]

	# ------------------------------------------------------------------ #
	# Private helpers                                                       #
	# ------------------------------------------------------------------ #

	def _doc_or_raise(self, document_id: str, tenant_id: str) -> EcDocument:
		doc = self._documents.get((tenant_id, document_id))
		if doc is None:
			raise KeyError(f"document {document_id!r} not found for tenant {tenant_id!r}")
		return doc

	def _policy_or_raise(self, policy_id: str, tenant_id: str) -> EcRetentionPolicy:
		policy = self._retention_policies.get((tenant_id, policy_id))
		if policy is None:
			raise KeyError(f"retention_policy {policy_id!r} not found for tenant {tenant_id!r}")
		return policy

	def _workflow_or_raise(self, workflow_id: str, tenant_id: str) -> EcWorkflowInstance:
		wf = self._workflows.get((tenant_id, workflow_id))
		if wf is None:
			raise KeyError(f"workflow {workflow_id!r} not found for tenant {tenant_id!r}")
		return wf

	def _find_version_by_number(
		self, document_id: str, version_number: int, tenant_id: str
	) -> EcDocumentVersion | None:
		version_ids = self._doc_version_index.get((tenant_id, document_id), [])
		for vid in version_ids:
			ver = self._versions.get((tenant_id, vid))
			if ver and ver.version_number == version_number:
				return ver
		return None

	def _count(self, store: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for (tid, _) in store if tid == tenant_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"ts": _now(),
			"processor": "bytewax",
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "ecm_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "ecm_policy_denied")


# Canonical alias used by the APG framework loader
CkmEcmService = EcmService
