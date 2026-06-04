"""Executable service layer for APG Intelligence Reporting."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_APPROVAL_TYPES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CITATION_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_DISTRIBUTION_TYPES,
		SUPPORTED_PRODUCT_TYPES,
		SUPPORTED_PUBLICATION_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SECTION_TYPES,
		SUPPORTED_TEMPLATE_TYPES,
		SUPPORTED_WORKSPACE_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		ReportingAgent,
		ReportingApproval,
		ReportingAuthority,
		ReportingCitation,
		ReportingDistribution,
		ReportingProduct,
		ReportingPublication,
		ReportingReview,
		ReportingSection,
		ReportingTemplate,
		ReportingWorkspace,
	)
	from .reporting_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPROVAL_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CITATION_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DISTRIBUTION_TYPES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_PUBLICATION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SECTION_TYPES, SUPPORTED_TEMPLATE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import ReportingAgent, ReportingApproval, ReportingAuthority, ReportingCitation, ReportingDistribution, ReportingProduct, ReportingPublication, ReportingReview, ReportingSection, ReportingTemplate, ReportingWorkspace  # type: ignore
	from reporting_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


# Report lifecycle states
REPORT_DRAFT = "draft"
REPORT_PEER_REVIEW = "peer_review"
REPORT_APPROVED = "approved"
REPORT_DISSEMINATED = "disseminated"
REPORT_ARCHIVED = "archived"

VALID_SEARCH_FIELDS = {"title", "classification", "product_type", "author_id"}


class IntelligenceReportingService:
	"""Tenant-scoped intelligence-reporting runtime for generated APG applications."""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self.authorities: dict[tuple[str, str], ReportingAuthority] = {}
		self.workspaces: dict[tuple[str, str], ReportingWorkspace] = {}
		self.templates: dict[tuple[str, str], ReportingTemplate] = {}
		self.products: dict[tuple[str, str], ReportingProduct] = {}
		self.sections: dict[tuple[str, str], ReportingSection] = {}
		self.citations: dict[tuple[str, str], ReportingCitation] = {}
		self.approvals: dict[tuple[str, str], ReportingApproval] = {}
		self.distributions: dict[tuple[str, str], ReportingDistribution] = {}
		self.publications: dict[tuple[str, str], ReportingPublication] = {}
		self.reviews: dict[tuple[str, str], ReportingReview] = {}
		self.agents: dict[tuple[str, str], ReportingAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Per-report metadata registry: product_id -> lifecycle state + feedback list
		self._report_state: dict[str, dict[str, Any]] = {}
		self._report_feedback: dict[str, list[dict[str, Any]]] = defaultdict(list)

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core CRUD – preserved
	# ------------------------------------------------------------------

	def record_authority(
		self,
		authority_id: str,
		tenant_id: str,
		authority_type: str,
		scope_reference: str,
		classification: str,
		approver_id: str,
		expires_at: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "reporting_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(
		self,
		workspace_id: str,
		tenant_id: str,
		workspace_type: str,
		name: str,
		classification: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_workspace",
			"workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES,
			"workspace_name_present": present(name),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = ReportingWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "reporting_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_template(
		self,
		template_id: str,
		tenant_id: str,
		workspace_id: str,
		template_type: str,
		template_reference: str,
		classification: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		template_type = normalize_code(template_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_template",
			"workspace_present": workspace is not None,
			"template_type_supported": template_type in SUPPORTED_TEMPLATE_TYPES,
			"template_reference_present": present(template_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"evidence_present": present(evidence_reference),
		})
		item = ReportingTemplate(template_id, tenant_id, workspace_id, template_type, template_reference, classification, evidence_reference)
		self.templates[self._tenant_key(tenant_id, template_id)] = item
		self._audit(tenant_id, "reporting_template_recorded", template_id)
		return item.to_dict()

	def record_product(
		self,
		product_id: str,
		tenant_id: str,
		template_id: str,
		product_type: str,
		title: str,
		author_id: str,
		classification: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		template = self._tenant_template_or_none(template_id, tenant_id)
		product_type = normalize_code(product_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_product",
			"template_present": template is not None,
			"product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES,
			"title_present": present(title),
			"author_present": present(author_id),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"evidence_present": present(evidence_reference),
		})
		item = ReportingProduct(product_id, tenant_id, template_id, product_type, title, author_id, classification, evidence_reference)
		self.products[self._tenant_key(tenant_id, product_id)] = item
		self._report_state[product_id] = {"status": REPORT_DRAFT, "created_at": _utcnow()}
		self._audit(tenant_id, "reporting_product_recorded", product_id)
		return item.to_dict()

	def record_section(
		self,
		section_id: str,
		tenant_id: str,
		product_id: str,
		section_type: str,
		section_reference: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		section_type = normalize_code(section_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_section",
			"product_present": product is not None,
			"section_type_supported": section_type in SUPPORTED_SECTION_TYPES,
			"section_reference_present": present(section_reference),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingSection(section_id, tenant_id, product_id, section_type, section_reference, float(confidence_score), evidence_reference)
		self.sections[self._tenant_key(tenant_id, section_id)] = item
		self._audit(tenant_id, "reporting_section_recorded", section_id)
		return item.to_dict()

	def record_citation(
		self,
		citation_id: str,
		tenant_id: str,
		section_id: str,
		citation_type: str,
		source_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		section = self._tenant_section_or_none(section_id, tenant_id)
		citation_type = normalize_code(citation_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_citation",
			"section_present": section is not None,
			"citation_type_supported": citation_type in SUPPORTED_CITATION_TYPES,
			"source_present": present(source_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingCitation(citation_id, tenant_id, section_id, citation_type, source_reference, evidence_reference)
		self.citations[self._tenant_key(tenant_id, citation_id)] = item
		self._audit(tenant_id, "reporting_citation_recorded", citation_id)
		return item.to_dict()

	def record_approval(
		self,
		approval_id: str,
		tenant_id: str,
		product_id: str,
		approval_type: str,
		approver_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		approval_type = normalize_code(approval_type)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_approval",
			"product_present": product is not None,
			"approval_type_supported": approval_type in SUPPORTED_APPROVAL_TYPES,
			"approver_present": present(approver_id),
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"evidence_present": present(evidence_reference),
		})
		item = ReportingApproval(approval_id, tenant_id, product_id, approval_type, approver_id, status, evidence_reference)
		self.approvals[self._tenant_key(tenant_id, approval_id)] = item
		self._audit(tenant_id, "reporting_approval_recorded", approval_id)
		return item.to_dict()

	def record_distribution(
		self,
		distribution_id: str,
		tenant_id: str,
		product_id: str,
		distribution_type: str,
		recipient_reference: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		distribution_type = normalize_code(distribution_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_distribution",
			"product_present": product is not None,
			"distribution_type_supported": distribution_type in SUPPORTED_DISTRIBUTION_TYPES,
			"recipient_present": present(recipient_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingDistribution(distribution_id, tenant_id, product_id, distribution_type, recipient_reference, approval_reference, evidence_reference)
		self.distributions[self._tenant_key(tenant_id, distribution_id)] = item
		self._audit(tenant_id, "reporting_distribution_recorded", distribution_id)
		return item.to_dict()

	def record_publication(
		self,
		publication_id: str,
		tenant_id: str,
		distribution_id: str,
		publication_type: str,
		publication_reference: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		distribution = self._tenant_distribution_or_none(distribution_id, tenant_id)
		publication_type = normalize_code(publication_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_publication",
			"distribution_present": distribution is not None,
			"publication_type_supported": publication_type in SUPPORTED_PUBLICATION_TYPES,
			"publication_reference_present": present(publication_reference),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingPublication(publication_id, tenant_id, distribution_id, publication_type, publication_reference, approval_reference, evidence_reference)
		self.publications[self._tenant_key(tenant_id, publication_id)] = item
		self._audit(tenant_id, "reporting_publication_recorded", publication_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = ReportingReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "reporting_review_recorded", reference_id)
		return item.to_dict()

	def register_reporting_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_reporting_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
			"agent_name_present": present(name),
			"agent_scope_present": present(scope),
		})
		item = ReportingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "reporting_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		uncited_claim_scope: bool = False,
		classification_downgrade_scope: bool = False,
		source_fabrication_scope: bool = False,
		privacy_bypass_scope: bool = False,
		autonomous_publication_scope: bool = False,
		unapproved_distribution_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "reporting_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"uncited_claim_scope": uncited_claim_scope,
			"classification_downgrade_scope": classification_downgrade_scope,
			"source_fabrication_scope": source_fabrication_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"autonomous_publication_scope": autonomous_publication_scope,
			"unapproved_distribution_scope": unapproved_distribution_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "reporting_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.reporting.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"workspace_count": self._count(self.workspaces, tenant_id),
			"template_count": self._count(self.templates, tenant_id),
			"product_count": self._count(self.products, tenant_id),
			"section_count": self._count(self.sections, tenant_id),
			"citation_count": self._count(self.citations, tenant_id),
			"approval_count": self._count(self.approvals, tenant_id),
			"distribution_count": self._count(self.distributions, tenant_id),
			"publication_count": self._count(self.publications, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented reporting operations
	# ------------------------------------------------------------------

	async def create_report(
		self,
		report_type: str,
		classification: str,
		title: str,
		author_id: str,
	) -> dict[str, Any]:
		"""Create a new report product using the first available template for *report_type*."""
		assert present(report_type), "report_type required"
		assert present(classification), "classification required"
		assert present(title), "title required"
		assert present(author_id), "author_id required"

		tenant_id = self.tenant_id
		product_type_norm = normalize_code(report_type)
		classification_norm = normalize_code(classification)

		# Find a template matching the report type in this tenant
		template_id = next(
			(tid for (tnid, tid), t in self.templates.items()
			 if tnid == tenant_id and normalize_code(getattr(t, "template_type", "")) == product_type_norm),
			None,
		)
		if template_id is None:
			raise RuntimeError(f"No template found for report_type={report_type}; register one first")

		product_id = f"rpt_{author_id}_{product_type_norm}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
		result = self.record_product(
			product_id=product_id,
			tenant_id=tenant_id,
			template_id=template_id,
			product_type=product_type_norm,
			title=title,
			author_id=author_id,
			classification=classification_norm,
			evidence_reference=f"created_by:{author_id}",
		)
		self._audit(tenant_id, "report_created", product_id)
		return {**result, "lifecycle_status": REPORT_DRAFT}

	async def add_section(
		self,
		report_id: str,
		section_type: str,
		content: str,
	) -> dict[str, Any]:
		"""Append a section with *content* to *report_id*."""
		assert present(report_id), "report_id required"
		assert present(section_type), "section_type required"
		assert present(content), "content required"

		tenant_id = self.tenant_id
		product = self._tenant_product_or_none(report_id, tenant_id)
		if product is None:
			raise KeyError(f"Report not found: {report_id}")

		state = self._report_state.get(report_id, {})
		if state.get("status") not in {REPORT_DRAFT, REPORT_PEER_REVIEW}:
			raise RuntimeError(f"Cannot add sections to report in status={state.get('status')}")

		section_id = f"sec_{report_id}_{section_type}_{len(self.sections) + 1}"
		# Infer a confidence from content length as a naive proxy
		confidence = min(1.0, max(0.1, len(content) / 5000.0))
		result = self.record_section(
			section_id=section_id,
			tenant_id=tenant_id,
			product_id=report_id,
			section_type=normalize_code(section_type),
			section_reference=content[:200],  # store first 200 chars as reference
			confidence_score=confidence,
			evidence_reference=f"report:{report_id}",
		)
		self._audit(tenant_id, "report_section_added", report_id)
		return result

	async def add_intelligence_item(
		self,
		report_id: str,
		intel_ids: list[str],
	) -> dict[str, Any]:
		"""Attach intelligence items as citations to the last section of *report_id*."""
		assert present(report_id), "report_id required"
		assert isinstance(intel_ids, list) and intel_ids, "intel_ids must be non-empty list"

		tenant_id = self.tenant_id
		product = self._tenant_product_or_none(report_id, tenant_id)
		if product is None:
			raise KeyError(f"Report not found: {report_id}")

		# Find latest section for this product
		product_sections = [
			(sid, s) for (tid, sid), s in self.sections.items()
			if tid == tenant_id and getattr(s, "product_id", "") == report_id
		]
		if not product_sections:
			raise RuntimeError(f"No sections found for report {report_id}; add a section first")

		# Use the most recently registered section (last in insertion order)
		latest_section_id = product_sections[-1][0]
		added: list[dict[str, Any]] = []
		for intel_id in intel_ids:
			citation_id = f"cit_{report_id}_{intel_id}"
			citation = self.record_citation(
				citation_id=citation_id,
				tenant_id=tenant_id,
				section_id=latest_section_id,
				citation_type=normalize_code("intelligence_item"),
				source_reference=intel_id,
				evidence_reference=f"intel:{intel_id}",
			)
			added.append(citation)

		self._audit(tenant_id, "intelligence_items_added", report_id)
		return {
			"report_id": report_id,
			"section_id": latest_section_id,
			"added_count": len(added),
			"citations": added,
		}

	async def peer_review(
		self,
		report_id: str,
		reviewer_id: str,
		comments: str,
	) -> dict[str, Any]:
		"""Submit peer review for *report_id* with *comments*."""
		assert present(report_id), "report_id required"
		assert present(reviewer_id), "reviewer_id required"
		assert present(comments), "comments required"

		tenant_id = self.tenant_id
		state = self._report_state.get(report_id, {})
		if state.get("status") not in {REPORT_DRAFT, REPORT_PEER_REVIEW}:
			raise RuntimeError(f"Report {report_id} is not in a reviewable state: {state.get('status')}")

		review_id = f"rev_{report_id}_{reviewer_id}"
		result = self.record_review(
			review_id=review_id,
			tenant_id=tenant_id,
			reference_id=report_id,
			reviewer_id=reviewer_id,
			status=normalize_code("pending"),
			evidence_reference=comments[:500],
		)
		# Advance lifecycle
		state["status"] = REPORT_PEER_REVIEW
		state["last_reviewer"] = reviewer_id
		state["review_comments"] = comments
		self._report_state[report_id] = state
		self._audit(tenant_id, "report_peer_reviewed", report_id)
		return {**result, "lifecycle_status": REPORT_PEER_REVIEW, "comments_length": len(comments)}

	async def approve_report(self, report_id: str, approver_id: str) -> dict[str, Any]:
		"""Approve *report_id* for dissemination."""
		assert present(report_id), "report_id required"
		assert present(approver_id), "approver_id required"

		tenant_id = self.tenant_id
		state = self._report_state.get(report_id, {})
		if state.get("status") != REPORT_PEER_REVIEW:
			raise RuntimeError(f"Report {report_id} must be in peer_review state before approval; current={state.get('status')}")

		approval_id = f"appr_{report_id}_{approver_id}"
		result = self.record_approval(
			approval_id=approval_id,
			tenant_id=tenant_id,
			product_id=report_id,
			approval_type=normalize_code("editorial"),
			approver_id=approver_id,
			status=normalize_code("approved"),
			evidence_reference=f"approved_by:{approver_id}",
		)
		state["status"] = REPORT_APPROVED
		state["approver_id"] = approver_id
		state["approved_at"] = _utcnow()
		self._report_state[report_id] = state
		self._audit(tenant_id, "report_approved", report_id)
		return {**result, "lifecycle_status": REPORT_APPROVED}

	async def disseminate_report(
		self,
		report_id: str,
		distribution_list: list[str],
	) -> dict[str, Any]:
		"""Disseminate *report_id* to each recipient in *distribution_list*."""
		assert present(report_id), "report_id required"
		assert isinstance(distribution_list, list) and distribution_list, "distribution_list must be non-empty"

		tenant_id = self.tenant_id
		state = self._report_state.get(report_id, {})
		if state.get("status") != REPORT_APPROVED:
			raise RuntimeError(f"Report {report_id} must be approved before dissemination; current={state.get('status')}")

		approver_id = state.get("approver_id", "system")
		distributions_created: list[dict[str, Any]] = []
		for recipient in distribution_list:
			dist_id = f"dist_{report_id}_{recipient.replace(' ', '_')}"
			dist = self.record_distribution(
				distribution_id=dist_id,
				tenant_id=tenant_id,
				product_id=report_id,
				distribution_type=normalize_code("recipient"),
				recipient_reference=recipient,
				approval_reference=f"approved_by:{approver_id}",
				evidence_reference=f"dissemination:{report_id}",
			)
			distributions_created.append(dist)

		state["status"] = REPORT_DISSEMINATED
		state["disseminated_at"] = _utcnow()
		state["recipient_count"] = len(distribution_list)
		self._report_state[report_id] = state
		self._audit(tenant_id, "report_disseminated", report_id)
		return {
			"report_id": report_id,
			"recipient_count": len(distribution_list),
			"distributions": distributions_created,
			"lifecycle_status": REPORT_DISSEMINATED,
			"disseminated_at": state["disseminated_at"],
		}

	async def report_feedback(
		self,
		report_id: str,
		recipient_id: str,
		feedback: str,
	) -> dict[str, Any]:
		"""Record post-dissemination feedback from *recipient_id*."""
		assert present(report_id), "report_id required"
		assert present(recipient_id), "recipient_id required"
		assert present(feedback), "feedback required"

		entry = {
			"recipient_id": recipient_id,
			"feedback": feedback,
			"submitted_at": _utcnow(),
		}
		self._report_feedback[report_id].append(entry)
		self._audit(self.tenant_id, "report_feedback_received", report_id)
		return {
			"report_id": report_id,
			"recipient_id": recipient_id,
			"feedback_count": len(self._report_feedback[report_id]),
			"recorded_at": entry["submitted_at"],
		}

	async def archive_report(self, report_id: str) -> dict[str, Any]:
		"""Mark *report_id* as archived."""
		assert present(report_id), "report_id required"
		tenant_id = self.tenant_id
		state = self._report_state.get(report_id)
		if state is None:
			raise KeyError(f"Report not found: {report_id}")
		state["status"] = REPORT_ARCHIVED
		state["archived_at"] = _utcnow()
		self._report_state[report_id] = state
		self._audit(tenant_id, "report_archived", report_id)
		return {
			"report_id": report_id,
			"lifecycle_status": REPORT_ARCHIVED,
			"archived_at": state["archived_at"],
		}

	async def report_search(
		self,
		query: str,
		filters: dict[str, str] | None = None,
	) -> list[dict[str, Any]]:
		"""Full-scan search across report products. *query* matched against title; *filters* on fields."""
		assert present(query) or filters, "query or filters required"
		tenant_id = self.tenant_id
		filters = filters or {}

		results: list[dict[str, Any]] = []
		for (tid, pid), product in self.products.items():
			if tid != tenant_id:
				continue
			title = getattr(product, "title", "")
			if query and query.lower() not in title.lower():
				continue
			# Apply filters
			match = True
			for field, value in filters.items():
				if field in VALID_SEARCH_FIELDS:
					attr = getattr(product, field, "")
					if value.lower() not in str(attr).lower():
						match = False
						break
			if not match:
				continue

			state = self._report_state.get(pid, {})
			entry = product.to_dict() if hasattr(product, "to_dict") else {"product_id": pid}
			entry["lifecycle_status"] = state.get("status", "unknown")
			results.append(entry)

		self._audit(tenant_id, "report_search_executed", f"q={query}")
		return results

	async def reporting_analytics(self, period: str = "30d") -> dict[str, Any]:
		"""Aggregate reporting statistics for the current tenant over *period*."""
		assert present(period), "period required"
		tenant_id = self.tenant_id

		# Product type distribution
		type_dist: dict[str, int] = defaultdict(int)
		classification_dist: dict[str, int] = defaultdict(int)
		lifecycle_dist: dict[str, int] = defaultdict(int)
		for (tid, pid), product in self.products.items():
			if tid != tenant_id:
				continue
			type_dist[getattr(product, "product_type", "unknown")] += 1
			classification_dist[getattr(product, "classification", "unknown")] += 1
			state = self._report_state.get(pid, {})
			lifecycle_dist[state.get("status", "unknown")] += 1

		# Citation confidence distribution
		confidence_scores = [
			getattr(s, "confidence_score", 0.0)
			for (tid, _), s in self.sections.items()
			if tid == tenant_id
		]
		avg_confidence = round(statistics.mean(confidence_scores), 4) if confidence_scores else 0.0

		# Feedback volume
		total_feedback = sum(len(v) for v in self._report_feedback.values())

		self._audit(tenant_id, "reporting_analytics_computed", period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"product_count": self._count(self.products, tenant_id),
			"section_count": self._count(self.sections, tenant_id),
			"citation_count": self._count(self.citations, tenant_id),
			"avg_section_confidence": avg_confidence,
			"by_product_type": dict(type_dist),
			"by_classification": dict(classification_dist),
			"by_lifecycle_status": dict(lifecycle_dist),
			"total_feedback_submissions": total_feedback,
			"computed_at": _utcnow(),
		}

	async def get_report_state(self, report_id: str) -> dict[str, Any]:
		"""Return full lifecycle state for *report_id*."""
		state = self._report_state.get(report_id)
		if state is None:
			raise KeyError(f"Report not found: {report_id}")
		return {"report_id": report_id, **state}

	async def citation_integrity_check(self, report_id: str) -> dict[str, Any]:
		"""Verify all sections in *report_id* have at least one citation."""
		tenant_id = self.tenant_id
		section_ids = {
			sid for (tid, sid), s in self.sections.items()
			if tid == tenant_id and getattr(s, "product_id", "") == report_id
		}
		cited_section_ids = {
			getattr(c, "section_id", "")
			for (tid, _), c in self.citations.items()
			if tid == tenant_id
		}
		uncited = section_ids - cited_section_ids
		return {
			"report_id": report_id,
			"total_sections": len(section_ids),
			"cited_sections": len(section_ids - uncited),
			"uncited_sections": list(uncited),
			"citation_coverage": round((len(section_ids) - len(uncited)) / len(section_ids), 4) if section_ids else 1.0,
			"checked_at": _utcnow(),
		}

	async def report_workflow(
		self,
		report_type: str,
		classification: str,
		title: str,
		author_id: str,
		distribution_list: list[str],
	) -> dict[str, Any]:
		"""End-to-end report workflow: create → peer_review stub → approve → disseminate."""
		assert present(report_type) and present(classification) and present(title) and present(author_id), "all params required"
		assert distribution_list, "distribution_list required"
		rpt = await self.create_report(report_type, classification, title, author_id)
		product_id = rpt.get("product_id", "")
		state = self._report_state.get(product_id, {})
		# Fast-track: directly approve (workflow shortcut for automation)
		state["status"] = REPORT_PEER_REVIEW
		self._report_state[product_id] = state
		approved = await self.approve_report(product_id, author_id)
		disseminated = await self.disseminate_report(product_id, distribution_list)
		self._audit(self.tenant_id, "report_workflow_completed", product_id)
		return {
			"product_id": product_id,
			"title": title,
			"classification": classification,
			"recipient_count": len(distribution_list),
			"lifecycle_status": REPORT_DISSEMINATED,
			"completed_at": _utcnow(),
		}

	async def dissemination_track(
		self,
		product_id: str,
	) -> dict[str, Any]:
		"""Track all dissemination records for *product_id*."""
		assert present(product_id), "product_id required"
		tenant_id = self.tenant_id
		dists = [
			{"distribution_id": did, "recipient": getattr(d, "recipient_reference", ""), "type": getattr(d, "distribution_type", "")}
			for (tid, did), d in self.distributions.items()
			if tid == tenant_id and getattr(d, "product_id", "") == product_id
		]
		pubs = [
			{"publication_id": pid, "type": getattr(p, "publication_type", "")}
			for (tid, pid), p in self.publications.items()
			if tid == tenant_id
		]
		self._audit(tenant_id, "dissemination_tracked", product_id)
		return {
			"product_id": product_id,
			"distribution_count": len(dists),
			"distributions": dists,
			"publication_count": len(pubs),
			"retrieved_at": _utcnow(),
		}

	async def intelligence_score(
		self,
		product_id: str,
	) -> dict[str, Any]:
		"""Compute an intelligence value score for *product_id* based on citation coverage and confidence."""
		assert present(product_id), "product_id required"
		tenant_id = self.tenant_id
		integrity = await self.citation_integrity_check(product_id)
		coverage = float(integrity.get("citation_coverage", 0.0))
		sections = integrity.get("total_sections", 0)
		section_scores = [
			getattr(s, "confidence_score", 0.0)
			for (tid, _), s in self.sections.items()
			if tid == tenant_id and getattr(s, "product_id", "") == product_id
		]
		mean_conf = round(statistics.mean(section_scores), 4) if section_scores else 0.0
		intel_score = round((coverage * 0.5 + mean_conf * 0.5), 4)
		self._audit(tenant_id, "intelligence_score_computed", product_id)
		return {
			"product_id": product_id,
			"citation_coverage": coverage,
			"mean_section_confidence": mean_conf,
			"section_count": sections,
			"intelligence_score": intel_score,
			"grade": "A" if intel_score >= 0.8 else "B" if intel_score >= 0.6 else "C" if intel_score >= 0.4 else "D",
			"computed_at": _utcnow(),
		}

	async def report_search_advanced(
		self,
		query: str,
		classification: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""Search reports with optional *classification* and lifecycle *status* filters."""
		filters: dict[str, str] = {}
		if classification:
			filters["classification"] = classification
		results = await self.report_search(query, filters)
		if status:
			results = [r for r in results if r.get("lifecycle_status") == status]
		self._audit(self.tenant_id, "report_search_advanced_executed", f"q={query}")
		return results

	async def report_archive_batch(
		self,
		product_ids: list[str],
	) -> dict[str, Any]:
		"""Archive multiple reports in a single call."""
		assert product_ids, "product_ids required"
		archived: list[str] = []
		failed: list[dict[str, Any]] = []
		for pid in product_ids:
			try:
				await self.archive_report(pid)
				archived.append(pid)
			except Exception as exc:
				failed.append({"product_id": pid, "error": str(exc)})
		self._audit(self.tenant_id, "report_archive_batch_completed", f"count={len(product_ids)}")
		return {"archived": archived, "failed": failed, "total": len(product_ids), "archived_at": _utcnow()}

	async def analytic_judgment(
		self,
		product_id: str,
		judgment: str,
		analyst_id: str,
	) -> dict[str, Any]:
		"""Record an analytic judgment for *product_id*."""
		assert present(product_id) and present(judgment) and present(analyst_id), "all params required"
		tenant_id = self.tenant_id
		judgment_id = f"aj_{product_id}_{analyst_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
		self._report_feedback[product_id].append({
			"type": "analytic_judgment",
			"analyst_id": analyst_id,
			"judgment": judgment,
			"recorded_at": _utcnow(),
		})
		self._audit(tenant_id, "analytic_judgment_recorded", judgment_id)
		return {"judgment_id": judgment_id, "product_id": product_id, "analyst_id": analyst_id, "judgment": judgment, "recorded_at": _utcnow()}

	async def key_judgment(
		self,
		product_id: str,
	) -> list[dict[str, Any]]:
		"""Extract key analytic judgments for *product_id*."""
		assert present(product_id), "product_id required"
		judgments = [
			fb for fb in self._report_feedback.get(product_id, [])
			if fb.get("type") == "analytic_judgment"
		]
		self._audit(self.tenant_id, "key_judgments_retrieved", product_id)
		return judgments

	async def caveat_add(
		self,
		product_id: str,
		caveat: str,
		analyst_id: str,
	) -> dict[str, Any]:
		"""Add a caveat to *product_id* (e.g. source limitations, time-sensitivity)."""
		assert present(product_id) and present(caveat) and present(analyst_id), "all params required"
		tenant_id = self.tenant_id
		caveat_id = f"cav_{product_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
		self._report_feedback[product_id].append({
			"type": "caveat",
			"analyst_id": analyst_id,
			"caveat": caveat,
			"recorded_at": _utcnow(),
		})
		self._audit(tenant_id, "caveat_added", caveat_id)
		return {"caveat_id": caveat_id, "product_id": product_id, "caveat": caveat, "analyst_id": analyst_id, "added_at": _utcnow()}

	async def report_analytics_extended(
		self,
		period: str = "30d",
	) -> dict[str, Any]:
		"""Extended report analytics including judgment and caveat counts."""
		base = await self.reporting_analytics(period)
		total_judgments = sum(
			1 for fb_list in self._report_feedback.values()
			for fb in fb_list if fb.get("type") == "analytic_judgment"
		)
		total_caveats = sum(
			1 for fb_list in self._report_feedback.values()
			for fb in fb_list if fb.get("type") == "caveat"
		)
		base["total_judgments"] = total_judgments
		base["total_caveats"] = total_caveats
		return base

	async def report_index(self) -> list[dict[str, Any]]:
		"""Return an index of all reports with their lifecycle status."""
		tenant_id = self.tenant_id
		index = []
		for (tid, pid), product in self.products.items():
			if tid != tenant_id:
				continue
			state = self._report_state.get(pid, {})
			index.append({
				"product_id": pid,
				"title": getattr(product, "title", ""),
				"product_type": getattr(product, "product_type", ""),
				"classification": getattr(product, "classification", ""),
				"lifecycle_status": state.get("status", "unknown"),
				"created_at": state.get("created_at", ""),
			})
		index.sort(key=lambda x: x["created_at"], reverse=True)
		self._audit(tenant_id, "report_index_retrieved", "all")
		return index

	async def pending_approvals(self) -> list[dict[str, Any]]:
		"""List reports currently in peer_review awaiting approval."""
		tenant_id = self.tenant_id
		pending = [
			{"report_id": pid, **state}
			for pid, state in self._report_state.items()
			if state.get("status") == REPORT_PEER_REVIEW
		]
		# Enrich with product metadata
		result = []
		for p in pending:
			product = self._tenant_product_or_none(p["report_id"], tenant_id)
			if product:
				p["title"] = getattr(product, "title", "")
				p["classification"] = getattr(product, "classification", "")
			result.append(p)
		return result

	async def template_usage_report(self) -> list[dict[str, Any]]:
		"""Count how many products were created from each template."""
		tenant_id = self.tenant_id
		template_usage: dict[str, int] = defaultdict(int)
		for (tid, _), product in self.products.items():
			if tid == tenant_id:
				tmpl = getattr(product, "template_id", "unknown")
				template_usage[tmpl] += 1

		result = []
		for (tid, tmpl_id), template in self.templates.items():
			if tid != tenant_id:
				continue
			result.append({
				"template_id": tmpl_id,
				"template_type": getattr(template, "template_type", ""),
				"product_count": template_usage.get(tmpl_id, 0),
			})
		result.sort(key=lambda x: x["product_count"], reverse=True)
		return result

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> ReportingAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> ReportingWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_template_or_none(self, item_id: str, tenant_id: str) -> ReportingTemplate | None:
		return self.templates.get(self._tenant_key(tenant_id, item_id))

	def _tenant_product_or_none(self, item_id: str, tenant_id: str) -> ReportingProduct | None:
		return self.products.get(self._tenant_key(tenant_id, item_id))

	def _tenant_section_or_none(self, item_id: str, tenant_id: str) -> ReportingSection | None:
		return self.sections.get(self._tenant_key(tenant_id, item_id))

	def _tenant_distribution_or_none(self, item_id: str, tenant_id: str) -> ReportingDistribution | None:
		return self.distributions.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _utcnow(),
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "reporting_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "reporting_policy_denied")


IntelReportingService = IntelligenceReportingService
