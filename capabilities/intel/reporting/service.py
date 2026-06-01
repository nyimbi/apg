"""Executable service layer for APG Intelligence Reporting."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPROVAL_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CITATION_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DISTRIBUTION_TYPES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_PUBLICATION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SECTION_TYPES, SUPPORTED_TEMPLATE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import ReportingAgent, ReportingApproval, ReportingAuthority, ReportingCitation, ReportingDistribution, ReportingProduct, ReportingPublication, ReportingReview, ReportingSection, ReportingTemplate, ReportingWorkspace
	from .reporting_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPROVAL_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CITATION_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_DISTRIBUTION_TYPES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_PUBLICATION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SECTION_TYPES, SUPPORTED_TEMPLATE_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import ReportingAgent, ReportingApproval, ReportingAuthority, ReportingCitation, ReportingDistribution, ReportingProduct, ReportingPublication, ReportingReview, ReportingSection, ReportingTemplate, ReportingWorkspace  # type: ignore
	from reporting_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class IntelligenceReportingService:
	"""Tenant-scoped intelligence-reporting runtime for generated APG applications."""

	def __init__(self) -> None:
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

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = ReportingAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "reporting_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = ReportingWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "reporting_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_template(self, template_id: str, tenant_id: str, workspace_id: str, template_type: str, template_reference: str, classification: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		template_type = normalize_code(template_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_template", "workspace_present": workspace is not None, "template_type_supported": template_type in SUPPORTED_TEMPLATE_TYPES, "template_reference_present": present(template_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "evidence_present": present(evidence_reference)})
		item = ReportingTemplate(template_id, tenant_id, workspace_id, template_type, template_reference, classification, evidence_reference)
		self.templates[self._tenant_key(tenant_id, template_id)] = item
		self._audit(tenant_id, "reporting_template_recorded", template_id)
		return item.to_dict()

	def record_product(self, product_id: str, tenant_id: str, template_id: str, product_type: str, title: str, author_id: str, classification: str, evidence_reference: str) -> dict[str, Any]:
		template = self._tenant_template_or_none(template_id, tenant_id)
		product_type = normalize_code(product_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_product", "template_present": template is not None, "product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES, "title_present": present(title), "author_present": present(author_id), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "evidence_present": present(evidence_reference)})
		item = ReportingProduct(product_id, tenant_id, template_id, product_type, title, author_id, classification, evidence_reference)
		self.products[self._tenant_key(tenant_id, product_id)] = item
		self._audit(tenant_id, "reporting_product_recorded", product_id)
		return item.to_dict()

	def record_section(self, section_id: str, tenant_id: str, product_id: str, section_type: str, section_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		section_type = normalize_code(section_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_section", "product_present": product is not None, "section_type_supported": section_type in SUPPORTED_SECTION_TYPES, "section_reference_present": present(section_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = ReportingSection(section_id, tenant_id, product_id, section_type, section_reference, float(confidence_score), evidence_reference)
		self.sections[self._tenant_key(tenant_id, section_id)] = item
		self._audit(tenant_id, "reporting_section_recorded", section_id)
		return item.to_dict()

	def record_citation(self, citation_id: str, tenant_id: str, section_id: str, citation_type: str, source_reference: str, evidence_reference: str) -> dict[str, Any]:
		section = self._tenant_section_or_none(section_id, tenant_id)
		citation_type = normalize_code(citation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_citation", "section_present": section is not None, "citation_type_supported": citation_type in SUPPORTED_CITATION_TYPES, "source_present": present(source_reference), "evidence_present": present(evidence_reference)})
		item = ReportingCitation(citation_id, tenant_id, section_id, citation_type, source_reference, evidence_reference)
		self.citations[self._tenant_key(tenant_id, citation_id)] = item
		self._audit(tenant_id, "reporting_citation_recorded", citation_id)
		return item.to_dict()

	def record_approval(self, approval_id: str, tenant_id: str, product_id: str, approval_type: str, approver_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		approval_type = normalize_code(approval_type)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_approval", "product_present": product is not None, "approval_type_supported": approval_type in SUPPORTED_APPROVAL_TYPES, "approver_present": present(approver_id), "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": present(evidence_reference)})
		item = ReportingApproval(approval_id, tenant_id, product_id, approval_type, approver_id, status, evidence_reference)
		self.approvals[self._tenant_key(tenant_id, approval_id)] = item
		self._audit(tenant_id, "reporting_approval_recorded", approval_id)
		return item.to_dict()

	def record_distribution(self, distribution_id: str, tenant_id: str, product_id: str, distribution_type: str, recipient_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		product = self._tenant_product_or_none(product_id, tenant_id)
		distribution_type = normalize_code(distribution_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_distribution", "product_present": product is not None, "distribution_type_supported": distribution_type in SUPPORTED_DISTRIBUTION_TYPES, "recipient_present": present(recipient_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = ReportingDistribution(distribution_id, tenant_id, product_id, distribution_type, recipient_reference, approval_reference, evidence_reference)
		self.distributions[self._tenant_key(tenant_id, distribution_id)] = item
		self._audit(tenant_id, "reporting_distribution_recorded", distribution_id)
		return item.to_dict()

	def record_publication(self, publication_id: str, tenant_id: str, distribution_id: str, publication_type: str, publication_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		distribution = self._tenant_distribution_or_none(distribution_id, tenant_id)
		publication_type = normalize_code(publication_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_publication", "distribution_present": distribution is not None, "publication_type_supported": publication_type in SUPPORTED_PUBLICATION_TYPES, "publication_reference_present": present(publication_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = ReportingPublication(publication_id, tenant_id, distribution_id, publication_type, publication_reference, approval_reference, evidence_reference)
		self.publications[self._tenant_key(tenant_id, publication_id)] = item
		self._audit(tenant_id, "reporting_publication_recorded", publication_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = ReportingReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "reporting_review_recorded", reference_id)
		return item.to_dict()

	def register_reporting_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_reporting_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES, "agent_name_present": present(name), "agent_scope_present": present(scope)})
		item = ReportingAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "reporting_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, uncited_claim_scope: bool = False, classification_downgrade_scope: bool = False, source_fabrication_scope: bool = False, privacy_bypass_scope: bool = False, autonomous_publication_scope: bool = False, unapproved_distribution_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "reporting_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "uncited_claim_scope": uncited_claim_scope, "classification_downgrade_scope": classification_downgrade_scope, "source_fabrication_scope": source_fabrication_scope, "privacy_bypass_scope": privacy_bypass_scope, "autonomous_publication_scope": autonomous_publication_scope, "unapproved_distribution_scope": unapproved_distribution_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "reporting_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.reporting.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "template_count": self._count(self.templates, tenant_id), "product_count": self._count(self.products, tenant_id), "section_count": self._count(self.sections, tenant_id), "citation_count": self._count(self.citations, tenant_id), "approval_count": self._count(self.approvals, tenant_id), "distribution_count": self._count(self.distributions, tenant_id), "publication_count": self._count(self.publications, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

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
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "reporting_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "reporting_policy_denied")


IntelReportingService = IntelligenceReportingService

