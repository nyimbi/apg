"""Executable service layer for APG InsurTech."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_CLAIM_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_PRODUCT_LINES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract
	from .insurance_runtime import normalize_code, normalize_currency, positive_minor, score_present
	from .models import ClaimRecord, InsuranceAlert, InsuranceDocument, InsuranceEvidence, InsuranceProduct, InsuranceReview, Policy, Policyholder, PremiumRecord, Quote, ReinsuranceAttachment, RiskAssessment
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_CLAIM_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DOCUMENT_TYPES, SUPPORTED_PRODUCT_LINES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from insurance_runtime import normalize_code, normalize_currency, positive_minor, score_present  # type: ignore
	from models import ClaimRecord, InsuranceAlert, InsuranceDocument, InsuranceEvidence, InsuranceProduct, InsuranceReview, Policy, Policyholder, PremiumRecord, Quote, ReinsuranceAttachment, RiskAssessment  # type: ignore


class InsurTechService:
	"""In-memory InsurTech runtime for generated APG applications."""

	def __init__(self) -> None:
		self.policyholders: dict[str, Policyholder] = {}
		self.products: dict[str, InsuranceProduct] = {}
		self.quotes: dict[str, Quote] = {}
		self.policies: dict[str, Policy] = {}
		self.premiums: dict[str, PremiumRecord] = {}
		self.claims: dict[str, ClaimRecord] = {}
		self.documents: dict[str, InsuranceDocument] = {}
		self.risk: dict[str, RiskAssessment] = {}
		self.reinsurance: dict[str, ReinsuranceAttachment] = {}
		self.compliance: dict[str, InsuranceAlert] = {}
		self.reviews: dict[str, InsuranceReview] = {}
		self.evidence: dict[str, InsuranceEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def onboard_policyholder(self, policyholder_id: str, tenant_id: str, name: str, kyc_reference: str, contact_reference: str, risk_profile_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_policyholder", "kyc_present": bool(kyc_reference), "contact_present": bool(contact_reference)})
		item = Policyholder(policyholder_id, tenant_id, name, kyc_reference, contact_reference, risk_profile_reference)
		self.policyholders[policyholder_id] = item
		self._audit(tenant_id, "policyholder_onboarded", policyholder_id)
		return item.to_dict()

	def publish_product(self, product_id: str, tenant_id: str, name: str, product_line: str, coverage_terms_reference: str, pricing_reference: str) -> dict[str, Any]:
		product_line = normalize_code(product_line)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_product", "product_line_supported": product_line in SUPPORTED_PRODUCT_LINES, "coverage_terms_present": bool(coverage_terms_reference)})
		item = InsuranceProduct(product_id, tenant_id, name, product_line, coverage_terms_reference, pricing_reference)
		self.products[product_id] = item
		self._audit(tenant_id, "insurance_product_published", product_id)
		return item.to_dict()

	def generate_quote(self, quote_id: str, tenant_id: str, policyholder_id: str, product_id: str, premium_minor: int, currency: str, underwriting_reference: str) -> dict[str, Any]:
		policyholder = self._tenant_policyholder_or_none(policyholder_id, tenant_id)
		product = self._tenant_product_or_none(product_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "generate_quote", "policyholder_present": policyholder is not None, "product_present": product is not None, "positive_premium": positive_minor(premium_minor), "underwriting_reference_present": bool(underwriting_reference)})
		item = Quote(quote_id, tenant_id, policyholder_id, product_id, int(premium_minor), currency, underwriting_reference)
		self.quotes[quote_id] = item
		self._audit(tenant_id, "quote_generated", quote_id)
		return item.to_dict()

	def bind_policy(self, policy_id: str, tenant_id: str, quote_id: str, effective_date: str, payment_reference: str) -> dict[str, Any]:
		quote = self._tenant_quote_or_none(quote_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "bind_policy", "quote_present": quote is not None, "payment_reference_present": bool(payment_reference)})
		item = Policy(policy_id, tenant_id, quote_id, effective_date, payment_reference)
		self.policies[policy_id] = item
		self._audit(tenant_id, "policy_bound", policy_id)
		return item.to_dict()

	def record_premium(self, premium_id: str, tenant_id: str, policy_id: str, amount_minor: int, currency: str, payment_reference: str) -> dict[str, Any]:
		policy = self._tenant_policy_or_none(policy_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_premium", "policy_present": policy is not None, "positive_amount": positive_minor(amount_minor), "currency_supported": currency in SUPPORTED_CURRENCIES, "payment_reference_present": bool(payment_reference)})
		item = PremiumRecord(premium_id, tenant_id, policy_id, int(amount_minor), currency, payment_reference)
		self.premiums[premium_id] = item
		self._audit(tenant_id, "premium_recorded", premium_id)
		return item.to_dict()

	def open_claim(self, claim_id: str, tenant_id: str, policy_id: str, claim_type: str, amount_minor: int, loss_date: str, evidence_reference: str) -> dict[str, Any]:
		policy = self._tenant_policy_or_none(policy_id, tenant_id)
		claim_type = normalize_code(claim_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_claim", "policy_present": policy is not None, "claim_type_supported": claim_type in SUPPORTED_CLAIM_TYPES, "positive_amount": positive_minor(amount_minor), "evidence_present": bool(evidence_reference) and bool(loss_date)})
		item = ClaimRecord(claim_id, tenant_id, policy_id, claim_type, int(amount_minor), loss_date, evidence_reference)
		self.claims[claim_id] = item
		self._audit(tenant_id, "claim_opened", claim_id)
		return item.to_dict()

	def record_document(self, document_id: str, tenant_id: str, reference_id: str, document_type: str, evidence_reference: str) -> dict[str, Any]:
		document_type = normalize_code(document_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_document", "document_type_supported": document_type in SUPPORTED_DOCUMENT_TYPES, "evidence_present": bool(evidence_reference) and bool(reference_id)})
		item = InsuranceDocument(document_id, tenant_id, reference_id, document_type, evidence_reference)
		self.documents[document_id] = item
		self._audit(tenant_id, "document_recorded", document_id)
		return item.to_dict()

	def record_risk_assessment(self, assessment_id: str, tenant_id: str, policyholder_id: str, score: float, source_reference: str) -> dict[str, Any]:
		policyholder = self._tenant_policyholder_or_none(policyholder_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk_assessment", "policyholder_present": policyholder is not None, "score_present": score_present(score), "source_present": bool(source_reference)})
		item = RiskAssessment(assessment_id, tenant_id, policyholder_id, float(score), source_reference)
		self.risk[assessment_id] = item
		self._audit(tenant_id, "risk_assessment_recorded", assessment_id)
		return item.to_dict()

	def record_reinsurance_attachment(self, attachment_id: str, tenant_id: str, policy_id: str, treaty_reference: str, share_percent: float) -> dict[str, Any]:
		policy = self._tenant_policy_or_none(policy_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_reinsurance_attachment", "policy_present": policy is not None, "treaty_reference_present": bool(treaty_reference), "positive_share": float(share_percent) > 0})
		item = ReinsuranceAttachment(attachment_id, tenant_id, policy_id, treaty_reference, float(share_percent))
		self.reinsurance[attachment_id] = item
		self._audit(tenant_id, "reinsurance_attachment_recorded", attachment_id)
		return item.to_dict()

	def record_compliance_alert(self, alert_id: str, tenant_id: str, reference_id: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_compliance_alert", "severity_supported": severity in SUPPORTED_ALERT_SEVERITIES, "evidence_present": bool(evidence_reference)})
		item = InsuranceAlert(alert_id, tenant_id, reference_id, severity, evidence_reference)
		self.compliance[alert_id] = item
		self._audit(tenant_id, "insurance_compliance_alert_recorded", alert_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": bool(evidence_reference) and bool(reviewer_id)})
		item = InsuranceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "insurance_review_recorded", review_id)
		return item.to_dict()

	def register_insurance_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_insurance_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = InsuranceEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = item
		self._audit(tenant_id, "insurance_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "insurance_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "insurance_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.insurance.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "policyholder_count": self._count(self.policyholders, tenant_id), "product_count": self._count(self.products, tenant_id), "quote_count": self._count(self.quotes, tenant_id), "policy_count": self._count(self.policies, tenant_id), "premium_count": self._count(self.premiums, tenant_id), "claim_count": self._count(self.claims, tenant_id), "document_count": self._count(self.documents, tenant_id), "risk_count": self._count(self.risk, tenant_id), "reinsurance_count": self._count(self.reinsurance, tenant_id), "compliance_count": self._count(self.compliance, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_policyholder_or_none(self, item_id: str, tenant_id: str) -> Policyholder | None:
		item = self.policyholders.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_product_or_none(self, item_id: str, tenant_id: str) -> InsuranceProduct | None:
		item = self.products.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_quote_or_none(self, item_id: str, tenant_id: str) -> Quote | None:
		item = self.quotes.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_policy_or_none(self, item_id: str, tenant_id: str) -> Policy | None:
		item = self.policies.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "insurance_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "insurance_policy_denied")
