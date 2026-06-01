"""Executable service layer for APG Know Your Customer."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CUSTOMER_TYPES, SUPPORTED_DOCUMENT_TYPES, evaluate_capability_rules, get_capability_contract
	from .kyc_runtime import normalize_code, normalize_confidence, normalize_country, normalize_risk_score, risk_band
	from .models import KycDecision, KycDocument, KycEvidence, KycProfile, KycScreening
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CUSTOMER_TYPES, SUPPORTED_DOCUMENT_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from kyc_runtime import normalize_code, normalize_confidence, normalize_country, normalize_risk_score, risk_band  # type: ignore
	from models import KycDecision, KycDocument, KycEvidence, KycProfile, KycScreening  # type: ignore


class KnowYourCustomerService:
	"""Dependency-light KYC lifecycle runtime for generated applications."""

	def __init__(self) -> None:
		self.profiles: dict[str, KycProfile] = {}
		self.documents: dict[str, KycDocument] = {}
		self.screenings: dict[str, KycScreening] = {}
		self.decisions: dict[str, KycDecision] = {}
		self.evidence: dict[str, KycEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def open_profile(self, profile_id: str, tenant_id: str, subject_reference: str, legal_name: str, customer_type: str, country_code: str, consent_reference: str, metadata: dict[str, Any] | None = None, policy_attached: bool = True) -> dict[str, Any]:
		customer_type = normalize_code(customer_type)
		country_code = normalize_country(country_code)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "open_profile", "subject_present": bool(subject_reference), "legal_name_present": bool(legal_name), "customer_type_supported": customer_type in SUPPORTED_CUSTOMER_TYPES, "country_present": bool(country_code), "consent_recorded": bool(consent_reference)})
		if profile_id in self.profiles:
			raise ValueError(f"profile already exists: {profile_id}")
		profile = KycProfile(profile_id, tenant_id, subject_reference, legal_name, customer_type, country_code, consent_reference, metadata=dict(metadata or {}))
		self.profiles[profile_id] = profile
		self._audit(tenant_id, "kyc_profile_opened", profile_id)
		return profile.to_dict()

	def register_document(self, document_id: str, tenant_id: str, profile_id: str, document_type: str, token_reference: str, extracted_subject: str, confidence: float | int | str) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		document_type = normalize_code(document_type)
		confidence_value = normalize_confidence(confidence)
		minimum = float(get_capability_contract(tenant_id)["configuration"]["documents"]["minimum_confidence"])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_document", "profile_present": profile is not None, "document_type_supported": document_type in SUPPORTED_DOCUMENT_TYPES, "token_reference_present": bool(token_reference), "extracted_subject_present": bool(extracted_subject), "confidence_below_minimum": confidence_value < minimum})
		document = KycDocument(document_id, tenant_id, profile_id, document_type, token_reference, extracted_subject, confidence_value)
		self.documents[document_id] = document
		self._audit(tenant_id, "kyc_document_registered", document_id)
		return document.to_dict()

	def record_screening(self, screening_id: str, tenant_id: str, profile_id: str, sanctions_hit: bool = False, pep_hit: bool = False, watchlist_hit: bool = False, adverse_media_hit: bool = False, review_id: str = "") -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		screening_hit = any([sanctions_hit, pep_hit, watchlist_hit, adverse_media_hit])
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_screening", "profile_present": profile is not None, "screening_hit": screening_hit, "review_recorded": bool(review_id)})
		status = "reviewed" if screening_hit else "clear"
		screening = KycScreening(screening_id, tenant_id, profile_id, sanctions_hit, pep_hit, watchlist_hit, adverse_media_hit, review_id, status)
		self.screenings[screening_id] = screening
		self._audit(tenant_id, "kyc_screening_recorded", screening_id)
		return screening.to_dict()

	def score_risk(self, decision_id: str, tenant_id: str, profile_id: str, risk_score: int | str, review_id: str = "") -> dict[str, Any]:
		self._tenant_profile(profile_id, tenant_id)
		score = normalize_risk_score(risk_score)
		limits = get_capability_contract(tenant_id)["configuration"]["risk"]
		band = risk_band(score, int(limits["high_risk_threshold"]), int(limits["medium_risk_threshold"])) if 0 <= score <= 100 else "invalid"
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "score_risk", "risk_score_out_of_range": not 0 <= score <= 100, "high_risk": band == "high", "review_recorded": bool(review_id)})
		decision = KycDecision(decision_id, tenant_id, profile_id, band, score, review_id)
		self.decisions[decision_id] = decision
		self._audit(tenant_id, "kyc_risk_scored", decision_id)
		return decision.to_dict()

	def record_decision(self, decision_id: str, tenant_id: str, profile_id: str, decision: str, risk_score: int | str, review_id: str = "") -> dict[str, Any]:
		self._tenant_profile(profile_id, tenant_id)
		score = normalize_risk_score(risk_score)
		identity_document_present = self._has_document(tenant_id, profile_id, {"passport", "national_id", "driver_license", "resident_permit"})
		address_document_present = self._has_document(tenant_id, profile_id, {"utility_bill", "bank_statement", "business_registration"})
		screening_present = any(item.tenant_id == tenant_id and item.profile_id == profile_id for item in self.screenings.values())
		risk_present = any(item.tenant_id == tenant_id and item.profile_id == profile_id for item in self.decisions.values())
		open_review_flags = any(item.profile_id == profile_id and item.status != "clear" and not item.review_id for item in self.screenings.values())
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_decision", "identity_document_present": identity_document_present, "address_document_present": address_document_present, "screening_present": screening_present, "risk_present": risk_present, "open_review_flags": open_review_flags})
		record = KycDecision(decision_id, tenant_id, profile_id, normalize_code(decision), score, review_id, "verified" if normalize_code(decision) == "approve" else "recorded")
		self.decisions[decision_id] = record
		self.profiles[profile_id].status = "verified" if record.decision == "approve" else record.decision
		self._audit(tenant_id, "kyc_decision_recorded", decision_id)
		return record.to_dict()

	def register_kyc_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_kyc_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "kyc_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "kyc_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.kyc.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		profiles = [profile for profile in self.profiles.values() if profile.tenant_id == tenant_id]
		return {"tenant_id": tenant_id, "profile_count": len(profiles), "document_count": sum(1 for item in self.documents.values() if item.tenant_id == tenant_id), "screening_count": sum(1 for item in self.screenings.values() if item.tenant_id == tenant_id), "decision_count": sum(1 for item in self.decisions.values() if item.tenant_id == tenant_id), "verified_count": sum(1 for item in profiles if item.status == "verified"), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_profiles(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		profiles = self.profiles.values()
		if tenant_id is not None:
			profiles = [profile for profile in profiles if profile.tenant_id == tenant_id]
		return [profile.to_dict() for profile in sorted(profiles, key=lambda item: item.id)]

	def _tenant_profile_or_none(self, profile_id: str, tenant_id: str) -> KycProfile | None:
		profile = self.profiles.get(profile_id)
		if profile is None or profile.tenant_id != tenant_id:
			return None
		return profile

	def _tenant_profile(self, profile_id: str, tenant_id: str) -> KycProfile:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		if profile is None:
			raise KeyError(f"unknown KYC profile: {profile_id}")
		return profile

	def _has_document(self, tenant_id: str, profile_id: str, document_types: set[str]) -> bool:
		return any(item.tenant_id == tenant_id and item.profile_id == profile_id and item.document_type in document_types for item in self.documents.values())

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = KycEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "kyc_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "kyc_policy_denied")


FintechKycService = KnowYourCustomerService
