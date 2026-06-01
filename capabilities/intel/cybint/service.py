"""Executable service layer for APG Cyber Intelligence."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_ENRICHMENT_TYPES, SUPPORTED_INDICATOR_TYPES, SUPPORTED_PROFILE_TYPES, SUPPORTED_RESPONSE_PRIORITIES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SEVERITIES, SUPPORTED_TLP, evaluate_capability_rules, get_capability_contract
	from .cybint_runtime import bounded_score, normalize_code, positive_int, present
	from .models import CYBINTAgent, CYBINTDissemination, CYBINTReview, CyberAuthority, CyberRiskAssessment, Enrichment, IncidentLink, Indicator, Sighting, ThreatProfile
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_ENRICHMENT_TYPES, SUPPORTED_INDICATOR_TYPES, SUPPORTED_PROFILE_TYPES, SUPPORTED_RESPONSE_PRIORITIES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SEVERITIES, SUPPORTED_TLP, evaluate_capability_rules, get_capability_contract  # type: ignore
	from cybint_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import CYBINTAgent, CYBINTDissemination, CYBINTReview, CyberAuthority, CyberRiskAssessment, Enrichment, IncidentLink, Indicator, Sighting, ThreatProfile  # type: ignore


class CyberIntelligenceService:
	"""Tenant-scoped defensive cyber-intelligence runtime for generated APG apps."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], CyberAuthority] = {}
		self.indicators: dict[tuple[str, str], Indicator] = {}
		self.sightings: dict[tuple[str, str], Sighting] = {}
		self.enrichments: dict[tuple[str, str], Enrichment] = {}
		self.profiles: dict[tuple[str, str], ThreatProfile] = {}
		self.risks: dict[tuple[str, str], CyberRiskAssessment] = {}
		self.incidents: dict[tuple[str, str], IncidentLink] = {}
		self.disseminations: dict[tuple[str, str], CYBINTDissemination] = {}
		self.reviews: dict[tuple[str, str], CYBINTReview] = {}
		self.agents: dict[tuple[str, str], CYBINTAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = CyberAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "cybint_authority_recorded", authority_id)
		return item.to_dict()

	def record_indicator(self, indicator_id: str, tenant_id: str, indicator_type: str, indicator_value: str, tlp: str, confidence_score: float, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		tlp = normalize_code(tlp)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_indicator", "indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES, "indicator_value_present": present(indicator_value), "tlp_supported": tlp in SUPPORTED_TLP, "confidence_valid": bounded_score(confidence_score), "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = Indicator(indicator_id, tenant_id, indicator_type, indicator_value, tlp, float(confidence_score), authority_id, evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "cybint_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_sighting(self, sighting_id: str, tenant_id: str, indicator_id: str, source_reference: str, observed_at: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_sighting", "indicator_present": indicator is not None, "source_reference_present": present(source_reference), "observed_at_present": present(observed_at), "severity_supported": severity in SUPPORTED_SEVERITIES, "evidence_present": present(evidence_reference)})
		item = Sighting(sighting_id, tenant_id, indicator_id, source_reference, observed_at, severity, evidence_reference)
		self.sightings[self._tenant_key(tenant_id, sighting_id)] = item
		self._audit(tenant_id, "cybint_sighting_recorded", sighting_id)
		return item.to_dict()

	def record_enrichment(self, enrichment_id: str, tenant_id: str, indicator_id: str, enrichment_type: str, provider_reference: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		enrichment_type = normalize_code(enrichment_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_enrichment", "indicator_present": indicator is not None, "enrichment_type_supported": enrichment_type in SUPPORTED_ENRICHMENT_TYPES, "provider_present": present(provider_reference), "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = Enrichment(enrichment_id, tenant_id, indicator_id, enrichment_type, provider_reference, float(confidence_score), analyst_id, evidence_reference)
		self.enrichments[self._tenant_key(tenant_id, enrichment_id)] = item
		self._audit(tenant_id, "cybint_enrichment_recorded", enrichment_id)
		return item.to_dict()

	def record_profile(self, profile_id: str, tenant_id: str, profile_type: str, name: str, classification: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		profile_type = normalize_code(profile_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_profile", "profile_type_supported": profile_type in SUPPORTED_PROFILE_TYPES, "name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ThreatProfile(profile_id, tenant_id, profile_type, name, classification, float(confidence_score), analyst_id, evidence_reference)
		self.profiles[self._tenant_key(tenant_id, profile_id)] = item
		self._audit(tenant_id, "cybint_profile_recorded", profile_id)
		return item.to_dict()

	def record_risk(self, assessment_id: str, tenant_id: str, indicator_id: str, profile_id: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk", "indicator_present": indicator is not None, "profile_present": profile is not None, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = CyberRiskAssessment(assessment_id, tenant_id, indicator_id, profile_id, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "cybint_risk_recorded", assessment_id)
		return item.to_dict()

	def record_incident_link(self, link_id: str, tenant_id: str, assessment_id: str, incident_reference: str, response_priority: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		response_priority = normalize_code(response_priority)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_incident_link", "assessment_present": assessment is not None, "incident_reference_present": present(incident_reference), "response_priority_supported": response_priority in SUPPORTED_RESPONSE_PRIORITIES, "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = IncidentLink(link_id, tenant_id, assessment_id, incident_reference, response_priority, owner_id, evidence_reference)
		self.incidents[self._tenant_key(tenant_id, link_id)] = item
		self._audit(tenant_id, "cybint_incident_link_recorded", link_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = CYBINTDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "cybint_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = CYBINTReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "cybint_review_recorded", review_id)
		return item.to_dict()

	def register_cybint_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_cybint_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = CYBINTAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "cybint_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, offensive_or_exploit_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "cybint_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "offensive_or_exploit_scope": offensive_or_exploit_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope, "offensive_or_exploit_scope": offensive_or_exploit_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "cybint_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.cybint.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "indicator_count": self._count(self.indicators, tenant_id), "sighting_count": self._count(self.sightings, tenant_id), "enrichment_count": self._count(self.enrichments, tenant_id), "profile_count": self._count(self.profiles, tenant_id), "risk_count": self._count(self.risks, tenant_id), "incident_count": self._count(self.incidents, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> CyberAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_indicator_or_none(self, item_id: str, tenant_id: str) -> Indicator | None:
		return self.indicators.get(self._tenant_key(tenant_id, item_id))

	def _tenant_profile_or_none(self, item_id: str, tenant_id: str) -> ThreatProfile | None:
		return self.profiles.get(self._tenant_key(tenant_id, item_id))

	def _tenant_risk_or_none(self, item_id: str, tenant_id: str) -> CyberRiskAssessment | None:
		return self.risks.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "cybint_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "cybint_policy_denied")


IntelCYBINTService = CyberIntelligenceService
