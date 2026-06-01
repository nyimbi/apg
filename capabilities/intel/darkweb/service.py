"""Executable service layer for APG Dark Web Monitoring."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_NETWORK_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract
	from .darkweb_runtime import bounded_score, normalize_code, positive_int, present
	from .models import DarkWebAgent, DarkWebDissemination, DarkWebObservation, DarkWebReferral, DarkWebReview, ExposureIndicator, HiddenServiceSource, MarketplaceRiskAssessment, MonitoringAuthority, MonitoringProgram, ThreatActorAssessment
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_INDICATOR_TYPES, SUPPORTED_NETWORK_TYPES, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SOURCE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from darkweb_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from models import DarkWebAgent, DarkWebDissemination, DarkWebObservation, DarkWebReferral, DarkWebReview, ExposureIndicator, HiddenServiceSource, MarketplaceRiskAssessment, MonitoringAuthority, MonitoringProgram, ThreatActorAssessment  # type: ignore


class DarkWebMonitoringService:
	"""Tenant-scoped dark-web monitoring runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], MonitoringAuthority] = {}
		self.programs: dict[tuple[str, str], MonitoringProgram] = {}
		self.sources: dict[tuple[str, str], HiddenServiceSource] = {}
		self.observations: dict[tuple[str, str], DarkWebObservation] = {}
		self.indicators: dict[tuple[str, str], ExposureIndicator] = {}
		self.marketplace_risks: dict[tuple[str, str], MarketplaceRiskAssessment] = {}
		self.threat_actors: dict[tuple[str, str], ThreatActorAssessment] = {}
		self.referrals: dict[tuple[str, str], DarkWebReferral] = {}
		self.disseminations: dict[tuple[str, str], DarkWebDissemination] = {}
		self.reviews: dict[tuple[str, str], DarkWebReview] = {}
		self.agents: dict[tuple[str, str], DarkWebAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = MonitoringAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "darkweb_authority_recorded", authority_id)
		return item.to_dict()

	def record_program(self, program_id: str, tenant_id: str, program_type: str, name: str, priority: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		program_type = normalize_code(program_type)
		priority = normalize_code(priority)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_program", "program_type_supported": program_type in SUPPORTED_PROGRAM_TYPES, "program_name_present": present(name), "priority_supported": priority in SUPPORTED_RISK_LEVELS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = MonitoringProgram(program_id, tenant_id, program_type, name, priority, authority_id, evidence_reference)
		self.programs[self._tenant_key(tenant_id, program_id)] = item
		self._audit(tenant_id, "darkweb_program_recorded", program_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, network_type: str, source_reference: str, custodian_id: str, authority_id: str, access_review_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		network_type = normalize_code(network_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "network_type_supported": network_type in SUPPORTED_NETWORK_TYPES, "source_reference_present": present(source_reference), "custodian_present": present(custodian_id), "authority_present": authority is not None, "access_review_present": present(access_review_reference), "evidence_present": present(evidence_reference)})
		item = HiddenServiceSource(source_id, tenant_id, source_type, network_type, source_reference, custodian_id, authority_id, access_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "darkweb_source_registered", source_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, program_id: str, source_id: str, observation_type: str, observation_reference: str, content_fingerprint: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		observation_type = normalize_code(observation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "program_present": program is not None, "source_present": source is not None, "program_source_authority_match": program is not None and source is not None and program.authority_id == source.authority_id, "observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES, "observation_reference_present": present(observation_reference), "fingerprint_present": present(content_fingerprint), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = DarkWebObservation(observation_id, tenant_id, program_id, source_id, observation_type, observation_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "darkweb_observation_recorded", observation_id)
		return item.to_dict()

	def record_indicator(self, indicator_id: str, tenant_id: str, observation_id: str, indicator_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		indicator_type = normalize_code(indicator_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_indicator", "observation_present": observation is not None, "indicator_type_supported": indicator_type in SUPPORTED_INDICATOR_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ExposureIndicator(indicator_id, tenant_id, observation_id, indicator_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.indicators[self._tenant_key(tenant_id, indicator_id)] = item
		self._audit(tenant_id, "darkweb_indicator_recorded", indicator_id)
		return item.to_dict()

	def record_marketplace_risk(self, assessment_id: str, tenant_id: str, indicator_id: str, assessment_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_marketplace_risk", "indicator_present": indicator is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = MarketplaceRiskAssessment(assessment_id, tenant_id, indicator_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.marketplace_risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "darkweb_marketplace_risk_recorded", assessment_id)
		return item.to_dict()

	def record_threat_actor(self, assessment_id: str, tenant_id: str, indicator_id: str, actor_reference: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		indicator = self._tenant_indicator_or_none(indicator_id, tenant_id)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_threat_actor", "indicator_present": indicator is not None, "actor_reference_present": present(actor_reference), "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = ThreatActorAssessment(assessment_id, tenant_id, indicator_id, actor_reference, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.threat_actors[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "darkweb_threat_actor_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = DarkWebReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "darkweb_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._assessment_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = DarkWebDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "darkweb_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = DarkWebReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "darkweb_review_recorded", reference_id)
		return item.to_dict()

	def register_darkweb_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_darkweb_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = DarkWebAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "darkweb_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, credential_use_scope: bool = False, exploit_procurement_scope: bool = False, contraband_transaction_scope: bool = False, evasion_scope: bool = False, doxxing_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "darkweb_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "credential_use_scope": credential_use_scope, "exploit_procurement_scope": exploit_procurement_scope, "contraband_transaction_scope": contraband_transaction_scope, "evasion_scope": evasion_scope, "doxxing_scope": doxxing_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "darkweb_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.darkweb.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "program_count": self._count(self.programs, tenant_id), "source_count": self._count(self.sources, tenant_id), "observation_count": self._count(self.observations, tenant_id), "indicator_count": self._count(self.indicators, tenant_id), "marketplace_risk_count": self._count(self.marketplace_risks, tenant_id), "threat_actor_count": self._count(self.threat_actors, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> MonitoringAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_program_or_none(self, item_id: str, tenant_id: str) -> MonitoringProgram | None:
		return self.programs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> HiddenServiceSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> DarkWebObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_indicator_or_none(self, item_id: str, tenant_id: str) -> ExposureIndicator | None:
		return self.indicators.get(self._tenant_key(tenant_id, item_id))

	def _assessment_or_none(self, item_id: str, tenant_id: str) -> MarketplaceRiskAssessment | ThreatActorAssessment | None:
		return self.marketplace_risks.get(self._tenant_key(tenant_id, item_id)) or self.threat_actors.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "darkweb_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "darkweb_policy_denied")


IntelDarkWebService = DarkWebMonitoringService
