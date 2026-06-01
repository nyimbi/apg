"""Executable service layer for APG Radio Intelligence Listener."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BAND_TYPES, SUPPORTED_CLASSIFICATION_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_EVENT_TYPES, SUPPORTED_RECEIVER_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SESSION_TYPES, SUPPORTED_SIGNAL_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import RadioAgent, RadioAuthority, RadioBandPlan, RadioCollectionSession, RadioDissemination, RadioEventAssessment, RadioReceiver, RadioReferral, RadioReview, RadioSignalObservation, RadioTransmissionClassification
	from .radio_runtime import bounded_score, nonnegative_float, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_BAND_TYPES, SUPPORTED_CLASSIFICATION_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_EVENT_TYPES, SUPPORTED_RECEIVER_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SESSION_TYPES, SUPPORTED_SIGNAL_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import RadioAgent, RadioAuthority, RadioBandPlan, RadioCollectionSession, RadioDissemination, RadioEventAssessment, RadioReceiver, RadioReferral, RadioReview, RadioSignalObservation, RadioTransmissionClassification  # type: ignore
	from radio_runtime import bounded_score, nonnegative_float, normalize_code, positive_int, present  # type: ignore


class RadioIntelligenceListenerService:
	"""Tenant-scoped radio monitoring runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], RadioAuthority] = {}
		self.band_plans: dict[tuple[str, str], RadioBandPlan] = {}
		self.receivers: dict[tuple[str, str], RadioReceiver] = {}
		self.sessions: dict[tuple[str, str], RadioCollectionSession] = {}
		self.observations: dict[tuple[str, str], RadioSignalObservation] = {}
		self.classifications: dict[tuple[str, str], RadioTransmissionClassification] = {}
		self.events: dict[tuple[str, str], RadioEventAssessment] = {}
		self.referrals: dict[tuple[str, str], RadioReferral] = {}
		self.disseminations: dict[tuple[str, str], RadioDissemination] = {}
		self.reviews: dict[tuple[str, str], RadioReview] = {}
		self.agents: dict[tuple[str, str], RadioAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = RadioAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "radio_authority_recorded", authority_id)
		return item.to_dict()

	def record_band_plan(self, band_id: str, tenant_id: str, band_type: str, name: str, frequency_min_mhz: float, frequency_max_mhz: float, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		band_type = normalize_code(band_type)
		frequency_min = float(frequency_min_mhz) if nonnegative_float(frequency_min_mhz) else -1.0
		frequency_max = float(frequency_max_mhz) if nonnegative_float(frequency_max_mhz) else -1.0
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_band_plan", "band_type_supported": band_type in SUPPORTED_BAND_TYPES, "band_name_present": present(name), "frequency_min_valid": nonnegative_float(frequency_min_mhz), "frequency_max_valid": nonnegative_float(frequency_max_mhz), "frequency_range_valid": frequency_max >= frequency_min and frequency_min >= 0, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = RadioBandPlan(band_id, tenant_id, band_type, name, frequency_min, frequency_max, authority_id, evidence_reference)
		self.band_plans[self._tenant_key(tenant_id, band_id)] = item
		self._audit(tenant_id, "radio_band_plan_recorded", band_id)
		return item.to_dict()

	def register_receiver(self, receiver_id: str, tenant_id: str, receiver_type: str, site_reference: str, custodian_id: str, authority_id: str, calibration_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		receiver_type = normalize_code(receiver_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_receiver", "receiver_type_supported": receiver_type in SUPPORTED_RECEIVER_TYPES, "site_reference_present": present(site_reference), "custodian_present": present(custodian_id), "authority_present": authority is not None, "calibration_present": present(calibration_reference), "evidence_present": present(evidence_reference)})
		item = RadioReceiver(receiver_id, tenant_id, receiver_type, site_reference, custodian_id, authority_id, calibration_reference, evidence_reference)
		self.receivers[self._tenant_key(tenant_id, receiver_id)] = item
		self._audit(tenant_id, "radio_receiver_registered", receiver_id)
		return item.to_dict()

	def record_session(self, session_id: str, tenant_id: str, band_id: str, receiver_id: str, session_type: str, started_at: str, ended_at: str, collection_plan_reference: str, evidence_reference: str) -> dict[str, Any]:
		band = self._tenant_band_or_none(band_id, tenant_id)
		receiver = self._tenant_receiver_or_none(receiver_id, tenant_id)
		session_type = normalize_code(session_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_session", "band_present": band is not None, "receiver_present": receiver is not None, "band_receiver_authority_match": band is not None and receiver is not None and band.authority_id == receiver.authority_id, "session_type_supported": session_type in SUPPORTED_SESSION_TYPES, "started_at_present": present(started_at), "collection_plan_present": present(collection_plan_reference), "evidence_present": present(evidence_reference)})
		item = RadioCollectionSession(session_id, tenant_id, band_id, receiver_id, session_type, started_at, ended_at, collection_plan_reference, evidence_reference)
		self.sessions[self._tenant_key(tenant_id, session_id)] = item
		self._audit(tenant_id, "radio_session_recorded", session_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, session_id: str, frequency_mhz: float, signal_type: str, signal_fingerprint: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		session = self._tenant_session_or_none(session_id, tenant_id)
		band = self._tenant_band_or_none(session.band_id, tenant_id) if session is not None else None
		frequency = float(frequency_mhz) if nonnegative_float(frequency_mhz) else -1.0
		signal_type = normalize_code(signal_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "session_present": session is not None, "frequency_valid": nonnegative_float(frequency_mhz), "frequency_in_band": band is not None and band.frequency_min_mhz <= frequency <= band.frequency_max_mhz, "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES, "fingerprint_present": present(signal_fingerprint), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = RadioSignalObservation(observation_id, tenant_id, session_id, frequency, signal_type, signal_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "radio_observation_recorded", observation_id)
		return item.to_dict()

	def record_classification(self, classification_id: str, tenant_id: str, observation_id: str, classification_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		classification_type = normalize_code(classification_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_classification", "observation_present": observation is not None, "classification_type_supported": classification_type in SUPPORTED_CLASSIFICATION_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = RadioTransmissionClassification(classification_id, tenant_id, observation_id, classification_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.classifications[self._tenant_key(tenant_id, classification_id)] = item
		self._audit(tenant_id, "radio_classification_recorded", classification_id)
		return item.to_dict()

	def record_event(self, assessment_id: str, tenant_id: str, classification_id: str, event_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		classification = self._tenant_classification_or_none(classification_id, tenant_id)
		event_type = normalize_code(event_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_event", "classification_present": classification is not None, "event_type_supported": event_type in SUPPORTED_EVENT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = RadioEventAssessment(assessment_id, tenant_id, classification_id, event_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.events[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "radio_event_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_event_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = RadioReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "radio_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_event_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = RadioDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "radio_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = RadioReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "radio_review_recorded", reference_id)
		return item.to_dict()

	def register_radio_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_radio_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = RadioAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "radio_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, transmit_scope: bool = False, unauthorized_interception_scope: bool = False, decryption_scope: bool = False, jamming_scope: bool = False, spoofing_scope: bool = False, interference_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "radio_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "transmit_scope": transmit_scope, "unauthorized_interception_scope": unauthorized_interception_scope, "decryption_scope": decryption_scope, "jamming_scope": jamming_scope, "spoofing_scope": spoofing_scope, "interference_scope": interference_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "radio_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.radio.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "band_plan_count": self._count(self.band_plans, tenant_id), "receiver_count": self._count(self.receivers, tenant_id), "session_count": self._count(self.sessions, tenant_id), "observation_count": self._count(self.observations, tenant_id), "classification_count": self._count(self.classifications, tenant_id), "event_count": self._count(self.events, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> RadioAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_band_or_none(self, item_id: str, tenant_id: str) -> RadioBandPlan | None:
		return self.band_plans.get(self._tenant_key(tenant_id, item_id))

	def _tenant_receiver_or_none(self, item_id: str, tenant_id: str) -> RadioReceiver | None:
		return self.receivers.get(self._tenant_key(tenant_id, item_id))

	def _tenant_session_or_none(self, item_id: str, tenant_id: str) -> RadioCollectionSession | None:
		return self.sessions.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> RadioSignalObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_classification_or_none(self, item_id: str, tenant_id: str) -> RadioTransmissionClassification | None:
		return self.classifications.get(self._tenant_key(tenant_id, item_id))

	def _tenant_event_or_none(self, item_id: str, tenant_id: str) -> RadioEventAssessment | None:
		return self.events.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "radio_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "radio_policy_denied")


IntelRadioService = RadioIntelligenceListenerService
