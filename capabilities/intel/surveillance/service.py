"""Executable service layer for APG Digital Surveillance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_ASSET_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SENSOR_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import MonitoredAsset, SurveillanceAgent, SurveillanceAlert, SurveillanceAuthority, SurveillanceDissemination, SurveillanceObservation, SurveillanceProgram, SurveillanceReferral, SurveillanceReview, SurveillanceRiskAssessment, SurveillanceSensor
	from .surveillance_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_ASSET_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SENSOR_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import MonitoredAsset, SurveillanceAgent, SurveillanceAlert, SurveillanceAuthority, SurveillanceDissemination, SurveillanceObservation, SurveillanceProgram, SurveillanceReferral, SurveillanceReview, SurveillanceRiskAssessment, SurveillanceSensor  # type: ignore
	from surveillance_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class DigitalSurveillanceService:
	"""Tenant-scoped digital surveillance runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], SurveillanceAuthority] = {}
		self.programs: dict[tuple[str, str], SurveillanceProgram] = {}
		self.assets: dict[tuple[str, str], MonitoredAsset] = {}
		self.sensors: dict[tuple[str, str], SurveillanceSensor] = {}
		self.observations: dict[tuple[str, str], SurveillanceObservation] = {}
		self.alerts: dict[tuple[str, str], SurveillanceAlert] = {}
		self.risks: dict[tuple[str, str], SurveillanceRiskAssessment] = {}
		self.referrals: dict[tuple[str, str], SurveillanceReferral] = {}
		self.disseminations: dict[tuple[str, str], SurveillanceDissemination] = {}
		self.reviews: dict[tuple[str, str], SurveillanceReview] = {}
		self.agents: dict[tuple[str, str], SurveillanceAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = SurveillanceAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "surveillance_authority_recorded", authority_id)
		return item.to_dict()

	def record_program(self, program_id: str, tenant_id: str, program_type: str, name: str, priority: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		program_type = normalize_code(program_type)
		priority = normalize_code(priority)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_program", "program_type_supported": program_type in SUPPORTED_PROGRAM_TYPES, "program_name_present": present(name), "priority_supported": priority in SUPPORTED_RISK_LEVELS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = SurveillanceProgram(program_id, tenant_id, program_type, name, priority, authority_id, evidence_reference)
		self.programs[self._tenant_key(tenant_id, program_id)] = item
		self._audit(tenant_id, "surveillance_program_recorded", program_id)
		return item.to_dict()

	def record_asset(self, asset_id: str, tenant_id: str, asset_type: str, asset_reference: str, owner_id: str, authority_id: str, privacy_review_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		asset_type = normalize_code(asset_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_asset", "asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES, "asset_reference_present": present(asset_reference), "owner_present": present(owner_id), "authority_present": authority is not None, "privacy_review_present": present(privacy_review_reference), "evidence_present": present(evidence_reference)})
		item = MonitoredAsset(asset_id, tenant_id, asset_type, asset_reference, owner_id, authority_id, privacy_review_reference, evidence_reference)
		self.assets[self._tenant_key(tenant_id, asset_id)] = item
		self._audit(tenant_id, "surveillance_asset_recorded", asset_id)
		return item.to_dict()

	def register_sensor(self, sensor_id: str, tenant_id: str, sensor_type: str, asset_id: str, sensor_reference: str, custodian_id: str, calibration_reference: str, evidence_reference: str) -> dict[str, Any]:
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		sensor_type = normalize_code(sensor_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_sensor", "sensor_type_supported": sensor_type in SUPPORTED_SENSOR_TYPES, "asset_present": asset is not None, "sensor_reference_present": present(sensor_reference), "custodian_present": present(custodian_id), "calibration_present": present(calibration_reference), "evidence_present": present(evidence_reference)})
		item = SurveillanceSensor(sensor_id, tenant_id, sensor_type, asset_id, sensor_reference, custodian_id, calibration_reference, evidence_reference)
		self.sensors[self._tenant_key(tenant_id, sensor_id)] = item
		self._audit(tenant_id, "surveillance_sensor_registered", sensor_id)
		return item.to_dict()

	def record_observation(self, observation_id: str, tenant_id: str, program_id: str, sensor_id: str, observation_type: str, observation_reference: str, content_fingerprint: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		sensor = self._tenant_sensor_or_none(sensor_id, tenant_id)
		asset = self._tenant_asset_or_none(sensor.asset_id, tenant_id) if sensor is not None else None
		observation_type = normalize_code(observation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_observation", "program_present": program is not None, "sensor_present": sensor is not None, "program_sensor_authority_match": program is not None and asset is not None and program.authority_id == asset.authority_id, "observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES, "observation_reference_present": present(observation_reference), "fingerprint_present": present(content_fingerprint), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = SurveillanceObservation(observation_id, tenant_id, program_id, sensor_id, observation_type, observation_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "surveillance_observation_recorded", observation_id)
		return item.to_dict()

	def record_alert(self, alert_id: str, tenant_id: str, observation_id: str, alert_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		alert_type = normalize_code(alert_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_alert", "observation_present": observation is not None, "alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = SurveillanceAlert(alert_id, tenant_id, observation_id, alert_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.alerts[self._tenant_key(tenant_id, alert_id)] = item
		self._audit(tenant_id, "surveillance_alert_recorded", alert_id)
		return item.to_dict()

	def record_risk(self, assessment_id: str, tenant_id: str, alert_id: str, assessment_type: str, risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_risk", "alert_present": alert is not None, "assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES, "risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = SurveillanceRiskAssessment(assessment_id, tenant_id, alert_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "surveillance_risk_recorded", assessment_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, assessment_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "assessment_present": assessment is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = SurveillanceReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "surveillance_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, assessment_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "assessment_present": assessment is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = SurveillanceDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "surveillance_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = SurveillanceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "surveillance_review_recorded", reference_id)
		return item.to_dict()

	def register_surveillance_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_surveillance_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = SurveillanceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "surveillance_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, covert_tracking_scope: bool = False, stalking_scope: bool = False, spyware_scope: bool = False, credential_capture_scope: bool = False, bypass_scope: bool = False, biometric_identification_scope: bool = False, exfiltration_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "surveillance_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "covert_tracking_scope": covert_tracking_scope, "stalking_scope": stalking_scope, "spyware_scope": spyware_scope, "credential_capture_scope": credential_capture_scope, "bypass_scope": bypass_scope, "biometric_identification_scope": biometric_identification_scope, "exfiltration_scope": exfiltration_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "surveillance_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.surveillance.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "program_count": self._count(self.programs, tenant_id), "asset_count": self._count(self.assets, tenant_id), "sensor_count": self._count(self.sensors, tenant_id), "observation_count": self._count(self.observations, tenant_id), "alert_count": self._count(self.alerts, tenant_id), "risk_count": self._count(self.risks, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> SurveillanceAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_program_or_none(self, item_id: str, tenant_id: str) -> SurveillanceProgram | None:
		return self.programs.get(self._tenant_key(tenant_id, item_id))

	def _tenant_asset_or_none(self, item_id: str, tenant_id: str) -> MonitoredAsset | None:
		return self.assets.get(self._tenant_key(tenant_id, item_id))

	def _tenant_sensor_or_none(self, item_id: str, tenant_id: str) -> SurveillanceSensor | None:
		return self.sensors.get(self._tenant_key(tenant_id, item_id))

	def _tenant_observation_or_none(self, item_id: str, tenant_id: str) -> SurveillanceObservation | None:
		return self.observations.get(self._tenant_key(tenant_id, item_id))

	def _tenant_alert_or_none(self, item_id: str, tenant_id: str) -> SurveillanceAlert | None:
		return self.alerts.get(self._tenant_key(tenant_id, item_id))

	def _tenant_risk_or_none(self, item_id: str, tenant_id: str) -> SurveillanceRiskAssessment | None:
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
		reasons = ", ".join(action.get("reason", action.get("rule", "surveillance_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "surveillance_policy_denied")


IntelSurveillanceService = DigitalSurveillanceService
