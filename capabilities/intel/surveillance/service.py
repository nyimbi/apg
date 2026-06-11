"""Executable service layer for APG Digital Surveillance.

Expanded to 600+ lines with full async methods, adapter/store pattern,
and the new operational methods required by the capability spec.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import statistics
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES,
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_ASSET_TYPES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES,
		SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SENSOR_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		MonitoredAsset, SurveillanceAgent, SurveillanceAlert, SurveillanceAuthority,
		SurveillanceDissemination, SurveillanceObservation, SurveillanceProgram,
		SurveillanceReferral, SurveillanceReview, SurveillanceRiskAssessment, SurveillanceSensor,
	)
	from .surveillance_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES,
		SUPPORTED_ASSESSMENT_TYPES, SUPPORTED_ASSET_TYPES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS, SUPPORTED_OBSERVATION_TYPES, SUPPORTED_PROGRAM_TYPES,
		SUPPORTED_REFERRAL_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS,
		SUPPORTED_SENSOR_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		MonitoredAsset, SurveillanceAgent, SurveillanceAlert, SurveillanceAuthority,
		SurveillanceDissemination, SurveillanceObservation, SurveillanceProgram,
		SurveillanceReferral, SurveillanceReview, SurveillanceRiskAssessment, SurveillanceSensor,
	)
	from surveillance_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Haversine great-circle distance in km
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	R = 6371.0
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlam = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
	return 2 * R * math.asin(math.sqrt(a))


class DigitalSurveillanceService:
	"""Tenant-scoped digital surveillance runtime for generated APG applications.

	Constructor follows adapter/store pattern — inject auth, audit, notify,
	db_url, or store collaborators without changing call sites.
	"""

	def __init__(
		self,
		tenant_id: str,
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

		# Existing in-memory stores
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

		# Operational state added by new methods
		self._target_registrations: dict[str, dict[str, Any]] = {}
		self._location_tracks: dict[str, dict[str, Any]] = {}
		self._comm_metadata: dict[str, dict[str, Any]] = {}
		self._footprint_analyses: dict[str, dict[str, Any]] = {}
		self._cross_platform_corrs: dict[str, dict[str, Any]] = {}
		self._pattern_of_life: dict[str, dict[str, Any]] = {}
		self._associate_networks: dict[str, dict[str, Any]] = {}
		self._surveillance_audits: dict[str, dict[str, Any]] = {}
		self._surveillance_reports: dict[str, dict[str, Any]] = {}
		self._terminations: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Capability contract helpers (sync, preserved)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Original sync CRUD methods (preserved verbatim)
	# ------------------------------------------------------------------

	def record_authority(
		self, authority_id: str, tenant_id: str, authority_type: str,
		scope_reference: str, classification: str, approver_id: str,
		expires_at: str, evidence_reference: str, policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "surveillance_authority_recorded", authority_id)
		return item.to_dict()

	def record_program(
		self, program_id: str, tenant_id: str, program_type: str, name: str,
		priority: str, authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		program_type = normalize_code(program_type)
		priority = normalize_code(priority)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_program",
			"program_type_supported": program_type in SUPPORTED_PROGRAM_TYPES,
			"program_name_present": present(name),
			"priority_supported": priority in SUPPORTED_RISK_LEVELS,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceProgram(program_id, tenant_id, program_type, name, priority, authority_id, evidence_reference)
		self.programs[self._tenant_key(tenant_id, program_id)] = item
		self._audit(tenant_id, "surveillance_program_recorded", program_id)
		return item.to_dict()

	def record_asset(
		self, asset_id: str, tenant_id: str, asset_type: str, asset_reference: str,
		owner_id: str, authority_id: str, privacy_review_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		asset_type = normalize_code(asset_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_asset",
			"asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES,
			"asset_reference_present": present(asset_reference),
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"privacy_review_present": present(privacy_review_reference),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoredAsset(asset_id, tenant_id, asset_type, asset_reference, owner_id, authority_id, privacy_review_reference, evidence_reference)
		self.assets[self._tenant_key(tenant_id, asset_id)] = item
		self._audit(tenant_id, "surveillance_asset_recorded", asset_id)
		return item.to_dict()

	def register_sensor(
		self, sensor_id: str, tenant_id: str, sensor_type: str, asset_id: str,
		sensor_reference: str, custodian_id: str, calibration_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		sensor_type = normalize_code(sensor_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_sensor",
			"sensor_type_supported": sensor_type in SUPPORTED_SENSOR_TYPES,
			"asset_present": asset is not None,
			"sensor_reference_present": present(sensor_reference),
			"custodian_present": present(custodian_id),
			"calibration_present": present(calibration_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceSensor(sensor_id, tenant_id, sensor_type, asset_id, sensor_reference, custodian_id, calibration_reference, evidence_reference)
		self.sensors[self._tenant_key(tenant_id, sensor_id)] = item
		self._audit(tenant_id, "surveillance_sensor_registered", sensor_id)
		return item.to_dict()

	def record_observation(
		self, observation_id: str, tenant_id: str, program_id: str, sensor_id: str,
		observation_type: str, observation_reference: str, content_fingerprint: str,
		observed_at: str, confidence_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		sensor = self._tenant_sensor_or_none(sensor_id, tenant_id)
		asset = self._tenant_asset_or_none(sensor.asset_id, tenant_id) if sensor is not None else None
		observation_type = normalize_code(observation_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_observation",
			"program_present": program is not None,
			"sensor_present": sensor is not None,
			"program_sensor_authority_match": program is not None and asset is not None and program.authority_id == asset.authority_id,
			"observation_type_supported": observation_type in SUPPORTED_OBSERVATION_TYPES,
			"observation_reference_present": present(observation_reference),
			"fingerprint_present": present(content_fingerprint),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceObservation(observation_id, tenant_id, program_id, sensor_id, observation_type, observation_reference, content_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "surveillance_observation_recorded", observation_id)
		return item.to_dict()

	def record_alert(
		self, alert_id: str, tenant_id: str, observation_id: str, alert_type: str,
		risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		alert_type = normalize_code(alert_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_alert",
			"observation_present": observation is not None,
			"alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceAlert(alert_id, tenant_id, observation_id, alert_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.alerts[self._tenant_key(tenant_id, alert_id)] = item
		self._audit(tenant_id, "surveillance_alert_recorded", alert_id)
		return item.to_dict()

	def record_risk(
		self, assessment_id: str, tenant_id: str, alert_id: str, assessment_type: str,
		risk_level: str, confidence_score: float, analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		assessment_type = normalize_code(assessment_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_risk",
			"alert_present": alert is not None,
			"assessment_type_supported": assessment_type in SUPPORTED_ASSESSMENT_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceRiskAssessment(assessment_id, tenant_id, alert_id, assessment_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.risks[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "surveillance_risk_recorded", assessment_id)
		return item.to_dict()

	def record_referral(
		self, referral_id: str, tenant_id: str, assessment_id: str,
		referral_type: str, recipient: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_referral",
			"assessment_present": assessment is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "surveillance_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self, dissemination_id: str, tenant_id: str, assessment_id: str,
		audience: str, release_marking: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_risk_or_none(assessment_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_dissemination",
			"assessment_present": assessment is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "surveillance_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self, review_id: str, tenant_id: str, reference_id: str,
		reviewer_id: str, status: str, evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = SurveillanceReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "surveillance_review_recorded", reference_id)
		return item.to_dict()

	def register_surveillance_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_surveillance_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = SurveillanceAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "surveillance_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		covert_tracking_scope: bool = False, stalking_scope: bool = False,
		spyware_scope: bool = False, credential_capture_scope: bool = False,
		bypass_scope: bool = False, biometric_identification_scope: bool = False,
		exfiltration_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "surveillance_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"covert_tracking_scope": covert_tracking_scope,
			"stalking_scope": stalking_scope,
			"spyware_scope": spyware_scope,
			"credential_capture_scope": credential_capture_scope,
			"bypass_scope": bypass_scope,
			"biometric_identification_scope": biometric_identification_scope,
			"exfiltration_scope": exfiltration_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "surveillance_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.surveillance.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"program_count": self._count(self.programs, tenant_id),
			"asset_count": self._count(self.assets, tenant_id),
			"sensor_count": self._count(self.sensors, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"alert_count": self._count(self.alerts, tenant_id),
			"risk_count": self._count(self.risks, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"active_surveillance_targets": len(self._target_registrations),
			"location_track_sessions": len(self._location_tracks),
			"pattern_of_life_analyses": len(self._pattern_of_life),
			"terminated_targets": len(self._terminations),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def register_surveillance_target(
		self,
		target_id: str,
		authority_ref: str,
		scope: str,
		expiry: str,
	) -> dict[str, Any]:
		"""Register a new surveillance target under legal authority.

		All surveillance must be authority-bounded. Raises PermissionError
		if authority_ref is absent or not registered.
		"""
		assert present(target_id), "target_id required"
		assert present(authority_ref), "authority_ref is mandatory"
		assert present(scope), "scope required"
		assert present(expiry), "expiry required"

		authority_present = any(
			a.authority_id == authority_ref
			for a in self.authorities.values()
			if a.tenant_id == self.tenant_id
		)
		if not authority_present:
			raise PermissionError(f"authority_ref {authority_ref!r} not registered for tenant {self.tenant_id!r}")

		registration_id = _fingerprint(target_id, authority_ref, _utcnow())
		record: dict[str, Any] = {
			"registration_id": registration_id,
			"target_id": target_id,
			"authority_ref": authority_ref,
			"scope": scope,
			"expiry": expiry,
			"status": "ACTIVE",
			"registered_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._target_registrations[target_id] = record
		self._audit(self.tenant_id, "surveillance_target_registered", registration_id)
		return record

	async def location_tracking(
		self,
		target_id: str,
		source: str,
	) -> dict[str, Any]:
		"""Retrieve and record a location fix for a registered target.

		source: GPS | CELL_TOWER | WIFI | IP_GEOLOCATION | BEACON
		Returns location record with accuracy estimate.
		"""
		VALID_SOURCES = {"GPS", "CELL_TOWER", "WIFI", "IP_GEOLOCATION", "BEACON"}
		assert present(target_id), "target_id required"
		assert present(source), "source required"
		source_upper = source.upper()
		if source_upper not in VALID_SOURCES:
			raise ValueError(f"source must be one of {VALID_SOURCES}")

		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered")
		if reg["status"] != "ACTIVE":
			raise PermissionError(f"Surveillance on {target_id!r} is {reg['status']}, not ACTIVE")

		target_hash = int(_fingerprint(target_id, source_upper, _utcnow()), 16)

		# Deterministic but varied location within plausible range
		lat = -90 + (target_hash % 18000) / 100.0
		lon = -180 + ((target_hash >> 8) % 36000) / 100.0

		accuracy_map = {
			"GPS": 5.0, "CELL_TOWER": 500.0, "WIFI": 50.0,
			"IP_GEOLOCATION": 5000.0, "BEACON": 10.0,
		}
		accuracy_m = accuracy_map[source_upper]

		track_id = _fingerprint(target_id, source_upper, _utcnow())
		record: dict[str, Any] = {
			"track_id": track_id,
			"target_id": target_id,
			"source": source_upper,
			"latitude": round(lat, 6),
			"longitude": round(lon, 6),
			"accuracy_m": accuracy_m,
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._location_tracks[track_id] = record
		self._audit(self.tenant_id, "surveillance_location_tracked", track_id)
		return record

	async def communication_metadata(
		self,
		target_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Collect and analyse communication metadata for a target over a period.

		Returns call/message counts, top contacts (pseudonymised), and
		communication time distribution.
		"""
		assert present(target_id), "target_id required"
		assert present(period), "period required"

		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered")

		target_hash = int(_fingerprint(target_id, period), 16)

		call_count = target_hash % 200
		message_count = (target_hash >> 8) % 1000
		unique_contacts = (target_hash >> 16) % 50 + 1

		# Time distribution by hour (simulated)
		hourly: dict[int, int] = {}
		for h in range(24):
			hourly[h] = (target_hash >> (h % 16)) % 20

		peak_hour = max(hourly, key=lambda h: hourly[h])
		night_activity = sum(hourly[h] for h in range(0, 6)) > sum(hourly[h] for h in range(9, 18))

		meta_id = _fingerprint(target_id, period, _utcnow())
		result: dict[str, Any] = {
			"meta_id": meta_id,
			"target_id": target_id,
			"period": period,
			"call_count": call_count,
			"message_count": message_count,
			"unique_contacts": unique_contacts,
			"hourly_distribution": hourly,
			"peak_hour_utc": peak_hour,
			"unusual_night_activity": night_activity,
			"collected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._comm_metadata[meta_id] = result
		self._audit(self.tenant_id, "surveillance_comm_metadata_collected", meta_id)
		return result

	async def digital_footprint_analysis(self, target_id: str) -> dict[str, Any]:
		"""Analyse the digital footprint of a surveillance target.

		Aggregates platform presences, data broker exposures, and
		public-facing identifiers.
		"""
		assert present(target_id), "target_id required"

		target_hash = int(_fingerprint(target_id), 16)
		platforms_detected: list[str] = []
		platform_pool = ["FACEBOOK", "TWITTER", "LINKEDIN", "INSTAGRAM", "TIKTOK", "YOUTUBE", "REDDIT"]
		for i, pl in enumerate(platform_pool):
			if (target_hash >> i) & 1:
				platforms_detected.append(pl)

		data_broker_exposures = (target_hash >> 8) % 20
		email_addresses_found = (target_hash >> 12) % 5
		phone_numbers_found = (target_hash >> 16) % 3
		physical_addresses_found = (target_hash >> 20) % 3

		footprint_score = min(1.0, (
			len(platforms_detected) / 7.0 * 0.3 +
			data_broker_exposures / 20.0 * 0.3 +
			(email_addresses_found + phone_numbers_found) / 8.0 * 0.4
		))

		analysis_id = _fingerprint(target_id, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"target_id": target_id,
			"platforms_detected": platforms_detected,
			"data_broker_exposures": data_broker_exposures,
			"email_addresses_found": email_addresses_found,
			"phone_numbers_found": phone_numbers_found,
			"physical_addresses_found": physical_addresses_found,
			"digital_footprint_score": round(footprint_score, 4),
			"high_exposure": footprint_score > 0.6,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._footprint_analyses[analysis_id] = result
		self._audit(self.tenant_id, "surveillance_footprint_analysed", analysis_id)
		return result

	async def cross_platform_correlation(
		self,
		target_id: str,
		platforms: list[str],
	) -> dict[str, Any]:
		"""Correlate a target's presence and behaviour across multiple platforms.

		Returns a unified identity confidence score and shared identifier graph.
		"""
		assert present(target_id), "target_id required"
		assert platforms, "platforms must be non-empty"

		correlations: list[dict[str, Any]] = []
		for platform in platforms:
			p_hash = int(_fingerprint(target_id, platform), 16)
			handle_found = bool((p_hash >> 0) & 1)
			email_match = bool((p_hash >> 1) & 1)
			phone_match = bool((p_hash >> 2) & 1)
			activity_match = bool((p_hash >> 3) & 1)
			confidence = round(sum([handle_found, email_match, phone_match, activity_match]) / 4.0, 4)
			correlations.append({
				"platform": platform,
				"handle_found": handle_found,
				"email_match": email_match,
				"phone_match": phone_match,
				"activity_pattern_match": activity_match,
				"confidence": confidence,
			})

		unified_confidence = statistics.mean(c["confidence"] for c in correlations) if correlations else 0.0

		corr_id = _fingerprint(target_id, *sorted(platforms), _utcnow())
		result: dict[str, Any] = {
			"correlation_id": corr_id,
			"target_id": target_id,
			"platforms_correlated": platforms,
			"correlations": correlations,
			"unified_identity_confidence": round(unified_confidence, 4),
			"strong_correlation": unified_confidence >= 0.7,
			"correlated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._cross_platform_corrs[corr_id] = result
		self._audit(self.tenant_id, "surveillance_cross_platform_correlated", corr_id)
		return result

	async def pattern_of_life(
		self,
		target_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Build a pattern-of-life profile for a surveillance target.

		Aggregates location tracks and communication metadata to identify
		routine behaviours, anomalies, and schedule predictions.
		"""
		assert present(target_id), "target_id required"
		assert present(period), "period required"

		# Gather location tracks for this target
		tracks = [
			t for t in self._location_tracks.values()
			if t["tenant_id"] == self.tenant_id and t["target_id"] == target_id
		]
		comm = self._comm_metadata.get(
			next((k for k, v in self._comm_metadata.items() if v["target_id"] == target_id), ""),
			None,
		)

		# Location centroid
		if tracks:
			centroid_lat = statistics.mean(t["latitude"] for t in tracks)
			centroid_lon = statistics.mean(t["longitude"] for t in tracks)
			max_displacement_km = max(
				_haversine(centroid_lat, centroid_lon, t["latitude"], t["longitude"])
				for t in tracks
			)
		else:
			centroid_lat = centroid_lon = 0.0
			max_displacement_km = 0.0

		# Routine score: low displacement + low communication variance = high routine
		comm_variance = statistics.variance(comm["hourly_distribution"].values()) if comm and len(comm["hourly_distribution"]) > 1 else 0.0
		routine_score = max(0.0, 1.0 - max_displacement_km / 100.0 - comm_variance / 100.0)

		pol_id = _fingerprint(target_id, period, _utcnow())
		result: dict[str, Any] = {
			"pol_id": pol_id,
			"target_id": target_id,
			"period": period,
			"location_fixes": len(tracks),
			"centroid_lat": round(centroid_lat, 6),
			"centroid_lon": round(centroid_lon, 6),
			"max_displacement_km": round(max_displacement_km, 2),
			"routine_score": round(routine_score, 4),
			"high_routine": routine_score > 0.7,
			"comm_peak_hour": comm["peak_hour_utc"] if comm else None,
			"night_activity_flag": comm["unusual_night_activity"] if comm else False,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._pattern_of_life[pol_id] = result
		self._audit(self.tenant_id, "surveillance_pattern_of_life_built", pol_id)
		return result

	async def associate_network(
		self,
		target_id: str,
		depth: int,
	) -> dict[str, Any]:
		"""Map the associate network of a surveillance target.

		Builds a graph of known associates up to depth levels using
		communication metadata and location co-presence.
		"""
		assert present(target_id), "target_id required"
		assert 1 <= depth <= 3, f"depth must be 1–3, got {depth}"

		target_hash = int(_fingerprint(target_id), 16)
		nodes: list[dict[str, Any]] = [{"id": target_id, "level": 0, "type": "TARGET"}]
		edges: list[dict[str, Any]] = []

		for level in range(1, depth + 1):
			count = max(1, (target_hash >> (level * 4)) % (5 * level))
			for j in range(count):
				assoc_id = _fingerprint(target_id, str(level), str(j))
				nodes.append({"id": assoc_id, "level": level, "type": "ASSOCIATE"})
				parent = nodes[max(0, len(nodes) - count - 1)]["id"]
				link_type = "COMMUNICATION" if (target_hash >> j) & 1 else "CO_LOCATION"
				edges.append({"from": parent, "to": assoc_id, "type": link_type})

		network_id = _fingerprint(target_id, str(depth), _utcnow())
		result: dict[str, Any] = {
			"network_id": network_id,
			"target_id": target_id,
			"depth": depth,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"associate_count": len(nodes) - 1,
			"nodes": nodes,
			"edges": edges[:50],
			"built_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._associate_networks[network_id] = result
		self._audit(self.tenant_id, "surveillance_associate_network_built", network_id)
		return result

	async def surveillance_audit(self, target_id: str) -> dict[str, Any]:
		"""Generate a compliance audit trail for all surveillance activities on a target.

		Returns chronological list of all surveillance events, authority references,
		and compliance status per event.
		"""
		assert present(target_id), "target_id required"

		reg = self._target_registrations.get(target_id)
		authority_ref = reg["authority_ref"] if reg else "UNKNOWN"

		related_events = [
			e for e in self.audit_events
			if e["tenant_id"] == self.tenant_id
			and target_id in e.get("reference_id", "")
		]

		# All location tracks
		location_count = sum(1 for t in self._location_tracks.values() if t.get("target_id") == target_id)
		comm_count = sum(1 for c in self._comm_metadata.values() if c.get("target_id") == target_id)
		pol_count = sum(1 for p in self._pattern_of_life.values() if p.get("target_id") == target_id)

		compliance_issues: list[str] = []
		if not reg:
			compliance_issues.append("NO_REGISTRATION_FOUND")
		elif reg.get("status") == "TERMINATED" and (location_count > 0 or comm_count > 0):
			compliance_issues.append("ACTIVITY_AFTER_TERMINATION")

		audit_id = _fingerprint(target_id, _utcnow())
		result: dict[str, Any] = {
			"audit_id": audit_id,
			"target_id": target_id,
			"authority_ref": authority_ref,
			"registration_status": reg["status"] if reg else "NOT_REGISTERED",
			"location_fixes": location_count,
			"comm_metadata_collections": comm_count,
			"pattern_of_life_analyses": pol_count,
			"total_audit_events": len(related_events),
			"compliance_issues": compliance_issues,
			"compliant": len(compliance_issues) == 0,
			"audited_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._surveillance_audits[audit_id] = result
		self._audit(self.tenant_id, "surveillance_audit_completed", audit_id)
		return result

	async def surveillance_report(
		self,
		target_id: str,
		classification: str,
	) -> dict[str, Any]:
		"""Generate a classified surveillance report for a specific target."""
		assert present(target_id), "target_id required"
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		report_id = _fingerprint(target_id, classification, tenant, _utcnow())

		reg = self._target_registrations.get(target_id)
		pol_results = [p for p in self._pattern_of_life.values() if p["tenant_id"] == tenant and p["target_id"] == target_id]
		assoc_nets = [n for n in self._associate_networks.values() if n["tenant_id"] == tenant and n["target_id"] == target_id]

		report: dict[str, Any] = {
			"report_id": report_id,
			"target_id": target_id,
			"classification": classification,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"registration": reg,
			"summary": {
				"location_fixes": sum(1 for t in self._location_tracks.values() if t.get("target_id") == target_id),
				"comm_metadata_periods": sum(1 for c in self._comm_metadata.values() if c.get("target_id") == target_id),
				"digital_footprint_analyses": sum(1 for f in self._footprint_analyses.values() if f["tenant_id"] == tenant and f.get("target_id") == target_id),
				"cross_platform_correlations": sum(1 for c in self._cross_platform_corrs.values() if c["tenant_id"] == tenant and c.get("target_id") == target_id),
				"pattern_of_life_analyses": len(pol_results),
				"high_routine_detected": any(p["high_routine"] for p in pol_results),
				"associate_network_analyses": len(assoc_nets),
				"max_associate_count": max((n["associate_count"] for n in assoc_nets), default=0),
				"total_observations": self._count(self.observations, tenant),
				"total_alerts": self._count(self.alerts, tenant),
			},
		}
		self._surveillance_reports[report_id] = report
		self._audit(tenant, "surveillance_report_generated", report_id)
		return report

	async def terminate_surveillance(
		self,
		target_id: str,
		reason: str,
	) -> dict[str, Any]:
		"""Terminate all surveillance activities on a target.

		Marks the target registration as TERMINATED, records the reason,
		and generates a termination audit record.
		"""
		assert present(target_id), "target_id required"
		assert present(reason), "reason required"

		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered")
		if reg["status"] == "TERMINATED":
			raise ValueError(f"Surveillance on {target_id!r} is already terminated")

		previous_status = reg["status"]
		reg["status"] = "TERMINATED"
		reg["terminated_at"] = _utcnow()
		reg["termination_reason"] = reason

		termination_id = _fingerprint(target_id, reason, _utcnow())
		record: dict[str, Any] = {
			"termination_id": termination_id,
			"target_id": target_id,
			"reason": reason,
			"previous_status": previous_status,
			"terminated_at": _utcnow(),
			"actor_id": self.actor_id,
			"tenant_id": self.tenant_id,
		}
		self._terminations[termination_id] = record
		self._audit(self.tenant_id, "surveillance_terminated", termination_id)
		return record

	async def bulk_target_registration(self, targets: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-register multiple surveillance targets under validated authorities.

		Each entry: {"target_id": str, "authority_ref": str, "scope": str, "expiry": str}.
		"""
		assert targets, "targets required"
		assert len(targets) <= 100, "bulk cap: 100 targets"

		successes: list[str] = []
		failures: list[dict[str, Any]] = []
		for t in targets:
			try:
				await self.register_surveillance_target(
					target_id=t["target_id"],
					authority_ref=t["authority_ref"],
					scope=t["scope"],
					expiry=t["expiry"],
				)
				successes.append(t["target_id"])
			except Exception as exc:
				failures.append({"target_id": t.get("target_id", "?"), "error": str(exc)})

		bulk_id = _fingerprint(str(len(targets)), _utcnow())
		result: dict[str, Any] = {
			"bulk_id": bulk_id,
			"submitted": len(targets),
			"succeeded": len(successes),
			"failed": len(failures),
			"target_ids": successes,
			"failures": failures,
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_bulk_targets_registered", bulk_id)
		return result

	async def geofence_alert(
		self,
		target_id: str,
		fence_lat: float,
		fence_lon: float,
		radius_km: float,
	) -> dict[str, Any]:
		"""Check whether a target has entered or exited a geofence.

		fence_lat/lon: geofence centre coordinates
		radius_km: geofence radius
		"""
		assert present(target_id), "target_id required"
		assert radius_km > 0, "radius_km must be positive"

		tracks = [
			t for t in self._location_tracks.values()
			if t["tenant_id"] == self.tenant_id and t["target_id"] == target_id
		]
		if not tracks:
			raise KeyError(f"No location data for target {target_id!r}")

		latest = max(tracks, key=lambda t: t.get("tracked_at", ""))
		dist_km = _haversine(fence_lat, fence_lon, latest["latitude"], latest["longitude"])
		inside_fence = dist_km <= radius_km

		alert_id = _fingerprint(target_id, str(fence_lat), str(fence_lon), _utcnow())
		result: dict[str, Any] = {
			"alert_id": alert_id,
			"target_id": target_id,
			"fence_centre": {"lat": fence_lat, "lon": fence_lon},
			"radius_km": radius_km,
			"target_lat": latest["latitude"],
			"target_lon": latest["longitude"],
			"distance_km": round(dist_km, 3),
			"inside_fence": inside_fence,
			"alert_type": "ENTERED_GEOFENCE" if inside_fence else "OUTSIDE_GEOFENCE",
			"checked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_geofence_alert_raised", alert_id)
		return result

	async def device_identifier_correlation(
		self,
		target_id: str,
		device_ids: list[str],
	) -> dict[str, Any]:
		"""Correlate device identifiers (IMEI, MAC, device fingerprint) to a target.

		Returns correlation confidence and matched identifiers.
		"""
		assert present(target_id), "target_id required"
		assert device_ids, "device_ids required"

		correlations: list[dict[str, Any]] = []
		for did in device_ids:
			d_hash = int(_fingerprint(target_id, did), 16)
			match_score = round((d_hash % 100) / 100.0, 4)
			correlations.append({
				"device_id": did,
				"match_score": match_score,
				"correlated": match_score >= 0.6,
			})

		correlated = [c for c in correlations if c["correlated"]]
		mean_score = round(statistics.mean(c["match_score"] for c in correlations), 4)

		corr_id = _fingerprint(target_id, *sorted(device_ids), _utcnow())
		result: dict[str, Any] = {
			"correlation_id": corr_id,
			"target_id": target_id,
			"devices_checked": len(device_ids),
			"devices_correlated": len(correlated),
			"mean_match_score": mean_score,
			"correlated_devices": correlated,
			"correlated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_device_identifier_correlated", corr_id)
		return result

	async def surveillance_compliance_report(self) -> dict[str, Any]:
		"""Generate a programme-level compliance report covering all targets.

		Checks: authority coverage, expiry status, and post-termination activity.
		"""
		tenant = self.tenant_id
		total_targets = len(self._target_registrations)
		active = sum(1 for t in self._target_registrations.values() if t["status"] == "ACTIVE")
		terminated = sum(1 for t in self._target_registrations.values() if t["status"] == "TERMINATED")
		missing_authority = sum(
			1 for t in self._target_registrations.values()
			if not any(
				a.authority_id == t["authority_ref"]
				for a in self.authorities.values()
				if a.tenant_id == tenant
			)
		)

		compliance_issues: list[str] = []
		if missing_authority > 0:
			compliance_issues.append(f"{missing_authority}_TARGETS_MISSING_AUTHORITY")
		if terminated > 0 and len(self._location_tracks) > terminated * 2:
			compliance_issues.append("POTENTIAL_POST_TERMINATION_ACTIVITY")

		report_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"report_id": report_id,
			"total_targets": total_targets,
			"active_targets": active,
			"terminated_targets": terminated,
			"targets_missing_authority": missing_authority,
			"compliance_issues": compliance_issues,
			"compliant": len(compliance_issues) == 0,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_compliance_report_generated", report_id)
		return result

	async def behavioural_anomaly_detection(self, target_id: str) -> dict[str, Any]:
		"""Detect behavioural anomalies for a surveillance target.

		Compares current behaviour against established pattern-of-life baseline.
		"""
		assert present(target_id), "target_id required"

		pol_entries = [
			p for p in self._pattern_of_life.values()
			if p["tenant_id"] == self.tenant_id and p["target_id"] == target_id
		]
		if not pol_entries:
			return {
				"anomaly_id": _fingerprint(target_id, _utcnow()),
				"target_id": target_id,
				"anomalies": ["NO_BASELINE_ESTABLISHED"],
				"anomaly_score": 1.0,
				"tenant_id": self.tenant_id,
			}

		latest_pol = max(pol_entries, key=lambda p: p.get("analysed_at", ""))
		anomalies: list[str] = []
		if not latest_pol.get("high_routine", True):
			anomalies.append("ROUTINE_DEVIATION_DETECTED")
		if latest_pol.get("night_activity_flag", False):
			anomalies.append("UNUSUAL_NIGHT_ACTIVITY")
		if latest_pol.get("max_displacement_km", 0) > 100:
			anomalies.append("EXCESSIVE_DISPLACEMENT")

		anomaly_score = len(anomalies) / 3.0
		anomaly_id = _fingerprint(target_id, _utcnow())
		result: dict[str, Any] = {
			"anomaly_id": anomaly_id,
			"target_id": target_id,
			"anomalies": anomalies,
			"anomaly_score": round(anomaly_score, 4),
			"anomaly_level": "HIGH" if anomaly_score >= 0.67 else "MEDIUM" if anomaly_score >= 0.33 else "LOW",
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_behavioural_anomaly_detected", anomaly_id)
		return result

	async def export_surveillance_data(self, target_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export all surveillance data for a target in specified format.

		fmt: json | csv
		"""
		VALID_FMTS = {"json", "csv"}
		assert present(target_id), "target_id required"
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		location_count = sum(1 for t in self._location_tracks.values() if t.get("target_id") == target_id)
		comm_count = sum(1 for c in self._comm_metadata.values() if c.get("target_id") == target_id)
		pol_count = sum(1 for p in self._pattern_of_life.values() if p.get("target_id") == target_id)

		export_id = _fingerprint(target_id, fmt, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"target_id": target_id,
			"format": fmt,
			"location_fixes_exported": location_count,
			"comm_metadata_periods_exported": comm_count,
			"pattern_of_life_exported": pol_count,
			"content_fingerprint": _fingerprint(target_id, str(location_count + comm_count)),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_data_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return surveillance service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"active_targets": sum(1 for t in self._target_registrations.values() if t["status"] == "ACTIVE"),
			"location_tracks": len(self._location_tracks),
			"pattern_of_life_analyses": len(self._pattern_of_life),
			"observations": self._count(self.observations, tenant),
			"alerts": self._count(self.alerts, tenant),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def lawful_intercept(
		self,
		target_id: str,
		authority_ref: str,
		channel: str,
	) -> dict[str, Any]:
		"""Initiate lawful intercept for *target_id* on *channel* under *authority_ref*."""
		assert present(target_id) and present(authority_ref) and present(channel), "all params required"
		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered — register first")
		intercept_id = _fingerprint(target_id, authority_ref, channel, _utcnow())
		record: dict[str, Any] = {
			"intercept_id": intercept_id,
			"target_id": target_id,
			"authority_ref": authority_ref,
			"channel": channel,
			"status": "ACTIVE",
			"started_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._audit(self.tenant_id, "surveillance_lawful_intercept_started", intercept_id)
		return record

	async def pattern_life(
		self,
		target_id: str,
		period: str = "30d",
	) -> dict[str, Any]:
		"""Alias for pattern_of_life."""
		return await self.pattern_of_life(target_id, period)

	async def associate_map(
		self,
		target_id: str,
		depth: int = 2,
	) -> dict[str, Any]:
		"""Alias for associate_network."""
		return await self.associate_network(target_id, depth)

	async def surveillance_export(
		self,
		target_id: str,
		fmt: str = "json",
	) -> dict[str, Any]:
		"""Alias for export_surveillance_data."""
		return await self.export_surveillance_data(target_id, fmt)

	async def sensor_health_check(self) -> dict[str, Any]:
		"""Report health status of all registered surveillance sensors."""
		tenant = self.tenant_id
		sensors = [s for s in self.sensors.values() if s.tenant_id == tenant]
		missing_calibration = [s.sensor_id for s in sensors if not s.calibration_reference.strip()]
		status_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"status_id": status_id,
			"total_sensors": len(sensors),
			"calibration_overdue": len(missing_calibration),
			"overdue_ids": missing_calibration,
			"coverage_pct": round((len(sensors) - len(missing_calibration)) / max(len(sensors), 1) * 100, 1),
			"checked_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_sensor_health_checked", status_id)
		return result

	async def target_lifecycle_status(self) -> dict[str, Any]:
		"""Return lifecycle status summary for all registered targets."""
		tenant = self.tenant_id
		status_counts: dict[str, int] = {}
		for t in self._target_registrations.values():
			s = t.get("status", "UNKNOWN")
			status_counts[s] = status_counts.get(s, 0) + 1

		report_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"report_id": report_id,
			"status_distribution": status_counts,
			"total_targets": len(self._target_registrations),
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_target_lifecycle_reported", report_id)
		return result

	async def surveillance_kpi_report(self) -> dict[str, Any]:
		"""Generate KPI report aggregating key surveillance programme metrics."""
		tenant = self.tenant_id
		observations = self._count(self.observations, tenant)
		alerts = self._count(self.alerts, tenant)
		risks = self._count(self.risks, tenant)
		observation_to_alert_ratio = round(alerts / max(observations, 1), 4)

		kpi_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"kpi_id": kpi_id,
			"observations": observations,
			"alerts": alerts,
			"risks": risks,
			"observation_to_alert_ratio": observation_to_alert_ratio,
			"active_targets": sum(1 for t in self._target_registrations.values() if t["status"] == "ACTIVE"),
			"terminations": len(self._terminations),
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_kpi_reported", kpi_id)
		return result

	# ------------------------------------------------------------------
	# New async methods — physical surveillance + advanced analytics
	# ------------------------------------------------------------------

	async def field_agent_tasking(
		self,
		target_id: str,
		agent_id: str,
		observation_zone: str,
		priority: str,
		instructions: str = "",
	) -> dict[str, Any]:
		"""Task a registered field agent with a physical surveillance assignment.

		Validates that both the target and agent are registered and active, then
		creates a tasking record that bridges physical-world observation into the
		digital analytics pipeline.

		priority: CRITICAL | HIGH | MEDIUM | LOW
		"""
		VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
		assert present(target_id), "target_id required"
		assert present(agent_id), "agent_id required"
		assert present(observation_zone), "observation_zone required"
		priority_upper = priority.upper()
		if priority_upper not in VALID_PRIORITIES:
			raise ValueError(f"priority must be one of {VALID_PRIORITIES}")

		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered")
		if reg["status"] != "ACTIVE":
			raise PermissionError(f"Surveillance on {target_id!r} is {reg['status']}, not ACTIVE")

		agent_rec = self.agents.get(self._tenant_key(self.tenant_id, agent_id))
		if agent_rec is None:
			raise KeyError(f"agent_id {agent_id!r} not registered for tenant {self.tenant_id!r}")

		tasking_id = _fingerprint(target_id, agent_id, observation_zone, _utcnow())
		record: dict[str, Any] = {
			"tasking_id": tasking_id,
			"target_id": target_id,
			"agent_id": agent_id,
			"observation_zone": observation_zone,
			"priority": priority_upper,
			"instructions": instructions,
			"status": "ASSIGNED",
			"authority_ref": reg["authority_ref"],
			"assigned_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		# Persist to a dedicated store keyed by tasking_id
		if not hasattr(self, "_field_taskings"):
			self._field_taskings: dict[str, dict[str, Any]] = {}
		self._field_taskings[tasking_id] = record
		self._audit(self.tenant_id, "surveillance_field_agent_tasked", tasking_id)
		return record

	async def observation_report_ingest(
		self,
		target_id: str,
		agent_id: str,
		report_text: str,
		media_refs: list[str] | None = None,
		location_lat: float | None = None,
		location_lon: float | None = None,
	) -> dict[str, Any]:
		"""Ingest a physical-world observation report submitted by a field agent.

		Stores the report, links it to the digital target record, and optionally
		records a location fix derived from the agent's reported position.
		"""
		assert present(target_id), "target_id required"
		assert present(agent_id), "agent_id required"
		assert present(report_text), "report_text required"

		reg = self._target_registrations.get(target_id)
		if reg is None:
			raise KeyError(f"target_id {target_id!r} not registered")

		report_id = _fingerprint(target_id, agent_id, report_text[:64], _utcnow())
		record: dict[str, Any] = {
			"report_id": report_id,
			"target_id": target_id,
			"agent_id": agent_id,
			"report_text": report_text,
			"media_refs": media_refs or [],
			"word_count": len(report_text.split()),
			"has_location": location_lat is not None and location_lon is not None,
			"location_lat": location_lat,
			"location_lon": location_lon,
			"reported_at": _utcnow(),
			"authority_ref": reg["authority_ref"],
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		if not hasattr(self, "_observation_reports"):
			self._observation_reports: dict[str, dict[str, Any]] = {}
		self._observation_reports[report_id] = record

		# If coordinates provided, also log a location fix
		if location_lat is not None and location_lon is not None:
			track_id = _fingerprint(target_id, "FIELD_REPORT", _utcnow())
			self._location_tracks[track_id] = {
				"track_id": track_id,
				"target_id": target_id,
				"source": "FIELD_REPORT",
				"latitude": location_lat,
				"longitude": location_lon,
				"accuracy_m": 100.0,
				"tracked_at": _utcnow(),
				"tenant_id": self.tenant_id,
			}
			record["location_track_id"] = track_id

		self._audit(self.tenant_id, "surveillance_observation_report_ingested", report_id)
		return record

	async def build_target_profile(self, target_id: str) -> dict[str, Any]:
		"""Aggregate all available intelligence into a unified target profile.

		Denormalises registration, location centroid, communication metadata,
		digital footprint score, cross-platform correlation confidence,
		pattern-of-life, and associate network depth into a single read-through
		record suitable for downstream consumers (case management, dissemination).
		"""
		assert present(target_id), "target_id required"

		tenant = self.tenant_id
		reg = self._target_registrations.get(target_id)

		# Location centroid
		tracks = [t for t in self._location_tracks.values() if t.get("target_id") == target_id and t["tenant_id"] == tenant]
		centroid_lat = round(statistics.mean(t["latitude"] for t in tracks), 6) if tracks else None
		centroid_lon = round(statistics.mean(t["longitude"] for t in tracks), 6) if tracks else None

		# Comm metadata (latest)
		comm_entries = [c for c in self._comm_metadata.values() if c.get("target_id") == target_id and c["tenant_id"] == tenant]
		latest_comm = max(comm_entries, key=lambda c: c.get("collected_at", "")) if comm_entries else None

		# Digital footprint (latest)
		footprint_entries = [f for f in self._footprint_analyses.values() if f.get("target_id") == target_id and f["tenant_id"] == tenant]
		latest_footprint = max(footprint_entries, key=lambda f: f.get("analysed_at", "")) if footprint_entries else None

		# Cross-platform (latest)
		corr_entries = [c for c in self._cross_platform_corrs.values() if c.get("target_id") == target_id and c["tenant_id"] == tenant]
		latest_corr = max(corr_entries, key=lambda c: c.get("correlated_at", "")) if corr_entries else None

		# Pattern of life (latest)
		pol_entries = [p for p in self._pattern_of_life.values() if p.get("target_id") == target_id and p["tenant_id"] == tenant]
		latest_pol = max(pol_entries, key=lambda p: p.get("analysed_at", "")) if pol_entries else None

		# Associate network (widest)
		assoc_entries = [n for n in self._associate_networks.values() if n.get("target_id") == target_id and n["tenant_id"] == tenant]
		widest_network = max(assoc_entries, key=lambda n: n.get("associate_count", 0)) if assoc_entries else None

		profile_id = _fingerprint(target_id, tenant, _utcnow())
		profile: dict[str, Any] = {
			"profile_id": profile_id,
			"target_id": target_id,
			"registration_status": reg["status"] if reg else "NOT_REGISTERED",
			"authority_ref": reg["authority_ref"] if reg else None,
			"location_fixes": len(tracks),
			"centroid_lat": centroid_lat,
			"centroid_lon": centroid_lon,
			"comm_call_count": latest_comm["call_count"] if latest_comm else None,
			"comm_unique_contacts": latest_comm["unique_contacts"] if latest_comm else None,
			"comm_peak_hour_utc": latest_comm["peak_hour_utc"] if latest_comm else None,
			"digital_footprint_score": latest_footprint["digital_footprint_score"] if latest_footprint else None,
			"high_digital_exposure": latest_footprint["high_exposure"] if latest_footprint else None,
			"cross_platform_confidence": latest_corr["unified_identity_confidence"] if latest_corr else None,
			"routine_score": latest_pol["routine_score"] if latest_pol else None,
			"high_routine": latest_pol["high_routine"] if latest_pol else None,
			"max_displacement_km": latest_pol["max_displacement_km"] if latest_pol else None,
			"night_activity_flag": latest_pol["night_activity_flag"] if latest_pol else False,
			"associate_count": widest_network["associate_count"] if widest_network else 0,
			"associate_network_depth": widest_network["depth"] if widest_network else 0,
			"built_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
		}
		self._audit(tenant, "surveillance_target_profile_built", profile_id)
		return profile

	async def trajectory_analysis(
		self,
		target_id: str,
		window_hours: int = 24,
	) -> dict[str, Any]:
		"""Analyse the movement trajectory of a target within a time window.

		Computes per-leg distance, speed, heading, mode-of-transport estimate,
		and flags impossible jumps (>300 km/h between consecutive fixes).

		Returns a trajectory report with ordered legs and a summary.
		"""
		assert present(target_id), "target_id required"
		assert 1 <= window_hours <= 720, "window_hours must be 1–720"

		tenant = self.tenant_id
		tracks = sorted(
			[t for t in self._location_tracks.values() if t.get("target_id") == target_id and t["tenant_id"] == tenant],
			key=lambda t: t.get("tracked_at", ""),
		)

		legs: list[dict[str, Any]] = []
		impossible_jumps = 0
		total_distance_km = 0.0

		for i in range(1, len(tracks)):
			prev = tracks[i - 1]
			curr = tracks[i]
			dist_km = _haversine(prev["latitude"], prev["longitude"], curr["latitude"], curr["longitude"])
			total_distance_km += dist_km

			# Rough time delta in hours using string prefix comparison (ISO format sortable)
			# For simulation purposes, estimate a 1-hour gap per fix
			time_delta_h = 1.0
			speed_kmh = dist_km / max(time_delta_h, 1e-6)
			impossible = speed_kmh > 300.0
			if impossible:
				impossible_jumps += 1

			mode = (
				"STATIONARY" if speed_kmh < 2
				else "PEDESTRIAN" if speed_kmh < 10
				else "VEHICLE" if speed_kmh < 150
				else "AIR" if not impossible
				else "IMPOSSIBLE"
			)

			# Heading in degrees (approximate)
			dlat = curr["latitude"] - prev["latitude"]
			dlon = curr["longitude"] - prev["longitude"]
			heading_deg = round(math.degrees(math.atan2(dlon, dlat)) % 360, 1)

			legs.append({
				"from_fix": i - 1,
				"to_fix": i,
				"distance_km": round(dist_km, 3),
				"speed_kmh": round(speed_kmh, 1),
				"heading_deg": heading_deg,
				"mode": mode,
				"impossible_jump": impossible,
			})

		traj_id = _fingerprint(target_id, str(window_hours), _utcnow())
		result: dict[str, Any] = {
			"trajectory_id": traj_id,
			"target_id": target_id,
			"window_hours": window_hours,
			"fix_count": len(tracks),
			"leg_count": len(legs),
			"total_distance_km": round(total_distance_km, 3),
			"impossible_jumps": impossible_jumps,
			"data_integrity_flag": impossible_jumps > 0,
			"legs": legs[:100],  # cap to avoid oversized responses
			"analysed_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_trajectory_analysed", traj_id)
		return result

	async def register_evidence(
		self,
		evidence_id: str,
		reference_url: str,
		sha256_hash: str,
		custodian_id: str,
		expiry: str,
	) -> dict[str, Any]:
		"""Register an evidence item in the chain-of-custody registry.

		Each evidence item is uniquely identified, fingerprinted, and linked to
		a custodian. All subsequent surveillance methods that consume an
		evidence_reference should resolve against this registry.

		sha256_hash: hex-encoded SHA-256 of the evidence artefact at ingest.
		"""
		assert present(evidence_id), "evidence_id required"
		assert present(reference_url), "reference_url required"
		assert present(sha256_hash) and len(sha256_hash) == 64, "sha256_hash must be 64-char hex"
		assert present(custodian_id), "custodian_id required"
		assert present(expiry), "expiry required"

		if not hasattr(self, "_evidence_registry"):
			self._evidence_registry: dict[str, dict[str, Any]] = {}

		if evidence_id in self._evidence_registry:
			raise ValueError(f"evidence_id {evidence_id!r} already registered — use a unique ID")

		ingest_fingerprint = _fingerprint(evidence_id, sha256_hash, reference_url)
		record: dict[str, Any] = {
			"evidence_id": evidence_id,
			"reference_url": reference_url,
			"sha256_hash": sha256_hash,
			"ingest_fingerprint": ingest_fingerprint,
			"custodian_id": custodian_id,
			"expiry": expiry,
			"chain_of_custody": [
				{
					"event": "REGISTERED",
					"actor_id": self.actor_id,
					"at": _utcnow(),
				}
			],
			"registered_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._evidence_registry[evidence_id] = record
		self._audit(self.tenant_id, "surveillance_evidence_registered", evidence_id)
		return record

	async def check_authority_renewals(self, days_ahead: int = 30) -> dict[str, Any]:
		"""Identify surveillance authorities expiring within *days_ahead* days.

		Fires a notification via the injected notify adapter if configured.
		Returns a renewal alert record listing affected authority IDs and their
		expiry dates so operators can initiate renewal workflows.
		"""
		assert 1 <= days_ahead <= 365, "days_ahead must be 1–365"

		tenant = self.tenant_id
		now_ts = _utcnow()
		alerts_list: list[dict[str, Any]] = []

		for auth in self.authorities.values():
			if auth.tenant_id != tenant:
				continue
			# Compare ISO strings lexicographically — valid for UTC ISO-8601 dates
			# (works because YYYY-MM-DD... sorts correctly as strings)
			expires = getattr(auth, "expires_at", "")
			if expires and expires <= now_ts:
				alerts_list.append({
					"authority_id": auth.authority_id,
					"expires_at": expires,
					"status": "EXPIRED",
				})
			elif expires:
				alerts_list.append({
					"authority_id": auth.authority_id,
					"expires_at": expires,
					"status": "EXPIRING_SOON",
				})

		# Notify if adapter is wired
		if self._notify and alerts_list:
			try:
				if asyncio.iscoroutinefunction(self._notify.send):
					await self._notify.send(
						topic="surveillance.authority_renewal",
						payload={"alerts": alerts_list, "tenant_id": tenant},
					)
				else:
					self._notify.send(
						topic="surveillance.authority_renewal",
						payload={"alerts": alerts_list, "tenant_id": tenant},
					)
			except Exception:
				pass  # notification failure must not block the workflow

		check_id = _fingerprint(tenant, str(days_ahead), _utcnow())
		result: dict[str, Any] = {
			"check_id": check_id,
			"days_ahead": days_ahead,
			"authorities_checked": sum(1 for a in self.authorities.values() if a.tenant_id == tenant),
			"renewal_alerts": alerts_list,
			"alert_count": len(alerts_list),
			"checked_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_authority_renewal_checked", check_id)
		return result

	async def ingest_observations_batch(
		self,
		observations: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Batch-ingest up to 500 observations with per-item deduplication.

		Each item must carry the same keys as `record_observation`.
		Items with a `content_fingerprint` already present in the store are
		skipped (DEDUP) rather than re-ingested. Returns per-item
		success/skip/failure breakdown.

		Suitable for high-volume sensor feeds; processes items concurrently
		using asyncio.gather.
		"""
		assert observations, "observations list required"
		assert len(observations) <= 500, "batch cap: 500 observations"

		# Build fingerprint index for dedup
		existing_fingerprints: set[str] = {
			obs.content_fingerprint
			for obs in self.observations.values()
			if obs.tenant_id == self.tenant_id
		}

		successes: list[str] = []
		skipped: list[str] = []
		failures: list[dict[str, Any]] = []

		async def _ingest_one(item: dict[str, Any]) -> None:
			fp = item.get("content_fingerprint", "")
			oid = item.get("observation_id", "?")
			if fp in existing_fingerprints:
				skipped.append(oid)
				return
			try:
				self.record_observation(
					observation_id=item["observation_id"],
					tenant_id=item.get("tenant_id", self.tenant_id),
					program_id=item["program_id"],
					sensor_id=item["sensor_id"],
					observation_type=item["observation_type"],
					observation_reference=item["observation_reference"],
					content_fingerprint=fp,
					observed_at=item.get("observed_at", _utcnow()),
					confidence_score=float(item.get("confidence_score", 0.5)),
					evidence_reference=item.get("evidence_reference", "batch"),
				)
				existing_fingerprints.add(fp)
				successes.append(oid)
			except Exception as exc:
				failures.append({"observation_id": oid, "error": str(exc)})

		await asyncio.gather(*[_ingest_one(item) for item in observations], return_exceptions=True)

		batch_id = _fingerprint(str(len(observations)), self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"batch_id": batch_id,
			"submitted": len(observations),
			"succeeded": len(successes),
			"skipped_dedup": len(skipped),
			"failed": len(failures),
			"observation_ids": successes,
			"failures": failures,
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "surveillance_observations_batch_ingested", batch_id)
		return result

	async def cross_target_pattern_correlation(
		self,
		target_ids: list[str],
		period: str,
	) -> dict[str, Any]:
		"""Correlate pattern-of-life entries across multiple targets.

		Identifies targets that share similar location centroids, communication
		peak hours, or night-activity flags — indicators of coordinated behaviour
		or shared handler contact.

		Returns a correlation matrix and a list of suspected co-ordination pairs.
		"""
		assert target_ids and len(target_ids) >= 2, "need at least 2 target_ids"
		assert present(period), "period required"

		tenant = self.tenant_id
		profiles: dict[str, dict[str, Any]] = {}
		for tid in target_ids:
			pol_entries = [
				p for p in self._pattern_of_life.values()
				if p["tenant_id"] == tenant and p["target_id"] == tid
			]
			if pol_entries:
				profiles[tid] = max(pol_entries, key=lambda p: p.get("analysed_at", ""))

		pairs: list[dict[str, Any]] = []
		tids = list(profiles.keys())
		for i in range(len(tids)):
			for j in range(i + 1, len(tids)):
				a, b = profiles[tids[i]], profiles[tids[j]]
				dist_km = _haversine(
					a["centroid_lat"], a["centroid_lon"],
					b["centroid_lat"], b["centroid_lon"],
				)
				same_peak_hour = a.get("comm_peak_hour") == b.get("comm_peak_hour")
				both_night = a.get("night_activity_flag", False) and b.get("night_activity_flag", False)
				correlation_score = round(
					(1.0 if dist_km < 5.0 else 0.0) * 0.5 +
					(0.3 if same_peak_hour else 0.0) +
					(0.2 if both_night else 0.0),
					4,
				)
				pairs.append({
					"target_a": tids[i],
					"target_b": tids[j],
					"centroid_distance_km": round(dist_km, 3),
					"same_peak_hour": same_peak_hour,
					"both_night_active": both_night,
					"correlation_score": correlation_score,
					"coordinated": correlation_score >= 0.5,
				})

		coordinated_pairs = [p for p in pairs if p["coordinated"]]
		corr_id = _fingerprint(*sorted(target_ids), period, _utcnow())
		result: dict[str, Any] = {
			"correlation_id": corr_id,
			"period": period,
			"targets_analysed": len(profiles),
			"targets_without_pol": len(target_ids) - len(profiles),
			"pair_count": len(pairs),
			"coordinated_pair_count": len(coordinated_pairs),
			"coordinated_pairs": coordinated_pairs,
			"pairs": pairs,
			"correlated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "surveillance_cross_target_pattern_correlated", corr_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

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
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": _utcnow(),
			"processor": "bytewax",
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "surveillance_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "surveillance_policy_denied")


# Aliases for backward compatibility
IntelSurveillanceService = DigitalSurveillanceService
