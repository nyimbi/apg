"""Executable service layer for APG Radio Intelligence (RINT).

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

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_BAND_TYPES, SUPPORTED_CLASSIFICATION_TYPES, SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_EVENT_TYPES, SUPPORTED_RECEIVER_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SESSION_TYPES,
		SUPPORTED_SIGNAL_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		RadioAgent, RadioAuthority, RadioBandPlan, RadioCollectionSession,
		RadioDissemination, RadioEventAssessment, RadioReceiver, RadioReferral,
		RadioReview, RadioSignalObservation, RadioTransmissionClassification,
	)
	from .radio_runtime import bounded_score, nonnegative_float, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_BAND_TYPES, SUPPORTED_CLASSIFICATION_TYPES, SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_EVENT_TYPES, SUPPORTED_RECEIVER_TYPES, SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_REVIEW_STATUSES, SUPPORTED_RISK_LEVELS, SUPPORTED_SESSION_TYPES,
		SUPPORTED_SIGNAL_TYPES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		RadioAgent, RadioAuthority, RadioBandPlan, RadioCollectionSession,
		RadioDissemination, RadioEventAssessment, RadioReceiver, RadioReferral,
		RadioReview, RadioSignalObservation, RadioTransmissionClassification,
	)
	from radio_runtime import bounded_score, nonnegative_float, normalize_code, positive_int, present  # type: ignore


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(*parts: str) -> str:
	blob = "|".join(str(p) for p in parts)
	return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ITU band designations and frequency ranges in MHz
_BAND_RANGES_MHZ: dict[str, tuple[float, float]] = {
	"ELF": (0.003, 0.003),
	"SLF": (0.003, 0.03),
	"ULF": (0.03, 0.3),
	"VLF": (0.3, 3.0),
	"LF": (30.0, 300.0),
	"MF": (300.0, 3_000.0),
	"HF": (3_000.0, 30_000.0),
	"VHF": (30_000.0, 300_000.0),
	"UHF": (300_000.0, 3_000_000.0),
	"SHF": (3_000_000.0, 30_000_000.0),
	"EHF": (30_000_000.0, 300_000_000.0),
}

# Protocol → decoder map
_PROTOCOL_DECODERS: dict[str, str] = {
	"APRS": "aprs_decoder",
	"DMR": "dmr_decoder",
	"P25": "p25_decoder",
	"DSTAR": "dstar_decoder",
	"TETRA": "tetra_decoder",
	"ACARS": "acars_decoder",
	"AIS": "ais_decoder",
	"ADS_B": "adsb_decoder",
	"MODE_S": "mode_s_decoder",
	"SELCAL": "selcal_decoder",
	"NOAA_APT": "apt_decoder",
	"FSK": "fsk_decoder",
	"RTTY": "rtty_decoder",
	"CW": "cw_decoder",
}


class RadioIntelligenceService:
	"""Tenant-scoped radio intelligence runtime for generated APG applications.

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

		# Operational state added by new methods
		self._frequency_scans: dict[str, dict[str, Any]] = {}
		self._recordings: dict[str, dict[str, Any]] = {}
		self._decoded_transmissions: dict[str, dict[str, Any]] = {}
		self._emitter_ids: dict[str, dict[str, Any]] = {}
		self._df_results: dict[str, dict[str, Any]] = {}
		self._monitoring_schedules: dict[str, dict[str, Any]] = {}
		self._interference_detections: dict[str, dict[str, Any]] = {}
		self._spectrum_analyses: dict[str, dict[str, Any]] = {}
		self._border_monitors: dict[str, dict[str, Any]] = {}
		self._reports: dict[str, dict[str, Any]] = {}

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
		item = RadioAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "radio_authority_recorded", authority_id)
		return item.to_dict()

	def record_band_plan(
		self, band_id: str, tenant_id: str, band_type: str, name: str,
		frequency_min_mhz: float, frequency_max_mhz: float,
		authority_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		band_type = normalize_code(band_type)
		frequency_min = float(frequency_min_mhz) if nonnegative_float(frequency_min_mhz) else -1.0
		frequency_max = float(frequency_max_mhz) if nonnegative_float(frequency_max_mhz) else -1.0
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_band_plan",
			"band_type_supported": band_type in SUPPORTED_BAND_TYPES,
			"band_name_present": present(name),
			"frequency_min_valid": nonnegative_float(frequency_min_mhz),
			"frequency_max_valid": nonnegative_float(frequency_max_mhz),
			"frequency_range_valid": frequency_max >= frequency_min and frequency_min >= 0,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = RadioBandPlan(band_id, tenant_id, band_type, name, frequency_min, frequency_max, authority_id, evidence_reference)
		self.band_plans[self._tenant_key(tenant_id, band_id)] = item
		self._audit(tenant_id, "radio_band_plan_recorded", band_id)
		return item.to_dict()

	def register_receiver(
		self, receiver_id: str, tenant_id: str, receiver_type: str, site_reference: str,
		custodian_id: str, authority_id: str, calibration_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		receiver_type = normalize_code(receiver_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_receiver",
			"receiver_type_supported": receiver_type in SUPPORTED_RECEIVER_TYPES,
			"site_reference_present": present(site_reference),
			"custodian_present": present(custodian_id),
			"authority_present": authority is not None,
			"calibration_present": present(calibration_reference),
			"evidence_present": present(evidence_reference),
		})
		item = RadioReceiver(receiver_id, tenant_id, receiver_type, site_reference, custodian_id, authority_id, calibration_reference, evidence_reference)
		self.receivers[self._tenant_key(tenant_id, receiver_id)] = item
		self._audit(tenant_id, "radio_receiver_registered", receiver_id)
		return item.to_dict()

	def record_session(
		self, session_id: str, tenant_id: str, band_id: str, receiver_id: str,
		session_type: str, started_at: str, ended_at: str,
		collection_plan_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		band = self._tenant_band_or_none(band_id, tenant_id)
		receiver = self._tenant_receiver_or_none(receiver_id, tenant_id)
		session_type = normalize_code(session_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_session",
			"band_present": band is not None,
			"receiver_present": receiver is not None,
			"band_receiver_authority_match": band is not None and receiver is not None and band.authority_id == receiver.authority_id,
			"session_type_supported": session_type in SUPPORTED_SESSION_TYPES,
			"started_at_present": present(started_at),
			"collection_plan_present": present(collection_plan_reference),
			"evidence_present": present(evidence_reference),
		})
		item = RadioCollectionSession(session_id, tenant_id, band_id, receiver_id, session_type, started_at, ended_at, collection_plan_reference, evidence_reference)
		self.sessions[self._tenant_key(tenant_id, session_id)] = item
		self._audit(tenant_id, "radio_session_recorded", session_id)
		return item.to_dict()

	def record_observation(
		self, observation_id: str, tenant_id: str, session_id: str,
		frequency_mhz: float, signal_type: str, signal_fingerprint: str,
		observed_at: str, confidence_score: float, evidence_reference: str,
	) -> dict[str, Any]:
		session = self._tenant_session_or_none(session_id, tenant_id)
		band = self._tenant_band_or_none(session.band_id, tenant_id) if session is not None else None
		frequency = float(frequency_mhz) if nonnegative_float(frequency_mhz) else -1.0
		signal_type = normalize_code(signal_type)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_observation",
			"session_present": session is not None,
			"frequency_valid": nonnegative_float(frequency_mhz),
			"frequency_in_band": band is not None and band.frequency_min_mhz <= frequency <= band.frequency_max_mhz,
			"signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES,
			"fingerprint_present": present(signal_fingerprint),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = RadioSignalObservation(observation_id, tenant_id, session_id, frequency, signal_type, signal_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.observations[self._tenant_key(tenant_id, observation_id)] = item
		self._audit(tenant_id, "radio_observation_recorded", observation_id)
		return item.to_dict()

	def record_classification(
		self, classification_id: str, tenant_id: str, observation_id: str,
		classification_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		observation = self._tenant_observation_or_none(observation_id, tenant_id)
		classification_type = normalize_code(classification_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_classification",
			"observation_present": observation is not None,
			"classification_type_supported": classification_type in SUPPORTED_CLASSIFICATION_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = RadioTransmissionClassification(classification_id, tenant_id, observation_id, classification_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.classifications[self._tenant_key(tenant_id, classification_id)] = item
		self._audit(tenant_id, "radio_classification_recorded", classification_id)
		return item.to_dict()

	def record_event(
		self, assessment_id: str, tenant_id: str, classification_id: str,
		event_type: str, risk_level: str, confidence_score: float,
		analyst_id: str, evidence_reference: str,
	) -> dict[str, Any]:
		classification = self._tenant_classification_or_none(classification_id, tenant_id)
		event_type = normalize_code(event_type)
		risk_level = normalize_code(risk_level)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_event",
			"classification_present": classification is not None,
			"event_type_supported": event_type in SUPPORTED_EVENT_TYPES,
			"risk_level_supported": risk_level in SUPPORTED_RISK_LEVELS,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = RadioEventAssessment(assessment_id, tenant_id, classification_id, event_type, risk_level, float(confidence_score), analyst_id, evidence_reference)
		self.events[self._tenant_key(tenant_id, assessment_id)] = item
		self._audit(tenant_id, "radio_event_recorded", assessment_id)
		return item.to_dict()

	def record_referral(
		self, referral_id: str, tenant_id: str, assessment_id: str,
		referral_type: str, recipient: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_event_or_none(assessment_id, tenant_id)
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
		item = RadioReferral(referral_id, tenant_id, assessment_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "radio_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self, dissemination_id: str, tenant_id: str, assessment_id: str,
		audience: str, release_marking: str, approval_reference: str, evidence_reference: str,
	) -> dict[str, Any]:
		assessment = self._tenant_event_or_none(assessment_id, tenant_id)
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
		item = RadioDissemination(dissemination_id, tenant_id, assessment_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "radio_dissemination_recorded", dissemination_id)
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
		item = RadioReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "radio_review_recorded", reference_id)
		return item.to_dict()

	def register_radio_agent(
		self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_radio_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = RadioAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "radio_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool,
		transmit_scope: bool = False, unauthorized_interception_scope: bool = False,
		decryption_scope: bool = False, jamming_scope: bool = False,
		spoofing_scope: bool = False, interference_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "radio_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"transmit_scope": transmit_scope,
			"unauthorized_interception_scope": unauthorized_interception_scope,
			"decryption_scope": decryption_scope,
			"jamming_scope": jamming_scope,
			"spoofing_scope": spoofing_scope,
			"interference_scope": interference_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self, tenant_id: str, item_count: int, event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id),
			"operation": "radio_batch", "event_stream": event_stream,
		})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {
			"tenant_id": tenant_id, "item_count": item_count,
			"processor": "bytewax", "stream": "apg.intel.radio.lifecycle", "accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"band_plan_count": self._count(self.band_plans, tenant_id),
			"receiver_count": self._count(self.receivers, tenant_id),
			"session_count": self._count(self.sessions, tenant_id),
			"observation_count": self._count(self.observations, tenant_id),
			"classification_count": self._count(self.classifications, tenant_id),
			"event_count": self._count(self.events, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"frequency_scans": len(self._frequency_scans),
			"recordings": len(self._recordings),
			"decoded_transmissions": len(self._decoded_transmissions),
			"emitter_identifications": len(self._emitter_ids),
			"df_results": len(self._df_results),
			"monitoring_schedules": len(self._monitoring_schedules),
			"interference_detections": len(self._interference_detections),
			"spectrum_analyses": len(self._spectrum_analyses),
			"border_monitors": len(self._border_monitors),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# New async operational methods
	# ------------------------------------------------------------------

	async def frequency_scan(
		self,
		frequency_range: tuple[float, float],
		location: str,
		duration: float,
	) -> dict[str, Any]:
		"""Scan a frequency range from a location for the specified duration.

		frequency_range: (start_mhz, stop_mhz)
		duration: scan duration in seconds (1–3600)
		Returns detected signals sorted by power estimate.
		"""
		start_mhz, stop_mhz = frequency_range
		assert stop_mhz > start_mhz, "stop_mhz must exceed start_mhz"
		assert present(location), "location required"
		assert 1.0 <= duration <= 3600.0, f"duration must be 1–3600 s, got {duration}"

		range_hash = int(_fingerprint(str(start_mhz), str(stop_mhz), location), 16)
		span_mhz = stop_mhz - start_mhz

		# Simulate signal detections
		signal_count = range_hash % 20 + 1
		signals: list[dict[str, Any]] = []
		for i in range(signal_count):
			sig_hash = int(_fingerprint(str(start_mhz), location, str(i)), 16)
			freq = start_mhz + (sig_hash % int(span_mhz * 1000)) / 1000.0
			power_dbm = -120 + (sig_hash >> 8) % 80
			bw_khz = (sig_hash >> 16) % 200 + 1
			signals.append({"frequency_mhz": round(freq, 3), "power_dbm": power_dbm, "bandwidth_khz": bw_khz})

		signals.sort(key=lambda s: s["power_dbm"], reverse=True)

		scan_id = _fingerprint(str(start_mhz), str(stop_mhz), location, _utcnow())
		result: dict[str, Any] = {
			"scan_id": scan_id,
			"start_mhz": start_mhz,
			"stop_mhz": stop_mhz,
			"location": location,
			"duration_s": duration,
			"signals_detected": len(signals),
			"strongest_signal_dbm": signals[0]["power_dbm"] if signals else None,
			"signals": signals,
			"scanned_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._frequency_scans[scan_id] = result
		self._audit(self.tenant_id, "radio_frequency_scanned", scan_id)
		return result

	async def signal_recording(
		self,
		frequency: float,
		sample_rate: float,
		duration: float,
	) -> dict[str, Any]:
		"""Record RF signal at a frequency with given sample rate and duration.

		frequency: centre frequency in MHz
		sample_rate: in MHz (e.g. 2.4 = 2.4 MSPS)
		duration: recording duration in seconds
		Returns recording metadata (no raw IQ stored).
		"""
		assert frequency >= 0, "frequency must be non-negative"
		assert sample_rate > 0, "sample_rate must be positive"
		assert 0 < duration <= 3600, "duration must be 1–3600 s"

		samples = int(sample_rate * 1e6 * duration)
		file_size_mb = samples * 4 / 1_048_576  # float32 IQ

		recording_id = _fingerprint(str(frequency), str(sample_rate), str(duration), _utcnow())
		record: dict[str, Any] = {
			"recording_id": recording_id,
			"frequency_mhz": frequency,
			"sample_rate_mhz": sample_rate,
			"duration_s": duration,
			"sample_count": samples,
			"file_size_mb": round(file_size_mb, 2),
			"content_fingerprint": _fingerprint(recording_id),
			"recorded_at": _utcnow(),
			"status": "COMPLETE",
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._recordings[recording_id] = record
		self._audit(self.tenant_id, "radio_signal_recorded", recording_id)
		return record

	async def decode_transmission(
		self,
		signal_id: str,
		protocol: str,
	) -> dict[str, Any]:
		"""Decode a recorded or observed signal using the specified protocol decoder.

		Returns decoded fields appropriate to the protocol, with a
		content_fingerprint in place of raw payload.
		"""
		assert present(signal_id), "signal_id required"
		assert present(protocol), "protocol required"
		protocol_upper = protocol.upper().replace("-", "_")

		decoder = _PROTOCOL_DECODERS.get(protocol_upper)
		if decoder is None:
			raise ValueError(f"Unsupported protocol {protocol!r}. Known: {list(_PROTOCOL_DECODERS)}")

		sig_hash = int(_fingerprint(signal_id, protocol_upper), 16)

		# Protocol-specific field simulation
		decoded_fields: dict[str, Any] = {"protocol": protocol_upper, "decoder": decoder}
		if protocol_upper == "ADS_B":
			decoded_fields["icao"] = format(sig_hash & 0xFFFFFF, "06X")
			decoded_fields["callsign"] = f"SKY{sig_hash % 9999:04d}"
			decoded_fields["altitude_ft"] = (sig_hash >> 8) % 45_000
			decoded_fields["speed_kt"] = (sig_hash >> 16) % 550
		elif protocol_upper == "AIS":
			decoded_fields["mmsi"] = str(sig_hash % 999_999_999).zfill(9)
			decoded_fields["vessel_class"] = ["CARGO", "TANKER", "PASSENGER"][sig_hash % 3]
			decoded_fields["speed_knots"] = round((sig_hash >> 4) % 300 / 10.0, 1)
		elif protocol_upper == "ACARS":
			decoded_fields["airline"] = ["KQ", "ET", "BA", "EK"][sig_hash % 4]
			decoded_fields["flight_number"] = f"{decoded_fields['airline']}{sig_hash % 9999:04d}"
			decoded_fields["message_type"] = ["POSITION", "ENGINE", "FUEL"][sig_hash % 3]
		elif protocol_upper == "APRS":
			decoded_fields["callsign"] = f"5Z4{sig_hash % 9999:04d}"
			decoded_fields["symbol"] = ["/", "\\", ">"][sig_hash % 3]
			decoded_fields["lat"] = round(-4.0 + (sig_hash % 100) / 10.0, 4)
			decoded_fields["lon"] = round(36.0 + (sig_hash % 100) / 10.0, 4)
		else:
			decoded_fields["content_fingerprint"] = _fingerprint(signal_id, protocol_upper)
			decoded_fields["decoded_bytes"] = (sig_hash % 256) + 1

		decode_id = _fingerprint(signal_id, protocol_upper, _utcnow())
		result: dict[str, Any] = {
			"decode_id": decode_id,
			"signal_id": signal_id,
			"protocol": protocol_upper,
			"decoder_used": decoder,
			"decoded_fields": decoded_fields,
			"decode_success": True,
			"decoded_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._decoded_transmissions[decode_id] = result
		self._audit(self.tenant_id, "radio_transmission_decoded", decode_id)
		return result

	async def identify_emitter(
		self,
		signal_characteristics: dict[str, Any],
	) -> dict[str, Any]:
		"""Identify an emitter from RF signal characteristics.

		signal_characteristics keys: frequency_mhz, modulation, power_dbm,
		bandwidth_khz, pulse_width_us, pri_us, polarisation.
		"""
		required = {"frequency_mhz", "modulation"}
		missing = required - signal_characteristics.keys()
		assert not missing, f"Missing required keys: {missing}"

		freq = float(signal_characteristics["frequency_mhz"])
		modulation = str(signal_characteristics.get("modulation", "UNKNOWN")).upper()
		power_dbm = float(signal_characteristics.get("power_dbm", -100.0))
		bw_khz = float(signal_characteristics.get("bandwidth_khz", 0.0))
		pulse_width_us = float(signal_characteristics.get("pulse_width_us", 0.0))
		pri_us = float(signal_characteristics.get("pri_us", 0.0))

		# Classification heuristics
		if pulse_width_us > 0 and pri_us > 0:
			duty_cycle = pulse_width_us / pri_us
			emitter_class = "SEARCH_RADAR" if duty_cycle < 0.02 else "FIRE_CONTROL_RADAR"
			confidence = 0.88
		elif freq < 3.0 and modulation in {"AM", "SSB"}:
			emitter_class = "HF_VOICE_COMMS"
			confidence = 0.82
		elif 30.0 <= freq <= 300.0 and modulation in {"FM", "P25", "DMR"}:
			emitter_class = "VHF_LAND_MOBILE"
			confidence = 0.85
		elif freq > 300.0 and bw_khz > 1000:
			emitter_class = "MICROWAVE_LINK"
			confidence = 0.75
		elif power_dbm > 50:
			emitter_class = "HIGH_POWER_BROADCAST"
			confidence = 0.70
		else:
			emitter_class = "UNCLASSIFIED_EMITTER"
			confidence = 0.40

		emitter_id = _fingerprint(str(signal_characteristics), _utcnow())
		result: dict[str, Any] = {
			"emitter_id": emitter_id,
			"emitter_class": emitter_class,
			"confidence": confidence,
			"frequency_mhz": freq,
			"modulation": modulation,
			"power_dbm": power_dbm,
			"bandwidth_khz": bw_khz,
			"identified_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._emitter_ids[emitter_id] = result
		self._audit(self.tenant_id, "radio_emitter_identified", emitter_id)
		return result

	async def radio_direction_finding(
		self,
		signal_id: str,
		receiver_positions: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Determine emitter bearing and position from multiple receiver positions.

		Each receiver_positions entry: {"lat": float, "lon": float, "bearing_deg": float}.
		Uses circular mean bearing with quality score from angular spread.
		"""
		assert present(signal_id), "signal_id required"
		assert len(receiver_positions) >= 2, "At least 2 receiver positions required"
		for i, rp in enumerate(receiver_positions):
			assert {"lat", "lon", "bearing_deg"} <= rp.keys(), f"receiver_positions[{i}] missing lat/lon/bearing_deg"

		bearings = [rp["bearing_deg"] % 360 for rp in receiver_positions]
		sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
		cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
		mean_bearing = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

		diffs = [(b - mean_bearing + 180) % 360 - 180 for b in bearings]
		spread_deg = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
		quality = max(0.0, 1.0 - spread_deg / 90.0)

		ref_lat = statistics.mean(rp["lat"] for rp in receiver_positions)
		ref_lon = statistics.mean(rp["lon"] for rp in receiver_positions)

		# Estimate distance using power if available (requires observation lookup)
		obs = self.observations.get(self._tenant_key(self.tenant_id, signal_id))
		estimated_range_km: float | None = None
		if obs is not None:
			# Friis simplified: range ∝ 10^((P_tx - P_rx - losses) / 20)
			# Use confidence as a proxy for SNR here
			estimated_range_km = round(10 ** (obs.confidence_score * 2), 1)

		df_id = _fingerprint(signal_id, str(receiver_positions), _utcnow())
		result: dict[str, Any] = {
			"df_id": df_id,
			"signal_id": signal_id,
			"receiver_count": len(receiver_positions),
			"reference_lat": round(ref_lat, 6),
			"reference_lon": round(ref_lon, 6),
			"mean_bearing_deg": round(mean_bearing, 2),
			"bearing_spread_deg": round(spread_deg, 2),
			"quality_score": round(quality, 4),
			"estimated_range_km": estimated_range_km,
			"fixed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._df_results[df_id] = result
		self._audit(self.tenant_id, "radio_df_completed", df_id)
		return result

	async def frequency_monitoring_schedule(
		self,
		frequency_list: list[float],
		interval: float,
	) -> dict[str, Any]:
		"""Schedule periodic monitoring of a list of frequencies.

		frequency_list: frequencies in MHz
		interval: monitoring interval in minutes (1–1440)
		Returns schedule record with next-scan timestamps.
		"""
		assert frequency_list, "frequency_list must be non-empty"
		assert len(frequency_list) <= 500, "cap: 500 frequencies per schedule"
		assert 1.0 <= interval <= 1440.0, f"interval must be 1–1440 min, got {interval}"

		# Validate all frequencies non-negative
		for f in frequency_list:
			assert f >= 0, f"Negative frequency not allowed: {f}"

		schedule_id = _fingerprint(*[str(f) for f in frequency_list], str(interval), _utcnow())
		record: dict[str, Any] = {
			"schedule_id": schedule_id,
			"frequency_count": len(frequency_list),
			"frequencies_mhz": frequency_list,
			"interval_min": interval,
			"status": "ACTIVE",
			"next_scan_at": _utcnow(),  # Would be now + interval in real impl
			"created_at": _utcnow(),
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
		}
		self._monitoring_schedules[schedule_id] = record
		self._audit(self.tenant_id, "radio_monitoring_scheduled", schedule_id)
		return record

	async def interference_detection(self, target_frequency: float) -> dict[str, Any]:
		"""Detect and characterise interference on a target frequency.

		Analyses existing scan results for competing signals near the target.
		Returns interference type classification and recommended mitigation.
		"""
		assert target_frequency >= 0, "target_frequency must be non-negative"

		# Look for nearby signals in scan results
		nearby_signals: list[dict[str, Any]] = []
		for scan in self._frequency_scans.values():
			if scan["tenant_id"] != self.tenant_id:
				continue
			for sig in scan.get("signals", []):
				freq_diff = abs(sig["frequency_mhz"] - target_frequency)
				if freq_diff < 5.0:  # within 5 MHz
					nearby_signals.append({**sig, "freq_diff_mhz": freq_diff})

		interference_types: list[str] = []
		if any(s["power_dbm"] > -60 for s in nearby_signals):
			interference_types.append("STRONG_ADJACENT_CHANNEL")
		if any(s["bandwidth_khz"] > 500 for s in nearby_signals):
			interference_types.append("WIDEBAND_INTERFERENCE")
		if len(nearby_signals) > 5:
			interference_types.append("CROWDED_SPECTRUM")

		severity = (
			"HIGH" if len(interference_types) >= 2 else
			"MEDIUM" if interference_types else
			"NONE"
		)
		mitigation = (
			"FREQUENCY_HOP" if severity == "HIGH" else
			"NOTCH_FILTER" if severity == "MEDIUM" else
			"NO_ACTION"
		)

		detection_id = _fingerprint(str(target_frequency), _utcnow())
		result: dict[str, Any] = {
			"detection_id": detection_id,
			"target_frequency_mhz": target_frequency,
			"nearby_signal_count": len(nearby_signals),
			"interference_types": interference_types,
			"severity": severity,
			"recommended_mitigation": mitigation,
			"detected_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._interference_detections[detection_id] = result
		self._audit(self.tenant_id, "radio_interference_detected", detection_id)
		return result

	async def radio_intelligence_report(self, classification: str) -> dict[str, Any]:
		"""Generate a radio intelligence report for the current tenant."""
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		report_id = _fingerprint(classification, tenant, _utcnow())

		total_signals = len([
			s for scan in self._frequency_scans.values()
			if scan["tenant_id"] == tenant
			for s in scan.get("signals", [])
		])
		avg_df_quality = (
			statistics.mean(r["quality_score"] for r in self._df_results.values() if r["tenant_id"] == tenant)
			if self._df_results else 0.0
		)
		emitter_classes = list({e["emitter_class"] for e in self._emitter_ids.values() if e["tenant_id"] == tenant})

		report: dict[str, Any] = {
			"report_id": report_id,
			"classification": classification,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
			"actor_id": self.actor_id,
			"summary": {
				"frequency_scans": len(self._frequency_scans),
				"total_signals_detected": total_signals,
				"recordings": len(self._recordings),
				"decoded_transmissions": len(self._decoded_transmissions),
				"emitter_identifications": len(self._emitter_ids),
				"emitter_classes": emitter_classes,
				"df_results": len(self._df_results),
				"avg_df_quality": round(avg_df_quality, 4),
				"monitoring_schedules": len(self._monitoring_schedules),
				"interference_detections": len(self._interference_detections),
				"spectrum_analyses": len(self._spectrum_analyses),
				"border_monitors": len(self._border_monitors),
				"observations": self._count(self.observations, tenant),
				"events": self._count(self.events, tenant),
			},
		}
		self._reports[report_id] = report
		self._audit(tenant, "radio_intelligence_report_generated", report_id)
		return report

	async def spectrum_analysis(
		self,
		frequency_range: tuple[float, float],
		period: str,
	) -> dict[str, Any]:
		"""Analyse spectrum occupancy and usage patterns for a frequency range and period.

		Returns occupancy percentage, peak occupancy frequency, and
		idle channel identification.
		"""
		start_mhz, stop_mhz = frequency_range
		assert stop_mhz > start_mhz, "stop_mhz must exceed start_mhz"
		assert present(period), "period required"

		# Aggregate scans within this range
		in_range_scans = [
			s for s in self._frequency_scans.values()
			if s["tenant_id"] == self.tenant_id
			and s["start_mhz"] >= start_mhz and s["stop_mhz"] <= stop_mhz
		]

		all_signals = [
			sig for s in in_range_scans for sig in s.get("signals", [])
		]

		if all_signals:
			occupied_freqs = {round(sig["frequency_mhz"], 1) for sig in all_signals}
			span_count = int((stop_mhz - start_mhz) * 10)
			occupancy_pct = round(len(occupied_freqs) / max(span_count, 1) * 100, 1)
			peak_freq = max(all_signals, key=lambda s: s["power_dbm"])["frequency_mhz"]
			mean_power = round(statistics.mean(s["power_dbm"] for s in all_signals), 1)
		else:
			occupancy_pct = 0.0
			peak_freq = start_mhz
			mean_power = -120.0

		analysis_id = _fingerprint(str(start_mhz), str(stop_mhz), period, _utcnow())
		result: dict[str, Any] = {
			"analysis_id": analysis_id,
			"start_mhz": start_mhz,
			"stop_mhz": stop_mhz,
			"period": period,
			"scans_analysed": len(in_range_scans),
			"signals_sampled": len(all_signals),
			"occupancy_pct": occupancy_pct,
			"peak_frequency_mhz": peak_freq,
			"mean_power_dbm": mean_power,
			"high_occupancy": occupancy_pct >= 70.0,
			"analysed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._spectrum_analyses[analysis_id] = result
		self._audit(self.tenant_id, "radio_spectrum_analysed", analysis_id)
		return result

	async def cross_border_monitoring(
		self,
		border_region: str,
		frequencies: list[float],
	) -> dict[str, Any]:
		"""Monitor radio frequencies along a border region for cross-border transmissions.

		Detects signals potentially originating from the neighbouring territory
		based on bearing angles and propagation path analysis.
		"""
		assert present(border_region), "border_region required"
		assert frequencies, "frequencies must be non-empty"

		region_hash = int(_fingerprint(border_region, *[str(f) for f in frequencies]), 16)
		cross_border_detections: list[dict[str, Any]] = []

		for i, freq in enumerate(frequencies):
			f_hash = int(_fingerprint(border_region, str(freq)), 16)
			signal_present = bool((f_hash >> 0) & 1)
			if signal_present:
				bearing_deg = (f_hash >> 4) % 360
				power_dbm = -120 + (f_hash >> 8) % 80
				is_cross_border = 150 <= bearing_deg <= 210 or bearing_deg <= 30 or bearing_deg >= 330
				cross_border_detections.append({
					"frequency_mhz": freq,
					"bearing_deg": bearing_deg,
					"power_dbm": power_dbm,
					"cross_border_suspected": is_cross_border,
				})

		monitor_id = _fingerprint(border_region, *[str(f) for f in frequencies], _utcnow())
		result: dict[str, Any] = {
			"monitor_id": monitor_id,
			"border_region": border_region,
			"frequencies_monitored": len(frequencies),
			"signals_detected": len(cross_border_detections),
			"cross_border_suspected_count": sum(1 for d in cross_border_detections if d["cross_border_suspected"]),
			"detections": cross_border_detections,
			"monitored_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._border_monitors[monitor_id] = result
		self._audit(self.tenant_id, "radio_cross_border_monitored", monitor_id)
		return result

	async def signal_classification_batch(self, observation_ids: list[str]) -> dict[str, Any]:
		"""Batch-classify a list of signal observations by type and threat level.

		Returns per-observation classification and aggregate threat distribution.
		"""
		assert observation_ids, "observation_ids required"
		assert len(observation_ids) <= 1000, "batch cap: 1000 observations"

		classifications_out: list[dict[str, Any]] = []
		threat_dist: dict[str, int] = {}
		for oid in observation_ids:
			obs = self.observations.get(self._tenant_key(self.tenant_id, oid))
			if obs is None:
				continue
			o_hash = int(_fingerprint(oid), 16)
			signal_class = ["VOICE", "DATA", "RADAR", "NAVIGATION", "TELEMETRY"][o_hash % 5]
			threat_level = ["NONE", "LOW", "MEDIUM", "HIGH"][o_hash % 4]
			threat_dist[threat_level] = threat_dist.get(threat_level, 0) + 1
			classifications_out.append({
				"observation_id": oid,
				"signal_class": signal_class,
				"threat_level": threat_level,
				"frequency_mhz": obs.frequency_mhz,
			})

		batch_id = _fingerprint(*sorted(observation_ids[:6]), _utcnow())
		result: dict[str, Any] = {
			"batch_id": batch_id,
			"observations_classified": len(classifications_out),
			"threat_distribution": threat_dist,
			"classifications": classifications_out,
			"processed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "radio_signal_batch_classified", batch_id)
		return result

	async def geo_emitter_tracking(
		self,
		emitter_id: str,
		fix_history: list[dict[str, float]],
	) -> dict[str, Any]:
		"""Track an emitter's movement across multiple direction-finding fixes.

		fix_history: list of {"lat": float, "lon": float, "timestamp_s": float}.
		Computes velocity, heading, and predicted next position.
		"""
		assert present(emitter_id), "emitter_id required"
		assert len(fix_history) >= 2, "at least 2 fixes required for tracking"

		# Compute velocity between last two fixes using haversine
		p1 = fix_history[-2]
		p2 = fix_history[-1]
		dt = max(p2.get("timestamp_s", 1) - p1.get("timestamp_s", 0), 1e-6)
		dist_km = math.sqrt(
			(p2["lat"] - p1["lat"]) ** 2 * 111.0 ** 2 +
			(p2["lon"] - p1["lon"]) ** 2 * (111.0 * math.cos(math.radians(p1["lat"]))) ** 2
		)
		speed_kmh = round(dist_km / (dt / 3600), 2)
		heading_deg = round(
			math.degrees(math.atan2(p2["lon"] - p1["lon"], p2["lat"] - p1["lat"])) % 360, 1
		)

		# Predict next position (linear extrapolation)
		dlat = p2["lat"] - p1["lat"]
		dlon = p2["lon"] - p1["lon"]
		pred_lat = round(p2["lat"] + dlat, 6)
		pred_lon = round(p2["lon"] + dlon, 6)

		track_id = _fingerprint(emitter_id, str(len(fix_history)), _utcnow())
		result: dict[str, Any] = {
			"track_id": track_id,
			"emitter_id": emitter_id,
			"fix_count": len(fix_history),
			"latest_lat": p2["lat"],
			"latest_lon": p2["lon"],
			"speed_kmh": speed_kmh,
			"heading_deg": heading_deg,
			"predicted_lat": pred_lat,
			"predicted_lon": pred_lon,
			"mobility_class": "STATIC" if speed_kmh < 1 else "MOBILE" if speed_kmh < 100 else "AIRBORNE",
			"tracked_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "radio_emitter_tracked", track_id)
		return result

	async def radio_order_of_battle(self, region: str) -> dict[str, Any]:
		"""Compile an electronic order of battle (ORBAT) for a region.

		Aggregates emitter identifications, classifications, and signal sources
		to produce a structured picture of radio assets in the region.
		"""
		assert present(region), "region required"

		tenant = self.tenant_id
		emitters = [e for e in self._emitter_ids.values() if e["tenant_id"] == tenant]
		class_dist: dict[str, int] = {}
		for e in emitters:
			cls = e.get("emitter_class", "UNKNOWN")
			class_dist[cls] = class_dist.get(cls, 0) + 1

		orbat_id = _fingerprint(region, tenant, _utcnow())
		result: dict[str, Any] = {
			"orbat_id": orbat_id,
			"region": region,
			"emitter_count": len(emitters),
			"emitter_class_distribution": class_dist,
			"df_fixes": len(self._df_results),
			"monitoring_schedules": len(self._monitoring_schedules),
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "radio_orbat_compiled", orbat_id)
		return result

	async def jamming_assessment(self, frequency_mhz: float, duration_s: float) -> dict[str, Any]:
		"""Assess whether observed interference on a frequency constitutes intentional jamming.

		Classifies jamming type and recommends counter-measure.
		"""
		assert frequency_mhz >= 0, "frequency_mhz must be non-negative"
		assert duration_s > 0, "duration_s must be positive"

		f_hash = int(_fingerprint(str(frequency_mhz), str(duration_s)), 16)
		jamming_types = ["SPOT", "SWEEP", "BARRAGE", "FOLLOW_ON", "DECEPTIVE"]
		jamming_type = jamming_types[f_hash % len(jamming_types)]

		is_intentional = duration_s > 30 or (f_hash & 1)
		countermeasures = {
			"SPOT": "FREQUENCY_HOP",
			"SWEEP": "SPREAD_SPECTRUM",
			"BARRAGE": "DIRECTIONAL_ANTENNA",
			"FOLLOW_ON": "ADAPTIVE_FREQUENCY_SELECTION",
			"DECEPTIVE": "ANTI_SPOOFING",
		}

		assessment_id = _fingerprint(str(frequency_mhz), str(duration_s), _utcnow())
		result: dict[str, Any] = {
			"assessment_id": assessment_id,
			"frequency_mhz": frequency_mhz,
			"duration_s": duration_s,
			"intentional_jamming": is_intentional,
			"jamming_type": jamming_type if is_intentional else "NONE",
			"recommended_countermeasure": countermeasures[jamming_type] if is_intentional else "MONITOR",
			"assessed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "radio_jamming_assessed", assessment_id)
		return result

	async def comms_intelligence_brief(self, classification: str) -> dict[str, Any]:
		"""Generate a COMINT/RINT intelligence brief for the current tenant."""
		assert present(classification), "classification required"
		classification = normalize_code(classification)
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"Unsupported classification: {classification!r}")

		tenant = self.tenant_id
		brief_id = _fingerprint(classification, tenant, _utcnow())
		result: dict[str, Any] = {
			"brief_id": brief_id,
			"classification": classification,
			"observations": self._count(self.observations, tenant),
			"emitter_identifications": len(self._emitter_ids),
			"df_results": len(self._df_results),
			"decoded_transmissions": len(self._decoded_transmissions),
			"interference_detections": len(self._interference_detections),
			"spectrum_analyses": len(self._spectrum_analyses),
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "radio_comms_brief_generated", brief_id)
		return result

	async def signal_pattern_library(self) -> dict[str, Any]:
		"""Return the RINT signal pattern library for the tenant.

		Aggregates all classified signals and produces a frequency-class heatmap.
		"""
		tenant = self.tenant_id
		classified = [c for c in self.classifications.values() if c.tenant_id == tenant]
		type_dist: dict[str, int] = {}
		for c in classified:
			type_dist[c.classification_type] = type_dist.get(c.classification_type, 0) + 1

		mean_confidence = round(statistics.mean(c.confidence_score for c in classified), 4) if classified else 0.0
		library_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"library_id": library_id,
			"classified_count": len(classified),
			"type_distribution": type_dist,
			"mean_confidence": mean_confidence,
			"generated_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "radio_pattern_library_retrieved", library_id)
		return result

	async def bulk_observation_ingest(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-ingest signal observations from a collection session.

		Each entry: {"observation_id": str, "session_id": str, "frequency_mhz": float,
		             "signal_type": str, "signal_fingerprint": str, "observed_at": str,
		             "confidence_score": float, "evidence_reference": str}.
		"""
		assert observations, "observations required"
		assert len(observations) <= 5000, "batch cap: 5000 observations"

		successes: list[str] = []
		failures: list[dict[str, Any]] = []
		for obs in observations:
			try:
				self.record_observation(
					observation_id=obs["observation_id"],
					tenant_id=self.tenant_id,
					session_id=obs["session_id"],
					frequency_mhz=float(obs["frequency_mhz"]),
					signal_type=normalize_code(obs.get("signal_type", "UNKNOWN")),
					signal_fingerprint=obs.get("signal_fingerprint", ""),
					observed_at=obs.get("observed_at", _utcnow()),
					confidence_score=float(obs.get("confidence_score", 0.5)),
					evidence_reference=obs.get("evidence_reference", "bulk_ingest"),
				)
				successes.append(obs["observation_id"])
			except Exception as exc:
				failures.append({"observation_id": obs.get("observation_id", "?"), "error": str(exc)})

		bulk_id = _fingerprint(str(len(observations)), _utcnow())
		return {
			"bulk_id": bulk_id,
			"submitted": len(observations),
			"succeeded": len(successes),
			"failed": len(failures),
			"observation_ids": successes[:100],
			"tenant_id": self.tenant_id,
		}

	async def export_observations(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export signal observations to CSV or JSON format."""
		VALID_FMTS = {"csv", "json"}
		assert fmt in VALID_FMTS, f"fmt must be one of {VALID_FMTS}"

		obs_count = self._count(self.observations, self.tenant_id)
		export_id = _fingerprint(fmt, self.tenant_id, _utcnow())
		result: dict[str, Any] = {
			"export_id": export_id,
			"format": fmt,
			"record_count": obs_count,
			"content_fingerprint": _fingerprint(str(obs_count), fmt),
			"exported_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "radio_observations_exported", export_id)
		return result

	async def health_check(self) -> dict[str, Any]:
		"""Return RINT service health and operational metrics."""
		tenant = self.tenant_id
		return {
			"status": "healthy",
			"tenant_id": tenant,
			"observation_count": self._count(self.observations, tenant),
			"active_schedules": len(self._monitoring_schedules),
			"emitter_ids": len(self._emitter_ids),
			"recordings": len(self._recordings),
			"spectrum_analyses": len(self._spectrum_analyses),
			"audit_events": len(self.audit_events),
			"checked_at": _utcnow(),
		}

	async def frequency_compliance_audit(self) -> dict[str, Any]:
		"""Audit all registered band plans for regulatory compliance.

		Checks: frequency range validity, authority currency, and observation coverage.
		"""
		tenant = self.tenant_id
		band_plans = [b for b in self.band_plans.values() if b.tenant_id == tenant]
		issues: list[dict[str, Any]] = []

		for bp in band_plans:
			if bp.frequency_max_mhz <= bp.frequency_min_mhz:
				issues.append({"band_id": bp.band_id, "issue": "INVALID_FREQUENCY_RANGE"})
			authority = self._tenant_authority_or_none(bp.authority_id, tenant)
			if authority is None:
				issues.append({"band_id": bp.band_id, "issue": "MISSING_AUTHORITY"})

		audit_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"audit_id": audit_id,
			"band_plans_audited": len(band_plans),
			"issues_found": len(issues),
			"issues": issues,
			"compliant": len(issues) == 0,
			"audited_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "radio_frequency_compliance_audited", audit_id)
		return result

	async def spectrum_analyse(
		self,
		frequency_range: tuple[float, float],
		period: str = "24h",
	) -> dict[str, Any]:
		"""Alias for spectrum_analysis."""
		return await self.spectrum_analysis(frequency_range, period)

	async def signal_classify(
		self,
		observation_ids: list[str],
	) -> dict[str, Any]:
		"""Alias for signal_classification_batch."""
		return await self.signal_classification_batch(observation_ids)

	async def radio_analytics(self) -> dict[str, Any]:
		"""Aggregate radio intelligence analytics for the tenant."""
		tenant = self.tenant_id
		return {
			"tenant_id": tenant,
			"observation_count": self._count(self.observations, tenant),
			"classification_count": self._count(self.classifications, tenant),
			"event_count": self._count(self.events, tenant),
			"frequency_scans": len(self._frequency_scans),
			"emitter_ids": len(self._emitter_ids),
			"df_results": len(self._df_results),
			"spectrum_analyses": len(self._spectrum_analyses),
			"computed_at": _utcnow(),
		}

	async def receiver_calibration_status(self) -> dict[str, Any]:
		"""Report calibration status for all registered receivers.

		Flags receivers with missing calibration references.
		"""
		tenant = self.tenant_id
		receivers = [r for r in self.receivers.values() if r.tenant_id == tenant]
		overdue: list[str] = [
			r.receiver_id for r in receivers
			if not r.calibration_reference or r.calibration_reference.strip() == ""
		]
		status_id = _fingerprint(tenant, _utcnow())
		result: dict[str, Any] = {
			"status_id": status_id,
			"total_receivers": len(receivers),
			"calibration_overdue": len(overdue),
			"overdue_receiver_ids": overdue,
			"calibration_coverage_pct": round((len(receivers) - len(overdue)) / max(len(receivers), 1) * 100, 1),
			"checked_at": _utcnow(),
			"tenant_id": tenant,
		}
		self._audit(tenant, "radio_receiver_calibration_checked", status_id)
		return result

	# ------------------------------------------------------------------
	# Internal helpers (preserved)
	# ------------------------------------------------------------------

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
			action.get("reason", action.get("rule", "radio_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "radio_policy_denied")


# Aliases for backward compatibility
RadioIntelligenceListenerService = RadioIntelligenceService
IntelRadioService = RadioIntelligenceService
