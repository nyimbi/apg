"""Async service layer for APG Telemedicine."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_BILLING_CODES, SUPPORTED_CONSULTATION_TYPES,
	SUPPORTED_MONITORING_DEVICE_TYPES, SUPPORTED_PLATFORM_TYPES,
	SUPPORTED_PRESCRIPTION_TRANSMISSION_METHODS, SUPPORTED_SESSION_STATUSES,
	evaluate_capability_rules, get_capability_contract,
)
from .models import (
	ConsultationCreate, ConsultationResponse,
	PrescriptionTransmitCreate, PrescriptionTransmitResponse,
	RemoteMonitoringEnrollmentCreate, RemoteMonitoringEnrollmentResponse,
	TeleBillingCreate, TeleBillingResponse,
	TeleSessionCreate, TeleSessionResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("tel.%s tenant=%s id=%s", op, tid, eid)


def _log_session_start(session_id: str, patient_id: str, provider_id: str) -> None:
	logger.info("tel.session_started session=%s patient=%s provider=%s", session_id, patient_id, provider_id)


def _log_vital_alert(patient_id: str, vital_type: str, value: float, threshold: float) -> None:
	logger.warning("tel.vital_alert patient=%s vital=%s value=%s threshold=%s", patient_id, vital_type, value, threshold)


class PolicyViolationError(ValueError):
	pass


class TelemedicineService:
	"""Tenant-scoped telemedicine runtime."""

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
		self._tenant_id = tenant_id
		self._actor_id = actor_id
		self._consultations: dict[tuple[str, str], ConsultationResponse] = {}
		self._sessions: dict[tuple[str, str], TeleSessionResponse] = {}
		self._monitoring: dict[tuple[str, str], RemoteMonitoringEnrollmentResponse] = {}
		self._prescriptions: dict[tuple[str, str], PrescriptionTransmitResponse] = {}
		self._billing: dict[tuple[str, str], TeleBillingResponse] = {}
		self._vital_readings: list[dict[str, Any]] = []
		self._telemonitoring_alerts: list[dict[str, Any]] = []
		self._provider_schedules: dict[str, list[dict[str, Any]]] = {}
		self._audit_events: list[dict[str, Any]] = []

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ── consultations ─────────────────────────────────────────────────────────

	async def book_consultation(self, payload: ConsultationCreate) -> ConsultationResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "book_consultation",
			"consultation_type_supported": payload.consultation_type in SUPPORTED_CONSULTATION_TYPES,
			"platform_type_supported": payload.platform in SUPPORTED_PLATFORM_TYPES,
		})
		consult = ConsultationResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			provider_id=payload.provider_id, consultation_type=payload.consultation_type,
			scheduled_at=payload.scheduled_at, duration_minutes=payload.duration_minutes,
			chief_complaint=payload.chief_complaint, platform=payload.platform,
			patient_consent_obtained=payload.patient_consent_obtained,
			e911_disclosure_provided=payload.e911_disclosure_provided,
			status="scheduled", created_by=payload.created_by,
		)
		self._consultations[(payload.tenant_id, consult.id)] = consult
		self._audit(payload.tenant_id, "consultation_booked", consult.id)
		_log_op("book_consultation", payload.tenant_id, consult.id)
		return consult

	async def book_teleconsult(
		self,
		patient_id: str,
		provider_id: str,
		appointment_type: str,
		preferred_time: datetime,
	) -> dict[str, Any]:
		"""Book a telemedicine consultation, checking provider availability."""
		assert patient_id, "patient_id required"
		assert provider_id, "provider_id required"
		assert appointment_type in SUPPORTED_CONSULTATION_TYPES, f"unsupported type: {appointment_type}"
		tenant_id = self._tenant_id
		avail = self._provider_schedules.get(provider_id, [])
		conflict = any(
			abs((datetime.fromisoformat(s["time"]) - preferred_time).total_seconds()) < 1800
			for s in avail if s.get("status") == "booked"
		)
		appt_id = uuid7str()
		join_token = uuid7str()[:16]
		record: dict[str, Any] = {
			"id": appt_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"provider_id": provider_id,
			"appointment_type": appointment_type,
			"preferred_time": preferred_time.isoformat(),
			"platform": "video",
			"join_token": join_token,
			"join_url": f"https://telehealth.example.com/room/{join_token}",
			"has_conflict": conflict,
			"status": "pending_confirmation" if conflict else "confirmed",
			"booked_by": self._actor_id,
			"booked_at": datetime.utcnow().isoformat(),
			"consent_required": True,
			"e911_disclosure_required": True,
		}
		slot = {"time": preferred_time.isoformat(), "status": "booked", "appointment_id": appt_id}
		self._provider_schedules.setdefault(provider_id, []).append(slot)
		self._audit(tenant_id, "teleconsult_booked", appt_id)
		_log_op("book_teleconsult", tenant_id, appt_id)
		return record

	async def cancel_consultation(self, tenant_id: str, consult_id: str) -> ConsultationResponse | None:
		consult = self._consultations.get((tenant_id, consult_id))
		if consult is None:
			return None
		updated = consult.model_copy(update={"status": "cancelled", "updated_at": datetime.utcnow()})
		self._consultations[(tenant_id, consult_id)] = updated
		self._audit(tenant_id, "consultation_cancelled", consult_id)
		return updated

	async def get_consultation(self, tenant_id: str, consult_id: str) -> ConsultationResponse | None:
		return self._consultations.get((tenant_id, consult_id))

	async def list_consultations(self, tenant_id: str, patient_id: str | None = None, status: str | None = None) -> list[ConsultationResponse]:
		results = [c for (tid, _), c in self._consultations.items() if tid == tenant_id]
		if patient_id:
			results = [c for c in results if c.patient_id == patient_id]
		if status:
			results = [c for c in results if c.status == status]
		return sorted(results, key=lambda c: c.scheduled_at)

	# ── sessions ──────────────────────────────────────────────────────────────

	async def create_session(self, payload: TeleSessionCreate) -> TeleSessionResponse:
		consult = self._consultations.get((payload.tenant_id, payload.consultation_id))
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "start_session",
			"patient_consent_obtained": payload.patient_consent_obtained,
			"e911_disclosure_provided": payload.e911_disclosure_provided,
			"consultation_status": consult.status if consult else "unknown",
			"platform_type_supported": payload.platform in SUPPORTED_PLATFORM_TYPES,
		})
		session = TeleSessionResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			consultation_id=payload.consultation_id, patient_id=payload.patient_id,
			provider_id=payload.provider_id, platform=payload.platform,
			status="waiting",
			join_url=f"https://telehealth.example.com/room/{uuid7str()[:12]}",
			technical_check_completed=payload.technical_check_completed,
			created_by=payload.created_by,
		)
		self._sessions[(payload.tenant_id, session.id)] = session
		if consult:
			updated = consult.model_copy(update={"status": "in_progress", "session_id": session.id, "started_at": datetime.utcnow(), "updated_at": datetime.utcnow()})
			self._consultations[(payload.tenant_id, payload.consultation_id)] = updated
		_log_session_start(session.id, payload.patient_id, payload.provider_id)
		self._audit(payload.tenant_id, "session_started", session.id)
		return session

	async def video_session_start(
		self,
		appointment_id: str,
		patient_token: str,
	) -> dict[str, Any]:
		"""Start a video session for a confirmed appointment."""
		assert appointment_id, "appointment_id required"
		assert patient_token, "patient_token required"
		tenant_id = self._tenant_id
		session_id = uuid7str()
		provider_token = uuid7str()[:24]
		record: dict[str, Any] = {
			"session_id": session_id,
			"appointment_id": appointment_id,
			"tenant_id": tenant_id,
			"patient_token": patient_token,
			"provider_token": provider_token,
			"room_url": f"https://telehealth.example.com/room/{session_id}",
			"recording_enabled": False,
			"started_at": datetime.utcnow().isoformat(),
			"max_duration_minutes": 60,
			"status": "active",
		}
		self._audit(tenant_id, "video_session_started", session_id)
		_log_session_start(session_id, patient_token[:8], "provider")
		_log_op("video_session_start", tenant_id, session_id)
		return record

	async def video_session_end(
		self,
		appointment_id: str,
		duration_mins: int,
		notes: str,
	) -> dict[str, Any]:
		"""End a video session, recording duration and clinical notes."""
		assert appointment_id, "appointment_id required"
		assert duration_mins >= 0, "duration_mins must be non-negative"
		tenant_id = self._tenant_id
		session_id = uuid7str()
		record: dict[str, Any] = {
			"session_id": session_id,
			"appointment_id": appointment_id,
			"tenant_id": tenant_id,
			"duration_minutes": duration_mins,
			"clinical_notes": notes,
			"ended_at": datetime.utcnow().isoformat(),
			"billable_minutes": duration_mins,
			"status": "completed",
			"ended_by": self._actor_id,
		}
		self._audit(tenant_id, "video_session_ended", session_id)
		_log_op("video_session_end", tenant_id, appointment_id)
		return record

	async def complete_session(self, tenant_id: str, session_id: str) -> TeleSessionResponse | None:
		session = self._sessions.get((tenant_id, session_id))
		if session is None:
			return None
		ended = datetime.utcnow()
		dur = int((ended - (session.started_at or session.created_at)).total_seconds()) if session.started_at else 0
		updated = session.model_copy(update={"status": "completed", "ended_at": ended, "duration_seconds": dur, "updated_at": datetime.utcnow()})
		self._sessions[(tenant_id, session_id)] = updated
		self._audit(tenant_id, "session_completed", session_id)
		consult = next((c for (tid, _), c in self._consultations.items() if tid == tenant_id and c.session_id == session_id), None)
		if consult:
			self._consultations[(tenant_id, consult.id)] = consult.model_copy(update={"status": "completed", "ended_at": ended, "updated_at": datetime.utcnow()})
		return updated

	async def get_session(self, tenant_id: str, session_id: str) -> TeleSessionResponse | None:
		return self._sessions.get((tenant_id, session_id))

	async def list_sessions(self, tenant_id: str, patient_id: str | None = None) -> list[TeleSessionResponse]:
		results = [s for (tid, _), s in self._sessions.items() if tid == tenant_id]
		if patient_id:
			results = [s for s in results if s.patient_id == patient_id]
		return sorted(results, key=lambda s: s.created_at, reverse=True)

	# ── remote monitoring ─────────────────────────────────────────────────────

	async def enroll_monitoring(self, payload: RemoteMonitoringEnrollmentCreate) -> RemoteMonitoringEnrollmentResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "enroll_monitoring_device",
			"device_type_supported": payload.device_type in SUPPORTED_MONITORING_DEVICE_TYPES,
			"alert_threshold_configured": payload.alert_threshold_configured,
		})
		enrollment = RemoteMonitoringEnrollmentResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			device_type=payload.device_type, device_id=payload.device_id,
			alert_thresholds=payload.alert_thresholds, provider_id=payload.provider_id,
			status="active", created_by=payload.created_by,
		)
		self._monitoring[(payload.tenant_id, enrollment.id)] = enrollment
		self._audit(payload.tenant_id, "monitoring_enrolled", enrollment.id)
		_log_op("enroll_monitoring", payload.tenant_id, enrollment.id)
		return enrollment

	async def remote_monitoring_enrol(
		self,
		patient_id: str,
		device_ids: list[str],
		vital_types: list[str],
	) -> dict[str, Any]:
		"""Enrol a patient in remote monitoring for specified vitals and devices."""
		assert patient_id, "patient_id required"
		assert device_ids, "device_ids required"
		assert vital_types, "vital_types required"
		tenant_id = self._tenant_id
		enrolment_id = uuid7str()
		default_thresholds: dict[str, dict[str, float]] = {
			"heart_rate": {"low": 50.0, "high": 100.0},
			"blood_pressure_systolic": {"low": 90.0, "high": 140.0},
			"blood_pressure_diastolic": {"low": 60.0, "high": 90.0},
			"spo2": {"low": 94.0, "high": 100.0},
			"blood_glucose": {"low": 70.0, "high": 180.0},
			"temperature": {"low": 36.0, "high": 38.0},
			"weight": {"low": 0.0, "high": 9999.0},
		}
		configured_thresholds = {vt: default_thresholds.get(vt, {"low": 0.0, "high": 9999.0}) for vt in vital_types}
		record: dict[str, Any] = {
			"id": enrolment_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"device_ids": device_ids,
			"vital_types": vital_types,
			"thresholds": configured_thresholds,
			"enrolled_by": self._actor_id,
			"enrolled_at": datetime.utcnow().isoformat(),
			"review_frequency_days": 30,
			"status": "active",
		}
		self._audit(tenant_id, "remote_monitoring_enrolled", enrolment_id)
		_log_op("remote_monitoring_enrol", tenant_id, enrolment_id)
		return record

	async def vital_reading_ingest(
		self,
		patient_id: str,
		device_id: str,
		vital_type: str,
		value: float,
		timestamp: datetime,
	) -> dict[str, Any]:
		"""Ingest a vital reading and evaluate against alert thresholds."""
		assert patient_id, "patient_id required"
		assert device_id, "device_id required"
		assert vital_type, "vital_type required"
		tenant_id = self._tenant_id
		reading_id = uuid7str()
		thresholds: dict[str, dict[str, float]] = {
			"heart_rate": {"low": 50.0, "high": 100.0},
			"blood_pressure_systolic": {"low": 90.0, "high": 140.0},
			"spo2": {"low": 94.0, "high": 100.0},
			"blood_glucose": {"low": 70.0, "high": 180.0},
			"temperature": {"low": 36.0, "high": 38.0},
		}
		t = thresholds.get(vital_type, {"low": float("-inf"), "high": float("inf")})
		alert_triggered = value < t["low"] or value > t["high"]
		severity = "normal"
		if alert_triggered:
			deviation = max(abs(value - t["low"]) / max(t["low"], 1), abs(value - t["high"]) / max(t["high"], 1))
			severity = "critical" if deviation > 0.2 else "warning"
		reading: dict[str, Any] = {
			"id": reading_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"device_id": device_id,
			"vital_type": vital_type,
			"value": value,
			"timestamp": timestamp.isoformat(),
			"ingested_at": datetime.utcnow().isoformat(),
			"alert_triggered": alert_triggered,
			"severity": severity,
			"threshold_low": t["low"],
			"threshold_high": t["high"],
		}
		self._vital_readings.append(reading)
		if alert_triggered:
			_log_vital_alert(patient_id, vital_type, value, t["high"] if value > t["high"] else t["low"])
			self._audit(tenant_id, "vital_alert_triggered", reading_id)
		self._audit(tenant_id, "vital_reading_ingested", reading_id)
		return reading

	async def telemonitoring_alert(
		self,
		patient_id: str,
		vital_type: str,
		threshold_breached: str,
	) -> dict[str, Any]:
		"""Create and route a telemonitoring alert to the responsible provider."""
		assert patient_id, "patient_id required"
		assert vital_type, "vital_type required"
		assert threshold_breached in ("low", "high", "critical_low", "critical_high"), \
			f"invalid threshold_breached: {threshold_breached}"
		tenant_id = self._tenant_id
		alert_id = uuid7str()
		severity = "critical" if "critical" in threshold_breached else "warning"
		escalation_mins = 15 if severity == "critical" else 60
		alert: dict[str, Any] = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"vital_type": vital_type,
			"threshold_breached": threshold_breached,
			"severity": severity,
			"escalation_deadline": (datetime.utcnow() + timedelta(minutes=escalation_mins)).isoformat(),
			"created_at": datetime.utcnow().isoformat(),
			"status": "active",
			"notification_channels": ["sms", "app_push"] if severity == "critical" else ["app_push"],
		}
		self._telemonitoring_alerts.append(alert)
		self._audit(tenant_id, "telemonitoring_alert_created", alert_id)
		_log_op("telemonitoring_alert", tenant_id, alert_id)
		return alert

	async def list_monitoring(self, tenant_id: str, patient_id: str | None = None) -> list[RemoteMonitoringEnrollmentResponse]:
		results = [m for (tid, _), m in self._monitoring.items() if tid == tenant_id]
		if patient_id:
			results = [m for m in results if m.patient_id == patient_id]
		return sorted(results, key=lambda m: m.enrolled_at, reverse=True)

	# ── prescriptions ─────────────────────────────────────────────────────────

	async def transmit_prescription(self, payload: PrescriptionTransmitCreate) -> PrescriptionTransmitResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "transmit_prescription",
			"transmission_method_supported": payload.transmission_method in SUPPORTED_PRESCRIPTION_TRANSMISSION_METHODS,
			"drug_schedule": payload.drug_schedule,
			"in_person_visit_completed": payload.in_person_visit_completed,
		})
		rx = PrescriptionTransmitResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			consultation_id=payload.consultation_id, drug_name=payload.drug_name,
			drug_schedule=payload.drug_schedule, dose=payload.dose, route=payload.route,
			frequency=payload.frequency, quantity=payload.quantity, refills=payload.refills,
			prescriber_id=payload.prescriber_id, pharmacy_id=payload.pharmacy_id,
			transmission_method=payload.transmission_method, status="transmitted",
			confirmation_number=f"RX-{uuid7str()[:8].upper()}",
			created_by=payload.created_by,
		)
		self._prescriptions[(payload.tenant_id, rx.id)] = rx
		self._audit(payload.tenant_id, "prescription_transmitted", rx.id)
		_log_op("transmit_prescription", payload.tenant_id, rx.id)
		return rx

	async def e_prescription(
		self,
		appointment_id: str,
		medications: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Issue an electronic prescription from a teleconsultation."""
		assert appointment_id, "appointment_id required"
		assert medications, "medications list required"
		tenant_id = self._tenant_id
		rx_id = uuid7str()
		rx_number = f"eRX-{datetime.utcnow().strftime('%Y%m%d')}-{rx_id[:6].upper()}"
		controlled_substances = [
			m for m in medications
			if m.get("schedule") in ("II", "III", "IV", "V")
		]
		items = []
		for med in medications:
			items.append({
				"drug_name": med.get("drug_name", ""),
				"dose": med.get("dose", ""),
				"route": med.get("route", "oral"),
				"frequency": med.get("frequency", ""),
				"quantity": med.get("quantity", 0),
				"refills": med.get("refills", 0),
				"schedule": med.get("schedule", "OTC"),
				"dea_required": med.get("schedule") in ("II", "III", "IV", "V"),
			})
		record: dict[str, Any] = {
			"id": rx_id,
			"rx_number": rx_number,
			"tenant_id": tenant_id,
			"appointment_id": appointment_id,
			"medications": items,
			"controlled_substances_count": len(controlled_substances),
			"requires_dea_auth": len(controlled_substances) > 0,
			"prescribed_by": self._actor_id,
			"prescribed_at": datetime.utcnow().isoformat(),
			"valid_until": (datetime.utcnow() + timedelta(days=30)).isoformat(),
			"transmission_method": "electronic",
			"status": "transmitted",
		}
		self._audit(tenant_id, "e_prescription_issued", rx_id)
		_log_op("e_prescription", tenant_id, rx_id)
		return record

	async def list_prescriptions(self, tenant_id: str, patient_id: str | None = None) -> list[PrescriptionTransmitResponse]:
		results = [p for (tid, _), p in self._prescriptions.items() if tid == tenant_id]
		if patient_id:
			results = [p for p in results if p.patient_id == patient_id]
		return sorted(results, key=lambda p: p.transmitted_at, reverse=True)

	# ── billing ───────────────────────────────────────────────────────────────

	async def create_billing_record(self, payload: TeleBillingCreate) -> TeleBillingResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_billing_record",
			"billing_code_supported": payload.billing_code in SUPPORTED_BILLING_CODES,
		})
		bill = TeleBillingResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, consultation_id=payload.consultation_id,
			patient_id=payload.patient_id, provider_id=payload.provider_id,
			billing_code=payload.billing_code, place_of_service=payload.place_of_service,
			diagnosis_codes=payload.diagnosis_codes, units=payload.units,
			status="pending", created_by=payload.created_by,
		)
		self._billing[(payload.tenant_id, bill.id)] = bill
		self._audit(payload.tenant_id, "billing_record_created", bill.id)
		_log_op("create_billing_record", payload.tenant_id, bill.id)
		return bill

	async def teleconsult_billing(self, appointment_id: str) -> dict[str, Any]:
		"""Generate a billing record for a completed teleconsultation."""
		assert appointment_id, "appointment_id required"
		tenant_id = self._tenant_id
		bill_id = uuid7str()
		claim_number = f"CLM-{datetime.utcnow().strftime('%Y%m')}-{bill_id[:8].upper()}"
		record: dict[str, Any] = {
			"id": bill_id,
			"claim_number": claim_number,
			"tenant_id": tenant_id,
			"appointment_id": appointment_id,
			"billing_code": "99213",
			"place_of_service": "02",
			"service_date": datetime.utcnow().date().isoformat(),
			"units": 1,
			"unit_charge": 150.0,
			"total_charge": 150.0,
			"payer": "insurance",
			"status": "pending",
			"billed_by": self._actor_id,
			"billed_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "teleconsult_billed", bill_id)
		_log_op("teleconsult_billing", tenant_id, appointment_id)
		return record

	async def list_billing(self, tenant_id: str, patient_id: str | None = None) -> list[TeleBillingResponse]:
		results = [b for (tid, _), b in self._billing.items() if tid == tenant_id]
		if patient_id:
			results = [b for b in results if b.patient_id == patient_id]
		return sorted(results, key=lambda b: b.created_at, reverse=True)

	# ── provider availability ─────────────────────────────────────────────────

	async def provider_availability(
		self,
		specialty: str,
		date: datetime,
	) -> dict[str, Any]:
		"""Return available time slots for providers of a given specialty."""
		assert specialty, "specialty required"
		tenant_id = self._tenant_id
		date_str = date.date().isoformat()
		standard_slots = [
			f"{date_str}T{h:02d}:00:00" for h in range(8, 17)
		]
		booked_slots: set[str] = set()
		for schedule in self._provider_schedules.values():
			for s in schedule:
				if s.get("time", "").startswith(date_str) and s.get("status") == "booked":
					booked_slots.add(s["time"][:16])
		available = [
			{"time": slot, "duration_minutes": 30, "available": slot[:16] not in booked_slots}
			for slot in standard_slots
		]
		available_count = sum(1 for s in available if s["available"])
		_log_op("provider_availability", tenant_id, specialty)
		return {
			"tenant_id": tenant_id,
			"specialty": specialty,
			"date": date_str,
			"total_slots": len(available),
			"available_slots": available_count,
			"booked_slots": len(booked_slots),
			"slots": available,
		}

	# ── analytics ─────────────────────────────────────────────────────────────

	async def teleconsult_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate teleconsultation metrics for a period."""
		assert period, "period required"
		tenant_id = self._tenant_id
		consults = [c for (tid, _), c in self._consultations.items() if tid == tenant_id]
		sessions = [s for (tid, _), s in self._sessions.items() if tid == tenant_id]
		monitoring = [m for (tid, _), m in self._monitoring.items() if tid == tenant_id]
		prescriptions = [p for (tid, _), p in self._prescriptions.items() if tid == tenant_id]
		billing = [b for (tid, _), b in self._billing.items() if tid == tenant_id]
		vitals = self._vital_readings
		alerts = self._telemonitoring_alerts
		completed_sessions = [s for s in sessions if s.status == "completed"]
		avg_duration = (
			sum(s.duration_seconds or 0 for s in completed_sessions) / len(completed_sessions) / 60
			if completed_sessions else 0.0
		)
		completion_rate = (
			len(completed_sessions) / len(sessions) * 100
			if sessions else 0.0
		)
		total_billed = sum(b.units * 150.0 for b in billing)
		_log_op("teleconsult_analytics", tenant_id, period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"consultations": {
				"total": len(consults),
				"scheduled": sum(1 for c in consults if c.status == "scheduled"),
				"completed": sum(1 for c in consults if c.status == "completed"),
				"cancelled": sum(1 for c in consults if c.status == "cancelled"),
				"completion_rate_pct": round(completion_rate, 1),
			},
			"sessions": {
				"total": len(sessions),
				"avg_duration_minutes": round(avg_duration, 1),
			},
			"remote_monitoring": {
				"enrolled_patients": len(monitoring),
				"active_enrollments": sum(1 for m in monitoring if m.status == "active"),
				"vital_readings": len(vitals),
				"alerts_triggered": len(alerts),
				"critical_alerts": sum(1 for a in alerts if a.get("severity") == "critical"),
			},
			"prescriptions": {"total": len(prescriptions)},
			"billing": {
				"total_claims": len(billing),
				"total_billed_amount": round(total_billed, 2),
			},
		}

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		consults = [c for (tid, _), c in self._consultations.items() if tid == tenant_id]
		sessions = [s for (tid, _), s in self._sessions.items() if tid == tenant_id]
		monitoring = [m for (tid, _), m in self._monitoring.items() if tid == tenant_id]
		prescriptions = [p for (tid, _), p in self._prescriptions.items() if tid == tenant_id]
		return {
			"tenant_id": tenant_id,
			"consultations": {"total": len(consults), "scheduled": sum(1 for c in consults if c.status == "scheduled"), "completed": sum(1 for c in consults if c.status == "completed")},
			"sessions": {"total": len(sessions), "active": sum(1 for s in sessions if s.status == "in_progress")},
			"monitoring": {"total": len(monitoring), "active": sum(1 for m in monitoring if m.status == "active")},
			"prescriptions": {"total": len(prescriptions)},
			"billing": {"total": len(self._billing)},
			"vital_readings": len(self._vital_readings),
			"active_alerts": sum(1 for a in self._telemonitoring_alerts if a.get("status") == "active"),
		}

	# ── patient consent management ────────────────────────────────────────────

	async def record_patient_consent(
		self,
		patient_id: str,
		consent_type: str,
		granted: bool,
		recorded_by: str,
	) -> dict[str, Any]:
		"""Record telehealth patient consent (video, data sharing, e-prescribing)."""
		assert patient_id, "patient_id required"
		assert consent_type in ("video_consultation", "data_sharing", "e_prescription", "remote_monitoring"), f"unsupported: {consent_type}"
		tenant_id = self._tenant_id
		consent_id = uuid7str()
		record: dict[str, Any] = {
			"id": consent_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"consent_type": consent_type,
			"granted": granted,
			"recorded_by": recorded_by,
			"recorded_at": datetime.utcnow().isoformat(),
			"valid_until": (datetime.utcnow() + timedelta(days=365)).isoformat(),
			"status": "active" if granted else "withdrawn",
		}
		self._audit(tenant_id, "patient_consent_recorded", consent_id)
		_log_op("record_patient_consent", tenant_id, consent_id)
		return record

	async def consent_check(self, patient_id: str, consent_type: str) -> dict[str, Any]:
		"""Verify a patient has active consent for a specific telehealth service."""
		assert patient_id, "patient_id required"
		tenant_id = self._tenant_id
		# In production would query consent store; stub returns advisory
		_log_op("consent_check", tenant_id, patient_id)
		return {
			"patient_id": patient_id,
			"consent_type": consent_type,
			"tenant_id": tenant_id,
			"check_result": "consent_required",
			"note": "Consent must be verified before proceeding",
			"checked_at": datetime.utcnow().isoformat(),
		}

	# ── clinical notes ────────────────────────────────────────────────────────

	async def create_clinical_note(
		self,
		consultation_id: str,
		note_type: str,
		content: str,
		authored_by: str,
	) -> dict[str, Any]:
		"""Create a structured clinical note linked to a teleconsultation."""
		assert consultation_id, "consultation_id required"
		assert note_type in ("soap", "progress", "discharge", "referral"), f"unsupported: {note_type}"
		assert content, "content required"
		tenant_id = self._tenant_id
		note_id = uuid7str()
		record: dict[str, Any] = {
			"id": note_id,
			"tenant_id": tenant_id,
			"consultation_id": consultation_id,
			"note_type": note_type,
			"content": content,
			"authored_by": authored_by,
			"authored_at": datetime.utcnow().isoformat(),
			"status": "final",
		}
		self._audit(tenant_id, "clinical_note_created", note_id)
		_log_op("create_clinical_note", tenant_id, note_id)
		return record

	async def get_consultation_notes(self, tenant_id: str, consultation_id: str) -> list[dict[str, Any]]:
		"""Return all clinical notes for a teleconsultation (stub — integrate EMR in production)."""
		_log_op("get_consultation_notes", tenant_id, consultation_id)
		return []

	# ── referral management ───────────────────────────────────────────────────

	async def create_referral(
		self,
		consultation_id: str,
		referring_provider_id: str,
		to_specialty: str,
		urgency: str,
		reason: str,
	) -> dict[str, Any]:
		"""Create a specialist referral from a teleconsultation."""
		assert consultation_id, "consultation_id required"
		assert to_specialty, "to_specialty required"
		assert urgency in ("routine", "urgent", "emergency"), f"invalid urgency: {urgency}"
		tenant_id = self._tenant_id
		referral_id = uuid7str()
		due_days = {"routine": 14, "urgent": 3, "emergency": 1}[urgency]
		record: dict[str, Any] = {
			"id": referral_id,
			"tenant_id": tenant_id,
			"consultation_id": consultation_id,
			"referring_provider_id": referring_provider_id,
			"to_specialty": to_specialty,
			"urgency": urgency,
			"reason": reason,
			"due_by": (datetime.utcnow() + timedelta(days=due_days)).isoformat(),
			"created_at": datetime.utcnow().isoformat(),
			"status": "pending",
		}
		self._audit(tenant_id, "referral_created", referral_id)
		_log_op("create_referral", tenant_id, referral_id)
		return record

	# ── follow-up scheduling ──────────────────────────────────────────────────

	async def schedule_follow_up(
		self,
		patient_id: str,
		provider_id: str,
		days_from_now: int,
		reason: str,
	) -> dict[str, Any]:
		"""Schedule a follow-up teleconsultation after an initial visit."""
		assert patient_id, "patient_id required"
		assert days_from_now >= 1, "days_from_now must be >= 1"
		tenant_id = self._tenant_id
		follow_up_id = uuid7str()
		follow_up_date = (datetime.utcnow() + timedelta(days=days_from_now)).isoformat()
		record: dict[str, Any] = {
			"id": follow_up_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"provider_id": provider_id,
			"follow_up_date": follow_up_date,
			"reason": reason,
			"days_from_now": days_from_now,
			"created_at": datetime.utcnow().isoformat(),
			"status": "scheduled",
		}
		self._audit(tenant_id, "follow_up_scheduled", follow_up_id)
		_log_op("schedule_follow_up", tenant_id, follow_up_id)
		return record

	# ── vital trends ──────────────────────────────────────────────────────────

	async def vital_trend_analysis(
		self,
		patient_id: str,
		vital_type: str,
		days: int = 30,
	) -> dict[str, Any]:
		"""Compute trend statistics for a vital type over the last N days."""
		assert patient_id, "patient_id required"
		assert vital_type, "vital_type required"
		tenant_id = self._tenant_id
		readings = [
			r for r in self._vital_readings
			if r["tenant_id"] == tenant_id and r["patient_id"] == patient_id and r["vital_type"] == vital_type
		]
		values = [r["value"] for r in readings]
		if not values:
			return {
				"patient_id": patient_id, "vital_type": vital_type,
				"tenant_id": tenant_id, "reading_count": 0,
				"trend": "no_data", "generated_at": datetime.utcnow().isoformat(),
			}
		avg = sum(values) / len(values)
		mn = min(values)
		mx = max(values)
		recent_avg = sum(values[-7:]) / min(len(values), 7)
		trend = "stable"
		if len(values) >= 2:
			if recent_avg > avg * 1.05:
				trend = "increasing"
			elif recent_avg < avg * 0.95:
				trend = "decreasing"
		alert_count = sum(1 for r in readings if r.get("alert_triggered"))
		_log_op("vital_trend_analysis", tenant_id, patient_id)
		return {
			"patient_id": patient_id,
			"vital_type": vital_type,
			"tenant_id": tenant_id,
			"reading_count": len(values),
			"mean": round(avg, 2),
			"min": round(mn, 2),
			"max": round(mx, 2),
			"recent_7_avg": round(recent_avg, 2),
			"trend": trend,
			"alert_count": alert_count,
			"days": days,
			"generated_at": datetime.utcnow().isoformat(),
		}

	# ── quality & compliance ──────────────────────────────────────────────────

	async def telehealth_quality_metrics(self, period: str) -> dict[str, Any]:
		"""Compute telehealth-specific quality metrics for a period."""
		assert period, "period required"
		tenant_id = self._tenant_id
		consults = [c for (tid, _), c in self._consultations.items() if tid == tenant_id]
		sessions = [s for (tid, _), s in self._sessions.items() if tid == tenant_id]
		completed = [c for c in consults if c.status == "completed"]
		consented = sum(1 for c in consults if c.patient_consent_obtained)
		consent_rate = round(consented / max(len(consults), 1) * 100, 1)
		e911_rate = round(sum(1 for c in consults if c.e911_disclosure_provided) / max(len(consults), 1) * 100, 1)
		_log_op("telehealth_quality_metrics", tenant_id, period)
		return {
			"period": period,
			"tenant_id": tenant_id,
			"total_consultations": len(consults),
			"completed_consultations": len(completed),
			"consent_rate_pct": consent_rate,
			"e911_disclosure_rate_pct": e911_rate,
			"active_monitoring_patients": sum(1 for (tid, _), m in self._monitoring.items() if tid == tenant_id and m.status == "active"),
			"vital_alerts_triggered": sum(1 for a in self._telemonitoring_alerts if a.get("tenant_id") == tenant_id),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def export_consultation_data(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export consultation records metadata."""
		consults = await self.list_consultations(tenant_id)
		export_id = uuid7str()
		_log_op("export_consultation_data", tenant_id, export_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"format": format,
			"record_count": len(consults),
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
			"status": "ready",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "TelemedicineService",
			"status": "healthy",
			"consultations": len(self._consultations),
			"sessions": len(self._sessions),
			"monitoring_enrollments": len(self._monitoring),
			"vital_readings": len(self._vital_readings),
			"prescriptions": len(self._prescriptions),
			"billing_records": len(self._billing),
			"audit_events": len(self._audit_events),
			"checked_at": datetime.utcnow().isoformat(),
		}

	async def update_consultation_status(self, tenant_id: str, consult_id: str, status: str) -> ConsultationResponse | None:
		"""Update a consultation status directly."""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_consultation",
			"consultation_status": status,
		})
		consult = self._consultations.get((tenant_id, consult_id))
		if consult is None:
			return None
		updated = consult.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._consultations[(tenant_id, consult_id)] = updated
		self._audit(tenant_id, "consultation_status_updated", consult_id)
		return updated

	async def active_monitoring_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Return summary of all active remote monitoring enrollments."""
		enrollments = [m for (tid, _), m in self._monitoring.items() if tid == tenant_id and m.status == "active"]
		device_types: dict[str, int] = {}
		for m in enrollments:
			device_types[m.device_type] = device_types.get(m.device_type, 0) + 1
		return {
			"tenant_id": tenant_id,
			"active_enrollments": len(enrollments),
			"unique_patients": len({m.patient_id for m in enrollments}),
			"by_device_type": device_types,
			"vital_readings_total": len(self._vital_readings),
			"active_alerts": sum(1 for a in self._telemonitoring_alerts if a.get("status") == "active"),
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			logger.warning("tel.rule_denied rule=%s", result["rule"])
			raise PolicyViolationError(result["reason"])

	async def teleconsult_kpi_summary(
		self,
		tenant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a concise teleconsultation KPI card for dashboard consumption."""
		consultations = [c for (tid, _), c in self._consultations.items() if tid == tenant_id]
		sessions = [s for (tid, _), s in self._sessions.items() if tid == tenant_id]
		monitoring = [m for (tid, _), m in self._monitoring.items() if tid == tenant_id]
		prescriptions = [p for (tid, _), p in self._prescriptions.items() if tid == tenant_id]
		completed = sum(1 for c in consultations if c.status == "completed")
		completion_rate = round(completed / max(len(consultations), 1) * 100, 1)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"total_consultations": len(consultations),
			"completed_consultations": completed,
			"completion_rate_pct": completion_rate,
			"active_sessions": sum(1 for s in sessions if s.status == "active"),
			"active_monitoring_enrollments": sum(1 for m in monitoring if m.status == "active"),
			"prescriptions_issued": len(prescriptions),
			"vital_readings": len(self._vital_readings),
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def ai_diagnosis_assist(
		self,
		tenant_id: str,
		patient_id: str,
		symptoms: list[str],
		vitals: dict[str, Any] | None = None,
		requested_by: str = "clinician",
	) -> dict[str, Any]:
		"""Provide AI-assisted differential diagnosis suggestions based on symptoms and vitals.

		Returns ranked differential diagnoses with confidence scores.
		In production this delegates to an Ollama-served clinical LLM.
		This in-memory implementation returns deterministic scored suggestions.
		"""
		assert symptoms, "at least one symptom required"
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "ai_diagnosis_assist",
		})
		# Deterministic scoring: each symptom contributes to known differential patterns
		_symptom_map: dict[str, list[tuple[str, float]]] = {
			"fever": [("influenza", 0.72), ("malaria", 0.68), ("typhoid", 0.55)],
			"chest_pain": [("angina", 0.75), ("myocardial_infarction", 0.60), ("pericarditis", 0.45)],
			"cough": [("upper_respiratory_infection", 0.80), ("pneumonia", 0.55), ("tuberculosis", 0.30)],
			"headache": [("migraine", 0.70), ("hypertension", 0.50), ("meningitis", 0.25)],
		}
		scores: dict[str, float] = {}
		for symptom in symptoms:
			for dx, conf in _symptom_map.get(symptom.lower().replace(" ", "_"), []):
				scores[dx] = max(scores.get(dx, 0.0), conf)
		differentials = sorted(
			[{"diagnosis": dx, "confidence": round(conf, 2)} for dx, conf in scores.items()],
			key=lambda x: x["confidence"],
			reverse=True,
		)[:5]
		assist_id = uuid7str()
		self._audit(tenant_id, "ai_diagnosis_assist_requested", assist_id)
		return {
			"assist_id": assist_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"symptoms": symptoms,
			"vitals": vitals or {},
			"differentials": differentials,
			"disclaimer": "AI-assisted suggestion only — clinical judgement required",
			"requested_by": requested_by,
			"generated_at": datetime.utcnow().isoformat(),
		}

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({"tenant_id": tenant_id, "event": event, "entity_id": entity_id, "timestamp": datetime.utcnow().isoformat()})
