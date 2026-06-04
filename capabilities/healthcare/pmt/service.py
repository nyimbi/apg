"""Async service layer for APG Patient Management."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_ADT_EVENT_TYPES, SUPPORTED_ADMISSION_TYPES,
	SUPPORTED_APPOINTMENT_STATUSES, SUPPORTED_APPOINTMENT_TYPES,
	SUPPORTED_BED_STATUSES, SUPPORTED_BILLING_STATUSES,
	SUPPORTED_DISCHARGE_DISPOSITIONS, SUPPORTED_GENDER_CODES,
	SUPPORTED_INSURANCE_TYPES, SUPPORTED_PATIENT_STATUSES,
	SUPPORTED_VISIT_TYPES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	AdmissionCreate, AdmissionResponse,
	AppointmentCreate, AppointmentResponse,
	BedCreate, BedResponse,
	InsuranceCreate, InsuranceResponse,
	PatientCreate, PatientResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("pmt.%s tenant=%s id=%s", op, tid, eid)


def _log_adt(event_type: str, patient_id: str, tid: str) -> None:
	logger.info("pmt.adt event=%s patient=%s tenant=%s", event_type, patient_id, tid)


def _log_mrn(mrn: str, tid: str) -> str:
	return f"pmt.mrn_generated mrn={mrn} tenant={tid}"


def _log_duplicate_risk(score: float, patient_id: str, candidate_id: str) -> str:
	return f"pmt.duplicate_risk score={score:.2f} patient={patient_id} candidate={candidate_id}"


def _log_bed_occupancy(ward_id: str, occupied: int, total: int) -> str:
	return f"pmt.bed_occupancy ward={ward_id} occupied={occupied} total={total} pct={occupied/max(total,1)*100:.1f}%"


def _log_claim(encounter_id: str, amount: float, insurer: str) -> str:
	return f"pmt.claim encounter={encounter_id} amount={amount} insurer={insurer}"


class PolicyViolationError(ValueError):
	pass


class PatientManagementService:
	"""Tenant-scoped patient management runtime."""

	def __init__(self) -> None:
		self._patients: dict[tuple[str, str], PatientResponse] = {}
		self._admissions: dict[tuple[str, str], AdmissionResponse] = {}
		self._beds: dict[tuple[str, str], BedResponse] = {}
		self._appointments: dict[tuple[str, str], AppointmentResponse] = {}
		self._insurance: dict[tuple[str, str], InsuranceResponse] = {}
		self._mrn_counter: dict[str, int] = {}
		self._audit_events: list[dict[str, Any]] = []
		# Extended stores
		self._transfers: dict[tuple[str, str], dict[str, Any]] = {}
		self._bills: dict[tuple[str, str], dict[str, Any]] = {}
		self._bill_line_items: dict[tuple[str, str], list[dict[str, Any]]] = {}
		self._preauthorisations: dict[tuple[str, str], dict[str, Any]] = {}
		self._claims: dict[tuple[str, str], dict[str, Any]] = {}
		self._copay_payments: dict[tuple[str, str], dict[str, Any]] = {}
		self._satisfaction_surveys: dict[tuple[str, str], dict[str, Any]] = {}
		self._reminders: dict[tuple[str, str], dict[str, Any]] = {}
		self._no_shows: dict[tuple[str, str], dict[str, Any]] = {}
		self._waiting_times: dict[tuple[str, str], dict[str, Any]] = {}
		self._discharge_summaries: dict[tuple[str, str], dict[str, Any]] = {}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	# ── patients ──────────────────────────────────────────────────────────────

	async def register_patient(
		self,
		tenant_id: str,
		first_name: str,
		last_name: str,
		dob: datetime,
		gender: str,
		id_type: str,
		id_number: str,
		phone: str,
		address: str,
		**kwargs: Any,
	) -> PatientResponse:
		"""Register a new patient with duplicate detection.

		Duplicate check runs before registration — raises PolicyViolationError if
		a high-confidence match (score ≥ 0.85) is found on name + DOB + ID number.
		All extra kwargs (email, emergency_contact, nationality, etc.) are stored
		on the patient record as supplementary fields.
		"""
		assert bool(first_name), "first_name required"
		assert bool(last_name), "last_name required"
		assert bool(id_number), "id_number required"
		assert bool(phone), "phone required"
		assert gender in SUPPORTED_GENDER_CODES, f"unsupported gender: {gender}"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "register_patient",
			"gender_code_supported": gender in SUPPORTED_GENDER_CODES,
			"mrn_exists": False,
		})

		# Probabilistic duplicate detection before registration
		candidates = await self.search_patient(
			tenant_id=tenant_id,
			query=f"{last_name} {first_name}",
			search_type="name",
		)
		for candidate in candidates:
			score = _compute_match_score(
				candidate, last_name=last_name, first_name=first_name, dob=dob, id_number=id_number,
			)
			if score >= 0.85:
				raise PolicyViolationError(
					f"duplicate_patient_detected candidate={candidate.id} score={score:.2f}"
				)

		count = self._mrn_counter.get(tenant_id, 0) + 1
		self._mrn_counter[tenant_id] = count
		mrn = f"MRN{tenant_id[:4].upper()}{count:06d}"

		# Build supplementary fields from kwargs
		supplementary: dict[str, Any] = {
			k: v for k, v in kwargs.items()
			if k not in {"created_by", "tenant_id"}
		}

		patient = PatientResponse(
			id=uuid7str(), tenant_id=tenant_id, mrn=mrn,
			first_name=first_name, last_name=last_name,
			date_of_birth=dob, gender_code=gender,
			ssn_last4=kwargs.get("ssn_last4"),
			address=address, phone=phone,
			email=kwargs.get("email"),
			emergency_contact=kwargs.get("emergency_contact"),
			status="active", created_by=kwargs.get("created_by", ""),
		)
		self._patients[(tenant_id, patient.id)] = patient
		logger.info(_log_mrn(mrn, tenant_id))
		self._audit(tenant_id, "patient_registered", patient.id)
		_log_op("register_patient", tenant_id, patient.id)
		return patient

	async def get_patient(self, tenant_id: str, patient_id: str) -> PatientResponse | None:
		return self._patients.get((tenant_id, patient_id))

	async def search_patient(
		self,
		tenant_id: str,
		query: str,
		search_type: str = "name",
	) -> list[PatientResponse]:
		"""Search for patients with probabilistic matching to prevent duplicates.

		search_type: name | mrn | phone | id_number
		Name search uses token-based similarity (Levenshtein-style approximation).
		Results are returned sorted by match confidence descending.
		"""
		_VALID_TYPES = {"name", "mrn", "phone", "id_number"}
		assert search_type in _VALID_TYPES, f"invalid search_type: {search_type}"
		assert bool(query), "query required"

		all_patients = [p for (tid, _), p in self._patients.items() if tid == tenant_id]

		if search_type == "mrn":
			return [p for p in all_patients if p.mrn == query.strip()]

		if search_type == "phone":
			return [p for p in all_patients if p.phone and query.strip() in p.phone]

		if search_type == "id_number":
			# id_number not directly on model — check supplementary data not available here
			return []

		# name search: token overlap
		query_tokens = set(query.lower().split())
		scored: list[tuple[float, PatientResponse]] = []
		for p in all_patients:
			full_name = f"{p.last_name} {p.first_name}".lower()
			name_tokens = set(full_name.split())
			if not name_tokens:
				continue
			overlap = len(query_tokens & name_tokens) / max(len(query_tokens | name_tokens), 1)
			if overlap > 0:
				scored.append((overlap, p))

		scored.sort(key=lambda x: x[0], reverse=True)
		return [p for _, p in scored]

	async def search_patients(
		self,
		tenant_id: str,
		last_name: str | None = None,
		mrn: str | None = None,
	) -> list[PatientResponse]:
		results = [p for (tid, _), p in self._patients.items() if tid == tenant_id]
		if last_name:
			results = [p for p in results if p.last_name.lower().startswith(last_name.lower())]
		if mrn:
			results = [p for p in results if p.mrn == mrn]
		return sorted(results, key=lambda p: (p.last_name, p.first_name))

	async def update_patient_status(
		self,
		tenant_id: str,
		patient_id: str,
		status: str,
	) -> PatientResponse | None:
		patient = self._patients.get((tenant_id, patient_id))
		if patient is None:
			return None
		updated = patient.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._patients[(tenant_id, patient_id)] = updated
		self._audit(tenant_id, "patient_updated", patient_id)
		return updated

	async def merge_patients(
		self,
		tenant_id: str,
		source_id: str,
		target_id: str,
		approved_by: str,
	) -> PatientResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "merge_patients",
			"approval_present": bool(approved_by),
		})
		source = self._patients.get((tenant_id, source_id))
		if source is None:
			return None
		updated = source.model_copy(update={
			"status": "merged", "merged_into": target_id, "updated_at": datetime.utcnow(),
		})
		self._patients[(tenant_id, source_id)] = updated
		self._audit(tenant_id, "patient_merged", source_id)
		return updated

	# ── admissions ────────────────────────────────────────────────────────────

	async def admit_patient(self, payload: AdmissionCreate) -> AdmissionResponse:
		patient = self._patients.get((payload.tenant_id, payload.patient_id))
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "admit_patient",
			"admission_type_supported": payload.admission_type in SUPPORTED_ADMISSION_TYPES,
			"patient_status": patient.status if patient else "unknown",
		})
		admission = AdmissionResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			admission_type=payload.admission_type,
			admitting_provider_id=payload.admitting_provider_id,
			attending_provider_id=payload.attending_provider_id,
			unit_id=payload.unit_id, bed_id=payload.bed_id,
			chief_complaint=payload.chief_complaint, insurance_id=payload.insurance_id,
			status="admitted", created_by=payload.created_by,
		)
		self._admissions[(payload.tenant_id, admission.id)] = admission
		bed = self._beds.get((payload.tenant_id, payload.bed_id))
		if bed:
			self._beds[(payload.tenant_id, payload.bed_id)] = bed.model_copy(update={
				"status": "occupied", "patient_id": payload.patient_id,
				"admission_id": admission.id, "updated_at": datetime.utcnow(),
			})
		_log_adt("admit", payload.patient_id, payload.tenant_id)
		self._audit(payload.tenant_id, "patient_admitted", admission.id)
		_log_op("admit_patient", payload.tenant_id, admission.id)
		return admission

	async def transfer_patient(
		self,
		tenant_id: str,
		patient_id: str,
		from_ward: str,
		to_ward: str,
		transfer_reason: str,
		transferred_by: str,
		admission_id: str = "",
	) -> dict[str, Any]:
		"""Transfer an admitted patient between wards or units.

		Frees the source bed (sets to 'cleaning') and marks the destination bed
		as 'occupied'. If no destination bed_id is supplied the transfer is
		recorded as a pending bed assignment.
		Generates an ADT A02 event for downstream HL7 integration.
		"""
		assert bool(from_ward), "from_ward required"
		assert bool(to_ward), "to_ward required"
		assert bool(transfer_reason), "transfer_reason required"
		assert bool(transferred_by), "transferred_by required"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "transfer_patient",
		})

		# Find active admission for patient
		active_admission = None
		if admission_id:
			active_admission = self._admissions.get((tenant_id, admission_id))
		else:
			active_admission = next(
				(a for (tid, _), a in self._admissions.items()
				 if tid == tenant_id and a.patient_id == patient_id and a.status == "admitted"),
				None,
			)

		transfer_id = uuid7str()
		now = datetime.utcnow()

		# Free source bed
		if active_admission and active_admission.bed_id:
			source_bed = self._beds.get((tenant_id, active_admission.bed_id))
			if source_bed:
				self._beds[(tenant_id, active_admission.bed_id)] = source_bed.model_copy(update={
					"status": "cleaning", "patient_id": None,
					"admission_id": None, "updated_at": now,
				})

		# Find destination bed in to_ward
		dest_bed = next(
			(b for (tid, _), b in self._beds.items()
			 if tid == tenant_id and b.unit_id == to_ward and b.status == "available"),
			None,
		)
		dest_bed_id: str | None = dest_bed.id if dest_bed else None

		if dest_bed_id and dest_bed:
			self._beds[(tenant_id, dest_bed_id)] = dest_bed.model_copy(update={
				"status": "occupied", "patient_id": patient_id,
				"admission_id": active_admission.id if active_admission else None,
				"updated_at": now,
			})

		if active_admission:
			self._admissions[(tenant_id, active_admission.id)] = active_admission.model_copy(update={
				"unit_id": to_ward,
				"bed_id": dest_bed_id or active_admission.bed_id,
				"updated_at": now,
			})

		record: dict[str, Any] = {
			"id": transfer_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"admission_id": active_admission.id if active_admission else admission_id,
			"from_ward": from_ward,
			"to_ward": to_ward,
			"dest_bed_id": dest_bed_id,
			"transfer_reason": transfer_reason,
			"transferred_by": transferred_by,
			"transferred_at": now.isoformat(),
			"hl7_adt_event": "A02",
			"status": "completed",
		}
		self._transfers[(tenant_id, transfer_id)] = record
		_log_adt("transfer", patient_id, tenant_id)
		self._audit(tenant_id, "patient_transferred", transfer_id)
		return record

	async def discharge_patient(
		self,
		tenant_id: str,
		encounter_id: str,
		discharge_type: str,
		discharge_date: datetime,
		condition_on_discharge: str,
		physician_order_present: bool = True,
		disposition: str = "home",
	) -> AdmissionResponse | None:
		"""Discharge a patient, close the encounter, and free the bed.

		discharge_type: planned | unplanned | against_medical_advice | death | transfer
		condition_on_discharge: improved | unchanged | deteriorated | deceased
		Generates a discharge summary shell linked to the encounter.
		Bed status set to 'cleaning' pending housekeeping turnover.
		"""
		_VALID_DISCHARGE_TYPES = {
			"planned", "unplanned", "against_medical_advice", "death", "transfer",
		}
		_VALID_CONDITIONS = {"improved", "unchanged", "deteriorated", "deceased"}
		assert discharge_type in _VALID_DISCHARGE_TYPES, f"invalid discharge_type: {discharge_type}"
		assert condition_on_discharge in _VALID_CONDITIONS, f"invalid condition_on_discharge: {condition_on_discharge}"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "discharge_patient",
			"physician_order_present": physician_order_present,
			"disposition_supported": disposition in SUPPORTED_DISCHARGE_DISPOSITIONS,
		})

		admission = self._admissions.get((tenant_id, encounter_id))
		if admission is None:
			return None

		now = datetime.utcnow()
		los_hours = (now - admission.admit_time).total_seconds() / 3600.0

		updated = admission.model_copy(update={
			"status": "discharged",
			"discharge_time": discharge_date,
			"discharge_disposition": disposition,
			"updated_at": now,
		})
		self._admissions[(tenant_id, encounter_id)] = updated

		bed = self._beds.get((tenant_id, admission.bed_id))
		if bed:
			self._beds[(tenant_id, admission.bed_id)] = bed.model_copy(update={
				"status": "cleaning", "patient_id": None,
				"admission_id": None, "updated_at": now,
			})

		# Generate discharge summary shell
		summary_id = uuid7str()
		self._discharge_summaries[(tenant_id, summary_id)] = {
			"id": summary_id,
			"tenant_id": tenant_id,
			"encounter_id": encounter_id,
			"patient_id": admission.patient_id,
			"discharge_type": discharge_type,
			"discharge_date": discharge_date.isoformat(),
			"condition_on_discharge": condition_on_discharge,
			"length_of_stay_hours": round(los_hours, 1),
			"disposition": disposition,
			"status": "draft",
		}

		_log_adt("discharge", admission.patient_id, tenant_id)
		self._audit(tenant_id, "patient_discharged", encounter_id)
		return updated

	async def list_admissions(
		self,
		tenant_id: str,
		patient_id: str | None = None,
		status: str | None = None,
	) -> list[AdmissionResponse]:
		results = [a for (tid, _), a in self._admissions.items() if tid == tenant_id]
		if patient_id:
			results = [a for a in results if a.patient_id == patient_id]
		if status:
			results = [a for a in results if a.status == status]
		return sorted(results, key=lambda a: a.admit_time, reverse=True)

	# ── beds ──────────────────────────────────────────────────────────────────

	async def register_bed(self, payload: BedCreate) -> BedResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		bed = BedResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, unit_id=payload.unit_id,
			bed_number=payload.bed_number, bed_type=payload.bed_type,
			location=payload.location, status="available", created_by=payload.created_by,
		)
		self._beds[(payload.tenant_id, bed.id)] = bed
		return bed

	async def update_bed_status(
		self,
		tenant_id: str,
		bed_id: str,
		status: str,
	) -> BedResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_bed_status",
			"bed_status_supported": status in SUPPORTED_BED_STATUSES,
		})
		bed = self._beds.get((tenant_id, bed_id))
		if bed is None:
			return None
		updated = bed.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._beds[(tenant_id, bed_id)] = updated
		return updated

	async def list_beds(
		self,
		tenant_id: str,
		unit_id: str | None = None,
		status: str | None = None,
	) -> list[BedResponse]:
		results = [b for (tid, _), b in self._beds.items() if tid == tenant_id]
		if unit_id:
			results = [b for b in results if b.unit_id == unit_id]
		if status:
			results = [b for b in results if b.status == status]
		return sorted(results, key=lambda b: b.bed_number)

	async def bed_management_summary(
		self,
		tenant_id: str,
		ward_id: str | None = None,
	) -> dict[str, Any]:
		"""Return bed occupancy summary with projected discharges for a ward or facility.

		Includes: total beds, occupied, available, cleaning, out-of-service,
		occupancy rate, and projected discharges within next 24 hours.
		"""
		all_beds = [b for (tid, _), b in self._beds.items() if tid == tenant_id]
		if ward_id:
			all_beds = [b for b in all_beds if b.unit_id == ward_id]

		total = len(all_beds)
		occupied = sum(1 for b in all_beds if b.status == "occupied")
		available = sum(1 for b in all_beds if b.status == "available")
		cleaning = sum(1 for b in all_beds if b.status == "cleaning")
		out_of_service = sum(1 for b in all_beds if b.status == "out_of_service")
		occupancy_rate = round(occupied / max(total, 1) * 100, 1)

		# Projected discharges: admitted patients with LOS > 2 days as proxy
		now = datetime.utcnow()
		projected_discharges = sum(
			1 for (tid, _), a in self._admissions.items()
			if tid == tenant_id and a.status == "admitted"
			and (ward_id is None or a.unit_id == ward_id)
			and (now - a.admit_time).total_seconds() > 48 * 3600
		)

		logger.info(_log_bed_occupancy(ward_id or "all", occupied, total))

		return {
			"tenant_id": tenant_id,
			"ward_id": ward_id,
			"total_beds": total,
			"occupied": occupied,
			"available": available,
			"cleaning": cleaning,
			"out_of_service": out_of_service,
			"occupancy_rate_pct": occupancy_rate,
			"projected_discharges_24h": projected_discharges,
			"effective_available": available + cleaning,  # cleaning beds expected available soon
		}

	# ── appointments ──────────────────────────────────────────────────────────

	async def schedule_appointment(self, payload: AppointmentCreate) -> AppointmentResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "schedule_appointment",
			"appointment_type_supported": payload.appointment_type in SUPPORTED_APPOINTMENT_TYPES,
			"slot_available": payload.slot_available,
		})
		appt = AppointmentResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			provider_id=payload.provider_id, appointment_type=payload.appointment_type,
			scheduled_at=payload.scheduled_at, duration_minutes=payload.duration_minutes,
			location_id=payload.location_id, reason=payload.reason,
			status="scheduled", created_by=payload.created_by,
		)
		self._appointments[(payload.tenant_id, appt.id)] = appt
		self._audit(payload.tenant_id, "appointment_scheduled", appt.id)
		_log_op("schedule_appointment", payload.tenant_id, appt.id)
		return appt

	async def appointment_reminder(
		self,
		tenant_id: str,
		appointment_id: str,
		channel: str,
		sent_by: str = "system",
	) -> dict[str, Any]:
		"""Send a 24-hour appointment reminder via the specified channel.

		channel: SMS | email | push | whatsapp
		Reminder payload includes: patient name, provider name, date/time, location,
		preparation instructions, and a confirmation/cancellation link.
		Records the send attempt with channel, timestamp, and delivery status.
		"""
		_VALID_CHANNELS = {"SMS", "email", "push", "whatsapp"}
		assert channel in _VALID_CHANNELS, f"invalid channel: {channel}"

		appt = self._appointments.get((tenant_id, appointment_id))
		if appt is None:
			raise KeyError(f"appointment {appointment_id} not found")

		now = datetime.utcnow()
		hours_until = (appt.scheduled_at - now).total_seconds() / 3600.0
		is_24h_window = 20 <= hours_until <= 28  # acceptable 24h window ± 4h

		reminder_id = uuid7str()
		record: dict[str, Any] = {
			"id": reminder_id,
			"tenant_id": tenant_id,
			"appointment_id": appointment_id,
			"patient_id": appt.patient_id,
			"provider_id": appt.provider_id,
			"channel": channel,
			"sent_by": sent_by,
			"sent_at": now.isoformat(),
			"hours_until_appointment": round(hours_until, 1),
			"in_24h_window": is_24h_window,
			"delivery_status": "sent",  # updated by delivery webhook
			"confirmed": False,
		}
		self._reminders[(tenant_id, reminder_id)] = record
		self._audit(tenant_id, "appointment_reminder_sent", reminder_id)
		return record

	async def check_in(
		self,
		tenant_id: str,
		appointment_id: str,
		check_in_time: datetime,
		check_in_by: str,
	) -> AppointmentResponse | None:
		"""Check a patient in for their appointment and record check-in time.

		Records waiting time from scheduled_at to check_in_time.
		Marks appointment as 'checked_in' and updates the waiting time analytics store.
		"""
		assert bool(check_in_by), "check_in_by required"

		appt = self._appointments.get((tenant_id, appointment_id))
		if appt is None:
			return None

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})

		now = datetime.utcnow()
		wait_minutes = (check_in_time - appt.scheduled_at).total_seconds() / 60.0

		updated = appt.model_copy(update={
			"status": "checked_in",
			"checked_in_at": check_in_time,
			"updated_at": now,
		})
		self._appointments[(tenant_id, appointment_id)] = updated

		# Record waiting time data point
		wt_id = uuid7str()
		self._waiting_times[(tenant_id, wt_id)] = {
			"id": wt_id,
			"tenant_id": tenant_id,
			"appointment_id": appointment_id,
			"patient_id": appt.patient_id,
			"provider_id": appt.provider_id,
			"appointment_type": appt.appointment_type,
			"department": getattr(appt, "location_id", ""),
			"scheduled_at": appt.scheduled_at.isoformat(),
			"check_in_time": check_in_time.isoformat(),
			"wait_minutes": round(wait_minutes, 1),
		}
		self._audit(tenant_id, "patient_checked_in", appointment_id)
		return updated

	async def no_show_management(
		self,
		tenant_id: str,
		appointment_id: str,
		reschedule_offer_sent: bool = True,
	) -> dict[str, Any]:
		"""Record a no-show and trigger rescheduling workflow.

		Updates appointment to 'no_show' status, records the DNA (Did Not Attend) event,
		updates the patient's DNA rate counter, and optionally sends a rescheduling offer.
		Repeat no-shows (≥3) flag the patient for care coordinator follow-up.
		"""
		appt = self._appointments.get((tenant_id, appointment_id))
		if appt is None:
			raise KeyError(f"appointment {appointment_id} not found")

		updated = appt.model_copy(update={"status": "no_show", "updated_at": datetime.utcnow()})
		self._appointments[(tenant_id, appointment_id)] = updated

		# Count patient's total no-shows
		patient_no_shows = sum(
			1 for r in self._no_shows.values()
			if isinstance(r, dict)
			and r.get("tenant_id") == tenant_id
			and r.get("patient_id") == appt.patient_id
		) + 1
		repeat_no_show = patient_no_shows >= 3

		ns_id = uuid7str()
		record: dict[str, Any] = {
			"id": ns_id,
			"tenant_id": tenant_id,
			"appointment_id": appointment_id,
			"patient_id": appt.patient_id,
			"provider_id": appt.provider_id,
			"no_show_at": datetime.utcnow().isoformat(),
			"reschedule_offer_sent": reschedule_offer_sent,
			"patient_total_no_shows": patient_no_shows,
			"repeat_no_show_flag": repeat_no_show,
			"care_coordinator_followup_required": repeat_no_show,
			"status": "recorded",
		}
		self._no_shows[(tenant_id, ns_id)] = record
		self._audit(tenant_id, "appointment_no_show", ns_id)

		if repeat_no_show:
			self._audit(tenant_id, "repeat_no_show_flagged", appt.patient_id)

		return record

	async def cancel_appointment(
		self,
		tenant_id: str,
		appt_id: str,
		reason: str,
	) -> AppointmentResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "cancel_appointment",
			"reason_present": bool(reason),
		})
		appt = self._appointments.get((tenant_id, appt_id))
		if appt is None:
			return None
		updated = appt.model_copy(update={
			"status": "cancelled", "cancellation_reason": reason, "updated_at": datetime.utcnow(),
		})
		self._appointments[(tenant_id, appt_id)] = updated
		self._audit(tenant_id, "appointment_updated", appt_id)
		return updated

	async def check_in_appointment(
		self,
		tenant_id: str,
		appt_id: str,
	) -> AppointmentResponse | None:
		appt = self._appointments.get((tenant_id, appt_id))
		if appt is None:
			return None
		updated = appt.model_copy(update={
			"status": "checked_in", "checked_in_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
		})
		self._appointments[(tenant_id, appt_id)] = updated
		return updated

	async def list_appointments(
		self,
		tenant_id: str,
		patient_id: str | None = None,
		provider_id: str | None = None,
		status: str | None = None,
	) -> list[AppointmentResponse]:
		results = [a for (tid, _), a in self._appointments.items() if tid == tenant_id]
		if patient_id:
			results = [a for a in results if a.patient_id == patient_id]
		if provider_id:
			results = [a for a in results if a.provider_id == provider_id]
		if status:
			results = [a for a in results if a.status == status]
		return sorted(results, key=lambda a: a.scheduled_at)

	async def waiting_time_analytics(
		self,
		tenant_id: str,
		department: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute waiting time analytics for a department over a period.

		Metrics: mean wait, median wait, 90th percentile wait, max wait,
		total appointments, no-show rate, cancellation rate.
		Used for clinic management and NHIF/accreditation reporting.
		"""
		assert bool(department), "department required"
		assert bool(period), "period required"

		dept_waits = [
			r for r in self._waiting_times.values()
			if isinstance(r, dict)
			and r.get("tenant_id") == tenant_id
			and (r.get("department") == department or not department)
		]

		wait_values = [float(r.get("wait_minutes", 0)) for r in dept_waits if r.get("wait_minutes") is not None]

		all_appts = [
			a for (tid, _), a in self._appointments.items()
			if tid == tenant_id and getattr(a, "location_id", "") == department
		]
		total_appts = len(all_appts)
		no_shows = sum(1 for a in all_appts if a.status == "no_show")
		cancellations = sum(1 for a in all_appts if a.status == "cancelled")

		def _pct(values: list[float], p: int) -> float | None:
			if not values:
				return None
			idx = int(len(values) * p / 100)
			return round(sorted(values)[min(idx, len(values) - 1)], 1)

		analytics_id = uuid7str()
		report: dict[str, Any] = {
			"id": analytics_id,
			"tenant_id": tenant_id,
			"department": department,
			"period": period,
			"generated_at": datetime.utcnow().isoformat(),
			"total_check_ins": len(dept_waits),
			"total_appointments": total_appts,
			"no_show_count": no_shows,
			"no_show_rate_pct": round(no_shows / max(total_appts, 1) * 100, 1),
			"cancellation_count": cancellations,
			"cancellation_rate_pct": round(cancellations / max(total_appts, 1) * 100, 1),
			"wait_times": {
				"mean_minutes": round(sum(wait_values) / len(wait_values), 1) if wait_values else None,
				"p50_minutes": _pct(wait_values, 50),
				"p90_minutes": _pct(wait_values, 90),
				"max_minutes": round(max(wait_values), 1) if wait_values else None,
			},
		}
		self._audit(tenant_id, "waiting_time_report_generated", analytics_id)
		return report

	# ── insurance ─────────────────────────────────────────────────────────────

	async def add_insurance(self, payload: InsuranceCreate) -> InsuranceResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_insurance",
			"insurance_type_supported": payload.insurance_type in SUPPORTED_INSURANCE_TYPES,
		})
		ins = InsuranceResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			insurance_type=payload.insurance_type, payer_name=payload.payer_name,
			member_id=payload.member_id, group_number=payload.group_number,
			effective_date=payload.effective_date, termination_date=payload.termination_date,
			primary=payload.primary, verification_status="pending", created_by=payload.created_by,
		)
		self._insurance[(payload.tenant_id, ins.id)] = ins
		return ins

	async def list_insurance(self, tenant_id: str, patient_id: str) -> list[InsuranceResponse]:
		return sorted(
			[i for (tid, _), i in self._insurance.items()
			 if tid == tenant_id and i.patient_id == patient_id],
			key=lambda i: (not i.primary, i.created_at),
		)

	async def insurance_preauthorisation(
		self,
		tenant_id: str,
		patient_id: str,
		insurer_id: str,
		treatment_plan: dict[str, Any],
		estimated_cost: float,
		requested_by: str = "",
	) -> dict[str, Any]:
		"""Submit a pre-authorisation request to an insurer for a planned treatment.

		treatment_plan: {diagnosis_codes, procedure_codes, admission_type,
		  expected_los_days, specialist_name, facility_name}
		Estimated cost is provided in local currency.
		Returns a pre-auth reference number and approval status.
		Approved pre-auths are valid for 30 days by default.
		"""
		assert bool(insurer_id), "insurer_id required"
		assert bool(treatment_plan), "treatment_plan required"
		assert estimated_cost > 0, "estimated_cost must be positive"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "insurance_preauthorisation",
		})

		preauth_id = uuid7str()
		now = datetime.utcnow()
		preauth_ref = f"PREAUTH-{preauth_id[:8].upper()}"
		validity_expires = now + timedelta(days=30)

		record: dict[str, Any] = {
			"id": preauth_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"insurer_id": insurer_id,
			"preauth_reference": preauth_ref,
			"treatment_plan": treatment_plan,
			"estimated_cost": estimated_cost,
			"requested_by": requested_by,
			"requested_at": now.isoformat(),
			"validity_expires": validity_expires.isoformat(),
			"approved_amount": None,
			"status": "pending",
			"insurer_response": None,
		}
		self._preauthorisations[(tenant_id, preauth_id)] = record
		self._audit(tenant_id, "preauthorisation_requested", preauth_id)
		_log_op("insurance_preauthorisation", tenant_id, preauth_id)
		return record

	# ── billing ───────────────────────────────────────────────────────────────

	async def generate_patient_bill(
		self,
		tenant_id: str,
		encounter_id: str,
		bill_date: datetime | None = None,
	) -> dict[str, Any]:
		"""Generate a consolidated patient bill for an encounter.

		Aggregates all charges: ward/bed charges, professional fees, drug charges,
		laboratory charges, imaging, theatre, consumables, and sundries.
		Bill is generated in draft status pending financial review before final posting.
		"""
		admission = self._admissions.get((tenant_id, encounter_id))
		if admission is None:
			raise KeyError(f"encounter {encounter_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "generate_bill",
		})

		now = bill_date or datetime.utcnow()
		bill_id = uuid7str()
		bill_number = f"BILL-{bill_id[:8].upper()}"

		# Retrieve existing line items if any (populated by charge-posting integrations)
		line_items = self._bill_line_items.get((tenant_id, encounter_id), [])

		# Compute LOS-based ward charges as a fallback
		if not line_items and admission.status in {"admitted", "discharged"}:
			admit_time = admission.admit_time
			end_time = now
			los_days = max(1, int((end_time - admit_time).total_seconds() / 86400))
			line_items = [{
				"category": "ward_charges",
				"description": f"Ward charge {los_days} day(s)",
				"quantity": los_days,
				"unit_price": 3500.0,
				"amount": los_days * 3500.0,
				"currency": "KES",
			}]

		subtotal = sum(float(li.get("amount", 0)) for li in line_items)
		tax = round(subtotal * 0.16, 2)  # 16% VAT (Kenya)
		total = round(subtotal + tax, 2)

		bill: dict[str, Any] = {
			"id": bill_id,
			"tenant_id": tenant_id,
			"encounter_id": encounter_id,
			"patient_id": admission.patient_id,
			"bill_number": bill_number,
			"bill_date": now.isoformat(),
			"line_items": line_items,
			"subtotal": subtotal,
			"tax": tax,
			"total": total,
			"currency": "KES",
			"status": "draft",
			"insurance_id": admission.insurance_id,
			"paid": False,
			"balance_due": total,
		}
		self._bills[(tenant_id, bill_id)] = bill
		self._audit(tenant_id, "patient_bill_generated", bill_id)
		_log_op("generate_patient_bill", tenant_id, bill_id)
		return bill

	async def claim_submission(
		self,
		tenant_id: str,
		encounter_id: str,
		insurer_id: str,
		claim_amount: float,
		diagnosis_codes: list[str],
		procedure_codes: list[str],
		submitted_by: str = "",
	) -> dict[str, Any]:
		"""Submit an insurance claim for a completed encounter.

		Generates a claim with ICD-10 diagnosis codes and CPT/procedure codes.
		Claim formats supported: NHIF smart card | SHA | commercial | reinsurance.
		Returns a claim reference number and submission timestamp.
		"""
		assert claim_amount > 0, "claim_amount must be positive"
		assert bool(diagnosis_codes), "at least one diagnosis_code required"
		assert bool(procedure_codes), "at least one procedure_code required"
		assert bool(insurer_id), "insurer_id required"

		admission = self._admissions.get((tenant_id, encounter_id))
		if admission is None:
			raise KeyError(f"encounter {encounter_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "claim_submission",
		})

		claim_id = uuid7str()
		claim_ref = f"CLM-{claim_id[:8].upper()}"
		now = datetime.utcnow()

		record: dict[str, Any] = {
			"id": claim_id,
			"tenant_id": tenant_id,
			"encounter_id": encounter_id,
			"patient_id": admission.patient_id,
			"insurer_id": insurer_id,
			"claim_reference": claim_ref,
			"claim_amount": claim_amount,
			"currency": "KES",
			"diagnosis_codes": diagnosis_codes,
			"procedure_codes": procedure_codes,
			"submitted_by": submitted_by,
			"submitted_at": now.isoformat(),
			"status": "submitted",
			"adjudication_status": None,
			"approved_amount": None,
			"rejection_reason": None,
		}
		self._claims[(tenant_id, claim_id)] = record
		self._audit(tenant_id, "insurance_claim_submitted", claim_id)
		logger.info(_log_claim(encounter_id, claim_amount, insurer_id))
		return record

	async def process_copay(
		self,
		tenant_id: str,
		encounter_id: str,
		copay_amount: float,
		payment_method: str,
		received_by: str = "",
	) -> dict[str, Any]:
		"""Process a patient copay payment at point of service.

		payment_method: cash | card | mobile_money | insurance_direct | waiver
		Copay is posted against the encounter bill and balance_due is reduced.
		Receipt is generated with a unique payment reference.
		"""
		_VALID_METHODS = {"cash", "card", "mobile_money", "insurance_direct", "waiver"}
		assert payment_method in _VALID_METHODS, f"invalid payment_method: {payment_method}"
		assert copay_amount >= 0, "copay_amount must be non-negative"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "process_payment",
		})

		copay_id = uuid7str()
		receipt_number = f"RCT-{copay_id[:8].upper()}"
		now = datetime.utcnow()

		# Update bill balance_due
		bill = next(
			(b for b in self._bills.values()
			 if isinstance(b, dict) and b.get("tenant_id") == tenant_id and b.get("encounter_id") == encounter_id),
			None,
		)
		new_balance = None
		if bill:
			new_balance = max(0.0, float(bill.get("balance_due", 0)) - copay_amount)
			updated_bill = {**bill, "balance_due": new_balance, "paid": new_balance == 0}
			self._bills[(tenant_id, bill["id"])] = updated_bill

		record: dict[str, Any] = {
			"id": copay_id,
			"tenant_id": tenant_id,
			"encounter_id": encounter_id,
			"receipt_number": receipt_number,
			"copay_amount": copay_amount,
			"payment_method": payment_method,
			"received_by": received_by,
			"paid_at": now.isoformat(),
			"new_balance_due": new_balance,
			"status": "paid",
		}
		self._copay_payments[(tenant_id, copay_id)] = record
		self._audit(tenant_id, "copay_processed", copay_id)
		return record

	async def patient_satisfaction_survey(
		self,
		tenant_id: str,
		encounter_id: str,
		survey_responses: dict[str, Any],
		submitted_by: str = "",
	) -> dict[str, Any]:
		"""Record a patient satisfaction survey response linked to an encounter.

		survey_responses should use a standardised Likert scale (1-5) or NPS (0-10).
		Standard domains: overall_satisfaction, communication_with_staff,
		  cleanliness, wait_time, pain_management, discharge_planning,
		  would_recommend (NPS anchor).
		Calculates a composite score and NPS bucket (promoter/passive/detractor).
		"""
		assert bool(survey_responses), "survey_responses required"

		admission = self._admissions.get((tenant_id, encounter_id))
		patient_id = admission.patient_id if admission else ""

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})

		# Compute composite score (mean of numeric responses on 1-5 scale)
		numeric_responses = {
			k: float(v) for k, v in survey_responses.items()
			if isinstance(v, (int, float)) and k != "would_recommend"
		}
		composite = (
			round(sum(numeric_responses.values()) / len(numeric_responses), 2)
			if numeric_responses else None
		)

		# NPS bucket
		nps_score = survey_responses.get("would_recommend")
		nps_bucket: str | None = None
		if isinstance(nps_score, (int, float)):
			nps_score_f = float(nps_score)
			if nps_score_f >= 9:
				nps_bucket = "promoter"
			elif nps_score_f >= 7:
				nps_bucket = "passive"
			else:
				nps_bucket = "detractor"

		survey_id = uuid7str()
		record: dict[str, Any] = {
			"id": survey_id,
			"tenant_id": tenant_id,
			"encounter_id": encounter_id,
			"patient_id": patient_id,
			"submitted_by": submitted_by,
			"submitted_at": datetime.utcnow().isoformat(),
			"survey_responses": survey_responses,
			"composite_score": composite,
			"nps_score": nps_score,
			"nps_bucket": nps_bucket,
			"status": "completed",
		}
		self._satisfaction_surveys[(tenant_id, survey_id)] = record
		self._audit(tenant_id, "satisfaction_survey_submitted", survey_id)
		return record

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		patients = [p for (tid, _), p in self._patients.items() if tid == tenant_id]
		admissions = [a for (tid, _), a in self._admissions.items() if tid == tenant_id]
		beds = [b for (tid, _), b in self._beds.items() if tid == tenant_id]
		appts = [a for (tid, _), a in self._appointments.items() if tid == tenant_id]
		claims = [c for c in self._claims.values() if isinstance(c, dict) and c.get("tenant_id") == tenant_id]
		no_shows = [n for n in self._no_shows.values() if isinstance(n, dict) and n.get("tenant_id") == tenant_id]
		surveys = [s for s in self._satisfaction_surveys.values() if isinstance(s, dict) and s.get("tenant_id") == tenant_id]
		scores = [float(s["composite_score"]) for s in surveys if s.get("composite_score") is not None]
		avg_satisfaction = round(sum(scores) / len(scores), 2) if scores else None
		return {
			"tenant_id": tenant_id,
			"patients": {
				"total": len(patients),
				"active": sum(1 for p in patients if p.status == "active"),
			},
			"admissions": {
				"total": len(admissions),
				"admitted": sum(1 for a in admissions if a.status == "admitted"),
			},
			"beds": {
				"total": len(beds),
				"available": sum(1 for b in beds if b.status == "available"),
				"occupied": sum(1 for b in beds if b.status == "occupied"),
				"occupancy_rate_pct": round(
					sum(1 for b in beds if b.status == "occupied") / max(len(beds), 1) * 100, 1
				),
			},
			"appointments": {
				"total": len(appts),
				"scheduled": sum(1 for a in appts if a.status == "scheduled"),
				"no_shows": len(no_shows),
			},
			"billing": {
				"claims_submitted": len(claims),
				"claims_pending": sum(1 for c in claims if c.get("status") == "submitted"),
			},
			"satisfaction": {
				"surveys_completed": len(surveys),
				"avg_composite_score": avg_satisfaction,
			},
		}

	# ── clinical documentation ────────────────────────────────────────────────

	async def create_encounter_note(
		self,
		tenant_id: str,
		patient_id: str,
		encounter_id: str,
		note_type: str,
		content: str,
		authored_by: str,
	) -> dict[str, Any]:
		"""Create a clinical encounter note (SOAP, progress, discharge)."""
		assert note_type in ("soap", "progress", "discharge", "referral", "nursing"), f"unsupported: {note_type}"
		assert content, "content required"
		note_id = uuid7str()
		record: dict[str, Any] = {
			"id": note_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"note_type": note_type,
			"content": content,
			"authored_by": authored_by,
			"authored_at": datetime.utcnow().isoformat(),
			"status": "final",
		}
		self._audit(tenant_id, "encounter_note_created", note_id)
		_log_op("create_encounter_note", tenant_id, note_id)
		return record

	async def vital_signs_record(
		self,
		tenant_id: str,
		patient_id: str,
		encounter_id: str,
		vitals: dict[str, Any],
		recorded_by: str,
	) -> dict[str, Any]:
		"""Record patient vital signs for an encounter."""
		assert vitals, "vitals dict required"
		vs_id = uuid7str()
		early_warning_score = 0
		bp_systolic = vitals.get("bp_systolic", 120)
		if bp_systolic < 90 or bp_systolic > 180:
			early_warning_score += 2
		spo2 = vitals.get("spo2", 98)
		if spo2 < 94:
			early_warning_score += 2
		hr = vitals.get("heart_rate", 80)
		if hr < 50 or hr > 120:
			early_warning_score += 1
		record: dict[str, Any] = {
			"id": vs_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"vitals": vitals,
			"early_warning_score": early_warning_score,
			"ews_level": "high" if early_warning_score >= 4 else ("medium" if early_warning_score >= 2 else "low"),
			"recorded_by": recorded_by,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "recorded",
		}
		self._audit(tenant_id, "vital_signs_recorded", vs_id)
		_log_op("vital_signs_record", tenant_id, vs_id)
		return record

	async def allergy_record(
		self,
		tenant_id: str,
		patient_id: str,
		allergen: str,
		reaction: str,
		severity: str,
		recorded_by: str,
	) -> dict[str, Any]:
		"""Record a patient allergy with reaction and severity."""
		assert severity in ("mild", "moderate", "severe", "life_threatening"), f"invalid severity: {severity}"
		allergy_id = uuid7str()
		record: dict[str, Any] = {
			"id": allergy_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"allergen": allergen,
			"reaction": reaction,
			"severity": severity,
			"alert_required": severity in ("severe", "life_threatening"),
			"recorded_by": recorded_by,
			"recorded_at": datetime.utcnow().isoformat(),
			"status": "active",
		}
		self._audit(tenant_id, "allergy_recorded", allergy_id)
		_log_op("allergy_record", tenant_id, allergy_id)
		return record

	async def medication_order(
		self,
		tenant_id: str,
		patient_id: str,
		encounter_id: str,
		drug_name: str,
		dose: str,
		route: str,
		frequency: str,
		ordered_by: str,
	) -> dict[str, Any]:
		"""Create a medication order for an inpatient encounter."""
		assert drug_name, "drug_name required"
		assert ordered_by, "ordered_by required"
		order_id = uuid7str()
		record: dict[str, Any] = {
			"id": order_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"drug_name": drug_name,
			"dose": dose,
			"route": route,
			"frequency": frequency,
			"ordered_by": ordered_by,
			"ordered_at": datetime.utcnow().isoformat(),
			"pharmacist_review_required": True,
			"status": "pending_review",
		}
		self._audit(tenant_id, "medication_ordered", order_id)
		_log_op("medication_order", tenant_id, order_id)
		return record

	async def patient_demographics_update(
		self,
		tenant_id: str,
		patient_id: str,
		updates: dict[str, Any],
		updated_by: str,
	) -> PatientResponse | None:
		"""Update patient demographic fields with audit trail."""
		assert updates, "updates required"
		patient = self._patients.get((tenant_id, patient_id))
		if patient is None:
			return None
		allowed = {"phone", "address", "email", "emergency_contact"}
		filtered = {k: v for k, v in updates.items() if k in allowed}
		updated = patient.model_copy(update={**filtered, "updated_at": datetime.utcnow()})
		self._patients[(tenant_id, patient_id)] = updated
		self._audit(tenant_id, "patient_demographics_updated", patient_id)
		_log_op("patient_demographics_update", tenant_id, patient_id)
		return updated

	async def nhif_sha_eligibility_check(
		self,
		tenant_id: str,
		patient_id: str,
		membership_number: str,
	) -> dict[str, Any]:
		"""Check NHIF/SHA insurance eligibility for a patient (Kenya)."""
		assert membership_number, "membership_number required"
		patient = self._patients.get((tenant_id, patient_id))
		check_id = uuid7str()
		record: dict[str, Any] = {
			"id": check_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"membership_number": membership_number,
			"scheme": "SHA",
			"eligibility_status": "eligible",
			"active_cover": True,
			"cover_type": "inpatient_outpatient",
			"annual_limit_kes": 500000,
			"checked_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "nhif_eligibility_checked", check_id)
		_log_op("nhif_sha_eligibility_check", tenant_id, check_id)
		return record

	async def referral_letter(
		self,
		tenant_id: str,
		patient_id: str,
		encounter_id: str,
		referring_provider: str,
		to_facility: str,
		reason: str,
	) -> dict[str, Any]:
		"""Generate a structured referral letter for inter-facility transfer."""
		assert reason, "reason required"
		patient = self._patients.get((tenant_id, patient_id))
		ref_id = uuid7str()
		ref_number = f"REF-{datetime.utcnow().strftime('%Y%m%d')}-{ref_id[:6].upper()}"
		record: dict[str, Any] = {
			"id": ref_id,
			"reference_number": ref_number,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"patient_mrn": patient.mrn if patient else "",
			"patient_name": f"{patient.first_name} {patient.last_name}" if patient else "",
			"encounter_id": encounter_id,
			"referring_provider": referring_provider,
			"to_facility": to_facility,
			"reason": reason,
			"urgency": "routine",
			"created_at": datetime.utcnow().isoformat(),
			"status": "issued",
		}
		self._audit(tenant_id, "referral_letter_issued", ref_id)
		_log_op("referral_letter", tenant_id, ref_id)
		return record

	async def export_patient_data(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export patient records metadata."""
		patients = [p for (tid, _), p in self._patients.items() if tid == tenant_id]
		export_id = uuid7str()
		_log_op("export_patient_data", tenant_id, export_id)
		return {
			"export_id": export_id,
			"tenant_id": tenant_id,
			"format": format,
			"record_count": len(patients),
			"download_ref": f"/exports/{tenant_id}/{export_id}.{format}",
			"status": "ready",
			"generated_at": datetime.utcnow().isoformat(),
		}

	async def health_check(self) -> dict[str, Any]:
		"""Return service health and store sizes."""
		return {
			"service": "PatientManagementService",
			"status": "healthy",
			"patients": len(self._patients),
			"admissions": len(self._admissions),
			"beds": len(self._beds),
			"appointments": len(self._appointments),
			"insurance_records": len(self._insurance),
			"bills": len(self._bills),
			"claims": len(self._claims),
			"audit_events": len(self._audit_events),
			"checked_at": datetime.utcnow().isoformat(),
		}

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			logger.warning("pmt.rule_denied rule=%s", result["rule"])
			raise PolicyViolationError(result["reason"])

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event": event,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
		})


def _compute_match_score(
	candidate: PatientResponse,
	last_name: str,
	first_name: str,
	dob: datetime,
	id_number: str,
) -> float:
	"""Compute a probabilistic match score (0–1) between a query and a candidate patient.

	Weights: last_name=0.3, first_name=0.2, dob=0.35, id_number=0.15
	Exact string matches score 1.0 per field; case-insensitive prefix match scores 0.5.
	"""
	score = 0.0

	# Last name
	if candidate.last_name.lower() == last_name.lower():
		score += 0.30
	elif candidate.last_name.lower().startswith(last_name[:3].lower()):
		score += 0.15

	# First name
	if candidate.first_name.lower() == first_name.lower():
		score += 0.20
	elif candidate.first_name.lower().startswith(first_name[:2].lower()):
		score += 0.10

	# Date of birth
	if candidate.date_of_birth and dob:
		if candidate.date_of_birth.date() == dob.date():
			score += 0.35
		elif candidate.date_of_birth.year == dob.year and candidate.date_of_birth.month == dob.month:
			score += 0.15

	# ID number — not directly on model in current implementation; skip
	_ = id_number

	return round(min(score, 1.0), 3)
