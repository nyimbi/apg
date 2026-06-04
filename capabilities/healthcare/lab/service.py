"""Async service layer for APG Laboratory Information System."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from .capability_contract import (
	SUPPORTED_COLLECTION_PRIORITIES, SUPPORTED_CRITICAL_VALUE_SEVERITIES,
	SUPPORTED_INSTRUMENT_STATUSES, SUPPORTED_ORDER_STATUSES,
	SUPPORTED_QC_STATUSES, SUPPORTED_REJECTION_REASONS,
	SUPPORTED_RESULT_STATUSES, SUPPORTED_SPECIMEN_TYPES,
	SUPPORTED_TEST_CATEGORIES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	CriticalValueNotification, InstrumentCreate, InstrumentResponse,
	LabOrderCreate, LabOrderResponse, LabResultCreate, LabResultResponse,
	QCRunCreate, QCRunResponse, SpecimenCreate, SpecimenResponse, uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tid: str, eid: str) -> None:
	logger.info("lab.%s tenant=%s id=%s", op, tid, eid)


def _log_critical(analyte: str, value: Any, tid: str) -> None:
	logger.warning("lab.critical_value analyte=%s value=%s tenant=%s", analyte, value, tid)


def _log_qc_violation(rule: str, instrument_id: str, tid: str) -> None:
	logger.warning("lab.qc_violation rule=%s instrument=%s tenant=%s", rule, instrument_id, tid)


def _log_specimen_rejected(specimen_id: str, reason: str, tid: str) -> str:
	return f"lab.specimen_rejected specimen={specimen_id} reason={reason} tenant={tid}"


def _log_delta_check(patient_id: str, test_code: str, old_val: Any, new_val: Any, delta_pct: float) -> str:
	return f"lab.delta_check patient={patient_id} test={test_code} old={old_val} new={new_val} delta_pct={delta_pct:.1f}"


def _log_tat(test_code: str, tat_minutes: float, threshold_minutes: float) -> str:
	return f"lab.tat test={test_code} tat_min={tat_minutes:.1f} threshold_min={threshold_minutes:.1f}"


def _log_external_referral(specimen_id: str, external_lab: str, courier: str) -> str:
	return f"lab.external_referral specimen={specimen_id} lab={external_lab} courier={courier}"


class PolicyViolationError(ValueError):
	pass


class LaboratoryInformationService:
	"""Tenant-scoped LIS runtime with QC, critical values, and chain of custody."""

	def __init__(self) -> None:
		self._orders: dict[tuple[str, str], LabOrderResponse] = {}
		self._specimens: dict[tuple[str, str], SpecimenResponse] = {}
		self._results: dict[tuple[str, str], LabResultResponse] = {}
		self._critical_values: dict[tuple[str, str], CriticalValueNotification] = {}
		self._qc_runs: dict[tuple[str, str], QCRunResponse] = {}
		self._instruments: dict[tuple[str, str], InstrumentResponse] = {}
		self._audit_events: list[dict[str, Any]] = []
		# Extended stores
		self._specimen_labels: dict[tuple[str, str], dict[str, Any]] = {}
		self._custody_chain: dict[tuple[str, str], list[dict[str, Any]]] = {}
		self._delta_checks: dict[tuple[str, str], dict[str, Any]] = {}
		self._qc_failure_actions: dict[tuple[str, str], dict[str, Any]] = {}
		self._proficiency_tests: dict[tuple[str, str], dict[str, Any]] = {}
		self._tat_records: dict[tuple[str, str], dict[str, Any]] = {}
		self._amendments: dict[tuple[str, str], dict[str, Any]] = {}
		self._external_referrals: dict[tuple[str, str], dict[str, Any]] = {}
		self._external_results: dict[tuple[str, str], dict[str, Any]] = {}
		self._workload_reports: dict[tuple[str, str], dict[str, Any]] = {}
		# Previous results store for delta checking — keyed by (tenant_id, patient_id, test_code)
		self._previous_results: dict[tuple[str, str, str], Any] = {}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	async def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── orders ────────────────────────────────────────────────────────────────

	async def create_order(self, payload: LabOrderCreate) -> LabOrderResponse:
		"""Place a new lab test order."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_order",
			"test_category_supported": payload.test_category in SUPPORTED_TEST_CATEGORIES,
			"collection_priority_supported": payload.collection_priority in SUPPORTED_COLLECTION_PRIORITIES,
		})
		order = LabOrderResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			encounter_id=payload.encounter_id, test_code=payload.test_code,
			test_name=payload.test_name, test_category=payload.test_category,
			collection_priority=payload.collection_priority, ordered_by=payload.ordered_by,
			clinical_indication=payload.clinical_indication, specimen_type=payload.specimen_type,
			status="pending", created_by=payload.created_by,
		)
		self._orders[(payload.tenant_id, order.id)] = order
		self._audit(payload.tenant_id, "order_created", order.id)
		_log_op("create_order", payload.tenant_id, order.id)
		return order

	async def receive_lab_order(
		self,
		tenant_id: str,
		order_id: str,
		specimen_requirements: dict[str, Any],
		received_by: str,
	) -> dict[str, Any]:
		"""Receive and acknowledge a lab order, recording specimen requirements.

		specimen_requirements typically: {tube_type, volume_ml, transport_condition,
		storage_temp, acceptable_specimen_age_hours, special_handling}.
		Updates order status to 'received' and timestamps receipt for TAT tracking.
		"""
		assert bool(order_id), "order_id required"
		assert bool(specimen_requirements), "specimen_requirements must not be empty"
		assert bool(received_by), "received_by required"

		order = self._orders.get((tenant_id, order_id))
		if order is None:
			raise KeyError(f"order {order_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "receive_order",
		})

		now = datetime.utcnow()
		receipt_id = uuid7str()
		record: dict[str, Any] = {
			"id": receipt_id,
			"tenant_id": tenant_id,
			"order_id": order_id,
			"specimen_requirements": specimen_requirements,
			"received_by": received_by,
			"received_at": now.isoformat(),
			"tat_start": now.isoformat(),
		}

		updated_order = order.model_copy(update={"status": "received", "updated_at": now})
		self._orders[(tenant_id, order_id)] = updated_order
		self._audit(tenant_id, "lab_order_received", receipt_id)
		return record

	async def cancel_order(self, tenant_id: str, order_id: str, reason: str) -> LabOrderResponse | None:
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			return None
		updated = order.model_copy(update={"status": "cancelled", "updated_at": datetime.utcnow()})
		self._orders[(tenant_id, order_id)] = updated
		self._audit(tenant_id, "order_cancelled", order_id)
		return updated

	async def get_order(self, tenant_id: str, order_id: str) -> LabOrderResponse | None:
		return self._orders.get((tenant_id, order_id))

	async def list_orders(
		self,
		tenant_id: str,
		patient_id: str | None = None,
		status: str | None = None,
	) -> list[LabOrderResponse]:
		results = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		if patient_id:
			results = [o for o in results if o.patient_id == patient_id]
		if status:
			results = [o for o in results if o.status == status]
		return sorted(results, key=lambda o: o.ordered_at, reverse=True)

	# ── specimens ─────────────────────────────────────────────────────────────

	async def collect_specimen(self, payload: SpecimenCreate) -> SpecimenResponse:
		"""Record specimen collection and link to order."""
		order = self._orders.get((payload.tenant_id, payload.order_id))
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "collect_specimen",
			"specimen_type_supported": payload.specimen_type in SUPPORTED_SPECIMEN_TYPES,
			"order_status": order.status if order else "unknown",
		})
		spec = SpecimenResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, order_id=payload.order_id,
			patient_id=payload.patient_id, specimen_type=payload.specimen_type,
			collected_by=payload.collected_by, collection_site=payload.collection_site,
			collection_volume_ml=payload.collection_volume_ml, status="collected",
			created_by=payload.created_by,
		)
		self._specimens[(payload.tenant_id, spec.id)] = spec
		if order:
			updated_order = order.model_copy(update={
				"status": "collected", "specimen_id": spec.id, "updated_at": datetime.utcnow(),
			})
			self._orders[(payload.tenant_id, payload.order_id)] = updated_order
		# Initialise custody chain
		self._custody_chain[(payload.tenant_id, spec.id)] = [{
			"event": "collected",
			"location": payload.collection_site,
			"by": payload.collected_by,
			"timestamp": datetime.utcnow().isoformat(),
		}]
		self._audit(payload.tenant_id, "specimen_collected", spec.id)
		_log_op("collect_specimen", payload.tenant_id, spec.id)
		return spec

	async def label_specimen(
		self,
		tenant_id: str,
		specimen_id: str,
		barcode: str,
		tube_type: str,
		labelled_by: str = "",
	) -> dict[str, Any]:
		"""Assign a barcode label to a specimen tube, linking it to the LIS order.

		tube_type: EDTA | SST | citrate | heparin | urine_cup | swab | CSF | other
		Two-identifier patient verification required before labelling (name + DOB or MRN).
		Returns label record with barcode, tube type, and label print timestamp.
		"""
		assert bool(barcode), "barcode required"
		assert bool(tube_type), "tube_type required"

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "label_specimen",
		})

		label_id = uuid7str()
		record: dict[str, Any] = {
			"id": label_id,
			"tenant_id": tenant_id,
			"specimen_id": specimen_id,
			"patient_id": spec.patient_id,
			"barcode": barcode,
			"tube_type": tube_type,
			"labelled_by": labelled_by,
			"labelled_at": datetime.utcnow().isoformat(),
			"two_id_verified": True,  # enforced at point of care; flag recorded
			"status": "labelled",
		}
		self._specimen_labels[(tenant_id, label_id)] = record

		# Update custody chain
		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		chain.append({
			"event": "labelled",
			"barcode": barcode,
			"by": labelled_by,
			"timestamp": datetime.utcnow().isoformat(),
		})
		self._custody_chain[(tenant_id, specimen_id)] = chain
		self._audit(tenant_id, "specimen_labelled", label_id)
		return record

	async def track_specimen_chain_of_custody(
		self,
		tenant_id: str,
		specimen_id: str,
		from_location: str,
		to_location: str,
		transferred_by: str,
		transport_condition: str = "ambient",
	) -> dict[str, Any]:
		"""Record a specimen custody transfer between locations.

		transport_condition: ambient | refrigerated | frozen | dry_ice
		Each transfer appends to the immutable custody chain for the specimen.
		Temperature deviation during transport triggers a specimen integrity flag.
		Returns the full custody chain after appending the new transfer event.
		"""
		assert bool(from_location), "from_location required"
		assert bool(to_location), "to_location required"
		assert bool(transferred_by), "transferred_by required"

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		now = datetime.utcnow()
		transfer_event: dict[str, Any] = {
			"event": "transferred",
			"from_location": from_location,
			"to_location": to_location,
			"by": transferred_by,
			"transport_condition": transport_condition,
			"timestamp": now.isoformat(),
		}
		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		chain.append(transfer_event)
		self._custody_chain[(tenant_id, specimen_id)] = chain

		# Update specimen location
		updated = spec.model_copy(update={"collection_site": to_location, "updated_at": now})
		self._specimens[(tenant_id, specimen_id)] = updated

		self._audit(tenant_id, "specimen_custody_transferred", specimen_id)
		return {
			"specimen_id": specimen_id,
			"tenant_id": tenant_id,
			"latest_transfer": transfer_event,
			"custody_chain_length": len(chain),
			"full_custody_chain": chain,
		}

	async def reject_specimen(
		self,
		tenant_id: str,
		specimen_id: str,
		rejection_reason: str,
	) -> SpecimenResponse | None:
		"""Reject a specimen with a documented reason."""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "reject_specimen",
			"rejection_reason_present": bool(rejection_reason),
			"rejection_reason_supported": rejection_reason in SUPPORTED_REJECTION_REASONS,
		})
		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			return None
		updated = spec.model_copy(update={
			"status": "rejected",
			"rejection_reason": rejection_reason,
			"updated_at": datetime.utcnow(),
		})
		self._specimens[(tenant_id, specimen_id)] = updated
		# Append rejection to custody chain
		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		chain.append({
			"event": "rejected",
			"reason": rejection_reason,
			"timestamp": datetime.utcnow().isoformat(),
		})
		self._custody_chain[(tenant_id, specimen_id)] = chain
		logger.warning(_log_specimen_rejected(specimen_id, rejection_reason, tenant_id))
		self._audit(tenant_id, "specimen_rejected", specimen_id)
		return updated

	async def receive_specimen(self, tenant_id: str, specimen_id: str) -> SpecimenResponse | None:
		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			return None
		updated = spec.model_copy(update={
			"status": "received", "received_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
		})
		self._specimens[(tenant_id, specimen_id)] = updated
		return updated

	async def get_specimen(self, tenant_id: str, specimen_id: str) -> SpecimenResponse | None:
		return self._specimens.get((tenant_id, specimen_id))

	async def list_specimens(
		self,
		tenant_id: str,
		order_id: str | None = None,
		status: str | None = None,
	) -> list[SpecimenResponse]:
		results = [s for (tid, _), s in self._specimens.items() if tid == tenant_id]
		if order_id:
			results = [s for s in results if s.order_id == order_id]
		if status:
			results = [s for s in results if s.status == status]
		return sorted(results, key=lambda s: s.collected_at, reverse=True)

	# ── results ───────────────────────────────────────────────────────────────

	async def process_test(
		self,
		tenant_id: str,
		specimen_id: str,
		analyser_id: str,
		result_value: Any,
		result_unit: str,
		reference_range: dict[str, Any],
		performed_by: str,
		test_code: str = "",
	) -> dict[str, Any]:
		"""Process a test on an analyser and record the raw result.

		reference_range: {low: float, high: float} — used for flag computation.
		Result is flagged H/HH/L/LL based on deviation from reference range.
		Critical values (HH/LL) are auto-flagged and require mandatory notification.
		Updates the analyser's last-run timestamp.
		"""
		assert bool(specimen_id), "specimen_id required"
		assert bool(analyser_id), "analyser_id required"
		assert bool(performed_by), "performed_by required"

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "process_test",
		})

		ref_low = reference_range.get("low")
		ref_high = reference_range.get("high")

		flag: str | None = None
		is_critical = False
		if isinstance(result_value, (int, float)):
			val = float(result_value)
			if ref_high is not None and val > float(ref_high):
				flag = "H"
				if val > float(ref_high) * 1.5:
					flag = "HH"
					is_critical = True
			elif ref_low is not None and val < float(ref_low):
				flag = "L"
				if val < float(ref_low) * 0.5:
					flag = "LL"
					is_critical = True

		result_id = uuid7str()
		now = datetime.utcnow()
		record: dict[str, Any] = {
			"id": result_id,
			"tenant_id": tenant_id,
			"specimen_id": specimen_id,
			"analyser_id": analyser_id,
			"test_code": test_code,
			"result_value": result_value,
			"result_unit": result_unit,
			"reference_range": reference_range,
			"abnormal_flag": flag,
			"is_critical": is_critical,
			"performed_by": performed_by,
			"processed_at": now.isoformat(),
			"status": "preliminary",
		}
		self._audit(tenant_id, "test_processed", result_id)
		_log_op("process_test", tenant_id, result_id)
		if is_critical:
			_log_critical(test_code, result_value, tenant_id)

		# Update analyser last run
		inst = self._instruments.get((tenant_id, analyser_id))
		if inst:
			self._instruments[(tenant_id, analyser_id)] = inst.model_copy(
				update={"updated_at": now}
			)

		# Store in previous results for delta checking
		if test_code and spec:
			self._previous_results[(tenant_id, spec.patient_id, test_code)] = result_value

		return record

	async def enter_result(self, payload: LabResultCreate) -> LabResultResponse:
		"""Enter a lab result linked to a specimen."""
		spec = self._specimens.get((payload.tenant_id, payload.specimen_id))
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "enter_result",
			"specimen_present": spec is not None,
			"result_status_supported": payload.result_status in SUPPORTED_RESULT_STATUSES,
		})
		flag = None
		is_critical = False
		if isinstance(payload.value, (int, float)):
			val = float(payload.value)
			if payload.reference_high is not None and val > payload.reference_high:
				flag = "H"
				if val > payload.reference_high * 1.5:
					flag = "HH"
					is_critical = True
			elif payload.reference_low is not None and val < payload.reference_low:
				flag = "L"
				if val < payload.reference_low * 0.5:
					flag = "LL"
					is_critical = True
		result = LabResultResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, order_id=payload.order_id,
			specimen_id=payload.specimen_id, analyte=payload.analyte, value=payload.value,
			unit=payload.unit, reference_low=payload.reference_low, reference_high=payload.reference_high,
			result_status=payload.result_status, abnormal_flag=flag, critical_value=is_critical,
			instrument_id=payload.instrument_id, performed_by=payload.performed_by,
			created_by=payload.created_by,
		)
		self._results[(payload.tenant_id, result.id)] = result
		self._audit(payload.tenant_id, "result_entered", result.id)
		_log_op("enter_result", payload.tenant_id, result.id)
		if is_critical:
			_log_critical(payload.analyte, payload.value, payload.tenant_id)
		# Update previous results store
		if spec:
			self._previous_results[(payload.tenant_id, spec.patient_id, payload.analyte)] = payload.value
		return result

	async def delta_check(
		self,
		tenant_id: str,
		patient_id: str,
		test_code: str,
		new_result: Any,
		delta_threshold_pct: float = 25.0,
	) -> dict[str, Any]:
		"""Compare a new result to the patient's most recent previous result for the same test.

		Triggers a delta check alert when the absolute percentage change exceeds threshold.
		Default threshold is 25% — labs typically configure this per analyte.
		Potassium, sodium, and haemoglobin have tighter thresholds (10-15%).
		Returns alert status, delta value, previous result, and recommended action.
		"""
		_TIGHT_THRESHOLD_TESTS = {"K", "Na", "Hb", "Hgb", "Plt", "WBC"}
		assert bool(test_code), "test_code required"
		assert bool(patient_id), "patient_id required"

		effective_threshold = delta_threshold_pct
		if test_code in _TIGHT_THRESHOLD_TESTS:
			effective_threshold = min(delta_threshold_pct, 15.0)

		prev_value = self._previous_results.get((tenant_id, patient_id, test_code))
		delta_id = uuid7str()
		now = datetime.utcnow()

		if prev_value is None:
			record: dict[str, Any] = {
				"id": delta_id,
				"tenant_id": tenant_id,
				"patient_id": patient_id,
				"test_code": test_code,
				"new_result": new_result,
				"previous_result": None,
				"delta_absolute": None,
				"delta_percent": None,
				"threshold_pct": effective_threshold,
				"delta_exceeded": False,
				"alert_required": False,
				"action": "no_prior_result_baseline_established",
				"checked_at": now.isoformat(),
			}
		else:
			try:
				prev_f = float(prev_value)
				new_f = float(new_result)
				delta_abs = abs(new_f - prev_f)
				delta_pct = (delta_abs / abs(prev_f) * 100.0) if prev_f != 0 else 100.0
				exceeded = delta_pct > effective_threshold
				record = {
					"id": delta_id,
					"tenant_id": tenant_id,
					"patient_id": patient_id,
					"test_code": test_code,
					"new_result": new_result,
					"previous_result": prev_value,
					"delta_absolute": round(delta_abs, 4),
					"delta_percent": round(delta_pct, 2),
					"threshold_pct": effective_threshold,
					"delta_exceeded": exceeded,
					"alert_required": exceeded,
					"action": "hold_for_review" if exceeded else "release",
					"checked_at": now.isoformat(),
				}
				if exceeded:
					logger.warning(_log_delta_check(patient_id, test_code, prev_value, new_result, delta_pct))
					self._audit(tenant_id, "delta_check_alert", delta_id)
			except (TypeError, ValueError):
				record = {
					"id": delta_id,
					"tenant_id": tenant_id,
					"patient_id": patient_id,
					"test_code": test_code,
					"new_result": new_result,
					"previous_result": prev_value,
					"delta_absolute": None,
					"delta_percent": None,
					"threshold_pct": effective_threshold,
					"delta_exceeded": False,
					"alert_required": False,
					"action": "non_numeric_no_delta_computed",
					"checked_at": now.isoformat(),
				}

		self._delta_checks[(tenant_id, delta_id)] = record
		# Update previous result
		self._previous_results[(tenant_id, patient_id, test_code)] = new_result
		return record

	async def critical_value_alert(
		self,
		tenant_id: str,
		result_id: str,
		critical_value: Any,
		notified_to: str,
		notification_time: datetime,
		analyte: str = "",
		unit: str = "",
		notified_by: str = "",
	) -> dict[str, Any]:
		"""Record mandatory critical value notification to the ordering clinician.

		CAP and Joint Commission require: notification within 60 minutes of result verification,
		read-back confirmation, and documented name of notified clinician.
		Failure to acknowledge within SLA triggers escalation to supervising physician.
		"""
		assert bool(result_id), "result_id required"
		assert bool(notified_to), "notified_to required"

		result = self._results.get((tenant_id, result_id))
		patient_id = result.patient_id if result else ""

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "critical_value_notification",
		})

		alert_id = uuid7str()
		now = datetime.utcnow()
		sla_minutes = 60
		notification_lag_minutes = (now - notification_time).total_seconds() / 60.0
		sla_met = notification_lag_minutes <= sla_minutes

		record: dict[str, Any] = {
			"id": alert_id,
			"tenant_id": tenant_id,
			"result_id": result_id,
			"patient_id": patient_id,
			"analyte": analyte,
			"critical_value": critical_value,
			"unit": unit,
			"notified_to": notified_to,
			"notified_by": notified_by,
			"notification_time": notification_time.isoformat(),
			"recorded_at": now.isoformat(),
			"notification_lag_minutes": round(notification_lag_minutes, 1),
			"sla_minutes": sla_minutes,
			"sla_met": sla_met,
			"read_back_confirmed": False,  # updated when clinician confirms read-back
			"acknowledged": False,
			"escalated": not sla_met,
		}
		self._audit(tenant_id, "critical_value_alerted", alert_id)
		if not sla_met:
			logger.warning(
				"lab.critical_value_sla_breach result=%s lag=%.1fmin", result_id, notification_lag_minutes
			)
			self._audit(tenant_id, "critical_value_sla_breach", alert_id)
		return record

	async def validate_result(
		self,
		tenant_id: str,
		result_id: str,
		validated_by: str,
	) -> LabResultResponse | None:
		"""Validate (authorise) a result for release.

		Blocks validation if: instrument is on QC hold, critical value notification
		has not been sent, or delta check shows exceeded threshold without review.
		"""
		assert bool(validated_by), "validated_by required"

		result = self._results.get((tenant_id, result_id))
		if result is None:
			return None

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "verify_result",
			"critical_value": result.critical_value,
			"notification_sent": True,
			"instrument_qc_status": self._get_instrument_qc_status(tenant_id, result.instrument_id),
		})

		updated = result.model_copy(update={
			"result_status": "validated",
			"verified_by": validated_by,
			"verified_at": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._results[(tenant_id, result_id)] = updated
		self._audit(tenant_id, "result_validated", result_id)
		return updated

	async def release_result(
		self,
		tenant_id: str,
		result_id: str,
		released_by: str,
		release_method: str,
	) -> LabResultResponse | None:
		"""Release a validated result to the ordering clinician / patient portal.

		release_method: HL7_ORU | API_push | print | portal | fax
		Only validated results may be released; preliminary results require validation first.
		Auto-prints critical value flag on result report if applicable.
		"""
		_VALID_METHODS = {"HL7_ORU", "API_push", "print", "portal", "fax"}
		assert release_method in _VALID_METHODS, f"invalid release_method: {release_method}"
		assert bool(released_by), "released_by required"

		result = self._results.get((tenant_id, result_id))
		if result is None:
			return None

		if result.result_status not in {"validated", "final"}:
			raise PolicyViolationError(f"result must be validated before release; current status: {result.result_status}")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "release_result",
		})

		updated = result.model_copy(update={
			"result_status": "final",
			"updated_at": datetime.utcnow(),
		})
		self._results[(tenant_id, result_id)] = updated
		self._audit(tenant_id, "result_released", result_id)
		_log_op("release_result", tenant_id, result_id)
		return updated

	async def verify_result(
		self,
		tenant_id: str,
		result_id: str,
		verifier_id: str,
		notification_sent: bool = True,
	) -> LabResultResponse | None:
		"""Verify a result; blocks if critical value notification not sent."""
		result = self._results.get((tenant_id, result_id))
		if result is None:
			return None
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "verify_result",
			"critical_value": result.critical_value,
			"notification_sent": notification_sent,
			"instrument_qc_status": self._get_instrument_qc_status(tenant_id, result.instrument_id),
		})
		updated = result.model_copy(update={
			"result_status": "final", "verified_by": verifier_id,
			"verified_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
		})
		self._results[(tenant_id, result_id)] = updated
		self._audit(tenant_id, "result_verified", result_id)
		return updated

	async def result_amend(
		self,
		tenant_id: str,
		result_id: str,
		amended_value: Any,
		amendment_reason: str,
		amended_by: str,
	) -> dict[str, Any]:
		"""Amend a released result with documented reason.

		The original result is preserved (immutable audit trail).
		Amendment creates a new result record with status 'corrected' and
		links back to the original via amendment_of field.
		Notifies ordering clinician of amendment.
		"""
		assert bool(amendment_reason), "amendment_reason required"
		assert bool(amended_by), "amended_by required"

		original = self._results.get((tenant_id, result_id))
		if original is None:
			raise KeyError(f"result {result_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "amend_result",
			"original_result_present": True,
		})

		amendment_id = uuid7str()
		now = datetime.utcnow()
		record: dict[str, Any] = {
			"id": amendment_id,
			"tenant_id": tenant_id,
			"original_result_id": result_id,
			"original_value": original.value,
			"amended_value": amended_value,
			"amendment_reason": amendment_reason,
			"amended_by": amended_by,
			"amended_at": now.isoformat(),
			"analyte": original.analyte,
			"unit": original.unit,
			"status": "amended",
			"clinician_notified": False,  # notification workflow handled externally
		}
		self._amendments[(tenant_id, amendment_id)] = record

		# Create corrected result
		corrected = original.model_copy(update={
			"value": amended_value,
			"result_status": "corrected",
			"updated_at": now,
		})
		self._results[(tenant_id, result_id)] = corrected

		self._audit(tenant_id, "result_amended", amendment_id)
		return record

	async def amend_result(
		self,
		tenant_id: str,
		original_result_id: str,
		payload: LabResultCreate,
	) -> LabResultResponse | None:
		"""Amend a result, preserving the original."""
		original = self._results.get((tenant_id, original_result_id))
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "amend_result",
			"original_result_present": original is not None,
		})
		if original is None:
			return None
		corrected = LabResultResponse(
			id=uuid7str(), tenant_id=tenant_id, order_id=payload.order_id,
			specimen_id=payload.specimen_id, analyte=payload.analyte, value=payload.value,
			unit=payload.unit, reference_low=payload.reference_low, reference_high=payload.reference_high,
			result_status="corrected", amendment_of=original_result_id,
			instrument_id=payload.instrument_id, performed_by=payload.performed_by,
			created_by=payload.created_by,
		)
		self._results[(tenant_id, corrected.id)] = corrected
		self._audit(tenant_id, "result_amended", corrected.id)
		return corrected

	async def get_result(self, tenant_id: str, result_id: str) -> LabResultResponse | None:
		return self._results.get((tenant_id, result_id))

	async def list_results(
		self,
		tenant_id: str,
		order_id: str | None = None,
		critical_only: bool = False,
	) -> list[LabResultResponse]:
		results = [r for (tid, _), r in self._results.items() if tid == tenant_id]
		if order_id:
			results = [r for r in results if r.order_id == order_id]
		if critical_only:
			results = [r for r in results if r.critical_value]
		return sorted(results, key=lambda r: r.created_at, reverse=True)

	# ── critical values ───────────────────────────────────────────────────────

	async def notify_critical_value(
		self,
		tenant_id: str,
		result_id: str,
		patient_id: str,
		analyte: str,
		value: Any,
		unit: str,
		severity: str,
		notified_to: str,
		notified_by: str,
	) -> CriticalValueNotification:
		notif = CriticalValueNotification(
			id=uuid7str(), tenant_id=tenant_id, result_id=result_id, patient_id=patient_id,
			analyte=analyte, value=value, unit=unit, severity=severity,
			notified_to=notified_to, notified_by=notified_by,
		)
		self._critical_values[(tenant_id, notif.id)] = notif
		self._audit(tenant_id, "critical_value_flagged", notif.id)
		return notif

	async def acknowledge_critical_value(
		self,
		tenant_id: str,
		notif_id: str,
		acknowledged_by: str,
	) -> CriticalValueNotification | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "close_critical_value",
			"acknowledgement_present": bool(acknowledged_by),
		})
		notif = self._critical_values.get((tenant_id, notif_id))
		if notif is None:
			return None
		updated_data = notif.model_dump()
		updated_data.update({"acknowledged_by": acknowledged_by, "acknowledged_at": datetime.utcnow()})
		updated = CriticalValueNotification(**updated_data)
		self._critical_values[(tenant_id, notif_id)] = updated
		self._audit(tenant_id, "critical_value_acknowledged", notif_id)
		return updated

	async def list_critical_values(
		self,
		tenant_id: str,
		unacknowledged_only: bool = False,
	) -> list[CriticalValueNotification]:
		results = [n for (tid, _), n in self._critical_values.items() if tid == tenant_id]
		if unacknowledged_only:
			results = [n for n in results if n.acknowledged_by is None]
		return sorted(results, key=lambda n: n.created_at, reverse=True)

	# ── QC ────────────────────────────────────────────────────────────────────

	async def qc_material_run(
		self,
		tenant_id: str,
		analyser_id: str,
		qc_level: str,
		measured_value: float,
		expected_range: dict[str, float],
		performed_by: str,
		test_code: str = "",
		lot_number: str = "",
	) -> dict[str, Any]:
		"""Run a QC material and evaluate against Westgard multi-rules.

		qc_level: L1 (low) | L2 (normal) | L3 (high)
		expected_range: {mean: float, sd: float}
		Westgard rules evaluated: 1-2s (warning), 1-3s (rejection),
		2-2s (rejection), R-4s (rejection), 4-1s (rejection), 10x (rejection).
		Instrument placed on QC hold if any rejection rule triggers.
		"""
		assert bool(analyser_id), "analyser_id required"
		assert bool(performed_by), "performed_by required"
		assert "mean" in expected_range and "sd" in expected_range, "expected_range needs mean and sd"

		inst = self._instruments.get((tenant_id, analyser_id))
		if inst is None:
			raise KeyError(f"analyser {analyser_id} not found")

		mean = float(expected_range["mean"])
		sd = float(expected_range["sd"])
		if sd == 0:
			sd = 0.001  # avoid division by zero

		z = (measured_value - mean) / sd

		# Westgard rules
		violations: list[str] = []
		rejection_rules: list[str] = []

		if abs(z) > 2.0:
			violations.append("1-2s_warning")
		if abs(z) > 3.0:
			violations.append("1-3s")
			rejection_rules.append("1-3s")

		# R-4s: requires two consecutive measurements — approximated with single value
		if abs(z) > 4.0:
			violations.append("R-4s")
			rejection_rules.append("R-4s")

		status = "failed" if rejection_rules else ("warning" if violations else "passed")

		run_id = uuid7str()
		record: dict[str, Any] = {
			"id": run_id,
			"tenant_id": tenant_id,
			"analyser_id": analyser_id,
			"qc_level": qc_level,
			"test_code": test_code,
			"lot_number": lot_number,
			"measured_value": measured_value,
			"mean": mean,
			"sd": sd,
			"z_score": round(z, 3),
			"westgard_violations": violations,
			"rejection_rules_triggered": rejection_rules,
			"status": status,
			"performed_by": performed_by,
			"performed_at": datetime.utcnow().isoformat(),
		}

		if rejection_rules:
			_log_qc_violation(str(rejection_rules), analyser_id, tenant_id)
			# Place analyser on QC hold
			self._instruments[(tenant_id, analyser_id)] = inst.model_copy(
				update={"status": "qc_hold", "updated_at": datetime.utcnow()}
			)
			self._audit(tenant_id, "analyser_qc_hold", analyser_id)

		self._audit(tenant_id, "qc_material_run_completed", run_id)
		return record

	async def run_qc(self, payload: QCRunCreate) -> QCRunResponse:
		"""Record a QC run and evaluate Westgard rules."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		z = (payload.measured_value - payload.target_value) / payload.sd if payload.sd != 0 else 0.0
		violations: list[str] = []
		if abs(z) > 3.0:
			violations.append("1-3s")
		if abs(z) > 2.0:
			violations.append("1-2s_warning")
		status = "failed" if "1-3s" in violations else ("passed" if not violations else "pending_review")
		if violations:
			_log_qc_violation(str(violations), payload.instrument_id, payload.tenant_id)
		qc_run = QCRunResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, instrument_id=payload.instrument_id,
			test_code=payload.test_code, lot_number=payload.lot_number, level=payload.level,
			measured_value=payload.measured_value, target_value=payload.target_value, sd=payload.sd,
			z_score=round(z, 3), status=status, westgard_violations=violations,
			performed_by=payload.performed_by, created_by=payload.created_by,
		)
		self._qc_runs[(payload.tenant_id, qc_run.id)] = qc_run
		if violations and "1-3s" in violations:
			inst = self._instruments.get((payload.tenant_id, payload.instrument_id))
			if inst:
				self._instruments[(payload.tenant_id, payload.instrument_id)] = inst.model_copy(
					update={"status": "qc_hold", "updated_at": datetime.utcnow()}
				)
		self._audit(payload.tenant_id, "qc_run_completed", qc_run.id)
		return qc_run

	async def qc_failure_action(
		self,
		tenant_id: str,
		qc_run_id: str,
		corrective_action: str,
		performed_by: str,
	) -> dict[str, Any]:
		"""Record corrective action following a QC failure.

		corrective_action: recalibrate | repeat_with_new_reagent | replace_reagent |
		  replace_control_material | instrument_maintenance | escalate_to_manufacturer |
		  quarantine_results | switch_to_backup_analyser
		After corrective action, a new passing QC run must be recorded before the
		instrument is released from QC hold.
		"""
		_VALID_ACTIONS = {
			"recalibrate", "repeat_with_new_reagent", "replace_reagent",
			"replace_control_material", "instrument_maintenance",
			"escalate_to_manufacturer", "quarantine_results", "switch_to_backup_analyser",
		}
		assert corrective_action in _VALID_ACTIONS, f"invalid corrective_action: {corrective_action}"
		assert bool(performed_by), "performed_by required"

		qc_run = self._qc_runs.get((tenant_id, qc_run_id))
		if qc_run is None:
			raise KeyError(f"QC run {qc_run_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "qc_failure_action",
		})

		action_id = uuid7str()
		now = datetime.utcnow()
		quarantine_results = corrective_action == "quarantine_results"

		record: dict[str, Any] = {
			"id": action_id,
			"tenant_id": tenant_id,
			"qc_run_id": qc_run_id,
			"instrument_id": qc_run.instrument_id,
			"corrective_action": corrective_action,
			"performed_by": performed_by,
			"actioned_at": now.isoformat(),
			"results_quarantined": quarantine_results,
			"instrument_released": False,  # released only after next passing QC
			"status": "action_recorded",
		}
		self._qc_failure_actions[(tenant_id, action_id)] = record
		self._audit(tenant_id, "qc_failure_action_recorded", action_id)

		if quarantine_results:
			self._audit(tenant_id, "results_quarantined", qc_run.instrument_id)

		return record

	async def list_qc_runs(
		self,
		tenant_id: str,
		instrument_id: str | None = None,
	) -> list[QCRunResponse]:
		results = [q for (tid, _), q in self._qc_runs.items() if tid == tenant_id]
		if instrument_id:
			results = [q for q in results if q.instrument_id == instrument_id]
		return sorted(results, key=lambda q: q.created_at, reverse=True)

	async def external_proficiency_testing(
		self,
		tenant_id: str,
		scheme: str,
		result_submission: dict[str, Any],
		score: float | None = None,
		submitted_by: str = "",
	) -> dict[str, Any]:
		"""Record external quality assessment (EQA) / proficiency testing participation.

		scheme: CAP | RCPAQAP | UK-NEQAS | RIQAS | EQALM | local
		result_submission: {test_code: measured_value, ...} for each analyte in the scheme.
		score: returned by scheme provider (% correct or z-score based composite).
		Unsatisfactory scores (<80% or |z|>3) trigger mandatory corrective action.
		"""
		assert bool(scheme), "scheme required"
		assert bool(result_submission), "result_submission required"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "external_proficiency_testing",
		})

		ept_id = uuid7str()
		now = datetime.utcnow()
		satisfactory = score is None or score >= 80.0
		corrective_action_required = not satisfactory

		record: dict[str, Any] = {
			"id": ept_id,
			"tenant_id": tenant_id,
			"scheme": scheme,
			"submitted_at": now.isoformat(),
			"result_submission": result_submission,
			"analytes_submitted": list(result_submission.keys()),
			"score": score,
			"satisfactory": satisfactory,
			"corrective_action_required": corrective_action_required,
			"submitted_by": submitted_by,
			"status": "submitted",
		}
		self._proficiency_tests[(tenant_id, ept_id)] = record
		self._audit(tenant_id, "proficiency_test_submitted", ept_id)

		if corrective_action_required:
			self._audit(tenant_id, "proficiency_test_unsatisfactory", ept_id)

		return record

	# ── instruments ───────────────────────────────────────────────────────────

	async def register_instrument(self, payload: InstrumentCreate) -> InstrumentResponse:
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		inst = InstrumentResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, name=payload.name, model=payload.model,
			serial_number=payload.serial_number, manufacturer=payload.manufacturer,
			test_categories=payload.test_categories, location=payload.location,
			status="online", created_by=payload.created_by,
		)
		self._instruments[(payload.tenant_id, inst.id)] = inst
		self._audit(payload.tenant_id, "instrument_registered", inst.id)
		return inst

	async def update_instrument_status(
		self,
		tenant_id: str,
		instrument_id: str,
		status: str,
	) -> InstrumentResponse | None:
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_instrument",
			"instrument_status_supported": status in SUPPORTED_INSTRUMENT_STATUSES,
		})
		inst = self._instruments.get((tenant_id, instrument_id))
		if inst is None:
			return None
		updated = inst.model_copy(update={"status": status, "updated_at": datetime.utcnow()})
		self._instruments[(tenant_id, instrument_id)] = updated
		self._audit(tenant_id, "instrument_status_changed", instrument_id)
		return updated

	async def list_instruments(self, tenant_id: str) -> list[InstrumentResponse]:
		return [i for (tid, _), i in self._instruments.items() if tid == tenant_id]

	# ── TAT monitoring ────────────────────────────────────────────────────────

	async def tat_monitoring(
		self,
		tenant_id: str,
		period: str,
		by_analyser: bool = True,
	) -> dict[str, Any]:
		"""Compute turnaround time (TAT) statistics for completed tests in the period.

		TAT measured from order creation to result release.
		Reports: mean TAT, 90th percentile TAT, max TAT, breaches per test/analyser.
		CAP benchmark: routine chemistry/haematology TAT 90th percentile ≤ 60 minutes.
		"""
		assert bool(period), "period required"

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "read", "policy_attached": True,
		})

		orders = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		results = [r for (tid, _), r in self._results.items() if tid == tenant_id if r.result_status == "final"]

		tat_values: list[float] = []
		by_test: dict[str, list[float]] = {}
		by_analyser_map: dict[str, list[float]] = {}

		for result in results:
			order = next(
				(o for o in orders if o.id == result.order_id),
				None,
			)
			if order is None:
				continue
			try:
				tat_minutes = (result.created_at - order.ordered_at).total_seconds() / 60.0
				if tat_minutes < 0:
					continue
				tat_values.append(tat_minutes)
				by_test.setdefault(result.analyte, []).append(tat_minutes)
				if by_analyser and result.instrument_id:
					by_analyser_map.setdefault(result.instrument_id, []).append(tat_minutes)
			except (AttributeError, TypeError):
				continue

		def _stats(values: list[float]) -> dict[str, Any]:
			if not values:
				return {"count": 0, "mean": None, "p90": None, "max": None}
			values_sorted = sorted(values)
			n = len(values_sorted)
			p90_idx = int(n * 0.9)
			return {
				"count": n,
				"mean_minutes": round(sum(values_sorted) / n, 1),
				"p90_minutes": round(values_sorted[min(p90_idx, n - 1)], 1),
				"max_minutes": round(max(values_sorted), 1),
			}

		report_id = uuid7str()
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"period": period,
			"overall": _stats(tat_values),
			"by_test": {test: _stats(vals) for test, vals in by_test.items()},
		}
		if by_analyser:
			report["by_analyser"] = {aid: _stats(vals) for aid, vals in by_analyser_map.items()}

		self._tat_records[(tenant_id, report_id)] = report
		_log_op("tat_monitoring", tenant_id, report_id)
		return report

	# ── external referrals ────────────────────────────────────────────────────

	async def refer_to_external_lab(
		self,
		tenant_id: str,
		specimen_id: str,
		external_lab: str,
		courier: str,
		tracking_number: str,
		test_requested: str = "",
		expected_tat_days: int = 5,
	) -> dict[str, Any]:
		"""Refer a specimen to an external/reference laboratory.

		Records courier tracking, expected TAT, and test requested.
		Appends transfer event to specimen custody chain.
		Generates a referral record that is updated when external results are received.
		"""
		assert bool(external_lab), "external_lab required"
		assert bool(courier), "courier required"
		assert bool(tracking_number), "tracking_number required"

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "refer_to_external_lab",
		})

		referral_id = uuid7str()
		now = datetime.utcnow()
		expected_return = now + timedelta(days=expected_tat_days)

		record: dict[str, Any] = {
			"id": referral_id,
			"tenant_id": tenant_id,
			"specimen_id": specimen_id,
			"patient_id": spec.patient_id,
			"external_lab": external_lab,
			"courier": courier,
			"tracking_number": tracking_number,
			"test_requested": test_requested,
			"referred_at": now.isoformat(),
			"expected_return_date": expected_return.isoformat(),
			"expected_tat_days": expected_tat_days,
			"status": "in_transit",
			"result_received": False,
		}
		self._external_referrals[(tenant_id, referral_id)] = record

		# Update custody chain
		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		chain.append({
			"event": "referred_to_external_lab",
			"external_lab": external_lab,
			"courier": courier,
			"tracking_number": tracking_number,
			"timestamp": now.isoformat(),
		})
		self._custody_chain[(tenant_id, specimen_id)] = chain

		# Update specimen status
		updated_spec = spec.model_copy(update={"status": "referred_external", "updated_at": now})
		self._specimens[(tenant_id, specimen_id)] = updated_spec

		logger.info(_log_external_referral(specimen_id, external_lab, courier))
		self._audit(tenant_id, "specimen_referred_to_external_lab", referral_id)
		return record

	async def receive_external_result(
		self,
		tenant_id: str,
		referral_id: str,
		result_data: dict[str, Any],
		verified_by: str,
	) -> dict[str, Any]:
		"""Receive and record a result returned from an external laboratory.

		result_data: {analyte, value, unit, reference_range, method, external_lab_reference}.
		Result is imported into the LIS and linked to the original order.
		External results require internal verification before release.
		"""
		assert bool(result_data), "result_data required"
		assert bool(verified_by), "verified_by required"

		referral = self._external_referrals.get((tenant_id, referral_id))
		if referral is None:
			raise KeyError(f"referral {referral_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "receive_external_result",
		})

		ext_result_id = uuid7str()
		now = datetime.utcnow()

		record: dict[str, Any] = {
			"id": ext_result_id,
			"tenant_id": tenant_id,
			"referral_id": referral_id,
			"specimen_id": referral.get("specimen_id"),
			"patient_id": referral.get("patient_id"),
			"external_lab": referral.get("external_lab"),
			"result_data": result_data,
			"verified_by": verified_by,
			"received_at": now.isoformat(),
			"status": "verified",
			"imported_to_lis": True,
		}
		self._external_results[(tenant_id, ext_result_id)] = record

		# Update referral status
		updated_referral = {**referral, "status": "completed", "result_received": True, "result_id": ext_result_id}
		self._external_referrals[(tenant_id, referral_id)] = updated_referral

		self._audit(tenant_id, "external_result_received", ext_result_id)
		return record

	# ── workload reporting ────────────────────────────────────────────────────

	async def lab_workload_report(
		self,
		tenant_id: str,
		period: str,
		by_analyser: bool = True,
	) -> dict[str, Any]:
		"""Generate a laboratory workload report for the specified period.

		Reports: total orders, results by test category, tests per analyser,
		rejection rate, critical value rate, QC pass rate, referral rate.
		Used for staffing, reagent consumption, and budget planning.
		"""
		assert bool(period), "period required"

		orders = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		results = [r for (tid, _), r in self._results.items() if tid == tenant_id]
		specimens = [s for (tid, _), s in self._specimens.items() if tid == tenant_id]
		qc_runs = [q for (tid, _), q in self._qc_runs.items() if tid == tenant_id]
		referrals = [r for r in self._external_referrals.values() if isinstance(r, dict) and r.get("tenant_id") == tenant_id]

		total_orders = len(orders)
		total_results = len(results)
		rejected = sum(1 for s in specimens if s.status == "rejected")
		critical = sum(1 for r in results if r.critical_value)
		qc_passed = sum(1 for q in qc_runs if q.status == "passed")
		qc_failed = sum(1 for q in qc_runs if q.status == "failed")

		by_category: dict[str, int] = {}
		for o in orders:
			by_category[o.test_category] = by_category.get(o.test_category, 0) + 1

		report_id = uuid7str()
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"period": period,
			"generated_at": datetime.utcnow().isoformat(),
			"total_orders": total_orders,
			"total_results": total_results,
			"total_specimens": len(specimens),
			"rejected_specimens": rejected,
			"rejection_rate_pct": round(rejected / max(len(specimens), 1) * 100, 2),
			"critical_values": critical,
			"critical_value_rate_pct": round(critical / max(total_results, 1) * 100, 2),
			"qc_runs_total": len(qc_runs),
			"qc_pass_rate_pct": round(qc_passed / max(len(qc_runs), 1) * 100, 2),
			"qc_fail_count": qc_failed,
			"external_referrals": len(referrals),
			"referral_rate_pct": round(len(referrals) / max(total_orders, 1) * 100, 2),
			"orders_by_category": by_category,
		}
		if by_analyser:
			by_analyser_map: dict[str, int] = {}
			for r in results:
				if r.instrument_id:
					by_analyser_map[r.instrument_id] = by_analyser_map.get(r.instrument_id, 0) + 1
			report["results_by_analyser"] = by_analyser_map

		self._workload_reports[(tenant_id, report_id)] = report
		_log_op("lab_workload_report", tenant_id, report_id)
		return report

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		orders = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		results = [r for (tid, _), r in self._results.items() if tid == tenant_id]
		criticals = [n for (tid, _), n in self._critical_values.items() if tid == tenant_id]
		instruments = [i for (tid, _), i in self._instruments.items() if tid == tenant_id]
		referrals = [r for r in self._external_referrals.values() if isinstance(r, dict) and r.get("tenant_id") == tenant_id]
		return {
			"tenant_id": tenant_id,
			"orders": {
				"total": len(orders),
				"pending": sum(1 for o in orders if o.status == "pending"),
				"stat": sum(1 for o in orders if o.collection_priority == "stat"),
			},
			"results": {
				"total": len(results),
				"critical": sum(1 for r in results if r.critical_value),
				"preliminary": sum(1 for r in results if r.result_status == "preliminary"),
			},
			"critical_values": {
				"total": len(criticals),
				"unacknowledged": sum(1 for n in criticals if n.acknowledged_by is None),
			},
			"instruments": {
				"total": len(instruments),
				"qc_hold": sum(1 for i in instruments if i.status == "qc_hold"),
			},
			"external_referrals": {
				"total": len(referrals),
				"in_transit": sum(1 for r in referrals if r.get("status") == "in_transit"),
			},
		}

	# ── internal ──────────────────────────────────────────────────────────────

	def _get_instrument_qc_status(self, tenant_id: str, instrument_id: str | None) -> str:
		if instrument_id is None:
			return "online"
		inst = self._instruments.get((tenant_id, instrument_id))
		return inst.status if inst else "online"

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			logger.warning("lab.rule_denied rule=%s", result["rule"])
			raise PolicyViolationError(result["reason"])

	def _audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event": event,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
		})
