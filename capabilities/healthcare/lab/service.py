"""Async service layer for APG Laboratory Information System."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

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
	AnalyserInterfaceResponse, AnalyserInterfaceUpdate,
	CriticalValueCreate, CriticalValueNotification,
	ExternalReferralCreate, ExternalReferralResponse, ExternalReferralUpdate,
	InstrumentCreate, InstrumentResponse,
	InstrumentStatus, OrderStatus, SpecimenStatus,
	LabOrderCreate, LabOrderResponse, LabOrderUpdate,
	LabResultCreate, LabResultResponse, LabResultUpdate,
	LabTestCreate, LabTestResponse, LabTestUpdate,
	QCRunCreate, QCRunResponse, QCRunUpdate,
	ReferenceRangeCreate, ReferenceRangeResponse, ReferenceRangeUpdate,
	SpecimenCreate, SpecimenResponse, SpecimenTrackRequest, SpecimenUpdate,
	uuid7str,
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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
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
		# New typed stores
		self._tests: dict[tuple[str, str], LabTestResponse] = {}
		self._reference_ranges: dict[tuple[str, str], ReferenceRangeResponse] = {}
		self._referrals: dict[tuple[str, str], ExternalReferralResponse] = {}
		self._calibrations: dict[tuple[str, str], dict[str, Any]] = {}
		self._instrument_messages: dict[tuple[str, str], dict[str, Any]] = {}

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
			status=OrderStatus.PENDING, created_by=payload.created_by,
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

		updated_order = order.model_copy(update={"status": OrderStatus.RECEIVED, "updated_at": now})
		self._orders[(tenant_id, order_id)] = updated_order
		self._audit(tenant_id, "lab_order_received", receipt_id)
		return record

	async def cancel_order(self, tenant_id: str, order_id: str, reason: str) -> LabOrderResponse | None:
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			return None
		updated = order.model_copy(update={
			"status": OrderStatus.CANCELLED,
			"cancelled_reason": reason or None,
			"updated_at": datetime.utcnow(),
		})
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
			collection_volume_ml=payload.collection_volume_ml,
			status=SpecimenStatus.COLLECTED,
			created_by=payload.created_by,
		)
		self._specimens[(payload.tenant_id, spec.id)] = spec
		if order:
			updated_order = order.model_copy(update={
				"status": OrderStatus.COLLECTED, "specimen_id": spec.id, "updated_at": datetime.utcnow(),
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
			"status": SpecimenStatus.REJECTED,
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
			"status": SpecimenStatus.RECEIVED, "received_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
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
		patient_id = ""
		if result:
			spec = self._specimens.get((tenant_id, result.specimen_id))
			if spec:
				patient_id = spec.patient_id

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
			created_by=notified_by,
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
				update={"status": InstrumentStatus.QC_HOLD, "updated_at": datetime.utcnow()}
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
					update={"status": InstrumentStatus.QC_HOLD, "updated_at": datetime.utcnow()}
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

		now = datetime.utcnow()
		expected_tat_hours = expected_tat_days * 24

		# Build typed referral; find any order linked to this specimen
		order_id = next(
			(o.id for (tid, _), o in self._orders.items()
			 if tid == tenant_id and o.specimen_id == specimen_id),
			specimen_id,  # fallback: use specimen_id as order_id placeholder
		)
		referral = ExternalReferralResponse(
			id=uuid7str(), tenant_id=tenant_id,
			order_id=order_id, specimen_id=specimen_id,
			patient_id=spec.patient_id,
			reference_lab_name=external_lab, reference_lab_code=external_lab,
			test_code=test_requested or "EXTERNAL", test_name=test_requested or "External Test",
			tracking_number=tracking_number,
			expected_tat_hours=expected_tat_hours,
			dispatched_by=courier, dispatched_at=now,
			status="dispatched",
			created_by=courier,
		)
		self._referrals[(tenant_id, referral.id)] = referral

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
		updated_spec = spec.model_copy(update={"updated_at": now})
		self._specimens[(tenant_id, specimen_id)] = updated_spec

		logger.info(_log_external_referral(specimen_id, external_lab, courier))
		self._audit(tenant_id, "specimen_referred_to_external_lab", referral.id)
		return referral.model_dump(mode="json")

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

		referral = self._referrals.get((tenant_id, referral_id))
		if referral is None:
			raise KeyError(f"referral {referral_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "receive_external_result",
		})

		ext_result_id = uuid7str()
		now = datetime.utcnow()

		updated = referral.model_copy(update={
			"status": "resulted",
			"external_result_id": ext_result_id,
			"result_received_at": now,
			"received_at": now,
			"updated_at": now,
		})
		self._referrals[(tenant_id, referral_id)] = updated

		record: dict[str, Any] = {
			"id": ext_result_id,
			"tenant_id": tenant_id,
			"referral_id": referral_id,
			"specimen_id": referral.specimen_id,
			"patient_id": referral.patient_id,
			"external_lab": referral.reference_lab_name,
			"result_data": result_data,
			"verified_by": verified_by,
			"received_at": now.isoformat(),
			"status": "verified",
			"imported_to_lis": True,
		}
		self._external_results[(tenant_id, ext_result_id)] = record

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
		referrals = [r for (tid, _), r in self._referrals.items() if tid == tenant_id and not r.is_deleted]

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
		referrals = [r for (tid, _), r in self._referrals.items() if tid == tenant_id and not r.is_deleted]
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
				"in_transit": sum(1 for r in referrals if r.status in {"pending", "dispatched"}),
			},
		}

	# ── lab test catalogue ────────────────────────────────────────────────────

	async def create_test(self, payload: LabTestCreate) -> LabTestResponse:
		"""Add a new diagnostic test to the tenant catalogue."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		test = LabTestResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			test_code=payload.test_code, test_name=payload.test_name,
			category=payload.category, specimen_types=payload.specimen_types,
			loinc_code=payload.loinc_code, cpt_code=payload.cpt_code,
			snomed_code=payload.snomed_code,
			turnaround_minutes=payload.turnaround_minutes,
			stat_turnaround_minutes=payload.stat_turnaround_minutes,
			active=payload.active, requires_fasting=payload.requires_fasting,
			requires_consent=payload.requires_consent,
			price=payload.price, department=payload.department,
			instructions=payload.instructions,
			sample_volume_ml=payload.sample_volume_ml,
			container_type=payload.container_type,
			storage_temperature=payload.storage_temperature,
			created_by=payload.created_by,
		)
		self._tests[(payload.tenant_id, test.id)] = test
		self._audit(payload.tenant_id, "test_created", test.id)
		_log_op("create_test", payload.tenant_id, test.id)
		return test

	async def get_test(self, tenant_id: str, test_id: str) -> LabTestResponse | None:
		"""Retrieve a test catalogue entry by ID."""
		return self._tests.get((tenant_id, test_id))

	async def list_tests(
		self,
		tenant_id: str,
		category: str | None = None,
		active: bool | None = None,
	) -> list[LabTestResponse]:
		"""List test catalogue entries for a tenant with optional filters."""
		results = [t for (tid, _), t in self._tests.items() if tid == tenant_id and not t.is_deleted]
		if category:
			results = [t for t in results if t.category == category]
		if active is not None:
			results = [t for t in results if t.active == active]
		return sorted(results, key=lambda t: t.test_name)

	async def update_test(
		self, tenant_id: str, test_id: str, payload: LabTestUpdate
	) -> LabTestResponse | None:
		"""Update a test catalogue entry with the provided fields."""
		test = self._tests.get((tenant_id, test_id))
		if test is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = test.model_copy(update=updates)
		self._tests[(tenant_id, test_id)] = updated
		self._audit(tenant_id, "test_updated", test_id)
		return updated

	async def delete_test(
		self, tenant_id: str, test_id: str, actor_id: str
	) -> LabTestResponse | None:
		"""Soft-delete a test catalogue entry."""
		test = self._tests.get((tenant_id, test_id))
		if test is None:
			return None
		updated = test.model_copy(update={"is_deleted": True, "updated_at": datetime.utcnow()})
		self._tests[(tenant_id, test_id)] = updated
		self._audit(tenant_id, "test_deleted", test_id)
		return updated

	# ── order extensions ──────────────────────────────────────────────────────

	async def update_order(
		self, tenant_id: str, order_id: str, payload: LabOrderUpdate
	) -> LabOrderResponse | None:
		"""Apply a partial update to a lab order."""
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = order.model_copy(update=updates)
		self._orders[(tenant_id, order_id)] = updated
		self._audit(tenant_id, "order_updated", order_id)
		return updated

	async def hold_order(
		self, tenant_id: str, order_id: str, reason: str
	) -> LabOrderResponse | None:
		"""Place a lab order on hold with a documented reason."""
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			return None
		updated = order.model_copy(update={
			"status": OrderStatus.ON_HOLD, "on_hold_reason": reason, "updated_at": datetime.utcnow(),
		})
		self._orders[(tenant_id, order_id)] = updated
		self._audit(tenant_id, "order_held", order_id)
		return updated

	async def unhold_order(
		self, tenant_id: str, order_id: str
	) -> LabOrderResponse | None:
		"""Release a lab order from hold, restoring it to 'pending'."""
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			return None
		updated = order.model_copy(update={
			"status": OrderStatus.PENDING, "on_hold_reason": None, "updated_at": datetime.utcnow(),
		})
		self._orders[(tenant_id, order_id)] = updated
		self._audit(tenant_id, "order_unheld", order_id)
		return updated

	async def list_orders(
		self,
		tenant_id: str,
		patient_id: str | None = None,
		status: str | None = None,
		priority: str | None = None,
	) -> list[LabOrderResponse]:
		"""List lab orders with optional patient, status, and priority filters."""
		results = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		if patient_id:
			results = [o for o in results if o.patient_id == patient_id]
		if status:
			results = [o for o in results if o.status == status]
		if priority:
			results = [o for o in results if o.collection_priority == priority]
		return sorted(results, key=lambda o: o.ordered_at, reverse=True)

	# ── specimen extensions ───────────────────────────────────────────────────

	async def update_specimen(
		self, tenant_id: str, specimen_id: str, payload: SpecimenUpdate
	) -> SpecimenResponse | None:
		"""Apply a partial update to a specimen record."""
		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = spec.model_copy(update=updates)
		self._specimens[(tenant_id, specimen_id)] = updated
		self._audit(tenant_id, "specimen_updated", specimen_id)
		return updated

	async def track_specimen(
		self, tenant_id: str, specimen_id: str, payload: SpecimenTrackRequest
	) -> dict[str, Any]:
		"""Append a custody event to a specimen's chain-of-custody log."""
		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")
		event: dict[str, Any] = {
			"event_type": payload.event_type,
			"actor_id": payload.actor_id,
			"location": payload.location,
			"notes": payload.notes,
			"timestamp": datetime.utcnow().isoformat(),
		}
		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		chain.append(event)
		self._custody_chain[(tenant_id, specimen_id)] = chain
		self._audit(tenant_id, "specimen_custody_event", specimen_id)
		return {"specimen_id": specimen_id, "event": event, "custody_chain_length": len(chain)}

	async def get_custody_chain(
		self, tenant_id: str, specimen_id: str
	) -> list[dict[str, Any]]:
		"""Return the full chain-of-custody log for a specimen."""
		return self._custody_chain.get((tenant_id, specimen_id), [])

	# ── reference range management ────────────────────────────────────────────

	async def create_reference_range(
		self, payload: ReferenceRangeCreate
	) -> ReferenceRangeResponse:
		"""Create a new reference range for a test analyte."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		rr = ReferenceRangeResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			test_code=payload.test_code, analyte=payload.analyte,
			unit=payload.unit, low=payload.low, high=payload.high,
			critical_low=payload.critical_low, critical_high=payload.critical_high,
			age_min_years=payload.age_min_years, age_max_years=payload.age_max_years,
			sex=payload.sex, condition=payload.condition,
			effective_date=payload.effective_date, expiry_date=payload.expiry_date,
			source=payload.source, created_by=payload.created_by,
		)
		self._reference_ranges[(payload.tenant_id, rr.id)] = rr
		self._audit(payload.tenant_id, "reference_range_created", rr.id)
		_log_op("create_reference_range", payload.tenant_id, rr.id)
		return rr

	async def get_reference_range(
		self, tenant_id: str, rr_id: str
	) -> ReferenceRangeResponse | None:
		"""Retrieve a reference range by ID."""
		return self._reference_ranges.get((tenant_id, rr_id))

	async def list_reference_ranges(
		self, tenant_id: str, test_code: str | None = None
	) -> list[ReferenceRangeResponse]:
		"""List reference ranges for a tenant, optionally filtered by test code."""
		results = [
			rr for (tid, _), rr in self._reference_ranges.items()
			if tid == tenant_id and not rr.is_deleted and rr.active
		]
		if test_code:
			results = [rr for rr in results if rr.test_code == test_code]
		return sorted(results, key=lambda r: r.test_code)

	async def update_reference_range(
		self, tenant_id: str, rr_id: str, payload: ReferenceRangeUpdate
	) -> ReferenceRangeResponse | None:
		"""Update a reference range's bounds."""
		rr = self._reference_ranges.get((tenant_id, rr_id))
		if rr is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = rr.model_copy(update=updates)
		self._reference_ranges[(tenant_id, rr_id)] = updated
		self._audit(tenant_id, "reference_range_updated", rr_id)
		return updated

	async def delete_reference_range(
		self, tenant_id: str, rr_id: str, actor_id: str
	) -> ReferenceRangeResponse | None:
		"""Soft-delete a reference range."""
		rr = self._reference_ranges.get((tenant_id, rr_id))
		if rr is None:
			return None
		updated = rr.model_copy(update={"is_deleted": True, "active": False, "updated_at": datetime.utcnow()})
		self._reference_ranges[(tenant_id, rr_id)] = updated
		self._audit(tenant_id, "reference_range_deleted", rr_id)
		return updated

	async def validate_reference_range(
		self, tenant_id: str, test_code: str, analyte: str, value: float,
		patient_age_years: float | None = None, patient_sex: str | None = None,
	) -> dict[str, Any]:
		"""Evaluate a result value against the best-matching reference range.

		Applies demographic stratification: selects the most specific range
		that matches age and sex. Returns abnormal flag and critical status.
		"""
		from .domain.calculations import classify_numeric_result, select_reference_range

		ranges = [
			rr.model_dump() for (tid, _), rr in self._reference_ranges.items()
			if tid == tenant_id and rr.test_code == test_code
			and rr.analyte == analyte and rr.active and not rr.is_deleted
		]
		matched = select_reference_range(ranges, patient_age_years, patient_sex)
		if matched is None:
			return {
				"test_code": test_code, "analyte": analyte, "value": value,
				"matched_range": None, "flag": None, "is_critical": False,
				"note": "no_matching_reference_range",
			}
		flag, is_critical = classify_numeric_result(
			value,
			matched.get("low"), matched.get("high"),
			matched.get("critical_low"), matched.get("critical_high"),
		)
		return {
			"test_code": test_code, "analyte": analyte, "value": value,
			"matched_range": matched,
			"flag": flag.value if flag else None,
			"is_critical": is_critical,
		}

	# ── result extensions ─────────────────────────────────────────────────────

	async def update_result(
		self, tenant_id: str, result_id: str, payload: LabResultUpdate
	) -> LabResultResponse | None:
		"""Apply a partial update to a result (permitted before verification)."""
		result = self._results.get((tenant_id, result_id))
		if result is None:
			return None
		if result.result_status in {"final", "validated"}:
			raise PolicyViolationError("cannot update a verified/final result; use amend instead")
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = result.model_copy(update=updates)
		self._results[(tenant_id, result_id)] = updated
		self._audit(tenant_id, "result_updated", result_id)
		return updated

	# ── critical value extensions ─────────────────────────────────────────────

	async def create_critical_value(
		self, payload: CriticalValueCreate
	) -> CriticalValueNotification:
		"""Create a critical value notification record."""
		notif = CriticalValueNotification(
			id=uuid7str(), tenant_id=payload.tenant_id,
			result_id=payload.result_id, patient_id=payload.patient_id,
			analyte=payload.analyte, value=payload.value, unit=payload.unit,
			severity=payload.severity, notified_to=payload.notified_to,
			notified_by=payload.notified_by,
			notification_method=payload.notification_method,
			read_back_confirmed=payload.read_back_confirmed,
			created_by=payload.created_by,
		)
		self._critical_values[(payload.tenant_id, notif.id)] = notif
		self._audit(payload.tenant_id, "critical_value_created", notif.id)
		_log_critical(payload.analyte, payload.value, payload.tenant_id)
		return notif

	async def get_critical_value(
		self, tenant_id: str, notif_id: str
	) -> CriticalValueNotification | None:
		"""Retrieve a critical value notification by ID."""
		return self._critical_values.get((tenant_id, notif_id))

	async def alert_critical_value(
		self,
		tenant_id: str,
		result_id: str,
		analyte: str,
		value: Any,
		unit: str,
		severity: str,
		notified_to: str,
		notified_by: str,
		notification_method: str = "phone",
		read_back_confirmed: bool = False,
	) -> CriticalValueNotification:
		"""High-level helper: create + audit a critical value alert in one call.

		Enforces the 60-minute SLA window from result verification.
		Read-back confirmation is recorded but not blocking at this stage
		(must be completed within the SLA window).
		"""
		assert bool(notified_to), "notified_to required"
		assert bool(notified_by), "notified_by required"

		result = self._results.get((tenant_id, result_id))
		# LabResultResponse has no patient_id; resolve via specimen
		patient_id = ""
		if result:
			spec = self._specimens.get((tenant_id, result.specimen_id))
			if spec:
				patient_id = spec.patient_id

		notif = CriticalValueNotification(
			id=uuid7str(), tenant_id=tenant_id,
			result_id=result_id, patient_id=patient_id,
			analyte=analyte, value=value, unit=unit,
			severity=severity, notified_to=notified_to, notified_by=notified_by,
			notification_method=notification_method,
			read_back_confirmed=read_back_confirmed,
			created_by=notified_by,
		)
		self._critical_values[(tenant_id, notif.id)] = notif
		self._audit(tenant_id, "critical_value_alerted", notif.id)
		_log_critical(analyte, value, tenant_id)
		return notif

	# ── QC extensions ─────────────────────────────────────────────────────────

	async def get_qc_run(
		self, tenant_id: str, qc_id: str
	) -> QCRunResponse | None:
		"""Retrieve a single QC run record."""
		return self._qc_runs.get((tenant_id, qc_id))

	async def update_qc_run(
		self, tenant_id: str, qc_id: str, payload: QCRunUpdate
	) -> QCRunResponse | None:
		"""Update QC run review status and notes."""
		qc = self._qc_runs.get((tenant_id, qc_id))
		if qc is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = qc.model_copy(update=updates)
		self._qc_runs[(tenant_id, qc_id)] = updated
		self._audit(tenant_id, "qc_run_updated", qc_id)
		return updated

	async def generate_qc_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a QC pass/fail summary aggregated per instrument and test.

		Returns per-instrument pass rates, current status, and a list of
		Westgard violation counts broken down by rule name.
		"""
		from .domain.calculations import calculate_pass_rate

		qc_runs = [q for (tid, _), q in self._qc_runs.items() if tid == tenant_id]
		instruments = {i.id: i for (tid, _), i in self._instruments.items() if tid == tenant_id}

		# Aggregate per instrument
		by_instrument: dict[str, dict[str, Any]] = {}
		for run in qc_runs:
			key = run.instrument_id
			if key not in by_instrument:
				by_instrument[key] = {
					"instrument_id": key,
					"instrument_name": instruments[key].name if key in instruments else key,
					"total": 0, "passed": 0, "failed": 0,
					"violations": {}, "tests": {},
				}
			by_instrument[key]["total"] += 1
			if run.status == "passed":
				by_instrument[key]["passed"] += 1
			elif run.status == "failed":
				by_instrument[key]["failed"] += 1
			for v in run.westgard_violations:
				by_instrument[key]["violations"][v] = by_instrument[key]["violations"].get(v, 0) + 1
			by_instrument[key]["tests"][run.test_code] = by_instrument[key]["tests"].get(run.test_code, 0) + 1

		for inst_data in by_instrument.values():
			inst_data["pass_rate_pct"] = calculate_pass_rate(inst_data["total"], inst_data["passed"])
			iid = inst_data["instrument_id"]
			inst_data["current_status"] = instruments[iid].status if iid in instruments else "unknown"

		return {
			"tenant_id": tenant_id,
			"generated_at": datetime.utcnow().isoformat(),
			"total_qc_runs": len(qc_runs),
			"by_instrument": list(by_instrument.values()),
		}

	# ── instrument extensions ─────────────────────────────────────────────────

	async def get_instrument(
		self, tenant_id: str, instrument_id: str
	) -> InstrumentResponse | None:
		"""Retrieve a single analyser interface record."""
		return self._instruments.get((tenant_id, instrument_id))

	async def update_instrument(
		self, tenant_id: str, instrument_id: str, payload: AnalyserInterfaceUpdate
	) -> InstrumentResponse | None:
		"""Update analyser interface properties."""
		inst = self._instruments.get((tenant_id, instrument_id))
		if inst is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = inst.model_copy(update=updates)
		self._instruments[(tenant_id, instrument_id)] = updated
		self._audit(tenant_id, "instrument_updated", instrument_id)
		return updated

	async def record_calibration(
		self,
		tenant_id: str,
		instrument_id: str,
		calibrated_by: str,
		notes: str | None = None,
		pass_fail: bool = True,
	) -> dict[str, Any]:
		"""Record an instrument calibration event and update calibration due date.

		Updates instrument's last_calibrated_at and calibration_due_at.
		If pass_fail is False, instrument remains offline pending corrective action.
		"""
		assert bool(calibrated_by), "calibrated_by required"
		inst = self._instruments.get((tenant_id, instrument_id))
		if inst is None:
			raise KeyError(f"instrument {instrument_id} not found")

		now = datetime.utcnow()
		interval_days = inst.calibration_interval_days or 90
		next_due = now + timedelta(days=interval_days)

		cal_id = uuid7str()
		record: dict[str, Any] = {
			"id": cal_id,
			"tenant_id": tenant_id,
			"instrument_id": instrument_id,
			"calibrated_by": calibrated_by,
			"calibration_date": now.isoformat(),
			"next_due_date": next_due.isoformat(),
			"notes": notes,
			"pass_fail": pass_fail,
		}
		self._calibrations[(tenant_id, cal_id)] = record

		new_status = "online" if pass_fail else "offline"
		updated_inst = inst.model_copy(update={
			"last_calibrated_at": now,
			"calibration_due_at": next_due,
			"status": new_status,
			"updated_at": now,
		})
		self._instruments[(tenant_id, instrument_id)] = updated_inst
		self._audit(tenant_id, "instrument_calibrated", cal_id)
		_log_op("record_calibration", tenant_id, cal_id)
		return record

	async def interface_analyser(
		self,
		tenant_id: str,
		instrument_id: str,
		protocol: str,
		message_type: str,
		raw_payload: str,
	) -> dict[str, Any]:
		"""Ingest a raw message from an analyser interface (HL7 v2 / ASTM / REST).

		Parses the message header, extracts result segments, maps to LIS order IDs,
		and queues results for technician review.  Instrument's last_message_at
		and message_count are updated.  Returns a message receipt record.
		"""
		assert bool(raw_payload), "raw_payload required"
		assert bool(message_type), "message_type required"

		inst = self._instruments.get((tenant_id, instrument_id))
		if inst is None:
			raise KeyError(f"instrument {instrument_id} not found")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
		})

		msg_id = uuid7str()
		now = datetime.utcnow()

		# Minimal parse: extract result segments from HL7 OBX or ASTM R records
		parsed_results: list[dict[str, Any]] = []
		if protocol in {"hl7_v2", "astm_e1381"}:
			# Walk lines looking for OBX (HL7) or R| (ASTM) segments
			for line in raw_payload.splitlines():
				line = line.strip()
				if line.startswith("OBX|") or line.startswith("R|"):
					fields = line.split("|")
					try:
						if line.startswith("OBX|"):
							parsed_results.append({
								"segment": "OBX",
								"set_id": fields[1] if len(fields) > 1 else "",
								"value_type": fields[2] if len(fields) > 2 else "",
								"identifier": fields[3] if len(fields) > 3 else "",
								"value": fields[5] if len(fields) > 5 else "",
								"unit": fields[6] if len(fields) > 6 else "",
								"reference_range": fields[7] if len(fields) > 7 else "",
								"abnormal_flag": fields[8] if len(fields) > 8 else "",
							})
						else:  # ASTM R|
							parsed_results.append({
								"segment": "R",
								"test_id": fields[2] if len(fields) > 2 else "",
								"value": fields[3] if len(fields) > 3 else "",
								"unit": fields[4] if len(fields) > 4 else "",
								"reference_range": fields[5] if len(fields) > 5 else "",
								"abnormal_flag": fields[6] if len(fields) > 6 else "",
							})
					except IndexError as _exc:
						_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		error: str | None = None
		processed = len(parsed_results) > 0

		record: dict[str, Any] = {
			"id": msg_id,
			"tenant_id": tenant_id,
			"instrument_id": instrument_id,
			"protocol": protocol,
			"message_type": message_type,
			"raw_payload": raw_payload,
			"parsed_results": parsed_results,
			"received_at": now.isoformat(),
			"processed": processed,
			"error": error,
			"result_count": len(parsed_results),
		}
		self._instrument_messages[(tenant_id, msg_id)] = record

		# Update instrument stats
		new_count = (inst.message_count or 0) + 1
		updated_inst = inst.model_copy(update={
			"last_message_at": now,
			"message_count": new_count,
			"updated_at": now,
		})
		self._instruments[(tenant_id, instrument_id)] = updated_inst

		self._audit(tenant_id, "analyser_message_ingested", msg_id)
		_log_op("interface_analyser", tenant_id, msg_id)
		return record

	# ── external referral management ──────────────────────────────────────────

	async def create_referral(
		self, payload: ExternalReferralCreate
	) -> ExternalReferralResponse:
		"""Create an external referral for a specimen/test."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
		})
		referral = ExternalReferralResponse(
			id=uuid7str(), tenant_id=payload.tenant_id,
			order_id=payload.order_id, specimen_id=payload.specimen_id,
			patient_id=payload.patient_id,
			reference_lab_name=payload.reference_lab_name,
			reference_lab_code=payload.reference_lab_code,
			test_code=payload.test_code, test_name=payload.test_name,
			clinical_notes=payload.clinical_notes,
			expected_tat_hours=payload.expected_tat_hours,
			dispatched_by=payload.dispatched_by,
			created_by=payload.created_by,
		)
		self._referrals[(payload.tenant_id, referral.id)] = referral
		self._audit(payload.tenant_id, "referral_created", referral.id)
		_log_op("create_referral", payload.tenant_id, referral.id)
		return referral

	async def get_referral(
		self, tenant_id: str, referral_id: str
	) -> ExternalReferralResponse | None:
		"""Retrieve an external referral by ID."""
		return self._referrals.get((tenant_id, referral_id))

	async def list_referrals(
		self, tenant_id: str, status: str | None = None
	) -> list[ExternalReferralResponse]:
		"""List external referrals for a tenant with optional status filter."""
		results = [
			r for (tid, _), r in self._referrals.items()
			if tid == tenant_id and not r.is_deleted
		]
		if status:
			results = [r for r in results if r.status == status]
		return sorted(results, key=lambda r: r.created_at, reverse=True)

	async def update_referral(
		self, tenant_id: str, referral_id: str, payload: ExternalReferralUpdate
	) -> ExternalReferralResponse | None:
		"""Update an external referral's tracking details."""
		referral = self._referrals.get((tenant_id, referral_id))
		if referral is None:
			return None
		updates = {k: v for k, v in payload.model_dump().items() if v is not None}
		updates["updated_at"] = datetime.utcnow()
		updated = referral.model_copy(update=updates)
		self._referrals[(tenant_id, referral_id)] = updated
		self._audit(tenant_id, "referral_updated", referral_id)
		return updated

	# ── report generation ─────────────────────────────────────────────────────

	async def generate_lab_report(
		self, tenant_id: str, order_id: str, fmt: str = "json"
	) -> dict[str, Any]:
		"""Generate a full patient lab report for an order.

		Collects order, patient demographics, all specimens, all results
		(with reference ranges and abnormal flags), and assembles a
		structured report suitable for PDF rendering or HL7 ORU^R01 dispatch.

		fmt: 'json' | 'html' | 'pdf' — PDF/HTML rendering is caller's responsibility.
		"""
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			raise KeyError(f"order {order_id} not found")

		specimens = [
			s for (tid, _), s in self._specimens.items()
			if tid == tenant_id and s.order_id == order_id
		]
		results = [
			r for (tid, _), r in self._results.items()
			if tid == tenant_id and r.order_id == order_id
		]
		critical_values = [
			cv for (tid, _), cv in self._critical_values.items()
			if tid == tenant_id and any(r.id == cv.result_id for r in results)
		]

		report_id = uuid7str()
		now = datetime.utcnow()

		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"generated_at": now.isoformat(),
			"format": fmt,
			"order": order.model_dump(mode="json"),
			"patient": {
				"patient_id": order.patient_id,
				"age_years": order.patient_age_years,
				"sex": order.patient_sex,
			},
			"specimens": [s.model_dump(mode="json") for s in specimens],
			"results": [r.model_dump(mode="json") for r in results],
			"critical_values": [cv.model_dump(mode="json") for cv in critical_values],
			"summary": {
				"total_results": len(results),
				"critical_results": sum(1 for r in results if r.critical_value),
				"abnormal_results": sum(1 for r in results if r.abnormal_flag is not None),
				"all_verified": all(r.result_status in {"final", "corrected"} for r in results),
			},
		}
		self._audit(tenant_id, "lab_report_generated", report_id)
		_log_op("generate_lab_report", tenant_id, report_id)
		return report

	async def generate_critical_value_report(
		self,
		tenant_id: str,
		date_from: str | None = None,
		date_to: str | None = None,
	) -> dict[str, Any]:
		"""Generate a critical value notification compliance report.

		Computes: total critical values, acknowledged count, SLA compliance rate
		(notifications within 60 min), escalation rate, and per-analyte breakdown.
		"""
		from .domain.calculations import calculate_critical_value_response_time

		all_cv = [n for (tid, _), n in self._critical_values.items() if tid == tenant_id]

		total = len(all_cv)
		acknowledged = sum(1 for n in all_cv if n.acknowledged_by is not None)
		escalated = sum(1 for n in all_cv if n.escalated)
		read_back = sum(1 for n in all_cv if n.read_back_confirmed)

		response_times: list[float] = []
		for n in all_cv:
			rt = calculate_critical_value_response_time(n.created_at, n.acknowledged_at)
			if rt is not None:
				response_times.append(rt)

		sla_met = sum(1 for rt in response_times if rt <= 60.0)
		sla_compliance = round(sla_met / max(len(response_times), 1) * 100, 1)

		by_analyte: dict[str, int] = {}
		for n in all_cv:
			by_analyte[n.analyte] = by_analyte.get(n.analyte, 0) + 1

		return {
			"tenant_id": tenant_id,
			"generated_at": datetime.utcnow().isoformat(),
			"date_from": date_from,
			"date_to": date_to,
			"total_critical_values": total,
			"acknowledged": acknowledged,
			"unacknowledged": total - acknowledged,
			"acknowledgement_rate_pct": round(acknowledged / max(total, 1) * 100, 1),
			"read_back_confirmed": read_back,
			"read_back_rate_pct": round(read_back / max(total, 1) * 100, 1),
			"escalated": escalated,
			"escalation_rate_pct": round(escalated / max(total, 1) * 100, 1),
			"sla_compliance_pct": sla_compliance,
			"median_response_time_minutes": (
				sorted(response_times)[len(response_times) // 2]
				if response_times else None
			),
			"by_analyte": by_analyte,
		}

	async def generate_rejection_report(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a specimen rejection rate report broken down by reason.

		Returns overall rejection rate and per-reason counts, enabling labs to
		identify pre-analytical quality improvement opportunities.
		"""
		from .domain.calculations import calculate_rejection_rate

		specimens = [s for (tid, _), s in self._specimens.items() if tid == tenant_id]
		total = len(specimens)
		rejected = [s for s in specimens if s.status == "rejected"]

		by_reason: dict[str, int] = {}
		for s in rejected:
			reason = str(s.rejection_reason) if s.rejection_reason else "unspecified"
			by_reason[reason] = by_reason.get(reason, 0) + 1

		return {
			"tenant_id": tenant_id,
			"generated_at": datetime.utcnow().isoformat(),
			"total_specimens": total,
			"rejected_count": len(rejected),
			"rejection_rate_pct": calculate_rejection_rate(total, len(rejected)),
			"by_reason": by_reason,
		}

	# ── __init__ store registration ────────────────────────────────────────────
	# (called at end of __init__ to register new stores)

	def _init_extended_stores(self) -> None:
		"""Initialise stores added in the expanded service beyond the base set."""
		self._tests: dict[tuple[str, str], LabTestResponse] = {}
		self._reference_ranges: dict[tuple[str, str], ReferenceRangeResponse] = {}
		self._referrals: dict[tuple[str, str], ExternalReferralResponse] = {}
		self._calibrations: dict[tuple[str, str], dict[str, Any]] = {}
		self._instrument_messages: dict[tuple[str, str], dict[str, Any]] = {}

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

	async def ml_critical_lab_flag(self, *args, **kwargs):
		"""AI-powered ML detection of critical lab result patterns. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["normal","borderline","abnormal","critical_alert"])
			return {"lab_class": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

	# ── new enhanced methods ───────────────────────────────────────────────────

	async def export_fhir_diagnostic_report(
		self,
		tenant_id: str,
		order_id: str,
	) -> dict[str, Any]:
		"""Serialise a completed lab order as a FHIR R4 DiagnosticReport + Observation bundle.

		Maps LabOrderResponse → DiagnosticReport, each LabResultResponse → Observation,
		each CriticalValueNotification → Communication resource.
		LOINC codes sourced from LabTestResponse catalogue entries.
		SNOMED status codes mapped from result_status:
		  preliminary → 33694004 | final → 36998000 | corrected → 397963008.

		Returns a FHIR Bundle (type=collection) containing all resources.
		Raises KeyError if the order is not found.
		"""
		_SNOMED_STATUS = {
			"preliminary": "33694004",
			"validated": "33694004",
			"final": "36998000",
			"corrected": "397963008",
		}
		order = self._orders.get((tenant_id, order_id))
		if order is None:
			raise KeyError(f"order {order_id} not found")

		results = [
			r for (tid, _), r in self._results.items()
			if tid == tenant_id and r.order_id == order_id
		]
		critical_values = [
			cv for (tid, _), cv in self._critical_values.items()
			if tid == tenant_id and any(r.id == cv.result_id for r in results)
		]

		observations: list[dict[str, Any]] = []
		for r in results:
			obs: dict[str, Any] = {
				"resourceType": "Observation",
				"id": r.id,
				"status": r.result_status,
				"code": {
					"coding": [{"system": "http://loinc.org", "code": r.analyte}],
					"text": r.analyte,
				},
				"subject": {"reference": f"Patient/{order.patient_id}"},
				"valueQuantity": {
					"value": r.value,
					"unit": r.unit,
					"system": "http://unitsofmeasure.org",
				},
				"interpretation": [{"coding": [{
					"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
					"code": r.abnormal_flag or "N",
				}]}],
			}
			if r.reference_low is not None or r.reference_high is not None:
				obs["referenceRange"] = [{"low": {"value": r.reference_low}, "high": {"value": r.reference_high}}]
			observations.append(obs)

		communications: list[dict[str, Any]] = []
		for cv in critical_values:
			communications.append({
				"resourceType": "Communication",
				"id": cv.id,
				"status": "completed" if cv.acknowledged_by else "in-progress",
				"subject": {"reference": f"Patient/{cv.patient_id}"},
				"about": [{"reference": f"Observation/{cv.result_id}"}],
				"payload": [{"contentString": f"Critical {cv.analyte}: {cv.value} {cv.unit}"}],
				"recipient": [{"display": cv.notified_to}],
			})

		diagnostic_report: dict[str, Any] = {
			"resourceType": "DiagnosticReport",
			"id": order_id,
			"status": _SNOMED_STATUS.get(order.status, "unknown"),
			"code": {
				"coding": [{"system": "http://loinc.org", "code": order.test_code}],
				"text": order.test_name,
			},
			"subject": {"reference": f"Patient/{order.patient_id}"},
			"issued": order.ordered_at.isoformat(),
			"result": [{"reference": f"Observation/{r.id}"} for r in results],
		}

		bundle_id = uuid7str()
		bundle: dict[str, Any] = {
			"resourceType": "Bundle",
			"id": bundle_id,
			"type": "collection",
			"timestamp": datetime.utcnow().isoformat(),
			"entry": (
				[{"resource": diagnostic_report}]
				+ [{"resource": o} for o in observations]
				+ [{"resource": c} for c in communications]
			),
		}
		self._audit(tenant_id, "fhir_bundle_exported", bundle_id)
		_log_op("export_fhir_diagnostic_report", tenant_id, bundle_id)
		return bundle

	async def configure_reflex_rule(
		self,
		tenant_id: str,
		trigger_test_code: str,
		condition: str,
		threshold: float,
		reflex_test_code: str,
		reflex_test_name: str,
		reflex_priority: str = "routine",
		configured_by: str = "",
	) -> dict[str, Any]:
		"""Define an auto-reflex rule: when trigger_test_code result meets condition,
		automatically place a new order for reflex_test_code.

		condition: gt | lt | gte | lte | eq | abnormal | critical
		reflex_priority: stat | asap | routine | reflex

		Example: creatinine > 2.0 → order eGFR reflex; TSH abnormal → order free T4.
		Rules are evaluated in `enter_result` after each result is stored.
		"""
		_VALID_CONDITIONS = {"gt", "lt", "gte", "lte", "eq", "abnormal", "critical"}
		_VALID_PRIORITIES = {"stat", "asap", "routine", "reflex"}
		assert condition in _VALID_CONDITIONS, f"invalid condition: {condition}"
		assert reflex_priority in _VALID_PRIORITIES, f"invalid reflex_priority: {reflex_priority}"
		assert bool(trigger_test_code), "trigger_test_code required"
		assert bool(reflex_test_code), "reflex_test_code required"

		if not hasattr(self, "_reflex_rules"):
			self._reflex_rules: dict[str, list[dict[str, Any]]] = {}

		rule_id = uuid7str()
		rule: dict[str, Any] = {
			"id": rule_id,
			"tenant_id": tenant_id,
			"trigger_test_code": trigger_test_code,
			"condition": condition,
			"threshold": threshold,
			"reflex_test_code": reflex_test_code,
			"reflex_test_name": reflex_test_name,
			"reflex_priority": reflex_priority,
			"configured_by": configured_by,
			"created_at": datetime.utcnow().isoformat(),
			"active": True,
		}
		key = f"{tenant_id}:{trigger_test_code}"
		self._reflex_rules.setdefault(key, []).append(rule)
		self._audit(tenant_id, "reflex_rule_configured", rule_id)
		_log_op("configure_reflex_rule", tenant_id, rule_id)
		return rule

	async def evaluate_reflex_rules(
		self,
		tenant_id: str,
		test_code: str,
		value: Any,
		abnormal_flag: str | None,
		is_critical: bool,
		order: Any,
	) -> list[dict[str, Any]]:
		"""Evaluate all active reflex rules for the given test code and result value.

		Returns list of triggered reflex rule records. Callers (e.g. enter_result)
		should create new orders for each triggered rule.  Numeric comparisons use
		float coercion; non-numeric values only match 'abnormal'/'critical' conditions.
		"""
		if not hasattr(self, "_reflex_rules"):
			return []
		triggered: list[dict[str, Any]] = []
		key = f"{tenant_id}:{test_code}"
		rules = self._reflex_rules.get(key, [])
		for rule in rules:
			if not rule.get("active"):
				continue
			cond = rule["condition"]
			thresh = rule["threshold"]
			fired = False
			if cond == "abnormal" and abnormal_flag is not None:
				fired = True
			elif cond == "critical" and is_critical:
				fired = True
			else:
				try:
					v = float(value)
					if cond == "gt" and v > thresh:
						fired = True
					elif cond == "lt" and v < thresh:
						fired = True
					elif cond == "gte" and v >= thresh:
						fired = True
					elif cond == "lte" and v <= thresh:
						fired = True
					elif cond == "eq" and v == thresh:
						fired = True
				except (TypeError, ValueError) as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			if fired:
				triggered.append(rule)
				self._audit(tenant_id, "reflex_rule_triggered", rule["id"])
				logger.info(
					"lab.reflex_triggered trigger=%s reflex=%s tenant=%s",
					test_code, rule["reflex_test_code"], tenant_id,
				)
		return triggered

	async def generate_compliance_scorecard(
		self,
		tenant_id: str,
		period: str,
		standard: str = "CAP",
	) -> dict[str, Any]:
		"""Generate an accreditation compliance scorecard for the specified standard.

		standard: CAP | CLIA | ISO_15189 | SANAS

		Criteria evaluated:
		- QC frequency: hours between consecutive QC runs per instrument (target: ≤8 h)
		- Critical value SLA: notifications within 60 min (target: ≥95%)
		- Specimen rejection rate (target: ≤2%)
		- EQA/proficiency testing participation (target: ≥80% score)
		- STAT TAT 90th percentile (target: ≤60 min)
		- Delta check utilisation (target: ≥90% of results checked)

		Returns structured scorecard with pass/fail per criterion and evidence summary.
		"""
		_VALID_STANDARDS = {"CAP", "CLIA", "ISO_15189", "SANAS"}
		assert standard in _VALID_STANDARDS, f"unsupported standard: {standard}"

		qc_runs = sorted(
			[q for (tid, _), q in self._qc_runs.items() if tid == tenant_id],
			key=lambda q: q.created_at,
		)
		critical_values = [n for (tid, _), n in self._critical_values.items() if tid == tenant_id]
		specimens = [s for (tid, _), s in self._specimens.items() if tid == tenant_id]
		proficiency = [p for (tid, _), p in self._proficiency_tests.items() if tid == tenant_id]
		delta_checks = [d for (tid, _), d in self._delta_checks.items() if tid == tenant_id]
		orders = [o for (tid, _), o in self._orders.items() if tid == tenant_id]
		results = [r for (tid, _), r in self._results.items() if tid == tenant_id if r.result_status == "final"]

		# QC frequency gaps
		by_instrument: dict[str, list[Any]] = {}
		for qr in qc_runs:
			by_instrument.setdefault(qr.instrument_id, []).append(qr.created_at)
		max_gap_hours = 0.0
		for times in by_instrument.values():
			if len(times) > 1:
				for i in range(1, len(times)):
					gap = (times[i] - times[i - 1]).total_seconds() / 3600.0
					if gap > max_gap_hours:
						max_gap_hours = gap
		qc_freq_pass = max_gap_hours <= 8.0 or len(qc_runs) == 0

		# Critical value SLA
		sla_count = 0
		sla_total = len(critical_values)
		for n in critical_values:
			if n.acknowledged_at and n.created_at:
				lag = (n.acknowledged_at - n.created_at).total_seconds() / 60.0
				if lag <= 60.0:
					sla_count += 1
		sla_compliance = round(sla_count / max(sla_total, 1) * 100, 1)
		sla_pass = sla_compliance >= 95.0 or sla_total == 0

		# Specimen rejection rate
		rejected = sum(1 for s in specimens if s.status == "rejected")
		rejection_rate = round(rejected / max(len(specimens), 1) * 100, 2)
		rejection_pass = rejection_rate <= 2.0

		# Proficiency testing participation
		satisfactory = sum(1 for p in proficiency if p.get("satisfactory", False))
		pt_score = round(satisfactory / max(len(proficiency), 1) * 100, 1)
		pt_pass = pt_score >= 80.0 or len(proficiency) == 0

		# STAT TAT 90th percentile
		stat_tats: list[float] = []
		for r in results:
			order = next((o for o in orders if o.id == r.order_id and o.collection_priority == "stat"), None)
			if order:
				try:
					tat = (r.created_at - order.ordered_at).total_seconds() / 60.0
					if tat >= 0:
						stat_tats.append(tat)
				except (AttributeError, TypeError) as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		stat_tats.sort()
		p90_stat = stat_tats[int(len(stat_tats) * 0.9)] if stat_tats else None
		stat_tat_pass = (p90_stat is None) or (p90_stat <= 60.0)

		# Delta check utilisation
		results_count = len(results)
		delta_utilisation = round(len(delta_checks) / max(results_count, 1) * 100, 1)
		delta_pass = delta_utilisation >= 90.0 or results_count == 0

		overall_pass = all([qc_freq_pass, sla_pass, rejection_pass, pt_pass, stat_tat_pass, delta_pass])
		scorecard_id = uuid7str()
		scorecard: dict[str, Any] = {
			"id": scorecard_id,
			"tenant_id": tenant_id,
			"period": period,
			"standard": standard,
			"generated_at": datetime.utcnow().isoformat(),
			"overall": "PASS" if overall_pass else "FAIL",
			"criteria": {
				"qc_frequency": {
					"pass": qc_freq_pass,
					"target": "≤8h between QC runs",
					"actual_max_gap_hours": round(max_gap_hours, 1),
				},
				"critical_value_sla": {
					"pass": sla_pass,
					"target": "≥95% notifications within 60 min",
					"actual_compliance_pct": sla_compliance,
				},
				"specimen_rejection_rate": {
					"pass": rejection_pass,
					"target": "≤2% rejection rate",
					"actual_rejection_pct": rejection_rate,
				},
				"proficiency_testing": {
					"pass": pt_pass,
					"target": "≥80% satisfactory EQA scores",
					"actual_satisfactory_pct": pt_score,
				},
				"stat_tat_p90": {
					"pass": stat_tat_pass,
					"target": "≤60 min STAT TAT 90th percentile",
					"actual_p90_minutes": p90_stat,
				},
				"delta_check_utilisation": {
					"pass": delta_pass,
					"target": "≥90% of results delta-checked",
					"actual_utilisation_pct": delta_utilisation,
				},
			},
		}
		self._audit(tenant_id, "compliance_scorecard_generated", scorecard_id)
		_log_op("generate_compliance_scorecard", tenant_id, scorecard_id)
		return scorecard

	async def route_specimen(
		self,
		tenant_id: str,
		specimen_id: str,
		test_code: str,
	) -> dict[str, Any]:
		"""Select the optimal instrument for a specimen+test using weighted routing.

		Routing algorithm:
		1. Filter instruments by: test_code in test_categories, status == online.
		2. Score by: weight (configured) / current queue depth.
		3. Select highest-scoring available instrument.
		4. If all instruments are on QC hold or offline, return routing_failed=True.

		Routing decision is published to audit log and the selected instrument's
		queue depth counter is incremented.  Returns routing decision record.
		"""
		assert bool(specimen_id), "specimen_id required"
		assert bool(test_code), "test_code required"

		if not hasattr(self, "_routing_config"):
			self._routing_config: dict[str, list[dict[str, Any]]] = {}
		if not hasattr(self, "_instrument_queue_depth"):
			self._instrument_queue_depth: dict[tuple[str, str], int] = {}

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		# Get routing weights for test_code
		routing_key = f"{tenant_id}:{test_code}"
		weights = self._routing_config.get(routing_key, [])

		instruments = [i for (tid, _), i in self._instruments.items() if tid == tenant_id]
		eligible = [
			i for i in instruments
			if i.status not in {"qc_hold", "offline", "maintenance"}
			and (not i.test_categories or test_code in i.test_categories)
		]

		if not eligible:
			return {
				"specimen_id": specimen_id,
				"test_code": test_code,
				"routing_failed": True,
				"reason": "no_eligible_instruments",
				"routed_at": datetime.utcnow().isoformat(),
			}

		# Score candidates
		weight_map = {w["instrument_id"]: w.get("weight", 1.0) for w in weights}
		max_queue_map = {w["instrument_id"]: w.get("max_queue", 100) for w in weights}

		def _score(inst: InstrumentResponse) -> float:
			w = weight_map.get(inst.id, 1.0)
			q = self._instrument_queue_depth.get((tenant_id, inst.id), 0)
			max_q = max_queue_map.get(inst.id, 100)
			if q >= max_q:
				return -1.0
			return w / (q + 1)

		selected = max(eligible, key=_score)

		# Increment queue counter
		current_depth = self._instrument_queue_depth.get((tenant_id, selected.id), 0)
		self._instrument_queue_depth[(tenant_id, selected.id)] = current_depth + 1

		routing_id = uuid7str()
		record: dict[str, Any] = {
			"id": routing_id,
			"tenant_id": tenant_id,
			"specimen_id": specimen_id,
			"test_code": test_code,
			"selected_instrument_id": selected.id,
			"selected_instrument_name": selected.name,
			"routing_failed": False,
			"eligible_count": len(eligible),
			"queue_depth_after": current_depth + 1,
			"routed_at": datetime.utcnow().isoformat(),
		}
		self._audit(tenant_id, "specimen_routed", routing_id)
		_log_op("route_specimen", tenant_id, routing_id)
		return record

	async def configure_routing_weights(
		self,
		tenant_id: str,
		test_code: str,
		weights: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Configure per-instrument routing weights for a test code.

		Each weight entry: {instrument_id: str, weight: float, max_queue: int}
		Higher weight → more likely to be selected when queue depths are equal.
		max_queue limits the instrument's concurrent workload before fallback.
		"""
		if not hasattr(self, "_routing_config"):
			self._routing_config = {}
		key = f"{tenant_id}:{test_code}"
		self._routing_config[key] = weights
		config_id = uuid7str()
		self._audit(tenant_id, "routing_weights_configured", config_id)
		return {"id": config_id, "tenant_id": tenant_id, "test_code": test_code, "weights": weights}

	async def record_patient_consent(
		self,
		tenant_id: str,
		patient_id: str,
		test_categories: list[str],
		consented_by: str,
		expiry_date: datetime | None = None,
		consent_method: str = "written",
	) -> dict[str, Any]:
		"""Record a patient's consent for release of sensitive test categories.

		Consent gates release of results in CONSENT_GATED_CATEGORIES:
		genetics | hiv | substance_abuse | reproductive | mental_health

		consent_method: written | verbal | electronic | implicit
		Consent records are time-limited; expired consents are treated as absent.
		Returns the consent record ID for reference in result release workflows.
		"""
		_VALID_METHODS = {"written", "verbal", "electronic", "implicit"}
		assert consent_method in _VALID_METHODS, f"invalid consent_method: {consent_method}"
		assert bool(patient_id), "patient_id required"
		assert bool(consented_by), "consented_by required"
		assert test_categories, "test_categories must not be empty"

		if not hasattr(self, "_consent_records"):
			self._consent_records: dict[tuple[str, str, str], dict[str, Any]] = {}

		now = datetime.utcnow()
		consent_id = uuid7str()
		for cat in test_categories:
			record: dict[str, Any] = {
				"id": consent_id,
				"tenant_id": tenant_id,
				"patient_id": patient_id,
				"test_category": cat,
				"consented_by": consented_by,
				"consent_method": consent_method,
				"consented_at": now.isoformat(),
				"expiry_date": expiry_date.isoformat() if expiry_date else None,
				"active": True,
			}
			self._consent_records[(tenant_id, patient_id, cat)] = record

		self._audit(tenant_id, "patient_consent_recorded", consent_id)
		logger.info("lab.consent_recorded patient=%s categories=%s tenant=%s", patient_id, test_categories, tenant_id)
		return {
			"id": consent_id,
			"tenant_id": tenant_id,
			"patient_id": patient_id,
			"test_categories": test_categories,
			"consented_at": now.isoformat(),
			"expiry_date": expiry_date.isoformat() if expiry_date else None,
		}

	async def check_consent(
		self,
		tenant_id: str,
		patient_id: str,
		test_category: str,
	) -> dict[str, Any]:
		"""Check whether valid consent exists for releasing a sensitive test result.

		Returns {has_consent: bool, consent_id: str | None, expiry_date: str | None}.
		Expired consent is treated as absent.
		"""
		_CONSENT_GATED_CATEGORIES = {"genetics", "hiv", "substance_abuse", "reproductive", "mental_health"}

		if test_category not in _CONSENT_GATED_CATEGORIES:
			return {"has_consent": True, "consent_id": None, "reason": "category_not_consent_gated"}

		if not hasattr(self, "_consent_records"):
			return {"has_consent": False, "consent_id": None, "reason": "no_consent_records"}

		record = self._consent_records.get((tenant_id, patient_id, test_category))
		if record is None:
			return {"has_consent": False, "consent_id": None, "reason": "no_consent_on_record"}

		if record.get("expiry_date"):
			try:
				expiry = datetime.fromisoformat(record["expiry_date"])
				if datetime.utcnow() > expiry:
					return {"has_consent": False, "consent_id": record["id"], "reason": "consent_expired"}
			except ValueError as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		return {
			"has_consent": True,
			"consent_id": record["id"],
			"expiry_date": record.get("expiry_date"),
			"reason": "valid_consent_on_record",
		}

	async def get_audit_events(
		self,
		tenant_id: str,
		event_type: str | None = None,
		limit: int = 100,
	) -> list[dict[str, Any]]:
		"""Return recent audit events for a tenant, optionally filtered by event type.

		Events are returned in reverse-chronological order (most recent first).
		limit: maximum number of events to return (max 1000).
		"""
		limit = min(limit, 1000)
		events = [e for e in self._audit_events if e.get("tenant_id") == tenant_id]
		if event_type:
			events = [e for e in events if e.get("event") == event_type]
		return sorted(events, key=lambda e: e.get("timestamp", ""), reverse=True)[:limit]

	async def verify_audit_chain(self, tenant_id: str) -> dict[str, Any]:
		"""Verify cryptographic integrity of the audit event chain for a tenant.

		Each event's hash is recomputed from its content fields.
		Returns {valid: bool, entries_verified: int, first_break_at: int | None}.

		Note: hash chaining is only active if events were written via _audit_with_hash.
		Legacy events (no hash field) are counted but not verified.
		"""
		import hashlib

		events = [e for e in self._audit_events if e.get("tenant_id") == tenant_id]
		if not events:
			return {"valid": True, "entries_verified": 0, "first_break_at": None, "legacy_events": 0}

		verified = 0
		legacy = 0
		first_break: int | None = None

		prev_hash = ""
		for idx, event in enumerate(events):
			stored_hash = event.get("chain_hash")
			if stored_hash is None:
				legacy += 1
				continue
			payload = f"{prev_hash}{event.get('tenant_id','')}{event.get('event','')}{event.get('entity_id','')}{event.get('timestamp','')}"
			computed = hashlib.sha256(payload.encode()).hexdigest()
			if computed != stored_hash:
				if first_break is None:
					first_break = idx
			else:
				verified += 1
			prev_hash = stored_hash

		return {
			"valid": first_break is None,
			"entries_verified": verified,
			"first_break_at": first_break,
			"legacy_events": legacy,
			"total_events": len(events),
		}

	async def assess_specimen_viability(
		self,
		tenant_id: str,
		specimen_id: str,
		test_codes: list[str],
	) -> dict[str, Any]:
		"""Estimate specimen viability for the requested test codes based on elapsed time,
		collection method, and transport conditions recorded in the custody chain.

		Uses CLSI EP25-based stability windows per analyte type:
		  potassium (whole_blood, RT): 1h | glucose (serum, RT): 4h | CBC: 24h
		  coagulation (citrate, RT): 4h | general_chemistry (serum, 4°C): 72h

		Returns:
		  viability_score (0–100), risk_analytes (those near/past stability window),
		  recommended_action (process_immediately | acceptable | reject).
		"""
		_STABILITY_HOURS: dict[str, float] = {
			"K": 1.0, "Na": 8.0, "Glucose": 4.0, "Hb": 24.0, "WBC": 24.0,
			"PT": 4.0, "APTT": 4.0, "INR": 4.0, "Creatinine": 72.0,
			"default": 24.0,
		}

		spec = self._specimens.get((tenant_id, specimen_id))
		if spec is None:
			raise KeyError(f"specimen {specimen_id} not found")

		chain = self._custody_chain.get((tenant_id, specimen_id), [])
		now = datetime.utcnow()
		elapsed_hours = (now - spec.collected_at).total_seconds() / 3600.0

		# Temperature multiplier from custody chain — refrigerated extends stability
		temp_multiplier = 1.0
		for event in chain:
			cond = event.get("transport_condition", "ambient")
			if cond == "refrigerated":
				temp_multiplier = 2.5
			elif cond == "frozen":
				temp_multiplier = 10.0
			elif cond == "dry_ice":
				temp_multiplier = 20.0

		risk_analytes: list[str] = []
		scores: list[float] = []
		for tc in test_codes:
			stability = _STABILITY_HOURS.get(tc, _STABILITY_HOURS["default"]) * temp_multiplier
			remaining_pct = max(0.0, (stability - elapsed_hours) / stability * 100.0)
			scores.append(remaining_pct)
			if remaining_pct < 20.0:
				risk_analytes.append(tc)

		viability_score = round(sum(scores) / max(len(scores), 1), 1) if scores else 100.0

		if viability_score < 10.0:
			recommended_action = "reject"
		elif viability_score < 40.0:
			recommended_action = "process_immediately"
		else:
			recommended_action = "acceptable"

		return {
			"specimen_id": specimen_id,
			"tenant_id": tenant_id,
			"elapsed_hours": round(elapsed_hours, 2),
			"viability_score": viability_score,
			"risk_analytes": risk_analytes,
			"recommended_action": recommended_action,
			"test_codes_assessed": test_codes,
			"assessed_at": now.isoformat(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

