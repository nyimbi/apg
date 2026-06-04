"""Comprehensive CI tests for the APG Laboratory Information System service.

Tests cover all service methods including new additions:
- LabTest catalogue CRUD
- Order lifecycle (create, update, hold, unhold, cancel, receive)
- Specimen tracking with custody chain
- Reference range management and validation
- Result entry, verification, release, amendment
- Critical value alerting and acknowledgement
- QC runs, Westgard rules, failure actions
- Instrument management, calibration, HL7 message ingestion
- External referral CRUD and result receipt
- Report generation (lab report, QC summary, critical value, rejection)
- TAT monitoring and workload reports
- Delta check algorithm
- Domain rules (domain/rules.py)
- Domain calculations (domain/calculations.py)

© 2025 Datacraft — nyimbi@gmail.com
"""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

# Ensure lab package is importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lab.models import (
	AnalyserInterfaceCreate,
	AnalyserInterfaceUpdate,
	CriticalValueCreate,
	ExternalReferralCreate,
	ExternalReferralUpdate,
	LabOrderCreate,
	LabOrderUpdate,
	LabResultCreate,
	LabTestCreate,
	LabTestUpdate,
	QCRunCreate,
	ReferenceRangeCreate,
	ReferenceRangeUpdate,
	SpecimenCreate,
	SpecimenTrackRequest,
	SpecimenUpdate,
)
from lab.service import LaboratoryInformationService, PolicyViolationError
from lab.domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_no_cross_tenant_access,
	assert_order_cancellable,
	assert_specimen_collectable,
	assert_specimen_type_supported,
	assert_rejection_reason_present,
	assert_critical_value_notification_sent,
	assert_result_validated_for_release,
	assert_reference_range_bounds_valid,
	assert_instrument_not_on_qc_hold,
	assert_qc_lot_not_expired,
	assert_calibration_not_overdue,
	calculate_z_score,
	calculate_cv_percent,
	calculate_delta_percent,
	calculate_rejection_rate_pct,
	calculate_qc_pass_rate_pct,
	calculate_tat_deadline,
	calculate_percentile,
)
from lab.domain.calculations import (
	classify_numeric_result,
	delta_check,
	evaluate_westgard_rules,
	calculate_tat_metrics,
	select_reference_range,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def fresh() -> LaboratoryInformationService:
	return LaboratoryInformationService()


TID = "tenant_ci"


def make_order(svc: LaboratoryInformationService, priority: str = "routine") -> any:
	return run(svc.create_order(LabOrderCreate(
		tenant_id=TID, patient_id="pat1", encounter_id="enc1",
		test_code="CBC", test_name="Complete Blood Count",
		test_category="hematology", collection_priority=priority,
		ordered_by="dr_smith", clinical_indication="Anaemia workup",
		specimen_type="blood_venous", created_by="dr_smith",
	)))


def make_specimen(svc, order_id: str) -> any:
	return run(svc.collect_specimen(SpecimenCreate(
		tenant_id=TID, order_id=order_id, patient_id="pat1",
		specimen_type="blood_venous", collected_by="nurse1",
		collection_site="left_antecubital", collection_volume_ml=5.0,
		created_by="nurse1",
	)))


def make_instrument(svc) -> any:
	return run(svc.register_instrument(AnalyserInterfaceCreate(
		tenant_id=TID, name="Sysmex XN-9100", model="XN-9100",
		serial_number="SN-CI-001", manufacturer="Sysmex",
		location="Haematology Lab", created_by="admin",
	)))


def make_result(svc, order_id: str, specimen_id: str, value: float = 95.0) -> any:
	return run(svc.enter_result(LabResultCreate(
		tenant_id=TID, order_id=order_id, specimen_id=specimen_id,
		analyte="Glucose", value=value, unit="mg/dL",
		reference_low=70.0, reference_high=110.0,
		result_status="preliminary", performed_by="tech1", created_by="tech1",
	)))


# ── LabTest catalogue ──────────────────────────────────────────────────────────

class TestLabTestCatalogue:

	def test_create_and_retrieve_test(self):
		svc = fresh()
		test = run(svc.create_test(LabTestCreate(
			tenant_id=TID, test_code="HbA1c", test_name="Glycated Haemoglobin",
			category="chemistry", specimen_types=["blood_venous"],
			loinc_code="4548-4", turnaround_minutes=240,
			stat_turnaround_minutes=120, created_by="admin",
		)))
		assert test.id
		assert test.test_code == "HbA1c"
		fetched = run(svc.get_test(TID, test.id))
		assert fetched.loinc_code == "4548-4"

	def test_list_tests_filter_by_category(self):
		svc = fresh()
		run(svc.create_test(LabTestCreate(
			tenant_id=TID, test_code="CBC", test_name="Complete Blood Count",
			category="hematology", specimen_types=["blood_venous"], created_by="admin",
		)))
		run(svc.create_test(LabTestCreate(
			tenant_id=TID, test_code="Glucose", test_name="Glucose",
			category="chemistry", specimen_types=["serum"], created_by="admin",
		)))
		hem = run(svc.list_tests(TID, category="hematology"))
		assert len(hem) == 1 and hem[0].test_code == "CBC"

	def test_update_test(self):
		svc = fresh()
		test = run(svc.create_test(LabTestCreate(
			tenant_id=TID, test_code="T1", test_name="Test One",
			category="chemistry", specimen_types=["serum"], created_by="admin",
		)))
		updated = run(svc.update_test(TID, test.id, LabTestUpdate(test_name="Test One Updated")))
		assert updated.test_name == "Test One Updated"

	def test_delete_test_soft(self):
		svc = fresh()
		test = run(svc.create_test(LabTestCreate(
			tenant_id=TID, test_code="DEL1", test_name="To Delete",
			category="chemistry", specimen_types=["serum"], created_by="admin",
		)))
		run(svc.delete_test(TID, test.id, "admin"))
		assert run(svc.list_tests(TID)) == []

	def test_get_nonexistent_test_returns_none(self):
		svc = fresh()
		assert run(svc.get_test(TID, "nonexistent")) is None


# ── Order lifecycle ────────────────────────────────────────────────────────────

class TestOrderLifecycle:

	def test_create_order_pending(self):
		svc = fresh()
		order = make_order(svc)
		assert order.status == "pending"
		assert order.tenant_id == TID

	def test_update_order_notes(self):
		svc = fresh()
		order = make_order(svc)
		updated = run(svc.update_order(TID, order.id, LabOrderUpdate(notes="fasting confirmed")))
		assert updated.notes == "fasting confirmed"

	def test_hold_and_unhold_order(self):
		svc = fresh()
		order = make_order(svc)
		held = run(svc.hold_order(TID, order.id, "awaiting consent"))
		assert held.status == "on_hold"
		assert held.on_hold_reason == "awaiting consent"
		released = run(svc.unhold_order(TID, order.id))
		assert released.status == "pending"
		assert released.on_hold_reason is None

	def test_cancel_order(self):
		svc = fresh()
		order = make_order(svc)
		cancelled = run(svc.cancel_order(TID, order.id, "patient_declined"))
		assert cancelled.status == "cancelled"
		assert cancelled.cancelled_reason == "patient_declined"

	def test_receive_lab_order(self):
		svc = fresh()
		order = make_order(svc)
		receipt = run(svc.receive_lab_order(
			TID, order.id,
			{"tube_type": "EDTA", "volume_ml": 3.0},
			"lab_receptionist",
		))
		assert receipt["order_id"] == order.id
		assert receipt["received_by"] == "lab_receptionist"
		updated = run(svc.get_order(TID, order.id))
		assert updated.status == "received"

	def test_list_orders_filter_by_priority(self):
		svc = fresh()
		make_order(svc, priority="routine")
		make_order(svc, priority="stat")
		stat = run(svc.list_orders(TID, priority="stat"))
		assert len(stat) == 1

	def test_cancel_nonexistent_order_returns_none(self):
		svc = fresh()
		assert run(svc.cancel_order(TID, "missing", "reason")) is None


# ── Specimen management ────────────────────────────────────────────────────────

class TestSpecimenManagement:

	def test_collect_specimen_links_order(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		assert spec.status == "collected"
		order_updated = run(svc.get_order(TID, order.id))
		assert order_updated.specimen_id == spec.id

	def test_update_specimen_storage_location(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		updated = run(svc.update_specimen(TID, spec.id, SpecimenUpdate(storage_location="Fridge-A2")))
		assert updated.storage_location == "Fridge-A2"

	def test_receive_specimen(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		received = run(svc.receive_specimen(TID, spec.id))
		assert received.status == "received"
		assert received.received_at is not None

	def test_reject_specimen_records_reason(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		rejected = run(svc.reject_specimen(TID, spec.id, "hemolyzed"))
		assert rejected.status == "rejected"
		assert rejected.rejection_reason == "hemolyzed"

	def test_reject_specimen_empty_reason_denied(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		with pytest.raises(PolicyViolationError):
			run(svc.reject_specimen(TID, spec.id, ""))

	def test_track_specimen_appends_custody(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = run(svc.track_specimen(TID, spec.id, SpecimenTrackRequest(
			event_type="transferred", actor_id="porter1", location="Biochemistry"
		)))
		assert result["custody_chain_length"] >= 2
		chain = run(svc.get_custody_chain(TID, spec.id))
		assert any(e.get("event_type") == "transferred" for e in chain)

	def test_get_custody_chain_empty_for_unknown(self):
		svc = fresh()
		chain = run(svc.get_custody_chain(TID, "unknown_specimen"))
		assert chain == []

	def test_track_unknown_specimen_raises(self):
		svc = fresh()
		with pytest.raises(KeyError):
			run(svc.track_specimen(TID, "ghost", SpecimenTrackRequest(
				event_type="transferred", actor_id="x"
			)))

	def test_chain_of_custody_transfer(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = run(svc.track_specimen_chain_of_custody(
			TID, spec.id, "Phlebotomy", "Haematology Lab", "porter2", "refrigerated"
		))
		assert result["custody_chain_length"] >= 2


# ── Reference range management ─────────────────────────────────────────────────

class TestReferenceRangeManagement:

	def _make_rr(self, svc, analyte="Glucose", low=70.0, high=110.0):
		return run(svc.create_reference_range(ReferenceRangeCreate(
			tenant_id=TID, test_code="GLUC", analyte=analyte,
			unit="mg/dL", low=low, high=high,
			critical_low=40.0, critical_high=400.0,
			created_by="admin",
		)))

	def test_create_and_retrieve(self):
		svc = fresh()
		rr = self._make_rr(svc)
		assert rr.id
		fetched = run(svc.get_reference_range(TID, rr.id))
		assert fetched.analyte == "Glucose"

	def test_list_by_test_code(self):
		svc = fresh()
		self._make_rr(svc, "Glucose")
		other = run(svc.create_reference_range(ReferenceRangeCreate(
			tenant_id=TID, test_code="CBC", analyte="Haemoglobin",
			unit="g/dL", low=12.0, high=17.0, created_by="admin",
		)))
		gluc_ranges = run(svc.list_reference_ranges(TID, test_code="GLUC"))
		assert all(r.test_code == "GLUC" for r in gluc_ranges)

	def test_update_bounds(self):
		svc = fresh()
		rr = self._make_rr(svc)
		updated = run(svc.update_reference_range(TID, rr.id, ReferenceRangeUpdate(high=120.0)))
		assert updated.high == 120.0

	def test_delete_soft(self):
		svc = fresh()
		rr = self._make_rr(svc)
		run(svc.delete_reference_range(TID, rr.id, "admin"))
		assert run(svc.list_reference_ranges(TID)) == []

	def test_validate_reference_range_normal(self):
		svc = fresh()
		self._make_rr(svc)
		result = run(svc.validate_reference_range(TID, "GLUC", "Glucose", 95.0))
		assert result["flag"] is None
		assert not result["is_critical"]

	def test_validate_reference_range_critical_high(self):
		svc = fresh()
		self._make_rr(svc)
		result = run(svc.validate_reference_range(TID, "GLUC", "Glucose", 450.0))
		assert result["is_critical"]

	def test_validate_no_matching_range(self):
		svc = fresh()
		result = run(svc.validate_reference_range(TID, "UNKNOWN", "Foo", 10.0))
		assert result["matched_range"] is None

	def test_model_rejects_inverted_bounds(self):
		with pytest.raises(ValidationError):
			ReferenceRangeCreate(
				tenant_id=TID, test_code="X", analyte="Y",
				unit="U", low=100.0, high=50.0, created_by="admin",
			)

	def test_model_rejects_critical_low_above_normal_low(self):
		with pytest.raises(ValidationError):
			ReferenceRangeCreate(
				tenant_id=TID, test_code="X", analyte="Y",
				unit="U", low=70.0, high=110.0,
				critical_low=80.0,  # must be < low
				created_by="admin",
			)


# ── Result entry and lifecycle ─────────────────────────────────────────────────

class TestResultLifecycle:

	def test_enter_normal_result(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id, value=95.0)
		assert result.result_status == "preliminary"
		assert result.abnormal_flag is None
		assert not result.critical_value

	def test_enter_high_result_flagged(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id, value=130.0)
		assert result.abnormal_flag == "H"

	def test_enter_critical_high_result(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id, value=200.0)
		assert result.critical_value
		assert result.abnormal_flag == "HH"

	def test_verify_result_ok(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id)
		verified = run(svc.verify_result(TID, result.id, "supervisor", notification_sent=True))
		assert verified.result_status == "final"
		assert verified.verified_by == "supervisor"

	def test_verify_critical_without_notification_denied(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id, value=200.0)
		assert result.critical_value
		with pytest.raises(PolicyViolationError):
			run(svc.verify_result(TID, result.id, "supervisor", notification_sent=False))

	def test_release_result_ok(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id)
		run(svc.verify_result(TID, result.id, "supervisor", notification_sent=True))
		released = run(svc.release_result(TID, result.id, "tech1", "portal"))
		assert released.result_status == "final"

	def test_release_unverified_result_denied(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id)
		with pytest.raises(PolicyViolationError):
			run(svc.release_result(TID, result.id, "tech1", "portal"))

	def test_amend_result_creates_corrected(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id)
		run(svc.verify_result(TID, result.id, "supervisor", notification_sent=True))
		amendment = run(svc.result_amend(
			TID, result.id, 105.0, "transcription error", "supervisor"
		))
		assert amendment["amended_value"] == 105.0
		assert amendment["original_result_id"] == result.id

	def test_update_result_blocked_after_verification(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id)
		run(svc.verify_result(TID, result.id, "supervisor", notification_sent=True))
		from lab.models import LabResultUpdate
		with pytest.raises(PolicyViolationError):
			run(svc.update_result(TID, result.id, LabResultUpdate(notes="updated")))

	def test_delta_check_exceeds_threshold(self):
		svc = fresh()
		# Seed a prior result
		svc._previous_results[(TID, "pat1", "K")] = 3.5
		delta = run(svc.delta_check(TID, "pat1", "K", 5.5, 15.0))
		assert delta["delta_exceeded"]
		assert delta["alert_required"]

	def test_delta_check_within_threshold(self):
		svc = fresh()
		svc._previous_results[(TID, "pat1", "K")] = 4.0
		delta = run(svc.delta_check(TID, "pat1", "K", 4.2, 15.0))
		assert not delta["delta_exceeded"]

	def test_delta_check_no_prior_result(self):
		svc = fresh()
		delta = run(svc.delta_check(TID, "pat1", "K", 4.0))
		assert delta["previous_result"] is None
		assert not delta["alert_required"]


# ── Critical value alerting ────────────────────────────────────────────────────

class TestCriticalValueAlerting:

	def test_create_critical_value_notification(self):
		svc = fresh()
		notif = run(svc.create_critical_value(CriticalValueCreate(
			tenant_id=TID, result_id="r1", patient_id="pat1",
			analyte="Potassium", value=7.8, unit="mEq/L",
			severity="critical_high", notified_to="dr_jones",
			notified_by="tech1", created_by="tech1",
		)))
		assert notif.id
		assert notif.acknowledged_by is None

	def test_acknowledge_critical_value(self):
		svc = fresh()
		notif = run(svc.notify_critical_value(
			TID, "r1", "pat1", "K", 7.8, "mEq/L",
			"critical_high", "dr_jones", "tech1"
		))
		acked = run(svc.acknowledge_critical_value(TID, notif.id, "dr_jones"))
		assert acked.acknowledged_by == "dr_jones"
		assert acked.acknowledged_at is not None

	def test_acknowledge_requires_acknowledger(self):
		svc = fresh()
		notif = run(svc.notify_critical_value(
			TID, "r1", "pat1", "K", 7.8, "mEq/L",
			"critical_high", "dr_jones", "tech1"
		))
		with pytest.raises(PolicyViolationError):
			run(svc.acknowledge_critical_value(TID, notif.id, ""))

	def test_list_critical_values_unacked_filter(self):
		svc = fresh()
		n1 = run(svc.notify_critical_value(TID, "r1", "pat1", "K", 7.8, "mEq/L", "critical_high", "dr1", "t1"))
		n2 = run(svc.notify_critical_value(TID, "r2", "pat1", "Na", 115.0, "mEq/L", "critical_low", "dr1", "t1"))
		run(svc.acknowledge_critical_value(TID, n1.id, "dr1"))
		unacked = run(svc.list_critical_values(TID, unacknowledged_only=True))
		assert len(unacked) == 1
		assert unacked[0].id == n2.id

	def test_alert_critical_value_high_level(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		result = make_result(svc, order.id, spec.id, value=200.0)
		alert = run(svc.alert_critical_value(
			TID, result.id, "Glucose", 200.0, "mg/dL",
			"critical_high", "dr_smith", "tech1",
		))
		assert alert.result_id == result.id
		assert alert.severity == "critical_high"


# ── QC management ─────────────────────────────────────────────────────────────

class TestQCManagement:

	def test_qc_run_passes(self):
		svc = fresh()
		inst = make_instrument(svc)
		qc = run(svc.run_qc(QCRunCreate(
			tenant_id=TID, instrument_id=inst.id,
			test_code="WBC", lot_number="L001", level="normal",
			measured_value=5.0, target_value=5.0, sd=0.2,
			performed_by="tech1", created_by="tech1",
		)))
		assert qc.z_score == 0.0
		assert qc.status == "passed"

	def test_qc_run_1_3s_violation_puts_instrument_on_hold(self):
		svc = fresh()
		inst = make_instrument(svc)
		qc = run(svc.run_qc(QCRunCreate(
			tenant_id=TID, instrument_id=inst.id,
			test_code="WBC", lot_number="L001", level="normal",
			measured_value=5.0 + 3.1 * 0.2, target_value=5.0, sd=0.2,
			performed_by="tech1", created_by="tech1",
		)))
		assert "1-3s" in qc.westgard_violations
		updated_inst = run(svc.get_instrument(TID, inst.id))
		assert updated_inst.status == "qc_hold"

	def test_qc_failure_action_recorded(self):
		svc = fresh()
		inst = make_instrument(svc)
		qc = run(svc.run_qc(QCRunCreate(
			tenant_id=TID, instrument_id=inst.id, test_code="WBC",
			lot_number="L001", level="normal",
			measured_value=5.62, target_value=5.0, sd=0.2,
			performed_by="tech1", created_by="tech1",
		)))
		action = run(svc.qc_failure_action(TID, qc.id, "recalibrate", "tech1"))
		assert action["corrective_action"] == "recalibrate"
		assert action["qc_run_id"] == qc.id

	def test_qc_material_run_westgard(self):
		svc = fresh()
		inst = make_instrument(svc)
		record = run(svc.qc_material_run(
			TID, inst.id, "L2", 5.0,
			{"mean": 5.0, "sd": 0.3}, "tech1", "WBC", "LOT001",
		))
		assert record["z_score"] == 0.0
		assert record["status"] == "passed"

	def test_get_qc_run(self):
		svc = fresh()
		inst = make_instrument(svc)
		qc = run(svc.run_qc(QCRunCreate(
			tenant_id=TID, instrument_id=inst.id, test_code="RBC",
			lot_number="L002", level="normal",
			measured_value=4.5, target_value=4.5, sd=0.15,
			performed_by="tech1", created_by="tech1",
		)))
		fetched = run(svc.get_qc_run(TID, qc.id))
		assert fetched.id == qc.id

	def test_generate_qc_summary(self):
		svc = fresh()
		inst = make_instrument(svc)
		run(svc.run_qc(QCRunCreate(
			tenant_id=TID, instrument_id=inst.id, test_code="WBC",
			lot_number="L001", level="normal",
			measured_value=5.0, target_value=5.0, sd=0.2,
			performed_by="tech1", created_by="tech1",
		)))
		summary = run(svc.generate_qc_summary(TID))
		assert summary["total_qc_runs"] == 1
		assert len(summary["by_instrument"]) == 1

	def test_external_proficiency_testing(self):
		svc = fresh()
		record = run(svc.external_proficiency_testing(
			TID, "CAP",
			{"WBC": 5.1, "RBC": 4.5, "Hb": 14.2},
			score=92.5, submitted_by="lab_manager",
		))
		assert record["satisfactory"]
		assert not record["corrective_action_required"]

	def test_external_proficiency_testing_unsatisfactory(self):
		svc = fresh()
		record = run(svc.external_proficiency_testing(
			TID, "CAP", {"WBC": 5.1}, score=72.0, submitted_by="lab_manager",
		))
		assert not record["satisfactory"]
		assert record["corrective_action_required"]


# ── Instrument management ─────────────────────────────────────────────────────

class TestInstrumentManagement:

	def test_register_and_retrieve_instrument(self):
		svc = fresh()
		inst = make_instrument(svc)
		fetched = run(svc.get_instrument(TID, inst.id))
		assert fetched.serial_number == "SN-CI-001"

	def test_update_instrument_status(self):
		svc = fresh()
		inst = make_instrument(svc)
		updated = run(svc.update_instrument_status(TID, inst.id, "maintenance"))
		assert updated.status == "maintenance"

	def test_update_instrument_properties(self):
		svc = fresh()
		inst = make_instrument(svc)
		updated = run(svc.update_instrument(
			TID, inst.id, AnalyserInterfaceUpdate(location="Biochemistry Lab")
		))
		assert updated.location == "Biochemistry Lab"

	def test_record_calibration_updates_dates(self):
		svc = fresh()
		inst = make_instrument(svc)
		cal = run(svc.record_calibration(TID, inst.id, "engineer1", notes="routine", pass_fail=True))
		assert cal["pass_fail"]
		updated_inst = run(svc.get_instrument(TID, inst.id))
		assert updated_inst.last_calibrated_at is not None
		assert updated_inst.calibration_due_at is not None
		assert updated_inst.status == "online"

	def test_failed_calibration_sets_offline(self):
		svc = fresh()
		inst = make_instrument(svc)
		run(svc.record_calibration(TID, inst.id, "engineer1", pass_fail=False))
		updated_inst = run(svc.get_instrument(TID, inst.id))
		assert updated_inst.status == "offline"

	def test_interface_analyser_hl7_message(self):
		svc = fresh()
		inst = make_instrument(svc)
		raw = (
			"MSH|^~\\&|LAB||HIS||20260604||ORU^R01|1|P|2.3\r"
			"OBX|1|NM|WBC^White Blood Count||5.2|10^3/uL|4.5-11.0|N|||F"
		)
		record = run(svc.interface_analyser(TID, inst.id, "hl7_v2", "ORU_R01", raw))
		assert record["instrument_id"] == inst.id
		assert record["result_count"] >= 1
		# Instrument message count incremented
		updated_inst = run(svc.get_instrument(TID, inst.id))
		assert updated_inst.message_count == 1

	def test_interface_analyser_astm_message(self):
		svc = fresh()
		inst = make_instrument(svc)
		raw = "H|\\^&|||Analyser|||||||P|1\rR|1|^^^WBC|5.2|10^3/uL|4.5-11.0|N|||F"
		record = run(svc.interface_analyser(TID, inst.id, "astm_e1381", "RESULT", raw))
		assert record["result_count"] >= 1


# ── External referrals ────────────────────────────────────────────────────────

class TestExternalReferrals:

	def _make_referral(self, svc):
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		return run(svc.create_referral(ExternalReferralCreate(
			tenant_id=TID, order_id=order.id, specimen_id=spec.id,
			patient_id="pat1", reference_lab_name="PathLab Kenya",
			reference_lab_code="PLK", test_code="HIV_PCR",
			test_name="HIV-1 PCR Quantitative",
			expected_tat_hours=72, dispatched_by="lab_manager",
			created_by="lab_manager",
		))), order, spec

	def test_create_and_retrieve_referral(self):
		svc = fresh()
		referral, _, _ = self._make_referral(svc)
		fetched = run(svc.get_referral(TID, referral.id))
		assert fetched.reference_lab_code == "PLK"
		assert fetched.status == "pending"

	def test_update_referral_tracking_number(self):
		svc = fresh()
		referral, _, _ = self._make_referral(svc)
		updated = run(svc.update_referral(TID, referral.id, ExternalReferralUpdate(
			tracking_number="TRK-12345", status="dispatched"
		)))
		assert updated.tracking_number == "TRK-12345"
		assert updated.status == "dispatched"

	def test_receive_external_result(self):
		svc = fresh()
		referral, _, _ = self._make_referral(svc)
		record = run(svc.receive_external_result(
			TID, referral.id,
			{"analyte": "HIV-1 RNA", "value": "Undetectable", "unit": "copies/mL"},
			"dr_smith",
		))
		assert record["referral_id"] == referral.id
		assert record["verified_by"] == "dr_smith"

	def test_list_referrals_by_status(self):
		svc = fresh()
		ref1, _, _ = self._make_referral(svc)
		run(svc.update_referral(TID, ref1.id, ExternalReferralUpdate(status="dispatched")))
		ref2, _, _ = self._make_referral(svc)
		dispatched = run(svc.list_referrals(TID, status="dispatched"))
		assert len(dispatched) == 1


# ── Report generation ─────────────────────────────────────────────────────────

class TestReportGeneration:

	def test_generate_lab_report(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		make_result(svc, order.id, spec.id)
		report = run(svc.generate_lab_report(TID, order.id))
		assert report["order"]["id"] == order.id
		assert len(report["results"]) == 1
		assert report["summary"]["total_results"] == 1

	def test_generate_lab_report_missing_order_raises(self):
		svc = fresh()
		with pytest.raises(KeyError):
			run(svc.generate_lab_report(TID, "nonexistent"))

	def test_generate_critical_value_report(self):
		svc = fresh()
		notif = run(svc.notify_critical_value(
			TID, "r1", "pat1", "K", 7.8, "mEq/L", "critical_high", "dr1", "tech1"
		))
		run(svc.acknowledge_critical_value(TID, notif.id, "dr1"))
		report = run(svc.generate_critical_value_report(TID))
		assert report["total_critical_values"] == 1
		assert report["acknowledged"] == 1

	def test_generate_rejection_report(self):
		svc = fresh()
		order = make_order(svc)
		spec = make_specimen(svc, order.id)
		run(svc.reject_specimen(TID, spec.id, "hemolyzed"))
		report = run(svc.generate_rejection_report(TID))
		assert report["rejected_count"] == 1
		assert report["rejection_rate_pct"] == 100.0
		assert "hemolyzed" in report["by_reason"]

	def test_generate_qc_summary_empty(self):
		svc = fresh()
		summary = run(svc.generate_qc_summary(TID))
		assert summary["total_qc_runs"] == 0
		assert summary["by_instrument"] == []

	def test_tat_monitoring(self):
		svc = fresh()
		make_order(svc)
		report = run(svc.tat_monitoring(TID, "today"))
		assert "overall" in report

	def test_workload_report(self):
		svc = fresh()
		make_order(svc)
		report = run(svc.lab_workload_report(TID, "today"))
		assert report["total_orders"] == 1

	def test_dashboard_summary(self):
		svc = fresh()
		make_order(svc)
		summary = run(svc.dashboard_summary(TID))
		assert summary["orders"]["total"] == 1


# ── Domain rules ──────────────────────────────────────────────────────────────

class TestDomainRules:

	def test_assert_tenant_context_empty_raises(self):
		with pytest.raises(RuleViolation) as exc_info:
			assert_tenant_context("")
		assert exc_info.value.rule_name == "tenant_context_required"

	def test_assert_tenant_context_none_raises(self):
		with pytest.raises(RuleViolation):
			assert_tenant_context(None)

	def test_assert_tenant_context_valid_passes(self):
		assert_tenant_context("my_tenant")  # no exception

	def test_cross_tenant_access_denied(self):
		with pytest.raises(RuleViolation) as exc_info:
			assert_no_cross_tenant_access("tenant_a", "tenant_b")
		assert exc_info.value.rule_name == "cross_tenant_access_denied"

	def test_same_tenant_access_allowed(self):
		assert_no_cross_tenant_access("t1", "t1")  # no exception

	def test_order_cancellable_from_pending(self):
		assert_order_cancellable("pending")  # no exception

	def test_order_not_cancellable_when_reported(self):
		with pytest.raises(RuleViolation):
			assert_order_cancellable("reported")

	def test_specimen_collectable_from_pending(self):
		assert_specimen_collectable("pending")

	def test_specimen_not_collectable_from_cancelled(self):
		with pytest.raises(RuleViolation):
			assert_specimen_collectable("cancelled")

	def test_rejection_reason_required(self):
		with pytest.raises(RuleViolation):
			assert_rejection_reason_present("")

	def test_rejection_reason_none_raises(self):
		with pytest.raises(RuleViolation):
			assert_rejection_reason_present(None)

	def test_critical_notification_required(self):
		with pytest.raises(RuleViolation):
			assert_critical_value_notification_sent(is_critical=True, notification_sent=False)

	def test_critical_notification_not_required_for_non_critical(self):
		assert_critical_value_notification_sent(is_critical=False, notification_sent=False)

	def test_result_release_requires_validation(self):
		with pytest.raises(RuleViolation):
			assert_result_validated_for_release("preliminary")

	def test_result_release_ok_when_final(self):
		assert_result_validated_for_release("final")

	def test_reference_range_inverted_bounds(self):
		with pytest.raises(RuleViolation):
			assert_reference_range_bounds_valid(low=100.0, high=50.0, critical_low=None, critical_high=None)

	def test_reference_range_critical_low_above_normal(self):
		with pytest.raises(RuleViolation):
			assert_reference_range_bounds_valid(low=70.0, high=110.0, critical_low=80.0, critical_high=None)

	def test_qc_hold_blocks_result(self):
		with pytest.raises(RuleViolation):
			assert_instrument_not_on_qc_hold("qc_hold")

	def test_qc_hold_ok_when_online(self):
		assert_instrument_not_on_qc_hold("online")

	def test_qc_lot_expired_raises(self):
		past = datetime.utcnow() - timedelta(days=1)
		with pytest.raises(RuleViolation):
			assert_qc_lot_not_expired(past)

	def test_qc_lot_valid_passes(self):
		future = datetime.utcnow() + timedelta(days=30)
		assert_qc_lot_not_expired(future)

	def test_calibration_overdue_raises(self):
		past = datetime.utcnow() - timedelta(days=1)
		with pytest.raises(RuleViolation):
			assert_calibration_not_overdue(past)

	def test_tat_deadline_stat(self):
		now = datetime.utcnow()
		deadline = calculate_tat_deadline(now, "stat", stat_tat_minutes=60)
		assert abs((deadline - now).total_seconds() - 3600) < 5

	def test_calculate_z_score(self):
		assert calculate_z_score(5.3, 5.0, 0.3) == pytest.approx(1.0, rel=1e-3)

	def test_calculate_cv_percent(self):
		assert calculate_cv_percent(5.0, 0.25) == pytest.approx(5.0)

	def test_calculate_delta_percent(self):
		assert calculate_delta_percent(5.5, 4.0) == pytest.approx(37.5)

	def test_calculate_rejection_rate(self):
		assert calculate_rejection_rate_pct(100, 3) == pytest.approx(3.0)

	def test_calculate_qc_pass_rate(self):
		assert calculate_qc_pass_rate_pct(10, 9) == pytest.approx(90.0)

	def test_calculate_percentile_median(self):
		vals = [1.0, 2.0, 3.0, 4.0, 5.0]
		assert calculate_percentile(vals, 50) == pytest.approx(3.0, rel=0.01)

	def test_calculate_percentile_empty(self):
		assert calculate_percentile([], 90) is None


# ── Domain calculations ────────────────────────────────────────────────────────

class TestDomainCalculations:

	def test_classify_normal(self):
		flag, is_critical = classify_numeric_result(95.0, 70.0, 110.0, 40.0, 400.0)
		assert flag is None
		assert not is_critical

	def test_classify_critical_high(self):
		flag, is_critical = classify_numeric_result(450.0, 70.0, 110.0, 40.0, 400.0)
		from lab.models import AbnormalFlag
		assert flag == AbnormalFlag.CRITICAL_HIGH
		assert is_critical

	def test_classify_critical_low(self):
		flag, is_critical = classify_numeric_result(35.0, 70.0, 110.0, 40.0, 400.0)
		from lab.models import AbnormalFlag
		assert flag == AbnormalFlag.CRITICAL_LOW
		assert is_critical

	def test_classify_high_no_explicit_critical(self):
		flag, is_critical = classify_numeric_result(170.0, 70.0, 110.0)
		# 170 > 110 * 1.5 → VERY_HIGH / critical
		from lab.models import AbnormalFlag
		assert flag == AbnormalFlag.VERY_HIGH
		assert is_critical

	def test_delta_check_exceeds(self):
		assert delta_check(5.5, 4.0, 25.0)  # 37.5% > 25%

	def test_delta_check_within(self):
		assert not delta_check(4.2, 4.0, 25.0)  # 5% < 25%

	def test_delta_check_zero_previous(self):
		assert delta_check(1.0, 0.0)

	def test_westgard_1_3s(self):
		violations, status = evaluate_westgard_rules([3.5])
		from lab.models import QCStatus
		assert "1-3s" in violations
		assert status == QCStatus.FAILED

	def test_westgard_1_2s_warning_only(self):
		violations, status = evaluate_westgard_rules([2.3])
		from lab.models import QCStatus
		assert "1-2s" in violations
		assert status == QCStatus.PENDING_REVIEW

	def test_westgard_r_4s(self):
		violations, status = evaluate_westgard_rules([-2.5, 2.5])
		assert "R-4s" in violations

	def test_westgard_10x(self):
		z_scores = [0.5] * 10
		violations, _ = evaluate_westgard_rules(z_scores)
		assert "10x" in violations

	def test_westgard_pass(self):
		violations, status = evaluate_westgard_rules([0.3, -0.5, 0.8])
		from lab.models import QCStatus
		assert status == QCStatus.PASSED
		assert violations == []

	def test_tat_metrics_basic(self):
		metrics = calculate_tat_metrics([30.0, 45.0, 60.0, 90.0, 120.0], [], 120.0)
		assert metrics["total_completed"] == 5
		assert metrics["overdue_count"] == 0

	def test_tat_metrics_overdue(self):
		metrics = calculate_tat_metrics([150.0, 200.0, 30.0], [], 120.0)
		assert metrics["overdue_count"] == 2

	def test_tat_metrics_empty(self):
		metrics = calculate_tat_metrics([], [], 120.0)
		assert metrics["total_completed"] == 0
		assert metrics["median_tat_minutes"] is None

	def test_select_reference_range_age_sex(self):
		ranges = [
			{"age_min_years": 0, "age_max_years": 17, "sex": None, "low": 4.0, "high": 10.0},
			{"age_min_years": 18, "age_max_years": None, "sex": "M", "low": 4.5, "high": 11.0},
			{"age_min_years": None, "age_max_years": None, "sex": None, "low": 4.0, "high": 10.0},
		]
		result = select_reference_range(ranges, 35.0, "M")
		assert result["low"] == 4.5  # age+sex specific match

	def test_select_reference_range_no_match(self):
		ranges = [
			{"age_min_years": 0, "age_max_years": 10, "sex": None, "low": 4.0, "high": 10.0}
		]
		result = select_reference_range(ranges, 40.0, "M")
		assert result is None

	def test_select_reference_range_universal(self):
		ranges = [
			{"age_min_years": None, "age_max_years": None, "sex": None, "low": 4.0, "high": 10.0}
		]
		result = select_reference_range(ranges, 50.0, "F")
		assert result is not None


# ── Tenant isolation ──────────────────────────────────────────────────────────

class TestTenantIsolation:

	def test_orders_isolated_by_tenant(self):
		svc = fresh()
		run(svc.create_order(LabOrderCreate(
			tenant_id="tenant_a", patient_id="p1", encounter_id="e1",
			test_code="CBC", test_name="CBC", test_category="hematology",
			ordered_by="dr1", clinical_indication="x", specimen_type="blood_venous",
			created_by="dr1",
		)))
		run(svc.create_order(LabOrderCreate(
			tenant_id="tenant_b", patient_id="p2", encounter_id="e2",
			test_code="CBC", test_name="CBC", test_category="hematology",
			ordered_by="dr2", clinical_indication="y", specimen_type="blood_venous",
			created_by="dr2",
		)))
		a_orders = run(svc.list_orders("tenant_a"))
		b_orders = run(svc.list_orders("tenant_b"))
		assert len(a_orders) == 1
		assert len(b_orders) == 1
		assert all(o.tenant_id == "tenant_a" for o in a_orders)

	def test_results_isolated_by_tenant(self):
		svc = fresh()
		for tid in ("t1", "t2"):
			order = run(svc.create_order(LabOrderCreate(
				tenant_id=tid, patient_id="p", encounter_id="e",
				test_code="CBC", test_name="CBC", test_category="hematology",
				ordered_by="dr", clinical_indication="x", specimen_type="blood_venous",
				created_by="dr",
			)))
			spec = run(svc.collect_specimen(SpecimenCreate(
				tenant_id=tid, order_id=order.id, patient_id="p",
				specimen_type="blood_venous", collected_by="n", collection_site="arm",
				created_by="n",
			)))
			run(svc.enter_result(LabResultCreate(
				tenant_id=tid, order_id=order.id, specimen_id=spec.id,
				analyte="Hb", value=13.5, unit="g/dL",
				reference_low=12.0, reference_high=17.0,
				result_status="preliminary", performed_by="tech", created_by="tech",
			)))
		assert len(run(svc.list_results("t1"))) == 1
		assert len(run(svc.list_results("t2"))) == 1
