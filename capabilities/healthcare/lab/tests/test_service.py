"""Tests for LaboratoryInformationService."""

from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lab.models import InstrumentCreate, LabOrderCreate, LabResultCreate, QCRunCreate, SpecimenCreate
from lab.service import LaboratoryInformationService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return LaboratoryInformationService()


def make_order(s: LaboratoryInformationService, tid: str = "t", priority: str = "routine") -> any:
	return run(s.create_order(LabOrderCreate(
		tenant_id=tid, patient_id="p1", encounter_id="e1",
		test_code="CBC", test_name="Complete Blood Count", test_category="hematology",
		collection_priority=priority, ordered_by="dr1", clinical_indication="Anemia workup",
		specimen_type="blood_venous", created_by="dr1",
	)))


def test_create_order():
	s = svc()
	order = make_order(s)
	assert order.id and order.status == "pending"


def test_create_order_unsupported_category_denied():
	"""Pydantic rejects invalid enums at validation time; both Pydantic ValidationError
	and PolicyViolationError represent a correctly rejected request."""
	from pydantic import ValidationError
	s = svc()
	try:
		run(s.create_order(LabOrderCreate(
			tenant_id="t", patient_id="p1", encounter_id="e1",
			test_code="X", test_name="X", test_category="unknown_cat",
			ordered_by="dr1", clinical_indication="x", specimen_type="blood_venous", created_by="dr1",
		)))
		assert False, "expected a rejection"
	except (PolicyViolationError, ValidationError):
		pass


def test_cancel_order():
	s = svc()
	order = make_order(s)
	cancelled = run(s.cancel_order("t", order.id, "patient_refused"))
	assert cancelled.status == "cancelled"


def test_collect_specimen_links_order():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(
		tenant_id="t", order_id=order.id, patient_id="p1",
		specimen_type="blood_venous", collected_by="nurse1",
		collection_site="left_antecubital", created_by="nurse1",
	)))
	assert spec.status == "collected"
	updated_order = run(s.get_order("t", order.id))
	assert updated_order.status == "collected"
	assert updated_order.specimen_id == spec.id


def test_collect_cancelled_order_denied():
	s = svc()
	order = make_order(s)
	run(s.cancel_order("t", order.id, "reason"))
	try:
		run(s.collect_specimen(SpecimenCreate(
			tenant_id="t", order_id=order.id, patient_id="p1",
			specimen_type="blood_venous", collected_by="nurse1",
			collection_site="arm", created_by="nurse1",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_reject_specimen_requires_reason():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(tenant_id="t", order_id=order.id, patient_id="p1", specimen_type="blood_venous", collected_by="u", collection_site="arm", created_by="u")))
	try:
		run(s.reject_specimen("t", spec.id, ""))
		assert False
	except PolicyViolationError:
		pass


def test_reject_specimen_valid():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(tenant_id="t", order_id=order.id, patient_id="p1", specimen_type="blood_venous", collected_by="u", collection_site="arm", created_by="u")))
	rejected = run(s.reject_specimen("t", spec.id, "hemolyzed"))
	assert rejected.status == "rejected" and rejected.rejection_reason == "hemolyzed"


def test_enter_result_flags_critical():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(tenant_id="t", order_id=order.id, patient_id="p1", specimen_type="blood_venous", collected_by="u", collection_site="arm", created_by="u")))
	result = run(s.enter_result(LabResultCreate(
		tenant_id="t", order_id=order.id, specimen_id=spec.id,
		analyte="Potassium", value=7.8, unit="mEq/L",
		reference_low=3.5, reference_high=5.0,
		result_status="preliminary", performed_by="tech1", created_by="tech1",
	)))
	assert result.critical_value is True
	assert result.abnormal_flag in ("HH",)


def test_verify_result_blocks_without_critical_notification():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(tenant_id="t", order_id=order.id, patient_id="p1", specimen_type="blood_venous", collected_by="u", collection_site="arm", created_by="u")))
	result = run(s.enter_result(LabResultCreate(tenant_id="t", order_id=order.id, specimen_id=spec.id, analyte="K", value=7.8, unit="mEq/L", reference_low=3.5, reference_high=5.0, result_status="preliminary", performed_by="tech1", created_by="tech1")))
	assert result.critical_value
	try:
		run(s.verify_result("t", result.id, "supervisor", notification_sent=False))
		assert False
	except PolicyViolationError:
		pass


def test_verify_result_with_notification():
	s = svc()
	order = make_order(s)
	spec = run(s.collect_specimen(SpecimenCreate(tenant_id="t", order_id=order.id, patient_id="p1", specimen_type="blood_venous", collected_by="u", collection_site="arm", created_by="u")))
	result = run(s.enter_result(LabResultCreate(tenant_id="t", order_id=order.id, specimen_id=spec.id, analyte="Glucose", value=95.0, unit="mg/dL", reference_low=70.0, reference_high=110.0, result_status="preliminary", performed_by="tech1", created_by="tech1")))
	verified = run(s.verify_result("t", result.id, "pathologist", notification_sent=True))
	assert verified.result_status == "final" and verified.verified_by == "pathologist"


def test_run_qc_westgard_pass():
	s = svc()
	inst = run(s.register_instrument(InstrumentCreate(tenant_id="t", name="Analyzer A", model="XN-9100", serial_number="SN001", manufacturer="Sysmex", location="Lab", created_by="admin")))
	qc = run(s.run_qc(QCRunCreate(tenant_id="t", instrument_id=inst.id, test_code="WBC", lot_number="L001", level="normal", measured_value=5.2, target_value=5.0, sd=0.3, performed_by="tech1", created_by="tech1")))
	assert qc.status in ("passed", "pending_review")
	assert abs(qc.z_score) < 2.0


def test_run_qc_westgard_fail_puts_instrument_on_hold():
	s = svc()
	inst = run(s.register_instrument(InstrumentCreate(tenant_id="t", name="B", model="M", serial_number="SN002", manufacturer="Mfr", location="Lab", created_by="admin")))
	run(s.run_qc(QCRunCreate(tenant_id="t", instrument_id=inst.id, test_code="WBC", lot_number="L001", level="normal", measured_value=6.0, target_value=5.0, sd=0.3, performed_by="tech1", created_by="tech1")))
	updated_inst = [i for i in run(s.list_instruments("t")) if i.id == inst.id][0]
	assert updated_inst.status == "qc_hold"


def test_critical_value_notify_and_acknowledge():
	s = svc()
	notif = run(s.notify_critical_value("t", "res_001", "p1", "K", 7.8, "mEq/L", "critical_high", "dr1", "tech1"))
	assert notif.acknowledged_by is None
	ack = run(s.acknowledge_critical_value("t", notif.id, "dr1"))
	assert ack.acknowledged_by == "dr1" and ack.acknowledged_at is not None


def test_dashboard_summary():
	s = svc()
	make_order(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["orders"]["total"] == 1
	assert "critical_values" in summary
