"""Tests for MedicalDeviceManagementService."""

from __future__ import annotations
import asyncio, sys, os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dev.models import AdverseEventCreate, CalibrationRecordCreate, DeviceCreate, MaintenanceScheduleCreate
from dev.service import MedicalDeviceManagementService, PolicyViolationError


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def svc():
	return MedicalDeviceManagementService()


def make_device(s, tid="t", device_class="class_i", udi=None) -> any:
	return run(s.register_device(DeviceCreate(
		tenant_id=tid, name="Pulse Oximeter", device_type="patient_monitor",
		device_class=device_class, manufacturer="Nellcor", model_number="PM-100",
		serial_number="SN12345", udi=udi, location="ICU", department="Critical Care",
		created_by="biomed",
	)))


def test_register_class_i_device_no_udi():
	s = svc()
	device = make_device(s, device_class="class_i")
	assert device.id and device.status == "active"


def test_register_class_ii_requires_udi():
	s = svc()
	try:
		make_device(s, device_class="class_ii", udi=None)
		assert False
	except PolicyViolationError:
		pass


def test_register_class_ii_with_udi():
	s = svc()
	device = make_device(s, device_class="class_ii", udi="(01)12345678901234")
	assert device.udi == "(01)12345678901234"


def test_update_device_status():
	s = svc()
	device = make_device(s)
	updated = run(s.update_device_status("t", device.id, "in_maintenance"))
	assert updated.status == "in_maintenance"


def test_schedule_maintenance():
	s = svc()
	device = make_device(s)
	sched = run(s.schedule_maintenance(MaintenanceScheduleCreate(
		tenant_id="t", device_id=device.id, maintenance_type="preventive",
		scheduled_date=datetime.utcnow() + timedelta(days=7),
		assigned_to="biomed_tech", estimated_hours=2.0,
		instructions="Annual PM checklist", created_by="biomed",
	)))
	assert sched.status == "open" and sched.work_order_id is not None


def test_complete_maintenance():
	s = svc()
	device = make_device(s)
	sched = run(s.schedule_maintenance(MaintenanceScheduleCreate(
		tenant_id="t", device_id=device.id, maintenance_type="calibration",
		scheduled_date=datetime.utcnow(), assigned_to="tech1",
		estimated_hours=1.0, instructions="Calibrate", created_by="biomed",
	)))
	completed = run(s.complete_maintenance("t", sched.id, "Completed per protocol"))
	assert completed.status == "completed" and completed.completed_at is not None


def test_record_calibration_updates_device():
	s = svc()
	device = make_device(s)
	cal = run(s.record_calibration(CalibrationRecordCreate(
		tenant_id="t", device_id=device.id, calibrated_by="tech1",
		calibration_date=datetime.utcnow(),
		next_due_date=datetime.utcnow() + timedelta(days=365),
		certificate_reference="CERT-001", result="pass", created_by="biomed",
	)))
	assert cal.result == "pass"
	updated_device = run(s.get_device("t", device.id))
	assert updated_device.calibration_status == "current"
	assert updated_device.last_calibrated_at is not None


def test_calibration_without_cert_denied():
	s = svc()
	device = make_device(s)
	try:
		run(s.record_calibration(CalibrationRecordCreate(
			tenant_id="t", device_id=device.id, calibrated_by="tech1",
			calibration_date=datetime.utcnow(),
			next_due_date=datetime.utcnow() + timedelta(days=365),
			certificate_reference="", result="pass", created_by="biomed",
		)))
		assert False
	except PolicyViolationError:
		pass


def test_report_adverse_event():
	s = svc()
	device = make_device(s)
	event = run(s.report_adverse_event(AdverseEventCreate(
		tenant_id="t", device_id=device.id, event_type="malfunction",
		severity="moderate", description="Device alarmed unexpectedly",
		occurred_at=datetime.utcnow(), reported_by="nurse1",
		immediate_action_taken="Switched to backup device", created_by="nurse1",
	)))
	assert event.status == "open" and event.severity == "moderate"


def test_serious_adverse_event_puts_device_in_maintenance():
	s = svc()
	device = make_device(s)
	run(s.report_adverse_event(AdverseEventCreate(
		tenant_id="t", device_id=device.id, event_type="patient_injury",
		severity="serious", description="Patient burned by heating pad",
		occurred_at=datetime.utcnow(), reported_by="nurse1",
		immediate_action_taken="Device removed from service", created_by="nurse1",
	)))
	updated_device = run(s.get_device("t", device.id))
	assert updated_device.status == "in_maintenance"


def test_close_adverse_event():
	s = svc()
	device = make_device(s)
	event = run(s.report_adverse_event(AdverseEventCreate(
		tenant_id="t", device_id=device.id, event_type="alarm_failure",
		severity="minor", description="False alarm",
		occurred_at=datetime.utcnow(), reported_by="tech1",
		immediate_action_taken="Acknowledged", created_by="tech1",
	)))
	closed = run(s.close_adverse_event("t", event.id, "Software defect", "Firmware update applied"))
	assert closed.status == "closed" and closed.root_cause is not None


def test_dashboard_summary():
	s = svc()
	make_device(s)
	summary = run(s.dashboard_summary("t"))
	assert summary["devices"]["total"] == 1
	assert "adverse_events" in summary and "maintenance" in summary
