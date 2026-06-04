"""Service tests for pharma_mfg."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest

from capabilities.pharma.mfg.service import PharmaceuticalManufacturingService
from capabilities.pharma.mfg.models import BatchRecordCreate


def svc():
	return PharmaceuticalManufacturingService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_mfg"


def test_create_batch():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-001", product_id="PROD-A",
		manufacturing_type="drug_product", master_formula_reference="MF-001",
		planned_quantity=100.0, unit_of_measure="kg", created_by="prod_mgr",
	)
	batch = s.create_batch(payload)
	assert batch.batch_number == "BTH-001"
	assert batch.status == "planned"


def test_create_batch_denied_no_master_formula():
	s = svc()
	with pytest.raises(PermissionError):
		s._enforce({
			"tenant_id": "t1",
			"tenant_context_present": True,
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_batch",
			"manufacturing_type_supported": True,
			"master_formula_present": False,
			"batch_number_present": True,
		})


def test_release_batch_requires_qp():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-002", product_id="PROD-B",
		manufacturing_type="bulk_drug_substance", master_formula_reference="MF-002",
		planned_quantity=200.0, unit_of_measure="L", created_by="prod",
	)
	batch = s.create_batch(payload)
	with pytest.raises(PermissionError):
		s.release_batch(batch.id, "t1", "", "SIG-001")


def test_release_batch_success():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-003", product_id="PROD-C",
		manufacturing_type="finished_dose", master_formula_reference="MF-003",
		planned_quantity=50.0, unit_of_measure="units", created_by="prod",
	)
	batch = s.create_batch(payload)
	released = s.release_batch(batch.id, "t1", "QP-001", "ESIG-001")
	assert released.status == "released"


def test_reject_batch():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-004", product_id="PROD-D",
		manufacturing_type="drug_product", master_formula_reference="MF-004",
		planned_quantity=75.0, unit_of_measure="kg", created_by="prod",
	)
	batch = s.create_batch(payload)
	rejected = s.reject_batch(batch.id, "t1", "OOS result")
	assert rejected.status == "rejected"


def test_register_equipment():
	s = svc()
	equip = s.register_equipment("t1", "EQ-001", "Mixer", "mixing", "Line 1", "maint")
	assert equip.equipment_id == "EQ-001"
	assert equip.status == "under_qualification"


def test_raise_deviation():
	s = svc()
	dev = s.raise_deviation("t1", "DEV-001", "process_deviation", "major",
							"Yield below spec", "operator")
	assert dev.deviation_number == "DEV-001"
	assert dev.status == "open"


def test_close_deviation():
	s = svc()
	dev = s.raise_deviation("t1", "DEV-002", "equipment_deviation", "minor",
							"Calibration drift", "tech")
	closed = s.close_deviation(dev.id, "t1", "Recalibrated equipment")
	assert closed.status == "closed"
	assert closed.root_cause == "Recalibrated equipment"


def test_record_yield():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-010", product_id="PROD-E",
		manufacturing_type="drug_product", master_formula_reference="MF-010",
		planned_quantity=100.0, unit_of_measure="kg", created_by="prod",
	)
	batch = s.create_batch(payload)
	record = s.record_yield("t1", batch.id, "step_yield", "Granulation", 100.0, 98.0, "prod")
	assert record.actual_quantity == 98.0
	assert record.variance_pct is not None


def test_receive_and_release_material():
	s = svc()
	material = s.receive_material(
		"t1", "MAT-001", "API-A", "active_ingredient", "VENDOR-001",
		"LOT-2026-001", 50.0, "kg", "controlled_room_temp",
		vendor_qualified=True, created_by="warehouse",
	)
	assert material.status == "quarantine"
	released = s.release_material(material.id, "t1", "QC-CERT-001")
	assert released.status == "released"


def test_dashboard_summary():
	s = svc()
	payload = BatchRecordCreate(
		tenant_id="t1", batch_number="BTH-020", product_id="PROD-F",
		manufacturing_type="drug_product", master_formula_reference="MF-020",
		planned_quantity=60.0, unit_of_measure="kg", created_by="prod",
	)
	s.create_batch(payload)
	summary = s.dashboard_summary("t1")
	assert summary["batch_count"] == 1
	assert "open_deviations" in summary
