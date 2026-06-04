"""Service tests for pharma_dis."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.dis.service import PharmaceuticalDistributionService
from capabilities.pharma.dis.models import ShipmentCreate


def svc():
	return PharmaceuticalDistributionService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_dis"


def test_create_shipment():
	s = svc()
	payload = ShipmentCreate(
		tenant_id="t1", shipment_number="SHP-001", distribution_channel="hospital",
		origin_site="SITE-A", destination_site="HOSP-B",
		transport_mode="road", transport_condition="ambient",
		created_by="warehouse",
	)
	shipment = s.create_shipment(payload)
	assert shipment.shipment_number == "SHP-001"
	assert shipment.status == "planned"


def test_dispatch_shipment():
	s = svc()
	payload = ShipmentCreate(
		tenant_id="t1", shipment_number="SHP-002", distribution_channel="hospital",
		origin_site="SITE-A", destination_site="HOSP-C",
		transport_mode="courier", transport_condition="ambient",
		created_by="warehouse",
	)
	shipment = s.create_shipment(payload)
	dispatched = s.dispatch_shipment(
		shipment.id, "t1", "PL-001", "COA-001",
	)
	assert dispatched.status == "dispatched"


def test_dispatch_wholesale_requires_wda():
	s = svc()
	payload = ShipmentCreate(
		tenant_id="t1", shipment_number="SHP-003", distribution_channel="wholesale",
		origin_site="SITE-A", destination_site="DIST-A",
		transport_mode="road", transport_condition="ambient",
		created_by="warehouse",
	)
	shipment = s.create_shipment(payload)
	with pytest.raises(PermissionError):
		s.dispatch_shipment(shipment.id, "t1", "PL-001", "COA-001")


def test_report_excursion():
	s = svc()
	payload = ShipmentCreate(
		tenant_id="t1", shipment_number="SHP-004", distribution_channel="hospital",
		origin_site="SITE-A", destination_site="HOSP-D",
		transport_mode="road", transport_condition="refrigerated_2_8",
		created_by="warehouse",
	)
	shipment = s.create_shipment(payload)
	cc = s.create_cold_chain_record(
		"t1", shipment.id, "PROD-A", "refrigerated_2_8",
		2.0, 8.0, "LOGGER-001", "QUAL-001", "warehouse",
	)
	excursion = s.report_excursion(
		"t1", cc.id, shipment.id,
		datetime.utcnow() - timedelta(hours=2),
		10.5, 12.0, "minor", "warehouse",
	)
	assert excursion.severity == "minor"


def test_verify_serialisation_not_found():
	s = svc()
	result = s.verify_serialisation("t1", "NONEXISTENT-SN")
	assert result["verified"] is False
	assert result["reason"] == "not_found"


def test_verify_serialisation_success():
	s = svc()
	s.serialise_product("t1", "PROD-A", "SN-12345", "BTH-001", "gs1_sgtin", "item", "GTIN123", "warehouse")
	result = s.verify_serialisation("t1", "SN-12345")
	assert result["verified"] is True


def test_initiate_recall():
	s = svc()
	recall = s.initiate_recall(
		"t1", "RCL-001", "class_ii", "PROD-A",
		["BTH-001", "BTH-002"], "Contamination found", "EU market", "qa_mgr",
	)
	assert recall.recall_number == "RCL-001"
	assert recall.recall_class == "class_ii"
	assert recall.status == "initiated"


def test_register_and_grant_wda():
	s = svc()
	wda = s.register_wda(
		"t1", "WDA-KE-001", "ke", "Pharma Co Ltd",
		"123 Industrial Area", ["wholesale", "retail_pharmacy"],
		"PPB Kenya", "ra_mgr",
	)
	assert wda.status == "applied"
	now = datetime.utcnow()
	granted = s.grant_wda(wda.id, "t1", now, now + timedelta(days=1095))
	assert granted.status == "granted"


def test_wda_expiry_check():
	s = svc()
	now = datetime.utcnow()
	wda = s.register_wda("t1", "WDA-001", "ke", "Company", "Address", [], "Authority", "ra")
	s.grant_wda(wda.id, "t1", now, now + timedelta(days=30))
	alerts = s.check_wda_expiry("t1")
	assert len(alerts) == 1


def test_dashboard_summary():
	s = svc()
	payload = ShipmentCreate(
		tenant_id="t1", shipment_number="SHP-010", distribution_channel="hospital",
		origin_site="A", destination_site="B", transport_mode="road",
		transport_condition="ambient", created_by="w",
	)
	s.create_shipment(payload)
	summary = s.dashboard_summary("t1")
	assert summary["shipment_count"] == 1
