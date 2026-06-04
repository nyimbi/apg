"""Service tests for pharma_sup."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.sup.service import PharmaceuticalSupplyChainService
from capabilities.pharma.sup.models import SupplierCreate


def svc():
	return PharmaceuticalSupplyChainService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_sup"


def test_create_supplier():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-001", name="API Manufacturer Ltd",
		supplier_type="api_manufacturer", country="IN", created_by="procurement",
	)
	supplier = s.create_supplier(payload)
	assert supplier.supplier_code == "SUP-001"
	assert supplier.qualification_status == "unqualified"


def test_qualify_supplier():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-002", name="Excipient Corp",
		supplier_type="excipient_supplier", country="DE", created_by="procurement",
	)
	supplier = s.create_supplier(payload)
	qualified = s.qualify_supplier(
		supplier.id, "t1", "QA-AGR-001",
		datetime.utcnow(), ["EXCIPIENT-A", "EXCIPIENT-B"],
	)
	assert qualified.qualification_status == "qualified"
	assert qualified.on_approved_supplier_list is True


def test_suspend_supplier():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-003", name="Problem Supplier",
		supplier_type="excipient_supplier", country="CN", created_by="procurement",
	)
	supplier = s.create_supplier(payload)
	s.qualify_supplier(supplier.id, "t1", "QA-001", datetime.utcnow(), [])
	suspended = s.suspend_supplier(supplier.id, "t1", "Quality failure")
	assert suspended.qualification_status == "suspended"
	assert suspended.on_approved_supplier_list is False


def test_place_order_denied_not_on_asl():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-004", name="New Supplier",
		supplier_type="api_manufacturer", country="US", created_by="p",
	)
	supplier = s.create_supplier(payload)
	with pytest.raises(PermissionError):
		s.place_order("t1", "PO-001", "purchase_order", supplier.id,
					"PROD-A", 100.0, "kg", "buyer")


def test_place_order_success():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-005", name="Qualified Supplier",
		supplier_type="api_manufacturer", country="IN", created_by="p",
	)
	supplier = s.create_supplier(payload)
	s.qualify_supplier(supplier.id, "t1", "QA-002", datetime.utcnow(), ["PROD-A"])
	order = s.place_order("t1", "PO-001", "purchase_order", supplier.id,
						"PROD-A", 50.0, "kg", "buyer")
	assert order.po_number == "PO-001"
	assert order.status == "placed"


def test_activate_cmo():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="CMO-001", name="Contract Mfg Co",
		supplier_type="cmo", country="NL", created_by="sc",
	)
	supplier = s.create_supplier(payload)
	cmo = s.activate_cmo(
		"t1", "CMO-NL-001", "CMO Netherlands", "drug_product",
		supplier.id, "TA-001", "QA-001", "sc_mgr",
	)
	assert cmo.cmo_code == "CMO-NL-001"
	assert cmo.active is True


def test_create_forecast():
	s = svc()
	forecast = s.create_forecast(
		"t1", "FC-Q1-2026", "PROD-A", "s_op", "Q1-2026",
		12, {"Jan": 100.0, "Feb": 120.0}, 30.0, "demand_planner",
	)
	assert forecast.forecast_number == "FC-Q1-2026"
	assert forecast.sop_approved is False


def test_apply_and_grant_import_license():
	s = svc()
	lic = s.apply_import_license(
		"t1", "IMP-KE-001", "import_permit", "ke",
		["PROD-A"], "AUTH-REF-001", "PPB Kenya", "Import of Drug X", "ra",
	)
	assert lic.license_number == "IMP-KE-001"
	assert lic.status == "applied"
	now = datetime.utcnow()
	granted = s.grant_import_license(lic.id, "t1", now, now + timedelta(days=365))
	assert granted.status == "active"


def test_import_license_expiry_alert():
	s = svc()
	now = datetime.utcnow()
	lic = s.apply_import_license(
		"t1", "IMP-001", "import_permit", "ke",
		["PROD-A"], "AUTH-001", "PPB", "scope", "ra",
	)
	s.grant_import_license(lic.id, "t1", now, now + timedelta(days=30))
	alerts = s.check_import_license_expiry("t1")
	assert len(alerts) == 1


def test_supply_security_update():
	s = svc()
	record = s.update_supply_security(
		"t1", "PROD-A", "secure", "low", "SUP-001", "sc_analyst"
	)
	assert record.supply_status == "secure"
	assert record.risk_level == "low"


def test_dashboard_summary():
	s = svc()
	payload = SupplierCreate(
		tenant_id="t1", supplier_code="SUP-010", name="Test Supplier",
		supplier_type="api_manufacturer", country="IN", created_by="p",
	)
	s.create_supplier(payload)
	summary = s.dashboard_summary("t1")
	assert summary["total_supplier_count"] == 1
	assert summary["qualified_supplier_count"] == 0
