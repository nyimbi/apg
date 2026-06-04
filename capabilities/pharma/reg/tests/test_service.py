"""Service tests for pharma_reg."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import pytest
from datetime import datetime, timedelta

from capabilities.pharma.reg.service import ProductRegistrationService
from capabilities.pharma.reg.models import ProductRegistrationCreate


def svc():
	return ProductRegistrationService()


def test_describe():
	s = svc()
	c = s.describe("t1")
	assert c["capability"] == "pharma_reg"


def test_create_registration():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-A", product_name="Drug X 10mg",
		product_type="small_molecule", registration_type="new_application",
		region="eu_ema", created_by="ra_mgr",
	)
	reg = s.create_registration(payload)
	assert reg.product_name == "Drug X 10mg"
	assert reg.status == "not_submitted"


def test_submit_registration():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-B", product_name="Drug Y 20mg",
		product_type="biologic", registration_type="new_application",
		region="us_fda", created_by="ra",
	)
	reg = s.create_registration(payload)
	dossier = s.compile_dossier("t1", "DOS-001", "PROD-B", "ctd_ectd", "1.0",
								["module_1", "module_2", "module_3"], "ra")
	s.validate_ectd(dossier.id, "t1")
	submitted = s.submit_registration(
		reg.id, "t1", dossier.id, "LOCAL-REP-001",
		qp_signed_off=True, ectd_validated=True,
	)
	assert submitted.status == "submitted"


def test_submit_denied_no_qp():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-C", product_name="Drug Z",
		product_type="small_molecule", registration_type="new_application",
		region="us_fda", created_by="ra",
	)
	reg = s.create_registration(payload)
	with pytest.raises(PermissionError):
		s.submit_registration(reg.id, "t1", "DOS-001", "LOCAL-001", qp_signed_off=False, ectd_validated=True)


def test_approve_registration():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-D", product_name="Drug W",
		product_type="small_molecule", registration_type="new_application",
		region="uk_mhra", created_by="ra",
	)
	reg = s.create_registration(payload)
	now = datetime.utcnow()
	approved = s.approve_registration(
		reg.id, "t1", "MA-UK-12345", now, now + timedelta(days=1825),
	)
	assert approved.status == "approved"
	assert approved.registration_number == "MA-UK-12345"


def test_compile_dossier():
	s = svc()
	dossier = s.compile_dossier(
		"t1", "DOS-001", "PROD-A", "ctd_ectd", "1.0",
		["module_1", "module_2"], "ra",
	)
	assert dossier.format == "ctd_ectd"
	assert not dossier.ectd_validated


def test_validate_ectd():
	s = svc()
	dossier = s.compile_dossier("t1", "DOS-002", "PROD-B", "ctd_ectd", "1.0", [], "ra")
	validated = s.validate_ectd(dossier.id, "t1")
	assert validated.ectd_validated is True


def test_record_authority_interaction():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-E", product_name="Drug V",
		product_type="small_molecule", registration_type="new_application",
		region="eu_ema", created_by="ra",
	)
	reg = s.create_registration(payload)
	interaction = s.record_interaction(
		"t1", reg.id, "scientific_advice", "EMA",
		datetime.utcnow(), "ra_mgr", "MINUTES-001",
	)
	assert interaction.interaction_type == "scientific_advice"


def test_renewal_alerts():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-F", product_name="Drug U",
		product_type="small_molecule", registration_type="renewal",
		region="eu_ema", created_by="ra",
	)
	reg = s.create_registration(payload)
	now = datetime.utcnow()
	s.approve_registration(reg.id, "t1", "MA-EU-001", now, now + timedelta(days=90))
	alerts = s.check_renewal_alerts("t1")
	assert len(alerts) == 1


def test_dashboard_summary():
	s = svc()
	payload = ProductRegistrationCreate(
		tenant_id="t1", product_id="PROD-G", product_name="Drug T",
		product_type="biosimilar", registration_type="new_application",
		region="us_fda", created_by="ra",
	)
	s.create_registration(payload)
	summary = s.dashboard_summary("t1")
	assert summary["registration_count"] == 1
