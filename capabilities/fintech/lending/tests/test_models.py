"""
Unit tests for APG Digital Lending Pydantic v2 models.

Run: cd capabilities/fintech/lending && python -m pytest tests/test_models.py -vxs
"""

from __future__ import annotations

import pytest
from datetime import date, timedelta
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
	LoanProductCreate, LoanProductUpdate, LoanProductResponse,
	LoanApplicationCreate, LoanApplicationUpdate,
	CreditScoreCreate, CreditScoreResponse,
	LoanOfferCreate, LoanOfferUpdate,
	LoanCreate, LoanScheduleItem,
	RepaymentTransactionCreate,
	CollateralItemCreate, GuarantorRecordCreate,
	RestructureCreate, WriteOffCreate,
	AmortisationScheduleRequest,
	LoanApplicationStatus, LoanStatus, LoanProductType,
	ScheduleType, RepaymentFrequency, CollateralType,
	uuid7str,
)


# ---------------------------------------------------------------------------
# uuid7str
# ---------------------------------------------------------------------------

def test_uuid7str_is_string():
	uid = uuid7str()
	assert isinstance(uid, str)
	assert len(uid) == 36  # standard UUID format


def test_uuid7str_unique():
	assert uuid7str() != uuid7str()


# ---------------------------------------------------------------------------
# LoanProductCreate
# ---------------------------------------------------------------------------

def test_loan_product_create_valid():
	p = LoanProductCreate(
		tenant_id="t1", created_by="admin",
		code="TERM01", name="Term Loan",
		product_type=LoanProductType.TERM_LOAN,
		min_amount=10_000, max_amount=500_000,
		min_tenor_months=3, max_tenor_months=60,
		base_annual_rate=0.18,
	)
	assert p.code == "TERM01"
	assert p.base_annual_rate == 0.18
	assert p.currency == "KES"


def test_loan_product_create_tenor_validation():
	with pytest.raises(ValidationError):
		LoanProductCreate(
			tenant_id="t1", created_by="admin",
			code="X", name="X",
			product_type="term_loan",
			min_amount=10_000, max_amount=500_000,
			min_tenor_months=24, max_tenor_months=12,  # invalid
			base_annual_rate=0.18,
		)


def test_loan_product_create_amount_validation():
	with pytest.raises(ValidationError):
		LoanProductCreate(
			tenant_id="t1", created_by="admin",
			code="X", name="X",
			product_type="term_loan",
			min_amount=500_000, max_amount=10_000,  # invalid
			min_tenor_months=3, max_tenor_months=24,
			base_annual_rate=0.18,
		)


def test_loan_product_rate_bounds():
	with pytest.raises(ValidationError):
		LoanProductCreate(
			tenant_id="t1", created_by="admin",
			code="X", name="X",
			product_type="term_loan",
			min_amount=1_000, max_amount=10_000,
			min_tenor_months=1, max_tenor_months=12,
			base_annual_rate=1.5,  # > 1.0
		)


def test_loan_product_update_partial():
	u = LoanProductUpdate(base_annual_rate=0.20)
	assert u.base_annual_rate == 0.20
	assert u.name is None


# ---------------------------------------------------------------------------
# LoanApplicationCreate
# ---------------------------------------------------------------------------

def test_application_create_valid():
	app = LoanApplicationCreate(
		tenant_id="t1", created_by="agent",
		borrower_id="B001", product_id="TERM01",
		requested_amount=50_000, requested_tenor_months=12,
		purpose="business",
		income_source="employed",
		monthly_income=80_000,
		kyc_ref="KYC001",
	)
	assert app.requested_amount == 50_000


def test_application_create_negative_amount():
	with pytest.raises(ValidationError):
		LoanApplicationCreate(
			tenant_id="t1", created_by="a",
			borrower_id="B1", product_id="P1",
			requested_amount=-100,  # invalid
			requested_tenor_months=12,
			purpose="business",
			income_source="employed",
			monthly_income=50_000,
			kyc_ref="KYC001",
		)


# ---------------------------------------------------------------------------
# CreditScoreCreate
# ---------------------------------------------------------------------------

def test_credit_score_create_valid():
	cs = CreditScoreCreate(
		tenant_id="t1", created_by="system",
		borrower_id="B001",
		bureau_score=680, behavioural_score=720,
		demographic_score=700,
		payment_ratio=0.95, utilisation_ratio=0.30,
	)
	assert cs.bureau_score == 680


def test_credit_score_bounds():
	with pytest.raises(ValidationError):
		CreditScoreCreate(
			tenant_id="t1", created_by="system",
			borrower_id="B001",
			bureau_score=200,  # < 300
			behavioural_score=500,
			demographic_score=500,
			payment_ratio=0.9,
			utilisation_ratio=0.3,
		)


# ---------------------------------------------------------------------------
# LoanOfferCreate
# ---------------------------------------------------------------------------

def test_offer_create_valid():
	o = LoanOfferCreate(
		tenant_id="t1", created_by="uw",
		application_id="APP001", credit_score_id="CS001",
		offered_amount=80_000, annual_rate=0.20,
		tenor_months=12,
		expiry_date=date.today() + timedelta(days=7),
	)
	assert o.offered_amount == 80_000


def test_offer_rate_zero_invalid():
	with pytest.raises(ValidationError):
		LoanOfferCreate(
			tenant_id="t1", created_by="uw",
			application_id="APP001", credit_score_id="CS001",
			offered_amount=80_000, annual_rate=0.0,  # invalid: must be > 0
			tenor_months=12,
			expiry_date=date.today() + timedelta(days=7),
		)


# ---------------------------------------------------------------------------
# AmortisationScheduleRequest
# ---------------------------------------------------------------------------

def test_amortisation_request_valid():
	r = AmortisationScheduleRequest(
		principal=100_000,
		annual_rate=0.18,
		tenor_months=24,
		start_date=date.today(),
	)
	assert r.schedule_type == ScheduleType.REDUCING_BALANCE


def test_amortisation_request_invalid_principal():
	with pytest.raises(ValidationError):
		AmortisationScheduleRequest(
			principal=-1, annual_rate=0.18,
			tenor_months=12, start_date=date.today(),
		)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_enums_have_expected_values():
	assert LoanApplicationStatus.SUBMITTED == "submitted"
	assert LoanStatus.ACTIVE == "active"
	assert LoanProductType.MORTGAGE == "mortgage"
	assert ScheduleType.REDUCING_BALANCE == "reducing_balance"
	assert RepaymentFrequency.MONTHLY == "monthly"
	assert CollateralType.PROPERTY == "property"


# ---------------------------------------------------------------------------
# CollateralItemCreate
# ---------------------------------------------------------------------------

def test_collateral_create_valid():
	c = CollateralItemCreate(
		tenant_id="t1", created_by="a",
		loan_id="L001",
		collateral_type=CollateralType.PROPERTY,
		description="3BR house in Nairobi",
		market_value=5_000_000,
	)
	assert c.market_value == 5_000_000


def test_collateral_negative_value():
	with pytest.raises(ValidationError):
		CollateralItemCreate(
			tenant_id="t1", created_by="a",
			loan_id="L001",
			collateral_type="property",
			description="house",
			market_value=-100,
		)


# ---------------------------------------------------------------------------
# RestructureCreate
# ---------------------------------------------------------------------------

def test_restructure_create_valid():
	r = RestructureCreate(
		tenant_id="t1", created_by="uw",
		loan_id="L001",
		restructure_type="tenor_extension",
		new_tenor_months=36,
		reason="COVID hardship",
		approved_by="manager",
	)
	assert r.restructure_type == "tenor_extension"


# ---------------------------------------------------------------------------
# WriteOffCreate
# ---------------------------------------------------------------------------

def test_writeoff_create_valid():
	w = WriteOffCreate(
		tenant_id="t1", created_by="uw",
		loan_id="L001",
		reason="non_performing",
		write_off_date=date.today(),
		approved_by="credit_committee",
	)
	assert w.recovery_prospect == 0.0
