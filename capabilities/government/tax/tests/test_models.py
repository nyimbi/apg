"""Unit tests for Tax Administration Pydantic v2 models.

No @pytest.mark.asyncio decorators — plain functions + async where needed.
Real objects, no mocks.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))

from models import (
	TaxpayerCreate, TaxpayerUpdate, TaxpayerResponse,
	TaxReturnCreate, TaxReturnResponse,
	TaxAssessmentCreate, TaxAssessmentResponse,
	TaxPaymentCreate, TaxPaymentResponse,
	TaxDebtCreate, TaxDebtResponse,
	TaxAuditCreate, TaxAuditResponse,
	AuditFindingCreate, AuditFindingResponse,
	ObjectionCreate, ObjectionResponse,
	AppealCreate, AppealResponse,
	TaxRefundCreate, TaxRefundResponse,
	PenaltyCreate, PenaltyResponse,
	InterestCreate, InterestResponse,
	TaxClearanceCertificateCreate, TaxClearanceCertificateResponse,
	TaxDashboardKPI, ComplianceRiskProfile, DebtAgingBucket, RevenueReport,
	DemandNotice, EOIRequest,
	TaxType, TaxpayerType, TaxpayerStatus, ReturnType, ReturnStatus,
	AssessmentType, AssessmentStatus, PaymentMethod, PaymentStatus,
	DebtStatus, AuditType, AuditStatus, FindingType, ObjectionStatus,
	AppealStatus, RefundStatus, PenaltyType, PenaltyStatus, InterestType,
	ClearanceCertificateStatus, ObligationStatus,
	uuid7str,
)
from datetime import date, datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tenant_id() -> str:
	return "test_tenant"


@pytest.fixture
def taxpayer_create(tenant_id) -> TaxpayerCreate:
	return TaxpayerCreate(
		tenant_id=tenant_id,
		taxpayer_type=TaxpayerType.INDIVIDUAL,
		tax_pin="A000000001X",
		national_id="12345678",
		taxpayer_name="Alice Wanjiku",
		email="alice@example.com",
		phone="0712345678",
		physical_address="Nairobi",
		tax_types=[TaxType.INCOME_TAX, TaxType.VAT],
		evidence_reference="reg_001",
	)


@pytest.fixture
def taxpayer_response(tenant_id) -> TaxpayerResponse:
	return TaxpayerResponse(
		tenant_id=tenant_id,
		taxpayer_type=TaxpayerType.COMPANY,
		tax_pin="P000000002B",
		taxpayer_name="Acme Ltd",
		tax_types=[TaxType.CORPORATE_TAX, TaxType.VAT],
		country_of_incorporation="KE",
		is_resident=True,
		evidence_reference="reg_002",
		status=TaxpayerStatus.ACTIVE,
	)


# ---------------------------------------------------------------------------
# Tests: Enumerations
# ---------------------------------------------------------------------------

def test_all_tax_types_parseable():
	values = [
		"income_tax", "vat", "corporate_tax", "withholding_tax",
		"capital_gains_tax", "excise_duty", "customs_duty", "stamp_duty",
		"rental_income_tax", "turnover_tax", "digital_services_tax", "presumptive_tax",
	]
	for v in values:
		assert TaxType(v) is not None


def test_all_taxpayer_statuses():
	for s in TaxpayerStatus:
		assert isinstance(s.value, str)


def test_all_return_statuses():
	for s in ReturnStatus:
		assert isinstance(s.value, str)


def test_all_assessment_statuses():
	for s in AssessmentStatus:
		assert isinstance(s.value, str)


# ---------------------------------------------------------------------------
# Tests: TaxpayerCreate
# ---------------------------------------------------------------------------

def test_taxpayer_create_valid(taxpayer_create):
	assert taxpayer_create.taxpayer_name == "Alice Wanjiku"
	assert taxpayer_create.taxpayer_type == TaxpayerType.INDIVIDUAL
	assert TaxType.VAT in taxpayer_create.tax_types


def test_taxpayer_create_strips_whitespace():
	tc = TaxpayerCreate(
		tenant_id="t1",
		taxpayer_type=TaxpayerType.INDIVIDUAL,
		tax_pin="A000000003C",
		national_id="99999",
		taxpayer_name="  Bob Otieno  ",
		evidence_reference=" reg_003 ",
	)
	assert tc.taxpayer_name == "Bob Otieno"
	assert tc.evidence_reference == "reg_003"


def test_taxpayer_create_requires_tenant():
	with pytest.raises(Exception):
		TaxpayerCreate(
			tenant_id="",
			taxpayer_type=TaxpayerType.INDIVIDUAL,
			tax_pin="A000000004D",
			national_id="11111",
			taxpayer_name="Carol",
			evidence_reference="ev",
		)


def test_taxpayer_create_extra_fields_forbidden():
	with pytest.raises(Exception):
		TaxpayerCreate(
			tenant_id="t1",
			taxpayer_type=TaxpayerType.INDIVIDUAL,
			tax_pin="A000000005E",
			national_id="22222",
			taxpayer_name="David",
			evidence_reference="ev",
			unknown_field="bad",
		)


# ---------------------------------------------------------------------------
# Tests: TaxpayerResponse
# ---------------------------------------------------------------------------

def test_taxpayer_response_has_id(taxpayer_response):
	assert taxpayer_response.id
	assert len(taxpayer_response.id) > 10


def test_taxpayer_response_default_status():
	tr = TaxpayerResponse(
		tenant_id="t1",
		taxpayer_type=TaxpayerType.NGO,
		tax_pin="P000000006F",
		taxpayer_name="Help Kenya NGO",
		tax_types=[],
		country_of_incorporation="KE",
		is_resident=True,
		evidence_reference="ev",
		status=TaxpayerStatus.PENDING,
	)
	assert tr.status == TaxpayerStatus.PENDING


def test_taxpayer_response_serializes_to_json(taxpayer_response):
	d = taxpayer_response.model_dump(mode="json")
	assert d["taxpayer_type"] == "company"
	assert d["status"] == "active"
	assert isinstance(d["created_at"], str)


# ---------------------------------------------------------------------------
# Tests: TaxReturnCreate / Response
# ---------------------------------------------------------------------------

def test_return_create_valid():
	rc = TaxReturnCreate(
		tenant_id="t1",
		taxpayer_id="tp_1",
		tax_pin="A000000007G",
		return_type=ReturnType.MONTHLY_VAT,
		tax_period_start=date(2025, 1, 1),
		tax_period_end=date(2025, 1, 31),
		gross_income=Decimal("500000"),
		allowable_deductions=Decimal("50000"),
		taxable_income=Decimal("450000"),
		tax_liability=Decimal("72000"),
		tax_credits=Decimal("0"),
		tax_paid=Decimal("72000"),
		net_tax_payable=Decimal("0"),
		evidence_reference="ret_ev_001",
	)
	assert rc.return_type == ReturnType.MONTHLY_VAT
	assert rc.net_tax_payable == Decimal("0")


def test_return_response_uuid():
	rr = TaxReturnResponse(
		tenant_id="t1",
		taxpayer_id="tp_1",
		tax_pin="A000000008H",
		return_type=ReturnType.ANNUAL_INCOME,
		tax_period_start=date(2025, 1, 1),
		tax_period_end=date(2025, 12, 31),
		gross_income=Decimal("1200000"),
		allowable_deductions=Decimal("200000"),
		taxable_income=Decimal("1000000"),
		tax_liability=Decimal("300000"),
		tax_credits=Decimal("0"),
		tax_paid=Decimal("300000"),
		net_tax_payable=Decimal("0"),
		status=ReturnStatus.FILED,
		evidence_reference="ev",
		is_amended=False,
	)
	assert rr.id.startswith("0") or len(rr.id) == 36  # UUID7 format


# ---------------------------------------------------------------------------
# Tests: TaxAssessmentCreate / Response
# ---------------------------------------------------------------------------

def test_assessment_create_valid():
	ac = TaxAssessmentCreate(
		tenant_id="t1",
		return_id="ret_001",
		taxpayer_id="tp_001",
		assessment_type=AssessmentType.AUDIT_ASSESSMENT,
		assessed_amount=Decimal("150000"),
		assessor_id="officer_1",
		assessment_date=date(2025, 3, 15),
		evidence_reference="ass_ev",
	)
	assert ac.assessed_amount == Decimal("150000")


def test_assessment_response_penalty_list():
	ar = TaxAssessmentResponse(
		tenant_id="t1",
		return_id="ret_001",
		taxpayer_id="tp_001",
		assessment_type=AssessmentType.BEST_JUDGEMENT,
		assessed_amount=Decimal("50000"),
		tax_liability_per_return=Decimal("0"),
		additional_tax=Decimal("50000"),
		assessor_id="officer_1",
		assessment_date=date(2025, 4, 1),
		evidence_reference="ev",
		status=AssessmentStatus.ISSUED,
	)
	assert ar.penalty_ids == []
	assert ar.interest_ids == []


# ---------------------------------------------------------------------------
# Tests: TaxPayment
# ---------------------------------------------------------------------------

def test_payment_create_valid():
	pc = TaxPaymentCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		payment_reference="PAY-2025-001",
		payment_method=PaymentMethod.MOBILE_MONEY,
		amount=Decimal("50000"),
		payment_date=date(2025, 4, 15),
		evidence_reference="pay_ev",
	)
	assert pc.amount == Decimal("50000")
	assert pc.payment_method == PaymentMethod.MOBILE_MONEY


# ---------------------------------------------------------------------------
# Tests: TaxDebt
# ---------------------------------------------------------------------------

def test_debt_create_valid():
	dc = TaxDebtCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		assessment_id="ass_001",
		principal_amount=Decimal("100000"),
		penalty_amount=Decimal("5000"),
		interest_amount=Decimal("1000"),
		due_date=date(2025, 5, 31),
	)
	assert dc.principal_amount == Decimal("100000")


def test_debt_response_balance():
	dr = TaxDebtResponse(
		tenant_id="t1",
		taxpayer_id="tp_001",
		assessment_id="ass_001",
		principal_amount=Decimal("100000"),
		penalty_amount=Decimal("5000"),
		interest_amount=Decimal("1000"),
		total_amount=Decimal("106000"),
		amount_paid=Decimal("50000"),
		balance=Decimal("56000"),
		due_date=date(2025, 5, 31),
		status=DebtStatus.PARTIALLY_PAID,
	)
	assert dr.balance == Decimal("56000")
	assert dr.status == DebtStatus.PARTIALLY_PAID


# ---------------------------------------------------------------------------
# Tests: TaxAudit / AuditFinding
# ---------------------------------------------------------------------------

def test_audit_create_valid():
	ac = TaxAuditCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		tax_pin="P000000009I",
		audit_type=AuditType.FIELD_AUDIT,
		auditor_id="auditor_1",
		audit_team=["auditor_1", "auditor_2"],
		tax_period_start=date(2024, 1, 1),
		tax_period_end=date(2024, 12, 31),
		evidence_reference="aud_ev",
	)
	assert ac.audit_type == AuditType.FIELD_AUDIT
	assert len(ac.audit_team) == 2


def test_finding_create_valid():
	fc = AuditFindingCreate(
		tenant_id="t1",
		audit_id="aud_001",
		taxpayer_id="tp_001",
		finding_type=FindingType.UNDERPAYMENT,
		description="Underpaid VAT for Q1 2024",
		additional_tax=Decimal("25000"),
		penalty_amount=Decimal("1250"),
		interest_amount=Decimal("500"),
		evidence_reference="find_ev",
	)
	assert fc.additional_tax == Decimal("25000")


def test_finding_response_total():
	fr = AuditFindingResponse(
		tenant_id="t1",
		audit_id="aud_001",
		taxpayer_id="tp_001",
		finding_type=FindingType.FRAUD,
		description="Fraudulent invoices",
		additional_tax=Decimal("100000"),
		penalty_amount=Decimal("100000"),
		interest_amount=Decimal("10000"),
		total_amount=Decimal("210000"),
		evidence_reference="ev",
	)
	assert fr.total_amount == Decimal("210000")


# ---------------------------------------------------------------------------
# Tests: Objection / Appeal
# ---------------------------------------------------------------------------

def test_objection_create_valid():
	oc = ObjectionCreate(
		tenant_id="t1",
		assessment_id="ass_001",
		taxpayer_id="tp_001",
		tax_pin="A000000010J",
		grounds="Double counting of expenses",
		amount_disputed=Decimal("30000"),
		evidence_reference="obj_ev",
	)
	assert oc.grounds == "Double counting of expenses"


def test_appeal_create_valid():
	ac = AppealCreate(
		tenant_id="t1",
		objection_id="obj_001",
		taxpayer_id="tp_001",
		grounds="KRA failed to consider all evidence",
		amount_in_dispute=Decimal("30000"),
		evidence_reference="app_ev",
	)
	assert ac.tribunal == "Tax Appeals Tribunal"


# ---------------------------------------------------------------------------
# Tests: TaxRefund
# ---------------------------------------------------------------------------

def test_refund_create_valid():
	rc = TaxRefundCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		tax_pin="A000000011K",
		return_id="ret_001",
		refund_type="input_vat_credit",
		claimed_amount=Decimal("45000"),
		bank_account_number="1234567890",
		bank_name="Equity Bank",
		evidence_reference="ref_ev",
	)
	assert rc.claimed_amount == Decimal("45000")


# ---------------------------------------------------------------------------
# Tests: Penalty / Interest
# ---------------------------------------------------------------------------

def test_penalty_create_valid():
	pc = PenaltyCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		penalty_type=PenaltyType.LATE_FILING,
		base_amount=Decimal("100000"),
		rate=Decimal("0.05"),
		calculated_amount=Decimal("5000"),
		period_days=15,
	)
	assert pc.calculated_amount == Decimal("5000")


def test_interest_create_valid():
	ic = InterestCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		interest_type=InterestType.LATE_PAYMENT,
		principal_amount=Decimal("100000"),
		annual_rate=Decimal("0.12"),
		from_date=date(2025, 1, 1),
		to_date=date(2025, 4, 1),
		calculated_amount=Decimal("3000"),
	)
	assert ic.interest_type == InterestType.LATE_PAYMENT


# ---------------------------------------------------------------------------
# Tests: TaxClearanceCertificate
# ---------------------------------------------------------------------------

def test_clearance_create_valid():
	cc = TaxClearanceCertificateCreate(
		tenant_id="t1",
		taxpayer_id="tp_001",
		tax_pin="A000000012L",
		purpose="government_tender",
		validity_months=6,
		evidence_reference="tcc_ev",
	)
	assert cc.purpose == "government_tender"
	assert cc.validity_months == 6


# ---------------------------------------------------------------------------
# Tests: Dashboard / Reporting models
# ---------------------------------------------------------------------------

def test_dashboard_kpi_valid():
	kpi = TaxDashboardKPI(
		tenant_id="t1",
		as_of=datetime.utcnow(),
		registered_taxpayers=100,
		active_taxpayers=90,
		returns_filed_ytd=450,
		returns_overdue=10,
		assessments_pending=5,
		total_tax_assessed=Decimal("10000000"),
		total_tax_collected=Decimal("9500000"),
		total_outstanding_debt=Decimal("500000"),
		open_objections=3,
		open_audits=7,
		pending_refunds=2,
		pending_clearance_certs=4,
		compliance_rate=Decimal("0.90"),
		collection_rate=Decimal("0.95"),
	)
	assert kpi.compliance_rate == Decimal("0.90")


def test_demand_notice_valid():
	dn = DemandNotice(
		tenant_id="t1",
		debt_id="debt_001",
		taxpayer_id="tp_001",
		tax_pin="A000000013M",
		amount_demanded=Decimal("150000"),
		due_date=date(2025, 6, 30),
		notice_number="DN-20250101-ABCDEF",
	)
	assert dn.notice_number.startswith("DN-")


def test_eoi_request_valid():
	eoi = EOIRequest(
		tenant_id="t1",
		treaty_partner="GB",
		subject_taxpayer_id="tp_001",
		subject_name="Global Corp Ltd",
		information_requested="account_balances",
	)
	assert eoi.urgency == "routine"
	assert eoi.legal_basis == "double_tax_agreement"


def test_compliance_risk_profile_valid():
	crp = ComplianceRiskProfile(
		taxpayer_id="tp_001",
		tax_pin="A000000014N",
		risk_score=Decimal("72.5"),
		risk_category="high",
		factors={"late_filings": 3, "open_audits": 1},
		recommended_action="schedule_field_audit",
	)
	assert crp.risk_category == "high"


def test_debt_aging_bucket_valid():
	bucket = DebtAgingBucket(
		bucket_label="31-90",
		taxpayer_count=15,
		total_amount=Decimal("750000"),
		principal=Decimal("600000"),
		penalty=Decimal("100000"),
		interest=Decimal("50000"),
	)
	assert bucket.taxpayer_count == 15


def test_uuid7str_generates_unique_ids():
	ids = {uuid7str() for _ in range(100)}
	assert len(ids) == 100
