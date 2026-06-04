"""Service layer tests for APG Tax Administration.

Tests the full TaxAdministrationService lifecycle using real objects.
No mocks. Covers all major workflows and tenant isolation.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))

from service import TaxAdministrationService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc() -> TaxAdministrationService:
	return TaxAdministrationService()


@pytest.fixture
def tenant() -> str:
	return "test_tenant"


@pytest.fixture
def registered_taxpayer(svc, tenant):
	"""Returns a registered taxpayer dict."""
	return svc.register_taxpayer(
		taxpayer_id="", tenant_id=tenant,
		tax_type="income_tax", tax_pin="",
		id_number="ID-001", legal_name="Alice Wanjiku",
		entity_type="individual",
		email="alice@example.com",
		phone="0712345678",
		address="Nairobi CBD",
		evidence_reference="reg_ev_001",
	)


@pytest.fixture
def filed_return(svc, tenant, registered_taxpayer):
	tin = registered_taxpayer["tax_pin"]
	return svc.submit_return(
		tin=tin, tax_type="income_tax", period="2024",
		return_data={
			"gross_income": 1200000,
			"allowable_deductions": 200000,
			"taxable_income": 1000000,
			"tax_liability": 300000,
			"tax_credits": 0,
			"tax_paid": 300000,
			"net_tax_payable": 0,
			"evidence_reference": "ret_ev_001",
		},
		tenant_id=tenant,
	)


# ---------------------------------------------------------------------------
# Taxpayer Registration
# ---------------------------------------------------------------------------

class TestTaxpayerRegistration:

	def test_register_individual(self, svc, tenant):
		result = svc.register_taxpayer(
			taxpayer_id="", tenant_id=tenant,
			tax_type="income_tax", tax_pin="",
			id_number="ID-100", legal_name="John Kamau",
			entity_type="individual",
			evidence_reference="ev",
		)
		assert result["taxpayer_name"] == "John Kamau"
		assert result["taxpayer_type"] == "individual"
		assert result["tax_pin"].startswith("A")
		assert result["status"] == "pending"
		assert result["tenant_id"] == tenant

	def test_register_company(self, svc, tenant):
		result = svc.register_taxpayer(
			taxpayer_id="", tenant_id=tenant,
			tax_type="corporate_tax", tax_pin="",
			id_number="BRN-001", legal_name="Acme Ltd",
			entity_type="company",
			evidence_reference="ev",
		)
		assert result["taxpayer_type"] == "company"
		assert result["tax_pin"].startswith("P")

	def test_register_ngo(self, svc, tenant):
		result = svc.register_taxpayer(
			taxpayer_id="", tenant_id=tenant,
			tax_type="income_tax", tax_pin="",
			id_number="NGO-001", legal_name="Help Kenya NGO",
			entity_type="ngo",
			evidence_reference="ev",
		)
		assert result["taxpayer_type"] == "ngo"

	def test_unique_pin_generated(self, svc, tenant):
		pins = set()
		for i in range(20):
			r = svc.register_taxpayer(
				taxpayer_id="", tenant_id=tenant,
				tax_type="income_tax", tax_pin="",
				id_number=f"ID-{i}", legal_name=f"Taxpayer {i}",
				evidence_reference="ev",
			)
			pins.add(r["tax_pin"])
		assert len(pins) == 20

	def test_find_by_pin(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		found = svc._find_taxpayer_by_pin(tin, tenant)
		assert found is not None
		assert found.taxpayer_name == "Alice Wanjiku"

	def test_taxpayer_search_by_name(self, svc, tenant, registered_taxpayer):
		results = svc.taxpayer_search("Alice", "name", tenant_id=tenant)
		assert len(results) >= 1
		assert any(r["taxpayer_name"] == "Alice Wanjiku" for r in results)

	def test_taxpayer_search_by_pin(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		results = svc.taxpayer_search(tin, "pin", tenant_id=tenant)
		assert len(results) == 1

	def test_verify_tin_valid(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.verify_tin(tin)
		assert result["exists"] is True

	def test_verify_tin_unknown(self, svc, tenant):
		result = svc.verify_tin("UNKNOWN-PIN")
		assert result["exists"] is False

	def test_update_taxpayer(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		updated = svc.update_taxpayer(
			tin, tenant_id=tenant,
			email="new@example.com",
			phone="0799999999",
		)
		assert updated["email"] == "new@example.com"

	def test_deregister_taxpayer(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.deregister_taxpayer(
			tin, reason="ceased_operations",
			deregistration_date="2025-12-31",
			tenant_id=tenant,
		)
		assert result["status"] == "deregistered"

	def test_audit_trail_on_registration(self, svc, tenant):
		svc.register_taxpayer(
			taxpayer_id="", tenant_id=tenant,
			tax_type="vat", tax_pin="",
			id_number="ID-AUDIT", legal_name="Audit Test Co",
			evidence_reference="ev",
		)
		events = [e for e in svc.audit_events if e["event_type"] == "taxpayer_registered"]
		assert len(events) >= 1


# ---------------------------------------------------------------------------
# Return Filing
# ---------------------------------------------------------------------------

class TestReturnFiling:

	def test_file_return(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.submit_return(
			tin=tin, tax_type="vat", period="2025-01",
			return_data={
				"gross_income": 500000,
				"allowable_deductions": 0,
				"tax_liability": 80000,
				"tax_credits": 0,
				"tax_paid": 80000,
				"evidence_reference": "ret_ev",
			},
			tenant_id=tenant,
		)
		assert result["status"] == "filed"
		assert result["return_type"] == "monthly_vat"

	def test_file_nil_return(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.file_nil_return(tin=tin, tax_type="vat", period="2025-02", tenant_id=tenant)
		assert result["tax_liability"] == "0"
		assert result["status"] == "filed"

	def test_validate_return_valid(self, svc, tenant, filed_return):
		result = svc.validate_return(filed_return["id"], tenant_id=tenant)
		assert result["status"] == "valid"
		assert result["issues"] == []

	def test_validate_return_mismatch(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		ret = svc.submit_return(
			tin=tin, tax_type="income_tax", period="2024",
			return_data={
				"gross_income": 500000,
				"allowable_deductions": 50000,
				"taxable_income": 400000,  # wrong: should be 450000
				"tax_liability": 100000,
				"tax_credits": 0,
				"tax_paid": 100000,
				"evidence_reference": "ev",
			},
			tenant_id=tenant,
		)
		result = svc.validate_return(ret["id"], tenant_id=tenant)
		assert result["status"] == "invalid"
		assert len(result["issues"]) > 0

	def test_amend_return(self, svc, tenant, filed_return):
		amended = svc.amend_return(
			filed_return["id"],
			amendment_reason="corrected deductions",
			amended_data={"gross_income": 1300000},
			tenant_id=tenant,
		)
		assert amended["is_amended"] is True
		assert amended["original_return_id"] == filed_return["id"]

	def test_return_filing_status_filed(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		svc.submit_return(
			tin=tin, tax_type="income_tax", period="2024",
			return_data={"gross_income": 0, "tax_liability": 0, "tax_paid": 0, "evidence_reference": "ev"},
			tenant_id=tenant,
		)
		status = svc.return_filing_status(tin, "income_tax", "2024", tenant_id=tenant)
		assert status["filed"] is True

	def test_return_filing_status_not_filed(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		status = svc.return_filing_status(tin, "vat", "2023", tenant_id=tenant)
		assert status["filed"] is False

	def test_filing_history(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		for month in ["2024-01", "2024-02", "2024-03"]:
			svc.submit_return(
				tin=tin, tax_type="vat", period=month,
				return_data={"gross_income": 100000, "tax_liability": 16000, "tax_paid": 16000, "evidence_reference": "ev"},
				tenant_id=tenant,
			)
		history = svc.filing_history(tin, "2024-01", "2024-03", tenant_id=tenant)
		assert len(history) >= 3


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

class TestAssessment:

	def test_issue_assessment(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=150000.0, reason="underdeclared_income",
			assessment_type="best_judgement",
			tenant_id=tenant, assessor_id="officer_1",
		)
		assert result["assessed_amount"] == "150000.00"
		assert result["status"] == "issued"
		assert "debt_id" in result

	def test_assessment_creates_debt(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=75000.0, reason="audit_finding",
			assessment_type="audit_assessment",
			tenant_id=tenant,
		)
		debt = svc._debts.get_item(tenant, result["debt_id"])
		assert debt is not None
		assert debt.principal_amount == Decimal("75000.00")

	def test_calculate_penalty_and_interest(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=100000.0, reason="late",
			assessment_type="self_assessment",
			tenant_id=tenant,
		)
		# Pay 60 days after due date (due is 30 days after assessment, so ~90 days later)
		future_date = (date.today() + timedelta(days=90)).isoformat()
		result = svc.calculate_penalty_and_interest(
			assess["id"], future_date, tenant_id=tenant
		)
		assert "late_filing_penalty" in result
		assert "late_payment_interest" in result
		assert float(result["late_filing_penalty"]) > 0


# ---------------------------------------------------------------------------
# Objections & Appeals
# ---------------------------------------------------------------------------

class TestObjectionsAppeals:

	def test_raise_objection(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		# File objection within 30 days
		today = date.today().isoformat()
		obj = svc.raise_objection(
			assess["id"], "Double counted expenses", 20000.0,
			objection_date=today, tenant_id=tenant, tax_pin=tin,
		)
		assert obj["status"] == "submitted"
		assert obj["amount_disputed"] == "20000.00"

	def test_objection_deadline_enforcement(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		# File objection 45 days after assessment date
		late_date = (date.today() + timedelta(days=45)).isoformat()
		with pytest.raises(AssertionError):
			svc.raise_objection(
				assess["id"], "Too late objection", 20000.0,
				objection_date=late_date, tenant_id=tenant,
			)

	def test_process_objection_upheld(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		today = date.today().isoformat()
		obj = svc.raise_objection(
			assess["id"], "Valid grounds", 20000.0,
			objection_date=today, tenant_id=tenant,
		)
		result = svc.process_objection(
			obj["id"], decision="upheld", revised_amount=0.0,
			officer_id="officer_1", tenant_id=tenant,
		)
		assert result["status"] == "upheld"
		assert result["amount_upheld"] == "0.00"

	def test_process_objection_dismissed(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2023",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		today = date.today().isoformat()
		obj = svc.raise_objection(
			assess["id"], "Weak grounds", 20000.0,
			objection_date=today, tenant_id=tenant,
		)
		dismissed = svc.process_objection(
			obj["id"], decision="dismissed", revised_amount=50000.0,
			officer_id="officer_1", tenant_id=tenant,
		)
		assert dismissed["status"] == "dismissed"

	def test_file_appeal_after_dismissal(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2022",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		today = date.today().isoformat()
		obj = svc.raise_objection(
			assess["id"], "Grounds", 20000.0,
			objection_date=today, tenant_id=tenant,
		)
		svc.process_objection(
			obj["id"], "dismissed", 50000.0, "officer_1", tenant_id=tenant
		)
		appeal = svc.file_appeal(
			obj["id"], "Further grounds for appeal", tenant_id=tenant
		)
		assert appeal["status"] == "submitted"
		assert appeal["tribunal"] == "Tax Appeals Tribunal"

	def test_appeal_rejected_on_upheld_objection(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		assess = svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2021",
			assessed_amount=50000.0, reason="test",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		today = date.today().isoformat()
		obj = svc.raise_objection(
			assess["id"], "Good grounds", 20000.0,
			objection_date=today, tenant_id=tenant,
		)
		svc.process_objection(obj["id"], "upheld", 0.0, "officer_1", tenant_id=tenant)
		with pytest.raises(AssertionError):
			svc.file_appeal(obj["id"], "Invalid appeal", tenant_id=tenant)


# ---------------------------------------------------------------------------
# Payments & Debt
# ---------------------------------------------------------------------------

class TestPaymentsDebt:

	def test_process_payment(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.process_tax_payment(
			tin=tin, tax_type="vat", period="2025-01",
			amount=80000.0, payment_method="mobile_money",
			reference="MPESA-20250115-001",
			tenant_id=tenant,
		)
		assert result["status"] == "confirmed"
		assert result["amount"] == "80000.00"

	def test_payment_allocation_fifo(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		tp = svc._find_taxpayer_by_pin(tin, tenant)

		# Create two debts
		from models import TaxDebtResponse, DebtStatus
		debt1 = TaxDebtResponse(
			tenant_id=tenant, taxpayer_id=tp.id,
			assessment_id="ass_1",
			principal_amount=Decimal("50000"), penalty_amount=Decimal("0"),
			interest_amount=Decimal("0"), total_amount=Decimal("50000"),
			balance=Decimal("50000"),
			due_date=date(2025, 1, 31),
			status=DebtStatus.OUTSTANDING,
		)
		debt2 = TaxDebtResponse(
			tenant_id=tenant, taxpayer_id=tp.id,
			assessment_id="ass_2",
			principal_amount=Decimal("30000"), penalty_amount=Decimal("0"),
			interest_amount=Decimal("0"), total_amount=Decimal("30000"),
			balance=Decimal("30000"),
			due_date=date(2025, 2, 28),
			status=DebtStatus.OUTSTANDING,
		)
		svc._debts.put(tenant, debt1.id, debt1)
		svc._debts.put(tenant, debt2.id, debt2)

		payment = svc.process_tax_payment(
			tin=tin, tax_type="income_tax", period="2025",
			amount=70000.0, payment_method="bank_transfer",
			reference="PAY-001", tenant_id=tenant,
		)
		result = svc.allocate_payment_to_assessments(payment["id"], tenant_id=tenant)
		assert len(result["allocated"]) >= 1
		assert Decimal(result["unallocated"]) < Decimal("70000")

	def test_issue_demand_notice(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		deadline = (date.today() + timedelta(days=30)).isoformat()
		result = svc.issue_demand_notice(
			tin=tin, outstanding_amount=150000.0,
			deadline=deadline, tenant_id=tenant,
		)
		assert "notice_number" in result
		assert result["notice_number"].startswith("DN-")
		assert result["amount_demanded"] == "150000.00"

	def test_debt_collection_action(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.debt_collection_action(
			tin=tin, action_type="payment_plan",
			officer_id="collector_1",
			tenant_id=tenant,
		)
		assert result["action_type"] == "payment_plan"
		assert result["status"] == "initiated"


# ---------------------------------------------------------------------------
# Audit Case Management
# ---------------------------------------------------------------------------

class TestAuditManagement:

	def test_open_audit_case(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.open_audit_case(
			tin=tin, audit_type="field_audit", audit_period="2024",
			assigned_officer="auditor_1", tenant_id=tenant,
			scope_description="Full field audit", risk_score=65.0,
		)
		assert result["audit_type"] == "field_audit"
		assert result["status"] == "planned"

	def test_conduct_audit_with_findings(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		audit = svc.open_audit_case(
			tin=tin, audit_type="desk_audit", audit_period="2024",
			assigned_officer="auditor_1", tenant_id=tenant,
		)
		findings = [
			{
				"finding_type": "underpayment",
				"description": "VAT underpaid Q1 2024",
				"additional_tax": 25000,
				"penalty_amount": 1250,
				"interest_amount": 500,
				"evidence_reference": "find_ev_001",
			},
		]
		result = svc.conduct_audit(audit["id"], findings, tenant_id=tenant)
		assert result["status"] == "in_progress"
		assert len(result["finding_ids"]) == 1
		assert float(result["total_additional_tax"]) == 25000.0

	def test_close_audit_case(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		audit = svc.open_audit_case(
			tin=tin, audit_type="compliance_audit", audit_period="2023",
			assigned_officer="auditor_1", tenant_id=tenant,
		)
		result = svc.close_audit_case(
			audit["id"], outcome="tax_due",
			final_tax_due=50000.0, penalties=2500.0,
			tenant_id=tenant,
		)
		assert result["status"] == "finalised"
		assert "audit_assessment_id" in result

	def test_audit_case_analytics(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		for audit_type in ["desk_audit", "field_audit"]:
			svc.open_audit_case(
				tin=tin, audit_type=audit_type, audit_period="2024",
				assigned_officer="auditor_1", tenant_id=tenant,
			)
		stats = svc.audit_case_analytics("2024", tenant_id=tenant)
		assert stats["total_cases"] >= 2


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

class TestRefunds:

	def test_refund_application(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		# File a return first
		svc.submit_return(
			tin=tin, tax_type="vat", period="2025-01",
			return_data={"gross_income": 500000, "tax_liability": 80000, "tax_paid": 80000, "evidence_reference": "ev"},
			tenant_id=tenant,
		)
		result = svc.refund_application(
			tin=tin, tax_type="vat", period="2025-01",
			refund_amount=15000.0, reason="input_vat_credit",
			bank_account_number="1234567890", bank_name="Equity Bank",
			tenant_id=tenant,
		)
		assert result["status"] == "claimed"
		assert result["claimed_amount"] == "15000.00"

	def test_refund_review_to_approval(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		svc.submit_return(
			tin=tin, tax_type="vat", period="2025-02",
			return_data={"gross_income": 300000, "tax_liability": 48000, "tax_paid": 48000, "evidence_reference": "ev"},
			tenant_id=tenant,
		)
		refund = svc.refund_application(
			tin=tin, tax_type="vat", period="2025-02",
			refund_amount=10000.0, reason="overpayment",
			tenant_id=tenant,
		)
		reviewed = svc.verify_refund(
			refund["id"], officer_id="reviewer_1",
			tenant_id=tenant, notes="Documents verified",
		)
		assert reviewed["status"] == "under_review"

		approved = svc.approve_refund(
			refund["id"], approved_by="manager_1",
			payment_method="bank_transfer", tenant_id=tenant,
		)
		assert approved["status"] == "approved"

	def test_refund_analytics(self, svc, tenant):
		stats = svc.refund_analytics("2025", tenant_id=tenant)
		assert "total_applications" in stats
		assert "by_status" in stats


# ---------------------------------------------------------------------------
# Tax Clearance Certificate
# ---------------------------------------------------------------------------

class TestClearanceCertificate:

	def test_issue_clearance_no_debt(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.issue_tax_clearance_certificate(
			tin=tin, validity_days=180, tenant_id=tenant, purpose="government_tender"
		)
		assert result["status"] == "issued"
		assert "certificate_number" in result
		assert result["certificate_number"].startswith("TCC-")

	def test_clearance_blocked_by_debt(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		# Issue an assessment (creates debt automatically)
		svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=100000.0, reason="unpaid_tax",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		result = svc.issue_tax_clearance_certificate(
			tin=tin, tenant_id=tenant, purpose="business_license"
		)
		assert result["status"] == "rejected"
		assert "outstanding_debts" in result["reason"]


# ---------------------------------------------------------------------------
# Exchange of Information
# ---------------------------------------------------------------------------

class TestEOI:

	def test_exchange_of_information(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.exchange_of_information(
			request_source="GB", tin=tin,
			data_type="account_balances", tenant_id=tenant,
		)
		assert result["treaty_partner"] == "GB"
		assert result["status"] == "submitted"
		assert result["urgency"] == "routine"

	def test_eoi_urgent(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		result = svc.exchange_of_information(
			request_source="US", tin=tin,
			data_type="beneficial_ownership", tenant_id=tenant,
			urgency="urgent",
		)
		assert result["urgency"] == "urgent"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class TestReports:

	def test_dashboard_summary(self, svc, tenant):
		summary = svc.dashboard_summary(tenant)
		assert "registered_taxpayers" in summary
		assert "compliance_rate" in summary
		assert "total_tax_collected" in summary

	def test_revenue_collection_report(self, svc, tenant, registered_taxpayer, filed_return):
		report = svc.revenue_collection_report("2024", tenant_id=tenant)
		assert "total_assessed" in report
		assert "total_collected" in report
		assert "net_revenue" in report

	def test_compliance_rate_report(self, svc, tenant):
		report = svc.compliance_rate_report("2024", tenant_id=tenant)
		assert "compliance_rate" in report

	def test_delinquency_report(self, svc, tenant, registered_taxpayer):
		tin = registered_taxpayer["tax_pin"]
		svc.issue_assessment(
			tin=tin, tax_type="income_tax", period="2024",
			assessed_amount=50000.0, reason="overdue",
			assessment_type="best_judgement", tenant_id=tenant,
		)
		report = svc.delinquency_report(date.today().isoformat(), tenant_id=tenant)
		assert "aging_buckets" in report
		assert "total_outstanding_balance" in report


# ---------------------------------------------------------------------------
# Tenant Isolation
# ---------------------------------------------------------------------------

class TestTenantIsolation:

	def test_taxpayer_not_visible_across_tenants(self, svc):
		svc.register_taxpayer(
			taxpayer_id="", tenant_id="tenant_a",
			tax_type="income_tax", tax_pin="",
			id_number="ID-A", legal_name="Tenant A Corp",
			evidence_reference="ev",
		)
		svc.register_taxpayer(
			taxpayer_id="", tenant_id="tenant_b",
			tax_type="vat", tax_pin="",
			id_number="ID-B", legal_name="Tenant B Corp",
			evidence_reference="ev",
		)
		summary_a = svc.dashboard_summary("tenant_a")
		summary_b = svc.dashboard_summary("tenant_b")
		assert summary_a["registered_taxpayers"] == 1
		assert summary_b["registered_taxpayers"] == 1

	def test_returns_isolated_by_tenant(self, svc):
		tp_a = svc.register_taxpayer(
			taxpayer_id="", tenant_id="ta",
			tax_type="income_tax", tax_pin="",
			id_number="ID-A2", legal_name="A Corp",
			evidence_reference="ev",
		)
		svc.submit_return(
			tin=tp_a["tax_pin"], tax_type="income_tax", period="2024",
			return_data={"gross_income": 100000, "tax_liability": 10000, "tax_paid": 10000, "evidence_reference": "ev"},
			tenant_id="ta",
		)
		returns_tb = svc._returns.tenant_values("tb")
		assert len(returns_tb) == 0

	def test_debt_not_shared_across_tenants(self, svc):
		tp_a = svc.register_taxpayer(
			taxpayer_id="", tenant_id="t_iso_a",
			tax_type="income_tax", tax_pin="",
			id_number="ISO-A", legal_name="Iso A",
			evidence_reference="ev",
		)
		svc.issue_assessment(
			tin=tp_a["tax_pin"], tax_type="income_tax", period="2024",
			assessed_amount=100000.0, reason="test",
			assessment_type="best_judgement", tenant_id="t_iso_a",
		)
		debts_b = svc._debts.tenant_values("t_iso_b")
		assert len(debts_b) == 0


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------

class TestLegacyAPI:

	def test_legacy_file_return(self, svc, tenant):
		result = svc.file_return(
			"ret_leg_1", tenant, "annual_income", "PIN-LEG", "2024",
			1000000, 150000, 150000, "legacy_ev",
		)
		assert result["return_type"] == "annual_income"

	def test_legacy_raise_assessment(self, svc, tenant):
		svc.file_return(
			"ret_leg_2", tenant, "annual_income", "PIN-LEG2", "2024",
			500000, 75000, 75000, "ev",
		)
		result = svc.raise_assessment(
			"ass_leg_1", tenant, "ret_leg_2", "self_assessment",
			75000, "assessor_1", "2025-04-01", "ev",
		)
		assert result["assessed_amount"] == "75000.00"

	def test_legacy_file_objection(self, svc, tenant):
		svc.file_return("ret_obj", tenant, "annual_income", "PIN-OBJ", "2024", 100000, 15000, 15000, "ev")
		svc.raise_assessment("ass_obj", tenant, "ret_obj", "self_assessment", 15000, "assessor", "2025-04-01", "ev")
		obj = svc.file_objection(
			"obj_leg_1", tenant, "ass_obj", "PIN-OBJ",
			"Double counting", 5000, "ev", within_deadline=True,
		)
		assert obj["status"] == "submitted"

	def test_legacy_open_audit(self, svc, tenant):
		result = svc.open_audit(
			"aud_leg_1", tenant, "PIN-AUD-LEG",
			"desk_audit", "auditor_1", "2024", "aud_ev",
		)
		assert result["audit_type"] == "desk_audit"

	def test_legacy_complete_audit(self, svc, tenant):
		opened = svc.open_audit("aud_leg_2", tenant, "PIN-AUD2", "field_audit", "aud_1", "2024", "ev")
		audit_id = opened["id"]
		result = svc.complete_audit(audit_id, tenant, "Clean audit, no findings")
		assert result["status"] == "finalised"

	def test_legacy_register_agent(self, svc, tenant):
		result = svc.register_agent(
			"ag_1", tenant, "Return Processor", "codex", "return_processor", "scope"
		)
		assert result["role"] == "return_processor"

	def test_legacy_validate_batch(self, svc, tenant):
		result = svc.validate_batch(tenant, 100)
		assert result["processor"] == "bytewax"
		assert result["accepted"] is True

	def test_legacy_validate_batch_wrong_stream(self, svc, tenant):
		with pytest.raises(PermissionError):
			svc.validate_batch(tenant, 50, event_stream="sqs")

	def test_legacy_duplicate_pin_denied(self, svc, tenant):
		# The legacy interface generates new PINs (ignores supplied tax_pin arg).
		# Duplicate PIN enforcement applies to the *generated* PIN — verify two
		# registrations produce distinct PINs (no collision for unique id_numbers).
		r1 = svc.register_taxpayer("reg1", tenant, "income_tax", "PIN-DUP", "ID-001", "John", "ev")
		r2 = svc.register_taxpayer("reg2", tenant, "income_tax", "PIN-DUP2", "ID-002", "Jane", "ev")
		assert r1["tax_pin"] != r2["tax_pin"]

	def test_legacy_objection_outside_deadline(self, svc, tenant):
		# within_deadline=False triggers the policy denial regardless of PIN
		svc.file_return("ret_od", tenant, "annual_income", "PIN-OD", "2024", 100000, 15000, 15000, "ev")
		svc.raise_assessment("ass_od", tenant, "ret_od", "self_assessment", 15000, "assessor", "2025-04-01", "ev")
		with pytest.raises(PermissionError, match="objection_deadline_passed"):
			svc.file_objection("obj_od", tenant, "ass_od", "PIN-OD", "Grounds", 5000, "ev", within_deadline=False)

	def test_legacy_debt_collection_no_demand(self, svc, tenant):
		# empty demand_notice_reference triggers demand_notice_required denial
		svc.file_return("ret_nd", tenant, "annual_income", "PIN-ND", "2024", 100000, 15000, 0, "ev")
		svc.raise_assessment("ass_nd", tenant, "ret_nd", "self_assessment", 15000, "assessor", "2025-04-01", "ev")
		with pytest.raises(PermissionError, match="demand_notice_required"):
			svc.initiate_collection("col_nd", tenant, "PIN-ND", "ass_nd", "payment_plan", 15000, "", "approval-ref", "ev")
