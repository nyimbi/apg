"""
Service-layer tests for APG Digital Lending.

Run: cd capabilities/fintech/lending && python -m pytest tests/test_service.py -vxs
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Fixtures (also defined in conftest.py — duplicated here for standalone run)
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
	from service import LendingService
	return LendingService()


@pytest.fixture
def tenant_id():
	return "test_tenant"


@pytest.fixture
def seeded(svc, tenant_id):
	svc.register_product(
		product_id="TERM01", tenant_id=tenant_id, name="Term Loan", owner_id="admin",
		product_type="term_loan", currency="KES",
		min_amount=5_000, max_amount=1_000_000,
		min_term_days=30, max_term_days=1_800,
		annual_rate=0.18, repayment_frequency="monthly",
	)
	svc.onboard_borrower(
		borrower_id="B001", tenant_id=tenant_id, customer_reference="CUST001",
		kyc_profile_id="KYC001", country="KE",
		income_evidence_id="INC001", consent_reference="CONSENT001",
	)
	return svc


@pytest.fixture
def app_id(seeded, tenant_id):
	r = seeded.submit_application(
		application_id="APP001", tenant_id=tenant_id,
		borrower_id="B001", product_id="TERM01",
		requested_amount=100_000, purpose="working_capital",
		affordability_reference="AFF001",
		bank_statement_reference="BS001",
		aml_reference="AML001", fraud_reference="FRAUD001",
		behavior_evidence_reference="BEH001", human_review="UW1",
	)
	return r["id"]


@pytest.fixture
def approved_app_id(seeded, app_id):
	seeded.underwriting_decision(app_id, "approve", [], "UW001")
	return app_id


@pytest.fixture
def loan_id(seeded, approved_app_id):
	r = seeded.disburse_loan(
		loan_id="LOAN001",
		application_id=approved_app_id,
		bank_account="KE0001234567",
		disbursement_date=(date.today() - timedelta(days=90)).isoformat(),
	)
	return r["loan_id"]


# ---------------------------------------------------------------------------
# Product management
# ---------------------------------------------------------------------------

def test_register_product(seeded, tenant_id):
	products = seeded.list_products()
	assert len(products) == 1
	assert products[0]["id"] == "TERM01"


def test_register_duplicate_product_raises(seeded, tenant_id):
	with pytest.raises(ValueError, match="already exists"):
		seeded.register_product(
			product_id="TERM01", tenant_id=tenant_id, name="Dup", owner_id="a",
			product_type="term_loan", currency="KES",
			min_amount=1_000, max_amount=500_000,
			min_term_days=30, max_term_days=360,
			annual_rate=0.18, repayment_frequency="monthly",
		)


def test_update_product_rates(seeded):
	result = seeded.update_product_rates("TERM01", {"annual_rate": 0.22}, "2026-01-01")
	assert result["new_annual_rate"] == 0.22


def test_product_performance_report(seeded):
	report = seeded.product_performance_report("TERM01", "2026-06")
	assert "total_applications" in report
	assert report["product_code"] == "TERM01"


# ---------------------------------------------------------------------------
# Borrower & application
# ---------------------------------------------------------------------------

def test_onboard_borrower(seeded, tenant_id):
	seeded.onboard_borrower(
		borrower_id="B002", tenant_id=tenant_id, customer_reference="CUST002",
		kyc_profile_id="KYC002", country="KE",
		income_evidence_id="INC002", consent_reference="CON002",
	)
	assert "B002" in seeded.borrowers


def test_submit_application(app_id, seeded):
	apps = seeded.list_applications(tenant_id="test_tenant")
	assert len(apps) == 1
	assert apps[0]["id"] == app_id


def test_retrieve_application(seeded, app_id):
	detail = seeded.retrieve_application(app_id)
	assert detail["id"] == app_id
	assert "required_documents" in detail


def test_withdraw_application(seeded, app_id):
	result = seeded.withdraw_application(app_id, "borrower_changed_mind")
	assert result["status"] == "withdrawn"


def test_assign_underwriter(seeded, app_id):
	result = seeded.assign_underwriter(app_id, "UW_BETA")
	assert result["underwriter_id"] == "UW_BETA"
	assert result["status"] == "under_review"


def test_request_documents(seeded, app_id):
	result = seeded.request_documents(app_id, ["bank_statement", "payslip"])
	assert "bank_statement" in result["requested_documents"]


def test_application_analytics(seeded, app_id):
	analytics = seeded.application_analytics("2026-06")
	assert analytics["total_applications"] >= 1
	assert "by_status" in analytics


# ---------------------------------------------------------------------------
# Credit assessment
# ---------------------------------------------------------------------------

def test_credit_bureau_check(seeded):
	result = seeded.credit_bureau_check("CUST001", "12345678", "KE")
	assert "score" in result
	assert result["country"] == "KE"
	assert result["bureau_name"] != ""


def test_income_verification_employed(seeded):
	result = seeded.income_verification(
		"CUST001", "employed", 80_000,
		["payslip_jan2026.pdf", "bank_statement.pdf"]
	)
	assert result["verified"] is True
	assert result["confidence"] >= 0.85


def test_income_verification_mobile_money(seeded):
	result = seeded.income_verification(
		"CUST001", "mobile_money", 50_000,
		["mpesa_statement.pdf"]
	)
	assert result["verified"] is True
	assert result["verified_amount"] < 50_000  # haircut applied


def test_income_verification_no_docs(seeded):
	result = seeded.income_verification("CUST001", "employed", 80_000, ["generic.pdf"])
	assert result["verified"] is False


def test_credit_score_calculate(seeded, loan_id):
	result = seeded.credit_score_calculate("CUST001")
	assert 300 <= result["score"] <= 850
	assert result["risk_grade"] in ("A", "B", "C", "D", "E", "F")
	assert 0 < result["probability_of_default"] < 1


def test_dsr_calculation(seeded, loan_id):
	seeded.income_verification("CUST001", "employed", 100_000, ["payslip.pdf"])
	result = seeded.debt_service_ratio("CUST001", 50_000, 0.18, 12)
	assert "dsr" in result
	assert result["new_emi"] > 0


def test_loan_eligibility(seeded):
	seeded.income_verification("CUST001", "employed", 150_000, ["payslip.pdf"])
	seeded.credit_score_calculate("CUST001")
	result = seeded.calculate_loan_eligibility("CUST001", "TERM01")
	assert "max_amount" in result
	assert "indicative_annual_rate" in result


def test_generate_loan_offers(seeded, approved_app_id):
	seeded.income_verification("CUST001", "employed", 150_000, ["payslip.pdf"])
	seeded.credit_score_calculate("CUST001")
	offers = seeded.generate_loan_offers(approved_app_id)
	assert len(offers) >= 2
	tiers = {o["tier"] for o in offers}
	assert "conservative" in tiers
	assert "standard" in tiers


# ---------------------------------------------------------------------------
# Loan lifecycle
# ---------------------------------------------------------------------------

def test_disburse_loan(seeded, approved_app_id):
	result = seeded.disburse_loan(
		"LOAN_X", approved_app_id, "KE0001111", date.today().isoformat()
	)
	assert result["loan_id"] == "LOAN_X"
	assert "schedule_summary" in result


def test_generate_repayment_schedule(seeded, loan_id):
	sched = seeded.generate_repayment_schedule(loan_id)
	assert sched["principal"] > 0
	assert len(sched["installments"]) > 0
	# All installments should have required keys
	for inst in sched["installments"]:
		assert "emi" in inst
		assert "principal" in inst
		assert "interest" in inst


def test_repayment_schedule_flat_rate(seeded, loan_id):
	sched = seeded.generate_repayment_schedule(loan_id, "flat_rate")
	assert sched["schedule_type"] == "flat_rate"


def test_process_repayment(seeded, loan_id):
	result = seeded.process_repayment(
		loan_id=loan_id,
		amount=10_000,
		payment_date=date.today().isoformat(),
		payment_method="mobile_money",
		reference="REF001",
	)
	assert result["payment_amount"] == 10_000
	assert result["principal_cleared"] > 0 or result["interest_cleared"] > 0
	assert result["outstanding_principal"] < 100_000


def test_early_settlement(seeded, loan_id):
	result = seeded.early_settlement(
		loan_id, (date.today() + timedelta(days=1)).isoformat()
	)
	assert result["total_settlement_amount"] > 0
	assert result["outstanding_principal"] > 0
	assert result["early_settlement_fee"] > 0


def test_close_settled_loan_fails_with_balance(seeded, loan_id):
	with pytest.raises(ValueError, match="outstanding principal"):
		seeded.close_loan(loan_id, "settled")


def test_close_loan_cancelled(seeded, loan_id):
	result = seeded.close_loan(loan_id, "cancelled")
	assert result["status"] == "closed"


def test_add_fee(seeded, loan_id):
	result = seeded.add_loan_fee(loan_id, "late_payment_penalty", 500, "30 DPD penalty")
	assert result["fee"]["amount"] == 500
	assert result["total_outstanding_fees"] >= 500


def test_waive_fee(seeded, loan_id):
	r = seeded.add_loan_fee(loan_id, "late_payment_penalty", 500, "30 DPD")
	fee_id = r["fee"]["fee_id"]
	waive_result = seeded.waive_fee_or_penalty(loan_id, fee_id, "financial hardship", "manager")
	assert waive_result["waived_amount"] == 500


def test_restructure_loan(seeded, loan_id):
	result = seeded.restructure_loan(
		loan_id,
		{"annual_rate": 0.15, "tenor_months": 24},
		"COVID hardship restructure",
		"credit_committee",
	)
	assert result["restructure_record"]["new_terms"]["annual_rate"] == 0.15
	assert result["new_outstanding_principal"] > 0


def test_restructure_sets_new_schedule(seeded, loan_id):
	seeded.restructure_loan(loan_id, {"tenor_months": 36}, "extension", "mgr")
	loan = seeded._require_loan(loan_id)
	assert loan.tenor_months == 36
	assert len(loan.installments) > 0


def test_write_off_loan(seeded, loan_id):
	result = seeded.write_off_loan(
		loan_id, "non_performing",
		(date.today() - timedelta(days=1)).isoformat(),
		"credit_committee",
	)
	assert result["write_off_amount"] > 0
	loan = seeded._require_loan(loan_id)
	assert loan.status == "written_off"


def test_loan_statement(seeded, loan_id):
	seeded.process_repayment(loan_id, 5_000, date.today().isoformat(), "mpesa", "REF_STMT")
	stmt = seeded.get_loan_statement(loan_id)
	assert stmt["loan_id"] == loan_id
	assert len(stmt["repayments"]) >= 1
	assert "installments" in stmt
	assert stmt["total_repaid"] >= 5_000


# ---------------------------------------------------------------------------
# DPD & Collections
# ---------------------------------------------------------------------------

def test_calculate_dpd(seeded, loan_id):
	result = seeded.calculate_dpd(loan_id)
	assert result["loan_id"] == loan_id
	assert "max_dpd" in result
	assert result["max_dpd"] >= 0
	assert result["delinquency_bucket"] in ("current", "1-30", "31-60", "61-90", "91-120", "120+")


def test_demand_notice_level_1(seeded, loan_id):
	notice = seeded.generate_demand_notice(loan_id, 1)
	assert notice["level"] == 1
	assert "notice_text" in notice


def test_assign_collector(seeded, loan_id):
	result = seeded.assign_to_collector(loan_id, "COLLECTOR001")
	assert result["collector_id"] == "COLLECTOR001"


def test_collection_activity(seeded, loan_id):
	seeded.assign_to_collector(loan_id, "COLL001")
	result = seeded.record_collection_activity(
		loan_id, "call", "promise_to_pay",
		"Borrower promised to pay by end of week", "follow_up_friday"
	)
	assert result["activity"]["outcome"] == "promise_to_pay"
	assert result["total_activities"] == 1


def test_legal_action(seeded, loan_id):
	result = seeded.legal_action(loan_id, "file_suit", "LAWYER001")
	assert result["legal_action"]["action_type"] == "file_suit"


def test_delinquency_report(seeded, loan_id):
	report = seeded.delinquency_report()
	assert "npl_ratio" in report
	assert "buckets" in report
	assert "par_30" in report


# ---------------------------------------------------------------------------
# Collateral
# ---------------------------------------------------------------------------

def test_assess_collateral(seeded):
	result = seeded.assess_collateral([
		{"type": "property", "market_value": 1_000_000, "description": "house"},
		{"type": "vehicle",  "market_value": 200_000,   "description": "car"},
	])
	assert result["total_market_value"] == 1_200_000
	assert result["total_collateral_value"] < 1_200_000  # haircuts applied
	assert len(result["items"]) == 2


def test_collateral_coverage_ratio(seeded):
	result = seeded.assess_collateral([
		{"type": "cash", "market_value": 200_000, "description": "FD", "requested_amount": 100_000},
	])
	# Cash haircut 90%: FSV = 180k, requested = 100k → coverage 1.8x
	assert result["sufficient"] is True


# ---------------------------------------------------------------------------
# Portfolio analytics
# ---------------------------------------------------------------------------

def test_portfolio_summary(seeded, loan_id):
	summary = seeded.portfolio_summary()
	assert summary["total_active_loans"] == 1
	assert summary["total_book"] > 0
	assert 0 <= summary["par_30"] <= 1


def test_provision_calculation(seeded, loan_id):
	result = seeded.provision_calculation("ifrs9")
	assert result["method"] == "ifrs9"
	assert "stage1" in result
	assert "total_ecl" in result
	assert result["total_ecl"] >= 0


def test_vintage_analysis(seeded, loan_id):
	result = seeded.vintage_analysis(12)
	assert result["cohort_months"] == 12
	assert isinstance(result["cohorts"], list)


def test_concentration_risk(seeded, loan_id):
	result = seeded.concentration_risk_report()
	assert "by_sector" in result
	assert "by_geography" in result
	assert "by_ticket_size" in result


def test_stress_test(seeded, loan_id):
	scenarios = [
		{"name": "mild",   "additional_default_rate": 0.05, "lgd": 0.40},
		{"name": "severe", "additional_default_rate": 0.30, "lgd": 0.45},
	]
	result = seeded.stress_test(scenarios)
	assert len(result["scenarios"]) == 2
	mild = next(s for s in result["scenarios"] if s["scenario"] == "mild")
	severe = next(s for s in result["scenarios"] if s["scenario"] == "severe")
	assert severe["incremental_loss"] > mild["incremental_loss"]


def test_collection_performance_report(seeded, loan_id):
	seeded.assign_to_collector(loan_id, "COLL001")
	result = seeded.collection_performance_report("2026-06", "COLL001")
	assert "recovery_rate" in result
	assert result["total_loans_assigned"] >= 1
