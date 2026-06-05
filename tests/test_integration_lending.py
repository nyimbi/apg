"""Fintech lending capability integration tests: LendingService.

All tests are sync (service methods are synchronous).
Uses in-memory dicts — zero config.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Lending service falls back to bare imports when loaded outside its package;
# inject the package directory so the fallback bare imports resolve.
_LENDING_DIR = os.path.join(os.path.dirname(__file__), "..", "capabilities", "fintech", "lending")
if _LENDING_DIR not in sys.path:
	sys.path.insert(0, _LENDING_DIR)

from datetime import date, timedelta


# ── helpers ───────────────────────────────────────────────────────────────────

def _svc():
	from capabilities.fintech.lending.service import LendingService
	return LendingService()


def _register_product(svc, tenant_id: str = "lend-tenant", product_id: str = "prod-001"):
	return svc.register_product(
		product_id=product_id,
		tenant_id=tenant_id,
		name="Personal Loan",
		owner_id="owner-001",
		product_type="term_loan",
		currency="KES",
		min_amount=10_000,
		max_amount=500_000,
		min_term_days=30,
		max_term_days=365,
		annual_rate=0.18,
		repayment_frequency="monthly",
	)


def _onboard_borrower(svc, tenant_id: str = "lend-tenant", borrower_id: str = "borrower-001"):
	return svc.onboard_borrower(
		borrower_id=borrower_id,
		tenant_id=tenant_id,
		customer_reference="cust-001",
		kyc_profile_id="kyc-001",
		country="KE",
		income_evidence_id="inc-001",
		consent_reference="consent-001",
	)


# ── 1. loan product creation ──────────────────────────────────────────────────

def test_lending_loan_product():
	"""create_loan_product (register_product) returns a product dict."""
	svc = _svc()
	product = _register_product(svc)
	assert isinstance(product, dict)
	# LoanProduct.to_dict() uses "id" key for the product identifier
	assert product.get("id") == "prod-001" or product.get("product_id") == "prod-001"
	assert product["name"] == "Personal Loan"
	assert product["currency"] == "KES"
	assert product["tenant_id"] == "lend-tenant"


# ── 2. loan eligibility ───────────────────────────────────────────────────────

def test_lending_eligibility():
	"""calculate_loan_eligibility returns a dict with a max_amount key."""
	svc = _svc()
	_register_product(svc)
	_onboard_borrower(svc)

	# Seed income verification so eligibility calculation has income data
	svc.income_verification(
		customer_id="cust-001",
		income_source="employed",
		stated_amount=80_000,
		docs=["payslip_jan2026.pdf"],
	)
	# Compute credit score first
	svc.credit_score_calculate("cust-001")

	result = svc.calculate_loan_eligibility("cust-001", "prod-001")
	assert isinstance(result, dict)
	assert "max_amount" in result
	assert isinstance(result["max_amount"], float)


# ── 3. credit score calculation ──────────────────────────────────────────────

def test_lending_credit_score():
	"""credit_score_calculate returns a score between 300 and 850."""
	svc = _svc()
	_onboard_borrower(svc)

	result = svc.credit_score_calculate("cust-001")
	assert isinstance(result, dict)
	assert "score" in result
	score = result["score"]
	assert 300 <= score <= 850, f"score {score} out of range 300–850"
	assert "risk_grade" in result
	assert result["risk_grade"] in ("A", "B", "C", "D", "E", "F")


# ── 4. amortisation / repayment schedule ─────────────────────────────────────

def test_lending_amortisation():
	"""generate_repayment_schedule returns a dict with a non-empty installments list."""
	svc = _svc()
	_register_product(svc)
	_onboard_borrower(svc)

	# Need to create a loan to generate its schedule.
	# Submit application → approve → disburse → then call generate_repayment_schedule.
	app = svc.submit_application(
		application_id="app-001",
		tenant_id="lend-tenant",
		borrower_id="borrower-001",
		product_id="prod-001",
		requested_amount=100_000,
		purpose="working_capital",
		affordability_reference="afford-001",
		bank_statement_reference="bank-stmt-001",
		aml_reference="aml-001",
		fraud_reference="fraud-001",
		behavior_evidence_reference="behav-001",
		human_review="hr-001",
	)
	# Approve via underwriting_decision (lightweight path)
	svc.underwriting_decision("app-001", "approve", [], "underwriter-001")

	loan = svc.disburse_loan(
		loan_id="loan-001",
		application_id="app-001",
		bank_account="KE123456789",
		disbursement_date=date.today().isoformat(),
	)
	assert loan["loan_id"] == "loan-001"

	schedule = svc.generate_repayment_schedule("loan-001")
	assert isinstance(schedule, dict)
	assert "installments" in schedule
	assert isinstance(schedule["installments"], list)
	assert len(schedule["installments"]) > 0


# ── 5. delinquency / collection queue ────────────────────────────────────────

def test_lending_collection_queue():
	"""delinquency_report returns a dict with PAR metrics."""
	svc = _svc()
	result = svc.delinquency_report()
	assert isinstance(result, dict)
	assert "par_30" in result
	assert "npl_ratio" in result
	assert "total_outstanding" in result
	assert isinstance(result["par_30"], float)


# ── 6. rule evaluation — deny missing tenant ─────────────────────────────────

def test_lending_rule_evaluation():
	"""evaluate_rules('fintech_lending', {}) denies when tenant context is absent."""
	from capabilities.fintech.lending.capability_contract import evaluate_capability_rules

	deny_result = evaluate_capability_rules({"tenant_context_present": False})
	assert deny_result["decision"] == "deny"

	allow_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation_type": "read",
		"policy_attached": True,
	})
	assert allow_result["decision"] == "allow"


# ── 7. manifest — service methods ────────────────────────────────────────────

def test_lending_manifest_navigation():
	"""get_capability('fintech_lending') exposes 60+ service methods."""
	from capabilities.manifest import get_capability
	cap = get_capability("fintech_lending")
	assert cap is not None, "fintech_lending not found in manifest"
	methods = cap.get("service_methods", [])
	assert len(methods) >= 60, f"expected ≥60 service methods, got {len(methods)}"
