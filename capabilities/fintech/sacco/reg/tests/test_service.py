"""Service tests for SACCARegulatoryService."""
from __future__ import annotations

from decimal import Decimal

import pytest

from capabilities.fintech.sacco.reg.models import (
	FilingStatus,
	LoanClassificationBand,
	ReturnType,
	TrafficLight,
)
from capabilities.fintech.sacco.reg.service import (
	SACCARegulatoryService,
	_CAR_MIN,
	_LIQUIDITY_MIN,
	_LDR_MAX,
)

TENANT = "test-sacco"


# ── Capital Adequacy ──────────────────────────────────────────────────────────

async def test_capital_adequacy_compliant(svc):
	result = await svc.calculate_capital_adequacy(TENANT, "2025-03-31")
	assert result.compliant is True
	assert result.capital_adequacy_ratio >= _CAR_MIN
	assert result.traffic_light in {TrafficLight.GREEN, TrafficLight.AMBER}
	assert result.shortfall == Decimal("0")


async def test_capital_adequacy_breach(svc_breach):
	result = await svc_breach.calculate_capital_adequacy(TENANT, "2025-03-31")
	assert result.compliant is False
	assert result.capital_adequacy_ratio < _CAR_MIN
	assert result.traffic_light == TrafficLight.RED
	assert result.shortfall > Decimal("0")


async def test_capital_adequacy_fields_populated(svc):
	result = await svc.calculate_capital_adequacy(TENANT, "2025-03-31")
	assert result.core_capital > 0
	assert result.secondary_capital >= 0
	assert result.institutional_capital == result.core_capital + result.secondary_capital
	assert result.risk_weighted_assets > 0
	assert result.total_assets > 0


# ── Liquidity ─────────────────────────────────────────────────────────────────

async def test_liquidity_compliant(svc):
	result = await svc.calculate_liquidity_ratio(TENANT, "2025-03-31")
	assert result.compliant is True
	assert result.liquidity_ratio >= _LIQUIDITY_MIN
	assert result.shortfall == Decimal("0")


async def test_liquidity_breach(svc_breach):
	result = await svc_breach.calculate_liquidity_ratio(TENANT, "2025-03-31")
	assert result.compliant is False
	assert result.traffic_light == TrafficLight.RED


async def test_liquidity_components(svc):
	result = await svc.calculate_liquidity_ratio(TENANT, "2025-03-31")
	assert result.total_liquid_assets == (
		result.cash_on_hand + result.bank_balances
		+ result.government_securities + result.other_liquid_assets
	)
	assert result.total_deposits_and_borrowings == result.total_deposits + result.total_borrowings


# ── Loan to Deposit Ratio ─────────────────────────────────────────────────────

async def test_ldr_compliant(svc):
	ldr = await svc.calculate_loan_to_deposit_ratio(TENANT, "2025-03-31")
	assert ldr <= _LDR_MAX


async def test_ldr_breach(svc_breach):
	ldr = await svc_breach.calculate_loan_to_deposit_ratio(TENANT, "2025-03-31")
	assert ldr > _LDR_MAX


# ── Loan Classification ───────────────────────────────────────────────────────

async def test_loan_classification_bands(svc):
	lc = await svc.classify_loan_portfolio(TENANT, "2025-03-31")
	band_names = {b.band for b in lc.bands}
	assert LoanClassificationBand.NORMAL in band_names
	assert LoanClassificationBand.WATCH in band_names
	assert LoanClassificationBand.SUBSTANDARD in band_names
	assert LoanClassificationBand.DOUBTFUL in band_names
	assert LoanClassificationBand.LOSS in band_names


async def test_loan_classification_totals(svc):
	lc = await svc.classify_loan_portfolio(TENANT, "2025-03-31")
	assert lc.total_gross_portfolio == sum(b.outstanding_balance for b in lc.bands)
	assert lc.total_required_provisions == sum(b.required_provision for b in lc.bands)


async def test_npl_ratio(svc):
	npl = await svc.calculate_npl_ratio(TENANT, "2025-03-31")
	lc = await svc.classify_loan_portfolio(TENANT, "2025-03-31")
	assert npl == lc.npl_ratio
	npl_bands = {LoanClassificationBand.SUBSTANDARD, LoanClassificationBand.DOUBTFUL, LoanClassificationBand.LOSS}
	npl_bal = sum(b.outstanding_balance for b in lc.bands if b.band in npl_bands)
	expected = (npl_bal / lc.total_gross_portfolio * 100).quantize(Decimal("0.0001"))
	assert abs(npl - expected) < Decimal("0.01")


async def test_par30_par90(svc):
	par30 = await svc.calculate_par(TENANT, "2025-03-31", 30)
	par90 = await svc.calculate_par(TENANT, "2025-03-31", 90)
	assert par30 >= par90


async def test_provision_rates_correct(svc):
	lc = await svc.classify_loan_portfolio(TENANT, "2025-03-31")
	rates = {b.band: b.provision_rate for b in lc.bands}
	assert rates[LoanClassificationBand.NORMAL] == Decimal("0")
	assert rates[LoanClassificationBand.WATCH] == Decimal("1")
	assert rates[LoanClassificationBand.SUBSTANDARD] == Decimal("25")
	assert rates[LoanClassificationBand.DOUBTFUL] == Decimal("50")
	assert rates[LoanClassificationBand.LOSS] == Decimal("100")


# ── Required Provisions & Coverage ───────────────────────────────────────────

async def test_required_provisions(svc):
	req = await svc.calculate_required_provisions(TENANT, "2025-03-31")
	assert req > 0


async def test_provisioning_coverage_below_100_when_underprovision(svc_breach):
	cov = await svc_breach.calculate_provisioning_coverage(TENANT, "2025-03-31")
	assert cov < Decimal("100")


# ── Quarterly Return ──────────────────────────────────────────────────────────

async def test_quarterly_return_structure(svc):
	qr = await svc.generate_quarterly_return(TENANT, 2025, 1)
	assert qr.year == 2025
	assert qr.quarter == 1
	assert qr.period_end == "2025-03-31"
	assert qr.form1_balance_sheet is not None
	assert qr.form2_income_statement is not None
	assert qr.form3_capital_adequacy is not None
	assert qr.form4_liquidity is not None
	assert qr.form5_loan_classification is not None


async def test_quarterly_return_compliant(svc):
	qr = await svc.generate_quarterly_return(TENANT, 2025, 1)
	assert qr.overall_compliant is True
	assert len(qr.violations) == 0


async def test_quarterly_return_violations(svc_breach):
	qr = await svc_breach.generate_quarterly_return(TENANT, 2025, 1)
	assert qr.overall_compliant is False
	assert len(qr.violations) > 0


async def test_quarterly_return_invalid_quarter(svc):
	with pytest.raises(AssertionError):
		await svc.generate_quarterly_return(TENANT, 2025, 5)


# ── Annual Report ─────────────────────────────────────────────────────────────

async def test_annual_report_structure(svc):
	ar = await svc.generate_annual_report(TENANT, 2025)
	assert ar.year == 2025
	assert ar.balance_sheet is not None
	assert ar.income_statement is not None
	assert ar.capital_adequacy is not None
	assert ar.liquidity is not None
	assert "capital_adequacy_ratio_pct" in ar.key_ratios
	assert "liquidity_ratio_pct" in ar.key_ratios
	assert len(ar.quarterly_snapshots) == 4


# ── Compliance Status ─────────────────────────────────────────────────────────

async def test_compliance_status_compliant(svc):
	cs = await svc.check_regulatory_compliance(TENANT, "2025-03-31")
	assert cs.overall_compliant is True
	assert len(cs.violations) == 0
	assert len(cs.ratios) >= 6


async def test_compliance_status_breach(svc_breach):
	cs = await svc_breach.check_regulatory_compliance(TENANT, "2025-03-31")
	assert cs.overall_compliant is False
	assert len(cs.violations) > 0


async def test_compliance_ratios_have_traffic_lights(svc):
	cs = await svc.check_regulatory_compliance(TENANT, "2025-03-31")
	for ratio in cs.ratios:
		assert ratio.traffic_light in {TrafficLight.GREEN, TrafficLight.AMBER, TrafficLight.RED}


# ── Filing Registry ───────────────────────────────────────────────────────────

async def test_file_return_and_retrieve(svc):
	filing = await svc.file_return(
		TENANT,
		ReturnType.QUARTERLY,
		"2025-Q1",
		{"note": "test"},
		"John Kamau",
	)
	assert filing.filing_status == FilingStatus.SUBMITTED
	assert filing.period == "2025-Q1"
	assert filing.tenant_id == TENANT

	history = await svc.get_filing_history(TENANT)
	assert any(f.id == filing.id for f in history)


async def test_filing_history_filtered_by_date(svc):
	await svc.file_return(TENANT, ReturnType.QUARTERLY, "2025-Q1", {}, "Officer A", submitted_at="2025-04-05T00:00:00Z")
	await svc.file_return(TENANT, ReturnType.QUARTERLY, "2025-Q2", {}, "Officer B", submitted_at="2025-07-10T00:00:00Z")

	history = await svc.get_filing_history(TENANT, from_date="2025-07-01")
	assert all(f.submitted_at >= "2025-07-01" for f in history)


async def test_file_return_requires_officer(svc):
	with pytest.raises(AssertionError):
		await svc.file_return(TENANT, ReturnType.QUARTERLY, "2025-Q1", {}, "")


# ── Regulatory Calendar ───────────────────────────────────────────────────────

async def test_regulatory_calendar_count(svc):
	cal = await svc.get_regulatory_calendar(TENANT, 2025)
	assert len(cal) == 5


async def test_regulatory_calendar_due_dates(svc):
	cal = await svc.get_regulatory_calendar(TENANT, 2025)
	periods = {d.period for d in cal}
	assert "2025-Q1" in periods
	assert "2025-Q4" in periods
	assert "2025-annual" in periods


async def test_regulatory_calendar_q1_due_date(svc):
	cal = await svc.get_regulatory_calendar(TENANT, 2025)
	q1 = next(d for d in cal if d.period == "2025-Q1")
	assert q1.due_date == "2025-04-30"


async def test_pending_filings_marks_filed(svc):
	await svc.file_return(TENANT, ReturnType.QUARTERLY, "2025-Q1", {}, "Officer")
	cal = await svc.get_regulatory_calendar(TENANT, 2025)
	q1 = next(d for d in cal if d.period == "2025-Q1")
	assert q1.filed is True


# ── Compliance Dashboard ──────────────────────────────────────────────────────

async def test_compliance_dashboard_green(svc):
	dash = await svc.get_compliance_dashboard(TENANT, "2025-03-31")
	assert dash.overall_status in {TrafficLight.GREEN, TrafficLight.AMBER}
	assert len(dash.ratios) >= 6


async def test_compliance_dashboard_red(svc_breach):
	dash = await svc_breach.get_compliance_dashboard(TENANT, "2025-03-31")
	assert dash.overall_status == TrafficLight.RED


# ── Board Report ──────────────────────────────────────────────────────────────

async def test_board_report_structure(svc):
	report = await svc.generate_board_report(TENANT, "2025-03-31")
	assert "key_ratios" in report
	assert "loan_portfolio" in report
	assert "overall_compliant" in report
	assert "executive_summary" in report
	assert "pending_filings" in report


# ── XML Return ────────────────────────────────────────────────────────────────

async def test_xml_return_valid_xml(svc):
	import xml.etree.ElementTree as ET
	xml_str = await svc.generate_sasra_xml_return(TENANT, 2025, 1)
	assert xml_str.startswith("<?xml")
	root = ET.fromstring(xml_str.split("\n", 1)[1])
	assert root.tag == "SASRAReturn"
	assert root.find("Header/Year").text == "2025"
	assert root.find("Header/Quarter").text == "1"


async def test_xml_return_contains_all_forms(svc):
	import xml.etree.ElementTree as ET
	xml_str = await svc.generate_sasra_xml_return(TENANT, 2025, 1)
	root = ET.fromstring(xml_str.split("\n", 1)[1])
	assert root.find("Form1BalanceSheet") is not None
	assert root.find("Form2IncomeStatement") is not None
	assert root.find("Form3CapitalAdequacy") is not None
	assert root.find("Form4Liquidity") is not None
	assert root.find("Form5LoanClassification") is not None


# ── Health Check ──────────────────────────────────────────────────────────────

async def test_health_check(svc):
	h = await svc.health_check()
	assert h["status"] == "healthy"
	assert h["service"] == "fintech_sacco_reg"


# ── Tenant isolation ──────────────────────────────────────────────────────────

async def test_tenant_isolation():
	svc_a = SACCARegulatoryService("tenant-a")
	svc_b = SACCARegulatoryService("tenant-b")
	svc_a.seed_ledger("tenant-a", "2025-03-31", {"core_capital": 10_000_000, "total_assets": 50_000_000, "member_deposits": 30_000_000})
	cap_b = await svc_b.calculate_capital_adequacy("tenant-b", "2025-03-31")
	assert cap_b.institutional_capital == Decimal("0")


# ── DPD band edge cases ───────────────────────────────────────────────────────

async def test_dpd_boundary_30_is_normal(svc):
	"""DPD exactly 30 = Normal band."""
	svc.seed_ledger(TENANT, "2025-06-30", {
		"loan_books": [{"outstanding_balance": 100_000, "days_past_due": 30}],
		"loan_loss_provisions": 0,
	})
	lc = await svc.classify_loan_portfolio(TENANT, "2025-06-30")
	normal = next(b for b in lc.bands if b.band == LoanClassificationBand.NORMAL)
	assert normal.outstanding_balance == Decimal("100000")
	assert normal.required_provision == Decimal("0")


async def test_dpd_boundary_31_is_watch(svc):
	"""DPD exactly 31 = Watch band (1% provision)."""
	svc.seed_ledger(TENANT, "2025-06-30", {
		"loan_books": [{"outstanding_balance": 100_000, "days_past_due": 31}],
		"loan_loss_provisions": 0,
	})
	lc = await svc.classify_loan_portfolio(TENANT, "2025-06-30")
	watch = next(b for b in lc.bands if b.band == LoanClassificationBand.WATCH)
	assert watch.outstanding_balance == Decimal("100000")
	assert watch.required_provision == Decimal("1000.00")  # 1% of 100k


async def test_dpd_loss_provision_100_pct(svc):
	"""Loss band = 100% provision."""
	svc.seed_ledger(TENANT, "2025-06-30", {
		"loan_books": [{"outstanding_balance": 50_000, "days_past_due": 400}],
		"loan_loss_provisions": 0,
	})
	lc = await svc.classify_loan_portfolio(TENANT, "2025-06-30")
	loss = next(b for b in lc.bands if b.band == LoanClassificationBand.LOSS)
	assert loss.required_provision == Decimal("50000.00")
