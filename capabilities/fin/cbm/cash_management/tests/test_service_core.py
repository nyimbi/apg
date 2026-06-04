"""
APG Cash Management — Core service tests.

APG standards: no @pytest.mark.asyncio, no mocks, real objects only.
Run: uv run pytest -vxs tests/test_service_core.py

© 2025 Datacraft. Author: Nyimbi Odero
"""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest

from ..service import CashManagementService
from ..domain.calculations import (
	calculate_simple_interest,
	calculate_compound_interest,
	convert_currency,
	calculate_fx_unrealised_pnl,
	calculate_liquidity_coverage_ratio,
	calculate_forecast_mape,
	calculate_overdraft_interest,
	calculate_reconciliation_variance,
	calculate_credit_headroom,
)
from ..domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_no_cross_tenant_access,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def svc() -> CashManagementService:
	return CashManagementService()


@pytest.fixture()
def tenant() -> str:
	return "tenant-cbm-test"


@pytest.fixture()
def bank(svc: CashManagementService, tenant: str) -> dict:
	return svc.create_bank("bk1", tenant, "KCBKE", "KCB Bank Kenya")


@pytest.fixture()
def account(svc: CashManagementService, tenant: str, bank: dict) -> dict:
	return svc.create_cash_account(
		"acct1", tenant, bank["id"], "001234567", "Operating KES",
		"operating", "KES", minimum_buffer=50_000
	)


@pytest.fixture()
def account_usd(svc: CashManagementService, tenant: str, bank: dict) -> dict:
	return svc.create_cash_account(
		"acct_usd", tenant, bank["id"], "001234568", "Operating USD",
		"operating", "USD", minimum_buffer=10_000
	)


# ---------------------------------------------------------------------------
# Bank CRUD
# ---------------------------------------------------------------------------

class TestBankLifecycle:
	def test_create_bank(self, svc: CashManagementService, tenant: str) -> None:
		b = svc.create_bank("bk2", tenant, "EQKENAD", "Equity Bank")
		assert b["code"] == "EQKENAD"
		assert b["tenant_id"] == tenant
		assert b["status"] == "active"

	def test_create_bank_missing_tenant_raises(self, svc: CashManagementService) -> None:
		with pytest.raises(PermissionError, match="tenant_context_required"):
			svc.create_bank("bk3", "", "TESTBANK", "Test Bank")

	def test_list_banks(self, svc: CashManagementService, tenant: str, bank: dict) -> None:
		banks = svc.list_banks(tenant)
		assert any(b["id"] == bank["id"] for b in banks)

	def test_tenant_isolation(self, svc: CashManagementService, bank: dict) -> None:
		other = svc.list_banks("other-tenant")
		assert not any(b["id"] == bank["id"] for b in other)


# ---------------------------------------------------------------------------
# Cash Account CRUD
# ---------------------------------------------------------------------------

class TestCashAccountLifecycle:
	def test_create_account(self, account: dict, tenant: str) -> None:
		assert account["account_type"] == "operating"
		assert account["currency"] == "KES"
		assert account["minimum_buffer"] == Decimal("50000")

	def test_unsupported_account_type_raises(
		self, svc: CashManagementService, tenant: str, bank: dict
	) -> None:
		with pytest.raises(PermissionError, match="account_type_not_supported"):
			svc.create_cash_account(
				"bad_acct", tenant, bank["id"], "999", "Bad", "crypto_wallet"
			)

	def test_unsupported_currency_raises(
		self, svc: CashManagementService, tenant: str, bank: dict
	) -> None:
		with pytest.raises(PermissionError):
			svc.create_cash_account(
				"bad_cur", tenant, bank["id"], "998", "Bad Cur", "operating", "XYZ"
			)


# ---------------------------------------------------------------------------
# Cash Position
# ---------------------------------------------------------------------------

class TestCashPosition:
	def test_record_position_above_buffer(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		pos = svc.record_cash_position(
			"pos1", tenant, account["id"], "2026-06-01", 100_000
		)
		assert pos["available_balance"] == Decimal("100000")
		assert pos["status"] == "recorded"

	def test_record_position_below_buffer_no_review_raises(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		with pytest.raises(PermissionError, match="liquidity_review_required"):
			svc.record_cash_position(
				"pos2", tenant, account["id"], "2026-06-01", 100
			)

	def test_record_position_below_buffer_with_review(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		pos = svc.record_cash_position(
			"pos3", tenant, account["id"], "2026-06-01", 100,
			liquidity_reviewed_by="treasurer@example.com"
		)
		assert pos["status"] == "reviewed"

	def test_bank_account_balance_returns_latest(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		svc.record_cash_position("p1", tenant, account["id"], "2026-05-01", 200_000)
		svc.record_cash_position("p2", tenant, account["id"], "2026-06-01", 300_000)
		bal = svc.bank_account_balance(account["id"], "2026-06-01", tenant)
		assert bal["available_balance"] == Decimal("300000")


# ---------------------------------------------------------------------------
# Cash Flow
# ---------------------------------------------------------------------------

class TestCashFlow:
	def test_record_inflow(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		flow = svc.record_cash_flow(
			"fl1", tenant, account["id"], "inflow", 500_000,
			"customer_receipts", "2026-06-15"
		)
		assert flow["flow_type"] == "inflow"
		assert flow["amount"] == Decimal("500000")

	def test_record_outflow(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		flow = svc.record_cash_flow(
			"fl2", tenant, account["id"], "outflow", 200_000,
			"supplier_payments", "2026-06-16"
		)
		assert flow["flow_type"] == "outflow"

	def test_unsupported_flow_type_raises(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		with pytest.raises(PermissionError):
			svc.record_cash_flow(
				"fl3", tenant, account["id"], "magic", 1000, "misc", "2026-06-16"
			)


# ---------------------------------------------------------------------------
# Cash Forecast
# ---------------------------------------------------------------------------

class TestCashForecast:
	def test_create_forecast(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		svc.record_cash_flow("f1", tenant, account["id"], "inflow", 1_000_000, "ar", "2026-06-20")
		svc.record_cash_flow("f2", tenant, account["id"], "outflow", 400_000, "ap", "2026-06-25")
		fc = svc.create_cash_forecast(
			"fc1", tenant, 90, "base", 0.85, reviewed_by="cfo@example.com"
		)
		assert fc["projected_net_cash"] == Decimal("600000")
		assert fc["scenario"] == "base"

	def test_low_confidence_no_review_raises(
		self, svc: CashManagementService, tenant: str
	) -> None:
		with pytest.raises(PermissionError):
			svc.create_cash_forecast("fc2", tenant, 30, "stress", 0.50)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class TestReconciliation:
	def test_reconciliation_within_tolerance(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		rec = svc.record_bank_reconciliation(
			"rec1", tenant, account["id"], 1_000_000, 1_000_000,
			reviewed_by="accountant@example.com"
		)
		assert rec["variance"] == Decimal("0")
		assert rec["status"] in ("reconciled", "matched")

	def test_reconciliation_variance_requires_review(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		with pytest.raises(PermissionError):
			svc.record_bank_reconciliation(
				"rec2", tenant, account["id"], 1_000_500, 1_000_000
			)


# ---------------------------------------------------------------------------
# Domain Calculations
# ---------------------------------------------------------------------------

class TestCalculations:
	def test_simple_interest(self) -> None:
		i = calculate_simple_interest(Decimal("1000000"), Decimal("0.12"), 30)
		# 1M * 12% * 30/365
		assert i > Decimal("9000") and i < Decimal("11000")

	def test_compound_interest_positive(self) -> None:
		i = calculate_compound_interest(Decimal("100000"), Decimal("0.08"), 90)
		assert i > Decimal("0")

	def test_convert_currency(self) -> None:
		result = convert_currency(Decimal("1000"), Decimal("130.50"))
		assert result == Decimal("130500.00")

	def test_convert_currency_inverse(self) -> None:
		result = convert_currency(Decimal("130500"), Decimal("130.50"), inverse=True)
		assert result == Decimal("1000.00")

	def test_fx_unrealised_pnl(self) -> None:
		pnl = calculate_fx_unrealised_pnl(
			Decimal("1000"), Decimal("130.00"), Decimal("135.00")
		)
		# 1000 units * (135 - 130) = 5000 positive PnL
		assert pnl == Decimal("5000.00")

	def test_lcr(self) -> None:
		lcr = calculate_liquidity_coverage_ratio(Decimal("1000000"), Decimal("800000"))
		assert lcr > Decimal("1.0")

	def test_mape(self) -> None:
		actuals = [Decimal("100"), Decimal("200"), Decimal("300")]
		forecasts = [Decimal("110"), Decimal("190"), Decimal("310")]
		mape = calculate_forecast_mape(actuals, forecasts)
		# Returns percentage (0-100+); these forecasts are within ~7% of actuals
		assert mape < 10.0

	def test_overdraft_interest(self) -> None:
		interest = calculate_overdraft_interest(
			Decimal("50000"), Decimal("0.24"), 30
		)
		assert interest > Decimal("0")

	def test_reconciliation_variance(self) -> None:
		var = calculate_reconciliation_variance(
			Decimal("1_000_500"), Decimal("1_000_000")
		)
		assert var == Decimal("500")

	def test_credit_headroom(self) -> None:
		headroom = calculate_credit_headroom(Decimal("5_000_000"), Decimal("2_000_000"))
		assert headroom == Decimal("3_000_000")


# ---------------------------------------------------------------------------
# Domain Rules
# ---------------------------------------------------------------------------

class TestDomainRules:
	def test_assert_tenant_context_passes(self) -> None:
		assert_tenant_context({"tenant_id": "t1"})  # no exception

	def test_assert_tenant_context_fails(self) -> None:
		with pytest.raises(RuleViolation, match="tenant_context_required"):
			assert_tenant_context({"tenant_id": ""})

	def test_cross_tenant_denied(self) -> None:
		with pytest.raises(RuleViolation, match="cross_tenant_access_denied"):
			assert_no_cross_tenant_access("tenant_a", "tenant_b")

	def test_same_tenant_passes(self) -> None:
		assert_no_cross_tenant_access("tenant_a", "tenant_a")  # no exception


# ---------------------------------------------------------------------------
# Import statement + MT940 parsing (SWIFT edge cases)
# ---------------------------------------------------------------------------

class TestBankStatementImport:
	def test_import_mt940_basic(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		"""Minimal MT940 statement import via service."""
		mt940_text = (
			":20:STARTUMS\r\n"
			":25:KCB/001234567\r\n"
			":28C:00001/001\r\n"
			":60F:C260601KES1000000,00\r\n"
			":61:2606010601C500000,00NTRFREF001//CUST001\r\n"
			":62F:C260601KES1500000,00\r\n"
		)
		result = svc.import_bank_statement(
			"stmt1", tenant, account["id"], mt940_text, "mt940"
		)
		assert result["format"] == "mt940"
		assert result["tenant_id"] == tenant

	def test_import_mpesa_statement(
		self, svc: CashManagementService, tenant: str, account: dict
	) -> None:
		"""M-Pesa CSV import."""
		mpesa_csv = (
			"Receipt No,Completion Time,Details,Transaction Status,Paid In,Withdrawn,Balance\r\n"
			"ABC123,01/06/2026 09:00,Customer Payment,Completed,50000,,1050000\r\n"
			"DEF456,01/06/2026 11:00,Utility Payment,Completed,,20000,1030000\r\n"
		)
		result = svc.import_bank_statement(
			"stmt2", tenant, account["id"], mpesa_csv, "mpesa"
		)
		assert result["format"] == "mpesa"
		assert result["transaction_count"] >= 2


# ---------------------------------------------------------------------------
# Cash concentration / sweep
# ---------------------------------------------------------------------------

class TestCashConcentration:
	def test_sweep_accounts(
		self, svc: CashManagementService, tenant: str,
		account: dict, account_usd: dict
	) -> None:
		svc.record_cash_position("p_src", tenant, account["id"], "2026-06-01", 2_000_000)
		result = svc.sweep_accounts(
			"sweep1", tenant, [account["id"]], account_usd["id"], "2026-06-01"
		)
		assert "sweep_id" in result
		assert result["tenant_id"] == tenant


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------

class TestDashboard:
	def test_dashboard_summary(
		self, svc: CashManagementService, tenant: str,
		bank: dict, account: dict
	) -> None:
		svc.record_cash_position("pd1", tenant, account["id"], "2026-06-01", 500_000)
		summary = svc.dashboard_summary(tenant)
		assert "bank_count" in summary
		assert summary["bank_count"] >= 1
		assert "total_cash_balance" in summary
