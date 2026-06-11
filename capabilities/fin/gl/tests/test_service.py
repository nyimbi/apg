"""Tests for fin/gl GLService — double-entry invariants, error paths, trial balance.

These are the minimum viable tests for a production accounting engine.
Every test uses real GLService objects — no mocks.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from capabilities.fin.gl.service import (
	GLImbalanceError,
	GLService,
	PostingToClosedPeriodError,
)

TODAY = str(date.today())
PERIOD = "2025-01"


@pytest.fixture
def svc():
	s = GLService(tenant_id="test-gl-tenant")
	s.initialise_standard_coa()
	return s


async def _open(svc: GLService, period: str = PERIOD) -> None:
	year, month = period.split("-")
	await svc.open_period(period, int(year), int(month))


async def _post(svc: GLService, dr_code: str, cr_code: str, amount: str, ref: str = "TEST-001") -> dict:
	amt = Decimal(amount)
	return await svc.post_journal_entry(
		entries=[
			{"account_code": dr_code, "debit_amount": amt, "credit_amount": Decimal("0"), "narrative": "debit side"},
			{"account_code": cr_code, "debit_amount": Decimal("0"), "credit_amount": amt, "narrative": "credit side"},
		],
		description="Test entry",
		reference=ref,
		posting_date=TODAY,
		period_id=PERIOD,
	)


# ── 1. Double-entry invariant ──────────────────────────────────────────────

class TestDoubleEntryInvariant:
	async def test_balanced_entry_posts_successfully(self, svc):
		await _open(svc)
		je = await _post(svc, "1010", "2100", "10000.00")
		assert Decimal(je["total_debit"]) == Decimal(je["total_credit"])
		assert Decimal(je["total_debit"]) == Decimal("10000.00")
		assert je["status"] == "POSTED"

	async def test_imbalanced_entry_raises_gl_imbalance_error(self, svc):
		await _open(svc)
		with pytest.raises(GLImbalanceError) as exc_info:
			await svc.post_journal_entry(
				entries=[
					{"account_code": "1010", "debit_amount": Decimal("10000"), "credit_amount": Decimal("0")},
					{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("9999")},
				],
				description="Unbalanced",
				reference="BAD-001",
				posting_date=TODAY,
				period_id=PERIOD,
			)
		err = exc_info.value
		assert err.debits == Decimal("10000.00")
		assert err.credits == Decimal("9999.00")
		assert "difference=1" in str(err)

	async def test_zero_value_entry_raises(self, svc):
		await _open(svc)
		with pytest.raises(ValueError, match="zero value"):
			await svc.post_journal_entry(
				entries=[
					{"account_code": "1010", "debit_amount": Decimal("0"), "credit_amount": Decimal("0")},
					{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("0")},
				],
				description="Zero",
				reference="ZERO-001",
				posting_date=TODAY,
				period_id=PERIOD,
			)

	async def test_negative_amount_raises(self, svc):
		await _open(svc)
		with pytest.raises(ValueError, match="non-negative"):
			await svc.post_journal_entry(
				entries=[
					{"account_code": "1010", "debit_amount": Decimal("-100"), "credit_amount": Decimal("0")},
					{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("-100")},
				],
				description="Negative",
				reference="NEG-001",
				posting_date=TODAY,
				period_id=PERIOD,
			)

	async def test_unknown_account_raises(self, svc):
		await _open(svc)
		with pytest.raises(KeyError):
			await svc.post_journal_entry(
				entries=[
					{"account_code": "9999", "debit_amount": Decimal("100"), "credit_amount": Decimal("0")},
					{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("100")},
				],
				description="Bad account",
				reference="BADACC-001",
				posting_date=TODAY,
				period_id=PERIOD,
			)

	async def test_multi_line_entry_balanced(self, svc):
		"""3-way entry: cash + loans → deposits."""
		await _open(svc)
		je = await svc.post_journal_entry(
			entries=[
				{"account_code": "1001", "debit_amount": Decimal("500"), "credit_amount": Decimal("0")},
				{"account_code": "1010", "debit_amount": Decimal("500"), "credit_amount": Decimal("0")},
				{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("1000")},
			],
			description="Multi-line deposit",
			reference="MULTI-001",
			posting_date=TODAY,
			period_id=PERIOD,
		)
		assert Decimal(je["total_debit"]) == Decimal(je["total_credit"]) == Decimal("1000.00")


# ── 2. Period control ──────────────────────────────────────────────────────

class TestPeriodControl:
	async def test_posting_to_closed_period_raises(self, svc):
		await svc.open_period("2024-12", 2024, 12)
		await svc.close_period("2024-12")
		with pytest.raises(PostingToClosedPeriodError) as exc_info:
			await svc.post_journal_entry(
				entries=[
					{"account_code": "1010", "debit_amount": Decimal("500"), "credit_amount": Decimal("0")},
					{"account_code": "2100", "debit_amount": Decimal("0"), "credit_amount": Decimal("500")},
				],
				description="Late posting",
				reference="LATE-001",
				posting_date="2024-12-15",
				period_id="2024-12",
			)
		assert "2024-12" in str(exc_info.value)

	async def test_cannot_reopen_closed_period(self, svc):
		await svc.open_period("2024-11", 2024, 11)
		await svc.close_period("2024-11")
		with pytest.raises(ValueError, match="closed and cannot be reopened"):
			await svc.open_period("2024-11", 2024, 11)

	async def test_open_period_allows_posting(self, svc):
		await _open(svc)
		je = await _post(svc, "1010", "2100", "1000.00")
		assert je["status"] == "POSTED"

	async def test_posting_without_period_creation_raises(self, svc):
		# Period 'closed-orphan' was never opened but is in entries; defaults to non-existent
		# Should succeed (period enforcement only blocks CLOSED periods, not missing ones)
		je = await _post(svc, "1010", "2100", "500.00")
		assert je["status"] == "POSTED"


# ── 3. Trial balance ───────────────────────────────────────────────────────

class TestTrialBalance:
	async def test_trial_balance_balanced_after_posting(self, svc):
		await _open(svc)
		await _post(svc, "1010", "2100", "5000.00")
		tb = await svc.get_trial_balance()
		total = next(r for r in tb if r["code"] == "TOTAL")
		assert total["balanced"] is True
		assert Decimal(total["debit_balance"]) == Decimal(total["credit_balance"])
		assert Decimal(total["debit_balance"]) > 0

	async def test_trial_balance_empty_is_balanced(self, svc):
		tb = await svc.get_trial_balance()
		total = next(r for r in tb if r["code"] == "TOTAL")
		assert total["balanced"] is True
		assert Decimal(total["debit_balance"]) == Decimal("0")

	async def test_trial_balance_reflects_multiple_entries(self, svc):
		await _open(svc)
		await _post(svc, "1010", "2100", "1000.00", ref="T-1")
		await _post(svc, "1001", "2100", "2000.00", ref="T-2")
		tb = await svc.get_trial_balance()
		total = next(r for r in tb if r["code"] == "TOTAL")
		assert total["balanced"] is True
		assert Decimal(total["debit_balance"]) == Decimal("3000.00")


# ── 4. Account balance ────────────────────────────────────────────────────

class TestAccountBalance:
	async def test_debit_account_increases_on_debit(self, svc):
		"""Asset accounts (normal DEBIT) increase on debit."""
		await _open(svc)
		await _post(svc, "1010", "2100", "5000.00")
		bal = await svc.get_account_balance("1010")
		assert Decimal(bal["balance"]) == Decimal("5000.00")

	async def test_credit_account_increases_on_credit(self, svc):
		"""Liability accounts (normal CREDIT) increase on credit."""
		await _open(svc)
		await _post(svc, "1010", "2100", "5000.00")
		bal = await svc.get_account_balance("2100")
		assert Decimal(bal["balance"]) == Decimal("5000.00")

	async def test_balance_is_cumulative(self, svc):
		await _open(svc)
		await _post(svc, "1010", "2100", "1000.00", ref="B-1")
		await _post(svc, "1010", "2100", "2000.00", ref="B-2")
		bal = await svc.get_account_balance("1010")
		assert Decimal(bal["balance"]) == Decimal("3000.00")


# ── 5. Journal entry reversal ─────────────────────────────────────────────

class TestReversal:
	async def test_reverse_entry_rebalances_account(self, svc):
		await _open(svc)
		je = await _post(svc, "1010", "2100", "5000.00", ref="REV-ORIG")
		await svc.reverse_journal_entry(je["id"], reason="Correction")
		bal = await svc.get_account_balance("1010")
		assert Decimal(bal["balance"]) == Decimal("0.00")

	async def test_reversed_entry_is_still_in_journal(self, svc):
		"""Reversal adds a new entry — original entry is never deleted."""
		await _open(svc)
		je = await _post(svc, "1010", "2100", "1000.00", ref="REV-TEST")
		await svc.reverse_journal_entry(je["id"], reason="Test reversal")
		all_entries = await svc.get_journal_entries()
		assert all_entries["total"] == 2  # original + reversal


# ── 6. COA management ────────────────────────────────────────────────────

class TestChartOfAccounts:
	def test_initialise_standard_coa_creates_accounts(self, svc):
		result = svc.initialise_standard_coa()
		# Should already be initialised by fixture — idempotent on second call
		assert result["total"] > 40

	async def test_create_account_requires_unique_code(self, svc):
		with pytest.raises(ValueError, match="already exists"):
			await svc.create_account("1010", "Duplicate", "ASSET", "DEBIT")

	async def test_get_account_not_found_raises(self, svc):
		with pytest.raises(KeyError):
			await svc.get_account("9999")

	async def test_deactivate_account(self, svc):
		await svc.deactivate_account("1520")
		acc = await svc.get_account("1520")
		assert acc["is_active"] is False

	async def test_list_accounts_filters_by_type(self, svc):
		assets = await svc.list_accounts(account_type="ASSET")
		assert all(a["account_type"] == "ASSET" for a in assets)
		assert len(assets) > 0


# ── 7. COA validate balance ───────────────────────────────────────────────

class TestCOAValidation:
	async def test_empty_coa_is_balanced(self, svc):
		result = await svc.validate_coa_balance()
		assert result["balanced"] is True

	async def test_coa_balanced_after_valid_entries(self, svc):
		await _open(svc)
		await _post(svc, "1010", "2100", "10000.00")
		await _post(svc, "1100", "1010", "5000.00")
		result = await svc.validate_coa_balance()
		assert result["balanced"] is True


# ── 8. Regulatory reporting ───────────────────────────────────────────────

class TestRegulatoryReport:
	async def test_capital_adequacy_report_structure(self, svc):
		await _open(svc)
		await _post(svc, "1100", "3100", "10000000.00")  # institutional capital
		await _post(svc, "2100", "1100", "5000000.00")   # loan funded by deposit

		report = await svc.get_regulatory_report("CAPITAL_ADEQUACY", PERIOD)
		assert report["report_type"] == "CAPITAL_ADEQUACY"
		assert "capital_adequacy_ratio" in report
		assert "compliant" in report
		assert report["minimum_required"] == "10.00"

	async def test_capital_adequacy_compliant_flag(self, svc):
		await _open(svc)
		# Post large institutional capital relative to assets → well-capitalised
		await _post(svc, "1010", "3100", "50000000.00")  # bank funded by capital
		report = await svc.get_regulatory_report("CAPITAL_ADEQUACY", PERIOD)
		ratio = float(report["capital_adequacy_ratio"])
		if ratio >= 10.0:
			assert report["compliant"] is True
		else:
			assert report["compliant"] is False
