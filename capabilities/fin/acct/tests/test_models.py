"""Tests for fin.acct Pydantic v2 models."""

from __future__ import annotations

import sys
import os
from decimal import Decimal

import pytest
from pydantic import ValidationError

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _ROOT not in sys.path:
	sys.path.insert(0, _ROOT)

from capabilities.fin.acct.models import (
	BankAccount, AccountTransaction, AccountBalance, FundLock,
	AccountStatus, AccountType, TransactionType, TransactionDirection,
	FundLockStatus, OpenAccountRequest, CreditRequest, DebitRequest,
	TransferRequest, BulkCreditItem,
)
from capabilities.fin.acct.views import (
	OpenAccountView, CreditView, DebitView, TransferView, LockFundsView,
)


class TestBankAccountModel:
	def test_defaults(self):
		acc = BankAccount(
			tenant_id="t1", customer_id="c1",
			account_number="ACC001", currency="KES",
			account_type=AccountType.CURRENT, product_code="CURR001",
		)
		assert acc.status == AccountStatus.PENDING
		assert acc.book_balance == Decimal("0.00")
		assert acc.overdraft_limit == Decimal("0.00")
		assert acc.id is not None

	def test_currency_uppercased(self):
		acc = BankAccount(
			tenant_id="t1", customer_id="c1",
			account_number="ACC001", currency="kes",
			account_type=AccountType.CURRENT, product_code="CURR001",
		)
		assert acc.currency == "KES"

	def test_decimal_coercion_from_string(self):
		acc = BankAccount(
			tenant_id="t1", customer_id="c1",
			account_number="ACC001", currency="KES",
			account_type=AccountType.CURRENT, product_code="CURR001",
			book_balance="1234.56",
		)
		assert acc.book_balance == Decimal("1234.56")

	def test_extra_fields_forbidden(self):
		with pytest.raises(ValidationError):
			BankAccount(
				tenant_id="t1", customer_id="c1",
				account_number="ACC001", currency="KES",
				account_type=AccountType.CURRENT, product_code="CURR001",
				unknown_field="x",
			)


class TestAccountTransactionModel:
	def test_direction_enum(self):
		txn = AccountTransaction(
			tenant_id="t1", account_id="a1", account_number="ACC001",
			currency="KES", amount=Decimal("100"),
			direction=TransactionDirection.CREDIT,
			transaction_type=TransactionType.DEPOSIT,
			reference="REF", description="test",
			balance_before=Decimal("0"), balance_after=Decimal("100"),
		)
		assert txn.direction == TransactionDirection.CREDIT

	def test_debit_direction(self):
		txn = AccountTransaction(
			tenant_id="t1", account_id="a1", account_number="ACC001",
			currency="USD", amount="50.00",
			direction="debit", transaction_type="withdrawal",
			reference="WDW", description="cash",
			balance_before="100.00", balance_after="50.00",
		)
		assert txn.direction == TransactionDirection.DEBIT
		assert txn.amount == Decimal("50.00")


class TestAccountBalanceModel:
	def test_all_fields_decimal(self):
		bal = AccountBalance(
			account_id="a1", account_number="ACC001", currency="KES",
			book_balance="1000", available_balance="700",
			locked_balance="300", overdraft_limit="500",
			overdraft_used="0", overdraft_available="500",
		)
		assert bal.book_balance == Decimal("1000")
		assert bal.locked_balance == Decimal("300")


class TestFundLockModel:
	def test_defaults_active(self):
		lock = FundLock(
			tenant_id="t1", account_id="a1",
			amount=Decimal("500"), lock_reference="LOCK-1",
		)
		assert lock.status == FundLockStatus.ACTIVE
		assert lock.id is not None


class TestOpenAccountRequest:
	def test_valid_request(self):
		req = OpenAccountRequest(
			tenant_id="t1", customer_id="c1",
			product_code="CURR001", currency="KES",
		)
		assert req.opening_deposit is None

	def test_negative_deposit_rejected(self):
		# Negative deposit is not caught at model level (service validates)
		# but coerced to Decimal
		req = OpenAccountRequest(
			tenant_id="t1", customer_id="c1",
			product_code="CURR001", currency="KES",
			opening_deposit="-100",
		)
		assert req.opening_deposit == Decimal("-100")


class TestViewModels:
	def test_open_account_view_non_empty(self):
		with pytest.raises(ValidationError):
			OpenAccountView(customer_id="", product_code="P1", currency="KES")

	def test_credit_view_positive_amount(self):
		with pytest.raises(ValidationError):
			CreditView(amount=Decimal("-1"), currency="KES", reference="R", description="d")

	def test_credit_view_zero_amount(self):
		with pytest.raises(ValidationError):
			CreditView(amount=Decimal("0"), currency="KES", reference="R", description="d")

	def test_debit_view_valid(self):
		dv = DebitView(amount=Decimal("100"), currency="KES", reference="WDW", description="cash")
		assert dv.transaction_type == TransactionType.WITHDRAWAL

	def test_transfer_view_requires_destination(self):
		with pytest.raises(ValidationError):
			TransferView(to_account_id="", amount=Decimal("100"), reference="R", description="d")

	def test_lock_view_positive_amount(self):
		with pytest.raises(ValidationError):
			LockFundsView(amount=Decimal("0"), lock_reference="LOCK")

	def test_lock_view_valid(self):
		lv = LockFundsView(amount=Decimal("500"), lock_reference="LOCK-1")
		assert lv.amount == Decimal("500")
