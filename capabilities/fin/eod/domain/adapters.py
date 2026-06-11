"""Domain adapters for EOD/BOD Processing Engine.

Swap these adapters to connect to real data stores (PostgreSQL via SQLAlchemy,
Redis for idempotency, Kafka for events). The service.py uses these hooks via
dependency injection — pass adapter instances to EODService.__init__().

Pattern: each adapter is a Protocol + default in-memory implementation.
"""
from __future__ import annotations

from abc import abstractmethod
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable


# ── Account Adapter ───────────────────────────────────────────────────────────

@runtime_checkable
class AccountAdapter(Protocol):
	"""Read account data needed for EOD batch jobs."""

	@abstractmethod
	async def get_interest_bearing_accounts(self, tenant_id: str, processing_date: str) -> list[dict[str, Any]]:
		"""Return accounts with interest_rate > 0 and status = ACTIVE."""
		...

	@abstractmethod
	async def get_fee_due_accounts(self, tenant_id: str, processing_date: str) -> list[dict[str, Any]]:
		"""Return accounts with fee schedules due on processing_date."""
		...

	@abstractmethod
	async def get_dormancy_candidates(self, tenant_id: str, processing_date: str, threshold_days: int) -> list[dict[str, Any]]:
		"""Return active accounts with no customer transaction for > threshold_days."""
		...

	@abstractmethod
	async def mark_dormant(self, tenant_id: str, account_ids: list[str], processing_date: str) -> int:
		"""Flip status = DORMANT for given account_ids. Returns count updated."""
		...

	@abstractmethod
	async def get_suspense_balance(self, tenant_id: str) -> Decimal:
		"""Return total balance in suspense GL accounts."""
		...

	@abstractmethod
	async def get_unposted_entry_count(self, tenant_id: str) -> int:
		"""Return count of unposted journal entries."""
		...


# ── Term Deposit Adapter ──────────────────────────────────────────────────────

@runtime_checkable
class TermDepositAdapter(Protocol):

	@abstractmethod
	async def get_maturing_deposits(self, tenant_id: str, maturity_date: str) -> list[dict[str, Any]]:
		"""Return term deposits maturing on maturity_date."""
		...

	@abstractmethod
	async def process_maturity(self, tenant_id: str, deposit_id: str, action: str, processing_date: str) -> dict[str, Any]:
		"""action: 'payout' or 'renew'. Returns transaction details."""
		...


# ── Loan Adapter ──────────────────────────────────────────────────────────────

@runtime_checkable
class LoanAdapter(Protocol):

	@abstractmethod
	async def get_repayments_due(self, tenant_id: str, due_date: str) -> list[dict[str, Any]]:
		"""Return loan instalments due on due_date."""
		...

	@abstractmethod
	async def process_repayment(self, tenant_id: str, instalment_id: str, processing_date: str) -> dict[str, Any]:
		"""Collect repayment; update arrears if insufficient funds."""
		...


# ── Standing Order Adapter ────────────────────────────────────────────────────

@runtime_checkable
class StandingOrderAdapter(Protocol):

	@abstractmethod
	async def get_orders_due(self, tenant_id: str, execution_date: str) -> list[dict[str, Any]]:
		...

	@abstractmethod
	async def execute_order(self, tenant_id: str, order_id: str, processing_date: str) -> dict[str, Any]:
		...


# ── GL / Period Adapter ───────────────────────────────────────────────────────

@runtime_checkable
class GLAdapter(Protocol):

	@abstractmethod
	async def is_period_open(self, tenant_id: str, year: int, month: int) -> bool:
		...

	@abstractmethod
	async def open_period(self, tenant_id: str, year: int, month: int) -> bool:
		...

	@abstractmethod
	async def lock_period(self, tenant_id: str, year: int, month: int) -> bool:
		...

	@abstractmethod
	async def post_journal(self, tenant_id: str, entries: list[dict[str, Any]], reference: str) -> str:
		"""Post balanced journal entries. Returns journal_id."""
		...

	@abstractmethod
	async def get_trial_balance(self, tenant_id: str, as_of: str) -> dict[str, Decimal]:
		"""Return {account_code: balance} as of date."""
		...


# ── FX Rate Adapter ───────────────────────────────────────────────────────────

@runtime_checkable
class FXAdapter(Protocol):

	@abstractmethod
	async def get_closing_rates(self, tenant_id: str, rate_date: str) -> dict[str, Decimal]:
		"""Return {currency_code: rate_to_base} for rate_date."""
		...

	@abstractmethod
	async def get_fcy_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return accounts with foreign currency balances."""
		...


# ── Event Bus Adapter ─────────────────────────────────────────────────────────

@runtime_checkable
class EventBusAdapter(Protocol):

	@abstractmethod
	async def publish(self, tenant_id: str, event_type: str, payload: dict[str, Any]) -> None:
		"""Publish an EOD event for downstream consumers (NATS, Kafka, etc.)."""
		...


# ── Null (no-op) adapters for in-memory / test usage ─────────────────────────

class NullAccountAdapter:
	async def get_interest_bearing_accounts(self, tenant_id: str, processing_date: str) -> list[dict[str, Any]]:
		return []
	async def get_fee_due_accounts(self, tenant_id: str, processing_date: str) -> list[dict[str, Any]]:
		return []
	async def get_dormancy_candidates(self, tenant_id: str, processing_date: str, threshold_days: int) -> list[dict[str, Any]]:
		return []
	async def mark_dormant(self, tenant_id: str, account_ids: list[str], processing_date: str) -> int:
		return 0
	async def get_suspense_balance(self, tenant_id: str) -> Decimal:
		return Decimal("0")
	async def get_unposted_entry_count(self, tenant_id: str) -> int:
		return 0


class NullTermDepositAdapter:
	async def get_maturing_deposits(self, tenant_id: str, maturity_date: str) -> list[dict[str, Any]]:
		return []
	async def process_maturity(self, tenant_id: str, deposit_id: str, action: str, processing_date: str) -> dict[str, Any]:
		return {}


class NullLoanAdapter:
	async def get_repayments_due(self, tenant_id: str, due_date: str) -> list[dict[str, Any]]:
		return []
	async def process_repayment(self, tenant_id: str, instalment_id: str, processing_date: str) -> dict[str, Any]:
		return {}


class NullStandingOrderAdapter:
	async def get_orders_due(self, tenant_id: str, execution_date: str) -> list[dict[str, Any]]:
		return []
	async def execute_order(self, tenant_id: str, order_id: str, processing_date: str) -> dict[str, Any]:
		return {}


class NullGLAdapter:
	async def is_period_open(self, tenant_id: str, year: int, month: int) -> bool:
		return True
	async def open_period(self, tenant_id: str, year: int, month: int) -> bool:
		return True
	async def lock_period(self, tenant_id: str, year: int, month: int) -> bool:
		return True
	async def post_journal(self, tenant_id: str, entries: list[dict[str, Any]], reference: str) -> str:
		return f"jnl:{reference}"
	async def get_trial_balance(self, tenant_id: str, as_of: str) -> dict[str, Decimal]:
		return {}


class NullFXAdapter:
	async def get_closing_rates(self, tenant_id: str, rate_date: str) -> dict[str, Decimal]:
		return {"USD": Decimal("130.50"), "EUR": Decimal("142.30"), "GBP": Decimal("165.00")}
	async def get_fcy_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
		return []


class NullEventBusAdapter:
	async def publish(self, tenant_id: str, event_type: str, payload: dict[str, Any]) -> None:
		pass  # discard in-memory


# ── Default adapter bundle ────────────────────────────────────────────────────

class DefaultAdapters:
	"""Bundle of null adapters suitable for testing and initial deployment."""

	accounts:       NullAccountAdapter      = NullAccountAdapter()
	term_deposits:  NullTermDepositAdapter  = NullTermDepositAdapter()
	loans:          NullLoanAdapter         = NullLoanAdapter()
	standing_orders: NullStandingOrderAdapter = NullStandingOrderAdapter()
	gl:             NullGLAdapter           = NullGLAdapter()
	fx:             NullFXAdapter           = NullFXAdapter()
	events:         NullEventBusAdapter     = NullEventBusAdapter()
