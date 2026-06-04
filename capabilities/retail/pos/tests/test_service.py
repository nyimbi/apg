"""Comprehensive service tests for APG Point of Sale.

Tests use real objects (no mocks), plain async functions, and
asyncio.get_event_loop() per CLAUDE.md standards.
"""
from __future__ import annotations

import asyncio

import pytest

from ..service import PointOfSaleService
from ..models import (
	CashEventType,
	DiscountType,
	PosSessionCreate,
	PosTerminalCreate,
	RefundReason,
	SessionStatus,
	TerminalStatus,
	TransactionStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


T1 = "tenant-001"
T2 = "tenant-002"
STORE = "store-nairobi-01"
CASHIER = "cashier-alice"
SUPERVISOR = "supervisor-bob"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
	return PointOfSaleService()


@pytest.fixture
def terminal(svc):
	result = run(svc.register_terminal(PosTerminalCreate(
		tenant_id=T1,
		store_id=STORE,
		terminal_code="T001",
		terminal_type="fixed_counter",
		floor_limit=10000.0,
		created_by="admin",
	)))
	return result


@pytest.fixture
def terminal2(svc):
	result = run(svc.register_terminal(PosTerminalCreate(
		tenant_id=T1,
		store_id=STORE,
		terminal_code="T002",
		terminal_type="mobile",
		created_by="admin",
	)))
	return result


@pytest.fixture
def session(svc, terminal):
	result = run(svc.open_session(
		terminal_id=terminal["id"],
		cashier_id=CASHIER,
		opening_float=1000.0,
		tenant_id=T1,
		store_id=STORE,
		created_by=CASHIER,
	))
	return result


@pytest.fixture
def txn(svc, session):
	"""Pending transaction with 2x SKU-MILK @ 120 = 240."""
	t = run(svc.begin_transaction(
		session_id=session["id"],
		tenant_id=T1,
		cashier_id=CASHIER,
		created_by=CASHIER,
	))
	svc._inventory.set_price(T1, "SKU-MILK", 120.0)
	svc._inventory.set_stock(STORE, "SKU-MILK", 50)
	run(svc.add_item(
		transaction_id=t["id"],
		sku="SKU-MILK",
		quantity=2,
		tenant_id=T1,
		description="Whole Milk 1L",
		created_by=CASHIER,
	))
	return run(svc.get_transaction(t["id"], tenant_id=T1))


# ===========================================================================
# TERMINAL TESTS
# ===========================================================================

class TestTerminal:

	def test_register_terminal(self, svc):
		t = run(svc.register_terminal(PosTerminalCreate(
			tenant_id=T1, store_id=STORE, terminal_code="TX1",
			terminal_type="kiosk", created_by="admin",
		)))
		assert t["id"]
		assert t["terminal_type"] == "kiosk"
		assert t["status"] == TerminalStatus.OFFLINE.value

	def test_duplicate_terminal_code_raises(self, svc, terminal):
		with pytest.raises(AssertionError):
			run(svc.register_terminal(PosTerminalCreate(
				tenant_id=T1, store_id=STORE, terminal_code="T001",
				terminal_type="fixed_counter", created_by="admin",
			)))

	def test_heartbeat_sets_online(self, svc, terminal):
		updated = run(svc.heartbeat_terminal(T1, terminal["id"]))
		assert updated["status"] == TerminalStatus.ONLINE.value
		assert updated["last_heartbeat_at"] is not None

	def test_get_terminal(self, svc, terminal):
		fetched = run(svc.get_terminal(T1, terminal["id"]))
		assert fetched["id"] == terminal["id"]

	def test_list_terminals_by_store(self, svc, terminal, terminal2):
		all_t = run(svc.list_terminals(T1, STORE))
		ids = [t["id"] for t in all_t]
		assert terminal["id"] in ids
		assert terminal2["id"] in ids

	def test_tenant_isolation_terminal(self, svc, terminal):
		other = run(svc.get_terminal(T2, terminal["id"]))
		assert other is None


# ===========================================================================
# SESSION TESTS
# ===========================================================================

class TestSession:

	def test_open_session_basic(self, svc, terminal):
		s = run(svc.open_session(
			terminal_id=terminal["id"],
			cashier_id=CASHIER,
			opening_float=500.0,
			tenant_id=T1,
			store_id=STORE,
			created_by=CASHIER,
		))
		assert s["id"]
		assert s["status"] == SessionStatus.OPEN.value
		assert s["opening_float"] == 500.0

	def test_cannot_open_duplicate_session(self, svc, terminal, session):
		with pytest.raises(AssertionError, match="open session"):
			run(svc.open_session(
				terminal_id=terminal["id"],
				cashier_id="another-cashier",
				opening_float=200.0,
				tenant_id=T1,
				store_id=STORE,
				created_by="another-cashier",
			))

	def test_close_session(self, svc, session):
		closed = run(svc.close_session(
			session_id=session["id"],
			closing_float=1000.0,
			tenant_id=T1,
		))
		assert closed["status"] == SessionStatus.CLOSED.value
		assert closed["closing_cash_counted"] == 1000.0
		assert closed["variance"] == pytest.approx(0.0)

	def test_session_variance_calculated(self, svc, session):
		"""Variance = counted - (opening_float + cash_sales). No sales → variance = counted - float."""
		closed = run(svc.close_session(
			session_id=session["id"],
			closing_float=950.0,
			tenant_id=T1,
		))
		assert closed["variance"] == pytest.approx(-50.0)

	def test_suspend_and_resume_session(self, svc, session):
		suspended = run(svc.suspend_session(T1, session["id"]))
		assert suspended["status"] == SessionStatus.SUSPENDED.value
		resumed = run(svc.resume_session(T1, session["id"]))
		assert resumed["status"] == SessionStatus.OPEN.value

	def test_get_session(self, svc, session):
		fetched = run(svc.get_session(session["id"], tenant_id=T1))
		assert fetched["id"] == session["id"]

	def test_session_tenant_isolation(self, svc, session):
		fetched = run(svc.get_session(session["id"], tenant_id=T2))
		assert fetched is None

	def test_session_summary_empty(self, svc, session):
		summary = run(svc.session_summary(session["id"], tenant_id=T1))
		assert summary["session_id"] == session["id"]
		assert summary["gross_sales"] == 0.0
		assert summary["transaction_count"] == 0

	def test_list_sessions(self, svc, session):
		sessions = run(svc.list_sessions(T1, store_id=STORE))
		assert any(s["id"] == session["id"] for s in sessions)


# ===========================================================================
# TRANSACTION TESTS
# ===========================================================================

class TestTransaction:

	def test_begin_transaction(self, svc, session):
		t = run(svc.begin_transaction(
			session_id=session["id"],
			tenant_id=T1,
			cashier_id=CASHIER,
			created_by=CASHIER,
		))
		assert t["id"]
		assert t["status"] == TransactionStatus.PENDING.value
		assert t["items"] == []

	def test_add_item(self, svc, session):
		svc._inventory.set_price(T1, "SKU-A", 100.0)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		updated = run(svc.add_item(
			transaction_id=t["id"], sku="SKU-A", quantity=3,
			tenant_id=T1, description="Widget A", created_by=CASHIER,
		))
		assert len(updated["items"]) == 1
		assert updated["items"][0]["sku"] == "SKU-A"
		assert updated["items"][0]["quantity"] == 3
		assert updated["subtotal"] == pytest.approx(300.0)

	def test_add_item_with_price_override(self, svc, session):
		svc._inventory.set_price(T1, "SKU-B", 200.0)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		updated = run(svc.add_item(
			transaction_id=t["id"], sku="SKU-B", quantity=1,
			price_override=150.0, tenant_id=T1, description="Widget B", created_by=CASHIER,
		))
		assert updated["items"][0]["unit_price"] == 150.0

	def test_remove_item(self, svc, session):
		svc._inventory.set_price(T1, "SKU-C", 50.0)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		run(svc.add_item(transaction_id=t["id"], sku="SKU-C", quantity=1, tenant_id=T1, description="C", created_by=CASHIER))
		updated = run(svc.remove_item(transaction_id=t["id"], line_id="SKU-C", tenant_id=T1))
		assert updated["items"] == []
		assert updated["subtotal"] == 0.0

	def test_complete_transaction_cash(self, svc, session, txn):
		"""2x SKU-MILK at 120 = 240. Tender 300, expect 60 change."""
		completed = run(svc.complete_transaction(
			transaction_id=txn["id"],
			payments=[{"method": "cash", "amount": 300.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert completed["status"] == TransactionStatus.COMPLETED.value
		assert completed["grand_total"] == pytest.approx(240.0)
		assert completed["change_due"] == pytest.approx(60.0)
		assert completed["receipt_number"] is not None

	def test_complete_transaction_updates_session(self, svc, session, txn):
		run(svc.complete_transaction(
			transaction_id=txn["id"],
			payments=[{"method": "cash", "amount": 300.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		s = run(svc.get_session(session["id"], tenant_id=T1))
		assert s["transaction_count"] == 1
		assert s["total_sales"] == pytest.approx(240.0)

	def test_complete_transaction_underpayment_raises(self, svc, session, txn):
		with pytest.raises(AssertionError):
			run(svc.complete_transaction(
				transaction_id=txn["id"],
				payments=[{"method": "cash", "amount": 10.0}],
				tenant_id=T1,
				created_by=CASHIER,
			))

	def test_void_transaction_pending(self, svc, session):
		svc._inventory.set_price(T1, "SKU-D", 99.0)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		run(svc.add_item(transaction_id=t["id"], sku="SKU-D", quantity=1, tenant_id=T1, description="D", created_by=CASHIER))
		voided = run(svc.void_transaction(
			transaction_id=t["id"],
			reason="customer changed mind",
			supervisor_id=SUPERVISOR,
			tenant_id=T1,
		))
		assert voided["status"] == TransactionStatus.VOIDED.value

	def test_void_requires_supervisor(self, svc, session, txn):
		with pytest.raises(AssertionError):
			run(svc.void_transaction(
				transaction_id=txn["id"],
				reason="test",
				supervisor_id="",
				tenant_id=T1,
			))

	def test_void_completed_reverses_session(self, svc, session, txn):
		run(svc.complete_transaction(
			transaction_id=txn["id"],
			payments=[{"method": "cash", "amount": 240.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		s_before = run(svc.get_session(session["id"], tenant_id=T1))
		assert s_before["total_sales"] == pytest.approx(240.0)

		run(svc.void_transaction(
			transaction_id=txn["id"],
			reason="duplicate",
			supervisor_id=SUPERVISOR,
			tenant_id=T1,
		))
		s_after = run(svc.get_session(session["id"], tenant_id=T1))
		assert s_after["total_sales"] == pytest.approx(0.0)

	def test_park_and_retrieve_transaction(self, svc, session, txn):
		parked = run(svc.park_transaction(txn["id"], tenant_id=T1))
		assert parked["status"] == TransactionStatus.SUSPENDED.value
		retrieved = run(svc.retrieve_parked_transaction(txn["id"], tenant_id=T1))
		assert retrieved["status"] == TransactionStatus.PENDING.value

	def test_transaction_tenant_isolation(self, svc, session, txn):
		fetched = run(svc.get_transaction(txn["id"], tenant_id=T2))
		assert fetched is None


# ===========================================================================
# SPLIT / MULTI-TENDER PAYMENT TESTS
# ===========================================================================

class TestSplitPayment:

	def test_split_cash_and_mpesa(self, svc, session, txn):
		"""Pay 240 as 100 cash + 140 M-Pesa."""
		result = run(svc.split_payment(
			transaction_id=txn["id"],
			payment_splits=[
				{"method": "cash", "amount": 100.0},
				{"method": "mpesa", "amount": 140.0},
			],
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert result["total_tendered"] == pytest.approx(240.0)
		assert result["change_due"] == pytest.approx(0.0)
		assert len(result["payments"]) == 2

	def test_split_insufficient_raises(self, svc, session, txn):
		with pytest.raises(AssertionError, match="insufficient tender"):
			run(svc.split_payment(
				transaction_id=txn["id"],
				payment_splits=[{"method": "cash", "amount": 50.0}],
				tenant_id=T1,
				created_by=CASHIER,
			))


# ===========================================================================
# DISCOUNT TESTS
# ===========================================================================

class TestDiscount:

	def test_percentage_discount(self, svc, session, txn):
		"""10% off 240 = 24 discount."""
		updated = run(svc.apply_discount(
			transaction_id=txn["id"],
			discount_type="percentage",
			value=10.0,
			approved_by=SUPERVISOR,
			tenant_id=T1,
		))
		assert updated["discount_total"] == pytest.approx(24.0)

	def test_fixed_discount(self, svc, session, txn):
		updated = run(svc.apply_discount(
			transaction_id=txn["id"],
			discount_type="fixed_amount",
			value=20.0,
			approved_by=SUPERVISOR,
			tenant_id=T1,
		))
		assert updated["discount_total"] == pytest.approx(20.0)

	def test_loyalty_discount_insufficient_points(self, svc, session, txn):
		"""Customer has 0 points; redeem 1000 should fail."""
		txn_obj = svc._store_transactions.get_item(T1, txn["id"])
		data = txn_obj.model_dump()
		data["customer_id"] = "cust-broke"
		from ..models import SaleTransactionResponse
		svc._store_transactions.put(T1, txn["id"], SaleTransactionResponse(**data))
		with pytest.raises(AssertionError, match="insufficient"):
			run(svc.apply_discount(
				transaction_id=txn["id"],
				discount_type="loyalty_points",
				value=1000,
				tenant_id=T1,
			))


# ===========================================================================
# REFUND TESTS
# ===========================================================================

class TestRefund:

	def _complete_sale(self, svc, session, sku="SKU-G", price=200.0):
		svc._inventory.set_price(T1, sku, price)
		svc._inventory.set_stock(STORE, sku, 10)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		run(svc.add_item(transaction_id=t["id"], sku=sku, quantity=2, tenant_id=T1, description="G", created_by=CASHIER))
		return run(svc.complete_transaction(
			transaction_id=t["id"],
			payments=[{"method": "cash", "amount": price * 2}],
			tenant_id=T1,
			created_by=CASHIER,
		))

	def test_process_refund(self, svc, session):
		from ..models import RefundCreate, SaleItemCreate
		completed = self._complete_sale(svc, session)
		refund = run(svc.process_refund(RefundCreate(
			tenant_id=T1,
			original_transaction_id=completed["id"],
			session_id=session["id"],
			terminal_id=completed["terminal_id"],
			items=[SaleItemCreate(
				sku="SKU-G", description="G", quantity=1,
				unit_price=200.0, line_total=200.0,
			)],
			reason=RefundReason.DEFECTIVE,
			created_by=CASHIER,
		)))
		assert refund["refund_amount"] == pytest.approx(200.0)
		assert refund["status"] == TransactionStatus.COMPLETED.value


# ===========================================================================
# CASH FLOAT TESTS
# ===========================================================================

class TestCashFloat:

	def test_cash_float_event_recorded(self, svc, session):
		from ..models import CashFloatCreate
		event = run(svc.record_cash_float(CashFloatCreate(
			tenant_id=T1,
			session_id=session["id"],
			terminal_id=session["terminal_id"],
			store_id=STORE,
			cashier_id=CASHIER,
			event_type=CashEventType.SAFE_DROP,
			amount=300.0,
			authorised_by=SUPERVISOR,
			created_by=CASHIER,
		)))
		assert event["id"]
		assert event["amount"] == 300.0

	def test_cash_count_reconciliation(self, svc, session):
		result = run(svc.cash_count_reconciliation(
			session_id=session["id"],
			counted_cash=950.0,
			denominations={"500": 1, "200": 2, "50": 1},
			tenant_id=T1,
			counted_by=CASHIER,
		))
		assert result["expected_cash"] == pytest.approx(1000.0)
		assert result["variance"] == pytest.approx(-50.0)


# ===========================================================================
# LOYALTY TESTS
# ===========================================================================

class TestLoyalty:

	def test_earn_points_on_sale(self, svc, session):
		svc._inventory.set_price(T1, "SKU-H", 100.0)
		svc._inventory.set_stock(STORE, "SKU-H", 10)
		t = run(svc.begin_transaction(
			session_id=session["id"],
			customer_id="cust-earner",
			tenant_id=T1,
			cashier_id=CASHIER,
			created_by=CASHIER,
		))
		run(svc.add_item(transaction_id=t["id"], sku="SKU-H", quantity=1, tenant_id=T1, description="H", created_by=CASHIER))
		run(svc.complete_transaction(
			transaction_id=t["id"],
			payments=[{"method": "cash", "amount": 100.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		balance = svc._loyalty.balance(T1, "cust-earner")
		assert balance == 100

	def test_loyalty_earn_redeem(self, svc, session):
		svc._loyalty.earn(T1, "cust-redeem", 5000)
		result = run(svc.loyalty_points_earn_redeem(
			customer_id="cust-redeem",
			transaction_id="dummy-txn",
			points_to_redeem=1000,
			points_earned=0,
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert result["points_redeemed"] == 1000
		assert result["points_balance_after"] == 4000

	def test_loyalty_insufficient_points_raises(self, svc):
		from ..domain.rules import RuleViolation
		svc._loyalty.earn(T1, "cust-poor", 10)
		with pytest.raises((AssertionError, RuleViolation)):
			run(svc.loyalty_points_earn_redeem(
				customer_id="cust-poor",
				transaction_id="dummy-txn",
				points_to_redeem=1000,
				points_earned=0,
				tenant_id=T1,
				created_by=CASHIER,
			))


# ===========================================================================
# OFFLINE SYNC TESTS
# ===========================================================================

class TestOfflineSync:

	def test_offline_sync_batch_accepted(self, svc, session, terminal):
		from ..models import OfflineSyncBatch, SaleTransactionCreate
		batch = OfflineSyncBatch(
			tenant_id=T1,
			terminal_id=terminal["id"],
			session_id=session["id"],
			transactions=[
				SaleTransactionCreate(
					tenant_id=T1,
					session_id=session["id"],
					terminal_id=terminal["id"],
					store_id=STORE,
					cashier_id=CASHIER,
					transaction_type="sale",
					offline_mode=True,
					offline_synced=False,
					created_by=CASHIER,
				)
			],
			sync_sequence=1,
			created_by=CASHIER,
		)
		result = run(svc.offline_mode_sync(batch, tenant_id=T1))
		assert result["sync_completed_at"] is not None
		assert "accepted" in result

	def test_offline_sync_duplicate_sequence_raises(self, svc, session, terminal):
		from ..models import OfflineSyncBatch
		from ..domain.rules import RuleViolation
		batch = OfflineSyncBatch(
			tenant_id=T1,
			terminal_id=terminal["id"],
			session_id=session["id"],
			transactions=[],
			sync_sequence=1,
			created_by=CASHIER,
		)
		run(svc.offline_mode_sync(batch, tenant_id=T1))
		with pytest.raises((AssertionError, RuleViolation)):
			run(svc.offline_mode_sync(batch, tenant_id=T1))


# ===========================================================================
# EOD REPORT TESTS
# ===========================================================================

class TestEOD:

	def test_end_of_day_closing(self, svc, terminal, session):
		run(svc.close_session(session_id=session["id"], closing_float=1000.0, tenant_id=T1))
		report = run(svc.end_of_day_closing(
			store_id=STORE,
			business_date="2026-06-04",
			tenant_id=T1,
			generated_by=SUPERVISOR,
			created_by=SUPERVISOR,
		))
		assert report["store_id"] == STORE
		assert report["business_date"] == "2026-06-04"
		assert report["status"] == "draft"

	def test_eod_idempotency_guard(self, svc, terminal, session):
		from ..domain.rules import RuleViolation
		run(svc.close_session(session_id=session["id"], closing_float=1000.0, tenant_id=T1))
		run(svc.end_of_day_closing(
			store_id=STORE, business_date="2026-06-05",
			tenant_id=T1, generated_by=SUPERVISOR, created_by=SUPERVISOR,
		))
		with pytest.raises((AssertionError, RuleViolation)):
			run(svc.end_of_day_closing(
				store_id=STORE, business_date="2026-06-05",
				tenant_id=T1, generated_by=SUPERVISOR, created_by=SUPERVISOR,
			))


# ===========================================================================
# RECEIPT TESTS
# ===========================================================================

class TestReceipt:

	def test_receipt_generation_thermal(self, svc, session, txn):
		run(svc.complete_transaction(
			transaction_id=txn["id"],
			payments=[{"method": "cash", "amount": 240.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		receipt = run(svc.receipt_generation(
			transaction_id=txn["id"],
			fmt="thermal",
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert receipt["receipt_format"] == "thermal"
		assert receipt["rendered_content"] is not None
		assert "Whole Milk" in receipt["rendered_content"]

	def test_receipt_generation_email(self, svc, session, txn):
		run(svc.complete_transaction(
			transaction_id=txn["id"],
			payments=[{"method": "cash", "amount": 240.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		receipt = run(svc.receipt_generation(
			transaction_id=txn["id"],
			fmt="email",
			recipient_email="test@datacraft.co.ke",
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert receipt["receipt_format"] == "email"
		assert receipt["recipient_email"] == "test@datacraft.co.ke"


# ===========================================================================
# PRICE CHECK TESTS
# ===========================================================================

class TestPriceCheck:

	def test_price_check_known_sku(self, svc, terminal):
		svc._inventory.set_price(T1, "SKU-PC", 299.0)
		result = run(svc.price_check(sku="SKU-PC", tenant_id=T1, store_id=STORE))
		assert result["sku"] == "SKU-PC"
		assert result["effective_price"] == pytest.approx(299.0)

	def test_price_check_unknown_sku(self, svc, terminal):
		result = run(svc.price_check(sku="UNKNOWN-XYZ", tenant_id=T1, store_id=STORE))
		assert result["base_price"] is None


# ===========================================================================
# SUPERVISOR OVERRIDE TESTS
# ===========================================================================

class TestSupervisorOverride:

	def test_supervisor_override(self, svc, session, terminal):
		from ..models import SupervisorOverrideCreate
		override = run(svc.supervisor_override(SupervisorOverrideCreate(
			tenant_id=T1,
			session_id=session["id"],
			terminal_id=terminal["id"],
			supervisor_id=SUPERVISOR,
			override_type="void",
			notes="Authorising void of wrong-item sale",
			created_by=CASHIER,
		)))
		assert override["id"]
		assert override["supervisor_id"] == SUPERVISOR

	def test_self_approval_raises(self, svc, session, terminal):
		from ..models import SupervisorOverrideCreate
		from ..domain.rules import RuleViolation
		with pytest.raises((AssertionError, RuleViolation)):
			run(svc.supervisor_override(SupervisorOverrideCreate(
				tenant_id=T1,
				session_id=session["id"],
				terminal_id=terminal["id"],
				supervisor_id=CASHIER,   # same as session cashier
				override_type="price_override",
				created_by=CASHIER,
			)))


# ===========================================================================
# INVENTORY DEDUCTION TESTS
# ===========================================================================

class TestInventoryDeduction:

	def test_stock_deducted_on_complete(self, svc, session):
		svc._inventory.set_price(T1, "SKU-INV", 50.0)
		svc._inventory.set_stock(STORE, "SKU-INV", 20)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		run(svc.add_item(transaction_id=t["id"], sku="SKU-INV", quantity=5, tenant_id=T1, description="Inv", created_by=CASHIER))
		run(svc.complete_transaction(
			transaction_id=t["id"],
			payments=[{"method": "cash", "amount": 250.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert svc._inventory.get_stock(STORE, "SKU-INV") == pytest.approx(15.0)

	def test_stock_restored_on_void(self, svc, session):
		svc._inventory.set_price(T1, "SKU-VOID", 100.0)
		svc._inventory.set_stock(STORE, "SKU-VOID", 10)
		t = run(svc.begin_transaction(session_id=session["id"], tenant_id=T1, cashier_id=CASHIER, created_by=CASHIER))
		run(svc.add_item(transaction_id=t["id"], sku="SKU-VOID", quantity=3, tenant_id=T1, description="V", created_by=CASHIER))
		run(svc.complete_transaction(
			transaction_id=t["id"],
			payments=[{"method": "cash", "amount": 300.0}],
			tenant_id=T1,
			created_by=CASHIER,
		))
		assert svc._inventory.get_stock(STORE, "SKU-VOID") == pytest.approx(7.0)
		run(svc.void_transaction(
			transaction_id=t["id"],
			reason="test void",
			supervisor_id=SUPERVISOR,
			tenant_id=T1,
		))
		assert svc._inventory.get_stock(STORE, "SKU-VOID") == pytest.approx(10.0)


# ===========================================================================
# DOMAIN RULES TESTS
# ===========================================================================

class TestRules:

	def test_cross_tenant_rule(self):
		from ..domain.rules import assert_no_cross_tenant_access, RuleViolation
		with pytest.raises(RuleViolation, match="cross_tenant"):
			assert_no_cross_tenant_access("t1", "t2")

	def test_same_tenant_passes(self):
		from ..domain.rules import assert_no_cross_tenant_access
		assert_no_cross_tenant_access("t1", "t1")

	def test_session_open_rule(self):
		from ..domain.rules import assert_session_open, RuleViolation
		with pytest.raises(RuleViolation, match="session_not_open"):
			assert_session_open("closed")

	def test_sufficient_payment_rule(self):
		from ..domain.rules import assert_sufficient_payment, RuleViolation
		with pytest.raises(RuleViolation, match="insufficient_payment"):
			assert_sufficient_payment(50.0, 100.0)

	def test_discount_percentage_bounds(self):
		from ..domain.rules import assert_discount_percentage_valid, RuleViolation
		with pytest.raises(RuleViolation):
			assert_discount_percentage_valid(0.0)
		with pytest.raises(RuleViolation):
			assert_discount_percentage_valid(101.0)
		assert_discount_percentage_valid(10.0)

	def test_loyalty_redemption_limit(self):
		from ..domain.rules import assert_loyalty_redemption_within_limit, RuleViolation
		with pytest.raises(RuleViolation, match="loyalty_redemption_limit_exceeded"):
			assert_loyalty_redemption_within_limit(300.0, 400.0, max_redemption_pct=0.5)

	def test_void_same_terminal(self):
		from ..domain.rules import assert_void_same_terminal, RuleViolation
		with pytest.raises(RuleViolation, match="cross_terminal"):
			assert_void_same_terminal("T1", "T2")

	def test_eod_not_twice(self):
		from ..domain.rules import assert_eod_not_already_run, RuleViolation
		with pytest.raises(RuleViolation, match="eod_already_run"):
			assert_eod_not_already_run("existing-id", "2026-06-04")

	def test_supervisor_not_self(self):
		from ..domain.rules import assert_supervisor_not_self_approving, RuleViolation
		with pytest.raises(RuleViolation, match="self_approval"):
			assert_supervisor_not_self_approving("alice", "alice")

	def test_tax_exempt_ref_required(self):
		from ..domain.rules import assert_tax_exempt_ref, RuleViolation
		with pytest.raises(RuleViolation):
			assert_tax_exempt_ref(True, None)
		assert_tax_exempt_ref(True, "TAX-EXEMPT-2026-001")


# ===========================================================================
# CALCULATIONS TESTS
# ===========================================================================

class TestCalculations:

	def test_item_subtotal(self):
		from ..domain.calculations import item_subtotal
		assert item_subtotal(100.0, 3) == pytest.approx(300.0)

	def test_vat_inclusive_breakdown(self):
		from ..domain.calculations import vat_inclusive_breakdown
		result = vat_inclusive_breakdown(116.0, 0.16)
		assert result["vat"] == pytest.approx(16.0, abs=0.01)
		assert result["net"] == pytest.approx(100.0, abs=0.01)

	def test_cash_variance(self):
		from ..domain.calculations import cash_variance
		assert cash_variance(1000.0, 950.0) == pytest.approx(-50.0)

	def test_suggest_denominations(self):
		from ..domain.calculations import suggest_denominations
		result = suggest_denominations(1750.0)
		total = sum(int(k) * v for k, v in result.items())
		assert total == 1750

	def test_earn_points(self):
		from ..domain.calculations import earn_points
		assert earn_points(500.0, 1.0) == 500

	def test_expected_cash_in_till(self):
		from ..domain.calculations import expected_cash_in_till
		result = expected_cash_in_till(
			opening_float=1000.0,
			cash_sales=500.0,
			cash_refunds=50.0,
			safe_drops=200.0,
			safe_pickups=0.0,
			petty_cash_out=30.0,
			petty_cash_in=0.0,
			till_loans=0.0,
			corrections=0.0,
		)
		assert result == pytest.approx(1220.0)

	def test_average_transaction_value(self):
		from ..domain.calculations import average_transaction_value
		assert average_transaction_value(1500.0, 5) == pytest.approx(300.0)
		assert average_transaction_value(1500.0, 0) == 0.0
