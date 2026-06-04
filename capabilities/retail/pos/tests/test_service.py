"""Service tests for retail_pos capability."""

import asyncio
import pytest

from ..service import PosService
from ..models import (
	PosTerminalCreate, PosSessionCreate, PosTransactionCreate,
	PosTransactionLineItem, PosCashEventCreate, PosReconciliationCreate,
	PosReceiptCreate, PosVoidCreate,
)


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return PosService()


@pytest.fixture
def terminal(svc):
	return run(svc.register_terminal(PosTerminalCreate(
		tenant_id="t1", store_id="store-01", terminal_code="T001",
		terminal_type="fixed_counter", created_by="admin",
	)))


@pytest.fixture
def session(svc, terminal):
	return run(svc.open_session(PosSessionCreate(
		tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
		cashier_id="cashier-001", opening_float=500.0, created_by="cashier",
	)))


def test_register_terminal(svc):
	t = run(svc.register_terminal(PosTerminalCreate(
		tenant_id="t1", store_id="s1", terminal_code="T100",
		terminal_type="mobile_pos", created_by="admin",
	)))
	assert t.id
	assert t.terminal_type == "mobile_pos"
	assert t.status == "offline"


def test_terminal_heartbeat(svc, terminal):
	rec = run(svc.heartbeat_terminal("t1", terminal.id))
	assert rec.status == "online"


def test_open_session(svc, terminal):
	s = run(svc.open_session(PosSessionCreate(
		tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
		cashier_id="cashier-002", opening_float=200.0, created_by="cashier",
	)))
	assert s.id
	assert s.status == "open"
	assert s.opening_float == 200.0


def test_cannot_open_duplicate_session(svc, terminal, session):
	with pytest.raises(AssertionError, match="open session"):
		run(svc.open_session(PosSessionCreate(
			tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
			cashier_id="cashier-003", opening_float=100.0, created_by="cashier",
		)))


def test_post_transaction(svc, session, terminal):
	items = [PosTransactionLineItem(
		sku="A1", description="Widget", quantity=2, unit_price=10.0,
		line_total=20.0, tax_rate=0.1, tax_amount=2.0,
	)]
	txn = run(svc.post_transaction(PosTransactionCreate(
		tenant_id="t1", session_id=session.id, terminal_id=terminal.id,
		store_id="store-01", cashier_id="cashier-001",
		transaction_type="sale", items=items,
		payment_method="card", tender_amount=22.0,
		created_by="cashier",
	)))
	assert txn.id
	assert txn.grand_total == 22.0
	assert txn.transaction_signed is True


def test_post_transaction_closed_session(svc, terminal):
	# Create and close a session
	s = run(svc.open_session(PosSessionCreate(
		tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
		cashier_id="c99", opening_float=100.0, created_by="c",
	)))
	run(svc.close_session("t1", s.id, 100.0))
	with pytest.raises(AssertionError, match="session must be open"):
		run(svc.post_transaction(PosTransactionCreate(
			tenant_id="t1", session_id=s.id, terminal_id=terminal.id,
			store_id="store-01", cashier_id="c99",
			transaction_type="sale", items=[], payment_method="cash",
			tender_amount=0.0, created_by="c",
		)))


def test_void_transaction(svc, session, terminal):
	items = [PosTransactionLineItem(sku="B1", description="Hat", quantity=1, unit_price=5.0, line_total=5.0)]
	txn = run(svc.post_transaction(PosTransactionCreate(
		tenant_id="t1", session_id=session.id, terminal_id=terminal.id,
		store_id="store-01", cashier_id="cashier-001",
		transaction_type="sale", items=items, payment_method="cash",
		tender_amount=5.0, created_by="cashier",
	)))
	void = run(svc.void_transaction(PosVoidCreate(
		tenant_id="t1", original_transaction_id=txn.id,
		session_id=session.id, terminal_id=terminal.id,
		void_reason="operator_error", created_by="cashier",
	)))
	assert void.status == "completed"


def test_void_cross_terminal_denied(svc, terminal):
	t2 = run(svc.register_terminal(PosTerminalCreate(
		tenant_id="t1", store_id="store-01", terminal_code="T002",
		terminal_type="fixed_counter", created_by="admin",
	)))
	s1 = run(svc.open_session(PosSessionCreate(
		tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
		cashier_id="c1", opening_float=100.0, created_by="c",
	)))
	txn = run(svc.post_transaction(PosTransactionCreate(
		tenant_id="t1", session_id=s1.id, terminal_id=terminal.id,
		store_id="store-01", cashier_id="c1",
		transaction_type="sale", items=[], payment_method="cash",
		tender_amount=0.0, created_by="c",
	)))
	with pytest.raises(AssertionError, match="originating terminal"):
		run(svc.void_transaction(PosVoidCreate(
			tenant_id="t1", original_transaction_id=txn.id,
			session_id=s1.id, terminal_id=t2.id,
			void_reason="operator_error", created_by="cashier",
		)))


def test_cash_event(svc, session, terminal):
	ev = run(svc.record_cash_event(PosCashEventCreate(
		tenant_id="t1", session_id=session.id, terminal_id=terminal.id,
		store_id="store-01", cashier_id="cashier-001",
		cash_event_type="safe_drop", amount=-200.0,
		authorised_by="manager", created_by="cashier",
	)))
	assert ev.id
	assert ev.balance_after == 300.0  # 500 float - 200 drop


def test_reconciliation(svc, session, terminal):
	rec = run(svc.create_reconciliation(PosReconciliationCreate(
		tenant_id="t1", session_id=session.id, terminal_id=terminal.id,
		store_id="store-01", cashier_id="cashier-001",
		system_cash_total=500.0, counted_cash_total=498.50,
		total_card_sales=200.0, total_mobile_money=0.0,
		total_gift_card=0.0, total_other=0.0,
		created_by="cashier",
	)))
	assert rec.variance == pytest.approx(-1.50)
	approved = run(svc.approve_reconciliation("t1", rec.id, "manager"))
	assert approved.status == "approved"


def test_session_summary(svc, session, terminal):
	summary = run(svc.session_summary("t1", session.id))
	assert "session" in summary
	assert "transaction_count" in summary
	assert "cash_balance" in summary


def test_tenant_isolation(svc, terminal):
	s = run(svc.open_session(PosSessionCreate(
		tenant_id="t1", terminal_id=terminal.id, store_id="store-01",
		cashier_id="c-x", opening_float=100.0, created_by="c",
	)))
	assert run(svc.get_session("t2", s.id)) is None
