"""Retail capability integration tests: Point of Sale (pos).

All tests use real in-memory service instances — no mocks.
Async service methods are called via asyncio.run().
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest


def _pos_service():
	from capabilities.retail.pos.service import PointOfSaleService
	return PointOfSaleService()


# ── 1. Open session ───────────────────────────────────────────────────────────

def test_pos_open_session():
	"""open_session returns a session dict with expected keys."""
	svc = _pos_service()
	session = asyncio.run(
		svc.open_session(
			terminal_id="T-001",
			cashier_id="cashier-alice",
			opening_float=5000.0,
			tenant_id="retail-test",
			store_id="store-nairobi-01",
		)
	)
	assert isinstance(session, dict)
	assert "id" in session
	assert session["terminal_id"] == "T-001"
	assert session["cashier_id"] == "cashier-alice"
	assert float(session["opening_float"]) == 5000.0
	assert session["store_id"] == "store-nairobi-01"
	assert session["status"] == "open"


# ── 2. Transaction: begin → add_item → complete ───────────────────────────────

def test_pos_transaction():
	"""begin_transaction → add_item → complete_transaction completes successfully."""
	svc = _pos_service()

	# Seed price
	svc._inventory.set_price("retail-test", "SKU-BREAD", 120.0)

	# Open session
	session = asyncio.run(
		svc.open_session(
			terminal_id="T-002",
			cashier_id="cashier-bob",
			opening_float=2000.0,
			tenant_id="retail-test",
			store_id="store-nairobi-01",
		)
	)
	session_id = session["id"]

	# Begin transaction
	txn = asyncio.run(
		svc.begin_transaction(
			session_id=session_id,
			tenant_id="retail-test",
			cashier_id="cashier-bob",
		)
	)
	txn_id = txn["id"]
	assert txn["status"] == "pending"

	# Add item
	updated = asyncio.run(
		svc.add_item(
			transaction_id=txn_id,
			sku="SKU-BREAD",
			quantity=2.0,
			tenant_id="retail-test",
		)
	)
	assert len(updated["items"]) == 1
	assert float(updated["grand_total"]) > 0

	# Complete with cash payment
	grand_total = float(updated["grand_total"])
	completed = asyncio.run(
		svc.complete_transaction(
			transaction_id=txn_id,
			payments=[{"method": "cash", "amount": grand_total + 50.0}],
			tenant_id="retail-test",
		)
	)
	assert completed["status"] == "completed"
	assert "receipt_number" in completed
	assert float(completed["change_due"]) == pytest.approx(50.0, abs=0.01)


# ── 3. Cash payment returns receipt-like dict ─────────────────────────────────

def test_pos_cash_payment():
	"""process_cash_payment returns a dict with payment_id, change_due, and method."""
	svc = _pos_service()
	svc._inventory.set_price("retail-test", "SKU-MILK", 80.0)

	session = asyncio.run(
		svc.open_session(
			terminal_id="T-003",
			cashier_id="cashier-carol",
			opening_float=1000.0,
			tenant_id="retail-test",
			store_id="store-nairobi-01",
		)
	)
	txn = asyncio.run(
		svc.begin_transaction(session_id=session["id"], tenant_id="retail-test")
	)
	asyncio.run(
		svc.add_item(
			transaction_id=txn["id"],
			sku="SKU-MILK",
			quantity=1.0,
			tenant_id="retail-test",
		)
	)

	receipt = asyncio.run(
		svc.process_cash_payment(
			transaction_id=txn["id"],
			amount_tendered=100.0,
			tenant_id="retail-test",
		)
	)
	assert isinstance(receipt, dict)
	assert "payment_id" in receipt
	assert receipt["payment_method"] == "cash"
	assert float(receipt["change_due"]) == pytest.approx(100.0 - 80.0, abs=0.01)
	assert receipt["transaction_id"] == txn["id"]


# ── 4. Void transaction ───────────────────────────────────────────────────────

def test_pos_void_transaction():
	"""void_transaction returns a voided status dict."""
	svc = _pos_service()
	svc._inventory.set_price("retail-test", "SKU-SUGAR", 150.0)

	session = asyncio.run(
		svc.open_session(
			terminal_id="T-004",
			cashier_id="cashier-dan",
			opening_float=3000.0,
			tenant_id="retail-test",
			store_id="store-nairobi-01",
		)
	)
	txn = asyncio.run(
		svc.begin_transaction(session_id=session["id"], tenant_id="retail-test")
	)
	asyncio.run(
		svc.add_item(
			transaction_id=txn["id"],
			sku="SKU-SUGAR",
			quantity=1.0,
			tenant_id="retail-test",
		)
	)

	voided = asyncio.run(
		svc.void_transaction(
			transaction_id=txn["id"],
			reason="Customer changed mind",
			supervisor_id="supervisor-001",
			tenant_id="retail-test",
		)
	)
	assert isinstance(voided, dict)
	assert voided["status"] == "voided"
	assert "supervisor-001" in voided.get("notes", "") or voided.get("supervisor_override_id") == "supervisor-001"


# ── 5. Rule evaluation — allow ────────────────────────────────────────────────

def test_pos_rule_evaluation():
	"""evaluate_rules('retail_pos', {tenant_context_present: True}) returns allow."""
	from capabilities.retail.pos.capability_contract import evaluate_capability_rules

	context = {
		"tenant_id": "retail-test",
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
	}
	result = evaluate_capability_rules(context)
	assert result["decision"] == "allow", (
		f"Expected allow, got {result['decision']}. Actions: {result.get('actions', [])}"
	)
	assert result["actions"] == []


# ── 6. Manifest — 5 retail capabilities ──────────────────────────────────────

def test_retail_manifest():
	"""There are exactly 5 retail capabilities in the manifest."""
	import glob

	retail_root = os.path.join(
		os.path.dirname(__file__), "..", "capabilities", "retail"
	)
	manifests = glob.glob(os.path.join(retail_root, "*/package_manifest.json"))
	assert len(manifests) == 5, (
		f"Expected 5 retail capability manifests, found {len(manifests)}: "
		f"{[os.path.dirname(m).split('/')[-1] for m in manifests]}"
	)


# ── 7. End of day report ──────────────────────────────────────────────────────

def test_pos_eod_report():
	"""end_of_day_report returns a summary dict with net_sales and session_count keys."""
	svc = _pos_service()
	from datetime import date

	# Run a minimal sale so the report has data
	svc._inventory.set_price("retail-test", "SKU-TEA", 60.0)
	today_str = date.today().isoformat()

	session = asyncio.run(
		svc.open_session(
			terminal_id="T-005",
			cashier_id="cashier-eve",
			opening_float=1000.0,
			tenant_id="retail-test",
			store_id="store-kisumu-01",
		)
	)
	txn = asyncio.run(
		svc.begin_transaction(session_id=session["id"], tenant_id="retail-test")
	)
	asyncio.run(svc.add_item(txn["id"], "SKU-TEA", 2.0, tenant_id="retail-test"))
	total = float((await_txn := asyncio.run(
		svc.add_item(txn["id"], "SKU-TEA", 0.0, tenant_id="retail-test")
		if False else
		svc.add_item(txn["id"], "SKU-TEA", 1.0, tenant_id="retail-test")
	))["grand_total"])
	asyncio.run(
		svc.complete_transaction(
			txn["id"],
			payments=[{"method": "cash", "amount": total + 10}],
			tenant_id="retail-test",
		)
	)

	report = asyncio.run(
		svc.end_of_day_report(
			store_id="store-kisumu-01",
			report_date=today_str,
			tenant_id="retail-test",
		)
	)
	assert isinstance(report, dict)
	assert "net_sales" in report
	assert "session_count" in report
	assert "transaction_count" in report
	assert "gross_sales" in report
	assert report["store_id"] == "store-kisumu-01"
	assert report["business_date"] == today_str
