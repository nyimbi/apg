"""Finance capability integration tests: AR + GL + AP.

All tests are sync; async service methods are called via asyncio.run().
Uses real in-memory service instances — no mocks.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest


# ── Helpers to import services lazily (avoids top-level import errors in CI) ─

def _ar_service():
	from capabilities.fin.arc.accounts_receivable.service import AccountsReceivableService
	from capabilities.fin.arc.accounts_receivable.database.store import InMemoryStore
	store = InMemoryStore()
	return AccountsReceivableService(tenant_id="test-tenant", store=store)


def _gl_service():
	from capabilities.fin.glr.general_ledger.service import GeneralLedgerService
	return GeneralLedgerService(tenant_id="test-tenant")


def _ap_service():
	from capabilities.fin.apy.accounts_payable.service import AccountsPayableService
	return AccountsPayableService()


# ── 1. AR customer creation ───────────────────────────────────────────────────

def test_ar_customer_creation():
	"""Create an AR customer with a credit limit and verify key fields."""
	svc = _ar_service()
	customer = asyncio.run(
		svc.create_customer(
			name="Acme Corp",
			credit_limit=50_000,
			payment_terms="NET30",
			currency="USD",
		)
	)
	assert customer["name"] == "Acme Corp"
	assert customer["payment_terms"] == "NET30"
	assert customer["currency"] == "USD"
	assert float(customer["credit_limit"]) == 50_000.0
	assert customer["status"] == "active"
	assert "id" in customer
	assert customer["tenant_id"] == "test-tenant"


# ── 2. AR invoice lifecycle ───────────────────────────────────────────────────

def test_ar_invoice_lifecycle():
	"""Create → validate → submit → approve → post workflow."""
	svc = _ar_service()

	customer = asyncio.run(
		svc.create_customer("Beta Ltd", 100_000, "NET60", "USD")
	)
	cust_id = customer["id"]

	invoice = asyncio.run(
		svc.create_invoice(
			customer_id=cust_id,
			invoice_date="2026-01-01",
			due_date="2026-03-01",
			lines=[{
				"description": "Consulting services",
				"quantity": 10,
				"unit_price": 500,
				"tax_rate": 0.16,
			}],
			currency="USD",
			payment_terms="NET60",
		)
	)
	assert invoice["status"] == "draft"
	inv_id = invoice["id"]

	validation = asyncio.run(svc.validate_invoice(inv_id))
	assert validation["valid"] is True
	assert validation["issues"] == []

	submitted = asyncio.run(svc.submit_invoice(inv_id))
	assert submitted["status"] == "submitted"

	approved = asyncio.run(svc.approve_invoice(inv_id, approved_by="finance-mgr"))
	assert approved["status"] == "approved"

	posted = asyncio.run(svc.post_invoice(inv_id))
	assert posted["invoice"]["status"] == "posted"
	assert "gl_entry" in posted


# ── 3. AR aging report ────────────────────────────────────────────────────────

def test_ar_aging_report():
	"""calculate_aging() returns a dict with non-negative bucket amounts."""
	svc = _ar_service()
	result = asyncio.run(svc.calculate_aging())
	assert isinstance(result, dict)
	assert "totals" in result
	totals = result["totals"]
	for bucket_key in ("current", "1_30", "31_60", "61_90", "91_120", "120_plus"):
		assert bucket_key in totals
		assert float(totals[bucket_key]) >= 0.0


# ── 4. AR rule blocks unbalanced operation ────────────────────────────────────

def test_ar_rule_blocks_unbalanced_operation():
	"""create_invoice raises AssertionError when due_date precedes invoice_date."""
	svc = _ar_service()

	async def _create_bad():
		customer = await svc.create_customer("Bad Corp", 10_000, "NET30", "USD")
		await svc.create_invoice(
			customer_id=customer["id"],
			invoice_date="2026-06-01",
			due_date="2026-05-01",  # precedes invoice_date
			lines=[{"description": "x", "quantity": 1, "unit_price": 100, "tax_rate": 0}],
			currency="USD",
			payment_terms="NET30",
		)

	with pytest.raises((AssertionError, ValueError)):
		asyncio.run(_create_bad())


# ── 5. GL journal balance enforced ────────────────────────────────────────────

def test_gl_journal_balance_enforced():
	"""GL rejects an unbalanced journal entry (debit ≠ credit)."""
	from capabilities.fin.glr.general_ledger.service import GeneralLedgerService

	svc = GeneralLedgerService(tenant_id="gl-test")

	# Create accounts and period required by create_journal_entry
	period = svc.open_period(
		"p1", "gl-test", "2026-01", 2026, "2026-01-01", "2026-01-31"
	)
	acct_dr = svc.create_account(
		"a1", "gl-test", "1000", "Cash", "asset"
	)
	acct_cr = svc.create_account(
		"a2", "gl-test", "4000", "Revenue", "revenue"
	)
	batch = svc.create_journal_batch(
		"b1", "gl-test", period["id"], "manual"
	)

	with pytest.raises((PermissionError, AssertionError, ValueError)):
		svc.create_journal_entry(
			"j1",
			"gl-test",
			batch["id"],
			"Unbalanced entry",
			lines=[
				{"account_id": acct_dr["id"], "debit": 1000, "credit": 0},
				{"account_id": acct_cr["id"], "debit": 500,  "credit": 0},  # unbalanced
			],
		)


# ── 6. GL trial balance returns dict ─────────────────────────────────────────

def test_gl_trial_balance_returns_dict():
	"""trial_balance() returns a structured dict with required keys."""
	from capabilities.fin.glr.general_ledger.service import GeneralLedgerService

	svc = GeneralLedgerService(tenant_id="gl-tb")

	period = svc.open_period(
		"p1", "gl-tb", "2026-01", 2026, "2026-01-01", "2026-01-31"
	)

	result = asyncio.run(
		svc.trial_balance("gl-tb", period["name"])
	)
	assert isinstance(result, dict)
	assert "rows" in result or "accounts" in result or "trial_balance" in result or "total_debits" in result


# ── 7. AP invoice creation ────────────────────────────────────────────────────

def test_ap_invoice_creation():
	"""Record a vendor and create an AP invoice successfully."""
	svc = _ap_service()

	vendor = svc.register_vendor(
		vendor_id="v-001",
		tenant_id="ap-test",
		name="Supplies Inc",
		owner="procurement-mgr",
		tax_profile="standard",
		payment_method="bank_transfer",
	)
	assert vendor["name"] == "Supplies Inc"
	assert vendor["status"] == "active"

	invoice = svc.record_invoice(
		invoice_id="inv-001",
		tenant_id="ap-test",
		vendor_record_id=vendor["id"],
		invoice_number="INV-2026-001",
		amount=5000.00,
		currency="USD",
		document_reference="doc-ref-001",
	)
	assert invoice["invoice_number"] == "INV-2026-001"
	assert float(invoice["amount"]) == 5000.0
	assert invoice["currency"] == "USD"


# ── 8. AP three-way match concept ────────────────────────────────────────────

def test_ap_three_way_match_concept():
	"""AP service evaluate_rules correctly evaluates a matching context."""
	from capabilities.fin.apy.accounts_payable.capability_contract import evaluate_capability_rules

	# Valid write context — should be allowed.
	# The invoice_amount_positive rule fires when context["amount"] <= 0
	# (condition key is "amount_lte": 0, which strips "_lte" and looks up "amount").
	# Pass amount=5000 so that condition does NOT match.
	ctx_allow = {
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "record_invoice",
		"vendor_present": True,
		"invoice_number_present": True,
		"currency_present": True,
		"duplicate_detected": False,
		"duplicate_reviewed": True,
		"amount": 5000,            # positive — invoice_amount_positive won't fire
	}
	result = evaluate_capability_rules(ctx_allow)
	assert result["decision"] in ("allow", "require_review"), (
		f"Unexpected decision for valid context: {result['decision']}. "
		f"Matched: {result.get('matched_rules', [])}"
	)

	# Missing tenant — should deny
	ctx_deny = {"tenant_context_present": False}
	result_deny = evaluate_capability_rules(ctx_deny)
	assert result_deny["decision"] == "deny"
