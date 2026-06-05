"""Telecom capability integration tests: Billing (bil).

All tests use real in-memory service instances — no mocks.
Async service methods are called via asyncio.run().
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest


def _bil_service(tenant_id: str = "telecom-test") -> "TelecomBillingService":
	from capabilities.telecom.bil.service import TelecomBillingService
	return TelecomBillingService(tenant_id=tenant_id, actor_id="test-actor")


# ── 1. Generate bill ──────────────────────────────────────────────────────────

def test_billing_subscriber_bill():
	"""generate_bill returns a bill dict with invoice_id and status='draft'."""
	svc = _bil_service()

	# Seed a charge so the bill has a non-zero amount
	svc.record_charge(
		charge_id="chg-001",
		customer_id="sub-001",
		charge_type="usage_based",
		rating_type="flat_rate",
		amount=450.00,
		currency="KES",
		tax_amount=72.00,
	)

	bill = asyncio.run(
		svc.generate_bill(
			account_id="sub-001",
			billing_period={"start": "2026-05-01", "end": "2026-05-31"},
		)
	)
	assert isinstance(bill, dict)
	assert "invoice_id" in bill
	assert bill["account_id"] == "sub-001"
	assert bill["status"] == "draft"
	assert bill["tenant_id"] == "telecom-test"
	assert float(bill["subtotal"]) >= 450.0
	assert bill["currency"] == "KES"


# ── 2. Payment processing ─────────────────────────────────────────────────────

def test_billing_payment():
	"""payment_processing records a payment and returns a payment_id."""
	svc = _bil_service()

	result = asyncio.run(
		svc.payment_processing(
			account_id="sub-002",
			amount=1200.00,
			payment_method="mobile_money",
			reference="QJK3XABCDE",
		)
	)
	assert isinstance(result, dict)
	assert "payment_id" in result
	assert result["account_id"] == "sub-002"
	assert result["status"] == "received"
	assert result["payment_method"] == "mobile_money"
	assert float(result["amount"]) == pytest.approx(1200.0, abs=0.01)

	# Balance should have been updated
	from decimal import Decimal
	balance = svc._balances["sub-002"]["main_balance"]
	assert balance == Decimal("1200.00")


# ── 3. Dunning workflow escalation ────────────────────────────────────────────

def test_billing_dunning():
	"""dunning_workflow escalates correctly based on days-past-due."""
	svc = _bil_service()

	# 5 DPD → reminder_1
	r1 = asyncio.run(svc.dunning_workflow(account_id="sub-010", dpd_days=5))
	assert r1["dunning_step"] == "reminder_1"
	assert r1["suspended"] is False

	# 25 DPD → service_suspended
	r2 = asyncio.run(svc.dunning_workflow(account_id="sub-011", dpd_days=25))
	assert r2["dunning_step"] == "service_suspended"
	assert r2["suspended"] is True
	assert "sub-011" in svc._suspended_accounts

	# 35 DPD → legal_notice
	r3 = asyncio.run(svc.dunning_workflow(account_id="sub-012", dpd_days=35))
	assert r3["dunning_step"] == "legal_notice"
	assert r3["suspended"] is False  # legal_notice step does not set suspended flag


# ── 4. Billing dispute ────────────────────────────────────────────────────────

def test_billing_dispute():
	"""raise_billing_dispute returns a dispute dict with dispute_id and status='open'."""
	svc = _bil_service()

	dispute = asyncio.run(
		svc.raise_billing_dispute(
			account_id="sub-020",
			invoice_id="inv-20260501",
			disputed_amount=350.00,
			reason="Charged for calls I did not make on 2026-05-15.",
		)
	)
	assert isinstance(dispute, dict)
	assert "dispute_id" in dispute
	assert dispute["account_id"] == "sub-020"
	assert dispute["status"] == "open"
	assert dispute["tenant_id"] == "telecom-test"
	assert float(dispute["disputed_amount"]) == pytest.approx(350.0, abs=0.01)
	assert dispute["currency"] == "KES"

	# Dispute should be stored
	assert dispute["dispute_id"] in svc._disputes


# ── 5. Rule evaluation — allow ────────────────────────────────────────────────

def test_billing_rule_evaluation():
	"""evaluate_rules('telecom_bil', {tenant_context_present: True}) returns allow."""
	from capabilities.telecom.bil.capability_contract import evaluate_capability_rules

	context = {
		"tenant_id": "telecom-test",
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
	}
	result = evaluate_capability_rules(context)
	assert result["decision"] == "allow", (
		f"Expected allow, got {result['decision']}. Actions: {result.get('actions', [])}"
	)
	assert result["actions"] == []


# ── 6. Manifest — 10 telecom capabilities ────────────────────────────────────

def test_telecom_manifest():
	"""There are exactly 10 telecom capabilities in the manifest."""
	import glob

	telecom_root = os.path.join(
		os.path.dirname(__file__), "..", "capabilities", "telecom"
	)
	manifests = glob.glob(os.path.join(telecom_root, "*/package_manifest.json"))
	assert len(manifests) == 10, (
		f"Expected 10 telecom capability manifests, found {len(manifests)}: "
		f"{[os.path.dirname(m).split('/')[-1] for m in manifests]}"
	)


# ── 7. Composability — all telecom requires satisfied ────────────────────────

def test_telecom_composability():
	"""Every telecom capability has a capability_contract.py and requires common infra."""
	import glob

	telecom_root = os.path.join(
		os.path.dirname(__file__), "..", "capabilities", "telecom"
	)
	manifests = glob.glob(os.path.join(telecom_root, "*/package_manifest.json"))
	assert manifests, "No telecom manifests found"

	for manifest_path in manifests:
		cap_dir = os.path.dirname(manifest_path)
		cap_code = os.path.basename(cap_dir)
		contract_path = os.path.join(cap_dir, "capability_contract.py")
		assert os.path.isfile(contract_path), (
			f"capability_contract.py missing for telecom/{cap_code}"
		)

	# Verify billing contract exposes requires with common infra
	from capabilities.telecom.bil.capability_contract import get_capability_contract
	contract = get_capability_contract("telecom-test")
	requires = contract.get("requires", [])
	assert isinstance(requires, list)
	assert len(requires) > 0, "telecom_bil should require at least one capability"
	for common in ("auth", "audl", "mten"):
		assert common in requires, (
			f"telecom_bil should require '{common}', got {requires}"
		)
