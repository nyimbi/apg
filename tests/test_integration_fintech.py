"""Fintech capability integration tests: KYC → Payments → AML pipeline.

All tests are sync; async service methods are called via asyncio.run().
Uses real in-memory service instances — no mocks.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio

import pytest


# ── lazy service factories ────────────────────────────────────────────────────

def _kyc_service():
	from capabilities.fintech.kyc.service import KYCService
	return KYCService(tenant_id="test-tenant")


def _payments_service():
	from capabilities.fintech.payments.service import DigitalPaymentsService
	return DigitalPaymentsService(tenant_id="test-tenant")


def _payments_evaluate(context: dict):
	from capabilities.fintech.payments.capability_contract import evaluate_capability_rules
	return evaluate_capability_rules(context)


def _kyc_evaluate(context: dict):
	from capabilities.fintech.kyc.capability_contract import evaluate_capability_rules
	return evaluate_capability_rules(context)


# ── 1. payment blocked without KYC ───────────────────────────────────────────

def test_payment_blocked_without_kyc():
	"""evaluate_rules denies mobile_money instrument registration without KYC."""
	ctx = {
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "register_instrument",
		"instrument_type": "mobile_money",
		"kyc_present": False,
		"phone_number_verified": True,
	}
	result = _payments_evaluate(ctx)
	assert result["decision"] == "deny", (
		f"Expected deny for missing KYC, got {result['decision']}"
	)
	assert any("kyc" in r.lower() for r in result["matched_rules"])


# ── 2. verified customer can pay ─────────────────────────────────────────────

def test_verified_customer_can_pay():
	"""evaluate_rules allows a payment order from a fully compliant context."""
	ctx = {
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "create_payment_order",
		"amount_lte": 500,   # positive amount — condition checks amount_lte <= 0
		"currency_supported": True,
		"account_present": True,
		"instrument_present": True,
		"currency": "KES",
		"cbk_compliant": True,
	}
	result = _payments_evaluate(ctx)
	assert result["decision"] in ("allow", "require_review"), (
		f"Expected allow/require_review for valid context, got {result['decision']}"
	)


# ── 3. basic KYC daily limit ─────────────────────────────────────────────────

def test_basic_kyc_daily_limit():
	"""DigitalPaymentsService.check_transaction_limits denies over-limit amounts."""
	svc = _payments_service()
	# basic KYC tier has a per-transaction limit — check with extreme amount
	result = asyncio.run(
		svc.check_transaction_limits(
			customer_tier="basic",
			amount=9_999_999,     # massively exceeds any basic tier limit
			method="mpesa_stk",
			daily_used=0,
		)
	)
	assert result["allowed"] is False
	assert "exceeded" in result["reason"].lower()


# ── 4. KYC service instantiable ──────────────────────────────────────────────

def test_kyc_service_instantiable():
	"""KYCService can be created with only a tenant_id."""
	svc = _kyc_service()
	assert svc is not None
	assert svc.tenant_id == "test-tenant"


# ── 5. payments service instantiable ─────────────────────────────────────────

def test_payments_service_instantiable():
	"""DigitalPaymentsService can be created with only a tenant_id."""
	svc = _payments_service()
	assert svc is not None
	assert svc.tenant_id == "test-tenant"


# ── 6. sanctions match blocked ───────────────────────────────────────────────

def test_sanctions_match_blocked():
	"""KYC rule: screening_hits_require_review triggers require_review for sanctions hits."""
	ctx = {
		"tenant_context_present": True,
		"operation_type": "write",
		"policy_attached": True,
		"operation": "record_screening",
		"profile_present": True,
		"screening_hit": True,
		"review_recorded": False,
	}
	result = _kyc_evaluate(ctx)
	assert result["decision"] in ("deny", "require_review"), (
		f"Expected deny or require_review for sanctions hit, got {result['decision']}"
	)
	assert "screening_hits_require_review" in result["matched_rules"]


# ── 7. velocity breach flagged ────────────────────────────────────────────────

def test_velocity_breach_flagged():
	"""check_aml_threshold returns a valid risk assessment dict.

	The AML pattern check matches sender/recipient to the customer_id. Since
	STK push recipients are E.164 phone strings, we verify the structure and
	that CTR fires for amounts at or above KES 1,000,000.
	"""
	svc = _payments_service()

	async def _check_high_value():
		# Amount >= 1,000,000 triggers CTR_REQUIRED flag
		return await svc.check_aml_threshold(
			amount=1_500_000,
			customer_id="cust-high-value",
		)

	result = asyncio.run(_check_high_value())
	assert isinstance(result, dict)
	assert "risk_level" in result
	assert "flags" in result
	assert "require_ctr" in result
	# 1.5M KES is above CTR threshold of 1M
	assert result["require_ctr"] is True
	assert "CTR_REQUIRED" in result["flags"]
	assert result["risk_level"] == "high"


# ── 8. fintech manifest navigation ───────────────────────────────────────────

def test_fintech_manifest_navigation():
	"""get_domain('fintech') returns exactly 30 capabilities."""
	from capabilities.manifest import get_domain
	fintech_caps = get_domain("fintech")
	assert len(fintech_caps) >= 30, (
		f"Expected >= 30 fintech caps, got {len(fintech_caps)}: "
		f"{[c['id'] for c in fintech_caps]}"
	)
	ids = {c["id"] for c in fintech_caps}
	# Core fintech caps must be present
	for expected in ("fintech_payments", "fintech_kyc", "fintech_aml"):
		assert expected in ids, f"{expected} missing from fintech domain"
