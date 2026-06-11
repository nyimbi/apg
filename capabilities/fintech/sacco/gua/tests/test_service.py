"""Tests for GuarantorService — real objects, no mocks, no @pytest.mark.asyncio."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from capabilities.fintech.sacco.gua.service import GuarantorService


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loop():
	return asyncio.get_event_loop()


@pytest.fixture
def svc():
	s = GuarantorService(tenant_id="t1")
	# seed members
	s.seed_member("m-borrower", savings=Decimal("100000"), shares=Decimal("50000"))
	s.seed_member("m-guarantor1", savings=Decimal("80000"), shares=Decimal("40000"))
	s.seed_member("m-guarantor2", savings=Decimal("60000"), shares=Decimal("30000"))
	s.seed_member("m-inactive", savings=Decimal("50000"), shares=Decimal("20000"), is_active=False)
	s.seed_member("m-defaulter", savings=Decimal("50000"), shares=Decimal("20000"), is_defaulter=True)
	# seed a loan
	s.seed_loan("loan-001", status="active", dpd=0)
	s.seed_loan("loan-overdue", status="active", dpd=45)
	s.seed_loan("loan-closed", status="closed", dpd=0)
	return s


# ── Eligibility ───────────────────────────────────────────────────────────────

def test_eligibility_pass(loop, svc):
	result = loop.run_until_complete(
		svc.check_guarantor_eligibility("t1", "m-guarantor1", Decimal("20000"))
	)
	assert result["eligible"] is True
	assert result["reasons"] == []


def test_eligibility_inactive_member(loop, svc):
	result = loop.run_until_complete(
		svc.check_guarantor_eligibility("t1", "m-inactive", Decimal("10000"))
	)
	assert result["eligible"] is False
	assert any("not_active" in r for r in result["reasons"])


def test_eligibility_defaulter(loop, svc):
	result = loop.run_until_complete(
		svc.check_guarantor_eligibility("t1", "m-defaulter", Decimal("10000"))
	)
	assert result["eligible"] is False
	assert any("defaulter" in r for r in result["reasons"])


def test_eligibility_insufficient_savings(loop, svc):
	# guarantor1 has 80000 savings; requesting 90000 guarantee needs 100% cover
	result = loop.run_until_complete(
		svc.check_guarantor_eligibility("t1", "m-guarantor1", Decimal("90000"))
	)
	assert result["eligible"] is False
	assert any("insufficient_savings" in r for r in result["reasons"])


def test_eligibility_exposure_exceeded(loop, svc):
	# shares=40000; max = 3*40000=120000; requesting 130000 → exceeds
	result = loop.run_until_complete(
		svc.check_guarantor_eligibility("t1", "m-guarantor1", Decimal("130000"))
	)
	assert result["eligible"] is False


# ── Request lifecycle ─────────────────────────────────────────────────────────

def test_request_guarantee_creates_pending(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("20000"), "Please help me")
	)
	assert req["status"] == "pending"
	assert req["loan_id"] == "loan-001"
	assert req["guarantor_member_id"] == "m-guarantor1"


def test_request_guarantee_ineligible_fails(loop, svc):
	with pytest.raises(ValueError, match="guarantor_ineligible"):
		loop.run_until_complete(
			svc.request_guarantee("t1", "loan-001", "m-inactive", Decimal("5000"))
		)


def test_accept_guarantee_freezes_savings(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("20000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	assert gua["status"] == "active"
	assert Decimal(gua["frozen_amount"]) == Decimal("20000")


def test_accept_requires_pin(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	with pytest.raises(ValueError, match="pin_verification_required"):
		loop.run_until_complete(
			svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=False)
		)


def test_accept_wrong_guarantor_blocked(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	with pytest.raises(PermissionError):
		loop.run_until_complete(
			svc.accept_guarantee("t1", req["id"], "m-guarantor2", pin_verified=True)
		)


def test_decline_guarantee(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	result = loop.run_until_complete(
		svc.decline_guarantee("t1", req["id"], "m-guarantor1", "Cannot commit")
	)
	assert result["status"] == "declined"
	assert result["decline_reason"] == "Cannot commit"


def test_cancel_request(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	result = loop.run_until_complete(
		svc.cancel_guarantee_request("t1", req["id"], "officer-1", "Wrong loan")
	)
	assert result["status"] == "cancelled"


# ── Release ───────────────────────────────────────────────────────────────────

def test_release_guarantee_unfreezes(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("15000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	released = loop.run_until_complete(
		svc.release_guarantee("t1", gua["id"], "loan_repaid", "system")
	)
	assert released["status"] == "released"
	assert Decimal(released["frozen_amount"]) == Decimal("0")


# ── Call guarantee ────────────────────────────────────────────────────────────

def test_call_guarantee_deducts_savings_and_posts_gl(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("30000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	result = loop.run_until_complete(
		svc.call_guarantee("t1", gua["id"], Decimal("10000"), "Default on loan-001")
	)
	assert result["guarantee"]["status"] == "called"
	assert Decimal(result["guarantee"]["amount_called"]) == Decimal("10000")
	assert result["gl_entry"]["debit_account"] == "Guarantor Savings"
	assert result["gl_entry"]["credit_account"] == "Loan Recovery"


def test_call_exceeds_frozen_fails(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	with pytest.raises(ValueError, match="call_exceeds_frozen"):
		loop.run_until_complete(
			svc.call_guarantee("t1", gua["id"], Decimal("15000"), "Exceeds frozen")
		)


# ── Exposure ──────────────────────────────────────────────────────────────────

def test_exposure_reflects_active_guarantees(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("25000"))
	)
	loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	exposure = loop.run_until_complete(svc.get_guarantor_exposure("t1", "m-guarantor1"))
	assert Decimal(exposure["total_guaranteed"]) == Decimal("25000")
	assert Decimal(exposure["frozen_savings"]) == Decimal("25000")


def test_exposure_limit_override(loop, svc):
	loop.run_until_complete(
		svc.set_exposure_limit("t1", "m-guarantor1", Decimal("200000"), "admin")
	)
	exposure = loop.run_until_complete(svc.get_guarantor_exposure("t1", "m-guarantor1"))
	assert Decimal(exposure["max_exposure_limit"]) == Decimal("200000")


# ── Substitute guarantor ──────────────────────────────────────────────────────

def test_substitute_guarantor(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("20000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	result = loop.run_until_complete(
		svc.substitute_guarantor("t1", gua["id"], "m-guarantor2", "Moving abroad", "officer-1")
	)
	assert result["released_guarantee"]["status"] == "substituted"
	assert result["new_request"]["guarantor_member_id"] == "m-guarantor2"


# ── At-risk ───────────────────────────────────────────────────────────────────

def test_at_risk_guarantees(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-overdue", "m-guarantor1", Decimal("20000"))
	)
	loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	at_risk = loop.run_until_complete(svc.get_at_risk_guarantees("t1"))
	assert len(at_risk) >= 1
	assert all(svc._loan_dpd.get(g["loan_id"], 0) > 30 for g in at_risk)


# ── Automatic releases ────────────────────────────────────────────────────────

def test_automatic_release_closed_loan(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-closed", "m-guarantor1", Decimal("10000"))
	)
	gua = loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	# Loan is already seeded as "closed"
	result = loop.run_until_complete(svc.process_automatic_releases("t1"))
	assert result["released_count"] >= 1
	# Check the guarantee is now released
	updated = svc._guarantees[gua["id"]]
	assert updated["status"] == "released"


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def test_portfolio_metrics(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("20000"))
	)
	loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	metrics = loop.run_until_complete(svc.get_guarantee_portfolio_metrics("t1"))
	assert metrics["total_active_guarantees"] >= 1
	assert Decimal(metrics["total_exposure"]) >= Decimal("20000")


# ── History ───────────────────────────────────────────────────────────────────

def test_guarantor_history(loop, svc):
	req = loop.run_until_complete(
		svc.request_guarantee("t1", "loan-001", "m-guarantor1", Decimal("10000"))
	)
	loop.run_until_complete(
		svc.accept_guarantee("t1", req["id"], "m-guarantor1", pin_verified=True)
	)
	history = loop.run_until_complete(svc.get_guarantor_history("t1", "m-guarantor1"))
	assert len(history["requests"]) >= 1
	assert len(history["guarantees"]) >= 1


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_check(loop, svc):
	result = loop.run_until_complete(svc.health_check())
	assert result["status"] == "healthy"
	assert result["service"] == "fintech_sacco_gua"
