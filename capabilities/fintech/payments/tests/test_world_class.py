"""Tests for the 10 world-class improvement methods.

All async — run with asyncio.run(), no @pytest.mark.asyncio decorators.
"""
from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from capabilities.fintech.payments.service import DigitalPaymentsService
from capabilities.fintech.payments.domain.calculations import (
	semantic_duplicate_score,
	float_exhaustion_eta,
	optimal_payment_route,
	behavioral_limit_multiplier,
	fx_rate_lock,
	chargeback_win_probability,
	classify_batch_failure,
	intraday_settlement_schedule,
)
from capabilities.fintech.payments.domain.rules import calculate_ctr_obligation


def run(coro):
	return asyncio.run(coro)


def svc(tenant: str = "wc-tenant") -> DigitalPaymentsService:
	return DigitalPaymentsService(tenant_id=tenant, actor_id="test")


# ─────────────────────────────────────────────────────────────
# 1. Semantic duplicate detection
# ─────────────────────────────────────────────────────────────

def test_semantic_duplicate_score_identical():
	score = semantic_duplicate_score(
		"INV-001", "INV-001",
		"254712345678", "254712345678",
		Decimal("1000"), Decimal("1000"),
		seconds_apart=10,
	)
	assert score > 0.85


def test_semantic_duplicate_score_different_phone():
	score = semantic_duplicate_score(
		"INV-001", "INV-001",
		"254712345678", "254700000001",
		Decimal("1000"), Decimal("1000"),
		seconds_apart=10,
	)
	assert score == 0.0


def test_semantic_duplicate_score_different_amount():
	score = semantic_duplicate_score(
		"INV-001", "INV-001",
		"254712345678", "254712345678",
		Decimal("1000"), Decimal("2000"),
		seconds_apart=10,
	)
	assert score == 0.0


def test_semantic_duplicate_score_outside_window():
	score = semantic_duplicate_score(
		"INV-001", "INV-001",
		"254712345678", "254712345678",
		Decimal("1000"), Decimal("1000"),
		seconds_apart=400,
		window=300,
	)
	assert score == 0.0


def test_semantic_duplicate_check_service():
	s = svc()
	# Create a transaction first
	txn = run(s.mpesa_stk_push("254712345678", Decimal("500"), "SEM-DUP-001"))
	result = run(s.semantic_duplicate_check(
		reference="SEM-DUP-001",
		amount=Decimal("500"),
		phone=txn.get("recipient", "254712345678"),
		window_seconds=300,
	))
	assert isinstance(result, dict)
	assert "is_duplicate" in result
	assert "score" in result


# ─────────────────────────────────────────────────────────────
# 2. Float exhaustion forecasting
# ─────────────────────────────────────────────────────────────

def test_float_exhaustion_eta_requires_topup():
	result = float_exhaustion_eta(
		current_float=Decimal("100000"),
		burn_rate_per_hour=Decimal("50000"),
		pending_batch_total=Decimal("150000"),
	)
	assert result["requires_topup"] is True
	assert Decimal(result["recommended_topup"]) > 0


def test_float_exhaustion_eta_sufficient():
	result = float_exhaustion_eta(
		current_float=Decimal("500000"),
		burn_rate_per_hour=Decimal("10000"),
		pending_batch_total=Decimal("100000"),
	)
	assert result["requires_topup"] is False
	assert Decimal(result["recommended_topup"]) == Decimal("0")


def test_float_exhaustion_eta_zero_burn_rate():
	result = float_exhaustion_eta(
		current_float=Decimal("100000"),
		burn_rate_per_hour=Decimal("0"),
		pending_batch_total=Decimal("50000"),
	)
	assert result["eta_hours"] == 999.0


def test_forecast_float_service():
	s = svc()
	result = run(s.forecast_float(current_float=Decimal("200000"), lookback_hours=24))
	assert "eta_hours" in result
	assert "requires_topup" in result
	assert "burn_rate_per_hour" in result


# ─────────────────────────────────────────────────────────────
# 3. Auto CTR filing
# ─────────────────────────────────────────────────────────────

def test_ctr_cbk_threshold():
	result = calculate_ctr_obligation(Decimal("1500000"), "KES")
	assert result["requires_ctr"] is True
	assert result["regulator"] == "CBK"


def test_ctr_cbn_threshold():
	result = calculate_ctr_obligation(Decimal("6000000"), "NGN")
	assert result["requires_ctr"] is True
	assert result["regulator"] == "CBN"


def test_ctr_below_threshold():
	result = calculate_ctr_obligation(Decimal("500000"), "KES")
	assert result["requires_ctr"] is False


def test_auto_file_ctr_not_triggered():
	s = svc()
	txn = run(s.mpesa_stk_push("254712345678", Decimal("500"), "CTR-SMALL"))
	run(s.confirm_payment(txn["id"], "CONF-001"))
	result = run(s.auto_file_ctr(txn["id"]))
	assert result["filed"] is False


def test_auto_file_ctr_triggered():
	"""Create a synthetic high-value transaction and auto-file CTR."""
	s = svc("ctr-tenant")
	# Manually insert a high-value completed transaction
	from capabilities.fintech.payments.models import (
		PaymentTransaction, PaymentMethod, CurrencyCode, PaymentStatus, TransactionType, uuid7str
	)
	from decimal import Decimal as D
	txn = PaymentTransaction(
		id=uuid7str(),
		tenant_id="ctr-tenant",
		order_id=uuid7str(),
		transaction_type=TransactionType.payment,
		method=PaymentMethod.bank_eft,
		amount=D("2000000"),
		currency=CurrencyCode.KES,
		status=PaymentStatus.completed,
		recipient="TEST-RECIPIENT",
		reference="CTR-HIGH-VALUE",
	)
	d = s._txn_dict(txn)
	run(s._save("payments_transactions", d))
	result = run(s.auto_file_ctr(txn.id))
	assert result["filed"] is True
	assert result["regulator"] == "CBK"


# ─────────────────────────────────────────────────────────────
# 4. Optimal payment routing
# ─────────────────────────────────────────────────────────────

def test_optimal_route_cheapest_first():
	routes = optimal_payment_route(
		amount=Decimal("5000"),
		recipient_capabilities=["mpesa", "airtel", "bank_eft"],
		currency="KES",
		priority="cost",
	)
	assert len(routes) >= 2
	# First route should have lowest cost_score
	assert routes[0]["cost_score"] <= routes[1]["cost_score"]


def test_optimal_route_fastest_first():
	routes = optimal_payment_route(
		amount=Decimal("5000"),
		recipient_capabilities=["mpesa", "bank_eft"],
		currency="KES",
		priority="speed",
	)
	# M-Pesa (30s) should rank above bank EFT (3600s)
	assert routes[0]["method"] == "mpesa_stk"


def test_optimal_route_most_reliable_first():
	routes = optimal_payment_route(
		amount=Decimal("50000"),
		recipient_capabilities=["mpesa", "bank_eft"],
		currency="KES",
		priority="reliability",
	)
	assert routes[0]["method"] == "bank_eft"


def test_get_optimal_route_service():
	s = svc()
	result = run(s.get_optimal_route(
		amount=Decimal("10000"),
		currency="KES",
		recipient_capabilities=["mpesa", "airtel"],
		priority="cost",
	))
	assert "recommended" in result
	assert "all_routes" in result
	assert result["recommended"] is not None


# ─────────────────────────────────────────────────────────────
# 5. Velocity-adaptive limits
# ─────────────────────────────────────────────────────────────

def test_behavioral_multiplier_clean_history():
	result = behavioral_limit_multiplier(
		account_age_days=360,
		total_txn_count=200,
		success_rate=0.99,
		dispute_rate=0.001,
		aml_flags=0,
		kyc_tier="basic",
	)
	assert Decimal(result["multiplier"]) > Decimal("1.0")


def test_behavioral_multiplier_aml_flag():
	result = behavioral_limit_multiplier(
		account_age_days=180,
		total_txn_count=50,
		success_rate=0.95,
		dispute_rate=0.001,
		aml_flags=2,
		kyc_tier="basic",
	)
	assert result["multiplier"] == "0.5"
	assert result["reviewable"] is True


def test_behavioral_multiplier_poor_success_rate():
	result = behavioral_limit_multiplier(
		account_age_days=90,
		total_txn_count=30,
		success_rate=0.80,
		dispute_rate=0.0,
		aml_flags=0,
		kyc_tier="standard",
	)
	assert Decimal(result["multiplier"]) < Decimal("1.5")


def test_get_dynamic_limit_service():
	s = svc()
	result = run(s.get_dynamic_limit(customer_id="cust-001", kyc_tier="basic"))
	assert "multiplier" in result
	assert "effective_daily_limit" in result
	assert "customer_id" in result


# ─────────────────────────────────────────────────────────────
# 6. FX rate lock
# ─────────────────────────────────────────────────────────────

def test_fx_rate_lock_structure():
	result = fx_rate_lock("USD", "KES", Decimal("100"), lock_duration_seconds=300)
	assert "lock_id" in result
	assert "locked_rate" in result
	assert "expires_at" in result
	assert "guaranteed_to_amount" in result
	assert Decimal(result["guaranteed_to_amount"]) > Decimal("10000")


def test_fx_rate_lock_same_currency():
	result = fx_rate_lock("KES", "KES", Decimal("1000"), lock_duration_seconds=60)
	assert Decimal(result["locked_rate"]) == Decimal("1")
	assert Decimal(result["guaranteed_to_amount"]) == Decimal("1000")


def test_lock_fx_rate_service():
	s = svc()
	result = run(s.lock_fx_rate("USD", "KES", Decimal("500")))
	assert "lock_id" in result
	assert "expires_at" in result
	assert Decimal(result["to_amount"]) > Decimal("50000")


# ─────────────────────────────────────────────────────────────
# 7. Chargeback win probability
# ─────────────────────────────────────────────────────────────

def test_chargeback_high_probability_with_3ds():
	result = chargeback_win_probability(
		three_ds_result="Y",
		avs_result="Y",
		cvv_result="M",
		customer_txn_history_count=10,
		minutes_since_txn=60.0,
		dispute_reason="unauthorised",
	)
	assert Decimal(result["win_probability"]) >= Decimal("0.70")
	assert result["recommended_action"] == "contest"


def test_chargeback_low_probability_no_evidence():
	result = chargeback_win_probability(
		three_ds_result=None,
		avs_result="N",
		cvv_result="N",
		customer_txn_history_count=0,
		minutes_since_txn=2.0,
		dispute_reason="item_not_received",
	)
	assert Decimal(result["win_probability"]) < Decimal("0.70")


def test_score_chargeback_service():
	s = svc()
	txn = run(s.mpesa_stk_push("254712345678", Decimal("3000"), "CB-SCORE-001"))
	run(s.confirm_payment(txn["id"], "CONF-CB-001"))
	dispute = run(s.raise_dispute(txn["id"], "unauthorised", "Test dispute for scoring"))
	result = run(s.score_chargeback(dispute["id"], three_ds_result="Y", avs_result="Y", cvv_result="M"))
	assert "win_probability" in result
	assert "recommended_action" in result


# ─────────────────────────────────────────────────────────────
# 8. Batch failure recovery
# ─────────────────────────────────────────────────────────────

def test_classify_batch_failure_retry():
	result = classify_batch_failure("network_timeout", Decimal("5000"), "254712345678")
	assert result["action"] == "retry"
	assert result["auto_recoverable"] is True
	assert "backoff_ms" in result["patched_params"]


def test_classify_batch_failure_reroute():
	result = classify_batch_failure("mpesa_insufficient_float", Decimal("50000"), "254712345678")
	assert result["action"] == "reroute"
	assert result["auto_recoverable"] is True


def test_classify_batch_failure_split():
	result = classify_batch_failure("kyc_per_txn_limit_exceeded", Decimal("300000"), "254712345678", "basic")
	assert result["action"] == "split"
	assert "split_amounts" in result["patched_params"]


def test_classify_batch_failure_escalate():
	result = classify_batch_failure("account_suspended", Decimal("5000"), "254712345678")
	assert result["action"] == "escalate"
	assert result["auto_recoverable"] is False


def test_recover_batch_failures_service():
	s = svc()
	payment_list = [
		{"phone": "254712000001", "amount": Decimal("500"), "reference": "r1", "method": "mpesa_b2c"},
	]
	batch = run(s.create_bulk_payment_batch("recovery-test", payment_list))
	run(s.validate_bulk_batch(batch["id"]))
	run(s.process_bulk_batch(batch["id"]))
	result = run(s.recover_batch_failures(batch["id"]))
	assert "batch_id" in result
	assert "total_failed" in result
	assert "auto_recovered" in result


# ─────────────────────────────────────────────────────────────
# 9. Intraday settlement
# ─────────────────────────────────────────────────────────────

def test_intraday_settlement_schedule_empty():
	cycles = intraday_settlement_schedule([], cycle_hours=4)
	assert cycles == []


def test_intraday_settlement_schedule_single_cycle():
	from datetime import datetime, timezone
	txns = [
		{"amount": "10000", "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed"},
		{"amount": "5000",  "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed"},
	]
	cycles = intraday_settlement_schedule(txns, cycle_hours=4)
	assert len(cycles) >= 1
	cycle = cycles[0]
	assert "gross" in cycle
	assert "provisional_credit" in cycle
	assert "final_credit" in cycle
	# provisional should be 90% of net
	net = Decimal(cycle["net"])
	provisional = Decimal(cycle["provisional_credit"])
	assert abs(provisional / net - Decimal("0.90")) < Decimal("0.01")


def test_intraday_settlement_service():
	s = svc()
	# Complete some transactions
	txn = run(s.mpesa_stk_push("254712345678", Decimal("2000"), "INTRA-001"))
	run(s.confirm_payment(txn["id"], "INTRA-CONF"))
	result = run(s.intraday_settlement(bank_account="ACC-INTRADAY", cycle_hours=4))
	assert "cycles" in result
	assert "settlement_ids" in result
	assert result["cycle_hours"] == 4


# ─────────────────────────────────────────────────────────────
# 10. Payment widget spec
# ─────────────────────────────────────────────────────────────

def test_widget_spec_structure():
	s = svc()
	result = run(s.payment_widget_spec(
		merchant_id="merch-001",
		amount=Decimal("1000"),
		currency="KES",
		methods=["mpesa_stk", "card_visa"],
	))
	assert result["version"] == "1.0"
	assert result["widget_type"] == "payment"
	assert "state_machine" in result
	assert "offline_contract" in result
	assert "ui_hints" in result
	assert "fee_estimates" in result


def test_widget_spec_state_machine_completeness():
	s = svc()
	result = run(s.payment_widget_spec("merch-002", Decimal("500")))
	sm = result["state_machine"]
	assert "idle" in sm["states"]
	assert "pending" in sm["states"]
	assert "offline_queue" in sm["states"]
	assert "completed" in sm["states"]
	assert "failed" in sm["states"]


def test_widget_spec_offline_contract():
	s = svc()
	result = run(s.payment_widget_spec("merch-003", Decimal("250"), currency="USD"))
	oc = result["offline_contract"]
	assert oc["sync_on_reconnect"] is True
	assert oc["conflict_resolution"] == "server_wins"
	# FX hint should be True for non-KES
	assert result["ui_hints"]["show_fx_rate"] is True


def test_widget_spec_fee_estimates_populated():
	s = svc()
	result = run(s.payment_widget_spec(
		"merch-fee",
		Decimal("5000"),
		currency="KES",
		methods=["mpesa_stk"],
	))
	assert "mpesa_stk" in result["fee_estimates"]
	# M-Pesa fee for 5000 KES = 57 + 11.40 excise = 68.40 total
	assert Decimal(result["fee_estimates"]["mpesa_stk"]) > Decimal("0")
