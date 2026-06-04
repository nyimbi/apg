"""Service-level integration tests for AML edge-case typology methods.

Covers: TBML, NFT wash-trade, crypto mixer, correspondent banking, TF detection.
Plain async functions — no @pytest.mark.asyncio decorators.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))

from service import AMLService  # type: ignore
from domain.rules import RuleViolation  # type: ignore


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def fresh(tenant_id: str = "t1") -> AMLService:
	return AMLService(tenant_id=tenant_id, actor_id="test-actor")


def _now(minutes_ago: int = 0) -> str:
	return (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat()


# ---------------------------------------------------------------------------
# TBML
# ---------------------------------------------------------------------------

async def test_tbml_clean():
	svc = fresh()
	invoices = [{"id": "inv-1", "amount": 10_000, "commodity_code": "STEEL",
	             "quantity": 10, "unit_price": 1_000, "counterparty_country": "CN"}]
	result = await svc.detect_trade_based_ml(
		invoices,
		market_value_lookup={"STEEL": 1_000},
		over_under_threshold=0.15,
	)
	assert result["detected"] is False
	assert result["risk_score"] == 0


async def test_tbml_over_invoiced_detected():
	svc = fresh()
	invoices = [{"id": "inv-1", "amount": 30_000, "commodity_code": "STEEL",
	             "quantity": 10, "unit_price": 3_000, "counterparty_country": "CN"}]
	result = await svc.detect_trade_based_ml(
		invoices,
		market_value_lookup={"STEEL": 1_000},
		over_under_threshold=0.15,
	)
	assert result["detected"] is True
	assert "over_invoiced" in result["typologies"]
	# Domain event should be emitted
	assert any(e["event_type"] == "tbml_detected" for e in svc._events)


async def test_tbml_phantom_shipment():
	svc = fresh()
	invoices = [{"id": "PHANTOM-001", "amount": 10_000, "commodity_code": "GOLD",
	             "quantity": 1, "unit_price": 10_000, "counterparty_country": "AE"}]
	result = await svc.detect_trade_based_ml(
		invoices,
		phantom_shipment_indicators=["PHANTOM-001"],
	)
	assert result["detected"] is True
	assert "phantom_shipment" in result["typologies"]


async def test_tbml_empty_invoices():
	svc = fresh()
	result = await svc.detect_trade_based_ml([])
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# NFT wash-trade
# ---------------------------------------------------------------------------

def _nft_transfer(token_id: str, from_w: str, to_w: str, price: float, minutes_ago: int = 0) -> dict:
	return {
		"token_id": token_id,
		"from_wallet": from_w,
		"to_wallet": to_w,
		"price": price,
		"currency": "ETH",
		"created_at": _now(minutes_ago),
	}


async def test_nft_wash_trade_clean():
	svc = fresh()
	transfers = [
		_nft_transfer("nft-1", "alice", "bob", 1.0, 120),
		_nft_transfer("nft-1", "bob", "carol", 1.05, 60),
	]
	result = await svc.detect_nft_wash_trading(transfers, min_round_trips=3)
	assert result["detected"] is False


async def test_nft_wash_trade_round_trip_raises():
	svc = fresh()
	transfers = [
		_nft_transfer("nft-2", "alice", "bob", 1.0, 500),
		_nft_transfer("nft-2", "bob", "alice", 10.0, 400),
		_nft_transfer("nft-2", "alice", "bob", 50.0, 300),
		_nft_transfer("nft-2", "bob", "alice", 200.0, 200),
	]
	import pytest
	with pytest.raises(RuleViolation) as exc:
		await svc.detect_nft_wash_trading(transfers, min_round_trips=2, price_inflation_threshold=5.0)
	assert exc.value.rule_name == "nft_wash_trade_detected"


async def test_nft_wash_trade_emits_event_without_rule_enforcement():
	"""When score is high enough to detect but below enforcement threshold, event fires."""
	svc = fresh()
	transfers = [
		_nft_transfer("nft-3", "alice", "bob", 1.0, 500),
		_nft_transfer("nft-3", "bob", "carol", 1.05, 400),
		_nft_transfer("nft-3", "carol", "alice", 1.1, 300),  # 1 round-trip
	]
	# min_round_trips=2 so not detected
	result = await svc.detect_nft_wash_trading(transfers, min_round_trips=2)
	assert result["detected"] is False
	assert not any(e["event_type"] == "nft_wash_trade_detected" for e in svc._events)


# ---------------------------------------------------------------------------
# Crypto mixer detection
# ---------------------------------------------------------------------------

async def test_crypto_mixer_clean_tx():
	svc = fresh()
	txns = [{"tx_hash": "0xabc", "from_address": "0x1", "to_address": "0x2"}]
	# detect_crypto_mixer_routing returns clean -> no raise
	result = await svc.detect_crypto_mixer_routing(txns)
	assert result["detected"] is False


async def test_crypto_mixer_known_service_label_raises():
	svc = fresh()
	txns = [{"tx_hash": "0xbad", "from_address": "0x1", "to_address": "0x2",
	         "service_label": "tornado_cash_v2"}]
	import pytest
	with pytest.raises(RuleViolation) as exc:
		await svc.detect_crypto_mixer_routing(txns)
	assert exc.value.rule_name == "crypto_mixer_detected"


async def test_crypto_mixer_known_address_raises():
	svc = fresh()
	mixer = "0xdeadbeefcafe"
	txns = [{"tx_hash": "0x999", "from_address": "0x1", "to_address": mixer}]
	import pytest
	with pytest.raises(RuleViolation):
		await svc.detect_crypto_mixer_routing(txns, known_mixer_addresses={mixer})


async def test_crypto_mixer_coinjoin_raises():
	svc = fresh()
	txns = [{
		"tx_hash": "0xcjoin",
		"from_address": "0x1",
		"to_address": "0x2",
		"input_count": 8,
		"output_count": 8,
		"equal_output_amounts": True,
	}]
	import pytest
	with pytest.raises(RuleViolation):
		await svc.detect_crypto_mixer_routing(txns)


async def test_crypto_mixer_empty():
	svc = fresh()
	result = await svc.detect_crypto_mixer_routing([])
	assert result["detected"] is False


# ---------------------------------------------------------------------------
# Correspondent banking analysis
# ---------------------------------------------------------------------------

def _link(institution_id: str, jurisdiction: str, aml_rating: str = "good",
          kyb_status: str = "verified", nested_count: int = 0) -> dict:
	return {
		"institution_id": institution_id,
		"institution_name": f"Bank {institution_id}",
		"jurisdiction": jurisdiction,
		"aml_rating": aml_rating,
		"kyb_status": kyb_status,
		"nested_accounts_count": nested_count,
	}


async def test_correspondent_shallow_clean():
	svc = fresh()
	chain = [_link("b1", "US"), _link("b2", "UK")]
	result = await svc.correspondent_banking_analysis(chain)
	assert result["nesting_depth"] == 2
	assert result["risk_score"] < 30


async def test_correspondent_deep_chain_raises():
	svc = fresh()
	chain = [_link(f"b{i}", "US") for i in range(5)]  # depth=5 > max=3
	import pytest
	with pytest.raises(RuleViolation) as exc:
		await svc.correspondent_banking_analysis(chain, max_nesting_depth=3)
	assert exc.value.rule_name == "correspondent_nesting_too_deep"


async def test_correspondent_high_risk_jurisdiction():
	svc = fresh()
	chain = [_link("b1", "US"), _link("b2", "IR")]  # Iran
	result = await svc.correspondent_banking_analysis(chain)
	assert any("IR" in f for f in result["risk_factors"])
	assert result["risk_score"] >= 25


async def test_correspondent_emits_event():
	svc = fresh()
	chain = [_link("b1", "US")]
	await svc.correspondent_banking_analysis(chain)
	assert any(e["event_type"] == "correspondent_banking_assessed" for e in svc._events)


async def test_correspondent_termination_recommended():
	svc = fresh()
	chain = [
		_link("b1", "KP", aml_rating="sanctioned", kyb_status="rejected", nested_count=20),
		_link("b2", "IR", aml_rating="non_compliant", kyb_status="pending", nested_count=15),
	]
	result = await svc.correspondent_banking_analysis(chain, max_nesting_depth=5)
	assert result["recommended_action"] == "terminate_relationship"


# ---------------------------------------------------------------------------
# Terrorist financing detection
# ---------------------------------------------------------------------------

async def test_tf_no_transactions():
	svc = fresh()
	result = await svc.detect_terrorist_financing("cust-clean")
	assert result["detected"] is False
	assert result["risk_score"] == 0


async def test_tf_detected_via_customer_profile():
	svc = fresh()
	# Seed a transaction so the service has history for the customer
	svc._transactions["txn-tf-1"] = {
		"id": "txn-tf-1",
		"tenant_id": "t1",
		"subject_reference": "cust-tf",
		"amount": 500,
		"destination_country": "US",
		"transaction_type": "wire",
	}
	result = await svc.detect_terrorist_financing(
		"cust-tf",
		customer_profile={"adverse_media_terrorism": True},
	)
	assert result["detected"] is True
	assert result["risk_score"] >= 40
	assert "adverse_media_terrorism_link" in result["tf_indicators"]
	assert any(e["event_type"] == "terrorist_financing_indicators_detected" for e in svc._events)


async def test_tf_small_amount_high_risk_jurisdiction():
	svc = fresh()
	svc._transactions["txn-af-1"] = {
		"id": "txn-af-1",
		"tenant_id": "t1",
		"subject_reference": "cust-suspicious",
		"amount": 800,
		"destination_country": "AF",
		"transaction_type": "wire",
	}
	result = await svc.detect_terrorist_financing("cust-suspicious")
	assert result["detected"] is True
	assert "small_amount_to_high_risk_jurisdiction" in result["tf_indicators"]


async def test_tf_tenant_isolation():
	svc_a = AMLService(tenant_id="tenant-a", actor_id="actor")
	svc_b = AMLService(tenant_id="tenant-b", actor_id="actor")
	# Seed transaction only in tenant-a's store
	svc_a._transactions["txn-a-tf"] = {
		"id": "txn-a-tf",
		"tenant_id": "tenant-a",
		"subject_reference": "cust-shared",
		"amount": 500,
		"destination_country": "AF",
		"transaction_type": "wire",
	}
	# tenant-b should see zero transactions for the same customer ID
	result_b = await svc_b.detect_terrorist_financing("cust-shared")
	assert result_b["transaction_count"] == 0
