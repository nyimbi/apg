"""Tests for AML edge-case typology detection: TBML, NFT wash-trade,
crypto mixer routing, correspondent nesting, terrorist financing indicators.

Plain sync tests — no async needed for pure-function domain calculations.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))

from domain.calculations import (  # type: ignore
	calculate_correspondent_nesting_risk,
	detect_crypto_mixer_routing,
	detect_nft_wash_trading,
	detect_terrorist_financing_indicators,
	detect_trade_based_ml,
)
from domain.rules import (  # type: ignore
	RuleViolation,
	assert_alert_close_has_disposition,
	assert_alert_escalation_has_reviewer,
	assert_alert_evidence_present,
	assert_alert_type_supported,
	assert_case_is_open_for_investigation,
	assert_case_type_supported,
	assert_correspondent_nesting_depth_acceptable,
	assert_crypto_mixer_not_detected,
	assert_ctr_amount_triggers_reporting,
	assert_investigator_assigned,
	assert_kyc_link_present,
	assert_match_score_valid,
	assert_no_cross_tenant_access,
	assert_nft_wash_trade_not_detected,
	assert_positive_amount,
	assert_sar_human_approval,
	assert_sar_jurisdiction_present,
	assert_sar_narrative_present,
	assert_severity_supported,
	assert_source_reference_present,
	assert_tbml_invoice_variance_acceptable,
	assert_tenant_context,
	assert_transaction_subject_present,
)


# ---------------------------------------------------------------------------
# domain/rules.py — assert_* coverage
# ---------------------------------------------------------------------------

class TestTenantRules:
	def test_tenant_context_required(self):
		with pytest.raises(RuleViolation) as exc:
			assert_tenant_context({"tenant_id": ""})
		assert exc.value.rule_name == "tenant_context_required"

	def test_tenant_context_passes(self):
		assert_tenant_context({"tenant_id": "t1"})  # no raise

	def test_cross_tenant_denied(self):
		with pytest.raises(RuleViolation) as exc:
			assert_no_cross_tenant_access("tenant-a", "tenant-b")
		assert exc.value.rule_name == "cross_tenant_access_denied"

	def test_same_tenant_passes(self):
		assert_no_cross_tenant_access("t1", "t1")  # no raise


class TestTransactionRules:
	def test_subject_required(self):
		with pytest.raises(RuleViolation):
			assert_transaction_subject_present("")

	def test_subject_whitespace_fails(self):
		with pytest.raises(RuleViolation):
			assert_transaction_subject_present("   ")

	def test_positive_amount(self):
		with pytest.raises(RuleViolation):
			assert_positive_amount(0.0)
		with pytest.raises(RuleViolation):
			assert_positive_amount(-100.0)

	def test_currency_required(self):
		with pytest.raises(RuleViolation):
			assert_currency_present("")

	def test_source_reference_required(self):
		with pytest.raises(RuleViolation):
			assert_source_reference_present("", "payments")
		with pytest.raises(RuleViolation):
			assert_source_reference_present("ref-1", "")

	def test_kyc_link_required(self):
		with pytest.raises(RuleViolation):
			assert_kyc_link_present("")


def assert_currency_present(c: str) -> None:
	from domain.rules import assert_currency_present as _f  # type: ignore
	_f(c)


def assert_source_reference_present(s: str, cap: str) -> None:
	from domain.rules import assert_source_reference_present as _f  # type: ignore
	_f(s, cap)


class TestAlertRules:
	def test_unsupported_alert_type(self):
		with pytest.raises(RuleViolation) as exc:
			assert_alert_type_supported("unicorn_laundering")
		assert exc.value.rule_name == "unsupported_alert_type"

	def test_supported_alert_type_passes(self):
		assert_alert_type_supported("structuring")

	def test_unsupported_severity(self):
		with pytest.raises(RuleViolation):
			assert_severity_supported("ultra_critical")

	def test_supported_severity_passes(self):
		assert_severity_supported("high")

	def test_evidence_required(self):
		with pytest.raises(RuleViolation):
			assert_alert_evidence_present([])

	def test_evidence_passes(self):
		assert_alert_evidence_present(["txn-1"])

	def test_escalation_requires_reviewer(self):
		with pytest.raises(RuleViolation):
			assert_alert_escalation_has_reviewer(True, "")

	def test_escalation_no_reviewer_if_not_escalating(self):
		assert_alert_escalation_has_reviewer(False, "")  # no raise

	def test_close_requires_disposition(self):
		with pytest.raises(RuleViolation):
			assert_alert_close_has_disposition(True, "")

	def test_close_no_disposition_if_not_closing(self):
		assert_alert_close_has_disposition(False, "")  # no raise


class TestCaseRules:
	def test_unsupported_case_type(self):
		with pytest.raises(RuleViolation):
			assert_case_type_supported("bad_type")

	def test_supported_case_type_passes(self):
		assert_case_type_supported("transaction_monitoring")

	def test_investigator_required(self):
		with pytest.raises(RuleViolation):
			assert_investigator_assigned("")

	def test_case_is_open(self):
		# Should not raise for open/under_investigation
		assert_case_is_open_for_investigation("open")
		assert_case_is_open_for_investigation("under_investigation")

	def test_terminal_case_blocked(self):
		for terminal in ("closed_no_action", "closed_action_taken", "referred_to_lea"):
			with pytest.raises(RuleViolation) as exc:
				assert_case_is_open_for_investigation(terminal)
			assert exc.value.rule_name == "case_already_closed"


class TestSARRules:
	def test_narrative_too_short(self):
		with pytest.raises(RuleViolation) as exc:
			assert_sar_narrative_present("too short")
		assert exc.value.rule_name == "sar_narrative_insufficient"

	def test_narrative_sufficient(self):
		assert_sar_narrative_present("A" * 50)  # exactly 50 chars — passes

	def test_jurisdiction_required(self):
		with pytest.raises(RuleViolation):
			assert_sar_jurisdiction_present("")

	def test_human_approval_required(self):
		with pytest.raises(RuleViolation):
			assert_sar_human_approval("")

	def test_human_approval_passes(self):
		assert_sar_human_approval("compliance-officer-1")


class TestCTRRules:
	def test_below_threshold_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_ctr_amount_triggers_reporting(5_000.0, 10_000.0)
		assert exc.value.rule_name == "ctr_threshold_not_met"

	def test_at_threshold_passes(self):
		assert_ctr_amount_triggers_reporting(10_000.0, 10_000.0)

	def test_above_threshold_passes(self):
		assert_ctr_amount_triggers_reporting(15_000.0, 10_000.0)


class TestWatchlistRules:
	def test_invalid_score_above_one(self):
		with pytest.raises(RuleViolation):
			assert_match_score_valid(1.5)

	def test_invalid_score_negative(self):
		with pytest.raises(RuleViolation):
			assert_match_score_valid(-0.1)

	def test_valid_score_passes(self):
		assert_match_score_valid(0.85)


# ---------------------------------------------------------------------------
# TBML domain rules
# ---------------------------------------------------------------------------

class TestTBMLRules:
	def test_invoice_variance_acceptable(self):
		assert_tbml_invoice_variance_acceptable(10_000, 10_500, tolerance_pct=0.15)  # 5% — ok

	def test_invoice_over_invoiced_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_tbml_invoice_variance_acceptable(20_000, 10_000, tolerance_pct=0.15)
		assert exc.value.rule_name == "tbml_invoice_variance_exceeded"
		assert "over_invoiced" in exc.value.reason

	def test_invoice_under_invoiced_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_tbml_invoice_variance_acceptable(5_000, 10_000, tolerance_pct=0.15)
		assert exc.value.rule_name == "tbml_invoice_variance_exceeded"
		assert "under_invoiced" in exc.value.reason

	def test_zero_market_value_raises(self):
		with pytest.raises(RuleViolation):
			assert_tbml_invoice_variance_acceptable(10_000, 0.0)


# ---------------------------------------------------------------------------
# Crypto/NFT domain rules
# ---------------------------------------------------------------------------

class TestCryptoRules:
	def test_mixer_detected_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_crypto_mixer_not_detected(["tornado_cash"])
		assert exc.value.rule_name == "crypto_mixer_detected"

	def test_no_mixer_passes(self):
		assert_crypto_mixer_not_detected([])

	def test_nft_wash_trade_above_threshold_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_nft_wash_trade_not_detected(0.85, threshold=0.7)
		assert exc.value.rule_name == "nft_wash_trade_detected"

	def test_nft_wash_trade_below_threshold_passes(self):
		assert_nft_wash_trade_not_detected(0.5, threshold=0.7)


# ---------------------------------------------------------------------------
# Correspondent nesting rules
# ---------------------------------------------------------------------------

class TestCorrespondentRules:
	def test_depth_acceptable(self):
		assert_correspondent_nesting_depth_acceptable(3, max_depth=3)

	def test_depth_exceeded_raises(self):
		with pytest.raises(RuleViolation) as exc:
			assert_correspondent_nesting_depth_acceptable(4, max_depth=3)
		assert exc.value.rule_name == "correspondent_nesting_too_deep"


# ---------------------------------------------------------------------------
# detect_trade_based_ml
# ---------------------------------------------------------------------------

def _invoice(
	inv_id: str,
	amount: float,
	unit_price: float,
	commodity: str = "WIDGET",
	counterparty: str = "CN",
) -> dict:
	return {
		"id": inv_id,
		"amount": amount,
		"commodity_code": commodity,
		"quantity": 10,
		"unit_price": unit_price,
		"counterparty_country": counterparty,
	}


class TestDetectTBML:
	def test_clean_invoices(self):
		invoices = [_invoice("inv-1", 10_000, 1_000)]
		result = detect_trade_based_ml(
			invoices,
			market_value_lookup={"WIDGET": 1_000},
			over_under_threshold=0.15,
		)
		assert result["detected"] is False

	def test_over_invoiced_detected(self):
		invoices = [_invoice("inv-1", 25_000, 2_500)]  # market = 1000 → 150% over
		result = detect_trade_based_ml(
			invoices,
			market_value_lookup={"WIDGET": 1_000},
			over_under_threshold=0.15,
		)
		assert result["detected"] is True
		assert any("over_invoiced" in t for t in result["typologies"])

	def test_under_invoiced_detected(self):
		invoices = [_invoice("inv-1", 3_000, 300)]  # market = 1000 → 70% under
		result = detect_trade_based_ml(
			invoices,
			market_value_lookup={"WIDGET": 1_000},
			over_under_threshold=0.15,
		)
		assert result["detected"] is True
		assert any("under_invoiced" in t for t in result["typologies"])

	def test_phantom_shipment_detected(self):
		invoices = [_invoice("phantom-ref-1", 10_000, 1_000)]
		result = detect_trade_based_ml(
			invoices,
			phantom_shipment_indicators=["phantom-ref-1"],
		)
		assert result["detected"] is True
		assert "phantom_shipment" in result["typologies"]

	def test_multiple_invoicing_detected(self):
		invoices = [
			_invoice("inv-1", 10_000, 1_000, commodity="WIDGET", counterparty="CN"),
			_invoice("inv-2", 10_000, 1_000, commodity="WIDGET", counterparty="CN"),  # same pair
		]
		result = detect_trade_based_ml(invoices)
		assert result["detected"] is True
		assert "multiple_invoicing" in result["typologies"]

	def test_empty_invoices(self):
		result = detect_trade_based_ml([])
		assert result["detected"] is False
		assert result["risk_score"] == 0


# ---------------------------------------------------------------------------
# detect_nft_wash_trading
# ---------------------------------------------------------------------------

def _transfer(token_id: str, from_w: str, to_w: str, price: float, minutes_ago: int = 0) -> dict:
	return {
		"token_id": token_id,
		"from_wallet": from_w,
		"to_wallet": to_w,
		"price": price,
		"currency": "ETH",
		"created_at": (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat(),
	}


class TestNFTWashTrading:
	def test_clean_transfers(self):
		transfers = [
			_transfer("nft-1", "alice", "bob", 1.0, 300),
			_transfer("nft-1", "bob", "carol", 1.1, 200),
		]
		result = detect_nft_wash_trading(transfers, min_round_trips=2)
		# bob → carol → different wallet, no round-trip ≥ 2
		assert result["wash_trade_score"] < 1.0

	def test_round_trip_wash_detected(self):
		transfers = [
			_transfer("nft-2", "alice", "bob", 1.0, 600),
			_transfer("nft-2", "bob", "alice", 5.0, 400),   # round-trip 1
			_transfer("nft-2", "alice", "bob", 15.0, 200),  # round-trip 2
			_transfer("nft-2", "bob", "alice", 50.0, 60),   # round-trip 3
		]
		result = detect_nft_wash_trading(transfers, min_round_trips=2)
		assert result["detected"] is True
		assert result["flagged_tokens"][0]["round_trips"] >= 2

	def test_price_inflation_flagged(self):
		transfers = [
			_transfer("nft-3", "alice", "bob", 1.0, 600),
			_transfer("nft-3", "bob", "carol", 1.0, 400),
			_transfer("nft-3", "carol", "dave", 1.0, 200),
			_transfer("nft-3", "dave", "eve", 5.0, 100),   # 5x inflation
		]
		result = detect_nft_wash_trading(transfers, price_inflation_threshold=3.0)
		assert result["detected"] is True
		assert "artificial_price_inflation" in result["patterns"]

	def test_empty_transfers(self):
		result = detect_nft_wash_trading([])
		assert result["detected"] is False
		assert result["wash_trade_score"] == 0.0


# ---------------------------------------------------------------------------
# detect_crypto_mixer_routing
# ---------------------------------------------------------------------------

class TestCryptoMixerDetection:
	def test_known_service_label_detected(self):
		txns = [{"tx_hash": "0xabc", "from_address": "0x1", "to_address": "0x2", "service_label": "tornado_cash_relay"}]
		result = detect_crypto_mixer_routing(txns)
		assert result["detected"] is True
		assert "tornado_cash" in result["mixer_indicators"]

	def test_known_address_detected(self):
		mixer_addr = "0xdeadbeef"
		txns = [{"tx_hash": "0x123", "from_address": "0xabc", "to_address": mixer_addr}]
		result = detect_crypto_mixer_routing(txns, known_mixer_addresses={mixer_addr})
		assert result["detected"] is True
		assert "known_mixer_address" in result["mixer_indicators"]

	def test_coinjoin_pattern_detected(self):
		txns = [{
			"tx_hash": "0xcoinjoin",
			"from_address": "0x1",
			"to_address": "0x2",
			"input_count": 7,
			"output_count": 7,
			"equal_output_amounts": True,
		}]
		result = detect_crypto_mixer_routing(txns)
		assert result["detected"] is True
		assert "coinjoin_pattern" in result["mixer_indicators"]

	def test_clean_transaction(self):
		txns = [{"tx_hash": "0xclean", "from_address": "0x1", "to_address": "0x2"}]
		result = detect_crypto_mixer_routing(txns)
		assert result["detected"] is False

	def test_empty(self):
		result = detect_crypto_mixer_routing([])
		assert result["detected"] is False


# ---------------------------------------------------------------------------
# calculate_correspondent_nesting_risk
# ---------------------------------------------------------------------------

def _link(institution_id: str, jurisdiction: str, aml_rating: str = "good", kyb_status: str = "verified", nested_count: int = 0) -> dict:
	return {
		"institution_id": institution_id,
		"institution_name": f"Bank {institution_id}",
		"jurisdiction": jurisdiction,
		"aml_rating": aml_rating,
		"kyb_status": kyb_status,
		"nested_accounts_count": nested_count,
	}


class TestCorrespondentNestingRisk:
	def test_shallow_low_risk_chain(self):
		chain = [_link("bank-a", "US"), _link("bank-b", "UK")]
		result = calculate_correspondent_nesting_risk(chain)
		assert result["nesting_depth"] == 2
		assert result["risk_score"] < 30
		assert result["recommended_action"] in ("standard_monitoring", "review_and_monitor")

	def test_deep_chain_penalty(self):
		chain = [
			_link("b1", "US"),
			_link("b2", "UK"),
			_link("b3", "DE"),
			_link("b4", "FR"),  # depth=4, triggers penalty
		]
		result = calculate_correspondent_nesting_risk(chain)
		assert result["nesting_depth"] == 4
		assert any("deep_nesting" in f for f in result["risk_factors"])

	def test_high_risk_jurisdiction_penalty(self):
		chain = [_link("b1", "US"), _link("b2", "IR")]  # Iran = high risk
		result = calculate_correspondent_nesting_risk(chain)
		assert any("high_risk_jurisdiction:IR" in f for f in result["risk_factors"])
		assert result["risk_score"] >= 25

	def test_poor_aml_rating_penalty(self):
		chain = [_link("b1", "US", aml_rating="poor")]
		result = calculate_correspondent_nesting_risk(chain)
		assert any("poor_aml_rating" in f for f in result["risk_factors"])

	def test_very_high_risk_recommends_termination(self):
		chain = [
			_link("b1", "IR", aml_rating="non_compliant", kyb_status="pending", nested_count=10),
			_link("b2", "KP", aml_rating="sanctioned", kyb_status="rejected", nested_count=10),
		]
		result = calculate_correspondent_nesting_risk(chain)
		assert result["recommended_action"] == "terminate_relationship"
		assert result["risk_score"] >= 70

	def test_empty_chain(self):
		result = calculate_correspondent_nesting_risk([])
		assert result["nesting_depth"] == 0
		assert result["risk_score"] == 0


# ---------------------------------------------------------------------------
# detect_terrorist_financing_indicators
# ---------------------------------------------------------------------------

def _tf_txn(amount: float, dest: str, txn_type: str = "wire", minutes_ago: int = 0) -> dict:
	return {
		"amount": amount,
		"destination_country": dest,
		"transaction_type": txn_type,
		"created_at": (datetime.utcnow() - timedelta(minutes=minutes_ago)).isoformat(),
	}


class TestTerroristFinancingDetection:
	def test_clean_profile_no_indicators(self):
		txns = [_tf_txn(500, "US")]
		result = detect_terrorist_financing_indicators(txns)
		assert result["detected"] is False
		assert result["risk_score"] == 0

	def test_adverse_media_terrorism_link(self):
		result = detect_terrorist_financing_indicators(
			[_tf_txn(100, "US")],
			customer_profile={"adverse_media_terrorism": True},
		)
		assert result["detected"] is True
		assert "adverse_media_terrorism_link" in result["tf_indicators"]
		assert result["risk_score"] >= 40

	def test_known_tf_associate(self):
		result = detect_terrorist_financing_indicators(
			[_tf_txn(100, "US")],
			customer_profile={"known_tf_associate": True},
		)
		assert result["detected"] is True
		assert result["risk_score"] >= 50

	def test_small_amount_high_risk_jurisdiction(self):
		txns = [_tf_txn(500, "AF")]  # Afghanistan, small amount
		result = detect_terrorist_financing_indicators(txns)
		assert result["detected"] is True
		assert "small_amount_to_high_risk_jurisdiction" in result["tf_indicators"]

	def test_hawala_pattern(self):
		txns = [_tf_txn(2000, "SY", txn_type="hawala_transfer")]
		result = detect_terrorist_financing_indicators(txns)
		assert result["detected"] is True
		assert "hawala_pattern" in result["tf_indicators"]

	def test_charity_misuse(self):
		txns = [_tf_txn(5000, "YE")]
		result = detect_terrorist_financing_indicators(
			txns,
			customer_profile={"charity_sector": True},
		)
		assert result["detected"] is True
		assert "charity_misuse" in result["tf_indicators"]

	def test_empty_transactions(self):
		result = detect_terrorist_financing_indicators([])
		assert result["detected"] is False
