"""Service-level tests for APG Multi-Currency Management."""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import date

_CAP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CAP_DIR)

from service import MultiCurrencyManagementService
from models import (
	CurrencyConfigCreate,
	CurrencyConfigUpdate,
	CurrencyTranslationCreate,
	ExchangeRateCreate,
	FxAccountCreate,
	McyAgentCreate,
	RevaluationCreate,
)


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def _svc():
	return MultiCurrencyManagementService()


TENANT = "test_tenant"


def _kes_payload(**kw):
	return CurrencyConfigCreate(
		tenant_id=TENANT,
		code=kw.get("code", "KES"),
		name=kw.get("name", "Kenyan Shilling"),
		symbol=kw.get("symbol", "KSh"),
		decimal_places=kw.get("decimal_places", 2),
		rounding_mode="round_half_even",
	)


def _fx_gain_account(svc, tenant=TENANT):
	return _run(svc.register_fx_account(FxAccountCreate(
		tenant_id=tenant,
		account_type="realised_gain",
		account_code="FX_GAIN_001",
		account_name="FX Realised Gain",
		currency="USD",
	)))


def _fx_loss_account(svc, tenant=TENANT):
	return _run(svc.register_fx_account(FxAccountCreate(
		tenant_id=tenant,
		account_type="realised_loss",
		account_code="FX_LOSS_001",
		account_name="FX Realised Loss",
		currency="USD",
	)))


def test_configure_currency():
	svc = _svc()
	c = _run(svc.configure_currency(_kes_payload()))
	assert c.code == "KES"
	assert c.status == "active"
	assert c.tenant_id == TENANT


def test_currency_code_uppercased():
	svc = _svc()
	c = _run(svc.configure_currency(_kes_payload(code="kes")))
	assert c.code == "KES"


def test_unsupported_currency_rejected():
	from pydantic import ValidationError
	try:
		CurrencyConfigCreate(
			tenant_id=TENANT, code="XXX", name="Fake", symbol="X",
			decimal_places=2, rounding_mode="round_half_even",
		)
		assert False, "expected ValidationError"
	except (AssertionError, ValidationError):
		pass


def test_list_currencies_empty():
	svc = _svc()
	assert _run(svc.list_currencies(TENANT)) == []


def test_list_currencies_after_configure():
	svc = _svc()
	_run(svc.configure_currency(_kes_payload()))
	assert len(_run(svc.list_currencies(TENANT))) == 1


def test_record_exchange_rate_spot():
	svc = _svc()
	# Use a future date to avoid the backdating rule
	rate = _run(svc.record_exchange_rate(ExchangeRateCreate(
		tenant_id=TENANT,
		from_currency="KES",
		to_currency="USD",
		rate=0.0077,
		rate_type="spot",
		rate_source="central_bank",
		effective_date=date(2026, 12, 31),
	)))
	assert rate.from_currency == "KES"
	assert rate.to_currency == "USD"
	assert rate.rate == 0.0077


def test_record_rate_negative_rejected():
	from pydantic import ValidationError
	try:
		ExchangeRateCreate(
			tenant_id=TENANT,
			from_currency="KES",
			to_currency="USD",
			rate=-1.0,
			rate_type="spot",
			rate_source="central_bank",
			effective_date=date(2026, 12, 31),
		)
		assert False, "expected ValidationError"
	except (AssertionError, ValidationError):
		pass


def test_self_conversion_rejected():
	from pydantic import ValidationError
	try:
		ExchangeRateCreate(
			tenant_id=TENANT,
			from_currency="USD",
			to_currency="USD",
			rate=1.0,
			rate_type="spot",
			rate_source="central_bank",
			effective_date=date(2026, 12, 31),
		)
		assert False, "expected ValidationError"
	except (AssertionError, ValidationError):
		pass


def test_convert_amount_direct():
	svc = _svc()
	_run(svc.record_exchange_rate(ExchangeRateCreate(
		tenant_id=TENANT,
		from_currency="KES",
		to_currency="USD",
		rate=0.0077,
		rate_type="spot",
		rate_source="central_bank",
		effective_date=date(2026, 12, 31),
	)))
	result = _run(svc.convert_amount(TENANT, 1000.0, "KES", "USD", date(2026, 12, 31)))
	assert result["converted_amount"] == round(1000.0 * 0.0077, 6)


def test_convert_same_currency():
	svc = _svc()
	result = _run(svc.convert_amount(TENANT, 500.0, "USD", "USD", date(2026, 6, 1)))
	assert result["converted_amount"] == 500.0
	assert result["rate"] == 1.0


def test_register_fx_account():
	svc = _svc()
	acct = _fx_gain_account(svc)
	assert acct.account_type == "realised_gain"
	assert acct.account_code == "FX_GAIN_001"


def test_create_revaluation():
	svc = _svc()
	gain = _fx_gain_account(svc)
	loss = _fx_loss_account(svc)
	rev = _run(svc.create_revaluation(RevaluationCreate(
		tenant_id=TENANT,
		entity_id="entity_001",
		period_start=date(2026, 1, 1),
		period_end=date(2026, 3, 31),
		revaluation_method="closing_rate",
		functional_currency="KES",
		fx_gain_account_id=gain.id,
		fx_loss_account_id=loss.id,
	)))
	assert rev.status == "draft"
	assert rev.revaluation_method == "closing_rate"


def test_post_revaluation_requires_approval():
	svc = _svc()
	gain = _fx_gain_account(svc)
	loss = _fx_loss_account(svc)
	rev = _run(svc.create_revaluation(RevaluationCreate(
		tenant_id=TENANT,
		entity_id="entity_001",
		period_start=date(2026, 1, 1),
		period_end=date(2026, 3, 31),
		revaluation_method="closing_rate",
		functional_currency="KES",
		fx_gain_account_id=gain.id,
		fx_loss_account_id=loss.id,
	)))
	try:
		_run(svc.post_revaluation(TENANT, rev.id))
		assert False, "expected PermissionError — no approval reference"
	except PermissionError:
		pass


def test_create_currency_translation():
	svc = _svc()
	reserve = _run(svc.register_fx_account(FxAccountCreate(
		tenant_id=TENANT,
		account_type="translation_reserve",
		account_code="TR_RESERVE_001",
		account_name="Translation Reserve",
		currency="USD",
	)))
	tr = _run(svc.create_translation(CurrencyTranslationCreate(
		tenant_id=TENANT,
		entity_id="entity_001",
		period_start=date(2026, 1, 1),
		period_end=date(2026, 3, 31),
		source_currency="KES",
		target_currency="USD",
		translation_method="current_rate",
		translation_reserve_account_id=reserve.id,
	)))
	assert tr.status == "draft"
	assert tr.source_currency == "KES"
	assert tr.target_currency == "USD"


def test_register_agent():
	svc = _svc()
	agent = _run(svc.register_agent(McyAgentCreate(
		tenant_id=TENANT,
		name="RateMonitor",
		runtime="claude_code",
		role="rate_feed_monitor",
		scope="exchange rate feed ingestion",
	)))
	assert agent.name == "RateMonitor"
	assert agent.role == "rate_feed_monitor"


def test_dashboard_summary():
	svc = _svc()
	_run(svc.configure_currency(_kes_payload()))
	summary = _run(svc.dashboard_summary(TENANT))
	assert summary["currency_count"] == 1
	assert summary["tenant_id"] == TENANT


def test_cross_tenant_isolation():
	svc = _svc()
	_run(svc.configure_currency(_kes_payload()))
	assert _run(svc.list_currencies("other_tenant")) == []


def test_get_rate_no_result():
	svc = _svc()
	rate = _run(svc.get_rate_for_date(TENANT, "USD", "GBP", date(2027, 1, 1)))
	assert rate is None


def test_fx_gain_loss_report():
	svc = _svc()
	report = _run(svc.generate_fx_report(TENANT, date(2026, 1, 1), date(2026, 3, 31)))
	assert report.tenant_id == TENANT
	assert report.net_fx_impact == 0.0
