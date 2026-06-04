"""Service tests for Property Valuation (val)."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from capabilities.realestate.val.service import ValService
from capabilities.realestate.val.models import (
	ValuerCreate, ValuerGrade,
	ComparableCreate, ComparableType,
	ValuationCreate, ValuationUpdate, ValuationMethod, ValuationPurpose,
	ValuationStatus, ReportType,
	DcfModelCreate,
	ValuationRollEntryCreate,
	MassAppraisalRunCreate, MassAppraisalModel,
	ValuationChallengeCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return ValService()


def _valuer(svc, independent=True, grade=ValuerGrade.rics_registered):
	return loop.run_until_complete(svc.register_valuer(ValuerCreate(
		tenant_id=T, name="John Valuer MRICS",
		grade=grade, email="jv@test.com",
		is_independent=independent, created_by="u",
	)))


def _valuation(svc, valuer_id, **kwargs):
	defaults = dict(
		tenant_id=T, property_id="prop-1",
		valuation_method=ValuationMethod.investment_method,
		purpose=ValuationPurpose.financial_reporting,
		report_type=ReportType.restricted_report,
		valuer_id=valuer_id,
		instruction_date=date(2025, 1, 10),
		created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.instruct_valuation(ValuationCreate(**defaults)))


# ── Valuer ────────────────────────────────────────────────────────────────────

def test_register_valuer():
	svc = _svc()
	v = _valuer(svc)
	assert v.id
	assert v.is_independent is True


def test_list_valuers_independent():
	svc = _svc()
	_valuer(svc, independent=True)
	_valuer(svc, independent=False)
	indep = loop.run_until_complete(svc.list_valuers(T, independent_only=True))
	assert len(indep) == 1


# ── Comparable ────────────────────────────────────────────────────────────────

def test_add_and_verify_comparable():
	svc = _svc()
	comp = loop.run_until_complete(svc.add_comparable(ComparableCreate(
		tenant_id=T, comparable_type=ComparableType.sale,
		address="45 Business Park, Nairobi",
		transaction_date=date(2024, 11, 1),
		price=Decimal("85000000"),
		area=Decimal("1200"), area_unit="sqm",
		source="market_report", created_by="u",
	)))
	assert comp.verified is False
	verified = loop.run_until_complete(svc.verify_comparable(comp.id, T, "senior_valuer"))
	assert verified.verified is True


def test_list_comparables_verified_only():
	svc = _svc()
	loop.run_until_complete(svc.add_comparable(ComparableCreate(
		tenant_id=T, comparable_type=ComparableType.sale,
		address="Addr1", transaction_date=date(2024, 1, 1),
		price=Decimal("50000000"), created_by="u",
	)))
	c2 = loop.run_until_complete(svc.add_comparable(ComparableCreate(
		tenant_id=T, comparable_type=ComparableType.sale,
		address="Addr2", transaction_date=date(2024, 6, 1),
		price=Decimal("60000000"), verified=True, created_by="u",
	)))
	verified_list = loop.run_until_complete(svc.list_comparables(T, verified_only=True))
	assert len(verified_list) == 1


# ── Valuation ─────────────────────────────────────────────────────────────────

def test_instruct_valuation():
	svc = _svc()
	v = _valuer(svc)
	val = _valuation(svc, v.id)
	assert val.ref.startswith("VAL-")
	assert val.status == ValuationStatus.instructed


def test_update_valuation():
	svc = _svc()
	v = _valuer(svc)
	val = _valuation(svc, v.id)
	updated = loop.run_until_complete(svc.update_valuation(val.id, T, ValuationUpdate(
		valuation_figure=Decimal("75000000"),
		status=ValuationStatus.draft_issued,
	)))
	assert updated.valuation_figure == Decimal("75000000")


def test_sign_off_requires_qualified_grade():
	svc = _svc()
	v = _valuer(svc, grade=ValuerGrade.internal_valuer)
	val = _valuation(svc, v.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.sign_off_valuation(val.id, T, "u", ValuerGrade.internal_valuer.value))


def test_sign_off_by_rics_valuer():
	svc = _svc()
	v = _valuer(svc, grade=ValuerGrade.rics_registered)
	val = _valuation(svc, v.id)
	signed = loop.run_until_complete(svc.sign_off_valuation(val.id, T, v.id, ValuerGrade.rics_registered.value))
	assert signed.status == ValuationStatus.signed_off


def test_publish_valuation():
	svc = _svc()
	v = _valuer(svc, independent=True, grade=ValuerGrade.rics_registered)
	val = _valuation(svc, v.id)
	loop.run_until_complete(svc.sign_off_valuation(val.id, T, v.id, ValuerGrade.rics_registered.value))
	published = loop.run_until_complete(svc.publish_valuation(val.id, T))
	assert published.status == ValuationStatus.published
	assert published.published_at is not None


def test_published_valuation_immutable():
	svc = _svc()
	v = _valuer(svc, grade=ValuerGrade.rics_registered)
	val = _valuation(svc, v.id)
	loop.run_until_complete(svc.sign_off_valuation(val.id, T, v.id, ValuerGrade.rics_registered.value))
	loop.run_until_complete(svc.publish_valuation(val.id, T))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.update_valuation(val.id, T, ValuationUpdate(valuation_figure=Decimal("1"))))


# ── DCF ───────────────────────────────────────────────────────────────────────

def test_run_dcf_model():
	svc = _svc()
	v = _valuer(svc)
	val = _valuation(svc, v.id)
	dcf = loop.run_until_complete(svc.run_dcf_model(DcfModelCreate(
		tenant_id=T, valuation_id=val.id, property_id="prop-1",
		discount_rate=Decimal("0.08"), holding_period_years=5,
		exit_yield=Decimal("0.07"), annual_rental_income=Decimal("6000000"),
		rental_growth_rate=Decimal("0.03"),
		created_by="u",
	)))
	assert dcf.npv > 0
	assert dcf.capital_value > 0
	assert len(dcf.cash_flow_schedule) == 5


def test_dcf_out_of_range_rate_raises():
	svc = _svc()
	v = _valuer(svc)
	val = _valuation(svc, v.id)
	with pytest.raises(Exception):
		DcfModelCreate(
			tenant_id=T, valuation_id=val.id, property_id="prop-1",
			discount_rate=Decimal("0.50"),  # out of range
			holding_period_years=5, exit_yield=Decimal("0.07"),
			annual_rental_income=Decimal("5000000"), created_by="u",
		)


# ── Valuation Roll ────────────────────────────────────────────────────────────

def test_valuation_roll_supersedes_previous():
	svc = _svc()
	v = _valuer(svc)
	val1 = _valuation(svc, v.id)
	loop.run_until_complete(svc.add_to_valuation_roll(ValuationRollEntryCreate(
		tenant_id=T, property_id="prop-1", valuation_id=val1.id,
		effective_date=date(2024, 12, 31), valuation_figure=Decimal("70000000"),
		created_by="u",
	)))
	val2 = _valuation(svc, v.id)
	loop.run_until_complete(svc.add_to_valuation_roll(ValuationRollEntryCreate(
		tenant_id=T, property_id="prop-1", valuation_id=val2.id,
		effective_date=date(2025, 6, 30), valuation_figure=Decimal("75000000"),
		created_by="u",
	)))
	roll = loop.run_until_complete(svc.get_valuation_roll(T, property_id="prop-1"))
	assert len(roll) == 1
	assert roll[0].valuation_figure == Decimal("75000000")


# ── Challenge ─────────────────────────────────────────────────────────────────

def test_raise_challenge():
	svc = _svc()
	v = _valuer(svc, grade=ValuerGrade.rics_registered)
	val = _valuation(svc, v.id)
	loop.run_until_complete(svc.sign_off_valuation(val.id, T, v.id, ValuerGrade.rics_registered.value))
	loop.run_until_complete(svc.publish_valuation(val.id, T))
	challenge = loop.run_until_complete(svc.raise_challenge(ValuationChallengeCreate(
		tenant_id=T, valuation_id=val.id,
		raised_by="owner-1", grounds="Below market evidence",
		counter_evidence_document_ids=["evidence.pdf"],
		counter_valuation_figure=Decimal("82000000"),
		created_by="u",
	)))
	assert challenge.status == "open"
	# Valuation should now show as challenged
	challenged_val = loop.run_until_complete(svc.get_valuation(val.id, T))
	assert challenged_val.status == ValuationStatus.challenged


def test_challenge_no_evidence_raises():
	svc = _svc()
	v = _valuer(svc)
	val = _valuation(svc, v.id)
	with pytest.raises(Exception):
		ValuationChallengeCreate(
			tenant_id=T, valuation_id=val.id, raised_by="owner-1",
			grounds="Too low", counter_evidence_document_ids=[],
			created_by="u",
		)


# ── Yield ─────────────────────────────────────────────────────────────────────

def test_calculate_yield():
	svc = _svc()
	result = loop.run_until_complete(svc.calculate_yield(
		T, "prop-1", Decimal("6000000"), Decimal("80000000"), "net_initial_yield"
	))
	assert result["yield_pct"] == 7.5
