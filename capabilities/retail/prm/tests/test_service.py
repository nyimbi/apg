"""Service tests for retail_prm capability."""

import asyncio
import pytest
from datetime import datetime, timedelta

from ..service import PrmService
from ..models import (
	PrmPromotionCreate, PrmPromotionUpdate, PrmTriggerCreate,
	PrmCouponCreate, PrmCouponRedemptionCreate,
	PrmPricingRuleCreate, PrmMarkdownCreate, PrmEffectivenessRecord,
)


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return PrmService()


def _promo(svc, name="Test Promo"):
	return run(svc.create_promotion(PrmPromotionCreate(
		tenant_id="t1", name=name, promotion_type="percentage_off",
		discount_type="percentage", discount_value=10.0,
		budget_cap=5000.0, margin_floor_pct=5.0,
		start_date=datetime.utcnow(),
		end_date=datetime.utcnow() + timedelta(days=30),
		created_by="admin",
	)))


def test_create_promotion(svc):
	p = _promo(svc)
	assert p.id
	assert p.approval_status == "draft"
	assert p.promotion_code.startswith("PROMO-")


def test_create_promotion_requires_budget(svc):
	with pytest.raises(AssertionError, match="budget cap"):
		run(svc.create_promotion(PrmPromotionCreate(
			tenant_id="t1", name="No Budget", promotion_type="percentage_off",
			discount_type="percentage", discount_value=10.0,
			budget_cap=0.0, margin_floor_pct=5.0,
			start_date=datetime.utcnow(),
			end_date=datetime.utcnow() + timedelta(days=1),
			created_by="admin",
		)))


def test_promotion_lifecycle(svc):
	p = _promo(svc)
	# Submit
	p = run(svc.submit_for_approval("t1", p.id, "author"))
	assert p.approval_status == "pending_review"
	# Approve
	p = run(svc.approve_promotion("t1", p.id, "manager"))
	assert p.approval_status == "approved"
	# Activate
	p = run(svc.activate_promotion("t1", p.id))
	assert p.approval_status == "active"
	# Pause
	p = run(svc.pause_promotion("t1", p.id))
	assert p.approval_status == "paused"


def test_activate_requires_approval(svc):
	p = _promo(svc)
	with pytest.raises(AssertionError, match="approved"):
		run(svc.activate_promotion("t1", p.id))


def test_reject_promotion(svc):
	p = _promo(svc)
	run(svc.submit_for_approval("t1", p.id, "author"))
	p = run(svc.reject_promotion("t1", p.id, "too aggressive", "manager"))
	assert p.approval_status == "rejected"


def test_apply_promotion(svc):
	p = _promo(svc)
	run(svc.submit_for_approval("t1", p.id, "author"))
	run(svc.approve_promotion("t1", p.id, "mgr"))
	run(svc.activate_promotion("t1", p.id))
	result = run(svc.apply_promotion("t1", p.id, 100.0, 3))
	assert result["applied"] is True
	assert result["discount_amount"] == pytest.approx(10.0)


def test_apply_promotion_margin_floor(svc):
	p = run(svc.create_promotion(PrmPromotionCreate(
		tenant_id="t1", name="Margin Breaker", promotion_type="percentage_off",
		discount_type="percentage", discount_value=99.0,
		budget_cap=50000.0, margin_floor_pct=5.0,
		start_date=datetime.utcnow(),
		end_date=datetime.utcnow() + timedelta(days=1),
		created_by="admin",
	)))
	run(svc.submit_for_approval("t1", p.id, "a"))
	run(svc.approve_promotion("t1", p.id, "m"))
	run(svc.activate_promotion("t1", p.id))
	result = run(svc.apply_promotion("t1", p.id, 100.0, 1))
	assert result["applied"] is False
	assert result["reason"] == "margin_floor_breach"


def test_coupon_create_and_redeem(svc):
	p = _promo(svc)
	coupon = run(svc.create_coupon(PrmCouponCreate(
		tenant_id="t1", promotion_id=p.id, coupon_type="single_use",
		coupon_code="SAVE10", max_uses=1,
		valid_from=datetime.utcnow(),
		valid_to=datetime.utcnow() + timedelta(days=30),
		created_by="admin",
	)))
	assert coupon.coupon_code == "SAVE10"
	red = run(svc.redeem_coupon(PrmCouponRedemptionCreate(
		tenant_id="t1", coupon_id=coupon.id, promotion_id=p.id,
		channel_id="ch-01", discount_applied=10.0, created_by="pos",
	)))
	assert red.id
	# Coupon should now be redeemed
	updated = run(svc.get_coupon_by_code("t1", "SAVE10"))
	assert updated.status == "redeemed"


def test_coupon_duplicate_code_rejected(svc):
	p = _promo(svc)
	run(svc.create_coupon(PrmCouponCreate(
		tenant_id="t1", promotion_id=p.id, coupon_type="single_use",
		coupon_code="DUPE", max_uses=1,
		valid_from=datetime.utcnow(),
		valid_to=datetime.utcnow() + timedelta(days=10),
		created_by="admin",
	)))
	with pytest.raises(ValueError, match="already exists"):
		run(svc.create_coupon(PrmCouponCreate(
			tenant_id="t1", promotion_id=p.id, coupon_type="single_use",
			coupon_code="DUPE", max_uses=1,
			valid_from=datetime.utcnow(),
			valid_to=datetime.utcnow() + timedelta(days=10),
			created_by="admin",
		)))


def test_markdown_creation(svc):
	md = run(svc.create_markdown(PrmMarkdownCreate(
		tenant_id="t1", name="End of Season", markdown_type="end_of_season",
		sku_list=["SKU-A", "SKU-B", "SKU-C"], markdown_pct=30.0,
		floor_margin_pct=10.0, effective_from=datetime.utcnow(),
		created_by="admin",
	)))
	assert md.items_affected == 3
	assert md.approval_status == "draft"


def test_markdown_floor_margin_breach(svc):
	with pytest.raises(AssertionError):
		run(svc.create_markdown(PrmMarkdownCreate(
			tenant_id="t1", name="Too Deep", markdown_type="clearance",
			sku_list=["X"], markdown_pct=101.0,
			floor_margin_pct=5.0, effective_from=datetime.utcnow(),
			created_by="admin",
		)))


def test_effectiveness_record(svc):
	p = _promo(svc)
	eff = run(svc.record_effectiveness(PrmEffectivenessRecord(
		tenant_id="t1", promotion_id=p.id,
		measurement_period_start=datetime.utcnow(),
		measurement_period_end=datetime.utcnow() + timedelta(days=7),
		redemption_rate=0.12, incremental_revenue=5000.0,
		margin_impact=-200.0, basket_uplift_pct=8.5,
		new_customer_acquisitions=30, repeat_purchase_rate=0.35, roi=2.5,
		calculated_by="analytics_agent",
	)))
	assert eff.roi == pytest.approx(2.5)


def test_tenant_isolation(svc):
	p = _promo(svc)
	assert run(svc.get_promotion("t2", p.id)) is None
