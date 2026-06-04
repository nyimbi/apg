"""Service tests for retail_loy capability."""

import asyncio
import pytest
from datetime import datetime, timedelta

from ..service import LoyService
from ..models import (
	LoyProgrammeCreate, LoyMemberCreate, LoyMemberUpdate,
	LoyTierCreate, LoyTransactionCreate, LoyCampaignCreate,
	LoyPartnerCreate, LoyRewardCreate, LoyClvSegmentRecord,
)


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return LoyService()


@pytest.fixture
def programme(svc):
	return run(svc.create_programme(LoyProgrammeCreate(
		tenant_id="t1", name="Acme Rewards", programme_type="points",
		created_by="admin",
	)))


@pytest.fixture
def member(svc, programme):
	return run(svc.enrol_member(LoyMemberCreate(
		tenant_id="t1", programme_id=programme.id,
		external_customer_id="cust-001",
		first_name="Alice", last_name="Smith",
		email="alice@example.com",
		consent_recorded=True, identity_verified=True,
		created_by="admin",
	)))


def test_create_programme(svc):
	prog = run(svc.create_programme(LoyProgrammeCreate(
		tenant_id="t1", name="Test Programme", programme_type="points", created_by="admin",
	)))
	assert prog.id
	assert prog.name == "Test Programme"
	assert prog.tenant_id == "t1"


def test_list_programmes_empty(svc):
	result = run(svc.list_programmes("unknown_tenant"))
	assert result == []


def test_enrol_member_requires_consent(svc, programme):
	with pytest.raises(AssertionError, match="consent"):
		run(svc.enrol_member(LoyMemberCreate(
			tenant_id="t1", programme_id=programme.id,
			external_customer_id="x", first_name="Bob", last_name="Jones",
			consent_recorded=False, identity_verified=True, created_by="admin",
		)))


def test_enrol_member_requires_identity(svc, programme):
	with pytest.raises(AssertionError, match="identity"):
		run(svc.enrol_member(LoyMemberCreate(
			tenant_id="t1", programme_id=programme.id,
			external_customer_id="x", first_name="Bob", last_name="Jones",
			consent_recorded=True, identity_verified=False, created_by="admin",
		)))


def test_enrol_member_success(svc, programme):
	m = run(svc.enrol_member(LoyMemberCreate(
		tenant_id="t1", programme_id=programme.id,
		external_customer_id="c2", first_name="Bob", last_name="Jones",
		consent_recorded=True, identity_verified=True, created_by="admin",
	)))
	assert m.id
	assert m.points_balance == 0
	assert m.status == "active"
	assert m.member_number.startswith("M")


def test_earn_points(svc, member):
	txn = run(svc.earn_points(LoyTransactionCreate(
		tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
		transaction_type="earn", points=500,
		earn_mechanism="purchase_amount", receipt_reference="rcpt-001",
		created_by="pos",
	)))
	assert txn.points == 500
	assert txn.balance_after == 500
	updated = run(svc.get_member("t1", member.id))
	assert updated.points_balance == 500


def test_earn_requires_receipt(svc, member):
	with pytest.raises(AssertionError, match="receipt"):
		run(svc.earn_points(LoyTransactionCreate(
			tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
			transaction_type="earn", points=100,
			earn_mechanism="purchase_amount", receipt_reference=None,
			created_by="pos",
		)))


def test_redeem_points(svc, member):
	# First earn
	run(svc.earn_points(LoyTransactionCreate(
		tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
		transaction_type="earn", points=1000,
		earn_mechanism="purchase_amount", receipt_reference="r1",
		created_by="pos",
	)))
	txn = run(svc.redeem_points(LoyTransactionCreate(
		tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
		transaction_type="redeem", points=-200,
		redeem_mechanism="discount", created_by="pos",
	)))
	assert txn.balance_after == 800


def test_redeem_insufficient_balance(svc, member):
	with pytest.raises(AssertionError, match="insufficient"):
		run(svc.redeem_points(LoyTransactionCreate(
			tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
			transaction_type="redeem", points=-9999,
			redeem_mechanism="discount", created_by="pos",
		)))


def test_adjust_prevents_negative_balance(svc, member):
	with pytest.raises(AssertionError, match="negative balance"):
		run(svc.adjust_points(LoyTransactionCreate(
			tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
			transaction_type="adjust", points=-1,
			created_by="admin",
		)))


def test_tenant_isolation(svc, programme):
	m1 = run(svc.enrol_member(LoyMemberCreate(
		tenant_id="t1", programme_id=programme.id,
		external_customer_id="c1", first_name="A", last_name="B",
		consent_recorded=True, identity_verified=True, created_by="a",
	)))
	# t2 cannot see t1 member
	assert run(svc.get_member("t2", m1.id)) is None


def test_freeze_and_reactivate(svc, member):
	frozen = run(svc.freeze_member("t1", member.id, "fraud review", "admin"))
	assert frozen.status == "frozen"
	active = run(svc.reactivate_member("t1", member.id, "admin"))
	assert active.status == "active"


def test_create_tier(svc, programme):
	tier = run(svc.create_tier(LoyTierCreate(
		tenant_id="t1", programme_id=programme.id,
		tier_name="gold", earn_multiplier=2.0,
		qualification_points=5000, created_by="admin",
	)))
	assert tier.tier_name == "gold"
	assert tier.earn_multiplier == 2.0


def test_create_campaign_requires_budget(svc, programme):
	with pytest.raises(AssertionError, match="budget cap"):
		run(svc.create_campaign(LoyCampaignCreate(
			tenant_id="t1", programme_id=programme.id,
			name="Xmas Bonus", campaign_type="bonus_points",
			start_date=datetime.utcnow(),
			end_date=datetime.utcnow() + timedelta(days=30),
			budget_cap_points=0,
			created_by="admin",
		)))


def test_campaign_lifecycle(svc, programme):
	c = run(svc.create_campaign(LoyCampaignCreate(
		tenant_id="t1", programme_id=programme.id,
		name="Spring Sale", campaign_type="double_points",
		start_date=datetime.utcnow(),
		end_date=datetime.utcnow() + timedelta(days=14),
		budget_cap_points=100000,
		created_by="admin",
	)))
	assert c.approval_status == "draft"
	approved = run(svc.approve_campaign("t1", c.id, "manager"))
	assert approved.approval_status == "approved"
	activated = run(svc.activate_campaign("t1", c.id))
	assert activated.approval_status == "active"


def test_activate_unapproved_campaign_fails(svc, programme):
	c = run(svc.create_campaign(LoyCampaignCreate(
		tenant_id="t1", programme_id=programme.id,
		name="Draft", campaign_type="bonus_points",
		start_date=datetime.utcnow(),
		end_date=datetime.utcnow() + timedelta(days=1),
		budget_cap_points=1000, created_by="admin",
	)))
	with pytest.raises(AssertionError, match="approved"):
		run(svc.activate_campaign("t1", c.id))


def test_clv_segment(svc, member):
	clv = run(svc.record_clv_segment(LoyClvSegmentRecord(
		tenant_id="t1", member_id=member.id, programme_id=member.programme_id,
		clv_score=0.85, clv_segment="high_value",
		predicted_12m_revenue=5000.0, recency_days=7,
		frequency_transactions=24, monetary_value=8000.0,
		calculated_by="clv_agent",
	)))
	assert clv.clv_segment == "high_value"
	fetched = run(svc.get_clv_segment("t1", member.id))
	assert fetched is not None
	assert fetched.clv_segment == "high_value"
	# member should be updated
	m = run(svc.get_member("t1", member.id))
	assert m.clv_segment == "high_value"
