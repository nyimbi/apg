"""Service tests for Property Management (prm)."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from capabilities.realestate.prm.service import PrmService
from capabilities.realestate.prm.models import (
	OwnerCreate, OwnerUpdate, OwnerType,
	PropertyCreate, PropertyUpdate, PropertyType, PropertyStatus,
	UnitCreate, UnitUpdate, UnitType, UnitStatus,
	PropertyAddress, OwnershipStructure, PortfolioTier, ManagementModel,
	KpiCalculationRequest, DistributionCreate, HandoverCreate,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return PrmService()


def _addr():
	return PropertyAddress(street="123 Main St", city="Nairobi", country="Kenya")


def _owner(svc):
	return loop.run_until_complete(svc.register_owner(OwnerCreate(
		tenant_id=T, name="Test Owner", owner_type=OwnerType.corporate,
		email="owner@test.com", created_by="u",
	)))


def _property(svc, owner_id):
	return loop.run_until_complete(svc.register_property(PropertyCreate(
		tenant_id=T, name="Test Tower", property_type=PropertyType.office,
		address=_addr(), owner_id=owner_id,
		ownership_structure=OwnershipStructure.freehold,
		portfolio_tier=PortfolioTier.core,
		management_model=ManagementModel.full_service,
		created_by="u",
	)))


# ── Owner ─────────────────────────────────────────────────────────────────────

def test_register_and_get_owner():
	svc = _svc()
	o = _owner(svc)
	assert o.id
	fetched = loop.run_until_complete(svc.get_owner(o.id, T))
	assert fetched.name == "Test Owner"


def test_list_owners():
	svc = _svc()
	for _ in range(3):
		_owner(svc)
	owners = loop.run_until_complete(svc.list_owners(T))
	assert len(owners) == 3


def test_update_owner():
	svc = _svc()
	o = _owner(svc)
	updated = loop.run_until_complete(svc.update_owner(o.id, T, OwnerUpdate(email="new@test.com")))
	assert updated.email == "new@test.com"


# ── Property ──────────────────────────────────────────────────────────────────

def test_register_and_get_property():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	assert p.id
	assert p.name == "Test Tower"
	fetched = loop.run_until_complete(svc.get_property(p.id, T))
	assert fetched.property_type == PropertyType.office


def test_property_linked_to_owner():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	owner = loop.run_until_complete(svc.get_owner(o.id, T))
	assert p.id in owner.property_ids


def test_list_properties_with_filters():
	svc = _svc()
	o = _owner(svc)
	for _ in range(2):
		_property(svc, o.id)
	all_props = loop.run_until_complete(svc.list_properties(T))
	assert len(all_props) == 2
	core = loop.run_until_complete(svc.list_properties(T, portfolio_tier="core"))
	assert len(core) == 2


def test_update_property_status():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	updated = loop.run_until_complete(svc.update_property(p.id, T, PropertyUpdate(status=PropertyStatus.vacant)))
	assert updated.status == PropertyStatus.vacant


def test_delete_property_without_board_approval_raises():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.delete_property(p.id, T, board_approved=False))


def test_delete_property_with_board_approval():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	deleted = loop.run_until_complete(svc.delete_property(p.id, T, board_approved=True))
	assert deleted is True


def test_sold_property_modification_denied():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	loop.run_until_complete(svc.update_property(p.id, T, PropertyUpdate(status=PropertyStatus.sold)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.update_property(p.id, T, PropertyUpdate(portfolio_tier=PortfolioTier.value_add)))


# ── Unit ──────────────────────────────────────────────────────────────────────

def test_create_and_get_unit():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	u = loop.run_until_complete(svc.create_unit(UnitCreate(
		tenant_id=T, property_id=p.id, unit_ref="U-101",
		unit_type=UnitType.office_suite, created_by="u",
	)))
	assert u.id
	assert u.status == UnitStatus.available


def test_list_units_by_status():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	for ref in ("U-1", "U-2", "U-3"):
		loop.run_until_complete(svc.create_unit(UnitCreate(
			tenant_id=T, property_id=p.id, unit_ref=ref,
			unit_type=UnitType.office_suite, created_by="u",
		)))
	void = loop.run_until_complete(svc.get_void_units(T, p.id))
	assert len(void) == 3


def test_update_unit_status():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	u = loop.run_until_complete(svc.create_unit(UnitCreate(
		tenant_id=T, property_id=p.id, unit_ref="U-201",
		unit_type=UnitType.retail_unit, created_by="u",
	)))
	updated = loop.run_until_complete(svc.update_unit(u.id, T, UnitUpdate(status=UnitStatus.let)))
	assert updated.status == UnitStatus.let


# ── KPI ───────────────────────────────────────────────────────────────────────

def test_calculate_occupancy_kpi():
	svc = _svc()
	o = _owner(svc)
	p = _property(svc, o.id)
	u1 = loop.run_until_complete(svc.create_unit(UnitCreate(tenant_id=T, property_id=p.id, unit_ref="K1", unit_type=UnitType.office_suite, created_by="u")))
	u2 = loop.run_until_complete(svc.create_unit(UnitCreate(tenant_id=T, property_id=p.id, unit_ref="K2", unit_type=UnitType.office_suite, created_by="u")))
	loop.run_until_complete(svc.update_unit(u1.id, T, UnitUpdate(status=UnitStatus.let)))
	req = KpiCalculationRequest(tenant_id=T, property_id=p.id, kpi_names=["occupancy_rate"], period="2025-01", requested_by="u")
	result = loop.run_until_complete(svc.calculate_kpis(req))
	occ = next(r for r in result.results if r.kpi_name == "occupancy_rate")
	assert occ.value == Decimal("50.00")


# ── Distribution ──────────────────────────────────────────────────────────────

def test_distribution_dual_control():
	svc = _svc()
	from datetime import date
	o = _owner(svc)
	p = _property(svc, o.id)
	dist = loop.run_until_complete(svc.create_distribution(DistributionCreate(
		tenant_id=T, owner_id=o.id, property_id=p.id,
		period="2025-01", gross_income=Decimal("200000"),
		net_distribution=Decimal("180000"), payment_date=date(2025, 1, 31), created_by="u",
	)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.approve_distribution(dist.id, T, "user1", "user1"))
	approved = loop.run_until_complete(svc.approve_distribution(dist.id, T, "user1", "user2"))
	assert approved.status == "approved"


# ── Portfolio Summary ─────────────────────────────────────────────────────────

def test_portfolio_summary():
	svc = _svc()
	summary = loop.run_until_complete(svc.get_portfolio_summary(T))
	assert "total_properties" in summary
	assert "occupancy_rate" in summary


# ── Search ────────────────────────────────────────────────────────────────────

def test_search_properties():
	svc = _svc()
	o = _owner(svc)
	_property(svc, o.id)
	results = loop.run_until_complete(svc.search_properties(T, "Test"))
	assert len(results) == 1
	no_results = loop.run_until_complete(svc.search_properties(T, "XYZNOTEXIST"))
	assert len(no_results) == 0
