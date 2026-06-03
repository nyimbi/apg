"""
Shared fixtures for Fleet Management tests.

No mocks — real objects only.  Async via plain asyncio.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from capabilities.transport.fle.models import (
	DriverCreate, FuelRecordCreate, InspectionCreate, InspectionResult,
	InspectionType, MaintenanceCreate, MaintenanceType, TripCreate,
	VehicleCreate, VehicleAssignmentCreate,
)
from capabilities.transport.fle.service import FleetService


class _DB:
	"""Minimal in-process store."""
	pass


@pytest.fixture
def db():
	return _DB()


@pytest.fixture
def svc(db):
	return FleetService(db, "tenant_test", "actor_test")


@pytest.fixture
def future_date():
	return datetime.utcnow() + timedelta(days=365 * 3)


@pytest.fixture
def vehicle_payload(future_date):
	return VehicleCreate(
		tenant_id="tenant_test",
		vehicle_type="rigid_truck",
		registration="KCA 001T",
		vin="WAUZZZ8K9BA123456",
		make="Mercedes-Benz",
		model="Actros",
		year=2022,
		fuel_type="diesel",
		ownership_type="owned",
		gross_vehicle_weight_kg=Decimal("26000"),
		payload_capacity_kg=Decimal("16000"),
	)


@pytest.fixture
def driver_payload(future_date):
	return DriverCreate(
		tenant_id="tenant_test",
		name="John Kamau",
		licence_number="DLN-001-KAMAU",
		licence_class="ce",
		licence_expiry=future_date,
		cpc_expiry=future_date,
		tacho_card_number="TC-001",
		phone="+254712000001",
	)


def run(coro):
	"""Run a coroutine synchronously in tests."""
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)
