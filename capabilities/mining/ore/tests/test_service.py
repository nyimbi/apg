"""Tests for OreService."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from capabilities.mining.ore.models import (
	AlertLevel,
	BalanceType,
	CircuitStatus,
	CircuitStatusUpdateCreate,
	DeviationAlertCreate,
	DeviationType,
	FeedSource,
	MetallurgicalBalanceCreate,
	PlantFeedCreate,
	ProductQualityCreate,
	ProductType,
	ReagentType,
	ReagentUsageCreate,
	ReconciliationStatus,
	StreamAssay,
)
from capabilities.mining.ore.service import OreService

TENANT = "test_ore_plant"


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def make_service():
	return OreService(tenant_id=TENANT)


def make_plant_feed() -> PlantFeedCreate:
	return PlantFeedCreate(
		tenant_id=TENANT,
		feed_source=FeedSource.ROM_ORE,
		period_start=datetime(2026, 1, 1, 0, 0),
		period_end=datetime(2026, 1, 1, 8, 0),
		dry_tonnes=2400.0,
		feed_grade=1.8,
		grade_units="g/t",
		moisture_pct=4.5,
		entered_by="plant_operator",
	)


def test_record_plant_feed():
	svc = make_service()
	result = run(svc.record_plant_feed(make_plant_feed(), created_by="plant_operator"))
	assert result.dry_tonnes == 2400.0
	assert result.feed_grade == 1.8


def test_list_plant_feeds():
	svc = make_service()
	run(svc.record_plant_feed(make_plant_feed(), created_by="plant_operator"))
	run(svc.record_plant_feed(
		PlantFeedCreate(
			tenant_id=TENANT, feed_source=FeedSource.CRUSHED_ORE,
			period_start=datetime(2026, 1, 1, 8, 0), period_end=datetime(2026, 1, 1, 16, 0),
			dry_tonnes=2200.0, feed_grade=1.6, grade_units="g/t", moisture_pct=3.8,
			entered_by="plant_operator",
		),
		created_by="plant_operator"
	))
	results = run(svc.list_plant_feeds())
	assert len(results) == 2


def test_record_reagent_usage():
	svc = make_service()
	usage = ReagentUsageCreate(
		tenant_id=TENANT,
		reagent_type=ReagentType.LIME,
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 1, 2),
		quantity_kg=1200.0,
		dosage_rate_g_t=500.0,
		circuit_id="leach_circuit_1",
		unit_cost=0.35,
		entered_by="plant_operator",
	)
	result = run(svc.record_reagent_usage(usage, created_by="plant_operator"))
	assert result.quantity_kg == 1200.0
	assert result.total_cost == pytest.approx(420.0)


def test_reagent_inventory_tracking():
	svc = make_service()
	run(svc.add_reagent_stock("lime", 5000.0))
	usage = ReagentUsageCreate(
		tenant_id=TENANT, reagent_type=ReagentType.LIME,
		period_start=datetime(2026, 1, 1), period_end=datetime(2026, 1, 2),
		quantity_kg=1000.0, dosage_rate_g_t=400.0, circuit_id="cil_1",
		entered_by="operator",
	)
	run(svc.record_reagent_usage(usage, created_by="operator"))
	inventory = run(svc.get_reagent_inventory())
	assert inventory["lime"] == pytest.approx(4000.0)


def test_submit_and_approve_met_balance():
	svc = make_service()
	balance = MetallurgicalBalanceCreate(
		tenant_id=TENANT,
		balance_type=BalanceType.DAILY,
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 1, 2),
		commodity="gold",
		recovery_method=__import__("capabilities.mining.ore.models", fromlist=["RecoveryMethod"]).RecoveryMethod.MASS_BALANCE,
		feed_stream=StreamAssay(sample_point="feed", dry_tonnes=7200.0, grade_value=1.8, grade_units="g/t"),
		concentrate_stream=StreamAssay(sample_point="concentrate", dry_tonnes=720.0, grade_value=15.2, grade_units="g/t"),
		tailings_stream=StreamAssay(sample_point="tailings", dry_tonnes=6480.0, grade_value=0.18, grade_units="g/t"),
		calculated_recovery_pct=85.3,
		prepared_by="met_engineer",
	)
	result = run(svc.submit_metallurgical_balance(balance, created_by="met_engineer"))
	assert result.status == ReconciliationStatus.OPEN

	approved = run(svc.approve_metallurgical_balance(result.id, "senior_met"))
	assert approved.status == ReconciliationStatus.APPROVED

	published = run(svc.publish_metallurgical_balance(result.id))
	assert published.published


def test_negative_recovery_rejected():
	"""Pydantic v2 rejects calculated_recovery_pct < 0 at model construction time."""
	from pydantic import ValidationError
	with pytest.raises(ValidationError):
		MetallurgicalBalanceCreate(
			tenant_id=TENANT,
			balance_type=BalanceType.DAILY,
			period_start=datetime(2026, 1, 1), period_end=datetime(2026, 1, 2),
			commodity="gold",
			recovery_method=__import__("capabilities.mining.ore.models", fromlist=["RecoveryMethod"]).RecoveryMethod.ASSAY_BASED,
			feed_stream=StreamAssay(sample_point="feed", dry_tonnes=1000.0, grade_value=2.0, grade_units="g/t"),
			calculated_recovery_pct=-5.0,
			prepared_by="met_engineer",
		)


def test_publish_requires_approval():
	svc = make_service()
	balance = MetallurgicalBalanceCreate(
		tenant_id=TENANT, balance_type=BalanceType.WEEKLY,
		period_start=datetime(2026, 1, 1), period_end=datetime(2026, 1, 8),
		commodity="copper",
		recovery_method=__import__("capabilities.mining.ore.models", fromlist=["RecoveryMethod"]).RecoveryMethod.MASS_BALANCE,
		feed_stream=StreamAssay(sample_point="feed", dry_tonnes=50000.0, grade_value=0.55, grade_units="%"),
		calculated_recovery_pct=87.0,
		prepared_by="met_engineer",
	)
	result = run(svc.submit_metallurgical_balance(balance, created_by="met_engineer"))
	with pytest.raises(PermissionError):
		run(svc.publish_metallurgical_balance(result.id))


def test_off_spec_product_warning():
	svc = make_service()
	quality = ProductQualityCreate(
		tenant_id=TENANT, product_type=ProductType.GOLD_DORE,
		lot_number="LOT-2026-001",
		sampled_at=datetime(2026, 1, 15),
		dry_weight_tonnes=0.025, moisture_pct=0.1,
		commodity_grade=85.0, grade_units="%",
		meets_specification=False,
		sampled_by="assayer_001",
	)
	result = run(svc.record_product_quality(quality, created_by="assayer_001"))
	assert not result.meets_specification


def test_raise_and_resolve_deviation_alert():
	svc = make_service()
	alert = DeviationAlertCreate(
		tenant_id=TENANT,
		deviation_type=DeviationType.RECOVERY_DEVIATION,
		alert_level=AlertLevel.HIGH,
		circuit_id="cil_circuit_1",
		description="Recovery dropped 5% below target",
		actual_value=80.0,
		target_value=88.0,
		units="%",
		detected_at=datetime(2026, 1, 20, 14, 0),
		detected_by="control_room",
	)
	result = run(svc.raise_deviation_alert(alert, created_by="control_room"))
	assert result.variance_pct == pytest.approx(9.09, rel=0.01)

	resolved = run(svc.resolve_deviation(result.id, "Carbon inventory replenished"))
	assert resolved.resolved


def test_update_circuit_status():
	svc = make_service()
	status_update = CircuitStatusUpdateCreate(
		tenant_id=TENANT,
		circuit_id="sag_mill_1",
		circuit_name="SAG Mill 1",
		circuit_type="sag_milling",
		status=CircuitStatus.RUNNING,
		throughput_tph=850.0,
		power_kw=12500.0,
		updated_by="control_room",
		updated_at=datetime(2026, 1, 20, 8, 0),
	)
	result = run(svc.update_circuit_status(status_update, created_by="control_room"))
	assert result.circuit_id == "sag_mill_1"
	assert result.status == CircuitStatus.RUNNING
