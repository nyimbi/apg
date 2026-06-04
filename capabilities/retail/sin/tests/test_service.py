"""Service tests for retail_sin capability."""

import asyncio
import pytest
from datetime import datetime, timedelta

from ..service import SinService
from ..models import (
	SinStoreCreate, SinZoneCreate, SinSensorCreate,
	SinTrafficCountCreate, SinPlanogramAuditCreate,
	SinShelfAlertCreate, SinConversionEventCreate,
	SinKpiSnapshotCreate, SinHeatmapCreate,
)


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


@pytest.fixture
def svc():
	return SinService()


@pytest.fixture
def store(svc):
	return run(svc.create_store(SinStoreCreate(
		tenant_id="t1", store_code="ST001", name="Main Street",
		store_format="supermarket",
		address={"street": "1 Main St", "city": "Nairobi"},
		latitude=-1.286389, longitude=36.817223,
		sqm_total=2000.0, sqm_selling=1500.0,
		created_by="admin",
	)))


@pytest.fixture
def zone(svc, store):
	return run(svc.create_zone(SinZoneCreate(
		tenant_id="t1", store_id=store.id, zone_code="Z01",
		zone_name="Entrance", zone_type="entrance",
		sqm=50.0, created_by="admin",
	)))


@pytest.fixture
def sensor(svc, store, zone):
	return run(svc.register_sensor(SinSensorCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		sensor_code="SEN-001", sensor_type="infrared_beam",
		created_by="admin",
	)))


def test_create_store(svc):
	s = run(svc.create_store(SinStoreCreate(
		tenant_id="t1", store_code="ST999", name="Test Store",
		store_format="convenience",
		address={"city": "Mombasa"},
		latitude=4.0435, longitude=39.6682,
		sqm_total=100.0, sqm_selling=80.0, created_by="admin",
	)))
	assert s.id
	assert s.store_format == "convenience"


def test_store_requires_positive_sqm(svc):
	with pytest.raises(AssertionError):
		run(svc.create_store(SinStoreCreate(
			tenant_id="t1", store_code="BAD", name="Bad",
			store_format="pop_up",
			address={}, latitude=0.0, longitude=0.0,
			sqm_total=0.0, sqm_selling=0.0, created_by="admin",
		)))


def test_create_zone(svc, store):
	z = run(svc.create_zone(SinZoneCreate(
		tenant_id="t1", store_id=store.id, zone_code="AISLE-1",
		zone_name="Beverages", zone_type="aisle",
		sqm=30.0, created_by="admin",
	)))
	assert z.zone_type == "aisle"


def test_sensor_heartbeat(svc, sensor):
	updated = run(svc.sensor_heartbeat("t1", sensor.id))
	assert updated.status == "online"
	assert updated.last_heartbeat_at is not None


def test_record_traffic_count(svc, store, zone, sensor):
	now = datetime.utcnow()
	tc = run(svc.record_traffic_count(SinTrafficCountCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		sensor_id=sensor.id,
		period_start=now, period_end=now + timedelta(minutes=1),
		entries=45, exits=30, occupancy_peak=120, dwell_avg_seconds=180.0,
		created_by="sensor",
	)))
	assert tc.entries == 45


def test_traffic_summary(svc, store, zone, sensor):
	now = datetime.utcnow()
	for i in range(3):
		run(svc.record_traffic_count(SinTrafficCountCreate(
			tenant_id="t1", store_id=store.id, zone_id=zone.id,
			sensor_id=sensor.id,
			period_start=now + timedelta(minutes=i),
			period_end=now + timedelta(minutes=i+1),
			entries=100, exits=80, occupancy_peak=200,
			dwell_avg_seconds=120.0, created_by="sensor",
		)))
	summary = run(svc.get_traffic_summary("t1", store.id, now, now + timedelta(minutes=5)))
	assert summary["total_entries"] == 300
	assert summary["record_count"] == 3


def test_planogram_audit(svc, store, zone):
	audit = run(svc.record_planogram_audit(SinPlanogramAuditCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		planogram_id="plano-001", audited_by="agent-01",
		audit_method="image_ai", compliance_status="minor_deviation",
		created_by="agent",
	)))
	assert audit.compliance_score_pct == 80.0


def test_compliance_rate_compliant(svc, store, zone):
	run(svc.record_planogram_audit(SinPlanogramAuditCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		planogram_id="p1", audited_by="a1", audit_method="manual",
		compliance_status="compliant", created_by="staff",
	)))
	rate = run(svc.get_store_compliance_rate("t1", store.id))
	assert rate == 100.0


def test_raise_and_resolve_shelf_alert(svc, store, zone, sensor):
	alert = run(svc.raise_shelf_alert(SinShelfAlertCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		sku="SKU-COLA", alert_type="out_of_stock", severity="critical",
		current_stock_level=0, detected_by=sensor.id, created_by="sensor",
	)))
	assert alert.status == "open"
	resolved = run(svc.resolve_shelf_alert("t1", alert.id, "restocked", "staff"))
	assert resolved.status == "resolved"


def test_replenishment_triggered(svc, store, zone, sensor):
	alert = run(svc.raise_shelf_alert(SinShelfAlertCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		sku="SKU-BREAD", alert_type="low_stock", severity="warning",
		current_stock_level=2, detected_by=sensor.id, created_by="sensor",
	)))
	rep = run(svc.trigger_replenishment("t1", alert.id))
	assert rep.replenishment_triggered is True


def test_heatmap_requires_pii_masking(svc, store):
	with pytest.raises(AssertionError, match="pii_masked"):
		run(svc.create_heatmap(SinHeatmapCreate(
			tenant_id="t1", store_id=store.id, floor_level=0,
			resolution="2m",
			period_start=datetime.utcnow(),
			period_end=datetime.utcnow() + timedelta(hours=1),
			grid_data=[], pii_masked=False, created_by="system",
		)))


def test_kpi_snapshot(svc, store):
	now = datetime.utcnow()
	snap = run(svc.record_kpi_snapshot(SinKpiSnapshotCreate(
		tenant_id="t1", store_id=store.id, kpi_category="traffic",
		period_type="daily", period_start=now,
		period_end=now + timedelta(hours=24),
		kpi_values={"total_entries": 3500.0, "conversion_rate": 0.32},
		benchmark_type="peer_group",
		benchmark_values={"total_entries": 3200.0, "conversion_rate": 0.30},
		created_by="analytics",
	)))
	assert snap.vs_benchmark_delta["total_entries"] == pytest.approx(300.0)


def test_store_performance_summary(svc, store, zone, sensor):
	run(svc.raise_shelf_alert(SinShelfAlertCreate(
		tenant_id="t1", store_id=store.id, zone_id=zone.id,
		sku="ITEM-X", alert_type="out_of_stock", severity="critical",
		detected_by=sensor.id, created_by="sensor",
	)))
	summary = run(svc.store_performance_summary("t1", store.id))
	assert summary["open_shelf_alerts"] >= 1
	assert summary["critical_alerts"] >= 1
	assert "planogram_compliance_pct" in summary


def test_tenant_isolation(svc, store):
	assert run(svc.get_store("t2", store.id)) is None
