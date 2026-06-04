"""Service tests for bia_anl Analytics Engine."""

from __future__ import annotations

import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service import AnalyticsEngineService


def _run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def test_describe_returns_contract():
	svc = AnalyticsEngineService()
	c = _run(svc.describe("acme"))
	assert c["capability"] == "bia_anl"


def test_register_datasource():
	svc = AnalyticsEngineService()
	ds = _run(svc.register_datasource(
		tenant_id="t1",
		name="Primary DB",
		datasource_type="postgresql",
		connection_config={"host": "localhost", "port": 5432, "dbname": "analytics"},
		credentials_vault_ref="vault://t1/primary-db",
		owner_id="user1",
	))
	assert ds["tenant_id"] == "t1"
	assert ds["name"] == "Primary DB"
	assert "id" in ds


def test_list_datasources_tenant_scoped():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS1", "postgresql", {}, "vault://t1/ds1", "u1"))
	_run(svc.register_datasource("t2", "DS2", "postgresql", {}, "vault://t2/ds2", "u2"))
	t1_list = _run(svc.list_datasources("t1"))
	assert all(d["tenant_id"] == "t1" for d in t1_list)
	assert len(t1_list) == 1


def test_save_and_get_query():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	ds_id = dss[0]["id"]
	q = _run(svc.save_query(
		tenant_id="t1",
		name="Revenue by Month",
		query_type="saved",
		sql_text="SELECT month, sum(revenue) FROM sales GROUP BY month",
		datasource_id=ds_id,
		owner_id="user1",
	))
	assert q["name"] == "Revenue by Month"
	fetched = _run(svc.get_query("t1", q["id"]))
	assert fetched["id"] == q["id"]


def test_query_not_found_returns_none():
	svc = AnalyticsEngineService()
	result = _run(svc.get_query("t1", "nonexistent-id"))
	assert result is None


def test_update_query():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	q = _run(svc.save_query("t1", "Q1", "saved", "SELECT 1", dss[0]["id"], "u1"))
	updated = _run(svc.update_query("t1", q["id"], {"name": "Q1 Updated"}))
	assert updated["name"] == "Q1 Updated"


def test_delete_query():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	q = _run(svc.save_query("t1", "ToDelete", "saved", "SELECT 1", dss[0]["id"], "u1"))
	ok = _run(svc.delete_query("t1", q["id"]))
	assert ok is True
	assert _run(svc.get_query("t1", q["id"])) is None


def test_create_and_refresh_cube():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	cube = _run(svc.create_cube(
		tenant_id="t1",
		name="Sales Cube",
		datasource_id=dss[0]["id"],
		dimensions=["time", "product"],
		measures=["sum", "count"],
		grain_sql="SELECT * FROM sales",
		owner_id="u1",
	))
	assert cube["state"] == "building"
	refreshed = _run(svc.refresh_cube("t1", cube["id"]))
	assert refreshed["state"] in ("active", "refreshing")


def test_define_metric():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	cube = _run(svc.create_cube("t1", "C1", dss[0]["id"], ["time"], ["sum"], "SELECT 1", "u1"))
	metric = _run(svc.define_metric(
		tenant_id="t1",
		name="Monthly Revenue",
		metric_type="kpi",
		formula="SUM(revenue)",
		cube_id=cube["id"],
		owner_id="u1",
		unit="USD",
	))
	assert metric["name"] == "Monthly Revenue"
	assert metric["unit"] == "USD"


def test_list_metrics_tenant_scoped():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	cube = _run(svc.create_cube("t1", "C1", dss[0]["id"], ["time"], ["sum"], "SELECT 1", "u1"))
	_run(svc.define_metric("t1", "M1", "kpi", "SUM(x)", cube["id"], "u1"))
	_run(svc.define_metric("t1", "M2", "derived", "AVG(y)", cube["id"], "u1"))
	metrics = _run(svc.list_metrics("t1"))
	assert len(metrics) == 2


def test_execute_query_returns_result():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	dss = _run(svc.list_datasources("t1"))
	q = _run(svc.save_query("t1", "Q", "ad_hoc", "SELECT 1 AS n", dss[0]["id"], "u1"))
	result = _run(svc.execute_query("t1", q["id"], {}))
	assert "columns" in result
	assert "rows" in result
	assert "execution_time_ms" in result


def test_audit_events_recorded():
	svc = AnalyticsEngineService()
	_run(svc.register_datasource("t1", "DS", "postgresql", {}, "vault://t1/ds", "u1"))
	events = _run(svc.get_audit_events("t1"))
	assert len(events) >= 1
	assert events[0]["tenant_id"] == "t1"


def test_dashboard_stats():
	svc = AnalyticsEngineService()
	stats = _run(svc.get_dashboard_stats("t1"))
	for key in ["query_count", "cube_count", "metric_count", "datasource_count"]:
		assert key in stats
