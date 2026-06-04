"""Service tests for bia_dsh Dashboard Management."""

from __future__ import annotations
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import DashboardService


def _run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def test_create_dashboard():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "Sales Overview", "user1"))
	assert d["name"] == "Sales Overview"
	assert d["state"] == "draft"


def test_publish_dashboard():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	p = _run(svc.publish_dashboard("t1", d["id"]))
	assert p["state"] == "published"


def test_archive_dashboard():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	a = _run(svc.archive_dashboard("t1", d["id"]))
	assert a["state"] == "archived"


def test_list_dashboards_tenant_scoped():
	svc = DashboardService()
	_run(svc.create_dashboard("t1", "D1", "u1"))
	_run(svc.create_dashboard("t2", "D2", "u2"))
	assert len(_run(svc.list_dashboards("t1"))) == 1


def test_delete_dashboard():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	ok = _run(svc.delete_dashboard("t1", d["id"]))
	assert ok is True
	assert _run(svc.get_dashboard("t1", d["id"])) is None


def test_add_and_list_widgets():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	w = _run(svc.add_widget("t1", d["id"], "Revenue Chart", "bar_chart", "metric", "metric-123", "u1"))
	assert w["widget_type"] == "bar_chart"
	widgets = _run(svc.list_widgets("t1", d["id"]))
	assert len(widgets) == 1


def test_remove_widget():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	w = _run(svc.add_widget("t1", d["id"], "W1", "kpi_card", "metric", "m1", "u1"))
	ok = _run(svc.remove_widget("t1", w["id"]))
	assert ok is True


def test_widget_count_tracked():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	_run(svc.add_widget("t1", d["id"], "W1", "table", "query", "q1", "u1"))
	_run(svc.add_widget("t1", d["id"], "W2", "line_chart", "query", "q2", "u1"))
	d2 = _run(svc.get_dashboard("t1", d["id"]))
	assert d2["widget_count"] == 2


def test_take_snapshot():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	snap = _run(svc.take_snapshot("t1", d["id"], "png", "u1", "Weekly snapshot"))
	assert snap["format"] == "png"
	assert "storage_ref" in snap


def test_add_filter():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	f = _run(svc.add_filter("t1", d["id"], "Date Filter", "date_range", "created_at", "u1"))
	assert f["filter_type"] == "date_range"
	filters = _run(svc.list_filters("t1", d["id"]))
	assert len(filters) == 1


def test_remove_filter():
	svc = DashboardService()
	d = _run(svc.create_dashboard("t1", "D1", "u1"))
	f = _run(svc.add_filter("t1", d["id"], "F1", "dropdown", "status", "u1"))
	ok = _run(svc.remove_filter("t1", f["id"]))
	assert ok is True


def test_dashboard_stats():
	svc = DashboardService()
	_run(svc.create_dashboard("t1", "D1", "u1"))
	stats = _run(svc.get_dashboard_stats("t1"))
	assert stats["dashboard_count"] == 1


def test_audit_events_recorded():
	svc = DashboardService()
	_run(svc.create_dashboard("t1", "D1", "u1"))
	events = _run(svc.get_audit_events("t1"))
	assert len(events) >= 1
