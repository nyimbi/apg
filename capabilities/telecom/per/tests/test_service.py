"""Service-level tests for telecom_per."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_describe_returns_contract():
	mod = _load("svc_desc_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_per"


def test_all_kpi_categories_accepted():
	mod = _load("svc_kpi_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	for i, cat in enumerate(["radio_access", "core_network", "transmission", "ims_voice"]):
		kpi = svc.record_kpi(f"k{i}", "t1", cat, f"metric-{i}", 90.0, 95.0, "%", "core", "2026-01-01")
		assert kpi["kpi_category"] == cat


def test_sla_breach_triggers_audit():
	mod = _load("svc_breach_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	svc.record_sla_compliance("sla-1", "t1", "availability", None, 99.9, 98.0, "2026-01", notification_sent=True)
	assert any(e["event_type"] == "sla_breach_detected" for e in svc.audit_events)


def test_capacity_congestion_triggers_audit():
	mod = _load("svc_cap_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	svc.record_capacity("cap-1", "t1", "BTS-001", "overloaded", 98.0, 90, "2026-01-01")
	assert any(e["event_type"] == "capacity_congestion_alert" for e in svc.audit_events)


def test_benchmark_gap_calculation():
	mod = _load("svc_bench_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	bench = svc.record_benchmark("b1", "t1", "internal_target", "Availability", 99.9, 98.5, "2026-01-01")
	assert bench["gap_pct"] > 0
	zero_gap = svc.record_benchmark("b2", "t1", "internal_target", "Availability", 99.9, 99.9, "2026-01-01")
	assert zero_gap["gap_pct"] == 0.0


def test_threshold_all_actions_accepted():
	mod = _load("svc_thr_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	for i, action in enumerate(["alert_only", "escalate", "trigger_capacity_plan"]):
		thr = svc.set_threshold(f"t{i}", "t1", "kpi", "core", 80.0, 95.0, action, f"approval-{i}", "analyst")
		assert thr["action"] == action


def test_trend_degrading_triggers_audit():
	mod = _load("svc_trend_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	svc.record_kpi("k1", "t1", "radio_access", "CDR", 2.5, 1.0, "%", "ran", "2026-01-01")
	svc.record_trend("tnd-1", "t1", "k1", "degrading", 30, 4.5, "2026-01-01")
	assert any(e["event_type"] == "trend_degradation_detected" for e in svc.audit_events)


def test_multi_tenant_kpi_isolation():
	mod = _load("svc_iso_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	svc.record_kpi("k1", "tenant-a", "radio_access", "CDR", 1.5, 1.0, "%", "ran", "2026-01-01")
	svc.record_kpi("k1", "tenant-b", "core_network", "Latency", 20.0, 15.0, "ms", "core", "2026-01-01")
	assert svc.kpis[("tenant-a", "k1")].kpi_category == "radio_access"
	assert svc.kpis[("tenant-b", "k1")].kpi_category == "core_network"
