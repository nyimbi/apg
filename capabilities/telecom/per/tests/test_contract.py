"""Tests for telecom_per capability contract and service."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def test_contract_shape():
	mod = _load("cc_per", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("t1")
	assert c["capability"] == "telecom_per"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "kpi_monitoring_workflow" in c["provides"]
	assert "capacity_utilisation_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20


def test_rule_engine():
	mod = _load("re_per", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "set_threshold", "threshold_action_supported": True, "approval_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "per_batch", "event_stream": "pubsub"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_performance_lifecycle():
	mod = _load("svc_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()

	kpi = svc.record_kpi("kpi-1", "t1", "radio_access", "Call Drop Rate", 2.5, 1.0, "%", "ran", "2026-01-01T00:00:00")
	updated_kpi = svc.update_kpi_status("kpi-1", "t1", "critical")
	sla = svc.record_sla_compliance("sla-1", "t1", "availability", "cust-ent-1", 99.9, 98.5, "2026-01")
	capacity = svc.record_capacity("cap-1", "t1", "BTS-001", "congested", 92.0, 90, "2026-01-01")
	trend = svc.record_trend("tnd-1", "t1", kpi["id"], "degrading", 30, 4.5, "2026-01-01")
	threshold = svc.set_threshold("thr-1", "t1", "Call Drop Rate", "ran", 2.0, 5.0, "alert_only", "approval-ref", "analyst")
	benchmark = svc.record_benchmark("bnch-1", "t1", "internal_target", "Call Drop Rate", 1.0, 2.5, "2026-01-01")
	report = svc.generate_report("rpt-1", "t1", "monthly", "pdf", "rpt-approval-ref", "analyst", "2026-02-01")
	agent = svc.register_agent("agt-1", "t1", "PER Agent", "codex", "kpi_analyst", "performance monitoring")
	batch = svc.validate_batch("t1", 5)
	summary = svc.dashboard_summary("t1")

	assert kpi["kpi_category"] == "radio_access"
	assert updated_kpi["status"] == "critical"
	assert sla["status"] == "breached"
	assert capacity["capacity_state"] == "congested"
	assert trend["trend_direction"] == "degrading"
	assert threshold["action"] == "alert_only"
	assert benchmark["gap_pct"] > 0
	assert report["report_period"] == "monthly"
	assert agent["role"] == "kpi_analyst"
	assert batch["processor"] == "bytewax"
	assert summary["sla_breach_count"] == 1
	assert summary["audit_event_count"] >= 8


def test_sla_compliance_pass():
	mod = _load("svc_sla_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()
	sla = svc.record_sla_compliance("sla-ok", "t1", "availability", None, 99.9, 99.95, "2026-01", notification_sent=False)
	assert sla["status"] == "compliant"


def test_guardrails():
	mod = _load("svc_guard_per", PACKAGE_DIR / "service.py")
	svc = mod.TelecomPerService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.record_kpi("k", "", "radio_access", "CDR", 1.0, 0.5, "%", "ran", "2026-01-01")
	with pytest.raises(PermissionError, match="kpi_category_not_supported"):
		svc.record_kpi("k", "t1", "alien_network", "metric", 1.0, 0.5, "%", "ran", "2026-01-01")
	with pytest.raises(PermissionError, match="threshold_change_approval_required"):
		svc.set_threshold("t", "t1", "CDR", "ran", 2.0, 5.0, "alert_only", "", "analyst")
	with pytest.raises(PermissionError, match="report_approval_required"):
		svc.generate_report("r", "t1", "monthly", "pdf", "", "analyst", "2026-02-01")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 1, event_stream="kinesis")
	with pytest.raises(PermissionError, match="unapproved_threshold_change_denied"):
		svc.validate_agent_action("t1", False, False, unapproved_threshold_change_scope=True)


def test_api_and_views():
	api = _load("api_per", PACKAGE_DIR / "api.py")
	views = _load("views_per", PACKAGE_DIR / "views.py")

	kpi = api.record_kpi({"tenant_id": "t-api", "kpi_id": "kpi-api", "kpi_category": "core_network", "kpi_name": "Latency", "value": 25.0, "baseline_value": 20.0, "unit": "ms"})
	sla = api.record_sla_compliance({"tenant_id": "t-api", "compliance_id": "sla-api", "sla_type": "latency", "target_value": 20.0, "actual_value": 25.0, "period": "2026-01", "notification_sent": True})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 2})
	db = views.dashboard_model(api.service(), "t-api")
	kpi_view = views.kpi_console_model(api.service(), "t-api")

	assert kpi["kpi_name"] == "Latency"
	assert sla["status"] == "breached"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["kpi_count"] == 1
	assert len(kpi_view["all_kpis"]) == 1
