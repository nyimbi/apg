"""Tests for telecom_ana capability contract and service."""

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


def test_contract_shape_and_required_keys():
	mod = _load("cc_ana", PACKAGE_DIR / "capability_contract.py")
	c = mod.get_capability_contract("tenant-test")
	assert c["capability"] == "telecom_ana"
	assert c["streaming"]["processor"] == "bytewax"
	assert c["theme"]["tokens"]["border.radius"] == "8px"
	assert "churn_prediction_workflow" in c["provides"]
	assert len(c["ui"]["routes"]) >= 8
	assert len(c["rule_engine"]["rules"]) >= 20
	assert "auth" in c["requires"]
	assert "mqeb" in c["requires"]


def test_rule_engine_blocks_missing_tenant_context():
	mod = _load("re_ana", PACKAGE_DIR / "capability_contract.py")
	assert mod.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "ana_batch", "event_stream": "kafka"})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True, "operation": "ana_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert mod.evaluate_capability_rules({"tenant_id": "t", "tenant_context_present": True})["decision"] == "allow"


def test_service_analytics_lifecycle():
	svc_mod = _load("svc_ana", PACKAGE_DIR / "service.py")
	svc = svc_mod.TelecomAnaService()

	model = svc.register_model("model-1", "t1", "regression", "Churn Model", "1.0.0", "val-ref", "analyst")
	run = svc.record_analysis_run("run-1", "t1", "churn_prediction", "owner-1", "daily", "2026-01-01", "2026-01-31", "evidence-ref")
	pred = svc.record_churn_prediction("pred-1", "t1", "cust-1", "high", 0.87, model["id"], "2026-01-31", "features-ref")
	rev = svc.record_revenue_event("rev-1", "t1", "data", 50000.0, "KES", "2026-01", "rev-evidence")
	seg = svc.record_segment("seg-1", "t1", "High Value", "custom", "arpu > 1000", 5000, "analyst")
	net = svc.record_network_analytics("net-1", "t1", "ran", "throughput", 80.0, 100.0, "2026-01-01T00:00:00")
	anom = svc.record_anomaly("anom-1", "t1", "revenue_leak", 0.92, "Missing CDRs", "evidence-ref", "2026-01-01")
	metric = svc.record_metric("metric-1", "t1", "kpi", "ARPU", 1250.0, "KES", 1000.0, "avg", "2026-01-01")
	report = svc.generate_report("rpt-1", "t1", "json", run["id"], "approval-ref", "analyst", "2026-02-01")
	batch = svc.validate_batch("t1", 10)
	summary = svc.dashboard_summary("t1")

	assert pred["risk_level"] == "high"
	assert rev["category"] == "data"
	assert seg["segment_name"] == "High Value"
	assert net["network_layer"] == "ran"
	assert anom["anomaly_type"] == "revenue_leak"
	assert metric["metric_name"] == "ARPU"
	assert report["report_format"] == "json"
	assert batch["processor"] == "bytewax"
	assert summary["model_count"] == 1
	assert summary["audit_event_count"] >= 9


def test_service_tenant_isolation():
	svc_mod = _load("svc_iso_ana", PACKAGE_DIR / "service.py")
	svc = svc_mod.TelecomAnaService()

	svc.register_model("m1", "tenant-a", "regression", "Model A", "1.0", "val-a", "analyst-a")
	svc.register_model("m1", "tenant-b", "classification", "Model B", "1.0", "val-b", "analyst-b")

	assert svc.dashboard_summary("tenant-a")["model_count"] == 1
	assert svc.dashboard_summary("tenant-b")["model_count"] == 1
	model_a = svc.models.get(("tenant-a", "m1"))
	model_b = svc.models.get(("tenant-b", "m1"))
	assert model_a.model_type == "regression"
	assert model_b.model_type == "classification"


def test_service_guardrails():
	svc_mod = _load("svc_guard_ana", PACKAGE_DIR / "service.py")
	svc = svc_mod.TelecomAnaService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.record_analysis_run("r", "", "churn_prediction", "owner", "daily", "", "", "ev")
	with pytest.raises(PermissionError, match="analysis_type_not_supported"):
		svc.record_analysis_run("r", "t1", "unknown_type", "owner", "daily", "", "", "ev")
	with pytest.raises(PermissionError, match="model_type_not_supported"):
		svc.register_model("m", "t1", "quantum_neural", "X", "1.0", "val", "owner")
	with pytest.raises(PermissionError, match="model_validation_required"):
		svc.register_model("m", "t1", "regression", "X", "1.0", "", "owner")
	with pytest.raises(PermissionError, match="anomaly_type_not_supported"):
		svc.record_anomaly("a", "t1", "alien_activity", 0.9, "desc", "ev", "2026-01-01")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="kafka")
	with pytest.raises(PermissionError, match="human_approval_required"):
		svc.validate_agent_action("t1", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="unapproved_model_deployment_scope_denied"):
		svc.validate_agent_action("t1", privileged_scope=False, human_approval_recorded=False, unapproved_model_deployment_scope=True)


def test_api_and_views():
	api = _load("api_ana", PACKAGE_DIR / "api.py")
	views = _load("views_ana", PACKAGE_DIR / "views.py")

	model = api.register_model({"tenant_id": "t-api", "model_id": "m-api", "model_type": "regression", "model_name": "API Model", "version": "1.0", "validation_reference": "val-ref", "registered_by": "analyst"})
	run = api.record_analysis_run({"tenant_id": "t-api", "run_id": "run-api", "analysis_type": "churn_prediction", "owner_id": "owner", "evidence_reference": "evidence"})
	batch = api.validate_batch({"tenant_id": "t-api", "item_count": 3})
	db = views.dashboard_model(api.service(), "t-api")
	console = views.analysis_console_model(api.service(), "t-api")
	registry = views.model_registry_model(api.service(), "t-api")

	assert model["model_type"] == "regression"
	assert run["analysis_type"] == "churn_prediction"
	assert batch["processor"] == "bytewax"
	assert db["summary"]["model_count"] == 1
	assert len(console["analysis_runs"]) == 1
	assert model["id"] in [m["id"] for m in registry["models"]]
