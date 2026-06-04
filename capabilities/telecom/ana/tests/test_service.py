"""Service-level tests for telecom_ana."""

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
	mod = _load("svc_desc_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	contract = svc.describe("tenant-x")
	assert contract["capability"] == "telecom_ana"
	assert contract["configuration"]["tenant_id"] == "tenant-x"


def test_evaluate_allows_valid_context():
	mod = _load("svc_eval_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	result = svc.evaluate({"tenant_id": "t1", "tenant_context_present": True})
	assert result["decision"] == "allow"


def test_model_required_before_churn_prediction():
	mod = _load("svc_model_dep_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	with pytest.raises(PermissionError, match="churn_model_required"):
		svc.record_churn_prediction("pred", "t1", "cust", "high", 0.8, "nonexistent-model-id", "2026-01-01", "feat")


def test_confidence_score_must_be_bounded():
	mod = _load("svc_conf_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	svc.register_model("m1", "t1", "regression", "Model", "1.0", "val-ref", "owner")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		svc.record_churn_prediction("pred", "t1", "cust", "high", 1.5, "m1", "2026-01-01", "feat")


def test_revenue_event_records_correctly():
	mod = _load("svc_rev_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	ev = svc.record_revenue_event("rev-1", "t1", "voice", 100000.0, "KES", "2026-01", "ev-ref")
	assert ev["category"] == "voice"
	assert ev["amount"] == 100000.0


def test_network_analytics_all_supported_layers():
	mod = _load("svc_net_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	for layer in ["core", "radio", "transport", "ims", "cdn", "edge"]:
		rec = svc.record_network_analytics(f"rec-{layer}", "t1", layer, "throughput", 100.0, 50.0, "2026-01-01")
		assert rec["network_layer"] == layer


def test_segment_stores_criteria():
	mod = _load("svc_seg_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	seg = svc.record_segment("seg-1", "t1", "Prepaid High Value", "custom", "arpu > 500 AND plan = prepaid", 12000, "analyst")
	assert seg["criteria"] == "arpu > 500 AND plan = prepaid"
	assert seg["customer_count"] == 12000


def test_report_requires_approval():
	mod = _load("svc_rpt_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	with pytest.raises(PermissionError, match="report_approval_required"):
		svc.generate_report("rpt", "t1", "pdf", "run-1", "", "analyst", "2026-01-01")


def test_batch_requires_positive_count():
	mod = _load("svc_batch_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	with pytest.raises(ValueError):
		svc.validate_batch("t1", 0)


def test_agent_scope_required():
	mod = _load("svc_scope_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	with pytest.raises(PermissionError, match="ana_agent_scope_required"):
		svc.register_agent("agt", "t1", "Agent", "codex", "data_analyst", "")


def test_cross_tenant_data_denied():
	mod = _load("svc_cross_ana", PACKAGE_DIR / "service.py")
	svc = mod.TelecomAnaService()
	with pytest.raises(PermissionError, match="cross_tenant_data_scope_denied"):
		svc.validate_agent_action("t1", False, False, cross_tenant_data_scope=True)
