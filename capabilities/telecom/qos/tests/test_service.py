"""Service-level tests for telecom_qos."""

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
	mod = _load("svc_desc_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_qos"


def test_all_qos_classes_accepted():
	mod = _load("svc_cls_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	for i, qclass in enumerate(["conversational", "streaming", "interactive", "background"]):
		pol = svc.create_qos_policy(f"pol-{i}", "t1", "bearer_qos", qclass, f"Policy {qclass}", "{}", f"approval-{i}", "eng")
		assert pol["qos_class"] == qclass


def test_downgrade_requires_approval():
	mod = _load("svc_down_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	svc.create_qos_policy("pol-1", "t1", "bearer_qos", "conversational", "VoLTE", "{}", "approval-1", "eng")
	with pytest.raises(PermissionError, match="qos_downgrade_approval_required"):
		svc.change_qos_policy("pol-1", "t1", '{"gbr": "64kbps"}', is_downgrade=True, approval_reference="")
	changed = svc.change_qos_policy("pol-1", "t1", '{"gbr": "64kbps"}', is_downgrade=True, approval_reference="downgrade-approval")
	assert changed["parameters"] == '{"gbr": "64kbps"}'


def test_sla_breach_direction_per_parameter():
	mod = _load("svc_sla_dir_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	latency_breach = svc.record_sla_measurement("m1", "t1", "max_latency_ms", 50.0, 20.0, None, "2026-01-01")
	assert latency_breach["is_breach"] is True
	throughput_breach = svc.record_sla_measurement("m2", "t1", "min_throughput_mbps", 5.0, 10.0, None, "2026-01-01")
	assert throughput_breach["is_breach"] is True
	latency_ok = svc.record_sla_measurement("m3", "t1", "max_latency_ms", 15.0, 20.0, None, "2026-01-01")
	assert latency_ok["is_breach"] is False


def test_degradation_evidence_required():
	mod = _load("svc_deg_ev_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	with pytest.raises(PermissionError, match="anomaly_evidence_required|degradation_cause_not_supported|degradation_evidence_required"):
		svc.record_degradation("d1", "t1", "congestion", 0.9, "desc", "res", "", "2026-01-01")


def test_disruptive_remediation_requires_approval():
	mod = _load("svc_disrupt_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	svc.create_qos_policy("pol-1", "t1", "bearer_qos", "conversational", "Policy", "{}", "approval-1", "eng")
	svc.record_degradation("deg-1", "t1", "congestion", 0.9, "desc", "BTS-001", "evidence-ref", "2026-01-01")
	with pytest.raises(PermissionError, match="disruptive_remediation_approval_required"):
		svc.trigger_remediation("rem-1", "t1", "deg-1", "bearer_reestablishment", is_disruptive=True, approval_reference=None, triggered_at="2026-01-01")
	rem = svc.trigger_remediation("rem-ok", "t1", "deg-1", "bearer_reestablishment", is_disruptive=True, approval_reference="approval-disruptive-1", triggered_at="2026-01-01")
	assert rem["is_disruptive"] is True


def test_multi_tenant_policy_isolation():
	mod = _load("svc_iso_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	svc.create_qos_policy("pol-1", "tenant-a", "bearer_qos", "conversational", "VoLTE-A", "{}", "approval-a", "eng")
	svc.create_qos_policy("pol-1", "tenant-b", "traffic_shaping", "streaming", "Video-B", "{}", "approval-b", "eng")
	assert svc.policies[("tenant-a", "pol-1")].policy_type == "bearer_qos"
	assert svc.policies[("tenant-b", "pol-1")].policy_type == "traffic_shaping"


def test_batch_stream_correct():
	mod = _load("svc_batch_qos", PACKAGE_DIR / "service.py")
	svc = mod.TelecomQosService()
	result = svc.validate_batch("t1", 5)
	assert result["stream"] == "apg.telecom.qos.lifecycle"
