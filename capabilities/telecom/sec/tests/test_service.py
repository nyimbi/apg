"""Service-level tests for telecom_sec."""

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
	mod = _load("svc_desc_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	c = svc.describe("t1")
	assert c["capability"] == "telecom_sec"


def test_fraud_case_full_lifecycle():
	mod = _load("svc_fraud_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	case = svc.raise_fraud_case("f1", "t1", "irsf", "+254700000001", 0.93, "cdr-ev", "2026-01-01")
	assert case["status"] == "open"
	blocked = svc.apply_fraud_block("f1", "t1", "cdr-ev")
	assert blocked["status"] == "blocked"


def test_all_supported_fraud_types_accepted():
	mod = _load("svc_fraud_types_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	for i, ftype in enumerate(["sim_swap_fraud", "wangiri", "irsf", "pbx_hacking", "account_takeover"]):
		case = svc.raise_fraud_case(f"f{i}", "t1", ftype, f"+25470000000{i}", 0.9, f"ev-{i}", "2026-01-01")
		assert case["fraud_type"] == ftype


def test_intercept_requires_both_warrant_and_authority():
	mod = _load("svc_int_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	with pytest.raises(PermissionError, match="intercept_warrant_required"):
		svc.activate_intercept("int", "t1", "voice_call", "+254700000001", "", "DCI", "2026-01-01", "2026-04-01")
	with pytest.raises(PermissionError, match="regulatory_authority_required"):
		svc.activate_intercept("int2", "t1", "voice_call", "+254700000001", "WARRANT-001", "", "2026-01-01", "2026-04-01")


def test_incident_full_lifecycle():
	mod = _load("svc_inc_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	inc = svc.open_incident("inc-1", "t1", "ss7_attack", "major", "SS7 probe detected", "pcap-ref", "2026-01-01")
	assert inc["status"] == "new"
	investigating = svc.update_incident_status("inc-1", "t1", "under_investigation")
	assert investigating["status"] == "under_investigation"
	resolved = svc.update_incident_status("inc-1", "t1", "closed", "2026-01-02")
	assert resolved["status"] == "closed"
	assert resolved["resolved_at"] == "2026-01-02"


def test_threat_intel_tlp_stored():
	mod = _load("svc_ti_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	ti = svc.record_threat_intel("ti-1", "t1", "carrier_community", "msisdn", "+254700000001", "red", "2026-01-01", "2026-07-01", False)
	assert ti["tlp_level"] == "red"
	assert ti["shared"] is False


def test_multi_tenant_fraud_isolation():
	mod = _load("svc_iso_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	svc.raise_fraud_case("f1", "tenant-a", "wangiri", "+25470000001", 0.9, "ev-a", "2026-01-01")
	svc.raise_fraud_case("f1", "tenant-b", "pbx_hacking", "+25470000002", 0.85, "ev-b", "2026-01-01")
	assert svc.dashboard_summary("tenant-a")["fraud_case_count"] == 1
	assert svc.dashboard_summary("tenant-b")["fraud_case_count"] == 1
	assert svc.fraud_cases[("tenant-a", "f1")].fraud_type == "wangiri"
	assert svc.fraud_cases[("tenant-b", "f1")].fraud_type == "pbx_hacking"


def test_ss7_and_diameter_attack_recorded():
	mod = _load("svc_sig_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	ss7 = svc.record_ss7_attack("s1", "t1", "call_interception", "src-001", "tgt-001", "pcap-1", "2026-01-01")
	dia = svc.record_diameter_attack("d1", "t1", "replay_attack", "evil.realm", "good.realm", "pcap-2", "2026-01-01")
	assert ss7["attack_type"] == "call_interception"
	assert dia["attack_type"] == "replay_attack"
	assert svc.dashboard_summary("t1")["ss7_attack_count"] == 1
	assert svc.dashboard_summary("t1")["diameter_attack_count"] == 1


def test_agent_requires_supported_runtime():
	mod = _load("svc_agt_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	with pytest.raises(PermissionError, match="sec_agent_runtime_not_supported"):
		svc.register_agent("agt", "t1", "Agent", "gpt5", "fraud_analyst", "security ops")


def test_validate_batch_correct():
	mod = _load("svc_vb_sec", PACKAGE_DIR / "service.py")
	svc = mod.TelecomSecService()
	result = svc.validate_batch("t1", 10)
	assert result["processor"] == "bytewax"
	assert result["stream"] == "apg.telecom.sec.lifecycle"
