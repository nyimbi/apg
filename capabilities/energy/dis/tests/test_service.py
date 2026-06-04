"""Service tests for energy_dis capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="dis"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_dis_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import DistributionNetworkService


def make_svc() -> DistributionNetworkService:
	return DistributionNetworkService()


def _feeder(svc, fid="f1", tid="t1"):
	return svc.register_feeder(fid, tid, f"Feeder-{fid}", "ss1", "mv_11kv", 5.0, 7.0)


def test_register_feeder():
	svc = make_svc()
	f = _feeder(svc)
	assert f["id"] == "f1"
	assert f["voltage_level"] == "mv_11kv"


def test_register_feeder_unsupported_voltage():
	svc = make_svc()
	with pytest.raises(ValueError, match="voltage_level_not_supported"):
		svc.register_feeder("f2", "t1", "Bad Feeder", "ss1", "mv_999kv", 5.0, 7.0)


def test_register_element():
	svc = make_svc()
	_feeder(svc)
	el = svc.register_element("e1", "t1", "transformer", "TX-001", "f1", "mv_11kv", "grid_loc_1")
	assert el["element_type"] == "transformer"


def test_register_element_unsupported_type():
	svc = make_svc()
	_feeder(svc)
	with pytest.raises(ValueError, match="element_type_not_supported"):
		svc.register_element("e2", "t1", "alien_device", "AX", "f1", "mv_11kv", "loc")


def test_report_and_isolate_fault():
	svc = make_svc()
	_feeder(svc)
	svc.register_element("e1", "t1", "feeder", "F1", "f1", "mv_11kv", "loc1")
	fault = svc.report_fault("fault1", "t1", "e1", "phase_to_ground", "pole_123", 50)
	assert fault["status"] == "detected"
	isolated = svc.isolate_fault("fault1", "t1")
	assert isolated["status"] == "isolated"


def test_report_fault_missing_element():
	svc = make_svc()
	with pytest.raises(ValueError, match="element_not_found"):
		svc.report_fault("fx", "t1", "nonexistent", "equipment_failure", "loc", 0)


def test_dispatch_crew_requires_isolation():
	svc = make_svc()
	_feeder(svc)
	svc.register_element("e1", "t1", "feeder", "F1", "f1", "mv_11kv", "loc1")
	svc.report_fault("f1", "t1", "e1", "phase_to_ground", "loc1", 30)
	with pytest.raises(ValueError, match="fault_must_be_isolated_before_crew_dispatch"):
		svc.dispatch_crew("f1", "t1", "crew-001")


def test_switching_order_lifecycle():
	svc = make_svc()
	_feeder(svc)
	svc.register_element("e1", "t1", "switch", "SW1", "f1", "mv_11kv", "loc1")
	order = svc.create_switching_order("so1", "t1", "e1", "open", "op1", "fault isolation")
	assert order["status"] == "pending"
	approved = svc.approve_switching_order("so1", "t1", "supervisor@acme.com")
	assert approved["status"] == "approved"
	executed = svc.execute_switching_order("so1", "t1")
	assert executed["status"] == "executed"


def test_execute_switching_without_approval_raises():
	svc = make_svc()
	_feeder(svc)
	svc.register_element("e1", "t1", "switch", "SW1", "f1", "mv_11kv", "loc1")
	svc.create_switching_order("so1", "t1", "e1", "close", "op1", "restoration")
	with pytest.raises(ValueError, match="switching_approval_required"):
		svc.execute_switching_order("so1", "t1")


def test_record_and_restore_outage():
	svc = make_svc()
	_feeder(svc)
	outage = svc.record_outage("out1", "t1", "f1", "equipment_failure", "2026-06-01T08:00:00Z", "manual_switching", 200)
	assert outage["affected_customers"] == 200
	restored = svc.restore_outage("out1", "t1", saidi_minutes=45.0)
	assert restored["saidi_minutes"] == 45.0


def test_scada_reading_processing():
	svc = make_svc()
	r = svc.process_scada_reading(
		"r1", "t1", "e1", "dnp3", "voltage_pu", 1.02, "pu", "good",
		"2026-06-01T12:00:00Z", heartbeat_valid=True,
	)
	assert r["value"] == 1.02
	assert r["protocol"] == "dnp3"


def test_scada_heartbeat_expired_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="scada_heartbeat_expired"):
		svc.process_scada_reading(
			"r2", "t1", "e1", "dnp3", "current_a", 100.0, "A", "good",
			"2026-06-01T12:00:00Z", heartbeat_valid=False,
		)


def test_load_balance_action():
	svc = make_svc()
	_feeder(svc)
	action = svc.apply_load_balance("lb1", "t1", "f1", "manual", "load_transfer", 2.0, 0.01)
	assert action["load_transferred_mw"] == 2.0


def test_register_agent():
	svc = make_svc()
	agent = svc.register_agent("a1", "t1", "FaultBot", "claude_code", "fault_detector")
	assert agent["role"] == "fault_detector"


def test_register_agent_bad_role_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="agent_role_not_supported"):
		svc.register_agent("a2", "t1", "WeirdBot", "codex", "unknown_role")


def test_dashboard_summary():
	svc = make_svc()
	_feeder(svc)
	summary = svc.dashboard_summary("t1")
	assert summary["total_feeders"] == 1
	assert summary["active_faults"] == 0


def test_tenant_isolation():
	svc = make_svc()
	_feeder(svc, "f1", "t1")
	_feeder(svc, "f2", "t2")
	assert len(svc.list_feeders("t1")) == 1
	assert len(svc.list_feeders("t2")) == 1
