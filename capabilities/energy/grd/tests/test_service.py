"""Service tests for energy_grd capability."""

from __future__ import annotations

import importlib.util as _ilu, sys as _sys, os as _os

def _load_cap(name, cap="grd"):
	_cap_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
	key = f"_ecap_grd_{name}"
	if key not in _sys.modules:
		spec = _ilu.spec_from_file_location(key, _os.path.join(_cap_dir, f"{name}.py"))
		mod  = _ilu.module_from_spec(spec)
		_sys.modules[key] = mod
		spec.loader.exec_module(mod)
	_sys.modules[name] = _sys.modules[key]

for _m in ("capability_contract", "models", "service", "views"):
	_load_cap(_m)

import pytest
from service import GridOperationsService


def make_svc() -> GridOperationsService:
	return GridOperationsService()


def test_run_state_estimation_converged():
	svc = make_svc()
	run = svc.run_state_estimation(
		"se1", "t1", "weighted_least_squares", "transmission",
		"model_ref_001", "meas_snap_001", iterations=5, converged=True, residual=1e-5,
	)
	assert run["converged"] is True
	assert run["status"] == "completed"


def test_run_state_estimation_not_converged():
	svc = make_svc()
	run = svc.run_state_estimation(
		"se2", "t1", "weighted_least_squares", "transmission",
		"model_ref_001", "meas_snap_001", iterations=50, converged=False, residual=0.5,
	)
	assert run["converged"] is False
	assert run["status"] == "failed_convergence"


def test_se_unsupported_type_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="state_estimator_type_not_supported"):
		svc.run_state_estimation("se3", "t1", "kalman_99", "transmission", "m", "s", 0, False, 0.0)


def test_get_latest_se_run():
	svc = make_svc()
	svc.run_state_estimation("se1", "t1", "weighted_least_squares", "transmission", "m", "s", 5, True, 1e-5)
	svc.run_state_estimation("se2", "t1", "weighted_least_squares", "transmission", "m", "s", 50, False, 0.5)
	latest = svc.get_latest_se_run("t1")
	assert latest["id"] == "se1"


def test_run_contingency_normal():
	svc = make_svc()
	case = svc.run_contingency(
		"c1", "t1", "n_minus_1", "Line-101-102",
		"base_case_ref", True, violations=[], max_overload_pct=80.0,
		min_voltage_pu=0.97, max_voltage_pu=1.03,
	)
	assert case["system_status"] == "normal"
	assert not case["has_violations"]


def test_run_contingency_emergency():
	svc = make_svc()
	case = svc.run_contingency(
		"c2", "t1", "n_minus_1", "Line-201", "base_ref", True,
		violations=[{"element": "Line-201", "overload_pct": 130}],
		max_overload_pct=130.0, min_voltage_pu=0.88, max_voltage_pu=1.03,
	)
	assert case["system_status"] == "emergency"
	assert case["has_violations"]


def test_contingency_no_base_case_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="converged_base_case_required"):
		svc.run_contingency("c3", "t1", "n_minus_1", "L1", "ref", base_case_converged=False,
		                    violations=[], max_overload_pct=0, min_voltage_pu=1.0, max_voltage_pu=1.0)


def test_apply_voltage_control():
	svc = make_svc()
	action = svc.apply_voltage_control("vc1", "t1", "tap_changer", "TX-101", 1.02, 1.018, "operator1")
	assert action["status"] == "completed"
	assert action["control_method"] == "tap_changer"


def test_voltage_control_no_approval_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="voltage_control_action_requires_approval"):
		svc.apply_voltage_control("vc2", "t1", "tap_changer", "TX-101", 1.02, 1.018, "")


def test_apply_frequency_control():
	svc = make_svc()
	action = svc.apply_frequency_control("fc1", "t1", "primary_frequency_response", 49.5, 50.0)
	assert action["response_mw"] == 50.0


def test_configure_ufls():
	svc = make_svc()
	result = svc.configure_ufls("t1", threshold_hz=49.0)
	assert result["ufls_threshold_hz"] == 49.0


def test_configure_ufls_invalid_threshold_raises():
	svc = make_svc()
	with pytest.raises(ValueError, match="ufls_threshold_invalid"):
		svc.configure_ufls("t1", threshold_hz=40.0)


def test_settle_market_interval():
	svc = make_svc()
	s = svc.settle_market_interval(
		"int1", "t1", "energy", "2026-06-01T00:00Z", "2026-06-01T00:30Z",
		100.0, 98.0, 50.0, "KES", "part1", "bid-ref-001",
	)
	assert s["status"] == "preliminary"
	assert s["imbalance_mwh"] == pytest.approx(2.0)
	assert s["settlement_amount"] == pytest.approx(5000.0)


def test_finalize_settlement():
	svc = make_svc()
	svc.settle_market_interval("int1", "t1", "energy", "S", "E", 100.0, 100.0, 50.0, "KES", "p1", "bid1")
	final = svc.finalize_settlement("int1", "t1")
	assert final["status"] == "final"


def test_raise_acknowledge_clear_alarm():
	svc = make_svc()
	alarm = svc.raise_alarm("al1", "t1", "thermal_overload", "warning", "Line-101", "Overload detected")
	assert alarm["status"] == "active"
	acked = svc.acknowledge_alarm("al1", "t1", "operator@acme.com")
	assert acked["acknowledged"] is True
	cleared = svc.clear_alarm("al1", "t1")
	assert cleared["status"] == "cleared"


def test_clear_critical_alarm_without_ack_raises():
	svc = make_svc()
	svc.raise_alarm("al2", "t1", "voltage_violation", "critical", "Bus-5", "Voltage collapse")
	with pytest.raises(ValueError, match="critical_alarm_must_be_acknowledged"):
		svc.clear_alarm("al2", "t1")


def test_execute_ems_function():
	svc = make_svc()
	result = svc.execute_ems_function("ems1", "t1", "optimal_power_flow", "real_time", "scheduler", {"cost_reduction_pct": 3.2})
	assert result["ems_function"] == "optimal_power_flow"
	assert result["status"] == "completed"


def test_register_agent():
	svc = make_svc()
	agent = svc.register_agent("ag1", "t1", "SEBot", "codex", "state_estimator")
	assert agent["role"] == "state_estimator"


def test_dashboard_summary():
	svc = make_svc()
	svc.run_state_estimation("se1", "t1", "weighted_least_squares", "transmission", "m", "s", 5, True, 1e-5)
	svc.raise_alarm("al1", "t1", "thermal_overload", "critical", "L1", "desc")
	summary = svc.dashboard_summary("t1")
	assert summary["last_se_converged"] is True
	assert summary["critical_alarms"] == 1
