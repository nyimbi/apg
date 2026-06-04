"""Service tests for PPM Project Planning & Scheduling (pps)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load(name: str, path: Path):
	# Evict generic module names that other capabilities may have cached,
	# then prepend this capability's directory so fallback imports resolve correctly.
	_pkg = str(path.parent)
	for _key in ("capability_contract", "models", "service"):
		sys.modules.pop(_key, None)
	if _pkg not in sys.path:
		sys.path.insert(0, _pkg)
	else:
		sys.path.remove(_pkg)
		sys.path.insert(0, _pkg)
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec and spec.loader
	mod = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def _svc():
	mod = _load(f"svc_ppm_pps_{id(object())}", PACKAGE_DIR / "service.py")
	return mod.ProjectPlanningService()


def test_full_scheduling_lifecycle():
	svc = _svc()
	project = svc.create_project("proj-1", "t1", "ERP Implementation", "planned", "waterfall", "pm-1", "2026-01-01", "2026-12-31", "evidence-p")
	phase = svc.add_wbs_element("wbs-1", "t1", "proj-1", None, "phase", "1.0", "Planning Phase", "Initial planning")
	wp = svc.add_wbs_element("wbs-2", "t1", "proj-1", "wbs-1", "work_package", "1.1", "Requirements", "Gather requirements")
	task1 = svc.add_task("t-1", "t1", "proj-1", "wbs-2", "work_package", "not_started", "Gather Req", 5.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "2026-01-05", "2026-01-10")
	task2 = svc.add_task("t-2", "t1", "proj-1", "wbs-2", "work_package", "not_started", "Analyse Req", 3.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "2026-01-11", "2026-01-14")
	dep = svc.link_dependency("dep-1", "t1", "t-1", "t-2", "finish_to_start", 0.0)
	updated = svc.update_task_status("t-1", "t1", "in_progress", 50.0)
	cpm = svc.calculate_critical_path("cpm-1", "t1", "proj-1", "cpm", '["t-1","t-2"]', 0.0, 8.0, "2026-01-05")
	levelling = svc.level_resources("lv-1", "t1", "proj-1", "priority_based", 2, 1.0, "2026-01-05")
	calendar = svc.create_calendar("cal-1", "t1", "Standard Work Week", "standard_5x8", 8.0, '["monday","tuesday","wednesday","thursday","friday"]')
	agent = svc.register_agent("ag-1", "t1", "Schedule Bot", "codex", "schedule_builder", "scheduling")

	assert project["status"] == "planned"
	assert phase["level"] == "phase"
	assert task1["task_type"] == "work_package"
	assert dep["dependency_type"] == "finish_to_start"
	assert updated["status"] == "in_progress"
	assert updated["progress_pct"] == 50.0
	assert cpm["method"] == "cpm"
	assert levelling["over_allocations_resolved"] == 2
	assert calendar["calendar_type"] == "standard_5x8"
	assert agent["role"] == "schedule_builder"


def test_milestone_tracking():
	svc = _svc()
	svc.create_project("proj-m", "t1", "Milestone Test", "planned", "waterfall", "pm", "2026-01-01", "2026-12-31", "ev")
	wbs = svc.add_wbs_element("wbs-m", "t1", "proj-m", None, "phase", "1.0", "Phase", "")
	svc.add_task("ms-1", "t1", "proj-m", "wbs-m", "milestone", "not_started", "Go-Live", 0.0, "fixed_duration", "must_finish_on", "milestones_achieved", 0.0, "2026-06-30", "2026-06-30")
	milestones = [v.to_dict() for v in svc.tasks.values() if v.task_type == "milestone"]
	assert len(milestones) == 1


def test_circular_dependency_detection():
	svc = _svc()
	svc.create_project("proj-c", "t1", "Circ Test", "planned", "waterfall", "pm", "2026-01-01", "", "ev")
	wbs = svc.add_wbs_element("wbs-c", "t1", "proj-c", None, "phase", "1.0", "P", "")
	svc.add_task("tc-1", "t1", "proj-c", "wbs-c", "work_package", "not_started", "T1", 1.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "", "")
	svc.add_task("tc-2", "t1", "proj-c", "wbs-c", "work_package", "not_started", "T2", 1.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "", "")
	svc.link_dependency("dep-c1", "t1", "tc-1", "tc-2", "finish_to_start")
	with pytest.raises(PermissionError, match="wbs_circular_dependency_denied"):
		svc.link_dependency("dep-c2", "t1", "tc-2", "tc-1", "finish_to_start")


def test_tenant_isolation():
	svc = _svc()
	svc.create_project("p-a", "tenant-a", "A", "planned", "waterfall", "pm", "2026-01-01", "", "ev")
	svc.create_project("p-a", "tenant-b", "A", "planned", "waterfall", "pm", "2026-01-01", "", "ev")
	assert svc.dashboard_summary("tenant-a")["project_count"] == 1
	assert svc.dashboard_summary("tenant-b")["project_count"] == 1


def test_guardrail_unsupported_project_status():
	svc = _svc()
	with pytest.raises(PermissionError, match="project_status_not_supported"):
		svc.create_project("p", "t1", "X", "vaporware", "waterfall", "pm", "2026-01-01", "", "ev")


def test_guardrail_task_requires_wbs():
	svc = _svc()
	svc.create_project("p-tw", "t1", "TW", "planned", "waterfall", "pm", "2026-01-01", "", "ev")
	with pytest.raises(PermissionError, match="wbs_element_required"):
		svc.add_task("t", "t1", "p-tw", "missing-wbs", "work_package", "not_started", "T", 1.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "", "")


def test_guardrail_zero_duration_task():
	svc = _svc()
	svc.create_project("p-zd", "t1", "ZD", "planned", "waterfall", "pm", "2026-01-01", "", "ev")
	wbs = svc.add_wbs_element("wbs-zd", "t1", "p-zd", None, "phase", "1.0", "P", "")
	with pytest.raises(PermissionError, match="task_duration_must_be_positive"):
		svc.add_task("t-zd", "t1", "p-zd", "wbs-zd", "work_package", "not_started", "Zero", 0.0, "fixed_duration", "as_soon_as_possible", "percent_complete", 0.0, "", "")


def test_batch_requires_bytewax():
	svc = _svc()
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		svc.validate_batch("t1", 5, event_stream="rabbitmq")
