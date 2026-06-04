"""Process-local API helpers for APG Project Baseline Management (pbl)."""

from __future__ import annotations

try:
	from .service import ProjectBaselineService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import ProjectBaselineService  # type: ignore

_SERVICE = ProjectBaselineService()


def service() -> ProjectBaselineService:
	return _SERVICE


def create_baseline(payload: dict):
	return _SERVICE.create_baseline(
		payload["baseline_id"], payload.get("tenant_id", "default"),
		payload["project_id"], payload["baseline_type"],
		payload.get("status", "draft"), payload["name"],
		payload.get("description", ""), payload["owner_id"],
		payload["approval_reference"], payload["evidence_reference"],
		payload.get("policy_attached", True),
	)


def approve_baseline(payload: dict):
	return _SERVICE.approve_baseline(
		payload["baseline_id"], payload.get("tenant_id", "default"),
		payload.get("designated_approver", True),
		payload["approval_reference"], payload["evidence_reference"],
	)


def submit_change_request(payload: dict):
	return _SERVICE.submit_change_request(
		payload["cr_id"], payload.get("tenant_id", "default"),
		payload["baseline_id"], payload["change_type"],
		payload.get("priority", "medium"), payload["title"],
		payload.get("description", ""), payload["submitter_id"],
		payload["impact_reference"], payload.get("approval_reference", ""),
		payload["evidence_reference"],
	)


def implement_change(payload: dict):
	return _SERVICE.implement_change(
		payload["cr_id"], payload.get("tenant_id", "default"),
		payload["approval_reference"],
	)


def assess_change_impact(payload: dict):
	return _SERVICE.assess_change_impact(
		payload["assessment_id"], payload.get("tenant_id", "default"),
		payload["change_request_id"], payload.get("impact_areas", "scope"),
		int(payload.get("schedule_impact_days", 0)),
		float(payload.get("cost_impact_amount", 0.0)),
		payload.get("scope_impact_description", ""),
		payload.get("risk_impact_description", ""),
		payload["assessor_id"], payload["evidence_reference"],
	)


def take_ev_snapshot(payload: dict):
	return _SERVICE.take_ev_snapshot(
		payload["snapshot_id"], payload.get("tenant_id", "default"),
		payload["baseline_id"], payload["snapshot_date"],
		float(payload["pv"]), float(payload["ev"]), float(payload["ac"]),
		float(payload["bac"]), payload["forecasting_method"],
		float(payload["eac"]), float(payload["etc"]),
	)


def generate_variance_report(payload: dict):
	return _SERVICE.generate_variance_report(
		payload["report_id"], payload.get("tenant_id", "default"),
		payload["baseline_id"], payload.get("report_period", ""),
		float(payload["schedule_variance"]), float(payload["cost_variance"]),
		float(payload["spi"]), float(payload["cpi"]),
		payload.get("variance_threshold", "standard"),
		payload.get("generated_by", "system"),
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "baseline management operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
