"""View models for APG Remote Workforce screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RemoteWorkforceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RemoteWorkforceService  # type: ignore


def _run(coro):
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view(service: RemoteWorkforceService, tenant_id: str = "default") -> dict[str, Any]:
	"""RWF top-level dashboard."""
	contract = get_capability_contract(tenant_id)
	summary = _run(service.dashboard_summary(tenant_id))
	return {
		"title": "Remote Workforce",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def work_policy_manager_view(service: RemoteWorkforceService, tenant_id: str, policy_type: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""Work policy manager list view."""
	policies = _run(service.list_work_policies(tenant_id, policy_type=policy_type, state=state))
	return {
		"tenant_id": tenant_id,
		"policies": [p.model_dump() for p in policies],
		"active_count": sum(1 for p in policies if p.state == "active"),
		"total_acknowledgments": sum(p.acknowledgment_count for p in policies),
	}


def policy_detail_view(service: RemoteWorkforceService, tenant_id: str, policy_id: str) -> dict[str, Any]:
	"""Work policy detail with acknowledgment list."""
	policy = _run(service.get_work_policy(tenant_id, policy_id))
	acks = _run(service.list_acknowledgments(tenant_id, policy_id=policy_id))
	return {
		"policy": policy.model_dump(),
		"acknowledgments": [a.model_dump() for a in acks],
		"acknowledgment_count": len(acks),
	}


def vpn_access_console_view(service: RemoteWorkforceService, tenant_id: str, employee_id: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""VPN access console view."""
	records = _run(service.list_vpn_access(tenant_id, employee_id=employee_id, state=state))
	return {
		"tenant_id": tenant_id,
		"vpn_access": [r.model_dump() for r in records],
		"active_count": sum(1 for r in records if r.state == "active"),
	}


def productivity_dashboard_view(service: RemoteWorkforceService, tenant_id: str) -> dict[str, Any]:
	"""Aggregated productivity dashboard."""
	metrics = _run(service.list_productivity_metrics(tenant_id))
	employee_ids = {m.employee_id for m in metrics}
	return {
		"tenant_id": tenant_id,
		"total_records": len(metrics),
		"employees_tracked": len(employee_ids),
		"metrics": [m.model_dump() for m in metrics],
	}


def employee_productivity_view(service: RemoteWorkforceService, tenant_id: str, employee_id: str) -> dict[str, Any]:
	"""Per-employee productivity view."""
	summary = _run(service.get_productivity_summary(tenant_id, employee_id))
	metrics = _run(service.list_productivity_metrics(tenant_id, employee_id=employee_id))
	return {
		"summary": summary,
		"metrics": [m.model_dump() for m in metrics],
	}


def equipment_inventory_view(service: RemoteWorkforceService, tenant_id: str, employee_id: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""Equipment inventory list view."""
	equipment = _run(service.list_equipment(tenant_id, employee_id=employee_id, state=state))
	return {
		"tenant_id": tenant_id,
		"equipment": [e.model_dump() for e in equipment],
		"pending_count": sum(1 for e in equipment if e.state == "requested"),
		"in_use_count": sum(1 for e in equipment if e.state == "delivered"),
	}


def onboarding_console_view(service: RemoteWorkforceService, tenant_id: str, state: str | None = None) -> dict[str, Any]:
	"""Onboarding console list view."""
	records = _run(service.list_onboarding_records(tenant_id, state=state))
	return {
		"tenant_id": tenant_id,
		"records": [r.model_dump() for r in records],
		"in_progress_count": sum(1 for r in records if r.state == "in_progress"),
		"completed_count": sum(1 for r in records if r.state == "completed"),
	}


def onboarding_detail_view(service: RemoteWorkforceService, tenant_id: str, record_id: str) -> dict[str, Any]:
	"""Onboarding detail view with steps."""
	record = _run(service.get_onboarding_record(tenant_id, record_id))
	return {
		"record": record.model_dump(),
		"progress_pct": int(len(record.completed_steps) / max(len(record.completed_steps) + len(record.pending_steps), 1) * 100),
	}


def compliance_dashboard_view(service: RemoteWorkforceService, tenant_id: str, result: str | None = None) -> dict[str, Any]:
	"""Remote compliance dashboard view."""
	checks = _run(service.list_compliance_checks(tenant_id, result=result))
	return {
		"tenant_id": tenant_id,
		"checks": [c.model_dump() for c in checks],
		"pass_count": sum(1 for c in checks if c.result == "pass"),
		"fail_count": sum(1 for c in checks if c.result == "fail"),
	}


def incident_queue_view(service: RemoteWorkforceService, tenant_id: str, state: str | None = "open") -> dict[str, Any]:
	"""Remote incident queue view."""
	incidents = _run(service.list_incidents(tenant_id, state=state))
	return {
		"tenant_id": tenant_id,
		"incidents": [i.model_dump() for i in incidents],
		"open_count": sum(1 for i in incidents if i.state == "open"),
	}
