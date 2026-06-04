"""View models for APG Mobile Device Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import MobileDeviceManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import MobileDeviceManagementService  # type: ignore


def _run(coro):
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view(service: MobileDeviceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	"""MDM dashboard view model."""
	contract = get_capability_contract(tenant_id)
	summary = _run(service.dashboard_summary(tenant_id))
	return {
		"title": "Mobile Device Management",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def device_inventory_view(service: MobileDeviceManagementService, tenant_id: str, os_platform: str | None = None, enrolment_state: str | None = None) -> dict[str, Any]:
	"""Device inventory list view."""
	devices = _run(service.list_devices(tenant_id, os_platform=os_platform, enrolment_state=enrolment_state))
	return {
		"tenant_id": tenant_id,
		"filters": {"os_platform": os_platform, "enrolment_state": enrolment_state},
		"devices": [d.model_dump() for d in devices],
		"count": len(devices),
	}


def device_detail_view(service: MobileDeviceManagementService, tenant_id: str, device_id: str) -> dict[str, Any]:
	"""Single device detail with policies, compliance, apps."""
	device = _run(service.get_device(tenant_id, device_id))
	assignments = _run(service.list_policy_assignments(tenant_id, device_id=device_id))
	compliance = _run(service.list_compliance_records(tenant_id, device_id=device_id))
	apps = _run(service.list_app_distributions(tenant_id, device_id=device_id))
	alerts = _run(service.list_alerts(tenant_id, device_id=device_id))
	return {
		"device": device.model_dump(),
		"policy_assignments": [a.model_dump() for a in assignments],
		"compliance_records": [c.model_dump() for c in compliance],
		"installed_apps": [a.model_dump() for a in apps],
		"alerts": [al.model_dump() for al in alerts],
	}


def policy_workbench_view(service: MobileDeviceManagementService, tenant_id: str, policy_type: str | None = None, state: str | None = None) -> dict[str, Any]:
	"""Policy workbench list view."""
	policies = _run(service.list_policies(tenant_id, policy_type=policy_type, state=state))
	return {
		"tenant_id": tenant_id,
		"policies": [p.model_dump() for p in policies],
		"count": len(policies),
		"active_count": sum(1 for p in policies if p.state == "active"),
	}


def compliance_dashboard_view(service: MobileDeviceManagementService, tenant_id: str, compliance_state: str | None = None) -> dict[str, Any]:
	"""Compliance dashboard view."""
	records = _run(service.list_compliance_records(tenant_id, compliance_state=compliance_state))
	devices = _run(service.list_devices(tenant_id))
	non_compliant = [d for d in devices if d.compliance_state == "non_compliant"]
	return {
		"tenant_id": tenant_id,
		"compliance_records": [r.model_dump() for r in records],
		"non_compliant_devices": [d.model_dump() for d in non_compliant],
		"non_compliant_count": len(non_compliant),
		"total_evaluated": len(records),
	}


def app_distribution_view(service: MobileDeviceManagementService, tenant_id: str, device_id: str | None = None) -> dict[str, Any]:
	"""App distribution console view."""
	dists = _run(service.list_app_distributions(tenant_id, device_id=device_id))
	return {
		"tenant_id": tenant_id,
		"distributions": [d.model_dump() for d in dists],
		"count": len(dists),
	}


def wipe_request_view(service: MobileDeviceManagementService, tenant_id: str, state: str | None = None) -> dict[str, Any]:
	"""Wipe request queue view."""
	wipes = _run(service.list_wipe_requests(tenant_id, state=state))
	return {
		"tenant_id": tenant_id,
		"wipe_requests": [w.model_dump() for w in wipes],
		"pending_count": sum(1 for w in wipes if w.state == "pending"),
	}


def profile_manager_view(service: MobileDeviceManagementService, tenant_id: str, profile_type: str | None = None) -> dict[str, Any]:
	"""Profile manager view."""
	profiles = _run(service.list_profiles(tenant_id, profile_type=profile_type))
	return {
		"tenant_id": tenant_id,
		"profiles": [p.model_dump() for p in profiles],
		"count": len(profiles),
	}


def alert_queue_view(service: MobileDeviceManagementService, tenant_id: str, resolved: bool | None = False) -> dict[str, Any]:
	"""MDM alert queue view."""
	alerts = _run(service.list_alerts(tenant_id, resolved=resolved))
	return {
		"tenant_id": tenant_id,
		"alerts": [a.model_dump() for a in alerts],
		"count": len(alerts),
	}
