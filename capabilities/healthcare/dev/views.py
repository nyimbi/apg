"""View model builders for APG Medical Device Management screens."""

from __future__ import annotations

import asyncio
from typing import Any

from .capability_contract import get_capability_contract
from .service import MedicalDeviceManagementService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: MedicalDeviceManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Medical Device Management", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def device_list_view_model(service: MedicalDeviceManagementService, tenant_id: str, device_type: str | None = None, status: str | None = None) -> dict[str, Any]:
	devices = _run(service.list_devices(tenant_id, device_type=device_type, status=status))
	recalled = [d for d in devices if d.status == "recalled"]
	return {"title": "Device Inventory", "tenant_id": tenant_id, "devices": [d.model_dump() for d in devices], "recalled_count": len(recalled), "filter": {"device_type": device_type, "status": status}}


def adverse_event_view_model(service: MedicalDeviceManagementService, tenant_id: str) -> dict[str, Any]:
	events = _run(service.list_adverse_events(tenant_id))
	open_serious = [e for e in events if e.status == "open" and e.severity in ("serious", "life_threatening", "death")]
	return {"title": "Adverse Events", "tenant_id": tenant_id, "events": [e.model_dump() for e in events], "open_serious_count": len(open_serious)}


def calibration_view_model(service: MedicalDeviceManagementService, tenant_id: str, device_id: str | None = None) -> dict[str, Any]:
	cals = _run(service.list_calibrations(tenant_id, device_id=device_id))
	return {"title": "Calibration Records", "tenant_id": tenant_id, "calibrations": [c.model_dump() for c in cals], "filter": {"device_id": device_id}}
