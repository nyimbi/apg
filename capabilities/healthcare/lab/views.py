"""View model builders for APG Laboratory Information System screens."""

from __future__ import annotations

import asyncio
from typing import Any

from .capability_contract import get_capability_contract
from .service import LaboratoryInformationService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: LaboratoryInformationService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Laboratory Information System", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def order_queue_view_model(service: LaboratoryInformationService, tenant_id: str, status: str | None = None) -> dict[str, Any]:
	orders = _run(service.list_orders(tenant_id, status=status))
	stat_count = sum(1 for o in orders if o.collection_priority == "stat")
	return {"title": "Lab Order Queue", "tenant_id": tenant_id, "orders": [o.model_dump() for o in orders], "stat_count": stat_count, "filter": {"status": status}}


def critical_values_view_model(service: LaboratoryInformationService, tenant_id: str) -> dict[str, Any]:
	all_cv = _run(service.list_critical_values(tenant_id))
	unack = [n for n in all_cv if n.acknowledged_by is None]
	return {"title": "Critical Values", "tenant_id": tenant_id, "notifications": [n.model_dump() for n in all_cv], "unacknowledged_count": len(unack)}


def qc_console_view_model(service: LaboratoryInformationService, tenant_id: str, instrument_id: str | None = None) -> dict[str, Any]:
	qc_runs = _run(service.list_qc_runs(tenant_id, instrument_id=instrument_id))
	failed = [q for q in qc_runs if q.status == "failed"]
	instruments = _run(service.list_instruments(tenant_id))
	return {
		"title": "QC Console",
		"tenant_id": tenant_id,
		"qc_runs": [q.model_dump() for q in qc_runs],
		"failed_count": len(failed),
		"instruments": [i.model_dump() for i in instruments],
		"filter": {"instrument_id": instrument_id},
	}


def specimen_tracker_view_model(service: LaboratoryInformationService, tenant_id: str, order_id: str | None = None) -> dict[str, Any]:
	specimens = _run(service.list_specimens(tenant_id, order_id=order_id))
	rejected = [s for s in specimens if s.status == "rejected"]
	return {"title": "Specimen Tracker", "tenant_id": tenant_id, "specimens": [s.model_dump() for s in specimens], "rejected_count": len(rejected), "filter": {"order_id": order_id}}
