"""View model builders for APG Clinical Management screens."""

from __future__ import annotations

import asyncio
from typing import Any

from .capability_contract import get_capability_contract
from .service import ClinicalManagementService


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def dashboard_view_model(service: ClinicalManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Clinical Management", "tenant_id": tenant_id, "summary": _run(service.dashboard_summary(tenant_id)), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def care_plan_list_view_model(service: ClinicalManagementService, tenant_id: str, patient_id: str | None = None, status: str | None = None) -> dict[str, Any]:
	plans = _run(service.list_care_plans(tenant_id, patient_id=patient_id, status=status))
	active = [p for p in plans if p.status == "active"]
	return {"title": "Care Plans", "tenant_id": tenant_id, "care_plans": [p.model_dump() for p in plans], "active_count": len(active), "filter": {"patient_id": patient_id, "status": status}}


def care_plan_detail_view_model(service: ClinicalManagementService, tenant_id: str, cp_id: str) -> dict[str, Any]:
	cp = _run(service.get_care_plan(tenant_id, cp_id))
	if cp is None:
		return {"error": "care_plan_not_found", "id": cp_id}
	workflows = _run(service.list_workflows(tenant_id, patient_id=cp.patient_id))
	return {"title": f"Care Plan: {cp.title}", "tenant_id": tenant_id, "care_plan": cp.model_dump(), "workflows": [w.model_dump() for w in workflows]}


def cds_alert_view_model(service: ClinicalManagementService, tenant_id: str, patient_id: str | None = None) -> dict[str, Any]:
	alerts = _run(service.list_cds_alerts(tenant_id, patient_id=patient_id))
	active = [a for a in alerts if a.status == "active"]
	critical = [a for a in active if a.priority == "critical"]
	return {"title": "Clinical Decision Support Alerts", "tenant_id": tenant_id, "alerts": [a.model_dump() for a in alerts], "active_count": len(active), "critical_count": len(critical)}


def handoff_view_model(service: ClinicalManagementService, tenant_id: str, patient_id: str | None = None) -> dict[str, Any]:
	handoffs = _run(service.list_handoffs(tenant_id, patient_id=patient_id))
	unack = [h for h in handoffs if h.acknowledged_by is None]
	return {"title": "Clinical Handoffs", "tenant_id": tenant_id, "handoffs": [h.model_dump() for h in handoffs], "unacknowledged_count": len(unack)}
