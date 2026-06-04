"""View models for APG Performance Management screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomPerService


def dashboard_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Performance Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def kpi_console_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	critical = [k.to_dict() for k in service.kpis.values() if k.tenant_id == tenant_id and k.status == "critical"]
	return {"tenant_id": tenant_id, "critical_kpis": critical, "all_kpis": _items(service.kpis, tenant_id)}


def sla_console_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	breached = [s.to_dict() for s in service.sla_compliance.values() if s.tenant_id == tenant_id and s.status == "breached"]
	return {"tenant_id": tenant_id, "breached_slas": breached, "all_sla_compliance": _items(service.sla_compliance, tenant_id)}


def capacity_console_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	congested = [r.to_dict() for r in service.capacity_records.values() if r.tenant_id == tenant_id and r.capacity_state in ("congested", "overloaded")]
	return {"tenant_id": tenant_id, "congested_resources": congested, "all_capacity": _items(service.capacity_records, tenant_id)}


def trend_console_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	degrading = [t.to_dict() for t in service.trends.values() if t.tenant_id == tenant_id and t.trend_direction == "degrading"]
	return {"tenant_id": tenant_id, "degrading_trends": degrading, "all_trends": _items(service.trends, tenant_id), "benchmarks": _items(service.benchmarks, tenant_id)}


def report_console_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "reports": _items(service.reports, tenant_id), "thresholds": _items(service.thresholds, tenant_id)}


def agent_workbench_model(service: TelecomPerService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
