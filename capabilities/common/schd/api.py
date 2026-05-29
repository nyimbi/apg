"""Dependency-light API helpers for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import SchdService


def capability_status(service: SchdService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or SchdService()
	contract = get_capability_contract(tenant_id)
	return {
		"capability": "schd",
		"status": "ready",
		"contract": contract,
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def create_calendar_policy(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.create_calendar_policy(**payload)


def register_worker_pool(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.register_worker_pool(**payload)


def define_job(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.define_job(**payload)


def create_schedule(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.create_schedule(**payload)


def trigger_run(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.trigger_run(**payload)


def complete_run(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.complete_run(**payload)


def disable_schedule(service: SchdService, **payload: Any) -> dict[str, Any]:
	return service.disable_schedule(**payload)


def create_record(service: SchdService, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
	return service.create_record(record_id, tenant_id, metadata, status)


def list_records(service: SchdService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_records(tenant_id)


def list_schedules(service: SchdService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_schedules(tenant_id)


def list_runs(service: SchdService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_runs(tenant_id)
