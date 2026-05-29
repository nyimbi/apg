"""Dependency-light API helpers for APG Custom Scripting Engine."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ScptService


def capability_status(service: ScptService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ScptService()
	contract = get_capability_contract(tenant_id)
	return {
		"capability": "scpt",
		"status": "ready",
		"contract": contract,
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def create_package_policy(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.create_package_policy(**payload)


def create_sandbox(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.create_sandbox(**payload)


def create_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.create_script(**payload)


def approve_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.approve_script(**payload)


def publish_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.publish_script(**payload)


def bind_workflow(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.bind_workflow(**payload)


def execute_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.execute_script(**payload)


def complete_execution(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.complete_execution(**payload)


def create_record(service: ScptService, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
	return service.create_record(record_id, tenant_id, metadata, status)


def list_records(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_records(tenant_id)


def list_scripts(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_scripts(tenant_id)


def list_executions(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_executions(tenant_id)
