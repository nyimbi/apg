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


def change_sandbox_state(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.change_sandbox_state(**payload)


def create_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.create_script(**payload)


def request_script_review(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.request_script_review(**payload)


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


def cancel_execution(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.cancel_execution(**payload)


def retire_script(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.retire_script(**payload)


def register_scripting_agent(service: ScptService, **payload: Any) -> dict[str, Any]:
	return service.register_scripting_agent(**payload)


def validate_batch_mutation(service: ScptService, event_stream: str) -> dict[str, Any]:
	return service.validate_batch_mutation(event_stream)


def create_record(service: ScptService, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
	return service.create_record(record_id, tenant_id, metadata, status)


def list_records(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_records(tenant_id)


def list_scripts(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_scripts(tenant_id)


def list_executions(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_executions(tenant_id)


def list_agents(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_agents(tenant_id)


def audit_events(service: ScptService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.audit_events(tenant_id)
