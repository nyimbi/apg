"""Dependency-light API helpers for APG Scraper/Data Harvesting."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ScrpService


def capability_status(service: ScrpService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = service or ScrpService()
	contract = get_capability_contract(tenant_id)
	return {
		"capability": "scrp",
		"status": "ready",
		"contract": contract,
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def register_source(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.register_source(**payload)


def create_extractor_profile(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.create_extractor_profile(**payload)


def create_harvest_job(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.create_harvest_job(**payload)


def run_harvest(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.run_harvest(**payload)


def complete_harvest_run(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.complete_harvest_run(**payload)


def register_harvest_agent(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.register_harvest_agent(**payload)


def change_harvest_job_state(service: ScrpService, **payload: Any) -> dict[str, Any]:
	return service.change_harvest_job_state(**payload)


def create_record(service: ScrpService, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
	return service.create_record(record_id, tenant_id, metadata, status)


def list_records(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_records(tenant_id)


def list_sources(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_sources(tenant_id)


def list_extractors(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_extractors(tenant_id)


def list_jobs(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_jobs(tenant_id)


def list_runs(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_runs(tenant_id)


def list_results(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_results(tenant_id)


def list_handoffs(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_handoffs(tenant_id)


def list_harvest_agents(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.list_harvest_agents(tenant_id)


def audit_events(service: ScrpService, tenant_id: str | None = None) -> list[dict[str, Any]]:
	return service.audit_events(tenant_id)
