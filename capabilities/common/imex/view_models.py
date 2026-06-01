"""Generated-app UI view models for the IMEX capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .imex_runtime import ImexService


def dashboard_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Import/Export", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "transfer_agents": service.list_transfer_agents(tenant_id), "lifecycle_batches": service.list_lifecycle_batches(tenant_id), "pending_reviews": service.list_pending_reviews(tenant_id), "review_evidence": contract["review_evidence"], "routes": contract["ui"]["routes"], "theme": contract["theme"]}


def job_designer_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "endpoints": service.list_endpoints(tenant_id), "mappings": service.list_mappings(tenant_id), "formats": contract["configuration"]["formats"], "actions": ["create_job", "validate_preview", "execute_job"]}


def mapping_workbench_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "mappings": service.list_mappings(tenant_id), "validation": get_capability_contract(tenant_id)["configuration"]["validation"]}


def transfer_monitor_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "jobs": service.list_jobs(tenant_id), "runs": service.list_runs(tenant_id), "observability": get_capability_contract(tenant_id)["configuration"]["observability"]}


def validation_console_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "jobs": service.list_jobs(tenant_id), "validation": get_capability_contract(tenant_id)["configuration"]["validation"]}


def import_workbench_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "jobs": [job for job in service.list_jobs(tenant_id) if job["direction"] in {"import", "migration"}], "endpoints": service.list_endpoints(tenant_id)}


def export_workbench_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "jobs": [job for job in service.list_jobs(tenant_id) if job["direction"] == "export"], "artifacts": service.list_artifacts(tenant_id)}


def approvals_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "reviews": service.list_reviews(tenant_id), "security": get_capability_contract(tenant_id)["configuration"]["security"]}


def artifacts_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "artifacts": service.list_artifacts(tenant_id), "retention_days": get_capability_contract(tenant_id)["configuration"]["jobs"]["retention_days"]}


def audit_timeline_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "events": service.list_audit_events(tenant_id), "observability": get_capability_contract(tenant_id)["configuration"]["observability"]}


def transfer_agent_roster_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "agents": service.list_transfer_agents(tenant_id), "pending_reviews": [record for record in service.list_pending_reviews(tenant_id) if record.get("role") in contract["agents"]["supported_roles"]], "agent_contract": contract["agents"], "supported_runtimes": contract["agents"]["supported_runtimes"], "supported_roles": contract["agents"]["supported_roles"], "privileged_roles": contract["agents"]["privileged_roles"], "required_fields": ["name", "runtime", "role", "scope", "owner", "purpose", "contribution_disclosed"]}


def lifecycle_batch_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "batches": service.list_lifecycle_batches(tenant_id), "streaming": contract["streaming"], "required_processor": contract["streaming"]["required_processor"], "summary": service.dashboard_summary(tenant_id)}


def settings_model(service: ImexService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "configuration": contract["configuration"], "schema": contract["configuration_schema"], "adapters": contract["configuration"]["adapters"], "agents": contract["agents"], "streaming": contract["streaming"], "review_evidence": contract["review_evidence"], "theme": contract["theme"]}
