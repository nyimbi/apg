"""Screen-model helpers for the GRC Document Management capability."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import GrcDocService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import GrcDocService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/grc-doc/dashboard", "icon": "layout-dashboard"},
	{"name": "Documents", "route": "/grc-doc/documents", "icon": "file-text"},
	{"name": "Templates", "route": "/grc-doc/templates", "icon": "copy"},
	{"name": "Reviews", "route": "/grc-doc/reviews", "icon": "clipboard-check"},
	{"name": "Retention", "route": "/grc-doc/retention", "icon": "archive"},
	{"name": "Access", "route": "/grc-doc/access", "icon": "lock-keyhole"},
	{"name": "Processing", "route": "/grc-doc/processing", "icon": "workflow"},
	{"name": "Agents", "route": "/grc-doc/agents", "icon": "bot"},
	{"name": "Settings", "route": "/grc-doc/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"drafts": len([record for record in service.documents.values() if record["tenant_id"] == tenant_id and record["status"] == "draft"]),
		"reviews": len([record for record in service.documents.values() if record["tenant_id"] == tenant_id and record["status"] == "in_review"]),
		"restricted": len([record for record in service.documents.values() if record["tenant_id"] == tenant_id and record["classification"] == "restricted"]),
		"processing": len([record for record in service.processing_jobs.values() if record["tenant_id"] == tenant_id and record["status"] == "queued"]),
	}
	return model


def document_repository_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("documents", tenant_id)
	model["records"] = service.list_records("documents", tenant_id)
	model["columns"] = ["title", "owner_id", "document_type", "classification", "version", "status"]
	return model


def template_library_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("templates", tenant_id)
	model["records"] = service.list_records("templates", tenant_id)
	model["columns"] = ["name", "owner_id", "classification", "status"]
	return model


def review_queue_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("reviews", tenant_id)
	model["records"] = [
		record for record in service.list_records("documents", tenant_id)
		if record["status"] in {"draft", "in_review", "approved"}
	]
	model["columns"] = ["title", "owner_id", "classification", "reviewed_by", "approved_by", "status"]
	return model


def retention_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("retention", tenant_id)
	model["records"] = service.list_records("retention_policies", tenant_id)
	model["columns"] = ["document_id", "retention_days", "legal_hold", "status"]
	return model


def access_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("access", tenant_id)
	model["records"] = service.list_records("access_grants", tenant_id)
	model["columns"] = ["document_id", "principal_id", "permission", "expires_on", "status"]
	return model


def processing_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("processing", tenant_id)
	model["records"] = service.list_records("processing_jobs", tenant_id)
	model["columns"] = ["document_id", "job_type", "processor", "status"]
	return model


def agent_workbench_model(service: GrcDocService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_document", "classify_document", "review_retention", "review_access", "prepare_publication"]
	return model
