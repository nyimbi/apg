"""Composable view models for APG Scraper/Data Harvesting."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ScrpService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Scraper/Data Harvesting",
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def source_registry_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"sources": service.list_sources(tenant_id),
		"guardrails": ["source_owner_required", "source_terms_required", "pii_policy_required", "sensitive_source_review_required"],
		"actions": ["register_source"],
	}


def job_monitor_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"jobs": service.list_jobs(tenant_id),
		"runs": service.list_runs(tenant_id),
		"actions": ["create_harvest_job", "run_harvest", "complete_harvest_run"],
	}


def extractor_workbench_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"extractors": service.list_extractors(tenant_id),
		"guardrails": ["schema_validation_required", "incremental_mode_supported"],
		"actions": ["create_extractor_profile"],
	}


def pipeline_handoff_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"handoffs": service.list_handoffs(tenant_id),
		"results": service.list_results(tenant_id),
		"guardrails": ["pipeline_handoff_required"],
	}


def compliance_review_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"sources": service.list_sources(tenant_id),
		"audit_events": service.audit_events(tenant_id),
		"guardrails": ["robots_policy_required", "pii_handling_policy_required", "restricted_source_review_required", "dlp_scan_required"],
	}


def results_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"results": service.list_results(tenant_id),
		"runs": service.list_runs(tenant_id),
	}


def settings_model(service: ScrpService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"permissions": ["scrp:view", "scrp:configure_sources", "scrp:run_jobs", "scrp:approve_harvests", "scrp:admin"],
	}
