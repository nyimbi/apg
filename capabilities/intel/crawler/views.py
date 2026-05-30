"""View models for APG intelligence crawler screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_CRAWLER_AGENT_ROLES, SUPPORTED_CRAWLER_AGENT_RUNTIMES, get_capability_contract
	from .service import IntelligenceCrawlerService
except ImportError:
	from capability_contract import SUPPORTED_CRAWLER_AGENT_ROLES, SUPPORTED_CRAWLER_AGENT_RUNTIMES, get_capability_contract
	from service import IntelligenceCrawlerService


def navigation_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "dashboard",
		"title": "Intelligence Crawler",
		"summary": service.dashboard_summary(tenant_id),
		"sections": ["sources", "crawl_jobs", "extractions", "datasets", "validation", "knowledge"],
	}


def source_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "sources",
		"records": service.list_sources(tenant_id),
		"columns": ["source_id", "name", "owner", "source_type", "allowed_domains", "status"],
		"actions": ["register_source", "review_policy", "create_crawl_job"],
	}


def crawl_job_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "crawl_jobs",
		"records": service.list_crawl_jobs(tenant_id),
		"columns": ["job_id", "source_id", "cadence", "max_depth", "rate_limit_per_minute", "status"],
		"actions": ["create_crawl_job", "complete_crawl_job", "record_extraction"],
	}


def extraction_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "extractions",
		"records": service.list_extractions(tenant_id),
		"columns": ["extraction_id", "source_id", "schema_name", "quality_score", "status"],
		"actions": ["record_extraction", "open_validation_session", "publish_dataset"],
	}


def dataset_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "datasets",
		"records": service.list_datasets(tenant_id),
		"columns": ["dataset_id", "source_id", "contains_pii", "privacy_reviewed_by", "status"],
		"actions": ["publish_dataset", "record_rag_plan", "record_graph_projection"],
	}


def validation_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "validation",
		"records": service.list_validation_sessions(tenant_id),
		"columns": ["session_id", "reviewer", "confidence", "decision", "status"],
		"actions": ["open_validation_session", "complete_validation_session"],
	}


def rag_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "rag",
		"records": service.list_rag_plans(tenant_id),
		"columns": ["plan_id", "dataset_record_id", "chunk_plan", "chunk_size", "embedding_model", "status"],
		"actions": ["record_rag_plan", "prepare_vector_index"],
	}


def graph_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "graph",
		"records": service.list_graph_projections(tenant_id),
		"columns": ["projection_id", "dataset_record_id", "entity_schema", "status"],
		"actions": ["record_graph_projection", "review_relationship_evidence"],
	}


def agent_workbench_model(service: IntelligenceCrawlerService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "agents",
		"records": service.list_crawler_agents(tenant_id),
		"supported_runtimes": SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_CRAWLER_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
