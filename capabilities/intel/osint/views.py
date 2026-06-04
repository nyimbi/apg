"""View models for OSINT screens.

Each function returns a plain dict ready for a Jinja2 template or a
JSON response — no rendering logic here.
"""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import OSINTService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import OSINTService  # type: ignore


# ---------------------------------------------------------------------------
# Async view model builders
# ---------------------------------------------------------------------------

async def dashboard_model(
	service: OSINTService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""KPI dashboard view model."""
	contract = get_capability_contract(tenant_id)
	summary = await service.dashboard_summary()
	return {
		"title": "Open Source Intelligence",
		"tenant_id": tenant_id,
		"summary": summary.model_dump(),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"streaming": contract["streaming"],
	}


async def source_list_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Source registry list view model."""
	filters = filters or {}
	sources = await service.list_sources(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	report = await service.source_health_report()
	return {
		"title": "Source Registry",
		"tenant_id": tenant_id,
		"sources": [s.model_dump() for s in sources],
		"health": report.model_dump(),
		"supported_source_types": get_capability_contract(tenant_id)["configuration"]["sources"]["supported_types"],
		"supported_risk_tiers": get_capability_contract(tenant_id)["configuration"]["sources"]["supported_risk_tiers"],
	}


async def task_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Collection task console view model."""
	filters = filters or {}
	tasks = await service.list_tasks(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Collection Tasks",
		"tenant_id": tenant_id,
		"tasks": [t.model_dump() for t in tasks],
		"supported_task_types": get_capability_contract(tenant_id)["configuration"]["tasks"]["supported_types"],
	}


async def raw_intel_ledger_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Raw intelligence ledger view model."""
	filters = filters or {}
	items = await service.list_raw_intel(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Raw Intelligence Ledger",
		"tenant_id": tenant_id,
		"items": [i.model_dump() for i in items],
	}


async def processed_intel_workbench_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Processed intelligence workbench view model."""
	filters = filters or {}
	items = await service.list_processed_intel(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Processed Intelligence",
		"tenant_id": tenant_id,
		"items": [i.model_dump() for i in items],
		"supported_assessment_types": get_capability_contract(tenant_id)["configuration"]["processed_intelligence"]["supported_assessment_types"],
	}


async def entity_graph_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Entity graph view model — includes relationship network."""
	filters = filters or {}
	entities = await service.list_entities(
		limit=filters.get("limit", 200),
		offset=filters.get("offset", 0),
	)
	relationships = await service.list_relationships(
		limit=filters.get("rel_limit", 500),
		offset=0,
	)
	network = await service.relationship_mapping()
	return {
		"title": "Entity Graph",
		"tenant_id": tenant_id,
		"entities": [e.model_dump() for e in entities],
		"relationships": [r.model_dump() for r in relationships],
		"clusters": network.clusters,
		"high_confidence_links": network.high_confidence_links,
		"supported_entity_types": get_capability_contract(tenant_id)["configuration"]["entities"]["supported_types"],
		"supported_relationship_types": get_capability_contract(tenant_id)["configuration"]["relationships"]["supported_types"],
	}


async def social_profile_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Social media profile console view model."""
	filters = filters or {}
	profiles = await service.list_social_profiles(
		platform=filters.get("platform"),
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Social Media Profiles",
		"tenant_id": tenant_id,
		"profiles": [p.model_dump() for p in profiles],
	}


async def web_content_ledger_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Web content ledger view model."""
	filters = filters or {}
	items = await service.list_web_content(
		task_id=filters.get("task_id"),
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Web Content",
		"tenant_id": tenant_id,
		"items": [i.model_dump() for i in items],
	}


async def domain_intel_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Domain intelligence console view model."""
	filters = filters or {}
	records = await service.find_domain_records(
		domain=filters.get("domain"),
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Domain Intelligence",
		"tenant_id": tenant_id,
		"records": [r.model_dump() for r in records],
	}


async def ip_intel_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""IP intelligence console view model."""
	filters = filters or {}
	items = await service.find_ip_intel(
		ip_address=filters.get("ip_address"),
		country_code=filters.get("country_code"),
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "IP Intelligence",
		"tenant_id": tenant_id,
		"items": [i.model_dump() for i in items],
	}


async def document_analysis_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Document analysis console view model."""
	filters = filters or {}
	analyses = await service.list_document_analyses(
		raw_intel_id=filters.get("raw_intel_id"),
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Document Analysis",
		"tenant_id": tenant_id,
		"analyses": [a.model_dump() for a in analyses],
	}


async def dissemination_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Dissemination console view model."""
	filters = filters or {}
	packages = await service.list_dissemination_packages(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "Intelligence Dissemination",
		"tenant_id": tenant_id,
		"packages": [p.model_dump() for p in packages],
		"supported_tlp": get_capability_contract(tenant_id)["configuration"]["dissemination"]["supported_tlp"],
	}


async def review_console_model(
	service: OSINTService,
	tenant_id: str = "default",
	filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Review console view model."""
	filters = filters or {}
	reviews = await service.list_reviews(
		limit=filters.get("limit", 50),
		offset=filters.get("offset", 0),
	)
	return {
		"title": "OSINT Reviews",
		"tenant_id": tenant_id,
		"reviews": [r.model_dump() for r in reviews],
		"supported_statuses": get_capability_contract(tenant_id)["configuration"]["reviews"]["supported_statuses"],
	}


async def agent_workbench_model(
	service: OSINTService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Agent workbench view model."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "OSINT Agent Workbench",
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


async def threat_landscape_model(
	service: OSINTService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Threat landscape report view model."""
	report = await service.threat_landscape_report()
	return {
		"title": "Threat Landscape",
		"tenant_id": tenant_id,
		"report": report.model_dump(),
	}


# ---------------------------------------------------------------------------
# Legacy synchronous view models (used by test_package_contract.py)
# ---------------------------------------------------------------------------

def dashboard_model(svc: OSINTService, tenant_id: str = "default") -> dict[str, Any]:  # type: ignore[override]
	"""Dashboard KPI view model — synchronous legacy interface.

	Args:
		svc: An OSINTService instance.
		tenant_id: Tenant context.

	Returns:
		Dict with 'summary' containing KPI counts.
	"""
	summary = svc._sync_dashboard_summary(tenant_id)
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Open Source Intelligence",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"streaming": contract["streaming"],
	}


def osint_console_model(svc: OSINTService, tenant_id: str = "default") -> dict[str, Any]:
	"""OSINT operations console view model — synchronous legacy interface.

	Returns collection plans, requirements, agents, and source lists for the
	given tenant.

	Args:
		svc: An OSINTService instance.
		tenant_id: Tenant context.

	Returns:
		Dict with 'collection_plans', 'requirements', 'agents', 'sources'.
	"""
	collection_plans = [
		v for (t, _), v in getattr(svc, "_collection_plans", {}).items()
		if t == tenant_id
	]
	requirements = [
		v for (t, _), v in getattr(svc, "_requirements", {}).items()
		if t == tenant_id
	]
	legacy_agents = [
		v for (t, _), v in getattr(svc, "_legacy_agents", {}).items()
		if t == tenant_id
	]
	legacy_sources = [
		v for (t, _), v in getattr(svc, "_legacy_sources", {}).items()
		if t == tenant_id
	]
	return {
		"title": "OSINT Operations Console",
		"tenant_id": tenant_id,
		"collection_plans": collection_plans,
		"requirements": requirements,
		"agents": legacy_agents,
		"sources": legacy_sources,
	}
