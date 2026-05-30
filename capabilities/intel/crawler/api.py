"""Dependency-light API helpers for APG intelligence crawler."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import IntelligenceCrawlerService
except ImportError:
	from capability_contract import get_capability_contract
	from service import IntelligenceCrawlerService


_SERVICE = IntelligenceCrawlerService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def register_source(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_source(
		payload["source_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["source_type"],
		payload["urls"],
		payload["allowed_domains"],
		payload.get("policy_reviewed_by"),
	)


def create_crawl_job(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_crawl_job(
		payload["job_id"],
		payload.get("tenant_id", "default"),
		payload["source_record_id"],
		payload["cadence"],
		payload["max_depth"],
		payload["rate_limit_per_minute"],
		payload.get("high_risk", False),
		payload.get("approved_by"),
	)


def record_extraction(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_extraction(
		payload["extraction_id"],
		payload.get("tenant_id", "default"),
		payload["job_record_id"],
		payload["schema_name"],
		payload["content"],
		payload["quality_score"],
	)


def publish_dataset(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_dataset(
		payload["dataset_id"],
		payload.get("tenant_id", "default"),
		payload["extraction_record_id"],
		payload["validation_recorded"],
		payload.get("contains_pii", False),
		payload.get("privacy_reviewed_by"),
	)


def register_crawler_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_crawler_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def service() -> IntelligenceCrawlerService:
	return _SERVICE
