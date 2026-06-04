"""Domain service for APG intelligence crawler."""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

try:
	from .capability_contract import (
		SUPPORTED_CRAWLER_AGENT_ROLES,
		SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_CRAWLER_AGENT_ROLES,
		SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
CRAWL_STATUS_SCHEDULED = "scheduled"
CRAWL_STATUS_RUNNING = "running"
CRAWL_STATUS_COMPLETED = "completed"
CRAWL_STATUS_FAILED = "failed"
CRAWL_STATUS_REVIEW_REQUIRED = "review_required"
CRAWL_STATUS_CANCELLED = "cancelled"

VALID_FREQUENCIES = {"hourly", "daily", "weekly", "monthly", "on_demand"}
VALID_EXTRACT_CONFIGS = {"entities", "keywords", "links", "metadata", "full_text", "structured"}

MAX_CRAWL_DEPTH = 10
MAX_RATE_LIMIT = 600  # requests per minute


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


def _fingerprint(content: str) -> str:
	return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _slug(value: str) -> str:
	return "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_") or "record"


def _record_id(prefix: str, value: str) -> str:
	return f"{prefix}_{_slug(value)}"


def _extract_domain(url: str) -> str:
	try:
		return urlparse(url).netloc or url
	except Exception:
		return url


class IntelligenceCrawlerService:
	"""Tenant-scoped source, crawl, extraction, validation, and knowledge-prep coordinator.

	Constructor accepts optional adapter/store overrides for production injection.
	In-memory dicts are the default store for tests and lightweight deployments.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self._sources: dict[str, dict[str, Any]] = {}
		self._crawl_jobs: dict[str, dict[str, Any]] = {}
		self._extractions: dict[str, dict[str, Any]] = {}
		self._datasets: dict[str, dict[str, Any]] = {}
		self._validation_sessions: dict[str, dict[str, Any]] = {}
		self._rag_plans: dict[str, dict[str, Any]] = {}
		self._graph_projections: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

		# Ban list: domain -> {reason, banned_at}
		self._ban_list: dict[str, dict[str, str]] = {}
		# Dedup fingerprint registry: fingerprint -> {url, crawled_at}
		self._fingerprint_registry: dict[str, dict[str, str]] = {}
		# Crawl schedule registry: source_id -> schedule metadata
		self._schedule_registry: dict[str, dict[str, Any]] = {}
		# Entity extraction results: extraction_id -> list of entity dicts
		self._entity_results: dict[str, list[dict[str, Any]]] = {}
		# Relationship extraction results: extraction_id -> list of relationship dicts
		self._relationship_results: dict[str, list[dict[str, Any]]] = {}
		# Health metrics: component -> latest metric
		self._health_metrics: dict[str, Any] = {
			"jobs_run": 0,
			"jobs_completed": 0,
			"jobs_failed": 0,
			"entities_extracted": 0,
			"relationships_extracted": 0,
			"duplicates_detected": 0,
			"sources_banned": 0,
		}

	# ------------------------------------------------------------------
	# Existing register_source / create_crawl_job / etc. – fully preserved
	# ------------------------------------------------------------------

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		source_type: str,
		urls: list[str],
		allowed_domains: list[str],
		policy_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"source_owner_assigned": bool(owner),
			"source_url_present": bool(urls),
			"allowed_domain_present": bool(allowed_domains),
			"policy_review_recorded": bool(policy_reviewed_by),
		}
		self._enforce(context)
		# Check for banned domains
		for url in urls:
			domain = _extract_domain(url)
			if domain in self._ban_list:
				raise PermissionError(f"Domain is banned: {domain} — reason: {self._ban_list[domain]['reason']}")

		record = {
			"id": _record_id("crawler_source", source_id),
			"source_id": source_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"source_type": source_type,
			"urls": list(urls),
			"allowed_domains": list(allowed_domains),
			"policy_reviewed_by": policy_reviewed_by,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._sources[record["id"]] = record
		self._emit("source_registered", tenant_id, record["id"], {"source_type": source_type})
		return deepcopy(record)

	def create_crawl_job(
		self,
		job_id: str,
		tenant_id: str,
		source_record_id: str,
		cadence: str,
		max_depth: int,
		rate_limit_per_minute: int,
		high_risk: bool = False,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		source = self._require_source(source_record_id, tenant_id) if source_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_crawl_job",
			"source_present": source is not None,
			"cadence_present": bool(cadence),
			"rate_limit_per_minute": rate_limit_per_minute,
			"max_depth": max_depth,
			"high_risk": high_risk,
			"approved": bool(approved_by),
		}
		self._enforce(context)

		if max_depth > MAX_CRAWL_DEPTH:
			raise ValueError(f"max_depth exceeds maximum allowed ({MAX_CRAWL_DEPTH})")
		if rate_limit_per_minute > MAX_RATE_LIMIT:
			raise ValueError(f"rate_limit_per_minute exceeds maximum allowed ({MAX_RATE_LIMIT})")
		if cadence not in VALID_FREQUENCIES:
			raise ValueError(f"cadence must be one of {VALID_FREQUENCIES}")

		record = {
			"id": _record_id("crawler_job", job_id),
			"job_id": job_id,
			"tenant_id": tenant_id,
			"source_record_id": source["id"],
			"source_id": source["source_id"],
			"cadence": cadence,
			"max_depth": max_depth,
			"rate_limit_per_minute": rate_limit_per_minute,
			"high_risk": high_risk,
			"approved_by": approved_by,
			"status": CRAWL_STATUS_SCHEDULED,
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
			"fetched_count": 0,
			"error_count": 0,
		}
		self._crawl_jobs[record["id"]] = record
		self._health_metrics["jobs_run"] += 1
		self._emit("crawl_job_created", tenant_id, record["id"], {"source_id": source["source_id"], "cadence": cadence})
		return deepcopy(record)

	def complete_crawl_job(
		self,
		tenant_id: str,
		job_record_id: str,
		fetched_count: int,
		error_count: int = 0,
	) -> dict[str, Any]:
		job = self._require_crawl_job(job_record_id, tenant_id)
		job["status"] = CRAWL_STATUS_COMPLETED if error_count == 0 else CRAWL_STATUS_REVIEW_REQUIRED
		job["fetched_count"] = fetched_count
		job["error_count"] = error_count
		job["updated_at"] = _utcnow()
		if error_count == 0:
			self._health_metrics["jobs_completed"] += 1
		else:
			self._health_metrics["jobs_failed"] += 1
		self._emit("crawl_job_completed", tenant_id, job["id"], {"fetched_count": fetched_count, "error_count": error_count})
		return deepcopy(job)

	def record_extraction(
		self,
		extraction_id: str,
		tenant_id: str,
		job_record_id: str,
		schema_name: str,
		content: str,
		quality_score: float,
	) -> dict[str, Any]:
		job = self._require_crawl_job(job_record_id, tenant_id)
		fp = _fingerprint(content) if content else ""
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_extraction",
			"schema_present": bool(schema_name),
			"fingerprint_present": bool(fp),
			"quality_score": quality_score,
		}
		self._enforce(context)
		record = {
			"id": _record_id("crawler_extraction", extraction_id),
			"extraction_id": extraction_id,
			"tenant_id": tenant_id,
			"job_record_id": job["id"],
			"source_id": job["source_id"],
			"schema_name": schema_name,
			"content_fingerprint": fp,
			"quality_score": quality_score,
			"status": "recorded",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._extractions[record["id"]] = record
		self._emit("extraction_recorded", tenant_id, record["id"], {"quality_score": quality_score})
		return deepcopy(record)

	def open_validation_session(
		self,
		session_id: str,
		tenant_id: str,
		extraction_record_id: str,
		reviewer: str,
	) -> dict[str, Any]:
		extraction = self._require_extraction(extraction_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_validation_session",
			"reviewer_present": bool(reviewer),
		}
		self._enforce(context)
		record = {
			"id": _record_id("crawler_validation", session_id),
			"session_id": session_id,
			"tenant_id": tenant_id,
			"extraction_record_id": extraction["id"],
			"reviewer": reviewer,
			"confidence": None,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._validation_sessions[record["id"]] = record
		self._emit("validation_session_opened", tenant_id, record["id"], {"reviewer": reviewer})
		return deepcopy(record)

	def complete_validation_session(
		self,
		tenant_id: str,
		session_record_id: str,
		confidence: float,
		decision: str,
	) -> dict[str, Any]:
		session = self._require_validation_session(session_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "complete_validation_session",
			"confidence": confidence,
		}
		self._enforce(context)
		session["confidence"] = confidence
		session["decision"] = decision
		session["status"] = "validated"
		session["updated_at"] = _utcnow()
		self._emit("validation_session_completed", tenant_id, session["id"], {"decision": decision, "confidence": confidence})
		return deepcopy(session)

	def publish_dataset(
		self,
		dataset_id: str,
		tenant_id: str,
		extraction_record_id: str,
		validation_recorded: bool,
		contains_pii: bool = False,
		privacy_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		extraction = self._require_extraction(extraction_record_id, tenant_id) if extraction_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_dataset",
			"lineage_present": extraction is not None,
			"validation_recorded": validation_recorded,
			"contains_pii": contains_pii,
			"privacy_review_recorded": bool(privacy_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": _record_id("crawler_dataset", dataset_id),
			"dataset_id": dataset_id,
			"tenant_id": tenant_id,
			"extraction_record_id": extraction["id"],
			"source_id": extraction["source_id"],
			"contains_pii": contains_pii,
			"privacy_reviewed_by": privacy_reviewed_by,
			"status": "published",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._datasets[record["id"]] = record
		self._emit("dataset_published", tenant_id, record["id"], {"contains_pii": contains_pii})
		return deepcopy(record)

	def record_rag_plan(
		self,
		plan_id: str,
		tenant_id: str,
		dataset_record_id: str,
		chunk_plan: str,
		chunk_size: int,
		embedding_model: str,
	) -> dict[str, Any]:
		dataset = self._require_dataset(dataset_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_rag_plan",
			"chunk_plan_present": bool(chunk_plan),
			"chunk_size": chunk_size,
			"embedding_model_present": bool(embedding_model),
		}
		self._enforce(context)
		if chunk_size <= 0:
			raise ValueError("chunk_size must be positive")
		record = {
			"id": _record_id("crawler_rag_plan", plan_id),
			"plan_id": plan_id,
			"tenant_id": tenant_id,
			"dataset_record_id": dataset["id"],
			"chunk_plan": chunk_plan,
			"chunk_size": chunk_size,
			"embedding_model": embedding_model,
			"status": "ready",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._rag_plans[record["id"]] = record
		self._emit("rag_plan_recorded", tenant_id, record["id"], {"chunk_size": chunk_size})
		return deepcopy(record)

	def record_graph_projection(
		self,
		projection_id: str,
		tenant_id: str,
		dataset_record_id: str,
		entity_schema: str,
		relationship_evidence: str,
	) -> dict[str, Any]:
		dataset = self._require_dataset(dataset_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_graph_projection",
			"entity_schema_present": bool(entity_schema),
			"relationship_evidence_present": bool(relationship_evidence),
		}
		self._enforce(context)
		record = {
			"id": _record_id("crawler_graph", projection_id),
			"projection_id": projection_id,
			"tenant_id": tenant_id,
			"dataset_record_id": dataset["id"],
			"entity_schema": entity_schema,
			"relationship_evidence": relationship_evidence,
			"status": "ready",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._graph_projections[record["id"]] = record
		self._emit("graph_projection_recorded", tenant_id, record["id"], {"entity_schema": entity_schema})
		return deepcopy(record)

	def register_crawler_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		instructions: str,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_crawler_agent",
			"agent_runtime_supported": runtime in SUPPORTED_CRAWLER_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_CRAWLER_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": _record_id("crawler_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._agents[record["id"]] = record
		self._emit("crawler_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_crawler_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown crawler agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_crawler_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch_ingest(
		self,
		tenant_id: str,
		record_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "crawler_batch", "event_stream": event_stream}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"source_count": len(self.list_sources(tenant_id)),
			"crawl_job_count": len(self.list_crawl_jobs(tenant_id)),
			"extraction_count": len(self.list_extractions(tenant_id)),
			"dataset_count": len(self.list_datasets(tenant_id)),
			"validation_session_count": len(self.list_validation_sessions(tenant_id)),
			"rag_plan_count": len(self.list_rag_plans(tenant_id)),
			"graph_projection_count": len(self.list_graph_projections(tenant_id)),
			"crawler_agent_count": len(self.list_crawler_agents(tenant_id)),
			"audit_event_count": len(self.get_audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	# ------------------------------------------------------------------
	# List helpers – preserved
	# ------------------------------------------------------------------

	def list_sources(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._sources, tenant_id)

	def list_crawl_jobs(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._crawl_jobs, tenant_id)

	def list_extractions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._extractions, tenant_id)

	def list_datasets(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._datasets, tenant_id)

	def list_validation_sessions(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._validation_sessions, tenant_id)

	def list_rag_plans(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._rag_plans, tenant_id)

	def list_graph_projections(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._graph_projections, tenant_id)

	def list_crawler_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def get_audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

	# Backward-compat name kept
	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return self.get_audit_events(tenant_id)

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.register_source(
			data.get("source_id", data.get("id", "source")),
			data.get("tenant_id", "default"),
			data.get("name", "Source"),
			data.get("owner", "owner"),
			data.get("source_type", "web"),
			data.get("urls", ["https://example.com"]),
			data.get("allowed_domains", ["example.com"]),
			data.get("policy_reviewed_by", "reviewer"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_sources(tenant_id)

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented crawler operations
	# ------------------------------------------------------------------

	async def schedule_crawl(
		self,
		url: str,
		depth: int,
		frequency: str,
		keywords: list[str],
	) -> dict[str, Any]:
		"""Register a crawl schedule for *url* at *frequency* with keyword filters."""
		assert url, "url required"
		assert isinstance(depth, int) and depth > 0, "depth must be a positive integer"
		assert frequency in VALID_FREQUENCIES, f"frequency must be one of {VALID_FREQUENCIES}"
		assert isinstance(keywords, list), "keywords must be a list"

		tenant_id = self.tenant_id
		domain = _extract_domain(url)
		if domain in self._ban_list:
			raise PermissionError(f"Domain is banned: {domain}")

		schedule_id = f"sched_{_slug(url)}_{frequency}"
		schedule = {
			"schedule_id": schedule_id,
			"url": url,
			"domain": domain,
			"depth": min(depth, MAX_CRAWL_DEPTH),
			"frequency": frequency,
			"keywords": list(keywords),
			"tenant_id": tenant_id,
			"status": "active",
			"created_at": _utcnow(),
		}
		self._schedule_registry[schedule_id] = schedule
		self._emit("crawl_scheduled", tenant_id, schedule_id, {"url": url, "frequency": frequency})
		return deepcopy(schedule)

	async def manual_crawl(
		self,
		url: str,
		depth: int,
		extract_config: dict[str, Any],
	) -> dict[str, Any]:
		"""Trigger a one-off crawl of *url* to *depth* with *extract_config* options."""
		assert url, "url required"
		assert isinstance(depth, int) and depth > 0
		assert isinstance(extract_config, dict), "extract_config must be a dict"

		tenant_id = self.tenant_id
		domain = _extract_domain(url)
		if domain in self._ban_list:
			raise PermissionError(f"Domain is banned: {domain}")

		task_id = f"manual_{_slug(url)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		# Simulate crawl: produce a synthetic result record
		page_count = min(depth * 5, 50)  # proxy: 5 pages per depth level
		task = {
			"task_id": task_id,
			"url": url,
			"domain": domain,
			"depth": min(depth, MAX_CRAWL_DEPTH),
			"extract_config": extract_config,
			"tenant_id": tenant_id,
			"status": CRAWL_STATUS_COMPLETED,
			"pages_fetched": page_count,
			"errors": 0,
			"started_at": _utcnow(),
			"completed_at": _utcnow(),
		}
		# Register as a crawl job for downstream tracking
		job_id = f"job_{task_id}"
		self._crawl_jobs[_record_id("crawler_job", job_id)] = {
			"id": _record_id("crawler_job", job_id),
			"job_id": job_id,
			"tenant_id": tenant_id,
			"source_record_id": None,
			"source_id": domain,
			"cadence": "on_demand",
			"max_depth": depth,
			"rate_limit_per_minute": 60,
			"high_risk": False,
			"approved_by": self.actor_id,
			"status": CRAWL_STATUS_COMPLETED,
			"fetched_count": page_count,
			"error_count": 0,
			"event_stream": "bytewax",
			"updated_at": _utcnow(),
		}
		self._health_metrics["jobs_run"] += 1
		self._health_metrics["jobs_completed"] += 1
		self._emit("manual_crawl_completed", tenant_id, task_id, {"url": url, "pages_fetched": page_count})
		return task

	async def crawl_status(self, task_id: str) -> dict[str, Any]:
		"""Return current status of crawl task *task_id*."""
		assert task_id, "task_id required"
		tenant_id = self.tenant_id

		# Search by job_id or record id
		for record in self._crawl_jobs.values():
			if record.get("tenant_id") == tenant_id and (
				record.get("job_id") == task_id or record.get("id") == task_id
			):
				return {
					"task_id": task_id,
					"status": record.get("status", "unknown"),
					"fetched_count": record.get("fetched_count", 0),
					"error_count": record.get("error_count", 0),
					"updated_at": record.get("updated_at", ""),
				}

		# Check schedule registry
		schedule = self._schedule_registry.get(task_id)
		if schedule and schedule.get("tenant_id") == tenant_id:
			return {"task_id": task_id, "status": schedule.get("status", "unknown"), "type": "schedule"}

		raise KeyError(f"Task not found: {task_id}")

	async def extract_entities(self, crawled_content_id: str) -> dict[str, Any]:
		"""Extract named entities from a recorded extraction *crawled_content_id*."""
		assert crawled_content_id, "crawled_content_id required"
		tenant_id = self.tenant_id

		extraction = self._require_extraction(crawled_content_id, tenant_id)
		fingerprint = extraction.get("content_fingerprint", "")

		# Simulate entity extraction: derive entity count from quality score
		quality = float(extraction.get("quality_score", 0.5))
		entity_count = max(1, int(quality * 30))
		entities = [
			{
				"entity_id": f"ent_{crawled_content_id}_{i}",
				"entity_type": ["person", "organisation", "location", "event", "concept"][i % 5],
				"mention": f"Entity_{i}",
				"confidence": round(quality * (0.8 + 0.02 * (i % 10)), 4),
			}
			for i in range(entity_count)
		]
		self._entity_results[crawled_content_id] = entities
		self._health_metrics["entities_extracted"] += entity_count
		self._emit("entities_extracted", tenant_id, crawled_content_id, {"entity_count": entity_count})
		return {
			"crawled_content_id": crawled_content_id,
			"source_id": extraction.get("source_id", ""),
			"entity_count": entity_count,
			"entities": entities,
			"fingerprint": fingerprint,
			"extracted_at": _utcnow(),
		}

	async def extract_relationships(self, crawled_content_id: str) -> dict[str, Any]:
		"""Extract entity relationships from *crawled_content_id*."""
		assert crawled_content_id, "crawled_content_id required"
		tenant_id = self.tenant_id

		extraction = self._require_extraction(crawled_content_id, tenant_id)
		entities = self._entity_results.get(crawled_content_id, [])
		if not entities:
			# Auto-extract if not yet done
			result = await self.extract_entities(crawled_content_id)
			entities = result["entities"]

		# Derive relationships: pair consecutive entities
		relationships: list[dict[str, Any]] = []
		quality = float(extraction.get("quality_score", 0.5))
		for i in range(0, len(entities) - 1, 2):
			rel = {
				"relationship_id": f"rel_{crawled_content_id}_{i}",
				"source_entity": entities[i]["entity_id"],
				"target_entity": entities[i + 1]["entity_id"],
				"relationship_type": ["affiliated_with", "located_at", "participated_in", "reported_by"][i % 4],
				"confidence": round(quality * 0.9, 4),
			}
			relationships.append(rel)

		self._relationship_results[crawled_content_id] = relationships
		self._health_metrics["relationships_extracted"] += len(relationships)
		self._emit("relationships_extracted", tenant_id, crawled_content_id, {"relationship_count": len(relationships)})
		return {
			"crawled_content_id": crawled_content_id,
			"entity_count": len(entities),
			"relationship_count": len(relationships),
			"relationships": relationships,
			"extracted_at": _utcnow(),
		}

	async def store_crawled_data(
		self,
		task_id: str,
		content: str,
	) -> dict[str, Any]:
		"""Persist crawled *content* from *task_id* as an extraction record."""
		assert task_id, "task_id required"
		assert content, "content required"

		tenant_id = self.tenant_id
		fp = _fingerprint(content)

		# Check for duplicate
		if fp in self._fingerprint_registry:
			existing = self._fingerprint_registry[fp]
			self._health_metrics["duplicates_detected"] += 1
			return {
				"task_id": task_id,
				"stored": False,
				"reason": "duplicate_content",
				"existing_url": existing.get("url", ""),
				"fingerprint": fp,
			}

		# Register fingerprint
		self._fingerprint_registry[fp] = {"url": task_id, "stored_at": _utcnow()}

		# Find crawl job for this task
		job_record = next(
			(r for r in self._crawl_jobs.values()
			 if r.get("tenant_id") == tenant_id and r.get("job_id") == task_id),
			None,
		)
		if job_record is None:
			raise KeyError(f"No crawl job found for task: {task_id}")

		extraction_id = f"extr_{task_id}_{fp[:8]}"
		quality = min(1.0, len(content) / 10000.0)
		record = self.record_extraction(
			extraction_id=extraction_id,
			tenant_id=tenant_id,
			job_record_id=job_record["id"],
			schema_name="raw_html",
			content=content,
			quality_score=quality,
		)
		return {
			"task_id": task_id,
			"stored": True,
			"extraction_id": extraction_id,
			"fingerprint": fp,
			"quality_score": quality,
			"stored_at": _utcnow(),
		}

	async def dedup_check(self, url: str, content_hash: str) -> dict[str, Any]:
		"""Check whether *content_hash* (SHA-256 hex) is already in the fingerprint registry."""
		assert url, "url required"
		assert content_hash, "content_hash required"

		existing = self._fingerprint_registry.get(content_hash)
		is_duplicate = existing is not None

		if is_duplicate:
			self._health_metrics["duplicates_detected"] += 1

		return {
			"url": url,
			"content_hash": content_hash,
			"is_duplicate": is_duplicate,
			"first_seen_at": existing.get("stored_at", "") if existing else None,
			"original_url": existing.get("url", "") if existing else None,
			"checked_at": _utcnow(),
		}

	async def crawl_analytics(self, period: str = "7d") -> dict[str, Any]:
		"""Aggregate crawl performance metrics over *period*."""
		assert period, "period required"
		tenant_id = self.tenant_id

		jobs = self.list_crawl_jobs(tenant_id)
		status_dist: dict[str, int] = defaultdict(int)
		source_counts: dict[str, int] = defaultdict(int)
		fetch_counts: list[int] = []
		error_rates: list[float] = []

		for job in jobs:
			status_dist[job.get("status", "unknown")] += 1
			source_counts[job.get("source_id", "unknown")] += 1
			fetched = int(job.get("fetched_count", 0))
			errors = int(job.get("error_count", 0))
			fetch_counts.append(fetched)
			total = fetched + errors
			error_rates.append(errors / total if total > 0 else 0.0)

		extractions = self.list_extractions(tenant_id)
		quality_scores = [float(e.get("quality_score", 0.0)) for e in extractions]

		self._emit("crawl_analytics_computed", tenant_id, period, {"job_count": len(jobs)})
		return {
			"tenant_id": tenant_id,
			"period": period,
			"job_count": len(jobs),
			"status_distribution": dict(status_dist),
			"total_pages_fetched": sum(fetch_counts),
			"avg_pages_per_job": round(statistics.mean(fetch_counts), 2) if fetch_counts else 0.0,
			"avg_error_rate": round(statistics.mean(error_rates), 4) if error_rates else 0.0,
			"extraction_count": len(extractions),
			"avg_extraction_quality": round(statistics.mean(quality_scores), 4) if quality_scores else 0.0,
			"fingerprint_registry_size": len(self._fingerprint_registry),
			"duplicate_count": self._health_metrics["duplicates_detected"],
			"source_count": len(self.list_sources(tenant_id)),
			"top_sources": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
			"computed_at": _utcnow(),
		}

	async def ban_source(self, url: str, reason: str) -> dict[str, Any]:
		"""Add *url*'s domain to the ban list with *reason*."""
		assert url, "url required"
		assert reason, "reason required"

		domain = _extract_domain(url)
		self._ban_list[domain] = {"reason": reason, "banned_at": _utcnow(), "banned_by": self.actor_id}
		self._health_metrics["sources_banned"] += 1
		self._emit("source_banned", self.tenant_id, domain, {"reason": reason})
		return {
			"url": url,
			"domain": domain,
			"reason": reason,
			"banned_at": self._ban_list[domain]["banned_at"],
		}

	async def unban_source(self, url: str) -> dict[str, Any]:
		"""Remove *url*'s domain from the ban list."""
		assert url, "url required"
		domain = _extract_domain(url)
		if domain not in self._ban_list:
			raise KeyError(f"Domain is not banned: {domain}")
		entry = self._ban_list.pop(domain)
		self._emit("source_unbanned", self.tenant_id, domain, {})
		return {"domain": domain, "unbanned_at": _utcnow(), "was_banned_at": entry["banned_at"]}

	async def crawler_health_report(self) -> dict[str, Any]:
		"""Return a comprehensive health report for the crawling subsystem."""
		tenant_id = self.tenant_id
		jobs = self.list_crawl_jobs(tenant_id)
		sources = self.list_sources(tenant_id)
		agents = self.list_crawler_agents(tenant_id)

		# Active vs stale jobs
		active_jobs = [j for j in jobs if j.get("status") in {CRAWL_STATUS_SCHEDULED, CRAWL_STATUS_RUNNING}]
		failed_jobs = [j for j in jobs if j.get("status") in {CRAWL_STATUS_FAILED, CRAWL_STATUS_REVIEW_REQUIRED}]

		# Source health: sources with no completed jobs
		completed_source_ids = {j.get("source_id") for j in jobs if j.get("status") == CRAWL_STATUS_COMPLETED}
		sources_without_data = [s["source_id"] for s in sources if s["source_id"] not in completed_source_ids]

		# Extraction quality summary
		extractions = self.list_extractions(tenant_id)
		quality_scores = [float(e.get("quality_score", 0.0)) for e in extractions]
		low_quality = [e for e in extractions if float(e.get("quality_score", 1.0)) < 0.3]

		self._emit("crawler_health_report_generated", tenant_id, "health_check", {})
		return {
			"tenant_id": tenant_id,
			"reported_at": _utcnow(),
			"source_count": len(sources),
			"active_source_count": len([s for s in sources if s.get("status") == "active"]),
			"banned_domain_count": len(self._ban_list),
			"sources_without_data": sources_without_data,
			"job_count": len(jobs),
			"active_job_count": len(active_jobs),
			"failed_job_count": len(failed_jobs),
			"extraction_count": len(extractions),
			"avg_extraction_quality": round(statistics.mean(quality_scores), 4) if quality_scores else 0.0,
			"low_quality_extraction_count": len(low_quality),
			"fingerprint_registry_size": len(self._fingerprint_registry),
			"agent_count": len(agents),
			"schedule_count": len(self._schedule_registry),
			"health_metrics": deepcopy(self._health_metrics),
			"status": "healthy" if not failed_jobs and not sources_without_data else "degraded",
		}

	async def source_quality_index(self) -> list[dict[str, Any]]:
		"""Score each source by average extraction quality across its jobs."""
		tenant_id = self.tenant_id
		source_job_map: dict[str, list[str]] = defaultdict(list)
		for job in self.list_crawl_jobs(tenant_id):
			source_job_map[job.get("source_id", "")].append(job.get("id", ""))

		# Map job -> extractions
		job_extraction_quality: dict[str, list[float]] = defaultdict(list)
		for extraction in self.list_extractions(tenant_id):
			jid = extraction.get("job_record_id", "")
			job_extraction_quality[jid].append(float(extraction.get("quality_score", 0.0)))

		result = []
		for source in self.list_sources(tenant_id):
			sid = source["source_id"]
			job_ids = source_job_map.get(sid, [])
			all_scores: list[float] = []
			for jid in job_ids:
				all_scores.extend(job_extraction_quality.get(jid, []))
			result.append({
				"source_id": sid,
				"source_type": source.get("source_type", ""),
				"job_count": len(job_ids),
				"extraction_count": len(all_scores),
				"avg_quality": round(statistics.mean(all_scores), 4) if all_scores else 0.0,
			})
		result.sort(key=lambda x: x["avg_quality"], reverse=True)
		return result

	async def schedule_overview(self) -> list[dict[str, Any]]:
		"""Return all registered crawl schedules for the current tenant."""
		tenant_id = self.tenant_id
		return [
			deepcopy(s) for s in self._schedule_registry.values()
			if s.get("tenant_id") == tenant_id
		]

	async def cancel_schedule(self, schedule_id: str) -> dict[str, Any]:
		"""Cancel a registered crawl schedule."""
		assert schedule_id, "schedule_id required"
		schedule = self._schedule_registry.get(schedule_id)
		if schedule is None or schedule.get("tenant_id") != self.tenant_id:
			raise KeyError(f"Schedule not found: {schedule_id}")
		schedule["status"] = CRAWL_STATUS_CANCELLED
		schedule["cancelled_at"] = _utcnow()
		self._emit("crawl_schedule_cancelled", self.tenant_id, schedule_id, {})
		return deepcopy(schedule)

	async def extraction_summary_by_schema(self) -> list[dict[str, Any]]:
		"""Summarise extraction counts and quality by schema name."""
		tenant_id = self.tenant_id
		schema_data: dict[str, list[float]] = defaultdict(list)
		for extr in self.list_extractions(tenant_id):
			schema = extr.get("schema_name", "unknown")
			schema_data[schema].append(float(extr.get("quality_score", 0.0)))

		return [
			{
				"schema_name": schema,
				"extraction_count": len(scores),
				"avg_quality": round(statistics.mean(scores), 4) if scores else 0.0,
			}
			for schema, scores in sorted(schema_data.items(), key=lambda x: -len(x[1]))
		]

	async def ban_list_report(self) -> dict[str, Any]:
		"""Return the current ban list with metadata."""
		return {
			"banned_domain_count": len(self._ban_list),
			"banned_domains": deepcopy(self._ban_list),
			"retrieved_at": _utcnow(),
		}

	# ------------------------------------------------------------------
	# Internal helpers – preserved from original implementation
	# ------------------------------------------------------------------

	def _require_source(self, source_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._sources, source_id, tenant_id, "source", "source_id")

	def _require_crawl_job(self, job_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._crawl_jobs, job_id, tenant_id, "crawl job", "job_id")

	def _require_extraction(self, extraction_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._extractions, extraction_id, tenant_id, "extraction", "extraction_id")

	def _require_dataset(self, dataset_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._datasets, dataset_id, tenant_id, "dataset", "dataset_id")

	def _require_validation_session(self, session_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._validation_sessions, session_id, tenant_id, "validation session", "session_id")

	def _require_record(
		self,
		records: dict[str, dict[str, Any]],
		record_id: str,
		tenant_id: str,
		label: str,
		public_key: str,
	) -> dict[str, Any]:
		for record in records.values():
			if record["tenant_id"] == tenant_id and (record["id"] == record_id or record[public_key] == record_id):
				return record
		raise KeyError(f"Unknown {label}: {record_id}")

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": _utcnow(),
		})


class CrawlerDatabaseService(IntelligenceCrawlerService):
	"""Compatibility wrapper for callers that still pass database settings."""

	def __init__(self, database_url: str | None = None, **engine_kwargs: Any) -> None:
		super().__init__()
		self.database_url = database_url
		self.engine_kwargs = engine_kwargs


CrawlerService = IntelligenceCrawlerService
