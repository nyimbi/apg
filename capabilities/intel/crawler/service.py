"""Domain service for APG intelligence crawler."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CRAWLER_AGENT_ROLES,
		SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_CRAWLER_AGENT_ROLES,
		SUPPORTED_CRAWLER_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


class IntelligenceCrawlerService:
	"""Tenant-scoped source, crawl, extraction, validation, and knowledge-prep coordinator."""

	def __init__(self) -> None:
		self._sources: dict[str, dict[str, Any]] = {}
		self._crawl_jobs: dict[str, dict[str, Any]] = {}
		self._extractions: dict[str, dict[str, Any]] = {}
		self._datasets: dict[str, dict[str, Any]] = {}
		self._validation_sessions: dict[str, dict[str, Any]] = {}
		self._rag_plans: dict[str, dict[str, Any]] = {}
		self._graph_projections: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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
		record = {
			"id": self._record_id("crawler_source", source_id),
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
			"updated_at": self._now(),
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
		record = {
			"id": self._record_id("crawler_job", job_id),
			"job_id": job_id,
			"tenant_id": tenant_id,
			"source_record_id": source["id"],
			"source_id": source["source_id"],
			"cadence": cadence,
			"max_depth": max_depth,
			"rate_limit_per_minute": rate_limit_per_minute,
			"high_risk": high_risk,
			"approved_by": approved_by,
			"status": "scheduled",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._crawl_jobs[record["id"]] = record
		self._emit("crawl_job_created", tenant_id, record["id"], {"source_id": source["source_id"], "cadence": cadence})
		return deepcopy(record)

	def complete_crawl_job(self, tenant_id: str, job_record_id: str, fetched_count: int, error_count: int = 0) -> dict[str, Any]:
		job = self._require_crawl_job(job_record_id, tenant_id)
		job["status"] = "completed" if error_count == 0 else "review_required"
		job["fetched_count"] = fetched_count
		job["error_count"] = error_count
		job["updated_at"] = self._now()
		self._emit("crawl_job_completed", tenant_id, job["id"], {"fetched_count": fetched_count, "error_count": error_count})
		return deepcopy(job)

	def record_extraction(self, extraction_id: str, tenant_id: str, job_record_id: str, schema_name: str, content: str, quality_score: float) -> dict[str, Any]:
		job = self._require_crawl_job(job_record_id, tenant_id)
		fingerprint = self._fingerprint(content) if content else ""
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_extraction",
			"schema_present": bool(schema_name),
			"fingerprint_present": bool(fingerprint),
			"quality_score": quality_score,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("crawler_extraction", extraction_id),
			"extraction_id": extraction_id,
			"tenant_id": tenant_id,
			"job_record_id": job["id"],
			"source_id": job["source_id"],
			"schema_name": schema_name,
			"content_fingerprint": fingerprint,
			"quality_score": quality_score,
			"status": "recorded",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._extractions[record["id"]] = record
		self._emit("extraction_recorded", tenant_id, record["id"], {"quality_score": quality_score})
		return deepcopy(record)

	def open_validation_session(self, session_id: str, tenant_id: str, extraction_record_id: str, reviewer: str) -> dict[str, Any]:
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
			"id": self._record_id("crawler_validation", session_id),
			"session_id": session_id,
			"tenant_id": tenant_id,
			"extraction_record_id": extraction["id"],
			"reviewer": reviewer,
			"confidence": None,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._validation_sessions[record["id"]] = record
		self._emit("validation_session_opened", tenant_id, record["id"], {"reviewer": reviewer})
		return deepcopy(record)

	def complete_validation_session(self, tenant_id: str, session_record_id: str, confidence: float, decision: str) -> dict[str, Any]:
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
		session["updated_at"] = self._now()
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
			"id": self._record_id("crawler_dataset", dataset_id),
			"dataset_id": dataset_id,
			"tenant_id": tenant_id,
			"extraction_record_id": extraction["id"],
			"source_id": extraction["source_id"],
			"contains_pii": contains_pii,
			"privacy_reviewed_by": privacy_reviewed_by,
			"status": "published",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._datasets[record["id"]] = record
		self._emit("dataset_published", tenant_id, record["id"], {"contains_pii": contains_pii})
		return deepcopy(record)

	def record_rag_plan(self, plan_id: str, tenant_id: str, dataset_record_id: str, chunk_plan: str, chunk_size: int, embedding_model: str) -> dict[str, Any]:
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
		record = {
			"id": self._record_id("crawler_rag_plan", plan_id),
			"plan_id": plan_id,
			"tenant_id": tenant_id,
			"dataset_record_id": dataset["id"],
			"chunk_plan": chunk_plan,
			"chunk_size": chunk_size,
			"embedding_model": embedding_model,
			"status": "ready",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._rag_plans[record["id"]] = record
		self._emit("rag_plan_recorded", tenant_id, record["id"], {"chunk_size": chunk_size})
		return deepcopy(record)

	def record_graph_projection(self, projection_id: str, tenant_id: str, dataset_record_id: str, entity_schema: str, relationship_evidence: str) -> dict[str, Any]:
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
			"id": self._record_id("crawler_graph", projection_id),
			"projection_id": projection_id,
			"tenant_id": tenant_id,
			"dataset_record_id": dataset["id"],
			"entity_schema": entity_schema,
			"relationship_evidence": relationship_evidence,
			"status": "ready",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._graph_projections[record["id"]] = record
		self._emit("graph_projection_recorded", tenant_id, record["id"], {"entity_schema": entity_schema})
		return deepcopy(record)

	def register_crawler_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
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
			"id": self._record_id("crawler_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("crawler_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_crawler_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
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

	def validate_batch_ingest(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
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
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

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

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

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

	def _require_record(self, records: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str, public_key: str) -> dict[str, Any]:
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
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _fingerprint(self, content: str) -> str:
		return hashlib.sha256(content.encode("utf-8")).hexdigest()

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


class CrawlerDatabaseService(IntelligenceCrawlerService):
	"""Compatibility wrapper for callers that still pass database settings."""

	def __init__(self, database_url: str | None = None, **engine_kwargs: Any) -> None:
		super().__init__()
		self.database_url = database_url
		self.engine_kwargs = engine_kwargs


CrawlerService = IntelligenceCrawlerService
