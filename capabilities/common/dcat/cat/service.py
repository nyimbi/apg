"""Data Catalog service — dataset registry, lineage, metadata, glossary."""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "dcat_cat"
SUPPORTED_FORMATS = {"csv", "parquet", "avro", "json", "orc", "delta", "iceberg", "unknown"}
SUPPORTED_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted", "pii"}

# PII field-name patterns for auto-detection
_PII_PATTERNS: list[re.Pattern[str]] = [
	re.compile(r, re.IGNORECASE)
	for r in [
		r"\bemail\b", r"\bphone\b", r"\bssn\b", r"\bpassport\b", r"\bdob\b",
		r"\bbirthdate\b", r"\bdate_of_birth\b", r"\bip_address\b", r"\bip\b",
		r"\bfirst_name\b", r"\blast_name\b", r"\bfull_name\b", r"\baddress\b",
		r"\bcredit_card\b", r"\bcard_number\b", r"\bnational_id\b", r"\buser_id\b",
		r"\bsocial_security\b", r"\bpassword\b", r"\bsalary\b", r"\bgps\b",
		r"\blatitude\b", r"\blongitude\b",
	]
]

# Completeness check fields — datasets are scored against these
_COMPLETENESS_FIELDS = [
	"description", "schema", "tags", "owner", "classification",
	"location_uri", "format", "domain",
]


class DataCatalogService:
	"""Dataset registry with lineage graph, metadata tagging, glossary, and ownership tracking."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.datasets: dict[str, dict[str, Any]] = {}
		self.lineage_edges: dict[str, dict[str, Any]] = {}
		self.tags: dict[str, dict[str, Any]] = {}
		self.glossary_terms: dict[str, dict[str, Any]] = {}
		self.ownership_records: dict[str, dict[str, Any]] = {}
		self.schema_versions: dict[str, list[dict[str, Any]]] = {}
		self._audit_events: list[dict[str, Any]] = []
		# New state stores
		self._quality_scores: dict[str, list[dict[str, Any]]] = {}          # dataset_id -> list of score records
		self._access_log: list[dict[str, Any]] = []                          # raw access events
		self._term_column_links: list[dict[str, Any]] = []                   # glossary-term → column bindings
		self._data_contracts: dict[str, dict[str, Any]] = {}                 # contract_id -> contract
		self._deprecations: dict[str, dict[str, Any]] = {}                   # dataset_id -> deprecation record
		self._embeddings: dict[str, list[float]] = {}                        # dataset_id -> embedding vector

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _emit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, payload: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"entity_id": entity_id,
			"entity_type": entity_type,
			"payload": payload or {},
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health."""
		return {
			"service": "dcat_cat",
			"status": "healthy",
			"dataset_count": len(self.datasets),
			"lineage_edge_count": len(self.lineage_edges),
			"glossary_term_count": len(self.glossary_terms),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return capability contract."""
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"supported_formats": sorted(SUPPORTED_FORMATS),
			"supported_classifications": sorted(SUPPORTED_CLASSIFICATIONS),
			"features": ["dataset_registry", "lineage_graph", "metadata_tagging", "glossary", "atlas_api", "ownership_tracking"],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Return audit trail for tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Dataset CRUD ─────────────────────────────────────────────

	async def create_dataset(
		self,
		tenant_id: str,
		name: str,
		owner: str,
		source_system: str,
		description: str = "",
		schema: dict[str, Any] | None = None,
		tags: list[str] | None = None,
		location_uri: str = "",
		format: str = "unknown",
		classification: str = "internal",
		domain: str = "default",
	) -> dict[str, Any]:
		"""Register a new dataset in the catalog."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(owner, "owner")
		guard_non_empty_string(source_system, "source_system")
		# Normalize classification
		if classification not in SUPPORTED_CLASSIFICATIONS:
			raise ValueError(f"classification must be one of {sorted(SUPPORTED_CLASSIFICATIONS)}")
		record: dict[str, Any] = {
			"id": self._id("ds"),
			"tenant_id": tenant,
			"name": name,
			"description": description,
			"schema": schema or {},
			"tags": list(tags or []),
			"owner": owner,
			"source_system": source_system,
			"location_uri": location_uri,
			"format": format if format in SUPPORTED_FORMATS else "unknown",
			"classification": classification,
			"domain": domain,
			"status": "active",
			"created_at": self._now(),
			"updated_at": None,
		}
		self.datasets[record["id"]] = record
		self.schema_versions[record["id"]] = [{"version": 1, "schema": deepcopy(schema or {}), "recorded_at": self._now()}]
		self._emit(tenant, "dataset_created", record["id"], "dataset", {"name": name, "owner": owner})
		_log.info("dataset created: %s tenant=%s", record["id"], tenant)
		return deepcopy(record)

	async def get_dataset(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Fetch a dataset by ID."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		return deepcopy(record)

	async def list_datasets(
		self,
		tenant_id: str,
		owner: str | None = None,
		domain: str | None = None,
		classification: str | None = None,
		tags: list[str] | None = None,
		status: str | None = None,
		source_system: str | None = None,
	) -> list[dict[str, Any]]:
		"""List datasets with optional filters."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.datasets.values() if r["tenant_id"] == tenant]
		if owner:
			items = [r for r in items if r["owner"] == owner]
		if domain:
			items = [r for r in items if r["domain"] == domain]
		if classification:
			items = [r for r in items if r["classification"] == classification]
		if tags:
			items = [r for r in items if all(t in r["tags"] for t in tags)]
		if status:
			items = [r for r in items if r["status"] == status]
		if source_system:
			items = [r for r in items if r["source_system"] == source_system]
		return items

	async def update_dataset(self, tenant_id: str, dataset_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update dataset metadata fields."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		allowed = {"description", "tags", "owner", "location_uri", "format", "classification", "domain", "schema"}
		for key, value in kwargs.items():
			if key in allowed and value is not None:
				if key == "schema":
					# Track schema evolution
					versions = self.schema_versions.setdefault(dataset_id, [])
					next_ver = len(versions) + 1
					versions.append({"version": next_ver, "schema": deepcopy(value), "recorded_at": self._now()})
				record[key] = value
		record["updated_at"] = self._now()
		self._emit(tenant, "dataset_updated", dataset_id, "dataset", {"fields": list(kwargs.keys())})
		return deepcopy(record)

	async def delete_dataset(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Soft-delete a dataset (marks as deleted)."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		record["status"] = "deleted"
		record["deleted_at"] = self._now()
		self._emit(tenant, "dataset_deleted", dataset_id, "dataset")
		return deepcopy(record)

	async def search_datasets(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Full-text search across dataset names, descriptions, and tags."""
		tenant = self._tenant(tenant_id)
		q = query.lower()
		results = []
		for r in self.datasets.values():
			if r["tenant_id"] != tenant or r["status"] == "deleted":
				continue
			if (q in r["name"].lower() or q in r["description"].lower()
					or any(q in t.lower() for t in r["tags"])
					or q in r["source_system"].lower()):
				results.append(deepcopy(r))
		return results

	async def get_schema_history(self, tenant_id: str, dataset_id: str) -> list[dict[str, Any]]:
		"""Return schema evolution history for a dataset."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		return deepcopy(self.schema_versions.get(dataset_id, []))

	# ── Lineage graph ────────────────────────────────────────────

	async def add_lineage_edge(
		self,
		tenant_id: str,
		source_dataset_id: str,
		target_dataset_id: str,
		transformation: str = "",
		job_name: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a lineage relationship between two datasets."""
		tenant = self._tenant(tenant_id)
		# Validate both datasets exist for this tenant
		src = self.datasets.get(source_dataset_id)
		tgt = self.datasets.get(target_dataset_id)
		if not src or src["tenant_id"] != tenant:
			raise KeyError(f"source dataset not found: {source_dataset_id}")
		if not tgt or tgt["tenant_id"] != tenant:
			raise KeyError(f"target dataset not found: {target_dataset_id}")
		record: dict[str, Any] = {
			"id": self._id("edge"),
			"tenant_id": tenant,
			"source_dataset_id": source_dataset_id,
			"target_dataset_id": target_dataset_id,
			"transformation": transformation,
			"job_name": job_name,
			"metadata": metadata or {},
			"created_at": self._now(),
		}
		self.lineage_edges[record["id"]] = record
		self._emit(tenant, "lineage_edge_added", record["id"], "lineage_edge", {
			"source": source_dataset_id, "target": target_dataset_id
		})
		return deepcopy(record)

	async def get_lineage_upstream(self, tenant_id: str, dataset_id: str, depth: int = 5) -> dict[str, Any]:
		"""Walk lineage graph upstream from dataset_id."""
		tenant = self._tenant(tenant_id)
		visited: set[str] = set()
		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []

		async def _walk(did: str, remaining: int) -> None:
			if did in visited or remaining == 0:
				return
			visited.add(did)
			ds = self.datasets.get(did)
			if ds and ds["tenant_id"] == tenant:
				nodes.append({"id": did, "name": ds["name"], "source_system": ds["source_system"]})
			for edge in self.lineage_edges.values():
				if edge["tenant_id"] == tenant and edge["target_dataset_id"] == did:
					edges.append(deepcopy(edge))
					await _walk(edge["source_dataset_id"], remaining - 1)

		await _walk(dataset_id, depth)
		return {"dataset_id": dataset_id, "direction": "upstream", "nodes": nodes, "edges": edges, "depth": depth}

	async def get_lineage_downstream(self, tenant_id: str, dataset_id: str, depth: int = 5) -> dict[str, Any]:
		"""Walk lineage graph downstream from dataset_id."""
		tenant = self._tenant(tenant_id)
		visited: set[str] = set()
		nodes: list[dict[str, Any]] = []
		edges: list[dict[str, Any]] = []

		async def _walk(did: str, remaining: int) -> None:
			if did in visited or remaining == 0:
				return
			visited.add(did)
			ds = self.datasets.get(did)
			if ds and ds["tenant_id"] == tenant:
				nodes.append({"id": did, "name": ds["name"], "source_system": ds["source_system"]})
			for edge in self.lineage_edges.values():
				if edge["tenant_id"] == tenant and edge["source_dataset_id"] == did:
					edges.append(deepcopy(edge))
					await _walk(edge["target_dataset_id"], remaining - 1)

		await _walk(dataset_id, depth)
		return {"dataset_id": dataset_id, "direction": "downstream", "nodes": nodes, "edges": edges, "depth": depth}

	async def list_lineage_edges(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all lineage edges for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self.lineage_edges.values() if e["tenant_id"] == tenant]

	async def delete_lineage_edge(self, tenant_id: str, edge_id: str) -> dict[str, Any]:
		"""Remove a lineage edge."""
		tenant = self._tenant(tenant_id)
		edge = self.lineage_edges.get(edge_id)
		if not edge or edge["tenant_id"] != tenant:
			raise KeyError(f"lineage edge not found: {edge_id}")
		del self.lineage_edges[edge_id]
		self._emit(tenant, "lineage_edge_deleted", edge_id, "lineage_edge")
		return deepcopy(edge)

	# ── Tags ─────────────────────────────────────────────────────

	async def create_tag(self, tenant_id: str, name: str, color: str = "#6366f1", description: str = "") -> dict[str, Any]:
		"""Create a metadata tag."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		tag: dict[str, Any] = {
			"id": self._id("tag"),
			"tenant_id": tenant,
			"name": name,
			"color": color,
			"description": description,
			"created_at": self._now(),
		}
		self.tags[tag["id"]] = tag
		self._emit(tenant, "tag_created", tag["id"], "tag")
		return deepcopy(tag)

	async def list_tags(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all tags for a tenant."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(t) for t in self.tags.values() if t["tenant_id"] == tenant]

	async def tag_dataset(self, tenant_id: str, dataset_id: str, tag_name: str) -> dict[str, Any]:
		"""Apply a tag to a dataset."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		if tag_name not in record["tags"]:
			record["tags"].append(tag_name)
			record["updated_at"] = self._now()
		self._emit(tenant, "dataset_tagged", dataset_id, "dataset", {"tag": tag_name})
		return deepcopy(record)

	async def untag_dataset(self, tenant_id: str, dataset_id: str, tag_name: str) -> dict[str, Any]:
		"""Remove a tag from a dataset."""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		record["tags"] = [t for t in record["tags"] if t != tag_name]
		record["updated_at"] = self._now()
		self._emit(tenant, "dataset_untagged", dataset_id, "dataset", {"tag": tag_name})
		return deepcopy(record)

	# ── Glossary ─────────────────────────────────────────────────

	async def create_glossary_term(
		self,
		tenant_id: str,
		term: str,
		definition: str,
		domain: str = "general",
		synonyms: list[str] | None = None,
		related_terms: list[str] | None = None,
	) -> dict[str, Any]:
		"""Add a business glossary term."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(term, "term")
		guard_non_empty_string(definition, "definition")
		record: dict[str, Any] = {
			"id": self._id("glos"),
			"tenant_id": tenant,
			"term": term,
			"definition": definition,
			"domain": domain,
			"synonyms": list(synonyms or []),
			"related_terms": list(related_terms or []),
			"status": "approved",
			"created_at": self._now(),
		}
		self.glossary_terms[record["id"]] = record
		self._emit(tenant, "glossary_term_created", record["id"], "glossary_term", {"term": term})
		return deepcopy(record)

	async def get_glossary_term(self, tenant_id: str, term_id: str) -> dict[str, Any]:
		"""Fetch a glossary term by ID."""
		tenant = self._tenant(tenant_id)
		record = self.glossary_terms.get(term_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"glossary term not found: {term_id}")
		return deepcopy(record)

	async def list_glossary_terms(self, tenant_id: str, domain: str | None = None) -> list[dict[str, Any]]:
		"""List glossary terms, optionally filtered by domain."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.glossary_terms.values() if r["tenant_id"] == tenant]
		if domain:
			items = [r for r in items if r["domain"] == domain]
		return items

	async def update_glossary_term(self, tenant_id: str, term_id: str, **kwargs: Any) -> dict[str, Any]:
		"""Update a glossary term."""
		tenant = self._tenant(tenant_id)
		record = self.glossary_terms.get(term_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"glossary term not found: {term_id}")
		for key in ("definition", "domain", "synonyms", "related_terms", "status"):
			if key in kwargs and kwargs[key] is not None:
				record[key] = kwargs[key]
		self._emit(tenant, "glossary_term_updated", term_id, "glossary_term")
		return deepcopy(record)

	async def delete_glossary_term(self, tenant_id: str, term_id: str) -> dict[str, Any]:
		"""Delete a glossary term."""
		tenant = self._tenant(tenant_id)
		record = self.glossary_terms.get(term_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"glossary term not found: {term_id}")
		del self.glossary_terms[term_id]
		self._emit(tenant, "glossary_term_deleted", term_id, "glossary_term")
		return deepcopy(record)

	async def search_glossary(self, tenant_id: str, query: str) -> list[dict[str, Any]]:
		"""Search glossary by term text or definition."""
		tenant = self._tenant(tenant_id)
		q = query.lower()
		return [
			deepcopy(r) for r in self.glossary_terms.values()
			if r["tenant_id"] == tenant and (
				q in r["term"].lower() or q in r["definition"].lower()
				or any(q in s.lower() for s in r["synonyms"])
			)
		]

	# ── Ownership tracking ────────────────────────────────────────

	async def assign_owner(
		self,
		tenant_id: str,
		dataset_id: str,
		owner: str,
		ownership_type: str = "technical",
	) -> dict[str, Any]:
		"""Assign or change dataset ownership."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(owner, "owner")
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		previous_owner = record.get("owner")
		record["owner"] = owner
		record["updated_at"] = self._now()
		ownership_record: dict[str, Any] = {
			"id": self._id("own"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"owner": owner,
			"ownership_type": ownership_type,
			"previous_owner": previous_owner,
			"assigned_at": self._now(),
		}
		self.ownership_records[ownership_record["id"]] = ownership_record
		self._emit(tenant, "ownership_assigned", dataset_id, "dataset", {
			"owner": owner, "previous_owner": previous_owner
		})
		return deepcopy(ownership_record)

	async def list_ownership_history(self, tenant_id: str, dataset_id: str) -> list[dict[str, Any]]:
		"""Return ownership history for a dataset."""
		tenant = self._tenant(tenant_id)
		return [
			deepcopy(r) for r in self.ownership_records.values()
			if r["tenant_id"] == tenant and r["dataset_id"] == dataset_id
		]

	# ── Apache Atlas-compatible API ───────────────────────────────

	async def atlas_get_entity(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Return Atlas-compatible entity representation."""
		tenant = self._tenant(tenant_id)
		record = await self.get_dataset(tenant_id, dataset_id)
		return {
			"entity": {
				"typeName": "hive_table",
				"guid": record["id"],
				"status": record["status"].upper(),
				"displayText": record["name"],
				"attributes": {
					"name": record["name"],
					"description": record["description"],
					"owner": record["owner"],
					"qualifiedName": f"{record['source_system']}.{record['name']}",
					"createTime": record["created_at"],
					"updateTime": record.get("updated_at"),
				},
				"classifications": [{"typeName": t} for t in record["tags"]],
			}
		}

	async def atlas_search(self, tenant_id: str, query: str, type_name: str = "hive_table") -> dict[str, Any]:
		"""Atlas-compatible basic search."""
		tenant = self._tenant(tenant_id)
		datasets = await self.search_datasets(tenant_id, query)
		entities = [
			{
				"guid": r["id"],
				"typeName": type_name,
				"displayText": r["name"],
				"status": r["status"].upper(),
			}
			for r in datasets
		]
		return {"queryType": "DSL", "queryText": query, "entities": entities, "count": len(entities)}

	async def atlas_create_lineage(self, tenant_id: str, process_qualified_name: str, inputs: list[str], outputs: list[str]) -> dict[str, Any]:
		"""Create lineage relationships in Atlas-compatible format."""
		tenant = self._tenant(tenant_id)
		edges_created = []
		errors = []
		for inp in inputs:
			for out in outputs:
				try:
					edge = await self.add_lineage_edge(
						tenant_id=tenant_id,
						source_dataset_id=inp,
						target_dataset_id=out,
						transformation=process_qualified_name,
						job_name=process_qualified_name,
					)
					edges_created.append(edge)
				except Exception as exc:
					_log.error("atlas lineage edge failed %s->%s: %s", inp, out, exc)
					errors.append({"source": inp, "target": out, "error": str(exc)})
		return {
			"process": process_qualified_name,
			"edges_created": len(edges_created),
			"errors": errors,
			"edges": edges_created,
		}

	# ── Statistics / dashboard ────────────────────────────────────

	async def catalog_statistics(self, tenant_id: str) -> dict[str, Any]:
		"""Return catalog-wide statistics for a tenant."""
		tenant = self._tenant(tenant_id)
		datasets = [r for r in self.datasets.values() if r["tenant_id"] == tenant]
		active = [r for r in datasets if r["status"] == "active"]
		by_domain: dict[str, int] = {}
		by_classification: dict[str, int] = {}
		by_source: dict[str, int] = {}
		for r in active:
			by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
			by_classification[r["classification"]] = by_classification.get(r["classification"], 0) + 1
			by_source[r["source_system"]] = by_source.get(r["source_system"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_datasets": len(datasets),
			"active_datasets": len(active),
			"lineage_edges": len([e for e in self.lineage_edges.values() if e["tenant_id"] == tenant]),
			"glossary_terms": len([t for t in self.glossary_terms.values() if t["tenant_id"] == tenant]),
			"tags": len([t for t in self.tags.values() if t["tenant_id"] == tenant]),
			"datasets_by_domain": by_domain,
			"datasets_by_classification": by_classification,
			"datasets_by_source_system": by_source,
			"generated_at": self._now(),
		}

	async def bulk_register_datasets(self, tenant_id: str, datasets: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-register multiple datasets."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		tasks = [
			self.create_dataset(
				tenant_id=tenant_id,
				name=d.get("name", ""),
				owner=d.get("owner", ""),
				source_system=d.get("source_system", ""),
				description=d.get("description", ""),
				schema=d.get("schema"),
				tags=d.get("tags"),
				location_uri=d.get("location_uri", ""),
				format=d.get("format", "unknown"),
				classification=d.get("classification", "internal"),
				domain=d.get("domain", "default"),
			)
			for d in datasets
		]
		for coro, d in zip(tasks, datasets):
			try:
				rec = await coro
				results.append(rec)
			except Exception as exc:
				_log.error("bulk_register_datasets failed for %s: %s", d.get("name"), exc)
				errors.append({"input": d, "error": str(exc)})
		return {"processed": len(results), "failed": len(errors), "datasets": results, "errors": errors}

	async def export_catalog(self, tenant_id: str, format: str = "json") -> dict[str, Any]:
		"""Export entire catalog metadata for a tenant."""
		tenant = self._tenant(tenant_id)
		datasets = await self.list_datasets(tenant_id)
		glossary = await self.list_glossary_terms(tenant_id)
		edges = await self.list_lineage_edges(tenant_id)
		return {
			"tenant_id": tenant,
			"format": format,
			"datasets": datasets,
			"glossary_terms": glossary,
			"lineage_edges": edges,
			"total_datasets": len(datasets),
			"exported_at": self._now(),
		}

	async def get_impact_analysis(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Identify all downstream datasets impacted by changes to this one."""
		tenant = self._tenant(tenant_id)
		downstream = await self.get_lineage_downstream(tenant_id, dataset_id, depth=10)
		impacted = [n for n in downstream["nodes"] if n["id"] != dataset_id]
		return {
			"dataset_id": dataset_id,
			"impacted_count": len(impacted),
			"impacted_datasets": impacted,
			"lineage_depth": downstream["depth"],
			"generated_at": self._now(),
		}

	# ── Data Quality ─────────────────────────────────────────────────

	async def record_quality_score(
		self,
		tenant_id: str,
		dataset_id: str,
		dimension: str,
		score: float,
		job_id: str = "",
		details: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Record a quality metric for a dataset dimension (completeness, freshness, validity…).

		Args:
			tenant_id:  Tenant namespace.
			dataset_id: Target dataset.
			dimension:  One of completeness | freshness | validity | uniqueness | accuracy.
			score:      Float in [0.0, 1.0].
			job_id:     Optional job/run that produced this score.
			details:    Arbitrary dict with per-rule breakdown.

		Returns:
			The persisted quality score record.
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		if not (0.0 <= score <= 1.0):
			raise ValueError("score must be in [0.0, 1.0]")
		entry: dict[str, Any] = {
			"id": self._id("qsc"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"dimension": dimension,
			"score": score,
			"job_id": job_id,
			"details": details or {},
			"measured_at": self._now(),
		}
		self._quality_scores.setdefault(dataset_id, []).append(entry)
		self._emit(tenant, "quality_score_recorded", dataset_id, "dataset", {
			"dimension": dimension, "score": score
		})
		_log.info("quality score recorded: dataset=%s dimension=%s score=%.3f", dataset_id, dimension, score)
		return deepcopy(entry)

	async def get_quality_profile(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Return the latest quality score per dimension for a dataset, with an aggregate.

		Computes the arithmetic mean of the most-recent score across all measured
		dimensions as the ``trust_score``.

		Returns:
			Dict with keys ``dataset_id``, ``trust_score``, ``dimensions``, ``measured_count``.
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		entries = self._quality_scores.get(dataset_id, [])
		# Latest score per dimension
		latest: dict[str, dict[str, Any]] = {}
		for e in entries:
			dim = e["dimension"]
			if dim not in latest or e["measured_at"] > latest[dim]["measured_at"]:
				latest[dim] = e
		scores = [v["score"] for v in latest.values()]
		trust_score = round(sum(scores) / len(scores), 4) if scores else None
		return {
			"dataset_id": dataset_id,
			"trust_score": trust_score,
			"dimensions": {dim: deepcopy(entry) for dim, entry in latest.items()},
			"measured_count": len(entries),
			"generated_at": self._now(),
		}

	# ── PII Detection ────────────────────────────────────────────────

	async def scan_pii_fields(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Scan schema column names for PII-indicative patterns.

		If PII columns are detected the dataset classification is promoted to ``pii``
		(never demoted). Returns a scan result with flagged fields and confidence.

		Returns:
			``{dataset_id, pii_detected, flagged_fields, classification_upgraded, ...}``
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		schema = record.get("schema") or {}
		columns: list[str] = list(schema.keys()) if isinstance(schema, dict) else []
		flagged: list[dict[str, Any]] = []
		for col in columns:
			matched = [p.pattern for p in _PII_PATTERNS if p.search(col)]
			if matched:
				flagged.append({"column": col, "matched_patterns": matched, "confidence": 0.9})
		upgraded = False
		if flagged and record["classification"] != "pii":
			record["classification"] = "pii"
			record["updated_at"] = self._now()
			upgraded = True
			self._emit(tenant, "classification_upgraded_to_pii", dataset_id, "dataset", {
				"flagged_fields": [f["column"] for f in flagged]
			})
		result: dict[str, Any] = {
			"dataset_id": dataset_id,
			"pii_detected": bool(flagged),
			"flagged_fields": flagged,
			"classification_upgraded": upgraded,
			"scanned_columns": len(columns),
			"scanned_at": self._now(),
		}
		_log.info("pii scan: dataset=%s flagged=%d upgraded=%s", dataset_id, len(flagged), upgraded)
		return result

	# ── Popularity / Usage ───────────────────────────────────────────

	async def record_dataset_access(
		self,
		tenant_id: str,
		dataset_id: str,
		accessor: str,
		access_type: str = "read",
	) -> dict[str, Any]:
		"""Log a dataset access event for popularity tracking.

		Args:
			tenant_id:   Tenant namespace.
			dataset_id:  Accessed dataset.
			accessor:    User or service that accessed the dataset.
			access_type: One of ``read`` | ``query`` | ``export`` | ``preview``.

		Returns:
			The persisted access event.
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		guard_non_empty_string(accessor, "accessor")
		event: dict[str, Any] = {
			"id": self._id("acc"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"accessor": accessor,
			"access_type": access_type,
			"accessed_at": self._now(),
		}
		self._access_log.append(event)
		return deepcopy(event)

	async def get_popular_datasets(
		self,
		tenant_id: str,
		limit: int = 10,
		since_days: int = 30,
	) -> list[dict[str, Any]]:
		"""Return datasets ranked by access frequency within a trailing window.

		Args:
			tenant_id:  Tenant namespace.
			limit:      Maximum results to return.
			since_days: Trailing window in calendar days.

		Returns:
			List of ``{dataset_id, name, access_count, unique_accessors}`` sorted descending.
		"""
		tenant = self._tenant(tenant_id)
		from datetime import timedelta
		cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat(timespec="seconds")
		counts: dict[str, int] = defaultdict(int)
		accessors: dict[str, set[str]] = defaultdict(set)
		for event in self._access_log:
			if event["tenant_id"] == tenant and event["accessed_at"] >= cutoff:
				did = event["dataset_id"]
				counts[did] += 1
				accessors[did].add(event["accessor"])
		ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
		result = []
		for did, count in ranked:
			ds = self.datasets.get(did)
			result.append({
				"dataset_id": did,
				"name": ds["name"] if ds else did,
				"access_count": count,
				"unique_accessors": len(accessors[did]),
			})
		return result

	# ── Catalog Completeness Scoring ─────────────────────────────────

	async def score_dataset_completeness(self, tenant_id: str, dataset_id: str) -> dict[str, Any]:
		"""Compute metadata completeness score for a single dataset.

		Checks presence and non-emptiness of description, schema, tags, owner,
		classification, location_uri, format, and domain. Also checks whether the
		dataset has at least one lineage edge and one quality score.

		Returns:
			``{dataset_id, score, missing, present, has_lineage, has_quality}``
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		present: list[str] = []
		missing: list[str] = []
		for field in _COMPLETENESS_FIELDS:
			val = record.get(field)
			if val and val not in ("unknown", "default", "internal", ""):
				present.append(field)
			elif field == "tags" and isinstance(val, list) and val:
				present.append(field)
			elif field == "schema" and isinstance(val, dict) and val:
				present.append(field)
			else:
				missing.append(field)
		has_lineage = any(
			e["tenant_id"] == tenant and (
				e["source_dataset_id"] == dataset_id or e["target_dataset_id"] == dataset_id
			)
			for e in self.lineage_edges.values()
		)
		has_quality = bool(self._quality_scores.get(dataset_id))
		bonus = (1 if has_lineage else 0) + (1 if has_quality else 0)
		total_checks = len(_COMPLETENESS_FIELDS) + 2
		score = round((len(present) + bonus) / total_checks, 4)
		return {
			"dataset_id": dataset_id,
			"score": score,
			"present": present,
			"missing": missing,
			"has_lineage": has_lineage,
			"has_quality": has_quality,
			"generated_at": self._now(),
		}

	async def get_governance_health(self, tenant_id: str) -> dict[str, Any]:
		"""Compute aggregate governance health across all active datasets for a tenant.

		Fans out completeness scoring across all active datasets concurrently and
		returns per-domain averages plus an overall ``health_score``.

		Returns:
			``{health_score, total_datasets, avg_by_domain, low_quality_datasets, ...}``
		"""
		tenant = self._tenant(tenant_id)
		active_ids = [
			did for did, ds in self.datasets.items()
			if ds["tenant_id"] == tenant and ds["status"] == "active"
		]
		if not active_ids:
			return {"health_score": None, "total_datasets": 0, "generated_at": self._now()}
		profiles = await asyncio.gather(*[
			self.score_dataset_completeness(tenant_id, did, return_exceptions=True) for did in active_ids
		])
		scores_by_domain: dict[str, list[float]] = defaultdict(list)
		low_quality: list[dict[str, Any]] = []
		all_scores: list[float] = []
		for profile in profiles:
			did = profile["dataset_id"]
			ds = self.datasets[did]
			domain = ds["domain"]
			sc = profile["score"]
			all_scores.append(sc)
			scores_by_domain[domain].append(sc)
			if sc < 0.5:
				low_quality.append({"dataset_id": did, "name": ds["name"], "score": sc})
		health_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
		avg_by_domain = {
			d: round(sum(vs) / len(vs), 4) for d, vs in scores_by_domain.items()
		}
		return {
			"health_score": health_score,
			"total_datasets": len(active_ids),
			"avg_by_domain": avg_by_domain,
			"low_quality_datasets": sorted(low_quality, key=lambda x: x["score"]),
			"generated_at": self._now(),
		}

	# ── Schema Diff ──────────────────────────────────────────────────

	async def compute_schema_diff(
		self,
		tenant_id: str,
		dataset_id: str,
		from_version: int,
		to_version: int,
	) -> dict[str, Any]:
		"""Compute a structured diff between two schema versions.

		Classifies each change as:

		- ``COMPATIBLE``  — new nullable column added
		- ``WARNING``     — column type changed
		- ``BREAKING``    — column removed or renamed

		Emits a ``schema_breaking_change`` audit event when any BREAKING changes exist.

		Returns:
			``{dataset_id, from_version, to_version, compatibility, changes}``
		"""
		tenant = self._tenant(tenant_id)
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		versions = self.schema_versions.get(dataset_id, [])
		version_map = {v["version"]: v["schema"] for v in versions}
		if from_version not in version_map:
			raise KeyError(f"schema version {from_version} not found for dataset {dataset_id}")
		if to_version not in version_map:
			raise KeyError(f"schema version {to_version} not found for dataset {dataset_id}")
		old_schema: dict[str, Any] = version_map[from_version]
		new_schema: dict[str, Any] = version_map[to_version]
		old_fields = set(old_schema.keys())
		new_fields = set(new_schema.keys())
		changes: list[dict[str, Any]] = []
		# Removed fields → BREAKING
		for col in old_fields - new_fields:
			changes.append({"column": col, "change": "removed", "severity": "BREAKING"})
		# Added fields → COMPATIBLE
		for col in new_fields - old_fields:
			changes.append({"column": col, "change": "added", "severity": "COMPATIBLE"})
		# Type changes → WARNING
		for col in old_fields & new_fields:
			old_type = old_schema[col]
			new_type = new_schema[col]
			if old_type != new_type:
				changes.append({"column": col, "change": "type_changed",
				                 "from": old_type, "to": new_type, "severity": "WARNING"})
		severity_rank = {"BREAKING": 2, "WARNING": 1, "COMPATIBLE": 0}
		overall = "COMPATIBLE"
		for c in changes:
			if severity_rank[c["severity"]] > severity_rank[overall]:
				overall = c["severity"]
		if overall == "BREAKING":
			self._emit(tenant, "schema_breaking_change", dataset_id, "dataset", {
				"from_version": from_version, "to_version": to_version,
				"breaking_columns": [c["column"] for c in changes if c["severity"] == "BREAKING"],
			})
		return {
			"dataset_id": dataset_id,
			"from_version": from_version,
			"to_version": to_version,
			"compatibility": overall,
			"changes": changes,
			"change_count": len(changes),
			"computed_at": self._now(),
		}

	# ── Dataset Deprecation ──────────────────────────────────────────

	async def deprecate_dataset(
		self,
		tenant_id: str,
		dataset_id: str,
		reason: str,
		successor_id: str | None = None,
		deprecation_date: str | None = None,
	) -> dict[str, Any]:
		"""Initiate a deprecation workflow for a dataset.

		Sets dataset status to ``deprecated``, records successor pointer and reason,
		and emits a ``dataset_deprecated`` event for downstream alert routing.

		Args:
			reason:          Human-readable deprecation rationale.
			successor_id:    ID of the recommended replacement dataset.
			deprecation_date: ISO date when the dataset will be removed (YYYY-MM-DD).

		Returns:
			The deprecation record.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(reason, "reason")
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		record["status"] = "deprecated"
		record["updated_at"] = self._now()
		deprecation: dict[str, Any] = {
			"id": self._id("dep"),
			"tenant_id": tenant,
			"dataset_id": dataset_id,
			"dataset_name": record["name"],
			"reason": reason,
			"successor_id": successor_id,
			"deprecation_date": deprecation_date,
			"created_at": self._now(),
		}
		self._deprecations[dataset_id] = deprecation
		self._emit(tenant, "dataset_deprecated", dataset_id, "dataset", {
			"reason": reason, "successor_id": successor_id
		})
		_log.info("dataset deprecated: %s tenant=%s successor=%s", dataset_id, tenant, successor_id)
		return deepcopy(deprecation)

	async def list_deprecated_datasets(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all deprecation records for a tenant, including days until removal.

		Returns:
			List of deprecation records sorted by ``deprecation_date`` ascending.
			Each record includes ``days_until_removal`` (None if no date set).
		"""
		tenant = self._tenant(tenant_id)
		result = []
		today_str = datetime.now(timezone.utc).date().isoformat()
		for dep in self._deprecations.values():
			if dep["tenant_id"] != tenant:
				continue
			entry = deepcopy(dep)
			if dep["deprecation_date"]:
				try:
					delta = (
						datetime.fromisoformat(dep["deprecation_date"]).date()
						- datetime.fromisoformat(today_str).date()
					)
					entry["days_until_removal"] = delta.days
				except ValueError:
					entry["days_until_removal"] = None
			else:
				entry["days_until_removal"] = None
			result.append(entry)
		return sorted(result, key=lambda x: x.get("deprecation_date") or "9999-12-31")

	# ── Catalog Facets ───────────────────────────────────────────────

	async def get_catalog_facets(
		self,
		tenant_id: str,
		active_filters: dict[str, str] | None = None,
	) -> dict[str, Any]:
		"""Return per-facet counts for the discovery sidebar in a single pass.

		Applies ``active_filters`` (e.g. ``{domain: "payments"}``) before computing
		counts so the UI can render dependent facet narrowing correctly.

		Facets returned: ``domain``, ``classification``, ``format``, ``source_system``,
		``status``, ``owner``.

		Returns:
			``{facets: {domain: {payments: 12, ...}, ...}, total_matching: int}``
		"""
		tenant = self._tenant(tenant_id)
		af = active_filters or {}
		facets: dict[str, dict[str, int]] = {
			"domain": {}, "classification": {}, "format": {},
			"source_system": {}, "status": {}, "owner": {},
		}
		total = 0
		for ds in self.datasets.values():
			if ds["tenant_id"] != tenant:
				continue
			# Apply active filters
			if af.get("domain") and ds["domain"] != af["domain"]:
				continue
			if af.get("classification") and ds["classification"] != af["classification"]:
				continue
			if af.get("format") and ds["format"] != af["format"]:
				continue
			if af.get("source_system") and ds["source_system"] != af["source_system"]:
				continue
			if af.get("status") and ds["status"] != af["status"]:
				continue
			total += 1
			for facet_key in facets:
				val = ds.get(facet_key, "unknown") or "unknown"
				facets[facet_key][val] = facets[facet_key].get(val, 0) + 1
		return {"facets": facets, "total_matching": total, "generated_at": self._now()}

	# ── Glossary-Column Linkage ──────────────────────────────────────

	async def link_term_to_column(
		self,
		tenant_id: str,
		term_id: str,
		dataset_id: str,
		column_name: str,
	) -> dict[str, Any]:
		"""Bind a glossary term to a specific column in a dataset.

		Enables column-level semantic search: "find all columns representing
		``net_revenue``" returns exactly the column, not the dataset.

		Returns:
			The created linkage record.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(column_name, "column_name")
		term = self.glossary_terms.get(term_id)
		if not term or term["tenant_id"] != tenant:
			raise KeyError(f"glossary term not found: {term_id}")
		record = self.datasets.get(dataset_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"dataset not found: {dataset_id}")
		link: dict[str, Any] = {
			"id": self._id("tcl"),
			"tenant_id": tenant,
			"term_id": term_id,
			"term_name": term["term"],
			"dataset_id": dataset_id,
			"dataset_name": record["name"],
			"column_name": column_name,
			"created_at": self._now(),
		}
		self._term_column_links.append(link)
		self._emit(tenant, "term_linked_to_column", term_id, "glossary_term", {
			"dataset_id": dataset_id, "column_name": column_name
		})
		return deepcopy(link)

	async def find_columns_by_term(self, tenant_id: str, term_id: str) -> list[dict[str, Any]]:
		"""Return all dataset columns linked to a glossary term.

		Returns:
			List of ``{dataset_id, dataset_name, column_name}`` records.
		"""
		tenant = self._tenant(tenant_id)
		term = self.glossary_terms.get(term_id)
		if not term or term["tenant_id"] != tenant:
			raise KeyError(f"glossary term not found: {term_id}")
		return [
			deepcopy(lnk) for lnk in self._term_column_links
			if lnk["tenant_id"] == tenant and lnk["term_id"] == term_id
		]

	# ── Federated Search ─────────────────────────────────────────────

	async def federate_search(
		self,
		root_tenant_id: str,
		query: str,
		child_tenant_ids: list[str],
	) -> dict[str, Any]:
		"""Fan out search across multiple tenant namespaces concurrently.

		Useful for data mesh topologies where each domain runs an isolated
		tenant but a root governance tenant needs cross-domain discovery.

		Returns:
			``{query, results: [{source_tenant, ...dataset}], total}``
		"""
		guard_non_empty_string(query, "query")
		all_tenants = [root_tenant_id] + (child_tenant_ids or [])
		tenant_results = await asyncio.gather(*[
			self.search_datasets(tid, query, return_exceptions=True) for tid in all_tenants
		], return_exceptions=True)
		merged: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for tid, res in zip(all_tenants, tenant_results):
			if isinstance(res, Exception):
				errors.append({"tenant_id": tid, "error": str(res)})
			else:
				for ds in res:
					annotated = deepcopy(ds)
					annotated["source_tenant"] = tid
					merged.append(annotated)
		return {
			"query": query,
			"results": merged,
			"total": len(merged),
			"tenants_searched": all_tenants,
			"errors": errors,
			"generated_at": self._now(),
		}

	# ── OpenMetadata / DCAT-AP Export ────────────────────────────────

	async def export_dcat_ap(self, tenant_id: str) -> dict[str, Any]:
		"""Serialise active datasets as W3C DCAT-AP JSON-LD.

		DCAT-AP is the EU application profile of DCAT, used by government open data
		portals and data mesh platforms. Returns a ``@graph`` of ``dcat:Dataset`` nodes.

		Returns:
			JSON-LD document with ``@context`` and ``@graph`` keys.
		"""
		tenant = self._tenant(tenant_id)
		datasets = [
			ds for ds in self.datasets.values()
			if ds["tenant_id"] == tenant and ds["status"] == "active"
		]
		graph: list[dict[str, Any]] = []
		for ds in datasets:
			node: dict[str, Any] = {
				"@type": "dcat:Dataset",
				"@id": f"urn:dcat:dataset:{ds['id']}",
				"dcterms:title": ds["name"],
				"dcterms:description": ds["description"],
				"dcterms:creator": ds["owner"],
				"dcterms:modified": ds.get("updated_at") or ds["created_at"],
				"dcterms:identifier": ds["id"],
				"dcat:keyword": ds["tags"],
				"dcterms:accessRights": ds["classification"],
				"dcat:distribution": [
					{
						"@type": "dcat:Distribution",
						"dcat:accessURL": ds.get("location_uri", ""),
						"dcterms:format": ds["format"],
					}
				] if ds.get("location_uri") else [],
			}
			graph.append(node)
		return {
			"@context": {
				"dcat": "http://www.w3.org/ns/dcat#",
				"dcterms": "http://purl.org/dc/terms/",
				"xsd": "http://www.w3.org/2001/XMLSchema#",
			},
			"@graph": graph,
			"generated_at": self._now(),
			"total": len(graph),
		}
