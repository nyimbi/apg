"""Data Catalog service — dataset registry, lineage, metadata, glossary."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "dcat_cat"
SUPPORTED_FORMATS = {"csv", "parquet", "avro", "json", "orc", "delta", "iceberg", "unknown"}
SUPPORTED_CLASSIFICATIONS = {"public", "internal", "confidential", "restricted", "pii"}


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
