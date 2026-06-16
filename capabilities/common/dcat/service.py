# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from .models import (
	Dataset,
	DatasetSearch,
	DatasetTag,
	DataQualityDimension,
	LineageEdge,
	QualityScore,
	uuid7str,
)

log = logging.getLogger(__name__)


def _log_pretty_id(dataset_id: str) -> str:
	return dataset_id[:8] + "…"


class DataCatalogService:
	"""
	In-process Data Catalog service. Stores state in dictionaries keyed by
	tenant_id so all operations are multi-tenant isolated.

	Swap out the internal dicts for SQLAlchemy + Postgres sessions to
	productionise — the public async API surface stays identical.
	"""

	def __init__(self) -> None:
		# tenant_id -> {dataset_id -> Dataset}
		self._datasets: dict[str, dict[str, Dataset]] = defaultdict(dict)
		# tenant_id -> list[LineageEdge]
		self._edges: dict[str, list[LineageEdge]] = defaultdict(list)
		# tenant_id -> {dataset_id -> QualityScore}
		self._scores: dict[str, dict[str, QualityScore]] = defaultdict(dict)
		self._lock = asyncio.Lock()

	# ------------------------------------------------------------------
	# Dataset registration
	# ------------------------------------------------------------------

	async def register_dataset(self, ds: Dataset) -> str:
		"""Register a new dataset (or upsert by id). Returns dataset id."""
		async with self._lock:
			ds.updated_at = datetime.utcnow()
			self._datasets[ds.tenant_id][ds.id] = ds
			log.info("dcat.register_dataset tenant=%s id=%s name=%s", ds.tenant_id, _log_pretty_id(ds.id), ds.name)
			return ds.id

	async def get_dataset(self, tenant_id: str, dataset_id: str) -> Dataset | None:
		"""Retrieve a single dataset by id."""
		return self._datasets[tenant_id].get(dataset_id)

	# ------------------------------------------------------------------
	# Tagging
	# ------------------------------------------------------------------

	async def tag_dataset(self, dataset_id: str, tenant_id: str, tags: list[DatasetTag]) -> None:
		"""Append or overwrite tags on a dataset (merge by key)."""
		async with self._lock:
			ds = self._datasets[tenant_id].get(dataset_id)
			if ds is None:
				raise KeyError(f"Dataset {dataset_id!r} not found for tenant {tenant_id!r}")
			existing = {t.key: t for t in ds.tags}
			for tag in tags:
				existing[tag.key] = tag
			ds.tags = list(existing.values())
			ds.updated_at = datetime.utcnow()
			log.info("dcat.tag_dataset tenant=%s id=%s tags=%s", tenant_id, _log_pretty_id(dataset_id), [t.key for t in tags])

	# ------------------------------------------------------------------
	# Lineage
	# ------------------------------------------------------------------

	async def add_lineage(self, edge: LineageEdge) -> None:
		"""Record a directed lineage edge source -> target."""
		async with self._lock:
			# Prevent duplicate edges (same source/target/type)
			for existing in self._edges[edge.tenant_id]:
				if (
					existing.source_id == edge.source_id
					and existing.target_id == edge.target_id
					and existing.edge_type == edge.edge_type
				):
					log.debug("dcat.add_lineage: duplicate edge skipped %s->%s", edge.source_id, edge.target_id)
					return
			self._edges[edge.tenant_id].append(edge)
			log.info(
				"dcat.add_lineage tenant=%s %s->%s type=%s",
				edge.tenant_id,
				_log_pretty_id(edge.source_id),
				_log_pretty_id(edge.target_id),
				edge.edge_type,
			)

	async def get_lineage(self, dataset_id: str, tenant_id: str, depth: int = 3) -> dict[str, Any]:
		"""
		BFS traversal of lineage graph up to `depth` hops.
		Returns {nodes: [...], edges: [...]} in a format compatible with
		Apache Atlas lineage response schema.
		"""
		edges_for_tenant = self._edges[tenant_id]

		# Build adjacency: node -> list of (neighbour, edge)
		adj: dict[str, list[tuple[str, LineageEdge]]] = defaultdict(list)
		for e in edges_for_tenant:
			adj[e.source_id].append((e.target_id, e))
			adj[e.target_id].append((e.source_id, e))  # bidirectional for traversal

		visited: set[str] = set()
		collected_edges: list[LineageEdge] = []
		queue: deque[tuple[str, int]] = deque([(dataset_id, 0)])

		while queue:
			node_id, current_depth = queue.popleft()
			if node_id in visited:
				continue
			visited.add(node_id)
			if current_depth >= depth:
				continue
			for neighbour, edge in adj[node_id]:
				if neighbour not in visited:
					queue.append((neighbour, current_depth + 1))
				if edge not in collected_edges:
					collected_edges.append(edge)

		nodes = []
		for nid in visited:
			ds = self._datasets[tenant_id].get(nid)
			nodes.append({
				"guid": nid,
				"typeName": ds.type_name if ds else "DataSet",
				"displayText": ds.name if ds else nid,
				"status": ds.status if ds else "UNKNOWN",
			})

		return {
			"baseEntityGuid": dataset_id,
			"depth": depth,
			"nodes": nodes,
			"edges": [
				{
					"fromEntityId": e.source_id,
					"toEntityId": e.target_id,
					"relationshipId": e.id,
					"type": e.edge_type,
				}
				for e in collected_edges
			],
		}

	# ------------------------------------------------------------------
	# Search
	# ------------------------------------------------------------------

	async def search_datasets(self, query: DatasetSearch) -> list[Dataset]:
		"""Filter datasets by free-text + structured criteria."""
		results: list[Dataset] = list(self._datasets[query.tenant_id].values())

		if query.status is not None:
			results = [d for d in results if d.status == query.status]

		if query.format is not None:
			results = [d for d in results if d.format == query.format]

		if query.owner is not None:
			results = [d for d in results if d.owner == query.owner]

		if query.tag_key is not None:
			if query.tag_value is not None:
				results = [
					d for d in results
					if any(t.key == query.tag_key and t.value == query.tag_value for t in d.tags)
				]
			else:
				results = [
					d for d in results
					if any(t.key == query.tag_key for t in d.tags)
				]

		if query.classification is not None:
			results = [d for d in results if query.classification in d.classifications]

		if query.query:
			q = query.query.lower()
			results = [
				d for d in results
				if q in d.name.lower()
				or (d.description and q in d.description.lower())
				or any(q in t.key or q in t.value for t in d.tags)
			]

		# Sort by name for stable output
		results.sort(key=lambda d: d.name.lower())

		return results[query.offset : query.offset + query.limit]

	# ------------------------------------------------------------------
	# Quality scoring
	# ------------------------------------------------------------------

	async def score_quality(self, dataset_id: str, tenant_id: str) -> QualityScore:
		"""
		Compute a heuristic quality score for a dataset.

		Dimensions evaluated:
		  - completeness: fraction of optional fields populated
		  - lineage_coverage: whether lineage edges exist
		  - tagging: whether tags are present
		  - status: penalise deprecated/archived datasets
		"""
		ds = self._datasets[tenant_id].get(dataset_id)
		if ds is None:
			raise KeyError(f"Dataset {dataset_id!r} not found for tenant {tenant_id!r}")

		dimensions: list[DataQualityDimension] = []

		# 1. Completeness — optional fields: description, location, owner, schema_def
		optional_fields = [ds.description, ds.location, ds.owner, ds.schema_def]
		completeness = sum(1 for f in optional_fields if f is not None) / len(optional_fields)
		dimensions.append(DataQualityDimension(name="completeness", score=completeness, details="Optional fields populated"))

		# 2. Lineage coverage
		tenant_edges = self._edges[tenant_id]
		has_lineage = any(e.source_id == dataset_id or e.target_id == dataset_id for e in tenant_edges)
		dimensions.append(DataQualityDimension(name="lineage_coverage", score=1.0 if has_lineage else 0.0, details="At least one lineage edge present"))

		# 3. Tagging
		tag_score = min(len(ds.tags) / 3.0, 1.0)  # 3+ tags = perfect
		dimensions.append(DataQualityDimension(name="tagging", score=tag_score, details=f"{len(ds.tags)} tag(s) present"))

		# 4. Status health
		status_scores = {"active": 1.0, "draft": 0.5, "deprecated": 0.25, "archived": 0.1}
		status_score = status_scores.get(ds.status.value, 0.5)
		dimensions.append(DataQualityDimension(name="status_health", score=status_score, details=f"Dataset status: {ds.status}"))

		overall = sum(d.score for d in dimensions) / len(dimensions)

		score = QualityScore(
			dataset_id=dataset_id,
			tenant_id=tenant_id,
			overall=round(overall, 4),
			dimensions=dimensions,
		)
		async with self._lock:
			self._scores[tenant_id][dataset_id] = score

		log.info("dcat.score_quality tenant=%s id=%s overall=%.3f", tenant_id, _log_pretty_id(dataset_id), overall)
		return score
