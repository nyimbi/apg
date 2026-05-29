"""Deterministic helpers for APG Knowledge Graph."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class KnowledgeRuntime:
	"""Dependency-light graph helper routines used by the KNGR service."""

	def stable_id(self, prefix: str, payload: dict[str, Any]) -> str:
		material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
		digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
		return f"{prefix}-{digest}"

	def normalize_confidence(self, value: float) -> float:
		return max(0.0, min(1.0, round(float(value), 4)))

	def relationship_status(self, confidence_score: float, review_recorded: bool) -> str:
		confidence = self.normalize_confidence(confidence_score)
		if confidence < 0.7 and not review_recorded:
			return "review_required"
		if confidence < 0.7:
			return "accepted_with_review"
		return "active"

	def entity_curation_status(self, curation_recorded: bool, confidence_score: float) -> str:
		if curation_recorded:
			return "curated"
		if self.normalize_confidence(confidence_score) < 0.7:
			return "review_required"
		return "draft"

	def path_depth(self, relationship_ids: tuple[str, ...]) -> int:
		return max(len(relationship_ids), 0)

	def publication_status(self, entity_count: int, relationship_count: int) -> str:
		if entity_count == 0:
			return "empty"
		if relationship_count == 0:
			return "entity_snapshot"
		return "published"

	def neighborhood(
		self,
		entity_id: str,
		entities: list[dict[str, Any]],
		relationships: list[dict[str, Any]],
	) -> dict[str, Any]:
		edges = [
			relationship for relationship in relationships
			if relationship["subject_entity_id"] == entity_id or relationship["object_entity_id"] == entity_id
		]
		neighbor_ids = {
			relationship["object_entity_id"] if relationship["subject_entity_id"] == entity_id else relationship["subject_entity_id"]
			for relationship in edges
		}
		neighbor_entities = [entity for entity in entities if entity["id"] in neighbor_ids or entity["id"] == entity_id]
		return {
			"entity_id": entity_id,
			"entities": sorted(neighbor_entities, key=lambda item: item["id"]),
			"relationships": sorted(edges, key=lambda item: item["id"]),
			"neighbor_count": len(neighbor_ids),
			"relationship_count": len(edges),
		}
