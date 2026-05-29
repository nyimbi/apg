"""Deterministic runtime helpers for ontology management."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Any


TERM_STATUSES = ("draft", "curated", "published", "deprecated")
MAPPING_TYPES = ("exact", "close", "broad", "narrow", "related")


def stable_id(prefix: str, *parts: object) -> str:
	"""Create short deterministic identifiers for repeatable APG tests and demos."""
	seed = "|".join(str(part) for part in parts if part is not None)
	digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def normalize_label(label: str) -> str:
	return " ".join(label.strip().lower().split())


def normalize_term_status(status: str | None) -> str:
	value = (status or "draft").strip().lower()
	if value not in TERM_STATUSES:
		raise ValueError(f"unsupported_term_status:{value}")
	return value


def normalize_mapping_type(mapping_type: str | None) -> str:
	value = (mapping_type or "exact").strip().lower()
	if value not in MAPPING_TYPES:
		raise ValueError(f"unsupported_mapping_type:{value}")
	return value


def normalize_confidence(confidence: float | int | str | None) -> float:
	if confidence is None:
		return 0.0
	value = float(confidence)
	if value < 0:
		return 0.0
	if value > 1:
		return 1.0
	return value


def bump_patch_version(version: str) -> str:
	parts = version.split(".")
	if len(parts) != 3 or not all(part.isdigit() for part in parts):
		return "0.1.0"
	major, minor, patch = (int(part) for part in parts)
	return f"{major}.{minor}.{patch + 1}"


def duplicate_labels(terms: list[dict[str, Any]]) -> list[str]:
	counts = Counter(normalize_label(term["label"]) for term in terms)
	return sorted(label for label, count in counts.items() if count > 1)


def taxonomy_has_cycle(edges: list[dict[str, Any]], parent_id: str, child_id: str) -> bool:
	if parent_id == child_id:
		return True
	children_by_parent: dict[str, list[str]] = {}
	for edge in edges:
		children_by_parent.setdefault(edge["parent_term_id"], []).append(edge["child_term_id"])
	children_by_parent.setdefault(parent_id, []).append(child_id)
	seen: set[str] = set()

	def visit(term_id: str, stack: set[str]) -> bool:
		if term_id in stack:
			return True
		if term_id in seen:
			return False
		seen.add(term_id)
		stack.add(term_id)
		for next_id in children_by_parent.get(term_id, []):
			if visit(next_id, stack):
				return True
		stack.remove(term_id)
		return False

	return visit(parent_id, set())


def mapping_requires_review(confidence: float, threshold: float) -> bool:
	return confidence < threshold


def publication_readiness(
	terms: list[dict[str, Any]],
	mappings: list[dict[str, Any]],
	duplicates: list[str],
	confidence_threshold: float = 0.8,
) -> tuple[bool, list[str]]:
	issues: list[str] = []
	if not terms:
		issues.append("term_required")
	if duplicates:
		issues.append("duplicate_term_detected")
	if any(term["status"] == "draft" for term in terms):
		issues.append("draft_terms_present")
	if any(mapping["confidence"] < confidence_threshold and not mapping["review_recorded"] for mapping in mappings):
		issues.append("mapping_review_required")
	return not issues, issues
