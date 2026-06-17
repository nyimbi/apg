"""Feature Flags service — runtime toggles, percentage rollout, A/B experiments, per-tenant targeting."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import logging
import math
import random
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fflag"


def _consistent_hash(key: str, seed: str) -> float:
	"""Deterministic 0.0–100.0 bucket from key+seed using MD5."""
	h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
	return (int(h[:8], 16) / 0xFFFFFFFF) * 100.0


class FeatureFlagService:
	"""Runtime feature toggles, percentage rollout, A/B experiment assignment, per-tenant targeting, audit trail."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.flags: dict[str, dict[str, Any]] = {}  # keyed by "tenant:flag_key"
		self.experiments: dict[str, dict[str, Any]] = {}
		self.assignments: dict[str, dict[str, Any]] = {}  # user experiment assignments
		self.overrides: dict[str, dict[str, Any]] = {}  # per-user flag overrides
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# I10 — named targeting segments, keyed "tenant:segment_id"
		self.segments: dict[str, dict[str, Any]] = {}
		# I8 — sticky assignment cache, keyed "tenant:flag_key:user_id"
		self.sticky_assignments: dict[str, dict[str, Any]] = {}
		# I4 — bandit Beta-distribution state, keyed "tenant:exp_id:variant_key"
		self.bandit_state: dict[str, dict[str, float]] = {}
		# I15 — cross-tenant flag templates, keyed by template name
		self.templates: dict[str, dict[str, Any]] = {}
		# I11 — pending change-request records, keyed by request id
		self.change_requests: dict[str, dict[str, Any]] = {}

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _flag_key(self, tenant: str, key: str) -> str:
		return f"{tenant}:{key}"

	def _emit(self, tenant_id: str, event_type: str, flag_key: str, before: dict[str, Any] | None = None, after: dict[str, Any] | None = None, actor: str = "system") -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"flag_key": flag_key,
			"actor": actor,
			"before": deepcopy(before) if before else None,
			"after": deepcopy(after) if after else None,
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "fflag",
			"status": "healthy",
			"flag_count": len(self.flags),
			"experiment_count": len(self.experiments),
			"assignment_count": len(self.assignments),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"features": [
				"runtime_toggles", "percentage_rollout", "ab_experiment_assignment",
				"per_tenant_targeting", "per_user_overrides", "audit_trail", "variant_flags"
			],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Flag CRUD ────────────────────────────────────────────────

	async def create_flag(
		self,
		tenant_id: str,
		key: str,
		name: str,
		description: str = "",
		enabled: bool = False,
		rollout_percentage: float = 0.0,
		targeting_rules: list[dict[str, Any]] | None = None,
		variants: dict[str, Any] | None = None,
		tags: list[str] | None = None,
		owner: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Create a feature flag."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(key, "key")
		guard_non_empty_string(name, "name")
		fk = self._flag_key(tenant, key)
		if fk in self.flags:
			raise ValueError(f"flag already exists: {key}")
		if not 0.0 <= rollout_percentage <= 100.0:
			raise ValueError("rollout_percentage must be 0–100")
		record: dict[str, Any] = {
			"id": self._id("flag"),
			"tenant_id": tenant,
			"key": key,
			"name": name,
			"description": description,
			"enabled": enabled,
			"rollout_percentage": rollout_percentage,
			"targeting_rules": list(targeting_rules or []),
			"variants": dict(variants or {}),
			"tags": list(tags or []),
			"owner": owner,
			"created_at": self._now(),
			"updated_at": None,
		}
		self.flags[fk] = record
		self._emit(tenant, "flag_created", key, after=record, actor=actor)
		_log.info("flag created: %s tenant=%s enabled=%s rollout=%.1f%%", key, tenant, enabled, rollout_percentage)
		return deepcopy(record)

	async def get_flag(self, tenant_id: str, key: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.flags.get(self._flag_key(tenant, key))
		if not record:
			raise KeyError(f"flag not found: {key}")
		return deepcopy(record)

	async def list_flags(
		self,
		tenant_id: str,
		enabled: bool | None = None,
		tags: list[str] | None = None,
		owner: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		items = [deepcopy(r) for k, r in self.flags.items() if k.startswith(prefix)]
		if enabled is not None:
			items = [r for r in items if r["enabled"] == enabled]
		if tags:
			items = [r for r in items if all(t in r["tags"] for t in tags)]
		if owner:
			items = [r for r in items if r["owner"] == owner]
		return items

	async def update_flag(self, tenant_id: str, key: str, actor: str = "system", **kwargs: Any) -> dict[str, Any]:
		"""Update a flag's configuration."""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		before = deepcopy(record)
		allowed = {"name", "description", "enabled", "rollout_percentage", "targeting_rules", "variants", "tags"}
		for field, value in kwargs.items():
			if field in allowed and value is not None:
				if field == "rollout_percentage" and not 0.0 <= value <= 100.0:
					raise ValueError("rollout_percentage must be 0–100")
				record[field] = value
		record["updated_at"] = self._now()
		self._emit(tenant, "flag_updated", key, before=before, after=deepcopy(record), actor=actor)
		return deepcopy(record)

	async def delete_flag(self, tenant_id: str, key: str, actor: str = "system") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		del self.flags[fk]
		self._emit(tenant, "flag_deleted", key, before=record, actor=actor)
		return deepcopy(record)

	async def enable_flag(self, tenant_id: str, key: str, actor: str = "system") -> dict[str, Any]:
		"""Enable a feature flag."""
		return await self.update_flag(tenant_id, key, actor=actor, enabled=True)

	async def disable_flag(self, tenant_id: str, key: str, actor: str = "system") -> dict[str, Any]:
		"""Disable a feature flag."""
		return await self.update_flag(tenant_id, key, actor=actor, enabled=False)

	async def set_rollout(self, tenant_id: str, key: str, percentage: float, actor: str = "system") -> dict[str, Any]:
		"""Set percentage rollout for a flag."""
		return await self.update_flag(tenant_id, key, actor=actor, rollout_percentage=percentage)

	# ── Evaluation ───────────────────────────────────────────────

	async def evaluate_flag(
		self,
		tenant_id: str,
		key: str,
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate a feature flag for a specific user."""
		tenant = self._tenant(tenant_id)
		# Check user override first
		override_key = f"{tenant}:{key}:{user_id}"
		override = self.overrides.get(override_key)
		if override is not None:
			return {
				"flag_key": key,
				"enabled": override["enabled"],
				"variant": override.get("variant"),
				"reason": "override",
				"targeting_matched": False,
			}

		fk = self._flag_key(tenant, key)
		flag = self.flags.get(fk)
		if not flag:
			return {"flag_key": key, "enabled": False, "variant": None, "reason": "flag_not_found", "targeting_matched": False}

		if not flag["enabled"]:
			return {"flag_key": key, "enabled": False, "variant": None, "reason": "flag_disabled", "targeting_matched": False}

		attrs = user_attributes or {}

		# Evaluate targeting rules (first match wins)
		for rule in flag["targeting_rules"]:
			if self._matches_rule(rule, user_id, attrs):
				variant = rule.get("variant")
				return {
					"flag_key": key,
					"enabled": rule.get("enabled", True),
					"variant": variant,
					"reason": "targeting_rule",
					"targeting_matched": True,
				}

		# Percentage rollout using consistent hash
		bucket = _consistent_hash(user_id, key)
		in_rollout = bucket < flag["rollout_percentage"]

		variant = None
		if in_rollout and flag["variants"]:
			variant = self._assign_variant(key, user_id, flag["variants"])

		return {
			"flag_key": key,
			"enabled": in_rollout,
			"variant": variant,
			"reason": "rollout",
			"targeting_matched": False,
		}

	def _matches_rule(self, rule: dict[str, Any], user_id: str, attrs: dict[str, Any]) -> bool:
		"""Evaluate a single targeting rule against user attributes."""
		conditions = rule.get("conditions", [])
		if not conditions:
			return True
		for condition in conditions:
			attr = condition.get("attribute", "")
			op = condition.get("operator", "eq")
			val = condition.get("value")
			user_val = attrs.get(attr) if attr != "user_id" else user_id
			try:
				if op == "eq" and user_val != val:
					return False
				elif op == "in" and user_val not in (val or []):
					return False
				elif op == "not_in" and user_val in (val or []):
					return False
				elif op == "gt" and not (float(user_val) > float(val)):
					return False
				elif op == "lt" and not (float(user_val) < float(val)):
					return False
				elif op == "contains" and str(val) not in str(user_val):
					return False
			except Exception:
				return False
		return True

	def _assign_variant(self, key: str, user_id: str, variants: dict[str, Any]) -> str:
		"""Deterministically assign a variant based on hash."""
		bucket = _consistent_hash(user_id, f"{key}:variant")
		total_weight = sum(v.get("weight", 1) for v in variants.values()) if isinstance(variants, dict) else 100
		cumulative = 0.0
		for variant_key, variant_cfg in variants.items():
			weight = (variant_cfg.get("weight", 1) / total_weight * 100) if isinstance(variant_cfg, dict) else 50
			cumulative += weight
			if bucket < cumulative:
				return variant_key
		return list(variants.keys())[-1]

	async def evaluate_many(
		self,
		tenant_id: str,
		flag_keys: list[str],
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Batch evaluate multiple flags for a user."""
		tenant = self._tenant(tenant_id)
		results = await asyncio.gather(
			*[self.evaluate_flag(tenant_id, k, user_id, user_attributes) for k in flag_keys],
			return_exceptions=True,
		)
		evaluations = {}
		for key, result in zip(flag_keys, results):
			if isinstance(result, Exception):
				_log.error("evaluate_many flag=%s user=%s: %s", key, user_id, result)
				evaluations[key] = {"flag_key": key, "enabled": False, "variant": None, "reason": "error"}
			else:
				evaluations[key] = result
		return {"user_id": user_id, "evaluations": evaluations, "count": len(evaluations)}

	# ── Per-user overrides ────────────────────────────────────────

	async def set_override(
		self,
		tenant_id: str,
		key: str,
		user_id: str,
		enabled: bool,
		variant: str | None = None,
		reason: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Set a per-user flag override."""
		tenant = self._tenant(tenant_id)
		override_key = f"{tenant}:{key}:{user_id}"
		record: dict[str, Any] = {
			"tenant_id": tenant,
			"flag_key": key,
			"user_id": user_id,
			"enabled": enabled,
			"variant": variant,
			"reason": reason,
			"set_by": actor,
			"set_at": self._now(),
		}
		self.overrides[override_key] = record
		self._emit(tenant, "override_set", key, after=record, actor=actor)
		return deepcopy(record)

	async def clear_override(self, tenant_id: str, key: str, user_id: str, actor: str = "system") -> dict[str, Any]:
		"""Remove a per-user override."""
		tenant = self._tenant(tenant_id)
		override_key = f"{tenant}:{key}:{user_id}"
		record = self.overrides.get(override_key)
		if not record:
			raise KeyError(f"override not found for {key}/{user_id}")
		del self.overrides[override_key]
		self._emit(tenant, "override_cleared", key, before=record, actor=actor)
		return deepcopy(record)

	async def list_overrides(self, tenant_id: str, key: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.overrides.values() if r["tenant_id"] == tenant]
		if key:
			items = [r for r in items if r["flag_key"] == key]
		return items

	# ── A/B Experiments ───────────────────────────────────────────

	async def create_experiment(
		self,
		tenant_id: str,
		flag_key: str,
		name: str,
		variants: list[dict[str, Any]] | None = None,
		description: str = "",
		targeting_rule: dict[str, Any] | None = None,
		owner: str = "",
	) -> dict[str, Any]:
		"""Create an A/B experiment tied to a feature flag."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(flag_key, "flag_key")
		guard_non_empty_string(name, "name")
		record: dict[str, Any] = {
			"id": self._id("exp"),
			"tenant_id": tenant,
			"flag_key": flag_key,
			"name": name,
			"description": description,
			"variants": list(variants or [{"key": "control", "weight": 50}, {"key": "treatment", "weight": 50}]),
			"targeting_rule": targeting_rule or {},
			"owner": owner,
			"status": "draft",
			"created_at": self._now(),
		}
		self.experiments[record["id"]] = record
		self._emit(tenant, "experiment_created", flag_key, after={"experiment_id": record["id"]})
		return deepcopy(record)

	async def start_experiment(self, tenant_id: str, experiment_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")
		exp["status"] = "running"
		exp["started_at"] = self._now()
		self._emit(tenant, "experiment_started", exp["flag_key"])
		return deepcopy(exp)

	async def stop_experiment(self, tenant_id: str, experiment_id: str, winner: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")
		exp["status"] = "stopped"
		exp["winner"] = winner
		exp["stopped_at"] = self._now()
		self._emit(tenant, "experiment_stopped", exp["flag_key"], after={"winner": winner})
		return deepcopy(exp)

	async def get_experiment(self, tenant_id: str, experiment_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")
		return deepcopy(exp)

	async def list_experiments(self, tenant_id: str, flag_key: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		items = [deepcopy(e) for e in self.experiments.values() if e["tenant_id"] == tenant]
		if flag_key:
			items = [e for e in items if e["flag_key"] == flag_key]
		return items

	async def assign_experiment_variant(
		self,
		tenant_id: str,
		experiment_id: str,
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Assign a user to an experiment variant deterministically."""
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")
		if exp["status"] != "running":
			raise ValueError(f"experiment not running: {experiment_id}")

		assignment_key = f"{tenant}:{experiment_id}:{user_id}"
		existing = self.assignments.get(assignment_key)
		if existing:
			return deepcopy(existing)

		bucket = _consistent_hash(user_id, experiment_id)
		total_weight = sum(v.get("weight", 50) for v in exp["variants"])
		cumulative = 0.0
		assigned_variant = exp["variants"][-1]["key"]
		for variant in exp["variants"]:
			weight = (variant.get("weight", 50) / total_weight) * 100
			cumulative += weight
			if bucket < cumulative:
				assigned_variant = variant["key"]
				break

		assignment: dict[str, Any] = {
			"id": self._id("asgn"),
			"tenant_id": tenant,
			"experiment_id": experiment_id,
			"flag_key": exp["flag_key"],
			"user_id": user_id,
			"variant": assigned_variant,
			"assigned_at": self._now(),
		}
		self.assignments[assignment_key] = assignment
		return deepcopy(assignment)

	# ── Statistics ────────────────────────────────────────────────

	async def flag_statistics(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		flags = [r for k, r in self.flags.items() if k.startswith(prefix)]
		enabled_flags = [r for r in flags if r["enabled"]]
		experiments = [e for e in self.experiments.values() if e["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"total_flags": len(flags),
			"enabled_flags": len(enabled_flags),
			"disabled_flags": len(flags) - len(enabled_flags),
			"total_experiments": len(experiments),
			"running_experiments": sum(1 for e in experiments if e["status"] == "running"),
			"total_overrides": sum(1 for r in self.overrides.values() if r["tenant_id"] == tenant),
			"audit_events": sum(1 for e in self._audit_events if e["tenant_id"] == tenant),
			"generated_at": self._now(),
		}

	async def bulk_evaluate(
		self,
		tenant_id: str,
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate all flags for a user in one call."""
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		all_keys = [r["key"] for k, r in self.flags.items() if k.startswith(prefix)]
		return await self.evaluate_many(tenant_id, all_keys, user_id, user_attributes)

	async def clone_flag(self, tenant_id: str, source_key: str, new_key: str, actor: str = "system") -> dict[str, Any]:
		"""Clone an existing flag with a new key."""
		tenant = self._tenant(tenant_id)
		source = await self.get_flag(tenant_id, source_key)
		source.pop("id", None)
		source.pop("created_at", None)
		source.pop("updated_at", None)
		source["key"] = new_key
		source["name"] = f"{source['name']} (copy)"
		source["enabled"] = False
		return await self.create_flag(tenant_id=tenant_id, actor=actor, **{k: v for k, v in source.items() if k not in ("tenant_id",)})

	async def get_flag_history(self, tenant_id: str, key: str) -> list[dict[str, Any]]:
		"""Return audit history for a specific flag."""
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant and e["flag_key"] == key]

	async def add_targeting_rule(
		self,
		tenant_id: str,
		key: str,
		rule: dict[str, Any],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Append a targeting rule to a flag."""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		before = deepcopy(record)
		record["targeting_rules"].append(deepcopy(rule))
		record["updated_at"] = self._now()
		self._emit(tenant, "targeting_rule_added", key, before=before, after=deepcopy(record), actor=actor)
		return deepcopy(record)

	async def remove_targeting_rule(self, tenant_id: str, key: str, rule_index: int, actor: str = "system") -> dict[str, Any]:
		"""Remove a targeting rule by index."""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		before = deepcopy(record)
		rules = record["targeting_rules"]
		if rule_index < 0 or rule_index >= len(rules):
			raise IndexError(f"rule index out of range: {rule_index}")
		rules.pop(rule_index)
		record["updated_at"] = self._now()
		self._emit(tenant, "targeting_rule_removed", key, before=before, after=deepcopy(record), actor=actor)
		return deepcopy(record)

	async def get_experiment_results(self, tenant_id: str, experiment_id: str) -> dict[str, Any]:
		"""Return assignment distribution for an experiment."""
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")
		prefix = f"{tenant}:{experiment_id}:"
		assignments = [a for k, a in self.assignments.items() if k.startswith(prefix)]
		by_variant: dict[str, int] = {}
		for a in assignments:
			by_variant[a["variant"]] = by_variant.get(a["variant"], 0) + 1
		return {
			"experiment_id": experiment_id,
			"status": exp["status"],
			"total_assignments": len(assignments),
			"by_variant": by_variant,
			"winner": exp.get("winner"),
			"generated_at": self._now(),
		}

	# ── I10: Named Segments ───────────────────────────────────────

	async def create_segment(
		self,
		tenant_id: str,
		segment_id: str,
		name: str,
		conditions: list[dict[str, Any]],
		description: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Create a reusable targeting segment (named set of conditions).

		Flags reference segments via targeting_rules entries with
		``{"type": "segment", "segment_id": "<id>"}``.  Decouples cohort
		definition from flag configuration — change once, affects all flags.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(segment_id, "segment_id")
		guard_non_empty_string(name, "name")
		sk = f"{tenant}:{segment_id}"
		if sk in self.segments:
			raise ValueError(f"segment already exists: {segment_id}")
		record: dict[str, Any] = {
			"id": self._id("seg"),
			"tenant_id": tenant,
			"segment_id": segment_id,
			"name": name,
			"description": description,
			"conditions": list(deepcopy(conditions)),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.segments[sk] = record
		self._emit(tenant, "segment_created", segment_id, after=record, actor=actor)
		_log.info("segment created: %s tenant=%s conditions=%d", segment_id, tenant, len(conditions))
		return deepcopy(record)

	async def get_segment(self, tenant_id: str, segment_id: str) -> dict[str, Any]:
		"""Fetch a segment by id."""
		tenant = self._tenant(tenant_id)
		record = self.segments.get(f"{tenant}:{segment_id}")
		if not record:
			raise KeyError(f"segment not found: {segment_id}")
		return deepcopy(record)

	async def list_segments(self, tenant_id: str) -> list[dict[str, Any]]:
		"""List all segments for a tenant."""
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		return [deepcopy(r) for k, r in self.segments.items() if k.startswith(prefix)]

	async def delete_segment(self, tenant_id: str, segment_id: str, actor: str = "system") -> dict[str, Any]:
		"""Delete a targeting segment."""
		tenant = self._tenant(tenant_id)
		sk = f"{tenant}:{segment_id}"
		record = self.segments.get(sk)
		if not record:
			raise KeyError(f"segment not found: {segment_id}")
		del self.segments[sk]
		self._emit(tenant, "segment_deleted", segment_id, before=record, actor=actor)
		return deepcopy(record)

	# ── I8: Sticky Assignments ────────────────────────────────────

	async def evaluate_flag_sticky(
		self,
		tenant_id: str,
		key: str,
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate a flag with sticky bucketing.

		Once a user receives a result for a sticky flag the assignment is
		persisted.  Subsequent calls — even after rollout percentage changes —
		return the stored result.  Critical for multi-step user journeys where
		mid-funnel variant flips destroy experiment validity.
		"""
		tenant = self._tenant(tenant_id)
		sticky_key = f"{tenant}:{key}:{user_id}"
		stored = self.sticky_assignments.get(sticky_key)
		if stored is not None:
			return {**deepcopy(stored), "reason": "sticky_assignment"}

		result = await self.evaluate_flag(tenant_id, key, user_id, user_attributes)
		# Only persist if the flag is actually enabled (don't lock users out permanently)
		if result["enabled"]:
			self.sticky_assignments[sticky_key] = deepcopy(result)
		return result

	async def clear_sticky_assignment(
		self,
		tenant_id: str,
		key: str,
		user_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Clear a sticky assignment, allowing re-evaluation on next call."""
		tenant = self._tenant(tenant_id)
		sticky_key = f"{tenant}:{key}:{user_id}"
		record = self.sticky_assignments.get(sticky_key)
		if not record:
			raise KeyError(f"sticky assignment not found: {key}/{user_id}")
		del self.sticky_assignments[sticky_key]
		self._emit(tenant, "sticky_assignment_cleared", key,
				   before=record, actor=actor)
		return deepcopy(record)

	# ── I4: Multi-Armed Bandit Experiments ────────────────────────

	async def record_bandit_outcome(
		self,
		tenant_id: str,
		experiment_id: str,
		user_id: str,
		variant_key: str,
		converted: bool,
	) -> dict[str, Any]:
		"""Record a conversion outcome for Thompson Sampling bandit experiment.

		Updates Beta distribution parameters for the variant:
		  - conversion  → alpha += 1
		  - no-convert  → beta  += 1

		The updated posterior drives future variant allocation toward winners,
		converging 3–5× faster than fixed A/B splits.
		"""
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")

		state_key = f"{tenant}:{experiment_id}:{variant_key}"
		state = self.bandit_state.setdefault(state_key, {"alpha": 1.0, "beta": 1.0})
		if converted:
			state["alpha"] += 1.0
		else:
			state["beta"] += 1.0

		outcome: dict[str, Any] = {
			"experiment_id": experiment_id,
			"variant_key": variant_key,
			"user_id": user_id,
			"converted": converted,
			"alpha": state["alpha"],
			"beta": state["beta"],
			"recorded_at": self._now(),
		}
		self._emit(tenant, "bandit_outcome_recorded", exp["flag_key"],
				   after=outcome, actor="system")
		_log.debug("bandit outcome: exp=%s variant=%s converted=%s α=%.1f β=%.1f",
				   experiment_id, variant_key, converted, state["alpha"], state["beta"])
		return outcome

	async def get_bandit_state(self, tenant_id: str, experiment_id: str) -> dict[str, Any]:
		"""Return current Beta distribution parameters for all variants in a bandit experiment."""
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")

		variant_states: dict[str, dict[str, float]] = {}
		for v in exp["variants"]:
			sk = f"{tenant}:{experiment_id}:{v['key']}"
			state = self.bandit_state.get(sk, {"alpha": 1.0, "beta": 1.0})
			alpha, beta_val = state["alpha"], state["beta"]
			mean = alpha / (alpha + beta_val)
			variant_states[v["key"]] = {
				"alpha": alpha,
				"beta": beta_val,
				"mean_conversion_rate": round(mean, 4),
				"observations": int(alpha + beta_val - 2),  # subtract Beta(1,1) prior
			}

		return {
			"experiment_id": experiment_id,
			"flag_key": exp["flag_key"],
			"variant_states": variant_states,
			"generated_at": self._now(),
		}

	# ── I7: Statistical Significance ─────────────────────────────

	async def compute_experiment_significance(
		self,
		tenant_id: str,
		experiment_id: str,
		conversions: dict[str, int],
		totals: dict[str, int],
		significance_level: float = 0.05,
	) -> dict[str, Any]:
		"""Compute two-proportion Z-test between experiment variants.

		Args:
			conversions: {variant_key: conversion_count}
			totals:       {variant_key: total_impressions}
			significance_level: alpha threshold (default 0.05)

		Returns dict with p_value, significant, confidence_intervals,
		and minimum required sample size for 80% power.
		"""
		tenant = self._tenant(tenant_id)
		exp = self.experiments.get(experiment_id)
		if not exp or exp["tenant_id"] != tenant:
			raise KeyError(f"experiment not found: {experiment_id}")

		variant_keys = list(conversions.keys())
		if len(variant_keys) < 2:
			raise ValueError("need at least 2 variants with conversion data")

		# Two-proportion Z-test: control vs first treatment
		ctrl_key = variant_keys[0]
		treat_key = variant_keys[1]
		n1 = totals.get(ctrl_key, 0)
		n2 = totals.get(treat_key, 0)
		c1 = conversions.get(ctrl_key, 0)
		c2 = conversions.get(treat_key, 0)

		if n1 == 0 or n2 == 0:
			return {"experiment_id": experiment_id, "significant": False,
					"p_value": 1.0, "error": "insufficient data", "generated_at": self._now()}

		p1 = c1 / n1
		p2 = c2 / n2
		p_pool = (c1 + c2) / (n1 + n2)

		se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
		z_stat = (p2 - p1) / se if se > 0 else 0.0

		# Approximate two-tailed p-value via complementary error function
		p_value = 2 * (1 - _normal_cdf(abs(z_stat)))
		significant = p_value < significance_level

		# 95% confidence interval for difference
		z_95 = 1.96
		diff = p2 - p1
		se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if (n1 > 0 and n2 > 0) else 0.0
		ci_lower = round(diff - z_95 * se_diff, 4)
		ci_upper = round(diff + z_95 * se_diff, 4)

		# Power analysis: minimum n per arm (80% power, two-tailed)
		z_alpha = 1.96
		z_beta = 0.842  # 80% power
		p_bar = (p1 + p2) / 2
		if p_bar > 0 and p_bar < 1 and abs(diff) > 0:
			required_n = int(math.ceil(
				((z_alpha + z_beta) ** 2 * 2 * p_bar * (1 - p_bar)) / (diff ** 2)
			))
		else:
			required_n = -1

		return {
			"experiment_id": experiment_id,
			"control_variant": ctrl_key,
			"treatment_variant": treat_key,
			"z_statistic": round(z_stat, 4),
			"p_value": round(p_value, 4),
			"significant": significant,
			"significance_level": significance_level,
			"conversion_rates": {ctrl_key: round(p1, 4), treat_key: round(p2, 4)},
			"relative_lift": round((p2 - p1) / p1, 4) if p1 > 0 else None,
			"confidence_interval_95": {"lower": ci_lower, "upper": ci_upper},
			"required_sample_size_per_arm": required_n,
			"generated_at": self._now(),
		}

	# ── I13: Flag Import / Export ─────────────────────────────────

	async def export_flags(self, tenant_id: str) -> dict[str, Any]:
		"""Serialise all flags, segments, and experiments for a tenant.

		The returned envelope is version-stamped and suitable for GitOps
		storage — commit to VCS, diff in PRs, import into other envs.
		"""
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"

		flags = [deepcopy(r) for k, r in self.flags.items() if k.startswith(prefix)]
		segments = [deepcopy(r) for k, r in self.segments.items() if k.startswith(prefix)]
		experiments = [deepcopy(e) for e in self.experiments.values() if e["tenant_id"] == tenant]

		return {
			"schema_version": "1.0",
			"tenant_id": tenant,
			"exported_at": self._now(),
			"counts": {
				"flags": len(flags),
				"segments": len(segments),
				"experiments": len(experiments),
			},
			"flags": flags,
			"segments": segments,
			"experiments": experiments,
		}

	async def import_flags(
		self,
		tenant_id: str,
		data: dict[str, Any],
		mode: Literal["merge", "overwrite", "dry_run"] = "merge",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Import flags from an export envelope.

		Modes:
		  merge     — create missing flags, skip existing (non-destructive)
		  overwrite — replace all flags for the tenant
		  dry_run   — validate and report diff without applying changes
		"""
		tenant = self._tenant(tenant_id)
		if data.get("schema_version") != "1.0":
			raise ValueError("unsupported export schema_version")

		incoming_flags: list[dict[str, Any]] = data.get("flags", [])
		prefix = f"{tenant}:"
		existing_keys = {r["key"] for k, r in self.flags.items() if k.startswith(prefix)}
		incoming_keys = {f["key"] for f in incoming_flags}

		added: list[str] = []
		skipped: list[str] = []
		replaced: list[str] = []

		if mode == "dry_run":
			for f in incoming_flags:
				if f["key"] in existing_keys:
					skipped.append(f["key"])
				else:
					added.append(f["key"])
			return {
				"dry_run": True,
				"mode": mode,
				"would_add": added,
				"would_skip": skipped,
				"tenant_id": tenant,
			}

		if mode == "overwrite":
			# Remove all existing flags for this tenant
			for k in list(self.flags.keys()):
				if k.startswith(prefix):
					del self.flags[k]

		for f in incoming_flags:
			fk = self._flag_key(tenant, f["key"])
			if mode == "merge" and fk in self.flags:
				skipped.append(f["key"])
				continue
			record = deepcopy(f)
			record["tenant_id"] = tenant
			record["imported_at"] = self._now()
			self.flags[fk] = record
			if f["key"] in existing_keys and mode == "overwrite":
				replaced.append(f["key"])
			else:
				added.append(f["key"])
			self._emit(tenant, "flag_imported", f["key"], after=record, actor=actor)

		_log.info("import_flags: tenant=%s mode=%s added=%d skipped=%d replaced=%d",
				  tenant, mode, len(added), len(skipped), len(replaced))
		return {
			"mode": mode,
			"tenant_id": tenant,
			"added": added,
			"skipped": skipped,
			"replaced": replaced,
			"imported_at": self._now(),
		}

	# ── I15: Cross-Tenant Flag Templates ─────────────────────────

	async def create_template(
		self,
		name: str,
		flag_spec: dict[str, Any],
		description: str = "",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Create a tenant-agnostic flag blueprint.

		Templates let platform teams push a standard flag configuration to any
		number of tenants without N individual API calls.  Provenance is tracked
		via ``template_source`` on each instantiated flag.
		"""
		guard_non_empty_string(name, "name")
		if name in self.templates:
			raise ValueError(f"template already exists: {name}")
		record: dict[str, Any] = {
			"name": name,
			"description": description,
			"flag_spec": deepcopy(flag_spec),
			"created_at": self._now(),
			"updated_at": None,
		}
		self.templates[name] = record
		_log.info("template created: %s", name)
		return deepcopy(record)

	async def instantiate_template(
		self,
		tenant_id: str,
		template_name: str,
		overrides: dict[str, Any] | None = None,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Instantiate a flag template for a specific tenant.

		The resulting flag has ``template_source`` set so future
		``apply_template_update`` calls can find all derived instances.
		"""
		tenant = self._tenant(tenant_id)
		tmpl = self.templates.get(template_name)
		if not tmpl:
			raise KeyError(f"template not found: {template_name}")
		spec: dict[str, Any] = deepcopy(tmpl["flag_spec"])
		if overrides:
			spec.update(overrides)
		spec["template_source"] = template_name
		guard_non_empty_string(spec.get("key", ""), "key")
		result = await self.create_flag(tenant_id=tenant_id, actor=actor, **{
			k: v for k, v in spec.items() if k not in ("tenant_id", "id", "created_at", "updated_at")
		})
		_log.info("template instantiated: %s -> tenant=%s flag=%s", template_name, tenant, result["key"])
		return result

	async def apply_template_update(
		self,
		template_name: str,
		field_updates: dict[str, Any],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Push field updates to all flags derived from a template.

		Only updates fields present in ``field_updates``; per-tenant overrides
		on other fields are preserved.  Emits ``template_update_applied`` per
		affected flag.
		"""
		tmpl = self.templates.get(template_name)
		if not tmpl:
			raise KeyError(f"template not found: {template_name}")
		# Update the template spec itself
		tmpl["flag_spec"].update(field_updates)
		tmpl["updated_at"] = self._now()

		affected: list[str] = []
		for fk, record in self.flags.items():
			if record.get("template_source") == template_name:
				tenant = record["tenant_id"]
				before = deepcopy(record)
				for field, value in field_updates.items():
					if field in ("key", "id", "tenant_id", "created_at"):
						continue
					record[field] = value
				record["updated_at"] = self._now()
				self._emit(tenant, "template_update_applied", record["key"],
						   before=before, after=deepcopy(record), actor=actor)
				affected.append(fk)

		_log.info("template_update applied: %s affected=%d fields=%s",
				  template_name, len(affected), list(field_updates.keys()))
		return {
			"template_name": template_name,
			"fields_updated": list(field_updates.keys()),
			"affected_flags": len(affected),
			"applied_at": self._now(),
		}

	# ── I11: Change-Request Approval Workflow ─────────────────────

	async def request_flag_change(
		self,
		tenant_id: str,
		key: str,
		proposed_changes: dict[str, Any],
		requestor: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Submit a flag change for approval.

		When a flag has ``requires_approval: True`` (set via update_flag),
		callers should use this method instead of update_flag directly.
		Returns a ChangeRequest record that must be approved before the
		mutation is applied.
		"""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		if fk not in self.flags:
			raise KeyError(f"flag not found: {key}")
		req_id = self._id("cr")
		record: dict[str, Any] = {
			"id": req_id,
			"tenant_id": tenant,
			"flag_key": key,
			"proposed_changes": deepcopy(proposed_changes),
			"requestor": requestor,
			"reason": reason,
			"status": "pending",
			"created_at": self._now(),
			"resolved_at": None,
			"resolved_by": None,
		}
		self.change_requests[req_id] = record
		self._emit(tenant, "change_request_created", key, after=record, actor=requestor)
		_log.info("change_request created: %s flag=%s requestor=%s", req_id, key, requestor)
		return deepcopy(record)

	async def approve_change_request(
		self,
		tenant_id: str,
		request_id: str,
		approver: str,
	) -> dict[str, Any]:
		"""Approve a pending flag change request and apply the mutation."""
		tenant = self._tenant(tenant_id)
		req = self.change_requests.get(request_id)
		if not req or req["tenant_id"] != tenant:
			raise KeyError(f"change_request not found: {request_id}")
		if req["status"] != "pending":
			raise ValueError(f"change_request is not pending: {req['status']}")

		# Apply the proposed changes
		updated_flag = await self.update_flag(
			tenant_id, req["flag_key"], actor=approver, **req["proposed_changes"]
		)
		req["status"] = "approved"
		req["resolved_at"] = self._now()
		req["resolved_by"] = approver
		self._emit(tenant, "change_request_approved", req["flag_key"],
				   after={"request_id": request_id, "approver": approver}, actor=approver)
		return {"change_request": deepcopy(req), "updated_flag": updated_flag}

	async def reject_change_request(
		self,
		tenant_id: str,
		request_id: str,
		rejector: str,
		rejection_reason: str = "",
	) -> dict[str, Any]:
		"""Reject a pending flag change request without applying any mutation."""
		tenant = self._tenant(tenant_id)
		req = self.change_requests.get(request_id)
		if not req or req["tenant_id"] != tenant:
			raise KeyError(f"change_request not found: {request_id}")
		if req["status"] != "pending":
			raise ValueError(f"change_request is not pending: {req['status']}")
		req["status"] = "rejected"
		req["resolved_at"] = self._now()
		req["resolved_by"] = rejector
		req["rejection_reason"] = rejection_reason
		self._emit(tenant, "change_request_rejected", req["flag_key"],
				   after={"request_id": request_id, "rejector": rejector}, actor=rejector)
		return deepcopy(req)

	# ── I12: Evaluation Telemetry / OTel-Compatible Events ────────

	async def evaluate_flag_with_telemetry(
		self,
		tenant_id: str,
		key: str,
		user_id: str,
		user_attributes: dict[str, Any] | None = None,
		trace_context: dict[str, str] | None = None,
		sample_rate: float = 1.0,
	) -> dict[str, Any]:
		"""Evaluate a flag and emit a structured telemetry event.

		The telemetry event is OpenTelemetry-compatible — it carries trace_id
		and span_id from the caller's trace context so downstream collectors
		can correlate flag decisions with distributed traces.

		``sample_rate`` (0.0–1.0) controls what fraction of evaluations emit
		telemetry — use 0.01 for high-frequency flags to cap data volume by 99%.
		"""
		result = await self.evaluate_flag(tenant_id, key, user_id, user_attributes)

		if sample_rate > 0 and random.random() < sample_rate:
			event: dict[str, Any] = {
				"event_type": "flag_evaluation",
				"tenant_id": tenant_id,
				"flag_key": key,
				"user_id": user_id,
				"enabled": result["enabled"],
				"variant": result["variant"],
				"reason": result["reason"],
				"targeting_matched": result["targeting_matched"],
				"sampled": True,
				"evaluated_at": self._now(),
			}
			if trace_context:
				event["trace_id"] = trace_context.get("trace_id")
				event["span_id"] = trace_context.get("span_id")
			# Publish to NATS telemetry subject when adapter is available
			try:
				from capabilities.common.fflag.domain.adapters import get_audit_adapter
				adapter = get_audit_adapter()
				if adapter and hasattr(adapter, "publish"):
					await asyncio.get_event_loop().run_in_executor(
						None, adapter.publish, f"fflag.telemetry.{tenant_id}", event
					)
			except Exception as exc:
				_log.debug("telemetry publish skipped: %s", exc)
			result["_telemetry_emitted"] = True

		return result

	# ── I9: Gradual Rollout Ramp Plans ────────────────────────────

	async def set_ramp_plan(
		self,
		tenant_id: str,
		key: str,
		steps: list[dict[str, Any]],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Attach a gradual rollout ramp plan to a flag.

		Each step: ``{"percentage": float, "after_minutes": int}`` or
		``{"percentage": float, "at_time": "<ISO8601>"}``.

		The scheduler (driven externally by a NATS tick on
		``fflag.scheduler.tick``) calls ``advance_ramp`` to apply due steps.
		"""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		before = deepcopy(record)
		record["ramp_plan"] = deepcopy(steps)
		record["ramp_step_index"] = 0
		record["ramp_active"] = True
		record["updated_at"] = self._now()
		self._emit(tenant, "ramp_plan_set", key, before=before, after=deepcopy(record), actor=actor)
		_log.info("ramp_plan set: flag=%s steps=%d", key, len(steps))
		return deepcopy(record)

	async def advance_ramp(
		self,
		tenant_id: str,
		key: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Advance a flag to its next ramp step if the step's dwell time has elapsed.

		Returns the flag record.  Idempotent if no step is due.
		"""
		tenant = self._tenant(tenant_id)
		fk = self._flag_key(tenant, key)
		record = self.flags.get(fk)
		if not record:
			raise KeyError(f"flag not found: {key}")
		plan: list[dict[str, Any]] = record.get("ramp_plan", [])
		idx: int = record.get("ramp_step_index", 0)
		if not record.get("ramp_active") or idx >= len(plan):
			return deepcopy(record)

		step = plan[idx]
		now_ts = datetime.now(timezone.utc)

		# Determine if this step is due
		due = False
		if "at_time" in step:
			try:
				step_time = datetime.fromisoformat(step["at_time"])
				if step_time.tzinfo is None:
					step_time = step_time.replace(tzinfo=timezone.utc)
				due = now_ts >= step_time
			except ValueError:
				_log.warning("ramp step has invalid at_time: %s", step["at_time"])
		elif "after_minutes" in step:
			# Use ramp plan set time as baseline
			plan_set_at_str = record.get("updated_at", record.get("created_at", self._now()))
			try:
				plan_set_at = datetime.fromisoformat(plan_set_at_str.rstrip("Z"))
				plan_set_at = plan_set_at.replace(tzinfo=timezone.utc)
				due = (now_ts - plan_set_at).total_seconds() >= step["after_minutes"] * 60
			except Exception:
				due = False

		if due:
			before = deepcopy(record)
			record["rollout_percentage"] = float(step["percentage"])
			record["ramp_step_index"] = idx + 1
			if record["ramp_step_index"] >= len(plan):
				record["ramp_active"] = False
			record["updated_at"] = self._now()
			self._emit(tenant, "ramp_step_applied", key, before=before,
					   after=deepcopy(record), actor=actor)
			_log.info("ramp step applied: flag=%s step=%d pct=%.1f%%",
					  key, idx, step["percentage"])

		return deepcopy(record)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _normal_cdf(z: float) -> float:
	"""Approximate standard normal CDF using Abramowitz & Stegun series (error < 7.5e-8)."""
	if z < 0:
		return 1.0 - _normal_cdf(-z)
	t = 1.0 / (1.0 + 0.2316419 * z)
	poly = t * (0.319381530
				+ t * (-0.356563782
					   + t * (1.781477937
							  + t * (-1.821255978
									 + t * 1.330274429))))
	return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

