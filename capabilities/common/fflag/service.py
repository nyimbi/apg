"""Feature Flags service — runtime toggles, percentage rollout, A/B experiments, per-tenant targeting."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from copy import deepcopy
from datetime import datetime
from typing import Any
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

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.flags: dict[str, dict[str, Any]] = {}  # keyed by "tenant:flag_key"
		self.experiments: dict[str, dict[str, Any]] = {}
		self.assignments: dict[str, dict[str, Any]] = {}  # user experiment assignments
		self.overrides: dict[str, dict[str, Any]] = {}  # per-user flag overrides
		self._audit_events: list[dict[str, Any]] = []

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
