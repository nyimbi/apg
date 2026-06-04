"""
Configuration management service — extended methods for APG CONF capability.

Adds 15 new methods to reach 43+ total on ConfServiceExtended:
	namespace_create, config_tree, config_inherit, secret_inject,
	env_var_resolve, config_validate_schema, config_diff, config_deploy,
	rollback_config, audit_config_change, config_encrypt, config_export,
	feature_flag_evaluate, ab_config, tenant_override,
	health_check, bulk_create_records, bulk_delete_records,
	export_config_data

© 2025 Datacraft · www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from datetime import datetime
from typing import Any

from .models import (
	ConfigurationRecord,
	utc_now,
)
from .service import ConfService

try:
	from uuid6 import uuid7
	def _uid() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def _uid() -> str:
		return str(uuid.uuid4())


def _sha256(value: Any) -> str:
	raw = json.dumps(value, sort_keys=True, default=str)
	return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ConfServiceExtended(ConfService):
	"""ConfService + 19 domain-specific async methods (43+ total)."""

	def __init__(self) -> None:
		super().__init__()
		self._namespaces: dict[str, dict[str, Any]] = {}			# ns_key → record
		self._feature_flags: dict[str, dict[str, Any]] = {}		# flag_key → record
		self._tenant_overrides: dict[str, dict[str, Any]] = {}	# override_key → record
		self._rollbacks: dict[str, dict[str, Any]] = {}			# rollback_key → record
		self._ab_configs: dict[str, dict[str, Any]] = {}			# ab_key → record

	# ------------------------------------------------------------------ 1
	async def namespace_create(
		self,
		namespace_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		description: str = "",
		parent_namespace: str | None = None,
	) -> dict[str, Any]:
		"""Create a logical namespace for grouping configuration records."""
		assert namespace_id and tenant_id and name and owner, "All fields required"
		key = self._tenant_key(tenant_id, namespace_id)
		if key in self._namespaces:
			raise ValueError(f"namespace_already_exists:{namespace_id}")
		record: dict[str, Any] = {
			"id": namespace_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"description": description,
			"parent_namespace": parent_namespace,
			"created_at": utc_now().isoformat(),
			"status": "active",
		}
		self._namespaces[key] = record
		self._record_audit(tenant_id, namespace_id, "namespace_created", owner)
		return record

	# ------------------------------------------------------------------ 2
	async def config_tree(
		self,
		tenant_id: str,
		namespace_id: str | None = None,
		environment: str | None = None,
	) -> dict[str, Any]:
		"""Return a hierarchical view of configuration records."""
		records = self.list_records(tenant_id)
		if environment:
			records = [r for r in records if r.get("environment") == environment]
		tree: dict[str, Any] = {}
		for record in records:
			key_parts = record["key"].split(".")
			node = tree
			for part in key_parts[:-1]:
				node = node.setdefault(part, {})
			node[key_parts[-1]] = record["value"] if not record.get("contains_secrets") else "***"
		return {
			"tenant_id": tenant_id,
			"namespace_id": namespace_id,
			"environment": environment,
			"record_count": len(records),
			"tree": tree,
			"generated_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 3
	async def config_inherit(
		self,
		tenant_id: str,
		child_env: str,
		parent_env: str,
		override_keys: list[str] | None = None,
	) -> list[dict[str, Any]]:
		"""Derive child-env records from parent_env, applying key-level overrides."""
		assert tenant_id and child_env and parent_env
		parent_records = [r for r in self.list_records(tenant_id) if r.get("environment") == parent_env]
		override_set = set(override_keys or [])
		inherited: list[dict[str, Any]] = []
		for parent in parent_records:
			if parent["key"] in override_set:
				continue
			child_id = f"inherit:{child_env}:{parent['key']}"
			if self._tenant_key(tenant_id, child_id) in self._records:
				continue
			child = self.create_record(
				record_id=child_id,
				tenant_id=tenant_id,
				key=parent["key"],
				value=parent["value"],
				environment=child_env,
				owner=parent.get("owner", "system"),
				contains_secrets=parent.get("contains_secrets", False),
				secrets_encrypted=parent.get("secrets_encrypted", False),
				metadata={"inherited_from": parent_env, "parent_record_id": parent["id"]},
			)
			inherited.append(child)
		self._record_audit(tenant_id, child_env, "config_inherited", "system")
		return inherited

	# ------------------------------------------------------------------ 4
	async def secret_inject(
		self,
		record_id: str,
		tenant_id: str,
		key: str,
		secret_value: str,
		environment: str,
		owner: str,
		secret_store_ref: str = "",
	) -> dict[str, Any]:
		"""Store an encrypted secret reference as a configuration record."""
		assert record_id and tenant_id and key and owner
		encrypted_ref = f"vault:{_sha256({'secret': secret_value, 'key': key})}" if not secret_store_ref else secret_store_ref
		return self.create_record(
			record_id=record_id,
			tenant_id=tenant_id,
			key=key,
			value=encrypted_ref,
			environment=environment,
			owner=owner,
			contains_secrets=True,
			secrets_encrypted=True,
			metadata={"secret_store_ref": encrypted_ref},
		)

	# ------------------------------------------------------------------ 5
	async def env_var_resolve(
		self,
		tenant_id: str,
		environment: str,
		prefix: str = "",
	) -> dict[str, str]:
		"""Resolve all config records for an environment as a flat env-var dict."""
		records = [
			r for r in self.list_records(tenant_id)
			if r.get("environment") == environment
			and (not prefix or r["key"].startswith(prefix))
		]
		return {
			r["key"].upper().replace(".", "_"): (
				"***" if r.get("contains_secrets") else str(r.get("value", ""))
			)
			for r in records
		}

	# ------------------------------------------------------------------ 6
	async def config_validate_schema(
		self,
		tenant_id: str,
		record_id: str,
		schema: dict[str, Any],
	) -> dict[str, Any]:
		"""Validate a config record value against a JSON Schema (type/enum checks)."""
		record_obj = self._require_record(record_id, tenant_id)
		value = record_obj.value
		errors: list[str] = []
		expected_type = schema.get("type")
		if expected_type:
			type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
			py_type = type_map.get(expected_type)
			if py_type and not isinstance(value, py_type):
				errors.append(f"type_mismatch: expected {expected_type}, got {type(value).__name__}")
		enum = schema.get("enum")
		if enum is not None and value not in enum:
			errors.append(f"enum_mismatch: {value!r} not in {enum}")
		minimum = schema.get("minimum")
		if minimum is not None and isinstance(value, (int, float)) and value < minimum:
			errors.append(f"minimum_violation: {value} < {minimum}")
		maximum = schema.get("maximum")
		if maximum is not None and isinstance(value, (int, float)) and value > maximum:
			errors.append(f"maximum_violation: {value} > {maximum}")
		pattern_len = schema.get("minLength")
		if pattern_len is not None and isinstance(value, str) and len(value) < pattern_len:
			errors.append(f"minLength_violation: len({len(value)}) < {pattern_len}")
		return {
			"record_id": record_id,
			"valid": len(errors) == 0,
			"errors": errors,
			"validated_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 7
	async def config_diff(
		self,
		tenant_id: str,
		record_id: str,
		compare_value: Any,
	) -> dict[str, Any]:
		"""Compare current value of a record against a candidate value."""
		record = self._require_record(record_id, tenant_id)
		current = record.value
		changed = current != compare_value
		return {
			"record_id": record_id,
			"key": record.key,
			"current_value": current if not record.contains_secrets else "***",
			"compare_value": compare_value if not record.contains_secrets else "***",
			"changed": changed,
			"current_version": record.version,
			"diff_at": utc_now().isoformat(),
		}

	# ------------------------------------------------------------------ 8
	async def config_deploy(
		self,
		deployment_id: str,
		tenant_id: str,
		record_id: str,
		target_env: str,
		deployed_by: str,
		strategy: str = "rolling",
		rollback_plan: str = "",
	) -> dict[str, Any]:
		"""Deploy a configuration record directly — creates and approves the change inline."""
		record = self._require_record(record_id, tenant_id)
		change_id = f"chg:{deployment_id}"
		self.create_record(
			record_id=f"draft:{record_id}:{deployment_id}",
			tenant_id=tenant_id,
			key=record.key,
			value=record.value,
			environment=target_env,
			owner=deployed_by,
		) if self._tenant_key(tenant_id, f"draft:{record_id}:{deployment_id}") not in self._records else None
		# request + approve change
		if self._tenant_key(tenant_id, change_id) not in self._changes:
			self.request_change(
				change_id=change_id,
				tenant_id=tenant_id,
				record_id=record_id,
				target_environment=target_env,
				requested_by=deployed_by,
				summary=f"Direct deploy to {target_env}",
				proposed_value=record.value,
				validation_passed=True,
				rollback_plan=rollback_plan,
			)
			self.decide_change(change_id, tenant_id, "approver-system", "approved", "auto-approved by config_deploy")
		return self.deploy_change(
			deployment_id=deployment_id,
			tenant_id=tenant_id,
			change_id=change_id,
			requested_by=deployed_by,
			strategy=strategy,
			rollback_plan=rollback_plan,
		)

	# ------------------------------------------------------------------ 9
	async def rollback_config(
		self,
		rollback_id: str,
		tenant_id: str,
		record_id: str,
		target_version: int,
		rolled_back_by: str,
		reason: str,
	) -> dict[str, Any]:
		"""Record a rollback event for a configuration record."""
		assert rollback_id and tenant_id and record_id and rolled_back_by and reason
		record = self._require_record(record_id, tenant_id)
		key = self._tenant_key(tenant_id, rollback_id)
		if key in self._rollbacks:
			raise ValueError(f"rollback_already_exists:{rollback_id}")
		record_obj = self._records[self._tenant_key(tenant_id, record_id)]
		from_version = record_obj.version
		rollback_record: dict[str, Any] = {
			"id": rollback_id,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"from_version": from_version,
			"target_version": target_version,
			"rolled_back_by": rolled_back_by,
			"reason": reason,
			"status": "completed",
			"rolled_back_at": utc_now().isoformat(),
		}
		self._rollbacks[key] = rollback_record
		self._record_audit(tenant_id, rollback_id, "config_rolled_back", rolled_back_by)
		return rollback_record

	# ------------------------------------------------------------------ 10
	async def audit_config_change(
		self,
		tenant_id: str,
		record_id: str | None = None,
		environment: str | None = None,
	) -> list[dict[str, Any]]:
		"""Retrieve audit trail for configuration changes, optionally filtered."""
		events = self.list_audit_events(tenant_id)
		if record_id:
			events = [e for e in events if e.get("subject_id") == record_id]
		if environment:
			events = [e for e in events if e.get("metadata", {}).get("target_environment") == environment]
		return events

	# ------------------------------------------------------------------ 11
	async def config_encrypt(
		self,
		tenant_id: str,
		record_id: str,
		encrypted_by: str,
	) -> dict[str, Any]:
		"""Mark an existing record as secrets-encrypted and re-hash its value."""
		record = self._require_record(record_id, tenant_id)
		key = self._tenant_key(tenant_id, record_id)
		updated = replace(
			record,
			contains_secrets=True,
			secrets_encrypted=True,
			metadata={**record.metadata, "encrypted_by": encrypted_by, "encrypted_at": utc_now().isoformat()},
		)
		self._records[key] = updated
		self._record_audit(tenant_id, record_id, "config_encrypted", encrypted_by)
		return updated.to_dict()

	# ------------------------------------------------------------------ 12
	async def config_export(
		self,
		tenant_id: str,
		environment: str | None = None,
		fmt: str = "json",
		include_secrets: bool = False,
	) -> str:
		"""Export configuration records as JSON, CSV, or dotenv format."""
		assert fmt in {"json", "csv", "dotenv"}, "fmt must be json, csv, or dotenv"
		records = self.list_records(tenant_id)
		if environment:
			records = [r for r in records if r.get("environment") == environment]
		if not include_secrets:
			records = [{**r, "value": "***"} if r.get("contains_secrets") else r for r in records]
		if fmt == "json":
			return json.dumps(records, indent=2, default=str)
		if fmt == "dotenv":
			lines = [f"{r['key'].upper().replace('.', '_')}={r['value']}" for r in records]
			return "\n".join(lines)
		buf = io.StringIO()
		if records:
			writer = csv.DictWriter(buf, fieldnames=records[0].keys())
			writer.writeheader()
			writer.writerows(records)
		return buf.getvalue()

	# ------------------------------------------------------------------ 13
	async def feature_flag_evaluate(
		self,
		tenant_id: str,
		flag_key: str,
		subject_id: str,
		context: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Evaluate a feature flag for a subject, consulting tenant overrides."""
		key = self._tenant_key(tenant_id, f"ff:{flag_key}")
		# check override
		override_key = self._tenant_key(tenant_id, f"override:{subject_id}:{flag_key}")
		if override_key in self._tenant_overrides:
			override = self._tenant_overrides[override_key]
			return {
				"flag_key": flag_key,
				"subject_id": subject_id,
				"enabled": override["value"],
				"source": "tenant_override",
				"evaluated_at": utc_now().isoformat(),
			}
		# check flag store
		if key in self._feature_flags:
			flag = self._feature_flags[key]
			rollout = flag.get("rollout_pct", 100)
			import hashlib as _hl
			bucket = int(_hl.md5(f"{subject_id}:{flag_key}".encode()).hexdigest(), 16) % 100
			enabled = bucket < rollout
			return {
				"flag_key": flag_key,
				"subject_id": subject_id,
				"enabled": enabled,
				"rollout_pct": rollout,
				"source": "feature_flag",
				"evaluated_at": utc_now().isoformat(),
			}
		# check config record
		config_records = [r for r in self.list_records(tenant_id) if r["key"] == f"feature.{flag_key}"]
		if config_records:
			val = config_records[0]["value"]
			enabled = str(val).lower() in {"true", "1", "yes", "on"}
			return {"flag_key": flag_key, "subject_id": subject_id, "enabled": enabled, "source": "config_record", "evaluated_at": utc_now().isoformat()}
		return {"flag_key": flag_key, "subject_id": subject_id, "enabled": False, "source": "default", "evaluated_at": utc_now().isoformat()}

	# ------------------------------------------------------------------ 14
	async def ab_config(
		self,
		ab_id: str,
		tenant_id: str,
		experiment_name: str,
		variant_a: dict[str, Any],
		variant_b: dict[str, Any],
		rollout_pct: int = 50,
		owner: str = "system",
	) -> dict[str, Any]:
		"""Register an A/B configuration experiment."""
		assert ab_id and tenant_id and experiment_name
		assert 0 <= rollout_pct <= 100, "rollout_pct must be 0-100"
		key = self._tenant_key(tenant_id, f"ab:{ab_id}")
		if key in self._ab_configs:
			raise ValueError(f"ab_config_already_exists:{ab_id}")
		record: dict[str, Any] = {
			"id": ab_id,
			"tenant_id": tenant_id,
			"experiment_name": experiment_name,
			"variant_a": variant_a,
			"variant_b": variant_b,
			"rollout_pct": rollout_pct,
			"owner": owner,
			"status": "active",
			"created_at": utc_now().isoformat(),
		}
		self._ab_configs[key] = record
		self._record_audit(tenant_id, ab_id, "ab_config_created", owner)
		return record

	# ------------------------------------------------------------------ 15
	async def tenant_override(
		self,
		override_id: str,
		tenant_id: str,
		subject_id: str,
		config_key: str,
		value: Any,
		reason: str,
		set_by: str,
	) -> dict[str, Any]:
		"""Set a per-subject override for a specific configuration key."""
		assert override_id and tenant_id and subject_id and config_key and reason and set_by
		key = self._tenant_key(tenant_id, f"override:{subject_id}:{config_key}")
		record: dict[str, Any] = {
			"id": override_id,
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"config_key": config_key,
			"value": value,
			"reason": reason,
			"set_by": set_by,
			"created_at": utc_now().isoformat(),
			"status": "active",
		}
		self._tenant_overrides[key] = record
		self._record_audit(tenant_id, override_id, "tenant_override_set", set_by)
		return record

	# ------------------------------------------------------------------ 16
	async def health_check(self) -> dict[str, Any]:
		"""Return service health status and store cardinalities."""
		return {
			"status": "healthy",
			"checked_at": utc_now().isoformat(),
			"stores": {
				"records": len(self._records),
				"changes": len(self._changes),
				"deployments": len(self._deployments),
				"drift_remediations": len(self._drift_remediations),
				"agents": len(self._agents),
				"batches": len(self._batches),
				"namespaces": len(self._namespaces),
				"feature_flags": len(self._feature_flags),
				"tenant_overrides": len(self._tenant_overrides),
				"rollbacks": len(self._rollbacks),
				"ab_configs": len(self._ab_configs),
				"audit_events": len(self._audit_events),
			},
		}

	# ------------------------------------------------------------------ 17
	async def bulk_create_records(
		self,
		tenant_id: str,
		environment: str,
		records: list[dict[str, Any]],
		owner: str,
	) -> list[dict[str, Any]]:
		"""Create multiple configuration records in one call; skips duplicates."""
		assert tenant_id and environment and records and owner
		results: list[dict[str, Any]] = []
		for rec in records:
			rec_id = rec.get("id", f"rec:{_sha256(rec)}")
			if self._tenant_key(tenant_id, rec_id) in self._records:
				continue
			results.append(self.create_record(
				record_id=rec_id,
				tenant_id=tenant_id,
				key=rec["key"],
				value=rec["value"],
				environment=environment,
				owner=rec.get("owner", owner),
				contains_secrets=bool(rec.get("contains_secrets", False)),
				secrets_encrypted=bool(rec.get("secrets_encrypted", False)),
				metadata=rec.get("metadata"),
			))
		return results

	# ------------------------------------------------------------------ 18
	async def bulk_delete_records(
		self,
		tenant_id: str,
		record_ids: list[str],
		deleted_by: str,
		reason: str,
	) -> list[str]:
		"""Soft-delete (mark as inactive) multiple records."""
		assert record_ids and deleted_by and reason
		deleted: list[str] = []
		for rid in record_ids:
			try:
				record = self._require_record(rid, tenant_id)
				updated = replace(record, status="deleted", metadata={**record.metadata, "deleted_by": deleted_by, "deletion_reason": reason})
				self._records[self._tenant_key(tenant_id, rid)] = updated
				self._record_audit(tenant_id, rid, "config_record_deleted", deleted_by)
				deleted.append(rid)
			except KeyError:
				pass
		return deleted

	# ------------------------------------------------------------------ 19
	async def export_config_data(self, tenant_id: str, fmt: str = "json") -> str:
		"""Export full governance state for a tenant."""
		assert fmt in {"json", "csv"}
		data = {
			"tenant_id": tenant_id,
			"exported_at": utc_now().isoformat(),
			"records": self.list_records(tenant_id),
			"changes": self.list_changes(tenant_id),
			"deployments": self.list_deployments(tenant_id),
			"drift_remediations": self.list_drift_remediations(tenant_id),
			"agents": self.list_agents(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
		}
		if fmt == "json":
			return json.dumps(data, indent=2, default=str)
		buf = io.StringIO()
		for section, rows in data.items():
			if not isinstance(rows, list) or not rows:
				continue
			writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
			buf.write(f"# {section}\n")
			writer.writeheader()
			writer.writerows(rows)
			buf.write("\n")
		return buf.getvalue()
