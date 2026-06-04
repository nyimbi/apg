"""APG Environment Management Service — expanded async runtime (42+ methods).

All state lives in _Store (in-memory). Every mutation emits an audit event.
Notifications sent on lifecycle transitions.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:
		return str(uuid.uuid4())

import logging

logger = logging.getLogger(__name__)

VALID_STAGES: set[str] = {"development", "test", "staging", "production"}
SUPPORTED_CHANNELS: set[str] = {"email", "sms", "webhook", "audit_log"}


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Store / Audit / Notify
# ---------------------------------------------------------------------------

class _Store:
	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
		self._data.setdefault(collection, {})[record["id"]] = record
		return record

	async def get(self, collection: str, record_id: str) -> dict[str, Any] | None:
		return self._data.get(collection, {}).get(record_id)

	async def list(self, collection: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(self._data.get(collection, {}).values())
		if tenant_id is not None:
			items = [i for i in items if i.get("tenant_id") == tenant_id]
		return sorted(items, key=lambda i: i.get("id", ""))

	async def delete(self, collection: str, record_id: str) -> bool:
		bucket = self._data.get(collection, {})
		if record_id in bucket:
			del bucket[record_id]
			return True
		return False


class _Audit:
	def __init__(self, store: _Store) -> None:
		self._store = store

	async def log_event(
		self,
		event_type: str,
		actor_id: str,
		tenant_id: str,
		subject_id: str,
		details: dict[str, Any] | None = None,
		severity: str = "info",
	) -> dict[str, Any]:
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"actor_id": actor_id,
			"subject_id": subject_id,
			"severity": severity,
			"details": details or {},
			"recorded_at": _utc_now(),
		}
		await self._store.put("envm_audit", record)
		return record


class _Notify:
	async def send(self, recipient: str, channel: str, subject: str, body: str) -> dict[str, Any]:
		if channel not in SUPPORTED_CHANNELS:
			raise ValueError(f"unsupported_channel:{channel}")
		return {
			"id": uuid7str(), "recipient": recipient, "channel": channel,
			"subject": subject, "body": body, "sent_at": _utc_now(),
		}


# ---------------------------------------------------------------------------
# EnvironmentManagementService
# ---------------------------------------------------------------------------

class EnvironmentManagementService:
	"""Async environment lifecycle management service — 42+ methods."""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id
		self._store = _Store()
		self._audit = _Audit(self._store)
		self._notify = _Notify()

	# ------------------------------------------------------------------
	# 1. env_create
	# ------------------------------------------------------------------
	async def env_create(
		self,
		tenant_id: str,
		env_id: str,
		name: str,
		stage: str,
		region: str,
		cloud_provider: str,
		owner: str,
		config: dict[str, Any] | None = None,
		rbac_policy: str = "default-rbac",
		secret_scope_policy: str = "default-secrets",
	) -> dict[str, Any]:
		"""Create and register a new environment."""
		stage = stage.lower()
		if stage not in VALID_STAGES:
			raise ValueError(f"invalid_stage:{stage}")
		fingerprint = hashlib.sha256(
			f"{tenant_id}:{env_id}:{stage}:{region}:{cloud_provider}".encode()
		).hexdigest()
		record = {
			"id": env_id,
			"tenant_id": tenant_id,
			"name": name,
			"stage": stage,
			"region": region,
			"cloud_provider": cloud_provider,
			"owner": owner,
			"config": json.dumps(config or {}, default=str),
			"rbac_policy": rbac_policy,
			"secret_scope_policy": secret_scope_policy,
			"fingerprint": fingerprint,
			"status": "active",
			"production_locked": stage == "production",
			"created_at": _utc_now(),
			"updated_at": _utc_now(),
		}
		await self._store.put("envm_environments", record)
		await self._audit.log_event("env_created", self.actor_id, tenant_id, env_id, {"stage": stage, "region": region})
		return record

	# ------------------------------------------------------------------
	# 2. env_clone
	# ------------------------------------------------------------------
	async def env_clone(
		self,
		tenant_id: str,
		source_env_id: str,
		target_name: str,
		target_stage: str | None = None,
		override_config: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Clone an environment with optional config overlay."""
		source = await self._require_env(tenant_id, source_env_id)
		resolved_stage = target_stage or source["stage"]
		if resolved_stage not in VALID_STAGES:
			raise ValueError(f"invalid_stage:{resolved_stage}")
		base_cfg: dict[str, Any] = {}
		try:
			base_cfg = json.loads(source["config"])
		except Exception:
			base_cfg = {"source_ref": source["config"]}
		merged = {**base_cfg, **(override_config or {})}
		new_id = f"{source_env_id}-clone-{uuid7str()[:8]}"
		clone = await self.env_create(
			tenant_id, new_id, target_name, resolved_stage,
			source["region"], source["cloud_provider"], source["owner"],
			merged, source["rbac_policy"], source["secret_scope_policy"],
		)
		await self._audit.log_event("env_cloned", self.actor_id, tenant_id, new_id, {"source": source_env_id})
		return {**clone, "cloned_from": source_env_id}

	# ------------------------------------------------------------------
	# 3. env_compare
	# ------------------------------------------------------------------
	async def env_compare(
		self,
		tenant_id: str,
		env1_id: str,
		env2_id: str,
	) -> dict[str, Any]:
		"""Return a structured diff between two environments."""
		env1 = await self._require_env(tenant_id, env1_id)
		env2 = await self._require_env(tenant_id, env2_id)

		def _cfg(src: str) -> dict[str, Any]:
			try:
				return json.loads(src)
			except Exception:
				return {"raw": src}

		cfg1, cfg2 = _cfg(env1["config"]), _cfg(env2["config"])
		all_keys = set(cfg1) | set(cfg2)
		config_diff = {k: {"env1": cfg1.get(k, "__missing__"), "env2": cfg2.get(k, "__missing__")}
					   for k in sorted(all_keys) if cfg1.get(k) != cfg2.get(k)}
		diffs = [f for f in ("stage", "region", "rbac_policy", "secret_scope_policy") if env1.get(f) != env2.get(f)]
		if env1.get("fingerprint") != env2.get("fingerprint"):
			diffs.append("fingerprint")
		if config_diff:
			diffs.append("configuration")
		return {
			"env1_id": env1_id, "env2_id": env2_id, "tenant_id": tenant_id,
			"identical": len(diffs) == 0, "differences": diffs, "config_diff": config_diff,
			"compared_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 4. env_promote
	# ------------------------------------------------------------------
	async def env_promote(
		self,
		tenant_id: str,
		source_env_id: str,
		target_stage: str,
		approved_by: str,
		artifact_ref: str,
	) -> dict[str, Any]:
		"""Promote an environment to the next stage."""
		source = await self._require_env(tenant_id, source_env_id)
		if target_stage not in VALID_STAGES:
			raise ValueError(f"invalid_target_stage:{target_stage}")
		if not approved_by:
			raise PermissionError("promotion_approval_required")
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"source_env_id": source_env_id,
			"source_stage": source["stage"],
			"target_stage": target_stage,
			"artifact_ref": artifact_ref,
			"approved_by": approved_by,
			"status": "promoted",
			"promoted_at": _utc_now(),
		}
		await self._store.put("envm_promotions", record)
		await self._audit.log_event("env_promoted", self.actor_id, tenant_id, record["id"], {"from": source["stage"], "to": target_stage}, severity="medium")
		await self._notify.send(approved_by, "audit_log", "Environment promoted", f"Env {source_env_id} promoted to {target_stage}")
		return record

	# ------------------------------------------------------------------
	# 5. env_snapshot
	# ------------------------------------------------------------------
	async def env_snapshot(
		self,
		tenant_id: str,
		env_id: str,
		snapshot_label: str = "",
	) -> dict[str, Any]:
		"""Capture a point-in-time snapshot of an environment's configuration."""
		env = await self._require_env(tenant_id, env_id)
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"label": snapshot_label or f"snap-{_utc_now()}",
			"snapshot_data": dict(env),
			"created_at": _utc_now(),
		}
		await self._store.put("envm_snapshots", record)
		await self._audit.log_event("env_snapshot_created", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "label": record["label"]})
		return {k: v for k, v in record.items() if k != "snapshot_data"} | {"snapshot_data_keys": list(env.keys())}

	# ------------------------------------------------------------------
	# 6. config_drift_check
	# ------------------------------------------------------------------
	async def config_drift_check(
		self,
		tenant_id: str,
		env_id: str,
		declared: dict[str, Any],
		observed: dict[str, Any],
	) -> dict[str, Any]:
		"""Detect configuration drift between declared and observed state."""
		all_keys = set(declared) | set(observed)
		drifted = {k: {"declared": declared.get(k), "observed": observed.get(k)}
				   for k in sorted(all_keys) if declared.get(k) != observed.get(k)}
		drift_pct = round(len(drifted) / max(len(all_keys), 1) * 100, 2)
		status = "compliant" if not drifted else ("review_required" if drift_pct > 20 else "drifted")
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"drift_percent": drift_pct,
			"drifted_keys": drifted,
			"drifted_key_count": len(drifted),
			"total_keys": len(all_keys),
			"status": status,
			"checked_at": _utc_now(),
		}
		await self._store.put("envm_drift_reports", record)
		await self._audit.log_event("config_drift_checked", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "drift_percent": drift_pct})
		if status == "review_required":
			await self._notify.send(self.actor_id, "audit_log", "Config drift review required", f"Env {env_id} drift: {drift_pct}%")
		return record

	# ------------------------------------------------------------------
	# 7. secret_rotation
	# ------------------------------------------------------------------
	async def secret_rotation(
		self,
		tenant_id: str,
		env_id: str,
		secret_name: str,
		new_vault_path: str,
		rotated_by: str,
	) -> dict[str, Any]:
		"""Rotate a secret reference in an environment."""
		await self._require_env(tenant_id, env_id)
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"secret_name": secret_name,
			"new_vault_path": new_vault_path,
			"rotated_by": rotated_by,
			"status": "rotated",
			"rotated_at": _utc_now(),
		}
		await self._store.put("envm_secret_rotations", record)
		await self._audit.log_event("secret_rotated", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "secret_name": secret_name}, severity="medium")
		await self._notify.send(rotated_by, "audit_log", "Secret rotated", f"Secret {secret_name} rotated in env {env_id}")
		return record

	# ------------------------------------------------------------------
	# 8. env_cost_track
	# ------------------------------------------------------------------
	async def env_cost_track(
		self,
		tenant_id: str,
		env_id: str,
		period: str,
		resource_costs: dict[str, float],
		currency: str = "USD",
	) -> dict[str, Any]:
		"""Record resource cost data for an environment."""
		await self._require_env(tenant_id, env_id)
		total = round(sum(resource_costs.values()), 4)
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"period": period,
			"currency": currency,
			"resource_costs": resource_costs,
			"total_cost": total,
			"recorded_at": _utc_now(),
		}
		await self._store.put("envm_costs", record)
		await self._audit.log_event("cost_recorded", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "period": period, "total": total})
		return record

	# ------------------------------------------------------------------
	# 9. env_lifecycle
	# ------------------------------------------------------------------
	async def env_lifecycle(
		self,
		tenant_id: str,
		env_id: str,
		action: str,
		actor: str,
		reason: str = "",
	) -> dict[str, Any]:
		"""Manage environment lifecycle: provision | deprovision | suspend | resume."""
		valid_actions = {"provision", "deprovision", "suspend", "resume"}
		if action not in valid_actions:
			raise ValueError(f"invalid_lifecycle_action:{action}")
		env = await self._require_env(tenant_id, env_id)
		if action == "deprovision" and env["stage"] == "production" and not reason:
			raise PermissionError("production_deprovision_requires_reason")
		status_map = {"provision": "provisioned", "deprovision": "deprovisioned", "suspend": "suspended", "resume": "active"}
		env["status"] = status_map[action]
		env["updated_at"] = _utc_now()
		await self._store.put("envm_environments", env)
		record = {
			"id": uuid7str(), "tenant_id": tenant_id, "env_id": env_id,
			"action": action, "actor": actor, "reason": reason,
			"new_status": env["status"], "performed_at": _utc_now(),
		}
		await self._store.put("envm_lifecycle_events", record)
		await self._audit.log_event(f"env_{action}", self.actor_id, tenant_id, env_id, {"actor": actor, "reason": reason}, severity="medium")
		await self._notify.send(actor, "audit_log", f"Env {action}", f"Env {env_id} {action} by {actor}")
		return record

	# ------------------------------------------------------------------
	# 10. compliance_check_env
	# ------------------------------------------------------------------
	async def compliance_check_env(
		self,
		tenant_id: str,
		env_id: str,
		framework: str = "CIS",
	) -> dict[str, Any]:
		"""Run a compliance check against a named framework."""
		env = await self._require_env(tenant_id, env_id)
		issues: list[str] = []
		if not env.get("rbac_policy"):
			issues.append("rbac_policy_missing")
		if not env.get("secret_scope_policy"):
			issues.append("secret_scope_policy_missing")
		if env["stage"] == "production" and env["status"] != "provisioned" and env["status"] != "active":
			issues.append("production_environment_not_active")
		drift_reports = await self._store.list("envm_drift_reports", tenant_id)
		env_drifts = [d for d in drift_reports if d["env_id"] == env_id and d["status"] == "review_required"]
		if env_drifts:
			issues.append(f"{len(env_drifts)}_unresolved_drift_reports")
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"framework": framework,
			"passed": len(issues) == 0,
			"issues": issues,
			"checked_at": _utc_now(),
		}
		await self._store.put("envm_compliance_checks", record)
		await self._audit.log_event("compliance_checked", self.actor_id, tenant_id, record["id"], {"framework": framework, "passed": record["passed"]})
		return record

	# ------------------------------------------------------------------
	# 11. env_access_audit
	# ------------------------------------------------------------------
	async def env_access_audit(
		self,
		tenant_id: str,
		env_id: str,
	) -> dict[str, Any]:
		"""Audit all access events recorded against an environment."""
		events = await self._store.list("envm_audit", tenant_id)
		env_events = [e for e in events if e.get("details", {}).get("env_id") == env_id or e.get("subject_id") == env_id]
		actors = list({e["actor_id"] for e in env_events})
		by_type: dict[str, int] = {}
		for e in env_events:
			by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
		return {
			"env_id": env_id,
			"tenant_id": tenant_id,
			"total_events": len(env_events),
			"unique_actors": actors,
			"events_by_type": by_type,
			"audited_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 12. env_export
	# ------------------------------------------------------------------
	async def env_export(
		self,
		tenant_id: str,
		env_id: str,
		fmt: str = "json",
	) -> str:
		"""Export environment configuration as JSON or CSV."""
		env = await self._require_env(tenant_id, env_id)
		if fmt == "json":
			return json.dumps(env, indent=2, default=str)
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=list(env.keys()))
		writer.writeheader()
		writer.writerow(env)
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 13. env_import
	# ------------------------------------------------------------------
	async def env_import(
		self,
		tenant_id: str,
		env_data: dict[str, Any],
		owner: str = "import",
	) -> dict[str, Any]:
		"""Import an environment from an exported record."""
		env_id = env_data.get("id") or uuid7str()
		return await self.env_create(
			tenant_id,
			env_id,
			name=str(env_data.get("name", "imported")),
			stage=str(env_data.get("stage", "development")),
			region=str(env_data.get("region", "unknown")),
			cloud_provider=str(env_data.get("cloud_provider", "unknown")),
			owner=owner,
			config=json.loads(env_data.get("config", "{}")),
			rbac_policy=str(env_data.get("rbac_policy", "default-rbac")),
			secret_scope_policy=str(env_data.get("secret_scope_policy", "default-secrets")),
		)

	# ------------------------------------------------------------------
	# 14. dependency_scan
	# ------------------------------------------------------------------
	async def dependency_scan(
		self,
		tenant_id: str,
		env_id: str,
		manifest: dict[str, str],
	) -> dict[str, Any]:
		"""Scan environment dependencies for known vulnerability patterns."""
		await self._require_env(tenant_id, env_id)
		# Stub: flag packages with 'beta', 'alpha', or 'rc' as risky
		risky = {k: v for k, v in manifest.items() if any(tag in v.lower() for tag in ("beta", "alpha", "rc", "dev"))}
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"total_dependencies": len(manifest),
			"risky_count": len(risky),
			"risky_packages": risky,
			"passed": len(risky) == 0,
			"scanned_at": _utc_now(),
		}
		await self._store.put("envm_dep_scans", record)
		await self._audit.log_event("dependency_scanned", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "risky_count": len(risky)})
		return record

	# ------------------------------------------------------------------
	# 15. env_analytics
	# ------------------------------------------------------------------
	async def env_analytics(self, tenant_id: str, period: str) -> dict[str, Any]:
		"""Aggregate environment analytics for a tenant."""
		envs = await self._store.list("envm_environments", tenant_id)
		costs = await self._store.list("envm_costs", tenant_id)
		drifts = await self._store.list("envm_drift_reports", tenant_id)
		period_costs = [c for c in costs if c["period"] == period]
		by_stage: dict[str, int] = {}
		for e in envs:
			by_stage[e["stage"]] = by_stage.get(e["stage"], 0) + 1
		drift_pcts = [d["drift_percent"] for d in drifts if d["env_id"] in {e["id"] for e in envs}]
		total_cost = round(sum(c["total_cost"] for c in period_costs), 4)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"environment_count": len(envs),
			"environments_by_stage": by_stage,
			"drift_report_count": len(drifts),
			"avg_drift_percent": round(statistics.mean(drift_pcts), 2) if drift_pcts else 0.0,
			"total_cost": total_cost,
			"cost_currency": "USD",
			"computed_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 16. health_check
	# ------------------------------------------------------------------
	async def health_check(self) -> dict[str, Any]:
		"""Return service health and collection sizes."""
		try:
			test_id = f"_health_{uuid7str()}"
			await self.env_create("_health", test_id, "_health_env", "development", "us-east-1", "aws", "system")
			await self._store.delete("envm_environments", test_id)
			status = "healthy"
		except Exception as exc:
			status = f"degraded:{exc}"
		return {
			"service": "EnvironmentManagementService",
			"status": status,
			"collections": {
				"environments": len(await self._store.list("envm_environments")),
				"promotions": len(await self._store.list("envm_promotions")),
				"drift_reports": len(await self._store.list("envm_drift_reports")),
				"audit_events": len(await self._store.list("envm_audit")),
			},
			"checked_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 17. dashboard_summary
	# ------------------------------------------------------------------
	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate KPI dashboard."""
		envs = await self._store.list("envm_environments", tenant_id)
		drifts = await self._store.list("envm_drift_reports", tenant_id)
		costs = await self._store.list("envm_costs", tenant_id)
		promotions = await self._store.list("envm_promotions", tenant_id)
		compliance = await self._store.list("envm_compliance_checks", tenant_id)
		return {
			"tenant_id": tenant_id,
			"total_environments": len(envs),
			"active_environments": sum(1 for e in envs if e["status"] == "active"),
			"production_environments": sum(1 for e in envs if e["stage"] == "production"),
			"deprovisioned_environments": sum(1 for e in envs if e["status"] == "deprovisioned"),
			"drift_reports": len(drifts),
			"drifted_environments": sum(1 for d in drifts if d["status"] != "compliant"),
			"total_promotions": len(promotions),
			"total_cost_recorded": round(sum(c["total_cost"] for c in costs), 4),
			"compliance_checks": len(compliance),
			"failed_compliance_checks": sum(1 for c in compliance if not c["passed"]),
			"audit_events": len(await self._store.list("envm_audit", tenant_id)),
			"generated_at": _utc_now(),
		}

	# ------------------------------------------------------------------
	# 18. register_environment (alias for env_create with named params)
	# ------------------------------------------------------------------
	async def register_environment(
		self,
		name: str,
		env_type: str,
		cloud_provider: str,
		region: str,
		config: dict[str, Any],
		tenant_id: str = "default",
		owner: str = "platform",
		stage: str = "development",
		rbac_policy: str = "envm-default-rbac",
		secret_scope_policy: str = "envm-default-secrets",
		approval_recorded: bool = True,
		status: str = "active",
		environment_id: str | None = None,
	) -> dict[str, Any]:
		"""Named-parameter environment registration (backward compat)."""
		env_id = environment_id or f"env-{uuid7str()[:8]}"
		merged_config = {**config, "env_type": env_type}
		rec = await self.env_create(tenant_id, env_id, name, stage, region, cloud_provider, owner, merged_config, rbac_policy, secret_scope_policy)
		if status != "active":
			rec["status"] = status
			await self._store.put("envm_environments", rec)
		return rec

	# ------------------------------------------------------------------
	# 19. provision_environment
	# ------------------------------------------------------------------
	async def provision_environment(
		self,
		env_id: str,
		template_id: str,
		approved_by: str,
		tenant_id: str = "default",
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Provision an environment from a template."""
		return await self.env_lifecycle(tenant_id, env_id, "provision", approved_by, f"template:{template_id}" + (" dry_run" if dry_run else ""))

	# ------------------------------------------------------------------
	# 20. deprovision_environment
	# ------------------------------------------------------------------
	async def deprovision_environment(
		self,
		env_id: str,
		reason: str,
		deprovisioned_by: str,
		tenant_id: str = "default",
		force: bool = False,
	) -> dict[str, Any]:
		"""Deprovision an environment."""
		return await self.env_lifecycle(tenant_id, env_id, "deprovision", deprovisioned_by, reason)

	# ------------------------------------------------------------------
	# 21. environment_health_check
	# ------------------------------------------------------------------
	async def environment_health_check(
		self,
		env_id: str,
		tenant_id: str = "default",
		checks: list[str] | None = None,
	) -> dict[str, Any]:
		"""Run synthetic health checks on an environment."""
		env = await self._require_env(tenant_id, env_id)
		checks_to_run = checks or ["connectivity", "secret_access", "rbac", "network", "config_sync"]
		results: dict[str, str] = {}
		for check in checks_to_run:
			if env["status"] in {"deprovisioned", "failed"}:
				results[check] = "fail"
			elif check == "config_sync" and env["status"] == "provisioning":
				results[check] = "pending"
			else:
				results[check] = "pass"
		overall = ("healthy" if all(v == "pass" for v in results.values())
				   else "degraded" if any(v == "pending" for v in results.values())
				   else "unhealthy")
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"checks": results,
			"overall": overall,
			"checked_at": _utc_now(),
		}
		await self._store.put("envm_health_checks", record)
		await self._audit.log_event("env_health_checked", self.actor_id, tenant_id, env_id, {"overall": overall})
		return record

	# ------------------------------------------------------------------
	# 22. configuration_drift_detection (sync-style compat)
	# ------------------------------------------------------------------
	async def configuration_drift_detection(
		self,
		env_id: str,
		tenant_id: str = "default",
		declared_version: str = "v1",
		observed_version: str = "v1",
		changed_items: int = 0,
		total_items: int = 1,
		drift_review_recorded: bool = False,
		remediation_action: str = "",
	) -> dict[str, Any]:
		"""Detect configuration drift from counts."""
		await self._require_env(tenant_id, env_id)
		drift_pct = round(changed_items / max(total_items, 1) * 100, 2)
		status = "compliant" if drift_pct == 0 else ("review_required" if drift_pct > 20 and not drift_review_recorded else "drifted")
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"declared_version": declared_version,
			"observed_version": observed_version,
			"drift_percent": drift_pct,
			"changed_items": changed_items,
			"total_items": total_items,
			"status": status,
			"drift_review_recorded": drift_review_recorded,
			"remediation_action": remediation_action or ("auto_remediate" if drift_pct < 20 else "manual_review"),
			"checked_at": _utc_now(),
		}
		await self._store.put("envm_drift_reports", record)
		await self._audit.log_event("drift_detected", self.actor_id, tenant_id, record["id"], {"env_id": env_id, "drift_percent": drift_pct})
		return record

	# ------------------------------------------------------------------
	# 23. secret_injection
	# ------------------------------------------------------------------
	async def secret_injection(
		self,
		env_id: str,
		secret_name: str,
		value: str,
		vault_path: str,
		tenant_id: str = "default",
		injected_by: str = "system",
		rotation_days: int = 90,
	) -> dict[str, Any]:
		"""Inject a secret reference into an environment (value NOT persisted)."""
		await self._require_env(tenant_id, env_id)
		record = {
			"id": uuid7str(),
			"tenant_id": tenant_id,
			"env_id": env_id,
			"secret_name": secret_name,
			"vault_path": vault_path,
			"rotation_days": rotation_days,
			"injected_by": injected_by,
			"injected_at": _utc_now(),
			"value_stored": False,
		}
		await self._store.put("envm_secrets", record)
		await self._audit.log_event("secret_injected", self.actor_id, tenant_id, record["id"], {"secret_name": secret_name, "vault_path": vault_path})
		return record

	# ------------------------------------------------------------------
	# 24. environment_clone (compat alias)
	# ------------------------------------------------------------------
	async def environment_clone(self, source_env_id: str, target_name: str, tenant_id: str = "default",
								cloned_by: str = "system", target_stage: str | None = None,
								override_config: dict[str, Any] | None = None) -> dict[str, Any]:
		return await self.env_clone(tenant_id, source_env_id, target_name, target_stage, override_config)

	# ------------------------------------------------------------------
	# 25. environment_comparison (compat alias)
	# ------------------------------------------------------------------
	async def environment_comparison(self, env1_id: str, env2_id: str, tenant_id: str = "default") -> dict[str, Any]:
		return await self.env_compare(tenant_id, env1_id, env2_id)

	# ------------------------------------------------------------------
	# 26. cost_tracking (compat alias)
	# ------------------------------------------------------------------
	async def cost_tracking(self, env_id: str, period: str, tenant_id: str = "default",
							resource_costs: dict[str, float] | None = None, currency: str = "USD",
							recorded_by: str = "system") -> dict[str, Any]:
		return await self.env_cost_track(tenant_id, env_id, period, resource_costs or {}, currency)

	# ------------------------------------------------------------------
	# 27. create_promotion_path
	# ------------------------------------------------------------------
	async def create_promotion_path(
		self,
		path_id: str,
		tenant_id: str,
		source_environment_id: str,
		target_environment_id: str,
		deployment_link: str,
		rollback_environment_id: str,
		approval_recorded: bool,
		promotion_path_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a promotion path between two environments."""
		await self._require_env(tenant_id, source_environment_id)
		await self._require_env(tenant_id, target_environment_id)
		if not deployment_link:
			raise PermissionError("deployment_link_required")
		record = {
			"id": path_id,
			"tenant_id": tenant_id,
			"source_environment_id": source_environment_id,
			"target_environment_id": target_environment_id,
			"deployment_link": deployment_link,
			"rollback_environment_id": rollback_environment_id,
			"approval_recorded": approval_recorded,
			"status": "active",
			"created_at": _utc_now(),
		}
		await self._store.put("envm_promotion_paths", record)
		await self._audit.log_event("promotion_path_created", self.actor_id, tenant_id, path_id)
		return record

	# ------------------------------------------------------------------
	# 28. run_promotion
	# ------------------------------------------------------------------
	async def run_promotion(
		self,
		run_id: str,
		tenant_id: str,
		promotion_path_id: str,
		requested_by: str,
		artifact_ref: str,
		approval_recorded: bool,
	) -> dict[str, Any]:
		"""Execute a promotion run."""
		path = await self._store.get("envm_promotion_paths", promotion_path_id)
		if path is None or path["tenant_id"] != tenant_id:
			raise KeyError(f"promotion_path_not_found:{promotion_path_id}")
		record = {
			"id": run_id,
			"tenant_id": tenant_id,
			"promotion_path_id": promotion_path_id,
			"requested_by": requested_by,
			"artifact_ref": artifact_ref,
			"approval_recorded": approval_recorded,
			"status": "promoted",
			"promoted_at": _utc_now(),
		}
		await self._store.put("envm_promotion_runs", record)
		await self._audit.log_event("env_promoted", self.actor_id, tenant_id, run_id, {"path_id": promotion_path_id}, severity="medium")
		return record

	# ------------------------------------------------------------------
	# 29. register_secret_scope
	# ------------------------------------------------------------------
	async def register_secret_scope(
		self,
		scope_id: str,
		tenant_id: str,
		environment_id: str,
		name: str,
		policy_ref: str,
		secret_refs: list[str] | tuple[str, ...],
		access_roles: list[str] | tuple[str, ...],
		secret_policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Register a secret scope for an environment."""
		await self._require_env(tenant_id, environment_id)
		record = {
			"id": scope_id,
			"tenant_id": tenant_id,
			"environment_id": environment_id,
			"name": name,
			"policy_ref": policy_ref,
			"secret_refs": list(secret_refs),
			"access_roles": list(access_roles),
			"status": "active",
			"created_at": _utc_now(),
		}
		await self._store.put("envm_secret_scopes", record)
		await self._audit.log_event("secret_scope_registered", self.actor_id, tenant_id, scope_id, {"env_id": environment_id})
		return record

	# ------------------------------------------------------------------
	# 30. register_envm_agent
	# ------------------------------------------------------------------
	async def register_envm_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		"""Register an environment management agent."""
		record = {
			"id": agent_id or uuid7str(),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": _normalize(runtime),
			"role": _normalize(role),
			"scope": scope,
			"contribution_disclosed": contribution_disclosed,
			"status": "active",
			"registered_at": _utc_now(),
		}
		await self._store.put("envm_agents", record)
		await self._audit.log_event("agent_registered", self.actor_id, tenant_id, record["id"], {"role": role})
		return record

	# ------------------------------------------------------------------
	# 31. bulk_create_environments
	# ------------------------------------------------------------------
	async def bulk_create_environments(
		self,
		tenant_id: str,
		environments: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Bulk-create multiple environments in parallel."""
		tasks = [
			self.env_create(
				tenant_id,
				e.get("id", uuid7str()),
				e["name"], e.get("stage", "development"), e.get("region", "us-east-1"),
				e.get("cloud_provider", "aws"), e.get("owner", "system"),
				e.get("config", {}),
			)
			for e in environments
		]
		results = await asyncio.gather(*tasks)
		await self._audit.log_event("bulk_envs_created", self.actor_id, tenant_id, "bulk", {"count": len(environments)})
		return list(results)

	# ------------------------------------------------------------------
	# 32. bulk_delete_environments
	# ------------------------------------------------------------------
	async def bulk_delete_environments(
		self,
		tenant_id: str,
		env_ids: list[str],
		reason: str = "bulk_delete",
	) -> list[dict[str, Any]]:
		"""Soft-delete multiple environments."""
		results = []
		for eid in env_ids:
			try:
				await self.env_lifecycle(tenant_id, eid, "deprovision", self.actor_id, reason)
				results.append({"env_id": eid, "status": "deprovisioned"})
			except Exception as exc:
				results.append({"env_id": eid, "status": "failed", "error": str(exc)})
		await self._audit.log_event("bulk_envs_deleted", self.actor_id, tenant_id, "bulk", {"count": len(env_ids)})
		return results

	# ------------------------------------------------------------------
	# 33. export_csv
	# ------------------------------------------------------------------
	async def export_csv(self, tenant_id: str, collection: str = "envm_environments") -> str:
		"""Export a collection to CSV."""
		records = await self._store.list(collection, tenant_id)
		if not records:
			return ""
		buf = io.StringIO()
		writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
		writer.writeheader()
		writer.writerows(records)
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 34. export_json
	# ------------------------------------------------------------------
	async def export_json(self, tenant_id: str, collection: str = "envm_environments") -> str:
		"""Export a collection to JSON."""
		records = await self._store.list(collection, tenant_id)
		return json.dumps(records, indent=2, default=str)

	# ------------------------------------------------------------------
	# 35. list_environments
	# ------------------------------------------------------------------
	async def list_environments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_environments", tenant_id)

	# ------------------------------------------------------------------
	# 36. list_drift_reports
	# ------------------------------------------------------------------
	async def list_drift_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_drift_reports", tenant_id)

	# ------------------------------------------------------------------
	# 37. list_promotions
	# ------------------------------------------------------------------
	async def list_promotions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_promotions", tenant_id)

	# ------------------------------------------------------------------
	# 38. list_promotion_paths
	# ------------------------------------------------------------------
	async def list_promotion_paths(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_promotion_paths", tenant_id)

	# ------------------------------------------------------------------
	# 39. list_promotion_runs
	# ------------------------------------------------------------------
	async def list_promotion_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_promotion_runs", tenant_id)

	# ------------------------------------------------------------------
	# 40. list_secret_scopes
	# ------------------------------------------------------------------
	async def list_secret_scopes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_secret_scopes", tenant_id)

	# ------------------------------------------------------------------
	# 41. list_audit_events
	# ------------------------------------------------------------------
	async def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_audit", tenant_id)

	# ------------------------------------------------------------------
	# 42. list_health_checks
	# ------------------------------------------------------------------
	async def list_health_checks(self, tenant_id: str | None = None, env_id: str | None = None) -> list[dict[str, Any]]:
		records = await self._store.list("envm_health_checks", tenant_id)
		if env_id:
			records = [r for r in records if r.get("env_id") == env_id]
		return records

	# ------------------------------------------------------------------
	# 43. list_envm_agents
	# ------------------------------------------------------------------
	async def list_envm_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return await self._store.list("envm_agents", tenant_id)

	# ------------------------------------------------------------------
	# 44. list_cost_records
	# ------------------------------------------------------------------
	async def list_cost_records(self, tenant_id: str | None = None, period: str | None = None) -> list[dict[str, Any]]:
		records = await self._store.list("envm_costs", tenant_id)
		if period:
			records = [r for r in records if r.get("period") == period]
		return records

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	async def _require_env(self, tenant_id: str, env_id: str) -> dict[str, Any]:
		rec = await self._store.get("envm_environments", env_id)
		if rec is None or rec["tenant_id"] != tenant_id:
			raise KeyError(f"environment_not_found:{env_id}")
		return rec


# Backward compat alias
EnvmService = EnvironmentManagementService

__all__ = ["EnvironmentManagementService", "EnvmService"]
