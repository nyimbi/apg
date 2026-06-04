"""
APG Backup & Restore (BKUP) - Expanded Service Implementation

Dependency-light in-memory store pattern. 42+ async methods covering
scheduling, incremental/differential backups, encryption, offsite sync,
RPO/RTO analysis, disaster recovery tests, and compliance reporting.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7

import logging

logger = logging.getLogger(__name__)


def uuid7str() -> str:
	return str(uuid7())


def _ts() -> str:
	return datetime.utcnow().isoformat(timespec="seconds")


class _R(dict[str, Any]):
	"""Thin dict wrapper for record instances."""


class BkupService:
	"""
	42+ async methods for backup plans, snapshots, schedules,
	encryption, offsite sync, incremental/differential snapshots,
	cataloguing, disaster recovery tests, RPO checks, RTO estimates,
	retention policies, and compliance reporting.
	"""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id

		self._plans:       dict[tuple[str, str], _R] = {}
		self._snapshots:   dict[tuple[str, str], _R] = {}
		self._schedules:   dict[tuple[str, str], _R] = {}
		self._restore_runs: dict[tuple[str, str], _R] = {}
		self._retention_policies: dict[tuple[str, str], _R] = {}
		self._offsite_syncs: dict[tuple[str, str], _R] = {}
		self._dr_tests:    dict[tuple[str, str], _R] = {}
		self._catalogue:   list[_R] = []
		self._audit_log:   list[_R] = []

	# ------------------------------------------------------------------
	# helpers
	# ------------------------------------------------------------------

	def _key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	async def _audit(self, event_type: str, record_id: str, details: dict[str, Any] | None = None) -> None:
		self._audit_log.append(_R(
			event_id=uuid7str(),
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			event_type=event_type,
			record_id=record_id,
			details=details or {},
			occurred_at=_ts(),
		))

	def _require_plan(self, plan_id: str) -> _R:
		r = self._plans.get(self._key(self.tenant_id, plan_id))
		if r is None:
			raise KeyError(f"backup plan not found: {plan_id}")
		return r

	def _require_snapshot(self, snapshot_id: str) -> _R:
		r = self._snapshots.get(self._key(self.tenant_id, snapshot_id))
		if r is None:
			raise KeyError(f"snapshot not found: {snapshot_id}")
		return r

	# ------------------------------------------------------------------
	# 1. Create backup plan
	# ------------------------------------------------------------------

	async def create_backup_plan(
		self,
		name: str,
		sources: list[str],
		retention_days: int = 30,
		rpo_minutes: int = 60,
		owner: str = "system",
	) -> _R:
		"""Define a backup plan covering one or more data sources."""
		assert name, "plan name required"
		assert sources, "at least one source required"
		assert retention_days > 0, "retention_days must be positive"
		plan_id = uuid7str()
		record = _R(
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			name=name,
			sources=list(sources),
			retention_days=retention_days,
			rpo_minutes=rpo_minutes,
			owner=owner,
			status="active",
			created_at=_ts(),
		)
		self._plans[self._key(self.tenant_id, plan_id)] = record
		await self._audit("plan_created", plan_id, {"name": name, "sources": sources})
		return record

	# ------------------------------------------------------------------
	# 2. Backup schedule
	# ------------------------------------------------------------------

	async def backup_schedule(
		self,
		plan_id: str,
		cron_expression: str,
		backup_type: str = "full",
		enabled: bool = True,
	) -> _R:
		"""Attach a cron-based schedule to a backup plan."""
		plan = self._require_plan(plan_id)
		schedule_id = uuid7str()
		record = _R(
			schedule_id=schedule_id,
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			cron_expression=cron_expression,
			backup_type=backup_type,
			enabled=enabled,
			created_at=_ts(),
		)
		self._schedules[self._key(self.tenant_id, schedule_id)] = record
		plan["schedule_id"] = schedule_id
		await self._audit("schedule_created", schedule_id, {"plan_id": plan_id, "cron": cron_expression})
		return record

	# ------------------------------------------------------------------
	# 3. Backup run
	# ------------------------------------------------------------------

	async def backup_run(
		self,
		plan_id: str,
		backup_type: str = "full",
		triggered_by: str = "scheduler",
	) -> _R:
		"""Execute a backup run for a plan (full, incremental, or differential)."""
		assert backup_type in {"full", "incremental", "differential"}, f"unknown backup_type: {backup_type}"
		plan = self._require_plan(plan_id)
		snapshot_id = uuid7str()
		size_bytes = 1024 * 1024 * (100 if backup_type == "full" else 10)
		record = _R(
			snapshot_id=snapshot_id,
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			backup_type=backup_type,
			size_bytes=size_bytes,
			encrypted=True,
			integrity_check_passed=True,
			status="available",
			triggered_by=triggered_by,
			created_at=_ts(),
		)
		self._snapshots[self._key(self.tenant_id, snapshot_id)] = record
		self._catalogue.append(_R(
			catalogue_id=uuid7str(),
			snapshot_id=snapshot_id,
			plan_id=plan_id,
			backup_type=backup_type,
			size_bytes=size_bytes,
			created_at=_ts(),
		))
		await self._audit("backup_run", snapshot_id, {"plan_id": plan_id, "type": backup_type})
		return record

	# ------------------------------------------------------------------
	# 4. Incremental backup
	# ------------------------------------------------------------------

	async def incremental_backup(self, plan_id: str, parent_snapshot_id: str) -> _R:
		"""Run an incremental backup relative to a parent snapshot."""
		plan = self._require_plan(plan_id)
		parent = self._require_snapshot(parent_snapshot_id)
		snapshot = await self.backup_run(plan_id, backup_type="incremental")
		snapshot["parent_snapshot_id"] = parent_snapshot_id
		snapshot["lineage"] = parent.get("lineage", []) + [parent_snapshot_id]
		await self._audit("incremental_backup", snapshot["snapshot_id"], {"parent": parent_snapshot_id})
		return snapshot

	# ------------------------------------------------------------------
	# 5. Differential backup
	# ------------------------------------------------------------------

	async def differential_backup(self, plan_id: str, base_snapshot_id: str) -> _R:
		"""Run a differential backup relative to the last full snapshot."""
		plan = self._require_plan(plan_id)
		base = self._require_snapshot(base_snapshot_id)
		assert base["backup_type"] == "full", "differential base must be a full snapshot"
		snapshot = await self.backup_run(plan_id, backup_type="differential")
		snapshot["base_snapshot_id"] = base_snapshot_id
		await self._audit("differential_backup", snapshot["snapshot_id"], {"base": base_snapshot_id})
		return snapshot

	# ------------------------------------------------------------------
	# 6. Restore from snapshot
	# ------------------------------------------------------------------

	async def restore_from(
		self,
		snapshot_id: str,
		target_environment: str,
		requested_by: str,
		point_in_time: str | None = None,
	) -> _R:
		"""Initiate a restore from a specific snapshot."""
		snapshot = self._require_snapshot(snapshot_id)
		assert snapshot["status"] == "available", "snapshot not available"
		restore_id = uuid7str()
		record = _R(
			restore_id=restore_id,
			snapshot_id=snapshot_id,
			tenant_id=self.tenant_id,
			target_environment=target_environment,
			requested_by=requested_by,
			point_in_time=point_in_time,
			status="completed",
			rto_minutes=15,
			started_at=_ts(),
			completed_at=_ts(),
		)
		self._restore_runs[self._key(self.tenant_id, restore_id)] = record
		await self._audit("restore_initiated", restore_id, {"snapshot_id": snapshot_id, "environment": target_environment})
		return record

	# ------------------------------------------------------------------
	# 7. Verify backup
	# ------------------------------------------------------------------

	async def verify_backup(self, snapshot_id: str) -> _R:
		"""Verify the integrity of a snapshot."""
		snapshot = self._require_snapshot(snapshot_id)
		# Simulate checksum verification
		check_passed = snapshot.get("integrity_check_passed", True)
		snapshot["verified_at"] = _ts()
		snapshot["verification_status"] = "passed" if check_passed else "failed"
		result = _R(
			snapshot_id=snapshot_id,
			integrity_passed=check_passed,
			verified_at=_ts(),
		)
		await self._audit("backup_verified", snapshot_id, {"integrity_passed": check_passed})
		return result

	# ------------------------------------------------------------------
	# 8. Test restore
	# ------------------------------------------------------------------

	async def test_restore(
		self,
		snapshot_id: str,
		sandbox_environment: str = "sandbox",
		requested_by: str = "system",
	) -> _R:
		"""Run a non-destructive test restore in a sandbox environment."""
		snapshot = self._require_snapshot(snapshot_id)
		restore = await self.restore_from(snapshot_id, sandbox_environment, requested_by)
		restore["test_restore"] = True
		restore["rto_minutes"] = 12
		result = _R(
			snapshot_id=snapshot_id,
			test_restore_id=restore["restore_id"],
			rto_minutes=restore["rto_minutes"],
			passed=True,
			sandbox=sandbox_environment,
			tested_at=_ts(),
		)
		await self._audit("test_restore_completed", restore["restore_id"], {"snapshot_id": snapshot_id, "rto_minutes": restore["rto_minutes"]})
		return result

	# ------------------------------------------------------------------
	# 9. Retention policy
	# ------------------------------------------------------------------

	async def retention_policy(
		self,
		plan_id: str,
		daily_copies: int = 7,
		weekly_copies: int = 4,
		monthly_copies: int = 12,
		yearly_copies: int = 3,
	) -> _R:
		"""Set GFS (Grandfather-Father-Son) retention policy for a plan."""
		plan = self._require_plan(plan_id)
		policy_id = uuid7str()
		record = _R(
			policy_id=policy_id,
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			daily_copies=daily_copies,
			weekly_copies=weekly_copies,
			monthly_copies=monthly_copies,
			yearly_copies=yearly_copies,
			created_at=_ts(),
		)
		self._retention_policies[self._key(self.tenant_id, policy_id)] = record
		plan["retention_policy_id"] = policy_id
		await self._audit("retention_policy_set", policy_id, {"plan_id": plan_id})
		return record

	# ------------------------------------------------------------------
	# 10. Encryption at rest
	# ------------------------------------------------------------------

	async def encryption_at_rest(self, snapshot_id: str, key_ref: str = "kms://default") -> _R:
		"""Apply or confirm encryption-at-rest for a snapshot."""
		snapshot = self._require_snapshot(snapshot_id)
		snapshot["encrypted"] = True
		snapshot["encryption_key_ref"] = key_ref
		snapshot["encrypted_at"] = _ts()
		await self._audit("snapshot_encrypted", snapshot_id, {"key_ref": key_ref})
		return snapshot

	# ------------------------------------------------------------------
	# 11. Offsite sync
	# ------------------------------------------------------------------

	async def offsite_sync(
		self,
		snapshot_id: str,
		destination: str,
		sync_type: str = "replicate",
	) -> _R:
		"""Sync a snapshot to an offsite location."""
		snapshot = self._require_snapshot(snapshot_id)
		sync_id = uuid7str()
		record = _R(
			sync_id=sync_id,
			snapshot_id=snapshot_id,
			tenant_id=self.tenant_id,
			destination=destination,
			sync_type=sync_type,
			status="completed",
			bytes_transferred=snapshot.get("size_bytes", 0),
			synced_at=_ts(),
		)
		self._offsite_syncs[self._key(self.tenant_id, sync_id)] = record
		snapshot["offsite_location"] = destination
		await self._audit("offsite_synced", sync_id, {"destination": destination, "snapshot_id": snapshot_id})
		return record

	# ------------------------------------------------------------------
	# 12. Backup catalogue
	# ------------------------------------------------------------------

	async def backup_catalogue(self, plan_id: str | None = None) -> list[_R]:
		"""List all catalogue entries, optionally filtered by plan."""
		entries = [
			e for e in self._catalogue
			if (plan_id is None or e["plan_id"] == plan_id)
		]
		await self._audit("catalogue_queried", "system", {"plan_id": plan_id, "count": len(entries)})
		return entries

	# ------------------------------------------------------------------
	# 13. Backup report
	# ------------------------------------------------------------------

	async def backup_report(self, plan_id: str | None = None) -> _R:
		"""Generate a summary backup report for a plan or all plans."""
		snapshots = [
			s for (tid, _), s in self._snapshots.items()
			if tid == self.tenant_id and (plan_id is None or s["plan_id"] == plan_id)
		]
		available = [s for s in snapshots if s["status"] == "available"]
		total_bytes = sum(s.get("size_bytes", 0) for s in available)
		report = _R(
			tenant_id=self.tenant_id,
			plan_id=plan_id,
			total_snapshots=len(snapshots),
			available_snapshots=len(available),
			total_size_bytes=total_bytes,
			encrypted_count=sum(1 for s in available if s.get("encrypted")),
			offsite_synced_count=sum(1 for s in available if s.get("offsite_location")),
			generated_at=_ts(),
		)
		await self._audit("backup_report_generated", "system", {"plan_id": plan_id})
		return report

	# ------------------------------------------------------------------
	# 14. Disaster recovery test
	# ------------------------------------------------------------------

	async def disaster_recovery_test(
		self,
		plan_id: str,
		scenario: str = "full_site_failure",
		requested_by: str = "system",
	) -> _R:
		"""Execute a simulated DR test for a backup plan."""
		plan = self._require_plan(plan_id)
		# Find most recent available snapshot
		plan_snapshots = [
			s for (tid, _), s in self._snapshots.items()
			if tid == self.tenant_id and s["plan_id"] == plan_id and s["status"] == "available"
		]
		assert plan_snapshots, "no available snapshots for DR test"
		latest = sorted(plan_snapshots, key=lambda s: s["created_at"])[-1]
		restore = await self.test_restore(latest["snapshot_id"], "dr_sandbox", requested_by)
		dr_test_id = uuid7str()
		record = _R(
			dr_test_id=dr_test_id,
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			scenario=scenario,
			snapshot_id=latest["snapshot_id"],
			rto_minutes=restore["rto_minutes"],
			passed=restore["passed"],
			tested_at=_ts(),
			requested_by=requested_by,
		)
		self._dr_tests[self._key(self.tenant_id, dr_test_id)] = record
		await self._audit("dr_test_completed", dr_test_id, {"plan_id": plan_id, "scenario": scenario, "passed": restore["passed"]})
		return record

	# ------------------------------------------------------------------
	# 15. RPO check
	# ------------------------------------------------------------------

	async def rpo_check(self, plan_id: str) -> _R:
		"""Check whether the most recent snapshot meets the plan's RPO target."""
		plan = self._require_plan(plan_id)
		rpo_minutes = plan.get("rpo_minutes", 60)
		plan_snapshots = sorted(
			[s for (tid, _), s in self._snapshots.items() if tid == self.tenant_id and s["plan_id"] == plan_id and s["status"] == "available"],
			key=lambda s: s["created_at"],
		)
		if not plan_snapshots:
			return _R(plan_id=plan_id, rpo_met=False, gap_minutes=None, reason="no_snapshots", checked_at=_ts())
		latest = plan_snapshots[-1]
		latest_dt = datetime.fromisoformat(latest["created_at"])
		gap_minutes = (datetime.utcnow() - latest_dt).total_seconds() / 60
		rpo_met = gap_minutes <= rpo_minutes
		result = _R(
			plan_id=plan_id,
			rpo_minutes=rpo_minutes,
			gap_minutes=round(gap_minutes, 2),
			rpo_met=rpo_met,
			latest_snapshot_id=latest["snapshot_id"],
			checked_at=_ts(),
		)
		await self._audit("rpo_checked", plan_id, {"rpo_met": rpo_met, "gap_minutes": round(gap_minutes, 2)})
		return result

	# ------------------------------------------------------------------
	# 16. RTO estimate
	# ------------------------------------------------------------------

	async def rto_estimate(self, plan_id: str) -> _R:
		"""Estimate RTO based on historical restore run durations."""
		restores = [
			r for (tid, _), r in self._restore_runs.items()
			if tid == self.tenant_id
		]
		plan_snapshots_ids = {
			s["snapshot_id"]
			for (tid, _), s in self._snapshots.items()
			if tid == self.tenant_id and s["plan_id"] == plan_id
		}
		plan_restores = [r for r in restores if r["snapshot_id"] in plan_snapshots_ids]
		if not plan_restores:
			return _R(plan_id=plan_id, estimated_rto_minutes=None, sample_size=0, estimated_at=_ts())
		durations = [r.get("rto_minutes", 15) for r in plan_restores]
		avg_rto = round(statistics.mean(durations), 2)
		p95_rto = round(sorted(durations)[int(len(durations) * 0.95)], 2) if len(durations) >= 2 else avg_rto
		result = _R(
			plan_id=plan_id,
			estimated_rto_minutes=avg_rto,
			p95_rto_minutes=p95_rto,
			sample_size=len(plan_restores),
			estimated_at=_ts(),
		)
		await self._audit("rto_estimated", plan_id, {"avg_rto": avg_rto, "samples": len(plan_restores)})
		return result

	# ------------------------------------------------------------------
	# 17. Bulk create snapshots
	# ------------------------------------------------------------------

	async def bulk_create_snapshots(self, plan_ids: list[str], backup_type: str = "full") -> list[_R]:
		"""Run a backup for each listed plan in one call."""
		results = []
		for pid in plan_ids:
			snapshot = await self.backup_run(pid, backup_type=backup_type)
			results.append(snapshot)
		await self._audit("bulk_snapshots_created", "system", {"count": len(results)})
		return results

	# ------------------------------------------------------------------
	# 18. Bulk delete snapshots
	# ------------------------------------------------------------------

	async def bulk_delete_snapshots(self, snapshot_ids: list[str]) -> _R:
		"""Expire multiple snapshots at once (retention enforcement)."""
		deleted = []
		for sid in snapshot_ids:
			snapshot = self._snapshots.get(self._key(self.tenant_id, sid))
			if snapshot:
				snapshot["status"] = "deleted"
				snapshot["deleted_at"] = _ts()
				deleted.append(sid)
		await self._audit("bulk_snapshots_deleted", "system", {"count": len(deleted)})
		return _R(deleted_count=len(deleted), snapshot_ids=deleted)

	# ------------------------------------------------------------------
	# 19. List plans
	# ------------------------------------------------------------------

	async def list_plans(self, status: str | None = None) -> list[_R]:
		"""List backup plans for the current tenant."""
		plans = [
			p for (tid, _), p in self._plans.items()
			if tid == self.tenant_id and (status is None or p["status"] == status)
		]
		return sorted(plans, key=lambda p: p["created_at"])

	# ------------------------------------------------------------------
	# 20. List snapshots
	# ------------------------------------------------------------------

	async def list_snapshots(self, plan_id: str | None = None, backup_type: str | None = None) -> list[_R]:
		"""List snapshots, optionally filtered by plan or type."""
		snaps = [
			s for (tid, _), s in self._snapshots.items()
			if tid == self.tenant_id
			and (plan_id is None or s["plan_id"] == plan_id)
			and (backup_type is None or s["backup_type"] == backup_type)
		]
		return sorted(snaps, key=lambda s: s["created_at"])

	# ------------------------------------------------------------------
	# 21. Update plan
	# ------------------------------------------------------------------

	async def update_plan(self, plan_id: str, **kwargs: Any) -> _R:
		"""Update mutable fields on a backup plan."""
		plan = self._require_plan(plan_id)
		allowed = {"name", "retention_days", "rpo_minutes", "owner", "status"}
		for k, v in kwargs.items():
			if k in allowed:
				plan[k] = v
		plan["updated_at"] = _ts()
		await self._audit("plan_updated", plan_id, {k: v for k, v in kwargs.items() if k in allowed})
		return plan

	# ------------------------------------------------------------------
	# 22. Delete plan
	# ------------------------------------------------------------------

	async def delete_plan(self, plan_id: str) -> _R:
		"""Soft-delete a backup plan."""
		plan = self._require_plan(plan_id)
		plan["status"] = "deleted"
		plan["deleted_at"] = _ts()
		await self._audit("plan_deleted", plan_id, {})
		return plan

	# ------------------------------------------------------------------
	# 23. Schedule update
	# ------------------------------------------------------------------

	async def schedule_update(self, schedule_id: str, cron_expression: str | None = None, enabled: bool | None = None) -> _R:
		"""Update a backup schedule."""
		schedule = self._schedules.get(self._key(self.tenant_id, schedule_id))
		assert schedule is not None, f"schedule not found: {schedule_id}"
		if cron_expression is not None:
			schedule["cron_expression"] = cron_expression
		if enabled is not None:
			schedule["enabled"] = enabled
		schedule["updated_at"] = _ts()
		await self._audit("schedule_updated", schedule_id, {"cron": cron_expression, "enabled": enabled})
		return schedule

	# ------------------------------------------------------------------
	# 24. Approve restore
	# ------------------------------------------------------------------

	async def approve_restore(self, restore_id: str, reviewer: str, notes: str = "") -> _R:
		"""Approve a pending restore run."""
		restore = self._restore_runs.get(self._key(self.tenant_id, restore_id))
		assert restore is not None, f"restore run not found: {restore_id}"
		assert reviewer, "reviewer required"
		restore["approved_by"] = reviewer
		restore["approval_notes"] = notes
		restore["approved_at"] = _ts()
		await self._audit("restore_approved", restore_id, {"reviewer": reviewer})
		return restore

	# ------------------------------------------------------------------
	# 25. List restore runs
	# ------------------------------------------------------------------

	async def list_restore_runs(self, plan_id: str | None = None) -> list[_R]:
		"""List restore runs for the tenant."""
		plan_snapshot_ids: set[str] | None = None
		if plan_id is not None:
			plan_snapshot_ids = {
				s["snapshot_id"]
				for (tid, _), s in self._snapshots.items()
				if tid == self.tenant_id and s["plan_id"] == plan_id
			}
		runs = [
			r for (tid, _), r in self._restore_runs.items()
			if tid == self.tenant_id
			and (plan_snapshot_ids is None or r["snapshot_id"] in plan_snapshot_ids)
		]
		return sorted(runs, key=lambda r: r["started_at"])

	# ------------------------------------------------------------------
	# 26. List offsite syncs
	# ------------------------------------------------------------------

	async def list_offsite_syncs(self) -> list[_R]:
		"""List offsite sync records for the tenant."""
		syncs = [s for (tid, _), s in self._offsite_syncs.items() if tid == self.tenant_id]
		return sorted(syncs, key=lambda s: s["synced_at"])

	# ------------------------------------------------------------------
	# 27. Export snapshots to CSV
	# ------------------------------------------------------------------

	async def export_snapshots_csv(self) -> str:
		"""Export snapshot metadata to CSV."""
		snapshots = await self.list_snapshots()
		buf = io.StringIO()
		fields = ["snapshot_id", "plan_id", "backup_type", "size_bytes", "status", "encrypted", "created_at"]
		writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(snapshots)
		await self._audit("snapshots_exported_csv", "system", {"count": len(snapshots)})
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 28. Export plan report to JSON
	# ------------------------------------------------------------------

	async def export_plan_report_json(self) -> str:
		"""Export all plans and their snapshot counts as JSON."""
		plans = await self.list_plans()
		for plan in plans:
			plan["snapshot_count"] = len(await self.list_snapshots(plan_id=plan["plan_id"]))
		await self._audit("plan_report_exported_json", "system", {"count": len(plans)})
		return json.dumps(plans, default=str, indent=2)

	# ------------------------------------------------------------------
	# 29. Health check
	# ------------------------------------------------------------------

	async def health_check(self) -> _R:
		"""Service health and storage summary."""
		plan_count = sum(1 for (tid, _) in self._plans if tid == self.tenant_id)
		snap_count = sum(1 for (tid, _) in self._snapshots if tid == self.tenant_id)
		available = sum(1 for (tid, _), s in self._snapshots.items() if tid == self.tenant_id and s["status"] == "available")
		return _R(
			status="healthy",
			tenant_id=self.tenant_id,
			plan_count=plan_count,
			snapshot_count=snap_count,
			available_snapshots=available,
			restore_run_count=sum(1 for (tid, _) in self._restore_runs if tid == self.tenant_id),
			audit_event_count=len(self._audit_log),
			checked_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 30. Dashboard / KPI summary
	# ------------------------------------------------------------------

	async def dashboard(self) -> _R:
		"""KPI dashboard aggregating key backup metrics."""
		plans = await self.list_plans()
		snapshots = await self.list_snapshots()
		restores = await self.list_restore_runs()
		total_bytes = sum(s.get("size_bytes", 0) for s in snapshots if s["status"] == "available")
		offsite_count = sum(1 for s in snapshots if s.get("offsite_location"))
		encrypted_count = sum(1 for s in snapshots if s.get("encrypted"))
		return _R(
			tenant_id=self.tenant_id,
			plan_count=len(plans),
			active_plans=sum(1 for p in plans if p["status"] == "active"),
			total_snapshots=len(snapshots),
			available_snapshots=sum(1 for s in snapshots if s["status"] == "available"),
			encrypted_snapshots=encrypted_count,
			offsite_synced_snapshots=offsite_count,
			total_backup_bytes=total_bytes,
			restore_runs=len(restores),
			dr_tests_passed=sum(1 for (tid, _), d in self._dr_tests.items() if tid == self.tenant_id and d["passed"]),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 31. Compliance report (SOC2 / ISO 27001)
	# ------------------------------------------------------------------

	async def compliance_report(self, framework: str = "SOC2") -> _R:
		"""Generate a backup compliance report for SOC2 or ISO 27001."""
		plans = await self.list_plans()
		snapshots = await self.list_snapshots()
		encrypted = sum(1 for s in snapshots if s.get("encrypted") and s["status"] == "available")
		offsite = sum(1 for s in snapshots if s.get("offsite_location") and s["status"] == "available")
		available = sum(1 for s in snapshots if s["status"] == "available")
		dr_tests = [d for (tid, _), d in self._dr_tests.items() if tid == self.tenant_id]
		last_dr_test = sorted(dr_tests, key=lambda d: d["tested_at"])[-1] if dr_tests else None
		report = _R(
			framework=framework,
			tenant_id=self.tenant_id,
			total_plans=len(plans),
			total_available_snapshots=available,
			encryption_rate=round(encrypted / max(available, 1), 4),
			offsite_rate=round(offsite / max(available, 1), 4),
			retention_policies_defined=sum(1 for (tid, _) in self._retention_policies if tid == self.tenant_id),
			dr_tests_executed=len(dr_tests),
			last_dr_test_at=last_dr_test["tested_at"] if last_dr_test else None,
			last_dr_test_passed=last_dr_test["passed"] if last_dr_test else None,
			audit_trail_complete=True,
			generated_at=_ts(),
		)
		await self._audit("compliance_report_generated", "system", {"framework": framework})
		return report

	# ------------------------------------------------------------------
	# 32. Audit trail
	# ------------------------------------------------------------------

	async def audit_trail(self, event_type: str | None = None) -> list[_R]:
		"""Return audit events for the tenant, optionally filtered."""
		events = [
			e for e in self._audit_log
			if e["tenant_id"] == self.tenant_id and (event_type is None or e["event_type"] == event_type)
		]
		return events

	# ------------------------------------------------------------------
	# 33. Snapshot search
	# ------------------------------------------------------------------

	async def snapshot_search(
		self,
		query: str,
		backup_type: str | None = None,
		status: str | None = None,
	) -> list[_R]:
		"""Search snapshots by plan name fragment or backup type."""
		plan_name_index = {
			p["plan_id"]: p["name"].lower()
			for (tid, _), p in self._plans.items()
			if tid == self.tenant_id
		}
		results = []
		for (tid, _), s in self._snapshots.items():
			if tid != self.tenant_id:
				continue
			if backup_type and s["backup_type"] != backup_type:
				continue
			if status and s["status"] != status:
				continue
			plan_name = plan_name_index.get(s["plan_id"], "")
			if query.lower() in plan_name or query.lower() in s["snapshot_id"]:
				results.append(s)
		return results

	# ------------------------------------------------------------------
	# 34. Storage utilisation report
	# ------------------------------------------------------------------

	async def storage_utilisation(self) -> _R:
		"""Report on backup storage utilisation by plan."""
		plan_bytes: dict[str, int] = {}
		for (tid, _), s in self._snapshots.items():
			if tid == self.tenant_id and s["status"] == "available":
				plan_bytes[s["plan_id"]] = plan_bytes.get(s["plan_id"], 0) + s.get("size_bytes", 0)
		total = sum(plan_bytes.values())
		breakdown = [{"plan_id": pid, "size_bytes": sz} for pid, sz in sorted(plan_bytes.items(), key=lambda x: -x[1])]
		return _R(
			tenant_id=self.tenant_id,
			total_bytes=total,
			plan_breakdown=breakdown,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 35. Expiry enforcement
	# ------------------------------------------------------------------

	async def enforce_expiry(self, plan_id: str) -> _R:
		"""Delete snapshots that exceed the plan's retention_days."""
		plan = self._require_plan(plan_id)
		retention_days = plan.get("retention_days", 30)
		cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
		expired = []
		for (tid, sid), s in self._snapshots.items():
			if tid == self.tenant_id and s["plan_id"] == plan_id and s["created_at"] < cutoff and s["status"] == "available":
				s["status"] = "expired"
				s["expired_at"] = _ts()
				expired.append(sid)
		result = _R(
			plan_id=plan_id,
			expired_count=len(expired),
			snapshot_ids=expired,
			enforced_at=_ts(),
		)
		await self._audit("expiry_enforced", plan_id, {"expired": len(expired)})
		return result

	# ------------------------------------------------------------------
	# 36. Clone snapshot to new plan
	# ------------------------------------------------------------------

	async def clone_snapshot(self, snapshot_id: str, target_plan_id: str) -> _R:
		"""Clone a snapshot's metadata into another backup plan."""
		source = self._require_snapshot(snapshot_id)
		target_plan = self._require_plan(target_plan_id)
		new_id = uuid7str()
		clone = _R(**source, snapshot_id=new_id, plan_id=target_plan_id, cloned_from=snapshot_id, created_at=_ts())
		self._snapshots[self._key(self.tenant_id, new_id)] = clone
		await self._audit("snapshot_cloned", new_id, {"source": snapshot_id, "target_plan": target_plan_id})
		return clone

	# ------------------------------------------------------------------
	# 37. Snapshot annotate
	# ------------------------------------------------------------------

	async def snapshot_annotate(self, snapshot_id: str, tags: dict[str, str]) -> _R:
		"""Add metadata tags to a snapshot."""
		snapshot = self._require_snapshot(snapshot_id)
		snapshot.setdefault("tags", {})
		snapshot["tags"].update(tags)
		await self._audit("snapshot_annotated", snapshot_id, {"tags": tags})
		return snapshot

	# ------------------------------------------------------------------
	# 38. Point-in-time restore
	# ------------------------------------------------------------------

	async def point_in_time_restore(
		self,
		plan_id: str,
		target_datetime: str,
		target_environment: str,
		requested_by: str,
	) -> _R:
		"""Find the best snapshot for a point-in-time restore and execute it."""
		plan_snapshots = sorted(
			[s for (tid, _), s in self._snapshots.items() if tid == self.tenant_id and s["plan_id"] == plan_id and s["status"] == "available"],
			key=lambda s: s["created_at"],
		)
		assert plan_snapshots, "no snapshots available for plan"
		best = max(
			[s for s in plan_snapshots if s["created_at"] <= target_datetime],
			key=lambda s: s["created_at"],
			default=plan_snapshots[0],
		)
		restore = await self.restore_from(best["snapshot_id"], target_environment, requested_by, point_in_time=target_datetime)
		restore["pit_target"] = target_datetime
		await self._audit("pit_restore", restore["restore_id"], {"target_datetime": target_datetime, "snapshot_id": best["snapshot_id"]})
		return restore

	# ------------------------------------------------------------------
	# 39. Backup chain validation
	# ------------------------------------------------------------------

	async def validate_backup_chain(self, snapshot_id: str) -> _R:
		"""Validate that an incremental/differential chain is unbroken."""
		snapshot = self._require_snapshot(snapshot_id)
		lineage = snapshot.get("lineage", [])
		broken = False
		missing = []
		for ancestor_id in lineage:
			ancestor = self._snapshots.get(self._key(self.tenant_id, ancestor_id))
			if ancestor is None or ancestor["status"] not in {"available"}:
				broken = True
				missing.append(ancestor_id)
		result = _R(
			snapshot_id=snapshot_id,
			chain_complete=not broken,
			lineage=lineage,
			missing_ancestors=missing,
			validated_at=_ts(),
		)
		await self._audit("chain_validated", snapshot_id, {"chain_complete": not broken})
		return result

	# ------------------------------------------------------------------
	# 40. Legal hold
	# ------------------------------------------------------------------

	async def legal_hold(self, plan_id: str, hold: bool, reason: str = "") -> _R:
		"""Place or lift a legal hold on all snapshots for a plan."""
		plan = self._require_plan(plan_id)
		plan["legal_hold"] = hold
		plan["legal_hold_reason"] = reason
		plan["legal_hold_updated_at"] = _ts()
		affected = 0
		for (tid, _), s in self._snapshots.items():
			if tid == self.tenant_id and s["plan_id"] == plan_id:
				s["legal_hold"] = hold
				affected += 1
		result = _R(plan_id=plan_id, legal_hold=hold, affected_snapshots=affected, updated_at=_ts())
		await self._audit("legal_hold_updated", plan_id, {"hold": hold, "affected": affected})
		return result

	# ------------------------------------------------------------------
	# 41. Summary for capability contract
	# ------------------------------------------------------------------

	async def continuity_summary(self) -> _R:
		"""High-level business continuity summary."""
		return await self.dashboard()

	# ------------------------------------------------------------------
	# 42. Full DR runbook execution
	# ------------------------------------------------------------------

	async def dr_runbook_execute(
		self,
		plan_id: str,
		scenario: str = "region_failure",
		requested_by: str = "system",
	) -> _R:
		"""Execute a full DR runbook: DR test + RPO check + RTO estimate."""
		dr_test = await self.disaster_recovery_test(plan_id, scenario, requested_by)
		rpo = await self.rpo_check(plan_id)
		rto = await self.rto_estimate(plan_id)
		runbook_id = uuid7str()
		result = _R(
			runbook_id=runbook_id,
			plan_id=plan_id,
			scenario=scenario,
			dr_test=dict(dr_test),
			rpo_check=dict(rpo),
			rto_estimate=dict(rto),
			overall_pass=dr_test["passed"] and rpo["rpo_met"],
			executed_at=_ts(),
		)
		await self._audit("dr_runbook_executed", runbook_id, {"plan_id": plan_id, "scenario": scenario, "overall_pass": result["overall_pass"]})
		return result
