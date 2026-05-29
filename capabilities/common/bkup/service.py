"""Service layer for APG Backup and Restore."""

from __future__ import annotations

from typing import Any

from .backup_engine import BackupEngine
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import BackupAuditEvent, BackupPlan, BackupSnapshot, ContinuityReport, RestoreRun


class BkupService:
	"""Backup plan registry, encrypted snapshot vault, restore console, and reports."""

	def __init__(self) -> None:
		self._plans: dict[str, BackupPlan] = {}
		self._snapshots: dict[str, BackupSnapshot] = {}
		self._restores: dict[str, RestoreRun] = {}
		self._reports: dict[str, ContinuityReport] = {}
		self._audit_events: dict[str, BackupAuditEvent] = {}
		self._engine = BackupEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_backup_plan(
		self,
		plan_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		schedule: str,
		sources: list[str] | tuple[str, ...],
		retention_days: int = 30,
		rpo_minutes: int = 60,
		legal_hold: bool = False,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_backup_plan",
			"plan_owner_assigned": bool(owner),
		})
		self._raise_if_denied(result)
		if not schedule:
			raise PermissionError("backup_schedule_required")
		if not sources:
			raise PermissionError("source_inventory_required")
		if retention_days <= 0:
			raise PermissionError("retention_policy_required")
		plan = BackupPlan(
			id=plan_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			schedule=schedule,
			sources=tuple(str(item) for item in sources),
			retention_days=int(retention_days),
			rpo_minutes=int(rpo_minutes),
			legal_hold=legal_hold,
		)
		self._plans[plan_id] = plan
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=plan_id,
			event_type="backup_plan_created",
			actor=owner,
			decision=result["decision"],
			metadata={"source_count": len(sources), "schedule": schedule},
		)
		return plan.to_dict()

	def list_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		plans = list(self._plans.values())
		if tenant_id is not None:
			plans = [item for item in plans if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(plans, key=lambda item: item.id)]

	def create_snapshot(
		self,
		snapshot_id: str,
		tenant_id: str,
		plan_id: str,
		source_id: str,
		size_bytes: int,
		encrypted: bool = True,
		integrity_check_passed: bool = True,
		lineage: list[str] | tuple[str, ...] | None = None,
		region: str = "primary",
		data_fingerprint: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		plan = self._require_plan(plan_id, tenant_id)
		if source_id not in plan.sources:
			raise KeyError(f"source not in backup plan: {source_id}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_snapshot",
			"snapshot_encrypted": bool(encrypted),
		})
		self._raise_if_denied(result)
		if not integrity_check_passed:
			raise PermissionError("snapshot_integrity_check_required")
		payload = {
			"id": snapshot_id,
			"tenant_id": tenant_id,
			"plan_id": plan_id,
			"source_id": source_id,
			"size_bytes": int(size_bytes),
			"lineage": list(lineage or ()),
			"region": region,
			"data_fingerprint": data_fingerprint or snapshot_id,
		}
		snapshot = BackupSnapshot(
			id=snapshot_id,
			tenant_id=tenant_id,
			plan_id=plan_id,
			source_id=source_id,
			snapshot_hash=self._engine.snapshot_hash(payload),
			size_bytes=int(size_bytes),
			encrypted=encrypted,
			integrity_status="passed",
			lineage=tuple(lineage or (plan_id,)),
			region=region,
		)
		self._snapshots[snapshot_id] = snapshot
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=snapshot_id,
			event_type="snapshot_created",
			actor=plan.owner,
			decision=result["decision"],
			metadata={"plan_id": plan_id, "source_id": source_id, "region": region},
		)
		return snapshot.to_dict()

	def list_snapshots(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		snapshots = list(self._snapshots.values())
		if tenant_id is not None:
			snapshots = [item for item in snapshots if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(snapshots, key=lambda item: item.id)]

	def restore_snapshot(
		self,
		restore_id: str,
		tenant_id: str,
		snapshot_id: str,
		target_environment: str,
		requested_by: str,
		integrity_check_passed: bool = True,
		approval_recorded: bool = False,
		point_in_time: str | None = None,
		days_since_restore_test: int = 0,
		restore_test_review_recorded: bool = True,
		rto_minutes: int = 0,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		snapshot = self._require_snapshot(snapshot_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "restore",
			"integrity_check_passed": bool(integrity_check_passed),
			"target_environment": target_environment,
			"approval_recorded": bool(approval_recorded),
			"days_since_restore_test": int(days_since_restore_test),
			"restore_test_review_recorded": bool(restore_test_review_recorded),
		})
		self._raise_if_denied(result)
		review_status = "required" if result["decision"] == "require_review" else "approved"
		status = "pending_review" if result["decision"] == "require_review" else "completed"
		restore = RestoreRun(
			id=restore_id,
			tenant_id=tenant_id,
			snapshot_id=snapshot_id,
			target_environment=target_environment,
			requested_by=requested_by,
			status=status,
			integrity_check_passed=integrity_check_passed,
			approval_recorded=approval_recorded,
			point_in_time=point_in_time,
			review_status=review_status,
			rto_minutes=int(rto_minutes),
		)
		self._restores[restore_id] = restore
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=restore_id,
			event_type="restore_requested",
			actor=requested_by,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"snapshot_id": snapshot.id, "target_environment": target_environment},
		)
		return restore.to_dict()

	def approve_restore(self, restore_id: str, reviewer: str) -> dict[str, Any]:
		restore = self._restores.get(restore_id)
		if restore is None:
			raise KeyError(f"unknown restore run: {restore_id}")
		if restore.status != "pending_review":
			return restore.to_dict()
		approved = RestoreRun(
			id=restore.id,
			tenant_id=restore.tenant_id,
			snapshot_id=restore.snapshot_id,
			target_environment=restore.target_environment,
			requested_by=restore.requested_by,
			status="completed",
			integrity_check_passed=restore.integrity_check_passed,
			approval_recorded=restore.approval_recorded,
			point_in_time=restore.point_in_time,
			review_status="approved",
			rto_minutes=restore.rto_minutes,
		)
		self._restores[restore_id] = approved
		self._record_audit(
			tenant_id=approved.tenant_id,
			subject_id=restore_id,
			event_type="restore_review_approved",
			actor=reviewer,
			decision="allow",
		)
		return approved.to_dict()

	def list_restores(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		restores = list(self._restores.values())
		if tenant_id is not None:
			restores = [item for item in restores if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(restores, key=lambda item: item.id)]

	def record_restore_test(
		self,
		report_id: str,
		tenant_id: str,
		plan_id: str,
		rto_minutes: int,
		days_since_restore_test: int = 0,
		restore_test_review_recorded: bool = True,
	) -> dict[str, Any]:
		plan = self._require_plan(plan_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"days_since_restore_test": int(days_since_restore_test),
			"restore_test_review_recorded": bool(restore_test_review_recorded),
		})
		self._raise_if_denied(result)
		review_status = "required" if result["decision"] == "require_review" else "current"
		findings = self._engine.continuity_findings(
			rpo_minutes=plan.rpo_minutes,
			rpo_target_minutes=self.describe(tenant_id)["configuration"]["plans"]["rpo_minutes"],
			rto_minutes=int(rto_minutes),
			rto_target_minutes=self.describe(tenant_id)["configuration"]["restore"]["rto_minutes"],
			days_since_restore_test=int(days_since_restore_test),
		)
		report = ContinuityReport(
			id=report_id,
			tenant_id=tenant_id,
			plan_id=plan_id,
			rpo_minutes=plan.rpo_minutes,
			rto_minutes=int(rto_minutes),
			restore_test_status="review_required" if review_status == "required" else "passed",
			days_since_restore_test=int(days_since_restore_test),
			review_status=review_status,
			findings=findings,
		)
		self._reports[report_id] = report
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=report_id,
			event_type="restore_test_recorded",
			actor=plan.owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"plan_id": plan_id, "findings": list(findings)},
		)
		return report.to_dict()

	def list_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		reports = list(self._reports.values())
		if tenant_id is not None:
			reports = [item for item in reports if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(reports, key=lambda item: item.id)]

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._audit_events.values())
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(events, key=lambda item: item.id)]

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing snapshots as BKUP records."""
		return self.list_snapshots(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records an auditable backup event."""
		self._require_tenant(tenant_id)
		metadata = dict(metadata or {})
		event = self._record_audit(
			tenant_id=tenant_id,
			subject_id=record_id,
			event_type=str(metadata.get("event_type") or "backup_note"),
			actor=str(metadata.get("actor") or "system"),
			decision=status,
			metadata=metadata,
		)
		return event.to_dict()

	def continuity_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		restores = self.list_restores(tenant_id)
		reports = self.list_reports(tenant_id)
		return {
			"plan_count": len(self.list_plans(tenant_id)),
			"snapshot_count": len(self.list_snapshots(tenant_id)),
			"restore_count": len(restores),
			"completed_restore_count": len([item for item in restores if item["status"] == "completed"]),
			"pending_review_count": len([item for item in restores if item["status"] == "pending_review"]),
			"continuity_report_count": len(reports),
			"review_required_report_count": len([item for item in reports if item["review_status"] == "required"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_plan(self, plan_id: str, tenant_id: str) -> BackupPlan:
		plan = self._plans.get(plan_id)
		if plan is None or plan.tenant_id != tenant_id:
			raise KeyError(f"unknown backup plan: {plan_id}")
		return plan

	def _require_snapshot(self, snapshot_id: str, tenant_id: str) -> BackupSnapshot:
		snapshot = self._snapshots.get(snapshot_id)
		if snapshot is None or snapshot.tenant_id != tenant_id:
			raise KeyError(f"unknown backup snapshot: {snapshot_id}")
		return snapshot

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> BackupAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = BackupAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "backup_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "backup_policy_blocked")
