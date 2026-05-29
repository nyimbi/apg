"""Service layer for APG Backup and Restore."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .backup_engine import BackupEngine
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	BackupAuditEvent,
	BackupPlan,
	BackupSnapshot,
	ContinuityReport,
	RestoreApproval,
	RestoreRun,
	RetentionDisposition,
)


class BkupService:
	"""Backup plan registry, snapshot vault, restore approvals, retention, and reports."""

	def __init__(self) -> None:
		self._plans: dict[tuple[str, str], BackupPlan] = {}
		self._snapshots: dict[tuple[str, str], BackupSnapshot] = {}
		self._restores: dict[tuple[str, str], RestoreRun] = {}
		self._restore_approvals: dict[tuple[str, str], RestoreApproval] = {}
		self._retention_dispositions: dict[tuple[str, str], RetentionDisposition] = {}
		self._reports: dict[tuple[str, str], ContinuityReport] = {}
		self._audit_events: dict[tuple[str, str], BackupAuditEvent] = {}
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
		self._ensure_new(self._plans, tenant_id, plan_id, "backup plan")
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
		self._plans[self._tenant_key(tenant_id, plan_id)] = plan
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=plan_id,
			event_type="backup_plan_created",
			actor=owner,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"source_count": len(sources), "schedule": schedule, "legal_hold": legal_hold},
		)
		return plan.to_dict()

	def list_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._plans.values(), tenant_id)

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
		self._ensure_new(self._snapshots, tenant_id, snapshot_id, "snapshot")
		plan = self._require_plan(plan_id, tenant_id)
		if source_id not in plan.sources:
			raise KeyError(f"source not in backup plan: {source_id}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_snapshot",
			"snapshot_encrypted": bool(encrypted),
			"snapshot_integrity_passed": bool(integrity_check_passed),
		})
		self._raise_if_denied(result)
		if not integrity_check_passed:
			raise PermissionError("snapshot_integrity_check_required")
		if size_bytes <= 0:
			raise PermissionError("snapshot_size_required")
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
		self._snapshots[self._tenant_key(tenant_id, snapshot_id)] = snapshot
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=snapshot_id,
			event_type="snapshot_created",
			actor=plan.owner,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"plan_id": plan_id, "source_id": source_id, "region": region},
		)
		return snapshot.to_dict()

	def list_snapshots(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._snapshots.values(), tenant_id)

	def request_restore_approval(
		self,
		approval_id: str,
		tenant_id: str,
		snapshot_id: str,
		target_environment: str,
		requested_by: str,
		justification: str,
		point_in_time: str | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._restore_approvals, tenant_id, approval_id, "restore approval")
		snapshot = self._require_snapshot(snapshot_id, tenant_id)
		if snapshot.status != "available":
			raise PermissionError("snapshot_not_available")
		if not requested_by:
			raise ValueError("restore_approval_requester_required")
		if not justification:
			raise ValueError("restore_approval_justification_required")
		approval = RestoreApproval(
			id=approval_id,
			tenant_id=tenant_id,
			snapshot_id=snapshot_id,
			target_environment=target_environment,
			requested_by=requested_by,
			justification=justification,
			point_in_time=point_in_time,
		)
		self._restore_approvals[self._tenant_key(tenant_id, approval_id)] = approval
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="restore_approval_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"snapshot_id": snapshot_id, "target_environment": target_environment},
		)
		return approval.to_dict()

	def decide_restore_approval(
		self,
		approval_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		approval = self._require_restore_approval(approval_id, tenant_id)
		if approval.status != "pending":
			raise ValueError("restore_approval_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("restore_approval_decision_invalid")
		if not reviewer:
			raise ValueError("restore_approval_reviewer_required")
		if not notes:
			raise ValueError("restore_approval_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_restore",
			"restore_reviewer_same_as_requester": reviewer == approval.requested_by,
		})
		self._raise_if_denied(result)
		decided = replace(approval, reviewer=reviewer, decision=decision, notes=notes, status=decision)
		self._restore_approvals[self._tenant_key(tenant_id, approval_id)] = decided
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="restore_approval_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"snapshot_id": approval.snapshot_id, "target_environment": approval.target_environment},
		)
		return decided.to_dict()

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
		approval_id: str | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._restores, tenant_id, restore_id, "restore")
		snapshot = self._require_snapshot(snapshot_id, tenant_id)
		if snapshot.status != "available":
			raise PermissionError("snapshot_not_available")
		approved_restore = self._approved_restore_approval(
			tenant_id=tenant_id,
			approval_id=approval_id,
			snapshot_id=snapshot_id,
			target_environment=target_environment,
			point_in_time=point_in_time,
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "restore",
			"integrity_check_passed": bool(integrity_check_passed),
			"target_environment": target_environment,
			"approval_recorded": approved_restore is not None,
			"days_since_restore_test": int(days_since_restore_test),
			"restore_test_review_recorded": False if int(days_since_restore_test) > 90 else bool(restore_test_review_recorded),
		})
		self._raise_if_denied(result)
		if not requested_by:
			raise ValueError("restore_requester_required")
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
			approval_recorded=approved_restore is not None,
			approval_id=approved_restore.id if approved_restore else "",
			point_in_time=point_in_time,
			review_status=review_status,
			rto_minutes=int(rto_minutes),
		)
		self._restores[self._tenant_key(tenant_id, restore_id)] = restore
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=restore_id,
			event_type="restore_requested",
			actor=requested_by,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"snapshot_id": snapshot.id, "target_environment": target_environment, "approval_id": approval_id},
		)
		return restore.to_dict()

	def approve_restore(self, restore_id: str, reviewer: str, tenant_id: str | None = None, notes: str = "Approved restore review.") -> dict[str, Any]:
		restore = self._require_restore(restore_id, tenant_id)
		if restore.status != "pending_review":
			return restore.to_dict()
		if reviewer == restore.requested_by:
			raise PermissionError("independent_restore_reviewer_required")
		if not notes:
			raise ValueError("restore_review_notes_required")
		approved = replace(
			restore,
			status="completed",
			review_status="approved",
			reviewer=reviewer,
			reviewer_notes=notes,
		)
		self._restores[self._tenant_key(restore.tenant_id, restore_id)] = approved
		self._record_audit(
			tenant_id=approved.tenant_id,
			subject_id=restore_id,
			event_type="restore_review_approved",
			actor=reviewer,
			decision="allow",
			metadata={"notes": notes},
		)
		return approved.to_dict()

	def list_restore_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._restore_approvals.values(), tenant_id)

	def list_restores(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._restores.values(), tenant_id)

	def record_restore_test(
		self,
		report_id: str,
		tenant_id: str,
		plan_id: str,
		rto_minutes: int,
		days_since_restore_test: int = 0,
		restore_test_review_recorded: bool = True,
	) -> dict[str, Any]:
		self._ensure_new(self._reports, tenant_id, report_id, "continuity report")
		plan = self._require_plan(plan_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"days_since_restore_test": int(days_since_restore_test),
			"restore_test_review_recorded": False if int(days_since_restore_test) > 90 else bool(restore_test_review_recorded),
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
		self._reports[self._tenant_key(tenant_id, report_id)] = report
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=report_id,
			event_type="restore_test_recorded",
			actor=plan.owner,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"plan_id": plan_id, "findings": list(findings)},
		)
		return report.to_dict()

	def list_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reports.values(), tenant_id)

	def request_retention_disposition(
		self,
		disposition_id: str,
		tenant_id: str,
		snapshot_id: str,
		action: str,
		requested_by: str,
		reason: str,
	) -> dict[str, Any]:
		self._ensure_new(self._retention_dispositions, tenant_id, disposition_id, "retention disposition")
		snapshot = self._require_snapshot(snapshot_id, tenant_id)
		plan = self._require_plan(snapshot.plan_id, tenant_id)
		normalized_action = action.strip().lower()
		if normalized_action not in {"delete", "archive"}:
			raise ValueError("retention_disposition_action_invalid")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retention_disposition",
			"legal_hold_active": bool(plan.legal_hold),
		})
		self._raise_if_denied(result)
		if snapshot.status != "available":
			raise PermissionError("snapshot_not_available")
		if not requested_by:
			raise ValueError("retention_disposition_requester_required")
		if not reason:
			raise ValueError("retention_disposition_reason_required")
		disposition = RetentionDisposition(
			id=disposition_id,
			tenant_id=tenant_id,
			snapshot_id=snapshot_id,
			action=normalized_action,
			requested_by=requested_by,
			reason=reason,
		)
		self._retention_dispositions[self._tenant_key(tenant_id, disposition_id)] = disposition
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=disposition_id,
			event_type="retention_disposition_requested",
			actor=requested_by,
			decision="require_review",
			reasons=self._reasons(result),
			metadata={"snapshot_id": snapshot_id, "action": normalized_action},
		)
		return disposition.to_dict()

	def decide_retention_disposition(
		self,
		disposition_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		disposition = self._require_retention_disposition(disposition_id, tenant_id)
		snapshot = self._require_snapshot(disposition.snapshot_id, tenant_id)
		if disposition.status != "pending":
			raise ValueError("retention_disposition_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("retention_disposition_decision_invalid")
		if not reviewer:
			raise ValueError("retention_disposition_reviewer_required")
		if not notes:
			raise ValueError("retention_disposition_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_retention_disposition",
			"retention_reviewer_same_as_requester": reviewer == disposition.requested_by,
		})
		self._raise_if_denied(result)
		decided = replace(disposition, reviewer=reviewer, decision=decision, notes=notes, status=decision)
		self._retention_dispositions[self._tenant_key(tenant_id, disposition_id)] = decided
		if decision == "approved":
			new_status = "deleted" if disposition.action == "delete" else "archived"
			self._snapshots[self._tenant_key(tenant_id, snapshot.id)] = replace(snapshot, status=new_status)
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=disposition_id,
			event_type="retention_disposition_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"snapshot_id": snapshot.id, "action": disposition.action},
		)
		return decided.to_dict()

	def list_retention_dispositions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._retention_dispositions.values(), tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events.values(), tenant_id)

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
			"available_snapshot_count": len([item for item in self.list_snapshots(tenant_id) if item["status"] == "available"]),
			"restore_count": len(restores),
			"completed_restore_count": len([item for item in restores if item["status"] == "completed"]),
			"pending_review_count": len([item for item in restores if item["status"] == "pending_review"]),
			"restore_approval_count": len(self.list_restore_approvals(tenant_id)),
			"pending_restore_approval_count": len([item for item in self.list_restore_approvals(tenant_id) if item["status"] == "pending"]),
			"retention_disposition_count": len(self.list_retention_dispositions(tenant_id)),
			"pending_retention_disposition_count": len([item for item in self.list_retention_dispositions(tenant_id) if item["status"] == "pending"]),
			"continuity_report_count": len(reports),
			"review_required_report_count": len([item for item in reports if item["review_status"] == "required"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _ensure_new(self, records: dict[tuple[str, str], Any], tenant_id: str, record_id: str, label: str) -> None:
		self._require_tenant(tenant_id)
		if not record_id:
			raise ValueError(f"{label}_id_required")
		if self._tenant_key(tenant_id, record_id) in records:
			raise ValueError(f"{label} already exists for tenant: {record_id}")

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_plan(self, plan_id: str, tenant_id: str) -> BackupPlan:
		plan = self._plans.get(self._tenant_key(tenant_id, plan_id))
		if plan is None:
			raise KeyError(f"unknown backup plan: {plan_id}")
		return plan

	def _require_snapshot(self, snapshot_id: str, tenant_id: str) -> BackupSnapshot:
		snapshot = self._snapshots.get(self._tenant_key(tenant_id, snapshot_id))
		if snapshot is None:
			raise KeyError(f"unknown backup snapshot: {snapshot_id}")
		return snapshot

	def _require_restore(self, restore_id: str, tenant_id: str | None = None) -> RestoreRun:
		if tenant_id is not None:
			restore = self._restores.get(self._tenant_key(tenant_id, restore_id))
			if restore is None:
				raise KeyError(f"unknown restore run: {restore_id}")
			return restore
		matches = [restore for (_, item_id), restore in self._restores.items() if item_id == restore_id]
		if len(matches) > 1:
			raise KeyError(f"restore ID is ambiguous across tenants: {restore_id}")
		if not matches:
			raise KeyError(f"unknown restore run: {restore_id}")
		return matches[0]

	def _require_restore_approval(self, approval_id: str, tenant_id: str) -> RestoreApproval:
		approval = self._restore_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise KeyError(f"unknown restore approval: {approval_id}")
		return approval

	def _require_retention_disposition(self, disposition_id: str, tenant_id: str) -> RetentionDisposition:
		disposition = self._retention_dispositions.get(self._tenant_key(tenant_id, disposition_id))
		if disposition is None:
			raise KeyError(f"unknown retention disposition: {disposition_id}")
		return disposition

	def _approved_restore_approval(
		self,
		tenant_id: str,
		approval_id: str | None,
		snapshot_id: str,
		target_environment: str,
		point_in_time: str | None,
	) -> RestoreApproval | None:
		if target_environment != "production":
			return None
		if approval_id is None:
			return None
		approval = self._restore_approvals.get(self._tenant_key(tenant_id, approval_id))
		if approval is None:
			raise PermissionError("production_restore_approval_required")
		if (
			approval.snapshot_id != snapshot_id
			or approval.target_environment != target_environment
			or approval.point_in_time != point_in_time
		):
			raise PermissionError("restore_approval_mismatch")
		if approval.status != "approved":
			raise PermissionError("restore_approval_not_approved")
		return approval

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
		self._audit_events[self._tenant_key(tenant_id, event_id)] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(self._reasons(result))
			raise PermissionError(reasons or "backup_policy_blocked")

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			str(action.get("reason") or action.get("required_action") or "backup_policy_blocked")
			for action in result.get("actions", [])
		)

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
