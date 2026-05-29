"""Dependency-light MTEN lifecycle runtime for package composition."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from uuid_extensions import uuid7str

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	CapacityApprovalRecord,
	IsolationIncidentRecord,
	LiveMigrationRecord,
	TenantEnvironmentRecord,
	TenantGovernanceEvent,
)


class MtenService:
	"""Tenant provisioning, isolation, capacity, and migration governance facade."""

	def __init__(self) -> None:
		self._tenants: dict[tuple[str, str], TenantEnvironmentRecord] = {}
		self._capacity_approvals: dict[tuple[str, str], CapacityApprovalRecord] = {}
		self._isolation_incidents: dict[tuple[str, str], IsolationIncidentRecord] = {}
		self._migrations: dict[tuple[str, str], LiveMigrationRecord] = {}
		self._governance_events: dict[tuple[str, str], TenantGovernanceEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_tenant(
		self,
		target_tenant_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		tier: str = "free",
		primary_domain: str = "",
		custom_domain: str = "",
		dns_validated: bool = False,
		projected_compute_units: int = 0,
		isolation_boundary_encrypted: bool = True,
		capacity_approval_id: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._ensure_new(self._tenants, tenant_id, target_tenant_id, "tenant")
		if not name:
			raise ValueError("tenant_name_required")
		if not owner:
			raise ValueError("tenant_owner_required")
		if not primary_domain:
			raise ValueError("primary_domain_required")
		approved_capacity = self._approved_capacity_approval(
			tenant_id=tenant_id,
			target_tenant_id=target_tenant_id,
			approval_id=capacity_approval_id,
			projected_compute_units=projected_compute_units,
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_tenant",
			"custom_domain_requested": bool(custom_domain),
			"dns_validated": bool(dns_validated),
			"projected_compute_units": int(projected_compute_units),
			"capacity_approval_recorded": approved_capacity is not None,
			"isolation_boundary_encrypted": bool(isolation_boundary_encrypted),
		})
		_raise_if_denied(result)
		if result["decision"] == "require_review":
			raise PermissionError("capacity_review_required")
		record = TenantEnvironmentRecord(
			id=target_tenant_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			tier=tier,
			primary_domain=primary_domain,
			custom_domain=custom_domain,
			dns_validated=dns_validated,
			projected_compute_units=int(projected_compute_units),
			isolation_boundary_encrypted=isolation_boundary_encrypted,
			capacity_approval_id=approved_capacity.id if approved_capacity else "",
			metadata=dict(metadata or {}),
		)
		self._tenants[self._tenant_key(tenant_id, target_tenant_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=target_tenant_id,
			event_type="tenant_registered",
			actor=owner,
			decision=result["decision"],
			reasons=self._reasons(result),
			metadata={"tier": tier, "custom_domain": custom_domain},
		)
		return record.to_dict()

	def activate_tenant(
		self,
		target_tenant_id: str,
		tenant_id: str,
		actor: str,
		dns_validated: bool | None = None,
	) -> dict[str, Any]:
		record = self._require_tenant(tenant_id, target_tenant_id)
		effective_dns = record.dns_validated if dns_validated is None else bool(dns_validated)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "activate_tenant",
			"tenant_status": record.status,
			"requested_operation_is_mutation": True,
			"custom_domain_requested": bool(record.custom_domain),
			"dns_validated": effective_dns,
			"isolation_boundary_encrypted": record.isolation_boundary_encrypted,
		})
		_raise_if_denied(result)
		activated = replace(record, status="active", dns_validated=effective_dns)
		self._tenants[self._tenant_key(tenant_id, target_tenant_id)] = activated
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=target_tenant_id,
			event_type="tenant_activated",
			actor=actor,
			decision=result["decision"],
			reasons=self._reasons(result),
		)
		return activated.to_dict()

	def request_capacity_approval(
		self,
		approval_id: str,
		tenant_id: str,
		target_tenant_id: str,
		requested_by: str,
		projected_compute_units: int,
		justification: str,
	) -> dict[str, Any]:
		self._ensure_new(self._capacity_approvals, tenant_id, approval_id, "capacity approval")
		if not requested_by:
			raise ValueError("capacity_requester_required")
		if not justification:
			raise ValueError("capacity_justification_required")
		record = CapacityApprovalRecord(
			id=approval_id,
			tenant_id=tenant_id,
			target_tenant_id=target_tenant_id,
			requested_by=requested_by,
			projected_compute_units=int(projected_compute_units),
			justification=justification,
		)
		self._capacity_approvals[self._tenant_key(tenant_id, approval_id)] = record
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="capacity_approval_requested",
			actor=requested_by,
			decision="require_review",
			metadata={"target_tenant_id": target_tenant_id, "projected_compute_units": projected_compute_units},
		)
		return record.to_dict()

	def decide_capacity_approval(
		self,
		approval_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		record = self._require_capacity_approval(tenant_id, approval_id)
		if record.status != "pending":
			raise ValueError("capacity_approval_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("capacity_approval_decision_invalid")
		if not reviewer:
			raise ValueError("capacity_reviewer_required")
		if not notes:
			raise ValueError("capacity_reviewer_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_capacity",
			"capacity_reviewer_same_as_requester": reviewer == record.requested_by,
		})
		_raise_if_denied(result)
		decided = replace(record, status=decision, decision=decision, reviewer=reviewer, notes=notes)
		self._capacity_approvals[self._tenant_key(tenant_id, approval_id)] = decided
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=approval_id,
			event_type="capacity_approval_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"target_tenant_id": record.target_tenant_id},
		)
		return decided.to_dict()

	def record_isolation_incident(
		self,
		incident_id: str,
		tenant_id: str,
		target_tenant_id: str,
		detected_by: str,
		breach_summary: str,
		severity: str = "high",
	) -> dict[str, Any]:
		self._ensure_new(self._isolation_incidents, tenant_id, incident_id, "isolation incident")
		record = self._require_tenant(tenant_id, target_tenant_id)
		if not detected_by:
			raise ValueError("isolation_detector_required")
		if not breach_summary:
			raise ValueError("isolation_breach_summary_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_isolation_incident",
			"isolation_breach_detected": True,
			"tenant_suspended": True,
		})
		_raise_if_denied(result)
		incident = IsolationIncidentRecord(
			id=incident_id,
			tenant_id=tenant_id,
			target_tenant_id=target_tenant_id,
			detected_by=detected_by,
			breach_summary=breach_summary,
			severity=severity,
		)
		self._isolation_incidents[self._tenant_key(tenant_id, incident_id)] = incident
		self._tenants[self._tenant_key(tenant_id, target_tenant_id)] = replace(record, status="suspended")
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=incident_id,
			event_type="isolation_incident_recorded",
			actor=detected_by,
			decision="suspend",
			reasons=self._reasons(result),
			metadata={"target_tenant_id": target_tenant_id, "severity": severity},
		)
		return incident.to_dict()

	def reactivate_tenant(
		self,
		target_tenant_id: str,
		tenant_id: str,
		actor: str,
		evidence: str,
	) -> dict[str, Any]:
		record = self._require_tenant(tenant_id, target_tenant_id)
		if not actor:
			raise ValueError("reactivation_actor_required")
		if not evidence:
			raise ValueError("reactivation_evidence_required")
		active = replace(record, status="active")
		self._tenants[self._tenant_key(tenant_id, target_tenant_id)] = active
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=target_tenant_id,
			event_type="tenant_reactivated",
			actor=actor,
			metadata={"evidence": evidence},
		)
		return active.to_dict()

	def request_live_migration(
		self,
		migration_id: str,
		tenant_id: str,
		target_tenant_id: str,
		requested_by: str,
		source_provider: str,
		target_provider: str,
		runbook: str,
	) -> dict[str, Any]:
		self._ensure_new(self._migrations, tenant_id, migration_id, "live migration")
		record = self._require_tenant(tenant_id, target_tenant_id)
		if not requested_by:
			raise ValueError("migration_requester_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "request_live_migration",
			"requested_operation": "live_migration",
			"tenant_status": record.status,
			"requested_operation_is_mutation": True,
			"runbook_attached": bool(runbook),
		})
		_raise_if_denied(result)
		migration = LiveMigrationRecord(
			id=migration_id,
			tenant_id=tenant_id,
			target_tenant_id=target_tenant_id,
			requested_by=requested_by,
			source_provider=source_provider,
			target_provider=target_provider,
			runbook=runbook,
		)
		self._migrations[self._tenant_key(tenant_id, migration_id)] = migration
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=migration_id,
			event_type="live_migration_requested",
			actor=requested_by,
			decision="require_review",
			reasons=self._reasons(result),
			metadata={"target_tenant_id": target_tenant_id, "target_provider": target_provider},
		)
		return migration.to_dict()

	def decide_live_migration(
		self,
		migration_id: str,
		tenant_id: str,
		reviewer: str,
		decision: str,
		notes: str,
	) -> dict[str, Any]:
		migration = self._require_migration(tenant_id, migration_id)
		if migration.status != "pending":
			raise ValueError("live_migration_already_decided")
		if decision not in {"approved", "rejected"}:
			raise ValueError("live_migration_decision_invalid")
		if not reviewer:
			raise ValueError("migration_reviewer_required")
		if not notes:
			raise ValueError("migration_reviewer_notes_required")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_live_migration",
			"migration_reviewer_same_as_requester": reviewer == migration.requested_by,
		})
		_raise_if_denied(result)
		decided = replace(migration, status=decision, decision=decision, reviewer=reviewer, notes=notes)
		self._migrations[self._tenant_key(tenant_id, migration_id)] = decided
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=migration_id,
			event_type="live_migration_decided",
			actor=reviewer,
			decision=decision,
			reasons=self._reasons(result),
			metadata={"target_tenant_id": migration.target_tenant_id},
		)
		return decided.to_dict()

	def execute_live_migration(self, migration_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
		migration = self._require_migration(tenant_id, migration_id)
		if migration.status != "approved":
			raise PermissionError("live_migration_approval_required")
		record = self._require_tenant(tenant_id, migration.target_tenant_id)
		if record.status == "suspended":
			raise PermissionError("tenant_suspended")
		executed = replace(migration, status="completed")
		self._migrations[self._tenant_key(tenant_id, migration_id)] = executed
		self._record_governance(
			tenant_id=tenant_id,
			subject_id=migration_id,
			event_type="live_migration_executed",
			actor=actor,
			metadata={"target_tenant_id": migration.target_tenant_id, "target_provider": migration.target_provider},
		)
		return executed.to_dict()

	def list_tenants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._tenants.values(), tenant_id)

	def list_capacity_approvals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._capacity_approvals.values(), tenant_id)

	def list_isolation_incidents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._isolation_incidents.values(), tenant_id)

	def list_live_migrations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._migrations.values(), tenant_id)

	def list_governance_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._governance_events.values(), tenant_id)

	def portfolio_summary(self, tenant_id: str | None = None) -> dict[str, int]:
		tenants = self.list_tenants(tenant_id)
		approvals = self.list_capacity_approvals(tenant_id)
		incidents = self.list_isolation_incidents(tenant_id)
		migrations = self.list_live_migrations(tenant_id)
		events = self.list_governance_events(tenant_id)
		return {
			"tenant_count": len(tenants),
			"active_tenant_count": len([item for item in tenants if item["status"] == "active"]),
			"suspended_tenant_count": len([item for item in tenants if item["status"] == "suspended"]),
			"capacity_approval_count": len(approvals),
			"isolation_incident_count": len(incidents),
			"live_migration_count": len(migrations),
			"governance_event_count": len(events),
		}

	def _approved_capacity_approval(
		self,
		tenant_id: str,
		target_tenant_id: str,
		approval_id: str | None,
		projected_compute_units: int,
	) -> CapacityApprovalRecord | None:
		if projected_compute_units <= 1000:
			return None
		if not approval_id:
			return None
		record = self._capacity_approvals.get(self._tenant_key(tenant_id, approval_id))
		if record is None or record.status != "approved":
			return None
		if record.target_tenant_id != target_tenant_id:
			return None
		if record.projected_compute_units < projected_compute_units:
			return None
		return record

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		if not tenant_id:
			raise ValueError("tenant_id_required")
		if not item_id:
			raise ValueError("id_required")
		return tenant_id, item_id

	def _ensure_new(self, store: dict[tuple[str, str], Any], tenant_id: str, item_id: str, label: str) -> None:
		key = self._tenant_key(tenant_id, item_id)
		if key in store:
			raise ValueError(f"duplicate {label}: {item_id}")

	def _require_tenant(self, tenant_id: str, target_tenant_id: str) -> TenantEnvironmentRecord:
		try:
			return self._tenants[self._tenant_key(tenant_id, target_tenant_id)]
		except KeyError as exc:
			raise KeyError(f"tenant not found: {target_tenant_id}") from exc

	def _require_capacity_approval(self, tenant_id: str, approval_id: str) -> CapacityApprovalRecord:
		try:
			return self._capacity_approvals[self._tenant_key(tenant_id, approval_id)]
		except KeyError as exc:
			raise KeyError(f"capacity approval not found: {approval_id}") from exc

	def _require_migration(self, tenant_id: str, migration_id: str) -> LiveMigrationRecord:
		try:
			return self._migrations[self._tenant_key(tenant_id, migration_id)]
		except KeyError as exc:
			raise KeyError(f"live migration not found: {migration_id}") from exc

	def _record_governance(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str = "allow",
		reasons: list[str] | tuple[str, ...] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = uuid7str()
		event = TenantGovernanceEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reasons or ()),
			metadata=dict(metadata or {}),
		)
		self._governance_events[self._tenant_key(tenant_id, event_id)] = event

	def _reasons(self, result: dict[str, Any]) -> list[str]:
		return [
			str(action.get("reason"))
			for action in result.get("actions", [])
			if action.get("reason")
		]

	def _list(self, values: Any, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [
			item.to_dict()
			for item in values
			if tenant_id is None or item.tenant_id == tenant_id
		]


def _raise_if_denied(result: dict[str, Any]) -> None:
	if result.get("decision") == "deny":
		reasons = [
			str(action.get("reason"))
			for action in result.get("actions", [])
			if action.get("reason")
		]
		raise PermissionError(reasons[0] if reasons else "tenant_operation_denied")
