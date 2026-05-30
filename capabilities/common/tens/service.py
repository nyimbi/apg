"""Service layer for the Tenants Legacy capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_TENS_AGENT_ROLES,
	SUPPORTED_TENS_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .tenant_runtime import (
	AccessBoundaryRecord,
	DeprecationPlanRecord,
	LegacyTenantRecord,
	MigrationPlanRecord,
	TensAgentRecord,
	TenantAuditEventRecord,
	TenantMappingRecord,
	stable_id,
	tenant_required_actions,
	utc_now,
)


class TensService:
	"""Deterministic legacy tenant service for APG composition."""

	def __init__(self) -> None:
		self.legacy_tenants: dict[str, LegacyTenantRecord] = {}
		self.mappings: dict[str, TenantMappingRecord] = {}
		self.boundaries: dict[str, AccessBoundaryRecord] = {}
		self.migrations: dict[str, MigrationPlanRecord] = {}
		self.deprecations: dict[str, DeprecationPlanRecord] = {}
		self.audit_events: dict[str, TenantAuditEventRecord] = {}
		self.tens_agents: dict[str, TensAgentRecord] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_legacy_tenant(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		source_system: str,
		owner: str,
		compatibility_scope: str,
		days_since_activity: int = 0,
		stale_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not str(legacy_tenant_id or "").strip():
			raise ValueError("legacy_tenant_id_required")
		context = {
			"tenant_context_present": True,
			"operation": "register_legacy_tenant",
			"legacy_owner_assigned": bool(str(owner or "").strip()),
			"source_system_present": bool(str(source_system or "").strip()),
			"compatibility_scope_present": bool(str(compatibility_scope or "").strip()),
			"days_since_activity": int(days_since_activity),
			"stale_review_recorded": bool(stale_review_recorded),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		status = "stale" if result["decision"] == "require_review" else "active"
		record = LegacyTenantRecord(
			id=stable_id("tens_legacy", tenant_id, legacy_tenant_id),
			tenant_id=tenant_id,
			legacy_tenant_id=legacy_tenant_id,
			source_system=source_system,
			owner=owner,
			compatibility_scope=compatibility_scope,
			status=status,
			days_since_activity=int(days_since_activity),
			required_actions=tenant_required_actions(result),
		)
		self.legacy_tenants[record.id] = record
		self._record_event(
			tenant_id,
			"legacy_tenant_registered",
			record.id,
			f"Legacy tenant registered: {legacy_tenant_id}",
			owner,
			"low",
			{"event_stream": event_stream_name(), "source_system": source_system},
		)
		return record.to_dict()

	def map_tenant(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		apg_tenant_id: str,
		validated_by: str,
		validation_ref: str,
		mapping_validated: bool = True,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not str(apg_tenant_id or "").strip():
			raise ValueError("apg_tenant_id_required")
		context = {
			"tenant_context_present": True,
			"operation": "map_tenant",
			"mapping_validated": bool(mapping_validated and str(validation_ref or "").strip()),
			"event_stream": str(event_stream or "").strip().lower(),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = TenantMappingRecord(
			id=stable_id("tens_mapping", tenant_id, legacy.id, apg_tenant_id),
			tenant_id=tenant_id,
			legacy_tenant_id=legacy.id,
			apg_tenant_id=apg_tenant_id,
			validated_by=validated_by,
			status="validated",
			validation_ref=validation_ref,
		)
		self.mappings[record.id] = record
		legacy.status = "mapped"
		legacy.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"tenant_mapped",
			record.id,
			f"Legacy tenant mapped to APG tenant: {apg_tenant_id}",
			validated_by,
			"medium",
			{"event_stream": event_stream_name(), "apg_tenant_id": apg_tenant_id},
		)
		return record.to_dict()

	def validate_access_boundary(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		auth_boundary_ref: str,
		role_mapping_ref: str,
		isolation_validation_ref: str,
		privileged_review_ref: str,
		actor: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		context = {
			"tenant_context_present": True,
			"operation": "validate_access_boundary",
			"auth_boundary_validated": bool(str(auth_boundary_ref or "").strip()),
			"role_mapping_present": bool(str(role_mapping_ref or "").strip()),
			"isolation_validation_present": bool(str(isolation_validation_ref or "").strip()),
			"privileged_review_present": bool(str(privileged_review_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = AccessBoundaryRecord(
			id=stable_id("tens_boundary", tenant_id, legacy.id),
			tenant_id=tenant_id,
			legacy_tenant_id=legacy.id,
			auth_boundary_ref=auth_boundary_ref,
			role_mapping_ref=role_mapping_ref,
			isolation_validation_ref=isolation_validation_ref,
			privileged_review_ref=privileged_review_ref,
			status="validated",
			actor=actor,
		)
		self.boundaries[record.id] = record
		self._record_event(
			tenant_id,
			"boundary_validated",
			record.id,
			f"Access boundary validated: {legacy.legacy_tenant_id}",
			actor,
			"medium",
			{"event_stream": event_stream_name()},
		)
		return record.to_dict()

	def create_migration_plan(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		mapping_id: str,
		owner: str,
		approval_ref: str,
		rollback_plan_ref: str,
		post_migration_validation_ref: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		mapping = self._get_mapping(tenant_id, mapping_id)
		self._get_boundary(tenant_id, legacy.id)
		if mapping.legacy_tenant_id != legacy.id:
			raise PermissionError("mapping_does_not_match_legacy_tenant")
		if not str(post_migration_validation_ref or "").strip():
			raise PermissionError("post_migration_validation_required")
		context = {
			"tenant_context_present": True,
			"operation": "migrate_tenant",
			"approval_recorded": bool(str(approval_ref or "").strip()),
			"rollback_plan_present": bool(str(rollback_plan_ref or "").strip()),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		record = MigrationPlanRecord(
			id=stable_id("tens_migration", tenant_id, legacy.id, mapping.id),
			tenant_id=tenant_id,
			legacy_tenant_id=legacy.id,
			mapping_id=mapping.id,
			owner=owner,
			approval_ref=approval_ref,
			rollback_plan_ref=rollback_plan_ref,
			post_migration_validation_ref=post_migration_validation_ref,
			status="approved",
		)
		self.migrations[record.id] = record
		legacy.status = "migration_ready"
		legacy.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"migration_plan_created",
			record.id,
			f"Migration plan approved: {legacy.legacy_tenant_id}",
			owner,
			"medium",
			{"event_stream": event_stream_name(), "mapping_id": mapping.id},
		)
		return record.to_dict()

	def complete_migration(
		self,
		tenant_id: str,
		migration_id: str,
		actor: str,
		post_migration_validation_ref: str,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		migration = self._get_migration(tenant_id, migration_id)
		context = {
			"tenant_context_present": True,
			"operation": "complete_migration",
			"post_migration_validation_present": bool(str(post_migration_validation_ref or "").strip()),
			"event_stream": str(event_stream or "").strip().lower(),
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		migration.status = "completed"
		migration.post_migration_validation_ref = post_migration_validation_ref
		migration.completed_at = utc_now()
		legacy = self._get_legacy_tenant(tenant_id, migration.legacy_tenant_id)
		legacy.status = "migrated"
		legacy.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"migration_completed",
			migration.id,
			f"Legacy tenant migrated: {legacy.legacy_tenant_id}",
			actor,
			"medium",
			{"event_stream": event_stream_name(), "post_migration_validation_ref": post_migration_validation_ref},
		)
		return migration.to_dict()

	def record_deprecation_plan(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		owner: str,
		deprecation_ref: str,
		target_date: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not str(deprecation_ref or "").strip():
			raise PermissionError("deprecation_plan_required")
		record = DeprecationPlanRecord(
			id=stable_id("tens_deprecation", tenant_id, legacy.id),
			tenant_id=tenant_id,
			legacy_tenant_id=legacy.id,
			owner=owner,
			deprecation_ref=deprecation_ref,
			target_date=target_date,
		)
		self.deprecations[record.id] = record
		legacy.status = "deprecated"
		legacy.updated_at = utc_now()
		self._record_event(
			tenant_id,
			"deprecation_planned",
			record.id,
			f"Deprecation plan recorded: {legacy.legacy_tenant_id}",
			owner,
			"medium",
			{"event_stream": event_stream_name(), "target_date": target_date},
		)
		return record.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_legacy_tenant(
			tenant_id=tenant_id,
			legacy_tenant_id=record_id,
			source_system=str(metadata.get("source_system") or "legacy"),
			owner=str(metadata.get("owner") or "compatibility-owner"),
			compatibility_scope=str(metadata.get("compatibility_scope") or status),
			days_since_activity=int(metadata.get("days_since_activity", 0)),
			stale_review_recorded=bool(metadata.get("stale_review_recorded", False)),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_legacy_tenants(tenant_id)

	def list_legacy_tenants(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.legacy_tenants, tenant_id)

	def list_mappings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.mappings, tenant_id)

	def list_boundaries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.boundaries, tenant_id)

	def list_migrations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.migrations, tenant_id)

	def list_deprecations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.deprecations, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.audit_events, tenant_id)

	def register_tens_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str = "tenant-admin",
		human_approval_required: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		runtime_value = self._normalize_token(runtime)
		role_value = self._normalize_token(role)
		context = {
			"tenant_context_present": True,
			"operation": "register_tens_agent",
			"agent_runtime_supported": runtime_value in SUPPORTED_TENS_AGENT_RUNTIMES,
			"agent_role_supported": role_value in SUPPORTED_TENS_AGENT_ROLES,
		}
		result = self.evaluate(context)
		if result["decision"] == "deny":
			self._raise_policy(result)
		if not str(name or "").strip():
			raise ValueError("tens_agent_name_required")
		if not str(scope or "").strip():
			raise ValueError("tens_agent_scope_required")
		record = TensAgentRecord(
			id=stable_id("tens_agent", tenant_id, name, runtime_value, role_value),
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=scope,
			owner=owner,
			human_approval_required=bool(human_approval_required),
		)
		self.tens_agents[record.id] = record
		self._record_event(
			tenant_id,
			"tens_agent_registered",
			record.id,
			f"TENS agent registered: {name}",
			owner,
			"low",
			{"runtime": runtime_value, "role": role_value, "event_stream": event_stream_name()},
		)
		return record.to_dict()

	def validate_agent_tenant_action(
		self,
		tenant_id: str,
		agent_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		agent = self.tens_agents.get(agent_id)
		if agent is None or agent.tenant_id != tenant_id:
			raise KeyError(f"tens_agent_not_found:{agent_id}")
		context = {
			"tenant_context_present": True,
			"operation": "agent_tenant_action",
			"privileged_scope": bool(privileged_scope),
			"human_approval_recorded": bool(human_approval_recorded),
		}
		return self.evaluate(context)

	def validate_batch_tenant_mapping(
		self,
		tenant_id: str,
		legacy_tenant_ids: list[str],
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not legacy_tenant_ids:
			raise ValueError("batch_tenant_mapping_targets_required")
		context = {
			"tenant_context_present": True,
			"operation": "batch_tenant_mapping",
			"event_stream": str(event_stream or "").strip().lower(),
		}
		return self.evaluate(context)

	def list_tens_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self.tens_agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		legacy_tenants = self.list_legacy_tenants(tenant_id)
		return {
			"tenant_id": tenant_id,
			"legacy_tenant_count": len(legacy_tenants),
			"stale_tenant_count": sum(1 for item in legacy_tenants if item["status"] == "stale"),
			"mapped_tenant_count": len(self.list_mappings(tenant_id)),
			"validated_boundary_count": len(self.list_boundaries(tenant_id)),
			"migration_count": len(self.list_migrations(tenant_id)),
			"completed_migration_count": sum(1 for item in self.list_migrations(tenant_id) if item["status"] == "completed"),
			"deprecation_count": len(self.list_deprecations(tenant_id)),
			"tens_agent_count": len(self.list_tens_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"recent_events": self.list_audit_events(tenant_id)[-5:],
			"streaming": streaming_manifest(),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		if not str(tenant_id or "").strip():
			self._raise_policy(self.evaluate({"tenant_context_present": False}))

	def _raise_policy(self, result: dict[str, Any]) -> None:
		reasons = ", ".join(action.get("reason", "tens_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "tens_policy_blocked")

	def _get_legacy_tenant(self, tenant_id: str, legacy_tenant_id: str) -> LegacyTenantRecord:
		legacy = self.legacy_tenants.get(legacy_tenant_id)
		if legacy is None:
			legacy = next((item for item in self.legacy_tenants.values() if item.tenant_id == tenant_id and item.legacy_tenant_id == legacy_tenant_id), None)
		if legacy is None or legacy.tenant_id != tenant_id:
			raise KeyError(f"legacy_tenant_not_found:{legacy_tenant_id}")
		return legacy

	def _get_mapping(self, tenant_id: str, mapping_id: str) -> TenantMappingRecord:
		mapping = self.mappings.get(mapping_id)
		if mapping is None or mapping.tenant_id != tenant_id:
			raise KeyError(f"tenant_mapping_not_found:{mapping_id}")
		return mapping

	def _get_boundary(self, tenant_id: str, legacy_tenant_id: str) -> AccessBoundaryRecord:
		boundary = self.boundaries.get(stable_id("tens_boundary", tenant_id, legacy_tenant_id))
		if boundary is None or boundary.tenant_id != tenant_id:
			raise PermissionError("auth_boundary_required")
		return boundary

	def _get_migration(self, tenant_id: str, migration_id: str) -> MigrationPlanRecord:
		migration = self.migrations.get(migration_id)
		if migration is None or migration.tenant_id != tenant_id:
			raise KeyError(f"tenant_migration_not_found:{migration_id}")
		return migration

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		record = TenantAuditEventRecord(
			id=stable_id("tens_event", tenant_id, event_type, subject_id, len(self.audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self.audit_events[record.id] = record
		return record.to_dict()

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = [record.to_dict() for record in records.values()]
		if tenant_id is not None:
			items = [item for item in items if item["tenant_id"] == tenant_id]
		return sorted(items, key=lambda item: item["id"])

	def _normalize_token(self, value: str) -> str:
		return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
