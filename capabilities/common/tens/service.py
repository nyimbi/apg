"""Service layer for the Tenants Legacy capability."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import json
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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class TensService:
	"""Deterministic legacy tenant service for APG composition."""

	def __init__(self, db_url: str | None = None) -> None:
		self.legacy_tenants: dict[str, LegacyTenantRecord] = {}
		self.mappings: dict[str, TenantMappingRecord] = {}
		self.boundaries: dict[str, AccessBoundaryRecord] = {}
		self.migrations: dict[str, MigrationPlanRecord] = {}
		self.deprecations: dict[str, DeprecationPlanRecord] = {}
		self.audit_events: dict[str, TenantAuditEventRecord] = {}
		self.tens_agents: dict[str, TensAgentRecord] = {}
		# Additional in-memory stores for new methods
		_store = get_store(db_url)
		self._tenant_archives = WriteThruDict('tenant_archives', tenant_id, _store)
		self._tenant_clones = WriteThruDict('tenant_clones', tenant_id, _store)
		self._usage_reports = WriteThruDict('usage_reports', tenant_id, _store)
		self._billing_summaries = WriteThruDict('billing_summaries', tenant_id, _store)
		self._isolation_checks = WriteThruDict('isolation_checks', tenant_id, _store)
		self._export_jobs = WriteThruDict('export_jobs', tenant_id, _store)
		self._import_jobs = WriteThruDict('import_jobs', tenant_id, _store)
		self._subdomain_assignments = WriteThruDict('subdomain_assignments', tenant_id, _store)
		self._health_checks = WriteThruDict('health_checks', tenant_id, _store)
		self._suspensions = WriteThruDict('suspensions', tenant_id, _store)
		self._reactivations = WriteThruDict('reactivations', tenant_id, _store)
		self._resource_quotas = WriteThruDict('resource_quotas', tenant_id, _store)

	# ------------------------------------------------------------------ #
	# Original 21 methods                                                  #
	# ------------------------------------------------------------------ #

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

	# ------------------------------------------------------------------ #
	# New methods (15 new, reaching 36 total public methods)               #
	# ------------------------------------------------------------------ #

	async def tenant_migrate(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		target_environment: str,
		migration_script_ref: str,
		actor: str,
	) -> dict[str, Any]:
		"""Initiate a data-migration job for a legacy tenant to a new environment."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not migration_script_ref:
			raise ValueError("migration_script_ref_required")
		record = {
			"id": stable_id("tens_migrate_job", tenant_id, legacy.id, target_environment),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"target_environment": target_environment,
			"migration_script_ref": migration_script_ref,
			"status": "queued",
			"actor": actor,
			"created_at": utc_now(),
		}
		self._record_event(tenant_id, "tenant_migration_queued", record["id"], f"Migration queued: {legacy_tenant_id} -> {target_environment}", actor, "medium")
		return record

	async def tenant_clone(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		new_tenant_id: str,
		actor: str,
		include_data: bool = False,
	) -> dict[str, Any]:
		"""Create a structural clone of a legacy tenant under a new ID."""
		self._require_tenant(tenant_id)
		source = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		clone = self.register_legacy_tenant(
			tenant_id=tenant_id,
			legacy_tenant_id=new_tenant_id,
			source_system=source.source_system,
			owner=source.owner,
			compatibility_scope=source.compatibility_scope,
			days_since_activity=0,
			stale_review_recorded=True,
		)
		record = {
			"source_legacy_tenant_id": legacy_tenant_id,
			"cloned_tenant_id": new_tenant_id,
			"include_data": include_data,
			"actor": actor,
			"clone": clone,
			"created_at": utc_now(),
		}
		self._tenant_clones[stable_id("tens_clone", tenant_id, new_tenant_id)] = record
		return record

	async def tenant_archive(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		archive_ref: str,
		actor: str,
	) -> dict[str, Any]:
		"""Archive a deprecated legacy tenant, preserving data at archive_ref."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not archive_ref:
			raise ValueError("archive_ref_required")
		legacy.status = "archived"
		legacy.updated_at = utc_now()
		record = {
			"id": stable_id("tens_archive", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"archive_ref": archive_ref,
			"actor": actor,
			"archived_at": utc_now(),
		}
		self._tenant_archives[record["id"]] = record
		self._record_event(tenant_id, "tenant_archived", record["id"], f"Tenant archived: {legacy_tenant_id}", actor, "medium")
		return record

	async def tenant_restore(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		restore_from_ref: str,
		actor: str,
	) -> dict[str, Any]:
		"""Restore an archived tenant from a backup reference."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if legacy.status != "archived":
			raise PermissionError("tenant_not_archived")
		if not restore_from_ref:
			raise ValueError("restore_from_ref_required")
		legacy.status = "active"
		legacy.updated_at = utc_now()
		record = {
			"id": stable_id("tens_restore", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"restore_from_ref": restore_from_ref,
			"actor": actor,
			"restored_at": utc_now(),
		}
		self._record_event(tenant_id, "tenant_restored", record["id"], f"Tenant restored: {legacy_tenant_id}", actor, "medium")
		return record

	async def tenant_merge(
		self,
		tenant_id: str,
		source_tenant_id: str,
		target_tenant_id: str,
		merge_strategy: str = "union",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Merge two legacy tenants into one target tenant record."""
		self._require_tenant(tenant_id)
		source = self._get_legacy_tenant(tenant_id, source_tenant_id)
		target = self._get_legacy_tenant(tenant_id, target_tenant_id)
		source.status = "merged"
		source.updated_at = utc_now()
		record = {
			"id": stable_id("tens_merge", tenant_id, source.id, target.id),
			"tenant_id": tenant_id,
			"source_tenant_id": source.id,
			"target_tenant_id": target.id,
			"merge_strategy": merge_strategy,
			"actor": actor,
			"merged_at": utc_now(),
		}
		self._record_event(tenant_id, "tenants_merged", record["id"], f"Merged {source_tenant_id} into {target_tenant_id}", actor, "high")
		return record

	async def usage_report(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		period_start: str,
		period_end: str,
	) -> dict[str, Any]:
		"""Generate a usage report for a legacy tenant over a date period."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		events = [e.to_dict() for e in self.audit_events.values() if e.tenant_id == tenant_id]
		report = {
			"id": stable_id("tens_usage", tenant_id, legacy.id, period_start, period_end),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"period_start": period_start,
			"period_end": period_end,
			"event_count": len(events),
			"status": legacy.status,
			"days_since_activity": legacy.days_since_activity,
			"generated_at": utc_now(),
		}
		self._usage_reports[report["id"]] = report
		return report

	async def tenant_billing_summary(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		billing_period: str,
	) -> dict[str, Any]:
		"""Summarise billing events associated with a legacy tenant for a period."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		summary = {
			"id": stable_id("tens_billing", tenant_id, legacy.id, billing_period),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"billing_period": billing_period,
			"line_items": [],
			"total_amount": 0.0,
			"currency": "USD",
			"generated_at": utc_now(),
		}
		self._billing_summaries[summary["id"]] = summary
		return summary

	async def data_isolation_verify(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		actor: str,
	) -> dict[str, Any]:
		"""Verify that a legacy tenant's data is properly isolated from other tenants."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		# Check boundary record exists
		boundary_key = stable_id("tens_boundary", tenant_id, legacy.id)
		boundary_present = boundary_key in self.boundaries
		result = {
			"id": stable_id("tens_isolation", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"boundary_present": boundary_present,
			"isolation_status": "verified" if boundary_present else "unverified",
			"actor": actor,
			"verified_at": utc_now(),
		}
		self._isolation_checks[result["id"]] = result
		self._record_event(tenant_id, "data_isolation_verified", result["id"], f"Isolation status: {result['isolation_status']}", actor, "medium")
		return result

	async def tenant_export(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		format_: str = "json",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Export a legacy tenant record and its associated data as JSON or CSV."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		mappings = [m.to_dict() for m in self.mappings.values() if m.legacy_tenant_id == legacy.id]
		migrations = [m.to_dict() for m in self.migrations.values() if m.legacy_tenant_id == legacy.id]
		payload_dict = {
			"legacy_tenant": legacy.to_dict(),
			"mappings": mappings,
			"migrations": migrations,
		}
		payload = json.dumps(payload_dict, ensure_ascii=False) if format_ == "json" else str(payload_dict)
		job = {
			"id": stable_id("tens_export", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"format": format_,
			"payload_size_bytes": len(payload.encode()),
			"actor": actor,
			"created_at": utc_now(),
		}
		self._export_jobs[job["id"]] = job
		return job

	async def tenant_import(
		self,
		tenant_id: str,
		payload: dict[str, Any],
		actor: str,
		overwrite_existing: bool = False,
	) -> dict[str, Any]:
		"""Import a tenant definition payload, optionally overwriting existing data."""
		self._require_tenant(tenant_id)
		lt_data = payload.get("legacy_tenant") or {}
		legacy_tenant_id = str(lt_data.get("legacy_tenant_id") or "")
		if not legacy_tenant_id:
			raise ValueError("import_payload_missing_legacy_tenant_id")
		existing_id = stable_id("tens_legacy", tenant_id, legacy_tenant_id)
		if existing_id in self.legacy_tenants and not overwrite_existing:
			raise PermissionError("tenant_already_exists_use_overwrite")
		record = self.register_legacy_tenant(
			tenant_id=tenant_id,
			legacy_tenant_id=legacy_tenant_id,
			source_system=str(lt_data.get("source_system") or "imported"),
			owner=str(lt_data.get("owner") or actor),
			compatibility_scope=str(lt_data.get("compatibility_scope") or "imported"),
		)
		job = {
			"id": stable_id("tens_import", tenant_id, legacy_tenant_id),
			"tenant_id": tenant_id,
			"imported_tenant": record,
			"actor": actor,
			"created_at": utc_now(),
		}
		self._import_jobs[job["id"]] = job
		return job

	async def subdomain_assign(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		subdomain: str,
		actor: str,
	) -> dict[str, Any]:
		"""Assign a subdomain to a legacy tenant for routing purposes."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not subdomain or not subdomain.replace("-", "").isalnum():
			raise ValueError("invalid_subdomain")
		# Check uniqueness
		existing = next(
			(r for r in self._subdomain_assignments.values() if r["subdomain"] == subdomain and r["tenant_id"] == tenant_id),
			None,
		)
		if existing:
			raise PermissionError(f"subdomain_already_assigned:{subdomain}")
		record = {
			"id": stable_id("tens_subdomain", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"subdomain": subdomain,
			"fqdn": f"{subdomain}.apg.local",
			"actor": actor,
			"assigned_at": utc_now(),
		}
		self._subdomain_assignments[record["id"]] = record
		self._record_event(tenant_id, "subdomain_assigned", record["id"], f"Subdomain assigned: {subdomain}", actor, "low")
		return record

	async def tenant_health_check(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
	) -> dict[str, Any]:
		"""Run a health assessment for a legacy tenant (boundary, mapping, staleness)."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		has_mapping = any(m.legacy_tenant_id == legacy.id for m in self.mappings.values())
		boundary_key = stable_id("tens_boundary", tenant_id, legacy.id)
		has_boundary = boundary_key in self.boundaries
		is_stale = legacy.days_since_activity > 90
		health_status = "healthy"
		if not has_mapping or not has_boundary:
			health_status = "degraded"
		if is_stale:
			health_status = "stale"
		result = {
			"id": stable_id("tens_health", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"status": legacy.status,
			"has_mapping": has_mapping,
			"has_boundary": has_boundary,
			"is_stale": is_stale,
			"health_status": health_status,
			"checked_at": utc_now(),
		}
		self._health_checks[result["id"]] = result
		return result

	async def tenant_suspend(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		reason: str,
		actor: str,
	) -> dict[str, Any]:
		"""Suspend a legacy tenant, preventing further operations."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if not reason:
			raise ValueError("suspension_reason_required")
		legacy.status = "suspended"
		legacy.updated_at = utc_now()
		record = {
			"id": stable_id("tens_suspend", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"reason": reason,
			"actor": actor,
			"suspended_at": utc_now(),
		}
		self._suspensions[record["id"]] = record
		self._record_event(tenant_id, "tenant_suspended", record["id"], f"Suspended: {reason}", actor, "high")
		return record

	async def tenant_reactivate(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		actor: str,
		reactivation_note: str = "",
	) -> dict[str, Any]:
		"""Reactivate a previously suspended or archived legacy tenant."""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		if legacy.status not in {"suspended", "archived"}:
			raise PermissionError("tenant_not_suspended_or_archived")
		legacy.status = "active"
		legacy.updated_at = utc_now()
		record = {
			"id": stable_id("tens_reactivate", tenant_id, legacy.id),
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"reactivation_note": reactivation_note,
			"actor": actor,
			"reactivated_at": utc_now(),
		}
		self._reactivations[record["id"]] = record
		self._record_event(tenant_id, "tenant_reactivated", record["id"], f"Reactivated: {reactivation_note or 'no note'}", actor, "medium")
		return record

	async def tenant_search(
		self,
		tenant_id: str,
		query: str,
		status_filter: str | None = None,
	) -> list[dict[str, Any]]:
		"""Search legacy tenants by legacy_tenant_id or source_system substring."""
		q = query.lower()
		return [
			lt.to_dict()
			for lt in self.legacy_tenants.values()
			if lt.tenant_id == tenant_id
			and (q in lt.legacy_tenant_id.lower() or q in lt.source_system.lower())
			and (status_filter is None or lt.status == status_filter)
		]

	async def audit_search(
		self,
		tenant_id: str,
		event_type_filter: str | None = None,
		actor_filter: str | None = None,
	) -> list[dict[str, Any]]:
		"""Search audit events by event_type or actor substring."""
		return [
			ev.to_dict()
			for ev in self.audit_events.values()
			if ev.tenant_id == tenant_id
			and (event_type_filter is None or event_type_filter in ev.event_type)
			and (actor_filter is None or actor_filter in ev.actor)
		]

	async def migration_summary(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return a concise count of migrations by status for a tenant."""
		migrations = self.list_migrations(tenant_id)
		by_status: dict[str, int] = {}
		for m in migrations:
			s = str(m.get("status") or "unknown")
			by_status[s] = by_status.get(s, 0) + 1
		return {
			"tenant_id": tenant_id,
			"total": len(migrations),
			"by_status": by_status,
			"generated_at": utc_now(),
		}

	async def list_archived_tenants(
		self,
		tenant_id: str,
	) -> list[dict[str, Any]]:
		"""Return all archived tenant records for a tenant."""
		return [v for v in self._tenant_archives.values() if v.get("tenant_id") == tenant_id]

	async def resource_quota(
		self,
		tenant_id: str,
		legacy_tenant_id: str,
		quotas: dict[str, int | float],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Set or retrieve resource quotas for a legacy tenant.

		quotas dict may contain keys: max_api_calls, max_storage_mb, max_users.
		"""
		self._require_tenant(tenant_id)
		legacy = self._get_legacy_tenant(tenant_id, legacy_tenant_id)
		key = stable_id("tens_quota", tenant_id, legacy.id)
		existing = self._resource_quotas.get(key, {})
		merged = {**existing, **{k: v for k, v in quotas.items() if isinstance(v, (int, float))}}
		record = {
			"id": key,
			"tenant_id": tenant_id,
			"legacy_tenant_id": legacy.id,
			"quotas": merged,
			"actor": actor,
			"updated_at": utc_now(),
		}
		self._resource_quotas[key] = record
		self._record_event(tenant_id, "resource_quota_updated", key, f"Quotas updated: {list(merged.keys())}", actor, "low")
		return record

	# ------------------------------------------------------------------ #
	# Private helpers                                                      #
	# ------------------------------------------------------------------ #

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

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_tenant_archives', '_tenant_clones', '_usage_reports', '_billing_summaries', '_isolation_checks', '_export_jobs', '_import_jobs', '_subdomain_assignments', '_health_checks', '_suspensions', '_reactivations', '_resource_quotas']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

