"""APG Data Synchronisation service."""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string
from .models import DsySyncConfig, DsySyncRun, DsySyncConflict, SyncStatus, SyncDirection

_log = logging.getLogger(__name__)


class DsyService:
	def __init__(self, tenant_id: str = "default") -> None:
		self._tenant_id = tenant_id
		self._configs: dict[str, DsySyncConfig] = {}
		self._runs: list[DsySyncRun] = []
		self._conflicts: list[DsySyncConflict] = []

	async def create_sync(
		self,
		name: str,
		source_capability: str,
		source_entity: str,
		target_capability: str,
		target_entity: str,
		field_mappings: list[dict[str, Any]] | None = None,
		direction: str = "bidirectional",
		frequency_minutes: int = 15,
		tenant_id: str | None = None,
	) -> DsySyncConfig:
		tid = tenant_id or self._tenant_id
		guard_tenant_id(tid)
		guard_non_empty_string(name, "name")
		from .models import DsyFieldMapping
		cfg = DsySyncConfig(
			tenant_id=tid, name=name,
			source_capability=source_capability, source_entity=source_entity,
			target_capability=target_capability, target_entity=target_entity,
			direction=SyncDirection(direction),
			field_mappings=[DsyFieldMapping(**m) for m in (field_mappings or [])],
			frequency_minutes=frequency_minutes,
		)
		self._configs[cfg.id] = cfg
		_log.info("Created sync '%s' (%s → %s)", name, source_capability, target_capability)
		return cfg

	async def run_sync(self, config_id: str, tenant_id: str | None = None) -> DsySyncRun:
		cfg = self._configs.get(config_id)
		assert cfg is not None, f"Sync config {config_id} not found"
		assert cfg.enabled, "Sync config is disabled"
		cfg.status = SyncStatus.RUNNING
		run = DsySyncRun(tenant_id=cfg.tenant_id, sync_config_id=config_id)
		self._runs.append(run)
		try:
			# Production: pull records from source, apply field mappings, upsert to target
			run.records_processed = 0
			run.status = "completed"
			run.completed_at = datetime.now(timezone.utc)
			cfg.last_sync_at = run.completed_at
			cfg.last_sync_records = run.records_processed
			cfg.status = SyncStatus.IDLE
		except Exception as exc:
			run.status = "failed"
			run.error_message = str(exc)
			run.completed_at = datetime.now(timezone.utc)
			cfg.status = SyncStatus.ERROR
			_log.error("Sync %s failed: %s", config_id, exc)
		return run

	async def list_syncs(self, tenant_id: str | None = None) -> list[DsySyncConfig]:
		tid = tenant_id or self._tenant_id
		return [c for c in self._configs.values() if c.tenant_id == tid]

	async def get_sync_history(self, config_id: str, tenant_id: str | None = None) -> list[DsySyncRun]:
		return [r for r in self._runs if r.sync_config_id == config_id]

	async def list_conflicts(self, config_id: str | None = None, tenant_id: str | None = None) -> list[DsySyncConflict]:
		tid = tenant_id or self._tenant_id
		return [c for c in self._conflicts if c.tenant_id == tid and (config_id is None or c.sync_config_id == config_id) and c.resolution == "pending"]

	async def resolve_conflict(self, conflict_id: str, resolution: str, resolved_by: str, tenant_id: str | None = None) -> DsySyncConflict:
		conflict = next((c for c in self._conflicts if c.id == conflict_id), None)
		assert conflict is not None, f"Conflict {conflict_id} not found"
		conflict.resolution = resolution
		conflict.resolved_by = resolved_by
		return conflict
