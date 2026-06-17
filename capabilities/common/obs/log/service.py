"""Log Aggregation service (obs_log).

Structured log ingestion, correlation ID injection, retention policies,
log level management, Loki export.
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "obs_log"
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_LEVEL_ORDER = {lvl: i for i, lvl in enumerate(LOG_LEVELS)}


def _now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sid() -> str:
	return uuid4().hex[:16]


def _cid() -> str:
	"""Generate a correlation ID."""
	return uuid4().hex


class LogAggregationService:
	"""In-memory async service for log aggregation lifecycle."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		guard_tenant_id(tenant_id)
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self._entries = WriteThruList('entries', tenant_id, _store)
		self._retention_policies = WriteThruDict('retention_policies', tenant_id, _store)
		self._level_overrides = WriteThruDict('level_overrides', tenant_id, _store)
		self._loki_configs = WriteThruDict('loki_configs', tenant_id, _store)
		self._correlation_contexts = WriteThruDict('correlation_contexts', tenant_id, _store)
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		_log.info("LogAggregationService initialised tenant=%s", self.tenant_id)

	# ------------------------------------------------------------------ helpers

	def _emit(self, event_type: str, record_id: str, record_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _sid(),
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"details": details or {},
			"emitted_at": _now(),
		})

	def _level_passes_policy(self, level: str, min_level: str) -> bool:
		return LOG_LEVEL_ORDER.get(level.upper(), 0) >= LOG_LEVEL_ORDER.get(min_level.upper(), 0)

	def _effective_min_level(self, service_name: str, logger_name: str | None) -> str:
		"""Get the effective minimum log level considering active overrides."""
		now_str = _now()
		for override in self._level_overrides.values():
			if not override["active"]:
				continue
			if override["expires_at"] and override["expires_at"] < now_str:
				override["active"] = False
				continue
			if override["service_name"] != service_name:
				continue
			if override["logger_name"] and logger_name and override["logger_name"] != logger_name:
				continue
			return override["level"]
		# Fall back to retention policy
		for policy in self._retention_policies.values():
			if not policy["enabled"]:
				continue
			if policy["service_name"] and policy["service_name"] != service_name:
				continue
			return policy["min_level"]
		return "DEBUG"

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy",
			"capability": CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"log_entries": len(self._entries),
			"retention_policies": len(self._retention_policies),
			"level_overrides": len(self._level_overrides),
			"loki_configs": len(self._loki_configs),
			"checked_at": _now(),
		}

	# ------------------------------------------------------------------ describe

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "observability",
			"description": "Structured log ingestion, correlation ID injection, retention policies, log level management, Loki export",
			"supported_levels": LOG_LEVELS,
			"tenant_id": self.tenant_id,
		}

	# ------------------------------------------------------------------ audit

	async def get_audit_events(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return deepcopy(events[-limit:])

	# ------------------------------------------------------------------ log ingestion

	async def ingest_log(
		self,
		service_name: str,
		level: str,
		message: str,
		timestamp: str | None = None,
		correlation_id: str | None = None,
		trace_id: str | None = None,
		span_id: str | None = None,
		fields: dict[str, Any] | None = None,
		source_file: str | None = None,
		source_line: int | None = None,
		logger_name: str | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		guard_non_empty_string(message, "message")
		level_upper = level.upper()
		if level_upper not in LOG_LEVEL_ORDER:
			raise ValueError(f"Unsupported log level: {level}. Supported: {LOG_LEVELS}")

		# Apply effective minimum level filter
		effective_min = self._effective_min_level(service_name, logger_name)
		if not self._level_passes_policy(level_upper, effective_min):
			_log.debug("log suppressed by level policy: service=%s level=%s min=%s", service_name, level_upper, effective_min)
			return {
				"suppressed": True,
				"reason": f"level {level_upper} below minimum {effective_min}",
				"service_name": service_name,
			}

		entry_id = _sid()
		record: dict[str, Any] = {
			"id": entry_id,
			"service_name": service_name,
			"level": level_upper,
			"message": message,
			"timestamp": timestamp or _now(),
			"correlation_id": correlation_id,
			"trace_id": trace_id,
			"span_id": span_id,
			"fields": fields or {},
			"source_file": source_file,
			"source_line": source_line,
			"logger_name": logger_name,
			"tenant_id": self.tenant_id,
			"ingested_at": _now(),
		}
		self._entries.append(record)
		# Prune to 500k entries
		if len(self._entries) > 500_000:
			self._entries = self._entries[-500_000:]
		return deepcopy(record)

	async def bulk_ingest_logs(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
		tasks = [
			self.ingest_log(
				service_name=e.get("service_name", ""),
				level=e.get("level", "INFO"),
				message=e.get("message", ""),
				timestamp=e.get("timestamp"),
				correlation_id=e.get("correlation_id"),
				trace_id=e.get("trace_id"),
				span_id=e.get("span_id"),
				fields=e.get("fields"),
				source_file=e.get("source_file"),
				source_line=e.get("source_line"),
				logger_name=e.get("logger_name"),
			)
			for e in entries
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		ingested = [r for r in results if not isinstance(r, Exception) and not r.get("suppressed")]
		suppressed = [r for r in results if not isinstance(r, Exception) and r.get("suppressed")]
		errors = [{"error": str(r)} for r in results if isinstance(r, Exception)]
		return {"ingested": len(ingested), "suppressed": len(suppressed), "errors": errors}

	async def get_log_entry(self, entry_id: str) -> dict[str, Any]:
		guard_non_empty_string(entry_id, "entry_id")
		for entry in self._entries:
			if entry["id"] == entry_id:
				return deepcopy(entry)
		raise KeyError(f"Log entry not found: {entry_id}")

	async def list_log_entries(
		self,
		service_name: str | None = None,
		level: str | None = None,
		min_level: str | None = None,
		correlation_id: str | None = None,
		trace_id: str | None = None,
		start_time: str | None = None,
		end_time: str | None = None,
		message_contains: str | None = None,
		fields_match: dict[str, Any] | None = None,
		page: int = 1,
		page_size: int = 100,
	) -> dict[str, Any]:
		entries = list(self._entries)
		if service_name:
			entries = [e for e in entries if e["service_name"] == service_name]
		if level:
			entries = [e for e in entries if e["level"] == level.upper()]
		if min_level:
			entries = [e for e in entries if self._level_passes_policy(e["level"], min_level)]
		if correlation_id:
			entries = [e for e in entries if e.get("correlation_id") == correlation_id]
		if trace_id:
			entries = [e for e in entries if e.get("trace_id") == trace_id]
		if start_time:
			entries = [e for e in entries if e["timestamp"] >= start_time]
		if end_time:
			entries = [e for e in entries if e["timestamp"] <= end_time]
		if message_contains:
			entries = [e for e in entries if message_contains.lower() in e["message"].lower()]
		if fields_match:
			entries = [e for e in entries if all(e["fields"].get(k) == v for k, v in fields_match.items())]

		total = len(entries)
		offset = (page - 1) * page_size
		page_entries = entries[offset: offset + page_size]
		return {
			"items": [deepcopy(e) for e in page_entries],
			"total": total,
			"page": page,
			"page_size": page_size,
			"has_more": offset + page_size < total,
		}

	async def delete_log_entry(self, entry_id: str) -> dict[str, Any]:
		guard_non_empty_string(entry_id, "entry_id")
		before = len(self._entries)
		self._entries = [e for e in self._entries if e["id"] != entry_id]
		if len(self._entries) == before:
			raise KeyError(f"Log entry not found: {entry_id}")
		self._emit("log_entry_deleted", entry_id, "log_entry")
		return {"deleted": True, "entry_id": entry_id}

	async def purge_log_entries(self, service_name: str | None = None, before_timestamp: str | None = None) -> dict[str, Any]:
		before = len(self._entries)
		entries = self._entries
		if service_name:
			entries = [e for e in entries if e["service_name"] != service_name]
		if before_timestamp:
			entries = [e for e in entries if e["timestamp"] >= before_timestamp]
		self._entries = entries if (service_name or before_timestamp) else []
		deleted = before - len(self._entries)
		self._emit("log_entries_purged", "bulk", "log_entry", {"deleted": deleted})
		return {"deleted": deleted}

	# ------------------------------------------------------------------ correlation ID injection

	async def create_correlation_context(
		self,
		service_name: str,
		correlation_id: str | None = None,
		trace_id: str | None = None,
		request_id: str | None = None,
		user_id: str | None = None,
		session_id: str | None = None,
		extra: dict[str, str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		ctx_id = _sid()
		resolved_correlation_id = correlation_id or _cid()
		record: dict[str, Any] = {
			"id": ctx_id,
			"correlation_id": resolved_correlation_id,
			"trace_id": trace_id,
			"request_id": request_id or _cid(),
			"user_id": user_id,
			"session_id": session_id,
			"service_name": service_name,
			"extra": extra or {},
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._correlation_contexts[ctx_id] = record
		self._emit("correlation_context_created", ctx_id, "correlation_context")
		return deepcopy(record)

	async def get_correlation_context(self, ctx_id: str) -> dict[str, Any]:
		guard_non_empty_string(ctx_id, "ctx_id")
		if ctx_id not in self._correlation_contexts:
			raise KeyError(f"Correlation context not found: {ctx_id}")
		return deepcopy(self._correlation_contexts[ctx_id])

	async def find_correlation_context(self, correlation_id: str) -> dict[str, Any] | None:
		guard_non_empty_string(correlation_id, "correlation_id")
		for ctx in self._correlation_contexts.values():
			if ctx["correlation_id"] == correlation_id:
				return deepcopy(ctx)
		return None

	async def list_correlation_contexts(self, service_name: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
		ctxs = list(self._correlation_contexts.values())
		if service_name:
			ctxs = [c for c in ctxs if c["service_name"] == service_name]
		total = len(ctxs)
		offset = (page - 1) * page_size
		return {"items": [deepcopy(c) for c in ctxs[offset: offset + page_size]], "total": total, "page": page, "page_size": page_size}

	async def delete_correlation_context(self, ctx_id: str) -> dict[str, Any]:
		guard_non_empty_string(ctx_id, "ctx_id")
		if ctx_id not in self._correlation_contexts:
			raise KeyError(f"Correlation context not found: {ctx_id}")
		del self._correlation_contexts[ctx_id]
		self._emit("correlation_context_deleted", ctx_id, "correlation_context")
		return {"deleted": True, "ctx_id": ctx_id}

	async def get_logs_by_correlation_id(self, correlation_id: str, page: int = 1, page_size: int = 100) -> dict[str, Any]:
		guard_non_empty_string(correlation_id, "correlation_id")
		return await self.list_log_entries(correlation_id=correlation_id, page=page, page_size=page_size)

	async def get_logs_by_trace_id(self, trace_id: str, page: int = 1, page_size: int = 100) -> dict[str, Any]:
		guard_non_empty_string(trace_id, "trace_id")
		return await self.list_log_entries(trace_id=trace_id, page=page, page_size=page_size)

	# ------------------------------------------------------------------ retention policies

	async def create_retention_policy(
		self,
		name: str,
		retention_days: int = 30,
		min_level: str = "DEBUG",
		service_name: str | None = None,
		archive_after_days: int | None = None,
		delete_after_days: int | None = None,
		compress_after_days: int | None = None,
		enabled: bool = True,
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		if min_level.upper() not in LOG_LEVEL_ORDER:
			raise ValueError(f"Invalid log level: {min_level}")
		policy_id = _sid()
		record: dict[str, Any] = {
			"id": policy_id,
			"name": name,
			"service_name": service_name,
			"min_level": min_level.upper(),
			"retention_days": retention_days,
			"archive_after_days": archive_after_days,
			"delete_after_days": delete_after_days,
			"compress_after_days": compress_after_days,
			"enabled": enabled,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
			"updated_at": None,
		}
		self._retention_policies[policy_id] = record
		self._emit("retention_policy_created", policy_id, "retention_policy", {"name": name})
		return deepcopy(record)

	async def update_retention_policy(
		self,
		policy_id: str,
		min_level: str | None = None,
		retention_days: int | None = None,
		archive_after_days: int | None = None,
		delete_after_days: int | None = None,
		compress_after_days: int | None = None,
		enabled: bool | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(policy_id, "policy_id")
		if policy_id not in self._retention_policies:
			raise KeyError(f"Retention policy not found: {policy_id}")
		pol = self._retention_policies[policy_id]
		if min_level is not None:
			if min_level.upper() not in LOG_LEVEL_ORDER:
				raise ValueError(f"Invalid log level: {min_level}")
			pol["min_level"] = min_level.upper()
		if retention_days is not None:
			pol["retention_days"] = retention_days
		if archive_after_days is not None:
			pol["archive_after_days"] = archive_after_days
		if delete_after_days is not None:
			pol["delete_after_days"] = delete_after_days
		if compress_after_days is not None:
			pol["compress_after_days"] = compress_after_days
		if enabled is not None:
			pol["enabled"] = enabled
		pol["updated_at"] = _now()
		self._emit("retention_policy_updated", policy_id, "retention_policy")
		return deepcopy(pol)

	async def get_retention_policy(self, policy_id: str) -> dict[str, Any]:
		guard_non_empty_string(policy_id, "policy_id")
		if policy_id not in self._retention_policies:
			raise KeyError(f"Retention policy not found: {policy_id}")
		return deepcopy(self._retention_policies[policy_id])

	async def list_retention_policies(self, enabled_only: bool = False) -> list[dict[str, Any]]:
		policies = list(self._retention_policies.values())
		if enabled_only:
			policies = [p for p in policies if p["enabled"]]
		return [deepcopy(p) for p in policies]

	async def delete_retention_policy(self, policy_id: str) -> dict[str, Any]:
		guard_non_empty_string(policy_id, "policy_id")
		if policy_id not in self._retention_policies:
			raise KeyError(f"Retention policy not found: {policy_id}")
		del self._retention_policies[policy_id]
		self._emit("retention_policy_deleted", policy_id, "retention_policy")
		return {"deleted": True, "policy_id": policy_id}

	async def apply_retention_policies(self) -> dict[str, Any]:
		"""Enforce all enabled retention policies, deleting expired entries."""
		total_deleted = 0
		for policy in self._retention_policies.values():
			if not policy["enabled"]:
				continue
			cutoff = datetime.now(timezone.utc) - timedelta(days=policy["retention_days"])
			cutoff_str = cutoff.isoformat(timespec="microseconds")
			before = len(self._entries)
			if policy["service_name"]:
				self._entries = [
					e for e in self._entries
					if not (e["service_name"] == policy["service_name"] and e["timestamp"] < cutoff_str)
				]
			else:
				self._entries = [e for e in self._entries if e["timestamp"] >= cutoff_str]
			deleted = before - len(self._entries)
			total_deleted += deleted
			_log.info("retention policy=%s deleted=%d entries", policy["name"], deleted)
		return {"total_deleted": total_deleted, "applied_at": _now()}

	# ------------------------------------------------------------------ log level management

	async def create_level_override(
		self,
		service_name: str,
		level: str,
		logger_name: str | None = None,
		duration_minutes: int | None = None,
		reason: str = "",
	) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		level_upper = level.upper()
		if level_upper not in LOG_LEVEL_ORDER:
			raise ValueError(f"Invalid log level: {level}")
		override_id = _sid()
		expires_at: str | None = None
		if duration_minutes:
			expires_at = (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat(timespec="microseconds")
		record: dict[str, Any] = {
			"id": override_id,
			"service_name": service_name,
			"logger_name": logger_name,
			"level": level_upper,
			"duration_minutes": duration_minutes,
			"reason": reason,
			"expires_at": expires_at,
			"active": True,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._level_overrides[override_id] = record
		self._emit("level_override_created", override_id, "level_override", {"service": service_name, "level": level_upper})
		return deepcopy(record)

	async def update_level_override(self, override_id: str, level: str | None = None, active: bool | None = None) -> dict[str, Any]:
		guard_non_empty_string(override_id, "override_id")
		if override_id not in self._level_overrides:
			raise KeyError(f"Level override not found: {override_id}")
		override = self._level_overrides[override_id]
		if level is not None:
			level_upper = level.upper()
			if level_upper not in LOG_LEVEL_ORDER:
				raise ValueError(f"Invalid log level: {level}")
			override["level"] = level_upper
		if active is not None:
			override["active"] = active
		self._emit("level_override_updated", override_id, "level_override")
		return deepcopy(override)

	async def get_level_override(self, override_id: str) -> dict[str, Any]:
		guard_non_empty_string(override_id, "override_id")
		if override_id not in self._level_overrides:
			raise KeyError(f"Level override not found: {override_id}")
		return deepcopy(self._level_overrides[override_id])

	async def list_level_overrides(self, service_name: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		overrides = list(self._level_overrides.values())
		if service_name:
			overrides = [o for o in overrides if o["service_name"] == service_name]
		if active_only:
			now_str = _now()
			overrides = [
				o for o in overrides
				if o["active"] and (o["expires_at"] is None or o["expires_at"] > now_str)
			]
		return [deepcopy(o) for o in overrides]

	async def delete_level_override(self, override_id: str) -> dict[str, Any]:
		guard_non_empty_string(override_id, "override_id")
		if override_id not in self._level_overrides:
			raise KeyError(f"Level override not found: {override_id}")
		del self._level_overrides[override_id]
		self._emit("level_override_deleted", override_id, "level_override")
		return {"deleted": True, "override_id": override_id}

	async def get_effective_log_level(self, service_name: str, logger_name: str | None = None) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		level = self._effective_min_level(service_name, logger_name)
		return {"service_name": service_name, "logger_name": logger_name, "effective_level": level, "resolved_at": _now()}

	# ------------------------------------------------------------------ Loki export

	async def create_loki_config(
		self,
		name: str,
		endpoint: str,
		tenant_header: str | None = None,
		extra_labels: dict[str, str] | None = None,
		batch_size: int = 1000,
		flush_interval_ms: int = 1000,
		max_retries: int = 3,
		enabled: bool = True,
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		guard_non_empty_string(endpoint, "endpoint")
		config_id = _sid()
		record: dict[str, Any] = {
			"id": config_id,
			"name": name,
			"endpoint": endpoint,
			"tenant_header": tenant_header,
			"extra_labels": extra_labels or {},
			"batch_size": batch_size,
			"flush_interval_ms": flush_interval_ms,
			"max_retries": max_retries,
			"enabled": enabled,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._loki_configs[config_id] = record
		self._emit("loki_config_created", config_id, "loki_config", {"name": name, "endpoint": endpoint})
		return deepcopy(record)

	async def update_loki_config(
		self,
		config_id: str,
		enabled: bool | None = None,
		batch_size: int | None = None,
		flush_interval_ms: int | None = None,
		extra_labels: dict[str, str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._loki_configs:
			raise KeyError(f"Loki config not found: {config_id}")
		cfg = self._loki_configs[config_id]
		if enabled is not None:
			cfg["enabled"] = enabled
		if batch_size is not None:
			cfg["batch_size"] = batch_size
		if flush_interval_ms is not None:
			cfg["flush_interval_ms"] = flush_interval_ms
		if extra_labels is not None:
			cfg["extra_labels"].update(extra_labels)
		self._emit("loki_config_updated", config_id, "loki_config")
		return deepcopy(cfg)

	async def get_loki_config(self, config_id: str) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._loki_configs:
			raise KeyError(f"Loki config not found: {config_id}")
		return deepcopy(self._loki_configs[config_id])

	async def list_loki_configs(self, enabled_only: bool = False) -> list[dict[str, Any]]:
		cfgs = list(self._loki_configs.values())
		if enabled_only:
			cfgs = [c for c in cfgs if c["enabled"]]
		return [deepcopy(c) for c in cfgs]

	async def delete_loki_config(self, config_id: str) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._loki_configs:
			raise KeyError(f"Loki config not found: {config_id}")
		del self._loki_configs[config_id]
		self._emit("loki_config_deleted", config_id, "loki_config")
		return {"deleted": True, "config_id": config_id}

	async def render_loki_push_payload(self, service_name: str | None = None, limit: int = 1000) -> dict[str, Any]:
		"""Produce a Loki push API compatible payload from recent log entries."""
		entries = list(self._entries)
		if service_name:
			entries = [e for e in entries if e["service_name"] == service_name]
		entries = entries[-limit:]

		# Group by service and level as stream labels
		from collections import defaultdict
		streams: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
		for entry in entries:
			key = (entry["service_name"], entry["level"])
			ts_ns = entry["timestamp"]
			streams[key].append([ts_ns, entry["message"]])

		loki_streams = [
			{
				"stream": {"service": svc, "level": lvl, "tenant": self.tenant_id},
				"values": vals,
			}
			for (svc, lvl), vals in streams.items()
		]
		return {"streams": loki_streams}

	# ------------------------------------------------------------------ analytics

	async def get_log_statistics(self, service_name: str | None = None) -> dict[str, Any]:
		entries = list(self._entries)
		if service_name:
			entries = [e for e in entries if e["service_name"] == service_name]
		level_counts: dict[str, int] = {}
		service_counts: dict[str, int] = {}
		for e in entries:
			level_counts[e["level"]] = level_counts.get(e["level"], 0) + 1
			service_counts[e["service_name"]] = service_counts.get(e["service_name"], 0) + 1
		return {
			"total_entries": len(entries),
			"level_distribution": level_counts,
			"service_distribution": service_counts,
			"service_name_filter": service_name,
			"computed_at": _now(),
		}

	async def search_logs(self, query: str, service_name: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
		"""Full-text search over log messages."""
		guard_non_empty_string(query, "query")
		pattern = re.compile(query, re.IGNORECASE)
		entries = list(self._entries)
		if service_name:
			entries = [e for e in entries if e["service_name"] == service_name]
		matched = [e for e in entries if pattern.search(e["message"])]
		total = len(matched)
		offset = (page - 1) * page_size
		return {
			"items": [deepcopy(e) for e in matched[offset: offset + page_size]],
			"total": total,
			"query": query,
			"page": page,
			"page_size": page_size,
		}

	async def get_error_summary(self, service_name: str | None = None, window_minutes: int = 60) -> dict[str, Any]:
		"""Return a summary of error and critical log entries."""
		entries = [e for e in self._entries if e["level"] in ("ERROR", "CRITICAL")]
		if service_name:
			entries = [e for e in entries if e["service_name"] == service_name]
		by_service: dict[str, int] = {}
		for e in entries:
			by_service[e["service_name"]] = by_service.get(e["service_name"], 0) + 1
		return {
			"total_errors": len(entries),
			"by_service": by_service,
			"window_minutes": window_minutes,
			"service_name_filter": service_name,
			"computed_at": _now(),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_retention_policies', '_level_overrides', '_loki_configs', '_correlation_contexts', '_entries', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

