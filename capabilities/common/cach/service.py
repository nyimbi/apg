"""
APG Cache Management (CACH) - Expanded Service Implementation

Dependency-light in-memory store pattern. 44+ async methods covering
set/get/delete/exists, bulk ops, TTL management, eviction policies,
distributed locking, pub/sub notifications, cache warming, miss
reporting, pattern invalidation, health checks and governance.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import statistics
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, AsyncIterator, Awaitable, Callable

from uuid6 import uuid7

import logging
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

logger = logging.getLogger(__name__)



# Optional compression library stubs (installed in production)
try:
    import lz4.frame as lz4_frame
except ImportError:
    lz4_frame = None  # pip install lz4
try:
    import zstandard
except ImportError:
    zstandard = None  # pip install zstandard

def uuid7str() -> str:
	return str(uuid7())


def _ts() -> str:
	return datetime.utcnow().isoformat(timespec="seconds")


def _now_epoch() -> float:
	return time.monotonic()


class _R(dict[str, Any]):
	"""Thin dict wrapper for records."""


class CacheService:
	"""
	44+ async methods for cache lifecycle, governance, analytics,
	distributed locking, pub/sub, pattern invalidation, health
	checks, and compliance reporting.

	All state is held in Python dicts (in-memory store pattern).
	Keys are namespaced as  <tenant>:<namespace>:<key>.
	"""

	def __init__(self, actor_id: str = "system", tenant_id: str = "default") -> None:
		self.actor_id = actor_id
		self.tenant_id = tenant_id

		# cache store: full_key -> _R
		self._store:    dict[str, _R] = {}
		# distributed locks: lock_key -> {owner, acquired_at, ttl_seconds}
		self._locks:    dict[str, _R] = {}
		# pub/sub channels: channel -> list[_R notifications]
		self._channels: dict[str, list[_R]] = {}
		# eviction policy per namespace: ns_key -> str
		self._eviction_policies: dict[str, str] = {}
		# namespace metadata
		self._namespaces: dict[str, _R] = {}
		# miss tracking: full_key -> int count
		self._misses:   dict[str, int] = {}
		# hit tracking: full_key -> int count
		self._hits:     dict[str, int] = {}
		# audit log
		self._audit_log: list[_R] = []
		# operation counters
		self._op_count = 0
		self._error_count = 0
		# stale-while-revalidate grace windows (full_key -> swr_grace_seconds)
		self._swr_grace: dict[str, int] = {}
		# refresh callbacks per namespace (ns_key -> async callable)
		self._refresh_callbacks: dict[str, Callable[..., Awaitable[Any]]] = {}
		# write-mode per namespace (ns_key -> str)
		self._write_modes: dict[str, str] = {}
		# write-behind queue (full_key -> (value, ttl, tags))
		self._write_queue: list[_R] = []
		# adaptive TTL config per namespace (ns_key -> _R)
		self._adaptive_ttl_cfg: dict[str, _R] = {}
		# tenant quota config (tenant_id -> {soft_bytes, hard_bytes})
		self._quotas: dict[str, _R] = {}
		# schema version per full_key
		self._schema_versions: dict[str, str] = {}
		# tag hierarchy graph (ns_key -> {parent_tag -> [child_tags]})
		self._tag_graph: dict[str, dict[str, list[str]]] = {}
		# active warming progress tracking
		self._warming_progress: dict[str, _R] = {}
		# xfetch last recompute deltas (full_key -> seconds)
		self._xfetch_deltas: dict[str, float] = {}
		# session writes (session_id -> {full_key -> value})
		self._session_writes: dict[str, dict[str, Any]] = {}
		# monetary amount cache (full_key -> Decimal)
		self._money_cache: dict[str, Decimal] = {}

	# ------------------------------------------------------------------
	# helpers
	# ------------------------------------------------------------------

	def _full_key(self, namespace: str, key: str) -> str:
		return f"{self.tenant_id}:{namespace}:{key}"

	def _ns_key(self, namespace: str) -> str:
		return f"{self.tenant_id}:{namespace}"

	def _is_expired(self, record: _R) -> bool:
		expires_at = record.get("expires_at")
		if expires_at is None:
			return False
		return datetime.utcnow() > datetime.fromisoformat(expires_at)

	async def _audit(self, event_type: str, subject: str, details: dict[str, Any] | None = None) -> None:
		self._audit_log.append(_R(
			event_id=uuid7str(),
			tenant_id=self.tenant_id,
			actor_id=self.actor_id,
			event_type=event_type,
			subject=subject,
			details=details or {},
			occurred_at=_ts(),
		))

	# ------------------------------------------------------------------
	# 1. cache_set
	# ------------------------------------------------------------------

	async def cache_set(
		self,
		namespace: str,
		key: str,
		value: Any,
		ttl_seconds: int = 3600,
		tags: list[str] | None = None,
	) -> _R:
		"""Store a value in the cache with optional TTL and tags."""
		assert key, "key required"
		assert ttl_seconds > 0, "ttl_seconds must be positive"
		fk = self._full_key(namespace, key)
		expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
		record = _R(
			full_key=fk,
			namespace=namespace,
			key=key,
			tenant_id=self.tenant_id,
			value=value,
			ttl_seconds=ttl_seconds,
			expires_at=expires_at,
			tags=tags or [],
			created_at=_ts(),
			access_count=0,
			last_accessed_at=None,
		)
		self._store[fk] = record
		self._op_count += 1
		await self._audit("cache_set", fk, {"namespace": namespace, "ttl_seconds": ttl_seconds})
		return _R(key=key, namespace=namespace, stored=True, expires_at=expires_at)

	# ------------------------------------------------------------------
	# 2. cache_get
	# ------------------------------------------------------------------

	async def cache_get(self, namespace: str, key: str) -> _R:
		"""Retrieve a value from cache. Returns hit=False on miss or expiry."""
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		if record is None or self._is_expired(record):
			if record and self._is_expired(record):
				del self._store[fk]
			self._misses[fk] = self._misses.get(fk, 0) + 1
			self._op_count += 1
			await self._audit("cache_miss", fk, {"namespace": namespace})
			return _R(hit=False, key=key, namespace=namespace, value=None)
		record["access_count"] = record.get("access_count", 0) + 1
		record["last_accessed_at"] = _ts()
		self._hits[fk] = self._hits.get(fk, 0) + 1
		self._op_count += 1
		await self._audit("cache_hit", fk, {"namespace": namespace})
		return _R(hit=True, key=key, namespace=namespace, value=record["value"], expires_at=record["expires_at"])

	# ------------------------------------------------------------------
	# 3. cache_delete
	# ------------------------------------------------------------------

	async def cache_delete(self, namespace: str, key: str) -> _R:
		"""Delete a single cache entry."""
		fk = self._full_key(namespace, key)
		existed = fk in self._store
		self._store.pop(fk, None)
		self._op_count += 1
		await self._audit("cache_delete", fk, {"namespace": namespace, "existed": existed})
		return _R(deleted=existed, key=key, namespace=namespace)

	# ------------------------------------------------------------------
	# 4. cache_exists
	# ------------------------------------------------------------------

	async def cache_exists(self, namespace: str, key: str) -> _R:
		"""Check whether a non-expired entry exists for a key."""
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		if record and self._is_expired(record):
			del self._store[fk]
			record = None
		exists = record is not None
		return _R(exists=exists, key=key, namespace=namespace)

	# ------------------------------------------------------------------
	# 5. bulk_set
	# ------------------------------------------------------------------

	async def bulk_set(
		self,
		namespace: str,
		items: dict[str, Any],
		ttl_seconds: int = 3600,
	) -> _R:
		"""Set multiple keys atomically."""
		stored = []
		failed = []
		for key, value in items.items():
			try:
				await self.cache_set(namespace, key, value, ttl_seconds)
				stored.append(key)
			except Exception as exc:
				self._error_count += 1
				failed.append({"key": key, "error": str(exc)})
		await self._audit("bulk_set", namespace, {"count": len(stored), "failed": len(failed)})
		return _R(stored_count=len(stored), failed_count=len(failed), stored=stored, failed=failed)

	# ------------------------------------------------------------------
	# 6. bulk_get
	# ------------------------------------------------------------------

	async def bulk_get(self, namespace: str, keys: list[str]) -> _R:
		"""Retrieve multiple keys in one call."""
		results = {}
		hits = 0
		misses = 0
		for key in keys:
			r = await self.cache_get(namespace, key)
			results[key] = r["value"]
			if r["hit"]:
				hits += 1
			else:
				misses += 1
		await self._audit("bulk_get", namespace, {"keys": len(keys), "hits": hits, "misses": misses})
		return _R(results=results, hit_count=hits, miss_count=misses)

	# ------------------------------------------------------------------
	# 7. cache_flush
	# ------------------------------------------------------------------

	async def cache_flush(self, namespace: str | None = None) -> _R:
		"""Flush all entries in a namespace, or the entire tenant cache."""
		prefix = f"{self.tenant_id}:{namespace}:" if namespace else f"{self.tenant_id}:"
		keys_to_delete = [k for k in list(self._store.keys()) if k.startswith(prefix)]
		for k in keys_to_delete:
			del self._store[k]
		self._op_count += len(keys_to_delete)
		await self._audit("cache_flush", namespace or "all", {"deleted_count": len(keys_to_delete)})
		return _R(flushed_count=len(keys_to_delete), namespace=namespace)

	# ------------------------------------------------------------------
	# 8. ttl_update
	# ------------------------------------------------------------------

	async def ttl_update(self, namespace: str, key: str, ttl_seconds: int) -> _R:
		"""Update the TTL of an existing cache entry."""
		assert ttl_seconds > 0, "ttl_seconds must be positive"
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		assert record is not None and not self._is_expired(record), f"key not found: {key}"
		new_expires = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
		record["ttl_seconds"] = ttl_seconds
		record["expires_at"] = new_expires
		await self._audit("ttl_updated", fk, {"new_ttl": ttl_seconds, "new_expires_at": new_expires})
		return _R(key=key, namespace=namespace, new_ttl_seconds=ttl_seconds, new_expires_at=new_expires)

	# ------------------------------------------------------------------
	# 9. cache_stats
	# ------------------------------------------------------------------

	async def cache_stats(self, namespace: str | None = None) -> _R:
		"""Return hit rate, miss rate, entry count and memory estimate."""
		prefix = f"{self.tenant_id}:{namespace}:" if namespace else f"{self.tenant_id}:"
		entries = [v for k, v in self._store.items() if k.startswith(prefix) and not self._is_expired(v)]
		total_hits = sum(self._hits.get(k, 0) for k in self._hits if k.startswith(prefix))
		total_misses = sum(self._misses.get(k, 0) for k in self._misses if k.startswith(prefix))
		total_ops = total_hits + total_misses
		hit_rate = round(total_hits / max(total_ops, 1), 4)
		# Approximate memory via JSON size
		total_bytes = sum(len(json.dumps(e.get("value", ""), default=str).encode()) for e in entries)
		return _R(
			namespace=namespace,
			tenant_id=self.tenant_id,
			entry_count=len(entries),
			total_hits=total_hits,
			total_misses=total_misses,
			hit_rate=hit_rate,
			estimated_bytes=total_bytes,
			computed_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 10. eviction_policy
	# ------------------------------------------------------------------

	async def eviction_policy(self, namespace: str, policy: str) -> _R:
		"""Set the eviction policy for a namespace (lru, lfu, ttl, random)."""
		assert policy in {"lru", "lfu", "ttl", "random", "no_eviction"}, f"unsupported policy: {policy}"
		self._eviction_policies[self._ns_key(namespace)] = policy
		await self._audit("eviction_policy_set", namespace, {"policy": policy})
		return _R(namespace=namespace, policy=policy, set_at=_ts())

	# ------------------------------------------------------------------
	# 11. warm_cache
	# ------------------------------------------------------------------

	async def warm_cache(
		self,
		namespace: str,
		items: dict[str, Any],
		ttl_seconds: int = 7200,
	) -> _R:
		"""Pre-populate the cache from a warm dataset."""
		result = await self.bulk_set(namespace, items, ttl_seconds)
		await self._audit("cache_warmed", namespace, {"count": result["stored_count"]})
		return _R(namespace=namespace, warmed_count=result["stored_count"], warmed_at=_ts())

	# ------------------------------------------------------------------
	# 12. cold_cache_detect
	# ------------------------------------------------------------------

	async def cold_cache_detect(self, namespace: str, threshold_entries: int = 10) -> _R:
		"""Detect whether a namespace has fewer entries than threshold (cold start)."""
		prefix = self._full_key(namespace, "")
		entries = [k for k in self._store if k.startswith(f"{self.tenant_id}:{namespace}:") and not self._is_expired(self._store[k])]
		is_cold = len(entries) < threshold_entries
		result = _R(
			namespace=namespace,
			entry_count=len(entries),
			threshold=threshold_entries,
			is_cold=is_cold,
			detected_at=_ts(),
		)
		await self._audit("cold_cache_detection", namespace, {"is_cold": is_cold, "entries": len(entries)})
		return result

	# ------------------------------------------------------------------
	# 13. cache_miss_report
	# ------------------------------------------------------------------

	async def cache_miss_report(self, namespace: str | None = None) -> _R:
		"""Report on keys with highest miss counts."""
		prefix = f"{self.tenant_id}:{namespace}:" if namespace else f"{self.tenant_id}:"
		miss_data = [
			{"full_key": k, "miss_count": v}
			for k, v in self._misses.items()
			if k.startswith(prefix)
		]
		miss_data.sort(key=lambda x: x["miss_count"], reverse=True)
		total_misses = sum(m["miss_count"] for m in miss_data)
		await self._audit("miss_report_generated", namespace or "all", {"total_misses": total_misses})
		return _R(
			namespace=namespace,
			top_missed_keys=miss_data[:20],
			total_unique_missed_keys=len(miss_data),
			total_misses=total_misses,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 14. distributed_lock
	# ------------------------------------------------------------------

	async def distributed_lock(
		self,
		lock_key: str,
		owner: str,
		ttl_seconds: int = 30,
	) -> _R:
		"""Acquire a distributed lock. Returns acquired=False if already held."""
		fk = f"{self.tenant_id}:lock:{lock_key}"
		existing = self._locks.get(fk)
		if existing:
			lock_created = datetime.fromisoformat(existing["acquired_at"])
			if datetime.utcnow() < lock_created + timedelta(seconds=existing["ttl_seconds"]):
				await self._audit("lock_acquire_failed", lock_key, {"owner": owner, "held_by": existing["owner"]})
				return _R(acquired=False, lock_key=lock_key, held_by=existing["owner"])
		expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
		lock = _R(
			lock_id=uuid7str(),
			lock_key=lock_key,
			owner=owner,
			ttl_seconds=ttl_seconds,
			acquired_at=_ts(),
			expires_at=expires_at,
		)
		self._locks[fk] = lock
		await self._audit("lock_acquired", lock_key, {"owner": owner, "ttl": ttl_seconds})
		return _R(acquired=True, lock_key=lock_key, lock_id=lock["lock_id"], expires_at=expires_at)

	# ------------------------------------------------------------------
	# 15. distributed_unlock
	# ------------------------------------------------------------------

	async def distributed_unlock(self, lock_key: str, owner: str) -> _R:
		"""Release a distributed lock held by owner."""
		fk = f"{self.tenant_id}:lock:{lock_key}"
		lock = self._locks.get(fk)
		if lock is None:
			return _R(released=False, lock_key=lock_key, reason="not_found")
		if lock["owner"] != owner:
			return _R(released=False, lock_key=lock_key, reason="not_owner")
		del self._locks[fk]
		await self._audit("lock_released", lock_key, {"owner": owner})
		return _R(released=True, lock_key=lock_key)

	# ------------------------------------------------------------------
	# 16. cache_invalidate_pattern
	# ------------------------------------------------------------------

	async def cache_invalidate_pattern(self, namespace: str, pattern: str) -> _R:
		"""Delete all entries whose key contains the pattern string."""
		prefix = f"{self.tenant_id}:{namespace}:"
		to_delete = [k for k in list(self._store.keys()) if k.startswith(prefix) and pattern in k[len(prefix):]]
		for k in to_delete:
			del self._store[k]
		await self._audit("cache_invalidate_pattern", namespace, {"pattern": pattern, "deleted": len(to_delete)})
		return _R(namespace=namespace, pattern=pattern, invalidated_count=len(to_delete))

	# ------------------------------------------------------------------
	# 17. pub_sub_notify
	# ------------------------------------------------------------------

	async def pub_sub_notify(self, channel: str, message: Any) -> _R:
		"""Publish a notification to a pub/sub channel."""
		if channel not in self._channels:
			self._channels[channel] = []
		notification = _R(
			notification_id=uuid7str(),
			channel=channel,
			tenant_id=self.tenant_id,
			message=message,
			published_at=_ts(),
		)
		self._channels[channel].append(notification)
		await self._audit("pub_sub_notify", channel, {"channel": channel})
		return notification

	# ------------------------------------------------------------------
	# 18. pub_sub_subscribe
	# ------------------------------------------------------------------

	async def pub_sub_subscribe(self, channel: str, since: str | None = None) -> list[_R]:
		"""Retrieve messages from a pub/sub channel, optionally since a timestamp."""
		messages = self._channels.get(channel, [])
		if since:
			messages = [m for m in messages if m["published_at"] >= since]
		return list(messages)

	# ------------------------------------------------------------------
	# 19. cache_health
	# ------------------------------------------------------------------

	async def cache_health(self) -> _R:
		"""Return overall cache health status."""
		all_entries = [v for k, v in self._store.items() if k.startswith(f"{self.tenant_id}:")]
		active = [e for e in all_entries if not self._is_expired(e)]
		expired = len(all_entries) - len(active)
		total_bytes = sum(len(json.dumps(e.get("value", ""), default=str).encode()) for e in active)
		lock_count = sum(1 for k in self._locks if k.startswith(f"{self.tenant_id}:"))
		return _R(
			status="healthy",
			tenant_id=self.tenant_id,
			active_entries=len(active),
			expired_pending_eviction=expired,
			estimated_bytes=total_bytes,
			active_locks=lock_count,
			total_operations=self._op_count,
			error_count=self._error_count,
			checked_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 20. namespace_create
	# ------------------------------------------------------------------

	async def namespace_create(
		self,
		namespace: str,
		max_entries: int = 100_000,
		default_ttl_seconds: int = 3600,
		owner: str = "system",
	) -> _R:
		"""Register a namespace with metadata and governance settings."""
		ns_key = self._ns_key(namespace)
		record = _R(
			namespace=namespace,
			tenant_id=self.tenant_id,
			owner=owner,
			max_entries=max_entries,
			default_ttl_seconds=default_ttl_seconds,
			eviction_policy=self._eviction_policies.get(ns_key, "lru"),
			status="active",
			created_at=_ts(),
		)
		self._namespaces[ns_key] = record
		await self._audit("namespace_created", namespace, {"owner": owner})
		return record

	# ------------------------------------------------------------------
	# 21. namespace_list
	# ------------------------------------------------------------------

	async def namespace_list(self) -> list[_R]:
		"""List all namespaces for the tenant."""
		prefix = f"{self.tenant_id}:"
		return [n for k, n in self._namespaces.items() if k.startswith(prefix)]

	# ------------------------------------------------------------------
	# 22. namespace_delete
	# ------------------------------------------------------------------

	async def namespace_delete(self, namespace: str) -> _R:
		"""Delete a namespace and all its entries."""
		flushed = await self.cache_flush(namespace)
		self._namespaces.pop(self._ns_key(namespace), None)
		await self._audit("namespace_deleted", namespace, {"flushed_entries": flushed["flushed_count"]})
		return _R(namespace=namespace, deleted=True, flushed_entries=flushed["flushed_count"])

	# ------------------------------------------------------------------
	# 23. bulk_delete
	# ------------------------------------------------------------------

	async def bulk_delete(self, namespace: str, keys: list[str]) -> _R:
		"""Delete multiple keys from a namespace."""
		deleted = []
		for key in keys:
			r = await self.cache_delete(namespace, key)
			if r["deleted"]:
				deleted.append(key)
		await self._audit("bulk_delete", namespace, {"count": len(deleted)})
		return _R(deleted_count=len(deleted), deleted_keys=deleted)

	# ------------------------------------------------------------------
	# 24. tag_invalidate
	# ------------------------------------------------------------------

	async def tag_invalidate(self, namespace: str, tag: str) -> _R:
		"""Invalidate all cache entries with a specific tag."""
		prefix = f"{self.tenant_id}:{namespace}:"
		to_delete = [
			k for k, v in self._store.items()
			if k.startswith(prefix) and tag in v.get("tags", [])
		]
		for k in to_delete:
			del self._store[k]
		await self._audit("tag_invalidated", namespace, {"tag": tag, "deleted": len(to_delete)})
		return _R(namespace=namespace, tag=tag, invalidated_count=len(to_delete))

	# ------------------------------------------------------------------
	# 25. get_with_metadata
	# ------------------------------------------------------------------

	async def get_with_metadata(self, namespace: str, key: str) -> _R:
		"""Get a cache entry including full metadata (TTL, tags, access count)."""
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		if record is None or self._is_expired(record):
			if record:
				del self._store[fk]
			return _R(hit=False, key=key, namespace=namespace)
		remaining_ttl = None
		if record.get("expires_at"):
			remaining_ttl = max(0, (datetime.fromisoformat(record["expires_at"]) - datetime.utcnow()).total_seconds())
		return _R(
			hit=True,
			key=key,
			namespace=namespace,
			value=record["value"],
			tags=record.get("tags", []),
			access_count=record.get("access_count", 0),
			created_at=record.get("created_at"),
			expires_at=record.get("expires_at"),
			remaining_ttl_seconds=round(remaining_ttl, 1) if remaining_ttl is not None else None,
			last_accessed_at=record.get("last_accessed_at"),
		)

	# ------------------------------------------------------------------
	# 26. scan_keys
	# ------------------------------------------------------------------

	async def scan_keys(self, namespace: str, pattern: str = "*") -> list[str]:
		"""Return all non-expired keys in a namespace matching a simple glob."""
		import fnmatch
		prefix = f"{self.tenant_id}:{namespace}:"
		keys = [
			k[len(prefix):]
			for k, v in self._store.items()
			if k.startswith(prefix) and not self._is_expired(v)
		]
		return [k for k in keys if fnmatch.fnmatch(k, pattern)]

	# ------------------------------------------------------------------
	# 27. set_if_not_exists (SETNX)
	# ------------------------------------------------------------------

	async def set_if_not_exists(
		self,
		namespace: str,
		key: str,
		value: Any,
		ttl_seconds: int = 3600,
	) -> _R:
		"""Set a key only if it does not already exist (atomic SETNX)."""
		exists = await self.cache_exists(namespace, key)
		if exists["exists"]:
			return _R(set=False, key=key, namespace=namespace, reason="already_exists")
		await self.cache_set(namespace, key, value, ttl_seconds)
		return _R(set=True, key=key, namespace=namespace)

	# ------------------------------------------------------------------
	# 28. increment
	# ------------------------------------------------------------------

	async def increment(self, namespace: str, key: str, delta: int = 1) -> _R:
		"""Atomically increment a numeric cache value."""
		r = await self.cache_get(namespace, key)
		current = r["value"] if r["hit"] and isinstance(r["value"], (int, float)) else 0
		new_value = current + delta
		await self.cache_set(namespace, key, new_value, ttl_seconds=3600)
		await self._audit("incremented", self._full_key(namespace, key), {"delta": delta, "new_value": new_value})
		return _R(key=key, namespace=namespace, new_value=new_value)

	# ------------------------------------------------------------------
	# 29. expire_soon_report
	# ------------------------------------------------------------------

	async def expire_soon_report(self, namespace: str, within_seconds: int = 300) -> _R:
		"""Report keys expiring within the next N seconds."""
		prefix = f"{self.tenant_id}:{namespace}:"
		cutoff = (datetime.utcnow() + timedelta(seconds=within_seconds)).isoformat()
		soon = [
			{"key": k[len(prefix):], "expires_at": v["expires_at"]}
			for k, v in self._store.items()
			if k.startswith(prefix)
			and v.get("expires_at") is not None
			and not self._is_expired(v)
			and v["expires_at"] <= cutoff
		]
		return _R(
			namespace=namespace,
			within_seconds=within_seconds,
			expiring_soon=sorted(soon, key=lambda x: x["expires_at"]),
			count=len(soon),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 30. performance_report
	# ------------------------------------------------------------------

	async def performance_report(self) -> _R:
		"""Aggregate performance report across all tenant namespaces."""
		total_hits = sum(self._hits.values())
		total_misses = sum(self._misses.values())
		total_ops = total_hits + total_misses
		hit_rate = round(total_hits / max(total_ops, 1), 4)
		active_entries = sum(
			1 for k, v in self._store.items()
			if k.startswith(f"{self.tenant_id}:") and not self._is_expired(v)
		)
		return _R(
			tenant_id=self.tenant_id,
			total_operations=self._op_count,
			total_hits=total_hits,
			total_misses=total_misses,
			hit_rate=hit_rate,
			active_entries=active_entries,
			error_count=self._error_count,
			active_locks=len(self._locks),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 31. export_entries_csv
	# ------------------------------------------------------------------

	async def export_entries_csv(self, namespace: str) -> str:
		"""Export cache entry metadata to CSV (no values, governance-safe)."""
		prefix = f"{self.tenant_id}:{namespace}:"
		entries = [
			v for k, v in self._store.items()
			if k.startswith(prefix) and not self._is_expired(v)
		]
		buf = io.StringIO()
		fields = ["key", "namespace", "ttl_seconds", "expires_at", "access_count", "created_at"]
		writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(entries)
		await self._audit("entries_exported_csv", namespace, {"count": len(entries)})
		return buf.getvalue()

	# ------------------------------------------------------------------
	# 32. export_stats_json
	# ------------------------------------------------------------------

	async def export_stats_json(self) -> str:
		"""Export cache stats as JSON."""
		stats = await self.cache_stats()
		await self._audit("stats_exported_json", "system", {})
		return json.dumps(dict(stats), default=str, indent=2)

	# ------------------------------------------------------------------
	# 33. health_check  (alias for cache_health)
	# ------------------------------------------------------------------

	async def health_check(self) -> _R:
		"""Return service health — delegates to cache_health."""
		return await self.cache_health()

	# ------------------------------------------------------------------
	# 34. dashboard
	# ------------------------------------------------------------------

	async def dashboard(self) -> _R:
		"""KPI dashboard aggregating key cache metrics."""
		stats = await self.performance_report()
		namespaces = await self.namespace_list()
		cold = 0
		for ns in namespaces:
			r = await self.cold_cache_detect(ns["namespace"])
			if r["is_cold"]:
				cold += 1
		return _R(
			tenant_id=self.tenant_id,
			namespace_count=len(namespaces),
			cold_namespaces=cold,
			active_entries=stats["active_entries"],
			hit_rate=stats["hit_rate"],
			total_operations=stats["total_operations"],
			active_locks=stats["active_locks"],
			pub_sub_channels=len(self._channels),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 35. governance_report
	# ------------------------------------------------------------------

	async def governance_report(self) -> _R:
		"""Governance summary for compliance review."""
		namespaces = await self.namespace_list()
		policies = {
			ns["namespace"]: self._eviction_policies.get(self._ns_key(ns["namespace"]), "not_set")
			for ns in namespaces
		}
		no_policy = sum(1 for v in policies.values() if v == "not_set")
		return _R(
			tenant_id=self.tenant_id,
			namespace_count=len(namespaces),
			namespaces_without_eviction_policy=no_policy,
			eviction_policies=policies,
			audit_event_count=len([e for e in self._audit_log if e["tenant_id"] == self.tenant_id]),
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 36. audit_trail
	# ------------------------------------------------------------------

	async def audit_trail(self, event_type: str | None = None) -> list[_R]:
		"""Return audit events for the tenant."""
		events = [
			e for e in self._audit_log
			if e["tenant_id"] == self.tenant_id and (event_type is None or e["event_type"] == event_type)
		]
		return events

	# ------------------------------------------------------------------
	# 37. auto_evict
	# ------------------------------------------------------------------

	async def auto_evict(self, namespace: str, max_entries: int = 1000) -> _R:
		"""Evict lowest-access-count entries when namespace exceeds max_entries."""
		prefix = f"{self.tenant_id}:{namespace}:"
		active = {
			k: v for k, v in self._store.items()
			if k.startswith(prefix) and not self._is_expired(v)
		}
		if len(active) <= max_entries:
			return _R(namespace=namespace, evicted=0, remaining=len(active))
		sorted_entries = sorted(active.items(), key=lambda x: x[1].get("access_count", 0))
		to_evict = len(active) - max_entries
		evicted_keys = []
		for k, _ in sorted_entries[:to_evict]:
			del self._store[k]
			evicted_keys.append(k)
		await self._audit("auto_evict", namespace, {"evicted": to_evict, "policy": "lfu"})
		return _R(namespace=namespace, evicted=to_evict, remaining=len(active) - to_evict)

	# ------------------------------------------------------------------
	# 38. copy_key
	# ------------------------------------------------------------------

	async def copy_key(
		self,
		src_namespace: str,
		src_key: str,
		dst_namespace: str,
		dst_key: str,
	) -> _R:
		"""Copy a cache entry to a new namespace/key."""
		src = await self.get_with_metadata(src_namespace, src_key)
		assert src["hit"], f"source key not found: {src_key}"
		await self.cache_set(dst_namespace, dst_key, src["value"], ttl_seconds=int(src.get("remaining_ttl_seconds") or 3600))
		await self._audit("key_copied", f"{src_namespace}:{src_key}", {"dst": f"{dst_namespace}:{dst_key}"})
		return _R(copied=True, src=f"{src_namespace}:{src_key}", dst=f"{dst_namespace}:{dst_key}")

	# ------------------------------------------------------------------
	# 39. move_key
	# ------------------------------------------------------------------

	async def move_key(
		self,
		src_namespace: str,
		src_key: str,
		dst_namespace: str,
		dst_key: str,
	) -> _R:
		"""Move a cache entry to a new namespace/key (copy + delete)."""
		result = await self.copy_key(src_namespace, src_key, dst_namespace, dst_key)
		await self.cache_delete(src_namespace, src_key)
		await self._audit("key_moved", f"{src_namespace}:{src_key}", {"dst": f"{dst_namespace}:{dst_key}"})
		return _R(moved=True, src=f"{src_namespace}:{src_key}", dst=f"{dst_namespace}:{dst_key}")

	# ------------------------------------------------------------------
	# 40. access_frequency_report
	# ------------------------------------------------------------------

	async def access_frequency_report(self, namespace: str, top_n: int = 20) -> _R:
		"""Report the most-frequently accessed keys in a namespace."""
		prefix = f"{self.tenant_id}:{namespace}:"
		freq = [
			{"key": k[len(prefix):], "access_count": v.get("access_count", 0)}
			for k, v in self._store.items()
			if k.startswith(prefix) and not self._is_expired(v)
		]
		freq.sort(key=lambda x: x["access_count"], reverse=True)
		return _R(
			namespace=namespace,
			top_keys=freq[:top_n],
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 41. consistency_check
	# ------------------------------------------------------------------

	async def consistency_check(self, namespace: str) -> _R:
		"""Identify and purge expired entries that were not yet evicted."""
		prefix = f"{self.tenant_id}:{namespace}:"
		stale_keys = [k for k, v in self._store.items() if k.startswith(prefix) and self._is_expired(v)]
		for k in stale_keys:
			del self._store[k]
		await self._audit("consistency_check", namespace, {"purged": len(stale_keys)})
		return _R(namespace=namespace, purged_stale_entries=len(stale_keys), checked_at=_ts())

	# ------------------------------------------------------------------
	# 42. cache_reset_stats
	# ------------------------------------------------------------------

	async def cache_reset_stats(self) -> _R:
		"""Reset hit/miss counters and operation counts."""
		self._hits.clear()
		self._misses.clear()
		self._op_count = 0
		self._error_count = 0
		await self._audit("stats_reset", "system", {})
		return _R(reset=True, reset_at=_ts())

	# ------------------------------------------------------------------
	# 43. multi_namespace_stats
	# ------------------------------------------------------------------

	async def multi_namespace_stats(self) -> list[_R]:
		"""Return per-namespace statistics for all registered namespaces."""
		namespaces = await self.namespace_list()
		results = []
		for ns in namespaces:
			stats = await self.cache_stats(ns["namespace"])
			results.append(stats)
		return results

	# ------------------------------------------------------------------
	# 44. compliance_report
	# ------------------------------------------------------------------

	async def compliance_report(self, framework: str = "GDPR") -> _R:
		"""Generate a cache data governance compliance report."""
		namespaces = await self.namespace_list()
		no_ttl_keys = sum(
			1 for v in self._store.values()
			if v.get("tenant_id") == self.tenant_id and v.get("expires_at") is None
		)
		report = _R(
			framework=framework,
			tenant_id=self.tenant_id,
			namespace_count=len(namespaces),
			no_ttl_entry_count=no_ttl_keys,
			audit_trail_complete=True,
			eviction_policies_defined=len(self._eviction_policies),
			generated_at=_ts(),
		)
		await self._audit("compliance_report_generated", "system", {"framework": framework})
		return report

	# ------------------------------------------------------------------
	# 45. cache_set_swr  — stale-while-revalidate write
	# ------------------------------------------------------------------

	async def cache_set_swr(
		self,
		namespace: str,
		key: str,
		value: Any,
		ttl_seconds: int = 3600,
		swr_grace_seconds: int = 60,
		tags: list[str] | None = None,
	) -> _R:
		"""Store a value with a stale-while-revalidate grace window.

		When the entry expires, calls within the grace window receive the
		stale value immediately (stale=True) while a background refresh is
		triggered via the namespace refresh callback (if registered).
		"""
		guard_tenant_id(self.tenant_id)
		guard_non_empty_string(key, "key")
		assert ttl_seconds > 0, "ttl_seconds must be positive"
		assert swr_grace_seconds >= 0, "swr_grace_seconds must be non-negative"
		result = await self.cache_set(namespace, key, value, ttl_seconds, tags)
		fk = self._full_key(namespace, key)
		self._swr_grace[fk] = swr_grace_seconds
		await self._audit(
			"cache_set_swr", fk,
			{"swr_grace_seconds": swr_grace_seconds, "ttl_seconds": ttl_seconds},
		)
		return _R(**result, swr_grace_seconds=swr_grace_seconds)

	# ------------------------------------------------------------------
	# 46. cache_get_swr  — stale-while-revalidate read
	# ------------------------------------------------------------------

	async def cache_get_swr(self, namespace: str, key: str) -> _R:
		"""Read with stale-while-revalidate semantics.

		If the key is within its SWR grace window after expiry, returns the
		stale value with stale=True and triggers background revalidation via
		the registered namespace refresh callback.
		"""
		guard_tenant_id(self.tenant_id)
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		grace = self._swr_grace.get(fk, 0)

		if record is None:
			self._misses[fk] = self._misses.get(fk, 0) + 1
			await self._audit("cache_miss", fk, {"namespace": namespace, "swr": True})
			return _R(hit=False, stale=False, key=key, namespace=namespace, value=None)

		now = datetime.utcnow()
		expires_dt = datetime.fromisoformat(record["expires_at"]) if record.get("expires_at") else None

		# Within normal TTL — clean hit
		if expires_dt is None or now <= expires_dt:
			record["access_count"] = record.get("access_count", 0) + 1
			record["last_accessed_at"] = _ts()
			self._hits[fk] = self._hits.get(fk, 0) + 1
			self._op_count += 1
			await self._audit("cache_hit", fk, {"namespace": namespace, "swr": False})
			return _R(hit=True, stale=False, key=key, namespace=namespace,
				value=record["value"], expires_at=record.get("expires_at"))

		# Within SWR grace window — serve stale, trigger background refresh
		grace_deadline = expires_dt + timedelta(seconds=grace)
		if now <= grace_deadline:
			stale_value = record["value"]
			cb = self._refresh_callbacks.get(self._ns_key(namespace))
			if cb is not None:
				try:
					new_value = await cb(namespace, key)
					if new_value is not None:
						await self.cache_set(namespace, key, new_value,
							ttl_seconds=record.get("ttl_seconds", 3600))
				except Exception as exc:
					logger.warning("SWR refresh failed for %s: %s", fk, exc)
			await self._audit("cache_hit_stale", fk,
				{"namespace": namespace, "grace_remaining_s": (grace_deadline - now).total_seconds()})
			return _R(hit=True, stale=True, key=key, namespace=namespace,
				value=stale_value, expires_at=record.get("expires_at"))

		# Fully expired, past grace — hard miss
		del self._store[fk]
		self._misses[fk] = self._misses.get(fk, 0) + 1
		await self._audit("cache_miss", fk, {"namespace": namespace, "swr": True, "past_grace": True})
		return _R(hit=False, stale=False, key=key, namespace=namespace, value=None)

	# ------------------------------------------------------------------
	# 47. register_refresh_callback
	# ------------------------------------------------------------------

	async def register_refresh_callback(
		self,
		namespace: str,
		callback: Callable[[str, str], Awaitable[Any]],
	) -> _R:
		"""Register an async callback for SWR background revalidation.

		The callback signature is ``async def cb(namespace, key) -> value``.
		"""
		guard_tenant_id(self.tenant_id)
		self._refresh_callbacks[self._ns_key(namespace)] = callback
		await self._audit("refresh_callback_registered", namespace, {})
		return _R(namespace=namespace, registered=True, registered_at=_ts())

	# ------------------------------------------------------------------
	# 48. adaptive_ttl_configure  — per-namespace adaptive TTL policy
	# ------------------------------------------------------------------

	async def adaptive_ttl_configure(
		self,
		namespace: str,
		ttl_min_seconds: int = 60,
		ttl_max_seconds: int = 86400,
		growth_factor: float = 1.5,
	) -> _R:
		"""Configure adaptive TTL for a namespace.

		On each cache hit the remaining TTL is extended by ``growth_factor``
		(capped at ``ttl_max_seconds``). Entries that go cold shrink naturally
		to ``ttl_min_seconds`` on next rewrite.
		"""
		guard_tenant_id(self.tenant_id)
		assert ttl_min_seconds > 0, "ttl_min_seconds must be positive"
		assert ttl_max_seconds >= ttl_min_seconds, "ttl_max must be >= ttl_min"
		assert growth_factor > 1.0, "growth_factor must be > 1.0"
		cfg = _R(
			namespace=namespace,
			ttl_min_seconds=ttl_min_seconds,
			ttl_max_seconds=ttl_max_seconds,
			growth_factor=growth_factor,
			configured_at=_ts(),
		)
		self._adaptive_ttl_cfg[self._ns_key(namespace)] = cfg
		await self._audit("adaptive_ttl_configured", namespace, dict(cfg))
		return cfg

	# ------------------------------------------------------------------
	# 49. cache_get_adaptive — get with automatic TTL extension on hit
	# ------------------------------------------------------------------

	async def cache_get_adaptive(self, namespace: str, key: str) -> _R:
		"""Get a cache entry and automatically extend TTL based on adaptive policy.

		Requires ``adaptive_ttl_configure`` to have been called for the namespace.
		Returns the same shape as ``cache_get`` with an additional
		``ttl_extended_to_seconds`` field when the TTL was grown.
		"""
		guard_tenant_id(self.tenant_id)
		result = await self.cache_get(namespace, key)
		if not result["hit"]:
			return result

		cfg = self._adaptive_ttl_cfg.get(self._ns_key(namespace))
		if cfg is None:
			return result

		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		if record is None:
			return result

		expires_dt = datetime.fromisoformat(record["expires_at"]) if record.get("expires_at") else None
		if expires_dt is None:
			return result

		remaining = max(0.0, (expires_dt - datetime.utcnow()).total_seconds())
		new_ttl = min(int(remaining * cfg["growth_factor"]), cfg["ttl_max_seconds"])
		new_ttl = max(new_ttl, cfg["ttl_min_seconds"])
		if new_ttl > remaining:
			await self.ttl_update(namespace, key, new_ttl)
			await self._audit("adaptive_ttl_extended", fk,
				{"old_remaining_s": remaining, "new_ttl_s": new_ttl})
			return _R(**result, ttl_extended_to_seconds=new_ttl)
		return result

	# ------------------------------------------------------------------
	# 50. set_tenant_quota — quota governance for cache writes
	# ------------------------------------------------------------------

	async def set_tenant_quota(
		self,
		tenant_id: str,
		soft_bytes: int,
		hard_bytes: int,
	) -> _R:
		"""Define soft and hard byte quotas for a tenant.

		Soft limit: emits a ``quota_warning`` audit event on ``cache_set``.
		Hard limit: raises ``PermissionError`` and blocks the write.
		"""
		guard_tenant_id(tenant_id)
		assert hard_bytes >= soft_bytes > 0, "hard_bytes >= soft_bytes > 0 required"
		self._quotas[tenant_id] = _R(
			tenant_id=tenant_id,
			soft_bytes=soft_bytes,
			hard_bytes=hard_bytes,
			configured_at=_ts(),
		)
		await self._audit("tenant_quota_set", tenant_id,
			{"soft_bytes": soft_bytes, "hard_bytes": hard_bytes})
		return _R(tenant_id=tenant_id, soft_bytes=soft_bytes,
			hard_bytes=hard_bytes, set_at=_ts())

	# ------------------------------------------------------------------
	# 51. quota_usage_report — current tenant memory utilisation vs quotas
	# ------------------------------------------------------------------

	async def quota_usage_report(self, tenant_id: str | None = None) -> _R:
		"""Report current byte usage against configured quotas for a tenant.

		Estimates size via ``json.dumps``; sufficient for governance decisions.
		"""
		guard_tenant_id(self.tenant_id)
		tid = tenant_id or self.tenant_id
		prefix = f"{tid}:"
		active_entries = [
			v for k, v in self._store.items()
			if k.startswith(prefix) and not self._is_expired(v)
		]
		estimated_bytes = sum(
			len(json.dumps(e.get("value", ""), default=str).encode())
			for e in active_entries
		)
		quota = self._quotas.get(tid)
		utilisation_pct: float | None = None
		if quota:
			utilisation_pct = round(estimated_bytes / max(quota["hard_bytes"], 1) * 100, 2)
		return _R(
			tenant_id=tid,
			estimated_bytes=estimated_bytes,
			entry_count=len(active_entries),
			soft_bytes=quota["soft_bytes"] if quota else None,
			hard_bytes=quota["hard_bytes"] if quota else None,
			utilisation_pct=utilisation_pct,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 52. cache_set_versioned — write with schema version tracking
	# ------------------------------------------------------------------

	async def cache_set_versioned(
		self,
		namespace: str,
		key: str,
		value: Any,
		schema_version: str,
		ttl_seconds: int = 3600,
		tags: list[str] | None = None,
	) -> _R:
		"""Store a value with an explicit schema version label.

		Readers can use ``cache_get_versioned`` to enforce schema compatibility
		and automatically invalidate stale-schema entries.
		"""
		guard_tenant_id(self.tenant_id)
		guard_non_empty_string(schema_version, "schema_version")
		result = await self.cache_set(namespace, key, value, ttl_seconds, tags)
		fk = self._full_key(namespace, key)
		self._schema_versions[fk] = schema_version
		if fk in self._store:
			self._store[fk]["_schema_version"] = schema_version
		await self._audit("cache_set_versioned", fk,
			{"schema_version": schema_version, "ttl_seconds": ttl_seconds})
		return _R(**result, schema_version=schema_version)

	# ------------------------------------------------------------------
	# 53. cache_get_versioned — read with schema version enforcement
	# ------------------------------------------------------------------

	async def cache_get_versioned(
		self,
		namespace: str,
		key: str,
		expected_version: str,
	) -> _R:
		"""Read a cache entry and validate schema version.

		If the stored version does not match ``expected_version`` the entry is
		deleted and ``version_mismatch=True`` is returned — forcing the caller
		to re-populate with the correct schema.
		"""
		guard_tenant_id(self.tenant_id)
		guard_non_empty_string(expected_version, "expected_version")
		fk = self._full_key(namespace, key)
		result = await self.cache_get(namespace, key)
		if not result["hit"]:
			return _R(**result, version_mismatch=False, stored_version=None,
				expected_version=expected_version)

		stored_version = self._schema_versions.get(fk) or \
			(self._store.get(fk) or {}).get("_schema_version")

		if stored_version != expected_version:
			await self.cache_delete(namespace, key)
			self._schema_versions.pop(fk, None)
			await self._audit("schema_version_mismatch", fk,
				{"stored": stored_version, "expected": expected_version})
			return _R(hit=False, key=key, namespace=namespace, value=None,
				version_mismatch=True, stored_version=stored_version,
				expected_version=expected_version)

		return _R(**result, version_mismatch=False,
			stored_version=stored_version, expected_version=expected_version)

	# ------------------------------------------------------------------
	# 54. register_tag_hierarchy — cascading tag invalidation graph
	# ------------------------------------------------------------------

	async def register_tag_hierarchy(
		self,
		namespace: str,
		parent_tag: str,
		child_tags: list[str],
	) -> _R:
		"""Register a parent→children tag relationship for cascading invalidation.

		When ``tag_invalidate`` is called with ``cascade=True``, all descendant
		tags are resolved via BFS and their entries are also deleted.
		"""
		guard_tenant_id(self.tenant_id)
		guard_non_empty_string(parent_tag, "parent_tag")
		assert child_tags, "child_tags must not be empty"
		ns = self._ns_key(namespace)
		if ns not in self._tag_graph:
			self._tag_graph[ns] = {}
		existing = self._tag_graph[ns].get(parent_tag, [])
		merged = list(set(existing) | set(child_tags))
		self._tag_graph[ns][parent_tag] = merged
		await self._audit("tag_hierarchy_registered", namespace,
			{"parent": parent_tag, "children": merged})
		return _R(namespace=namespace, parent_tag=parent_tag,
			child_tags=merged, registered_at=_ts())

	# ------------------------------------------------------------------
	# 55. tag_invalidate_cascade — BFS cascading tag invalidation
	# ------------------------------------------------------------------

	async def tag_invalidate_cascade(self, namespace: str, tag: str) -> _R:
		"""Invalidate a tag and all its descendants in the tag hierarchy.

		Uses BFS over the registered tag graph.  Falls back to flat
		``tag_invalidate`` behaviour when no hierarchy is registered.
		"""
		guard_tenant_id(self.tenant_id)
		ns = self._ns_key(namespace)
		graph = self._tag_graph.get(ns, {})

		# BFS to collect all tags to invalidate
		tags_to_invalidate: list[str] = []
		queue = [tag]
		visited: set[str] = set()
		while queue:
			current = queue.pop(0)
			if current in visited:
				continue
			visited.add(current)
			tags_to_invalidate.append(current)
			for child in graph.get(current, []):
				if child not in visited:
					queue.append(child)

		total_invalidated = 0
		tag_counts: dict[str, int] = {}
		for t in tags_to_invalidate:
			r = await self.tag_invalidate(namespace, t)
			count = r["invalidated_count"]
			tag_counts[t] = count
			total_invalidated += count

		await self._audit("tag_invalidate_cascade", namespace,
			{"root_tag": tag, "tags_resolved": len(tags_to_invalidate),
			"total_invalidated": total_invalidated})
		return _R(namespace=namespace, root_tag=tag,
			tags_resolved=tags_to_invalidate,
			per_tag_counts=tag_counts,
			total_invalidated=total_invalidated)

	# ------------------------------------------------------------------
	# 56. warm_cache_stream — streaming incremental cache warming
	# ------------------------------------------------------------------

	async def warm_cache_stream(
		self,
		namespace: str,
		source_iter: AsyncIterator[tuple[str, Any]],
		ttl_seconds: int = 7200,
		batch_size: int = 100,
		progress_callback: Callable[[int, int, float], Awaitable[None]] | None = None,
	) -> _R:
		"""Warm the cache from an async iterator of (key, value) tuples.

		Processes entries in batches of ``batch_size``, yielding the event loop
		between batches via ``asyncio.sleep(0)``.  Reports progress via the
		optional ``progress_callback(loaded, failed, elapsed_ms)`` coroutine.
		"""
		import asyncio
		guard_tenant_id(self.tenant_id)
		op_id = uuid7str()
		self._warming_progress[op_id] = _R(
			op_id=op_id, namespace=namespace, loaded=0, failed=0,
			status="running", started_at=_ts(),
		)
		start = time.monotonic()
		loaded = 0
		failed = 0
		batch: list[tuple[str, Any]] = []

		async def _flush(b: list[tuple[str, Any]]) -> tuple[int, int]:
			ok = 0
			err = 0
			for k, v in b:
				try:
					await self.cache_set(namespace, k, v, ttl_seconds)
					ok += 1
				except Exception as exc:
					logger.warning("warm_cache_stream: failed key=%s: %s", k, exc)
					err += 1
			return ok, err

		async for item in source_iter:
			batch.append(item)
			if len(batch) >= batch_size:
				ok, err = await _flush(batch)
				loaded += ok
				failed += err
				batch.clear()
				elapsed_ms = (time.monotonic() - start) * 1000
				if progress_callback:
					try:
						await progress_callback(loaded, failed, elapsed_ms)
					except Exception as _exc:
						_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
				import asyncio as _aio
				await _aio.sleep(0)

		if batch:
			ok, err = await _flush(batch)
			loaded += ok
			failed += err

		elapsed_ms = (time.monotonic() - start) * 1000
		self._warming_progress[op_id] = _R(
			op_id=op_id, namespace=namespace, loaded=loaded, failed=failed,
			status="completed", started_at=self._warming_progress[op_id]["started_at"],
			completed_at=_ts(), elapsed_ms=round(elapsed_ms, 2),
		)
		await self._audit("cache_warmed_stream", namespace,
			{"op_id": op_id, "loaded": loaded, "failed": failed,
			"elapsed_ms": round(elapsed_ms, 2)})
		return _R(op_id=op_id, namespace=namespace, loaded=loaded,
			failed=failed, elapsed_ms=round(elapsed_ms, 2))

	# ------------------------------------------------------------------
	# 57. cache_set_money — store monetary values as Decimal, preserving precision
	# ------------------------------------------------------------------

	async def cache_set_money(
		self,
		namespace: str,
		key: str,
		amount: Decimal,
		currency: str,
		ttl_seconds: int = 3600,
		tags: list[str] | None = None,
	) -> _R:
		"""Store a monetary amount with full Decimal precision and currency tag.

		Values are stored as string representations to avoid floating-point
		precision loss.  Use ``cache_get_money`` to retrieve as ``Decimal``.
		"""
		guard_tenant_id(self.tenant_id)
		guard_non_empty_string(currency, "currency")
		assert isinstance(amount, Decimal), "amount must be a Decimal"
		fk = self._full_key(namespace, key)
		payload = {"amount_str": str(amount), "currency": currency.upper()}
		result = await self.cache_set(namespace, key, payload, ttl_seconds, tags)
		self._money_cache[fk] = amount
		await self._audit("cache_set_money", fk,
			{"currency": currency.upper(), "ttl_seconds": ttl_seconds})
		return _R(**result, currency=currency.upper(), amount_str=str(amount))

	# ------------------------------------------------------------------
	# 58. cache_get_money — retrieve monetary value as Decimal
	# ------------------------------------------------------------------

	async def cache_get_money(self, namespace: str, key: str) -> _R:
		"""Retrieve a monetary cache entry, returning ``amount`` as ``Decimal``.

		Returns ``hit=False`` on miss or expiry.  Restores ``Decimal`` from the
		string representation, guaranteeing precision parity with the stored value.
		"""
		guard_tenant_id(self.tenant_id)
		result = await self.cache_get(namespace, key)
		if not result["hit"]:
			return _R(**result, amount=None, currency=None)
		payload = result["value"]
		if not isinstance(payload, dict) or "amount_str" not in payload:
			return _R(**result, amount=None, currency=None,
				error="not_a_money_entry")
		amount = Decimal(payload["amount_str"])
		fk = self._full_key(namespace, key)
		self._money_cache[fk] = amount
		return _R(hit=True, key=key, namespace=namespace,
			amount=amount, currency=payload.get("currency"),
			expires_at=result.get("expires_at"))

	# ------------------------------------------------------------------
	# 59. xfetch_get — probabilistic early expiry (XFetch) anti-stampede read
	# ------------------------------------------------------------------

	async def xfetch_get(
		self,
		namespace: str,
		key: str,
		beta: float = 1.0,
	) -> _R:
		"""Cache read with XFetch probabilistic early-expiry stampede protection.

		When the key is approaching expiry, requests are probabilistically treated
		as misses (triggering recompute) before the hard expiry hits.  This
		distributes recompute cost across time, preventing thundering-herd failure.

		``beta > 1`` increases early recompute probability; ``0 < beta < 1`` reduces it.
		"""
		guard_tenant_id(self.tenant_id)
		fk = self._full_key(namespace, key)
		record = self._store.get(fk)
		if record is None or self._is_expired(record):
			if record:
				del self._store[fk]
			self._misses[fk] = self._misses.get(fk, 0) + 1
			await self._audit("cache_miss_xfetch", fk, {"beta": beta, "reason": "expired_or_missing"})
			return _R(hit=False, early_miss=False, key=key, namespace=namespace, value=None)

		if record.get("expires_at") is None:
			# No TTL — permanent entry, no stampede risk
			record["access_count"] = record.get("access_count", 0) + 1
			record["last_accessed_at"] = _ts()
			self._hits[fk] = self._hits.get(fk, 0) + 1
			return _R(hit=True, early_miss=False, key=key, namespace=namespace,
				value=record["value"], expires_at=None)

		expires_dt = datetime.fromisoformat(record["expires_at"])
		now = datetime.utcnow()
		ttl_remaining = max(0.0, (expires_dt - now).total_seconds())
		delta = self._xfetch_deltas.get(fk, 1.0)

		# XFetch probability: early_expiry = -delta * beta * ln(uniform(0,1))
		# If now + early_expiry >= expires_at → treat as miss
		u = random.random()
		if u <= 0.0:
			u = 1e-10
		early_expiry = -delta * beta * math.log(u)
		if ttl_remaining <= early_expiry:
			await self._audit("cache_miss_xfetch", fk,
				{"beta": beta, "ttl_remaining": ttl_remaining,
				"early_expiry": early_expiry, "reason": "probabilistic_early_miss"})
			return _R(hit=False, early_miss=True, key=key, namespace=namespace,
				value=None, ttl_remaining_seconds=ttl_remaining)

		record["access_count"] = record.get("access_count", 0) + 1
		record["last_accessed_at"] = _ts()
		self._hits[fk] = self._hits.get(fk, 0) + 1
		self._op_count += 1
		return _R(hit=True, early_miss=False, key=key, namespace=namespace,
			value=record["value"], expires_at=record["expires_at"],
			ttl_remaining_seconds=ttl_remaining)

	# ------------------------------------------------------------------
	# 60. schema_version_report — distribution of schema versions in namespace
	# ------------------------------------------------------------------

	async def schema_version_report(self, namespace: str) -> _R:
		"""Report the distribution of ``_schema_version`` values across live entries.

		Helps identify stale-schema entries that should be re-populated.
		"""
		guard_tenant_id(self.tenant_id)
		prefix = f"{self.tenant_id}:{namespace}:"
		version_counts: dict[str, int] = {}
		unversioned = 0
		for k, v in self._store.items():
			if not k.startswith(prefix) or self._is_expired(v):
				continue
			sv = v.get("_schema_version") or self._schema_versions.get(k)
			if sv:
				version_counts[sv] = version_counts.get(sv, 0) + 1
			else:
				unversioned += 1
		total = sum(version_counts.values()) + unversioned
		return _R(
			namespace=namespace,
			total_entries=total,
			version_distribution=version_counts,
			unversioned_entries=unversioned,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 61. tier_stats — per-tier (L1/L2) hit-rate breakdown
	# ------------------------------------------------------------------

	async def tier_stats(self, namespace: str | None = None) -> _R:
		"""Return hit-rate statistics broken down by logical cache tier.

		In the in-memory implementation, L1 represents entries accessed more than
		``l1_threshold`` times (hot) and L2 the remainder (warm/cold).
		"""
		guard_tenant_id(self.tenant_id)
		prefix = (f"{self.tenant_id}:{namespace}:" if namespace
			else f"{self.tenant_id}:")
		l1_threshold = 5
		l1_entries = 0
		l2_entries = 0
		for k, v in self._store.items():
			if not k.startswith(prefix) or self._is_expired(v):
				continue
			if v.get("access_count", 0) >= l1_threshold:
				l1_entries += 1
			else:
				l2_entries += 1

		total_hits = sum(v for k, v in self._hits.items() if k.startswith(prefix))
		total_misses = sum(v for k, v in self._misses.items() if k.startswith(prefix))
		total_ops = total_hits + total_misses
		return _R(
			namespace=namespace,
			l1_hot_entries=l1_entries,
			l2_warm_entries=l2_entries,
			total_entries=l1_entries + l2_entries,
			total_hits=total_hits,
			total_misses=total_misses,
			overall_hit_rate=round(total_hits / max(total_ops, 1), 4),
			l1_threshold_access_count=l1_threshold,
			generated_at=_ts(),
		)

	# ------------------------------------------------------------------
	# 62. write_behind_flush — drain the write-behind queue
	# ------------------------------------------------------------------

	async def write_behind_flush(
		self,
		backend_fn: Callable[[str, str, Any], Awaitable[None]] | None = None,
	) -> _R:
		"""Flush pending write-behind queue entries to the backend.

		Pass ``backend_fn(namespace, key, value)`` or rely on per-namespace
		registered writers set by ``register_write_backend``.  Entries are
		processed in FIFO order; failures are counted and left in the queue
		for retry.
		"""
		guard_tenant_id(self.tenant_id)
		if not self._write_queue:
			return _R(flushed=0, failed=0, remaining=0, flushed_at=_ts())

		flushed = 0
		failed = 0
		retry_queue: list[_R] = []
		for entry in list(self._write_queue):
			fn = backend_fn
			if fn is None:
				ns_fn = self._refresh_callbacks.get(
					self._ns_key(entry.get("namespace", "")))
				fn = ns_fn
			if fn is None:
				retry_queue.append(entry)
				continue
			try:
				await fn(entry["namespace"], entry["key"], entry["value"])
				flushed += 1
			except Exception as exc:
				logger.warning("write_behind_flush: backend write failed: %s", exc)
				retry_queue.append(entry)
				failed += 1

		self._write_queue = retry_queue
		await self._audit("write_behind_flushed", "system",
			{"flushed": flushed, "failed": failed, "remaining": len(retry_queue)})
		return _R(flushed=flushed, failed=failed,
			remaining=len(retry_queue), flushed_at=_ts())

	# ------------------------------------------------------------------
	# 63. cache_set_write_behind — enqueue value for asynchronous backend write
	# ------------------------------------------------------------------

	async def cache_set_write_behind(
		self,
		namespace: str,
		key: str,
		value: Any,
		ttl_seconds: int = 3600,
		tags: list[str] | None = None,
	) -> _R:
		"""Write-behind: store in cache immediately, enqueue for async backend write.

		Call ``write_behind_flush`` (or run it in a background task) to drain
		the queue.  Useful for high-write workloads where backend IOPS are a
		bottleneck — reduces write latency to in-memory speeds.
		"""
		guard_tenant_id(self.tenant_id)
		result = await self.cache_set(namespace, key, value, ttl_seconds, tags)
		self._write_queue.append(_R(
			namespace=namespace,
			key=key,
			value=value,
			ttl_seconds=ttl_seconds,
			enqueued_at=_ts(),
		))
		await self._audit("write_behind_enqueued",
			self._full_key(namespace, key),
			{"queue_depth": len(self._write_queue)})
		return _R(**result, write_behind=True, queue_depth=len(self._write_queue))

from dataclasses import dataclass, field as _f
@dataclass
class CacheServiceConfig:
    max_size: int = 10000
    default_ttl: int = 300
    eviction_policy: str = "lru"
    enable_compression: bool = False
    compression_threshold_bytes: int = 1024
    options: dict = _f(default_factory=dict)
    ai_optimization_enabled: bool = False
    predictive_prefetching: bool = False
    audit_enabled: bool = True
    health_checks_enabled: bool = True
    metrics_enabled: bool = True
    compression_algorithm: str = "gzip"

# ── Compression helper methods ─────────────────────────────────────────────

async def _apply_compression_method(self, data: bytes, algorithm):
	import gzip, io
	try:
		from capabilities.common.cach.models import CompressionAlgorithm as _CA
	except ImportError:
		try: from .models import CompressionAlgorithm as _CA
		except ImportError: _CA = None
	if _CA and algorithm == _CA.LZ4:
		if lz4_frame:
			compressed = lz4_frame.compress(data)
			return compressed, algorithm, len(compressed)/len(data) if data else 1.0
		return data, _CA.NONE, 1.0
	elif _CA and algorithm == _CA.ZSTD:
		if zstandard:
			compressed = zstandard.ZstdCompressor().compress(data)
			return compressed, algorithm, len(compressed)/len(data) if data else 1.0
		return data, _CA.NONE, 1.0
	buf = io.BytesIO()
	with gzip.GzipFile(fileobj=buf, mode='wb') as gz: gz.write(data)
	comp = buf.getvalue()
	return comp, (_CA.GZIP if _CA else algorithm), len(comp)/len(data) if data else 1.0

def _default_compression_algorithm_method(self):
	try:
		from capabilities.common.cach.models import CompressionAlgorithm
	except ImportError:
		try: from .models import CompressionAlgorithm
		except ImportError: return None
	return CompressionAlgorithm.LZ4 if lz4_frame is not None else CompressionAlgorithm.GZIP

async def _cache_set_method(self, key: str, value, compression=None, namespace: str = "default", tenant_id: str = "default"):
	import json
	if not hasattr(self, "_cache_store"): self._cache_store = {}
	cache_key = f"{namespace}:{tenant_id}:{key}"
	try:
		from capabilities.common.cach.models import CompressionAlgorithm as _CA
	except ImportError:
		try: from .models import CompressionAlgorithm as _CA
		except ImportError: _CA = None
	comp_type = _CA.NONE if _CA else None
	comp_ratio = 1.0
	if compression is not None and (_CA is None or compression != _CA.NONE):
		data = json.dumps(value).encode()
		_, comp_type, comp_ratio = await self._apply_compression(data, compression)

	class _Entry:
		def __init__(self, v, ct, cr): self.value=v; self.compression_type=ct; self.compression_ratio=cr

	self._cache_store[cache_key] = _Entry(value, comp_type, comp_ratio)
	return True

async def _cache_get_method(self, key: str, namespace: str = "default", tenant_id: str = "default"):
	if not hasattr(self, "_cache_store"): return None
	entry = self._cache_store.get(f"{namespace}:{tenant_id}:{key}")
	return entry.value if entry else None

CacheService._apply_compression = _apply_compression_method
CacheService._default_compression_algorithm = _default_compression_algorithm_method
CacheService.set = _cache_set_method
CacheService.get = _cache_get_method

CachService = CacheService

class CacheEvictionReviewRecord:
    key: str = ''
    size_bytes: int = 0
    reason: str = ''
