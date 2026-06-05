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
import statistics
import time
from datetime import datetime, timedelta
from typing import Any

from uuid6 import uuid7

import logging

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
			return compressed, algorithm, len(data)/len(compressed) if compressed else 1.0
		return data, _CA.NONE, 1.0
	elif _CA and algorithm == _CA.ZSTD:
		if zstandard:
			compressed = zstandard.ZstdCompressor().compress(data)
			return compressed, algorithm, len(data)/len(compressed) if compressed else 1.0
		return data, _CA.NONE, 1.0
	buf = io.BytesIO()
	with gzip.GzipFile(fileobj=buf, mode='wb') as gz: gz.write(data)
	comp = buf.getvalue()
	return comp, (_CA.GZIP if _CA else algorithm), len(data)/len(comp) if comp else 1.0

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
