"""GraphQL Federation Gateway service — federated gateway, schema stitching, DataLoader, persisted queries."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "gql_gw"

# Minimal GQL introspection type stubs
_INTROSPECTION_SCHEMA = {
	"__schema": {
		"queryType": {"name": "Query"},
		"mutationType": {"name": "Mutation"},
		"subscriptionType": None,
		"types": [],
		"directives": [],
	}
}


class GraphQLGatewayService:
	"""Federated GraphQL gateway: subgraph registry, schema composition, DataLoader batching, persisted queries, introspection."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.subgraphs: dict[str, dict[str, Any]] = {}
		self.persisted_queries: dict[str, dict[str, Any]] = {}
		self.query_log: list[dict[str, Any]] = []
		self.dataloader_batches: dict[str, dict[str, Any]] = {}
		self.schema_cache: dict[str, dict[str, Any]] = {}
		self.rate_limits: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _id(self, prefix: str = "") -> str:
		return f"{prefix}-{uuid4().hex[:12]}" if prefix else uuid4().hex[:12]

	def _subgraph_key(self, tenant: str, name: str) -> str:
		return f"{tenant}:{name}"

	def _emit(self, tenant_id: str, event_type: str, operation: str = "", subgraph: str = "", payload: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"operation": operation,
			"subgraph": subgraph,
			"payload": payload or {},
			"created_at": self._now(),
		})

	# ── Health / describe ────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "gql_gw",
			"status": "healthy",
			"subgraph_count": len(self.subgraphs),
			"persisted_query_count": len(self.persisted_queries),
			"query_log_entries": len(self.query_log),
			"checked_at": self._now(),
		}

	async def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"tenant_id": tenant,
			"features": [
				"federated_gateway", "subgraph_registry", "schema_stitching",
				"auto_schema_from_semantic_model", "dataloader_batching",
				"persisted_queries", "introspection", "rate_limiting", "query_audit_log"
			],
		}

	async def get_audit_events(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]

	# ── Subgraph registry ────────────────────────────────────────

	async def register_subgraph(
		self,
		tenant_id: str,
		name: str,
		url: str,
		schema_sdl: str = "",
		health_check_path: str = "/health",
		timeout_ms: int = 5000,
		enabled: bool = True,
	) -> dict[str, Any]:
		"""Register a GraphQL subgraph with the federation gateway."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(url, "url")
		sk = self._subgraph_key(tenant, name)
		record: dict[str, Any] = {
			"id": self._id("sg"),
			"tenant_id": tenant,
			"name": name,
			"url": url,
			"schema_sdl": schema_sdl,
			"health_check_path": health_check_path,
			"timeout_ms": timeout_ms,
			"enabled": enabled,
			"status": "active",
			"created_at": self._now(),
		}
		self.subgraphs[sk] = record
		# Invalidate schema cache
		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "subgraph_registered", subgraph=name, payload={"url": url})
		_log.info("subgraph registered: %s tenant=%s url=%s", name, tenant, url)
		return deepcopy(record)

	async def get_subgraph(self, tenant_id: str, name: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sg = self.subgraphs.get(self._subgraph_key(tenant, name))
		if not sg:
			raise KeyError(f"subgraph not found: {name}")
		return deepcopy(sg)

	async def list_subgraphs(self, tenant_id: str, enabled: bool | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		items = [deepcopy(r) for k, r in self.subgraphs.items() if k.startswith(prefix)]
		if enabled is not None:
			items = [r for r in items if r["enabled"] == enabled]
		return items

	async def update_subgraph(self, tenant_id: str, name: str, **kwargs: Any) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sg = self.subgraphs.get(self._subgraph_key(tenant, name))
		if not sg:
			raise KeyError(f"subgraph not found: {name}")
		for key in ("url", "schema_sdl", "timeout_ms", "enabled"):
			if key in kwargs and kwargs[key] is not None:
				sg[key] = kwargs[key]
		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "subgraph_updated", subgraph=name)
		return deepcopy(sg)

	async def delete_subgraph(self, tenant_id: str, name: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		sk = self._subgraph_key(tenant, name)
		sg = self.subgraphs.get(sk)
		if not sg:
			raise KeyError(f"subgraph not found: {name}")
		del self.subgraphs[sk]
		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "subgraph_deleted", subgraph=name)
		return deepcopy(sg)

	async def probe_subgraph_health(self, tenant_id: str, name: str) -> dict[str, Any]:
		"""Probe a subgraph's health endpoint (in-memory simulation)."""
		tenant = self._tenant(tenant_id)
		sg = self.subgraphs.get(self._subgraph_key(tenant, name))
		if not sg:
			raise KeyError(f"subgraph not found: {name}")
		# Real implementation would do HTTP GET; here we simulate
		start = time.monotonic()
		# Simulate 1–5ms latency
		await asyncio.sleep(0.001)
		latency_ms = round((time.monotonic() - start) * 1000, 2)
		return {
			"subgraph": name,
			"url": sg["url"],
			"healthy": sg["enabled"],
			"latency_ms": latency_ms,
			"checked_at": self._now(),
		}

	# ── Schema composition ────────────────────────────────────────

	async def compose_schema(self, tenant_id: str) -> dict[str, Any]:
		"""Compose a federated schema from all registered subgraphs."""
		tenant = self._tenant(tenant_id)
		cached = self.schema_cache.get(tenant)
		if cached:
			return deepcopy(cached)
		subgraphs = await self.list_subgraphs(tenant_id, enabled=True)
		composed_types: list[str] = []
		for sg in subgraphs:
			if sg["schema_sdl"]:
				composed_types.append(f"# Subgraph: {sg['name']}\n{sg['schema_sdl']}")
		schema: dict[str, Any] = {
			"tenant_id": tenant,
			"subgraph_count": len(subgraphs),
			"subgraphs": [sg["name"] for sg in subgraphs],
			"composed_sdl": "\n\n".join(composed_types),
			"composed_at": self._now(),
		}
		self.schema_cache[tenant] = schema
		self._emit(tenant, "schema_composed", payload={"subgraph_count": len(subgraphs)})
		return deepcopy(schema)

	async def auto_schema_from_semantic_model(self, tenant_id: str, semantic_model: dict[str, Any]) -> dict[str, Any]:
		"""Generate GraphQL SDL from a semantic_model.json structure."""
		tenant = self._tenant(tenant_id)
		entities = semantic_model.get("entities", semantic_model.get("tables", []))
		sdl_parts = []
		for entity in entities:
			name = entity.get("name", "Unknown")
			columns = entity.get("columns", entity.get("fields", []))
			fields_sdl = []
			for col in columns:
				col_name = col.get("name", "field")
				col_type = col.get("type", "String")
				nullable = col.get("nullable", True)
				gql_type = self._map_sql_to_gql_type(col_type)
				required = "!" if not nullable else ""
				fields_sdl.append(f"  {col_name}: {gql_type}{required}")
			fields_block = "\n".join(fields_sdl) if fields_sdl else "  _placeholder: String"
			sdl_parts.append(f"type {name} {{\n{fields_block}\n}}")

		# Auto-generate Query type
		query_fields = "\n".join(
			f"  {e.get('name', 'entity').lower()}(id: ID!): {e.get('name', 'Entity')}\n"
			f"  {e.get('name', 'entity').lower()}List(limit: Int, offset: Int): [{e.get('name', 'Entity')}!]!"
			for e in entities
		)
		sdl_parts.append(f"type Query {{\n{query_fields}\n}}")

		sdl = "\n\n".join(sdl_parts)
		result: dict[str, Any] = {
			"tenant_id": tenant,
			"entity_count": len(entities),
			"generated_sdl": sdl,
			"generated_at": self._now(),
		}
		self._emit(tenant, "schema_auto_generated", payload={"entity_count": len(entities)})
		return result

	def _map_sql_to_gql_type(self, sql_type: str) -> str:
		mapping = {
			"varchar": "String", "text": "String", "char": "String",
			"int": "Int", "integer": "Int", "bigint": "Int", "smallint": "Int",
			"float": "Float", "double": "Float", "decimal": "Float", "numeric": "Float",
			"boolean": "Boolean", "bool": "Boolean",
			"timestamp": "String", "date": "String", "datetime": "String",
			"uuid": "ID", "json": "JSON", "jsonb": "JSON",
		}
		return mapping.get(sql_type.lower().split("(")[0], "String")

	async def introspect(self, tenant_id: str) -> dict[str, Any]:
		"""Return introspection schema representation."""
		tenant = self._tenant(tenant_id)
		schema = await self.compose_schema(tenant_id)
		introspection = deepcopy(_INTROSPECTION_SCHEMA)
		# Populate types from subgraph SDL (simplified parsing)
		for sg_name in schema.get("subgraphs", []):
			sg = self.subgraphs.get(self._subgraph_key(tenant, sg_name), {})
			sdl = sg.get("schema_sdl", "")
			for line in sdl.split("\n"):
				line = line.strip()
				if line.startswith("type "):
					type_name = line.split("{")[0].replace("type ", "").strip()
					if type_name not in ("Query", "Mutation", "Subscription"):
						introspection["__schema"]["types"].append({
							"kind": "OBJECT", "name": type_name, "fields": []
						})
		return introspection

	# ── Query execution ───────────────────────────────────────────

	async def execute_query(
		self,
		tenant_id: str,
		query: str,
		variables: dict[str, Any] | None = None,
		operation_name: str | None = None,
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Execute a GraphQL query against the federated gateway."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")
		start = time.monotonic()

		# Rate limit check
		await self._check_rate_limit(tenant, user_id)

		# Route to appropriate subgraph(s)
		subgraphs = await self.list_subgraphs(tenant_id, enabled=True)
		if not subgraphs:
			return {"data": None, "errors": [{"message": "no_subgraphs_registered"}], "extensions": {}}

		# Parse operation type from query
		op_type = "query"
		stripped = query.strip().lower()
		if stripped.startswith("mutation"):
			op_type = "mutation"
		elif stripped.startswith("subscription"):
			op_type = "subscription"

		# Simulate routing: pick first matching subgraph
		# In real federation this would parse the query and route to subgraphs per field
		target_sg = subgraphs[0]
		duration_ms = round((time.monotonic() - start) * 1000, 2)

		# Log query
		log_entry: dict[str, Any] = {
			"id": self._id("qlog"),
			"tenant_id": tenant,
			"user_id": user_id,
			"operation_type": op_type,
			"operation_name": operation_name,
			"subgraph": target_sg["name"],
			"query_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
			"variables": variables or {},
			"duration_ms": duration_ms,
			"executed_at": self._now(),
		}
		self.query_log.append(log_entry)

		self._emit(tenant, "query_executed", operation=operation_name or op_type, subgraph=target_sg["name"], payload={
			"duration_ms": duration_ms
		})

		# Return simulated response
		return {
			"data": {f"_gateway": {"subgraph": target_sg["name"], "operation": op_type}},
			"errors": None,
			"extensions": {
				"gateway": {"duration_ms": duration_ms, "subgraph": target_sg["name"]},
			},
		}

	async def _check_rate_limit(self, tenant_id: str, user_id: str) -> None:
		"""Simple sliding window rate limit (100 req/min per user)."""
		now = time.time()
		key = f"{tenant_id}:{user_id}"
		rl = self.rate_limits.setdefault(key, {"window_start": now, "count": 0})
		if now - rl["window_start"] > 60:
			rl["window_start"] = now
			rl["count"] = 0
		rl["count"] += 1
		if rl["count"] > 100:
			raise PermissionError("rate_limit_exceeded")

	# ── Persisted queries ─────────────────────────────────────────

	async def register_persisted_query(
		self,
		tenant_id: str,
		query_id: str,
		document: str,
		name: str = "",
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		"""Register a persisted query document."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query_id, "query_id")
		guard_non_empty_string(document, "document")
		pq_key = f"{tenant}:{query_id}"
		record: dict[str, Any] = {
			"id": self._id("pq"),
			"tenant_id": tenant,
			"query_id": query_id,
			"document": document,
			"name": name,
			"tags": list(tags or []),
			"doc_hash": hashlib.sha256(document.encode()).hexdigest(),
			"created_at": self._now(),
		}
		self.persisted_queries[pq_key] = record
		self._emit(tenant, "persisted_query_registered", operation=query_id)
		return deepcopy(record)

	async def execute_persisted_query(
		self,
		tenant_id: str,
		query_id: str,
		variables: dict[str, Any] | None = None,
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Execute a previously registered persisted query."""
		tenant = self._tenant(tenant_id)
		pq = self.persisted_queries.get(f"{tenant}:{query_id}")
		if not pq:
			raise KeyError(f"persisted query not found: {query_id}")
		return await self.execute_query(tenant_id, pq["document"], variables, query_id, user_id)

	async def list_persisted_queries(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		prefix = f"{tenant}:"
		return [deepcopy(r) for k, r in self.persisted_queries.items() if k.startswith(prefix)]

	async def delete_persisted_query(self, tenant_id: str, query_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		pq_key = f"{tenant}:{query_id}"
		pq = self.persisted_queries.get(pq_key)
		if not pq:
			raise KeyError(f"persisted query not found: {query_id}")
		del self.persisted_queries[pq_key]
		self._emit(tenant, "persisted_query_deleted", operation=query_id)
		return deepcopy(pq)

	# ── DataLoader batching ───────────────────────────────────────

	async def dataloader_batch(
		self,
		tenant_id: str,
		loader_key: str,
		ids: list[str],
	) -> dict[str, Any]:
		"""Batch-load entities by IDs using DataLoader semantics (deduplication + batching)."""
		tenant = self._tenant(tenant_id)
		unique_ids = list(dict.fromkeys(ids))  # deduplicate preserving order
		# Simulate batch fetch — real implementation calls the subgraph
		batch_id = self._id("batch")
		results = {uid: {"id": uid, "loaded_from": loader_key, "batch_id": batch_id} for uid in unique_ids}
		batch: dict[str, Any] = {
			"id": batch_id,
			"tenant_id": tenant,
			"loader_key": loader_key,
			"requested_ids": ids,
			"unique_ids": unique_ids,
			"results": results,
			"dedup_savings": len(ids) - len(unique_ids),
			"executed_at": self._now(),
		}
		self.dataloader_batches[batch_id] = batch
		return deepcopy(batch)

	async def list_dataloader_batches(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(b) for b in self.dataloader_batches.values() if b["tenant_id"] == tenant]

	# ── Query log ─────────────────────────────────────────────────

	async def get_query_log(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		entries = [deepcopy(e) for e in self.query_log if e["tenant_id"] == tenant]
		return entries[-limit:]

	async def query_analytics(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		entries = [e for e in self.query_log if e["tenant_id"] == tenant]
		if not entries:
			return {"tenant_id": tenant, "total_queries": 0, "avg_duration_ms": 0.0}
		by_op: dict[str, int] = {}
		by_sg: dict[str, int] = {}
		durations = []
		for e in entries:
			by_op[e["operation_type"]] = by_op.get(e["operation_type"], 0) + 1
			by_sg[e["subgraph"]] = by_sg.get(e["subgraph"], 0) + 1
			durations.append(e["duration_ms"])
		return {
			"tenant_id": tenant,
			"total_queries": len(entries),
			"avg_duration_ms": round(sum(durations) / len(durations), 2),
			"max_duration_ms": max(durations),
			"by_operation_type": by_op,
			"by_subgraph": by_sg,
			"generated_at": self._now(),
		}

	# ── Rate limiting / access control ────────────────────────────

	async def set_rate_limit(self, tenant_id: str, user_id: str, requests_per_minute: int) -> dict[str, Any]:
		"""Configure rate limit for a specific user."""
		tenant = self._tenant(tenant_id)
		key = f"{tenant}:{user_id}"
		config: dict[str, Any] = {
			"tenant_id": tenant,
			"user_id": user_id,
			"requests_per_minute": requests_per_minute,
			"set_at": self._now(),
		}
		self.rate_limits[key] = {**self.rate_limits.get(key, {}), **config}
		return deepcopy(config)

	async def gateway_statistics(self, tenant_id: str) -> dict[str, Any]:
		"""Return gateway-wide statistics."""
		tenant = self._tenant(tenant_id)
		subgraphs = await self.list_subgraphs(tenant_id)
		pqs = await self.list_persisted_queries(tenant_id)
		log = await self.get_query_log(tenant_id, limit=10000)
		return {
			"tenant_id": tenant,
			"subgraph_count": len(subgraphs),
			"enabled_subgraphs": sum(1 for s in subgraphs if s["enabled"]),
			"persisted_query_count": len(pqs),
			"total_queries": len(log),
			"dataloader_batches": len([b for b in self.dataloader_batches.values() if b["tenant_id"] == tenant]),
			"generated_at": self._now(),
		}

	async def flush_schema_cache(self, tenant_id: str) -> dict[str, Any]:
		"""Invalidate the composed schema cache for a tenant."""
		tenant = self._tenant(tenant_id)
		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "schema_cache_flushed")
		return {"tenant_id": tenant, "flushed": True, "flushed_at": self._now()}

	async def probe_all_subgraphs(self, tenant_id: str) -> dict[str, Any]:
		"""Health-probe all subgraphs concurrently."""
		tenant = self._tenant(tenant_id)
		subgraphs = await self.list_subgraphs(tenant_id)
		probes = await asyncio.gather(
			*[self.probe_subgraph_health(tenant_id, sg["name"]) for sg in subgraphs],
			return_exceptions=True,
		)
		results = []
		for sg, probe in zip(subgraphs, probes):
			if isinstance(probe, Exception):
				_log.error("probe failed for %s: %s", sg["name"], probe)
				results.append({"subgraph": sg["name"], "healthy": False, "error": str(probe)})
			else:
				results.append(probe)
		return {
			"tenant_id": tenant,
			"total": len(results),
			"healthy": sum(1 for r in results if r.get("healthy")),
			"results": results,
			"probed_at": self._now(),
		}

	async def get_schema_diff(self, tenant_id: str, subgraph_name: str, new_sdl: str) -> dict[str, Any]:
		"""Detect breaking changes between current and proposed SDL."""
		tenant = self._tenant(tenant_id)
		sg = self.subgraphs.get(self._subgraph_key(tenant, subgraph_name))
		if not sg:
			raise KeyError(f"subgraph not found: {subgraph_name}")
		current_sdl = sg.get("schema_sdl", "")
		current_lines = set(current_sdl.split("\n"))
		new_lines = set(new_sdl.split("\n"))
		added = [l for l in new_lines - current_lines if l.strip()]
		removed = [l for l in current_lines - new_lines if l.strip()]
		# Heuristic: removing a type or field is breaking
		breaking = [l for l in removed if l.strip() and not l.strip().startswith("#")]
		return {
			"subgraph": subgraph_name,
			"added_lines": len(added),
			"removed_lines": len(removed),
			"breaking_changes": breaking[:20],
			"has_breaking_changes": bool(breaking),
			"analysed_at": self._now(),
		}
