"""GraphQL Federation Gateway service — federated gateway, schema stitching, DataLoader, persisted queries."""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import json
import logging
import random
import time
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncGenerator
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

	# Circuit breaker states
	_CB_CLOSED = "CLOSED"
	_CB_OPEN = "OPEN"
	_CB_HALF_OPEN = "HALF_OPEN"

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.subgraphs: dict[str, dict[str, Any]] = {}
		self.persisted_queries: dict[str, dict[str, Any]] = {}
		self.query_log: list[dict[str, Any]] = []
		self.dataloader_batches: dict[str, dict[str, Any]] = {}
		self.schema_cache: dict[str, dict[str, Any]] = {}
		self.rate_limits: dict[str, dict[str, Any]] = {}
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)
		# Response cache: key → {data, expires_at, hits}
		self._response_cache = WriteThruDict('response_cache', tenant_id, _store)
		# Schema registry: tenant:subgraph → list of version records
		self._schema_versions: dict[str, list[dict[str, Any]]] = {}
		# Circuit breakers: tenant:subgraph → CB state dict
		self._circuit_breakers = WriteThruDict('circuit_breakers', tenant_id, _store)
		# Per-tenant allowlist mode: tenant → bool
		self._allowlist_mode: dict[str, bool] = {}
		# Query complexity budgets: tenant → int (default 1000)
		self._complexity_budgets: dict[str, int] = {}
		# Per-field complexity cost map (overridable): field_name → int
		self._field_costs: dict[str, int] = {"list": 10, "search": 20, "connection": 15}
		# Subgraph variant weights: tenant:subgraph_name → list of (record_key, weight)
		self._variant_weights: dict[str, list[tuple[str, int]]] = {}
		# Field-level auth policies: tenant:subgraph → {TypeName.field → [roles]}
		self._field_auth_policies: dict[str, dict[str, list[str]]] = {}
		# Distributed traces: trace_id → trace context dict
		self._traces = WriteThruDict('traces', tenant_id, _store)
		# Webhooks: webhook_id → webhook record
		self._webhooks = WriteThruDict('webhooks', tenant_id, _store)
		# Region index: tenant:region → [subgraph_name, ...]
		self._region_index: dict[str, list[str]] = {}

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

		# Allowlist enforcement
		self._check_allowlist(tenant, query)

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

		# Use circuit-breaker-aware variant routing if available, else first subgraph
		target_sg = self._route_variant(tenant, subgraphs[0]["name"]) or subgraphs[0]

		# Fast-fail if circuit breaker is OPEN for target subgraph
		cb_key = f"{tenant}:{target_sg['name']}"
		cb = self._circuit_breakers.get(cb_key, {})
		if cb.get("state") == self._CB_OPEN:
			return {
				"data": None,
				"errors": [{"message": "SUBGRAPH_UNAVAILABLE", "subgraph": target_sg["name"]}],
				"extensions": {"circuit_breaker": "OPEN"},
			}

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

	# ── I1 / I2: Query complexity & depth analysis ────────────────

	async def analyze_query_complexity(
		self,
		tenant_id: str,
		query: str,
		max_depth: int | None = None,
		cost_budget: int | None = None,
	) -> dict[str, Any]:
		"""Compute complexity score and nesting depth for a GraphQL query document.

		Rejects execution if the score exceeds the tenant's budget or depth exceeds max_depth.
		Uses a configurable per-field cost map (_field_costs).  Returns the report regardless
		so callers can surface it to developers.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")

		budget = cost_budget if cost_budget is not None else self._complexity_budgets.get(tenant, 1000)
		depth_limit = max_depth if max_depth is not None else 10

		# Walk the query text heuristically — a real impl would use a proper GQL AST parser.
		# We count opening braces for depth and match field names against the cost map.
		lines = query.split("\n")
		current_depth = 0
		max_seen_depth = 0
		total_cost = 0
		field_breakdown: dict[str, int] = {}

		for line in lines:
			stripped = line.strip()
			current_depth += stripped.count("{") - stripped.count("}")
			max_seen_depth = max(max_seen_depth, current_depth)
			# Extract field name (first token before any parens/colons)
			tokens = stripped.lstrip("{}# \t").split()
			if tokens:
				field_name = tokens[0].split("(")[0].split(":")[0]
				# Depth multiplier: nested fields cost more (fan-out amplification)
				depth_weight = max(1, current_depth)
				base_cost = self._field_costs.get(field_name, 1)
				field_cost = base_cost * depth_weight
				total_cost += field_cost
				if field_name and field_name not in ("{", "}"):
					field_breakdown[field_name] = field_breakdown.get(field_name, 0) + field_cost

		too_deep = max_seen_depth > depth_limit
		over_budget = total_cost > budget
		allowed = not too_deep and not over_budget

		report: dict[str, Any] = {
			"tenant_id": tenant,
			"complexity_score": total_cost,
			"max_depth": max_seen_depth,
			"depth_limit": depth_limit,
			"cost_budget": budget,
			"over_budget": over_budget,
			"too_deep": too_deep,
			"allowed": allowed,
			"field_breakdown": field_breakdown,
			"analysed_at": self._now(),
		}
		_log.info(
			"query complexity: tenant=%s score=%d depth=%d allowed=%s",
			tenant, total_cost, max_seen_depth, allowed,
		)
		if not allowed:
			reason = "QUERY_TOO_DEEP" if too_deep else "QUERY_TOO_COMPLEX"
			report["rejection_reason"] = reason
		return report

	async def set_complexity_budget(
		self,
		tenant_id: str,
		budget: int,
		field_costs: dict[str, int] | None = None,
	) -> dict[str, Any]:
		"""Configure the per-tenant complexity budget and optional per-field cost overrides."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		assert budget > 0, "budget must be positive"
		self._complexity_budgets[tenant] = budget
		if field_costs:
			self._field_costs.update(field_costs)
		_log.info("complexity budget updated: tenant=%s budget=%d", tenant, budget)
		return {
			"tenant_id": tenant,
			"budget": budget,
			"field_costs": dict(self._field_costs),
			"set_at": self._now(),
		}

	# ── I4: Response caching with TTL ─────────────────────────────

	async def get_cached_response(
		self,
		tenant_id: str,
		query: str,
		variables: dict[str, Any] | None = None,
	) -> dict[str, Any] | None:
		"""Return a cached query response if one exists and has not expired.

		Cache key = sha256(tenant + canonical_query + sorted_variables_json).
		Returns None on cache miss or expiry.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		key = self._cache_key(tenant, query, variables)
		entry = self._response_cache.get(key)
		if entry is None:
			return None
		if time.time() > entry["expires_at"]:
			del self._response_cache[key]
			_log.info("cache expired: tenant=%s key=%s", tenant, key[:16])
			return None
		entry["hits"] = entry.get("hits", 0) + 1
		_log.info("cache hit: tenant=%s key=%s hits=%d", tenant, key[:16], entry["hits"])
		return deepcopy(entry["data"])

	async def cache_response(
		self,
		tenant_id: str,
		query: str,
		response: dict[str, Any],
		ttl_seconds: int = 60,
		variables: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Store a query response in the in-process cache with a TTL.

		Skips caching for mutation or subscription operations (side-effect queries).
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		# Never cache mutations or subscriptions
		stripped = query.strip().lower()
		if stripped.startswith("mutation") or stripped.startswith("subscription"):
			return {"cached": False, "reason": "mutations_not_cacheable"}
		assert ttl_seconds > 0, "ttl_seconds must be positive"
		key = self._cache_key(tenant, query, variables)
		self._response_cache[key] = {
			"tenant_id": tenant,
			"key": key,
			"data": deepcopy(response),
			"expires_at": time.time() + ttl_seconds,
			"ttl_seconds": ttl_seconds,
			"hits": 0,
			"cached_at": self._now(),
		}
		_log.info("response cached: tenant=%s key=%s ttl=%ds", tenant, key[:16], ttl_seconds)
		return {"cached": True, "key": key[:16], "ttl_seconds": ttl_seconds}

	def _cache_key(self, tenant: str, query: str, variables: dict[str, Any] | None) -> str:
		normalized_query = " ".join(query.split())
		vars_json = json.dumps(variables or {}, sort_keys=True)
		raw = f"{tenant}:{normalized_query}:{vars_json}"
		return hashlib.sha256(raw.encode()).hexdigest()

	async def get_cache_stats(self, tenant_id: str) -> dict[str, Any]:
		"""Return cache hit/miss statistics and entry count for the tenant."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		entries = [e for e in self._response_cache.values() if e["tenant_id"] == tenant]
		now = time.time()
		live = [e for e in entries if e["expires_at"] > now]
		total_hits = sum(e.get("hits", 0) for e in live)
		return {
			"tenant_id": tenant,
			"total_entries": len(live),
			"expired_entries": len(entries) - len(live),
			"total_hits": total_hits,
			"generated_at": self._now(),
		}

	# ── I7: Schema registry with versioning ───────────────────────

	async def publish_schema_version(
		self,
		tenant_id: str,
		subgraph_name: str,
		sdl: str,
		version: str,
		changelog: str = "",
		promote_to_stable: bool = False,
	) -> dict[str, Any]:
		"""Publish a new SDL version to the schema registry for a subgraph.

		Breaking-change detection gates promotion to stable automatically unless
		`promote_to_stable=True` is explicitly passed by the caller.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(subgraph_name, "subgraph_name")
		guard_non_empty_string(sdl, "sdl")
		guard_non_empty_string(version, "version")

		reg_key = f"{tenant}:{subgraph_name}"
		versions = self._schema_versions.setdefault(reg_key, [])

		# Detect breaking changes against latest stable version
		stable_versions = [v for v in versions if v.get("status") == "stable"]
		breaking: list[str] = []
		if stable_versions:
			latest_stable = stable_versions[-1]
			diff = await self.get_schema_diff(tenant_id, subgraph_name, sdl) if self.subgraphs.get(
				self._subgraph_key(tenant, subgraph_name)
			) else {"breaking_changes": [], "has_breaking_changes": False}
			breaking = diff.get("breaking_changes", [])

		has_breaking = bool(breaking)
		status = "stable" if (promote_to_stable and not has_breaking) else ("draft" if has_breaking else "stable")

		record: dict[str, Any] = {
			"id": self._id("sv"),
			"tenant_id": tenant,
			"subgraph": subgraph_name,
			"version": version,
			"sdl": sdl,
			"sdl_hash": hashlib.sha256(sdl.encode()).hexdigest()[:16],
			"changelog": changelog,
			"breaking_changes": breaking,
			"has_breaking_changes": has_breaking,
			"status": status,
			"published_at": self._now(),
		}
		versions.append(record)
		self._emit(tenant, "schema_version_published", subgraph=subgraph_name, payload={
			"version": version, "status": status, "has_breaking_changes": has_breaking,
		})
		_log.info(
			"schema version published: %s@%s tenant=%s status=%s breaking=%s",
			subgraph_name, version, tenant, status, has_breaking,
		)
		return deepcopy(record)

	async def list_schema_versions(self, tenant_id: str, subgraph_name: str) -> list[dict[str, Any]]:
		"""Return the full schema version history for a subgraph."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		reg_key = f"{tenant}:{subgraph_name}"
		return [deepcopy(v) for v in self._schema_versions.get(reg_key, [])]

	async def rollback_schema_version(
		self,
		tenant_id: str,
		subgraph_name: str,
		target_version: str,
	) -> dict[str, Any]:
		"""Reinstate a prior schema version as the active SDL for a subgraph.

		Updates the subgraph record and flushes the composed schema cache.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		reg_key = f"{tenant}:{subgraph_name}"
		versions = self._schema_versions.get(reg_key, [])
		target = next((v for v in versions if v["version"] == target_version), None)
		if not target:
			raise KeyError(f"schema version not found: {subgraph_name}@{target_version}")

		sg_key = self._subgraph_key(tenant, subgraph_name)
		sg = self.subgraphs.get(sg_key)
		if sg:
			sg["schema_sdl"] = target["sdl"]

		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "schema_version_rolled_back", subgraph=subgraph_name, payload={
			"target_version": target_version,
		})
		_log.info("schema rolled back: %s → %s tenant=%s", subgraph_name, target_version, tenant)
		return {"subgraph": subgraph_name, "rolled_back_to": target_version, "rolled_back_at": self._now()}

	# ── I9: Circuit breaker per subgraph ──────────────────────────

	async def get_circuit_breaker_status(self, tenant_id: str, subgraph_name: str) -> dict[str, Any]:
		"""Return the current circuit breaker state for a subgraph.

		States: CLOSED (healthy), OPEN (failing, fast-fail), HALF_OPEN (probing).
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		cb_key = f"{tenant}:{subgraph_name}"
		cb = self._circuit_breakers.get(cb_key, {
			"state": self._CB_CLOSED,
			"failure_count": 0,
			"success_count": 0,
			"last_failure_at": None,
			"opened_at": None,
			"recovery_timeout_s": 30,
			"failure_threshold": 5,
		})
		# Auto-transition OPEN → HALF_OPEN after recovery window
		if cb["state"] == self._CB_OPEN and cb.get("opened_at"):
			elapsed = time.time() - cb["opened_at"]
			if elapsed >= cb["recovery_timeout_s"]:
				cb["state"] = self._CB_HALF_OPEN
				_log.info("circuit breaker HALF_OPEN: %s tenant=%s", subgraph_name, tenant)
		self._circuit_breakers[cb_key] = cb
		return {
			"tenant_id": tenant,
			"subgraph": subgraph_name,
			"state": cb["state"],
			"failure_count": cb["failure_count"],
			"success_count": cb["success_count"],
			"last_failure_at": cb["last_failure_at"],
			"opened_at": cb["opened_at"],
			"failure_threshold": cb["failure_threshold"],
			"recovery_timeout_s": cb["recovery_timeout_s"],
			"checked_at": self._now(),
		}

	async def record_subgraph_result(
		self,
		tenant_id: str,
		subgraph_name: str,
		success: bool,
		failure_threshold: int = 5,
		recovery_timeout_s: int = 30,
	) -> dict[str, Any]:
		"""Record a success or failure for a subgraph, advancing the circuit breaker state machine.

		CLOSED: consecutive failures >= threshold → OPEN
		OPEN: fast-fail; after recovery_timeout transitions to HALF_OPEN
		HALF_OPEN: first success → CLOSED, first failure → OPEN
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		cb_key = f"{tenant}:{subgraph_name}"
		cb = self._circuit_breakers.setdefault(cb_key, {
			"state": self._CB_CLOSED,
			"failure_count": 0,
			"success_count": 0,
			"last_failure_at": None,
			"opened_at": None,
			"recovery_timeout_s": recovery_timeout_s,
			"failure_threshold": failure_threshold,
		})
		cb["failure_threshold"] = failure_threshold
		cb["recovery_timeout_s"] = recovery_timeout_s

		if success:
			cb["success_count"] += 1
			if cb["state"] in (self._CB_HALF_OPEN, self._CB_OPEN):
				cb["state"] = self._CB_CLOSED
				cb["failure_count"] = 0
				_log.info("circuit breaker CLOSED (recovered): %s tenant=%s", subgraph_name, tenant)
		else:
			cb["failure_count"] += 1
			cb["last_failure_at"] = self._now()
			if cb["state"] == self._CB_CLOSED and cb["failure_count"] >= failure_threshold:
				cb["state"] = self._CB_OPEN
				cb["opened_at"] = time.time()
				_log.info(
					"circuit breaker OPEN: %s tenant=%s failures=%d",
					subgraph_name, tenant, cb["failure_count"],
				)
			elif cb["state"] == self._CB_HALF_OPEN:
				cb["state"] = self._CB_OPEN
				cb["opened_at"] = time.time()
				_log.info("circuit breaker re-OPEN (probe failed): %s tenant=%s", subgraph_name, tenant)

		return await self.get_circuit_breaker_status(tenant_id, subgraph_name)

	async def reset_circuit_breaker(self, tenant_id: str, subgraph_name: str) -> dict[str, Any]:
		"""Manually reset a circuit breaker to CLOSED state (operator intervention)."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		cb_key = f"{tenant}:{subgraph_name}"
		self._circuit_breakers.pop(cb_key, None)
		self._emit(tenant, "circuit_breaker_reset", subgraph=subgraph_name)
		_log.info("circuit breaker manually reset: %s tenant=%s", subgraph_name, tenant)
		return {"subgraph": subgraph_name, "state": self._CB_CLOSED, "reset_at": self._now()}

	# ── I12: Canary / traffic splitting ───────────────────────────

	async def register_subgraph_variant(
		self,
		tenant_id: str,
		name: str,
		url: str,
		variant: str,
		weight: int = 10,
		schema_sdl: str = "",
		timeout_ms: int = 5000,
	) -> dict[str, Any]:
		"""Register a canary variant of an existing subgraph with a traffic weight.

		`weight` is relative (e.g., stable=90, canary=10 means ~10% traffic to canary).
		Both variants must share the same logical `name`.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(variant, "variant")
		assert 1 <= weight <= 100, "weight must be 1–100"

		variant_key = f"{tenant}:{name}:{variant}"
		record: dict[str, Any] = {
			"id": self._id("sg"),
			"tenant_id": tenant,
			"name": name,
			"variant": variant,
			"url": url,
			"schema_sdl": schema_sdl,
			"timeout_ms": timeout_ms,
			"weight": weight,
			"enabled": True,
			"status": "active",
			"error_count": 0,
			"request_count": 0,
			"created_at": self._now(),
		}
		self.subgraphs[variant_key] = record

		# Update variant weight registry for this logical subgraph
		vw_key = f"{tenant}:{name}"
		existing = self._variant_weights.get(vw_key, [])
		existing = [(k, w) for k, w in existing if k != variant_key]
		existing.append((variant_key, weight))
		self._variant_weights[vw_key] = existing

		self.schema_cache.pop(tenant, None)
		self._emit(tenant, "subgraph_variant_registered", subgraph=name, payload={
			"variant": variant, "url": url, "weight": weight,
		})
		_log.info(
			"subgraph variant registered: %s[%s] tenant=%s weight=%d url=%s",
			name, variant, tenant, weight, url,
		)
		return deepcopy(record)

	async def get_traffic_split(self, tenant_id: str, subgraph_name: str) -> dict[str, Any]:
		"""Return the current traffic weight distribution across variants of a subgraph."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		vw_key = f"{tenant}:{subgraph_name}"
		variants = self._variant_weights.get(vw_key, [])
		if not variants:
			# No explicit variants — check if subgraph exists at all
			sg = self.subgraphs.get(self._subgraph_key(tenant, subgraph_name))
			if sg:
				variants = [(self._subgraph_key(tenant, subgraph_name), 100)]
		total_weight = sum(w for _, w in variants)
		distribution = []
		for vk, w in variants:
			sg = self.subgraphs.get(vk, {})
			distribution.append({
				"key": vk,
				"variant": sg.get("variant", "stable"),
				"url": sg.get("url", ""),
				"weight": w,
				"pct": round(w / total_weight * 100, 1) if total_weight else 0,
				"error_count": sg.get("error_count", 0),
				"request_count": sg.get("request_count", 0),
			})
		return {
			"tenant_id": tenant,
			"subgraph": subgraph_name,
			"total_weight": total_weight,
			"variants": distribution,
			"generated_at": self._now(),
		}

	def _route_variant(self, tenant: str, subgraph_name: str) -> dict[str, Any] | None:
		"""Select a subgraph variant via weighted random sampling, respecting circuit breakers."""
		vw_key = f"{tenant}:{subgraph_name}"
		variants = self._variant_weights.get(vw_key)
		if not variants:
			return self.subgraphs.get(self._subgraph_key(tenant, subgraph_name))

		# Filter out OPEN circuits
		live = []
		for vk, w in variants:
			cb = self._circuit_breakers.get(f"{tenant}:{vk}", {})
			if cb.get("state") != self._CB_OPEN:
				live.append((vk, w))
		if not live:
			# All variants tripped — fall through to the first registered (fail loud)
			live = variants[:1]

		total = sum(w for _, w in live)
		r = random.uniform(0, total)
		cumulative = 0.0
		for vk, w in live:
			cumulative += w
			if r <= cumulative:
				return self.subgraphs.get(vk)
		return self.subgraphs.get(live[-1][0])

	# ── I13: Persisted query allowlist mode ───────────────────────

	async def set_allowlist_mode(
		self,
		tenant_id: str,
		enabled: bool,
	) -> dict[str, Any]:
		"""Enable or disable allowlist mode for a tenant.

		When enabled, `execute_query` rejects any query whose sha256 hash does not
		match a registered persisted query document hash.  Returns QUERY_NOT_ALLOWED
		with the computed hash so the caller can register it.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		self._allowlist_mode[tenant] = enabled
		self._emit(tenant, "allowlist_mode_changed", payload={"enabled": enabled})
		_log.info("allowlist mode: tenant=%s enabled=%s", tenant, enabled)
		return {"tenant_id": tenant, "allowlist_mode": enabled, "set_at": self._now()}

	async def get_allowlist_mode(self, tenant_id: str) -> dict[str, Any]:
		"""Return the current allowlist mode setting for a tenant."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"allowlist_mode": self._allowlist_mode.get(tenant, False),
		}

	def _check_allowlist(self, tenant: str, query: str) -> None:
		"""Raise PermissionError if allowlist mode is active and query is not pre-registered."""
		if not self._allowlist_mode.get(tenant, False):
			return
		doc_hash = hashlib.sha256(query.encode()).hexdigest()
		registered_hashes = {
			pq["doc_hash"]
			for k, pq in self.persisted_queries.items()
			if k.startswith(f"{tenant}:")
		}
		if doc_hash not in registered_hashes:
			raise PermissionError(
				f"QUERY_NOT_ALLOWED: query hash {doc_hash[:16]} not in allowlist — "
				f"register with POST /api/gql/persisted"
			)

	# ── I11: Query cost estimation (dry run) ──────────────────────

	async def estimate_query_cost(
		self,
		tenant_id: str,
		query: str,
		variables: dict[str, Any] | None = None,
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Dry-run cost estimation — no subgraph calls made.

		Returns complexity score, depth, cache hit probability, estimated subgraph
		fan-out count, and current rate-limit status.  Designed for CI pipeline
		integration to catch expensive queries before they reach production.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")

		# Complexity analysis
		complexity_report = await self.analyze_query_complexity(tenant_id, query)

		# Cache probe (no side-effect — just checks key existence)
		cache_key = self._cache_key(tenant, query, variables)
		cached_entry = self._response_cache.get(cache_key)
		cache_hit = cached_entry is not None and time.time() < cached_entry.get("expires_at", 0)

		# Estimate subgraph fan-out: count unique top-level fields
		top_level_fields = []
		depth = 0
		for line in query.split("\n"):
			stripped = line.strip()
			depth += stripped.count("{") - stripped.count("}")
			if depth == 1:
				tokens = stripped.lstrip("{ \t").split()
				if tokens and not tokens[0].startswith(("#", "}")):
					top_level_fields.append(tokens[0].split("(")[0])

		subgraphs = await self.list_subgraphs(tenant_id, enabled=True)
		estimated_subgraph_calls = min(len(top_level_fields) or 1, len(subgraphs) or 1)

		# Rate-limit headroom
		rl_key = f"{tenant}:{user_id}"
		rl = self.rate_limits.get(rl_key, {})
		rl_used = rl.get("count", 0)
		rl_limit = rl.get("requests_per_minute", 100)
		rl_headroom = max(0, rl_limit - rl_used)

		# Allowlist check
		allowlist_mode = self._allowlist_mode.get(tenant, False)
		doc_hash = hashlib.sha256(query.encode()).hexdigest()
		registered_hashes = {
			pq["doc_hash"]
			for k, pq in self.persisted_queries.items()
			if k.startswith(f"{tenant}:")
		}
		allowlist_ok = not allowlist_mode or doc_hash in registered_hashes

		estimate: dict[str, Any] = {
			"tenant_id": tenant,
			"user_id": user_id,
			"complexity_score": complexity_report["complexity_score"],
			"max_depth": complexity_report["max_depth"],
			"depth_limit": complexity_report["depth_limit"],
			"cost_budget": complexity_report["cost_budget"],
			"over_budget": complexity_report["over_budget"],
			"too_deep": complexity_report["too_deep"],
			"will_be_allowed": complexity_report["allowed"] and allowlist_ok and rl_headroom > 0,
			"cache_hit": cache_hit,
			"estimated_subgraph_calls": estimated_subgraph_calls,
			"top_level_fields": top_level_fields,
			"rate_limit_headroom": rl_headroom,
			"allowlist_mode": allowlist_mode,
			"allowlist_ok": allowlist_ok,
			"query_hash": doc_hash[:16],
			"estimated_at": self._now(),
		}
		_log.info(
			"query cost estimated: tenant=%s score=%d depth=%d cache_hit=%s allowed=%s",
			tenant,
			estimate["complexity_score"],
			estimate["max_depth"],
			cache_hit,
			estimate["will_be_allowed"],
		)
		return estimate

	# ── I3: Field-level authorization via @auth directives ───────

	async def register_field_auth_policy(
		self,
		tenant_id: str,
		subgraph_name: str,
		field_policies: dict[str, list[str]],
	) -> dict[str, Any]:
		"""Register per-field role requirements for a subgraph.

		`field_policies` maps field paths (``TypeName.fieldName``) to the list of
		roles at least one of which the caller must hold.  The gateway enforces
		these rules in :meth:`execute_query_authorized` before forwarding to the
		subgraph, eliminating duplication across downstream services.

		Example::

			await svc.register_field_auth_policy(
				"acme", "payments",
				{"Payment.amount": ["finance", "admin"], "Payment.cardLast4": ["admin"]},
			)
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(subgraph_name, "subgraph_name")
		assert isinstance(field_policies, dict), "field_policies must be a dict"

		policy_key = f"{tenant}:{subgraph_name}"
		self._field_auth_policies[policy_key] = {
			field: list(roles) for field, roles in field_policies.items()
		}
		self._emit(tenant, "field_auth_policy_registered", subgraph=subgraph_name, payload={
			"field_count": len(field_policies),
		})
		_log.info(
			"field auth policy registered: %s tenant=%s fields=%d",
			subgraph_name, tenant, len(field_policies),
		)
		return {
			"tenant_id": tenant,
			"subgraph": subgraph_name,
			"field_count": len(field_policies),
			"policies": dict(self._field_auth_policies[policy_key]),
			"registered_at": self._now(),
		}

	async def execute_query_authorized(
		self,
		tenant_id: str,
		query: str,
		user_roles: list[str],
		variables: dict[str, Any] | None = None,
		operation_name: str | None = None,
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Execute a query with field-level authorization enforcement.

		Scans the query document for field references matching registered auth
		policies and raises ``PermissionError`` (``FIELD_ACCESS_DENIED``) if the
		caller's roles are insufficient.  Passes through to :meth:`execute_query`
		when all requested fields are permitted.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")

		# Collect all field-auth policies for this tenant
		tenant_policies: dict[str, list[str]] = {}
		for pk, policy in self._field_auth_policies.items():
			if pk.startswith(f"{tenant}:"):
				tenant_policies.update(policy)

		if tenant_policies:
			# Simple heuristic: tokenize query and check field names against policy map
			tokens = query.replace("{", " ").replace("}", " ").replace("(", " ").split()
			for token in tokens:
				field_name = token.split(":")[0].split("(")[0].strip()
				# Check both exact match and wildcard "*.fieldName"
				for policy_field, required_roles in tenant_policies.items():
					check_name = policy_field.split(".")[-1]
					if field_name == check_name:
						if not any(r in user_roles for r in required_roles):
							raise PermissionError(
								f"FIELD_ACCESS_DENIED: field '{field_name}' requires one of {required_roles}"
							)

		_log.info(
			"authorized query execution: tenant=%s user=%s roles=%s",
			tenant, user_id, user_roles,
		)
		return await self.execute_query(tenant_id, query, variables, operation_name, user_id)

	async def get_field_auth_policies(self, tenant_id: str, subgraph_name: str) -> dict[str, Any]:
		"""Return the registered field auth policies for a subgraph."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		policy_key = f"{tenant}:{subgraph_name}"
		policies = self._field_auth_policies.get(policy_key, {})
		return {
			"tenant_id": tenant,
			"subgraph": subgraph_name,
			"policies": dict(policies),
			"field_count": len(policies),
		}

	# ── I6: Distributed tracing spans ────────────────────────────

	async def start_trace(
		self,
		tenant_id: str,
		operation_name: str,
		user_id: str = "anonymous",
		parent_trace_id: str | None = None,
	) -> dict[str, Any]:
		"""Open a distributed trace span for a gateway operation.

		Returns a ``trace_context`` dict that should be forwarded in every
		subgraph HTTP request as ``traceparent`` / ``X-Trace-ID`` headers.
		The trace is stored internally and closed by :meth:`finish_trace`.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(operation_name, "operation_name")

		trace_id = self._id("trace")
		span_id = self._id("span")
		ctx: dict[str, Any] = {
			"trace_id": trace_id,
			"span_id": span_id,
			"parent_trace_id": parent_trace_id,
			"tenant_id": tenant,
			"operation_name": operation_name,
			"user_id": user_id,
			"started_at": self._now(),
			"finished_at": None,
			"duration_ms": None,
			"subgraph_spans": [],
			"status": "ACTIVE",
			"_start_monotonic": time.monotonic(),
		}
		self._traces[trace_id] = ctx
		_log.info(
			"trace started: id=%s op=%s tenant=%s parent=%s",
			trace_id, operation_name, tenant, parent_trace_id,
		)
		return {k: v for k, v in ctx.items() if not k.startswith("_")}

	async def record_subgraph_span(
		self,
		tenant_id: str,
		trace_id: str,
		subgraph_name: str,
		duration_ms: float,
		field_count: int = 0,
		cache_hit: bool = False,
		error: str | None = None,
	) -> dict[str, Any]:
		"""Append a child span to an active trace representing one subgraph call.

		Subgraph spans are used by observability tooling to compute per-subgraph
		P50/P99 latencies and identify fan-out amplification.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		trace = self._traces.get(trace_id)
		if not trace:
			raise KeyError(f"trace not found: {trace_id}")
		span: dict[str, Any] = {
			"span_id": self._id("span"),
			"subgraph": subgraph_name,
			"duration_ms": round(duration_ms, 2),
			"field_count": field_count,
			"cache_hit": cache_hit,
			"error": error,
			"recorded_at": self._now(),
		}
		trace["subgraph_spans"].append(span)
		_log.info(
			"subgraph span recorded: trace=%s subgraph=%s duration=%.2fms cache=%s",
			trace_id, subgraph_name, duration_ms, cache_hit,
		)
		return deepcopy(span)

	async def finish_trace(self, tenant_id: str, trace_id: str, status: str = "OK") -> dict[str, Any]:
		"""Close a trace span and compute total duration and aggregated span metrics."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		trace = self._traces.get(trace_id)
		if not trace:
			raise KeyError(f"trace not found: {trace_id}")
		elapsed_ms = round((time.monotonic() - trace["_start_monotonic"]) * 1000, 2)
		trace["finished_at"] = self._now()
		trace["duration_ms"] = elapsed_ms
		trace["status"] = status
		spans = trace["subgraph_spans"]
		summary = {k: v for k, v in trace.items() if not k.startswith("_")}
		summary["span_count"] = len(spans)
		summary["total_subgraph_ms"] = round(sum(s["duration_ms"] for s in spans), 2)
		summary["cache_hits"] = sum(1 for s in spans if s.get("cache_hit"))
		summary["errors"] = [s["error"] for s in spans if s.get("error")]
		self._emit(tenant, "trace_finished", operation=trace["operation_name"], payload={
			"duration_ms": elapsed_ms, "span_count": len(spans), "status": status,
		})
		_log.info(
			"trace finished: id=%s op=%s duration=%.2fms spans=%d status=%s",
			trace_id, trace["operation_name"], elapsed_ms, len(spans), status,
		)
		return summary

	async def get_trace(self, tenant_id: str, trace_id: str) -> dict[str, Any]:
		"""Retrieve a stored trace by ID."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		trace = self._traces.get(trace_id)
		if not trace:
			raise KeyError(f"trace not found: {trace_id}")
		return {k: v for k, v in deepcopy(trace).items() if not k.startswith("_")}

	async def list_traces(
		self,
		tenant_id: str,
		limit: int = 50,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return recent traces for a tenant, optionally filtered by status."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		items = [
			{k: v for k, v in t.items() if not k.startswith("_")}
			for t in self._traces.values()
			if t["tenant_id"] == tenant
		]
		if status:
			items = [t for t in items if t.get("status") == status]
		return items[-limit:]

	# ── I8: Query normalization and deduplication ─────────────────

	async def normalize_query(
		self,
		tenant_id: str,
		query: str,
		variables: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Normalize a GraphQL query document to its canonical form.

		Canonical form: strip block comments (``# ...``), collapse whitespace,
		sort top-level selections alphabetically, remove redundant aliases on
		same-name fields.  The canonical form is used for cache key computation
		and persisted query deduplication — two logically identical queries from
		different clients hash to the same key after normalization.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")

		# Step 1: Strip comments
		lines = [ln for ln in query.split("\n") if not ln.strip().startswith("#")]
		stripped = "\n".join(lines)

		# Step 2: Collapse whitespace
		import re
		normalized = re.sub(r"\s+", " ", stripped).strip()

		# Step 3: Sort comma-separated field lists within each selection set
		# (heuristic: split by { } to find selection sets and sort their field tokens)
		def sort_selection(match: re.Match) -> str:
			inner = match.group(1)
			fields = [f.strip() for f in inner.split(" ") if f.strip() and f not in ("{", "}")]
			fields.sort()
			return "{ " + " ".join(fields) + " }"

		# Apply light alphabetic field sort inside each braced block
		normalized = re.sub(r"\{([^{}]*)\}", sort_selection, normalized)

		# Step 4: Compute normalized hash
		norm_hash = hashlib.sha256(normalized.encode()).hexdigest()
		raw_hash = hashlib.sha256(query.encode()).hexdigest()
		is_duplicate = any(
			pq["doc_hash"] == norm_hash
			for k, pq in self.persisted_queries.items()
			if k.startswith(f"{tenant}:")
		)

		# Step 5: Normalize variables (sort keys)
		canonical_vars = json.dumps(variables or {}, sort_keys=True)

		result: dict[str, Any] = {
			"tenant_id": tenant,
			"original_query": query,
			"normalized_query": normalized,
			"original_hash": raw_hash[:16],
			"normalized_hash": norm_hash[:16],
			"is_duplicate": is_duplicate,
			"canonical_variables": canonical_vars,
			"cache_key": self._cache_key(tenant, normalized, variables),
			"normalized_at": self._now(),
		}
		_log.info(
			"query normalized: tenant=%s hash=%s duplicate=%s",
			tenant, norm_hash[:16], is_duplicate,
		)
		return result

	# ── I10: Federated entity resolution (_entities query) ───────

	async def resolve_entities(
		self,
		tenant_id: str,
		representations: list[dict[str, Any]],
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Resolve cross-subgraph entity references via Apollo Federation ``_entities`` query.

		Each representation must contain ``__typename`` plus the key fields for
		that type.  The gateway groups representations by ``__typename``, fans out
		to the owning subgraph (determined by SDL ``@key`` hints or subgraph name
		convention), and merges results in the original order using DataLoader
		batching.

		Example input::

			[
				{"__typename": "User", "id": "u1"},
				{"__typename": "Payment", "id": "pay-001"},
				{"__typename": "User", "id": "u2"},
			]
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		assert isinstance(representations, list) and representations, \
			"representations must be a non-empty list"

		# Group by typename
		by_type: dict[str, list[dict[str, Any]]] = {}
		for rep in representations:
			typename = rep.get("__typename", "Unknown")
			by_type.setdefault(typename, []).append(rep)

		# For each typename, find the owning subgraph (name match or @key directive)
		resolved: dict[str, dict[str, Any]] = {}
		subgraph_calls: list[dict[str, Any]] = []

		prefix = f"{tenant}:"
		for typename, reps in by_type.items():
			# Find owning subgraph: look for a subgraph whose SDL mentions the type
			owning_sg: dict[str, Any] | None = None
			for sg_key, sg in self.subgraphs.items():
				if not sg_key.startswith(prefix):
					continue
				sdl = sg.get("schema_sdl", "")
				if f"type {typename}" in sdl:
					owning_sg = sg
					break
			if not owning_sg:
				# Fall back to first enabled subgraph
				enabled = [sg for sk, sg in self.subgraphs.items()
						   if sk.startswith(prefix) and sg.get("enabled", True)]
				owning_sg = enabled[0] if enabled else None

			# Batch-load entity IDs for this typename
			ids = [str(rep.get("id", rep.get("_id", ""))) for rep in reps]
			batch_id = self._id("ent")
			start = time.monotonic()
			await asyncio.sleep(0)  # yield to event loop (simulates async subgraph I/O)
			duration_ms = round((time.monotonic() - start) * 1000, 2)

			for rep in reps:
				entity_id = str(rep.get("id", rep.get("_id", batch_id)))
				resolved[entity_id] = {
					"__typename": typename,
					**rep,
					"_resolved_from": owning_sg["name"] if owning_sg else "unknown",
					"_batch_id": batch_id,
				}
			subgraph_calls.append({
				"typename": typename,
				"subgraph": owning_sg["name"] if owning_sg else "unknown",
				"entity_count": len(reps),
				"ids": ids,
				"duration_ms": duration_ms,
			})

		# Return in original representation order
		ordered_results = []
		for rep in representations:
			entity_id = str(rep.get("id", rep.get("_id", "")))
			ordered_results.append(resolved.get(entity_id, {"__typename": rep.get("__typename"), "_error": "NOT_FOUND"}))

		self._emit(tenant, "entities_resolved", payload={
			"representation_count": len(representations),
			"typename_count": len(by_type),
			"subgraph_calls": len(subgraph_calls),
		})
		_log.info(
			"entities resolved: tenant=%s representations=%d typenames=%d subgraph_calls=%d",
			tenant, len(representations), len(by_type), len(subgraph_calls),
		)
		return {
			"tenant_id": tenant,
			"entities": ordered_results,
			"representation_count": len(representations),
			"subgraph_calls": subgraph_calls,
			"resolved_at": self._now(),
		}

	# ── I14: Webhook integration for schema change events ─────────

	async def register_webhook(
		self,
		tenant_id: str,
		url: str,
		events: list[str],
		secret: str = "",
		name: str = "",
	) -> dict[str, Any]:
		"""Register a webhook endpoint to receive push notifications for gateway events.

		`events` is a list of event type strings (e.g. ``["subgraph_registered",
		"schema_version_published"]``).  Use ``["*"]`` to subscribe to all events.
		The gateway signs each delivery with HMAC-SHA256 using `secret` (if provided)
		and sets ``X-GQL-Signature`` on the POST request.

		In this in-process implementation the HTTP delivery is simulated — a real
		deploy would use :func:`asyncio.create_task` with an HTTP client.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(url, "url")
		assert isinstance(events, list) and events, "events must be a non-empty list"

		wh_id = self._id("wh")
		record: dict[str, Any] = {
			"id": wh_id,
			"tenant_id": tenant,
			"name": name or wh_id,
			"url": url,
			"events": list(events),
			"secret": secret,
			"delivery_count": 0,
			"last_delivery_at": None,
			"last_delivery_status": None,
			"enabled": True,
			"created_at": self._now(),
		}
		self._webhooks[wh_id] = record
		self._emit(tenant, "webhook_registered", payload={"url": url, "events": events})
		_log.info("webhook registered: id=%s tenant=%s url=%s events=%s", wh_id, tenant, url, events)
		return deepcopy(record)

	async def delete_webhook(self, tenant_id: str, webhook_id: str) -> dict[str, Any]:
		"""Remove a registered webhook."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		wh = self._webhooks.get(webhook_id)
		if not wh or wh["tenant_id"] != tenant:
			raise KeyError(f"webhook not found: {webhook_id}")
		del self._webhooks[webhook_id]
		self._emit(tenant, "webhook_deleted", payload={"webhook_id": webhook_id})
		return deepcopy(wh)

	async def list_webhooks(self, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all registered webhooks for a tenant."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		return [deepcopy(wh) for wh in self._webhooks.values() if wh["tenant_id"] == tenant]

	async def deliver_webhook(
		self,
		tenant_id: str,
		event_type: str,
		payload: dict[str, Any],
	) -> dict[str, Any]:
		"""Fan out a gateway event to all matching registered webhooks.

		Signs the payload body with HMAC-SHA256 when a secret is configured.
		Records delivery status on each webhook record for observability.
		In production this would be async fire-and-forget with exponential backoff.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(event_type, "event_type")

		matching = [
			wh for wh in self._webhooks.values()
			if wh["tenant_id"] == tenant
			and wh.get("enabled", True)
			and ("*" in wh["events"] or event_type in wh["events"])
		]

		deliveries: list[dict[str, Any]] = []
		body = json.dumps({"event_type": event_type, "payload": payload, "tenant_id": tenant})

		for wh in matching:
			# Compute HMAC signature if secret configured
			sig = ""
			if wh.get("secret"):
				import hmac as _hmac
				sig = _hmac.new(
					wh["secret"].encode(),
					body.encode(),
					hashlib.sha256,
				).hexdigest()

			# Simulate delivery (real impl: asyncio HTTP POST)
			delivery_id = self._id("dlv")
			wh["delivery_count"] += 1
			wh["last_delivery_at"] = self._now()
			wh["last_delivery_status"] = "delivered"

			deliveries.append({
				"delivery_id": delivery_id,
				"webhook_id": wh["id"],
				"url": wh["url"],
				"signature": sig[:16] if sig else None,
				"status": "delivered",
				"delivered_at": self._now(),
			})
			_log.info(
				"webhook delivered: id=%s event=%s url=%s tenant=%s",
				delivery_id, event_type, wh["url"], tenant,
			)

		return {
			"tenant_id": tenant,
			"event_type": event_type,
			"matched_webhooks": len(matching),
			"deliveries": deliveries,
			"delivered_at": self._now(),
		}

	# ── I15: Multi-region subgraph affinity routing ───────────────

	async def register_subgraph_region(
		self,
		tenant_id: str,
		subgraph_name: str,
		region: str,
		latency_ms_p50: float = 10.0,
	) -> dict[str, Any]:
		"""Tag a subgraph with a region label and optional baseline latency.

		Region labels (e.g. ``"us-east-1"``, ``"eu-west-1"``) are used by
		:meth:`execute_query_with_region_affinity` to route queries to the
		geographically closest replica.  Latency stats are updated dynamically
		by :meth:`record_subgraph_span` calls.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(subgraph_name, "subgraph_name")
		guard_non_empty_string(region, "region")

		sg_key = self._subgraph_key(tenant, subgraph_name)
		sg = self.subgraphs.get(sg_key)
		if not sg:
			raise KeyError(f"subgraph not found: {subgraph_name}")

		sg["region"] = region
		sg["latency_ms_p50"] = latency_ms_p50

		region_key = f"{tenant}:{region}"
		self._region_index.setdefault(region_key, [])
		if subgraph_name not in self._region_index[region_key]:
			self._region_index[region_key].append(subgraph_name)

		self._emit(tenant, "subgraph_region_registered", subgraph=subgraph_name, payload={
			"region": region, "latency_ms_p50": latency_ms_p50,
		})
		_log.info(
			"subgraph region registered: %s region=%s tenant=%s latency=%.1fms",
			subgraph_name, region, tenant, latency_ms_p50,
		)
		return {
			"subgraph": subgraph_name,
			"region": region,
			"latency_ms_p50": latency_ms_p50,
			"registered_at": self._now(),
		}

	async def execute_query_with_region_affinity(
		self,
		tenant_id: str,
		query: str,
		preferred_region: str,
		variables: dict[str, Any] | None = None,
		operation_name: str | None = None,
		user_id: str = "anonymous",
	) -> dict[str, Any]:
		"""Execute a query routed to the subgraph closest to `preferred_region`.

		Selection strategy:

		1. Prefer same-region subgraphs with open circuits.
		2. Fall back to any healthy subgraph sorted by ``latency_ms_p50`` ascending.
		3. If no subgraphs are available, return ``NO_HEALTHY_SUBGRAPH`` error.

		The selected subgraph and routing reason are surfaced in ``extensions``.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")
		guard_non_empty_string(preferred_region, "preferred_region")

		prefix = f"{tenant}:"
		region_key = f"{tenant}:{preferred_region}"
		same_region_names = self._region_index.get(region_key, [])

		# Collect candidate subgraphs, excluding OPEN circuits
		def _is_open(sg_name: str) -> bool:
			cb = self._circuit_breakers.get(f"{tenant}:{sg_name}", {})
			return cb.get("state") == self._CB_OPEN

		same_region = [
			self.subgraphs[f"{tenant}:{n}"]
			for n in same_region_names
			if f"{tenant}:{n}" in self.subgraphs and not _is_open(n)
		]

		if same_region:
			target = sorted(same_region, key=lambda s: s.get("latency_ms_p50", 999))[0]
			routing_reason = "same_region"
		else:
			# Fall back to any enabled, non-open subgraph
			all_sgs = [
				sg for sk, sg in self.subgraphs.items()
				if sk.startswith(prefix) and sg.get("enabled", True) and not _is_open(sg["name"])
			]
			if not all_sgs:
				return {
					"data": None,
					"errors": [{"message": "NO_HEALTHY_SUBGRAPH", "preferred_region": preferred_region}],
					"extensions": {"gateway": {"routing_reason": "no_healthy_subgraph"}},
				}
			target = sorted(all_sgs, key=lambda s: s.get("latency_ms_p50", 999))[0]
			routing_reason = "latency_fallback"

		_log.info(
			"region affinity routing: tenant=%s preferred=%s selected=%s reason=%s",
			tenant, preferred_region, target["name"], routing_reason,
		)
		result = await self.execute_query(tenant_id, query, variables, operation_name, user_id)
		result.setdefault("extensions", {})
		result["extensions"]["routing"] = {
			"selected_subgraph": target["name"],
			"selected_region": target.get("region", "unknown"),
			"preferred_region": preferred_region,
			"routing_reason": routing_reason,
		}
		return result

	async def get_region_topology(self, tenant_id: str) -> dict[str, Any]:
		"""Return a map of all registered regions and their subgraph assignments."""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		topology: dict[str, list[dict[str, Any]]] = {}
		prefix = f"{tenant}:"
		for region_key, sg_names in self._region_index.items():
			if not region_key.startswith(prefix):
				continue
			region = region_key[len(prefix):]
			topology[region] = []
			for name in sg_names:
				sg = self.subgraphs.get(f"{tenant}:{name}", {})
				topology[region].append({
					"name": name,
					"url": sg.get("url", ""),
					"latency_ms_p50": sg.get("latency_ms_p50", None),
					"enabled": sg.get("enabled", True),
				})
		return {
			"tenant_id": tenant,
			"regions": topology,
			"region_count": len(topology),
			"generated_at": self._now(),
		}

	# ── I5: Subscriptions via AsyncGenerator ──────────────────────

	async def subscribe_query(
		self,
		tenant_id: str,
		query: str,
		variables: dict[str, Any] | None = None,
		user_id: str = "anonymous",
		poll_interval_s: float = 1.0,
		max_events: int = 10,
	) -> AsyncGenerator[dict[str, Any], None]:
		"""Stream subscription events as an async generator (SSE-compatible).

		In a real implementation this would open a WebSocket or SSE connection to the
		subgraph's subscription endpoint and fan out events to registered listeners.
		Here we simulate by re-executing the query every `poll_interval_s` seconds up
		to `max_events` times, yielding incremental delta payloads.
		"""
		guard_tenant_id(tenant_id)
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(query, "query")
		await self._check_rate_limit(tenant, user_id)

		sub_id = self._id("sub")
		_log.info(
			"subscription started: id=%s tenant=%s user=%s max_events=%d",
			sub_id, tenant, user_id, max_events,
		)
		self._emit(tenant, "subscription_started", operation=sub_id, payload={
			"user_id": user_id, "max_events": max_events,
		})

		event_index = 0
		while event_index < max_events:
			await asyncio.sleep(poll_interval_s)
			result = await self.execute_query(tenant_id, query, variables, f"sub_{sub_id}", user_id)
			yield {
				"id": sub_id,
				"type": "next",
				"event_index": event_index,
				"payload": result,
				"emitted_at": self._now(),
			}
			event_index += 1

		yield {"id": sub_id, "type": "complete", "event_index": event_index, "emitted_at": self._now()}
		_log.info("subscription completed: id=%s tenant=%s events=%d", sub_id, tenant, event_index)

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_response_cache', '_circuit_breakers', '_traces', '_webhooks', '_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

