"""USSD Flow Designer service — visual menu builder, conditional routing, A/B tests."""

from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import asyncio
import hashlib
import logging
import re
from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

SUPPORTED_NODE_TYPES = {"menu", "input", "decision", "action", "end"}
SUPPORTED_FLOW_STATUSES = {"draft", "active", "archived"}
SUPPORTED_AB_STATUSES = {"active", "paused", "concluded"}
MAX_NODES_PER_FLOW = 200
MAX_EDGES_PER_FLOW = 400


class UssdFloService:
	"""In-memory USSD Flow Designer: build, version, translate, and A/B-test USSD flows."""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		self.tenant_id = tenant_id
		_store = get_store(db_url)
		self.flows: dict[str, dict[str, Any]] = {}
		self.nodes: dict[str, dict[str, Any]] = {}   # key = f"{flow_id}:{node_id}"
		self.edges: dict[str, dict[str, Any]] = {}   # key = edge record id
		self.translations: dict[str, dict[str, Any]] = {}  # key = f"{flow_id}:{language}"
		self.ab_tests: dict[str, dict[str, Any]] = {}
		self.flow_versions: dict[str, list[dict[str, Any]]] = {}  # flow_id -> version list
		self._audit_events = WriteThruList('audit_events', tenant_id, _store)

	# ── Utility ─────────────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _emit(self, tenant_id: str, event_type: str, resource_id: str, resource_type: str, details: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": self._record_id("audit"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"resource_id": resource_id,
			"resource_type": resource_type,
			"details": details or {},
			"emitted_at": self._now(),
		})

	def _flow_node_key(self, flow_id: str, node_id: str) -> str:
		return f"{flow_id}:{node_id}"

	def _eval_condition(self, condition: str, context: dict[str, Any]) -> bool:
		"""Evaluate a simple comparison expression against a context dict."""
		try:
			match = re.match(r"(\w+)\s*(==|!=|>|<|>=|<=)\s*(.+)", condition.strip())
			if not match:
				return True
			key, op, rhs = match.group(1), match.group(2), match.group(3).strip().strip("'\"")
			lhs = str(context.get(key, ""))
			ops = {"==": lhs == rhs, "!=": lhs != rhs, ">": lhs > rhs, "<": lhs < rhs, ">=": lhs >= rhs, "<=": lhs <= rhs}
			return ops.get(op, True)
		except Exception as exc:
			_log.debug("condition eval error '%s': %s", condition, exc)
			return True

	def _flow_checksum(self, flow_id: str) -> str:
		"""Compute a deterministic checksum of a flow's nodes and edges."""
		flow_nodes = sorted(
			[v for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")],
			key=lambda n: n["node_id"],
		)
		flow_edges = sorted(
			[v for v in self.edges.values() if v["flow_id"] == flow_id],
			key=lambda e: (e["source_node_id"], e["target_node_id"]),
		)
		payload = str(flow_nodes) + str(flow_edges)
		return hashlib.sha256(payload.encode()).hexdigest()[:16]

	# ── Health & describe ────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": "ussd_flo",
			"status": "healthy",
			"total_flows": len(self.flows),
			"active_flows": sum(1 for f in self.flows.values() if f["status"] == "active"),
			"total_nodes": len(self.nodes),
			"total_edges": len(self.edges),
			"ab_tests": len(self.ab_tests),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": "ussd_flo",
			"domain": "common",
			"version": "1.0.0",
			"description": "Visual USSD menu flow builder with conditional routing, multi-language support, A/B test flows",
			"supported_node_types": list(SUPPORTED_NODE_TYPES),
			"max_nodes_per_flow": MAX_NODES_PER_FLOW,
			"max_edges_per_flow": MAX_EDGES_PER_FLOW,
		}

	# ── Flow CRUD ────────────────────────────────────────────────────────────

	async def create_flow(
		self,
		name: str,
		service_code: str,
		root_node_id: str,
		tenant_id: str | None = None,
		description: str = "",
		languages: list[str] | None = None,
		tags: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		guard_non_empty_string(service_code, "service_code")
		guard_non_empty_string(root_node_id, "root_node_id")
		record = {
			"id": self._record_id("flow"),
			"tenant_id": tenant,
			"name": name,
			"description": description,
			"service_code": service_code,
			"root_node_id": root_node_id,
			"languages": languages or ["en"],
			"tags": tags or [],
			"status": "draft",
			"node_count": 0,
			"edge_count": 0,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.flows[record["id"]] = record
		self.flow_versions[record["id"]] = []
		self._emit(tenant, "flow_created", record["id"], "ussd_flow", {"name": name, "service_code": service_code})
		_log.info("flow created: %s service=%s tenant=%s", name, service_code, tenant)
		return deepcopy(record)

	async def get_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.flows.get(flow_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		return deepcopy(record)

	async def list_flows(
		self,
		tenant_id: str | None = None,
		service_code: str | None = None,
		status: str | None = None,
		tag: str | None = None,
	) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		results = [deepcopy(r) for r in self.flows.values() if r["tenant_id"] == tenant]
		if service_code:
			results = [r for r in results if r["service_code"] == service_code]
		if status:
			results = [r for r in results if r["status"] == status]
		if tag:
			results = [r for r in results if tag in r.get("tags", [])]
		return results

	async def update_flow(
		self,
		flow_id: str,
		tenant_id: str | None = None,
		name: str | None = None,
		description: str | None = None,
		root_node_id: str | None = None,
		languages: list[str] | None = None,
		tags: list[str] | None = None,
		status: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.flows.get(flow_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		if name is not None:
			record["name"] = name
		if description is not None:
			record["description"] = description
		if root_node_id is not None:
			record["root_node_id"] = root_node_id
		if languages is not None:
			record["languages"] = languages
		if tags is not None:
			record["tags"] = tags
		if status is not None:
			if status not in SUPPORTED_FLOW_STATUSES:
				raise ValueError(f"status must be one of {SUPPORTED_FLOW_STATUSES}")
			record["status"] = status
		if metadata is not None:
			record["metadata"].update(metadata)
		record["updated_at"] = self._now()
		self._emit(tenant, "flow_updated", flow_id, "ussd_flow")
		return deepcopy(record)

	async def delete_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.flows.get(flow_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		# Remove all nodes and edges belonging to this flow
		for key in [k for k in self.nodes if k.startswith(f"{flow_id}:")]:
			del self.nodes[key]
		for eid in [eid for eid, e in self.edges.items() if e["flow_id"] == flow_id]:
			del self.edges[eid]
		del self.flows[flow_id]
		self.flow_versions.pop(flow_id, None)
		self._emit(tenant, "flow_deleted", flow_id, "ussd_flow")
		return deepcopy(record)

	async def activate_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Validate and activate a draft flow."""
		tenant = self._tenant(tenant_id)
		record = self.flows.get(flow_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		# Basic validation: root node must exist
		root_key = self._flow_node_key(flow_id, record["root_node_id"])
		if root_key not in self.nodes:
			raise ValueError("root_node_not_found — cannot activate flow without a root node")
		if record["node_count"] == 0:
			raise ValueError("flow_has_no_nodes — add nodes before activating")
		record["status"] = "active"
		record["updated_at"] = self._now()
		self._emit(tenant, "flow_activated", flow_id, "ussd_flow")
		_log.info("flow activated: %s tenant=%s", flow_id, tenant)
		return deepcopy(record)

	async def archive_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.flows.get(flow_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		record["status"] = "archived"
		record["updated_at"] = self._now()
		self._emit(tenant, "flow_archived", flow_id, "ussd_flow")
		return deepcopy(record)

	# ── Node CRUD ────────────────────────────────────────────────────────────

	async def add_node(
		self,
		flow_id: str,
		node_id: str,
		node_type: str,
		title: str,
		tenant_id: str | None = None,
		body: str = "",
		items: list[dict[str, Any]] | None = None,
		position_x: float = 0.0,
		position_y: float = 0.0,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		if node_type not in SUPPORTED_NODE_TYPES:
			raise ValueError(f"node_type must be one of {SUPPORTED_NODE_TYPES}")
		if flow["node_count"] >= MAX_NODES_PER_FLOW:
			raise ValueError(f"flow exceeds max node limit ({MAX_NODES_PER_FLOW})")
		guard_non_empty_string(node_id, "node_id")
		key = self._flow_node_key(flow_id, node_id)
		if key in self.nodes:
			raise ValueError(f"node_already_exists: {node_id} in flow {flow_id}")
		record = {
			"id": self._record_id("node"),
			"flow_id": flow_id,
			"node_id": node_id,
			"node_type": node_type,
			"title": title,
			"body": body,
			"items": deepcopy(items or []),
			"position_x": position_x,
			"position_y": position_y,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.nodes[key] = record
		flow["node_count"] += 1
		flow["updated_at"] = self._now()
		self._emit(tenant, "node_added", record["id"], "ussd_node", {"flow_id": flow_id, "node_id": node_id, "node_type": node_type})
		return deepcopy(record)

	async def get_node(self, flow_id: str, node_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		key = self._flow_node_key(flow_id, node_id)
		record = self.nodes.get(key)
		if not record:
			raise KeyError(f"node_not_found: {node_id}")
		return deepcopy(record)

	async def list_nodes(self, flow_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		return [deepcopy(v) for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")]

	async def update_node(
		self,
		flow_id: str,
		node_id: str,
		tenant_id: str | None = None,
		title: str | None = None,
		body: str | None = None,
		items: list[dict[str, Any]] | None = None,
		position_x: float | None = None,
		position_y: float | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		key = self._flow_node_key(flow_id, node_id)
		record = self.nodes.get(key)
		if not record:
			raise KeyError(f"node_not_found: {node_id}")
		if title is not None:
			record["title"] = title
		if body is not None:
			record["body"] = body
		if items is not None:
			record["items"] = deepcopy(items)
		if position_x is not None:
			record["position_x"] = position_x
		if position_y is not None:
			record["position_y"] = position_y
		if metadata is not None:
			record["metadata"].update(metadata)
		record["updated_at"] = self._now()
		flow["updated_at"] = self._now()
		self._emit(tenant, "node_updated", record["id"], "ussd_node", {"flow_id": flow_id, "node_id": node_id})
		return deepcopy(record)

	async def delete_node(self, flow_id: str, node_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		key = self._flow_node_key(flow_id, node_id)
		record = self.nodes.get(key)
		if not record:
			raise KeyError(f"node_not_found: {node_id}")
		del self.nodes[key]
		flow["node_count"] = max(0, flow["node_count"] - 1)
		flow["updated_at"] = self._now()
		# Remove edges connected to this node
		to_delete = [
			eid for eid, e in self.edges.items()
			if e["flow_id"] == flow_id and (e["source_node_id"] == node_id or e["target_node_id"] == node_id)
		]
		for eid in to_delete:
			del self.edges[eid]
			flow["edge_count"] = max(0, flow["edge_count"] - 1)
		self._emit(tenant, "node_deleted", record["id"], "ussd_node", {"flow_id": flow_id, "node_id": node_id})
		return deepcopy(record)

	# ── Edge CRUD ────────────────────────────────────────────────────────────

	async def add_edge(
		self,
		flow_id: str,
		source_node_id: str,
		target_node_id: str,
		tenant_id: str | None = None,
		label: str = "",
		condition: str | None = None,
		priority: int = 0,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		if flow["edge_count"] >= MAX_EDGES_PER_FLOW:
			raise ValueError(f"flow exceeds max edge limit ({MAX_EDGES_PER_FLOW})")
		# Both nodes must exist
		if self._flow_node_key(flow_id, source_node_id) not in self.nodes:
			raise KeyError(f"source_node_not_found: {source_node_id}")
		if self._flow_node_key(flow_id, target_node_id) not in self.nodes:
			raise KeyError(f"target_node_not_found: {target_node_id}")
		record = {
			"id": self._record_id("edge"),
			"flow_id": flow_id,
			"source_node_id": source_node_id,
			"target_node_id": target_node_id,
			"label": label,
			"condition": condition,
			"priority": priority,
			"metadata": metadata or {},
			"created_at": self._now(),
		}
		self.edges[record["id"]] = record
		flow["edge_count"] += 1
		flow["updated_at"] = self._now()
		self._emit(tenant, "edge_added", record["id"], "ussd_edge", {"flow_id": flow_id, "source": source_node_id, "target": target_node_id})
		return deepcopy(record)

	async def list_edges(self, flow_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		return [deepcopy(e) for e in self.edges.values() if e["flow_id"] == flow_id]

	async def delete_edge(self, edge_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.edges.get(edge_id)
		if not record:
			raise KeyError(f"edge_not_found: {edge_id}")
		flow = self.flows.get(record["flow_id"])
		if not flow or flow["tenant_id"] != tenant:
			raise PermissionError("edge_tenant_mismatch")
		del self.edges[edge_id]
		flow["edge_count"] = max(0, flow["edge_count"] - 1)
		flow["updated_at"] = self._now()
		self._emit(tenant, "edge_deleted", edge_id, "ussd_edge")
		return deepcopy(record)

	# ── Conditional routing ───────────────────────────────────────────────────

	async def resolve_next_node(
		self,
		flow_id: str,
		current_node_id: str,
		context: dict[str, Any],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Evaluate outgoing edges from current_node and return the first matching target."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		outgoing = [
			e for e in self.edges.values()
			if e["flow_id"] == flow_id and e["source_node_id"] == current_node_id
		]
		outgoing.sort(key=lambda e: e["priority"])
		for edge in outgoing:
			condition = edge.get("condition")
			if not condition or self._eval_condition(condition, context):
				target_key = self._flow_node_key(flow_id, edge["target_node_id"])
				target_node = self.nodes.get(target_key)
				return {
					"matched_edge": deepcopy(edge),
					"target_node": deepcopy(target_node) if target_node else None,
					"resolved_at": self._now(),
				}
		return {"matched_edge": None, "target_node": None, "resolved_at": self._now()}

	async def get_reachable_nodes(self, flow_id: str, start_node_id: str, tenant_id: str | None = None) -> list[str]:
		"""BFS traversal — returns all node IDs reachable from start_node_id."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		visited: set[str] = set()
		queue = [start_node_id]
		while queue:
			node_id = queue.pop(0)
			if node_id in visited:
				continue
			visited.add(node_id)
			outgoing = [
				e["target_node_id"] for e in self.edges.values()
				if e["flow_id"] == flow_id and e["source_node_id"] == node_id
			]
			queue.extend(n for n in outgoing if n not in visited)
		return list(visited)

	async def detect_cycles(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Detect cycles in the flow graph using DFS."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		# Build adjacency list
		adj: dict[str, list[str]] = {}
		for e in self.edges.values():
			if e["flow_id"] != flow_id:
				continue
			adj.setdefault(e["source_node_id"], []).append(e["target_node_id"])
		visited: set[str] = set()
		rec_stack: set[str] = set()
		cycles: list[list[str]] = []

		def dfs(node: str, path: list[str]) -> None:
			visited.add(node)
			rec_stack.add(node)
			for neighbor in adj.get(node, []):
				if neighbor not in visited:
					dfs(neighbor, path + [neighbor])
				elif neighbor in rec_stack:
					cycle_start = path.index(neighbor) if neighbor in path else 0
					cycles.append(path[cycle_start:] + [neighbor])
			rec_stack.discard(node)

		for node_key in self.nodes:
			if not node_key.startswith(f"{flow_id}:"):
				continue
			node_id = node_key.split(":", 1)[1]
			if node_id not in visited:
				dfs(node_id, [node_id])
		return {"flow_id": flow_id, "has_cycles": len(cycles) > 0, "cycles": cycles, "checked_at": self._now()}

	async def validate_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Run structural validation on a flow before activation."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		errors: list[str] = []
		warnings: list[str] = []
		# Root node must exist
		root_key = self._flow_node_key(flow_id, flow["root_node_id"])
		if root_key not in self.nodes:
			errors.append(f"root_node_missing: {flow['root_node_id']}")
		# At least one end node
		node_types = [v["node_type"] for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")]
		if "end" not in node_types:
			warnings.append("no_end_node — sessions may never terminate cleanly")
		# Check for cycles
		cycle_check = await self.detect_cycles(flow_id, tenant_id)
		if cycle_check["has_cycles"]:
			warnings.append(f"cycles_detected: {len(cycle_check['cycles'])} cycle(s)")
		# Orphaned nodes (no incoming edges and not root)
		all_targets = {e["target_node_id"] for e in self.edges.values() if e["flow_id"] == flow_id}
		for nkey, node in self.nodes.items():
			if not nkey.startswith(f"{flow_id}:"):
				continue
			if node["node_id"] != flow["root_node_id"] and node["node_id"] not in all_targets:
				warnings.append(f"orphaned_node: {node['node_id']}")
		return {
			"flow_id": flow_id,
			"valid": len(errors) == 0,
			"errors": errors,
			"warnings": warnings,
			"node_count": flow["node_count"],
			"edge_count": flow["edge_count"],
			"validated_at": self._now(),
		}

	# ── Translation management ────────────────────────────────────────────────

	async def add_translation(
		self,
		flow_id: str,
		language: str,
		translations: dict[str, dict[str, str]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Add or replace translations for all nodes in a flow for a given language."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		guard_non_empty_string(language, "language")
		key = f"{flow_id}:{language}"
		record = {
			"id": self._record_id("trans"),
			"flow_id": flow_id,
			"tenant_id": tenant,
			"language": language,
			"translations": deepcopy(translations),
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.translations[key] = record
		# Register language in flow
		if language not in flow["languages"]:
			flow["languages"].append(language)
		flow["updated_at"] = self._now()
		self._emit(tenant, "translation_added", record["id"], "ussd_translation", {"flow_id": flow_id, "language": language})
		_log.info("translation added: flow=%s lang=%s tenant=%s", flow_id, language, tenant)
		return deepcopy(record)

	async def get_translation(self, flow_id: str, language: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		key = f"{flow_id}:{language}"
		record = self.translations.get(key)
		if not record:
			raise KeyError(f"translation_not_found: {flow_id}:{language}")
		return deepcopy(record)

	async def list_translations(self, flow_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		return [deepcopy(v) for k, v in self.translations.items() if k.startswith(f"{flow_id}:")]

	async def delete_translation(self, flow_id: str, language: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		key = f"{flow_id}:{language}"
		record = self.translations.get(key)
		if not record:
			raise KeyError(f"translation_not_found: {flow_id}:{language}")
		del self.translations[key]
		if language in flow["languages"] and language != "en":
			flow["languages"].remove(language)
		self._emit(tenant, "translation_deleted", record["id"], "ussd_translation")
		return deepcopy(record)

	async def render_node_translated(
		self,
		flow_id: str,
		node_id: str,
		language: str,
		tenant_id: str | None = None,
		variables: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Render a node with translations applied and variable substitution."""
		tenant = self._tenant(tenant_id)
		node = await self.get_node(flow_id, node_id, tenant_id)
		trans_key = f"{flow_id}:{language}"
		trans = self.translations.get(trans_key, {})
		node_trans = trans.get("translations", {}).get(node_id, {})
		title = node_trans.get("title", node["title"])
		body = node_trans.get("body", node["body"])
		vars_ = variables or {}
		for k, v in vars_.items():
			title = title.replace(f"{{{k}}}", str(v))
			body = body.replace(f"{{{k}}}", str(v))
		items = deepcopy(node["items"])
		item_labels = node_trans.get("item_labels", {})
		for i, item in enumerate(items):
			label_key = str(i)
			if label_key in item_labels:
				item["label"] = item_labels[label_key]
		return {"node_id": node_id, "language": language, "title": title, "body": body, "items": items, "rendered_at": self._now()}

	# ── A/B test management ───────────────────────────────────────────────────

	async def create_ab_test(
		self,
		name: str,
		service_code: str,
		control_flow_id: str,
		variant_flow_id: str,
		tenant_id: str | None = None,
		split_percentage: float = 50.0,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(name, "name")
		if not 0 < split_percentage < 100:
			raise ValueError("split_percentage must be between 0 and 100 (exclusive)")
		# Both flows must exist and belong to tenant
		for fid in (control_flow_id, variant_flow_id):
			f = self.flows.get(fid)
			if not f or f["tenant_id"] != tenant:
				raise KeyError(f"flow_not_found: {fid}")
		record = {
			"id": self._record_id("ab"),
			"tenant_id": tenant,
			"name": name,
			"service_code": service_code,
			"control_flow_id": control_flow_id,
			"variant_flow_id": variant_flow_id,
			"split_percentage": split_percentage,
			"status": "active",
			"control_sessions": 0,
			"variant_sessions": 0,
			"control_completions": 0,
			"variant_completions": 0,
			"metadata": metadata or {},
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.ab_tests[record["id"]] = record
		self._emit(tenant, "ab_test_created", record["id"], "ussd_ab_test", {
			"name": name, "control": control_flow_id, "variant": variant_flow_id,
		})
		_log.info("ab_test created: %s tenant=%s", name, tenant)
		return deepcopy(record)

	async def get_ab_test(self, test_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		return deepcopy(record)

	async def list_ab_tests(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		results = [deepcopy(r) for r in self.ab_tests.values() if r["tenant_id"] == tenant]
		if status:
			results = [r for r in results if r["status"] == status]
		return results

	async def update_ab_test(
		self,
		test_id: str,
		tenant_id: str | None = None,
		split_percentage: float | None = None,
		status: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		if split_percentage is not None:
			if not 0 < split_percentage < 100:
				raise ValueError("split_percentage must be between 0 and 100 (exclusive)")
			record["split_percentage"] = split_percentage
		if status is not None:
			if status not in SUPPORTED_AB_STATUSES:
				raise ValueError(f"status must be one of {SUPPORTED_AB_STATUSES}")
			record["status"] = status
		if metadata is not None:
			record["metadata"].update(metadata)
		record["updated_at"] = self._now()
		self._emit(tenant, "ab_test_updated", test_id, "ussd_ab_test")
		return deepcopy(record)

	async def delete_ab_test(self, test_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		del self.ab_tests[test_id]
		self._emit(tenant, "ab_test_deleted", test_id, "ussd_ab_test")
		return deepcopy(record)

	async def assign_ab_flow(
		self,
		test_id: str,
		session_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Deterministically assign a session to control or variant based on session_id hash."""
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		if record["status"] != "active":
			raise ValueError(f"ab_test not active: {record['status']}")
		digest = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
		bucket = digest % 100
		is_variant = bucket < record["split_percentage"]
		assigned_flow = record["variant_flow_id"] if is_variant else record["control_flow_id"]
		arm = "variant" if is_variant else "control"
		record[f"{arm}_sessions"] += 1
		record["updated_at"] = self._now()
		return {
			"test_id": test_id,
			"session_id": session_id,
			"arm": arm,
			"assigned_flow_id": assigned_flow,
			"assigned_at": self._now(),
		}

	async def record_ab_completion(self, test_id: str, arm: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Record a session completion for the given arm (control/variant)."""
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		if arm not in ("control", "variant"):
			raise ValueError("arm must be 'control' or 'variant'")
		record[f"{arm}_completions"] += 1
		record["updated_at"] = self._now()
		return deepcopy(record)

	async def get_ab_test_results(self, test_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Compute conversion rates for control vs variant."""
		tenant = self._tenant(tenant_id)
		record = self.ab_tests.get(test_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"ab_test_not_found: {test_id}")
		ctrl_rate = (record["control_completions"] / record["control_sessions"] * 100) if record["control_sessions"] else 0.0
		var_rate = (record["variant_completions"] / record["variant_sessions"] * 100) if record["variant_sessions"] else 0.0
		lift = var_rate - ctrl_rate
		return {
			"test_id": test_id,
			"name": record["name"],
			"status": record["status"],
			"control": {
				"flow_id": record["control_flow_id"],
				"sessions": record["control_sessions"],
				"completions": record["control_completions"],
				"completion_rate_pct": round(ctrl_rate, 2),
			},
			"variant": {
				"flow_id": record["variant_flow_id"],
				"sessions": record["variant_sessions"],
				"completions": record["variant_completions"],
				"completion_rate_pct": round(var_rate, 2),
			},
			"lift_pct": round(lift, 2),
			"generated_at": self._now(),
		}

	# ── Flow versioning ───────────────────────────────────────────────────────

	async def snapshot_flow(self, flow_id: str, label: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Save a point-in-time snapshot of a flow's nodes and edges."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		nodes_snapshot = [deepcopy(v) for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")]
		edges_snapshot = [deepcopy(e) for e in self.edges.values() if e["flow_id"] == flow_id]
		checksum = self._flow_checksum(flow_id)
		version = {
			"version_id": self._record_id("ver"),
			"flow_id": flow_id,
			"label": label,
			"checksum": checksum,
			"node_count": len(nodes_snapshot),
			"edge_count": len(edges_snapshot),
			"nodes": nodes_snapshot,
			"edges": edges_snapshot,
			"snapshotted_at": self._now(),
		}
		self.flow_versions.setdefault(flow_id, []).append(version)
		self._emit(tenant, "flow_snapshot_created", flow_id, "ussd_flow", {"label": label, "checksum": checksum})
		return {k: v for k, v in version.items() if k not in ("nodes", "edges")}

	async def list_flow_versions(self, flow_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		versions = self.flow_versions.get(flow_id, [])
		return [{k: v for k, v in ver.items() if k not in ("nodes", "edges")} for ver in versions]

	async def restore_flow_version(self, flow_id: str, version_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Restore a flow to a previous snapshot."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		versions = self.flow_versions.get(flow_id, [])
		version = next((v for v in versions if v["version_id"] == version_id), None)
		if not version:
			raise KeyError(f"version_not_found: {version_id}")
		# Clear current nodes and edges
		for key in [k for k in self.nodes if k.startswith(f"{flow_id}:")]:
			del self.nodes[key]
		for eid in [eid for eid, e in self.edges.items() if e["flow_id"] == flow_id]:
			del self.edges[eid]
		# Restore from snapshot
		for node in version["nodes"]:
			self.nodes[self._flow_node_key(flow_id, node["node_id"])] = deepcopy(node)
		for edge in version["edges"]:
			self.edges[edge["id"]] = deepcopy(edge)
		flow["node_count"] = version["node_count"]
		flow["edge_count"] = version["edge_count"]
		flow["updated_at"] = self._now()
		self._emit(tenant, "flow_version_restored", flow_id, "ussd_flow", {"version_id": version_id})
		return {"flow_id": flow_id, "version_id": version_id, "restored_at": self._now()}

	# ── Export / import ───────────────────────────────────────────────────────

	async def export_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Export a complete flow (nodes, edges, translations) as a portable dict."""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		nodes = [deepcopy(v) for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")]
		edges = [deepcopy(e) for e in self.edges.values() if e["flow_id"] == flow_id]
		translations = [deepcopy(v) for k, v in self.translations.items() if k.startswith(f"{flow_id}:")]
		return {
			"schema_version": "1.0",
			"flow": deepcopy(flow),
			"nodes": nodes,
			"edges": edges,
			"translations": translations,
			"checksum": self._flow_checksum(flow_id),
			"exported_at": self._now(),
		}

	async def import_flow(
		self,
		flow_export: dict[str, Any],
		tenant_id: str | None = None,
		overwrite: bool = False,
	) -> dict[str, Any]:
		"""Import a flow from an export dict. Assigns new IDs to avoid collisions."""
		tenant = self._tenant(tenant_id)
		source_flow = flow_export.get("flow", {})
		new_flow_id = self._record_id("flow")
		# Create the flow record
		new_flow = deepcopy(source_flow)
		new_flow["id"] = new_flow_id
		new_flow["tenant_id"] = tenant
		new_flow["status"] = "draft"
		new_flow["created_at"] = self._now()
		new_flow["updated_at"] = self._now()
		self.flows[new_flow_id] = new_flow
		self.flow_versions[new_flow_id] = []
		# Import nodes
		node_id_map: dict[str, str] = {}
		for node in flow_export.get("nodes", []):
			new_node = deepcopy(node)
			old_key = self._flow_node_key(source_flow.get("id", ""), node["node_id"])
			new_key = self._flow_node_key(new_flow_id, node["node_id"])
			new_node["flow_id"] = new_flow_id
			new_node["id"] = self._record_id("node")
			self.nodes[new_key] = new_node
			node_id_map[node["node_id"]] = node["node_id"]
		# Import edges
		for edge in flow_export.get("edges", []):
			new_edge = deepcopy(edge)
			new_edge["id"] = self._record_id("edge")
			new_edge["flow_id"] = new_flow_id
			self.edges[new_edge["id"]] = new_edge
		# Import translations
		for trans in flow_export.get("translations", []):
			new_trans = deepcopy(trans)
			new_trans["id"] = self._record_id("trans")
			new_trans["flow_id"] = new_flow_id
			new_trans["tenant_id"] = tenant
			key = f"{new_flow_id}:{trans['language']}"
			self.translations[key] = new_trans
		self._emit(tenant, "flow_imported", new_flow_id, "ussd_flow", {"source_name": source_flow.get("name")})
		return deepcopy(new_flow)

	# ── Bulk operations ───────────────────────────────────────────────────────

	async def bulk_add_nodes(self, flow_id: str, nodes: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
		"""Add multiple nodes to a flow in one call."""
		tenant = self._tenant(tenant_id)
		results, errors = [], []
		tasks = [
			self.add_node(
				flow_id=flow_id, node_id=n["node_id"], node_type=n["node_type"],
				title=n["title"], tenant_id=tenant, body=n.get("body", ""),
				items=n.get("items"), position_x=n.get("position_x", 0.0),
				position_y=n.get("position_y", 0.0), metadata=n.get("metadata"),
			)
			for n in nodes
		]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		for n, r in zip(nodes, raw):
			if isinstance(r, Exception):
				errors.append({"input": n, "error": str(r)})
			else:
				results.append(r)
		return {"added": len(results), "failed": len(errors), "nodes": results, "errors": errors}

	# ── Analytics ─────────────────────────────────────────────────────────────

	async def get_flow_stats(self, tenant_id: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		flows = [f for f in self.flows.values() if f["tenant_id"] == tenant]
		by_status: dict[str, int] = {}
		for f in flows:
			by_status[f["status"]] = by_status.get(f["status"], 0) + 1
		return {
			"tenant_id": tenant,
			"total_flows": len(flows),
			"by_status": by_status,
			"total_nodes": sum(f["node_count"] for f in flows),
			"total_edges": sum(f["edge_count"] for f in flows),
			"total_translations": len([k for k in self.translations if any(k.startswith(f"{f['id']}:") for f in flows)]),
			"active_ab_tests": len([t for t in self.ab_tests.values() if t["tenant_id"] == tenant and t["status"] == "active"]),
			"generated_at": self._now(),
		}

	async def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		return await self.get_flow_stats(tenant_id)

	# ── Audit events ──────────────────────────────────────────────────────────

	async def get_audit_events(self, tenant_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		return events[-limit:]

	# ── Session simulation ────────────────────────────────────────────────────

	async def simulate_session(
		self,
		flow_id: str,
		script: list[str],
		tenant_id: str | None = None,
		language: str = "en",
		context_seed: dict[str, Any] | None = None,
		expected_terminal_node: str | None = None,
	) -> dict[str, Any]:
		"""
		Dry-run a scripted USSD conversation against a flow without deploying.

		Walk the flow graph node-by-node, feeding items from *script* as
		successive user inputs.  Each step renders the current node (with
		translation), evaluates outgoing edges with the accumulated context,
		and advances to the matched target.

		Returns a full SessionTrace: every screen rendered, every edge taken,
		every context variable accumulated, plus a pass/fail verdict against
		*expected_terminal_node* if provided.

		Args:
			flow_id: The flow to simulate.
			script: Ordered list of user inputs, one per interaction step.
			language: BCP-47 language code used for rendering.
			context_seed: Initial context variables (e.g. subscriber info).
			expected_terminal_node: If given, the trace fails when the session
				terminates at a different node.

		Returns:
			Trace dict with keys: flow_id, steps, terminal_node_id, passed,
			failure_reason, simulated_at.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")

		context: dict[str, Any] = deepcopy(context_seed or {})
		steps: list[dict[str, Any]] = []
		current_node_id = flow["root_node_id"]
		script_idx = 0
		max_steps = min(len(script) + 1, MAX_NODES_PER_FLOW)
		terminal_node_id: str | None = None
		failure_reason: str | None = None

		for _ in range(max_steps):
			node_key = self._flow_node_key(flow_id, current_node_id)
			node = self.nodes.get(node_key)
			if not node:
				failure_reason = f"node_not_found_during_sim: {current_node_id}"
				break

			# Render with translation
			rendered = await self.render_node_translated(
				flow_id, current_node_id, language, tenant_id, variables=context
			)

			# Capture user input if any remains in script
			user_input: str | None = None
			if script_idx < len(script):
				user_input = script[script_idx]
				script_idx += 1
				# Store input under conventional key for condition evaluation
				context["user_input"] = user_input
				context[f"input_{current_node_id}"] = user_input
				# If it's an input node, store under node_id key too
				if node["node_type"] == "input":
					context[current_node_id] = user_input

			steps.append({
				"step": len(steps) + 1,
				"node_id": current_node_id,
				"node_type": node["node_type"],
				"rendered_title": rendered["title"],
				"rendered_body": rendered["body"],
				"user_input": user_input,
				"context_snapshot": deepcopy(context),
			})

			if node["node_type"] == "end":
				terminal_node_id = current_node_id
				break

			# Resolve next node
			resolution = await self.resolve_next_node(
				flow_id, current_node_id, context, tenant_id
			)
			if not resolution["target_node"]:
				failure_reason = f"no_edge_matched_at: {current_node_id}"
				terminal_node_id = current_node_id
				break

			current_node_id = resolution["target_node"]["node_id"]

		# Assess verdict
		passed = True
		if failure_reason:
			passed = False
		elif expected_terminal_node and terminal_node_id != expected_terminal_node:
			passed = False
			failure_reason = (
				f"expected terminal={expected_terminal_node}, "
				f"got={terminal_node_id}"
			)

		_log.info(
			"simulate_session flow=%s steps=%d passed=%s tenant=%s",
			flow_id, len(steps), passed, tenant,
		)
		return {
			"flow_id": flow_id,
			"language": language,
			"steps": steps,
			"terminal_node_id": terminal_node_id,
			"expected_terminal_node": expected_terminal_node,
			"passed": passed,
			"failure_reason": failure_reason,
			"simulated_at": self._now(),
		}

	# ── Flow scoring ──────────────────────────────────────────────────────────

	async def score_flow(self, flow_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""
		Compute a usability scorecard for a flow.

		Metrics:
		- avg_path_depth / max_path_depth: mean and max hops from root to any
		  end node via BFS on the unweighted graph.
		- cyclomatic_complexity: E - N + 2 (McCabe).
		- avg_branching_factor: average outgoing edge count across menu nodes.
		- estimated_session_seconds: avg_path_depth * 10 (10 s/node heuristic).
		- usability_score: 0-100 composite with penalties for depth > 5,
		  cyclomatic complexity > 20, branching factor > 8.

		Returns:
			FlowScorecard dict.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")

		# Build adjacency list
		adj: dict[str, list[str]] = {}
		for e in self.edges.values():
			if e["flow_id"] != flow_id:
				continue
			adj.setdefault(e["source_node_id"], []).append(e["target_node_id"])

		flow_nodes = {
			k.split(":", 1)[1]: v
			for k, v in self.nodes.items()
			if k.startswith(f"{flow_id}:")
		}
		end_nodes = {nid for nid, n in flow_nodes.items() if n["node_type"] == "end"}

		# BFS from root to all end nodes — collect path depths
		root = flow["root_node_id"]
		from collections import deque
		depth_map: dict[str, int] = {root: 0}
		q: deque[str] = deque([root])
		while q:
			nid = q.popleft()
			for nbr in adj.get(nid, []):
				if nbr not in depth_map:
					depth_map[nbr] = depth_map[nid] + 1
					q.append(nbr)

		end_depths = [depth_map[n] for n in end_nodes if n in depth_map]
		avg_path_depth = sum(end_depths) / len(end_depths) if end_depths else 0.0
		max_path_depth = max(end_depths) if end_depths else 0

		# Cyclomatic complexity: E - N + 2
		n_edges = sum(1 for e in self.edges.values() if e["flow_id"] == flow_id)
		n_nodes = len(flow_nodes)
		cyclomatic = n_edges - n_nodes + 2

		# Branching factor from menu nodes
		menu_out = [
			len(adj.get(nid, []))
			for nid, n in flow_nodes.items()
			if n["node_type"] == "menu"
		]
		avg_branching = sum(menu_out) / len(menu_out) if menu_out else 0.0

		estimated_session_seconds = avg_path_depth * 10

		# Usability score
		score = 100.0
		if avg_path_depth > 5:
			score -= min(30, (avg_path_depth - 5) * 6)
		if cyclomatic > 20:
			score -= min(20, (cyclomatic - 20) * 1)
		if avg_branching > 8:
			score -= min(15, (avg_branching - 8) * 3)
		if n_nodes == 0:
			score = 0.0
		score = max(0.0, round(score, 1))

		return {
			"flow_id": flow_id,
			"node_count": n_nodes,
			"edge_count": n_edges,
			"avg_path_depth": round(avg_path_depth, 2),
			"max_path_depth": max_path_depth,
			"cyclomatic_complexity": cyclomatic,
			"avg_branching_factor": round(avg_branching, 2),
			"estimated_session_seconds": round(estimated_session_seconds, 1),
			"usability_score": score,
			"scored_at": self._now(),
		}

	# ── Translation completeness ──────────────────────────────────────────────

	async def check_translation_completeness(
		self,
		flow_id: str,
		language: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Report missing translation keys for a given language.

		For each node in the flow, check that title, body, and all item labels
		are present in the translation dict.  Returns a per-node coverage
		breakdown and an overall coverage percentage.

		Args:
			flow_id: Target flow.
			language: BCP-47 code to check (e.g. 'sw', 'fr').

		Returns:
			TranslationCoverageReport with keys: flow_id, language,
			coverage_pct, missing, node_reports.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		guard_non_empty_string(language, "language")

		trans_key = f"{flow_id}:{language}"
		trans_record = self.translations.get(trans_key, {})
		trans_map: dict[str, dict[str, str]] = trans_record.get("translations", {})

		flow_nodes = [
			v for k, v in self.nodes.items() if k.startswith(f"{flow_id}:")
		]

		total_keys = 0
		missing_keys = 0
		missing: list[dict[str, Any]] = []
		node_reports: list[dict[str, Any]] = []

		for node in flow_nodes:
			nid = node["node_id"]
			node_trans = trans_map.get(nid, {})
			node_missing: list[str] = []

			# title
			total_keys += 1
			if not node_trans.get("title"):
				missing_keys += 1
				node_missing.append("title")

			# body (only if non-empty in source)
			if node.get("body"):
				total_keys += 1
				if not node_trans.get("body"):
					missing_keys += 1
					node_missing.append("body")

			# item labels
			for idx in range(len(node.get("items", []))):
				total_keys += 1
				label_key = str(idx)
				item_labels = node_trans.get("item_labels", {})
				if not item_labels.get(label_key):
					missing_keys += 1
					node_missing.append(f"item_labels[{label_key}]")

			if node_missing:
				missing.append({"node_id": nid, "missing_keys": node_missing})

			node_reports.append({
				"node_id": nid,
				"node_type": node["node_type"],
				"missing": node_missing,
				"complete": len(node_missing) == 0,
			})

		coverage_pct = (
			round((total_keys - missing_keys) / total_keys * 100, 1)
			if total_keys else 100.0
		)

		return {
			"flow_id": flow_id,
			"language": language,
			"total_keys": total_keys,
			"missing_keys": missing_keys,
			"coverage_pct": coverage_pct,
			"missing": missing,
			"node_reports": node_reports,
			"checked_at": self._now(),
		}

	# ── Screen budget validation ───────────────────────────────────────────────

	async def validate_screen_budgets(
		self,
		flow_id: str,
		tenant_id: str | None = None,
		budget: int = 182,
	) -> dict[str, Any]:
		"""
		Check that every node's rendered content fits within the USSD character budget.

		Most networks (MTN, Safaricom) truncate USSD pages at 182 bytes.
		This method renders each node in each registered language and flags
		content that would be silently cut by the network layer.

		Args:
			flow_id: Target flow.
			budget: Maximum bytes per screen (default 182, UTF-8 encoded).

		Returns:
			ScreenBudgetReport with over-budget nodes per language.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")
		if budget < 1:
			raise ValueError("budget must be >= 1")

		languages = flow.get("languages", ["en"])
		violations: list[dict[str, Any]] = []
		checks_total = 0

		for language in languages:
			for k, node in self.nodes.items():
				if not k.startswith(f"{flow_id}:"):
					continue
				nid = node["node_id"]
				rendered = await self.render_node_translated(
					flow_id, nid, language, tenant_id
				)
				# Build the full USSD page string (title + newline + body + items)
				page_parts = [rendered["title"]]
				if rendered["body"]:
					page_parts.append(rendered["body"])
				for idx, item in enumerate(rendered.get("items", []), 1):
					page_parts.append(f"{idx}. {item.get('label', '')}")
				page_text = "\n".join(page_parts)
				byte_len = len(page_text.encode("utf-8"))
				checks_total += 1
				if byte_len > budget:
					violations.append({
						"node_id": nid,
						"language": language,
						"byte_length": byte_len,
						"budget": budget,
						"overflow": byte_len - budget,
						"preview": page_text[:80] + ("..." if len(page_text) > 80 else ""),
					})

		return {
			"flow_id": flow_id,
			"budget": budget,
			"languages_checked": languages,
			"nodes_checked": checks_total,
			"violations": violations,
			"violation_count": len(violations),
			"passed": len(violations) == 0,
			"validated_at": self._now(),
		}

	# ── Flow diff ─────────────────────────────────────────────────────────────

	async def diff_flow_versions(
		self,
		flow_id: str,
		version_id_a: str,
		version_id_b: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Compute a structural diff between two flow snapshots.

		Compares node sets and edge sets by their logical IDs (node_id,
		source+target pair).  Does NOT require version_id_b to be newer than
		version_id_a — the diff is directional: a → b.

		Returns:
			FlowDiff with added_nodes, removed_nodes, modified_nodes,
			added_edges, removed_edges.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")

		versions = self.flow_versions.get(flow_id, [])
		ver_a = next((v for v in versions if v["version_id"] == version_id_a), None)
		ver_b = next((v for v in versions if v["version_id"] == version_id_b), None)
		if not ver_a:
			raise KeyError(f"version_not_found: {version_id_a}")
		if not ver_b:
			raise KeyError(f"version_not_found: {version_id_b}")

		# Index nodes by node_id
		nodes_a = {n["node_id"]: n for n in ver_a["nodes"]}
		nodes_b = {n["node_id"]: n for n in ver_b["nodes"]}

		added_nodes = [deepcopy(nodes_b[nid]) for nid in nodes_b if nid not in nodes_a]
		removed_nodes = [deepcopy(nodes_a[nid]) for nid in nodes_a if nid not in nodes_b]
		modified_nodes: list[dict[str, Any]] = []
		for nid in nodes_a:
			if nid in nodes_b:
				fields_changed: list[str] = []
				for field in ("title", "body", "items", "node_type"):
					if nodes_a[nid].get(field) != nodes_b[nid].get(field):
						fields_changed.append(field)
				if fields_changed:
					modified_nodes.append({
						"node_id": nid,
						"fields_changed": fields_changed,
						"before": {f: nodes_a[nid].get(f) for f in fields_changed},
						"after": {f: nodes_b[nid].get(f) for f in fields_changed},
					})

		# Index edges by (source, target) pair
		def edge_key(e: dict[str, Any]) -> str:
			return f"{e['source_node_id']}→{e['target_node_id']}"

		edges_a = {edge_key(e): e for e in ver_a["edges"]}
		edges_b = {edge_key(e): e for e in ver_b["edges"]}

		added_edges = [deepcopy(edges_b[k]) for k in edges_b if k not in edges_a]
		removed_edges = [deepcopy(edges_a[k]) for k in edges_a if k not in edges_b]

		return {
			"flow_id": flow_id,
			"version_a": version_id_a,
			"version_b": version_id_b,
			"added_nodes": added_nodes,
			"removed_nodes": removed_nodes,
			"modified_nodes": modified_nodes,
			"added_edges": added_edges,
			"removed_edges": removed_edges,
			"summary": {
				"nodes_added": len(added_nodes),
				"nodes_removed": len(removed_nodes),
				"nodes_modified": len(modified_nodes),
				"edges_added": len(added_edges),
				"edges_removed": len(removed_edges),
			},
			"diffed_at": self._now(),
		}

	# ── Bulk flow migration ───────────────────────────────────────────────────

	async def migrate_service_code(
		self,
		old_code: str,
		new_code: str,
		tenant_id: str | None = None,
		dry_run: bool = True,
	) -> dict[str, Any]:
		"""
		Rename a USSD service code across all affected flows.

		Scans all tenant flows for ``old_code`` in their ``service_code`` field
		and node metadata.  In dry-run mode only reports; when ``dry_run=False``
		snapshots every affected flow first, then applies the rename atomically.

		Args:
			old_code: The service code to replace (e.g. '*123#').
			new_code: The replacement service code.
			dry_run: When True (default) only report — do not mutate.

		Returns:
			MigrationReport with migration_id (only populated on live run),
			affected_flows, dry_run flag, and per-flow details.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(old_code, "old_code")
		guard_non_empty_string(new_code, "new_code")
		if old_code == new_code:
			raise ValueError("old_code and new_code must differ")

		tenant_flows = [f for f in self.flows.values() if f["tenant_id"] == tenant]
		affected: list[dict[str, Any]] = []

		for flow in tenant_flows:
			flow_touched = False
			node_updates: list[str] = []

			if flow["service_code"] == old_code:
				flow_touched = True

			# Check node metadata for service_code references
			for nk, node in self.nodes.items():
				if not nk.startswith(f"{flow['id']}:"):
					continue
				if node.get("metadata", {}).get("service_code") == old_code:
					node_updates.append(node["node_id"])

			if flow_touched or node_updates:
				affected.append({
					"flow_id": flow["id"],
					"flow_name": flow["name"],
					"service_code_changed": flow_touched,
					"node_metadata_updated": node_updates,
				})

		migration_id: str | None = None

		if not dry_run and affected:
			migration_id = self._record_id("mig")
			for entry in affected:
				fid = entry["flow_id"]
				# Auto-snapshot before mutation
				await self.snapshot_flow(fid, f"pre-migration-{migration_id}", tenant_id)
				# Apply mutation
				if entry["service_code_changed"]:
					self.flows[fid]["service_code"] = new_code
					self.flows[fid]["updated_at"] = self._now()
				for nid in entry["node_metadata_updated"]:
					nk = self._flow_node_key(fid, nid)
					if nk in self.nodes:
						self.nodes[nk]["metadata"]["service_code"] = new_code
						self.nodes[nk]["updated_at"] = self._now()

			self._emit(
				tenant, "service_code_migrated", migration_id, "ussd_flow",
				{"old_code": old_code, "new_code": new_code, "flows_affected": len(affected)},
			)
			_log.info(
				"migrate_service_code old=%s new=%s flows=%d tenant=%s",
				old_code, new_code, len(affected), tenant,
			)

		return {
			"migration_id": migration_id,
			"old_code": old_code,
			"new_code": new_code,
			"dry_run": dry_run,
			"affected_flows": len(affected),
			"details": affected,
			"generated_at": self._now(),
		}

	# ── Dead-path analysis ────────────────────────────────────────────────────

	async def compute_dead_paths(
		self,
		flow_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Identify nodes that can never be reached or can never reach an end node.

		A node is *forward-dead* if it is not reachable from the root.
		A node is *backward-dead* if no end node is reachable from it.
		A node is *fully-dead* if both conditions apply.

		Returns:
			DeadPathReport with forward_dead, backward_dead, fully_dead node
			lists and suggested actions.
		"""
		tenant = self._tenant(tenant_id)
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")

		all_node_ids = {
			k.split(":", 1)[1]
			for k in self.nodes
			if k.startswith(f"{flow_id}:")
		}
		if not all_node_ids:
			return {
				"flow_id": flow_id,
				"forward_dead": [],
				"backward_dead": [],
				"fully_dead": [],
				"suggestions": [],
				"analysed_at": self._now(),
			}

		# Forward reachability from root via BFS
		forward_reachable = set(
			await self.get_reachable_nodes(flow_id, flow["root_node_id"], tenant_id)
		)
		forward_dead = all_node_ids - forward_reachable

		# Backward reachability: reverse graph BFS from end nodes
		rev_adj: dict[str, list[str]] = {}
		for e in self.edges.values():
			if e["flow_id"] != flow_id:
				continue
			rev_adj.setdefault(e["target_node_id"], []).append(e["source_node_id"])

		end_nodes = {
			k.split(":", 1)[1]
			for k, v in self.nodes.items()
			if k.startswith(f"{flow_id}:") and v["node_type"] == "end"
		}

		backward_reachable: set[str] = set()
		queue = list(end_nodes)
		while queue:
			nid = queue.pop(0)
			if nid in backward_reachable:
				continue
			backward_reachable.add(nid)
			queue.extend(n for n in rev_adj.get(nid, []) if n not in backward_reachable)

		backward_dead = all_node_ids - backward_reachable
		fully_dead = forward_dead & backward_dead

		suggestions: list[str] = []
		for nid in sorted(forward_dead):
			suggestions.append(f"add_incoming_edge_to: {nid}")
		for nid in sorted(backward_dead - forward_dead):
			suggestions.append(f"add_path_to_end_from: {nid}")
		for nid in sorted(fully_dead):
			suggestions.append(f"safe_to_delete: {nid}")

		return {
			"flow_id": flow_id,
			"forward_dead": sorted(forward_dead),
			"backward_dead": sorted(backward_dead),
			"fully_dead": sorted(fully_dead),
			"suggestions": suggestions,
			"analysed_at": self._now(),
		}

	# ── Session event recording ───────────────────────────────────────────────

	async def record_session_event(
		self,
		session_id: str,
		flow_id: str,
		node_id: str,
		user_input: str | None = None,
		context_snapshot: dict[str, Any] | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Record a live session interaction event for later replay and debugging.

		Events are stored in ``self.session_events`` keyed by *session_id*.
		Each event captures the node visited, user input received, and a full
		context snapshot so the exact session state can be reconstructed.

		Args:
			session_id: Unique identifier for the subscriber's session.
			flow_id: The flow being executed.
			node_id: The node displayed to the subscriber.
			user_input: Raw user response (may be None for first display).
			context_snapshot: Full session context at the moment of the event.

		Returns:
			The recorded session event dict.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(session_id, "session_id")
		guard_non_empty_string(flow_id, "flow_id")
		guard_non_empty_string(node_id, "node_id")

		if not hasattr(self, "session_events"):
			self.session_events: dict[str, list[dict[str, Any]]] = {}

		event: dict[str, Any] = {
			"id": self._record_id("se"),
			"session_id": session_id,
			"flow_id": flow_id,
			"node_id": node_id,
			"user_input": user_input,
			"context_snapshot": deepcopy(context_snapshot or {}),
			"tenant_id": tenant,
			"recorded_at": self._now(),
		}
		self.session_events.setdefault(session_id, []).append(event)
		return deepcopy(event)

	async def replay_session(
		self,
		session_id: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Replay a recorded live session through the current flow graph.

		Walks the stored session events and re-evaluates each routing decision
		against the *current* flow graph.  Divergences (where the current graph
		would have routed differently from what was observed) are flagged, making
		it trivial to identify which flow change caused a subscriber complaint.

		Args:
			session_id: The session to replay.

		Returns:
			SessionReplayReport with steps, divergences, and verdict.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(session_id, "session_id")

		if not hasattr(self, "session_events"):
			self.session_events = {}

		events = self.session_events.get(session_id, [])
		if not events:
			raise KeyError(f"session_not_found: {session_id}")

		flow_id = events[0]["flow_id"]
		flow = self.flows.get(flow_id)
		if not flow or flow["tenant_id"] != tenant:
			raise KeyError(f"flow_not_found: {flow_id}")

		replay_steps: list[dict[str, Any]] = []
		divergences: list[dict[str, Any]] = []
		context: dict[str, Any] = {}

		for i, event in enumerate(events):
			current_node_id = event["node_id"]
			user_input = event.get("user_input")

			if user_input is not None:
				context["user_input"] = user_input
				context[f"input_{current_node_id}"] = user_input

			# Determine what the *current* graph would route to
			resolution = await self.resolve_next_node(
				flow_id, current_node_id, context, tenant_id
			)
			current_next = (
				resolution["target_node"]["node_id"]
				if resolution["target_node"] else None
			)

			# What was actually observed
			observed_next = events[i + 1]["node_id"] if i + 1 < len(events) else None

			step: dict[str, Any] = {
				"step": i + 1,
				"node_id": current_node_id,
				"user_input": user_input,
				"observed_next": observed_next,
				"replayed_next": current_next,
				"diverged": current_next != observed_next,
			}
			replay_steps.append(step)

			if step["diverged"]:
				divergences.append({
					"step": i + 1,
					"node_id": current_node_id,
					"observed_next": observed_next,
					"replayed_next": current_next,
				})

		return {
			"session_id": session_id,
			"flow_id": flow_id,
			"total_steps": len(replay_steps),
			"divergences": divergences,
			"divergence_count": len(divergences),
			"clean_replay": len(divergences) == 0,
			"steps": replay_steps,
			"replayed_at": self._now(),
		}

	# ── Bulk edge operations ──────────────────────────────────────────────────

	async def bulk_add_edges(
		self,
		flow_id: str,
		edges: list[dict[str, Any]],
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Add multiple edges to a flow in a single call.

		Mirrors ``bulk_add_nodes``.  Each entry in *edges* must contain at
		minimum ``source_node_id`` and ``target_node_id``.  Optional fields:
		``label``, ``condition``, ``priority``, ``metadata``.

		Returns:
			Summary dict with added count, failed count, edges list, errors list.
		"""
		tenant = self._tenant(tenant_id)
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []

		tasks = [
			self.add_edge(
				flow_id=flow_id,
				source_node_id=e["source_node_id"],
				target_node_id=e["target_node_id"],
				tenant_id=tenant,
				label=e.get("label", ""),
				condition=e.get("condition"),
				priority=e.get("priority", 0),
				metadata=e.get("metadata"),
			)
			for e in edges
		]
		raw = await asyncio.gather(*tasks, return_exceptions=True)
		for e, r in zip(edges, raw):
			if isinstance(r, Exception):
				errors.append({"input": e, "error": str(r)})
			else:
				results.append(r)

		return {
			"added": len(results),
			"failed": len(errors),
			"edges": results,
			"errors": errors,
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_audit_events']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

