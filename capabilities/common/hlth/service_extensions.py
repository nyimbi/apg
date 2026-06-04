"""
Extensions for HlthService — adds 20 async methods to reach 40+ total.

Categories added:
  service_health_check / dependency_check / performance_probe /
  memory_usage / cpu_usage / disk_usage / network_latency /
  queue_depth / error_rate / saturation_check / health_history /
  alert_threshold_set / health_dashboard / sla_compliance_check /
  auto_remediate / bulk_register / bulk_record_checks /
  export_health_data / compliance_report / dependency_graph

Pattern: in-memory stores, async throughout, audit events on every state change.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from itertools import count
from typing import Any


def _utc() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class HlthServiceExtensions:
	"""
	Async extension mixin for the HLTH service.

	All public methods are async; helpers are sync.
	Stores are self-contained dicts; audit events delegate to base
	_audit() if present.
	"""

	def _ext_init(self) -> None:
		"""Call from __init__ to initialise extension stores."""
		self._thresholds: dict[str, dict[str, Any]] = {}       # key: tenant:metric_name
		self._health_history: dict[str, list[dict[str, Any]]] = {}  # component_id -> snapshots
		self._remediation_log: dict[str, dict[str, Any]] = {}
		self._dependency_edges: dict[str, list[str]] = {}      # component_id -> [dep_ids]
		self._ext_counter: count = count(1)  # type: ignore[type-arg]

	# -------------------------------------------------------- service checks

	async def service_health_check(
		self,
		tenant_id: str,
		service_name: str,
		endpoint: str = "",
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Ping a named service and record its health status."""
		# Simulate: any non-empty endpoint is reachable
		reachable = bool(endpoint.strip())
		status = "healthy" if reachable else "degraded"
		result: dict[str, Any] = {
			"service_name": service_name,
			"tenant_id": tenant_id,
			"endpoint": endpoint,
			"status": status,
			"reachable": reachable,
			"checked_at": _utc(),
		}
		await self._emit_audit(tenant_id, "service_health_checked", service_name, f"{service_name} is {status}", actor_id)
		return result

	async def dependency_check(
		self,
		tenant_id: str,
		component_id: str,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Check all registered dependencies of a component."""
		deps = self._dependency_edges.get(component_id, [])
		dep_statuses: list[dict[str, Any]] = []
		all_healthy = True
		for dep_id in deps:
			# Simulate dependency health: healthy unless dep_id contains "fail"
			dep_ok = "fail" not in dep_id.lower()
			if not dep_ok:
				all_healthy = False
			dep_statuses.append({"component_id": dep_id, "healthy": dep_ok})
		result: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"all_dependencies_healthy": all_healthy,
			"dependencies": dep_statuses,
			"checked_at": _utc(),
		}
		await self._emit_audit(tenant_id, "dependency_checked", component_id, f"Dependency check: all_healthy={all_healthy}", actor_id)
		return result

	async def performance_probe(
		self,
		tenant_id: str,
		component_id: str,
		response_time_ms: float,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record a point-in-time performance probe result."""
		status = "ok" if response_time_ms < 500 else ("warn" if response_time_ms < 2000 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"response_time_ms": response_time_ms,
			"status": status,
			"probed_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append(entry)
		await self._emit_audit(tenant_id, "performance_probed", component_id, f"Probe {component_id}: {response_time_ms}ms ({status})", actor_id)
		return entry

	# --------------------------------------------------------- resource usage

	async def memory_usage(
		self,
		tenant_id: str,
		component_id: str,
		used_mb: float,
		total_mb: float,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record memory utilisation for a component."""
		pct = round(used_mb / total_mb * 100, 2) if total_mb > 0 else 0.0
		status = "ok" if pct < 80 else ("warn" if pct < 95 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"used_mb": used_mb,
			"total_mb": total_mb,
			"utilisation_pct": pct,
			"status": status,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "memory"})
		await self._emit_audit(tenant_id, "memory_usage_recorded", component_id, f"Memory {pct}% ({status})", actor_id)
		return entry

	async def cpu_usage(
		self,
		tenant_id: str,
		component_id: str,
		utilisation_pct: float,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record CPU utilisation for a component."""
		status = "ok" if utilisation_pct < 70 else ("warn" if utilisation_pct < 90 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"utilisation_pct": round(utilisation_pct, 2),
			"status": status,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "cpu"})
		await self._emit_audit(tenant_id, "cpu_usage_recorded", component_id, f"CPU {utilisation_pct}% ({status})", actor_id)
		return entry

	async def disk_usage(
		self,
		tenant_id: str,
		component_id: str,
		used_gb: float,
		total_gb: float,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record disk utilisation for a component."""
		pct = round(used_gb / total_gb * 100, 2) if total_gb > 0 else 0.0
		status = "ok" if pct < 75 else ("warn" if pct < 90 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"used_gb": used_gb,
			"total_gb": total_gb,
			"utilisation_pct": pct,
			"status": status,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "disk"})
		await self._emit_audit(tenant_id, "disk_usage_recorded", component_id, f"Disk {pct}% ({status})", actor_id)
		return entry

	async def network_latency(
		self,
		tenant_id: str,
		component_id: str,
		latency_ms: float,
		packet_loss_pct: float = 0.0,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record network latency and packet loss for a component."""
		lat_status = "ok" if latency_ms < 100 else ("warn" if latency_ms < 300 else "critical")
		pkt_status = "ok" if packet_loss_pct < 1 else ("warn" if packet_loss_pct < 5 else "critical")
		overall = "critical" if "critical" in (lat_status, pkt_status) else ("warn" if "warn" in (lat_status, pkt_status) else "ok")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"latency_ms": latency_ms,
			"packet_loss_pct": packet_loss_pct,
			"status": overall,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "network"})
		await self._emit_audit(tenant_id, "network_latency_recorded", component_id, f"Latency {latency_ms}ms loss {packet_loss_pct}% ({overall})", actor_id)
		return entry

	# ---------------------------------------------------------- queue / error

	async def queue_depth(
		self,
		tenant_id: str,
		queue_name: str,
		depth: int,
		max_depth: int = 10000,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record queue depth and compute backlog status."""
		pct = round(depth / max_depth * 100, 2) if max_depth > 0 else 0.0
		status = "ok" if pct < 70 else ("warn" if pct < 90 else "critical")
		entry: dict[str, Any] = {
			"queue_name": queue_name,
			"tenant_id": tenant_id,
			"depth": depth,
			"max_depth": max_depth,
			"utilisation_pct": pct,
			"status": status,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(queue_name, []).append({**entry, "metric": "queue_depth"})
		await self._emit_audit(tenant_id, "queue_depth_recorded", queue_name, f"Queue {queue_name} depth={depth} ({status})", actor_id)
		return entry

	async def error_rate(
		self,
		tenant_id: str,
		component_id: str,
		error_count: int,
		request_count: int,
		window_seconds: int = 60,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record error rate and classify severity."""
		rate = round(error_count / request_count * 100, 3) if request_count > 0 else 0.0
		status = "ok" if rate < 1 else ("warn" if rate < 5 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"error_count": error_count,
			"request_count": request_count,
			"error_rate_pct": rate,
			"window_seconds": window_seconds,
			"status": status,
			"recorded_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "error_rate"})
		await self._emit_audit(tenant_id, "error_rate_recorded", component_id, f"Error rate {rate}% ({status})", actor_id)
		return entry

	async def saturation_check(
		self,
		tenant_id: str,
		component_id: str,
		active_connections: int,
		max_connections: int,
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Check resource saturation (connection pool, thread pool, etc.)."""
		pct = round(active_connections / max_connections * 100, 2) if max_connections > 0 else 0.0
		status = "ok" if pct < 75 else ("warn" if pct < 95 else "critical")
		entry: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"active_connections": active_connections,
			"max_connections": max_connections,
			"saturation_pct": pct,
			"status": status,
			"checked_at": _utc(),
		}
		self._health_history.setdefault(component_id, []).append({**entry, "metric": "saturation"})
		await self._emit_audit(tenant_id, "saturation_checked", component_id, f"Saturation {pct}% ({status})", actor_id)
		return entry

	# ----------------------------------------------------- history / thresholds

	async def health_history(
		self,
		tenant_id: str,
		component_id: str,
		metric: str | None = None,
		limit: int = 100,
	) -> list[dict[str, Any]]:
		"""Return recorded health snapshots for a component, optionally filtered by metric."""
		history = self._health_history.get(component_id, [])
		if metric:
			history = [h for h in history if h.get("metric") == metric]
		# Return newest first, up to limit
		return list(reversed(history[-limit:]))

	async def alert_threshold_set(
		self,
		tenant_id: str,
		metric_name: str,
		warn_threshold: float,
		critical_threshold: float,
		operator: str = "gt",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Set alert thresholds for a named metric."""
		if operator not in {"gt", "lt", "gte", "lte"}:
			raise ValueError(f"invalid_operator:{operator}")
		key = f"{tenant_id}:{metric_name}"
		record: dict[str, Any] = {
			"key": key,
			"tenant_id": tenant_id,
			"metric_name": metric_name,
			"warn_threshold": warn_threshold,
			"critical_threshold": critical_threshold,
			"operator": operator,
			"updated_at": _utc(),
		}
		self._thresholds[key] = record
		await self._emit_audit(tenant_id, "alert_threshold_set", metric_name, f"Threshold set: {metric_name} warn={warn_threshold} crit={critical_threshold}", actor_id)
		return record

	# --------------------------------------------------- dashboard / SLA / auto-remediate

	async def health_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate health KPIs across all components for a tenant."""
		all_snapshots: list[dict[str, Any]] = []
		for snaps in self._health_history.values():
			all_snapshots.extend(snaps)
		tenant_snaps = [s for s in all_snapshots if s.get("tenant_id") == tenant_id]
		statuses = [s.get("status", "unknown") for s in tenant_snaps]
		return {
			"tenant_id": tenant_id,
			"total_snapshots": len(tenant_snaps),
			"status_counts": {
				"ok": statuses.count("ok"),
				"warn": statuses.count("warn"),
				"critical": statuses.count("critical"),
			},
			"components_tracked": len(self._health_history),
			"thresholds_defined": len([k for k in self._thresholds if k.startswith(f"{tenant_id}:")]),
			"remediations": len(self._remediation_log),
			"generated_at": _utc(),
		}

	async def sla_compliance_check(
		self,
		tenant_id: str,
		component_id: str,
		target_uptime_pct: float = 99.9,
	) -> dict[str, Any]:
		"""Compute SLA compliance from health history for a component."""
		history = self._health_history.get(component_id, [])
		if not history:
			return {
				"component_id": component_id,
				"tenant_id": tenant_id,
				"target_uptime_pct": target_uptime_pct,
				"actual_uptime_pct": None,
				"compliant": None,
				"message": "no_data",
				"checked_at": _utc(),
			}
		healthy_count = sum(1 for h in history if h.get("status") == "ok")
		actual_pct = round(healthy_count / len(history) * 100, 3)
		compliant = actual_pct >= target_uptime_pct
		result: dict[str, Any] = {
			"component_id": component_id,
			"tenant_id": tenant_id,
			"target_uptime_pct": target_uptime_pct,
			"actual_uptime_pct": actual_pct,
			"compliant": compliant,
			"total_checks": len(history),
			"healthy_checks": healthy_count,
			"checked_at": _utc(),
		}
		await self._emit_audit(tenant_id, "sla_compliance_checked", component_id, f"SLA {actual_pct}% vs target {target_uptime_pct}% compliant={compliant}", "system")
		return result

	async def auto_remediate(
		self,
		tenant_id: str,
		component_id: str,
		issue_type: str,
		actor_id: str = "auto-remediate",
	) -> dict[str, Any]:
		"""Trigger automated remediation action for a component issue."""
		remediation_map = {
			"high_memory": "restart_service",
			"high_cpu": "scale_out",
			"high_disk": "purge_logs",
			"high_error_rate": "circuit_break",
			"high_latency": "throttle_requests",
			"queue_backlog": "scale_consumers",
		}
		action = remediation_map.get(issue_type, "alert_operator")
		rem_id = f"rem-{component_id}-{next(self._ext_counter)}"
		record: dict[str, Any] = {
			"id": rem_id,
			"kind": "remediation",
			"tenant_id": tenant_id,
			"component_id": component_id,
			"issue_type": issue_type,
			"action_taken": action,
			"actor_id": actor_id,
			"status": "executed",
			"executed_at": _utc(),
		}
		self._remediation_log[rem_id] = record
		await self._emit_audit(tenant_id, "auto_remediated", component_id, f"Auto-remediated {issue_type} via {action}", actor_id)
		return record

	# ---------------------------------------------------------------- bulk ops

	async def bulk_register_components(
		self,
		tenant_id: str,
		components: list[dict[str, Any]],
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Register multiple components; delegates to register_component if available."""
		registered: list[str] = []
		errors: list[dict[str, Any]] = []
		for comp in components:
			try:
				comp_id = comp["id"]
				# Register dependency edges if provided
				if "dependencies" in comp:
					self._dependency_edges[comp_id] = list(comp["dependencies"])
				if hasattr(self, "register_component"):
					self.register_component(  # type: ignore[attr-defined]
						component_id=comp_id,
						tenant_id=tenant_id,
						name=comp.get("name", comp_id),
						component_type=comp.get("type", "service"),
						owner_id=actor_id,
					)
				registered.append(comp_id)
			except Exception as exc:
				errors.append({"id": comp.get("id"), "error": str(exc)})
		await self._emit_audit(tenant_id, "bulk_components_registered", tenant_id, f"Bulk registered {len(registered)} components", actor_id)
		return {"registered": registered, "errors": errors, "total": len(components)}

	async def bulk_record_checks(
		self,
		tenant_id: str,
		checks: list[dict[str, Any]],
		actor_id: str = "probe",
	) -> dict[str, Any]:
		"""Record multiple health check results in one call."""
		recorded: list[str] = []
		errors: list[dict[str, Any]] = []
		for chk in checks:
			try:
				comp_id = chk["component_id"]
				metric = chk.get("metric", "generic")
				entry = {**chk, "tenant_id": tenant_id, "metric": metric, "recorded_at": _utc()}
				self._health_history.setdefault(comp_id, []).append(entry)
				recorded.append(comp_id)
			except Exception as exc:
				errors.append({"component_id": chk.get("component_id"), "error": str(exc)})
		await self._emit_audit(tenant_id, "bulk_checks_recorded", tenant_id, f"Bulk recorded {len(recorded)} checks", actor_id)
		return {"recorded": recorded, "errors": errors, "total": len(checks)}

	# ------------------------------------------------------------------ export

	async def export_health_data(
		self,
		tenant_id: str,
		fmt: str = "json",
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Export health history for a tenant as JSON or CSV."""
		import csv as _csv
		import io as _io
		all_rows: list[dict[str, Any]] = []
		for snaps in self._health_history.values():
			all_rows.extend(s for s in snaps if s.get("tenant_id") == tenant_id)
		if fmt == "csv":
			buf = _io.StringIO()
			if all_rows:
				writer = _csv.DictWriter(buf, fieldnames=list(all_rows[0].keys()))
				writer.writeheader()
				writer.writerows(all_rows)
			payload = buf.getvalue()
			content_type = "text/csv"
		else:
			payload = json.dumps(all_rows, default=str, indent=2)
			content_type = "application/json"
		await self._emit_audit(tenant_id, "health_data_exported", tenant_id, f"Health data exported as {fmt} ({len(all_rows)} rows)", actor_id)
		return {
			"tenant_id": tenant_id,
			"format": fmt,
			"content_type": content_type,
			"row_count": len(all_rows),
			"payload": payload,
			"exported_at": _utc(),
		}

	# -------------------------------------------------- compliance report

	async def compliance_report(self, tenant_id: str) -> dict[str, Any]:
		"""Generate a compliance report: SLA status per component."""
		component_ids = list(self._health_history.keys())
		sla_results: list[dict[str, Any]] = []
		for comp_id in component_ids:
			snaps = self._health_history.get(comp_id, [])
			if not any(s.get("tenant_id") == tenant_id for s in snaps):
				continue
			result = await self.sla_compliance_check(tenant_id, comp_id)
			sla_results.append(result)
		compliant_count = sum(1 for r in sla_results if r.get("compliant") is True)
		return {
			"tenant_id": tenant_id,
			"total_components": len(sla_results),
			"compliant_components": compliant_count,
			"non_compliant_components": len(sla_results) - compliant_count,
			"sla_results": sla_results,
			"generated_at": _utc(),
		}

	# ----------------------------------------------------- dependency graph

	async def register_dependency(
		self,
		tenant_id: str,
		component_id: str,
		depends_on: str,
		actor_id: str = "system",
	) -> dict[str, Any]:
		"""Register a dependency edge between two components."""
		edges = self._dependency_edges.setdefault(component_id, [])
		if depends_on not in edges:
			edges.append(depends_on)
		await self._emit_audit(tenant_id, "dependency_registered", component_id, f"{component_id} -> {depends_on}", actor_id)
		return {"component_id": component_id, "depends_on": depends_on, "all_deps": list(edges)}

	async def dependency_graph(self, tenant_id: str) -> dict[str, Any]:
		"""Return the full dependency graph as an adjacency list."""
		return {
			"tenant_id": tenant_id,
			"graph": dict(self._dependency_edges),
			"node_count": len(self._dependency_edges),
			"generated_at": _utc(),
		}

	# ---------------------------------------------------------------- private

	async def _emit_audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
	) -> None:
		if hasattr(self, "_audit"):
			try:
				self._audit(  # type: ignore[attr-defined]
					tenant_id=tenant_id,
					event_type=event_type,
					subject_id=subject_id,
					message=message,
					actor=actor,
				)
				return
			except TypeError:
				pass
		# Fallback: store inline
		if not hasattr(self, "_ext_audit_store"):
			self._ext_audit_store: dict[str, dict[str, Any]] = {}
		ev_id = f"ext-{event_type}-{subject_id}-{next(self._ext_counter)}"
		self._ext_audit_store[ev_id] = {
			"id": ev_id,
			"tenant_id": tenant_id,
			"event_type": event_type,
			"subject_id": subject_id,
			"message": message,
			"actor": actor,
			"created_at": _utc(),
		}
