"""Metrics & SLO service (obs_mtx).

RED metrics (Rate/Error/Duration), SLO definition, burn rate alerts,
Prometheus export, dashboard generation.
"""
from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "obs_mtx"
SUPPORTED_METRIC_TYPES = {"counter", "gauge", "histogram", "summary"}
SUPPORTED_SLO_TYPES = {"availability", "latency", "error_rate", "throughput"}
SUPPORTED_SEVERITIES = {"critical", "warning", "info"}


def _now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sid() -> str:
	return uuid4().hex[:16]


class MetricsSLOService:
	"""In-memory async service for metrics collection, SLO management, and burn rate alerting."""

	def __init__(self, tenant_id: str = "default") -> None:
		guard_tenant_id(tenant_id)
		self.tenant_id = tenant_id
		self._metric_definitions: dict[str, dict[str, Any]] = {}
		self._data_points: list[dict[str, Any]] = []
		self._slos: dict[str, dict[str, Any]] = {}
		self._burn_rate_alerts: dict[str, dict[str, Any]] = {}
		self._dashboards: dict[str, dict[str, Any]] = {}
		self._prometheus_config: dict[str, Any] | None = None
		self._audit_events: list[dict[str, Any]] = []
		_log.info("MetricsSLOService initialised tenant=%s", self.tenant_id)

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

	def _percentile(self, data: list[float], pct: float) -> float:
		if not data:
			return 0.0
		sorted_data = sorted(data)
		idx = int(math.ceil(len(sorted_data) * pct / 100)) - 1
		return sorted_data[max(0, idx)]

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy",
			"capability": CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"metric_definitions": len(self._metric_definitions),
			"data_points": len(self._data_points),
			"slos": len(self._slos),
			"burn_rate_alerts": len(self._burn_rate_alerts),
			"dashboards": len(self._dashboards),
			"checked_at": _now(),
		}

	# ------------------------------------------------------------------ describe

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "observability",
			"description": "RED metrics (Rate/Error/Duration), SLO definition, burn rate alerts, Prometheus export, dashboard generation",
			"supported_metric_types": list(SUPPORTED_METRIC_TYPES),
			"supported_slo_types": list(SUPPORTED_SLO_TYPES),
			"supported_severities": list(SUPPORTED_SEVERITIES),
			"tenant_id": self.tenant_id,
		}

	# ------------------------------------------------------------------ audit

	async def get_audit_events(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return deepcopy(events[-limit:])

	# ------------------------------------------------------------------ metric definitions

	async def create_metric_definition(
		self,
		name: str,
		service_name: str,
		metric_type: str,
		description: str = "",
		unit: str = "",
		labels: list[str] | None = None,
		namespace: str = "apg",
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		guard_non_empty_string(service_name, "service_name")
		if metric_type not in SUPPORTED_METRIC_TYPES:
			raise ValueError(f"Unsupported metric_type: {metric_type}")
		# Check for name collision
		existing = [m for m in self._metric_definitions.values() if m["name"] == name and m["service_name"] == service_name]
		if existing:
			raise ValueError(f"Metric '{name}' already defined for service '{service_name}'")
		def_id = _sid()
		record: dict[str, Any] = {
			"id": def_id,
			"name": name,
			"description": description,
			"metric_type": metric_type,
			"unit": unit,
			"labels": labels or [],
			"service_name": service_name,
			"namespace": namespace,
			"enabled": True,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._metric_definitions[def_id] = record
		self._emit("metric_definition_created", def_id, "metric_definition", {"name": name, "type": metric_type})
		_log.debug("metric definition created name=%s service=%s", name, service_name)
		return deepcopy(record)

	async def update_metric_definition(
		self,
		def_id: str,
		description: str | None = None,
		labels: list[str] | None = None,
		enabled: bool | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(def_id, "def_id")
		if def_id not in self._metric_definitions:
			raise KeyError(f"Metric definition not found: {def_id}")
		defn = self._metric_definitions[def_id]
		if description is not None:
			defn["description"] = description
		if labels is not None:
			defn["labels"] = labels
		if enabled is not None:
			defn["enabled"] = enabled
		self._emit("metric_definition_updated", def_id, "metric_definition")
		return deepcopy(defn)

	async def get_metric_definition(self, def_id: str) -> dict[str, Any]:
		guard_non_empty_string(def_id, "def_id")
		if def_id not in self._metric_definitions:
			raise KeyError(f"Metric definition not found: {def_id}")
		return deepcopy(self._metric_definitions[def_id])

	async def list_metric_definitions(
		self,
		service_name: str | None = None,
		metric_type: str | None = None,
		enabled_only: bool = True,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		defs = list(self._metric_definitions.values())
		if service_name:
			defs = [d for d in defs if d["service_name"] == service_name]
		if metric_type:
			defs = [d for d in defs if d["metric_type"] == metric_type]
		if enabled_only:
			defs = [d for d in defs if d["enabled"]]
		total = len(defs)
		offset = (page - 1) * page_size
		return {"items": [deepcopy(d) for d in defs[offset: offset + page_size]], "total": total, "page": page, "page_size": page_size}

	async def delete_metric_definition(self, def_id: str) -> dict[str, Any]:
		guard_non_empty_string(def_id, "def_id")
		if def_id not in self._metric_definitions:
			raise KeyError(f"Metric definition not found: {def_id}")
		del self._metric_definitions[def_id]
		self._emit("metric_definition_deleted", def_id, "metric_definition")
		return {"deleted": True, "def_id": def_id}

	# ------------------------------------------------------------------ data points

	async def record_metric(
		self,
		metric_name: str,
		value: float,
		service_name: str,
		labels: dict[str, str] | None = None,
		timestamp: str | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(metric_name, "metric_name")
		guard_non_empty_string(service_name, "service_name")
		point: dict[str, Any] = {
			"id": _sid(),
			"metric_name": metric_name,
			"value": value,
			"labels": labels or {},
			"timestamp": timestamp or _now(),
			"service_name": service_name,
			"tenant_id": self.tenant_id,
		}
		self._data_points.append(point)
		# Prune in-memory store beyond 100k points to prevent unbounded growth
		if len(self._data_points) > 100_000:
			self._data_points = self._data_points[-100_000:]
		return deepcopy(point)

	async def bulk_record_metrics(self, points: list[dict[str, Any]]) -> dict[str, Any]:
		tasks = [
			self.record_metric(
				metric_name=p.get("metric_name", ""),
				value=float(p.get("value", 0)),
				service_name=p.get("service_name", ""),
				labels=p.get("labels"),
				timestamp=p.get("timestamp"),
			)
			for p in points
		]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		ok = [r for r in results if not isinstance(r, Exception)]
		errs = [{"error": str(r)} for r in results if isinstance(r, Exception)]
		return {"recorded": len(ok), "errors": errs}

	async def query_metric(
		self,
		metric_name: str,
		service_name: str | None = None,
		labels_match: dict[str, str] | None = None,
		start_time: str | None = None,
		end_time: str | None = None,
		limit: int = 1000,
	) -> list[dict[str, Any]]:
		guard_non_empty_string(metric_name, "metric_name")
		points = [p for p in self._data_points if p["metric_name"] == metric_name]
		if service_name:
			points = [p for p in points if p["service_name"] == service_name]
		if labels_match:
			points = [p for p in points if all(p["labels"].get(k) == v for k, v in labels_match.items())]
		if start_time:
			points = [p for p in points if p["timestamp"] >= start_time]
		if end_time:
			points = [p for p in points if p["timestamp"] <= end_time]
		return [deepcopy(p) for p in points[-limit:]]

	# ------------------------------------------------------------------ RED metrics

	async def compute_red_metrics(self, service_name: str, window_minutes: int = 5) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		# Derive from recorded data points by convention: metric names
		rate_points = [p for p in self._data_points if p["service_name"] == service_name and p["metric_name"].endswith("_requests_total")]
		error_points = [p for p in self._data_points if p["service_name"] == service_name and p["metric_name"].endswith("_errors_total")]
		duration_points = [p for p in self._data_points if p["service_name"] == service_name and p["metric_name"].endswith("_duration_ms")]
		durations = [p["value"] for p in duration_points]
		total_requests = sum(p["value"] for p in rate_points) if rate_points else len(rate_points)
		total_errors = sum(p["value"] for p in error_points) if error_points else 0
		return {
			"service_name": service_name,
			"window_minutes": window_minutes,
			"request_rate": len(rate_points) / max(window_minutes, 1),
			"error_rate": total_errors / max(total_requests, 1) if total_requests > 0 else 0.0,
			"p50_duration_ms": self._percentile(durations, 50),
			"p95_duration_ms": self._percentile(durations, 95),
			"p99_duration_ms": self._percentile(durations, 99),
			"total_requests": int(total_requests),
			"total_errors": int(total_errors),
			"tenant_id": self.tenant_id,
			"computed_at": _now(),
		}

	async def compute_red_metrics_all_services(self, window_minutes: int = 5) -> list[dict[str, Any]]:
		services = list({p["service_name"] for p in self._data_points})
		tasks = [self.compute_red_metrics(svc, window_minutes) for svc in services]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		return [r for r in results if not isinstance(r, Exception)]

	# ------------------------------------------------------------------ SLOs

	async def create_slo(
		self,
		name: str,
		service_name: str,
		slo_type: str,
		target: float,
		window_days: int = 30,
		description: str = "",
		good_query: str = "",
		total_query: str = "",
		latency_threshold_ms: float | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		guard_non_empty_string(service_name, "service_name")
		if slo_type not in SUPPORTED_SLO_TYPES:
			raise ValueError(f"Unsupported SLO type: {slo_type}")
		if not (0.0 <= target <= 100.0):
			raise ValueError("target must be between 0.0 and 100.0")
		slo_id = _sid()
		record: dict[str, Any] = {
			"id": slo_id,
			"name": name,
			"description": description,
			"service_name": service_name,
			"slo_type": slo_type,
			"target": target,
			"window_days": window_days,
			"good_query": good_query,
			"total_query": total_query,
			"latency_threshold_ms": latency_threshold_ms,
			"enabled": True,
			"current_compliance": None,
			"error_budget_remaining": None,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
			"updated_at": None,
		}
		self._slos[slo_id] = record
		self._emit("slo_created", slo_id, "slo", {"name": name, "type": slo_type, "target": target})
		_log.debug("SLO created id=%s name=%s target=%.2f%%", slo_id, name, target)
		return deepcopy(record)

	async def update_slo(
		self,
		slo_id: str,
		description: str | None = None,
		target: float | None = None,
		window_days: int | None = None,
		good_query: str | None = None,
		total_query: str | None = None,
		latency_threshold_ms: float | None = None,
		enabled: bool | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(slo_id, "slo_id")
		if slo_id not in self._slos:
			raise KeyError(f"SLO not found: {slo_id}")
		slo = self._slos[slo_id]
		if description is not None:
			slo["description"] = description
		if target is not None:
			if not (0.0 <= target <= 100.0):
				raise ValueError("target must be between 0.0 and 100.0")
			slo["target"] = target
		if window_days is not None:
			slo["window_days"] = window_days
		if good_query is not None:
			slo["good_query"] = good_query
		if total_query is not None:
			slo["total_query"] = total_query
		if latency_threshold_ms is not None:
			slo["latency_threshold_ms"] = latency_threshold_ms
		if enabled is not None:
			slo["enabled"] = enabled
		slo["updated_at"] = _now()
		self._emit("slo_updated", slo_id, "slo")
		return deepcopy(slo)

	async def get_slo(self, slo_id: str) -> dict[str, Any]:
		guard_non_empty_string(slo_id, "slo_id")
		if slo_id not in self._slos:
			raise KeyError(f"SLO not found: {slo_id}")
		return deepcopy(self._slos[slo_id])

	async def list_slos(
		self,
		service_name: str | None = None,
		slo_type: str | None = None,
		enabled_only: bool = True,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		slos = list(self._slos.values())
		if service_name:
			slos = [s for s in slos if s["service_name"] == service_name]
		if slo_type:
			slos = [s for s in slos if s["slo_type"] == slo_type]
		if enabled_only:
			slos = [s for s in slos if s["enabled"]]
		total = len(slos)
		offset = (page - 1) * page_size
		return {"items": [deepcopy(s) for s in slos[offset: offset + page_size]], "total": total, "page": page, "page_size": page_size}

	async def delete_slo(self, slo_id: str) -> dict[str, Any]:
		guard_non_empty_string(slo_id, "slo_id")
		if slo_id not in self._slos:
			raise KeyError(f"SLO not found: {slo_id}")
		del self._slos[slo_id]
		self._emit("slo_deleted", slo_id, "slo")
		return {"deleted": True, "slo_id": slo_id}

	async def evaluate_slo_compliance(self, slo_id: str) -> dict[str, Any]:
		"""Evaluate current SLO compliance against recorded data points."""
		guard_non_empty_string(slo_id, "slo_id")
		if slo_id not in self._slos:
			raise KeyError(f"SLO not found: {slo_id}")
		slo = self._slos[slo_id]
		svc = slo["service_name"]
		slo_type = slo["slo_type"]
		target = slo["target"]

		if slo_type == "availability":
			up_points = [p for p in self._data_points if p["service_name"] == svc and p["metric_name"].endswith("_up")]
			compliance = (sum(p["value"] for p in up_points) / max(len(up_points), 1)) * 100 if up_points else 100.0
		elif slo_type == "error_rate":
			err_points = [p for p in self._data_points if p["service_name"] == svc and p["metric_name"].endswith("_errors_total")]
			req_points = [p for p in self._data_points if p["service_name"] == svc and p["metric_name"].endswith("_requests_total")]
			total_req = sum(p["value"] for p in req_points) or 1
			total_err = sum(p["value"] for p in err_points)
			compliance = (1 - total_err / total_req) * 100
		elif slo_type == "latency":
			dur_points = [p["value"] for p in self._data_points if p["service_name"] == svc and p["metric_name"].endswith("_duration_ms")]
			threshold = slo.get("latency_threshold_ms") or 1000.0
			good = sum(1 for d in dur_points if d <= threshold)
			compliance = (good / max(len(dur_points), 1)) * 100
		else:
			compliance = 100.0

		error_budget_consumed = max(0.0, (target - compliance) / max(100.0 - target, 0.001))
		error_budget_remaining = max(0.0, 100.0 - error_budget_consumed * 100)

		slo["current_compliance"] = round(compliance, 4)
		slo["error_budget_remaining"] = round(error_budget_remaining, 4)
		slo["updated_at"] = _now()

		return {
			"slo_id": slo_id,
			"slo_name": slo["name"],
			"service_name": svc,
			"target": target,
			"current_compliance": round(compliance, 4),
			"error_budget_remaining": round(error_budget_remaining, 4),
			"meeting_slo": compliance >= target,
			"evaluated_at": _now(),
		}

	async def evaluate_all_slos(self) -> list[dict[str, Any]]:
		tasks = [self.evaluate_slo_compliance(sid) for sid in self._slos]
		results = await asyncio.gather(*tasks, return_exceptions=True)
		return [r for r in results if not isinstance(r, Exception)]

	# ------------------------------------------------------------------ burn rate alerts

	async def create_burn_rate_alert(
		self,
		slo_id: str,
		name: str,
		short_window_minutes: int = 60,
		long_window_minutes: int = 360,
		burn_rate_threshold: float = 14.4,
		severity: str = "critical",
		notification_channels: list[str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(slo_id, "slo_id")
		guard_non_empty_string(name, "name")
		if slo_id not in self._slos:
			raise KeyError(f"SLO not found: {slo_id}")
		if severity not in SUPPORTED_SEVERITIES:
			raise ValueError(f"Unsupported severity: {severity}")
		alert_id = _sid()
		record: dict[str, Any] = {
			"id": alert_id,
			"slo_id": slo_id,
			"name": name,
			"short_window_minutes": short_window_minutes,
			"long_window_minutes": long_window_minutes,
			"burn_rate_threshold": burn_rate_threshold,
			"severity": severity,
			"notification_channels": notification_channels or [],
			"enabled": True,
			"firing": False,
			"last_fired_at": None,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._burn_rate_alerts[alert_id] = record
		self._emit("burn_rate_alert_created", alert_id, "burn_rate_alert", {"slo_id": slo_id, "severity": severity})
		return deepcopy(record)

	async def update_burn_rate_alert(
		self,
		alert_id: str,
		enabled: bool | None = None,
		burn_rate_threshold: float | None = None,
		severity: str | None = None,
		notification_channels: list[str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(alert_id, "alert_id")
		if alert_id not in self._burn_rate_alerts:
			raise KeyError(f"Burn rate alert not found: {alert_id}")
		alert = self._burn_rate_alerts[alert_id]
		if enabled is not None:
			alert["enabled"] = enabled
		if burn_rate_threshold is not None:
			alert["burn_rate_threshold"] = burn_rate_threshold
		if severity is not None:
			if severity not in SUPPORTED_SEVERITIES:
				raise ValueError(f"Unsupported severity: {severity}")
			alert["severity"] = severity
		if notification_channels is not None:
			alert["notification_channels"] = notification_channels
		self._emit("burn_rate_alert_updated", alert_id, "burn_rate_alert")
		return deepcopy(alert)

	async def get_burn_rate_alert(self, alert_id: str) -> dict[str, Any]:
		guard_non_empty_string(alert_id, "alert_id")
		if alert_id not in self._burn_rate_alerts:
			raise KeyError(f"Burn rate alert not found: {alert_id}")
		return deepcopy(self._burn_rate_alerts[alert_id])

	async def list_burn_rate_alerts(self, slo_id: str | None = None, firing_only: bool = False) -> list[dict[str, Any]]:
		alerts = list(self._burn_rate_alerts.values())
		if slo_id:
			alerts = [a for a in alerts if a["slo_id"] == slo_id]
		if firing_only:
			alerts = [a for a in alerts if a["firing"]]
		return [deepcopy(a) for a in alerts]

	async def delete_burn_rate_alert(self, alert_id: str) -> dict[str, Any]:
		guard_non_empty_string(alert_id, "alert_id")
		if alert_id not in self._burn_rate_alerts:
			raise KeyError(f"Burn rate alert not found: {alert_id}")
		del self._burn_rate_alerts[alert_id]
		self._emit("burn_rate_alert_deleted", alert_id, "burn_rate_alert")
		return {"deleted": True, "alert_id": alert_id}

	async def evaluate_burn_rate(self, alert_id: str) -> dict[str, Any]:
		"""Calculate current burn rate and determine if alert fires."""
		guard_non_empty_string(alert_id, "alert_id")
		if alert_id not in self._burn_rate_alerts:
			raise KeyError(f"Burn rate alert not found: {alert_id}")
		alert = self._burn_rate_alerts[alert_id]
		slo = self._slos.get(alert["slo_id"])
		if not slo:
			raise KeyError(f"SLO {alert['slo_id']} not found for alert {alert_id}")

		compliance = await self.evaluate_slo_compliance(alert["slo_id"])
		error_budget_remaining = compliance["error_budget_remaining"]
		window_fraction = alert["short_window_minutes"] / (slo["window_days"] * 24 * 60)
		burn_rate = (100 - error_budget_remaining) / (100 * window_fraction) if window_fraction > 0 else 0.0
		fires = burn_rate >= alert["burn_rate_threshold"]

		if fires and not alert["firing"]:
			alert["firing"] = True
			alert["last_fired_at"] = _now()
			self._emit("burn_rate_alert_fired", alert_id, "burn_rate_alert", {"burn_rate": burn_rate, "severity": alert["severity"]})
		elif not fires:
			alert["firing"] = False

		return {
			"alert_id": alert_id,
			"slo_id": alert["slo_id"],
			"burn_rate": round(burn_rate, 4),
			"threshold": alert["burn_rate_threshold"],
			"firing": fires,
			"error_budget_remaining": error_budget_remaining,
			"evaluated_at": _now(),
		}

	# ------------------------------------------------------------------ Prometheus export

	async def configure_prometheus_export(
		self,
		endpoint: str = "/metrics",
		port: int = 9090,
		scrape_interval_seconds: int = 15,
		include_namespaces: list[str] | None = None,
		exclude_labels: list[str] | None = None,
	) -> dict[str, Any]:
		config_id = _sid()
		self._prometheus_config = {
			"id": config_id,
			"endpoint": endpoint,
			"port": port,
			"scrape_interval_seconds": scrape_interval_seconds,
			"include_namespaces": include_namespaces or ["apg"],
			"exclude_labels": exclude_labels or [],
			"enabled": True,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._emit("prometheus_config_set", config_id, "prometheus_config")
		return deepcopy(self._prometheus_config)

	async def get_prometheus_config(self) -> dict[str, Any] | None:
		return deepcopy(self._prometheus_config) if self._prometheus_config else None

	async def render_prometheus_metrics(self) -> str:
		"""Render current data points in Prometheus text exposition format."""
		lines: list[str] = []
		# Group data points by metric name
		grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
		for p in self._data_points:
			grouped[p["metric_name"]].append(p)

		for metric_name, points in grouped.items():
			defn = next((d for d in self._metric_definitions.values() if d["name"] == metric_name), None)
			mtype = defn["metric_type"] if defn else "untyped"
			if defn and defn.get("description"):
				lines.append(f"# HELP {metric_name} {defn['description']}")
			lines.append(f"# TYPE {metric_name} {mtype}")
			for point in points[-100:]:  # last 100 per metric
				labels = point.get("labels", {})
				label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
				if label_str:
					lines.append(f"{metric_name}{{{label_str}}} {point['value']}")
				else:
					lines.append(f"{metric_name} {point['value']}")

		return "\n".join(lines) + "\n"

	# ------------------------------------------------------------------ dashboards

	async def create_dashboard(
		self,
		name: str,
		description: str = "",
		service_name: str | None = None,
		panels: list[dict[str, Any]] | None = None,
		variables: list[dict[str, Any]] | None = None,
		refresh_interval_seconds: int = 30,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		dash_id = _sid()
		record: dict[str, Any] = {
			"id": dash_id,
			"name": name,
			"description": description,
			"service_name": service_name,
			"panels": panels or [],
			"variables": variables or [],
			"refresh_interval_seconds": refresh_interval_seconds,
			"tags": tags or [],
			"tenant_id": self.tenant_id,
			"created_at": _now(),
			"updated_at": None,
		}
		self._dashboards[dash_id] = record
		self._emit("dashboard_created", dash_id, "dashboard", {"name": name})
		return deepcopy(record)

	async def update_dashboard(
		self,
		dash_id: str,
		name: str | None = None,
		description: str | None = None,
		panels: list[dict[str, Any]] | None = None,
		variables: list[dict[str, Any]] | None = None,
		refresh_interval_seconds: int | None = None,
		tags: list[str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(dash_id, "dash_id")
		if dash_id not in self._dashboards:
			raise KeyError(f"Dashboard not found: {dash_id}")
		dash = self._dashboards[dash_id]
		if name is not None:
			dash["name"] = name
		if description is not None:
			dash["description"] = description
		if panels is not None:
			dash["panels"] = panels
		if variables is not None:
			dash["variables"] = variables
		if refresh_interval_seconds is not None:
			dash["refresh_interval_seconds"] = refresh_interval_seconds
		if tags is not None:
			dash["tags"] = tags
		dash["updated_at"] = _now()
		self._emit("dashboard_updated", dash_id, "dashboard")
		return deepcopy(dash)

	async def get_dashboard(self, dash_id: str) -> dict[str, Any]:
		guard_non_empty_string(dash_id, "dash_id")
		if dash_id not in self._dashboards:
			raise KeyError(f"Dashboard not found: {dash_id}")
		return deepcopy(self._dashboards[dash_id])

	async def list_dashboards(self, service_name: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
		dashes = list(self._dashboards.values())
		if service_name:
			dashes = [d for d in dashes if d["service_name"] == service_name]
		total = len(dashes)
		offset = (page - 1) * page_size
		return {"items": [deepcopy(d) for d in dashes[offset: offset + page_size]], "total": total, "page": page, "page_size": page_size}

	async def delete_dashboard(self, dash_id: str) -> dict[str, Any]:
		guard_non_empty_string(dash_id, "dash_id")
		if dash_id not in self._dashboards:
			raise KeyError(f"Dashboard not found: {dash_id}")
		del self._dashboards[dash_id]
		self._emit("dashboard_deleted", dash_id, "dashboard")
		return {"deleted": True, "dash_id": dash_id}

	async def generate_red_dashboard(self, service_name: str) -> dict[str, Any]:
		"""Auto-generate a RED metrics dashboard for a service."""
		guard_non_empty_string(service_name, "service_name")
		panels = [
			{
				"title": "Request Rate",
				"type": "timeseries",
				"query": f'rate({service_name}_requests_total[5m])',
				"unit": "req/s",
				"position": {"x": 0, "y": 0, "w": 8, "h": 4},
			},
			{
				"title": "Error Rate",
				"type": "timeseries",
				"query": f'rate({service_name}_errors_total[5m]) / rate({service_name}_requests_total[5m])',
				"unit": "%",
				"position": {"x": 8, "y": 0, "w": 8, "h": 4},
			},
			{
				"title": "Request Duration P99",
				"type": "timeseries",
				"query": f'histogram_quantile(0.99, rate({service_name}_duration_ms_bucket[5m]))',
				"unit": "ms",
				"position": {"x": 16, "y": 0, "w": 8, "h": 4},
			},
		]
		return await self.create_dashboard(
			name=f"{service_name} RED Metrics",
			description=f"Auto-generated RED dashboard for {service_name}",
			service_name=service_name,
			panels=panels,
			tags=["auto-generated", "red-metrics"],
		)
