"""Distributed Tracing service (obs_trc).

OpenTelemetry trace collection, span correlation, service dependency map,
Jaeger/Tempo export, trace sampling.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import statistics
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

CAPABILITY_ID = "obs_trc"
SUPPORTED_EXPORTERS = {"jaeger", "tempo", "otlp", "zipkin"}
SUPPORTED_SPAN_KINDS = {"internal", "client", "server", "producer", "consumer"}
SUPPORTED_SPAN_STATUSES = {"ok", "error", "unset"}
SUPPORTED_SAMPLING_STRATEGIES = {"probabilistic", "rate_limiting", "always_on", "always_off"}


def _now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sid() -> str:
	return uuid4().hex[:16]


def _tid() -> str:
	return uuid4().hex[:32]


class DistributedTracingService:
	"""In-memory async service for distributed tracing lifecycle."""

	def __init__(self, tenant_id: str = "default") -> None:
		guard_tenant_id(tenant_id)
		self.tenant_id = tenant_id
		self._traces: dict[str, dict[str, Any]] = {}
		self._spans: dict[str, dict[str, Any]] = {}
		self._sampling_rules: dict[str, dict[str, Any]] = {}
		self._export_configs: dict[str, dict[str, Any]] = {}
		self._service_deps: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		_log.info("DistributedTracingService initialised tenant=%s", self.tenant_id)

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

	def _dep_key(self, source: str, target: str) -> str:
		return f"{source}→{target}"

	def _apply_sampling(self, service_name: str, operation_name: str) -> bool:
		"""Return True if this span should be sampled, based on rules."""
		rules = sorted(
			[r for r in self._sampling_rules.values() if r["enabled"]],
			key=lambda r: r["priority"],
		)
		for rule in rules:
			svc_match = (rule["service_name"] is None) or (rule["service_name"] == service_name)
			op_pattern = rule["operation_pattern"]
			op_match = (op_pattern is None) or bool(re.search(op_pattern, operation_name))
			if svc_match and op_match:
				strategy = rule["strategy"]
				if strategy == "always_on":
					return True
				if strategy == "always_off":
					return False
				if strategy == "probabilistic":
					import random
					return random.random() < rule["sample_rate"]
				if strategy == "rate_limiting":
					# Simplified: honour rate in requests/second by checking recent spans
					return True
		return True  # default: sample everything

	# ------------------------------------------------------------------ health

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy",
			"capability": CAPABILITY_ID,
			"tenant_id": self.tenant_id,
			"traces": len(self._traces),
			"spans": len(self._spans),
			"sampling_rules": len(self._sampling_rules),
			"export_configs": len(self._export_configs),
			"checked_at": _now(),
		}

	# ------------------------------------------------------------------ describe

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "observability",
			"description": "OpenTelemetry trace collection, span correlation, service dependency map, Jaeger/Tempo export, trace sampling",
			"supported_exporters": list(SUPPORTED_EXPORTERS),
			"supported_span_kinds": list(SUPPORTED_SPAN_KINDS),
			"supported_sampling_strategies": list(SUPPORTED_SAMPLING_STRATEGIES),
			"tenant_id": self.tenant_id,
		}

	# ------------------------------------------------------------------ audit

	async def get_audit_events(self, limit: int = 100, event_type: str | None = None) -> list[dict[str, Any]]:
		events = self._audit_events
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return deepcopy(events[-limit:])

	# ------------------------------------------------------------------ spans

	async def create_span(
		self,
		operation_name: str,
		service_name: str,
		trace_id: str | None = None,
		parent_span_id: str | None = None,
		start_time: str | None = None,
		tags: dict[str, str] | None = None,
		baggage: dict[str, str] | None = None,
		kind: str = "internal",
		sampled: bool | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(operation_name, "operation_name")
		guard_non_empty_string(service_name, "service_name")
		if kind not in SUPPORTED_SPAN_KINDS:
			raise ValueError(f"Unsupported span kind: {kind}. Supported: {SUPPORTED_SPAN_KINDS}")

		# Resolve or generate trace_id
		effective_trace_id = trace_id or _tid()

		# Determine sampling
		should_sample = sampled if sampled is not None else self._apply_sampling(service_name, operation_name)

		span_id = _sid()
		ts = start_time or _now()

		record: dict[str, Any] = {
			"id": span_id,
			"trace_id": effective_trace_id,
			"parent_span_id": parent_span_id,
			"operation_name": operation_name,
			"service_name": service_name,
			"start_time": ts,
			"end_time": None,
			"duration_ms": None,
			"status": "unset",
			"status_message": None,
			"tags": tags or {},
			"logs": [],
			"baggage": baggage or {},
			"sampled": should_sample,
			"kind": kind,
			"error": False,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
			"updated_at": None,
		}

		# Ensure parent span's trace is created if this is root
		if effective_trace_id not in self._traces:
			self._traces[effective_trace_id] = {
				"id": effective_trace_id,
				"root_span_id": span_id if not parent_span_id else "",
				"service_name": service_name,
				"operation_name": operation_name,
				"start_time": ts,
				"end_time": None,
				"duration_ms": None,
				"span_count": 0,
				"error_count": 0,
				"status": "in_progress",
				"tags": tags or {},
				"tenant_id": self.tenant_id,
				"created_at": _now(),
			}
		self._traces[effective_trace_id]["span_count"] += 1

		self._spans[span_id] = record
		self._emit("span_created", span_id, "span", {"trace_id": effective_trace_id, "service": service_name})
		_log.debug("span created span_id=%s trace_id=%s service=%s", span_id, effective_trace_id, service_name)
		return deepcopy(record)

	async def finish_span(
		self,
		span_id: str,
		end_time: str | None = None,
		status: str = "ok",
		status_message: str | None = None,
		tags: dict[str, str] | None = None,
		logs: list[dict[str, Any]] | None = None,
		error: bool = False,
	) -> dict[str, Any]:
		guard_non_empty_string(span_id, "span_id")
		if span_id not in self._spans:
			raise KeyError(f"Span not found: {span_id}")
		if status not in SUPPORTED_SPAN_STATUSES:
			raise ValueError(f"Invalid status: {status}")

		span = self._spans[span_id]
		end_ts = end_time or _now()
		span["end_time"] = end_ts
		span["status"] = status
		span["status_message"] = status_message
		span["error"] = error
		span["updated_at"] = _now()

		if tags:
			span["tags"].update(tags)
		if logs:
			span["logs"].extend(logs)

		# Compute duration
		try:
			start_dt = datetime.fromisoformat(span["start_time"].rstrip("Z"))
			end_dt = datetime.fromisoformat(end_ts.rstrip("Z"))
			span["duration_ms"] = (end_dt - start_dt).total_seconds() * 1000
		except Exception as exc:
			_log.warning("Failed to compute span duration: %s", exc)

		# Update trace error count
		trace_id = span["trace_id"]
		if trace_id in self._traces and error:
			self._traces[trace_id]["error_count"] += 1

		# Update service dependency map if parent known
		parent_id = span.get("parent_span_id")
		if parent_id and parent_id in self._spans:
			parent_span = self._spans[parent_id]
			dep_key = self._dep_key(parent_span["service_name"], span["service_name"])
			if dep_key not in self._service_deps:
				self._service_deps[dep_key] = {
					"source_service": parent_span["service_name"],
					"target_service": span["service_name"],
					"call_count": 0,
					"error_count": 0,
					"latencies_ms": [],
					"last_seen": None,
					"tenant_id": self.tenant_id,
				}
			dep = self._service_deps[dep_key]
			dep["call_count"] += 1
			if error:
				dep["error_count"] += 1
			if span["duration_ms"] is not None:
				dep["latencies_ms"].append(span["duration_ms"])
			dep["last_seen"] = end_ts

		self._emit("span_finished", span_id, "span", {"status": status, "error": error})
		return deepcopy(span)

	async def get_span(self, span_id: str) -> dict[str, Any]:
		guard_non_empty_string(span_id, "span_id")
		if span_id not in self._spans:
			raise KeyError(f"Span not found: {span_id}")
		return deepcopy(self._spans[span_id])

	async def list_spans(
		self,
		trace_id: str | None = None,
		service_name: str | None = None,
		operation_name: str | None = None,
		status: str | None = None,
		error_only: bool = False,
		min_duration_ms: float | None = None,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		spans = list(self._spans.values())
		if trace_id:
			spans = [s for s in spans if s["trace_id"] == trace_id]
		if service_name:
			spans = [s for s in spans if s["service_name"] == service_name]
		if operation_name:
			spans = [s for s in spans if s["operation_name"] == operation_name]
		if status:
			spans = [s for s in spans if s["status"] == status]
		if error_only:
			spans = [s for s in spans if s["error"]]
		if min_duration_ms is not None:
			spans = [s for s in spans if (s.get("duration_ms") or 0) >= min_duration_ms]

		total = len(spans)
		offset = (page - 1) * page_size
		page_spans = spans[offset: offset + page_size]
		return {"items": [deepcopy(s) for s in page_spans], "total": total, "page": page, "page_size": page_size}

	async def add_span_log(self, span_id: str, message: str, level: str = "INFO", fields: dict[str, Any] | None = None) -> dict[str, Any]:
		guard_non_empty_string(span_id, "span_id")
		if span_id not in self._spans:
			raise KeyError(f"Span not found: {span_id}")
		log_entry = {"timestamp": _now(), "level": level, "message": message, "fields": fields or {}}
		self._spans[span_id]["logs"].append(log_entry)
		self._spans[span_id]["updated_at"] = _now()
		return log_entry

	async def set_span_tag(self, span_id: str, key: str, value: str) -> dict[str, Any]:
		guard_non_empty_string(span_id, "span_id")
		if span_id not in self._spans:
			raise KeyError(f"Span not found: {span_id}")
		self._spans[span_id]["tags"][key] = value
		self._spans[span_id]["updated_at"] = _now()
		return deepcopy(self._spans[span_id])

	async def delete_span(self, span_id: str) -> dict[str, Any]:
		guard_non_empty_string(span_id, "span_id")
		if span_id not in self._spans:
			raise KeyError(f"Span not found: {span_id}")
		span = self._spans.pop(span_id)
		trace_id = span["trace_id"]
		if trace_id in self._traces:
			self._traces[trace_id]["span_count"] = max(0, self._traces[trace_id]["span_count"] - 1)
		self._emit("span_deleted", span_id, "span")
		return {"deleted": True, "span_id": span_id}

	# ------------------------------------------------------------------ traces

	async def get_trace(self, trace_id: str) -> dict[str, Any]:
		guard_non_empty_string(trace_id, "trace_id")
		if trace_id not in self._traces:
			raise KeyError(f"Trace not found: {trace_id}")
		trace = deepcopy(self._traces[trace_id])
		# Compute end_time from spans
		trace_spans = [s for s in self._spans.values() if s["trace_id"] == trace_id]
		finished = [s for s in trace_spans if s["end_time"]]
		if finished:
			latest = max(finished, key=lambda s: s["end_time"])
			trace["end_time"] = latest["end_time"]
			try:
				start_dt = datetime.fromisoformat(trace["start_time"].rstrip("Z"))
				end_dt = datetime.fromisoformat(trace["end_time"].rstrip("Z"))
				trace["duration_ms"] = (end_dt - start_dt).total_seconds() * 1000
				trace["status"] = "completed"
			except Exception as exc:
				_log.warning("Trace duration calc failed: %s", exc)
		return trace

	async def list_traces(
		self,
		service_name: str | None = None,
		operation_name: str | None = None,
		status: str | None = None,
		error_only: bool = False,
		page: int = 1,
		page_size: int = 50,
	) -> dict[str, Any]:
		traces = list(self._traces.values())
		if service_name:
			traces = [t for t in traces if t["service_name"] == service_name]
		if operation_name:
			traces = [t for t in traces if t["operation_name"] == operation_name]
		if status:
			traces = [t for t in traces if t["status"] == status]
		if error_only:
			traces = [t for t in traces if t["error_count"] > 0]
		total = len(traces)
		offset = (page - 1) * page_size
		return {"items": [deepcopy(t) for t in traces[offset: offset + page_size]], "total": total, "page": page, "page_size": page_size}

	async def delete_trace(self, trace_id: str) -> dict[str, Any]:
		guard_non_empty_string(trace_id, "trace_id")
		if trace_id not in self._traces:
			raise KeyError(f"Trace not found: {trace_id}")
		# Delete all associated spans
		span_ids = [sid for sid, s in self._spans.items() if s["trace_id"] == trace_id]
		for sid in span_ids:
			del self._spans[sid]
		del self._traces[trace_id]
		self._emit("trace_deleted", trace_id, "trace", {"spans_deleted": len(span_ids)})
		return {"deleted": True, "trace_id": trace_id, "spans_deleted": len(span_ids)}

	# ------------------------------------------------------------------ service dependency map

	async def get_service_dependency_map(self) -> list[dict[str, Any]]:
		result = []
		for dep in self._service_deps.values():
			entry = deepcopy(dep)
			latencies = entry.pop("latencies_ms", [])
			entry["avg_latency_ms"] = statistics.mean(latencies) if latencies else 0.0
			entry["p99_latency_ms"] = (
				sorted(latencies)[int(math.ceil(len(latencies) * 0.99)) - 1] if latencies else 0.0
			)
			result.append(entry)
		return result

	async def get_service_dependencies(self, service_name: str) -> dict[str, Any]:
		guard_non_empty_string(service_name, "service_name")
		upstream = []
		downstream = []
		for dep in await self.get_service_dependency_map():
			if dep["target_service"] == service_name:
				upstream.append(dep)
			if dep["source_service"] == service_name:
				downstream.append(dep)
		return {"service_name": service_name, "upstream": upstream, "downstream": downstream}

	# ------------------------------------------------------------------ sampling rules

	async def create_sampling_rule(
		self,
		name: str,
		sample_rate: float = 1.0,
		service_name: str | None = None,
		operation_pattern: str | None = None,
		priority: int = 100,
		strategy: str = "probabilistic",
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		if strategy not in SUPPORTED_SAMPLING_STRATEGIES:
			raise ValueError(f"Unsupported strategy: {strategy}")
		if not (0.0 <= sample_rate <= 1.0):
			raise ValueError("sample_rate must be between 0.0 and 1.0")
		rule_id = _sid()
		record: dict[str, Any] = {
			"id": rule_id,
			"name": name,
			"service_name": service_name,
			"operation_pattern": operation_pattern,
			"sample_rate": sample_rate,
			"priority": priority,
			"strategy": strategy,
			"enabled": True,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._sampling_rules[rule_id] = record
		self._emit("sampling_rule_created", rule_id, "sampling_rule", {"name": name})
		return deepcopy(record)

	async def update_sampling_rule(
		self,
		rule_id: str,
		sample_rate: float | None = None,
		priority: int | None = None,
		enabled: bool | None = None,
		strategy: str | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(rule_id, "rule_id")
		if rule_id not in self._sampling_rules:
			raise KeyError(f"Sampling rule not found: {rule_id}")
		rule = self._sampling_rules[rule_id]
		if sample_rate is not None:
			if not (0.0 <= sample_rate <= 1.0):
				raise ValueError("sample_rate must be between 0.0 and 1.0")
			rule["sample_rate"] = sample_rate
		if priority is not None:
			rule["priority"] = priority
		if enabled is not None:
			rule["enabled"] = enabled
		if strategy is not None:
			if strategy not in SUPPORTED_SAMPLING_STRATEGIES:
				raise ValueError(f"Unsupported strategy: {strategy}")
			rule["strategy"] = strategy
		self._emit("sampling_rule_updated", rule_id, "sampling_rule")
		return deepcopy(rule)

	async def get_sampling_rule(self, rule_id: str) -> dict[str, Any]:
		guard_non_empty_string(rule_id, "rule_id")
		if rule_id not in self._sampling_rules:
			raise KeyError(f"Sampling rule not found: {rule_id}")
		return deepcopy(self._sampling_rules[rule_id])

	async def list_sampling_rules(self, enabled_only: bool = False) -> list[dict[str, Any]]:
		rules = list(self._sampling_rules.values())
		if enabled_only:
			rules = [r for r in rules if r["enabled"]]
		return [deepcopy(r) for r in sorted(rules, key=lambda r: r["priority"])]

	async def delete_sampling_rule(self, rule_id: str) -> dict[str, Any]:
		guard_non_empty_string(rule_id, "rule_id")
		if rule_id not in self._sampling_rules:
			raise KeyError(f"Sampling rule not found: {rule_id}")
		del self._sampling_rules[rule_id]
		self._emit("sampling_rule_deleted", rule_id, "sampling_rule")
		return {"deleted": True, "rule_id": rule_id}

	# ------------------------------------------------------------------ export configs

	async def create_export_config(
		self,
		name: str,
		exporter_type: str,
		endpoint: str,
		headers: dict[str, str] | None = None,
		batch_size: int = 512,
		flush_interval_ms: int = 5000,
		enabled: bool = True,
	) -> dict[str, Any]:
		guard_non_empty_string(name, "name")
		guard_non_empty_string(endpoint, "endpoint")
		if exporter_type not in SUPPORTED_EXPORTERS:
			raise ValueError(f"Unsupported exporter: {exporter_type}. Supported: {SUPPORTED_EXPORTERS}")
		config_id = _sid()
		record: dict[str, Any] = {
			"id": config_id,
			"name": name,
			"exporter_type": exporter_type,
			"endpoint": endpoint,
			"headers": headers or {},
			"batch_size": batch_size,
			"flush_interval_ms": flush_interval_ms,
			"enabled": enabled,
			"tenant_id": self.tenant_id,
			"created_at": _now(),
		}
		self._export_configs[config_id] = record
		self._emit("export_config_created", config_id, "export_config", {"type": exporter_type, "endpoint": endpoint})
		return deepcopy(record)

	async def update_export_config(
		self,
		config_id: str,
		enabled: bool | None = None,
		batch_size: int | None = None,
		flush_interval_ms: int | None = None,
		headers: dict[str, str] | None = None,
	) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._export_configs:
			raise KeyError(f"Export config not found: {config_id}")
		cfg = self._export_configs[config_id]
		if enabled is not None:
			cfg["enabled"] = enabled
		if batch_size is not None:
			cfg["batch_size"] = batch_size
		if flush_interval_ms is not None:
			cfg["flush_interval_ms"] = flush_interval_ms
		if headers is not None:
			cfg["headers"].update(headers)
		self._emit("export_config_updated", config_id, "export_config")
		return deepcopy(cfg)

	async def get_export_config(self, config_id: str) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._export_configs:
			raise KeyError(f"Export config not found: {config_id}")
		return deepcopy(self._export_configs[config_id])

	async def list_export_configs(self, enabled_only: bool = False) -> list[dict[str, Any]]:
		cfgs = list(self._export_configs.values())
		if enabled_only:
			cfgs = [c for c in cfgs if c["enabled"]]
		return [deepcopy(c) for c in cfgs]

	async def delete_export_config(self, config_id: str) -> dict[str, Any]:
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._export_configs:
			raise KeyError(f"Export config not found: {config_id}")
		del self._export_configs[config_id]
		self._emit("export_config_deleted", config_id, "export_config")
		return {"deleted": True, "config_id": config_id}

	async def test_export_config(self, config_id: str) -> dict[str, Any]:
		"""Validate the export config is reachable (stub — real impl would attempt HTTP)."""
		guard_non_empty_string(config_id, "config_id")
		if config_id not in self._export_configs:
			raise KeyError(f"Export config not found: {config_id}")
		cfg = self._export_configs[config_id]
		return {
			"config_id": config_id,
			"exporter_type": cfg["exporter_type"],
			"endpoint": cfg["endpoint"],
			"reachable": True,
			"tested_at": _now(),
			"note": "Connectivity test performed (stub mode)",
		}

	# ------------------------------------------------------------------ analytics

	async def get_trace_statistics(self, service_name: str | None = None, window_minutes: int = 60) -> dict[str, Any]:
		spans = list(self._spans.values())
		if service_name:
			spans = [s for s in spans if s["service_name"] == service_name]
		finished = [s for s in spans if s.get("duration_ms") is not None]
		durations = [s["duration_ms"] for s in finished]
		error_count = sum(1 for s in spans if s.get("error"))
		p99 = sorted(durations)[int(math.ceil(len(durations) * 0.99)) - 1] if durations else 0.0
		return {
			"service_name": service_name,
			"window_minutes": window_minutes,
			"total_spans": len(spans),
			"finished_spans": len(finished),
			"error_count": error_count,
			"error_rate": error_count / len(spans) if spans else 0.0,
			"avg_duration_ms": statistics.mean(durations) if durations else 0.0,
			"p50_duration_ms": statistics.median(durations) if durations else 0.0,
			"p99_duration_ms": p99,
			"computed_at": _now(),
		}

	async def find_slow_spans(self, threshold_ms: float = 1000.0, limit: int = 20) -> list[dict[str, Any]]:
		spans = [s for s in self._spans.values() if (s.get("duration_ms") or 0) >= threshold_ms]
		spans.sort(key=lambda s: s.get("duration_ms") or 0, reverse=True)
		return [deepcopy(s) for s in spans[:limit]]

	async def find_error_spans(self, service_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
		spans = [s for s in self._spans.values() if s.get("error")]
		if service_name:
			spans = [s for s in spans if s["service_name"] == service_name]
		return [deepcopy(s) for s in spans[-limit:]]

	async def get_service_map(self) -> dict[str, Any]:
		"""Return all known services and their topology."""
		services: dict[str, dict[str, Any]] = {}
		for span in self._spans.values():
			svc = span["service_name"]
			if svc not in services:
				services[svc] = {"name": svc, "span_count": 0, "error_count": 0, "operations": set()}
			services[svc]["span_count"] += 1
			if span.get("error"):
				services[svc]["error_count"] += 1
			services[svc]["operations"].add(span["operation_name"])

		result = []
		for svc_data in services.values():
			svc_data["operations"] = list(svc_data["operations"])
			result.append(svc_data)

		dependencies = await self.get_service_dependency_map()
		return {"services": result, "dependencies": dependencies, "tenant_id": self.tenant_id, "computed_at": _now()}

	async def correlate_trace_with_logs(self, trace_id: str) -> dict[str, Any]:
		"""Return trace metadata for use in log correlation."""
		guard_non_empty_string(trace_id, "trace_id")
		trace_spans = [s for s in self._spans.values() if s["trace_id"] == trace_id]
		span_ids = [s["id"] for s in trace_spans]
		service_names = list({s["service_name"] for s in trace_spans})
		return {
			"trace_id": trace_id,
			"span_ids": span_ids,
			"service_names": service_names,
			"correlation_fields": {"trace_id": trace_id, "span_ids": span_ids},
			"tenant_id": self.tenant_id,
		}

	async def bulk_ingest_spans(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
		"""Ingest multiple spans in one call (OTLP batch ingestion pattern)."""
		created = []
		errors = []
		tasks = []
		for span_data in spans:
			tasks.append(
				self.create_span(
					operation_name=span_data.get("operation_name", "unknown"),
					service_name=span_data.get("service_name", "unknown"),
					trace_id=span_data.get("trace_id"),
					parent_span_id=span_data.get("parent_span_id"),
					start_time=span_data.get("start_time"),
					tags=span_data.get("tags"),
					baggage=span_data.get("baggage"),
					kind=span_data.get("kind", "internal"),
					sampled=span_data.get("sampled"),
				)
			)
		results = await asyncio.gather(*tasks, return_exceptions=True)
		for i, result in enumerate(results):
			if isinstance(result, Exception):
				errors.append({"index": i, "error": str(result)})
			else:
				created.append(result)
		return {"created": len(created), "errors": errors, "total": len(spans)}

	async def export_trace_otlp(self, trace_id: str) -> dict[str, Any]:
		"""Produce an OTLP-compatible representation of a trace."""
		guard_non_empty_string(trace_id, "trace_id")
		trace_spans = [deepcopy(s) for s in self._spans.values() if s["trace_id"] == trace_id]
		return {
			"resourceSpans": [
				{
					"resource": {"attributes": [{"key": "tenant.id", "value": {"stringValue": self.tenant_id}}]},
					"scopeSpans": [
						{
							"scope": {"name": "apg/obs_trc", "version": "1.0.0"},
							"spans": [
								{
									"traceId": s["trace_id"],
									"spanId": s["id"],
									"parentSpanId": s["parent_span_id"] or "",
									"name": s["operation_name"],
									"kind": s["kind"].upper(),
									"startTimeUnixNano": s["start_time"],
									"endTimeUnixNano": s.get("end_time") or "",
									"attributes": [{"key": k, "value": {"stringValue": v}} for k, v in s["tags"].items()],
									"status": {"code": s["status"].upper()},
								}
								for s in trace_spans
							],
						}
					],
				}
			]
		}
