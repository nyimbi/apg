# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""ObservabilityService — unified facade over obs_trc, obs_mtx, obs_log subcapabilities.

Provides a single entry point for:
  - Distributed trace span recording (delegating to trc.DistributedTracingService)
  - RED metric ingestion (delegating to mtx.MetricsSLOService)
  - Structured log events (delegating to log.LogAggregationService)
  - Composite health checks
  - SLO status queries
  - Multi-tenant operation via explicit tenant_id parameter
"""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import (
	AlertRule,
	HealthStatus,
	LogEntry,
	Metric,
	SLOConfig,
	TraceSpan,
)
from .trc.service import DistributedTracingService
from .mtx.service import MetricsSLOService
from .log.service import LogAggregationService

_log = logging.getLogger(__name__)

CAPABILITY_ID = "obs"


def _now() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ObservabilityService:
	"""Unified observability facade — multi-tenant, all async.

	One instance per process is sufficient; tenant isolation is maintained
	by the underlying subcapability services which key their state on
	tenant_id.

	Usage::

		svc = ObservabilityService()
		span_id = await svc.record_span(span, tenant_id="acme")
		await svc.record_metric(metric, tenant_id="acme")
		health = await svc.get_health_status("api-gateway", tenant_id="acme")
	"""

	def __init__(self) -> None:
		# Lazy per-tenant subcapability service pools
		self._trc: dict[str, DistributedTracingService] = {}
		self._mtx: dict[str, MetricsSLOService] = {}
		self._log: dict[str, LogAggregationService] = {}
		# In-process SLO / alert rule registry (keyed by tenant → id)
		self._slos: dict[str, dict[str, dict[str, Any]]] = {}
		self._alerts: dict[str, dict[str, dict[str, Any]]] = {}
		_log.info("ObservabilityService initialised (capability=%s)", CAPABILITY_ID)

	# ------------------------------------------------------------------ helpers

	def _trc_svc(self, tenant_id: str) -> DistributedTracingService:
		if tenant_id not in self._trc:
			self._trc[tenant_id] = DistributedTracingService(tenant_id=tenant_id)
		return self._trc[tenant_id]

	def _mtx_svc(self, tenant_id: str) -> MetricsSLOService:
		if tenant_id not in self._mtx:
			self._mtx[tenant_id] = MetricsSLOService(tenant_id=tenant_id)
		return self._mtx[tenant_id]

	def _log_svc(self, tenant_id: str) -> LogAggregationService:
		if tenant_id not in self._log:
			self._log[tenant_id] = LogAggregationService(tenant_id=tenant_id)
		return self._log[tenant_id]

	# ------------------------------------------------------------------ spans

	async def record_span(self, span: TraceSpan, tenant_id: str = "default") -> str:
		"""Record a trace span and return its assigned span ID.

		The span is forwarded to the distributed tracing subcapability.
		If span.id is already set it is used as a hint; the trc service
		generates a deterministic hex span ID internally.

		Returns:
			str: The span ID assigned by the tracing service.
		"""
		svc = self._trc_svc(tenant_id)
		result = await svc.create_span(
			operation_name=span.operation_name,
			service_name=span.service_name,
			trace_id=span.trace_id,
			parent_span_id=span.parent_span_id,
			start_time=span.start_time.isoformat() if span.start_time else None,
			tags=span.tags,
			baggage=span.baggage,
			kind=span.kind,
			sampled=span.sampled,
		)
		span_id: str = result["id"]

		# If the caller supplied end_time/status finish the span inline
		if span.end_time is not None:
			await svc.finish_span(
				span_id=span_id,
				end_time=span.end_time.isoformat(),
				status=span.status,
				status_message=span.status_message,
				error=span.error,
			)

		_log.debug("record_span span_id=%s trace_id=%s tenant=%s", span_id, span.trace_id, tenant_id)
		return span_id

	# ------------------------------------------------------------------ metrics

	async def record_metric(self, metric: Metric, tenant_id: str = "default") -> None:
		"""Ingest a single metric data point into the metrics subcapability.

		counter and gauge metrics are recorded as raw time-series samples;
		histogram and summary types are recorded as individual observations.
		"""
		svc = self._mtx_svc(tenant_id)
		await svc.record_metric(
			metric_name=metric.name,
			value=metric.value,
			service_name=metric.service_name,
			labels=metric.labels,
		)
		_log.debug("record_metric name=%s value=%s tenant=%s", metric.name, metric.value, tenant_id)

	# ------------------------------------------------------------------ logs

	async def log_event(self, entry: LogEntry, tenant_id: str = "default") -> None:
		"""Ingest a structured log entry into the log aggregation subcapability.

		Correlation fields (trace_id, span_id, correlation_id) are forwarded
		so logs can be joined with traces in the UI.
		"""
		svc = self._log_svc(tenant_id)
		extra_fields: dict[str, Any] = dict(entry.fields)
		if entry.trace_id:
			extra_fields["trace_id"] = entry.trace_id
		if entry.span_id:
			extra_fields["span_id"] = entry.span_id
		if entry.correlation_id:
			extra_fields["correlation_id"] = entry.correlation_id

		await svc.ingest_log(
			service_name=entry.service_name,
			level=entry.level,
			message=entry.message,
			correlation_id=entry.correlation_id,
			trace_id=entry.trace_id,
			span_id=entry.span_id,
			fields=extra_fields,
			source_file=entry.source_file,
			source_line=entry.source_line,
			logger_name=entry.logger_name or entry.service_name,
		)
		_log.debug("log_event level=%s service=%s tenant=%s", entry.level, entry.service_name, tenant_id)

	# ------------------------------------------------------------------ health

	async def get_health_status(self, service_name: str, tenant_id: str = "default") -> HealthStatus:
		"""Return a composite HealthStatus for the named service.

		Aggregates health signals from all three subcapabilities:
		  - trc: span rate and error rate in last 60 s
		  - mtx: SLO burn rate
		  - log: recent ERROR/CRITICAL log volume

		Overall status is degraded if any subcapability reports anomalies,
		unhealthy if errors are dominant.
		"""
		if not service_name.strip():
			raise ValueError("service_name must not be blank")

		trc_health = await self._trc_svc(tenant_id).health_check()
		mtx_health = await self._mtx_svc(tenant_id).health_check()
		log_health = await self._log_svc(tenant_id).health_check()

		# Derive overall status from subcapability signals.
		# trc health_check returns: traces, spans counts (no error_count key at top level)
		# mtx health_check returns: data_points, slos, burn_rate_alerts counts
		# log health_check returns: entry_count, level_overrides counts
		trc_status: str = trc_health.get("status", "healthy")
		mtx_status: str = mtx_health.get("status", "healthy")
		log_status: str = log_health.get("status", "healthy")

		all_statuses = {trc_status, mtx_status, log_status}
		if "unhealthy" in all_statuses:
			overall = "unhealthy"
		elif "degraded" in all_statuses:
			overall = "degraded"
		else:
			overall = "healthy"

		return HealthStatus(
			service_name=service_name,
			status=overall,
			checks={
				"tracing": trc_health,
				"metrics": mtx_health,
				"logging": log_health,
			},
			message=f"Composite health for {service_name}",
			checked_at=datetime.now(timezone.utc),
			tenant_id=tenant_id,
		)

	# ------------------------------------------------------------------ SLO management

	async def create_slo(self, config: SLOConfig, tenant_id: str = "default") -> dict[str, Any]:
		"""Register an SLO with the metrics subcapability and the local registry.

		SLOConfig.target is stored as a ratio (0-1); the mtx service expects 0-100.
		SLOConfig.window_seconds is converted to window_days (floor division by 86400).
		"""
		svc = self._mtx_svc(tenant_id)
		window_days = max(1, config.window_seconds // 86400)
		result = await svc.create_slo(
			name=config.name,
			service_name=config.service_name,
			slo_type=config.slo_type,
			target=config.target * 100.0,  # ratio → percentage
			window_days=window_days,
			latency_threshold_ms=config.latency_threshold_ms,
			description=config.description or "",
		)
		# Cache in local registry
		self._slos.setdefault(tenant_id, {})[result["id"]] = result
		return result

	async def get_slo_status(self, slo_id: str, tenant_id: str = "default") -> dict[str, Any]:
		"""Return current SLO compliance status including error budget and burn rate.

		Delegates to MetricsSLOService.evaluate_slo_compliance which computes
		current_compliance, error_budget_remaining, and burn_rate from the
		in-memory metric time-series.
		"""
		if not slo_id.strip():
			raise ValueError("slo_id must not be blank")
		svc = self._mtx_svc(tenant_id)
		return await svc.evaluate_slo_compliance(slo_id)

	async def list_slos(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""List all SLOs registered for the tenant."""
		svc = self._mtx_svc(tenant_id)
		return await svc.list_slos()

	# ------------------------------------------------------------------ alert rules

	async def create_alert_rule(self, rule: AlertRule, tenant_id: str = "default") -> dict[str, Any]:
		"""Register a burn-rate alert rule and return the stored record.

		Maps AlertRule fields onto MetricsSLOService.create_burn_rate_alert.
		rule.slo_id is required for burn-rate alerts; rule.threshold is used as
		the burn_rate_threshold.
		"""
		if not rule.slo_id:
			raise ValueError("AlertRule.slo_id is required for burn-rate alerts")
		svc = self._mtx_svc(tenant_id)
		result = await svc.create_burn_rate_alert(
			slo_id=rule.slo_id,
			name=rule.name,
			short_window_minutes=rule.window_seconds // 60,
			burn_rate_threshold=rule.threshold,
			severity=rule.severity,
			notification_channels=rule.notify_channels or None,
		)
		self._alerts.setdefault(tenant_id, {})[result["id"]] = result
		return result

	async def evaluate_alert_rules(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		"""Evaluate all SLOs and return those that are out of compliance."""
		svc = self._mtx_svc(tenant_id)
		return await svc.evaluate_all_slos()

	# ------------------------------------------------------------------ describe / health

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "common",
			"description": (
				"Unified observability facade: OpenTelemetry tracing, RED metrics, "
				"structured logging, SLO management, composite health."
			),
			"subcapabilities": ["obs_trc", "obs_mtx", "obs_log"],
			"multi_tenant": True,
		}

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		trc_h = await self._trc_svc(tenant_id).health_check()
		mtx_h = await self._mtx_svc(tenant_id).health_check()
		log_h = await self._log_svc(tenant_id).health_check()
		return {
			"status": "healthy",
			"capability": CAPABILITY_ID,
			"tenant_id": tenant_id,
			"subcapabilities": {"trc": trc_h, "mtx": mtx_h, "log": log_h},
			"checked_at": _now(),
		}
