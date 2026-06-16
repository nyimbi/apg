# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025
"""Async tests for ObservabilityService (obs capability).

Covers:
  1. record_span — span created and ID returned
  2. record_metric — metric ingested without error
  3. get_health_status — composite HealthStatus returned with correct shape
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from capabilities.common.obs.models import LogEntry, Metric, TraceSpan
from capabilities.common.obs.service import ObservabilityService


# ---------------------------------------------------------------------------
# Test 1: record_span
# ---------------------------------------------------------------------------

async def test_record_span_returns_span_id():
	"""record_span should create a span in the tracing subcapability and return a non-empty ID."""
	svc = ObservabilityService()
	span = TraceSpan(
		trace_id="aabbccddeeff00112233445566778899",
		operation_name="handle_request",
		service_name="api-gateway",
		start_time=datetime.now(timezone.utc),
	)

	span_id = await svc.record_span(span, tenant_id="test-tenant")

	assert isinstance(span_id, str), "span_id must be a string"
	assert len(span_id) > 0, "span_id must not be empty"

	# Verify the span is persisted in the trc subcapability
	trc = svc._trc_svc("test-tenant")
	stored = await trc.get_span(span_id)
	assert stored["operation_name"] == "handle_request"
	assert stored["service_name"] == "api-gateway"
	assert stored["trace_id"] == "aabbccddeeff00112233445566778899"


# ---------------------------------------------------------------------------
# Test 2: record_metric
# ---------------------------------------------------------------------------

async def test_record_metric_ingests_data_point():
	"""record_metric should ingest a data point into the mtx subcapability without raising."""
	svc = ObservabilityService()
	metric = Metric(
		name="http_requests_total",
		metric_type="counter",
		value=42.0,
		service_name="api-gateway",
		labels={"method": "GET", "status": "200"},
	)

	# Should not raise
	await svc.record_metric(metric, tenant_id="test-tenant")

	# Verify the data point is stored
	mtx = svc._mtx_svc("test-tenant")
	result = await mtx.query_metric(
		metric_name="http_requests_total",
		service_name="api-gateway",
	)
	# query_metric returns a list of data point dicts
	assert isinstance(result, list), "query_metric must return a list"
	assert len(result) >= 1, "At least one data point should be stored"
	values = [p["value"] for p in result]
	assert 42.0 in values, "The ingested value must appear in the stored points"


# ---------------------------------------------------------------------------
# Test 3: get_health_status
# ---------------------------------------------------------------------------

async def test_get_health_status_returns_valid_health_status():
	"""get_health_status should return a HealthStatus with the expected shape."""
	from capabilities.common.obs.models import HealthStatus

	svc = ObservabilityService()

	status = await svc.get_health_status("payment-service", tenant_id="test-tenant")

	assert isinstance(status, HealthStatus), "Must return a HealthStatus instance"
	assert status.service_name == "payment-service"
	assert status.status in {"healthy", "degraded", "unhealthy", "unknown"}, \
		f"Unexpected status value: {status.status!r}"
	assert "tracing" in status.checks, "checks must include 'tracing' key"
	assert "metrics" in status.checks, "checks must include 'metrics' key"
	assert "logging" in status.checks, "checks must include 'logging' key"
	assert status.tenant_id == "test-tenant"
