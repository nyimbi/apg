# SCRP — World-Class Improvement Plan

**Capability**: Screen Processing / Scraper & Data Harvesting (`scrp`)
**Author**: Nyimbi Odero | © 2025 Datacraft

---

## Overview

The current SCRP implementation provides a solid synchronous foundation for
governed web scraping and data harvesting. The 15 improvements below move it
toward production-grade, async-first, composable data infrastructure.

---

## Improvement 1 — Full Async Service Layer

**Problem**: All service methods are synchronous. Under concurrent load, I/O
waits (network, DB, vault lookups) block the event loop.

**Solution**: Convert the entire service to `async def` methods using
`asyncio`. Introduce `AsyncScraperDataHarvestingService` extending the
synchronous base for backward compatibility. Use `asyncio.gather` for fan-out
operations such as multi-URL crawl scheduling.

**Impact**: 10–50× throughput improvement on I/O-bound harvest workloads.

---

## Improvement 2 — Persistent Storage via AsyncPG

**Problem**: All state is held in Python dicts (`_sources`, `_runs`, etc.).
Process restart loses all data.

**Solution**: Replace in-memory dicts with an `asyncpg`-backed store matching
the existing Alembic migration schema. Provide a `StorageBackend` protocol so
tests can still inject an in-memory implementation.

**Impact**: Production durability, crash recovery, multi-process deployment.

---

## Improvement 3 — Real HTTP Fetching via HTTPX Async Client

**Problem**: `run_scrape` and `javascript_rendered_scrape` return synthetic
records — no real HTTP calls are made.

**Solution**: Integrate `httpx.AsyncClient` with connection pooling, retry
middleware (exponential backoff with jitter), and respect for rate-limit
headers (`Retry-After`, `X-RateLimit-*`). Honour per-domain `rate_limit_rpm`
from the rate-limits store.

**Impact**: Real-world usability; scraping actually fetches pages.

---

## Improvement 4 — Playwright-Backed JS Rendering

**Problem**: `javascript_rendered_scrape` simulates rendering with a fixed
3.5 s synthetic timer; it never launches a browser.

**Solution**: Use `playwright.async_api` with a shared browser context pool.
Support `wait_for_selector`, `wait_for_network_idle`, screenshot capture, and
HAR export. Integrate with the proxy-rotation store so each render uses the
next proxy in the pool.

**Impact**: Accurate rendering of SPAs, React, Angular, and Vue frontends.

---

## Improvement 5 — Real robots.txt Parsing

**Problem**: `robots_respect` makes up disallowed prefixes in code; it never
fetches the actual `robots.txt`.

**Solution**: Fetch `robots.txt` with HTTPX, parse it with the `robotparser`
stdlib module, cache per domain with a configurable TTL (default 24 h), and
honour `Crawl-delay`. Expose a `robots_prefetch` method that warms the cache
for all registered sources.

**Impact**: Legal compliance; avoids accidental crawling of disallowed paths.

---

## Improvement 6 — Distributed Rate Limiting via Redis

**Problem**: Rate-limit state is per-process Python dict. Concurrent workers
share no state, so limits are trivially bypassed.

**Solution**: Replace `_rate_limits` with a Redis sliding-window counter using
`aioredis`. Implement a `TokenBucket` async context manager that blocks until
a token is available. Fall back to in-process bucket when Redis is unavailable.

**Impact**: Correct rate limiting across all worker processes.

---

## Improvement 7 — Screen Capture & OCR Pipeline

**Problem**: The capability description mentions "Screen capture, OCR, UI
automation" but none of these are implemented.

**Solution**: Add `capture_screen`, `ocr_image`, and `ocr_extract_table`
async methods. `capture_screen` wraps Playwright's `page.screenshot()` or
the `mss` library for desktop capture. `ocr_image` sends to a local Tesseract
or Ollama vision model. `ocr_extract_table` post-processes OCR output into
structured rows.

**Impact**: Closes the gap between the stated capability description and
implementation reality.

---

## Improvement 8 — RPA Action Execution

**Problem**: No Robotic Process Automation primitives exist despite RPA being
named in the capability description.

**Solution**: Add `rpa_click`, `rpa_type`, `rpa_navigate`, `rpa_extract`,
and `rpa_workflow_run` async methods backed by Playwright. Workflows are
expressed as a list of `RPAStep` dicts (action, selector, value). Steps are
replayed with retry on stale-element errors.

**Impact**: Enables full UI automation and form filling use cases.

---

## Improvement 9 — Structured LLM Extraction via Ollama

**Problem**: `extract_structured_data` does substring matching on selectors —
it cannot handle unstructured or semi-structured content.

**Solution**: Add `llm_extract` async method that sends raw text/HTML to a
locally-hosted Ollama model (default `mistral-nemo`) with a JSON schema
prompt. Parse and validate the response against the provided Pydantic schema.
Cache extraction results by content hash.

**Impact**: Handles messy real-world content that CSS/XPath selectors cannot.

---

## Improvement 10 — Incremental Cursor State Management

**Problem**: `create_harvest_job` stores `incremental_cursor_field` but there
is no mechanism to read, advance, or reset cursors between runs.

**Solution**: Add `cursor_read`, `cursor_advance`, and `cursor_reset` async
methods. Store cursor state keyed by `(tenant_id, job_id)`. Expose last
cursor value in `run_harvest` response so downstream extractors know where
to continue.

**Impact**: Correct incremental harvesting without re-processing old records.

---

## Improvement 11 — Webhook Delivery for Pipeline Handoffs

**Problem**: `PipelineHandoff` records are created with status `queued` but
nothing ever delivers them to the pipeline target.

**Solution**: Add `handoff_dispatch` and `handoff_retry` async methods. On
success, mark status `delivered`; on failure after `max_retries`, mark
`dead_lettered` and emit an audit event. Support HMAC-signed HTTP POST
webhooks and internal event-bus emission.

**Impact**: End-to-end pipeline connectivity; no silent handoff drops.

---

## Improvement 12 — Content Diff & Change Alerting

**Problem**: `change_detect` only stores/compares a pre-computed hash. It
does not produce a human-readable diff or trigger downstream alerting.

**Solution**: Add `content_diff` async method that fetches two snapshot refs
from storage and returns a unified diff or structured JSON delta. Emit
`source_changed` events to the `intel.alerts` capability when a significant
change is detected (>threshold% of content altered).

**Impact**: Turns passive change detection into an actionable alerting feed.

---

## Improvement 13 — Data Quality Scoring

**Problem**: `extract_structured_data` computes a raw `quality_score` but
there are no thresholds, trending, or per-field breakdown.

**Solution**: Add `quality_report` async method that computes per-field
completeness, type validity, format conformance, and outlier rates. Persist
scores in a time-series store keyed by `(tenant_id, extraction_id, run_date)`.
Expose trend charts via the analytics model.

**Impact**: Proactive data quality governance before records reach the
warehouse.

---

## Improvement 14 — Multi-Tenant Isolation Hardening

**Problem**: `_require_owned` checks `item.tenant_id` but several newer
methods (e.g. `crawler_schedule`, `source_monitor`) use `hasattr` guards
and ad-hoc dicts that bypass the isolation check.

**Solution**: Refactor all ad-hoc `hasattr`-initialised stores into typed
`__init__` attributes. Centralise cross-tenant isolation into a single
`_require_tenant_isolation(item, tenant_id)` guard called in every lookup.
Add property-based tests that assert no method leaks data across tenants.

**Impact**: Eliminates a class of multi-tenant data-leak bugs.

---

## Improvement 15 — Observability: Structured Logging, Metrics & Traces

**Problem**: Events are stored in `_audit_events` (in-memory list) but there
is no integration with external observability platforms.

**Solution**: Add `emit_metrics` async method that ships Prometheus counters
(`scrp_runs_total`, `scrp_records_harvested_total`, `scrp_errors_total`) via
`prometheus_client`. Add OpenTelemetry span context propagation through
`run_harvest` → `complete_harvest_run` → `handoff_dispatch`. Use
`structlog` for JSON-structured log lines with `tenant_id`, `run_id`, and
`trace_id` fields.

**Impact**: Full production observability stack; integrates with Grafana,
Jaeger, and any OpenTelemetry-compatible backend.
