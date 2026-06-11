# ACCS - World Class Improvements

Fifteen targeted improvements that would elevate ACCS from a solid in-memory
governance tracker to a best-in-class enterprise accessibility platform.

---

### I1. Real-Time WCAG Axe-Core Integration via Playwright

**Category**: Audit Engine
**Justification**: Current audit engine uses deterministic heuristics against stored target metadata. A real headless browser integration (Playwright + axe-core) would catch actual DOM violations — the only ground truth for WCAG conformance. This is what Deque WorldSpace and SiteImprove do. Without it, the audit results are approximations.
**Implementation**: Add an optional `AxePlaywrightAdapter` that spawns a headless Chromium session, injects axe-core via `page.evaluate()`, maps the axe violations JSON to `AccessibilityFinding` records, and stores them through the normal `record_finding` pipeline. Keep the in-memory engine as the offline/test path. Gate via `ACCS_HEADLESS_AUDIT=true` env var.
**Competitor**: Deque WorldSpace Attest, SiteImprove Accessibility

---

### I2. WCAG 2.2 / ARIA 1.3 Rule Set as Structured Catalog

**Category**: Rule Engine
**Justification**: The current engine produces generic finding descriptions. Mapping every check to the canonical WCAG 2.2 success criterion ID (e.g. `1.4.3`, `2.1.1`) and the corresponding ARIA technique enables automated reporting to regulators, EU Web Accessibility Directive dashboards, and US Section 508 tools.
**Implementation**: Replace the freeform `rule` string with a `WcagCriterion` dataclass carrying `criterion_id`, `level`, `title`, `technique_ids`, and `test_procedure`. Validate criterion IDs against a bundled JSON catalog (`data/wcag_2_2_criteria.json`). Surface criterion metadata in finding dicts and the compliance_summary.
**Competitor**: Level Access AMP, Siteimprove, WAVE

---

### I3. Async Batch Audit Pipeline with Asyncio Task Groups

**Category**: Performance
**Justification**: `run_audit()` is synchronous and sequential across targets. For tenants with dozens of routes, this is a serial bottleneck. Concurrent auditing would reduce total wall time proportionally. Python 3.11+ `asyncio.TaskGroup` makes this clean and cancellation-safe.
**Implementation**: Add `async def run_audit_batch(self, audit_ids, tenant_id, standard_id, target_ids_per_audit)` using `async with asyncio.TaskGroup() as tg` to fire one coroutine per target, collect results, and merge findings. Propagate the first exception that escapes the group.
**Competitor**: Evinced, Deque attest CI pipeline

---

### I4. Finding Deduplication with Content-Addressed IDs

**Category**: Data Integrity
**Justification**: Repeated audits on the same target produce duplicate findings for the same rule. Currently callers get `ValueError: duplicate`. A deterministic SHA-256 of `(tenant_id, target_id, rule, severity)` as the finding ID, plus an upsert that bumps a `recurrence_count` field, gives idempotent re-audits and accurate trend analysis.
**Implementation**: Add `_deterministic_finding_id(tenant_id, target_id, rule, severity) -> str` computing `sha256("|".join([...]))[:16]`. Change `record_finding` to upsert: if key exists, increment `recurrence_count` and update `last_seen`. Add `recurrence_count: int = 1` and `last_seen: str` to `AccessibilityFinding`.
**Competitor**: Jira Accessibility Plugin, GitHub Accessibility Issues

---

### I5. Remediation SLA Enforcement with Decimal-Based Priority Scoring

**Category**: Remediation Workflow
**Justification**: Open remediation tasks have no urgency signal beyond severity string. A Decimal-based SLA score combining `severity_weight * age_days / sla_days` gives a sortable, unambiguous priority queue. SLA days per severity tier (critical: 3, high: 14, medium: 30, low: 90) enforce the ADA and EN 301 549 expectation of timely correction.
**Implementation**: Add `async def score_remediation_queue(self, tenant_id: str) -> list[dict[str, Any]]` using `Decimal` arithmetic. `sla_days` map keyed on severity. Compute `age = (today - opened_date).days`, `score = Decimal(str(severity_weight)) * Decimal(age) / Decimal(sla_days)`. Sort descending. Include `sla_status: str` — `"ok"`, `"warning"` (>80%), `"breached"`.
**Competitor**: Remedy Force Accessibility, ServiceNow Accessibility Module

---

### I6. Tenant-Isolated Persistent Storage via PostgreSQL Adapter

**Category**: Storage / Multi-Tenancy
**Justification**: The in-memory dict store is reset on every process restart. Enterprise tenants require durable, queryable audit history, finding trends, and exception logs. PostgreSQL with row-level security (RLS) enforces strict tenant isolation at the database layer, not just at the Python layer.
**Implementation**: Add `database/pg_store.py` with a `PgAccsStore` that mirrors the in-memory interface using asyncpg connection pools. Each table (`accs_findings`, `accs_audits`, etc.) has a `tenant_id UUID NOT NULL` column covered by an RLS policy `USING (tenant_id = current_setting('app.tenant_id')::uuid)`. Wire via `AccsService(store=PgAccsStore(...))` injection point.
**Competitor**: Equally Accessible DB, CivicPlus Accessibility Suite

---

### I7. OpenTelemetry Span Instrumentation on Every Audit Step

**Category**: Observability
**Justification**: Accessibility audits touch multiple targets, rules, and findings in one call tree. Without distributed tracing, slow audits and cascading failures are invisible. OTel spans on every `run_audit`, `record_finding`, `close_finding`, and `validate_publication` call enable latency breakdown, error attribution, and SLO dashboards in Grafana/Tempo.
**Implementation**: Wrap service methods with `@contextlib.contextmanager` OTel spans via `opentelemetry-sdk`. Tag spans with `tenant_id`, `audit_id`, `finding_id`, `rule`. Add `trace_id` to every event dict. Gate behind `ACCS_OTEL_ENABLED=true`. Zero-overhead when disabled (noop tracer).
**Competitor**: Deque WorldSpace, Level Access AMP (both instrument audit pipelines)

---

### I8. Machine-Readable VPAT / ACR Report Generator

**Category**: Compliance Reporting
**Justification**: Government and enterprise procurement requires a Voluntary Product Accessibility Template (VPAT / ACR) documenting conformance per WCAG criterion. Generating this automatically from audit findings eliminates weeks of manual documentation work — a key pain point ACCS is positioned to solve.
**Implementation**: Add `async def generate_vpat(self, tenant_id: str, product_name: str, version: str) -> dict[str, Any]` that iterates the bundled WCAG criterion catalog, maps open/closed findings to each criterion, and emits a structured dict conforming to the ITI VPAT 2.5 schema. Add a Markdown/HTML renderer as a second pass.
**Competitor**: Level Access VPAT Express, Equalize Digital VPAT Generator

---

### I9. AI-Driven Remediation Suggestion Engine (Ollama)

**Category**: AI / Automation
**Justification**: Developers with low accessibility expertise stall on how to fix findings. An LLM-powered suggestion engine that explains the violation and provides a concrete code patch draft (HTML, CSS, or ARIA change) accelerates closure. Routing through a locally-hosted Ollama model keeps the IP on-premises and avoids SaaS data residency concerns.
**Implementation**: Add `async def suggest_remediation(self, finding_id: str, tenant_id: str) -> dict[str, Any]` that builds a prompt from finding `rule`, `description`, and `evidence`, posts to `http://localhost:11434/api/generate` using `httpx.AsyncClient`, and returns `{"suggestion": str, "confidence": float, "model": str}`. Cache responses in `BoundedCache` keyed on finding hash.
**Competitor**: Deque Axe Assistant, Microsoft Accessibility Insights AI suggestions

---

### I10. Keyboard Navigation Simulation with Focus-Trap Detection

**Category**: Assistive Technology Testing
**Justification**: Most keyboard navigation failures (focus traps, skip-link gaps, modal prison) cannot be detected without simulating Tab/Shift+Tab sequences. Static metadata (`keyboard_navigation_present: bool`) is too coarse. A simulated walk of the focusable element graph catches structural issues before users with motor disabilities encounter them.
**Implementation**: Add `async def simulate_keyboard_walk(self, tenant_id: str, target_id: str, max_steps: int = 200) -> dict[str, Any]` that (via Playwright adapter when enabled) traverses focusable elements, detects cycles longer than `max_steps`, identifies missing skip links (`<a href="#main">`), and returns a step trace + trap locations as findings.
**Competitor**: Accessibility Insights for Web (Tab Stops visualiser), axe DevTools Pro

---

### I11. Color-Blind Simulation and Automated Palette Validation

**Category**: Visual Accessibility
**Justification**: Contrast ratio (4.5:1) is necessary but not sufficient for users with color-vision deficiency. Deuteranopia, protanopia, and tritanopia simulations reveal UI elements that pass contrast ratios but convey information by color alone — a WCAG 1.4.1 violation. Automated palette analysis makes this routine rather than expert-dependent.
**Implementation**: Add `async def color_blindness_audit(self, tenant_id: str, target_id: str, palette: list[str], simulation_types: list[str] | None = None) -> dict[str, Any]` using a pure-Python DaltonLens-style matrix transform on hex color pairs. Score each pair for hue-only distinction under each simulation. Record WCAG 1.4.1 findings for failing pairs.
**Competitor**: Stark (Figma plugin), Colour Contrast Analyser, Sim Daltonism

---

### I12. Structured Audit Evidence Package for Legal Defence

**Category**: Governance / Legal
**Justification**: Under the EU Web Accessibility Directive and US ADA Title III, organisations must demonstrate due diligence. A cryptographically-signed, timestamped evidence bundle (audit metadata, finding records, review decisions, exception approvals) creates a defensible paper trail that can be produced in litigation or regulatory review.
**Implementation**: Add `async def export_evidence_bundle(self, tenant_id: str, audit_id: str | None = None) -> dict[str, Any]` that collects all findings, reviews, exceptions, and events for the tenant (or a single audit), serialises to canonical JSON, computes a SHA-256 digest, and stores `{"bundle": {...}, "sha256": str, "generated_at": str}`. Sign with `hmac` if `ACCS_EVIDENCE_KEY` is set.
**Competitor**: Level Access AMP Evidence Vault, Siteimprove Compliance Manager

---

### I13. Real-Time Accessibility Score Stream via Server-Sent Events

**Category**: Real-Time UX
**Justification**: Audit results today are available only after a full `run_audit` call. Streaming partial scores as each rule completes lets developers see violations appear in the IDE or CI log in real time — a 10x reduction in feedback latency. This is the accessibility equivalent of a live linting panel.
**Implementation**: Add `async def stream_audit_scores(self, audit_id: str, tenant_id: str) -> AsyncIterator[dict[str, Any]]` as an `async_generator`. Each yield emits `{"rule": str, "target_id": str, "passed": bool, "finding": dict | None}` as soon as a rule check completes. Wire into the Flask API as an SSE endpoint (`text/event-stream`).
**Competitor**: Evinced Devtools live stream, axe Monitor CI streaming

---

### I14. Multi-Language / i18n Accessibility Validation

**Category**: Internationalisation
**Justification**: RTL layouts (Arabic, Hebrew), CJK font scaling, and `lang` attribute correctness are accessibility requirements often missed in i18n pipelines. Validating these at audit time catches WCAG 3.1.1 (language of page) and 3.1.2 (language of parts) violations before users with AT encounter broken reading order.
**Implementation**: Add `async def i18n_accessibility_audit(self, tenant_id: str, target_id: str, declared_lang: str, rtl_layout: bool = False, cjk_font_scale: float = 1.0) -> dict[str, Any]`. Validate `declared_lang` against BCP 47 tag list. Check `rtl_layout` is set when lang is `ar`/`he`/`fa`/`ur`. Record WCAG 3.1.1 findings for mismatches.
**Competitor**: Siteimprove Multilingual Accessibility, Pope Tech i18n checks

---

### I15. Accessibility Debt Quantification with Financial Impact Scoring

**Category**: Business Intelligence
**Justification**: Accessibility backlogs are deprioritised because severity strings do not translate to business risk. Assigning an estimated remediation cost (developer-hours * rate) and a litigation risk score (based on finding severity, user-reach, and jurisdiction) converts the finding list into a CFO-readable risk ledger — the same framing that Level Access and Deque use in enterprise sales.
**Implementation**: Add `async def debt_ledger(self, tenant_id: str, hourly_rate: Decimal = Decimal("150"), jurisdiction: str = "us") -> dict[str, Any]`. Map severity to `effort_hours` (critical: 8, high: 4, medium: 2, low: 0.5). Compute `cost = Decimal(str(effort_hours)) * hourly_rate` per finding. Add `litigation_risk_score` based on severity weight and `published_ui` flag. Sum to tenant-level `total_estimated_cost: Decimal`.
**Competitor**: Level Access ROI Calculator, Deque Business Case Builder
