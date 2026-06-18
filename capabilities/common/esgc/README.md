# ESGC ESG and Carbon Tracking

`esgc` is the APG common ESG and carbon tracking capability. It lets generated
applications compose tenant-scoped emissions inventories, factor libraries,
activity emissions, sustainability reports, reduction targets, compliance
evidence, Bytewax stream governance, visual theme metadata, and AI-agent
assistance.

The package is dependency-light. It defines the executable service, rule
engine, UI route metadata, theme metadata, Bytewax stream declaration, API
helpers, view models, and semantic evidence. Meter integrations, forecasting
models, compliance filings, durable audit stores, geospatial providers, and
stream-worker deployments are adapter responsibilities.

## What It Provides

- Emissions inventory with organization owner, reporting year, boundary,
  geospatial boundary, and compliance framework.
- Approved emission factor library with source evidence, versioning, scope,
  units, and conversion rates.
- Activity data recording with evidence, anomaly review, scope classification,
  and carbon dioxide equivalent calculation.
- Scope 1/2/3 calculation with market-based, location-based, and category
  breakdowns.
- Sustainability report publishing with approval, compliance mapping, and audit
  evidence.
- Full GHG report generation (Scope 1+2+3 net) per standard and period.
- Reduction target tracking with baseline, target year, target reduction, and
  live progress calculation.
- Carbon credit trading with registry, counterparty, and settled-state recording.
- Carbon offset purchase and verification with registry and standard tracking.
- Net-zero target setting with pathway and interim milestones.
- Supplier ESG scoring with weighted metric aggregation and AAA–B rating.
- Biodiversity impact assessment with risk-level classification.
- Water usage recording with source and stress-level metadata.
- Waste tracking with disposal method and recycling percentage.
- Energy audit recording with renewable/non-renewable split.
- ESG rating (E/S/G/combined) with letter-grade output.
- SDG alignment mapping for UN goals 1–17.
- TCFD-aligned climate risk assessment (physical and transition risks).
- ESG disclosure generation for GRI, SASB, TCFD, ISSB, CDP, and CSRD.
- Green bond reporting (ICMA GBP) with use-of-proceeds and impact metrics.
- Bulk activity ingestion with parallel asyncio execution.
- CSV and JSON export for any collection.
- Full audit trail: every mutation writes an immutable audit event.
- AI ESGC-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch ESG mutation.
- Dashboard summary and health check.
- UI routes and visual theme tokens for generated APG applications.

## Quick Start

```python
import asyncio
from capabilities.common.esgc import EsgcService

service = EsgcService()

async def main():
    await service.create_inventory(
        inventory_id="inventory-2026",
        tenant_id="tenant-acme",
        organization="Acme Manufacturing",
        owner="sustainability-lead",
        reporting_year=2026,
        boundary_ref="boundary:operations",
        geospatial_boundary="geos:ke-operations",
        compliance_framework="GHG Protocol",
    )

    await service.register_factor(
        factor_id="factor-grid-ke",
        tenant_id="tenant-acme",
        name="Kenya grid electricity",
        scope="scope_2",
        unit="kwh",
        co2e_per_unit=0.00025,
        source="national-grid-factor",
        source_evidence="audl:evidence-grid-2026",
        version="2026.1",
        approved_source=True,
    )

asyncio.run(main())
```

All `EsgcService` methods are async. Use `asyncio.run()` or await them inside
an existing event loop.

## AI Agent Registration

AI agents are first-class ESG contributors only after registration:

```python
agent = await service.register_esgc_agent(
    tenant_id="tenant-acme",
    name="Report reviewer",
    runtime="codex",
    role="report_reviewer",
    scope="review report evidence, approval, and compliance mapping",
    contribution_disclosed=True,
)
```

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.
Supported roles: `inventory_reviewer`, `factor_reviewer`, `activity_reviewer`,
`report_reviewer`, `target_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- inventory owner or reporting boundary is missing;
- factor source is not approved;
- factor source evidence or version is missing;
- activity evidence is missing;
- activity references a factor with a different unit;
- report approval, compliance mapping, or audit evidence is missing;
- reduction target baseline is missing;
- activity anomaly lacks review;
- an AI ESGC agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch ESG mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch ESG mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_esgc_mutation("bytewax")
blocked  = service.validate_batch_esgc_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.esgc.lifecycle` and state for inventories,
factors, activities, reports, targets, ESGC agents, and audit events.

## API Reference

| Method | Description |
|---|---|
| `create_inventory(...)` | Create a tenant-scoped emissions inventory |
| `register_factor(...)` | Register an approved emission factor |
| `record_activity(...)` | Record an emission activity with CO2e calculation |
| `scope1_record(...)` | Convenience recorder for Scope 1 direct emissions |
| `scope1_calculation(...)` | Aggregate all Scope 1 emissions for a period |
| `scope2_calculate(...)` | Market-based and location-based Scope 2 totals |
| `scope3_estimate(...)` | Value-chain Scope 3 estimate by category |
| `bulk_record_activities(...)` | Parallel bulk ingestion of emission activities |
| `publish_report(...)` | Publish an approved sustainability report |
| `ghg_report(...)` | Full Scope 1+2+3 GHG report with net emissions |
| `create_target(...)` | Create a reduction target with live progress |
| `net_zero_target_setting(...)` | Set a net-zero pathway with interim milestones |
| `carbon_offset_purchase(...)` | Record a carbon offset retirement |
| `carbon_offset_verify(...)` | Verify offset status and standard |
| `carbon_credit_trade(...)` | Record a buy/sell carbon credit trade |
| `esg_analytics(...)` | Aggregate analytics across all entities |
| `esg_score_calculation(...)` | Compute weighted E/S/G/combined score |
| `esg_rating(...)` | Compute and store a combined ESG rating |
| `esg_disclosure_generation(...)` | Generate structured framework disclosure |
| `supplier_esg_score(...)` | Weighted ESG score for a supplier |
| `green_certification(...)` | Record a green certification (ISO 14001, LEED…) |
| `biodiversity_impact(...)` | Record a biodiversity impact assessment |
| `water_usage(...)` | Record water consumption with stress-level metadata |
| `waste_tracking(...)` | Record waste with disposal method and recycling % |
| `energy_audit(...)` | Record energy audit with renewable split |
| `sdg_alignment(...)` | Map entity contributions to UN SDGs 1–17 |
| `climate_risk(...)` | TCFD-aligned physical + transition risk assessment |
| `green_bond_reporting(...)` | ICMA GBP green bond report |
| `generate_narrative_section(...)` | LLM-assisted narrative paragraph for a report section |
| `csrd_gap_analysis(...)` | ESRS E1–G1 completeness matrix |
| `carbon_intensity(...)` | tCO2e per revenue/headcount/floor-area/output |
| `check_target_progress(...)` | Live progress check with at_risk/off_track alerts |
| `scope3_financed_emissions(...)` | PCAF Scope 3 Cat-15 financed emissions |
| `export_csv(...)` | CSV export of any collection |
| `export_json(...)` | JSON export of any collection |
| `dashboard_summary(...)` | Counts and totals across all collections |
| `health_check()` | Service liveness check |
| `list_inventories(...)` | List inventories (optionally scoped to tenant) |
| `list_factors(...)` | List factors |
| `list_activities(...)` | List activities |
| `list_reports(...)` | List reports |
| `list_targets(...)` | List targets |
| `list_offsets(...)` | List offsets |
| `list_esg_scores(...)` | List ESG scores |
| `list_disclosures(...)` | List disclosures |
| `list_credit_trades(...)` | List credit trades |
| `list_audit_events(...)` | List audit events |
| `list_esgc_agents(...)` | List registered AI agents |
| `register_esgc_agent(...)` | Register an AI agent with role and scope |

## World-Class Enhancements (v2.0)

These 15 improvements elevate `esgc` from a prototype to production-grade status.
Adapter implementations are out-of-scope for the core capability; the interfaces
below are the contracts to target.

1. **Persistent Storage Backend** — Swap `_Store` for a SQLAlchemy async engine
   (PostgreSQL + asyncpg). `_Store` interface preserved for in-memory test usage.
   Enables durable audit trails and multi-process deployments.

2. **Real-Time Emissions Streaming via Bytewax** — `BytewaxEmissionStream`
   publishes every `record_activity` event to `apg.esgc.lifecycle` and fans out
   anomaly detection + target-progress recalculation as stateful windowed
   operators. Sub-second anomaly alerting; live dashboard deltas without polling.

3. **ML-Based Anomaly Detection** — Isolation Forest (or z-score baseline) per
   `(tenant_id, scope, activity_type)` on historical quantities. Outliers require
   `anomaly_review_recorded=True` before an activity can reach `settled`. Catches
   data-entry errors and greenwashing before they enter official reports.

4. **SBTi-Aligned Pathway Validation** — `SBTiPathwayValidator` checks target
   year and reduction percentage against 1.5°C and well-below-2°C sector
   benchmarks. Returns `sbti_aligned: bool` + `gap_analysis` dict. Automatic
   compliance signalling to investors and rating agencies.

5. **Multi-Currency Carbon Credit Pricing** — Pluggable FX adapter normalises all
   `total_value` fields to USD with a 1-hour rate cache. Exposes
   `price_usd_equivalent` on every trade. Enables portfolio-level carbon cost
   analysis across EUA, RGGI, and VCS markets.

6. **Scope 3 Category 15 (Investments) Calculator** — `scope3_financed_emissions`
   uses PCAF methodology: weighted average carbon intensity per asset class (equity,
   bonds, real estate, loans). Enables financial institutions to report TCFD Scope
   3 Cat-15 without bespoke spreadsheet models.

7. **Automated CSRD / ESRS Data Gap Analysis** — `csrd_gap_analysis` maps
   ESRS E1–E5, S1–S4, G1 data points to stored records and returns a structured
   completeness matrix (`missing` / `partial` / `complete` per disclosure
   requirement). Turns weeks of manual gap analysis into a single API call.

8. **Geospatial Water Stress Integration** — WRI Aqueduct API adapter
   auto-populates `water_stress_level` from `(latitude, longitude)` with a 30-day
   location cache. Required for TNFD and CSRD ESRS E3 water disclosures.

9. **Regulatory Filing Export (XBRL/iXBRL)** — `export_xbrl` renders an
   IFRS S2 / GRI-compliant inline XBRL document (via `arelle` or a lightweight
   template engine). Enables direct machine-readable filings to SEC, ESMA, and
   Kenya CMA.

10. **Verifiable Credentials for Audit Evidence** — W3C VC (JSON-LD, Ed25519)
    issued for each audit-critical record. VC proof stored in the audit event.
    `verify_credential(vc_jwt)` lets third-party verifiers confirm data provenance
    cryptographically without raw storage access.

11. **Automated Target Progress Notifications** — `check_target_progress` compares
    live emissions against all active targets and fires `_Notify.send` on
    `["email", "webhook"]` when a target is `at_risk` (within 10% of missing the
    annual milestone) or `off_track`. Closes the feedback loop between data
    recording and strategic decisions.

12. **Carbon Intensity KPIs** — `carbon_intensity` computes tCO2e per unit of
    revenue, headcount, floor area, or product output with time-series storage for
    trend charting. Enables like-for-like peer comparisons and CDP sector
    benchmarks.

13. **Dual-Register Offset Integrity Check** — `_retire_ledger` keyed by
    `(registry, serial_number)` rejects retirement if a serial was already retired
    under any tenant. Cross-tenant conflicts logged at `severity="critical"`.
    Prevents offset double-counting — the most common greenwashing vector
    identified by SEC and FCA investigations.

14. **LLM-Assisted Narrative Generation** — `generate_narrative_section` calls a
    locally hosted Ollama model (e.g. `llama3.2`) with structured report data as
    context and returns a draft disclosure paragraph stored under
    `report.narrative_sections`. Cuts report writing from days to minutes while
    keeping humans in the approval loop.

15. **Event-Driven Capability Composition Hooks** — Direct capability imports
    replaced by an async event bus (`asyncio.Queue` internally; Bytewax adapter
    externally). Emits `esgc.inventory.created`, `esgc.activity.recorded`,
    `esgc.report.published`. Subscribing capabilities replace hard-wired calls —
    hot-swappable integrations and replay-based testing without mocks.

## New Methods

### `ghg_report` — Full GHG report generation

```python
report = await service.ghg_report(
    tenant_id="tenant-acme",
    report_id="ghg-2026-q1",
    entity_id="inventory-2026",
    standard="ISSB",
    period="2026-Q1",
    approved_by="cfo@acme.com",
)
# report["net_total_co2e_tonnes"] is gross minus all retired offsets
```

### `carbon_intensity` — Intensity KPI per business denominator

```python
intensity = await service.carbon_intensity(
    tenant_id="tenant-acme",
    period="2026",
    denominator_key="revenue_usd",
    denominator_value=5_000_000.0,
)
# intensity["tco2e_per_unit"] — comparable to CDP sector benchmarks
```

### `csrd_gap_analysis` — ESRS completeness matrix

```python
gaps = await service.csrd_gap_analysis(
    tenant_id="tenant-acme",
    entity_id="acme-corp",
    period="2026",
)
# gaps["matrix"]["E1"]["climate_risk"] -> "missing" | "partial" | "complete"
```

### `generate_narrative_section` — Ollama-backed disclosure narrative

```python
draft = await service.generate_narrative_section(
    tenant_id="tenant-acme",
    report_id="ghg-2026-q1",
    section="climate_strategy",
    model="llama3.2",
)
# draft["narrative"] -> ready-for-review paragraph; stored under report
```

### `check_target_progress` — Proactive milestone alerting

```python
status = await service.check_target_progress(tenant_id="tenant-acme")
# status["at_risk"] -> list of targets within 10% of missing annual milestone
# status["off_track"] -> list of targets already behind; email + webhook fired
```

## Composition

Generated APG applications should compose `esgc` through:

- capability ID: `esgc`
- provided services: emissions inventory, factor library, activity emissions,
  sustainability reporting, target tracking, ESG evidence, and ESGC agents
- required services: `auth`, `conf`, `audl`, `geos`, `pred`, and `comp`
- API prefix: `/esgc/api/v1`
- UI routes: dashboard, emissions, factors, data sources, reports, targets,
  agents, rules, audit, and settings
- theme: `esgc_sustainability_ops`
- stream processor: `bytewax`

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/esgc/__init__.py capabilities/common/esgc/capability_contract.py capabilities/common/esgc/models.py capabilities/common/esgc/service.py capabilities/common/esgc/api.py capabilities/common/esgc/views.py capabilities/common/esgc/app.py capabilities/common/esgc/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/esgc/test_capability_contract.py
./.venv/bin/python -c "import asyncio; from capabilities.common.esgc import EsgcService; service = EsgcService(); print(asyncio.run(service.dashboard_summary('tenant-proof')))"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/esgc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/esgc --json
```
