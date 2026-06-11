# ESGC World-Class Improvements

15 high-priority improvements to elevate the ESG/Carbon tracking capability to production-grade status.

---

## 1. Persistent Storage Backend

**Current**: In-memory `_Store` dict — all state lost on restart.
**Improvement**: Swap `_Store` for a SQLAlchemy async engine (PostgreSQL). Use `asyncpg` + `sqlalchemy[asyncio]`. Keep `_Store` as an interface so the in-memory variant remains for tests.
**Impact**: Durable audit trails, multi-process deployments, regulatory data retention.

---

## 2. Real-Time Emissions Streaming via Bytewax

**Current**: Batch mutation validated by a flag check; no actual stream processing.
**Improvement**: Implement a `BytewaxEmissionStream` that publishes every `record_activity` event to topic `apg.esgc.lifecycle` and fans out anomaly detection + target-progress recalculation as stateful windowed Bytewax operators.
**Impact**: Sub-second anomaly alerting; live dashboard deltas without polling.

---

## 3. ML-Based Anomaly Detection

**Current**: `anomaly_detected` field always set to `False`.
**Improvement**: Fit an Isolation Forest (or z-score baseline) per `(tenant_id, scope, activity_type)` on historical quantities. Flag outliers and require `anomaly_review_recorded=True` before the activity can reach `settled` status.
**Impact**: Catches data entry errors and potential greenwashing before they enter official reports.

---

## 4. SBTi-Aligned Pathway Validation

**Current**: `net_zero_target_setting` accepts any `pathway` string with no validation.
**Improvement**: Add a `SBTiPathwayValidator` that checks target year and reduction percentage against 1.5°C and well-below-2°C sector-specific benchmarks. Return `sbti_aligned: bool` + `gap_analysis` dict.
**Impact**: Automatic compliance signalling to investors and rating agencies.

---

## 5. Multi-Currency Carbon Credit Pricing

**Current**: `carbon_credit_trade` stores a single price; no FX conversion.
**Improvement**: Integrate an FX rate provider (pluggable adapter) to normalise all `total_value` fields to a base currency (USD). Cache rates with a 1-hour TTL. Expose `price_usd_equivalent` on every trade record.
**Impact**: Consolidated portfolio-level carbon cost analysis across markets (EUA, RGGI, VCS).

---

## 6. Scope 3 Category 15 (Investments) Calculator

**Current**: Scope 3 estimate is a flat filter on stored activities; financed emissions absent.
**Improvement**: Add `scope3_financed_emissions(tenant_id, portfolio, period)` using PCAF methodology — weighted average carbon intensity per asset class (equity, bonds, real estate, loans).
**Impact**: Enables financial institutions to report TCFD Scope 3 Cat-15 without bespoke spreadsheet models.

---

## 7. Automated CSRD / ESRS Data Gap Analysis

**Current**: `esg_disclosure_generation` lists sections but does not validate data completeness.
**Improvement**: Add `csrd_gap_analysis(tenant_id, entity_id, period)` that maps ESRS E1–E5, S1–S4, G1 data points to stored records and returns a structured completeness matrix with `missing`, `partial`, `complete` per disclosure requirement.
**Impact**: Turns weeks of manual gap analysis into a single API call ahead of external auditors.

---

## 8. Geospatial Water Stress Integration

**Current**: `water_usage` takes `water_stress_level` as a free-form string.
**Improvement**: Integrate the WRI Aqueduct API (async HTTP adapter) to auto-lookup `water_stress_level` from `(latitude, longitude)`. Cache results for 30 days per location.
**Impact**: Removes manual lookup error; required for TNFD and CSRD ESRS E3 water disclosures.

---

## 9. Regulatory Filing Export (XBRL/iXBRL)

**Current**: Exports only CSV and JSON.
**Improvement**: Add `export_xbrl(tenant_id, disclosure_id, taxonomy)` that renders an IFRS S2 / GRI-compliant inline XBRL document. Use `arelle` or a lightweight template engine.
**Impact**: Direct machine-readable filings to SEC (ESG disclosures), ESMA, and Kenya CMA.

---

## 10. Verifiable Credentials for Audit Evidence

**Current**: `evidence_ref` is an opaque string with no integrity check.
**Improvement**: Issue W3C Verifiable Credentials (JSON-LD, Ed25519 signatures) for each audit-critical record (factors, activities, reports). Store the VC proof in the audit event. Expose `verify_credential(vc_jwt)`.
**Impact**: Third-party verifiers and rating agencies can cryptographically confirm data provenance without accessing raw storage.

---

## 11. Automated Target Progress Notifications

**Current**: Targets are created and forgotten; no alerting when progress slips.
**Improvement**: Add `check_target_progress(tenant_id)` that compares live emissions against all active targets and fires `_Notify.send` on channels `["email", "webhook"]` when a target is `at_risk` (within 10 % of missing the annual milestone) or `off_track`.
**Impact**: Proactive sustainability management; closes the feedback loop between data recording and strategic decisions.

---

## 12. Carbon Intensity KPIs

**Current**: Absolute emissions only — no normalisation to business activity.
**Improvement**: Add `carbon_intensity(tenant_id, period, denominator_key, denominator_value)` that computes tCO2e per unit of revenue, headcount, floor area, or product output. Store time-series for trend charting.
**Impact**: Enables like-for-like comparisons against peers and CDP sector benchmarks.

---

## 13. Dual-Register Offset Integrity Check

**Current**: Offset purchase sets `status="retired"` immediately; no double-counting guard.
**Improvement**: Maintain a `_retire_ledger` keyed by `(registry, serial_number)`. Reject retirement if the serial was already retired under any tenant. Log cross-tenant conflicts with `severity="critical"`.
**Impact**: Prevents offset double-counting — the most common greenwashing vector identified by SEC and FCA investigations.

---

## 14. LLM-Assisted Narrative Generation

**Current**: Reports are structured dicts; no human-readable narrative.
**Improvement**: Add `generate_narrative_section(tenant_id, report_id, section, model)` that calls a locally hosted Ollama model (e.g., `llama3.2`) with structured report data as context and returns a draft disclosure paragraph. Store draft under the report's `narrative_sections` key.
**Impact**: Cuts report writing from days to minutes while keeping humans in the approval loop.

---

## 15. Event-Driven Capability Composition Hooks

**Current**: `esgc` calls other capabilities (auth, geos, pred) via direct import; tight coupling.
**Improvement**: Replace direct calls with an async event bus (`asyncio.Queue` internally, Kafka adapter externally). Emit `esgc.inventory.created`, `esgc.activity.recorded`, `esgc.report.published` events. Other capabilities subscribe rather than being called.
**Impact**: Decouples esgc from concrete adapter implementations, enabling hot-swappable integrations and replay-based testing without mocks.
