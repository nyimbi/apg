# Pharmacovigilance Intelligence — World-Class Improvements

**Capability**: `pharma_pvi` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Full Async Service Layer

**Current state**: All core methods (`create_case`, `process_case`, `submit_icsr`, etc.) are synchronous despite I/O-bound database and regulatory API calls. Only auto-generated expansion stubs are async.

**Improvement**: Convert every public method to `async def`. Gate all DB reads/writes through `asyncpg` connection pools, enabling concurrent ICSR submissions and signal detection runs without blocking the event loop.

**Impact**: 10–40x throughput increase for batch processing. Eliminates deadlocks during simultaneous multi-product PSUR data collection.

---

## 2. Persistent PostgreSQL Store with Alembic Migrations

**Current state**: All stores are in-memory dicts (`self._cases`, `self._signals`, etc.). Process restart loses all data.

**Improvement**: Replace in-memory dicts with `asyncpg`-backed `DatabaseStore` (already scaffolded in `database/store.py`). Full Alembic migration suite with proper indices on `(tenant_id, status)`, `(product_id, meddra_pt)` and `(submission_date, due_date)` for timeline compliance queries.

**Impact**: Production-grade persistence. Enables multi-instance horizontal scaling. Supports audit trail forensics across restarts.

---

## 3. Structured Timeline Compliance Engine

**Current state**: Timeline checks (`within_7d`, `within_15d`) are passed as boolean flags by the caller — honour-system enforcement.

**Improvement**: Implement `TimelineComplianceEngine` that computes elapsed days from `report_date` to `submission_date`, evaluates against ICH E2A / EMA / FDA deadline tables keyed by `(case_type, regulatory_database)`, and raises `TimelineViolationError` with structured breach metadata. Emit `timeline_breach_detected` stream events for downstream alerting.

**Impact**: Eliminates regulatory citations from missed deadlines. Supports automated breach notification to `ntfy` capability.

---

## 4. MedDRA Hierarchy Traversal and Validation

**Current state**: `meddra_pt` and `meddra_soc` are free-text strings; no validation against the MedDRA hierarchy.

**Improvement**: Integrate a local MedDRA SQLite lookup (shipped as a capability asset) providing `validate_meddra_pt()`, `pt_to_soc()`, `pt_to_hlgt()`, and `pt_to_hlt()` with full hierarchy traversal. Validate PT/SOC on `process_case` and reject unknown terms.

**Impact**: Prevents miscoded cases from reaching regulatory databases. Required for WHO VigiBase submissions. Enables SOC-level signal aggregation.

---

## 5. Statistical Disproportionality Suite (ROR + PRR + EBGM)

**Current state**: `signal_detection` computes a simplified pseudo-ROR with no confidence intervals and a hard-coded `n >= 3` threshold.

**Improvement**: Implement proper Reporting Odds Ratio with 95% CI (Woolf method), Proportional Reporting Ratio with chi-squared test, and Empirical Bayes Geometric Mean (EBGM) using the multi-item gamma Poisson shrinker (MGPS). Store background rates from a configurable reference dataset per `(database_source, analysis_period)`.

**Impact**: Reduces false-positive signal burden by 60–80%. Meets EMA PRAC and FDA SRS statistical methodology requirements. Enables ranking signals by EBGM05.

---

## 6. ICH E2B(R3) XML Serialiser and Validator

**Current state**: `submit_to_eudravigilance` and `submit_to_fda_aers` assume upstream formatting is done. The actual E2B(R3) XML is never produced.

**Improvement**: Implement `E2BSerializer.serialize(case: AdvEventCase) -> str` producing valid ICH E2B(R3) ICHICSR XML with all mandatory elements (N.1.*, C.1.*, D.*, E.*, F.*, G.*). Validate against the official ICHICSR XSD before submission. Assign `e2b_r3_message_id` from UUID7.

**Impact**: Enables direct gateway integration with EudraVigilance EVWEB and FDA ESG. Removes dependency on external formatting tools.

---

## 7. Automated Literature Screening Scheduler

**Current state**: `record_literature` is a manual intake step. No periodic automated screening is modelled.

**Improvement**: `async def schedule_literature_screening(tenant_id, databases, products, cron_expr)` that persists a screening schedule, runs async HTTP queries against PubMed, EMBASE, and WHO ICSDb APIs, deduplicates against existing `article_reference` values, and queues new hits for medical assessment. Emit `literature_screening_completed` events.

**Impact**: Meets ICH E6(R2) GVP Module VI requirement for weekly literature screening of medicinal products. Eliminates manual gap risk.

---

## 8. Benefit-Risk Assessment Structured Framework

**Current state**: `submit_psur` accepts a boolean `benefit_risk_assessed` flag. No structured benefit-risk content is captured.

**Improvement**: Implement `BenefitRiskAssessment` Pydantic model covering: identified risks, potential risks, missing information, risk minimisation measures, benefit characterisation, and overall B/R conclusion per EU RMP Annex I format. Link assessment to `PsurReport.signal_evaluation_reference`. Gate `submit_psur` on a fully-populated assessment object.

**Impact**: Produces audit-ready B/R documentation. Satisfies EMA GVP Module V and ICH E2C(R2) Section 16 requirements. Enables cross-product B/R portfolio analysis.

---

## 9. Aggregate Reporting (DSUR / SUSAR Line Listings)

**Current state**: Only PSUR/PBRER generation is implemented. Development Safety Update Reports and SUSAR line listings are absent.

**Improvement**: `async def generate_dsur(drug_id, trial_id, period, tenant_id)` producing Development Safety Update Reports per ICH E2F. `async def generate_susar_line_listing(trial_id, tenant_id)` producing EudraCT-compliant line listings. Both methods collect from `pharma_ctr` feed via event bus.

**Impact**: Covers clinical trial PV obligations. Required for all IND/IND-equivalent products. Eliminates double-handling between clinical and post-market PV teams.

---

## 10. Multi-Tenant RBAC with PV Role Hierarchy

**Current state**: `_enforce` evaluates a flat policy context. There is no role differentiation between PV officers, medical reviewers, qualified persons for PV (QPPV), and auditors.

**Improvement**: Introduce `PvRole` enum (`pv_officer`, `medical_reviewer`, `qppv`, `pv_auditor`, `regulatory_affairs`). Bind each method to required minimum role via `@requires_pv_role(PvRole.medical_reviewer)` decorator. QPPV role required for PSUR submission and label update approval. Audit all role elevations.

**Impact**: Satisfies EU GVP Module I QPPV accountability requirements. Prevents unauthorised PSUR submissions. Enables segregation of duties audit trail.

---

## 11. Case Deduplication via Phonetic and Semantic Matching

**Current state**: `mark_duplicate` is a manual, explicit operation. No automatic duplicate detection algorithm is implemented.

**Improvement**: `async def auto_detect_duplicates(case_id, tenant_id)` computing similarity scores across: reporter name (Metaphone), patient demographics (age ± 2y, sex), suspect drug, MedDRA PT, and onset date (± 7 days). Return ranked candidates with similarity scores. Above configurable threshold, propose auto-linkage pending QPPV approval.

**Impact**: Reduces duplicate ICSRs reaching regulatory databases (a major FDA Form 483 finding). Typical deduplication catch rate 15–25% of spontaneous reports.

---

## 12. Streaming Event Bus Integration (Bytewax / Kafka)

**Current state**: `_audit` appends to a Python list with hardcoded `"processor": "bytewax"`. No actual stream is produced.

**Improvement**: Replace `_audit` with `async def _emit(event_type, payload)` that publishes typed `CloudEvent` (CE 1.0) to a configurable Kafka / Bytewax topic (`apg.pharma.pvi.lifecycle`). Integrate with `mqeb` capability. Provide consumer stubs for downstream `pharma_rec` and `pharma_reg` signal feeds.

**Impact**: Enables real-time signal dashboards, cross-capability composition, and regulatory timeline alerting without polling. Decouples PV data producers from consumers.

---

## 13. PSUR Submission Deadline Tracker with EMA EURD List

**Current state**: PSUR timelines are not validated against the EMA EURD (European Union Reference Dates) list.

**Improvement**: Ingest EMA EURD list (published quarterly as XML) into a local reference table keyed by `(active_substance, din_code)`. `async def check_psur_eurd_deadline(drug_id, tenant_id)` returns next PSUR due date, submission window, and EURD DLP. Emit `psur_deadline_approaching` events at 90, 30, and 7 days.

**Impact**: Eliminates missed PSUR windows (€100k+ EMA penalty range). Automates EURD compliance calendar for multi-product portfolios.

---

## 14. AI-Assisted Narrative Generation via Local LLM

**Current state**: `ml_adverse_event_classify` stub calls `MLCapability` for classification only. Narrative writing is entirely manual.

**Improvement**: `async def generate_case_narrative(case_id, tenant_id)` calls a locally-hosted Ollama model (e.g., `llama3.1:8b` with a structured PV narrative prompt template per ICH E2B Section G.k.9) to draft a case narrative from structured fields. Treat output as draft requiring medical reviewer sign-off. Store draft in `AdvEventCase.narrative` with `ai_generated=True` flag.

**Impact**: Reduces narrative writing time from 45–90 min to 5–10 min per case. Consistent ICH-compliant structure. Fully audited — AI authorship disclosed in ICSR metadata.

---

## 15. Risk Management Plan (RMP) Linkage and Tracking

**Current state**: Label update proposals (`label_update_proposal`) exist as standalone records with no linkage to EU RMP or REMS structures.

**Improvement**: Implement `RiskManagementPlan` model covering: safety concerns (identified/potential/missing information), pharmacovigilance activities (routine/additional), risk minimisation measures (routine/additional), and RMP version history. `async def update_rmp_safety_concern(rmp_id, concern_id, signal_id, tenant_id)` links confirmed PV signals to RMP updates. Trigger `rmp_update_required` events to `pharma_reg`.

**Impact**: Closes the signal-to-RMP update loop required by EMA GVP Module V. Enables automated RMP version tracking and submission scheduling. Prevents orphaned signals that never reach product labelling.
