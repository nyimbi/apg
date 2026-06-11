# Data Quality — World-Class Improvement Proposals

## I1. Column-Level Distribution Fingerprinting
**Category**: Profiling
**Justification**: Point-in-time null/distinct counts miss distributional shape. Distribution fingerprints (histogram bucket hashes) catch data substitution and silent schema evolution that standard completeness rules miss entirely.
**Implementation**: For each numeric column compute equal-width histogram buckets (default 20), store bucket counts + a SHA-256 fingerprint. On subsequent profiles compare fingerprint distance using L1 norm. Expose `distribution_drift_score` per column in profile comparison.
**Competitor Reference**: Great Expectations `expect_column_kl_divergence_to_be_less_than` + Monte Carlo Data's distribution drift monitor.

---

## I2. Regex Rule Evaluation Against Real Data
**Category**: Rule Execution
**Justification**: `regex` is listed as a supported rule type but `_evaluate_rule` has no implementation for it — the fallback score is always 1.0, silently marking every regex rule as passing. This is a correctness bug with compliance consequences.
**Implementation**: Compile the `expression` field as a regex pattern, evaluate fraction of non-null column values that match it, compare against threshold. Cache compiled patterns keyed by `(rule_id, expression)`.
**Competitor Reference**: dbt `accepted_values` + Soda Core `matches_regex` check.

---

## I3. Referential Integrity Rule Evaluation
**Category**: Rule Execution
**Justification**: Same as I2 — `referential` rule type is declared but not implemented. Critical for FK-style validation (e.g. product_id must exist in products table).
**Implementation**: `expression` field encodes the reference set as a comma-separated list or JSON array. Compute fraction of column values present in the reference set. Expose `missing_values` list (capped at 100) in rule result for debugging.
**Competitor Reference**: Great Expectations `expect_column_values_to_be_in_set`; dbt `relationships` test.

---

## I4. Freshness Rule with Timestamp Column Evaluation
**Category**: Rule Execution
**Justification**: Freshness rule currently always scores 1.0. For near-real-time pipelines (IoT, payments) stale data causes downstream model failures hours before anyone notices.
**Implementation**: `column` specifies the timestamp column; `expression` = max allowed age in ISO-8601 duration (e.g. `PT6H`). Parse durations via `isodate` / manual parsing. Find max timestamp in sample, compute elapsed seconds vs allowed, score = 0.0 if exceeded, 1.0 if within.
**Competitor Reference**: dbt `freshness` source block; Airflow `SLAMissCallback`.

---

## I5. Multi-Dimensional DQ Scorecard (6 ISO 25012 Dimensions)
**Category**: Scoring
**Justification**: A single aggregate quality score conflates orthogonal failure modes. ISO 25012 defines: Completeness, Consistency, Accuracy, Timeliness, Uniqueness, Validity. Separating dimensions lets data owners act on the right remediation lever.
**Implementation**: Group rule results by dimension (map rule_type → dimension), compute per-dimension score. Persist `dimension_scores` dict in scorecard. Add weighted composite score with configurable dimension weights.
**Competitor Reference**: Collibra DQ dimension scores; Informatica CDGC quality dimensions.

---

## I6. Statistical Outlier Detection Per Column (IQR + Z-Score)
**Category**: Anomaly Detection
**Justification**: Current anomaly detection only fires on aggregate score drops, missing column-level value outliers in numeric data that indicate sensor faults, ETL bugs, or fraud signals.
**Implementation**: For numeric columns in a data sample: compute IQR and flag values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`; also flag values beyond 3-sigma. Return `outlier_rate` per column and a `sample_outliers` list. Store as `column_anomaly` type in self.anomalies.
**Competitor Reference**: Anomalo column-level anomaly detection; Monte Carlo automatic outlier monitors.

---

## I7. Rule Template Library
**Category**: Usability / Governance
**Justification**: DQ practitioners spend significant time hand-crafting the same rules for common patterns (email format, phone number, UUID, positive integer, ISO date). A template library collapses this to a single call.
**Implementation**: Curated `RULE_TEMPLATES` dict mapping template name → `{rule_type, expression, description, column_hint}`. `create_rule_from_template(template_name, dataset_id, column)` expands template then delegates to `create_rule`. Expose `list_rule_templates()`.
**Competitor Reference**: dbt built-in generic tests (`not_null`, `unique`, `accepted_values`); Soda pre-built check library.

---

## I8. Incremental Data Profiling (Partition-Aware)
**Category**: Profiling Performance
**Justification**: Re-profiling a 500M-row dataset on every DQ run is infeasible. Partition-aware incremental profiling merges statistics from a new partition into the existing profile using Welford's online algorithm for mean/variance — O(1) per new partition.
**Implementation**: `profile_dataset_incremental(tenant_id, dataset_id, partition_id, partition_profiles)` merges partition statistics into existing profile using Welford's method for streaming mean/variance. Stores partition registry for auditability.
**Competitor Reference**: Apache Griffin incremental profiling; dbt incremental materialisation + re-profiling hooks.

---

## I9. Data Lineage Impact Score
**Category**: Governance / Observability
**Justification**: A quality failure in a source dataset has compounding downstream impact. An impact score quantifying "how many downstream datasets/models depend on this dataset" lets teams triage DQ issues by business risk, not just raw score.
**Implementation**: `register_lineage(tenant_id, source_dataset_id, downstream_dataset_ids)` builds a dependency graph. `get_impact_score(tenant_id, dataset_id)` returns: direct dependents count, transitive dependents count, impact_tier (critical/high/medium/low). Factor impact_tier into alert severity escalation.
**Competitor Reference**: DataHub lineage graph + impact analysis; Atlan impact score.

---

## I10. Expectations Catalog (Declarative YAML/JSON Import)
**Category**: Configurability
**Justification**: Data teams manage DQ expectations in version-controlled YAML alongside dbt models. A bulk-import mechanism closes the gap between declarative config and the runtime rule store — essential for GitOps workflows.
**Implementation**: `import_expectations(tenant_id, expectations_doc)` parses a list of rule specs (supports dbt-style and GE-style schemas via a normalisation layer), bulk-creates rules, returns import summary with success/skip/error counts and rule IDs.
**Competitor Reference**: Great Expectations expectation suites JSON; dbt `schema.yml` tests block.

---

## I11. SLA Breach Tracking and Escalation
**Category**: Governance / Alerting
**Justification**: Knowing that quality dropped is less useful than knowing it violated an agreed SLA (e.g. "completeness must stay above 0.95 during business hours"). SLA breach history is audit evidence for data contracts.
**Implementation**: `define_sla(tenant_id, dataset_id, dimension, min_score, business_hours_only)` stores SLA. After each run, `_check_slas(tenant_id, dataset_id, run_result)` evaluates SLAs, creates `sla_breach` records with breach duration tracking, emits `sla_breach_detected` audit event.
**Competitor Reference**: Atlan data contracts + SLA monitoring; Monte Carlo SLA manager.

---

## I12. Cross-Dataset Consistency Validation
**Category**: Rule Execution
**Justification**: Row counts, referential keys, and aggregate totals must be consistent across related datasets (e.g. orders total in transactions must equal orders total in warehouse). No existing rule type covers cross-dataset assertions.
**Implementation**: New rule_type `cross_dataset`. `expression` encodes `{source_metric}:{target_dataset_id}:{target_metric}:{tolerance}`. `run_cross_dataset_check(tenant_id, dataset_a, dataset_b, metric, tolerance)` compares aggregate metrics across two dataset profiles, returns consistency score.
**Competitor Reference**: Great Expectations `expect_multicolumn_sum_to_equal`; Soda `cross` checks.

---

## I13. Automated Rule Suggestion via Column Profiling
**Category**: Intelligence / Automation
**Justification**: Writing rules from scratch requires domain expertise. Column statistics contain implicit rule hints: a column with 0 nulls historically → suggest completeness >= 0.99; a column with 100% distinct values → suggest uniqueness == 1.0. Auto-suggestion accelerates DQ adoption.
**Implementation**: `suggest_rules(tenant_id, dataset_id)` reads the latest profile, applies heuristics (null rate < 0.01 → completeness rule; uniqueness > 0.98 → uniqueness rule; min/max bounds present → range rule), returns list of candidate `DQRuleCreate` payloads with confidence scores. Does not auto-create rules — requires explicit confirmation.
**Competitor Reference**: Informatica CDGC auto-discovery; Monte Carlo automated monitors.

---

## I14. Quality Score Trend Forecasting (EWMA)
**Category**: Analytics / Predictive
**Justification**: A degrading quality score detected after it crosses a threshold is too late for prevention. Exponentially weighted moving average (EWMA) trend extrapolation gives an advance warning window proportional to the trend velocity.
**Implementation**: `forecast_quality_score(tenant_id, dataset_id, horizon_runs)` reads score history from scorecard, fits EWMA (configurable alpha=0.3), extrapolates `horizon_runs` steps forward, returns `{forecast: [...], will_breach_threshold: bool, breach_at_run: int | None}`.
**Competitor Reference**: Anomalo predictive quality forecasting; DataBand trend anomaly detection.

---

## I15. Data Contract Enforcement
**Category**: Governance
**Justification**: Data contracts (OpenDataContract standard) formalise agreements between data producers and consumers. Integrating contract enforcement into DQ runs means violations are surfaced as first-class events with contract_id, producer, consumer, and breach severity — essential for federated data mesh architectures.
**Implementation**: `register_data_contract(tenant_id, contract)` stores contract spec (dataset_id, producer_team, consumer_teams, expectations list, sla, owner_contact). `evaluate_contract(tenant_id, contract_id, run_id)` cross-references run results against contract expectations, computes contract_compliance_score, emits `contract_breach` events for failures. Expose `list_contracts` and `get_contract_compliance_history`.
**Competitor Reference**: PayPal Open Data Contract Standard; Atlan data contracts; Soda data contracts.
