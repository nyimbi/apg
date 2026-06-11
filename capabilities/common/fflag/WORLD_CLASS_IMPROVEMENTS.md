# Feature Flags — World-Class Improvements

15 high-impact improvements to elevate fflag from solid to production-grade.

---

### I1. Scheduled Flag Lifecycle (Temporal Triggers)
**Category**: Automation
**Justification**: Flags without time-bounds become permanent debt. LaunchDarkly reports >40% of flags outlive their intent by 6+ months. Scheduled activation/expiry enforces hygiene automatically and enables "dark launches" with precise rollout windows — eliminating engineer coordination overhead.
**Implementation**: Add `activation_time: datetime | None` and `expiry_time: datetime | None` fields to each flag. `evaluate_flag` checks UTC now against these bounds before rollout logic. A background NATS subscriber (subject `fflag.scheduler.tick`) drives a periodic sweep that auto-disables expired flags and emits `flag_expired` audit events.
**Competitor**: LaunchDarkly Scheduled Flag Changes, Split.io Rollout Automation

---

### I2. NATS-Backed Real-Time Flag Change Propagation
**Category**: Distribution / Consistency
**Justification**: In-memory state is siloed per-process. Any multi-instance deployment diverges within milliseconds of a flag change. NATS pub/sub gives sub-1ms fan-out to all service replicas with zero polling overhead — 10× lower propagation latency than Redis keyspace notifications.
**Implementation**: On every mutating method (`create_flag`, `update_flag`, `delete_flag`, etc.) publish to `fflag.changes.{tenant_id}.{key}` via `NATSEventAdapter`. Each service instance subscribes on startup and applies the delta to its local cache. Use optimistic locking (version counter) to reject stale updates.
**Competitor**: Unleash with Redis sync, Flipt with gRPC streaming, GrowthBook SSE

---

### I3. SDK-Style Cached Evaluation with Stale-While-Revalidate
**Category**: Performance
**Justification**: Flag evaluation at 10k RPS with DB-backed storage adds 2–5ms per call. A local in-process cache with SWR semantics reduces p99 to <0.1ms while tolerating a configurable staleness window. This is the architecture AWS AppConfig and LaunchDarkly SDKs use to serve billions of evaluations per day.
**Implementation**: Add `FlagCache` dataclass wrapping a `dict[str, tuple[dict, float]]` (value, expires_at). `evaluate_flag` checks cache first; on miss or stale hit triggers async background refresh via NATS request-reply. `cache_ttl_seconds: float = 5.0` is configurable per service instance.
**Competitor**: LaunchDarkly In-Memory Cache, AWS AppConfig deployment cache

---

### I4. Multi-Armed Bandit Experiment Assignment
**Category**: Experimentation Science
**Justification**: Static 50/50 A/B splits waste traffic on clearly losing variants. Thompson Sampling (Bayesian bandit) continuously reallocates traffic toward winning variants, converging to the optimal variant 3–5× faster than fixed splits. Used by Netflix, Booking.com, and Airbnb.
**Implementation**: Add `experiment_type: Literal["ab", "bandit"]` field. For bandit experiments, maintain per-variant Beta distribution parameters `(alpha, beta)` in `self.bandit_state`. `assign_experiment_variant` samples from each variant's Beta distribution and selects the highest draw. `record_bandit_outcome` updates the winning variant's alpha (success) or beta (failure) counter, then republishes updated weights to NATS.
**Competitor**: Netflix's Experimentation Platform, VWO SmartStats, Optimizely Stats Engine

---

### I5. Flag Dependency Graph with Circular-Dependency Detection
**Category**: Correctness / Safety
**Justification**: Complex feature systems develop implicit flag dependencies ("flag B only makes sense when flag A is on"). Without explicit dependency tracking, enabling B without A causes subtle bugs that take hours to diagnose. Explicit prerequisite modelling with DAG validation prevents entire classes of misconfiguration.
**Implementation**: Add `prerequisites: list[str]` to flag records. `evaluate_flag` resolves prerequisites recursively (depth-limited to 10) before evaluating the current flag. `create_flag` and `add_prerequisite` run a topological sort to detect cycles, raising `CyclicDependencyError` on detection. Prerequisites are published in the flag payload over NATS.
**Competitor**: LaunchDarkly Flag Prerequisites, Unleash Flag Dependencies

---

### I6. Contextual Attribute Schema Validation
**Category**: Data Integrity
**Justification**: `user_attributes` is an untyped `dict[str, Any]` — operators can reference attributes that never exist (typos, schema drift) causing silent rule mismatches. A per-flag attribute schema with type coercion and required-field validation catches targeting rule errors at authoring time, not at 2AM.
**Implementation**: Add `attribute_schema: dict[str, dict]` to flag records (JSON Schema subset: `type`, `required`, `enum`). `_matches_rule` validates and coerces each referenced attribute against the schema before comparison. `validate_flag_attributes` exposes this as a standalone async method. Schema violations surface as `SCHEMA_VIOLATION` audit events.
**Competitor**: Statsig's Conditions, LaunchDarkly Custom Attributes with type hints

---

### I7. Statistical Significance Calculator for Experiments
**Category**: Experimentation Science
**Justification**: Engineers stop experiments when they "look right" rather than at statistical significance, leading to underpowered conclusions. A built-in Chi-squared / Z-test calculator that signals readiness prevents both premature stops and indefinite running — a direct LaunchDarkly Experiment Insights equivalent.
**Implementation**: `compute_experiment_significance` takes conversion counts per variant and computes two-proportion Z-test p-value and 95% confidence intervals using pure Python (no scipy dependency, hand-implemented). Returns `significant: bool`, `p_value: float`, `required_sample_size: int` (power analysis at 80% power, 5% α). Threshold configurable per experiment via `significance_level` field.
**Competitor**: Optimizely Stats Engine, LaunchDarkly Experimentation, VWO SmartStats

---

### I8. Flag Stickiness Groups (Cohort Stability)
**Category**: UX Consistency
**Justification**: When rollout percentage changes from 10% to 20%, naive hashing can flip users who were already in the 10%. Sticky groups ensure users already assigned stay assigned — critical for checkout flows, onboarding sequences, or any multi-step user journey where mid-funnel variant changes destroy the experiment.
**Implementation**: Add `sticky: bool = False` to flags. When sticky is enabled, `evaluate_flag` writes a `sticky_assignment` record keyed by `tenant:flag_key:user_id` on first evaluation. Subsequent calls return the stored assignment regardless of percentage changes. `migrate_sticky_assignments` provides a sweep to re-assign users when a flag is restructured.
**Competitor**: LaunchDarkly Sticky Bucketing, Unleash Stickiness, GrowthBook Sticky Bucketing

---

### I9. Gradual Rollout Schedules (Ramp Plans)
**Category**: Risk Management
**Justification**: Manual percentage bumping requires human attention every step. A ramp plan (0% → 5% → 25% → 100% with configured dwell times) enables safe automated canary deployment without engineer babysitting. Correlates with error-rate monitoring to auto-halt on anomalies.
**Implementation**: Add `ramp_plan: list[dict]` to flags — each step has `percentage: float`, `at_time: str` (ISO8601) or `after_minutes: int`. `apply_ramp_step` is called by the scheduler tick NATS subscriber. If an anomaly signal is received on `fflag.anomaly.{tenant}.{key}`, the ramp halts and emits `ramp_halted` audit event.
**Competitor**: AWS CodeDeploy canary deployments, LaunchDarkly Progressive Rollout, Spinnaker Canary

---

### I10. Flag Segments (Reusable Targeting Cohorts)
**Category**: DX / Maintainability
**Justification**: Targeting rules duplicated across 50 flags for "internal-beta-users" become a maintenance nightmare — a user group change requires 50 flag updates. Named segments decouple cohort definition from flag configuration, mirroring how LaunchDarkly Segments and Unleash Segments reduce targeting rule count by 60–80%.
**Implementation**: Add `self.segments: dict[str, dict]` to the service. `create_segment` defines a named set of conditions (same `conditions` list format as targeting rules). Flags reference segments via `{"type": "segment", "segment_id": "beta-users"}` in their `targeting_rules`. `_matches_rule` resolves segment references before condition evaluation. Segments are tenant-scoped.
**Competitor**: LaunchDarkly Segments, Unleash Segments, Split.io Segments

---

### I11. Flag Change Approval Workflows
**Category**: Governance / Compliance
**Justification**: A misfire on a payment flag at 100% rollout can take down revenue. Approval workflows gate production changes behind a second actor — essential for SOC2 / ISO27001 compliance, where change management requires documented four-eyes checks. LaunchDarkly Enterprise reports 73% of enterprise customers require flag change approvals.
**Implementation**: Add `requires_approval: bool` and `approvers: list[str]` to flags. Mutations on approved-required flags create a pending `ChangeRequest` record instead of applying immediately. `approve_change_request` and `reject_change_request` methods drive the workflow. Only on approval does the actual mutation execute. All state published to NATS for notification fanout.
**Competitor**: LaunchDarkly Approvals, Statsig Approvals, Harness Feature Flags Approvals

---

### I12. Evaluation Context Propagation (OpenTelemetry-Compatible)
**Category**: Observability
**Justification**: Flag evaluations are invisible in distributed traces today. Injecting flag decisions as span attributes (standard OpenTelemetry semantic conventions) means engineers can filter production traces by "users in treatment variant" — cutting debug time for variant-correlated issues from hours to minutes.
**Implementation**: `evaluate_flag` accepts an optional `trace_context: dict[str, str] | None` parameter. When present, emits a structured evaluation event to NATS subject `fflag.telemetry.{tenant_id}` with trace_id, span_id, flag_key, variant, and reason. An OpenTelemetry-compatible collector can consume this stream and attach flag attributes to the originating trace span.
**Competitor**: LaunchDarkly OpenTelemetry hook, Statsig diagnostics, GrowthBook data pipeline

---

### I13. Tenant Flag Import/Export (Portable Configurations)
**Category**: Operations
**Justification**: Moving flag configurations between environments (dev → staging → prod) manually is error-prone and time-consuming. Structured export/import with schema validation and conflict detection enables GitOps-style flag management — flags as code, reviewable in pull requests.
**Implementation**: `export_flags(tenant_id)` serialises all flags, segments, and experiments to a canonical JSON envelope (versioned schema). `import_flags(tenant_id, data, mode)` supports `merge` (non-destructive), `overwrite` (full replace), and `dry_run` (validation only, no mutations) modes. Diff report shows added/updated/deleted flag keys.
**Competitor**: LaunchDarkly Terraform provider, Unleash state import/export, Flagsmith environment clone

---

### I14. Dynamic Sampling Rate Control
**Category**: Performance / Cost
**Justification**: Emitting a full audit event for every flag evaluation at 100k evaluations/sec generates gigabytes of useless data. Dynamic sampling (1-in-N for high-frequency flags, always-on for mutations) reduces storage cost by 99% while preserving full fidelity for changes and overrides.
**Implementation**: Add `evaluation_sample_rate: float = 0.0` to each flag (0.0 = no evaluation logging, 1.0 = log all). `evaluate_flag` uses `random.random() < flag["evaluation_sample_rate"]` to decide whether to emit to NATS telemetry. Mutations always emit regardless of sample rate. `set_flag_sample_rate` exposes this as a first-class operation.
**Competitor**: LaunchDarkly Data Export sampling, Statsig sampling, AWS CloudWatch EMF sampling

---

### I15. Cross-Tenant Flag Templates
**Category**: Platform / Multi-Tenancy
**Justification**: SaaS platforms deploying features across 500+ tenants need a way to push a standard flag configuration to all tenants simultaneously — the "platform default" pattern. Without templates, bootstrapping new tenants requires N individual API calls and diverges over time.
**Implementation**: `self.templates: dict[str, dict]` stores tenant-agnostic flag blueprints. `create_template(name, flag_spec)` stores the blueprint. `instantiate_template(tenant_id, template_name, overrides)` calls `create_flag` with template defaults merged with per-tenant overrides. `apply_template_update(template_name, actor)` re-applies updated template fields to all tenant instances derived from the template — tracking provenance via `template_source` field on each flag.
**Competitor**: LaunchDarkly Flag Templates (Enterprise), Statsig Gates, Harness Feature Flag Templates
