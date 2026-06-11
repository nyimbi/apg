# Resource Management — World-Class Improvements

**Capability**: `ppm_res` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Skill Taxonomy with Ontology Graph

Replace flat string skill matching with an RDF-style skill ontology. Skills inherit from parent nodes (e.g. `react` isa `frontend` isa `software_engineering`). Queries against `frontend` automatically surface React, Vue, and Angular resources. Eliminates the current fragile 4-char prefix fuzzy match.

**Impact**: Reduces gap misclassification by ~40%; enables semantic skill search via embedding similarity.

---

## 2. Continuous Utilisation Time-Series

Replace per-period snapshots with a true time-series store (daily granularity). Enables rolling-window utilisation trends, anomaly detection, and burn-rate projections rather than static monthly snapshots.

**Impact**: Enables predictive over-allocation alerts 2-3 weeks before breach.

---

## 3. Constraint-Based Optimal Assignment (CP-SAT)

Replace greedy first-N matching in `team_builder` with a constraint programming solver (OR-Tools CP-SAT). Optimises simultaneously for skill coverage, utilisation balance, cost minimisation, and leave avoidance.

**Impact**: 15-25% reduction in team bench cost; guaranteed feasibility over infeasible greedy result.

---

## 4. Real-Time Over-Allocation Webhook Delivery

Current over-allocation detection is pull-based (detected at `assign_resource` call time). Add a background monitor that fires webhook events to `ntfy` when cumulative allocation crosses 80%, 95%, and 100% thresholds without requiring an explicit check.

**Impact**: Eliminates silent over-allocation that slips through when allocations are created by different actors.

---

## 5. Evidence-Verified Skill Proficiency Pipeline

Replace the boolean `evidence_reference` field with an async verification pipeline: submitted evidence (cert URL, manager attestation, test score) enters a verification queue; proficiency is held at `claimed` status until verification completes, then transitions to `verified`. Prevents skill fabrication at the model layer, not just the policy layer.

**Impact**: Eliminates the current weakness where any non-empty string passes the `evidence_present` check.

---

## 6. Multi-Currency Cost Normalisation

`CostRate` stores a single currency string with no conversion logic. Add a `CurrencyNormalisationService` integration that converts all rates to a tenant base currency at time-of-record exchange rates. Portfolio-level cost analytics become meaningful when resources span geographies.

**Impact**: Enables accurate cross-geography capacity cost comparisons; essential for global delivery models.

---

## 7. Leave Impact Propagation

When a leave record is created, automatically identify all active allocations during that period and compute impacted tasks, project delay risk, and backfill cost. Publish `leave_impact_computed` events to allow `ppm_pps` to reschedule automatically.

**Impact**: Eliminates manual re-planning effort triggered by leave entries.

---

## 8. Resource Carbon / Sustainability Accounting

Add `carbon_kg_per_day` to `Resource` for equipment and facility types. The allocation engine can then report carbon footprint alongside cost, feeding ESG reporting dashboards. Increasingly required for public-sector and enterprise RFPs.

**Impact**: Differentiator capability; unlocks compliance with CSRD and similar mandates.

---

## 9. Probabilistic Demand Forecasting

Replace deterministic `forecast_demand_fte = sum(active_allocation_pct / 100)` with a Monte Carlo simulation over historical allocation variance and pipeline win rates. Output P50/P80/P90 FTE demand bands rather than a single point estimate.

**Impact**: Allows hiring decisions with explicit confidence intervals; reduces reactive over/under-hiring cycles.

---

## 10. Skill Endorsement Social Graph

Add a `SkillEndorsement` model: any resource with `proficiency_level >= expert` on skill X can endorse a colleague's claim to skill X. Endorsement count and network centrality become additional proficiency signals. Mirrors LinkedIn-style peer validation without external dependency.

**Impact**: Reduces cold-start problem for new resources; provides distributed evidence beyond manager attestation.

---

## 11. Allocation Marketplace (Internal Gig Board)

Resources with bench time > 30% can post availability to an internal marketplace. Project managers browse and request resources; notifications fire to bench resource's manager. This converts bench time from invisible cost to visible opportunity.

**Impact**: Estimated 10-15% reduction in bench cost through proactive utilisation.

---

## 12. Cost Rate Versioning with Bi-Temporal Model

`CostRate` has a single `effective_date` field. Adopt bi-temporal modelling: `valid_time` (when the rate is contractually valid) and `transaction_time` (when it was recorded in the system). Enables retroactive corrections without losing audit trail of what was believed to be true at decision time.

**Impact**: Eliminates re-computation bugs in historical cost reports when backdated rate changes occur.

---

## 13. Role-Based Capacity Pools

Group resources by role/grade into capacity pools (e.g. `senior_engineer`, `architect`). Allocations can target a pool rather than a named individual; the engine resolves the best available member at scheduling time. Decouples project planning from individual availability.

**Impact**: Eliminates the planning bottleneck where project managers cannot plan until specific people are identified.

---

## 14. Automated Capacity Plan Generation

`create_capacity_plan` currently accepts pre-serialised JSON strings for `demand_data` and `supply_data`. Add `generate_capacity_plan` which computes these fields automatically from live allocation and forecast data, then applies configurable gap-closure strategies (hire, contract, re-deploy, train) with timeline and cost projections.

**Impact**: Reduces capacity planning cycle from days to minutes; makes plans a living artifact rather than a point-in-time document.

---

## 15. RBAC-Integrated Delegation Chains

The service uses a flat `actor_id` field. Replace with a delegation chain model: a resource manager can delegate allocation authority to a team lead for a specific project and date range. The `_enforce` method validates the full delegation chain at policy evaluation time. Prevents authority creep and supports SOX-compliant resource governance.

**Impact**: Required for enterprise compliance; eliminates the current trust-on-presentation of `actor_id`.

---

*© 2025 Datacraft. All improvements are additive and backward-compatible with the existing service contract.*
