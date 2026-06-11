# World-Class Improvements — Project Baseline Management (ppm_pbl)

**Capability**: Project Baseline Management | **Version Target**: 2.0.0
**Date**: 2026-06-11 | **Author**: Nyimbi Odero

---

## 1. Integrated Baseline Review (IBR) Automation

**Current gap**: No cross-baseline consistency enforcement — scope, schedule, and cost baselines can exist independently with undetected mismatches (e.g., cost baseline doesn't cover all WBS elements in scope baseline).

**Improvement**: Add `integrated_baseline_review()` that cross-validates all three baseline types for a project, verifying WBS coverage parity, schedule-to-cost alignment, and resource loading consistency. Returns a scored IBR report with pass/fail by dimension and a composite IBR health index (0–100).

---

## 2. Rolling Wave Baseline Segments

**Current gap**: Baselines are monolithic snapshots. Projects using rolling wave or agile approaches need multi-horizon baselines where near-term work is fully planned and far-term work is a rolling estimate.

**Improvement**: Add `set_rolling_wave_baseline()` with `horizon_weeks` and `detail_level` parameters per horizon. Track "committed" vs "planning" segments separately. Enables EVM on the committed segment while preserving flexibility on the planning horizon.

---

## 3. Probabilistic Cost and Schedule Reserves

**Current gap**: Cost and schedule baselines capture deterministic values only. No modeling of management reserve vs contingency reserve, nor Monte Carlo confidence intervals.

**Improvement**: Add `set_reserve_analysis()` storing management reserve, contingency reserve, and P80/P90 confidence bounds. Integrate with `variance_analysis()` to report remaining reserve ratio alongside CPI/SPI. Critical for DCSA/EVMS compliance programs.

---

## 4. Time-Phased Budget Distribution (S-Curve)

**Current gap**: Cost baseline is a single total figure. EVM requires time-phased planned value — the S-curve — to compute meaningful SPI/CPI over time.

**Improvement**: Add `set_time_phased_budget()` accepting period-by-period PV distributions. `take_ev_snapshot()` then auto-computes the correct PV for the snapshot date from the S-curve, eliminating manual PV entry errors that corrupt EVM integrity.

---

## 5. Retroactive Integrity Audit Trail

**Current gap**: Audit events capture operation names but not before/after field diffs. A forensic review cannot reconstruct exactly what changed.

**Improvement**: Add structured `_audit_diff()` that captures `{field, before, after}` tuples for every mutating operation. Store diffs in a tamper-evident append-only log. Add `get_audit_trail()` method with filtering by date range, actor, and resource_id.

---

## 6. Baseline Lock Mechanism

**Current gap**: Approved baselines can be overwritten by calling `set_scope_baseline()` again. The version increment is informational only — no write protection.

**Improvement**: Add `lock_baseline()` and `unlock_baseline()` with explicit lock owner tracking. Locked baselines reject any mutation unless unlocked by an authorized actor. Integrate with the policy engine to enforce `baseline_locked` as a first-class rule condition.

---

## 7. Change Request Dependency Graph

**Current gap**: Change requests are independent records. In practice, CR-002 may be blocked by CR-001, or CR-003 may supersede CR-001 — these relationships are invisible.

**Improvement**: Add `link_change_requests()` with relationship types `blocks`, `depends_on`, `supersedes`, `relates_to`. Add `get_cr_dependency_graph()` returning a DAG structure. Change approval workflow then checks that blocking CRs are resolved before a dependent CR can be approved.

---

## 8. Earned Schedule (ES) Metrics

**Current gap**: Variance analysis computes SV and SPI in cost terms only (BCWP-based). Earned Schedule — a more accurate time-predictor for late projects — is absent.

**Improvement**: Extend `variance_analysis()` and `take_ev_snapshot()` to compute Earned Schedule (ES), Schedule Variance(t) = ES − AT, and SPI(t) = ES / AT. ES-based SPI(t) correctly converges to 1.0 at project completion unlike cost-based SPI. Add `es_forecast_completion_date()` using IEAC(t) = PD / SPI(t).

---

## 9. Multi-Baseline Portfolio View

**Current gap**: `baseline_analytics()` is project-scoped. Portfolio managers need a cross-project rolled-up view: aggregate BAC, ETC, VAC, and baseline health across all active projects.

**Improvement**: Add `portfolio_baseline_summary()` that aggregates all project baselines for the tenant, computing portfolio-level CPI, SPI, EAC, and a risk-tiered project list (red/amber/green by variance threshold breach count).

---

## 10. Automated Variance Threshold Escalation

**Current gap**: `variance_analysis()` returns a health colour but takes no action. Threshold breaches require manual follow-up.

**Improvement**: Add `configure_variance_escalation()` to define escalation rules: e.g., `{threshold: "red", consecutive_periods: 2, escalate_to: ["pmo_director"], action: "notify+freeze_cr"}`. Add `evaluate_escalation_rules()` called from `variance_analysis()` to auto-trigger notifications and optional CR freeze when rules fire.

---

## 11. Baseline Freeze Periods

**Current gap**: No mechanism to prevent change requests during critical periods (month-end reporting lock, contract performance report windows).

**Improvement**: Add `set_freeze_period()` with start/end dates and an optional scope filter (e.g., freeze cost baseline only). The `change_request()` method checks active freeze periods and rejects submissions with `BASELINE_FROZEN` unless the CR has `emergency` priority, matching EVMS practice.

---

## 12. WBS-Linked Scope Baseline

**Current gap**: Scope baseline stores deliverables as an unstructured list. No linkage to a formal Work Breakdown Structure, making EVM control account mapping impossible.

**Improvement**: Extend `set_scope_baseline()` to accept `wbs_elements: list[{id, name, level, parent_id, control_account}]`. Add `get_wbs_element()` and `list_control_accounts()`. Control accounts become the integration point for schedule (activities) and cost (budgets), enabling proper PMB construction.

---

## 13. Variance At Completion (VAC) Forecasting

**Current gap**: EV snapshots capture EAC but not VAC = BAC − EAC, nor TCPI (To-Complete Performance Index), which tells the team the CPI they must sustain to finish on budget.

**Improvement**: Compute and store VAC = BAC − EAC and TCPI = (BAC − EV) / (BAC − AC) for every EV snapshot. Add `forecast_completion()` returning EAC under three methods (BAC/CPI, AC+ETC_typical, AC+ETC_atypical) with a recommended method flag based on SPI/CPI stability over last 3 snapshots.

---

## 14. Baseline Deviation Score (BDS)

**Current gap**: No single composite metric captures overall project baseline health for quick portfolio triage. SPI and CPI require interpretation; threshold colours are coarse.

**Improvement**: Compute a Baseline Deviation Score (0–100, lower is better) combining SV%, CV%, scope change velocity (CRs/week), and time-since-last-rebase. Weights are configurable via `conf`. Add `get_baseline_deviation_scores()` returning BDS for all projects, ranked worst-first, for dashboard KPI tiles.

---

## 15. Baseline Version History and Diff

**Current gap**: Version numbers increment on rebase but prior baseline content is discarded. Post-project reviews and audits cannot reconstruct the original approved baseline vs final state.

**Improvement**: Add `save_baseline_version()` that snapshots the full baseline record (including WBS, time-phased budget, and all approved CRs) into an append-only version history. Add `diff_baseline_versions()` returning a structured diff (added/removed deliverables, budget delta by period, schedule delta by milestone) between any two version numbers. Mandatory before any rebase operation.
