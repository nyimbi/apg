# HLTH Capability - World Class Improvements

15 targeted improvements to elevate the Health Monitoring capability to production-grade quality.

---

## 1. SLA Burn-Rate Tracking

Track error-budget consumption over rolling windows (1h, 6h, 24h, 72h) using the multi-window burn-rate algorithm from the SRE workbook. Emit fast-burn alerts when the 1h window exceeds 14x the allowed error rate. Current implementation only checks point-in-time thresholds — it misses slow burns that accumulate over days before breaching the SLA.

## 2. Dependency Health Propagation with Circuit Breaker State

When a dependency enters UNHEALTHY status, propagate a synthetic degraded score to all dependents using a weighted fan-out. Integrate circuit-breaker state (CLOSED / OPEN / HALF_OPEN) so dependents distinguish between "dependency slow" and "dependency tripped open." Currently, dependent components silently absorb cascading failures with no causal linkage in the health record.

## 3. Percentile-Based Latency Health Scoring

Replace mean-based response time scoring with p50/p95/p99 percentile tracking per component. A component whose mean latency is 200ms but whose p99 is 8s is not healthy. Store rolling histograms (using t-digest or HDR histogram) and score against percentile SLO targets rather than averages.

## 4. Composite Health Check Groups

Allow health checks to be grouped into logical units (e.g., "payment stack", "auth cluster") with aggregate pass/fail logic (ALL_MUST_PASS, MAJORITY_MUST_PASS, AT_LEAST_ONE_MUST_PASS). Report group-level health alongside component-level health. Enables deployment gates to block on logical service health rather than individual component health.

## 5. Adaptive Threshold Tuning via Bayesian Updating

Replace static thresholds with Bayesian-updated posteriors. Each new observation shifts the prior based on the posterior predictive distribution. After N observations the threshold narrows toward the true baseline. Prevents both premature alerting on initial deployment and threshold drift after years of operation.

## 6. Correlated Root Cause Surfacing via Granger Causality

When multiple components degrade simultaneously, apply Granger causality tests to the health time-series to identify which component degraded first and with what lag. Return a ranked list of root-cause candidates with confidence scores. Current correlation code counts co-occurring alerts but does not establish temporal causation direction.

## 7. On-Call Schedule Integration for Escalation Routing

Read on-call schedules from PagerDuty / OpsGenie / plain JSON rotation files and use the current on-call engineer as the first escalation target, falling back to the team lead. Route notifications through the preferred channel of the on-call person. Eliminates hardcoded escalation paths that send alerts to engineers who are off-shift.

## 8. Canary and Blue/Green Deployment Health Differentiation

Tag health checks with `deployment_slot` (canary, blue, green, stable) so the service can compare canary health scores against stable traffic health scores in real time. Automatically promote or roll back canary traffic based on health delta thresholds. Currently, canary and stable traffic share component identity with no slot differentiation.

## 9. Cost-Aware Remediation Prioritisation

Before triggering auto-remediation, estimate the cost of inaction (revenue per minute at current degradation rate) versus the risk-adjusted cost of the remediation action (blast radius * historical failure rate). Only execute remediations whose expected value is positive. Prevents expensive scale-out events for low-revenue components during off-peak hours.

## 10. Health Evidence Ledger with Merkle Chaining

Hash each `HlthCheckRecord` and chain it to the previous hash (Merkle-style), storing the chain root in the tenant audit log. Auditors can verify that no historical health record was modified after the fact without touching live state. Required for SOC 2 Type II and ISO 27001 audit trails.

## 11. Synthetic Transaction Monitoring Integration

Issue synthetic HTTP/gRPC probes on a configurable cadence and feed the results back as `HealthMetric` objects with `source=synthetic`. Distinguish synthetic from real traffic in scoring so that a component with 100% synthetic availability but 0% real-traffic availability is correctly scored as unhealthy.

## 12. MTTR and MTBF Calculation per Component

Maintain rolling mean-time-to-repair and mean-time-between-failures per component using incident open/close timestamps. Surface these as reliability KPIs in the health report. Expose MTBF trends so degrading reliability (shortening MTBF) triggers a predictive alert before the next failure occurs.

## 13. Kubernetes / Nomad Native Health Source Adapter

Consume pod ready/not-ready events, liveness probe failures, and OOMKill events directly from the cluster API using a watch stream. Translate these into typed `HealthMetric` events without requiring application-side instrumentation. Currently requires all metrics to be pushed in; pull-based cluster observation is absent.

## 14. Health Score Caching with Cache Stampede Prevention

Replace the current dict-based cache with a probabilistic early expiry (PER) cache. Entries are refreshed slightly before expiry with probability proportional to how close they are to expiry, preventing the thundering-herd problem when many cached scores expire simultaneously under load. Cache miss latency spikes are eliminated without sacrificing freshness.

## 15. Multi-Region Health Aggregation with Split-Brain Detection

Collect health scores from agents running in multiple regions and aggregate them with a configurable quorum policy (ANY, MAJORITY, ALL). Detect split-brain conditions where region A reports a component healthy and region B reports it unhealthy, and emit a split-brain alert rather than masking the disagreement behind an average.
