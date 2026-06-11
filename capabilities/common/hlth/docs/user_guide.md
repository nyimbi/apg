# HLTH User Guide

Practical guide for the APG Health Monitoring (HLTH) capability, covering
core workflows and the 2026 additions.  See `user-manual.md` for full
reference documentation.

---

## Quick Start

```python
import asyncio
from capabilities.common.hlth.service import HlthService

svc = HlthService()

# Register a component
svc.register_component(
    tenant_id="acme",
    component_id="orders-api",
    name="Orders API",
    component_type="service",
    environment="production",
    owner="platform-team",
    criticality="critical",
)

# Record a health check
check = svc.record_check(
    tenant_id="acme",
    component_id="orders-api",
    dimension="availability",
    score=92.5,
    summary="Availability within SLO",
)
```

All methods that accept `tenant_id` validate it via `guard_tenant_id` at the
boundary — an empty or whitespace tenant_id raises `ValueError` immediately.

---

## SLA Burn-Rate Tracking

Track how quickly your error budget is being consumed before the SLA is
breached.  The multi-window model detects both fast burns (large incident) and
slow burns (chronic degradation).

```python
result = await svc.get_sla_burn_rate(
    tenant_id="acme",
    slo_id="orders-api-availability",
    window_hours=1,   # 1 | 6 | 24 | 72
)

print(result["burn_rate"])               # e.g. 18.4 (fast burn)
print(result["fast_burn"])               # True when > 14x over 1h
print(result["budget_remaining_minutes"])# Decimal string, e.g. "42.1800"
print(result["projected_depletion"])     # ISO-8601 or None
print(result["decision"])                # "alert" | "warn" | "ok"
```

Budget remaining and projected depletion use `Decimal` arithmetic internally
to prevent floating-point accumulation errors across thousands of tenants.

---

## Blast-Radius Simulation

Before responding to a degradation, understand which downstream components are
at risk.

```python
result = await svc.simulate_failure_blast_radius(
    tenant_id="acme",
    component_id="orders-db",
    failure_mode="unavailable",  # | "degraded" | "slow"
)

for comp in result["affected_components"]:
    print(
        f"  {comp['component_id']}: score drops by "
        f"{comp['estimated_score_delta']} pts "
        f"(p={comp['probability']:.2f}, depth={comp['depth']})"
    )

print("Blast radius score:", result["blast_radius_score"])  # 0-100
print("Critical path hit:", result["critical_path_hit"])
```

The BFS is pruned at depth 6 and probability < 0.05 to remain fast even in
large dependency graphs.

---

## Chaos-Resilience Score

Understand how well a component would survive a fault before one happens.

```python
result = await svc.assess_chaos_resilience(
    tenant_id="acme",
    component_id="orders-api",
)

print("Resilience score:", result["resilience_score"], "/ 100")
for gap in result["gaps"]:
    print("  Gap:", gap)
for rec in result["recommendations"]:
    print("  Fix:", rec)
```

The score is broken down across five dimensions: circuit_breaker, retry,
timeout, redundancy, and recovery_speed.  Each is 0-20 points.

Register circuit-breaker and retry metadata via component tags to improve
the score:

```python
svc.register_component(
    ...,
    tags=["circuit_breaker", "retry", "production"],
    metadata={"replicas": 3, "timeout_ms": 5000},
)
```

---

## Certificate and Secret Expiry Tracking

Prevent outages from expired TLS certificates, API keys, and passwords.

```python
from datetime import datetime, timedelta

await svc.register_expiry_asset(
    tenant_id="acme",
    component_id="orders-api",
    asset_type="tls_cert",           # tls_cert | api_key | password | ca_cert | token
    expiry_date=datetime.utcnow() + timedelta(days=45),
    notify_days_before=30,
)
# Returns health_score=50.0 (halfway through the 30-day notification window)
# An alert is emitted immediately because the asset is within notify_days_before
```

The background task `_check_expiry_assets()` re-evaluates all registered assets
daily and emits `HlthAlertRecord` entries as expiry approaches.

---

## Alert Deduplication

Prevent notification storms by fingerprinting alerts before dispatch.

```python
fingerprint = await svc.compute_alert_fingerprint(alert)
# fingerprint = "a3f7c9d1e2b4..." (32-char hex)

# Use as a key in a TTL cache:
if fingerprint not in dedup_cache:
    dedup_cache[fingerprint] = {"count": 1, "last_seen": datetime.utcnow()}
    await dispatch_alert(alert)
else:
    dedup_cache[fingerprint]["count"] += 1
    if dedup_cache[fingerprint]["count"] > ESCALATION_THRESHOLD:
        await escalate_alert(alert)
```

---

## Health Score Explainability

Understand why a component received a particular score.

```python
result = await svc.explain_health_score(
    tenant_id="acme",
    component_id="orders-api",
)

print(result["summary"])
# "Score of 43.2/100. Largest drag: availability (-18.4). Top factors:
#  availability, performance, cascade_risk."

for factor in result["factors"]:
    print(
        f"  {factor['factor']}: {factor['contribution_delta']:+.2f} "
        f"({factor['direction']}) — {factor['description']}"
    )
```

---

## Deployment Regression Detection

Close the deploy-to-detect feedback loop.

```python
# 1. Register the deployment before it goes live
snap = await svc.register_deployment_event(
    tenant_id="acme",
    deployment_id="orders-api-v2.3.1",
    component_ids=["orders-api", "orders-worker"],
)
print("Pre-deploy scores:", snap["pre_deploy_scores"])

# 2. After the deployment window, check for regressions
result = await svc.check_deployment_health_regression(
    tenant_id="acme",
    deployment_id="orders-api-v2.3.1",
    window_minutes=15,
    regression_threshold_delta=10.0,
)

if result["overall_verdict"] == "regression_detected":
    for r in result["regressions"]:
        print(
            f"REGRESSION: {r['component_id']} "
            f"{r['pre_deploy_score']} -> {r['post_deploy_score']} "
            f"(delta={r['delta']}, severity={r['severity']})"
        )
```

---

## SLA Narrative Report

Generate executive-ready reports for weekly reviews and board packs.

```python
report = await svc.generate_sla_narrative_report(
    tenant_id="acme",
    period_days=30,
)

# Markdown for email / Confluence
print(report["markdown_report"])

# JSON for dashboard embedding
print(report["json_report"]["overall_uptime_pct"])  # Decimal string

for comp in report["json_report"]["per_component"]:
    print(
        f"  {comp['component_name']}: {comp['uptime_pct']}% "
        f"({'OK' if comp['sla_met'] else 'BREACH'}) "
        f"{comp['incident_count']} incidents"
    )
```

---

## Financial Cost Precision

Any method that computes financial impact (SLA burn-rate budget, cost-per-hour
assessments) uses `Decimal` with `ROUND_HALF_UP` at 4dp.  Do not coerce
these values to `float` before storing or summing them — use `Decimal`
arithmetic throughout the billing pipeline.

```python
from decimal import Decimal

budget = Decimal(report["json_report"]["overall_uptime_pct"])
sla_target = Decimal("99.9")
gap = sla_target - budget
print(f"SLA gap: {gap} percentage points")
```

---

## Tenant Isolation

Every public method validates `tenant_id` via `guard_tenant_id` from
`capabilities.common.reliability`.  Passing an empty string, `None`, or a
value longer than 128 characters raises `ValueError` before any data is
accessed.  Cross-tenant data access is structurally prevented by keying all
internal stores on `tenant_id`.

---

## See Also

- `docs/api-reference.md` — full method signatures
- `docs/architecture.md` — system design and adapter boundaries
- `docs/getting-started.md` — installation and first run
- `SPECIFICATION.md` — guardrail rules and contract
- `WORLD_CLASS_IMPROVEMENTS.md` — roadmap of 15 planned enhancements
