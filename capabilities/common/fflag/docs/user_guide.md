# Feature Flags User Guide

## Overview

The Feature Flags capability (`fflag`) provides runtime feature toggles with percentage rollout,
A/B experiment assignment, per-user targeting rules, sticky bucketing, multi-armed bandit
experiments, change-request approvals, cross-tenant templates, gradual ramp plans, and a
complete immutable audit trail.

All evaluation is deterministic — the same user always gets the same result for an unchanged
flag configuration.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Flag** | Named boolean toggle with optional variants and rollout % |
| **Segment** | Named reusable set of targeting conditions shared across flags |
| **Rollout** | % of users receiving the flag (consistent hash bucketing) |
| **Targeting Rule** | User-attribute conditions that override rollout (first match wins) |
| **Override** | Hard per-user flag assignment; bypasses all other logic |
| **Sticky Assignment** | Persisted bucket — users stay in their variant even as % changes |
| **Experiment** | A/B test or bandit attached to a flag with weighted variants |
| **Ramp Plan** | Scheduled step-wise rollout (canary → general availability) |
| **Change Request** | Four-eyes gate before applying a flag mutation in production |
| **Template** | Tenant-agnostic flag blueprint for platform-level defaults |

---

## Evaluation Priority

```
1. Per-user override           ← highest priority
2. Targeting rules             ← first matching rule wins
3. Sticky assignment           ← if flag has sticky=True and user was previously bucketed
4. Percentage rollout          ← consistent MD5 hash of user_id
5. Flag disabled               ← returns enabled=false
```

---

## Quickstart

### Create and evaluate a basic flag

```http
POST /api/fflag/flags
{
  "tenant_id": "acme",
  "key": "new_checkout_flow",
  "name": "New Checkout Flow",
  "enabled": true,
  "rollout_percentage": 25.0,
  "owner": "payments-team"
}

GET /api/fflag/evaluate/new_checkout_flow?tenant_id=acme&user_id=user-123
```

```json
{
  "flag_key": "new_checkout_flow",
  "enabled": true,
  "variant": null,
  "reason": "rollout",
  "targeting_matched": false
}
```

### Add a variant flag (multivariate)

```http
PUT /api/fflag/flags/new_checkout_flow
{
  "tenant_id": "acme",
  "variants": {
    "control":   {"weight": 50, "config": {"cta": "Buy Now"}},
    "treatment": {"weight": 50, "config": {"cta": "Checkout Fast"}}
  }
}
```

---

## Targeting

### Add a targeting rule

Rules are evaluated before rollout.  First match wins.

```http
POST /api/fflag/flags/new_checkout_flow/targeting-rules
{
  "tenant_id": "acme",
  "rule": {
    "conditions": [
      {"attribute": "user_tier", "operator": "eq", "value": "beta"}
    ],
    "enabled": true,
    "variant": "treatment"
  }
}
```

Supported operators: `eq`, `in`, `not_in`, `gt`, `lt`, `contains`.

### Named segments

Define once, reference from many flags:

```http
POST /api/fflag/segments
{
  "tenant_id": "acme",
  "segment_id": "beta-users",
  "name": "Beta Users",
  "conditions": [
    {"attribute": "user_tier", "operator": "eq", "value": "beta"}
  ]
}
```

Reference in a targeting rule:

```json
{
  "targeting_rules": [
    {"type": "segment", "segment_id": "beta-users", "enabled": true, "variant": "treatment"}
  ]
}
```

---

## Sticky Bucketing

Sticky bucketing ensures users who receive a flag stay in their variant even if the rollout
percentage is later changed — essential for multi-step flows.

```http
GET /api/fflag/evaluate/new_checkout_flow/sticky?tenant_id=acme&user_id=user-123
```

On first call the result is computed and persisted.  Subsequent calls return the stored result
with `"reason": "sticky_assignment"`.

To reset a user (e.g. after a major flag redesign):

```http
DELETE /api/fflag/overrides/sticky
{"tenant_id": "acme", "key": "new_checkout_flow", "user_id": "user-123"}
```

---

## A/B Experiments

### Create, start, and assign

```http
POST /api/fflag/experiments
{
  "tenant_id": "acme",
  "flag_key": "new_checkout_flow",
  "name": "Checkout A/B Test",
  "variants": [
    {"key": "control",   "weight": 50},
    {"key": "treatment", "weight": 50}
  ]
}

POST /api/fflag/experiments/{id}/start

POST /api/fflag/experiments/{id}/assign
{"user_id": "user-456"}
```

Assignment is deterministic — calling assign twice returns the same variant.

### Statistical significance

Once you have impression and conversion counts, check readiness:

```http
POST /api/fflag/experiments/{id}/significance
{
  "tenant_id": "acme",
  "conversions": {"control": 120, "treatment": 145},
  "totals":       {"control": 800, "treatment": 790},
  "significance_level": 0.05
}
```

```json
{
  "significant": true,
  "p_value": 0.0312,
  "relative_lift": 0.1521,
  "confidence_interval_95": {"lower": 0.012, "upper": 0.058},
  "required_sample_size_per_arm": 612
}
```

---

## Multi-Armed Bandit

For faster convergence on the winning variant, use Thompson Sampling (Bayesian bandit):

```http
POST /api/fflag/experiments
{
  "tenant_id": "acme",
  "flag_key": "homepage_hero",
  "name": "Hero Image Bandit",
  "experiment_type": "bandit",
  "variants": [
    {"key": "image_a", "weight": 33},
    {"key": "image_b", "weight": 33},
    {"key": "image_c", "weight": 34}
  ]
}
```

After each user action, record the outcome:

```http
POST /api/fflag/experiments/{id}/bandit/outcome
{
  "user_id": "user-789",
  "variant_key": "image_b",
  "converted": true
}
```

Inspect current Beta posteriors (drives allocation):

```http
GET /api/fflag/experiments/{id}/bandit/state
```

```json
{
  "variant_states": {
    "image_a": {"alpha": 12, "beta": 88, "mean_conversion_rate": 0.12},
    "image_b": {"alpha": 34, "beta": 66, "mean_conversion_rate": 0.34},
    "image_c": {"alpha": 8,  "beta": 92, "mean_conversion_rate": 0.08}
  }
}
```

---

## Gradual Rollout Ramp Plans

Avoid manual percentage bumping.  Define a ramp and let the scheduler drive it:

```http
POST /api/fflag/flags/new_checkout_flow/ramp
{
  "tenant_id": "acme",
  "steps": [
    {"percentage": 5,   "after_minutes": 0},
    {"percentage": 25,  "after_minutes": 60},
    {"percentage": 50,  "after_minutes": 360},
    {"percentage": 100, "after_minutes": 1440}
  ]
}
```

The NATS scheduler subscriber calls `advance_ramp` on each tick.  Alternatively, advance
manually:

```http
POST /api/fflag/flags/new_checkout_flow/ramp/advance?tenant_id=acme
```

---

## Change-Request Approvals

For flags with `requires_approval: true`, route changes through the approval workflow:

```http
POST /api/fflag/change-requests
{
  "tenant_id": "acme",
  "key": "kill_switch_payments",
  "proposed_changes": {"enabled": false},
  "requestor": "alice",
  "reason": "Degraded API response times"
}
```

An approver reviews and applies:

```http
POST /api/fflag/change-requests/{request_id}/approve
{"approver": "bob"}
```

Or rejects:

```http
POST /api/fflag/change-requests/{request_id}/reject
{"rejector": "bob", "rejection_reason": "Metrics look normal now"}
```

---

## Cross-Tenant Templates

Roll out a standard flag configuration to multiple tenants:

```http
POST /api/fflag/templates
{
  "name": "dark-mode-flag",
  "description": "Standard dark mode feature flag",
  "flag_spec": {
    "key": "dark_mode",
    "name": "Dark Mode",
    "enabled": false,
    "rollout_percentage": 0.0,
    "tags": ["ui", "platform"]
  }
}
```

Instantiate for each tenant (tenant-specific overrides supported):

```http
POST /api/fflag/templates/dark-mode-flag/instantiate
{
  "tenant_id": "acme",
  "overrides": {"rollout_percentage": 50.0}
}
```

Push an update to all tenant instances at once:

```http
POST /api/fflag/templates/dark-mode-flag/apply-update
{
  "field_updates": {"rollout_percentage": 100.0},
  "actor": "platform-team"
}
```

---

## Import / Export (GitOps)

### Export for storage in VCS

```http
GET /api/fflag/export?tenant_id=acme
```

Returns a versioned JSON envelope with all flags, segments, and experiments.

### Import into another environment

```http
POST /api/fflag/import
{
  "tenant_id": "acme-staging",
  "mode": "dry_run",
  "data": { ...export envelope... }
}
```

Modes:
- `dry_run` — report what would change, apply nothing
- `merge` — add missing flags, leave existing untouched
- `overwrite` — full replacement of tenant's flags

---

## Telemetry (OpenTelemetry-Compatible)

Emit flag decisions as structured events consumable by an OTel collector:

```http
POST /api/fflag/evaluate/new_checkout_flow/telemetry
{
  "tenant_id": "acme",
  "user_id": "user-123",
  "trace_context": {
    "trace_id": "abc123",
    "span_id": "def456"
  },
  "sample_rate": 0.1
}
```

Events publish to NATS subject `fflag.telemetry.{tenant_id}`.  A collector can attach
`flag_key`, `variant`, and `reason` as span attributes to the originating distributed trace.

Use `sample_rate < 1.0` for high-frequency flags to cap data volume.

---

## NATS Integration

Set `NATS_URL=nats://localhost:4222`.

| Subject | Description |
|---------|-------------|
| `fflag.changes.{tenant}.{key}` | Flag mutation events (create/update/delete) |
| `fflag.telemetry.{tenant}` | Evaluation telemetry (sampled) |
| `fflag.scheduler.tick` | Drives ramp advancement and expiry sweeps |
| `fflag.anomaly.{tenant}.{key}` | External anomaly signal → halts active ramp |

---

## Audit Trail

Every mutation emits an immutable audit event with `before`/`after` snapshots:

```http
GET /api/fflag/audit?tenant_id=acme

GET /api/fflag/flags/new_checkout_flow/history?tenant_id=acme
```

Event types: `flag_created`, `flag_updated`, `flag_deleted`, `flag_imported`,
`targeting_rule_added`, `targeting_rule_removed`, `override_set`, `override_cleared`,
`sticky_assignment_cleared`, `experiment_created`, `experiment_started`, `experiment_stopped`,
`bandit_outcome_recorded`, `segment_created`, `segment_deleted`, `change_request_created`,
`change_request_approved`, `change_request_rejected`, `ramp_plan_set`, `ramp_step_applied`,
`template_created`, `template_update_applied`.
