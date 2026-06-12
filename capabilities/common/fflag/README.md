# Feature Flags (fflag)

Runtime feature toggles, percentage rollout, A/B experiment assignment, per-tenant targeting,
sticky bucketing, multi-armed bandit experiments, named segments, change-request approvals,
flag templates, ramp plans, import/export, statistical significance, and OTel-compatible telemetry.

## API

### Core Flag CRUD

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fflag/health | Service health |
| GET | /api/fflag/flags | List flags (filterable by enabled, tags, owner) |
| POST | /api/fflag/flags | Create flag |
| GET | /api/fflag/flags/{key} | Get flag |
| PUT | /api/fflag/flags/{key} | Update flag |
| DELETE | /api/fflag/flags/{key} | Delete flag |
| POST | /api/fflag/flags/{key}/enable | Enable flag |
| POST | /api/fflag/flags/{key}/disable | Disable flag |
| POST | /api/fflag/flags/{key}/rollout | Set rollout % |
| POST | /api/fflag/flags/{key}/clone | Clone flag to new key |
| GET | /api/fflag/flags/{key}/history | Flag change audit history |
| POST | /api/fflag/flags/{key}/targeting-rules | Append targeting rule |
| DELETE | /api/fflag/flags/{key}/targeting-rules/{index} | Remove targeting rule |

### Evaluation

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fflag/evaluate/{key} | Evaluate flag for user |
| GET | /api/fflag/evaluate/{key}/sticky | Evaluate with sticky bucketing |
| POST | /api/fflag/evaluate/batch | Batch evaluate named flags |
| POST | /api/fflag/evaluate/all | Evaluate all flags for user |
| POST | /api/fflag/evaluate/{key}/telemetry | Evaluate + emit OTel-compatible event |

### Per-User Overrides

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/overrides | Set user override |
| DELETE | /api/fflag/overrides | Clear user override |
| DELETE | /api/fflag/overrides/sticky | Clear sticky assignment |
| GET | /api/fflag/overrides | List overrides |

### Segments

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/segments | Create named targeting segment |
| GET | /api/fflag/segments | List segments |
| GET | /api/fflag/segments/{segment_id} | Get segment |
| DELETE | /api/fflag/segments/{segment_id} | Delete segment |

### A/B Experiments

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/experiments | Create experiment |
| GET | /api/fflag/experiments | List experiments |
| GET | /api/fflag/experiments/{id} | Get experiment |
| POST | /api/fflag/experiments/{id}/start | Start experiment |
| POST | /api/fflag/experiments/{id}/stop | Stop experiment (declare winner) |
| GET | /api/fflag/experiments/{id}/results | Variant assignment distribution |
| POST | /api/fflag/experiments/{id}/assign | Assign user to variant |
| POST | /api/fflag/experiments/{id}/significance | Compute statistical significance |

### Multi-Armed Bandit

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/experiments/{id}/bandit/outcome | Record conversion outcome |
| GET | /api/fflag/experiments/{id}/bandit/state | Beta distribution parameters per variant |

### Ramp Plans

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/flags/{key}/ramp | Set gradual rollout ramp plan |
| POST | /api/fflag/flags/{key}/ramp/advance | Advance ramp to next due step |

### Change Approvals

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/change-requests | Submit flag change for approval |
| POST | /api/fflag/change-requests/{id}/approve | Approve and apply change |
| POST | /api/fflag/change-requests/{id}/reject | Reject change request |

### Templates

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/fflag/templates | Create flag template |
| POST | /api/fflag/templates/{name}/instantiate | Instantiate template for a tenant |
| POST | /api/fflag/templates/{name}/apply-update | Push field updates to all derived flags |

### Operations

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/fflag/statistics | Flag statistics |
| GET | /api/fflag/audit | Audit trail |
| GET | /api/fflag/export | Export flags/segments/experiments |
| POST | /api/fflag/import | Import flags (merge/overwrite/dry_run) |

## Architecture

```
FeatureFlagService
├── flags{}              — flag records keyed "tenant:key"
├── segments{}           — named targeting cohorts keyed "tenant:segment_id"
├── experiments{}        — A/B and bandit experiments
├── assignments{}        — deterministic variant assignments
├── overrides{}          — per-user hard overrides
├── sticky_assignments{} — bucketing persistence (I8)
├── bandit_state{}       — Beta(α,β) per variant (I4)
├── templates{}          — cross-tenant flag blueprints (I15)
├── change_requests{}    — pending approval records (I11)
└── _audit_events[]      — full immutable audit trail
```

### Evaluation Priority

1. Per-user override (highest)
2. Targeting rules — first match wins (supports segment references)
3. Sticky assignment (if flag is sticky-bucketed)
4. Percentage rollout (consistent MD5 hash)
5. Flag disabled → `false`

### NATS Integration

Mutation events publish to `fflag.changes.{tenant_id}.{key}`.
Evaluation telemetry publishes to `fflag.telemetry.{tenant_id}`.
Ramp scheduler subscribes to `fflag.scheduler.tick`.
Set `NATS_URL=nats://localhost:4222` to activate.

## Evaluation Determinism

All bucketing uses `MD5(seed:user_id)` mapped to 0.0–100.0.  The same user
always receives the same flag result for an unchanged flag configuration.
Sticky bucketing persists the first positive result so rollout percentage
changes do not flip existing participants.

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Scheduled Flag Lifecycle (Temporal Triggers)** [Automation]
- **I2. NATS-Backed Real-Time Flag Change Propagation** [Distribution / Consistency]
- **I3. SDK-Style Cached Evaluation with Stale-While-Revalidate** [Performance]
- **I4. Multi-Armed Bandit Experiment Assignment** [Experimentation Science]
- **I5. Flag Dependency Graph with Circular-Dependency Detection** [Correctness / Safety]
- **I6. Contextual Attribute Schema Validation** [Data Integrity]
- **I7. Statistical Significance Calculator for Experiments** [Experimentation Science]
- **I8. Flag Stickiness Groups (Cohort Stability)** [UX Consistency]
- **I9. Gradual Rollout Schedules (Ramp Plans)** [Risk Management]
- **I10. Flag Segments (Reusable Targeting Cohorts)** [DX / Maintainability]
- **I11. Flag Change Approval Workflows** [Governance / Compliance]
- **I12. Evaluation Context Propagation (OpenTelemetry-Compatible)** [Observability]
- **I13. Tenant Flag Import/Export (Portable Configurations)** [Operations]
- **I14. Dynamic Sampling Rate Control** [Performance / Cost]
- **I15. Cross-Tenant Flag Templates** [Platform / Multi-Tenancy]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
