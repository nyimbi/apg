# Process Mining User Guide

## Overview

The Process Mining capability (`pmin`) automatically infers BPMN process models from event streams (NATS or direct ingestion). It provides conformance checking to measure how well real process executions match the discovered model, bottleneck analysis, and variant discovery to find the most common and deviant execution paths.

## Key Concepts

- **Event Log**: a named stream of process events, each with a `case_id`, `activity`, `timestamp`, and optional `resource`
- **BPMN Model**: a Directly-Follows Graph (DFG) or Alpha-mined model showing the process flow
- **Variant**: a unique sequence of activities for a case (the "happy path" is the most frequent)
- **Conformance**: how closely the observed process matches the discovered model (fitness 0–1)
- **Bottleneck**: transitions with the highest average waiting time

## Supported Discovery Algorithms

| Algorithm | Description |
|-----------|-------------|
| `alpha_miner` | Classic α-algorithm — precise, sensitive to noise |
| `heuristics_miner` | Frequency-based, noise-tolerant |
| `inductive_miner` | Guarantees sound process tree |
| `directly_follows` | DFG only — fastest, least expressive |

## Quickstart

### 1. Create an event log

```http
POST /api/pmin/logs
{
  "tenant_id": "acme",
  "name": "Order Fulfillment",
  "subject_filter": "orders.>",
  "case_id_field": "order_id",
  "activity_field": "event_type"
}
```

### 2. Ingest events

```http
POST /api/pmin/logs/{log_id}/events
{
  "tenant_id": "acme",
  "events": [
    {"order_id": "O-001", "event_type": "Order Received", "timestamp": "2026-06-01T08:00:00Z"},
    {"order_id": "O-001", "event_type": "Payment Verified", "timestamp": "2026-06-01T08:15:00Z"},
    {"order_id": "O-001", "event_type": "Shipped", "timestamp": "2026-06-01T10:00:00Z"},
    {"order_id": "O-001", "event_type": "Delivered", "timestamp": "2026-06-02T14:00:00Z"}
  ]
}
```

### 3. Discover the BPMN model

```http
POST /api/pmin/logs/{log_id}/discover
{"tenant_id": "acme", "algorithm": "heuristics_miner", "noise_threshold": 0.2}
```

### 4. Check conformance

```http
POST /api/pmin/conformance
{"tenant_id": "acme", "log_id": "...", "model_id": "..."}
```

Returns `fitness`, `precision`, `generalization`, `simplicity`, and `deviating_cases`.

### 5. Find bottlenecks

```http
POST /api/pmin/logs/{log_id}/bottlenecks
{"tenant_id": "acme", "top_n": 5}
```

### 6. Discover variants

```http
POST /api/pmin/logs/{log_id}/variants
{"tenant_id": "acme", "top_n": 20}
```

Returns the `happy_path` (most frequent sequence) and all variant frequencies.

### 7. Simulate the model

```http
POST /api/pmin/models/{model_id}/simulate
{"tenant_id": "acme", "simulation_cases": 500}
```

Returns `completion_rate` and average trace length from Monte Carlo simulation.

## NATS Integration

Events published to NATS subjects matching `subject_filter` can be ingested directly:

```http
POST /api/pmin/logs/{log_id}/events/nats
{
  "tenant_id": "acme",
  "messages": [
    {"data": {"order_id": "O-002", "event_type": "Order Received", "timestamp": "..."}}
  ]
}
```

---

## Advanced Features (v1.1)

### SLA / KPI Breach Alerting

Configure per-log SLA rules and scan for breaches after each ingestion batch.

```python
# Configure rules
await svc.configure_sla_rules("acme", log_id, rules=[
    {"name": "Payment T+2", "activity": "Payment Verified",
     "max_duration_s": 172800, "scope": "transition"},
    {"name": "Total case 5 days", "activity": "Order Received",
     "max_duration_s": 432000, "scope": "case"},
])

# Scan for breaches
result = await svc.check_sla_breaches("acme", log_id)
# result["breaches"][0] → {rule_name, breach_count, breach_rate, breaching_cases}
```

Two scopes are supported:
- `transition` — gap between the named activity and the *next* activity in the case
- `case` — total case duration from first to last event

---

### Predictive Completion Time

Estimate how long open cases will take to complete based on the empirical distribution of
historical cases that followed the same activity prefix.

```python
result = await svc.predict_completion_time("acme", log_id, partial_traces=[
    {
        "case_id": "O-999",
        "activities": ["Order Received", "Payment Verified"],
        "started_at": "2026-06-11T08:00:00Z",
    }
])
# result["predictions"][0] →
#   {remaining_p50_s, remaining_p75_s, remaining_p95_s, matched_historical_cases}
```

Shorter prefix fallback is applied automatically when no exact match exists.

---

### Happy-Path Alignment Score

Quantify how closely each case follows the happy path (most frequent variant) using
Levenshtein edit distance.

```python
result = await svc.compute_happy_path_alignment("acme", log_id)
# result["avg_alignment_score"]  → e.g. 0.87
# result["most_deviant_cases"]   → bottom 10% of cases by alignment
```

Score of 1.0 = perfect match; 0.0 = completely unrelated sequence.

---

### Deviation Root-Cause Analysis

After running `check_conformance`, identify which case attributes statistically explain
why some cases deviate from the model.

```python
result = await svc.analyze_deviation_root_causes("acme", log_id, model_id)
# result["top_drivers"] →
#   [{attribute, value, lift, p_value_approx, count_in_deviating, count_in_conforming}, ...]
```

Ranked by lift (ratio of deviation rate to conformance rate). Run `check_conformance` first.

---

### Process Cost Analysis

Compute per-activity and per-variant costs by combining resource hourly rates with
transition durations. All monetary values use `Decimal` for precision.

```python
result = await svc.analyze_process_costs("acme", log_id, resource_rates={
    "agent_tier1": "45.00",
    "agent_tier2": "90.00",
    "default": "60.00",
})
# result["total_process_cost"]    → "12450.00"
# result["activity_costs"]["Underwriting"]["avg_cost_per_case"] → "320.50"
```

A `"default"` key is used as the fallback rate for resources not listed.

---

### Multi-Log Process Comparison

Compare two event logs from the same tenant to surface structural differences — useful
for benchmarking sites, periods, or product lines.

```python
result = await svc.compare_event_logs("acme", log_id_region_a, log_id_region_b)
# result["jaccard_activity_similarity"]  → 0.82
# result["jaccard_edge_similarity"]      → 0.67
# result["edges_only_in_a"]             → [{"edge": "Review → Approve", "frequency_a": 120}]
# result["top_duration_divergences"]    → ranked list of shared edges with time deltas
```

---

### Streaming Conformance

Maintain per-case running conformance state. Push new events as they arrive; the first
deviation per case emits a `conformance_deviation` NATS audit event.

```python
result = await svc.update_streaming_conformance(
    "acme", log_id, model_id,
    new_events=[
        {"order_id": "O-100", "event_type": "Shipped"},
        {"order_id": "O-101", "event_type": "Cancelled"},  # out-of-model
    ]
)
# result["newly_deviating_this_batch"] → ["O-101"]
# result["currently_deviating"]        → 1
```

State is persisted on the log record so subsequent calls are additive.

---

### Case Attribute Enrichment and Segmented Analysis

Attach business dimensions to cases, then re-run analyses on a filtered subset.

```python
# Step 1 — enrich
await svc.enrich_case_attributes("acme", log_id, case_attributes={
    "O-001": {"region": "APAC", "tier": "gold", "amount": 15000},
    "O-002": {"region": "EMEA", "tier": "standard", "amount": 800},
})

# Step 2 — segment
result = await svc.segment_analysis(
    "acme", log_id,
    segment_filter={"region": "APAC"},
    analysis_type="bottlenecks",
)
# result["matched_cases"] → 1
# result["result"]        → standard bottleneck report scoped to APAC cases
```

`analysis_type` is either `"variants"` or `"bottlenecks"`.
