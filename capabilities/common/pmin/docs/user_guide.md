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
