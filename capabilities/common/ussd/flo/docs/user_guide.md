# ussd_flo User Guide

## Overview

`ussd_flo` is the APG USSD Flow Designer. It provides a graph-based USSD menu builder with:

- Visual node/edge flow definition
- Conditional routing between nodes
- Multi-language translation support with completeness checking
- Point-in-time version snapshots and structural diffs
- A/B test framework for comparing two flow variants
- Session simulation (dry-run without deploying to `ussd_eng`)
- Screen-budget validation (182-byte USSD page limit)
- Usability scoring with complexity metrics
- Dead-path analysis and pruning advisor
- Live session recording and replay for debugging
- Service-code migration with dry-run and auto-snapshot rollback

Flows defined in `ussd_flo` are exported and loaded into `ussd_eng` for runtime execution.

---

## Concepts

### Flow

A flow is a directed graph of nodes connected by edges. It has a single root node (entry point)
and one or more end nodes. Lifecycle: `draft` → `active` → `archived`.

Activate via `POST /flows/<id>/activate` — runs structural validation first; refuses to activate
if the root node is missing or the flow has zero nodes.

### Nodes

Five node types:

| Type | Description |
|------|-------------|
| `menu` | Displays a numbered list; subscriber selects by digit |
| `input` | Captures free-text input into `context[node_id]` |
| `decision` | Evaluates conditions silently; no subscriber-visible content |
| `action` | Executes a server-side handler; carries handler reference in metadata |
| `end` | Terminates the session cleanly |

Node position (`position_x`, `position_y`) is used by visual canvas tools and has no effect on
routing.

### Edges

Edges connect nodes. Each edge carries:

- `condition` — expression like `"balance > 0"` or `"language == sw"`. Evaluated in `priority`
  order; the first match wins.
- `label` — human-readable for the designer canvas.
- `priority` — integer; lower values are evaluated first (default 0).

When no condition is set, the edge always matches (unconditional fallback). Place unconditional
edges at the highest priority value to act as defaults.

### Condition Language

Conditions are single-clause expressions:

```
user_input == 1
balance > 500
tier != gold
language == sw
```

Keys are resolved from the session context dict. Values are compared as strings unless the
context value is numeric. Unsupported operators default to `True` (permissive) — use
`validate_flow` to surface warnings about non-evaluable conditions before activation.

### Multi-language Support

Add translations for any language via the translations API. Node titles, bodies, and item labels
are all translatable. At runtime (`ussd_eng`), the session language selects the right translation
with English as fallback.

Check coverage before activating:

```python
report = await svc.check_translation_completeness(flow_id, "sw")
# report["coverage_pct"] should be 100.0 for production flows
```

### Screen Budget

USSD networks (Safaricom, MTN) truncate pages at 182 bytes. Run budget validation before
activation:

```python
report = await svc.validate_screen_budgets(flow_id, budget=182)
if not report["passed"]:
    for v in report["violations"]:
        print(v["node_id"], v["language"], v["overflow"], "bytes over")
```

### Flow Versioning

Call `snapshot_flow(flow_id, label)` to save a checkpoint. Snapshots can be restored at any
time — enabling rollback after a bad deployment.

Compare two snapshots:

```python
diff = await svc.diff_flow_versions(flow_id, ver_id_old, ver_id_new)
# diff["summary"] -> {"nodes_added": 2, "nodes_removed": 0, ...}
```

### A/B Testing

Create an A/B test pairing a control flow with a variant. Traffic is split deterministically by
session ID hash — reproducible, no state. Record completions and retrieve conversion rates:

```python
test = await svc.create_ab_test("welcome_v2", "*123#", ctrl_id, var_id, split_percentage=30.0)
assignment = await svc.assign_ab_flow(test["id"], session_id)
# ... session completes ...
await svc.record_ab_completion(test["id"], assignment["arm"])
results = await svc.get_ab_test_results(test["id"])
# results["lift_pct"] = variant_rate - control_rate
```

---

## Standard Workflow

1. `POST /flows` — create a draft flow
2. `POST /flows/<id>/nodes` (repeat) — build the node graph
3. `POST /flows/<id>/edges` (repeat) — connect nodes with optional conditions
4. `POST /flows/<id>/simulate` — dry-run with a scripted conversation
5. `GET /flows/<id>/validate` — check for structural errors and orphaned nodes
6. `GET /flows/<id>/translations/<lang>/completeness` — verify i18n coverage
7. `GET /flows/<id>/screen-budget` — confirm no node exceeds 182 bytes
8. `GET /flows/<id>/score` — review usability metrics
9. `GET /flows/<id>/dead-paths` — prune unreachable nodes
10. `POST /flows/<id>/versions` — snapshot before go-live
11. `POST /flows/<id>/activate` — promote from draft to active
12. `POST /flows/<id>/export` — export for ingestion by `ussd_eng`

---

## Session Simulation

Simulate a scripted conversation without deploying:

```python
trace = await svc.simulate_session(
    flow_id,
    script=["1", "2", "500"],   # user inputs per step
    language="sw",
    context_seed={"balance": "1500", "tier": "gold"},
    expected_terminal_node="end_success",
)
assert trace["passed"]
for step in trace["steps"]:
    print(step["step"], step["node_id"], step["rendered_title"])
```

The trace reports every screen rendered, every edge taken, and the full context at each step. Use
`expected_terminal_node` for automated regression testing of flows before activation.

---

## Flow Usability Scoring

```python
scorecard = await svc.score_flow(flow_id)
# {
#   "avg_path_depth": 3.5,
#   "cyclomatic_complexity": 8,
#   "avg_branching_factor": 3.2,
#   "estimated_session_seconds": 35.0,
#   "usability_score": 87.0
# }
```

Penalties applied to usability score:

| Metric | Threshold | Penalty |
|--------|-----------|---------|
| avg_path_depth | > 5 | 6 pts per level (max 30) |
| cyclomatic_complexity | > 20 | 1 pt per unit (max 20) |
| avg_branching_factor | > 8 | 3 pts per unit (max 15) |

---

## Dead-Path Analysis

```python
report = await svc.compute_dead_paths(flow_id)
# report["forward_dead"] — not reachable from root
# report["backward_dead"] — no end node reachable from here
# report["fully_dead"] — safe to delete
# report["suggestions"] — ["add_incoming_edge_to: node_x", "safe_to_delete: node_y"]
```

---

## Service-Code Migration

```python
# Preview first
preview = await svc.migrate_service_code("*123#", "*456#", dry_run=True)
print(f"{preview['affected_flows']} flows would be updated")

# Apply
result = await svc.migrate_service_code("*123#", "*456#", dry_run=False)
# Pre-migration snapshots created automatically for all affected flows
```

---

## Live Session Debugging

Record events during a live session in `ussd_eng`:

```python
await svc.record_session_event(session_id, flow_id, node_id, user_input, context)
```

Replay later to find where routing diverged after a flow change:

```python
replay = await svc.replay_session(session_id)
if not replay["clean_replay"]:
    for d in replay["divergences"]:
        print(f"Step {d['step']}: observed {d['observed_next']}, now routes to {d['replayed_next']}")
```

---

## NATS Streaming Integration

All audit events can be forwarded to NATS subjects `ussd.flo.events.<event_type>` for
downstream bytewax pipelines:

- Real-time session abandonment monitoring
- A/B test anomaly detection
- Cross-tenant flow change alerts

Inject a `NatsEventPublisher` adapter into `UssdFloService.__init__` via the optional
`event_publisher` protocol parameter.

---

## Copyright

© 2025 Datacraft — www.datacraft.co.ke | nyimbi@gmail.com
