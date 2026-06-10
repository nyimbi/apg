# ussd_flo User Guide

## Overview

`ussd_flo` is the APG USSD Flow Designer. It provides a graph-based USSD menu builder with:
- Visual node/edge flow definition
- Conditional routing between nodes
- Multi-language translation support
- Point-in-time version snapshots
- A/B test framework for comparing two flow variants

Flows defined in `ussd_flo` are exported and loaded into `ussd_eng` for runtime execution.

## Concepts

### Flow
A flow is a directed graph of nodes connected by edges. It has a single root node (entry point) and one or more end nodes. Flows start as `draft`, are `active` during use, and `archived` when retired.

### Nodes
Five node types:
- `menu` — displays a menu with numbered items
- `input` — captures free-text input into a session variable
- `decision` — evaluates conditions to route to different targets (no user-visible content)
- `action` — executes a server-side handler
- `end` — terminates the session

### Edges
Edges connect nodes. Each edge can carry:
- `condition` — expression like `"balance > 0"` or `"language == sw"`. Edges are evaluated in `priority` order; the first match wins.
- `label` — human-readable description for the designer UI

### Multi-language Support
Add translations for any language via the translations API. Node titles, bodies, and item labels are all translatable. At runtime (`ussd_eng`), the session language selects the right translation with English as fallback.

### Flow Versioning
Call `POST /flows/<id>/versions` to snapshot the current graph. Snapshots can be restored at any time, enabling rollback after bad deployments.

### A/B Testing
Create an A/B test pairing a control flow with a variant. Traffic is split deterministically by session ID hash (reproducible, no state). Record completions via `record_ab_completion()` and retrieve conversion rates via `GET /abtests/<id>/results`.

## Workflow

1. `POST /flows` — create a draft flow
2. `POST /flows/<id>/nodes` (repeat) — build the node graph
3. `POST /flows/<id>/edges` (repeat) — connect nodes with optional conditions
4. `GET /flows/<id>/validate` — check for errors and orphaned nodes
5. `POST /flows/<id>/activate` — promote from draft to active
6. `POST /flows/<id>/export` — export for ingestion by `ussd_eng`
