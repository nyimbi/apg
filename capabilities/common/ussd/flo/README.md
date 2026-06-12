# ussd_flo — USSD Flow Designer

Visual USSD menu flow builder with conditional routing, multi-language support, A/B test flows,
session simulation, screen-budget validation, flow scoring, dead-path analysis, and service-code
migration.

## API

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/health` | Service health |
| GET | `/api/ussd/flo/describe` | Capability descriptor |
| GET | `/api/ussd/flo/dashboard` | Summary dashboard |
| GET | `/api/ussd/flo/audit` | Audit event log |

### Flows

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/flows` | List flows |
| POST | `/api/ussd/flo/flows` | Create flow |
| GET | `/api/ussd/flo/flows/<id>` | Get flow |
| PUT | `/api/ussd/flo/flows/<id>` | Update flow |
| DELETE | `/api/ussd/flo/flows/<id>` | Delete flow |
| POST | `/api/ussd/flo/flows/<id>/activate` | Activate flow |
| POST | `/api/ussd/flo/flows/<id>/archive` | Archive flow |
| GET | `/api/ussd/flo/flows/<id>/validate` | Validate flow graph |
| POST | `/api/ussd/flo/flows/<id>/export` | Export flow |
| POST | `/api/ussd/flo/flows/import` | Import flow |

### Nodes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/flows/<id>/nodes` | List nodes |
| POST | `/api/ussd/flo/flows/<id>/nodes` | Add node |
| POST | `/api/ussd/flo/flows/<id>/nodes/bulk` | Bulk add nodes |
| GET | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Get node |
| PUT | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Update node |
| DELETE | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Delete node |

### Edges

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/flows/<id>/edges` | List edges |
| POST | `/api/ussd/flo/flows/<id>/edges` | Add edge |
| POST | `/api/ussd/flo/flows/<id>/edges/bulk` | Bulk add edges |
| DELETE | `/api/ussd/flo/flows/<id>/edges/<eid>` | Delete edge |

### Routing & Analysis

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ussd/flo/flows/<id>/route` | Resolve next node |
| GET | `/api/ussd/flo/flows/<id>/reachable/<nid>` | BFS reachable nodes |
| GET | `/api/ussd/flo/flows/<id>/cycles` | Cycle detection |
| GET | `/api/ussd/flo/flows/<id>/dead-paths` | Dead-path analysis |
| GET | `/api/ussd/flo/flows/<id>/score` | Usability scorecard |
| GET | `/api/ussd/flo/flows/<id>/screen-budget` | Screen budget check |

### Simulation & Replay

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ussd/flo/flows/<id>/simulate` | Dry-run scripted session |
| POST | `/api/ussd/flo/sessions/<sid>/events` | Record session event |
| GET | `/api/ussd/flo/sessions/<sid>/replay` | Replay recorded session |

### Translations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/flows/<id>/translations` | List translations |
| POST | `/api/ussd/flo/flows/<id>/translations` | Add translation |
| GET | `/api/ussd/flo/flows/<id>/translations/<lang>` | Get translation |
| DELETE | `/api/ussd/flo/flows/<id>/translations/<lang>` | Delete translation |
| GET | `/api/ussd/flo/flows/<id>/translations/<lang>/completeness` | Coverage check |
| POST | `/api/ussd/flo/flows/<id>/nodes/<nid>/render` | Render translated node |

### Versions

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/flows/<id>/versions` | List versions |
| POST | `/api/ussd/flo/flows/<id>/versions` | Snapshot flow |
| POST | `/api/ussd/flo/flows/<id>/versions/<vid>/restore` | Restore version |
| GET | `/api/ussd/flo/flows/<id>/versions/<va>/diff/<vb>` | Diff two versions |

### A/B Tests

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/abtests` | List A/B tests |
| POST | `/api/ussd/flo/abtests` | Create A/B test |
| GET | `/api/ussd/flo/abtests/<id>` | Get A/B test |
| PUT | `/api/ussd/flo/abtests/<id>` | Update A/B test |
| DELETE | `/api/ussd/flo/abtests/<id>` | Delete A/B test |
| POST | `/api/ussd/flo/abtests/<id>/assign` | Assign session to arm |
| POST | `/api/ussd/flo/abtests/<id>/complete` | Record arm completion |
| GET | `/api/ussd/flo/abtests/<id>/results` | A/B test results |

### Operations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ussd/flo/migrate-service-code` | Rename service code across flows |

## New Service Methods (v1.1)

| Method | Category | Description |
|--------|----------|-------------|
| `simulate_session` | Testing | Dry-run scripted USSD conversation; returns full SessionTrace |
| `score_flow` | Analytics | Usability scorecard: depth, cyclomatic complexity, branching factor |
| `check_translation_completeness` | i18n | Per-language coverage report with missing-key list |
| `validate_screen_budgets` | Compliance | Flag nodes exceeding 182-byte USSD page limit |
| `diff_flow_versions` | Collaboration | Structural diff between two snapshots |
| `migrate_service_code` | Operations | Dry-run or live rename of a USSD service code |
| `compute_dead_paths` | Flow Quality | Forward/backward dead-node analysis with removal suggestions |
| `record_session_event` | Debugging | Capture live session events for replay |
| `replay_session` | Debugging | Re-evaluate a recorded session; flags routing divergences |
| `bulk_add_edges` | Productivity | Add N edges in a single async-gathered call |

## Node Types

| Type | Purpose |
|------|---------|
| `menu` | Display numbered options; user selects by digit |
| `input` | Capture free-text input into session context |
| `decision` | Evaluate conditions; no visible content |
| `action` | Execute server-side handler |
| `end` | Terminate the session |

## Edge Conditions

Conditions are simple expressions evaluated against the session context:

```
user_input == 1
balance > 0
language != en
tier IN gold,platinum
```

Edges are evaluated in ascending `priority` order; the first matching edge wins.

## A/B Assignment

Assignment is deterministic: `md5(session_id) % 100 < split_percentage` routes to the variant.
No server-side state is required for assignment — the same session always gets the same arm.

## Streaming Integration

Publish `ussd.flo.events.*` NATS subjects to feed bytewax pipelines for real-time session
abandonment detection and A/B anomaly alerting.

## Copyright

© 2025 Datacraft — www.datacraft.co.ke

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Intent-Aware Conditional DSL** [Routing Engine]
- **I2. Session Simulation & Dry-Run Engine** [Testing]
- **I3. NATS-Based Real-Time Event Streaming** [Observability]
- **I4. Automatic Dead-Path Pruning Advisor** [Flow Quality]
- **I5. Multi-Variant A/B Testing (N-way Splits)** [Experimentation]
- **I6. Pluggable Variable Interpolation with Template Functions** [Content Rendering]
- **I7. Flow Diff and Merge** [Collaboration]
- **I8. Accessibility & Screen-Budget Validator** [Compliance]
- **I9. Flow Template Library** [Productivity]
- **I10. Role-Based Edit Permissions per Flow** [Security]
- **I11. USSD Session Replay from Logs** [Debugging]
- **I12. Composite Flow (Sub-flow Inclusion)** [Composability]
- **I13. Automated Translation Completeness Check** [i18n Quality]
- **I14. Flow Performance Scoring** [Analytics]
- **I15. Bulk Flow Migration Across Service Codes** [Operations]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
