# Matter Management (leg_mat) — User Guide

## Overview

The Matter Management capability manages the full lifecycle of legal matters: opening, staffing, tracking tasks and deadlines, logging court docket events, recording time entries, running conflict checks, computing risk scores, generating privilege logs, and closing.

## Use Cases

- **Litigation matters**: track court dates, pleading deadlines, chained deadline trees, and assigned counsel.
- **Advisory engagements**: manage deliverable tasks and client deadlines.
- **Transactional matters**: coordinate multi-party task lists and document checklists via templates.
- **Workload management**: view attorney capacity and load scores across all active matters.
- **Risk triage**: batch-score all open matters; surface critical matters requiring immediate partner attention.
- **Compliance**: auto-generate privilege logs with tamper hash; run conflict-of-interest checks on intake.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Matter | Root work unit — a legal engagement with a client |
| Task | Discrete action item assigned to a team member |
| Deadline | Date-anchored obligation (court, filing, statutory, contractual) |
| Deadline Chain | Set of derived deadlines triggered by a single court event |
| Docket Entry | Court calendar event tied to a matter |
| Note | Privileged or non-privileged memo on a matter |
| Time Entry | Logged billable/non-billable hours with rate and narrative |
| Time Budget | Approved hours allocation; tracks burn vs. remaining |
| Conflict Check | Cross-matter party name scan to surface potential conflicts |
| Risk Score | Composite 0–100 score reflecting matter urgency and risk exposure |
| Privilege Log | Structured FRCP 26(b)(5)-conforming export of privileged notes |

## API Reference

### Create a Matter

```http
POST /api/legal/mat/matters
Content-Type: application/json

{
  "tenant_id": "acme",
  "title": "Smith v Jones",
  "matter_type": "litigation",
  "client_id": "client-001",
  "lead_attorney_id": "atty-007",
  "practice_area": "Commercial Litigation",
  "jurisdiction": "Kenya",
  "priority": "high",
  "budget": 50000.00
}
```

### Transition Matter Status (FSM)

```http
POST /api/legal/mat/matters/{id}/transition
{
  "tenant_id": "acme",
  "new_status": "active",
  "actor_id": "atty-007",
  "reason": "Parties confirmed — proceeding to active phase"
}
```

Valid transitions:
- `open` → `active`, `on_hold`, `archived`
- `active` → `on_hold`, `closed`, `archived`
- `on_hold` → `active`, `closed`, `archived`
- `closed` → `active`
- `archived` → terminal

### Apply Matter Template

Seed a matter with standard tasks in one call:

```http
POST /api/legal/mat/matters/{id}/template
{
  "tenant_id": "acme",
  "template_name": "litigation",
  "start_date": "2026-06-15",
  "assigned_to_id": "atty-007"
}
```

Available templates: `litigation`, `transactional`, `advisory`.

### Log a Time Entry

All monetary values use `Decimal` precision internally.

```http
POST /api/legal/mat/time-entries
{
  "tenant_id": "acme",
  "matter_id": "mat-abc123",
  "attorney_id": "atty-007",
  "hours": "2.5",
  "rate": "350.00",
  "narrative": "Reviewed pleadings and prepared motion outline",
  "billable": true,
  "entry_date": "2026-06-10"
}
```

### Get Budget Burn Report

```http
GET /api/legal/mat/matters/{id}/budget-burn?tenant_id=acme
```

Returns: total billable hours, billed amount, budget remaining, burn%, projected overrun date.

### Run Conflict Check

```http
POST /api/legal/mat/matters/{id}/conflict-check
{
  "tenant_id": "acme",
  "party_names": ["Jones Holdings Ltd", "ABC Finance"]
}
```

Returns: `status` (`clear` | `flagged`), list of conflicting matters with reasons.

### Create Chained Deadlines

Trigger a set of derived deadlines from a single court event:

```http
POST /api/legal/mat/deadlines/chain
{
  "tenant_id": "acme",
  "matter_id": "mat-abc123",
  "trigger_event": "complaint_filed",
  "trigger_date": "2026-06-15"
}
```

Supported trigger events: `complaint_filed`, `defence_filed`.

### Compute Risk Score

```http
GET /api/legal/mat/matters/{id}/risk?tenant_id=acme
```

Returns: `score` (0–100), `risk_level` (`low`/`medium`/`high`/`critical`), `contributing_factors`.

### Batch Risk Scores

```http
GET /api/legal/mat/risk/batch?tenant_id=acme
```

Returns all active matters sorted by score descending — use for partner triage dashboard.

### Generate Privilege Log

```http
GET /api/legal/mat/matters/{id}/privilege-log?tenant_id=acme
```

Returns FRCP 26(b)(5)-conforming entries plus `log_hash` (SHA-256) for tamper detection.

### Team Capacity Report

```http
GET /api/legal/mat/capacity?tenant_id=acme
```

Returns per-attorney: active matter count, pending/overdue tasks, upcoming deadlines (14d), `load_score` (0–100).

### List Open Matters

```http
GET /api/legal/mat/matters?tenant_id=acme&status=open
```

### Create a Deadline

```http
POST /api/legal/mat/deadlines
{
  "tenant_id": "acme",
  "matter_id": "mat-abc123",
  "title": "File Defence",
  "deadline_date": "2026-07-15",
  "deadline_type": "court",
  "reminder_days": [14, 7, 2, 1]
}
```

### Get Overdue Deadlines

```http
GET /api/legal/mat/deadlines?tenant_id=acme&overdue_only=true
```

### Dashboard

```http
GET /api/legal/mat/dashboard?tenant_id=acme
```

Returns: matter counts by type/status, open tasks, overdue deadlines.

## Matter Statuses

`open` → `active` → `on_hold` → `closed` → `archived`

Use `transition_matter_status` for all status changes. Guards prevent invalid transitions and closing with open tasks.

## Task Statuses

`pending` → `in_progress` → `completed` | `cancelled`

## Deadline Types

`court`, `filing`, `statute_of_limitations`, `contractual`, `regulatory`, `internal`

## Risk Score Breakdown

| Factor | Points | Notes |
|--------|--------|-------|
| Overdue tasks | 10 each, max 30 | Any task past due date |
| Overdue deadlines | 15 each, max 30 | Any pending deadline past date |
| SoL within 30 days | 20 | Once if any SoL deadline is imminent |
| Budget burn > 80% | 15 | Based on time entries vs. budget |
| Unresolved conflict flag | 25 | Any flagged conflict check |
| Inactivity > 30 days | 10 | No audit events in 30 days |

## Money and Precision

All financial values in `log_time_entry` are accepted as strings (`"350.00"`) and stored using `decimal.Decimal` arithmetic to avoid floating-point errors. Budget burn reports return amounts as Decimal strings.

## Audit Trail

Every mutation emits an audit event retrievable via:

```http
GET /api/legal/mat/audit?tenant_id=acme&limit=100
```
