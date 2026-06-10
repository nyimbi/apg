# Matter Management (leg_mat) — User Guide

## Overview

The Matter Management capability manages the full lifecycle of legal matters: opening, staffing, tracking tasks and deadlines, logging court docket events, and closing.

## Use Cases

- **Litigation matters**: track court dates, pleading deadlines, and assigned counsel.
- **Advisory engagements**: manage deliverable tasks and client deadlines.
- **Transactional matters**: coordinate multi-party task lists and document checklists.
- **Workload management**: view attorney workload across all active matters.

## Key Concepts

| Concept | Description |
|---------|-------------|
| Matter | The root work unit — a legal engagement with a client |
| Task | A discrete action item assigned to a team member |
| Deadline | A date-anchored obligation (court, filing, statutory, contractual) |
| Docket Entry | A court calendar event tied to a matter |
| Note | A privileged or non-privileged memo on a matter |
| Time Budget | An approved hours allocation for a matter |

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
  "priority": "high"
}
```

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

## Task Statuses

`pending` → `in_progress` → `completed` | `cancelled`

## Deadline Types

`court`, `filing`, `statute_of_limitations`, `contractual`, `regulatory`, `internal`
