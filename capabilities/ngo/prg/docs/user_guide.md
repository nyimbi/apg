# Programme & Project Monitoring (ngo_prg) — User Guide

## Overview

Manages programme and project lifecycle: logframes, activity scheduling, output/outcome recording,
and field data collection. Integrates with M&E (ngo_me) for indicator tracking.

## Programme Lifecycle

`planning → active → completed`

## Logframe Structure

- **Goal**: Long-term development objective
- **Purpose**: Immediate objective the programme achieves
- **Outputs**: Concrete deliverables per activity
- **Activities**: Tasks with budget and schedule
- **Assumptions**: External factors required for success

## API Examples

### Create a Programme

```
POST /api/ngo/prg/
{
  "name": "Hunger Relief Programme 2026",
  "code": "HRP-2026",
  "start_date": "2026-01-01",
  "end_date": "2026-12-31",
  "sector": "food_security",
  "budget": 10000000,
  "lead_staff": "pm@org.ke"
}
```

### Record an Output

Via activity endpoint after creating an activity, then:
```
POST /api/ngo/prg/<programme_id>/activities
{
  "name": "Community Kitchen Training",
  "planned_start": "2026-02-01",
  "planned_end": "2026-03-31"
}
```

### Submit Field Data

```
POST /api/ngo/prg/<programme_id>/field-data
{
  "collector": "field_officer_jane",
  "collection_date": "2026-02-15",
  "location": "Turkana South",
  "data_type": "observation",
  "data": {"households_visited": 120, "children_fed": 340}
}
```
