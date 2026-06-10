# M&E — Monitoring & Evaluation (ngo_me) — User Guide

## Overview

Provides the full M&E framework: SMART indicators, periodic data collection, progress reporting
with auto-snapshots, evaluation recording, and structured learning cycles.

## Indicator Types

`input`, `output`, `outcome`, `impact`, `process`

## Evaluation Ratings (OECD/DAC Scale)

`highly_satisfactory`, `satisfactory`, `moderately_satisfactory`,
`moderately_unsatisfactory`, `unsatisfactory`, `highly_unsatisfactory`

## Workflow

```
Create Indicators → Set Baselines → Collect Data Periodically
       ↓
  Progress Reports (auto-snapshot indicators)
       ↓
  Evaluations (mid-term, final)
       ↓
  Learning Cycles (findings → action points)
```

## API Examples

### Create an Indicator

```
POST /api/ngo/me/indicators
{
  "programme_id": "prg-001",
  "name": "Households with food security",
  "code": "OUT-01",
  "indicator_type": "outcome",
  "target_value": 5000,
  "target_date": "2026-12-31",
  "unit": "households",
  "baseline_value": 1200,
  "baseline_date": "2026-01-01"
}
```

### Collect a Data Point

```
POST /api/ngo/me/data-collections
{
  "indicator_id": "ind-xxx",
  "programme_id": "prg-001",
  "value": 3200,
  "collection_date": "2026-06-30",
  "collected_by": "me_officer@org.ke",
  "period": "Q2 2026"
}
```

### Create a Mid-Term Evaluation

```
POST /api/ngo/me/evaluations
{
  "programme_id": "prg-001",
  "evaluator": "External Evaluator Ltd",
  "evaluation_date": "2026-07-01",
  "evaluation_type": "mid_term",
  "rating": "satisfactory",
  "findings": "Programme is on track...",
  "recommendations": "Strengthen targeting..."
}
```
