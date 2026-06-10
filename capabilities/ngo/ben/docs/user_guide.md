# Beneficiary Registry (ngo_ben) — User Guide

## Overview

Manages the full beneficiary lifecycle: registration, programme enrolment, vulnerability scoring,
cash/in-kind transfers, and deduplication to prevent double-dipping.

## Vulnerability Scoring

Five dimensions (0–100 each) are averaged to produce a composite score:

| Score Range | Category |
|-------------|----------|
| 80–100 | critical |
| 60–79 | high |
| 40–59 | medium |
| 20–39 | low |
| 0–19 | none |

## API Examples

### Register a Beneficiary

```
POST /api/ngo/ben/
{
  "first_name": "Amina",
  "last_name": "Wanjiru",
  "national_id": "12345678",
  "gender": "female",
  "county": "Turkana",
  "household_size": 5
}
```

### Run Vulnerability Assessment

```
POST /api/ngo/ben/<id>/assessments
{
  "assessor": "field_worker_1",
  "assessment_date": "2026-03-01",
  "food_security_score": 75,
  "shelter_score": 60,
  "health_score": 55,
  "income_score": 80,
  "protection_score": 45
}
```

### Create Transfer

```
POST /api/ngo/ben/<id>/transfers
{
  "programme_id": "prg-hunger-relief",
  "amount": 3000,
  "transfer_date": "2026-03-15",
  "reference": "MPESA-001",
  "approved_by": "supervisor@org.ke",
  "payment_method": "mpesa"
}
```
