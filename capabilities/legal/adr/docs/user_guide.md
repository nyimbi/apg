# ADR / Dispute Resolution (leg_adr) — User Guide

## Overview

Manages arbitration, mediation, conciliation, and expert determination cases from filing through award enforcement or settlement.

## Supported Case Types

`arbitration`, `mediation`, `conciliation`, `expert_determination`, `adjudication`

## Case Status Flow

`filed` → `notice_served` → `panel_constituted` → `preliminary_conference` → `hearings` → `deliberation` → `award_rendered` | `settled` → `enforcement` → `closed`

## Neutral Roles

`sole_arbitrator`, `presiding_arbitrator`, `co_arbitrator`, `mediator`, `conciliator`, `expert`

## Award Types

`final`, `partial`, `interim`, `consent`, `default`

## API Reference

### File an Arbitration

```http
POST /api/legal/adr/cases
{
  "tenant_id": "acme",
  "title": "Datacraft v TechCorp — Contract Dispute",
  "case_type": "arbitration",
  "claimant_id": "entity-datacraft",
  "respondent_id": "entity-techcorp",
  "seat": "Nairobi",
  "rules": "Nairobi_Centre",
  "governing_law": "Kenya",
  "claim_amount": 12000000,
  "currency": "KES"
}
```

Returns `case_number` like `ARB-2026-2001`.

### Appoint an Arbitrator

```http
POST /api/legal/adr/neutrals
{
  "tenant_id": "acme",
  "case_id": "adr-001",
  "neutral_id": "arb-dr-kimani",
  "role": "sole_arbitrator",
  "appointed_by": "agreement",
  "appointment_date": "2026-07-01",
  "fee_rate": 50000
}
```

### Render an Award

```http
POST /api/legal/adr/awards
{
  "tenant_id": "acme",
  "case_id": "adr-001",
  "award_type": "final",
  "award_date": "2026-11-15",
  "awarded_to_id": "entity-datacraft",
  "award_amount": 10500000,
  "costs_awarded": 850000,
  "interest_rate": 14.0,
  "summary": "Claimant succeeds on all counts. Respondent to pay damages plus costs."
}
```

### Record a Settlement

```http
POST /api/legal/adr/settlements
{
  "tenant_id": "acme",
  "case_id": "adr-001",
  "settlement_date": "2026-10-01",
  "settlement_amount": 8000000,
  "terms_summary": "Full and final settlement; TechCorp pays KES 8M within 14 days",
  "signed_by_claimant_id": "ceo-datacraft",
  "signed_by_respondent_id": "ceo-techcorp",
  "confidentiality_clause": true
}
```

### ADR Dashboard

```http
GET /api/legal/adr/dashboard?tenant_id=acme
```

Returns total cases, active case count, claim value, award value, settlement count.
