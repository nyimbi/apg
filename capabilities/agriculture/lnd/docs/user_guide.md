# Land Management — User Guide

## Overview

`agr_lnd` is the APG digital land registry capability. It covers the full lifecycle of
land administration: registering parcels, capturing GPS boundaries, issuing titles,
managing transfers (including multi-signature approval for high-value transactions),
adjudicating disputes, registering encumbrances, subdividing or amalgamating parcels,
generating the valuation roll and rate bills, formalising customary tenure, performing
geospatial searches, and reconstructing the chain of title. All monetary values use
`Decimal` arithmetic for precision.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Parcel** | Base unit of land: parcel number, area (ha), tenure type, owner, location. |
| **GPS Boundary** | Polygon of lat/lng waypoints. Area auto-computed via Shoelace + equirectangular projection. |
| **Title** | Issued ownership document. Can be invalidated (audit trail preserved) or deleted. |
| **Transfer** | Ownership change workflow: initiated → approved → registered. Multi-sig required ≥ KES 10M. |
| **Dispute** | Adjudication case: filed → evidence_collection → hearing_scheduled → adjudicated. Locks parcel. |
| **Encumbrance** | Mortgage, caveat, lien, easement, or charge registered against a parcel. |
| **Valuation** | Assessed value record used to compute land-rate bills. |
| **Rate Bill** | `assessed_value × levy_rate_pct`, stored as Decimal, status: draft/issued/paid/overdue. |
| **Formalisation** | 6-stage customary→statutory tenure workflow per Kenya Community Land Act 2016. |
| **Chain of Title** | Full ownership provenance with encumbrance overlay per period. |
| **Webhook** | HMAC-SHA-256 signed event push to registered URLs for real-time downstream integration. |

---

## Example Workflows

### 1. Register a Parcel

```
POST /api/agriculture/lnd/parcels
{
  "parcel_number": "NAKURU/NAIVASHA/1234",
  "area_ha": 3.5,
  "tenure_type": "freehold",
  "owner_id": "farmer-001",
  "owner_name": "Jane Wanjiku",
  "location_county": "Nakuru",
  "location_sub_county": "Naivasha",
  "land_use": "arable"
}
```

### 2. Capture GPS Boundary

```
POST /api/agriculture/lnd/boundaries
{
  "parcel_id": "lnd-abc",
  "captured_by": "surveyor-001",
  "waypoints": [
    {"lat": -0.7167, "lng": 36.4333},
    {"lat": -0.7180, "lng": 36.4350},
    {"lat": -0.7160, "lng": 36.4360},
    {"lat": -0.7167, "lng": 36.4333}
  ],
  "accuracy_m": 2.5
}
```

Area is automatically computed and written back to the parcel record.

### 3. Issue a Title

```
POST /api/agriculture/lnd/titles
{
  "parcel_id": "lnd-abc",
  "title_number": "NAKURU/NVS/1234",
  "issued_by": "lands-officer-001",
  "issue_date": "2025-06-01",
  "tenure_type": "freehold"
}
```

### 4. Initiate a Land Transfer

```
POST /api/agriculture/lnd/transfers
{
  "parcel_id": "lnd-abc",
  "from_owner_id": "farmer-001",
  "to_owner_id": "farmer-002",
  "to_owner_name": "Peter Mwangi",
  "transfer_value": 1500000,
  "currency": "KES",
  "reason": "sale"
}
```

For transfers ≥ KES 10,000,000, two approver signatures are required before the status
advances to `approved`.

### 5. Multi-Signature Transfer Approval

```
POST /api/agriculture/lnd/transfers/{id}/sign
{
  "approver_id": "registrar-county-001",
  "role": "county_registrar",
  "signature_hash": "<sha256-of-approval-data>"
}
```

A second signature from a different approver triggers automatic advancement to `approved`.

### 6. File a Dispute

```
POST /api/agriculture/lnd/disputes
{
  "parcel_id": "lnd-abc",
  "complainant_id": "person-003",
  "complainant_name": "Mary Njeri",
  "description": "Boundary encroachment on northern edge",
  "evidence_urls": ["https://files.example.com/evidence1.pdf"]
}
```

Parcel status is immediately set to `disputed`. Use `PUT /disputes/{id}/advance` with
`{ "stage": "evidence_collection" }` to move through the adjudication pipeline.

### 7. Register an Encumbrance (Mortgage)

```
POST /api/agriculture/lnd/encumbrances
{
  "parcel_id": "lnd-abc",
  "type": "mortgage",
  "holder_id": "bank-001",
  "holder_name": "Equity Bank Kenya",
  "amount": 3000000,
  "currency": "KES",
  "notes": "Home loan ref HL-2025-00123"
}
```

Amount is stored as a `Decimal` string. Parcel status becomes `encumbered`.

To discharge: `POST /api/agriculture/lnd/encumbrances/{id}/discharge`

### 8. Subdivide a Parcel

```
POST /api/agriculture/lnd/parcels/{parent_id}/subdivide
{
  "children": [
    { "parcel_number": "NAKURU/NVS/1234-A", "area_ha": 1.5, "owner_id": "farmer-001",
      "owner_name": "Jane Wanjiku", "tenure_type": "freehold",
      "location_county": "Nakuru" },
    { "parcel_number": "NAKURU/NVS/1234-B", "area_ha": 2.0, "owner_id": "farmer-001",
      "owner_name": "Jane Wanjiku", "tenure_type": "freehold",
      "location_county": "Nakuru" }
  ]
}
```

Child areas must sum to ≤ parent area (tolerance 0.01 ha). Parent is cancelled with
`superseded_by` reference; each child carries `parent_id` back-reference.

### 9. Record a Valuation and Generate a Rate Bill

```
POST /api/agriculture/lnd/valuations
{
  "parcel_id": "lnd-abc",
  "assessed_value": 5000000,
  "currency": "KES",
  "method": "market_comparison",
  "valuation_date": "2025-01-01",
  "valued_by": "valuer-001"
}

POST /api/agriculture/lnd/parcels/lnd-abc/rate-bill
{
  "financial_year": "2025/2026",
  "levy_rate_pct": 0.01
}
```

Bill amount = KES 5,000,000 × 0.01 = **KES 50,000** (stored as Decimal `"50000.00"`).

### 10. Initiate Tenure Formalisation

```
POST /api/agriculture/lnd/formalisations
{
  "parcel_id": "lnd-abc",
  "community_id": "community-naivasha-west",
  "initiated_by": "community-liaison-001",
  "workflow_type": "community"
}
```

Advances through: `community_consent → demarcation → survey → adjudication →
registration → title_issued`.

Each stage: `POST /formalisations/{id}/advance { "officer_id": "..." }`

### 11. Geospatial Search

```
GET /api/agriculture/lnd/parcels/search/location?lat=-0.7167&lng=36.4333&radius_m=500
```

Returns all parcels whose boundary centroid is within 500 m, sorted by distance. Uses
Haversine formula.

```
GET /api/agriculture/lnd/parcels/search/point?lat=-0.7170&lng=36.4340
```

Point-in-polygon test (ray-casting) — returns the parcel containing that coordinate.

### 12. Chain of Title

```
GET /api/agriculture/lnd/parcels/{id}/chain-of-title
```

Returns full provenance: each prior owner, acquisition type, transfer value, title number,
and encumbrances active during that ownership period.

### 13. Register a Webhook

```
POST /api/agriculture/lnd/webhooks
{
  "url": "https://county-finance.example.go.ke/hooks/land",
  "events": ["transfer.registered", "title.issued", "encumbrance.registered"],
  "secret": "s3cr3t-hmac-key"
}
```

All matching events are dispatched with `X-Signature: sha256=<hmac>` header so receivers
can verify authenticity.

---

## Decimal Precision Policy

All monetary fields (`transfer_value`, `amount` on encumbrances, `assessed_value`,
`bill_amount`, `levy_rate_pct`) are persisted as `str(Decimal(...).quantize("0.01"))`.
Never pass these values through `float` arithmetic — use `Decimal` at every step.

## Audit Trail

Every mutation emits an audit event via `_emit`. Retrieve with:

```
GET /api/agriculture/lnd/audit?limit=200
```

The audit log is append-only and never purged.

## Tenant Isolation

`LandManagementService` is instantiated per tenant. All records carry `tenant_id`.
`guard_tenant_id` enforces non-empty, max-128-char tenant IDs at construction time.
