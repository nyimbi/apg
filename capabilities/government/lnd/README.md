# Land Registry (gov_lnd)

Parcel cadastre, title issuance, land transfer, adjudication, encumbrance registry, valuation rolls.

## Overview

Complete land administration system for managing land parcels, title deeds, ownership transfers, dispute adjudication, encumbrances/charges, and property valuations. Designed for national/county land registries in East Africa.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/lnd/health | Service health check |
| GET | /api/government/lnd/dashboard | Dashboard metrics |
| GET | /api/government/lnd/parcels | List parcels |
| GET | /api/government/lnd/parcels/{id} | Get parcel |
| POST | /api/government/lnd/parcels | Register parcel |
| PUT | /api/government/lnd/parcels/{id} | Update parcel |
| DELETE | /api/government/lnd/parcels/{id} | Deregister parcel |
| GET | /api/government/lnd/titles | List titles |
| GET | /api/government/lnd/titles/{id} | Get title |
| POST | /api/government/lnd/titles | Issue title |
| PUT | /api/government/lnd/titles/{id} | Update title |
| GET | /api/government/lnd/transfers | List transfers |
| POST | /api/government/lnd/transfers | Initiate transfer |
| POST | /api/government/lnd/transfers/{id}/complete | Complete transfer |
| GET | /api/government/lnd/adjudications | List adjudications |
| POST | /api/government/lnd/adjudications | Submit adjudication |
| GET | /api/government/lnd/encumbrances | List encumbrances |
| POST | /api/government/lnd/encumbrances | Register encumbrance |
| POST | /api/government/lnd/encumbrances/{id}/discharge | Discharge encumbrance |
| GET | /api/government/lnd/valuations | List valuations |
| POST | /api/government/lnd/valuations | Record valuation |
| GET | /api/government/lnd/audit-events | List audit events |
| POST | /api/government/lnd/transfers/{id}/stamp-duty | Compute stamp duty |
| POST | /api/government/lnd/transfers/{id}/duty-payment | Record duty payment |
| POST | /api/government/lnd/parcels/{id}/subdivide | Subdivide a parcel |
| GET | /api/government/lnd/parcels/{id}/chain | Title chain of ownership |
| POST | /api/government/lnd/leases | Register a lease |
| PUT | /api/government/lnd/leases/{id}/renew | Renew a lease |
| POST | /api/government/lnd/cautions | Lodge a caution |
| PUT | /api/government/lnd/cautions/{id}/confirm | Confirm caution (court order) |
| PUT | /api/government/lnd/cautions/{id}/withdraw | Withdraw a caution |
| POST | /api/government/lnd/cautions/expire-stale | Expire stale cautions |
| POST | /api/government/lnd/spousal-consents | Register spousal consent |
| POST | /api/government/lnd/titles/{id}/flag-matrimonial | Flag matrimonial property |
| POST | /api/government/lnd/rates/{id}/payment | Record rates payment |
| GET | /api/government/lnd/parcels/{id}/rates-arrears | Compute rates arrears |
| POST | /api/government/lnd/surveyors | Register surveyor |
| POST | /api/government/lnd/survey-plans | Deposit survey plan |
| GET | /api/government/lnd/parcels/{id}/survey-plans | List survey plans |
| POST | /api/government/lnd/adjudications/{id}/escalate | Escalate adjudication |
| POST | /api/government/lnd/escalations/{id}/decision | Record tribunal decision |
| POST | /api/government/lnd/titles/{id}/certificate | Generate title certificate |

## New Features (v1.1.0)

### Stamp Duty Computation
Calculates stamp duty, CGT, and registration fees per Kenya Stamp Duty Act Cap 480. Supports
residential (4%), agricultural (2%), and government/conservation (0%) rates. Uses `Decimal`
arithmetic throughout. Payment is recorded via `record_duty_payment`; transfers require
`duty_paid` status before completion.

### Parcel Subdivision
`subdivide_parcel` splits a parent parcel into 2+ children, validating that child areas do not
exceed the parent. Each child inherits county/sub-county and owner. Parent is marked
`subdivided`. Requires a survey reference from a registered surveyor.

### Title Chain of Ownership
`get_title_chain` returns the full provenance trail from original issuance through all
completed transfers, with a SHA-256 integrity hash for tamper detection.

### Lease Management
Full leasehold lifecycle: `register_lease` computes expiry date; `renew_lease` extends term
and recalculates total rent. All monetary fields use `Decimal`.

### Caution Workflow (LRA 2012 s.71–73)
Structured caution lifecycle: lodge → confirm (via court order) → withdraw. Cautions
auto-expire after 60 days (configurable). `expire_stale_cautions` bulk-expires overdue
lodged cautions. Active cautions are surfaced in `conduct_land_search`.

### Spousal Consent & Matrimonial Property (LRA 2012 s.93)
`flag_matrimonial_property` marks a title; subsequent transfers require a spousal consent
record via `register_spousal_consent`, or they raise `PermissionError("spousal_consent_required")`.

### Rates Ledger & Arrears
`record_rates_payment` tracks partial/full rate payments; `compute_rates_arrears` calculates
outstanding principal plus 2%/month statutory penalty interest from assessment date.

### Survey Plan Registry
`register_surveyor` (with licence expiry tracking) and `deposit_survey_plan` (validates
licence currency) implement Survey Act Cap 299 compliance. `list_survey_plans` returns all
plans for a parcel.

### Dispute Escalation
`escalate_adjudication` routes decided/submitted adjudications to the Land Dispute Tribunal,
ELC, or High Court. `record_tribunal_decision` propagates the judgement back to the original
adjudication record.

### Title Certificate Generation
`generate_title_certificate` assembles a structured certificate payload including QR-code seed
(SHA-256) and digital signature placeholder, ready for downstream PDF rendering per NLIMS spec.

## Usage Examples

```python
import asyncio
from capabilities.government.lnd.service import LandRegistryService

async def main():
    svc = LandRegistryService(tenant_id="lands_kenya")

    # Register a parcel
    p = await svc.register_parcel(
        "NRBI/WSTL/001", "Nairobi", "Westlands", "Parklands",
        0.5, tenant_id="lands_kenya", land_use="residential",
    )

    # Issue a title
    t = await svc.issue_title(
        p["id"], "IR-12345", "owner-1", "Alice Kamau",
        "2025-01-15", "Registrar of Titles", tenant_id="lands_kenya",
    )

    # Compute and record stamp duty on a transfer
    tr = await svc.initiate_transfer(
        t["id"], "owner-1", "Alice Kamau", "owner-2", "Bob Otieno",
        8_500_000, "2025-03-01", "TRANS-2025-001", "Registrar",
        tenant_id="lands_kenya",
    )
    duty = await svc.compute_stamp_duty(tr["id"], 8_500_000, "residential", tenant_id="lands_kenya")
    # duty["total_payable_kes"] == "448500.00"
    await svc.record_duty_payment(tr["id"], "KRA-PAY-001", 448500, "RCT-001", "owner-1", tenant_id="lands_kenya")

    # Generate a title certificate
    cert = await svc.generate_title_certificate(t["id"], "Registrar", tenant_id="lands_kenya")
    print(cert["qr_code_seed"])

asyncio.run(main())
```
