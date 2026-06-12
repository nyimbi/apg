# leg_ip — Intellectual Property Registry

Patent, trademark, copyright portfolio management, renewal deadlines, licensing, royalties.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/ip/health | Health check |
| GET | /api/legal/ip/assets | List IP assets |
| GET | /api/legal/ip/assets/{id} | Get asset |
| POST | /api/legal/ip/assets | Register asset |
| PUT | /api/legal/ip/assets/{id} | Update asset |
| DELETE | /api/legal/ip/assets/{id} | Abandon asset |
| POST | /api/legal/ip/assets/{id}/register | Record registration |
| GET | /api/legal/ip/renewals | List renewals |
| POST | /api/legal/ip/renewals | File renewal |
| POST | /api/legal/ip/renewals/{id}/confirm | Confirm renewal |
| GET | /api/legal/ip/licenses | List licenses |
| POST | /api/legal/ip/licenses | Grant license |
| DELETE | /api/legal/ip/licenses/{id} | Terminate license |
| GET | /api/legal/ip/royalties | List royalties |
| POST | /api/legal/ip/royalties | Record royalty |
| POST | /api/legal/ip/royalties/{id}/pay | Pay royalty |
| GET | /api/legal/ip/expiring | Expiring assets |
| GET | /api/legal/ip/portfolio | Portfolio summary |
| GET | /api/legal/ip/audit | Audit events |

## Service Class

`IntellectualPropertyService` — full IP lifecycle from application to expiry, with exclusive license conflict detection, royalty calculation, and auto renewal-due-date computation.

## World-Class Enhancements (v2.0)

Fifteen improvements that elevate `leg_ip` from basic registry to enterprise-grade IP intelligence, competitive with Anaqua, CPA Global (Clarivate), Dennemeyer, and IPfolio.

**I1. Tiered Royalty Engine** — multi-bracket royalty schedules with `Decimal` precision for audited financials [Feature]

**I2. Smart Renewal Deadline Engine** — jurisdiction-aware lead times (`USPTO`, `EUIPO`, `KIPI`) with multi-milestone dates instead of a naive 30-day offset [Compliance]

**I3. Opposition / Cancellation Case Tracker** — `OPPOSITION_STAGES` state machine with gate-validated transitions, hearing dates, and cascading asset deadlines [Feature]

**I4. AI-Assisted Trademark Clearance Search** — trigram + character-level similarity scoring against same-class marks, returns ranked hit list with recommended action [AI/ML]

**I5. Assignment Chain with Chain-of-Title Validation** — `assign_asset` validates ownership continuity; `get_chain_of_title` returns ordered history with gap flags [Feature]

**I6. Portfolio Valuation Engine** — cost approach (prosecution + renewal) and income approach (discounted royalty streams) per IAS 38, all in `Decimal` [Feature]

**I7. Renewal Calendar Export** — RFC 5545-compliant `.ics` feed with `VALARM` components plus structured JSON, for native Outlook / Google Calendar subscription [UX]

**I8. License Revenue Forecasting** — linear trend fit per license from historical royalties, returns `{period: Decimal}` monthly forecast with confidence bands [AI/ML]

**I9. Lapsed Asset Revival Candidate Finder** — identifies lapsed assets still within statutory revival windows (`REVIVAL_WINDOWS` registry) with deadline and fee estimates [Feature]

**I10. Multi-Currency Royalty Settlement** — `record_royalty` accepts `settlement_currency` + `fx_rate: Decimal`, stores both home and foreign amounts for audit reproducibility [Feature]

**I11. IP Due Diligence Report Generator** — assembles ownership chain, active licenses, open oppositions, expiry schedule, and encumbered value into a structured dict for PDF/DOCX rendering [UX]

**I12. Watch Service with Similarity-Based Hit Detection** — `process_watch_results` scores incoming trademark feeds against registered marks and emits `watch_hit_detected` events [Integration]

**I13. Prosecution Cost Ledger per Asset** — `record_prosecution_cost` appends itemised filing/examination costs per asset for IAS 38 capitalisation [Feature]

**I14. Regulatory Compliance Checklist per Jurisdiction** — `get_compliance_checklist` surfaces jurisdiction-specific maintenance obligations with due dates and risk levels [Compliance]

**I15. Embodiment / Product-Asset Mapping** — `link_asset_to_product` maps patents to product SKUs; `get_assets_for_product` enables rapid freedom-to-operate responses [Integration]

## New Methods

### `record_royalty_tiered` — Tiered Royalty Calculation (I1)

```python
svc = IntellectualPropertyService(tenant_id="acme")

result = await svc.record_royalty_tiered(
    tenant_id="acme",
    license_id="lic_abc123",
    revenue_base=Decimal("12_000_000"),   # KES
    tiers=[
        {"threshold": Decimal("5_000_000"), "rate": Decimal("0.05")},
        {"threshold": None,                 "rate": Decimal("0.08")},
    ],
    period="2025-Q4",
)
# result["royalty_amount"]    => Decimal("810000.00")
# result["tier_breakdown"]    => [{"bracket": ..., "amount": ...}, ...]
```

### `trademark_clearance_search` — AI-Assisted Clearance (I4)

```python
hits = await svc.trademark_clearance_search(
    tenant_id="acme",
    proposed_mark="SwiftPay",
    nice_class=36,          # financial services
    similarity_threshold=Decimal("0.70"),
)
# hits => [
#   {"mark": "SwiftPay Africa", "similarity": 0.92, "status": "registered", "action": "HIGH RISK"},
#   {"mark": "Swiftpay Ltd",    "similarity": 0.85, "status": "pending",    "action": "REVIEW"},
# ]
```

### `generate_due_diligence_report` — M&A IP Due Diligence (I11)

```python
report = await svc.generate_due_diligence_report(
    tenant_id="acme",
    target_owner_ids=["owner_xyz", "owner_abc"],
)
# report keys:
#   "assets"           — list of asset details with current status
#   "ownership_chains" — chain-of-title per asset
#   "active_licenses"  — licensee, scope, expiry
#   "open_oppositions" — stage, hearing dates
#   "expiry_schedule"  — assets expiring within 24 months
#   "encumbered_value" — Decimal total across valuation method
```
