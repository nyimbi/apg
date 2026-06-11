# SASRA Regulatory Reporting — `fintech/sacco/reg`

SASRA (Sacco Societies Regulatory Authority) prudential return generation, ratio monitoring, filing registry, and compliance dashboard for deposit-taking SACCOs.

## What it does

| Feature | Description |
|---------|-------------|
| **Forms 1-5** | Balance sheet, income statement, capital adequacy, liquidity, loan portfolio quality |
| **Ratio engine** | CAR, liquidity, LDR, NPL, PAR30/90, provisioning coverage — all vs SASRA minimums |
| **Loan classification** | DPD-based SASRA matrix: Normal/Watch/Substandard/Doubtful/Loss with provision rates |
| **Compliance dashboard** | Traffic-light status (RED/AMBER/GREEN) per ratio |
| **Filing registry** | Record submissions with period, officer, and data snapshot |
| **Regulatory calendar** | All quarterly + annual deadlines with days-remaining and overdue flag |
| **XML export** | SASRA portal-compatible XML (Forms 1-5) |
| **Board report** | Compact pack with ratios, NPL, pending filings, executive summary |

## SASRA Thresholds Enforced

| Ratio | Minimum | Maximum | SASRA Reference |
|-------|---------|---------|-----------------|
| Capital Adequacy (CAR) | 10% | — | Reg 17(1) |
| Core Capital / Total Assets | 8% | — | Reg 17(2) |
| Liquidity | 15% | — | Reg 23(1) |
| Loan to Deposit | — | 70% | Reg 24(1) |
| NPL (warning) | — | 5% | Prudential Guidelines |
| NPL (breach) | — | 10% | Prudential Guidelines |

## DPD Provision Matrix

| Band | DPD Range | Provision Rate |
|------|-----------|----------------|
| Normal | 0-30 | 0% |
| Watch | 31-90 | 1% |
| Substandard | 91-180 | 25% |
| Doubtful | 181-365 | 50% |
| Loss | >365 | 100% |

## Quick Start

```python
from capabilities.fintech.sacco.reg.service import SACCARegulatoryService

svc = SACCARegulatoryService("my-sacco")

# Seed ledger data (in production, use domain/adapters.py LedgerAdapter)
svc.seed_ledger("my-sacco", "2025-03-31", {
    "core_capital": 25_000_000,
    "secondary_capital": 5_000_000,
    "total_assets": 113_000_000,
    "gross_loan_portfolio": 60_000_000,
    "government_securities": 10_000_000,
    "cash_on_hand": 5_000_000,
    "bank_balances": 20_000_000,
    "member_deposits": 100_000_000,
    "external_borrowings": 15_000_000,
    "loan_books": [
        {"outstanding_balance": 55_000_000, "days_past_due": 0},
        {"outstanding_balance": 3_000_000,  "days_past_due": 45},
        {"outstanding_balance": 2_000_000,  "days_past_due": 400},
    ],
    "loan_loss_provisions": 2_100_000,
})

import asyncio

# Generate quarterly return
qr = asyncio.run(svc.generate_quarterly_return("my-sacco", 2025, 1))
print(f"Compliant: {qr.overall_compliant}")
print(f"CAR: {qr.form3_capital_adequacy.capital_adequacy_ratio}%")

# Check compliance
cs = asyncio.run(svc.check_regulatory_compliance("my-sacco", "2025-03-31"))
for ratio in cs.ratios:
    print(f"{ratio.name}: {ratio.actual:.2f}% [{ratio.traffic_light}]")

# File a return
filing = asyncio.run(svc.file_return(
    "my-sacco",
    return_type=ReturnType.QUARTERLY,
    period="2025-Q1",
    data={"quarterly_return_id": qr.id},
    filing_officer="Jane Wanjiku",
))
```

## API Endpoints

Base: `/api/fintech/sacco/reg` · Header: `X-Tenant-ID`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/returns/quarterly?year=&quarter=` | Generate quarterly return |
| GET | `/returns/annual?year=` | Generate annual report |
| GET | `/returns/xml?year=&quarter=` | SASRA portal XML |
| GET | `/ratios/capital-adequacy` | CAR calculation |
| GET | `/ratios/liquidity` | Liquidity ratio |
| GET | `/ratios/loan-to-deposit` | LDR |
| GET | `/ratios/npl` | NPL ratio |
| GET | `/ratios/par?days=30` | PAR30 or PAR90 |
| GET | `/loan-classification` | DPD band breakdown |
| GET | `/provisions/required` | Required provisions total |
| GET | `/provisions/coverage` | Provisioning coverage % |
| GET | `/compliance` | Full compliance status |
| GET | `/compliance/dashboard` | Dashboard with traffic lights |
| GET | `/board-report` | Board pack data |
| POST | `/filings` | Record a filing |
| GET | `/filings` | Filing history |
| GET | `/calendar` | Full regulatory calendar |
| GET | `/calendar/pending` | Overdue + upcoming filings |
| POST | `/ledger/seed` | Inject test data |

## Filing Deadlines

- **Quarterly**: 30 days after quarter-end (Q1→Apr 30, Q2→Jul 31, Q3→Oct 31, Q4→Jan 31)
- **Annual**: April 30 of following year

Non-compliance with filing deadlines risks SASRA licence suspension.

## Tests

```bash
python -m pytest capabilities/fintech/sacco/reg/tests/ -v
```
