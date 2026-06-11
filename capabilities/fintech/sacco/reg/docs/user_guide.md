# SASRA Regulatory Reporting — User Guide

## Overview

This capability automates SASRA (Sacco Societies Regulatory Authority) prudential return preparation for deposit-taking SACCOs in Kenya. It generates all five SASRA return forms, monitors compliance ratios against statutory minimums, maintains a filing registry, and produces a compliance dashboard with traffic-light indicators.

**Regulatory basis**: SACCO Societies (Deposit-Taking) Regulations, 2010 and SASRA Prudential Guidelines.

---

## Getting Started

### Prerequisites

- APG platform with `fintech/sacco` capabilities installed
- Ledger data from `fintech/sacco/dep` (deposits) and `fintech/sacco/lnd` (loans)
- SACCO registered with SASRA and holding a valid licence

### Seeding Ledger Data

In production, the `LedgerAdapter` in `domain/adapters.py` pulls live balances from the deposit and lending services. For testing or standalone use, seed data directly:

```python
svc.seed_ledger("my-sacco", "2025-03-31", {
    # Balance sheet items
    "cash_on_hand": 5_000_000,
    "bank_balances": 20_000_000,
    "government_securities": 10_000_000,
    "gross_loan_portfolio": 60_000_000,
    "loan_loss_provisions": 2_100_000,
    "member_deposits": 100_000_000,
    "external_borrowings": 15_000_000,
    "core_capital": 25_000_000,
    "secondary_capital": 5_000_000,
    "total_assets": 113_000_000,
    # Loan portfolio (per-loan DPD data)
    "loan_books": [
        {"outstanding_balance": 55_000_000, "days_past_due": 0},
        {"outstanding_balance": 3_000_000,  "days_past_due": 60},
        {"outstanding_balance": 2_000_000,  "days_past_due": 400},
    ],
})
```

---

## Core Workflows

### 1. Generate Quarterly Prudential Return

Produces all five SASRA forms for a quarter-end date.

```python
qr = await svc.generate_quarterly_return("my-sacco", year=2025, quarter=1)

print(f"Period end: {qr.period_end}")
print(f"Overall compliant: {qr.overall_compliant}")
print(f"Violations: {qr.violations}")

# Form 3: Capital Adequacy
car = qr.form3_capital_adequacy
print(f"CAR: {car.capital_adequacy_ratio:.2f}% (min 10%) — {car.traffic_light}")

# Form 5: Loan Portfolio
lc = qr.form5_loan_classification
print(f"NPL ratio: {lc.npl_ratio:.2f}%")
print(f"PAR30: {lc.par30:.2f}%")
```

**Quarter-end dates**: Q1 = March 31, Q2 = June 30, Q3 = September 30, Q4 = December 31

### 2. Check Regulatory Compliance

Evaluates all key SASRA ratios against minimums at a point in time.

```python
cs = await svc.check_regulatory_compliance("my-sacco", "2025-03-31")

print(f"Compliant: {cs.overall_compliant}")
for ratio in cs.ratios:
    status = "✓" if ratio.compliant else "✗"
    print(f"{status} {ratio.name}: {ratio.actual:.2f}% [{ratio.traffic_light}]")
```

Traffic lights:
- **GREEN**: Ratio passes with margin (>2pp above minimum / below maximum)
- **AMBER**: Within 2 percentage points of the threshold — take corrective action
- **RED**: Ratio breaches SASRA minimum — regulatory violation

### 3. Compliance Dashboard

Single call returning all ratios, pending filings, and overall status.

```python
dash = await svc.get_compliance_dashboard("my-sacco")

print(f"Overall: {dash.overall_status}")
for pf in dash.pending_filings:
    overdue = "OVERDUE" if pf.overdue else f"{pf.days_remaining} days"
    print(f"{pf.period}: {pf.return_type} — {overdue}")
```

### 4. Classify Loan Portfolio (SASRA DPD Matrix)

```python
lc = await svc.classify_loan_portfolio("my-sacco", "2025-03-31")

for band in lc.bands:
    print(f"{band.band.value:12s} ({band.dpd_range:7s} DPD): "
          f"KES {band.outstanding_balance:>15,.0f}  "
          f"provision rate {band.provision_rate}%  "
          f"required KES {band.required_provision:>12,.0f}")

print(f"\nTotal gross portfolio: KES {lc.total_gross_portfolio:,.0f}")
print(f"Required provisions:   KES {lc.total_required_provisions:,.0f}")
print(f"Actual provisions:     KES {lc.actual_provisions_held:,.0f}")
print(f"Coverage:              {lc.provisioning_coverage:.1f}%")
```

**DPD Band Provision Rates** (SASRA Prudential Guidelines):

| Band | DPD Range | Provision |
|------|-----------|-----------|
| Normal | 0–30 days | 0% |
| Watch | 31–90 days | 1% |
| Substandard | 91–180 days | 25% |
| Doubtful | 181–365 days | 50% |
| Loss | >365 days | 100% |

### 5. File a Return

Records the submission in APG. Actual upload to the SASRA portal is done manually or via `SASRAPortalAdapter`.

```python
filing = await svc.file_return(
    tenant_id="my-sacco",
    return_type=ReturnType.QUARTERLY,
    period="2025-Q1",
    data={"quarterly_return_id": qr.id, "notes": "Verified by CEO"},
    filing_officer="Jane Wanjiku",
    submitted_at="2025-04-28T09:00:00Z",
)
print(f"Filing ID: {filing.id}")
print(f"Status: {filing.filing_status}")
```

### 6. Regulatory Calendar

```python
cal = await svc.get_regulatory_calendar("my-sacco", year=2025)

for deadline in cal:
    filed = "Filed" if deadline.filed else ("OVERDUE" if deadline.overdue else f"{deadline.days_remaining}d")
    print(f"{deadline.period:12s} due {deadline.due_date}  [{filed}]")
```

**Filing deadlines**:
- **Quarterly returns**: 30 days after quarter-end
- **Annual audited accounts**: April 30 of the following year

### 7. Generate SASRA XML Return

Produces XML for upload to the SASRA portal.

```python
xml = await svc.generate_sasra_xml_return("my-sacco", year=2025, quarter=1)
with open("sasra_q1_2025.xml", "w") as f:
    f.write(xml)
```

### 8. Board Report

```python
report = await svc.generate_board_report("my-sacco", period="2025-03-31")
print(report["executive_summary"])
print(report["key_ratios"])
```

---

## SASRA Minimum Requirements

| Ratio | Formula | Minimum | Maximum |
|-------|---------|---------|---------|
| Capital Adequacy (CAR) | Institutional capital / Risk-weighted assets | **10%** | — |
| Core Capital Ratio | Core capital / Total assets | **8%** | — |
| Liquidity Ratio | Liquid assets / (Deposits + Borrowings) | **15%** | — |
| Loan to Deposit | Gross loans / Member deposits | — | **70%** |
| NPL Ratio (warning) | NPL balance / Gross portfolio | — | 5% |
| NPL Ratio (breach) | NPL balance / Gross portfolio | — | **10%** |
| Provisioning Coverage | Actual provisions / Required provisions | **100%** | — |

**Institutional capital** = Core capital (paid-up share capital + retained earnings) + Secondary capital (general provisions + statutory reserves)

---

## REST API

All endpoints require `X-Tenant-ID` header. KES amounts returned as Decimal strings.

### Examples

```bash
# Health
curl -H "X-Tenant-ID: my-sacco" http://localhost:5000/api/fintech/sacco/reg/health

# Generate Q1 2025 return
curl -H "X-Tenant-ID: my-sacco" \
  "http://localhost:5000/api/fintech/sacco/reg/returns/quarterly?year=2025&quarter=1"

# Compliance dashboard
curl -H "X-Tenant-ID: my-sacco" \
  http://localhost:5000/api/fintech/sacco/reg/compliance/dashboard

# Regulatory calendar for 2025
curl -H "X-Tenant-ID: my-sacco" \
  "http://localhost:5000/api/fintech/sacco/reg/calendar?year=2025"

# File a quarterly return
curl -X POST -H "X-Tenant-ID: my-sacco" \
  -H "Content-Type: application/json" \
  -d '{"return_type":"quarterly","period":"2025-Q1","filing_officer":"Jane Wanjiku","data":{}}' \
  http://localhost:5000/api/fintech/sacco/reg/filings

# Export XML for SASRA portal
curl -H "X-Tenant-ID: my-sacco" \
  "http://localhost:5000/api/fintech/sacco/reg/returns/xml?year=2025&quarter=1" \
  -o sasra_q1_2025.xml
```

---

## Consequences of Non-Compliance

Per the SACCO Societies Act (Cap. 490B):

- **Filing late**: Penalty fees per SASRA tariff
- **Ratio breach sustained > 2 quarters**: Formal notice, corrective action plan required
- **Ratio breach unresolved**: Restriction on member dividend payments
- **Persistent non-compliance**: Licence suspension or revocation

The AMBER traffic light (within 2pp of threshold) is your early-warning indicator — act before it turns RED.

---

## Multi-Tenancy

Each SACCO operates as an isolated tenant. Pass `tenant_id` to all service calls or set it at construction time:

```python
svc = SACCARegulatoryService("umoja-sacco")
# All calls default to "umoja-sacco" if tenant_id omitted
```

Ledger data, filings, and audit events are strictly isolated per tenant.
