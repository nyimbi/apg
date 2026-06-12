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

## World-Class Enhancements (v2.0)

**I1. Stress-Test Simulation Engine** — apply parameterised shocks (NPL spike, deposit run, securities haircut) to a ledger snapshot and surface the first threshold-breaching scenario [Risk Analytics]

**I2. Trend Analysis & Ratio Trajectory Forecasting** — pull historical snapshots, fit a linear trend per ratio, project N quarters forward, and flag trajectories crossing thresholds before the next filing [Predictive Compliance]

**I3. Corrective Action Plan (CAP) Generator** — detect all ratio breaches, quantify the capital/liquidity restoration amounts, generate structured action items with 30/60-day milestones, and emit a PDF-ready dict (SACCO Societies Act Cap 490B §35) [Regulatory Workflow]

**I4. Statutory Reserve Adequacy Monitor** — track the 10% annual net-surplus transfer obligation vs cumulative statutory reserve and compute the top-up needed to reach the minimum core-capital threshold (Reg 22) [Capital Management]

**I5. Dividend Restriction Enforcer** — gate proposed dividend payments against all SASRA ratio floors and return a SASRA-language eligibility decision with per-ratio breakdown [Member Protection / Regulatory Gate]

**I6. Multi-Period Peer Benchmarking** — accept SASRA sector statistics (median, p25, p75 per ratio) and return each ratio's percentile position with outlier flags [Comparative Analytics]

**I7. Liquidity Stress Buffer Calculator** — model an N-day deposit withdrawal at a configurable run rate, compute residual liquid assets, and report survival-horizon days vs the 5-day minimum [Liquidity Risk]

**I8. Regulatory Filing Reminder & Penalty Estimator** — compute days overdue, apply SASRA penalty rates (KES 2,000/day quarterly, KES 5,000/day annual), and flag the 90-day suspension trigger [Compliance Operations]

**I9. Loan Write-Off Recommendation Engine** — identify fully-provisioned loss-band loans (DPD > threshold), verify no ratio breach post-write-off, and return a board-resolution-ready write-off schedule [Asset Quality]

**I10. Regulatory Ratio Sensitivity Analysis** — sweep the primary driver of any ratio across a delta range and return a sensitivity table showing headroom to breach [Capital Planning]

**I11. Cross-Ratio Conflict Detector** — run balance-sheet identity checks and cross-ratio invariant validations, flagging arithmetic inconsistencies with specific field references before SASRA portal submission [Regulatory Quality Assurance]

**I12. SASRA Examination Readiness Score** — score all CAMEL dimensions using available ratio and governance data, return a composite 1-5 SASRA scale score with the three highest-impact remediation actions [Supervisory Readiness]

**I13. Capital Injection Planning Tool** — model equal monthly member share-capital calls to restore a target CAR within a specified number of months and return a full amortisation schedule [Capital Management]

**I14. Consolidated Group Reporting** — consolidate balance sheets across a primary SACCO and subsidiary entities (eliminating intra-group balances), recompute group-level ratios, and provide entity-level drill-down [Multi-Entity Compliance]

**I15. Automated Audit Trail & Evidence Package** — emit a per-ratio evidence record (source fields, formula, result, threshold, outcome, ledger snapshot hash) bundled into a signed JSON manifest exportable as a ZIP [Governance / Audit]

## New Methods

Three high-impact async methods from the v2.0 roadmap, illustrating signature and usage patterns.

### `stress_test_capital_adequacy`

Model capital impact of simultaneous shocks before filing.

```python
from capabilities.fintech.sacco.reg.service import SACCARegulatoryService, StressScenario

svc = SACCARegulatoryService("my-sacco")

scenarios = [
    StressScenario(name="Base",       npl_spike_pct=5,  deposit_run_pct=0,  securities_haircut_pct=0),
    StressScenario(name="Moderate",   npl_spike_pct=10, deposit_run_pct=10, securities_haircut_pct=5),
    StressScenario(name="Severe",     npl_spike_pct=20, deposit_run_pct=20, securities_haircut_pct=15),
]

result = await svc.stress_test_capital_adequacy("my-sacco", scenarios, as_of_date="2025-03-31")
# result.first_breach_scenario  -> "Severe"
# result.delta_tables[scenario] -> {ratio: (before, after, delta)}
for s in result.delta_tables:
    print(f"{s.name}: CAR {s.pre_car:.2f}% → {s.post_car:.2f}% (Δ {s.delta_car:.2f}%)")
```

### `generate_corrective_action_plan`

Produce a structured CAP within the SASRA-mandated 30-day window.

```python
cap = await svc.generate_corrective_action_plan("my-sacco", as_of_date="2025-03-31")

# cap.breaches          -> list of breached ratios with shortfall amounts
# cap.action_items      -> ranked remediation steps (member capital call, loan recovery, etc.)
# cap.milestones        -> [30-day checkpoint, 60-day checkpoint]
# cap.pdf_export_dict   -> ready for PDF renderer

for item in cap.action_items:
    print(f"[{item.priority}] {item.description}: {item.target_amount:,.0f} KES by {item.due_date}")
```

### `generate_audit_evidence_package`

Build a tamper-evident evidence bundle for SASRA examination response.

```python
pkg = await svc.generate_audit_evidence_package("my-sacco", year=2025, quarter=1)

# pkg.records           -> list[EvidenceRecord] — one per ratio
# pkg.manifest_hash     -> SHA-256 of ledger snapshot at computation time
# pkg.zip_bytes         -> bytes — write directly to disk or attach to email

for rec in pkg.records:
    print(f"{rec.ratio_name}: {rec.result:.2f}% | compliant={rec.compliant} | hash={rec.ledger_hash[:8]}")

with open("audit_evidence_2025_Q1.zip", "wb") as f:
    f.write(pkg.zip_bytes)
```

## Tests

```bash
python -m pytest capabilities/fintech/sacco/reg/tests/ -v
```
