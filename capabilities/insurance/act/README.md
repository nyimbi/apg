# Actuarial Tools (ins_act)

Mortality tables, loss ratios, reserve calculations, IBNR, pricing models, experience analysis.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/insurance/act/health | Health check |
| GET | /api/insurance/act/describe | Capability description |
| GET | /api/insurance/act/mortality-tables | List mortality tables |
| POST | /api/insurance/act/mortality-tables | Load mortality table |
| GET | /api/insurance/act/mortality-tables/{id} | Get table |
| DELETE | /api/insurance/act/mortality-tables/{id} | Retire table |
| POST | /api/insurance/act/loss-ratio | Calculate loss ratio |
| GET | /api/insurance/act/loss-ratios | List loss ratio reports |
| POST | /api/insurance/act/reserves | Calculate reserve |
| GET | /api/insurance/act/reserves | List reserves |
| POST | /api/insurance/act/ibnr | Estimate IBNR |
| GET | /api/insurance/act/ibnr | List IBNR estimates |
| POST | /api/insurance/act/pricing-models | Create pricing model |
| GET | /api/insurance/act/pricing-models | List pricing models |
| POST | /api/insurance/act/pricing-models/{id}/apply | Apply model |
| POST | /api/insurance/act/experience-analysis | Run analysis |
| GET | /api/insurance/act/experience-analyses | List analyses |
| GET | /api/insurance/act/summary | Actuarial summary |
| GET | /api/insurance/act/audit | Audit trail |

## World-Class Enhancements (v2.0)

Fifteen targeted improvements that bring ins_act to parity with Willis Towers Watson ResQ, Milliman MG-ALFA, and Guidewire's actuarial suite.

**I1. Full Chain-Ladder Development Factor Engine** — Volume-weighted age-to-age factors, CDF-to-ultimate, per-AY IBNR; replaces the non-defensible stub. [Feature]

**I2. Bornhuetter-Ferguson Reserve Method** — Stabilises IBNR for thin-data AYs; mandatory for IAS 37 reserve adequacy testing. [Feature]

**I3. Mortality Improvement Projection (SOA MP Scales)** — Age-specific annual improvement factors projected forward; required for IFRS 17 / Solvency II longevity risk on policies >10 yr. [Feature]

**I4. Credibility-Weighted Experience Rating (Bühlmann-Straub)** — Full structural parameter derivation; blends observed and prior rates into a filing-ready credibility-adjusted renewal premium. [Feature]

**I5. Scenario-Based Catastrophe Stress Testing** — Named 1-in-200/1-in-250 CAT scenarios re-run reserves under each multiplier; produces audit-trail-ready stress tables for Solvency II SCR / NAIC RBC. [Compliance]

**I6. Multi-Treaty Reinsurance Cession Calculator** — Tower-of-coverage logic for quota-share, surplus, and XL treaties; returns gross/ceded/net split per layer. [Feature]

**I7. LDF Curve Fitting with Tail Factor Extrapolation** — Weighted-average / medial-average LDF selection plus exponential tail extrapolation, AIC-selected; eliminates manual selection bias. [AI/ML]

**I8. Expense Loading and Profit Margin Decomposition** — Computes needed premium from pure premium + expense loads + target profit; returns regulator-ready waterfall. [Feature]

**I9. IFRS 17 Contractual Service Margin (CSM) Tracking** — CSM inception, coverage-unit amortisation, and interest accretion; GMM vs PAA reserve split per contract group. [Compliance]

**I10. Discount Rate Yield Curve Management** — Versioned par/zero curves with bootstrapping; discounts cash-flow schedules to NPV with duration and convexity output. [Feature]

**I11. Solvency II SCR Underwriting Risk Calculator** — Standard formula premium and reserve risk sub-modules with ρ=0.5 correlation aggregation; returns component SCRs and diversification benefit. [Compliance]

**I12. Peer-Review Workflow with Actuarial Sign-Off Locking** — ASOP 41 digital sign-off: immutable lock on approval, full review chain audit trail, version snapshot. [Compliance]

**I13. Real-Time Profitability Snapshot (Dashboard Feed)** — Live combined-ratio, loss-ratio, expense-ratio, premium adequacy index; 60-second BoundedCache; dashboard-ready payload for ins_dashboard. [UX]

**I14. Stochastic Reserve Distribution via Bootstrap Simulation** — 10 000-scenario chain-ladder residual bootstrap; returns p25/p50/p75/p95/p99.5 percentile reserves and CoV per AY. [Feature]

**I15. Actuarial Assumption Change Tracking (Experience Unlock)** — Versioned assumption snapshots with reserve-impact sensitivity delta; IFRS 17 reconciliation waterfall on demand. [Compliance]

## New Methods

The three highest-impact additions planned for v2.0, shown with their async signatures and example usage.

### `compute_chain_ladder` — Full triangle development to ultimate

```python
svc = ActuarialToolsService(tenant_id="acme")

# First upload a cumulative claims triangle
triangle = await svc.upload_claims_triangle(
    tenant_id="acme",
    product_code="WC",
    accident_years=[2019, 2020, 2021, 2022, 2023],
    development_periods=[12, 24, 36, 48, 60],
    incremental_data=[[...], [...], ...],  # 5×5 triangle
)

result = await svc.compute_chain_ladder(
    tenant_id="acme",
    triangle_id=triangle["id"],
)
# result keys: development_factors, cdf_to_ultimate, ibnr_by_accident_year,
#              total_ibnr, tail_factor, created_at
print(result["total_ibnr"])          # Decimal — regulatory-defensible aggregate IBNR
print(result["ibnr_by_accident_year"])  # {2023: Decimal(...), ...}
```

### `bootstrap_reserve_distribution` — Stochastic reserve percentiles for capital modelling

```python
dist = await svc.bootstrap_reserve_distribution(
    tenant_id="acme",
    triangle_id=triangle["id"],
    n_simulations=10_000,
    seed=42,
)
# dist keys: p25, p50, p75, p95, p99_5 (per-AY and aggregate), cov_by_accident_year
print(dist["p99_5"]["aggregate"])    # 99.5th-percentile reserve — feeds Solvency II ORSA
print(dist["cov_by_accident_year"])  # CoV per AY — flags thin-data years
```

### `calculate_scr_underwriting` — Solvency II standard formula SCR

```python
scr = await svc.calculate_scr_underwriting(
    tenant_id="acme",
    product_code="MOT",
    lob="motor_vehicle_liability",
    net_written_premium=Decimal("12_500_000"),
    best_estimate_reserve=Decimal("8_200_000"),
    premium_sigma=Decimal("0.10"),   # EIOPA line-of-business sigma factor
    reserve_sigma=Decimal("0.09"),
)
# scr keys: scr_premium, scr_reserve, diversification_benefit, bscr, created_at
print(scr["bscr"])                   # Basic SCR underwriting component
print(scr["diversification_benefit"])  # Capital saved by ρ=0.5 correlation assumption
```
