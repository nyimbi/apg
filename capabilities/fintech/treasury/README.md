# Treasury Management System

© 2025 Datacraft | Author: Nyimbi Odero

## Overview

`fintech_treasury` is a world-class, standalone-deployable corporate treasury management capability for the APG platform. It covers cash pooling, FX hedging, interest rate management, ALCO governance, liquidity forecasting, regulatory reporting (LCR/NSFR/Basel III), payment factory, SWIFT gpi tracking, and AI-powered decision support via locally-hosted Ollama models.

## Capability ID

`fintech_treasury` | Version: `1.2.0`

## Provides

| Service | Description |
|---------|-------------|
| `cash_position_management` | Intraday and end-of-day cash positioning across currencies |
| `treasury_dealing_workflow` | FX forwards, options, swaps, MM placements — full deal lifecycle |
| `counterparty_limit_governance` | Dealer and counterparty limit monitoring with breach detection |
| `settlement_instruction_workflow` | Payment factory, SWIFT MT103/MT202/MT760, gpi tracking |
| `fx_rate_management` | Rate feeds, FX option pricing (Garman-Kohlhagen), Greeks |
| `liquidity_management` | LCR, NSFR, CFaR, maturity ladder, contingency plan activation |
| `alco_governance` | Motion creation, four-eyes voting, resolution workflow |
| `hedge_accounting` | IFRS 9 effectiveness tests, hedge portfolio summary |
| `regulatory_reporting` | Basel III CAR, CBK returns, LCR/NSFR, transfer pricing (CUP) |
| `nostro_reconciliation` | MT940 statement import, multi-pass matching engine |
| `ai_copilot` | Ollama-backed treasury action recommendations |

## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Authentication and authorisation |
| `audl` | Immutable audit trail |
| `ntfy` | Email/SMS/NATS notifications |
| `keym` | Key management for deal signing |
| `fintech_payments` | Payment settlement integration |

## Installation

```bash
pip install apg-fintech-treasury
```

## Standalone Usage

```python
from apg_fintech_treasury import CorporateTreasuryService

svc = CorporateTreasuryService(db_url="postgresql+asyncpg://user:pass@localhost/treasury")

# Intraday cash position
pos = await svc.cash_position("ENTITY-1", "2026-06-01", ["KES", "USD", "EUR"])

# Liquidity Coverage Ratio
lcr = await svc.lcr_daily_calculation("ENTITY-1", "2026-06-01")

# FX option pricing (Garman-Kohlhagen)
option = await svc.fx_option_price(
    "ENTITY-1", spot=130.5, strike=133.0,
    domestic_rate_pct=10.5, foreign_rate_pct=5.25,
    vol_pct=8.5, tenor_days=90, option_type="call",
    currency_pair="USD/KES", notional=5_000_000,
)

# AI treasury co-pilot (requires OLLAMA_BASE_URL)
recs = await svc.treasury_copilot_recommend("ENTITY-1", focus="placement")
```

## Running the Standalone Server

```bash
# InMemory store (development)
apg-fintech-treasury --port 8080

# PostgreSQL persistence (production)
apg-fintech-treasury --db-url postgresql+asyncpg://user:pass@localhost/treasury --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/fintech-treasury/dashboard` | `fintech_treasury:view` |
| cash_management | `/fintech-treasury/cash` | `fintech_treasury:manage_cash` |
| dealing | `/fintech-treasury/dealing` | `fintech_treasury:deal` |
| limits | `/fintech-treasury/limits` | `fintech_treasury:manage_limits` |
| settlement | `/fintech-treasury/settlement` | `fintech_treasury:settle` |
| fx | `/fintech-treasury/fx` | `fintech_treasury:manage_fx` |
| liquidity | `/fintech-treasury/liquidity` | `fintech_treasury:manage_liquidity` |
| nostro | `/fintech-treasury/nostro` | `fintech_treasury:reconcile` |
| alco | `/fintech-treasury/alco` | `fintech_treasury:alco_vote` |
| copilot | `/fintech-treasury/copilot` | `fintech_treasury:view` |

## HTTP Endpoints

```
GET  /health                  Liveness probe
GET  /contract                Full capability contract JSON
POST /evaluate                Evaluate governance rules
GET  /api/v1/...              Domain-specific REST API
```

## Key Service Methods

### Cash Management
- `cash_position(entity_id, as_of_date, currencies)` — Intraday multi-currency cash position
- `liquidity_forecast(entity_id, days, method)` — Forward cash flow projection
- `cash_pooling(pool_id, value_date, method)` — Notional or physical cash pool sweep
- `intraday_liquidity_monitoring(entity_id)` — Real-time RTGS settlement monitoring

### FX & Derivatives
- `fx_forward_booking(...)` — Book FX forward with rule engine validation
- `fx_option_price(...)` — Garman-Kohlhagen option pricing with full Greeks
- `fx_exposure_report(entity_id, as_of_date)` — FX exposure by currency pair
- `fx_rate_feed(currency_pairs)` — Indicative FX rate snapshot
- `swap_valuation(swap_id, market_rate)` — Mark-to-market swap NPV

### Hedging & Risk
- `hedge_instrument_create(...)` — Book FX forward / option / swap instrument
- `hedge_effectiveness_test(hedge_id, period, method)` — IAS 39 / IFRS 9 effectiveness
- `hedge_portfolio_summary(entity_id)` — Active hedge book summary
- `fx_hedge_effectiveness_report(entity_id, period)` — Period effectiveness statistics
- `scenario_analysis(entity_id, scenario_type, parameters)` — Stress test / what-if
- `cashflow_at_risk(entity_id, horizon_days, simulations)` — Monte Carlo CFaR

### Regulatory & Compliance
- `lcr_daily_calculation(entity_id, as_of_date)` — Basel III LCR with HQLA classification
- `nsfr_calculation(entity_id, as_of_date)` — Basel III NSFR and maturity ladder
- `regulatory_capital_report(entity_id, period)` — CAR / Tier 1 / Tier 2 / RWA
- `cbk_returns_filing(entity_id, period, return_type, submitted_by)` — CBK prudential return
- `covenant_monitoring(facility_id, financial_ratios)` — Financial covenant surveillance
- `transfer_pricing_report(period)` — Intercompany loan TP entries
- `transfer_pricing_benchmark_rate(currency, tenor_months, credit_rating)` — CUP arm's-length rate

### ALCO Governance
- `alco_motion_create(...)` — Create committee motion with quorum threshold
- `alco_motion_vote(motion_id, voter_id, vote)` — Cast and record vote
- *(auto-resolves when quorum reached)*

### Payments & Settlement
- `payment_factory(entity_id, payments, payment_date)` — Batch payment processing
- `swift_message_send(entity_id, message_type, payload)` — Outbound SWIFT message
- `swift_gpi_status_check(uetr)` — gpi end-to-end payment tracking
- `nostro_reconciliation_run(account_id, statement_entries, as_of_date)` — MT940 matching engine
- `netting_calculation(entities, currency, period)` — Multilateral netting

### Funding & Lending
- `money_market_placement(...)` — Place funds at bank
- `intercompany_loan(...)` — Intercompany lending with interest schedule
- `bank_relationship_management(...)` — Facility limit and utilisation
- `cost_of_funds_report(entity_id, period)` — Blended WACOF

### Analytics & Reporting
- `treasury_kpi_dashboard(entity_id)` — KPI dashboard
- `treasury_analytics(entity_id, period)` — Deal and placement performance metrics
- `interest_rate_risk_report(entity_id, as_of_date)` — BPV / repricing gap
- `counterparty_risk_report(entity_id, period)` — Credit exposure by counterparty
- `dealer_limit_monitoring(dealer_id, deal_type)` — Dealing limit utilisation
- `benchmark_rate_submission(entity_id, rate_type, rate_value, submission_date)` — KIBOR submission

### AI Co-Pilot
- `treasury_copilot_recommend(entity_id, focus)` — Ollama-backed action recommendations

## Composability

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["fintech_treasury"].contract
```

```apg
use fintech_treasury;
```

## NATS Integration

The capability publishes events to the following NATS subjects (requires `NATS_URL`):

| Subject | Event |
|---------|-------|
| `treasury.deals.booked` | Every FX/MM deal booking |
| `treasury.risk.var.{entity_id}` | VaR calculation results |
| `treasury.reconciliation.breaks.{account_id}` | Open nostro breaks |
| `treasury.alco.motion.{id}` | ALCO motion notifications |
| `treasury.copilot.recommendations.{entity_id}` | AI recommendations |
| `treasury.swift.gpi.{uetr}` | SWIFT gpi status transitions |
| `treasury.limits.breach.{dealer_id}` | Real-time limit breach alerts |

The platform uses **bytewax + NATS** (not NATS JetStream) for streaming dataflows.

## Configuration

Set via the `conf` capability or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FINTECH_TREASURY_DB_URL` | in-memory | PostgreSQL connection URL |
| `NATS_URL` | — | NATS server URL for event streaming |
| `OLLAMA_BASE_URL` | — | Ollama API URL for AI co-pilot |
| `FINTECH_TREASURY_TENANT_ID` | `default` | Tenant context |

## Development

```bash
# Run tests
uv run pytest tests/ -q

# Type check
uv run pyright

# Build wheel
python -m build --wheel .

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## License

Proprietary — © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Real-Time Intraday Cash Position via NATS Event Sourcing** [Architecture / Cash Management]
- **I2. Monte Carlo VaR Engine for FX Hedge Portfolio** [Risk Analytics]
- **I3. ALCO Committee Decision Workflow with Four-Eyes Approval Chain** [Governance]
- **I4. Dynamic Liquidity Coverage Ratio (LCR) Calculator with HQLA Buffer Tracking** [Regulatory Compliance]
- **I5. Yield Curve Construction and Interest Rate Sensitivity (DV01/BPV) Engine** [Market Risk]
- **I6. Automated Nostro Reconciliation with Matching Engine** [Operations / Settlement]
- **I7. FX Options Pricing (Black-Scholes / Garman-Kohlhagen) with Greeks** [Derivatives Pricing]
- **I8. Cash Flow at Risk (CFaR) with AR/AP Schedule Integration** [Liquidity Risk]
- **I9. Multi-Entity Cash Pooling with Overlay Structure and In-Pool Interest Allocation** [Cash Management]
- **I10. Transfer Pricing Arm's-Length Rate Engine with Comparable Uncontrolled Price (CUP) Method** [Tax / Compliance]
- **I11. Real-Time FX Position Limit Breach Detection via NATS Streaming** [Risk Controls]
- **I12. Net Stable Funding Ratio (NSFR) Calculator with Asset/Liability Maturity Ladder** [Regulatory Compliance]
- **I13. Cross-Currency Basis Swap Pricing and Hedge Accounting Documentation Generator** [Derivatives / Accounting]
- **I14. Treasury Workstation AI Co-Pilot (Ollama-Backed Deal Recommendation Engine)** [AI-Augmented Treasury]
- **I15. Automated SWIFT gpi Tracker Integration with Payment Certainty Dashboard** [Payments / Operations]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
