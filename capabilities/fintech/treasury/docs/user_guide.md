# Treasury Management System — User Guide

**Capability ID**: `fintech_treasury` | **Domain**: `fintech` | **Version**: `1.2.0`
**Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Introduction

The `fintech_treasury` capability provides a complete corporate treasury management system covering:

- Multi-currency cash positioning and cash pooling
- FX dealing (forwards, options, swaps) with full pricing and Greeks
- Liquidity risk management: LCR, NSFR, Cash Flow at Risk, maturity ladder
- ALCO governance with four-eyes approval workflow
- Basel III / CBK regulatory reporting
- SWIFT message handling and gpi payment tracking
- Nostro reconciliation (MT940 matching engine)
- Transfer pricing with CUP arm's-length benchmarking
- AI-powered treasury co-pilot via locally-hosted Ollama

All business logic lives in `service.py` as `async` methods on `CorporateTreasuryService`.
The capability follows the APG adapter pattern — auth, audit, and notify are injected dependencies.

---

## 2. Installation

```bash
pip install apg-fintech-treasury
```

For development:

```bash
cd capabilities/fintech/treasury
uv sync
uv run pytest tests/ -q
```

---

## 3. Quick Start

```python
import asyncio
from apg_fintech_treasury import CorporateTreasuryService

async def main():
    svc = CorporateTreasuryService(
        db_url="postgresql+asyncpg://user:pass@localhost/treasury",
        tenant_id="my_entity",
    )

    # 1. Cash position
    pos = await svc.cash_position("ENTITY-1", "2026-06-01", ["KES", "USD"])
    print(f"KES confirmed: {pos['positions']['KES']['confirmed']:,.0f}")

    # 2. Liquidity forecast
    forecast = await svc.liquidity_forecast("ENTITY-1", days=30)
    print(f"Net 30-day position: {forecast['net_position']:,.0f}")

    # 3. LCR calculation
    lcr = await svc.lcr_daily_calculation("ENTITY-1", "2026-06-01")
    print(f"LCR: {lcr['lcr_pct']:.1f}%  (compliant: {lcr['lcr_compliant']})")

asyncio.run(main())
```

---

## 4. Core Modules

### 4.1 Cash Management

#### Cash Position
```python
pos = await svc.cash_position(
    entity_id="ENTITY-1",
    as_of_date="2026-06-01",
    currencies=["KES", "USD", "EUR"],
)
# Returns: positions per currency — confirmed, float_amt, total
```

#### Cash Pooling
```python
# Physical zero-balancing sweep
sweep = await svc.cash_pooling(
    pool_id="POOL-001",
    value_date="2026-06-01",
    method="physical",         # or "notional"
)
```

#### Intraday Liquidity Monitoring
```python
monitor = await svc.intraday_liquidity_monitoring("ENTITY-1")
print(f"RTGS buffer: {monitor['buffer']:,.0f} KES — {monitor['status']}")
```

---

### 4.2 FX Dealing

#### Book FX Forward
```python
deal = await svc.fx_forward_booking(
    entity_id="ENTITY-1",
    buy_currency="USD",
    sell_currency="KES",
    amount=1_000_000.0,
    settlement_date="2026-09-01",
    forward_rate=131.25,
)
```

#### FX Option Pricing (Garman-Kohlhagen)
```python
option = await svc.fx_option_price(
    entity_id="ENTITY-1",
    spot=130.5,
    strike=133.0,
    domestic_rate_pct=10.5,     # KIBOR
    foreign_rate_pct=5.25,      # SOFR
    vol_pct=8.5,
    tenor_days=90,
    option_type="call",
    currency_pair="USD/KES",
    notional=5_000_000.0,
)
print(f"Premium: KES {option['premium_total']:,.2f}")
print(f"Delta: {option['greeks']['delta']:.4f}")
print(f"Vega: {option['greeks']['vega']:.4f} per 1% vol")
```

#### FX Rate Feed
```python
rates = await svc.fx_rate_feed(["USD/KES", "EUR/KES", "GBP/KES"])
```

---

### 4.3 Hedging

#### Create Hedge Instrument
```python
hedge = await svc.hedge_instrument_create(
    instrument_type="fx_forward",    # fx_forward | fx_option | interest_rate_swap | cross_currency_swap
    notional=10_000_000.0,
    currency_pair="USD/KES",
    strike=131.50,
    maturity="2026-12-31",
    entity_id="ENTITY-1",
    counterparty_id="BANK-001",
)
```

#### Hedge Effectiveness Test (IFRS 9)
```python
test = await svc.hedge_effectiveness_test(
    hedge_id=hedge["id"],
    period="2026-Q2",
    method="dollar_offset",         # dollar_offset | regression | hypothetical_derivative
)
print(f"Effectiveness: {test['effectiveness_ratio_pct']:.1f}%  effective: {test['effective']}")
```

#### Cash Flow at Risk (Monte Carlo)
```python
cfar = await svc.cashflow_at_risk(
    entity_id="ENTITY-1",
    horizon_days=90,
    simulations=1_000,
)
print(f"P5 worst-case: KES {cfar['percentiles']['P5']:,.0f}")
print(f"Expected shortfall: KES {cfar['expected_shortfall_p5']:,.0f}")
```

---

### 4.4 Liquidity Risk — LCR and NSFR

#### LCR (Basel III)
```python
lcr = await svc.lcr_daily_calculation("ENTITY-1", "2026-06-01")
# Fields: level1_hqla, level2a_hqla, level2b_hqla, total_hqla,
#         net_cash_outflows_30d, lcr_pct, lcr_compliant, buffer_adequate
```

LCR triggers an email alert when the ratio falls below 100% (regulatory) or 120% (internal buffer).

#### NSFR (Basel III) + Maturity Ladder
```python
nsfr = await svc.nsfr_calculation("ENTITY-1", "2026-06-01")
print(f"NSFR: {nsfr['nsfr_pct']:.1f}%")
for bucket, flows in nsfr['maturity_ladder'].items():
    print(f"  {bucket:>4}: inflow {flows['inflow']:>12,.0f}  outflow {flows['outflow']:>12,.0f}  net {flows['net']:>12,.0f}")
```

---

### 4.5 ALCO Governance

#### Create a Motion
```python
motion = await svc.alco_motion_create(
    entity_id="ENTITY-1",
    motion_type="limit_change",            # limit_change | policy_update | hedge_strategy | ...
    description="Increase USD FX dealing limit from USD 50M to USD 75M",
    proposer_id="treasurer-001",
    participants=["cfo-001", "risk-001", "ceo-001", "treasurer-001"],
    quorum=3,
    meeting_date="2026-06-15",
)
motion_id = motion["id"]
```

#### Vote on a Motion
```python
result = await svc.alco_motion_vote(
    motion_id=motion_id,
    voter_id="cfo-001",
    vote="for",
    rationale="FX volumes have grown 40% YoY; limit is constraining",
)
# Motion auto-resolves when quorum votes cast
print(f"Status: {result['status']}")   # open | approved | rejected
```

---

### 4.6 Regulatory Reporting

#### Basel III Capital Adequacy
```python
report = await svc.regulatory_capital_report("ENTITY-1", "2026-Q2")
print(f"CAR: {report['capital_adequacy_ratio_pct']:.2f}%  compliant: {report['car_compliant']}")
```

#### CBK Return Filing
```python
filing = await svc.cbk_returns_filing(
    entity_id="ENTITY-1",
    period="2026-Q2",
    return_type="capital_adequacy",
    submitted_by="head-of-treasury",
)
```

#### Transfer Pricing Benchmark Rate (CUP Method)
```python
tp_rate = await svc.transfer_pricing_benchmark_rate(
    currency="KES",
    tenor_months=12,
    credit_rating="BBB",
)
print(f"Arm's length range: {tp_rate['arm_length_range']['low_pct']:.2f}% – {tp_rate['arm_length_range']['high_pct']:.2f}%")
```

---

### 4.7 Nostro Reconciliation

```python
# Import MT940 statement entries
statement_entries = [
    {"value_date": "2026-06-01", "amount": 5_000_000.0, "currency": "KES", "reference": "REF001", "direction": "credit"},
    {"value_date": "2026-06-01", "amount": 2_000_000.0, "currency": "KES", "reference": "REF002", "direction": "debit"},
]

recon = await svc.nostro_reconciliation_run(
    account_id="NOSTRO-KES-001",
    statement_entries=statement_entries,
    as_of_date="2026-06-01",
)
print(f"Match rate: {recon['match_rate_pct']:.1f}%")
print(f"Open breaks: {len(recon['open_breaks'])}")
```

---

### 4.8 SWIFT & Payments

#### Send SWIFT Message
```python
msg = await svc.swift_message_send(
    entity_id="ENTITY-1",
    message_type="MT103",           # MT103 | MT202 | MT202COV | MT760 | MT300
    payload={"amount": 500_000, "currency": "USD", "beneficiary": "BANK-XYZ"},
)
uetr = msg["reference"]
```

#### SWIFT gpi Tracking
```python
gpi = await svc.swift_gpi_status_check(uetr=uetr)
print(f"Payment status: {gpi['status']}")   # initiated | in_progress | credited | completed
```

---

### 4.9 AI Treasury Co-Pilot

Requires `OLLAMA_BASE_URL` environment variable pointing to a running Ollama instance (e.g. `http://localhost:11434`).

```python
import os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

recs = await svc.treasury_copilot_recommend(
    entity_id="ENTITY-1",
    focus="placement",     # placement | hedging | funding | all
)

for r in recs["recommendations"]:
    print(f"[{r['priority']}] {r['action']}")
    print(f"    Rationale: {r['rationale']}")
    print(f"    Expected NII impact: +{r['expected_nii_improvement_pct']:.2f}%")
```

Falls back to rule-based heuristics if Ollama is unavailable, ensuring the method is always callable in production.

---

## 5. NATS Streaming Integration

Events are published to NATS (requires `NATS_URL`) using bytewax dataflows:

```
treasury.deals.booked              — FX/MM deal bookings
treasury.risk.var.{entity_id}      — VaR results
treasury.reconciliation.breaks.*   — Nostro open breaks
treasury.alco.motion.{id}          — ALCO notifications
treasury.copilot.recommendations.* — AI recommendations
treasury.swift.gpi.{uetr}          — gpi status transitions
treasury.limits.breach.{dealer_id} — Limit breach alerts
```

---

## 6. Configuration Reference

| Variable | Description |
|----------|-------------|
| `FINTECH_TREASURY_DB_URL` | PostgreSQL `asyncpg` connection URL |
| `NATS_URL` | NATS server for streaming events |
| `OLLAMA_BASE_URL` | Ollama API for AI co-pilot |
| `FINTECH_TREASURY_TENANT_ID` | Tenant namespace (default: `default`) |

---

## 7. Permissions

| Permission | Grants Access To |
|------------|-----------------|
| `fintech_treasury:view` | Dashboards, reports, co-pilot |
| `fintech_treasury:manage_cash` | Cash positioning, pooling |
| `fintech_treasury:deal` | FX/MM deal booking |
| `fintech_treasury:manage_limits` | Dealer and counterparty limits |
| `fintech_treasury:settle` | Payment factory, SWIFT messages |
| `fintech_treasury:manage_fx` | FX rate management, option pricing |
| `fintech_treasury:manage_liquidity` | LCR, NSFR, forecasting |
| `fintech_treasury:reconcile` | Nostro reconciliation |
| `fintech_treasury:alco_vote` | ALCO motion creation and voting |

---

## 8. Testing

```bash
# Unit tests (CI)
uv run pytest tests/ci/ -vxs

# All tests
uv run pytest tests/ -q

# Type checking
uv run pyright
```

Test fixtures in `tests/conftest.py` provide an in-memory store and stub adapters so no external dependencies are needed for CI.

---

## 9. Composability

```apg
use fintech_treasury;
```

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["fintech_treasury"].contract
```

---

## 10. Further Reading

| File | Purpose |
|------|---------|
| `service.py` | Complete business logic implementation |
| `models.py` | SQLAlchemy and Pydantic data models |
| `api.py` | REST API endpoints |
| `views.py` | Flask-AppBuilder views and Pydantic schemas |
| `capability_contract.py` | Governance rules and contract metadata |
| `WORLD_CLASS_IMPROVEMENTS.md` | 15 prioritised capability enhancements roadmap |
| `SPECIFICATION.md` | Detailed capability specification |
| `README.md` | Quick reference |
