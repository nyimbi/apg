# Agency Banking

## Overview

Agency Banking extends financial services reach through a network of accredited third-party outlets — retail shops, pharmacies, petrol stations, mobile agents, cooperatives, and community banks — operating under a governed program structure. Each outlet holds a float account, serves KYC/AML-verified customers, and processes transactions across services including cash-in/out, bill payment, airtime, loan disbursement, card services, and government payments.

The capability manages the full lifecycle: program registration and outlet onboarding, agent accreditation and float management, transaction recording, commission settlement, dispute handling, field supervision visits, AI-assisted agent workflows, and v2.0 network intelligence features. All activity streams to `apg.fintech.agency.lifecycle` via Bytewax.

## Capability ID

`fintech_agency`  Version: 2.0.0

## Provides

| Service | Description |
|---------|-------------|
| agency_program_governance | Register and govern agency programs with owner, country, currency, and settlement-model constraints |
| agency_outlet_lifecycle | Onboard outlets with license, location, and security-plan evidence; manage channel assignments |
| agency_agent_accreditation | Accredit human agents with identity, training, and background-check evidence |
| agency_float_management | Open and reconcile float accounts linked to outlets with ledger references |
| agency_customer_workflow | Onboard customers with KYC, AML, fraud, and consent evidence under tier controls |
| agency_transaction_workflow | Record channel transactions with float sufficiency checks and high-value approval gates |
| agency_cash_movement_workflow | Record float top-ups, cash pickups, vault rebalances with custodian assignment |
| agency_commission_settlement_workflow | Settle outlet commissions with reconciliation and payment-reference evidence |
| agency_dispute_workflow | Open and resolve agent transaction disputes with evidence and reviewer assignment |
| agency_supervision_workflow | Record field supervision visits with outcome tracking and remediation plans |
| agency_ai_agent_workflow | Register and govern AI agents acting in agency banking roles |
| agency_liquidity_intelligence | Predictive float forecasting, rebalancing, and credit facility management |
| agency_network_analytics | Network-wide performance scoring, leaderboards, and geo coverage analysis |
| agency_regulatory_reporting | CBK/BoU/BoT/BNR automated return generation |
| agency_fraud_velocity | Behavioural fraud velocity rules with real-time breach detection |
| agency_esg_reporting | Carbon-credit micro-offset pool and ESG impact reporting |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication and session management |
| audl | Immutable audit trail |
| ntfy | Customer and agent notifications |
| nlpc | NLP for narrative and evidence analysis |
| keym | Key management (also used for offline queue HMAC) |
| fintech_payments | Payment execution for transactions |
| fintech_wallets | Wallet backing for float accounts |
| fintech_cards | Card service operations at outlets |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening for customers and transactions |
| fintech_fraud | Fraud signal scoring |
| fintech_remittance | Cross-border remittance via agent networks |
| fintech_neobanking | Account opening at outlets |
| fintech_lending | Loan disbursement, collection, and float credit facilities |

## Quick Start

```python
import asyncio
from capabilities.fintech.agency.service import AgencyBankingService

svc = AgencyBankingService()

async def main():
    # Register a program
    svc.register_program(
        program_id="prog-1", tenant_id="acme",
        name="ACME Agency", owner_id="owner-1",
        country="KE", currency="KES",
        settlement_model="daily_net",
        services=["cash_in", "cash_out", "bill_payment"],
    )

    # Register an agent (creates outlet automatically)
    agent = await svc.register_agent(
        name="Jane Mwangi", location="Kiambu Road, Nairobi",
        phone="+254712345678", id_number="12345678",
        float_account="fa-ref-001", tenant_id="acme",
        program_id="prog-1",
    )

    # Check float and process a deposit
    balance = await svc.agent_float_check(agent["id"], tenant_id="acme")
    deposit  = await svc.customer_deposit(
        agent_id=agent["id"], customer_phone="+254700111222",
        amount=5000, tenant_id="acme",
    )

asyncio.run(main())
```

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-agency/dashboard | GET | fintech_agency:view | Overview |
| programs | /fintech-agency/programs | GET/POST | fintech_agency:manage_programs | Programs |
| outlets | /fintech-agency/outlets | GET/POST | fintech_agency:manage_outlets | Network |
| agents | /fintech-agency/agents | GET/POST | fintech_agency:manage_agents | Network |
| float_accounts | /fintech-agency/float-accounts | GET/POST | fintech_agency:float | Liquidity |
| customers | /fintech-agency/customers | GET/POST | fintech_agency:customers | Customers |
| transactions | /fintech-agency/transactions | GET/POST | fintech_agency:transactions | Transactions |
| cash_movements | /fintech-agency/cash-movements | GET/POST | fintech_agency:liquidity | Liquidity |
| commissions | /fintech-agency/commissions | GET/POST | fintech_agency:commissions | Settlement |
| disputes | /fintech-agency/disputes | GET/POST | fintech_agency:disputes | Servicing |
| supervision | /fintech-agency/supervision | GET/POST | fintech_agency:supervision | Field Control |
| ai_agents | /fintech-agency/ai-agents | GET/POST | fintech_agency:admin | Automation |
| settings | /fintech-agency/settings | GET/POST | fintech_agency:admin | Administration |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| transactions.daily_limit | number | 200000 | Per-outlet daily transaction ceiling |
| transactions.high_value_threshold | number | 100000 | Threshold requiring human approval |
| cash_movements.high_value_threshold | number | 100000 | Cash movement approval threshold |
| outlets.minimum_initial_float | number | 500 | Minimum float to onboard outlet |
| customers.supported_tiers | list | tier_1, tier_2, tier_3 | Allowed KYC tiers |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| tenant_context_required | No tenant context present | deny |
| agency_write_requires_policy | Write without policy evidence | deny |
| outlet_initial_float_valid | Below minimum initial float | deny |
| transaction_float_sufficient | Cash-out with insufficient float | deny |
| transaction_limit_valid | Amount exceeds daily limit | deny |
| transaction_currency_matches_float | Currency differs from float account | deny |
| high_value_transaction_requires_approval | Amount > 100,000 without human approval | require_review |
| high_value_cash_movement_requires_approval | Cash movement > 100,000 without approval | require_review |
| supervision_findings_require_remediation | Findings present without remediation plan | require_review |
| agency_batch_requires_bytewax | Batch without Bytewax stream | deny |
| privileged_agency_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| AgencyProgram | id, tenant_id, name, owner_id, country, currency, settlement_model, services, status |
| AgencyOutlet | id, program_id, outlet_type, country, license_reference, location_reference, initial_float, status |
| AccreditedAgent | id, outlet_id, name, identity_reference, training_reference, background_check_reference, status |
| FloatAccount | id, outlet_id, currency, ledger_reference, balance, status |
| AgencyCustomer | id, outlet_id, customer_reference, kyc_profile_id, tier, consent_reference, aml_reference, fraud_reference |
| AgencyTransaction | id, outlet_id, agent_id, customer_id, float_account_id, service, channel, currency, amount, status |
| CashMovement | id, outlet_id, movement_type, currency, amount, custodian_id, status |
| CommissionSettlement | id, outlet_id, currency, amount, reconciliation_reference, payment_reference |
| AgencyDispute | id, transaction_id, reason, evidence_references, reviewer_id, status |
| SupervisionVisit | id, outlet_id, supervisor_id, outcome, findings, remediation_plan, evidence_references |

## Streaming Events

Events emitted to the fintech event stream via Bytewax.

| Event | Trigger |
|-------|---------|
| agency_program_registered | New program created |
| agency_outlet_onboarded | Outlet passes onboarding checks |
| agency_agent_accredited | Agent accreditation approved |
| float_account_opened | Float account opened for outlet |
| agency_customer_onboarded | Customer enrolled at outlet |
| agency_transaction_recorded | Transaction posted |
| cash_movement_recorded | Float cash movement recorded |
| commission_settlement_recorded | Commission paid to outlet |
| agency_dispute_opened | Dispute filed against transaction |
| supervision_visit_recorded | Field supervision visit recorded |
| agency_ai_agent_registered | AI agent registered for agency role |
| float_topup_alert | Predicted EOD balance below 2× threshold |
| fraud_velocity_alert | Agent velocity window breached |
| agent_tier_upgraded | Agent promoted to higher performance tier |

## New Methods

### Float top-up and balance check

```python
# Top up an agent's float from a bank disbursement
topup = await svc.agent_float_top_up(
    agent_id="agent-12345678-2025-01-01",
    amount=50_000,
    source="bank_transfer",
    tenant_id="acme",
    approved_by="ops-manager-1",
)
# {"agent_id": ..., "prior_balance": 3200, "new_balance": 53200, ...}

# Check balance + threshold status + recent top-up history
balance = await svc.agent_float_check("agent-12345678-2025-01-01", tenant_id="acme")
# {"available_balance": 53200, "below_threshold": False, "recent_topups": [...], ...}
```

### Compliance check and dormancy management

```python
# Six-point compliance check — returns score and actionable issues
report = await svc.agent_compliance_check("agent-12345678-2025-01-01", tenant_id="acme")
# {"compliance_score": 83.3, "issues": ["recent_supervision_visit"], "compliant": False}

# Flag agents with no transactions in the last 60 days
dormancy = await svc.dormant_agent_management(days_inactive=60, tenant_id="acme")
# {"dormant_count": 4, "active_count": 18, "dormant_agents": [...]}
```

### Network-wide analytics

```python
analytics = await svc.agent_network_analytics(period="2025-Q1", tenant_id="acme")
# {
#   "agent_count": 22, "total_volume": 4_820_000, "total_commission_payable": 14460,
#   "top_agents": [{"agent_id": ..., "volume": 310000}, ...],
#   "outlet_type_distribution": {"retail_agent": 12, "pharmacy": 5, ...},
#   "avg_compliance_score": 88.1,
# }
```

### Regulatory reporting and float rebalancing

```python
# CBK Agency Banking monthly return (draft)
cbk = await svc.cbk_agency_banking_return(period="2025-01", tenant_id="acme")
# {"report_type": "CBK_AGENCY_BANKING_RETURN", "cash_in": 2100000, "cash_out": 1850000, ...}

# Rebalance float: drain excess outlets, top up deficient ones
rebalance = await svc.float_rebalancing(tenant_id="acme")
# {"accounts_reviewed": 22, "transfers": [{"from_outlet": ..., "to_outlet": ..., "amount": 12500}]}
```

### Specialised transaction helpers

```python
# Airtime
airtime = await svc.mobile_money_airtime_purchase(
    agent_id=agent["id"], customer_id="cust-001",
    phone="+254700111222", amount=500,
    provider="safaricom", tenant_id="acme",
)

# Government payments (eCitizen, NHIF, NSSF, county)
gov = await svc.government_payments_agent(
    agent_id=agent["id"], customer_id="cust-001",
    service_code="NHIF", amount=1700,
    reference="nhif-ref-XYZ", tenant_id="acme",
)
```

## World-Class Enhancements (v2.0)

15 production-grade improvements modelled on Equity Bank, M-PESA, MTN MoMo, and CBK regulatory requirements:

| # | Enhancement | Category | Key Method |
|---|-------------|----------|------------|
| 1 | **Real-Time Float Liquidity Forecasting** — Holt-Winters per-outlet model; emits `float_topup_alert` when predicted EOD balance < 2× threshold | Liquidity Intelligence | `forecast_float_needs(tenant_id, horizon_days)` |
| 2 | **Tiered Commission Engine** — Volume-bracket basis-point rates with real-time accrual; rewards high-volume agents | Revenue Management | `compute_tiered_commission(outlet_id, period)` |
| 3 | **Geospatial Network Optimisation** — Grid-cell outlet density vs. population; surfaces regulatory coverage gaps | Network Expansion | `geo_gap_analysis(country, grid_resolution_km)` |
| 4 | **Behavioural Fraud Velocity Rules** — Rolling 1h/24h/7d windows on count, value, unique customers, reversal rate | Risk & Fraud | `evaluate_agent_fraud_velocity(agent_id, tenant_id)` |
| 5 | **Automated Regulatory Reporting** — Pre-built CBK, BoU, BoT, BNR templates; XLSX/PDF output in minutes | Regulatory Compliance | `generate_regulatory_report(country, report_type, period, tenant_id)` |
| 6 | **Multi-Tier Agent Hierarchy** — Super-agent → master → sub-agent with float-credit and liability chain enforcement | Network Governance | `assign_sub_agent(...)` / `get_agent_hierarchy(...)` |
| 7 | **Dynamic Transaction Limit Engine** — Tenure, compliance score, dispute rate → limit tier; nightly recompute | Risk & Limits | `compute_dynamic_limit(agent_id, tenant_id)` |
| 8 | **Offline Transaction Queue** — Device HMAC keys via `keym`; cryptographic sequence verification on reconnection | Resilience | `reconcile_offline_queue(outlet_id, signed_batch, tenant_id)` |
| 9 | **Customer Spend Analytics & Nudges** — Monthly spend by service, top biller, frequency; template nudge messages | Customer Intelligence | `customer_spend_profile(customer_id, period, tenant_id)` |
| 10 | **Float Credit Facility** — Intraday credit lines for under-capitalised agents; daily interest accrual | Liquidity Products | `apply_for_float_credit(...)` / `repay_float_credit(...)` |
| 11 | **Transaction Reversal Workflow** — Multi-party approval chain (agent → supervisor → finance) before ledger reversal | Operations | `request_transaction_reversal(...)` / `execute_reversal(...)` |
| 12 | **Agent Performance Scoring & Gamification** — Weekly score (volume 40%, growth 20%, compliance 20%, float 20%); Bronze/Silver/Gold/Platinum tiers | Network Engagement | `get_agent_leaderboard(program_id, period, top_n, tenant_id)` |
| 13 | **Interoperability Gateway** — Shared outlet routing across MFIs/SACCOs/banks; dual-tenant audit trail | Composability | `register_shared_outlet(...)` |
| 14 | **AI-Assisted SAR Generation** — Ollama (Mistral-7B-Instruct) drafts POCAMLA/FATF-compliant SARs with completeness score | Compliance / AI | `draft_sar(transaction_ids, agent_id, reason, tenant_id)` |
| 15 | **Carbon-Credit Micro-Offset Programme** — Per-transaction levy → VCU pool; Verra/Gold Standard API integration | ESG / Innovation | `generate_esg_impact_report(program_id, period, tenant_id)` |

## Composability

- **Upstream**: `fintech_kyc` and `fintech_aml` provide identity and AML evidence before customers can be onboarded; `fintech_fraud` provides fraud signals per transaction
- **Downstream**: `fintech_payments` and `fintech_wallets` execute underlying money movement; `fintech_lending` uses agent outlets as disbursement/collection points and backs float credit facilities; `fintech_cards` enables card services at physical outlets
- **Peer**: Deployed alongside `fintech_mobile` (USSD/mobile channels feeding agency transactions) and `fintech_remittance` (cross-border payouts via agent networks)

## Edge Cases Handled

- Cash-out blocked when available float is insufficient — check fires before posting; callers must compute `float_sufficient` in context before invoking the rule engine
- Transaction currency must match float account currency; mismatches are denied even if both appear in `SUPPORTED_CURRENCIES`
- High-value transactions (>100,000) require a human approval record — not just a flag — before the transaction is posted
- Supervision visits with a non-empty findings list require an attached remediation plan before the visit can be closed
- Agent AI runtimes and roles are validated against allow-lists at registration time; unknown values are denied, not silently ignored
- `same_country` is blocked for cross-border corridors; agency transactions are domestic by design
- Offline queue reconciliation rejects HMAC-invalid or out-of-sequence entries; partial batches are processed up to the first invalid entry

## Development Notes

- All write operations require both tenant context and attached policy evidence — two separate checks, both must pass
- `agency_batch_requires_bytewax` enforces Bytewax as the only valid stream processor; direct DB writes bypass event ordering and are rejected
- Float sufficiency is a caller-computed context flag (`float_sufficient`); the rule engine does not query balances directly
- Condition keys ending in `_ne` invert the match: `event_stream_ne: "bytewax"` fires when the stream is NOT bytewax
- Supported countries and currencies are fixed compile-time lists; adding a new corridor requires updating the constant arrays and redeploying the contract
- v2.0 new async methods (e.g. `forecast_float_needs`, `draft_sar`, `get_agent_leaderboard`) are stubs ready for full implementation; core CRUD methods are production-ready
