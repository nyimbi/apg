# Agency Banking — World-Class Improvement Catalogue

© 2025 Datacraft · Author: Nyimbi Odero · www.datacraft.co.ke

---

## 1. Real-Time Float Liquidity Forecasting

**Category**: Liquidity Intelligence

**Justification**: Agent float shortfalls are the #1 cause of transaction failures in African agency networks. Equity Bank's agent network (Kenya) and Capitec (South Africa) both built predictive float models that cut cashout failure rates by 30–40 %. Reactive top-up is too slow; predictive liquidity management prevents revenue loss.

**Implementation**: Collect per-outlet daily transaction velocity over a 90-day rolling window. Fit a lightweight autoregressive model (simple exponential smoothing or Holt-Winters) per outlet. Emit `float_topup_alert` when predicted balance-at-end-of-day drops below 2× the configured threshold. Expose `async forecast_float_needs(tenant_id, horizon_days)` returning ranked outlet alerts.

**Competitor**: Cellulant's Tingg platform; Wave Mobile Money (Senegal/Francophone Africa).

---

## 2. Tiered Commission Engine with Real-Time Accrual

**Category**: Revenue Management

**Justification**: Fixed-rate commission is a commodity; tiered schedules reward high-volume agents and anchor network stickiness. Safaricom M-PESA's tiered agent commission tables (publicly documented) created a competitive moat that took MTN MoMo years to replicate.

**Implementation**: Store a `CommissionTier` table per program with volume brackets and basis-point rates. On every `record_transaction` call, look up the current-period cumulative volume for the outlet and apply the matching tier rate. Accrue in real time to `CommissionSettlement` ledger. Expose `async compute_tiered_commission(outlet_id, period)` returning accrued amount and current tier.

**Competitor**: MTN MoMo Agent Commission Framework; Airtel Money tiered incentive schedules.

---

## 3. Geospatial Agent Network Optimisation

**Category**: Network Expansion

**Justification**: Regulators (CBK, Bank of Tanzania, BOU) require demonstration of financial inclusion depth — outlets per 10,000 population, proximity to underserved areas. Networks that quantify coverage gaps win franchise agreements.

**Implementation**: Store latitude/longitude on `AgencyOutlet`. Implement `async geo_gap_analysis(country, grid_resolution_km)` that bins outlets into grid cells, counts population (via a GeoJSON reference dataset), and returns cells with outlet density below the regulatory target. Flag candidates for new outlet deployment.

**Competitor**: Equity Bank GIS coverage dashboards; Mastercard's Financial Inclusion Gateway.

---

## 4. Behavioural Fraud Velocity Rules Engine

**Category**: Risk & Fraud

**Justification**: Agency fraud — agent collusion, round-tripping, cash substitution — costs networks 0.3–0.8 % of gross volume annually. Pure AML screening misses intra-agent patterns; velocity rules catch them within seconds.

**Implementation**: On every `record_transaction`, compute per-agent rolling windows (last 1 h, 24 h, 7 d) for: transaction count, total value, unique customers, reversal rate. Compare against per-service velocity limits stored in config. Emit a `fraud_velocity_alert` event and set a `review_hold` flag when any window is breached. Expose `async evaluate_agent_fraud_velocity(agent_id, tenant_id)` returning breach details and recommended action.

**Competitor**: Interswitch Fraud Intelligence; FIS Fraud Management; Temenos Financial Crime Mitigation.

---

## 5. Automated Regulatory Reporting Pack (CBK/BoU/BoT/BNR)

**Category**: Regulatory Compliance

**Justification**: Manual preparation of CBK Agency Banking Monthly Returns takes 3–5 days at mid-sized FIs. Automating extraction and formatting cuts that to minutes and eliminates transcription errors, reducing regulatory risk.

**Implementation**: Implement a `RegRptTemplate` registry keyed by `(country, regulator, report_type)`. Each template maps source fields to the regulator's prescribed columns, applies any required aggregations, and formats output as XLSX/PDF. Expose `async generate_regulatory_report(country, report_type, period, tenant_id)`. Pre-built templates for: CBK Agency Banking Return, BoU e-money report, BoT agent network return, BNR digital payment report.

**Competitor**: Oracle OFSAA; Wolters Kluwer OneSumX; Temenos Regulatory Reporting.

---

## 6. Multi-Tier Agent Hierarchy Management

**Category**: Network Governance

**Justification**: Super-agent → master agent → sub-agent structures are required under CBK Agent Banking Guidelines (Regulation 32) and are standard in M-PESA, Equity, and Co-op Bank networks. Flat agent models cannot express liability chains or float-lending relationships.

**Implementation**: Add `parent_agent_id: str | None` and `hierarchy_level: int` to `AccreditedAgent`. Implement `async assign_sub_agent(parent_agent_id, child_agent_id, tenant_id)` enforcing that the parent holds sufficient float credit and has supervisor role. Expose `async get_agent_hierarchy(root_agent_id, tenant_id)` returning the full tree with aggregate volume and float stats at each node.

**Competitor**: Cellulant Super-Agent Framework; Interswitch Quickteller agent hierarchy.

---

## 7. Dynamic Transaction Limit Engine

**Category**: Risk & Limits

**Justification**: Static KES 200,000 daily limits are blunt instruments. High-performing, long-tenured agents with clean compliance records should earn higher limits dynamically, reducing friction for premium customers while retaining guardrails for new agents.

**Implementation**: Compute an `AgentLimitScore` from: tenure (days since accreditation), rolling 90-day compliance score, dispute rate, fraud velocity score, and average monthly volume. Map the score to a limit tier (e.g., score ≥ 85 → KES 500,000; ≥ 70 → KES 350,000; default KES 200,000). Recompute nightly. Expose `async compute_dynamic_limit(agent_id, tenant_id)` and cache result on the agent record.

**Competitor**: Mastercard Dynamic Limits API; Visa Transaction Controls.

---

## 8. Offline Transaction Queue with Cryptographic Reconciliation

**Category**: Resilience

**Justification**: Agent POS devices in low-connectivity zones (rural Kenya, Northern Uganda) must be able to queue transactions offline. Without a cryptographically secured queue, offline transactions cannot be trusted on reconnection.

**Implementation**: Assign each outlet a device-specific HMAC key (via `keym`). Offline transactions are signed with a sequence number and queued locally. On reconnection, `async reconcile_offline_queue(outlet_id, signed_batch, tenant_id)` verifies each message's HMAC and sequence integrity, then replays valid transactions in order, rejecting tampered or out-of-sequence entries.

**Competitor**: Ecobank OMNI Lite offline mode; KopoKopo offline collection.

---

## 9. Customer-Level Spend Analytics and Nudges

**Category**: Customer Intelligence

**Justification**: Agents who can show customers their spending patterns (utility bills, airtime, loan repayments) sell more services and improve retention. First Bank Nigeria and Standard Bank's agent networks use customer analytics to cross-sell credit products at the point of service.

**Implementation**: Build per-customer aggregated spend views: monthly totals by service, top biller, frequency. Expose `async customer_spend_profile(customer_id, period, tenant_id)` returning the profile. Generate nudge messages (e.g., "You have paid KPLC KES 3,200 this month — consider pre-paying before month-end.") using template strings.

**Competitor**: Standard Bank Agent Analytics; Zenith Bank AgentPay analytics.

---

## 10. Float Insurance and Credit Facility Integration

**Category**: Liquidity Products

**Justification**: Under-capitalised agents churn out of networks because they cannot sustain float. Equity's Equitel and KCB's mobi bank both offer intraday float credit lines to agents with good track records, dramatically reducing churn and increasing transaction capacity.

**Implementation**: Model a `FloatCreditFacility` linked to an outlet with limit, drawdown balance, and daily interest rate. Implement `async apply_for_float_credit(outlet_id, requested_limit, tenant_id)` that evaluates the outlet's 90-day volume history and compliance score. On approval, top up float and accrue interest daily. Expose `async repay_float_credit(outlet_id, amount, tenant_id)`.

**Competitor**: Equity Bank Float Advance; KCB Agent Banking Credit.

---

## 11. End-to-End Transaction Reversal Workflow

**Category**: Operations / Dispute Resolution

**Justification**: Transaction reversals are high-risk and frequently abused. Without a structured, multi-party approval workflow (agent request → supervisor review → finance authorisation → ledger reversal), networks incur both financial loss and regulatory censure.

**Implementation**: Extend `AgencyDispute` with a `reversal_requested: bool` flag and a `reversal_approval_chain: list[ApprovalStep]` field. Implement `async request_transaction_reversal(transaction_id, reason, evidence, tenant_id)`, `async approve_reversal_step(dispute_id, approver_id, step, tenant_id)`, and `async execute_reversal(dispute_id, payment_reference, tenant_id)`. All steps emit auditable events.

**Competitor**: Flutterwave Refund API; Stripe Dispute Resolution; Paystack Refund workflow.

---

## 12. Agent Performance Scoring and Gamification

**Category**: Network Engagement

**Justification**: Agent churn in emerging-market agency networks runs 25–40 % annually. Leaderboards, badges, and performance tiers (Bronze/Silver/Gold/Platinum) that unlock better commission rates are proven retention levers — used by M-PESA Lipa Na M-PESA, Airtel Money, and MTN MoMo.

**Implementation**: Compute a weekly `AgentPerformanceScore` from: transaction volume (40 %), customer growth (20 %), compliance score (20 %), and float utilisation (20 %). Assign tiers and badges. Expose `async get_agent_leaderboard(program_id, period, top_n, tenant_id)` returning ranked agents with score breakdown and tier badge. Emit `agent_tier_upgraded` events for downstream notification.

**Competitor**: M-PESA Super-Agent Program; MTN MoMo Star Agent.

---

## 13. Interoperability Gateway for Third-Party Aggregators

**Category**: Composability / Integration

**Justification**: Licensing a single agency network to multiple MFIs, SACCOs, and banks multiplies revenue without multiplying costs. PesaLink, Interswitch, and Cellulant operate shared agent networks on exactly this model.

**Implementation**: Model a `NetworkSharingAgreement` between programs from different tenants. Implement `async register_shared_outlet(outlet_id, guest_program_id, guest_tenant_id, host_tenant_id)` and transaction routing logic that correctly attributes float, commission, and settlement to the originating program while clearing through the shared outlet. Full audit trail on both tenants.

**Competitor**: PesaLink Shared Agent Network; Cellulant Shared Agent Hub.

---

## 14. AI-Assisted Suspicious Activity Report (SAR) Generation

**Category**: Compliance / AI

**Justification**: Filing SARs manually takes compliance teams 2–4 hours per case. FIs regulated under POCAMLA (Kenya), AML Act (Uganda), and FATF recommendations are obligated to file within 48 hours. AI-drafted SARs cut this to minutes while ensuring regulatory completeness.

**Implementation**: Implement `async draft_sar(transaction_ids, agent_id, reason, tenant_id)` which aggregates the transaction chain, dispute history, customer profile, and velocity signals. Feed the structured context into a locally-hosted Ollama model (e.g., Mistral-7B-Instruct) to generate a narrative. Return `draft_sar_text`, `supporting_evidence_refs`, and a `completeness_score`. Human reviewer required before submission.

**Competitor**: NICE Actimize SAR generation; Oracle Financial Services Anti Money Laundering.

---

## 15. Carbon-Credit Micro-Offset Programme for Agent Transactions

**Category**: ESG / Innovation

**Justification**: ESG mandates are entering emerging-market financial regulation. Agents processing transactions for smallholder farmers, clean-energy retailers, and rural health workers can generate Verified Carbon Units (VCU) via a micro-offset pool. This is a novel revenue stream with zero marginal cost for the network operator — and a first-mover ESG differentiator.

**Implementation**: Maintain a `CarbonOffsetPool` per program with an optional per-transaction levy (e.g., 0.01 % of value, capped per transaction). Implement `async record_carbon_contribution(transaction_id, offset_amount_kg_co2, tenant_id)` and `async generate_esg_impact_report(program_id, period, tenant_id)` summarising total offset credits, equivalent trees, and beneficiary categories. Integrate with Verra or Gold Standard API for optional VCU issuance.

**Competitor**: Mastercard Carbon Calculator; Doconomy True Cost API.
