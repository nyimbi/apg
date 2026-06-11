# Energy Billing (energy_bil) — World-Class Improvements

## 1. Granular Time-of-Use (ToU) Interval Billing
**Category**: Tariff Intelligence
**Justification**: Flat-rate and crude block tariffs leave 15–25% of revenue value on the table vs. real-time marginal cost pricing. Kenya Power's EPRA-mandated MELT tariff reform (2023) already requires interval granularity. Without it, residential NEM customers can game billing windows.
**Implementation**: Accept 15-min or 30-min interval meter reads (DLMS/COSEM format) stored in a time-series store (TimescaleDB). Price each interval against a ToU window schedule (on-peak / mid-peak / off-peak / super-off-peak) loaded per tariff. Aggregate into bill line items per window. Store `interval_billing_id` on `EnergyBill`.
**Competitor**: Itron OpenWay Riva, Oracle Utilities Customer Cloud Service

## 2. Automated Bill Run Scheduler with Cycle Alignment
**Category**: Operations Automation
**Justification**: Manual bill run triggers cause 3–7 day delays in receivables, inflating DSO (Days Sales Outstanding). Automated scheduling aligned to billing cycle calendar (monthly, bi-monthly, seasonal) with cut-off date enforcement closes this gap.
**Implementation**: Persist a `BillingCycleSchedule` model keyed by `(tenant_id, cycle_code)`. A background task (APScheduler / Celery beat) triggers `bulk_generate_bills` on the configured cut-off datetime. Failed runs are journaled to `BillingRunLog` with retry state. Expose `/energy-bil/api/v1/billing-runs` endpoint.
**Competitor**: SAP IS-U, Gentrack Velocity

## 3. Predictive Consumption Estimation with ML Imputation
**Category**: AI/ML
**Justification**: ~8% of smart meter reads fail per month (comms outage, tamper, CRC error). Falling back to flat estimates causes customer disputes and revenue leakage. ML imputation using LSTM or gradient-boosted trees trained on 12-month history reduces estimation error to <2%.
**Implementation**: `estimate_consumption()` async method: (1) pull last N reads from `_consumption_records`; (2) if OLLAMA_BASE_URL present, call local `llama3` / `phi3` with structured prompt requesting kWh estimate + confidence interval; (3) fall back to weighted moving average. Store `estimation_method` and `confidence_pct` on consumption record.
**Competitor**: AutoGrid, Bidgee, Landis+Gyr AIM

## 4. Multi-Currency and FX Rate Management
**Category**: Financial Governance
**Justification**: East African utilities increasingly bill industrial customers in USD or EUR (fuel cost pass-through clauses). Without FX rate snapshots at bill generation time, restatement risk on multi-currency AR is unauditable.
**Implementation**: `FXRateSnapshot` model storing `(base_currency, quote_currency, rate, valid_from, source)`. `generate_bill()` accepts `billing_currency` and `settlement_currency`; stores FX rate at generation time. Expose `record_fx_rate()` and `convert_bill_currency()` service methods. Block bill generation if no valid FX rate for the pair.
**Competitor**: Oracle Utilities, SAP FICA, Zuora

## 5. Net Energy Metering (NEM) Billing for Solar Prosumers
**Category**: Renewable Integration
**Justification**: Kenya's Energy Act 2019 mandates NEM for licensed prosumers. Without NEM billing logic, utilities cannot onboard rooftop solar customers legally or accurately — a growing segment (>40k installations in Kenya as of 2024).
**Implementation**: `calculate_nem_credit()` method: compare export_kwh vs. import_kwh per interval; apply NEM compensation rate (typically retail rate or avoided cost rate per regulatory tariff); net export credit appears as negative charge on bill. Export surplus rolls to next month up to NEM rollover limit. Store `nem_export_kwh`, `nem_credit_amount` on `EnergyBill`.
**Competitor**: Bidgee, Fluentgrid Optimus, Utility Cloud

## 6. Prepaid / Token-Based Vending Integration
**Category**: Payment Models
**Justification**: >60% of KPLC residential customers are on prepaid STS tokens. Without prepaid reconciliation, post-paid billing service cannot handle migration, hybrid, or fallback scenarios. STS (Standard Transfer Specification IEC 62055-41) is the dominant open standard.
**Implementation**: `PrepaidToken` model: `(token_code, units_kwh, amount, vended_at, meter_id, redeemed)`. `vend_token()` method: validate account eligibility, deduct outstanding balance first (NERSA-style forced debt recovery), compute net units, generate 20-digit STS-formatted token. `reconcile_prepaid_vending()` reconciles tokens against meter reads.
**Competitor**: Conlog, Landis+Gyr, Honeywell Elster

## 7. Dunning Workflow Engine with Escalation Tiers
**Category**: Collections
**Justification**: Flat "flag + disconnect notice" logic misses 6 industry-standard escalation tiers (reminder → 1st notice → 2nd notice → disconnect warrant → physical disconnection → legal referral). Each tier has regulatory hold periods (EPRA consumer protection rules). Implementing a proper dunning state machine reduces write-offs by 15–30% in comparable utilities.
**Implementation**: `DunningProcess` model with `status` enum mapping to tier. `advance_dunning()` method: validates that minimum hold period has elapsed, transitions to next tier, triggers `ntfy` notification template, logs action. `DunningConfig` stores per-tier hold days, notification template ID, and auto-escalation flag.
**Competitor**: Gentrack, Hansen Technologies, Oracle CC&B

## 8. Revenue Leakage Detection with Rule Engine
**Category**: Revenue Assurance
**Justification**: Sub-Saharan utilities lose 15–25% of potential revenue to technical/commercial losses, unbilled connections, and billing errors. A rule-engine layer scanning bills, payments, and meter data automatically surfaces leakage patterns in near-real-time vs. monthly audit cycles.
**Implementation**: `RevenueLeakageScanner` async method: runs configurable rule set — (1) bills with zero consumption for >60 days; (2) meter IDs not billed in >45 days; (3) payments without matching bills; (4) tariff applied does not match customer class; (5) demand charges below minimum contract demand; (6) credits exceeding 20% of bill amount. Each hit creates a `RevenueAssuranceFlag` with ML confidence score if Ollama available.
**Competitor**: Itron Revenue Guard, ABB Ability Network Manager, Netcracker

## 9. Regulatory Tariff Compliance Checking (EPRA/KPLC)
**Category**: Compliance
**Justification**: EPRA tariff orders mandate specific rate caps, levy calculations (REP, REREC, ERC, VAT), and customer class segmentation. Non-compliance at bill generation exposes the utility to regulatory fines and consumer protection claims. Automated compliance checking at the point of bill generation prevents this.
**Implementation**: `validate_tariff_compliance()` method: load tariff rule set from `comp` capability or embedded EPRA rate schedule; assert that each rate block rate is within EPRA approved range; assert that all required levies are present; assert that customer class mapping is valid; return compliance verdict with per-rule findings. Called in `generate_bill()` pre-condition.
**Competitor**: Enverus Regulatory Intelligence, Oracle Utilities Policy, C2 Global Technologies

## 10. Smart Invoice with QR Payment and Digital Delivery
**Category**: Customer Experience
**Justification**: Paper bills cost KES 80–150 each to print and deliver in Kenya. E-invoicing with embedded M-Pesa paybill QR code and WhatsApp/email delivery reduces bill-to-cash cycle from 14 days to <2 hours and cuts DSO. Kenya's e-invoicing VAT rules (2023 KRA VAT Act amendment) also require structured invoice formats.
**Implementation**: `render_invoice()` async method: accepts `bill_id` and `output_format` (pdf | html | json | ubl_xml); generates structured invoice object; embeds M-Pesa QR code (Safaricom QR-Gen API or local qrcode lib); signs with utility digital certificate; dispatches via `ntfy` channel (email/WhatsApp/SMS). Return `InvoiceRendering` with delivery status.
**Competitor**: Billtrust, Fluentgrid, Oracle Utilities Customer Self-Service

## 11. Payment Plan (Instalment Agreement) with Covenant Monitoring
**Category**: Collections / Customer Service
**Justification**: Blanket write-offs destroy AR value. Structured payment plans with auto-monitoring reduce credit losses by 40–60% in comparable utility programs. Most utilities' arrears management is ad-hoc; a covenant-monitored plan with auto-default triggers is a material improvement.
**Implementation**: `PaymentPlan` model: `(account_id, total_arrears, instalment_amount, frequency, instalments_paid, status, next_due_date, covenant_miss_count, max_misses)`. `create_payment_plan()` creates the plan. `check_payment_plan_covenants()` scheduled task: compares expected vs. actual instalments; increments miss counter; after `max_misses` transitions to `defaulted` and triggers dunning re-escalation. Expose plan status in dashboard.
**Competitor**: SAP IS-U Collections, Gentrack Velocity, Hansen SBS

## 12. Carbon Emission Billing and Scope 2 Reporting
**Category**: ESG / Sustainability
**Justification**: Kenya's Carbon Markets Taskforce (2023) and SEC-aligned ESG reporting requirements increasingly require grid emission factors embedded in invoices. Industrial customers need Scope 2 market-based accounting data. Utilities that provide this data become preferred suppliers vs. those that do not.
**Implementation**: `CarbonEmissionRecord` model: `(bill_id, consumption_kwh, grid_emission_factor_kg_per_kwh, location_based_kg, market_based_kg, certificate_id)`. `calculate_carbon_charges()` method: multiply consumption by grid emission factor (loaded from `energy_grd` grid mix data or static table), produce emission certificate reference. If green tariff applies, look up renewable energy certificate (REC) to offset. Include `carbon_summary` block in rendered invoice.
**Competitor**: Veridium, WattTime, Oracle Utilities Sustainability

## 13. Automated Meter Data Validation (AMDV) Pipeline
**Category**: Data Quality / Revenue Protection
**Justification**: Raw AMI meter reads contain ~2–5% anomalies (stuck registers, rollover errors, communication gaps, tamper events). Billing from unvalidated reads causes over/under-billing, disputes, and revenue leakage. AMDV is a prerequisite for accurate ToU and NEM billing.
**Implementation**: `validate_meter_reads()` async method: (1) detect zero/null reads — flag as `missed_read`; (2) detect register rollover — compute modulo max register; (3) detect negative increments — flag as `reverse_tamper`; (4) detect statistical outliers (>3σ from 90-day rolling mean) — flag as `suspect_high`; (5) cross-validate against estimated reads where available. Return `MeterReadValidation` with per-read status and estimated substitution values.
**Competitor**: Itron AnalytIQ, Oracle MDM, Silver Spring Networks Riva

## 14. Cross-Subsidiary Billing Netting for Large Accounts
**Category**: Enterprise Billing
**Justification**: Multi-site industrial customers (factories, retail chains) want a single consolidated invoice with subsidiary-level detail and central payment settlement. Without netting, each site generates a separate bill, increasing AR complexity and customer payment friction.
**Implementation**: `AccountGroup` model: `(group_id, primary_account_id, subsidiary_account_ids, netting_enabled, consolidated_invoice)`. `generate_consolidated_bill()` method: aggregate consumption and charges across all subsidiary accounts; apply group-level volume discount tiers; produce master bill with attached sub-bills; route payment to primary account. Expose `create_account_group()` and `get_group_summary()` endpoints.
**Competitor**: SAP IS-U, Oracle Utilities, Gentrack G2.0

## 15. Real-Time Collection Dashboard with Predictive KPIs
**Category**: BI / Analytics
**Justification**: Static monthly reports arrive after the fact. Revenue-at-risk and collection likelihood scores computed daily (or intra-day for large portfolios) allow collections teams to prioritise outreach before accounts slip into arrears. Predictive KPIs (expected collection rate, 30/60/90-day arrears forecast) are now table stakes at Tier-1 utilities.
**Implementation**: `collection_forecast()` async method: for each open bill compute `days_to_due`, historical payment probability by payment method and customer class (loaded from past 24 months), expected payment date distribution. Aggregate to portfolio level: expected_collected_30d, expected_collected_60d, at_risk_amount. Return `CollectionForecast` with per-segment breakdown. If Ollama available, pass summary stats to local model for narrative insight generation.
**Competitor**: Bidgee Analytics, Itron Analytics, DataRobot for Utilities
