# Policy Administration (ins_pol) — World-Class Improvements

## Overview

15 improvements that elevate `ins_pol` from a competent lifecycle manager to a
platform competitors cannot easily replicate.

---

### I1. AI-Driven Premium Re-Rating on Endorsement

**Category**: AI/ML
**Justification**: Mid-term endorsements today use a flat additive delta; that
leaves money on the table and misprices risk. Real-time re-rating on every
change—driven by an embedded rating engine fed current telematics or
underwriting factors—closes the gap that carriers like Tractable and Cape
Analytics exploit.
**Implementation**: Accept an optional `rating_factors` dict on
`create_endorsement`; compute the revised premium via a pluggable
`RatingEngine.rate()` coroutine and store both the old premium and the
AI-derived adjustment, surfacing the delta to the caller.
**Competitive reference**: Majesco Rating & Underwriting (AI re-rate on every
mid-term change)

---

### I2. Proactive Renewal Scoring & Churn Prediction

**Category**: AI/ML
**Justification**: Insurers lose 15-25 % of the book at renewal because no one
acts early enough. A churn-probability score surfaced 90 days out lets agents
intervene with tailored retention offers before the client shops around.
**Implementation**: `score_renewal_risk()` computes a heuristic score from
lapse history, endorsement frequency, days-to-expiry, and premium delta,
returning a `[0,1]` float with a `risk_band` label (`low/medium/high`).
**Competitive reference**: Guidewire Predict (renewal propensity scoring)

---

### I3. Regulatory Compliance Guardrails (IRA Kenya / IAIS)

**Category**: Compliance
**Justification**: Underwriters in regulated markets face fines for policies
that breach minimum-sum-insured floors, exceed premium-to-sum-insured caps, or
lack mandatory clauses. Baking the rules in-service catches violations at
issuance rather than audit time.
**Implementation**: `validate_regulatory_compliance()` checks product-specific
floors/ceilings (configurable per jurisdiction via a `ComplianceRuleset`) and
raises `ComplianceViolationError` with structured violation details before any
record is persisted.
**Competitive reference**: Duck Creek Policy (built-in compliance guardrails for
IRA, FCA, NAIC)

---

### I4. Instalment Schedule & Premium Financing Tracking

**Category**: Feature
**Justification**: Over 60 % of retail motor policies in East Africa are sold on
monthly instalments; managing instalment schedules inside `ins_pol` eliminates
a common integration gap that leads to incorrect lapse triggers.
**Implementation**: `create_instalment_schedule()` generates dated instalment
records linked to a policy; `record_instalment_payment()` marks each paid,
recomputes the outstanding balance, and automatically un-lapses the policy when
arrears clear.
**Competitive reference**: Majesco PolicyCore (instalment schedule management)

---

### I5. Co-Insurance & Treaty Reinsurance Split Tracking

**Category**: Feature
**Justification**: Large commercial risks are co-insured across multiple
carriers; tracking each lead/follower share at the policy level is mandatory for
IRA statutory returns and reinsurance recoveries.
**Implementation**: `set_coinsurance_structure()` attaches a list of
`{carrier_id, share_pct, premium_share, sum_insured_share}` records; `get_coinsurance_structure()` returns the split; shares are validated to sum to 100 %.
**Competitive reference**: Xuber (coinsurance split ledger)

---

### I6. Digital Policy Wallet (QR / PDF / Passkit)

**Category**: UX
**Justification**: Regulators now mandate digital proof-of-insurance; generating
a signed, verifiable PDF with an embedded QR code linking to a live policy
check endpoint transforms customer experience and reduces fraud.
**Implementation**: `generate_digital_certificate()` builds a structured
`CertificatePayload`, signs it with an HMAC key, and returns a `download_url`
plus a `verify_url`; the verify path authenticates the signature without a
database round-trip.
**Competitive reference**: Verisk FAST (digital certificate generation with
QR verification)

---

### I7. No-Claim Bonus (NCB) Ledger

**Category**: Feature
**Justification**: Motor insurers apply NCB discounts of up to 50 % for
claim-free years; without an in-policy ledger the discount is applied
inconsistently and the audit trail is lost.
**Implementation**: `update_ncb()` increments or resets the `ncb_years` counter
and recomputes the `ncb_discount_pct` using a configurable step table; the
value flows into `initiate_renewal()` as an automatic premium reducer.
**Competitive reference**: Insurity PolicyPlus (NCB schedule management)

---

### I8. Lien & Financier Notification Workflow

**Category**: Integration
**Justification**: Vehicle and property policies financed by banks require
automatic notification to the lien-holder on cancellation or lapses; missing
this triggers contractual breaches.
**Implementation**: `register_lien()` stores `{financier_id, financier_name,
loan_reference, notification_email}`; `_notify_lienholders()` is called
internally on every `cancel_policy()` and `lapse_policy()`, emitting a
`lienholder_notified` audit event.
**Competitive reference**: Applied Systems Epic (lien-holder notification on
status change)

---

### I9. Policy Comparison & Version Diffing

**Category**: Feature
**Justification**: Underwriters and auditors need to see exactly what changed
between any two versions of a policy (original vs. post-endorsement); without
versioning, this is reconstructed manually from audit logs.
**Implementation**: `snapshot_policy()` persists a deep-copy version record
with a monotonically increasing `version` number; `diff_policy_versions()` returns
a structured `{field: {from, to}}` diff between any two snapshots.
**Competitive reference**: ContractPodAi (contract versioning and diff)

---

### I10. Automated Expiry & Grace-Period Engine

**Category**: Automation
**Justification**: Most policy systems require a nightly batch job to expire
policies; an in-service grace-period engine that automatically transitions
`active → grace → lapsed` removes the cron dependency and surfaces the correct
status in real time.
**Implementation**: `get_effective_status()` computes the live status considering
a configurable `grace_period_days` per product without mutating the record;
`run_expiry_engine()` batch-transitions only policies whose grace period has
also elapsed.
**Competitive reference**: Guidewire PolicyCenter (grace period management)

---

### I11. Multi-Currency & FX Rate Support

**Category**: Feature
**Justification**: Reinsurance treaties and cross-border group policies are
priced in USD or EUR while premiums are collected in KES; storing only one
currency creates reconciliation nightmares at close of accounts.
**Implementation**: `convert_policy_currency()` accepts a target currency and
an `fx_rate` (Decimal), recomputes `premium`, `sum_insured`, and any open
instalment balances, and records the conversion in the audit trail with the
applied rate.
**Competitive reference**: Majesco PolicyCore (multi-currency ledger)

---

### I12. STP (Straight-Through Processing) Eligibility Scoring

**Category**: AI/ML
**Justification**: Underwriters spend 70 % of time on risks that could be
auto-approved; an STP score gates simple risks for instant issuance while
routing complex ones to manual review queues.
**Implementation**: `stp_score()` evaluates sum-insured thresholds, product
type, insured credit tier, and prior claims count, returning a `{score, eligible,
referral_reasons}` dict; `create_policy()` checks this when `auto_bind=True` is
passed.
**Competitive reference**: EXL STP Underwriting (automated bind eligibility)

---

### I13. Claims-Loss Ratio Linkage

**Category**: Integration
**Justification**: Portfolio managers need loss ratios per policy at a glance;
linking claim reserves and payments from `ins_clm` into the policy record
enables real-time combined ratio computation without a warehouse query.
**Implementation**: `attach_claim_summary()` stores `{claim_count, total_incurred,
total_paid}` against the policy; `compute_loss_ratio()` returns
`total_incurred / premium` as a Decimal, flagging ratios above a configurable
threshold.
**Competitive reference**: Gallagher Bassett (claim-loss ratio dashboard per
policy)

---

### I14. Beneficiary & Succession Management

**Category**: Feature
**Justification**: Life and health policies require a managed beneficiary
register with percentage splits and succession ordering; without it, insurers
cannot automate claims settlement or comply with estate-law requirements.
**Implementation**: `set_beneficiaries()` replaces the beneficiary list,
validates that `sum_pct` totals 100 %, and creates an endorsement record;
`get_beneficiaries()` returns the current ordered list with effective date.
**Competitive reference**: Insurity Life (beneficiary register management)

---

### I15. Embedded Telematics / IoT Data Hooks

**Category**: Integration
**Justification**: Pay-how-you-drive motor and usage-based health products
require real-time telematics feeds to adjust premiums monthly; embedding the
hook in `ins_pol` rather than a separate microservice keeps the premium ledger
authoritative.
**Implementation**: `ingest_telematics_snapshot()` accepts a `{period,
odometer_km, risk_events, score}` payload, recomputes the period premium using a
configurable `TelematisBand` table, and records an automatic premium-adjustment
endorsement if the change exceeds a materiality threshold.
**Competitive reference**: LexisNexis Telematics (usage-based insurance premium
adjustment)
