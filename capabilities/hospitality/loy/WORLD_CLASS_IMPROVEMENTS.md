# Guest Loyalty Programme — World-Class Improvements

Fifteen targeted enhancements to elevate `hos_loy` from a basic points ledger to a full competitive-grade loyalty engine.

---

### I1. Decimal-Safe Monetary Arithmetic
**Category**: Compliance
**Justification**: Float arithmetic silently loses precision on currency values, producing incorrect point valuations and audit discrepancies that compound across millions of transactions. Regulatory bodies in EAC/EU require exact decimal accounting for financial ledgers.
**Implementation**: Replace all `float` monetary fields with `Decimal`; use `ROUND_HALF_UP` for point-to-cash conversions; store `cash_value` and `point_value_kes` as `Decimal` with explicit quantisation to 2 decimal places.
**Competitive reference**: Marriott Bonvoy, IHG One Rewards — both use exact decimal ledgers for financial integrity.

---

### I2. FIFO Points Expiry Ledger with Advance Notice
**Category**: Compliance
**Justification**: Un-governed expiry is a balance-sheet liability and a class-action risk. FIFO expiry with 60/30-day warning events reduces member attrition by ~18% while controlling outstanding liability.
**Implementation**: Track point blocks with `expires_at` timestamps; `sweep_expired_points` processes FIFO expiry and emits `points_expiry_notice` events at 60 and 30 days; `get_expiring_points_report` surfaces members near expiry for CRM dispatch.
**Competitive reference**: Marriott Bonvoy (24-month rolling expiry with activity reset), IHG One Rewards.

---

### I3. Tier Downgrade Protection with Grace Period
**Category**: Feature
**Justification**: Abrupt downgrades cause 40% of affected members to disengage. A configurable grace period (typically 90 days post-qualifying year) maintains tier benefits while giving members a retention window.
**Implementation**: `evaluate_tier_retention` computes qualifying metrics at anniversary; if thresholds unmet, sets `tier_grace_until` rather than immediately downgrading; emits `tier_downgrade_warning` for CRM actioning.
**Competitive reference**: Hilton Honors tier extension, World of Hyatt 90-day grace period.

---

### I4. Real-Time Spend Projection to Next Tier
**Category**: UX
**Justification**: Showing members exactly how many more KES/nights they need to reach the next tier is the single highest-impact UX feature for driving incremental spend (Phocuswire 2024 loyalty benchmark).
**Implementation**: `tier_progress_for_member` returns `{"current_tier", "next_tier", "points_needed", "nights_needed", "spend_needed_kes"}` computed live from member state and tenant tier rules using `Decimal` arithmetic.
**Competitive reference**: Accor Live Limitless — real-time tier progress bar in mobile app.

---

### I5. Points Transfer and Account Pooling
**Category**: Feature
**Justification**: Corporate travel managers demand account pooling to accelerate redemption; Marriott reports 22% higher LTV from pooled accounts. Family pooling reduces churn among high-value leisure segments.
**Implementation**: `transfer_points(from_member_id, to_member_id, points, fee_pct)` atomically deducts/credits both accounts with full ledger entries; `create_pool` and `add_member_to_pool` govern pool eligibility and spending limits.
**Competitive reference**: Hilton Honors Points Pooling, Marriott Bonvoy Points Transfer.

---

### I6. Digital Wallet Token (QR-Scannable Loyalty Card)
**Category**: UX
**Justification**: 67% of guests expect a mobile-presentable loyalty card at check-in (Oracle Hospitality 2024). A signed wallet token eliminates manual lookup and reduces check-in time by ~45 seconds.
**Implementation**: `generate_wallet_token(member_id)` returns a signed HMAC-SHA256 JWT containing membership number, tier, and current balance, valid 24 hours; front-desk validates via `verify_wallet_token(token)`.
**Competitive reference**: Marriott Bonvoy Apple Wallet / Google Wallet, Accor ALL digital card.

---

### I7. Stay-Based Milestone Awards
**Category**: Feature
**Justification**: Milestone rewards (5th stay certificate, 25-night anniversary gift) drive booking cadence and create emotional brand moments; properties using milestone programmes report 12% higher repeat-booking rates (STR 2023).
**Implementation**: `check_and_award_milestones` evaluates a configurable milestone table after each stay earn; issues bonus-point grants and fires `milestone_achieved` events; milestones are idempotent (awarded exactly once per threshold crossing).
**Competitive reference**: Hyatt Globalist milestone awards, IHG One Rewards milestone certificates.

---

### I8. Velocity-Based Fraud Detection
**Category**: Security
**Justification**: Loyalty fraud (account takeover, fictitious stay claims, duplicate earn) costs the global industry USD 1B+ annually (LoyaltyOne 2023). Rule-based velocity checks catch 80% of patterns at near-zero false-positive cost.
**Implementation**: `assess_transaction_risk(member_id, txn_type, points, reference_id)` checks same-day earn frequency, duplicate `reference_id`, and earn/redeem velocity vs. historical baseline; returns `RiskVerdict` and emits `fraud_flag` audit event on HIGH verdict.
**Competitive reference**: Marriott Bonvoy fraud operations, Collinson Group loyalty fraud platform.

---

### I9. AI-Driven Churn Risk Score
**Category**: AI/ML
**Justification**: Identifying at-risk members before they lapse enables proactive win-back offers with 3–5x better ROI than post-lapse reactivation; Deloitte estimates 5% churn reduction = 25% profit increase for loyalty programmes.
**Implementation**: `compute_churn_risk(member_id)` scores on RFM dimensions (days-since-last-transaction, transaction frequency, monetary trend); returns `{"risk_score", "risk_tier": "low|medium|high", "recommended_action"}` without external ML dependency.
**Competitive reference**: Amadeus Loyalty Analytics, Duetto loyalty revenue intelligence.

---

### I10. Configurable Challenge Engine
**Category**: Feature
**Justification**: Targeted challenges ("Stay 3 nights in 30 days, earn 2 000 bonus points") lift occupancy by 8–12% in A/B tests by creating short-term incentive loops that static earn rates cannot replicate.
**Implementation**: `create_challenge`, `opt_in_challenge`, `evaluate_challenge_progress`, `complete_challenge` — progress is tracked per-member per-challenge; completion triggers automatic bonus award and `challenge_completed` event.
**Competitive reference**: Marriott Bonvoy Challenges & Promotions portal.

---

### I11. Partner Points Conversion (Airline Mile Transfer)
**Category**: Integration
**Justification**: Airline mile conversion is the #1 redemption request in African hotel loyalty programmes (IATA Africa 2024); enabling configurable ratio conversion removes the single largest redemption barrier.
**Implementation**: `convert_to_partner_currency(member_id, partner_id, points, target_account_ref)` deducts hotel points at the configured `conversion_ratio`, creates a `PartnerConversion` ledger record, and returns a payload formatted for the partner's API contract.
**Competitive reference**: Marriott Bonvoy → airline transfer, Hilton Honors → airline mile conversion.

---

### I12. Birthday and Anniversary Lifecycle Rewards
**Category**: UX
**Justification**: Birthday/anniversary bonus points generate 4.2x higher email open rates and 2.8x redemption events versus standard promotions (Epsilon 2023); emotional touchpoints build brand affinity disproportionate to their cost.
**Implementation**: `check_lifecycle_rewards(member_id, event_date)` matches against stored `date_of_birth` and `enrollment_anniversary`; auto-issues configurable bonus points with `transaction_type="lifecycle_reward"` and idempotency guard (one award per calendar year per type).
**Competitive reference**: IHG One Rewards birthday bonus, Accor ALL anniversary offers.

---

### I13. GDPR/PDPA Consent Capture and PII Erasure
**Category**: Compliance / Security
**Justification**: Kenya Data Protection Act 2019 and GDPR impose right-to-erasure with fines up to 4% of global turnover; auditable consent trails and anonymisation pipelines are non-negotiable for any guest data platform.
**Implementation**: `record_consent(member_id, consent_type, granted, ip_address)` persists an immutable consent record; `request_data_erasure(member_id)` anonymises PII fields (name/email/phone → SHA-256 tokens) while retaining anonymised financial ledger for accounting compliance.
**Competitive reference**: OneTrust hospitality consent management, Salesforce GDPR compliance for loyalty.

---

### I14. Multi-Currency Earn Normalisation
**Category**: Feature / Compliance
**Justification**: East African properties transact in KES, UGX, TZS, USD; without normalisation a USD spend earns ~100x more points than a KES-equivalent, creating inequity and gaming vectors.
**Implementation**: `earn_points_multi_currency(member_id, amount, currency_code, ...)` normalises to KES using a `fx_rates` dict stored per tenant; `set_fx_rate(currency_code, rate)` lets operators update rates; all arithmetic uses `Decimal`.
**Competitive reference**: Marriott Bonvoy normalises earn to USD base across 130+ currencies.

---

### I15. Configurable Redemption Blackout Rules
**Category**: Feature / Compliance
**Justification**: Revenue-critical dates (Christmas, NYE, peak conference season) must be protected from points redemptions that displace full-rate guests; without blackout rules, loyalty cannibalises RevPAR.
**Implementation**: `create_blackout_rule(date_from, date_to, affected_offer_types, property_ids)` stores date ranges per property; `redeem_points` / `redeem_offer` validate against active blackouts before processing; returns `blackout_violation` error with next eligible date.
**Competitive reference**: Four Seasons redemption blackout calendar enforced at booking engine and POS integration layers.
