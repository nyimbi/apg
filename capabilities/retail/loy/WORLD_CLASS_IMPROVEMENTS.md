# Loyalty Programme — World-Class Improvements

**Capability**: `retail_loy` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Gamified Streak & Challenge Engine

**Problem**: Flat earn rates produce diminishing engagement. Members earn passively and have no behavioural hook.

**Improvement**: Introduce a `ChallengeEngine` — time-boxed micro-challenges (e.g. "shop 3 Fridays in a row", "spend $200 in electronics this month"). Each challenge has a `streak_target`, `bonus_points`, and a `reset_policy`. Completing a streak unlocks a badge and a one-time multiplier burst. Challenges are authored by marketing staff and targeted by CLV segment + tier.

**Impact**: 15–30% lift in purchase frequency among mid-tier members (industry benchmark from Starbucks Rewards, Woolworths Everyday Rewards).

---

## 2. Dynamic Points Pricing (Yield Management)

**Problem**: Fixed `POINTS_CASH_RATE = 0.01` ignores supply/demand, reward inventory, and programme liability.

**Improvement**: Replace the static rate with a `PointsPricingEngine` that adjusts the redemption rate in real-time based on: outstanding liability (points float), redemption demand pressure, reward inventory levels, and time-to-expiry proximity. Rates are bounded `[min_rate, max_rate]` per programme config and published via an event to downstream pricing systems.

**Impact**: Reduces points liability by up to 20% while increasing perceived reward value at low-demand periods (yield management, analogous to airline miles pricing).

---

## 3. Fraud Detection — Velocity & Graph Anomaly Rules

**Problem**: `ml_detect_loyalty_fraud` is a stub that returns `{"ml_enhanced": False}` when Ollama is absent. The earn pipeline has no inline fraud gates.

**Improvement**: Add deterministic velocity checks inline in `earn_points` and `redeem_points`: max earn per member per day, earn-then-immediately-redeem latency gate (min 24h), device fingerprint cross-member sharing detection. Pair with an async graph anomaly scan that flags synthetic member rings (multiple members sharing `external_customer_id`, email, or mobile). Results feed a `FraudScore` model persisted per transaction.

**Impact**: Industry studies show rule-based velocity gates alone block 60–80% of loyalty fraud at near-zero false-positive cost.

---

## 4. Referral & Social Earn

**Problem**: Earn is entirely transactional (POS/online). No viral growth mechanism.

**Improvement**: Add `referral_earn` — a member generates a unique referral code. When a referee enrols and completes a qualifying purchase, both referrer and referee earn configurable bonus points. Track referral tree depth (max 2 levels to prevent pyramid exploitation). Expose referral leaderboards as a gamification surface.

**Impact**: Referral programmes deliver 3–5× lower CAC than paid acquisition channels in retail loyalty (Yotpo benchmarks).

---

## 5. Omnichannel Earn Attribution & Deduplication

**Problem**: `earn_points` trusts the caller's `transaction_id` without idempotency enforcement. Duplicate POS submissions double-credit members.

**Improvement**: Maintain a `receipt_hash → transaction_id` deduplication index (Redis/Postgres `UNIQUE` constraint). Any duplicate earn attempt returns the original transaction with `duplicate=True` rather than creating a second credit. Extend to cover online order IDs, mobile wallet tokens, and e-receipt QR codes.

**Impact**: Prevents the most common operator error in multi-channel deployments; eliminates the liability from double-earning exploits.

---

## 6. Coalition Real-Time Point Conversion API

**Problem**: `coalition_transfer` is one-directional (outbound), uses a hard-coded 5% fee, and has no partner exchange rate negotiation.

**Improvement**: Add a `CoalitionExchangeEngine` supporting bidirectional transfer: inbound points from partner programmes (earn at partner, redeem at home programme). Support per-partner exchange rates (stored in `LoyPartnerResponse`), settlement batching (daily/weekly), and a reconciliation ledger (`coalition_reconciliation` table). Expose a webhook endpoint partners call to push inbound transfer confirmations.

**Impact**: Enables true coalition loyalty (analogous to Star Alliance miles or Nectar/Avios interchange) — dramatically expands the effective earn and burn universe for members.

---

## 7. Predictive Churn Intervention

**Problem**: `loyalty_analytics` counts churn-risk members but takes no action. Intervention is manual.

**Improvement**: Add a `ChurnInterventionService` that runs on a scheduler: for members with recency > `churn_threshold_days` (configurable per programme), automatically triggers a personalised win-back offer (bonus points, discount voucher) via the `ntfy` adapter. Track offer acceptance rate and update CLV segment on re-engagement. Uses RFM scores already computed in `LoyClvSegmentRecord`.

**Impact**: Automated win-back campaigns recover 15–25% of at-risk members before they lapse (Antavo Global Customer Loyalty Report 2024).

---

## 8. Tier Downgrade Grace Period Enforcement

**Problem**: `tier_upgrade_check` upgrades members correctly, but the service lacks a matching `tier_downgrade_check` that enforces the `downgrade_grace_days` window defined in `LoyTierCreate`.

**Improvement**: Add `tier_downgrade_check` that evaluates whether a member's rolling-window qualification points (last `qualification_window_days` days of earn transactions) fall below the current tier's `qualification_points`. If so, set a `downgrade_scheduled_at` timestamp. Only execute the downgrade once `downgrade_grace_days` have elapsed without recovery. Log the change and fire a `tier_downgraded` notification.

**Impact**: Closes a material business rule gap. Current code allows members to stay at Gold indefinitely with zero activity — a significant programme liability.

---

## 9. Points Float & Liability Reporting

**Problem**: Outstanding points are a deferred revenue liability on the P&L. There is no actuarial breakage model.

**Improvement**: Add `points_liability_report` that computes: total outstanding points × `points_to_currency_rate`, breakage estimate (expected percentage of points that will expire unredeemed, derived from historical expiry runs), net liability = outstanding − breakage. Segment by tier and CLV segment. Export as JSON or CSV for finance systems. Include a Monte Carlo scenario (3 redemption rate assumptions).

**Impact**: Enables CFO-grade loyalty programme accounting, mandatory for programmes exceeding ~$1M liability in most jurisdictions.

---

## 10. Batch Earn via Event Stream (Bytewax / Kafka)

**Problem**: `earn_points` is synchronous and single-member. High-volume batch earn (e.g. end-of-day POS reconciliation, online order fulfilment runs) is not supported.

**Improvement**: Add `batch_earn_points` that accepts a list of `EarnRecord` objects, validates each (idempotency check, member lookup, multiplier), and processes them as an async generator — yielding results as they complete. Emit a `batch_earn_completed` event to the Bytewax stream with aggregate stats. Include a partial-failure model: failed records are collected in an `errors` list; successes are committed individually (no all-or-nothing semantics to prevent large rollbacks).

**Impact**: Required for any programme processing >1,000 transactions/day. Current synchronous design cannot handle POS batch uploads.

---

## 11. Member Merge & Duplicate Detection

**Problem**: Members enrol multiple times across channels (in-store, app, web), creating duplicate accounts with split point balances.

**Improvement**: Add `merge_members` — given a primary and secondary `member_id`, validate both belong to the same tenant, transfer the secondary's `points_balance` and `lifetime_points_earned` to primary, reclassify all secondary's transactions under the primary, mark secondary as `merged`, and log an immutable audit entry. Add a `find_duplicate_candidates` method using fuzzy match on `(email, mobile, first_name, last_name)`.

**Impact**: Member deduplication is consistently rated the #1 data quality issue in loyalty programme operators (Loyalty360 survey).

---

## 12. Tiered Reward Gating

**Problem**: `list_rewards` returns all `available` rewards regardless of member tier. A platinum-only experience is indistinguishable from a bronze member's catalogue.

**Improvement**: Extend `LoyRewardCreate` with `min_tier_name: str | None` and `allowed_segments: list[str] | None`. Add `list_rewards_for_member` that filters the catalogue by the member's current tier and CLV segment. Attempting to redeem a gated reward raises `IneligibleRewardError` with the qualifying tier name in the message.

**Impact**: Tier-exclusive rewards are the primary perceived differentiator of premium loyalty tiers — critical for programme health at the top end.

---

## 13. Campaign ROI Measurement

**Problem**: `LoyCampaignResponse.points_issued_to_date` tracks cost but not revenue impact. There is no way to measure whether a campaign generated incremental revenue.

**Improvement**: Add `record_campaign_attribution` — when a member earns points on a transaction while a campaign is active, link the transaction to the campaign. Compute: incremental revenue (transactions × avg order value attributed to campaign), cost (bonus points × `points_to_currency_rate`), ROI = (incremental revenue − cost) / cost. Expose `get_campaign_roi` returning a structured `CampaignRoiReport`.

**Impact**: Campaign ROI measurement is the prerequisite for marketing budget allocation and programme optimisation — without it, campaigns are run on intuition.

---

## 14. Privacy & Consent Lifecycle Management

**Problem**: `consent_recorded` is a boolean with no timestamp, version, or withdrawal mechanism. Non-compliant with GDPR Art. 7 / Kenya DPA 2019.

**Improvement**: Replace `consent_recorded: bool` with a `ConsentRecord` model: `consent_version`, `consent_at`, `consent_channel`, `consent_ip`, `withdrawn_at | None`. Add `withdraw_consent` method that sets `withdrawn_at`, freezes the member, and schedules a data deletion task. Add `export_member_data` (GDPR Data Subject Access Request) returning all data held for a member as structured JSON.

**Impact**: Required for compliance in any jurisdiction with a modern data protection law. Eliminates legal liability from a boolean flag that cannot prove meaningful consent.

---

## 15. Real-Time Personalisation via Contextual Bandits

**Problem**: `personalised_offer` uses static if/elif rules mapping `clv_segment` → offer parameters. It never learns which offer type actually drives conversion.

**Improvement**: Replace the rule table with a contextual bandit model (epsilon-greedy or UCB1) backed by a lightweight `OfferConversionLedger` (offer sent → offer redeemed boolean). The bandit selects the offer arm with the highest estimated reward given the member's context vector `[tier, clv_segment, recency_days, balance_ratio]`. On each redemption event, update the arm's reward estimate. Fall back to the existing rule table when fewer than `min_trials` (default 100) have been recorded per arm.

**Impact**: Contextual bandits outperform static segmentation rules by 20–40% in offer conversion rate (Facebook/Meta ads research, adaptable to loyalty). Requires no external ML infrastructure — pure Python running in-process.
