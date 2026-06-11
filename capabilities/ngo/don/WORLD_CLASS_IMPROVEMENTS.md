# World-Class Improvements — Donor Relationship Management (ngo_don)

Fifteen targeted improvements that elevate ngo_don from functional CRUD to a revenue-generating,
AI-augmented donor intelligence platform competitive with Salesforce NPSP, Blackbaud Raiser's Edge,
and Bloomerang.

---

### I1. AI-Powered Donor Propensity Scoring
**Category**: AI/ML
**Justification**: Fundraising teams spend 70% of cultivation time on the wrong donors. A real-time score (0–100) derived from giving frequency, recency, gift size trend, and engagement velocity lets relationship managers triage a 3,000-donor portfolio and focus effort on the 80 donors who drive 80% of revenue — achieving the same results with a smaller team.
**Implementation**: Compute a weighted score from four sub-signals (RFM: recency, frequency, monetary; plus stewardship engagement ratio) stored per donor on every pledge/receipt write; expose `score_donor()` and `bulk_rescore()` async methods so the API and nightly batch both use the same path.
**Competitive reference**: Blackbaud Raiser's Edge NXT "AI Donor Insights"; Salesforce Nonprofit Success Pack "Scoring Engine"

---

### I2. Lapsed-Donor Win-Back Campaign Engine
**Category**: Feature
**Justification**: Industry average donor lapse rate is 45–65% year-on-year; automated win-back sequences that trigger at 90/180/365 days of silence recover 18–25% of lapsed donors without incremental staff cost — equivalent to hiring a full-time major-gifts officer.
**Implementation**: `compute_lapse_risk()` compares last-receipt date against configurable thresholds and writes `lapse_risk` (none/low/medium/high/critical) to each donor; `get_winback_candidates()` returns segmented lists with recommended messaging cadence based on prior gift history.
**Competitive reference**: HubSpot CRM win-back workflows; DonorPerfect re-engagement sequences

---

### I3. Recurring-Gift Scheduling and Auto-Fulfillment Tracking
**Category**: Feature
**Justification**: Monthly giving programmes (sustainer donors) generate 4x higher lifetime value than one-time gifts. Without automated instalment scheduling, staff manually reconcile every bank statement line — an error-prone process that misattributes or loses 5–12% of recurring income in mid-size NGOs.
**Implementation**: `schedule_recurring_pledge()` generates a forward-dated instalment schedule (calendar entries) as child records under a parent pledge; `advance_instalment()` marks each instalment paid, auto-generates a receipt, and projects the next due date; overdue instalments surface in `overdue_pledges()`.
**Competitive reference**: Bloomerang recurring giving; Stripe Billing subscription invoicing

---

### I4. Soft-Credit and Household Attribution
**Category**: Feature
**Justification**: Grant-making foundations, advised funds, and family offices route gifts through intermediaries; without soft-credit tracking the true philanthropic relationship is invisible in the database, causing major donors to receive generic outreach instead of bespoke cultivation.
**Implementation**: `link_household()` groups donor records into a household/entity with a designated primary contact; `soft_credit()` assigns a percentage of a receipt's value to a linked advisor or household member, stored as `ngo_soft_credit` records; `donor_giving_history()` includes both hard and soft totals.
**Competitive reference**: Raiser's Edge NXT Households; CharityEngine relationship mapping

---

### I5. Tax-Deductibility Compliance Engine
**Category**: Compliance
**Justification**: Kenya Revenue Authority (and OECD peers) require NGOs to issue annual tax certificates within 30 days of the fiscal year end; non-compliance risks donor tax-deductibility claims and triggers KRA audit exposure. Automating this converts a 2-week manual process into a same-day batch run.
**Implementation**: `generate_annual_tax_certificate()` aggregates all receipts for a donor in a given fiscal year, applies jurisdiction-specific deductibility rules (KE: NGO Act S.10, US: IRS 501(c)(3)), and produces a structured certificate record; `bulk_issue_tax_certificates()` runs concurrently via `asyncio.gather`.
**Competitive reference**: Salesforce Nonprofit cloud Tax Receipt automation; Andar/360 Canadian T4A integration

---

### I6. Donor Portal Self-Service Token Generation
**Category**: UX / Security
**Justification**: Donors who can self-serve — view giving history, download receipts, update address — reduce inbound support queries by 40% (sector benchmark) and feel more connected to the organisation, driving higher renewal intent.
**Implementation**: `generate_portal_token()` issues a signed, time-limited (24 h) token scoped to a single donor ID; `validate_portal_token()` verifies the token and returns the read-only donor view; tokens are stored as `ngo_portal_token` records with usage audit.
**Competitive reference**: Blackbaud Online Express donor self-service; Kindful donor portal

---

### I7. Duplicate-Donor Detection and Merge
**Category**: Data Quality / UX
**Justification**: CRM databases accumulate 10–30% duplicate records through multi-channel data entry; duplicates fragment giving history, distort retention metrics, and cause major donors to receive duplicate solicitations — a relationship-damaging error.
**Implementation**: `find_duplicate_candidates()` scores pairs using a weighted similarity function (name token overlap + email exact + phone fuzzy); `merge_donors()` consolidates two records, re-parents all pledges/receipts/communications to the survivor, and soft-deletes the duplicate with a provenance pointer.
**Competitive reference**: DonorPerfect Duplicate Check; Salesforce NPSP Merge function

---

### I8. Multi-Currency Pledge Reporting with FX Normalization
**Category**: Compliance / Finance
**Justification**: International NGOs receive grants in USD, EUR, GBP, and KES; without FX normalization the portfolio summary mixes currencies into a meaningless total. Donor segmentation and board reporting require a single functional currency view.
**Implementation**: `set_fx_rate()` stores exchange rates (date-stamped, per-currency-pair) as `ngo_fx_rate` records; `portfolio_summary()` and `donor_giving_history()` gain a `reporting_currency` parameter that converts all Decimal amounts using the nearest-date rate.
**Competitive reference**: Blackbaud Financial Edge NXT multi-currency; NetSuite OneWorld FX management

---

### I9. Stewardship Touchpoint Compliance Dashboard
**Category**: Feature / UX
**Justification**: Principal and legacy donors (top 2% of portfolio) expect a minimum contact frequency; missing a scheduled touchpoint is a leading predictor of lapse for gifts over USD 25,000. A real-time compliance view surfaces at-risk relationships before the window closes.
**Implementation**: `stewardship_compliance_report()` calculates touchpoint completion rate per plan (completed / required × 100), identifies plans behind schedule, and returns a ranked list ordered by risk tier; `touchpoints_due_this_month()` returns plans needing contact in the current calendar month.
**Competitive reference**: Salesforce Nonprofit cadence plans; Virtuous CRM stewardship scoring

---

### I10. Donation Impact Reporting Linkage
**Category**: Feature / Integration
**Justification**: 76% of major donors say "seeing impact" is the primary driver of repeat giving (AFP 2024 study). Linking a donation receipt to a specific programme outcome and surfacing that in donor communications converts a transactional relationship into a transformational one.
**Implementation**: `link_donation_to_impact()` stores a `ngo_impact_link` record associating a receipt with an external `programme_id` and `impact_metric`; `donor_impact_statement()` aggregates all linked outcomes for a donor across all time, suitable for embedding in a personalised annual report.
**Competitive reference**: Apricot by Bonterra outcomes tracking; Salesforce Program Management Module

---

### I11. GDPR / Data-Privacy Consent Management
**Category**: Compliance / Security
**Justification**: GDPR Art. 7 and Kenya's Data Protection Act 2019 require documented, specific, revocable consent for marketing communications; a single data-subject access request without a consent audit trail can result in KSh 5M (approx USD 38k) fines or KRA deregistration.
**Implementation**: `record_consent()` stores consent events (granted/withdrawn) per channel as immutable `ngo_consent_event` records; `get_current_consent()` returns the effective consent state per channel by replaying the event log; `log_communication()` guards against sending on a non-consented channel.
**Competitive reference**: HubSpot GDPR consent tools; Salesforce Privacy Center consent management

---

### I12. Automated Receipt Delivery via Notification Hub
**Category**: Integration / UX
**Justification**: Printed/manually-emailed receipts take 3–10 days to reach donors; same-day digital delivery increases donor satisfaction (NPS +18 points, Bloomerang 2023) and reduces inbound "where is my receipt" queries by 80%.
**Implementation**: `queue_receipt_delivery()` writes a `ngo_receipt_delivery_job` record with channel (email/whatsapp), donor contact details, and receipt payload; delivery status is tracked on the receipt record; integrates with APG `ngo_msg` capability event bus if available.
**Competitive reference**: Bloomerang instant receipt; Donorbox automated receipt emails

---

### I13. Pledge Reminder Escalation Workflow
**Category**: Feature / Automation
**Justification**: 22% of open pledges in NGO databases are 30+ days overdue due to missed follow-up; a structured escalation path (automated reminder → staff follow-up → relationship manager escalation) reduces overdue pledge balance by 35% without additional headcount.
**Implementation**: `get_pledge_reminder_schedule()` computes the next reminder date and escalation owner based on days overdue (7 d: automated, 30 d: assigned staff, 60 d: relationship manager); `record_pledge_reminder_sent()` logs each reminder as a linked communication record.
**Competitive reference**: Salesforce Flow pledge reminder automation; DonorPerfect pledge reminders

---

### I14. Donor Lifecycle Stage Classification
**Category**: AI/ML / Feature
**Justification**: Treating a first-time $500 donor identically to a 10-year $1M cumulative donor destroys relationship capital; lifecycle-aware messaging (prospect → first-time → repeat → major → legacy) doubles average gift size at upgrade moments (AFP benchmark data).
**Implementation**: `classify_lifecycle_stage()` derives stage from cumulative giving, giving streak, and years-active using a rule-based classifier; `get_upgrade_candidates()` returns donors within 20% of the Decimal threshold for the next stage with a recommended ask amount; stage is written back on each receipt event.
**Competitive reference**: Blackbaud donor lifecycle management; Virtuous responsive fundraising lifecycle

---

### I15. Board-Ready Giving Trend Export
**Category**: Feature / Compliance
**Justification**: NGO boards require quarterly giving trend data in formats consumable by non-technical trustees; manual Excel exports take 4–6 staff hours per quarter and introduce transcription errors that undermine donor confidence in financial stewardship.
**Implementation**: `generate_trend_report()` aggregates monthly receipt totals by donor type over a configurable date range; returns a structured dict consumable by the APG `fin_rpt` reporting capability or serialisable directly to CSV/JSON; includes YoY and MoM percentage change for each segment.
**Competitive reference**: Salesforce Nonprofit Analytics; Blackbaud Financial Edge NXT board dashboards
