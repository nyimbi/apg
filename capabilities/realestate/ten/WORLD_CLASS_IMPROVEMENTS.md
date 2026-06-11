# Tenant Management — World Class Improvements

**Capability**: `realestate_ten` | **Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. Deposit Protection Lifecycle Management

Current state: deposit_registration is a single onboarding step flag with no financial tracking.

Improvement: Full deposit lifecycle — record deposit amount, custodian scheme, certificate reference, interest accrual, dispute claims, and automated return processing. Statutory compliance deadlines enforced (e.g., 30-day registration window). Breach of deposit protection legislation flagged and escalated automatically.

Impact: Eliminates regulatory exposure; deposit disputes drop 60–80% when evidence trails exist from day one.

---

## 2. Rent Arrears Tracking with Automated Escalation Ladders

Current state: Arrears are only mentioned as an EscalationType enum value with no financial tracking.

Improvement: `track_rent_arrears()` records each payment period, amount due, amount paid, days overdue, and arrears balance. Configurable escalation ladder: reminder at day 7, formal notice at day 14, legal referral trigger at day 28. Each step generates a compliant notice document and audit trail.

Impact: Average arrears recovery rate improves 35–50% with systematic early intervention versus ad-hoc chasing.

---

## 3. Tenancy Agreement Version Control and Digital Signing

Current state: lease_signing is an onboarding step with no document model.

Improvement: `create_tenancy_agreement()` records agreement version, clauses, special conditions, break clause dates, and rolling/fixed term flags. Digital signature workflow with multi-party sign tracking. Version history preserved — every amendment creates a new version linked to the original. Clause diff visible for renewals.

Impact: Dispute resolution time cut in half when exact agreed terms are recoverable per version with audit timestamps.

---

## 4. Predictive Churn Scoring with Behavioural Signals

Current state: Retention risk uses a single score threshold (< 40) with no forward-looking model.

Improvement: `predict_churn_probability()` computes a churn probability from weighted signals: satisfaction trend direction, days since last communication, service request resolution rate, lease expiry proximity, and rent-to-market ratio. Produces a 0–1 probability with contributing factor breakdown. Threshold configurable per tenant segment.

Impact: 6-month churn prediction at 75%+ accuracy enables proactive retention outreach before tenants enter notice period.

---

## 5. Compliance Calendar with Automated Reminders

Current state: Lease covenant compliance is a point-in-time record with a manual next_review_date.

Improvement: `compliance_calendar()` generates a forward-looking schedule of all statutory and contractual obligations: gas safety (annual), EICR (5-year), EPC review, rent review dates, break clause windows, lease expiry, deposit renewal. Each item has an owner, deadline, and days-to-deadline. Automated pre-deadline reminders via notify adapter.

Impact: Zero missed statutory inspections, eliminating fines that routinely reach £5,000–£30,000 per breach in UK residential portfolios.

---

## 6. Bulk Tenant Communication Campaigns

Current state: send_communication() handles one-to-one messages only.

Improvement: `send_bulk_communication()` targets tenant cohorts by status, type, property, or custom filter. Template variable substitution per recipient (name, unit, property). Delivery tracking aggregated per campaign. Unsubscribe handling per channel. Rate limiting to respect carrier throttles.

Impact: Operational communication (rent reviews, maintenance windows, building notices) moves from manual mail-merge to zero-effort automated dispatch.

---

## 7. Service Request SLA Analytics Dashboard Data

Current state: SLA breach is a boolean flag on individual requests with no aggregate view.

Improvement: `get_sla_performance_report()` computes per-request-type SLA metrics: response rate, average resolution time, breach count, breach rate, P50/P95 resolution time. Compares against configured targets. Highlights systemic underperformance by request type or assigned team member.

Impact: Facilities managers can identify which request types are chronically under-resourced rather than reacting to individual complaints.

---

## 8. Tenant Document Vault with Expiry Tracking

Current state: Document IDs are stored in onboarding step records but no document model exists.

Improvement: `register_document()` captures document type (passport, visa, proof of address, insurance cert, guarantor deed), issue date, expiry date, and issuing authority. `get_expiring_documents()` returns documents expiring within N days across the portfolio. Automated renewal reminders dispatched at configurable lead times.

Impact: Right-to-rent compliance failures (£3,000–£10,000 per tenant in UK law) are eliminated when expiry tracking is systematic.

---

## 9. Guarantor Management

Current state: No guarantor model exists despite being a standard tenancy risk mitigation tool.

Improvement: `register_guarantor()` links a guarantor entity to a tenant, records guarantee type (limited/unlimited), guarantee amount, credit check reference, and signed deed reference. `validate_guarantor_coverage()` checks whether total guaranteed exposure covers outstanding obligations. Guarantor notification workflows for rent arrears.

Impact: Portfolio credit exposure calculation becomes accurate; guarantor enforcement starts from a documented basis rather than a verbal agreement.

---

## 10. Move-In / Move-Out Inventory Integration

Current state: inventory is an onboarding step flag with no structured data.

Improvement: `record_inventory_inspection()` captures room-by-room condition records, fixture states, meter readings (gas, electric, water), and photographic evidence references. Checkout creates a diff against check-in: new damage items flagged with estimated remediation cost, feeding directly into deposit deduction claims.

Impact: Deposit dispute adjudication success rate rises from ~50% to 85%+ when condition evidence is structured and timestamped rather than narrative PDFs.

---

## 11. Tenant Self-Service Portal Action Log

Current state: portal_active is a boolean with no record of what tenants do via the portal.

Improvement: `log_portal_action()` records every self-service action: payment made, document uploaded, service request raised, survey submitted, communication read. Produces an activity feed per tenant. Inactivity detection flags tenants who haven't logged in for N days for proactive outreach.

Impact: Reduces inbound calls by 30–40% when tenants are engaged with self-service; disengagement detected early enables re-engagement before it signals churn.

---

## 12. Multi-Tenancy Subletting Detection and Control

Current state: subletting_unauthorised exists as an escalation type but no detection mechanism exists.

Improvement: `register_subletting_consent()` tracks approved subletting: sublessee details, consent period, rent pass-through terms, and head-lease restrictions. `detect_subletting_indicators()` flags anomalies: unusual access patterns, multiple occupants on utility accounts, short-let platform listings matched against tenant addresses via integration.

Impact: Protects landlord from insurance voidance and lease forfeiture exposure while providing a compliant route for legitimate subletting.

---

## 13. Rent-Free and Incentive Period Tracking

Current state: No lease economic terms model exists.

Improvement: `record_lease_incentive()` captures rent-free periods, fit-out contributions, stepped rent schedules, and rent caps. `get_effective_rent_schedule()` returns the actual cash flow for any lease period accounting for all incentives. Accounting amortisation schedule generated for straight-line rent recognition per IFRS 16.

Impact: Finance teams stop reconciling rent-free periods manually in spreadsheets; effective rent reporting becomes real-time.

---

## 14. Tenant Relationship Health Score Composite

Current state: tenant_score uses a single model with a simple low-score flag.

Improvement: `compute_relationship_health_score()` produces a composite score from four weighted dimensions: Financial Health (payment history, credit grade, arrears), Operational Health (service request frequency, SLA adherence by tenant), Engagement (survey response rate, portal activity, communication responsiveness), and Compliance (covenant compliance rate, onboarding completeness). Score drives automated tier classification (Platinum/Gold/Silver/Standard) with differentiated service levels.

Impact: Portfolio segmentation enables resource allocation proportional to relationship value rather than uniform service delivery.

---

## 15. Lease Break Clause Workflow

Current state: No break clause model or workflow exists.

Improvement: `register_break_clause()` records break date, notice period required, break conditions (e.g., no rent arrears, vacant possession), and current activation status. `check_break_clause_eligibility()` evaluates whether conditions are met at any given date. `activate_break_clause()` initiates the checkout workflow, generates required legal notices, and freezes the vacating timeline. Both landlord-break and tenant-break supported.

Impact: Break clause mismanagement is one of the highest-cost lease disputes (six-figure litigation); systematic tracking eliminates ambiguity about whether conditions were met.
