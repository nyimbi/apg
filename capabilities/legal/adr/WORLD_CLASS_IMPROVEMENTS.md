# leg_adr — World-Class Improvement Roadmap

Fifteen improvements that move leg_adr from functional to best-in-class.

---

### I1. Decimal-precision monetary fields
**Category**: Compliance
**Justification**: `float` arithmetic introduces rounding errors that violate financial-reporting standards (IFRS 9, IAS 37) and create audit failures on cross-border enforcement filings where even 0.01 KES discrepancies are material. Competing products that use floats (older Relativity implementations) have issued compliance advisories.
**Implementation**: Replace every `float` monetary field (`claim_amount`, `award_amount`, `settlement_amount`, `fee_rate`, `costs_awarded`) with `Decimal`; accept `str | Decimal` at the boundary and coerce via `Decimal(str(v))`.
**Competitive reference**: Gallagher Bassett ClaimSpace — uses exact decimal types in all financial fields for SOX compliance.

---

### I2. Statute-of-limitations / time-bar guardian
**Category**: Compliance
**Justification**: Missing a limitation deadline is the single most common malpractice claim in ADR. No mid-market ADR SaaS product proactively blocks filings that are time-barred or warns counsel when a deadline is within 30 days. Winning this gap captures the risk-management buyer.
**Implementation**: `async def check_limitation_period(tenant_id, case_id, governing_law, cause_of_action, incident_date)` calculates days remaining under a configurable rule table keyed by `(governing_law, cause_of_action)` and returns a `{days_remaining, deadline, status: ok|warning|expired}` payload.
**Competitive reference**: ContractPodAi — surfaces contract deadline risk in-UI; no ADR platform has replicated this for limitation periods.

---

### I3. Procedural deadline calendar with overdue surfacing
**Category**: Feature
**Justification**: Missed procedural deadlines (response to notice, statement of claim, exchange of documents) auto-result in default awards — an existential risk. No open-source ADR product bundles an integrated deadline scheduler; lawyers currently track these in spreadsheets.
**Implementation**: `create_deadline` stores deadlines keyed to case/proceeding with `label`, `due_date`, `responsible_party_id`; `list_overdue_deadlines` returns all past-due items with computed `days_overdue` for webhook dispatch.
**Competitive reference**: Clio Manage — comprehensive deadline calendar for litigation; this brings it natively to ADR workflows.

---

### I4. Multi-party / multi-claimant support
**Category**: Feature
**Justification**: Complex construction, infrastructure and insurance disputes routinely involve 4–8 parties (e.g., main contractor + sub-contractors + insurer + employer). Current schema has a single `claimant_id` and `respondent_id`, blocking real-world complex arbitrations. ICC and SIAC rules explicitly govern multi-party proceedings.
**Implementation**: `add_party(tenant_id, case_id, party_id, party_role, counsel_id, joinder_date)` with `party_role ∈ {claimant, respondent, third_party, intervener}` and `list_parties(tenant_id, case_id)` with full role breakdown.
**Competitive reference**: Tymetrix 360 — handles multi-party matter management natively.

---

### I5. Cost ledger with running arbitration cost exposure
**Category**: Feature / Performance
**Justification**: Arbitration costs are often disputed at the award stage; a running ledger that tracks each neutral's billable time against their fee rate, plus institution fees and disbursements, enables real-time cost-exposure dashboards. FTI Consulting charges $500/hr to produce these manually.
**Implementation**: `record_cost_entry(tenant_id, case_id, entry_type, amount: Decimal, payable_by, description)` adds a ledger line; `get_case_costs(tenant_id, case_id)` aggregates by payable_by and entry_type, returning a `total_costs` Decimal.
**Competitive reference**: FTI Consulting ADR cost-tracking module exports LEDES billing entries per matter.

---

### I6. Conflict-of-interest disclosure registry
**Category**: Security / Compliance
**Justification**: IBA Guidelines on Conflicts of Interest require neutrals to disclose any connection to parties or counsel; storing and querying these disclosures electronically is a prerequisite for ISO-accredited arbitration institutions.
**Implementation**: `record_coi_disclosure(tenant_id, case_id, neutral_id, disclosure_text, waived)` creates a disclosure record; `list_coi_disclosures(tenant_id, case_id)` enables institution secretaries to review outstanding waivers before panel constitution.
**Competitive reference**: ICSID Administrative and Financial Regulation 6 mandates written disclosure and secretary review.

---

### I7. Settlement installment plan tracking
**Category**: Feature / Compliance
**Justification**: Many settlements are structured as installment payments over 6–36 months; the current single `settlement_amount` field cannot model this, forcing counsel to create side-spreadsheets and missing payment events.
**Implementation**: `add_settlement_installment(tenant_id, settlement_id, due_date, amount: Decimal)` attaches a payment schedule entry; `mark_installment_paid` updates status; `list_overdue_installments` surfaces missed payments.
**Competitive reference**: Gallagher Bassett structured settlement module tracks installment compliance with automated alerts.

---

### I8. New York Convention enforcement eligibility check
**Category**: Compliance / Feature
**Justification**: Whether a foreign award can be enforced in a given jurisdiction depends on New York Convention membership; surfacing this at the enforcement-filing stage prevents futile enforcement attempts and protects the firm from malpractice exposure.
**Implementation**: `check_enforcement_eligibility(tenant_id, award_id, jurisdiction)` cross-references against a bundled `NYC_CONTRACTING_STATES` set and returns `{eligible, basis, caveats}` — no external API call required.
**Competitive reference**: Wolters Kluwer Kluwer Arbitration database provides treaty-based enforcement eligibility checks per jurisdiction.

---

### I9. Document bundle management with hash integrity
**Category**: Security
**Justification**: Chain of custody for submitted evidence and pleadings is critical for award enforceability; tampering allegations are a common set-aside ground. Recording SHA-256 hashes at submission time provides immutable proof of document integrity.
**Implementation**: `add_document(tenant_id, case_id, reference, doc_type, filed_by, document_hash, page_count)` stores metadata + hash; `list_documents(tenant_id, case_id)` returns the full bundle filtered by doc_type.
**Competitive reference**: NetDocuments + Epiq ADR integration — document integrity tracking for arbitration.

---

### I10. Case risk score (0–100) with factor drill-down
**Category**: AI/ML / UX
**Justification**: General counsel need a single number that aggregates procedural health (deadlines met, neutrals confirmed, submissions filed) to flag at-risk cases before they become emergencies — a standard feature of mature legal ops platforms.
**Implementation**: `compute_case_risk_score(tenant_id, case_id)` calculates a 0–100 integer from weighted factors: overdue deadlines (+25 each), challenged neutrals (+20 each), no panel after 30 days (+15), award overdue (+20); returns `{score, risk_band, factors}`.
**Competitive reference**: Onit Enterprise Legal Management surfaces a case health score with contributing factor drill-down.

---

### I11. Procedural timetable with Gantt-ready schedule
**Category**: Feature / UX
**Justification**: Parties routinely agree a procedural timetable at the preliminary conference; without a structured representation this lives in a Word document, making automated reminders and compliance tracking impossible.
**Implementation**: `create_timetable_item(tenant_id, case_id, milestone, due_date, responsible_party)` adds a milestone; `get_timetable(tenant_id, case_id)` returns items sorted by `due_date` with computed `days_remaining`.
**Competitive reference**: Opus 2 Magnum renders interactive Gantt views of arbitration timetables for all parties.

---

### I12. Seat-specific default rules engine
**Category**: Feature / Compliance
**Justification**: UNCITRAL, ICC, LCIA, Nairobi Centre, and SIAC each impose different default timelines and notice periods; hard-coding "KES / Nairobi" defaults while being unaware of seat-imposed constraints is a silent compliance gap.
**Implementation**: `SEAT_RULES` dict maps seat/institution codes to a dict of default timelines (e.g., `response_days`, `award_deadline_days`); `get_seat_defaults` exposes these so `create_case` can auto-populate preliminary deadlines.
**Competitive reference**: UNCITRAL Rules Revision 2021 mandated institution-specific defaults in all UNODC-compliant systems.

---

### I13. Multi-currency award conversion with FX audit trail
**Category**: Feature / Performance
**Justification**: Cross-border arbitrations regularly issue awards in one currency while enforcement occurs in another; without a recorded conversion rate at the time of award, parties dispute the enforceable amount years later.
**Implementation**: `record_currency_conversion(tenant_id, award_id, from_currency, to_currency, rate, rate_date)` stores the conversion record; `get_award_in_currency` returns `converted_amount` Decimal using the stored rate.
**Competitive reference**: ICC International Court of Arbitration practice direction on multi-currency awards explicitly requires rate documentation.

---

### I14. Panel registry with specialisation filtering
**Category**: Integration / Feature
**Justification**: Appointing-authorities maintain lists of accredited arbitrators with specialisations and availability; exposing a `search_panel_registry` hook that filters the tenant's neutral roster by seat, specialisation, and language enables institution-quality panel selection in seconds rather than hours.
**Implementation**: `register_panel_member(tenant_id, neutral_id, specialisations, languages, seat_accreditations)` adds to tenant roster; `search_panel_registry(tenant_id, specialisation, language, seat)` returns filtered matches sorted by active_appointment_count.
**Competitive reference**: SIAC panel search and LCIA Directory both expose structured specialisation/language filtering.

---

### I15. Case-level SLA monitoring
**Category**: Performance
**Justification**: ADR institutions (Nairobi Centre, NCIA, LCIA) are contractually bound to SLAs on case management — e.g., panel constituted within 30 days of filing, award within 12 months. Tracking SLA breach risk proactively protects institutional reputation and enables performance benchmarking.
**Implementation**: `compute_case_sla_status(tenant_id, case_id)` measures elapsed time at each status transition against configurable SLA thresholds, returning `{sla_status: on_track|at_risk|breached, days_elapsed, days_budget, transitions}`.
**Competitive reference**: Huron Consulting ADR benchmarking — SLA analytics for dispute resolution institutions.
