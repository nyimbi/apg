# leg_ip — World-Class Improvements

Fifteen targeted improvements that elevate this capability from basic registry to enterprise-grade IP intelligence platform, rivalling Anaqua, CPA Global (Clarivate), Dennemeyer, and IPfolio.

---

### I1. Tiered Royalty Engine with Decimal Precision
**Category**: Feature
**Justification**: Enterprise licensors use multi-tier royalty brackets (e.g., 5% on first KES 5M, 8% above). Float arithmetic silently loses fractional cents at volume; Decimal precision is non-negotiable for audited financials. Competitors like Anaqua Auto-Invoice use tiered schedules backed by precise arithmetic.
**Implementation**: Replace `float` royalty fields with `Decimal`; add `royalty_tiers` list-of-`(threshold, rate)` on licenses. `record_royalty_tiered` applies bracket logic and returns `Decimal` amounts with per-tier breakdowns stored on the royalty record.
**Competitive reference**: Anaqua AQX tiered royalty schedules; CPA Global Royalties module.

---

### I2. Smart Renewal Deadline Engine (jurisdiction-aware lead times)
**Category**: Compliance
**Justification**: The current `_compute_renewal_due` subtracts ~30 days naively. USPTO, EUIPO, and KIPI each have distinct statutory renewal windows. Missing the correct date can invalidate a patent worth millions. Dennemeyer and IP.com both embed jurisdiction deadline tables.
**Implementation**: Introduce a `RENEWAL_LEAD_DAYS` registry keyed by `(asset_type, jurisdiction)`. `_compute_renewal_due` looks up the correct lead time, falling back to a conservative 90-day default. Returns multiple milestone dates (180/90/30/7 days).
**Competitive reference**: Dennemeyer DIAMS iQ jurisdiction-aware docketing engine.

---

### I3. Opposition / Cancellation Case Tracker with Stage Machine
**Category**: Feature
**Justification**: Trademark oppositions and patent re-examinations have hard statutory deadlines and multiple parties. Tracking them as raw records with no workflow enforcement creates malpractice risk. IPfolio and FoundationIP both model opposition stages with gate enforcement.
**Implementation**: Add `OPPOSITION_STAGES` state machine (filed, evidence, hearing_set, suspended, decided, appealed). `advance_opposition_stage` validates legal transitions, records hearing dates, and emits audit events. Deadlines cascade to connected assets.
**Competitive reference**: IPfolio — opposition/cancellation case management with stage gate enforcement.

---

### I4. AI-Assisted Trademark Clearance Search
**Category**: AI/ML
**Justification**: Pre-filing clearance searches catch 80% of oppositions before they happen. CompuMark and Corsearch use phonetic + visual similarity algorithms; replicating even a simplified version eliminates a KES 300K+ attorney clearance fee per mark.
**Implementation**: `trademark_clearance_search` normalises the proposed mark using trigram overlap + character-level similarity, compares against same-class registered marks in the tenant portfolio, and returns a similarity-scored hit list with recommended action.
**Competitive reference**: CompuMark (Clarivate) AI trademark screening; Corsearch Brand Protection platform.

---

### I5. Assignment Chain with Chain-of-Title Validation
**Category**: Feature
**Justification**: IP ownership changes hands through M&A, spin-offs, and collateral agreements. Chain-of-title gaps invalidate enforcement actions. The current `assignments` dict stores records but never validates continuity. Anaqua enforces sequential assignment chains.
**Implementation**: `assign_asset` validates that the current `owner_id` matches `from_owner_id`, records the assignment with notarisation metadata and `consideration: Decimal`, updates `owner_id` atomically. `get_chain_of_title` returns ordered history with gap flags.
**Competitive reference**: Anaqua AQX chain-of-title reporting; CPA Global IP Management Suite.

---

### I6. Portfolio Valuation Engine (cost + income approaches)
**Category**: Feature
**Justification**: CFOs and investors need IP balance-sheet valuations for IAS 38 compliance. Cost approach (accumulated prosecution + renewal costs) and income approach (discounted royalty streams) are standard methods unavailable in any open-source IP tool.
**Implementation**: `compute_asset_valuation(method)` aggregates renewal fees and prosecution costs (cost approach), or discounts projected royalties at a configurable discount rate (income approach). Returns `Decimal` valuation with method assumptions embedded in the record.
**Competitive reference**: ktMINE royalty rate benchmarking; Ocean Tomo IP valuation reports.

---

### I7. Renewal Calendar Export (iCal + JSON feed)
**Category**: UX
**Justification**: IP managers live in Outlook and Google Calendar. Exporting renewal deadlines as iCal (`.ics`) lets them subscribe and receive native device alerts without logging into another portal. Competitors rely on static email reminders; a live iCal feed is a hard differentiator.
**Implementation**: `export_renewal_calendar` iterates expiring assets within a configurable window, generates RFC 5545-compliant VEVENT blocks with VALARM components, and returns the `.ics` payload as a string alongside a structured JSON feed.
**Competitive reference**: Dennemeyer calendar integration; Questel Orbit renewal reminders.

---

### I8. License Revenue Forecasting (time-series projection)
**Category**: AI/ML
**Justification**: Boards need "what royalties will we earn next 12 months?" Linear extrapolation of historical royalty records stratified by licensee growth gives a defensible forecast. No basic IP tool does this; enterprise platforms bill separately for analytics modules.
**Implementation**: `forecast_royalty_revenue(months_ahead)` groups paid royalties by license and period, fits a linear trend per license, sums projected monthly revenues, and returns a `{period: Decimal}` map with confidence bands based on growth-rate variance.
**Competitive reference**: Anaqua Analytics dashboards; Dennemeyer Financial Forecasting module.

---

### I9. Lapsed Asset Revival Candidate Finder
**Category**: Feature
**Justification**: ~15% of lapsed trademarks and patents can be revived within statutory grace periods (USPTO 6-month revival, EUIPO 2-month + appeal). Identifying lapsed assets still within their revival window recovers portfolio value that would otherwise be permanently lost.
**Implementation**: `lapsed_revival_candidates` queries assets with status `lapsed` or `expired`, computes days since lapse, filters to those within jurisdiction revival windows (from `REVIVAL_WINDOWS` registry), and returns revival deadline and fee estimates.
**Competitive reference**: Dennemeyer Revival Services; NovumIP lapse monitoring.

---

### I10. Multi-Currency Royalty Settlement with FX Snapshots
**Category**: Feature
**Justification**: Cross-border licenses settle in USD, EUR, or GBP while the licensor books in KES. Recording royalties without FX conversion creates phantom gains/losses. CPA Global and Anaqua support multi-currency settlement with configurable FX rate sources and audit-grade rate snapshots.
**Implementation**: `record_royalty` accepts optional `settlement_currency` and `fx_rate: Decimal`. Returns both `royalty_amount` in license currency and `royalty_amount_home` in tenant base currency. The rate used is stored on the record for full reproducibility.
**Competitive reference**: Anaqua Financial Management multi-currency reconciliation; CPA Global currency module.

---

### I11. IP Due Diligence Report Generator
**Category**: UX
**Justification**: M&A and investment diligence demands a structured IP report covering ownership, encumbrances, renewal status, and litigation risk. Generating this manually takes a paralegal 2–3 days. Automating it from structured data compresses this to seconds.
**Implementation**: `generate_due_diligence_report(target_owner_ids)` assembles asset details, ownership chain, active licenses, open oppositions, expiry schedules, and total encumbered value into a structured dict ready for PDF/DOCX rendering via APG's document capabilities.
**Competitive reference**: Anaqua Due Diligence module; CPA Global IP Audit Reports.

---

### I12. Watch Service with Similarity-Based Hit Detection
**Category**: Integration
**Justification**: Proactive brand protection requires monitoring competitor trademark filings. The existing `watches` dict is inert — it stores records but never triggers on external events. Corsearch and CompuMark charge premium fees for exactly this functionality.
**Implementation**: `create_watch` stores `watch_terms`, `jurisdictions`, `similarity_threshold: Decimal`. `process_watch_results(watch_id, hits)` ingests external feed results, scores similarity against registered marks using trigram overlap, and emits `watch_hit_detected` events for manual review.
**Competitive reference**: Corsearch automated trademark watch; CompuMark Global Watch.

---

### I13. Prosecution Cost Ledger per Asset
**Category**: Feature
**Justification**: Patent prosecution costs (filing, examination, appeal, translation) must be amortised over asset life under IAS 38. Without line-item cost tracking, finance cannot capitalise IP correctly. No free IP tool tracks prosecution costs; this feature alone justifies enterprise pricing.
**Implementation**: `record_prosecution_cost(asset_id, cost_type, amount: Decimal, currency, invoice_ref)` appends to a per-asset cost ledger. `get_prosecution_costs(asset_id)` returns itemised costs and `Decimal` total. `portfolio_summary` includes `total_prosecution_costs`.
**Competitive reference**: Anaqua Cost Management; Dennemeyer cost budget modules.

---

### I14. Regulatory Compliance Checklist per Jurisdiction
**Category**: Compliance
**Justification**: Each jurisdiction imposes unique maintenance obligations (working requirements for patents, use obligations for trademarks, mandatory recordal for assignments). Embedding jurisdiction compliance checklists converts passive data into active risk management.
**Implementation**: `get_compliance_checklist(asset_id)` returns obligation items from a static `JURISDICTION_OBLIGATIONS` registry, each with `due_date`, `completion_status`, and `risk_level`. `complete_compliance_item` marks an obligation met and emits an audit event.
**Competitive reference**: Dennemeyer Compliance Management; CPA Global Compliance Watch.

---

### I15. Embodiment / Product-Asset Mapping
**Category**: Integration
**Justification**: IP litigation defence requires proving which products embody which patents. Without a product-to-patent map, a cease-and-desist letter cannot be answered quickly. This is the foundation of Lex Machina's licensing analytics and Anaqua's product-IP mapping.
**Implementation**: `link_asset_to_product(asset_id, product_id, embodiment_description, linked_by)` records the mapping with timestamp. `get_assets_for_product(product_id)` surfaces all IP protecting a given product SKU, enabling rapid freedom-to-operate responses.
**Competitive reference**: Lex Machina patent analytics; Anaqua AQX product-IP mapping.
