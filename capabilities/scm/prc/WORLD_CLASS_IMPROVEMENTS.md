# World-Class Improvements — Procurement Management (scm_prc)

## 1. Dynamic Tolerance Tiers for Three-Way Match

Replace the hardcoded 1%/5% tolerance with configurable tiers stored per-tenant (e.g. by commodity class, vendor tier, or PO value band). Allows tight control on high-value POs while relaxing checks on petty purchases, reducing noise in AP queues without sacrificing compliance.

## 2. Supplier Risk Scoring Engine

Compute a composite risk score per vendor from late delivery rate, dispute frequency, financial health indicators, and geographic/geo-political exposure. Surfaces vendors approaching risk thresholds before they become active disruptions, enabling proactive dual-sourcing decisions.

## 3. Automated PO Approval Workflows

Introduce configurable approval chains (single-tier, two-tier, budget-holder matrix) triggered by PO value, commodity category, or vendor risk tier. Stores approval history, delegation paths, and escalation timestamps for audit; blocks send until chain is complete.

## 4. Contract Spend-Down Tracking

Track consumed value against each contract ceiling in real time as POs are raised. Emit `contract_nearing_limit` events at 80%/95% utilisation. Prevents maverick spend beyond contract value and gives category managers early warning for renewal negotiations.

## 5. RFQ Comparative Scoring (Weighted Criteria)

Replace single-criterion award with a configurable weighted scorecard (price, lead time, quality certification, sustainability score). Produces an auditable award recommendation that can be overridden with a mandatory justification note.

## 6. Catalog Integration & Punchout Support

Link line items to an internal item catalog with approved suppliers, standard unit prices, and lead-time benchmarks. Flag off-catalog items requiring additional approval. Provides the foundation for guided buying and spend-under-management improvement.

## 7. Delivery Schedule & Milestone Tracking

Allow a PO to carry an ordered delivery schedule (multiple expected receipt dates per line). Track schedule adherence, compute on-time delivery rate per vendor, and escalate overdue milestones to the buyer automatically.

## 8. Invoice Discounting & Early-Payment Programs

Model early-payment discount terms (e.g. "2/10 NET30") per PO. Surface discount windows in the AP worklist, calculate annualised yield, and record whether the discount was captured — feeds dynamic discounting analytics.

## 9. Spend Forecasting with Seasonality

Generate a forward spend forecast by vendor and category using historical PO patterns, open PO commitments, and contract run-rates. Outputs monthly bucket forecasts to feed cash-flow planning and working-capital models.

## 10. Commodity Price Index Benchmarking

Compare quoted unit prices against an external commodity index (e.g. Bloomberg BCOM, internal category benchmarks). Compute price-competitiveness scores per RFQ response and flag quotes more than a configurable percentage above index.

## 11. ESG / Sustainability Supplier Scorecard

Extend vendor evaluation with ESG dimensions: carbon footprint, labour standards certification, diversity classification. Enables supplier sustainability ranking and regulatory reporting (e.g. CSRD, scope 3 supply chain emissions).

## 12. Exception-Based Alerts & SLA Engine

Define configurable SLAs per document type (e.g. "PO must be acknowledged within 48 hours", "disputed invoice must be resolved within 5 business days"). Emit alert events when SLA breaches are imminent or have occurred, triggering escalation paths.

## 13. Audit Trail Immutability & Tamper Evidence

Persist audit events to an append-only log with SHA-256 chaining (each event includes a hash of the previous event). Makes the procurement audit trail cryptographically tamper-evident, satisfying financial control requirements without a full blockchain solution.

## 14. Multi-Currency Normalisation

All monetary values stored in both transaction currency and a base reporting currency using dated exchange rates. Spend analytics aggregated in reporting currency regardless of vendor invoicing currency, eliminating FX distortions in management reporting.

## 15. Procurement Process Mining & Bottleneck Detection

Reconstruct the actual process graph from audit events (e.g. mean time from RFQ issue to award, PO draft to acknowledgement). Compare against target cycle times and surface bottlenecks — feeds continuous-improvement programmes and buyer productivity metrics.
