# Pharmaceutical Supply Chain — World-Class Improvements

**Capability**: `pharma_sup` | **Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Serialisation & Track-and-Trace (GS1/DSCSA/FMD)

**Gap**: No unit-level serialisation. Current PO workflow tracks bulk quantity only.

**Improvement**: Add `SerialNumber` model with GTIN + serial + lot + expiry quadruplet. Implement `serialise_batch()` to issue GS1-compliant SSCCs, `verify_serial()` for FMD/DSCSA verification queries, and `decommission_serial()` on dispense/destruction. Wire events to the EU Medicines Verification System (EMVS) stub. This eliminates counterfeit exposure at every custody transfer.

---

## 2. Cold Chain Continuous Monitoring

**Gap**: `transport_condition` field on PO is a free-text string. No temperature excursion workflow.

**Improvement**: Add `ColdChainShipment` model capturing logger device ID, set-point range (e.g. 2–8 °C), geo-route, and a time-series of readings. Implement `record_temperature_reading()`, `detect_excursion()` (MKT calculation per ASTM E1484), and `evaluate_excursion_impact()` (stability budget comparison). Auto-trigger quarantine via `update_supply_security()` and create CAPA record. WHO PQP and EMA GDP Annex 15 compliant.

---

## 3. GDP Compliance Gate on Every Shipment

**Gap**: GDP rules are referenced in the contract but not enforced programmatically at shipment creation.

**Improvement**: Implement `gdp_compliance_gate()` as a pre-shipment check that validates: qualified carrier on approved list, transport mode matches product GDP category, temperature logger commissioned, CMR/AWB document set complete, no open GDP deviations for this lane. Returns a structured `GdpClearance` object with pass/fail and blocking reason. Gate is mandatory before `place_order()` transitions to `in_transit`.

---

## 4. Automated Recall Management (Class I/II/III)

**Gap**: No recall capability exists. Shortage management does not cover recall-triggered withdrawals.

**Improvement**: Add `initiate_recall()` that classifies recall by FDA/EMA tier, generates a DUNS-linked affected-lot matrix from batch genealogy, and creates customer notification batches via `ntfy`. Implement `track_recall_progress()` to reconcile units acknowledged vs units distributed for regulatory effectiveness check. Audit trail satisfies 21 CFR Part 7 and EMA GMP Annex 16 requirements.

---

## 5. Supplier Performance Scorecard (KPI-Driven Re-qualification)

**Gap**: `qualify_supplier()` is a one-time gate; no ongoing performance measurement.

**Improvement**: Implement `calculate_supplier_scorecard()` that aggregates OTIF, right-first-time batch release rate, CoA deviations per 100 deliveries, complaint rate, and audit finding closure time into a weighted score (0–100). Scores below threshold trigger automatic `requalification_required` status and schedule an audit via `schd`. High scorers get preferred-supplier designation used by `demand_planning()` to rank source selection.

---

## 6. Multi-Tier Supply Chain Visibility (n-Tier Mapping)

**Gap**: `supply_risk_assessment()` accepts a flat node list. Real pharma chains are multi-tier (Tier 1 API supplier → Tier 2 reagent supplier → Tier 3 mine).

**Improvement**: Add `SupplyChainNode` model with `tier`, `parent_node_id`, and `criticality_class`. Implement `map_supply_chain()` to build a directed acyclic graph, `identify_single_points_of_failure()` using topological sort, and `propagate_disruption_impact()` for scenario simulation. Output feeds `security_of_supply_monitoring()` with tier-aware risk scores.

---

## 7. Intelligent Demand Sensing with AI Uplift

**Gap**: `demand_planning()` uses simple compound-growth projection. No real signal integration.

**Improvement**: Implement `demand_sensing()` that ingests point-of-dispensing data, epidemiological signals (disease prevalence APIs), and historical fill-rate corrections. Feed signals into a Prophet-compatible decomposition via local Ollama (`llama3`). Produce a 13-week rolling forecast with prediction intervals. Uncertainty quantification drives dynamic safety stock adjustment beyond the fixed 1.5× factor.

---

## 8. Regulatory Dossier Linkage (CTD Module 3 / DMF)

**Gap**: Supplier qualification stores a quality agreement reference but no link to the regulatory submission dossier.

**Improvement**: Add `regulatory_dossier` field on `Supplier` linking to Drug Master File (DMF) number, ASMF reference, or CEP. Implement `validate_dossier_currency()` to check DMF annual report submission status and flag if the dossier is outdated. Block `place_order()` if the supplier's DMF is withdrawn or pending annual update.

---

## 9. Contract Price & Volume Commitment Enforcement

**Gap**: `SupplyContract` has no financial terms. There is no mechanism to detect over/under-purchase against contracted volumes.

**Improvement**: Add `ContractTerm` model with `min_annual_quantity`, `max_annual_quantity`, `unit_price`, `price_escalation_clause`, and `currency`. Implement `check_commitment_adherence()` that aggregates annual PO volumes against committed bands, warns on under-purchase (rebate risk) or over-purchase (price break missed), and triggers renegotiation workflow at 80% of contract volume.

---

## 10. Counterfeit Detection & Supply Chain Integrity

**Gap**: No authentication layer on received materials. Serialisation alone is insufficient if EPCIS events are spoofed.

**Improvement**: Implement `verify_supply_chain_integrity()` that cross-references received serial numbers against manufacturer-published EPCIS event repository, checks ATP (Aggregation-to-Package) tree completeness, and flags orphan serials (no upstream custody event). Integrate with local Ollama vision model to analyse pack-level hologram images. Non-conforming units auto-quarantined.

---

## 11. Shortage Prediction (Proactive vs Reactive)

**Gap**: `shortage_management()` is triggered after a shortage is declared. Prediction horizon is zero.

**Improvement**: Implement `predict_shortage_risk()` that triangulates: forecast coverage vs. supplier lead time, supplier financial health score (Dun & Bradstreet feed), geopolitical risk index for country-of-manufacture, and inventory days on hand. Output a 90-day forward-looking shortage probability per product, enabling proactive safety stock build or alternate-source qualification before a shortage materialises.

---

## 12. Dual Sourcing Workflow Automation

**Gap**: `supply_risk_assessment()` recommends dual sourcing but does not trigger the sourcing workflow.

**Improvement**: Implement `initiate_dual_sourcing()` that: identifies candidate alternate suppliers from the unqualified supplier pool matching the required material, creates a qualification plan with milestones, assigns tasks via `wflo`, and tracks progress through to ASL inclusion. Sets a provisional `contingency_supplier_id` on the `SupplySecurityRecord` once the candidate passes analytical testing, enabling partial coverage before full qualification.

---

## 13. Audit Trail Cryptographic Integrity (ALCOA+)

**Gap**: `_audit()` stores plain dicts in memory with no tamper-evidence. Fails ALCOA+ attributable/contemporaneous/original requirements.

**Improvement**: Each audit event should be hash-chained: `event_hash = SHA256(prev_hash + json(event))`. Store the chain root in a `AuditChain` table. Implement `verify_audit_chain()` to detect any retrospective modification. Sign events with the actor's key (HMAC-SHA256 using tenant-scoped secret). Export compliant with 21 CFR Part 11 and EU Annex 11 electronic records requirements.

---

## 14. Batch Genealogy Integration with `pharma_mfg`

**Gap**: CMO orders (`cmo_order()`) are stored as plain dicts with no link to the manufacturing batch record.

**Improvement**: Return a typed `CmoOrder` model from `cmo_order()`. Add `batch_genealogy_id` field populated when `pharma_mfg` creates the batch record. Implement `get_batch_supply_genealogy()` that walks from finished product batch → CMO order → API PO → supplier DMF. This complete audit thread is required for quality investigations and Qualified Person (QP) batch certification.

---

## 15. Regulatory Intelligence Feed (EMA/FDA Supply Notifications)

**Gap**: Import license expiry is the only regulatory compliance event type. No awareness of regulatory agency communications affecting the supply chain.

**Improvement**: Implement `ingest_regulatory_intelligence()` as an async background task that polls EMA SCENIHR feed and FDA drug shortage database RSS, normalises entries against the tenant's product portfolio, and auto-creates `SupplySecurityRecord` updates or shortage records when a listed product matches. This converts reactive shortage response into proactive regulatory-informed supply planning, typically providing 30–90 days of advance warning.
