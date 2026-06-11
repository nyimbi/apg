# Underwriting Engine — World-Class Improvements

© 2025 Datacraft | Author: Nyimbi Odero

---

### I1. Predictive Loss-Cost Modelling with Gradient Boosting
**Category**: AI/ML
**Justification**: Pure heuristic scoring mis-prices tail risk by 20–40%. Gradient-boosted loss-cost models trained on portfolio history cut mis-pricing by 60%, enabling profitable growth at lower rates than competitors — the same approach that lifted Swiss Re's combined ratio 3 points.
**Implementation**: Persist a feature vector (age, vehicle_age, claim_history, product, region) per submission; expose `score_with_model(submission_id)` that calls a local Ollama-hosted LLM or scikit-learn GBRT endpoint, returning loss-cost estimate and confidence interval alongside the existing heuristic band.
**Competitive reference**: Swiss Re iptiQ, Quantemplate

---

### I2. Real-Time Telematics / IoT Risk Adjustment
**Category**: Feature
**Justification**: Static risk attributes miss dynamic exposure. Insurers with telematics (e.g., Root Insurance) charge 15–30% lower loss ratios by rating on actual behaviour. Offering mid-term rating adjustment on live sensor data is a hard differentiator for SME fleet and agri products.
**Implementation**: Add `ingest_telematics_event(submission_id, sensor_payload)` that accumulates driving-score deltas and calls `recompute_risk_score()`, emitting a `telematics_adjustment` audit event and flagging if re-rating is warranted.
**Competitive reference**: Root Insurance, Discovery Vitality Drive

---

### I3. Multi-Layer Reinsurance Programme Optimisation
**Category**: Feature
**Justification**: Single-treaty cession ignores programme layering (quota-share → XL → stop-loss). Munich Re's TreatyMate and similar tools show 8–12% improvement in net retained profit by optimising layer attachment points automatically.
**Implementation**: Add `optimise_reinsurance_programme(tenant_id, gross_premium, sum_insured)` that iterates active treaties ordered by layer priority, applies each in sequence, and returns a full cession waterfall with retained net exposure per layer.
**Competitive reference**: Munich Re TreatyMate, RMS TreatyIQ

---

### I4. Regulatory Solvency II / IFRS 17 Capital Allocation
**Category**: Compliance
**Justification**: Without SCR attribution per submission, insurers cannot price for required capital cost (typically 6–8% of SCR). IFRS 17 CSM computation also requires contractual service margin per group-of-contracts. Competitors offering embedded capital attribution (e.g., Earnix) command 40% pricing premium.
**Implementation**: Add `compute_scr_allocation(submission_id, risk_module)` using simplified standard-formula factors (premium risk, reserve risk, CAT risk) to return SCR contribution and minimum required premium loading for target ROE.
**Competitive reference**: Earnix, FIS Prophet, Moody's RMS

---

### I5. Automated Sanctions & AML Screening
**Category**: Security
**Justification**: Writing risk for OFAC/UN-sanctioned entities exposes insurers to regulatory fines exceeding policy premium by 1000×. Lloyd's requires PQ screening on all submissions. Embedding this at submission time is a compliance gate that prevents downstream liability.
**Implementation**: Add `screen_proposer(tenant_id, submission_id)` that hashes proposer name + ID against a configurable sanctions list (loaded from local file or REST endpoint), returns `CLEAR | HIT | POSSIBLE_MATCH` with a match score, and hard-blocks submission acceptance on `HIT`.
**Competitive reference**: Refinitiv World-Check, LexisNexis Bridger Insight

---

### I6. Dynamic Pricing Corridor with Min/Max Rate Guards
**Category**: Feature
**Justification**: Unconstrained underwriter rate adjustments create adverse selection spirals. Beazley and Markel maintain algorithmic rate corridors that prevent individual underwriters from going below technical price — reducing unprofitable business written by ~18%.
**Implementation**: Add `validate_rate_corridor(product_code, proposed_rate, risk_score)` that enforces configurable `min_rate_floor` and `max_rate_ceiling` per product/risk-band, returning `approved | floor_applied | ceiling_applied` with the adjusted rate.
**Competitive reference**: Beazley JAZZ, Markel IronNet

---

### I7. Portfolio Accumulation & Catastrophe Exposure Aggregation
**Category**: Risk Management
**Justification**: Insurers that lack real-time PML (Probable Maximum Loss) aggregation face unplanned CAT losses that breach their reinsurance XL attachment. RMS and AIR platform users avoid this with sub-second aggregation dashboards.
**Implementation**: Add `compute_pml_exposure(tenant_id, peril, region)` that aggregates committed sum-insured by geographic region/peril from all active submissions, computes 100-year PML using configurable damage factors, and flags breach of any XL treaty layer attachment.
**Competitive reference**: Verisk AIR Touchstone, RMS Risk Intelligence

---

### I8. Straight-Through Processing (STP) Pipeline
**Category**: Performance
**Justification**: Manual referral queues add 3–5 business days to placement. Convex and Brit achieved 70–80% STP rates by chaining: submit → assess → capacity_check → rate → bind in one atomic async pipeline for qualifying risks — compressing time-to-quote from days to seconds.
**Implementation**: Add `straight_through_process(tenant_id, submission_payload)` that executes submit → assess → check_capacity → rate_risk in sequence, auto-binds if all gates pass, and returns a full `stp_result` record with each step's outcome or the first failure reason.
**Competitive reference**: Convex Insurance, Brit Syndicate STP Engine

---

### I9. Underwriter Performance Scorecard
**Category**: UX
**Justification**: Without visibility into individual underwriter loss ratios and override patterns, management cannot detect adverse selection from manual overrides. Gallagher Bassett's UW analytics shows 12% combined-ratio uplift from underwriter-level accountability metrics.
**Implementation**: Add `underwriter_scorecard(tenant_id, underwriter_id, from_date, to_date)` that aggregates submissions assessed/overridden by that underwriter, computes override rate, recommended vs. actual premium variance, and referral resolution time.
**Competitive reference**: Gallagher Bassett, Xceedance Analytics

---

### I10. Behavioural Cohort Renewal Pricing
**Category**: AI/ML
**Justification**: Flat renewal rates drive profitable customers to competitors while retaining adverse ones. Zurich's renewal engine segments cohorts by CLV and claims experience, applying targeted adjustments that improved 13-month retention by 8%.
**Implementation**: Add `compute_renewal_adjustment(tenant_id, proposer_id, product_code)` that retrieves prior-period premium, claims, and risk-score trend for the proposer, then returns a recommended renewal loading/discount as a Decimal multiplier with justification string.
**Competitive reference**: Zurich Renewal Engine, Majesco Digital1st

---

### I11. Embedded Parametric Trigger Evaluation
**Category**: Feature
**Justification**: Parametric products (drought index, earthquake PGA, flood depth) settle instantly without loss adjustment — a 10× customer experience improvement over indemnity. WorldCover and ARC offer this natively; traditional UW engines cannot model trigger thresholds.
**Implementation**: Add `evaluate_parametric_trigger(tenant_id, submission_id, index_value, threshold, payout_schedule)` that compares an observed index value against contract thresholds and returns triggered payout amount as a Decimal, plus a `trigger_audit` record.
**Competitive reference**: WorldCover, African Risk Capacity (ARC), Etherisc

---

### I12. Peer-Comparable Rate Benchmarking
**Category**: UX
**Justification**: Underwriters with no market context over-price by 5–15% or under-price by 10–20%. Zywave and Verisk MarketStance provide real-time rate adequacy scores against anonymised peer benchmarks, improving pricing confidence dramatically.
**Implementation**: Add `benchmark_rate(product_code, risk_band, proposed_rate)` that compares proposed_rate against stored peer percentiles (p25/p50/p75) per product/band and returns an adequacy score (`above_market | at_market | below_market`) with percentile rank.
**Competitive reference**: Verisk MarketStance, Zywave BrokerBridge

---

### I13. Document-Driven Risk Extraction (AI OCR)
**Category**: AI/ML
**Justification**: 60–70% of underwriting time is spent reading submission documents. Cytora and Sequel claim 10× throughput improvement by extracting structured risk data from PDFs/emails into risk-attribute dicts automatically.
**Implementation**: Add `extract_risk_attributes_from_document(tenant_id, document_text, product_code)` that sends the document to a local Ollama model with a structured extraction prompt and returns a typed `risk_attributes` dict ready for `submit_risk`, with a confidence score per field.
**Competitive reference**: Cytora, Sequel Business Insights, Concirrus

---

### I14. Facultative Placement Workflow
**Category**: Feature
**Justification**: Large or unusual risks require facultative reinsurance placement before acceptance. Without a structured workflow, underwriters track placement via email — introducing errors and compliance gaps. AdvantageGo's OPUS platform reduced facultative placement time by 35%.
**Implementation**: Add `create_facultative_placement(tenant_id, submission_id, cedant_retention, markets)` that creates a structured facultative slip record, tracks percentage placed per market, and auto-completes once 100% is covered or flags partial placement risk.
**Competitive reference**: AdvantageGo OPUS, Brit Fac Placement Hub

---

### I15. Underwriting Letter / Policy Schedule Generation
**Category**: UX
**Justification**: Post-decision document generation is a bottleneck: 40% of straight-through rejections stall at the letter stage. Instanda and VIPR automate schedule generation from structured underwriting data, cutting policy issuance from 3 days to minutes.
**Implementation**: Add `generate_underwriting_letter(tenant_id, assessment_id, letter_type)` that resolves assessment + submission data, fills a Jinja2 template (acceptance/decline/referral letter types), and returns the rendered text plus metadata for downstream document storage.
**Competitive reference**: Instanda, VIPR, Applied Epic
