# World-Class Improvements — Extension Services (agr_ext)

Fifteen improvements that elevate agr_ext from a basic CRUD log to a competitive
precision-extension platform.

---

### I1. Farmer Profiling & Advisory Personalisation
**Category**: AI/ML
**Justification**: Generic advisories fail 40–60% of the time because they ignore farm
context (soil type, altitude, prior inputs). Personalised advisory delivery increases
adoption rates by 3–5x — the core metric funders and extension programmes report on.
**Implementation**: Maintain a per-farmer profile (crop portfolio, input budget tier,
literacy level, preferred language) and score incoming advisories for relevance; surface
the match score and auto-translate short advisories via a pluggable LLM adapter.
**Competitive reference**: OneSoil Advisor, Farmerline ICTD personalisation engine

---

### I2. Multi-Channel Delivery Tracking with Read Receipts
**Category**: Feature
**Justification**: Extension programmes run on SMS, IVR, WhatsApp and field visits
simultaneously. Without delivery state (sent → delivered → read → responded) programme
managers cannot tell whether an advisory reached the farmer or sat unsent.
**Implementation**: Add `delivery_status` lifecycle FSM per advisory record with
timestamps for each state transition; emit `advisory.delivery_state_changed` events
consumable by downstream notification capabilities.
**Competitive reference**: Twilio Engage, Digital Green Farmer.CHAT

---

### I3. Cost-per-Advisory & Programme Budget Tracking
**Category**: Compliance
**Justification**: Donor-funded extension programmes are required to report cost per
beneficiary reached. Without Decimal-precise cost tracking the reporting spreadsheet
becomes a manual reconciliation exercise every quarter.
**Implementation**: Accept `cost_amount` (Decimal) + `cost_currency` on advisory and
training creation; expose `get_programme_cost_summary()` that aggregates total cost,
cost per unique farmer reached, and cost per training seat filled.
**Competitive reference**: GSMA AgriTech Impact Framework, USAID CLA cost reporting

---

### I4. Demo Plot Yield Comparison Engine
**Category**: Feature
**Justification**: Demo plots exist specifically to prove technology adoption benefits.
Without a structured before/after yield comparison, field staff record text outcomes
that no one analyses — eliminating the entire scientific value of the plot.
**Implementation**: Add `baseline_yield_kg_ha` and `demonstrated_yield_kg_ha` fields to
demo plots; `get_demo_plot_impact_report()` computes yield lift %, breakeven input cost,
and technology adoption recommendation per crop type.
**Competitive reference**: CIMMYT On-Farm Trial Manager, IITA DemoPlotsDB

---

### I5. Training Certification & CPD Credit Tracking
**Category**: Compliance
**Justification**: Kenya and most SSA governments now require Continuing Professional
Development (CPD) records for licenced extension agents. Lacking built-in CPD tracking
forces agencies to duplicate records in paper registries.
**Implementation**: Attach `cpd_credits` (Decimal) and `certificate_number` to completed
trainings; `get_cpd_statement(worker_id)` returns a dated credit ledger for regulatory
submission.
**Competitive reference**: Corteva Agriscience Learning Portal, ATVET CPD registry

---

### I6. Seasonal Advisory Calendar & Agronomic Scheduling
**Category**: Feature
**Justification**: Extension workers spend 30% of planning time manually computing when
to send which advisory for which crop stage. An automated calendar tied to planting
dates eliminates that and ensures no critical crop-stage window is missed.
**Implementation**: `create_advisory_schedule(farmer_id, crop_type, planting_date)`
generates a date-keyed list of recommended advisory touchpoints derived from a crop
phenology template; supports long rains, short rains, and irrigated season types.
**Competitive reference**: John Deere Operations Center agronomic calendar, aWhere

---

### I7. Knowledge Base Versioning & Approval Workflow
**Category**: Compliance
**Justification**: Agronomic recommendations change (pesticide re-registrations, new
resistant varieties). Publishing outdated guidance without an approval chain exposes
extension agencies to regulatory liability and farmer harm.
**Implementation**: Add `version` (int), `status` (draft/review/approved/archived) and
`reviewed_by` to knowledge articles; `publish_knowledge_article()` transitions to
approved only after reviewer sign-off is recorded.
**Competitive reference**: FMC AgronomyIQ content governance, Bayer CropScience portal

---

### I8. Geospatial Advisory Heat-Map Data Export
**Category**: Integration
**Justification**: Programme managers need to show donors a map of where advisories were
delivered vs. where follow-up is still needed. Without structured geo data, they
manually geocode CSV exports.
**Implementation**: Accept optional `latitude`/`longitude` on advisories and demo plots;
`get_advisory_geo_export(bbox)` returns GeoJSON FeatureCollection of advisory points
filterable by date range and crop type for direct Mapbox/Leaflet consumption.
**Competitive reference**: ESRI ArcGIS for Agriculture, One Acre Fund reach dashboards

---

### I9. Extension Worker Gamification & Performance Leaderboard
**Category**: UX
**Justification**: Extension worker motivation is the single largest determinant of
programme quality in SSA. Organisations that introduced simple leaderboards and badges
report 20–35% improvements in advisory completeness without salary changes.
**Implementation**: `get_leaderboard(period)` ranks workers by weighted score:
(advisories × 1) + (follow-ups completed × 2) + (unique farmers × 3) + (trainings × 5);
includes badge assignment logic for first_advisory, 50_farmers, 100_follow_ups milestones.
**Competitive reference**: Grameen Foundation Motech, Vodafone m-Farm gamification

---

### I10. Farmer Feedback & Advisory Effectiveness Rating
**Category**: Feature
**Justification**: The only way to know if an advisory actually changed farming practice
is to capture structured feedback from the farmer. Without it, extension programmes
cannot demonstrate behaviour-change impact to investors.
**Implementation**: `submit_advisory_feedback(advisory_id, farmer_id, rating, adopted,
notes)` records 1–5 star rating plus a boolean `adopted_recommendation`; aggregate
`get_advisory_effectiveness()` reports adoption rate by topic, channel, and worker.
**Competitive reference**: Esoko Advisory Feedback, Hello Tractor Service Rating

---

### I11. Bulk SMS / IVR Advisory Broadcast
**Category**: Feature
**Justification**: When a pest outbreak or drought alert requires reaching 10 000 farmers
immediately, creating individual advisory records one-by-one is impractical. Bulk
broadcast + delivery tracking is table-stakes for any modern extension platform.
**Implementation**: `broadcast_advisory(farmer_ids, payload, channel)` atomically creates
individual advisory records in a batch transaction, returns a broadcast summary with
batch_id, count, and estimated delivery time; plugs into the agr_comm capability.
**Competitive reference**: Safaricom Digifarm, CropIn SmartFarm SMS blasts

---

### I12. Integration Bridge — Farm Parcel & Weather Data
**Category**: Integration
**Justification**: An advisory about "apply fertiliser now" is meaningless without the
weather forecast. Linking advisories to the agr_parcel and agr_weather capabilities
surfaces the farm context (soil type, last rainfall, ETo) inline on each advisory record.
**Implementation**: `enrich_advisory_with_context(advisory_id)` calls agr_parcel for
parcel attributes and agr_weather for 7-day forecast; appends a `context_snapshot` dict
to the advisory record, invalidated after 24 h using BoundedCache TTL.
**Competitive reference**: Trimble Ag Connect, Proagrica advisory enrichment

---

### I13. Off-line Queue & Sync for Field Operations
**Category**: Performance
**Justification**: 60–70% of advisory delivery happens in areas with no mobile data. Field
staff capture advisories on device and sync when connectivity returns. Without a formal
sync protocol, duplicate records corrupt programme reports.
**Implementation**: `submit_offline_batch(records, device_id, captured_at)` applies
idempotent upsert keyed on `(device_id, local_ref)`; returns per-record
`created | merged | rejected` status with conflict details.
**Competitive reference**: Twiga Foods offline-first agent app, Safaricom Digifarm lite

---

### I14. Automated Follow-Up Escalation
**Category**: Feature
**Justification**: Overdue follow-ups are invisible in current reports until manually
audited. Automatic escalation after a configurable SLA window reduces farmer churn from
unresolved agronomic problems.
**Implementation**: `get_overdue_follow_ups(sla_days)` scans advisories where
`follow_up_required=True`, `follow_up_done=False`, and `delivered_at` older than
`sla_days`; returns sorted list with days overdue, worker contact, and suggested
escalation action.
**Competitive reference**: Salesforce Agentforce SLA escalation, CRM-AG follow-up rules

---

### I15. Multilingual Knowledge Translation Pipeline
**Category**: AI/ML
**Justification**: Kenya has 42+ languages; Tanzania 126. A Swahili-only knowledge base
excludes 30–40% of smallholder farmers by literacy. Auto-translation with human review
breaks the language barrier at near-zero marginal cost.
**Implementation**: `request_translation(article_id, target_language)` creates a
translation job record linked to the source article; `submit_translation_result(job_id,
translated_content, translator_id)` creates a new article variant with
`source_article_id` back-reference and `status=review`; `approve_translation(article_id)`
publishes it.
**Competitive reference**: CGIAR BigFarm knowledge portal, FAO e-agriculture multilingual
