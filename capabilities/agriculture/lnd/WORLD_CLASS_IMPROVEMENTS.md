# Land Management (agr_lnd) — World Class Improvements

### I1. Adjudication & Dispute Resolution Workflow
**Category**: Feature
**Justification**: Unresolved land disputes stall credit access and development. A structured adjudication pipeline with evidence attachments, hearing schedules, and binding resolutions reduces informal corruption and gives lenders confidence — 10x over ad-hoc status flags.
**Implementation**: Add `_disputes` store with states: `filed → evidence_collection → hearing_scheduled → adjudicated → appealed`. Each dispute links to the parcel, claimants, evidence URLs, assigned adjudicator, and a final resolution record. `parcel.status` locks to `disputed` while open.
**Competitor**: Trimble AgriData (Australia), Kenya National Land Commission NLC-DRS

### I2. Encumbrance & Charge Registry
**Category**: Compliance
**Justification**: Banks, microfinance institutions, and co-operatives need a reliable way to register mortgages, caveats, and liens against parcels. Without this, double-collateralisation fraud is routine. Matches the legal requirement under Kenya's Land Registration Act §59.
**Implementation**: `_encumbrances` store: `{ parcel_id, type: mortgage|caveat|lien|easement, holder_id, amount, currency, registered_at, discharged_at }`. Parcel listing surfaces active encumbrances. Discharge workflow flips `discharged_at` and emits `encumbrance.discharged`.
**Competitor**: ESRI ArcGIS Land Records, Infor Public Sector Land Management

### I3. Valuation Roll & Automated Rate Assessment
**Category**: Feature
**Justification**: County governments levy land rates against assessed values. A live valuation roll with mass appraisal (comparable sales + land-use multipliers) eliminates manual spreadsheets and reduces revenue leakage by 40–60% — value realised by councils in Kenya's devolution era.
**Implementation**: `_valuations` store per parcel, `_valuation_runs` for batch campaigns. `compute_valuation` applies location-adjusted market-rate tables (price/ha by county × land-use class) with recency decay. `generate_valuation_roll` aggregates county-wide. Rate bills derived from assessed value × levy rate.
**Competitor**: Tyler Technologies MUNIS Land Records, Patriot Properties

### I4. Spatial Overlap & Boundary Conflict Detection
**Category**: Feature
**Justification**: Overlapping title polygons are the single largest source of land disputes in sub-Saharan Africa. Automated detection at capture time — before the human reviews — eliminates a class of errors that take years and thousands of dollars to unwind.
**Implementation**: On `capture_boundary`, run a Sutherland-Hodgman polygon intersection test against all existing boundaries in the same county. Return `overlap_alerts: [{ boundary_id, parcel_id, overlap_ha }]` in the response. Flag parcel status `potential_overlap` for review. Uses bounding-box pre-filter for O(log n) performance.
**Competitor**: Trimble Spectra Geospatial, Esri Parcel Fabric

### I5. Offline-First Mobile Sync Protocol
**Category**: UX
**Justification**: Field surveyors in rural Kenya operate with intermittent connectivity. GPS waypoints captured offline must queue and sync deterministically without duplicates or ordering conflicts. Competitors who nail this own the field-ops market.
**Implementation**: `submit_offline_batch(payload)` accepts an array of boundary captures with client-generated idempotency keys, timestamp of capture, and device fingerprint. Server deduplicates via idempotency key index, replays in capture-time order, returns per-item status. Conflicts (same parcel, overlapping times) quarantined to `_offline_conflicts` for review.
**Competitor**: Survey123 for ArcGIS, Fulcrum

### I6. Title Certificate PDF Generation
**Category**: Feature
**Justification**: Farmers need a physical or printable title deed they can take to a bank. Generating a compliant PDF (with parcel map sketch, QR verification code, official signatures) from the digital record closes the gap between digital and paper administration — primary ask from land offices in 6 counties.
**Implementation**: `generate_title_certificate(title_id)` builds a structured document dict: header, parcel details, boundary sketch (SVG polygon from waypoints), owner details, encumbrances, QR code payload (title_id + parcel_number + tenant_id), issuer block. Returns base64-encoded PDF bytes + SHA-256 fingerprint stored on the title record.
**Competitor**: Clio (legal docs), DocuSign CLM

### I7. AI-Powered Parcel Description Extraction (OCR + NLP)
**Category**: AI/ML
**Justification**: Legacy paper deeds contain parcel descriptions in narrative text ("...bounded to the north by the Naivasha road, thence 200 links east..."). Auto-extraction of structured coordinates and acreage from scanned documents slashes digitisation cost by 80%.
**Implementation**: `ingest_legacy_deed(payload)` accepts base64-encoded scan. Calls local Ollama `llava` model for OCR of deed image, then `mistral` for entity extraction (parcel_number, area, tenure_type, owner_name, boundary_description). Returns structured `extracted_parcel` dict with confidence scores. Human review required before `create_parcel` commit. Full Ollama HTTP integration, no cloud APIs.
**Competitor**: Kofax TotalAgility, ABBYY FlexiCapture

### I8. Land Use Change Detection via Satellite Imagery
**Category**: AI/ML
**Justification**: Unauthorised land-use changes (e.g. converting gazetted forest to farmland) are enforceable only if detected. Automated NDVI change alerts integrated with the parcel registry give enforcement agencies actionable intelligence on 100% of parcels, not the 2% that get field-inspected.
**Implementation**: `run_land_use_change_scan(parcel_ids, before_date, after_date)` fetches Sentinel-2 NDVI tiles (or mock tiles) for parcel bounding boxes, computes change magnitude, stores `_landuse_alerts` records with `change_score`, `before_ndvi`, `after_ndvi`. Alerts above threshold trigger `landuse.change_detected` audit event. Ollama `llava` can classify tile pairs.
**Competitor**: Regrow, Satellogic, Planet Labs Monitoring

### I9. Tenure Formalisation Workflow (Customary → Statutory)
**Category**: Compliance
**Justification**: Over 60% of African smallholders hold land under customary tenure with no formal title. A structured workflow guiding: community consent → adjudication → survey → title issuance removes administrative ambiguity and is required by the Kenya Community Land Act 2016.
**Implementation**: `initiate_formalisation(parcel_id, community_id, workflow_type)` creates a `_formalisation` record with stages: `community_consent → demarcation → survey → adjudication → registration → title_issued`. Each stage gates on the previous, records officers, dates, and required documents (list of attachment references). Progress queryable via `get_formalisation_status`.
**Competitor**: GLTN (UN-Habitat), Cadasta Platform

### I10. Parcel Subdivision & Amalgamation
**Category**: Feature
**Justification**: Smallholder inheritance routinely splits parcels; consolidation projects merge them. Without a formal subdivision/amalgamation API, registrars manually cancel and re-create records — losing audit history. Courts require the traceable lineage.
**Implementation**: `subdivide_parcel(parent_id, children_payloads)` validates child areas sum ≤ parent area, cancels parent parcel (status `cancelled`, `superseded_by: [child_ids]`), creates children with `parent_id` back-reference, retains boundary history. `amalgamate_parcels(source_ids, target_payload)` reverse: cancel sources, create merged parcel with `merged_from` list.
**Competitor**: Esri Parcel Fabric, Trimble AgriData

### I11. Multi-Signature Approval Chains
**Category**: Security
**Justification**: High-value transfers (>KES 10M) require multiple registrar sign-offs under Kenya's Land Act. Hardcoding single-approver workflows creates fraud vectors. Multi-sig with role-based quorum (e.g. county registrar + national lands) eliminates single points of compromise.
**Implementation**: `_approval_chains` config per tenant: `{ threshold_value, required_roles: [role1, role2], quorum: 2 }`. `sign_transfer(transfer_id, approver_id, role, signature_hash)` appends to `transfer.signatures`. Status advances to `approved` only when quorum reached. Each signature includes timestamp + role + approver identity. Tampering with any signature invalidates the chain (SHA-256 of prior state included).
**Competitor**: DocuSign, Gallagher Bassett (insurance multi-party), Yardi Voyager

### I12. Land Tax & Rates Billing Integration
**Category**: Integration
**Justification**: County finance systems (IFMIS, Oracle Financials) need parcel-level land-rate bills generated from the valuation roll. Direct integration eliminates the parallel spreadsheet maintained by every county treasurer — a major source of revenue leakage and corruption.
**Implementation**: `generate_rate_bill(parcel_id, financial_year)` computes: `assessed_value × levy_rate_pct`. Stores `_rate_bills` with `status: draft|issued|paid|overdue`. `bulk_generate_bills(county, financial_year)` batches for all county parcels. Integration stub `push_bills_to_ifmis(bills)` serialises to IFMIS-compatible JSON envelope. Arrears tracked with compound interest per Kenya Revenue Authority rules.
**Competitor**: Tyler Technologies MUNIS, Infor CloudSuite Public Sector

### I13. Geospatial Search by Coordinates
**Category**: Feature
**Justification**: "Which parcels fall within 500m of this borehole?" is an everyday query for planners, banks, and NGOs. A spatial point-in-polygon and radius search replaces GIS specialists doing ad-hoc queries in QGIS — empowering non-technical users.
**Implementation**: `search_parcels_by_location(lat, lng, radius_m)` iterates all parcels with boundaries, computes centroid distance using Haversine formula, returns matches within `radius_m` sorted by distance. `find_parcel_at_point(lat, lng)` does point-in-polygon test (ray-casting algorithm) against all stored boundary waypoints. Returns parcel ID, owner, title status.
**Competitor**: HERE Maps Land Registry API, Google Maps Places API (parcel overlays)

### I14. Ownership History & Chain of Title
**Category**: Compliance
**Justification**: Banks and lawyers performing due diligence need the complete chain of title — every owner, transfer date, price, encumbrance — going back to original registration. Without this, every conveyancing search is a manual archive dig taking days.
**Implementation**: `get_chain_of_title(parcel_id)` reconstructs ownership lineage by walking `_transfers` for the parcel in chronological order, stitching: `[{ owner_id, owner_name, tenure_from, tenure_to, acquisition_type, transfer_value, title_number }]`. Includes encumbrance overlaps per ownership period. Returns a linked-list-style provenance chain. `verify_title_authenticity(title_id)` cross-checks title against chain for consistency.
**Competitor**: First American Title (DataTree), Verisite

### I15. Webhook & Event Streaming for Third-Party Integration
**Category**: Integration
**Justification**: County land offices, NSSF, KRA, banks, and NGOs each need real-time parcel events (transfer registered, title issued, dispute filed). A webhook registry eliminates polling and enables event-driven downstream workflows — the architecture of every modern SaaS platform.
**Implementation**: `_webhooks` store: `{ url, events: [event_types], secret, active }`. `register_webhook`, `list_webhooks`, `delete_webhook` management APIs. `_dispatch_webhooks(event_type, payload)` called from `_emit`: signs payload with HMAC-SHA-256 using the webhook secret (X-Signature header), POSTs to registered URLs matching the event type. Retry logic: 3 attempts with exponential backoff. Failed deliveries logged to `_webhook_failures`.
**Competitor**: Twilio, Stripe Webhooks, Salesforce Platform Events
