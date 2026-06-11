# Electronic Medical Records — World-Class Improvements

### I1. Real-Time Clinical Event Streaming via NATS
**Category**: Integration / Streaming
**Justification**: Current implementation writes to in-memory audit list only; downstream systems (analytics, pharmacy, billing) cannot react to chart events without polling. A streaming backbone multiplies the utility of every write operation by an order of magnitude.
**Implementation**: Publish structured CloudEvents to a NATS subject hierarchy (`emr.{tenant_id}.{resource_type}.{event_type}`) on every mutating service method. Bytewax pipelines can consume these for real-time dashboards, risk scoring, and HL7 v2/v3 relay. Use `nats.py` async client; inject a `NATSAdapter` through the existing adapter pattern so the null adapter is used in tests.
**Competitor**: Epic Systems uses HL7 messaging buses internally; Cerner uses Kafka-backed event streams. NATS provides sub-millisecond latency with JetStream persistence, outperforming both for low-latency alert delivery.

---

### I2. Longitudinal Vital Trend Analysis with Statistical Anomaly Detection
**Category**: Clinical Decision Support
**Justification**: Recording vitals without trend analysis loses >80% of the clinical signal. A single blood pressure reading is noise; three readings over 30 minutes with a rising mean and widening pulse pressure is a haemodynamic deterioration pattern. Automated trend detection catches deterioration 60–90 minutes earlier than manual observation per NICE EWS research.
**Implementation**: Add `analyse_vital_trend()` method that retrieves a time-windowed vital series, computes rolling mean/stddev, fits a linear regression to detect directional trend, and fires a NATS alert if slope exceeds configurable thresholds or if any single reading is >3 SD from the patient's own baseline. Store per-patient baseline statistics keyed to `(tenant_id, patient_id, vital_type)`.
**Competitor**: Philips IntelliVue trend alarming; Epic Deterioration Index (EDI) uses vital trend slopes as primary input.

---

### I3. SMART on FHIR OAuth2 App Launch Integration
**Category**: Interoperability / Security
**Justification**: The FHIR export is currently a raw bundle with no authorisation context. SMART on FHIR is the industry standard (used by all US EHRs under ONC 21st Century Cures) for authorised third-party app access. Without it, the FHIR layer is unsuitable for real-world integration.
**Implementation**: Add `generate_smart_launch_context()` method that issues a signed JWT containing `patient`, `encounter`, `tenant` claims conforming to SMART v2 spec. Add `validate_smart_token()` for inbound verification. Integrate with the existing `_NullAuthAdapter` pattern so production deployments wire a real JWKS-backed validator.
**Competitor**: Epic App Orchard, Cerner Code, and all major EHRs use SMART on FHIR as the mandatory integration path for third-party clinical applications.

---

### I4. AI-Assisted Clinical Note Structuring via Local LLM (Ollama)
**Category**: AI / Clinical Productivity
**Justification**: Clinicians spend 35–55% of their time on documentation. Automated extraction of SOAP components from free-text dictation reduces cognitive load and improves note completeness. Local Ollama deployment avoids PHI leaving the facility boundary.
**Implementation**: Add `ai_structure_note()` async method that sends free-text note content to a locally-hosted Ollama model (e.g. `llama3:8b` or `medllama2`) via HTTP. Prompt instructs the model to extract Subjective/Objective/Assessment/Plan fields, suggested ICD-10 codes, and medication mentions. Results populate the `ClinicalNoteCreate` payload fields. Fallback gracefully when Ollama is unavailable.
**Competitor**: Nuance DAX (Dragon Ambient eXperience) and Suki AI provide cloud-hosted equivalents. Local LLM approach provides equivalent functionality with zero PHI egress.

---

### I5. Predictive Readmission Risk Scoring (LACE+ Index)
**Category**: Population Health / Risk Stratification
**Justification**: 30-day hospital readmissions cost health systems $26B annually in the US. The LACE+ index (Length of stay, Acuity, Charlson comorbidity index, Emergency department visits) has AUC 0.72 for predicting 30-day readmission and is fully computable from EMR data. Early identification enables targeted discharge planning.
**Implementation**: Add `compute_lace_plus_score()` method that aggregates: encounter LOS, admission acuity from encounter type, Charlson Comorbidity Index from problem list ICD-10 codes, and ED visit count from encounter history. Returns risk tier (low/moderate/high/very-high), score, contributing factors, and recommended interventions. Emit score to NATS for population health dashboards.
**Competitor**: Epic's Readmission Risk model (proprietary ML); Cerner HealtheIntent uses similar index-based approaches. LACE+ is open, clinically validated, and requires no ML infrastructure.

---

### I6. Charlson Comorbidity Index (CCI) Calculation
**Category**: Clinical Decision Support / Risk Adjustment
**Justification**: CCI is used for risk-adjusted outcome measurement, bundled payment calculations, and research cohort stratification. Every ICD-10 code on the problem list contributes to this score. Without automated CCI computation, clinicians and administrators manually calculate it — a reliable error source.
**Implementation**: Add `compute_charlson_comorbidity_index()` method. Maintain a mapping of ICD-10 prefixes to CCI weights (MI → 1, CHF → 1, PVD → 1, CVD → 1, dementia → 1, COPD → 1, CLD → 1, diabetes → 1, diabetes+complications → 2, hemiplegia → 2, CKD moderate → 2, CKD severe → 2, solid tumour → 2, leukaemia → 2, lymphoma → 2, liver disease moderate-severe → 3, metastatic tumour → 6, AIDS → 6). Age-adjust for patients ≥50 years.
**Competitor**: Used natively in Epic Clarity for outcomes reporting; IBM Watson Health used CCI for risk stratification. The implementation here makes it a first-class real-time API.

---

### I7. Structured Medication Administration Record (eMAR) with Dose Tracking
**Category**: Patient Safety / Medication Management
**Justification**: The current MAR is derived from prescriptions, not actual administration events. A true eMAR records each individual dose given, by whom, at what time, and any variance (dose refused, held, modified). This is the primary tool for preventing medication errors in inpatient settings.
**Implementation**: Add `record_dose_administration()` method that creates a `DoseAdministrationEvent` linked to prescription, encounter, and patient. Fields: drug, dose, route, administered_by, administered_at, variance_code (none/held/refused/partial/modified), variance_reason. Add `get_emar_report()` to generate a tabular view of due/given/overdue doses for a time window. Store in `_dose_administrations` dict keyed `(tenant_id, event_id)`.
**Competitor**: Pyxis MedStation (BD), Omnicell, and all inpatient EHRs treat eMAR as foundational safety infrastructure. Five Rights verification (right patient, drug, dose, route, time) is enforced at administration event creation.

---

### I8. Sepsis Bundle Compliance Tracking (Sepsis-6/Hour-1)
**Category**: Quality / Patient Safety
**Justification**: Sepsis kills 11 million people annually. Evidence-based Sepsis-6 bundle (blood cultures, lactate, IV antibiotics, IV fluids, oxygen, urine output measurement) within 1 hour of recognition reduces mortality by 25%. Automated bundle tracking ensures compliance and supports mandatory quality reporting.
**Implementation**: Add `track_sepsis_bundle()` method. When a patient has qSOFA ≥ 2 or NEWS2 ≥ 7, create a `SepsisBundleTracker` record with timestamp of recognition. Track completion of each bundle element by checking lab orders (blood culture, lactate), prescriptions (antibiotic, IV fluid), and vitals. Compute bundle compliance percentage. Alert via NATS if any element is incomplete at 45-minute mark.
**Competitor**: Epic Sepsis Management module; Cerner CareAware iBus sepsis alerting. Sepsis is a CMS quality measure (SEP-1) required for hospital accreditation.

---

### I9. Genomics/Pharmacogenomics Integration Layer
**Category**: Precision Medicine
**Justification**: ~25% of patients are poor or ultra-rapid metabolisers of CYP2C19/CYP2D6 substrates (SSRIs, clopidogrel, opioids, antipsychotics). Prescribing standard doses to poor metabolisers can cause toxicity; to ultra-rapid metabolisers, therapeutic failure. Pharmacogenomics (PGx) testing is becoming standard of care for psychiatric and cardiology patients.
**Implementation**: Add `record_pgx_result()` method to store pharmacogenomics test results (gene, diplotype, phenotype, e.g. `CYP2C19 *1/*17 → rapid metaboliser`). Add `check_pgx_prescribing()` that cross-references the drug against a PGx interaction table and returns dose adjustment recommendations per CPIC guidelines. Store results in `_pgx_results` dict; expose via FHIR MolecularSequence resource in bundle export.
**Competitor**: OneOme RightMed, Genomind, and health systems like Vanderbilt University Medical Center (PREDICT program) have deployed PGx at scale. Epic stores PGx results in a native Gene-Drug Interaction module.

---

### I10. Patient-Generated Health Data (PGHD) Ingestion
**Category**: Connected Health / Patient Engagement
**Justification**: Wearables (CGMs, BP monitors, pulse oximeters, ECG patches) generate longitudinal data that is clinically superior to isolated in-clinic measurements. Integration of PGHD enables continuous monitoring between visits, catching deterioration in outpatients. AHA/ACC guidelines now recommend ambulatory BP monitoring over office measurements for hypertension diagnosis.
**Implementation**: Add `ingest_patient_generated_data()` method accepting a list of observation dicts in FHIR Observation format. Validate device type, timestamp, and value ranges. Map to `VitalSignCreate` payloads and bulk-insert. Tag with `source: patient_device` and `device_id`. Add `get_pghd_summary()` to aggregate device-generated readings per period. Support NATS-based push from device gateways.
**Competitor**: Apple Health Records, CommonHealth, and Google Fit all support FHIR export. Epic MyChart ingests Apple Health data. This feature closes the loop from outpatient device to clinical record.

---

### I11. Audit Trail Immutability with Cryptographic Hash Chaining
**Category**: Compliance / Security
**Justification**: A mutable audit log is legally worthless. HIPAA requires that audit trails cannot be retroactively modified. Hash chaining (each audit record includes SHA-256 of the previous record) makes tampering detectable — any modification breaks the chain. This is a legal and accreditation requirement in most jurisdictions.
**Implementation**: Modify `_record_audit()` to compute `sha256(previous_hash + current_record_json)` and store it on each audit event. Add `verify_audit_chain()` method that walks the tenant's audit events in sequence, recomputes each hash, and returns a verification report with any detected breaks. Persist the current chain head in `_audit_chain_head` per tenant. In production, use PostgreSQL `BYTEA` column with append-only row-level security policy.
**Competitor**: Hyperledger Fabric-based audit trails (used in some health data exchanges); Epic's Chronicles audit journal (proprietary immutability mechanism). Hash chaining provides a practical, verifiable substitute without blockchain overhead.

---

### I12. Advance Directive and POLST/MOLST Management
**Category**: Patient Safety / End-of-Life Care
**Justification**: Advance directives (living wills, healthcare proxies) and POLST/MOLST forms govern end-of-life care decisions. Absence of documented preferences leads to unwanted resuscitation attempts and legal disputes. Joint Commission requires accessible documentation of patient wishes. 45% of ICU deaths involve patients without documented advance directives.
**Implementation**: Add `record_advance_directive()` method storing: directive_type (living_will/dnr/dnar/polst/molst/healthcare_proxy), provisions, agent_name, agent_contact, signed_date, witness_ids, document_url. Add `get_active_directives()` to surface directives at encounter creation. Integrate into discharge checklist via `clinical_reminder_check`. Flag at top of clinical summary and FHIR export as `Consent` resources.
**Competitor**: Epic stores advance directives in a dedicated module with alert banners in clinical documentation. CMS requires documentation of advance care planning discussions (CPT 99497/99498) for Medicare billing.

---

### I13. Outpatient Population Health Cohort Builder
**Category**: Population Health / Analytics
**Justification**: Reactive care is 3–5x more expensive than preventive management. Cohort-based outreach (identifying all diabetic patients with HbA1c > 9 who haven't been seen in 6 months) is the primary mechanism for closing care gaps at scale. Without a cohort API, population health initiatives require bespoke SQL queries against raw tables.
**Implementation**: Add `build_patient_cohort()` method accepting filter criteria: `diagnosis_prefixes` (ICD-10), `active_medications`, `age_range`, `last_encounter_before`, `missing_labs`, `missing_screenings`. Returns list of `PatientCohortMember` dicts with patient_id, last_encounter_date, open_care_gaps. Emit cohort snapshots to NATS for downstream analytics. Cache cohort definitions for scheduled refresh.
**Competitor**: Epic Reporting Workbench; Arcadia Data; Azara Healthcare. These tools charge $50–200K/year for population health analytics on top of the base EHR license.

---

### I14. Multi-Factor Diagnostic Confidence Scoring
**Category**: Clinical Decision Support / AI
**Justification**: Current `suggest_diagnoses()` uses keyword matching with three confidence tiers. Real diagnostic decision support must weight: symptom-diagnosis likelihood ratios, lab result patterns, vital abnormalities, demographics, and medication history. Bayesian probability revision as new data arrives increases diagnostic accuracy and reduces unnecessary testing.
**Implementation**: Refactor `suggest_diagnoses()` to accept an optional `patient_id` argument. When provided, retrieve problem list, recent labs, vitals, age/gender, and active medications, then apply a weighted scoring model: keyword match (base score) × demographic likelihood factor × lab signal multiplier × vital signal multiplier. Sort by posterior probability. Output includes `pre_test_probability`, `posterior_probability`, `likelihood_ratio`, and `supporting_evidence` list per suggestion. Fall back to keyword-only when patient context is absent.
**Competitor**: Isabel DDx, DXplain, Diagnosis Pro. All commercial DDx tools use multi-factor probabilistic models. The open implementation here enables equivalent functionality without third-party data egress.

---

### I15. Cross-Encounter Longitudinal Disease Trajectory Tracking
**Category**: Chronic Disease Management
**Justification**: For chronic conditions (diabetes, heart failure, CKD, hypertension), what matters clinically is the trajectory of control metrics over time — not any single encounter snapshot. A patient whose HbA1c went 11.2 → 9.4 → 7.8 over 18 months is doing well; one going 7.8 → 9.4 → 11.2 needs urgent intervention. Without trajectory data, each encounter is evaluated in isolation.
**Implementation**: Add `get_disease_trajectory()` method accepting `patient_id` and `icd10_prefix`. Collects all lab results matching the condition's key biomarkers (e.g. E11 → HbA1c, eGFR, urine albumin; I10 → systolic BP; I50 → BNP, ejection fraction), vital trends, and medication changes over time. Returns a sorted timeline of `TrajectoryPoint` objects with value, timestamp, contextual note excerpt, and computed trend direction (improving/stable/worsening) using linear regression on the last 3–5 data points.
**Competitor**: Epic Care Everywhere trend view; Cerner PowerChart longitudinal viewer. Both are proprietary implementations of the same concept; this open implementation integrates trajectory data into the FHIR export as DiagnosticReport series.
