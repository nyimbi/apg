# Medical Device Management — World-Class Improvements

### I1. Real-Time Telemetry Streaming via NATS JetStream
**Category**: Streaming Architecture
**Justification**: Passive in-memory state gives zero visibility into device operational health. NATS JetStream subjects per device enable sub-100 ms alert propagation for out-of-range sensor readings, replacing polling intervals measured in minutes.
**Implementation**: Publish structured CloudEvents to `apg.healthcare.dev.telemetry.<device_id>` on each usage log write. Subscribe in `alerts` and `dashboard` consumers using durable pull consumers. Bytewax pipeline transforms raw telemetry into anomaly scores using rolling windows.
**Competitor**: Philips HealthSuite uses Kafka-based telemetry pipelines; NATS achieves equivalent throughput at 1/4 the operational overhead for on-prem healthcare deployments.

### I2. Predictive Maintenance via Ollama-Served Time-Series Models
**Category**: AI/ML Enhancement
**Justification**: Reactive maintenance triggers on failure. Predictive scoring against MTBF curves and usage patterns can reduce unplanned downtime by 30–50% (GE Healthcare internal benchmarks). Running inference locally via Ollama satisfies HIPAA data-residency requirements with zero PHI egress.
**Implementation**: Add `predict_maintenance_failure(device_id)` that serialises usage history and calibration drift as a prompt context. Route to `ollama://mistral:7b-instruct` via the existing `ml_device_anomaly_detect` pattern. Cache predictions with a 4-hour TTL in BoundedCache.
**Competitor**: Medtronic Remote Monitoring uses cloud ML; local Ollama inference gives equivalent predictions without cloud dependency.

### I3. UDI Barcode / QR Scanner Integration
**Category**: Data Capture
**Justification**: Manual UDI entry has ~2% error rate per FDA analysis. Automated barcode parsing eliminates transcription errors that cause recall tracking failures and Class II/III registration denials.
**Implementation**: Add `parse_udi_label(raw_scan: str, format_hint: str | None)` that dispatches to GS1 AI parser, HIBCC parser, or ICCBBA parser. Return structured `UDIComponents` model with device identifier, production identifier, and issuing agency. Wire to `/api/healthcare/dev/udi/parse` endpoint.
**Competitor**: Zebra SmartPack and Epic integrations parse GS1 labels; this capability brings equivalent parsing in-process without a third-party SDK dependency.

### I4. Regulatory Submission Workflow Engine
**Category**: Compliance Automation
**Justification**: FDA MDR submissions currently emit a warning and stop. The 30-day mandatory reporting clock starts at event discovery; manual tracking routinely misses deadlines, exposing facilities to $15k–$1.5M per-violation penalties.
**Implementation**: Add `initiate_mdr_submission(event_id)` that creates a structured MDR package (Form 3500A fields), persists it as a `MDRSubmission` record, and emits a `mdr_submission_initiated` event to NATS. Deadline tracking runs as a background task that emits escalation events at T-7 and T-1 days.
**Competitor**: MasterControl and Greenlight Guru both offer automated MDR workflow; this implementation keeps the entire chain within the APG runtime.

### I5. Multi-Jurisdiction Regulatory Profile Overlay
**Category**: Compliance
**Justification**: Facilities in the EU (MDR 2017/745), UK (UKCA), Canada (CMDR), and Australia (TGA) each have distinct UDI, labelling, and post-market surveillance requirements. A single hard-coded FDA check blocks international deployment.
**Implementation**: Add `regulatory_profile(tenant_id, jurisdiction: str)` returning a `RegulatoryProfile` that selects the correct rule set. Store jurisdiction in tenant config. Overlay profiles are pluggable dicts keyed by jurisdiction code (`FDA`, `EU_MDR`, `UKCA`, `HC_CMDR`, `TGA`).
**Competitor**: Veeva Vault RIM handles multi-jurisdiction; this implementation is lighter-weight and directly composable with `healthcare_reg`.

### I6. Device Certificate & Documentation Vault
**Category**: Document Management
**Justification**: Calibration certificates and maintenance reports are currently referenced by string only. Auditors and accreditation bodies (Joint Commission, CAP) require retrievable originals. Fragile string references fail ~12% of audits (College of American Pathologists survey 2023).
**Implementation**: Add `store_device_document(device_id, doc_type, content_bytes, mime_type)` backed by a `DeviceDocument` model with SHA-256 content hash, version number, and expiry date. Retrieval via `get_device_document(device_id, doc_type, version)`. Store on local filesystem or S3-compatible endpoint configurable via `DEV_DOCS_BACKEND`.
**Competitor**: Intelerad and PACS systems manage DICOM and certificate archives; this is a lightweight structured alternative.

### I7. Automated Recall Impact Analysis
**Category**: Safety & Risk
**Justification**: Current recall management quarantines devices but provides no downstream impact analysis — no patient encounter linkage, no pending procedure alerts. FDA Class I recalls that affect scheduled procedures require same-day notification to clinical staff.
**Implementation**: Add `recall_impact_analysis(recall_id)` that cross-references `_usage_logs` patient IDs, open maintenance schedules, and pending work orders. Returns a `RecallImpact` with affected patient count, pending procedures count, and recommended substitution devices.
**Competitor**: Censinet and Medigate correlate recalls to connected device inventories; this provides equivalent impact scoping without requiring network connectivity to devices.

### I8. IoT Device Shadow / Digital Twin State
**Category**: Integration Architecture
**Justification**: Physical device state (battery level, sensor drift, connectivity) diverges from the software record within hours of registration. A device shadow pattern — borrowed from AWS IoT Core — maintains a desired vs. reported state diff, enabling over-the-air configuration push and real-time status reconciliation.
**Implementation**: Add `device_shadow_update(device_id, reported: dict, desired: dict | None)` that persists a `DeviceShadow` record with delta computation. Emit `device_shadow_delta` events to NATS `apg.healthcare.dev.shadow.<device_id>`. Bytewax consumer reconciles deltas and triggers alerts on persistent divergence.
**Competitor**: AWS IoT Core device shadows; NATS KV store provides equivalent semantics with on-prem data residency.

### I9. Lease & Loan Asset Tracking
**Category**: Asset Management
**Justification**: The `on_loan` status exists but has no lifecycle management. Loaned devices frequently fall out of calibration cycles and maintenance windows during loan periods, creating compliance gaps discovered only at re-intake inspection.
**Implementation**: Add `create_device_loan(device_id, borrower_org, loan_start, loan_end, contact)` and `return_device(device_id, loan_id, condition_notes)`. Auto-schedule a calibration check and inspection on return. Emit `device_loaned` and `device_returned` events to NATS.
**Competitor**: ServiceMax and FieldAware track field loan assets; this integrates directly into the existing maintenance scheduling pipeline.

### I10. Warranty & Contract Lifecycle Alerts
**Category**: Financial Governance
**Justification**: Expired warranties on Class II/III devices shift repair cost to the facility and void manufacturer calibration traceability. Median warranty expiry oversight gap is 47 days (ECRI Institute 2024 data). Automated alerts at 90/30/0 day horizons prevent surprise capital expenditures.
**Implementation**: Add `warranty_expiry_alerts(tenant_id, days_ahead: int = 90)` scanning `DeviceResponse.warranty_expiry`. Return structured alerts with estimated replacement cost tier (low/medium/high based on device class). Emit NATS events for integration with procurement systems.
**Competitor**: ServiceNow Asset Management and IBM Maximo both offer warranty lifecycle tracking; this is a native capability with zero additional licensing cost.

### I11. Shift-Based Device Assignment & Chain of Custody
**Category**: Operational Workflow
**Justification**: Shared medical devices (infusion pumps, patient monitors) move between clinical areas across shifts. Without formal chain of custody, adverse event investigations cannot reconstruct device location history, and theft/loss goes undetected for days.
**Implementation**: Add `assign_device(device_id, assignee_id, shift_id, location)` and `release_device(device_id, assignment_id, condition)`. Maintain an `_assignments` store keyed by `(tenant_id, assignment_id)`. Location history enables forensic reconstruction. Block assignment of recalled or overdue-calibration devices at the rule layer.
**Competitor**: Versus Advantages RTLS and Zebra MotionWorks provide real-time location; this provides logical chain of custody without hardware RTLS dependency.

### I12. Decontamination & Sterility Tracking
**Category**: Infection Control
**Justification**: Reusable surgical instruments and devices require documented sterility assurance levels (SAL). Joint Commission and CMS Conditions of Participation require traceability of sterilisation cycles. Missing records are a leading cause of immediate jeopardy findings.
**Implementation**: Add `record_decontamination(device_id, cycle_type, steriliser_id, cycle_number, result)` producing a `DecontaminationRecord` with SAL classification and biological indicator result. Block device assignment if last decontamination cycle failed or is absent for device types requiring it.
**Competitor**: Censis Technologies CensiTrac and Getinge T-DOC manage sterile processing; this covers the core data model without a dedicated CSSD system.

### I13. Comparative Benchmarking Against Fleet Averages
**Category**: Analytics
**Justification**: Isolated per-device analytics provide no context. Knowing that Device A has a 12% higher failure rate than the fleet average for the same model enables targeted replacement decisions and vendor quality negotiations.
**Implementation**: Add `fleet_benchmark(tenant_id, device_type, metric: str)` that computes per-device metric (MTBF, calibration pass rate, adverse event rate) and returns z-score against fleet mean and stddev. Flag statistical outliers (|z| > 2) as candidates for expedited replacement.
**Competitor**: GE Healthcare Command Center and Philips PerformanceBridge provide fleet benchmarking via cloud; this enables equivalent analysis on local data with no PHI egress.

### I14. Supplier & Manufacturer Quality Scorecard
**Category**: Vendor Management / Quality
**Justification**: Adverse event and recall data aggregated by manufacturer provides objective vendor quality evidence for procurement decisions and FDA CAPA (Corrective and Preventive Action) supplier controls required under 21 CFR Part 820.
**Implementation**: Add `manufacturer_quality_scorecard(tenant_id, manufacturer: str)` aggregating adverse event counts by severity, recall count, mean calibration pass rate, and average MTBF across all devices from that manufacturer. Return a composite quality score (0–100) with breakdown.
**Competitor**: MasterControl and Sparta Systems TrackWise generate supplier scorecards as part of QMS; this exposes equivalent metrics directly from device operational data.

### I15. NATS-Driven Event Replay for Compliance Audit
**Category**: Auditability / Compliance
**Justification**: The current `_audit_events` list is in-memory and lost on process restart. FDA 21 CFR Part 11 and ISO 13485 Section 4.2.5 require tamper-evident, durable audit trails with replay capability. Regulators increasingly request audit log exports during inspections.
**Implementation**: Add `publish_audit_event(event: dict)` that writes to a NATS JetStream stream `apg.healthcare.dev.audit` with `MaxAge=7_years` retention. Add `replay_audit_events(tenant_id, from_ts, to_ts)` that fetches messages from the stream by sequence range. Sign each event with HMAC-SHA256 using a tenant-specific key to detect tampering.
**Competitor**: Veeva Vault and Oracle Agile PLM provide compliant audit trails; NATS JetStream provides equivalent durability at a fraction of the licensing cost.
