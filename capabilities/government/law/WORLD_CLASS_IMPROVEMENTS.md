# Law Enforcement Capability — World-Class Improvements

**Capability**: `government_law`  
**Domain**: Government  
**Author**: Nyimbi Odero — Datacraft © 2025  

---

## 1. Async Service Layer

All synchronous methods should be converted to `async def`. The service currently blocks on in-memory stores; the next evolution is async I/O to PostgreSQL (via `asyncpg`) and the event bus (via `aiokafka`/`bytewax`). Converting now avoids a disruptive migration later and unlocks concurrent case processing under real load.

**Impact**: Throughput, correctness under concurrent requests, testability.

---

## 2. PostgreSQL-Backed Persistence

Replace the in-memory `dict` stores with an async `asyncpg` connection pool. All writes go through explicit SQL transactions — evidence intake, custody actions, and prosecution hand-overs are wrapped in `SERIALIZABLE` transactions so no two concurrent operations can corrupt chain-of-custody lineage.

**Impact**: Durability, crash-safety, multi-process deployment.

---

## 3. Cryptographic Evidence Integrity

Evidence items currently carry a `hash(description + chain_of_custody)` Python hash — not cryptographically safe. Replace with SHA-256 (or BLAKE3 for throughput) over a canonical JSON blob that includes `(evidence_id, tenant_id, description, received_by, received_at)`. Store the hex digest and re-verify on every custody action. Any mismatch aborts the transfer and raises an immutable tamper alert.

**Impact**: Legal admissibility, tamper-detection, audit defensibility.

---

## 4. Immutable Append-Only Audit Trail via Event Sourcing

The current `audit_events` list is mutable in-memory state. Real law enforcement systems require an immutable ledger. Replace with an append-only `audit_ledger` table (PostgreSQL `INSERT`-only, no `UPDATE`/`DELETE` grants on that table for the app role) streaming over bytewax. Each event carries a `prev_hash` linking to the prior event — a lightweight Merkle chain over each tenant's audit stream.

**Impact**: Legal compliance, governance, non-repudiation.

---

## 5. Structured Domain Events with CloudEvents Envelope

Audit calls currently push bare dicts. Wrap every domain event in a [CloudEvents](https://cloudevents.io) envelope (`specversion`, `source`, `type`, `id`, `time`, `datacontenttype`, `data`). This makes the bytewax stream consumable by any CE-compliant subscriber (Kafka, NATS, HTTP webhook, GCP Pub/Sub) without custom deserialization glue.

**Impact**: Composability, federation, reduced integration surface.

---

## 6. Chain-of-Custody Graph — Directed Acyclic Graph Model

The current `custody_actions` store is a flat list. Model the chain as a DAG: each `CustodyAction` node has a `parent_action_id` foreign key to the previous action on that evidence item. Traversal becomes `SELECT … WITH RECURSIVE`, and integrity checks (is the chain unbroken? does it terminate at seizure?) become single SQL queries rather than application-level list scans.

**Impact**: Query performance, integrity proof generation, court-ready reports.

---

## 7. CIMS Integration Adapter

Add a `CIMSAdapter` that translates internal `IncidentReport` / `CaseDocket` models to and from the CIMS (Criminal Information Management System) XML/REST format used by Kenya Police and the DCI. The adapter exposes `async push_to_cims(incident_id)` and `async pull_from_cims(cims_ref)`, with a configurable `base_url` and mutual-TLS client certificate. Authentication uses a rotating token refreshed every 55 minutes.

**Impact**: Interoperability with national systems, mandatory for production deployment.

---

## 8. Automated Docket SLA Monitoring

Cases sit open indefinitely without consequence. Add an `async check_sla_violations()` method that scans all open dockets, computes `age_days = (now - opened_date)`, compares against per-incident-type SLA thresholds (configurable per tenant), and emits a `docket_sla_breach` domain event for any docket past its threshold. The event triggers an `ntfy` notification to the supervisor and a `moni` gauge increment.

**Impact**: Accountability, case clearance rate improvement, oversight compliance.

---

## 9. Geospatial Crime Hotspot Analysis

`crime_map_query` returns a crude set of `location_reference` strings. Replace with a PostGIS-backed query: store `location_point GEOGRAPHY(POINT, 4326)` on `IncidentReport`, then run a `ST_ClusterDBSCAN` query to return genuine geographic clusters with centroid, radius, and constituent incident IDs. Expose as a GeoJSON `FeatureCollection` for direct consumption by Leaflet/MapLibre.

**Impact**: Operational policing intelligence, patrol resource allocation.

---

## 10. Digital Evidence Hash Verification on Transfer

Every `record_custody_action` call that transfers evidence to a new custodian (action type `transferred` or `court_submitted`) should re-hash the evidence descriptor and compare with the stored hash from intake. If the hash differs, the transfer is blocked and a `evidence_integrity_violation` event is emitted — not just an exception. This gives the CIMS integration a discrete signal it can act on independently.

**Impact**: Legal chain-of-custody integrity, tamper-detection feedback loop.

---

## 11. Warrant Lifecycle State Machine

`warrant_issue` creates a warrant but there is no lifecycle beyond `active`. Add a `WarrantRecord` dataclass with statuses `issued → served → expired | cancelled` and an `async update_warrant_status()` method. Warrants past their `valid_until` date (configurable, default 14 days) are auto-transitioned to `expired` by the SLA monitor. Served warrants automatically link back to an `arrest_record`.

**Impact**: Legal completeness, prevents executing stale warrants, audit trail closure.

---

## 12. Victim / Complainant Case Portal (Read-Only Scoped Token)

Issue short-lived (24 h) read-only JWT tokens scoped to a single `ob_number`. The token allows a complainant to query incident status, assigned officer name (not ID), and next hearing date — nothing more. The service exposes `async generate_complainant_token(ob_number)` returning a signed JWT and `async get_case_status_for_complainant(token)`. This fulfills Article 50(2)(k) of the Kenyan Constitution (access to justice).

**Impact**: Public trust, constitutional compliance, reduced counter inquiry load.

---

## 13. Bulk Evidence Import via Structured Manifest

Field units collect many items simultaneously. Add `async bulk_evidence_import(manifest: list[dict])` that processes up to 100 items in a single database transaction, assigns exhibit numbers sequentially (`EXH-<docket>-001` … `EXH-<docket>-100`), and returns a manifest receipt with per-item status. Invalid items are rejected with structured errors; valid items are committed atomically. Integrates with the forensic lab LIMS via an adapter.

**Impact**: Operational efficiency, reduces data-entry errors in field conditions.

---

## 14. ML-Assisted Incident Classification

Feed incoming `incident_report` description text to a locally hosted Ollama model (e.g., `mistral:7b-instruct`) to suggest `incident_type` when the officer leaves it blank or selects `other`. The suggestion is surfaced in the UI as a non-binding hint; the officer confirms or overrides. The method `async classify_incident(description)` returns `{"suggested_type": str, "confidence": float, "model": str}`. All inference is local — no data leaves the deployment boundary.

**Impact**: Data quality, reduces misclassification that distorts crime statistics.

---

## 15. Redaction and Right-to-Be-Forgotten Workflow

Add `async redact_personal_data(record_type, record_id, fields)` that replaces specified PII fields with `[REDACTED-<timestamp>]` and writes a `data_redaction_event` to the audit ledger. The original values are archived in a separately encrypted column (AES-256-GCM, key stored in a tenant-specific KMS slot) accessible only under a court order workflow. This satisfies the Kenya Data Protection Act 2019 §§ 38–40 and GDPR Art. 17 for cross-border deployments.

**Impact**: Legal compliance (DPA 2019, GDPR), reduces liability exposure, enables regional deployments.
