# PHI Classifier — 15 World-Class Improvements

## 1. Probabilistic Confidence Scoring (Multi-Signal Bayesian)

Current classification returns a binary `0.9` or `0.0` confidence. Replace with a Bayesian signal combiner that weighs field-name entropy, value-pattern match strength, surrounding context, and co-occurrence priors to emit a calibrated `[0.0, 1.0]` posterior. Enables threshold-gated redaction policies ("redact if confidence > 0.7") and downstream risk-scoring.

## 2. NLP-Backed Free-Text PHI Extraction (NER via Ollama)

`scan_document` currently matches 3 regex patterns. Route through a locally-hosted Ollama NER model (e.g. `llama3`/`phi3`) to detect person names, organisations, locations, dates, and medical jargon in running prose. Return span-level annotations with entity type, character offsets, and model confidence. Degrades gracefully to regex-only when Ollama is unreachable.

## 3. Streaming Redaction Pipeline

Add `redact_stream(records: AsyncIterator[dict]) -> AsyncIterator[dict]` to support high-throughput ELT pipelines. Internally batches records in configurable windows, applies concurrent classification, and yields redacted records with back-pressure via `asyncio.Queue`. Avoids loading entire datasets into memory.

## 4. Structured Audit Ledger (Append-Only Event Store)

Replace the no-op `log_phi_access` stub with a real append-only audit ledger backed by a PostgreSQL `phi_audit_events` table. Each event captures `accessor_id`, `tenant_id`, `record_id`, `operation`, `phi_fields_touched`, `purpose`, and an RFC 3339 timestamp. Expose a paginated `get_audit_events(cursor, limit)` API for HIPAA audit-log requirements.

## 5. Field-Level Encryption (Format-Preserving AES-FF3)

Add `encrypt_phi(record, key_ref)` that applies AES-FF3-1 format-preserving encryption to each PHI field value so the field structure (length, character class) is preserved for downstream systems that need schema compatibility. Paired with `decrypt_phi(record, key_ref)` and a key-rotation helper `rotate_phi_key(old_ref, new_ref)`.

## 6. Differential Privacy Noise Injection

For aggregate analytics on PHI-adjacent data (ages, zip codes, counts), add `apply_differential_privacy(values, epsilon, sensitivity)` implementing the Laplace mechanism. Allows publishing statistics with mathematically bounded privacy loss, meeting HIPAA Expert Determination requirements without full redaction.

## 7. De-identification Quality Metrics (Re-identification Risk Score)

After de-identification, score the remaining quasi-identifiers (age, gender, zip prefix) against the k-anonymity and l-diversity criteria. `score_reidentification_risk(cohort)` returns `k_value`, `l_value`, `t_closeness`, and a plain-English `risk_summary`. Surfaces hidden linkage attack exposure invisible to simple PHI scanning.

## 8. FHIR R4 Resource PHI Extractor

Add `classify_fhir_resource(resource: dict)` that understands FHIR R4 resource schemas (Patient, Encounter, Observation, DiagnosticReport). Maps FHIR paths (`Patient.name[0].family`, `Patient.birthDate`) directly to HIPAA categories instead of relying on flat field-name heuristics. Returns a FHIR-path-addressed result set.

## 9. Schema-Aware Column Classifier (SQL/Parquet/Arrow)

Add `classify_schema(schema: dict[str, str])` that accepts a column-name→dtype map from a SQL table or Parquet file and returns column-level PHI classification without requiring row data. Powers automated data-catalog tagging and prevents PHI columns from being added to unprotected tables at schema-migration time.

## 10. Redaction Policy DSL

Replace the single `replace_with_placeholder` strategy with a YAML/JSON policy DSL:

```yaml
rules:
  - category: ssn
    action: mask          # show last 4: ***-**-6789
  - category: email
    action: hash          # SHA-256 hex
  - category: name
    action: pseudonymise  # deterministic fake name via Faker seeded by tenant+value hash
  - category: date
    action: generalise    # retain year only
  - default:
    action: redact
```

Policies are loaded per-tenant, versioned, and hot-reloadable without service restart.

## 11. Provenance-Preserving Reversible Pseudonymisation

Add `pseudonymise(record, namespace)` that replaces PHI values with deterministic synthetic stand-ins (consistent fake name, synthetic MRN, shifted date) derived from `HMAC(tenant_secret, original_value)`. Add paired `depseudonymise(record, namespace)` for authorised re-identification. Enables analytics on pseudonymised data while retaining re-identification capability for care teams.

## 12. Cross-Tenant PHI Linkage Detection

In multi-tenant deployments, the same individual may appear across tenants under different record IDs. Add `detect_cross_tenant_links(tenant_a_records, tenant_b_records)` using a privacy-preserving record linkage protocol (Bloom-filter PPRL) to surface probable same-individual links without exposing raw PHI values across tenant boundaries.

## 13. HIPAA Minimum-Necessary OPA Policy Export

Auto-generate Open Policy Agent (OPA) Rego policies from the `_MINIMUM_NECESSARY` map so that API gateway enforcement stays in sync with the Python service logic. `export_opa_policy(purpose)` returns a valid `.rego` bundle and a corresponding `policy_test.rego` for CI validation.

## 14. Real-Time PHI Velocity Alerting

Track per-accessor PHI access counts within rolling windows (60 s, 5 min, 1 h) using a Redis sorted-set counter. `check_phi_access_velocity(accessor_id)` returns `is_anomalous: bool` and raises a NATS `phi.access.anomaly` event when velocity exceeds configured thresholds. Detects bulk data exfiltration, insider threats, and misconfigured ETL jobs.

## 15. Synthetic PHI Test-Data Generator

Add `generate_synthetic_phi_record(locale, seed)` that produces statistically realistic HIPAA-profile records (name, DOB, SSN, MRN, diagnosis codes, medications) using locale-aware Faker providers and real ICD-10/CPT code distributions. Used in unit tests, load testing, and demo environments without touching real patient data. Emits records in both flat-dict and FHIR R4 Patient bundle formats.
