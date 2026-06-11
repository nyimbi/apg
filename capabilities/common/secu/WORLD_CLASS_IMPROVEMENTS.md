# SECU Capability — World-Class Improvement Roadmap

**Capability**: Security (`secu`) | **Domain**: `common`
**Author**: Nyimbi Odero | **© 2025 Datacraft**

---

## 1. MITRE ATT&CK Tactic-to-Technique Correlation Engine

Map detected threat indicators directly to MITRE ATT&CK tactics (Initial Access, Execution, Persistence, etc.) and specific techniques (T1078, T1190, etc.). Produces kill-chain phase labels on every `ThreatIndicator`, enabling responders to see not just *that* an attack is happening but *where in the kill chain* it sits. Correlate multiple lower-confidence signals that share the same tactic to escalate confidence without requiring any single signal to cross a threshold.

## 2. Probabilistic Bayesian Risk Fusion

Replace the current static weighted-average risk calculation with a Bayesian network where each risk dimension (behavioral, device, network, temporal) is a node with prior and likelihood distributions. Evidence from each dimension updates the posterior probability of a compromise. The engine automatically recalibrates priors as tenant-level false-positive feedback accumulates, approaching an asymptotically calibrated scorer rather than a hand-tuned one.

## 3. Impossible-Travel Geospatial Engine

Implement real impossible-travel detection using a haversine/geodesic distance calculator coupled with access-event timestamps. Given two authentication events for the same user at coordinates (lat₁, lon₁, t₁) and (lat₂, lon₂, t₂), compute the required travel speed. Speeds above the threshold of a commercial aircraft (~900 km/h) flag a `ThreatType.CREDENTIAL_STUFFING` indicator with confidence proportional to how far above the threshold the implied speed is.

## 4. Streaming Threat Feed Ingestion (STIX/TAXII + OpenCTI)

Add an async adapter layer that polls STIX 2.1 bundles from TAXII 2.1 collection endpoints and OpenCTI GraphQL subscriptions. Normalise every indicator-of-compromise (IoC) into `ThreatIndicatorRecord` rows with TTL-based automatic expiry. New IoCs surface as quarantine or challenge triggers within the assessment pipeline within seconds of ingestion, with deduplication keyed on `indicator_type + value`.

## 5. Zero-Trust Continuous Verification Loop

Shift the current point-in-time assessment to a continuous monitoring posture. Attach a short-lived `SecurityContext` token (e.g., 5-minute sliding window) to each session. A background coroutine re-evaluates risk on every significant event (new API call, privilege change, network hop) and forces a step-up authentication or session revocation when the risk delta exceeds a configurable threshold. This eliminates the implicit trust assumption between an initial authentication and subsequent actions.

## 6. Federated Compliance Evidence Graph

Build a directed acyclic graph (DAG) where compliance controls map to evidence artefacts and those artefacts are versioned and content-addressed (SHA-256 over evidence bytes). Controls can inherit evidence from upstream nodes—enabling a SOC 2 Type II control to partially satisfy a GDPR Article 32 evidence requirement without duplicating collection. The DAG is serialised to the `evidence_repository` field and queryable by framework, control, or evidence hash.

## 7. ML-Driven False-Positive Feedback Loop

Add a `feedback_on_threat` async method that accepts analyst verdicts (true_positive / false_positive) on `ThreatIndicator` records. Verdicts are stored per-tenant and used to retrain a lightweight online learning model (e.g., Vowpal Wabbit or a simple logistic regression over feature hashes). The retrained model adjusts `false_positive_likelihood` on future indicators from the same source/detector combination, reducing alert fatigue without requiring a full MLOps pipeline.

## 8. Hardware Security Module (HSM) / TPM Attestation Integration

Extend `DeviceContext` to carry a TPM 2.0 quote blob or HSM attestation token. Add an async verification method that validates the quote against a known public endorsement key (EK), checks PCR values for OS integrity, and upgrades the device `trust_level` to `TRUSTED` only on successful attestation. Devices that fail attestation are immediately downgraded to `COMPROMISED`. This eliminates software-only device trust assertions that are trivially forged.

## 9. SIEM Push Adapter with CEF/LEEF/JSON-LD Normalisation

Implement an async SIEM push adapter that translates every `SecurityAuditEventRecord` to three output formats: Common Event Format (CEF), Log Event Extended Format (LEEF), and a JSON-LD envelope with a `schema.org/SecurityEvent` vocabulary. The adapter uses a non-blocking queue (asyncio.Queue) so SIEM latency never blocks the hot assessment path. Back-pressure is handled by dropping the oldest event when the queue exceeds a configurable high-water mark, ensuring low-latency operations even under SIEM outage.

## 10. Automated Playbook Execution Engine

Add a `SecurityPlaybook` model with trigger conditions (a subset of the existing rule DSL) and an ordered list of `PlaybookStep` actions (isolate_device, revoke_session, notify_soc, create_ticket, escalate_incident). An async `execute_playbook` method evaluates trigger conditions against the current `SecurityContext`, then executes steps sequentially with per-step timeout and rollback annotation. Steps that touch external systems (ticketing, IAM) use the adapter boundary so the engine runs safely without live integrations during testing.

## 11. Data-Loss Prevention (DLP) Classification Tagging

Intercept resource-access events and classify the sensitivity of requested resources using a combination of regex rules (credit-card PANs, national IDs, health data patterns) and an Ollama-hosted embedding model for semantic classification. Attach a `DataSensitivityTag` to the `SecurityContext` resource field. High-sensitivity resource access from elevated-risk contexts triggers automatic step-up challenges and audit-evidence requirements without requiring a separate DLP system.

## 12. Role-Based Segregation-of-Duties (SoD) Conflict Detection

Add an async `check_sod_conflicts` method that cross-references the `authorization_grants` in a `SecurityContext` against a tenant-level SoD conflict matrix (e.g., "create_payment AND approve_payment is always forbidden"). Conflicts generate a `ComplianceControlRecord` gap automatically and can be configured to deny access outright or require a time-bound waiver. The conflict matrix is expressed as a simple list of `frozenset` pairs for O(n) evaluation without quadratic enumeration.

## 13. Cryptographic Audit Chain (Merkle-Root Tamper Evidence)

Chain `SecurityAuditEventRecord` entries into an append-only Merkle tree keyed per-tenant. Each event's hash incorporates the previous chain tip, producing a tamper-evident audit log where any retroactive modification invalidates all subsequent hashes. Add a `verify_audit_chain` async method that returns the chain root, last verified height, and any detected breaks. This provides cryptographic non-repudiation required by SOX Section 404 and GDPR Article 5(2) accountability obligations.

## 14. Adaptive MFA Challenge Orchestration

Instead of a binary "challenge / no-challenge" decision, implement a risk-graduated MFA selector: score 0–49 → no-op, 50–69 → TOTP or push notification, 70–84 → hardware FIDO2 key required, 85–100 → out-of-band voice call + session freeze. The selector is tenant-configurable per resource classification. The async `select_mfa_challenge` method returns the required factor type and a one-time challenge token, and records the challenge issuance as an audit event to close the evidence loop.

## 15. Threat-Hunting Query Language (SECU-QL)

Design a minimal declarative query language for retrospective threat hunting over the in-memory audit event and assessment stores. Syntax: `HUNT events WHERE risk_score > 70 AND event_type IN ['device_quarantined', 'access_deny'] WITHIN 24h GROUPBY user_id`. The query is parsed with a hand-written recursive-descent parser (no external deps), evaluated lazily over the store iterators, and returns a ranked list of user or device identifiers with aggregated suspicion scores. This enables SOC analysts to build custom hunt rules without writing Python.
