# World-Class Improvements: Compliance Management (comp)

## 1. Async-First Service Layer

All mutating and query methods are currently synchronous. Converting to `async def` throughout enables non-blocking I/O against real PostgreSQL, audit sinks, and evidence repositories. The service can integrate with `asyncpg`/`SQLAlchemy` async sessions without API surface changes once the backing store is swapped.

## 2. Persistent PostgreSQL Backend via SQLAlchemy Async

The in-memory dicts are a test harness, not a production store. Backing every entity with an async SQLAlchemy session and Alembic-managed migrations gives durable, queryable state with foreign-key integrity, JSONB audit payloads, and row-level tenant isolation via RLS policies.

## 3. Continuous Control Monitoring (CCM)

Today assessments are point-in-time. A `continuous_monitor` background task can periodically re-evaluate each control's evidence freshness, open findings, and testing-frequency SLA, emitting `control_degraded` events when coverage slips below threshold — moving from periodic to real-time assurance posture.

## 4. Risk-Adjusted Control Prioritisation

`risk_integrate` stores a flat score. A scoring engine that combines likelihood × impact, control effectiveness, residual risk, and regulatory weight produces a prioritised remediation queue. Integrating with the APG `risk` capability provides a unified heat-map across all domains.

## 5. Automated Evidence Collection via Adapters

Evidence today requires manual `record_evidence` calls. An `evidence_collector` adapter interface that polls source systems (cloud config scanners, SIEM exports, identity governance APIs, CI/CD pipelines) and auto-records fresh encrypted evidence eliminates manual toil and latency in the evidence chain.

## 6. Cross-Framework Control Mapping and Reuse

Controls are siloed per framework. A `cross_framework_map` operation that detects overlapping obligations across SOC 2, ISO 27001, GDPR, NIST CSF, and PCI-DSS allows a single control to satisfy multiple obligations, reducing duplicated assessments and evidence collection by up to 60%.

## 7. Machine-Readable Regulatory Change Feed Integration

`regulatory_alert` is a manual record. Connecting to live regulatory RSS/API feeds (EUR-Lex, US Federal Register, FCA, CBK) and running NLP-based impact classification against registered frameworks auto-triggers gap assessments when new obligations appear.

## 8. Cryptographic Evidence Chain with Merkle Proofs

The current `stable_digest` is SHA-256 over a dict. Chaining evidence records into a Merkle tree where each leaf is `H(evidence_id || collected_at || payload_hash)` and the root is recorded in the audit event provides tamper-evident provenance that can be independently verified by an auditor without exposing raw evidence.

## 9. Multi-Party Attestation Workflows

Attestation today is single-actor. A workflow that routes a report through N required attestors (CFO, CISO, DPO, Board Audit Committee) with configurable quorum thresholds, deadline enforcement, and escalation to the compliance steward closes the gap between lite and enterprise GRC suites.

## 10. AI-Assisted Finding Triage and Remediation Suggestions

When a finding is opened, an LLM agent (using the already-registered `ComplianceAgentRecord` infrastructure) can classify the finding against known control patterns, suggest a remediation plan from a curated library, estimate effort, and auto-assign to the right owner — reducing mean time to remediation.

## 11. Compliance Posture Score and Trend Analytics

A time-series `posture_score` model that snapshots coverage, open findings, overdue assessments, and escalation counts daily enables trend charts and regression detection. Alerting when posture degrades more than X% week-over-week gives leadership early warning before audits.

## 12. Exception Management Lifecycle

Exceptions (approved deviations from controls) are referenced in the UI routes but have no service layer. A full exception lifecycle — request, risk-acceptance, approval, time-bound expiry, renewal, and auto-reopening of finding when exception lapses — closes this gap cleanly.

## 13. Immutable Audit Log Export and Regulator Submission Package

The current audit log is an in-memory list. An `export_audit_package` method that serialises the Merkle-chained audit log, associated evidence references, attestations, and framework metadata into a signed ZIP artefact (PGP or JWS) ready for regulator or external auditor submission reduces audit-prep time from weeks to hours.

## 14. Fine-Grained RBAC and Separation-of-Duties Enforcement

Current role checks are coarse (owner vs. tester, approver vs. preparer). A full RBAC layer integrated with the APG `auth` capability, enforcing segregation of duties rules (e.g., finding opener ≠ finding resolver, evidence collector ≠ control assessor) at the service level prevents internal collusion and satisfies auditor SoD requirements.

## 15. Webhook and Event Bus Integration for Real-Time Notifications

Compliance state changes (finding escalated, report published, evidence expired, control degraded) currently produce audit records only. Emitting structured CloudEvents to the APG `ntfy` capability's event bus enables real-time Slack/Teams/PagerDuty/email alerts, workflow triggers in `wflo`, and cross-capability reactions in the wider APG composition graph.
