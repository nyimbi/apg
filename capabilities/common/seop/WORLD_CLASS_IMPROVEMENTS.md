# SEOP — World-Class Improvement Proposals

Capability: Security Operations (`seop`)
Author: Nyimbi Odero
Date: 2026-06-11

---

## 1. Full async surface

Every public method is synchronous, blocking the event loop when called from FastAPI or async playbook runners. Convert the entire public API to `async def` methods so they can be awaited natively. Store state in async-safe collections guarded by `asyncio.Lock` rather than plain dicts.

## 2. Persistent backing store via async SQLAlchemy

All state lives in process memory and is lost on restart. Replace in-memory dicts with an async SQLAlchemy repository layer backed by PostgreSQL. Implement a `SeopRepository` ABC with `InMemorySeopRepository` (for tests) and `PostgresSeopRepository` (for production) so the service layer is storage-agnostic.

## 3. Structured event publishing via CloudEvents

`_record_event` emits to an internal dict. Wire it to a real CloudEvents publisher that serialises to JSON and delivers to Bytewax, Kafka, or any ASGI-compatible message bus. The method signature already exposes the right metadata; the plumbing is missing.

## 4. MITRE ATT&CK enrichment pipeline

SIEM rules and detections have `mitre_tactics` lists but no normalisation or enrichment. Add an async `enrich_mitre_context` method that maps raw tactic/technique strings to canonical ATT&CK IDs (T1059, TA0002 etc.), fetches technique descriptions from a local MITRE ATT&CK STIX bundle, and attaches enriched context to detections and hunts.

## 5. Correlation engine for multi-signal detections

Currently each detection is independent. Implement `correlate_signals` which groups detections sharing IOCs, SIEM rule IDs, or asset IDs within a configurable time window and produces a composite correlation record. This reduces analyst fatigue by surfacing campaign-level patterns rather than per-alert noise.

## 6. Risk-scored incident prioritisation

`open_incident` sets severity from a caller-supplied string with no cross-validation. Add `compute_incident_risk_score` that aggregates linked detection confidence scores, CVSS scores from associated vulnerability scans, asset criticality, and active threat-intel feed matches to produce a numeric risk score (0–100) used to sort the incident queue.

## 7. Automated playbook selection

`execute_response` requires the caller to supply a `playbook_id`. Add `recommend_playbook` which takes an incident and uses severity, detection type, and MITRE tactics to return the ranked list of approved playbooks that best match the incident. Remove the burden from the caller of knowing playbook IDs upfront.

## 8. SLA tracking and breach alerting

Open incidents have no SLA metadata. Add configurable per-severity SLA targets (e.g. critical → 4 h, high → 8 h) and an `async check_sla_breaches` method that scans open incidents, computes elapsed time, and emits `sla_breached` audit events for overdue items. Expose breach counts in the dashboard.

## 9. Threat-intel feed deduplication and expiry

`threat_intelligence_integration` appends indicators with no deduplication or TTL. Add `async refresh_threat_intel_feed` which diffs incoming indicators against the existing store, inserts new ones, marks stale ones expired based on a configurable `ttl_hours`, and returns a diff summary. Prevents the indicator store from growing unboundedly.

## 10. Compliance control mapping service

`close_incident` accepts a `compliance_mapping` string but does no validation. Add `async map_to_compliance_controls` which accepts a list of control frameworks (ISO 27001, NIST CSF, SOC 2) and returns structured mappings with control ID, domain, and evidence status. Integrates with the APG `comp` capability when present.

## 11. Analyst workload balancing

Alerts and incidents are assigned manually via free-text strings. Add `async assign_workload` which accepts a list of analyst IDs with capacity metadata and distributes open alerts and incidents to the analyst with the lowest open-item count, respecting severity caps per analyst tier.

## 12. Detection quality metrics and false-positive feedback loop

Detection precision is computed at dashboard time but is never fed back to tune rule thresholds. Add `async record_analyst_feedback` which accepts a detection ID and analyst rating (true_positive / false_positive / tuning_needed), persists the feedback, and computes running precision per SIEM rule. Expose per-rule false-positive rates in `siem_rule_management` output.

## 13. Automated evidence collection harness

Forensic evidence is recorded manually via `forensic_capture`. Add `async schedule_evidence_collection` which, on incident creation above a configurable severity, automatically schedules evidence collection tasks (log export, memory snapshot, network capture) by emitting structured task events. Analysts confirm or cancel; collected refs are appended to the incident automatically.

## 14. Zero-trust posture continuous monitoring

`record_posture_control` is a one-shot snapshot. Add `async evaluate_posture_drift` which compares the current posture snapshot against the previous one, identifies controls that regressed from `covered` to `partial` or `gap`, and emits `posture_drift_detected` events with a drift magnitude score. Enables continuous rather than point-in-time posture visibility.

## 15. Capability composition bus integration

SEOP is consumed by other APG capabilities (threat intelligence, anomaly detection, monitoring) but the integration is hand-wired. Add a `SeopCompositionBus` adapter that exposes typed composition hooks (`on_anomaly_detected`, `on_vulnerability_found`, `on_posture_change`) so upstream capabilities can push events into SEOP without knowing its internal structure. Each hook validates the payload against a Pydantic contract before dispatch.
