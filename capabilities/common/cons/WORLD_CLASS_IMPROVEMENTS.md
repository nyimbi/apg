# CONS - World Class Improvements

15 high-leverage improvements to make CONS a reference-grade consent management engine.

---

### I1. Cryptographic Consent Receipt Standard (ISO/IEC 29184)

**Category**: Compliance & Trust
**Justification**: Every consent capture should produce a tamper-proof, self-contained receipt — a signed JSON blob the data subject can verify independently of the controller's system. This is what OneTrust Certified Receipts and the Kantara Consent Receipt Specification mandate. Absence means consent is only as trustworthy as the controller's DB backup.
**Implementation**: On `capture_consent`, serialise the consent payload, sign it with an Ed25519 key loaded from `CONS_SIGNING_KEY_PATH`, embed the public key fingerprint and ISO 8601 timestamp, then store `receipt_jwt` on `ConsentEvent`. Expose `verify_consent_receipt(receipt_jwt: str) -> bool` that verifies the signature without any DB access.
**Competitor**: OneTrust Consent Receipt, Kantara Initiative CRS v1.1, Consentric

---

### I2. Consent Propagation Bus (Event-Driven Downstream Sync)

**Category**: Architecture & Integration
**Justification**: Consent status changes must propagate to every downstream system (CRM, email platform, analytics, data warehouse) within seconds, not via batch. Failure to propagate is the single largest cause of GDPR non-compliance post-capture. Salesforce CDP, mParticle, and Segment all provide real-time consent signal APIs for exactly this reason.
**Implementation**: Add an async `_emit_consent_event(event_type: str, payload: dict)` that publishes to a configurable backend — CloudEvents-shaped dict to an asyncio Queue by default, with optional `CONS_KAFKA_BOOTSTRAP` / `CONS_WEBHOOK_URL` env vars. All state-changing methods (`capture_consent`, `withdraw_consent`, `right_to_erasure`) call `await _emit_consent_event(...)` after persisting. Downstream adapters subscribe via `register_consent_listener(fn: Callable[[dict], Awaitable[None]])`.
**Competitor**: Segment Consent Manager, OneTrust Consent Signal SDK, mParticle Consent State

---

### I3. Async-Native Service (Full asyncio Rewrite)

**Category**: Performance & Scalability
**Justification**: All I/O-bound methods (DB writes, event emission, webhook calls, SLA timers) block the event loop when implemented synchronously. At 10k consent captures/second — a realistic peak for a marketing opt-in campaign — synchronous methods become a bottleneck and prevent use in FastAPI, Starlette, or async LangGraph agents.
**Implementation**: Convert every public method to `async def`. Replace internal dict stores with an `AsyncConsentStore` protocol backed by `asyncpg` connection pools. Gate the in-memory fallback behind `CONS_STORE=memory`. Use `asyncio.gather` for batch operations in `consent_analytics` and `review_stale_consents`.
**Competitor**: Usercentrics Consent Management Platform (async REST), ConsentKit (async Python SDK)

---

### I4. Versioned Consent Lineage Graph

**Category**: Auditability & Compliance
**Justification**: GDPR Recital 42 requires proof that consent was freely given and specific to the notice version in force at the time. When a notice is updated, regulators expect a traceable lineage: which subjects were on v1, which migrated to v2, which were re-consented. Current implementation tracks `notice_id` on a consent but has no versioned lineage model.
**Implementation**: Add `ConsentLineageNode(consent_id, notice_version, superseded_by, migrated_at)`. On `privacy_notice_version`, generate a re-consent campaign record linking old consents to the new notice version. Add `consent_lineage(tenant_id, subject_id) -> list[ConsentLineageNode]` that returns the full chain from original capture to current status.
**Competitor**: TrustArc Consent Versioning, Usercentrics Consent History API

---

### I5. Granular Consent Expiry with Auto-Renewal Prompts

**Category**: Compliance Operations
**Justification**: GDPR Article 7 requires consent to remain verifiably current. Many DPAs (German BfDI, CNIL, ICO) have fined organisations for relying on stale consent that was never explicitly re-confirmed. `review_stale_consents` identifies staleness but has no mechanism to set per-purpose expiry windows or trigger re-consent nudges.
**Implementation**: Add `expiry_days: int | None` field to `PrivacyPurpose`. On `capture_consent`, compute `expires_at = captured_at + timedelta(days=purpose.expiry_days)` if set. Add `async get_expiring_consents(tenant_id, within_days: int) -> list[dict]` that returns consents expiring within the window, sorted by urgency. Add `async schedule_reconsent_campaign(tenant_id, purpose_id, expiry_window_days: int, notifier_fn)` that calls `notifier_fn` for each affected subject.
**Competitor**: OneTrust Consent Renewal Campaigns, Didomi Re-consent Workflows

---

### I6. Preference Centre as First-Class Tenant-Branded Widget

**Category**: UX & Product
**Justification**: Preference centres that mirror the brand reduce opt-out rates by 30-50% (Acquia/Monsido data). Current `PreferenceProfile` is a data model with no rendering contract. Competitors ship embeddable, tenant-branded preference centre widgets that auto-ingest purpose definitions.
**Implementation**: Add `async render_preference_centre(tenant_id, subject_id, theme_overrides: dict | None) -> dict` that returns a JSON schema describing channels, purposes (name, legal basis, current consent status, description), toggle states, and a `submit_url`. Add `async apply_preference_centre_submission(tenant_id, subject_id, submission: dict, captured_by: str) -> dict` that atomically updates all preferences and withdraws/captures consent records accordingly.
**Competitor**: Usercentrics Preference Center, Didomi Preference Management, OneTrust Universal Preference Center

---

### I7. Cross-Regulation Rule Engine (GDPR / POPIA / CCPA / LGPD)

**Category**: Compliance & Multi-Jurisdiction
**Justification**: A single tenant may process data subjects from Kenya (PDPA), South Africa (POPIA), the EU (GDPR), California (CCPA), and Brazil (LGPD) simultaneously. Each has materially different requirements: CCPA has no consent requirement for sale of data under $250k revenue, POPIA mandates a 72-hour breach notification SLA, LGPD requires DPO registration. Current rule engine assumes GDPR-only semantics.
**Implementation**: Add `regulation: str` field to `PrivacyPurpose` and `PrivacyRequest`. Implement `RegulationRuleSet` protocol with concrete implementations for `GDPRRuleSet`, `CCPARuleSet`, `POPIARuleSet`, `LGPDRuleSet`. `evaluate_capability_rules` dispatches to the correct rule set based on `subject_jurisdiction` in the evaluation context. Add `async jurisdiction_check(tenant_id, subject_id, processing_activity) -> dict` that returns applicable regulations and required actions.
**Competitor**: OneTrust Global Privacy Control, TrustArc Multi-Jurisdiction Engine, WireWheel Privacy Ops

---

### I8. Consent Proof Ledger with Merkle-Tree Tamper Evidence

**Category**: Auditability & Legal Defence
**Justification**: The current `provenance_hash` is a single SHA-256 over the consent payload — sufficient for integrity checking of a single record but not for proving the audit log itself was not tampered with after the fact. UK ICO enforcement expects a verifiable audit trail. A Merkle tree over audit events allows O(log n) inclusion proofs.
**Implementation**: Maintain an append-only `_merkle_chain: list[str]` where each element is `SHA256(previous_root || audit_event_hash)`. Add `get_audit_proof(audit_event_id: str) -> dict` that returns the inclusion path. Add `verify_audit_chain(tenant_id: str) -> bool` that re-derives the root and compares to stored root. Store root snapshots in `_chain_checkpoints: dict[str, str]` keyed by UTC day.
**Competitor**: IBM OpenPages Audit Evidence, Palantir Foundry Audit Graph, custom Hyperledger Fabric deployments

---

### I9. Consent Score and Privacy Health Dashboard

**Category**: Analytics & Observability
**Justification**: Privacy officers need a single number to answer "how well is our consent estate managed?" Current `consent_analytics` returns raw counts. A normalised 0-100 consent score — weighted by coverage, staleness ratio, open DSR overdue rate, and unresolved breach count — enables trend monitoring and SLA alerting.
**Implementation**: Add `async consent_health_score(tenant_id: str) -> dict` returning `{"score": float, "breakdown": {...}, "trend": "improving"|"stable"|"declining", "risk_flags": list[str]}`. Score formula: `100 - (staleness_penalty + overdue_dsr_penalty + breach_penalty + low_coverage_penalty)`, each capped at 25 points. Store daily snapshots in `_health_snapshots: list[dict]` for trend computation over 7/30-day windows.
**Competitor**: OneTrust Privacy Health Index, TrustArc Privacy Score, Transcend Privacy Center Dashboard

---

### I10. Data Minimisation Enforcement at Capture

**Category**: Privacy by Design
**Justification**: GDPR Article 5(1)(c) requires data minimisation — only the personal data actually necessary for the stated purpose may be collected. Current capture methods accept arbitrary `data_categories` without validating them against a purpose's declared categories. This means a rogue integration can silently capture categories beyond what the notice discloses.
**Implementation**: On `capture_consent`, cross-check `data_categories` being collected against the linked `PrivacyPurpose.data_categories`. Raise `ValueError("data_categories_exceed_purpose_scope")` if the set is wider than declared. Add `async validate_minimisation(tenant_id, purpose_id, requested_categories: list[str]) -> dict` that returns `{"permitted": [...], "excess": [...], "compliant": bool}`.
**Competitor**: BigID Data Minimization, Securiti.ai Data Minimization Engine, Informatica Privacy Management

---

### I11. Automated DSAR Workflow with SLA Escalation

**Category**: Operations & Compliance
**Justification**: GDPR Article 12 mandates response to DSARs within 30 days; POPIA within 30 days; CCPA within 45 days. Current `submit_privacy_request` sets a `due_at` but there is no workflow state machine, assignee routing, or escalation mechanism. Unmanaged DSAR backlogs are the most common ICO enforcement trigger in 2024-2025.
**Implementation**: Add `WorkflowStage(name, assignee_role, sla_days, auto_advance: bool)` and a `DSARWorkflow` with stages `[identity_verification, data_collection, legal_review, response_delivery, closure]`. `submit_privacy_request` creates workflow stages. Add `async advance_dsar_stage(request_id, tenant_id, actor, notes) -> dict`, `async reassign_dsar(request_id, tenant_id, new_assignee, reason) -> dict`, and `async escalate_overdue_dsars(tenant_id) -> list[dict]` that auto-escalates all open requests past their SLA.
**Competitor**: OneTrust DSAR Automation, BigID DSR Manager, Exterro Privacy Rights Management

---

### I12. Consent Fatigue Detection and Optimisation

**Category**: UX & Conversion
**Justification**: Showing consent banners or preference centre prompts too frequently drives banner blindness and reduces meaningful consent quality. Research by Bösch et al. (2016) and Nielsen Norman Group shows that consent fatigue causes up to 80% of users to click "accept all" without reading. A fatigue detector reduces noise, improves signal quality, and demonstrates GDPR good-faith principle.
**Implementation**: Add `async detect_consent_fatigue(tenant_id, subject_id) -> dict` that computes `prompt_frequency_7d`, `accept_all_ratio`, `average_decision_time_ms`, and a `fatigue_risk: "low"|"medium"|"high"` label. Add `async optimise_consent_presentation(tenant_id, subject_id, purposes: list[str]) -> dict` that returns recommended prompt timing, grouping, and format (banner vs. in-flow vs. settings) based on fatigue score.
**Competitor**: Didomi Adaptive Consent UX, CookieYes Smart Blocking, Cookiebot Adaptive Consent

---

### I13. Decentralised Identity and Self-Sovereign Consent (DID/VC)

**Category**: Future-Proofing & Standards
**Justification**: W3C DIDs and Verifiable Credentials are becoming the legal standard for portable, user-controlled consent in the EU Digital Identity Wallet (eIDAS 2.0) and Kenya's emerging digital ID framework. Controllers that adopt DID-based consent now avoid costly re-platforming when the regulatory mandate arrives. Microsoft Entra Verified ID and Dock.io already offer production VC consent flows.
**Implementation**: Add `async issue_consent_vc(tenant_id, consent_id, holder_did: str) -> dict` that produces a W3C VC JSON-LD object with `credentialSubject` containing the consent terms, signed by the controller's DID. Add `async verify_consent_vc(vc_jwt: str) -> dict` that resolves the issuer DID and verifies the signature. Store `vc_id` on `ConsentEvent`.
**Competitor**: Microsoft Entra Verified ID, Dock.io Consent Credentials, Mattr Global VC Platform

---

### I14. Real-Time Consent Signal API (GPC / TCF 2.2 / IAB)

**Category**: Standards Compliance & Ad-Tech Integration
**Justification**: The IAB Transparency and Consent Framework (TCF 2.2) and Global Privacy Control (GPC) are legal consent signals under CCPA and increasingly under GDPR. Publishers and ad-tech stacks (Google, Meta, The Trade Desk) require a machine-readable TC string to lawfully process personal data for targeted advertising. Absence means blanket denial of ad revenue on consent-dependent inventory.
**Implementation**: Add `async generate_tc_string(tenant_id, subject_id, vendor_consents: dict[int, bool], purpose_consents: dict[int, bool]) -> str` that encodes a valid TCF 2.2 TC string (base64url-encoded bitfield). Add `async parse_tc_string(tc_string: str) -> dict` for decoding incoming signals. Add `async apply_gpc_signal(tenant_id, subject_id, gpc_header_value: str) -> dict` that interprets `Sec-GPC: 1` and auto-withdraws all sale/sharing consents.
**Competitor**: Quantcast Choice (TCF 2.2), LiveRamp Safe Haven, Prebid.js Consent Management Module

---

### I15. AI Consent Explainability and Algorithmic Transparency Notices

**Category**: AI Governance & Emerging Regulation
**Justification**: EU AI Act Article 13 and 50 require transparency notices for AI systems that process personal data in high-risk contexts (credit scoring, HR, law enforcement). GDPR Article 22 mandates explanation of solely automated decisions. Current `PrivacyAgent` registration captures the agent's existence but not what it decided or why. Future-proofing for the AI Act is commercially critical for any AI-enabled product.
**Implementation**: Add `AlgorithmicTransparencyNotice(agent_id, decision_type, input_features, output_type, explainability_method, human_oversight_available)`. Add `async register_algorithmic_notice(tenant_id, agent_id, notice: AlgorithmicTransparencyNotice) -> dict`. Add `async record_automated_decision(tenant_id, decision_id, agent_id, subject_id, input_summary: dict, output_summary: dict, explanation: str, human_reviewable: bool) -> dict` and `async challenge_automated_decision(tenant_id, decision_id, subject_id, challenge_reason: str) -> dict` for Art. 22 right-to-contest.
**Competitor**: IBM AI Fairness 360 + OpenPages, Microsoft Responsible AI Dashboard, Truera Model Intelligence
