# CONS - Consent and Privacy Management

CONS is the APG capability for governed consent, privacy preferences, privacy
requests, consent-gated processing, and auditable privacy operations. It lets
generated APG applications publish notices, register lawful purposes, capture
and withdraw consent, manage subject preferences, evaluate processing
decisions, fulfill privacy requests, register scoped AI privacy agents, and
audit every meaningful privacy transition.

The implementation is dependency-light and side-effect free. It records privacy
state and guardrail evidence without calling live identity, DLP, audit-log,
notification, workflow, regulator, or marketing systems.

## What CONS Provides

- Privacy notice publication with version, URL, language, purpose list, and
  publishing actor.
- Purpose registry with owner, legal basis, retention policy, notice linkage,
  data categories, and active state.
- Consent ledger with subject, purpose, notice, source, capture actor,
  provenance hash, status, and withdrawal timestamp.
- Preference profiles for channels and purposes.
- Consent-gated processing decisions with allow/deny evidence.
- Privacy request queue with identity verification, evidence reference, SLA
  due date, completion state, and resolution.
- First-class AI privacy agents for Codex, Claude Code, OpenCode, Pi, and
  compatible runtime adapters.
- Deterministic rules for tenant context, legal basis, purpose owner,
  retention policy, notice linkage, active consent, identity verification,
  request evidence, stale-consent review, AI-agent governance, state-change
  audit, cross-tenant isolation, and Bytewax batch mutation streams.
- View models for dashboard, subject privacy, privacy agents, audit,
  analytics, and settings.
- Theme metadata for APG Studio and generated Python applications.

## How To Use It

```python
from capabilities.common.cons.service import ConsService

service = ConsService()
tenant_id = "tenant-cons"

notice = service.publish_notice(
    notice_id="notice-v1",
    tenant_id=tenant_id,
    version="2026.1",
    url="https://privacy.example/notice",
    language="en",
    purposes=["marketing"],
    published_by="privacy-owner",
)

purpose = service.create_purpose(
    purpose_id="purpose-marketing",
    tenant_id=tenant_id,
    name="Product marketing",
    owner="privacy-owner",
    legal_basis="consent",
    retention_policy="retain-24-months",
    notice_id=notice["id"],
    data_categories=["email", "profile"],
)

agent = service.register_privacy_agent(
    tenant_id=tenant_id,
    agent_id="codex-notice-reviewer",
    name="Codex Notice Reviewer",
    runtime="codex",
    role="notice_reviewer",
    scope="Review notice wording, purpose linkage, and privacy request evidence.",
    contribution_disclosed=True,
    policy_ref="policy:cons:agents:v1",
)

consent = service.capture_consent(
    consent_id="consent-001",
    tenant_id=tenant_id,
    subject_id="subject-001",
    purpose_id=purpose["id"],
    notice_id=notice["id"],
    source="web-form",
    captured_by="preference-center",
)

decision = service.process_consent_gated_data(
    decision_id="decision-001",
    tenant_id=tenant_id,
    subject_id="subject-001",
    purpose_id=purpose["id"],
)
```

Use `api.py` when composing generated application handlers, and `views.py` for
framework-neutral UI state:

```python
from capabilities.common.cons.views import dashboard_model, privacy_agents_model

dashboard = dashboard_model(service, tenant_id)
agents = privacy_agents_model(service, tenant_id)
```

## Contract And Composition

`get_capability_contract()` publishes:

- configuration for purposes, consents, privacy requests, privacy agents,
  governance, observability, adapters, UI, and theme;
- JSON-schema-style configuration requirements;
- deterministic rule engine;
- UI routes under `/cons`;
- theme tokens under `cons_privacy_center`;
- Bytewax lifecycle-stream metadata.

CONS depends on `comp`, `auth`, and `dlpd`. Optional adapter boundaries include
`i18n`, `audl`, `mchn`, `wsbl`, `bytewax`, `ntfy`, and `wflo`.

## Guardrail Summary

CONS denies or requires review when:

- tenant context is missing;
- a purpose lacks legal basis, owner, retention policy, or notice linkage;
- consent capture lacks notice evidence;
- consent-gated processing lacks active consent;
- a privacy request lacks identity verification or request evidence;
- stale consent requires review;
- an AI privacy agent is unregistered, uses an unsupported runtime or role,
  lacks explicit scope, or has undisclosed contributions;
- a purpose state change lacks a reason or audit evidence;
- a cross-tenant access attempt is detected;
- a batch privacy mutation does not declare Bytewax.

## Focused Verification

Battery-conscious CONS checks:

```bash
./.venv/bin/python -m py_compile capabilities/common/cons/__init__.py capabilities/common/cons/capability_contract.py capabilities/common/cons/models.py capabilities/common/cons/privacy_engine.py capabilities/common/cons/service.py capabilities/common/cons/api.py capabilities/common/cons/views.py capabilities/common/cons/app.py capabilities/common/cons/test_capability_contract.py capabilities/common/cons/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/cons/test_capability_contract.py capabilities/common/cons/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/cons --json
./.venv/bin/apg capabilities publish-plan capabilities/common/cons --json
```

---

## World-Class Enhancements (v2.0)

Fifteen targeted improvements over baseline implementation:

- **I1. Cryptographic Consent Receipt Standard (ISO/IEC 29184)** [Compliance & Trust]
- **I2. Consent Propagation Bus (Event-Driven Downstream Sync)** [Architecture & Integration]
- **I3. Async-Native Service (Full asyncio Rewrite)** [Performance & Scalability]
- **I4. Versioned Consent Lineage Graph** [Auditability & Compliance]
- **I5. Granular Consent Expiry with Auto-Renewal Prompts** [Compliance Operations]
- **I6. Preference Centre as First-Class Tenant-Branded Widget** [UX & Product]
- **I7. Cross-Regulation Rule Engine (GDPR / POPIA / CCPA / LGPD)** [Compliance & Multi-Jurisdiction]
- **I8. Consent Proof Ledger with Merkle-Tree Tamper Evidence** [Auditability & Legal Defence]
- **I9. Consent Score and Privacy Health Dashboard** [Analytics & Observability]
- **I10. Data Minimisation Enforcement at Capture** [Privacy by Design]
- **I11. Automated DSAR Workflow with SLA Escalation** [Operations & Compliance]
- **I12. Consent Fatigue Detection and Optimisation** [UX & Conversion]
- **I13. Decentralised Identity and Self-Sovereign Consent (DID/VC)** [Future-Proofing & Standards]
- **I14. Real-Time Consent Signal API (GPC / TCF 2.2 / IAB)** [Standards Compliance & Ad-Tech Integration]
- **I15. AI Consent Explainability and Algorithmic Transparency Notices** [AI Governance & Emerging Regulation]

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
