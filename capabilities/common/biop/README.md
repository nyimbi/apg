# BIOP - Biometric Processing

BIOP provides governed biometric processing for APG applications. It covers consent, enrollment, encrypted template metadata, liveness-backed verification, match confidence review, cross-border privacy review, first-class biometric governance agents, Bytewax lifecycle batch validation, revocation, retirement, audit evidence, and route-ready UI models.

The generated-application surface is dependency-light. `biometric_runtime.py`, `api_helpers.py`, and `view_models.py` can be imported without biometric hardware, Flask, Flask-AppBuilder, database sessions, model servers, computer-vision engines, or external AI-agent clients. Production deployments can bind sensors, matchers, HSM-backed vaults, CVSN, MFAU, AICR, AUDL, ENCR, and Bytewax adapters behind the same contract.

## What BIOP Provides

- Explicit biometric consent records with purpose, modality scope, jurisdiction scope, actor, evidence, and revocation state.
- Encrypted biometric template metadata for face, voice, fingerprint, iris, palm, behavioral, and document-derived modalities.
- Template quality, retention, retirement, and revocation guardrails.
- Liveness-backed verification with presentation-attack evidence.
- Low-confidence match routing to independent review.
- Cross-border biometric processing routing to independent privacy review.
- Provider-neutral biometric governance-agent registration for Codex, Claude Code, opencode, and Pi style runtimes.
- Bytewax-first lifecycle batch validation for consent, template, verification, liveness, review, retention, and agent changes.
- Tenant-local state isolation and audit events for each lifecycle transition.
- Deterministic rule evaluation for generated APG applications and package tests.
- UI route metadata and compact theme components for biometric operations.

## Package Structure

- `SPECIFICATION.md` defines the functional and governance requirements.
- `PLAN.md` records the implementation and review checklist.
- `capability_contract.py` declares configuration, guardrails, UI routes, theme, and adapters.
- `biometric_runtime.py` implements the generated-app runtime.
- `api_helpers.py` exposes dependency-light API helper functions.
- `view_models.py` exposes route-ready UI model helpers.
- `app.py` derives semantic model and package manifest data from the live contract.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide focused verification.

## Basic Usage

```python
from capabilities.common.biop.biometric_runtime import BiopService

service = BiopService()
tenant_id = "tenant-bio"

consent = service.record_consent(
    consent_id="consent-alice",
    tenant_id=tenant_id,
    subject_id="alice",
    purpose="workforce authentication",
    modalities=["face"],
    jurisdictions=["KE"],
    granted_by="alice",
    evidence="signed-consent:v1",
)

template = service.enroll_template(
    template_id="template-alice-face",
    tenant_id=tenant_id,
    subject_id="alice",
    modality="face",
    template_hash="sha256:face-template",
    encrypted=True,
    quality_score=0.95,
    consent_id=consent["id"],
    retention_policy="workforce-biometric-365d",
)

verification = service.request_verification(
    verification_id="verify-alice-login",
    tenant_id=tenant_id,
    subject_id="alice",
    template_id=template["id"],
    modality="face",
    requested_by="access-service",
    match_confidence=0.94,
    liveness_score=0.91,
    source_jurisdiction="KE",
    target_jurisdiction="KE",
)

assert verification["status"] == "verified"

agent = service.register_biometric_agent(
    agent_id="agent-privacy",
    tenant_id=tenant_id,
    name="Privacy Reviewer",
    runtime="codex",
    role="privacy_reviewer",
    scope="cross-border biometric review",
    owner="privacy-office",
    purpose="Review biometric transfer risk",
    contribution_disclosed=True,
    human_approval_required=True,
)

batch = service.validate_biop_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=2,
    operation="biometric_agent_batch",
)

assert agent["status"] == "active"
assert batch["required_processor"] == "bytewax"
```

## Review Flows

Low-confidence matches become `pending_match_review`. Cross-border processing becomes `pending_privacy_review`. Reviews must be requested and decided by independent actors; the runtime rejects self-approval and stale review mutations.

```python
review = service.request_match_review(
    review_id="review-low-confidence",
    tenant_id=tenant_id,
    verification_id="verify-alice-login",
    requested_by="access-service",
    justification="Confidence below tenant threshold.",
)

service.decide_match_review(
    review_id=review["id"],
    tenant_id=tenant_id,
    reviewer="identity-reviewer",
    decision="approved",
    notes="Secondary evidence supports the match.",
)
```

## Agent and Lifecycle Guardrails

BIOP agents are composition records, not hardwired tool clients. The contract supports `codex`, `claude_code`, `opencode`, and `pi` runtimes through the `aicr_provider_neutral_biometric_agent_adapter` boundary. Unsupported runtimes, unsupported roles, missing scope, missing owner, missing purpose, and missing machine-contribution disclosure are denied. Privileged roles without human approval are retained as `pending_review`.

Lifecycle batches are Bytewax-first. A broker-specific queue-routed batch is intentionally denied:

```python
from capabilities.common.biop.capability_contract import evaluate_capability_rules

result = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "validate_biop_lifecycle_batch",
    "event_stream": "legacy_queue",
    "mutation_count": 1,
})

assert result["decision"] == "deny"
assert "bytewax_biop_stream_required" in result["matched_rules"]
```

## Composition Notes

BIOP depends on `mfau`, `cvsn`, `aicr`, `encr`, `audl`, and `conf`. Optional adapters include `auth`, `frec`, `moni`, `cach`, and Bytewax event streams. Applications should compose BIOP through the capability contract and dependency-light runtime helpers, not by importing production-only web, hardware integration, or external agent internals.

Raw biometric samples are outside the package runtime. BIOP stores template metadata and encrypted-template evidence only.
