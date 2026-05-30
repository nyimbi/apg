# BIOP - Biometric Processing

BIOP provides governed biometric processing for APG applications. It covers consent, enrollment, encrypted template metadata, liveness-backed verification, match confidence review, cross-border privacy review, revocation, retirement, audit evidence, and route-ready UI models.

The generated-application surface is dependency-light. `biometric_runtime.py`, `api_helpers.py`, and `view_models.py` can be imported without biometric hardware, Flask, Flask-AppBuilder, database sessions, model servers, or computer-vision engines. Production deployments can bind sensors, matchers, HSM-backed vaults, CVSN, MFAU, AICR, AUDL, ENCR, and Bytewax adapters behind the same contract.

## What BIOP Provides

- Explicit biometric consent records with purpose, modality scope, jurisdiction scope, actor, evidence, and revocation state.
- Encrypted biometric template metadata for face, voice, fingerprint, iris, palm, behavioral, and document-derived modalities.
- Template quality, retention, retirement, and revocation guardrails.
- Liveness-backed verification with presentation-attack evidence.
- Low-confidence match routing to independent review.
- Cross-border biometric processing routing to independent privacy review.
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

## Composition Notes

BIOP depends on `mfau`, `cvsn`, and `aicr`. Optional adapters include `auth`, `audl`, `encr`, `frec`, `moni`, `cach`, and Bytewax event streams. Applications should compose BIOP through the capability contract and dependency-light runtime helpers, not by importing production-only web or hardware integration internals.

Raw biometric samples are outside the package runtime. BIOP stores template metadata and encrypted-template evidence only.
