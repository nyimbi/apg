# FREC - Facial Recognition

FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, emotion-analysis governance, review queues, audit evidence, UI metadata, and visual theming.

The generated-application surface is dependency-light. `face_runtime.py`, `api_helpers.py`, and `view_models.py` can run without camera hardware, model servers, Flask, Flask-AppBuilder, databases, or computer-vision engines. Production deployments can connect real capture, matching, liveness, anti-spoofing, CVSN, BIOP, MFAU, AICR, ENCR, AUDL, and Bytewax adapters behind the same capability contract.

## What FREC Provides

- Face consent records with scope, purpose, evidence, and revocation state.
- Tenant-scoped face template metadata with quality and encryption guardrails.
- Liveness evidence with threshold, spoof, and deepfake controls.
- Face verification with quality, liveness, active-template, and match-confidence checks.
- Watchlist management and governed one-to-many identification.
- Emotion analysis only when an explicit approved purpose is recorded.
- Review routing for low-quality captures, low-confidence matches, and watchlist hits.
- Audit events for lifecycle transitions and generated-app dashboards.
- UI route metadata and compact theme components for identity workflows.

## Package Structure

- `SPECIFICATION.md` defines functional requirements and acceptance criteria.
- `PLAN.md` records the implementation plan and review checklist.
- `capability_contract.py` declares configuration, guardrails, UI routes, theme, and adapters.
- `face_runtime.py` implements the generated-app runtime.
- `api_helpers.py` exposes dependency-light API helper functions.
- `view_models.py` exposes route-ready UI model helpers.
- `app.py` derives semantic model and component manifest data from the live contract.
- `test_capability_contract.py` and `tests/test_package_contract.py` provide focused verification.

## Basic Usage

```python
from capabilities.common.frec.face_runtime import FrecService

service = FrecService()
tenant_id = "tenant-face"

consent = service.record_face_consent(
    consent_id="consent-alice-face",
    tenant_id=tenant_id,
    subject_id="alice",
    purpose="workforce authentication",
    evidence="signed-consent:v1",
)

template = service.enroll_face(
    template_id="template-alice-face",
    tenant_id=tenant_id,
    subject_id="alice",
    consent_id=consent["id"],
    template_hash="sha256:face-template",
    face_quality=0.94,
    template_encrypted=True,
)

liveness = service.record_liveness(
    liveness_id="live-alice-login",
    tenant_id=tenant_id,
    subject_id="alice",
    liveness_score=0.93,
)

verification = service.verify_face(
    verification_id="verify-alice-login",
    tenant_id=tenant_id,
    subject_id="alice",
    template_id=template["id"],
    liveness_id=liveness["id"],
    match_confidence=0.93,
)

assert verification["status"] == "verified"
```

## Watchlists

Identification requires an active watchlist policy. FREC keeps the generated-app runtime deterministic by comparing supplied candidate confidence values against tenant thresholds rather than running face-model inference.

```python
watchlist = service.create_watchlist(
    watchlist_id="wl-access-deny",
    tenant_id=tenant_id,
    name="Access deny list",
    policy_id="policy-watchlist-1",
    owner="security",
    reason="Physical access control",
)

service.add_watchlist_subject(
    watchlist_id=watchlist["id"],
    tenant_id=tenant_id,
    subject_id="alice",
    template_id=template["id"],
    added_by="security",
    reason="authorized access baseline",
)
```

## Composition Notes

FREC depends on `biop`, `cvsn`, and `aicr`. Optional adapters include `auth`, `audl`, `encr`, `mfau`, `moni`, and `cach`. Batch recognition events should use Bytewax through the `event_stream` adapter.

Generated applications should compose FREC through the contract and dependency-light helper modules. Production web views, database integrations, real model inference, camera capture, and hardware integrations remain adapter concerns.
