# FREC - Facial Recognition

FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, emotion-analysis governance, review queues, first-class facial-recognition governance agents, Bytewax lifecycle batch validation, audit evidence, UI metadata, and visual theming.

The generated-application surface is dependency-light. `face_runtime.py`, `api_helpers.py`, and `view_models.py` can run without camera hardware, model servers, Flask, Flask-AppBuilder, databases, computer-vision engines, durable stream processors, or external AI-agent clients. Production deployments can connect real capture, matching, liveness, anti-spoofing, CVSN, BIOP, MFAU, AICR, ENCR, AUDL, and Bytewax adapters behind the same capability contract.

## What FREC Provides

- Face consent records with scope, purpose, evidence, and revocation state.
- Tenant-scoped face template metadata with quality and encryption guardrails.
- Liveness evidence with threshold, spoof, and deepfake controls.
- Face verification with quality, liveness, active-template, and match-confidence checks.
- Watchlist management and governed one-to-many identification.
- Emotion analysis only when an explicit approved purpose is recorded.
- Provider-neutral facial-recognition governance agents for Codex, Claude Code, opencode, Pi, and future runtimes through adapter contracts.
- Bytewax-first lifecycle batch validation for consent, template, liveness, verification, watchlist, identification, emotion, review, and agent changes.
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

## Production API Image Sources

The generated runtime stores governed metadata and does not perform raw image
inference. The production Flask API adapter still needs to normalize request
images before handing them to recognition services. Enrollment, verification,
and identification endpoints accept strict base64 `image_data`, base64 `data:`
image URLs, and governed `http`/`https` image URLs.

Remote URL handling is deliberately fail-closed: unsupported schemes,
private-network or loopback host resolution, empty payloads, payloads over 10
MiB, and non-image content types are rejected before service invocation. Live
capture devices, model servers, and image storage systems remain external
adapters.

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

## Agent Composition And Lifecycle Batches

FREC treats AI agents as governed composition records, not hidden implementation details. A generated app can register an agent that reviews facial-recognition evidence while the real agent runtime remains behind an AICR adapter.

```python
agent = service.register_facial_recognition_agent(
    agent_id="agent-face-governance",
    tenant_id=tenant_id,
    name="Face Governance Agent",
    runtime="codex",
    role="consent_reviewer",
    scope="consent and enrollment evidence",
    owner="identity-governance",
    purpose="review FREC lifecycle evidence before production rollout",
)

batch = service.validate_frec_lifecycle_batch(
    tenant_id=tenant_id,
    event_stream="bytewax",
    mutation_count=3,
    operation="facial_recognition_agent_batch",
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

Privileged roles such as `verification_reviewer`, `watchlist_reviewer`, `identification_reviewer`, `emotion_governance_reviewer`, `privacy_reviewer`, `lifecycle_batch_reviewer`, and `facial_recognition_steward` are marked `pending_review` unless human approval evidence is supplied. Non-Bytewax lifecycle batches are intentionally denied by the rule engine.

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
    reason="authorized access record",
)
```

## Composition Notes

FREC depends on `biop`, `cvsn`, `aicr`, `encr`, `audl`, `conf`, and `mfau`. Optional adapters include `auth`, `moni`, and `cach`. Batch recognition and lifecycle events should use Bytewax through the `event_stream` and lifecycle stream contracts.

Generated applications should compose FREC through the contract and dependency-light helper modules. Production web views, database integrations, real model inference, camera capture, and hardware integrations remain adapter concerns.

## New Capabilities (v1.1)

### Watchlist Management

```python
wl = await svc.create_watchlist("wl-deny", "Deny list", policy_id="pol-1",
                                 owner="security", reason="access control", match_threshold=0.90)
await svc.add_watchlist_subject("wl-deny", subject_id="suspect-007", added_by="officer", reason="court order")
result = await svc.watchlist_match(probe_image, "wl-deny")
# → {"hit_count": 1, "hits": [{"subject_id": "suspect-007", "score": 0.94}]}
```

### Deepfake and Morphing Attack Detection

```python
df = await svc.deepfake_detect(image)         # FFT spectral + DCT artifact analysis
morph = await svc.morphing_attack_detect(image)  # landmark asymmetry + seam scoring
```

### Demographic Bias Audit (ISO/IEC 19795-10)

```python
report = await svc.bias_audit_report(cohort_field="demographic_group", min_samples=30)
print(report["bias_flags"])  # cohorts with > 5pp differential FAR/FRR
```

### GDPR Explainability (Art. 22)

```python
exp = await svc.explain_verification(verification_id)
print(exp["plain_language_summary"])
```

### Template Aging and Re-enrollment

```python
aging = await svc.template_aging_report("gallery-staff", drift_threshold=0.05)
await svc.reenroll_subject("alice-001", new_image, reason="drift_detected")
```

### Continuous Ambient Re-authentication

```python
async for event in svc.continuous_auth_stream("alice", frames, interval_frames=30, revoke_on_fail_count=3):
    if event["status"] == "revoked":
        revoke_access(event["subject_id"])
```

### Cross-Tenant Federated Identification

```python
result = await svc.federated_identify(probe, [
    {"tenant_id": "org-a", "gallery_id": "gal-a", "consent_proof": "cp-1"},
    {"tenant_id": "org-b", "gallery_id": "gal-b", "consent_proof": "cp-2"},
])
```

### Consent Portability (GDPR Art. 20)

```python
exported = await svc.export_consent_portable("alice")   # W3C VC JSON-LD
imported = await svc.import_consent_portable("alice", exported["credential"])
```

### ISO/IEC 30107-3 Compliance Report

```python
report = await svc.liveness_compliance_report(labelled_test_results)
print(report["compliant"], report["APCER"], report["BPCER"])
# Level 4: APCER <= 0.5%, BPCER <= 0.5%
```

See `WORLD_CLASS_IMPROVEMENTS.md` for 15 architectural improvement areas and `docs/user_guide.md` for complete API reference.
