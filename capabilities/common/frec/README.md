# FREC - Facial Recognition

FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, deepfake and morphing attack detection, demographic bias auditing, GDPR explainability, continuous ambient re-authentication, federated cross-tenant identification, consent portability, template aging management, ISO/IEC 30107-3 liveness compliance, and visual theming.

The generated-application surface is dependency-light. `face_runtime.py`, `api_helpers.py`, and `view_models.py` can run without camera hardware, model servers, Flask, Flask-AppBuilder, databases, computer-vision engines, durable stream processors, or external AI-agent clients. Production deployments can connect real capture, matching, liveness, anti-spoofing, CVSN, BIOP, MFAU, AICR, ENCR, AUDL, and Bytewax adapters behind the same capability contract.

## What FREC Provides

- Face consent records with scope, purpose, evidence, and revocation state.
- Tenant-scoped face template metadata with quality and encryption guardrails.
- Liveness evidence: passive sequence checks, active challenge/response, ISO/IEC 30107-3 Level 4 compliance evaluation.
- Face verification (1:1) with quality, liveness, active-template, and match-confidence checks.
- Face identification (1:N) with linear gallery scan and per-gallery access control.
- Watchlist management and governed one-to-many watchlist matching with per-watchlist thresholds.
- Deepfake detection via FFT spectral anomaly and DCT artifact analysis.
- Morphing attack detection via landmark asymmetry and Laplacian seam scoring.
- Demographic bias auditing per ISO/IEC 19795-10 with per-cohort FAR/FRR reporting.
- GDPR Art. 22 explainability: binding constraint, counterfactual threshold, plain-language summary.
- Continuous ambient re-authentication as an async generator stream.
- Cross-tenant federated identification with per-tenant consent proof enforcement.
- Consent portability as W3C Verifiable Credential JSON-LD (GDPR Art. 20).
- Template aging reports with drift detection and scheduled re-enrollment.
- Provider-neutral facial-recognition governance agents for Codex, Claude Code, opencode, Pi, and future runtimes through adapter contracts.
- Bytewax-first lifecycle batch validation for consent, template, liveness, verification, watchlist, identification, emotion, review, and agent changes.
- Review routing for low-quality captures, low-confidence matches, and watchlist hits.
- Audit events for lifecycle transitions and generated-app dashboards.
- UI route metadata and compact theme components for identity workflows.

## Package Structure

- `SPECIFICATION.md` — functional requirements and acceptance criteria.
- `PLAN.md` — implementation plan and review checklist.
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement areas for v2.0+.
- `capability_contract.py` — configuration, guardrails, UI routes, theme, and adapters.
- `service.py` — core `FacialRecognitionService` implementation.
- `face_runtime.py` — generated-app runtime.
- `api_helpers.py` — dependency-light API helper functions.
- `view_models.py` — route-ready UI model helpers.
- `app.py` — derives semantic model and component manifest data from the live contract.
- `tests/` — unit, integration, and compliance tests.
- `docs/user_guide.md` — full API reference.

## Production API Image Sources

The generated runtime stores governed metadata and does not perform raw image inference. The production Flask API adapter normalizes request images before handing them to recognition services. Enrollment, verification, and identification endpoints accept strict base64 `image_data`, base64 `data:` image URLs, and governed `http`/`https` image URLs.

Remote URL handling is fail-closed: unsupported schemes, private-network or loopback host resolution, empty payloads, payloads over 10 MiB, and non-image content types are rejected before service invocation. Live capture devices, model servers, and image storage systems remain external adapters.

## Quick Start

```python
from capabilities.common.frec.service import FacialRecognitionService

svc = FacialRecognitionService(
    database_url="postgresql://localhost/frec",
    encryption_key="<32-byte-key>",
    tenant_id="org-1",
)
await svc.initialize()

# Record consent
await svc.record_consent("alice", purpose="workforce_authentication",
                          obtained_by="hr-system", expiry="2027-01-01T00:00:00Z")

# Enroll
result = await svc.enroll_face("alice", image_array)
assert result["success"]

# 1:1 Verify
verification = await svc.verify_face("alice", probe_image)
assert verification["verified"]

# 1:N Identify
gallery = await svc.create_gallery("gallery-staff", "Staff", max_subjects=5000)
hits = await svc.identify_face(probe_image, "gallery-staff", top_k=3)
```

## Core API

| Method | Signature | Description |
|--------|-----------|-------------|
| `enroll_face` | `(subject_id, image_data, quality_threshold=0.85)` | Enroll a face template; requires active consent |
| `verify_face` | `(subject_id, probe_image, verification_config=None)` | 1:1 verification; returns `{verified, confidence, score}` |
| `identify_face` | `(probe_image, gallery_id, top_k=5)` | 1:N ranked identification in a gallery |
| `update_face_template` | `(subject_id, new_image)` | Replace templates with fresh enrollment |
| `delete_face_template` | `(subject_id)` | Hard-delete all templates for subject |
| `batch_enroll` | `(subjects_list)` | Concurrent multi-subject enrollment |
| `batch_verify` | `(probes_list)` | Concurrent multi-subject verification |
| `face_quality_score` | `(image_data)` | Quality assessment: score, issues, sharpness, brightness, contrast |
| `compare_faces` | `(image_a, image_b)` | Raw cosine similarity between two face images |
| `liveness_check` | `(image_sequence, method='passive')` | Passive liveness from frame sequence |
| `active_liveness_challenge` | `(session_id)` | Generate blink/turn/nod challenge |
| `validate_challenge_response` | `(session_id, response)` | Validate active liveness response |
| `presentation_attack_detect` | `(image)` | Detect print, replay, 3D mask attacks |
| `texture_analysis` | `(image)` | Laplacian + DoG texture naturalness check |
| `depth_check` | `(image_pair)` | Stereo/sequential depth disparity |
| `replay_detect` | `(video_metadata)` | Replay attack heuristics from video metadata |
| `liveness_score` | `(all_checks)` | Aggregate multi-signal liveness score |
| `create_gallery` | `(gallery_id, name, max_subjects, access_level)` | Create subject gallery |
| `delete_gallery` | `(gallery_id)` | Delete gallery |
| `gallery_stats` | `(gallery_id)` | Quality distribution and average score |
| `merge_galleries` | `(src_id, dst_id)` | Merge subjects from src into dst |
| `purge_expired` | `(gallery_id, expiry_days=365)` | Remove stale templates by age |
| `clone_gallery` | `(src_id, new_id, tenant_id)` | Clone gallery structure to another tenant |
| `estimate_age` | `(image)` | Age-range estimate (requires model adapter) |
| `detect_emotion` | `(image)` | Emotion detection (requires `emotion_intelligence` adapter) |
| `detect_occlusion` | `(image)` | Mask, glasses, hat detection |
| `face_detect_in_frame` | `(image)` | All bounding boxes in frame |
| `face_count` | `(image)` | Number of faces in image |
| `record_consent` | `(subject_id, purpose, obtained_by, expiry)` | Record biometric consent |
| `check_consent` | `(subject_id, purpose)` | Check consent validity |
| `revoke_consent` | `(subject_id)` | Revoke all consents for subject |
| `data_subject_erasure` | `(subject_id)` | GDPR/PDPA right-to-erasure |
| `compliance_report` | `(period, jurisdiction)` | Consent and erasure compliance summary |
| `create_watchlist` | `(watchlist_id, name, policy_id, owner, reason, match_threshold)` | Policy-bound watchlist |
| `add_watchlist_subject` | `(watchlist_id, subject_id, added_by, reason, expiry)` | Add subject to watchlist |
| `watchlist_match` | `(probe_image, watchlist_id)` | 1:N match against watchlist |
| `deepfake_detect` | `(image)` | FFT spectral + DCT deepfake analysis |
| `morphing_attack_detect` | `(image)` | Landmark asymmetry + seam morph detection |
| `bias_audit_report` | `(cohort_field, min_samples)` | Per-cohort FAR/FRR ISO/IEC 19795-10 audit |
| `explain_verification` | `(verification_id)` | GDPR Art. 22 verification explanation |
| `template_aging_report` | `(gallery_id, drift_threshold)` | Flag templates with match confidence drift |
| `reenroll_subject` | `(subject_id, new_image, quality_threshold, reason)` | Hard-delete and re-enroll |
| `continuous_auth_stream` | `(subject_id, frame_source, interval_frames, revoke_on_fail_count)` | Ambient re-auth async generator |
| `federated_identify` | `(probe_image, tenants, top_k)` | Cross-tenant parallel identification |
| `export_consent_portable` | `(subject_id)` | W3C VC JSON-LD consent export |
| `import_consent_portable` | `(subject_id, credential, obtained_by)` | Import and activate portable consent |
| `liveness_compliance_report` | `(test_results)` | ISO/IEC 30107-3 APCER/BPCER/ACER evaluation |
| `recognition_latency_report` | `(period)` | P95/P99 latency from audit events |
| `accuracy_metrics` | `(test_dataset_id)` | FAR, FRR, EER from labelled audit data |
| `recognition_volume_report` | `(period)` | Enrollment and verification volumes |
| `health_check` | `()` | Service and database health |
| `get_service_statistics` | `()` | Threshold config and 30-day analytics |

## World-Class Enhancements (v2.0)

All 15 improvements documented in `WORLD_CLASS_IMPROVEMENTS.md`, in order:

1. **Adaptive Threshold Calibration** — Bayesian per-subject dynamic thresholds derived from historical match score distributions; tighten for fresh templates, relax for aged ones, never below the tenant floor.

2. **Morphing Attack Detection (MAD)** — `morphing_attack_detect`: FFT compression boundary artifact detection + landmark asymmetry scoring, distinct from PAD.

3. **Continuous Authentication Streaming** — `continuous_auth_stream` async generator: passive ambient re-verification every N frames; emits REVOKE and stops after configurable consecutive failure count.

4. **Cross-Tenant Federated Identity Matching** — `federated_identify`: parallel fan-out across `(tenant_id, gallery_id)` pairs with per-tenant consent proof enforcement and merged ranked results.

5. **Template Aging and Re-enrollment Scheduling** — `template_aging_report` + `reenroll_subject`: track rolling match confidence per subject; flag drift exceeding configurable threshold; hard-delete and re-enroll.

6. **Privacy-Preserving Federated Learning** — `federated_model_update_round` stub: accept encrypted gradient updates from edge nodes, aggregate via secure aggregation, apply to local feature extractor — no raw biometric data crosses tenant boundaries.

7. **Demographic Bias Monitoring** — `bias_audit_report`: per-cohort FAR/FRR per ISO/IEC 19795-10; flags cohorts with >5pp differential. Required under EU AI Act and Kenya Data Protection Act.

8. **Deepfake and GAN-Generated Face Detection** — `deepfake_detect`: Fourier spectrum anomaly + DCT block artifact analysis; pluggable backend contract for FaceForensics++-trained classifier.

9. **Secure Enclave Template Storage with HSM Integration** — `HsmTemplateStore` adapter contract + `enroll_face_with_hsm`: PKCS#11-compatible HSM or cloud KMS (AWS KMS, Azure Key Vault, GCP Cloud KMS) for all wrap/unwrap operations; plaintext feature vector never in application memory.

10. **Real-Time Watchlist Hit Streaming via Event Bus** — `watchlist_hit_stream` async generator: publish confirmed hits to Bytewax, NATS, or APG `evtb` adapter with sub-second latency.

11. **Multi-Modal Fusion: Face + Voice + Iris** — `multimodal_verify`: score-level fusion via logistic regression over face, optional voice embedding, optional iris hash; degrades gracefully to face-only with adjusted threshold.

12. **Explainability API for Verification Decisions** — `explain_verification`: binding constraint, counterfactual threshold, contributing landmarks, plain-language summary; structured for GDPR Art. 22 compliance.

13. **Incremental 1:N Search Using FAISS/HNSW Indexing** — `build_gallery_index`, `update_gallery_index`, `identify_face_indexed`: replace O(N) linear scan with ANN index; incremental add/remove without full rebuild; persisted across restarts.

14. **ISO/IEC 30107-3 Level 4 Liveness Compliance Report** — `liveness_compliance_report`: APCER/BPCER/ACER evaluation against labelled test data; certifies Level 4 compliance (APCER ≤ 0.5%, BPCER ≤ 0.5%).

15. **Consent Portability and Cross-System Sync** — `export_consent_portable` / `import_consent_portable`: W3C Verifiable Credential JSON-LD consent records with cryptographic proof (sign via `encr` adapter); enables consent to follow the subject across organizational boundaries without re-enrollment.

## New Methods

### Deepfake and Morphing Attack Detection

```python
# Deepfake: FFT spectral + DCT artifact analysis
df = await svc.deepfake_detect(image)
# → {"is_deepfake": False, "risk_score": 0.12,
#    "indicators": {"spectral_anomaly": False, "dct_artifact": False, ...}}

# Morphing attack: landmark asymmetry + Laplacian seam score
morph = await svc.morphing_attack_detect(image)
# → {"is_morph": False, "morph_score": 0.08,
#    "indicators": {"landmark_asymmetry": 0.02, "seam_anomaly": False, ...}}
```

Wire an AICR adapter backed by a FaceForensics++-trained classifier for production deepfake detection.

### Demographic Bias Audit (ISO/IEC 19795-10)

```python
report = await svc.bias_audit_report(cohort_field="demographic_group", min_samples=30)
print(report["bias_flags"])   # cohorts with > 5pp differential FAR or FRR
print(report["overall_FAR"], report["overall_FRR"])
# cohorts below min_samples threshold are marked "insufficient_samples"
```

### Continuous Ambient Re-authentication

```python
async for event in svc.continuous_auth_stream(
    "alice", frame_generator, interval_frames=30, revoke_on_fail_count=3
):
    if event["status"] == "revoked":
        revoke_access(event["subject_id"])
        break
    elif event["status"] == "warning":
        log_warning(event["consecutive_failures"])
```

### Cross-Tenant Federated Identification

```python
result = await svc.federated_identify(probe_image, [
    {"tenant_id": "org-a", "gallery_id": "gal-a", "consent_proof": "cp-token-1"},
    {"tenant_id": "org-b", "gallery_id": "gal-b", "consent_proof": "cp-token-2"},
], top_k=5)
# → {"candidates": [{"subject_id": "...", "score": 0.94, "tenant_id": "org-a", "rank": 1}, ...]}
# Tenants without consent_proof are silently excluded.
```

### Template Aging Report and Re-enrollment

```python
# Flag subjects whose match confidence has drifted from enrollment quality
aging = await svc.template_aging_report("gallery-staff", drift_threshold=0.05)
print(aging["flagged_subjects"])
# → [{"subject_id": "alice", "enroll_quality": 0.92, "recent_avg_confidence": 0.84, "drift": 0.08}]

# Hard-delete and re-enroll a flagged subject
result = await svc.reenroll_subject(
    "alice", new_image, quality_threshold=0.85, reason="drift_detected"
)
```

### ISO/IEC 30107-3 Liveness Compliance

```python
test_data = [
    {"is_live_predicted": True, "is_bona_fide": True, "confidence": 0.97},
    {"is_live_predicted": False, "is_bona_fide": False, "attack_type": "print", "confidence": 0.82},
]
report = await svc.liveness_compliance_report(test_data)
print(report["compliant"], report["APCER"], report["BPCER"])
# Level 4 threshold: APCER <= 0.005, BPCER <= 0.005
```

### GDPR Art. 22 Verification Explanation

```python
exp = await svc.explain_verification(verification_id)
print(exp["binding_constraint"])       # "biometric_similarity" | "input_image_quality" | ...
print(exp["counterfactual"])           # what would need to change to flip the outcome
print(exp["plain_language_summary"])   # subject-readable disclosure text
```

### Consent Portability (GDPR Art. 20)

```python
# Export as W3C VC JSON-LD
exported = await svc.export_consent_portable("alice")
credential = exported["credential"]   # BiometricConsentCredential

# Import on another system
imported = await svc.import_consent_portable("alice", credential)
print(imported["imported_purposes"])  # activated purpose list
# Note: wire the `encr` adapter to verify the credential's cryptographic proof in production.
```

## Agent Composition and Lifecycle Batches

FREC treats AI agents as governed composition records. A generated app can register an agent that reviews facial-recognition evidence while the real agent runtime remains behind an AICR adapter.

```python
from capabilities.common.frec.face_runtime import FrecService

service = FrecService()
tenant_id = "tenant-face"

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

Privileged roles (`verification_reviewer`, `watchlist_reviewer`, `identification_reviewer`, `emotion_governance_reviewer`, `privacy_reviewer`, `lifecycle_batch_reviewer`, `facial_recognition_steward`) are marked `pending_review` unless human approval evidence is supplied. Non-Bytewax lifecycle batches are denied by the rule engine.

## Composition Notes

FREC depends on `biop`, `cvsn`, `aicr`, `encr`, `audl`, `conf`, and `mfau`. Optional adapters: `auth`, `moni`, `cach`. Batch recognition and lifecycle events should use Bytewax through the `event_stream` and lifecycle stream contracts.

Generated applications should compose FREC through the contract and dependency-light helper modules. Production web views, database integrations, real model inference, camera capture, and hardware integrations remain adapter concerns.

---

© 2025 Datacraft — www.datacraft.co.ke
