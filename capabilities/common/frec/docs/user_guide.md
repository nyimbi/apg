# Facial Recognition

**Capability ID**: `frec` | **Domain**: `common` | **Version**: `1.0.0`

## Description

FREC provides governed facial recognition for APG applications. It covers face consent, face-template enrollment, liveness evidence, one-to-one verification, one-to-many identification, watchlist policy, emotion-analysis governance, review queues, first-class facial-recognition governance agents, Bytewax lifecycle batch validation, audit evidence, UI metadata, and visual theming.

## Installation

```bash
pip install apg-common-frec
```

## Provides

- `facial_recognition`
- `face_identification`
- `facial_recognition_agent_composition`

## Requires

- `biop`
- `cvsn`
- `aicr`
- `encr`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/frec/dashboard` | `frec:view` | Overview |
| `/frec/subjects` | `frec:view` | Identity |
| `/frec/consents` | `frec:enroll` | Identity |
| `/frec/enrollment` | `frec:enroll` | Identity |
| `/frec/templates` | `frec:enroll` | Identity |
| `/frec/verification` | `frec:verify` | Identity |
| `/frec/identification` | `frec:identify` | Identity |
| `/frec/liveness` | `frec:verify` | Security |

## Key Service Methods

### Core Identity

- `initialize()` — Create DB tables and warm engines.
- `create_user(user_data)` — Register a subject.
- `enroll_face(subject_id, image_data, quality_threshold)` — Enroll face template.
- `verify_face(subject_id, probe_image, config)` — 1:1 verification.
- `identify_face(probe_image, gallery_id, top_k)` — 1:N gallery identification.
- `batch_enroll(subjects_list)` — Concurrent multi-subject enrollment.
- `batch_verify(probes_list)` — Concurrent multi-subject verification.
- `update_face_template(subject_id, new_image)` — Replace template.
- `delete_face_template(subject_id)` — Hard-delete all templates.

### Liveness and Anti-Spoofing

- `liveness_check(image_sequence, method)` — Passive multi-frame liveness.
- `active_liveness_challenge(session_id)` — Generate blink/turn/nod challenge.
- `validate_challenge_response(session_id, response)` — Validate active challenge.
- `presentation_attack_detect(image)` — Detect print/replay/3D mask.
- `texture_analysis(image)` — Laplacian + DoG natural texture scoring.
- `depth_check(image_pair)` — Stereo/sequential depth disparity.
- `replay_detect(video_metadata)` — Metadata-based replay detection.
- `liveness_score(all_checks)` — Weighted aggregate liveness score.
- `deepfake_detect(image)` — FFT spectral + DCT artifact analysis. _(new)_
- `morphing_attack_detect(image)` — Landmark asymmetry + seam scoring. _(new)_
- `liveness_compliance_report(test_results)` — ISO/IEC 30107-3 APCER/BPCER. _(new)_

### Watchlist

- `create_watchlist(watchlist_id, name, policy_id, owner, reason, match_threshold)` _(new)_
- `add_watchlist_subject(watchlist_id, subject_id, added_by, reason, expiry)` _(new)_
- `watchlist_match(probe_image, watchlist_id)` _(new)_

### Gallery Management

- `create_gallery(gallery_id, name, max_subjects, access_level)`
- `delete_gallery(gallery_id)`
- `gallery_stats(gallery_id)`
- `merge_galleries(src_id, dst_id)`
- `clone_gallery(src_id, new_id, tenant_id)`
- `purge_expired(gallery_id, expiry_days)`
- `export_gallery_metadata(gallery_id)`
- `list_enrolled(gallery_id, filters)`

### Template Aging and Re-enrollment

- `template_aging_report(gallery_id, drift_threshold)` _(new)_
- `reenroll_subject(subject_id, new_image, quality_threshold, reason)` _(new)_

### Continuous Re-authentication

- `continuous_auth_stream(subject_id, frame_source, interval_frames, revoke_on_fail_count)` — AsyncGenerator. _(new)_

### Federated Identification

- `federated_identify(probe_image, tenants, top_k)` — Cross-tenant parallel 1:N. _(new)_

### Attributes and Frame Analysis

- `estimate_age(image)`
- `detect_emotion(image)`
- `detect_occlusion(image)`
- `face_detect_in_frame(image)`
- `face_count(image)`
- `face_quality_score(image)`
- `compare_faces(image_a, image_b)`

### Consent and Compliance

- `record_consent(subject_id, purpose, obtained_by, expiry)`
- `check_consent(subject_id, purpose)`
- `revoke_consent(subject_id)`
- `data_subject_erasure(subject_id)` — GDPR/PDPA full erasure.
- `compliance_report(period, jurisdiction)`
- `export_consent_portable(subject_id)` — W3C VC JSON-LD export. _(new)_
- `import_consent_portable(subject_id, credential, obtained_by)` _(new)_

### Bias and Explainability

- `bias_audit_report(cohort_field, min_samples)` — ISO/IEC 19795-10 FAR/FRR by cohort. _(new)_
- `explain_verification(verification_id)` — GDPR Art. 22 binding constraint + plain language. _(new)_

### Analytics and Performance

- `recognition_latency_report(period)` — avg/p95/p99 latency.
- `recognition_volume_report(period)` — enrollment/verification volumes.
- `accuracy_metrics(test_dataset_id)` — FAR/FRR/EER from audit.
- `model_accuracy_trend(periods)` — EER trend.
- `false_match_investigate(match_id)` — Detailed false match audit.
- `health_check()` — Service health status.
- `system_capacity_check()` — Gallery/subject/consent counts.
- `get_service_statistics()` — Combined threshold + analytics.

_(See `service.py` for complete signatures and docstrings.)_

## Interoperability

`frec` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use frec;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FREC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
