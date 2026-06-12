# POSE - Pose Estimation

POSE is APG's governed human pose-estimation capability. It provides
tenant-scoped model registration, tracking sessions, frame capture, pose
estimates, biomechanical analysis, 3D reconstruction records, AI pose-agent
registration, quality governance, audit events, UI model surfaces, and Bytewax
lifecycle stream policy.

The package is dependency-light and deterministic for generated applications.
Production model inference, camera pipelines, multi-camera fusion, edge
deployment, and medical/ergonomic engines attach through explicit APG adapters.

## What POSE Provides

- Pose model registration with owner, model type, model policy, keypoint
  confidence threshold, and edge-readiness metadata.
- Tracking session lifecycle with owner, source reference, subject consent,
  secure-stream enforcement, sensitive-use approval, and max-person policy.
- Frame capture with timestamp and image dimensions.
- Pose estimate records with keypoints, confidence, quality score, person count,
  and review evidence.
- Biomechanical analysis records with medical-review guardrails.
- 3D reconstruction records with camera-calibration guardrails.
- First-class AI pose agents for runtimes such as Codex, Claude Code, OpenCode,
  and Pi.
- Bytewax lifecycle stream contract for batch and runtime pose mutations.
- Joint angle extraction for biomechanical and ergonomic analysis.
- Confidence-weighted consensus fusion of multi-model or multi-camera estimates.
- Anatomical anomaly detection against keypoint topology constraints.
- Laplace-mechanism privacy anonymisation of pose estimates.
- Tamper-evident quality certification with SHA-256 content hashing.
- Linear frame interpolation for dropped-frame gap filling.
- EWMA-based model confidence drift detection.
- Display-ready skeleton overlay segment generation (COCO-17, Halpe-26, minimal).
- Temporal keypoint smoothing (EMA / boxcar) with residual noise RMS.
- Per-keypoint velocity and acceleration kinematics from skeletal tracks.
- ISO 8551 / AAOS range-of-motion measurement with clinical classification.
- Bilateral movement asymmetry detection with injury-severity audit events.
- ISO 11226 Posture Alignment Index (0-100) with traffic-light band.
- Evidence-based biomechanical injury risk rules engine with corrective cues.
- Concurrent batch frame + estimate ingestion with asyncio semaphore control.
- Cross-session longitudinal comparison with pairwise similarity matrix and trend vectors.

## Minimal Usage

```python
from capabilities.common.pose.service import PoseService

service = PoseService()
tenant_id = "tenant-pose"

model = service.register_model("rtmpose", tenant_id, "RTMPose", "rtmpose", "vision-team", "pose-policy:default")
session = service.start_session(
    "session-001",
    tenant_id,
    "Movement Study",
    "coach",
    "camera:studio-a",
    model["id"],
    subject_consent_recorded=True,
    secure_stream=True,
    realtime_stream=True,
)
frame = service.record_frame("frame-001", tenant_id, session["id"], 1, "2026-05-30T10:00:00Z", "frame://001", 1920, 1080)
estimate = service.estimate_pose(
    "estimate-001",
    tenant_id,
    session["id"],
    frame["id"],
    model["id"],
    [{"name": "left_shoulder", "x": 100, "y": 120, "confidence": 0.95}],
)
analysis = service.analyze_pose("analysis-001", tenant_id, estimate["id"], "biomechanical")
```

## Async Methods Quick Reference

| Method | Description |
|--------|-------------|
| `multi_person_pose()` | Batch estimate all persons in a single frame |
| `skeletal_track()` | Link estimates across frames into a temporal track |
| `action_recognise()` | Classify a pose sequence into a recognised action |
| `gesture_detect()` | Detect hand gesture from a single estimate |
| `fall_detect()` | Detect falls via hip-centroid vertical drop |
| `gait_analysis()` | Compute cadence, symmetry, stride variability |
| `pose_compare()` | Keypoint-by-keypoint similarity between two estimates |
| `exercise_count()` | Count rep oscillations from hip-y series |
| `ergonomics_assess()` | RULA-style ergonomics risk scoring |
| `pose_export()` | Export session estimates to JSON or plain text |
| `pose_annotate()` | Attach training labels to estimates |
| `pose_analytics()` | Aggregate statistics across a tenant |
| `real_time_pose()` | Low-latency path for realtime streams |
| `pose_normalize()` | Scale keypoints to reference skeleton height |
| `model_benchmark()` | Pass-rate benchmark for a model |
| `session_summary()` | Lightweight session statistics |
| `estimate_search()` | Filter estimates by session and confidence band |
| `annotation_list()` | List annotations, optionally by estimate |
| `model_list()` | List models with type and edge_ready filters |
| `extract_joint_angles()` | Anatomical joint angles + bilateral symmetry deltas |
| `fuse_estimates()` | Confidence-weighted consensus fusion of estimates |
| `flag_anatomical_anomalies()` | Topology constraint violation detection |
| `anonymise_estimate()` | Gaussian-noise privacy anonymisation |
| `certify_estimate_quality()` | SHA-256 tamper-evident quality certificate |
| `interpolate_missing_frames()` | Linear fill for dropped-frame gaps |
| `detect_model_drift()` | EWMA confidence drift detection |
| `build_skeleton_overlay()` | Display-ready edge segments for rendering |
| `smooth_keypoint_track()` | EMA/boxcar temporal noise filtering over a skeletal track |
| `compute_kinematics()` | Per-keypoint velocity, acceleration, and peak-velocity from a track |
| `measure_rom()` | ISO 8551 range-of-motion with clinical classification |
| `detect_asymmetry()` | Bilateral joint-pair asymmetry with injury-risk severity audit |
| `compute_posture_score()` | ISO 11226 Posture Alignment Index (0-100) with traffic-light band |
| `score_injury_risk()` | Evidence-based biomechanical injury risk rules engine (0-10 score) |
| `ingest_frame_batch()` | Concurrent batch frame + estimate ingestion with semaphore control |
| `longitudinal_compare()` | Cross-session similarity matrix and trend vectors |

## World-Class Enhancements (v2.0)

The following 15 improvements were designed to bring POSE to parity with or
beyond leading commercial platforms (Kinovea, Vicon, DARI Motion, Move.ai,
Sword Health, Sparta Science, etc.).

| # | Name | Category | Key Output |
|---|------|----------|------------|
| I1 | **Temporal Keypoint Smoothing** | Signal Processing | EMA / boxcar filter per keypoint; residual noise RMS per joint |
| I2 | **Velocity & Acceleration Kinematics** | Biomechanical Analytics | v (units/frame and units/s), a (units/frame²), peak-velocity frame index, kinetic energy proxy |
| I3 | **Pose-to-Text LLM Narration** | Generative AI / Accessibility | Plain-English narrative from joint angles + risk score via local Ollama — no cloud egress |
| I4 | **Streaming Pose Ring Buffer** | Real-Time Infrastructure | Circular buffer with eviction on overflow; head/tail/fill-ratio metadata |
| I5 | **Activity Spatial Heat Map** | Spatial Analytics | Normalised 2D density grid (default 64×64) of per-keypoint positional deltas |
| I6 | **Cross-Session Longitudinal Comparison** | Progress Tracking | Per-session confidence distribution; pairwise cosine similarity matrix; improving/stable/declining trend vectors |
| I7 | **Batch Frame Ingestion with Concurrency Control** | Throughput / DX | asyncio semaphore-bounded gather; per-frame success/failure; total latency summary |
| I8 | **Range-of-Motion (ROM) Measurement** | Rehabilitation / Clinical | Angular delta vs ISO 8551/AAOS normals; clinical classification: normal/restricted/hypermobile |
| I9 | **Bilateral Movement Asymmetry Detection** | Injury Prevention | Left/right joint-pair speed ratios; symmetric/mild/severe classification; high-severity audit on breach |
| I10 | **Posture Alignment Index Scoring** | Occupational Health | PAI 0-100 from head position, shoulder level, spinal alignment, pelvic tilt; green/amber/red band |
| I11 | **Biomechanical Injury Risk Rules Engine** | Predictive Health | Configurable rules (operator, threshold, evidence level A/B/C, weight); composite score 0-10; corrective cues |
| I12 | **Skeleton-to-BVH Export** | Interoperability | HIERARCHY + MOTION blocks from COCO-17 topology; Euler angles; compatible with Blender, Unity, Qualisys |
| I13 | **Differential Privacy Budget Accounting** | Privacy Engineering | Per-tenant/subject RDP epsilon ledger; reject anonymisation when budget exhausted; GDPR Art. 89 compliant |
| I14 | **Model Latency Profiling & SLA Alerting** | Performance Operations | P50/P95/P99 latency, fps, memory delta; high-severity audit when P99 exceeds SLA |
| I15 | **Pose Re-Identification Risk Assessment** | Identity / Security | Gait signature vector (stride, cadence, keypoint covariance); cosine similarity vs stored signatures; risk score + recommended anonymisation params |

Improvements I1, I2, I6, I7, I8, I9, I10, I11 are fully implemented in
`service.py`. I3, I4, I5, I12, I13, I14, I15 are specified and ready for
production adapter attachment.

## New Methods

### I1 — Temporal Keypoint Smoothing

Eliminates jitter from raw frame-to-frame keypoint noise before downstream
biomechanical analysis. Build the skeletal track first, then smooth it.

```python
# Build track first
track = await service.skeletal_track("track-001", tenant_id, session["id"], estimate_ids)

# Apply EMA smoothing, window=7
smoothed = await service.smooth_keypoint_track(
    "smooth-001", tenant_id, track["id"],
    window_size=7, filter_type="ema"
)
# smoothed["smoothed_series"]          — dict[keypoint_name, list[{frame_index, x, y}]]
# smoothed["noise_rms_per_keypoint"]   — dict[keypoint_name, float]
```

### I2 — Velocity and Acceleration Kinematics

Second-order finite differences over a skeletal track. Essential for sports
science, fall prediction, and rep counting without external preprocessing.

```python
kinematics = await service.compute_kinematics(
    "kin-001", tenant_id, track["id"], fps=30.0
)
for kp in kinematics["kinematics"]:
    print(kp["keypoint"], "peak_v =", kp["peak_velocity"],
          "ke_proxy =", kp["kinetic_energy_proxy"])
```

### I8 — Range-of-Motion Clinical Measurement

Computes angular delta for a named joint between two pose estimates and
classifies against ISO 8551 / AAOS normal ROM tables.

```python
rom = await service.measure_rom(
    "rom-001", tenant_id,
    estimate_id_start="estimate-start",
    estimate_id_end="estimate-end",
    joint="left_knee",           # or right_knee, left_hip, left_shoulder, etc.
)
# rom["rom_degrees"]              — float
# rom["percent_of_normal_rom"]    — float
# rom["clinical_classification"]  — "normal" | "restricted" | "hypermobile"
```

### I9 — Bilateral Movement Asymmetry Detection

Flags left/right imbalances above clinical thresholds. Raises a high-severity
audit event automatically when any pair exceeds the severe threshold.

```python
asym = await service.detect_asymmetry(
    "asym-001", tenant_id, track["id"],
    mild_threshold_pct=10.0,
    severe_threshold_pct=15.0,
)
for pair in asym["joint_pair_results"]:
    print(pair["joint_pair"], pair["asymmetry_pct"], "%", pair["classification"])
# asym["severe_asymmetry_detected"]  — bool (also triggers audit event)
```

### I10 & I11 — Posture Score + Injury Risk

Combine a single-frame posture snapshot with a configurable rules engine to
produce actionable clinical output.

```python
# Posture Alignment Index (ISO 11226)
score = await service.compute_posture_score("score-001", tenant_id, estimate["id"])
print(score["posture_alignment_index"], score["traffic_light_band"])  # e.g. 74.5, "amber"

# Biomechanical injury risk (configurable rules, defaults cover ACL/hamstring/lower-back)
angles_report = await service.extract_joint_angles("angles-001", tenant_id, estimate["id"])
risk = await service.score_injury_risk("risk-001", tenant_id, angles_report["id"])
print(risk["composite_injury_risk_score"], risk["risk_tier"])  # e.g. 3.5, "moderate"
for r in risk["rule_results"]:
    if r["triggered"]:
        print(r["rule_id"], "->", r["corrective_cue"])
```

### I7 — Batch Frame Ingestion

Eliminates caller-side concurrency management for long-video processing.

```python
frames = [
    {
        "frame_id": f"f{i}", "frame_number": i,
        "occurred_at": "2026-06-01T10:00:00Z",
        "source_ref": f"video://clip-01/frame/{i}",
        "width": 1920, "height": 1080,
        "keypoints": [{"name": "left_shoulder", "x": 0.5, "y": 0.4, "confidence": 0.92}],
    }
    for i in range(100)
]
batch = await service.ingest_frame_batch(
    "batch-001", tenant_id, session["id"], model["id"],
    frames=frames,
    max_concurrency=16,
)
print(batch["success_count"], "/", batch["total_frames"], "in", batch["elapsed_ms"], "ms")
```

## Guardrail Summary

POSE denies operations that lack tenant context, model owner, model policy,
session owner, subject consent, source reference, secure stream for realtime
tracking, sensitive-use approval, frame timestamp, keypoints, medical review,
camera calibration, AI pose-agent registration/runtime/scope/disclosure,
state-change reason/audit, tenant isolation, or Bytewax stream evidence for
batch mutations.

Low-quality estimates return a review requirement and are blocked by the local
service until quality review evidence is recorded.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/pose/__init__.py capabilities/common/pose/models.py capabilities/common/pose/service.py capabilities/common/pose/api.py capabilities/common/pose/views.py capabilities/common/pose/capability_contract.py capabilities/common/pose/app.py capabilities/common/pose/test_capability_contract.py capabilities/common/pose/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/pose/test_capability_contract.py capabilities/common/pose/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.pose import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/pose --json
./.venv/bin/apg capabilities publish-plan capabilities/common/pose --json
```
