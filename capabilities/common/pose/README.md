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
