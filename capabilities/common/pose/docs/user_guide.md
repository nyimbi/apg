# Pose Estimation - User Guide

**Capability ID**: `pose` | **Domain**: `common` | **Version**: `1.1.0`
**Copyright**: (c) 2025 Datacraft | **Author**: Nyimbi Odero

---

## Overview

POSE is APG's governed human pose-estimation capability. It provides a complete
service layer for tracking human movement: raw keypoint registration, biomechanical
analysis, ergonomic risk scoring, gait assessment, exercise counting, privacy
anonymisation, and rendering-ready skeleton overlays.

All operations are tenant-scoped, audited, and governed by capability rules
enforced at the service layer. No operation silently succeeds without satisfying
its guardrail contract.

---

## Quick Start

```python
import asyncio
from capabilities.common.pose.service import PoseService

svc = PoseService()
T = "my-tenant"

model = svc.register_model("m1", T, "RTMPose-S", "rtmpose", "vision-team", "pose-policy:default")
session = svc.start_session(
    "sess-1", T, "Lab Study", "researcher", "camera:lab-a",
    model["id"], subject_consent_recorded=True, secure_stream=True
)
frame = svc.record_frame("frm-1", T, session["id"], 1, "2026-06-01T09:00:00Z", "frame://001", 1920, 1080)
keypoints = [
    {"name": "left_shoulder",  "x": 0.35, "y": 0.30, "confidence": 0.94},
    {"name": "right_shoulder", "x": 0.65, "y": 0.30, "confidence": 0.93},
    {"name": "left_hip",       "x": 0.36, "y": 0.58, "confidence": 0.91},
    {"name": "right_hip",      "x": 0.64, "y": 0.58, "confidence": 0.90},
    {"name": "left_knee",      "x": 0.37, "y": 0.78, "confidence": 0.88},
    {"name": "right_knee",     "x": 0.63, "y": 0.78, "confidence": 0.87},
    {"name": "left_ankle",     "x": 0.38, "y": 0.95, "confidence": 0.85},
    {"name": "right_ankle",    "x": 0.62, "y": 0.95, "confidence": 0.84},
    {"name": "nose",           "x": 0.50, "y": 0.10, "confidence": 0.96},
    {"name": "left_elbow",     "x": 0.25, "y": 0.50, "confidence": 0.89},
]
estimate = svc.estimate_pose("est-1", T, session["id"], frame["id"], model["id"], keypoints,
                              quality_review_recorded=True)
analysis = svc.analyze_pose("ana-1", T, estimate["id"], "biomechanical")
print(analysis["metrics"])
```

---

## Core Workflow

### Model Registration

```python
model = svc.register_model(
    model_id="vitpose-h",
    tenant_id=T,
    name="ViTPose-H (COCO)",
    model_type="vitpose",          # movenet | rtmpose | vitpose | swin_pose | edge_pose
    owner="cv-team",
    policy_ref="pose-policy:v2",
    minimum_keypoint_confidence=0.80,
    edge_ready=False,
)
```

### Session Lifecycle

```python
session = svc.start_session("s1", T, "Sprint Analysis", "coach", "rtsp://camera-01",
    model["id"], subject_consent_recorded=True, secure_stream=True,
    realtime_stream=True, max_persons=4)

svc.change_session_state(T, session["id"], "paused",    reason="lunch break")
svc.change_session_state(T, session["id"], "active",    reason="resumed")
svc.change_session_state(T, session["id"], "completed", reason="end of session")
```

### Real-Time Stream Path

For latency-sensitive ingestion use `real_time_pose()`. The session must have
`realtime_stream=True`.

```python
rt_est = asyncio.run(svc.real_time_pose(T, session["id"], "frm-rt-001", model["id"], keypoints))
```

---

## Biomechanical Analysis

### Joint Angle Extraction

Returns angles in degrees for each connected keypoint triple plus bilateral
symmetry deltas.

```python
angles = asyncio.run(svc.extract_joint_angles("ang-1", T, estimate["id"]))
# angles["joint_angles"]            -> [{joint, angle_degrees, confidence}, ...]
# angles["symmetry_deltas_degrees"] -> {joint_name: delta_degrees}

# Custom topology
angles = asyncio.run(svc.extract_joint_angles("ang-2", T, estimate["id"],
    skeleton_topology=[("left_hip", "left_knee", "left_ankle")]))
```

### Gait Analysis

```python
gait = asyncio.run(svc.gait_analysis("gait-1", T, session["id"], estimate_ids))
# gait["cadence_rpm"], gait["symmetry_score"], gait["mean_confidence"]
```

### Exercise Rep Counting

```python
reps = asyncio.run(svc.exercise_count("reps-1", T, session["id"], estimate_ids,
    exercise="squat", rep_threshold=0.05))
print(reps["repetitions"])
```

### Ergonomics Assessment

Requires `subject_consent_recorded=True` on the session.

```python
ergo = asyncio.run(svc.ergonomics_assess("ergo-1", T, estimate["id"],
    workstation_ref="desk-station-7"))
# ergo["risk_level"]  -> "low" | "medium" | "high"
# ergo["risk_score"]  -> int 1-7
```

### Pose Comparison

```python
cmp = asyncio.run(svc.pose_compare("cmp-1", T, "est-a", "est-b"))
# cmp["similarity_score"] -> 0.0-1.0
```

---

## Movement Recognition

### Action Recognition

```python
action = asyncio.run(svc.action_recognise("act-1", T, session["id"], estimate_ids,
    threshold=0.75))
print(action["recognised_action"])
```

### Gesture Detection

```python
gesture = asyncio.run(svc.gesture_detect("gest-1", T, session["id"], estimate["id"],
    hand="right"))
print(gesture["detected_gesture"])
```

### Fall Detection

```python
fall = asyncio.run(svc.fall_detect("fall-1", T, session["id"], estimate_ids,
    vertical_drop_threshold=0.35))
if fall["fall_detected"]:
    print(f"Fall detected! Drop: {fall['vertical_drop']}")
```

---

## Multi-Person Tracking

```python
batch = asyncio.run(svc.multi_person_pose(
    "batch-1", T, session["id"], frame["id"], model["id"],
    persons=[keypoints_person_a, keypoints_person_b]
))
# batch["person_count"], batch["estimates"]

track = asyncio.run(svc.skeletal_track("track-1", T, session["id"], estimate_ids))
# track["snapshots"] -> sorted by frame_number
```

---

## Quality and Governance

### Quality Certification

```python
cert = asyncio.run(svc.certify_estimate_quality(
    "cert-1", T, estimate["id"],
    reviewer="dr.smith",
    min_confidence=0.85,
    min_keypoints=10,
))
print(cert["grade"])         # "certified" | "rejected"
print(cert["content_hash"])  # SHA-256 of estimate payload
```

### Anatomical Anomaly Detection

```python
anomalies = asyncio.run(svc.flag_anatomical_anomalies("anom-1", T, estimate["id"]))
# anomalies["overall_severity"] -> "none" | "low" | "medium" | "high"
# anomalies["violations"]       -> list of constraint violations with delta values
```

### Model Drift Detection

```python
drift = asyncio.run(svc.detect_model_drift("drift-1", T, model["id"],
    window_size=30, drift_threshold=0.08))
if drift["drift_detected"]:
    print(f"Drift detected! Deviation: {drift['deviation']}")
# drift["ewma_series"] -> EWMA confidence values over the window
```

### Model Benchmarking

```python
bench = asyncio.run(svc.model_benchmark("bench-1", T, model["id"], test_estimate_ids))
print(bench["pass_rate"])
```

---

## Privacy and Anonymisation

Apply Gaussian noise to keypoint coordinates before sharing across tenant
boundaries.

```python
anon = asyncio.run(svc.anonymise_estimate(
    "anon-1", T, estimate["id"],
    noise_scale=0.02,   # larger = more private, less accurate
    seed=42,            # reproducible anonymisation
))
# anon["anonymised_keypoints"] -> noised coordinates
```

Anonymised keypoints are returned but NOT persisted as estimates, preventing
accidental record linkage.

---

## Rendering and Visualisation

### Skeleton Overlay Segments

```python
overlay = asyncio.run(svc.build_skeleton_overlay(
    "ovl-1", T, estimate["id"],
    topology="coco17",   # "coco17" | "halpe26" | "minimal"
))
for seg in overlay["segments"]:
    print(f"{seg['from']} -> {seg['to']}  colour={seg['colour']}  conf={seg['confidence']}")
```

Edge colours: green (#00cc44) >= 0.8 confidence, yellow (#ffcc00) >= 0.5,
red (#ff3300) below 0.5.

---

## Gap Filling and Interpolation

```python
filled = asyncio.run(svc.interpolate_missing_frames(
    "interp-1", T, session["id"], estimate_ids
))
print(f"Filled {filled['interpolated_count']} synthetic frames")
for frm in filled["frames"]:
    if frm["synthetic"]:
        print(f"  [SYNTHETIC] frame_number={frm['frame_number']}")
```

Synthetic frames are clearly flagged and not persisted as real estimates.

---

## Multi-Model Fusion

```python
fusion = asyncio.run(svc.fuse_estimates(
    "fus-1", T,
    [estimate_id_model_a, estimate_id_model_b, estimate_id_model_c],
    outlier_iqr_factor=1.5
))
print(fusion["fused_confidence"])
print(fusion["fused_keypoints"])
```

---

## Pose Normalisation

```python
normalised = asyncio.run(svc.pose_normalize(T, estimate["id"], reference_height=1.0))
print(normalised["scale_factor"])
```

---

## Data Export and Search

```python
# Export session estimates
export = asyncio.run(svc.pose_export("exp-1", T, session["id"], format_="json"))
print(export["estimate_count"], export["payload_size_bytes"])

# Filter by confidence band
results = asyncio.run(svc.estimate_search(T, session_id=session["id"],
    min_confidence=0.85, max_confidence=1.0))

# Session statistics
summary = asyncio.run(svc.session_summary(T, session["id"]))

# Tenant-wide analytics
analytics = asyncio.run(svc.pose_analytics(T, session_id=session["id"]))
print(analytics["fall_event_count"], analytics["low_quality_count"])
```

---

## Annotation for Training Pipelines

```python
annotation = asyncio.run(svc.pose_annotate(
    "ann-1", T, estimate["id"],
    label="correct_squat_form",
    notes="Full depth achieved, knees tracking over toes",
    annotator="coach.jones",
))
annotations = asyncio.run(svc.annotation_list(T, estimate_id=estimate["id"]))
```

---

## Guardrails Reference

| Guard | Enforced On |
|-------|-------------|
| `tenant_context_present` | All operations |
| `model_owner_present` | `register_model` |
| `model_policy_attached` | `register_model` |
| `session_owner_assigned` | `start_session` |
| `source_reference_present` | `start_session` |
| `subject_consent_recorded` | `start_session`, `ergonomics_assess` |
| `secure_stream` | Realtime sessions |
| `sensitive_use` + `approval_recorded` | Sensitive sessions |
| `frame_timestamp_present` | `record_frame` |
| `keypoint_confidence` in [0, 1] | `estimate_pose` |
| `quality_review_recorded` | Low-confidence estimates |
| `medical_review_recorded` | Medical-grade analyses |
| `camera_calibration_present` | `reconstruct_3d` |
| `agent_registered` + `runtime_supported` | Agent registration |
| `state_change_reason_present` | `change_session_state` |
| `reviewer_required` | `certify_estimate_quality` |
| `at_least_two_estimates_required` | `fuse_estimates` |

---

## Capability Composition

```apg
use pose;
```

POSE integrates with:
- `cvsn` - computer vision pre-processing
- `aicr` - AI coordination and routing
- `mlcm` - model lifecycle management

---

## Further Reading

- `service.py` - Complete business logic and all 27 async methods
- `models.py` - Domain dataclasses
- `api.py` - REST API endpoints
- `views.py` - Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` - 15 improvement proposals with rationale
- `SPECIFICATION.md` - Full capability specification
- `cap_spec.md` - APG capability contract
