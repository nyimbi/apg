# Pose Estimation — World-Class Improvement Catalogue

**Capability**: `pose` | **Domain**: `common` | **Author**: Nyimbi Odero
**Date**: 2026-06-11 | **Copyright**: © 2025 Datacraft

---

### I1. Temporal Keypoint Smoothing Pipeline

**Category**: Signal Processing | **Justification**: Raw keypoint estimates are noisy frame-to-frame; jitter propagates into gait analysis, rep counting, and biomechanical metrics producing clinically unreliable output — Savitzky-Golay filtering eliminates this at source, matching Kinovea and DARI Motion pre-analytics. | **Implementation**: `smooth_keypoint_track(smoothed_id, tenant_id, track_id, window_size, filter_type)` where filter_type is `ema` or `boxcar`; per-keypoint 1D filter over time axis; return smoothed trajectories and residual noise RMS per keypoint. | **Competitor**: Kinovea (built-in smoothing), DARI Motion (Butterworth filter on every joint)

---

### I2. Velocity and Acceleration Kinematics

**Category**: Biomechanical Analytics | **Justification**: Velocity and acceleration are the primary signals for sports science and fall prediction yet callers must reinvent finite-difference logic — exposing them directly collapses an entire preprocessing layer that every biomechanics platform provides natively. | **Implementation**: `compute_kinematics(report_id, tenant_id, track_id, fps)` — second-order finite differences per keypoint; return velocity in units/frame and units/second, acceleration, peak-velocity frame index, and kinetic energy proxy. | **Competitor**: Vicon Nexus (6-DOF kinematics auto-computed), OpenPose velocity extension

---

### I3. Pose-to-Text LLM Narration

**Category**: Accessibility / Generative AI | **Justification**: Non-technical stakeholders cannot consume raw keypoint data — a locally hosted LLM closes the last-mile gap without cloud egress, directly aligned with APG's Ollama-first generative AI strategy. | **Implementation**: `narrate_pose_analysis(narration_id, tenant_id, analysis_id, role, model)` — build structured prompt from joint angles, risk score, session metadata; route to local Ollama; return plain-English narrative with confidence caveat. | **Competitor**: Move.ai (AI movement descriptions), Tempus (LLM radiology narratives)

---

### I4. Streaming Pose Ring Buffer

**Category**: Real-Time Infrastructure | **Justification**: `real_time_pose()` processes one frame at a time with no buffering abstraction; callers managing 30fps streams duplicate buffering logic with memory leak risk — a service-native ring buffer removes this and feeds action recognition directly. | **Implementation**: `push_to_stream_buffer(tenant_id, session_id, estimate_id, capacity)` and `drain_stream_buffer(tenant_id, session_id)` — circular buffer evicting oldest on overflow; track head/tail pointers and fill ratio in audit metadata. | **Competitor**: AWS Kinesis Video Streams (ring buffer built-in), Azure Percept (edge stream buffer)

---

### I5. Activity Spatial Heat Map

**Category**: Spatial Analytics | **Justification**: There is no surface for understanding which body regions are kinematically active across a session — coaching and therapy applications need this to identify compensatory movement patterns without external preprocessing. | **Implementation**: `generate_activity_heatmap(heatmap_id, tenant_id, session_id, resolution)` — accumulate per-keypoint positional deltas into normalised 2D density grid (default 64×64); return grid with per-keypoint contribution weights and dominant motion zone labels. | **Competitor**: Dartfish (activity overlay heatmaps), Hudl Sportscode (spatial density maps)

---

### I6. Cross-Session Longitudinal Comparison

**Category**: Progress Tracking | **Justification**: `pose_compare()` compares two individual estimates but there is no mechanism to compare movement quality across sessions over time — the core use case for physical therapy progress and athletic periodisation. | **Implementation**: `longitudinal_compare(report_id, tenant_id, session_ids)` — per-session aggregate confidence distributions; pairwise cosine similarity matrix; trend vectors (improving / stable / declining) per session relative to baseline. | **Competitor**: PhysiTrack (longitudinal ROM tracking), Hudl IQ (multi-session trend analysis)

---

### I7. Batch Frame Ingestion with Concurrency Control

**Category**: Throughput / DX | **Justification**: Processing long video requires hundreds of sequential `record_frame()` + `estimate_pose()` calls; absence of a batch interface forces callers to manage concurrency, partial failures, and progress independently, multiplying integration cost. | **Implementation**: `ingest_frame_batch(batch_id, tenant_id, session_id, model_id, frames, max_concurrency)` — asyncio semaphore-bounded gather; return per-frame success/failure status, total latency, and progress summary. | **Competitor**: MediaPipe Batch (async frame pipeline), AWS Rekognition Video (batch job API)

---

### I8. Range-of-Motion (ROM) Clinical Measurement

**Category**: Rehabilitation / Clinical | **Justification**: ROM is the standard clinical metric for joint injury assessment yet the service has no dedicated ROM interface — therapists must manually compare angle values across sessions without normal-range context. | **Implementation**: `measure_rom(rom_id, tenant_id, estimate_id_start, estimate_id_end, joint)` — compute angular delta; compare against ISO 8551 / AAOS normal ranges; return ROM degrees, percent of normal, and clinical classification (normal / restricted / hypermobile). | **Competitor**: Sword Health (digital ROM), MooveCare (automated ROM tracking)

---

### I9. Bilateral Movement Asymmetry Detection

**Category**: Injury Prevention | **Justification**: Left-right asymmetry above 10-15% is a validated injury precursor in running biomechanics; the service has no proactive asymmetry alerting so coaches miss the injury window completely. | **Implementation**: `detect_asymmetry(report_id, tenant_id, track_id, mild_threshold_pct, severe_threshold_pct)` — bilateral joint-pair speed ratios from keypoint velocities; classify as symmetric / mild / severe; raise high-severity audit on breach. | **Competitor**: Sparta Science (movement quality asymmetry), Catapult Sports (bilateral load asymmetry)

---

### I10. Posture Alignment Index Scoring

**Category**: Occupational Health | **Justification**: Spinal alignment scores are the primary deliverable in occupational health risk assessments; a single normalised score is more actionable than raw joint angles and enables dashboard trending against ISO 11226. | **Implementation**: `compute_posture_score(score_id, tenant_id, estimate_id)` — evaluate head forward position, shoulder level, spinal vertical alignment, pelvic tilt; compute PAI 0-100; traffic-light bands green/amber/red. | **Competitor**: PostureScreen (posture score), Dorsavi (wearable posture index), Upright Go (real-time score)

---

### I11. Biomechanical Injury Risk Rules Engine

**Category**: Predictive Health | **Justification**: Clinical evidence links specific joint angle patterns (knee valgus, hip drop, trunk lean > 15°) to injury risk — a lightweight rules engine closes the gap between raw kinematics and actionable clinical insight without requiring an ML model. | **Implementation**: `score_injury_risk(risk_id, tenant_id, joint_angles_report_id, rules)` — evaluate configurable rules (operator, threshold, evidence level, weight); composite score 0-10; corrective cues mapped to triggered rules. | **Competitor**: Fusionetics (injury risk), Sparta Science (Load/Explode/Drive signature), Zone7 (AI injury prediction)

---

### I12. Skeleton-to-BVH Motion Capture Export

**Category**: Interoperability | **Justification**: BVH is the universal interchange format for motion capture used by every 3D animation tool and biomechanics lab — without a BVH exporter, pose data is stranded in the APG ecosystem and cannot reach Blender, Unity, or Qualisys. | **Implementation**: `export_to_bvh(export_id, tenant_id, track_id, fps)` — HIERARCHY block from COCO-17 topology with joint offsets; MOTION block from per-frame Euler angles via `extract_joint_angles()`; return BVH string. | **Competitor**: Xsens MVN (BVH export), Qualisys (BVH/C3D), Vicon (native BVH)

---

### I13. Differential Privacy Budget Accounting

**Category**: Privacy Engineering | **Justification**: `anonymise_estimate()` applies per-call noise but has no epsilon budget tracker — repeated queries exhaust budget and allow reconstruction attacks; proper DP accounting is required for GDPR article 89 research exemptions. | **Implementation**: `track_privacy_budget(tenant_id, subject_ref, epsilon_used, delta)` — maintain per-tenant/subject RDP composition ledger; reject anonymisation when remaining budget falls below floor; expose composition history in audit. | **Competitor**: Google DP Library (epsilon tracking), Apple DP (per-day budget enforcement)

---

### I14. Model Latency Profiling and SLA Alerting

**Category**: Performance Operations | **Justification**: Cold-start latency for Ollama/ONNX models can exceed 500ms; without profiling, SLA violations at session start go undetected until production incidents — proactive P99 tracking closes this gap. | **Implementation**: `profile_model_latency(profile_id, tenant_id, model_id, rounds, sla_p99_ms)` — synthetic inference rounds; record P50/P95/P99 latency, fps, memory delta; persist in `_latency_profiles`; raise high audit if P99 exceeds SLA. | **Competitor**: NVIDIA Triton (built-in profiling), TorchServe (latency benchmarking API)

---

### I15. Pose Re-Identification Risk Assessment

**Category**: Identity / Security | **Justification**: Gait signatures derived from pose data can re-identify subjects across sessions even after PII removal; without an explicit biometric linkage control, unintentional re-identification attacks are possible through data aggregation. | **Implementation**: `assess_reidentification_risk(report_id, tenant_id, session_id)` — compute gait signature vector (stride length, cadence, keypoint covariance fingerprint); cosine similarity against stored tenant signatures; return risk score and recommended anonymisation parameters. | **Competitor**: NIST PRTM (biometric risk framework), Socure (identity risk scoring)
