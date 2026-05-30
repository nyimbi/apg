# POSE Capability Specification

## Identity

- Capability ID: `pose`
- Display name: Pose Estimation
- Category: common
- Runtime target: Python capability package
- Primary purpose: govern pose-estimation workflows from model registration and
  source capture through keypoint estimates, biomechanical analysis, 3D
  reconstruction, AI pose agents, quality review, and audit.

## Goals

POSE must provide a practical executable spine for applications that work with
human pose data while keeping consent, source security, medical review,
quality, tenant isolation, and AI-agent contributions explicit.

The capability must support:

- Model registration and model-use policy.
- Tracking session lifecycle.
- Frame capture and keypoint estimates.
- Quality review for low-confidence estimates.
- Biomechanical analysis with medical-grade review.
- 3D reconstruction with camera calibration evidence.
- AI pose-agent composition.
- Bytewax lifecycle event policy.
- UI models for operations, review, audit, and analytics.

## Lifecycle

1. **Register model**: capture owner, model type, policy reference, confidence
   threshold, and edge readiness.
2. **Start session**: create a tenant-scoped tracking session with owner, source
   reference, subject consent, secure-stream posture, sensitive-use approval,
   model binding, and max-person policy.
3. **Record frame**: capture frame number, timestamp, source reference, and
   dimensions.
4. **Estimate pose**: store normalized keypoints, person count, confidence,
   quality score, and review evidence.
5. **Analyze pose**: attach biomechanical or movement metrics, with medical
   review required for medical-grade analysis.
6. **Reconstruct 3D**: create deterministic 3D records only when camera
   calibration evidence is present.
7. **Register agents**: configure AI pose agents by runtime, role, scope,
   policy, registration, and contribution disclosure.
8. **Change state**: require reason and audit evidence for session lifecycle
   changes.

## Domain Model

- `PoseModelRecord`: model metadata and governance policy.
- `PoseSessionRecord`: tracking session and consent/source policy.
- `PoseFrameRecord`: frame timestamp, source, and dimensions.
- `PoseEstimateRecord`: keypoint estimate and quality metadata.
- `PoseAnalysisRecord`: biomechanical or movement analysis.
- `PoseReconstructionRecord`: 3D reconstruction output and calibration evidence.
- `PoseAgentRecord`: first-class AI pose-analysis collaborator.
- `PoseAuditEvent`: local audit evidence for pose operations.

## Rule Engine

The deterministic rule engine returns `allow`, `require_review`, or `deny`.
Rules cover tenant context, model owner/policy, session owner, subject consent,
source reference, secure realtime streams, sensitive use approval, frame
timestamp, keypoints, max-person limit, low-quality review, medical review, 3D
camera calibration, AI pose agents, state changes, cross-tenant access, and
Bytewax batch mutation enforcement.

## UI Contract

POSE exposes APG Python UI model routes:

- `/pose/dashboard`
- `/pose/estimate`
- `/pose/tracking`
- `/pose/analysis`
- `/pose/reconstruction`
- `/pose/sessions`
- `/pose/models`
- `/pose/quality`
- `/pose/agents`
- `/pose/audit`
- `/pose/analytics`
- `/pose/settings`

## Adapter Boundaries

POSE does not download or execute live model runtimes inside the generated
package. Production integrations should attach through:

- `cvsn` for computer vision inference, camera streams, and reconstruction.
- `aicr` for AI orchestration and pose agents.
- `mlcm` for model lifecycle and approval.
- `edge` for edge model deployment.
- `audl` for durable audit.
- `moni` for metrics and observability.
- `bytewax` for lifecycle events and batch pose mutation.

## Non-Goals For This Packet

- Live camera capture.
- Live HuggingFace/OpenCV/ONNX inference.
- Rendered browser UI verification.
- Medical-device certification.
- Full performance benchmarking.

Those are production adapter and validation passes after the executable
capability spine is stable.
