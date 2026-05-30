# POSE Model Selection

POSE keeps model selection policy explicit instead of hardcoding live model
downloads into the generated package.

Supported model families in the executable contract:

- `movenet`
- `rtmpose`
- `vitpose`
- `swin_pose`
- `edge_pose`

Production model selection should be handled by MLCM/CVSN adapters using the
registered `PoseModelRecord` policy fields:

- `owner`
- `policy_ref`
- `minimum_keypoint_confidence`
- `edge_ready`
- tenant-specific quality and consent policy

The local package uses deterministic keypoint records supplied by callers so
tests and generated applications remain dependency-light.
