"""3D reconstruction adapter boundary for POSE.

The local `PoseService.reconstruct_3d()` method provides deterministic package
behavior. Production 3D reconstruction engines should attach through CVSN or
edge adapters with camera-calibration evidence.
"""

from __future__ import annotations

from typing import Any


def reconstruction_requirements() -> dict[str, Any]:
	return {
		"requires": ["pose_estimate", "camera_calibration_ref", "quality_policy"],
		"adapter": "cvsn.pose_3d_reconstruction",
		"event_stream": "bytewax",
	}
