"""Multi-camera fusion adapter boundary for POSE.

This module intentionally keeps the package dependency-light. Production
multi-camera synchronization and fusion should be supplied by CVSN/edge
adapters and recorded through `PoseService.record_frame()` and
`PoseService.reconstruct_3d()`.
"""

from __future__ import annotations

from typing import Any


def fusion_requirements() -> dict[str, Any]:
	return {
		"requires": ["camera_calibration_ref", "synchronized_frame_timestamps", "tenant_policy"],
		"adapter": "cvsn.multi_camera_fusion",
		"event_stream": "bytewax",
	}
