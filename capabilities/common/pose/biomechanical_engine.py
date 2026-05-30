"""Biomechanical analysis adapter boundary for POSE.

The executable package stores deterministic analysis records through
`PoseService.analyze_pose()`. Production medical, ergonomic, or sports
biomechanics engines should be attached through governed adapters and must
respect the medical-review guardrail.
"""

from __future__ import annotations

from typing import Any


def biomechanical_requirements() -> dict[str, Any]:
	return {
		"requires": ["pose_estimate", "analysis_policy", "reviewer_for_medical_grade"],
		"adapter": "cvsn.biomechanical_analysis",
		"event_stream": "bytewax",
	}
