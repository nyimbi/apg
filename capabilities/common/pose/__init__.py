"""APG Pose Estimation (POSE) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "2.0.0"
__capability_id__ = "pose"
__capability_name__ = "Pose Estimation"
__apg_dependencies__ = ["cvsn", "aicr", "mlcm"]

capability_metadata: dict[str, Any] = {
	"name": "pose",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Governed pose model registration, tracking sessions, frame capture, estimation, biomechanical analysis, 3D reconstruction, AI pose agents, quality review, and audit",
	"category": "specialized_ai_analytics",
	"subcategory": "pose_estimation",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["pose_estimation", "multi_person_tracking", "biomechanical_analysis", "pose_3d_reconstruction", "edge_pose_inference", "pose_agents", "pose_quality_governance"],
	"permissions": ["pose:view", "pose:estimate", "pose:track", "pose:analyze", "pose:manage_models", "pose:audit", "pose:admin"]
}

CAPABILITY_INFO = capability_metadata


def register_capability() -> dict[str, Any]:
	"""Register POSE with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "pose",
		"aliases": ["pose_estimation", "human_pose", "biomechanical_analysis"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["colb", "edge", "audl", "geos", "bytewax", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"pose_estimation": "Estimate human keypoints from tenant-scoped image and video streams",
			"multi_person_tracking": "Track multiple people with temporal consistency and session policy",
			"biomechanical_analysis": "Analyze movement, posture, angles, and ergonomic indicators",
			"pose_3d_reconstruction": "Reconstruct 3D pose from camera streams when policy permits",
			"pose_agents": "Register Codex, Claude Code, OpenCode, Pi, and future runtimes as scoped pose-analysis collaborators",
			"pose_quality_governance": "Enforce consent, secure streams, quality review, medical review, and tenant isolation",
			"capability_rules": "Evaluate deterministic pose-estimation governance rules",
			"visual_theming": "Apply pose-intelligence theme tokens and components"
		},
		"endpoints": {
			"estimate": "/pose/api/v1/estimate",
			"tracking": "/pose/api/v1/tracking",
			"analysis": "/pose/api/v1/analysis",
			"reconstruction": "/pose/api/v1/reconstruction",
			"sessions": "/pose/api/v1/sessions",
			"models": "/pose/api/v1/models",
			"agents": "/pose/api/v1/agents"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get POSE capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["CAPABILITY_INFO", "capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
