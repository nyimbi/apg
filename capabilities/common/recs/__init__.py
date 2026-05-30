"""APG Recommender Systems (RECS) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "recs"
__capability_name__ = "Recommender Systems"
__apg_dependencies__ = ["pred", "aicr", "nlpc"]

capability_metadata: dict[str, Any] = {
	"name": "recs",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware recommendation datasets, models, deployments, ranking policies, feedback loops, experiments, AI recommender agents, and personalization governance",
	"category": "specialized_ai_analytics",
	"subcategory": "recommender_systems",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["personalized_recommendations", "ranking_policies", "catalog_matching", "interaction_datasets", "model_training", "model_deployments", "feedback_loops", "experiment_optimization", "profile_features", "recommender_agents"],
	"permissions": ["recs:view", "recs:recommend", "recs:manage_data", "recs:manage_models", "recs:deploy", "recs:run_experiments", "recs:audit", "recs:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register RECS with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "recs",
		"aliases": ["recommendations", "recommender_systems", "personalization"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["mdm", "etlp", "audl", "comp", "bytewax", "moni"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"personalized_recommendations": "Generate governed recommendations from tenant-scoped profiles and catalogs",
			"interaction_datasets": "Register governed datasets and interaction events for training and feedback",
			"model_training": "Train, approve, and monitor deterministic recommendation models",
			"model_deployments": "Deploy approved recommendation models with target, approval, and rollback evidence",
			"ranking_policies": "Apply ranking objectives, constraints, diversity, and safety policies",
			"catalog_matching": "Match items, content, and entities using predictive and NLP features",
			"feedback_loops": "Capture impressions, clicks, dismissals, conversions, and ratings for recommendation learning",
			"experiment_optimization": "Run recommendation experiments with guardrails and business metrics",
			"recommender_agents": "Register Codex, Claude Code, OpenCode, Pi, and future agent runtimes as scoped recommender collaborators",
			"capability_rules": "Evaluate deterministic recommendation-governance rules",
			"visual_theming": "Apply recommendation-console theme tokens and components"
		},
		"endpoints": {
			"recommendations": "/recs/api/v1/recommendations",
			"datasets": "/recs/api/v1/datasets",
			"models": "/recs/api/v1/models",
			"deployments": "/recs/api/v1/deployments",
			"catalogs": "/recs/api/v1/catalogs",
			"profiles": "/recs/api/v1/profiles",
			"feedback": "/recs/api/v1/feedback",
			"experiments": "/recs/api/v1/experiments",
			"agents": "/recs/api/v1/agents"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get RECS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
