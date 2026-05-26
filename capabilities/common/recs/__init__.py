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
	"description": "Tenant-aware recommendation models, catalogs, user profiles, ranking policies, experiments, and personalization governance",
	"category": "specialized_ai_analytics",
	"subcategory": "recommender_systems",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["personalized_recommendations", "ranking_policies", "catalog_matching", "experiment_optimization", "profile_features"],
	"permissions": ["recs:view", "recs:recommend", "recs:manage_models", "recs:run_experiments", "recs:admin"]
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
		"optional_dependencies": ["mdm", "etlp", "audl", "comp"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"personalized_recommendations": "Generate governed recommendations from tenant-scoped profiles and catalogs",
			"ranking_policies": "Apply ranking objectives, constraints, diversity, and safety policies",
			"catalog_matching": "Match items, content, and entities using predictive and NLP features",
			"experiment_optimization": "Run recommendation experiments with guardrails and business metrics",
			"capability_rules": "Evaluate deterministic recommendation-governance rules",
			"visual_theming": "Apply recommendation-console theme tokens and components"
		},
		"endpoints": {
			"recommendations": "/recs/api/v1/recommendations",
			"models": "/recs/api/v1/models",
			"catalogs": "/recs/api/v1/catalogs",
			"profiles": "/recs/api/v1/profiles",
			"experiments": "/recs/api/v1/experiments"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get RECS capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
