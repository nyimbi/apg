"""Capability registry recommendation ranking regressions."""

from __future__ import annotations

import pytest

from capabilities.composition.registry.service import CapabilityRegistryService


@pytest.mark.asyncio
async def test_capability_recommendations_rank_intent_matches_above_popularity():
	service = object.__new__(CapabilityRegistryService)
	search_results = [
		{
			"capability_id": "cap-payroll",
			"capability_name": "Payroll Operations",
			"description": "Payroll and employee compensation processing",
			"category": "hcm",
			"subcategory": "payroll",
			"composition_keywords": ["payroll", "employee"],
			"provides_services": ["payroll-service"],
			"complexity_score": 3.0,
			"quality_score": 0.90,
			"popularity_score": 0.45,
			"usage_count": 120,
		},
		{
			"capability_id": "cap-popular",
			"capability_name": "Popular Analytics",
			"description": "High usage dashboards",
			"category": "analytics",
			"subcategory": "bi",
			"composition_keywords": ["dashboard"],
			"provides_services": ["reporting-service"],
			"complexity_score": 2.0,
			"quality_score": 0.95,
			"popularity_score": 0.95,
			"usage_count": 900,
		},
	]

	recommendations = await service._generate_capability_recommendations(
		"build employee payroll rules",
		search_results,
	)

	assert recommendations[0]["capability_id"] == "cap-payroll"
	assert recommendations[0]["matched_terms"] == ["employee", "payroll"]
	assert recommendations[0]["confidence_score"] > recommendations[1]["confidence_score"]
	assert recommendations[0]["score_breakdown"]["intent_match"] > 0


@pytest.mark.asyncio
async def test_capability_recommendations_fall_back_to_quality_without_query():
	service = object.__new__(CapabilityRegistryService)
	search_results = [
		{
			"capability_id": "cap-low",
			"capability_name": "Low Quality",
			"description": "",
			"category": "misc",
			"composition_keywords": [],
			"provides_services": [],
			"complexity_score": 8.0,
			"quality_score": 0.30,
			"popularity_score": 0.20,
			"usage_count": 5,
		},
		{
			"capability_id": "cap-high",
			"capability_name": "High Quality",
			"description": "",
			"category": "misc",
			"composition_keywords": [],
			"provides_services": [],
			"complexity_score": 2.0,
			"quality_score": 0.90,
			"popularity_score": 0.70,
			"usage_count": 200,
		},
	]

	recommendations = await service._generate_capability_recommendations(None, search_results)

	assert recommendations[0]["capability_id"] == "cap-high"
	assert recommendations[0]["recommendation_reason"] == "High quality capability for this search result set"
	assert recommendations[0]["matched_terms"] == []
