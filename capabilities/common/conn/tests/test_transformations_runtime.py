"""Focused tests for executable CONN transformation expression behavior."""

import pytest

from capabilities.common.conn.transformations import DataTransformationEngine


@pytest.mark.asyncio
async def test_jq_expression_reads_nested_field():
	engine = DataTransformationEngine()

	result = await engine.transform_json_to_json(
		{"customer": {"profile": {"email": "user@example.test"}}},
		transformation_rules=[],
		jq_expressions=[".customer.profile.email"]
	)

	assert result == {"email": "user@example.test"}


@pytest.mark.asyncio
async def test_jq_expression_assigns_nested_field():
	engine = DataTransformationEngine()

	result = await engine.transform_json_to_json(
		{"source": {"name": "Amina"}, "target": {}},
		transformation_rules=[],
		jq_expressions=[".target.display_name = .source.name"]
	)

	assert result["source"]["name"] == "Amina"
	assert result["target"]["display_name"] == "Amina"


@pytest.mark.asyncio
async def test_jq_expression_maps_list_values():
	engine = DataTransformationEngine()

	result = await engine.transform_json_to_json(
		{"records": [{"amount": 10}, {"amount": 25}]},
		transformation_rules=[],
		jq_expressions=[".records | map(.amount)"]
	)

	assert result == {"result": [10, 25]}


@pytest.mark.asyncio
async def test_jq_expression_reads_array_index():
	engine = DataTransformationEngine()

	result = await engine.transform_json_to_json(
		{"records": [{"id": "first"}, {"id": "second"}]},
		transformation_rules=[],
		jq_expressions=[".records[1].id"]
	)

	assert result == {"id": "second"}
