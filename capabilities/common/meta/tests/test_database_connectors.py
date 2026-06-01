"""Regression coverage for META database connector executable fallbacks."""

from __future__ import annotations

import pytest

from capabilities.common.meta.connectors.base_connector import ConnectorConfig, DataType
from capabilities.common.meta.connectors.database_connectors import (
	BigQueryConnector,
	OracleConnector,
	RedisConnector,
	SQLServerConnector,
)


def _config(source: str, asset_name: str, column_type: str) -> ConnectorConfig:
	return ConnectorConfig(
		connection_string=f"{source}://offline",
		database="erp",
		schema="catalog",
		include_patterns=["*"],
		additional_params={
			"offline_catalog": [
				{
					"name": asset_name,
					"schema": "catalog",
					"asset_type": "table",
					"row_count": 2,
					"columns": [
						{
							"name": "id",
							"data_type": column_type,
							"primary_key": True,
							"is_nullable": False,
						},
						{
							"name": "email",
							"data_type": "VARCHAR",
							"sample_values": ["owner@example.com"],
						},
					],
					"sample_data": [
						{"id": 1, "email": "owner@example.com"},
						{"id": 2, "email": "steward@example.com"},
					],
				}
			]
		},
	)


@pytest.mark.asyncio
@pytest.mark.parametrize(
	("connector_cls", "source", "column_type", "expected_type"),
	[
		(OracleConnector, "oracle", "NUMBER", DataType.FLOAT),
		(SQLServerConnector, "sqlserver", "INT", DataType.INTEGER),
		(RedisConnector, "redis", "hash", DataType.JSON),
		(BigQueryConnector, "bigquery", "INT64", DataType.INTEGER),
	],
)
async def test_fixture_backed_database_connectors_discover_schema_and_samples(
	connector_cls,
	source: str,
	column_type: str,
	expected_type: DataType,
):
	connector = connector_cls(_config(source, "customers", column_type))

	connection = await connector.test_connection()
	discovery = await connector.discover_assets()
	schema = await connector.get_asset_schema("catalog.customers")
	samples = await connector.sample_asset_data("catalog.customers", 1)

	assert connection["status"] == "success"
	assert connection["mode"] == "metadata_fixture"
	assert discovery.total_assets == 1
	assert discovery.errors == []
	assert schema is not None
	assert schema.source_system == source
	assert schema.columns[0].data_type == expected_type
	assert schema.columns[0].is_primary_key is True
	assert schema.columns[1].contains_pii is True
	assert samples == [{"id": 1, "email": "owner@example.com"}]


@pytest.mark.asyncio
async def test_fixture_backed_database_connectors_empty_catalog_is_operable():
	connector = BigQueryConnector(
		ConnectorConfig(
			connection_string="bigquery://offline",
			database="analytics",
			additional_params={"offline_catalog": []},
		)
	)

	connection = await connector.test_connection()
	discovery = await connector.discover_assets()

	assert connection["status"] == "success"
	assert discovery.total_assets == 0
	assert discovery.errors == []
	assert discovery.warnings


@pytest.mark.asyncio
async def test_bigquery_datetime_fixture_type_keeps_datetime_precision():
	connector = BigQueryConnector(_config("bigquery", "events", "DATETIME"))

	schema = await connector.get_asset_schema("catalog.events")

	assert schema is not None
	assert schema.columns[0].data_type == DataType.DATETIME
