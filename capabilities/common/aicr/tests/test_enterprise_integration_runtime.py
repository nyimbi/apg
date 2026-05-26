"""Focused tests for executable AICR enterprise integration adapters."""

import pytest

from capabilities.common.aicr.enterprise_integration import (
	DatabaseConfig,
	DatabaseIntegration,
	DatabaseType,
	MessageQueueConfig,
	MessageQueueIntegration,
	MessageQueueType
)


@pytest.mark.asyncio
async def test_kafka_adapter_publishes_and_replays_local_messages():
	config = MessageQueueConfig(
		queue_type=MessageQueueType.APACHE_KAFKA,
		connection_url="kafka://offline",
		queue_name="aicr-events",
		routing_key="model-events"
	)
	integration = MessageQueueIntegration(config)
	await integration.initialize()

	assert await integration.publish_message({"event": "model.deployed"})

	received = []
	await integration.consume_messages(lambda message: received.append(message))

	assert received == [{
		"message_id": received[0]["message_id"],
		"timestamp": received[0]["timestamp"],
		"data": {"event": "model.deployed"}
	}]
	assert integration._local_topics["model-events"][0]["offset"] == 0


@pytest.mark.asyncio
async def test_kafka_adapter_delivers_to_registered_async_consumer():
	config = MessageQueueConfig(
		queue_type=MessageQueueType.APACHE_KAFKA,
		connection_url="kafka://offline",
		queue_name="aicr-events"
	)
	integration = MessageQueueIntegration(config)
	await integration.initialize()
	received = []

	async def callback(message):
		received.append(message["data"])

	await integration.consume_messages(callback)
	await integration.publish_message({"event": "model.updated"})

	assert received == [{"event": "model.updated"}]


@pytest.mark.asyncio
async def test_oracle_adapter_executes_metadata_backed_query():
	config = DatabaseConfig(
		database_type=DatabaseType.ORACLE,
		host="oracle.local",
		port=1521,
		database_name="AICR",
		username="aicr",
		password="secret",
		metadata={
			"tables": {
				"models": [
					{"id": "m1", "status": "active"},
					{"id": "m2", "status": "retired"}
				]
			}
		}
	)
	integration = DatabaseIntegration(config)
	await integration.initialize()

	rows = await integration.execute_query("SELECT * FROM models WHERE status = :1", ["active"])

	assert rows == [{"id": "m1", "status": "active"}]
	assert integration._local_query_log[0]["query"] == "SELECT * FROM models WHERE status = :1"


@pytest.mark.asyncio
async def test_sql_server_adapter_uses_configured_query_results():
	config = DatabaseConfig(
		database_type=DatabaseType.SQL_SERVER,
		host="sqlserver.local",
		port=1433,
		database_name="AICR",
		username="aicr",
		password="secret",
		metadata={
			"query_results": {
				"select count(*) as total from audit_events": [{"total": 3}]
			}
		}
	)
	integration = DatabaseIntegration(config)
	await integration.initialize()

	rows = await integration.execute_query("SELECT COUNT(*) AS total FROM audit_events")

	assert rows == [{"total": 3}]
