"""Focused tests for executable CONN composition runtime behavior."""

import pytest

from capabilities.common.conn.composition_api import (
	CapabilityEvent,
	CapabilityInterface,
	CapabilityType,
	CompositionContract,
	ConnectionCapabilityComposer,
	IntegrationMethod
)
from capabilities.common.conn.service import ConnectionManager


def _composer():
	return ConnectionCapabilityComposer(
		ConnectionManager(audit_enabled=False, monitoring_enabled=False),
		tenant_id="tenant-a"
	)


def _target_interface(methods):
	return CapabilityInterface(
		name="analytics_target",
		version="1.0.0",
		capability_type=CapabilityType.ANALYTICS,
		supported_methods=methods,
		endpoints={"analytics.ingest": "/analytics/ingest", "default": "/analytics/events"},
		event_types=["analytics.ingest"],
		data_formats=["json"]
	)


def _event(records=None):
	return CapabilityEvent(
		event_id="event-1",
		source_capability="connection_management",
		target_capability=None,
		event_type="connection.created",
		timestamp="2026-05-26T16:41:00+03:00",
		payload={"records": records or [{"status": "active", "amount": 10}, {"status": "inactive", "amount": 4}]},
		metadata={"source": "test"}
	)


@pytest.mark.asyncio
async def test_data_stream_composition_executes_transformations_and_validation():
	composer = _composer()
	target = _target_interface([IntegrationMethod.DATA_STREAM])
	await composer.register_capability(target)
	contract = CompositionContract(
		source_capability=composer.own_interface.capability_id,
		target_capability=target.capability_id,
		integration_method=IntegrationMethod.DATA_STREAM,
		data_flow_direction="source_to_target",
		event_mappings={"connection.created": "analytics.ingest"},
		data_transformations=[
			{"type": "filter_data", "conditions": [{"field": "status", "operator": "equals", "value": "active"}]},
			{"type": "aggregate", "operations": [{"op": "sum", "field": "amount", "name": "total_amount"}]}
		],
		validation_rules=[
			{"type": "required_fields", "fields": ["record_count", "total_amount"]},
			{"type": "data_types", "types": {"total_amount": "number"}}
		],
		error_handling={}
	)
	composition_id = await composer.create_composition(contract)

	result = await composer.execute_composition(composition_id, _event())

	assert result == {"record_count": 1, "total_amount": 10}
	assert composer.composition_events[0]["status"] == "stream_ready"


@pytest.mark.asyncio
async def test_api_call_composition_prepares_endpoint_call():
	composer = _composer()
	target = _target_interface([IntegrationMethod.API_CALL])
	await composer.register_capability(target)
	contract = CompositionContract(
		source_capability=composer.own_interface.capability_id,
		target_capability=target.capability_id,
		integration_method=IntegrationMethod.API_CALL,
		data_flow_direction="source_to_target",
		event_mappings={"connection.created": "analytics.ingest"},
		data_transformations=[],
		validation_rules=[],
		error_handling={}
	)
	composition_id = await composer.create_composition(contract)

	result = await composer.execute_composition(composition_id, _event())

	assert result["status"] == "prepared"
	assert result["endpoint"] == "/analytics/ingest"
	assert result["payload"]["records"][0]["amount"] == 10


@pytest.mark.asyncio
async def test_composition_error_notification_is_recorded():
	composer = _composer()
	target = _target_interface([IntegrationMethod.DATA_STREAM])
	await composer.register_capability(target)
	contract = CompositionContract(
		source_capability=composer.own_interface.capability_id,
		target_capability=target.capability_id,
		integration_method=IntegrationMethod.DATA_STREAM,
		data_flow_direction="source_to_target",
		event_mappings={"connection.created": "analytics.ingest"},
		data_transformations=[],
		validation_rules=[{"type": "required_fields", "fields": ["missing_field"]}],
		error_handling={"strategy": "notify"}
	)
	composition_id = await composer.create_composition(contract)

	with pytest.raises(ValueError, match="Missing required fields"):
		await composer.execute_composition(composition_id, _event(records=[{"amount": 10}]))

	assert composer.composition_errors[0]["composition_id"] == composition_id
	assert composer.composition_events[-1]["status"] == "error_notified"
