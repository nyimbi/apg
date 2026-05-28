"""Executable checks for MQEB edge IoT protocol adapters."""

from __future__ import annotations

import asyncio

from capabilities.common.mqeb.edge_computing import (
	EdgeBroker,
	EdgeBrokerConfig,
	EdgeDeploymentType,
	IoTDevice,
	IoTProtocol,
	SynchronizationStrategy,
)


def test_generic_iot_protocol_adapter_supports_device_round_trip():
	async def scenario() -> None:
		config = EdgeBrokerConfig(
			broker_id="edge_generic",
			deployment_type=EdgeDeploymentType.MICRO_EDGE,
			location="factory-floor",
			region="ke-nairobi",
			protocols_enabled=[IoTProtocol.OPC_UA],
			sync_strategy=SynchronizationStrategy.THRESHOLD,
		)
		broker = EdgeBroker(config)
		await broker.initialize()
		try:
			device = IoTDevice(
				device_id="opcua_device_001",
				device_type="plc",
				protocol=IoTProtocol.OPC_UA,
				battery_level=87.0,
			)

			assert await broker.connect_device(device) is True
			adapter = broker.protocol_adapters[IoTProtocol.OPC_UA]
			assert await adapter.send_message(device.device_id, b"set-speed=15") is True
			assert list(adapter.outbound_messages[device.device_id]) == [b"set-speed=15"]

			assert await adapter.inject_received_message(device.device_id, b"speed=15") is True
			assert await adapter.receive_message(device.device_id) == b"speed=15"

			assert broker.metrics["messages_processed"] == 1
			assert broker.message_buffer[-1].topic == "iot.opc_ua.opcua_device_001"
			assert broker.message_buffer[-1].payload == b"speed=15"
		finally:
			await broker.shutdown()

	asyncio.run(scenario())


def test_lorawan_and_coap_adapters_have_executable_receive_paths():
	async def scenario() -> None:
		config = EdgeBrokerConfig(
			broker_id="edge_mixed",
			deployment_type=EdgeDeploymentType.MINI_EDGE,
			location="warehouse",
			region="us-west",
			protocols_enabled=[IoTProtocol.LORAWAN, IoTProtocol.COAP],
		)
		broker = EdgeBroker(config)
		await broker.initialize()
		try:
			lora = IoTDevice("lora_001", "temperature", IoTProtocol.LORAWAN)
			coap = IoTDevice("coap_001", "actuator", IoTProtocol.COAP)

			assert await broker.connect_device(lora) is True
			assert await broker.connect_device(coap) is True

			lora_adapter = broker.protocol_adapters[IoTProtocol.LORAWAN]
			coap_adapter = broker.protocol_adapters[IoTProtocol.COAP]

			assert await lora_adapter.inject_received_message(lora.device_id, b"temp=21.5") is True
			assert await lora_adapter.receive_message(lora.device_id) == b"temp=21.5"
			assert await coap_adapter.inject_received_message(coap.device_id, b"state=ready") is True
			assert await coap_adapter.receive_message(coap.device_id) == b"state=ready"

			topics = [message.topic for message in broker.message_buffer]
			assert "iot.lorawan.lora_001" in topics
			assert "iot.coap.coap_001" in topics
		finally:
			await broker.shutdown()

	asyncio.run(scenario())
