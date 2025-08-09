#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Edge Computing Tests
Tests for edge-native brokers and IoT protocol support

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from uuid_extensions import uuid7str

# Import MQEB components
from ..models import MQMessage, MessagePriority
from ..service import MQEBService
from ..edge_computing import (
	EdgeBroker, EdgeOrchestrator, IoTProtocolAdapter, MQTTAdapter, LoRaWANAdapter, CoAPAdapter,
	EdgeDeploymentType, IoTProtocol, SynchronizationStrategy,
	EdgeBrokerConfig, IoTDevice, EdgeSyncEvent,
	create_edge_broker, create_edge_orchestrator
)


class TestIoTDevice:
	"""Test IoT device functionality"""
	
	def test_iot_device_creation(self):
		"""Test IoT device creation"""
		device = IoTDevice(
			device_id="sensor_001",
			device_type="temperature_sensor",
			protocol=IoTProtocol.MQTT_5_0,
			location="Factory Floor 1",
			battery_level=85.5,
			signal_strength=-45.0,
			firmware_version="1.2.3"
		)
		
		assert device.device_id == "sensor_001"
		assert device.protocol == IoTProtocol.MQTT_5_0
		assert device.battery_level == 85.5
		assert device.is_online() == False  # No last_seen set
		assert device.is_low_battery() == False  # Above 20% threshold
	
	def test_device_online_status(self):
		"""Test device online status checking"""
		device = IoTDevice(
			device_id="sensor_002",
			device_type="pressure_sensor",
			protocol=IoTProtocol.COAP
		)
		
		# Device without last_seen should be offline
		assert device.is_online() == False
		
		# Device with recent last_seen should be online
		device.last_seen = datetime.utcnow()
		assert device.is_online() == True
		
		# Device with old last_seen should be offline
		device.last_seen = datetime.utcnow() - timedelta(minutes=10)
		assert device.is_online(timeout_seconds=300) == False
	
	def test_low_battery_detection(self):
		"""Test low battery detection"""
		device = IoTDevice(
			device_id="sensor_003",
			device_type="motion_sensor",
			protocol=IoTProtocol.LORAWAN,
			battery_level=15.0  # Low battery
		)
		
		assert device.is_low_battery() == True
		assert device.is_low_battery(threshold=10.0) == False
		
		device.battery_level = 50.0
		assert device.is_low_battery() == False


class TestMQTTAdapter:
	"""Test MQTT protocol adapter"""
	
	@pytest.fixture
	async def mqtt_adapter(self):
		"""Create MQTT adapter for testing"""
		adapter = MQTTAdapter()
		await adapter.initialize()
		yield adapter
		await adapter.shutdown()
	
	@pytest.mark.asyncio
	async def test_mqtt_adapter_initialization(self, mqtt_adapter):
		"""Test MQTT adapter initialization"""
		assert mqtt_adapter.protocol == IoTProtocol.MQTT_5_0
		assert mqtt_adapter.broker_host == "localhost"
		assert mqtt_adapter.broker_port == 1883
		assert len(mqtt_adapter.connected_devices) == 0
	
	@pytest.mark.asyncio
	async def test_device_connection(self, mqtt_adapter):
		"""Test MQTT device connection"""
		device = IoTDevice(
			device_id="mqtt_device_001",
			device_type="smart_meter",
			protocol=IoTProtocol.MQTT_5_0,
			location="Building A"
		)
		
		# Connect device
		success = await mqtt_adapter.connect_device(device)
		assert success == True
		assert device.device_id in mqtt_adapter.connected_devices
		assert device.last_seen is not None
		
		# Check topic subscription
		device_topic = f"devices/{device.device_id}/+"
		assert device_topic in mqtt_adapter.topics
		assert device.device_id in mqtt_adapter.topics[device_topic]
	
	@pytest.mark.asyncio
	async def test_device_disconnection(self, mqtt_adapter):
		"""Test MQTT device disconnection"""
		device = IoTDevice(
			device_id="mqtt_device_002",
			device_type="gateway",
			protocol=IoTProtocol.MQTT_5_0
		)
		
		# Connect then disconnect
		await mqtt_adapter.connect_device(device)
		assert device.device_id in mqtt_adapter.connected_devices
		
		success = await mqtt_adapter.disconnect_device(device.device_id)
		assert success == True
		assert device.device_id not in mqtt_adapter.connected_devices
	
	@pytest.mark.asyncio
	async def test_message_sending(self, mqtt_adapter):
		"""Test sending MQTT messages"""
		device = IoTDevice(
			device_id="mqtt_device_003",
			device_type="actuator",
			protocol=IoTProtocol.MQTT_5_0
		)
		
		await mqtt_adapter.connect_device(device)
		
		# Send message to connected device
		message = b'{"command": "turn_on", "value": true}'
		success = await mqtt_adapter.send_message(device.device_id, message)
		assert success == True
		
		# Try to send to non-existent device
		success = await mqtt_adapter.send_message("non_existent_device", message)
		assert success == False
	
	@pytest.mark.asyncio
	async def test_telemetry_publishing(self, mqtt_adapter):
		"""Test telemetry data publishing"""
		device = IoTDevice(
			device_id="mqtt_device_004",
			device_type="environmental_sensor",
			protocol=IoTProtocol.MQTT_5_0
		)
		
		await mqtt_adapter.connect_device(device)
		
		# Publish telemetry data
		telemetry_data = {
			"temperature": 23.5,
			"humidity": 65.2,
			"timestamp": datetime.utcnow().isoformat()
		}
		
		success = await mqtt_adapter.publish_telemetry(device.device_id, telemetry_data)
		assert success == True
		
		# Check that device last_seen was updated
		updated_device = mqtt_adapter.connected_devices[device.device_id]
		assert updated_device.last_seen is not None


class TestLoRaWANAdapter:
	"""Test LoRaWAN protocol adapter"""
	
	@pytest.fixture
	async def lorawan_adapter(self):
		"""Create LoRaWAN adapter for testing"""
		adapter = LoRaWANAdapter()
		await adapter.initialize()
		yield adapter
		await adapter.shutdown()
	
	@pytest.mark.asyncio
	async def test_lorawan_initialization(self, lorawan_adapter):
		"""Test LoRaWAN adapter initialization"""
		assert lorawan_adapter.protocol == IoTProtocol.LORAWAN
		assert lorawan_adapter.gateway_eui == "1234567890ABCDEF"
		assert len(lorawan_adapter.spreading_factors) == 6
	
	@pytest.mark.asyncio
	async def test_device_join_procedure(self, lorawan_adapter):
		"""Test LoRaWAN device join procedure"""
		device = IoTDevice(
			device_id="lora_device_001",
			device_type="soil_sensor",
			protocol=IoTProtocol.LORAWAN,
			metadata={"app_eui": "1111111111111111"}
		)
		
		# Simulate device join
		success = await lorawan_adapter.connect_device(device)
		assert success == True
		assert device.device_id in lorawan_adapter.connected_devices
		assert device.last_seen is not None
	
	@pytest.mark.asyncio
	async def test_uplink_reception(self, lorawan_adapter):
		"""Test LoRaWAN uplink message reception"""
		device = IoTDevice(
			device_id="lora_device_002",
			device_type="water_level_sensor",
			protocol=IoTProtocol.LORAWAN
		)
		
		await lorawan_adapter.connect_device(device)
		
		# Simulate uplink message
		payload = b'\x01\x23\x45\x67'  # Example sensor data
		rssi = -95.0  # Received Signal Strength Indicator
		snr = 8.5     # Signal-to-Noise Ratio
		
		success = await lorawan_adapter.receive_uplink(device.device_id, payload, rssi, snr)
		assert success == True
		
		# Check that device metadata was updated
		updated_device = lorawan_adapter.connected_devices[device.device_id]
		assert updated_device.signal_strength == rssi
		assert updated_device.metadata['snr'] == snr
	
	@pytest.mark.asyncio
	async def test_downlink_transmission(self, lorawan_adapter):
		"""Test LoRaWAN downlink message transmission"""
		device = IoTDevice(
			device_id="lora_device_003",
			device_type="valve_controller",
			protocol=IoTProtocol.LORAWAN
		)
		
		await lorawan_adapter.connect_device(device)
		
		# Update last_seen to simulate recent uplink
		device.last_seen = datetime.utcnow()
		
		# Send downlink message
		command = b'\xFF\x01'  # Example command
		success = await lorawan_adapter.send_message(device.device_id, command)
		assert success == True
		
		# Test sending when not in receive window
		device.last_seen = datetime.utcnow() - timedelta(seconds=10)
		success = await lorawan_adapter.send_message(device.device_id, command)
		assert success == False


class TestCoAPAdapter:
	"""Test CoAP protocol adapter"""
	
	@pytest.fixture
	async def coap_adapter(self):
		"""Create CoAP adapter for testing"""
		adapter = CoAPAdapter()
		await adapter.initialize()
		yield adapter
		await adapter.shutdown()
	
	@pytest.mark.asyncio
	async def test_coap_initialization(self, coap_adapter):
		"""Test CoAP adapter initialization"""
		assert coap_adapter.protocol == IoTProtocol.COAP
		assert coap_adapter.server_port == 5683
		assert len(coap_adapter.resources) == 0
	
	@pytest.mark.asyncio
	async def test_device_registration(self, coap_adapter):
		"""Test CoAP device registration"""
		device = IoTDevice(
			device_id="coap_device_001",
			device_type="air_quality_sensor",
			protocol=IoTProtocol.COAP,
			location="Office Building"
		)
		
		success = await coap_adapter.connect_device(device)
		assert success == True
		assert device.device_id in coap_adapter.connected_devices
		
		# Check that device resources were registered
		base_path = f"/devices/{device.device_id}"
		assert f"{base_path}/status" in coap_adapter.resources
		assert f"{base_path}/config" in coap_adapter.resources
		assert f"{base_path}/telemetry" in coap_adapter.resources
	
	@pytest.mark.asyncio
	async def test_observe_request_handling(self, coap_adapter):
		"""Test CoAP Observe request handling"""
		device = IoTDevice(
			device_id="coap_device_002",
			device_type="light_sensor",
			protocol=IoTProtocol.COAP
		)
		
		await coap_adapter.connect_device(device)
		
		# Handle observe request
		resource_path = f"/devices/{device.device_id}/telemetry"
		success = await coap_adapter.handle_observe_request(device.device_id, resource_path)
		assert success == True
		
		# Check observe relationship was established
		assert device.device_id in coap_adapter.observe_relationships[resource_path]


class TestEdgeBroker:
	"""Test edge broker functionality"""
	
	@pytest.fixture
	def edge_config(self):
		"""Create edge broker configuration"""
		return EdgeBrokerConfig(
			broker_id="test_edge_broker",
			deployment_type=EdgeDeploymentType.MINI_EDGE,
			location="Test Lab",
			region="us-west-1",
			protocols_enabled=[IoTProtocol.MQTT_5_0, IoTProtocol.COAP],
			max_connections=100,
			max_memory_mb=256,
			max_storage_gb=5
		)
	
	@pytest.fixture
	async def edge_broker(self, edge_config):
		"""Create edge broker for testing"""
		broker = EdgeBroker(edge_config)
		await broker.initialize()
		yield broker
		await broker.shutdown()
	
	@pytest.mark.asyncio
	async def test_edge_broker_initialization(self, edge_broker):
		"""Test edge broker initialization"""
		assert edge_broker.running == True
		assert edge_broker.deployment_type == EdgeDeploymentType.MINI_EDGE
		assert len(edge_broker.protocol_adapters) == 2  # MQTT and CoAP
		assert IoTProtocol.MQTT_5_0 in edge_broker.protocol_adapters
		assert IoTProtocol.COAP in edge_broker.protocol_adapters
	
	@pytest.mark.asyncio
	async def test_device_connection_management(self, edge_broker):
		"""Test device connection management"""
		device = IoTDevice(
			device_id="edge_device_001",
			device_type="multi_sensor",
			protocol=IoTProtocol.MQTT_5_0,
			location="Test Location",
			battery_level=90.0
		)
		
		# Connect device
		success = await edge_broker.connect_device(device)
		assert success == True
		assert device.device_id in edge_broker.connected_devices
		assert edge_broker.metrics['devices_connected'] == 1
		
		# Disconnect device
		success = await edge_broker.disconnect_device(device.device_id)
		assert success == True
		assert device.device_id not in edge_broker.connected_devices
		assert edge_broker.metrics['devices_connected'] == 0
	
	@pytest.mark.asyncio
	async def test_low_battery_alert_handling(self, edge_broker):
		"""Test low battery alert handling"""
		device = IoTDevice(
			device_id="edge_device_002",
			device_type="battery_sensor",
			protocol=IoTProtocol.MQTT_5_0,
			battery_level=15.0  # Low battery
		)
		
		# Connect device with low battery
		await edge_broker.connect_device(device)
		
		# Check that battery alert was generated
		assert edge_broker.metrics['battery_alerts'] > 0
		
		# Check that alert message was buffered
		alert_messages = [
			msg for msg in edge_broker.message_buffer
			if 'battery' in msg.topic and msg.headers.get('alert_type') == 'low_battery'
		]
		assert len(alert_messages) > 0
	
	@pytest.mark.asyncio
	async def test_cloud_synchronization(self, edge_broker):
		"""Test cloud synchronization"""
		# Add some messages to sync queue
		for i in range(3):
			message = MQMessage(
				topic=f"test.sync.{i}",
				payload=f"Test sync message {i}".encode(),
				tenant_id="edge_test",
				source_application="test_app"
			)
			edge_broker.sync_queue.append(message)
		
		# Perform cloud sync
		cloud_url = "https://test-cloud.example.com"
		sync_event = await edge_broker.sync_with_cloud(cloud_url)
		
		assert sync_event.success == True
		assert sync_event.message_count == 3
		assert sync_event.data_size_bytes > 0
		assert sync_event.sync_duration_ms > 0
		assert len(edge_broker.sync_queue) == 0  # Queue should be cleared
	
	@pytest.mark.asyncio
	async def test_edge_status_reporting(self, edge_broker):
		"""Test edge status reporting"""
		# Connect some devices
		for i in range(2):
			device = IoTDevice(
				device_id=f"status_device_{i}",
				device_type="test_sensor",
				protocol=IoTProtocol.MQTT_5_0,
				battery_level=80.0 - (i * 10)
			)
			await edge_broker.connect_device(device)
		
		# Get status
		status = await edge_broker.get_edge_status()
		
		assert status['broker_id'] == edge_broker.broker_id
		assert status['deployment_type'] == EdgeDeploymentType.MINI_EDGE.value
		assert status['running'] == True
		assert status['connected_devices'] == 2
		assert len(status['devices']) == 2
		assert len(status['protocols_enabled']) == 2


class TestEdgeOrchestrator:
	"""Test edge orchestrator functionality"""
	
	@pytest.fixture
	async def cloud_service(self):
		"""Create cloud MQEB service"""
		service = MQEBService()
		await service.initialize()
		yield service
		await service.shutdown()
	
	@pytest.fixture
	async def orchestrator(self, cloud_service):
		"""Create edge orchestrator"""
		return await create_edge_orchestrator(cloud_service)
	
	@pytest.mark.asyncio
	async def test_orchestrator_initialization(self, orchestrator):
		"""Test edge orchestrator initialization"""
		assert len(orchestrator.deployment_templates) == 4
		assert EdgeDeploymentType.MICRO_EDGE in orchestrator.deployment_templates
		assert EdgeDeploymentType.INDUSTRIAL_EDGE in orchestrator.deployment_templates
	
	@pytest.mark.asyncio
	async def test_edge_broker_deployment(self, orchestrator):
		"""Test deploying edge brokers"""
		# Deploy micro edge broker
		broker_id = await orchestrator.deploy_edge_broker(
			EdgeDeploymentType.MICRO_EDGE,
			"Factory Gateway 1",
			"us-west-2"
		)
		
		assert broker_id is not None
		assert broker_id in orchestrator.edge_brokers
		
		deployed_broker = orchestrator.edge_brokers[broker_id]
		assert deployed_broker.deployment_type == EdgeDeploymentType.MICRO_EDGE
		assert deployed_broker.config.location == "Factory Gateway 1"
		assert deployed_broker.running == True
	
	@pytest.mark.asyncio
	async def test_edge_broker_undeployment(self, orchestrator):
		"""Test undeploying edge brokers"""
		# Deploy then undeploy
		broker_id = await orchestrator.deploy_edge_broker(
			EdgeDeploymentType.REGIONAL_EDGE,
			"Regional Hub",
			"eu-central-1"
		)
		
		assert broker_id in orchestrator.edge_brokers
		
		success = await orchestrator.undeploy_edge_broker(broker_id)
		assert success == True
		assert broker_id not in orchestrator.edge_brokers
	
	@pytest.mark.asyncio
	async def test_iot_device_registration(self, orchestrator):
		"""Test IoT device registration with edge brokers"""
		# Deploy edge broker
		broker_id = await orchestrator.deploy_edge_broker(
			EdgeDeploymentType.MINI_EDGE,
			"Test Facility",
			"us-east-1"
		)
		
		# Register IoT device
		device = IoTDevice(
			device_id="orchestrator_device_001",
			device_type="temperature_humidity_sensor",
			protocol=IoTProtocol.MQTT_5_0,
			location="Test Facility - Room A"
		)
		
		success = await orchestrator.register_iot_device(broker_id, device)
		assert success == True
		
		# Check device is registered with edge broker
		edge_broker = orchestrator.edge_brokers[broker_id]
		assert device.device_id in edge_broker.connected_devices
	
	@pytest.mark.asyncio
	async def test_bulk_synchronization(self, orchestrator):
		"""Test synchronizing all edge brokers"""
		# Deploy multiple edge brokers
		broker_ids = []
		for i in range(2):
			broker_id = await orchestrator.deploy_edge_broker(
				EdgeDeploymentType.MINI_EDGE,
				f"Site {i+1}",
				"us-west-1"
			)
			broker_ids.append(broker_id)
		
		# Add some data to sync
		for broker_id in broker_ids:
			broker = orchestrator.edge_brokers[broker_id]
			for j in range(2):
				message = MQMessage(
					topic=f"sync.test.{j}",
					payload=f"Sync test data from {broker_id}".encode(),
					tenant_id="sync_test",
					source_application="orchestrator_test"
				)
				broker.sync_queue.append(message)
		
		# Sync all brokers
		sync_events = await orchestrator.sync_all_brokers()
		
		assert len(sync_events) == 2
		assert all(event.success for event in sync_events)
		assert all(event.message_count == 2 for event in sync_events)
	
	@pytest.mark.asyncio
	async def test_orchestrator_status(self, orchestrator):
		"""Test orchestrator status reporting"""
		# Deploy some edge brokers and register devices
		for i in range(2):
			broker_id = await orchestrator.deploy_edge_broker(
				EdgeDeploymentType.MINI_EDGE,
				f"Test Site {i+1}",
				"us-central-1"
			)
			
			# Register devices with each broker
			for j in range(2):
				device = IoTDevice(
					device_id=f"status_test_device_{i}_{j}",
					device_type="multi_sensor",
					protocol=IoTProtocol.MQTT_5_0
				)
				await orchestrator.register_iot_device(broker_id, device)
		
		# Get orchestrator status
		status = await orchestrator.get_orchestrator_status()
		
		assert status['edge_brokers'] == 2
		assert status['total_connected_devices'] == 4
		assert len(status['brokers']) == 2
		
		for broker_info in status['brokers']:
			assert broker_info['deployment_type'] == EdgeDeploymentType.MINI_EDGE.value
			assert broker_info['connected_devices'] == 2
			assert broker_info['running'] == True


class TestDeploymentTemplates:
	"""Test deployment templates for different edge types"""
	
	@pytest.mark.asyncio
	async def test_micro_edge_deployment(self):
		"""Test micro edge deployment template"""
		cloud_service = MQEBService()
		await cloud_service.initialize()
		
		try:
			orchestrator = await create_edge_orchestrator(cloud_service)
			
			# Deploy micro edge with custom config
			broker_id = await orchestrator.deploy_edge_broker(
				EdgeDeploymentType.MICRO_EDGE,
				"Solar Panel Controller",
				"remote-site-1",
				custom_config={
					'battery_powered': True,
					'sync_strategy': SynchronizationStrategy.BATTERY_AWARE
				}
			)
			
			edge_broker = orchestrator.edge_brokers[broker_id]
			assert edge_broker.config.max_connections == 50  # Template value
			assert edge_broker.config.max_memory_mb == 128   # Template value
			assert edge_broker.config.battery_powered == True  # Custom value
			
		finally:
			await cloud_service.shutdown()
	
	@pytest.mark.asyncio
	async def test_industrial_edge_deployment(self):
		"""Test industrial edge deployment template"""
		cloud_service = MQEBService()
		await cloud_service.initialize()
		
		try:
			orchestrator = await create_edge_orchestrator(cloud_service)
			
			# Deploy industrial edge
			broker_id = await orchestrator.deploy_edge_broker(
				EdgeDeploymentType.INDUSTRIAL_EDGE,
				"Manufacturing Floor Control",
				"factory-1"
			)
			
			edge_broker = orchestrator.edge_brokers[broker_id]
			assert edge_broker.config.max_connections == 1000  # Industrial template
			assert IoTProtocol.MODBUS_TCP in edge_broker.config.protocols_enabled
			assert IoTProtocol.OPC_UA in edge_broker.config.protocols_enabled
			assert edge_broker.config.sync_strategy == SynchronizationStrategy.IMMEDIATE
			
		finally:
			await cloud_service.shutdown()


if __name__ == "__main__":
	# Run tests if script is executed directly
	pytest.main([__file__, "-v"])