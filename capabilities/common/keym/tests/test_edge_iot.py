#!/usr/bin/env python3
"""
APG Key Management - Edge Computing & IoT Tests
Comprehensive tests for edge computing and IoT device key management

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import ssl

from ..edge_iot_integration import (
	IoTDeviceManager, IoTDevice, EdgeNode, EdgeCryptoService,
	DeviceType, EdgeLocation, ConnectivityType, SecurityLevel,
	create_iot_device_manager
)
from ..service import KeyManagementService


@pytest.fixture
async def mock_service():
	"""Mock key management service"""
	service = Mock(spec=KeyManagementService)
	service._log_audit_event = AsyncMock()
	return service


@pytest.fixture
def mqtt_config():
	"""MQTT configuration for testing"""
	return {
		'mqtt': {
			'broker': 'localhost',
			'port': 1883,
			'username': 'test_user',
			'password': 'test_pass',
			'use_tls': False
		}
	}


@pytest.fixture
async def iot_manager(mock_service, mqtt_config):
	"""IoT device manager instance"""
	with patch('paho.mqtt.client.Client'):
		manager = IoTDeviceManager(mock_service)
		await manager.initialize(mqtt_config)
		yield manager
		await manager.shutdown()


@pytest.fixture
def sample_device_spec():
	"""Sample device specification"""
	return {
		'name': 'Temperature Sensor 001',
		'type': 'sensor',
		'manufacturer': 'AcmeSensors',
		'model': 'TempPro-2000',
		'firmware_version': '1.2.3',
		'location': 'factory_floor',
		'connectivity': ['wifi', 'bluetooth'],
		'security_level': 'enhanced',
		'tenant_id': 'test-tenant',
		'supports_hardware_crypto': True,
		'has_secure_element': True,
		'supported_algorithms': ['AES-256-GCM', 'ChaCha20-Poly1305'],
		'max_key_size': 256
	}


@pytest.fixture
def sample_edge_node_spec():
	"""Sample edge node specification"""
	return {
		'name': 'Factory Floor Gateway',
		'location': 'factory_floor',
		'cpu_cores': 8,
		'memory_gb': 16,
		'storage_gb': 128,
		'has_gpu': True,
		'has_tpm': True,
		'ip_address': '192.168.1.100',
		'max_device_capacity': 200,
		'tenant_id': 'test-tenant',
		'supports_offline_crypto': True
	}


class TestIoTDevice:
	"""Test IoTDevice class"""
	
	def test_device_creation(self):
		"""Test IoT device creation"""
		device = IoTDevice(
			device_name="Test Sensor",
			device_type=DeviceType.SENSOR,
			manufacturer="TestCorp",
			edge_location=EdgeLocation.FACTORY_FLOOR,
			security_level=SecurityLevel.ENHANCED
		)
		
		assert device.device_name == "Test Sensor"
		assert device.device_type == DeviceType.SENSOR
		assert device.manufacturer == "TestCorp"
		assert device.edge_location == EdgeLocation.FACTORY_FLOOR
		assert device.security_level == SecurityLevel.ENHANCED
		assert device.device_id is not None
		assert device.status == "active"
		assert isinstance(device.created_at, datetime)
	
	def test_device_to_dict(self):
		"""Test device serialization"""
		device = IoTDevice(
			device_name="Test Device",
			device_type=DeviceType.CAMERA,
			connectivity=[ConnectivityType.WIFI, ConnectivityType.CELLULAR_5G]
		)
		
		device_dict = device.to_dict()
		
		assert device_dict['device_name'] == "Test Device"
		assert device_dict['device_type'] == "camera"
		assert device_dict['connectivity'] == ["wifi", "cellular_5g"]
		assert 'device_id' in device_dict
		assert 'created_at' in device_dict


class TestEdgeNode:
	"""Test EdgeNode class"""
	
	def test_edge_node_creation(self):
		"""Test edge node creation"""
		node = EdgeNode(
			node_name="Test Gateway",
			location=EdgeLocation.WAREHOUSE,
			cpu_cores=4,
			memory_gb=8,
			has_tpu=True
		)
		
		assert node.node_name == "Test Gateway"
		assert node.location == EdgeLocation.WAREHOUSE
		assert node.cpu_cores == 4
		assert node.memory_gb == 8
		assert node.node_id is not None
		assert node.status == "online"
		assert len(node.managed_devices) == 0


class TestEdgeCryptoService:
	"""Test EdgeCryptoService class"""
	
	@pytest.fixture
	def crypto_service(self):
		"""Edge crypto service instance"""
		return EdgeCryptoService()
	
	@pytest.mark.asyncio
	async def test_generate_aes_key(self, crypto_service):
		"""Test AES key generation"""
		key = await crypto_service.generate_device_key("AES-256-GCM", 256)
		
		assert isinstance(key, bytes)
		assert len(key) == 32  # 256 bits / 8
	
	@pytest.mark.asyncio
	async def test_generate_chacha20_key(self, crypto_service):
		"""Test ChaCha20 key generation"""
		key = await crypto_service.generate_device_key("ChaCha20-Poly1305", 256)
		
		assert isinstance(key, bytes)
		assert len(key) == 32  # ChaCha20 uses 256-bit keys
	
	@pytest.mark.asyncio
	async def test_generate_rsa_key(self, crypto_service):
		"""Test RSA key generation"""
		key = await crypto_service.generate_device_key("RSA-2048", 2048)
		
		assert isinstance(key, bytes)
		assert b'-----BEGIN PRIVATE KEY-----' in key
	
	@pytest.mark.asyncio
	async def test_aes_encrypt_decrypt(self, crypto_service):
		"""Test AES encryption/decryption"""
		key = await crypto_service.generate_device_key("AES-256-GCM", 256)
		plaintext = b"Hello, IoT World!"
		
		# Encrypt
		encrypted = await crypto_service.encrypt_data(plaintext, key, "AES-256-GCM")
		
		assert 'ciphertext' in encrypted
		assert 'iv' in encrypted
		assert 'tag' in encrypted
		assert encrypted['algorithm'] == 'AES-GCM'
		
		# Decrypt
		decrypted = await crypto_service.decrypt_data(encrypted, key, "AES-256-GCM")
		assert decrypted == plaintext
	
	@pytest.mark.asyncio
	async def test_chacha20_encrypt_decrypt(self, crypto_service):
		"""Test ChaCha20 encryption/decryption"""
		key = await crypto_service.generate_device_key("ChaCha20-Poly1305", 256)
		plaintext = b"IoT sensor data"
		
		# Encrypt
		encrypted = await crypto_service.encrypt_data(plaintext, key, "ChaCha20-Poly1305")
		
		assert 'ciphertext' in encrypted
		assert 'nonce' in encrypted
		assert encrypted['algorithm'] == 'ChaCha20-Poly1305'
		
		# Decrypt
		decrypted = await crypto_service.decrypt_data(encrypted, key, "ChaCha20-Poly1305")
		assert decrypted == plaintext
	
	@pytest.mark.asyncio
	async def test_unsupported_algorithm(self, crypto_service):
		"""Test handling of unsupported algorithms"""
		with pytest.raises(ValueError, match="Unsupported algorithm"):
			await crypto_service.generate_device_key("UNKNOWN-ALGO", 256)
		
		with pytest.raises(ValueError, match="Unsupported algorithm"):
			await crypto_service.encrypt_data(b"data", b"key", "UNKNOWN-ALGO")


class TestIoTDeviceManager:
	"""Test IoTDeviceManager class"""
	
	@pytest.mark.asyncio
	async def test_manager_initialization(self, mock_service, mqtt_config):
		"""Test IoT device manager initialization"""
		with patch('paho.mqtt.client.Client') as mock_mqtt:
			mock_client = MagicMock()
			mock_mqtt.return_value = mock_client
			
			manager = IoTDeviceManager(mock_service)
			await manager.initialize(mqtt_config)
			
			assert manager._is_running is True
			assert manager.mqtt_config == mqtt_config['mqtt']
			mock_client.connect.assert_called_once()
			
			await manager.shutdown()
	
	@pytest.mark.asyncio
	async def test_register_device(self, iot_manager, sample_device_spec):
		"""Test device registration"""
		device = await iot_manager.register_device(sample_device_spec, "test-user")
		
		assert device.device_name == "Temperature Sensor 001"
		assert device.device_type == DeviceType.SENSOR
		assert device.manufacturer == "AcmeSensors"
		assert device.security_level == SecurityLevel.ENHANCED
		assert device.device_id in iot_manager.devices
		assert len(device.device_keys) > 0  # Keys should be provisioned
		
		# Check audit logging was called
		iot_manager.service._log_audit_event.assert_called()
	
	@pytest.mark.asyncio
	async def test_register_edge_node(self, iot_manager, sample_edge_node_spec):
		"""Test edge node registration"""
		node = await iot_manager.register_edge_node(sample_edge_node_spec, "test-user")
		
		assert node.node_name == "Factory Floor Gateway"
		assert node.location == EdgeLocation.FACTORY_FLOOR
		assert node.cpu_cores == 8
		assert node.has_tpm is True
		assert node.node_id in iot_manager.edge_nodes
		
		# Check audit logging was called
		iot_manager.service._log_audit_event.assert_called()
	
	@pytest.mark.asyncio
	async def test_assign_device_to_edge_node(self, iot_manager, sample_device_spec, sample_edge_node_spec):
		"""Test assigning device to edge node"""
		# Register device and edge node
		device = await iot_manager.register_device(sample_device_spec)
		node = await iot_manager.register_edge_node(sample_edge_node_spec)
		
		# Assign device to node
		await iot_manager.assign_device_to_edge_node(device.device_id, node.node_id, "test-user")
		
		assert device.device_id in node.managed_devices
		
		# Check audit logging was called
		assert iot_manager.service._log_audit_event.call_count >= 3  # 2 registrations + 1 assignment
	
	@pytest.mark.asyncio
	async def test_assign_device_capacity_exceeded(self, iot_manager, sample_device_spec, sample_edge_node_spec):
		"""Test device assignment when edge node is at capacity"""
		# Create node with capacity of 1
		sample_edge_node_spec['max_device_capacity'] = 1
		
		# Register devices and edge node
		device1 = await iot_manager.register_device(sample_device_spec)
		device2_spec = sample_device_spec.copy()
		device2_spec['name'] = 'Device 2'
		device2 = await iot_manager.register_device(device2_spec)
		node = await iot_manager.register_edge_node(sample_edge_node_spec)
		
		# Assign first device (should succeed)
		await iot_manager.assign_device_to_edge_node(device1.device_id, node.node_id)
		
		# Assign second device (should fail)
		with pytest.raises(ValueError, match="at capacity"):
			await iot_manager.assign_device_to_edge_node(device2.device_id, node.node_id)
	
	@pytest.mark.asyncio
	async def test_rotate_device_keys(self, iot_manager, sample_device_spec):
		"""Test device key rotation"""
		device = await iot_manager.register_device(sample_device_spec)
		original_keys = device.device_keys.copy()
		original_rotation_time = device.last_key_rotation
		
		# Wait a small amount to ensure timestamps differ
		await asyncio.sleep(0.01)
		
		# Rotate keys
		new_keys = await iot_manager.rotate_device_keys(device.device_id, "test-user")
		
		assert len(new_keys) == len(device.supported_algorithms)
		assert device.last_key_rotation > original_rotation_time
		
		# Keys should be different
		for algorithm in device.supported_algorithms:
			if algorithm in original_keys:
				assert device.device_keys[algorithm] != original_keys[algorithm]
	
	@pytest.mark.asyncio
	async def test_rotate_nonexistent_device_keys(self, iot_manager):
		"""Test key rotation for non-existent device"""
		with pytest.raises(ValueError, match="Device not found"):
			await iot_manager.rotate_device_keys("non-existent-device")
	
	@pytest.mark.asyncio
	async def test_get_device_status(self, iot_manager, sample_device_spec):
		"""Test getting device status"""
		device = await iot_manager.register_device(sample_device_spec)
		
		status = await iot_manager.get_device_status(device.device_id)
		
		assert status['device_id'] == device.device_id
		assert status['status'] == device.status
		assert 'last_seen' in status
		assert 'keys_provisioned' in status
		assert 'security_level' in status
		assert status['keys_provisioned'] > 0
	
	@pytest.mark.asyncio
	async def test_get_devices_by_location(self, iot_manager, sample_device_spec):
		"""Test getting devices by location"""
		# Register device at factory floor
		device1 = await iot_manager.register_device(sample_device_spec)
		
		# Register device at warehouse
		warehouse_spec = sample_device_spec.copy()
		warehouse_spec['location'] = 'warehouse'
		warehouse_spec['name'] = 'Warehouse Sensor'
		device2 = await iot_manager.register_device(warehouse_spec)
		
		# Get devices by location
		factory_devices = await iot_manager.get_devices_by_location(EdgeLocation.FACTORY_FLOOR)
		warehouse_devices = await iot_manager.get_devices_by_location(EdgeLocation.WAREHOUSE)
		
		assert len(factory_devices) == 1
		assert len(warehouse_devices) == 1
		assert factory_devices[0].device_id == device1.device_id
		assert warehouse_devices[0].device_id == device2.device_id
	
	@pytest.mark.asyncio
	async def test_get_edge_node_devices(self, iot_manager, sample_device_spec, sample_edge_node_spec):
		"""Test getting devices managed by edge node"""
		# Register components
		device1 = await iot_manager.register_device(sample_device_spec)
		device2_spec = sample_device_spec.copy()
		device2_spec['name'] = 'Device 2'
		device2 = await iot_manager.register_device(device2_spec)
		node = await iot_manager.register_edge_node(sample_edge_node_spec)
		
		# Assign devices to node
		await iot_manager.assign_device_to_edge_node(device1.device_id, node.node_id)
		await iot_manager.assign_device_to_edge_node(device2.device_id, node.node_id)
		
		# Get node devices
		node_devices = await iot_manager.get_edge_node_devices(node.node_id)
		
		assert len(node_devices) == 2
		device_ids = {device.device_id for device in node_devices}
		assert device1.device_id in device_ids
		assert device2.device_id in device_ids
	
	@pytest.mark.asyncio
	async def test_get_security_summary(self, iot_manager, sample_device_spec):
		"""Test getting security summary"""
		# Register devices with different security levels
		enhanced_spec = sample_device_spec.copy()
		enhanced_spec['security_level'] = 'enhanced'
		enhanced_spec['has_secure_element'] = True
		
		standard_spec = sample_device_spec.copy()
		standard_spec['security_level'] = 'standard'
		standard_spec['has_secure_element'] = False
		standard_spec['name'] = 'Standard Device'
		
		await iot_manager.register_device(enhanced_spec)
		await iot_manager.register_device(standard_spec)
		
		summary = await iot_manager.get_security_summary()
		
		assert summary['total_devices'] == 2
		assert summary['devices_by_security_level']['enhanced'] == 1
		assert summary['devices_by_security_level']['standard'] == 1
		assert summary['devices_with_secure_element'] == 1
		assert summary['secure_element_percentage'] == 50.0
		assert summary['rotation_compliance'] == 100.0  # Just registered, no rotation needed yet
	
	@pytest.mark.asyncio
	async def test_mqtt_message_handling(self, iot_manager, sample_device_spec):
		"""Test MQTT message handling"""
		device = await iot_manager.register_device(sample_device_spec)
		
		# Test heartbeat message
		heartbeat_payload = {
			'timestamp': datetime.utcnow().isoformat(),
			'battery_level': 85.0,
			'signal_strength': -45.0
		}
		
		await iot_manager._handle_device_heartbeat(device.device_id, heartbeat_payload)
		
		# Check device was updated
		updated_device = iot_manager.devices[device.device_id]
		assert updated_device.battery_level == 85.0
		assert updated_device.signal_strength == -45.0
		assert updated_device.status == "active"
	
	@pytest.mark.asyncio
	async def test_key_request_handling(self, iot_manager, sample_device_spec):
		"""Test device key request handling"""
		device = await iot_manager.register_device(sample_device_spec)
		original_keys = device.device_keys.copy()
		
		# Test key request
		key_request_payload = {
			'algorithm': 'AES-256-GCM',
			'reason': 'scheduled_rotation'
		}
		
		await iot_manager._handle_key_request(device.device_id, key_request_payload)
		
		# Keys should be rotated
		updated_device = iot_manager.devices[device.device_id]
		assert updated_device.device_keys != original_keys
	
	@pytest.mark.asyncio
	async def test_device_status_update_handling(self, iot_manager, sample_device_spec):
		"""Test device status update handling"""
		device = await iot_manager.register_device(sample_device_spec)
		
		# Test status update
		status_payload = {
			'cpu_usage': 45.2,
			'memory_usage': 67.8,
			'temperature': 42.5
		}
		
		await iot_manager._handle_device_status(device.device_id, status_payload)
		
		# Check metadata was updated
		updated_device = iot_manager.devices[device.device_id]
		assert updated_device.metadata['cpu_usage'] == 45.2
		assert updated_device.metadata['memory_usage'] == 67.8
		assert updated_device.metadata['temperature'] == 42.5


class TestEdgeComputingScenarios:
	"""Test edge computing scenarios"""
	
	@pytest.mark.asyncio
	async def test_factory_floor_deployment(self, iot_manager):
		"""Test complete factory floor deployment scenario"""
		# Register edge gateway
		gateway_spec = {
			'name': 'Factory Gateway',
			'location': 'factory_floor',
			'cpu_cores': 8,
			'memory_gb': 32,
			'has_tpm': True,
			'max_device_capacity': 50,
			'tenant_id': 'factory-tenant'
		}
		gateway = await iot_manager.register_edge_node(gateway_spec)
		
		# Register various IoT devices
		sensor_devices = []
		for i in range(5):
			sensor_spec = {
				'name': f'Temperature Sensor {i+1}',
				'type': 'sensor',
				'location': 'factory_floor',
				'security_level': 'enhanced',
				'tenant_id': 'factory-tenant',
				'connectivity': ['wifi'],
				'supported_algorithms': ['AES-256-GCM']
			}
			sensor = await iot_manager.register_device(sensor_spec)
			sensor_devices.append(sensor)
			
			# Assign to gateway
			await iot_manager.assign_device_to_edge_node(sensor.device_id, gateway.node_id)
		
		# Register industrial actuators
		actuator_spec = {
			'name': 'Robotic Arm Controller',
			'type': 'actuator',
			'location': 'factory_floor',
			'security_level': 'critical',
			'tenant_id': 'factory-tenant',
			'connectivity': ['ethernet'],
			'supported_algorithms': ['AES-256-GCM', 'RSA-2048']
		}
		actuator = await iot_manager.register_device(actuator_spec)
		await iot_manager.assign_device_to_edge_node(actuator.device_id, gateway.node_id)
		
		# Verify deployment
		assert len(gateway.managed_devices) == 6
		
		factory_devices = await iot_manager.get_devices_by_location(EdgeLocation.FACTORY_FLOOR)
		assert len(factory_devices) == 6
		
		security_summary = await iot_manager.get_security_summary()
		assert security_summary['total_devices'] == 6
		assert security_summary['devices_by_security_level']['enhanced'] == 5
		assert security_summary['devices_by_security_level']['critical'] == 1
	
	@pytest.mark.asyncio
	async def test_smart_home_deployment(self, iot_manager):
		"""Test smart home deployment scenario"""
		# Register home gateway
		gateway_spec = {
			'name': 'Smart Home Hub',
			'location': 'home',
			'cpu_cores': 4,
			'memory_gb': 4,
			'has_tpm': False,
			'max_device_capacity': 20,
			'tenant_id': 'home-tenant'
		}
		gateway = await iot_manager.register_edge_node(gateway_spec)
		
		# Register smart home devices
		devices = [
			{
				'name': 'Smart Thermostat',
				'type': 'sensor',
				'location': 'home',
				'security_level': 'standard',
				'connectivity': ['wifi'],
				'supported_algorithms': ['AES-128-GCM']
			},
			{
				'name': 'Security Camera',
				'type': 'camera',
				'location': 'home',
				'security_level': 'enhanced',
				'connectivity': ['wifi'],
				'supported_algorithms': ['AES-256-GCM']
			},
			{
				'name': 'Smart Lock',
				'type': 'actuator',
				'location': 'home',
				'security_level': 'critical',
				'connectivity': ['zigbee'],
				'supported_algorithms': ['AES-256-GCM', 'ChaCha20-Poly1305']
			}
		]
		
		registered_devices = []
		for device_spec in devices:
			device_spec['tenant_id'] = 'home-tenant'
			device = await iot_manager.register_device(device_spec)
			registered_devices.append(device)
			await iot_manager.assign_device_to_edge_node(device.device_id, gateway.node_id)
		
		# Verify deployment
		assert len(gateway.managed_devices) == 3
		
		home_devices = await iot_manager.get_devices_by_location(EdgeLocation.HOME)
		assert len(home_devices) == 3
		
		# Verify different connectivity types
		connectivity_types = set()
		for device in registered_devices:
			connectivity_types.update(device.connectivity)
		
		expected_types = {ConnectivityType.WIFI, ConnectivityType.ZIGBEE}
		assert connectivity_types == expected_types
	
	@pytest.mark.asyncio
	async def test_vehicle_deployment(self, iot_manager):
		"""Test vehicle IoT deployment scenario"""
		# Register vehicle edge node
		vehicle_spec = {
			'name': 'Connected Vehicle ECU',
			'location': 'vehicle',
			'cpu_cores': 6,
			'memory_gb': 8,
			'has_gpu': True,
			'has_tpm': True,
			'max_device_capacity': 15,
			'tenant_id': 'vehicle-tenant'
		}
		vehicle_node = await iot_manager.register_edge_node(vehicle_spec)
		
		# Register vehicle sensors and systems
		vehicle_devices = [
			{
				'name': 'Engine Control Unit',
				'type': 'industrial',
				'security_level': 'critical',
				'connectivity': ['ethernet'],
				'supported_algorithms': ['AES-256-GCM', 'RSA-2048']
			},
			{
				'name': 'GPS Navigation System',
				'type': 'sensor',
				'security_level': 'enhanced',
				'connectivity': ['cellular_5g', 'wifi'],
				'supported_algorithms': ['AES-256-GCM']
			},
			{
				'name': 'Dashboard Camera',
				'type': 'camera',
				'security_level': 'standard',
				'connectivity': ['ethernet'],
				'supported_algorithms': ['ChaCha20-Poly1305']
			}
		]
		
		for device_spec in vehicle_devices:
			device_spec['location'] = 'vehicle'
			device_spec['tenant_id'] = 'vehicle-tenant'
			device = await iot_manager.register_device(device_spec)
			await iot_manager.assign_device_to_edge_node(device.device_id, vehicle_node.node_id)
		
		# Verify vehicle deployment
		vehicle_devices_list = await iot_manager.get_devices_by_location(EdgeLocation.VEHICLE)
		assert len(vehicle_devices_list) == 3
		
		# Check critical system security
		critical_devices = [d for d in vehicle_devices_list if d.security_level == SecurityLevel.CRITICAL]
		assert len(critical_devices) == 1
		assert critical_devices[0].device_name == 'Engine Control Unit'


@pytest.mark.asyncio
async def test_factory_function(mock_service, mqtt_config):
	"""Test IoT device manager factory function"""
	with patch('paho.mqtt.client.Client'):
		manager = await create_iot_device_manager(mock_service, mqtt_config)
		
		assert isinstance(manager, IoTDeviceManager)
		assert manager._is_running is True
		assert manager.service == mock_service
		
		await manager.shutdown()


class TestErrorHandling:
	"""Test error handling scenarios"""
	
	@pytest.mark.asyncio
	async def test_device_not_found_errors(self, iot_manager):
		"""Test handling of device not found errors"""
		# Test device status for non-existent device
		with pytest.raises(ValueError, match="Device not found"):
			await iot_manager.get_device_status("non-existent-device")
		
		# Test key rotation for non-existent device
		with pytest.raises(ValueError, match="Device not found"):
			await iot_manager.rotate_device_keys("non-existent-device")
	
	@pytest.mark.asyncio
	async def test_node_not_found_errors(self, iot_manager, sample_device_spec):
		"""Test handling of edge node not found errors"""
		device = await iot_manager.register_device(sample_device_spec)
		
		# Test assignment to non-existent node
		with pytest.raises(ValueError, match="Edge node not found"):
			await iot_manager.assign_device_to_edge_node(device.device_id, "non-existent-node")
	
	@pytest.mark.asyncio
	async def test_mqtt_connection_failure(self, mock_service, mqtt_config):
		"""Test handling of MQTT connection failures"""
		with patch('paho.mqtt.client.Client') as mock_mqtt:
			mock_client = MagicMock()
			mock_client.connect.side_effect = Exception("Connection failed")
			mock_mqtt.return_value = mock_client
			
			manager = IoTDeviceManager(mock_service)
			
			# Should not raise exception, just log error
			await manager.initialize(mqtt_config)
			
			await manager.shutdown()


if __name__ == "__main__":
	pytest.main([__file__, "-v"])