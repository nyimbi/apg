#!/usr/bin/env python3
"""
APG Message Queue Event Bus (MQEB) - Edge Computing & IoT Integration
Edge-native message brokers and IoT protocol support

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import secrets
from uuid_extensions import uuid7str

from .models import MQMessage, MessagePriority, ProtocolType
from .service import MQEBService


class EdgeDeploymentType(str, Enum):
	"""Types of edge deployments"""
	MICRO_EDGE = "micro_edge"          # IoT gateways, sensors
	MINI_EDGE = "mini_edge"            # Edge servers, local data centers
	REGIONAL_EDGE = "regional_edge"    # Regional distribution centers
	INDUSTRIAL_EDGE = "industrial_edge" # Factory floors, manufacturing


class IoTProtocol(str, Enum):
	"""Supported IoT protocols"""
	MQTT_5_0 = "mqtt_5_0"
	COAP = "coap"
	LORAWAN = "lorawan"
	ZIGBEE = "zigbee"
	BLUETOOTH_LE = "bluetooth_le"
	MODBUS_TCP = "modbus_tcp"
	OPC_UA = "opc_ua"
	HTTP_IOT = "http_iot"


class SynchronizationStrategy(str, Enum):
	"""Data synchronization strategies"""
	IMMEDIATE = "immediate"        # Sync immediately when connected
	PERIODIC = "periodic"          # Sync at regular intervals
	THRESHOLD = "threshold"        # Sync when data reaches threshold
	INTELLIGENT = "intelligent"    # AI-driven sync optimization
	BATTERY_AWARE = "battery_aware" # Optimize for battery life


@dataclass
class EdgeBrokerConfig:
	"""Configuration for edge broker deployment"""
	broker_id: str
	deployment_type: EdgeDeploymentType
	location: str
	region: str
	protocols_enabled: List[IoTProtocol]
	max_connections: int = 1000
	max_memory_mb: int = 512
	max_storage_gb: int = 10
	battery_powered: bool = False
	sync_strategy: SynchronizationStrategy = SynchronizationStrategy.INTELLIGENT
	offline_buffer_size: int = 10000
	compression_enabled: bool = True
	encryption_required: bool = True


@dataclass
class IoTDevice:
	"""IoT device representation"""
	device_id: str
	device_type: str
	protocol: IoTProtocol
	location: Optional[str] = None
	last_seen: Optional[datetime] = None
	battery_level: Optional[float] = None  # 0-100%
	signal_strength: Optional[float] = None  # dBm
	firmware_version: Optional[str] = None
	capabilities: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)
	
	def is_online(self, timeout_seconds: int = 300) -> bool:
		"""Check if device is considered online"""
		if self.last_seen is None:
			return False
		return (datetime.utcnow() - self.last_seen).total_seconds() < timeout_seconds
	
	def is_low_battery(self, threshold: float = 20.0) -> bool:
		"""Check if device has low battery"""
		return self.battery_level is not None and self.battery_level < threshold


@dataclass
class EdgeSyncEvent:
	"""Event for data synchronization between edge and cloud"""
	event_id: str
	edge_broker_id: str
	sync_type: str  # upstream, downstream, bidirectional
	data_size_bytes: int
	message_count: int
	compression_ratio: Optional[float]
	sync_duration_ms: float
	success: bool
	error_details: Optional[str] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)


class IoTProtocolAdapter:
	"""Base class for IoT protocol adapters"""
	
	def __init__(self, protocol: IoTProtocol):
		self.protocol = protocol
		self.connected_devices: Dict[str, IoTDevice] = {}
		self.message_handlers: List[Callable] = []
		self.outbound_messages: Dict[str, deque[bytes]] = defaultdict(deque)
		self.inbound_messages: Dict[str, deque[bytes]] = defaultdict(deque)
		self.logger = logging.getLogger(f'mqeb.iot.{protocol.value}')
	
	async def initialize(self) -> None:
		"""Initialize protocol adapter"""
		self.logger.info(f"Initializing {self.protocol.value} protocol adapter")
	
	async def shutdown(self) -> None:
		"""Shutdown protocol adapter"""
		self.logger.info(f"Shutting down {self.protocol.value} protocol adapter")
	
	async def connect_device(self, device: IoTDevice) -> bool:
		"""Connect IoT device"""
		if device.protocol != self.protocol:
			self.logger.error(
				f"Device {device.device_id} uses {device.protocol.value}, not {self.protocol.value}"
			)
			return False
		self.connected_devices[device.device_id] = device
		device.last_seen = datetime.utcnow()
		self.outbound_messages.setdefault(device.device_id, deque())
		self.inbound_messages.setdefault(device.device_id, deque())
		self.logger.info(f"{self.protocol.value} device {device.device_id} connected")
		return True
	
	async def disconnect_device(self, device_id: str) -> bool:
		"""Disconnect IoT device"""
		if device_id not in self.connected_devices:
			return False
		del self.connected_devices[device_id]
		self.outbound_messages.pop(device_id, None)
		self.inbound_messages.pop(device_id, None)
		self.logger.info(f"{self.protocol.value} device {device_id} disconnected")
		return True
	
	async def send_message(self, device_id: str, message: bytes) -> bool:
		"""Send message to IoT device"""
		if device_id not in self.connected_devices:
			return False
		self.outbound_messages[device_id].append(bytes(message))
		self.connected_devices[device_id].last_seen = datetime.utcnow()
		self.logger.debug(f"Queued {len(message)} bytes for {self.protocol.value} device {device_id}")
		return True
	
	async def receive_message(self, device_id: str) -> Optional[bytes]:
		"""Receive message from IoT device"""
		if device_id not in self.connected_devices:
			return None
		if not self.inbound_messages[device_id]:
			return None
		message = self.inbound_messages[device_id].popleft()
		self.connected_devices[device_id].last_seen = datetime.utcnow()
		await self._handle_received_message(device_id, message)
		return message

	async def inject_received_message(self, device_id: str, message: bytes) -> bool:
		"""Inject an inbound device message for local/offline protocol execution."""
		if device_id not in self.connected_devices:
			return False
		self.inbound_messages[device_id].append(bytes(message))
		return True
	
	def add_message_handler(self, handler: Callable):
		"""Add message handler"""
		self.message_handlers.append(handler)
	
	async def _handle_received_message(self, device_id: str, message: bytes):
		"""Handle received message from IoT device"""
		for handler in self.message_handlers:
			try:
				await handler(device_id, message, self.protocol)
			except Exception as e:
				self.logger.error(f"Message handler error: {e}")


class MQTTAdapter(IoTProtocolAdapter):
	"""MQTT 5.0 protocol adapter"""
	
	def __init__(self):
		super().__init__(IoTProtocol.MQTT_5_0)
		self.broker_host = "localhost"
		self.broker_port = 1883
		self.client = None
		self.topics: Dict[str, List[str]] = defaultdict(list)  # topic -> device_ids
	
	async def initialize(self) -> None:
		await super().initialize()
		# Initialize MQTT client
		# In production, would use actual MQTT library like paho-mqtt
		self.logger.info(f"MQTT broker initialized on {self.broker_host}:{self.broker_port}")
	
	async def connect_device(self, device: IoTDevice) -> bool:
		"""Connect MQTT device"""
		try:
			self.connected_devices[device.device_id] = device
			device.last_seen = datetime.utcnow()
			
			# Subscribe to device topics
			device_topic = f"devices/{device.device_id}/+"
			self.topics[device_topic].append(device.device_id)
			
			self.logger.info(f"MQTT device {device.device_id} connected")
			return True
		except Exception as e:
			self.logger.error(f"Failed to connect MQTT device {device.device_id}: {e}")
			return False
	
	async def disconnect_device(self, device_id: str) -> bool:
		"""Disconnect MQTT device"""
		try:
			if device_id in self.connected_devices:
				del self.connected_devices[device_id]
				
				# Unsubscribe from device topics
				for topic, device_ids in self.topics.items():
					if device_id in device_ids:
						device_ids.remove(device_id)
				
				self.logger.info(f"MQTT device {device_id} disconnected")
			return True
		except Exception as e:
			self.logger.error(f"Failed to disconnect MQTT device {device_id}: {e}")
			return False
	
	async def send_message(self, device_id: str, message: bytes) -> bool:
		"""Send message to MQTT device"""
		try:
			if not await super().send_message(device_id, message):
				return False
			
			topic = f"devices/{device_id}/commands"
			# In production, would publish to actual MQTT broker
			self.logger.debug(f"Publishing to MQTT topic {topic}: {len(message)} bytes")
			return True
		except Exception as e:
			self.logger.error(f"Failed to send MQTT message to {device_id}: {e}")
			return False
	
	async def receive_message(self, device_id: str) -> Optional[bytes]:
		"""Receive message from MQTT device."""
		return await super().receive_message(device_id)
	
	async def publish_telemetry(self, device_id: str, telemetry_data: Dict[str, Any]) -> bool:
		"""Publish telemetry data from IoT device"""
		try:
			topic = f"devices/{device_id}/telemetry"
			payload = json.dumps(telemetry_data).encode()
			
			# Update device last seen
			if device_id in self.connected_devices:
				self.connected_devices[device_id].last_seen = datetime.utcnow()
			
			# Process telemetry through message handlers
			await self._handle_received_message(device_id, payload)
			return True
		except Exception as e:
			self.logger.error(f"Failed to publish telemetry for {device_id}: {e}")
			return False


class LoRaWANAdapter(IoTProtocolAdapter):
	"""LoRaWAN protocol adapter"""
	
	def __init__(self):
		super().__init__(IoTProtocol.LORAWAN)
		self.gateway_eui = "1234567890ABCDEF"
		self.network_server = None
		self.spreading_factors = [7, 8, 9, 10, 11, 12]  # SF7-SF12
	
	async def initialize(self) -> None:
		await super().initialize()
		self.logger.info(f"LoRaWAN gateway {self.gateway_eui} initialized")
	
	async def connect_device(self, device: IoTDevice) -> bool:
		"""Connect LoRaWAN device (join procedure)"""
		try:
			# Simulate OTAA (Over-The-Air Activation)
			dev_eui = device.device_id
			app_eui = device.metadata.get('app_eui', '0000000000000000')
			
			self.connected_devices[device.device_id] = device
			device.last_seen = datetime.utcnow()
			
			self.logger.info(f"LoRaWAN device {dev_eui} joined network")
			return True
		except Exception as e:
			self.logger.error(f"Failed to join LoRaWAN device {device.device_id}: {e}")
			return False
	
	async def send_message(self, device_id: str, message: bytes) -> bool:
		"""Send downlink message to LoRaWAN device"""
		try:
			if device_id not in self.connected_devices:
				return False
			
			device = self.connected_devices[device_id]
			
			# LoRaWAN has limited downlink opportunities
			# Check if device is in receive window
			if not self._is_in_receive_window(device):
				self.logger.warning(f"LoRaWAN device {device_id} not in receive window")
				return False
			
			# Simulate downlink transmission
			self.outbound_messages[device_id].append(bytes(message))
			self.logger.debug(f"Sending LoRaWAN downlink to {device_id}: {len(message)} bytes")
			return True
		except Exception as e:
			self.logger.error(f"Failed to send LoRaWAN message to {device_id}: {e}")
			return False
	
	def _is_in_receive_window(self, device: IoTDevice) -> bool:
		"""Check if device is in receive window (simplified)"""
		# LoRaWAN devices have specific receive windows after uplink
		# This is a simplified check
		if device.last_seen is None:
			return False
		
		time_since_uplink = (datetime.utcnow() - device.last_seen).total_seconds()
		return time_since_uplink < 2  # 2-second receive window
	
	async def receive_uplink(self, device_id: str, payload: bytes, rssi: float, snr: float) -> bool:
		"""Receive uplink message from LoRaWAN device"""
		try:
			if device_id in self.connected_devices:
				device = self.connected_devices[device_id]
				device.last_seen = datetime.utcnow()
				device.signal_strength = rssi
				device.metadata.update({'snr': snr})
			
			# Process uplink message
			await self._handle_received_message(device_id, payload)
			return True
		except Exception as e:
			self.logger.error(f"Failed to process LoRaWAN uplink from {device_id}: {e}")
			return False


class CoAPAdapter(IoTProtocolAdapter):
	"""CoAP (Constrained Application Protocol) adapter"""
	
	def __init__(self):
		super().__init__(IoTProtocol.COAP)
		self.server_port = 5683
		self.resources: Dict[str, Any] = {}
		self.observe_relationships: Dict[str, List[str]] = defaultdict(list)
	
	async def initialize(self) -> None:
		await super().initialize()
		self.logger.info(f"CoAP server initialized on port {self.server_port}")
	
	async def connect_device(self, device: IoTDevice) -> bool:
		"""Register CoAP device"""
		try:
			self.connected_devices[device.device_id] = device
			device.last_seen = datetime.utcnow()
			
			# Register device resources
			base_path = f"/devices/{device.device_id}"
			self.resources[f"{base_path}/status"] = {"type": "status"}
			self.resources[f"{base_path}/config"] = {"type": "config"}
			self.resources[f"{base_path}/telemetry"] = {"type": "telemetry"}
			
			self.logger.info(f"CoAP device {device.device_id} registered")
			return True
		except Exception as e:
			self.logger.error(f"Failed to register CoAP device {device.device_id}: {e}")
			return False
	
	async def send_message(self, device_id: str, message: bytes) -> bool:
		"""Send CoAP message to device"""
		try:
			if not await super().send_message(device_id, message):
				return False
			
			# Simulate CoAP PUT or POST request
			resource_path = f"/devices/{device_id}/commands"
			self.logger.debug(f"Sending CoAP message to {resource_path}: {len(message)} bytes")
			return True
		except Exception as e:
			self.logger.error(f"Failed to send CoAP message to {device_id}: {e}")
			return False
	
	async def handle_observe_request(self, device_id: str, resource_path: str) -> bool:
		"""Handle CoAP Observe request"""
		try:
			self.observe_relationships[resource_path].append(device_id)
			self.logger.debug(f"CoAP observe registered: {device_id} -> {resource_path}")
			return True
		except Exception as e:
			self.logger.error(f"Failed to handle CoAP observe: {e}")
			return False


class EdgeBroker:
	"""Edge-native message broker for IoT and edge computing"""
	
	def __init__(self, config: EdgeBrokerConfig):
		self.config = config
		self.broker_id = config.broker_id
		self.deployment_type = config.deployment_type
		
		# Protocol adapters
		self.protocol_adapters: Dict[IoTProtocol, IoTProtocolAdapter] = {}
		self.connected_devices: Dict[str, IoTDevice] = {}
		
		# Message handling
		self.message_buffer: deque = deque(maxlen=config.offline_buffer_size)
		self.sync_queue: deque = deque()
		self.cloud_connection_status = False
		
		# Metrics
		self.metrics = {
			'messages_processed': 0,
			'devices_connected': 0,
			'sync_events': 0,
			'battery_alerts': 0,
			'protocol_errors': 0
		}
		
		# Background tasks
		self._background_tasks: Set[asyncio.Task] = set()
		self.running = False
		
		self.logger = logging.getLogger(f'mqeb.edge.{self.broker_id}')
	
	async def initialize(self) -> None:
		"""Initialize edge broker"""
		self.logger.info(f"Initializing edge broker {self.broker_id} ({self.deployment_type.value})")
		
		# Initialize protocol adapters
		await self._initialize_protocol_adapters()
		
		# Start background tasks
		await self._start_background_tasks()
		
		self.running = True
		self.logger.info(f"Edge broker {self.broker_id} initialized with {len(self.protocol_adapters)} protocols")
	
	async def shutdown(self) -> None:
		"""Shutdown edge broker"""
		self.logger.info(f"Shutting down edge broker {self.broker_id}")
		self.running = False
		
		# Shutdown protocol adapters
		for adapter in self.protocol_adapters.values():
			await adapter.shutdown()
		
		# Cancel background tasks
		for task in self._background_tasks:
			task.cancel()
		
		await asyncio.gather(*self._background_tasks, return_exceptions=True)
		self.logger.info(f"Edge broker {self.broker_id} shut down")
	
	async def _initialize_protocol_adapters(self) -> None:
		"""Initialize IoT protocol adapters"""
		for protocol in self.config.protocols_enabled:
			try:
				if protocol == IoTProtocol.MQTT_5_0:
					adapter = MQTTAdapter()
				elif protocol == IoTProtocol.LORAWAN:
					adapter = LoRaWANAdapter()
				elif protocol == IoTProtocol.COAP:
					adapter = CoAPAdapter()
				else:
					# Generic adapter for other protocols
					adapter = IoTProtocolAdapter(protocol)
				
				# Add message handler
				adapter.add_message_handler(self._handle_iot_message)
				
				await adapter.initialize()
				self.protocol_adapters[protocol] = adapter
				
			except Exception as e:
				self.logger.error(f"Failed to initialize {protocol.value} adapter: {e}")
	
	async def connect_device(self, device: IoTDevice) -> bool:
		"""Connect IoT device to edge broker"""
		try:
			adapter = self.protocol_adapters.get(device.protocol)
			if not adapter:
				self.logger.error(f"No adapter for protocol {device.protocol.value}")
				return False
			
			success = await adapter.connect_device(device)
			if success:
				self.connected_devices[device.device_id] = device
				self.metrics['devices_connected'] += 1
				
				# Check battery level if applicable
				if device.is_low_battery():
					await self._handle_low_battery_alert(device)
				
				self.logger.info(f"Device {device.device_id} connected via {device.protocol.value}")
			
			return success
		except Exception as e:
			self.logger.error(f"Failed to connect device {device.device_id}: {e}")
			return False
	
	async def disconnect_device(self, device_id: str) -> bool:
		"""Disconnect IoT device"""
		try:
			device = self.connected_devices.get(device_id)
			if not device:
				return False
			
			adapter = self.protocol_adapters.get(device.protocol)
			if adapter:
				await adapter.disconnect_device(device_id)
			
			del self.connected_devices[device_id]
			self.metrics['devices_connected'] -= 1
			
			self.logger.info(f"Device {device_id} disconnected")
			return True
		except Exception as e:
			self.logger.error(f"Failed to disconnect device {device_id}: {e}")
			return False
	
	async def _handle_iot_message(self, device_id: str, message: bytes, protocol: IoTProtocol):
		"""Handle incoming IoT message"""
		try:
			# Create MQEB message from IoT message
			mqeb_message = MQMessage(
				topic=f"iot.{protocol.value}.{device_id}",
				payload=message,
				tenant_id="edge",
				source_application=f"edge_broker_{self.broker_id}",
				priority=MessagePriority.NORMAL,
				headers={
					'device_id': device_id,
					'protocol': protocol.value,
					'edge_broker': self.broker_id,
					'timestamp': datetime.utcnow().isoformat()
				}
			)
			
			# Buffer message for processing
			self.message_buffer.append(mqeb_message)
			self.metrics['messages_processed'] += 1
			
			# Queue for cloud sync if connected
			if self.cloud_connection_status:
				self.sync_queue.append(mqeb_message)
			
			self.logger.debug(f"Processed IoT message from {device_id}")
			
		except Exception as e:
			self.logger.error(f"Failed to handle IoT message from {device_id}: {e}")
			self.metrics['protocol_errors'] += 1
	
	async def _handle_low_battery_alert(self, device: IoTDevice):
		"""Handle low battery alert for IoT device"""
		try:
			alert_message = MQMessage(
				topic=f"iot.alerts.battery.{device.device_id}",
				payload=json.dumps({
					'device_id': device.device_id,
					'battery_level': device.battery_level,
					'alert_type': 'low_battery',
					'location': device.location,
					'timestamp': datetime.utcnow().isoformat()
				}).encode(),
				tenant_id="edge",
				source_application=f"edge_broker_{self.broker_id}",
				priority=MessagePriority.HIGH,
				headers={
					'alert_type': 'low_battery',
					'device_id': device.device_id,
					'edge_broker': self.broker_id
				}
			)
			
			self.message_buffer.append(alert_message)
			self.metrics['battery_alerts'] += 1
			
			self.logger.warning(f"Low battery alert for device {device.device_id}: {device.battery_level}%")
			
		except Exception as e:
			self.logger.error(f"Failed to handle low battery alert: {e}")
	
	async def sync_with_cloud(self, cloud_broker_url: str) -> EdgeSyncEvent:
		"""Synchronize data with cloud broker"""
		sync_event = EdgeSyncEvent(
			event_id=f"sync_{uuid7str()}",
			edge_broker_id=self.broker_id,
			sync_type="upstream",
			data_size_bytes=0,
			message_count=0,
			sync_duration_ms=0,
			success=False
		)
		
		start_time = time.time()
		
		try:
			# Prepare messages for sync
			messages_to_sync = list(self.sync_queue)
			if not messages_to_sync:
				sync_event.success = True
				return sync_event
			
			# Calculate data size
			total_size = sum(msg.size_bytes() for msg in messages_to_sync)
			sync_event.data_size_bytes = total_size
			sync_event.message_count = len(messages_to_sync)
			
			# Compress data if enabled
			if self.config.compression_enabled:
				# Simulate compression
				sync_event.compression_ratio = 0.3  # 70% compression
			
			# Simulate cloud sync (in production, would use actual HTTP/gRPC client)
			await asyncio.sleep(0.1)  # Simulate network latency
			
			# Clear sync queue on success
			self.sync_queue.clear()
			sync_event.success = True
			self.metrics['sync_events'] += 1
			
			self.logger.info(f"Synced {len(messages_to_sync)} messages to cloud ({total_size} bytes)")
			
		except Exception as e:
			sync_event.error_details = str(e)
			self.logger.error(f"Cloud sync failed: {e}")
		
		finally:
			sync_event.sync_duration_ms = (time.time() - start_time) * 1000
		
		return sync_event
	
	async def _start_background_tasks(self) -> None:
		"""Start background tasks"""
		
		# Device health monitoring
		task = asyncio.create_task(self._device_health_monitoring_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Cloud synchronization
		task = asyncio.create_task(self._cloud_sync_loop())
		self._background_tasks.add(task)
		task.add_done_callback(self._background_tasks.discard)
		
		# Battery monitoring
		if self.config.battery_powered:
			task = asyncio.create_task(self._battery_optimization_loop())
			self._background_tasks.add(task)
			task.add_done_callback(self._background_tasks.discard)
	
	async def _device_health_monitoring_loop(self) -> None:
		"""Background task to monitor device health"""
		while self.running:
			try:
				await asyncio.sleep(60)  # Check every minute
				
				offline_devices = []
				for device_id, device in self.connected_devices.items():
					if not device.is_online():
						offline_devices.append(device_id)
						self.logger.warning(f"Device {device_id} appears offline")
				
				# Clean up offline devices
				for device_id in offline_devices:
					await self.disconnect_device(device_id)
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Device health monitoring error: {e}")
	
	async def _cloud_sync_loop(self) -> None:
		"""Background task for cloud synchronization"""
		while self.running:
			try:
				# Determine sync interval based on strategy
				sync_interval = self._calculate_sync_interval()
				await asyncio.sleep(sync_interval)
				
				# Check if sync is needed
				if len(self.sync_queue) > 0:
					cloud_url = "https://cloud.mqeb.example.com"  # Would be configurable
					sync_event = await self.sync_with_cloud(cloud_url)
					
					if not sync_event.success:
						self.logger.error(f"Cloud sync failed: {sync_event.error_details}")
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Cloud sync loop error: {e}")
	
	def _calculate_sync_interval(self) -> int:
		"""Calculate sync interval based on strategy"""
		strategy = self.config.sync_strategy
		
		if strategy == SynchronizationStrategy.IMMEDIATE:
			return 10  # 10 seconds
		elif strategy == SynchronizationStrategy.PERIODIC:
			return 300  # 5 minutes
		elif strategy == SynchronizationStrategy.THRESHOLD:
			# Sync more frequently when queue is full
			queue_ratio = len(self.sync_queue) / max(1, self.config.offline_buffer_size)
			return max(30, int(300 * (1 - queue_ratio)))  # 30s to 5min
		elif strategy == SynchronizationStrategy.INTELLIGENT:
			# AI-driven optimization (simplified)
			base_interval = 300
			device_count_factor = min(2.0, len(self.connected_devices) / 100)
			return max(60, int(base_interval / device_count_factor))
		elif strategy == SynchronizationStrategy.BATTERY_AWARE:
			# Longer intervals to save battery
			return 900  # 15 minutes
		
		return 300  # Default 5 minutes
	
	async def _battery_optimization_loop(self) -> None:
		"""Background task for battery optimization"""
		while self.running:
			try:
				await asyncio.sleep(1800)  # Check every 30 minutes
				
				# Implement battery-saving measures
				low_battery_devices = [
					device for device in self.connected_devices.values()
					if device.is_low_battery(30.0)  # 30% threshold
				]
				
				if low_battery_devices:
					self.logger.info(f"Optimizing for {len(low_battery_devices)} low-battery devices")
					
					# Reduce sync frequency
					if self.config.sync_strategy != SynchronizationStrategy.BATTERY_AWARE:
						self.config.sync_strategy = SynchronizationStrategy.BATTERY_AWARE
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"Battery optimization error: {e}")
	
	async def get_edge_status(self) -> Dict[str, Any]:
		"""Get current edge broker status"""
		return {
			'broker_id': self.broker_id,
			'deployment_type': self.deployment_type.value,
			'location': self.config.location,
			'running': self.running,
			'cloud_connected': self.cloud_connection_status,
			'protocols_enabled': [p.value for p in self.config.protocols_enabled],
			'connected_devices': len(self.connected_devices),
			'message_buffer_size': len(self.message_buffer),
			'sync_queue_size': len(self.sync_queue),
			'metrics': self.metrics.copy(),
			'devices': [
				{
					'device_id': device.device_id,
					'protocol': device.protocol.value,
					'battery_level': device.battery_level,
					'last_seen': device.last_seen.isoformat() if device.last_seen else None,
					'online': device.is_online()
				}
				for device in self.connected_devices.values()
			]
		}


class EdgeOrchestrator:
	"""Orchestrates multiple edge brokers and cloud connectivity"""
	
	def __init__(self, cloud_service: MQEBService):
		self.cloud_service = cloud_service
		self.edge_brokers: Dict[str, EdgeBroker] = {}
		self.sync_events: List[EdgeSyncEvent] = []
		
		# Edge deployment management
		self.deployment_templates: Dict[EdgeDeploymentType, Dict] = self._initialize_deployment_templates()
		
		self.logger = logging.getLogger('mqeb.edge_orchestrator')
	
	def _initialize_deployment_templates(self) -> Dict[EdgeDeploymentType, Dict]:
		"""Initialize deployment templates for different edge types"""
		return {
			EdgeDeploymentType.MICRO_EDGE: {
				'max_connections': 50,
				'max_memory_mb': 128,
				'max_storage_gb': 1,
				'protocols': [IoTProtocol.MQTT_5_0, IoTProtocol.COAP],
				'sync_strategy': SynchronizationStrategy.BATTERY_AWARE
			},
			EdgeDeploymentType.MINI_EDGE: {
				'max_connections': 500,
				'max_memory_mb': 512,
				'max_storage_gb': 10,
				'protocols': [IoTProtocol.MQTT_5_0, IoTProtocol.COAP, IoTProtocol.LORAWAN],
				'sync_strategy': SynchronizationStrategy.INTELLIGENT
			},
			EdgeDeploymentType.REGIONAL_EDGE: {
				'max_connections': 5000,
				'max_memory_mb': 2048,
				'max_storage_gb': 100,
				'protocols': [proto for proto in IoTProtocol],
				'sync_strategy': SynchronizationStrategy.THRESHOLD
			},
			EdgeDeploymentType.INDUSTRIAL_EDGE: {
				'max_connections': 1000,
				'max_memory_mb': 1024,
				'max_storage_gb': 50,
				'protocols': [IoTProtocol.MODBUS_TCP, IoTProtocol.OPC_UA, IoTProtocol.MQTT_5_0],
				'sync_strategy': SynchronizationStrategy.IMMEDIATE
			}
		}
	
	async def deploy_edge_broker(self, deployment_type: EdgeDeploymentType,
								location: str, region: str,
								custom_config: Optional[Dict] = None) -> str:
		"""Deploy new edge broker"""
		try:
			# Get template configuration
			template = self.deployment_templates[deployment_type]
			
			# Create broker configuration
			config = EdgeBrokerConfig(
				broker_id=f"edge_{deployment_type.value}_{uuid7str()[:8]}",
				deployment_type=deployment_type,
				location=location,
				region=region,
				protocols_enabled=template['protocols'],
				max_connections=template['max_connections'],
				max_memory_mb=template['max_memory_mb'],
				max_storage_gb=template['max_storage_gb'],
				sync_strategy=template['sync_strategy']
			)
			
			# Apply custom configuration
			if custom_config:
				for key, value in custom_config.items():
					if hasattr(config, key):
						setattr(config, key, value)
			
			# Create and initialize edge broker
			edge_broker = EdgeBroker(config)
			await edge_broker.initialize()
			
			# Register edge broker
			self.edge_brokers[config.broker_id] = edge_broker
			
			self.logger.info(f"Deployed edge broker {config.broker_id} at {location}")
			return config.broker_id
			
		except Exception as e:
			self.logger.error(f"Failed to deploy edge broker: {e}")
			raise
	
	async def undeploy_edge_broker(self, broker_id: str) -> bool:
		"""Undeploy edge broker"""
		try:
			if broker_id in self.edge_brokers:
				edge_broker = self.edge_brokers[broker_id]
				await edge_broker.shutdown()
				del self.edge_brokers[broker_id]
				self.logger.info(f"Undeployed edge broker {broker_id}")
				return True
			return False
		except Exception as e:
			self.logger.error(f"Failed to undeploy edge broker {broker_id}: {e}")
			return False
	
	async def register_iot_device(self, broker_id: str, device: IoTDevice) -> bool:
		"""Register IoT device with edge broker"""
		try:
			edge_broker = self.edge_brokers.get(broker_id)
			if not edge_broker:
				self.logger.error(f"Edge broker {broker_id} not found")
				return False
			
			return await edge_broker.connect_device(device)
		except Exception as e:
			self.logger.error(f"Failed to register device {device.device_id}: {e}")
			return False
	
	async def sync_all_brokers(self) -> List[EdgeSyncEvent]:
		"""Synchronize all edge brokers with cloud"""
		sync_events = []
		
		for broker_id, edge_broker in self.edge_brokers.items():
			try:
				cloud_url = f"https://cloud.mqeb.{edge_broker.config.region}.example.com"
				sync_event = await edge_broker.sync_with_cloud(cloud_url)
				sync_events.append(sync_event)
				self.sync_events.append(sync_event)
			except Exception as e:
				self.logger.error(f"Failed to sync edge broker {broker_id}: {e}")
		
		return sync_events
	
	async def get_orchestrator_status(self) -> Dict[str, Any]:
		"""Get edge orchestrator status"""
		total_devices = sum(len(broker.connected_devices) for broker in self.edge_brokers.values())
		total_messages = sum(broker.metrics['messages_processed'] for broker in self.edge_brokers.values())
		
		return {
			'edge_brokers': len(self.edge_brokers),
			'total_connected_devices': total_devices,
			'total_messages_processed': total_messages,
			'recent_sync_events': len([e for e in self.sync_events if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]),
			'brokers': [
				{
					'broker_id': broker.broker_id,
					'deployment_type': broker.deployment_type.value,
					'location': broker.config.location,
					'connected_devices': len(broker.connected_devices),
					'running': broker.running
				}
				for broker in self.edge_brokers.values()
			]
		}


# Factory functions
async def create_edge_broker(config: EdgeBrokerConfig) -> EdgeBroker:
	"""Create and initialize edge broker"""
	broker = EdgeBroker(config)
	await broker.initialize()
	return broker


async def create_edge_orchestrator(cloud_service: MQEBService) -> EdgeOrchestrator:
	"""Create edge orchestrator"""
	return EdgeOrchestrator(cloud_service)


# Export components
__all__ = [
	'EdgeBroker', 'EdgeOrchestrator', 'IoTProtocolAdapter', 'MQTTAdapter', 'LoRaWANAdapter', 'CoAPAdapter',
	'EdgeDeploymentType', 'IoTProtocol', 'SynchronizationStrategy',
	'EdgeBrokerConfig', 'IoTDevice', 'EdgeSyncEvent',
	'create_edge_broker', 'create_edge_orchestrator'
]
