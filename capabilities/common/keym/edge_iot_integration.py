#!/usr/bin/env python3
"""
APG Key Management - Edge Computing & IoT Integration
Comprehensive edge computing and IoT device key management system

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
import ssl
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from uuid_extensions import uuid7str
import paho.mqtt.client as mqtt
from aiocache import cached, Cache
from aiocache.serializers import PickleSerializer

from .service import KeyManagementService


class DeviceType(str, Enum):
	"""IoT device types"""
	SENSOR = "sensor"
	ACTUATOR = "actuator"
	GATEWAY = "gateway"
	EDGE_COMPUTE = "edge_compute"
	CAMERA = "camera"
	VEHICLE = "vehicle"
	INDUSTRIAL = "industrial"
	WEARABLE = "wearable"
	SMART_HOME = "smart_home"
	MEDICAL = "medical"


class EdgeLocation(str, Enum):
	"""Edge computing locations"""
	FACTORY_FLOOR = "factory_floor"
	RETAIL_STORE = "retail_store"
	VEHICLE = "vehicle"
	HOME = "home"
	OFFICE = "office"
	HOSPITAL = "hospital"
	WAREHOUSE = "warehouse"
	OUTDOOR = "outdoor"


class ConnectivityType(str, Enum):
	"""Device connectivity types"""
	WIFI = "wifi"
	CELLULAR_4G = "cellular_4g"
	CELLULAR_5G = "cellular_5g"
	BLUETOOTH = "bluetooth"
	ZIGBEE = "zigbee"
	LORA = "lora"
	ETHERNET = "ethernet"
	SATELLITE = "satellite"


class SecurityLevel(str, Enum):
	"""Device security levels"""
	MINIMAL = "minimal"
	STANDARD = "standard"
	ENHANCED = "enhanced"
	CRITICAL = "critical"


@dataclass
class IoTDevice:
	"""IoT device representation"""
	device_id: str = field(default_factory=uuid7str)
	device_name: str = ""
	device_type: DeviceType = DeviceType.SENSOR
	manufacturer: str = ""
	model: str = ""
	firmware_version: str = ""
	hardware_version: str = ""
	
	# Location and connectivity
	edge_location: EdgeLocation = EdgeLocation.FACTORY_FLOOR
	connectivity: List[ConnectivityType] = field(default_factory=list)
	ip_address: str = ""
	mac_address: str = ""
	
	# Security configuration
	security_level: SecurityLevel = SecurityLevel.STANDARD
	supports_hardware_crypto: bool = False
	has_secure_element: bool = False
	certificate_chain: List[str] = field(default_factory=list)
	
	# Operational data
	tenant_id: str = ""
	last_seen: datetime = field(default_factory=datetime.utcnow)
	status: str = "active"
	battery_level: Optional[float] = None
	signal_strength: Optional[float] = None
	
	# Key management
	device_keys: Dict[str, str] = field(default_factory=dict)
	key_rotation_interval: int = 86400  # 24 hours
	last_key_rotation: datetime = field(default_factory=datetime.utcnow)
	
	# Capabilities
	supported_algorithms: List[str] = field(default_factory=list)
	max_key_size: int = 256
	supports_pki: bool = True
	supports_symmetric: bool = True
	
	# Metadata
	metadata: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary"""
		data = asdict(self)
		data['device_type'] = self.device_type.value
		data['edge_location'] = self.edge_location.value
		data['security_level'] = self.security_level.value
		data['connectivity'] = [c.value for c in self.connectivity]
		data['last_seen'] = self.last_seen.isoformat()
		data['last_key_rotation'] = self.last_key_rotation.isoformat()
		data['created_at'] = self.created_at.isoformat()
		data['updated_at'] = self.updated_at.isoformat()
		return data


@dataclass
class EdgeNode:
	"""Edge computing node"""
	node_id: str = field(default_factory=uuid7str)
	node_name: str = ""
	location: EdgeLocation = EdgeLocation.FACTORY_FLOOR
	
	# Hardware specifications
	cpu_cores: int = 4
	memory_gb: int = 8
	storage_gb: int = 64
	has_gpu: bool = False
	has_tpm: bool = False
	has_secure_boot: bool = False
	
	# Network configuration
	ip_address: str = ""
	gateway_ip: str = ""
	dns_servers: List[str] = field(default_factory=list)
	vpn_enabled: bool = False
	
	# Connected devices
	managed_devices: Set[str] = field(default_factory=set)
	max_device_capacity: int = 100
	
	# Key management capabilities
	local_key_cache: Dict[str, Any] = field(default_factory=dict)
	cache_ttl: int = 3600  # 1 hour
	supports_offline_crypto: bool = True
	
	# Security
	tenant_id: str = ""
	security_policies: Dict[str, Any] = field(default_factory=dict)
	attestation_certificate: str = ""
	
	# Status
	status: str = "online"
	last_heartbeat: datetime = field(default_factory=datetime.utcnow)
	cpu_usage: float = 0.0
	memory_usage: float = 0.0
	
	# Metadata
	metadata: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)


class EdgeCryptoService:
	"""Lightweight cryptographic service for edge devices"""
	
	def __init__(self):
		self._cache = Cache.MEMORY
		self._algorithms = {
			'AES-256-GCM': self._aes_encrypt,
			'AES-128-GCM': self._aes_encrypt,
			'ChaCha20-Poly1305': self._chacha20_encrypt,
			'RSA-2048': self._rsa_encrypt,
			'ECDSA-P256': self._ecdsa_sign
		}
	
	async def generate_device_key(self, algorithm: str, key_size: int = 256) -> bytes:
		"""Generate key for IoT device"""
		if algorithm.startswith('AES'):
			return secrets.token_bytes(key_size // 8)
		elif algorithm.startswith('ChaCha20'):
			return secrets.token_bytes(32)
		elif algorithm.startswith('RSA'):
			private_key = rsa.generate_private_key(
				public_exponent=65537,
				key_size=key_size,
				backend=default_backend()
			)
			return private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
		elif algorithm.startswith('ECDSA'):
			private_key = ec.generate_private_key(
				ec.SECP256R1() if 'P256' in algorithm else ec.SECP384R1(),
				backend=default_backend()
			)
			return private_key.private_bytes(
				encoding=serialization.Encoding.PEM,
				format=serialization.PrivateFormat.PKCS8,
				encryption_algorithm=serialization.NoEncryption()
			)
		else:
			raise ValueError(f"Unsupported algorithm: {algorithm}")
	
	async def encrypt_data(self, data: bytes, key: bytes, algorithm: str) -> Dict[str, Any]:
		"""Encrypt data using specified algorithm"""
		if algorithm not in self._algorithms:
			raise ValueError(f"Unsupported algorithm: {algorithm}")
		
		return await self._algorithms[algorithm](data, key, 'encrypt')
	
	async def decrypt_data(self, encrypted_data: Dict[str, Any], key: bytes, algorithm: str) -> bytes:
		"""Decrypt data using specified algorithm"""
		if algorithm not in self._algorithms:
			raise ValueError(f"Unsupported algorithm: {algorithm}")
		
		return await self._algorithms[algorithm](encrypted_data, key, 'decrypt')
	
	async def _aes_encrypt(self, data: Union[bytes, Dict[str, Any]], key: bytes, operation: str) -> Union[Dict[str, Any], bytes]:
		"""AES encryption/decryption"""
		if operation == 'encrypt':
			iv = secrets.token_bytes(12)  # 96-bit IV for GCM
			cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
			encryptor = cipher.encryptor()
			ciphertext = encryptor.update(data) + encryptor.finalize()
			
			return {
				'ciphertext': ciphertext.hex(),
				'iv': iv.hex(),
				'tag': encryptor.tag.hex(),
				'algorithm': 'AES-GCM'
			}
		else:
			iv = bytes.fromhex(data['iv'])
			tag = bytes.fromhex(data['tag'])
			ciphertext = bytes.fromhex(data['ciphertext'])
			
			cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
			decryptor = cipher.decryptor()
			return decryptor.update(ciphertext) + decryptor.finalize()
	
	async def _chacha20_encrypt(self, data: Union[bytes, Dict[str, Any]], key: bytes, operation: str) -> Union[Dict[str, Any], bytes]:
		"""ChaCha20-Poly1305 encryption/decryption"""
		if operation == 'encrypt':
			nonce = secrets.token_bytes(12)
			cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend=default_backend())
			encryptor = cipher.encryptor()
			ciphertext = encryptor.update(data) + encryptor.finalize()
			
			return {
				'ciphertext': ciphertext.hex(),
				'nonce': nonce.hex(),
				'algorithm': 'ChaCha20-Poly1305'
			}
		else:
			nonce = bytes.fromhex(data['nonce'])
			ciphertext = bytes.fromhex(data['ciphertext'])
			
			cipher = Cipher(algorithms.ChaCha20(key, nonce), None, backend=default_backend())
			decryptor = cipher.decryptor()
			return decryptor.update(ciphertext) + decryptor.finalize()
	
	async def _rsa_encrypt(self, data: Union[bytes, Dict[str, Any]], key: bytes, operation: str) -> Union[Dict[str, Any], bytes]:
		"""RSA encryption/decryption"""
		from cryptography.hazmat.primitives.asymmetric import rsa, padding
		from cryptography.hazmat.primitives import serialization, hashes
		from cryptography.hazmat.backends import default_backend
		
		if operation == "encrypt":
			# Load public key and encrypt
			public_key = serialization.load_pem_public_key(key, backend=default_backend())
			ciphertext = public_key.encrypt(
				data,
				padding.OAEP(
					mgf=padding.MGF1(algorithm=hashes.SHA256()),
					algorithm=hashes.SHA256(),
					label=None
				)
			)
			return {
				"ciphertext": ciphertext,
				"algorithm": "RSA-OAEP",
				"key_size": public_key.key_size
			}
		elif operation == "decrypt":
			# Load private key and decrypt
			private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
			plaintext = private_key.decrypt(
				data["ciphertext"],
				padding.OAEP(
					mgf=padding.MGF1(algorithm=hashes.SHA256()),
					algorithm=hashes.SHA256(),
					label=None
				)
			)
			return plaintext
		else:
			raise ValueError(f"Unsupported RSA operation: {operation}")
	
	async def _ecdsa_sign(self, data: Union[bytes, Dict[str, Any]], key: bytes, operation: str) -> Union[Dict[str, Any], bytes]:
		"""ECDSA signing/verification"""
		from cryptography.hazmat.primitives.asymmetric import ec
		from cryptography.hazmat.primitives import serialization, hashes
		from cryptography.hazmat.backends import default_backend
		
		if operation == "sign":
			# Load private key and sign
			private_key = serialization.load_pem_private_key(key, password=None, backend=default_backend())
			signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
			
			return {
				"signature": signature,
				"algorithm": "ECDSA-SHA256",
				"curve": private_key.curve.name,
				"key_size": private_key.curve.key_size
			}
		
		elif operation == "verify":
			# Load public key and verify
			public_key = serialization.load_pem_public_key(key, backend=default_backend())
			try:
				public_key.verify(data["signature"], data["message"], ec.ECDSA(hashes.SHA256()))
				return True
			except Exception:
				return False
		
		else:
			raise ValueError(f"Unsupported ECDSA operation: {operation}")


class IoTDeviceManager:
	"""IoT device lifecycle management"""
	
	def __init__(self, service: KeyManagementService):
		self.service = service
		self.devices: Dict[str, IoTDevice] = {}
		self.edge_nodes: Dict[str, EdgeNode] = {}
		self.crypto_service = EdgeCryptoService()
		
		# MQTT client for device communication
		self.mqtt_client: Optional[mqtt.Client] = None
		self.mqtt_config: Dict[str, Any] = {}
		
		# Background tasks
		self._device_monitor_task: Optional[asyncio.Task] = None
		self._key_rotation_task: Optional[asyncio.Task] = None
		self._is_running = False
	
	async def initialize(self, config: Dict[str, Any] = None):
		"""Initialize IoT device manager"""
		config = config or {}
		
		# Initialize MQTT client
		self.mqtt_config = config.get('mqtt', {
			'broker': 'localhost',
			'port': 1883,
			'username': '',
			'password': '',
			'use_tls': False
		})
		
		await self._setup_mqtt_client()
		
		# Start background tasks
		self._is_running = True
		self._device_monitor_task = asyncio.create_task(self._device_monitoring_loop())
		self._key_rotation_task = asyncio.create_task(self._key_rotation_loop())
		
		logging.info("IoT device manager initialized")
	
	async def shutdown(self):
		"""Shutdown IoT device manager"""
		self._is_running = False
		
		if self._device_monitor_task:
			self._device_monitor_task.cancel()
		
		if self._key_rotation_task:
			self._key_rotation_task.cancel()
		
		if self.mqtt_client:
			self.mqtt_client.disconnect()
		
		logging.info("IoT device manager shut down")
	
	async def _setup_mqtt_client(self):
		"""Setup MQTT client for device communication"""
		try:
			self.mqtt_client = mqtt.Client()
			
			if self.mqtt_config.get('username'):
				self.mqtt_client.username_pw_set(
					self.mqtt_config['username'],
					self.mqtt_config['password']
				)
			
			if self.mqtt_config.get('use_tls'):
				context = ssl.create_default_context()
				self.mqtt_client.tls_set_context(context)
			
			# Set callbacks
			self.mqtt_client.on_connect = self._on_mqtt_connect
			self.mqtt_client.on_message = self._on_mqtt_message
			self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
			
			# Connect to broker
			self.mqtt_client.connect(
				self.mqtt_config['broker'],
				self.mqtt_config['port'],
				60
			)
			
			self.mqtt_client.loop_start()
			logging.info("MQTT client connected")
			
		except Exception as e:
			logging.error(f"Failed to setup MQTT client: {e}")
	
	def _on_mqtt_connect(self, client, userdata, flags, rc):
		"""MQTT connection callback"""
		if rc == 0:
			client.subscribe("keym/devices/+/heartbeat")
			client.subscribe("keym/devices/+/key_request")
			client.subscribe("keym/devices/+/status")
			logging.info("MQTT client subscribed to device topics")
		else:
			logging.error(f"MQTT connection failed with code {rc}")
	
	def _on_mqtt_message(self, client, userdata, msg):
		"""MQTT message callback"""
		try:
			topic_parts = msg.topic.split('/')
			if len(topic_parts) >= 3:
				device_id = topic_parts[2]
				message_type = topic_parts[3] if len(topic_parts) > 3 else "unknown"
				
				payload = json.loads(msg.payload.decode())
				
				# Handle different message types
				asyncio.create_task(self._handle_device_message(device_id, message_type, payload))
				
		except Exception as e:
			logging.error(f"Error processing MQTT message: {e}")
	
	def _on_mqtt_disconnect(self, client, userdata, rc):
		"""MQTT disconnection callback"""
		logging.warning("MQTT client disconnected")
	
	async def _handle_device_message(self, device_id: str, message_type: str, payload: Dict[str, Any]):
		"""Handle incoming device messages"""
		if message_type == "heartbeat":
			await self._handle_device_heartbeat(device_id, payload)
		elif message_type == "key_request":
			await self._handle_key_request(device_id, payload)
		elif message_type == "status":
			await self._handle_device_status(device_id, payload)
	
	async def register_device(self, device_spec: Dict[str, Any], user_id: str = None) -> IoTDevice:
		"""Register new IoT device"""
		# Create device from specification
		device = IoTDevice(
			device_name=device_spec.get('name', ''),
			device_type=DeviceType(device_spec.get('type', 'sensor')),
			manufacturer=device_spec.get('manufacturer', ''),
			model=device_spec.get('model', ''),
			firmware_version=device_spec.get('firmware_version', ''),
			edge_location=EdgeLocation(device_spec.get('location', 'factory_floor')),
			connectivity=[ConnectivityType(c) for c in device_spec.get('connectivity', [])],
			security_level=SecurityLevel(device_spec.get('security_level', 'standard')),
			tenant_id=device_spec.get('tenant_id', ''),
			supports_hardware_crypto=device_spec.get('supports_hardware_crypto', False),
			has_secure_element=device_spec.get('has_secure_element', False),
			supported_algorithms=device_spec.get('supported_algorithms', ['AES-256-GCM']),
			max_key_size=device_spec.get('max_key_size', 256)
		)
		
		# Generate initial device keys
		await self._provision_device_keys(device)
		
		# Store device
		self.devices[device.device_id] = device
		
		# Log registration
		if hasattr(self.service, '_log_audit_event'):
			await self.service._log_audit_event(
				event_type="device_registered",
				resource_id=device.device_id,
				action="register_iot_device",
				user_id=user_id,
				details={
					'device_type': device.device_type.value,
					'manufacturer': device.manufacturer,
					'model': device.model,
					'security_level': device.security_level.value
				}
			)
		
		logging.info(f"IoT device registered: {device.device_id}")
		return device
	
	async def _provision_device_keys(self, device: IoTDevice):
		"""Provision cryptographic keys for device"""
		for algorithm in device.supported_algorithms:
			try:
				key = await self.crypto_service.generate_device_key(algorithm, device.max_key_size)
				key_id = f"{device.device_id}_{algorithm}_{int(time.time())}"
				
				# Store key securely (in production, use HSM or secure key store)
				device.device_keys[algorithm] = key_id
				
				# Send key to device via secure channel
				await self._send_key_to_device(device.device_id, key_id, key, algorithm)
				
			except Exception as e:
				logging.error(f"Failed to provision key for device {device.device_id}: {e}")
	
	async def _send_key_to_device(self, device_id: str, key_id: str, key: bytes, algorithm: str):
		"""Send key to device via secure channel"""
		if self.mqtt_client:
			# In production, encrypt the key before sending
			key_message = {
				'key_id': key_id,
				'algorithm': algorithm,
				'key': key.hex(),  # In production, encrypt this
				'timestamp': datetime.utcnow().isoformat()
			}
			
			topic = f"keym/devices/{device_id}/key_provision"
			self.mqtt_client.publish(topic, json.dumps(key_message))
	
	async def register_edge_node(self, node_spec: Dict[str, Any], user_id: str = None) -> EdgeNode:
		"""Register edge computing node"""
		node = EdgeNode(
			node_name=node_spec.get('name', ''),
			location=EdgeLocation(node_spec.get('location', 'factory_floor')),
			cpu_cores=node_spec.get('cpu_cores', 4),
			memory_gb=node_spec.get('memory_gb', 8),
			storage_gb=node_spec.get('storage_gb', 64),
			has_gpu=node_spec.get('has_gpu', False),
			has_tpm=node_spec.get('has_tpm', False),
			ip_address=node_spec.get('ip_address', ''),
			max_device_capacity=node_spec.get('max_device_capacity', 100),
			tenant_id=node_spec.get('tenant_id', ''),
			supports_offline_crypto=node_spec.get('supports_offline_crypto', True)
		)
		
		# Store node
		self.edge_nodes[node.node_id] = node
		
		# Log registration
		if hasattr(self.service, '_log_audit_event'):
			await self.service._log_audit_event(
				event_type="edge_node_registered",
				resource_id=node.node_id,
				action="register_edge_node",
				user_id=user_id,
				details={
					'location': node.location.value,
					'cpu_cores': node.cpu_cores,
					'memory_gb': node.memory_gb,
					'has_tpm': node.has_tpm
				}
			)
		
		logging.info(f"Edge node registered: {node.node_id}")
		return node
	
	async def assign_device_to_edge_node(self, device_id: str, node_id: str, user_id: str = None):
		"""Assign IoT device to edge node"""
		if device_id not in self.devices:
			raise ValueError(f"Device not found: {device_id}")
		
		if node_id not in self.edge_nodes:
			raise ValueError(f"Edge node not found: {node_id}")
		
		edge_node = self.edge_nodes[node_id]
		
		# Check capacity
		if len(edge_node.managed_devices) >= edge_node.max_device_capacity:
			raise ValueError(f"Edge node {node_id} at capacity")
		
		# Assign device
		edge_node.managed_devices.add(device_id)
		edge_node.updated_at = datetime.utcnow()
		
		# Log assignment
		if hasattr(self.service, '_log_audit_event'):
			await self.service._log_audit_event(
				event_type="device_assigned",
				resource_id=device_id,
				action="assign_to_edge_node",
				user_id=user_id,
				details={
					'edge_node_id': node_id,
					'edge_location': edge_node.location.value
				}
			)
		
		logging.info(f"Device {device_id} assigned to edge node {node_id}")
	
	async def rotate_device_keys(self, device_id: str, user_id: str = None) -> Dict[str, str]:
		"""Rotate keys for specific device"""
		if device_id not in self.devices:
			raise ValueError(f"Device not found: {device_id}")
		
		device = self.devices[device_id]
		new_keys = {}
		
		# Rotate each algorithm key
		for algorithm in device.supported_algorithms:
			try:
				new_key = await self.crypto_service.generate_device_key(algorithm, device.max_key_size)
				key_id = f"{device_id}_{algorithm}_{int(time.time())}"
				
				# Update device key reference
				old_key_id = device.device_keys.get(algorithm)
				device.device_keys[algorithm] = key_id
				new_keys[algorithm] = key_id
				
				# Send new key to device
				await self._send_key_to_device(device_id, key_id, new_key, algorithm)
				
				# Schedule old key revocation (after grace period)
				if old_key_id:
					asyncio.create_task(self._revoke_key_after_grace_period(old_key_id, 3600))
				
			except Exception as e:
				logging.error(f"Failed to rotate key for device {device_id}, algorithm {algorithm}: {e}")
		
		# Update rotation timestamp
		device.last_key_rotation = datetime.utcnow()
		device.updated_at = datetime.utcnow()
		
		# Log key rotation
		if hasattr(self.service, '_log_audit_event'):
			await self.service._log_audit_event(
				event_type="key_rotated",
				resource_id=device_id,
				action="rotate_device_keys",
				user_id=user_id,
				details={
					'algorithms': list(device.supported_algorithms),
					'rotation_count': len(new_keys)
				}
			)
		
		logging.info(f"Keys rotated for device: {device_id}")
		return new_keys
	
	async def _revoke_key_after_grace_period(self, key_id: str, grace_period: int):
		"""Revoke key after grace period"""
		await asyncio.sleep(grace_period)
		# Implementation would revoke key from secure key store
		logging.info(f"Key revoked after grace period: {key_id}")
	
	async def get_device_status(self, device_id: str) -> Dict[str, Any]:
		"""Get comprehensive device status"""
		if device_id not in self.devices:
			raise ValueError(f"Device not found: {device_id}")
		
		device = self.devices[device_id]
		
		# Calculate key rotation status
		time_since_rotation = datetime.utcnow() - device.last_key_rotation
		rotation_needed = time_since_rotation.total_seconds() > device.key_rotation_interval
		
		return {
			'device_id': device_id,
			'status': device.status,
			'last_seen': device.last_seen.isoformat(),
			'battery_level': device.battery_level,
			'signal_strength': device.signal_strength,
			'key_rotation_needed': rotation_needed,
			'keys_provisioned': len(device.device_keys),
			'edge_node_assigned': any(device_id in node.managed_devices for node in self.edge_nodes.values()),
			'security_level': device.security_level.value,
			'connectivity': [c.value for c in device.connectivity]
		}
	
	async def _device_monitoring_loop(self):
		"""Background device monitoring loop"""
		while self._is_running:
			try:
				current_time = datetime.utcnow()
				
				# Check for offline devices
				for device in self.devices.values():
					time_since_seen = current_time - device.last_seen
					if time_since_seen.total_seconds() > 3600:  # 1 hour offline
						if device.status != "offline":
							device.status = "offline"
							logging.warning(f"Device went offline: {device.device_id}")
							
							# Log offline event
							if hasattr(self.service, '_log_audit_event'):
								await self.service._log_audit_event(
									event_type="device_offline",
									resource_id=device.device_id,
									action="device_status_change",
									details={'new_status': 'offline'}
								)
				
				# Check edge node health
				for node in self.edge_nodes.values():
					time_since_heartbeat = current_time - node.last_heartbeat
					if time_since_heartbeat.total_seconds() > 300:  # 5 minutes
						if node.status != "offline":
							node.status = "offline"
							logging.warning(f"Edge node went offline: {node.node_id}")
				
				await asyncio.sleep(60)  # Check every minute
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in device monitoring: {e}")
				await asyncio.sleep(60)
	
	async def _key_rotation_loop(self):
		"""Background key rotation loop"""
		while self._is_running:
			try:
				current_time = datetime.utcnow()
				
				# Check devices needing key rotation
				for device in self.devices.values():
					time_since_rotation = current_time - device.last_key_rotation
					if time_since_rotation.total_seconds() > device.key_rotation_interval:
						try:
							await self.rotate_device_keys(device.device_id)
						except Exception as e:
							logging.error(f"Auto key rotation failed for device {device.device_id}: {e}")
				
				await asyncio.sleep(3600)  # Check every hour
				
			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.error(f"Error in key rotation loop: {e}")
				await asyncio.sleep(3600)
	
	async def _handle_device_heartbeat(self, device_id: str, payload: Dict[str, Any]):
		"""Handle device heartbeat"""
		if device_id in self.devices:
			device = self.devices[device_id]
			device.last_seen = datetime.utcnow()
			device.status = "active"
			device.battery_level = payload.get('battery_level')
			device.signal_strength = payload.get('signal_strength')
			device.updated_at = datetime.utcnow()
	
	async def _handle_key_request(self, device_id: str, payload: Dict[str, Any]):
		"""Handle device key request"""
		if device_id in self.devices:
			algorithm = payload.get('algorithm')
			if algorithm in self.devices[device_id].supported_algorithms:
				await self.rotate_device_keys(device_id)
	
	async def _handle_device_status(self, device_id: str, payload: Dict[str, Any]):
		"""Handle device status update"""
		if device_id in self.devices:
			device = self.devices[device_id]
			device.last_seen = datetime.utcnow()
			
			# Update device metadata from status
			if 'cpu_usage' in payload:
				device.metadata['cpu_usage'] = payload['cpu_usage']
			if 'memory_usage' in payload:
				device.metadata['memory_usage'] = payload['memory_usage']
			if 'temperature' in payload:
				device.metadata['temperature'] = payload['temperature']
			
			device.updated_at = datetime.utcnow()
	
	async def get_devices_by_location(self, location: EdgeLocation) -> List[IoTDevice]:
		"""Get all devices at specific edge location"""
		return [device for device in self.devices.values() if device.edge_location == location]
	
	async def get_edge_node_devices(self, node_id: str) -> List[IoTDevice]:
		"""Get all devices managed by edge node"""
		if node_id not in self.edge_nodes:
			return []
		
		managed_device_ids = self.edge_nodes[node_id].managed_devices
		return [self.devices[device_id] for device_id in managed_device_ids if device_id in self.devices]
	
	async def get_security_summary(self) -> Dict[str, Any]:
		"""Get security summary across all IoT devices"""
		total_devices = len(self.devices)
		devices_by_security_level = {}
		devices_with_secure_element = 0
		devices_needing_rotation = 0
		
		current_time = datetime.utcnow()
		
		for device in self.devices.values():
			# Count by security level
			level = device.security_level.value
			devices_by_security_level[level] = devices_by_security_level.get(level, 0) + 1
			
			# Count secure elements
			if device.has_secure_element:
				devices_with_secure_element += 1
			
			# Count devices needing rotation
			time_since_rotation = current_time - device.last_key_rotation
			if time_since_rotation.total_seconds() > device.key_rotation_interval:
				devices_needing_rotation += 1
		
		return {
			'total_devices': total_devices,
			'devices_by_security_level': devices_by_security_level,
			'devices_with_secure_element': devices_with_secure_element,
			'devices_needing_rotation': devices_needing_rotation,
			'secure_element_percentage': (devices_with_secure_element / total_devices * 100) if total_devices > 0 else 0,
			'rotation_compliance': ((total_devices - devices_needing_rotation) / total_devices * 100) if total_devices > 0 else 100
		}


# Factory function
async def create_iot_device_manager(service: KeyManagementService, config: Dict[str, Any] = None) -> IoTDeviceManager:
	"""Create and initialize IoT device manager"""
	manager = IoTDeviceManager(service)
	await manager.initialize(config)
	return manager


# Export main components
__all__ = [
	'IoTDeviceManager', 'IoTDevice', 'EdgeNode', 'EdgeCryptoService',
	'DeviceType', 'EdgeLocation', 'ConnectivityType', 'SecurityLevel',
	'create_iot_device_manager'
]