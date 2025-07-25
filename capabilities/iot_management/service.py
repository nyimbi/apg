#!/usr/bin/env python3
"""
APG IoT Management Capability
=============================

Comprehensive IoT device management, sensor data collection, edge computing,
and real-time monitoring for industrial and smart applications.
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid
import queue
import sqlite3
from pathlib import Path
import hashlib
import ssl

# Optional imports for advanced features
try:
	import paho.mqtt.client as mqtt
	HAS_MQTT = True
except ImportError:
	HAS_MQTT = False

try:
	import serial
	HAS_SERIAL = True
except ImportError:
	HAS_SERIAL = False

try:
	import requests
	HAS_REQUESTS = True
except ImportError:
	HAS_REQUESTS = False

try:
	import websocket
	HAS_WEBSOCKET = True
except ImportError:
	HAS_WEBSOCKET = False

class DeviceStatus(Enum):
	"""IoT device status"""
	UNKNOWN = "unknown"
	ONLINE = "online"
	OFFLINE = "offline"
	ERROR = "error"
	MAINTENANCE = "maintenance"
	UPDATING = "updating"

class DeviceType(Enum):
	"""Types of IoT devices"""
	SENSOR = "sensor"
	ACTUATOR = "actuator"
	GATEWAY = "gateway"
	CONTROLLER = "controller"
	CAMERA = "camera"
	DISPLAY = "display"
	CUSTOM = "custom"

class SensorType(Enum):
	"""Types of sensors"""
	TEMPERATURE = "temperature"
	HUMIDITY = "humidity"
	PRESSURE = "pressure"
	LIGHT = "light"
	MOTION = "motion"
	PROXIMITY = "proximity"
	ACCELEROMETER = "accelerometer"
	GYROSCOPE = "gyroscope"
	GPS = "gps"
	AIR_QUALITY = "air_quality"
	SOUND = "sound"
	VIBRATION = "vibration"
	MAGNETIC = "magnetic"
	CUSTOM = "custom"

class ConnectionType(Enum):
	"""Device connection types"""
	WIFI = "wifi"
	BLUETOOTH = "bluetooth"
	ETHERNET = "ethernet"
	CELLULAR = "cellular"
	LORA = "lora"
	ZIGBEE = "zigbee"
	MQTT = "mqtt"
	HTTP = "http"
	SERIAL = "serial"
	MODBUS = "modbus"

@dataclass
class DeviceCredentials:
	"""Device authentication credentials"""
	device_id: str
	username: str = ""
	password: str = ""
	api_key: str = ""
	certificate_path: str = ""
	private_key_path: str = ""
	auth_token: str = ""
	expires_at: Optional[datetime] = None

@dataclass
class SensorReading:
	"""Individual sensor reading"""
	sensor_id: str
	sensor_type: SensorType
	value: Union[float, int, str, Dict[str, Any]]
	unit: str
	timestamp: datetime = field(default_factory=datetime.utcnow)
	quality: float = 1.0  # 0-1, quality of reading
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceInfo:
	"""IoT device information"""
	device_id: str
	name: str
	device_type: DeviceType
	manufacturer: str = ""
	model: str = ""
	firmware_version: str = ""
	hardware_version: str = ""
	mac_address: str = ""
	ip_address: str = ""
	connection_type: ConnectionType = ConnectionType.WIFI
	location: Dict[str, Any] = field(default_factory=dict)
	capabilities: List[str] = field(default_factory=list)
	sensors: List[SensorType] = field(default_factory=list)
	status: DeviceStatus = DeviceStatus.UNKNOWN
	last_seen: Optional[datetime] = None
	battery_level: Optional[float] = None
	signal_strength: Optional[float] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class DeviceCommand:
	"""Command to send to device"""
	command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	device_id: str = ""
	command: str = ""
	parameters: Dict[str, Any] = field(default_factory=dict)
	timestamp: datetime = field(default_factory=datetime.utcnow)
	expires_at: Optional[datetime] = None
	priority: int = 1  # 1-10, higher = more urgent
	retry_count: int = 0
	max_retries: int = 3
	status: str = "pending"  # pending, sent, acknowledged, failed

@dataclass
class AlertRule:
	"""IoT monitoring alert rule"""
	rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	name: str = ""
	description: str = ""
	device_id: str = ""
	sensor_type: Optional[SensorType] = None
	condition: str = ""  # e.g., "value > 30", "status == 'offline'"
	threshold_value: Optional[float] = None
	action: str = "log"  # log, email, webhook, command
	action_config: Dict[str, Any] = field(default_factory=dict)
	enabled: bool = True
	last_triggered: Optional[datetime] = None
	trigger_count: int = 0

class DeviceManager:
	"""Core IoT device management"""
	
	def __init__(self, database_path: str = "iot_devices.db"):
		self.database_path = database_path
		self.devices: Dict[str, DeviceInfo] = {}
		self.credentials: Dict[str, DeviceCredentials] = {}
		self.command_queue: queue.Queue = queue.Queue()
		self.alert_rules: Dict[str, AlertRule] = {}
		
		self.logger = logging.getLogger("device_manager")
		
		# Initialize database
		self._initialize_database()
		
		# Load existing devices
		self._load_devices()
	
	def _initialize_database(self):
		"""Initialize SQLite database"""
		conn = sqlite3.connect(self.database_path)
		cursor = conn.cursor()
		
		# Devices table
		cursor.execute('''
			CREATE TABLE IF NOT EXISTS devices (
				device_id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				device_type TEXT NOT NULL,
				manufacturer TEXT,
				model TEXT,
				firmware_version TEXT,
				ip_address TEXT,
				connection_type TEXT,
				status TEXT,
				last_seen TIMESTAMP,
				battery_level REAL,
				signal_strength REAL,
				location TEXT,
				capabilities TEXT,
				sensors TEXT,
				metadata TEXT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		''')
		
		# Sensor readings table
		cursor.execute('''
			CREATE TABLE IF NOT EXISTS sensor_readings (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				device_id TEXT NOT NULL,
				sensor_id TEXT NOT NULL,
				sensor_type TEXT NOT NULL,
				value TEXT NOT NULL,
				unit TEXT,
				quality REAL DEFAULT 1.0,
				timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				metadata TEXT,
				FOREIGN KEY (device_id) REFERENCES devices (device_id)
			)
		''')
		
		# Commands table
		cursor.execute('''
			CREATE TABLE IF NOT EXISTS device_commands (
				command_id TEXT PRIMARY KEY,
				device_id TEXT NOT NULL,
				command TEXT NOT NULL,
				parameters TEXT,
				timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				status TEXT DEFAULT 'pending',
				retry_count INTEGER DEFAULT 0,
				FOREIGN KEY (device_id) REFERENCES devices (device_id)
			)
		''')
		
		# Alert rules table
		cursor.execute('''
			CREATE TABLE IF NOT EXISTS alert_rules (
				rule_id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				device_id TEXT,
				sensor_type TEXT,
				condition_expr TEXT,
				threshold_value REAL,
				action TEXT,
				action_config TEXT,
				enabled INTEGER DEFAULT 1,
				last_triggered TIMESTAMP,
				trigger_count INTEGER DEFAULT 0,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		''')
		
		conn.commit()
		conn.close()
		
		self.logger.info("Database initialized successfully")
	
	def _load_devices(self):
		"""Load devices from database"""
		conn = sqlite3.connect(self.database_path)
		cursor = conn.cursor()
		
		cursor.execute("SELECT * FROM devices")
		rows = cursor.fetchall()
		
		for row in rows:
			device_info = DeviceInfo(
				device_id=row[0],
				name=row[1],
				device_type=DeviceType(row[2]),
				manufacturer=row[3] or "",
				model=row[4] or "",
				firmware_version=row[5] or "",
				ip_address=row[6] or "",
				connection_type=ConnectionType(row[7]) if row[7] else ConnectionType.WIFI,
				status=DeviceStatus(row[8]) if row[8] else DeviceStatus.UNKNOWN,
				last_seen=datetime.fromisoformat(row[9]) if row[9] else None,
				battery_level=row[10],
				signal_strength=row[11],
				location=json.loads(row[12]) if row[12] else {},
				capabilities=json.loads(row[13]) if row[13] else [],
				sensors=[SensorType(s) for s in json.loads(row[14])] if row[14] else [],
				metadata=json.loads(row[15]) if row[15] else {}
			)
			self.devices[device_info.device_id] = device_info
		
		conn.close()
		self.logger.info(f"Loaded {len(self.devices)} devices from database")
	
	async def register_device(self, device_info: DeviceInfo) -> bool:
		"""Register a new IoT device"""
		try:
			# Update timestamps
			device_info.created_at = datetime.utcnow()
			device_info.updated_at = datetime.utcnow()
			
			# Store in memory
			self.devices[device_info.device_id] = device_info
			
			# Store in database
			conn = sqlite3.connect(self.database_path)
			cursor = conn.cursor()
			
			cursor.execute('''
				INSERT OR REPLACE INTO devices 
				(device_id, name, device_type, manufacturer, model, firmware_version,
				 ip_address, connection_type, status, last_seen, battery_level,
				 signal_strength, location, capabilities, sensors, metadata, 
				 created_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			''', (
				device_info.device_id,
				device_info.name,
				device_info.device_type.value,
				device_info.manufacturer,
				device_info.model,
				device_info.firmware_version,
				device_info.ip_address,
				device_info.connection_type.value,
				device_info.status.value,
				device_info.last_seen.isoformat() if device_info.last_seen else None,
				device_info.battery_level,
				device_info.signal_strength,
				json.dumps(device_info.location),
				json.dumps(device_info.capabilities),
				json.dumps([s.value for s in device_info.sensors]),
				json.dumps(device_info.metadata),
				device_info.created_at.isoformat(),
				device_info.updated_at.isoformat()
			))
			
			conn.commit()
			conn.close()
			
			self.logger.info(f"Device registered: {device_info.name} ({device_info.device_id})")
			return True
			
		except Exception as e:
			self.logger.error(f"Error registering device: {e}")
			return False
	
	async def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
		"""Update device status"""
		if device_id not in self.devices:
			return False
		
		try:
			device = self.devices[device_id]
			device.status = status
			device.last_seen = datetime.utcnow()
			device.updated_at = datetime.utcnow()
			
			# Update database
			conn = sqlite3.connect(self.database_path)
			cursor = conn.cursor()
			
			cursor.execute('''
				UPDATE devices 
				SET status = ?, last_seen = ?, updated_at = ?
				WHERE device_id = ?
			''', (status.value, device.last_seen.isoformat(), 
				  device.updated_at.isoformat(), device_id))
			
			conn.commit()
			conn.close()
			
			return True
			
		except Exception as e:
			self.logger.error(f"Error updating device status: {e}")
			return False
	
	def get_device(self, device_id: str) -> Optional[DeviceInfo]:
		"""Get device information"""
		return self.devices.get(device_id)
	
	def list_devices(
		self, 
		device_type: Optional[DeviceType] = None,
		status: Optional[DeviceStatus] = None
	) -> List[DeviceInfo]:
		"""List devices with optional filtering"""
		devices = list(self.devices.values())
		
		if device_type:
			devices = [d for d in devices if d.device_type == device_type]
		
		if status:
			devices = [d for d in devices if d.status == status]
		
		return devices
	
	async def send_command(self, command: DeviceCommand) -> bool:
		"""Send command to device"""
		try:
			# Store command in database
			conn = sqlite3.connect(self.database_path)
			cursor = conn.cursor()
			
			cursor.execute('''
				INSERT INTO device_commands 
				(command_id, device_id, command, parameters, timestamp, status)
				VALUES (?, ?, ?, ?, ?, ?)
			''', (
				command.command_id,
				command.device_id,
				command.command,
				json.dumps(command.parameters),
				command.timestamp.isoformat(),
				command.status
			))
			
			conn.commit()
			conn.close()
			
			# Add to command queue for processing
			self.command_queue.put(command)
			
			self.logger.info(f"Command queued: {command.command} for {command.device_id}")
			return True
			
		except Exception as e:
			self.logger.error(f"Error sending command: {e}")
			return False

class SensorDataManager:
	"""Manage sensor data collection and storage"""
	
	def __init__(self, database_path: str = "iot_devices.db"):
		self.database_path = database_path
		self.data_buffer: Dict[str, List[SensorReading]] = {}
		self.buffer_size = 1000
		self.flush_interval = 60  # seconds
		
		self.logger = logging.getLogger("sensor_data")
		
		# Start data flushing thread
		self.flush_thread = threading.Thread(target=self._flush_data_periodically, daemon=True)
		self.flush_thread.start()
	
	async def record_reading(self, reading: SensorReading) -> bool:
		"""Record a sensor reading"""
		try:
			device_id = reading.sensor_id.split('_')[0]  # Extract device ID from sensor ID
			
			# Add to buffer
			if device_id not in self.data_buffer:
				self.data_buffer[device_id] = []
			
			self.data_buffer[device_id].append(reading)
			
			# Flush if buffer is full
			if len(self.data_buffer[device_id]) >= self.buffer_size:
				await self._flush_device_data(device_id)
			
			return True
			
		except Exception as e:
			self.logger.error(f"Error recording reading: {e}")
			return False
	
	async def _flush_device_data(self, device_id: str):
		"""Flush buffered data for a device to database"""
		if device_id not in self.data_buffer or not self.data_buffer[device_id]:
			return
		
		try:
			conn = sqlite3.connect(self.database_path)
			cursor = conn.cursor()
			
			readings = self.data_buffer[device_id]
			for reading in readings:
				cursor.execute('''
					INSERT INTO sensor_readings 
					(device_id, sensor_id, sensor_type, value, unit, quality, timestamp, metadata)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				''', (
					device_id,
					reading.sensor_id,
					reading.sensor_type.value,
					json.dumps(reading.value) if isinstance(reading.value, (dict, list)) else str(reading.value),
					reading.unit,
					reading.quality,
					reading.timestamp.isoformat(),
					json.dumps(reading.metadata)
				))
			
			conn.commit()
			conn.close()
			
			# Clear buffer
			self.data_buffer[device_id] = []
			
			self.logger.debug(f"Flushed {len(readings)} readings for device {device_id}")
			
		except Exception as e:
			self.logger.error(f"Error flushing data for device {device_id}: {e}")
	
	def _flush_data_periodically(self):
		"""Periodically flush all buffered data"""
		while True:
			try:
				time.sleep(self.flush_interval)
				
				# Flush all device buffers
				for device_id in list(self.data_buffer.keys()):
					asyncio.run(self._flush_device_data(device_id))
					
			except Exception as e:
				self.logger.error(f"Error in periodic flush: {e}")
	
	async def get_readings(
		self,
		device_id: str,
		sensor_type: Optional[SensorType] = None,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None,
		limit: int = 1000
	) -> List[SensorReading]:
		"""Get sensor readings with filtering"""
		
		try:
			conn = sqlite3.connect(self.database_path)
			cursor = conn.cursor()
			
			# Build query
			query = "SELECT * FROM sensor_readings WHERE device_id = ?"
			params = [device_id]
			
			if sensor_type:
				query += " AND sensor_type = ?"
				params.append(sensor_type.value)
			
			if start_time:
				query += " AND timestamp >= ?"
				params.append(start_time.isoformat())
			
			if end_time:
				query += " AND timestamp <= ?"
				params.append(end_time.isoformat())
			
			query += " ORDER BY timestamp DESC LIMIT ?"
			params.append(limit)
			
			cursor.execute(query, params)
			rows = cursor.fetchall()
			
			readings = []
			for row in rows:
				reading = SensorReading(
					sensor_id=row[2],
					sensor_type=SensorType(row[3]),
					value=json.loads(row[4]) if row[4].startswith(('[', '{')) else row[4],
					unit=row[5] or "",
					timestamp=datetime.fromisoformat(row[7]),
					quality=row[6] or 1.0,
					metadata=json.loads(row[8]) if row[8] else {}
				)
				readings.append(reading)
			
			conn.close()
			return readings
			
		except Exception as e:
			self.logger.error(f"Error getting readings: {e}")
			return []
	
	async def get_latest_reading(
		self, 
		device_id: str, 
		sensor_type: SensorType
	) -> Optional[SensorReading]:
		"""Get latest reading for a specific sensor"""
		
		readings = await self.get_readings(
			device_id=device_id,
			sensor_type=sensor_type,
			limit=1
		)
		
		return readings[0] if readings else None
	
	async def calculate_statistics(
		self,
		device_id: str,
		sensor_type: SensorType,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None
	) -> Dict[str, Any]:
		"""Calculate statistics for sensor data"""
		
		readings = await self.get_readings(
			device_id=device_id,
			sensor_type=sensor_type,
			start_time=start_time,
			end_time=end_time,
			limit=10000
		)
		
		if not readings:
			return {}
		
		# Extract numeric values
		values = []
		for reading in readings:
			try:
				if isinstance(reading.value, (int, float)):
					values.append(float(reading.value))
				elif isinstance(reading.value, str):
					values.append(float(reading.value))
			except (ValueError, TypeError):
				continue
		
		if not values:
			return {}
		
		import statistics
		
		return {
			"count": len(values),
			"min": min(values),
			"max": max(values),
			"mean": statistics.mean(values),
			"median": statistics.median(values),
			"std_dev": statistics.stdev(values) if len(values) > 1 else 0,
			"range": max(values) - min(values),
			"first_reading": readings[-1].timestamp.isoformat(),
			"last_reading": readings[0].timestamp.isoformat()
		}

class MQTTConnector:
	"""MQTT connectivity for IoT devices"""
	
	def __init__(self, broker_host: str, broker_port: int = 1883):
		if not HAS_MQTT:
			raise ImportError("paho-mqtt required for MQTT connectivity")
		
		self.broker_host = broker_host
		self.broker_port = broker_port
		self.client = mqtt.Client()
		self.connected = False
		self.subscriptions: Dict[str, Callable] = {}
		
		self.logger = logging.getLogger("mqtt_connector")
		
		# Setup callbacks
		self.client.on_connect = self._on_connect
		self.client.on_disconnect = self._on_disconnect
		self.client.on_message = self._on_message
	
	def _on_connect(self, client, userdata, flags, rc):
		"""MQTT connection callback"""
		if rc == 0:
			self.connected = True
			self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
			
			# Re-subscribe to topics
			for topic in self.subscriptions:
				client.subscribe(topic)
		else:
			self.logger.error(f"Failed to connect to MQTT broker, code: {rc}")
	
	def _on_disconnect(self, client, userdata, rc):
		"""MQTT disconnection callback"""
		self.connected = False
		self.logger.warning("Disconnected from MQTT broker")
	
	def _on_message(self, client, userdata, msg):
		"""MQTT message callback"""
		topic = msg.topic
		payload = msg.payload.decode()
		
		# Call registered handler
		if topic in self.subscriptions:
			try:
				self.subscriptions[topic](topic, payload)
			except Exception as e:
				self.logger.error(f"Error processing MQTT message: {e}")
	
	async def connect(self, username: str = None, password: str = None) -> bool:
		"""Connect to MQTT broker"""
		try:
			if username and password:
				self.client.username_pw_set(username, password)
			
			self.client.connect(self.broker_host, self.broker_port, 60)
			self.client.loop_start()
			
			# Wait for connection
			for _ in range(50):  # 5 second timeout
				if self.connected:
					return True
				await asyncio.sleep(0.1)
			
			return False
			
		except Exception as e:
			self.logger.error(f"Error connecting to MQTT broker: {e}")
			return False
	
	def subscribe(self, topic: str, handler: Callable[[str, str], None]):
		"""Subscribe to MQTT topic"""
		self.subscriptions[topic] = handler
		if self.connected:
			self.client.subscribe(topic)
		self.logger.info(f"Subscribed to topic: {topic}")
	
	async def publish(self, topic: str, payload: str, qos: int = 0) -> bool:
		"""Publish message to MQTT topic"""
		if not self.connected:
			return False
		
		try:
			result = self.client.publish(topic, payload, qos)
			return result.rc == mqtt.MQTT_ERR_SUCCESS
		except Exception as e:
			self.logger.error(f"Error publishing to MQTT: {e}")
			return False
	
	def disconnect(self):
		"""Disconnect from MQTT broker"""
		if self.connected:
			self.client.loop_stop()
			self.client.disconnect()
			self.connected = False

class AlertManager:
	"""IoT monitoring and alerting system"""
	
	def __init__(self, device_manager: DeviceManager, sensor_data: SensorDataManager):
		self.device_manager = device_manager
		self.sensor_data = sensor_data
		self.alert_rules: Dict[str, AlertRule] = {}
		self.alert_handlers: Dict[str, Callable] = {}
		
		self.logger = logging.getLogger("alert_manager")
		
		# Load existing alert rules
		self._load_alert_rules()
		
		# Start monitoring thread
		self.monitoring_thread = threading.Thread(target=self._monitor_devices, daemon=True)
		self.monitoring_active = True
		self.monitoring_thread.start()
	
	def _load_alert_rules(self):
		"""Load alert rules from database"""
		try:
			conn = sqlite3.connect(self.device_manager.database_path)
			cursor = conn.cursor()
			
			cursor.execute("SELECT * FROM alert_rules WHERE enabled = 1")
			rows = cursor.fetchall()
			
			for row in rows:
				rule = AlertRule(
					rule_id=row[0],
					name=row[1],
					device_id=row[2] or "",
					sensor_type=SensorType(row[3]) if row[3] else None,
					condition=row[4] or "",
					threshold_value=row[5],
					action=row[6] or "log",
					action_config=json.loads(row[7]) if row[7] else {},
					enabled=bool(row[8]),
					last_triggered=datetime.fromisoformat(row[9]) if row[9] else None,
					trigger_count=row[10] or 0
				)
				self.alert_rules[rule.rule_id] = rule
			
			conn.close()
			self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")
			
		except Exception as e:
			self.logger.error(f"Error loading alert rules: {e}")
	
	def add_alert_rule(self, rule: AlertRule) -> bool:
		"""Add new alert rule"""
		try:
			# Store in memory
			self.alert_rules[rule.rule_id] = rule
			
			# Store in database
			conn = sqlite3.connect(self.device_manager.database_path)
			cursor = conn.cursor()
			
			cursor.execute('''
				INSERT OR REPLACE INTO alert_rules
				(rule_id, name, device_id, sensor_type, condition_expr, threshold_value,
				 action, action_config, enabled, trigger_count)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			''', (
				rule.rule_id,
				rule.name,
				rule.device_id,
				rule.sensor_type.value if rule.sensor_type else None,
				rule.condition,
				rule.threshold_value,
				rule.action,
				json.dumps(rule.action_config),
				1 if rule.enabled else 0,
				rule.trigger_count
			))
			
			conn.commit()
			conn.close()
			
			self.logger.info(f"Alert rule added: {rule.name}")
			return True
			
		except Exception as e:
			self.logger.error(f"Error adding alert rule: {e}")
			return False
	
	def register_alert_handler(self, action_type: str, handler: Callable):
		"""Register custom alert handler"""
		self.alert_handlers[action_type] = handler
		self.logger.info(f"Registered alert handler for: {action_type}")
	
	def _monitor_devices(self):
		"""Monitor devices and trigger alerts"""
		while self.monitoring_active:
			try:
				self._check_device_alerts()
				self._check_sensor_alerts()
				time.sleep(30)  # Check every 30 seconds
				
			except Exception as e:
				self.logger.error(f"Error in device monitoring: {e}")
				time.sleep(60)
	
	def _check_device_alerts(self):
		"""Check device status alerts"""
		for device in self.device_manager.devices.values():
			for rule in self.alert_rules.values():
				if not rule.enabled or rule.device_id != device.device_id:
					continue
				
				# Check device status conditions
				if "status" in rule.condition:
					should_trigger = self._evaluate_condition(
						rule.condition,
						{"status": device.status.value, "device": device}
					)
					
					if should_trigger:
						self._trigger_alert(rule, device)
	
	def _check_sensor_alerts(self):
		"""Check sensor data alerts"""
		for rule in self.alert_rules.values():
			if not rule.enabled or not rule.sensor_type:
				continue
			
			# Get latest sensor reading
			try:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				reading = loop.run_until_complete(
					self.sensor_data.get_latest_reading(rule.device_id, rule.sensor_type)
				)
				
				loop.close()
				
				if reading:
					should_trigger = self._evaluate_condition(
						rule.condition,
						{"value": reading.value, "reading": reading}
					)
					
					if should_trigger:
						self._trigger_alert(rule, reading)
						
			except Exception as e:
				self.logger.error(f"Error checking sensor alert for rule {rule.name}: {e}")
	
	def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
		"""Evaluate alert condition"""
		try:
			# Simple condition evaluation (in production, use a safe evaluator)
			# This is a simplified version - production should use ast.literal_eval or similar
			
			# Replace context variables
			for key, value in context.items():
				if isinstance(value, str):
					condition = condition.replace(key, f"'{value}'")
				else:
					condition = condition.replace(key, str(value))
			
			# Basic safety check
			dangerous_terms = ['import', 'exec', 'eval', '__', 'open', 'file']
			if any(term in condition.lower() for term in dangerous_terms):
				return False
			
			return eval(condition)
			
		except Exception as e:
			self.logger.error(f"Error evaluating condition '{condition}': {e}")
			return False
	
	def _trigger_alert(self, rule: AlertRule, context: Any):
		"""Trigger an alert"""
		try:
			# Update rule trigger info
			rule.last_triggered = datetime.utcnow()
			rule.trigger_count += 1
			
			# Execute action
			if rule.action == "log":
				self.logger.warning(f"ALERT: {rule.name} - {rule.description}")
			
			elif rule.action in self.alert_handlers:
				self.alert_handlers[rule.action](rule, context)
			
			elif rule.action == "webhook" and "url" in rule.action_config:
				self._send_webhook_alert(rule, context)
			
			# Update database
			conn = sqlite3.connect(self.device_manager.database_path)
			cursor = conn.cursor()
			
			cursor.execute('''
				UPDATE alert_rules 
				SET last_triggered = ?, trigger_count = ?
				WHERE rule_id = ?
			''', (rule.last_triggered.isoformat(), rule.trigger_count, rule.rule_id))
			
			conn.commit()
			conn.close()
			
		except Exception as e:
			self.logger.error(f"Error triggering alert: {e}")
	
	def _send_webhook_alert(self, rule: AlertRule, context: Any):
		"""Send webhook alert"""
		if not HAS_REQUESTS:
			self.logger.error("requests library required for webhook alerts")
			return
		
		try:
			url = rule.action_config.get("url")
			if not url:
				return
			
			payload = {
				"rule_name": rule.name,
				"rule_id": rule.rule_id,
				"description": rule.description,
				"timestamp": datetime.utcnow().isoformat(),
				"context": str(context)
			}
			
			response = requests.post(url, json=payload, timeout=10)
			response.raise_for_status()
			
		except Exception as e:
			self.logger.error(f"Error sending webhook alert: {e}")

class IoTManagementCapability:
	"""Main IoT management capability interface"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		
		# Initialize core components
		database_path = self.config.get("database_path", "iot_devices.db")
		self.device_manager = DeviceManager(database_path)
		self.sensor_data = SensorDataManager(database_path)
		self.alert_manager = AlertManager(self.device_manager, self.sensor_data)
		
		# Optional MQTT connector
		self.mqtt_connector: Optional[MQTTConnector] = None
		if self.config.get("mqtt_broker"):
			self.mqtt_connector = MQTTConnector(
				self.config["mqtt_broker"],
				self.config.get("mqtt_port", 1883)
			)
		
		self.logger = logging.getLogger("iot_capability")
		self.logger.info("IoT Management Capability initialized")
	
	# Public API Methods
	
	async def register_device(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Register a new IoT device"""
		try:
			device_info = DeviceInfo(
				device_id=device_data["device_id"],
				name=device_data["name"],
				device_type=DeviceType(device_data.get("device_type", "custom")),
				manufacturer=device_data.get("manufacturer", ""),
				model=device_data.get("model", ""),
				firmware_version=device_data.get("firmware_version", ""),
				ip_address=device_data.get("ip_address", ""),
				connection_type=ConnectionType(device_data.get("connection_type", "wifi")),
				location=device_data.get("location", {}),
				capabilities=device_data.get("capabilities", []),
				sensors=[SensorType(s) for s in device_data.get("sensors", [])],
				metadata=device_data.get("metadata", {})
			)
			
			success = await self.device_manager.register_device(device_info)
			
			return {
				"success": success,
				"device_id": device_info.device_id,
				"message": "Device registered successfully" if success else "Registration failed",
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def get_device_info(self, device_id: str) -> Dict[str, Any]:
		"""Get device information"""
		device = self.device_manager.get_device(device_id)
		
		if not device:
			return {
				"success": False,
				"error": "Device not found",
				"timestamp": datetime.utcnow().isoformat()
			}
		
		return {
			"success": True,
			"device": asdict(device),
			"timestamp": datetime.utcnow().isoformat()
		}
	
	async def list_devices(
		self, 
		device_type: str = None, 
		status: str = None
	) -> Dict[str, Any]:
		"""List registered devices"""
		try:
			device_type_enum = DeviceType(device_type) if device_type else None
			status_enum = DeviceStatus(status) if status else None
			
			devices = self.device_manager.list_devices(device_type_enum, status_enum)
			
			return {
				"success": True,
				"devices": [asdict(device) for device in devices],
				"count": len(devices),
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def record_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Record sensor reading"""
		try:
			reading = SensorReading(
				sensor_id=sensor_data["sensor_id"],
				sensor_type=SensorType(sensor_data["sensor_type"]),
				value=sensor_data["value"],
				unit=sensor_data.get("unit", ""),
				quality=sensor_data.get("quality", 1.0),
				metadata=sensor_data.get("metadata", {})
			)
			
			success = await self.sensor_data.record_reading(reading)
			
			return {
				"success": success,
				"sensor_id": reading.sensor_id,
				"timestamp": reading.timestamp.isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def get_sensor_readings(
		self,
		device_id: str,
		sensor_type: str = None,
		hours_back: int = 24,
		limit: int = 1000
	) -> Dict[str, Any]:
		"""Get sensor readings for a device"""
		try:
			sensor_type_enum = SensorType(sensor_type) if sensor_type else None
			start_time = datetime.utcnow() - timedelta(hours=hours_back)
			
			readings = await self.sensor_data.get_readings(
				device_id=device_id,
				sensor_type=sensor_type_enum,
				start_time=start_time,
				limit=limit
			)
			
			return {
				"success": True,
				"device_id": device_id,
				"readings": [asdict(reading) for reading in readings],
				"count": len(readings),
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def get_sensor_statistics(
		self,
		device_id: str,
		sensor_type: str,
		hours_back: int = 24
	) -> Dict[str, Any]:
		"""Get sensor data statistics"""
		try:
			sensor_type_enum = SensorType(sensor_type)
			start_time = datetime.utcnow() - timedelta(hours=hours_back)
			
			stats = await self.sensor_data.calculate_statistics(
				device_id=device_id,
				sensor_type=sensor_type_enum,
				start_time=start_time
			)
			
			return {
				"success": True,
				"device_id": device_id,
				"sensor_type": sensor_type,
				"statistics": stats,
				"period_hours": hours_back,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def send_device_command(
		self,
		device_id: str,
		command: str,
		parameters: Dict[str, Any] = None
	) -> Dict[str, Any]:
		"""Send command to device"""
		try:
			device_command = DeviceCommand(
				device_id=device_id,
				command=command,
				parameters=parameters or {}
			)
			
			success = await self.device_manager.send_command(device_command)
			
			return {
				"success": success,
				"command_id": device_command.command_id,
				"device_id": device_id,
				"command": command,
				"timestamp": device_command.timestamp.isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	async def create_alert_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Create monitoring alert rule"""
		try:
			rule = AlertRule(
				name=rule_data["name"],
				description=rule_data.get("description", ""),
				device_id=rule_data.get("device_id", ""),
				sensor_type=SensorType(rule_data["sensor_type"]) if rule_data.get("sensor_type") else None,
				condition=rule_data["condition"],
				threshold_value=rule_data.get("threshold_value"),
				action=rule_data.get("action", "log"),
				action_config=rule_data.get("action_config", {}),
				enabled=rule_data.get("enabled", True)
			)
			
			success = self.alert_manager.add_alert_rule(rule)
			
			return {
				"success": success,
				"rule_id": rule.rule_id,
				"name": rule.name,
				"timestamp": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			return {
				"success": False,
				"error": str(e),
				"timestamp": datetime.utcnow().isoformat()
			}
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""Get capability information"""
		return {
			"name": "iot_management",
			"version": "1.0.0",
			"description": "Comprehensive IoT device management, sensor data collection, and real-time monitoring",
			"features": [
				"Device registration and management",
				"Real-time sensor data collection",
				"MQTT connectivity support",
				"Alert and monitoring system",
				"Device command and control",
				"Data analytics and statistics",
				"Multi-protocol device support"
			],
			"supported_devices": [dt.value for dt in DeviceType],
			"supported_sensors": [st.value for st in SensorType],
			"connection_types": [ct.value for ct in ConnectionType],
			"dependencies": {
				"sqlite3": "built-in",
				"paho-mqtt": ">=1.5.0 (optional)",
				"pyserial": ">=3.4 (optional)",
				"requests": ">=2.25.0 (optional)",
				"websocket-client": ">=1.0.0 (optional)"
			}
		}

# APG Integration
CAPABILITY_INFO = {
	"name": "iot_management",
	"version": "1.0.0",
	"provides": ["IoTManagementCapability"],
	"integrates_with": ["flask", "fastapi", "django", "mqtt", "influxdb"],
	"apg_templates": ["iot_platform", "sensor_dashboard", "smart_home", "industrial_monitoring"],
	"category": "iot_hardware",
	"tags": ["iot", "sensors", "mqtt", "device-management", "monitoring", "alerts"]
}