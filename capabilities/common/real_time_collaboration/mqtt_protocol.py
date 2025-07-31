"""
MQTT Protocol Implementation for APG Real-Time Collaboration

Provides MQTT publish/subscribe messaging for IoT device integration,
sensor data streaming, and distributed collaboration events.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path

try:
	import paho.mqtt.client as mqtt
	import asyncio_mqtt
	from asyncio_mqtt import Client as AsyncMQTTClient
except ImportError:
	print("MQTT dependencies not installed. Run: pip install paho-mqtt asyncio-mqtt")
	mqtt = None
	asyncio_mqtt = None
	AsyncMQTTClient = None

logger = logging.getLogger(__name__)


class MQTTQoS(Enum):
	"""MQTT Quality of Service levels"""
	AT_MOST_ONCE = 0	# Fire and forget
	AT_LEAST_ONCE = 1	# Acknowledged delivery
	EXACTLY_ONCE = 2	# Assured delivery


class MQTTTopicType(Enum):
	"""MQTT topic types for collaboration"""
	PRESENCE = "presence"
	COLLABORATION = "collaboration"
	IOT_SENSORS = "iot/sensors"
	DEVICE_STATUS = "device/status"
	ALERTS = "alerts"
	ANALYTICS = "analytics"
	FILE_SYNC = "file/sync"
	WORKFLOW_EVENTS = "workflow/events"


@dataclass
class MQTTMessage:
	"""MQTT message structure"""
	topic: str
	payload: Dict[str, Any]
	qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE
	retain: bool = False
	timestamp: datetime = field(default_factory=datetime.utcnow)
	message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"topic": self.topic,
			"payload": self.payload,
			"qos": self.qos.value,
			"retain": self.retain,
			"timestamp": self.timestamp.isoformat(),
			"message_id": self.message_id
		}


@dataclass
class MQTTSubscription:
	"""MQTT subscription tracking"""
	topic: str
	qos: MQTTQoS
	callback: Callable[[MQTTMessage], None]
	subscriber_id: str
	created_at: datetime = field(default_factory=datetime.utcnow)
	message_count: int = 0


class MQTTProtocolManager:
	"""Manages MQTT protocol for real-time collaboration"""
	
	def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
				 client_id: str = None, username: str = None, password: str = None):
		self.broker_host = broker_host
		self.broker_port = broker_port
		self.client_id = client_id or f"apg_rtc_{uuid.uuid4().hex[:8]}"
		self.username = username
		self.password = password
		
		# Client management
		self.mqtt_client: Optional[AsyncMQTTClient] = None
		self.is_connected = False
		self.connection_attempts = 0
		self.max_connection_attempts = 5
		
		# Subscription management
		self.subscriptions: Dict[str, MQTTSubscription] = {}
		self.topic_handlers: Dict[str, List[Callable]] = {}
		
		# Message tracking
		self.message_history: List[MQTTMessage] = []
		self.max_history_size = 1000
		
		# APG integration
		self.apg_namespace = "apg/rtc"
		self.session_id = str(uuid.uuid4())
		
		# Statistics
		self.stats = {
			"messages_published": 0,
			"messages_received": 0,
			"bytes_sent": 0,
			"bytes_received": 0,
			"connection_uptime": None,
			"last_activity": None
		}
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize MQTT connection"""
		try:
			if not AsyncMQTTClient:
				return {"error": "MQTT dependencies not installed"}
			
			# Create MQTT client
			self.mqtt_client = AsyncMQTTClient(
				hostname=self.broker_host,
				port=self.broker_port,
				client_id=self.client_id,
				username=self.username,
				password=self.password,
				keepalive=60,
				will_topic=f"{self.apg_namespace}/presence/{self.client_id}/status",
				will_payload="offline",
				will_qos=1,
				will_retain=True
			)
			
			# Connect to broker
			await self.mqtt_client.__aenter__()
			self.is_connected = True
			self.stats["connection_uptime"] = datetime.utcnow()
			
			# Subscribe to system topics
			await self._subscribe_system_topics()
			
			# Announce presence
			await self.publish_presence_update("online")
			
			logger.info(f"MQTT client connected to {self.broker_host}:{self.broker_port}")
			
			return {
				"status": "connected",
				"client_id": self.client_id,
				"broker": f"{self.broker_host}:{self.broker_port}",
				"session_id": self.session_id
			}
			
		except Exception as e:
			logger.error(f"Failed to initialize MQTT client: {e}")
			self.is_connected = False
			return {"error": f"MQTT initialization failed: {str(e)}"}
	
	async def disconnect(self) -> Dict[str, Any]:
		"""Disconnect from MQTT broker"""
		try:
			if self.mqtt_client and self.is_connected:
				# Announce offline status
				await self.publish_presence_update("offline")
				
				# Disconnect
				await self.mqtt_client.__aexit__(None, None, None)
				self.is_connected = False
				
				logger.info("MQTT client disconnected")
				
			return {"status": "disconnected"}
			
		except Exception as e:
			logger.error(f"Error disconnecting MQTT client: {e}")
			return {"error": f"Disconnect failed: {str(e)}"}
	
	async def publish_message(self, topic: str, payload: Dict[str, Any], 
							  qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE, 
							  retain: bool = False) -> Dict[str, Any]:
		"""Publish message to MQTT topic"""
		try:
			if not self.is_connected or not self.mqtt_client:
				return {"error": "MQTT client not connected"}
			
			# Create message
			message = MQTTMessage(
				topic=topic,
				payload=payload,
				qos=qos,
				retain=retain
			)
			
			# Add APG context
			enhanced_payload = {
				**payload,
				"_apg_metadata": {
					"client_id": self.client_id,
					"session_id": self.session_id,
					"timestamp": message.timestamp.isoformat(),
					"message_id": message.message_id
				}
			}
			
			# Publish message
			payload_json = json.dumps(enhanced_payload)
			await self.mqtt_client.publish(
				topic, 
				payload_json, 
				qos=qos.value, 
				retain=retain
			)
			
			# Update statistics
			self.stats["messages_published"] += 1
			self.stats["bytes_sent"] += len(payload_json)
			self.stats["last_activity"] = datetime.utcnow()
			
			# Store in history
			self._add_to_history(message)
			
			logger.debug(f"Published MQTT message to {topic}")
			
			return {
				"status": "published",
				"topic": topic,
				"message_id": message.message_id,
				"size_bytes": len(payload_json)
			}
			
		except Exception as e:
			logger.error(f"Failed to publish MQTT message: {e}")
			return {"error": f"Publish failed: {str(e)}"}
	
	async def subscribe_topic(self, topic: str, callback: Callable[[MQTTMessage], None], 
							  qos: MQTTQoS = MQTTQoS.AT_LEAST_ONCE,
							  subscriber_id: str = None) -> Dict[str, Any]:
		"""Subscribe to MQTT topic"""
		try:
			if not self.is_connected or not self.mqtt_client:
				return {"error": "MQTT client not connected"}
			
			subscriber_id = subscriber_id or str(uuid.uuid4())
			
			# Subscribe to topic
			await self.mqtt_client.subscribe(topic, qos=qos.value)
			
			# Store subscription
			subscription = MQTTSubscription(
				topic=topic,
				qos=qos,
				callback=callback,
				subscriber_id=subscriber_id
			)
			self.subscriptions[f"{topic}:{subscriber_id}"] = subscription
			
			# Add to topic handlers
			if topic not in self.topic_handlers:
				self.topic_handlers[topic] = []
			self.topic_handlers[topic].append(callback)
			
			logger.info(f"Subscribed to MQTT topic: {topic}")
			
			return {
				"status": "subscribed",
				"topic": topic,
				"subscriber_id": subscriber_id,
				"qos": qos.value
			}
			
		except Exception as e:
			logger.error(f"Failed to subscribe to MQTT topic: {e}")
			return {"error": f"Subscribe failed: {str(e)}"}
	
	async def unsubscribe_topic(self, topic: str, subscriber_id: str = None) -> Dict[str, Any]:
		"""Unsubscribe from MQTT topic"""
		try:
			if not self.is_connected or not self.mqtt_client:
				return {"error": "MQTT client not connected"}
			
			# Remove specific subscription or all for topic
			if subscriber_id:
				subscription_key = f"{topic}:{subscriber_id}"
				if subscription_key in self.subscriptions:
					subscription = self.subscriptions[subscription_key]
					self.topic_handlers[topic].remove(subscription.callback)
					del self.subscriptions[subscription_key]
			else:
				# Remove all subscriptions for topic
				keys_to_remove = [k for k in self.subscriptions.keys() if k.startswith(f"{topic}:")]
				for key in keys_to_remove:
					del self.subscriptions[key]
				self.topic_handlers.pop(topic, None)
			
			# Unsubscribe from broker if no more handlers
			if topic not in self.topic_handlers or not self.topic_handlers[topic]:
				await self.mqtt_client.unsubscribe(topic)
			
			logger.info(f"Unsubscribed from MQTT topic: {topic}")
			
			return {
				"status": "unsubscribed",
				"topic": topic,
				"subscriber_id": subscriber_id
			}
			
		except Exception as e:
			logger.error(f"Failed to unsubscribe from MQTT topic: {e}")
			return {"error": f"Unsubscribe failed: {str(e)}"}
	
	async def _subscribe_system_topics(self):
		"""Subscribe to system-level MQTT topics"""
		system_topics = [
			f"{self.apg_namespace}/presence/+/status",
			f"{self.apg_namespace}/collaboration/+/events",
			f"{self.apg_namespace}/system/broadcasts"
		]
		
		for topic in system_topics:
			await self.subscribe_topic(topic, self._handle_system_message)
	
	async def _handle_system_message(self, message: MQTTMessage):
		"""Handle system-level MQTT messages"""
		try:
			topic_parts = message.topic.split('/')
			
			if "presence" in topic_parts:
				await self._handle_presence_message(message)
			elif "collaboration" in topic_parts:
				await self._handle_collaboration_message(message)
			elif "system" in topic_parts:
				await self._handle_system_broadcast(message)
			
		except Exception as e:
			logger.error(f"Error handling system MQTT message: {e}")
	
	async def _handle_presence_message(self, message: MQTTMessage):
		"""Handle presence-related MQTT messages"""
		try:
			payload = message.payload
			if isinstance(payload, dict):
				client_id = payload.get("client_id")
				status = payload.get("status")
				
				logger.info(f"Presence update: {client_id} is {status}")
				
				# Notify other components about presence change
				# This would integrate with the WebSocket manager
				
		except Exception as e:
			logger.error(f"Error handling presence message: {e}")
	
	async def _handle_collaboration_message(self, message: MQTTMessage):
		"""Handle collaboration-related MQTT messages"""
		try:
			payload = message.payload
			event_type = payload.get("event_type")
			
			logger.debug(f"Collaboration event: {event_type}")
			
			# Process collaboration events
			# This would integrate with the collaboration system
			
		except Exception as e:
			logger.error(f"Error handling collaboration message: {e}")
	
	async def _handle_system_broadcast(self, message: MQTTMessage):
		"""Handle system broadcast messages"""
		try:
			payload = message.payload
			broadcast_type = payload.get("broadcast_type")
			
			logger.info(f"System broadcast: {broadcast_type}")
			
		except Exception as e:
			logger.error(f"Error handling system broadcast: {e}")
	
	async def publish_presence_update(self, status: str, metadata: Dict[str, Any] = None):
		"""Publish presence update via MQTT"""
		topic = f"{self.apg_namespace}/presence/{self.client_id}/status"
		payload = {
			"client_id": self.client_id,
			"status": status,
			"timestamp": datetime.utcnow().isoformat(),
			"metadata": metadata or {}
		}
		
		return await self.publish_message(topic, payload, retain=True)
	
	async def publish_collaboration_event(self, event_type: str, event_data: Dict[str, Any]):
		"""Publish collaboration event via MQTT"""
		topic = f"{self.apg_namespace}/collaboration/{self.session_id}/events"
		payload = {
			"event_type": event_type,
			"event_data": event_data,
			"client_id": self.client_id,
			"session_id": self.session_id,
			"timestamp": datetime.utcnow().isoformat()
		}
		
		return await self.publish_message(topic, payload)
	
	async def publish_iot_sensor_data(self, device_id: str, sensor_type: str, 
									  sensor_data: Dict[str, Any]):
		"""Publish IoT sensor data via MQTT"""
		topic = f"{self.apg_namespace}/iot/sensors/{device_id}/{sensor_type}"
		payload = {
			"device_id": device_id,
			"sensor_type": sensor_type,
			"sensor_data": sensor_data,
			"timestamp": datetime.utcnow().isoformat()
		}
		
		return await self.publish_message(topic, payload)
	
	async def publish_workflow_event(self, workflow_id: str, event_type: str, 
									 event_data: Dict[str, Any]):
		"""Publish workflow event via MQTT"""
		topic = f"{self.apg_namespace}/workflow/events/{workflow_id}"
		payload = {
			"workflow_id": workflow_id,
			"event_type": event_type,
			"event_data": event_data,
			"client_id": self.client_id,
			"timestamp": datetime.utcnow().isoformat()
		}
		
		return await self.publish_message(topic, payload)
	
	def _add_to_history(self, message: MQTTMessage):
		"""Add message to history with size management"""
		self.message_history.append(message)
		
		# Maintain history size limit
		if len(self.message_history) > self.max_history_size:
			self.message_history = self.message_history[-self.max_history_size:]
	
	def get_subscriptions(self) -> List[Dict[str, Any]]:
		"""Get current MQTT subscriptions"""
		return [
			{
				"topic": sub.topic,
				"qos": sub.qos.value,
				"subscriber_id": sub.subscriber_id,
				"created_at": sub.created_at.isoformat(),
				"message_count": sub.message_count
			}
			for sub in self.subscriptions.values()
		]
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get MQTT protocol statistics"""
		uptime_seconds = 0
		if self.stats["connection_uptime"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["connection_uptime"]).total_seconds())
		
		return {
			**self.stats,
			"is_connected": self.is_connected,
			"client_id": self.client_id,
			"broker": f"{self.broker_host}:{self.broker_port}",
			"active_subscriptions": len(self.subscriptions),
			"message_history_size": len(self.message_history),
			"uptime_seconds": uptime_seconds,
			"connection_uptime": self.stats["connection_uptime"].isoformat() if self.stats["connection_uptime"] else None,
			"last_activity": self.stats["last_activity"].isoformat() if self.stats["last_activity"] else None
		}
	
	async def start_message_listener(self):
		"""Start listening for incoming MQTT messages"""
		if not self.mqtt_client or not self.is_connected:
			logger.error("Cannot start message listener: MQTT client not connected")
			return
		
		try:
			async with self.mqtt_client.filtered_messages("") as messages:
				async for mqtt_message in messages:
					try:
						# Parse message
						payload = json.loads(mqtt_message.payload.decode())
						
						# Create MQTTMessage object
						message = MQTTMessage(
							topic=mqtt_message.topic,
							payload=payload,
							qos=MQTTQoS(mqtt_message.qos)
						)
						
						# Update statistics
						self.stats["messages_received"] += 1
						self.stats["bytes_received"] += len(mqtt_message.payload)
						self.stats["last_activity"] = datetime.utcnow()
						
						# Add to history
						self._add_to_history(message)
						
						# Route message to handlers
						await self._route_message(message)
						
					except Exception as e:
						logger.error(f"Error processing MQTT message: {e}")
						
		except Exception as e:
			logger.error(f"Error in MQTT message listener: {e}")
	
	async def _route_message(self, message: MQTTMessage):
		"""Route incoming MQTT message to appropriate handlers"""
		try:
			# Find matching topic handlers
			for topic_pattern, handlers in self.topic_handlers.items():
				if self._topic_matches(message.topic, topic_pattern):
					for handler in handlers:
						try:
							if asyncio.iscoroutinefunction(handler):
								await handler(message)
							else:
								handler(message)
						except Exception as e:
							logger.error(f"Error in MQTT message handler: {e}")
			
		except Exception as e:
			logger.error(f"Error routing MQTT message: {e}")
	
	def _topic_matches(self, topic: str, pattern: str) -> bool:
		"""Check if topic matches MQTT topic pattern"""
		# Simple MQTT wildcard matching
		topic_parts = topic.split('/')
		pattern_parts = pattern.split('/')
		
		if len(pattern_parts) > len(topic_parts):
			return False
		
		for i, pattern_part in enumerate(pattern_parts):
			if pattern_part == '#':
				return True  # Multi-level wildcard
			elif pattern_part == '+':
				continue  # Single-level wildcard
			elif i >= len(topic_parts) or pattern_part != topic_parts[i]:
				return False
		
		return len(pattern_parts) == len(topic_parts)


# Global MQTT manager instance
mqtt_protocol_manager = None


async def initialize_mqtt_protocol(broker_host: str = "localhost", 
								   broker_port: int = 1883,
								   client_id: str = None,
								   username: str = None, 
								   password: str = None) -> Dict[str, Any]:
	"""Initialize global MQTT protocol manager"""
	global mqtt_protocol_manager
	
	mqtt_protocol_manager = MQTTProtocolManager(
		broker_host=broker_host,
		broker_port=broker_port,
		client_id=client_id,
		username=username,
		password=password
	)
	
	result = await mqtt_protocol_manager.initialize()
	
	if result.get("status") == "connected":
		# Start message listener in background
		asyncio.create_task(mqtt_protocol_manager.start_message_listener())
	
	return result


def get_mqtt_manager() -> Optional[MQTTProtocolManager]:
	"""Get global MQTT protocol manager"""
	return mqtt_protocol_manager


if __name__ == "__main__":
	# Test MQTT protocol implementation
	async def test_mqtt():
		print("Testing MQTT protocol implementation...")
		
		# Initialize MQTT manager
		result = await initialize_mqtt_protocol()
		print(f"MQTT initialization result: {result}")
		
		if result.get("status") == "connected":
			manager = get_mqtt_manager()
			
			# Test message publishing
			pub_result = await manager.publish_collaboration_event(
				"test_event",
				{"data": "test collaboration event"}
			)
			print(f"Publish result: {pub_result}")
			
			# Test presence update
			presence_result = await manager.publish_presence_update(
				"active",
				{"location": "test_session"}
			)
			print(f"Presence result: {presence_result}")
			
			# Test IoT sensor data
			iot_result = await manager.publish_iot_sensor_data(
				"sensor_001",
				"temperature",
				{"value": 23.5, "unit": "celsius"}
			)
			print(f"IoT result: {iot_result}")
			
			# Get statistics
			stats = manager.get_statistics()
			print(f"MQTT statistics: {stats}")
			
			# Wait a bit for message processing
			await asyncio.sleep(2)
			
			# Disconnect
			disconnect_result = await manager.disconnect()
			print(f"Disconnect result: {disconnect_result}")
		
		print("✅ MQTT protocol test completed")
	
	asyncio.run(test_mqtt())