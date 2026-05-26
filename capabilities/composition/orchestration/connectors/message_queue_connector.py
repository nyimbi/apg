"""
APG Workflow Orchestration Message Queue Connectors

High-performance message queue connectors for Bytewax, RabbitMQ, Redis, and
other messaging systems with producer/consumer support and reliable delivery.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timezone
import logging

# RabbitMQ
import aio_pika
from aio_pika.exceptions import AMQPException

# Redis
import redis.asyncio as redis
from redis.exceptions import RedisError

from pydantic import Field

from .base_connector import BaseConnector, ConnectorConfiguration

logger = logging.getLogger(__name__)

class BytewaxConfiguration(ConnectorConfiguration):
	"""Bytewax stream connector configuration."""

	stream_names: List[str] = Field(default_factory=lambda: ["workflow-events"], description="Bytewax stream names")
	flow_id: str = Field(default="apg-workflow-orchestration")
	consumer_name: Optional[str] = Field(default=None, description="Logical consumer name")
	max_records: int = Field(default=500, ge=1, le=10000)
	replay_from_start: bool = Field(default=False)

class RabbitMQConfiguration(ConnectorConfiguration):
	"""RabbitMQ message queue configuration."""
	
	host: str = Field(default="localhost")
	port: int = Field(default=5672, ge=1, le=65535)
	virtual_host: str = Field(default="/")
	username: str = Field(default="guest")
	password: str = Field(default="guest")
	ssl: bool = Field(default=False)
	ssl_context: Optional[Dict[str, Any]] = Field(default=None)
	heartbeat: int = Field(default=60, ge=0, le=3600)
	connection_attempts: int = Field(default=3, ge=1, le=10)
	retry_delay: float = Field(default=2.0, ge=0.1, le=60.0)
	prefetch_count: int = Field(default=10, ge=1, le=1000)
	confirm_delivery: bool = Field(default=True)
	mandatory_publish: bool = Field(default=False)

class RedisQueueConfiguration(ConnectorConfiguration):
	"""Redis message queue configuration."""
	
	host: str = Field(default="localhost")
	port: int = Field(default=6379, ge=1, le=65535)
	db: int = Field(default=0, ge=0, le=15)
	password: Optional[str] = Field(default=None)
	ssl: bool = Field(default=False)
	ssl_cert_reqs: str = Field(default="required")
	ssl_ca_certs: Optional[str] = Field(default=None)
	ssl_certfile: Optional[str] = Field(default=None)
	ssl_keyfile: Optional[str] = Field(default=None)
	max_connections: int = Field(default=50, ge=1, le=1000)
	socket_keepalive: bool = Field(default=True)
	socket_keepalive_options: Dict[str, int] = Field(default_factory=dict)
	decode_responses: bool = Field(default=True)
	encoding: str = Field(default="utf-8")
	stream_maxlen: int = Field(default=10000, ge=100)
	consumer_group: str = Field(default="workflow-orchestration")
	consumer_name: str = Field(default="consumer-1")

class BytewaxConnector(BaseConnector):
	"""Dependency-light Bytewax stream connector."""
	
	def __init__(self, config: BytewaxConfiguration):
		super().__init__(config)
		self.config: BytewaxConfiguration = config
		self.streams: Dict[str, List[Dict[str, Any]]] = {}
		self.subscribed_streams: Set[str] = set()
		self.stream_cursors: Dict[str, int] = {}
		self.message_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
		self.consumer_task: Optional[asyncio.Task] = None
		self.is_consuming = False
	
	async def _connect(self) -> None:
		"""Initialize Bytewax stream ledgers."""
		self.streams = {stream_name: [] for stream_name in self.config.stream_names}
		logger.info(self._log_connector_info("Bytewax connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close Bytewax stream connector."""
		# Stop consuming
		if self.consumer_task:
			self.consumer_task.cancel()
			await asyncio.gather(self.consumer_task, return_exceptions=True)
		
		self.subscribed_streams.clear()
		self.is_consuming = False
		logger.info(self._log_connector_info("Bytewax connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Bytewax stream operation."""
		
		if operation == "produce":
			return await self._produce_message(parameters)
		elif operation == "consume":
			return await self._consume_messages(parameters)
		elif operation == "subscribe":
			return await self._subscribe_streams(parameters)
		elif operation == "unsubscribe":
			return await self._unsubscribe_streams(parameters)
		else:
			raise ValueError(f"Unsupported Bytewax operation: {operation}")
	
	async def _produce_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Emit item to a Bytewax stream."""
		
		stream = params.get("stream")
		message = params.get("message")
		key = params.get("key")
		headers = params.get("headers", {})

		if not stream or message is None:
			raise ValueError("Stream and message are required for Bytewax produce operation")

		self.streams.setdefault(stream, [])
		record = {
			"stream": stream,
			"sequence": len(self.streams[stream]),
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"key": key,
			"value": message,
			"headers": headers
		}
		self.streams[stream].append(record)
			
		return {
			"stream": stream,
			"sequence": record["sequence"],
			"timestamp": record["timestamp"],
			"success": True
		}
	
	async def _consume_messages(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Consume items from Bytewax streams."""
		
		streams = params.get("streams") or list(self.subscribed_streams) or self.config.stream_names
		max_records = params.get("max_records", self.config.max_records)
		start_sequence = 0 if self.config.replay_from_start else params.get("start_sequence", 0)
		messages = []

		for stream in streams:
			for record in self.streams.get(stream, []):
				if record["sequence"] >= start_sequence:
					messages.append(record)
				if len(messages) >= max_records:
					break
			if len(messages) >= max_records:
				break
		
		return {
			"messages": messages,
			"count": len(messages),
			"success": True
		}
	
	async def _subscribe_streams(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Subscribe to Bytewax streams."""
		
		streams = params.get("streams", [])
		if not streams:
			raise ValueError("Streams list is required for subscribe operation")
		
		for stream in streams:
			self.streams.setdefault(stream, [])
			self.subscribed_streams.add(stream)
			self.stream_cursors.setdefault(stream, 0 if self.config.replay_from_start else len(self.streams[stream]))

		# Start consuming task if not already running
		if not self.is_consuming:
			self.consumer_task = asyncio.create_task(self._consume_loop())
			self.is_consuming = True

		return {
			"subscribed_streams": streams,
			"success": True
		}
	
	async def _unsubscribe_streams(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Unsubscribe from Bytewax streams."""
		
		streams = params.get("streams")
		if streams:
			for stream in streams:
				self.subscribed_streams.discard(stream)
				self.stream_cursors.pop(stream, None)
		else:
			self.subscribed_streams.clear()
			self.stream_cursors.clear()
		
		# Stop consuming task if no streams remain
		if not self.subscribed_streams and self.consumer_task:
			self.consumer_task.cancel()
			self.is_consuming = False
		
		return {"success": True}
	
	async def _consume_loop(self) -> None:
		"""Background message consumption loop."""
		while self.is_consuming:
			try:
				# Consume messages and call handlers
				for stream in list(self.subscribed_streams):
					cursor = self.stream_cursors.get(stream, 0)
					for message in self.streams.get(stream, []):
						if message["sequence"] < cursor:
							continue
						if stream in self.message_handlers:
							try:
								await self.message_handlers[stream](message)
							except Exception as e:
								logger.error(self._log_connector_info(f"Message handler error for stream {stream}: {e}"))
						self.stream_cursors[stream] = message["sequence"] + 1
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(self._log_connector_info(f"Consume loop error: {e}"))
				await asyncio.sleep(1)
			await asyncio.sleep(0.1)
	
	async def _health_check(self) -> bool:
		"""Check Bytewax stream connector readiness."""
		return self.streams is not None
	
	def add_message_handler(self, stream: str, handler: Callable) -> None:
		"""Add message handler for a specific stream."""
		self.message_handlers[stream] = handler

class RabbitMQConnector(BaseConnector):
	"""High-performance RabbitMQ message queue connector."""
	
	def __init__(self, config: RabbitMQConfiguration):
		super().__init__(config)
		self.config: RabbitMQConfiguration = config
		self.connection: Optional[aio_pika.Connection] = None
		self.channel: Optional[aio_pika.Channel] = None
		self.exchanges: Dict[str, aio_pika.Exchange] = {}
		self.queues: Dict[str, aio_pika.Queue] = {}
	
	async def _connect(self) -> None:
		"""Initialize RabbitMQ connection."""
		
		# Build connection URL
		url = f"amqp://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}{self.config.virtual_host}"
		
		if self.config.ssl:
			url = url.replace("amqp://", "amqps://")
		
		try:
			self.connection = await aio_pika.connect_robust(
				url,
				heartbeat=self.config.heartbeat,
				connection_attempts=self.config.connection_attempts,
				retry_delay=self.config.retry_delay
			)
			
			self.channel = await self.connection.channel()
			await self.channel.set_qos(prefetch_count=self.config.prefetch_count)
			
			if self.config.confirm_delivery:
				await self.channel.confirm_delivery()
			
			logger.info(self._log_connector_info("RabbitMQ connector initialized"))
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to connect to RabbitMQ: {e}"))
			raise
	
	async def _disconnect(self) -> None:
		"""Close RabbitMQ connection."""
		
		if self.channel:
			await self.channel.close()
			self.channel = None
		
		if self.connection:
			await self.connection.close()
			self.connection = None
		
		self.exchanges.clear()
		self.queues.clear()
		
		logger.info(self._log_connector_info("RabbitMQ connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute RabbitMQ operation."""
		
		if operation == "publish":
			return await self._publish_message(parameters)
		elif operation == "consume":
			return await self._consume_message(parameters)
		elif operation == "declare_exchange":
			return await self._declare_exchange(parameters)
		elif operation == "declare_queue":
			return await self._declare_queue(parameters)
		elif operation == "bind_queue":
			return await self._bind_queue(parameters)
		else:
			raise ValueError(f"Unsupported RabbitMQ operation: {operation}")
	
	async def _publish_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Publish message to RabbitMQ exchange."""
		
		exchange_name = params.get("exchange", "")
		routing_key = params.get("routing_key", "")
		message = params.get("message")
		properties = params.get("properties", {})
		
		if message is None:
			raise ValueError("Message is required for RabbitMQ publish operation")
		
		# Serialize message
		if isinstance(message, dict):
			message_body = json.dumps(message).encode("utf-8")
			content_type = "application/json"
		elif isinstance(message, str):
			message_body = message.encode("utf-8")
			content_type = "text/plain"
		else:
			message_body = message
			content_type = "application/octet-stream"
		
		# Create message
		message_obj = aio_pika.Message(
			message_body,
			content_type=content_type,
			**properties
		)
		
		try:
			# Get or create exchange
			if exchange_name:
				if exchange_name not in self.exchanges:
					self.exchanges[exchange_name] = await self.channel.get_exchange(exchange_name)
				exchange = self.exchanges[exchange_name]
			else:
				exchange = self.channel.default_exchange
			
			# Publish message
			await exchange.publish(
				message_obj,
				routing_key=routing_key,
				mandatory=self.config.mandatory_publish
			)
			
			return {
				"exchange": exchange_name,
				"routing_key": routing_key,
				"message_size": len(message_body),
				"success": True
			}
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to publish message: {e}"))
			raise
	
	async def _consume_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Consume message from RabbitMQ queue."""
		
		queue_name = params.get("queue")
		no_ack = params.get("no_ack", False)
		timeout = params.get("timeout", 1.0)
		
		if not queue_name:
			raise ValueError("Queue name is required for RabbitMQ consume operation")
		
		try:
			# Get or declare queue
			if queue_name not in self.queues:
				self.queues[queue_name] = await self.channel.declare_queue(queue_name, passive=True)
			queue = self.queues[queue_name]
			
			# Get message
			message = await queue.get(timeout=timeout, no_ack=no_ack)
			
			if message:
				# Deserialize message body
				try:
					if message.content_type == "application/json":
						body = json.loads(message.body.decode("utf-8"))
					else:
						body = message.body.decode("utf-8")
				except (json.JSONDecodeError, UnicodeDecodeError):
					body = message.body
				
				return {
					"message_id": message.message_id,
					"body": body,
					"properties": {
						"content_type": message.content_type,
						"delivery_mode": message.delivery_mode,
						"priority": message.priority,
						"correlation_id": message.correlation_id,
						"reply_to": message.reply_to,
						"expiration": message.expiration,
						"timestamp": message.timestamp.isoformat() if message.timestamp else None,
						"headers": dict(message.headers) if message.headers else {}
					},
					"routing_key": message.routing_key,
					"exchange": message.exchange,
					"delivery_tag": message.delivery_tag,
					"success": True
				}
			else:
				return {"message": None, "success": True}
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to consume message: {e}"))
			raise
	
	async def _declare_exchange(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Declare RabbitMQ exchange."""
		
		name = params.get("name")
		exchange_type = params.get("type", "direct")
		durable = params.get("durable", True)
		auto_delete = params.get("auto_delete", False)
		
		if not name:
			raise ValueError("Exchange name is required")
		
		try:
			exchange = await self.channel.declare_exchange(
				name,
				type=exchange_type,
				durable=durable,
				auto_delete=auto_delete
			)
			self.exchanges[name] = exchange
			
			return {
				"name": name,
				"type": exchange_type,
				"durable": durable,
				"auto_delete": auto_delete,
				"success": True
			}
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to declare exchange: {e}"))
			raise
	
	async def _declare_queue(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Declare RabbitMQ queue."""
		
		name = params.get("name", "")
		durable = params.get("durable", True)
		exclusive = params.get("exclusive", False)
		auto_delete = params.get("auto_delete", False)
		
		try:
			queue = await self.channel.declare_queue(
				name,
				durable=durable,
				exclusive=exclusive,
				auto_delete=auto_delete
			)
			self.queues[queue.name] = queue
			
			return {
				"name": queue.name,
				"durable": durable,
				"exclusive": exclusive,
				"auto_delete": auto_delete,
				"success": True
			}
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to declare queue: {e}"))
			raise
	
	async def _bind_queue(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Bind queue to exchange."""
		
		queue_name = params.get("queue")
		exchange_name = params.get("exchange")
		routing_key = params.get("routing_key", "")
		
		if not queue_name or not exchange_name:
			raise ValueError("Queue and exchange names are required for binding")
		
		try:
			queue = self.queues.get(queue_name)
			exchange = self.exchanges.get(exchange_name)
			
			if not queue:
				raise ValueError(f"Queue {queue_name} not found")
			if not exchange:
				raise ValueError(f"Exchange {exchange_name} not found")
			
			await queue.bind(exchange, routing_key)
			
			return {
				"queue": queue_name,
				"exchange": exchange_name,
				"routing_key": routing_key,
				"success": True
			}
		
		except AMQPException as e:
			logger.error(self._log_connector_info(f"Failed to bind queue: {e}"))
			raise
	
	async def _health_check(self) -> bool:
		"""Check RabbitMQ connectivity."""
		try:
			if self.connection and not self.connection.is_closed:
				# Try to declare a temporary queue
				temp_queue = await self.channel.declare_queue("", exclusive=True, auto_delete=True)
				await temp_queue.delete()
				return True
			return False
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False

class RedisQueueConnector(BaseConnector):
	"""High-performance Redis message queue connector."""
	
	def __init__(self, config: RedisQueueConfiguration):
		super().__init__(config)
		self.config: RedisQueueConfiguration = config
		self.redis_client: Optional[redis.Redis] = None
	
	async def _connect(self) -> None:
		"""Initialize Redis connection."""
		
		connection_params = {
			"host": self.config.host,
			"port": self.config.port,
			"db": self.config.db,
			"password": self.config.password,
			"ssl": self.config.ssl,
			"max_connections": self.config.max_connections,
			"socket_keepalive": self.config.socket_keepalive,
			"socket_keepalive_options": self.config.socket_keepalive_options,
			"decode_responses": self.config.decode_responses,
			"encoding": self.config.encoding
		}
		
		if self.config.ssl:
			ssl_params = {
				"ssl_cert_reqs": self.config.ssl_cert_reqs
			}
			if self.config.ssl_ca_certs:
				ssl_params["ssl_ca_certs"] = self.config.ssl_ca_certs
			if self.config.ssl_certfile:
				ssl_params["ssl_certfile"] = self.config.ssl_certfile
			if self.config.ssl_keyfile:
				ssl_params["ssl_keyfile"] = self.config.ssl_keyfile
			
			connection_params.update(ssl_params)
		
		self.redis_client = redis.Redis(**connection_params)
		
		# Test connection
		await self.redis_client.ping()
		
		logger.info(self._log_connector_info("Redis queue connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close Redis connection."""
		if self.redis_client:
			await self.redis_client.close()
			self.redis_client = None
		
		logger.info(self._log_connector_info("Redis queue connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Redis queue operation."""
		
		if operation == "push":
			return await self._push_message(parameters)
		elif operation == "pop":
			return await self._pop_message(parameters)
		elif operation == "stream_add":
			return await self._stream_add(parameters)
		elif operation == "stream_read":
			return await self._stream_read(parameters)
		elif operation == "stream_create_group":
			return await self._stream_create_group(parameters)
		elif operation == "pubsub_publish":
			return await self._pubsub_publish(parameters)
		elif operation == "pubsub_subscribe":
			return await self._pubsub_subscribe(parameters)
		else:
			raise ValueError(f"Unsupported Redis operation: {operation}")
	
	async def _push_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Push message to Redis list."""
		
		queue_name = params.get("queue")
		message = params.get("message")
		direction = params.get("direction", "left")  # left or right
		
		if not queue_name or message is None:
			raise ValueError("Queue name and message are required for push operation")
		
		# Serialize message
		if isinstance(message, (dict, list)):
			message_data = json.dumps(message)
		else:
			message_data = str(message)
		
		try:
			if direction == "left":
				result = await self.redis_client.lpush(queue_name, message_data)
			else:
				result = await self.redis_client.rpush(queue_name, message_data)
			
			return {
				"queue": queue_name,
				"message_size": len(message_data),
				"queue_length": result,
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to push message: {e}"))
			raise
	
	async def _pop_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Pop message from Redis list."""
		
		queue_name = params.get("queue")
		direction = params.get("direction", "left")  # left or right
		timeout = params.get("timeout", 1)  # blocking timeout
		
		if not queue_name:
			raise ValueError("Queue name is required for pop operation")
		
		try:
			if direction == "left":
				result = await self.redis_client.blpop(queue_name, timeout=timeout)
			else:
				result = await self.redis_client.brpop(queue_name, timeout=timeout)
			
			if result:
				queue, message_data = result
				
				# Try to deserialize JSON
				try:
					message = json.loads(message_data)
				except json.JSONDecodeError:
					message = message_data
				
				return {
					"queue": queue,
					"message": message,
					"success": True
				}
			else:
				return {"message": None, "success": True}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to pop message: {e}"))
			raise
	
	async def _stream_add(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Add message to Redis stream."""
		
		stream_name = params.get("stream")
		fields = params.get("fields", {})
		message_id = params.get("id", "*")
		maxlen = params.get("maxlen", self.config.stream_maxlen)
		
		if not stream_name or not fields:
			raise ValueError("Stream name and fields are required for stream add operation")
		
		try:
			result = await self.redis_client.xadd(
				stream_name,
				fields,
				id=message_id,
				maxlen=maxlen
			)
			
			return {
				"stream": stream_name,
				"message_id": result,
				"fields": fields,
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to add to stream: {e}"))
			raise
	
	async def _stream_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Read messages from Redis stream."""
		
		streams = params.get("streams", {})
		count = params.get("count", 10)
		block = params.get("block", 1000)
		group = params.get("group")
		consumer = params.get("consumer", self.config.consumer_name)
		
		if not streams:
			raise ValueError("Streams are required for stream read operation")
		
		try:
			if group:
				# Read as consumer group
				result = await self.redis_client.xreadgroup(
					group,
					consumer,
					streams,
					count=count,
					block=block
				)
			else:
				# Read directly from stream
				result = await self.redis_client.xread(
					streams,
					count=count,
					block=block
				)
			
			messages = []
			for stream_name, stream_messages in result:
				for message_id, fields in stream_messages:
					messages.append({
						"stream": stream_name,
						"id": message_id,
						"fields": fields
					})
			
			return {
				"messages": messages,
				"count": len(messages),
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to read from stream: {e}"))
			raise
	
	async def _stream_create_group(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Create Redis stream consumer group."""
		
		stream_name = params.get("stream")
		group_name = params.get("group", self.config.consumer_group)
		start_id = params.get("start_id", "$")
		mkstream = params.get("mkstream", True)
		
		if not stream_name:
			raise ValueError("Stream name is required for group creation")
		
		try:
			await self.redis_client.xgroup_create(
				stream_name,
				group_name,
				start_id,
				mkstream=mkstream
			)
			
			return {
				"stream": stream_name,
				"group": group_name,
				"start_id": start_id,
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to create stream group: {e}"))
			raise
	
	async def _pubsub_publish(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Publish message to Redis pub/sub channel."""
		
		channel = params.get("channel")
		message = params.get("message")
		
		if not channel or message is None:
			raise ValueError("Channel and message are required for pub/sub publish")
		
		# Serialize message
		if isinstance(message, (dict, list)):
			message_data = json.dumps(message)
		else:
			message_data = str(message)
		
		try:
			subscribers = await self.redis_client.publish(channel, message_data)
			
			return {
				"channel": channel,
				"message_size": len(message_data),
				"subscribers": subscribers,
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to publish message: {e}"))
			raise
	
	async def _pubsub_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Subscribe to Redis pub/sub channels."""
		
		channels = params.get("channels", [])
		patterns = params.get("patterns", [])
		
		if not channels and not patterns:
			raise ValueError("Channels or patterns are required for subscription")
		
		try:
			pubsub = self.redis_client.pubsub()
			
			if channels:
				await pubsub.subscribe(*channels)
			if patterns:
				await pubsub.psubscribe(*patterns)
			
			return {
				"subscribed_channels": channels,
				"subscribed_patterns": patterns,
				"pubsub_object": pubsub,  # This would need special handling in real implementation
				"success": True
			}
		
		except RedisError as e:
			logger.error(self._log_connector_info(f"Failed to subscribe: {e}"))
			raise
	
	async def _health_check(self) -> bool:
		"""Check Redis connectivity."""
		try:
			await self.redis_client.ping()
			return True
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False

# Export message queue connector classes
__all__ = [
	"BytewaxConnector",
	"BytewaxConfiguration",
	"RabbitMQConnector",
	"RabbitMQConfiguration",
	"RedisQueueConnector",
	"RedisQueueConfiguration"
]
