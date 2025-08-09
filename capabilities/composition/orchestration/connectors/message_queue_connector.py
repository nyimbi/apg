"""
APG Workflow Orchestration Message Queue Connectors

High-performance message queue connectors for Kafka, RabbitMQ, Redis, and other
messaging systems with producer/consumer support and reliable delivery.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone
import logging

# Kafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

# RabbitMQ
import aio_pika
from aio_pika.exceptions import AMQPException

# Redis
import redis.asyncio as redis
from redis.exceptions import RedisError

from pydantic import BaseModel, Field, ConfigDict, validator

from .base_connector import BaseConnector, ConnectorConfiguration

logger = logging.getLogger(__name__)

class KafkaConfiguration(ConnectorConfiguration):
	"""Kafka message queue configuration."""
	
	bootstrap_servers: List[str] = Field(..., description="Kafka bootstrap servers")
	security_protocol: str = Field(default="PLAINTEXT", regex="^(PLAINTEXT|SSL|SASL_PLAINTEXT|SASL_SSL)$")
	sasl_mechanism: Optional[str] = Field(default=None, regex="^(PLAIN|SCRAM-SHA-256|SCRAM-SHA-512|GSSAPI)$")
	sasl_username: Optional[str] = Field(default=None)
	sasl_password: Optional[str] = Field(default=None)
	ssl_cafile: Optional[str] = Field(default=None, description="SSL CA certificate file path")
	ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file path")
	ssl_keyfile: Optional[str] = Field(default=None, description="SSL key file path")
	client_id: str = Field(default="apg-workflow-orchestration")
	group_id: Optional[str] = Field(default=None, description="Consumer group ID")
	auto_offset_reset: str = Field(default="latest", regex="^(earliest|latest)$")
	enable_auto_commit: bool = Field(default=True)
	max_poll_records: int = Field(default=500, ge=1, le=10000)
	session_timeout_ms: int = Field(default=30000, ge=1000, le=300000)
	heartbeat_interval_ms: int = Field(default=3000, ge=1000, le=30000)
	compression_type: str = Field(default="none", regex="^(none|gzip|snappy|lz4|zstd)$")

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

class KafkaConnector(BaseConnector):
	"""High-performance Kafka message queue connector."""
	
	def __init__(self, config: KafkaConfiguration):
		super().__init__(config)
		self.config: KafkaConfiguration = config
		self.producer: Optional[AIOKafkaProducer] = None
		self.consumer: Optional[AIOKafkaConsumer] = None
		self.message_handlers: Dict[str, Callable] = {}
		self.consumer_task: Optional[asyncio.Task] = None
		self.is_consuming = False
	
	async def _connect(self) -> None:
		"""Initialize Kafka producer and consumer."""
		
		# Common connection parameters
		kafka_params = {
			"bootstrap_servers": self.config.bootstrap_servers,
			"client_id": self.config.client_id,
			"security_protocol": self.config.security_protocol
		}
		
		# Add SASL authentication if configured
		if self.config.sasl_mechanism:
			kafka_params.update({
				"sasl_mechanism": self.config.sasl_mechanism,
				"sasl_plain_username": self.config.sasl_username,
				"sasl_plain_password": self.config.sasl_password
			})
		
		# Add SSL configuration if needed
		if self.config.security_protocol in ["SSL", "SASL_SSL"]:
			ssl_params = {}
			if self.config.ssl_cafile:
				ssl_params["ssl_cafile"] = self.config.ssl_cafile
			if self.config.ssl_certfile:
				ssl_params["ssl_certfile"] = self.config.ssl_certfile
			if self.config.ssl_keyfile:
				ssl_params["ssl_keyfile"] = self.config.ssl_keyfile
			kafka_params.update(ssl_params)
		
		# Initialize producer
		self.producer = AIOKafkaProducer(
			compression_type=self.config.compression_type,
			**kafka_params
		)
		await self.producer.start()
		
		# Initialize consumer if group_id is provided
		if self.config.group_id:
			self.consumer = AIOKafkaConsumer(
				group_id=self.config.group_id,
				auto_offset_reset=self.config.auto_offset_reset,
				enable_auto_commit=self.config.enable_auto_commit,
				max_poll_records=self.config.max_poll_records,
				session_timeout_ms=self.config.session_timeout_ms,
				heartbeat_interval_ms=self.config.heartbeat_interval_ms,
				**kafka_params
			)
			await self.consumer.start()
		
		logger.info(self._log_connector_info("Kafka connector initialized"))
	
	async def _disconnect(self) -> None:
		"""Close Kafka connections."""
		
		# Stop consuming
		if self.consumer_task:
			self.consumer_task.cancel()
			await asyncio.gather(self.consumer_task, return_exceptions=True)
		
		# Close consumer
		if self.consumer:
			await self.consumer.stop()
			self.consumer = None
		
		# Close producer
		if self.producer:
			await self.producer.stop()
			self.producer = None
		
		logger.info(self._log_connector_info("Kafka connector disconnected"))
	
	async def _execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute Kafka operation."""
		
		if operation == "produce":
			return await self._produce_message(parameters)
		elif operation == "consume":
			return await self._consume_messages(parameters)
		elif operation == "subscribe":
			return await self._subscribe_topics(parameters)
		elif operation == "unsubscribe":
			return await self._unsubscribe_topics(parameters)
		else:
			raise ValueError(f"Unsupported Kafka operation: {operation}")
	
	async def _produce_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Produce message to Kafka topic."""
		
		topic = params.get("topic")
		message = params.get("message")
		key = params.get("key")
		headers = params.get("headers", {})
		partition = params.get("partition")
		
		if not topic or message is None:
			raise ValueError("Topic and message are required for Kafka produce operation")
		
		# Serialize message
		if isinstance(message, dict):
			message_bytes = json.dumps(message).encode("utf-8")
		elif isinstance(message, str):
			message_bytes = message.encode("utf-8")
		else:
			message_bytes = message
		
		# Serialize key if provided
		key_bytes = None
		if key:
			if isinstance(key, str):
				key_bytes = key.encode("utf-8")
			else:
				key_bytes = key
		
		# Convert headers to bytes
		headers_bytes = {}
		for k, v in headers.items():
			if isinstance(v, str):
				headers_bytes[k] = v.encode("utf-8")
			else:
				headers_bytes[k] = v
		
		try:
			# Send message
			record_metadata = await self.producer.send_and_wait(
				topic=topic,
				value=message_bytes,
				key=key_bytes,
				headers=list(headers_bytes.items()) if headers_bytes else None,
				partition=partition
			)
			
			return {
				"topic": record_metadata.topic,
				"partition": record_metadata.partition,
				"offset": record_metadata.offset,
				"timestamp": record_metadata.timestamp,
				"success": True
			}
		
		except KafkaError as e:
			logger.error(self._log_connector_info(f"Failed to produce message: {e}"))
			raise
	
	async def _consume_messages(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Consume messages from Kafka topics."""
		
		timeout_ms = params.get("timeout_ms", 1000)
		max_records = params.get("max_records", 100)
		
		if not self.consumer:
			raise ValueError("Consumer not initialized. Provide group_id in configuration.")
		
		messages = []
		try:
			# Poll for messages
			message_batch = await self.consumer.getmany(timeout_ms=timeout_ms, max_records=max_records)
			
			for topic_partition, messages_list in message_batch.items():
				for message in messages_list:
					# Deserialize message
					try:
						value = json.loads(message.value.decode("utf-8"))
					except (json.JSONDecodeError, UnicodeDecodeError):
						value = message.value.decode("utf-8", errors="ignore")
					
					# Deserialize key
					key = None
					if message.key:
						try:
							key = message.key.decode("utf-8")
						except UnicodeDecodeError:
							key = str(message.key)
					
					# Deserialize headers
					headers = {}
					if message.headers:
						for header_key, header_value in message.headers:
							try:
								headers[header_key] = header_value.decode("utf-8")
							except UnicodeDecodeError:
								headers[header_key] = str(header_value)
					
					messages.append({
						"topic": message.topic,
						"partition": message.partition,
						"offset": message.offset,
						"timestamp": message.timestamp,
						"key": key,
						"value": value,
						"headers": headers
					})
			
			return {
				"messages": messages,
				"count": len(messages),
				"success": True
			}
		
		except KafkaError as e:
			logger.error(self._log_connector_info(f"Failed to consume messages: {e}"))
			raise
	
	async def _subscribe_topics(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Subscribe to Kafka topics."""
		
		topics = params.get("topics", [])
		if not topics:
			raise ValueError("Topics list is required for subscribe operation")
		
		if not self.consumer:
			raise ValueError("Consumer not initialized. Provide group_id in configuration.")
		
		try:
			self.consumer.subscribe(topics)
			
			# Start consuming task if not already running
			if not self.is_consuming:
				self.consumer_task = asyncio.create_task(self._consume_loop())
				self.is_consuming = True
			
			return {
				"subscribed_topics": topics,
				"success": True
			}
		
		except KafkaError as e:
			logger.error(self._log_connector_info(f"Failed to subscribe to topics: {e}"))
			raise
	
	async def _unsubscribe_topics(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Unsubscribe from Kafka topics."""
		
		if not self.consumer:
			raise ValueError("Consumer not initialized.")
		
		try:
			self.consumer.unsubscribe()
			
			# Stop consuming task
			if self.consumer_task:
				self.consumer_task.cancel()
				self.is_consuming = False
			
			return {"success": True}
		
		except KafkaError as e:
			logger.error(self._log_connector_info(f"Failed to unsubscribe: {e}"))
			raise
	
	async def _consume_loop(self) -> None:
		"""Background message consumption loop."""
		while self.is_consuming:
			try:
				# Consume messages and call handlers
				result = await self._consume_messages({"timeout_ms": 1000, "max_records": 100})
				
				for message in result.get("messages", []):
					topic = message["topic"]
					if topic in self.message_handlers:
						try:
							await self.message_handlers[topic](message)
						except Exception as e:
							logger.error(self._log_connector_info(f"Message handler error for topic {topic}: {e}"))
			
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(self._log_connector_info(f"Consume loop error: {e}"))
				await asyncio.sleep(1)
	
	async def _health_check(self) -> bool:
		"""Check Kafka connectivity."""
		try:
			if self.producer:
				# Try to get metadata
				metadata = await self.producer.client.bootstrap()
				return len(metadata.brokers) > 0
			return False
		except Exception as e:
			logger.warning(self._log_connector_info(f"Health check failed: {e}"))
			return False
	
	def add_message_handler(self, topic: str, handler: Callable) -> None:
		"""Add message handler for specific topic."""
		self.message_handlers[topic] = handler

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
	"KafkaConnector",
	"KafkaConfiguration",
	"RabbitMQConnector",
	"RabbitMQConfiguration",
	"RedisQueueConnector",
	"RedisQueueConfiguration"
]