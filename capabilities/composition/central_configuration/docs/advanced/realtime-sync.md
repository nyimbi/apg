# Real-Time Synchronization Guide

## Overview

The APG Central Configuration capability provides comprehensive real-time synchronization across distributed systems using WebSocket, Kafka, MQTT, and Redis technologies. This enables instant configuration propagation, conflict resolution, and multi-region consistency.

## Architecture

### Synchronization Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   WebSocket     │    │      Kafka       │    │      MQTT       │
│   Clients       │    │   Event Stream   │    │  Lightweight    │
│                 │    │                  │    │   Clients       │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────┬───────────┼───────────┬───────────┘
                     │           │           │
                ┌────┴────────────┴───────────┴────┐
                │      Realtime Sync Manager      │
                │                                  │
                │  • Event Broadcasting           │
                │  • Conflict Resolution          │
                │  • Operational Transforms       │
                │  • Lock Management              │
                └─────────────┬────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │      Redis        │
                    │   Pub/Sub & Cache │
                    └───────────────────┘
```

### Event Flow

```
Config Change → Validation → Encryption → Broadcast → Transform → Apply
     ↑                                        │
     └── Conflict Detection ←── Merge ←───────┘
```

## Configuration

### Basic Setup

```python
from capabilities.composition.central_configuration.realtime_sync import (
    RealtimeSyncManager, create_realtime_sync_manager
)
import redis.asyncio as redis

# Initialize Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Create sync manager
sync_manager = await create_realtime_sync_manager(
    redis_client=redis_client,
    kafka_bootstrap_servers=["localhost:9092"],
    mqtt_broker_host="localhost",
    node_id="node-1"
)
```

### Advanced Configuration

```python
# Comprehensive sync configuration
sync_config = {
    "redis": {
        "host": "redis-cluster.internal",
        "port": 6379,
        "password": "redis-password",
        "ssl": True,
        "pool_size": 20
    },
    "kafka": {
        "bootstrap_servers": [
            "kafka-1.internal:9092",
            "kafka-2.internal:9092", 
            "kafka-3.internal:9092"
        ],
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
        "sasl_username": "apg-config",
        "sasl_password": "kafka-password",
        "compression_type": "gzip",
        "batch_size": 1000,
        "linger_ms": 10
    },
    "mqtt": {
        "broker_host": "mqtt.internal",
        "broker_port": 8883,
        "username": "apg-mqtt",
        "password": "mqtt-password",
        "tls_enabled": True,
        "keepalive": 60,
        "qos": 1,
        "retain": True
    },
    "websocket": {
        "host": "0.0.0.0",
        "port": 8080,
        "ssl_context": {
            "certfile": "/etc/ssl/certs/websocket.crt",
            "keyfile": "/etc/ssl/private/websocket.key"
        },
        "ping_interval": 30,
        "ping_timeout": 10,
        "max_connections": 1000
    },
    "sync_settings": {
        "heartbeat_interval": 30,
        "conflict_resolution_interval": 10,
        "cleanup_interval": 300,
        "max_event_age_hours": 24,
        "batch_size": 100,
        "retry_attempts": 3,
        "retry_backoff": 2
    }
}

sync_manager = RealtimeSyncManager(
    redis_client=redis_client,
    **sync_config
)
await sync_manager.initialize()
```

## WebSocket Real-Time Updates

### Server-Side WebSocket Handler

```python
import websockets
import json
from websockets.server import serve

class WebSocketHandler:
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.connections = {}
    
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        connection_id = f"ws_{uuid7str()[:8]}"
        
        try:
            # Register connection
            await self.sync_manager.add_websocket_connection(
                connection_id,
                websocket,
                subscription_patterns=["*"]  # Subscribe to all by default
            )
            
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            # Handle incoming messages
            async for message in websocket:
                await self.handle_message(connection_id, json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Clean up connection
            await self.sync_manager.remove_websocket_connection(connection_id)
    
    async def handle_message(self, connection_id, message):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Update subscription patterns
            patterns = message.get("patterns", [])
            await self.sync_manager.update_subscription_patterns(
                connection_id, 
                patterns
            )
        
        elif message_type == "config_change":
            # Handle real-time configuration change
            await self.handle_config_change(connection_id, message)
        
        elif message_type == "ping":
            # Respond to ping
            await self.send_message(connection_id, {"type": "pong"})
    
    async def handle_config_change(self, connection_id, message):
        """Handle configuration change from WebSocket client"""
        try:
            config_key = message["config_key"]
            new_value = message["new_value"]
            user_id = message.get("user_id", "anonymous")
            
            # Attempt to acquire lock
            lock_acquired = await self.sync_manager.acquire_config_lock(
                config_key,
                user_id,
                timeout=60
            )
            
            if lock_acquired:
                # Process the change
                success = await self.process_config_change(
                    config_key, 
                    new_value, 
                    user_id
                )
                
                # Release lock
                await self.sync_manager.release_config_lock(
                    config_key, 
                    user_id
                )
                
                # Send response
                await self.send_message(connection_id, {
                    "type": "config_change_response",
                    "success": success,
                    "config_key": config_key
                })
            else:
                # Lock acquisition failed
                await self.send_message(connection_id, {
                    "type": "config_change_response",
                    "success": False,
                    "error": "Configuration is locked by another user",
                    "config_key": config_key
                })
                
        except Exception as e:
            await self.send_message(connection_id, {
                "type": "error",
                "message": str(e)
            })

# Start WebSocket server
handler = WebSocketHandler(sync_manager)
start_server = serve(handler.handle_connection, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

### Client-Side JavaScript

```javascript
class ConfigWebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.connectionId = null;
        this.subscriptions = new Set();
        this.eventHandlers = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    async connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected');
                this.handleDisconnection();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this.handleReconnection();
        }
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'connection_established':
                this.connectionId = message.connection_id;
                this.onConnectionEstablished();
                break;
                
            case 'config_sync':
                this.handleConfigSync(message);
                break;
                
            case 'config_locked':
                this.handleConfigLocked(message);
                break;
                
            case 'config_unlocked':
                this.handleConfigUnlocked(message);
                break;
                
            case 'error':
                this.handleError(message);
                break;
        }
    }
    
    handleConfigSync(message) {
        const { config_key, new_value, old_value, event_type } = message;
        
        // Update local configuration
        this.updateLocalConfig(config_key, new_value);
        
        // Trigger event handlers
        const handlers = this.eventHandlers.get('config_changed') || [];
        handlers.forEach(handler => {
            handler({
                key: config_key,
                newValue: new_value,
                oldValue: old_value,
                eventType: event_type
            });
        });
        
        // Update UI if needed
        this.updateUI(config_key, new_value);
    }
    
    subscribe(patterns) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            patterns.forEach(pattern => this.subscriptions.add(pattern));
            
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                patterns: Array.from(this.subscriptions)
            }));
        }
    }
    
    unsubscribe(patterns) {
        patterns.forEach(pattern => this.subscriptions.delete(pattern));
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'subscribe',
                patterns: Array.from(this.subscriptions)
            }));
        }
    }
    
    setConfig(key, value, userId = 'anonymous') {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'config_change',
                config_key: key,
                new_value: value,
                user_id: userId,
                timestamp: new Date().toISOString()
            }));
        }
    }
    
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    off(event, handler) {
        const handlers = this.eventHandlers.get(event) || [];
        const index = handlers.indexOf(handler);
        if (index !== -1) {
            handlers.splice(index, 1);
        }
    }
    
    handleDisconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
                this.reconnectAttempts++;
                this.connect();
            }, Math.pow(2, this.reconnectAttempts) * 1000);
        }
    }
    
    updateLocalConfig(key, value) {
        // Update local configuration store
        if (window.appConfig) {
            const keys = key.split('.');
            let current = window.appConfig;
            
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]]) {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            
            current[keys[keys.length - 1]] = value;
        }
    }
    
    updateUI(key, value) {
        // Update UI elements based on configuration changes
        const elements = document.querySelectorAll(`[data-config="${key}"]`);
        elements.forEach(element => {
            if (element.tagName === 'INPUT') {
                element.value = value;
            } else {
                element.textContent = value;
            }
        });
    }
}

// Usage example
const client = new ConfigWebSocketClient('ws://localhost:8765');

// Set up event handlers
client.on('config_changed', (event) => {
    console.log(`Configuration changed: ${event.key} = ${event.newValue}`);
    
    // Handle specific configuration changes
    if (event.key.startsWith('app.database.')) {
        console.log('Database configuration changed, considering reconnection...');
    } else if (event.key.startsWith('app.features.')) {
        console.log('Feature flag changed, updating UI...');
        updateFeatureFlags();
    }
});

// Connect and subscribe
await client.connect();
client.subscribe([
    'app.database.*',
    'app.features.*',
    'app.ui.*'
]);
```

## Kafka Event Streaming

### Producer Configuration

```python
from aiokafka import AIOKafkaProducer
import json

class ConfigEventProducer:
    def __init__(self, bootstrap_servers, security_config=None):
        self.bootstrap_servers = bootstrap_servers
        self.security_config = security_config or {}
        self.producer = None
    
    async def initialize(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: x.encode('utf-8') if x else None,
            compression_type="gzip",
            max_request_size=10485760,  # 10MB
            request_timeout_ms=30000,
            retry_backoff_ms=1000,
            retries=5,
            acks='all',  # Wait for all replicas
            enable_idempotence=True,  # Prevent duplicates
            **self.security_config
        )
        await self.producer.start()
    
    async def send_config_event(self, event_data, config_key=None):
        """Send configuration change event to Kafka"""
        topic = self.get_topic_for_config(config_key)
        partition_key = self.get_partition_key(config_key, event_data)
        
        try:
            record_metadata = await self.producer.send(
                topic,
                value=event_data,
                key=partition_key,
                headers={
                    'event_type': event_data.get('event_type', '').encode(),
                    'tenant_id': event_data.get('tenant_id', '').encode(),
                    'timestamp': str(int(time.time())).encode()
                }
            )
            
            logger.info(f"Event sent to Kafka: topic={topic}, partition={record_metadata.partition}, offset={record_metadata.offset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event to Kafka: {e}")
            return False
    
    def get_topic_for_config(self, config_key):
        """Determine Kafka topic based on configuration key"""
        if not config_key:
            return "apg-config-general"
        elif config_key.startswith("app.database."):
            return "apg-config-database"
        elif config_key.startswith("app.security."):
            return "apg-config-security"
        elif config_key.startswith("app.features."):
            return "apg-config-features"
        else:
            return "apg-config-general"
    
    def get_partition_key(self, config_key, event_data):
        """Generate partition key for even distribution"""
        tenant_id = event_data.get('tenant_id', 'default')
        return f"{tenant_id}:{config_key or 'system'}"

    async def close(self):
        if self.producer:
            await self.producer.stop()
```

### Consumer Configuration

```python
from aiokafka import AIOKafkaConsumer
import json

class ConfigEventConsumer:
    def __init__(self, bootstrap_servers, group_id, security_config=None):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.security_config = security_config or {}
        self.consumer = None
        self.event_handlers = {}
    
    async def initialize(self):
        self.consumer = AIOKafkaConsumer(
            "apg-config-general",
            "apg-config-database", 
            "apg-config-security",
            "apg-config-features",
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            **self.security_config
        )
        await self.consumer.start()
    
    async def consume_events(self):
        """Consume configuration events from Kafka"""
        try:
            async for message in self.consumer:
                try:
                    await self.process_event(message)
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    # Continue processing other messages
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            raise
    
    async def process_event(self, message):
        """Process individual configuration event"""
        event_data = message.value
        headers = {h[0]: h[1].decode() for h in message.headers or []}
        
        event_type = headers.get('event_type', event_data.get('event_type'))
        tenant_id = headers.get('tenant_id', event_data.get('tenant_id'))
        
        # Skip events from our own node to prevent loops
        source_node = event_data.get('source_node')
        if source_node == self.node_id:
            return
        
        # Route to appropriate handler
        handler = self.event_handlers.get(event_type)
        if handler:
            await handler(event_data, headers)
        else:
            # Default handler
            await self.handle_generic_config_event(event_data, headers)
    
    def register_handler(self, event_type, handler):
        """Register event handler for specific event type"""
        self.event_handlers[event_type] = handler
    
    async def handle_generic_config_event(self, event_data, headers):
        """Generic handler for configuration events"""
        config_key = event_data.get('config_key')
        new_value = event_data.get('new_value')
        tenant_id = event_data.get('tenant_id')
        
        logger.info(f"Received config event: {config_key} = {new_value} (tenant: {tenant_id})")
        
        # Apply configuration change locally
        await self.apply_config_change(config_key, new_value, tenant_id)
    
    async def apply_config_change(self, config_key, new_value, tenant_id):
        """Apply configuration change to local system"""
        # This would integrate with your local configuration management
        # For example, updating application settings, reloading configs, etc.
        pass

    async def close(self):
        if self.consumer:
            await self.consumer.stop()
```

## MQTT Lightweight Messaging

### MQTT Publisher

```python
import asyncio_mqtt
import json

class MQTTConfigPublisher:
    def __init__(self, broker_host, broker_port=1883, client_id=None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"apg_config_pub_{uuid7str()[:8]}"
        self.client = None
    
    async def initialize(self):
        self.client = asyncio_mqtt.Client(
            hostname=self.broker_host,
            port=self.broker_port,
            client_id=self.client_id,
            keepalive=60,
            will=asyncio_mqtt.Will(
                topic=f"apg/config/status/{self.client_id}",
                payload="offline",
                qos=1,
                retain=True
            )
        )
    
    async def publish_config_change(self, config_key, new_value, tenant_id=None):
        """Publish configuration change to MQTT"""
        topic = self.build_topic(config_key, tenant_id)
        payload = {
            "config_key": config_key,
            "new_value": new_value,
            "timestamp": datetime.utcnow().isoformat(),
            "publisher_id": self.client_id
        }
        
        async with self.client:
            await self.client.publish(
                topic,
                payload=json.dumps(payload),
                qos=1,
                retain=False
            )
            
            # Publish status
            await self.client.publish(
                f"apg/config/status/{self.client_id}",
                payload="online",
                qos=1,
                retain=True
            )
    
    def build_topic(self, config_key, tenant_id):
        """Build MQTT topic from configuration key"""
        tenant_part = tenant_id or "global"
        config_parts = config_key.split('.')
        
        # Create hierarchical topic structure
        return f"apg/config/{tenant_part}/{'/'.join(config_parts)}"

# Usage example
publisher = MQTTConfigPublisher("mqtt.internal", 1883)
await publisher.initialize()
await publisher.publish_config_change(
    "app.database.host", 
    "new-db-server.internal",
    "company-tenant"
)
```

### MQTT Subscriber

```python
class MQTTConfigSubscriber:
    def __init__(self, broker_host, broker_port=1883, client_id=None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id or f"apg_config_sub_{uuid7str()[:8]}"
        self.client = None
        self.subscriptions = []
        self.message_handlers = {}
    
    async def initialize(self):
        self.client = asyncio_mqtt.Client(
            hostname=self.broker_host,
            port=self.broker_port,
            client_id=self.client_id,
            keepalive=60
        )
    
    async def subscribe_to_configs(self, patterns):
        """Subscribe to configuration patterns"""
        async with self.client:
            # Subscribe to patterns
            for pattern in patterns:
                topic = self.pattern_to_mqtt_topic(pattern)
                await self.client.subscribe(topic)
                self.subscriptions.append(topic)
            
            # Listen for messages
            async for message in self.client.messages:
                await self.handle_message(message)
    
    def pattern_to_mqtt_topic(self, pattern):
        """Convert glob pattern to MQTT topic pattern"""
        # Convert app.database.* to apg/config/+/app/database/+
        parts = pattern.replace('*', '+').split('.')
        return f"apg/config/+/{'/'.join(parts)}"
    
    async def handle_message(self, message):
        """Handle incoming MQTT message"""
        try:
            payload = json.loads(message.payload.decode())
            topic_parts = str(message.topic).split('/')
            
            # Extract tenant and config key from topic
            tenant_id = topic_parts[2] if len(topic_parts) > 2 else "global"
            config_key = '.'.join(topic_parts[3:]) if len(topic_parts) > 3 else ""
            
            await self.process_config_update(
                config_key=config_key,
                new_value=payload.get('new_value'),
                tenant_id=tenant_id,
                metadata=payload
            )
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    async def process_config_update(self, config_key, new_value, tenant_id, metadata):
        """Process configuration update from MQTT"""
        logger.info(f"MQTT config update: {config_key} = {new_value} (tenant: {tenant_id})")
        
        # Check if we have a specific handler
        handler = self.message_handlers.get(config_key)
        if handler:
            await handler(config_key, new_value, tenant_id, metadata)
        else:
            # Default processing
            await self.default_config_handler(config_key, new_value, tenant_id)
    
    def register_handler(self, config_key, handler):
        """Register handler for specific configuration key"""
        self.message_handlers[config_key] = handler
    
    async def default_config_handler(self, config_key, new_value, tenant_id):
        """Default handler for configuration updates"""
        # Apply the configuration change locally
        # This would integrate with your application's config system
        pass

# Usage example
subscriber = MQTTConfigSubscriber("mqtt.internal", 1883)
await subscriber.initialize()

# Register specific handlers
subscriber.register_handler("app.database.host", handle_database_config_change)
subscriber.register_handler("app.features.*", handle_feature_flag_change)

# Start listening
await subscriber.subscribe_to_configs([
    "app.database.*",
    "app.features.*",
    "app.security.*"
])
```

## Conflict Resolution

### Operational Transforms

```python
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from datetime import datetime

@dataclass
class Operation:
    """Represents a single operation in operational transforms"""
    op_type: str  # 'insert', 'delete', 'retain', 'replace'
    position: int
    content: Any = None
    length: int = 0
    timestamp: datetime = None
    author: str = None

class OperationalTransform:
    """Operational Transform implementation for conflict resolution"""
    
    def __init__(self):
        self.operations = []
    
    def transform_against(self, other_op: 'OperationalTransform') -> 'OperationalTransform':
        """Transform this operation against another operation"""
        transformed = OperationalTransform()
        
        # Implement transformation logic based on operation types
        for op1 in self.operations:
            for op2 in other_op.operations:
                transformed_op = self._transform_single_operations(op1, op2)
                if transformed_op:
                    transformed.operations.append(transformed_op)
        
        return transformed
    
    def _transform_single_operations(self, op1: Operation, op2: Operation) -> Optional[Operation]:
        """Transform two individual operations"""
        # Insert vs Insert
        if op1.op_type == 'insert' and op2.op_type == 'insert':
            if op1.position <= op2.position:
                return Operation(
                    op_type=op2.op_type,
                    position=op2.position + len(op1.content),
                    content=op2.content,
                    author=op2.author,
                    timestamp=op2.timestamp
                )
            else:
                return op2
        
        # Insert vs Delete
        elif op1.op_type == 'insert' and op2.op_type == 'delete':
            if op1.position <= op2.position:
                return Operation(
                    op_type=op2.op_type,
                    position=op2.position + len(op1.content),
                    length=op2.length,
                    author=op2.author,
                    timestamp=op2.timestamp
                )
            else:
                return op2
        
        # Delete vs Insert
        elif op1.op_type == 'delete' and op2.op_type == 'insert':
            if op1.position < op2.position:
                return Operation(
                    op_type=op2.op_type,
                    position=max(op1.position, op2.position - op1.length),
                    content=op2.content,
                    author=op2.author,
                    timestamp=op2.timestamp
                )
            else:
                return op2
        
        # Delete vs Delete
        elif op1.op_type == 'delete' and op2.op_type == 'delete':
            # Complex case - overlapping deletes
            if op1.position + op1.length <= op2.position:
                # Non-overlapping, op2 comes after op1
                return Operation(
                    op_type=op2.op_type,
                    position=op2.position - op1.length,
                    length=op2.length,
                    author=op2.author,
                    timestamp=op2.timestamp
                )
            elif op2.position + op2.length <= op1.position:
                # Non-overlapping, op1 comes after op2
                return op2
            else:
                # Overlapping deletes - merge them
                new_position = min(op1.position, op2.position)
                new_length = max(
                    op1.position + op1.length,
                    op2.position + op2.length
                ) - new_position
                
                return Operation(
                    op_type='delete',
                    position=new_position,
                    length=new_length,
                    author=f"{op1.author}+{op2.author}",
                    timestamp=max(op1.timestamp, op2.timestamp)
                )
        
        return op2  # Default case

class ConfigurationConflictResolver:
    """Resolves configuration conflicts using various strategies"""
    
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.conflict_strategies = {
            ConflictResolutionStrategy.LAST_WRITE_WINS: self._last_write_wins,
            ConflictResolutionStrategy.FIRST_WRITE_WINS: self._first_write_wins,
            ConflictResolutionStrategy.OPERATIONAL_TRANSFORM: self._operational_transform,
            ConflictResolutionStrategy.MERGE_STRATEGIES: self._merge_strategies,
            ConflictResolutionStrategy.MANUAL_RESOLUTION: self._manual_resolution
        }
    
    async def resolve_conflict(
        self, 
        config_key: str, 
        competing_changes: List[Dict[str, Any]],
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
    ) -> Dict[str, Any]:
        """Resolve configuration conflict using specified strategy"""
        
        resolver = self.conflict_strategies.get(strategy)
        if not resolver:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")
        
        resolution_result = await resolver(config_key, competing_changes)
        
        # Log conflict resolution
        await self._log_conflict_resolution(
            config_key, 
            competing_changes, 
            resolution_result,
            strategy
        )
        
        return resolution_result
    
    async def _last_write_wins(self, config_key: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflict using last-write-wins strategy"""
        latest_change = max(
            changes,
            key=lambda x: datetime.fromisoformat(x.get('timestamp', '1970-01-01T00:00:00Z'))
        )
        
        return {
            'resolved_value': latest_change.get('new_value'),
            'resolution_method': 'last_write_wins',
            'winning_change': latest_change,
            'discarded_changes': [c for c in changes if c != latest_change]
        }
    
    async def _first_write_wins(self, config_key: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflict using first-write-wins strategy"""
        earliest_change = min(
            changes,
            key=lambda x: datetime.fromisoformat(x.get('timestamp', '9999-12-31T23:59:59Z'))
        )
        
        return {
            'resolved_value': earliest_change.get('new_value'),
            'resolution_method': 'first_write_wins',
            'winning_change': earliest_change,
            'discarded_changes': [c for c in changes if c != earliest_change]
        }
    
    async def _operational_transform(self, config_key: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflict using operational transforms"""
        if len(changes) < 2:
            return changes[0] if changes else None
        
        # Sort changes by timestamp
        sorted_changes = sorted(
            changes,
            key=lambda x: datetime.fromisoformat(x.get('timestamp', '1970-01-01T00:00:00Z'))
        )
        
        base_value = sorted_changes[0].get('old_value', {})
        operations = []
        
        # Convert each change to an operation
        for change in sorted_changes:
            op = self._change_to_operation(change)
            operations.append(op)
        
        # Apply operational transforms
        transformed_ops = []
        for i, op in enumerate(operations):
            current_op = op
            
            # Transform against all previous operations
            for prev_op in transformed_ops:
                ot = OperationalTransform()
                ot.operations = [current_op]
                
                other_ot = OperationalTransform()
                other_ot.operations = [prev_op]
                
                transformed_ot = ot.transform_against(other_ot)
                if transformed_ot.operations:
                    current_op = transformed_ot.operations[0]
            
            transformed_ops.append(current_op)
        
        # Apply all transformed operations to base value
        result_value = base_value
        for op in transformed_ops:
            result_value = self._apply_operation(result_value, op)
        
        return {
            'resolved_value': result_value,
            'resolution_method': 'operational_transform',
            'operations_applied': len(transformed_ops),
            'base_value': base_value
        }
    
    async def _merge_strategies(self, config_key: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflict using intelligent merge strategies"""
        if not changes:
            return None
        
        new_values = [change.get('new_value') for change in changes]
        
        # Strategy 1: Dictionary merge
        if all(isinstance(v, dict) for v in new_values):
            merged_dict = {}
            for value in new_values:
                merged_dict.update(value)
            
            return {
                'resolved_value': merged_dict,
                'resolution_method': 'dictionary_merge',
                'merged_sources': len(new_values)
            }
        
        # Strategy 2: List concatenation and deduplication
        elif all(isinstance(v, list) for v in new_values):
            merged_list = []
            for value in new_values:
                merged_list.extend(value)
            
            # Remove duplicates while preserving order
            seen = set()
            deduplicated = []
            for item in merged_list:
                if item not in seen:
                    seen.add(item)
                    deduplicated.append(item)
            
            return {
                'resolved_value': deduplicated,
                'resolution_method': 'list_merge',
                'original_length': len(merged_list),
                'deduplicated_length': len(deduplicated)
            }
        
        # Strategy 3: Numeric aggregation
        elif all(isinstance(v, (int, float)) for v in new_values):
            # For numeric values, use average as compromise
            average_value = sum(new_values) / len(new_values)
            
            return {
                'resolved_value': average_value,
                'resolution_method': 'numeric_average',
                'original_values': new_values,
                'min_value': min(new_values),
                'max_value': max(new_values)
            }
        
        # Strategy 4: Fall back to last write wins for incompatible types
        else:
            return await self._last_write_wins(config_key, changes)
    
    async def _manual_resolution(self, config_key: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mark conflict for manual resolution"""
        # Store conflict for manual review
        conflict_id = uuid7str()
        
        await self.sync_manager.redis_client.set(
            f"conflict:{conflict_id}",
            json.dumps({
                'conflict_id': conflict_id,
                'config_key': config_key,
                'competing_changes': changes,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'pending_manual_resolution',
                'assigned_to': None
            }),
            ex=86400 * 7  # Expire in 7 days
        )
        
        # Notify administrators
        await self._notify_manual_resolution_needed(conflict_id, config_key, changes)
        
        return {
            'resolved_value': None,
            'resolution_method': 'manual_resolution_pending',
            'conflict_id': conflict_id,
            'requires_human_intervention': True
        }
    
    def _change_to_operation(self, change: Dict[str, Any]) -> Operation:
        """Convert a configuration change to an operational transform operation"""
        # This is a simplified conversion - in practice, you'd need more sophisticated
        # analysis of the actual changes
        return Operation(
            op_type='replace',
            position=0,
            content=change.get('new_value'),
            author=change.get('user_id', 'unknown'),
            timestamp=datetime.fromisoformat(change.get('timestamp', datetime.utcnow().isoformat()))
        )
    
    def _apply_operation(self, base_value: Any, operation: Operation) -> Any:
        """Apply an operation to a base value"""
        if operation.op_type == 'replace':
            return operation.content
        elif operation.op_type == 'insert':
            if isinstance(base_value, list):
                result = base_value.copy()
                result.insert(operation.position, operation.content)
                return result
            elif isinstance(base_value, str):
                return base_value[:operation.position] + operation.content + base_value[operation.position:]
        elif operation.op_type == 'delete':
            if isinstance(base_value, list):
                result = base_value.copy()
                del result[operation.position:operation.position + operation.length]
                return result
            elif isinstance(base_value, str):
                return base_value[:operation.position] + base_value[operation.position + operation.length:]
        
        return base_value
    
    async def _log_conflict_resolution(
        self, 
        config_key: str, 
        changes: List[Dict[str, Any]], 
        resolution: Dict[str, Any],
        strategy: ConflictResolutionStrategy
    ):
        """Log conflict resolution for audit purposes"""
        log_entry = {
            'event_type': 'conflict_resolved',
            'config_key': config_key,
            'resolution_strategy': strategy.value,
            'competing_changes_count': len(changes),
            'resolution_method': resolution.get('resolution_method'),
            'resolved_value': resolution.get('resolved_value'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log to audit system
        logger.info(f"Conflict resolved for {config_key}: {resolution.get('resolution_method')}")
    
    async def _notify_manual_resolution_needed(
        self, 
        conflict_id: str, 
        config_key: str, 
        changes: List[Dict[str, Any]]
    ):
        """Notify administrators that manual conflict resolution is needed"""
        # This would integrate with your notification system
        # (email, Slack, PagerDuty, etc.)
        pass
```

## Configuration Locking

### Distributed Locking

```python
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

class DistributedConfigLock:
    """Distributed locking for configuration changes"""
    
    def __init__(self, redis_client, default_timeout=300):
        self.redis = redis_client
        self.default_timeout = default_timeout
        self.lock_prefix = "config_lock:"
        self.heartbeat_interval = 30
        self.heartbeat_tasks = {}
    
    async def acquire_lock(
        self, 
        config_key: str, 
        user_id: str, 
        tenant_id: Optional[str] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Acquire distributed lock for configuration key"""
        lock_key = self._build_lock_key(config_key, tenant_id)
        lock_timeout = timeout or self.default_timeout
        
        lock_data = {
            'config_key': config_key,
            'user_id': user_id,
            'tenant_id': tenant_id,
            'acquired_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(seconds=lock_timeout)).isoformat(),
            'heartbeat_interval': self.heartbeat_interval,
            'metadata': metadata or {}
        }
        
        # Try to acquire lock using Redis SET with NX (not exists) and EX (expiry)
        acquired = await self.redis.set(
            lock_key,
            json.dumps(lock_data),
            nx=True,  # Only set if key doesn't exist
            ex=lock_timeout  # Expire after timeout seconds
        )
        
        if acquired:
            # Start heartbeat to keep lock alive
            await self._start_heartbeat(lock_key, user_id)
            logger.info(f"Lock acquired for {config_key} by {user_id}")
            return True
        else:
            # Check if we already own this lock
            existing_lock = await self.redis.get(lock_key)
            if existing_lock:
                lock_info = json.loads(existing_lock)
                if lock_info.get('user_id') == user_id:
                    # Extend existing lock
                    await self._extend_lock(lock_key, lock_timeout)
                    return True
            
            logger.warning(f"Failed to acquire lock for {config_key} by {user_id}")
            return False
    
    async def release_lock(
        self, 
        config_key: str, 
        user_id: str, 
        tenant_id: Optional[str] = None
    ) -> bool:
        """Release distributed lock"""
        lock_key = self._build_lock_key(config_key, tenant_id)
        
        # Check if we own the lock
        existing_lock = await self.redis.get(lock_key)
        if not existing_lock:
            return False  # Lock doesn't exist
        
        lock_info = json.loads(existing_lock)
        if lock_info.get('user_id') != user_id:
            return False  # We don't own this lock
        
        # Release the lock
        deleted = await self.redis.delete(lock_key)
        
        if deleted:
            # Stop heartbeat
            await self._stop_heartbeat(lock_key)
            logger.info(f"Lock released for {config_key} by {user_id}")
            return True
        
        return False
    
    async def get_lock_info(
        self, 
        config_key: str, 
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about current lock"""
        lock_key = self._build_lock_key(config_key, tenant_id)
        lock_data = await self.redis.get(lock_key)
        
        if lock_data:
            return json.loads(lock_data)
        return None
    
    async def list_active_locks(
        self, 
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all active locks for a tenant"""
        pattern = self._build_lock_key("*", tenant_id)
        lock_keys = await self.redis.keys(pattern)
        
        locks = []
        for key_bytes in lock_keys:
            key = key_bytes.decode('utf-8')
            lock_data = await self.redis.get(key)
            if lock_data:
                lock_info = json.loads(lock_data)
                lock_info['lock_key'] = key
                locks.append(lock_info)
        
        return locks
    
    async def force_release_lock(
        self, 
        config_key: str, 
        admin_user_id: str,
        tenant_id: Optional[str] = None,
        reason: str = "Administrative override"
    ) -> bool:
        """Force release a lock (admin operation)"""
        lock_key = self._build_lock_key(config_key, tenant_id)
        
        # Get current lock info for audit
        existing_lock = await self.redis.get(lock_key)
        lock_info = json.loads(existing_lock) if existing_lock else {}
        
        # Force delete the lock
        deleted = await self.redis.delete(lock_key)
        
        if deleted:
            # Stop heartbeat
            await self._stop_heartbeat(lock_key)
            
            # Log administrative override
            logger.warning(
                f"Lock forcibly released for {config_key} by admin {admin_user_id}. "
                f"Reason: {reason}. Original owner: {lock_info.get('user_id', 'unknown')}"
            )
            
            return True
        
        return False
    
    def _build_lock_key(self, config_key: str, tenant_id: Optional[str]) -> str:
        """Build Redis key for lock"""
        tenant_part = f"{tenant_id}:" if tenant_id else "global:"
        return f"{self.lock_prefix}{tenant_part}{config_key}"
    
    async def _start_heartbeat(self, lock_key: str, user_id: str):
        """Start heartbeat to keep lock alive"""
        async def heartbeat_loop():
            while lock_key in self.heartbeat_tasks:
                try:
                    # Extend lock expiry
                    await self._extend_lock(lock_key, self.default_timeout)
                    await asyncio.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat failed for lock {lock_key}: {e}")
                    break
        
        # Start heartbeat task
        task = asyncio.create_task(heartbeat_loop())
        self.heartbeat_tasks[lock_key] = task
    
    async def _stop_heartbeat(self, lock_key: str):
        """Stop heartbeat for lock"""
        if lock_key in self.heartbeat_tasks:
            task = self.heartbeat_tasks[lock_key]
            task.cancel()
            del self.heartbeat_tasks[lock_key]
    
    async def _extend_lock(self, lock_key: str, timeout: int):
        """Extend lock expiry"""
        await self.redis.expire(lock_key, timeout)
```

## Performance Monitoring

### Sync Performance Metrics

```python
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SyncMetrics:
    """Real-time synchronization performance metrics"""
    total_events: int = 0
    successful_syncs: int = 0
    failed_syncs: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    average_sync_latency: float = 0.0
    websocket_connections: int = 0
    kafka_messages_sent: int = 0
    kafka_messages_received: int = 0
    mqtt_messages_sent: int = 0
    mqtt_messages_received: int = 0
    redis_operations: int = 0
    locks_acquired: int = 0
    locks_released: int = 0
    lock_contentions: int = 0

class SyncPerformanceMonitor:
    """Monitor real-time synchronization performance"""
    
    def __init__(self, window_size=300):  # 5 minute window
        self.window_size = window_size
        self.metrics = SyncMetrics()
        self.latency_samples = deque(maxlen=1000)
        self.event_rates = defaultdict(lambda: deque(maxlen=window_size))
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
    
    def record_sync_event(self, event_type: str, latency: float, success: bool):
        """Record synchronization event metrics"""
        current_time = time.time()
        
        self.metrics.total_events += 1
        
        if success:
            self.metrics.successful_syncs += 1
        else:
            self.metrics.failed_syncs += 1
            self.error_counts[event_type] += 1
        
        # Record latency
        self.latency_samples.append(latency)
        self._update_average_latency()
        
        # Record event rate
        self.event_rates[event_type].append(current_time)
    
    def record_conflict(self, resolved: bool):
        """Record conflict detection and resolution"""
        self.metrics.conflicts_detected += 1
        if resolved:
            self.metrics.conflicts_resolved += 1
    
    def record_connection_change(self, connection_type: str, delta: int):
        """Record connection count changes"""
        if connection_type == "websocket":
            self.metrics.websocket_connections += delta
    
    def record_message(self, transport: str, direction: str):
        """Record message transmission"""
        if transport == "kafka":
            if direction == "sent":
                self.metrics.kafka_messages_sent += 1
            else:
                self.metrics.kafka_messages_received += 1
        elif transport == "mqtt":
            if direction == "sent":
                self.metrics.mqtt_messages_sent += 1
            else:
                self.metrics.mqtt_messages_received += 1
    
    def record_lock_operation(self, operation: str, success: bool):
        """Record lock operations"""
        if operation == "acquire":
            if success:
                self.metrics.locks_acquired += 1
            else:
                self.metrics.lock_contentions += 1
        elif operation == "release":
            self.metrics.locks_released += 1
    
    def record_redis_operation(self):
        """Record Redis operation"""
        self.metrics.redis_operations += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        uptime = time.time() - self.start_time
        event_rate = self.metrics.total_events / uptime if uptime > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_events': self.metrics.total_events,
            'event_rate_per_second': event_rate,
            'success_rate': self._calculate_success_rate(),
            'average_latency_ms': self.metrics.average_sync_latency * 1000,
            'percentile_latencies': self._calculate_latency_percentiles(),
            'conflict_resolution_rate': self._calculate_conflict_resolution_rate(),
            'active_connections': {
                'websocket': self.metrics.websocket_connections
            },
            'message_throughput': {
                'kafka_sent': self.metrics.kafka_messages_sent,
                'kafka_received': self.metrics.kafka_messages_received,
                'mqtt_sent': self.metrics.mqtt_messages_sent,
                'mqtt_received': self.metrics.mqtt_messages_received
            },
            'lock_statistics': {
                'acquired': self.metrics.locks_acquired,
                'released': self.metrics.locks_released,
                'contentions': self.metrics.lock_contentions,
                'contention_rate': self._calculate_lock_contention_rate()
            },
            'error_breakdown': dict(self.error_counts),
            'redis_operations': self.metrics.redis_operations
        }
    
    def get_event_rates(self) -> Dict[str, float]:
        """Get event rates by type"""
        current_time = time.time()
        rates = {}
        
        for event_type, timestamps in self.event_rates.items():
            # Count events in the last window
            recent_events = [
                ts for ts in timestamps 
                if current_time - ts <= self.window_size
            ]
            rates[event_type] = len(recent_events) / self.window_size
        
        return rates
    
    def _update_average_latency(self):
        """Update average latency calculation"""
        if self.latency_samples:
            self.metrics.average_sync_latency = sum(self.latency_samples) / len(self.latency_samples)
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate percentage"""
        total = self.metrics.successful_syncs + self.metrics.failed_syncs
        if total == 0:
            return 100.0
        return (self.metrics.successful_syncs / total) * 100.0
    
    def _calculate_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latency_samples:
            return {'p50': 0, 'p95': 0, 'p99': 0}
        
        sorted_latencies = sorted(self.latency_samples)
        count = len(sorted_latencies)
        
        return {
            'p50': sorted_latencies[int(count * 0.50)] * 1000,  # Convert to ms
            'p95': sorted_latencies[int(count * 0.95)] * 1000,
            'p99': sorted_latencies[int(count * 0.99)] * 1000
        }
    
    def _calculate_conflict_resolution_rate(self) -> float:
        """Calculate conflict resolution rate"""
        if self.metrics.conflicts_detected == 0:
            return 100.0
        return (self.metrics.conflicts_resolved / self.metrics.conflicts_detected) * 100.0
    
    def _calculate_lock_contention_rate(self) -> float:
        """Calculate lock contention rate"""
        total_attempts = self.metrics.locks_acquired + self.metrics.lock_contentions
        if total_attempts == 0:
            return 0.0
        return (self.metrics.lock_contentions / total_attempts) * 100.0
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = SyncMetrics()
        self.latency_samples.clear()
        self.event_rates.clear()
        self.error_counts.clear()
        self.start_time = time.time()

# Integration with sync manager
class MonitoredRealtimeSyncManager(RealtimeSyncManager):
    """RealtimeSyncManager with performance monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_monitor = SyncPerformanceMonitor()
    
    async def broadcast_sync_event(self, event: SyncEvent):
        """Override to add performance monitoring"""
        start_time = time.time()
        success = False
        
        try:
            await super().broadcast_sync_event(event)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            latency = time.time() - start_time
            self.performance_monitor.record_sync_event(
                event.event_type.value,
                latency,
                success
            )
    
    async def detect_and_resolve_conflicts(self, *args, **kwargs):
        """Override to monitor conflict resolution"""
        try:
            result = await super().detect_and_resolve_conflicts(*args, **kwargs)
            self.performance_monitor.record_conflict(result.resolved)
            return result
        except Exception:
            self.performance_monitor.record_conflict(False)
            raise
    
    async def acquire_config_lock(self, *args, **kwargs):
        """Override to monitor lock operations"""
        try:
            result = await super().acquire_config_lock(*args, **kwargs)
            self.performance_monitor.record_lock_operation("acquire", result)
            return result
        except Exception:
            self.performance_monitor.record_lock_operation("acquire", False)
            raise
    
    async def release_config_lock(self, *args, **kwargs):
        """Override to monitor lock operations"""
        result = await super().release_config_lock(*args, **kwargs)
        self.performance_monitor.record_lock_operation("release", result)
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_monitor.get_current_metrics()
```

## Best Practices

### Scalability
1. **Use connection pooling for database and Redis**
2. **Implement message batching for high throughput**
3. **Partition Kafka topics by tenant or configuration type**
4. **Use MQTT QoS levels appropriately**
5. **Monitor and tune WebSocket connection limits**

### Reliability
1. **Implement circuit breakers for external services**
2. **Use message acknowledgments and retries**
3. **Set up health checks for all components**
4. **Monitor message lag and processing times**
5. **Implement graceful degradation**

### Security
1. **Use TLS for all network communication**
2. **Implement proper authentication for message brokers**
3. **Validate all incoming messages**
4. **Use encryption for sensitive configuration data**
5. **Audit all synchronization events**

### Monitoring
1. **Track synchronization latency and throughput**
2. **Monitor conflict resolution rates**
3. **Alert on failed synchronizations**
4. **Dashboard for real-time system health**
5. **Log all critical synchronization events**

## Next Steps

- Configure [Machine Learning Models](ml-models.md)
- Set up [Security and Encryption](../security/encryption.md)
- Review [Troubleshooting Guide](../troubleshooting/common-issues.md)
- Learn about [Performance Tuning](performance-tuning.md)