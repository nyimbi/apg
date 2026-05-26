"""
APG Connection Management Real-Time Notifications
WebSocket-based real-time notifications and event streaming

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from functools import wraps
import weakref
from collections import defaultdict, deque
import time

# WebSocket and async imports
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosedError
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available. Real-time notifications disabled.")

# Socket.IO for more advanced real-time features
try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

# Flask-SocketIO for Flask integration
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
    FLASK_SOCKETIO_AVAILABLE = True
except ImportError:
    FLASK_SOCKETIO_AVAILABLE = False

from .error_handling import APGError, ErrorContext
from .security import SecurityContext, require_authentication
from .monitoring import global_metrics_collector

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications"""
    CONNECTION_STATUS = "connection_status"
    FLOW_STATUS = "flow_status"
    DATA_LINEAGE = "data_lineage"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    CAPABILITY_EVENT = "capability_event"
    ERROR_EVENT = "error_event"
    SECURITY_EVENT = "security_event"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ChannelType(str, Enum):
    """Types of notification channels"""
    USER = "user"              # User-specific notifications
    TENANT = "tenant"          # Tenant-wide notifications
    CONNECTION = "connection"  # Connection-specific notifications
    FLOW = "flow"             # Flow-specific notifications
    SYSTEM = "system"         # System-wide notifications
    BROADCAST = "broadcast"   # All users


@dataclass
class NotificationMessage:
    """Notification message structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NotificationType = NotificationType.USER_ACTION
    priority: NotificationPriority = NotificationPriority.NORMAL
    title: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    sender_id: Optional[str] = None
    tenant_id: str = "system"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'sender_id': self.sender_id,
            'tenant_id': self.tenant_id,
            'tags': self.tags
        }

    def is_expired(self) -> bool:
        """Check if notification has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class WebSocketClient:
    """WebSocket client connection info"""
    connection_id: str
    websocket: Any  # WebSocket connection object
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    subscriptions: Set[str] = field(default_factory=set)
    last_ping: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationFilter:
    """Filter for notification routing"""

    def __init__(self,
                 types: List[NotificationType] = None,
                 priorities: List[NotificationPriority] = None,
                 tenant_ids: List[str] = None,
                 user_ids: List[str] = None,
                 tags: List[str] = None):
        self.types = types or []
        self.priorities = priorities or []
        self.tenant_ids = tenant_ids or []
        self.user_ids = user_ids or []
        self.tags = tags or []

    def matches(self, notification: NotificationMessage) -> bool:
        """Check if notification matches filter criteria"""
        if self.types and notification.type not in self.types:
            return False

        if self.priorities and notification.priority not in self.priorities:
            return False

        if self.tenant_ids and notification.tenant_id not in self.tenant_ids:
            return False

        if self.user_ids and notification.sender_id not in self.user_ids:
            return False

        if self.tags and not any(tag in notification.tags for tag in self.tags):
            return False

        return True


class WebSocketNotificationServer:
    """WebSocket server for real-time notifications"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketClient] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # channel -> client_ids
        self.message_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.server = None
        self.running = False

        # Statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_failed': 0,
            'uptime_start': datetime.now(timezone.utc)
        }

    async def start_server(self):
        """Start WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            raise APGError(
                message="WebSockets not available",
                context=ErrorContext(tenant_id="system", operation="start_websocket_server")
            )

        self.server = await websockets.serve(
            self.handle_client_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10**6,  # 1MB max message size
            compression=None
        )

        self.running = True
        logger.info(f"WebSocket notification server started on {self.host}:{self.port}")

        # Start background tasks
        asyncio.create_task(self._cleanup_expired_clients())
        asyncio.create_task(self._send_periodic_stats())

    async def stop_server(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            logger.info("WebSocket notification server stopped")

    async def handle_client_connection(self, websocket, path):
        """Handle new client WebSocket connection"""
        connection_id = str(uuid.uuid4())
        client = WebSocketClient(
            connection_id=connection_id,
            websocket=websocket
        )

        self.clients[connection_id] = client
        self.stats['total_connections'] += 1
        self.stats['active_connections'] += 1

        logger.info(f"New WebSocket client connected: {connection_id}")

        try:
            # Send welcome message
            await self.send_to_client(connection_id, NotificationMessage(
                type=NotificationType.SYSTEM_ALERT,
                title="Connected",
                message="Successfully connected to APG Connection Management notifications",
                priority=NotificationPriority.LOW
            ))

            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(connection_id, message)

        except ConnectionClosedError:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket client error {connection_id}: {e}")
        finally:
            await self._disconnect_client(connection_id)

    async def _handle_client_message(self, connection_id: str, message: str):
        """Handle message from WebSocket client"""
        try:
            data = json.loads(message)
            action = data.get('action')

            if action == 'authenticate':
                await self._authenticate_client(connection_id, data)
            elif action == 'subscribe':
                await self._subscribe_client(connection_id, data)
            elif action == 'unsubscribe':
                await self._unsubscribe_client(connection_id, data)
            elif action == 'ping':
                await self._handle_ping(connection_id)
            else:
                await self.send_to_client(connection_id, NotificationMessage(
                    type=NotificationType.ERROR_EVENT,
                    title="Invalid Action",
                    message=f"Unknown action: {action}",
                    priority=NotificationPriority.NORMAL
                ))

        except json.JSONDecodeError:
            await self.send_to_client(connection_id, NotificationMessage(
                type=NotificationType.ERROR_EVENT,
                title="Invalid Message",
                message="Message must be valid JSON",
                priority=NotificationPriority.NORMAL
            ))
        except Exception as e:
            logger.error(f"Error handling client message {connection_id}: {e}")

    async def _authenticate_client(self, connection_id: str, data: Dict[str, Any]):
        """Authenticate WebSocket client"""
        client = self.clients.get(connection_id)
        if not client:
            return

        # Extract authentication info
        token = data.get('token')
        user_id = data.get('user_id')
        tenant_id = data.get('tenant_id')
        session_id = data.get('session_id')

        # TODO: Validate token with security manager
        # For now, accept any authentication data

        client.user_id = user_id
        client.tenant_id = tenant_id
        client.session_id = session_id

        await self.send_to_client(connection_id, NotificationMessage(
            type=NotificationType.USER_ACTION,
            title="Authenticated",
            message=f"Successfully authenticated as {user_id}",
            priority=NotificationPriority.LOW
        ))

        logger.info(f"Client {connection_id} authenticated as {user_id}")

    async def _subscribe_client(self, connection_id: str, data: Dict[str, Any]):
        """Subscribe client to channels"""
        client = self.clients.get(connection_id)
        if not client:
            return

        channels = data.get('channels', [])
        subscribed = []

        for channel in channels:
            # Validate subscription permissions
            if self._can_subscribe_to_channel(client, channel):
                client.subscriptions.add(channel)
                self.subscriptions[channel].add(connection_id)
                subscribed.append(channel)

        await self.send_to_client(connection_id, NotificationMessage(
            type=NotificationType.USER_ACTION,
            title="Subscribed",
            message=f"Subscribed to channels: {', '.join(subscribed)}",
            data={'channels': subscribed},
            priority=NotificationPriority.LOW
        ))

    async def _unsubscribe_client(self, connection_id: str, data: Dict[str, Any]):
        """Unsubscribe client from channels"""
        client = self.clients.get(connection_id)
        if not client:
            return

        channels = data.get('channels', [])
        unsubscribed = []

        for channel in channels:
            if channel in client.subscriptions:
                client.subscriptions.remove(channel)
                self.subscriptions[channel].discard(connection_id)
                unsubscribed.append(channel)

        await self.send_to_client(connection_id, NotificationMessage(
            type=NotificationType.USER_ACTION,
            title="Unsubscribed",
            message=f"Unsubscribed from channels: {', '.join(unsubscribed)}",
            data={'channels': unsubscribed},
            priority=NotificationPriority.LOW
        ))

    async def _handle_ping(self, connection_id: str):
        """Handle ping from client"""
        client = self.clients.get(connection_id)
        if client:
            client.last_ping = datetime.now(timezone.utc)

            # Send pong response
            await client.websocket.send(json.dumps({
                'action': 'pong',
                'timestamp': client.last_ping.isoformat()
            }))

    def _can_subscribe_to_channel(self, client: WebSocketClient, channel: str) -> bool:
        """Check if client can subscribe to channel"""
        # Basic permission checking
        if channel.startswith('user:'):
            # User-specific channel
            user_id = channel.split(':', 1)[1]
            return client.user_id == user_id

        elif channel.startswith('tenant:'):
            # Tenant-specific channel
            tenant_id = channel.split(':', 1)[1]
            return client.tenant_id == tenant_id

        elif channel in ['system', 'broadcast']:
            # System channels - allow all authenticated users
            return client.user_id is not None

        else:
            # Default: allow subscription
            return True

    async def _disconnect_client(self, connection_id: str):
        """Disconnect and cleanup client"""
        client = self.clients.get(connection_id)
        if not client:
            return

        # Remove from all subscriptions
        for channel in client.subscriptions:
            self.subscriptions[channel].discard(connection_id)

        # Remove client
        del self.clients[connection_id]
        self.stats['active_connections'] -= 1

        logger.info(f"Client {connection_id} disconnected and cleaned up")

    async def _cleanup_expired_clients(self):
        """Background task to cleanup expired clients"""
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                expired_clients = []

                for connection_id, client in self.clients.items():
                    # Check if client hasn't pinged for too long (5 minutes)
                    if (current_time - client.last_ping).total_seconds() > 300:
                        expired_clients.append(connection_id)

                # Disconnect expired clients
                for connection_id in expired_clients:
                    try:
                        await self._disconnect_client(connection_id)
                    except Exception as e:
                        logger.error(f"Error disconnecting expired client {connection_id}: {e}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

    async def _send_periodic_stats(self):
        """Send periodic statistics to system channels"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                stats_notification = NotificationMessage(
                    type=NotificationType.PERFORMANCE_METRIC,
                    title="System Statistics",
                    message="Periodic system statistics update",
                    data=self.get_stats(),
                    priority=NotificationPriority.LOW,
                    tags=['system', 'statistics']
                )

                await self.broadcast_to_channel('system', stats_notification)

            except Exception as e:
                logger.error(f"Error sending periodic stats: {e}")

    async def send_to_client(self, connection_id: str, notification: NotificationMessage) -> bool:
        """Send notification to specific client"""
        client = self.clients.get(connection_id)
        if not client:
            return False

        try:
            message = json.dumps({
                'action': 'notification',
                'notification': notification.to_dict()
            })

            await client.websocket.send(message)
            self.stats['messages_sent'] += 1

            # Store in message history
            self.message_history[connection_id].append(notification.to_dict())

            return True

        except Exception as e:
            logger.error(f"Error sending message to client {connection_id}: {e}")
            self.stats['messages_failed'] += 1
            return False

    async def broadcast_to_channel(self, channel: str, notification: NotificationMessage) -> int:
        """Broadcast notification to all clients in channel"""
        if channel not in self.subscriptions:
            return 0

        client_ids = list(self.subscriptions[channel])
        sent_count = 0

        for client_id in client_ids:
            if await self.send_to_client(client_id, notification):
                sent_count += 1

        logger.debug(f"Broadcast to channel '{channel}': {sent_count}/{len(client_ids)} clients")
        return sent_count

    async def send_to_user(self, user_id: str, notification: NotificationMessage) -> int:
        """Send notification to all connections of a user"""
        sent_count = 0

        for client in self.clients.values():
            if client.user_id == user_id:
                if await self.send_to_client(client.connection_id, notification):
                    sent_count += 1

        return sent_count

    async def send_to_tenant(self, tenant_id: str, notification: NotificationMessage) -> int:
        """Send notification to all users in a tenant"""
        sent_count = 0

        for client in self.clients.values():
            if client.tenant_id == tenant_id:
                if await self.send_to_client(client.connection_id, notification):
                    sent_count += 1

        return sent_count

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        uptime = (datetime.now(timezone.utc) - self.stats['uptime_start']).total_seconds()

        return {
            **self.stats,
            'uptime_seconds': uptime,
            'channels': len(self.subscriptions),
            'subscriptions': sum(len(clients) for clients in self.subscriptions.values())
        }

    def get_client_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific client"""
        client = self.clients.get(connection_id)
        if not client:
            return None

        return {
            'connection_id': client.connection_id,
            'user_id': client.user_id,
            'tenant_id': client.tenant_id,
            'session_id': client.session_id,
            'subscriptions': list(client.subscriptions),
            'connected_at': client.connected_at.isoformat(),
            'last_ping': client.last_ping.isoformat(),
            'metadata': client.metadata
        }


class FlaskSocketIONotificationServer:
    """Flask-SocketIO integration for notifications"""

    def __init__(self, app=None):
        self.app = app
        self.socketio = None
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)

        if app and FLASK_SOCKETIO_AVAILABLE:
            self.init_app(app)

    def init_app(self, app):
        """Initialize Flask-SocketIO with Flask app"""
        if not FLASK_SOCKETIO_AVAILABLE:
            logger.warning("Flask-SocketIO not available")
            return

        self.app = app
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode='threading',
            ping_timeout=60,
            ping_interval=25,
            logger=True
        )

        # Register event handlers
        self._register_handlers()

        logger.info("Flask-SocketIO notification server initialized")

    def _register_handlers(self):
        """Register Socket.IO event handlers"""
        if not self.socketio:
            return

        @self.socketio.on('connect')
        def handle_connect(auth=None):
            session_id = request.sid
            logger.info(f"Socket.IO client connected: {session_id}")

            self.clients[session_id] = {
                'connected_at': datetime.now(timezone.utc),
                'user_id': None,
                'tenant_id': None,
                'subscriptions': set()
            }

            emit('notification', {
                'type': 'system_alert',
                'message': 'Connected to APG notifications',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            session_id = request.sid
            logger.info(f"Socket.IO client disconnected: {session_id}")

            if session_id in self.clients:
                # Remove from subscriptions
                client = self.clients[session_id]
                for channel in client['subscriptions']:
                    self.subscriptions[channel].discard(session_id)

                del self.clients[session_id]

        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            session_id = request.sid
            user_id = data.get('user_id')
            tenant_id = data.get('tenant_id')
            token = data.get('token')

            # TODO: Validate token

            if session_id in self.clients:
                self.clients[session_id]['user_id'] = user_id
                self.clients[session_id]['tenant_id'] = tenant_id

                emit('authenticated', {
                    'user_id': user_id,
                    'tenant_id': tenant_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            session_id = request.sid
            channels = data.get('channels', [])

            if session_id not in self.clients:
                return

            client = self.clients[session_id]
            subscribed = []

            for channel in channels:
                if self._can_subscribe_to_channel(session_id, channel):
                    client['subscriptions'].add(channel)
                    self.subscriptions[channel].add(session_id)
                    subscribed.append(channel)

            emit('subscribed', {
                'channels': subscribed,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            session_id = request.sid
            channels = data.get('channels', [])

            if session_id not in self.clients:
                return

            client = self.clients[session_id]
            unsubscribed = []

            for channel in channels:
                if channel in client['subscriptions']:
                    client['subscriptions'].remove(channel)
                    self.subscriptions[channel].discard(session_id)
                    unsubscribed.append(channel)

            emit('unsubscribed', {
                'channels': unsubscribed,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

    def _can_subscribe_to_channel(self, session_id: str, channel: str) -> bool:
        """Check if client can subscribe to channel"""
        client = self.clients.get(session_id)
        if not client:
            return False

        if channel.startswith('user:'):
            user_id = channel.split(':', 1)[1]
            return client.get('user_id') == user_id

        elif channel.startswith('tenant:'):
            tenant_id = channel.split(':', 1)[1]
            return client.get('tenant_id') == tenant_id

        return True

    def send_to_client(self, session_id: str, notification: NotificationMessage):
        """Send notification to specific client"""
        if not self.socketio or session_id not in self.clients:
            return False

        try:
            self.socketio.emit('notification', notification.to_dict(), room=session_id)
            return True
        except Exception as e:
            logger.error(f"Error sending notification to {session_id}: {e}")
            return False

    def broadcast_to_channel(self, channel: str, notification: NotificationMessage) -> int:
        """Broadcast notification to channel"""
        if not self.socketio or channel not in self.subscriptions:
            return 0

        sent_count = 0
        for session_id in self.subscriptions[channel]:
            if self.send_to_client(session_id, notification):
                sent_count += 1

        return sent_count

    def send_to_user(self, user_id: str, notification: NotificationMessage) -> int:
        """Send notification to all connections of a user"""
        sent_count = 0

        for session_id, client in self.clients.items():
            if client.get('user_id') == user_id:
                if self.send_to_client(session_id, notification):
                    sent_count += 1

        return sent_count

    def send_to_tenant(self, tenant_id: str, notification: NotificationMessage) -> int:
        """Send notification to all users in tenant"""
        sent_count = 0

        for session_id, client in self.clients.items():
            if client.get('tenant_id') == tenant_id:
                if self.send_to_client(session_id, notification):
                    sent_count += 1

        return sent_count


class NotificationManager:
    """Main notification management system"""

    def __init__(self):
        self.websocket_server: Optional[WebSocketNotificationServer] = None
        self.flask_socketio_server: Optional[FlaskSocketIONotificationServer] = None
        self.message_history: deque = deque(maxlen=10000)
        self.filters: Dict[str, NotificationFilter] = {}
        self.event_handlers: Dict[NotificationType, List[Callable]] = defaultdict(list)

        # Statistics
        self.stats = {
            'total_notifications': 0,
            'notifications_by_type': defaultdict(int),
            'notifications_by_priority': defaultdict(int),
            'active_clients': 0
        }

    async def initialize(self, websocket_config: Dict[str, Any] = None):
        """Initialize notification servers"""

        # Initialize WebSocket server
        if websocket_config and WEBSOCKETS_AVAILABLE:
            host = websocket_config.get('host', 'localhost')
            port = websocket_config.get('port', 8765)

            self.websocket_server = WebSocketNotificationServer(host, port)
            await self.websocket_server.start_server()

        logger.info("Notification manager initialized")

    def init_flask_socketio(self, app):
        """Initialize Flask-SocketIO integration"""
        if FLASK_SOCKETIO_AVAILABLE:
            self.flask_socketio_server = FlaskSocketIONotificationServer(app)

    async def shutdown(self):
        """Shutdown notification servers"""
        if self.websocket_server:
            await self.websocket_server.stop_server()

    def register_event_handler(self, notification_type: NotificationType, handler: Callable):
        """Register event handler for notification type"""
        self.event_handlers[notification_type].append(handler)

    def create_filter(self, filter_name: str, filter_config: NotificationFilter):
        """Create named notification filter"""
        self.filters[filter_name] = filter_config

    async def send_notification(self,
                              notification: NotificationMessage,
                              channel_type: ChannelType = ChannelType.BROADCAST,
                              channel_id: str = None) -> Dict[str, int]:
        """Send notification through all available channels"""

        # Update statistics
        self.stats['total_notifications'] += 1
        self.stats['notifications_by_type'][notification.type.value] += 1
        self.stats['notifications_by_priority'][notification.priority.value] += 1

        # Store in history
        self.message_history.append(notification.to_dict())

        # Call event handlers
        handlers = self.event_handlers.get(notification.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")

        results = {}

        # Send through WebSocket server
        if self.websocket_server:
            if channel_type == ChannelType.BROADCAST:
                count = await self.websocket_server.broadcast_to_channel('broadcast', notification)
                results['websocket_broadcast'] = count
            elif channel_type == ChannelType.USER and channel_id:
                count = await self.websocket_server.send_to_user(channel_id, notification)
                results['websocket_user'] = count
            elif channel_type == ChannelType.TENANT and channel_id:
                count = await self.websocket_server.send_to_tenant(channel_id, notification)
                results['websocket_tenant'] = count
            elif channel_type in [ChannelType.CONNECTION, ChannelType.FLOW, ChannelType.SYSTEM] and channel_id:
                channel_name = f"{channel_type.value}:{channel_id}"
                count = await self.websocket_server.broadcast_to_channel(channel_name, notification)
                results[f'websocket_{channel_type.value}'] = count

        # Send through Flask-SocketIO server
        if self.flask_socketio_server:
            if channel_type == ChannelType.BROADCAST:
                count = self.flask_socketio_server.broadcast_to_channel('broadcast', notification)
                results['socketio_broadcast'] = count
            elif channel_type == ChannelType.USER and channel_id:
                count = self.flask_socketio_server.send_to_user(channel_id, notification)
                results['socketio_user'] = count
            elif channel_type == ChannelType.TENANT and channel_id:
                count = self.flask_socketio_server.send_to_tenant(channel_id, notification)
                results['socketio_tenant'] = count

        # Record metrics
        total_sent = sum(results.values())
        global_metrics_collector.record_counter("notifications_sent", total_sent)
        global_metrics_collector.record_counter("notifications_total", 1, {
            'type': notification.type.value,
            'priority': notification.priority.value
        })

        return results

    async def notify_connection_status(self, connection_id: str, status: str,
                                     message: str = None, tenant_id: str = None):
        """Send connection status notification"""
        notification = NotificationMessage(
            type=NotificationType.CONNECTION_STATUS,
            title=f"Connection {status.title()}",
            message=message or f"Connection {connection_id} is now {status}",
            data={
                'connection_id': connection_id,
                'status': status
            },
            tenant_id=tenant_id or "system",
            priority=NotificationPriority.HIGH if status == 'failed' else NotificationPriority.NORMAL,
            tags=['connection', status]
        )

        # Send to connection-specific channel and tenant
        results = {}
        results.update(await self.send_notification(notification, ChannelType.CONNECTION, connection_id))

        if tenant_id:
            results.update(await self.send_notification(notification, ChannelType.TENANT, tenant_id))

        return results

    async def notify_flow_status(self, flow_id: str, status: str,
                               message: str = None, tenant_id: str = None,
                               flow_data: Dict[str, Any] = None):
        """Send flow status notification"""
        notification = NotificationMessage(
            type=NotificationType.FLOW_STATUS,
            title=f"Flow {status.title()}",
            message=message or f"Data flow {flow_id} is now {status}",
            data={
                'flow_id': flow_id,
                'status': status,
                **(flow_data or {})
            },
            tenant_id=tenant_id or "system",
            priority=NotificationPriority.HIGH if status in ['failed', 'error'] else NotificationPriority.NORMAL,
            tags=['flow', status]
        )

        results = {}
        results.update(await self.send_notification(notification, ChannelType.FLOW, flow_id))

        if tenant_id:
            results.update(await self.send_notification(notification, ChannelType.TENANT, tenant_id))

        return results

    async def notify_system_alert(self, title: str, message: str,
                                priority: NotificationPriority = NotificationPriority.NORMAL,
                                data: Dict[str, Any] = None):
        """Send system-wide alert"""
        notification = NotificationMessage(
            type=NotificationType.SYSTEM_ALERT,
            title=title,
            message=message,
            data=data or {},
            priority=priority,
            tags=['system', 'alert']
        )

        return await self.send_notification(notification, ChannelType.SYSTEM, "alerts")

    async def notify_data_lineage_update(self, source_id: str, target_id: str,
                                       tenant_id: str = None, update_data: Dict[str, Any] = None):
        """Send data lineage update notification"""
        notification = NotificationMessage(
            type=NotificationType.DATA_LINEAGE,
            title="Data Lineage Updated",
            message=f"Lineage updated between {source_id} and {target_id}",
            data={
                'source_id': source_id,
                'target_id': target_id,
                **(update_data or {})
            },
            tenant_id=tenant_id or "system",
            priority=NotificationPriority.LOW,
            tags=['lineage', 'update']
        )

        results = {}
        results.update(await self.send_notification(notification, ChannelType.SYSTEM, "lineage"))

        if tenant_id:
            results.update(await self.send_notification(notification, ChannelType.TENANT, tenant_id))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        active_clients = 0

        if self.websocket_server:
            active_clients += self.websocket_server.stats['active_connections']

        if self.flask_socketio_server:
            active_clients += len(self.flask_socketio_server.clients)

        return {
            **dict(self.stats),
            'active_clients': active_clients,
            'message_history_size': len(self.message_history),
            'websocket_server_running': self.websocket_server is not None and self.websocket_server.running,
            'socketio_server_available': self.flask_socketio_server is not None
        }

    def get_recent_notifications(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent notifications from history"""
        return list(self.message_history)[-limit:]


# Global notification manager instance
global_notification_manager = NotificationManager()


# Convenience functions
async def send_notification(notification: NotificationMessage,
                          channel_type: ChannelType = ChannelType.BROADCAST,
                          channel_id: str = None) -> Dict[str, int]:
    """Global notification sending function"""
    return await global_notification_manager.send_notification(notification, channel_type, channel_id)


async def notify_connection_status(connection_id: str, status: str,
                                 message: str = None, tenant_id: str = None):
    """Global connection status notification"""
    return await global_notification_manager.notify_connection_status(
        connection_id, status, message, tenant_id
    )


async def notify_flow_status(flow_id: str, status: str,
                           message: str = None, tenant_id: str = None,
                           flow_data: Dict[str, Any] = None):
    """Global flow status notification"""
    return await global_notification_manager.notify_flow_status(
        flow_id, status, message, tenant_id, flow_data
    )


async def notify_system_alert(title: str, message: str,
                            priority: NotificationPriority = NotificationPriority.NORMAL,
                            data: Dict[str, Any] = None):
    """Global system alert notification"""
    return await global_notification_manager.notify_system_alert(title, message, priority, data)


# Notification decorator for automatic event notifications
def notify_on_event(notification_type: NotificationType,
                   channel_type: ChannelType = ChannelType.SYSTEM,
                   priority: NotificationPriority = NotificationPriority.NORMAL):
    """Decorator to automatically send notifications on function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)

                # Send success notification
                notification = NotificationMessage(
                    type=notification_type,
                    title=f"{func.__name__} completed successfully",
                    message=f"Operation {func.__name__} completed",
                    data={'function': func.__name__, 'result_type': type(result).__name__},
                    priority=priority,
                    tags=['success', func.__name__]
                )

                await global_notification_manager.send_notification(notification, channel_type)
                return result

            except Exception as e:
                # Send error notification
                notification = NotificationMessage(
                    type=NotificationType.ERROR_EVENT,
                    title=f"{func.__name__} failed",
                    message=f"Operation {func.__name__} failed: {str(e)}",
                    data={'function': func.__name__, 'error': str(e)},
                    priority=NotificationPriority.HIGH,
                    tags=['error', func.__name__]
                )

                await global_notification_manager.send_notification(notification, channel_type)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use a simplified approach
            try:
                result = func(*args, **kwargs)
                logger.info(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator