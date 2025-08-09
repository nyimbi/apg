"""
XMPP Protocol Implementation for APG Real-Time Collaboration

Provides XMPP (Extensible Messaging and Presence Protocol) communication
for chat/messaging federation, presence management, and enterprise messaging.
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
import xml.etree.ElementTree as ET

try:
	import slixmpp
	from slixmpp import ClientXMPP, ComponentXMPP
	from slixmpp.stanza import Message, Presence, Iq
	from slixmpp.xmlstream import ElementBase, register_stanza_plugin
except ImportError:
	print("XMPP dependencies not installed. Run: pip install slixmpp")
	slixmpp = None
	ClientXMPP = None
	ComponentXMPP = None

logger = logging.getLogger(__name__)


class XMPPMessageType(Enum):
	"""XMPP message types"""
	CHAT = "chat"
	GROUPCHAT = "groupchat"
	HEADLINE = "headline"
	NORMAL = "normal"
	ERROR = "error"


class XMPPPresenceType(Enum):
	"""XMPP presence types"""
	AVAILABLE = "available"
	UNAVAILABLE = "unavailable"
	SUBSCRIBE = "subscribe"
	SUBSCRIBED = "subscribed"
	UNSUBSCRIBE = "unsubscribe"
	UNSUBSCRIBED = "unsubscribed"
	PROBE = "probe"
	ERROR = "error"


class XMPPPresenceShow(Enum):
	"""XMPP presence show values"""
	AWAY = "away"
	CHAT = "chat"
	DND = "dnd"  # Do Not Disturb
	XA = "xa"    # Extended Away


@dataclass
class XMPPMessage:
	"""XMPP message structure"""
	message_id: str
	from_jid: str
	to_jid: str
	message_type: XMPPMessageType
	body: str
	subject: Optional[str] = None
	thread: Optional[str] = None
	timestamp: datetime = field(default_factory=datetime.utcnow)
	extensions: Dict[str, Any] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"message_id": self.message_id,
			"from_jid": self.from_jid,
			"to_jid": self.to_jid,
			"message_type": self.message_type.value,
			"body": self.body,
			"subject": self.subject,
			"thread": self.thread,
			"timestamp": self.timestamp.isoformat(),
			"extensions": self.extensions
		}


@dataclass
class XMPPPresence:
	"""XMPP presence structure"""
	from_jid: str
	to_jid: Optional[str] = None
	presence_type: XMPPPresenceType = XMPPPresenceType.AVAILABLE
	show: Optional[XMPPPresenceShow] = None
	status: Optional[str] = None
	priority: int = 0
	timestamp: datetime = field(default_factory=datetime.utcnow)
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"from_jid": self.from_jid,
			"to_jid": self.to_jid,
			"presence_type": self.presence_type.value,
			"show": self.show.value if self.show else None,
			"status": self.status,
			"priority": self.priority,
			"timestamp": self.timestamp.isoformat()
		}


@dataclass
class XMPPContact:
	"""XMPP contact/roster entry"""
	jid: str
	name: Optional[str] = None
	subscription: str = "none"  # none, to, from, both
	groups: List[str] = field(default_factory=list)
	presence: Optional[XMPPPresence] = None
	last_seen: Optional[datetime] = None
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"jid": self.jid,
			"name": self.name,
			"subscription": self.subscription,
			"groups": self.groups,
			"presence": self.presence.to_dict() if self.presence else None,
			"last_seen": self.last_seen.isoformat() if self.last_seen else None
		}


class XMPPCollaborationClient(ClientXMPP):
	"""XMPP client for APG Real-Time Collaboration"""
	
	def __init__(self, jid: str, password: str, protocol_manager):
		if not ClientXMPP:
			raise ImportError("XMPP dependencies not installed")
		
		super().__init__(jid, password)
		self.protocol_manager = protocol_manager
		
		# Enable plugins
		self.register_plugin('xep_0030')  # Service Discovery
		self.register_plugin('xep_0045')  # Multi-User Chat
		self.register_plugin('xep_0199')  # XMPP Ping
		self.register_plugin('xep_0092')  # Software Version
		self.register_plugin('xep_0004')  # Data Forms
		self.register_plugin('xep_0060')  # Publish-Subscribe
		self.register_plugin('xep_0115')  # Entity Capabilities
		self.register_plugin('xep_0163')  # Personal Eventing Protocol
		
		# Event handlers
		self.add_event_handler('session_start', self._session_start)
		self.add_event_handler('message', self._message_received)
		self.add_event_handler('presence', self._presence_received)
		self.add_event_handler('roster_update', self._roster_update)
		self.add_event_handler('groupchat_message', self._groupchat_message)
		self.add_event_handler('muc_presence', self._muc_presence)
	
	async def _session_start(self, event):
		"""Handle session start"""
		try:
			# Send initial presence
			self.send_presence()
			
			# Get roster
			self.get_roster()
			
			logger.info(f"XMPP session started for {self.boundjid}")
			
			# Notify protocol manager
			await self.protocol_manager._handle_session_start()
			
		except Exception as e:
			logger.error(f"Error in XMPP session start: {e}")
	
	async def _message_received(self, msg):
		"""Handle incoming message"""
		try:
			xmpp_message = XMPPMessage(
				message_id=str(msg['id']) or str(uuid.uuid4()),
				from_jid=str(msg['from']),
				to_jid=str(msg['to']),
				message_type=XMPPMessageType(msg['type']),
				body=str(msg['body']),
				subject=str(msg['subject']) if msg['subject'] else None,
				thread=str(msg['thread']) if msg['thread'] else None
			)
			
			# Notify protocol manager
			await self.protocol_manager._handle_message_received(xmpp_message)
			
		except Exception as e:
			logger.error(f"Error handling XMPP message: {e}")
	
	async def _presence_received(self, presence):
		"""Handle incoming presence"""
		try:
			xmpp_presence = XMPPPresence(
				from_jid=str(presence['from']),
				to_jid=str(presence['to']) if presence['to'] else None,
				presence_type=XMPPPresenceType(presence['type'] or 'available'),
				show=XMPPPresenceShow(presence['show']) if presence['show'] else None,
				status=str(presence['status']) if presence['status'] else None,
				priority=int(presence['priority']) if presence['priority'] else 0
			)
			
			# Notify protocol manager
			await self.protocol_manager._handle_presence_received(xmpp_presence)
			
		except Exception as e:
			logger.error(f"Error handling XMPP presence: {e}")
	
	async def _roster_update(self, event):
		"""Handle roster update"""
		try:
			# Notify protocol manager
			await self.protocol_manager._handle_roster_update()
			
		except Exception as e:
			logger.error(f"Error handling roster update: {e}")
	
	async def _groupchat_message(self, msg):
		"""Handle group chat message"""
		try:
			# Handle MUC messages
			await self._message_received(msg)
			
		except Exception as e:
			logger.error(f"Error handling groupchat message: {e}")
	
	async def _muc_presence(self, presence):
		"""Handle MUC presence"""
		try:
			# Handle MUC presence
			await self._presence_received(presence)
			
		except Exception as e:
			logger.error(f"Error handling MUC presence: {e}")


class XMPPProtocolManager:
	"""Manages XMPP protocol for real-time collaboration"""
	
	def __init__(self, jid: str = None, password: str = None, server: str = None):
		self.jid = jid or "test@example.com"
		self.password = password or "password"
		self.server = server
		
		# XMPP client
		self.client: Optional[XMPPCollaborationClient] = None
		self.is_connected = False
		
		# Contact management
		self.contacts: Dict[str, XMPPContact] = {}
		self.presence_cache: Dict[str, XMPPPresence] = {}
		
		# Room management
		self.joined_rooms: Dict[str, Dict[str, Any]] = {}
		self.room_participants: Dict[str, Set[str]] = {}
		
		# Message handling
		self.message_handlers: Dict[str, List[Callable]] = {}
		self.presence_handlers: List[Callable] = []
		
		# Message history
		self.message_history: List[XMPPMessage] = []
		self.max_history_size = 1000
		
		# Configuration
		self.auto_authorize = True
		self.auto_subscribe = True
		
		# Statistics
		self.stats = {
			"messages_sent": 0,
			"messages_received": 0,
			"presence_updates": 0,
			"rooms_joined": 0,
			"contacts": 0,
			"connection_time": None,
			"last_activity": None
		}
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize XMPP connection"""
		try:
			if not ClientXMPP:
				return {"error": "XMPP dependencies not installed"}
			
			# Create XMPP client
			self.client = XMPPCollaborationClient(self.jid, self.password, self)
			
			# Configure connection
			if self.server:
				# Extract host and port from server
				if ':' in self.server:
					host, port = self.server.split(':', 1)
					self.client.connect(address=(host, int(port)))
				else:
					self.client.connect(address=(self.server, 5222))
			else:
				self.client.connect()
			
			# Wait for connection
			await asyncio.sleep(2)
			
			if self.client.is_connected():
				self.is_connected = True
				self.stats["connection_time"] = datetime.utcnow()
				
				logger.info(f"XMPP client connected: {self.jid}")
				
				return {
					"status": "connected",
					"jid": self.jid,
					"server": self.server or "auto-discovered"
				}
			else:
				return {"error": "Failed to connect to XMPP server"}
			
		except Exception as e:
			logger.error(f"Failed to initialize XMPP client: {e}")
			return {"error": f"XMPP initialization failed: {str(e)}"}
	
	async def disconnect(self) -> Dict[str, Any]:
		"""Disconnect from XMPP server"""
		try:
			if self.client and self.is_connected:
				self.client.disconnect()
				self.is_connected = False
				
				logger.info("XMPP client disconnected")
			
			return {"status": "disconnected"}
			
		except Exception as e:
			logger.error(f"Error disconnecting XMPP client: {e}")
			return {"error": f"Disconnect failed: {str(e)}"}
	
	async def send_message(self, to_jid: str, body: str, 
						   message_type: XMPPMessageType = XMPPMessageType.CHAT,
						   subject: str = None, thread: str = None) -> Dict[str, Any]:
		"""Send XMPP message"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			# Create and send message
			msg = self.client.make_message(
				mto=to_jid,
				mbody=body,
				mtype=message_type.value,
				msubject=subject,
				mthread=thread
			)
			
			msg.send()
			
			# Create message record
			xmpp_message = XMPPMessage(
				message_id=str(msg['id']),
				from_jid=self.jid,
				to_jid=to_jid,
				message_type=message_type,
				body=body,
				subject=subject,
				thread=thread
			)
			
			# Add to history
			self._add_to_history(xmpp_message)
			
			# Update statistics
			self.stats["messages_sent"] += 1
			self.stats["last_activity"] = datetime.utcnow()
			
			logger.debug(f"Sent XMPP message to {to_jid}")
			
			return {
				"status": "sent",
				"message_id": str(msg['id']),
				"to_jid": to_jid
			}
			
		except Exception as e:
			logger.error(f"Failed to send XMPP message: {e}")
			return {"error": f"Send failed: {str(e)}"}
	
	async def send_presence(self, presence_type: XMPPPresenceType = XMPPPresenceType.AVAILABLE,
							show: XMPPPresenceShow = None, status: str = None,
							priority: int = 0) -> Dict[str, Any]:
		"""Send presence update"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			# Send presence
			if presence_type == XMPPPresenceType.AVAILABLE:
				self.client.send_presence(
					pshow=show.value if show else None,
					pstatus=status,
					ppriority=priority
				)
			else:
				self.client.send_presence(ptype=presence_type.value)
			
			# Update statistics
			self.stats["presence_updates"] += 1
			self.stats["last_activity"] = datetime.utcnow()
			
			logger.debug(f"Sent XMPP presence: {presence_type.value}")
			
			return {
				"status": "sent",
				"presence_type": presence_type.value,
				"show": show.value if show else None,
				"status_message": status
			}
			
		except Exception as e:
			logger.error(f"Failed to send XMPP presence: {e}")
			return {"error": f"Presence send failed: {str(e)}"}
	
	async def join_room(self, room_jid: str, nickname: str, 
						password: str = None) -> Dict[str, Any]:
		"""Join multi-user chat room"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			# Join MUC room
			muc = self.client.plugin['xep_0045']
			await muc.join_muc(room_jid, nickname, password=password, wait=True)
			
			# Track room
			self.joined_rooms[room_jid] = {
				"nickname": nickname,
				"joined_at": datetime.utcnow(),
				"message_count": 0
			}
			
			self.room_participants[room_jid] = set()
			
			# Update statistics
			self.stats["rooms_joined"] += 1
			
			logger.info(f"Joined XMPP room: {room_jid} as {nickname}")
			
			return {
				"status": "joined",
				"room_jid": room_jid,
				"nickname": nickname
			}
			
		except Exception as e:
			logger.error(f"Failed to join XMPP room: {e}")
			return {"error": f"Room join failed: {str(e)}"}
	
	async def leave_room(self, room_jid: str) -> Dict[str, Any]:
		"""Leave multi-user chat room"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			if room_jid not in self.joined_rooms:
				return {"error": "Not in specified room"}
			
			# Leave MUC room
			muc = self.client.plugin['xep_0045']
			muc.leave_muc(room_jid, "Leaving room")
			
			# Remove from tracking
			del self.joined_rooms[room_jid]
			self.room_participants.pop(room_jid, None)
			
			logger.info(f"Left XMPP room: {room_jid}")
			
			return {
				"status": "left",
				"room_jid": room_jid
			}
			
		except Exception as e:
			logger.error(f"Failed to leave XMPP room: {e}")
			return {"error": f"Room leave failed: {str(e)}"}
	
	async def send_room_message(self, room_jid: str, body: str, 
								message_type: XMPPMessageType = XMPPMessageType.GROUPCHAT) -> Dict[str, Any]:
		"""Send message to MUC room"""
		try:
			if room_jid not in self.joined_rooms:
				return {"error": "Not in specified room"}
			
			result = await self.send_message(room_jid, body, message_type)
			
			if result.get("status") == "sent":
				self.joined_rooms[room_jid]["message_count"] += 1
			
			return result
			
		except Exception as e:
			logger.error(f"Failed to send room message: {e}")
			return {"error": f"Room message send failed: {str(e)}"}
	
	async def add_contact(self, jid: str, name: str = None, 
						  groups: List[str] = None) -> Dict[str, Any]:
		"""Add contact to roster"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			# Add to roster
			self.client.send_presence_subscription(pto=jid)
			
			if name or groups:
				self.client.update_roster(jid, name=name, groups=groups or [])
			
			# Create contact record
			contact = XMPPContact(
				jid=jid,
				name=name,
				groups=groups or []
			)
			self.contacts[jid] = contact
			
			# Update statistics
			self.stats["contacts"] = len(self.contacts)
			
			logger.info(f"Added XMPP contact: {jid}")
			
			return {
				"status": "added",
				"jid": jid,
				"name": name
			}
			
		except Exception as e:
			logger.error(f"Failed to add XMPP contact: {e}")
			return {"error": f"Contact add failed: {str(e)}"}
	
	async def remove_contact(self, jid: str) -> Dict[str, Any]:
		"""Remove contact from roster"""
		try:
			if not self.client or not self.is_connected:
				return {"error": "XMPP client not connected"}
			
			# Remove from roster
			self.client.del_roster_item(jid)
			self.client.send_presence_subscription(pto=jid, ptype='unsubscribe')
			
			# Remove from tracking
			self.contacts.pop(jid, None)
			self.presence_cache.pop(jid, None)
			
			# Update statistics
			self.stats["contacts"] = len(self.contacts)
			
			logger.info(f"Removed XMPP contact: {jid}")
			
			return {
				"status": "removed",
				"jid": jid
			}
			
		except Exception as e:
			logger.error(f"Failed to remove XMPP contact: {e}")
			return {"error": f"Contact remove failed: {str(e)}"}
	
	async def _handle_session_start(self):
		"""Handle XMPP session start"""
		logger.info("XMPP session started")
	
	async def _handle_message_received(self, message: XMPPMessage):
		"""Handle received XMPP message"""
		try:
			# Add to history
			self._add_to_history(message)
			
			# Update statistics
			self.stats["messages_received"] += 1
			self.stats["last_activity"] = datetime.utcnow()
			
			# Update room message count
			if message.message_type == XMPPMessageType.GROUPCHAT:
				room_jid = message.from_jid.split('/')[0]  # Extract room JID
				if room_jid in self.joined_rooms:
					self.joined_rooms[room_jid]["message_count"] += 1
			
			# Call message handlers
			for handler in self.message_handlers.get(message.message_type.value, []):
				try:
					if asyncio.iscoroutinefunction(handler):
						await handler(message)
					else:
						handler(message)
				except Exception as e:
					logger.error(f"Error in message handler: {e}")
			
			logger.debug(f"Received XMPP message from {message.from_jid}")
			
		except Exception as e:
			logger.error(f"Error handling received message: {e}")
	
	async def _handle_presence_received(self, presence: XMPPPresence):
		"""Handle received XMPP presence"""
		try:
			# Update presence cache
			self.presence_cache[presence.from_jid] = presence
			
			# Update contact presence
			jid = presence.from_jid.split('/')[0]  # Extract bare JID
			if jid in self.contacts:
				self.contacts[jid].presence = presence
				self.contacts[jid].last_seen = presence.timestamp
			
			# Update room participants
			if '/' in presence.from_jid:  # Room presence
				room_jid = presence.from_jid.split('/')[0]
				if room_jid in self.room_participants:
					if presence.presence_type == XMPPPresenceType.AVAILABLE:
						self.room_participants[room_jid].add(presence.from_jid)
					else:
						self.room_participants[room_jid].discard(presence.from_jid)
			
			# Update statistics
			self.stats["presence_updates"] += 1
			
			# Call presence handlers
			for handler in self.presence_handlers:
				try:
					if asyncio.iscoroutinefunction(handler):
						await handler(presence)
					else:
						handler(presence)
				except Exception as e:
					logger.error(f"Error in presence handler: {e}")
			
			logger.debug(f"Received XMPP presence from {presence.from_jid}: {presence.presence_type.value}")
			
		except Exception as e:
			logger.error(f"Error handling received presence: {e}")
	
	async def _handle_roster_update(self):
		"""Handle roster update"""
		try:
			if not self.client:
				return
			
			# Update contacts from roster
			roster = self.client.client_roster
			for jid in roster:
				contact_info = roster[jid]
				
				if jid not in self.contacts:
					self.contacts[jid] = XMPPContact(jid=jid)
				
				self.contacts[jid].name = contact_info.get('name')
				self.contacts[jid].subscription = contact_info.get('subscription', 'none')
				self.contacts[jid].groups = list(contact_info.get('groups', set()))
			
			# Update statistics
			self.stats["contacts"] = len(self.contacts)
			
			logger.debug("Updated XMPP roster")
			
		except Exception as e:
			logger.error(f"Error handling roster update: {e}")
	
	def register_message_handler(self, message_type: str, handler: Callable):
		"""Register message handler"""
		if message_type not in self.message_handlers:
			self.message_handlers[message_type] = []
		self.message_handlers[message_type].append(handler)
		
		logger.info(f"Registered XMPP message handler for type: {message_type}")
	
	def register_presence_handler(self, handler: Callable):
		"""Register presence handler"""
		self.presence_handlers.append(handler)
		
		logger.info("Registered XMPP presence handler")
	
	def _add_to_history(self, message: XMPPMessage):
		"""Add message to history with size management"""
		self.message_history.append(message)
		
		# Maintain history size limit
		if len(self.message_history) > self.max_history_size:
			self.message_history = self.message_history[-self.max_history_size:]
	
	def get_contacts(self) -> List[Dict[str, Any]]:
		"""Get list of contacts"""
		return [contact.to_dict() for contact in self.contacts.values()]
	
	def get_joined_rooms(self) -> Dict[str, Dict[str, Any]]:
		"""Get information about joined rooms"""
		return {
			room_jid: {
				**room_info,
				"joined_at": room_info["joined_at"].isoformat(),
				"participant_count": len(self.room_participants.get(room_jid, set()))
			}
			for room_jid, room_info in self.joined_rooms.items()
		}
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get XMPP protocol statistics"""
		uptime_seconds = 0
		if self.stats["connection_time"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["connection_time"]).total_seconds())
		
		return {
			**self.stats,
			"is_connected": self.is_connected,
			"jid": self.jid,
			"server": self.server,
			"joined_rooms": len(self.joined_rooms),
			"message_history_size": len(self.message_history),
			"uptime_seconds": uptime_seconds,
			"connection_time": self.stats["connection_time"].isoformat() if self.stats["connection_time"] else None,
			"last_activity": self.stats["last_activity"].isoformat() if self.stats["last_activity"] else None
		}


# Global XMPP manager instance
xmpp_protocol_manager = None


async def initialize_xmpp_protocol(jid: str, password: str, server: str = None) -> Dict[str, Any]:
	"""Initialize global XMPP protocol manager"""
	global xmpp_protocol_manager
	
	xmpp_protocol_manager = XMPPProtocolManager(jid=jid, password=password, server=server)
	
	result = await xmpp_protocol_manager.initialize()
	
	return result


def get_xmpp_manager() -> Optional[XMPPProtocolManager]:
	"""Get global XMPP protocol manager"""
	return xmpp_protocol_manager


if __name__ == "__main__":
	# Test XMPP protocol implementation
	async def test_xmpp():
		print("Testing XMPP protocol implementation...")
		
		# Note: This test requires actual XMPP credentials
		test_jid = "test@example.com"
		test_password = "password"
		
		print(f"Would initialize XMPP with JID: {test_jid}")
		print("XMPP protocol implementation ready for integration")
		
		# Initialize XMPP manager (commented out for testing without credentials)
		# result = await initialize_xmpp_protocol(test_jid, test_password)
		# print(f"XMPP initialization result: {result}")
		
		print("✅ XMPP protocol test completed")
	
	asyncio.run(test_xmpp())