"""
WebRTC Signaling Server for APG Real-Time Collaboration

Handles WebRTC signaling, peer connection establishment, and media negotiation.
Integrates with the existing WebSocket infrastructure for seamless real-time communication.
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

try:
	from .websocket_manager import websocket_manager
	from .models import RTCVideoCall, RTCVideoParticipant
	from .service import RTCService
	from .config import get_config
except ImportError:
	# Fallback for development
	pass

logger = logging.getLogger(__name__)


class SignalingMessageType(Enum):
	# WebRTC signaling messages
	OFFER = "webrtc_offer"
	ANSWER = "webrtc_answer"
	ICE_CANDIDATE = "webrtc_ice_candidate"
	
	# Call management
	CALL_START = "webrtc_call_start"
	CALL_END = "webrtc_call_end"
	CALL_JOIN = "webrtc_call_join"
	CALL_LEAVE = "webrtc_call_leave"
	
	# Media management
	MEDIA_TOGGLE = "webrtc_media_toggle"
	SCREEN_SHARE_START = "webrtc_screen_share_start"
	SCREEN_SHARE_STOP = "webrtc_screen_share_stop"
	
	# Quality and optimization
	QUALITY_CHANGE = "webrtc_quality_change"
	BANDWIDTH_UPDATE = "webrtc_bandwidth_update"
	
	# Recording
	RECORDING_START = "webrtc_recording_start"
	RECORDING_STOP = "webrtc_recording_stop"
	
	# Error handling
	CONNECTION_ERROR = "webrtc_connection_error"
	MEDIA_ERROR = "webrtc_media_error"


@dataclass
class WebRTCPeer:
	"""Represents a WebRTC peer connection"""
	user_id: str
	call_id: str
	session_id: str
	peer_id: str
	connection_state: str = "new"
	ice_connection_state: str = "new"
	media_tracks: Set[str] = None
	screen_sharing: bool = False
	audio_enabled: bool = True
	video_enabled: bool = True
	bandwidth_limit: Optional[int] = None
	quality_profile: str = "hd"
	joined_at: datetime = None
	
	def __post_init__(self):
		if self.media_tracks is None:
			self.media_tracks = set()
		if self.joined_at is None:
			self.joined_at = datetime.utcnow()


class WebRTCSignalingServer:
	"""WebRTC signaling server for peer-to-peer connections"""
	
	def __init__(self):
		self.active_calls: Dict[str, Dict[str, WebRTCPeer]] = {}
		self.peer_connections: Dict[str, WebRTCPeer] = {}
		self.ice_servers = self._get_ice_servers()
		self.rtc_service = None
		
	def _get_ice_servers(self) -> List[Dict[str, Any]]:
		"""Get ICE servers configuration"""
		config = get_config()
		
		# Default STUN servers (public)
		ice_servers = [
			{"urls": "stun:stun.l.google.com:19302"},
			{"urls": "stun:stun1.l.google.com:19302"},
			{"urls": "stun:stun2.l.google.com:19302"}
		]
		
		# Add custom TURN servers if configured
		if hasattr(config, 'webrtc') and config.webrtc.turn_servers:
			for turn_server in config.webrtc.turn_servers:
				ice_servers.append({
					"urls": turn_server["url"],
					"username": turn_server.get("username"),
					"credential": turn_server.get("credential")
				})
		
		return ice_servers
	
	async def initialize(self):
		"""Initialize the signaling server"""
		try:
			from service import RTCService
			self.rtc_service = RTCService()
			logger.info("WebRTC signaling server initialized")
		except ImportError:
			logger.warning("RTCService not available, using mock for development")
	
	async def handle_signaling_message(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle incoming WebRTC signaling messages"""
		try:
			message_type = SignalingMessageType(message.get("type"))
			
			handler_map = {
				SignalingMessageType.CALL_START: self._handle_call_start,
				SignalingMessageType.CALL_JOIN: self._handle_call_join,
				SignalingMessageType.CALL_LEAVE: self._handle_call_leave,
				SignalingMessageType.CALL_END: self._handle_call_end,
				SignalingMessageType.OFFER: self._handle_offer,
				SignalingMessageType.ANSWER: self._handle_answer,
				SignalingMessageType.ICE_CANDIDATE: self._handle_ice_candidate,
				SignalingMessageType.MEDIA_TOGGLE: self._handle_media_toggle,
				SignalingMessageType.SCREEN_SHARE_START: self._handle_screen_share_start,
				SignalingMessageType.SCREEN_SHARE_STOP: self._handle_screen_share_stop,
				SignalingMessageType.QUALITY_CHANGE: self._handle_quality_change,
				SignalingMessageType.RECORDING_START: self._handle_recording_start,
				SignalingMessageType.RECORDING_STOP: self._handle_recording_stop,
			}
			
			handler = handler_map.get(message_type)
			if handler:
				return await handler(user_id, message)
			else:
				logger.warning(f"Unknown WebRTC message type: {message_type}")
				return {"error": f"Unknown message type: {message_type}"}
				
		except ValueError as e:
			logger.error(f"Invalid WebRTC message type: {message.get('type')} - {e}")
			return {"error": "Invalid message type"}
		except Exception as e:
			logger.error(f"Error handling WebRTC signaling message: {e}")
			return {"error": "Internal signaling error"}
	
	async def _handle_call_start(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle call start request"""
		call_id = message.get("call_id")
		session_id = message.get("session_id")
		
		if not call_id or not session_id:
			return {"error": "Missing call_id or session_id"}
		
		# Initialize call in active calls
		if call_id not in self.active_calls:
			self.active_calls[call_id] = {}
		
		# Create peer for the host
		peer_id = f"{user_id}_{call_id}"
		peer = WebRTCPeer(
			user_id=user_id,
			call_id=call_id,
			session_id=session_id,
			peer_id=peer_id,
			connection_state="connecting"
		)
		
		self.active_calls[call_id][user_id] = peer
		self.peer_connections[peer_id] = peer
		
		# Broadcast call start to session participants
		await self._broadcast_to_session(session_id, {
			"type": "webrtc_call_started",
			"call_id": call_id,
			"host_user_id": user_id,
			"ice_servers": self.ice_servers,
			"timestamp": datetime.utcnow().isoformat()
		}, exclude_user=user_id)
		
		logger.info(f"WebRTC call started: {call_id} by {user_id}")
		
		return {
			"status": "call_started",
			"call_id": call_id,
			"peer_id": peer_id,
			"ice_servers": self.ice_servers
		}
	
	async def _handle_call_join(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle call join request"""
		call_id = message.get("call_id")
		session_id = message.get("session_id")
		
		if not call_id or not session_id:
			return {"error": "Missing call_id or session_id"}
		
		if call_id not in self.active_calls:
			return {"error": "Call not found or ended"}
		
		# Create peer for joining participant
		peer_id = f"{user_id}_{call_id}"
		peer = WebRTCPeer(
			user_id=user_id,
			call_id=call_id,
			session_id=session_id,
			peer_id=peer_id,
			connection_state="connecting"
		)
		
		self.active_calls[call_id][user_id] = peer
		self.peer_connections[peer_id] = peer
		
		# Get existing participants for peer connection setup
		existing_peers = [
			{
				"user_id": p.user_id,
				"peer_id": p.peer_id,
				"audio_enabled": p.audio_enabled,
				"video_enabled": p.video_enabled,
				"screen_sharing": p.screen_sharing
			}
			for p in self.active_calls[call_id].values()
			if p.user_id != user_id
		]
		
		# Notify existing participants about new peer
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_peer_joined",
			"user_id": user_id,
			"peer_id": peer_id,
			"timestamp": datetime.utcnow().isoformat()
		}, exclude_user=user_id)
		
		logger.info(f"User {user_id} joined WebRTC call: {call_id}")
		
		return {
			"status": "call_joined",
			"call_id": call_id,
			"peer_id": peer_id,
			"ice_servers": self.ice_servers,
			"existing_peers": existing_peers
		}
	
	async def _handle_call_leave(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle call leave request"""
		call_id = message.get("call_id")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		# Remove peer from call
		if user_id in self.active_calls[call_id]:
			peer = self.active_calls[call_id][user_id]
			del self.active_calls[call_id][user_id]
			del self.peer_connections[peer.peer_id]
		
		# Notify remaining participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_peer_left",
			"user_id": user_id,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		# End call if no participants left
		if not self.active_calls[call_id]:
			del self.active_calls[call_id]
			logger.info(f"WebRTC call ended (no participants): {call_id}")
		
		logger.info(f"User {user_id} left WebRTC call: {call_id}")
		
		return {"status": "call_left", "call_id": call_id}
	
	async def _handle_call_end(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle call end request"""
		call_id = message.get("call_id")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		# Notify all participants that call is ending
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_call_ended",
			"ended_by": user_id,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		# Clean up all peer connections for this call
		for peer in self.active_calls[call_id].values():
			if peer.peer_id in self.peer_connections:
				del self.peer_connections[peer.peer_id]
		
		del self.active_calls[call_id]
		
		logger.info(f"WebRTC call ended: {call_id} by {user_id}")
		
		return {"status": "call_ended", "call_id": call_id}
	
	async def _handle_offer(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle WebRTC offer"""
		call_id = message.get("call_id")
		target_user_id = message.get("target_user_id")
		offer = message.get("offer")
		
		if not all([call_id, target_user_id, offer]):
			return {"error": "Missing required fields for offer"}
		
		# Forward offer to target peer
		await self._send_to_user(target_user_id, {
			"type": "webrtc_offer",
			"call_id": call_id,
			"from_user_id": user_id,
			"offer": offer,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		logger.debug(f"WebRTC offer forwarded from {user_id} to {target_user_id}")
		
		return {"status": "offer_sent"}
	
	async def _handle_answer(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle WebRTC answer"""
		call_id = message.get("call_id")
		target_user_id = message.get("target_user_id")
		answer = message.get("answer")
		
		if not all([call_id, target_user_id, answer]):
			return {"error": "Missing required fields for answer"}
		
		# Forward answer to target peer
		await self._send_to_user(target_user_id, {
			"type": "webrtc_answer",
			"call_id": call_id,
			"from_user_id": user_id,
			"answer": answer,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		logger.debug(f"WebRTC answer forwarded from {user_id} to {target_user_id}")
		
		return {"status": "answer_sent"}
	
	async def _handle_ice_candidate(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle ICE candidate"""
		call_id = message.get("call_id")
		target_user_id = message.get("target_user_id")
		candidate = message.get("candidate")
		
		if not all([call_id, target_user_id, candidate]):
			return {"error": "Missing required fields for ICE candidate"}
		
		# Forward ICE candidate to target peer
		await self._send_to_user(target_user_id, {
			"type": "webrtc_ice_candidate",
			"call_id": call_id,
			"from_user_id": user_id,
			"candidate": candidate,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		logger.debug(f"ICE candidate forwarded from {user_id} to {target_user_id}")
		
		return {"status": "candidate_sent"}
	
	async def _handle_media_toggle(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle media enable/disable"""
		call_id = message.get("call_id")
		media_type = message.get("media_type")  # "audio" or "video"
		enabled = message.get("enabled", True)
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		if user_id not in self.active_calls[call_id]:
			return {"error": "User not in call"}
		
		# Update peer media state
		peer = self.active_calls[call_id][user_id]
		if media_type == "audio":
			peer.audio_enabled = enabled
		elif media_type == "video":
			peer.video_enabled = enabled
		
		# Broadcast media state change to all participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_media_toggle",
			"user_id": user_id,
			"media_type": media_type,
			"enabled": enabled,
			"timestamp": datetime.utcnow().isoformat()
		}, exclude_user=user_id)
		
		logger.info(f"Media toggle: {user_id} {media_type} {enabled} in call {call_id}")
		
		return {"status": "media_toggled", "media_type": media_type, "enabled": enabled}
	
	async def _handle_screen_share_start(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle screen sharing start"""
		call_id = message.get("call_id")
		share_type = message.get("share_type", "screen")  # "screen", "window", "tab"
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		if user_id not in self.active_calls[call_id]:
			return {"error": "User not in call"}
		
		# Update peer screen sharing state
		peer = self.active_calls[call_id][user_id]
		peer.screen_sharing = True
		
		# Stop any existing screen shares in the call (only one at a time)
		for other_peer in self.active_calls[call_id].values():
			if other_peer.user_id != user_id and other_peer.screen_sharing:
				other_peer.screen_sharing = False
				await self._send_to_user(other_peer.user_id, {
					"type": "webrtc_screen_share_stopped",
					"reason": "another_user_started_sharing",
					"timestamp": datetime.utcnow().isoformat()
				})
		
		# Broadcast screen share start to all participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_screen_share_started",
			"user_id": user_id,
			"share_type": share_type,
			"timestamp": datetime.utcnow().isoformat()
		}, exclude_user=user_id)
		
		logger.info(f"Screen sharing started: {user_id} in call {call_id}")
		
		return {"status": "screen_share_started", "share_type": share_type}
	
	async def _handle_screen_share_stop(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle screen sharing stop"""
		call_id = message.get("call_id")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		if user_id not in self.active_calls[call_id]:
			return {"error": "User not in call"}
		
		# Update peer screen sharing state
		peer = self.active_calls[call_id][user_id]
		peer.screen_sharing = False
		
		# Broadcast screen share stop to all participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_screen_share_stopped",
			"user_id": user_id,
			"timestamp": datetime.utcnow().isoformat()
		}, exclude_user=user_id)
		
		logger.info(f"Screen sharing stopped: {user_id} in call {call_id}")
		
		return {"status": "screen_share_stopped"}
	
	async def _handle_quality_change(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle video quality change"""
		call_id = message.get("call_id")
		quality_profile = message.get("quality_profile", "hd")  # "sd", "hd", "4k"
		bandwidth_limit = message.get("bandwidth_limit")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		if user_id not in self.active_calls[call_id]:
			return {"error": "User not in call"}
		
		# Update peer quality settings
		peer = self.active_calls[call_id][user_id]
		peer.quality_profile = quality_profile
		if bandwidth_limit:
			peer.bandwidth_limit = bandwidth_limit
		
		logger.info(f"Quality changed: {user_id} to {quality_profile} in call {call_id}")
		
		return {"status": "quality_changed", "quality_profile": quality_profile}
	
	async def _handle_recording_start(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle recording start"""
		call_id = message.get("call_id")
		recording_type = message.get("recording_type", "full_meeting")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		# Broadcast recording start to all participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_recording_started",
			"started_by": user_id,
			"recording_type": recording_type,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		logger.info(f"Recording started: {call_id} by {user_id}")
		
		return {"status": "recording_started", "recording_type": recording_type}
	
	async def _handle_recording_stop(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle recording stop"""
		call_id = message.get("call_id")
		
		if not call_id or call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		# Broadcast recording stop to all participants
		await self._broadcast_to_call(call_id, {
			"type": "webrtc_recording_stopped",
			"stopped_by": user_id,
			"timestamp": datetime.utcnow().isoformat()
		})
		
		logger.info(f"Recording stopped: {call_id} by {user_id}")
		
		return {"status": "recording_stopped"}
	
	async def _broadcast_to_call(self, call_id: str, message: Dict[str, Any], exclude_user: str = None):
		"""Broadcast message to all participants in a call"""
		if call_id not in self.active_calls:
			return
		
		for user_id in self.active_calls[call_id]:
			if user_id != exclude_user:
				await self._send_to_user(user_id, message)
	
	async def _broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: str = None):
		"""Broadcast message to all users in a session"""
		try:
			if websocket_manager:
				await websocket_manager.broadcast_to_session(session_id, message, exclude_user)
		except Exception as e:
			logger.error(f"Error broadcasting to session {session_id}: {e}")
	
	async def _send_to_user(self, user_id: str, message: Dict[str, Any]):
		"""Send message to a specific user"""
		try:
			if websocket_manager:
				await websocket_manager.send_to_user(user_id, message)
		except Exception as e:
			logger.error(f"Error sending message to user {user_id}: {e}")
	
	def get_call_statistics(self, call_id: str) -> Dict[str, Any]:
		"""Get statistics for a specific call"""
		if call_id not in self.active_calls:
			return {"error": "Call not found"}
		
		participants = self.active_calls[call_id]
		
		return {
			"call_id": call_id,
			"participant_count": len(participants),
			"participants": [
				{
					"user_id": peer.user_id,
					"connection_state": peer.connection_state,
					"audio_enabled": peer.audio_enabled,
					"video_enabled": peer.video_enabled,
					"screen_sharing": peer.screen_sharing,
					"quality_profile": peer.quality_profile,
					"joined_at": peer.joined_at.isoformat()
				}
				for peer in participants.values()
			],
			"screen_sharing_active": any(p.screen_sharing for p in participants.values())
		}
	
	def get_server_statistics(self) -> Dict[str, Any]:
		"""Get overall server statistics"""
		total_calls = len(self.active_calls)
		total_participants = sum(len(call) for call in self.active_calls.values())
		total_screen_shares = sum(
			sum(1 for peer in call.values() if peer.screen_sharing)
			for call in self.active_calls.values()
		)
		
		return {
			"active_calls": total_calls,
			"total_participants": total_participants,
			"active_screen_shares": total_screen_shares,
			"ice_servers_count": len(self.ice_servers),
			"timestamp": datetime.utcnow().isoformat()
		}


# Global WebRTC signaling server instance
webrtc_signaling = WebRTCSignalingServer()


# Integration function for WebSocket manager
async def handle_webrtc_message(user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
	"""Handle incoming WebRTC messages from WebSocket connections"""
	return await webrtc_signaling.handle_signaling_message(user_id, message)


if __name__ == "__main__":
	# Test the signaling server
	async def test_signaling():
		print("Testing WebRTC signaling server...")
		
		await webrtc_signaling.initialize()
		
		# Test call start
		result = await webrtc_signaling.handle_signaling_message("test_user_1", {
			"type": "webrtc_call_start",
			"call_id": "test_call_123",
			"session_id": "test_session_456"
		})
		print(f"Call start result: {result}")
		
		# Test call join
		result = await webrtc_signaling.handle_signaling_message("test_user_2", {
			"type": "webrtc_call_join",
			"call_id": "test_call_123",
			"session_id": "test_session_456"
		})
		print(f"Call join result: {result}")
		
		# Test statistics
		stats = webrtc_signaling.get_call_statistics("test_call_123")
		print(f"Call statistics: {stats}")
		
		server_stats = webrtc_signaling.get_server_statistics()
		print(f"Server statistics: {server_stats}")
		
		print("✅ WebRTC signaling server test completed")
	
	asyncio.run(test_signaling())