"""
RTMP Protocol Implementation for APG Real-Time Collaboration

Provides RTMP (Real-Time Messaging Protocol) communication for live streaming,
video broadcasting, and real-time media distribution capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import struct
import hashlib
import hmac
from pathlib import Path

try:
	import av
	from av import VideoFrame, AudioFrame
	import numpy as np
except ImportError:
	print("RTMP/Media dependencies not installed. Run: pip install av numpy")
	av = None
	VideoFrame = None
	AudioFrame = None
	np = None

logger = logging.getLogger(__name__)


class RTMPMessageType(Enum):
	"""RTMP message types"""
	SET_CHUNK_SIZE = 1
	ABORT_MESSAGE = 2
	ACKNOWLEDGEMENT = 3
	USER_CONTROL_MESSAGE = 4
	WINDOW_ACKNOWLEDGEMENT_SIZE = 5
	SET_PEER_BANDWIDTH = 6
	AUDIO = 8
	VIDEO = 9
	DATA_AMF3 = 15
	SHARED_OBJECT_AMF3 = 16
	COMMAND_AMF3 = 17
	DATA_AMF0 = 18
	SHARED_OBJECT_AMF0 = 19
	COMMAND_AMF0 = 20
	AGGREGATE = 22


class RTMPCommandType(Enum):
	"""RTMP command types"""
	CONNECT = "connect"
	CONNECT_RESULT = "connectResult"
	CALL = "call"
	CREATE_STREAM = "createStream"
	CREATE_STREAM_RESULT = "createStreamResult"
	PLAY = "play"
	PLAY_RESULT = "playResult"
	PUBLISH = "publish"
	PUBLISH_RESULT = "publishResult"
	DELETE_STREAM = "deleteStream"
	CLOSE_STREAM = "closeStream"
	RECEIVE_AUDIO = "receiveAudio"
	RECEIVE_VIDEO = "receiveVideo"
	ON_STATUS = "onStatus"
	ON_BW_DONE = "onBWDone"
	ON_FC_PUBLISH = "onFCPublish"
	ON_FC_UNPUBLISH = "onFCUnpublish"


class RTMPStreamState(Enum):
	"""RTMP stream states"""
	IDLE = "idle"
	CONNECTING = "connecting"
	CONNECTED = "connected"
	PUBLISHING = "publishing"
	PLAYING = "playing"
	PAUSED = "paused"
	STOPPED = "stopped"
	ERROR = "error"


@dataclass
class RTMPMessage:
	"""RTMP message structure"""
	message_type: RTMPMessageType
	timestamp: int
	stream_id: int
	data: bytes
	chunk_stream_id: int = 2
	message_length: Optional[int] = None
	created_at: datetime = field(default_factory=datetime.utcnow)
	message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"message_type": self.message_type.value,
			"timestamp": self.timestamp,
			"stream_id": self.stream_id,
			"chunk_stream_id": self.chunk_stream_id,
			"message_length": self.message_length or len(self.data),
			"data_size": len(self.data),
			"created_at": self.created_at.isoformat(),
			"message_id": self.message_id
		}


@dataclass
class RTMPStream:
	"""RTMP stream session"""
	stream_id: int
	stream_key: str
	app_name: str
	state: RTMPStreamState = RTMPStreamState.IDLE
	stream_type: str = "live"  # live, record, append
	client_id: Optional[str] = None
	publisher_id: Optional[str] = None
	subscribers: Set[str] = field(default_factory=set)
	metadata: Dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	
	# Media statistics
	video_frames_sent: int = 0
	audio_frames_sent: int = 0
	bytes_sent: int = 0
	bytes_received: int = 0
	bitrate_kbps: float = 0.0
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"stream_id": self.stream_id,
			"stream_key": self.stream_key,
			"app_name": self.app_name,
			"state": self.state.value,
			"stream_type": self.stream_type,
			"client_id": self.client_id,
			"publisher_id": self.publisher_id,
			"subscriber_count": len(self.subscribers),
			"metadata": self.metadata,
			"created_at": self.created_at.isoformat(),
			"last_activity": self.last_activity.isoformat(),
			"video_frames_sent": self.video_frames_sent,
			"audio_frames_sent": self.audio_frames_sent,
			"bytes_sent": self.bytes_sent,
			"bytes_received": self.bytes_received,
			"bitrate_kbps": self.bitrate_kbps
		}


@dataclass
class RTMPClient:
	"""RTMP client connection"""
	client_id: str
	connection_state: str = "connected"
	app_name: Optional[str] = None
	stream_key: Optional[str] = None
	client_type: str = "publisher"  # publisher, subscriber
	ip_address: Optional[str] = None
	user_agent: Optional[str] = None
	connected_at: datetime = field(default_factory=datetime.utcnow)
	last_activity: datetime = field(default_factory=datetime.utcnow)
	
	# Protocol state
	chunk_size: int = 128
	window_ack_size: int = 2500000
	peer_bandwidth: int = 2500000
	sequence_number: int = 0
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"client_id": self.client_id,
			"connection_state": self.connection_state,
			"app_name": self.app_name,
			"stream_key": self.stream_key,
			"client_type": self.client_type,
			"ip_address": self.ip_address,
			"user_agent": self.user_agent,
			"connected_at": self.connected_at.isoformat(),
			"last_activity": self.last_activity.isoformat(),
			"chunk_size": self.chunk_size,
			"window_ack_size": self.window_ack_size,
			"peer_bandwidth": self.peer_bandwidth
		}


class RTMPProtocolManager:
	"""Manages RTMP protocol for live streaming"""
	
	def __init__(self, host: str = "0.0.0.0", port: int = 1935):
		self.host = host
		self.port = port
		
		# Network components
		self.server: Optional[asyncio.Server] = None
		self.is_running = False
		
		# Client and stream management
		self.clients: Dict[str, RTMPClient] = {}
		self.streams: Dict[str, RTMPStream] = {}  # stream_key -> stream
		self.client_streams: Dict[str, str] = {}  # client_id -> stream_key
		
		# Message handling
		self.message_handlers: Dict[RTMPMessageType, List[Callable]] = {}
		self.command_handlers: Dict[str, Callable] = {}
		
		# Stream keys and authentication
		self.valid_stream_keys: Set[str] = set()
		self.stream_authentication: Dict[str, str] = {}  # stream_key -> secret
		
		# Configuration
		self.max_connections = 1000
		self.max_streams = 100
		self.chunk_size = 4096
		self.window_ack_size = 2500000
		
		# Statistics
		self.stats = {
			"connections": 0,
			"disconnections": 0,
			"streams_created": 0,
			"streams_terminated": 0,
			"total_bytes_sent": 0,
			"total_bytes_received": 0,
			"video_frames_processed": 0,
			"audio_frames_processed": 0,
			"uptime_start": None
		}
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize RTMP server"""
		try:
			# Create RTMP server
			self.server = await asyncio.start_server(
				self._handle_client_connection,
				self.host,
				self.port
			)
			
			self.is_running = True
			self.stats["uptime_start"] = datetime.utcnow()
			
			# Register default command handlers
			self._register_default_handlers()
			
			logger.info(f"RTMP server started on {self.host}:{self.port}")
			
			return {
				"status": "started",
				"address": f"rtmp://{self.host}:{self.port}",
				"max_connections": self.max_connections,
				"max_streams": self.max_streams
			}
			
		except Exception as e:
			logger.error(f"Failed to initialize RTMP server: {e}")
			return {"error": f"RTMP initialization failed: {str(e)}"}
	
	async def shutdown(self) -> Dict[str, Any]:
		"""Shutdown RTMP server"""
		try:
			# Terminate all streams
			for stream_key in list(self.streams.keys()):
				await self.terminate_stream(stream_key)
			
			# Close server
			if self.server:
				self.server.close()
				await self.server.wait_closed()
			
			self.is_running = False
			
			logger.info("RTMP server stopped")
			
			return {"status": "stopped"}
			
		except Exception as e:
			logger.error(f"Error stopping RTMP server: {e}")
			return {"error": f"Shutdown failed: {str(e)}"}
	
	async def _handle_client_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
		"""Handle new RTMP client connection"""
		client_id = str(uuid.uuid4())
		addr = writer.get_extra_info('peername')
		
		try:
			# Create client record
			client = RTMPClient(
				client_id=client_id,
				ip_address=addr[0] if addr else None
			)
			
			self.clients[client_id] = client
			self.stats["connections"] += 1
			
			logger.info(f"RTMP client connected: {client_id} from {addr}")
			
			# Handle RTMP handshake
			if await self._handle_rtmp_handshake(reader, writer, client):
				# Process RTMP messages
				await self._process_rtmp_messages(reader, writer, client)
			
		except Exception as e:
			logger.error(f"Error handling RTMP client {client_id}: {e}")
		
		finally:
			# Cleanup client
			await self._cleanup_client(client_id, writer)
	
	async def _handle_rtmp_handshake(self, reader: asyncio.StreamReader, 
									writer: asyncio.StreamWriter, client: RTMPClient) -> bool:
		"""Handle RTMP handshake process"""
		try:
			# C0: Version (1 byte)
			c0 = await reader.read(1)
			if not c0 or c0[0] != 3:
				logger.error("Invalid RTMP version in C0")
				return False
			
			# C1: Handshake (1536 bytes)
			c1 = await reader.read(1536)
			if len(c1) != 1536:
				logger.error("Invalid C1 handshake size")
				return False
			
			# Send S0: Version
			writer.write(b'\x03')
			
			# Send S1: Handshake
			s1 = self._create_handshake_packet()
			writer.write(s1)
			
			# Send S2: Echo of C1
			writer.write(c1)
			
			await writer.drain()
			
			# Receive C2: Echo of S1
			c2 = await reader.read(1536)
			if len(c2) != 1536:
				logger.error("Invalid C2 handshake size")
				return False
			
			logger.debug(f"RTMP handshake completed for client {client.client_id}")
			return True
			
		except Exception as e:
			logger.error(f"RTMP handshake failed: {e}")
			return False
	
	def _create_handshake_packet(self) -> bytes:
		"""Create RTMP handshake packet"""
		# Timestamp (4 bytes)
		timestamp = int(datetime.utcnow().timestamp())
		packet = struct.pack('>I', timestamp)
		
		# Zero (4 bytes)
		packet += b'\x00\x00\x00\x00'
		
		# Random data (1528 bytes)
		random_data = bytes([i % 256 for i in range(1528)])
		packet += random_data
		
		return packet
	
	async def _process_rtmp_messages(self, reader: asyncio.StreamReader,
									writer: asyncio.StreamWriter, client: RTMPClient):
		"""Process RTMP messages from client"""
		try:
			while self.is_running:
				# Read chunk header
				chunk_header = await self._read_chunk_header(reader)
				if not chunk_header:
					break
				
				# Read message data
				message_data = await reader.read(chunk_header["message_length"])
				if len(message_data) != chunk_header["message_length"]:
					break
				
				# Create RTMP message
				message = RTMPMessage(
					message_type=RTMPMessageType(chunk_header["message_type"]),
					timestamp=chunk_header["timestamp"],
					stream_id=chunk_header["stream_id"],
					data=message_data,
					chunk_stream_id=chunk_header["chunk_stream_id"]
				)
				
				# Process message
				await self._handle_rtmp_message(message, client, writer)
				
				# Update client activity
				client.last_activity = datetime.utcnow()
				
		except Exception as e:
			logger.error(f"Error processing RTMP messages: {e}")
	
	async def _read_chunk_header(self, reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
		"""Read RTMP chunk header"""
		try:
			# Basic header (1-3 bytes)
			basic_header = await reader.read(1)
			if not basic_header:
				return None
			
			chunk_stream_id = basic_header[0] & 0x3F
			header_type = (basic_header[0] >> 6) & 0x03
			
			# Extended chunk stream ID
			if chunk_stream_id == 0:
				ext_id = await reader.read(1)
				chunk_stream_id = ext_id[0] + 64
			elif chunk_stream_id == 1:
				ext_id = await reader.read(2)
				chunk_stream_id = struct.unpack('>H', ext_id)[0] + 64
			
			# Message header (0, 3, 7, or 11 bytes based on header type)
			header_sizes = {0: 11, 1: 7, 2: 3, 3: 0}
			header_size = header_sizes[header_type]
			
			if header_size > 0:
				message_header = await reader.read(header_size)
				if len(message_header) != header_size:
					return None
				
				# Parse message header
				if header_type == 0:  # Type 0: Full header
					timestamp = struct.unpack('>I', b'\x00' + message_header[0:3])[0]
					message_length = struct.unpack('>I', b'\x00' + message_header[3:6])[0]
					message_type = message_header[6]
					stream_id = struct.unpack('<I', message_header[7:11])[0]
				else:
					# For simplified implementation, use defaults
					timestamp = 0
					message_length = 1024  # Default size
					message_type = RTMPMessageType.COMMAND_AMF0.value
					stream_id = 0
			else:
				# Type 3: No header, use previous values
				timestamp = 0
				message_length = 1024
				message_type = RTMPMessageType.COMMAND_AMF0.value
				stream_id = 0
			
			return {
				"chunk_stream_id": chunk_stream_id,
				"header_type": header_type,
				"timestamp": timestamp,
				"message_length": message_length,
				"message_type": message_type,
				"stream_id": stream_id
			}
			
		except Exception as e:
			logger.error(f"Error reading chunk header: {e}")
			return None
	
	async def _handle_rtmp_message(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle RTMP message"""
		try:
			if message.message_type == RTMPMessageType.COMMAND_AMF0:
				await self._handle_command_message(message, client, writer)
			elif message.message_type == RTMPMessageType.VIDEO:
				await self._handle_video_message(message, client)
			elif message.message_type == RTMPMessageType.AUDIO:
				await self._handle_audio_message(message, client)
			elif message.message_type == RTMPMessageType.DATA_AMF0:
				await self._handle_data_message(message, client)
			elif message.message_type == RTMPMessageType.SET_CHUNK_SIZE:
				await self._handle_chunk_size_message(message, client)
			
			# Call registered handlers
			for handler in self.message_handlers.get(message.message_type, []):
				try:
					if asyncio.iscoroutinefunction(handler):
						await handler(message, client)
					else:
						handler(message, client)
				except Exception as e:
					logger.error(f"Error in message handler: {e}")
			
		except Exception as e:
			logger.error(f"Error handling RTMP message: {e}")
	
	async def _handle_command_message(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle RTMP command message"""
		try:
			# Parse AMF0 command (simplified)
			command_name = self._parse_amf0_string(message.data)
			
			if command_name == "connect":
				await self._handle_connect_command(message, client, writer)
			elif command_name == "createStream":
				await self._handle_create_stream_command(message, client, writer)
			elif command_name == "publish":
				await self._handle_publish_command(message, client, writer)
			elif command_name == "play":
				await self._handle_play_command(message, client, writer)
			elif command_name == "deleteStream":
				await self._handle_delete_stream_command(message, client, writer)
			
			logger.debug(f"Handled RTMP command: {command_name}")
			
		except Exception as e:
			logger.error(f"Error handling command message: {e}")
	
	async def _handle_connect_command(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle RTMP connect command"""
		try:
			# Parse connect parameters (simplified)
			client.app_name = "live"  # Default app name
			
			# Send connect response
			response = self._create_connect_response()
			await self._send_rtmp_message(writer, response)
			
			logger.info(f"Client {client.client_id} connected to app: {client.app_name}")
			
		except Exception as e:
			logger.error(f"Error handling connect command: {e}")
	
	async def _handle_create_stream_command(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle create stream command"""
		try:
			# Create new stream ID
			stream_id = len(self.streams) + 1
			
			# Send create stream response
			response = self._create_stream_response(stream_id)
			await self._send_rtmp_message(writer, response)
			
			logger.info(f"Created stream {stream_id} for client {client.client_id}")
			
		except Exception as e:
			logger.error(f"Error handling create stream command: {e}")
	
	async def _handle_publish_command(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle publish command"""
		try:
			# Parse stream key (simplified)
			stream_key = "default_stream"  # Would parse from AMF0 data
			
			# Create stream
			stream = RTMPStream(
				stream_id=message.stream_id,
				stream_key=stream_key,
				app_name=client.app_name or "live",
				state=RTMPStreamState.PUBLISHING,
				client_id=client.client_id,
				publisher_id=client.client_id
			)
			
			self.streams[stream_key] = stream
			self.client_streams[client.client_id] = stream_key
			client.stream_key = stream_key
			client.client_type = "publisher"
			
			# Update statistics
			self.stats["streams_created"] += 1
			
			# Send publish response
			response = self._create_publish_response()
			await self._send_rtmp_message(writer, response)
			
			logger.info(f"Client {client.client_id} started publishing stream: {stream_key}")
			
		except Exception as e:
			logger.error(f"Error handling publish command: {e}")
	
	async def _handle_play_command(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle play command"""
		try:
			# Parse stream key (simplified)
			stream_key = "default_stream"  # Would parse from AMF0 data
			
			if stream_key in self.streams:
				stream = self.streams[stream_key]
				stream.subscribers.add(client.client_id)
				
				client.stream_key = stream_key
				client.client_type = "subscriber"
				
				# Send play response
				response = self._create_play_response()
				await self._send_rtmp_message(writer, response)
				
				logger.info(f"Client {client.client_id} started playing stream: {stream_key}")
			else:
				logger.warning(f"Stream not found: {stream_key}")
			
		except Exception as e:
			logger.error(f"Error handling play command: {e}")
	
	async def _handle_delete_stream_command(self, message: RTMPMessage, client: RTMPClient, writer: asyncio.StreamWriter):
		"""Handle delete stream command"""
		try:
			if client.client_id in self.client_streams:
				stream_key = self.client_streams[client.client_id]
				await self.terminate_stream(stream_key)
			
			logger.info(f"Client {client.client_id} deleted stream")
			
		except Exception as e:
			logger.error(f"Error handling delete stream command: {e}")
	
	async def _handle_video_message(self, message: RTMPMessage, client: RTMPClient):
		"""Handle video data message"""
		try:
			if client.client_id in self.client_streams:
				stream_key = self.client_streams[client.client_id]
				stream = self.streams[stream_key]
				
				# Update stream statistics
				stream.video_frames_sent += 1
				stream.bytes_sent += len(message.data)
				stream.last_activity = datetime.utcnow()
				
				# Update global statistics
				self.stats["video_frames_processed"] += 1
				self.stats["total_bytes_received"] += len(message.data)
				
				# Broadcast to subscribers (simplified)
				await self._broadcast_to_subscribers(stream_key, message)
				
				logger.debug(f"Processed video frame for stream {stream_key}")
			
		except Exception as e:
			logger.error(f"Error handling video message: {e}")
	
	async def _handle_audio_message(self, message: RTMPMessage, client: RTMPClient):
		"""Handle audio data message"""
		try:
			if client.client_id in self.client_streams:
				stream_key = self.client_streams[client.client_id]
				stream = self.streams[stream_key]
				
				# Update stream statistics
				stream.audio_frames_sent += 1
				stream.bytes_sent += len(message.data)
				stream.last_activity = datetime.utcnow()
				
				# Update global statistics
				self.stats["audio_frames_processed"] += 1
				self.stats["total_bytes_received"] += len(message.data)
				
				# Broadcast to subscribers (simplified)
				await self._broadcast_to_subscribers(stream_key, message)
				
				logger.debug(f"Processed audio frame for stream {stream_key}")
			
		except Exception as e:
			logger.error(f"Error handling audio message: {e}")
	
	async def _handle_data_message(self, message: RTMPMessage, client: RTMPClient):
		"""Handle data message (metadata)"""
		try:
			if client.client_id in self.client_streams:
				stream_key = self.client_streams[client.client_id]
				stream = self.streams[stream_key]
				
				# Parse metadata (simplified)
				metadata = {"type": "onMetaData", "data": "parsed_metadata"}
				stream.metadata.update(metadata)
				
				logger.debug(f"Updated metadata for stream {stream_key}")
			
		except Exception as e:
			logger.error(f"Error handling data message: {e}")
	
	async def _handle_chunk_size_message(self, message: RTMPMessage, client: RTMPClient):
		"""Handle chunk size message"""
		try:
			if len(message.data) >= 4:
				new_chunk_size = struct.unpack('>I', message.data[:4])[0]
				client.chunk_size = new_chunk_size
				
				logger.debug(f"Updated chunk size to {new_chunk_size} for client {client.client_id}")
			
		except Exception as e:
			logger.error(f"Error handling chunk size message: {e}")
	
	async def _broadcast_to_subscribers(self, stream_key: str, message: RTMPMessage):
		"""Broadcast message to stream subscribers"""
		try:
			if stream_key in self.streams:
				stream = self.streams[stream_key]
				
				# In a real implementation, would send to actual subscriber connections
				# For now, just update statistics
				
				for subscriber_id in stream.subscribers:
					self.stats["total_bytes_sent"] += len(message.data)
				
				logger.debug(f"Broadcasted message to {len(stream.subscribers)} subscribers")
			
		except Exception as e:
			logger.error(f"Error broadcasting to subscribers: {e}")
	
	def _parse_amf0_string(self, data: bytes) -> str:
		"""Parse AMF0 string (simplified)"""
		try:
			if len(data) >= 3:
				# AMF0 string marker (0x02) + length (2 bytes) + string data
				if data[0] == 0x02:
					string_length = struct.unpack('>H', data[1:3])[0]
					if len(data) >= 3 + string_length:
						return data[3:3+string_length].decode('utf-8')
			
			return "unknown"
			
		except Exception as e:
			logger.error(f"Error parsing AMF0 string: {e}")
			return "unknown"
	
	def _create_connect_response(self) -> RTMPMessage:
		"""Create connect response message"""
		# Simplified connect response
		response_data = b'\x02\x00\x07connect\x00\x00\x00\x00\x00\x00\x00\x00\x00'
		
		return RTMPMessage(
			message_type=RTMPMessageType.COMMAND_AMF0,
			timestamp=0,
			stream_id=0,
			data=response_data
		)
	
	def _create_stream_response(self, stream_id: int) -> RTMPMessage:
		"""Create stream response message"""
		# Simplified stream response
		response_data = struct.pack('>I', stream_id)
		
		return RTMPMessage(
			message_type=RTMPMessageType.COMMAND_AMF0,
			timestamp=0,
			stream_id=0,
			data=response_data
		)
	
	def _create_publish_response(self) -> RTMPMessage:
		"""Create publish response message"""
		# Simplified publish response
		response_data = b'\x02\x00\x08onStatus\x00\x00\x00\x00\x00\x00\x00\x00\x00'
		
		return RTMPMessage(
			message_type=RTMPMessageType.COMMAND_AMF0,
			timestamp=0,
			stream_id=1,
			data=response_data
		)
	
	def _create_play_response(self) -> RTMPMessage:
		"""Create play response message"""
		# Simplified play response
		response_data = b'\x02\x00\x08onStatus\x00\x00\x00\x00\x00\x00\x00\x00\x00'
		
		return RTMPMessage(
			message_type=RTMPMessageType.COMMAND_AMF0,
			timestamp=0,
			stream_id=1,
			data=response_data
		)
	
	async def _send_rtmp_message(self, writer: asyncio.StreamWriter, message: RTMPMessage):
		"""Send RTMP message to client"""
		try:
			# Create chunk header (simplified)
			chunk_header = struct.pack('B', message.chunk_stream_id)
			
			# Create message header (Type 0 - full header)
			timestamp_bytes = struct.pack('>I', message.timestamp)[1:]  # 3 bytes
			length_bytes = struct.pack('>I', len(message.data))[1:]  # 3 bytes
			type_byte = struct.pack('B', message.message_type.value)
			stream_id_bytes = struct.pack('<I', message.stream_id)
			
			message_header = timestamp_bytes + length_bytes + type_byte + stream_id_bytes
			
			# Send complete message
			writer.write(chunk_header + message_header + message.data)
			await writer.drain()
			
		except Exception as e:
			logger.error(f"Error sending RTMP message: {e}")
	
	async def _cleanup_client(self, client_id: str, writer: asyncio.StreamWriter):
		"""Cleanup client connection"""
		try:
			# Remove from stream subscribers
			if client_id in self.client_streams:
				stream_key = self.client_streams[client_id]
				if stream_key in self.streams:
					stream = self.streams[stream_key]
					stream.subscribers.discard(client_id)
					
					# If this was the publisher, terminate stream
					if stream.publisher_id == client_id:
						await self.terminate_stream(stream_key)
				
				del self.client_streams[client_id]
			
			# Remove client record
			if client_id in self.clients:
				del self.clients[client_id]
			
			# Close writer
			writer.close()
			await writer.wait_closed()
			
			# Update statistics
			self.stats["disconnections"] += 1
			
			logger.info(f"Cleaned up RTMP client: {client_id}")
			
		except Exception as e:
			logger.error(f"Error cleaning up client {client_id}: {e}")
	
	def _register_default_handlers(self):
		"""Register default RTMP message handlers"""
		# Register command handlers
		self.command_handlers[RTMPCommandType.CONNECT.value] = self._handle_connect_command
		self.command_handlers[RTMPCommandType.CREATE_STREAM.value] = self._handle_create_stream_command
		self.command_handlers[RTMPCommandType.PUBLISH.value] = self._handle_publish_command
		self.command_handlers[RTMPCommandType.PLAY.value] = self._handle_play_command
	
	async def create_stream(self, stream_key: str, app_name: str = "live") -> Dict[str, Any]:
		"""Create new RTMP stream"""
		try:
			if stream_key in self.streams:
				return {"error": "Stream already exists"}
			
			stream_id = len(self.streams) + 1
			
			stream = RTMPStream(
				stream_id=stream_id,
				stream_key=stream_key,
				app_name=app_name,
				state=RTMPStreamState.IDLE
			)
			
			self.streams[stream_key] = stream
			self.valid_stream_keys.add(stream_key)
			
			logger.info(f"Created RTMP stream: {stream_key}")
			
			return {
				"status": "created",
				"stream_key": stream_key,
				"stream_id": stream_id,
				"app_name": app_name
			}
			
		except Exception as e:
			logger.error(f"Failed to create RTMP stream: {e}")
			return {"error": f"Stream creation failed: {str(e)}"}
	
	async def terminate_stream(self, stream_key: str) -> Dict[str, Any]:
		"""Terminate RTMP stream"""
		try:
			if stream_key not in self.streams:
				return {"error": "Stream not found"}
			
			stream = self.streams[stream_key]
			stream.state = RTMPStreamState.STOPPED
			
			# Remove stream
			del self.streams[stream_key]
			self.valid_stream_keys.discard(stream_key)
			
			# Update statistics
			self.stats["streams_terminated"] += 1
			
			logger.info(f"Terminated RTMP stream: {stream_key}")
			
			return {
				"status": "terminated",
				"stream_key": stream_key
			}
			
		except Exception as e:
			logger.error(f"Failed to terminate RTMP stream: {e}")
			return {"error": f"Stream termination failed: {str(e)}"}
	
	def register_message_handler(self, message_type: RTMPMessageType, handler: Callable):
		"""Register RTMP message handler"""
		if message_type not in self.message_handlers:
			self.message_handlers[message_type] = []
		self.message_handlers[message_type].append(handler)
		
		logger.info(f"Registered RTMP message handler for type: {message_type.name}")
	
	def add_stream_key(self, stream_key: str, secret: str = None):
		"""Add valid stream key"""
		self.valid_stream_keys.add(stream_key)
		if secret:
			self.stream_authentication[stream_key] = secret
		
		logger.info(f"Added stream key: {stream_key}")
	
	def remove_stream_key(self, stream_key: str):
		"""Remove stream key"""
		self.valid_stream_keys.discard(stream_key)
		self.stream_authentication.pop(stream_key, None)
		
		logger.info(f"Removed stream key: {stream_key}")
	
	def get_active_streams(self) -> List[Dict[str, Any]]:
		"""Get list of active streams"""
		return [stream.to_dict() for stream in self.streams.values()]
	
	def get_connected_clients(self) -> List[Dict[str, Any]]:
		"""Get list of connected clients"""
		return [client.to_dict() for client in self.clients.values()]
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get RTMP protocol statistics"""
		uptime_seconds = 0
		if self.stats["uptime_start"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["uptime_start"]).total_seconds())
		
		return {
			**self.stats,
			"is_running": self.is_running,
			"address": f"rtmp://{self.host}:{self.port}",
			"connected_clients": len(self.clients),
			"active_streams": len(self.streams),
			"valid_stream_keys": len(self.valid_stream_keys),
			"max_connections": self.max_connections,
			"max_streams": self.max_streams,
			"uptime_seconds": uptime_seconds,
			"uptime_start": self.stats["uptime_start"].isoformat() if self.stats["uptime_start"] else None
		}


# Global RTMP manager instance
rtmp_protocol_manager = None


async def initialize_rtmp_protocol(host: str = "0.0.0.0", port: int = 1935) -> Dict[str, Any]:
	"""Initialize global RTMP protocol manager"""
	global rtmp_protocol_manager
	
	rtmp_protocol_manager = RTMPProtocolManager(host=host, port=port)
	
	result = await rtmp_protocol_manager.initialize()
	
	return result


def get_rtmp_manager() -> Optional[RTMPProtocolManager]:
	"""Get global RTMP protocol manager"""
	return rtmp_protocol_manager


if __name__ == "__main__":
	# Test RTMP protocol implementation
	async def test_rtmp():
		print("Testing RTMP protocol implementation...")
		
		# Initialize RTMP manager
		result = await initialize_rtmp_protocol()
		print(f"RTMP initialization result: {result}")
		
		if result.get("status") == "started":
			manager = get_rtmp_manager()
			
			# Create test stream
			stream_result = await manager.create_stream("test_stream", "live")
			print(f"Stream creation result: {stream_result}")
			
			# Add stream key
			manager.add_stream_key("test_stream", "secret123")
			
			# Get statistics
			stats = manager.get_statistics()
			print(f"RTMP statistics: {stats}")
			
			# Keep server running for a bit
			print("RTMP server running. Connect with OBS or similar to test...")
			await asyncio.sleep(5)
			
			# Shutdown
			shutdown_result = await manager.shutdown()
			print(f"Shutdown result: {shutdown_result}")
		
		print("✅ RTMP protocol test completed")
	
	asyncio.run(test_rtmp())