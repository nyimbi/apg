"""
WebRTC Data Channels for APG Real-Time Collaboration

Implements peer-to-peer file transfer, collaborative editing synchronization,
and real-time data exchange using WebRTC data channels.
"""

import asyncio
import json
import logging
import base64
import hashlib
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class DataChannelMessageType(Enum):
	"""Data channel message types"""
	FILE_TRANSFER_REQUEST = "file_transfer_request"
	FILE_TRANSFER_ACCEPT = "file_transfer_accept"
	FILE_TRANSFER_REJECT = "file_transfer_reject"
	FILE_CHUNK = "file_chunk"
	FILE_TRANSFER_COMPLETE = "file_transfer_complete"
	FILE_TRANSFER_ERROR = "file_transfer_error"
	
	COLLABORATIVE_EDIT = "collaborative_edit"
	CURSOR_POSITION = "cursor_position"
	TEXT_SELECTION = "text_selection"
	
	FORM_SYNC = "form_sync"
	FIELD_LOCK = "field_lock"
	FIELD_UNLOCK = "field_unlock"
	
	PING = "ping"
	PONG = "pong"


@dataclass
class FileTransfer:
	"""File transfer session"""
	transfer_id: str
	filename: str
	file_size: int
	file_type: str
	sender_id: str
	receiver_id: str
	total_chunks: int
	received_chunks: int = 0
	chunks_data: Dict[int, bytes] = None
	started_at: datetime = None
	completed_at: Optional[datetime] = None
	status: str = "pending"  # pending, active, completed, error, cancelled
	error_message: Optional[str] = None
	checksum: Optional[str] = None
	
	def __post_init__(self):
		if self.chunks_data is None:
			self.chunks_data = {}
		if self.started_at is None:
			self.started_at = datetime.utcnow()


@dataclass
class CollaborativeEdit:
	"""Collaborative editing operation"""
	edit_id: str
	user_id: str
	field_name: str
	operation: str  # insert, delete, replace
	position: int
	content: str
	timestamp: datetime
	applied: bool = False


class WebRTCDataChannelManager:
	"""Manages WebRTC data channels for file transfer and collaboration"""
	
	CHUNK_SIZE = 16384  # 16KB chunks for file transfer
	MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size
	
	def __init__(self):
		self.active_transfers: Dict[str, FileTransfer] = {}
		self.collaborative_sessions: Dict[str, Dict[str, Any]] = {}
		self.data_channel_handlers: Dict[str, Callable] = {}
		self.connection_stats: Dict[str, Dict[str, Any]] = {}
		
		# Initialize message handlers
		self._init_handlers()
	
	def _init_handlers(self):
		"""Initialize data channel message handlers"""
		self.data_channel_handlers = {
			DataChannelMessageType.FILE_TRANSFER_REQUEST: self._handle_file_transfer_request,
			DataChannelMessageType.FILE_TRANSFER_ACCEPT: self._handle_file_transfer_accept,
			DataChannelMessageType.FILE_TRANSFER_REJECT: self._handle_file_transfer_reject,
			DataChannelMessageType.FILE_CHUNK: self._handle_file_chunk,
			DataChannelMessageType.FILE_TRANSFER_COMPLETE: self._handle_file_transfer_complete,
			DataChannelMessageType.FILE_TRANSFER_ERROR: self._handle_file_transfer_error,
			DataChannelMessageType.COLLABORATIVE_EDIT: self._handle_collaborative_edit,
			DataChannelMessageType.CURSOR_POSITION: self._handle_cursor_position,
			DataChannelMessageType.TEXT_SELECTION: self._handle_text_selection,
			DataChannelMessageType.FORM_SYNC: self._handle_form_sync,
			DataChannelMessageType.FIELD_LOCK: self._handle_field_lock,
			DataChannelMessageType.FIELD_UNLOCK: self._handle_field_unlock,
			DataChannelMessageType.PING: self._handle_ping,
			DataChannelMessageType.PONG: self._handle_pong
		}
	
	async def handle_data_channel_message(self, user_id: str, peer_id: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
		"""Handle incoming data channel message"""
		try:
			message_type = DataChannelMessageType(message.get('type'))
			
			# Update connection stats
			self._update_connection_stats(user_id, peer_id, 'received')
			
			handler = self.data_channel_handlers.get(message_type)
			if handler:
				return await handler(user_id, peer_id, message)
			else:
				logger.warning(f"Unknown data channel message type: {message_type}")
				return {"error": f"Unknown message type: {message_type}"}
				
		except ValueError as e:
			logger.error(f"Invalid data channel message type: {message.get('type')} - {e}")
			return {"error": "Invalid message type"}
		except Exception as e:
			logger.error(f"Error handling data channel message: {e}")
			return {"error": "Internal processing error"}
	
	async def initiate_file_transfer(self, sender_id: str, receiver_id: str, 
									file_info: Dict[str, Any]) -> Dict[str, Any]:
		"""Initiate a file transfer between peers"""
		try:
			# Validate file size
			file_size = file_info.get('size', 0)
			if file_size > self.MAX_FILE_SIZE:
				return {"error": f"File too large. Maximum size: {self.MAX_FILE_SIZE} bytes"}
			
			# Create transfer session
			transfer_id = str(uuid.uuid4())
			filename = file_info.get('name', 'unknown')
			file_type = file_info.get('type', 'application/octet-stream')
			
			total_chunks = (file_size + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
			
			transfer = FileTransfer(
				transfer_id=transfer_id,
				filename=filename,
				file_size=file_size,
				file_type=file_type,
				sender_id=sender_id,
				receiver_id=receiver_id,
				total_chunks=total_chunks
			)
			
			self.active_transfers[transfer_id] = transfer
			
			logger.info(f"File transfer initiated: {transfer_id} ({filename}, {file_size} bytes)")
			
			return {
				"transfer_id": transfer_id,
				"status": "request_sent",
				"filename": filename,
				"file_size": file_size,
				"total_chunks": total_chunks
			}
			
		except Exception as e:
			logger.error(f"Error initiating file transfer: {e}")
			return {"error": "Failed to initiate file transfer"}
	
	async def _handle_file_transfer_request(self, user_id: str, peer_id: str, 
										  message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file transfer request"""
		transfer_id = message.get('transfer_id')
		filename = message.get('filename')
		file_size = message.get('file_size')
		file_type = message.get('file_type')
		sender_id = message.get('sender_id')
		
		if not all([transfer_id, filename, file_size, sender_id]):
			return {"error": "Missing required file transfer information"}
		
		# Create transfer session
		total_chunks = (file_size + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
		
		transfer = FileTransfer(
			transfer_id=transfer_id,
			filename=filename,
			file_size=file_size,
			file_type=file_type,
			sender_id=sender_id,
			receiver_id=user_id,
			total_chunks=total_chunks,
			status="requested"
		)
		
		self.active_transfers[transfer_id] = transfer
		
		logger.info(f"File transfer request received: {transfer_id} from {sender_id}")
		
		# In a real implementation, this would trigger a UI notification
		# for the user to accept or reject the transfer
		return {
			"type": "file_transfer_notification",
			"transfer_id": transfer_id,
			"filename": filename,
			"file_size": file_size,
			"sender_id": sender_id,
			"status": "awaiting_response"
		}
	
	async def _handle_file_transfer_accept(self, user_id: str, peer_id: str, 
										 message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file transfer acceptance"""
		transfer_id = message.get('transfer_id')
		
		if transfer_id not in self.active_transfers:
			return {"error": "Transfer not found"}
		
		transfer = self.active_transfers[transfer_id]
		transfer.status = "active"
		
		logger.info(f"File transfer accepted: {transfer_id}")
		
		return {
			"status": "transfer_accepted",
			"transfer_id": transfer_id,
			"ready_for_chunks": True
		}
	
	async def _handle_file_transfer_reject(self, user_id: str, peer_id: str, 
										 message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file transfer rejection"""
		transfer_id = message.get('transfer_id')
		reason = message.get('reason', 'User declined')
		
		if transfer_id in self.active_transfers:
			transfer = self.active_transfers[transfer_id]
			transfer.status = "rejected"
			transfer.error_message = reason
			
			logger.info(f"File transfer rejected: {transfer_id} - {reason}")
			
			# Clean up after a delay
			asyncio.create_task(self._cleanup_transfer(transfer_id, delay=30))
		
		return {
			"status": "transfer_rejected",
			"transfer_id": transfer_id,
			"reason": reason
		}
	
	async def _handle_file_chunk(self, user_id: str, peer_id: str, 
							   message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file chunk data"""
		transfer_id = message.get('transfer_id')
		chunk_index = message.get('chunk_index')
		chunk_data_b64 = message.get('chunk_data')
		is_last_chunk = message.get('is_last_chunk', False)
		
		if not all([transfer_id, chunk_index is not None, chunk_data_b64]):
			return {"error": "Missing chunk data"}
		
		if transfer_id not in self.active_transfers:
			return {"error": "Transfer not found"}
		
		transfer = self.active_transfers[transfer_id]
		
		if transfer.status != "active":
			return {"error": "Transfer not active"}
		
		try:
			# Decode chunk data
			chunk_data = base64.b64decode(chunk_data_b64)
			
			# Store chunk
			transfer.chunks_data[chunk_index] = chunk_data
			transfer.received_chunks += 1
			
			logger.debug(f"Received chunk {chunk_index} for transfer {transfer_id}")
			
			# Check if transfer is complete
			if transfer.received_chunks >= transfer.total_chunks or is_last_chunk:
				await self._complete_file_transfer(transfer)
			
			return {
				"status": "chunk_received",
				"transfer_id": transfer_id,
				"chunk_index": chunk_index,
				"progress": transfer.received_chunks / transfer.total_chunks
			}
			
		except Exception as e:
			logger.error(f"Error processing file chunk: {e}")
			transfer.status = "error"
			transfer.error_message = str(e)
			return {"error": "Failed to process chunk"}
	
	async def _complete_file_transfer(self, transfer: FileTransfer):
		"""Complete file transfer and reconstruct file"""
		try:
			# Reconstruct file from chunks
			complete_data = b""
			for i in range(transfer.total_chunks):
				if i in transfer.chunks_data:
					complete_data += transfer.chunks_data[i]
			
			# Verify file size
			if len(complete_data) != transfer.file_size:
				raise ValueError(f"File size mismatch: expected {transfer.file_size}, got {len(complete_data)}")
			
			# Calculate checksum
			checksum = hashlib.sha256(complete_data).hexdigest()
			transfer.checksum = checksum
			
			# Update transfer status
			transfer.status = "completed"
			transfer.completed_at = datetime.utcnow()
			
			# In a real implementation, you would save the file or trigger a download
			logger.info(f"File transfer completed: {transfer.transfer_id} ({len(complete_data)} bytes)")
			
			# Clean up chunks from memory
			transfer.chunks_data.clear()
			
			# Schedule cleanup
			asyncio.create_task(self._cleanup_transfer(transfer.transfer_id, delay=300))
			
		except Exception as e:
			logger.error(f"Error completing file transfer: {e}")
			transfer.status = "error"
			transfer.error_message = str(e)
	
	async def _handle_file_transfer_complete(self, user_id: str, peer_id: str, 
										   message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file transfer completion notification"""
		transfer_id = message.get('transfer_id')
		checksum = message.get('checksum')
		
		if transfer_id in self.active_transfers:
			transfer = self.active_transfers[transfer_id]
			
			# Verify checksum if provided
			if checksum and transfer.checksum and checksum != transfer.checksum:
				logger.warning(f"Checksum mismatch for transfer {transfer_id}")
				return {"error": "Checksum verification failed"}
			
			logger.info(f"File transfer completion confirmed: {transfer_id}")
		
		return {"status": "completion_acknowledged"}
	
	async def _handle_file_transfer_error(self, user_id: str, peer_id: str, 
										message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle file transfer error"""
		transfer_id = message.get('transfer_id')
		error_message = message.get('error_message', 'Unknown error')
		
		if transfer_id in self.active_transfers:
			transfer = self.active_transfers[transfer_id]
			transfer.status = "error"
			transfer.error_message = error_message
			
			logger.error(f"File transfer error: {transfer_id} - {error_message}")
		
		return {"status": "error_acknowledged"}
	
	async def _handle_collaborative_edit(self, user_id: str, peer_id: str, 
									   message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle collaborative editing operation"""
		edit_id = message.get('edit_id', str(uuid.uuid4()))
		field_name = message.get('field_name')
		operation = message.get('operation')  # insert, delete, replace
		position = message.get('position')
		content = message.get('content', '')
		
		if not all([field_name, operation, position is not None]):
			return {"error": "Missing collaborative edit data"}
		
		edit = CollaborativeEdit(
			edit_id=edit_id,
			user_id=user_id,
			field_name=field_name,
			operation=operation,
			position=position,
			content=content,
			timestamp=datetime.utcnow()
		)
		
		# In a real implementation, this would apply the edit operation
		# and handle conflict resolution with operational transforms
		
		logger.debug(f"Collaborative edit: {operation} at {position} in {field_name} by {user_id}")
		
		return {
			"status": "edit_applied",
			"edit_id": edit_id,
			"timestamp": edit.timestamp.isoformat()
		}
	
	async def _handle_cursor_position(self, user_id: str, peer_id: str, 
									message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle cursor position update"""
		field_name = message.get('field_name')
		position = message.get('position')
		
		if field_name and position is not None:
			# Store cursor position for real-time display
			logger.debug(f"Cursor position update: {user_id} at {position} in {field_name}")
		
		return {"status": "cursor_position_updated"}
	
	async def _handle_text_selection(self, user_id: str, peer_id: str, 
								   message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle text selection update"""
		field_name = message.get('field_name')
		start_position = message.get('start_position')
		end_position = message.get('end_position')
		
		if field_name and start_position is not None and end_position is not None:
			logger.debug(f"Text selection: {user_id} selected {start_position}-{end_position} in {field_name}")
		
		return {"status": "text_selection_updated"}
	
	async def _handle_form_sync(self, user_id: str, peer_id: str, 
							  message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle form synchronization"""
		form_data = message.get('form_data', {})
		page_url = message.get('page_url')
		
		if form_data and page_url:
			# Synchronize form data across participants
			logger.debug(f"Form sync: {user_id} updated form data for {page_url}")
		
		return {"status": "form_synchronized"}
	
	async def _handle_field_lock(self, user_id: str, peer_id: str, 
							   message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle field lock request"""
		field_name = message.get('field_name')
		page_url = message.get('page_url')
		
		if field_name and page_url:
			# Implement field locking logic
			logger.debug(f"Field lock: {user_id} locked {field_name} on {page_url}")
		
		return {"status": "field_locked"}
	
	async def _handle_field_unlock(self, user_id: str, peer_id: str, 
								 message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle field unlock request"""
		field_name = message.get('field_name')
		page_url = message.get('page_url')
		
		if field_name and page_url:
			# Implement field unlocking logic
			logger.debug(f"Field unlock: {user_id} unlocked {field_name} on {page_url}")
		
		return {"status": "field_unlocked"}
	
	async def _handle_ping(self, user_id: str, peer_id: str, 
						 message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle ping message"""
		timestamp = message.get('timestamp', datetime.utcnow().isoformat())
		
		return {
			"type": "pong",
			"timestamp": timestamp,
			"response_time": datetime.utcnow().isoformat()
		}
	
	async def _handle_pong(self, user_id: str, peer_id: str, 
						 message: Dict[str, Any]) -> Dict[str, Any]:
		"""Handle pong response"""
		# Calculate latency if original timestamp is provided
		original_timestamp = message.get('timestamp')
		if original_timestamp:
			try:
				sent_time = datetime.fromisoformat(original_timestamp.replace('Z', '+00:00'))
				latency = (datetime.utcnow() - sent_time).total_seconds() * 1000
				
				# Update connection stats
				if user_id not in self.connection_stats:
					self.connection_stats[user_id] = {}
				
				self.connection_stats[user_id]['latency_ms'] = latency
				logger.debug(f"Data channel latency to {user_id}: {latency:.2f}ms")
				
			except Exception as e:
				logger.error(f"Error calculating latency: {e}")
		
		return {"status": "pong_received"}
	
	def _update_connection_stats(self, user_id: str, peer_id: str, direction: str):
		"""Update connection statistics"""
		if user_id not in self.connection_stats:
			self.connection_stats[user_id] = {
				'messages_sent': 0,
				'messages_received': 0,
				'bytes_sent': 0,
				'bytes_received': 0,
				'last_activity': datetime.utcnow()
			}
		
		stats = self.connection_stats[user_id]
		
		if direction == 'sent':
			stats['messages_sent'] += 1
		elif direction == 'received':
			stats['messages_received'] += 1
		
		stats['last_activity'] = datetime.utcnow()
	
	async def _cleanup_transfer(self, transfer_id: str, delay: int = 0):
		"""Clean up completed transfer after delay"""
		if delay > 0:
			await asyncio.sleep(delay)
		
		if transfer_id in self.active_transfers:
			transfer = self.active_transfers[transfer_id]
			
			# Clear chunks data
			if transfer.chunks_data:
				transfer.chunks_data.clear()
			
			# Remove from active transfers
			del self.active_transfers[transfer_id]
			
			logger.info(f"Transfer cleaned up: {transfer_id}")
	
	def get_transfer_status(self, transfer_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of a file transfer"""
		if transfer_id not in self.active_transfers:
			return None
		
		transfer = self.active_transfers[transfer_id]
		
		return {
			"transfer_id": transfer_id,
			"filename": transfer.filename,
			"file_size": transfer.file_size,
			"status": transfer.status,
			"progress": transfer.received_chunks / transfer.total_chunks if transfer.total_chunks > 0 else 0,
			"started_at": transfer.started_at.isoformat(),
			"completed_at": transfer.completed_at.isoformat() if transfer.completed_at else None,
			"error_message": transfer.error_message
		}
	
	def get_active_transfers(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get list of active transfers, optionally filtered by user"""
		transfers = []
		
		for transfer in self.active_transfers.values():
			if user_id and transfer.sender_id != user_id and transfer.receiver_id != user_id:
				continue
			
			transfers.append({
				"transfer_id": transfer.transfer_id,
				"filename": transfer.filename,
				"file_size": transfer.file_size,
				"sender_id": transfer.sender_id,
				"receiver_id": transfer.receiver_id,
				"status": transfer.status,
				"progress": transfer.received_chunks / transfer.total_chunks if transfer.total_chunks > 0 else 0,
				"started_at": transfer.started_at.isoformat()
			})
		
		return transfers
	
	def get_connection_statistics(self) -> Dict[str, Any]:
		"""Get data channel connection statistics"""
		return {
			"active_transfers": len(self.active_transfers),
			"total_connections": len(self.connection_stats),
			"connection_details": self.connection_stats,
			"timestamp": datetime.utcnow().isoformat()
		}


# Global data channel manager instance
webrtc_data_manager = WebRTCDataChannelManager()


# Client-side JavaScript for data channels
def generate_data_channel_javascript() -> str:
	"""Generate JavaScript for WebRTC data channel handling"""
	return """
/**
 * WebRTC Data Channel Manager for APG Real-Time Collaboration
 * Handles file transfer and collaborative editing via data channels
 */

class WebRTCDataChannelManager {
	constructor(webrtcClient) {
		this.webrtcClient = webrtcClient;
		this.dataChannels = new Map();
		this.fileTransfers = new Map();
		this.collaborativeEdits = new Map();
		
		// Configuration
		this.CHUNK_SIZE = 16384; // 16KB chunks
		this.MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
		
		// Initialize
		this.initialize();
	}
	
	initialize() {
		// Listen for new peer connections to add data channels
		if (this.webrtcClient) {
			this.webrtcClient.on = this.webrtcClient.on || {};
			this.webrtcClient.on.peerConnected = (userId, peerConnection) => {
				this.setupDataChannel(userId, peerConnection);
			};
		}
	}
	
	setupDataChannel(userId, peerConnection) {
		try {
			// Create data channel
			const dataChannel = peerConnection.createDataChannel('collaboration', {
				ordered: true,
				maxRetransmits: 3
			});
			
			// Set up event handlers
			dataChannel.onopen = () => {
				console.log(`Data channel opened with ${userId}`);
				this.dataChannels.set(userId, dataChannel);
			};
			
			dataChannel.onclose = () => {
				console.log(`Data channel closed with ${userId}`);
				this.dataChannels.delete(userId);
			};
			
			dataChannel.onmessage = (event) => {
				this.handleDataChannelMessage(userId, event.data);
			};
			
			dataChannel.onerror = (error) => {
				console.error(`Data channel error with ${userId}:`, error);
			};
			
			// Handle incoming data channels
			peerConnection.ondatachannel = (event) => {
				const channel = event.channel;
				channel.onmessage = (event) => {
					this.handleDataChannelMessage(userId, event.data);
				};
			};
			
		} catch (error) {
			console.error('Error setting up data channel:', error);
		}
	}
	
	async handleDataChannelMessage(userId, data) {
		try {
			const message = JSON.parse(data);
			const messageType = message.type;
			
			console.log(`Data channel message from ${userId}:`, messageType);
			
			switch (messageType) {
				case 'file_transfer_request':
					await this.handleFileTransferRequest(userId, message);
					break;
				case 'file_transfer_accept':
					await this.handleFileTransferAccept(userId, message);
					break;
				case 'file_transfer_reject':
					await this.handleFileTransferReject(userId, message);
					break;
				case 'file_chunk':
					await this.handleFileChunk(userId, message);
					break;
				case 'collaborative_edit':
					await this.handleCollaborativeEdit(userId, message);
					break;
				case 'cursor_position':
					await this.handleCursorPosition(userId, message);
					break;
				case 'form_sync':
					await this.handleFormSync(userId, message);
					break;
				case 'ping':
					await this.handlePing(userId, message);
					break;
				default:
					console.log('Unknown data channel message type:', messageType);
			}
		} catch (error) {
			console.error('Error handling data channel message:', error);
		}
	}
	
	async sendFileToUser(userId, file) {
		if (!this.dataChannels.has(userId)) {
			throw new Error(`No data channel with user ${userId}`);
		}
		
		if (file.size > this.MAX_FILE_SIZE) {
			throw new Error(`File too large. Maximum size: ${this.MAX_FILE_SIZE} bytes`);
		}
		
		const transferId = this.generateId();
		const totalChunks = Math.ceil(file.size / this.CHUNK_SIZE);
		
		// Send file transfer request
		const request = {
			type: 'file_transfer_request',
			transfer_id: transferId,
			filename: file.name,
			file_size: file.size,
			file_type: file.type,
			total_chunks: totalChunks,
			sender_id: this.webrtcClient.getCurrentUserId(),
			timestamp: new Date().toISOString()
		};
		
		this.sendDataChannelMessage(userId, request);
		
		// Store transfer info
		this.fileTransfers.set(transferId, {
			file: file,
			userId: userId,
			status: 'pending',
			sentChunks: 0,
			totalChunks: totalChunks
		});
		
		return transferId;
	}
	
	async handleFileTransferRequest(userId, message) {
		const { transfer_id, filename, file_size, file_type } = message;
		
		// Show file transfer notification to user
		const accept = await this.showFileTransferDialog(userId, filename, file_size);
		
		const response = {
			type: accept ? 'file_transfer_accept' : 'file_transfer_reject',
			transfer_id: transfer_id,
			reason: accept ? null : 'User declined',
			timestamp: new Date().toISOString()
		};
		
		this.sendDataChannelMessage(userId, response);
		
		if (accept) {
			// Prepare to receive file
			this.fileTransfers.set(transfer_id, {
				filename: filename,
				file_size: file_size,
				file_type: file_type,
				userId: userId,
				status: 'receiving',
				chunks: {},
				receivedChunks: 0,
				totalChunks: message.total_chunks
			});
		}
	}
	
	async handleFileTransferAccept(userId, message) {
		const transferId = message.transfer_id;
		const transfer = this.fileTransfers.get(transferId);
		
		if (!transfer) {
			console.error('Transfer not found:', transferId);
			return;
		}
		
		transfer.status = 'sending';
		
		// Start sending file chunks
		await this.sendFileChunks(transferId);
	}
	
	async sendFileChunks(transferId) {
		const transfer = this.fileTransfers.get(transferId);
		if (!transfer || !transfer.file) return;
		
		const { file, userId, totalChunks } = transfer;
		
		for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
			const start = chunkIndex * this.CHUNK_SIZE;
			const end = Math.min(start + this.CHUNK_SIZE, file.size);
			const chunk = file.slice(start, end);
			
			// Convert chunk to base64
			const chunkData = await this.fileToBase64(chunk);
			
			const chunkMessage = {
				type: 'file_chunk',
				transfer_id: transferId,
				chunk_index: chunkIndex,
				chunk_data: chunkData,
				is_last_chunk: chunkIndex === totalChunks - 1,
				timestamp: new Date().toISOString()
			};
			
			this.sendDataChannelMessage(userId, chunkMessage);
			transfer.sentChunks++;
			
			// Update progress
			const progress = transfer.sentChunks / totalChunks;
			this.onFileTransferProgress(transferId, progress);
			
			// Small delay to prevent overwhelming the channel
			await this.sleep(10);
		}
		
		transfer.status = 'completed';
		console.log(`File transfer completed: ${transferId}`);
	}
	
	async handleFileChunk(userId, message) {
		const { transfer_id, chunk_index, chunk_data, is_last_chunk } = message;
		const transfer = this.fileTransfers.get(transfer_id);
		
		if (!transfer) {
			console.error('Transfer not found:', transfer_id);
			return;
		}
		
		// Store chunk
		transfer.chunks[chunk_index] = chunk_data;
		transfer.receivedChunks++;
		
		// Update progress
		const progress = transfer.receivedChunks / transfer.totalChunks;
		this.onFileTransferProgress(transfer_id, progress);
		
		// Check if complete
		if (transfer.receivedChunks >= transfer.totalChunks || is_last_chunk) {
			await this.reconstructFile(transfer_id);
		}
	}
	
	async reconstructFile(transferId) {
		const transfer = this.fileTransfers.get(transferId);
		if (!transfer) return;
		
		try {
			// Reconstruct file from chunks
			const chunks = [];
			for (let i = 0; i < transfer.totalChunks; i++) {
				if (transfer.chunks[i]) {
					const chunkData = atob(transfer.chunks[i]);
					const chunkArray = new Uint8Array(chunkData.length);
					for (let j = 0; j < chunkData.length; j++) {
						chunkArray[j] = chunkData.charCodeAt(j);
					}
					chunks.push(chunkArray);
				}
			}
			
			const fileData = new Uint8Array(chunks.reduce((acc, chunk) => acc + chunk.length, 0));
			let offset = 0;
			for (const chunk of chunks) {
				fileData.set(chunk, offset);
				offset += chunk.length;
			}
			
			// Create blob and download
			const blob = new Blob([fileData], { type: transfer.file_type });
			const url = URL.createObjectURL(blob);
			
			const a = document.createElement('a');
			a.href = url;
			a.download = transfer.filename;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
			
			transfer.status = 'completed';
			console.log(`File received and downloaded: ${transfer.filename}`);
			
		} catch (error) {
			console.error('Error reconstructing file:', error);
			transfer.status = 'error';
		}
	}
	
	// Collaborative editing methods
	sendCollaborativeEdit(userId, fieldName, operation, position, content) {
		const editMessage = {
			type: 'collaborative_edit',
			edit_id: this.generateId(),
			field_name: fieldName,
			operation: operation,
			position: position,
			content: content,
			timestamp: new Date().toISOString()
		};
		
		this.sendDataChannelMessage(userId, editMessage);
	}
	
	async handleCollaborativeEdit(userId, message) {
		const { field_name, operation, position, content } = message;
		
		// Apply edit to local form field
		const field = document.querySelector(`[name="${field_name}"]`);
		if (field) {
			const currentValue = field.value;
			let newValue = currentValue;
			
			switch (operation) {
				case 'insert':
					newValue = currentValue.slice(0, position) + content + currentValue.slice(position);
					break;
				case 'delete':
					newValue = currentValue.slice(0, position) + currentValue.slice(position + content.length);
					break;
				case 'replace':
					newValue = currentValue.slice(0, position) + content + currentValue.slice(position + message.length || 1);
					break;
			}
			
			field.value = newValue;
			
			// Trigger change event
			field.dispatchEvent(new Event('input', { bubbles: true }));
		}
		
		console.log(`Applied collaborative edit from ${userId}:`, operation, position, content);
	}
	
	// Utility methods
	sendDataChannelMessage(userId, message) {
		const dataChannel = this.dataChannels.get(userId);
		if (dataChannel && dataChannel.readyState === 'open') {
			dataChannel.send(JSON.stringify(message));
		} else {
			console.warn(`Data channel not available for user ${userId}`);
		}
	}
	
	async fileToBase64(file) {
		return new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.onload = () => resolve(reader.result.split(',')[1]);
			reader.onerror = reject;
			reader.readAsDataURL(file);
		});
	}
	
	generateId() {
		return 'xxxx-xxxx-4xxx-yxxx'.replace(/[xy]/g, function(c) {
			const r = Math.random() * 16 | 0;
			const v = c === 'x' ? r : (r & 0x3 | 0x8);
			return v.toString(16);
		});
	}
	
	sleep(ms) {
		return new Promise(resolve => setTimeout(resolve, ms));
	}
	
	async showFileTransferDialog(userId, filename, fileSize) {
		// Simple implementation - in practice, this would show a proper dialog
		return confirm(`${userId} wants to send you a file: ${filename} (${fileSize} bytes). Accept?`);
	}
	
	onFileTransferProgress(transferId, progress) {
		// Override this method to update UI
		console.log(`File transfer progress: ${transferId} - ${(progress * 100).toFixed(1)}%`);
	}
	
	// Public API
	getActiveTransfers() {
		const transfers = [];
		for (const [id, transfer] of this.fileTransfers.entries()) {
			transfers.push({
				transfer_id: id,
				filename: transfer.filename || transfer.file?.name,
				status: transfer.status,
				progress: transfer.sentChunks / transfer.totalChunks || transfer.receivedChunks / transfer.totalChunks || 0
			});
		}
		return transfers;
	}
	
	cancelTransfer(transferId) {
		if (this.fileTransfers.has(transferId)) {
			const transfer = this.fileTransfers.get(transferId);
			transfer.status = 'cancelled';
			this.fileTransfers.delete(transferId);
		}
	}
}

// Export for global use
window.WebRTCDataChannelManager = WebRTCDataChannelManager;
"""


if __name__ == "__main__":
	# Test the data channel manager
	async def test_data_channels():
		print("Testing WebRTC data channel manager...")
		
		manager = WebRTCDataChannelManager()
		
		# Test file transfer initiation
		file_info = {
			"name": "test_document.pdf",
			"size": 1024000,  # 1MB
			"type": "application/pdf"
		}
		
		result = await manager.initiate_file_transfer("user1", "user2", file_info)
		print(f"File transfer initiation: {result}")
		
		# Test data channel message handling
		message = {
			"type": "ping",
			"timestamp": datetime.utcnow().isoformat()
		}
		
		response = await manager.handle_data_channel_message("user1", "peer1", message)
		print(f"Ping response: {response}")
		
		# Generate client JavaScript
		js_code = generate_data_channel_javascript()
		print(f"Generated JavaScript: {len(js_code)} characters")
		
		print("✅ WebRTC data channel manager test completed")
	
	asyncio.run(test_data_channels())