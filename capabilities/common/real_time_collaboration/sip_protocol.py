"""
SIP Protocol Implementation for APG Real-Time Collaboration

Provides SIP (Session Initiation Protocol) communication for traditional
telephony integration, VoIP calls, and enterprise phone system integration.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import socket
import re
from pathlib import Path

try:
	import aiortc
	from aiortc import RTCPeerConnection, RTCSessionDescription
	from aiortc.contrib.media import MediaPlayer, MediaRecorder
except ImportError:
	print("SIP/RTC dependencies not installed. Run: pip install aiortc")
	aiortc = None
	RTCPeerConnection = None

logger = logging.getLogger(__name__)


class SIPMethod(Enum):
	"""SIP method types"""
	INVITE = "INVITE"
	ACK = "ACK"
	BYE = "BYE"
	CANCEL = "CANCEL"
	REGISTER = "REGISTER"
	OPTIONS = "OPTIONS"
	SUBSCRIBE = "SUBSCRIBE"
	NOTIFY = "NOTIFY"
	MESSAGE = "MESSAGE"
	INFO = "INFO"
	PRACK = "PRACK"
	UPDATE = "UPDATE"
	REFER = "REFER"


class SIPResponseCode(Enum):
	"""SIP response codes"""
	# 1xx - Provisional
	TRYING = 100
	RINGING = 180
	SESSION_PROGRESS = 183
	
	# 2xx - Success
	OK = 200
	ACCEPTED = 202
	
	# 3xx - Redirection
	MULTIPLE_CHOICES = 300
	MOVED_PERMANENTLY = 301
	MOVED_TEMPORARILY = 302
	
	# 4xx - Client Error
	BAD_REQUEST = 400
	UNAUTHORIZED = 401
	FORBIDDEN = 403
	NOT_FOUND = 404
	METHOD_NOT_ALLOWED = 405
	REQUEST_TIMEOUT = 408
	BUSY_HERE = 486
	
	# 5xx - Server Error
	INTERNAL_SERVER_ERROR = 500
	NOT_IMPLEMENTED = 501
	BAD_GATEWAY = 502
	SERVICE_UNAVAILABLE = 503
	
	# 6xx - Global Failure
	BUSY_EVERYWHERE = 600
	DECLINE = 603


class SIPCallState(Enum):
	"""SIP call states"""
	IDLE = "idle"
	CALLING = "calling"
	INCOMING = "incoming"
	RINGING = "ringing"
	CONNECTED = "connected"
	HOLD = "hold"
	TERMINATED = "terminated"
	ERROR = "error"


@dataclass
class SIPMessage:
	"""SIP message structure"""
	method: Optional[SIPMethod] = None
	response_code: Optional[SIPResponseCode] = None
	request_uri: Optional[str] = None
	headers: Dict[str, str] = field(default_factory=dict)
	body: str = ""
	raw_message: str = ""
	timestamp: datetime = field(default_factory=datetime.utcnow)
	message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"method": self.method.value if self.method else None,
			"response_code": self.response_code.value if self.response_code else None,
			"request_uri": self.request_uri,
			"headers": self.headers,
			"body": self.body,
			"timestamp": self.timestamp.isoformat(),
			"message_id": self.message_id
		}


@dataclass
class SIPCall:
	"""SIP call session"""
	call_id: str
	from_uri: str
	to_uri: str
	state: SIPCallState = SIPCallState.IDLE
	direction: str = "outbound"  # outbound, inbound
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	duration_seconds: int = 0
	local_sdp: Optional[str] = None
	remote_sdp: Optional[str] = None
	rtc_connection: Optional[Any] = None  # RTCPeerConnection
	media_recorder: Optional[Any] = None
	tags: Dict[str, str] = field(default_factory=dict)
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"call_id": self.call_id,
			"from_uri": self.from_uri,
			"to_uri": self.to_uri,
			"state": self.state.value,
			"direction": self.direction,
			"start_time": self.start_time.isoformat() if self.start_time else None,
			"end_time": self.end_time.isoformat() if self.end_time else None,
			"duration_seconds": self.duration_seconds,
			"tags": self.tags
		}


@dataclass
class SIPAccount:
	"""SIP account configuration"""
	username: str
	password: str
	domain: str
	proxy_server: Optional[str] = None
	outbound_proxy: Optional[str] = None
	port: int = 5060
	transport: str = "UDP"  # UDP, TCP, TLS
	display_name: Optional[str] = None
	expires: int = 3600
	
	@property
	def uri(self) -> str:
		return f"sip:{self.username}@{self.domain}"
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"username": self.username,
			"domain": self.domain,
			"proxy_server": self.proxy_server,
			"outbound_proxy": self.outbound_proxy,
			"port": self.port,
			"transport": self.transport,
			"display_name": self.display_name,
			"uri": self.uri
		}


class SIPProtocolManager:
	"""Manages SIP protocol for telephony integration"""
	
	def __init__(self, local_host: str = "0.0.0.0", local_port: int = 5060):
		self.local_host = local_host
		self.local_port = local_port
		
		# Network components
		self.udp_transport: Optional[asyncio.DatagramTransport] = None
		self.udp_protocol: Optional[asyncio.DatagramProtocol] = None
		self.is_running = False
		
		# Account management
		self.accounts: Dict[str, SIPAccount] = {}
		self.registered_accounts: Set[str] = set()
		
		# Call management
		self.active_calls: Dict[str, SIPCall] = {}
		self.call_history: List[SIPCall] = []
		
		# Message handling
		self.message_handlers: Dict[str, List[Callable]] = {}
		self.call_handlers: List[Callable] = []
		
		# SIP transaction tracking
		self.transactions: Dict[str, Dict[str, Any]] = {}
		self.sequence_numbers: Dict[str, int] = {}
		
		# Configuration
		self.user_agent = "APG-RTC-SIP/1.0"
		self.supported_methods = [method.value for method in SIPMethod]
		self.supported_codecs = ["PCMU", "PCMA", "G722", "opus"]
		
		# Statistics
		self.stats = {
			"messages_sent": 0,
			"messages_received": 0,
			"calls_initiated": 0,
			"calls_received": 0,
			"calls_completed": 0,
			"calls_failed": 0,
			"registration_attempts": 0,
			"successful_registrations": 0,
			"uptime_start": None
		}
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize SIP protocol stack"""
		try:
			# Create UDP transport for SIP signaling
			loop = asyncio.get_event_loop()
			
			# Create protocol
			self.udp_protocol = SIPUDPProtocol(self)
			
			# Create transport
			transport, protocol = await loop.create_datagram_endpoint(
				lambda: self.udp_protocol,
				local_addr=(self.local_host, self.local_port)
			)
			
			self.udp_transport = transport
			self.is_running = True
			self.stats["uptime_start"] = datetime.utcnow()
			
			logger.info(f"SIP protocol stack started on {self.local_host}:{self.local_port}")
			
			return {
				"status": "started",
				"address": f"{self.local_host}:{self.local_port}",
				"transport": "UDP",
				"user_agent": self.user_agent
			}
			
		except Exception as e:
			logger.error(f"Failed to initialize SIP protocol: {e}")
			return {"error": f"SIP initialization failed: {str(e)}"}
	
	async def shutdown(self) -> Dict[str, Any]:
		"""Shutdown SIP protocol stack"""
		try:
			# Terminate active calls
			for call_id in list(self.active_calls.keys()):
				await self.terminate_call(call_id)
			
			# Unregister accounts
			for account_uri in list(self.registered_accounts):
				await self.unregister_account(account_uri)
			
			# Close transport
			if self.udp_transport:
				self.udp_transport.close()
			
			self.is_running = False
			
			logger.info("SIP protocol stack stopped")
			
			return {"status": "stopped"}
			
		except Exception as e:
			logger.error(f"Error stopping SIP protocol: {e}")
			return {"error": f"Shutdown failed: {str(e)}"}
	
	async def add_account(self, account: SIPAccount) -> Dict[str, Any]:
		"""Add SIP account"""
		try:
			self.accounts[account.uri] = account
			
			logger.info(f"Added SIP account: {account.uri}")
			
			return {
				"status": "added",
				"account_uri": account.uri,
				"display_name": account.display_name
			}
			
		except Exception as e:
			logger.error(f"Failed to add SIP account: {e}")
			return {"error": f"Account add failed: {str(e)}"}
	
	async def register_account(self, account_uri: str) -> Dict[str, Any]:
		"""Register SIP account with server"""
		try:
			if account_uri not in self.accounts:
				return {"error": "Account not found"}
			
			account = self.accounts[account_uri]
			
			# Create REGISTER message
			register_msg = self._create_register_message(account)
			
			# Send registration
			await self._send_sip_message(register_msg, account.proxy_server or account.domain, account.port)
			
			# Update statistics
			self.stats["registration_attempts"] += 1
			
			logger.info(f"Sent SIP REGISTER for {account_uri}")
			
			return {
				"status": "registration_sent",
				"account_uri": account_uri
			}
			
		except Exception as e:
			logger.error(f"Failed to register SIP account: {e}")
			return {"error": f"Registration failed: {str(e)}"}
	
	async def unregister_account(self, account_uri: str) -> Dict[str, Any]:
		"""Unregister SIP account"""
		try:
			if account_uri not in self.accounts:
				return {"error": "Account not found"}
			
			account = self.accounts[account_uri]
			
			# Create REGISTER message with Expires: 0
			register_msg = self._create_register_message(account, expires=0)
			
			# Send unregistration
			await self._send_sip_message(register_msg, account.proxy_server or account.domain, account.port)
			
			# Remove from registered accounts
			self.registered_accounts.discard(account_uri)
			
			logger.info(f"Sent SIP unregister for {account_uri}")
			
			return {
				"status": "unregister_sent",
				"account_uri": account_uri
			}
			
		except Exception as e:
			logger.error(f"Failed to unregister SIP account: {e}")
			return {"error": f"Unregister failed: {str(e)}"}
	
	async def initiate_call(self, from_account_uri: str, to_uri: str) -> Dict[str, Any]:
		"""Initiate outbound SIP call"""
		try:
			if from_account_uri not in self.accounts:
				return {"error": "From account not found"}
			
			if from_account_uri not in self.registered_accounts:
				return {"error": "Account not registered"}
			
			account = self.accounts[from_account_uri]
			
			# Create call session
			call_id = str(uuid.uuid4())
			call = SIPCall(
				call_id=call_id,
				from_uri=from_account_uri,
				to_uri=to_uri,
				state=SIPCallState.CALLING,
				direction="outbound",
				start_time=datetime.utcnow()
			)
			
			# Create RTC connection if aiortc available
			if RTCPeerConnection:
				call.rtc_connection = RTCPeerConnection()
				
				# Add audio track
				# In real implementation, would set up media streams
				
				# Create offer
				offer = await call.rtc_connection.createOffer()
				await call.rtc_connection.setLocalDescription(offer)
				call.local_sdp = str(offer.sdp)
			
			# Create INVITE message
			invite_msg = self._create_invite_message(account, to_uri, call_id, call.local_sdp)
			
			# Store call
			self.active_calls[call_id] = call
			
			# Send INVITE
			await self._send_sip_message(invite_msg, account.proxy_server or account.domain, account.port)
			
			# Update statistics
			self.stats["calls_initiated"] += 1
			
			logger.info(f"Initiated SIP call from {from_account_uri} to {to_uri}")
			
			return {
				"status": "call_initiated",
				"call_id": call_id,
				"from_uri": from_account_uri,
				"to_uri": to_uri
			}
			
		except Exception as e:
			logger.error(f"Failed to initiate SIP call: {e}")
			self.stats["calls_failed"] += 1
			return {"error": f"Call initiation failed: {str(e)}"}
	
	async def answer_call(self, call_id: str, accept: bool = True) -> Dict[str, Any]:
		"""Answer incoming SIP call"""
		try:
			if call_id not in self.active_calls:
				return {"error": "Call not found"}
			
			call = self.active_calls[call_id]
			
			if call.state != SIPCallState.INCOMING:
				return {"error": "Call not in incoming state"}
			
			if accept:
				# Create RTC connection if aiortc available
				if RTCPeerConnection and not call.rtc_connection:
					call.rtc_connection = RTCPeerConnection()
					
					# Create answer
					if call.remote_sdp:
						remote_desc = RTCSessionDescription(call.remote_sdp, "offer")
						await call.rtc_connection.setRemoteDescription(remote_desc)
						
						answer = await call.rtc_connection.createAnswer()
						await call.rtc_connection.setLocalDescription(answer)
						call.local_sdp = str(answer.sdp)
				
				# Send 200 OK
				response_msg = self._create_response_message(
					SIPResponseCode.OK, 
					call,
					body=call.local_sdp
				)
				
				call.state = SIPCallState.CONNECTED
				
			else:
				# Send 603 Decline
				response_msg = self._create_response_message(
					SIPResponseCode.DECLINE,
					call
				)
				
				call.state = SIPCallState.TERMINATED
				call.end_time = datetime.utcnow()
				
				# Move to history
				self.call_history.append(call)
				del self.active_calls[call_id]
			
			# Send response (simulate sending to originator)
			# In real implementation, would send to actual SIP endpoint
			
			# Update statistics
			if accept:
				self.stats["calls_completed"] += 1
			else:
				self.stats["calls_failed"] += 1
			
			logger.info(f"{'Answered' if accept else 'Declined'} SIP call {call_id}")
			
			return {
				"status": "answered" if accept else "declined",
				"call_id": call_id,
				"accepted": accept
			}
			
		except Exception as e:
			logger.error(f"Failed to answer SIP call: {e}")
			return {"error": f"Call answer failed: {str(e)}"}
	
	async def terminate_call(self, call_id: str) -> Dict[str, Any]:
		"""Terminate active SIP call"""
		try:
			if call_id not in self.active_calls:
				return {"error": "Call not found"}
			
			call = self.active_calls[call_id]
			
			# Send BYE message
			bye_msg = self._create_bye_message(call)
			
			# Close RTC connection
			if call.rtc_connection:
				await call.rtc_connection.close()
			
			# Update call state
			call.state = SIPCallState.TERMINATED
			call.end_time = datetime.utcnow()
			
			if call.start_time:
				call.duration_seconds = int((call.end_time - call.start_time).total_seconds())
			
			# Move to history
			self.call_history.append(call)
			del self.active_calls[call_id]
			
			# Update statistics
			self.stats["calls_completed"] += 1
			
			logger.info(f"Terminated SIP call {call_id}")
			
			return {
				"status": "terminated",
				"call_id": call_id,
				"duration_seconds": call.duration_seconds
			}
			
		except Exception as e:
			logger.error(f"Failed to terminate SIP call: {e}")
			return {"error": f"Call termination failed: {str(e)}"}
	
	async def hold_call(self, call_id: str) -> Dict[str, Any]:
		"""Put SIP call on hold"""
		try:
			if call_id not in self.active_calls:
				return {"error": "Call not found"}
			
			call = self.active_calls[call_id]
			
			if call.state != SIPCallState.CONNECTED:
				return {"error": "Call not connected"}
			
			# Send re-INVITE with hold SDP
			# In real implementation, would modify SDP for hold
			
			call.state = SIPCallState.HOLD
			
			logger.info(f"Put SIP call {call_id} on hold")
			
			return {
				"status": "on_hold",
				"call_id": call_id
			}
			
		except Exception as e:
			logger.error(f"Failed to hold SIP call: {e}")
			return {"error": f"Call hold failed: {str(e)}"}
	
	async def unhold_call(self, call_id: str) -> Dict[str, Any]:
		"""Take SIP call off hold"""
		try:
			if call_id not in self.active_calls:
				return {"error": "Call not found"}
			
			call = self.active_calls[call_id]
			
			if call.state != SIPCallState.HOLD:
				return {"error": "Call not on hold"}
			
			# Send re-INVITE with unhold SDP
			# In real implementation, would modify SDP for unhold
			
			call.state = SIPCallState.CONNECTED
			
			logger.info(f"Took SIP call {call_id} off hold")
			
			return {
				"status": "off_hold",
				"call_id": call_id
			}
			
		except Exception as e:
			logger.error(f"Failed to unhold SIP call: {e}")
			return {"error": f"Call unhold failed: {str(e)}"}
	
	def _create_register_message(self, account: SIPAccount, expires: int = None) -> str:
		"""Create SIP REGISTER message"""
		expires = expires if expires is not None else account.expires
		
		# Get next sequence number
		seq_num = self._get_next_sequence_number(account.uri)
		
		# Create headers
		headers = {
			"Via": f"SIP/2.0/{account.transport} {self.local_host}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
			"Max-Forwards": "70",
			"From": f'"{account.display_name or account.username}" <{account.uri}>;tag={uuid.uuid4().hex[:8]}',
			"To": f'"{account.display_name or account.username}" <{account.uri}>',
			"Call-ID": str(uuid.uuid4()),
			"CSeq": f"{seq_num} REGISTER",
			"Contact": f'<sip:{account.username}@{self.local_host}:{self.local_port};transport={account.transport.lower()}>',
			"Expires": str(expires),
			"User-Agent": self.user_agent,
			"Content-Length": "0"
		}
		
		# Build message
		request_line = f"REGISTER sip:{account.domain} SIP/2.0"
		header_lines = [f"{name}: {value}" for name, value in headers.items()]
		
		message = "\r\n".join([request_line] + header_lines + ["", ""])
		
		return message
	
	def _create_invite_message(self, account: SIPAccount, to_uri: str, call_id: str, sdp: str = None) -> str:
		"""Create SIP INVITE message"""
		seq_num = self._get_next_sequence_number(account.uri)
		
		# Create basic SDP if none provided
		if not sdp:
			sdp = self._create_basic_sdp()
		
		headers = {
			"Via": f"SIP/2.0/{account.transport} {self.local_host}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
			"Max-Forwards": "70",
			"From": f'"{account.display_name or account.username}" <{account.uri}>;tag={uuid.uuid4().hex[:8]}',
			"To": f"<{to_uri}>",
			"Call-ID": call_id,
			"CSeq": f"{seq_num} INVITE",
			"Contact": f'<sip:{account.username}@{self.local_host}:{self.local_port}>',
			"Content-Type": "application/sdp",
			"Content-Length": str(len(sdp)),
			"User-Agent": self.user_agent
		}
		
		# Build message
		request_line = f"INVITE {to_uri} SIP/2.0"
		header_lines = [f"{name}: {value}" for name, value in headers.items()]
		
		message = "\r\n".join([request_line] + header_lines + ["", sdp])
		
		return message
	
	def _create_response_message(self, response_code: SIPResponseCode, call: SIPCall, body: str = "") -> str:
		"""Create SIP response message"""
		# Simplified response creation
		status_line = f"SIP/2.0 {response_code.value} {response_code.name.replace('_', ' ').title()}"
		headers = {
			"Content-Length": str(len(body))
		}
		
		header_lines = [f"{name}: {value}" for name, value in headers.items()]
		
		message = "\r\n".join([status_line] + header_lines + ["", body])
		
		return message
	
	def _create_bye_message(self, call: SIPCall) -> str:
		"""Create SIP BYE message"""
		seq_num = self._get_next_sequence_number(call.from_uri)
		
		headers = {
			"Via": f"SIP/2.0/UDP {self.local_host}:{self.local_port};branch=z9hG4bK{uuid.uuid4().hex[:8]}",
			"Max-Forwards": "70",
			"From": f"<{call.from_uri}>;tag={uuid.uuid4().hex[:8]}",
			"To": f"<{call.to_uri}>",
			"Call-ID": call.call_id,
			"CSeq": f"{seq_num} BYE",
			"User-Agent": self.user_agent,
			"Content-Length": "0"
		}
		
		request_line = f"BYE {call.to_uri} SIP/2.0"
		header_lines = [f"{name}: {value}" for name, value in headers.items()]
		
		message = "\r\n".join([request_line] + header_lines + ["", ""])
		
		return message
	
	def _create_basic_sdp(self) -> str:
		"""Create basic SDP for audio call"""
		session_id = int(datetime.utcnow().timestamp())
		
		sdp = f"""v=0
o=- {session_id} {session_id} IN IP4 {self.local_host}
s=APG RTC Session
c=IN IP4 {self.local_host}
t=0 0
m=audio 5004 RTP/AVP 0 8 96
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=rtpmap:96 opus/48000/2
a=sendrecv"""
		
		return sdp
	
	def _get_next_sequence_number(self, account_uri: str) -> int:
		"""Get next sequence number for account"""
		if account_uri not in self.sequence_numbers:
			self.sequence_numbers[account_uri] = 1
		else:
			self.sequence_numbers[account_uri] += 1
		
		return self.sequence_numbers[account_uri]
	
	async def _send_sip_message(self, message: str, host: str, port: int):
		"""Send SIP message via UDP"""
		try:
			if self.udp_transport:
				self.udp_transport.sendto(message.encode(), (host, port))
				self.stats["messages_sent"] += 1
				
				logger.debug(f"Sent SIP message to {host}:{port}")
			
		except Exception as e:
			logger.error(f"Failed to send SIP message: {e}")
	
	async def _handle_incoming_message(self, data: bytes, addr: tuple):
		"""Handle incoming SIP message"""
		try:
			message_text = data.decode('utf-8')
			sip_message = self._parse_sip_message(message_text)
			
			self.stats["messages_received"] += 1
			
			# Route message based on method/response
			if sip_message.method:
				await self._handle_sip_request(sip_message, addr)
			elif sip_message.response_code:
				await self._handle_sip_response(sip_message, addr)
			
			logger.debug(f"Processed SIP message from {addr}")
			
		except Exception as e:
			logger.error(f"Error handling incoming SIP message: {e}")
	
	def _parse_sip_message(self, message_text: str) -> SIPMessage:
		"""Parse SIP message"""
		lines = message_text.split('\r\n')
		
		if not lines:
			return SIPMessage(raw_message=message_text)
		
		# Parse first line (request or response)
		first_line = lines[0]
		sip_message = SIPMessage(raw_message=message_text)
		
		if first_line.startswith('SIP/2.0'):
			# Response
			parts = first_line.split(' ', 2)
			if len(parts) >= 2:
				try:
					sip_message.response_code = SIPResponseCode(int(parts[1]))
				except ValueError:
					pass
		else:
			# Request
			parts = first_line.split(' ')
			if len(parts) >= 2:
				try:
					sip_message.method = SIPMethod(parts[0])
					sip_message.request_uri = parts[1]
				except ValueError:
					pass
		
		# Parse headers
		header_section = True
		body_lines = []
		
		for line in lines[1:]:
			if header_section:
				if line == '':
					header_section = False
					continue
				
				if ':' in line:
					name, value = line.split(':', 1)
					sip_message.headers[name.strip()] = value.strip()
			else:
				body_lines.append(line)
		
		sip_message.body = '\r\n'.join(body_lines)
		
		return sip_message
	
	async def _handle_sip_request(self, message: SIPMessage, addr: tuple):
		"""Handle SIP request"""
		if message.method == SIPMethod.INVITE:
			await self._handle_invite_request(message, addr)
		elif message.method == SIPMethod.ACK:
			await self._handle_ack_request(message, addr)
		elif message.method == SIPMethod.BYE:
			await self._handle_bye_request(message, addr)
		elif message.method == SIPMethod.CANCEL:
			await self._handle_cancel_request(message, addr)
		elif message.method == SIPMethod.OPTIONS:
			await self._handle_options_request(message, addr)
	
	async def _handle_sip_response(self, message: SIPMessage, addr: tuple):
		"""Handle SIP response"""
		if message.response_code == SIPResponseCode.OK:
			await self._handle_ok_response(message, addr)
		elif message.response_code == SIPResponseCode.RINGING:
			await self._handle_ringing_response(message, addr)
		elif message.response_code in [SIPResponseCode.BUSY_HERE, SIPResponseCode.DECLINE]:
			await self._handle_reject_response(message, addr)
	
	async def _handle_invite_request(self, message: SIPMessage, addr: tuple):
		"""Handle incoming INVITE request"""
		call_id = message.headers.get('Call-ID')
		from_uri = message.headers.get('From', '').split('<')[1].split('>')[0] if '<' in message.headers.get('From', '') else ''
		to_uri = message.headers.get('To', '').split('<')[1].split('>')[0] if '<' in message.headers.get('To', '') else ''
		
		# Create incoming call
		call = SIPCall(
			call_id=call_id or str(uuid.uuid4()),
			from_uri=from_uri,
			to_uri=to_uri,
			state=SIPCallState.INCOMING,
			direction="inbound",
			remote_sdp=message.body if message.body else None
		)
		
		self.active_calls[call.call_id] = call
		
		# Update statistics
		self.stats["calls_received"] += 1
		
		# Notify call handlers
		for handler in self.call_handlers:
			try:
				if asyncio.iscoroutinefunction(handler):
					await handler("incoming_call", call)
				else:
					handler("incoming_call", call)
			except Exception as e:
				logger.error(f"Error in call handler: {e}")
		
		logger.info(f"Received incoming INVITE from {from_uri}")
	
	async def _handle_ack_request(self, message: SIPMessage, addr: tuple):
		"""Handle ACK request"""
		logger.debug("Received ACK request")
	
	async def _handle_bye_request(self, message: SIPMessage, addr: tuple):
		"""Handle BYE request"""
		call_id = message.headers.get('Call-ID')
		
		if call_id in self.active_calls:
			call = self.active_calls[call_id]
			call.state = SIPCallState.TERMINATED
			call.end_time = datetime.utcnow()
			
			if call.start_time:
				call.duration_seconds = int((call.end_time - call.start_time).total_seconds())
			
			# Move to history
			self.call_history.append(call)
			del self.active_calls[call_id]
			
			logger.info(f"Call {call_id} terminated by remote party")
	
	async def _handle_cancel_request(self, message: SIPMessage, addr: tuple):
		"""Handle CANCEL request"""
		logger.debug("Received CANCEL request")
	
	async def _handle_options_request(self, message: SIPMessage, addr: tuple):
		"""Handle OPTIONS request"""
		logger.debug("Received OPTIONS request")
	
	async def _handle_ok_response(self, message: SIPMessage, addr: tuple):
		"""Handle 200 OK response"""
		call_id = message.headers.get('Call-ID')
		
		if call_id in self.active_calls:
			call = self.active_calls[call_id]
			call.state = SIPCallState.CONNECTED
			call.remote_sdp = message.body if message.body else call.remote_sdp
			
			logger.info(f"Call {call_id} connected")
	
	async def _handle_ringing_response(self, message: SIPMessage, addr: tuple):
		"""Handle 180 Ringing response"""
		call_id = message.headers.get('Call-ID')
		
		if call_id in self.active_calls:
			call = self.active_calls[call_id]
			call.state = SIPCallState.RINGING
			
			logger.info(f"Call {call_id} ringing")
	
	async def _handle_reject_response(self, message: SIPMessage, addr: tuple):
		"""Handle call rejection responses"""
		call_id = message.headers.get('Call-ID')
		
		if call_id in self.active_calls:
			call = self.active_calls[call_id]
			call.state = SIPCallState.TERMINATED
			call.end_time = datetime.utcnow()
			
			# Move to history
			self.call_history.append(call)
			del self.active_calls[call_id]
			
			self.stats["calls_failed"] += 1
			
			logger.info(f"Call {call_id} rejected")
	
	def register_call_handler(self, handler: Callable):
		"""Register call event handler"""
		self.call_handlers.append(handler)
		logger.info("Registered SIP call handler")
	
	def get_accounts(self) -> List[Dict[str, Any]]:
		"""Get list of SIP accounts"""
		return [account.to_dict() for account in self.accounts.values()]
	
	def get_active_calls(self) -> List[Dict[str, Any]]:
		"""Get list of active calls"""
		return [call.to_dict() for call in self.active_calls.values()]
	
	def get_call_history(self, limit: int = 50) -> List[Dict[str, Any]]:
		"""Get call history"""
		return [call.to_dict() for call in self.call_history[-limit:]]
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get SIP protocol statistics"""
		uptime_seconds = 0
		if self.stats["uptime_start"]:
			uptime_seconds = int((datetime.utcnow() - self.stats["uptime_start"]).total_seconds())
		
		return {
			**self.stats,
			"is_running": self.is_running,
			"local_address": f"{self.local_host}:{self.local_port}",
			"registered_accounts": len(self.registered_accounts),
			"active_calls": len(self.active_calls),
			"call_history_size": len(self.call_history),
			"supported_codecs": self.supported_codecs,
			"uptime_seconds": uptime_seconds,
			"uptime_start": self.stats["uptime_start"].isoformat() if self.stats["uptime_start"] else None
		}


class SIPUDPProtocol(asyncio.DatagramProtocol):
	"""UDP protocol handler for SIP messages"""
	
	def __init__(self, sip_manager: SIPProtocolManager):
		self.sip_manager = sip_manager
	
	def connection_made(self, transport):
		self.transport = transport
	
	def datagram_received(self, data, addr):
		# Handle incoming SIP message
		asyncio.create_task(self.sip_manager._handle_incoming_message(data, addr))
	
	def error_received(self, exc):
		logger.error(f"SIP UDP protocol error: {exc}")


# Global SIP manager instance
sip_protocol_manager = None


async def initialize_sip_protocol(local_host: str = "0.0.0.0", local_port: int = 5060) -> Dict[str, Any]:
	"""Initialize global SIP protocol manager"""
	global sip_protocol_manager
	
	sip_protocol_manager = SIPProtocolManager(local_host=local_host, local_port=local_port)
	
	result = await sip_protocol_manager.initialize()
	
	return result


def get_sip_manager() -> Optional[SIPProtocolManager]:
	"""Get global SIP protocol manager"""
	return sip_protocol_manager


if __name__ == "__main__":
	# Test SIP protocol implementation
	async def test_sip():
		print("Testing SIP protocol implementation...")
		
		# Initialize SIP manager
		result = await initialize_sip_protocol()
		print(f"SIP initialization result: {result}")
		
		if result.get("status") == "started":
			manager = get_sip_manager()
			
			# Add test account
			test_account = SIPAccount(
				username="testuser",
				password="password",
				domain="example.com",
				display_name="Test User"
			)
			
			account_result = await manager.add_account(test_account)
			print(f"Account add result: {account_result}")
			
			# Test registration (would need real SIP server)
			# register_result = await manager.register_account(test_account.uri)
			# print(f"Registration result: {register_result}")
			
			# Get statistics
			stats = manager.get_statistics()
			print(f"SIP statistics: {stats}")
			
			# Wait a bit for testing
			await asyncio.sleep(2)
			
			# Shutdown
			shutdown_result = await manager.shutdown()
			print(f"Shutdown result: {shutdown_result}")
		
		print("✅ SIP protocol test completed")
	
	asyncio.run(test_sip())