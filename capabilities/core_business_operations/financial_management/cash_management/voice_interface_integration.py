#!/usr/bin/env python3
"""APG Cash Management - Voice Interface Integration

Advanced voice interface with speech recognition, natural language processing,
and voice-driven cash management operations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import time
import base64
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager
import io
import wave

import asyncpg
import redis.asyncio as redis
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import speech_recognition as sr
import pyttsx3
import openai
from threading import Thread
import queue
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCommand(str, Enum):
	"""Voice command types."""
	QUERY_BALANCE = "query_balance"
	GET_FORECAST = "get_forecast"
	LIST_TRANSACTIONS = "list_transactions"
	CREATE_TRANSFER = "create_transfer"
	SET_ALERT = "set_alert"
	GENERATE_REPORT = "generate_report"
	SHOW_DASHBOARD = "show_dashboard"
	SCHEDULE_PAYMENT = "schedule_payment"
	CHECK_RISKS = "check_risks"
	EXPORT_DATA = "export_data"

class SpeechEngine(str, Enum):
	"""Speech recognition engines."""
	GOOGLE = "google"
	AZURE = "azure"
	AWS_TRANSCRIBE = "aws_transcribe"
	WHISPER = "whisper"
	LOCAL_SPHINX = "local_sphinx"

class VoiceLanguage(str, Enum):
	"""Supported voice languages."""
	ENGLISH_US = "en-US"
	ENGLISH_UK = "en-UK"
	SPANISH = "es-ES"
	FRENCH = "fr-FR"
	GERMAN = "de-DE"
	ITALIAN = "it-IT"
	JAPANESE = "ja-JP"
	CHINESE = "zh-CN"

class ResponseMode(str, Enum):
	"""Voice response modes."""
	VOICE_ONLY = "voice_only"
	VOICE_AND_TEXT = "voice_and_text"
	TEXT_ONLY = "text_only"
	VISUAL_AND_VOICE = "visual_and_voice"

@dataclass
class VoiceSession:
	"""Voice interaction session."""
	session_id: str
	user_id: str
	language: VoiceLanguage
	start_time: datetime
	last_interaction: datetime
	conversation_history: List[Dict[str, Any]] = field(default_factory=list)
	context: Dict[str, Any] = field(default_factory=dict)
	active: bool = True

class VoiceConfiguration(BaseModel):
	"""Voice interface configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	speech_engine: SpeechEngine = SpeechEngine.WHISPER
	default_language: VoiceLanguage = VoiceLanguage.ENGLISH_US
	response_mode: ResponseMode = ResponseMode.VOICE_AND_TEXT
	voice_timeout_seconds: int = Field(default=300, ge=30, le=1800)
	confidence_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
	
	# TTS Configuration
	tts_voice: str = "default"
	tts_speed: int = Field(default=150, ge=50, le=300)
	tts_volume: float = Field(default=0.8, ge=0.1, le=1.0)
	
	# Advanced features
	multi_language_support: bool = True
	context_awareness: bool = True
	interrupt_detection: bool = True
	noise_cancellation: bool = True
	wake_word_enabled: bool = True
	wake_word: str = "Hey APG"

class VoiceInterfaceIntegration:
	"""Advanced voice interface for cash management operations."""
	
	def __init__(
		self,
		tenant_id: str,
		db_pool: asyncpg.Pool,
		config: Optional[VoiceConfiguration] = None,
		openai_api_key: Optional[str] = None
	):
		self.tenant_id = tenant_id
		self.db_pool = db_pool
		self.config = config or VoiceConfiguration()
		self.openai_api_key = openai_api_key
		
		# Voice processing components
		self.recognizer = sr.Recognizer()
		self.microphone = sr.Microphone()
		self.tts_engine = None
		
		# Session management
		self.active_sessions: Dict[str, VoiceSession] = {}
		self.command_queue = asyncio.Queue()
		
		# NLP and context
		self.conversation_context: Dict[str, Any] = {}
		self.command_handlers: Dict[VoiceCommand, Callable] = {}
		
		# Audio processing
		self.audio_buffer = queue.Queue()
		self.listening_active = False
		
		logger.info(f"Initialized VoiceInterfaceIntegration for tenant {tenant_id}")
	
	async def initialize(self) -> None:
		"""Initialize voice interface components."""
		try:
			# Initialize TTS engine
			await self._initialize_tts_engine()
			
			# Initialize speech recognition
			await self._initialize_speech_recognition()
			
			# Register command handlers
			await self._register_command_handlers()
			
			# Start voice processing loop
			await self._start_voice_processing()
			
			# Calibrate microphone for ambient noise
			await self._calibrate_microphone()
			
			logger.info("Voice interface initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize voice interface: {e}")
			raise
	
	async def _initialize_tts_engine(self) -> None:
		"""Initialize text-to-speech engine."""
		try:
			self.tts_engine = pyttsx3.init()
			
			# Configure TTS settings
			voices = self.tts_engine.getProperty('voices')
			if voices:
				# Select voice based on configuration
				voice_index = 0
				if self.config.tts_voice != "default":
					for i, voice in enumerate(voices):
						if self.config.tts_voice.lower() in voice.name.lower():
							voice_index = i
							break
				
				self.tts_engine.setProperty('voice', voices[voice_index].id)
			
			self.tts_engine.setProperty('rate', self.config.tts_speed)
			self.tts_engine.setProperty('volume', self.config.tts_volume)
			
			logger.info("TTS engine initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize TTS engine: {e}")
			raise
	
	async def _initialize_speech_recognition(self) -> None:
		"""Initialize speech recognition system."""
		try:
			# Adjust for ambient noise
			with self.microphone as source:
				self.recognizer.adjust_for_ambient_noise(source, duration=2)
			
			# Configure recognition settings
			self.recognizer.energy_threshold = 4000
			self.recognizer.dynamic_energy_threshold = True
			self.recognizer.pause_threshold = 0.8
			self.recognizer.phrase_threshold = 0.3
			self.recognizer.non_speaking_duration = 0.5
			
			logger.info("Speech recognition initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize speech recognition: {e}")
			raise
	
	async def _register_command_handlers(self) -> None:
		"""Register handlers for voice commands."""
		self.command_handlers = {
			VoiceCommand.QUERY_BALANCE: self._handle_balance_query,
			VoiceCommand.GET_FORECAST: self._handle_forecast_request,
			VoiceCommand.LIST_TRANSACTIONS: self._handle_transaction_list,
			VoiceCommand.CREATE_TRANSFER: self._handle_transfer_creation,
			VoiceCommand.SET_ALERT: self._handle_alert_setup,
			VoiceCommand.GENERATE_REPORT: self._handle_report_generation,
			VoiceCommand.SHOW_DASHBOARD: self._handle_dashboard_request,
			VoiceCommand.SCHEDULE_PAYMENT: self._handle_payment_scheduling,
			VoiceCommand.CHECK_RISKS: self._handle_risk_check,
			VoiceCommand.EXPORT_DATA: self._handle_data_export
		}
		
		logger.info("Command handlers registered")
	
	async def _start_voice_processing(self) -> None:
		"""Start voice processing background tasks."""
		# Start listening loop
		asyncio.create_task(self._continuous_listening_loop())
		
		# Start command processing loop
		asyncio.create_task(self._command_processing_loop())
		
		# Start session cleanup loop
		asyncio.create_task(self._session_cleanup_loop())
		
		logger.info("Voice processing loops started")
	
	async def _calibrate_microphone(self) -> None:
		"""Calibrate microphone for optimal performance."""
		try:
			logger.info("Calibrating microphone for ambient noise...")
			
			with self.microphone as source:
				# Extended calibration for better accuracy
				self.recognizer.adjust_for_ambient_noise(source, duration=3)
			
			# Test microphone
			await self._test_microphone()
			
			logger.info("Microphone calibration completed")
			
		except Exception as e:
			logger.warning(f"Microphone calibration failed: {e}")
	
	async def _test_microphone(self) -> None:
		"""Test microphone functionality."""
		try:
			logger.info("Testing microphone... Please say 'test'")
			
			with self.microphone as source:
				audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
			
			# Test recognition
			text = self.recognizer.recognize_google(audio, language=self.config.default_language.value)
			logger.info(f"Microphone test successful: '{text}'")
			
		except sr.WaitTimeoutError:
			logger.warning("Microphone test timeout - no speech detected")
		except sr.UnknownValueError:
			logger.warning("Microphone test - could not understand speech")
		except Exception as e:
			logger.error(f"Microphone test failed: {e}")
	
	async def start_voice_session(
		self,
		user_id: str,
		language: Optional[VoiceLanguage] = None
	) -> str:
		"""Start a new voice interaction session."""
		try:
			session = VoiceSession(
				session_id=uuid7str(),
				user_id=user_id,
				language=language or self.config.default_language,
				start_time=datetime.now(),
				last_interaction=datetime.now()
			)
			
			self.active_sessions[session.session_id] = session
			
			# Welcome message
			welcome_msg = f"Hello! APG Cash Management voice interface is ready. How can I help you today?"
			await self._speak_response(welcome_msg, session.session_id)
			
			logger.info(f"Started voice session: {session.session_id} for user {user_id}")
			return session.session_id
			
		except Exception as e:
			logger.error(f"Error starting voice session: {e}")
			raise
	
	async def _continuous_listening_loop(self) -> None:
		"""Continuous listening loop for voice commands."""
		while True:
			try:
				if not self.listening_active:
					await asyncio.sleep(0.1)
					continue
				
				# Listen for audio
				await self._listen_for_audio()
				
			except Exception as e:
				logger.error(f"Listening loop error: {e}")
				await asyncio.sleep(1)
	
	async def _listen_for_audio(self) -> None:
		"""Listen for audio input and process."""
		try:
			with self.microphone as source:
				# Listen for audio with timeout
				audio = self.recognizer.listen(
					source,
					timeout=1,
					phrase_time_limit=10
				)
			
			# Process audio in background
			asyncio.create_task(self._process_audio(audio))
			
		except sr.WaitTimeoutError:
			# Normal timeout, continue listening
			pass
		except Exception as e:
			logger.error(f"Audio listening error: {e}")
	
	async def _process_audio(self, audio: sr.AudioData) -> None:
		"""Process captured audio."""
		try:
			# Recognize speech
			text = await self._recognize_speech(audio)
			
			if text:
				# Check for wake word if enabled
				if self.config.wake_word_enabled:
					if not self._detect_wake_word(text):
						return
				
				# Parse and queue command
				command_data = await self._parse_voice_command(text)
				if command_data:
					await self.command_queue.put(command_data)
			
		except Exception as e:
			logger.error(f"Audio processing error: {e}")
	
	async def _recognize_speech(self, audio: sr.AudioData) -> Optional[str]:
		"""Recognize speech from audio data."""
		try:
			if self.config.speech_engine == SpeechEngine.GOOGLE:
				text = self.recognizer.recognize_google(
					audio,
					language=self.config.default_language.value
				)
			elif self.config.speech_engine == SpeechEngine.WHISPER:
				# Use OpenAI Whisper if available
				if self.openai_api_key:
					text = await self._recognize_with_whisper(audio)
				else:
					# Fallback to Google
					text = self.recognizer.recognize_google(
						audio,
						language=self.config.default_language.value
					)
			else:
				# Default to Google
				text = self.recognizer.recognize_google(
					audio,
					language=self.config.default_language.value
				)
			
			logger.debug(f"Recognized speech: {text}")
			return text.strip()
			
		except sr.UnknownValueError:
			logger.debug("Could not understand speech")
			return None
		except sr.RequestError as e:
			logger.error(f"Speech recognition error: {e}")
			return None
	
	async def _recognize_with_whisper(self, audio: sr.AudioData) -> str:
		"""Recognize speech using OpenAI Whisper."""
		try:
			# Convert audio to the format expected by Whisper
			audio_data = audio.get_wav_data()
			
			# Create a temporary file-like object
			audio_file = io.BytesIO(audio_data)
			audio_file.name = "audio.wav"
			
			# Use OpenAI API for transcription
			client = openai.OpenAI(api_key=self.openai_api_key)
			transcript = client.audio.transcriptions.create(
				model="whisper-1",
				file=audio_file,
				language=self.config.default_language.value.split('-')[0]
			)
			
			return transcript.text
			
		except Exception as e:
			logger.error(f"Whisper recognition error: {e}")
			raise
	
	def _detect_wake_word(self, text: str) -> bool:
		"""Detect wake word in text."""
		text_lower = text.lower()
		wake_word_lower = self.config.wake_word.lower()
		return wake_word_lower in text_lower
	
	async def _parse_voice_command(self, text: str) -> Optional[Dict[str, Any]]:
		"""Parse voice command using NLP."""
		try:
			# Use OpenAI for advanced NLP if available
			if self.openai_api_key:
				command_data = await self._parse_with_openai(text)
			else:
				command_data = await self._parse_with_rules(text)
			
			if command_data:
				command_data["original_text"] = text
				command_data["timestamp"] = datetime.now().isoformat()
			
			return command_data
			
		except Exception as e:
			logger.error(f"Command parsing error: {e}")
			return None
	
	async def _parse_with_openai(self, text: str) -> Optional[Dict[str, Any]]:
		"""Parse command using OpenAI GPT."""
		try:
			client = openai.OpenAI(api_key=self.openai_api_key)
			
			prompt = f"""
			You are an AI assistant for a cash management system. Parse the following voice command and extract:
			1. Command type (one of: query_balance, get_forecast, list_transactions, create_transfer, set_alert, generate_report, show_dashboard, schedule_payment, check_risks, export_data)
			2. Parameters (account numbers, amounts, dates, etc.)
			3. Context information
			
			Voice command: "{text}"
			
			Respond in JSON format:
			{{
				"command": "command_type",
				"parameters": {{"param1": "value1", "param2": "value2"}},
				"confidence": 0.95,
				"context": "additional context"
			}}
			"""
			
			response = client.chat.completions.create(
				model="gpt-3.5-turbo",
				messages=[{"role": "user", "content": prompt}],
				max_tokens=200,
				temperature=0.1
			)
			
			result = json.loads(response.choices[0].message.content)
			
			# Validate command type
			if result.get("command") in [cmd.value for cmd in VoiceCommand]:
				return result
			
			return None
			
		except Exception as e:
			logger.error(f"OpenAI parsing error: {e}")
			return None
	
	async def _parse_with_rules(self, text: str) -> Optional[Dict[str, Any]]:
		"""Parse command using rule-based approach."""
		text_lower = text.lower()
		
		# Simple keyword-based parsing
		if any(word in text_lower for word in ["balance", "how much", "current amount"]):
			return {
				"command": VoiceCommand.QUERY_BALANCE.value,
				"parameters": self._extract_account_info(text),
				"confidence": 0.8
			}
		
		elif any(word in text_lower for word in ["forecast", "prediction", "future", "expect"]):
			return {
				"command": VoiceCommand.GET_FORECAST.value,
				"parameters": self._extract_time_period(text),
				"confidence": 0.8
			}
		
		elif any(word in text_lower for word in ["transactions", "payments", "history", "list"]):
			return {
				"command": VoiceCommand.LIST_TRANSACTIONS.value,
				"parameters": {
					**self._extract_account_info(text),
					**self._extract_time_period(text)
				},
				"confidence": 0.7
			}
		
		elif any(word in text_lower for word in ["transfer", "send", "move money"]):
			return {
				"command": VoiceCommand.CREATE_TRANSFER.value,
				"parameters": {
					**self._extract_account_info(text),
					**self._extract_amount(text)
				},
				"confidence": 0.7
			}
		
		elif any(word in text_lower for word in ["alert", "notify", "reminder"]):
			return {
				"command": VoiceCommand.SET_ALERT.value,
				"parameters": self._extract_alert_info(text),
				"confidence": 0.7
			}
		
		elif any(word in text_lower for word in ["report", "summary", "analysis"]):
			return {
				"command": VoiceCommand.GENERATE_REPORT.value,
				"parameters": self._extract_report_type(text),
				"confidence": 0.7
			}
		
		elif any(word in text_lower for word in ["dashboard", "show", "display"]):
			return {
				"command": VoiceCommand.SHOW_DASHBOARD.value,
				"parameters": {},
				"confidence": 0.8
			}
		
		return None
	
	def _extract_account_info(self, text: str) -> Dict[str, Any]:
		"""Extract account information from text."""
		import re
		
		# Look for account numbers or names
		account_patterns = [
			r"account (\w+)",
			r"from (\w+)",
			r"to (\w+)",
			r"checking",
			r"savings",
			r"business"
		]
		
		accounts = {}
		for pattern in account_patterns:
			matches = re.findall(pattern, text.lower())
			if matches:
				if "from" in pattern:
					accounts["from_account"] = matches[0]
				elif "to" in pattern:
					accounts["to_account"] = matches[0]
				else:
					accounts["account"] = matches[0]
		
		return accounts
	
	def _extract_amount(self, text: str) -> Dict[str, Any]:
		"""Extract monetary amounts from text."""
		import re
		
		# Look for amounts with currency symbols or words
		amount_patterns = [
			r"\$([0-9,]+(?:\.[0-9]{2})?)",
			r"([0-9,]+(?:\.[0-9]{2})?) dollars?",
			r"([0-9,]+) thousand",
			r"([0-9,]+) million"
		]
		
		for pattern in amount_patterns:
			matches = re.findall(pattern, text)
			if matches:
				amount_str = matches[0].replace(",", "")
				try:
					if "thousand" in pattern:
						amount = float(amount_str) * 1000
					elif "million" in pattern:
						amount = float(amount_str) * 1000000
					else:
						amount = float(amount_str)
					
					return {"amount": amount}
				except ValueError:
					continue
		
		return {}
	
	def _extract_time_period(self, text: str) -> Dict[str, Any]:
		"""Extract time period from text."""
		text_lower = text.lower()
		
		if "today" in text_lower:
			return {"period": "today"}
		elif "yesterday" in text_lower:
			return {"period": "yesterday"}
		elif "this week" in text_lower or "last 7 days" in text_lower:
			return {"period": "week"}
		elif "this month" in text_lower or "last 30 days" in text_lower:
			return {"period": "month"}
		elif "this year" in text_lower:
			return {"period": "year"}
		
		return {"period": "week"}  # Default
	
	def _extract_alert_info(self, text: str) -> Dict[str, Any]:
		"""Extract alert information from text."""
		text_lower = text.lower()
		
		alert_info = {}
		
		if "low balance" in text_lower:
			alert_info["type"] = "low_balance"
		elif "high spending" in text_lower:
			alert_info["type"] = "high_spending"
		elif "unusual activity" in text_lower:
			alert_info["type"] = "unusual_activity"
		
		# Extract threshold amounts
		amount_info = self._extract_amount(text)
		if amount_info:
			alert_info["threshold"] = amount_info["amount"]
		
		return alert_info
	
	def _extract_report_type(self, text: str) -> Dict[str, Any]:
		"""Extract report type from text."""
		text_lower = text.lower()
		
		if "cash flow" in text_lower:
			return {"report_type": "cash_flow"}
		elif "transaction" in text_lower:
			return {"report_type": "transaction_summary"}
		elif "forecast" in text_lower:
			return {"report_type": "forecast"}
		elif "risk" in text_lower:
			return {"report_type": "risk_analysis"}
		
		return {"report_type": "summary"}
	
	async def _command_processing_loop(self) -> None:
		"""Process queued voice commands."""
		while True:
			try:
				# Get command from queue
				command_data = await self.command_queue.get()
				
				# Process command
				await self._process_voice_command(command_data)
				
			except Exception as e:
				logger.error(f"Command processing error: {e}")
				await asyncio.sleep(1)
	
	async def _process_voice_command(self, command_data: Dict[str, Any]) -> None:
		"""Process a parsed voice command."""
		try:
			command_type = VoiceCommand(command_data["command"])
			handler = self.command_handlers.get(command_type)
			
			if handler:
				# Execute command handler
				response = await handler(command_data)
				
				# Send response
				if response:
					await self._speak_response(response)
			else:
				await self._speak_response("I'm sorry, I don't understand that command.")
			
		except Exception as e:
			logger.error(f"Error processing voice command: {e}")
			await self._speak_response("I encountered an error processing your request.")
	
	async def _handle_balance_query(self, command_data: Dict[str, Any]) -> str:
		"""Handle balance query command."""
		try:
			params = command_data.get("parameters", {})
			account = params.get("account", "all")
			
			# Query database for balance
			async with self.db_pool.acquire() as conn:
				if account == "all":
					query = """
						SELECT SUM(current_balance) as total_balance
						FROM cm_accounts 
						WHERE tenant_id = $1 AND active = true
					"""
					result = await conn.fetchrow(query, self.tenant_id)
					total_balance = result['total_balance'] or 0
					
					return f"Your total cash position across all accounts is ${total_balance:,.2f}"
				else:
					query = """
						SELECT current_balance, account_name
						FROM cm_accounts 
						WHERE tenant_id = $1 AND (account_id = $2 OR account_name ILIKE $3)
						AND active = true
					"""
					result = await conn.fetchrow(query, self.tenant_id, account, f"%{account}%")
					
					if result:
						return f"The balance for {result['account_name']} is ${result['current_balance']:,.2f}"
					else:
						return f"I couldn't find an account matching '{account}'"
			
		except Exception as e:
			logger.error(f"Error handling balance query: {e}")
			return "I encountered an error retrieving the balance information."
	
	async def _handle_forecast_request(self, command_data: Dict[str, Any]) -> str:
		"""Handle forecast request command."""
		try:
			params = command_data.get("parameters", {})
			period = params.get("period", "week")
			
			# Generate forecast
			if period == "week":
				days = 7
			elif period == "month":
				days = 30
			else:
				days = 7
			
			# Simplified forecast calculation
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT AVG(amount) as avg_daily_flow
					FROM cm_cash_flows 
					WHERE tenant_id = $1 AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'
				"""
				result = await conn.fetchrow(query, self.tenant_id)
				avg_flow = result['avg_daily_flow'] or 0
				
				forecast = avg_flow * days
				
				return f"Based on recent trends, I forecast a net cash flow of ${forecast:,.2f} over the next {period}."
			
		except Exception as e:
			logger.error(f"Error handling forecast request: {e}")
			return "I encountered an error generating the forecast."
	
	async def _handle_transaction_list(self, command_data: Dict[str, Any]) -> str:
		"""Handle transaction list command."""
		try:
			params = command_data.get("parameters", {})
			period = params.get("period", "week")
			account = params.get("account")
			
			# Query transactions
			async with self.db_pool.acquire() as conn:
				where_clause = "WHERE tenant_id = $1"
				query_params = [self.tenant_id]
				
				if period == "today":
					where_clause += " AND transaction_date = CURRENT_DATE"
				elif period == "week":
					where_clause += " AND transaction_date >= CURRENT_DATE - INTERVAL '7 days'"
				elif period == "month":
					where_clause += " AND transaction_date >= CURRENT_DATE - INTERVAL '30 days'"
				
				if account:
					where_clause += f" AND account_id = ${len(query_params) + 1}"
					query_params.append(account)
				
				query = f"""
					SELECT COUNT(*) as count, SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as inflow,
						   SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as outflow
					FROM cm_cash_flows 
					{where_clause}
				"""
				
				result = await conn.fetchrow(query, *query_params)
				
				count = result['count'] or 0
				inflow = result['inflow'] or 0
				outflow = result['outflow'] or 0
				
				return f"In the {period}, you had {count} transactions with ${inflow:,.2f} in inflows and ${outflow:,.2f} in outflows."
			
		except Exception as e:
			logger.error(f"Error handling transaction list: {e}")
			return "I encountered an error retrieving transaction information."
	
	async def _handle_transfer_creation(self, command_data: Dict[str, Any]) -> str:
		"""Handle transfer creation command."""
		# For security, transfers should require additional authentication
		return "Transfer requests require additional security verification. Please use the web interface to complete this transaction."
	
	async def _handle_alert_setup(self, command_data: Dict[str, Any]) -> str:
		"""Handle alert setup command."""
		try:
			params = command_data.get("parameters", {})
			alert_type = params.get("type", "low_balance")
			threshold = params.get("threshold")
			
			if not threshold:
				return "Please specify a threshold amount for the alert."
			
			# Create alert (simplified)
			return f"I've set up a {alert_type.replace('_', ' ')} alert with a threshold of ${threshold:,.2f}. You'll be notified when this condition is met."
			
		except Exception as e:
			logger.error(f"Error handling alert setup: {e}")
			return "I encountered an error setting up the alert."
	
	async def _handle_report_generation(self, command_data: Dict[str, Any]) -> str:
		"""Handle report generation command."""
		try:
			params = command_data.get("parameters", {})
			report_type = params.get("report_type", "summary")
			
			return f"I'm generating your {report_type.replace('_', ' ')} report. It will be available in your dashboard shortly."
			
		except Exception as e:
			logger.error(f"Error handling report generation: {e}")
			return "I encountered an error generating the report."
	
	async def _handle_dashboard_request(self, command_data: Dict[str, Any]) -> str:
		"""Handle dashboard request command."""
		return "I'm opening your cash management dashboard now. You can see your latest financial overview there."
	
	async def _handle_payment_scheduling(self, command_data: Dict[str, Any]) -> str:
		"""Handle payment scheduling command."""
		return "Payment scheduling requires additional verification. Please use the secure web interface to schedule payments."
	
	async def _handle_risk_check(self, command_data: Dict[str, Any]) -> str:
		"""Handle risk check command."""
		try:
			# Simplified risk assessment
			async with self.db_pool.acquire() as conn:
				query = """
					SELECT COUNT(*) as low_balance_accounts
					FROM cm_accounts 
					WHERE tenant_id = $1 AND current_balance < minimum_balance AND active = true
				"""
				result = await conn.fetchrow(query, self.tenant_id)
				
				low_balance_count = result['low_balance_accounts'] or 0
				
				if low_balance_count > 0:
					return f"I found {low_balance_count} account(s) with low balances that require attention."
				else:
					return "All accounts are currently above their minimum balance requirements. No immediate risks detected."
			
		except Exception as e:
			logger.error(f"Error handling risk check: {e}")
			return "I encountered an error checking for risks."
	
	async def _handle_data_export(self, command_data: Dict[str, Any]) -> str:
		"""Handle data export command."""
		return "I'm preparing your data export. You'll receive an email with the download link shortly."
	
	async def _speak_response(
		self,
		text: str,
		session_id: Optional[str] = None
	) -> None:
		"""Speak response using TTS."""
		try:
			if self.config.response_mode in [ResponseMode.VOICE_ONLY, ResponseMode.VOICE_AND_TEXT, ResponseMode.VISUAL_AND_VOICE]:
				# Use TTS to speak the response
				if self.tts_engine:
					self.tts_engine.say(text)
					self.tts_engine.runAndWait()
			
			# Log the response
			logger.info(f"Voice response: {text}")
			
			# Update session if provided
			if session_id and session_id in self.active_sessions:
				session = self.active_sessions[session_id]
				session.conversation_history.append({
					"type": "response",
					"text": text,
					"timestamp": datetime.now().isoformat()
				})
				session.last_interaction = datetime.now()
			
		except Exception as e:
			logger.error(f"Error speaking response: {e}")
	
	async def _session_cleanup_loop(self) -> None:
		"""Clean up inactive voice sessions."""
		while True:
			try:
				now = datetime.now()
				inactive_sessions = []
				
				for session_id, session in self.active_sessions.items():
					if (now - session.last_interaction).total_seconds() > self.config.voice_timeout_seconds:
						inactive_sessions.append(session_id)
				
				for session_id in inactive_sessions:
					await self.end_voice_session(session_id)
				
				await asyncio.sleep(60)  # Check every minute
				
			except Exception as e:
				logger.error(f"Session cleanup error: {e}")
				await asyncio.sleep(60)
	
	async def end_voice_session(self, session_id: str) -> None:
		"""End a voice session."""
		try:
			if session_id in self.active_sessions:
				session = self.active_sessions[session_id]
				session.active = False
				
				await self._speak_response("Thank you for using APG voice interface. Goodbye!", session_id)
				
				# Save session data if needed
				await self._save_session_data(session)
				
				del self.active_sessions[session_id]
				
				logger.info(f"Ended voice session: {session_id}")
			
		except Exception as e:
			logger.error(f"Error ending voice session: {e}")
	
	async def _save_session_data(self, session: VoiceSession) -> None:
		"""Save session data for analytics."""
		try:
			# Save session interaction data for improving the voice interface
			session_data = {
				"session_id": session.session_id,
				"user_id": session.user_id,
				"language": session.language.value,
				"duration_seconds": (session.last_interaction - session.start_time).total_seconds(),
				"interaction_count": len(session.conversation_history),
				"conversation_history": session.conversation_history[-10:]  # Keep last 10 interactions
			}
			
			# In a real implementation, this would be saved to a database
			logger.debug(f"Session data saved for {session.session_id}")
			
		except Exception as e:
			logger.error(f"Error saving session data: {e}")
	
	def enable_listening(self) -> None:
		"""Enable voice listening."""
		self.listening_active = True
		logger.info("Voice listening enabled")
	
	def disable_listening(self) -> None:
		"""Disable voice listening."""
		self.listening_active = False
		logger.info("Voice listening disabled")
	
	async def cleanup(self) -> None:
		"""Cleanup voice interface resources."""
		# End all active sessions
		for session_id in list(self.active_sessions.keys()):
			await self.end_voice_session(session_id)
		
		# Stop listening
		self.disable_listening()
		
		# Cleanup TTS engine
		if self.tts_engine:
			self.tts_engine.stop()
		
		logger.info("Voice interface cleanup completed")

# Global voice interface instance
_voice_interface: Optional[VoiceInterfaceIntegration] = None

async def get_voice_interface(
	tenant_id: str,
	db_pool: asyncpg.Pool,
	config: Optional[VoiceConfiguration] = None
) -> VoiceInterfaceIntegration:
	"""Get or create voice interface instance."""
	global _voice_interface
	
	if _voice_interface is None or _voice_interface.tenant_id != tenant_id:
		_voice_interface = VoiceInterfaceIntegration(tenant_id, db_pool, config)
		await _voice_interface.initialize()
	
	return _voice_interface

if __name__ == "__main__":
	async def main():
		# Example usage would require a real database connection
		print("Voice Interface Integration initialized")
		print("This module provides advanced voice-driven cash management")
	
	asyncio.run(main())