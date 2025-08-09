"""
Conversational Payment Interface Service
Natural language payment processing with voice support and intelligent automation.

Copyright (c) 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)


class ConversationLanguage(str, Enum):
	ENGLISH = "en"
	SPANISH = "es"
	FRENCH = "fr"
	GERMAN = "de"
	ITALIAN = "it"
	PORTUGUESE = "pt"
	DUTCH = "nl"
	JAPANESE = "ja"
	CHINESE = "zh"
	KOREAN = "ko"
	ARABIC = "ar"
	RUSSIAN = "ru"
	SWAHILI = "sw"


class ConversationChannel(str, Enum):
	TEXT_CHAT = "text_chat"
	VOICE_CALL = "voice_call"
	VIDEO_CALL = "video_call"
	SMS = "sms"
	EMAIL = "email"
	WHATSAPP = "whatsapp"
	TELEGRAM = "telegram"
	WEB_WIDGET = "web_widget"
	MOBILE_APP = "mobile_app"


class IntentType(str, Enum):
	MAKE_PAYMENT = "make_payment"
	CHECK_BALANCE = "check_balance"
	PAYMENT_HISTORY = "payment_history"
	REFUND_REQUEST = "refund_request"
	DISPUTE_PAYMENT = "dispute_payment"
	UPDATE_PAYMENT_METHOD = "update_payment_method"
	CANCEL_SUBSCRIPTION = "cancel_subscription"
	PAYMENT_HELP = "payment_help"
	TECHNICAL_SUPPORT = "technical_support"
	ACCOUNT_INQUIRY = "account_inquiry"


class ConversationState(str, Enum):
	INITIATED = "initiated"
	AUTHENTICATING = "authenticating"
	PROCESSING_INTENT = "processing_intent"
	COLLECTING_DETAILS = "collecting_details"
	CONFIRMING_ACTION = "confirming_action"
	EXECUTING_PAYMENT = "executing_payment"
	COMPLETED = "completed"
	FAILED = "failed"
	ESCALATED = "escalated"


class BiometricType(str, Enum):
	VOICE_PRINT = "voice_print"
	FACE_RECOGNITION = "face_recognition"
	FINGERPRINT = "fingerprint"
	BEHAVIORAL_PATTERN = "behavioral_pattern"


class ConversationMessage(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	conversation_id: str
	sender_type: str  # "user" or "assistant"
	content: str
	message_type: str = "text"  # text, voice, image, video
	language: ConversationLanguage = ConversationLanguage.ENGLISH
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
	intent_detected: Optional[IntentType] = None
	entities_extracted: Dict[str, Any] = Field(default_factory=dict)
	response_time_ms: Optional[int] = None


class ConversationSession(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	customer_id: Optional[str] = None
	channel: ConversationChannel
	language: ConversationLanguage = ConversationLanguage.ENGLISH
	state: ConversationState = ConversationState.INITIATED
	context: Dict[str, Any] = Field(default_factory=dict)
	authenticated: bool = False
	authentication_method: Optional[str] = None
	current_intent: Optional[IntentType] = None
	payment_in_progress: Optional[str] = None
	escalation_reason: Optional[str] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_activity: datetime = Field(default_factory=datetime.utcnow)
	session_duration: Optional[int] = None
	satisfaction_score: Optional[float] = None


class PaymentIntent(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	conversation_id: str
	intent_type: IntentType
	amount: Optional[Decimal] = None
	currency: str = "USD"
	recipient: Optional[str] = None
	payment_method: Optional[str] = None
	description: Optional[str] = None
	metadata: Dict[str, Any] = Field(default_factory=dict)
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
	extracted_entities: Dict[str, Any] = Field(default_factory=dict)
	requires_confirmation: bool = True
	confirmed: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)


class VoiceProfile(BaseModel):
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(default_factory=uuid7str)
	customer_id: str
	voice_features: Dict[str, Any] = Field(default_factory=dict)
	biometric_template: Optional[str] = None
	enrollment_quality: float = Field(default=0.0, ge=0.0, le=1.0)
	verification_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
	last_verification: Optional[datetime] = None
	verification_count: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationalPaymentService:
	"""Advanced conversational payment interface with NLP and voice processing."""
	
	def __init__(self):
		self._conversations: Dict[str, ConversationSession] = {}
		self._messages: Dict[str, List[ConversationMessage]] = {}
		self._payment_intents: Dict[str, PaymentIntent] = {}
		self._voice_profiles: Dict[str, VoiceProfile] = {}
		
		# NLP models and processors (simulated)
		self._intent_classifiers: Dict[ConversationLanguage, Any] = {}
		self._entity_extractors: Dict[ConversationLanguage, Any] = {}
		self._sentiment_analyzers: Dict[ConversationLanguage, Any] = {}
		
		# Voice processing components
		self._speech_to_text_engines: Dict[ConversationLanguage, Any] = {}
		self._text_to_speech_engines: Dict[ConversationLanguage, Any] = {}
		self._voice_biometric_engine: Optional[Any] = None
		
		# Conversation templates and responses
		self._response_templates: Dict[str, Dict[ConversationLanguage, List[str]]] = {}
		self._conversation_flows: Dict[IntentType, Dict[str, Any]] = {}
		
		# Initialize default configuration
		asyncio.create_task(self._initialize_conversational_ai())
	
	async def _initialize_conversational_ai(self) -> None:
		"""Initialize conversational AI models and templates."""
		# Initialize response templates
		self._response_templates = {
			"greeting": {
				ConversationLanguage.ENGLISH: [
					"Hello! I'm your payment assistant. How can I help you today?",
					"Hi there! I'm here to help with your payments. What can I do for you?",
					"Welcome! I can help you with payments, refunds, and account questions. What do you need?"
				],
				ConversationLanguage.SPANISH: [
					"¡Hola! Soy tu asistente de pagos. ¿Cómo puedo ayudarte hoy?",
					"¡Hola! Estoy aquí para ayudarte con tus pagos. ¿Qué puedo hacer por ti?",
					"¡Bienvenido! Puedo ayudarte con pagos, reembolsos y preguntas de cuenta. ¿Qué necesitas?"
				],
				ConversationLanguage.FRENCH: [
					"Bonjour! Je suis votre assistant de paiement. Comment puis-je vous aider aujourd'hui?",
					"Salut! Je suis là pour vous aider avec vos paiements. Que puis-je faire pour vous?",
					"Bienvenue! Je peux vous aider avec les paiements, remboursements et questions de compte. Que voulez-vous?"
				],
				ConversationLanguage.SWAHILI: [
					"Hujambo! Mimi ni msaidizi wako wa malipo. Ninawezaje kukusaidia leo?",
					"Habari! Nipo hapa kukusaidia na malipo yako. Ninaweza kufanya nini kwa ajili yako?",
					"Karibu! Ninaweza kukusaidia na malipo, marejesho na maswali ya akaunti. Unahitaji nini?"
				]
			},
			"payment_confirmation": {
				ConversationLanguage.ENGLISH: [
					"I understand you want to make a payment of {amount} {currency} to {recipient}. Is that correct?",
					"Let me confirm: you're paying {amount} {currency} to {recipient}. Should I proceed?",
					"Just to verify: {amount} {currency} payment to {recipient}. Correct?"
				],
				ConversationLanguage.SPANISH: [
					"Entiendo que quieres hacer un pago de {amount} {currency} a {recipient}. ¿Es correcto?",
					"Déjame confirmar: estás pagando {amount} {currency} a {recipient}. ¿Debo proceder?",
					"Solo para verificar: pago de {amount} {currency} a {recipient}. ¿Correcto?"
				],
				ConversationLanguage.SWAHILI: [
					"Naelewa unataka kulipa {amount} {currency} kwa {recipient}. Je, ni sahihi?",
					"Hebu nikakisi: unalipa {amount} {currency} kwa {recipient}. Je, niendelee?",
					"Tu kukagua: malipo ya {amount} {currency} kwa {recipient}. Sahihi?"
				]
			},
			"authentication_request": {
				ConversationLanguage.ENGLISH: [
					"For security, I need to verify your identity. Can you provide your account PIN or use voice verification?",
					"To protect your account, please authenticate using your PIN, fingerprint, or voice.",
					"Security check: Please verify your identity to continue with this payment."
				],
				ConversationLanguage.SPANISH: [
					"Por seguridad, necesito verificar tu identidad. ¿Puedes proporcionar tu PIN de cuenta o usar verificación de voz?",
					"Para proteger tu cuenta, por favor autentícate usando tu PIN, huella digital o voz.",
					"Verificación de seguridad: Por favor verifica tu identidad para continuar con este pago."
				],
				ConversationLanguage.SWAHILI: [
					"Kwa usalama, nahitaji kuthibitisha utambulisho wako. Je, unaweza kutoa PIN yako au kutumia uthibitisho wa sauti?",
					"Kulinda akaunti yako, tafadhali jithibitishe kwa kutumia PIN, alama ya kidole, au sauti.",
					"Ukaguzi wa usalama: Tafadhali thibitisha utambulisho wako kuendelea na malipo haya."
				]
			},
			"payment_success": {
				ConversationLanguage.ENGLISH: [
					"Great! Your payment of {amount} {currency} to {recipient} was successful. Transaction ID: {transaction_id}",
					"Payment completed! {amount} {currency} has been sent to {recipient}. Reference: {transaction_id}",
					"Success! Your {amount} {currency} payment is complete. Transaction: {transaction_id}"
				],
				ConversationLanguage.SPANISH: [
					"¡Excelente! Tu pago de {amount} {currency} a {recipient} fue exitoso. ID de transacción: {transaction_id}",
					"¡Pago completado! {amount} {currency} ha sido enviado a {recipient}. Referencia: {transaction_id}",
					"¡Éxito! Tu pago de {amount} {currency} está completo. Transacción: {transaction_id}"
				],
				ConversationLanguage.SWAHILI: [
					"Vizuri! Malipo yako ya {amount} {currency} kwa {recipient} yamefanikiwa. Kitambulisho cha muamala: {transaction_id}",
					"Malipo yamekamilika! {amount} {currency} imetumwa kwa {recipient}. Marejeleo: {transaction_id}",
					"Mafanikio! Malipo yako ya {amount} {currency} yamekamilika. Muamala: {transaction_id}"
				]
			},
			"payment_error": {
				ConversationLanguage.ENGLISH: [
					"I'm sorry, there was an issue processing your payment. Error: {error_message}. Would you like to try again?",
					"Payment failed: {error_message}. Let me help you resolve this issue.",
					"Unfortunately, your payment couldn't be completed: {error_message}. How can I help fix this?"
				],
				ConversationLanguage.SPANISH: [
					"Lo siento, hubo un problema procesando tu pago. Error: {error_message}. ¿Te gustaría intentar de nuevo?",
					"Pago falló: {error_message}. Déjame ayudarte a resolver este problema.",
					"Desafortunadamente, tu pago no pudo completarse: {error_message}. ¿Cómo puedo ayudar a solucionarlo?"
				],
				ConversationLanguage.SWAHILI: [
					"Samahani, kulikuwa na tatizo katika kuchakata malipo yako. Hitilafu: {error_message}. Je, ungependa kujaribu tena?",
					"Malipo yameshindwa: {error_message}. Acha nikusaidie kutatua tatizo hili.",
					"Kwa bahati mbaya, malipo yako hayakuweza kukamilika: {error_message}. Ninawezaje kusaidia kurekebisha hii?"
				]
			}
		}
		
		# Initialize conversation flows
		self._conversation_flows = {
			IntentType.MAKE_PAYMENT: {
				"required_entities": ["amount", "currency", "recipient"],
				"optional_entities": ["payment_method", "description"],
				"confirmation_required": True,
				"authentication_required": True,
				"steps": [
					"extract_payment_details",
					"validate_payment_info",
					"authenticate_user",
					"confirm_payment",
					"process_payment",
					"provide_confirmation"
				]
			},
			IntentType.CHECK_BALANCE: {
				"required_entities": [],
				"optional_entities": ["account_type"],
				"confirmation_required": False,
				"authentication_required": True,
				"steps": [
					"authenticate_user",
					"retrieve_balance",
					"format_balance_response"
				]
			},
			IntentType.REFUND_REQUEST: {
				"required_entities": ["transaction_id"],
				"optional_entities": ["reason", "amount"],
				"confirmation_required": True,
				"authentication_required": True,
				"steps": [
					"extract_refund_details",
					"validate_transaction",
					"authenticate_user",
					"confirm_refund",
					"process_refund",
					"provide_confirmation"
				]
			}
		}
		
		logger.info("Conversational AI models and templates initialized")
	
	async def start_conversation(
		self,
		channel: ConversationChannel,
		customer_id: Optional[str] = None,
		language: ConversationLanguage = ConversationLanguage.ENGLISH,
		context: Optional[Dict[str, Any]] = None
	) -> str:
		"""Start a new conversation session."""
		conversation = ConversationSession(
			customer_id=customer_id,
			channel=channel,
			language=language,
			context=context or {}
		)
		
		self._conversations[conversation.id] = conversation
		self._messages[conversation.id] = []
		
		# Send initial greeting
		greeting = await self._get_response_template("greeting", language)
		await self._add_assistant_message(conversation.id, greeting, language)
		
		logger.info(f"Started conversation {conversation.id} for customer {customer_id}")
		return conversation.id
	
	async def process_user_message(
		self,
		conversation_id: str,
		message_content: str,
		message_type: str = "text",
		audio_data: Optional[bytes] = None
	) -> Dict[str, Any]:
		"""Process incoming user message and generate response."""
		if conversation_id not in self._conversations:
			raise ValueError(f"Conversation {conversation_id} not found")
		
		conversation = self._conversations[conversation_id]
		conversation.last_activity = datetime.utcnow()
		
		# Convert speech to text if audio message
		if message_type == "voice" and audio_data:
			message_content = await self._speech_to_text(audio_data, conversation.language)
		
		# Process the message
		start_time = datetime.utcnow()
		
		# Add user message
		user_message = await self._add_user_message(
			conversation_id, message_content, message_type, conversation.language
		)
		
		# Extract intent and entities
		intent_result = await self._extract_intent_and_entities(
			message_content, conversation.language, conversation.context
		)
		
		user_message.intent_detected = intent_result.get("intent")
		user_message.entities_extracted = intent_result.get("entities", {})
		user_message.confidence_score = intent_result.get("confidence", 0.0)
		
		# Update conversation state and context
		await self._update_conversation_state(conversation, intent_result)
		
		# Generate response
		response = await self._generate_response(conversation, intent_result)
		
		# Add assistant response
		assistant_message = await self._add_assistant_message(
			conversation_id, response["content"], conversation.language
		)
		
		# Calculate response time
		end_time = datetime.utcnow()
		response_time_ms = int((end_time - start_time).total_seconds() * 1000)
		assistant_message.response_time_ms = response_time_ms
		
		return {
			"conversation_id": conversation_id,
			"response": response["content"],
			"response_type": response.get("type", "text"),
			"audio_response": response.get("audio_data"),
			"intent_detected": user_message.intent_detected,
			"confidence_score": user_message.confidence_score,
			"conversation_state": conversation.state,
			"requires_action": response.get("requires_action", False),
			"action_type": response.get("action_type"),
			"response_time_ms": response_time_ms
		}
	
	async def _speech_to_text(self, audio_data: bytes, language: ConversationLanguage) -> str:
		"""Convert speech audio to text."""
		# Simulate speech recognition (in real implementation, use services like Google Speech-to-Text, Azure Speech, etc.)
		
		# Simulate processing delay
		await asyncio.sleep(0.5)
		
		# Mock STT results based on common payment phrases
		mock_phrases = [
			"I want to send fifty dollars to John Smith",
			"Can you help me make a payment",
			"What's my account balance",
			"I need to request a refund for transaction twelve three four",
			"Transfer one hundred euros to my savings account",
			"Pay the electricity bill",
			"Send money to Maria Garcia"
		]
		
		import random
		return random.choice(mock_phrases)
	
	async def _text_to_speech(self, text: str, language: ConversationLanguage) -> bytes:
		"""Convert text to speech audio."""
		# Simulate text-to-speech conversion
		await asyncio.sleep(0.3)
		
		# Return mock audio data (in real implementation, use TTS services)
		return b"mock_audio_data_" + text.encode()[:50]
	
	async def _extract_intent_and_entities(
		self,
		message: str,
		language: ConversationLanguage,
		context: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Extract intent and entities from user message using NLP."""
		message_lower = message.lower()
		
		# Intent classification (simplified rule-based for demo)
		intent = None
		confidence = 0.0
		entities = {}
		
		# Payment intent patterns
		payment_patterns = [
			r'(?:send|transfer|pay|give)\s+.*?(\d+(?:\.\d{2})?)\s*(dollars?|euros?|pounds?|usd|eur|gbp|ksh)',
			r'make\s+a\s+payment',
			r'i\s+want\s+to\s+pay',
			r'pay\s+(?:the\s+)?(.+?)(?:\s+bill)?'
		]
		
		for pattern in payment_patterns:
			match = re.search(pattern, message_lower)
			if match:
				intent = IntentType.MAKE_PAYMENT
				confidence = 0.9
				
				# Extract amount and currency
				amount_match = re.search(r'(\d+(?:\.\d{2})?)', message_lower)
				if amount_match:
					entities["amount"] = float(amount_match.group(1))
				
				currency_match = re.search(r'(dollars?|euros?|pounds?|usd|eur|gbp|ksh)', message_lower)
				if currency_match:
					currency_text = currency_match.group(1)
					entities["currency"] = self._normalize_currency(currency_text)
				
				# Extract recipient
				recipient_patterns = [
					r'to\s+([A-Za-z\s]+?)(?:\s|$)',
					r'for\s+([A-Za-z\s]+?)(?:\s|$)'
				]
				for rec_pattern in recipient_patterns:
					rec_match = re.search(rec_pattern, message)
					if rec_match:
						entities["recipient"] = rec_match.group(1).strip()
						break
				
				break
		
		# Balance inquiry patterns
		balance_patterns = [
			r'(?:check|what\'?s)\s+my\s+balance',
			r'account\s+balance',
			r'how\s+much\s+(?:money\s+)?(?:do\s+)?i\s+have'
		]
		
		for pattern in balance_patterns:
			if re.search(pattern, message_lower):
				intent = IntentType.CHECK_BALANCE
				confidence = 0.95
				break
		
		# Refund request patterns
		refund_patterns = [
			r'(?:request|need|want)\s+a?\s*refund',
			r'refund\s+(?:for\s+)?(?:transaction\s+)?(\w+)',
			r'cancel\s+(?:the\s+)?payment'
		]
		
		for pattern in refund_patterns:
			match = re.search(pattern, message_lower)
			if match:
				intent = IntentType.REFUND_REQUEST
				confidence = 0.85
				
				# Extract transaction ID
				tx_match = re.search(r'(?:transaction\s+)?([a-z0-9\-_]+)', message_lower)
				if tx_match:
					entities["transaction_id"] = tx_match.group(1)
				break
		
		# Help patterns
		help_patterns = [
			r'help',
			r'what\s+can\s+you\s+do',
			r'how\s+(?:do\s+)?(?:i|can)',
			r'support'
		]
		
		for pattern in help_patterns:
			if re.search(pattern, message_lower):
				intent = IntentType.PAYMENT_HELP
				confidence = 0.8
				break
		
		# If no specific intent found, default to help
		if intent is None:
			intent = IntentType.PAYMENT_HELP
			confidence = 0.3
		
		# Extract common entities
		entities.update(await self._extract_common_entities(message))
		
		return {
			"intent": intent,
			"confidence": confidence,
			"entities": entities,
			"original_message": message
		}
	
	def _normalize_currency(self, currency_text: str) -> str:
		"""Normalize currency text to standard codes."""
		currency_map = {
			"dollar": "USD", "dollars": "USD", "usd": "USD",
			"euro": "EUR", "euros": "EUR", "eur": "EUR",
			"pound": "GBP", "pounds": "GBP", "gbp": "GBP",
			"shilling": "KES", "shillings": "KES", "ksh": "KES"
		}
		return currency_map.get(currency_text.lower(), "USD")
	
	async def _extract_common_entities(self, message: str) -> Dict[str, Any]:
		"""Extract common entities like names, numbers, dates."""
		entities = {}
		
		# Extract names (simple pattern)
		name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
		name_matches = re.findall(name_pattern, message)
		if name_matches:
			entities["person_names"] = name_matches
		
		# Extract phone numbers
		phone_pattern = r'\b(\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})\b'
		phone_matches = re.findall(phone_pattern, message)
		if phone_matches:
			entities["phone_numbers"] = phone_matches
		
		# Extract email addresses
		email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
		email_matches = re.findall(email_pattern, message)
		if email_matches:
			entities["email_addresses"] = email_matches
		
		return entities
	
	async def _update_conversation_state(
		self,
		conversation: ConversationSession,
		intent_result: Dict[str, Any]
	) -> None:
		"""Update conversation state based on intent and current context."""
		intent = intent_result.get("intent")
		entities = intent_result.get("entities", {})
		confidence = intent_result.get("confidence", 0.0)
		
		# Update current intent if confidence is high enough
		if confidence > 0.7:
			conversation.current_intent = intent
		
		# Update context with extracted entities
		conversation.context.update(entities)
		
		# State transitions based on intent and current state
		if conversation.state == ConversationState.INITIATED:
			if intent in [IntentType.MAKE_PAYMENT, IntentType.REFUND_REQUEST]:
				if not conversation.authenticated:
					conversation.state = ConversationState.AUTHENTICATING
				else:
					conversation.state = ConversationState.COLLECTING_DETAILS
			elif intent == IntentType.CHECK_BALANCE:
				if not conversation.authenticated:
					conversation.state = ConversationState.AUTHENTICATING
				else:
					conversation.state = ConversationState.PROCESSING_INTENT
			else:
				conversation.state = ConversationState.PROCESSING_INTENT
		
		elif conversation.state == ConversationState.AUTHENTICATING:
			# Check if authentication info was provided
			if self._has_authentication_info(entities):
				conversation.state = ConversationState.COLLECTING_DETAILS
		
		elif conversation.state == ConversationState.COLLECTING_DETAILS:
			# Check if we have all required details
			if await self._has_required_details(conversation):
				conversation.state = ConversationState.CONFIRMING_ACTION
		
		elif conversation.state == ConversationState.CONFIRMING_ACTION:
			if self._is_confirmation(intent_result.get("original_message", "")):
				conversation.state = ConversationState.EXECUTING_PAYMENT
		
		logger.debug(f"Updated conversation {conversation.id} state to {conversation.state}")
	
	def _has_authentication_info(self, entities: Dict[str, Any]) -> bool:
		"""Check if authentication information was provided."""
		auth_indicators = ["pin", "password", "voice", "fingerprint", "biometric"]
		message_text = entities.get("original_message", "").lower()
		
		return any(indicator in message_text for indicator in auth_indicators)
	
	async def _has_required_details(self, conversation: ConversationSession) -> bool:
		"""Check if conversation has all required details for the current intent."""
		intent = conversation.current_intent
		context = conversation.context
		
		if intent == IntentType.MAKE_PAYMENT:
			required = ["amount", "recipient"]
			return all(field in context for field in required)
		
		elif intent == IntentType.REFUND_REQUEST:
			return "transaction_id" in context
		
		elif intent == IntentType.CHECK_BALANCE:
			return True  # No additional details required
		
		return True
	
	def _is_confirmation(self, message: str) -> bool:
		"""Check if message is a confirmation."""
		confirmation_patterns = [
			r'\b(yes|yeah|yep|correct|right|proceed|confirm|ok|okay)\b',
			r'\b(si|oui|ja|sim)\b'  # Multi-language confirmations
		]
		
		message_lower = message.lower()
		return any(re.search(pattern, message_lower) for pattern in confirmation_patterns)
	
	async def _generate_response(
		self,
		conversation: ConversationSession,
		intent_result: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate appropriate response based on conversation state and intent."""
		intent = intent_result.get("intent")
		entities = intent_result.get("entities", {})
		language = conversation.language
		
		if conversation.state == ConversationState.AUTHENTICATING:
			return await self._generate_authentication_response(conversation, language)
		
		elif conversation.state == ConversationState.COLLECTING_DETAILS:
			return await self._generate_detail_collection_response(conversation, intent, entities, language)
		
		elif conversation.state == ConversationState.CONFIRMING_ACTION:
			return await self._generate_confirmation_response(conversation, language)
		
		elif conversation.state == ConversationState.EXECUTING_PAYMENT:
			return await self._execute_payment_action(conversation, language)
		
		elif conversation.state == ConversationState.PROCESSING_INTENT:
			return await self._process_intent_action(conversation, intent, entities, language)
		
		else:
			# Default help response
			return await self._generate_help_response(conversation, language)
	
	async def _generate_authentication_response(
		self,
		conversation: ConversationSession,
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Generate authentication request response."""
		template = await self._get_response_template("authentication_request", language)
		
		return {
			"content": template,
			"type": "text",
			"requires_action": True,
			"action_type": "authentication"
		}
	
	async def _generate_detail_collection_response(
		self,
		conversation: ConversationSession,
		intent: IntentType,
		entities: Dict[str, Any],
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Generate response to collect missing details."""
		context = conversation.context
		
		if intent == IntentType.MAKE_PAYMENT:
			if "amount" not in context:
				responses = {
					ConversationLanguage.ENGLISH: "How much would you like to send?",
					ConversationLanguage.SPANISH: "¿Cuánto te gustaría enviar?",
					ConversationLanguage.SWAHILI: "Unataka kutuma kiasi gani?"
				}
				return {"content": responses.get(language, responses[ConversationLanguage.ENGLISH]), "type": "text"}
			
			elif "recipient" not in context:
				responses = {
					ConversationLanguage.ENGLISH: "Who would you like to send the money to?",
					ConversationLanguage.SPANISH: "¿A quién te gustaría enviar el dinero?",
					ConversationLanguage.SWAHILI: "Unataka kumtumia nani pesa?"
				}
				return {"content": responses.get(language, responses[ConversationLanguage.ENGLISH]), "type": "text"}
			
			elif "currency" not in context:
				responses = {
					ConversationLanguage.ENGLISH: "What currency should I use? (USD, EUR, KES, etc.)",
					ConversationLanguage.SPANISH: "¿Qué moneda debo usar? (USD, EUR, KES, etc.)",
					ConversationLanguage.SWAHILI: "Sarafu gani nitumie? (USD, EUR, KES, n.k.)"
				}
				return {"content": responses.get(language, responses[ConversationLanguage.ENGLISH]), "type": "text"}
		
		elif intent == IntentType.REFUND_REQUEST:
			if "transaction_id" not in context:
				responses = {
					ConversationLanguage.ENGLISH: "Please provide the transaction ID for the refund.",
					ConversationLanguage.SPANISH: "Por favor proporciona el ID de transacción para el reembolso.",
					ConversationLanguage.SWAHILI: "Tafadhali toa kitambulisho cha muamala kwa ajili ya kurejesha."
				}
				return {"content": responses.get(language, responses[ConversationLanguage.ENGLISH]), "type": "text"}
		
		# If we have all details, move to confirmation
		return await self._generate_confirmation_response(conversation, language)
	
	async def _generate_confirmation_response(
		self,
		conversation: ConversationSession,
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Generate payment confirmation response."""
		intent = conversation.current_intent
		context = conversation.context
		
		if intent == IntentType.MAKE_PAYMENT:
			template = await self._get_response_template("payment_confirmation", language)
			content = template.format(
				amount=context.get("amount", ""),
				currency=context.get("currency", "USD"),
				recipient=context.get("recipient", "")
			)
			
			return {
				"content": content,
				"type": "text",
				"requires_action": True,
				"action_type": "confirmation"
			}
		
		elif intent == IntentType.REFUND_REQUEST:
			responses = {
				ConversationLanguage.ENGLISH: f"I'll process your refund request for transaction {context.get('transaction_id', '')}. Confirm?",
				ConversationLanguage.SPANISH: f"Procesaré tu solicitud de reembolso para la transacción {context.get('transaction_id', '')}. ¿Confirmar?",
				ConversationLanguage.SWAHILI: f"Nitachakata ombi lako la kurejesha kwa muamala {context.get('transaction_id', '')}. Thibitisha?"
			}
			return {"content": responses.get(language, responses[ConversationLanguage.ENGLISH]), "type": "text"}
		
		return {"content": "Please confirm to proceed.", "type": "text"}
	
	async def _execute_payment_action(
		self,
		conversation: ConversationSession,
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Execute the payment action."""
		context = conversation.context
		intent = conversation.current_intent
		
		try:
			if intent == IntentType.MAKE_PAYMENT:
				# Create payment intent
				payment_intent = PaymentIntent(
					conversation_id=conversation.id,
					intent_type=intent,
					amount=Decimal(str(context.get("amount", 0))),
					currency=context.get("currency", "USD"),
					recipient=context.get("recipient"),
					confirmed=True
				)
				
				self._payment_intents[payment_intent.id] = payment_intent
				
				# Simulate payment processing
				await asyncio.sleep(1)  # Simulate processing time
				
				# Mock success (90% success rate)
				import random
				if random.random() < 0.9:
					transaction_id = uuid7str()
					conversation.state = ConversationState.COMPLETED
					
					template = await self._get_response_template("payment_success", language)
					content = template.format(
						amount=context.get("amount", ""),
						currency=context.get("currency", "USD"),
						recipient=context.get("recipient", ""),
						transaction_id=transaction_id
					)
					
					return {
						"content": content,
						"type": "text",
						"requires_action": False,
						"transaction_id": transaction_id
					}
				else:
					conversation.state = ConversationState.FAILED
					template = await self._get_response_template("payment_error", language)
					content = template.format(error_message="Insufficient funds")
					
					return {
						"content": content,
						"type": "text",
						"requires_action": True,
						"action_type": "retry"
					}
			
			elif intent == IntentType.REFUND_REQUEST:
				# Process refund
				await asyncio.sleep(0.5)
				conversation.state = ConversationState.COMPLETED
				
				responses = {
					ConversationLanguage.ENGLISH: f"Refund processed successfully for transaction {context.get('transaction_id', '')}. You'll receive the funds within 3-5 business days.",
					ConversationLanguage.SPANISH: f"Reembolso procesado exitosamente para la transacción {context.get('transaction_id', '')}. Recibirás los fondos en 3-5 días hábiles.",
					ConversationLanguage.SWAHILI: f"Kurejesha kumechakatwa kwa mafanikio kwa muamala {context.get('transaction_id', '')}. Utapokea fedha ndani ya siku 3-5 za kazi."
				}
				
				return {
					"content": responses.get(language, responses[ConversationLanguage.ENGLISH]),
					"type": "text",
					"requires_action": False
				}
		
		except Exception as e:
			logger.error(f"Error executing payment action: {str(e)}")
			conversation.state = ConversationState.FAILED
			
			template = await self._get_response_template("payment_error", language)
			content = template.format(error_message=str(e))
			
			return {
				"content": content,
				"type": "text",
				"requires_action": True,
				"action_type": "retry"
			}
	
	async def _process_intent_action(
		self,
		conversation: ConversationSession,
		intent: IntentType,
		entities: Dict[str, Any],
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Process non-payment intents."""
		if intent == IntentType.CHECK_BALANCE:
			# Mock balance check
			await asyncio.sleep(0.3)
			
			mock_balance = {
				"checking": 1250.75,
				"savings": 5000.00,
				"credit_available": 2500.00
			}
			
			responses = {
				ConversationLanguage.ENGLISH: f"Your account balances:\n• Checking: ${mock_balance['checking']:.2f}\n• Savings: ${mock_balance['savings']:.2f}\n• Credit Available: ${mock_balance['credit_available']:.2f}",
				ConversationLanguage.SPANISH: f"Los saldos de tu cuenta:\n• Corriente: ${mock_balance['checking']:.2f}\n• Ahorros: ${mock_balance['savings']:.2f}\n• Crédito Disponible: ${mock_balance['credit_available']:.2f}",
				ConversationLanguage.SWAHILI: f"Mizani ya akaunti yako:\n• Ya kuhifadhi: ${mock_balance['checking']:.2f}\n• Ya akiba: ${mock_balance['savings']:.2f}\n• Mkopo Unapatikana: ${mock_balance['credit_available']:.2f}"
			}
			
			conversation.state = ConversationState.COMPLETED
			return {
				"content": responses.get(language, responses[ConversationLanguage.ENGLISH]),
				"type": "text",
				"requires_action": False
			}
		
		elif intent == IntentType.PAYMENT_HELP:
			return await self._generate_help_response(conversation, language)
		
		else:
			return await self._generate_help_response(conversation, language)
	
	async def _generate_help_response(
		self,
		conversation: ConversationSession,
		language: ConversationLanguage
	) -> Dict[str, Any]:
		"""Generate help response."""
		help_responses = {
			ConversationLanguage.ENGLISH: """I can help you with:
• Making payments - Just say "Send $50 to John" or "Pay my electricity bill"
• Checking balances - Ask "What's my balance?"
• Requesting refunds - Say "I need a refund for transaction 123"
• Payment history - Ask "Show my recent payments"
• Account questions - I'm here to help!

What would you like to do?""",
			
			ConversationLanguage.SPANISH: """Puedo ayudarte con:
• Hacer pagos - Solo di "Envía $50 a Juan" o "Paga mi factura de electricidad"
• Verificar saldos - Pregunta "¿Cuál es mi saldo?"
• Solicitar reembolsos - Di "Necesito un reembolso para la transacción 123"
• Historial de pagos - Pregunta "Muestra mis pagos recientes"
• Preguntas de cuenta - ¡Estoy aquí para ayudar!

¿Qué te gustaría hacer?""",
			
			ConversationLanguage.SWAHILI: """Ninaweza kukusaidia na:
• Kufanya malipo - Sema tu "Tuma $50 kwa John" au "Lipa bili yangu ya umeme"
• Kuangalia mizani - Uliza "Mizani yangu ni kiasi gani?"
• Kuomba kurejesha - Sema "Nahitaji kurejesha kwa muamala 123"
• Historia ya malipo - Uliza "Onyesha malipo yangu ya hivi karibuni"
• Maswali ya akaunti - Nipo hapa kusaidia!

Ungependa kufanya nini?"""
		}
		
		return {
			"content": help_responses.get(language, help_responses[ConversationLanguage.ENGLISH]),
			"type": "text",
			"requires_action": False
		}
	
	async def _get_response_template(
		self,
		template_key: str,
		language: ConversationLanguage
	) -> str:
		"""Get response template for language."""
		templates = self._response_templates.get(template_key, {})
		language_templates = templates.get(language, templates.get(ConversationLanguage.ENGLISH, []))
		
		if language_templates:
			import random
			return random.choice(language_templates)
		
		return "I'm here to help with your payments. What can I do for you?"
	
	async def _add_user_message(
		self,
		conversation_id: str,
		content: str,
		message_type: str,
		language: ConversationLanguage
	) -> ConversationMessage:
		"""Add user message to conversation."""
		message = ConversationMessage(
			conversation_id=conversation_id,
			sender_type="user",
			content=content,
			message_type=message_type,
			language=language
		)
		
		self._messages[conversation_id].append(message)
		return message
	
	async def _add_assistant_message(
		self,
		conversation_id: str,
		content: str,
		language: ConversationLanguage
	) -> ConversationMessage:
		"""Add assistant message to conversation."""
		message = ConversationMessage(
			conversation_id=conversation_id,
			sender_type="assistant",
			content=content,
			message_type="text",
			language=language
		)
		
		self._messages[conversation_id].append(message)
		return message
	
	async def authenticate_with_voice(
		self,
		conversation_id: str,
		voice_sample: bytes,
		customer_id: str
	) -> Dict[str, Any]:
		"""Authenticate user using voice biometrics."""
		if conversation_id not in self._conversations:
			raise ValueError(f"Conversation {conversation_id} not found")
		
		conversation = self._conversations[conversation_id]
		
		# Check if voice profile exists
		voice_profile = None
		for profile in self._voice_profiles.values():
			if profile.customer_id == customer_id:
				voice_profile = profile
				break
		
		if not voice_profile:
			return {
				"authenticated": False,
				"reason": "voice_profile_not_found",
				"message": "Voice profile not found. Please enroll first."
			}
		
		# Perform voice verification (simulated)
		await asyncio.sleep(1)  # Simulate processing time
		
		# Mock verification result (90% success rate for demo)
		import random
		verification_score = random.uniform(0.7, 1.0)
		is_verified = verification_score >= voice_profile.verification_threshold
		
		if is_verified:
			conversation.authenticated = True
			conversation.authentication_method = "voice_biometric"
			voice_profile.last_verification = datetime.utcnow()
			voice_profile.verification_count += 1
			
			return {
				"authenticated": True,
				"verification_score": verification_score,
				"message": "Voice authentication successful"
			}
		else:
			return {
				"authenticated": False,
				"verification_score": verification_score,
				"reason": "verification_failed",
				"message": "Voice verification failed. Please try again or use alternative authentication."
			}
	
	async def enroll_voice_profile(
		self,
		customer_id: str,
		voice_samples: List[bytes],
		language: ConversationLanguage = ConversationLanguage.ENGLISH
	) -> str:
		"""Enroll a new voice profile for customer."""
		if len(voice_samples) < 3:
			raise ValueError("At least 3 voice samples required for enrollment")
		
		# Process voice samples (simulated)
		await asyncio.sleep(2)  # Simulate enrollment processing
		
		# Create voice profile
		voice_profile = VoiceProfile(
			customer_id=customer_id,
			voice_features={
				"mfcc_features": [1.2, 3.4, 5.6],  # Mock MFCC features
				"pitch_range": [80, 300],
				"formant_frequencies": [800, 1200, 2500],
				"enrollment_language": language.value
			},
			biometric_template="mock_biometric_template_hash",
			enrollment_quality=0.95,  # High quality enrollment
			verification_threshold=0.8
		)
		
		self._voice_profiles[voice_profile.id] = voice_profile
		
		logger.info(f"Voice profile enrolled for customer {customer_id}")
		return voice_profile.id
	
	async def get_conversation_history(
		self,
		conversation_id: str,
		limit: int = 50
	) -> List[Dict[str, Any]]:
		"""Get conversation message history."""
		if conversation_id not in self._messages:
			return []
		
		messages = self._messages[conversation_id][-limit:]
		
		return [
			{
				"id": msg.id,
				"sender_type": msg.sender_type,
				"content": msg.content,
				"message_type": msg.message_type,
				"timestamp": msg.timestamp.isoformat(),
				"intent_detected": msg.intent_detected.value if msg.intent_detected else None,
				"confidence_score": msg.confidence_score,
				"response_time_ms": msg.response_time_ms
			}
			for msg in messages
		]
	
	async def get_conversation_analytics(
		self,
		time_range: Optional[Dict[str, datetime]] = None
	) -> Dict[str, Any]:
		"""Get analytics for conversations."""
		start_time = (time_range or {}).get('start', datetime.utcnow() - timedelta(days=7))
		end_time = (time_range or {}).get('end', datetime.utcnow())
		
		# Filter conversations by time range
		filtered_conversations = [
			conv for conv in self._conversations.values()
			if start_time <= conv.created_at <= end_time
		]
		
		if not filtered_conversations:
			return {"message": "No conversation data available for the specified time range"}
		
		# Calculate analytics
		total_conversations = len(filtered_conversations)
		completed_conversations = sum(1 for c in filtered_conversations if c.state == ConversationState.COMPLETED)
		failed_conversations = sum(1 for c in filtered_conversations if c.state == ConversationState.FAILED)
		
		# Channel distribution
		channel_distribution = {}
		for conv in filtered_conversations:
			channel = conv.channel.value
			channel_distribution[channel] = channel_distribution.get(channel, 0) + 1
		
		# Language distribution
		language_distribution = {}
		for conv in filtered_conversations:
			lang = conv.language.value
			language_distribution[lang] = language_distribution.get(lang, 0) + 1
		
		# Intent distribution
		intent_distribution = {}
		for conv in filtered_conversations:
			if conv.current_intent:
				intent = conv.current_intent.value
				intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
		
		# Calculate average response time
		all_messages = []
		for conv_id in self._messages:
			if conv_id in [c.id for c in filtered_conversations]:
				all_messages.extend(self._messages[conv_id])
		
		response_times = [msg.response_time_ms for msg in all_messages if msg.response_time_ms and msg.sender_type == "assistant"]
		avg_response_time = sum(response_times) / len(response_times) if response_times else 0
		
		# Calculate session durations
		session_durations = []
		for conv in filtered_conversations:
			if conv.state in [ConversationState.COMPLETED, ConversationState.FAILED]:
				duration = (conv.last_activity - conv.created_at).total_seconds()
				session_durations.append(duration)
		
		avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
		
		return {
			"time_range": {
				"start": start_time.isoformat(),
				"end": end_time.isoformat()
			},
			"total_conversations": total_conversations,
			"completion_rate": (completed_conversations / total_conversations * 100) if total_conversations > 0 else 0,
			"failure_rate": (failed_conversations / total_conversations * 100) if total_conversations > 0 else 0,
			"channel_distribution": channel_distribution,
			"language_distribution": language_distribution,
			"intent_distribution": intent_distribution,
			"average_response_time_ms": avg_response_time,
			"average_session_duration_seconds": avg_session_duration,
			"authentication_success_rate": sum(1 for c in filtered_conversations if c.authenticated) / total_conversations * 100 if total_conversations > 0 else 0
		}
	
	async def end_conversation(
		self,
		conversation_id: str,
		satisfaction_score: Optional[float] = None
	) -> Dict[str, Any]:
		"""End conversation and collect feedback."""
		if conversation_id not in self._conversations:
			raise ValueError(f"Conversation {conversation_id} not found")
		
		conversation = self._conversations[conversation_id]
		conversation.state = ConversationState.COMPLETED
		conversation.session_duration = int((datetime.utcnow() - conversation.created_at).total_seconds())
		conversation.satisfaction_score = satisfaction_score
		
		# Generate summary
		messages = self._messages.get(conversation_id, [])
		
		summary = {
			"conversation_id": conversation_id,
			"duration_seconds": conversation.session_duration,
			"message_count": len(messages),
			"intents_processed": [conversation.current_intent.value] if conversation.current_intent else [],
			"authentication_method": conversation.authentication_method,
			"final_state": conversation.state.value,
			"satisfaction_score": satisfaction_score,
			"channel": conversation.channel.value,
			"language": conversation.language.value
		}
		
		logger.info(f"Conversation {conversation_id} ended with summary: {summary}")
		return summary