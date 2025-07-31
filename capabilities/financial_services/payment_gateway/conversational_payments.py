"""
Conversational Payment Interface - Natural Language Payment Processing

Revolutionary conversational AI system for payment processing using natural language,
voice commands, and intelligent conversation flows for APG Payment Gateway.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from dataclasses import dataclass

# NLP and AI Libraries
try:
	import spacy
	from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
	import openai
	from textblob import TextBlob
except ImportError:
	print("⚠️  NLP libraries not available - using simplified conversation parsing")
	spacy = None

from .models import PaymentTransaction, PaymentMethod, PaymentStatus, PaymentMethodType
from .payment_processor import PaymentResult
from .ml_fraud_detection import MLFraudDetectionEngine

class ConversationState(str, Enum):
	"""Conversation states for payment flow"""
	GREETING = "greeting"
	INTENT_RECOGNITION = "intent_recognition"
	PAYMENT_DETAILS = "payment_details"
	AMOUNT_CONFIRMATION = "amount_confirmation"
	PAYMENT_METHOD = "payment_method"
	AUTHENTICATION = "authentication"
	PROCESSING = "processing"
	CONFIRMATION = "confirmation"
	ERROR_HANDLING = "error_handling"
	COMPLETION = "completion"

class PaymentIntent(str, Enum):
	"""Payment intents recognizable from conversation"""
	MAKE_PAYMENT = "make_payment"
	CHECK_BALANCE = "check_balance"
	TRANSFER_MONEY = "transfer_money"
	PAY_BILL = "pay_bill"
	SEND_MONEY = "send_money"
	BUY_PRODUCT = "buy_product"
	SUBSCRIPTION = "subscription"
	REFUND_REQUEST = "refund_request"
	PAYMENT_STATUS = "payment_status"
	HELP = "help"

class ConversationChannel(str, Enum):
	"""Conversation channels"""
	VOICE = "voice"
	TEXT_CHAT = "text_chat"
	SMS = "sms"
	WHATSAPP = "whatsapp"
	MESSENGER = "messenger"
	TELEGRAM = "telegram"
	WEB_WIDGET = "web_widget"

@dataclass
class ConversationContext:
	"""Conversation context and state"""
	conversation_id: str
	user_id: str
	channel: ConversationChannel
	state: ConversationState
	intent: Optional[PaymentIntent]
	extracted_entities: Dict[str, Any]
	conversation_history: List[Dict[str, Any]]
	user_preferences: Dict[str, Any]
	authentication_status: bool
	session_data: Dict[str, Any]
	created_at: datetime
	last_updated: datetime

@dataclass
class ConversationResponse:
	"""Response from conversational AI"""
	response_id: str
	conversation_id: str
	message: str
	suggested_actions: List[str]
	quick_replies: List[str]
	requires_input: bool
	next_state: ConversationState
	payment_data: Optional[Dict[str, Any]]
	confidence: float
	metadata: Dict[str, Any]
	created_at: datetime

class ConversationalPaymentEngine:
	"""
	Revolutionary conversational AI payment engine
	
	Enables natural language payment processing with voice commands,
	intelligent conversation flows, and multi-channel support.
	"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.engine_id = uuid7str()
		
		# Conversation configuration
		self.supported_languages = config.get("supported_languages", ["en", "sw", "fr", "es"])
		self.default_language = config.get("default_language", "en")
		self.enable_voice_processing = config.get("enable_voice_processing", True)
		self.enable_multilingual = config.get("enable_multilingual", True)
		self.conversation_timeout_minutes = config.get("conversation_timeout_minutes", 15)
		
		# AI configuration
		self.use_advanced_nlp = config.get("use_advanced_nlp", True)
		self.confidence_threshold = config.get("confidence_threshold", 0.7)
		self.enable_context_awareness = config.get("enable_context_awareness", True)
		self.enable_sentiment_analysis = config.get("enable_sentiment_analysis", True)
		
		# Security configuration
		self.require_authentication = config.get("require_authentication", True)
		self.enable_voice_biometrics = config.get("enable_voice_biometrics", False)
		self.secure_amount_threshold = config.get("secure_amount_threshold", 100000)  # $1000
		
		# NLP models and processors
		self._nlp_models: Dict[str, Any] = {}
		self._intent_classifier = None
		self._entity_extractor = None
		self._sentiment_analyzer = None
		
		# Conversation management
		self._active_conversations: Dict[str, ConversationContext] = {}
		self._conversation_templates: Dict[str, Dict[str, Any]] = {}
		self._response_cache: Dict[str, ConversationResponse] = {}
		
		# Payment integration
		self._payment_processor_manager = None
		self._fraud_detection_engine = None
		
		# Analytics
		self._conversation_analytics: Dict[str, Any] = {}
		self._user_interactions: List[Dict[str, Any]] = []
		
		self._initialized = False
		
		self._log_engine_created()
	
	async def initialize(self, payment_manager=None, fraud_engine=None) -> Dict[str, Any]:
		"""Initialize conversational payment engine"""
		self._log_engine_initialization_start()
		
		try:
			# Store references to other engines
			self._payment_processor_manager = payment_manager
			self._fraud_detection_engine = fraud_engine
			
			# Initialize NLP models
			await self._initialize_nlp_models()
			
			# Initialize conversation templates
			await self._initialize_conversation_templates()
			
			# Set up intent recognition
			await self._setup_intent_recognition()
			
			# Initialize entity extraction
			await self._setup_entity_extraction()
			
			# Set up sentiment analysis
			if self.enable_sentiment_analysis:
				await self._setup_sentiment_analysis()
			
			# Initialize conversation flows
			await self._initialize_conversation_flows()
			
			self._initialized = True
			
			self._log_engine_initialization_complete()
			
			return {
				"status": "initialized",
				"engine_id": self.engine_id,
				"supported_languages": self.supported_languages,
				"voice_processing": self.enable_voice_processing,
				"multilingual": self.enable_multilingual,
				"conversation_timeout": self.conversation_timeout_minutes
			}
			
		except Exception as e:
			self._log_engine_initialization_error(str(e))
			raise
	
	async def process_conversation_message(
		self,
		message: str,
		user_id: str,
		channel: ConversationChannel,
		conversation_id: str | None = None,
		context: Dict[str, Any] | None = None
	) -> ConversationResponse:
		"""
		Process conversational message and generate intelligent response
		
		Args:
			message: User message (text or transcribed voice)
			user_id: User identifier
			channel: Communication channel
			conversation_id: Existing conversation ID or None for new
			context: Additional context (location, device, etc.)
			
		Returns:
			ConversationResponse with intelligent reply and actions
		"""
		if not self._initialized:
			raise RuntimeError("Conversational payment engine not initialized")
		
		context = context or {}
		
		# Get or create conversation context
		if conversation_id and conversation_id in self._active_conversations:
			conv_context = self._active_conversations[conversation_id]
		else:
			conversation_id = uuid7str()
			conv_context = await self._create_conversation_context(
				conversation_id, user_id, channel, context
			)
		
		self._log_message_processing_start(conversation_id, message[:50])
		
		try:
			# Preprocess message
			processed_message = await self._preprocess_message(message, conv_context)
			
			# Detect language
			detected_language = await self._detect_language(processed_message)
			
			# Analyze sentiment
			sentiment = await self._analyze_sentiment(processed_message) if self.enable_sentiment_analysis else {"polarity": 0.0}
			
			# Extract intent
			intent, intent_confidence = await self._extract_intent(processed_message, conv_context)
			
			# Extract entities
			entities = await self._extract_entities(processed_message, conv_context)
			
			# Update conversation context
			conv_context.extracted_entities.update(entities)
			conv_context.conversation_history.append({
				"timestamp": datetime.now(timezone.utc),
				"message": message,
				"processed_message": processed_message,
				"intent": intent.value if intent else None,
				"entities": entities,
				"sentiment": sentiment
			})
			conv_context.last_updated = datetime.now(timezone.utc)
			
			# Generate response based on current state and intent
			response = await self._generate_response(
				conv_context, intent, entities, sentiment, intent_confidence
			)
			
			# Update conversation state
			conv_context.state = response.next_state
			conv_context.intent = intent
			
			# Store updated context
			self._active_conversations[conversation_id] = conv_context
			
			# Record interaction for analytics
			await self._record_interaction(conv_context, message, response)
			
			self._log_message_processing_complete(conversation_id, response.message[:50])
			
			return response
			
		except Exception as e:
			self._log_message_processing_error(conversation_id, str(e))
			
			# Return fallback response
			return await self._create_fallback_response(conversation_id, conv_context)
	
	async def process_voice_payment(
		self,
		audio_data: bytes,
		user_id: str,
		context: Dict[str, Any] | None = None
	) -> ConversationResponse:
		"""Process voice payment command"""
		try:
			# Convert speech to text (placeholder - would use real STT)
			transcribed_text = await self._speech_to_text(audio_data)
			
			# Process as conversation message
			return await self.process_conversation_message(
				transcribed_text, user_id, ConversationChannel.VOICE, context=context
			)
			
		except Exception as e:
			self._log_voice_processing_error(user_id, str(e))
			raise
	
	async def execute_payment_from_conversation(
		self,
		conversation_id: str
	) -> PaymentResult:
		"""Execute payment based on conversation data"""
		try:
			conv_context = self._active_conversations.get(conversation_id)
			if not conv_context:
				raise ValueError("Conversation not found")
			
			# Extract payment data from conversation
			payment_data = await self._extract_payment_data(conv_context)
			
			# Validate payment data
			validation_result = await self._validate_payment_data(payment_data)
			if not validation_result["valid"]:
				raise ValueError(f"Invalid payment data: {validation_result['error']}")
			
			# Create payment transaction
			transaction = await self._create_transaction_from_conversation(conv_context, payment_data)
			
			# Process payment through payment manager
			if self._payment_processor_manager:
				result = await self._payment_processor_manager.process_payment(
					transaction,
					payment_data["payment_method"],
					{"conversation_id": conversation_id, "channel": conv_context.channel.value}
				)
			else:
				# Simulation for development
				result = PaymentResult(
					success=True,
					status=PaymentStatus.COMPLETED,
					processor_transaction_id=uuid7str(),
					metadata={"simulated": True}
				)
			
			# Update conversation with payment result
			await self._update_conversation_with_payment_result(conv_context, result)
			
			return result
			
		except Exception as e:
			self._log_payment_execution_error(conversation_id, str(e))
			raise
	
	async def get_conversation_analytics(
		self,
		user_id: str | None = None,
		date_range: Tuple[datetime, datetime] | None = None
	) -> Dict[str, Any]:
		"""Get conversation analytics and insights"""
		try:
			# Filter interactions
			filtered_interactions = self._user_interactions
			
			if user_id:
				filtered_interactions = [
					i for i in filtered_interactions 
					if i.get("user_id") == user_id
				]
			
			if date_range:
				start_date, end_date = date_range
				filtered_interactions = [
					i for i in filtered_interactions
					if start_date <= i.get("timestamp", datetime.min) <= end_date
				]
			
			# Calculate analytics
			analytics = await self._calculate_conversation_analytics(filtered_interactions)
			
			return analytics
			
		except Exception as e:
			self._log_analytics_error(str(e))
			return {"status": "error", "error": str(e)}
	
	# Core NLP and conversation processing methods
	
	async def _preprocess_message(
		self,
		message: str,
		context: ConversationContext
	) -> str:
		"""Preprocess message for NLP analysis"""
		
		# Basic text cleaning
		processed = message.strip().lower()
		
		# Remove extra whitespace
		processed = re.sub(r'\s+', ' ', processed)
		
		# Handle common abbreviations and slang
		abbreviations = {
			"u": "you",
			"ur": "your",
			"pls": "please",
			"thx": "thanks",
			"cant": "cannot",
			"wont": "will not",
			"dont": "do not"
		}
		
		for abbrev, full in abbreviations.items():
			processed = re.sub(r'\b' + abbrev + r'\b', full, processed)
		
		# Handle numbers and currency
		processed = re.sub(r'\$(\d+)', r'\1 dollars', processed)
		processed = re.sub(r'(\d+)\s*k\b', r'\1000', processed)
		
		return processed
	
	async def _detect_language(self, message: str) -> str:
		"""Detect message language"""
		if not self.enable_multilingual:
			return self.default_language
		
		# Simplified language detection
		if spacy:
			try:
				# Use basic keyword detection for common languages
				swahili_keywords = ["pesa", "malipo", "lipa", "kutuma", "kununua"]
				french_keywords = ["payer", "argent", "acheter", "envoyer"]
				spanish_keywords = ["pagar", "dinero", "comprar", "enviar"]
				
				if any(keyword in message.lower() for keyword in swahili_keywords):
					return "sw"
				elif any(keyword in message.lower() for keyword in french_keywords):
					return "fr"
				elif any(keyword in message.lower() for keyword in spanish_keywords):
					return "es"
			except Exception:
				pass
		
		return self.default_language
	
	async def _analyze_sentiment(self, message: str) -> Dict[str, float]:
		"""Analyze message sentiment"""
		try:
			if TextBlob:
				blob = TextBlob(message)
				return {
					"polarity": blob.sentiment.polarity,
					"subjectivity": blob.sentiment.subjectivity
				}
		except Exception:
			pass
		
		# Fallback sentiment analysis
		positive_words = ["good", "great", "excellent", "love", "like", "yes", "please", "thanks"]
		negative_words = ["bad", "terrible", "hate", "no", "cancel", "stop", "angry", "frustrated"]
		
		positive_count = sum(1 for word in positive_words if word in message.lower())
		negative_count = sum(1 for word in negative_words if word in message.lower())
		
		if positive_count > negative_count:
			polarity = 0.5
		elif negative_count > positive_count:
			polarity = -0.5
		else:
			polarity = 0.0
		
		return {"polarity": polarity, "subjectivity": 0.5}
	
	async def _extract_intent(
		self,
		message: str,
		context: ConversationContext
	) -> Tuple[Optional[PaymentIntent], float]:
		"""Extract payment intent from message"""
		
		# Intent patterns
		intent_patterns = {
			PaymentIntent.MAKE_PAYMENT: [
				r"pay\s+for", r"make\s+payment", r"pay\s+\$?\d+", r"i want to pay",
				r"lipa", r"malipo"  # Swahili
			],
			PaymentIntent.SEND_MONEY: [
				r"send\s+money", r"transfer\s+\$?\d+", r"send\s+\$?\d+", r"tuma pesa"
			],
			PaymentIntent.CHECK_BALANCE: [
				r"check\s+balance", r"my\s+balance", r"how much", r"salio"
			],
			PaymentIntent.BUY_PRODUCT: [
				r"buy\s+", r"purchase\s+", r"order\s+", r"kununua"
			],
			PaymentIntent.PAY_BILL: [
				r"pay\s+bill", r"pay\s+my\s+", r"bill\s+payment", r"lipa bili"
			],
			PaymentIntent.REFUND_REQUEST: [
				r"refund", r"return\s+money", r"get\s+my\s+money\s+back"
			],
			PaymentIntent.PAYMENT_STATUS: [
				r"status", r"check\s+payment", r"where\s+is", r"hali ya malipo"
			],
			PaymentIntent.HELP: [
				r"help", r"how\s+to", r"what\s+can", r"msaada"
			]
		}
		
		best_intent = None
		best_confidence = 0.0
		
		for intent, patterns in intent_patterns.items():
			for pattern in patterns:
				if re.search(pattern, message.lower()):
					confidence = 0.8  # Base confidence for pattern match
					
					# Boost confidence based on context
					if context.state == ConversationState.INTENT_RECOGNITION:
						confidence += 0.1
					
					if confidence > best_confidence:
						best_confidence = confidence
						best_intent = intent
		
		return best_intent, best_confidence
	
	async def _extract_entities(
		self,
		message: str,
		context: ConversationContext
	) -> Dict[str, Any]:
		"""Extract entities from message"""
		entities = {}
		
		# Extract amounts
		amount_patterns = [
			r'\$(\d+(?:\.\d{2})?)',  # $123.45
			r'(\d+(?:\.\d{2})?)\s*dollars?',  # 123.45 dollars
			r'(\d+(?:\.\d{2})?)\s*usd',  # 123.45 USD
			r'(\d+)\s*shillings?',  # 123 shillings
			r'ksh\s*(\d+)',  # KSH 123
		]
		
		for pattern in amount_patterns:
			match = re.search(pattern, message.lower())
			if match:
				try:
					amount = float(match.group(1))
					entities["amount"] = int(amount * 100)  # Convert to cents
					break
				except (ValueError, IndexError):
					continue
		
		# Extract phone numbers
		phone_patterns = [
			r'(\+?254\d{9})',  # Kenyan format
			r'(0\d{9})',       # Local format
			r'(\d{10})'        # 10-digit format
		]
		
		for pattern in phone_patterns:
			match = re.search(pattern, message)
			if match:
				entities["phone_number"] = match.group(1)
				break
		
		# Extract payment methods
		payment_method_keywords = {
			"mpesa": PaymentMethodType.MPESA,
			"m-pesa": PaymentMethodType.MPESA,
			"card": PaymentMethodType.CREDIT_CARD,
			"credit card": PaymentMethodType.CREDIT_CARD,
			"debit card": PaymentMethodType.DEBIT_CARD,
			"paypal": PaymentMethodType.PAYPAL,
			"bank transfer": PaymentMethodType.BANK_TRANSFER
		}
		
		for keyword, method in payment_method_keywords.items():
			if keyword in message.lower():
				entities["payment_method"] = method
				break
		
		# Extract recipient information
		if "to" in message.lower():
			# Simple recipient extraction
			to_match = re.search(r'to\s+([a-zA-Z\s]+)', message.lower())
			if to_match:
				entities["recipient"] = to_match.group(1).strip()
		
		# Extract confirmation words
		confirmation_words = ["yes", "confirm", "proceed", "ok", "ndiyo", "sawa"]
		if any(word in message.lower() for word in confirmation_words):
			entities["confirmation"] = True
		
		cancellation_words = ["no", "cancel", "stop", "hapana", "acha"]
		if any(word in message.lower() for word in cancellation_words):
			entities["cancellation"] = True
		
		return entities
	
	# Response generation methods
	
	async def _generate_response(
		self,
		context: ConversationContext,
		intent: Optional[PaymentIntent],
		entities: Dict[str, Any],
		sentiment: Dict[str, float],
		confidence: float
	) -> ConversationResponse:
		"""Generate intelligent response based on context and intent"""
		
		response_id = uuid7str()
		
		# Handle low confidence or unclear intent
		if confidence < self.confidence_threshold:
			return await self._generate_clarification_response(response_id, context, entities)
		
		# Handle negative sentiment
		if sentiment.get("polarity", 0) < -0.3:
			return await self._generate_empathy_response(response_id, context, intent, entities)
		
		# Generate response based on current state and intent
		if context.state == ConversationState.GREETING:
			return await self._generate_greeting_response(response_id, context, intent, entities)
		
		elif context.state == ConversationState.INTENT_RECOGNITION:
			return await self._generate_intent_response(response_id, context, intent, entities)
		
		elif context.state == ConversationState.PAYMENT_DETAILS:
			return await self._generate_details_response(response_id, context, intent, entities)
		
		elif context.state == ConversationState.AMOUNT_CONFIRMATION:
			return await self._generate_amount_confirmation_response(response_id, context, entities)
		
		elif context.state == ConversationState.PAYMENT_METHOD:
			return await self._generate_payment_method_response(response_id, context, entities)
		
		elif context.state == ConversationState.AUTHENTICATION:
			return await self._generate_authentication_response(response_id, context, entities)
		
		elif context.state == ConversationState.PROCESSING:
			return await self._generate_processing_response(response_id, context)
		
		else:
			return await self._generate_default_response(response_id, context, intent, entities)
	
	async def _generate_greeting_response(
		self,
		response_id: str,
		context: ConversationContext,
		intent: Optional[PaymentIntent],
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate greeting response"""
		
		greetings = {
			"en": "Hello! I'm your payment assistant. How can I help you today?",
			"sw": "Hujambo! Mimi ni msaidizi wako wa malipo. Nawezaje kukusaidia leo?",
			"fr": "Bonjour! Je suis votre assistant de paiement. Comment puis-je vous aider aujourd'hui?",
			"es": "¡Hola! Soy tu asistente de pagos. ¿Cómo puedo ayudarte hoy?"
		}
		
		language = context.user_preferences.get("language", "en")
		message = greetings.get(language, greetings["en"])
		
		quick_replies = ["Make a payment", "Send money", "Check balance", "Pay a bill"]
		
		next_state = ConversationState.INTENT_RECOGNITION
		if intent:
			next_state = ConversationState.PAYMENT_DETAILS
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["select_payment_type"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data=None,
			confidence=0.9,
			metadata={"greeting_type": "initial"},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_intent_response(
		self,
		response_id: str,
		context: ConversationContext,
		intent: PaymentIntent,
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate response for recognized intent"""
		
		if intent == PaymentIntent.MAKE_PAYMENT:
			message = "I'll help you make a payment. "
			if "amount" in entities:
				amount_display = f"${entities['amount'] / 100:.2f}"
				message += f"I see you want to pay {amount_display}. "
			message += "What would you like to pay for?"
			
			next_state = ConversationState.PAYMENT_DETAILS
			quick_replies = ["Product purchase", "Bill payment", "Service fee"]
		
		elif intent == PaymentIntent.SEND_MONEY:
			message = "I'll help you send money. "
			if "amount" in entities:
				amount_display = f"${entities['amount'] / 100:.2f}"
				message += f"You want to send {amount_display}. "
			message += "Who would you like to send money to?"
			
			next_state = ConversationState.PAYMENT_DETAILS
			quick_replies = ["Enter phone number", "Select contact", "Enter account"]
		
		elif intent == PaymentIntent.CHECK_BALANCE:
			# This would integrate with account services
			message = "Let me check your balance. Please wait a moment..."
			next_state = ConversationState.PROCESSING
			quick_replies = []
		
		elif intent == PaymentIntent.PAY_BILL:
			message = "I'll help you pay a bill. What type of bill would you like to pay?"
			next_state = ConversationState.PAYMENT_DETAILS
			quick_replies = ["Electricity", "Water", "Internet", "Phone", "Other"]
		
		else:
			message = "I understand you need help with payments. Could you tell me more specifically what you'd like to do?"
			next_state = ConversationState.INTENT_RECOGNITION
			quick_replies = ["Make payment", "Send money", "Pay bill", "Check status"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["collect_payment_details"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data={"intent": intent.value, "entities": entities},
			confidence=0.8,
			metadata={"intent_recognized": intent.value},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_details_response(
		self,
		response_id: str,
		context: ConversationContext,
		intent: Optional[PaymentIntent],
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate response for collecting payment details"""
		
		# Check what details we still need
		needed_details = []
		current_entities = context.extracted_entities
		
		if "amount" not in current_entities and "amount" not in entities:
			needed_details.append("amount")
		
		if context.intent == PaymentIntent.SEND_MONEY and "phone_number" not in current_entities:
			needed_details.append("recipient")
		
		if "payment_method" not in current_entities:
			needed_details.append("payment_method")
		
		# Generate response based on missing details
		if "amount" in needed_details:
			message = "How much would you like to pay?"
			quick_replies = ["$10", "$25", "$50", "$100", "Other amount"]
			next_state = ConversationState.PAYMENT_DETAILS
		
		elif "recipient" in needed_details:
			message = "Please provide the recipient's phone number or account details."
			quick_replies = ["Enter phone number", "Select from contacts"]
			next_state = ConversationState.PAYMENT_DETAILS
		
		elif "payment_method" in needed_details:
			message = "How would you like to pay?"
			quick_replies = ["M-PESA", "Credit Card", "Bank Transfer", "PayPal"]
			next_state = ConversationState.PAYMENT_METHOD
		
		else:
			# All details collected, move to confirmation
			amount_display = f"${current_entities.get('amount', 0) / 100:.2f}"
			message = f"Great! Let me confirm: you want to pay {amount_display}"
			
			if context.intent == PaymentIntent.SEND_MONEY and "phone_number" in current_entities:
				message += f" to {current_entities['phone_number']}"
			
			message += ". Is this correct?"
			quick_replies = ["Yes, proceed", "No, change details"]
			next_state = ConversationState.AMOUNT_CONFIRMATION
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["collect_missing_details"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data={"missing_details": needed_details},
			confidence=0.8,
			metadata={"details_collection": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_amount_confirmation_response(
		self,
		response_id: str,
		context: ConversationContext,
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate amount confirmation response"""
		
		if entities.get("confirmation"):
			# User confirmed, proceed to payment method or authentication
			current_entities = context.extracted_entities
			amount = current_entities.get("amount", 0)
			
			# Check if amount requires additional security
			if amount > self.secure_amount_threshold:
				message = "This is a high-value transaction. For security, I'll need to verify your identity first."
				next_state = ConversationState.AUTHENTICATION
				quick_replies = ["Verify with PIN", "Verify with biometrics", "Send SMS code"]
			else:
				message = "Perfect! Now, how would you like to pay?"
				next_state = ConversationState.PAYMENT_METHOD
				quick_replies = ["M-PESA", "Credit Card", "Bank Transfer", "PayPal"]
		
		elif entities.get("cancellation"):
			message = "No problem! What would you like to change?"
			next_state = ConversationState.PAYMENT_DETAILS
			quick_replies = ["Change amount", "Change recipient", "Start over"]
		
		else:
			# Ask for confirmation
			current_entities = context.extracted_entities
			amount_display = f"${current_entities.get('amount', 0) / 100:.2f}"
			message = f"Please confirm: you want to pay {amount_display}. Is this correct?"
			next_state = ConversationState.AMOUNT_CONFIRMATION
			quick_replies = ["Yes, that's correct", "No, change amount"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["confirm_amount"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data=None,
			confidence=0.9,
			metadata={"confirmation_step": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_payment_method_response(
		self,
		response_id: str,
		context: ConversationContext,
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate payment method selection response"""
		
		if "payment_method" in entities:
			method = entities["payment_method"]
			method_name = method.value.replace("_", " ").title()
			
			message = f"Great! You've selected {method_name}. "
			
			if method == PaymentMethodType.MPESA:
				message += "Please enter your M-PESA PIN to complete the payment."
				quick_replies = ["Enter PIN", "Use different method"]
			elif method in [PaymentMethodType.CREDIT_CARD, PaymentMethodType.DEBIT_CARD]:
				message += "I'll redirect you to secure card payment."
				quick_replies = ["Proceed to card payment", "Use different method"]
			else:
				message += "Let me set up the payment for you."
				quick_replies = ["Proceed", "Use different method"]
			
			next_state = ConversationState.AUTHENTICATION
		else:
			message = "Which payment method would you prefer?"
			quick_replies = ["M-PESA", "Credit Card", "Bank Transfer", "PayPal"]
			next_state = ConversationState.PAYMENT_METHOD
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["process_payment_method"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data={"payment_method": entities.get("payment_method")},
			confidence=0.8,
			metadata={"payment_method_selection": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_authentication_response(
		self,
		response_id: str,
		context: ConversationContext,
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate authentication response"""
		
		if context.authentication_status:
			message = "Authentication successful! Processing your payment now..."
			next_state = ConversationState.PROCESSING
			quick_replies = []
		else:
			message = "For security, please verify your identity. How would you like to authenticate?"
			next_state = ConversationState.AUTHENTICATION
			quick_replies = ["SMS Code", "PIN", "Biometric", "Voice verification"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["authenticate_user"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=next_state,
			payment_data=None,
			confidence=0.9,
			metadata={"authentication_required": not context.authentication_status},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_processing_response(
		self,
		response_id: str,
		context: ConversationContext
	) -> ConversationResponse:
		"""Generate processing response"""
		
		message = "Processing your payment... Please wait a moment."
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["execute_payment"],
			quick_replies=[],
			requires_input=False,
			next_state=ConversationState.CONFIRMATION,
			payment_data={"ready_to_process": True},
			confidence=1.0,
			metadata={"processing": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_clarification_response(
		self,
		response_id: str,
		context: ConversationContext,
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate clarification response for unclear input"""
		
		message = "I'm not quite sure what you'd like to do. Could you please clarify?"
		quick_replies = ["Make a payment", "Send money", "Check balance", "Pay a bill", "Get help"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["clarify_intent"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=ConversationState.INTENT_RECOGNITION,
			payment_data=None,
			confidence=0.5,
			metadata={"clarification_needed": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_empathy_response(
		self,
		response_id: str,
		context: ConversationContext,
		intent: Optional[PaymentIntent],
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate empathetic response for negative sentiment"""
		
		message = "I understand you might be frustrated. I'm here to help make this as easy as possible for you. What can I do to assist you?"
		quick_replies = ["Start over", "Speak to human agent", "Continue", "Cancel"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["provide_support"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=ConversationState.INTENT_RECOGNITION,
			payment_data=None,
			confidence=0.7,
			metadata={"empathy_response": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _generate_default_response(
		self,
		response_id: str,
		context: ConversationContext,
		intent: Optional[PaymentIntent],
		entities: Dict[str, Any]
	) -> ConversationResponse:
		"""Generate default response"""
		
		message = "How can I help you with your payment today?"
		quick_replies = ["Make payment", "Send money", "Check status", "Get help"]
		
		return ConversationResponse(
			response_id=response_id,
			conversation_id=context.conversation_id,
			message=message,
			suggested_actions=["general_assistance"],
			quick_replies=quick_replies,
			requires_input=True,
			next_state=ConversationState.INTENT_RECOGNITION,
			payment_data=None,
			confidence=0.6,
			metadata={"default_response": True},
			created_at=datetime.now(timezone.utc)
		)
	
	async def _create_fallback_response(
		self,
		conversation_id: str,
		context: ConversationContext | None
	) -> ConversationResponse:
		"""Create fallback response for errors"""
		
		return ConversationResponse(
			response_id=uuid7str(),
			conversation_id=conversation_id,
			message="I'm sorry, I'm having trouble understanding. Could you please try again or speak to a human agent?",
			suggested_actions=["contact_support"],
			quick_replies=["Try again", "Human agent", "Start over"],
			requires_input=True,
			next_state=ConversationState.ERROR_HANDLING,
			payment_data=None,
			confidence=0.3,
			metadata={"fallback": True, "error": True},
			created_at=datetime.now(timezone.utc)
		)
	
	# Utility and helper methods
	
	async def _create_conversation_context(
		self,
		conversation_id: str,
		user_id: str,
		channel: ConversationChannel,
		context: Dict[str, Any]
	) -> ConversationContext:
		"""Create new conversation context"""
		
		# Get user preferences (would integrate with user service)
		user_preferences = context.get("user_preferences", {
			"language": self.default_language,
			"preferred_payment_method": None,
			"notification_preferences": {}
		})
		
		return ConversationContext(
			conversation_id=conversation_id,
			user_id=user_id,
			channel=channel,
			state=ConversationState.GREETING,
			intent=None,
			extracted_entities={},
			conversation_history=[],
			user_preferences=user_preferences,
			authentication_status=False,
			session_data=context,
			created_at=datetime.now(timezone.utc),
			last_updated=datetime.now(timezone.utc)
		)
	
	async def _speech_to_text(self, audio_data: bytes) -> str:
		"""Convert speech to text (placeholder)"""
		# In production, this would use services like:
		# - Google Speech-to-Text
		# - Azure Speech Services
		# - AWS Transcribe
		# - OpenAI Whisper
		
		# For demo purposes, return a placeholder
		return "I want to send 1000 shillings to 0712345678 using mpesa"
	
	async def _extract_payment_data(self, context: ConversationContext) -> Dict[str, Any]:
		"""Extract payment data from conversation context"""
		entities = context.extracted_entities
		
		return {
			"amount": entities.get("amount"),
			"recipient": entities.get("phone_number") or entities.get("recipient"),
			"payment_method_type": entities.get("payment_method"),
			"description": f"Conversational payment via {context.channel.value}",
			"metadata": {
				"conversation_id": context.conversation_id,
				"channel": context.channel.value,
				"intent": context.intent.value if context.intent else None
			}
		}
	
	async def _validate_payment_data(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate payment data extracted from conversation"""
		
		if not payment_data.get("amount"):
			return {"valid": False, "error": "Amount is required"}
		
		if payment_data["amount"] <= 0:
			return {"valid": False, "error": "Amount must be positive"}
		
		if not payment_data.get("payment_method_type"):
			return {"valid": False, "error": "Payment method is required"}
		
		return {"valid": True}
	
	async def _create_transaction_from_conversation(
		self,
		context: ConversationContext,
		payment_data: Dict[str, Any]
	) -> PaymentTransaction:
		"""Create payment transaction from conversation data"""
		
		# This would integrate with the payment models
		# For now, create a simplified transaction object
		return PaymentTransaction(
			tenant_id="default",
			merchant_id="conversational_payments",
			customer_id=context.user_id,
			amount=payment_data["amount"],
			currency="USD",  # Would be determined by user location/preferences
			description=payment_data["description"],
			payment_method_type=payment_data["payment_method_type"],
			metadata=payment_data["metadata"],
			created_by=context.user_id
		)
	
	async def _update_conversation_with_payment_result(
		self,
		context: ConversationContext,
		result: PaymentResult
	):
		"""Update conversation with payment result"""
		
		context.session_data["payment_result"] = {
			"success": result.success,
			"status": result.status.value if hasattr(result.status, 'value') else str(result.status),
			"processor_transaction_id": result.processor_transaction_id,
			"timestamp": datetime.now(timezone.utc).isoformat()
		}
		
		context.state = ConversationState.COMPLETION
		context.last_updated = datetime.now(timezone.utc)
	
	async def _record_interaction(
		self,
		context: ConversationContext,
		message: str,
		response: ConversationResponse
	):
		"""Record interaction for analytics"""
		
		interaction = {
			"conversation_id": context.conversation_id,
			"user_id": context.user_id,
			"channel": context.channel.value,
			"user_message": message,
			"bot_response": response.message,
			"intent": context.intent.value if context.intent else None,
			"state": context.state.value,
			"confidence": response.confidence,
			"timestamp": datetime.now(timezone.utc)
		}
		
		self._user_interactions.append(interaction)
	
	async def _calculate_conversation_analytics(
		self,
		interactions: List[Dict[str, Any]]
	) -> Dict[str, Any]:
		"""Calculate conversation analytics"""
		
		if not interactions:
			return {"status": "no_data"}
		
		total_conversations = len(set(i["conversation_id"] for i in interactions))
		total_messages = len(interactions)
		
		# Intent distribution
		intents = [i.get("intent") for i in interactions if i.get("intent")]
		intent_counts = {}
		for intent in intents:
			intent_counts[intent] = intent_counts.get(intent, 0) + 1
		
		# Channel distribution
		channels = [i.get("channel") for i in interactions]
		channel_counts = {}
		for channel in channels:
			channel_counts[channel] = channel_counts.get(channel, 0) + 1
		
		# Average confidence
		confidences = [i.get("confidence", 0) for i in interactions]
		avg_confidence = sum(confidences) / len(confidences) if confidences else 0
		
		return {
			"total_conversations": total_conversations,
			"total_messages": total_messages,
			"average_confidence": avg_confidence,
			"intent_distribution": intent_counts,
			"channel_distribution": channel_counts,
			"messages_per_conversation": total_messages / total_conversations if total_conversations > 0 else 0
		}
	
	# Initialization methods
	
	async def _initialize_nlp_models(self):
		"""Initialize NLP models"""
		
		if self.use_advanced_nlp and spacy:
			try:
				# Load spaCy model for English
				self._nlp_models["en"] = spacy.load("en_core_web_sm")
			except Exception:
				self._log_nlp_model_warning("English model not available")
		
		self._log_nlp_models_initialized()
	
	async def _initialize_conversation_templates(self):
		"""Initialize conversation templates"""
		
		self._conversation_templates = {
			"greeting": {
				"en": "Hello! I'm your payment assistant.",
				"sw": "Hujambo! Mimi ni msaidizi wako wa malipo.",
				"fr": "Bonjour! Je suis votre assistant de paiement.",
				"es": "¡Hola! Soy tu asistente de pagos."
			},
			"payment_confirmation": {
				"en": "Payment of {amount} has been processed successfully.",
				"sw": "Malipo ya {amount} yamefanikiwa.",
				"fr": "Le paiement de {amount} a été traité avec succès.",
				"es": "El pago de {amount} se ha procesado exitosamente."
			}
		}
		
		self._log_conversation_templates_initialized()
	
	async def _setup_intent_recognition(self):
		"""Set up intent recognition"""
		
		# Initialize intent classifier (would use trained model in production)
		self._intent_classifier = None
		
		self._log_intent_recognition_setup()
	
	async def _setup_entity_extraction(self):
		"""Set up entity extraction"""
		
		# Initialize entity extractor
		self._entity_extractor = None
		
		self._log_entity_extraction_setup()
	
	async def _setup_sentiment_analysis(self):
		"""Set up sentiment analysis"""
		
		# Initialize sentiment analyzer
		self._sentiment_analyzer = None
		
		self._log_sentiment_analysis_setup()
	
	async def _initialize_conversation_flows(self):
		"""Initialize conversation flows"""
		
		self._log_conversation_flows_initialized()
	
	# Logging methods following APG patterns
	
	def _log_engine_created(self):
		"""Log engine creation"""
		print(f"🗣️  Conversational Payment Engine created")
		print(f"   Engine ID: {self.engine_id}")
		print(f"   Languages: {self.supported_languages}")
		print(f"   Voice Processing: {self.enable_voice_processing}")
		print(f"   Multilingual: {self.enable_multilingual}")
	
	def _log_engine_initialization_start(self):
		"""Log engine initialization start"""
		print(f"🚀 Initializing Conversational Payment Engine...")
		print(f"   Advanced NLP: {self.use_advanced_nlp}")
		print(f"   Sentiment Analysis: {self.enable_sentiment_analysis}")
		print(f"   Voice Biometrics: {self.enable_voice_biometrics}")
	
	def _log_engine_initialization_complete(self):
		"""Log engine initialization complete"""
		print(f"✅ Conversational Payment Engine initialized successfully")
		print(f"   NLP Models: {len(self._nlp_models)}")
		print(f"   Conversation Templates: {len(self._conversation_templates)}")
	
	def _log_engine_initialization_error(self, error: str):
		"""Log engine initialization error"""
		print(f"❌ Conversational Payment Engine initialization failed: {error}")
	
	def _log_message_processing_start(self, conversation_id: str, message_preview: str):
		"""Log message processing start"""
		print(f"💬 Processing message in conversation {conversation_id}: '{message_preview}...'")
	
	def _log_message_processing_complete(self, conversation_id: str, response_preview: str):
		"""Log message processing completion"""
		print(f"✅ Generated response for {conversation_id}: '{response_preview}...'")
	
	def _log_message_processing_error(self, conversation_id: str, error: str):
		"""Log message processing error"""
		print(f"❌ Message processing failed for {conversation_id}: {error}")
	
	def _log_voice_processing_error(self, user_id: str, error: str):
		"""Log voice processing error"""
		print(f"❌ Voice processing failed for user {user_id}: {error}")
	
	def _log_payment_execution_error(self, conversation_id: str, error: str):
		"""Log payment execution error"""
		print(f"❌ Payment execution failed for conversation {conversation_id}: {error}")
	
	def _log_analytics_error(self, error: str):
		"""Log analytics error"""
		print(f"❌ Analytics calculation failed: {error}")
	
	def _log_nlp_model_warning(self, warning: str):
		"""Log NLP model warning"""
		print(f"⚠️  NLP Model Warning: {warning}")
	
	def _log_nlp_models_initialized(self):
		"""Log NLP models initialization"""
		print(f"🧠 NLP models initialized")
	
	def _log_conversation_templates_initialized(self):
		"""Log conversation templates initialization"""
		print(f"💬 Conversation templates initialized")
	
	def _log_intent_recognition_setup(self):
		"""Log intent recognition setup"""
		print(f"🎯 Intent recognition configured")
	
	def _log_entity_extraction_setup(self):
		"""Log entity extraction setup"""
		print(f"🔍 Entity extraction configured")
	
	def _log_sentiment_analysis_setup(self):
		"""Log sentiment analysis setup"""
		print(f"😊 Sentiment analysis configured")
	
	def _log_conversation_flows_initialized(self):
		"""Log conversation flows initialization"""
		print(f"🔄 Conversation flows initialized")

# Factory function for creating conversational payment engine
def create_conversational_payment_engine(config: Dict[str, Any]) -> ConversationalPaymentEngine:
	"""Factory function to create conversational payment engine"""
	return ConversationalPaymentEngine(config)

def _log_conversational_payments_module_loaded():
	"""Log conversational payments module loaded"""
	print("🗣️  Conversational Payment Interface module loaded")
	print("   - Natural language processing")
	print("   - Multi-language support (English, Swahili, French, Spanish)")
	print("   - Voice payment processing")
	print("   - Intelligent conversation flows")
	print("   - Sentiment analysis & empathy responses")
	print("   - Multi-channel support (Voice, SMS, Chat, WhatsApp)")

# Execute module loading log
_log_conversational_payments_module_loaded()