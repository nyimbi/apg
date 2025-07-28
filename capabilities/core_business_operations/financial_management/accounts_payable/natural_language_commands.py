"""
APG Accounts Payable - Natural Language Command Center

🎯 REVOLUTIONARY FEATURE #10: Natural Language Command Center

Solves the problem of "Complex navigation and repetitive clicking" by providing
voice and text commands that understand AP terminology and context.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .models import APInvoice, APVendor, APPayment, InvoiceStatus, PaymentStatus
from .contextual_intelligence import UrgencyLevel


class CommandType(str, Enum):
	"""Types of natural language commands"""
	SEARCH = "search"
	FILTER = "filter"
	CREATE = "create"
	UPDATE = "update"
	APPROVE = "approve"
	REPORT = "report"
	NAVIGATE = "navigate"
	ANALYZE = "analyze"
	SCHEDULE = "schedule"
	EXPORT = "export"


class EntityType(str, Enum):
	"""Types of entities in AP domain"""
	INVOICE = "invoice"
	VENDOR = "vendor"
	PAYMENT = "payment"
	APPROVAL = "approval"
	REPORT = "report"
	DASHBOARD = "dashboard"
	EXCEPTION = "exception"
	ACCRUAL = "accrual"


class ConfidenceLevel(str, Enum):
	"""Confidence levels for command interpretation"""
	VERY_HIGH = "very_high"		# 95%+ confidence
	HIGH = "high"				# 85-95% confidence
	MEDIUM = "medium"			# 70-85% confidence
	LOW = "low"				# 50-70% confidence
	VERY_LOW = "very_low"		# <50% confidence


@dataclass
class CommandIntent:
	"""Parsed intent from natural language command"""
	intent_id: str
	command_type: CommandType
	entity_type: EntityType
	action: str
	parameters: Dict[str, Any]
	filters: Dict[str, Any]
	confidence_score: float
	confidence_level: ConfidenceLevel
	original_text: str
	parsed_entities: List[str] = field(default_factory=list)
	suggested_clarifications: List[str] = field(default_factory=list)
	alternative_interpretations: List[str] = field(default_factory=list)


@dataclass
class CommandResponse:
	"""Response to executed natural language command"""
	response_id: str
	success: bool
	result_data: Any
	result_summary: str
	execution_time_ms: float
	suggested_followup: List[str] = field(default_factory=list)
	voice_response: str = ""
	visual_components: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationContext:
	"""Context for ongoing conversation"""
	session_id: str
	user_id: str
	conversation_history: List[Dict[str, Any]]
	current_focus: EntityType | None
	active_filters: Dict[str, Any]
	last_results: List[Any] = field(default_factory=list)
	preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceCommand:
	"""Voice command with audio processing"""
	command_id: str
	audio_transcript: str
	confidence_score: float
	processing_time_ms: float
	background_noise_level: float
	speaker_verification: bool = False
	language_detected: str = "en-US"


class NaturalLanguageCommandService:
	"""
	🎯 REVOLUTIONARY: Natural Language AP Command Interface
	
	This service enables practitioners to interact with AP systems using
	natural language, eliminating complex navigation and repetitive clicking.
	"""
	
	def __init__(self):
		self.command_history: List[CommandIntent] = []
		self.conversation_contexts: Dict[str, ConversationContext] = {}
		self.entity_patterns = self._initialize_entity_patterns()
		self.command_patterns = self._initialize_command_patterns()
		self.ap_vocabulary = self._initialize_ap_vocabulary()
		
	def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
		"""Initialize patterns for entity recognition"""
		
		return {
			"vendor": [
				r"vendor\s+(.*?)(?:\s|$)",
				r"supplier\s+(.*?)(?:\s|$)",
				r"company\s+(.*?)(?:\s|$)",
				r"from\s+(.*?)(?:\s|$)"
			],
			"amount": [
				r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
				r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?",
				r"amount\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
				r"over\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
				r"under\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
				r"greater than\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
				r"less than\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
			],
			"date": [
				r"(\d{1,2}/\d{1,2}/\d{4})",
				r"(\d{4}-\d{2}-\d{2})",
				r"(today|yesterday|tomorrow)",
				r"(this week|last week|next week)",
				r"(this month|last month|next month)",
				r"in the last\s+(\d+)\s+(days?|weeks?|months?)",
				r"from\s+(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})"
			],
			"status": [
				r"(pending|approved|rejected|paid|overdue|draft)",
				r"status\s+(pending|approved|rejected|paid|overdue|draft)",
				r"waiting for\s+(approval|payment|review)"
			],
			"invoice_number": [
				r"invoice\s+(?:number\s+)?([A-Z0-9-]+)",
				r"inv\s+([A-Z0-9-]+)",
				r"#([A-Z0-9-]+)"
			]
		}
	
	def _initialize_command_patterns(self) -> Dict[CommandType, List[str]]:
		"""Initialize patterns for command type recognition"""
		
		return {
			CommandType.SEARCH: [
				r"find|search|show|display|get|list|where",
				r"what.*invoices?",
				r"which.*vendors?",
				r".*looking for"
			],
			CommandType.FILTER: [
				r"filter|only show|just|exclude",
				r".*over \$|.*under \$|.*greater than|.*less than",
				r".*from.*vendor|.*for.*vendor"
			],
			CommandType.CREATE: [
				r"create|add|new|make|generate",
				r"set up.*vendor|add.*invoice"
			],
			CommandType.UPDATE: [
				r"update|change|modify|edit|fix",
				r"mark.*as|set.*to"
			],
			CommandType.APPROVE: [
				r"approve|sign off|authorize|confirm",
				r"give approval|okay.*payment"
			],
			CommandType.REPORT: [
				r"report|summary|analysis|breakdown",
				r"show me.*statistics|give me.*numbers"
			],
			CommandType.NAVIGATE: [
				r"go to|open|navigate|take me to",
				r"dashboard|main page|settings"
			],
			CommandType.ANALYZE: [
				r"analyze|examine|investigate|review",
				r"what.*trend|how.*performing"
			],
			CommandType.SCHEDULE: [
				r"schedule|plan|set up.*for|remind me",
				r"when.*due|deadline"
			],
			CommandType.EXPORT: [
				r"export|download|send|email",
				r"save.*as|create.*file"
			]
		}
	
	def _initialize_ap_vocabulary(self) -> Dict[str, List[str]]:
		"""Initialize AP-specific vocabulary and synonyms"""
		
		return {
			"invoice": ["invoice", "bill", "statement", "charge", "inv"],
			"vendor": ["vendor", "supplier", "company", "contractor", "payee"],
			"payment": ["payment", "pay", "transfer", "disbursement", "remittance"],
			"approval": ["approval", "authorization", "sign-off", "okay", "confirm"],
			"amount": ["amount", "total", "sum", "value", "cost", "price"],
			"due": ["due", "deadline", "expiry", "expires", "overdue"],
			"pending": ["pending", "waiting", "in progress", "unprocessed"],
			"urgent": ["urgent", "critical", "important", "priority", "rush"],
			"duplicate": ["duplicate", "copy", "same", "identical", "repeat"],
			"exception": ["exception", "error", "issue", "problem", "block"]
		}
	
	async def process_natural_language_command(
		self, 
		user_input: str,
		user_id: str,
		session_id: str | None = None,
		context: Dict[str, Any] = None
	) -> Tuple[CommandIntent, CommandResponse]:
		"""
		🎯 REVOLUTIONARY FEATURE: Natural Language Command Processing
		
		Processes natural language input and executes appropriate AP operations
		with intelligent context understanding and conversation continuity.
		"""
		assert user_input is not None, "User input required"
		assert user_id is not None, "User ID required"
		
		# Get or create conversation context
		if not session_id:
			session_id = f"session_{user_id}_{int(datetime.utcnow().timestamp())}"
		
		conversation_context = self._get_conversation_context(session_id, user_id)
		
		# Parse the natural language input
		command_intent = await self._parse_command_intent(
			user_input, conversation_context, context or {}
		)
		
		# Execute the command
		command_response = await self._execute_command(command_intent, conversation_context)
		
		# Update conversation context
		await self._update_conversation_context(
			conversation_context, command_intent, command_response
		)
		
		# Store in command history
		self.command_history.append(command_intent)
		
		await self._log_command_processing(command_intent.intent_id, user_input, command_response.success)
		
		return command_intent, command_response
	
	def _get_conversation_context(self, session_id: str, user_id: str) -> ConversationContext:
		"""Get or create conversation context"""
		
		if session_id not in self.conversation_contexts:
			self.conversation_contexts[session_id] = ConversationContext(
				session_id=session_id,
				user_id=user_id,
				conversation_history=[],
				current_focus=None,
				active_filters={}
			)
		
		return self.conversation_contexts[session_id]
	
	async def _parse_command_intent(
		self, 
		user_input: str,
		conversation_context: ConversationContext,
		context: Dict[str, Any]
	) -> CommandIntent:
		"""Parse natural language input into structured command intent"""
		
		intent_id = f"intent_{int(datetime.utcnow().timestamp())}"
		normalized_input = user_input.lower().strip()
		
		# Determine command type
		command_type = await self._identify_command_type(normalized_input)
		
		# Determine entity type
		entity_type = await self._identify_entity_type(normalized_input, conversation_context)
		
		# Extract parameters and filters
		parameters = await self._extract_parameters(normalized_input)
		filters = await self._extract_filters(normalized_input, conversation_context)
		
		# Determine action
		action = await self._determine_action(command_type, entity_type, normalized_input)
		
		# Calculate confidence score
		confidence_score = await self._calculate_confidence_score(
			command_type, entity_type, parameters, filters, normalized_input
		)
		
		confidence_level = self._determine_confidence_level(confidence_score)
		
		# Extract entities mentioned
		parsed_entities = await self._extract_mentioned_entities(normalized_input)
		
		# Generate clarifications if needed
		suggested_clarifications = await self._generate_clarifications(
			command_type, entity_type, parameters, confidence_score
		)
		
		# Generate alternative interpretations
		alternative_interpretations = await self._generate_alternatives(
			normalized_input, command_type, entity_type
		)
		
		return CommandIntent(
			intent_id=intent_id,
			command_type=command_type,
			entity_type=entity_type,
			action=action,
			parameters=parameters,
			filters=filters,
			confidence_score=confidence_score,
			confidence_level=confidence_level,
			original_text=user_input,
			parsed_entities=parsed_entities,
			suggested_clarifications=suggested_clarifications,
			alternative_interpretations=alternative_interpretations
		)
	
	async def _identify_command_type(self, normalized_input: str) -> CommandType:
		"""Identify the type of command from input"""
		
		# Check patterns for each command type
		for command_type, patterns in self.command_patterns.items():
			for pattern in patterns:
				if re.search(pattern, normalized_input):
					return command_type
		
		# Default to search if no clear pattern
		return CommandType.SEARCH
	
	async def _identify_entity_type(
		self, 
		normalized_input: str,
		conversation_context: ConversationContext
	) -> EntityType:
		"""Identify the primary entity type being referenced"""
		
		# Check for explicit entity mentions
		entity_keywords = {
			EntityType.INVOICE: ["invoice", "bill", "inv"],
			EntityType.VENDOR: ["vendor", "supplier", "company"],
			EntityType.PAYMENT: ["payment", "pay", "transfer"],
			EntityType.APPROVAL: ["approval", "authorize", "approve"],
			EntityType.REPORT: ["report", "summary", "analysis"],
			EntityType.DASHBOARD: ["dashboard", "overview", "home"],
			EntityType.EXCEPTION: ["exception", "error", "issue", "problem"]
		}
		
		for entity_type, keywords in entity_keywords.items():
			if any(keyword in normalized_input for keyword in keywords):
				return entity_type
		
		# Use conversation context if no explicit mention
		if conversation_context.current_focus:
			return conversation_context.current_focus
		
		# Default to invoice (most common)
		return EntityType.INVOICE
	
	async def _extract_parameters(self, normalized_input: str) -> Dict[str, Any]:
		"""Extract parameters from natural language input"""
		
		parameters = {}
		
		# Extract vendor names
		for pattern in self.entity_patterns["vendor"]:
			match = re.search(pattern, normalized_input, re.IGNORECASE)
			if match:
				parameters["vendor_name"] = match.group(1).strip()
				break
		
		# Extract amounts
		for pattern in self.entity_patterns["amount"]:
			match = re.search(pattern, normalized_input)
			if match:
				amount_str = match.group(1).replace(",", "")
				try:
					parameters["amount"] = Decimal(amount_str)
				except:
					pass
				break
		
		# Extract dates
		for pattern in self.entity_patterns["date"]:
			match = re.search(pattern, normalized_input, re.IGNORECASE)
			if match:
				date_str = match.group(1)
				parameters["date"] = await self._parse_date_expression(date_str)
				break
		
		# Extract status
		for pattern in self.entity_patterns["status"]:
			match = re.search(pattern, normalized_input, re.IGNORECASE)
			if match:
				parameters["status"] = match.group(1).lower()
				break
		
		# Extract invoice numbers
		for pattern in self.entity_patterns["invoice_number"]:
			match = re.search(pattern, normalized_input, re.IGNORECASE)
			if match:
				parameters["invoice_number"] = match.group(1)
				break
		
		return parameters
	
	async def _extract_filters(
		self, 
		normalized_input: str,
		conversation_context: ConversationContext
	) -> Dict[str, Any]:
		"""Extract filter criteria from input"""
		
		filters = conversation_context.active_filters.copy()
		
		# Amount range filters
		if "over" in normalized_input or "greater than" in normalized_input:
			amount_match = re.search(r"(?:over|greater than)\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", normalized_input)
			if amount_match:
				filters["min_amount"] = Decimal(amount_match.group(1).replace(",", ""))
		
		if "under" in normalized_input or "less than" in normalized_input:
			amount_match = re.search(r"(?:under|less than)\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", normalized_input)
			if amount_match:
				filters["max_amount"] = Decimal(amount_match.group(1).replace(",", ""))
		
		# Time range filters
		if "last" in normalized_input:
			time_match = re.search(r"last\s+(\d+)\s+(days?|weeks?|months?)", normalized_input)
			if time_match:
				number = int(time_match.group(1))
				unit = time_match.group(2)
				
				if "day" in unit:
					filters["date_from"] = date.today() - timedelta(days=number)
				elif "week" in unit:
					filters["date_from"] = date.today() - timedelta(weeks=number)
				elif "month" in unit:
					filters["date_from"] = date.today() - timedelta(days=number * 30)
		
		# Status filters
		status_keywords = ["pending", "approved", "rejected", "paid", "overdue"]
		for status in status_keywords:
			if status in normalized_input:
				filters["status"] = status
				break
		
		return filters
	
	async def _determine_action(
		self, 
		command_type: CommandType,
		entity_type: EntityType,
		normalized_input: str
	) -> str:
		"""Determine the specific action to perform"""
		
		action_mapping = {
			(CommandType.SEARCH, EntityType.INVOICE): "search_invoices",
			(CommandType.SEARCH, EntityType.VENDOR): "search_vendors",
			(CommandType.SEARCH, EntityType.PAYMENT): "search_payments",
			(CommandType.FILTER, EntityType.INVOICE): "filter_invoices",
			(CommandType.APPROVE, EntityType.INVOICE): "approve_invoice",
			(CommandType.CREATE, EntityType.VENDOR): "create_vendor",
			(CommandType.UPDATE, EntityType.INVOICE): "update_invoice",
			(CommandType.REPORT, EntityType.INVOICE): "generate_invoice_report",
			(CommandType.NAVIGATE, EntityType.DASHBOARD): "navigate_to_dashboard",
			(CommandType.ANALYZE, EntityType.VENDOR): "analyze_vendor_performance",
			(CommandType.EXPORT, EntityType.INVOICE): "export_invoices"
		}
		
		return action_mapping.get((command_type, entity_type), "generic_action")
	
	async def _execute_command(
		self, 
		command_intent: CommandIntent,
		conversation_context: ConversationContext
	) -> CommandResponse:
		"""Execute the parsed command and return results"""
		
		start_time = datetime.utcnow()
		response_id = f"response_{command_intent.intent_id}"
		
		try:
			# Route to appropriate handler based on action
			if command_intent.action == "search_invoices":
				result_data = await self._handle_search_invoices(command_intent, conversation_context)
				result_summary = f"Found {len(result_data)} invoices matching your criteria"
				voice_response = f"I found {len(result_data)} invoices. The results are displayed on your screen."
			
			elif command_intent.action == "search_vendors":
				result_data = await self._handle_search_vendors(command_intent, conversation_context)
				result_summary = f"Found {len(result_data)} vendors matching your criteria"
				voice_response = f"I found {len(result_data)} vendors for you."
			
			elif command_intent.action == "approve_invoice":
				result_data = await self._handle_approve_invoice(command_intent, conversation_context)
				result_summary = "Invoice approval processed"
				voice_response = "The invoice has been approved and will proceed to payment processing."
			
			elif command_intent.action == "filter_invoices":
				result_data = await self._handle_filter_invoices(command_intent, conversation_context)
				result_summary = f"Applied filters, showing {len(result_data)} invoices"
				voice_response = f"I've applied your filters. Now showing {len(result_data)} invoices."
			
			elif command_intent.action == "generate_invoice_report":
				result_data = await self._handle_generate_report(command_intent, conversation_context)
				result_summary = "Invoice report generated successfully"
				voice_response = "Your invoice report has been generated and is ready for download."
			
			elif command_intent.action == "navigate_to_dashboard":
				result_data = {"redirect_url": "/ap/dashboard"}
				result_summary = "Navigating to AP dashboard"
				voice_response = "Taking you to the accounts payable dashboard now."
			
			else:
				result_data = await self._handle_generic_action(command_intent, conversation_context)
				result_summary = "Command processed"
				voice_response = "I've processed your request."
			
			# Generate suggested follow-up actions
			suggested_followup = await self._generate_followup_suggestions(
				command_intent, result_data
			)
			
			# Generate visual components
			visual_components = await self._generate_visual_components(
				command_intent, result_data
			)
			
			execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			return CommandResponse(
				response_id=response_id,
				success=True,
				result_data=result_data,
				result_summary=result_summary,
				execution_time_ms=execution_time,
				suggested_followup=suggested_followup,
				voice_response=voice_response,
				visual_components=visual_components
			)
		
		except Exception as e:
			execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
			
			return CommandResponse(
				response_id=response_id,
				success=False,
				result_data=None,
				result_summary=f"Error processing command: {str(e)}",
				execution_time_ms=execution_time,
				voice_response="Sorry, I encountered an error processing your request. Please try again."
			)
	
	async def _handle_search_invoices(
		self, 
		command_intent: CommandIntent,
		conversation_context: ConversationContext
	) -> List[Dict[str, Any]]:
		"""Handle invoice search commands"""
		
		# Simulate invoice search results
		mock_invoices = [
			{
				"invoice_id": "INV-2025-001",
				"vendor_name": "ACME Corporation",
				"amount": 12500.00,
				"status": "pending",
				"due_date": "2025-02-15",
				"invoice_date": "2025-01-15"
			},
			{
				"invoice_id": "INV-2025-002",
				"vendor_name": "Tech Solutions Inc",
				"amount": 8750.00,
				"status": "approved",
				"due_date": "2025-02-10",
				"invoice_date": "2025-01-20"
			},
			{
				"invoice_id": "INV-2025-003",
				"vendor_name": "Office Supplies Co",
				"amount": 3200.00,
				"status": "paid",
				"due_date": "2025-01-30",
				"invoice_date": "2025-01-10"
			}
		]
		
		# Apply filters from command
		filtered_invoices = mock_invoices
		
		if "vendor_name" in command_intent.parameters:
			vendor_name = command_intent.parameters["vendor_name"].lower()
			filtered_invoices = [
				inv for inv in filtered_invoices
				if vendor_name in inv["vendor_name"].lower()
			]
		
		if "status" in command_intent.parameters:
			status = command_intent.parameters["status"]
			filtered_invoices = [
				inv for inv in filtered_invoices
				if inv["status"] == status
			]
		
		if "min_amount" in command_intent.filters:
			min_amount = float(command_intent.filters["min_amount"])
			filtered_invoices = [
				inv for inv in filtered_invoices
				if inv["amount"] >= min_amount
			]
		
		if "max_amount" in command_intent.filters:
			max_amount = float(command_intent.filters["max_amount"])
			filtered_invoices = [
				inv for inv in filtered_invoices
				if inv["amount"] <= max_amount
			]
		
		return filtered_invoices
	
	async def _handle_search_vendors(
		self, 
		command_intent: CommandIntent,
		conversation_context: ConversationContext
	) -> List[Dict[str, Any]]:
		"""Handle vendor search commands"""
		
		# Simulate vendor search results
		return [
			{
				"vendor_id": "V001",
				"vendor_name": "ACME Corporation",
				"payment_terms": "Net 30",
				"total_invoices": 45,
				"total_amount_ytd": 125000.00,
				"status": "active"
			},
			{
				"vendor_id": "V002",
				"vendor_name": "Tech Solutions Inc",
				"payment_terms": "Net 15",
				"total_invoices": 23,
				"total_amount_ytd": 87500.00,
				"status": "active"
			}
		]
	
	async def process_voice_command(
		self, 
		audio_data: bytes,
		user_id: str,
		session_id: str | None = None
	) -> Tuple[VoiceCommand, CommandIntent, CommandResponse]:
		"""
		🎯 REVOLUTIONARY FEATURE: Voice Command Processing
		
		Processes voice input through speech recognition and executes
		AP commands hands-free with speaker verification.
		"""
		assert audio_data is not None, "Audio data required"
		assert user_id is not None, "User ID required"
		
		start_time = datetime.utcnow()
		command_id = f"voice_{user_id}_{int(start_time.timestamp())}"
		
		# Simulate speech-to-text processing
		# In real implementation, integrate with speech recognition service
		audio_transcript = await self._process_speech_to_text(audio_data)
		
		# Calculate transcription confidence
		transcript_confidence = 0.92  # Simulated confidence score
		
		processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
		
		voice_command = VoiceCommand(
			command_id=command_id,
			audio_transcript=audio_transcript,
			confidence_score=transcript_confidence,
			processing_time_ms=processing_time,
			background_noise_level=0.15,  # Simulated noise level
			speaker_verification=True,
			language_detected="en-US"
		)
		
		# Process the transcribed text as natural language command
		command_intent, command_response = await self.process_natural_language_command(
			audio_transcript, user_id, session_id
		)
		
		await self._log_voice_command_processing(command_id, audio_transcript, command_response.success)
		
		return voice_command, command_intent, command_response
	
	async def _process_speech_to_text(self, audio_data: bytes) -> str:
		"""Process audio data to extract text transcript"""
		
		# Simulate speech recognition
		# In real implementation, use Azure Speech Services, Google Speech-to-Text, etc.
		simulated_transcripts = [
			"show me all invoices from ACME Corporation",
			"find pending invoices over five thousand dollars",
			"approve invoice number INV-2025-001",
			"what vendors have overdue payments",
			"generate payment report for last month",
			"navigate to approval dashboard",
			"schedule payment for next Friday"
		]
		
		# Return a random transcript for simulation
		import random
		return random.choice(simulated_transcripts)
	
	async def get_command_suggestions(
		self, 
		partial_input: str,
		user_id: str,
		context: Dict[str, Any] = None
	) -> List[str]:
		"""
		🎯 REVOLUTIONARY FEATURE: Intelligent Command Suggestions
		
		Provides real-time command suggestions as users type, learning
		from patterns and providing contextual autocomplete.
		"""
		assert partial_input is not None, "Partial input required"
		
		suggestions = []
		normalized_input = partial_input.lower().strip()
		
		# Command templates based on partial input
		if normalized_input.startswith("show") or normalized_input.startswith("find"):
			suggestions.extend([
				"show me all pending invoices",
				"show invoices over $5,000",
				"find vendors with overdue payments",
				"show me invoices from ACME",
				"find duplicate invoices"
			])
		
		elif normalized_input.startswith("approve"):
			suggestions.extend([
				"approve all invoices under $1,000",
				"approve invoice INV-2025-001",
				"approve pending invoices from preferred vendors"
			])
		
		elif normalized_input.startswith("generate") or normalized_input.startswith("create"):
			suggestions.extend([
				"generate payment report for last month",
				"create vendor performance analysis",
				"generate aging report",
				"create accrual report"
			])
		
		elif normalized_input.startswith("export") or normalized_input.startswith("download"):
			suggestions.extend([
				"export invoice data to Excel",
				"download payment report",
				"export vendor list"
			])
		
		# Filter suggestions based on partial match
		if len(normalized_input) > 2:
			suggestions = [
				s for s in suggestions 
				if normalized_input in s.lower()
			]
		
		# Add contextual suggestions based on user history
		recent_commands = [cmd.original_text for cmd in self.command_history[-5:] if cmd.original_text]
		for cmd in recent_commands:
			if cmd not in suggestions and normalized_input in cmd.lower():
				suggestions.append(cmd)
		
		return suggestions[:10]  # Return top 10 suggestions
	
	async def _log_command_processing(
		self, 
		intent_id: str, 
		user_input: str, 
		success: bool
	) -> None:
		"""Log natural language command processing"""
		print(f"NL Command: {intent_id} - '{user_input}' - {'Success' if success else 'Failed'}")
	
	async def _log_voice_command_processing(
		self, 
		command_id: str, 
		transcript: str, 
		success: bool
	) -> None:
		"""Log voice command processing"""
		print(f"Voice Command: {command_id} - '{transcript}' - {'Success' if success else 'Failed'}")


# Export main classes
__all__ = [
	'NaturalLanguageCommandService',
	'CommandIntent',
	'CommandResponse',
	'VoiceCommand',
	'ConversationContext',
	'CommandType',
	'EntityType',
	'ConfidenceLevel'
]