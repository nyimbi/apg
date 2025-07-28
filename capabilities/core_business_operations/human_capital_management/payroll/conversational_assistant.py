"""
APG Payroll Management - Conversational Payroll Assistant

Revolutionary AI-powered conversational interface for payroll operations.
Natural language processing, voice commands, and intelligent automation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, ConfigDict

# APG Platform Imports
from ...ai_orchestration.services import (
	AIOrchestrationService, 
	NLPService, 
	VoiceProcessingService,
	ConversationService
)
from ...ai_orchestration.models import (
	ConversationContext,
	NLPRequest,
	NLPResponse,
	IntentPrediction,
	EntityExtraction
)
from ...notification_engine.services import NotificationService
from ...employee_data_management.services import EmployeeDataService
from .models import (
	PRPayrollPeriod, PRPayrollRun, PREmployeePayroll, PRPayComponent,
	PayrollStatus, PayComponentType, PayFrequency
)
from .ai_intelligence_engine import PayrollIntelligenceEngine

# Configure logging
logger = logging.getLogger(__name__)


class ConversationIntent(str, Enum):
	"""Supported conversation intents for payroll operations."""
	
	# Payroll Information Queries
	GET_PAY_STUB = "get_pay_stub"
	GET_PAYROLL_SUMMARY = "get_payroll_summary"
	GET_EMPLOYEE_PAYROLL = "get_employee_payroll"
	GET_PAYROLL_STATUS = "get_payroll_status"
	
	# Payroll Processing Commands
	START_PAYROLL_RUN = "start_payroll_run"
	APPROVE_PAYROLL = "approve_payroll"
	CANCEL_PAYROLL = "cancel_payroll"
	FINALIZE_PAYROLL = "finalize_payroll"
	
	# Analytics and Reporting
	GET_PAYROLL_ANALYTICS = "get_payroll_analytics"
	COMPARE_PAYROLL_PERIODS = "compare_payroll_periods"
	DETECT_ANOMALIES = "detect_anomalies"
	FORECAST_COSTS = "forecast_costs"
	
	# Employee Management
	ADD_EMPLOYEE_TO_PAYROLL = "add_employee_to_payroll"
	UPDATE_EMPLOYEE_PAYROLL = "update_employee_payroll"
	REMOVE_EMPLOYEE_FROM_PAYROLL = "remove_employee_from_payroll"
	
	# Compliance and Validation
	CHECK_COMPLIANCE = "check_compliance"
	VALIDATE_PAYROLL = "validate_payroll"
	GET_TAX_CALCULATIONS = "get_tax_calculations"
	
	# Help and Guidance
	GET_HELP = "get_help"
	EXPLAIN_CALCULATION = "explain_calculation"
	TROUBLESHOOT_ISSUE = "troubleshoot_issue"
	
	# Voice Commands
	VOICE_COMMAND = "voice_command"
	
	# Unknown Intent
	UNKNOWN = "unknown"


class ResponseType(str, Enum):
	"""Types of responses from the conversational assistant."""
	TEXT = "text"
	DATA = "data"
	ACTION = "action"
	CONFIRMATION = "confirmation"
	ERROR = "error"
	HELP = "help"
	VOICE = "voice"


@dataclass
class ConversationEntity:
	"""Extracted entity from conversation."""
	entity_type: str
	entity_value: str
	confidence: float
	start_position: int
	end_position: int


@dataclass
class ConversationResponse:
	"""Response from the conversational assistant."""
	response_type: ResponseType
	message: str
	data: Optional[Dict[str, Any]] = None
	suggested_actions: Optional[List[str]] = None
	requires_confirmation: bool = False
	confidence_score: float = 1.0
	voice_response: Optional[str] = None
	context_updates: Optional[Dict[str, Any]] = None


class ConversationConfig(BaseModel):
	"""Configuration for conversational assistant."""
	model_config = ConfigDict(extra='forbid')
	
	# NLP Settings
	nlp_model: str = Field(default="gpt-4")
	intent_confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
	entity_confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)
	
	# Voice Settings
	voice_enabled: bool = Field(default=True)
	voice_language: str = Field(default="en-US")
	voice_response_enabled: bool = Field(default=True)
	
	# Context Settings
	context_memory_turns: int = Field(default=10, ge=1, le=50)
	session_timeout_minutes: int = Field(default=30, ge=5, le=120)
	
	# Security Settings
	require_authentication: bool = Field(default=True)
	sensitive_data_masking: bool = Field(default=True)
	audit_conversations: bool = Field(default=True)
	
	# Response Settings
	max_response_length: int = Field(default=500, ge=50, le=2000)
	include_suggestions: bool = Field(default=True)
	personalization_enabled: bool = Field(default=True)


class ConversationalPayrollAssistant:
	"""Revolutionary conversational interface for payroll operations.
	
	Provides natural language processing, voice commands, and intelligent
	automation for all payroll-related tasks and queries.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		ai_service: AIOrchestrationService,
		nlp_service: NLPService,
		voice_service: VoiceProcessingService,
		conversation_service: ConversationService,
		notification_service: NotificationService,
		employee_service: EmployeeDataService,
		intelligence_engine: PayrollIntelligenceEngine,
		config: Optional[ConversationConfig] = None
	):
		self.db = db_session
		self.ai_service = ai_service
		self.nlp_service = nlp_service
		self.voice_service = voice_service
		self.conversation_service = conversation_service
		self.notification_service = notification_service
		self.employee_service = employee_service
		self.intelligence_engine = intelligence_engine
		self.config = config or ConversationConfig()
		
		# Intent and entity mappings
		self._intent_patterns = self._initialize_intent_patterns()
		self._entity_patterns = self._initialize_entity_patterns()
		
		# Conversation context storage
		self._conversation_contexts = {}
		
		# Response templates
		self._response_templates = self._initialize_response_templates()
	
	def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
		"""Initialize intent recognition patterns."""
		return {
			ConversationIntent.GET_PAY_STUB: [
				r"show me (?:my )?pay stub",
				r"get (?:my )?pay stub for (.+)",
				r"payslip for (.+)",
				r"earnings statement",
				r"what did (.+) earn"
			],
			ConversationIntent.GET_PAYROLL_SUMMARY: [
				r"payroll summary",
				r"show me payroll totals",
				r"total payroll cost",
				r"payroll overview",
				r"summarize payroll"
			],
			ConversationIntent.START_PAYROLL_RUN: [
				r"start payroll",
				r"run payroll for (.+)",
				r"process payroll",
				r"begin payroll run",
				r"initiate payroll processing"
			],
			ConversationIntent.APPROVE_PAYROLL: [
				r"approve payroll",
				r"approve the payroll run",
				r"give approval for payroll",
				r"sign off on payroll"
			],
			ConversationIntent.GET_PAYROLL_STATUS: [
				r"payroll status",
				r"what's the status of payroll",
				r"is payroll ready",
				r"payroll progress",
				r"how is payroll doing"
			],
			ConversationIntent.GET_PAYROLL_ANALYTICS: [
				r"payroll analytics",
				r"show me payroll trends",
				r"payroll insights",
				r"analyze payroll data",
				r"payroll dashboard"
			],
			ConversationIntent.DETECT_ANOMALIES: [
				r"check for anomalies",
				r"find payroll errors",
				r"detect issues",
				r"what's wrong with payroll",
				r"payroll problems"
			],
			ConversationIntent.CHECK_COMPLIANCE: [
				r"check compliance",
				r"is payroll compliant",
				r"compliance status",
				r"regulatory check",
				r"audit payroll"
			],
			ConversationIntent.GET_HELP: [
				r"help",
				r"what can you do",
				r"available commands",
				r"how to use",
				r"instructions"
			]
		}
	
	def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
		"""Initialize entity extraction patterns."""
		return {
			"employee_name": [
				r"(?:for |of |employee )?([A-Z][a-z]+ [A-Z][a-z]+)",
				r"employee ([A-Z][a-z]+ [A-Z][a-z]+)"
			],
			"employee_id": [
				r"employee (?:id |#)?(\d+)",
				r"emp (?:id )?(\d+)",
				r"ID (\d+)"
			],
			"period": [
				r"(?:for |in |period )?(January|February|March|April|May|June|July|August|September|October|November|December) (\d{4})",
				r"(?:for |in |period )?(\d{1,2})/(\d{4})",
				r"(?:for |in |period )?Q([1-4]) (\d{4})",
				r"this (?:week|month|quarter|year)",
				r"last (?:week|month|quarter|year)"
			],
			"date": [
				r"(\d{1,2})/(\d{1,2})/(\d{4})",
				r"(\d{4})-(\d{1,2})-(\d{1,2})",
				r"(today|yesterday|tomorrow)"
			],
			"amount": [
				r"\$?([\d,]+\.?\d*)",
				r"([\d,]+\.?\d*) dollars?"
			],
			"department": [
				r"(?:department |dept )?([A-Z][a-zA-Z\s]+(?:Department|Dept)?)",
				r"in ([A-Z][a-zA-Z\s]+)"
			]
		}
	
	def _initialize_response_templates(self) -> Dict[str, str]:
		"""Initialize response templates for different scenarios."""
		return {
			"greeting": "Hello! I'm your AI payroll assistant. I can help you with payroll processing, employee information, analytics, and more. What would you like to do today?",
			
			"payroll_summary": "Here's your payroll summary for {period}:\n• Total Employees: {employee_count}\n• Gross Pay: ${gross_pay:,.2f}\n• Total Taxes: ${total_taxes:,.2f}\n• Net Pay: ${net_pay:,.2f}\n• Processing Status: {status}",
			
			"employee_payroll": "Payroll information for {employee_name}:\n• Gross Earnings: ${gross_earnings:,.2f}\n• Deductions: ${deductions:,.2f}\n• Taxes: ${taxes:,.2f}\n• Net Pay: ${net_pay:,.2f}\n• Hours: {hours}",
			
			"payroll_started": "✅ Payroll run has been started for {period}. Run ID: {run_id}\nEstimated completion time: {estimated_time} minutes.\nI'll notify you when it's ready for review.",
			
			"anomalies_found": "🚨 I found {anomaly_count} anomalies in the current payroll:\n{anomaly_list}\n\nWould you like me to investigate these further or apply automatic fixes where possible?",
			
			"compliance_check": "✅ Compliance check complete:\n• Tax Calculations: {tax_status}\n• Minimum Wage: {wage_status}\n• Overtime Rules: {overtime_status}\n• Overall Score: {compliance_score}%",
			
			"help_response": "I can help you with:\n• Payroll processing and approval\n• Employee pay information\n• Payroll analytics and insights\n• Anomaly detection\n• Compliance checking\n• Voice commands\n\nJust ask me naturally! For example: 'Show me John's pay stub' or 'Start payroll for this month'",
			
			"error": "I apologize, but I encountered an error: {error_message}. Please try again or contact support if the issue persists.",
			
			"unknown_intent": "I'm not sure I understand. Could you please rephrase your request? You can also say 'help' to see what I can do."
		}
	
	async def process_conversation(
		self,
		user_input: str,
		user_id: str,
		tenant_id: str,
		session_id: Optional[str] = None,
		is_voice: bool = False
	) -> ConversationResponse:
		"""Process a conversational input and return an appropriate response."""
		
		try:
			logger.info(f"Processing conversation input from user {user_id}: {user_input[:100]}...")
			
			# Get or create conversation context
			context = await self._get_conversation_context(user_id, session_id, tenant_id)
			
			# Process voice input if applicable
			if is_voice:
				user_input = await self._process_voice_input(user_input)
			
			# Extract intent and entities
			intent, entities = await self._extract_intent_and_entities(user_input, context)
			
			# Validate user permissions
			if not await self._validate_user_permissions(user_id, intent, tenant_id):
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="You don't have permission to perform this action.",
					confidence_score=1.0
				)
			
			# Process the intent
			response = await self._process_intent(intent, entities, context, user_input)
			
			# Update conversation context
			await self._update_conversation_context(context, user_input, response)
			
			# Generate voice response if enabled
			if self.config.voice_response_enabled and (is_voice or context.voice_mode):
				response.voice_response = await self._generate_voice_response(response.message)
			
			# Log conversation for audit
			if self.config.audit_conversations:
				await self._log_conversation(user_id, user_input, response, context)
			
			logger.info(f"Conversation processed successfully for user {user_id}")
			return response
			
		except Exception as e:
			logger.error(f"Conversation processing failed: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message=f"I encountered an error processing your request: {str(e)}",
				confidence_score=0.0
			)
	
	async def _process_voice_input(self, audio_input: str) -> str:
		"""Process voice input and convert to text."""
		try:
			# Convert audio to text using voice service
			transcription = await self.voice_service.speech_to_text(
				audio_data=audio_input,
				language=self.config.voice_language
			)
			return transcription.text
		except Exception as e:
			logger.error(f"Voice processing failed: {e}")
			return "I couldn't understand the voice input. Please try again."
	
	async def _extract_intent_and_entities(
		self, 
		user_input: str, 
		context: ConversationContext
	) -> Tuple[ConversationIntent, List[ConversationEntity]]:
		"""Extract intent and entities from user input."""
		
		# Use NLP service for intent classification
		nlp_request = NLPRequest(
			text=user_input,
			context=context.to_dict(),
			tasks=["intent_classification", "entity_extraction"],
			model=self.config.nlp_model
		)
		
		nlp_response = await self.nlp_service.process(nlp_request)
		
		# Extract intent
		intent = ConversationIntent.UNKNOWN
		if nlp_response.intent_prediction:
			if nlp_response.intent_prediction.confidence >= self.config.intent_confidence_threshold:
				intent = ConversationIntent(nlp_response.intent_prediction.intent)
		
		# Fallback to pattern matching if NLP confidence is low
		if intent == ConversationIntent.UNKNOWN:
			intent = self._match_intent_patterns(user_input)
		
		# Extract entities
		entities = []
		if nlp_response.entities:
			for entity in nlp_response.entities:
				if entity.confidence >= self.config.entity_confidence_threshold:
					entities.append(ConversationEntity(
						entity_type=entity.entity_type,
						entity_value=entity.value,
						confidence=entity.confidence,
						start_position=entity.start_position,
						end_position=entity.end_position
					))
		
		# Fallback to pattern matching for entities
		pattern_entities = self._extract_entities_with_patterns(user_input)
		entities.extend(pattern_entities)
		
		logger.info(f"Extracted intent: {intent}, entities: {len(entities)}")
		return intent, entities
	
	def _match_intent_patterns(self, user_input: str) -> ConversationIntent:
		"""Match intent using regex patterns."""
		user_input_lower = user_input.lower()
		
		for intent, patterns in self._intent_patterns.items():
			for pattern in patterns:
				if re.search(pattern, user_input_lower):
					return ConversationIntent(intent)
		
		return ConversationIntent.UNKNOWN
	
	def _extract_entities_with_patterns(self, user_input: str) -> List[ConversationEntity]:
		"""Extract entities using regex patterns."""
		entities = []
		
		for entity_type, patterns in self._entity_patterns.items():
			for pattern in patterns:
				matches = re.finditer(pattern, user_input, re.IGNORECASE)
				for match in matches:
					entities.append(ConversationEntity(
						entity_type=entity_type,
						entity_value=match.group(1) if match.groups() else match.group(0),
						confidence=0.8,  # Pattern matching confidence
						start_position=match.start(),
						end_position=match.end()
					))
		
		return entities
	
	async def _process_intent(
		self,
		intent: ConversationIntent,
		entities: List[ConversationEntity],
		context: ConversationContext,
		user_input: str
	) -> ConversationResponse:
		"""Process the identified intent and return appropriate response."""
		
		# Create entity dictionary for easier access
		entity_dict = {entity.entity_type: entity.entity_value for entity in entities}
		
		# Route to appropriate handler
		if intent == ConversationIntent.GET_PAY_STUB:
			return await self._handle_get_pay_stub(entity_dict, context)
		
		elif intent == ConversationIntent.GET_PAYROLL_SUMMARY:
			return await self._handle_get_payroll_summary(entity_dict, context)
		
		elif intent == ConversationIntent.GET_EMPLOYEE_PAYROLL:
			return await self._handle_get_employee_payroll(entity_dict, context)
		
		elif intent == ConversationIntent.START_PAYROLL_RUN:
			return await self._handle_start_payroll_run(entity_dict, context)
		
		elif intent == ConversationIntent.APPROVE_PAYROLL:
			return await self._handle_approve_payroll(entity_dict, context)
		
		elif intent == ConversationIntent.GET_PAYROLL_STATUS:
			return await self._handle_get_payroll_status(entity_dict, context)
		
		elif intent == ConversationIntent.GET_PAYROLL_ANALYTICS:
			return await self._handle_get_payroll_analytics(entity_dict, context)
		
		elif intent == ConversationIntent.DETECT_ANOMALIES:
			return await self._handle_detect_anomalies(entity_dict, context)
		
		elif intent == ConversationIntent.CHECK_COMPLIANCE:
			return await self._handle_check_compliance(entity_dict, context)
		
		elif intent == ConversationIntent.GET_HELP:
			return await self._handle_get_help(entity_dict, context)
		
		else:
			return await self._handle_unknown_intent(user_input, context)
	
	async def _handle_get_pay_stub(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle pay stub retrieval requests."""
		
		try:
			# Determine employee to get pay stub for
			employee_id = None
			employee_name = None
			
			if "employee_id" in entities:
				employee_id = entities["employee_id"]
			elif "employee_name" in entities:
				employee_name = entities["employee_name"]
				# Look up employee ID by name
				employee_id = await self._get_employee_id_by_name(employee_name, context.tenant_id)
			else:
				# Default to current user if they're an employee
				employee_id = await self._get_current_user_employee_id(context.user_id, context.tenant_id)
			
			if not employee_id:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="I couldn't find the employee. Please specify an employee name or ID.",
					suggested_actions=["Try: 'Show me John Smith's pay stub'", "Or: 'Get pay stub for employee 123'"]
				)
			
			# Determine period
			period = entities.get("period", "current")
			
			# Get payroll data
			payroll_data = await self._get_employee_payroll_data(employee_id, period, context.tenant_id)
			
			if not payroll_data:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message=f"No payroll data found for the specified employee and period.",
					suggested_actions=["Check the employee name and period", "Try a different time period"]
				)
			
			# Format response
			message = self._response_templates["employee_payroll"].format(
				employee_name=payroll_data.get("employee_name", "Unknown"),
				gross_earnings=payroll_data.get("gross_earnings", 0),
				deductions=payroll_data.get("total_deductions", 0),
				taxes=payroll_data.get("total_taxes", 0),
				net_pay=payroll_data.get("net_pay", 0),
				hours=payroll_data.get("regular_hours", 0)
			)
			
			return ConversationResponse(
				response_type=ResponseType.DATA,
				message=message,
				data=payroll_data,
				suggested_actions=["View detailed breakdown", "Compare with previous period", "Download pay stub PDF"]
			)
			
		except Exception as e:
			logger.error(f"Failed to get pay stub: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message="I couldn't retrieve the pay stub. Please try again.",
				confidence_score=0.5
			)
	
	async def _handle_get_payroll_summary(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle payroll summary requests."""
		
		try:
			# Determine period
			period = entities.get("period", "current")
			
			# Get payroll summary data
			summary_data = await self._get_payroll_summary_data(period, context.tenant_id)
			
			if not summary_data:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="No payroll data found for the specified period.",
					suggested_actions=["Try a different period", "Check if payroll has been processed"]
				)
			
			# Format response
			message = self._response_templates["payroll_summary"].format(
				period=summary_data.get("period_name", period),
				employee_count=summary_data.get("employee_count", 0),
				gross_pay=summary_data.get("total_gross", 0),
				total_taxes=summary_data.get("total_taxes", 0),
				net_pay=summary_data.get("total_net", 0),
				status=summary_data.get("status", "Unknown")
			)
			
			# Add insights if available
			if summary_data.get("insights"):
				message += "\\n\\n📊 Key Insights:\\n" + "\\n".join(summary_data["insights"])
			
			return ConversationResponse(
				response_type=ResponseType.DATA,
				message=message,
				data=summary_data,
				suggested_actions=["View detailed analytics", "Compare with previous period", "Check for anomalies"]
			)
			
		except Exception as e:
			logger.error(f"Failed to get payroll summary: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message="I couldn't retrieve the payroll summary. Please try again."
			)
	
	async def _handle_start_payroll_run(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle payroll run start requests."""
		
		try:
			# Determine period
			period = entities.get("period", "current")
			
			# Check if user has permission to start payroll
			if not await self._check_payroll_permission(context.user_id, "start_payroll", context.tenant_id):
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="You don't have permission to start payroll runs."
				)
			
			# Get period information
			period_info = await self._get_period_info(period, context.tenant_id)
			if not period_info:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="I couldn't find the specified payroll period.",
					suggested_actions=["Check the period name", "Create a new payroll period first"]
				)
			
			# Check if payroll is already running
			existing_run = await self._get_active_payroll_run(period_info["period_id"], context.tenant_id)
			if existing_run:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message=f"Payroll is already running for {period}. Current status: {existing_run['status']}",
					suggested_actions=["Check payroll status", "Wait for current run to complete"]
				)
			
			# Start payroll run
			run_info = await self._start_payroll_run(period_info["period_id"], context.user_id, context.tenant_id)
			
			# Estimate completion time
			estimated_time = await self._estimate_payroll_completion_time(run_info["run_id"])
			
			message = self._response_templates["payroll_started"].format(
				period=period_info["period_name"],
				run_id=run_info["run_id"],
				estimated_time=estimated_time
			)
			
			return ConversationResponse(
				response_type=ResponseType.ACTION,
				message=message,
				data=run_info,
				suggested_actions=["Monitor progress", "Check for anomalies", "View processing details"]
			)
			
		except Exception as e:
			logger.error(f"Failed to start payroll run: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message="I couldn't start the payroll run. Please try again or contact support."
			)
	
	async def _handle_detect_anomalies(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle anomaly detection requests."""
		
		try:
			# Get current or specified payroll run
			run_id = await self._get_current_payroll_run_id(context.tenant_id)
			if not run_id:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="No active payroll run found to check for anomalies.",
					suggested_actions=["Start a payroll run first", "Specify a payroll period"]
				)
			
			# Run anomaly detection
			analysis_report = await self.intelligence_engine.analyze_payroll_run(run_id, context.tenant_id)
			anomalies = analysis_report.get("anomalies", [])
			
			if not anomalies:
				return ConversationResponse(
					response_type=ResponseType.DATA,
					message="✅ Great news! I didn't find any significant anomalies in the current payroll.",
					data={"anomaly_count": 0, "health_score": analysis_report.get("overall_health_score", 100)},
					suggested_actions=["View payroll summary", "Proceed with approval", "Generate reports"]
				)
			
			# Format anomaly information
			anomaly_list = []
			for anomaly in anomalies[:5]:  # Show top 5 anomalies
				anomaly_list.append(f"• {anomaly.get('employee_name', 'Unknown')}: {anomaly.get('description', 'Unknown issue')}")
			
			if len(anomalies) > 5:
				anomaly_list.append(f"• ... and {len(anomalies) - 5} more")
			
			message = self._response_templates["anomalies_found"].format(
				anomaly_count=len(anomalies),
				anomaly_list="\\n".join(anomaly_list)
			)
			
			return ConversationResponse(
				response_type=ResponseType.DATA,
				message=message,
				data={"anomalies": anomalies, "analysis_report": analysis_report},
				suggested_actions=["Investigate anomalies", "Apply automatic fixes", "Review manually"],
				requires_confirmation=True
			)
			
		except Exception as e:
			logger.error(f"Failed to detect anomalies: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message="I couldn't check for anomalies. Please try again."
			)
	
	async def _handle_check_compliance(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle compliance check requests."""
		
		try:
			# Get current payroll run
			run_id = await self._get_current_payroll_run_id(context.tenant_id)
			if not run_id:
				return ConversationResponse(
					response_type=ResponseType.ERROR,
					message="No active payroll run found to check compliance.",
					suggested_actions=["Start a payroll run first"]
				)
			
			# Run compliance analysis
			analysis_report = await self.intelligence_engine.analyze_payroll_run(run_id, context.tenant_id)
			compliance_data = analysis_report.get("compliance_risks", [])
			
			# Calculate compliance scores
			tax_status = "✅ Compliant"
			wage_status = "✅ Compliant"
			overtime_status = "✅ Compliant"
			overall_score = 95  # Default high score
			
			# Check for compliance issues
			for risk in compliance_data:
				if risk.get("risk_level") == "high":
					overall_score -= 20
					if "tax" in risk.get("target_entity", "").lower():
						tax_status = "⚠️ Issues Found"
					elif "wage" in risk.get("target_entity", "").lower():
						wage_status = "⚠️ Issues Found"
					elif "overtime" in risk.get("target_entity", "").lower():
						overtime_status = "⚠️ Issues Found"
			
			message = self._response_templates["compliance_check"].format(
				tax_status=tax_status,
				wage_status=wage_status,
				overtime_status=overtime_status,
				compliance_score=overall_score
			)
			
			if compliance_data:
				message += "\\n\\n🔍 Compliance Issues Found:\\n"
				for risk in compliance_data[:3]:  # Show top 3 issues
					message += f"• {risk.get('description', 'Unknown issue')}\\n"
			
			return ConversationResponse(
				response_type=ResponseType.DATA,
				message=message,
				data={
					"compliance_score": overall_score,
					"compliance_risks": compliance_data,
					"tax_compliant": "✅" in tax_status,
					"wage_compliant": "✅" in wage_status,
					"overtime_compliant": "✅" in overtime_status
				},
				suggested_actions=["View detailed compliance report", "Address compliance issues", "Get regulatory guidance"]
			)
			
		except Exception as e:
			logger.error(f"Failed to check compliance: {e}")
			return ConversationResponse(
				response_type=ResponseType.ERROR,
				message="I couldn't check compliance. Please try again."
			)
	
	async def _handle_get_help(
		self, 
		entities: Dict[str, str], 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle help requests."""
		
		message = self._response_templates["help_response"]
		
		# Add personalized suggestions based on user role
		user_role = context.user_data.get("role", "employee")
		if user_role in ["payroll_admin", "hr_admin"]:
			message += "\\n\\nAs an admin, you can also:\\n• Start and approve payroll runs\\n• Access analytics and reports\\n• Manage compliance settings"
		
		return ConversationResponse(
			response_type=ResponseType.HELP,
			message=message,
			suggested_actions=[
				"Show me payroll summary",
				"Check for anomalies",
				"Start payroll for this month",
				"What's the status of payroll?"
			]
		)
	
	async def _handle_unknown_intent(
		self, 
		user_input: str, 
		context: ConversationContext
	) -> ConversationResponse:
		"""Handle unknown or unclear intents."""
		
		# Try to provide helpful suggestions based on keywords
		suggestions = []
		user_input_lower = user_input.lower()
		
		if any(word in user_input_lower for word in ["pay", "salary", "earning"]):
			suggestions.append("Show me pay stub for [employee name]")
		
		if any(word in user_input_lower for word in ["total", "summary", "overview"]):
			suggestions.append("Show me payroll summary")
		
		if any(word in user_input_lower for word in ["start", "run", "process"]):
			suggestions.append("Start payroll for [period]")
		
		if any(word in user_input_lower for word in ["error", "problem", "wrong"]):
			suggestions.append("Check for anomalies")
		
		if not suggestions:
			suggestions = ["Show me payroll summary", "Check payroll status", "Help"]
		
		return ConversationResponse(
			response_type=ResponseType.ERROR,
			message=self._response_templates["unknown_intent"],
			suggested_actions=suggestions,
			confidence_score=0.3
		)
	
	# Helper methods for data access and operations
	
	async def _get_conversation_context(
		self, 
		user_id: str, 
		session_id: Optional[str], 
		tenant_id: str
	) -> ConversationContext:
		"""Get or create conversation context."""
		
		context_key = f"{user_id}_{session_id or 'default'}"
		
		if context_key not in self._conversation_contexts:
			# Create new context
			user_data = await self._get_user_data(user_id, tenant_id)
			
			context = ConversationContext(
				user_id=user_id,
				session_id=session_id or f"session_{datetime.now().timestamp()}",
				tenant_id=tenant_id,
				user_data=user_data,
				conversation_history=[],
				entities={},
				preferences={},
				voice_mode=False,
				created_at=datetime.utcnow(),
				last_activity=datetime.utcnow()
			)
			
			self._conversation_contexts[context_key] = context
		
		return self._conversation_contexts[context_key]
	
	async def _get_user_data(self, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get user data for context."""
		try:
			# This would integrate with auth_rbac service
			return {
				"role": "payroll_admin",  # Default for now
				"permissions": ["read_payroll", "write_payroll", "approve_payroll"],
				"department": "HR",
				"preferences": {"language": "en", "timezone": "UTC"}
			}
		except Exception as e:
			logger.error(f"Failed to get user data: {e}")
			return {}
	
	async def _validate_user_permissions(
		self, 
		user_id: str, 
		intent: ConversationIntent, 
		tenant_id: str
	) -> bool:
		"""Validate user permissions for the requested action."""
		
		# Define required permissions for each intent
		permission_map = {
			ConversationIntent.START_PAYROLL_RUN: "start_payroll",
			ConversationIntent.APPROVE_PAYROLL: "approve_payroll",
			ConversationIntent.CANCEL_PAYROLL: "cancel_payroll",
			ConversationIntent.FINALIZE_PAYROLL: "finalize_payroll",
			ConversationIntent.UPDATE_EMPLOYEE_PAYROLL: "update_payroll",
		}
		
		required_permission = permission_map.get(intent)
		if not required_permission:
			return True  # No special permission required
		
		# Check user permissions
		user_permissions = await self._get_user_permissions(user_id, tenant_id)
		return required_permission in user_permissions
	
	async def _get_user_permissions(self, user_id: str, tenant_id: str) -> List[str]:
		"""Get user permissions."""
		try:
			# This would integrate with auth_rbac service
			return ["read_payroll", "write_payroll", "start_payroll", "approve_payroll"]
		except Exception as e:
			logger.error(f"Failed to get user permissions: {e}")
			return ["read_payroll"]  # Default minimal permissions
	
	async def _check_payroll_permission(self, user_id: str, permission: str, tenant_id: str) -> bool:
		"""Check specific payroll permission."""
		permissions = await self._get_user_permissions(user_id, tenant_id)
		return permission in permissions
	
	async def _get_employee_id_by_name(self, employee_name: str, tenant_id: str) -> Optional[str]:
		"""Get employee ID by name."""
		try:
			query = select(PREmployeePayroll).join(
				# This would join with employee table
			).where(
				and_(
					func.lower(PREmployeePayroll.employee_name).contains(employee_name.lower()),
					PREmployeePayroll.tenant_id == tenant_id
				)
			).limit(1)
			
			result = await self.db.execute(query)
			payroll = result.scalar_one_or_none()
			return payroll.employee_id if payroll else None
		except Exception as e:
			logger.error(f"Failed to get employee ID by name: {e}")
			return None
	
	async def _get_current_user_employee_id(self, user_id: str, tenant_id: str) -> Optional[str]:
		"""Get employee ID for current user."""
		try:
			# This would integrate with employee service
			employee_data = await self.employee_service.get_employee_by_user_id(user_id, tenant_id)
			return employee_data.get("employee_id") if employee_data else None
		except Exception as e:
			logger.error(f"Failed to get current user employee ID: {e}")
			return None
	
	async def _get_employee_payroll_data(
		self, 
		employee_id: str, 
		period: str, 
		tenant_id: str
	) -> Optional[Dict[str, Any]]:
		"""Get employee payroll data for specified period."""
		try:
			# Get the most recent payroll for the employee
			query = select(PREmployeePayroll).join(PRPayrollRun).join(PRPayrollPeriod).where(
				and_(
					PREmployeePayroll.employee_id == employee_id,
					PREmployeePayroll.tenant_id == tenant_id,
					PRPayrollRun.status.in_([PayrollStatus.COMPLETED, PayrollStatus.APPROVED])
				)
			).order_by(PRPayrollRun.completed_at.desc()).limit(1)
			
			result = await self.db.execute(query)
			payroll = result.scalar_one_or_none()
			
			if payroll:
				return {
					"employee_id": payroll.employee_id,
					"employee_name": payroll.employee_name,
					"gross_earnings": float(payroll.gross_earnings),
					"total_deductions": float(payroll.total_deductions),
					"total_taxes": float(payroll.total_taxes),
					"net_pay": float(payroll.net_pay),
					"regular_hours": float(payroll.regular_hours),
					"overtime_hours": float(payroll.overtime_hours),
					"period_name": "Current Period"  # Would get from joined period
				}
			
			return None
		except Exception as e:
			logger.error(f"Failed to get employee payroll data: {e}")
			return None
	
	async def _get_payroll_summary_data(self, period: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get payroll summary data for specified period."""
		try:
			# Get summary from materialized view or calculate
			query = text("""
				SELECT 
					period_name,
					total_employees,
					total_gross,
					total_deductions,
					total_taxes,
					total_net,
					avg_processing_score
				FROM mv_payroll_summary 
				WHERE tenant_id = :tenant_id
				ORDER BY period_name DESC 
				LIMIT 1
			""")
			
			result = await self.db.execute(query, {"tenant_id": tenant_id})
			row = result.fetchone()
			
			if row:
				return {
					"period_name": row.period_name,
					"employee_count": row.total_employees,
					"total_gross": float(row.total_gross),
					"total_taxes": float(row.total_taxes),
					"total_deductions": float(row.total_deductions),
					"total_net": float(row.total_net),
					"status": "Completed",
					"processing_score": float(row.avg_processing_score),
					"insights": [
						f"Average pay per employee: ${float(row.total_gross) / max(row.total_employees, 1):,.2f}",
						f"Tax rate: {float(row.total_taxes) / float(row.total_gross) * 100:.1f}%"
					]
				}
			
			return None
		except Exception as e:
			logger.error(f"Failed to get payroll summary data: {e}")
			return None
	
	async def _get_period_info(self, period: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get payroll period information."""
		try:
			query = select(PRPayrollPeriod).where(
				and_(
					PRPayrollPeriod.tenant_id == tenant_id,
					PRPayrollPeriod.is_active == True
				)
			).order_by(PRPayrollPeriod.start_date.desc()).limit(1)
			
			result = await self.db.execute(query)
			period_obj = result.scalar_one_or_none()
			
			if period_obj:
				return {
					"period_id": period_obj.period_id,
					"period_name": period_obj.period_name,
					"start_date": period_obj.start_date.isoformat(),
					"end_date": period_obj.end_date.isoformat(),
					"pay_date": period_obj.pay_date.isoformat()
				}
			
			return None
		except Exception as e:
			logger.error(f"Failed to get period info: {e}")
			return None
	
	async def _get_active_payroll_run(self, period_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
		"""Get active payroll run for period."""
		try:
			query = select(PRPayrollRun).where(
				and_(
					PRPayrollRun.period_id == period_id,
					PRPayrollRun.tenant_id == tenant_id,
					PRPayrollRun.status.in_([
						PayrollStatus.DRAFT, 
						PayrollStatus.PROCESSING, 
						PayrollStatus.AI_VALIDATION,
						PayrollStatus.COMPLIANCE_CHECK
					])
				)
			).limit(1)
			
			result = await self.db.execute(query)
			run = result.scalar_one_or_none()
			
			if run:
				return {
					"run_id": run.run_id,
					"status": run.status,
					"started_at": run.started_at.isoformat() if run.started_at else None
				}
			
			return None
		except Exception as e:
			logger.error(f"Failed to get active payroll run: {e}")
			return None
	
	async def _start_payroll_run(self, period_id: str, user_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Start a new payroll run."""
		try:
			# Create new payroll run
			new_run = PRPayrollRun(
				tenant_id=tenant_id,
				period_id=period_id,
				run_type="regular",
				status=PayrollStatus.PROCESSING,
				started_at=datetime.utcnow(),
				processed_by=user_id,
				run_number=1  # Would calculate next run number
			)
			
			self.db.add(new_run)
			await self.db.commit()
			await self.db.refresh(new_run)
			
			# Start background processing
			asyncio.create_task(self._process_payroll_run_async(new_run.run_id, tenant_id))
			
			return {
				"run_id": new_run.run_id,
				"status": new_run.status,
				"started_at": new_run.started_at.isoformat()
			}
		except Exception as e:
			logger.error(f"Failed to start payroll run: {e}")
			await self.db.rollback()
			raise
	
	async def _process_payroll_run_async(self, run_id: str, tenant_id: str) -> None:
		"""Process payroll run asynchronously."""
		try:
			# This would trigger the full payroll processing pipeline
			logger.info(f"Starting async processing for payroll run {run_id}")
			
			# Simulate processing time
			await asyncio.sleep(5)
			
			# Update run status
			query = select(PRPayrollRun).where(PRPayrollRun.run_id == run_id)
			result = await self.db.execute(query)
			run = result.scalar_one_or_none()
			
			if run:
				run.status = PayrollStatus.AI_VALIDATION
				run.progress_percentage = 50.0
				await self.db.commit()
			
			# Continue with AI validation and other steps...
			
		except Exception as e:
			logger.error(f"Async payroll processing failed: {e}")
	
	async def _estimate_payroll_completion_time(self, run_id: str) -> int:
		"""Estimate payroll completion time in minutes."""
		try:
			# Get employee count and complexity factors
			query = select(func.count(PREmployeePayroll.employee_id)).where(
				PREmployeePayroll.run_id == run_id
			)
			result = await self.db.execute(query)
			employee_count = result.scalar() or 0
			
			# Simple estimation: 0.1 minutes per employee + base 2 minutes
			estimated_minutes = max(2, int(employee_count * 0.1 + 2))
			return estimated_minutes
		except Exception as e:
			logger.error(f"Failed to estimate completion time: {e}")
			return 10  # Default 10 minutes
	
	async def _get_current_payroll_run_id(self, tenant_id: str) -> Optional[str]:
		"""Get current active payroll run ID."""
		try:
			query = select(PRPayrollRun).where(
				and_(
					PRPayrollRun.tenant_id == tenant_id,
					PRPayrollRun.status.in_([
						PayrollStatus.PROCESSING,
						PayrollStatus.AI_VALIDATION,
						PayrollStatus.COMPLIANCE_CHECK,
						PayrollStatus.APPROVED
					])
				)
			).order_by(PRPayrollRun.started_at.desc()).limit(1)
			
			result = await self.db.execute(query)
			run = result.scalar_one_or_none()
			return run.run_id if run else None
		except Exception as e:
			logger.error(f"Failed to get current payroll run ID: {e}")
			return None
	
	async def _generate_voice_response(self, text_response: str) -> str:
		"""Generate voice response from text."""
		try:
			voice_response = await self.voice_service.text_to_speech(
				text=text_response,
				language=self.config.voice_language,
				voice_style="professional"
			)
			return voice_response.audio_url
		except Exception as e:
			logger.error(f"Voice response generation failed: {e}")
			return ""
	
	async def _update_conversation_context(
		self,
		context: ConversationContext,
		user_input: str,
		response: ConversationResponse
	) -> None:
		"""Update conversation context with new interaction."""
		try:
			# Add to conversation history
			context.conversation_history.append({
				"timestamp": datetime.utcnow().isoformat(),
				"user_input": user_input,
				"response": response.message,
				"response_type": response.response_type.value
			})
			
			# Limit history length
			if len(context.conversation_history) > self.config.context_memory_turns:
				context.conversation_history = context.conversation_history[-self.config.context_memory_turns:]
			
			# Update last activity
			context.last_activity = datetime.utcnow()
			
			# Update context with any response data
			if response.context_updates:
				context.entities.update(response.context_updates)
			
		except Exception as e:
			logger.error(f"Failed to update conversation context: {e}")
	
	async def _log_conversation(
		self,
		user_id: str,
		user_input: str,
		response: ConversationResponse,
		context: ConversationContext
	) -> None:
		"""Log conversation for audit and analytics."""
		try:
			# This would log to audit system
			log_data = {
				"user_id": user_id,
				"tenant_id": context.tenant_id,
				"session_id": context.session_id,
				"timestamp": datetime.utcnow().isoformat(),
				"user_input": user_input if not self.config.sensitive_data_masking else "[MASKED]",
				"response_type": response.response_type.value,
				"confidence_score": response.confidence_score,
				"voice_interaction": bool(response.voice_response)
			}
			
			logger.info(f"Conversation logged for user {user_id}")
			
		except Exception as e:
			logger.error(f"Failed to log conversation: {e}")


# Example usage and testing
async def example_usage():
	"""Example of how to use the ConversationalPayrollAssistant."""
	
	# This would be set up with actual services
	# assistant = ConversationalPayrollAssistant(
	#     db_session=db_session,
	#     ai_service=ai_service,
	#     nlp_service=nlp_service,
	#     voice_service=voice_service,
	#     conversation_service=conversation_service,
	#     notification_service=notification_service,
	#     employee_service=employee_service,
	#     intelligence_engine=intelligence_engine
	# )
	
	# # Process a conversation
	# response = await assistant.process_conversation(
	#     user_input="Show me John Smith's pay stub for this month",
	#     user_id="user123",
	#     tenant_id="tenant456",
	#     session_id="session789"
	# )
	
	# print(f"Response: {response.message}")
	# print(f"Type: {response.response_type}")
	# print(f"Suggestions: {response.suggested_actions}")
	
	pass


if __name__ == "__main__":
	# Run example
	asyncio.run(example_usage())