"""
APG Financial Reporting - Revolutionary Conversational Interface

Voice and text-based conversational interface for natural language financial reporting
with multi-language support and context-aware interaction.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import speech_recognition as sr
import pyttsx3
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated

from .models import (
	CFRFConversationalInterface, CFRFReportTemplate,
	ConversationalIntentType, ReportIntelligenceLevel
)
from .nlp_engine import FinancialNLPEngine, ConversationalRequest, NLPResponse
from .ai_assistant import AIFinancialAssistant
from ...auth_rbac.models import db


class ConversationMode(str, Enum):
	"""Conversation interaction modes."""
	TEXT_ONLY = "text_only"
	VOICE_ONLY = "voice_only"
	MULTIMODAL = "multimodal"
	DICTATION = "dictation"


class LanguageCode(str, Enum):
	"""Supported language codes for conversation."""
	ENGLISH_US = "en-US"
	ENGLISH_UK = "en-UK"
	SPANISH = "es-ES"
	FRENCH = "fr-FR"
	GERMAN = "de-DE"
	CHINESE = "zh-CN"
	JAPANESE = "ja-JP"
	PORTUGUESE = "pt-BR"


class VoiceSettings(BaseModel):
	"""Voice interaction settings configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	enabled: bool = True
	language: LanguageCode = LanguageCode.ENGLISH_US
	voice_speed: int = Field(ge=50, le=200, default=150)
	voice_volume: float = Field(ge=0.0, le=1.0, default=0.8)
	voice_gender: str = Field(default="neutral")  # "male", "female", "neutral"
	wake_word: str = Field(default="APG Assistant")
	auto_listen: bool = False
	noise_threshold: float = Field(ge=0.0, le=1.0, default=0.3)


@dataclass
class ConversationContext:
	"""Comprehensive conversation context and state."""
	session_id: str
	user_id: str
	tenant_id: str
	conversation_mode: ConversationMode
	language: LanguageCode
	voice_settings: VoiceSettings
	current_topic: Optional[str] = None
	active_report: Optional[str] = None
	conversation_history: List[Dict[str, Any]] = field(default_factory=list)
	user_preferences: Dict[str, Any] = field(default_factory=dict)
	context_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
	"""Single conversation turn with user input and AI response."""
	turn_id: str
	timestamp: datetime
	user_input: str
	input_mode: str  # "text", "voice", "gesture"
	ai_response: str
	response_mode: str  # "text", "voice", "visual"
	confidence_score: float
	processing_time_ms: int
	artifacts_generated: Dict[str, Any]
	user_satisfaction: Optional[int] = None  # 1-5 rating


class ConversationalFinancialInterface:
	"""Revolutionary conversational interface for financial reporting."""
	
	def __init__(self, tenant_id: str, openai_api_key: str):
		self.tenant_id = tenant_id
		self.nlp_engine = FinancialNLPEngine(tenant_id, openai_api_key)
		self.ai_assistant = AIFinancialAssistant(tenant_id, openai_api_key)
		
		# Voice components
		self.speech_recognizer = sr.Recognizer()
		self.text_to_speech = pyttsx3.init()
		self.microphone = sr.Microphone()
		
		# Active conversations
		self.active_sessions: Dict[str, ConversationContext] = {}
		self.conversation_history: Dict[str, List[ConversationTurn]] = {}
		
		# Conversation handlers
		self.intent_handlers = self._register_intent_handlers()
		
	async def start_conversation_session(self, user_id: str, 
									   conversation_mode: ConversationMode = ConversationMode.TEXT_ONLY,
									   language: LanguageCode = LanguageCode.ENGLISH_US,
									   voice_settings: Optional[VoiceSettings] = None) -> str:
		"""Start a new conversational session."""
		
		session_id = uuid7str()
		
		# Create conversation context
		context = ConversationContext(
			session_id=session_id,
			user_id=user_id,
			tenant_id=self.tenant_id,
			conversation_mode=conversation_mode,
			language=language,
			voice_settings=voice_settings or VoiceSettings()
		)
		
		# Configure voice settings if voice mode enabled
		if conversation_mode in [ConversationMode.VOICE_ONLY, ConversationMode.MULTIMODAL]:
			await self._configure_voice_interface(context.voice_settings)
		
		# Store active session
		self.active_sessions[session_id] = context
		self.conversation_history[session_id] = []
		
		# Send welcome message
		welcome_message = await self._generate_welcome_message(context)
		await self._deliver_response(context, welcome_message, "text")
		
		return session_id
	
	async def process_user_input(self, session_id: str, user_input: str, 
								input_mode: str = "text") -> Dict[str, Any]:
		"""Process user input and generate appropriate response."""
		
		context = self.active_sessions.get(session_id)
		if not context:
			raise ValueError("Invalid session ID or session expired")
		
		start_time = datetime.now()
		
		try:
			# Process input based on mode
			if input_mode == "voice":
				user_input = await self._process_voice_input(user_input, context)
			
			# Generate AI response
			conversation_request = await self.nlp_engine.process_natural_language_query(
				user_input, context.user_id, session_id
			)
			
			ai_response = await self.nlp_engine.generate_ai_response(conversation_request)
			
			# Handle specific intents
			enhanced_response = await self._handle_intent(conversation_request, ai_response, context)
			
			# Determine response mode
			response_mode = self._determine_response_mode(context, enhanced_response)
			
			# Deliver response
			await self._deliver_response(context, enhanced_response.response_text, response_mode)
			
			# Record conversation turn
			processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
			turn = ConversationTurn(
				turn_id=uuid7str(),
				timestamp=start_time,
				user_input=user_input,
				input_mode=input_mode,
				ai_response=enhanced_response.response_text,
				response_mode=response_mode,
				confidence_score=enhanced_response.confidence_score,
				processing_time_ms=processing_time,
				artifacts_generated=enhanced_response.generated_artifacts
			)
			
			self.conversation_history[session_id].append(turn)
			context.conversation_history.append({
				'turn_id': turn.turn_id,
				'timestamp': turn.timestamp.isoformat(),
				'user_input': user_input,
				'ai_response': enhanced_response.response_text,
				'confidence': enhanced_response.confidence_score
			})
			
			return {
				'turn_id': turn.turn_id,
				'session_id': session_id,
				'ai_response': enhanced_response.response_text,
				'response_mode': response_mode,
				'artifacts': enhanced_response.generated_artifacts,
				'follow_up_suggestions': enhanced_response.suggested_follow_ups,
				'confidence_score': enhanced_response.confidence_score,
				'processing_time_ms': processing_time
			}
			
		except Exception as e:
			await self._handle_conversation_error(context, str(e))
			raise
	
	async def process_voice_command(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
		"""Process voice command from audio data."""
		
		context = self.active_sessions.get(session_id)
		if not context:
			raise ValueError("Invalid session ID")
		
		if context.conversation_mode not in [ConversationMode.VOICE_ONLY, ConversationMode.MULTIMODAL]:
			raise ValueError("Voice mode not enabled for this session")
		
		try:
			# Convert audio to text
			with sr.AudioFile(audio_data) as audio_source:
				audio = self.speech_recognizer.record(audio_source)
				user_input = self.speech_recognizer.recognize_google(
					audio, 
					language=context.language.value
				)
			
			# Process as regular input
			return await self.process_user_input(session_id, user_input, "voice")
			
		except sr.UnknownValueError:
			error_message = "I couldn't understand what you said. Could you please repeat that?"
			await self._deliver_response(context, error_message, "voice")
			return {
				'error': 'speech_recognition_failed',
				'message': error_message
			}
		except sr.RequestError as e:
			error_message = f"Speech recognition service error: {str(e)}"
			await self._deliver_response(context, error_message, "voice")
			return {
				'error': 'speech_service_error',
				'message': error_message
			}
	
	async def activate_voice_mode(self, session_id: str, wake_word_detected: bool = False) -> bool:
		"""Activate voice listening mode for continuous interaction."""
		
		context = self.active_sessions.get(session_id)
		if not context or not context.voice_settings.enabled:
			return False
		
		try:
			# Adjust for ambient noise
			with self.microphone as source:
				self.speech_recognizer.adjust_for_ambient_noise(source)
			
			if wake_word_detected or context.voice_settings.auto_listen:
				# Start listening for voice commands
				await self._start_voice_listening(context)
				return True
			
			return False
			
		except Exception as e:
			await self._handle_conversation_error(context, f"Voice activation failed: {str(e)}")
			return False
	
	async def end_conversation_session(self, session_id: str) -> Dict[str, Any]:
		"""End conversation session and provide summary."""
		
		context = self.active_sessions.get(session_id)
		if not context:
			return {'error': 'Session not found'}
		
		# Generate conversation summary
		conversation_turns = self.conversation_history.get(session_id, [])
		summary = await self._generate_conversation_summary(conversation_turns, context)
		
		# Update final conversation record
		await self._update_final_conversation_record(session_id, summary)
		
		# Cleanup session
		del self.active_sessions[session_id]
		if session_id in self.conversation_history:
			del self.conversation_history[session_id]
		
		return {
			'session_id': session_id,
			'duration_minutes': summary['duration_minutes'],
			'total_turns': summary['total_turns'],
			'topics_discussed': summary['topics_discussed'],
			'artifacts_created': summary['artifacts_created'],
			'user_satisfaction': summary.get('average_satisfaction'),
			'summary': summary['conversation_summary']
		}
	
	async def get_conversation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Get current conversation context and history."""
		
		context = self.active_sessions.get(session_id)
		if not context:
			return None
		
		return {
			'session_id': session_id,
			'user_id': context.user_id,
			'conversation_mode': context.conversation_mode.value,
			'language': context.language.value,
			'current_topic': context.current_topic,
			'active_report': context.active_report,
			'turn_count': len(context.conversation_history),
			'last_interaction': context.conversation_history[-1] if context.conversation_history else None,
			'user_preferences': context.user_preferences,
			'context_variables': context.context_variables
		}
	
	def _register_intent_handlers(self) -> Dict[ConversationalIntentType, Callable]:
		"""Register handlers for different conversation intents."""
		return {
			ConversationalIntentType.REPORT_CREATION: self._handle_report_creation_intent,
			ConversationalIntentType.DATA_ANALYSIS: self._handle_data_analysis_intent,
			ConversationalIntentType.TEMPLATE_MANAGEMENT: self._handle_template_management_intent,
			ConversationalIntentType.HELP_GUIDANCE: self._handle_help_guidance_intent,
			ConversationalIntentType.GENERAL_INQUIRY: self._handle_general_inquiry_intent
		}
	
	async def _handle_intent(self, request: ConversationalRequest, response: NLPResponse, 
							context: ConversationContext) -> NLPResponse:
		"""Handle specific conversation intent with specialized processing."""
		
		handler = self.intent_handlers.get(request.intent_type)
		if handler:
			return await handler(request, response, context)
		
		return response
	
	async def _handle_report_creation_intent(self, request: ConversationalRequest, 
											response: NLPResponse, context: ConversationContext) -> NLPResponse:
		"""Handle report creation intent with guided workflow."""
		
		# Extract report requirements from request
		entities = request.extracted_entities.get('entities', [])
		
		# Build guided report creation workflow
		workflow_steps = [
			"Confirm report type and format",
			"Select reporting period and entities",
			"Choose template or create custom layout",
			"Configure AI enhancements and automation",
			"Review and generate report"
		]
		
		# Enhance response with workflow guidance
		enhanced_artifacts = response.generated_artifacts.copy()
		enhanced_artifacts.update({
			'report_creation_workflow': {
				'steps': workflow_steps,
				'current_step': 1,
				'extracted_requirements': entities,
				'suggested_templates': await self._get_relevant_templates(entities),
				'customization_options': self._get_customization_options()
			}
		})
		
		# Update conversation context
		context.current_topic = "report_creation"
		context.context_variables['report_workflow_active'] = True
		
		return NLPResponse(
			response_text=f"{response.response_text}\n\nI'll guide you through creating your report. Let's start by confirming the report type and format you need.",
			response_type="guided_workflow",
			generated_artifacts=enhanced_artifacts,
			suggested_follow_ups=[
				"I need a monthly balance sheet",
				"Create an income statement with variance analysis",
				"Generate a cash flow report for Q3"
			],
			confidence_score=response.confidence_score
		)
	
	async def _handle_data_analysis_intent(self, request: ConversationalRequest, 
										  response: NLPResponse, context: ConversationContext) -> NLPResponse:
		"""Handle data analysis intent with intelligent insights."""
		
		# Get relevant insights from AI assistant
		insights = await self.ai_assistant.provide_intelligent_guidance(
			request.user_query, context.user_id, context.session_id
		)
		
		# Enhance response with analytical capabilities
		enhanced_artifacts = response.generated_artifacts.copy()
		enhanced_artifacts.update({
			'analysis_insights': insights.get('proactive_insights', []),
			'recommended_analysis': [
				"Variance analysis compared to budget",
				"Trend analysis over multiple periods",
				"Ratio analysis and benchmarking",
				"Predictive analytics and forecasting"
			],
			'visualization_options': [
				"Interactive charts and graphs",
				"Drill-down capability by account",
				"Comparative period analysis",
				"Exception-based reporting"
			]
		})
		
		context.current_topic = "data_analysis"
		
		return NLPResponse(
			response_text=f"{response.response_text}\n\nI can perform comprehensive analysis on your financial data. What specific analysis would you like me to conduct?",
			response_type="analytical_guidance",
			generated_artifacts=enhanced_artifacts,
			suggested_follow_ups=[
				"Show me variance analysis for this month",
				"Compare performance to last year",
				"Identify trends in revenue growth"
			],
			confidence_score=response.confidence_score
		)
	
	async def _handle_template_management_intent(self, request: ConversationalRequest, 
												response: NLPResponse, context: ConversationContext) -> NLPResponse:
		"""Handle template management intent."""
		
		templates = await self._get_available_templates()
		
		enhanced_artifacts = response.generated_artifacts.copy()
		enhanced_artifacts.update({
			'available_templates': templates,
			'template_actions': [
				"Create new template",
				"Modify existing template",
				"Copy and customize template",
				"Set template as default"
			],
			'ai_enhancements': [
				"Enable adaptive formatting",
				"Add predictive insights",
				"Configure natural language interface",
				"Set up automated generation"
			]
		})
		
		context.current_topic = "template_management"
		
		return NLPResponse(
			response_text=f"{response.response_text}\n\nI can help you manage your report templates. What would you like to do?",
			response_type="template_guidance",
			generated_artifacts=enhanced_artifacts,
			suggested_follow_ups=[
				"Create a new income statement template",
				"Modify the standard balance sheet format",
				"Enable AI features for existing templates"
			],
			confidence_score=response.confidence_score
		)
	
	async def _handle_help_guidance_intent(self, request: ConversationalRequest, 
										  response: NLPResponse, context: ConversationContext) -> NLPResponse:
		"""Handle help and guidance intent."""
		
		help_topics = [
			"Creating financial reports",
			"Using AI features",
			"Voice commands and controls",
			"Data analysis and insights",
			"Template customization",
			"Collaboration features"
		]
		
		enhanced_artifacts = response.generated_artifacts.copy()
		enhanced_artifacts.update({
			'help_topics': help_topics,
			'tutorials_available': [
				"Getting started with APG Financial Reporting",
				"Advanced AI features walkthrough",
				"Voice command reference",
				"Customization best practices"
			],
			'quick_actions': [
				"Show me how to create a report",
				"Explain AI capabilities",
				"Voice command help"
			]
		})
		
		context.current_topic = "help_guidance"
		
		return NLPResponse(
			response_text=f"{response.response_text}\n\nI'm here to help! What specific topic would you like guidance on?",
			response_type="help_guidance",
			generated_artifacts=enhanced_artifacts,
			suggested_follow_ups=[
				"How do I create my first report?",
				"What AI features are available?",
				"Show me voice command examples"
			],
			confidence_score=response.confidence_score
		)
	
	async def _handle_general_inquiry_intent(self, request: ConversationalRequest, 
											response: NLPResponse, context: ConversationContext) -> NLPResponse:
		"""Handle general inquiry intent."""
		
		return NLPResponse(
			response_text=f"{response.response_text}\n\nIs there anything specific about financial reporting I can help you with?",
			response_type="general_response",
			generated_artifacts=response.generated_artifacts,
			suggested_follow_ups=[
				"Create a financial report",
				"Analyze financial data",
				"Help with templates"
			],
			confidence_score=response.confidence_score
		)
	
	async def _configure_voice_interface(self, voice_settings: VoiceSettings):
		"""Configure voice interface with user preferences."""
		
		# Configure text-to-speech settings
		self.text_to_speech.setProperty('rate', voice_settings.voice_speed)
		self.text_to_speech.setProperty('volume', voice_settings.voice_volume)
		
		# Configure speech recognition sensitivity
		self.speech_recognizer.energy_threshold = voice_settings.noise_threshold * 4000
		self.speech_recognizer.dynamic_energy_threshold = True
		self.speech_recognizer.pause_threshold = 1.0
	
	async def _deliver_response(self, context: ConversationContext, response_text: str, mode: str):
		"""Deliver response in appropriate mode (text/voice/multimodal)."""
		
		if mode in ["voice", "multimodal"] and context.voice_settings.enabled:
			# Use text-to-speech for voice response
			self.text_to_speech.say(response_text)
			self.text_to_speech.runAndWait()
		
		# Log response delivery
		print(f"[Conversation] Response delivered in {mode} mode: {response_text[:100]}...")
	
	def _determine_response_mode(self, context: ConversationContext, response: NLPResponse) -> str:
		"""Determine appropriate response delivery mode."""
		
		if context.conversation_mode == ConversationMode.VOICE_ONLY:
			return "voice"
		elif context.conversation_mode == ConversationMode.MULTIMODAL:
			return "multimodal"
		else:
			return "text"
	
	# Utility and helper methods
	
	async def _generate_welcome_message(self, context: ConversationContext) -> str:
		"""Generate personalized welcome message."""
		
		mode_text = ""
		if context.conversation_mode == ConversationMode.VOICE_ONLY:
			mode_text = " I'm ready to assist you with voice commands."
		elif context.conversation_mode == ConversationMode.MULTIMODAL:
			mode_text = " You can interact with me using voice or text."
		
		return f"Hello! I'm your APG Financial Reporting AI Assistant.{mode_text} How can I help you with your financial reporting today?"
	
	async def _process_voice_input(self, audio_input: str, context: ConversationContext) -> str:
		"""Process and clean voice input."""
		# In production, this would handle audio processing
		return audio_input.strip()
	
	async def _start_voice_listening(self, context: ConversationContext):
		"""Start continuous voice listening mode."""
		# This would implement continuous voice monitoring
		pass
	
	async def _handle_conversation_error(self, context: ConversationContext, error: str):
		"""Handle conversation errors gracefully."""
		error_message = "I encountered an issue processing your request. Could you please try rephrasing your question?"
		await self._deliver_response(context, error_message, "text")
	
	async def _generate_conversation_summary(self, turns: List[ConversationTurn], 
											context: ConversationContext) -> Dict[str, Any]:
		"""Generate comprehensive conversation summary."""
		
		if not turns:
			return {
				'duration_minutes': 0,
				'total_turns': 0,
				'topics_discussed': [],
				'artifacts_created': [],
				'conversation_summary': 'No conversation activity'
			}
		
		start_time = turns[0].timestamp
		end_time = turns[-1].timestamp
		duration = (end_time - start_time).total_seconds() / 60
		
		# Extract topics and artifacts
		topics = list(set([context.current_topic] + [
			turn.artifacts_generated.get('topic') for turn in turns 
			if turn.artifacts_generated.get('topic')
		]))
		
		artifacts = []
		for turn in turns:
			if turn.artifacts_generated:
				artifacts.extend(turn.artifacts_generated.keys())
		
		return {
			'duration_minutes': round(duration, 2),
			'total_turns': len(turns),
			'topics_discussed': [t for t in topics if t],
			'artifacts_created': list(set(artifacts)),
			'average_confidence': sum(turn.confidence_score for turn in turns) / len(turns),
			'conversation_summary': f"Productive conversation covering {len(set(topics))} topics with {len(turns)} exchanges"
		}
	
	async def _update_final_conversation_record(self, session_id: str, summary: Dict[str, Any]):
		"""Update final conversation record in database."""
		# Update conversation records with final summary
		pass
	
	async def _get_relevant_templates(self, entities: List[Dict]) -> List[Dict]:
		"""Get relevant templates based on extracted entities."""
		return []  # Simplified for demonstration
	
	def _get_customization_options(self) -> List[str]:
		"""Get available customization options."""
		return [
			"AI-powered adaptive formatting",
			"Natural language narrative generation",
			"Predictive insights and alerts",
			"Real-time collaboration features",
			"Voice-activated controls"
		]
	
	async def _get_available_templates(self) -> List[Dict]:
		"""Get available report templates."""
		templates = db.session.query(CFRFReportTemplate).filter(
			CFRFReportTemplate.tenant_id == self.tenant_id,
			CFRFReportTemplate.is_active == True
		).limit(10).all()
		
		return [
			{
				'template_id': t.template_id,
				'template_name': t.template_name,
				'statement_type': t.statement_type,
				'ai_intelligence_level': t.ai_intelligence_level
			}
			for t in templates
		]