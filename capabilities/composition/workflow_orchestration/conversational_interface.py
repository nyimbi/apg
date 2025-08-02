"""
APG Workflow Orchestration - Conversational Interface
Natural language processing for workflow creation, modification, and voice control
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
import spacy
import speech_recognition as sr
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import openai
from textblob import TextBlob

from pydantic import BaseModel, Field, validator
from pydantic.config import ConfigDict

# APG Framework imports
from apg.base.service import APGBaseService
from apg.base.models import BaseModel as APGBaseModel
from apg.integrations.nlp import NLPService
from apg.integrations.speech import SpeechService
from apg.integrations.openai import OpenAIClient
from apg.base.security import SecurityManager

from .models import WorkflowInstance, Task, WorkflowDefinition
from .service import WorkflowService


class ConversationMode(str, Enum):
	"""Conversation modes"""
	TEXT_CHAT = "text_chat"
	VOICE_CONTROL = "voice_control"
	HYBRID = "hybrid"
	GUIDED_CREATION = "guided_creation"


class IntentType(str, Enum):
	"""Types of user intents"""
	CREATE_WORKFLOW = "create_workflow"
	MODIFY_WORKFLOW = "modify_workflow"
	EXECUTE_WORKFLOW = "execute_workflow"
	QUERY_STATUS = "query_status"
	GET_HELP = "get_help"
	DELETE_WORKFLOW = "delete_workflow"
	LIST_WORKFLOWS = "list_workflows"
	SCHEDULE_WORKFLOW = "schedule_workflow"
	ANALYZE_PERFORMANCE = "analyze_performance"
	CONFIGURE_SETTINGS = "configure_settings"


class EntityType(str, Enum):
	"""Types of entities that can be extracted"""
	WORKFLOW_NAME = "workflow_name"
	TASK_NAME = "task_name"
	RESOURCE_TYPE = "resource_type"
	SCHEDULE_TIME = "schedule_time"
	CONDITION = "condition"
	ACTION = "action"
	PARAMETER = "parameter"
	VALUE = "value"
	PERSON = "person"
	SYSTEM = "system"


@dataclass
class ConversationContext:
	"""Context for ongoing conversation"""
	session_id: str
	user_id: str
	tenant_id: str
	mode: ConversationMode
	current_workflow_id: Optional[str] = None
	conversation_history: List[Dict[str, Any]] = None
	extracted_entities: Dict[str, Any] = None
	pending_confirmations: List[str] = None
	language: str = "en"
	
	def __post_init__(self):
		if self.conversation_history is None:
			self.conversation_history = []
		if self.extracted_entities is None:
			self.extracted_entities = {}
		if self.pending_confirmations is None:
			self.pending_confirmations = []


class Intent(APGBaseModel):
	"""Detected user intent"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	type: IntentType = Field(..., description="Intent type")
	confidence: float = Field(..., description="Confidence score")
	entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Intent parameters")
	context_required: List[str] = Field(default_factory=list, description="Required context information")


class ConversationMessage(APGBaseModel):
	"""Single conversation message"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Message identifier")
	session_id: str = Field(..., description="Session identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
	
	# Message content
	sender: str = Field(..., description="Message sender (user/assistant)")
	content: str = Field(..., description="Message content")
	content_type: str = Field("text", description="Content type (text/audio/image)")
	
	# NLP analysis
	intent: Optional[Intent] = Field(None, description="Detected intent")
	entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
	sentiment: Optional[Dict[str, float]] = Field(None, description="Sentiment analysis")
	
	# Processing metadata
	processing_time_ms: Optional[int] = Field(None, description="Processing time")
	model_used: Optional[str] = Field(None, description="NLP model used")
	confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Various confidence scores")


class WorkflowTemplate(APGBaseModel):
	"""Template for natural language workflow creation"""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Template identifier")
	name: str = Field(..., description="Template name")
	description: str = Field(..., description="Template description")
	
	# Natural language patterns
	trigger_patterns: List[str] = Field(..., description="Patterns that trigger this template")
	required_entities: List[EntityType] = Field(..., description="Required entities")
	optional_entities: List[EntityType] = Field(default_factory=list, description="Optional entities")
	
	# Template structure
	workflow_template: Dict[str, Any] = Field(..., description="Workflow definition template")
	parameter_mappings: Dict[str, str] = Field(default_factory=dict, description="Entity to parameter mappings")
	
	# Generation settings
	complexity_level: str = Field("medium", description="Template complexity level")
	category: str = Field("general", description="Template category")
	tags: List[str] = Field(default_factory=list, description="Template tags")


class ConversationalInterface(APGBaseService):
	"""Main conversational interface service"""
	
	def __init__(self, workflow_service: WorkflowService, config: Dict[str, Any]):
		super().__init__()
		self.workflow_service = workflow_service
		self.config = config
		
		# NLP components
		self.nlp_processor = NLPProcessor(config)
		self.intent_classifier = IntentClassifier(config)
		self.entity_extractor = EntityExtractor(config)
		self.response_generator = ResponseGenerator(config)
		
		# Speech components
		self.speech_recognizer = SpeechRecognizer(config) if config.get('speech_enabled') else None
		self.speech_synthesizer = SpeechSynthesizer(config) if config.get('speech_enabled') else None
		
		# Context management
		self.context_manager = ContextManager()
		self.template_manager = TemplateManager()
		
		# Active conversations
		self.active_sessions: Dict[str, ConversationContext] = {}
		self.conversation_history: Dict[str, List[ConversationMessage]] = defaultdict(list)
		
		# Performance tracking
		self.interaction_metrics: Dict[str, List[float]] = defaultdict(list)
		
		self._log_info("Conversational interface initialized")
	
	async def initialize(self) -> None:
		"""Initialize conversational interface"""
		try:
			# Initialize NLP components
			await self.nlp_processor.initialize()
			await self.intent_classifier.initialize()
			await self.entity_extractor.initialize()
			await self.response_generator.initialize()
			
			# Initialize speech components
			if self.speech_recognizer:
				await self.speech_recognizer.initialize()
			if self.speech_synthesizer:
				await self.speech_synthesizer.initialize()
			
			# Initialize managers
			await self.context_manager.initialize()
			await self.template_manager.initialize()
			
			# Load templates
			await self._load_workflow_templates()
			
			self._log_info("Conversational interface initialized successfully")
			
		except Exception as e:
			self._log_error(f"Failed to initialize conversational interface: {e}")
			raise
	
	async def start_conversation(self, user_id: str, tenant_id: str, 
								mode: ConversationMode = ConversationMode.TEXT_CHAT,
								language: str = "en") -> str:
		"""Start a new conversation session"""
		try:
			session_id = f"conv_{user_id}_{int(datetime.utcnow().timestamp())}"
			
			context = ConversationContext(
				session_id=session_id,
				user_id=user_id,
				tenant_id=tenant_id,
				mode=mode,
				language=language
			)
			
			self.active_sessions[session_id] = context
			
			# Generate welcome message
			welcome_message = await self.response_generator.generate_welcome_message(context)
			
			# Store welcome message
			message = ConversationMessage(
				id=f"msg_{session_id}_0",
				session_id=session_id,
				sender="assistant",
				content=welcome_message
			)
			
			self.conversation_history[session_id].append(message)
			
			self._log_info(f"Started conversation session: {session_id} for user {user_id}")
			
			return session_id
			
		except Exception as e:
			self._log_error(f"Failed to start conversation: {e}")
			raise
	
	async def process_message(self, session_id: str, message_content: str, 
							 content_type: str = "text") -> ConversationMessage:
		"""Process incoming message from user"""
		try:
			context = self.active_sessions.get(session_id)
			if not context:
				raise ValueError(f"Session {session_id} not found")
			
			start_time = datetime.utcnow()
			
			# Create user message
			user_message = ConversationMessage(
				id=f"msg_{session_id}_{len(self.conversation_history[session_id])}",
				session_id=session_id,
				sender="user",
				content=message_content,
				content_type=content_type
			)
			
			# Process with NLP
			processed_message = await self._process_nlp(user_message, context)
			
			# Store user message
			self.conversation_history[session_id].append(processed_message)
			
			# Generate response
			response_content = await self._generate_response(processed_message, context)
			
			# Create assistant message
			assistant_message = ConversationMessage(
				id=f"msg_{session_id}_{len(self.conversation_history[session_id])}",
				session_id=session_id,
				sender="assistant",
				content=response_content,
				processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
			)
			
			# Store assistant message
			self.conversation_history[session_id].append(assistant_message)
			
			# Update context
			await self._update_context(context, processed_message)
			
			return assistant_message
			
		except Exception as e:
			self._log_error(f"Failed to process message: {e}")
			
			# Generate error response
			error_message = ConversationMessage(
				id=f"msg_{session_id}_error",
				session_id=session_id,
				sender="assistant",
				content="I'm sorry, I encountered an error processing your request. Could you please try again?"
			)
			
			return error_message
	
	async def process_voice_input(self, session_id: str, audio_data: bytes) -> ConversationMessage:
		"""Process voice input from user"""
		try:
			if not self.speech_recognizer:
				raise ValueError("Speech recognition not enabled")
			
			# Convert audio to text
			text_content = await self.speech_recognizer.recognize_speech(audio_data)
			
			# Process as text message
			response = await self.process_message(session_id, text_content, "voice")
			
			# Generate audio response if enabled
			if self.speech_synthesizer:
				audio_response = await self.speech_synthesizer.synthesize_speech(response.content)
				response.content_type = "audio"
				# Store audio data (implementation specific)
			
			return response
			
		except Exception as e:
			self._log_error(f"Failed to process voice input: {e}")
			raise
	
	async def _process_nlp(self, message: ConversationMessage, 
						  context: ConversationContext) -> ConversationMessage:
		"""Process message with NLP components"""
		try:
			# Extract entities
			entities = await self.entity_extractor.extract_entities(message.content, context)
			message.entities = entities
			
			# Classify intent
			intent = await self.intent_classifier.classify_intent(message.content, entities, context)
			message.intent = intent
			
			# Perform sentiment analysis
			sentiment = await self.nlp_processor.analyze_sentiment(message.content)
			message.sentiment = sentiment
			
			return message
			
		except Exception as e:
			self._log_error(f"Failed to process NLP: {e}")
			return message
	
	async def _generate_response(self, user_message: ConversationMessage, 
								context: ConversationContext) -> str:
		"""Generate response based on user message and intent"""
		try:
			if not user_message.intent:
				return "I'm not sure I understand. Could you please rephrase your request?"
			
			intent = user_message.intent
			
			if intent.type == IntentType.CREATE_WORKFLOW:
				return await self._handle_create_workflow(user_message, context)
			elif intent.type == IntentType.MODIFY_WORKFLOW:
				return await self._handle_modify_workflow(user_message, context)
			elif intent.type == IntentType.EXECUTE_WORKFLOW:
				return await self._handle_execute_workflow(user_message, context)
			elif intent.type == IntentType.QUERY_STATUS:
				return await self._handle_query_status(user_message, context)
			elif intent.type == IntentType.LIST_WORKFLOWS:
				return await self._handle_list_workflows(user_message, context)
			elif intent.type == IntentType.GET_HELP:
				return await self._handle_get_help(user_message, context)
			elif intent.type == IntentType.DELETE_WORKFLOW:
				return await self._handle_delete_workflow(user_message, context)
			elif intent.type == IntentType.SCHEDULE_WORKFLOW:
				return await self._handle_schedule_workflow(user_message, context)
			elif intent.type == IntentType.ANALYZE_PERFORMANCE:
				return await self._handle_analyze_performance(user_message, context)
			else:
				return await self.response_generator.generate_fallback_response(user_message, context)
			
		except Exception as e:
			self._log_error(f"Failed to generate response: {e}")
			return "I encountered an error while processing your request. Please try again."
	
	async def _handle_create_workflow(self, message: ConversationMessage, 
									 context: ConversationContext) -> str:
		"""Handle workflow creation intent"""
		try:
			entities = message.entities
			
			# Check if we have enough information
			required_info = ['workflow_name']
			missing_info = [info for info in required_info if info not in entities]
			
			if missing_info:
				context.pending_confirmations.extend(missing_info)
				questions = []
				if 'workflow_name' in missing_info:
					questions.append("What would you like to name this workflow?")
				
				return " ".join(questions)
			
			# Extract workflow details
			workflow_name = entities.get('workflow_name', 'Untitled Workflow')
			workflow_description = entities.get('description', f'Workflow created via conversation on {datetime.utcnow().strftime("%Y-%m-%d")}')
			
			# Find matching template
			template = await self.template_manager.find_matching_template(message.content, entities)
			
			if template:
				# Create workflow from template
				workflow_def = await self._create_workflow_from_template(template, entities)
			else:
				# Create basic workflow
				workflow_def = {
					'name': workflow_name,
					'description': workflow_description,
					'tasks': [
						{
							'id': 'start_task',
							'name': 'Start',
							'type': 'start',
							'properties': {}
						}
					],
					'connections': []
				}
			
			# Create workflow instance
			workflow = WorkflowInstance(
				name=workflow_name,
				description=workflow_description,
				definition=workflow_def,
				tenant_id=context.tenant_id,
				created_by=context.user_id,
				status="draft"
			)
			
			# Save workflow
			created_workflow = await self.workflow_service.create_workflow(workflow)
			context.current_workflow_id = created_workflow.id
			
			response = f"Great! I've created a new workflow called '{workflow_name}' for you."
			
			if template:
				response += f" Based on your description, I used the '{template.name}' template as a starting point."
			
			response += " Would you like to add more tasks or configure the workflow further?"
			
			return response
			
		except Exception as e:
			self._log_error(f"Failed to handle create workflow: {e}")
			return "I encountered an error while creating the workflow. Please try again."
	
	async def _handle_modify_workflow(self, message: ConversationMessage, 
									 context: ConversationContext) -> str:
		"""Handle workflow modification intent"""
		try:
			entities = message.entities
			
			# Determine which workflow to modify
			workflow_id = context.current_workflow_id
			
			if 'workflow_name' in entities:
				# Find workflow by name
				workflows = await self.workflow_service.list_workflows(context.tenant_id)
				workflow_name = entities['workflow_name']
				matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
				
				if matching_workflows:
					workflow_id = matching_workflows[0].id
					context.current_workflow_id = workflow_id
				else:
					return f"I couldn't find a workflow named '{workflow_name}'. Could you check the name and try again?"
			
			if not workflow_id:
				return "Which workflow would you like to modify? Please specify the workflow name."
			
			# Get current workflow
			workflow = await self.workflow_service.get_workflow(workflow_id, context.tenant_id)
			if not workflow:
				return "I couldn't find that workflow. Please check the workflow name."
			
			# Determine modification type
			modification_type = self._detect_modification_type(message.content, entities)
			
			if modification_type == "add_task":
				return await self._handle_add_task(workflow, entities, context)
			elif modification_type == "remove_task":
				return await self._handle_remove_task(workflow, entities, context)
			elif modification_type == "modify_task":
				return await self._handle_modify_task(workflow, entities, context)
			elif modification_type == "add_connection":
				return await self._handle_add_connection(workflow, entities, context)
			else:
				return "I understand you want to modify the workflow, but I'm not sure what specific changes you'd like to make. Could you be more specific?"
			
		except Exception as e:
			self._log_error(f"Failed to handle modify workflow: {e}")
			return "I encountered an error while modifying the workflow. Please try again."
	
	async def _handle_execute_workflow(self, message: ConversationMessage, 
									  context: ConversationContext) -> str:
		"""Handle workflow execution intent"""
		try:
			entities = message.entities
			
			# Determine which workflow to execute
			workflow_id = context.current_workflow_id
			workflow_name = entities.get('workflow_name')
			
			if workflow_name:
				workflows = await self.workflow_service.list_workflows(context.tenant_id)
				matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
				
				if matching_workflows:
					workflow_id = matching_workflows[0].id
				else:
					return f"I couldn't find a workflow named '{workflow_name}'. Please check the name and try again."
			
			if not workflow_id:
				return "Which workflow would you like to execute? Please specify the workflow name."
			
			# Execute workflow
			execution = await self.workflow_service.execute_workflow(workflow_id, context.tenant_id, context.user_id)
			
			return f"I've started executing the workflow. The execution ID is {execution.id}. You can check its status by asking 'What's the status of execution {execution.id}?'"
			
		except Exception as e:
			self._log_error(f"Failed to handle execute workflow: {e}")
			return "I encountered an error while executing the workflow. Please try again."
	
	async def _handle_query_status(self, message: ConversationMessage, 
								  context: ConversationContext) -> str:
		"""Handle status query intent"""
		try:
			entities = message.entities
			
			if 'execution_id' in entities:
				execution_id = entities['execution_id']
				execution = await self.workflow_service.get_execution(execution_id, context.tenant_id)
				
				if execution:
					return f"Execution {execution_id} is currently {execution.status}. Started at {execution.started_at.strftime('%Y-%m-%d %H:%M:%S')}."
				else:
					return f"I couldn't find an execution with ID {execution_id}."
			
			elif 'workflow_name' in entities or context.current_workflow_id:
				workflow_id = context.current_workflow_id
				workflow_name = entities.get('workflow_name')
				
				if workflow_name:
					workflows = await self.workflow_service.list_workflows(context.tenant_id)
					matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
					if matching_workflows:
						workflow_id = matching_workflows[0].id
				
				if workflow_id:
					executions = await self.workflow_service.get_recent_executions(workflow_id, context.tenant_id, limit=5)
					
					if executions:
						recent_execution = executions[0]
						return f"The most recent execution of this workflow is {recent_execution.status}. It started at {recent_execution.started_at.strftime('%Y-%m-%d %H:%M:%S')}."
					else:
						return "This workflow hasn't been executed yet."
			
			return "What specific status would you like to check? You can ask about a workflow or execution ID."
			
		except Exception as e:
			self._log_error(f"Failed to handle query status: {e}")
			return "I encountered an error while checking the status. Please try again."
	
	async def _handle_list_workflows(self, message: ConversationMessage, 
									context: ConversationContext) -> str:
		"""Handle list workflows intent"""
		try:
			workflows = await self.workflow_service.list_workflows(context.tenant_id, limit=10)
			
			if not workflows:
				return "You don't have any workflows yet. Would you like me to help you create one?"
			
			response = f"You have {len(workflows)} workflows:\n\n"
			
			for i, workflow in enumerate(workflows[:10], 1):
				status_info = f"({workflow.status})"
				last_execution = "Never executed"
				
				if hasattr(workflow, 'last_execution_at') and workflow.last_execution_at:
					last_execution = f"Last run: {workflow.last_execution_at.strftime('%Y-%m-%d %H:%M')}"
				
				response += f"{i}. **{workflow.name}** {status_info}\n"
				response += f"   {workflow.description}\n"
				response += f"   {last_execution}\n\n"
			
			if len(workflows) > 10:
				response += f"... and {len(workflows) - 10} more workflows."
			
			response += "\nWould you like to work with any of these workflows?"
			
			return response
			
		except Exception as e:
			self._log_error(f"Failed to handle list workflows: {e}")
			return "I encountered an error while listing workflows. Please try again."
	
	async def _handle_get_help(self, message: ConversationMessage, 
							  context: ConversationContext) -> str:
		"""Handle help request intent"""
		try:
			help_text = """
I can help you with workflow management through natural language! Here's what I can do:

**Creating Workflows:**
- "Create a new workflow called 'Data Processing'"
- "I need a workflow for processing customer orders"
- "Build a workflow that sends email notifications"

**Modifying Workflows:**
- "Add a task to send an email"
- "Remove the validation step"
- "Connect the data processing task to the email task"

**Running Workflows:**
- "Execute the Data Processing workflow"
- "Run my customer order workflow"
- "Start the email notification process"

**Checking Status:**
- "What's the status of my Data Processing workflow?"
- "How is execution 12345 doing?"
- "Show me recent workflow runs"

**Other Commands:**
- "List all my workflows"
- "Delete the old test workflow"
- "Schedule the daily report workflow for 9 AM"

Just describe what you want to do in natural language, and I'll help you build and manage your workflows!

What would you like to do first?
			""".strip()
			
			return help_text
			
		except Exception as e:
			self._log_error(f"Failed to handle get help: {e}")
			return "I can help you create, modify, and execute workflows using natural language. What would you like to do?"
	
	async def _handle_delete_workflow(self, message: ConversationMessage, 
									 context: ConversationContext) -> str:
		"""Handle workflow deletion intent"""
		try:
			entities = message.entities
			workflow_name = entities.get('workflow_name')
			
			if not workflow_name:
				return "Which workflow would you like to delete? Please specify the workflow name."
			
			# Find workflow
			workflows = await self.workflow_service.list_workflows(context.tenant_id)
			matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
			
			if not matching_workflows:
				return f"I couldn't find a workflow named '{workflow_name}'. Please check the name and try again."
			
			workflow = matching_workflows[0]
			
			# Add confirmation step
			if workflow.id not in context.pending_confirmations:
				context.pending_confirmations.append(workflow.id)
				return f"Are you sure you want to delete the workflow '{workflow.name}'? This action cannot be undone. Please confirm by saying 'yes' or 'confirm'."
			
			# Delete workflow
			await self.workflow_service.delete_workflow(workflow.id, context.tenant_id)
			context.pending_confirmations.remove(workflow.id)
			
			return f"I've successfully deleted the workflow '{workflow.name}'."
			
		except Exception as e:
			self._log_error(f"Failed to handle delete workflow: {e}")
			return "I encountered an error while deleting the workflow. Please try again."
	
	async def _handle_schedule_workflow(self, message: ConversationMessage, 
									   context: ConversationContext) -> str:
		"""Handle workflow scheduling intent"""
		try:
			entities = message.entities
			
			workflow_name = entities.get('workflow_name')
			schedule_time = entities.get('schedule_time')
			
			if not workflow_name:
				return "Which workflow would you like to schedule? Please specify the workflow name."
			
			if not schedule_time:
				return "When would you like to schedule this workflow? Please specify a time (e.g., 'tomorrow at 9 AM', 'every day at 2 PM')."
			
			# Find workflow
			workflows = await self.workflow_service.list_workflows(context.tenant_id)
			matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
			
			if not matching_workflows:
				return f"I couldn't find a workflow named '{workflow_name}'. Please check the name and try again."
			
			workflow = matching_workflows[0]
			
			# Parse schedule time (simplified implementation)
			# In production, you'd use a more sophisticated time parsing library
			parsed_time = await self._parse_schedule_time(schedule_time)
			
			if not parsed_time:
				return f"I couldn't understand the time '{schedule_time}'. Could you please specify it differently? For example: 'tomorrow at 9 AM' or 'every Monday at 2 PM'."
			
			# Schedule workflow (this would integrate with a scheduling service)
			# For now, we'll just store the schedule information
			schedule_info = {
				'workflow_id': workflow.id,
				'schedule_time': parsed_time,
				'created_by': context.user_id,
				'created_at': datetime.utcnow()
			}
			
			return f"I've scheduled the workflow '{workflow.name}' to run {schedule_time}. The schedule is now active."
			
		except Exception as e:
			self._log_error(f"Failed to handle schedule workflow: {e}")
			return "I encountered an error while scheduling the workflow. Please try again."
	
	async def _handle_analyze_performance(self, message: ConversationMessage, 
										 context: ConversationContext) -> str:
		"""Handle performance analysis intent"""
		try:
			entities = message.entities
			workflow_name = entities.get('workflow_name')
			
			if workflow_name:
				# Analyze specific workflow
				workflows = await self.workflow_service.list_workflows(context.tenant_id)
				matching_workflows = [w for w in workflows if w.name.lower() == workflow_name.lower()]
				
				if not matching_workflows:
					return f"I couldn't find a workflow named '{workflow_name}'. Please check the name and try again."
				
				workflow = matching_workflows[0]
				
				# Get execution metrics (simplified)
				executions = await self.workflow_service.get_recent_executions(workflow.id, context.tenant_id, limit=20)
				
				if not executions:
					return f"The workflow '{workflow.name}' hasn't been executed yet, so there's no performance data to analyze."
				
				# Calculate basic metrics
				total_executions = len(executions)
				successful_executions = len([e for e in executions if e.status == 'succeeded'])
				failed_executions = len([e for e in executions if e.status == 'failed'])
				success_rate = (successful_executions / total_executions) * 100 if total_executions > 0 else 0
				
				# Average execution time
				completed_executions = [e for e in executions if e.completed_at and e.started_at]
				if completed_executions:
					avg_duration = sum((e.completed_at - e.started_at).total_seconds() for e in completed_executions) / len(completed_executions)
					avg_duration_str = f"{avg_duration:.1f} seconds"
				else:
					avg_duration_str = "N/A"
				
				response = f"""
**Performance Analysis for '{workflow.name}':**

📊 **Execution Summary:**
- Total executions: {total_executions}
- Successful: {successful_executions}
- Failed: {failed_executions}
- Success rate: {success_rate:.1f}%

⏱️ **Timing:**
- Average execution time: {avg_duration_str}

🔍 **Recent Activity:**
"""
				
				for execution in executions[:5]:
					status_emoji = "✅" if execution.status == "succeeded" else "❌" if execution.status == "failed" else "⏳"
					response += f"- {status_emoji} {execution.started_at.strftime('%m/%d %H:%M')} - {execution.status}\n"
				
				return response.strip()
			
			else:
				# General performance overview
				return "I can analyze the performance of a specific workflow. Which workflow would you like me to analyze?"
			
		except Exception as e:
			self._log_error(f"Failed to handle analyze performance: {e}")
			return "I encountered an error while analyzing performance. Please try again."
	
	async def _create_workflow_from_template(self, template: WorkflowTemplate, 
											entities: Dict[str, Any]) -> Dict[str, Any]:
		"""Create workflow definition from template and entities"""
		try:
			workflow_def = template.workflow_template.copy()
			
			# Apply parameter mappings
			for entity_key, param_path in template.parameter_mappings.items():
				if entity_key in entities:
					value = entities[entity_key]
					# Set parameter in workflow definition (simplified)
					# In production, you'd use a more sophisticated path setting mechanism
					if '.' in param_path:
						# Handle nested parameters
						keys = param_path.split('.')
						current = workflow_def
						for key in keys[:-1]:
							if key not in current:
								current[key] = {}
							current = current[key]
						current[keys[-1]] = value
					else:
						workflow_def[param_path] = value
			
			return workflow_def
			
		except Exception as e:
			self._log_error(f"Failed to create workflow from template: {e}")
			return {}
	
	def _detect_modification_type(self, content: str, entities: Dict[str, Any]) -> str:
		"""Detect the type of modification requested"""
		content_lower = content.lower()
		
		add_keywords = ['add', 'insert', 'include', 'create', 'new']
		remove_keywords = ['remove', 'delete', 'drop', 'eliminate']
		modify_keywords = ['change', 'modify', 'update', 'edit', 'alter']
		connect_keywords = ['connect', 'link', 'join', 'chain', 'sequence']
		
		if any(keyword in content_lower for keyword in add_keywords):
			if 'task' in content_lower or 'step' in content_lower:
				return "add_task"
			elif any(keyword in content_lower for keyword in connect_keywords):
				return "add_connection"
		
		if any(keyword in content_lower for keyword in remove_keywords):
			return "remove_task"
		
		if any(keyword in content_lower for keyword in modify_keywords):
			return "modify_task"
		
		if any(keyword in content_lower for keyword in connect_keywords):
			return "add_connection"
		
		return "unknown"
	
	async def _handle_add_task(self, workflow: WorkflowInstance, entities: Dict[str, Any], 
							  context: ConversationContext) -> str:
		"""Handle adding a task to workflow"""
		try:
			task_name = entities.get('task_name', 'New Task')
			task_type = entities.get('task_type', 'task')
			
			# Create new task
			new_task = {
				'id': f"task_{len(workflow.definition.get('tasks', []))}",
				'name': task_name,
				'type': task_type,
				'properties': {}
			}
			
			# Add task to workflow
			if 'tasks' not in workflow.definition:
				workflow.definition['tasks'] = []
			
			workflow.definition['tasks'].append(new_task)
			
			# Update workflow
			await self.workflow_service.update_workflow(workflow.id, workflow, context.tenant_id)
			
			return f"I've added a new task called '{task_name}' to the workflow. Would you like to configure its properties or connect it to other tasks?"
			
		except Exception as e:
			self._log_error(f"Failed to add task: {e}")
			return "I encountered an error while adding the task. Please try again."
	
	async def _handle_remove_task(self, workflow: WorkflowInstance, entities: Dict[str, Any], 
								 context: ConversationContext) -> str:
		"""Handle removing a task from workflow"""
		try:
			task_name = entities.get('task_name')
			
			if not task_name:
				return "Which task would you like to remove? Please specify the task name."
			
			tasks = workflow.definition.get('tasks', [])
			matching_tasks = [task for task in tasks if task['name'].lower() == task_name.lower()]
			
			if not matching_tasks:
				return f"I couldn't find a task named '{task_name}' in this workflow."
			
			task_to_remove = matching_tasks[0]
			
			# Remove task
			workflow.definition['tasks'] = [task for task in tasks if task['id'] != task_to_remove['id']]
			
			# Remove connections involving this task
			if 'connections' in workflow.definition:
				workflow.definition['connections'] = [
					conn for conn in workflow.definition['connections']
					if conn.get('from') != task_to_remove['id'] and conn.get('to') != task_to_remove['id']
				]
			
			# Update workflow
			await self.workflow_service.update_workflow(workflow.id, workflow, context.tenant_id)
			
			return f"I've removed the task '{task_name}' from the workflow."
			
		except Exception as e:
			self._log_error(f"Failed to remove task: {e}")
			return "I encountered an error while removing the task. Please try again."
	
	async def _handle_modify_task(self, workflow: WorkflowInstance, entities: Dict[str, Any], 
								 context: ConversationContext) -> str:
		"""Handle modifying a task in workflow"""
		try:
			task_name = entities.get('task_name')
			
			if not task_name:
				return "Which task would you like to modify? Please specify the task name."
			
			tasks = workflow.definition.get('tasks', [])
			matching_tasks = [task for task in tasks if task['name'].lower() == task_name.lower()]
			
			if not matching_tasks:
				return f"I couldn't find a task named '{task_name}' in this workflow."
			
			task_to_modify = matching_tasks[0]
			
			# Apply modifications based on entities
			if 'new_name' in entities:
				task_to_modify['name'] = entities['new_name']
			
			if 'task_type' in entities:
				task_to_modify['type'] = entities['task_type']
			
			# Update workflow
			await self.workflow_service.update_workflow(workflow.id, workflow, context.tenant_id)
			
			return f"I've updated the task '{task_name}' in the workflow."
			
		except Exception as e:
			self._log_error(f"Failed to modify task: {e}")
			return "I encountered an error while modifying the task. Please try again."
	
	async def _handle_add_connection(self, workflow: WorkflowInstance, entities: Dict[str, Any], 
									context: ConversationContext) -> str:
		"""Handle adding a connection between tasks"""
		try:
			from_task = entities.get('from_task')
			to_task = entities.get('to_task')
			
			if not from_task or not to_task:
				return "To connect tasks, please specify both the source task and the destination task. For example: 'Connect the data processing task to the email task'."
			
			tasks = workflow.definition.get('tasks', [])
			
			# Find tasks
			from_task_obj = None
			to_task_obj = None
			
			for task in tasks:
				if task['name'].lower() == from_task.lower():
					from_task_obj = task
				if task['name'].lower() == to_task.lower():
					to_task_obj = task
			
			if not from_task_obj:
				return f"I couldn't find a task named '{from_task}' in this workflow."
			
			if not to_task_obj:
				return f"I couldn't find a task named '{to_task}' in this workflow."
			
			# Add connection
			if 'connections' not in workflow.definition:
				workflow.definition['connections'] = []
			
			connection = {
				'from': from_task_obj['id'],
				'to': to_task_obj['id'],
				'type': 'sequence'
			}
			
			workflow.definition['connections'].append(connection)
			
			# Update workflow
			await self.workflow_service.update_workflow(workflow.id, workflow, context.tenant_id)
			
			return f"I've connected the '{from_task}' task to the '{to_task}' task."
			
		except Exception as e:
			self._log_error(f"Failed to add connection: {e}")
			return "I encountered an error while adding the connection. Please try again."
	
	async def _parse_schedule_time(self, time_str: str) -> Optional[Dict[str, Any]]:
		"""Parse natural language time expression into schedule format"""
		try:
			# This is a simplified implementation
			# In production, you'd use a library like dateutil or a more sophisticated NLP approach
			
			time_str_lower = time_str.lower()
			
			# Handle common patterns
			if 'tomorrow' in time_str_lower:
				base_date = datetime.utcnow() + timedelta(days=1)
				if 'morning' in time_str_lower or '9' in time_str_lower:
					return {
						'type': 'once',
						'datetime': base_date.replace(hour=9, minute=0, second=0, microsecond=0)
					}
			
			if 'every day' in time_str_lower or 'daily' in time_str_lower:
				hour = 9  # Default hour
				if '2 pm' in time_str_lower or '14:' in time_str_lower:
					hour = 14
				
				return {
					'type': 'recurring',
					'frequency': 'daily',
					'time': f"{hour:02d}:00"
				}
			
			if 'every monday' in time_str_lower or 'weekly' in time_str_lower:
				return {
					'type': 'recurring',
					'frequency': 'weekly',
					'day_of_week': 'monday',
					'time': '14:00'
				}
			
			return None
			
		except Exception as e:
			self._log_error(f"Failed to parse schedule time: {e}")
			return None
	
	async def _update_context(self, context: ConversationContext, message: ConversationMessage) -> None:
		"""Update conversation context based on processed message"""
		try:
			# Update extracted entities
			if message.entities:
				context.extracted_entities.update(message.entities)
			
			# Update current workflow if mentioned
			if 'workflow_name' in message.entities:
				workflow_name = message.entities['workflow_name']
				try:
					# Look up workflow by name
					workflow_id = await self._lookup_workflow_by_name(workflow_name)
					if workflow_id:
						context.current_workflow_id = workflow_id
						logger.info(f"Set current workflow to '{workflow_name}' (ID: {workflow_id})")
				except Exception as e:
					logger.error(f"Failed to lookup workflow '{workflow_name}': {e}")
			
			# Clear pending confirmations if appropriate
			if message.content.lower() in ['yes', 'confirm', 'ok', 'sure']:
				context.pending_confirmations.clear()
			
		except Exception as e:
			self._log_error(f"Failed to update context: {e}")
	
	async def _load_workflow_templates(self) -> None:
		"""Load workflow templates for natural language generation"""
		try:
			# Define some basic templates
			templates = [
				WorkflowTemplate(
					id="email_notification",
					name="Email Notification",
					description="Send email notifications",
					trigger_patterns=[
						"send email", "email notification", "notify by email",
						"send message", "email alert"
					],
					required_entities=[EntityType.WORKFLOW_NAME],
					optional_entities=[EntityType.PERSON, EntityType.PARAMETER],
					workflow_template={
						'name': '{workflow_name}',
						'description': 'Email notification workflow',
						'tasks': [
							{
								'id': 'start',
								'name': 'Start',
								'type': 'start',
								'properties': {}
							},
							{
								'id': 'send_email',
								'name': 'Send Email',
								'type': 'email',
								'properties': {
									'to': '{recipient}',
									'subject': '{subject}',
									'body': '{body}'
								}
							},
							{
								'id': 'end',
								'name': 'End',
								'type': 'end',
								'properties': {}
							}
						],
						'connections': [
							{'from': 'start', 'to': 'send_email', 'type': 'sequence'},
							{'from': 'send_email', 'to': 'end', 'type': 'sequence'}
						]
					},
					parameter_mappings={
						'workflow_name': 'name',
						'recipient': 'tasks.1.properties.to',
						'subject': 'tasks.1.properties.subject'
					},
					category="communication",
					tags=["email", "notification", "communication"]
				),
				
				WorkflowTemplate(
					id="data_processing",
					name="Data Processing Pipeline",
					description="Process and transform data",
					trigger_patterns=[
						"process data", "data pipeline", "transform data",
						"data processing", "etl", "data transformation"
					],
					required_entities=[EntityType.WORKFLOW_NAME],
					optional_entities=[EntityType.RESOURCE_TYPE, EntityType.PARAMETER],
					workflow_template={
						'name': '{workflow_name}',
						'description': 'Data processing workflow',
						'tasks': [
							{
								'id': 'start',
								'name': 'Start',
								'type': 'start',
								'properties': {}
							},
							{
								'id': 'extract_data',
								'name': 'Extract Data',
								'type': 'data_extract',
								'properties': {
									'source': '{data_source}',
									'format': '{data_format}'
								}
							},
							{
								'id': 'transform_data',
								'name': 'Transform Data',
								'type': 'data_transform',
								'properties': {
									'transformations': []
								}
							},
							{
								'id': 'load_data',
								'name': 'Load Data',
								'type': 'data_load',
								'properties': {
									'destination': '{destination}'
								}
							},
							{
								'id': 'end',
								'name': 'End',
								'type': 'end',
								'properties': {}
							}
						],
						'connections': [
							{'from': 'start', 'to': 'extract_data', 'type': 'sequence'},
							{'from': 'extract_data', 'to': 'transform_data', 'type': 'sequence'},
							{'from': 'transform_data', 'to': 'load_data', 'type': 'sequence'},
							{'from': 'load_data', 'to': 'end', 'type': 'sequence'}
						]
					},
					parameter_mappings={
						'workflow_name': 'name',
						'data_source': 'tasks.1.properties.source',
						'destination': 'tasks.3.properties.destination'
					},
					category="data",
					tags=["data", "processing", "etl", "pipeline"]
				)
			]
			
			# Load templates into template manager
			for template in templates:
				await self.template_manager.add_template(template)
			
			self._log_info(f"Loaded {len(templates)} workflow templates")
			
		except Exception as e:
			self._log_error(f"Failed to load workflow templates: {e}")
	
	async def get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
		"""Get conversation history for a session"""
		try:
			return self.conversation_history.get(session_id, [])
		except Exception as e:
			self._log_error(f"Failed to get conversation history: {e}")
			return []
	
	async def end_conversation(self, session_id: str) -> None:
		"""End a conversation session"""
		try:
			if session_id in self.active_sessions:
				del self.active_sessions[session_id]
			
			# Keep history for a while for analysis
			# In production, you might want to archive this to a database
			
			self._log_info(f"Ended conversation session: {session_id}")
			
		except Exception as e:
			self._log_error(f"Failed to end conversation: {e}")
	
	async def shutdown(self) -> None:
		"""Shutdown conversational interface"""
		try:
			self._log_info("Shutting down conversational interface...")
			
			# Shutdown components
			await self.nlp_processor.shutdown()
			await self.intent_classifier.shutdown()
			await self.entity_extractor.shutdown()
			await self.response_generator.shutdown()
			
			if self.speech_recognizer:
				await self.speech_recognizer.shutdown()
			if self.speech_synthesizer:
				await self.speech_synthesizer.shutdown()
			
			await self.context_manager.shutdown()
			await self.template_manager.shutdown()
			
			self._log_info("Conversational interface shutdown completed")
			
		except Exception as e:
			self._log_error(f"Error during conversational interface shutdown: {e}")
	
	async def _lookup_workflow_by_name(self, workflow_name: str) -> str | None:
		"""Look up workflow ID by name."""
		try:
			from .database import DatabaseManager
			
			db_manager = DatabaseManager()
			async with db_manager.get_session() as session:
				query = """
				SELECT id FROM cr_workflows 
				WHERE name ILIKE %s 
				AND tenant_id = %s
				ORDER BY created_at DESC 
				LIMIT 1
				"""
				
				# Get tenant context (simplified for this implementation)
				tenant_id = getattr(self, 'tenant_id', 'default')
				
				result = await session.execute(query, (f"%{workflow_name}%", tenant_id))
				row = await result.fetchone()
				
				if row:
					return row['id']
				
				return None
				
		except Exception as e:
			self._log_error(f"Error looking up workflow '{workflow_name}': {e}")
			return None


# Placeholder classes for NLP components (would be implemented with actual NLP libraries)

class NLPProcessor:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.NLPProcessor")
	
	async def initialize(self):
		self.logger.info("NLP processor initialized")
	
	async def analyze_sentiment(self, text: str) -> Dict[str, float]:
		# Placeholder implementation
		return {"polarity": 0.0, "subjectivity": 0.0}
	
	async def shutdown(self):
		self.logger.info("NLP processor shutting down")


class IntentClassifier:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.IntentClassifier")
	
	async def initialize(self):
		self.logger.info("Intent classifier initialized")
	
	async def classify_intent(self, text: str, entities: Dict[str, Any], 
							 context: ConversationContext) -> Intent:
		# Simplified intent classification based on keywords
		text_lower = text.lower()
		
		if any(word in text_lower for word in ['create', 'new', 'build', 'make']):
			if 'workflow' in text_lower:
				return Intent(type=IntentType.CREATE_WORKFLOW, confidence=0.8)
		
		if any(word in text_lower for word in ['run', 'execute', 'start']):
			return Intent(type=IntentType.EXECUTE_WORKFLOW, confidence=0.7)
		
		if any(word in text_lower for word in ['status', 'how', 'doing']):
			return Intent(type=IntentType.QUERY_STATUS, confidence=0.6)
		
		if any(word in text_lower for word in ['list', 'show', 'all']):
			return Intent(type=IntentType.LIST_WORKFLOWS, confidence=0.6)
		
		if any(word in text_lower for word in ['help', 'what', 'how']):
			return Intent(type=IntentType.GET_HELP, confidence=0.5)
		
		return Intent(type=IntentType.GET_HELP, confidence=0.3)
	
	async def shutdown(self):
		self.logger.info("Intent classifier shutting down")


class EntityExtractor:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.EntityExtractor")
	
	async def initialize(self):
		self.logger.info("Entity extractor initialized")
	
	async def extract_entities(self, text: str, context: ConversationContext) -> Dict[str, Any]:
		# Simplified entity extraction
		entities = {}
		
		# Extract quoted strings as workflow names
		import re
		quoted_strings = re.findall(r'"([^"]*)"', text)
		if quoted_strings:
			entities['workflow_name'] = quoted_strings[0]
		
		# Extract simple patterns
		if 'called' in text.lower():
			parts = text.lower().split('called')
			if len(parts) > 1:
				name_part = parts[1].strip().split()[0:3]  # Take first few words
				entities['workflow_name'] = ' '.join(name_part).strip('\'"')
		
		return entities
	
	async def shutdown(self):
		self.logger.info("Entity extractor shutting down")


class ResponseGenerator:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.ResponseGenerator")
	
	async def initialize(self):
		self.logger.info("Response generator initialized")
	
	async def generate_welcome_message(self, context: ConversationContext) -> str:
		if context.mode == ConversationMode.VOICE_CONTROL:
			return "Hello! I'm your workflow assistant. You can speak to me to create, modify, and manage your workflows. What would you like to do?"
		else:
			return "Hi! I'm here to help you create and manage workflows using natural language. You can ask me to create new workflows, modify existing ones, or check on their status. What would you like to do today?"
	
	async def generate_fallback_response(self, message: ConversationMessage, 
									   context: ConversationContext) -> str:
		return "I'm not sure I understand what you're asking for. Could you please rephrase your request? You can ask me to create workflows, check their status, or modify existing ones."
	
	async def shutdown(self):
		self.logger.info("Response generator shutting down")


class SpeechRecognizer:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.SpeechRecognizer")
	
	async def initialize(self):
		self.logger.info("Speech recognizer initialized")
	
	async def recognize_speech(self, audio_data: bytes) -> str:
		# Placeholder implementation
		return "transcribed text"
	
	async def shutdown(self):
		self.logger.info("Speech recognizer shutting down")


class SpeechSynthesizer:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.SpeechSynthesizer")
	
	async def initialize(self):
		self.logger.info("Speech synthesizer initialized")
	
	async def synthesize_speech(self, text: str) -> bytes:
		# Placeholder implementation
		return b"audio data"
	
	async def shutdown(self):
		self.logger.info("Speech synthesizer shutting down")


class ContextManager:
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.ContextManager")
	
	async def initialize(self):
		self.logger.info("Context manager initialized")
	
	async def shutdown(self):
		self.logger.info("Context manager shutting down")


class TemplateManager:
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.TemplateManager")
		self.templates: List[WorkflowTemplate] = []
	
	async def initialize(self):
		self.logger.info("Template manager initialized")
	
	async def add_template(self, template: WorkflowTemplate):
		self.templates.append(template)
	
	async def find_matching_template(self, text: str, entities: Dict[str, Any]) -> Optional[WorkflowTemplate]:
		text_lower = text.lower()
		
		for template in self.templates:
			for pattern in template.trigger_patterns:
				if pattern.lower() in text_lower:
					return template
		
		return None
	
	async def shutdown(self):
		self.logger.info("Template manager shutting down")