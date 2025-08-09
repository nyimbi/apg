"""
APG Financial Reporting - Revolutionary Natural Language Processing Engine

AI-powered conversational interface for financial reporting with advanced natural language
understanding, intent classification, and intelligent query processing.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator
from annotated_types import Annotated

from .models import (
	CFRFConversationalInterface, CFRFReportTemplate, CFRFReportDefinition,
	ConversationalIntentType, ReportIntelligenceLevel
)
from ...auth_rbac.models import db
from ...ai_orchestration.service import AIOrchestrationService
from ...nlp_processing.service import NLPProcessingService
from ...generative_ai.service import GenerativeAIService


class FinancialTerminologyType(str, Enum):
	"""Financial terminology classification types."""
	ACCOUNT_TYPE = "account_type"
	STATEMENT_ELEMENT = "statement_element"
	RATIO_METRIC = "ratio_metric"
	PERIOD_REFERENCE = "period_reference"
	ENTITY_REFERENCE = "entity_reference"
	CURRENCY_AMOUNT = "currency_amount"
	PERCENTAGE = "percentage"
	TREND_ANALYSIS = "trend_analysis"


class QueryComplexity(str, Enum):
	"""Query complexity levels for processing optimization."""
	SIMPLE = "simple"			# Single metric, single period
	MODERATE = "moderate"		# Multiple metrics, basic analysis
	COMPLEX = "complex"			# Advanced analysis, multiple entities
	ENTERPRISE = "enterprise"	# Complex consolidation, forecasting


@dataclass
class FinancialEntity:
	"""Extracted financial entity from natural language."""
	entity_type: FinancialTerminologyType
	value: str
	normalized_value: str
	confidence: float
	context: Optional[Dict[str, Any]] = None


@dataclass
class QueryIntent:
	"""Parsed intent from natural language query."""
	intent_type: ConversationalIntentType
	confidence: float
	entities: List[FinancialEntity]
	complexity: QueryComplexity
	suggested_actions: List[str]
	context_requirements: Dict[str, Any]


class ConversationalRequest(BaseModel):
	"""Validated conversational request with AI processing."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	request_id: str = Field(default_factory=uuid7str)
	user_query: str = Field(min_length=1, max_length=1000)
	intent_type: ConversationalIntentType
	confidence_score: float = Field(ge=0.0, le=1.0)
	extracted_entities: Dict[str, Any] = Field(default_factory=dict)
	context_understanding: Dict[str, Any] = Field(default_factory=dict)


class NLPResponse(BaseModel):
	"""AI-generated response with artifacts."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	response_id: str = Field(default_factory=uuid7str)
	response_text: str
	response_type: str = "interactive"
	generated_artifacts: Dict[str, Any] = Field(default_factory=dict)
	suggested_follow_ups: List[str] = Field(default_factory=list)
	confidence_score: float = Field(ge=0.0, le=1.0)


class FinancialNLPEngine:
	"""Revolutionary Natural Language Processing Engine for Financial Reporting using APG AI facilities."""
	
	def __init__(self, tenant_id: str, ai_config: Optional[Dict[str, Any]] = None):
		self.tenant_id = tenant_id
		
		# Initialize APG AI services
		self.ai_orchestration = AIOrchestrationService(tenant_id)
		self.nlp_service = NLPProcessingService(tenant_id)
		self.generative_ai = GenerativeAIService(tenant_id)
		
		# Configure AI preferences (OpenAI, Ollama, or hybrid)
		self.ai_config = ai_config or {
			'primary_provider': 'openai',  # 'openai', 'ollama', or 'hybrid'
			'fallback_provider': 'ollama',
			'model_preferences': {
				'openai': 'gpt-4',
				'ollama': 'llama2:7b'
			},
			'financial_domain_optimization': True
		}
		
		self.financial_terminology = self._load_financial_terminology()
		self.query_patterns = self._compile_query_patterns()
		
	async def process_natural_language_query(self, user_query: str, user_id: str, 
											session_id: str) -> ConversationalRequest:
		"""Process natural language query with advanced AI understanding."""
		assert user_query and len(user_query.strip()) > 0, "Query cannot be empty"
		
		# Analyze query complexity and intent
		query_intent = await self._analyze_query_intent(user_query)
		
		# Extract financial entities
		entities = await self._extract_financial_entities(user_query)
		
		# Build context understanding
		context = await self._build_context_understanding(user_query, entities, query_intent)
		
		# Create conversational interface record
		conversation = CFRFConversationalInterface(
			tenant_id=self.tenant_id,
			user_id=user_id,
			session_id=session_id,
			user_query=user_query,
			intent_classification=query_intent.intent_type.value,
			confidence_score=Decimal(str(query_intent.confidence)),
			extracted_entities=self._entities_to_dict(entities),
			context_understanding=context
		)
		
		db.session.add(conversation)
		db.session.commit()
		
		return ConversationalRequest(
			request_id=conversation.conversation_id,
			user_query=user_query,
			intent_type=query_intent.intent_type,
			confidence_score=query_intent.confidence,
			extracted_entities=self._entities_to_dict(entities),
			context_understanding=context
		)
	
	async def generate_ai_response(self, request: ConversationalRequest) -> NLPResponse:
		"""Generate intelligent AI response using APG AI services with provider flexibility."""
		
		# Build comprehensive prompt
		system_prompt = self._build_system_prompt()
		user_prompt = self._build_user_prompt(request)
		
		try:
			# Use APG AI Orchestration for intelligent provider selection
			ai_request = {
				'prompt': user_prompt,
				'system_prompt': system_prompt,
				'task_type': 'financial_conversation',
				'domain': 'financial_reporting',
				'temperature': 0.3,
				'max_tokens': 2000,
				'functions': [
					{
						"name": "create_financial_report",
						"description": "Create a financial report based on user requirements",
						"parameters": {
							"type": "object",
							"properties": {
								"report_type": {"type": "string"},
								"period": {"type": "string"},
								"entities": {"type": "array", "items": {"type": "string"}},
								"metrics": {"type": "array", "items": {"type": "string"}}
							}
						}
					},
					{
						"name": "analyze_financial_data",
						"description": "Perform financial analysis on specified data",
						"parameters": {
							"type": "object",
							"properties": {
								"analysis_type": {"type": "string"},
								"accounts": {"type": "array", "items": {"type": "string"}},
								"comparison_periods": {"type": "array", "items": {"type": "string"}}
							}
						}
					}
				],
				'provider_preference': self.ai_config.get('primary_provider', 'openai'),
				'fallback_provider': self.ai_config.get('fallback_provider', 'ollama')
			}
			
			# Generate response using APG AI Orchestration
			ai_response = await self.ai_orchestration.generate_completion(ai_request)
			
			response_text = ai_response.get('response', '')
			
			# Handle function calls if present
			artifacts = {}
			if ai_response.get('function_call'):
				artifacts = await self._process_function_call(ai_response['function_call'], request)
			
			# Enhance with financial domain insights using APG NLP
			financial_insights = await self.nlp_service.extract_financial_insights(
				response_text, request.extracted_entities
			)
			artifacts.update({'financial_insights': financial_insights})
			
			# Generate follow-up suggestions
			follow_ups = await self._generate_follow_up_suggestions(request, response_text)
			
			# Update conversation record
			await self._update_conversation_response(request.request_id, response_text, artifacts)
			
			return NLPResponse(
				response_text=response_text,
				generated_artifacts=artifacts,
				suggested_follow_ups=follow_ups,
				confidence_score=ai_response.get('confidence_score', 0.95)
			)
			
		except Exception as e:
			self._log_error(f"AI response generation failed: {str(e)}")
			
			# Fallback to APG Generative AI with simpler prompt
			try:
				fallback_response = await self.generative_ai.generate_text(
					prompt=user_prompt,
					provider=self.ai_config.get('fallback_provider', 'ollama'),
					model=self.ai_config['model_preferences'].get('ollama', 'llama2:7b'),
					temperature=0.3
				)
				
				return NLPResponse(
					response_text=fallback_response.get('text', 'I apologize, but I encountered an issue processing your request.'),
					generated_artifacts={"provider": "fallback", "error": str(e)},
					suggested_follow_ups=[
						"Can you specify which financial statement you need?",
						"What time period should this report cover?",
						"Would you like to see comparative data?"
					],
					confidence_score=fallback_response.get('confidence', 0.7)
				)
			except Exception as fallback_error:
				self._log_error(f"Fallback AI response failed: {str(fallback_error)}")
				
				# Final fallback response
				return NLPResponse(
					response_text="I understand you're asking about financial reporting. Let me help you create the report you need.",
					generated_artifacts={"error": str(e), "fallback_error": str(fallback_error)},
					suggested_follow_ups=[
						"Can you specify which financial statement you need?",
						"What time period should this report cover?",
						"Would you like to see comparative data?"
					],
					confidence_score=0.6
				)
	
	async def _analyze_query_intent(self, query: str) -> QueryIntent:
		"""Analyze user query to determine intent and complexity."""
		query_lower = query.lower()
		
		# Intent classification using pattern matching
		intent_type = ConversationalIntentType.GENERAL_INQUIRY
		confidence = 0.5
		
		# Report creation patterns
		if any(word in query_lower for word in ['create', 'generate', 'build', 'make']):
			intent_type = ConversationalIntentType.REPORT_CREATION
			confidence = 0.8
		
		# Data analysis patterns
		elif any(word in query_lower for word in ['analyze', 'compare', 'trend', 'variance']):
			intent_type = ConversationalIntentType.DATA_ANALYSIS
			confidence = 0.8
		
		# Help and guidance patterns
		elif any(word in query_lower for word in ['help', 'how', 'explain', 'guide']):
			intent_type = ConversationalIntentType.HELP_GUIDANCE
			confidence = 0.9
		
		# Template management patterns
		elif any(word in query_lower for word in ['template', 'format', 'layout']):
			intent_type = ConversationalIntentType.TEMPLATE_MANAGEMENT
			confidence = 0.8
		
		# Determine complexity
		complexity = self._assess_query_complexity(query)
		
		return QueryIntent(
			intent_type=intent_type,
			confidence=confidence,
			entities=[],  # Will be populated separately
			complexity=complexity,
			suggested_actions=[],
			context_requirements={}
		)
	
	async def _extract_financial_entities(self, query: str) -> List[FinancialEntity]:
		"""Extract financial entities using APG NLP services with financial domain optimization."""
		
		# Use APG NLP service for enhanced entity extraction
		nlp_result = await self.nlp_service.extract_entities(
			text=query,
			domain='financial',
			entity_types=[
				'account_type', 'statement_element', 'period_reference', 
				'currency_amount', 'percentage', 'financial_metric'
			]
		)
		
		entities = []
		
		# Convert APG NLP results to our financial entity format
		for nlp_entity in nlp_result.get('entities', []):
			entity_type_mapping = {
				'account_type': FinancialTerminologyType.ACCOUNT_TYPE,
				'statement_element': FinancialTerminologyType.STATEMENT_ELEMENT,
				'period_reference': FinancialTerminologyType.PERIOD_REFERENCE,
				'currency_amount': FinancialTerminologyType.CURRENCY_AMOUNT,
				'percentage': FinancialTerminologyType.PERCENTAGE,
				'financial_metric': FinancialTerminologyType.RATIO_METRIC
			}
			
			entity_type = entity_type_mapping.get(
				nlp_entity.get('type'), 
				FinancialTerminologyType.ACCOUNT_TYPE
			)
			
			entities.append(FinancialEntity(
				entity_type=entity_type,
				value=nlp_entity.get('text', ''),
				normalized_value=nlp_entity.get('normalized_value', nlp_entity.get('text', '')),
				confidence=nlp_entity.get('confidence', 0.8),
				context=nlp_entity.get('context', {})
			))
		
		# Fallback to pattern-based extraction if APG NLP doesn't find entities
		if not entities:
			entities = await self._fallback_entity_extraction(query)
		
		return entities
	
	async def _fallback_entity_extraction(self, query: str) -> List[FinancialEntity]:
		"""Fallback entity extraction using pattern matching."""
		entities = []
		
		# Extract account types
		for account_type in ['assets', 'liabilities', 'equity', 'revenue', 'expenses']:
			if account_type in query.lower():
				entities.append(FinancialEntity(
					entity_type=FinancialTerminologyType.ACCOUNT_TYPE,
					value=account_type,
					normalized_value=account_type.upper(),
					confidence=0.9
				))
		
		# Extract statement types
		statement_patterns = {
			'balance sheet': 'BALANCE_SHEET',
			'income statement': 'INCOME_STATEMENT',
			'p&l': 'INCOME_STATEMENT',
			'profit and loss': 'INCOME_STATEMENT',
			'cash flow': 'CASH_FLOW',
			'statement of equity': 'EQUITY_STATEMENT'
		}
		
		for pattern, normalized in statement_patterns.items():
			if pattern in query.lower():
				entities.append(FinancialEntity(
					entity_type=FinancialTerminologyType.STATEMENT_ELEMENT,
					value=pattern,
					normalized_value=normalized,
					confidence=0.95
				))
		
		# Extract period references
		period_patterns = {
			r'q[1-4]': 'QUARTERLY',
			r'\d{4}': 'YEARLY',
			r'month': 'MONTHLY',
			r'ytd': 'YEAR_TO_DATE',
			r'current': 'CURRENT_PERIOD'
		}
		
		for pattern, period_type in period_patterns.items():
			matches = re.finditer(pattern, query.lower())
			for match in matches:
				entities.append(FinancialEntity(
					entity_type=FinancialTerminologyType.PERIOD_REFERENCE,
					value=match.group(),
					normalized_value=period_type,
					confidence=0.8
				))
		
		# Extract currency amounts
		currency_pattern = r'\$[\d,]+(?:\.\d{2})?'
		for match in re.finditer(currency_pattern, query):
			entities.append(FinancialEntity(
				entity_type=FinancialTerminologyType.CURRENCY_AMOUNT,
				value=match.group(),
				normalized_value=match.group().replace(',', ''),
				confidence=0.9
			))
		
		return entities
	
	async def _build_context_understanding(self, query: str, entities: List[FinancialEntity], 
										  intent: QueryIntent) -> Dict[str, Any]:
		"""Build comprehensive context understanding for the query."""
		
		context = {
			'query_analysis': {
				'word_count': len(query.split()),
				'has_comparison': any(word in query.lower() for word in ['vs', 'versus', 'compared to', 'against']),
				'has_time_reference': any(e.entity_type == FinancialTerminologyType.PERIOD_REFERENCE for e in entities),
				'has_financial_terms': len([e for e in entities if e.entity_type in [
					FinancialTerminologyType.ACCOUNT_TYPE, 
					FinancialTerminologyType.STATEMENT_ELEMENT
				]]) > 0
			},
			'suggested_templates': await self._suggest_relevant_templates(entities, intent),
			'required_permissions': self._determine_required_permissions(intent),
			'estimated_complexity': intent.complexity.value,
			'processing_hints': self._generate_processing_hints(query, entities)
		}
		
		return context
	
	async def _suggest_relevant_templates(self, entities: List[FinancialEntity], 
										 intent: QueryIntent) -> List[Dict[str, str]]:
		"""Suggest relevant report templates based on extracted entities."""
		
		# Get statement types from entities
		statement_types = [
			e.normalized_value for e in entities 
			if e.entity_type == FinancialTerminologyType.STATEMENT_ELEMENT
		]
		
		if not statement_types:
			# Default suggestions based on intent
			if intent.intent_type == ConversationalIntentType.REPORT_CREATION:
				statement_types = ['BALANCE_SHEET', 'INCOME_STATEMENT']
		
		suggestions = []
		for statement_type in statement_types:
			templates = db.session.query(CFRFReportTemplate).filter(
				CFRFReportTemplate.tenant_id == self.tenant_id,
				CFRFReportTemplate.statement_type == statement_type.lower(),
				CFRFReportTemplate.is_active == True
			).limit(3).all()
			
			for template in templates:
				suggestions.append({
					'template_id': template.template_id,
					'template_name': template.template_name,
					'statement_type': template.statement_type,
					'ai_intelligence_level': template.ai_intelligence_level
				})
		
		return suggestions
	
	def _assess_query_complexity(self, query: str) -> QueryComplexity:
		"""Assess the complexity level of the user query."""
		complexity_indicators = {
			QueryComplexity.SIMPLE: ['show', 'display', 'what is'],
			QueryComplexity.MODERATE: ['compare', 'analyze', 'calculate', 'variance'],
			QueryComplexity.COMPLEX: ['consolidate', 'forecast', 'trend analysis', 'multiple entities'],
			QueryComplexity.ENTERPRISE: ['consolidation', 'multi-currency', 'forecasting model', 'predictive']
		}
		
		query_lower = query.lower()
		
		# Check for enterprise complexity first
		if any(indicator in query_lower for indicator in complexity_indicators[QueryComplexity.ENTERPRISE]):
			return QueryComplexity.ENTERPRISE
		elif any(indicator in query_lower for indicator in complexity_indicators[QueryComplexity.COMPLEX]):
			return QueryComplexity.COMPLEX
		elif any(indicator in query_lower for indicator in complexity_indicators[QueryComplexity.MODERATE]):
			return QueryComplexity.MODERATE
		else:
			return QueryComplexity.SIMPLE
	
	def _build_system_prompt(self) -> str:
		"""Build comprehensive system prompt for GPT-4."""
		return """You are an advanced AI assistant specialized in financial reporting and analysis. 
		You have deep expertise in:
		- Financial statement preparation (Balance Sheet, Income Statement, Cash Flow, Equity)
		- Financial analysis and variance reporting
		- Consolidation and multi-entity reporting
		- Regulatory compliance (SOX, IFRS, GAAP)
		- Management reporting and KPI analysis
		
		Your responses should be:
		- Professional and accurate
		- Specific to financial reporting contexts
		- Actionable with clear next steps
		- Compliant with accounting standards
		
		Always provide specific, actionable recommendations and offer to create the requested reports or analysis."""
	
	def _build_user_prompt(self, request: ConversationalRequest) -> str:
		"""Build user prompt with context for GPT-4."""
		entities_text = ", ".join([
			f"{e.get('entity_type', '')}: {e.get('normalized_value', '')}" 
			for e in request.extracted_entities.get('entities', [])
		])
		
		return f"""
		User Query: {request.user_query}
		
		Detected Intent: {request.intent_type.value}
		Confidence: {request.confidence_score}
		
		Extracted Financial Entities: {entities_text}
		
		Context: {json.dumps(request.context_understanding, indent=2)}
		
		Please provide a helpful response and determine if you need to call any functions to assist the user.
		"""
	
	async def _process_function_call(self, function_call, request: ConversationalRequest) -> Dict[str, Any]:
		"""Process function calls from AI response."""
		function_name = function_call.name
		function_args = json.loads(function_call.arguments)
		
		artifacts = {
			'function_called': function_name,
			'function_arguments': function_args,
			'timestamp': datetime.now().isoformat()
		}
		
		if function_name == "create_financial_report":
			artifacts['report_template'] = await self._create_report_template_artifact(function_args)
		elif function_name == "analyze_financial_data":
			artifacts['analysis_config'] = await self._create_analysis_config_artifact(function_args)
		
		return artifacts
	
	async def _create_report_template_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Create report template artifact from function arguments."""
		return {
			'suggested_template': {
				'report_type': args.get('report_type', 'income_statement'),
				'period': args.get('period', 'monthly'),
				'entities': args.get('entities', []),
				'metrics': args.get('metrics', []),
				'format': 'comparative'
			},
			'next_steps': [
				'Review suggested template configuration',
				'Specify reporting period and entities',
				'Customize report layout and formatting',
				'Generate report with AI assistance'
			]
		}
	
	async def _create_analysis_config_artifact(self, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Create analysis configuration artifact."""
		return {
			'analysis_configuration': {
				'analysis_type': args.get('analysis_type', 'variance'),
				'accounts': args.get('accounts', []),
				'comparison_periods': args.get('comparison_periods', []),
				'metrics': ['variance_percentage', 'trend_analysis', 'ratio_analysis']
			},
			'visualization_options': [
				'Trend charts',
				'Variance analysis tables',
				'Ratio comparison charts',
				'Executive summary dashboard'
			]
		}
	
	async def _generate_follow_up_suggestions(self, request: ConversationalRequest, 
											 response_text: str) -> List[str]:
		"""Generate intelligent follow-up suggestions."""
		suggestions = []
		
		if request.intent_type == ConversationalIntentType.REPORT_CREATION:
			suggestions.extend([
				"Would you like to customize the report format?",
				"Should I include comparative periods?",
				"Do you need this report distributed to stakeholders?"
			])
		elif request.intent_type == ConversationalIntentType.DATA_ANALYSIS:
			suggestions.extend([
				"Would you like me to create visualizations for this analysis?",
				"Should I generate predictive insights?",
				"Do you want to drill down into specific accounts?"
			])
		elif request.intent_type == ConversationalIntentType.HELP_GUIDANCE:
			suggestions.extend([
				"Would you like a tutorial on this feature?",
				"Should I show you examples?",
				"Do you need help with specific report types?"
			])
		
		return suggestions[:3]  # Limit to 3 suggestions
	
	async def _update_conversation_response(self, conversation_id: str, response_text: str, 
										   artifacts: Dict[str, Any]):
		"""Update conversation record with AI response."""
		conversation = db.session.query(CFRFConversationalInterface).filter(
			CFRFConversationalInterface.conversation_id == conversation_id
		).first()
		
		if conversation:
			conversation.ai_response = response_text
			conversation.generated_artifacts = artifacts
			conversation.processing_time_ms = 1500  # Estimated processing time
			conversation.model_version = "gpt-4"
			conversation.updated_at = datetime.now()
			
			db.session.commit()
	
	def _entities_to_dict(self, entities: List[FinancialEntity]) -> Dict[str, Any]:
		"""Convert financial entities to dictionary format."""
		return {
			'entities': [
				{
					'entity_type': entity.entity_type.value,
					'value': entity.value,
					'normalized_value': entity.normalized_value,
					'confidence': entity.confidence,
					'context': entity.context or {}
				}
				for entity in entities
			],
			'entity_count': len(entities),
			'high_confidence_entities': [
				e for e in entities if e.confidence > 0.8
			]
		}
	
	def _determine_required_permissions(self, intent: QueryIntent) -> List[str]:
		"""Determine required permissions for the query."""
		permissions = ['financial_reporting_read']
		
		if intent.intent_type == ConversationalIntentType.REPORT_CREATION:
			permissions.append('financial_reporting_create')
		elif intent.intent_type == ConversationalIntentType.TEMPLATE_MANAGEMENT:
			permissions.extend(['template_read', 'template_create'])
		
		return permissions
	
	def _generate_processing_hints(self, query: str, entities: List[FinancialEntity]) -> List[str]:
		"""Generate processing hints for optimal query handling."""
		hints = []
		
		if not entities:
			hints.append("Consider being more specific about financial statements or metrics")
		
		if 'comparison' in query.lower() and not any(
			e.entity_type == FinancialTerminologyType.PERIOD_REFERENCE for e in entities
		):
			hints.append("Specify time periods for comparison analysis")
		
		if len(query.split()) < 5:
			hints.append("More detailed queries yield better AI assistance")
		
		return hints
	
	def _load_financial_terminology(self) -> Dict[str, Any]:
		"""Load financial terminology for enhanced NLP processing."""
		return {
			'accounting_terms': [
				'assets', 'liabilities', 'equity', 'revenue', 'expenses',
				'depreciation', 'amortization', 'accruals', 'deferrals'
			],
			'financial_ratios': [
				'current ratio', 'debt to equity', 'roe', 'roa', 'gross margin',
				'operating margin', 'net margin', 'asset turnover'
			],
			'statement_types': [
				'balance sheet', 'income statement', 'cash flow statement',
				'statement of equity', 'notes to financial statements'
			],
			'period_types': [
				'monthly', 'quarterly', 'annually', 'ytd', 'current', 'prior'
			]
		}
	
	def _compile_query_patterns(self) -> Dict[str, Any]:
		"""Compile regex patterns for query analysis."""
		return {
			'currency_amounts': re.compile(r'\$[\d,]+(?:\.\d{2})?'),
			'percentages': re.compile(r'\d+(?:\.\d+)?%'),
			'dates': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'),
			'quarters': re.compile(r'q[1-4]', re.IGNORECASE),
			'years': re.compile(r'\b20\d{2}\b')
		}
	
	def _log_error(self, error_message: str):
		"""Log NLP processing errors."""
		print(f"[NLP Engine Error] {error_message}")
		# In production, this would integrate with proper logging system