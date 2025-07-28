"""
APG Financial Reporting - Ollama Integration for Open-Source AI Models

Integration with Ollama for running open-weight models locally, providing privacy-focused
AI capabilities with support for financial domain-specific fine-tuned models.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class OllamaModelType(str, Enum):
	"""Supported Ollama model types for financial analysis."""
	LLAMA2_7B = "llama2:7b"
	LLAMA2_13B = "llama2:13b"
	LLAMA2_70B = "llama2:70b"
	CODELLAMA_7B = "codellama:7b"
	CODELLAMA_13B = "codellama:13b"
	MISTRAL_7B = "mistral:7b"
	MIXTRAL_8X7B = "mixtral:8x7b"
	NEURAL_CHAT = "neural-chat:7b"
	VICUNA_7B = "vicuna:7b"
	VICUNA_13B = "vicuna:13b"
	# Financial domain-specific models
	FINLLAMA_7B = "finllama:7b"				# Custom financial model
	ACCOUNTING_GPT = "accounting-gpt:7b"		# Accounting-focused model
	RISK_ANALYZER = "risk-analyzer:7b"		# Risk analysis model


@dataclass
class OllamaConfig:
	"""Configuration for Ollama integration."""
	base_url: str = "http://localhost:11434"
	model: OllamaModelType = OllamaModelType.LLAMA2_7B
	timeout: int = 60
	temperature: float = 0.3
	top_p: float = 0.9
	top_k: int = 40
	num_predict: int = 2048
	context_length: int = 4096
	system_prompt: Optional[str] = None
	financial_domain_optimization: bool = True


class OllamaResponse(BaseModel):
	"""Structured response from Ollama API."""
	model_config = ConfigDict(extra='forbid')
	
	response: str
	model: str
	created_at: datetime
	done: bool
	context: Optional[List[int]] = None
	total_duration: Optional[int] = None
	load_duration: Optional[int] = None
	prompt_eval_count: Optional[int] = None
	prompt_eval_duration: Optional[int] = None
	eval_count: Optional[int] = None
	eval_duration: Optional[int] = None


class OllamaFinancialAI:
	"""Ollama integration for open-source AI in financial reporting."""
	
	def __init__(self, config: OllamaConfig):
		self.config = config
		self.session: Optional[aiohttp.ClientSession] = None
		self.available_models: List[str] = []
		self.model_info: Dict[str, Dict] = {}
		
		# Financial domain prompts
		self.financial_system_prompts = {
			'general': """You are an expert financial analyst and accountant with deep knowledge of:
- Financial statement preparation (Balance Sheet, Income Statement, Cash Flow, Equity)
- Financial analysis and variance reporting
- Consolidation and multi-entity reporting
- Regulatory compliance (SOX, IFRS, GAAP)
- Management reporting and KPI analysis

Provide accurate, professional, and actionable financial insights. Always consider accounting standards and regulatory requirements.""",
			
			'variance_analysis': """You are a financial variance analysis expert. Analyze financial data to:
- Identify significant variances between actual and budgeted amounts
- Explain potential causes of variances
- Provide actionable recommendations for variance management
- Consider seasonal patterns and business drivers""",
			
			'consolidation': """You are an expert in financial consolidation and multi-entity reporting. Your expertise includes:
- Intercompany elimination procedures
- Currency translation and hedging
- Purchase price allocation
- Minority interest calculations
- Consolidation compliance requirements""",
			
			'risk_assessment': """You are a financial risk assessment specialist. Analyze financial data to:
- Identify potential financial risks
- Assess credit and liquidity risks
- Evaluate operational and market risks
- Provide risk mitigation recommendations"""
		}
	
	async def __aenter__(self):
		"""Async context manager entry."""
		self.session = aiohttp.ClientSession(
			timeout=aiohttp.ClientTimeout(total=self.config.timeout)
		)
		await self._initialize_connection()
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		"""Async context manager exit."""
		if self.session:
			await self.session.close()
	
	async def _initialize_connection(self):
		"""Initialize connection and check available models."""
		try:
			await self._check_ollama_status()
			await self._load_available_models()
			await self._ensure_model_available()
		except Exception as e:
			logger.error(f"Failed to initialize Ollama connection: {e}")
			raise
	
	async def _check_ollama_status(self):
		"""Check if Ollama server is running and accessible."""
		try:
			async with self.session.get(f"{self.config.base_url}/api/tags") as response:
				if response.status != 200:
					raise ConnectionError(f"Ollama server returned status {response.status}")
		except aiohttp.ClientError as e:
			raise ConnectionError(f"Cannot connect to Ollama server at {self.config.base_url}: {e}")
	
	async def _load_available_models(self):
		"""Load list of available models from Ollama."""
		try:
			async with self.session.get(f"{self.config.base_url}/api/tags") as response:
				data = await response.json()
				self.available_models = [model['name'] for model in data.get('models', [])]
				
				# Store model information
				for model in data.get('models', []):
					self.model_info[model['name']] = {
						'size': model.get('size', 0),
						'modified_at': model.get('modified_at'),
						'digest': model.get('digest')
					}
		except Exception as e:
			logger.warning(f"Failed to load available models: {e}")
			self.available_models = []
	
	async def _ensure_model_available(self):
		"""Ensure the configured model is available, pull if necessary."""
		model_name = self.config.model.value
		
		if model_name not in self.available_models:
			logger.info(f"Model {model_name} not found locally, attempting to pull...")
			await self._pull_model(model_name)
	
	async def _pull_model(self, model_name: str):
		"""Pull a model from Ollama registry."""
		try:
			pull_data = {"name": model_name}
			
			async with self.session.post(
				f"{self.config.base_url}/api/pull",
				json=pull_data
			) as response:
				async for line in response.content:
					if line:
						status = json.loads(line.decode())
						if status.get('status') == 'success':
							logger.info(f"Successfully pulled model {model_name}")
							break
						elif 'error' in status:
							raise Exception(f"Failed to pull model: {status['error']}")
		except Exception as e:
			logger.error(f"Failed to pull model {model_name}: {e}")
			raise
	
	async def generate_financial_analysis(self, prompt: str, 
										 analysis_type: str = 'general',
										 context: Optional[Dict[str, Any]] = None) -> OllamaResponse:
		"""Generate financial analysis using Ollama model."""
		
		# Build system prompt
		system_prompt = self.financial_system_prompts.get(analysis_type, self.financial_system_prompts['general'])
		
		# Add context if provided
		if context:
			context_text = self._format_financial_context(context)
			prompt = f"Context:\n{context_text}\n\nAnalysis Request:\n{prompt}"
		
		# Add financial domain optimization
		if self.config.financial_domain_optimization:
			prompt = self._optimize_for_financial_domain(prompt, analysis_type)
		
		return await self._generate_completion(prompt, system_prompt)
	
	async def analyze_financial_variance(self, variance_data: Dict[str, Any]) -> OllamaResponse:
		"""Analyze financial variances using specialized prompt."""
		
		prompt = f"""
		Analyze the following financial variance data and provide insights:
		
		Variance Summary:
		{json.dumps(variance_data, indent=2)}
		
		Please provide:
		1. Identification of significant variances (>10% or material amounts)
		2. Potential causes for each significant variance
		3. Recommended follow-up actions
		4. Risk assessment of the variances
		5. Suggested management attention areas
		"""
		
		return await self.generate_financial_analysis(prompt, 'variance_analysis', variance_data)
	
	async def generate_financial_narrative(self, financial_data: Dict[str, Any],
										  narrative_type: str = 'executive_summary') -> OllamaResponse:
		"""Generate financial narrative and commentary."""
		
		prompt_templates = {
			'executive_summary': """
			Create an executive summary for the following financial data:
			
			{financial_data}
			
			The summary should be concise, highlight key performance indicators, 
			identify trends, and provide strategic insights for management.
			""",
			
			'variance_explanation': """
			Provide detailed explanations for the variances in this financial data:
			
			{financial_data}
			
			Focus on material variances and their business implications.
			""",
			
			'trend_analysis': """
			Analyze trends and patterns in this financial data:
			
			{financial_data}
			
			Identify emerging trends, seasonal patterns, and potential future implications.
			"""
		}
		
		prompt = prompt_templates.get(narrative_type, prompt_templates['executive_summary'])
		prompt = prompt.format(financial_data=json.dumps(financial_data, indent=2))
		
		return await self.generate_financial_analysis(prompt, 'general', financial_data)
	
	async def process_natural_language_query(self, user_query: str,
											financial_context: Optional[Dict] = None) -> OllamaResponse:
		"""Process natural language queries about financial data."""
		
		system_prompt = """You are a financial reporting AI assistant. Help users understand and analyze financial data through natural language conversation. 

Capabilities:
- Answer questions about financial statements
- Explain financial ratios and metrics
- Provide variance analysis
- Generate report recommendations
- Help with financial data interpretation

Always provide accurate, professional responses grounded in accounting principles."""
		
		if financial_context:
			context_text = self._format_financial_context(financial_context)
			enhanced_query = f"""
			Financial Context:
			{context_text}
			
			User Question: {user_query}
			
			Please provide a helpful response based on the financial context provided.
			"""
		else:
			enhanced_query = user_query
		
		return await self._generate_completion(enhanced_query, system_prompt)
	
	async def detect_financial_anomalies(self, financial_data: List[Dict[str, Any]]) -> OllamaResponse:
		"""Detect potential anomalies in financial data."""
		
		prompt = f"""
		Analyze the following financial data for potential anomalies or unusual patterns:
		
		Financial Data:
		{json.dumps(financial_data, indent=2)}
		
		Please identify:
		1. Statistical outliers or unusual values
		2. Unexpected patterns or trends
		3. Data quality issues
		4. Potential fraud indicators
		5. Recommended investigation areas
		
		Provide a structured analysis with confidence levels for each finding.
		"""
		
		return await self.generate_financial_analysis(prompt, 'risk_assessment', {'data': financial_data})
	
	async def generate_consolidation_insights(self, entity_data: Dict[str, Any],
											 consolidation_rules: Dict[str, Any]) -> OllamaResponse:
		"""Generate insights for financial consolidation."""
		
		prompt = f"""
		Analyze this multi-entity financial data for consolidation:
		
		Entity Data:
		{json.dumps(entity_data, indent=2)}
		
		Consolidation Rules:
		{json.dumps(consolidation_rules, indent=2)}
		
		Please provide:
		1. Intercompany transaction analysis
		2. Elimination entry recommendations
		3. Currency translation considerations
		4. Consolidation risk assessment
		5. Compliance check recommendations
		"""
		
		return await self.generate_financial_analysis(prompt, 'consolidation', {
			'entities': entity_data,
			'rules': consolidation_rules
		})
	
	async def _generate_completion(self, prompt: str, system_prompt: Optional[str] = None) -> OllamaResponse:
		"""Generate completion using Ollama API."""
		
		# Prepare request data
		request_data = {
			"model": self.config.model.value,
			"prompt": prompt,
			"options": {
				"temperature": self.config.temperature,
				"top_p": self.config.top_p,
				"top_k": self.config.top_k,
				"num_predict": self.config.num_predict,
			},
			"stream": False
		}
		
		# Add system prompt if provided
		if system_prompt or self.config.system_prompt:
			request_data["system"] = system_prompt or self.config.system_prompt
		
		try:
			async with self.session.post(
				f"{self.config.base_url}/api/generate",
				json=request_data
			) as response:
				if response.status != 200:
					error_text = await response.text()
					raise Exception(f"Ollama API error {response.status}: {error_text}")
				
				result = await response.json()
				
				return OllamaResponse(
					response=result.get('response', ''),
					model=result.get('model', self.config.model.value),
					created_at=datetime.now(),
					done=result.get('done', True),
					context=result.get('context'),
					total_duration=result.get('total_duration'),
					load_duration=result.get('load_duration'),
					prompt_eval_count=result.get('prompt_eval_count'),
					prompt_eval_duration=result.get('prompt_eval_duration'),
					eval_count=result.get('eval_count'),
					eval_duration=result.get('eval_duration')
				)
		
		except Exception as e:
			logger.error(f"Failed to generate completion: {e}")
			raise
	
	def _format_financial_context(self, context: Dict[str, Any]) -> str:
		"""Format financial context for inclusion in prompts."""
		formatted_lines = []
		
		for key, value in context.items():
			if isinstance(value, dict):
				formatted_lines.append(f"{key.title()}:")
				for sub_key, sub_value in value.items():
					formatted_lines.append(f"  {sub_key}: {sub_value}")
			elif isinstance(value, list):
				formatted_lines.append(f"{key.title()}: {len(value)} items")
			else:
				formatted_lines.append(f"{key.title()}: {value}")
		
		return "\n".join(formatted_lines)
	
	def _optimize_for_financial_domain(self, prompt: str, analysis_type: str) -> str:
		"""Optimize prompt for financial domain specificity."""
		
		domain_optimizations = {
			'general': "Focus on financial accuracy, accounting standards compliance, and business insights.",
			'variance_analysis': "Emphasize materiality thresholds, business drivers, and actionable recommendations.",
			'consolidation': "Consider accounting standards (IFRS/GAAP), elimination procedures, and regulatory compliance.",
			'risk_assessment': "Evaluate financial, operational, and regulatory risks with quantitative analysis."
		}
		
		optimization = domain_optimizations.get(analysis_type, domain_optimizations['general'])
		
		return f"{prompt}\n\nImportant: {optimization}"
	
	async def get_model_info(self) -> Dict[str, Any]:
		"""Get information about the current model."""
		return {
			'model': self.config.model.value,
			'available_models': self.available_models,
			'model_info': self.model_info.get(self.config.model.value, {}),
			'config': {
				'base_url': self.config.base_url,
				'temperature': self.config.temperature,
				'context_length': self.config.context_length,
				'financial_optimization': self.config.financial_domain_optimization
			}
		}
	
	async def switch_model(self, new_model: OllamaModelType):
		"""Switch to a different model."""
		old_model = self.config.model
		self.config.model = new_model
		
		try:
			await self._ensure_model_available()
			logger.info(f"Successfully switched from {old_model.value} to {new_model.value}")
		except Exception as e:
			# Revert to old model on failure
			self.config.model = old_model
			logger.error(f"Failed to switch to {new_model.value}, reverted to {old_model.value}: {e}")
			raise


class OllamaModelManager:
	"""Manager for Ollama models with financial domain optimization."""
	
	def __init__(self, base_url: str = "http://localhost:11434"):
		self.base_url = base_url
		self.session: Optional[aiohttp.ClientSession] = None
	
	async def __aenter__(self):
		self.session = aiohttp.ClientSession()
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		if self.session:
			await self.session.close()
	
	async def list_models(self) -> List[Dict[str, Any]]:
		"""List all available models."""
		async with self.session.get(f"{self.base_url}/api/tags") as response:
			data = await response.json()
			return data.get('models', [])
	
	async def pull_financial_models(self) -> Dict[str, bool]:
		"""Pull recommended financial models."""
		recommended_models = [
			OllamaModelType.LLAMA2_7B,
			OllamaModelType.MISTRAL_7B,
			OllamaModelType.NEURAL_CHAT
		]
		
		results = {}
		for model in recommended_models:
			try:
				await self._pull_model(model.value)
				results[model.value] = True
			except Exception as e:
				logger.error(f"Failed to pull {model.value}: {e}")
				results[model.value] = False
		
		return results
	
	async def _pull_model(self, model_name: str):
		"""Pull a specific model."""
		pull_data = {"name": model_name}
		
		async with self.session.post(
			f"{self.base_url}/api/pull",
			json=pull_data
		) as response:
			async for line in response.content:
				if line:
					status = json.loads(line.decode())
					if status.get('status') == 'success':
						break
					elif 'error' in status:
						raise Exception(f"Failed to pull model: {status['error']}")
	
	async def get_model_recommendations(self, use_case: str = 'general') -> List[OllamaModelType]:
		"""Get model recommendations based on use case."""
		
		recommendations = {
			'general': [OllamaModelType.LLAMA2_7B, OllamaModelType.MISTRAL_7B],
			'analysis': [OllamaModelType.LLAMA2_13B, OllamaModelType.MIXTRAL_8X7B],
			'conversation': [OllamaModelType.NEURAL_CHAT, OllamaModelType.VICUNA_7B],
			'code': [OllamaModelType.CODELLAMA_7B, OllamaModelType.CODELLAMA_13B],
			'finance': [OllamaModelType.FINLLAMA_7B, OllamaModelType.ACCOUNTING_GPT, OllamaModelType.RISK_ANALYZER]
		}
		
		return recommendations.get(use_case, recommendations['general'])