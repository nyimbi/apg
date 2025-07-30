"""
APG Document Content Management - Generative AI Service

Generative AI integration for content interaction including summarization,
question-answering, content generation, and translation capabilities.

Copyright © 2025 Datacraft
Author: Nyimbi Odero <nyimbi@gmail.com>
Website: www.datacraft.co.ke
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from .models import DCMDocument, DCMGenerativeAI, DCMContentIntelligence


class GenerativeAIEngine:
	"""Generative AI engine for content interaction and enhancement"""
	
	def __init__(self, apg_ai_client=None, apg_rag_client=None, apg_genai_client=None):
		"""Initialize generative AI engine with APG clients"""
		self.apg_ai_client = apg_ai_client
		self.apg_rag_client = apg_rag_client
		self.apg_genai_client = apg_genai_client
		self.logger = logging.getLogger(__name__)
		
		# Interaction types and their configurations
		self.interaction_types = {
			'summarize': {
				'description': 'Generate document summary',
				'max_input_tokens': 4000,
				'max_output_tokens': 500,
				'temperature': 0.3,
				'requires_context': True
			},
			'qa': {
				'description': 'Answer questions about document content',
				'max_input_tokens': 3000,
				'max_output_tokens': 300,
				'temperature': 0.1,
				'requires_context': True
			},
			'translate': {
				'description': 'Translate document content',
				'max_input_tokens': 2000,
				'max_output_tokens': 2000,
				'temperature': 0.2,
				'requires_context': False
			},
			'enhance': {
				'description': 'Enhance and improve content quality',
				'max_input_tokens': 1500,
				'max_output_tokens': 1500,
				'temperature': 0.4,
				'requires_context': False
			},
			'generate': {
				'description': 'Generate new content based on templates',
				'max_input_tokens': 1000,
				'max_output_tokens': 2000,
				'temperature': 0.7,
				'requires_context': True
			},
			'extract': {
				'description': 'Extract specific information from content',
				'max_input_tokens': 3000,
				'max_output_tokens': 500,
				'temperature': 0.1,
				'requires_context': True
			},
			'compare': {
				'description': 'Compare multiple documents',
				'max_input_tokens': 4000,
				'max_output_tokens': 800,
				'temperature': 0.2,
				'requires_context': True
			},
			'analyze': {
				'description': 'Analyze document content for insights',
				'max_input_tokens': 3500,
				'max_output_tokens': 600,
				'temperature': 0.3,
				'requires_context': True
			}
		}
		
		# Language support
		self.supported_languages = {
			'en': 'English',
			'es': 'Spanish',
			'fr': 'French',
			'de': 'German',
			'it': 'Italian',
			'pt': 'Portuguese',
			'ru': 'Russian',
			'zh': 'Chinese',
			'ja': 'Japanese',
			'ko': 'Korean',
			'ar': 'Arabic',
			'hi': 'Hindi'
		}
		
		# Content templates
		self.content_templates = {
			'executive_summary': {
				'structure': ['overview', 'key_findings', 'recommendations', 'conclusion'],
				'tone': 'professional',
				'length': 'concise'
			},
			'technical_report': {
				'structure': ['abstract', 'introduction', 'methodology', 'results', 'discussion'],
				'tone': 'technical',
				'length': 'detailed'
			},
			'meeting_minutes': {
				'structure': ['attendees', 'agenda', 'discussions', 'decisions', 'action_items'],
				'tone': 'formal',
				'length': 'structured'
			},
			'policy_document': {
				'structure': ['purpose', 'scope', 'policy_statement', 'procedures', 'compliance'],
				'tone': 'authoritative',
				'length': 'comprehensive'
			}
		}
		
		# Performance tracking
		self.genai_stats = {
			'total_interactions': 0,
			'successful_responses': 0,
			'average_response_time': 0.0,
			'user_satisfaction_scores': [],
			'token_usage': {
				'input_tokens': 0,
				'output_tokens': 0
			},
			'interaction_types': {}
		}
	
	async def process_interaction(
		self,
		document: DCMDocument,
		user_prompt: str,
		interaction_type: str,
		user_id: str,
		context_documents: Optional[List[str]] = None,
		options: Optional[Dict[str, Any]] = None
	) -> DCMGenerativeAI:
		"""Process generative AI interaction with document content"""
		start_time = time.time()
		options = options or {}
		
		try:
			# Validate interaction type
			if interaction_type not in self.interaction_types:
				raise ValueError(f"Unsupported interaction type: {interaction_type}")
			
			# Get interaction configuration
			config = self.interaction_types[interaction_type]
			
			# Prepare context
			context_data = await self._prepare_context(
				document,
				context_documents,
				config['requires_context']
			)
			
			# Generate RAG context if needed
			rag_context_id = None
			if config['requires_context'] and self.apg_rag_client:
				rag_context_id = await self._create_rag_context(
					document.id,
					user_prompt,
					context_documents
				)
			
			# Process interaction based on type
			if interaction_type == 'summarize':
				response = await self._summarize_content(user_prompt, context_data, options)
			elif interaction_type == 'qa':
				response = await self._answer_question(user_prompt, context_data, options)
			elif interaction_type == 'translate':
				response = await self._translate_content(user_prompt, context_data, options)
			elif interaction_type == 'enhance':
				response = await self._enhance_content(user_prompt, context_data, options)
			elif interaction_type == 'generate':
				response = await self._generate_content(user_prompt, context_data, options)
			elif interaction_type == 'extract':
				response = await self._extract_information(user_prompt, context_data, options)
			elif interaction_type == 'compare':
				response = await self._compare_documents(user_prompt, context_data, options)
			elif interaction_type == 'analyze':
				response = await self._analyze_content(user_prompt, context_data, options)
			else:
				response = await self._generic_interaction(user_prompt, context_data, options)
			
			# Create interaction record
			interaction_record = DCMGenerativeAI(
				tenant_id=document.tenant_id,
				created_by=user_id,
				updated_by=user_id,
				document_id=document.id,
				user_id=user_id,
				interaction_type=interaction_type,
				user_prompt=user_prompt,
				context_documents=context_documents or [],
				rag_context=rag_context_id,
				genai_response=response['text'],
				response_sources=response.get('sources', []),
				confidence_score=response.get('confidence', 0.8),
				token_count=response.get('token_count', 0),
				processing_time_ms=int((time.time() - start_time) * 1000),
				model_version=response.get('model_version', 'apg-genai-v1')
			)
			
			# Update statistics
			self._update_genai_stats(interaction_record)
			
			self.logger.info(f"GenAI interaction completed: {interaction_type} for document {document.id}")
			return interaction_record
			
		except Exception as e:
			self.logger.error(f"GenAI interaction error: {str(e)}")
			
			# Create error record
			return DCMGenerativeAI(
				tenant_id=document.tenant_id,
				created_by=user_id,
				updated_by=user_id,
				document_id=document.id,
				user_id=user_id,
				interaction_type=interaction_type,
				user_prompt=user_prompt,
				context_documents=context_documents or [],
				genai_response=f"Error processing request: {str(e)}",
				response_sources=[],
				confidence_score=0.0,
				token_count=0,
				processing_time_ms=int((time.time() - start_time) * 1000),
				model_version="error"
			)
	
	async def _prepare_context(
		self,
		document: DCMDocument,
		context_documents: Optional[List[str]],
		requires_context: bool
	) -> Dict[str, Any]:
		"""Prepare context data for AI interaction"""
		context = {
			'primary_document': {
				'id': document.id,
				'title': document.title,
				'type': document.document_type.value,
				'content': await self._get_document_content(document.id)
			},
			'additional_documents': [],
			'metadata': {
				'language': document.language,
				'created_at': document.created_at.isoformat(),
				'keywords': document.keywords,
				'categories': document.categories
			}
		}
		
		# Add context documents if provided
		if context_documents and requires_context:
			for doc_id in context_documents:
				try:
					doc_content = await self._get_document_content(doc_id)
					context['additional_documents'].append({
						'id': doc_id,
						'content': doc_content
					})
				except Exception as e:
					self.logger.warning(f"Failed to load context document {doc_id}: {str(e)}")
		
		return context
	
	async def _get_document_content(self, document_id: str) -> str:
		"""Retrieve document content for AI processing"""
		# This would typically retrieve content from storage or database
		# For now, return placeholder
		return f"Document content for {document_id}"
	
	async def _create_rag_context(
		self,
		document_id: str,
		user_prompt: str,
		context_documents: Optional[List[str]]
	) -> Optional[str]:
		"""Create RAG context for enhanced AI responses"""
		if not self.apg_rag_client:
			return None
		
		try:
			rag_context = await self.apg_rag_client.create_context(
				primary_document=document_id,
				query=user_prompt,
				additional_documents=context_documents or [],
				context_window=2000
			)
			return rag_context.get('context_id')
		except Exception as e:
			self.logger.warning(f"Failed to create RAG context: {str(e)}")
			return None
	
	async def _summarize_content(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate content summary using AI"""
		summary_type = options.get('summary_type', 'executive')
		max_length = options.get('max_length', 300)
		
		if self.apg_genai_client:
			try:
				# Use APG GenAI for sophisticated summarization
				prompt = self._build_summarization_prompt(
					user_prompt,
					context_data,
					summary_type,
					max_length
				)
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=max_length + 50,
					temperature=0.3
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.8),
					'sources': self._extract_sources(context_data),
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI summarization failed: {str(e)}")
		
		# Fallback summarization
		return await self._fallback_summarization(context_data, max_length)
	
	def _build_summarization_prompt(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		summary_type: str,
		max_length: int
	) -> str:
		"""Build prompt for document summarization"""
		content = context_data['primary_document']['content']
		doc_title = context_data['primary_document']['title']
		
		base_prompt = f"""
		Summarize the following document titled "{doc_title}":
		
		{content[:3000]}  # Limit content to avoid token limits
		
		Requirements:
		- Summary type: {summary_type}
		- Maximum length: {max_length} words
		- Focus on: {user_prompt if user_prompt else 'key points and main findings'}
		- Maintain accuracy and objectivity
		
		Summary:
		"""
		
		return base_prompt.strip()
	
	async def _answer_question(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Answer questions about document content"""
		answer_style = options.get('answer_style', 'comprehensive')
		include_citations = options.get('include_citations', True)
		
		if self.apg_genai_client:
			try:
				prompt = self._build_qa_prompt(user_prompt, context_data, answer_style)
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=400,
					temperature=0.1
				)
				
				answer_text = response.get('text', '')
				if include_citations:
					answer_text += self._add_citations(context_data)
				
				return {
					'text': answer_text,
					'confidence': response.get('confidence', 0.7),
					'sources': self._extract_sources(context_data),
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI Q&A failed: {str(e)}")
		
		# Fallback Q&A
		return await self._fallback_qa(user_prompt, context_data)
	
	def _build_qa_prompt(
		self,
		question: str,
		context_data: Dict[str, Any],
		answer_style: str
	) -> str:
		"""Build prompt for question answering"""
		content = context_data['primary_document']['content']
		doc_title = context_data['primary_document']['title']
		
		prompt = f"""
		Based on the document "{doc_title}", answer the following question:
		
		Question: {question}
		
		Document content:
		{content[:3000]}
		
		Instructions:
		- Answer style: {answer_style}
		- Base your answer only on the provided document content
		- If the information is not in the document, say so clearly
		- Be accurate and cite specific parts of the document when possible
		
		Answer:
		"""
		
		return prompt.strip()
	
	async def _translate_content(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Translate document content to specified language"""
		target_language = options.get('target_language', 'es')
		source_language = options.get('source_language', 'auto')
		preserve_formatting = options.get('preserve_formatting', True)
		
		if target_language not in self.supported_languages:
			raise ValueError(f"Unsupported target language: {target_language}")
		
		if self.apg_genai_client:
			try:
				content = context_data['primary_document']['content']
				
				prompt = f"""
				Translate the following text from {source_language} to {self.supported_languages[target_language]}:
				
				{content[:2000]}
				
				Requirements:
				- Maintain professional tone
				- Preserve technical terminology when appropriate
				- Keep formatting: {preserve_formatting}
				- Ensure accuracy and cultural appropriateness
				
				Translation:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=2200,
					temperature=0.2
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.8),
					'sources': [{'type': 'translation', 'target_language': target_language}],
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI translation failed: {str(e)}")
		
		# Fallback translation (basic)
		return {
			'text': f'Translation to {self.supported_languages[target_language]} not available',
			'confidence': 0.0,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _enhance_content(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Enhance and improve content quality"""
		enhancement_type = options.get('enhancement_type', 'general')
		focus_areas = options.get('focus_areas', ['clarity', 'grammar', 'structure'])
		
		if self.apg_genai_client:
			try:
				content = context_data['primary_document']['content']
				
				prompt = f"""
				Enhance the following content to improve its quality:
				
				Original content:
				{content[:1500]}
				
				Enhancement requirements:
				- Type: {enhancement_type}
				- Focus on: {', '.join(focus_areas)}
				- Specific request: {user_prompt}
				- Maintain original meaning and intent
				- Improve readability and professional tone
				
				Enhanced content:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=1700,
					temperature=0.4
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.7),
					'sources': [{'type': 'enhancement', 'original_length': len(content)}],
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI enhancement failed: {str(e)}")
		
		# Fallback enhancement
		return {
			'text': 'Content enhancement not available',
			'confidence': 0.0,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _generate_content(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Generate new content based on templates and context"""
		template_type = options.get('template_type', 'general')
		content_length = options.get('content_length', 'medium')
		tone = options.get('tone', 'professional')
		
		if self.apg_genai_client:
			try:
				# Get template structure if available
				template_config = self.content_templates.get(template_type, {})
				structure = template_config.get('structure', [])
				
				context_info = context_data['primary_document']['content'][:1000]
				
				prompt = f"""
				Generate new content based on the following requirements:
				
				Request: {user_prompt}
				Template type: {template_type}
				Content length: {content_length}
				Tone: {tone}
				Structure: {', '.join(structure) if structure else 'flexible'}
				
				Reference context:
				{context_info}
				
				Requirements:
				- Create original, high-quality content
				- Follow the specified structure and tone
				- Incorporate relevant information from context
				- Ensure professional and accurate language
				
				Generated content:
				"""
				
				max_tokens = {'short': 500, 'medium': 1000, 'long': 2000}.get(content_length, 1000)
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=max_tokens,
					temperature=0.7
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.6),
					'sources': [{'type': 'generation', 'template': template_type}],
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI generation failed: {str(e)}")
		
		# Fallback generation
		return {
			'text': f'Generated content for: {user_prompt}',
			'confidence': 0.3,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _extract_information(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Extract specific information from document content"""
		extraction_type = options.get('extraction_type', 'general')
		output_format = options.get('output_format', 'text')
		
		if self.apg_genai_client:
			try:
				content = context_data['primary_document']['content']
				
				prompt = f"""
				Extract the following information from the document:
				
				Extraction request: {user_prompt}
				Extraction type: {extraction_type}
				Output format: {output_format}
				
				Document content:
				{content[:3000]}
				
				Instructions:
				- Extract only the requested information
				- Provide accurate quotes or data points
				- If information is not found, state clearly
				- Format output as requested: {output_format}
				
				Extracted information:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=600,
					temperature=0.1
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.8),
					'sources': self._extract_sources(context_data),
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI extraction failed: {str(e)}")
		
		# Fallback extraction
		return {
			'text': f'Information extraction for: {user_prompt}',
			'confidence': 0.4,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _compare_documents(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Compare multiple documents and highlight differences"""
		comparison_type = options.get('comparison_type', 'content')
		highlight_differences = options.get('highlight_differences', True)
		
		primary_doc = context_data['primary_document']
		additional_docs = context_data['additional_documents']
		
		if not additional_docs:
			return {
				'text': 'No additional documents provided for comparison',
				'confidence': 0.0,
				'sources': [],
				'token_count': 0,
				'model_version': 'error'
			}
		
		if self.apg_genai_client:
			try:
				doc_contents = [primary_doc['content'][:1500]]
				doc_contents.extend([doc['content'][:1500] for doc in additional_docs[:2]])  # Limit to 3 docs
				
				prompt = f"""
				Compare the following documents and provide analysis:
				
				Comparison focus: {user_prompt}
				Comparison type: {comparison_type}
				Highlight differences: {highlight_differences}
				
				Document 1:
				{doc_contents[0]}
				
				Document 2:
				{doc_contents[1] if len(doc_contents) > 1 else 'N/A'}
				
				Document 3:
				{doc_contents[2] if len(doc_contents) > 2 else 'N/A'}
				
				Provide a detailed comparison focusing on:
				- Key similarities and differences
				- Content structure variations
				- Important discrepancies
				- Recommendations or insights
				
				Comparison analysis:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=900,
					temperature=0.2
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.7),
					'sources': [{'type': 'comparison', 'documents_compared': len(doc_contents)}],
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI comparison failed: {str(e)}")
		
		# Fallback comparison
		return {
			'text': f'Document comparison for: {user_prompt}',
			'confidence': 0.3,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _analyze_content(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Analyze document content for insights and patterns"""
		analysis_type = options.get('analysis_type', 'comprehensive')
		focus_areas = options.get('focus_areas', ['themes', 'sentiment', 'key_points'])
		
		if self.apg_genai_client:
			try:
				content = context_data['primary_document']['content']
				
				prompt = f"""
				Analyze the following document content:
				
				Analysis request: {user_prompt}
				Analysis type: {analysis_type}
				Focus areas: {', '.join(focus_areas)}
				
				Document content:
				{content[:3500]}
				
				Provide comprehensive analysis including:
				- Main themes and topics
				- Sentiment and tone analysis
				- Key insights and findings
				- Patterns or trends identified
				- Actionable recommendations
				
				Analysis:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=700,
					temperature=0.3
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.7),
					'sources': self._extract_sources(context_data),
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI analysis failed: {str(e)}")
		
		# Fallback analysis
		return {
			'text': f'Content analysis for: {user_prompt}',
			'confidence': 0.4,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	async def _generic_interaction(
		self,
		user_prompt: str,
		context_data: Dict[str, Any],
		options: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Handle generic AI interactions"""
		if self.apg_genai_client:
			try:
				content = context_data['primary_document']['content']
				
				prompt = f"""
				Based on the following document, respond to the user request:
				
				User request: {user_prompt}
				
				Document content:
				{content[:2000]}
				
				Provide a helpful and accurate response based on the document content.
				
				Response:
				"""
				
				response = await self.apg_genai_client.generate_completion(
					prompt=prompt,
					max_tokens=500,
					temperature=0.5
				)
				
				return {
					'text': response.get('text', ''),
					'confidence': response.get('confidence', 0.6),
					'sources': self._extract_sources(context_data),
					'token_count': response.get('token_count', 0),
					'model_version': response.get('model_version', 'apg-genai-v1')
				}
			except Exception as e:
				self.logger.warning(f"APG GenAI generic interaction failed: {str(e)}")
		
		# Fallback response
		return {
			'text': f'Response to: {user_prompt}',
			'confidence': 0.3,
			'sources': [],
			'token_count': 0,
			'model_version': 'fallback'
		}
	
	def _extract_sources(self, context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Extract source information from context data"""
		sources = []
		
		# Primary document source
		primary_doc = context_data['primary_document']
		sources.append({
			'type': 'document',
			'id': primary_doc['id'],
			'title': primary_doc['title'],
			'relevance': 1.0
		})
		
		# Additional document sources
		for doc in context_data.get('additional_documents', []):
			sources.append({
				'type': 'context_document',
				'id': doc['id'],
				'title': doc.get('title', 'Untitled'),
				'relevance': 0.8
			})
		
		return sources
	
	def _add_citations(self, context_data: Dict[str, Any]) -> str:
		"""Add citations to response text"""
		primary_doc = context_data['primary_document']
		citation = f"\n\nSource: {primary_doc['title']} (Document ID: {primary_doc['id']})"
		
		additional_docs = context_data.get('additional_documents', [])
		if additional_docs:
			citation += "\nAdditional sources: "
			citation += ", ".join([doc.get('title', doc['id']) for doc in additional_docs])
		
		return citation
	
	async def _fallback_summarization(
		self,
		context_data: Dict[str, Any],
		max_length: int
	) -> Dict[str, Any]:
		"""Fallback summarization when AI services unavailable"""
		content = context_data['primary_document']['content']
		
		# Simple extractive summarization - take first few sentences
		sentences = content.split('.')
		summary_sentences = []
		word_count = 0
		
		for sentence in sentences[:10]:  # Consider first 10 sentences
			sentence_words = len(sentence.split())
			if word_count + sentence_words <= max_length:
				summary_sentences.append(sentence.strip())
				word_count += sentence_words
			else:
				break
		
		summary_text = '. '.join(summary_sentences)
		if summary_text and not summary_text.endswith('.'):
			summary_text += '.'
		
		return {
			'text': summary_text or 'Unable to generate summary',
			'confidence': 0.5,
			'sources': self._extract_sources(context_data),
			'token_count': word_count,
			'model_version': 'fallback'
		}
	
	async def _fallback_qa(
		self,
		question: str,
		context_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Fallback Q&A when AI services unavailable"""
		content = context_data['primary_document']['content'].lower()
		question_lower = question.lower()
		
		# Simple keyword matching
		question_words = set(question_lower.split())
		content_words = set(content.split())
		
		common_words = question_words.intersection(content_words)
		
		if common_words:
			# Find sentences containing question keywords
			sentences = context_data['primary_document']['content'].split('.')
			relevant_sentences = []
			
			for sentence in sentences:
				if any(word in sentence.lower() for word in common_words):
					relevant_sentences.append(sentence.strip())
					if len(relevant_sentences) >= 3:
						break
			
			answer = '. '.join(relevant_sentences)
			confidence = min(0.7, len(common_words) / len(question_words))
		else:
			answer = "I cannot find specific information to answer your question in the provided document."
			confidence = 0.0
		
		return {
			'text': answer,
			'confidence': confidence,
			'sources': self._extract_sources(context_data),
			'token_count': len(answer.split()),
			'model_version': 'fallback'
		}
	
	def _update_genai_stats(self, interaction_record: DCMGenerativeAI):
		"""Update generative AI performance statistics"""
		self.genai_stats['total_interactions'] += 1
		
		if interaction_record.confidence_score > 0.5:
			self.genai_stats['successful_responses'] += 1
		
		# Update average response time
		total_time = (
			self.genai_stats['average_response_time'] * 
			(self.genai_stats['total_interactions'] - 1) + 
			interaction_record.processing_time_ms
		)
		self.genai_stats['average_response_time'] = total_time / self.genai_stats['total_interactions']
		
		# Update token usage
		self.genai_stats['token_usage']['output_tokens'] += interaction_record.token_count
		
		# Update interaction type stats
		interaction_type = interaction_record.interaction_type
		if interaction_type not in self.genai_stats['interaction_types']:
			self.genai_stats['interaction_types'][interaction_type] = 0
		self.genai_stats['interaction_types'][interaction_type] += 1
	
	async def record_user_feedback(
		self,
		interaction_id: str,
		rating: int,
		feedback_text: Optional[str] = None
	):
		"""Record user feedback for continuous improvement"""
		try:
			if 1 <= rating <= 5:
				self.genai_stats['user_satisfaction_scores'].append(rating)
			
			self.logger.info(f"User feedback recorded: interaction={interaction_id}, rating={rating}")
		except Exception as e:
			self.logger.error(f"Error recording feedback: {str(e)}")
	
	async def get_genai_analytics(self) -> Dict[str, Any]:
		"""Get generative AI performance analytics"""
		stats = self.genai_stats
		
		return {
			"total_interactions": stats['total_interactions'],
			"success_rate": (
				stats['successful_responses'] / stats['total_interactions']
			) if stats['total_interactions'] > 0 else 0,
			"average_response_time_ms": stats['average_response_time'],
			"average_user_satisfaction": (
				sum(stats['user_satisfaction_scores']) / len(stats['user_satisfaction_scores'])
			) if stats['user_satisfaction_scores'] else 0,
			"token_usage": stats['token_usage'],
			"interaction_types": stats['interaction_types'],
			"supported_languages": len(self.supported_languages),
			"available_templates": len(self.content_templates)
		}
	
	async def get_supported_interactions(self) -> Dict[str, Any]:
		"""Get information about supported interaction types"""
		return {
			"interaction_types": self.interaction_types,
			"supported_languages": self.supported_languages,
			"content_templates": self.content_templates
		}