"""
APG Document Service Composition Integration

Implements APG composition patterns and capability registration for seamless
integration with other APG capabilities and services.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass

from .service import APGDocumentService
from .models import DocumentCreateRequest, DocumentUpdateRequest, ClassificationLevel

logger = logging.getLogger(__name__)


@dataclass
class CompositionCapability:
	"""Represents a capability available for composition"""
	name: str
	description: str
	keywords: List[str]
	handler: Callable
	parameters: Dict[str, Any]
	return_type: str


@dataclass
class CompositionContext:
	"""Context for capability composition"""
	requesting_capability: str
	user_id: str
	tenant_id: str
	session_id: Optional[str] = None
	metadata: Dict[str, Any] = None
	
	def __post_init__(self):
		if self.metadata is None:
			self.metadata = {}


class APGDocumentComposition:
	"""
	APG Document Service Composition Engine
	
	Handles capability registration, keyword processing, and inter-capability
	communication following APG composition patterns.
	"""
	
	def __init__(self, document_service: APGDocumentService):
		assert document_service, "Document service is required"
		
		self.document_service = document_service
		self._capabilities: Dict[str, CompositionCapability] = {}
		self._keyword_map: Dict[str, str] = {}
		self._initialized = False
		
		logger.info("APG Document Composition engine created")
	
	async def initialize(self) -> None:
		"""Initialize composition engine and register capabilities"""
		assert not self._initialized, "Composition engine already initialized"
		
		logger.info("Initializing APG Document Composition engine")
		
		try:
			# Register core document capabilities
			await self._register_core_capabilities()
			
			# Register advanced capabilities
			await self._register_advanced_capabilities()
			
			# Register AI processing capabilities
			await self._register_ai_capabilities()
			
			# Register with APG platform composition engine
			await self._register_with_apg_platform()
			
			self._initialized = True
			logger.info(f"Document composition engine initialized with {len(self._capabilities)} capabilities")
			
		except Exception as e:
			logger.error(f"Composition engine initialization failed: {e}")
			raise
	
	async def _register_core_capabilities(self) -> None:
		"""Register core document management capabilities"""
		
		# Document Creation
		await self._register_capability(CompositionCapability(
			name="create_document",
			description="Create a new document with content and metadata",
			keywords=["create_document", "new_document", "make_document", "add_document"],
			handler=self._handle_create_document,
			parameters={
				"title": {"type": "string", "required": True, "description": "Document title"},
				"content": {"type": "string", "required": False, "description": "Document content"},
				"classification": {"type": "string", "required": False, "default": "internal", "description": "Security classification"},
				"tags": {"type": "array", "required": False, "description": "Document tags"},
				"metadata": {"type": "object", "required": False, "description": "Custom metadata"}
			},
			return_type="DocumentResponse"
		))
		
		# Document Retrieval
		await self._register_capability(CompositionCapability(
			name="get_document",
			description="Retrieve document by ID with authorization",
			keywords=["get_document", "retrieve_document", "fetch_document", "load_document"],
			handler=self._handle_get_document,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"}
			},
			return_type="DSDocument"
		))
		
		# Document Update
		await self._register_capability(CompositionCapability(
			name="update_document",
			description="Update existing document content and metadata",
			keywords=["update_document", "modify_document", "edit_document", "change_document"],
			handler=self._handle_update_document,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"},
				"title": {"type": "string", "required": False, "description": "New document title"},
				"content": {"type": "string", "required": False, "description": "New document content"},
				"tags": {"type": "array", "required": False, "description": "New document tags"}
			},
			return_type="DocumentResponse"
		))
		
		# Document Deletion
		await self._register_capability(CompositionCapability(
			name="delete_document",
			description="Delete document (soft or hard delete)",
			keywords=["delete_document", "remove_document", "destroy_document"],
			handler=self._handle_delete_document,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"},
				"hard_delete": {"type": "boolean", "required": False, "default": False, "description": "Permanent deletion"}
			},
			return_type="boolean"
		))
	
	async def _register_advanced_capabilities(self) -> None:
		"""Register advanced document operations"""
		
		# Document Search
		await self._register_capability(CompositionCapability(
			name="search_documents",
			description="Search documents with intelligent filtering",
			keywords=["search_documents", "find_documents", "query_documents", "lookup_documents"],
			handler=self._handle_search_documents,
			parameters={
				"query": {"type": "string", "required": True, "description": "Search query"},
				"filters": {"type": "object", "required": False, "description": "Search filters"},
				"limit": {"type": "integer", "required": False, "default": 20, "description": "Result limit"}
			},
			return_type="DocumentSearchResponse"
		))
		
		# Template Creation
		await self._register_capability(CompositionCapability(
			name="create_template",
			description="Create reusable document template",
			keywords=["create_template", "make_template", "new_template", "define_template"],
			handler=self._handle_create_template,
			parameters={
				"name": {"type": "string", "required": True, "description": "Template name"},
				"content": {"type": "string", "required": True, "description": "Template content with variables"},
				"variables": {"type": "object", "required": False, "description": "Template variables"}
			},
			return_type="DSTemplate"
		))
		
		# Analytics
		await self._register_capability(CompositionCapability(
			name="get_document_analytics",
			description="Get comprehensive document analytics and metrics",
			keywords=["analytics", "document_metrics", "document_stats", "document_analytics"],
			handler=self._handle_get_analytics,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"}
			},
			return_type="object"
		))
		
		# Collaboration
		await self._register_capability(CompositionCapability(
			name="start_collaboration",
			description="Start real-time collaboration session",
			keywords=["collaborate", "start_collaboration", "edit_together", "share_edit"],
			handler=self._handle_start_collaboration,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"}
			},
			return_type="object"
		))
	
	async def _register_ai_capabilities(self) -> None:
		"""Register AI-powered document processing capabilities"""
		
		# Document Processing
		await self._register_capability(CompositionCapability(
			name="process_document",
			description="Process document with AI for entity extraction, sentiment analysis, and summarization",
			keywords=["process_document", "analyze_document", "ai_process", "extract_insights"],
			handler=self._handle_process_document,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"}
			},
			return_type="object"
		))
		
		# Smart Classification
		await self._register_capability(CompositionCapability(
			name="classify_document",
			description="Automatically classify document based on content",
			keywords=["classify_document", "auto_classify", "smart_classification", "categorize_document"],
			handler=self._handle_classify_document,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"}
			},
			return_type="object"
		))
		
		# Content Extraction
		await self._register_capability(CompositionCapability(
			name="extract_content",
			description="Extract structured content from document using AI",
			keywords=["extract_content", "parse_document", "extract_data", "content_extraction"],
			handler=self._handle_extract_content,
			parameters={
				"document_id": {"type": "string", "required": True, "description": "Document identifier"},
				"extraction_type": {"type": "string", "required": False, "default": "all", "description": "Type of content to extract"}
			},
			return_type="object"
		))
	
	async def _register_capability(self, capability: CompositionCapability) -> None:
		"""Register a capability with the composition engine"""
		self._capabilities[capability.name] = capability
		
		# Map keywords to capability name
		for keyword in capability.keywords:
			if keyword in self._keyword_map:
				logger.warning(f"Keyword '{keyword}' already mapped, overriding")
			self._keyword_map[keyword] = capability.name
		
		logger.debug(f"Registered capability: {capability.name} with {len(capability.keywords)} keywords")
	
	async def _register_with_apg_platform(self) -> None:
		"""Register capabilities with APG platform composition engine"""
		# This would register our capabilities with the central APG composition engine
		# so other capabilities can discover and use our document services
		
		registration_data = {
			"service_name": "document_service",
			"service_version": "1.0.0",
			"capabilities": [
				{
					"name": cap.name,
					"description": cap.description,
					"keywords": cap.keywords,
					"parameters": cap.parameters,
					"return_type": cap.return_type
				}
				for cap in self._capabilities.values()
			],
			"tenant_id": self.document_service.apg_context.tenant_id
		}
		
		logger.info(f"Registered {len(self._capabilities)} capabilities with APG platform")
		logger.info(f"Available keywords: {list(self._keyword_map.keys())}")
	
	# Capability Handlers
	
	async def _handle_create_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document creation requests"""
		try:
			request = DocumentCreateRequest(
				title=kwargs["title"],
				description=kwargs.get("description"),
				content=kwargs.get("content"),
				classification=ClassificationLevel(kwargs.get("classification", "internal")),
				tags=kwargs.get("tags", []),
				custom_metadata=kwargs.get("metadata", {})
			)
			
			response = await self.document_service.create_document(
				request, context.user_id, context.tenant_id
			)
			
			logger.info(f"Document created via composition: {response.document_id}")
			return response
			
		except Exception as e:
			logger.error(f"Composition create_document failed: {e}")
			raise
	
	async def _handle_get_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document retrieval requests"""
		try:
			document_id = kwargs["document_id"]
			
			document = await self.document_service.get_document(
				document_id, context.user_id, context.tenant_id
			)
			
			logger.debug(f"Document retrieved via composition: {document_id}")
			return document
			
		except Exception as e:
			logger.error(f"Composition get_document failed: {e}")
			raise
	
	async def _handle_update_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document update requests"""
		try:
			document_id = kwargs["document_id"]
			
			request = DocumentUpdateRequest(
				title=kwargs.get("title"),
				description=kwargs.get("description"),
				content=kwargs.get("content"),
				tags=kwargs.get("tags"),
				custom_metadata=kwargs.get("metadata")
			)
			
			response = await self.document_service.update_document(
				document_id, request, context.user_id, context.tenant_id
			)
			
			logger.info(f"Document updated via composition: {document_id}")
			return response
			
		except Exception as e:
			logger.error(f"Composition update_document failed: {e}")
			raise
	
	async def _handle_delete_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document deletion requests"""
		try:
			document_id = kwargs["document_id"]
			hard_delete = kwargs.get("hard_delete", False)
			
			result = await self.document_service.delete_document(
				document_id, context.user_id, context.tenant_id, hard_delete
			)
			
			logger.info(f"Document deleted via composition: {document_id}")
			return result
			
		except Exception as e:
			logger.error(f"Composition delete_document failed: {e}")
			raise
	
	async def _handle_search_documents(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document search requests"""
		try:
			from .models import DocumentSearchRequest
			
			request = DocumentSearchRequest(
				query=kwargs["query"],
				filters=kwargs.get("filters", {}),
				limit=kwargs.get("limit", 20),
				offset=kwargs.get("offset", 0)
			)
			
			response = await self.document_service.search_documents(
				request, context.user_id, context.tenant_id
			)
			
			logger.info(f"Document search via composition: {len(response.results)} results")
			return response
			
		except Exception as e:
			logger.error(f"Composition search_documents failed: {e}")
			raise
	
	async def _handle_create_template(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle template creation requests"""
		try:
			template_data = {
				"name": kwargs["name"],
				"content": kwargs["content"],
				"variables": kwargs.get("variables", {}),
				"description": kwargs.get("description"),
				"category": kwargs.get("category", "general")
			}
			
			template = await self.document_service.create_template(
				template_data, context.user_id, context.tenant_id
			)
			
			logger.info(f"Template created via composition: {template.template_id}")
			return template
			
		except Exception as e:
			logger.error(f"Composition create_template failed: {e}")
			raise
	
	async def _handle_get_analytics(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle analytics requests"""
		try:
			document_id = kwargs["document_id"]
			
			analytics = await self.document_service.get_document_analytics(
				document_id, context.user_id, context.tenant_id
			)
			
			logger.debug(f"Analytics retrieved via composition: {document_id}")
			return analytics
			
		except Exception as e:
			logger.error(f"Composition get_analytics failed: {e}")
			raise
	
	async def _handle_start_collaboration(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle collaboration session start requests"""
		try:
			document_id = kwargs["document_id"]
			
			session_details = await self.document_service.start_collaboration_session(
				document_id, context.user_id, context.tenant_id
			)
			
			logger.info(f"Collaboration started via composition: {document_id}")
			return session_details
			
		except Exception as e:
			logger.error(f"Composition start_collaboration failed: {e}")
			raise
	
	async def _handle_process_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle AI document processing requests"""
		try:
			document_id = kwargs["document_id"]
			
			# Trigger processing by scheduling it
			job_id = await self.document_service._schedule_document_processing(
				document_id, context.tenant_id, context.user_id
			)
			
			logger.info(f"Document processing scheduled via composition: {document_id}")
			return {"job_id": job_id, "status": "scheduled"}
			
		except Exception as e:
			logger.error(f"Composition process_document failed: {e}")
			raise
	
	async def _handle_classify_document(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle document classification requests"""
		try:
			document_id = kwargs["document_id"]
			
			# Get document content
			document = await self.document_service.get_document(
				document_id, context.user_id, context.tenant_id
			)
			
			# Validate classification
			if document.content:
				validated_classification = await self.document_service.security_manager.validate_document_classification(
					document.content, document.classification
				)
				
				classification_result = {
					"document_id": document_id,
					"current_classification": document.classification.value,
					"recommended_classification": validated_classification.value,
					"confidence": 0.85,
					"needs_upgrade": validated_classification != document.classification
				}
				
				logger.info(f"Document classified via composition: {document_id}")
				return classification_result
			else:
				return {"error": "No content to classify"}
			
		except Exception as e:
			logger.error(f"Composition classify_document failed: {e}")
			raise
	
	async def _handle_extract_content(self, context: CompositionContext, **kwargs) -> Any:
		"""Handle content extraction requests"""
		try:
			document_id = kwargs["document_id"]
			extraction_type = kwargs.get("extraction_type", "all")
			
			# Get document
			document = await self.document_service.get_document(
				document_id, context.user_id, context.tenant_id
			)
			
			# Extract content based on type
			extraction_result = {
				"document_id": document_id,
				"extraction_type": extraction_type,
				"extracted_data": {}
			}
			
			if extraction_type in ["all", "entities"]:
				extraction_result["extracted_data"]["entities"] = document.extracted_entities
			if extraction_type in ["all", "topics"]:
				extraction_result["extracted_data"]["topics"] = document.topics
			if extraction_type in ["all", "sentiment"]:
				extraction_result["extracted_data"]["sentiment"] = document.sentiment_analysis
			if extraction_type in ["all", "summary"]:
				extraction_result["extracted_data"]["summary"] = document.content_summary
			
			logger.info(f"Content extracted via composition: {document_id}")
			return extraction_result
			
		except Exception as e:
			logger.error(f"Composition extract_content failed: {e}")
			raise
	
	# Public Interface
	
	async def execute_capability(self, capability_name: str, context: CompositionContext, **kwargs) -> Any:
		"""Execute a registered capability"""
		assert self._initialized, "Composition engine must be initialized"
		
		if capability_name not in self._capabilities:
			raise ValueError(f"Unknown capability: {capability_name}")
		
		capability = self._capabilities[capability_name]
		
		try:
			logger.debug(f"Executing capability: {capability_name} for user {context.user_id}")
			result = await capability.handler(context, **kwargs)
			return result
			
		except Exception as e:
			logger.error(f"Capability execution failed: {capability_name} - {e}")
			raise
	
	async def execute_keyword(self, keyword: str, context: CompositionContext, **kwargs) -> Any:
		"""Execute capability by keyword"""
		if keyword not in self._keyword_map:
			raise ValueError(f"Unknown keyword: {keyword}")
		
		capability_name = self._keyword_map[keyword]
		return await self.execute_capability(capability_name, context, **kwargs)
	
	def get_available_capabilities(self) -> Dict[str, CompositionCapability]:
		"""Get all registered capabilities"""
		return self._capabilities.copy()
	
	def get_capability_info(self, capability_name: str) -> Optional[CompositionCapability]:
		"""Get information about a specific capability"""
		return self._capabilities.get(capability_name)
	
	def get_keywords(self) -> List[str]:
		"""Get all available keywords"""
		return list(self._keyword_map.keys())
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check composition engine health"""
		return {
			"healthy": self._initialized,
			"capabilities_count": len(self._capabilities),
			"keywords_count": len(self._keyword_map),
			"service": "document_composition",
			"tenant_id": self.document_service.apg_context.tenant_id
		}


async def create_document_composition(document_service: APGDocumentService) -> APGDocumentComposition:
	"""Create and initialize document composition engine"""
	composition = APGDocumentComposition(document_service)
	await composition.initialize()
	return composition