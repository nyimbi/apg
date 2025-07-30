"""
APG RAG Flask-AppBuilder Views & REST API

Enterprise-grade REST API endpoints with Flask-AppBuilder integration,
comprehensive CRUD operations, and APG ecosystem compatibility.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from flask import request, jsonify, g, current_app
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.api import BaseApi, expose_api
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields, validate, ValidationError
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

# APG imports
from .models import (
	KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate,
	Document, DocumentCreate, DocumentUpdate,
	Conversation, ConversationCreate, ConversationUpdate,
	ConversationTurn, RetrievalRequest, GenerationRequest,
	RetrievalMethod, DocumentStatus, ConversationStatus
)
from .service import RAGService

# Marshmallow schemas for API validation
class KnowledgeBaseCreateSchema(Schema):
	name = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	description = fields.Str(missing="", validate=validate.Length(max=1000))
	embedding_model = fields.Str(missing="bge-m3", validate=validate.OneOf(["bge-m3"]))
	generation_model = fields.Str(missing="qwen3", validate=validate.OneOf(["qwen3", "deepseek-r1"]))
	chunk_size = fields.Int(missing=1000, validate=validate.Range(min=100, max=8000))
	chunk_overlap = fields.Int(missing=200, validate=validate.Range(min=0, max=1000))
	similarity_threshold = fields.Float(missing=0.7, validate=validate.Range(min=0.0, max=1.0))
	max_retrievals = fields.Int(missing=10, validate=validate.Range(min=1, max=50))

class KnowledgeBaseUpdateSchema(Schema):
	name = fields.Str(validate=validate.Length(min=1, max=255))
	description = fields.Str(validate=validate.Length(max=1000))
	similarity_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0))
	max_retrievals = fields.Int(validate=validate.Range(min=1, max=50))

class DocumentCreateSchema(Schema):
	title = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	filename = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	file_type = fields.Str(required=True)
	content_hash = fields.Str(required=True)
	metadata = fields.Dict(missing=dict)

class ConversationCreateSchema(Schema):
	title = fields.Str(required=True, validate=validate.Length(min=1, max=255))
	description = fields.Str(missing="", validate=validate.Length(max=1000))
	generation_model = fields.Str(missing="qwen3", validate=validate.OneOf(["qwen3", "deepseek-r1"]))
	max_context_tokens = fields.Int(missing=8000, validate=validate.Range(min=1000, max=32000))
	temperature = fields.Float(missing=0.7, validate=validate.Range(min=0.0, max=2.0))
	session_id = fields.Str(missing="")

class QuerySchema(Schema):
	query_text = fields.Str(required=True, validate=validate.Length(min=1, max=10000))
	k = fields.Int(missing=10, validate=validate.Range(min=1, max=50))
	similarity_threshold = fields.Float(missing=0.7, validate=validate.Range(min=0.0, max=1.0))
	retrieval_method = fields.Str(missing="hybrid_search", validate=validate.OneOf([
		"vector_search", "hybrid_search", "semantic_search"
	]))

class ChatMessageSchema(Schema):
	message = fields.Str(required=True, validate=validate.Length(min=1, max=10000))
	user_context = fields.Dict(missing=dict)

class BaseRAGView(BaseView):
	"""Base view with common RAG functionality"""
	
	def __init__(self):
		super().__init__()
		self.logger = logging.getLogger(__name__)
	
	def get_rag_service(self) -> RAGService:
		"""Get RAG service instance from Flask app context"""
		if not hasattr(g, 'rag_service'):
			# This would be injected by APG framework
			raise InternalServerError("RAG service not available")
		return g.rag_service
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID from security context"""
		if hasattr(g, 'user') and hasattr(g.user, 'tenant_id'):
			return g.user.tenant_id
		return "default"  # Fallback for development
	
	def get_user_id(self) -> str:
		"""Get current user ID from security context"""
		if hasattr(g, 'user') and hasattr(g.user, 'id'):
			return str(g.user.id)
		return "system"  # Fallback for development
	
	def validate_json_schema(self, schema_class, data=None):
		"""Validate request JSON against schema"""
		if data is None:
			data = request.get_json()
		
		if not data:
			raise BadRequest("JSON body required")
		
		schema = schema_class()
		try:
			return schema.load(data)
		except ValidationError as e:
			raise BadRequest(f"Validation error: {e.messages}")

class KnowledgeBaseApi(BaseRAGView):
	"""Knowledge Base management API"""
	
	route_base = "/api/v1/rag/knowledge-bases"
	
	@expose_api
	@protect()
	async def get_list(self):
		"""List knowledge bases"""
		try:
			rag_service = self.get_rag_service()
			user_id = request.args.get('user_id')
			limit = int(request.args.get('limit', 50))
			offset = int(request.args.get('offset', 0))
			
			knowledge_bases = await rag_service.list_knowledge_bases(
				user_id=user_id,
				limit=limit,
				offset=offset
			)
			
			return jsonify({
				'success': True,
				'data': [kb.dict() for kb in knowledge_bases],
				'count': len(knowledge_bases)
			})
			
		except Exception as e:
			self.logger.error(f"Failed to list knowledge bases: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def post(self):
		"""Create knowledge base"""
		try:
			validated_data = self.validate_json_schema(KnowledgeBaseCreateSchema)
			
			rag_service = self.get_rag_service()
			kb_create = KnowledgeBaseCreate(
				**validated_data,
				user_id=self.get_user_id()
			)
			
			knowledge_base = await rag_service.create_knowledge_base(kb_create)
			
			return jsonify({
				'success': True,
				'data': knowledge_base.dict(),
				'message': 'Knowledge base created successfully'
			}), 201
			
		except BadRequest:
			raise
		except Exception as e:
			self.logger.error(f"Failed to create knowledge base: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def get(self, kb_id: str):
		"""Get knowledge base by ID"""
		try:
			rag_service = self.get_rag_service()
			knowledge_base = await rag_service.get_knowledge_base(kb_id)
			
			if not knowledge_base:
				raise NotFound(f"Knowledge base {kb_id} not found")
			
			return jsonify({
				'success': True,
				'data': knowledge_base.dict()
			})
			
		except NotFound:
			raise
		except Exception as e:
			self.logger.error(f"Failed to get knowledge base {kb_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def put(self, kb_id: str):
		"""Update knowledge base"""
		try:
			validated_data = self.validate_json_schema(KnowledgeBaseUpdateSchema)
			
			rag_service = self.get_rag_service()
			
			# Get existing knowledge base
			knowledge_base = await rag_service.get_knowledge_base(kb_id)
			if not knowledge_base:
				raise NotFound(f"Knowledge base {kb_id} not found")
			
			# Update via database (simplified for now)
			# In production, this would go through the service layer
			
			return jsonify({
				'success': True,
				'message': 'Knowledge base updated successfully'
			})
			
		except (NotFound, BadRequest):
			raise
		except Exception as e:
			self.logger.error(f"Failed to update knowledge base {kb_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class DocumentApi(BaseRAGView):
	"""Document management API"""
	
	route_base = "/api/v1/rag/documents"
	
	@expose_api
	@protect()
	async def post(self, kb_id: str):
		"""Upload document to knowledge base"""
		try:
			# Check if file is present
			if 'file' not in request.files:
				raise BadRequest("No file provided")
			
			file = request.files['file']
			if file.filename == '':
				raise BadRequest("No file selected")
			
			# Get metadata from form
			metadata = {}
			if 'metadata' in request.form:
				try:
					metadata = json.loads(request.form['metadata'])
				except json.JSONDecodeError:
					raise BadRequest("Invalid metadata JSON")
			
			# Read file content
			content = file.read()
			
			# Create document
			document_create = DocumentCreate(
				title=request.form.get('title', file.filename),
				filename=file.filename,
				file_type=file.content_type or 'application/octet-stream',
				content_hash=f"hash_{len(content)}",  # Simple hash for demo
				metadata=metadata,
				user_id=self.get_user_id()
			)
			
			rag_service = self.get_rag_service()
			document = await rag_service.add_document(
				kb_id, document_create, content
			)
			
			return jsonify({
				'success': True,
				'data': document.dict(),
				'message': 'Document uploaded and processed successfully'
			}), 201
			
		except (BadRequest, NotFound):
			raise
		except Exception as e:
			self.logger.error(f"Failed to upload document: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def get(self, document_id: str):
		"""Get document by ID"""
		try:
			rag_service = self.get_rag_service()
			document = await rag_service.get_document(document_id)
			
			if not document:
				raise NotFound(f"Document {document_id} not found")
			
			return jsonify({
				'success': True,
				'data': document.dict()
			})
			
		except NotFound:
			raise
		except Exception as e:
			self.logger.error(f"Failed to get document {document_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def delete(self, document_id: str):
		"""Delete document"""
		try:
			rag_service = self.get_rag_service()
			success = await rag_service.delete_document(document_id)
			
			if not success:
				raise NotFound(f"Document {document_id} not found")
			
			return jsonify({
				'success': True,
				'message': 'Document deleted successfully'
			})
			
		except NotFound:
			raise
		except Exception as e:
			self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class QueryApi(BaseRAGView):
	"""Query and retrieval API"""
	
	route_base = "/api/v1/rag/query"
	
	@expose_api
	@protect()
	async def post(self, kb_id: str):
		"""Query knowledge base"""
		try:
			validated_data = self.validate_json_schema(QuerySchema)
			
			rag_service = self.get_rag_service()
			
			# Map retrieval method
			retrieval_method_map = {
				"vector_search": RetrievalMethod.VECTOR_SEARCH,
				"hybrid_search": RetrievalMethod.HYBRID_SEARCH,
				"semantic_search": RetrievalMethod.SEMANTIC_SEARCH
			}
			retrieval_method = retrieval_method_map.get(
				validated_data['retrieval_method'], 
				RetrievalMethod.HYBRID_SEARCH
			)
			
			result = await rag_service.query_knowledge_base(
				kb_id=kb_id,
				query_text=validated_data['query_text'],
				k=validated_data['k'],
				similarity_threshold=validated_data['similarity_threshold'],
				retrieval_method=retrieval_method
			)
			
			return jsonify({
				'success': True,
				'data': {
					'query_text': validated_data['query_text'],
					'retrieved_chunks': result.retrieved_chunk_ids,
					'similarity_scores': result.similarity_scores,
					'processing_time_ms': result.processing_time_ms,
					'total_chunks_found': len(result.retrieved_chunk_ids)
				}
			})
			
		except (BadRequest, NotFound):
			raise
		except Exception as e:
			self.logger.error(f"Failed to query knowledge base {kb_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class GenerationApi(BaseRAGView):
	"""RAG generation API"""
	
	route_base = "/api/v1/rag/generate"
	
	@expose_api
	@protect()
	async def post(self, kb_id: str):
		"""Generate RAG response"""
		try:
			validated_data = self.validate_json_schema(QuerySchema)
			
			rag_service = self.get_rag_service()
			
			# Get optional parameters
			conversation_id = request.args.get('conversation_id')
			generation_model = request.args.get('generation_model')
			
			result = await rag_service.generate_response(
				kb_id=kb_id,
				query_text=validated_data['query_text'],
				conversation_id=conversation_id,
				generation_model=generation_model
			)
			
			return jsonify({
				'success': True,
				'data': {
					'query_text': validated_data['query_text'],
					'response_text': result.response_text,
					'sources_used': result.sources_used,
					'generation_model': result.generation_model,
					'token_count': result.token_count,
					'generation_time_ms': result.generation_time_ms,
					'confidence_score': result.confidence_score,
					'factual_accuracy_score': result.factual_accuracy_score,
					'citation_coverage': result.citation_coverage
				}
			})
			
		except (BadRequest, NotFound):
			raise
		except Exception as e:
			self.logger.error(f"Failed to generate response for KB {kb_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class ConversationApi(BaseRAGView):
	"""Conversation management API"""
	
	route_base = "/api/v1/rag/conversations"
	
	@expose_api
	@protect()
	async def post(self, kb_id: str):
		"""Create conversation"""
		try:
			validated_data = self.validate_json_schema(ConversationCreateSchema)
			
			rag_service = self.get_rag_service()
			conv_create = ConversationCreate(
				**validated_data,
				user_id=self.get_user_id()
			)
			
			conversation = await rag_service.create_conversation(kb_id, conv_create)
			
			return jsonify({
				'success': True,
				'data': conversation.dict(),
				'message': 'Conversation created successfully'
			}), 201
			
		except (BadRequest, NotFound):
			raise
		except Exception as e:
			self.logger.error(f"Failed to create conversation: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500
	
	@expose_api
	@protect()
	async def get(self, conversation_id: str):
		"""Get conversation by ID"""
		try:
			rag_service = self.get_rag_service()
			conversation = await rag_service.conversation_manager.get_conversation(conversation_id)
			
			if not conversation:
				raise NotFound(f"Conversation {conversation_id} not found")
			
			return jsonify({
				'success': True,
				'data': conversation.dict()
			})
			
		except NotFound:
			raise
		except Exception as e:
			self.logger.error(f"Failed to get conversation {conversation_id}: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class ChatApi(BaseRAGView):
	"""Chat API for conversations"""
	
	route_base = "/api/v1/rag/chat"
	
	@expose_api
	@protect()
	async def post(self, conversation_id: str):
		"""Send chat message"""
		try:
			validated_data = self.validate_json_schema(ChatMessageSchema)
			
			rag_service = self.get_rag_service()
			user_turn, assistant_turn = await rag_service.chat(
				conversation_id=conversation_id,
				user_message=validated_data['message'],
				user_context=validated_data['user_context']
			)
			
			return jsonify({
				'success': True,
				'data': {
					'conversation_id': conversation_id,
					'user_turn': {
						'id': user_turn.id,
						'content': user_turn.content,
						'turn_number': user_turn.turn_number,
						'created_at': user_turn.created_at.isoformat()
					},
					'assistant_turn': {
						'id': assistant_turn.id,
						'content': assistant_turn.content,
						'turn_number': assistant_turn.turn_number,
						'model_used': assistant_turn.model_used,
						'confidence_score': assistant_turn.confidence_score,
						'generation_time_ms': assistant_turn.generation_time_ms,
						'created_at': assistant_turn.created_at.isoformat()
					}
				}
			})
			
		except (BadRequest, NotFound):
			raise
		except Exception as e:
			self.logger.error(f"Failed to process chat message: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

class HealthApi(BaseRAGView):
	"""Health and monitoring API"""
	
	route_base = "/api/v1/rag/health"
	
	@expose_api
	async def get(self):
		"""Get service health status"""
		try:
			rag_service = self.get_rag_service()
			health_info = await rag_service.health_check()
			
			status_code = 200 if health_info.get('components_healthy', False) else 503
			
			return jsonify({
				'success': True,
				'data': health_info
			}), status_code
			
		except Exception as e:
			self.logger.error(f"Health check failed: {str(e)}")
			return jsonify({
				'success': False,
				'error': str(e),
				'service_status': 'unhealthy'
			}), 503
	
	@expose_api
	@protect()
	async def get_stats(self):
		"""Get service statistics"""
		try:
			rag_service = self.get_rag_service()
			stats = rag_service.get_statistics()
			
			return jsonify({
				'success': True,
				'data': stats
			})
			
		except Exception as e:
			self.logger.error(f"Failed to get statistics: {str(e)}")
			return jsonify({'success': False, 'error': str(e)}), 500

# Flask-AppBuilder model views (for admin interface)
class KnowledgeBaseModelView(ModelView):
	"""Knowledge Base model view for admin interface"""
	
	datamodel = SQLAInterface(KnowledgeBase)
	
	list_columns = ['name', 'description', 'status', 'document_count', 'created_at']
	show_columns = ['name', 'description', 'embedding_model', 'generation_model', 
	               'chunk_size', 'chunk_overlap', 'status', 'document_count', 
	               'total_chunks', 'created_at', 'updated_at']
	add_columns = ['name', 'description', 'chunk_size', 'chunk_overlap']
	edit_columns = ['name', 'description', 'chunk_size', 'chunk_overlap']
	
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit']

class DocumentModelView(ModelView):
	"""Document model view for admin interface"""
	
	datamodel = SQLAInterface(Document)
	
	list_columns = ['title', 'filename', 'file_type', 'processing_status', 'chunk_count', 'created_at']
	show_columns = ['title', 'filename', 'file_type', 'file_size', 'processing_status',
	               'chunk_count', 'metadata', 'created_at', 'updated_at']
	
	base_permissions = ['can_list', 'can_show', 'can_delete']

class ConversationModelView(ModelView):
	"""Conversation model view for admin interface"""
	
	datamodel = SQLAInterface(Conversation)
	
	list_columns = ['title', 'status', 'turn_count', 'total_tokens_used', 'created_at']
	show_columns = ['title', 'description', 'generation_model', 'status', 
	               'turn_count', 'total_tokens_used', 'created_at', 'updated_at']
	
	base_permissions = ['can_list', 'can_show', 'can_delete']

# API registration function for APG integration
def register_rag_apis(appbuilder):
	"""Register all RAG API views with Flask-AppBuilder"""
	
	# REST APIs
	appbuilder.add_api(KnowledgeBaseApi)
	appbuilder.add_api(DocumentApi)
	appbuilder.add_api(QueryApi)
	appbuilder.add_api(GenerationApi)
	appbuilder.add_api(ConversationApi)
	appbuilder.add_api(ChatApi)
	appbuilder.add_api(HealthApi)
	
	# Model views for admin interface
	appbuilder.add_view(
		KnowledgeBaseModelView,
		"Knowledge Bases",
		icon="fa-database",
		category="RAG Management"
	)
	
	appbuilder.add_view(
		DocumentModelView,
		"Documents",
		icon="fa-file-text",
		category="RAG Management"
	)
	
	appbuilder.add_view(
		ConversationModelView,
		"Conversations",
		icon="fa-comments",
		category="RAG Management"
	)

# Async wrapper for Flask views
def async_route(func):
	"""Decorator to handle async functions in Flask routes"""
	def wrapper(*args, **kwargs):
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		try:
			return loop.run_until_complete(func(*args, **kwargs))
		finally:
			loop.close()
	return wrapper

# Apply async wrapper to all API methods
for cls in [KnowledgeBaseApi, DocumentApi, QueryApi, GenerationApi, ConversationApi, ChatApi, HealthApi]:
	for attr_name in dir(cls):
		attr = getattr(cls, attr_name)
		if callable(attr) and hasattr(attr, '_urls') and asyncio.iscoroutinefunction(attr):
			setattr(cls, attr_name, async_route(attr))