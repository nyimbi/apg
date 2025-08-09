"""
APG Document Service Flask-AppBuilder Views

Comprehensive web interface for document management with APG integration,
security, and Flask-AppBuilder best practices.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from flask import request, jsonify, flash, current_app, g
from flask_appbuilder import BaseView, ModelView, expose, has_access, permission_name
from flask_appbuilder.api import BaseApi, safe, rison
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields, ValidationError
from sqlalchemy import desc, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from .models import (
	Document, DocumentTemplate, DocumentAccess, Metric, MetricSummary,
	DocumentCreateRequest, DocumentUpdateRequest, DocumentSearchRequest,
	ClassificationLevel, DocumentStatus, ProcessingStatus, DocumentType
)
from .service import APGDocumentService, create_document_service
from .database import create_database_manager
from .security import create_security_manager
from .apg_context import create_apg_context
from .config import APGDocumentConfig

logger = logging.getLogger(__name__)


# Marshmallow Schemas for API validation

class DocumentCreateSchema(Schema):
	"""Schema for document creation requests"""
	title = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)
	description = fields.Str(missing=None)
	content = fields.Str(missing=None)
	classification = fields.Str(missing="internal", validate=lambda x: x in [e.value for e in ClassificationLevel])
	tags = fields.List(fields.Str(), missing=[])
	template_id = fields.Str(missing=None)
	custom_metadata = fields.Dict(missing={})


class DocumentUpdateSchema(Schema):
	"""Schema for document update requests"""
	title = fields.Str(validate=lambda x: len(x.strip()) > 0 if x else True)
	description = fields.Str()
	content = fields.Str()
	classification = fields.Str(validate=lambda x: x in [e.value for e in ClassificationLevel] if x else True)
	tags = fields.List(fields.Str())
	custom_metadata = fields.Dict()


class DocumentSearchSchema(Schema):
	"""Schema for document search requests"""
	query = fields.Str(required=True)
	filters = fields.Dict(missing={})
	sort_by = fields.Str(missing="relevance", validate=lambda x: x in ["relevance", "created_at", "modified_at", "title"])
	limit = fields.Int(missing=20, validate=lambda x: 1 <= x <= 100)
	offset = fields.Int(missing=0, validate=lambda x: x >= 0)


class TemplateCreateSchema(Schema):
	"""Schema for template creation requests"""
	name = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)
	description = fields.Str(missing=None)
	content = fields.Str(required=True)
	category = fields.Str(missing="general")
	variables = fields.Dict(missing={})
	classification = fields.Str(missing="internal", validate=lambda x: x in [e.value for e in ClassificationLevel])
	tags = fields.List(fields.Str(), missing=[])


# API Classes

class DocumentServiceApi(BaseApi):
	"""RESTful API for APG Document Service"""
	
	def __init__(self):
		super().__init__()
		self._service: Optional[APGDocumentService] = None
	
	async def _get_service(self) -> APGDocumentService:
		"""Get or create document service instance"""
		if self._service is None:
			# Initialize APG document service
			config = APGDocumentConfig.from_environment()
			apg_context = await create_apg_context(config, self._get_current_tenant())
			db_manager = await create_database_manager(config)
			security_manager = await create_security_manager(apg_context, config)
			
			self._service = await create_document_service(config, apg_context, db_manager, security_manager)
		
		return self._service
	
	def _get_current_user(self) -> str:
		"""Get current user ID"""
		return g.user.username if hasattr(g, 'user') and g.user else 'anonymous'
	
	def _get_current_tenant(self) -> str:
		"""Get current tenant ID"""
		return getattr(g.user, 'tenant_id', 'default') if hasattr(g, 'user') and g.user else 'default'
	
	@expose('/documents', methods=['POST'])
	@protect()
	@safe
	def create_document(self):
		"""Create a new document"""
		try:
			# Validate request
			schema = DocumentCreateSchema()
			try:
				data = schema.load(request.json or {})
			except ValidationError as e:
				return self.response_400(message="Invalid request data", errors=e.messages)
			
			# Create document request
			doc_request = DocumentCreateRequest(
				title=data['title'],
				description=data.get('description'),
				content=data.get('content'),
				classification=ClassificationLevel(data['classification']),
				tags=data['tags'],
				template_id=data.get('template_id'),
				custom_metadata=data['custom_metadata']
			)
			
			# Get service and create document
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				response = loop.run_until_complete(
					service.create_document(doc_request, self._get_current_user())
				)
				
				return self.response(201, document_id=response.document_id, 
								   title=response.title, status=response.status.value,
								   created_at=response.created_at.isoformat())
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Document creation failed: {e}")
			return self.response_500(message=str(e))
	
	@expose('/documents/<document_id>', methods=['GET'])
	@protect()
	@safe
	def get_document(self, document_id: str):
		"""Get document by ID"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				document = loop.run_until_complete(
					service.get_document(document_id, self._get_current_user())
				)
				
				# Convert to dict for JSON response
				doc_data = {
					"document_id": document.document_id,
					"title": document.title,
					"description": document.description,
					"content": document.content,
					"classification": document.classification.value,
					"status": document.status.value,
					"processing_status": document.processing_status.value,
					"created_at": document.created_at.isoformat(),
					"modified_at": document.modified_at.isoformat() if document.modified_at else None,
					"tags": document.tags,
					"custom_metadata": document.custom_metadata,
					"file_size": document.file_size,
					"view_count": document.view_count,
					"download_count": document.download_count
				}
				
				return self.response(200, **doc_data)
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Get document failed: {e}")
			if "not found" in str(e).lower():
				return self.response_404(message="Document not found")
			elif "access denied" in str(e).lower():
				return self.response_403(message="Access denied")
			return self.response_500(message=str(e))
	
	@expose('/documents/<document_id>', methods=['PUT'])
	@protect()
	@safe
	def update_document(self, document_id: str):
		"""Update document"""
		try:
			# Validate request
			schema = DocumentUpdateSchema()
			try:
				data = schema.load(request.json or {})
			except ValidationError as e:
				return self.response_400(message="Invalid request data", errors=e.messages)
			
			# Create update request
			update_request = DocumentUpdateRequest(
				title=data.get('title'),
				description=data.get('description'),
				content=data.get('content'),
				classification=ClassificationLevel(data['classification']) if data.get('classification') else None,
				tags=data.get('tags'),
				custom_metadata=data.get('custom_metadata')
			)
			
			# Update document
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				response = loop.run_until_complete(
					service.update_document(document_id, update_request, self._get_current_user())
				)
				
				return self.response(200, document_id=response.document_id,
								   title=response.title, status=response.status.value,
								   modified_at=response.modified_at.isoformat() if response.modified_at else None)
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Document update failed: {e}")
			if "not found" in str(e).lower():
				return self.response_404(message="Document not found")
			elif "access denied" in str(e).lower():
				return self.response_403(message="Access denied")
			return self.response_500(message=str(e))
	
	@expose('/documents/<document_id>', methods=['DELETE'])
	@protect()
	@safe
	def delete_document(self, document_id: str):
		"""Delete document"""
		try:
			hard_delete = request.args.get('hard', 'false').lower() == 'true'
			
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				success = loop.run_until_complete(
					service.delete_document(document_id, self._get_current_user(), hard_delete=hard_delete)
				)
				
				if success:
					return self.response(200, message="Document deleted successfully", hard_delete=hard_delete)
				else:
					return self.response_500(message="Delete operation failed")
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Document deletion failed: {e}")
			if "not found" in str(e).lower():
				return self.response_404(message="Document not found")
			elif "access denied" in str(e).lower():
				return self.response_403(message="Access denied")
			return self.response_500(message=str(e))
	
	@expose('/documents/search', methods=['POST'])
	@protect()
	@safe
	def search_documents(self):
		"""Search documents"""
		try:
			# Validate request
			schema = DocumentSearchSchema()
			try:
				data = schema.load(request.json or {})
			except ValidationError as e:
				return self.response_400(message="Invalid request data", errors=e.messages)
			
			# Create search request
			search_request = DocumentSearchRequest(
				query=data['query'],
				filters=data['filters'],
				sort_by=data['sort_by'],
				limit=data['limit'],
				offset=data['offset']
			)
			
			# Search documents
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				response = loop.run_until_complete(
					service.search_documents(search_request, self._get_current_user())
				)
				
				# Convert results to dict
				results = []
				for doc in response.results:
					results.append({
						"document_id": doc.document_id,
						"title": doc.title,
						"status": doc.status.value,
						"processing_status": doc.processing_status.value,
						"created_at": doc.created_at.isoformat(),
						"modified_at": doc.modified_at.isoformat() if doc.modified_at else None,
						"file_size": doc.file_size
					})
				
				return self.response(200, 
								   total_count=response.total_count,
								   results=results,
								   search_time_ms=response.search_time_ms,
								   facets=response.facets)
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Document search failed: {e}")
			return self.response_500(message=str(e))
	
	@expose('/templates', methods=['POST'])
	@protect()
	@safe
	def create_template(self):
		"""Create document template"""
		try:
			# Validate request
			schema = TemplateCreateSchema()
			try:
				data = schema.load(request.json or {})
			except ValidationError as e:
				return self.response_400(message="Invalid request data", errors=e.messages)
			
			# Create template
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				template = loop.run_until_complete(
					service.create_template(data, self._get_current_user())
				)
				
				return self.response(201, 
								   template_id=template.template_id,
								   name=template.name,
								   category=template.category.value,
								   created_at=template.created_at.isoformat())
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Template creation failed: {e}")
			return self.response_500(message=str(e))
	
	@expose('/documents/<document_id>/analytics')
	@protect()
	@safe
	def get_document_analytics(self, document_id: str):
		"""Get document analytics"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				analytics = loop.run_until_complete(
					service.get_document_analytics(document_id, self._get_current_user())
				)
				
				return self.response(200, **analytics)
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Document analytics failed: {e}")
			if "not found" in str(e).lower():
				return self.response_404(message="Document not found")
			elif "access denied" in str(e).lower():
				return self.response_403(message="Access denied")
			return self.response_500(message=str(e))
	
	@expose('/documents/<document_id>/collaboration', methods=['POST'])
	@protect()
	@safe
	def start_collaboration(self, document_id: str):
		"""Start collaboration session"""
		try:
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			
			try:
				service = loop.run_until_complete(self._get_service())
				session_details = loop.run_until_complete(
					service.start_collaboration_session(document_id, self._get_current_user())
				)
				
				return self.response(201, **session_details)
			finally:
				loop.close()
				
		except Exception as e:
			logger.error(f"Collaboration start failed: {e}")
			if "not found" in str(e).lower():
				return self.response_404(message="Document not found")
			elif "access denied" in str(e).lower():
				return self.response_403(message="Access denied")
			return self.response_500(message=str(e))
	
	@expose('/health')
	def health_check(self):
		"""Service health check"""
		try:
			if self._service:
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				try:
					health = loop.run_until_complete(self._service.health_check())
					return self.response(200, **health)
				finally:
					loop.close()
			else:
				return self.response(200, healthy=False, message="Service not initialized")
				
		except Exception as e:
			logger.error(f"Health check failed: {e}")
			return self.response_500(message=str(e))


# Web UI Views

class DocumentModelView(ModelView):
	"""Flask-AppBuilder ModelView for Documents"""
	
	datamodel = SQLAInterface(Document)
	
	# Column configuration
	list_columns = ['title', 'document_type', 'classification', 'status', 
					'processing_status', 'created_by', 'created_at', 'view_count']
	show_columns = ['document_id', 'title', 'description', 'document_type', 
					'classification', 'status', 'processing_status', 'created_by', 
					'created_at', 'modified_by', 'modified_at', 'view_count', 
					'download_count', 'tags', 'custom_metadata']
	search_columns = ['title', 'description', 'content', 'created_by']
	edit_columns = ['title', 'description', 'classification', 'tags']
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Formatting
	formatters_columns = {
		'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else '',
		'modified_at': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else '',
		'classification': lambda x: x.title() if x else '',
		'status': lambda x: x.title() if x else '',
	}
	
	# Labels
	label_columns = {
		'document_id': 'Document ID',
		'document_type': 'Type',
		'classification': 'Security Level',
		'processing_status': 'Processing',
		'created_by': 'Created By',
		'created_at': 'Created',
		'modified_by': 'Modified By',
		'modified_at': 'Modified',
		'view_count': 'Views',
		'download_count': 'Downloads',
		'custom_metadata': 'Metadata'
	}
	
	# Default ordering
	base_order = ('created_at', 'desc')


class DocumentTemplateModelView(ModelView):
	"""Flask-AppBuilder ModelView for Document Templates"""
	
	datamodel = SQLAInterface(DocumentTemplate)
	
	# Column configuration
	list_columns = ['name', 'category', 'document_type', 'is_active', 
					'usage_count', 'created_by', 'created_at']
	show_columns = ['template_id', 'name', 'description', 'category', 
					'document_type', 'template_content', 'template_variables',
					'default_classification', 'default_tags', 'is_active',
					'usage_count', 'created_by', 'created_at']
	search_columns = ['name', 'description', 'created_by']
	edit_columns = ['name', 'description', 'category', 'template_content', 
					'template_variables', 'default_classification', 'default_tags', 'is_active']
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']
	
	# Formatting
	formatters_columns = {
		'created_at': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else '',
		'is_active': lambda x: 'Active' if x else 'Inactive',
		'template_content': lambda x: x[:100] + '...' if x and len(x) > 100 else x,
	}
	
	# Labels
	label_columns = {
		'template_id': 'Template ID',
		'template_content': 'Content',
		'template_variables': 'Variables',
		'default_classification': 'Default Security Level',
		'default_tags': 'Default Tags',
		'is_active': 'Active',
		'usage_count': 'Usage Count',
		'created_by': 'Created By',
		'created_at': 'Created'
	}


class DocumentServiceView(BaseView):
	"""Main dashboard view for Document Service"""
	
	default_view = 'dashboard'
	
	@expose('/dashboard/')
	@has_access
	@permission_name('read')
	def dashboard(self):
		"""Document service dashboard"""
		try:
			# Get basic statistics
			session = self.appbuilder.get_session
			
			# Document statistics
			total_documents = session.query(Document).count()
			documents_today = session.query(Document).filter(
				func.date(Document.created_at) == func.date(func.now())
			).count()
			
			# Recent documents
			recent_documents = session.query(Document).order_by(
				desc(Document.created_at)
			).limit(10).all()
			
			# Document type distribution
			type_stats = session.query(
				Document.document_type, 
				func.count(Document.document_id).label('count')
			).group_by(Document.document_type).all()
			
			# Classification distribution  
			classification_stats = session.query(
				Document.classification,
				func.count(Document.document_id).label('count')
			).group_by(Document.classification).all()
			
			stats = {
				'total_documents': total_documents,
				'documents_today': documents_today,
				'type_distribution': dict(type_stats),
				'classification_distribution': dict(classification_stats)
			}
			
			return self.render_template(
				'document_service/dashboard.html',
				stats=stats,
				recent_documents=recent_documents
			)
			
		except Exception as e:
			logger.error(f"Dashboard error: {e}")
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('document_service/error.html')
	
	@expose('/metrics/')
	@has_access
	@permission_name('read')
	def metrics_view(self):
		"""Metrics dashboard view"""
		try:
			# Get metrics from database
			session = self.appbuilder.get_session
			
			# Recent metrics
			recent_metrics = session.query(Metric).order_by(
				desc(Metric.timestamp)
			).limit(100).all()
			
			# Metric summaries
			metric_summaries = session.query(MetricSummary).order_by(
				desc(MetricSummary.created_at)
			).limit(20).all()
			
			return self.render_template(
				'document_service/metrics.html',
				recent_metrics=recent_metrics,
				metric_summaries=metric_summaries
			)
			
		except Exception as e:
			logger.error(f"Metrics view error: {e}")
			flash(f'Error loading metrics: {str(e)}', 'error')
			return self.render_template('document_service/error.html')
	
	@expose('/analytics/')
	@has_access  
	@permission_name('read')
	def analytics_view(self):
		"""Analytics dashboard view"""
		try:
			session = self.appbuilder.get_session
			
			# Usage analytics
			total_views = session.query(func.sum(Document.view_count)).scalar() or 0
			total_downloads = session.query(func.sum(Document.download_count)).scalar() or 0
			
			# Access logs
			recent_access = session.query(DocumentAccess).order_by(
				desc(DocumentAccess.accessed_at)
			).limit(50).all()
			
			# Top accessed documents
			top_documents = session.query(Document).order_by(
				desc(Document.view_count)
			).limit(10).all()
			
			analytics = {
				'total_views': total_views,
				'total_downloads': total_downloads,
				'recent_access': recent_access,
				'top_documents': top_documents
			}
			
			return self.render_template(
				'document_service/analytics.html',
				analytics=analytics
			)
			
		except Exception as e:
			logger.error(f"Analytics view error: {e}")
			flash(f'Error loading analytics: {str(e)}', 'error')
			return self.render_template('document_service/error.html')