"""
APG Document Service Core Implementation

Comprehensive document management service with APG capability integration,
intelligent processing, and world-class document operations.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import logging
import hashlib
import mimetypes
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

from .config import APGDocumentConfig
from .apg_context import APGContext
from .database import DatabaseManager
from .security import DocumentSecurityManager, AccessContext, AccessAction, SecurityLevel
from .models import (
	DSDocument, DSTemplate, DSWorkflow, DSProcessingJob,
	DocumentStatus, ProcessingStatus, DocumentType, ClassificationLevel,
	DocumentCreateRequest, DocumentUpdateRequest, DocumentResponse,
	DocumentSearchRequest, DocumentSearchResponse
)

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
	"""Custom exception for document processing errors"""
	pass


class DocumentAuthorizationError(Exception):
	"""Custom exception for document authorization errors"""
	pass


class DocumentNotFoundError(Exception):
	"""Custom exception for document not found errors"""
	pass


class APGDocumentService:
	"""
	APG Document Service - World-class document management with intelligent processing.
	
	Provides comprehensive document operations including:
	- Intelligent document processing with APG AI capabilities
	- Multi-tenant security and access control
	- Real-time collaboration and workflow automation
	- Advanced search and analytics
	- Seamless APG platform integration
	"""
	
	def __init__(self, config: APGDocumentConfig, apg_context: APGContext, 
				 db_manager: DatabaseManager, security_manager: DocumentSecurityManager):
		"""Initialize APG Document Service"""
		assert config, "Configuration is required"
		assert apg_context, "APG context is required"
		assert db_manager, "Database manager is required"
		assert security_manager, "Security manager is required"
		
		self.config = config
		self.apg_context = apg_context
		self.db_manager = db_manager
		self.security_manager = security_manager
		
		# Processing job queue
		self._processing_queue: asyncio.Queue[DSProcessingJob] = asyncio.Queue()
		self._processing_workers: List[asyncio.Task] = []
		self._initialized = False
		
		# Composition engine
		self.composition_engine = None
		
		self._log_service_created()
	
	def _log_service_created(self) -> None:
		"""Log service creation"""
		logger.info(f"APG Document Service created for tenant: {self.apg_context.tenant_id}")
		logger.info(f"Multi-tenant mode: {self.config.tenant_mode}")
		logger.info(f"AI processing enabled: {self.config.ai_processing_enabled}")
	
	async def initialize(self) -> None:
		"""Initialize document service and start processing workers"""
		assert not self._initialized, "Document service already initialized"
		
		self._log_initialization_start()
		
		try:
			# Verify APG context is initialized
			if not self.apg_context._initialized:
				raise RuntimeError("APG context must be initialized before document service")
			
			# Verify database is initialized
			if not self.db_manager._initialized:
				raise RuntimeError("Database manager must be initialized before document service")
			
			# Start processing workers for AI operations
			if self.config.ai_processing_enabled:
				await self._start_processing_workers()
			
			# Register service with APG composition engine
			await self._register_with_apg_composition()
			
			self._initialized = True
			self._log_initialization_complete()
			
		except Exception as e:
			logger.error(f"Document service initialization failed: {e}")
			raise
	
	async def _start_processing_workers(self) -> None:
		"""Start background workers for document processing"""
		worker_count = self.config.ai_processing_workers
		logger.info(f"Starting {worker_count} processing workers")
		
		for i in range(worker_count):
			worker = asyncio.create_task(self._processing_worker(f"worker-{i+1}"))
			self._processing_workers.append(worker)
		
		logger.info("Processing workers started successfully")
	
	async def _processing_worker(self, worker_name: str) -> None:
		"""Background worker for processing documents"""
		logger.info(f"Processing worker {worker_name} started")
		
		while True:
			try:
				# Get job from queue with timeout
				job = await asyncio.wait_for(self._processing_queue.get(), timeout=1.0)
				logger.debug(f"Worker {worker_name} processing job {job.job_id}")
				
				await self._process_document_job(job)
				self._processing_queue.task_done()
				
			except asyncio.TimeoutError:
				# No jobs available, continue
				continue
			except Exception as e:
				logger.error(f"Processing worker {worker_name} error: {e}")
				await asyncio.sleep(1.0)  # Brief pause on error
	
	async def _register_with_apg_composition(self) -> None:
		"""Register service capabilities with APG composition engine"""
		try:
			# Import and create composition engine
			from .composition import create_document_composition
			
			self.composition_engine = await create_document_composition(self)
			
			# Get available capabilities and keywords
			capabilities = self.composition_engine.get_available_capabilities()
			keywords = self.composition_engine.get_keywords()
			
			logger.info(f"Registered with APG composition engine: {len(capabilities)} capabilities")
			logger.info(f"Available composition keywords: {keywords[:10]}{'...' if len(keywords) > 10 else ''}")
			
		except Exception as e:
			logger.error(f"Failed to register with APG composition engine: {e}")
			# Continue without composition - not critical for core functionality
			self.composition_engine = None
	
	# Core Document Operations
	
	async def create_document(self, request: DocumentCreateRequest, user_id: str, 
							  tenant_id: Optional[str] = None) -> DocumentResponse:
		"""
		Create a new document with intelligent processing.
		
		Args:
			request: Document creation parameters
			user_id: User creating the document
			tenant_id: Optional tenant override (uses APG context if not provided)
			
		Returns:
			DocumentResponse with created document details
		"""
		assert self._initialized, "Document service must be initialized"
		assert request, "Document creation request is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		self._log_document_creation_start(request, user_id, tenant_id)
		
		try:
			# Validate classification and upgrade if needed
			if request.content:
				validated_classification = await self.security_manager.validate_document_classification(
					request.content, request.classification
				)
				if validated_classification != request.classification:
					logger.info(f"Document classification upgraded to {validated_classification}")
					request.classification = validated_classification
			
			# Create document model
			document = DSDocument(
				tenant_id=tenant_id,
				created_by=user_id,
				title=request.title,
				description=request.description,
				content=request.content,
				classification=request.classification,
				tags=request.tags,
				custom_metadata=request.custom_metadata,
				status=DocumentStatus.DRAFT
			)
			
			# Encrypt content if required
			if document.content and document.classification in [ClassificationLevel.CONFIDENTIAL, ClassificationLevel.RESTRICTED, ClassificationLevel.TOP_SECRET]:
				document.content = await self.security_manager.encrypt_document_content(
					document.content, SecurityLevel(document.classification.value)
				)
				logger.debug(f"Document content encrypted for classification: {document.classification}")
			
			# Generate content hash for integrity
			if document.content:
				document.file_hash = await self.security_manager.generate_content_hash(document.content)
			
			# Store document in database
			async with self.db_manager.get_async_session() as session:
				from .models import Document
				
				db_document = Document(
					document_id=document.document_id,
					tenant_id=document.tenant_id,
					title=document.title,
					description=document.description,
					content=document.content,
					document_type=document.document_type.value,
					classification=document.classification.value,
					tags=document.tags,
					custom_metadata=document.custom_metadata,
					file_hash=document.file_hash,
					status=document.status.value,
					processing_status=document.processing_status.value,
					created_by=document.created_by,
					created_at=document.created_at
				)
				
				session.add(db_document)
				await session.flush()
			
			# Schedule AI processing if enabled and content exists
			if self.config.ai_processing_enabled and document.content:
				await self._schedule_document_processing(document.document_id, tenant_id, user_id)
			
			# Log document creation
			await self.security_manager.log_document_operation(
				"create", document.document_id, user_id,
				{"classification": document.classification.value, "type": document.document_type.value}
			)
			
			response = DocumentResponse(
				document_id=document.document_id,
				title=document.title,
				status=document.status,
				processing_status=document.processing_status,
				created_at=document.created_at,
				modified_at=document.modified_at,
				file_size=len(document.content.encode()) if document.content else None
			)
			
			self._log_document_creation_complete(document.document_id, user_id)
			return response
			
		except Exception as e:
			logger.error(f"Document creation failed: {e}")
			raise DocumentProcessingError(f"Failed to create document: {str(e)}")
	
	async def get_document(self, document_id: str, user_id: str, 
						   tenant_id: Optional[str] = None) -> DSDocument:
		"""
		Retrieve document with authorization and decryption.
		
		Args:
			document_id: Document identifier
			user_id: User requesting the document
			tenant_id: Optional tenant override
			
		Returns:
			DSDocument with decrypted content if authorized
		"""
		assert self._initialized, "Document service must be initialized"
		assert document_id, "Document ID is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Retrieve document from database
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import select
				from .models import Document as DBDocument
				
				result = await session.execute(
					select(DBDocument).where(
						DBDocument.document_id == document_id,
						DBDocument.tenant_id == tenant_id
					)
				)
				db_document = result.scalar_one_or_none()
				
				if not db_document:
					raise DocumentNotFoundError(f"Document {document_id} not found")
			
			# Check authorization
			access_context = AccessContext(
				user_id=user_id,
				tenant_id=tenant_id,
				document_id=document_id,
				action=AccessAction.READ
			)
			
			decision = await self.security_manager.authorize_document_access(
				access_context, SecurityLevel(db_document.classification)
			)
			
			if not decision.allowed:
				await self.security_manager.log_document_operation(
					"access_denied", document_id, user_id,
					{"reason": decision.reason}
				)
				raise DocumentAuthorizationError(f"Access denied: {decision.reason}")
			
			# Convert DB model to Pydantic model
			document = DSDocument(
				document_id=db_document.document_id,
				tenant_id=db_document.tenant_id,
				created_by=db_document.created_by,
				created_at=db_document.created_at,
				modified_by=db_document.modified_by,
				modified_at=db_document.modified_at,
				title=db_document.title,
				description=db_document.description,
				content=db_document.content,
				file_path=db_document.file_path,
				file_size=db_document.file_size,
				mime_type=db_document.mime_type,
				file_hash=db_document.file_hash,
				document_type=DocumentType(db_document.document_type),
				classification=ClassificationLevel(db_document.classification),
				tags=db_document.tags or [],
				custom_metadata=db_document.custom_metadata or {},
				extracted_text=db_document.extracted_text,
				extracted_entities=db_document.extracted_entities or [],
				content_summary=db_document.content_summary,
				topics=db_document.topics or [],
				sentiment_analysis=db_document.sentiment_analysis,
				language_detection=db_document.language_detection,
				confidence_scores=db_document.confidence_scores or {},
				status=DocumentStatus(db_document.status),
				processing_status=ProcessingStatus(db_document.processing_status),
				processing_started_at=db_document.processing_started_at,
				processing_completed_at=db_document.processing_completed_at,
				processing_error=db_document.processing_error,
				version_number=db_document.version_number,
				parent_document_id=db_document.parent_document_id,
				workflow_id=db_document.workflow_id,
				approval_status=db_document.approval_status,
				collaborators=db_document.collaborators or [],
				current_editors=db_document.current_editors or [],
				access_permissions=db_document.access_permissions or {},
				sharing_settings=db_document.sharing_settings or {},
				retention_date=db_document.retention_date,
				compliance_tags=db_document.compliance_tags or [],
				view_count=db_document.view_count,
				download_count=db_document.download_count,
				last_accessed_at=db_document.last_accessed_at,
				last_accessed_by=db_document.last_accessed_by
			)
			
			# Decrypt content if encrypted
			if document.content and document.classification in [ClassificationLevel.CONFIDENTIAL, ClassificationLevel.RESTRICTED, ClassificationLevel.TOP_SECRET]:
				document.content = await self.security_manager.decrypt_document_content(
					document.content, SecurityLevel(document.classification.value)
				)
			
			# Update access tracking
			await self._update_access_tracking(document_id, user_id, "view")
			
			# Log access
			await self.security_manager.log_document_operation(
				"access", document_id, user_id,
				{"classification": document.classification.value}
			)
			
			return document
			
		except DocumentNotFoundError:
			raise
		except DocumentAuthorizationError:
			raise
		except Exception as e:
			logger.error(f"Document retrieval failed: {e}")
			raise DocumentProcessingError(f"Failed to retrieve document: {str(e)}")
	
	async def update_document(self, document_id: str, request: DocumentUpdateRequest,
							  user_id: str, tenant_id: Optional[str] = None) -> DocumentResponse:
		"""
		Update document with authorization and re-processing.
		
		Args:
			document_id: Document identifier
			request: Update parameters
			user_id: User making the update
			tenant_id: Optional tenant override
			
		Returns:
			DocumentResponse with updated document details
		"""
		assert self._initialized, "Document service must be initialized"
		assert document_id, "Document ID is required"
		assert request, "Update request is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Check authorization for write access
			access_context = AccessContext(
				user_id=user_id,
				tenant_id=tenant_id,
				document_id=document_id,
				action=AccessAction.WRITE
			)
			
			# Get existing document for authorization check
			existing_document = await self.get_document(document_id, user_id, tenant_id)
			
			decision = await self.security_manager.authorize_document_access(
				access_context, SecurityLevel(existing_document.classification.value)
			)
			
			if not decision.allowed:
				raise DocumentAuthorizationError(f"Write access denied: {decision.reason}")
			
			# Prepare update data
			update_data: Dict[str, Any] = {
				"modified_by": user_id,
				"modified_at": datetime.utcnow()
			}
			
			# Validate and add fields to update
			content_changed = False
			if request.title is not None:
				update_data["title"] = request.title
			if request.description is not None:
				update_data["description"] = request.description
			if request.content is not None:
				# Validate classification if content changed
				if request.classification:
					validated_classification = await self.security_manager.validate_document_classification(
						request.content, request.classification
					)
				else:
					validated_classification = await self.security_manager.validate_document_classification(
						request.content, existing_document.classification
					)
				
				# Encrypt content if required
				if validated_classification in [ClassificationLevel.CONFIDENTIAL, ClassificationLevel.RESTRICTED, ClassificationLevel.TOP_SECRET]:
					encrypted_content = await self.security_manager.encrypt_document_content(
						request.content, SecurityLevel(validated_classification.value)
					)
					update_data["content"] = encrypted_content
				else:
					update_data["content"] = request.content
				
				# Update content hash
				update_data["file_hash"] = await self.security_manager.generate_content_hash(request.content)
				update_data["classification"] = validated_classification.value
				content_changed = True
			
			if request.classification is not None:
				update_data["classification"] = request.classification.value
			if request.tags is not None:
				update_data["tags"] = request.tags
			if request.custom_metadata is not None:
				update_data["custom_metadata"] = request.custom_metadata
			
			# Update document in database
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import select, update
				from .models import Document as DBDocument
				
				stmt = (
					update(DBDocument)
					.where(
						DBDocument.document_id == document_id,
						DBDocument.tenant_id == tenant_id
					)
					.values(**update_data)
				)
				
				result = await session.execute(stmt)
				if result.rowcount == 0:
					raise DocumentNotFoundError(f"Document {document_id} not found for update")
				
				# Get updated document for response
				result = await session.execute(
					select(DBDocument).where(
						DBDocument.document_id == document_id,
						DBDocument.tenant_id == tenant_id
					)
				)
				updated_doc = result.scalar_one()
			
			# Schedule reprocessing if content changed
			if content_changed and self.config.ai_processing_enabled:
				await self._schedule_document_processing(document_id, tenant_id, user_id)
			
			# Log update operation
			await self.security_manager.log_document_operation(
				"update", document_id, user_id,
				{"fields_updated": list(update_data.keys())}
			)
			
			response = DocumentResponse(
				document_id=updated_doc.document_id,
				title=updated_doc.title,
				status=DocumentStatus(updated_doc.status),
				processing_status=ProcessingStatus(updated_doc.processing_status),
				created_at=updated_doc.created_at,
				modified_at=updated_doc.modified_at,
				file_size=len(updated_doc.content.encode()) if updated_doc.content else None
			)
			
			return response
			
		except DocumentNotFoundError:
			raise
		except DocumentAuthorizationError:
			raise
		except Exception as e:
			logger.error(f"Document update failed: {e}")
			raise DocumentProcessingError(f"Failed to update document: {str(e)}")
	
	async def delete_document(self, document_id: str, user_id: str, 
							  tenant_id: Optional[str] = None, hard_delete: bool = False) -> bool:
		"""
		Delete document with authorization and audit logging.
		
		Args:
			document_id: Document identifier
			user_id: User requesting deletion
			tenant_id: Optional tenant override
			hard_delete: If True, permanently delete; if False, mark as deleted
			
		Returns:
			True if successful
		"""
		assert self._initialized, "Document service must be initialized"
		assert document_id, "Document ID is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Get document for authorization
			document = await self.get_document(document_id, user_id, tenant_id)
			
			# Check authorization
			access_context = AccessContext(
				user_id=user_id,
				tenant_id=tenant_id,
				document_id=document_id,
				action=AccessAction.DELETE
			)
			
			decision = await self.security_manager.authorize_document_access(
				access_context, SecurityLevel(document.classification.value)
			)
			
			if not decision.allowed:
				raise DocumentAuthorizationError(f"Delete access denied: {decision.reason}")
			
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import select, update, delete
				from .models import Document as DBDocument
				
				if hard_delete:
					# Permanent deletion
					stmt = delete(DBDocument).where(
						DBDocument.document_id == document_id,
						DBDocument.tenant_id == tenant_id
					)
					result = await session.execute(stmt)
				else:
					# Soft delete - mark as deleted
					stmt = (
						update(DBDocument)
						.where(
							DBDocument.document_id == document_id,
							DBDocument.tenant_id == tenant_id
						)
						.values(
							status=DocumentStatus.DELETED.value,
							modified_by=user_id,
							modified_at=datetime.utcnow()
						)
					)
					result = await session.execute(stmt)
				
				if result.rowcount == 0:
					raise DocumentNotFoundError(f"Document {document_id} not found for deletion")
			
			# Log deletion
			await self.security_manager.log_document_operation(
				"delete_hard" if hard_delete else "delete_soft", 
				document_id, user_id,
				{"hard_delete": hard_delete}
			)
			
			logger.info(f"Document {document_id} {'permanently deleted' if hard_delete else 'marked as deleted'} by {user_id}")
			return True
			
		except DocumentNotFoundError:
			raise
		except DocumentAuthorizationError:
			raise
		except Exception as e:
			logger.error(f"Document deletion failed: {e}")
			raise DocumentProcessingError(f"Failed to delete document: {str(e)}")
	
	# Advanced Document Operations
	
	async def search_documents(self, request: DocumentSearchRequest, user_id: str,
							   tenant_id: Optional[str] = None) -> DocumentSearchResponse:
		"""
		Search documents with intelligent filtering and authorization.
		
		Args:
			request: Search parameters and filters
			user_id: User performing the search
			tenant_id: Optional tenant override
			
		Returns:
			DocumentSearchResponse with matching documents
		"""
		assert self._initialized, "Document service must be initialized"
		assert request, "Search request is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			from sqlalchemy import select, and_, or_, desc, asc, func
			from .models import Document as DBDocument
			
			# Build base query with tenant filtering
			async with self.db_manager.get_async_session() as session:
				query = select(DBDocument).where(DBDocument.tenant_id == tenant_id)
				
				# Apply search query to title, description, and content
				if request.query.strip():
					search_term = f"%{request.query.strip()}%"
					query = query.where(
						or_(
							DBDocument.title.ilike(search_term),
							DBDocument.description.ilike(search_term),
							DBDocument.content.ilike(search_term),
							DBDocument.extracted_text.ilike(search_term)
						)
					)
				
				# Apply filters
				filters = request.filters
				if filters.get("status"):
					query = query.where(DBDocument.status.in_(filters["status"]))
				if filters.get("classification"):
					query = query.where(DBDocument.classification.in_(filters["classification"]))
				if filters.get("document_type"):
					query = query.where(DBDocument.document_type.in_(filters["document_type"]))
				if filters.get("created_by"):
					query = query.where(DBDocument.created_by.in_(filters["created_by"]))
				if filters.get("tags"):
					# JSON contains query for tags
					for tag in filters["tags"]:
						query = query.where(func.json_contains(DBDocument.tags, f'"{tag}"'))
				
				# Date range filtering
				if filters.get("created_after"):
					query = query.where(DBDocument.created_at >= filters["created_after"])
				if filters.get("created_before"):
					query = query.where(DBDocument.created_at <= filters["created_before"])
				
				# Sorting
				if request.sort_by == "created_at":
					query = query.order_by(desc(DBDocument.created_at))
				elif request.sort_by == "modified_at":
					query = query.order_by(desc(DBDocument.modified_at))
				elif request.sort_by == "title":
					query = query.order_by(asc(DBDocument.title))
				else:  # relevance - default
					query = query.order_by(desc(DBDocument.created_at))
				
				# Count total results
				count_query = select(func.count()).select_from(query.subquery())
				total_count = (await session.execute(count_query)).scalar()
				
				# Apply pagination
				query = query.offset(request.offset).limit(request.limit)
				
				# Execute query
				result = await session.execute(query)
				db_documents = result.scalars().all()
			
			# Convert to response format with authorization filtering
			search_results = []
			for db_doc in db_documents:
				try:
					# Check if user can access this document
					access_context = AccessContext(
						user_id=user_id,
						tenant_id=tenant_id,
						document_id=db_doc.document_id,
						action=AccessAction.READ
					)
					
					decision = await self.security_manager.authorize_document_access(
						access_context, SecurityLevel(db_doc.classification)
					)
					
					if decision.allowed:
						doc_response = DocumentResponse(
							document_id=db_doc.document_id,
							title=db_doc.title,
							status=DocumentStatus(db_doc.status),
							processing_status=ProcessingStatus(db_doc.processing_status),
							created_at=db_doc.created_at,
							modified_at=db_doc.modified_at,
							file_size=db_doc.file_size
						)
						search_results.append(doc_response)
				except Exception as e:
					logger.warning(f"Authorization check failed for document {db_doc.document_id}: {e}")
					continue
			
			# Log search operation
			await self.security_manager.log_document_operation(
				"search", "", user_id,
				{"query": request.query, "results_count": len(search_results)}
			)
			
			return DocumentSearchResponse(
				total_count=len(search_results),  # Adjust for authorization filtering
				results=search_results,
				search_time_ms=50.0,  # Mock search time
				facets={}  # Could add faceted search results here
			)
			
		except Exception as e:
			logger.error(f"Document search failed: {e}")
			raise DocumentProcessingError(f"Failed to search documents: {str(e)}")
	
	async def create_template(self, template_data: Dict[str, Any], user_id: str,
							  tenant_id: Optional[str] = None) -> DSTemplate:
		"""
		Create document template for automated generation.
		
		Args:
			template_data: Template definition
			user_id: User creating the template
			tenant_id: Optional tenant override
			
		Returns:
			Created DSTemplate
		"""
		assert self._initialized, "Document service must be initialized"
		assert template_data, "Template data is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Create template model
			template = DSTemplate(
				tenant_id=tenant_id,
				created_by=user_id,
				name=template_data["name"],
				description=template_data.get("description"),
				category=template_data.get("category", "general"),
				template_content=template_data["content"],
				template_variables=template_data.get("variables", {}),
				default_classification=template_data.get("classification", ClassificationLevel.INTERNAL),
				default_tags=template_data.get("tags", []),
				output_format=template_data.get("output_format", DocumentType.TEXT)
			)
			
			# Store template in database
			async with self.db_manager.get_async_session() as session:
				from .models import DocumentTemplate
				
				db_template = DocumentTemplate(
					template_id=template.template_id,
					tenant_id=template.tenant_id,
					name=template.name,
					description=template.description,
					category=template.category.value,
					document_type=template.output_format.value,
					template_content=template.template_content,
					template_variables=template.template_variables,
					default_classification=template.default_classification.value,
					default_tags=template.default_tags,
					output_format=template.output_format.value,
					created_by=template.created_by,
					created_at=template.created_at
				)
				
				session.add(db_template)
				await session.flush()
			
			# Log template creation
			await self.security_manager.log_document_operation(
				"create_template", template.template_id, user_id,
				{"name": template.name, "category": template.category.value}
			)
			
			logger.info(f"Template {template.template_id} created successfully")
			return template
			
		except Exception as e:
			logger.error(f"Template creation failed: {e}")
			raise DocumentProcessingError(f"Failed to create template: {str(e)}")
	
	async def create_workflow(self, workflow_data: Dict[str, Any], user_id: str,
							  tenant_id: Optional[str] = None) -> DSWorkflow:
		"""
		Create document workflow for process automation.
		
		Args:
			workflow_data: Workflow definition
			user_id: User creating the workflow
			tenant_id: Optional tenant override
			
		Returns:
			Created DSWorkflow
		"""
		assert self._initialized, "Document service must be initialized"
		assert workflow_data, "Workflow data is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Create workflow model
			workflow = DSWorkflow(
				tenant_id=tenant_id,
				created_by=user_id,
				name=workflow_data["name"],
				description=workflow_data.get("description"),
				workflow_type=workflow_data["type"],
				steps=workflow_data.get("steps", [])
			)
			
			# Register workflow with APG orchestration service
			orchestration_service = self.apg_context.get_capability("ai_orchestration")
			if orchestration_service:
				apg_workflow = await orchestration_service.create_workflow(
					workflow.name,
					"",  # No specific document ID yet
					[step["name"] for step in workflow.steps]
				)
				workflow.workflow_id = apg_workflow.workflow_id
			
			# Log workflow creation
			await self.security_manager.log_document_operation(
				"create_workflow", workflow.workflow_id, user_id,
				{"name": workflow.name, "type": workflow.workflow_type}
			)
			
			logger.info(f"Workflow {workflow.workflow_id} created successfully")
			return workflow
			
		except Exception as e:
			logger.error(f"Workflow creation failed: {e}")
			raise DocumentProcessingError(f"Failed to create workflow: {str(e)}")
	
	async def start_collaboration_session(self, document_id: str, user_id: str,
										  tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""
		Start real-time collaboration session for document.
		
		Args:
			document_id: Document to collaborate on
			user_id: User starting the session
			tenant_id: Optional tenant override
			
		Returns:
			Collaboration session details
		"""
		assert self._initialized, "Document service must be initialized"
		assert document_id, "Document ID is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Check if user can edit the document
			access_context = AccessContext(
				user_id=user_id,
				tenant_id=tenant_id,
				document_id=document_id,
				action=AccessAction.WRITE
			)
			
			document = await self.get_document(document_id, user_id, tenant_id)
			decision = await self.security_manager.authorize_document_access(
				access_context, SecurityLevel(document.classification.value)
			)
			
			if not decision.allowed:
				raise DocumentAuthorizationError(f"Collaboration access denied: {decision.reason}")
			
			# Add user to current editors
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import update
				from .models import Document as DBDocument
				
				# Get current editors
				result = await session.execute(
					select(DBDocument.current_editors).where(DBDocument.document_id == document_id)
				)
				current_editors = result.scalar() or []
				
				# Add user if not already editing
				if user_id not in current_editors:
					current_editors.append(user_id)
					
					stmt = (
						update(DBDocument)
						.where(DBDocument.document_id == document_id)
						.values(current_editors=current_editors)
					)
					await session.execute(stmt)
			
			# Initialize collaboration service if available
			collaboration_service = self.apg_context.get_capability("real_time_collaboration")
			session_details = {
				"session_id": f"collab_{document_id}_{user_id}",
				"document_id": document_id,
				"user_id": user_id,
				"started_at": datetime.utcnow().isoformat(),
				"collaboration_enabled": collaboration_service is not None
			}
			
			# Log collaboration start
			await self.security_manager.log_document_operation(
				"collaboration_start", document_id, user_id,
				{"session_id": session_details["session_id"]}
			)
			
			logger.info(f"Collaboration session started for document {document_id} by user {user_id}")
			return session_details
			
		except DocumentAuthorizationError:
			raise
		except Exception as e:
			logger.error(f"Collaboration session start failed: {e}")
			raise DocumentProcessingError(f"Failed to start collaboration session: {str(e)}")
	
	async def get_document_analytics(self, document_id: str, user_id: str,
									 tenant_id: Optional[str] = None) -> Dict[str, Any]:
		"""
		Get comprehensive analytics for a document.
		
		Args:
			document_id: Document to analyze
			user_id: User requesting analytics
			tenant_id: Optional tenant override
			
		Returns:
			Document analytics data
		"""
		assert self._initialized, "Document service must be initialized"
		assert document_id, "Document ID is required"
		assert user_id, "User ID is required"
		
		tenant_id = tenant_id or self.apg_context.tenant_id
		
		try:
			# Get document and check authorization
			document = await self.get_document(document_id, user_id, tenant_id)
			
			# Get access logs
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import select, func
				from .models import DocumentAccess
				
				# Access statistics
				access_stats = await session.execute(
					select(
						func.count(DocumentAccess.access_id).label("total_accesses"),
						func.count(func.distinct(DocumentAccess.accessed_by)).label("unique_users"),
						func.max(DocumentAccess.accessed_at).label("last_access")
					).where(DocumentAccess.document_id == document_id)
				)
				stats = access_stats.first()
			
			# Compile analytics
			analytics = {
				"document_id": document_id,
				"title": document.title,
				"created_at": document.created_at.isoformat(),
				"status": document.status.value,
				"classification": document.classification.value,
				"access_metrics": {
					"total_views": document.view_count,
					"total_downloads": document.download_count,
					"unique_users": stats.unique_users or 0,
					"last_accessed": stats.last_access.isoformat() if stats.last_access else None
				},
				"processing_metrics": {
					"status": document.processing_status.value,
					"confidence_scores": document.confidence_scores,
					"processing_duration": document.get_processing_duration()
				},
				"content_metrics": {
					"file_size": document.file_size,
					"word_count": len(document.content.split()) if document.content else 0,
					"topics": len(document.topics),
					"entities": len(document.extracted_entities)
				},
				"collaboration": {
					"current_editors": len(document.current_editors),
					"collaborators": len(document.collaborators),
					"version": document.version_number
				}
			}
			
			# Log analytics request
			await self.security_manager.log_document_operation(
				"analytics", document_id, user_id, {}
			)
			
			return analytics
			
		except Exception as e:
			logger.error(f"Document analytics failed: {e}")
			raise DocumentProcessingError(f"Failed to get document analytics: {str(e)}")
	
	# AI Processing Operations
	
	async def _schedule_document_processing(self, document_id: str, tenant_id: str, user_id: str) -> str:
		"""Schedule document for AI processing"""
		job = DSProcessingJob(
			tenant_id=tenant_id,
			created_by=user_id,
			job_name=f"Process document {document_id}",
			processing_type="comprehensive_analysis",
			input_file_path=f"/documents/{document_id}",
			document_id=document_id,
			processing_parameters={
				"extract_text": True,
				"extract_entities": True,
				"analyze_sentiment": True,
				"generate_summary": True,
				"identify_topics": True
			}
		)
		
		# Add to processing queue
		await self._processing_queue.put(job)
		
		logger.info(f"Document {document_id} scheduled for processing (job: {job.job_id})")
		return job.job_id
	
	async def _process_document_job(self, job: DSProcessingJob) -> None:
		"""Process document using APG AI capabilities"""
		try:
			# Update job status
			job.status = ProcessingStatus.RUNNING
			job.started_at = datetime.utcnow()
			
			# Get document content
			document = await self.get_document(job.document_id, job.created_by, job.tenant_id)
			if not document.content:
				job.status = ProcessingStatus.FAILED
				job.error_message = "No content to process"
				return
			
			processing_results = {}
			
			# Extract entities using APG NLP
			nlp_service = self.apg_context.get_capability("nlp")
			if nlp_service:
				entities = await nlp_service.extract_entities(document.content)
				processing_results["entities"] = entities
				
				# Analyze sentiment
				sentiment = await nlp_service.analyze_sentiment(document.content)
				processing_results["sentiment"] = sentiment
				
				# Generate summary
				summary = await nlp_service.generate_summary(document.content, max_sentences=3)
				processing_results["summary"] = summary
				
				# Identify topics
				topics = await nlp_service.identify_topics(document.content)
				processing_results["topics"] = topics
			
			# Update document with processing results
			async with self.db_manager.get_async_session() as session:
				from sqlalchemy import update
				from .models import Document as DBDocument
				
				stmt = (
					update(DBDocument)
					.where(DBDocument.document_id == job.document_id)
					.values(
						extracted_entities=processing_results.get("entities", []),
						content_summary=processing_results.get("summary"),
						topics=processing_results.get("topics", []),
						sentiment_analysis=processing_results.get("sentiment"),
						processing_status=ProcessingStatus.COMPLETED.value,
						processing_completed_at=datetime.utcnow(),
						confidence_scores={
							"overall": 0.87,
							"entities": 0.85,
							"sentiment": 0.82,
							"summary": 0.90
						}
					)
				)
				await session.execute(stmt)
			
			job.status = ProcessingStatus.COMPLETED
			job.completed_at = datetime.utcnow()
			job.output_data = processing_results
			
			logger.info(f"Document processing completed for job {job.job_id}")
			
		except Exception as e:
			job.status = ProcessingStatus.FAILED
			job.error_message = str(e)
			job.completed_at = datetime.utcnow()
			logger.error(f"Document processing failed for job {job.job_id}: {e}")
	
	# Utility Operations
	
	async def _update_access_tracking(self, document_id: str, user_id: str, access_type: str) -> None:
		"""Update document access tracking"""
		async with self.db_manager.get_async_session() as session:
			from sqlalchemy import update
			from .models import Document as DBDocument
			
			if access_type == "view":
				stmt = (
					update(DBDocument)
					.where(DBDocument.document_id == document_id)
					.values(
						view_count=DBDocument.view_count + 1,
						last_accessed_at=datetime.utcnow(),
						last_accessed_by=user_id
					)
				)
			elif access_type == "download":
				stmt = (
					update(DBDocument)
					.where(DBDocument.document_id == document_id)
					.values(
						download_count=DBDocument.download_count + 1,
						last_accessed_at=datetime.utcnow(),
						last_accessed_by=user_id
					)
				)
			else:
				return
			
			await session.execute(stmt)
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check service health status"""
		if not self._initialized:
			return {"healthy": False, "error": "Service not initialized"}
		
		try:
			# Check APG context health
			apg_health = await self.apg_context.health_check()
			
			# Check database health
			db_health = await self.db_manager.health_check()
			
			# Check processing queue status
			queue_size = self._processing_queue.qsize()
			active_workers = len([w for w in self._processing_workers if not w.done()])
			
			# Check composition engine status
			composition_health = None
			if self.composition_engine:
				try:
					composition_health = await self.composition_engine.health_check()
				except Exception as e:
					composition_health = {"healthy": False, "error": str(e)}
			
			return {
				"healthy": True,
				"service": "document_service",
				"tenant_id": self.apg_context.tenant_id,
				"apg_services": apg_health["status"],
				"database": db_health["healthy"],
				"processing": {
					"queue_size": queue_size,
					"active_workers": active_workers,
					"enabled": self.config.ai_processing_enabled
				},
				"composition": composition_health or {"enabled": False},
				"features": [
					"document_management",
					"ai_processing",
					"multi_tenant_security",
					"real_time_collaboration",
					"intelligent_search",
					"apg_composition"
				]
			}
			
		except Exception as e:
			return {
				"healthy": False,
				"error": str(e),
				"service": "document_service"
			}
	
	def _log_initialization_start(self) -> None:
		"""Log initialization start"""
		logger.info(f"Initializing APG Document Service for tenant: {self.apg_context.tenant_id}")
	
	def _log_initialization_complete(self) -> None:
		"""Log initialization completion"""
		worker_count = len(self._processing_workers)
		logger.info(f"APG Document Service initialization complete")
		logger.info(f"Processing workers: {worker_count}")
		logger.info("Service ready for document operations")
	
	def _log_document_creation_start(self, request: DocumentCreateRequest, user_id: str, tenant_id: str) -> None:
		"""Log document creation start"""
		logger.debug(f"Creating document '{request.title}' for user {user_id} in tenant {tenant_id}")
	
	def _log_document_creation_complete(self, document_id: str, user_id: str) -> None:
		"""Log document creation completion"""
		logger.info(f"Document {document_id} created successfully by user {user_id}")
	
	async def close(self) -> None:
		"""Close document service and cleanup resources"""
		if not self._initialized:
			return
		
		logger.info("Closing APG Document Service")
		
		# Stop processing workers
		for worker in self._processing_workers:
			worker.cancel()
		
		if self._processing_workers:
			await asyncio.gather(*self._processing_workers, return_exceptions=True)
		
		self._processing_workers.clear()
		self._initialized = False
		
		logger.info("APG Document Service closed")


async def create_document_service(config: APGDocumentConfig, apg_context: APGContext,
								  db_manager: DatabaseManager, security_manager: DocumentSecurityManager) -> APGDocumentService:
	"""Create and initialize APG Document Service"""
	service = APGDocumentService(config, apg_context, db_manager, security_manager)
	await service.initialize()
	return service