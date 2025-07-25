"""
Document Management Service

Comprehensive business logic for document management operations including
document lifecycle, version control, security, workflows, and retention management.
"""

import os
import hashlib
import shutil
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, date, timedelta
from decimal import Decimal
import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import IntegrityError
from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from .models import (
	GCDMDocument, GCDMDocumentVersion, GCDMDocumentCategory, GCDMDocumentType,
	GCDMFolder, GCDMPermission, GCDMCheckout, GCDMWorkflow, GCDMReview,
	GCDMTag, GCDMDocumentTag, GCDMMetadata, GCDMRetentionPolicy, 
	GCDMArchive, GCDMAuditLog, DocumentStatus, DocumentType, 
	PermissionLevel, WorkflowStatus, ReviewStatus
)

logger = logging.getLogger(__name__)

class DocumentPermissionError(Exception):
	"""Raised when user lacks permission for document operation."""
	pass

class DocumentCheckoutError(Exception):
	"""Raised when document checkout/checkin operation fails."""
	pass

class DocumentVersionError(Exception):
	"""Raised when document version operation fails."""
	pass

class DocumentRetentionError(Exception):
	"""Raised when document retention operation fails."""
	pass

class DocumentService:
	"""
	Comprehensive document management service providing all business logic
	for document operations, security, workflows, and compliance.
	"""
	
	def __init__(self, db_session: Session):
		"""Initialize document service."""
		self.db = db_session
		self.storage_root = current_app.config.get('DOCUMENT_STORAGE_ROOT', '/var/apg/documents')
		self.max_file_size = current_app.config.get('MAX_DOCUMENT_SIZE_MB', 100) * 1024 * 1024
		self.allowed_extensions = current_app.config.get('ALLOWED_DOCUMENT_EXTENSIONS', [
			'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'rtf'
		])
		
		# Ensure storage directory exists
		Path(self.storage_root).mkdir(parents=True, exist_ok=True)
		
		logger.info("DocumentService initialized")
	
	# Document CRUD Operations
	
	def create_document(self, 
					   tenant_id: str,
					   title: str,
					   file_data: FileStorage,
					   category_id: str,
					   document_type_id: str,
					   folder_id: str,
					   owner_user_id: str,
					   description: Optional[str] = None,
					   metadata: Optional[Dict[str, Any]] = None,
					   tags: Optional[List[str]] = None) -> GCDMDocument:
		"""Create a new document."""
		try:
			# Validate file
			self._validate_file(file_data)
			
			# Generate document number
			document_number = self._generate_document_number(tenant_id)
			
			# Calculate file hash
			file_hash = self._calculate_file_hash(file_data)
			
			# Determine storage path
			file_path = self._get_storage_path(tenant_id, document_number, file_data.filename)
			
			# Create document record
			document = GCDMDocument(
				tenant_id=tenant_id,
				document_number=document_number,
				title=title,
				description=description,
				category_id=category_id,
				document_type_id=document_type_id,
				folder_id=folder_id,
				file_name=secure_filename(file_data.filename),
				file_path=file_path,
				file_size_bytes=self._get_file_size(file_data),
				file_extension=self._get_file_extension(file_data.filename),
				mime_type=file_data.mimetype or mimetypes.guess_type(file_data.filename)[0],
				file_hash=file_hash,
				owner_user_id=owner_user_id,
				status=DocumentStatus.DRAFT.value
			)
			
			# Save file to storage
			self._save_file_to_storage(file_data, file_path)
			
			self.db.add(document)
			self.db.flush()  # Get document ID
			
			# Create initial version
			self._create_initial_version(document, owner_user_id)
			
			# Apply metadata
			if metadata:
				self._apply_document_metadata(document.id, metadata)
			
			# Apply tags
			if tags:
				self._apply_document_tags(document.id, tags, owner_user_id)
			
			# Extract text content for search
			self._extract_document_content(document)
			
			# Apply retention policy
			self._apply_retention_policy(document)
			
			# Log audit event
			self._log_audit_event(
				tenant_id=tenant_id,
				document_id=document.id,
				activity_type="document_created",
				activity_description=f"Document '{title}' created",
				user_id=owner_user_id
			)
			
			self.db.commit()
			return document
			
		except Exception as e:
			self.db.rollback()
			logger.error(f"Failed to create document: {e}")
			raise
	
	def get_document(self, document_id: str, user_id: str) -> Optional[GCDMDocument]:
		"""Get document by ID with permission check."""
		document = self.db.query(GCDMDocument).filter_by(
			id=document_id, 
			is_active=True, 
			is_deleted=False
		).first()
		
		if not document:
			return None
		
		# Check read permission
		if not self._check_document_permission(document, user_id, PermissionLevel.READ):
			raise DocumentPermissionError("Insufficient permissions to view document")
		
		# Update view statistics
		document.view_count += 1
		document.last_viewed_date = datetime.utcnow()
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_viewed",
			activity_description=f"Document viewed",
			user_id=user_id
		)
		
		self.db.commit()
		return document
	
	def update_document(self, 
					   document_id: str, 
					   user_id: str,
					   updates: Dict[str, Any]) -> GCDMDocument:
		"""Update document metadata."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check write permission
		if not self._check_document_permission(document, user_id, PermissionLevel.WRITE):
			raise DocumentPermissionError("Insufficient permissions to update document")
		
		# Check if document is checked out by someone else
		if document.is_checked_out and document.checked_out_by != user_id:
			raise DocumentCheckoutError("Document is checked out by another user")
		
		# Store before values for audit
		before_values = {
			'title': document.title,
			'description': document.description,
			'category_id': document.category_id,
			'document_type_id': document.document_type_id
		}
		
		# Apply updates
		allowed_fields = ['title', 'description', 'category_id', 'document_type_id', 'keywords']
		for field, value in updates.items():
			if field in allowed_fields and hasattr(document, field):
				setattr(document, field, value)
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_updated",
			activity_description=f"Document metadata updated",
			user_id=user_id,
			before_values=json.dumps(before_values),
			after_values=json.dumps(updates)
		)
		
		self.db.commit()
		return document
	
	def delete_document(self, document_id: str, user_id: str, hard_delete: bool = False) -> bool:
		"""Delete document (soft delete by default)."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check delete permission
		if not self._check_document_permission(document, user_id, PermissionLevel.DELETE):
			raise DocumentPermissionError("Insufficient permissions to delete document")
		
		# Check if document is checked out
		if document.is_checked_out:
			raise DocumentCheckoutError("Cannot delete checked out document")
		
		# Check legal hold
		if document.legal_hold:
			raise DocumentRetentionError("Cannot delete document under legal hold")
		
		if hard_delete:
			# Remove file from storage
			self._remove_file_from_storage(document.file_path)
			
			# Delete from database
			self.db.delete(document)
		else:
			# Soft delete
			document.is_deleted = True
			document.deleted_date = datetime.utcnow()
			document.is_active = False
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_deleted",
			activity_description=f"Document deleted ({'hard' if hard_delete else 'soft'})",
			user_id=user_id
		)
		
		self.db.commit()
		return True
	
	# Document Search and Discovery
	
	def search_documents(self, 
						tenant_id: str,
						user_id: str,
						query: Optional[str] = None,
						category_id: Optional[str] = None,
						document_type_id: Optional[str] = None,
						folder_id: Optional[str] = None,
						status: Optional[str] = None,
						tags: Optional[List[str]] = None,
						date_from: Optional[date] = None,
						date_to: Optional[date] = None,
						owner_user_id: Optional[str] = None,
						page: int = 1,
						page_size: int = 50) -> Tuple[List[GCDMDocument], int]:
		"""Search documents with filters."""
		
		query_obj = self.db.query(GCDMDocument).filter(
			GCDMDocument.tenant_id == tenant_id,
			GCDMDocument.is_active == True,
			GCDMDocument.is_deleted == False
		)
		
		# Apply filters
		if query:
			search_filter = or_(
				GCDMDocument.title.ilike(f"%{query}%"),
				GCDMDocument.description.ilike(f"%{query}%"),
				GCDMDocument.content_text.ilike(f"%{query}%"),
				GCDMDocument.keywords.ilike(f"%{query}%")
			)
			query_obj = query_obj.filter(search_filter)
		
		if category_id:
			query_obj = query_obj.filter(GCDMDocument.category_id == category_id)
		
		if document_type_id:
			query_obj = query_obj.filter(GCDMDocument.document_type_id == document_type_id)
		
		if folder_id:
			query_obj = query_obj.filter(GCDMDocument.folder_id == folder_id)
		
		if status:
			query_obj = query_obj.filter(GCDMDocument.status == status)
		
		if owner_user_id:
			query_obj = query_obj.filter(GCDMDocument.owner_user_id == owner_user_id)
		
		if date_from:
			query_obj = query_obj.filter(GCDMDocument.document_date >= date_from)
		
		if date_to:
			query_obj = query_obj.filter(GCDMDocument.document_date <= date_to)
		
		# Filter by tags if specified
		if tags:
			query_obj = query_obj.join(GCDMDocumentTag).join(GCDMTag).filter(
				GCDMTag.name.in_(tags)
			)
		
		# Count total results
		total_count = query_obj.count()
		
		# Apply pagination
		offset = (page - 1) * page_size
		documents = query_obj.order_by(desc(GCDMDocument.created_on)).offset(offset).limit(page_size).all()
		
		# Filter by permissions (check each document individually)
		accessible_documents = []
		for doc in documents:
			if self._check_document_permission(doc, user_id, PermissionLevel.READ):
				accessible_documents.append(doc)
		
		return accessible_documents, total_count
	
	# Document Version Management
	
	def create_new_version(self, 
						  document_id: str,
						  user_id: str,
						  file_data: FileStorage,
						  change_description: str,
						  change_type: str = "minor") -> GCDMDocumentVersion:
		"""Create a new version of a document."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check write permission
		if not self._check_document_permission(document, user_id, PermissionLevel.WRITE):
			raise DocumentPermissionError("Insufficient permissions to create new version")
		
		# Check if document is checked out by current user
		if document.is_checked_out and document.checked_out_by != user_id:
			raise DocumentCheckoutError("Document must be checked out to create new version")
		
		# Validate file
		self._validate_file(file_data)
		
		# Generate new version number
		new_version = self._generate_version_number(document, change_type)
		
		# Calculate file hash
		file_hash = self._calculate_file_hash(file_data)
		
		# Determine storage path for new version
		version_file_path = self._get_version_storage_path(
			document.tenant_id, 
			document.document_number, 
			new_version, 
			file_data.filename
		)
		
		# Create version record
		version = GCDMDocumentVersion(
			document_id=document.id,
			version_number=new_version,
			file_name=secure_filename(file_data.filename),
			file_path=version_file_path,
			file_size_bytes=self._get_file_size(file_data),
			file_hash=file_hash,
			change_description=change_description,
			change_type=change_type,
			changed_by=user_id,
			status="draft"
		)
		
		# Save new version file
		self._save_file_to_storage(file_data, version_file_path)
		
		# Mark previous version as not current
		self.db.query(GCDMDocumentVersion).filter_by(
			document_id=document.id,
			is_current=True
		).update({'is_current': False})
		
		# Mark new version as current
		version.is_current = True
		
		# Update document with new version info
		document.version_number = new_version
		document.file_name = version.file_name
		document.file_path = version_file_path
		document.file_size_bytes = version.file_size_bytes
		document.file_hash = file_hash
		
		self.db.add(version)
		
		# Extract content for search
		self._extract_document_content(document)
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="version_created",
			activity_description=f"New version {new_version} created: {change_description}",
			user_id=user_id
		)
		
		self.db.commit()
		return version
	
	def get_document_versions(self, document_id: str, user_id: str) -> List[GCDMDocumentVersion]:
		"""Get all versions of a document."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		versions = self.db.query(GCDMDocumentVersion).filter_by(
			document_id=document.id
		).order_by(desc(GCDMDocumentVersion.created_on)).all()
		
		return versions
	
	def revert_to_version(self, document_id: str, version_id: str, user_id: str) -> bool:
		"""Revert document to a specific version."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check write permission
		if not self._check_document_permission(document, user_id, PermissionLevel.WRITE):
			raise DocumentPermissionError("Insufficient permissions to revert document")
		
		version = self.db.query(GCDMDocumentVersion).filter_by(
			id=version_id,
			document_id=document.id
		).first()
		
		if not version:
			raise DocumentVersionError("Version not found")
		
		# Update document to use version file
		document.file_name = version.file_name
		document.file_path = version.file_path
		document.file_size_bytes = version.file_size_bytes
		document.file_hash = version.file_hash
		
		# Update version status
		self.db.query(GCDMDocumentVersion).filter_by(
			document_id=document.id,
			is_current=True
		).update({'is_current': False})
		
		version.is_current = True
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="version_reverted",
			activity_description=f"Reverted to version {version.version_number}",
			user_id=user_id
		)
		
		self.db.commit()
		return True
	
	# Document Checkout/Checkin
	
	def checkout_document(self, 
						 document_id: str, 
						 user_id: str,
						 checkout_reason: Optional[str] = None,
						 expected_return_hours: Optional[int] = None) -> GCDMCheckout:
		"""Check out a document for exclusive editing."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check write permission
		if not self._check_document_permission(document, user_id, PermissionLevel.WRITE):
			raise DocumentPermissionError("Insufficient permissions to checkout document")
		
		# Check if already checked out
		if document.is_checked_out:
			if document.checked_out_by == user_id:
				raise DocumentCheckoutError("Document already checked out by you")
			else:
				raise DocumentCheckoutError("Document is checked out by another user")
		
		# Create checkout record
		checkout = GCDMCheckout(
			document_id=document.id,
			checked_out_by=user_id,
			checkout_reason=checkout_reason,
			expected_return_date=datetime.utcnow() + timedelta(hours=expected_return_hours) if expected_return_hours else None
		)
		
		# Update document checkout status
		document.is_checked_out = True
		document.checked_out_by = user_id
		document.checked_out_date = datetime.utcnow()
		
		self.db.add(checkout)
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_checked_out",
			activity_description=f"Document checked out for editing",
			user_id=user_id
		)
		
		self.db.commit()
		return checkout
	
	def checkin_document(self, 
						document_id: str, 
						user_id: str,
						return_comments: Optional[str] = None) -> bool:
		"""Check in a document."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check if document is checked out by current user
		if not document.is_checked_out or document.checked_out_by != user_id:
			raise DocumentCheckoutError("Document is not checked out by you")
		
		# Find active checkout
		checkout = self.db.query(GCDMCheckout).filter_by(
			document_id=document.id,
			checked_out_by=user_id,
			is_active=True
		).first()
		
		if checkout:
			checkout.returned_date = datetime.utcnow()
			checkout.return_comments = return_comments
			checkout.is_active = False
		
		# Update document checkout status
		document.is_checked_out = False
		document.checked_out_by = None
		document.checked_out_date = None
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_checked_in",
			activity_description=f"Document checked in",
			user_id=user_id
		)
		
		self.db.commit()
		return True
	
	# Document Download
	
	def download_document(self, document_id: str, user_id: str) -> Tuple[str, str]:
		"""Download document content."""
		document = self.get_document(document_id, user_id)
		if not document:
			raise ValueError("Document not found")
		
		# Check read permission
		if not self._check_document_permission(document, user_id, PermissionLevel.READ):
			raise DocumentPermissionError("Insufficient permissions to download document")
		
		# Verify file exists
		if not os.path.exists(document.file_path):
			raise FileNotFoundError("Document file not found in storage")
		
		# Update download statistics
		document.download_count += 1
		
		# Log audit event
		self._log_audit_event(
			tenant_id=document.tenant_id,
			document_id=document.id,
			activity_type="document_downloaded",
			activity_description=f"Document downloaded",
			user_id=user_id
		)
		
		self.db.commit()
		
		return document.file_path, document.file_name
	
	# Helper Methods
	
	def _validate_file(self, file_data: FileStorage) -> None:
		"""Validate uploaded file."""
		if not file_data or not file_data.filename:
			raise ValueError("No file provided")
		
		# Check file size
		file_size = self._get_file_size(file_data)
		if file_size > self.max_file_size:
			raise ValueError(f"File size exceeds maximum allowed size of {self.max_file_size // (1024*1024)}MB")
		
		# Check file extension
		file_ext = self._get_file_extension(file_data.filename)
		if file_ext.lower() not in self.allowed_extensions:
			raise ValueError(f"File type '{file_ext}' not allowed")
	
	def _generate_document_number(self, tenant_id: str) -> str:
		"""Generate unique document number."""
		# Get next sequence number for tenant
		result = self.db.execute(text("""
			SELECT COALESCE(MAX(CAST(SUBSTRING(document_number FROM '[0-9]+$') AS INTEGER)), 0) + 1
			FROM gc_dm_document 
			WHERE tenant_id = :tenant_id
		"""), {'tenant_id': tenant_id}).scalar()
		
		return f"DOC-{result:06d}"
	
	def _calculate_file_hash(self, file_data: FileStorage) -> str:
		"""Calculate SHA-256 hash of file."""
		file_data.stream.seek(0)
		hash_sha256 = hashlib.sha256()
		for chunk in iter(lambda: file_data.stream.read(4096), b""):
			hash_sha256.update(chunk)
		file_data.stream.seek(0)
		return hash_sha256.hexdigest()
	
	def _get_file_size(self, file_data: FileStorage) -> int:
		"""Get file size in bytes."""
		file_data.stream.seek(0, 2)  # Seek to end
		size = file_data.stream.tell()
		file_data.stream.seek(0)  # Reset to beginning
		return size
	
	def _get_file_extension(self, filename: str) -> str:
		"""Get file extension."""
		return filename.rsplit('.', 1)[1] if '.' in filename else ''
	
	def _get_storage_path(self, tenant_id: str, document_number: str, filename: str) -> str:
		"""Generate storage path for document."""
		safe_filename = secure_filename(filename)
		year = datetime.now().year
		month = datetime.now().month
		
		path = Path(self.storage_root) / tenant_id / str(year) / f"{month:02d}" / document_number
		path.mkdir(parents=True, exist_ok=True)
		
		return str(path / safe_filename)
	
	def _get_version_storage_path(self, tenant_id: str, document_number: str, version: str, filename: str) -> str:
		"""Generate storage path for document version."""
		safe_filename = secure_filename(filename)
		year = datetime.now().year
		month = datetime.now().month
		
		path = Path(self.storage_root) / tenant_id / str(year) / f"{month:02d}" / document_number / "versions"
		path.mkdir(parents=True, exist_ok=True)
		
		version_filename = f"v{version}_{safe_filename}"
		return str(path / version_filename)
	
	def _save_file_to_storage(self, file_data: FileStorage, file_path: str) -> None:
		"""Save file to storage."""
		file_data.save(file_path)
	
	def _remove_file_from_storage(self, file_path: str) -> None:
		"""Remove file from storage."""
		if os.path.exists(file_path):
			os.remove(file_path)
	
	def _create_initial_version(self, document: GCDMDocument, user_id: str) -> None:
		"""Create initial version record for new document."""
		version = GCDMDocumentVersion(
			document_id=document.id,
			version_number="1.0",
			version_label="Initial version",
			is_current=True,
			file_name=document.file_name,
			file_path=document.file_path,
			file_size_bytes=document.file_size_bytes,
			file_hash=document.file_hash,
			change_description="Initial document creation",
			change_type="initial",
			changed_by=user_id,
			status="published"
		)
		self.db.add(version)
	
	def _apply_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
		"""Apply custom metadata to document."""
		for key, value in metadata.items():
			meta = GCDMMetadata(
				document_id=document_id,
				metadata_key=key,
				metadata_value=str(value),
				data_type=self._detect_data_type(value)
			)
			self.db.add(meta)
	
	def _apply_document_tags(self, document_id: str, tag_names: List[str], user_id: str) -> None:
		"""Apply tags to document."""
		for tag_name in tag_names:
			# Find or create tag
			tag = self.db.query(GCDMTag).filter_by(name=tag_name).first()
			if not tag:
				tag = GCDMTag(name=tag_name, tenant_id=user_id)  # Use appropriate tenant_id
				self.db.add(tag)
				self.db.flush()
			
			# Create document-tag association
			doc_tag = GCDMDocumentTag(
				document_id=document_id,
				tag_id=tag.id,
				applied_by=user_id
			)
			self.db.add(doc_tag)
	
	def _extract_document_content(self, document: GCDMDocument) -> None:
		"""Extract text content from document for search indexing."""
		# This is a placeholder - in a real implementation, you would use
		# libraries like python-docx, PyPDF2, etc. to extract text content
		# For now, we'll just store the filename as searchable content
		document.content_text = f"{document.file_name} {document.title} {document.description or ''}"
	
	def _apply_retention_policy(self, document: GCDMDocument) -> None:
		"""Apply appropriate retention policy to document."""
		# Find applicable retention policy based on document type/category
		doc_type = self.db.query(GCDMDocumentType).filter_by(id=document.document_type_id).first()
		if doc_type and doc_type.retention_policy_id:
			policy = self.db.query(GCDMRetentionPolicy).filter_by(id=doc_type.retention_policy_id).first()
			if policy:
				document.retention_policy_id = policy.id
				# Calculate retention date
				retention_years = policy.retention_period_years
				retention_months = policy.retention_period_months
				retention_days = policy.retention_period_days
				
				retention_date = document.document_date
				if retention_years:
					retention_date = retention_date.replace(year=retention_date.year + retention_years)
				if retention_months:
					month = retention_date.month + retention_months
					year = retention_date.year + (month - 1) // 12
					month = ((month - 1) % 12) + 1
					retention_date = retention_date.replace(year=year, month=month)
				if retention_days:
					retention_date = retention_date + timedelta(days=retention_days)
				
				document.retention_date = retention_date
	
	def _generate_version_number(self, document: GCDMDocument, change_type: str) -> str:
		"""Generate next version number."""
		current_version = document.version_number
		parts = current_version.split('.')
		
		major = int(parts[0])
		minor = int(parts[1]) if len(parts) > 1 else 0
		patch = int(parts[2]) if len(parts) > 2 else 0
		
		if change_type == "major":
			major += 1
			minor = 0
			patch = 0
		elif change_type == "minor":
			minor += 1
			patch = 0
		else:  # patch
			patch += 1
		
		return f"{major}.{minor}.{patch}"
	
	def _check_document_permission(self, document: GCDMDocument, user_id: str, required_permission: PermissionLevel) -> bool:
		"""Check if user has required permission for document."""
		# Owner always has full access
		if document.owner_user_id == user_id:
			return True
		
		# Check if document is public and permission is read
		if document.is_public and required_permission == PermissionLevel.READ:
			return True
		
		# Check explicit permissions
		permission = self.db.query(GCDMPermission).filter(
			GCDMPermission.document_id == document.id,
			GCDMPermission.subject_type == "user",
			GCDMPermission.subject_id == user_id,
			GCDMPermission.is_active == True
		).first()
		
		if permission:
			return self._permission_level_allows(permission.permission_level, required_permission)
		
		# Check folder permissions (inherited)
		folder_permission = self.db.query(GCDMPermission).filter(
			GCDMPermission.folder_id == document.folder_id,
			GCDMPermission.subject_type == "user",
			GCDMPermission.subject_id == user_id,
			GCDMPermission.is_active == True,
			GCDMPermission.applies_to_children == True
		).first()
		
		if folder_permission:
			return self._permission_level_allows(folder_permission.permission_level, required_permission)
		
		return False
	
	def _permission_level_allows(self, granted_level: str, required_level: PermissionLevel) -> bool:
		"""Check if granted permission level allows required operation."""
		level_hierarchy = {
			PermissionLevel.NONE.value: 0,
			PermissionLevel.READ.value: 1,
			PermissionLevel.WRITE.value: 2,
			PermissionLevel.DELETE.value: 3,
			PermissionLevel.ADMIN.value: 4
		}
		
		granted_value = level_hierarchy.get(granted_level, 0)
		required_value = level_hierarchy.get(required_level.value, 0)
		
		return granted_value >= required_value
	
	def _detect_data_type(self, value: Any) -> str:
		"""Detect data type of metadata value."""
		if isinstance(value, bool):
			return "boolean"
		elif isinstance(value, int):
			return "number"
		elif isinstance(value, float):
			return "number"
		elif isinstance(value, (dict, list)):
			return "json"
		else:
			return "string"
	
	def _log_audit_event(self, 
					   tenant_id: str,
					   activity_type: str,
					   activity_description: str,
					   user_id: str,
					   document_id: Optional[str] = None,
					   before_values: Optional[str] = None,
					   after_values: Optional[str] = None) -> None:
		"""Log audit event."""
		audit_log = GCDMAuditLog(
			tenant_id=tenant_id,
			document_id=document_id,
			activity_type=activity_type,
			activity_description=activity_description,
			user_id=user_id,
			before_values=before_values,
			after_values=after_values
		)
		self.db.add(audit_log)