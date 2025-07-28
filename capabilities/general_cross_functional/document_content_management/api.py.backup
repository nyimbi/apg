"""
Document Management REST API

Comprehensive REST API endpoints for document management operations including
CRUD operations, search, file upload/download, version control, and workflow management.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, date
import json
import logging
import os

from flask import Blueprint, request, jsonify, send_file, current_app
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized, Forbidden
from marshmallow import Schema, fields as ma_fields, ValidationError
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .models import (
	GCDMDocument, GCDMDocumentVersion, GCDMDocumentCategory, GCDMDocumentType,
	GCDMFolder, GCDMPermission, GCDMCheckout, GCDMWorkflow, GCDMReview,
	GCDMTag, GCDMRetentionPolicy, GCDMArchive, GCDMAuditLog
)
from .service import DocumentService, DocumentPermissionError, DocumentCheckoutError, DocumentVersionError

logger = logging.getLogger(__name__)

# Create API blueprint
api_bp = Blueprint('document_management_api', __name__, url_prefix='/api/general_cross_functional/dm')
api = Api(api_bp, doc='/doc/', title='Document Management API', version='1.0', description='Document Management REST API')

# Create namespace
ns = Namespace('documents', description='Document management operations')
api.add_namespace(ns)

# API Models for Swagger documentation
document_model = api.model('Document', {
	'id': fields.String(description='Document ID'),
	'document_number': fields.String(description='Document number'),
	'title': fields.String(required=True, description='Document title'),
	'description': fields.String(description='Document description'),
	'category_id': fields.String(required=True, description='Category ID'),
	'document_type_id': fields.String(required=True, description='Document type ID'),
	'folder_id': fields.String(required=True, description='Folder ID'),
	'status': fields.String(description='Document status'),
	'version_number': fields.String(description='Version number'),
	'file_name': fields.String(description='File name'),
	'file_size_bytes': fields.Integer(description='File size in bytes'),
	'owner_user_id': fields.String(description='Owner user ID'),
	'document_date': fields.Date(description='Document date'),
	'created_on': fields.DateTime(description='Creation timestamp'),
	'changed_on': fields.DateTime(description='Last modified timestamp')
})

document_create_model = api.model('DocumentCreate', {
	'title': fields.String(required=True, description='Document title'),
	'description': fields.String(description='Document description'),
	'category_id': fields.String(required=True, description='Category ID'),
	'document_type_id': fields.String(required=True, description='Document type ID'),
	'folder_id': fields.String(required=True, description='Folder ID'),
	'keywords': fields.String(description='Keywords'),
	'tags': fields.List(fields.String, description='Tag list')
})

document_update_model = api.model('DocumentUpdate', {
	'title': fields.String(description='Document title'),
	'description': fields.String(description='Document description'),
	'category_id': fields.String(description='Category ID'),
	'document_type_id': fields.String(description='Document type ID'),
	'keywords': fields.String(description='Keywords')
})

version_model = api.model('DocumentVersion', {
	'id': fields.String(description='Version ID'),
	'version_number': fields.String(description='Version number'),
	'change_description': fields.String(description='Change description'),
	'change_type': fields.String(description='Change type'),
	'changed_by': fields.String(description='Changed by user'),
	'is_current': fields.Boolean(description='Is current version'),
	'created_on': fields.DateTime(description='Creation timestamp')
})

search_model = api.model('DocumentSearch', {
	'query': fields.String(description='Search query'),
	'category_id': fields.String(description='Category filter'),
	'document_type_id': fields.String(description='Document type filter'),
	'folder_id': fields.String(description='Folder filter'),
	'status': fields.String(description='Status filter'),
	'tags': fields.List(fields.String, description='Tag filters'),
	'date_from': fields.Date(description='Date from'),
	'date_to': fields.Date(description='Date to'),
	'owner_user_id': fields.String(description='Owner filter'),
	'page': fields.Integer(description='Page number', default=1),
	'page_size': fields.Integer(description='Page size', default=50)
})

folder_model = api.model('Folder', {
	'id': fields.String(description='Folder ID'),
	'name': fields.String(required=True, description='Folder name'),
	'description': fields.String(description='Folder description'),
	'folder_path': fields.String(description='Folder path'),
	'parent_folder_id': fields.String(description='Parent folder ID'),
	'document_count': fields.Integer(description='Number of documents'),
	'is_active': fields.Boolean(description='Is active')
})

permission_model = api.model('Permission', {
	'id': fields.String(description='Permission ID'),
	'document_id': fields.String(description='Document ID'),
	'folder_id': fields.String(description='Folder ID'),
	'subject_type': fields.String(required=True, description='Subject type (user/role/group)'),
	'subject_id': fields.String(required=True, description='Subject ID'),
	'permission_level': fields.String(required=True, description='Permission level'),
	'can_read': fields.Boolean(description='Can read'),
	'can_write': fields.Boolean(description='Can write'),
	'can_delete': fields.Boolean(description='Can delete'),
	'can_share': fields.Boolean(description='Can share'),
	'can_approve': fields.Boolean(description='Can approve')
})

# Marshmallow schemas for validation
class DocumentSearchSchema(Schema):
	query = ma_fields.String(allow_none=True)
	category_id = ma_fields.String(allow_none=True)
	document_type_id = ma_fields.String(allow_none=True)
	folder_id = ma_fields.String(allow_none=True)
	status = ma_fields.String(allow_none=True)
	tags = ma_fields.List(ma_fields.String(), allow_none=True)
	date_from = ma_fields.Date(allow_none=True)
	date_to = ma_fields.Date(allow_none=True)
	owner_user_id = ma_fields.String(allow_none=True)
	page = ma_fields.Integer(missing=1, validate=lambda x: x > 0)
	page_size = ma_fields.Integer(missing=50, validate=lambda x: 1 <= x <= 1000)

class DocumentCreateSchema(Schema):
	title = ma_fields.String(required=True, validate=lambda x: len(x.strip()) > 0)
	description = ma_fields.String(allow_none=True)
	category_id = ma_fields.String(required=True)
	document_type_id = ma_fields.String(required=True)
	folder_id = ma_fields.String(required=True)
	keywords = ma_fields.String(allow_none=True)
	tags = ma_fields.List(ma_fields.String(), allow_none=True)

class DocumentUpdateSchema(Schema):
	title = ma_fields.String(validate=lambda x: len(x.strip()) > 0 if x else True)
	description = ma_fields.String(allow_none=True)
	category_id = ma_fields.String()
	document_type_id = ma_fields.String()
	keywords = ma_fields.String(allow_none=True)

# Helper functions
def get_current_user_id() -> str:
	"""Get current user ID from request context."""
	# This should be implemented based on your authentication system
	return request.headers.get('X-User-ID', 'anonymous')

def get_current_tenant_id() -> str:
	"""Get current tenant ID from request context."""
	# This should be implemented based on your multi-tenancy system
	return request.headers.get('X-Tenant-ID', 'default')

def get_db_session() -> Session:
	"""Get database session."""
	# This should return the current database session
	# Implementation depends on your database setup
	from flask import g
	return g.db_session

def serialize_document(document: GCDMDocument) -> Dict[str, Any]:
	"""Serialize document to dictionary."""
	return {
		'id': document.id,
		'tenant_id': document.tenant_id,
		'document_number': document.document_number,
		'title': document.title,
		'description': document.description,
		'category_id': document.category_id,
		'document_type_id': document.document_type_id,
		'folder_id': document.folder_id,
		'file_name': document.file_name,
		'file_size_bytes': document.file_size_bytes,
		'file_extension': document.file_extension,
		'mime_type': document.mime_type,
		'version_number': document.version_number,
		'status': document.status,
		'owner_user_id': document.owner_user_id,
		'document_date': document.document_date.isoformat() if document.document_date else None,
		'effective_date': document.effective_date.isoformat() if document.effective_date else None,
		'expiry_date': document.expiry_date.isoformat() if document.expiry_date else None,
		'is_checked_out': document.is_checked_out,
		'checked_out_by': document.checked_out_by,
		'security_classification': document.security_classification,
		'view_count': document.view_count,
		'download_count': document.download_count,
		'keywords': document.keywords,
		'created_on': document.created_on.isoformat() if document.created_on else None,
		'changed_on': document.changed_on.isoformat() if document.changed_on else None
	}

def serialize_version(version: GCDMDocumentVersion) -> Dict[str, Any]:
	"""Serialize document version to dictionary."""
	return {
		'id': version.id,
		'document_id': version.document_id,
		'version_number': version.version_number,
		'version_label': version.version_label,
		'is_current': version.is_current,
		'change_description': version.change_description,
		'change_type': version.change_type,
		'changed_by': version.changed_by,
		'status': version.status,
		'file_name': version.file_name,
		'file_size_bytes': version.file_size_bytes,
		'created_on': version.created_on.isoformat() if version.created_on else None
	}

def handle_api_error(e: Exception) -> Tuple[Dict[str, Any], int]:
	"""Handle API errors and return appropriate response."""
	if isinstance(e, DocumentPermissionError):
		return {'error': 'Permission denied', 'message': str(e)}, 403
	elif isinstance(e, DocumentCheckoutError):
		return {'error': 'Checkout error', 'message': str(e)}, 409
	elif isinstance(e, DocumentVersionError):
		return {'error': 'Version error', 'message': str(e)}, 409
	elif isinstance(e, ValidationError):
		return {'error': 'Validation error', 'message': e.messages}, 400
	elif isinstance(e, ValueError):
		return {'error': 'Invalid input', 'message': str(e)}, 400
	elif isinstance(e, FileNotFoundError):
		return {'error': 'File not found', 'message': str(e)}, 404
	else:
		logger.error(f"API error: {e}")
		return {'error': 'Internal server error', 'message': 'An unexpected error occurred'}, 500

# API Endpoints

@ns.route('/documents')
class DocumentListAPI(Resource):
	@api.doc('list_documents')
	@api.marshal_list_with(document_model)
	def get(self):
		"""Get list of documents with optional filtering."""
		try:
			# Parse query parameters
			args = request.args.to_dict()
			search_schema = DocumentSearchSchema()
			search_params = search_schema.load(args)
			
			# Get service
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			tenant_id = get_current_tenant_id()
			
			# Perform search
			documents, total_count = service.search_documents(
				tenant_id=tenant_id,
				user_id=user_id,
				**search_params
			)
			
			# Serialize results
			result = {
				'documents': [serialize_document(doc) for doc in documents],
				'total_count': total_count,
				'page': search_params['page'],
				'page_size': search_params['page_size']
			}
			
			return result
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code
	
	@api.doc('create_document')
	@api.expect(document_create_model)
	@api.marshal_with(document_model, code=201)
	def post(self):
		"""Create a new document with file upload."""
		try:
			# Validate JSON data
			json_data = request.get_json() or {}
			create_schema = DocumentCreateSchema()
			document_data = create_schema.load(json_data)
			
			# Check for file upload
			if 'file' not in request.files:
				return {'error': 'No file provided'}, 400
			
			file_data = request.files['file']
			if not file_data or file_data.filename == '':
				return {'error': 'No file selected'}, 400
			
			# Get service
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			tenant_id = get_current_tenant_id()
			
			# Create document
			document = service.create_document(
				tenant_id=tenant_id,
				title=document_data['title'],
				file_data=file_data,
				category_id=document_data['category_id'],
				document_type_id=document_data['document_type_id'],
				folder_id=document_data['folder_id'],
				owner_user_id=user_id,
				description=document_data.get('description'),
				tags=document_data.get('tags')
			)
			
			return serialize_document(document), 201
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>')
class DocumentAPI(Resource):
	@api.doc('get_document')
	@api.marshal_with(document_model)
	def get(self, document_id):
		"""Get document by ID."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			document = service.get_document(document_id, user_id)
			if not document:
				return {'error': 'Document not found'}, 404
			
			return serialize_document(document)
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code
	
	@api.doc('update_document')
	@api.expect(document_update_model)
	@api.marshal_with(document_model)
	def put(self, document_id):
		"""Update document metadata."""
		try:
			# Validate JSON data
			json_data = request.get_json() or {}
			update_schema = DocumentUpdateSchema()
			update_data = update_schema.load(json_data)
			
			# Get service
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			# Update document
			document = service.update_document(document_id, user_id, update_data)
			
			return serialize_document(document)
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code
	
	@api.doc('delete_document')
	def delete(self, document_id):
		"""Delete document."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			hard_delete = request.args.get('hard_delete', 'false').lower() == 'true'
			success = service.delete_document(document_id, user_id, hard_delete)
			
			if success:
				return {'message': 'Document deleted successfully'}, 200
			else:
				return {'error': 'Failed to delete document'}, 400
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>/download')
class DocumentDownloadAPI(Resource):
	@api.doc('download_document')
	def get(self, document_id):
		"""Download document file."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			file_path, file_name = service.download_document(document_id, user_id)
			
			return send_file(
				file_path,
				as_attachment=True,
				download_name=file_name,
				mimetype='application/octet-stream'
			)
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>/checkout')
class DocumentCheckoutAPI(Resource):
	@api.doc('checkout_document')
	def post(self, document_id):
		"""Checkout document for editing."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			json_data = request.get_json() or {}
			checkout_reason = json_data.get('reason')
			expected_return_hours = json_data.get('expected_return_hours')
			
			checkout = service.checkout_document(
				document_id, 
				user_id, 
				checkout_reason, 
				expected_return_hours
			)
			
			return {
				'message': 'Document checked out successfully',
				'checkout_id': checkout.id,
				'checkout_date': checkout.checkout_date.isoformat()
			}, 200
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>/checkin')
class DocumentCheckinAPI(Resource):
	@api.doc('checkin_document')
	def post(self, document_id):
		"""Checkin document."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			json_data = request.get_json() or {}
			return_comments = json_data.get('comments')
			
			success = service.checkin_document(document_id, user_id, return_comments)
			
			if success:
				return {'message': 'Document checked in successfully'}, 200
			else:
				return {'error': 'Failed to checkin document'}, 400
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>/versions')
class DocumentVersionsAPI(Resource):
	@api.doc('get_document_versions')
	@api.marshal_list_with(version_model)
	def get(self, document_id):
		"""Get document versions."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			versions = service.get_document_versions(document_id, user_id)
			
			return [serialize_version(version) for version in versions]
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code
	
	@api.doc('create_document_version')
	def post(self, document_id):
		"""Create new document version."""
		try:
			# Check for file upload
			if 'file' not in request.files:
				return {'error': 'No file provided'}, 400
			
			file_data = request.files['file']
			if not file_data or file_data.filename == '':
				return {'error': 'No file selected'}, 400
			
			# Get form data
			change_description = request.form.get('change_description', '')
			change_type = request.form.get('change_type', 'minor')
			
			if not change_description:
				return {'error': 'Change description is required'}, 400
			
			# Get service
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			# Create new version
			version = service.create_new_version(
				document_id, 
				user_id, 
				file_data, 
				change_description, 
				change_type
			)
			
			return serialize_version(version), 201
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/documents/<string:document_id>/versions/<string:version_id>/revert')
class DocumentVersionRevertAPI(Resource):
	@api.doc('revert_document_version')
	def post(self, document_id, version_id):
		"""Revert document to specific version."""
		try:
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			
			success = service.revert_to_version(document_id, version_id, user_id)
			
			if success:
				return {'message': 'Document reverted successfully'}, 200
			else:
				return {'error': 'Failed to revert document'}, 400
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/folders')
class FolderListAPI(Resource):
	@api.doc('list_folders')
	@api.marshal_list_with(folder_model)
	def get(self):
		"""Get list of folders."""
		try:
			db_session = get_db_session()
			tenant_id = get_current_tenant_id()
			
			folders = db_session.query(GCDMFolder).filter(
				GCDMFolder.tenant_id == tenant_id,
				GCDMFolder.is_active == True
			).all()
			
			result = []
			for folder in folders:
				result.append({
					'id': folder.id,
					'name': folder.name,
					'description': folder.description,
					'folder_path': folder.folder_path,
					'parent_folder_id': folder.parent_folder_id,
					'document_count': folder.document_count,
					'is_active': folder.is_active
				})
			
			return result
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/search')
class DocumentSearchAPI(Resource):
	@api.doc('search_documents')
	@api.expect(search_model)
	def post(self):
		"""Advanced document search."""
		try:
			# Validate search parameters
			json_data = request.get_json() or {}
			search_schema = DocumentSearchSchema()
			search_params = search_schema.load(json_data)
			
			# Get service
			service = DocumentService(get_db_session())
			user_id = get_current_user_id()
			tenant_id = get_current_tenant_id()
			
			# Perform search
			documents, total_count = service.search_documents(
				tenant_id=tenant_id,
				user_id=user_id,
				**search_params
			)
			
			# Serialize results
			result = {
				'documents': [serialize_document(doc) for doc in documents],
				'total_count': total_count,
				'page': search_params['page'],
				'page_size': search_params['page_size'],
				'search_params': search_params
			}
			
			return result
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

@ns.route('/dashboard')
class DocumentDashboardAPI(Resource):
	@api.doc('get_dashboard')
	def get(self):
		"""Get dashboard statistics."""
		try:
			db_session = get_db_session()
			tenant_id = get_current_tenant_id()
			
			# Get basic statistics
			from sqlalchemy import func
			
			# Total documents
			total_docs = db_session.query(func.count(GCDMDocument.id)).filter(
				GCDMDocument.tenant_id == tenant_id,
				GCDMDocument.is_active == True,
				GCDMDocument.is_deleted == False
			).scalar()
			
			# Documents by status
			status_stats = db_session.query(
				GCDMDocument.status,
				func.count(GCDMDocument.id)
			).filter(
				GCDMDocument.tenant_id == tenant_id,
				GCDMDocument.is_active == True,
				GCDMDocument.is_deleted == False
			).group_by(GCDMDocument.status).all()
			
			# Checked out documents
			checked_out = db_session.query(func.count(GCDMDocument.id)).filter(
				GCDMDocument.tenant_id == tenant_id,
				GCDMDocument.is_checked_out == True
			).scalar()
			
			# Total storage used
			total_storage = db_session.query(func.sum(GCDMDocument.file_size_bytes)).filter(
				GCDMDocument.tenant_id == tenant_id,
				GCDMDocument.is_active == True
			).scalar() or 0
			
			# Recent documents
			recent_docs = db_session.query(GCDMDocument).filter(
				GCDMDocument.tenant_id == tenant_id,
				GCDMDocument.is_active == True,
				GCDMDocument.is_deleted == False
			).order_by(GCDMDocument.created_on.desc()).limit(10).all()
			
			dashboard_data = {
				'statistics': {
					'total_documents': total_docs,
					'status_distribution': dict(status_stats),
					'checked_out_count': checked_out,
					'total_storage_mb': total_storage / (1024 * 1024),
					'average_file_size_mb': (total_storage / total_docs / (1024 * 1024)) if total_docs > 0 else 0
				},
				'recent_documents': [serialize_document(doc) for doc in recent_docs]
			}
			
			return dashboard_data
			
		except Exception as e:
			error_response, status_code = handle_api_error(e)
			return error_response, status_code

# Error handlers
@api.errorhandler(ValidationError)
def handle_validation_error(error):
	return {'error': 'Validation failed', 'messages': error.messages}, 400

@api.errorhandler(BadRequest)
def handle_bad_request(error):
	return {'error': 'Bad request', 'message': str(error)}, 400

@api.errorhandler(NotFound)
def handle_not_found(error):
	return {'error': 'Not found', 'message': str(error)}, 404

@api.errorhandler(Unauthorized)
def handle_unauthorized(error):
	return {'error': 'Unauthorized', 'message': str(error)}, 401

@api.errorhandler(Forbidden)
def handle_forbidden(error):
	return {'error': 'Forbidden', 'message': str(error)}, 403

# Register blueprint
def register_api(app):
	"""Register API blueprint with Flask app."""
	app.register_blueprint(api_bp)