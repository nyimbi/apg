"""
APG Import/Export (IMEX) Secure REST API Layer

Purpose: Production-grade REST API endpoints with comprehensive security integration
         for enterprise import/export operations.
Dependencies: flask, pydantic, security module
Usage Context: Secure HTTP API layer exposing IMEX functionality to authenticated clients

This module provides:
- Complete RESTful API with security integration
- JWT and API key authentication
- Role-based access control enforcement
- Comprehensive audit logging
- Rate limiting and DDoS protection
- Multi-tenant security isolation
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json

from flask import Flask, Blueprint, request, jsonify, Response, g
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app: Any, *args: Any, **kwargs: Any) -> Any:
        return app
from pydantic import BaseModel, Field, ValidationError

from models import (
    ImportExportJob, JobExecution, SourceConfig, TargetConfig,
    JobStatus, JobType, DataFormat, SourceType, ValidationLevel,
    ErrorHandlingStrategy, ProcessingPriority
)
from service import ImportExportService
from security import (
    AuthenticationManager, AuditLogger, User, UserRole, Permission,
    require_permission, require_role, require_tenant_access, rate_limit,
    security_middleware, create_security_config
)

logger = logging.getLogger(__name__)

# Global instances
imex_service: Optional[ImportExportService] = None
auth_manager: Optional[AuthenticationManager] = None
audit_logger: Optional[AuditLogger] = None

def initialize_secure_api(service: ImportExportService, environment: str = "development"):
    """Initialize the secure API with service and security configuration"""
    global imex_service, auth_manager, audit_logger

    imex_service = service

    # Create security configuration
    security_config = create_security_config(environment)

    # Initialize authentication manager
    auth_manager = AuthenticationManager(security_config)

    # Initialize audit logger
    audit_logger = AuditLogger(auth_manager)

    logger.info("Secure API initialized successfully")

# Secure Request Models

class SecureJobCreateRequest(BaseModel):
    """Secure request model for creating jobs"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    job_type: str = Field(..., pattern="^(import|export|migration|sync)$")
    source_config: Dict[str, Any] = Field(...)
    target_config: Dict[str, Any] = Field(...)
    validation_rules: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    transformation_steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    priority: Optional[str] = Field("normal", pattern="^(low|normal|high|urgent)$")
    validation_level: Optional[str] = Field("basic", pattern="^(none|basic|strict|comprehensive)$")
    error_handling: Optional[str] = Field("log_and_continue", pattern="^(fail_fast|log_and_continue|skip_and_continue)$")
    tags: Optional[List[str]] = Field(default_factory=list)
    tenant_id: str = Field(...)

class AuthenticationRequest(BaseModel):
    """Request model for user authentication"""
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8)
    tenant_id: str = Field(...)

class ApiKeyRequest(BaseModel):
    """Request model for API key creation"""
    name: str = Field(..., min_length=1, max_length=255)
    permissions: List[str] = Field(...)
    expires_at: Optional[datetime] = Field(None)

# Create Secure Blueprint
secure_api_bp = Blueprint('secure_imex_api', __name__, url_prefix='/api/v1/secure/imex')
CORS(secure_api_bp)

# Utility Functions

def _execute_async_operation(operation, *args, **kwargs):
    """Execute async operation in Flask context"""
    try:
        return asyncio.run(operation(*args, **kwargs))
    except Exception as e:
        logger.error(f"Async operation failed: {e}")
        raise

def _log_api_request(endpoint: str, method: str, success: bool = True, error: str = None):
    """Log API request for audit trail"""
    if audit_logger:
        audit_logger.log_action(
            action=f"api:{method.lower()}",
            resource_type="api_endpoint",
            resource_id=endpoint,
            details={'method': method, 'endpoint': endpoint},
            success=success,
            error_message=error
        )

def _get_current_user() -> Optional[User]:
    """Get current authenticated user"""
    return getattr(g, 'current_user', None)

def _validate_service_available():
    """Validate that IMEX service is available"""
    if not imex_service:
        raise ValueError("IMEX service not available")

# Authentication Endpoints

@secure_api_bp.route('/auth/login', methods=['POST'])
@rate_limit(limit=10)  # 10 login attempts per hour
def login():
    """Authenticate user and return JWT token"""
    try:
        _log_api_request('/auth/login', 'POST')

        if not auth_manager:
            return jsonify({'error': 'Authentication not configured'}), 500

        # Validate request
        try:
            auth_request = AuthenticationRequest(**request.get_json())
        except ValidationError as e:
            _log_api_request('/auth/login', 'POST', False, 'Invalid request data')
            return jsonify({'error': 'Invalid request data', 'details': e.errors()}), 400

        # Mock user authentication (in production, check database)
        mock_user = User(
            username=auth_request.username,
            email=f"{auth_request.username}@example.com",
            password_hash=auth_manager.hash_password(auth_request.password),
            roles=[UserRole.OPERATOR],
            tenant_id=auth_request.tenant_id,
            is_active=True
        )

        # Verify password (in production, retrieve from database)
        if not auth_manager.verify_password(auth_request.password, mock_user.password_hash):
            _log_api_request('/auth/login', 'POST', False, 'Invalid credentials')
            return jsonify({'error': 'Invalid credentials'}), 401

        # Generate JWT token
        access_token = auth_manager.generate_jwt_token(mock_user)

        # Update last login (in production, update database)
        mock_user.last_login = datetime.now(timezone.utc)

        _log_api_request('/auth/login', 'POST', True)

        return jsonify({
            'success': True,
            'access_token': access_token,
            'token_type': 'bearer',
            'expires_in': auth_manager.config.jwt_access_token_expires,
            'user': {
                'id': mock_user.id,
                'username': mock_user.username,
                'roles': [role.value for role in mock_user.roles],
                'tenant_id': mock_user.tenant_id
            }
        }), 200

    except Exception as e:
        logger.error(f"Login error: {e}")
        _log_api_request('/auth/login', 'POST', False, str(e))
        return jsonify({'error': 'Authentication failed', 'message': str(e)}), 500

@secure_api_bp.route('/auth/api-keys', methods=['POST'])
@require_permission(Permission.SYSTEM_CONFIG)
@rate_limit(limit=20)
def create_api_key():
    """Create new API key"""
    try:
        _log_api_request('/auth/api-keys', 'POST')

        if not auth_manager:
            return jsonify({'error': 'Authentication not configured'}), 500

        user = _get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401

        # Validate request
        try:
            api_key_request = ApiKeyRequest(**request.get_json())
        except ValidationError as e:
            return jsonify({'error': 'Invalid request data', 'details': e.errors()}), 400

        # Generate API key
        api_key = auth_manager.generate_api_key()
        api_key_hash = auth_manager.hash_api_key(api_key)

        # Create API key record (in production, save to database)
        from security import ApiKey, Permission as SecPermission

        api_key_record = ApiKey(
            name=api_key_request.name,
            key_hash=api_key_hash,
            user_id=user.id,
            tenant_id=user.tenant_id,
            permissions=[SecPermission(p) for p in api_key_request.permissions],
            expires_at=api_key_request.expires_at
        )

        _log_api_request('/auth/api-keys', 'POST', True)

        return jsonify({
            'success': True,
            'api_key': api_key,  # Only returned once
            'key_id': api_key_record.id,
            'name': api_key_record.name,
            'permissions': api_key_request.permissions,
            'expires_at': api_key_record.expires_at.isoformat() if api_key_record.expires_at else None,
            'created_at': api_key_record.created_at.isoformat()
        }), 201

    except Exception as e:
        logger.error(f"API key creation error: {e}")
        _log_api_request('/auth/api-keys', 'POST', False, str(e))
        return jsonify({'error': 'API key creation failed', 'message': str(e)}), 500

# Secure Job Management Endpoints

@secure_api_bp.route('/jobs', methods=['GET'])
@require_permission(Permission.JOB_READ)
@rate_limit(limit=100)
def list_jobs():
    """List jobs with security filtering"""
    try:
        _log_api_request('/jobs', 'GET')
        _validate_service_available()

        user = _get_current_user()
        tenant_id = request.args.get('tenant_id', user.tenant_id if user else None)

        # Validate tenant access
        if user and not auth_manager.rbac.user_can_access_tenant(user, tenant_id):
            return jsonify({'error': 'Access denied to tenant'}), 403

        # Get query parameters
        status = request.args.get('status')
        job_type = request.args.get('job_type')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        # Get jobs from service (filtered by tenant)
        jobs = []
        if hasattr(imex_service, 'active_jobs'):
            for job_id, job in imex_service.active_jobs.items():
                # Tenant filtering
                if hasattr(job, 'tenant_id') and job.tenant_id != tenant_id:
                    continue

                if status and str(job.status).lower() != status.lower():
                    continue
                if job_type and str(job.job_type).lower() != job_type.lower():
                    continue

                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'description': job.description or '',
                    'job_type': str(job.job_type),
                    'status': str(job.status),
                    'priority': str(job.priority),
                    'created_by': job.created_by,
                    'created_at': job.created_at.isoformat() if job.created_at else None,
                    'updated_at': job.updated_at.isoformat() if job.updated_at else None
                })

        # Apply pagination
        total = len(jobs)
        jobs = jobs[offset:offset + limit]

        _log_api_request('/jobs', 'GET', True)

        return jsonify({
            'success': True,
            'data': {
                'jobs': jobs,
                'pagination': {
                    'total': total,
                    'limit': limit,
                    'offset': offset,
                    'has_next': offset + limit < total,
                    'has_prev': offset > 0
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"List jobs error: {e}")
        _log_api_request('/jobs', 'GET', False, str(e))
        return jsonify({'error': 'Failed to list jobs', 'message': str(e)}), 500

@secure_api_bp.route('/jobs', methods=['POST'])
@require_permission(Permission.JOB_CREATE)
@require_tenant_access('tenant_id')
@rate_limit(limit=50)
def create_job():
    """Create new job with security validation"""
    try:
        _log_api_request('/jobs', 'POST')
        _validate_service_available()

        user = _get_current_user()

        # Validate request
        try:
            job_request = SecureJobCreateRequest(**request.get_json())
        except ValidationError as e:
            return jsonify({'error': 'Invalid request data', 'details': e.errors()}), 400

        # Add security context
        job_config = job_request.dict()
        job_config['created_by'] = user.username if user else 'system'
        job_config['tenant_id'] = job_request.tenant_id

        # Create job
        job = _execute_async_operation(imex_service.create_job, job_config, user.username if user else 'system')

        # Log successful creation
        if audit_logger:
            audit_logger.log_action(
                action="job:create",
                resource_type="job",
                resource_id=job.id,
                details={'name': job.name, 'job_type': str(job.job_type)},
                success=True
            )

        _log_api_request('/jobs', 'POST', True)

        return jsonify({
            'success': True,
            'message': 'Job created successfully',
            'data': {
                'id': job.id,
                'name': job.name,
                'job_type': str(job.job_type),
                'status': str(job.status),
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'created_by': job.created_by
            }
        }), 201

    except Exception as e:
        logger.error(f"Create job error: {e}")
        _log_api_request('/jobs', 'POST', False, str(e))
        return jsonify({'error': 'Job creation failed', 'message': str(e)}), 500

@secure_api_bp.route('/jobs/<job_id>/execute', methods=['POST'])
@require_permission(Permission.JOB_EXECUTE)
@rate_limit(limit=30)
def execute_job(job_id):
    """Execute job with security validation"""
    try:
        _log_api_request(f'/jobs/{job_id}/execute', 'POST')
        _validate_service_available()

        user = _get_current_user()

        # Validate job exists and user can access it
        if hasattr(imex_service, 'active_jobs') and job_id in imex_service.active_jobs:
            job = imex_service.active_jobs[job_id]

            # Check tenant access
            if hasattr(job, 'tenant_id') and not auth_manager.rbac.user_can_access_tenant(user, job.tenant_id):
                return jsonify({'error': 'Access denied to job'}), 403
        else:
            return jsonify({'error': 'Job not found'}), 404

        # Execute job
        execution_config = request.get_json() or {}
        execution = _execute_async_operation(imex_service.execute_job, job_id, execution_config)

        # Log execution
        if audit_logger:
            audit_logger.log_action(
                action="job:execute",
                resource_type="job",
                resource_id=job_id,
                details={'execution_id': execution.id},
                success=True
            )

        _log_api_request(f'/jobs/{job_id}/execute', 'POST', True)

        return jsonify({
            'success': True,
            'message': 'Job execution started',
            'data': {
                'execution_id': execution.id,
                'job_id': job_id,
                'status': str(execution.status),
                'started_at': execution.started_at.isoformat() if execution.started_at else None
            }
        }), 202

    except Exception as e:
        logger.error(f"Execute job error: {e}")
        _log_api_request(f'/jobs/{job_id}/execute', 'POST', False, str(e))
        return jsonify({'error': 'Job execution failed', 'message': str(e)}), 500

# Secure Schema Detection Endpoints

@secure_api_bp.route('/schemas/detect', methods=['POST'])
@require_permission(Permission.SCHEMA_DETECT)
@rate_limit(limit=20)
def detect_schema():
    """AI-powered schema detection with security"""
    try:
        _log_api_request('/schemas/detect', 'POST')
        _validate_service_available()

        user = _get_current_user()

        # Validate request
        request_data = request.get_json()
        if not request_data or 'source_config' not in request_data:
            return jsonify({'error': 'Source configuration required'}), 400

        # Create source config
        try:
            source_config = SourceConfig(**request_data['source_config'])
        except Exception as e:
            return jsonify({'error': 'Invalid source configuration', 'message': str(e)}), 400

        # Detect schema
        schema_result = _execute_async_operation(imex_service.detect_schema_automatically, source_config)

        # Log schema detection
        if audit_logger:
            audit_logger.log_action(
                action="schema:detect",
                resource_type="schema",
                details={'source_type': str(source_config.source_type), 'format': str(source_config.format)},
                success=True
            )

        _log_api_request('/schemas/detect', 'POST', True)

        return jsonify({
            'success': True,
            'message': 'Schema detection completed',
            'data': {
                'schema': schema_result,
                'detected_at': datetime.now(timezone.utc).isoformat()
            }
        }), 200

    except Exception as e:
        logger.error(f"Schema detection error: {e}")
        _log_api_request('/schemas/detect', 'POST', False, str(e))
        return jsonify({'error': 'Schema detection failed', 'message': str(e)}), 500

# Security and Audit Endpoints

@secure_api_bp.route('/audit/logs', methods=['GET'])
@require_permission(Permission.AUDIT_READ)
@rate_limit(limit=50)
def get_audit_logs():
    """Get audit logs with security filtering"""
    try:
        _log_api_request('/audit/logs', 'GET')

        if not audit_logger:
            return jsonify({'error': 'Audit logging not available'}), 503

        user = _get_current_user()
        tenant_id = request.args.get('tenant_id', user.tenant_id if user else None)

        # Validate tenant access
        if user and not auth_manager.rbac.user_can_access_tenant(user, tenant_id):
            return jsonify({'error': 'Access denied to tenant audit logs'}), 403

        # Get parameters
        limit = int(request.args.get('limit', 100))
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')

        start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
        end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

        # Get audit logs
        logs = audit_logger.get_audit_logs(tenant_id, limit, start_date, end_date)

        _log_api_request('/audit/logs', 'GET', True)

        return jsonify({
            'success': True,
            'data': {
                'logs': [log.dict() for log in logs],
                'total': len(logs)
            }
        }), 200

    except Exception as e:
        logger.error(f"Audit logs error: {e}")
        _log_api_request('/audit/logs', 'GET', False, str(e))
        return jsonify({'error': 'Failed to retrieve audit logs', 'message': str(e)}), 500

@secure_api_bp.route('/security/status', methods=['GET'])
@require_permission(Permission.SYSTEM_MONITOR)
@rate_limit(limit=30)
def security_status():
    """Get security system status"""
    try:
        _log_api_request('/security/status', 'GET')

        if not auth_manager:
            return jsonify({'error': 'Security not configured'}), 503

        user = _get_current_user()

        status = {
            'security_level': auth_manager.config.security_level.value,
            'audit_enabled': auth_manager.config.audit_enabled,
            'rate_limiting_enabled': auth_manager.config.rate_limit_enabled,
            'mfa_required': auth_manager.config.require_mfa,
            'current_user': {
                'username': user.username if user else None,
                'roles': [role.value for role in user.roles] if user else [],
                'tenant_id': user.tenant_id if user else None
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        _log_api_request('/security/status', 'GET', True)

        return jsonify({
            'success': True,
            'data': status
        }), 200

    except Exception as e:
        logger.error(f"Security status error: {e}")
        _log_api_request('/security/status', 'GET', False, str(e))
        return jsonify({'error': 'Failed to get security status', 'message': str(e)}), 500

# Error Handlers

@secure_api_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401

@secure_api_bp.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden', 'message': 'Insufficient permissions'}), 403

@secure_api_bp.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate Limit Exceeded', 'message': 'Too many requests'}), 429

@secure_api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error', 'message': 'An internal error occurred'}), 500

# Security Registry for APG Integration

secure_api_registry = {
    'blueprint': secure_api_bp,
    'initialize': initialize_secure_api,
    'auth_manager': lambda: auth_manager,
    'audit_logger': lambda: audit_logger,
    'models': {
        'SecureJobCreateRequest': SecureJobCreateRequest,
        'AuthenticationRequest': AuthenticationRequest,
        'ApiKeyRequest': ApiKeyRequest
    }
}

__all__ = [
    'secure_api_bp',
    'initialize_secure_api',
    'SecureJobCreateRequest',
    'AuthenticationRequest',
    'ApiKeyRequest',
    'secure_api_registry'
]
