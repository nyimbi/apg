#!/usr/bin/env python3
"""
APG Key Management API
REST API endpoints following APG patterns

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from uuid_extensions import uuid7str

from .models import (
	KeySpec, Key, KeyOperation, AuditEvent, KeyUsageStats,
	KeyAlgorithm, KeyUsage, KeyState, SecurityLevel, create_key_spec_async
)
from .service import KeyManagementService, KeymService, create_key_management_service


# APG API Response Models
class APGResponse(dict):
	"""Standard APG API response format"""
	def __init__(self, success: bool = True, data: Any = None, error: Dict[str, Any] | None = None, 
				 metadata: Dict[str, Any] | None = None):
		super().__init__()
		self['success'] = success
		if data is not None:
			self['data'] = data
		if error:
			self['error'] = error
		self['metadata'] = metadata or {
			'request_id': uuid7str(),
			'timestamp': datetime.utcnow().isoformat(),
			'api_version': 'v1'
		}


# Security configuration
security = HTTPBearer()
app = FastAPI(title="APG Key Management API", version="1.0.0")


# Global service instance
keym_service: KeyManagementService | None = None
SERVICE = KeymService()


def _required_tenant_id(payload: Dict[str, Any]) -> str:
	tenant_id = str(payload.get("tenant_id") or "").strip()
	if not tenant_id:
		raise PermissionError("tenant_context_required")
	return tenant_id


def capability_status(tenant_id: str = "default") -> Dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"key_count": summary["key_count"],
		"operation_count": summary["operation_count"],
		"key_agent_count": summary["key_agent_count"],
		"denied_operation_count": summary["denied_operation_count"],
		"review_required_count": summary["review_required_count"],
		"pending_export_approval_count": summary["pending_export_approval_count"],
		"pending_rotation_exception_count": summary["pending_rotation_exception_count"],
		"scheduled_rotation_count": summary["scheduled_rotation_count"],
		"compromised_key_count": summary["compromised_key_count"],
	}


def create_managed_key(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.create_managed_key(
		tenant_id=_required_tenant_id(payload),
		key_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		algorithm=str(payload.get("algorithm") or "AES-256"),
		key_class=str(payload.get("key_class") or "data"),
		policy_ref=str(payload.get("policy_ref") or ""),
		hsm_attested=bool(payload.get("hsm_attested", False)),
		rotation_age_days=int(payload.get("rotation_age_days", 0) or 0),
		status=str(payload.get("status") or "active"),
	)


def evaluate_key_operation(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.evaluate_key_operation(
		tenant_id=_required_tenant_id(payload),
		operation_id=str(payload["id"]),
		key_id=str(payload["key_id"]),
		operation=str(payload.get("operation") or "use_key"),
	)


def request_export_approval(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.request_export_approval(
		tenant_id=_required_tenant_id(payload),
		approval_id=str(payload["id"]),
		key_id=str(payload["key_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
	)


def decide_export_approval(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.decide_export_approval(
		tenant_id=_required_tenant_id(payload),
		approval_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def request_rotation_exception(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.request_rotation_exception(
		tenant_id=_required_tenant_id(payload),
		exception_id=str(payload["id"]),
		key_id=str(payload["key_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
	)


def decide_rotation_exception(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.decide_rotation_exception(
		tenant_id=_required_tenant_id(payload),
		exception_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def schedule_rotation(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.schedule_rotation(
		tenant_id=_required_tenant_id(payload),
		rotation_id=str(payload["id"]),
		key_id=str(payload["key_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
	)


def complete_rotation(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.complete_rotation(
		tenant_id=_required_tenant_id(payload),
		rotation_id=str(payload["id"]),
		actor=str(payload.get("actor") or ""),
		evidence=str(payload.get("evidence") or ""),
	)


def mark_key_compromised(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.mark_key_compromised(
		tenant_id=_required_tenant_id(payload),
		key_id=str(payload["key_id"]),
		actor=str(payload.get("actor") or ""),
		evidence=str(payload.get("evidence") or ""),
	)


def register_key_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.register_key_agent(
		tenant_id=_required_tenant_id(payload),
		agent_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or ""),
		purpose=str(payload.get("purpose") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
		policy_ref=payload.get("policy_ref"),
	)


def validate_key_lifecycle_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.validate_key_lifecycle_batch(
		tenant_id=_required_tenant_id(payload),
		event_stream=str(payload.get("event_stream") or ""),
		mutation_count=int(payload.get("mutation_count") or 0),
	)


def create_record(payload: Dict[str, Any]) -> Dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=_required_tenant_id(payload),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[Dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_key_posture(tenant_id: str = "default") -> Dict[str, Any]:
	return {
		"keys": SERVICE.list_keys(tenant_id),
		"operations": SERVICE.list_operations(tenant_id),
		"export_approvals": SERVICE.list_export_approvals(tenant_id),
		"rotation_exceptions": SERVICE.list_rotation_exceptions(tenant_id),
		"rotations": SERVICE.list_rotations(tenant_id),
		"key_agents": SERVICE.list_key_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


async def get_service() -> KeyManagementService:
	"""Get key management service instance"""
	global keym_service
	if not keym_service:
		keym_service = await create_key_management_service()
	return keym_service


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
	"""Extract user ID from APG authentication token"""
	if not credentials:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Authentication required",
			headers={"WWW-Authenticate": "Bearer"}
		)
	
	try:
		# Parse JWT token
		import jwt
		from jwt.exceptions import InvalidTokenError
		
		token = credentials.credentials
		
		# In production, use proper JWT secret from configuration
		# For now, decode without verification for development
		try:
			payload = jwt.decode(token, options={"verify_signature": False})
			
			# Extract user ID from token payload
			user_id = payload.get('user_id') or payload.get('sub') or payload.get('username')
			
			if not user_id:
				raise HTTPException(
					status_code=status.HTTP_401_UNAUTHORIZED,
					detail="Invalid token: missing user identifier"
				)
			
			return str(user_id)
			
		except InvalidTokenError as e:
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail=f"Invalid token: {str(e)}"
			)
			
	except Exception as e:
		# Fallback for development - extract from basic auth or headers
		return "development_user"


async def get_tenant_id(request: Request) -> str:
	"""Extract tenant ID from APG request headers"""
	tenant_id = request.headers.get("X-Tenant-ID")
	if not tenant_id:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="X-Tenant-ID header required"
		)
	return tenant_id


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
	return JSONResponse(
		status_code=status.HTTP_400_BAD_REQUEST,
		content=APGResponse(success=False, error={
			'code': 'VALIDATION_ERROR',
			'message': str(exc),
			'request_id': uuid7str()
		})
	)


@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
	return JSONResponse(
		status_code=status.HTTP_403_FORBIDDEN,
		content=APGResponse(success=False, error={
			'code': 'PERMISSION_DENIED',
			'message': str(exc),
			'request_id': uuid7str()
		})
	)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content=APGResponse(success=False, error={
			'code': 'OPERATION_FAILED',
			'message': str(exc),
			'request_id': uuid7str()
		})
	)


# API Endpoints

@app.get("/health")
async def health_check():
	"""Health check endpoint"""
	service = await get_service()
	health = await service.get_service_health()
	return APGResponse(data=health)


@app.post("/keys")
async def create_key(
	key_request: Dict[str, Any],
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Create new cryptographic key"""
	try:
		# Create key specification
		spec = await create_key_spec_async(
			tenant_id=tenant_id,
			algorithm=KeyAlgorithm(key_request.get('algorithm', 'AES-256')),
			usage=[KeyUsage(u) for u in key_request.get('usage', ['encrypt', 'decrypt'])],
			name=key_request.get('name', 'Untitled Key'),
			created_by=user_id,
			description=key_request.get('description'),
			key_size=key_request.get('key_size'),
			security_level=SecurityLevel(key_request.get('security_level', 'internal'))
		)
		
		# Create key
		key = await service.create_key(spec, user_id)
		
		# Return key without sensitive material
		key_response = key.model_dump(exclude={'key_material', 'hsm_key_id'})
		
		return APGResponse(data={
			'key': key_response,
			'message': 'Key created successfully'
		})
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Key creation failed: {e}")


@app.get("/keys")
async def list_keys(
	algorithm: Optional[str] = None,
	state: Optional[str] = None,
	limit: int = 50,
	offset: int = 0,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""List keys for tenant"""
	try:
		# Build filters
		filters = {}
		if algorithm:
			filters['algorithm'] = KeyAlgorithm(algorithm)
		if state:
			filters['state'] = KeyState(state)
		
		# Get keys
		keys = await service.list_keys(tenant_id, user_id, filters)
		
		# Apply pagination
		paginated_keys = keys[offset:offset + limit]
		
		# Convert to response format
		keys_response = [key.model_dump(exclude={'key_material', 'hsm_key_id'}) 
						for key in paginated_keys]
		
		return APGResponse(data={
			'keys': keys_response,
			'pagination': {
				'total': len(keys),
				'limit': limit,
				'offset': offset,
				'has_more': offset + limit < len(keys)
			}
		})
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to list keys: {e}")


@app.get("/keys/{key_id}")
async def get_key(
	key_id: str,
	include_material: bool = False,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Get key details"""
	try:
		key = await service.retrieve_key(key_id, user_id, include_material)
		
		if not key:
			raise HTTPException(status_code=404, detail="Key not found")
		
		# Exclude sensitive material unless specifically requested and authorized
		exclude_fields = set()
		if not include_material:
			exclude_fields.update({'key_material', 'hsm_key_id'})
		
		key_response = key.model_dump(exclude=exclude_fields)
		
		return APGResponse(data={'key': key_response})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to retrieve key: {e}")


@app.put("/keys/{key_id}")
async def update_key(
	key_id: str,
	update_request: Dict[str, Any],
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Update key metadata and policies"""
	try:
		# Retrieve existing key
		key = await service.retrieve_key(key_id, user_id, include_material=False)
		
		if not key:
			raise HTTPException(status_code=404, detail="Key not found")
		
		# Update allowed fields
		if 'metadata' in update_request:
			metadata_update = update_request['metadata']
			if 'name' in metadata_update:
				key.spec.metadata.name = metadata_update['name']
			if 'description' in metadata_update:
				key.spec.metadata.description = metadata_update['description']
			if 'tags' in metadata_update:
				key.spec.metadata.tags.update(metadata_update['tags'])
		
		if 'policy' in update_request:
			policy_update = update_request['policy']
			if 'rotation_interval_days' in policy_update:
				key.spec.policy.rotation_interval_days = policy_update['rotation_interval_days']
			if 'auto_rotate' in policy_update:
				key.spec.policy.auto_rotate = policy_update['auto_rotate']
		
		# Update timestamp
		key.spec.updated_at = datetime.utcnow()
		
		# Return updated key
		key_response = key.model_dump(exclude={'key_material', 'hsm_key_id'})
		
		return APGResponse(data={
			'key': key_response,
			'message': 'Key updated successfully'
		})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to update key: {e}")


@app.delete("/keys/{key_id}")
async def delete_key(
	key_id: str,
	secure_delete: bool = True,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Delete cryptographic key"""
	try:
		success = await service.delete_key(key_id, user_id, secure_delete)
		
		if not success:
			raise HTTPException(status_code=404, detail="Key not found")
		
		return APGResponse(data={
			'key_id': key_id,
			'deleted': True,
			'secure_delete': secure_delete,
			'message': 'Key deleted successfully'
		})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to delete key: {e}")


@app.post("/keys/{key_id}/rotate")
async def rotate_key(
	key_id: str,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Rotate cryptographic key"""
	try:
		key = await service.rotate_key(key_id, user_id)
		
		# Return rotated key without sensitive material
		key_response = key.model_dump(exclude={'key_material', 'hsm_key_id'})
		
		return APGResponse(data={
			'key': key_response,
			'message': 'Key rotated successfully',
			'previous_versions': len(key.previous_versions)
		})
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Key rotation failed: {e}")


@app.post("/keys/{key_id}/encrypt")
async def encrypt_data(
	key_id: str,
	encrypt_request: Dict[str, Any],
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Encrypt data using specified key"""
	try:
		# Get data to encrypt (expected as base64 encoded)
		import base64
		data_b64 = encrypt_request.get('data')
		if not data_b64:
			raise HTTPException(status_code=400, detail="Data required")
		
		try:
			data = base64.b64decode(data_b64)
		except Exception:
			raise HTTPException(status_code=400, detail="Invalid base64 data")
		
		# Encrypt data
		encrypted_data = await service.encrypt_data(
			key_id, 
			data, 
			user_id,
			encrypt_request.get('parameters', {})
		)
		
		# Return encrypted data as base64
		encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
		
		return APGResponse(data={
			'key_id': key_id,
			'encrypted_data': encrypted_b64,
			'original_size': len(data),
			'encrypted_size': len(encrypted_data),
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Encryption failed: {e}")


@app.post("/keys/{key_id}/decrypt")
async def decrypt_data(
	key_id: str,
	decrypt_request: Dict[str, Any],
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Decrypt data using specified key"""
	try:
		# Get encrypted data (expected as base64 encoded)
		import base64
		encrypted_data_b64 = decrypt_request.get('encrypted_data')
		if not encrypted_data_b64:
			raise HTTPException(status_code=400, detail="Encrypted data required")
		
		try:
			encrypted_data = base64.b64decode(encrypted_data_b64)
		except Exception:
			raise HTTPException(status_code=400, detail="Invalid base64 encrypted data")
		
		# Decrypt data
		decrypted_data = await service.decrypt_data(
			key_id,
			encrypted_data,
			user_id,
			decrypt_request.get('parameters', {})
		)
		
		# Return decrypted data as base64
		decrypted_b64 = base64.b64encode(decrypted_data).decode('utf-8')
		
		return APGResponse(data={
			'key_id': key_id,
			'decrypted_data': decrypted_b64,
			'encrypted_size': len(encrypted_data),
			'decrypted_size': len(decrypted_data),
			'timestamp': datetime.utcnow().isoformat()
		})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Decryption failed: {e}")


@app.get("/keys/{key_id}/stats")
async def get_key_stats(
	key_id: str,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user)
):
	"""Get key usage statistics"""
	try:
		stats = await service.get_key_usage_stats(key_id, user_id)
		
		if not stats:
			raise HTTPException(status_code=404, detail="Key or statistics not found")
		
		stats_response = stats.model_dump()
		
		return APGResponse(data={'statistics': stats_response})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {e}")


@app.get("/audit")
async def get_audit_logs(
	event_type: Optional[str] = None,
	resource_id: Optional[str] = None,
	user_id_filter: Optional[str] = None,
	limit: int = 100,
	offset: int = 0,
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get audit logs for tenant"""
	try:
		# Build filters
		filters = {}
		if event_type:
			filters['event_type'] = event_type
		if resource_id:
			filters['resource_id'] = resource_id
		if user_id_filter:
			filters['user_id'] = user_id_filter
		
		# Get audit events
		events = await service.get_audit_events(tenant_id, user_id, filters)
		
		# Apply pagination
		paginated_events = events[offset:offset + limit]
		
		# Convert to response format
		events_response = [event.model_dump() for event in paginated_events]
		
		return APGResponse(data={
			'events': events_response,
			'pagination': {
				'total': len(events),
				'limit': limit,
				'offset': offset,
				'has_more': offset + limit < len(events)
			}
		})
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to retrieve audit logs: {e}")


@app.get("/metrics")
async def get_metrics(
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Get key management metrics"""
	try:
		# Get service health which includes metrics
		health = await service.get_service_health()
		
		# Get tenant-specific keys for more detailed metrics
		keys = await service.list_keys(tenant_id, user_id)
		
		# Calculate additional metrics
		algorithm_distribution = {}
		state_distribution = {}
		
		for key in keys:
			algo = key.spec.algorithm.value
			state = key.spec.state.value
			
			algorithm_distribution[algo] = algorithm_distribution.get(algo, 0) + 1
			state_distribution[state] = state_distribution.get(state, 0) + 1
		
		metrics = {
			'service_health': health,
			'tenant_metrics': {
				'total_keys': len(keys),
				'algorithm_distribution': algorithm_distribution,
				'state_distribution': state_distribution
			},
			'timestamp': datetime.utcnow().isoformat()
		}
		
		return APGResponse(data=metrics)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {e}")


# Batch operations
@app.post("/keys/batch")
async def batch_create_keys(
	batch_request: Dict[str, Any],
	service: KeyManagementService = Depends(get_service),
	user_id: str = Depends(get_current_user),
	tenant_id: str = Depends(get_tenant_id)
):
	"""Create multiple keys in batch"""
	try:
		key_requests = batch_request.get('keys', [])
		if not key_requests:
			raise HTTPException(status_code=400, detail="No keys specified for batch creation")
		
		results = []
		
		# Process each key request
		for i, key_request in enumerate(key_requests):
			try:
				# Create key specification
				spec = await create_key_spec_async(
					tenant_id=tenant_id,
					algorithm=KeyAlgorithm(key_request.get('algorithm', 'AES-256')),
					usage=[KeyUsage(u) for u in key_request.get('usage', ['encrypt', 'decrypt'])],
					name=key_request.get('name', f'Batch Key {i+1}'),
					created_by=user_id,
					description=key_request.get('description'),
					key_size=key_request.get('key_size')
				)
				
				# Create key
				key = await service.create_key(spec, user_id)
				
				results.append({
					'index': i,
					'success': True,
					'key_id': key.spec.id,
					'key': key.model_dump(exclude={'key_material', 'hsm_key_id'})
				})
				
			except Exception as e:
				results.append({
					'index': i,
					'success': False,
					'error': str(e)
				})
		
		# Calculate summary
		successful = len([r for r in results if r['success']])
		failed = len(results) - successful
		
		return APGResponse(data={
			'results': results,
			'summary': {
				'total_requests': len(key_requests),
				'successful': successful,
				'failed': failed,
				'success_rate': successful / len(key_requests) * 100 if key_requests else 0
			}
		})
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Batch key creation failed: {e}")


# Export FastAPI app
__all__ = ["app", "APGResponse"]
