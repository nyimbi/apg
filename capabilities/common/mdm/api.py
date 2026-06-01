#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - RESTful API Layer
Advanced RESTful API with GraphQL support and APG ecosystem integration

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from functools import wraps
import traceback

from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, validator
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import strawberry
from strawberry.fastapi import GraphQLRouter

from .service import (
    MDMService,
    MDMOperationType,
    MDMOperationContext,
    MdmCrossReferenceRecord,
    MdmDataAgentRecord,
    MdmDuplicateCandidateRecord,
    MdmEntityRecord,
    MdmGoldenRecord,
    MdmLifecycleBatchRecord,
    MdmMergeRequestRecord,
    MdmPublishRecord,
    MdmQualityRecord,
    MdmService,
)
from .models import (
    MdEntityCreate, MdEntityUpdate, MdDataQualityScore, MdDuplicateDetectionResult,
    EntityType, EntityStatus, DataQualityStatus
)


SERVICE = MdmService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
    contract = SERVICE.describe(tenant_id)
    return {
        "capability": contract["capability"],
        "display_name": contract["display_name"],
        "tenant_id": tenant_id,
        "route_count": len(contract["ui"]["routes"]),
        "rule_count": len(contract["rule_engine"]["rules"]),
        "record_count": len(SERVICE.list_records(tenant_id)),
    }


def register_entity_record(**kwargs: Any) -> MdmEntityRecord:
    return SERVICE.register_entity(**kwargs)


def assess_quality_record(**kwargs: Any) -> MdmQualityRecord:
    return SERVICE.assess_quality(**kwargs)


def create_duplicate_candidate_record(**kwargs: Any) -> MdmDuplicateCandidateRecord:
    return SERVICE.create_duplicate_candidate(**kwargs)


def review_duplicate_candidate_record(**kwargs: Any) -> MdmDuplicateCandidateRecord:
    return SERVICE.review_duplicate_candidate(**kwargs)


def create_golden_record(**kwargs: Any) -> MdmGoldenRecord:
    return SERVICE.create_golden_record(**kwargs)


def merge_golden_record(**kwargs: Any) -> MdmMergeRequestRecord:
    return SERVICE.merge_golden_record(**kwargs)


def update_cross_reference_record(**kwargs: Any) -> MdmCrossReferenceRecord:
    return SERVICE.update_cross_reference(**kwargs)


def publish_entity_record(**kwargs: Any) -> MdmPublishRecord:
    return SERVICE.publish_entity(**kwargs)


def register_data_agent(**kwargs: Any) -> MdmDataAgentRecord:
    return SERVICE.register_data_agent(**kwargs)


def validate_mdm_lifecycle_batch(**kwargs: Any) -> MdmLifecycleBatchRecord:
    return SERVICE.validate_mdm_lifecycle_batch(**kwargs)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
    return SERVICE.create_record(
        record_id=str(payload["id"]),
        tenant_id=str(payload.get("tenant_id") or "default"),
        metadata=dict(payload.get("metadata") or {}),
        status=str(payload.get("status") or "active"),
    )


def list_records(tenant_id: str | None = None, record_type: str | None = None) -> list[dict[str, Any]]:
    return SERVICE.list_records(tenant_id, record_type)


def list_pending_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
    return SERVICE.list_pending_reviews(tenant_id)


def list_mdm(tenant_id: str | None = None) -> dict[str, Any]:
    return {
        "summary": SERVICE.dashboard_summary(tenant_id),
        "entities": SERVICE.list_records(tenant_id, "entities"),
        "quality_assessments": SERVICE.list_records(tenant_id, "quality_assessments"),
        "duplicate_candidates": SERVICE.list_records(tenant_id, "duplicate_candidates"),
        "golden_records": SERVICE.list_records(tenant_id, "golden_records"),
        "merge_requests": SERVICE.list_records(tenant_id, "merge_requests"),
        "cross_references": SERVICE.list_records(tenant_id, "cross_references"),
        "publish_records": SERVICE.list_records(tenant_id, "publish_records"),
        "data_agents": SERVICE.list_records(tenant_id, "data_agents"),
        "lifecycle_batches": SERVICE.list_records(tenant_id, "lifecycle_batches"),
        "pending_reviews": SERVICE.list_pending_reviews(tenant_id),
        "audit_events": SERVICE.list_records(tenant_id, "audit_events"),
    }


# API Request/Response Models with Pydantic v2

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class EntityCreateRequest(MdEntityCreate):
    """Entity creation API request"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)


class EntityUpdateRequest(MdEntityUpdate):
    """Entity update API request"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)


class EntitySearchRequest(BaseModel):
    """Entity search API request"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    entity_type: Optional[EntityType] = None
    entity_name: Optional[str] = Field(None, max_length=255)
    business_key: Optional[str] = Field(None, max_length=100)
    source_system: Optional[str] = Field(None, max_length=100)
    status: Optional[EntityStatus] = None
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    is_golden_record: Optional[bool] = None
    data_classification: Optional[str] = Field(None, max_length=50)
    created_after: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    attributes: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    sort_by: str = Field("updated_at", max_length=50)
    sort_order: str = Field("desc", regex="^(asc|desc)$")
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class BulkOperationRequest(BaseModel):
    """Bulk operation API request"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    operation_type: str = Field(..., regex="^(create|update|delete)$")
    entities: List[Dict[str, Any]] = Field(..., min_items=1, max_items=1000)
    batch_options: Optional[Dict[str, Any]] = None


class QualityAssessmentRequest(BaseModel):
    """Quality assessment API request"""
    model_config = ConfigDict(extra='forbid', validate_by_name=True)
    
    entity_ids: List[str] = Field(..., min_items=1, max_items=100)
    include_recommendations: bool = Field(True)
    include_issues: bool = Field(True)


# Authentication and Authorization

class MDMSecurityManager:
    """APG-integrated security manager for MDM API"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bearer_scheme = HTTPBearer(auto_error=False)
        self.rate_limits = {
            'default': {'requests': 1000, 'window': 3600},  # 1000 req/hour
            'premium': {'requests': 5000, 'window': 3600},   # 5000 req/hour
            'enterprise': {'requests': 20000, 'window': 3600} # 20000 req/hour
        }
        self.request_tracking = {}  # In production, use Redis
    
    async def authenticate_request(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Authenticate API request using APG auth integration"""
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            # In production, integrate with APG auth capability
            token = credentials.credentials
            
            # Validate token with APG auth service integration
            user_context = await self._validate_token(token)
            
            if not user_context:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            return user_context
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
    
    async def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token with APG auth service"""
        try:
            # Integrate with APG auth capability for token validation
            import jwt
            from cryptography.hazmat.primitives import serialization
            import aiohttp
            import os
            
            # Get APG auth service endpoint from environment
            auth_service_url = os.getenv('APG_AUTH_SERVICE_URL', 'http://localhost:8080/auth')
            
            # Option 1: Validate locally with shared secret (faster)
            jwt_secret = os.getenv('APG_JWT_SECRET')
            if jwt_secret:
                try:
                    payload = jwt.decode(token, jwt_secret, algorithms=['HS256'])
                    
                    # Validate token structure and claims
                    required_claims = ['user_id', 'tenant_id', 'permissions', 'exp']
                    for claim in required_claims:
                        if claim not in payload:
                            return None
                    
                    # Check token expiration
                    import time
                    if payload.get('exp', 0) < time.time():
                        return None
                    
                    return {
                        'user_id': payload['user_id'],
                        'tenant_id': payload['tenant_id'], 
                        'permissions': payload.get('permissions', []),
                        'tier': payload.get('tier', 'standard'),
                        'email': payload.get('email'),
                        'name': payload.get('name'),
                        'roles': payload.get('roles', [])
                    }
                except jwt.InvalidTokenError:
                    pass
            
            # Option 2: Validate with APG auth service (fallback)
            async with aiohttp.ClientSession() as session:
                validation_url = f"{auth_service_url}/validate"
                headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
                
                async with session.post(validation_url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        validation_result = await response.json()
                        
                        if validation_result.get('valid', False):
                            user_context = validation_result.get('user_context', {})
                            
                            # Ensure required fields are present
                            if user_context.get('user_id') and user_context.get('tenant_id'):
                                return {
                                    'user_id': user_context['user_id'],
                                    'tenant_id': user_context['tenant_id'],
                                    'permissions': user_context.get('permissions', []),
                                    'tier': user_context.get('tier', 'standard'),
                                    'email': user_context.get('email'),
                                    'name': user_context.get('name'),
                                    'roles': user_context.get('roles', [])
                                }
            
            return None
            
        except Exception as e:
            # Log validation error but don't expose details
            print(f"Token validation error: {str(e)}")
            return None
    
    async def check_permissions(self, user_context: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = user_context.get('permissions', [])
        return required_permission in user_permissions or 'mdm.admin' in user_permissions
    
    async def rate_limit_check(self, request: Request, user_context: Dict[str, Any]) -> None:
        """Check rate limiting based on user tier"""
        client_id = f"{user_context['user_id']}:{user_context['tenant_id']}"
        user_tier = user_context.get('tier', 'default')
        
        rate_config = self.rate_limits.get(user_tier, self.rate_limits['default'])
        
        # Simple in-memory rate limiting (use Redis in production)
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=rate_config['window'])
        
        if client_id not in self.request_tracking:
            self.request_tracking[client_id] = []
        
        # Clean old requests
        self.request_tracking[client_id] = [
            req_time for req_time in self.request_tracking[client_id] 
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.request_tracking[client_id]) >= rate_config['requests']:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Add current request
        self.request_tracking[client_id].append(now)


# GraphQL Schema for advanced querying

@strawberry.type
class EntityGraphQL:
    """GraphQL entity type"""
    entity_id: str
    entity_type: str
    entity_name: str
    business_key: str
    source_system: str
    status: str
    quality_score: float
    is_golden_record: bool
    created_at: str
    updated_at: str
    attributes: strawberry.scalars.JSON
    tags: List[str]


@strawberry.type
class QualityAssessmentGraphQL:
    """GraphQL quality assessment type"""
    entity_id: str
    overall_score: float
    quality_status: str
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    assessment_timestamp: str


@strawberry.type
class Query:
    """GraphQL query root"""
    
    @strawberry.field
    async def entity(self, entity_id: str, info: strawberry.Info) -> Optional[EntityGraphQL]:
        """Get single entity by ID"""
        # Extract context from GraphQL info
        request = info.context["request"]
        mdm_service = request.app.state.mdm_service
        
        # Authentication would be handled by middleware
        tenant_id = "test_tenant"  # Extract from context
        
        result = await mdm_service.entity_service.get_entity(
            entity_id, tenant_id, include_quality=True
        )
        
        if result['status'] == 'success':
            entity_data = result['entity']
            return EntityGraphQL(
                entity_id=entity_data['entity_id'],
                entity_type=entity_data['entity_type'],
                entity_name=entity_data['entity_name'],
                business_key=entity_data['business_key'],
                source_system=entity_data['source_system'],
                status=entity_data['status'],
                quality_score=entity_data['quality_score'],
                is_golden_record=entity_data['is_golden_record'],
                created_at=entity_data['created_at'],
                updated_at=entity_data['updated_at'],
                attributes=entity_data['attributes'],
                tags=entity_data['tags']
            )
        return None
    
    @strawberry.field
    async def entities(self, limit: int = 50, offset: int = 0, 
                      entity_type: Optional[str] = None,
                      info: strawberry.Info = None) -> List[EntityGraphQL]:
        """Search entities with filtering"""
        request = info.context["request"]
        mdm_service = request.app.state.mdm_service
        
        tenant_id = "test_tenant"  # Extract from context
        
        search_criteria = {'limit': limit, 'offset': offset}
        if entity_type:
            search_criteria['entity_type'] = entity_type
        
        result = await mdm_service.entity_service.search_entities(
            tenant_id, search_criteria
        )
        
        entities = []
        if result['status'] == 'success':
            for entity_data in result['entities']:
                entities.append(EntityGraphQL(
                    entity_id=entity_data['entity_id'],
                    entity_type=entity_data['entity_type'],
                    entity_name=entity_data['entity_name'],
                    business_key=entity_data['business_key'],
                    source_system=entity_data['source_system'],
                    status=entity_data['status'],
                    quality_score=entity_data['quality_score'],
                    is_golden_record=entity_data['is_golden_record'],
                    created_at=entity_data['created_at'],
                    updated_at=entity_data['updated_at'],
                    attributes=entity_data.get('attributes', {}),
                    tags=entity_data.get('tags', [])
                ))
        
        return entities


@strawberry.type
class Mutation:
    """GraphQL mutation root"""
    
    @strawberry.field
    async def create_entity(self, entity_data: strawberry.scalars.JSON,
                           info: strawberry.Info) -> str:
        """Create new entity via GraphQL"""
        request = info.context["request"]
        mdm_service = request.app.state.mdm_service
        
        # Would extract from authenticated context
        context = MDMOperationContext(
            tenant_id="test_tenant",
            user_id="test_user",
            operation_type=MDMOperationType.CREATE_ENTITY
        )
        
        # Convert JSON to EntityCreate model
        entity_create = MdEntityCreate(**entity_data)
        
        result = await mdm_service.entity_service.create_entity(entity_create, context)
        
        if result['status'] == 'success':
            return result['entity_id']
        else:
            raise Exception(result['message'])


# Main API Application

class MDMAPI:
    """Main MDM API application with comprehensive endpoints"""
    
    def __init__(self, mdm_service: MDMService, config: Dict[str, Any] = None):
        self.mdm_service = mdm_service
        self.config = config or {}
        self.security_manager = MDMSecurityManager(config)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="APG Master Data Management API",
            description="Advanced multi-tenant MDM with AI-enhanced data quality",
            version="1.0.0",
            docs_url="/api/v1/docs",
            redoc_url="/api/v1/redoc"
        )
        
        # Store service in app state
        self.app.state.mdm_service = mdm_service
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup GraphQL
        self._setup_graphql()
    
    def _setup_middleware(self):
        """Configure API middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("allowed_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.get("allowed_hosts", ["*"])
        )
        
        # Custom request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = datetime.utcnow()
            
            # Generate request ID
            request_id = f"mdm-{int(start_time.timestamp() * 1000000)}"
            request.state.request_id = request_id
            
            # Process request
            response = await call_next(request)
            
            # Log request details
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            print(f"[MDM-API] {request.method} {request.url.path} - "
                  f"Status: {response.status_code} - "
                  f"Time: {process_time:.3f}s - "
                  f"RequestID: {request_id}")
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """System health check"""
            health_result = await self.mdm_service.health_check()
            
            if health_result['status'] == 'healthy':
                return APIResponse(
                    success=True,
                    message="MDM service is healthy",
                    data=health_result
                )
            else:
                return APIResponse(
                    success=False,
                    message="MDM service health issues detected",
                    data=health_result
                ), 503
        
        # Entity management endpoints
        @self.app.post("/api/v1/entities", response_model=APIResponse)
        async def create_entity(
            request: Request,
            entity_data: EntityCreateRequest,
            background_tasks: BackgroundTasks,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Create new master data entity"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.write'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Create operation context
                context = self.mdm_service.create_operation_context(
                    tenant_id=user_context['tenant_id'],
                    user_id=user_context['user_id'],
                    operation_type=MDMOperationType.CREATE_ENTITY,
                    source_system="api",
                    client_ip=request.client.host,
                    user_agent=request.headers.get('user-agent')
                )
                
                # Create entity
                result = await self.mdm_service.entity_service.create_entity(entity_data, context)
                
                if result['status'] == 'success':
                    return APIResponse(
                        success=True,
                        message="Entity created successfully",
                        data=result,
                        request_id=request.state.request_id
                    )
                else:
                    return APIResponse(
                        success=False,
                        message=result['message'],
                        errors=[result['message']],
                        request_id=request.state.request_id
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Create entity error: {str(e)}")
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/api/v1/entities/{entity_id}", response_model=APIResponse)
        async def get_entity(
            entity_id: str,
            request: Request,
            include_versions: bool = False,
            include_quality: bool = False,
            include_cross_refs: bool = False,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Get entity by ID with optional related data"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.read'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Get entity
                result = await self.mdm_service.entity_service.get_entity(
                    entity_id, user_context['tenant_id'],
                    include_versions=include_versions,
                    include_quality=include_quality,
                    include_cross_refs=include_cross_refs
                )
                
                if result['status'] == 'success':
                    return APIResponse(
                        success=True,
                        message="Entity retrieved successfully",
                        data=result,
                        request_id=request.state.request_id
                    )
                else:
                    if 'not found' in result['message'].lower():
                        raise HTTPException(status_code=404, detail=result['message'])
                    else:
                        return APIResponse(
                            success=False,
                            message=result['message'],
                            errors=[result['message']],
                            request_id=request.state.request_id
                        )
                        
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Get entity error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.put("/api/v1/entities/{entity_id}", response_model=APIResponse)
        async def update_entity(
            entity_id: str,
            entity_updates: EntityUpdateRequest,
            request: Request,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Update existing entity"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.write'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Create operation context
                context = self.mdm_service.create_operation_context(
                    tenant_id=user_context['tenant_id'],
                    user_id=user_context['user_id'],
                    operation_type=MDMOperationType.UPDATE_ENTITY,
                    entity_id=entity_id,
                    source_system="api",
                    client_ip=request.client.host,
                    user_agent=request.headers.get('user-agent')
                )
                
                # Update entity
                result = await self.mdm_service.entity_service.update_entity(
                    entity_id, entity_updates, context
                )
                
                if result['status'] == 'success':
                    return APIResponse(
                        success=True,
                        message="Entity updated successfully",
                        data=result,
                        request_id=request.state.request_id
                    )
                else:
                    if 'not found' in result['message'].lower():
                        raise HTTPException(status_code=404, detail=result['message'])
                    else:
                        return APIResponse(
                            success=False,
                            message=result['message'],
                            errors=[result['message']],
                            request_id=request.state.request_id
                        )
                        
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Update entity error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.delete("/api/v1/entities/{entity_id}", response_model=APIResponse)
        async def delete_entity(
            entity_id: str,
            request: Request,
            soft_delete: bool = True,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Delete entity (soft or hard delete)"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.write'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Create operation context
                context = self.mdm_service.create_operation_context(
                    tenant_id=user_context['tenant_id'],
                    user_id=user_context['user_id'],
                    operation_type=MDMOperationType.DELETE_ENTITY,
                    entity_id=entity_id,
                    source_system="api",
                    client_ip=request.client.host,
                    user_agent=request.headers.get('user-agent')
                )
                
                # Delete entity
                result = await self.mdm_service.entity_service.delete_entity(
                    entity_id, context, soft_delete=soft_delete
                )
                
                if result['status'] == 'success':
                    return APIResponse(
                        success=True,
                        message="Entity deleted successfully",
                        data=result,
                        request_id=request.state.request_id
                    )
                else:
                    if 'not found' in result['message'].lower():
                        raise HTTPException(status_code=404, detail=result['message'])
                    else:
                        return APIResponse(
                            success=False,
                            message=result['message'],
                            errors=[result['message']],
                            request_id=request.state.request_id
                        )
                        
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Delete entity error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/api/v1/entities/search", response_model=APIResponse)
        async def search_entities(
            search_request: EntitySearchRequest,
            request: Request,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Advanced entity search with filtering and pagination"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.read'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Convert request to search criteria
                search_criteria = search_request.dict(exclude_unset=True)
                
                # Search entities
                result = await self.mdm_service.entity_service.search_entities(
                    user_context['tenant_id'], search_criteria,
                    limit=search_request.limit, offset=search_request.offset
                )
                
                if result['status'] == 'success':
                    return APIResponse(
                        success=True,
                        message=f"Found {len(result['entities'])} entities",
                        data=result,
                        request_id=request.state.request_id
                    )
                else:
                    return APIResponse(
                        success=False,
                        message=result['message'],
                        errors=[result['message']],
                        request_id=request.state.request_id
                    )
                    
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Search entities error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Quality assessment endpoints
        @self.app.post("/api/v1/quality/assess", response_model=APIResponse)
        async def assess_quality(
            assessment_request: QualityAssessmentRequest,
            request: Request,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Assess data quality for multiple entities"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.read'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting
                await self.security_manager.rate_limit_check(request, user_context)
                
                quality_results = []
                
                for entity_id in assessment_request.entity_ids:
                    # Get entity first
                    entity_result = await self.mdm_service.entity_service.get_entity(
                        entity_id, user_context['tenant_id']
                    )
                    
                    if entity_result['status'] == 'success':
                        entity_data = entity_result['entity']
                        
                        # Assess quality
                        quality_result = await self.mdm_service.quality_service.assess_quality(
                            entity_id, user_context['tenant_id'],
                            entity_data['attributes'], entity_data['entity_type']
                        )
                        
                        quality_results.append(quality_result)
                    else:
                        quality_results.append({
                            'entity_id': entity_id,
                            'error': entity_result['message']
                        })
                
                return APIResponse(
                    success=True,
                    message=f"Quality assessed for {len(quality_results)} entities",
                    data={'quality_assessments': quality_results},
                    request_id=request.state.request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Quality assessment error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        # Bulk operations endpoint
        @self.app.post("/api/v1/entities/bulk", response_model=APIResponse)
        async def bulk_operations(
            bulk_request: BulkOperationRequest,
            request: Request,
            background_tasks: BackgroundTasks,
            user_context: Dict[str, Any] = Depends(self.security_manager.authenticate_request)
        ):
            """Perform bulk operations on multiple entities"""
            try:
                # Check permissions
                if not await self.security_manager.check_permissions(user_context, 'mdm.write'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Rate limiting (stricter for bulk operations)
                await self.security_manager.rate_limit_check(request, user_context)
                
                # Process bulk operations asynchronously
                async def process_bulk_operations():
                    results = []
                    for i, entity_data in enumerate(bulk_request.entities):
                        try:
                            if bulk_request.operation_type == 'create':
                                entity_create = MdEntityCreate(**entity_data)
                                context = self.mdm_service.create_operation_context(
                                    tenant_id=user_context['tenant_id'],
                                    user_id=user_context['user_id'],
                                    operation_type=MDMOperationType.CREATE_ENTITY
                                )
                                result = await self.mdm_service.entity_service.create_entity(
                                    entity_create, context
                                )
                            elif bulk_request.operation_type == 'update':
                                # Implementation for bulk update
                                if 'entity_id' not in entity_data:
                                    result = {'status': 'error', 'message': 'entity_id required for update operations'}
                                else:
                                    entity_id = entity_data.pop('entity_id')
                                    entity_update = MdEntityUpdate(**entity_data)
                                    context = self.mdm_service.create_operation_context(
                                        tenant_id=user_context['tenant_id'],
                                        user_id=user_context['user_id'],
                                        operation_type=MDMOperationType.UPDATE_ENTITY,
                                        entity_id=entity_id
                                    )
                                    result = await self.mdm_service.entity_service.update_entity(
                                        entity_id, user_context['tenant_id'], entity_update, context
                                    )
                            elif bulk_request.operation_type == 'delete':
                                # Implementation for bulk delete
                                if 'entity_id' not in entity_data:
                                    result = {'status': 'error', 'message': 'entity_id required for delete operations'}
                                else:
                                    entity_id = entity_data['entity_id']
                                    context = self.mdm_service.create_operation_context(
                                        tenant_id=user_context['tenant_id'],
                                        user_id=user_context['user_id'],
                                        operation_type=MDMOperationType.DELETE_ENTITY,
                                        entity_id=entity_id
                                    )
                                    result = await self.mdm_service.entity_service.delete_entity(
                                        entity_id, user_context['tenant_id'], context
                                    )
                            
                            results.append({
                                'index': i,
                                'entity_data': entity_data,
                                'result': result
                            })
                            
                        except Exception as e:
                            results.append({
                                'index': i,
                                'entity_data': entity_data,
                                'result': {'status': 'error', 'message': str(e)}
                            })
                    
                    return results
                
                # Run bulk operations in background
                background_tasks.add_task(process_bulk_operations)
                
                return APIResponse(
                    success=True,
                    message=f"Bulk {bulk_request.operation_type} operation initiated for {len(bulk_request.entities)} entities",
                    data={'operation_id': request.state.request_id},
                    request_id=request.state.request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"[MDM-API] Bulk operations error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    def _setup_graphql(self):
        """Setup GraphQL endpoint"""
        schema = strawberry.Schema(query=Query, mutation=Mutation)
        graphql_app = GraphQLRouter(schema, context_getter=lambda request: {"request": request})
        
        self.app.include_router(graphql_app, prefix="/api/v1/graphql")
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app


# Factory function for creating the API application
def create_mdm_api(mdm_service: MDMService, config: Dict[str, Any] = None) -> FastAPI:
    """Create and configure the MDM API application"""
    mdm_api = MDMAPI(mdm_service, config)
    return mdm_api.get_app()


# Export main classes
__all__ = [
    'MDMAPI', 'create_mdm_api', 'MDMSecurityManager',
    'APIResponse', 'EntityCreateRequest', 'EntityUpdateRequest', 'EntitySearchRequest',
    'BulkOperationRequest', 'QualityAssessmentRequest'
]
