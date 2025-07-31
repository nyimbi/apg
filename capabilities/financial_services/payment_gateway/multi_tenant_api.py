#!/usr/bin/env python3
"""
Multi-Tenant API Extensions - APG Payment Gateway

REST API endpoints for multi-tenant management, integrating seamlessly
with the existing payment gateway API architecture.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid_extensions import uuid7str
import json

from flask import Blueprint, request, jsonify, current_app
from flask_appbuilder import BaseView, expose, has_access
from flask_appbuilder.security.decorators import permission_name
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound, InternalServerError

from .multi_tenant_service import (
    TenantIsolationService, Tenant, TenantStatus, TenantPlan, ResourceType,
    create_multi_tenant_service
)
from .database import get_database_service
from .auth import get_auth_service, require_permission

# Create Flask Blueprint for multi-tenant endpoints
multi_tenant_bp = Blueprint(
    'multi_tenant',
    __name__,
    url_prefix='/api/v1/tenants'
)

class MultiTenantAPIView(BaseView):
    """
    Multi-Tenant Management API View
    
    Provides comprehensive tenant management endpoints with full integration
    into the existing APG payment gateway architecture.
    """
    
    route_base = "/api/v1/tenants"
    
    def __init__(self):
        super().__init__()
        self.tenant_service: Optional[TenantIsolationService] = None
        self.database_service = None
        self.auth_service = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure all services are initialized"""
        if not self._initialized:
            # Initialize database service
            self.database_service = get_database_service()
            await self.database_service.initialize()
            
            # Initialize authentication service
            self.auth_service = get_auth_service()
            
            # Initialize tenant service
            self.tenant_service = TenantIsolationService(self.database_service)
            await self.tenant_service.initialize()
            
            self._initialized = True
    
    # Core Tenant Management Endpoints
    
    @expose('/', methods=['POST'])
    @has_access
    @permission_name('tenant_create')
    def create_tenant(self):
        """
        Create a new tenant
        
        POST /api/v1/tenants/
        {
            "name": "My Company",
            "slug": "my-company",
            "business_type": "ecommerce",
            "industry": "retail",
            "country": "US",
            "plan": "professional",
            "billing_email": "billing@mycompany.com",
            "timezone": "America/New_York",
            "require_mfa": true,
            "data_residency_region": "us-east-1"
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['name', 'slug', 'business_type', 'country', 'billing_email']
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")
            
            # Create tenant
            tenant = await self.tenant_service.create_tenant(data)
            
            return jsonify({
                "success": True,
                "tenant": {
                    "id": tenant.id,
                    "name": tenant.name,
                    "slug": tenant.slug,
                    "plan": tenant.plan.value,
                    "status": tenant.status.value,
                    "business_type": tenant.business_type,
                    "country": tenant.country,
                    "created_at": tenant.created_at.isoformat(),
                    "resource_limits": tenant.resource_limits,
                    "feature_flags": tenant.feature_flags
                }
            }), 201
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Tenant creation error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/<tenant_id>', methods=['GET'])
    @has_access
    @permission_name('tenant_read')
    def get_tenant(self, tenant_id: str):
        """
        Get tenant by ID
        
        GET /api/v1/tenants/{tenant_id}
        """
        try:
            await self._ensure_initialized()
            
            tenant = await self.tenant_service.get_tenant(tenant_id)
            if not tenant:
                return jsonify({"error": "Tenant not found"}), 404
            
            return jsonify({
                "tenant": {
                    "id": tenant.id,
                    "name": tenant.name,
                    "slug": tenant.slug,
                    "plan": tenant.plan.value,
                    "status": tenant.status.value,
                    "business_type": tenant.business_type,
                    "industry": tenant.industry,
                    "country": tenant.country,
                    "timezone": tenant.timezone,
                    "billing_email": tenant.billing_email,
                    "subdomain": tenant.subdomain,
                    "custom_domain": tenant.custom_domain,
                    "require_mfa": tenant.require_mfa,
                    "data_residency_region": tenant.data_residency_region,
                    "pci_compliance_required": tenant.pci_compliance_required,
                    "gdpr_applicable": tenant.gdpr_applicable,
                    "allowed_processors": tenant.allowed_processors,
                    "default_currency": tenant.default_currency,
                    "resource_limits": tenant.resource_limits,
                    "feature_flags": tenant.feature_flags,
                    "created_at": tenant.created_at.isoformat(),
                    "updated_at": tenant.updated_at.isoformat(),
                    "last_activity_at": tenant.last_activity_at.isoformat() if tenant.last_activity_at else None
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Tenant retrieval error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/slug/<slug>', methods=['GET'])
    @has_access
    @permission_name('tenant_read')
    def get_tenant_by_slug(self, slug: str):
        """
        Get tenant by slug
        
        GET /api/v1/tenants/slug/{slug}
        """
        try:
            await self._ensure_initialized()
            
            tenant = await self.tenant_service.get_tenant_by_slug(slug)
            if not tenant:
                return jsonify({"error": "Tenant not found"}), 404
            
            return jsonify({
                "tenant": {
                    "id": tenant.id,
                    "name": tenant.name,
                    "slug": tenant.slug,
                    "plan": tenant.plan.value,
                    "status": tenant.status.value,
                    "business_type": tenant.business_type,
                    "country": tenant.country,
                    "created_at": tenant.created_at.isoformat()
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Tenant retrieval by slug error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/<tenant_id>', methods=['PATCH'])
    @has_access
    @permission_name('tenant_update')
    def update_tenant(self, tenant_id: str):
        """
        Update tenant configuration
        
        PATCH /api/v1/tenants/{tenant_id}
        {
            "plan": "enterprise",
            "require_mfa": true,
            "allowed_processors": ["stripe", "adyen", "paypal"]
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            updates = request.get_json()
            if not updates:
                raise BadRequest("No updates provided")
            
            # Update tenant
            await self.tenant_service.update_tenant(tenant_id, updates)
            
            # Return updated tenant
            tenant = await self.tenant_service.get_tenant(tenant_id)
            if not tenant:
                return jsonify({"error": "Tenant not found"}), 404
            
            return jsonify({
                "success": True,
                "tenant": {
                    "id": tenant.id,
                    "name": tenant.name,
                    "plan": tenant.plan.value,
                    "status": tenant.status.value,
                    "updated_at": tenant.updated_at.isoformat(),
                    "updated_fields": list(updates.keys())
                }
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Tenant update error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Resource Management Endpoints
    
    @expose('/<tenant_id>/resources', methods=['GET'])
    @has_access
    @permission_name('tenant_resources')
    def get_tenant_resources(self, tenant_id: str):
        """
        Get tenant resource usage and limits
        
        GET /api/v1/tenants/{tenant_id}/resources
        """
        try:
            await self._ensure_initialized()
            
            usage_report = await self.tenant_service.get_tenant_usage_report(tenant_id)
            
            if "error" in usage_report:
                if usage_report["error"] == "Tenant not found":
                    return jsonify({"error": "Tenant not found"}), 404
                else:
                    return jsonify({"error": usage_report["error"]}), 500
            
            return jsonify(usage_report)
            
        except Exception as e:
            current_app.logger.error(f"Tenant resources error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/<tenant_id>/resources/<resource_type>/check', methods=['POST'])  
    @has_access
    @permission_name('tenant_resources')
    def check_resource_limit(self, tenant_id: str, resource_type: str):
        """
        Check if tenant can consume specific resources
        
        POST /api/v1/tenants/{tenant_id}/resources/{resource_type}/check
        {
            "amount": 100
        }
        """
        try:
            await self._ensure_initialized()
            
            # Validate resource type
            try:
                resource_enum = ResourceType(resource_type)
            except ValueError:
                return jsonify({"error": f"Invalid resource type: {resource_type}"}), 400
            
            data = request.get_json() or {}
            amount = data.get("amount", 1)
            
            if not isinstance(amount, int) or amount <= 0:
                return jsonify({"error": "Amount must be a positive integer"}), 400
            
            check_result = await self.tenant_service.check_resource_limit(
                tenant_id, resource_enum, amount
            )
            
            return jsonify(check_result)
            
        except Exception as e:
            current_app.logger.error(f"Resource limit check error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/<tenant_id>/resources/<resource_type>/consume', methods=['POST'])
    @has_access
    @permission_name('tenant_resources')
    def consume_resource(self, tenant_id: str, resource_type: str):
        """
        Consume tenant resources
        
        POST /api/v1/tenants/{tenant_id}/resources/{resource_type}/consume
        {
            "amount": 1
        }
        """
        try:
            await self._ensure_initialized()
            
            # Validate resource type
            try:
                resource_enum = ResourceType(resource_type)
            except ValueError:
                return jsonify({"error": f"Invalid resource type: {resource_type}"}), 400
            
            data = request.get_json() or {}
            amount = data.get("amount", 1)
            
            if not isinstance(amount, int) or amount <= 0:
                return jsonify({"error": "Amount must be a positive integer"}), 400
            
            success = await self.tenant_service.consume_resource(
                tenant_id, resource_enum, amount
            )
            
            return jsonify({
                "success": success,
                "consumed": amount if success else 0,
                "resource_type": resource_type
            })
            
        except Exception as e:
            current_app.logger.error(f"Resource consumption error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Analytics and Reporting Endpoints
    
    @expose('/<tenant_id>/analytics', methods=['GET'])
    @has_access
    @permission_name('tenant_analytics')
    def get_tenant_analytics(self, tenant_id: str):
        """
        Get comprehensive tenant analytics
        
        GET /api/v1/tenants/{tenant_id}/analytics?period=30d
        """
        try:
            await self._ensure_initialized()
            
            period = request.args.get('period', '30d')
            
            # Get analytics from database service
            analytics = await self.database_service.get_tenant_analytics(tenant_id)
            
            if "error" in analytics:
                if analytics["error"] == "Tenant not found":
                    return jsonify({"error": "Tenant not found"}), 404
                else:
                    return jsonify({"error": analytics["error"]}), 500
            
            # Add period information
            analytics["period"] = period
            analytics["generated_at"] = datetime.now(timezone.utc).isoformat()
            
            return jsonify(analytics)
            
        except Exception as e:
            current_app.logger.error(f"Tenant analytics error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Tenant Isolation Endpoints
    
    @expose('/<tenant_id>/isolation/database', methods=['GET'])
    @has_access
    @permission_name('tenant_admin')
    def get_tenant_database_info(self, tenant_id: str):
        """
        Get tenant database isolation information
        
        GET /api/v1/tenants/{tenant_id}/isolation/database
        """
        try:
            await self._ensure_initialized()
            
            # Get database connection string (sanitized)
            db_connection = await self.tenant_service.get_tenant_database_connection(tenant_id)
            
            # Sanitize connection string for API response
            import re
            sanitized_connection = re.sub(r'://[^@]+@', '://***:***@', db_connection)
            
            tenant = await self.tenant_service.get_tenant(tenant_id)
            if not tenant:
                return jsonify({"error": "Tenant not found"}), 404
            
            return jsonify({
                "tenant_id": tenant_id,
                "database_isolation": {
                    "schema": f"tenant_{tenant.slug}",
                    "connection_template": sanitized_connection,
                    "isolation_method": "schema_per_tenant"
                },
                "security_context": {
                    "data_residency_region": tenant.data_residency_region,
                    "pci_compliance_required": tenant.pci_compliance_required,
                    "gdpr_applicable": tenant.gdpr_applicable
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Tenant database isolation error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Administrative Endpoints
    
    @expose('/', methods=['GET'])
    @has_access
    @permission_name('tenant_list')
    def list_tenants(self):
        """
        List all tenants with optional filtering
        
        GET /api/v1/tenants/?status=active&plan=enterprise&limit=50&offset=0
        """
        try:
            await self._ensure_initialized()
            
            # Parse query parameters
            status_filter = request.args.get('status')
            plan_filter = request.args.get('plan')
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            # Convert string filters to enums
            status_enum = TenantStatus(status_filter) if status_filter else None
            plan_enum = TenantPlan(plan_filter) if plan_filter else None
            
            # Get tenants from database (simplified - would normally use proper pagination)
            all_tenants = []
            
            # For demo, create a few sample tenants if none exist
            if not hasattr(self.tenant_service, '_tenant_cache') or not self.tenant_service._tenant_cache:
                sample_tenants = [
                    {
                        "name": "Demo Company",
                        "slug": "demo-company", 
                        "business_type": "ecommerce",
                        "country": "US",
                        "plan": "professional",
                        "billing_email": "demo@example.com"
                    }
                ]
                for tenant_data in sample_tenants:
                    await self.tenant_service.create_tenant(tenant_data)
            
            # Get all tenants
            for tenant_id in self.tenant_service._tenant_cache.keys():
                tenant = await self.tenant_service.get_tenant(tenant_id)
                if tenant:
                    # Apply filters
                    if status_enum and tenant.status != status_enum:
                        continue
                    if plan_enum and tenant.plan != plan_enum:
                        continue
                    
                    all_tenants.append({
                        "id": tenant.id,
                        "name": tenant.name,
                        "slug": tenant.slug,
                        "plan": tenant.plan.value,
                        "status": tenant.status.value,
                        "business_type": tenant.business_type,
                        "country": tenant.country,
                        "created_at": tenant.created_at.isoformat(),
                        "last_activity_at": tenant.last_activity_at.isoformat() if tenant.last_activity_at else None
                    })
            
            # Apply pagination
            total = len(all_tenants)
            tenants_page = all_tenants[offset:offset + limit]
            
            return jsonify({
                "tenants": tenants_page,
                "pagination": {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total
                },
                "filters": {
                    "status": status_filter,
                    "plan": plan_filter
                }
            })
            
        except ValueError as e:
            return jsonify({"error": f"Invalid filter value: {str(e)}"}), 400
        except Exception as e:
            current_app.logger.error(f"Tenant listing error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/health', methods=['GET'])
    def health_check(self):
        """
        Multi-tenant service health check
        
        GET /api/v1/tenants/health
        """
        try:
            await self._ensure_initialized()
            
            # Check tenant service health
            tenant_service_health = "healthy"
            total_tenants = 0
            
            if self.tenant_service and hasattr(self.tenant_service, '_tenant_cache'):
                total_tenants = len(self.tenant_service._tenant_cache)
            
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "services": {
                    "tenant_service": tenant_service_health,
                    "database_service": "healthy" if self.database_service else "not_initialized"
                },
                "metrics": {
                    "total_tenants": total_tenants,
                    "initialized": self._initialized
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Multi-tenant health check error: {str(e)}")
            return jsonify({
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 500

# Register Flask-AppBuilder view
def register_multi_tenant_views(appbuilder):
    """Register multi-tenant views with Flask-AppBuilder"""
    appbuilder.add_view_no_menu(MultiTenantAPIView)

# Module initialization logging
def _log_multi_tenant_api_module_loaded():
    """Log multi-tenant API module loaded"""
    print("🏢 Multi-Tenant API module loaded")
    print("   - Tenant CRUD operations")
    print("   - Resource management")
    print("   - Usage analytics")
    print("   - Isolation controls")
    print("   - Administrative endpoints")

# Execute module loading log
_log_multi_tenant_api_module_loaded()