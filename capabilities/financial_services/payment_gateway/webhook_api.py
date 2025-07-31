#!/usr/bin/env python3
"""
Webhook Management API - APG Payment Gateway

RESTful API endpoints for comprehensive webhook management,
integrating with the existing payment gateway architecture.

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

from .webhook_service import (
    WebhookService, WebhookEndpoint, WebhookEvent, WebhookEventType, 
    WebhookStatus, WebhookSecurityType, create_webhook_service
)
from .database import get_database_service
from .auth import get_auth_service, require_permission

# Create Flask Blueprint for webhook endpoints
webhook_bp = Blueprint(
    'webhooks',
    __name__,
    url_prefix='/api/v1/webhooks'
)

class WebhookAPIView(BaseView):
    """
    Comprehensive Webhook Management API View
    
    Provides all webhook management endpoints with full integration
    into the existing APG payment gateway architecture.
    """
    
    route_base = "/api/v1/webhooks"
    
    def __init__(self):
        super().__init__()
        self.webhook_service: Optional[WebhookService] = None
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
            
            # Initialize webhook service
            self.webhook_service = WebhookService(self.database_service)
            await self.webhook_service.initialize()
            
            self._initialized = True
    
    # Webhook Endpoint Management
    
    @expose('/endpoints', methods=['POST'])
    @has_access
    @permission_name('webhook_create')
    def create_webhook_endpoint(self):
        """
        Create a new webhook endpoint
        
        POST /api/v1/webhooks/endpoints
        {
            "name": "Payment Notifications",
            "url": "https://myapp.com/webhooks/payments",
            "enabled_events": ["payment.completed", "payment.failed"],
            "security_type": "hmac_sha256",
            "secret": "your_webhook_secret_key",
            "custom_headers": {
                "X-Custom-Header": "value"
            },
            "timeout_seconds": 30,
            "description": "Webhook for payment status updates"
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['name', 'url', 'secret']
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")
            
            # Add tenant context from authentication
            # In production, this would come from authenticated user context
            data['tenant_id'] = data.get('tenant_id', 'default_tenant')
            
            # Validate security type
            if 'security_type' in data:
                try:
                    WebhookSecurityType(data['security_type'])
                except ValueError:
                    valid_types = [t.value for t in WebhookSecurityType]
                    raise BadRequest(f"Invalid security_type. Must be one of: {valid_types}")
            
            # Validate event types
            if 'enabled_events' in data:
                try:
                    for event in data['enabled_events']:
                        WebhookEventType(event)
                except ValueError as e:
                    valid_events = [t.value for t in WebhookEventType]
                    raise BadRequest(f"Invalid event type. Must be one of: {valid_events}")
            
            # Create endpoint
            endpoint = await self.webhook_service.create_endpoint(data)
            
            return jsonify({
                "success": True,
                "endpoint": {
                    "id": endpoint.id,
                    "name": endpoint.name,
                    "url": str(endpoint.url),
                    "enabled_events": [e.value for e in endpoint.enabled_events],
                    "all_events": endpoint.all_events,
                    "security_type": endpoint.security_type.value,
                    "enabled": endpoint.enabled,
                    "created_at": endpoint.created_at.isoformat(),
                    "success_rate": endpoint.success_rate,
                    "is_healthy": endpoint.is_healthy
                }
            }), 201
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Webhook endpoint creation error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/endpoints/<endpoint_id>', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def get_webhook_endpoint(self, endpoint_id: str):
        """
        Get webhook endpoint details
        
        GET /api/v1/webhooks/endpoints/{endpoint_id}
        """
        try:
            await self._ensure_initialized()
            
            endpoint = await self.webhook_service.get_endpoint(endpoint_id)
            if not endpoint:
                return jsonify({"error": "Webhook endpoint not found"}), 404
            
            return jsonify({
                "endpoint": {
                    "id": endpoint.id,
                    "name": endpoint.name,
                    "url": str(endpoint.url),
                    "description": endpoint.description,
                    "enabled_events": [e.value for e in endpoint.enabled_events],
                    "all_events": endpoint.all_events,
                    "security_type": endpoint.security_type.value,
                    "custom_headers": endpoint.custom_headers,
                    "timeout_seconds": endpoint.timeout_seconds,
                    "enabled": endpoint.enabled,
                    "retry_config": {
                        "max_attempts": endpoint.retry_config.max_attempts,
                        "initial_delay_seconds": endpoint.retry_config.initial_delay_seconds,
                        "max_delay_seconds": endpoint.retry_config.max_delay_seconds,
                        "backoff_multiplier": endpoint.retry_config.backoff_multiplier
                    },
                    "statistics": {
                        "total_deliveries": endpoint.total_deliveries,
                        "successful_deliveries": endpoint.successful_deliveries,
                        "success_rate": endpoint.success_rate,
                        "consecutive_failures": endpoint.consecutive_failures,
                        "is_healthy": endpoint.is_healthy
                    },
                    "timestamps": {
                        "created_at": endpoint.created_at.isoformat(),
                        "updated_at": endpoint.updated_at.isoformat(),
                        "last_success_at": endpoint.last_success_at.isoformat() if endpoint.last_success_at else None,
                        "last_failure_at": endpoint.last_failure_at.isoformat() if endpoint.last_failure_at else None
                    }
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Webhook endpoint retrieval error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/endpoints', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def list_webhook_endpoints(self):
        """
        List webhook endpoints
        
        GET /api/v1/webhooks/endpoints?tenant_id=tenant123&enabled=true
        """
        try:
            await self._ensure_initialized()
            
            # Parse query parameters
            tenant_id = request.args.get('tenant_id', 'default_tenant')
            merchant_id = request.args.get('merchant_id')
            enabled_only = request.args.get('enabled', '').lower() == 'true'
            
            # Get endpoints
            endpoints = await self.webhook_service.list_endpoints(tenant_id, merchant_id)
            
            # Apply filters
            if enabled_only:
                endpoints = [e for e in endpoints if e.enabled]
            
            # Format response
            endpoint_list = []
            for endpoint in endpoints:
                endpoint_list.append({
                    "id": endpoint.id,
                    "name": endpoint.name,
                    "url": str(endpoint.url),
                    "enabled_events": [e.value for e in endpoint.enabled_events],
                    "enabled": endpoint.enabled,
                    "success_rate": endpoint.success_rate,
                    "is_healthy": endpoint.is_healthy,
                    "created_at": endpoint.created_at.isoformat(),
                    "last_success_at": endpoint.last_success_at.isoformat() if endpoint.last_success_at else None
                })
            
            return jsonify({
                "endpoints": endpoint_list,
                "total": len(endpoint_list),
                "filters": {
                    "tenant_id": tenant_id,
                    "merchant_id": merchant_id,
                    "enabled_only": enabled_only
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Webhook endpoints listing error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/endpoints/<endpoint_id>', methods=['PATCH'])
    @has_access
    @permission_name('webhook_update')
    def update_webhook_endpoint(self, endpoint_id: str):
        """
        Update webhook endpoint
        
        PATCH /api/v1/webhooks/endpoints/{endpoint_id}
        {
            "enabled": false,
            "enabled_events": ["payment.completed"],
            "timeout_seconds": 45
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            updates = request.get_json()
            if not updates:
                raise BadRequest("No updates provided")
            
            # Validate security type if provided
            if 'security_type' in updates:
                try:
                    WebhookSecurityType(updates['security_type'])
                except ValueError:
                    valid_types = [t.value for t in WebhookSecurityType]
                    raise BadRequest(f"Invalid security_type. Must be one of: {valid_types}")
            
            # Validate event types if provided
            if 'enabled_events' in updates:
                try:
                    for event in updates['enabled_events']:
                        WebhookEventType(event)
                except ValueError:
                    valid_events = [t.value for t in WebhookEventType]
                    raise BadRequest(f"Invalid event type. Must be one of: {valid_events}")
            
            # Update endpoint
            endpoint = await self.webhook_service.update_endpoint(endpoint_id, updates)
            if not endpoint:
                return jsonify({"error": "Webhook endpoint not found"}), 404
            
            return jsonify({
                "success": True,
                "endpoint": {
                    "id": endpoint.id,
                    "name": endpoint.name,
                    "enabled": endpoint.enabled,
                    "updated_at": endpoint.updated_at.isoformat(),
                    "updated_fields": list(updates.keys())
                }
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Webhook endpoint update error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/endpoints/<endpoint_id>', methods=['DELETE'])
    @has_access
    @permission_name('webhook_delete')
    def delete_webhook_endpoint(self, endpoint_id: str):
        """
        Delete webhook endpoint
        
        DELETE /api/v1/webhooks/endpoints/{endpoint_id}
        """
        try:
            await self._ensure_initialized()
            
            success = await self.webhook_service.delete_endpoint(endpoint_id)
            if not success:
                return jsonify({"error": "Webhook endpoint not found"}), 404
            
            return jsonify({"success": True, "message": "Webhook endpoint deleted"})
            
        except Exception as e:
            current_app.logger.error(f"Webhook endpoint deletion error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Webhook Event Management
    
    @expose('/events', methods=['POST'])
    @has_access
    @permission_name('webhook_send')
    def send_webhook_event(self):
        """
        Send webhook event manually
        
        POST /api/v1/webhooks/events
        {
            "tenant_id": "tenant123",
            "event_type": "payment.completed",
            "payload": {
                "transaction_id": "txn_12345",
                "amount": 1000,
                "currency": "USD",
                "status": "completed"
            },
            "merchant_id": "merchant456"
        }
        """
        try:
            await self._ensure_initialized()
            
            if not request.is_json:
                raise BadRequest("Request must be JSON")
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['tenant_id', 'event_type', 'payload']
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")
            
            # Validate event type
            try:
                event_type = WebhookEventType(data['event_type'])
            except ValueError:
                valid_events = [t.value for t in WebhookEventType]
                raise BadRequest(f"Invalid event_type. Must be one of: {valid_events}")
            
            # Send webhook
            event_ids = await self.webhook_service.send_webhook(
                tenant_id=data['tenant_id'],
                event_type=event_type,
                payload=data['payload'],
                merchant_id=data.get('merchant_id'),
                metadata=data.get('metadata', {})
            )
            
            return jsonify({
                "success": True,
                "events_queued": len(event_ids),
                "event_ids": event_ids,
                "event_type": event_type.value
            })
            
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            current_app.logger.error(f"Webhook event sending error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/events/<event_id>', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def get_webhook_event(self, event_id: str):
        """
        Get webhook event details
        
        GET /api/v1/webhooks/events/{event_id}
        """
        try:
            await self._ensure_initialized()
            
            event = await self.webhook_service.get_event(event_id)
            if not event:
                return jsonify({"error": "Webhook event not found"}), 404
            
            return jsonify({
                "event": {
                    "id": event.id,
                    "tenant_id": event.tenant_id,
                    "endpoint_id": event.endpoint_id,
                    "event_type": event.event_type.value,
                    "status": event.status.value,
                    "attempt_count": event.attempt_count,
                    "payload": event.payload,
                    "delivery_time_ms": event.delivery_time_ms,
                    "last_response_status": event.last_response_status,
                    "last_response_body": event.last_response_body,
                    "error_message": event.error_message,
                    "created_at": event.created_at.isoformat(),
                    "delivered_at": event.delivered_at.isoformat() if event.delivered_at else None,
                    "next_retry_at": event.next_retry_at.isoformat() if event.next_retry_at else None,
                    "is_expired": event.is_expired
                }
            })
            
        except Exception as e:
            current_app.logger.error(f"Webhook event retrieval error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/events/<event_id>/retry', methods=['POST'])
    @has_access
    @permission_name('webhook_retry')
    def retry_webhook_event(self, event_id: str):
        """
        Retry webhook event delivery
        
        POST /api/v1/webhooks/events/{event_id}/retry
        """
        try:
            await self._ensure_initialized()
            
            success = await self.webhook_service.retry_event(event_id)
            if not success:
                return jsonify({"error": "Webhook event not found or cannot be retried"}), 404
            
            return jsonify({
                "success": True,
                "message": "Webhook event queued for retry",
                "event_id": event_id
            })
            
        except Exception as e:
            current_app.logger.error(f"Webhook event retry error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Statistics and Monitoring
    
    @expose('/endpoints/<endpoint_id>/statistics', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def get_endpoint_statistics(self, endpoint_id: str):
        """
        Get detailed webhook endpoint statistics
        
        GET /api/v1/webhooks/endpoints/{endpoint_id}/statistics
        """
        try:
            await self._ensure_initialized()
            
            stats = await self.webhook_service.get_endpoint_statistics(endpoint_id)
            
            if "error" in stats:
                if stats["error"] == "Endpoint not found":
                    return jsonify({"error": "Webhook endpoint not found"}), 404
                else:
                    return jsonify({"error": stats["error"]}), 500
            
            return jsonify(stats)
            
        except Exception as e:
            current_app.logger.error(f"Webhook endpoint statistics error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/health', methods=['GET'])
    def get_webhook_service_health(self):
        """
        Get webhook service health status
        
        GET /api/v1/webhooks/health
        """
        try:
            await self._ensure_initialized()
            
            health = await self.webhook_service.get_service_health()
            
            return jsonify({
                "webhook_service": health,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0"
            })
            
        except Exception as e:
            current_app.logger.error(f"Webhook service health check error: {str(e)}")
            return jsonify({
                "webhook_service": {
                    "status": "unhealthy",
                    "error": str(e)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 500
    
    # Event Type Information
    
    @expose('/event-types', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def list_webhook_event_types(self):
        """
        List all supported webhook event types
        
        GET /api/v1/webhooks/event-types
        """
        try:
            event_types = []
            for event_type in WebhookEventType:
                # Categorize events
                category = "payment"
                if event_type.value.startswith("subscription"):
                    category = "subscription"
                elif event_type.value.startswith("invoice"):
                    category = "invoice"
                elif event_type.value.startswith("merchant"):
                    category = "merchant"
                elif event_type.value.startswith("fraud"):
                    category = "fraud"
                elif event_type.value.startswith("custom"):
                    category = "custom"
                
                event_types.append({
                    "type": event_type.value,
                    "category": category,
                    "description": self._get_event_type_description(event_type)
                })
            
            # Group by category
            categories = {}
            for event in event_types:
                cat = event["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(event)
            
            return jsonify({
                "event_types": event_types,
                "categories": categories,
                "total_types": len(event_types)
            })
            
        except Exception as e:
            current_app.logger.error(f"Event types listing error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @expose('/security-types', methods=['GET'])
    @has_access
    @permission_name('webhook_read')
    def list_webhook_security_types(self):
        """
        List all supported webhook security types
        
        GET /api/v1/webhooks/security-types
        """
        try:
            security_types = []
            for sec_type in WebhookSecurityType:
                security_types.append({
                    "type": sec_type.value,
                    "description": self._get_security_type_description(sec_type),
                    "recommended": sec_type in [WebhookSecurityType.HMAC_SHA256, WebhookSecurityType.HMAC_SHA512]
                })
            
            return jsonify({
                "security_types": security_types,
                "total_types": len(security_types)
            })
            
        except Exception as e:
            current_app.logger.error(f"Security types listing error: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    # Helper methods
    
    def _get_event_type_description(self, event_type: WebhookEventType) -> str:
        """Get human-readable description for event type"""
        descriptions = {
            WebhookEventType.PAYMENT_COMPLETED: "Payment transaction completed successfully",
            WebhookEventType.PAYMENT_FAILED: "Payment transaction failed",
            WebhookEventType.PAYMENT_AUTHORIZED: "Payment authorized but not captured",
            WebhookEventType.PAYMENT_CAPTURED: "Previously authorized payment captured",
            WebhookEventType.PAYMENT_REFUNDED: "Payment refunded to customer",
            WebhookEventType.PAYMENT_DISPUTED: "Payment disputed by customer",
            WebhookEventType.PAYMENT_CHARGEBACK: "Chargeback received for payment",
            WebhookEventType.SUBSCRIPTION_CREATED: "New subscription created",
            WebhookEventType.SUBSCRIPTION_UPDATED: "Subscription configuration updated",
            WebhookEventType.SUBSCRIPTION_CANCELLED: "Subscription cancelled",
            WebhookEventType.SUBSCRIPTION_PAYMENT_FAILED: "Subscription payment failed",
            WebhookEventType.INVOICE_CREATED: "New invoice generated",
            WebhookEventType.INVOICE_PAID: "Invoice payment received",
            WebhookEventType.INVOICE_OVERDUE: "Invoice payment overdue",
            WebhookEventType.MERCHANT_CREATED: "New merchant account created",
            WebhookEventType.MERCHANT_VERIFIED: "Merchant account verified",
            WebhookEventType.FRAUD_DETECTED: "Potentially fraudulent activity detected",
            WebhookEventType.CUSTOM_EVENT: "Custom event type"
        }
        return descriptions.get(event_type, "No description available")
    
    def _get_security_type_description(self, security_type: WebhookSecurityType) -> str:
        """Get human-readable description for security type"""
        descriptions = {
            WebhookSecurityType.HMAC_SHA256: "HMAC-SHA256 signature verification (recommended)",
            WebhookSecurityType.HMAC_SHA512: "HMAC-SHA512 signature verification (most secure)",
            WebhookSecurityType.JWT: "JSON Web Token authentication",
            WebhookSecurityType.BASIC_AUTH: "HTTP Basic Authentication",
            WebhookSecurityType.API_KEY: "API key in custom header",
            WebhookSecurityType.NONE: "No authentication (not recommended for production)"
        }
        return descriptions.get(security_type, "No description available")

# Register Flask-AppBuilder view
def register_webhook_views(appbuilder):
    """Register webhook views with Flask-AppBuilder"""
    appbuilder.add_view_no_menu(WebhookAPIView)

# Module initialization logging
def _log_webhook_api_module_loaded():
    """Log webhook API module loaded"""
    print("🔗 Webhook Management API module loaded")
    print("   - Endpoint CRUD operations")
    print("   - Event delivery management")
    print("   - Retry and monitoring")
    print("   - Security configuration")
    print("   - Statistics and analytics")

# Execute module loading log
_log_webhook_api_module_loaded()