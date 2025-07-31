#!/usr/bin/env python3
"""
Advanced Webhook Management Service - APG Payment Gateway

Comprehensive webhook management with delivery guarantees, retry logic,
security features, and real-time monitoring capabilities.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import hmac
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable
from uuid_extensions import uuid7str
from dataclasses import dataclass, field
import logging
import aiohttp
import ssl
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ConfigDict, field_validator, HttpUrl
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

logger = logging.getLogger(__name__)

# Webhook data models
class WebhookEventType(str, Enum):
    """Webhook event types for payment gateway"""
    PAYMENT_COMPLETED = "payment.completed"
    PAYMENT_FAILED = "payment.failed"
    PAYMENT_AUTHORIZED = "payment.authorized"
    PAYMENT_CAPTURED = "payment.captured"
    PAYMENT_REFUNDED = "payment.refunded"
    PAYMENT_DISPUTED = "payment.disputed"
    PAYMENT_CHARGEBACK = "payment.chargeback"
    SUBSCRIPTION_CREATED = "subscription.created"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_PAYMENT_FAILED = "subscription.payment_failed"
    INVOICE_CREATED = "invoice.created"
    INVOICE_PAID = "invoice.paid"
    INVOICE_OVERDUE = "invoice.overdue"
    MERCHANT_CREATED = "merchant.created"
    MERCHANT_VERIFIED = "merchant.verified"
    FRAUD_DETECTED = "fraud.detected"
    CUSTOM_EVENT = "custom.event"

class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    DISABLED = "disabled"

class WebhookSecurityType(str, Enum):
    """Webhook security/signature types"""
    HMAC_SHA256 = "hmac_sha256"
    HMAC_SHA512 = "hmac_sha512"
    JWT = "jwt"
    BASIC_AUTH = "basic_auth"
    API_KEY = "api_key"
    NONE = "none"

@dataclass
class WebhookRetryConfig:
    """Webhook retry configuration"""
    max_attempts: int = 5
    initial_delay_seconds: int = 5
    max_delay_seconds: int = 300
    backoff_multiplier: float = 2.0
    timeout_seconds: int = 30
    
    def get_delay_for_attempt(self, attempt: int) -> int:
        """Calculate delay for given retry attempt"""
        delay = self.initial_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(int(delay), self.max_delay_seconds)

class WebhookEndpoint(BaseModel):
    """
    Webhook endpoint configuration with advanced features
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    # Core identification
    id: str = Field(default_factory=uuid7str, description="Unique endpoint ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    merchant_id: str | None = Field(None, description="Optional merchant identifier")
    
    # Endpoint configuration
    name: str = Field(..., min_length=1, max_length=100, description="Endpoint name")
    url: HttpUrl = Field(..., description="Webhook URL")
    description: str | None = Field(None, max_length=500, description="Endpoint description")
    
    # Event configuration
    enabled_events: Set[WebhookEventType] = Field(default_factory=set, description="Enabled event types")
    all_events: bool = Field(default=False, description="Listen to all events")
    
    # Security configuration
    security_type: WebhookSecurityType = Field(default=WebhookSecurityType.HMAC_SHA256, description="Security method")
    secret: str = Field(..., min_length=8, description="Webhook secret for signing")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    
    # Delivery configuration
    retry_config: WebhookRetryConfig = Field(default_factory=WebhookRetryConfig, description="Retry configuration")
    timeout_seconds: int = Field(default=30, ge=5, le=300, description="Request timeout")
    
    # Filtering and transformation
    event_filters: Dict[str, Any] = Field(default_factory=dict, description="Event filtering rules")
    payload_transform: Dict[str, Any] = Field(default_factory=dict, description="Payload transformation rules")
    
    # Status and monitoring
    enabled: bool = Field(default=True, description="Endpoint enabled status")
    last_success_at: datetime | None = Field(None, description="Last successful delivery")
    last_failure_at: datetime | None = Field(None, description="Last failed delivery")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")
    total_deliveries: int = Field(default=0, description="Total delivery attempts")
    successful_deliveries: int = Field(default=0, description="Successful deliveries")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('enabled_events')
    @classmethod
    def validate_enabled_events(cls, v):
        """Validate enabled events are proper enum values"""
        if isinstance(v, set):
            return {WebhookEventType(event) if isinstance(event, str) else event for event in v}
        return set()
    
    @property
    def success_rate(self) -> float:
        """Calculate delivery success rate"""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy (low consecutive failures)"""
        return self.consecutive_failures < 5

class WebhookEvent(BaseModel):
    """
    Webhook event with payload and delivery tracking
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    # Core identification
    id: str = Field(default_factory=uuid7str, description="Unique event ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    endpoint_id: str = Field(..., description="Target endpoint ID")
    
    # Event data
    event_type: WebhookEventType = Field(..., description="Event type")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    
    # Delivery tracking
    status: WebhookStatus = Field(default=WebhookStatus.PENDING, description="Delivery status")
    attempt_count: int = Field(default=0, description="Delivery attempt count")
    next_retry_at: datetime | None = Field(None, description="Next retry time")
    
    # Response tracking
    last_response_status: int | None = Field(None, description="Last HTTP response status")
    last_response_body: str | None = Field(None, description="Last response body")
    last_response_headers: Dict[str, str] = Field(default_factory=dict, description="Last response headers")
    
    # Timing
    delivery_time_ms: int | None = Field(None, description="Delivery time in milliseconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: datetime | None = Field(None, description="Successful delivery time")
    
    # Error tracking
    error_message: str | None = Field(None, description="Last error message")
    error_details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    
    @property
    def is_expired(self) -> bool:
        """Check if event has expired (older than 7 days)"""
        expiry_time = self.created_at + timedelta(days=7)
        return datetime.now(timezone.utc) > expiry_time

class WebhookService:
    """
    Advanced webhook management service with delivery guarantees
    """
    
    def __init__(self, database_service=None):
        self._database_service = database_service
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._pending_events: Dict[str, WebhookEvent] = {}
        self._delivery_workers: List[asyncio.Task] = []
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._event_handlers: Dict[WebhookEventType, List[Callable]] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize webhook service"""
        try:
            # Create HTTP session with proper SSL configuration
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'APG-Payment-Gateway-Webhook/1.0',
                    'Content-Type': 'application/json'
                }
            )
            
            # Load existing endpoints from database
            await self._load_endpoints_from_database()
            
            # Start delivery workers
            await self._start_delivery_workers()
            
            self._running = True
            self._initialized = True
            self._log_service_initialized()
            
        except Exception as e:
            logger.error("webhook_service_initialization_failed", error=str(e))
            raise
    
    # Endpoint Management
    
    async def create_endpoint(self, endpoint_data: Dict[str, Any]) -> WebhookEndpoint:
        """Create a new webhook endpoint"""
        try:
            # Validate URL accessibility
            await self._validate_endpoint_url(endpoint_data['url'])
            
            # Convert string events to enums
            if 'enabled_events' in endpoint_data:
                enabled_events = {WebhookEventType(e) for e in endpoint_data['enabled_events']}
                endpoint_data['enabled_events'] = enabled_events
            
            endpoint = WebhookEndpoint(**endpoint_data)
            
            # Store endpoint
            self._endpoints[endpoint.id] = endpoint
            
            if self._database_service:
                await self._database_service.create_webhook_endpoint(endpoint)
            
            logger.info("webhook_endpoint_created", 
                endpoint_id=endpoint.id, 
                url=str(endpoint.url),
                events=len(endpoint.enabled_events)
            )
            
            return endpoint
            
        except Exception as e:
            logger.error("webhook_endpoint_creation_failed", error=str(e))
            raise
    
    async def get_endpoint(self, endpoint_id: str) -> WebhookEndpoint | None:
        """Get webhook endpoint by ID"""
        return self._endpoints.get(endpoint_id)
    
    async def list_endpoints(self, tenant_id: str, merchant_id: str | None = None) -> List[WebhookEndpoint]:
        """List webhook endpoints for tenant/merchant"""
        endpoints = []
        for endpoint in self._endpoints.values():
            if endpoint.tenant_id == tenant_id:
                if merchant_id is None or endpoint.merchant_id == merchant_id:
                    endpoints.append(endpoint)
        return endpoints
    
    async def update_endpoint(self, endpoint_id: str, updates: Dict[str, Any]) -> WebhookEndpoint | None:
        """Update webhook endpoint configuration"""
        try:
            endpoint = self._endpoints.get(endpoint_id)
            if not endpoint:
                return None
            
            # Validate URL if being updated
            if 'url' in updates:
                await self._validate_endpoint_url(updates['url'])
            
            # Convert string events to enums if needed
            if 'enabled_events' in updates:
                enabled_events = {WebhookEventType(e) for e in updates['enabled_events']}
                updates['enabled_events'] = enabled_events
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(endpoint, key):
                    setattr(endpoint, key, value)
            
            endpoint.updated_at = datetime.now(timezone.utc)
            
            # Update in database
            if self._database_service:
                await self._database_service.update_webhook_endpoint(endpoint_id, updates)
            
            logger.info("webhook_endpoint_updated", 
                endpoint_id=endpoint_id,
                updated_fields=list(updates.keys())
            )
            
            return endpoint
            
        except Exception as e:
            logger.error("webhook_endpoint_update_failed", 
                endpoint_id=endpoint_id, 
                error=str(e)
            )
            raise
    
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint"""
        try:
            if endpoint_id in self._endpoints:
                del self._endpoints[endpoint_id]
                
                if self._database_service:
                    await self._database_service.delete_webhook_endpoint(endpoint_id)
                
                logger.info("webhook_endpoint_deleted", endpoint_id=endpoint_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("webhook_endpoint_deletion_failed", 
                endpoint_id=endpoint_id, 
                error=str(e)
            )
            return False
    
    # Event Management
    
    async def send_webhook(self, tenant_id: str, event_type: WebhookEventType, payload: Dict[str, Any], 
                          merchant_id: str | None = None, metadata: Dict[str, Any] = None) -> List[str]:
        """Send webhook to all matching endpoints"""
        try:
            event_ids = []
            
            # Find matching endpoints
            matching_endpoints = []
            for endpoint in self._endpoints.values():
                if endpoint.tenant_id == tenant_id and endpoint.enabled:
                    if merchant_id is None or endpoint.merchant_id == merchant_id:
                        if endpoint.all_events or event_type in endpoint.enabled_events:
                            matching_endpoints.append(endpoint)
            
            # Create events for each endpoint
            for endpoint in matching_endpoints:
                # Apply event filters
                if not await self._should_deliver_event(endpoint, event_type, payload):
                    continue
                
                # Transform payload if needed
                transformed_payload = await self._transform_payload(endpoint, payload)
                
                event = WebhookEvent(
                    tenant_id=tenant_id,
                    endpoint_id=endpoint.id,
                    event_type=event_type,
                    payload=transformed_payload
                )
                
                self._pending_events[event.id] = event
                event_ids.append(event.id)
                
                logger.info("webhook_event_queued", 
                    event_id=event.id,
                    endpoint_id=endpoint.id,
                    event_type=event_type.value
                )
            
            return event_ids
            
        except Exception as e:
            logger.error("webhook_send_failed", 
                event_type=event_type.value, 
                error=str(e)
            )
            return []
    
    async def get_event(self, event_id: str) -> WebhookEvent | None:
        """Get webhook event by ID"""
        return self._pending_events.get(event_id)
    
    async def retry_event(self, event_id: str) -> bool:
        """Manually retry webhook event"""
        try:
            event = self._pending_events.get(event_id)
            if not event:
                return False
            
            if event.status in [WebhookStatus.DELIVERED, WebhookStatus.EXPIRED]:
                return False
            
            event.status = WebhookStatus.PENDING
            event.next_retry_at = datetime.now(timezone.utc)
            
            logger.info("webhook_event_retry_requested", event_id=event_id)
            return True
            
        except Exception as e:
            logger.error("webhook_event_retry_failed", 
                event_id=event_id, 
                error=str(e)
            )
            return False
    
    # Event Delivery
    
    async def _start_delivery_workers(self):
        """Start background webhook delivery workers"""
        worker_count = 3  # Configurable number of workers
        
        for i in range(worker_count):
            worker = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self._delivery_workers.append(worker)
        
        logger.info("webhook_delivery_workers_started", count=worker_count)
    
    async def _delivery_worker(self, worker_name: str):
        """Background worker for webhook delivery"""
        logger.info("webhook_delivery_worker_started", worker=worker_name)
        
        while self._running:
            try:
                # Find events ready for delivery
                ready_events = []
                now = datetime.now(timezone.utc)
                
                for event in list(self._pending_events.values()):
                    if event.status == WebhookStatus.PENDING:
                        ready_events.append(event)
                    elif event.status == WebhookStatus.RETRYING and event.next_retry_at and event.next_retry_at <= now:
                        ready_events.append(event)
                    elif event.is_expired:
                        event.status = WebhookStatus.EXPIRED
                        self._pending_events.pop(event.id, None)
                
                # Deliver events
                for event in ready_events[:5]:  # Process up to 5 events per cycle
                    await self._deliver_event(event)
                
                # Sleep before next cycle
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("webhook_delivery_worker_error", 
                    worker=worker_name, 
                    error=str(e)
                )
                await asyncio.sleep(5)
    
    async def _deliver_event(self, event: WebhookEvent):
        """Deliver a single webhook event"""
        try:
            endpoint = self._endpoints.get(event.endpoint_id)
            if not endpoint or not endpoint.enabled:
                event.status = WebhookStatus.FAILED
                event.error_message = "Endpoint not found or disabled"
                return
            
            event.status = WebhookStatus.DELIVERING
            event.attempt_count += 1
            
            # Prepare request
            headers = self._prepare_headers(endpoint, event)
            signature = self._generate_signature(endpoint, event.payload)
            if signature:
                headers['X-Webhook-Signature'] = signature
            
            # Add custom headers
            headers.update(endpoint.custom_headers)
            
            # Make HTTP request
            start_time = time.time()
            
            async with self._session.post(
                str(endpoint.url),
                json=event.payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            ) as response:
                delivery_time = int((time.time() - start_time) * 1000)
                
                event.delivery_time_ms = delivery_time
                event.last_response_status = response.status
                event.last_response_headers = dict(response.headers)
                
                # Read response body
                try:
                    response_body = await response.text()
                    event.last_response_body = response_body[:1000]  # Limit size
                except Exception:
                    event.last_response_body = "Unable to read response body"
                
                # Check if delivery was successful
                if 200 <= response.status < 300:
                    event.status = WebhookStatus.DELIVERED
                    event.delivered_at = datetime.now(timezone.utc)
                    
                    # Update endpoint statistics
                    endpoint.last_success_at = datetime.now(timezone.utc)
                    endpoint.consecutive_failures = 0
                    endpoint.successful_deliveries += 1
                    endpoint.total_deliveries += 1
                    
                    # Remove from pending events
                    self._pending_events.pop(event.id, None)
                    
                    logger.info("webhook_delivered_successfully", 
                        event_id=event.id,
                        endpoint_id=endpoint.id,
                        status_code=response.status,
                        delivery_time_ms=delivery_time
                    )
                    
                else:
                    await self._handle_delivery_failure(event, endpoint, f"HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            await self._handle_delivery_failure(event, endpoint, "Request timeout")
        except Exception as e:
            await self._handle_delivery_failure(event, endpoint, str(e))
    
    async def _handle_delivery_failure(self, event: WebhookEvent, endpoint: WebhookEndpoint, error_message: str):
        """Handle webhook delivery failure"""
        event.error_message = error_message
        event.error_details = {
            "attempt": event.attempt_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_message
        }
        
        # Update endpoint statistics
        endpoint.last_failure_at = datetime.now(timezone.utc)
        endpoint.consecutive_failures += 1
        endpoint.total_deliveries += 1
        
        # Check if we should retry
        if event.attempt_count < endpoint.retry_config.max_attempts:
            event.status = WebhookStatus.RETRYING
            delay = endpoint.retry_config.get_delay_for_attempt(event.attempt_count)
            event.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
            
            logger.warning("webhook_delivery_failed_retrying", 
                event_id=event.id,
                endpoint_id=endpoint.id,
                attempt=event.attempt_count,
                retry_in_seconds=delay,
                error=error_message
            )
        else:
            event.status = WebhookStatus.FAILED
            self._pending_events.pop(event.id, None)
            
            logger.error("webhook_delivery_failed_permanently", 
                event_id=event.id,
                endpoint_id=endpoint.id,
                attempts=event.attempt_count,
                error=error_message
            )
    
    # Helper Methods
    
    def _prepare_headers(self, endpoint: WebhookEndpoint, event: WebhookEvent) -> Dict[str, str]:
        """Prepare HTTP headers for webhook request"""
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Event-Type': event.event_type.value,
            'X-Webhook-Event-ID': event.id,
            'X-Webhook-Delivery-ID': uuid7str(),
            'X-Webhook-Timestamp': str(int(time.time())),
            'User-Agent': 'APG-Payment-Gateway-Webhook/1.0'
        }
        
        if endpoint.security_type == WebhookSecurityType.API_KEY:
            headers['X-API-Key'] = endpoint.secret
        elif endpoint.security_type == WebhookSecurityType.BASIC_AUTH:
            import base64
            auth_string = base64.b64encode(f"webhook:{endpoint.secret}".encode()).decode()
            headers['Authorization'] = f'Basic {auth_string}'
        
        return headers
    
    def _generate_signature(self, endpoint: WebhookEndpoint, payload: Dict[str, Any]) -> str | None:
        """Generate webhook signature"""
        if endpoint.security_type == WebhookSecurityType.NONE:
            return None
        
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        
        if endpoint.security_type == WebhookSecurityType.HMAC_SHA256:
            signature = hmac.new(
                endpoint.secret.encode('utf-8'),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            return f'sha256={signature}'
        
        elif endpoint.security_type == WebhookSecurityType.HMAC_SHA512:
            signature = hmac.new(
                endpoint.secret.encode('utf-8'),
                payload_bytes,
                hashlib.sha512
            ).hexdigest()
            return f'sha512={signature}'
        
        return None
    
    async def _validate_endpoint_url(self, url: str):
        """Validate webhook endpoint URL"""
        try:
            parsed = urlparse(str(url))
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValueError("URL must use HTTP or HTTPS")
            
            # Check for localhost in production (would be configurable)
            if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                logger.warning("webhook_localhost_url_detected", url=url)
            
            # Test connectivity (optional)
            # await self._test_endpoint_connectivity(url)
            
        except Exception as e:
            raise ValueError(f"Invalid webhook URL: {str(e)}")
    
    async def _should_deliver_event(self, endpoint: WebhookEndpoint, event_type: WebhookEventType, payload: Dict[str, Any]) -> bool:
        """Check if event should be delivered to endpoint based on filters"""
        if not endpoint.event_filters:
            return True
        
        # Apply custom filtering logic here
        # For now, always deliver
        return True
    
    async def _transform_payload(self, endpoint: WebhookEndpoint, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform payload based on endpoint configuration"""
        if not endpoint.payload_transform:
            return payload
        
        # Apply custom transformation logic here
        # For now, return as-is
        return payload
    
    async def _load_endpoints_from_database(self):
        """Load existing webhook endpoints from database"""
        if self._database_service:
            # In production, this would load from database
            # For now, create empty collection
            pass
    
    def _log_service_initialized(self):
        """Log service initialization"""
        logger.info("webhook_service_initialized", 
            endpoints_loaded=len(self._endpoints),
            workers_started=len(self._delivery_workers)
        )
    
    # Analytics and Monitoring
    
    async def get_endpoint_statistics(self, endpoint_id: str) -> Dict[str, Any]:
        """Get detailed statistics for webhook endpoint"""
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint:
            return {"error": "Endpoint not found"}
        
        # Count pending events for this endpoint
        pending_events = sum(
            1 for event in self._pending_events.values() 
            if event.endpoint_id == endpoint_id
        )
        
        return {
            "endpoint_id": endpoint_id,
            "url": str(endpoint.url),
            "enabled": endpoint.enabled,
            "total_deliveries": endpoint.total_deliveries,
            "successful_deliveries": endpoint.successful_deliveries,
            "success_rate": endpoint.success_rate,
            "consecutive_failures": endpoint.consecutive_failures,
            "is_healthy": endpoint.is_healthy,
            "pending_events": pending_events,
            "last_success_at": endpoint.last_success_at.isoformat() if endpoint.last_success_at else None,
            "last_failure_at": endpoint.last_failure_at.isoformat() if endpoint.last_failure_at else None,
            "enabled_events": [e.value for e in endpoint.enabled_events],
            "security_type": endpoint.security_type.value
        }
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get overall webhook service health"""
        total_endpoints = len(self._endpoints)
        healthy_endpoints = sum(1 for e in self._endpoints.values() if e.is_healthy)
        pending_events = len(self._pending_events)
        active_workers = len([w for w in self._delivery_workers if not w.done()])
        
        return {
            "status": "healthy" if self._running else "unhealthy",
            "initialized": self._initialized,
            "total_endpoints": total_endpoints,
            "healthy_endpoints": healthy_endpoints,
            "unhealthy_endpoints": total_endpoints - healthy_endpoints,
            "pending_events": pending_events,
            "active_workers": active_workers,
            "session_active": self._session is not None and not self._session.closed
        }
    
    async def close(self):
        """Shutdown webhook service"""
        try:
            self._running = False
            
            # Cancel delivery workers
            for worker in self._delivery_workers:
                worker.cancel()
            
            await asyncio.gather(*self._delivery_workers, return_exceptions=True)
            
            # Close HTTP session
            if self._session and not self._session.closed:
                await self._session.close()
            
            logger.info("webhook_service_shutdown_complete")
            
        except Exception as e:
            logger.error("webhook_service_shutdown_error", error=str(e))

# Factory function
def create_webhook_service(database_service=None) -> WebhookService:
    """Create and initialize webhook service"""
    return WebhookService(database_service)

# Test utility
async def test_webhook_service():
    """Test webhook service functionality"""
    print("🔗 Testing Advanced Webhook Service")
    print("=" * 50)
    
    # Initialize service
    service = create_webhook_service()
    await service.initialize()
    
    # Create test endpoint
    endpoint_data = {
        "tenant_id": "test_tenant",
        "name": "Test Webhook",
        "url": "https://httpbin.org/post",
        "enabled_events": ["payment.completed", "payment.failed"],
        "security_type": "hmac_sha256",
        "secret": "test_secret_key_12345678"
    }
    
    endpoint = await service.create_endpoint(endpoint_data)
    print(f"✅ Created webhook endpoint: {endpoint.name}")
    print(f"   URL: {endpoint.url}")
    print(f"   Events: {len(endpoint.enabled_events)}")
    print(f"   Security: {endpoint.security_type.value}")
    
    # Send test webhooks
    test_payload = {
        "transaction_id": "txn_12345",
        "amount": 1000,
        "currency": "USD",
        "status": "completed"
    }
    
    event_ids = await service.send_webhook(
        "test_tenant",
        WebhookEventType.PAYMENT_COMPLETED,
        test_payload
    )
    print(f"📤 Queued {len(event_ids)} webhook events")
    
    # Wait for delivery
    await asyncio.sleep(3)
    
    # Check statistics
    stats = await service.get_endpoint_statistics(endpoint.id)
    print(f"📊 Endpoint statistics:")
    print(f"   Total deliveries: {stats['total_deliveries']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Is healthy: {stats['is_healthy']}")
    
    # Service health
    health = await service.get_service_health()
    print(f"💚 Service health: {health['status']}")
    print(f"   Active workers: {health['active_workers']}")
    print(f"   Pending events: {health['pending_events']}")
    
    await service.close()
    print("🎉 Webhook service test completed!")

if __name__ == "__main__":
    asyncio.run(test_webhook_service())