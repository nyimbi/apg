"""
APG Developer-First Billing API Ecosystem

Complete developer-focused billing API with SDK generation, interactive documentation,
sandbox environments, webhook management, and comprehensive testing tools that make
integration effortless and delightful for developers.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import GenerateJsonSchema

from .models import BLCustomer, BLSubscription, BLPayment, BLInvoice
from .service import get_billing_service
from .audit_compliance import get_audit_compliance_system, AuditEventType


class APIVersion(Enum):
	"""API version enumeration"""
	V1 = "v1"
	V2 = "v2"
	BETA = "beta"


class SDKLanguage(Enum):
	"""Supported SDK languages"""
	PYTHON = "python"
	JAVASCRIPT = "javascript"
	TYPESCRIPT = "typescript"
	GO = "go"
	RUBY = "ruby"
	PHP = "php"
	JAVA = "java"
	CSHARP = "csharp"
	SWIFT = "swift"
	KOTLIN = "kotlin"


class WebhookEventType(Enum):
	"""Webhook event types"""
	CUSTOMER_CREATED = "customer.created"
	CUSTOMER_UPDATED = "customer.updated"
	SUBSCRIPTION_CREATED = "subscription.created"
	SUBSCRIPTION_UPDATED = "subscription.updated"
	SUBSCRIPTION_CANCELLED = "subscription.cancelled"
	PAYMENT_SUCCEEDED = "payment.succeeded"
	PAYMENT_FAILED = "payment.failed"
	INVOICE_CREATED = "invoice.created"
	INVOICE_PAID = "invoice.paid"
	INVOICE_OVERDUE = "invoice.overdue"
	DISPUTE_CREATED = "dispute.created"
	DISPUTE_RESOLVED = "dispute.resolved"


class APIEndpoint(BaseModel):
	"""API endpoint definition"""
	model_config = ConfigDict(extra='forbid')
	
	path: str = Field(..., description="Endpoint path")
	method: str = Field(..., description="HTTP method")
	summary: str = Field(..., description="Endpoint summary")
	description: str = Field(..., description="Detailed description")
	parameters: List[Dict[str, Any]] = Field(default_factory=list)
	request_body: Optional[Dict[str, Any]] = None
	responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
	examples: List[Dict[str, Any]] = Field(default_factory=list)
	rate_limit: Optional[int] = None
	requires_auth: bool = True
	scopes: List[str] = Field(default_factory=list)
	tags: List[str] = Field(default_factory=list)


class WebhookEndpoint(BaseModel):
	"""Webhook endpoint configuration"""
	model_config = ConfigDict(extra='forbid')
	
	id: str = Field(default_factory=uuid7str)
	url: str = Field(..., description="Webhook URL")
	events: List[WebhookEventType] = Field(..., description="Subscribed events")
	secret: str = Field(..., description="Webhook signing secret")
	active: bool = True
	max_retries: int = 3
	retry_delay_seconds: int = 60
	timeout_seconds: int = 30
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_success: Optional[datetime] = None
	last_failure: Optional[datetime] = None
	failure_count: int = 0
	headers: Dict[str, str] = Field(default_factory=dict)
	metadata: Dict[str, Any] = Field(default_factory=dict)


class SDKConfiguration(BaseModel):
	"""SDK generation configuration"""
	model_config = ConfigDict(extra='forbid')
	
	language: SDKLanguage
	package_name: str
	version: str
	author: str = "Datacraft"
	description: str = "APG Billing API SDK"
	repository_url: Optional[str] = None
	license: str = "MIT"
	includes_examples: bool = True
	includes_tests: bool = True
	async_support: bool = True
	retry_logic: bool = True
	rate_limiting: bool = True
	custom_types: bool = True


class DeveloperAPIEcosystem:
	"""Complete developer-first billing API ecosystem"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.DeveloperAPIEcosystem")
		self.billing_service = get_billing_service()
		self.audit_system = get_audit_compliance_system()
		
		# API configuration
		self.api_endpoints: Dict[str, APIEndpoint] = {}
		self.webhook_endpoints: Dict[str, WebhookEndpoint] = {}
		self.rate_limits: Dict[str, Dict[str, Any]] = {}
		
		# SDK management
		self.sdk_configurations: Dict[SDKLanguage, SDKConfiguration] = {}
		
		# Developer resources
		self.code_examples: Dict[str, Dict[str, str]] = {}
		self.interactive_docs: Dict[str, Any] = {}
		
		# Initialize API ecosystem
		asyncio.create_task(self._initialize_api_ecosystem())
	
	async def _initialize_api_ecosystem(self) -> None:
		"""Initialize the complete API ecosystem"""
		try:
			await self._setup_core_endpoints()
			await self._setup_webhook_system()
			await self._generate_sdk_configurations()
			await self._create_interactive_documentation()
			await self._setup_sandbox_environment()
			
			self.logger.info("✅ Developer API ecosystem initialized successfully")
			
		except Exception as e:
			self.logger.error(f"API ecosystem initialization failed: {e}")
	
	async def _setup_core_endpoints(self) -> None:
		"""Setup core billing API endpoints"""
		
		# Customer endpoints
		self.api_endpoints["create_customer"] = APIEndpoint(
			path="/v1/customers",
			method="POST",
			summary="Create a new customer",
			description="Create a new customer with billing information and preferences",
			request_body={
				"type": "object",
				"required": ["email", "name"],
				"properties": {
					"email": {"type": "string", "format": "email"},
					"name": {"type": "string"},
					"phone": {"type": "string"},
					"address": {"$ref": "#/components/schemas/Address"},
					"metadata": {"type": "object"}
				}
			},
			responses={
				"201": {
					"description": "Customer created successfully",
					"content": {
						"application/json": {
							"schema": {"$ref": "#/components/schemas/Customer"}
						}
					}
				},
				"400": {"description": "Invalid request data"},
				"409": {"description": "Customer already exists"}
			},
			examples=[
				{
					"name": "Basic customer creation",
					"request": {
						"email": "john@example.com",
						"name": "John Doe",
						"phone": "+1-555-0123"
					},
					"response": {
						"id": "cust_123456",
						"email": "john@example.com",
						"name": "John Doe",
						"created_at": "2025-01-01T00:00:00Z"
					}
				}
			],
			rate_limit=100,
			scopes=["customers:write"],
			tags=["Customers"]
		)
		
		# Subscription endpoints
		self.api_endpoints["create_subscription"] = APIEndpoint(
			path="/v1/subscriptions",
			method="POST",
			summary="Create a subscription",
			description="Create a new subscription for a customer with specified plan and billing cycle",
			request_body={
				"type": "object",
				"required": ["customer_id", "plan_id"],
				"properties": {
					"customer_id": {"type": "string"},
					"plan_id": {"type": "string"},
					"billing_period": {"type": "string", "enum": ["monthly", "yearly"]},
					"trial_days": {"type": "integer", "minimum": 0},
					"proration_behavior": {"type": "string", "enum": ["create_prorations", "none"]},
					"metadata": {"type": "object"}
				}
			},
			responses={
				"201": {
					"description": "Subscription created successfully",
					"content": {
						"application/json": {
							"schema": {"$ref": "#/components/schemas/Subscription"}
						}
					}
				}
			},
			examples=[
				{
					"name": "Monthly subscription with trial",
					"request": {
						"customer_id": "cust_123456",
						"plan_id": "plan_pro_monthly",
						"billing_period": "monthly",
						"trial_days": 14
					}
				}
			],
			rate_limit=50,
			scopes=["subscriptions:write"],
			tags=["Subscriptions"]
		)
		
		# Payment endpoints
		self.api_endpoints["create_payment"] = APIEndpoint(
			path="/v1/payments",
			method="POST",
			summary="Process a payment",
			description="Process a one-time payment or retry a failed payment",
			request_body={
				"type": "object",
				"required": ["customer_id", "amount", "currency"],
				"properties": {
					"customer_id": {"type": "string"},
					"amount": {"type": "number", "minimum": 0.01},
					"currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
					"payment_method_id": {"type": "string"},
					"description": {"type": "string"},
					"metadata": {"type": "object"}
				}
			},
			responses={
				"201": {
					"description": "Payment processed successfully",
					"content": {
						"application/json": {
							"schema": {"$ref": "#/components/schemas/Payment"}
						}
					}
				}
			},
			rate_limit=200,
			scopes=["payments:write"],
			tags=["Payments"]
		)
		
		# Analytics endpoints
		self.api_endpoints["get_analytics"] = APIEndpoint(
			path="/v1/analytics/dashboard",
			method="GET",
			summary="Get billing analytics",
			description="Retrieve comprehensive billing analytics and insights",
			parameters=[
				{
					"name": "scope",
					"in": "query",
					"schema": {"type": "string", "enum": ["daily", "weekly", "monthly"]},
					"description": "Analytics time scope"
				},
				{
					"name": "metrics",
					"in": "query",
					"schema": {"type": "array", "items": {"type": "string"}},
					"description": "Specific metrics to include"
				}
			],
			responses={
				"200": {
					"description": "Analytics data retrieved successfully",
					"content": {
						"application/json": {
							"schema": {"$ref": "#/components/schemas/AnalyticsDashboard"}
						}
					}
				}
			},
			rate_limit=10,
			scopes=["analytics:read"],
			tags=["Analytics"]
		)
		
		self.logger.info(f"✅ Setup {len(self.api_endpoints)} core API endpoints")
	
	async def _setup_webhook_system(self) -> None:
		"""Setup webhook management system"""
		
		# Create webhook management endpoints
		self.api_endpoints["create_webhook"] = APIEndpoint(
			path="/v1/webhooks",
			method="POST",
			summary="Create webhook endpoint",
			description="Register a new webhook endpoint to receive real-time billing events",
			request_body={
				"type": "object",
				"required": ["url", "events"],
				"properties": {
					"url": {"type": "string", "format": "uri"},
					"events": {
						"type": "array",
						"items": {"type": "string", "enum": [e.value for e in WebhookEventType]}
					},
					"secret": {"type": "string", "minLength": 16},
					"headers": {"type": "object"},
					"metadata": {"type": "object"}
				}
			},
			responses={
				"201": {
					"description": "Webhook created successfully",
					"content": {
						"application/json": {
							"schema": {"$ref": "#/components/schemas/WebhookEndpoint"}
						}
					}
				}
			},
			examples=[
				{
					"name": "Payment events webhook",
					"request": {
						"url": "https://api.example.com/webhooks/billing",
						"events": ["payment.succeeded", "payment.failed"],
						"secret": "whsec_1234567890abcdef"
					}
				}
			],
			rate_limit=10,
			scopes=["webhooks:write"],
			tags=["Webhooks"]
		)
		
		# Start webhook delivery system
		asyncio.create_task(self._start_webhook_delivery_system())
		
		self.logger.info("✅ Webhook system initialized")
	
	async def _start_webhook_delivery_system(self) -> None:
		"""Start background webhook delivery system"""
		while True:
			try:
				await self._process_webhook_queue()
				await asyncio.sleep(5)  # Process every 5 seconds
			except Exception as e:
				self.logger.error(f"Webhook delivery system error: {e}")
				await asyncio.sleep(30)
	
	async def _process_webhook_queue(self) -> None:
		"""Process pending webhook deliveries"""
		try:
			# Get pending webhook deliveries
			pending_deliveries = await self._get_pending_webhook_deliveries()
			
			for delivery in pending_deliveries:
				try:
					# Process individual delivery
					success = await self._deliver_webhook(delivery)
					
					if success:
						delivery.status = WebhookDeliveryStatus.DELIVERED
						delivery.delivered_at = datetime.utcnow()
						self.logger.info(f"Successfully delivered webhook {delivery.delivery_id}")
					else:
						delivery.retry_count += 1
						if delivery.retry_count >= 5:  # Max retries
							delivery.status = WebhookDeliveryStatus.FAILED
							self.logger.error(f"Webhook delivery failed after max retries: {delivery.delivery_id}")
						else:
							# Schedule retry with exponential backoff
							retry_delay = min(300, 2 ** delivery.retry_count)  # Max 5 minutes
							delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)
							self.logger.warning(f"Webhook delivery failed, scheduling retry: {delivery.delivery_id}")
					
					# Update delivery record
					await self._update_webhook_delivery(delivery)
					
				except Exception as e:
					self.logger.error(f"Error processing webhook delivery {delivery.delivery_id}: {e}")
			
		except Exception as e:
			self.logger.error(f"Webhook queue processing failed: {e}")

	async def _get_pending_webhook_deliveries(self) -> List[WebhookDelivery]:
		"""Get pending webhook deliveries from queue"""
		try:
			current_time = datetime.utcnow()
			pending_deliveries = []
			
			# Filter pending deliveries that are ready for processing
			for delivery in self.webhook_deliveries.values():
				if (delivery.status == WebhookDeliveryStatus.PENDING and
					(delivery.next_retry_at is None or delivery.next_retry_at <= current_time)):
					pending_deliveries.append(delivery)
			
			# Sort by created time (FIFO)
			pending_deliveries.sort(key=lambda d: d.created_at)
			
			return pending_deliveries[:50]  # Process up to 50 at a time
			
		except Exception as e:
			self.logger.error(f"Failed to get pending webhook deliveries: {e}")
			return []

	async def _deliver_webhook(self, delivery: WebhookDelivery) -> bool:
		"""Deliver individual webhook"""
		try:
			import aiohttp
			
			webhook = self.webhook_endpoints.get(delivery.webhook_id)
			if not webhook:
				self.logger.error(f"Webhook endpoint not found: {delivery.webhook_id}")
				return False
			
			# Prepare webhook payload
			payload = {
				'id': delivery.delivery_id,
				'event_type': delivery.event_type,
				'timestamp': delivery.created_at.isoformat(),
				'data': delivery.payload
			}
			
			# Generate signature for webhook security
			signature = self._generate_webhook_signature(webhook.secret, payload)
			
			headers = {
				'Content-Type': 'application/json',
				'X-APG-Signature': signature,
				'X-APG-Event-Type': delivery.event_type,
				'X-APG-Delivery-ID': delivery.delivery_id,
				'User-Agent': 'APG-Billing-Webhook/1.0'
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(
					webhook.url,
					json=payload,
					headers=headers,
					timeout=aiohttp.ClientTimeout(total=30)
				) as response:
					delivery.http_status_code = response.status
					delivery.response_body = await response.text()
					
					if 200 <= response.status < 300:
						return True
					else:
						self.logger.warning(
							f"Webhook delivery failed with status {response.status}: {delivery.delivery_id}"
						)
						return False
		
		except Exception as e:
			delivery.error_message = str(e)
			self.logger.error(f"Webhook delivery exception: {e}")
			return False

	def _generate_webhook_signature(self, secret: str, payload: Dict[str, Any]) -> str:
		"""Generate webhook signature for security verification"""
		import hmac
		import hashlib
		import json
		
		# Convert payload to canonical JSON string
		payload_string = json.dumps(payload, sort_keys=True, separators=(',', ':'))
		
		# Generate HMAC signature
		signature = hmac.new(
			secret.encode('utf-8'),
			payload_string.encode('utf-8'),
			hashlib.sha256
		).hexdigest()
		
		return f"sha256={signature}"

	async def _update_webhook_delivery(self, delivery: WebhookDelivery) -> None:
		"""Update webhook delivery record"""
		try:
			# Update the delivery in storage
			self.webhook_deliveries[delivery.delivery_id] = delivery
			
			# Log delivery metrics
			self._log_webhook_metrics(delivery)
			
		except Exception as e:
			self.logger.error(f"Failed to update webhook delivery: {e}")

	def _log_webhook_metrics(self, delivery: WebhookDelivery) -> None:
		"""Log webhook delivery metrics for monitoring"""
		metrics = {
			'delivery_id': delivery.delivery_id,
			'webhook_id': delivery.webhook_id,
			'event_type': delivery.event_type,
			'status': delivery.status,
			'retry_count': delivery.retry_count,
			'http_status_code': delivery.http_status_code,
			'processing_time_ms': (
				(delivery.delivered_at or datetime.utcnow()) - delivery.created_at
			).total_seconds() * 1000
		}
		
		self.logger.info(f"Webhook delivery metrics: {metrics}")
	
	async def _generate_sdk_configurations(self) -> None:
		"""Generate SDK configurations for multiple languages"""
		
		# Python SDK
		self.sdk_configurations[SDKLanguage.PYTHON] = SDKConfiguration(
			language=SDKLanguage.PYTHON,
			package_name="apg-billing",
			version="1.0.0",
			description="APG Billing API Python SDK with async support",
			repository_url="https://github.com/datacraft/apg-billing-python",
			async_support=True
		)
		
		# JavaScript/TypeScript SDK
		self.sdk_configurations[SDKLanguage.TYPESCRIPT] = SDKConfiguration(
			language=SDKLanguage.TYPESCRIPT,
			package_name="@datacraft/apg-billing",
			version="1.0.0",
			description="APG Billing API TypeScript SDK with full type safety",
			repository_url="https://github.com/datacraft/apg-billing-typescript"
		)
		
		# Go SDK
		self.sdk_configurations[SDKLanguage.GO] = SDKConfiguration(
			language=SDKLanguage.GO,
			package_name="github.com/datacraft/apg-billing-go",
			version="v1.0.0",
			description="APG Billing API Go SDK with comprehensive error handling"
		)
		
		# Add more languages as needed
		
		self.logger.info(f"✅ Generated {len(self.sdk_configurations)} SDK configurations")
	
	async def _create_interactive_documentation(self) -> None:
		"""Create interactive API documentation"""
		
		openapi_spec = {
			"openapi": "3.0.3",
			"info": {
				"title": "APG Billing API",
				"version": "1.0.0",
				"description": "Complete billing and subscription management API",
				"contact": {
					"name": "Datacraft Support",
					"email": "support@datacraft.co.ke",
					"url": "https://datacraft.co.ke/support"
				},
				"license": {
					"name": "MIT",
					"url": "https://opensource.org/licenses/MIT"
				}
			},
			"servers": [
				{
					"url": "https://api.datacraft.co.ke",
					"description": "Production server"
				},
				{
					"url": "https://sandbox-api.datacraft.co.ke",
					"description": "Sandbox server"
				}
			],
			"security": [
				{"ApiKeyAuth": []},
				{"BearerAuth": []}
			],
			"components": {
				"securitySchemes": {
					"ApiKeyAuth": {
						"type": "apiKey",
						"in": "header",
						"name": "X-API-Key"
					},
					"BearerAuth": {
						"type": "http",
						"scheme": "bearer",
						"bearerFormat": "JWT"
					}
				},
				"schemas": self._generate_api_schemas()
			},
			"paths": self._generate_api_paths(),
			"webhooks": self._generate_webhook_docs()
		}
		
		self.interactive_docs = openapi_spec
		self.logger.info("✅ Interactive API documentation created")
	
	def _generate_api_schemas(self) -> Dict[str, Any]:
		"""Generate OpenAPI schemas"""
		return {
			"Customer": {
				"type": "object",
				"required": ["id", "email", "name", "created_at"],
				"properties": {
					"id": {"type": "string", "example": "cust_123456"},
					"email": {"type": "string", "format": "email"},
					"name": {"type": "string"},
					"phone": {"type": "string"},
					"created_at": {"type": "string", "format": "date-time"},
					"metadata": {"type": "object"}
				}
			},
			"Subscription": {
				"type": "object",
				"required": ["id", "customer_id", "plan_id", "status"],
				"properties": {
					"id": {"type": "string", "example": "sub_123456"},
					"customer_id": {"type": "string"},
					"plan_id": {"type": "string"},
					"status": {"type": "string", "enum": ["active", "cancelled", "past_due"]},
					"billing_period": {"type": "string", "enum": ["monthly", "yearly"]},
					"created_at": {"type": "string", "format": "date-time"}
				}
			},
			"Payment": {
				"type": "object",
				"required": ["id", "customer_id", "amount", "currency", "status"],
				"properties": {
					"id": {"type": "string", "example": "pay_123456"},
					"customer_id": {"type": "string"},
					"amount": {"type": "number", "example": 29.99},
					"currency": {"type": "string", "example": "USD"},
					"status": {"type": "string", "enum": ["succeeded", "failed", "pending"]},
					"created_at": {"type": "string", "format": "date-time"}
				}
			},
			"WebhookEndpoint": {
				"type": "object",
				"required": ["id", "url", "events", "active"],
				"properties": {
					"id": {"type": "string"},
					"url": {"type": "string", "format": "uri"},
					"events": {"type": "array", "items": {"type": "string"}},
					"active": {"type": "boolean"},
					"created_at": {"type": "string", "format": "date-time"}
				}
			},
			"AnalyticsDashboard": {
				"type": "object",
				"properties": {
					"scope": {"type": "string"},
					"kpis": {"type": "object"},
					"mrr_metrics": {"type": "object"},
					"customer_health": {"type": "object"},
					"insights": {"type": "array", "items": {"type": "object"}}
				}
			}
		}
	
	def _generate_api_paths(self) -> Dict[str, Any]:
		"""Generate OpenAPI paths from endpoints"""
		paths = {}
		
		for endpoint_name, endpoint in self.api_endpoints.items():
			if endpoint.path not in paths:
				paths[endpoint.path] = {}
			
			paths[endpoint.path][endpoint.method.lower()] = {
				"summary": endpoint.summary,
				"description": endpoint.description,
				"tags": endpoint.tags,
				"security": [{"ApiKeyAuth": []}] if endpoint.requires_auth else [],
				"parameters": endpoint.parameters,
				"responses": endpoint.responses
			}
			
			if endpoint.request_body:
				paths[endpoint.path][endpoint.method.lower()]["requestBody"] = {
					"required": True,
					"content": {
						"application/json": {
							"schema": endpoint.request_body
						}
					}
				}
		
		return paths
	
	def _generate_webhook_docs(self) -> Dict[str, Any]:
		"""Generate webhook documentation"""
		webhook_docs = {}
		
		for event_type in WebhookEventType:
			webhook_docs[event_type.value] = {
				"post": {
					"summary": f"Webhook for {event_type.value}",
					"description": f"Sent when {event_type.value.replace('.', ' ')} occurs",
					"requestBody": {
						"required": True,
						"content": {
							"application/json": {
								"schema": {
									"type": "object",
									"properties": {
										"id": {"type": "string"},
										"type": {"type": "string", "example": event_type.value},
										"created": {"type": "integer"},
										"data": {"type": "object"}
									}
								}
							}
						}
					},
					"responses": {
						"200": {"description": "Webhook received successfully"}
					}
				}
			}
		
		return webhook_docs
	
	async def _setup_sandbox_environment(self) -> None:
		"""Setup sandbox environment for testing"""
		
		# Create sandbox-specific configurations
		self.sandbox_config = {
			"base_url": "https://sandbox-api.datacraft.co.ke",
			"features": {
				"test_credit_cards": True,
				"webhook_testing": True,
				"rate_limit_bypass": True,
				"debug_mode": True
			},
			"test_data": {
				"customers": self._generate_test_customers(),
				"plans": self._generate_test_plans(),
				"payment_methods": self._generate_test_payment_methods()
			}
		}
		
		self.logger.info("✅ Sandbox environment configured")
	
	def _generate_test_customers(self) -> List[Dict[str, Any]]:
		"""Generate test customer data for sandbox"""
		return [
			{
				"id": "cust_test_001",
				"email": "john.doe@example.com",
				"name": "John Doe",
				"description": "Test customer for API integration"
			},
			{
				"id": "cust_test_002",
				"email": "jane.smith@example.com",
				"name": "Jane Smith",
				"description": "Test customer with subscription"
			}
		]
	
	def _generate_test_plans(self) -> List[Dict[str, Any]]:
		"""Generate test billing plans for sandbox"""
		return [
			{
				"id": "plan_test_basic",
				"name": "Test Basic Plan",
				"amount": 999,  # $9.99
				"currency": "USD",
				"interval": "month"
			},
			{
				"id": "plan_test_pro",
				"name": "Test Pro Plan",
				"amount": 2999,  # $29.99
				"currency": "USD",
				"interval": "month"
			}
		]
	
	def _generate_test_payment_methods(self) -> List[Dict[str, Any]]:
		"""Generate test payment methods for sandbox"""
		return [
			{
				"id": "pm_test_visa",
				"type": "card",
				"card": {
					"brand": "visa",
					"last4": "4242",
					"exp_month": 12,
					"exp_year": 2030
				},
				"description": "Test Visa card (always succeeds)"
			},
			{
				"id": "pm_test_decline",
				"type": "card",
				"card": {
					"brand": "visa",
					"last4": "0002",
					"exp_month": 12,
					"exp_year": 2030
				},
				"description": "Test card (always declines)"
			}
		]
	
	async def generate_sdk(self, language: SDKLanguage, output_path: str = None) -> Dict[str, Any]:
		"""Generate SDK for specified language"""
		try:
			config = self.sdk_configurations.get(language)
			if not config:
				raise ValueError(f"No SDK configuration found for {language.value}")
			
			# Generate SDK based on OpenAPI spec and language
			sdk_files = await self._generate_sdk_files(language, config)
			
			sdk_info = {
				"language": language.value,
				"package_name": config.package_name,
				"version": config.version,
				"files_generated": len(sdk_files),
				"features": {
					"async_support": config.async_support,
					"retry_logic": config.retry_logic,
					"rate_limiting": config.rate_limiting,
					"custom_types": config.custom_types
				},
				"installation": self._generate_installation_instructions(language, config),
				"quick_start": self._generate_quick_start_guide(language, config),
				"generated_at": datetime.utcnow().isoformat()
			}
			
			# Log SDK generation
			await self.audit_system.log_audit_event({
				"event_type": AuditEventType.SYSTEM_CONFIG.value,
				"user_id": "system",
				"resource_type": "sdk_generation",
				"resource_id": f"sdk_{language.value}_{datetime.utcnow().strftime('%Y%m%d')}",
				"action": "sdk_generated",
				"description": f"SDK generated for {language.value}",
				"metadata": {
					"language": language.value,
					"package_name": config.package_name,
					"version": config.version
				}
			})
			
			return sdk_info
			
		except Exception as e:
			self.logger.error(f"SDK generation failed for {language.value}: {e}")
			raise
	
	async def _generate_sdk_files(self, language: SDKLanguage, config: SDKConfiguration) -> Dict[str, str]:
		"""Generate SDK files for specific language"""
		files = {}
		
		if language == SDKLanguage.PYTHON:
			files.update(await self._generate_python_sdk(config))
		elif language == SDKLanguage.TYPESCRIPT:
			files.update(await self._generate_typescript_sdk(config))
		elif language == SDKLanguage.GO:
			files.update(await self._generate_go_sdk(config))
		# Add more languages as needed
		
		return files
	
	async def _generate_python_sdk(self, config: SDKConfiguration) -> Dict[str, str]:
		"""Generate Python SDK files"""
		files = {}
		
		# Main client file
		files["apg_billing/client.py"] = '''"""
APG Billing API Python SDK
"""

import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
from decimal import Decimal


class APGBillingClient:
    """APG Billing API client with async support"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.datacraft.co.ke"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new customer"""
        async with self.session.post(
            f"{self.base_url}/v1/customers",
            json=customer_data
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def create_subscription(self, subscription_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new subscription"""
        async with self.session.post(
            f"{self.base_url}/v1/subscriptions",
            json=subscription_data
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a payment"""
        async with self.session.post(
            f"{self.base_url}/v1/payments",
            json=payment_data
        ) as response:
            response.raise_for_status()
            return await response.json()
'''
		
		# Setup file
		files["setup.py"] = f'''from setuptools import setup, find_packages

setup(
    name="{config.package_name}",
    version="{config.version}",
    description="{config.description}",
    author="{config.author}",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8+",
    ],
)'''
		
		# README
		files["README.md"] = f'''# {config.package_name}

{config.description}

## Installation

```bash
pip install {config.package_name}
```

## Quick Start

```python
import asyncio
from apg_billing import APGBillingClient

async def main():
    async with APGBillingClient("your_api_key") as client:
        # Create a customer
        customer = await client.create_customer({{
            "email": "customer@example.com",
            "name": "John Doe"
        }})
        
        # Create a subscription
        subscription = await client.create_subscription({{
            "customer_id": customer["id"],
            "plan_id": "plan_pro_monthly"
        }})

asyncio.run(main())
```

## Documentation

Full API documentation is available at: https://docs.datacraft.co.ke/billing-api
'''
		
		return files
	
	async def _generate_typescript_sdk(self, config: SDKConfiguration) -> Dict[str, str]:
		"""Generate TypeScript SDK files"""
		files = {}
		
		# Main client file
		files["src/client.ts"] = '''import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

export interface CustomerData {
  email: string;
  name: string;
  phone?: string;
  metadata?: Record<string, any>;
}

export interface SubscriptionData {
  customer_id: string;
  plan_id: string;
  billing_period?: 'monthly' | 'yearly';
  trial_days?: number;
  metadata?: Record<string, any>;
}

export interface PaymentData {
  customer_id: string;
  amount: number;
  currency: string;
  payment_method_id?: string;
  description?: string;
  metadata?: Record<string, any>;
}

export class APGBillingClient {
  private api: AxiosInstance;

  constructor(apiKey: string, baseURL: string = 'https://api.datacraft.co.ke') {
    this.api = axios.create({
      baseURL,
      headers: {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
      },
    });
  }

  async createCustomer(customerData: CustomerData): Promise<any> {
    const response = await this.api.post('/v1/customers', customerData);
    return response.data;
  }

  async createSubscription(subscriptionData: SubscriptionData): Promise<any> {
    const response = await this.api.post('/v1/subscriptions', subscriptionData);
    return response.data;
  }

  async processPayment(paymentData: PaymentData): Promise<any> {
    const response = await this.api.post('/v1/payments', paymentData);
    return response.data;
  }
}'''
		
		# Package.json
		files["package.json"] = f'''{{
  "name": "{config.package_name}",
  "version": "{config.version}",
  "description": "{config.description}",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {{
    "build": "tsc",
    "test": "jest",
    "prepublishOnly": "npm run build"
  }},
  "dependencies": {{
    "axios": "^1.0.0"
  }},
  "devDependencies": {{
    "typescript": "^4.0.0",
    "@types/node": "^18.0.0",
    "jest": "^29.0.0"
  }},
  "keywords": ["billing", "api", "subscription", "payments"],
  "author": "{config.author}",
  "license": "{config.license}"
}}'''
		
		return files
	
	async def _generate_go_sdk(self, config: SDKConfiguration) -> Dict[str, str]:
		"""Generate Go SDK files"""
		files = {}
		
		# Main client file
		files["client.go"] = '''package apgbilling

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type Client struct {
	APIKey  string
	BaseURL string
	HTTPClient *http.Client
}

type CustomerData struct {
	Email    string                 `json:"email"`
	Name     string                 `json:"name"`
	Phone    string                 `json:"phone,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type SubscriptionData struct {
	CustomerID     string                 `json:"customer_id"`
	PlanID         string                 `json:"plan_id"`
	BillingPeriod  string                 `json:"billing_period,omitempty"`
	TrialDays      int                    `json:"trial_days,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:  apiKey,
		BaseURL: "https://api.datacraft.co.ke",
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *Client) CreateCustomer(data CustomerData) (map[string]interface{}, error) {
	return c.makeRequest("POST", "/v1/customers", data)
}

func (c *Client) CreateSubscription(data SubscriptionData) (map[string]interface{}, error) {
	return c.makeRequest("POST", "/v1/subscriptions", data)
}

func (c *Client) makeRequest(method, path string, data interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(method, c.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}

	req.Header.Set("X-API-Key", c.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&result)
	return result, err
}'''
		
		# Go module file
		files["go.mod"] = f'''module {config.package_name}

go 1.19

require ()'''
		
		return files
	
	def _generate_installation_instructions(self, language: SDKLanguage, config: SDKConfiguration) -> Dict[str, str]:
		"""Generate installation instructions for SDK"""
		if language == SDKLanguage.PYTHON:
			return {
				"pip": f"pip install {config.package_name}",
				"conda": f"conda install -c conda-forge {config.package_name}"
			}
		elif language == SDKLanguage.TYPESCRIPT:
			return {
				"npm": f"npm install {config.package_name}",
				"yarn": f"yarn add {config.package_name}"
			}
		elif language == SDKLanguage.GO:
			return {
				"go_get": f"go get {config.package_name}"
			}
		return {}
	
	def _generate_quick_start_guide(self, language: SDKLanguage, config: SDKConfiguration) -> str:
		"""Generate quick start guide for SDK"""
		if language == SDKLanguage.PYTHON:
			return '''```python
import asyncio
from apg_billing import APGBillingClient

async def main():
    async with APGBillingClient("your_api_key") as client:
        customer = await client.create_customer({
            "email": "customer@example.com",
            "name": "John Doe"
        })
        print(f"Created customer: {customer['id']}")

asyncio.run(main())
```'''
		elif language == SDKLanguage.TYPESCRIPT:
			return '''```typescript
import { APGBillingClient } from '@datacraft/apg-billing';

const client = new APGBillingClient('your_api_key');

async function main() {
  const customer = await client.createCustomer({
    email: 'customer@example.com',
    name: 'John Doe'
  });
  console.log(`Created customer: ${customer.id}`);
}

main();
```'''
		elif language == SDKLanguage.GO:
			return '''```go
package main

import (
    "fmt"
    "log"
    "github.com/datacraft/apg-billing-go"
)

func main() {
    client := apgbilling.NewClient("your_api_key")
    
    customer, err := client.CreateCustomer(apgbilling.CustomerData{
        Email: "customer@example.com",
        Name:  "John Doe",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Created customer: %s\\n", customer["id"])
}
```'''
		return ""
	
	async def create_webhook_endpoint(self, webhook_data: Dict[str, Any]) -> WebhookEndpoint:
		"""Create a new webhook endpoint"""
		try:
			# Validate webhook URL
			if not webhook_data.get('url'):
				raise ValueError("Webhook URL is required")
			
			# Generate secret if not provided
			if not webhook_data.get('secret'):
				webhook_data['secret'] = f"whsec_{uuid7str()}"
			
			webhook = WebhookEndpoint(
				url=webhook_data['url'],
				events=[WebhookEventType(event) for event in webhook_data['events']],
				secret=webhook_data['secret'],
				headers=webhook_data.get('headers', {}),
				metadata=webhook_data.get('metadata', {})
			)
			
			self.webhook_endpoints[webhook.id] = webhook
			
			# Log webhook creation
			await self.audit_system.log_audit_event({
				"event_type": AuditEventType.SYSTEM_CONFIG.value,
				"user_id": "system",
				"resource_type": "webhook_endpoint",
				"resource_id": webhook.id,
				"action": "webhook_created",
				"description": f"Webhook endpoint created: {webhook.url}",
				"metadata": {
					"url": webhook.url,
					"events": [e.value for e in webhook.events]
				}
			})
			
			return webhook
			
		except Exception as e:
			self.logger.error(f"Webhook creation failed: {e}")
			raise
	
	async def send_webhook_event(self, event_type: WebhookEventType, data: Dict[str, Any]) -> None:
		"""Send webhook event to all subscribed endpoints"""
		try:
			# Find webhooks subscribed to this event
			subscribed_webhooks = [
				webhook for webhook in self.webhook_endpoints.values()
				if webhook.active and event_type in webhook.events
			]
			
			if not subscribed_webhooks:
				return
			
			# Prepare webhook payload
			webhook_payload = {
				"id": uuid7str(),
				"type": event_type.value,
				"created": int(datetime.utcnow().timestamp()),
				"data": data
			}
			
			# Send to all subscribed webhooks
			for webhook in subscribed_webhooks:
				asyncio.create_task(self._deliver_webhook(webhook, webhook_payload))
			
		except Exception as e:
			self.logger.error(f"Webhook event sending failed: {e}")
	
	async def _deliver_webhook(self, webhook: WebhookEndpoint, payload: Dict[str, Any]) -> None:
		"""Deliver webhook payload to endpoint"""
		try:
			import aiohttp
			import hmac
			import hashlib
			
			# Create signature
			payload_str = json.dumps(payload, sort_keys=True)
			signature = hmac.new(
				webhook.secret.encode(),
				payload_str.encode(),
				hashlib.sha256
			).hexdigest()
			
			headers = {
				"Content-Type": "application/json",
				"X-Webhook-Signature": f"sha256={signature}",
				**webhook.headers
			}
			
			async with aiohttp.ClientSession() as session:
				async with session.post(
					webhook.url,
					json=payload,
					headers=headers,
					timeout=aiohttp.ClientTimeout(total=webhook.timeout_seconds)
				) as response:
					if response.status == 200:
						webhook.last_success = datetime.utcnow()
						webhook.failure_count = 0
					else:
						await self._handle_webhook_failure(webhook, response.status)
		
		except Exception as e:
			await self._handle_webhook_failure(webhook, str(e))
	
	async def _handle_webhook_failure(self, webhook: WebhookEndpoint, error: Union[int, str]) -> None:
		"""Handle webhook delivery failure"""
		webhook.last_failure = datetime.utcnow()
		webhook.failure_count += 1
		
		# Disable webhook after too many failures
		if webhook.failure_count >= webhook.max_retries:
			webhook.active = False
			self.logger.warning(f"Webhook {webhook.id} disabled after {webhook.failure_count} failures")
		
		self.logger.error(f"Webhook delivery failed for {webhook.url}: {error}")
	
	async def get_api_documentation(self, format: str = "openapi") -> Dict[str, Any]:
		"""Get API documentation in specified format"""
		if format == "openapi":
			return self.interactive_docs
		elif format == "postman":
			return await self._generate_postman_collection()
		else:
			raise ValueError(f"Unsupported documentation format: {format}")
	
	async def _generate_postman_collection(self) -> Dict[str, Any]:
		"""Generate Postman collection from API endpoints"""
		collection = {
			"info": {
				"name": "APG Billing API",
				"description": "Complete billing and subscription management API",
				"version": "1.0.0"
			},
			"auth": {
				"type": "apikey",
				"apikey": [
					{"key": "key", "value": "X-API-Key"},
					{"key": "value", "value": "{{api_key}}"}
				]
			},
			"variable": [
				{"key": "base_url", "value": "https://api.datacraft.co.ke"},
				{"key": "api_key", "value": "your_api_key_here"}
			],
			"item": []
		}
		
		# Convert endpoints to Postman format
		for endpoint_name, endpoint in self.api_endpoints.items():
			item = {
				"name": endpoint.summary,
				"request": {
					"method": endpoint.method,
					"header": [],
					"url": {
						"raw": "{{base_url}}" + endpoint.path,
						"host": ["{{base_url}}"],
						"path": endpoint.path.strip("/").split("/")
					}
				},
				"response": []
			}
			
			if endpoint.request_body:
				item["request"]["body"] = {
					"mode": "raw",
					"raw": json.dumps(endpoint.examples[0]["request"] if endpoint.examples else {}, indent=2),
					"options": {"raw": {"language": "json"}}
				}
			
			collection["item"].append(item)
		
		return collection


# Global developer API ecosystem
_api_ecosystem_instance: Optional[DeveloperAPIEcosystem] = None

def get_developer_api_ecosystem() -> DeveloperAPIEcosystem:
	"""Get global developer API ecosystem instance"""
	global _api_ecosystem_instance
	if _api_ecosystem_instance is None:
		_api_ecosystem_instance = DeveloperAPIEcosystem()
	return _api_ecosystem_instance


__all__ = [
	'DeveloperAPIEcosystem',
	'APIEndpoint',
	'WebhookEndpoint',
	'SDKConfiguration',
	'APIVersion',
	'SDKLanguage',
	'WebhookEventType',
	'get_developer_api_ecosystem'
]