"""
APG Accounts Receivable - FastAPI Endpoints
Comprehensive REST API for accounts receivable operations with AI integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union
from uuid import UUID

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

from apg.core.fastapi_integration import APGFastAPI, APGDependencies
from apg.auth_rbac import get_current_user, get_current_tenant, require_permission
from apg.audit_compliance import audit_endpoint
from apg.rate_limiting import rate_limit

from .models import (
	ARCustomer, ARInvoice, ARPayment, ARCollectionActivity, ARCreditAssessment, ARDispute,
	ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus, ARCollectionPriority
)
from .service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)
from .ai_credit_scoring import APGCreditScoringService, CreditScoringResult
from .ai_collections_optimization import (
	APGCollectionsAIService, CollectionStrategyRecommendation, CollectionCampaignPlan
)
from .ai_cashflow_forecasting import (
	APGCashFlowForecastingService, CashFlowDataPoint, CashFlowForecastSummary,
	CashFlowScenarioComparison, CashFlowForecastInput
)
from .views import (
	ARCustomerCreateView, ARCustomerUpdateView, ARCustomerDetailView,
	ARInvoiceCreateView, ARInvoiceUpdateView, ARInvoiceDetailView,
	ARPaymentCreateView, ARPaymentUpdateView, ARPaymentDetailView
)


# =============================================================================
# Pydantic Response Models
# =============================================================================

class ARCustomerResponse(BaseModel):
	"""Customer response model."""
	id: str
	tenant_id: str
	customer_code: str
	legal_name: str
	display_name: Optional[str] = None
	customer_type: ARCustomerType
	status: ARCustomerStatus
	credit_limit: Decimal
	payment_terms_days: int
	total_outstanding: Decimal
	overdue_amount: Decimal
	contact_email: Optional[str] = None
	contact_phone: Optional[str] = None
	billing_address: Optional[str] = None
	created_at: datetime
	updated_at: datetime

	class Config:
		from_attributes = True
		json_encoders = {
			Decimal: float,
			datetime: lambda v: v.isoformat()
		}


class ARInvoiceResponse(BaseModel):
	"""Invoice response model."""
	id: str
	tenant_id: str
	customer_id: str
	invoice_number: str
	invoice_date: date
	due_date: date
	total_amount: Decimal
	paid_amount: Decimal
	outstanding_amount: Decimal
	currency_code: str
	status: ARInvoiceStatus
	payment_status: str
	description: Optional[str] = None
	created_at: datetime
	updated_at: datetime

	class Config:
		from_attributes = True
		json_encoders = {
			Decimal: float,
			date: lambda v: v.isoformat(),
			datetime: lambda v: v.isoformat()
		}


class ARPaymentResponse(BaseModel):
	"""Payment response model."""
	id: str
	tenant_id: str
	customer_id: str
	payment_reference: str
	payment_date: date
	payment_amount: Decimal
	payment_method: str
	status: ARPaymentStatus
	currency_code: str
	bank_reference: Optional[str] = None
	notes: Optional[str] = None
	processed_at: Optional[datetime] = None
	created_at: datetime

	class Config:
		from_attributes = True
		json_encoders = {
			Decimal: float,
			date: lambda v: v.isoformat(),
			datetime: lambda v: v.isoformat()
		}


class ARCollectionActivityResponse(BaseModel):
	"""Collection activity response model."""
	id: str
	tenant_id: str
	customer_id: str
	activity_type: str
	activity_date: date
	priority: ARCollectionPriority
	contact_method: str
	outcome: str
	status: str
	notes: Optional[str] = None
	created_at: datetime

	class Config:
		from_attributes = True
		json_encoders = {
			date: lambda v: v.isoformat(),
			datetime: lambda v: v.isoformat()
		}


class APIResponse(BaseModel):
	"""Generic API response wrapper."""
	success: bool = True
	message: Optional[str] = None
	data: Optional[Any] = None
	errors: Optional[List[str]] = None


class PaginatedResponse(BaseModel):
	"""Paginated response model."""
	items: List[Any]
	total: int
	page: int
	per_page: int
	pages: int
	has_next: bool
	has_prev: bool


# =============================================================================
# Request Models
# =============================================================================

class CreditAssessmentRequest(BaseModel):
	"""Credit assessment request."""
	customer_id: str
	assessment_type: str = Field(default="standard", regex="^(standard|comprehensive|monitoring)$")
	include_explanations: bool = True
	generate_recommendations: bool = True
	update_customer_record: bool = False
	notes: Optional[str] = Field(None, max_length=1000)


class CollectionsOptimizationRequest(BaseModel):
	"""Collections optimization request."""
	customer_ids: Optional[List[str]] = None
	optimization_scope: str = Field(..., regex="^(single|batch|campaign)$")
	scenario_type: str = Field(default="realistic", regex="^(realistic|optimistic|pessimistic|custom)$")
	include_ai_recommendations: bool = True
	generate_campaign_plan: bool = False


class CashFlowForecastRequest(BaseModel):
	"""Cash flow forecast request."""
	forecast_start_date: date = Field(default_factory=date.today)
	forecast_end_date: date
	forecast_period: str = Field(default="daily", regex="^(daily|weekly|monthly)$")
	scenario_type: str = Field(default="realistic", regex="^(realistic|optimistic|pessimistic|comparison)$")
	include_seasonal_trends: bool = True
	include_external_factors: bool = True
	confidence_level: float = Field(default=0.95, ge=0.1, le=0.99)

	@validator('forecast_end_date')
	def validate_end_date(cls, v, values):
		if 'forecast_start_date' in values and v <= values['forecast_start_date']:
			raise ValueError('End date must be after start date')
		return v


# =============================================================================
# Dependencies
# =============================================================================

async def get_ar_customer_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> ARCustomerService:
	"""Get AR customer service instance."""
	return ARCustomerService(tenant_id, user_id)


async def get_ar_invoice_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> ARInvoiceService:
	"""Get AR invoice service instance."""
	return ARInvoiceService(tenant_id, user_id)


async def get_ar_payment_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> ARCashApplicationService:
	"""Get AR payment service instance."""
	return ARCashApplicationService(tenant_id, user_id)


async def get_ar_collections_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> ARCollectionsService:
	"""Get AR collections service instance."""
	return ARCollectionsService(tenant_id, user_id)


async def get_ar_analytics_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> ARAnalyticsService:
	"""Get AR analytics service instance."""
	return ARAnalyticsService(tenant_id, user_id)


async def get_credit_scoring_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> APGCreditScoringService:
	"""Get credit scoring service instance."""
	return APGCreditScoringService(tenant_id, user_id, None)


async def get_collections_ai_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> APGCollectionsAIService:
	"""Get collections AI service instance."""
	return APGCollectionsAIService(tenant_id, user_id, None)


async def get_cashflow_forecasting_service(
	tenant_id: str = Depends(get_current_tenant),
	user_id: str = Depends(get_current_user)
) -> APGCashFlowForecastingService:
	"""Get cash flow forecasting service instance."""
	return APGCashFlowForecastingService(tenant_id, user_id, None)


# =============================================================================
# FastAPI App Creation
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan manager."""
	# Startup
	print("Starting APG Accounts Receivable API...")
	yield
	# Shutdown
	print("Shutting down APG Accounts Receivable API...")


def create_ar_api() -> FastAPI:
	"""Create FastAPI application for Accounts Receivable."""
	
	app = APGFastAPI(
		title="APG Accounts Receivable API",
		description="Comprehensive API for accounts receivable operations with AI integration",
		version="1.0.0",
		docs_url="/ar/docs",
		redoc_url="/ar/redoc",
		openapi_url="/ar/openapi.json",
		lifespan=lifespan
	)
	
	# Add CORS middleware
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],  # Configure appropriately for production
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	
	# Register all AR routes with the FastAPI app
	from .api_endpoints import register_ar_routes
	register_ar_routes(app)
	
	return app