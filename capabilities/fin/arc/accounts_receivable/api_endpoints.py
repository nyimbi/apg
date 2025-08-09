"""
APG Accounts Receivable - API Endpoints
FastAPI route definitions for all AR operations

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from typing import List, Optional
from datetime import date, datetime

# Direct imports to avoid circular dependency
from pydantic import BaseModel, Field, validator
from decimal import Decimal
from .service import (
	ARCustomerService, ARInvoiceService, ARCollectionsService,
	ARCashApplicationService, ARAnalyticsService
)
from .ai_credit_scoring import APGCreditScoringService
from .ai_collections_optimization import APGCollectionsAIService
from .ai_cashflow_forecasting import APGCashFlowForecastingService, CashFlowForecastInput
from .models import ARCustomerType, ARCustomerStatus, ARInvoiceStatus, ARPaymentStatus, ARCollectionPriority
from apg.auth_rbac import require_permission, get_current_user, get_current_tenant
from apg.audit_compliance import audit_endpoint
from apg.rate_limiting import rate_limit


# =============================================================================
# Response Models (copied to avoid circular import)
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
	data: Optional[dict] = None
	errors: Optional[List[str]] = None


class PaginatedResponse(BaseModel):
	"""Paginated response model."""
	items: List[dict]
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
# Customer Management Endpoints
# =============================================================================

customer_router = APIRouter(prefix="/customers", tags=["Customers"])


@customer_router.get("", response_model=List[ARCustomerResponse])
@require_permission("ar_customer_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("list_customers")
async def list_customers(
	page: int = Query(1, ge=1),
	per_page: int = Query(20, ge=1, le=100),
	customer_type: Optional[str] = Query(None),
	status: Optional[str] = Query(None),
	search: Optional[str] = Query(None),
	service: ARCustomerService = Depends(get_ar_customer_service)
):
	"""List customers with filtering and pagination."""
	try:
		customers = await service.get_customers_filtered(
			page=page,
			per_page=per_page,
			customer_type=customer_type,
			status=status,
			search=search
		)
		return [ARCustomerResponse.from_orm(customer) for customer in customers]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@customer_router.get("/{customer_id}", response_model=ARCustomerResponse)
@require_permission("ar_customer_view")
@rate_limit(requests=200, window=60)
@audit_endpoint("get_customer")
async def get_customer(
	customer_id: str = Path(...),
	service: ARCustomerService = Depends(get_ar_customer_service)
):
	"""Get customer by ID."""
	try:
		customer = await service.get_customer(customer_id)
		if not customer:
			raise HTTPException(status_code=404, detail="Customer not found")
		return ARCustomerResponse.from_orm(customer)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@customer_router.post("", response_model=ARCustomerResponse, status_code=status.HTTP_201_CREATED)
@require_permission("ar_customer_create")
@rate_limit(requests=20, window=60)
@audit_endpoint("create_customer")
async def create_customer(
	customer_data: dict = Body(...),
	service: ARCustomerService = Depends(get_ar_customer_service)
):
	"""Create new customer."""
	try:
		customer = await service.create_customer(customer_data)
		return ARCustomerResponse.from_orm(customer)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@customer_router.put("/{customer_id}", response_model=ARCustomerResponse)
@require_permission("ar_customer_edit")
@rate_limit(requests=50, window=60)
@audit_endpoint("update_customer")
async def update_customer(
	customer_id: str = Path(...),
	customer_data: dict = Body(...),
	service: ARCustomerService = Depends(get_ar_customer_service)
):
	"""Update customer."""
	try:
		customer = await service.update_customer(customer_id, customer_data)
		if not customer:
			raise HTTPException(status_code=404, detail="Customer not found")
		return ARCustomerResponse.from_orm(customer)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@customer_router.post("/{customer_id}/credit-assessment", response_model=APIResponse)
@require_permission("ar_credit_assessment")
@rate_limit(requests=10, window=60)
@audit_endpoint("assess_customer_credit")
async def assess_customer_credit(
	customer_id: str = Path(...),
	request_data: CreditAssessmentRequest = Body(...),
	service: APGCreditScoringService = Depends(get_credit_scoring_service)
):
	"""Perform AI credit assessment for customer."""
	try:
		assessment = await service.assess_customer_credit(
			customer_id=customer_id,
			assessment_options={
				'include_explanations': request_data.include_explanations,
				'generate_recommendations': request_data.generate_recommendations,
				'update_customer_record': request_data.update_customer_record
			}
		)
		return APIResponse(
			success=True,
			message="Credit assessment completed successfully",
			data=assessment.dict()
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@customer_router.get("/{customer_id}/summary", response_model=APIResponse)
@require_permission("ar_customer_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("get_customer_summary")
async def get_customer_summary(
	customer_id: str = Path(...),
	service: ARCustomerService = Depends(get_ar_customer_service)
):
	"""Get customer summary with analytics."""
	try:
		summary = await service.get_customer_summary(customer_id)
		return APIResponse(
			success=True,
			data=summary.dict()
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Invoice Management Endpoints
# =============================================================================

invoice_router = APIRouter(prefix="/invoices", tags=["Invoices"])


@invoice_router.get("", response_model=List[ARInvoiceResponse])
@require_permission("ar_invoice_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("list_invoices")
async def list_invoices(
	page: int = Query(1, ge=1),
	per_page: int = Query(20, ge=1, le=100),
	customer_id: Optional[str] = Query(None),
	status: Optional[str] = Query(None),
	date_from: Optional[date] = Query(None),
	date_to: Optional[date] = Query(None),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""List invoices with filtering and pagination."""
	try:
		invoices = await service.get_invoices_filtered(
			page=page,
			per_page=per_page,
			customer_id=customer_id,
			status=status,
			date_from=date_from,
			date_to=date_to
		)
		return [ARInvoiceResponse.from_orm(invoice) for invoice in invoices]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@invoice_router.get("/{invoice_id}", response_model=ARInvoiceResponse)
@require_permission("ar_invoice_view")
@rate_limit(requests=200, window=60)
@audit_endpoint("get_invoice")
async def get_invoice(
	invoice_id: str = Path(...),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""Get invoice by ID."""
	try:
		invoice = await service.get_invoice(invoice_id)
		if not invoice:
			raise HTTPException(status_code=404, detail="Invoice not found")
		return ARInvoiceResponse.from_orm(invoice)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@invoice_router.post("", response_model=ARInvoiceResponse, status_code=status.HTTP_201_CREATED)
@require_permission("ar_invoice_create")
@rate_limit(requests=20, window=60)
@audit_endpoint("create_invoice")
async def create_invoice(
	invoice_data: dict = Body(...),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""Create new invoice."""
	try:
		invoice = await service.create_invoice(invoice_data)
		return ARInvoiceResponse.from_orm(invoice)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@invoice_router.put("/{invoice_id}", response_model=ARInvoiceResponse)
@require_permission("ar_invoice_edit")
@rate_limit(requests=50, window=60)
@audit_endpoint("update_invoice")
async def update_invoice(
	invoice_id: str = Path(...),
	invoice_data: dict = Body(...),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""Update invoice."""
	try:
		invoice = await service.update_invoice(invoice_id, invoice_data)
		if not invoice:
			raise HTTPException(status_code=404, detail="Invoice not found")
		return ARInvoiceResponse.from_orm(invoice)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@invoice_router.post("/{invoice_id}/mark-overdue", response_model=APIResponse)
@require_permission("ar_invoice_edit")
@rate_limit(requests=20, window=60)
@audit_endpoint("mark_invoice_overdue")
async def mark_invoice_overdue(
	invoice_id: str = Path(...),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""Mark invoice as overdue."""
	try:
		await service.mark_invoice_overdue(invoice_id)
		return APIResponse(
			success=True,
			message="Invoice marked as overdue successfully"
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@invoice_router.get("/{invoice_id}/payment-prediction", response_model=APIResponse)
@require_permission("ar_invoice_view")
@rate_limit(requests=50, window=60)
@audit_endpoint("predict_payment_date")
async def predict_payment_date(
	invoice_id: str = Path(...),
	service: ARInvoiceService = Depends(get_ar_invoice_service)
):
	"""Predict payment date using AI."""
	try:
		prediction = await service.predict_payment_date(invoice_id)
		return APIResponse(
			success=True,
			data=prediction.dict()
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Payment Management Endpoints
# =============================================================================

payment_router = APIRouter(prefix="/payments", tags=["Payments"])


@payment_router.get("", response_model=List[ARPaymentResponse])
@require_permission("ar_payment_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("list_payments")
async def list_payments(
	page: int = Query(1, ge=1),
	per_page: int = Query(20, ge=1, le=100),
	customer_id: Optional[str] = Query(None),
	status: Optional[str] = Query(None),
	date_from: Optional[date] = Query(None),
	date_to: Optional[date] = Query(None),
	service: ARCashApplicationService = Depends(get_ar_payment_service)
):
	"""List payments with filtering and pagination."""
	try:
		payments = await service.get_payments_filtered(
			page=page,
			per_page=per_page,
			customer_id=customer_id,
			status=status,
			date_from=date_from,
			date_to=date_to
		)
		return [ARPaymentResponse.from_orm(payment) for payment in payments]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@payment_router.get("/{payment_id}", response_model=ARPaymentResponse)
@require_permission("ar_payment_view")
@rate_limit(requests=200, window=60)
@audit_endpoint("get_payment")
async def get_payment(
	payment_id: str = Path(...),
	service: ARCashApplicationService = Depends(get_ar_payment_service)
):
	"""Get payment by ID."""
	try:
		payment = await service.get_payment(payment_id)
		if not payment:
			raise HTTPException(status_code=404, detail="Payment not found")
		return ARPaymentResponse.from_orm(payment)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@payment_router.post("", response_model=ARPaymentResponse, status_code=status.HTTP_201_CREATED)
@require_permission("ar_payment_create")
@rate_limit(requests=20, window=60)
@audit_endpoint("create_payment")
async def create_payment(
	payment_data: dict = Body(...),
	service: ARCashApplicationService = Depends(get_ar_payment_service)
):
	"""Create new payment."""
	try:
		payment = await service.create_payment(payment_data)
		return ARPaymentResponse.from_orm(payment)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@payment_router.post("/{payment_id}/process", response_model=APIResponse)
@require_permission("ar_payment_process")
@rate_limit(requests=20, window=60)
@audit_endpoint("process_payment")
async def process_payment(
	payment_id: str = Path(...),
	service: ARCashApplicationService = Depends(get_ar_payment_service)
):
	"""Process payment."""
	try:
		await service.process_payment(payment_id)
		return APIResponse(
			success=True,
			message="Payment processed successfully"
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Collections Management Endpoints
# =============================================================================

collections_router = APIRouter(prefix="/collections", tags=["Collections"])


@collections_router.get("", response_model=List[ARCollectionActivityResponse])
@require_permission("ar_collections_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("list_collection_activities")
async def list_collection_activities(
	page: int = Query(1, ge=1),
	per_page: int = Query(20, ge=1, le=100),
	customer_id: Optional[str] = Query(None),
	activity_type: Optional[str] = Query(None),
	date_from: Optional[date] = Query(None),
	date_to: Optional[date] = Query(None),
	service: ARCollectionsService = Depends(get_ar_collections_service)
):
	"""List collection activities with filtering and pagination."""
	try:
		activities = await service.get_collection_activities_filtered(
			page=page,
			per_page=per_page,
			customer_id=customer_id,
			activity_type=activity_type,
			date_from=date_from,
			date_to=date_to
		)
		return [ARCollectionActivityResponse.from_orm(activity) for activity in activities]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@collections_router.post("/optimize", response_model=APIResponse)
@require_permission("ar_collections_optimization")
@rate_limit(requests=5, window=60)
@audit_endpoint("optimize_collections")
async def optimize_collections(
	request_data: CollectionsOptimizationRequest = Body(...),
	service: APGCollectionsAIService = Depends(get_collections_ai_service)
):
	"""Optimize collection strategies using AI."""
	try:
		if request_data.optimization_scope == "single":
			if not request_data.customer_ids or len(request_data.customer_ids) != 1:
				raise HTTPException(status_code=400, detail="Single customer ID required for single optimization")
			
			result = await service.optimize_collection_strategy(
				customer_id=request_data.customer_ids[0]
			)
		elif request_data.optimization_scope == "batch":
			if not request_data.customer_ids:
				raise HTTPException(status_code=400, detail="Customer IDs required for batch optimization")
			
			result = await service.batch_optimize_strategies(request_data.customer_ids)
		else:  # campaign
			result = await service.create_campaign_plan(
				customer_ids=request_data.customer_ids or []
			)
		
		return APIResponse(
			success=True,
			message="Collections optimization completed successfully",
			data=result.dict() if hasattr(result, 'dict') else result
		)
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@collections_router.post("/{invoice_id}/send-reminder", response_model=APIResponse)
@require_permission("ar_collections_activity")
@rate_limit(requests=10, window=60)
@audit_endpoint("send_payment_reminder")
async def send_payment_reminder(
	invoice_id: str = Path(...),
	service: ARCollectionsService = Depends(get_ar_collections_service)
):
	"""Send payment reminder for invoice."""
	try:
		await service.send_payment_reminder(invoice_id)
		return APIResponse(
			success=True,
			message="Payment reminder sent successfully"
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@collections_router.get("/metrics", response_model=APIResponse)
@require_permission("ar_collections_view")
@rate_limit(requests=50, window=60)
@audit_endpoint("get_collections_metrics")
async def get_collections_metrics(
	service: ARCollectionsService = Depends(get_ar_collections_service)
):
	"""Get collections performance metrics."""
	try:
		metrics = await service.get_collections_metrics()
		return APIResponse(
			success=True,
			data=metrics
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AI-Powered Analytics Endpoints
# =============================================================================

analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])


@analytics_router.post("/cashflow-forecast", response_model=APIResponse)
@require_permission("ar_cashflow_forecast")
@rate_limit(requests=5, window=60)
@audit_endpoint("generate_cashflow_forecast")
async def generate_cashflow_forecast(
	request_data: CashFlowForecastRequest = Body(...),
	service: APGCashFlowForecastingService = Depends(get_cashflow_forecasting_service)
):
	"""Generate AI-powered cash flow forecast."""
	try:
		forecast_input = CashFlowForecastInput(
			tenant_id=service.tenant_id,
			forecast_start_date=request_data.forecast_start_date,
			forecast_end_date=request_data.forecast_end_date,
			forecast_period=request_data.forecast_period,
			scenario_type=request_data.scenario_type,
			include_seasonal_trends=request_data.include_seasonal_trends,
			include_external_factors=request_data.include_external_factors,
			confidence_level=request_data.confidence_level
		)
		
		if request_data.scenario_type == "comparison":
			result = await service.generate_scenario_comparison(forecast_input)
		else:
			forecast_points, summary = await service.generate_cash_flow_forecast(forecast_input)
			result = {
				'forecast_points': [fp.dict() for fp in forecast_points],
				'summary': summary.dict()
			}
		
		return APIResponse(
			success=True,
			message="Cash flow forecast generated successfully",
			data=result
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/dashboard", response_model=APIResponse)
@require_permission("ar_dashboard_view")
@rate_limit(requests=100, window=60)
@audit_endpoint("get_dashboard_metrics")
async def get_dashboard_metrics(
	service: ARAnalyticsService = Depends(get_ar_analytics_service)
):
	"""Get AR dashboard metrics and KPIs."""
	try:
		metrics = await service.get_ar_dashboard_metrics()
		return APIResponse(
			success=True,
			data=metrics
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/aging-analysis", response_model=APIResponse)
@require_permission("ar_analytics_view")
@rate_limit(requests=50, window=60)
@audit_endpoint("get_aging_analysis")
async def get_aging_analysis(
	as_of_date: Optional[date] = Query(None),
	service: ARAnalyticsService = Depends(get_ar_analytics_service)
):
	"""Get accounts receivable aging analysis."""
	try:
		aging_analysis = await service.get_aging_analysis(as_of_date)
		return APIResponse(
			success=True,
			data=aging_analysis
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@analytics_router.get("/collection-performance", response_model=APIResponse)
@require_permission("ar_analytics_view")
@rate_limit(requests=50, window=60)
@audit_endpoint("get_collection_performance_metrics")
async def get_collection_performance_metrics(
	service: ARAnalyticsService = Depends(get_ar_analytics_service)
):
	"""Get collections performance metrics."""
	try:
		performance = await service.get_collection_performance_metrics()
		return APIResponse(
			success=True,
			data=performance
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Router Registration
# =============================================================================

def register_ar_routes(app: FastAPI):
	"""Register all AR routes with the FastAPI app."""
	
	app.include_router(customer_router, prefix="/api/ar")
	app.include_router(invoice_router, prefix="/api/ar")
	app.include_router(payment_router, prefix="/api/ar")
	app.include_router(collections_router, prefix="/api/ar")
	app.include_router(analytics_router, prefix="/api/ar")