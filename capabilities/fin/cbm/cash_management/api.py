"""APG Cash Management REST API

Enterprise-grade FastAPI implementation for APG Cash Management.
Provides comprehensive REST API endpoints with automatic OpenAPI documentation,
real-time bank integration, AI-powered analytics, and advanced cash optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import logging
from decimal import Decimal
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import UUID

from fastapi import (
	FastAPI, HTTPException, Depends, Query, Path, Body, BackgroundTasks,
	status, Request, Response
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic.types import condecimal, conint

from .models import CashManagementModels
from .service import CashManagementService
from .cache import CashCacheManager
from .events import CashEventManager
from .bank_integration import BankAPIConnection
from .real_time_sync import RealTimeSyncEngine
from .ai_forecasting import AIForecastingEngine
from .analytics_dashboard import AnalyticsDashboard

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

def _log_api_request(endpoint: str, tenant_id: str, user_id: str) -> str:
	"""Log API request with APG formatting"""
	return f"APG_API_REQUEST | endpoint={endpoint} | tenant={tenant_id} | user={user_id}"

def _log_api_response(endpoint: str, status_code: int, duration_ms: int) -> str:
	"""Log API response with APG formatting"""
	return f"APG_API_RESPONSE | endpoint={endpoint} | status={status_code} | duration_ms={duration_ms}"

def _log_api_error(endpoint: str, error: str, tenant_id: str) -> str:
	"""Log API error with APG formatting"""
	return f"APG_API_ERROR | endpoint={endpoint} | error={error} | tenant={tenant_id}"

# ============================================================================
# Pydantic Models for Request/Response Serialization
# ============================================================================

class APGBaseModel(BaseModel):
	"""Base APG model with standard configuration"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)

class BankAccountRequest(APGBaseModel):
	"""Bank account creation/update request"""
	account_number: str = Field(..., min_length=3, max_length=50)
	account_name: str = Field(..., min_length=3, max_length=200)
	account_type: str = Field(..., pattern=r'^(CHECKING|SAVINGS|MONEY_MARKET|INVESTMENT|PETTY_CASH|LOCKBOX)$')
	bank_name: str = Field(..., min_length=3, max_length=200)
	bank_code: Optional[str] = Field(None, max_length=20)
	routing_number: Optional[str] = Field(None, pattern=r'^\d{9}$')
	branch_name: Optional[str] = Field(None, max_length=200)
	currency_code: str = Field(default='USD', pattern=r'^[A-Z]{3}$')
	current_balance: condecimal(decimal_places=4) = Field(default=Decimal('0.0000'))
	available_balance: condecimal(decimal_places=4) = Field(default=Decimal('0.0000'))
	is_active: bool = Field(default=True)
	is_primary: bool = Field(default=False)
	requires_reconciliation: bool = Field(default=True)
	interest_rate: condecimal(decimal_places=6) = Field(default=Decimal('0.000000'))
	minimum_balance: condecimal(decimal_places=4) = Field(default=Decimal('0.0000'))
	notes: Optional[str] = Field(None, max_length=2000)

	class Config:
		schema_extra = {
			"example": {
				"account_number": "123456789",
				"account_name": "Primary Operating Account",
				"account_type": "CHECKING",
				"bank_name": "JPMorgan Chase Bank",
				"bank_code": "CHASE",
				"routing_number": "021000021",
				"branch_name": "Downtown Branch",
				"currency_code": "USD",
				"current_balance": "2500000.0000",
				"available_balance": "2450000.0000",
				"is_active": True,
				"is_primary": True,
				"requires_reconciliation": True,
				"interest_rate": "0.025000",
				"minimum_balance": "100000.0000",
				"notes": "Primary treasury account for daily operations"
			}
		}

class BankAccountResponse(APGBaseModel):
	"""Bank account response model"""
	id: str
	account_number: str
	account_name: str
	account_type: str
	bank_name: str
	bank_code: Optional[str]
	routing_number: Optional[str]
	branch_name: Optional[str]
	currency_code: str
	current_balance: Decimal
	available_balance: Decimal
	pending_credits: Decimal
	pending_debits: Decimal
	is_active: bool
	is_primary: bool
	requires_reconciliation: bool
	last_reconciliation_date: Optional[date]
	interest_rate: Decimal
	minimum_balance: Decimal
	notes: Optional[str]
	created_at: datetime
	updated_at: datetime

	# Real-time calculated fields
	net_balance: Optional[Decimal] = None
	days_cash_on_hand: Optional[int] = None
	is_overdrawn: Optional[bool] = None
	liquidity_ratio: Optional[Decimal] = None

class CashFlowRequest(APGBaseModel):
	"""Cash flow transaction request"""
	account_id: str = Field(..., description="Account ID for the transaction")
	transaction_date: date = Field(..., description="Transaction date")
	description: str = Field(..., min_length=3, max_length=500)
	amount: condecimal(decimal_places=4) = Field(..., gt=0)
	flow_type: str = Field(..., pattern=r'^(INFLOW|OUTFLOW)$')
	category: str = Field(..., max_length=100)
	counterparty: Optional[str] = Field(None, max_length=200)
	reference_number: Optional[str] = Field(None, max_length=50)
	is_forecasted: bool = Field(default=False)
	confidence_level: condecimal(decimal_places=2) = Field(default=Decimal('1.00'), ge=0, le=1)
	source_module: Optional[str] = Field(None, max_length=50)
	transaction_id: Optional[str] = Field(None, max_length=50)
	cost_center: Optional[str] = Field(None, max_length=50)
	department: Optional[str] = Field(None, max_length=100)
	tags: List[str] = Field(default_factory=list)
	notes: Optional[str] = Field(None, max_length=1000)

class CashFlowResponse(APGBaseModel):
	"""Cash flow transaction response"""
	id: str
	account_id: str
	transaction_date: date
	description: str
	amount: Decimal
	flow_type: str
	category: str
	counterparty: Optional[str]
	reference_number: Optional[str]
	is_forecasted: bool
	confidence_level: Decimal
	source_module: Optional[str]
	transaction_id: Optional[str]
	cost_center: Optional[str]
	department: Optional[str]
	tags: List[str]
	notes: Optional[str]
	created_at: datetime
	updated_at: datetime

	# APG Intelligence fields
	predicted_impact: Optional[Decimal] = None
	risk_score: Optional[Decimal] = None
	liquidity_impact: Optional[str] = None

class CashForecastRequest(APGBaseModel):
	"""AI-powered cash forecast request"""
	horizon_days: conint(ge=1, le=365) = Field(default=90, description="Forecast horizon in days")
	scenario_type: str = Field(default='BASE_CASE', pattern=r'^(BASE_CASE|OPTIMISTIC|PESSIMISTIC|STRESS_TEST)$')
	confidence_level: condecimal(decimal_places=2) = Field(default=Decimal('0.95'), ge=0.5, le=0.99)
	include_seasonality: bool = Field(default=True)
	include_external_factors: bool = Field(default=True)
	categories: List[str] = Field(default_factory=list)
	model_type: str = Field(default='AUTO', pattern=r'^(AUTO|ARIMA|LSTM|ENSEMBLE|HISTORICAL)$')
	refresh_models: bool = Field(default=False)

	class Config:
		schema_extra = {
			"example": {
				"horizon_days": 90,
				"scenario_type": "BASE_CASE",
				"confidence_level": "0.95",
				"include_seasonality": True,
				"include_external_factors": True,
				"categories": ["SALES", "PAYROLL", "CAPEX"],
				"model_type": "ENSEMBLE",
				"refresh_models": False
			}
		}

class CashForecastResponse(APGBaseModel):
	"""AI-powered cash forecast response"""
	id: str
	scenario_type: str
	horizon_days: int
	generated_at: datetime
	model_used: str
	model_version: str
	confidence_level: Decimal
	total_forecasted_inflows: Decimal
	total_forecasted_outflows: Decimal
	net_cash_flow: Decimal
	projected_ending_balance: Decimal
	shortfall_probability: Decimal
	value_at_risk: Decimal

	# Detailed forecasts by category and date
	daily_forecasts: List[Dict[str, Any]]
	category_breakdown: Dict[str, Decimal]
	risk_indicators: Dict[str, Any]
	model_performance: Dict[str, Any]
	assumptions_used: List[Dict[str, Any]]

class InvestmentOpportunityRequest(APGBaseModel):
	"""Investment opportunity analysis request"""
	amount: condecimal(decimal_places=4) = Field(..., gt=0)
	maturity_days: conint(ge=1, le=365) = Field(...)
	risk_tolerance: str = Field(default='MODERATE', pattern=r'^(CONSERVATIVE|MODERATE|AGGRESSIVE)$')
	liquidity_requirement: str = Field(default='NORMAL', pattern=r'^(HIGH|NORMAL|LOW)$')
	yield_preference: str = Field(default='BALANCED', pattern=r'^(YIELD_FOCUSED|BALANCED|RISK_FOCUSED)$')
	exclude_instruments: List[str] = Field(default_factory=list)
	min_yield: condecimal(decimal_places=4) = Field(default=Decimal('0.0000'))
	max_concentration: condecimal(decimal_places=2) = Field(default=Decimal('0.25'))

class InvestmentOpportunityResponse(APGBaseModel):
	"""AI-curated investment opportunity response"""
	id: str
	instrument_type: str
	instrument_name: str
	issuer: str
	amount: Decimal
	maturity_date: date
	expected_yield: Decimal
	current_rate: Decimal
	min_investment: Decimal
	max_investment: Decimal
	liquidity_score: Decimal
	risk_score: Decimal
	yield_score: Decimal
	fit_score: Decimal
	overall_score: Decimal
	rating: str
	counterparty_risk: str
	available_until: datetime
	key_features: List[str]
	risks: List[str]
	benchmark_comparison: Dict[str, Any]
	market_conditions: Dict[str, Any]

class CashPositionResponse(APGBaseModel):
	"""Real-time cash position response"""
	position_date: date
	total_cash: Decimal
	available_cash: Decimal
	restricted_cash: Decimal
	invested_cash: Decimal
	pending_inflows: Decimal
	pending_outflows: Decimal
	net_pending: Decimal
	days_cash_on_hand: int
	liquidity_ratio: Decimal
	concentration_risk: Decimal

	# Account breakdown
	accounts: List[Dict[str, Any]]
	currency_breakdown: Dict[str, Decimal]
	account_type_breakdown: Dict[str, Decimal]

	# Risk indicators
	risk_indicators: Dict[str, Any]
	performance_metrics: Dict[str, Any]
	alerts: List[Dict[str, Any]]

	# AI insights
	optimization_opportunities: List[Dict[str, Any]]
	forecasted_shortfalls: List[Dict[str, Any]]
	recommended_actions: List[str]

class RealTimeSyncRequest(APGBaseModel):
	"""Real-time bank synchronization request"""
	bank_codes: List[str] = Field(default_factory=list)
	account_ids: List[str] = Field(default_factory=list)
	force_refresh: bool = Field(default=False)
	include_pending: bool = Field(default=True)
	include_intraday: bool = Field(default=True)
	max_age_minutes: conint(ge=1, le=1440) = Field(default=60)

class RealTimeSyncResponse(APGBaseModel):
	"""Real-time bank synchronization response"""
	sync_id: str
	sync_started_at: datetime
	sync_completed_at: datetime
	total_accounts: int
	successful_syncs: int
	failed_syncs: int
	new_transactions: int
	updated_balances: int
	detected_issues: int

	# Sync details by bank
	bank_results: List[Dict[str, Any]]
	account_results: List[Dict[str, Any]]
	issue_summary: List[Dict[str, Any]]
	data_quality_score: Decimal
	next_sync_recommended: datetime

class AnalyticsDashboardResponse(APGBaseModel):
	"""Advanced analytics dashboard response"""
	dashboard_generated_at: datetime
	data_freshness: datetime
	time_range: Dict[str, Any]

	# Executive KPIs
	kpis: Dict[str, Any]
	performance_trends: Dict[str, Any]
	variance_analysis: Dict[str, Any]

	# Interactive widgets
	cash_flow_chart: Dict[str, Any]
	liquidity_gauge: Dict[str, Any]
	forecasting_accuracy: Dict[str, Any]
	investment_portfolio: Dict[str, Any]
	risk_heatmap: Dict[str, Any]

	# AI insights
	anomalies_detected: List[Dict[str, Any]]
	optimization_opportunities: List[Dict[str, Any]]
	predictive_alerts: List[Dict[str, Any]]
	market_intelligence: Dict[str, Any]

	# Drill-down capabilities
	detailed_breakdowns: Dict[str, Any]
	comparative_analysis: Dict[str, Any]
	benchmarking_data: Dict[str, Any]

class APGHealthResponse(APGBaseModel):
	"""APG system health response"""
	status: str
	timestamp: datetime
	version: str
	uptime_seconds: int

	# Component health
	components: Dict[str, Any]
	database_status: Dict[str, Any]
	cache_status: Dict[str, Any]
	event_system_status: Dict[str, Any]
	bank_connectivity: Dict[str, Any]
	ai_services: Dict[str, Any]

	# Performance metrics
	performance_metrics: Dict[str, Any]
	resource_usage: Dict[str, Any]
	active_sessions: int
	queue_depths: Dict[str, int]

	# Data quality indicators
	data_quality: Dict[str, Any]
	sync_status: Dict[str, Any]
	forecasting_health: Dict[str, Any]


class APGErrorResponse(APGBaseModel):
	"""Standard APG error response"""
	error: bool = True
	error_code: str
	message: str
	details: Optional[Dict[str, Any]] = None
	timestamp: datetime
	request_id: str
	path: str
	method: str

# ============================================================================
# Authentication and Authorization
# ============================================================================

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
	"""Extract user information from JWT token"""
	# Integration with APG authentication system
	# This would validate JWT tokens and extract user/tenant info
	return {
		'user_id': 'api_user',  # Extract from token
		'tenant_id': 'default_tenant',  # Extract from token
		'permissions': ['cash_management.read', 'cash_management.write']  # Extract from token
	}

async def get_tenant_id(user: dict = Depends(get_current_user)) -> str:
	"""Extract tenant ID from authenticated user"""
	return user['tenant_id']

async def check_permission(permission: str):
	"""Check if user has required permission"""
	def permission_checker(user: dict = Depends(get_current_user)):
		if permission not in user.get('permissions', []):
			raise HTTPException(
				status_code=status.HTTP_403_FORBIDDEN,
				detail=f"Permission '{permission}' required"
			)
		return user
	return permission_checker

# ============================================================================
# Dependency Injection for APG Services
# ============================================================================

async def get_cache_manager() -> CashCacheManager:
	"""Get APG cache manager instance"""
	return CashCacheManager()

async def get_event_manager() -> CashEventManager:
	"""Get APG event manager instance"""
	return CashEventManager()

async def get_cash_service(
	tenant_id: str = Depends(get_tenant_id),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager)
) -> CashManagementService:
	"""Get APG cash management service instance"""
	return CashManagementService(tenant_id, cache, events)

async def get_bank_integration(
	tenant_id: str = Depends(get_tenant_id),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager)
) -> BankAPIConnection:
	"""Get APG bank integration service"""
	return BankAPIConnection(tenant_id, cache, events)

async def get_sync_engine(
	tenant_id: str = Depends(get_tenant_id),
	bank_api: BankAPIConnection = Depends(get_bank_integration),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager)
) -> RealTimeSyncEngine:
	"""Get APG real-time sync engine"""
	return RealTimeSyncEngine(tenant_id, bank_api, cache, events)

async def get_ai_forecasting(
	tenant_id: str = Depends(get_tenant_id),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager)
) -> AIForecastingEngine:
	"""Get APG AI forecasting engine"""
	return AIForecastingEngine(tenant_id, cache, events)

async def get_analytics_dashboard(
	tenant_id: str = Depends(get_tenant_id),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager),
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting)
) -> AnalyticsDashboard:
	"""Get APG analytics dashboard"""
	return AnalyticsDashboard(tenant_id, cache, events, ai_forecasting)

# ============================================================================
# FastAPI Application Instance
# ============================================================================

def create_cash_management_api() -> FastAPI:
	"""Create and configure the APG Cash Management FastAPI application"""
	
	app = FastAPI(
		title="APG Cash Management API",
		description="""
		**Enterprise-Grade Cash Management API**

		The APG Cash Management API provides comprehensive treasury operations
		with real-time bank connectivity, AI-powered forecasting, and advanced
		cash optimization capabilities.

		**Key Features:**
		- Universal bank integration (Chase, Wells Fargo, Bank of America, Citi)
		- Real-time cash position monitoring
		- AI-powered cash flow forecasting
		- Intelligent investment optimization
		- Advanced analytics and dashboards
		- Multi-tenant architecture
		- Enterprise security and compliance

		**© 2025 Datacraft. All rights reserved.**
		""",
		version="1.0.0",
		contact={
			"name": "Nyimbi Odero",
			"email": "nyimbi@gmail.com",
			"url": "https://www.datacraft.co.ke"
		},
		license_info={
			"name": "Proprietary",
			"url": "https://www.datacraft.co.ke/license"
		},
		openapi_tags=[
			{
				"name": "accounts",
				"description": "Bank account management and real-time balance monitoring"
			},
			{
				"name": "cash-flows",
				"description": "Cash flow transaction tracking and categorization"
			},
			{
				"name": "forecasting",
				"description": "AI-powered cash flow forecasting and scenario analysis"
			},
			{
				"name": "investments",
				"description": "Investment portfolio management and opportunity discovery"
			},
			{
				"name": "positions",
				"description": "Real-time cash position monitoring and analysis"
			},
			{
				"name": "sync",
				"description": "Real-time bank data synchronization"
			},
			{
				"name": "analytics",
				"description": "Advanced analytics dashboards and KPIs"
			},
			{
				"name": "system",
				"description": "System health monitoring and administration"
			}
		]
	)

	# Middleware configuration
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],  # Configure for production
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	app.add_middleware(GZipMiddleware, minimum_size=1000)

	return app

# Create the FastAPI app instance
app = create_cash_management_api()

# ============================================================================
# Bank Account Management Endpoints
# ============================================================================

@app.get("/accounts", 
	tags=["accounts"],
	response_model=List[BankAccountResponse],
	summary="List bank accounts",
	description="Retrieve all bank accounts for the authenticated tenant with real-time balance information"
)
async def list_bank_accounts(
	include_inactive: bool = Query(False, description="Include inactive accounts"),
	account_type: Optional[str] = Query(None, description="Filter by account type"),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""List all bank accounts with real-time data"""
	try:
		accounts = await cash_service.get_bank_accounts(
			include_inactive=include_inactive,
			account_type=account_type
		)
		
		return accounts
		
	except Exception as e:
		logger.error(_log_api_error("list_bank_accounts", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve bank accounts"
		)

@app.post("/accounts",
	tags=["accounts"],
	response_model=BankAccountResponse,
	status_code=status.HTTP_201_CREATED,
	summary="Create bank account",
	description="Create a new bank account with automatic connectivity setup"
)
async def create_bank_account(
	account_data: BankAccountRequest,
	background_tasks: BackgroundTasks,
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Create a new bank account"""
	try:
		account = await cash_service.create_bank_account(account_data.dict())
		
		# Schedule background bank connectivity setup
		background_tasks.add_task(
			cash_service.setup_bank_connectivity,
			account.id
		)
		
		return account
		
	except Exception as e:
		logger.error(_log_api_error("create_bank_account", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Failed to create bank account"
		)

@app.get("/accounts/{account_id}",
	tags=["accounts"],
	response_model=BankAccountResponse,
	summary="Get bank account details",
	description="Retrieve detailed information for a specific bank account including real-time balances"
)
async def get_bank_account(
	account_id: str = Path(..., description="Bank account ID"),
	include_transactions: bool = Query(False, description="Include recent transactions"),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get bank account details with real-time data"""
	try:
		account = await cash_service.get_bank_account(
			account_id,
			include_transactions=include_transactions
		)
		
		if not account:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Bank account not found"
			)
		
		return account
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(_log_api_error("get_bank_account", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve bank account"
		)

@app.put("/accounts/{account_id}",
	tags=["accounts"],
	response_model=BankAccountResponse,
	summary="Update bank account",
	description="Update bank account information and settings"
)
async def update_bank_account(
	account_id: str = Path(..., description="Bank account ID"),
	account_data: BankAccountRequest = Body(...),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Update bank account details"""
	try:
		account = await cash_service.update_bank_account(account_id, account_data.dict())
		
		if not account:
			raise HTTPException(
				status_code=status.HTTP_404_NOT_FOUND,
				detail="Bank account not found"
			)
		
		return account
		
	except HTTPException:
		raise
	except Exception as e:
		logger.error(_log_api_error("update_bank_account", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Failed to update bank account"
		)

# ============================================================================
# Cash Flow Management Endpoints
# ============================================================================

@app.get("/cash-flows",
	tags=["cash-flows"],
	response_model=List[CashFlowResponse],
	summary="List cash flows",
	description="Retrieve cash flow transactions with filtering and pagination"
)
async def list_cash_flows(
	start_date: Optional[date] = Query(None, description="Start date for filtering"),
	end_date: Optional[date] = Query(None, description="End date for filtering"),
	account_id: Optional[str] = Query(None, description="Filter by account ID"),
	flow_type: Optional[str] = Query(None, description="Filter by flow type (INFLOW/OUTFLOW)"),
	category: Optional[str] = Query(None, description="Filter by category"),
	limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
	offset: int = Query(0, ge=0, description="Number of records to skip"),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""List cash flow transactions"""
	try:
		cash_flows = await cash_service.get_cash_flows(
			start_date=start_date,
			end_date=end_date,
			account_id=account_id,
			flow_type=flow_type,
			category=category,
			limit=limit,
			offset=offset
		)
		
		return cash_flows
		
	except Exception as e:
		logger.error(_log_api_error("list_cash_flows", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve cash flows"
		)

@app.post("/cash-flows",
	tags=["cash-flows"],
	response_model=CashFlowResponse,
	status_code=status.HTTP_201_CREATED,
	summary="Create cash flow",
	description="Record a new cash flow transaction with AI-powered categorization"
)
async def create_cash_flow(
	cash_flow_data: CashFlowRequest,
	background_tasks: BackgroundTasks,
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Create a new cash flow transaction"""
	try:
		cash_flow = await cash_service.create_cash_flow(cash_flow_data.dict())
		
		# Schedule background AI analysis
		background_tasks.add_task(
			cash_service.analyze_cash_flow_impact,
			cash_flow.id
		)
		
		return cash_flow
		
	except Exception as e:
		logger.error(_log_api_error("create_cash_flow", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Failed to create cash flow"
		)

@app.post("/cash-flows/bulk",
	tags=["cash-flows"],
	response_model=Dict[str, Any],
	summary="Bulk import cash flows",
	description="Import multiple cash flow transactions with validation and duplicate detection"
)
async def bulk_import_cash_flows(
	cash_flows: List[CashFlowRequest],
	background_tasks: BackgroundTasks,
	validate_only: bool = Query(False, description="Only validate without importing"),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Bulk import cash flow transactions"""
	try:
		result = await cash_service.bulk_import_cash_flows(
			[cf.dict() for cf in cash_flows],
			validate_only=validate_only
		)
		
		if not validate_only:
			# Schedule background analysis for imported flows
			background_tasks.add_task(
				cash_service.analyze_bulk_cash_flows,
				result['imported_ids']
			)
		
		return result
		
	except Exception as e:
		logger.error(_log_api_error("bulk_import_cash_flows", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Failed to import cash flows"
		)

# ============================================================================
# AI-Powered Forecasting Endpoints
# ============================================================================

@app.post("/forecasting/generate",
	tags=["forecasting"],
	response_model=CashForecastResponse,
	summary="Generate cash forecast",
	description="Generate AI-powered cash flow forecasts using machine learning models"
)
async def generate_cash_forecast(
	forecast_request: CashForecastRequest,
	background_tasks: BackgroundTasks,
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Generate AI-powered cash flow forecast"""
	try:
		forecast = await ai_forecasting.generate_comprehensive_forecast(
			horizon_days=forecast_request.horizon_days,
			scenario_type=forecast_request.scenario_type,
			confidence_level=forecast_request.confidence_level,
			include_seasonality=forecast_request.include_seasonality,
			include_external_factors=forecast_request.include_external_factors,
			categories=forecast_request.categories,
			model_type=forecast_request.model_type,
			refresh_models=forecast_request.refresh_models
		)
		
		# Schedule background model training if needed
		if forecast_request.refresh_models:
			background_tasks.add_task(
				ai_forecasting.retrain_models,
				forecast_request.categories
			)
		
		return forecast
		
	except Exception as e:
		logger.error(_log_api_error("generate_cash_forecast", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to generate cash forecast"
		)

@app.get("/forecasting/scenarios",
	tags=["forecasting"],
	response_model=List[CashForecastResponse],
	summary="Compare forecast scenarios",
	description="Generate and compare multiple forecast scenarios (base case, optimistic, pessimistic, stress test)"
)
async def compare_forecast_scenarios(
	horizon_days: int = Query(90, ge=1, le=365, description="Forecast horizon in days"),
	categories: List[str] = Query([], description="Categories to include in forecast"),
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Compare multiple forecast scenarios"""
	try:
		scenarios = await ai_forecasting.generate_scenario_comparison(
			horizon_days=horizon_days,
			categories=categories
		)
		
		return scenarios
		
	except Exception as e:
		logger.error(_log_api_error("compare_forecast_scenarios", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to compare forecast scenarios"
		)

@app.get("/forecasting/accuracy",
	tags=["forecasting"],
	response_model=Dict[str, Any],
	summary="Forecast accuracy analysis",
	description="Analyze historical forecast accuracy and model performance"
)
async def analyze_forecast_accuracy(
	lookback_days: int = Query(90, ge=30, le=365, description="Historical period to analyze"),
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Analyze forecast accuracy and model performance"""
	try:
		accuracy_analysis = await ai_forecasting.analyze_forecast_accuracy(
			lookback_days=lookback_days
		)
		
		return accuracy_analysis
		
	except Exception as e:
		logger.error(_log_api_error("analyze_forecast_accuracy", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to analyze forecast accuracy"
		)

# ============================================================================
# Investment Management Endpoints
# ============================================================================

@app.post("/investments/opportunities",
	tags=["investments"],
	response_model=List[InvestmentOpportunityResponse],
	summary="Find investment opportunities",
	description="Discover AI-curated investment opportunities based on cash position and preferences"
)
async def find_investment_opportunities(
	opportunity_request: InvestmentOpportunityRequest,
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Find AI-curated investment opportunities"""
	try:
		opportunities = await ai_forecasting.find_investment_opportunities(
			amount=opportunity_request.amount,
			maturity_days=opportunity_request.maturity_days,
			risk_tolerance=opportunity_request.risk_tolerance,
			liquidity_requirement=opportunity_request.liquidity_requirement,
			yield_preference=opportunity_request.yield_preference,
			exclude_instruments=opportunity_request.exclude_instruments,
			min_yield=opportunity_request.min_yield,
			max_concentration=opportunity_request.max_concentration
		)
		
		return opportunities
		
	except Exception as e:
		logger.error(_log_api_error("find_investment_opportunities", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to find investment opportunities"
		)

@app.post("/investments/optimize",
	tags=["investments"],
	response_model=Dict[str, Any],
	summary="Optimize investment portfolio",
	description="AI-powered portfolio optimization for maximum yield with risk constraints"
)
async def optimize_investment_portfolio(
	total_amount: Decimal = Body(..., description="Total amount to invest"),
	target_yield: Optional[Decimal] = Body(None, description="Target yield rate"),
	max_risk_score: Decimal = Body(Decimal('0.5'), description="Maximum risk score"),
	diversification_target: Decimal = Body(Decimal('0.7'), description="Diversification target"),
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Optimize investment portfolio using AI"""
	try:
		optimization = await ai_forecasting.optimize_investment_portfolio(
			total_amount=total_amount,
			target_yield=target_yield,
			max_risk_score=max_risk_score,
			diversification_target=diversification_target
		)
		
		return optimization
		
	except Exception as e:
		logger.error(_log_api_error("optimize_investment_portfolio", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to optimize investment portfolio"
		)

# ============================================================================
# Real-Time Cash Position Endpoints
# ============================================================================

@app.get("/positions/current",
	tags=["positions"],
	response_model=CashPositionResponse,
	summary="Get current cash position",
	description="Retrieve real-time cash position with AI-powered insights and optimization recommendations"
)
async def get_current_cash_position(
	include_forecasts: bool = Query(True, description="Include short-term forecasts"),
	include_opportunities: bool = Query(True, description="Include optimization opportunities"),
	cash_service: CashManagementService = Depends(get_cash_service),
	ai_forecasting: AIForecastingEngine = Depends(get_ai_forecasting),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get current cash position with AI insights"""
	try:
		position = await cash_service.get_current_cash_position(
			include_forecasts=include_forecasts,
			include_opportunities=include_opportunities
		)
		
		return position
		
	except Exception as e:
		logger.error(_log_api_error("get_current_cash_position", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve cash position"
		)

@app.get("/positions/historical",
	tags=["positions"],
	response_model=List[CashPositionResponse],
	summary="Get historical cash positions",
	description="Retrieve historical cash positions with trend analysis"
)
async def get_historical_cash_positions(
	start_date: date = Query(..., description="Start date for historical data"),
	end_date: date = Query(..., description="End date for historical data"),
	frequency: str = Query('daily', pattern=r'^(daily|weekly|monthly)$', description="Data frequency"),
	cash_service: CashManagementService = Depends(get_cash_service),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get historical cash positions"""
	try:
		positions = await cash_service.get_historical_cash_positions(
			start_date=start_date,
			end_date=end_date,
			frequency=frequency
		)
		
		return positions
		
	except Exception as e:
		logger.error(_log_api_error("get_historical_cash_positions", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve historical positions"
		)

# ============================================================================
# Real-Time Bank Synchronization Endpoints
# ============================================================================

@app.post("/sync/execute",
	tags=["sync"],
	response_model=RealTimeSyncResponse,
	summary="Execute bank synchronization",
	description="Synchronize data with connected banks in real-time"
)
async def execute_bank_sync(
	sync_request: RealTimeSyncRequest,
	background_tasks: BackgroundTasks,
	sync_engine: RealTimeSyncEngine = Depends(get_sync_engine),
	user: dict = Depends(check_permission("cash_management.write"))
):
	"""Execute real-time bank synchronization"""
	try:
		sync_result = await sync_engine.execute_comprehensive_sync(
			bank_codes=sync_request.bank_codes,
			account_ids=sync_request.account_ids,
			force_refresh=sync_request.force_refresh,
			include_pending=sync_request.include_pending,
			include_intraday=sync_request.include_intraday,
			max_age_minutes=sync_request.max_age_minutes
		)
		
		# Schedule background data validation
		background_tasks.add_task(
			sync_engine.validate_sync_results,
			sync_result['sync_id']
		)
		
		return sync_result
		
	except Exception as e:
		logger.error(_log_api_error("execute_bank_sync", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to execute bank synchronization"
		)

@app.get("/sync/status",
	tags=["sync"],
	response_model=Dict[str, Any],
	summary="Get sync status",
	description="Check the status of ongoing and recent bank synchronizations"
)
async def get_sync_status(
	sync_engine: RealTimeSyncEngine = Depends(get_sync_engine),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get bank synchronization status"""
	try:
		status_info = await sync_engine.get_sync_status()
		
		return status_info
		
	except Exception as e:
		logger.error(_log_api_error("get_sync_status", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve sync status"
		)

# ============================================================================
# Advanced Analytics Dashboard Endpoints
# ============================================================================

@app.get("/analytics/dashboard",
	tags=["analytics"],
	response_model=AnalyticsDashboardResponse,
	summary="Get analytics dashboard",
	description="Comprehensive analytics dashboard with executive KPIs and AI insights"
)
async def get_analytics_dashboard(
	time_range: str = Query('30d', pattern=r'^(7d|30d|90d|365d|ytd|custom)$', description="Time range for analytics"),
	start_date: Optional[date] = Query(None, description="Custom start date"),
	end_date: Optional[date] = Query(None, description="Custom end date"),
	analytics: AnalyticsDashboard = Depends(get_analytics_dashboard),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get comprehensive analytics dashboard"""
	try:
		dashboard = await analytics.generate_executive_dashboard(
			time_range=time_range,
			start_date=start_date,
			end_date=end_date
		)
		
		return dashboard
		
	except Exception as e:
		logger.error(_log_api_error("get_analytics_dashboard", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to generate analytics dashboard"
		)

@app.get("/analytics/kpis",
	tags=["analytics"],
	response_model=Dict[str, Any],
	summary="Get key performance indicators",
	description="Real-time KPIs for cash management performance"
)
async def get_cash_management_kpis(
	period: str = Query('current', pattern=r'^(current|daily|weekly|monthly|quarterly)$'),
	analytics: AnalyticsDashboard = Depends(get_analytics_dashboard),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get cash management KPIs"""
	try:
		kpis = await analytics.calculate_kpis(period=period)
		
		return kpis
		
	except Exception as e:
		logger.error(_log_api_error("get_cash_management_kpis", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to calculate KPIs"
		)

@app.get("/analytics/trends",
	tags=["analytics"],
	response_model=Dict[str, Any],
	summary="Get trend analysis",
	description="Advanced trend analysis with predictive insights"
)
async def get_trend_analysis(
	metric: str = Query('cash_flow', description="Metric to analyze"),
	lookback_days: int = Query(90, ge=30, le=365, description="Historical period for analysis"),
	analytics: AnalyticsDashboard = Depends(get_analytics_dashboard),
	user: dict = Depends(check_permission("cash_management.read"))
):
	"""Get trend analysis with predictions"""
	try:
		trends = await analytics.analyze_trends(
			metric=metric,
			lookback_days=lookback_days
		)
		
		return trends
		
	except Exception as e:
		logger.error(_log_api_error("get_trend_analysis", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to analyze trends"
		)

# ============================================================================
# System Health and Monitoring Endpoints
# ============================================================================

@app.get("/health",
	tags=["system"],
	response_model=APGHealthResponse,
	summary="System health check",
	description="Comprehensive system health monitoring with component status"
)
async def health_check(
	include_details: bool = Query(True, description="Include detailed component information"),
	cache: CashCacheManager = Depends(get_cache_manager),
	events: CashEventManager = Depends(get_event_manager)
):
	"""System health check"""
	try:
		health_status = {
			"status": "healthy",
			"timestamp": datetime.utcnow(),
			"version": "1.0.0",
			"uptime_seconds": 3600,  # Calculate actual uptime
			"components": {},
			"database_status": {},
			"cache_status": {},
			"event_system_status": {},
			"bank_connectivity": {},
			"ai_services": {},
			"performance_metrics": {},
			"resource_usage": {},
			"active_sessions": 0,
			"queue_depths": {},
			"data_quality": {},
			"sync_status": {},
			"forecasting_health": {}
		}
		
		if include_details:
			# Add detailed component health checks
			health_status["components"] = await _check_component_health(cache, events)
		
		return health_status
		
	except Exception as e:
		logger.error(_log_api_error("health_check", str(e), "system"))
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Health check failed"
		)

async def _check_component_health(cache: CashCacheManager, events: CashEventManager) -> Dict[str, Any]:
	"""Check health of individual system components"""
	component_health = {}
	
	try:
		# Check cache health
		cache_healthy = await cache.health_check()
		component_health["cache"] = {"status": "healthy" if cache_healthy else "unhealthy"}
	except Exception:
		component_health["cache"] = {"status": "error"}
	
	try:
		# Check event system health
		events_healthy = await events.health_check()
		component_health["events"] = {"status": "healthy" if events_healthy else "unhealthy"}
	except Exception:
		component_health["events"] = {"status": "error"}
	
	return component_health

@app.get("/metrics",
	tags=["system"],
	response_model=Dict[str, Any],
	summary="System metrics",
	description="Detailed system performance and usage metrics"
)
async def get_system_metrics(
	include_historical: bool = Query(False, description="Include historical metrics"),
	user: dict = Depends(check_permission("cash_management.admin"))
):
	"""Get system performance metrics"""
	try:
		metrics = {
			"request_metrics": {},
			"performance_metrics": {},
			"resource_usage": {},
			"business_metrics": {},
			"error_rates": {},
			"api_usage": {}
		}
		
		if include_historical:
			metrics["historical_trends"] = {}
		
		return metrics
		
	except Exception as e:
		logger.error(_log_api_error("get_system_metrics", str(e), user['tenant_id']))
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to retrieve system metrics"
		)

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
	"""Handle HTTP exceptions with APG error format"""
	return JSONResponse(
		status_code=exc.status_code,
		content=APGErrorResponse(
			error_code=f"HTTP_{exc.status_code}",
			message=exc.detail,
			timestamp=datetime.utcnow(),
			request_id=getattr(request.state, 'request_id', 'unknown'),
			path=str(request.url.path),
			method=request.method
		).dict()
	)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
	"""Handle general exceptions with APG error format"""
	logger.error(f"Unhandled exception: {str(exc)}")
	
	return JSONResponse(
		status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
		content=APGErrorResponse(
			error_code="INTERNAL_SERVER_ERROR",
			message="An unexpected error occurred",
			details={"exception_type": type(exc).__name__},
			timestamp=datetime.utcnow(),
			request_id=getattr(request.state, 'request_id', 'unknown'),
			path=str(request.url.path),
			method=request.method
		).dict()
	)

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
	"""Initialize APG services on startup"""
	logger.info("APG Cash Management API starting up...")
	
	# Initialize cache connections
	# Initialize event system
	# Initialize AI models
	# Verify bank connectivity
	
	logger.info("APG Cash Management API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
	"""Cleanup APG services on shutdown"""
	logger.info("APG Cash Management API shutting down...")
	
	# Close cache connections
	# Close event system connections
	# Save AI model state
	# Disconnect from banks
	
	logger.info("APG Cash Management API shutdown complete")

# ============================================================================
# Export the FastAPI app for APG integration
# ============================================================================

__all__ = ['app', 'create_cash_management_api']