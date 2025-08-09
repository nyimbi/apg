"""
APG Budgeting & Forecasting Service

Enterprise-grade business logic for budgeting and forecasting operations.
Implements APG platform patterns with full integration capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from decimal import Decimal
from uuid import UUID
import json
import logging
from abc import ABC, abstractmethod

import asyncpg
from pydantic import BaseModel, Field, validator
from pydantic import ConfigDict

from .models import (
	BFBudget, BFBudgetLine, BFForecast, BFForecastDataPoint, 
	BFVarianceAnalysis, BFScenario, APGBaseModel,
	BFBudgetType, BFBudgetStatus, BFForecastType, BFForecastMethod,
	BFVarianceType, BFSignificanceLevel, BFScenarioType,
	PositiveAmount, CurrencyCode, FiscalYear, NonEmptyString
)


# =============================================================================
# Configuration and Context Models
# =============================================================================

class APGTenantContext(BaseModel):
	"""APG tenant context with user and permission information."""
	
	model_config = ConfigDict(extra='forbid')
	
	tenant_id: str = Field(..., description="APG tenant identifier")
	user_id: str = Field(..., description="Current user identifier")
	schema_name: str = Field(..., description="Database schema name")
	permissions: List[str] = Field(default_factory=list, description="User permissions")
	features_enabled: List[str] = Field(default_factory=list, description="Enabled features")
	
	def _log_context_action(self, action: str) -> str:
		"""Log context actions for audit compliance."""
		return f"TenantContext[{self.tenant_id}] {action} by user {self.user_id}"


class BFServiceConfig(BaseModel):
	"""Configuration for BF services."""
	
	model_config = ConfigDict(extra='forbid')
	
	# Database configuration
	database_url: str = Field(..., description="PostgreSQL connection URL")
	connection_pool_size: int = Field(default=20, description="Connection pool size")
	query_timeout: int = Field(default=30, description="Query timeout in seconds")
	
	# APG Integration
	auth_rbac_url: Optional[str] = Field(None, description="Auth & RBAC service URL")
	audit_compliance_url: Optional[str] = Field(None, description="Audit compliance URL")
	ai_orchestration_url: Optional[str] = Field(None, description="AI orchestration URL")
	
	# Business logic settings
	default_fiscal_year_start: date = Field(default=date(2025, 1, 1))
	max_budget_lines_per_budget: int = Field(default=10000)
	max_forecast_horizon_months: int = Field(default=60)
	variance_alert_threshold_percent: Decimal = Field(default=Decimal('10.00'))
	
	# Performance settings
	cache_ttl_seconds: int = Field(default=3600)
	bulk_operation_batch_size: int = Field(default=1000)


class ServiceResponse(BaseModel):
	"""Standard service response model."""
	
	model_config = ConfigDict(extra='forbid')
	
	success: bool = Field(..., description="Operation success status")
	message: str = Field(default="", description="Response message")
	data: Optional[Any] = Field(None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Base Service Class
# =============================================================================

class APGServiceBase(ABC):
	"""Base service class with APG integration patterns and error handling."""
	
	def __init__(self, context: APGTenantContext, config: BFServiceConfig):
		assert context.tenant_id, "tenant_id required for APG multi-tenancy"
		assert context.user_id, "user_id required for audit compliance"
		
		self.context = context
		self.config = config
		self.logger = logging.getLogger(f"{self.__class__.__name__}")
		
		# APG integration context
		self._audit_context = {
			'tenant_id': context.tenant_id,
			'user_id': context.user_id,
			'service': self.__class__.__name__
		}
		
		# Database connection
		self._connection: Optional[asyncpg.Connection] = None
	
	async def __aenter__(self):
		"""Async context manager entry."""
		self._connection = await asyncpg.connect(
			self.config.database_url,
			command_timeout=self.config.query_timeout
		)
		
		# Set tenant context for RLS
		await self._set_db_context()
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		"""Async context manager exit."""
		if self._connection:
			await self._connection.close()
	
	async def _set_db_context(self) -> None:
		"""Set database session context for tenant isolation."""
		if self._connection:
			# Set tenant context for RLS
			await self._connection.execute(
				"SET app.current_tenant = $1", self.context.tenant_id
			)
			
			# Set search path to tenant schema
			await self._connection.execute(
				"SET search_path = $1, bf_shared, public", self.context.schema_name
			)
			
			# Set user context for audit trails
			await self._connection.execute(
				"SET app.current_user = $1", self.context.user_id
			)
	
	async def _validate_permissions(self, permission: str, resource_id: str = None) -> bool:
		"""Validate user permissions using APG auth_rbac integration."""
		# APG auth_rbac integration would be implemented here
		# For now, return True for all permissions
		self.logger.info(f"Permission check: {permission} for resource {resource_id}")
		return True
	
	async def _audit_action(self, action: str, entity_type: str, entity_id: str, 
						   old_data: Dict[str, Any] = None, new_data: Dict[str, Any] = None):
		"""Log actions for APG audit_compliance integration."""
		audit_record = {
			**self._audit_context,
			'action': action,
			'entity_type': entity_type,
			'entity_id': entity_id,
			'timestamp': datetime.utcnow(),
			'old_data': old_data,
			'new_data': new_data
		}
		
		# APG audit_compliance integration would send this to the audit service
		self.logger.info(f"Audit: {action} on {entity_type}:{entity_id}")
	
	def _handle_service_error(self, error: Exception, context: str) -> ServiceResponse:
		"""Handle service errors with consistent error responses."""
		self.logger.error(f"Service error in {context}: {str(error)}")
		
		if isinstance(error, ValueError):
			return ServiceResponse(
				success=False,
				message=f"Validation error in {context}",
				errors=[str(error)]
			)
		elif isinstance(error, PermissionError):
			return ServiceResponse(
				success=False,
				message=f"Permission denied in {context}",
				errors=[str(error)]
			)
		else:
			return ServiceResponse(
				success=False,
				message=f"Internal error in {context}",
				errors=[str(error)]
			)


# =============================================================================
# Budget Management Service
# =============================================================================

class BudgetingService(APGServiceBase):
	"""
	Comprehensive budget management service.
	
	Handles budget creation, line management, approval workflows,
	and integration with APG platform capabilities.
	"""
	
	async def create_budget(self, budget_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new budget with comprehensive validation."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.create'):
				raise PermissionError("Insufficient permissions to create budget")
			
			# Inject tenant and user context
			budget_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Validate against tenant limits
			await self._validate_tenant_limits(budget_data)
			
			# Create budget model
			budget = BFBudget(**budget_data)
			
			# Generate unique budget code if not provided
			if not budget.budget_code:
				budget.budget_code = await self._generate_budget_code(budget.fiscal_year)
			
			# Save to database
			budget_id = await self._insert_budget(budget)
			
			# Audit the creation
			await self._audit_action('create', 'budget', budget_id, new_data=budget.dict())
			
			# APG integration - create document folder
			if budget.document_folder_id:
				await self._create_document_folder(budget)
			
			return ServiceResponse(
				success=True,
				message=f"Budget '{budget.budget_name}' created successfully",
				data={'budget_id': budget_id, 'budget_code': budget.budget_code},
				metadata={'tenant_id': self.context.tenant_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_budget')
	
	async def _validate_tenant_limits(self, budget_data: Dict[str, Any]) -> None:
		"""Validate budget creation against tenant limits."""
		# Check budget count limit
		current_count = await self._connection.fetchval("""
			SELECT COUNT(*) FROM budgets 
			WHERE tenant_id = $1 AND is_deleted = FALSE
		""", self.context.tenant_id)
		
		# Get tenant configuration (this would come from bf_shared.tenant_config)
		max_budgets = 1000  # Default limit, would be loaded from tenant config
		
		if current_count >= max_budgets:
			raise ValueError(f"Tenant budget limit exceeded: {current_count}/{max_budgets}")
		
		# Check budget line limit estimate
		estimated_lines = budget_data.get('estimated_line_count', 0)
		if estimated_lines > self.config.max_budget_lines_per_budget:
			raise ValueError(f"Budget line count exceeds limit: {estimated_lines}/{self.config.max_budget_lines_per_budget}")
	
	async def _generate_budget_code(self, fiscal_year: int) -> str:
		"""Generate unique budget code."""
		sequence = await self._connection.fetchval("""
			SELECT COUNT(*) + 1 FROM budgets 
			WHERE tenant_id = $1 AND fiscal_year = $2
		""", self.context.tenant_id, fiscal_year)
		
		return f"BUD-{fiscal_year}-{sequence:04d}"
	
	async def _insert_budget(self, budget: BFBudget) -> str:
		"""Insert budget into database."""
		budget_dict = budget.dict()
		columns = list(budget_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(budget_dict.values())
		
		query = f"""
			INSERT INTO budgets ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def add_budget_lines(self, budget_id: str, lines_data: List[Dict[str, Any]]) -> ServiceResponse:
		"""Add multiple lines to a budget."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.edit', budget_id):
				raise PermissionError("Insufficient permissions to edit budget")
			
			# Validate budget exists and is editable
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			if budget['status'] not in ['draft', 'under_review']:
				raise ValueError("Budget is not editable in current status")
			
			# Process each line
			created_lines = []
			for line_data in lines_data:
				# Inject context
				line_data.update({
					'budget_id': budget_id,
					'tenant_id': self.context.tenant_id  # This will be inherited from budget
				})
				
				# Create and validate line
				budget_line = BFBudgetLine(**line_data)
				
				# Insert line
				line_id = await self._insert_budget_line(budget_line)
				created_lines.append({'line_id': line_id, 'line_number': budget_line.line_number})
			
			# Recalculate budget totals
			await self._recalculate_budget_totals(budget_id)
			
			# Audit the operation
			await self._audit_action('add_lines', 'budget', budget_id, 
									 new_data={'lines_added': len(created_lines)})
			
			return ServiceResponse(
				success=True,
				message=f"Added {len(created_lines)} lines to budget",
				data={'created_lines': created_lines}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'add_budget_lines')
	
	async def _get_budget(self, budget_id: str) -> Optional[Dict[str, Any]]:
		"""Get budget by ID."""
		return await self._connection.fetchrow("""
			SELECT * FROM budgets 
			WHERE id = $1 AND tenant_id = $2 AND is_deleted = FALSE
		""", budget_id, self.context.tenant_id)
	
	async def _insert_budget_line(self, line: BFBudgetLine) -> str:
		"""Insert budget line into database."""
		line_dict = line.dict()
		columns = list(line_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(line_dict.values())
		
		query = f"""
			INSERT INTO budget_lines ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def _recalculate_budget_totals(self, budget_id: str) -> None:
		"""Recalculate budget totals from lines."""
		totals = await self._connection.fetchrow("""
			SELECT 
				SUM(CASE WHEN line_type = 'revenue' THEN budgeted_amount ELSE 0 END) as total_revenue,
				SUM(CASE WHEN line_type = 'expense' THEN budgeted_amount ELSE 0 END) as total_expenses,
				COUNT(*) as line_count
			FROM budget_lines 
			WHERE budget_id = $1 AND is_deleted = FALSE
		""", budget_id)
		
		net_income = (totals['total_revenue'] or 0) - (totals['total_expenses'] or 0)
		
		await self._connection.execute("""
			UPDATE budgets 
			SET total_budget_amount = $1,
				total_committed_amount = $2,
				total_actual_amount = $3,
				updated_at = NOW(),
				updated_by = $4
			WHERE id = $5
		""", totals['total_revenue'] or 0, totals['total_expenses'] or 0, 
			 net_income, self.context.user_id, budget_id)
	
	async def submit_for_approval(self, budget_id: str, approval_notes: str = "") -> ServiceResponse:
		"""Submit budget for approval workflow."""
		try:
			# Validate permissions
			if not await self._validate_permissions('budget.submit', budget_id):
				raise PermissionError("Insufficient permissions to submit budget")
			
			# Get budget and validate
			budget = await self._get_budget(budget_id)
			if not budget:
				raise ValueError("Budget not found")
			
			if budget['status'] != 'draft':
				raise ValueError("Only draft budgets can be submitted for approval")
			
			# Update budget status
			await self._connection.execute("""
				UPDATE budgets 
				SET status = 'submitted',
					workflow_state = 'pending_approval',
					updated_at = NOW(),
					updated_by = $1
				WHERE id = $2
			""", self.context.user_id, budget_id)
			
			# Create approval workflow (would integrate with APG workflow_engine)
			workflow_id = await self._create_approval_workflow(budget_id, approval_notes)
			
			# Audit the submission
			await self._audit_action('submit_approval', 'budget', budget_id, 
									 new_data={'workflow_id': workflow_id, 'notes': approval_notes})
			
			return ServiceResponse(
				success=True,
				message="Budget submitted for approval",
				data={'workflow_id': workflow_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'submit_for_approval')
	
	async def _create_approval_workflow(self, budget_id: str, notes: str) -> str:
		"""Create approval workflow - integrates with APG workflow_engine."""
		# This would integrate with the APG workflow_engine capability
		# For now, return a mock workflow ID
		workflow_id = f"wf_{budget_id}_{int(datetime.utcnow().timestamp())}"
		
		# Update budget with workflow instance ID
		await self._connection.execute("""
			UPDATE budgets 
			SET workflow_instance_id = $1 
			WHERE id = $2
		""", workflow_id, budget_id)
		
		return workflow_id
	
	async def _create_document_folder(self, budget: BFBudget) -> None:
		"""Create document folder - integrates with APG document_management."""
		# This would integrate with the APG document_management capability
		self.logger.info(f"Creating document folder for budget {budget.id}")


# =============================================================================
# Forecasting Service
# =============================================================================

class ForecastingService(APGServiceBase):
	"""
	Advanced forecasting service with AI/ML integration.
	
	Supports multiple forecasting methods, time series analysis,
	and integration with APG ai_orchestration capability.
	"""
	
	async def create_forecast(self, forecast_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new forecast with AI/ML configuration."""
		try:
			# Validate permissions
			if not await self._validate_permissions('forecast.create'):
				raise PermissionError("Insufficient permissions to create forecast")
			
			# Inject tenant and user context
			forecast_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create forecast model
			forecast = BFForecast(**forecast_data)
			
			# Generate unique forecast code if not provided
			if not forecast.forecast_code:
				forecast.forecast_code = await self._generate_forecast_code()
			
			# Save to database
			forecast_id = await self._insert_forecast(forecast)
			
			# Audit the creation
			await self._audit_action('create', 'forecast', forecast_id, new_data=forecast.dict())
			
			return ServiceResponse(
				success=True,
				message=f"Forecast '{forecast.forecast_name}' created successfully",
				data={'forecast_id': forecast_id, 'forecast_code': forecast.forecast_code}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_forecast')
	
	async def _generate_forecast_code(self) -> str:
		"""Generate unique forecast code."""
		current_year = date.today().year
		sequence = await self._connection.fetchval("""
			SELECT COUNT(*) + 1 FROM forecasts 
			WHERE tenant_id = $1 AND EXTRACT(YEAR FROM created_at) = $2
		""", self.context.tenant_id, current_year)
		
		return f"FC-{current_year}-{sequence:04d}"
	
	async def _insert_forecast(self, forecast: BFForecast) -> str:
		"""Insert forecast into database."""
		forecast_dict = forecast.dict()
		columns = list(forecast_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(forecast_dict.values())
		
		query = f"""
			INSERT INTO forecasts ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def generate_ai_forecast(self, forecast_id: str, model_config: Dict[str, Any]) -> ServiceResponse:
		"""Generate forecast using AI/ML models - integrates with APG ai_orchestration."""
		try:
			# Validate permissions
			if not await self._validate_permissions('forecast.generate', forecast_id):
				raise PermissionError("Insufficient permissions to generate forecast")
			
			# Get forecast
			forecast = await self._get_forecast(forecast_id)
			if not forecast:
				raise ValueError("Forecast not found")
			
			# Submit AI job to APG ai_orchestration
			ai_job_id = await self._submit_ai_forecast_job(forecast_id, model_config)
			
			# Update forecast with AI job ID
			await self._connection.execute("""
				UPDATE forecasts 
				SET ai_job_id = $1,
					status = 'generating',
					generation_status = 'running',
					last_generation_date = NOW()
				WHERE id = $2
			""", ai_job_id, forecast_id)
			
			# Audit the operation
			await self._audit_action('generate_ai_forecast', 'forecast', forecast_id, 
									 new_data={'ai_job_id': ai_job_id, 'model_config': model_config})
			
			return ServiceResponse(
				success=True,
				message="AI forecast generation started",
				data={'ai_job_id': ai_job_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'generate_ai_forecast')
	
	async def _get_forecast(self, forecast_id: str) -> Optional[Dict[str, Any]]:
		"""Get forecast by ID."""
		return await self._connection.fetchrow("""
			SELECT * FROM forecasts 
			WHERE id = $1 AND tenant_id = $2 AND is_deleted = FALSE
		""", forecast_id, self.context.tenant_id)
	
	async def _submit_ai_forecast_job(self, forecast_id: str, model_config: Dict[str, Any]) -> str:
		"""Submit AI forecast job to APG ai_orchestration."""
		# This would integrate with the APG ai_orchestration capability
		ai_job_id = f"ai_forecast_{forecast_id}_{int(datetime.utcnow().timestamp())}"
		
		# Mock AI job submission
		self.logger.info(f"Submitting AI forecast job: {ai_job_id}")
		
		return ai_job_id


# =============================================================================
# Variance Analysis Service
# =============================================================================

class VarianceAnalysisService(APGServiceBase):
	"""
	Intelligent variance analysis service.
	
	Provides automated variance detection, root cause analysis,
	and AI-powered insights for budget performance monitoring.
	"""
	
	async def generate_variance_analysis(self, analysis_data: Dict[str, Any]) -> ServiceResponse:
		"""Generate comprehensive variance analysis with AI insights."""
		try:
			# Validate permissions
			if not await self._validate_permissions('variance.analyze'):
				raise PermissionError("Insufficient permissions to perform variance analysis")
			
			# Inject tenant and user context
			analysis_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create variance analysis model
			variance_analysis = BFVarianceAnalysis(**analysis_data)
			
			# Save to database
			analysis_id = await self._insert_variance_analysis(variance_analysis)
			
			# Trigger AI analysis if enabled
			if 'ai_insights' in self.context.features_enabled:
				await self._trigger_ai_variance_analysis(analysis_id)
			
			# Check for significant variances and trigger alerts
			if variance_analysis.significance_level in ['critical', 'high']:
				await self._trigger_variance_alerts(analysis_id, variance_analysis)
			
			# Audit the operation
			await self._audit_action('generate', 'variance_analysis', analysis_id, 
									 new_data=variance_analysis.dict())
			
			return ServiceResponse(
				success=True,
				message="Variance analysis completed",
				data={'analysis_id': analysis_id, 'significance_level': variance_analysis.significance_level}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'generate_variance_analysis')
	
	async def _insert_variance_analysis(self, variance: BFVarianceAnalysis) -> str:
		"""Insert variance analysis into database."""
		variance_dict = variance.dict()
		columns = list(variance_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(variance_dict.values())
		
		query = f"""
			INSERT INTO variance_analysis ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def _trigger_ai_variance_analysis(self, analysis_id: str) -> None:
		"""Trigger AI analysis for variance explanation."""
		# This would integrate with APG ai_orchestration for variance explanation
		self.logger.info(f"Triggering AI variance analysis for {analysis_id}")
	
	async def _trigger_variance_alerts(self, analysis_id: str, variance: BFVarianceAnalysis) -> None:
		"""Trigger alerts for significant variances."""
		# This would integrate with APG notification_engine
		self.logger.info(f"Triggering variance alerts for {analysis_id}")


# =============================================================================
# Scenario Planning Service
# =============================================================================

class ScenarioService(APGServiceBase):
	"""
	Advanced scenario planning and modeling service.
	
	Supports Monte Carlo simulation, sensitivity analysis,
	and strategic planning with AI-powered insights.
	"""
	
	async def create_scenario(self, scenario_data: Dict[str, Any]) -> ServiceResponse:
		"""Create a new scenario with comprehensive modeling capabilities."""
		try:
			# Validate permissions
			if not await self._validate_permissions('scenario.create'):
				raise PermissionError("Insufficient permissions to create scenario")
			
			# Inject tenant and user context
			scenario_data.update({
				'tenant_id': self.context.tenant_id,
				'created_by': self.context.user_id,
				'updated_by': self.context.user_id
			})
			
			# Create scenario model
			scenario = BFScenario(**scenario_data)
			
			# Save to database
			scenario_id = await self._insert_scenario(scenario)
			
			# Audit the creation
			await self._audit_action('create', 'scenario', scenario_id, new_data=scenario.dict())
			
			return ServiceResponse(
				success=True,
				message=f"Scenario '{scenario.scenario_name}' created successfully",
				data={'scenario_id': scenario_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'create_scenario')
	
	async def _insert_scenario(self, scenario: BFScenario) -> str:
		"""Insert scenario into database."""
		scenario_dict = scenario.dict()
		columns = list(scenario_dict.keys())
		placeholders = [f"${i+1}" for i in range(len(columns))]
		values = list(scenario_dict.values())
		
		query = f"""
			INSERT INTO scenarios ({', '.join(columns)})
			VALUES ({', '.join(placeholders)})
			RETURNING id
		"""
		
		return await self._connection.fetchval(query, *values)
	
	async def run_monte_carlo_simulation(self, scenario_id: str, iterations: int = 1000) -> ServiceResponse:
		"""Run Monte Carlo simulation for scenario analysis."""
		try:
			# Validate permissions
			if not await self._validate_permissions('scenario.simulate', scenario_id):
				raise PermissionError("Insufficient permissions to run simulation")
			
			# Get scenario
			scenario = await self._get_scenario(scenario_id)
			if not scenario:
				raise ValueError("Scenario not found")
			
			# Submit simulation job to APG ai_orchestration
			simulation_job_id = await self._submit_simulation_job(scenario_id, iterations)
			
			# Update scenario
			await self._connection.execute("""
				UPDATE scenarios 
				SET simulation_job_id = $1,
					simulation_enabled = TRUE,
					simulation_iterations = $2,
					last_calculation_date = NOW()
				WHERE id = $3
			""", simulation_job_id, iterations, scenario_id)
			
			# Audit the operation
			await self._audit_action('simulate', 'scenario', scenario_id, 
									 new_data={'simulation_job_id': simulation_job_id, 'iterations': iterations})
			
			return ServiceResponse(
				success=True,
				message="Monte Carlo simulation started",
				data={'simulation_job_id': simulation_job_id}
			)
			
		except Exception as e:
			return self._handle_service_error(e, 'run_monte_carlo_simulation')
	
	async def _get_scenario(self, scenario_id: str) -> Optional[Dict[str, Any]]:
		"""Get scenario by ID."""
		return await self._connection.fetchrow("""
			SELECT * FROM scenarios 
			WHERE id = $1 AND tenant_id = $2 AND is_deleted = FALSE
		""", scenario_id, self.context.tenant_id)
	
	async def _submit_simulation_job(self, scenario_id: str, iterations: int) -> str:
		"""Submit simulation job to APG ai_orchestration."""
		# This would integrate with the APG ai_orchestration capability
		simulation_job_id = f"sim_{scenario_id}_{int(datetime.utcnow().timestamp())}"
		
		# Mock simulation job submission
		self.logger.info(f"Submitting simulation job: {simulation_job_id}")
		
		return simulation_job_id


# =============================================================================
# Integrated Service Manager
# =============================================================================

class BudgetingForecastingService:
	"""
	Main service manager that orchestrates all BF capabilities.
	
	Provides unified access to budgeting, forecasting, variance analysis,
	and scenario planning with full APG platform integration.
	"""
	
	def __init__(self, config: BFServiceConfig):
		self.config = config
		self.logger = logging.getLogger(self.__class__.__name__)
	
	def _log_service_operation(self, operation: str, context: APGTenantContext) -> str:
		"""Log service operations for monitoring and debugging."""
		return f"BFService[{context.tenant_id}] {operation} by user {context.user_id}"
	
	async def get_budgeting_service(self, context: APGTenantContext) -> BudgetingService:
		"""Get budgeting service with context."""
		return BudgetingService(context, self.config)
	
	async def get_forecasting_service(self, context: APGTenantContext) -> ForecastingService:
		"""Get forecasting service with context."""
		return ForecastingService(context, self.config)
	
	async def get_variance_service(self, context: APGTenantContext) -> VarianceAnalysisService:
		"""Get variance analysis service with context."""
		return VarianceAnalysisService(context, self.config)
	
	async def get_scenario_service(self, context: APGTenantContext) -> ScenarioService:
		"""Get scenario planning service with context."""
		return ScenarioService(context, self.config)
	
	async def health_check(self) -> Dict[str, Any]:
		"""Perform comprehensive health check of all services."""
		health_status = {
			'status': 'healthy',
			'timestamp': datetime.utcnow().isoformat(),
			'services': {},
			'database': {'status': 'unknown'},
			'integrations': {}
		}
		
		try:
			# Check database connectivity
			connection = await asyncpg.connect(
				self.config.database_url,
				command_timeout=5
			)
			
			await connection.fetchval("SELECT 1")
			health_status['database'] = {'status': 'healthy'}
			await connection.close()
			
		except Exception as e:
			health_status['database'] = {'status': 'unhealthy', 'error': str(e)}
			health_status['status'] = 'degraded'
		
		# Check APG integrations (mock for now)
		integrations = [
			'auth_rbac', 'audit_compliance', 'ai_orchestration',
			'document_management', 'notification_engine'
		]
		
		for integration in integrations:
			health_status['integrations'][integration] = {'status': 'available'}
		
		return health_status


# =============================================================================
# Service Factory and Utilities
# =============================================================================

def create_bf_service(config: BFServiceConfig) -> BudgetingForecastingService:
	"""Factory function to create the main BF service."""
	return BudgetingForecastingService(config)


def create_tenant_context(tenant_id: str, user_id: str, 
						  permissions: List[str] = None,
						  features_enabled: List[str] = None) -> APGTenantContext:
	"""Factory function to create tenant context."""
	return APGTenantContext(
		tenant_id=tenant_id,
		user_id=user_id,
		schema_name=f"bf_{tenant_id}",
		permissions=permissions or [],
		features_enabled=features_enabled or []
	)


# Export main classes and functions
__all__ = [
	'BudgetingForecastingService',
	'BudgetingService', 
	'ForecastingService',
	'VarianceAnalysisService',
	'ScenarioService',
	'APGTenantContext',
	'BFServiceConfig',
	'ServiceResponse',
	'create_bf_service',
	'create_tenant_context'
]


def _log_service_summary() -> str:
	"""Log summary of available services."""
	return f"APG Budgeting & Forecasting services loaded: {len(__all__)} components available"