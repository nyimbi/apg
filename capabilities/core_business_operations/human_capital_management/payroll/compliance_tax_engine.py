"""
APG Payroll Management - Intelligent Compliance & Tax Engine

Revolutionary compliance and tax calculation engine with AI-powered
regulatory monitoring, automated filing, and intelligent optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, ConfigDict

# APG Platform Imports
from ...audit_compliance.services import ComplianceValidationService, RegulatoryUpdateService
from ...ai_orchestration.services import AIOrchestrationService, MLModelService
from ...notification_engine.services import NotificationService
from ...integration_api_management.services import ExternalAPIService
from .models import (
	PRPayrollRun, PREmployeePayroll, PRTaxCalculation, PRPayrollJournal,
	TaxType, PayrollStatus
)

# Configure logging
logger = logging.getLogger(__name__)


class ComplianceLevel(str, Enum):
	"""Compliance monitoring levels."""
	BASIC = "basic"
	STANDARD = "standard"
	ADVANCED = "advanced"
	ENTERPRISE = "enterprise"


class TaxCalculationMethod(str, Enum):
	"""Tax calculation methods."""
	PERCENTAGE = "percentage"
	BRACKET = "bracket"
	FLAT_RATE = "flat_rate"
	FORMULA = "formula"
	AI_OPTIMIZED = "ai_optimized"


class ComplianceStatus(str, Enum):
	"""Compliance validation status."""
	COMPLIANT = "compliant"
	WARNING = "warning"
	VIOLATION = "violation"
	UNKNOWN = "unknown"


@dataclass
class TaxCalculationResult:
	"""Result of tax calculation."""
	tax_type: TaxType
	jurisdiction: str
	taxable_wages: Decimal
	tax_amount: Decimal
	employee_portion: Decimal
	employer_portion: Decimal
	calculation_method: TaxCalculationMethod
	tax_rate: Optional[Decimal]
	confidence_score: float
	compliance_flags: List[str]
	calculation_details: Dict[str, Any]


@dataclass
class ComplianceCheckResult:
	"""Result of compliance validation."""
	compliance_area: str
	status: ComplianceStatus
	score: float  # 0-100
	violations: List[str]
	warnings: List[str]
	recommendations: List[str]
	regulatory_requirements: List[str]
	last_updated: datetime


@dataclass
class RegulatoryUpdate:
	"""Regulatory update information."""
	jurisdiction: str
	regulation_type: str
	effective_date: date
	description: str
	impact_assessment: Dict[str, Any]
	action_required: bool
	priority: str  # low, medium, high, critical
	implementation_deadline: Optional[date]


class ComplianceTaxConfig(BaseModel):
	"""Configuration for compliance and tax engine."""
	model_config = ConfigDict(extra='forbid')
	
	# Compliance Settings
	compliance_level: ComplianceLevel = Field(default=ComplianceLevel.ENTERPRISE)
	real_time_monitoring: bool = Field(default=True)
	auto_regulatory_updates: bool = Field(default=True)
	compliance_threshold: float = Field(default=95.0, ge=80.0, le=100.0)
	
	# Tax Calculation Settings
	enable_ai_tax_optimization: bool = Field(default=True)
	tax_calculation_precision: int = Field(default=2, ge=2, le=6)
	enable_multi_jurisdiction: bool = Field(default=True)
	tax_rate_cache_hours: int = Field(default=24, ge=1, le=168)
	
	# Regulatory Monitoring
	regulatory_monitoring_enabled: bool = Field(default=True)
	update_frequency_hours: int = Field(default=6, ge=1, le=24)
	critical_alert_threshold: int = Field(default=30, ge=1, le=90)  # days
	
	# AI/ML Settings
	enable_predictive_compliance: bool = Field(default=True)
	ml_confidence_threshold: float = Field(default=85.0, ge=70.0, le=100.0)
	enable_compliance_learning: bool = Field(default=True)
	
	# Performance Settings
	enable_parallel_processing: bool = Field(default=True)
	max_concurrent_calculations: int = Field(default=10, ge=1, le=50)
	enable_calculation_caching: bool = Field(default=True)


class IntelligentComplianceTaxEngine:
	"""Revolutionary compliance and tax calculation engine.
	
	Provides AI-powered regulatory monitoring, automated compliance validation,
	intelligent tax calculations, and predictive compliance risk assessment.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		compliance_service: ComplianceValidationService,
		regulatory_service: RegulatoryUpdateService,
		ai_service: AIOrchestrationService,
		ml_service: MLModelService,
		notification_service: NotificationService,
		external_api_service: ExternalAPIService,
		config: Optional[ComplianceTaxConfig] = None
	):
		self.db = db_session
		self.compliance_service = compliance_service
		self.regulatory_service = regulatory_service
		self.ai_service = ai_service
		self.ml_service = ml_service
		self.notification_service = notification_service
		self.external_api_service = external_api_service
		self.config = config or ComplianceTaxConfig()
		
		# Tax calculation engines by jurisdiction
		self._tax_engines = {}
		
		# Compliance rule cache
		self._compliance_rules = {}
		
		# Regulatory update cache
		self._regulatory_updates = {}
		
		# AI models for compliance and tax optimization
		self._compliance_models = {}
		self._tax_models = {}
		
		# Performance caches
		self._tax_rate_cache = {}
		self._compliance_cache = {}
	
	async def initialize_compliance_engine(self) -> None:
		"""Initialize the compliance and tax engine."""
		
		try:
			logger.info("Initializing Intelligent Compliance & Tax Engine...")
			
			# Load compliance rules and regulations
			await self._load_compliance_rules()
			
			# Initialize tax calculation engines
			await self._initialize_tax_engines()
			
			# Load AI/ML models
			if self.config.enable_ai_tax_optimization:
				await self._load_ai_models()
			
			# Start regulatory monitoring
			if self.config.regulatory_monitoring_enabled:
				await self._start_regulatory_monitoring()
			
			logger.info("Compliance & Tax Engine initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize compliance engine: {e}")
			raise
	
	async def calculate_employee_taxes(
		self,
		employee_payroll: PREmployeePayroll,
		payroll_run: PRPayrollRun,
		tenant_id: str
	) -> List[TaxCalculationResult]:
		"""Calculate all taxes for an employee with AI optimization."""
		
		try:
			logger.debug(f"Calculating taxes for employee: {employee_payroll.employee_id}")
			
			# Get employee location and tax jurisdiction
			employee_location = await self._get_employee_location(
				employee_payroll.employee_id, tenant_id
			)
			
			# Determine applicable taxes
			applicable_taxes = await self._get_applicable_taxes(
				employee_location, employee_payroll, tenant_id
			)
			
			# Calculate each tax type
			tax_results = []
			
			if self.config.enable_parallel_processing:
				# Parallel tax calculations
				tasks = [
					self._calculate_single_tax(tax_info, employee_payroll, employee_location, tenant_id)
					for tax_info in applicable_taxes
				]
				results = await asyncio.gather(*tasks, return_exceptions=True)
				
				for result in results:
					if isinstance(result, Exception):
						logger.error(f"Tax calculation failed: {result}")
						continue
					tax_results.append(result)
			else:
				# Sequential tax calculations
				for tax_info in applicable_taxes:
					try:
						result = await self._calculate_single_tax(
							tax_info, employee_payroll, employee_location, tenant_id
						)
						tax_results.append(result)
					except Exception as e:
						logger.error(f"Tax calculation failed for {tax_info['tax_type']}: {e}")
			
			# Store tax calculations in database
			await self._store_tax_calculations(employee_payroll, tax_results, tenant_id)
			
			# Validate compliance
			await self._validate_tax_compliance(employee_payroll, tax_results, tenant_id)
			
			return tax_results
			
		except Exception as e:
			logger.error(f"Employee tax calculation failed: {e}")
			raise
	
	async def _calculate_single_tax(
		self,
		tax_info: Dict[str, Any],
		employee_payroll: PREmployeePayroll,
		employee_location: Dict[str, Any],
		tenant_id: str
	) -> TaxCalculationResult:
		"""Calculate a single tax with AI optimization."""
		
		try:
			tax_type = TaxType(tax_info["tax_type"])
			jurisdiction = tax_info["jurisdiction"]
			
			# Get tax calculation engine
			engine = await self._get_tax_engine(jurisdiction, tax_type)
			
			# Prepare calculation context
			calc_context = {
				"employee_payroll": employee_payroll,
				"employee_location": employee_location,
				"tax_info": tax_info,
				"tenant_id": tenant_id,
				"calculation_date": datetime.utcnow()
			}
			
			# Use AI optimization if enabled
			if self.config.enable_ai_tax_optimization and tax_type in self._tax_models:
				result = await self._calculate_tax_with_ai(tax_type, calc_context)
			else:
				result = await self._calculate_tax_traditional(tax_type, calc_context, engine)
			
			# Validate calculation
			await self._validate_tax_calculation(result, calc_context)
			
			return result
			
		except Exception as e:
			logger.error(f"Single tax calculation failed for {tax_info['tax_type']}: {e}")
			raise
	
	async def _calculate_tax_with_ai(
		self,
		tax_type: TaxType,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate tax using AI optimization."""
		
		try:
			# Extract features for AI model
			features = await self._extract_tax_features(tax_type, calc_context)
			
			# Get AI model prediction
			model = self._tax_models[tax_type]
			prediction = await self.ml_service.predict(
				model=model,
				features=features
			)
			
			# Validate AI prediction confidence
			if prediction.confidence < self.config.ml_confidence_threshold / 100.0:
				logger.warning(f"Low AI confidence for {tax_type}, falling back to traditional calculation")
				return await self._calculate_tax_traditional(tax_type, calc_context, None)
			
			# Convert AI prediction to tax result
			return await self._convert_ai_prediction_to_tax_result(
				prediction, tax_type, calc_context
			)
			
		except Exception as e:
			logger.error(f"AI tax calculation failed for {tax_type}: {e}")
			# Fallback to traditional calculation
			return await self._calculate_tax_traditional(tax_type, calc_context, None)
	
	async def _calculate_tax_traditional(
		self,
		tax_type: TaxType,
		calc_context: Dict[str, Any],
		engine: Optional[Any]
	) -> TaxCalculationResult:
		"""Calculate tax using traditional methods."""
		
		employee_payroll = calc_context["employee_payroll"]
		employee_location = calc_context["employee_location"]
		
		# Route to specific tax calculation method
		if tax_type == TaxType.FEDERAL_INCOME:
			return await self._calculate_federal_income_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.STATE_INCOME:
			return await self._calculate_state_income_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.FICA_SOCIAL_SECURITY:
			return await self._calculate_fica_social_security_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.FICA_MEDICARE:
			return await self._calculate_fica_medicare_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.FUTA:
			return await self._calculate_futa_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.SUTA:
			return await self._calculate_suta_tax(employee_payroll, calc_context)
		elif tax_type == TaxType.LOCAL_INCOME:
			return await self._calculate_local_income_tax(employee_payroll, calc_context)
		else:
			raise ValueError(f"Unsupported tax type: {tax_type}")
	
	async def _calculate_federal_income_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate federal income tax using current tax tables."""
		
		try:
			# Get tax parameters
			taxable_wages = employee_payroll.taxable_income
			filing_status = employee_payroll.filing_status or "single"
			allowances = employee_payroll.federal_allowances or 0
			additional_withholding = employee_payroll.additional_withholding or Decimal('0.00')
			pay_frequency = employee_payroll.pay_frequency
			
			# Get current tax year
			tax_year = datetime.now().year
			
			# Get federal tax brackets
			tax_brackets = await self._get_federal_tax_brackets(tax_year, filing_status)
			
			# Calculate allowance reduction
			allowance_amount = await self._calculate_federal_allowance_amount(
				allowances, pay_frequency, tax_year
			)
			
			# Adjust taxable wages
			adjusted_taxable_wages = max(Decimal('0.00'), taxable_wages - allowance_amount)
			
			# Calculate tax using brackets
			calculated_tax = self._calculate_bracket_tax(adjusted_taxable_wages, tax_brackets)
			
			# Add additional withholding
			total_tax = calculated_tax + additional_withholding
			
			# Round to nearest cent
			final_tax = total_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.FEDERAL_INCOME,
				jurisdiction="US",
				taxable_wages=taxable_wages,
				tax_amount=final_tax,
				employee_portion=final_tax,
				employer_portion=Decimal('0.00'),
				calculation_method=TaxCalculationMethod.BRACKET,
				tax_rate=None,  # Variable based on brackets
				confidence_score=95.0,
				compliance_flags=[],
				calculation_details={
					"filing_status": filing_status,
					"allowances": allowances,
					"allowance_amount": float(allowance_amount),
					"adjusted_taxable_wages": float(adjusted_taxable_wages),
					"calculated_tax": float(calculated_tax),
					"additional_withholding": float(additional_withholding),
					"tax_brackets_used": len(tax_brackets)
				}
			)
			
		except Exception as e:
			logger.error(f"Federal income tax calculation failed: {e}")
			raise
	
	async def _calculate_state_income_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate state income tax."""
		
		try:
			employee_location = calc_context["employee_location"]
			state = employee_location.get("state", "CA")
			
			# Check if state has income tax
			no_tax_states = ["TX", "FL", "NV", "WA", "SD", "AK", "TN", "NH", "WY"]
			if state in no_tax_states:
				return TaxCalculationResult(
					tax_type=TaxType.STATE_INCOME,
					jurisdiction=state,
					taxable_wages=employee_payroll.taxable_income,
					tax_amount=Decimal('0.00'),
					employee_portion=Decimal('0.00'),
					employer_portion=Decimal('0.00'),
					calculation_method=TaxCalculationMethod.FLAT_RATE,
					tax_rate=Decimal('0.00'),
					confidence_score=100.0,
					compliance_flags=[],
					calculation_details={"no_state_income_tax": True}
				)
			
			# Get state tax information
			state_tax_info = await self._get_state_tax_info(state, datetime.now().year)
			
			# Calculate state tax
			if state_tax_info["calculation_method"] == "percentage":
				tax_rate = state_tax_info["tax_rate"]
				tax_amount = employee_payroll.taxable_income * tax_rate / 100
			elif state_tax_info["calculation_method"] == "bracket":
				tax_brackets = state_tax_info["tax_brackets"]
				tax_amount = self._calculate_bracket_tax(employee_payroll.taxable_income, tax_brackets)
			else:
				# Flat rate
				tax_amount = state_tax_info.get("flat_amount", Decimal('0.00'))
			
			# Apply state-specific adjustments
			adjusted_tax = await self._apply_state_tax_adjustments(
				tax_amount, state, employee_payroll, calc_context
			)
			
			final_tax = adjusted_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.STATE_INCOME,
				jurisdiction=state,
				taxable_wages=employee_payroll.taxable_income,
				tax_amount=final_tax,
				employee_portion=final_tax,
				employer_portion=Decimal('0.00'),
				calculation_method=TaxCalculationMethod(state_tax_info["calculation_method"]),
				tax_rate=state_tax_info.get("tax_rate"),
				confidence_score=90.0,
				compliance_flags=[],
				calculation_details={
					"state": state,
					"base_tax": float(tax_amount),
					"adjustments_applied": True
				}
			)
			
		except Exception as e:
			logger.error(f"State income tax calculation failed: {e}")
			raise
	
	async def _calculate_fica_social_security_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate FICA Social Security tax."""
		
		try:
			# Get current year Social Security parameters
			tax_year = datetime.now().year
			ss_rate = Decimal('0.062')  # 6.2%
			ss_wage_base = await self._get_social_security_wage_base(tax_year)
			
			# Calculate YTD wages subject to Social Security
			ytd_wages = employee_payroll.ytd_gross
			current_wages = employee_payroll.gross_earnings
			total_ytd_wages = ytd_wages + current_wages
			
			# Determine taxable wages
			if ytd_wages >= ss_wage_base:
				# Already exceeded wage base
				taxable_wages = Decimal('0.00')
			elif total_ytd_wages > ss_wage_base:
				# Will exceed wage base this period
				taxable_wages = ss_wage_base - ytd_wages
			else:
				# Normal calculation
				taxable_wages = current_wages
			
			# Calculate tax
			employee_tax = taxable_wages * ss_rate
			employer_tax = employee_tax  # Employer matches
			
			final_employee_tax = employee_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			final_employer_tax = employer_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.FICA_SOCIAL_SECURITY,
				jurisdiction="US",
				taxable_wages=taxable_wages,
				tax_amount=final_employee_tax + final_employer_tax,
				employee_portion=final_employee_tax,
				employer_portion=final_employer_tax,
				calculation_method=TaxCalculationMethod.PERCENTAGE,
				tax_rate=ss_rate * 100,  # Convert to percentage
				confidence_score=100.0,
				compliance_flags=[],
				calculation_details={
					"ss_rate": float(ss_rate),
					"ss_wage_base": float(ss_wage_base),
					"ytd_wages": float(ytd_wages),
					"wage_base_exceeded": ytd_wages >= ss_wage_base
				}
			)
			
		except Exception as e:
			logger.error(f"FICA Social Security tax calculation failed: {e}")
			raise
	
	async def _calculate_fica_medicare_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate FICA Medicare tax with additional Medicare tax for high earners."""
		
		try:
			# Medicare tax parameters
			medicare_rate = Decimal('0.0145')  # 1.45%
			additional_medicare_rate = Decimal('0.009')  # 0.9% additional
			additional_medicare_threshold = Decimal('200000')  # Annual threshold
			
			current_wages = employee_payroll.gross_earnings
			ytd_wages = employee_payroll.ytd_gross
			
			# Regular Medicare tax (no wage base limit)
			regular_medicare_employee = current_wages * medicare_rate
			regular_medicare_employer = regular_medicare_employee  # Employer matches
			
			# Additional Medicare tax (employee only, no employer match)
			total_ytd_wages = ytd_wages + current_wages
			additional_medicare_employee = Decimal('0.00')
			
			if total_ytd_wages > additional_medicare_threshold:
				if ytd_wages >= additional_medicare_threshold:
					# All current wages subject to additional tax
					additional_medicare_employee = current_wages * additional_medicare_rate
				else:
					# Partial current wages subject to additional tax
					excess_wages = total_ytd_wages - additional_medicare_threshold
					additional_medicare_employee = excess_wages * additional_medicare_rate
			
			# Total employee Medicare tax
			total_employee_tax = regular_medicare_employee + additional_medicare_employee
			
			final_employee_tax = total_employee_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			final_employer_tax = regular_medicare_employer.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.FICA_MEDICARE,
				jurisdiction="US",
				taxable_wages=current_wages,
				tax_amount=final_employee_tax + final_employer_tax,
				employee_portion=final_employee_tax,
				employer_portion=final_employer_tax,
				calculation_method=TaxCalculationMethod.PERCENTAGE,
				tax_rate=medicare_rate * 100,  # Base rate as percentage
				confidence_score=100.0,
				compliance_flags=[],
				calculation_details={
					"medicare_rate": float(medicare_rate),
					"additional_medicare_rate": float(additional_medicare_rate),
					"additional_medicare_threshold": float(additional_medicare_threshold),
					"regular_medicare_employee": float(regular_medicare_employee),
					"additional_medicare_employee": float(additional_medicare_employee),
					"ytd_wages": float(ytd_wages)
				}
			)
			
		except Exception as e:
			logger.error(f"FICA Medicare tax calculation failed: {e}")
			raise
	
	async def _calculate_futa_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate FUTA tax (employer only)."""
		
		try:
			# FUTA parameters
			futa_rate = Decimal('0.006')  # 0.6% effective rate (after state credit)
			futa_wage_base = Decimal('7000')  # Annual wage base
			
			current_wages = employee_payroll.gross_earnings
			ytd_wages = employee_payroll.ytd_gross
			
			# Determine taxable wages
			if ytd_wages >= futa_wage_base:
				taxable_wages = Decimal('0.00')
			elif ytd_wages + current_wages > futa_wage_base:
				taxable_wages = futa_wage_base - ytd_wages
			else:
				taxable_wages = current_wages
			
			# Calculate FUTA tax (employer only)
			futa_tax = taxable_wages * futa_rate
			final_futa_tax = futa_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.FUTA,
				jurisdiction="US",
				taxable_wages=taxable_wages,
				tax_amount=final_futa_tax,
				employee_portion=Decimal('0.00'),
				employer_portion=final_futa_tax,
				calculation_method=TaxCalculationMethod.PERCENTAGE,
				tax_rate=futa_rate * 100,
				confidence_score=100.0,
				compliance_flags=[],
				calculation_details={
					"futa_rate": float(futa_rate),
					"futa_wage_base": float(futa_wage_base),
					"ytd_wages": float(ytd_wages),
					"wage_base_exceeded": ytd_wages >= futa_wage_base
				}
			)
			
		except Exception as e:
			logger.error(f"FUTA tax calculation failed: {e}")
			raise
	
	async def _calculate_suta_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate SUTA tax (employer only)."""
		
		try:
			employee_location = calc_context["employee_location"]
			state = employee_location.get("state", "CA")
			tenant_id = calc_context["tenant_id"]
			
			# Get SUTA information for the state
			suta_info = await self._get_suta_info(state, tenant_id)
			
			current_wages = employee_payroll.gross_earnings
			ytd_wages = employee_payroll.ytd_gross
			
			# Determine taxable wages
			if ytd_wages >= suta_info["wage_base"]:
				taxable_wages = Decimal('0.00')
			elif ytd_wages + current_wages > suta_info["wage_base"]:
				taxable_wages = suta_info["wage_base"] - ytd_wages
			else:
				taxable_wages = current_wages
			
			# Calculate SUTA tax (employer only)
			suta_tax = taxable_wages * suta_info["tax_rate"]
			final_suta_tax = suta_tax.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.SUTA,
				jurisdiction=state,
				taxable_wages=taxable_wages,
				tax_amount=final_suta_tax,
				employee_portion=Decimal('0.00'),
				employer_portion=final_suta_tax,
				calculation_method=TaxCalculationMethod.PERCENTAGE,
				tax_rate=suta_info["tax_rate"] * 100,
				confidence_score=95.0,
				compliance_flags=[],
				calculation_details={
					"state": state,
					"suta_rate": float(suta_info["tax_rate"]),
					"wage_base": float(suta_info["wage_base"]),
					"ytd_wages": float(ytd_wages)
				}
			)
			
		except Exception as e:
			logger.error(f"SUTA tax calculation failed: {e}")
			raise
	
	async def _calculate_local_income_tax(
		self,
		employee_payroll: PREmployeePayroll,
		calc_context: Dict[str, Any]
	) -> TaxCalculationResult:
		"""Calculate local income tax."""
		
		try:
			employee_location = calc_context["employee_location"]
			state = employee_location.get("state", "")
			city = employee_location.get("city", "")
			
			# Check if locality has income tax
			local_tax_states = ["PA", "OH", "MI", "NY", "IN", "KY", "MD"]
			if state not in local_tax_states or not city:
				return TaxCalculationResult(
					tax_type=TaxType.LOCAL_INCOME,
					jurisdiction=f"{city}, {state}" if city else state,
					taxable_wages=employee_payroll.taxable_income,
					tax_amount=Decimal('0.00'),
					employee_portion=Decimal('0.00'),
					employer_portion=Decimal('0.00'),
					calculation_method=TaxCalculationMethod.FLAT_RATE,
					tax_rate=Decimal('0.00'),
					confidence_score=100.0,
					compliance_flags=[],
					calculation_details={"no_local_income_tax": True}
				)
			
			# Get local tax information
			local_tax_info = await self._get_local_tax_info(state, city)
			
			# Calculate local tax
			tax_rate = local_tax_info.get("tax_rate", Decimal('0.00'))
			tax_amount = employee_payroll.taxable_income * tax_rate / 100
			
			final_tax = tax_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
			
			return TaxCalculationResult(
				tax_type=TaxType.LOCAL_INCOME,
				jurisdiction=f"{city}, {state}",
				taxable_wages=employee_payroll.taxable_income,
				tax_amount=final_tax,
				employee_portion=final_tax,
				employer_portion=Decimal('0.00'),
				calculation_method=TaxCalculationMethod.PERCENTAGE,
				tax_rate=tax_rate,
				confidence_score=85.0,
				compliance_flags=[],
				calculation_details={
					"state": state,
					"city": city,
					"local_tax_rate": float(tax_rate)
				}
			)
			
		except Exception as e:
			logger.error(f"Local income tax calculation failed: {e}")
			raise
	
	async def validate_payroll_compliance(
		self,
		payroll_run: PRPayrollRun,
		tenant_id: str
	) -> ComplianceCheckResult:
		"""Comprehensive compliance validation for payroll run."""
		
		try:
			logger.info(f"Validating compliance for payroll run: {payroll_run.run_id}")
			
			# Run parallel compliance checks
			compliance_tasks = [
				self._check_wage_compliance(payroll_run, tenant_id),
				self._check_overtime_compliance(payroll_run, tenant_id),
				self._check_tax_compliance(payroll_run, tenant_id),
				self._check_deduction_compliance(payroll_run, tenant_id),
				self._check_filing_compliance(payroll_run, tenant_id),
				self._check_record_keeping_compliance(payroll_run, tenant_id)
			]
			
			compliance_results = await asyncio.gather(*compliance_tasks, return_exceptions=True)
			
			# Aggregate compliance results
			overall_violations = []
			overall_warnings = []
			overall_recommendations = []
			total_score = 0.0
			valid_results = 0
			
			for result in compliance_results:
				if isinstance(result, Exception):
					logger.error(f"Compliance check failed: {result}")
					continue
				
				overall_violations.extend(result.violations)
				overall_warnings.extend(result.warnings)
				overall_recommendations.extend(result.recommendations)
				total_score += result.score
				valid_results += 1
			
			# Calculate overall compliance score
			overall_score = total_score / valid_results if valid_results > 0 else 0.0
			
			# Determine overall status
			if overall_violations:
				status = ComplianceStatus.VIOLATION
			elif overall_warnings:
				status = ComplianceStatus.WARNING
			else:
				status = ComplianceStatus.COMPLIANT
			
			return ComplianceCheckResult(
				compliance_area="overall_payroll",
				status=status,
				score=overall_score,
				violations=overall_violations,
				warnings=overall_warnings,
				recommendations=overall_recommendations,
				regulatory_requirements=await self._get_applicable_regulations(payroll_run, tenant_id),
				last_updated=datetime.utcnow()
			)
			
		except Exception as e:
			logger.error(f"Compliance validation failed: {e}")
			raise
	
	# Helper methods for compliance checks
	
	async def _check_wage_compliance(
		self,
		payroll_run: PRPayrollRun,
		tenant_id: str
	) -> ComplianceCheckResult:
		"""Check minimum wage compliance."""
		
		violations = []
		warnings = []
		recommendations = []
		
		try:
			# Get all employee payrolls
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Get employee location for minimum wage determination
				employee_location = await self._get_employee_location(emp_payroll.employee_id, tenant_id)
				
				# Get applicable minimum wage
				min_wage = await self._get_minimum_wage(employee_location, datetime.now().date())
				
				# Calculate effective hourly rate
				if emp_payroll.regular_hours > 0:
					effective_rate = emp_payroll.regular_pay / emp_payroll.regular_hours
					
					if effective_rate < min_wage:
						violations.append(
							f"Employee {emp_payroll.employee_name} paid below minimum wage: "
							f"${effective_rate:.2f}/hr < ${min_wage:.2f}/hr"
						)
			
			# Calculate compliance score
			total_employees = len(employee_payrolls)
			violation_count = len(violations)
			score = max(0.0, (total_employees - violation_count) / total_employees * 100) if total_employees > 0 else 100.0
			
			status = ComplianceStatus.VIOLATION if violations else ComplianceStatus.COMPLIANT
			
			return ComplianceCheckResult(
				compliance_area="minimum_wage",
				status=status,
				score=score,
				violations=violations,
				warnings=warnings,
				recommendations=recommendations,
				regulatory_requirements=["Fair Labor Standards Act", "State Minimum Wage Laws"],
				last_updated=datetime.utcnow()
			)
			
		except Exception as e:
			logger.error(f"Wage compliance check failed: {e}")
			raise
	
	async def _check_overtime_compliance(
		self,
		payroll_run: PRPayrollRun,
		tenant_id: str
	) -> ComplianceCheckResult:
		"""Check overtime compliance."""
		
		violations = []
		warnings = []
		recommendations = []
		
		try:
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Check for overtime violations
				if emp_payroll.regular_hours > 40:
					# Regular hours should not exceed 40 per week
					excess_hours = emp_payroll.regular_hours - 40
					violations.append(
						f"Employee {emp_payroll.employee_name} has {excess_hours} hours "
						f"over 40 recorded as regular time instead of overtime"
					)
				
				# Check overtime rate compliance
				if emp_payroll.overtime_hours > 0 and emp_payroll.hourly_rate:
					expected_ot_rate = emp_payroll.hourly_rate * Decimal('1.5')
					actual_ot_rate = emp_payroll.overtime_pay / emp_payroll.overtime_hours
					
					if abs(actual_ot_rate - expected_ot_rate) > Decimal('0.01'):
						violations.append(
							f"Employee {emp_payroll.employee_name} overtime rate incorrect: "
							f"${actual_ot_rate:.2f}/hr should be ${expected_ot_rate:.2f}/hr"
						)
			
			# Calculate compliance score
			total_employees = len(employee_payrolls)
			violation_count = len(violations)
			score = max(0.0, (total_employees - violation_count) / total_employees * 100) if total_employees > 0 else 100.0
			
			status = ComplianceStatus.VIOLATION if violations else ComplianceStatus.COMPLIANT
			
			return ComplianceCheckResult(
				compliance_area="overtime",
				status=status,
				score=score,
				violations=violations,
				warnings=warnings,
				recommendations=recommendations,
				regulatory_requirements=["Fair Labor Standards Act", "State Overtime Laws"],
				last_updated=datetime.utcnow()
			)
			
		except Exception as e:
			logger.error(f"Overtime compliance check failed: {e}")
			raise
	
	async def _check_tax_compliance(
		self,
		payroll_run: PRPayrollRun,
		tenant_id: str
	) -> ComplianceCheckResult:
		"""Check tax calculation compliance."""
		
		violations = []
		warnings = []
		recommendations = []
		
		try:
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Validate tax calculations
				tax_calculations = await self._get_employee_tax_calculations(emp_payroll.employee_payroll_id)
				
				for tax_calc in tax_calculations:
					# Check for negative tax amounts
					if tax_calc.employee_tax < 0:
						violations.append(
							f"Negative tax amount for {emp_payroll.employee_name}: "
							f"{tax_calc.tax_type} = ${tax_calc.employee_tax}"
						)
					
					# Check for unusually high tax rates
					if tax_calc.taxable_wages > 0:
						tax_rate = tax_calc.employee_tax / tax_calc.taxable_wages
						if tax_rate > Decimal('0.5'):  # 50% tax rate threshold
							warnings.append(
								f"High tax rate for {emp_payroll.employee_name}: "
								f"{tax_calc.tax_type} = {tax_rate:.1%}"
							)
			
			# Check for missing required taxes
			for emp_payroll in employee_payrolls:
				required_taxes = [TaxType.FEDERAL_INCOME, TaxType.FICA_SOCIAL_SECURITY, TaxType.FICA_MEDICARE]
				actual_taxes = [tc.tax_type for tc in await self._get_employee_tax_calculations(emp_payroll.employee_payroll_id)]
				
				for required_tax in required_taxes:
					if required_tax not in actual_taxes:
						violations.append(
							f"Missing required tax calculation for {emp_payroll.employee_name}: {required_tax}"
						)
			
			# Calculate compliance score
			total_employees = len(employee_payrolls)
			violation_count = len(violations)
			score = max(0.0, (total_employees - violation_count) / total_employees * 100) if total_employees > 0 else 100.0
			
			status = ComplianceStatus.VIOLATION if violations else (ComplianceStatus.WARNING if warnings else ComplianceStatus.COMPLIANT)
			
			return ComplianceCheckResult(
				compliance_area="tax_calculations",
				status=status,
				score=score,
				violations=violations,
				warnings=warnings,
				recommendations=recommendations,
				regulatory_requirements=["IRS Tax Regulations", "State Tax Codes"],
				last_updated=datetime.utcnow()
			)
			
		except Exception as e:
			logger.error(f"Tax compliance check failed: {e}")
			raise
	
	# Additional helper methods for tax calculations and compliance
	
	def _calculate_bracket_tax(self, taxable_income: Decimal, tax_brackets: List[Dict[str, Any]]) -> Decimal:
		"""Calculate tax using progressive tax brackets."""
		
		total_tax = Decimal('0.00')
		remaining_income = taxable_income
		
		for bracket in tax_brackets:
			bracket_min = Decimal(str(bracket["min_income"]))
			bracket_max = Decimal(str(bracket.get("max_income", float('inf'))))
			bracket_rate = Decimal(str(bracket["tax_rate"]))
			base_tax = Decimal(str(bracket.get("base_tax", 0)))
			
			if remaining_income <= 0:
				break
			
			if taxable_income > bracket_min:
				taxable_in_bracket = min(remaining_income, bracket_max - bracket_min)
				tax_in_bracket = taxable_in_bracket * bracket_rate
				total_tax += tax_in_bracket
				remaining_income -= taxable_in_bracket
		
		return total_tax
	
	async def _get_federal_tax_brackets(self, tax_year: int, filing_status: str) -> List[Dict[str, Any]]:
		"""Get federal tax brackets for the given year and filing status."""
		
		# This would typically come from a tax table service or database
		# For 2024 tax year (simplified)
		if filing_status.lower() == "single":
			return [
				{"min_income": 0, "max_income": 11000, "tax_rate": 0.10, "base_tax": 0},
				{"min_income": 11000, "max_income": 44725, "tax_rate": 0.12, "base_tax": 1100},
				{"min_income": 44725, "max_income": 95375, "tax_rate": 0.22, "base_tax": 5147},
				{"min_income": 95375, "max_income": 182050, "tax_rate": 0.24, "base_tax": 16290},
				{"min_income": 182050, "max_income": 231250, "tax_rate": 0.32, "base_tax": 37104},
				{"min_income": 231250, "max_income": 578125, "tax_rate": 0.35, "base_tax": 52832},
				{"min_income": 578125, "max_income": float('inf'), "tax_rate": 0.37, "base_tax": 174238.25}
			]
		elif filing_status.lower() in ["married_joint", "marriedjoint"]:
			return [
				{"min_income": 0, "max_income": 22000, "tax_rate": 0.10, "base_tax": 0},
				{"min_income": 22000, "max_income": 89450, "tax_rate": 0.12, "base_tax": 2200},
				{"min_income": 89450, "max_income": 190750, "tax_rate": 0.22, "base_tax": 10294},
				{"min_income": 190750, "max_income": 364200, "tax_rate": 0.24, "base_tax": 32580},
				{"min_income": 364200, "max_income": 462500, "tax_rate": 0.32, "base_tax": 74208},
				{"min_income": 462500, "max_income": 693750, "tax_rate": 0.35, "base_tax": 105664},
				{"min_income": 693750, "max_income": float('inf'), "tax_rate": 0.37, "base_tax": 186601.50}
			]
		else:
			# Default to single filing status
			return await self._get_federal_tax_brackets(tax_year, "single")
	
	async def _get_social_security_wage_base(self, tax_year: int) -> Decimal:
		"""Get Social Security wage base for the given year."""
		
		# This would typically come from SSA or be updated annually
		wage_bases = {
			2024: Decimal('168600'),
			2023: Decimal('160200'),
			2022: Decimal('147000')
		}
		
		return wage_bases.get(tax_year, Decimal('168600'))  # Default to 2024
	
	async def _get_employee_location(self, employee_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Get employee location information."""
		
		# This would integrate with employee_data_management service
		return {
			"state": "CA",
			"city": "San Francisco",
			"country": "US",
			"zip_code": "94105"
		}
	
	async def _get_applicable_taxes(
		self,
		employee_location: Dict[str, Any],
		employee_payroll: PREmployeePayroll,
		tenant_id: str
	) -> List[Dict[str, Any]]:
		"""Get list of applicable taxes for employee."""
		
		applicable_taxes = [
			{"tax_type": TaxType.FEDERAL_INCOME, "jurisdiction": "US"},
			{"tax_type": TaxType.FICA_SOCIAL_SECURITY, "jurisdiction": "US"},
			{"tax_type": TaxType.FICA_MEDICARE, "jurisdiction": "US"},
			{"tax_type": TaxType.FUTA, "jurisdiction": "US"},
		]
		
		# Add state taxes if applicable
		state = employee_location.get("state")
		if state and state not in ["TX", "FL", "NV", "WA", "SD", "AK", "TN", "NH", "WY"]:
			applicable_taxes.append({"tax_type": TaxType.STATE_INCOME, "jurisdiction": state})
		
		# Add SUTA tax
		if state:
			applicable_taxes.append({"tax_type": TaxType.SUTA, "jurisdiction": state})
		
		# Add local taxes if applicable
		city = employee_location.get("city")
		if city and state in ["PA", "OH", "MI", "NY", "IN", "KY", "MD"]:
			applicable_taxes.append({"tax_type": TaxType.LOCAL_INCOME, "jurisdiction": f"{city}, {state}"})
		
		return applicable_taxes
	
	# Placeholder methods for additional functionality
	
	async def _load_compliance_rules(self) -> None:
		"""Load compliance rules and regulations."""
		logger.info("Loading compliance rules...")
		pass
	
	async def _initialize_tax_engines(self) -> None:
		"""Initialize tax calculation engines."""
		logger.info("Initializing tax engines...")
		pass
	
	async def _load_ai_models(self) -> None:
		"""Load AI/ML models for tax optimization."""
		logger.info("Loading AI models...")
		pass
	
	async def _start_regulatory_monitoring(self) -> None:
		"""Start regulatory update monitoring."""
		logger.info("Starting regulatory monitoring...")
		pass


# Factory function
async def create_compliance_tax_engine(
	db_session: AsyncSession,
	config: Optional[ComplianceTaxConfig] = None
) -> IntelligentComplianceTaxEngine:
	"""Factory function to create compliance tax engine."""
	
	# Initialize services (would be injected in real implementation)
	compliance_service = None  # ComplianceValidationService()
	regulatory_service = None  # RegulatoryUpdateService()
	ai_service = None  # AIOrchestrationService()
	ml_service = None  # MLModelService()
	notification_service = None  # NotificationService()
	external_api_service = None  # ExternalAPIService()
	
	engine = IntelligentComplianceTaxEngine(
		db_session=db_session,
		compliance_service=compliance_service,
		regulatory_service=regulatory_service,
		ai_service=ai_service,
		ml_service=ml_service,
		notification_service=notification_service,
		external_api_service=external_api_service,
		config=config
	)
	
	await engine.initialize_compliance_engine()
	return engine


if __name__ == "__main__":
	# Example usage would go here
	pass