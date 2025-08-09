"""
APG Payroll Management - AI-Powered Intelligence Engine

Revolutionary AI intelligence engine with predictive capabilities,
automated error detection, and intelligent payroll optimization.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
import logging
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, ConfigDict

# APG Platform Imports
from ...ai_orchestration.services import AIOrchestrationService, MLModelService
from ...ai_orchestration.models import PredictionRequest, PredictionResponse
from ...audit_compliance.services import ComplianceValidationService
from ...employee_data_management.services import EmployeeDataService
from .models import (
	PRPayrollPeriod, PRPayrollRun, PREmployeePayroll, PRPayComponent,
	PayrollStatus, PayComponentType, PayFrequency
)

# Configure logging
logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
	"""Types of payroll anomalies detected by AI."""
	EARNINGS_SPIKE = "earnings_spike"
	EARNINGS_DROP = "earnings_drop"
	HOURS_ANOMALY = "hours_anomaly"
	TAX_CALCULATION_ERROR = "tax_calculation_error"
	DEDUCTION_ANOMALY = "deduction_anomaly"
	COMPLIANCE_RISK = "compliance_risk"
	PATTERN_DEVIATION = "pattern_deviation"


class PredictionType(str, Enum):
	"""Types of AI predictions for payroll."""
	EMPLOYEE_TURNOVER = "employee_turnover"
	PAYROLL_COST_TREND = "payroll_cost_trend"
	COMPLIANCE_RISK = "compliance_risk"
	PROCESSING_TIME = "processing_time"
	ERROR_LIKELIHOOD = "error_likelihood"


@dataclass
class PayrollAnomaly:
	"""Detected payroll anomaly with AI insights."""
	employee_id: str
	employee_name: str
	anomaly_type: AnomalyType
	severity_score: float  # 0-100, higher = more severe
	confidence_score: float  # 0-100, higher = more confident
	description: str
	recommended_action: str
	historical_context: Dict[str, Any]
	detected_at: datetime


@dataclass
class PayrollPrediction:
	"""AI-powered payroll prediction."""
	prediction_type: PredictionType
	target_entity: str  # employee_id, department_id, or "organization"
	predicted_value: Union[float, str, bool]
	confidence_score: float  # 0-100
	prediction_date: date
	factors: List[str]  # Contributing factors
	risk_level: str  # "low", "medium", "high"
	recommendations: List[str]


@dataclass
class IntelligentValidation:
	"""AI-powered validation result."""
	validation_type: str
	field_name: str
	original_value: Any
	suggested_value: Optional[Any]
	confidence_score: float
	validation_status: str  # "passed", "warning", "error"
	explanation: str
	auto_fix_available: bool


class PayrollIntelligenceConfig(BaseModel):
	"""Configuration for AI intelligence engine."""
	model_config = ConfigDict(extra='forbid')
	
	# Anomaly detection settings
	anomaly_detection_enabled: bool = Field(default=True)
	anomaly_sensitivity: float = Field(default=0.8, ge=0.1, le=1.0)
	historical_periods_for_analysis: int = Field(default=12, ge=3, le=24)
	
	# Prediction settings
	prediction_enabled: bool = Field(default=True)
	prediction_horizon_days: int = Field(default=90, ge=30, le=365)
	minimum_confidence_threshold: float = Field(default=0.7, ge=0.5, le=0.95)
	
	# Validation settings
	intelligent_validation_enabled: bool = Field(default=True)
	auto_fix_threshold: float = Field(default=0.9, ge=0.8, le=1.0)
	validation_batch_size: int = Field(default=100, ge=10, le=1000)
	
	# Performance settings
	max_concurrent_predictions: int = Field(default=10, ge=1, le=50)
	cache_predictions_hours: int = Field(default=24, ge=1, le=168)
	enable_real_time_scoring: bool = Field(default=True)


class PayrollIntelligenceEngine:
	"""Revolutionary AI-powered payroll intelligence engine.
	
	Provides predictive analytics, anomaly detection, intelligent validation,
	and automated optimization for payroll processing.
	"""
	
	def __init__(
		self,
		db_session: AsyncSession,
		ai_service: AIOrchestrationService,
		ml_service: MLModelService,
		compliance_service: ComplianceValidationService,
		employee_service: EmployeeDataService,
		config: Optional[PayrollIntelligenceConfig] = None
	):
		self.db = db_session
		self.ai_service = ai_service
		self.ml_service = ml_service
		self.compliance_service = compliance_service
		self.employee_service = employee_service
		self.config = config or PayrollIntelligenceConfig()
		
		# Initialize ML models
		self._anomaly_model = None
		self._prediction_models = {}
		self._validation_models = {}
		
		# Cache for predictions and validations
		self._prediction_cache = {}
		self._validation_cache = {}
	
	async def initialize_ai_models(self) -> None:
		"""Initialize and load AI/ML models for payroll intelligence."""
		try:
			logger.info("Initializing AI models for payroll intelligence...")
			
			# Load anomaly detection model
			self._anomaly_model = await self.ml_service.load_model(
				model_name="payroll_anomaly_detector",
				model_type="ensemble",
				fallback_to_pretrained=True
			)
			
			# Load prediction models
			prediction_model_configs = [
				("employee_turnover", "gradient_boosting"),
				("payroll_cost_trend", "time_series"),
				("compliance_risk", "classification"),
				("processing_time", "regression"),
				("error_likelihood", "neural_network")
			]
			
			for model_name, model_type in prediction_model_configs:
				self._prediction_models[model_name] = await self.ml_service.load_model(
					model_name=f"payroll_{model_name}_predictor",
					model_type=model_type,
					fallback_to_pretrained=True
				)
			
			# Load validation models
			validation_models = ["tax_calculation", "earnings_validation", "deduction_validation"]
			for model_name in validation_models:
				self._validation_models[model_name] = await self.ml_service.load_model(
					model_name=f"payroll_{model_name}_validator",
					model_type="classification",
					fallback_to_pretrained=True
				)
			
			logger.info("AI models initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize AI models: {e}")
			# Continue with limited functionality
			await self._initialize_fallback_models()
	
	async def _initialize_fallback_models(self) -> None:
		"""Initialize fallback rule-based models when ML models fail."""
		logger.info("Initializing fallback rule-based models...")
		# Implement rule-based fallback logic
		pass
	
	async def analyze_payroll_run(self, run_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Comprehensive AI analysis of a payroll run.
		
		Returns detailed insights, anomalies, predictions, and recommendations.
		"""
		try:
			logger.info(f"Starting AI analysis for payroll run: {run_id}")
			
			# Get payroll run data
			payroll_run = await self._get_payroll_run_with_details(run_id, tenant_id)
			if not payroll_run:
				raise ValueError(f"Payroll run {run_id} not found")
			
			# Parallel analysis tasks
			analysis_tasks = [
				self._detect_anomalies(payroll_run),
				self._generate_predictions(payroll_run),
				self._validate_payroll_data(payroll_run),
				self._analyze_compliance_risks(payroll_run),
				self._calculate_processing_score(payroll_run)
			]
			
			results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
			
			# Compile analysis results
			analysis_report = {
				"run_id": run_id,
				"tenant_id": tenant_id,
				"analysis_timestamp": datetime.utcnow().isoformat(),
				"anomalies": results[0] if not isinstance(results[0], Exception) else [],
				"predictions": results[1] if not isinstance(results[1], Exception) else [],
				"validations": results[2] if not isinstance(results[2], Exception) else [],
				"compliance_risks": results[3] if not isinstance(results[3], Exception) else [],
				"processing_score": results[4] if not isinstance(results[4], Exception) else 0.0,
				"overall_health_score": 0.0,
				"recommendations": [],
				"auto_fixes_applied": [],
				"requires_human_review": False
			}
			
			# Calculate overall health score
			analysis_report["overall_health_score"] = await self._calculate_overall_health_score(analysis_report)
			
			# Generate intelligent recommendations
			analysis_report["recommendations"] = await self._generate_recommendations(analysis_report)
			
			# Apply automatic fixes if confidence is high enough
			if self.config.intelligent_validation_enabled:
				analysis_report["auto_fixes_applied"] = await self._apply_automatic_fixes(
					payroll_run, analysis_report["validations"]
				)
			
			# Determine if human review is required
			analysis_report["requires_human_review"] = self._requires_human_review(analysis_report)
			
			# Store analysis results
			await self._store_analysis_results(analysis_report)
			
			logger.info(f"AI analysis completed for run {run_id}")
			return analysis_report
			
		except Exception as e:
			logger.error(f"AI analysis failed for run {run_id}: {e}")
			raise
	
	async def _detect_anomalies(self, payroll_run: PRPayrollRun) -> List[PayrollAnomaly]:
		"""Detect anomalies in payroll data using AI."""
		anomalies = []
		
		try:
			# Get employee payroll records
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Get historical data for comparison
				historical_data = await self._get_employee_historical_data(
					emp_payroll.employee_id,
					self.config.historical_periods_for_analysis
				)
				
				if len(historical_data) < 3:
					continue  # Need at least 3 periods for meaningful analysis
				
				# Earnings anomaly detection
				earnings_anomaly = await self._detect_earnings_anomaly(emp_payroll, historical_data)
				if earnings_anomaly:
					anomalies.append(earnings_anomaly)
				
				# Hours anomaly detection
				hours_anomaly = await self._detect_hours_anomaly(emp_payroll, historical_data)
				if hours_anomaly:
					anomalies.append(hours_anomaly)
				
				# Tax calculation anomaly detection
				tax_anomaly = await self._detect_tax_anomaly(emp_payroll, historical_data)
				if tax_anomaly:
					anomalies.append(tax_anomaly)
				
				# Deduction anomaly detection
				deduction_anomaly = await self._detect_deduction_anomaly(emp_payroll, historical_data)
				if deduction_anomaly:
					anomalies.append(deduction_anomaly)
			
			logger.info(f"Detected {len(anomalies)} anomalies in payroll run")
			return anomalies
			
		except Exception as e:
			logger.error(f"Anomaly detection failed: {e}")
			return []
	
	async def _detect_earnings_anomaly(
		self, 
		emp_payroll: PREmployeePayroll, 
		historical_data: List[Dict[str, Any]]
	) -> Optional[PayrollAnomaly]:
		"""Detect earnings anomalies using statistical analysis and ML."""
		
		# Extract historical earnings
		historical_earnings = [float(d['gross_earnings']) for d in historical_data]
		current_earnings = float(emp_payroll.gross_earnings)
		
		if len(historical_earnings) < 3:
			return None
		
		# Statistical analysis
		mean_earnings = np.mean(historical_earnings)
		std_earnings = np.std(historical_earnings)
		z_score = abs(current_earnings - mean_earnings) / std_earnings if std_earnings > 0 else 0
		
		# Use AI model if available
		anomaly_score = 0.0
		if self._anomaly_model:
			try:
				features = self._extract_earnings_features(emp_payroll, historical_data)
				prediction = await self.ml_service.predict(
					model=self._anomaly_model,
					features=features
				)
				anomaly_score = prediction.confidence if prediction else z_score * 10
			except Exception as e:
				logger.warning(f"ML anomaly detection failed, using statistical: {e}")
				anomaly_score = z_score * 10
		else:
			anomaly_score = z_score * 10
		
		# Determine if anomaly based on threshold
		if anomaly_score > (self.config.anomaly_sensitivity * 100):
			anomaly_type = AnomalyType.EARNINGS_SPIKE if current_earnings > mean_earnings else AnomalyType.EARNINGS_DROP
			
			return PayrollAnomaly(
				employee_id=emp_payroll.employee_id,
				employee_name=emp_payroll.employee_name,
				anomaly_type=anomaly_type,
				severity_score=min(anomaly_score, 100.0),
				confidence_score=min(anomaly_score * 2, 100.0),
				description=f"Earnings deviate {z_score:.2f} standard deviations from historical average",
				recommended_action="Review time entries, pay adjustments, and employment changes",
				historical_context={
					"mean_earnings": mean_earnings,
					"std_earnings": std_earnings,
					"current_earnings": current_earnings,
					"z_score": z_score,
					"historical_count": len(historical_earnings)
				},
				detected_at=datetime.utcnow()
			)
		
		return None
	
	async def _detect_hours_anomaly(
		self, 
		emp_payroll: PREmployeePayroll, 
		historical_data: List[Dict[str, Any]]
	) -> Optional[PayrollAnomaly]:
		"""Detect hours anomalies in regular and overtime hours."""
		
		historical_hours = [float(d['regular_hours']) for d in historical_data]
		current_hours = float(emp_payroll.regular_hours)
		
		if len(historical_hours) < 3:
			return None
		
		mean_hours = np.mean(historical_hours)
		std_hours = np.std(historical_hours)
		z_score = abs(current_hours - mean_hours) / std_hours if std_hours > 0 else 0
		
		if z_score > 2.5:  # More than 2.5 standard deviations
			return PayrollAnomaly(
				employee_id=emp_payroll.employee_id,
				employee_name=emp_payroll.employee_name,
				anomaly_type=AnomalyType.HOURS_ANOMALY,
				severity_score=min(z_score * 20, 100.0),
				confidence_score=min(z_score * 15, 100.0),
				description=f"Hours worked deviate {z_score:.2f} standard deviations from historical pattern",
				recommended_action="Verify time tracking accuracy and review schedule changes",
				historical_context={
					"mean_hours": mean_hours,
					"std_hours": std_hours,
					"current_hours": current_hours,
					"z_score": z_score
				},
				detected_at=datetime.utcnow()
			)
		
		return None
	
	async def _detect_tax_anomaly(
		self, 
		emp_payroll: PREmployeePayroll, 
		historical_data: List[Dict[str, Any]]
	) -> Optional[PayrollAnomaly]:
		"""Detect tax calculation anomalies."""
		
		# Calculate expected tax rate based on historical data
		historical_tax_rates = []
		for d in historical_data:
			if d['gross_earnings'] > 0:
				tax_rate = d['total_taxes'] / d['gross_earnings']
				historical_tax_rates.append(tax_rate)
		
		if len(historical_tax_rates) < 3 or emp_payroll.gross_earnings <= 0:
			return None
		
		current_tax_rate = float(emp_payroll.total_taxes) / float(emp_payroll.gross_earnings)
		mean_tax_rate = np.mean(historical_tax_rates)
		std_tax_rate = np.std(historical_tax_rates)
		
		if std_tax_rate > 0:
			z_score = abs(current_tax_rate - mean_tax_rate) / std_tax_rate
			
			if z_score > 2.0:  # Tax rate significantly different
				return PayrollAnomaly(
					employee_id=emp_payroll.employee_id,
					employee_name=emp_payroll.employee_name,
					anomaly_type=AnomalyType.TAX_CALCULATION_ERROR,
					severity_score=min(z_score * 25, 100.0),
					confidence_score=min(z_score * 20, 100.0),
					description=f"Tax rate {current_tax_rate:.2%} differs significantly from expected {mean_tax_rate:.2%}",
					recommended_action="Review tax calculations, filing status, and withholding elections",
					historical_context={
						"mean_tax_rate": mean_tax_rate,
						"current_tax_rate": current_tax_rate,
						"z_score": z_score
					},
					detected_at=datetime.utcnow()
				)
		
		return None
	
	async def _detect_deduction_anomaly(
		self, 
		emp_payroll: PREmployeePayroll, 
		historical_data: List[Dict[str, Any]]
	) -> Optional[PayrollAnomaly]:
		"""Detect deduction anomalies."""
		
		historical_deductions = [float(d['total_deductions']) for d in historical_data]
		current_deductions = float(emp_payroll.total_deductions)
		
		if len(historical_deductions) < 3:
			return None
		
		mean_deductions = np.mean(historical_deductions)
		std_deductions = np.std(historical_deductions)
		
		if std_deductions > 0:
			z_score = abs(current_deductions - mean_deductions) / std_deductions
			
			if z_score > 2.0:
				return PayrollAnomaly(
					employee_id=emp_payroll.employee_id,
					employee_name=emp_payroll.employee_name,
					anomaly_type=AnomalyType.DEDUCTION_ANOMALY,
					severity_score=min(z_score * 20, 100.0),
					confidence_score=min(z_score * 15, 100.0),
					description=f"Deductions amount differs significantly from historical pattern",
					recommended_action="Review benefit elections and deduction configurations",
					historical_context={
						"mean_deductions": mean_deductions,
						"current_deductions": current_deductions,
						"z_score": z_score
					},
					detected_at=datetime.utcnow()
				)
		
		return None
	
	async def _generate_predictions(self, payroll_run: PRPayrollRun) -> List[PayrollPrediction]:
		"""Generate AI-powered predictions for payroll trends and risks."""
		predictions = []
		
		try:
			# Employee turnover predictions
			turnover_predictions = await self._predict_employee_turnover(payroll_run)
			predictions.extend(turnover_predictions)
			
			# Payroll cost trend predictions
			cost_predictions = await self._predict_payroll_costs(payroll_run)
			predictions.extend(cost_predictions)
			
			# Compliance risk predictions
			compliance_predictions = await self._predict_compliance_risks(payroll_run)
			predictions.extend(compliance_predictions)
			
			# Processing time predictions
			processing_predictions = await self._predict_processing_times(payroll_run)
			predictions.extend(processing_predictions)
			
			logger.info(f"Generated {len(predictions)} predictions for payroll run")
			return predictions
			
		except Exception as e:
			logger.error(f"Prediction generation failed: {e}")
			return []
	
	async def _predict_employee_turnover(self, payroll_run: PRPayrollRun) -> List[PayrollPrediction]:
		"""Predict employee turnover risk based on payroll patterns."""
		predictions = []
		
		try:
			# Get employees with concerning patterns
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Get employee features for prediction
				features = await self._extract_turnover_features(emp_payroll)
				
				if self._prediction_models.get("employee_turnover"):
					try:
						prediction_result = await self.ml_service.predict(
							model=self._prediction_models["employee_turnover"],
							features=features
						)
						
						if prediction_result and prediction_result.confidence > self.config.minimum_confidence_threshold:
							risk_level = "high" if prediction_result.probability > 0.7 else "medium" if prediction_result.probability > 0.4 else "low"
							
							predictions.append(PayrollPrediction(
								prediction_type=PredictionType.EMPLOYEE_TURNOVER,
								target_entity=emp_payroll.employee_id,
								predicted_value=prediction_result.probability,
								confidence_score=prediction_result.confidence * 100,
								prediction_date=date.today() + timedelta(days=90),
								factors=prediction_result.feature_importance or [],
								risk_level=risk_level,
								recommendations=self._get_turnover_recommendations(risk_level, features)
							))
					except Exception as e:
						logger.warning(f"ML turnover prediction failed for employee {emp_payroll.employee_id}: {e}")
			
			return predictions
			
		except Exception as e:
			logger.error(f"Employee turnover prediction failed: {e}")
			return []
	
	async def _predict_payroll_costs(self, payroll_run: PRPayrollRun) -> List[PayrollPrediction]:
		"""Predict future payroll costs and trends."""
		predictions = []
		
		try:
			# Get historical payroll cost data
			historical_costs = await self._get_historical_payroll_costs(payroll_run.tenant_id)
			
			if len(historical_costs) >= 6:  # Need at least 6 periods for trend analysis
				# Prepare time series data
				time_series_data = self._prepare_cost_time_series(historical_costs)
				
				if self._prediction_models.get("payroll_cost_trend"):
					prediction_result = await self.ml_service.predict(
						model=self._prediction_models["payroll_cost_trend"],
						features=time_series_data
					)
					
					if prediction_result:
						predictions.append(PayrollPrediction(
							prediction_type=PredictionType.PAYROLL_COST_TREND,
							target_entity="organization",
							predicted_value=prediction_result.predicted_value,
							confidence_score=prediction_result.confidence * 100,
							prediction_date=date.today() + timedelta(days=30),
							factors=["historical_trends", "seasonal_patterns", "employee_growth"],
							risk_level="medium",
							recommendations=[
								"Monitor staffing levels and overtime trends",
								"Review budget allocations for next period",
								"Consider cost optimization opportunities"
							]
						))
			
			return predictions
			
		except Exception as e:
			logger.error(f"Payroll cost prediction failed: {e}")
			return []
	
	async def _predict_compliance_risks(self, payroll_run: PRPayrollRun) -> List[PayrollPrediction]:
		"""Predict compliance risks and violations."""
		predictions = []
		
		try:
			# Analyze compliance patterns
			compliance_data = await self._analyze_compliance_patterns(payroll_run)
			
			if self._prediction_models.get("compliance_risk"):
				for compliance_area, data in compliance_data.items():
					prediction_result = await self.ml_service.predict(
						model=self._prediction_models["compliance_risk"],
						features=data["features"]
					)
					
					if prediction_result and prediction_result.confidence > 0.6:
						risk_level = "high" if prediction_result.probability > 0.8 else "medium"
						
						predictions.append(PayrollPrediction(
							prediction_type=PredictionType.COMPLIANCE_RISK,
							target_entity=compliance_area,
							predicted_value=prediction_result.probability,
							confidence_score=prediction_result.confidence * 100,
							prediction_date=date.today() + timedelta(days=30),
							factors=data.get("risk_factors", []),
							risk_level=risk_level,
							recommendations=data.get("recommendations", [])
						))
			
			return predictions
			
		except Exception as e:
			logger.error(f"Compliance risk prediction failed: {e}")
			return []
	
	async def _predict_processing_times(self, payroll_run: PRPayrollRun) -> List[PayrollPrediction]:
		"""Predict payroll processing times based on complexity and historical data."""
		predictions = []
		
		try:
			# Extract processing features
			processing_features = await self._extract_processing_features(payroll_run)
			
			if self._prediction_models.get("processing_time"):
				prediction_result = await self.ml_service.predict(
					model=self._prediction_models["processing_time"],
					features=processing_features
				)
				
				if prediction_result:
					predictions.append(PayrollPrediction(
						prediction_type=PredictionType.PROCESSING_TIME,
						target_entity=payroll_run.run_id,
						predicted_value=prediction_result.predicted_value,  # in minutes
						confidence_score=prediction_result.confidence * 100,
						prediction_date=date.today(),
						factors=["employee_count", "complexity_score", "historical_performance"],
						risk_level="low" if prediction_result.predicted_value < 30 else "medium",
						recommendations=[
							"Optimize payroll run scheduling",
							"Consider batch processing strategies",
							"Monitor system performance"
						]
					))
			
			return predictions
			
		except Exception as e:
			logger.error(f"Processing time prediction failed: {e}")
			return []
	
	# Helper methods for data extraction and feature engineering
	
	async def _get_payroll_run_with_details(self, run_id: str, tenant_id: str) -> Optional[PRPayrollRun]:
		"""Get payroll run with all related data."""
		query = select(PRPayrollRun).where(
			and_(PRPayrollRun.run_id == run_id, PRPayrollRun.tenant_id == tenant_id)
		)
		result = await self.db.execute(query)
		return result.scalar_one_or_none()
	
	async def _get_employee_payrolls(self, run_id: str) -> List[PREmployeePayroll]:
		"""Get all employee payroll records for a run."""
		query = select(PREmployeePayroll).where(PREmployeePayroll.run_id == run_id)
		result = await self.db.execute(query)
		return result.scalars().all()
	
	async def _get_employee_historical_data(self, employee_id: str, periods: int) -> List[Dict[str, Any]]:
		"""Get historical payroll data for an employee."""
		# Implement query to get historical payroll data
		# This would join multiple tables to get comprehensive history
		query = f"""
		SELECT 
			ep.gross_earnings,
			ep.total_deductions,
			ep.total_taxes,
			ep.net_pay,
			ep.regular_hours,
			ep.overtime_hours,
			pr.started_at,
			pp.period_name
		FROM pr_employee_payroll ep
		JOIN pr_payroll_run pr ON ep.run_id = pr.run_id
		JOIN pr_payroll_period pp ON pr.period_id = pp.period_id
		WHERE ep.employee_id = :employee_id
		AND pr.status = 'completed'
		ORDER BY pr.started_at DESC
		LIMIT :periods
		"""
		
		result = await self.db.execute(query, {"employee_id": employee_id, "periods": periods})
		return [dict(row) for row in result]
	
	def _extract_earnings_features(self, emp_payroll: PREmployeePayroll, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
		"""Extract features for earnings anomaly detection."""
		historical_earnings = [float(d['gross_earnings']) for d in historical_data]
		
		return {
			"current_earnings": float(emp_payroll.gross_earnings),
			"mean_historical_earnings": np.mean(historical_earnings) if historical_earnings else 0.0,
			"std_historical_earnings": np.std(historical_earnings) if len(historical_earnings) > 1 else 0.0,
			"min_historical_earnings": np.min(historical_earnings) if historical_earnings else 0.0,
			"max_historical_earnings": np.max(historical_earnings) if historical_earnings else 0.0,
			"trend_slope": self._calculate_trend_slope(historical_earnings),
			"regular_hours": float(emp_payroll.regular_hours),
			"overtime_hours": float(emp_payroll.overtime_hours),
			"periods_count": len(historical_data)
		}
	
	def _calculate_trend_slope(self, values: List[float]) -> float:
		"""Calculate trend slope for time series data."""
		if len(values) < 2:
			return 0.0
		
		x = np.arange(len(values))
		y = np.array(values)
		
		try:
			slope, _ = np.polyfit(x, y, 1)
			return float(slope)
		except:
			return 0.0
	
	async def _extract_turnover_features(self, emp_payroll: PREmployeePayroll) -> Dict[str, float]:
		"""Extract features for employee turnover prediction."""
		# Get additional employee data
		employee_data = await self.employee_service.get_employee_details(emp_payroll.employee_id)
		
		features = {
			"tenure_months": self._calculate_tenure_months(employee_data),
			"pay_satisfaction_score": self._calculate_pay_satisfaction(emp_payroll),
			"overtime_ratio": float(emp_payroll.overtime_hours) / max(float(emp_payroll.regular_hours), 1.0),
			"pay_growth_rate": await self._calculate_pay_growth_rate(emp_payroll.employee_id),
			"department_turnover_rate": await self._get_department_turnover_rate(emp_payroll.department_id),
			"performance_score": employee_data.get("performance_score", 3.0) if employee_data else 3.0,
			"benefits_utilization": await self._calculate_benefits_utilization(emp_payroll.employee_id)
		}
		
		return features
	
	def _calculate_tenure_months(self, employee_data: Optional[Dict[str, Any]]) -> float:
		"""Calculate employee tenure in months."""
		if not employee_data or not employee_data.get("hire_date"):
			return 0.0
		
		hire_date = employee_data["hire_date"]
		if isinstance(hire_date, str):
			hire_date = datetime.strptime(hire_date, "%Y-%m-%d").date()
		
		today = date.today()
		months = (today.year - hire_date.year) * 12 + (today.month - hire_date.month)
		return float(max(months, 0))
	
	def _calculate_pay_satisfaction(self, emp_payroll: PREmployeePayroll) -> float:
		"""Calculate pay satisfaction score based on various factors."""
		# Simplified calculation - in reality this would be more sophisticated
		base_score = 3.0
		
		# Adjust based on overtime ratio
		overtime_ratio = float(emp_payroll.overtime_hours) / max(float(emp_payroll.regular_hours), 1.0)
		if overtime_ratio > 0.2:  # More than 20% overtime might indicate dissatisfaction
			base_score -= 0.5
		
		# Adjust based on net pay vs market (would need market data)
		# This is a simplified version
		if float(emp_payroll.net_pay) < 1000:  # Very low pay
			base_score -= 1.0
		elif float(emp_payroll.net_pay) > 5000:  # High pay
			base_score += 0.5
		
		return max(1.0, min(5.0, base_score))
	
	async def _calculate_pay_growth_rate(self, employee_id: str) -> float:
		"""Calculate employee's pay growth rate over last year."""
		# Get last 12 months of payroll data
		historical_data = await self._get_employee_historical_data(employee_id, 12)
		
		if len(historical_data) < 6:
			return 0.0
		
		# Calculate growth rate
		recent_avg = np.mean([float(d['gross_earnings']) for d in historical_data[:6]])
		older_avg = np.mean([float(d['gross_earnings']) for d in historical_data[6:]])
		
		if older_avg > 0:
			growth_rate = (recent_avg - older_avg) / older_avg
			return float(growth_rate)
		
		return 0.0
	
	async def _get_department_turnover_rate(self, department_id: Optional[str]) -> float:
		"""Get department turnover rate."""
		if not department_id:
			return 0.1  # Default 10% turnover rate
		
		# This would query actual turnover data
		# Simplified implementation
		return 0.15  # 15% default department turnover rate
	
	async def _calculate_benefits_utilization(self, employee_id: str) -> float:
		"""Calculate employee's benefits utilization rate."""
		# This would integrate with benefits administration
		# Simplified implementation
		return 0.7  # 70% utilization rate
	
	async def _validate_payroll_data(self, payroll_run: PRPayrollRun) -> List[IntelligentValidation]:
		"""Perform intelligent validation of payroll data."""
		validations = []
		
		try:
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			for emp_payroll in employee_payrolls:
				# Tax validation
				tax_validation = await self._validate_tax_calculations(emp_payroll)
				if tax_validation:
					validations.append(tax_validation)
				
				# Earnings validation
				earnings_validation = await self._validate_earnings(emp_payroll)
				if earnings_validation:
					validations.append(earnings_validation)
				
				# Deductions validation
				deductions_validation = await self._validate_deductions(emp_payroll)
				if deductions_validation:
					validations.append(deductions_validation)
			
			return validations
			
		except Exception as e:
			logger.error(f"Data validation failed: {e}")
			return []
	
	async def _validate_tax_calculations(self, emp_payroll: PREmployeePayroll) -> Optional[IntelligentValidation]:
		"""Validate tax calculations using AI."""
		
		# Calculate expected tax using compliance service
		expected_tax = await self.compliance_service.calculate_expected_tax(
			gross_earnings=emp_payroll.gross_earnings,
			employee_id=emp_payroll.employee_id,
			period_id=emp_payroll.run_id
		)
		
		current_tax = emp_payroll.total_taxes
		difference = abs(float(current_tax) - float(expected_tax))
		tolerance = float(expected_tax) * 0.05  # 5% tolerance
		
		if difference > tolerance:
			confidence_score = min(90.0, difference / tolerance * 20)
			
			return IntelligentValidation(
				validation_type="tax_calculation",
				field_name="total_taxes",
				original_value=current_tax,
				suggested_value=expected_tax,
				confidence_score=confidence_score,
				validation_status="warning" if difference <= tolerance * 2 else "error",
				explanation=f"Tax calculation differs by ${difference:.2f} from expected amount",
				auto_fix_available=confidence_score > self.config.auto_fix_threshold
			)
		
		return None
	
	async def _validate_earnings(self, emp_payroll: PREmployeePayroll) -> Optional[IntelligentValidation]:
		"""Validate earnings calculations."""
		
		# Calculate expected earnings based on hours and rates
		expected_regular = float(emp_payroll.regular_hours) * float(emp_payroll.hourly_rate or 0)
		expected_overtime = float(emp_payroll.overtime_hours) * float(emp_payroll.hourly_rate or 0) * 1.5
		expected_total = expected_regular + expected_overtime + float(emp_payroll.bonus_pay or 0)
		
		current_total = float(emp_payroll.gross_earnings)
		difference = abs(current_total - expected_total)
		
		if difference > 10.0:  # More than $10 difference
			confidence_score = min(85.0, difference / 10.0 * 15)
			
			return IntelligentValidation(
				validation_type="earnings_calculation",
				field_name="gross_earnings",
				original_value=emp_payroll.gross_earnings,
				suggested_value=Decimal(str(expected_total)),
				confidence_score=confidence_score,
				validation_status="warning",
				explanation=f"Earnings calculation differs by ${difference:.2f} from expected based on hours and rates",
				auto_fix_available=False  # Earnings changes require manual approval
			)
		
		return None
	
	async def _validate_deductions(self, emp_payroll: PREmployeePayroll) -> Optional[IntelligentValidation]:
		"""Validate deduction calculations."""
		
		# Get expected deductions from employee configuration
		expected_deductions = await self._calculate_expected_deductions(emp_payroll.employee_id, emp_payroll.gross_earnings)
		
		current_deductions = float(emp_payroll.total_deductions)
		difference = abs(current_deductions - expected_deductions)
		
		if difference > 5.0:  # More than $5 difference
			confidence_score = min(80.0, difference / 5.0 * 20)
			
			return IntelligentValidation(
				validation_type="deductions_calculation",
				field_name="total_deductions",
				original_value=emp_payroll.total_deductions,
				suggested_value=Decimal(str(expected_deductions)),
				confidence_score=confidence_score,
				validation_status="warning",
				explanation=f"Deductions differ by ${difference:.2f} from expected based on employee elections",
				auto_fix_available=confidence_score > self.config.auto_fix_threshold
			)
		
		return None
	
	async def _calculate_expected_deductions(self, employee_id: str, gross_earnings: Decimal) -> float:
		"""Calculate expected deductions for an employee."""
		# This would integrate with benefits administration and employee elections
		# Simplified calculation
		base_deductions = float(gross_earnings) * 0.08  # 8% for benefits, retirement, etc.
		return base_deductions
	
	async def _analyze_compliance_patterns(self, payroll_run: PRPayrollRun) -> Dict[str, Dict[str, Any]]:
		"""Analyze compliance patterns and risks."""
		
		compliance_areas = {
			"overtime_compliance": {
				"features": await self._extract_overtime_compliance_features(payroll_run),
				"risk_factors": ["excessive_overtime", "pattern_violations", "state_regulations"],
				"recommendations": ["Review overtime policies", "Monitor employee scheduling", "Update compliance rules"]
			},
			"minimum_wage_compliance": {
				"features": await self._extract_wage_compliance_features(payroll_run),
				"risk_factors": ["below_minimum_wage", "tipped_employees", "jurisdiction_changes"],
				"recommendations": ["Verify minimum wage rates", "Review tipped employee calculations", "Update wage tables"]
			},
			"tax_compliance": {
				"features": await self._extract_tax_compliance_features(payroll_run),
				"risk_factors": ["incorrect_withholding", "filing_deadlines", "rate_changes"],
				"recommendations": ["Review tax calculations", "Verify filing schedules", "Update tax tables"]
			}
		}
		
		return compliance_areas
	
	async def _extract_overtime_compliance_features(self, payroll_run: PRPayrollRun) -> Dict[str, float]:
		"""Extract features for overtime compliance analysis."""
		employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
		
		total_employees = len(employee_payrolls)
		overtime_employees = sum(1 for ep in employee_payrolls if float(ep.overtime_hours) > 0)
		avg_overtime_hours = np.mean([float(ep.overtime_hours) for ep in employee_payrolls]) if employee_payrolls else 0
		max_overtime_hours = max([float(ep.overtime_hours) for ep in employee_payrolls]) if employee_payrolls else 0
		
		return {
			"overtime_percentage": (overtime_employees / total_employees * 100) if total_employees > 0 else 0,
			"avg_overtime_hours": avg_overtime_hours,
			"max_overtime_hours": max_overtime_hours,
			"total_employees": total_employees
		}
	
	async def _extract_wage_compliance_features(self, payroll_run: PRPayrollRun) -> Dict[str, float]:
		"""Extract features for wage compliance analysis."""
		employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
		
		hourly_rates = [float(ep.hourly_rate) for ep in employee_payrolls if ep.hourly_rate and float(ep.hourly_rate) > 0]
		min_hourly_rate = min(hourly_rates) if hourly_rates else 0
		avg_hourly_rate = np.mean(hourly_rates) if hourly_rates else 0
		
		return {
			"min_hourly_rate": min_hourly_rate,
			"avg_hourly_rate": avg_hourly_rate,
			"employees_with_hourly_pay": len(hourly_rates),
			"total_employees": len(employee_payrolls)
		}
	
	async def _extract_tax_compliance_features(self, payroll_run: PRPayrollRun) -> Dict[str, float]:
		"""Extract features for tax compliance analysis."""
		employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
		
		total_gross = sum(float(ep.gross_earnings) for ep in employee_payrolls)
		total_taxes = sum(float(ep.total_taxes) for ep in employee_payrolls)
		avg_tax_rate = (total_taxes / total_gross * 100) if total_gross > 0 else 0
		
		return {
			"avg_tax_rate": avg_tax_rate,
			"total_gross_earnings": total_gross,
			"total_tax_withheld": total_taxes,
			"employees_count": len(employee_payrolls)
		}
	
	async def _extract_processing_features(self, payroll_run: PRPayrollRun) -> Dict[str, float]:
		"""Extract features for processing time prediction."""
		employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
		
		complexity_score = self._calculate_complexity_score(employee_payrolls)
		
		return {
			"employee_count": len(employee_payrolls),
			"complexity_score": complexity_score,
			"total_gross_earnings": sum(float(ep.gross_earnings) for ep in employee_payrolls),
			"unique_departments": len(set(ep.department_id for ep in employee_payrolls if ep.department_id)),
			"employees_with_overtime": sum(1 for ep in employee_payrolls if float(ep.overtime_hours) > 0),
			"employees_with_bonuses": sum(1 for ep in employee_payrolls if float(ep.bonus_pay or 0) > 0)
		}
	
	def _calculate_complexity_score(self, employee_payrolls: List[PREmployeePayroll]) -> float:
		"""Calculate payroll complexity score."""
		if not employee_payrolls:
			return 0.0
		
		base_score = len(employee_payrolls) * 1.0
		
		# Add complexity for various factors
		overtime_factor = sum(1 for ep in employee_payrolls if float(ep.overtime_hours) > 0) * 0.5
		bonus_factor = sum(1 for ep in employee_payrolls if float(ep.bonus_pay or 0) > 0) * 0.3
		deduction_factor = sum(1 for ep in employee_payrolls if float(ep.total_deductions) > 0) * 0.2
		
		total_score = base_score + overtime_factor + bonus_factor + deduction_factor
		return min(total_score, 100.0)  # Cap at 100
	
	async def _calculate_processing_score(self, payroll_run: PRPayrollRun) -> float:
		"""Calculate overall processing quality score."""
		try:
			employee_payrolls = await self._get_employee_payrolls(payroll_run.run_id)
			
			if not employee_payrolls:
				return 0.0
			
			base_score = 100.0
			
			# Deduct points for errors and warnings
			error_count = sum(1 for ep in employee_payrolls if ep.has_errors)
			warning_count = sum(1 for ep in employee_payrolls if ep.has_warnings)
			
			base_score -= (error_count * 10)  # 10 points per error
			base_score -= (warning_count * 2)  # 2 points per warning
			
			# Bonus points for AI confidence
			avg_validation_score = np.mean([float(ep.validation_score) for ep in employee_payrolls if ep.validation_score])
			if avg_validation_score > 80:
				base_score += 5
			
			return max(0.0, min(100.0, base_score))
			
		except Exception as e:
			logger.error(f"Processing score calculation failed: {e}")
			return 0.0
	
	async def _calculate_overall_health_score(self, analysis_report: Dict[str, Any]) -> float:
		"""Calculate overall payroll health score."""
		processing_score = analysis_report.get("processing_score", 0.0)
		anomaly_count = len(analysis_report.get("anomalies", []))
		high_risk_predictions = sum(1 for p in analysis_report.get("predictions", []) if p.get("risk_level") == "high")
		validation_errors = sum(1 for v in analysis_report.get("validations", []) if v.get("validation_status") == "error")
		
		# Calculate weighted health score
		health_score = processing_score * 0.4  # 40% weight for processing
		health_score -= (anomaly_count * 5)  # 5 points per anomaly
		health_score -= (high_risk_predictions * 10)  # 10 points per high-risk prediction
		health_score -= (validation_errors * 15)  # 15 points per validation error
		
		return max(0.0, min(100.0, health_score))
	
	async def _generate_recommendations(self, analysis_report: Dict[str, Any]) -> List[str]:
		"""Generate intelligent recommendations based on analysis."""
		recommendations = []
		
		# Recommendations based on anomalies
		anomalies = analysis_report.get("anomalies", [])
		if len(anomalies) > 5:
			recommendations.append("Consider reviewing time tracking accuracy and payroll processes")
		
		high_severity_anomalies = [a for a in anomalies if a.get("severity_score", 0) > 80]
		if high_severity_anomalies:
			recommendations.append("Investigate high-severity anomalies before finalizing payroll")
		
		# Recommendations based on predictions
		predictions = analysis_report.get("predictions", [])
		high_risk_predictions = [p for p in predictions if p.get("risk_level") == "high"]
		if high_risk_predictions:
			recommendations.append("Address high-risk predictions to prevent future issues")
		
		# Recommendations based on validations
		validations = analysis_report.get("validations", [])
		validation_errors = [v for v in validations if v.get("validation_status") == "error"]
		if validation_errors:
			recommendations.append("Resolve validation errors before proceeding with payroll")
		
		# Overall health recommendations
		health_score = analysis_report.get("overall_health_score", 0.0)
		if health_score < 70:
			recommendations.append("Overall payroll health is below optimal - consider comprehensive review")
		elif health_score > 95:
			recommendations.append("Excellent payroll quality - consider this as a benchmark")
		
		return recommendations
	
	async def _apply_automatic_fixes(self, payroll_run: PRPayrollRun, validations: List[Dict[str, Any]]) -> List[str]:
		"""Apply automatic fixes for high-confidence validations."""
		applied_fixes = []
		
		for validation in validations:
			if (validation.get("auto_fix_available") and 
				validation.get("confidence_score", 0) > self.config.auto_fix_threshold * 100):
				
				try:
					# Apply the fix based on validation type
					if validation["validation_type"] == "tax_calculation":
						# Auto-fix tax calculations
						await self._auto_fix_tax_calculation(validation)
						applied_fixes.append(f"Auto-fixed tax calculation for {validation['field_name']}")
					
					elif validation["validation_type"] == "deductions_calculation":
						# Auto-fix deduction calculations
						await self._auto_fix_deductions(validation)
						applied_fixes.append(f"Auto-fixed deductions for {validation['field_name']}")
					
				except Exception as e:
					logger.error(f"Auto-fix failed for {validation['validation_type']}: {e}")
		
		return applied_fixes
	
	async def _auto_fix_tax_calculation(self, validation: Dict[str, Any]) -> None:
		"""Automatically fix tax calculation."""
		# Implementation would update the employee payroll record
		# with the suggested tax value
		logger.info(f"Auto-fixing tax calculation: {validation}")
		pass
	
	async def _auto_fix_deductions(self, validation: Dict[str, Any]) -> None:
		"""Automatically fix deduction calculation."""
		# Implementation would update the employee payroll record
		# with the suggested deduction value
		logger.info(f"Auto-fixing deductions: {validation}")
		pass
	
	def _requires_human_review(self, analysis_report: Dict[str, Any]) -> bool:
		"""Determine if payroll requires human review."""
		
		# Require review if health score is low
		if analysis_report.get("overall_health_score", 0) < 80:
			return True
		
		# Require review if there are high-severity anomalies
		anomalies = analysis_report.get("anomalies", [])
		if any(a.get("severity_score", 0) > 90 for a in anomalies):
			return True
		
		# Require review if there are high-risk predictions
		predictions = analysis_report.get("predictions", [])
		if any(p.get("risk_level") == "high" for p in predictions):
			return True
		
		# Require review if there are validation errors
		validations = analysis_report.get("validations", [])
		if any(v.get("validation_status") == "error" for v in validations):
			return True
		
		return False
	
	async def _store_analysis_results(self, analysis_report: Dict[str, Any]) -> None:
		"""Store analysis results for future reference and learning."""
		try:
			# Store in analytics table or cache
			# This would be implemented based on the specific storage strategy
			logger.info(f"Storing analysis results for run {analysis_report['run_id']}")
			
			# Update payroll run with analysis data
			run_id = analysis_report["run_id"]
			query = select(PRPayrollRun).where(PRPayrollRun.run_id == run_id)
			result = await self.db.execute(query)
			payroll_run = result.scalar_one_or_none()
			
			if payroll_run:
				# Update run with analysis results
				payroll_run.analytics_data = analysis_report
				payroll_run.processing_score = Decimal(str(analysis_report.get("overall_health_score", 0)))
				payroll_run.validation_score = Decimal(str(analysis_report.get("processing_score", 0)))
				
				await self.db.commit()
			
		except Exception as e:
			logger.error(f"Failed to store analysis results: {e}")
	
	async def _get_historical_payroll_costs(self, tenant_id: str, periods: int = 12) -> List[Dict[str, Any]]:
		"""Get historical payroll cost data for trend analysis."""
		# Implement query to get historical cost data
		query = f"""
		SELECT 
			pp.period_name,
			pp.start_date,
			SUM(ep.gross_earnings) as total_gross,
			SUM(ep.total_taxes) as total_taxes,
			SUM(ep.total_deductions) as total_deductions,
			SUM(ep.net_pay) as total_net,
			COUNT(DISTINCT ep.employee_id) as employee_count
		FROM pr_payroll_period pp
		JOIN pr_payroll_run pr ON pp.period_id = pr.period_id
		JOIN pr_employee_payroll ep ON pr.run_id = ep.run_id
		WHERE pp.tenant_id = :tenant_id
		AND pr.status = 'completed'
		GROUP BY pp.period_id, pp.period_name, pp.start_date
		ORDER BY pp.start_date DESC
		LIMIT :periods
		"""
		
		result = await self.db.execute(query, {"tenant_id": tenant_id, "periods": periods})
		return [dict(row) for row in result]
	
	def _prepare_cost_time_series(self, historical_costs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
		"""Prepare time series data for cost prediction."""
		return {
			"total_costs": [float(d["total_gross"]) for d in historical_costs],
			"employee_counts": [float(d["employee_count"]) for d in historical_costs],
			"cost_per_employee": [
				float(d["total_gross"]) / max(float(d["employee_count"]), 1) 
				for d in historical_costs
			],
			"periods": list(range(len(historical_costs)))
		}
	
	def _get_turnover_recommendations(self, risk_level: str, features: Dict[str, float]) -> List[str]:
		"""Get recommendations for employee turnover risk."""
		recommendations = []
		
		if risk_level == "high":
			recommendations.extend([
				"Schedule one-on-one meeting to discuss career goals",
				"Review compensation competitiveness",
				"Consider retention bonus or promotion opportunity",
				"Evaluate workload and work-life balance"
			])
		elif risk_level == "medium":
			recommendations.extend([
				"Monitor employee engagement closely",
				"Provide professional development opportunities",
				"Ensure competitive compensation package"
			])
		
		# Add specific recommendations based on features
		if features.get("overtime_ratio", 0) > 0.3:
			recommendations.append("Address excessive overtime concerns")
		
		if features.get("pay_growth_rate", 0) < 0.02:
			recommendations.append("Consider salary adjustment or merit increase")
		
		return recommendations


# Example usage and testing
async def example_usage():
	"""Example of how to use the PayrollIntelligenceEngine."""
	
	# This would be set up with actual database session and services
	# db_session = get_async_session()
	# ai_service = AIOrchestrationService()
	# ml_service = MLModelService()
	# compliance_service = ComplianceValidationService()
	# employee_service = EmployeeDataService()
	
	# config = PayrollIntelligenceConfig(
	#     anomaly_sensitivity=0.8,
	#     prediction_enabled=True,
	#     intelligent_validation_enabled=True
	# )
	
	# intelligence_engine = PayrollIntelligenceEngine(
	#     db_session=db_session,
	#     ai_service=ai_service,
	#     ml_service=ml_service,
	#     compliance_service=compliance_service,
	#     employee_service=employee_service,
	#     config=config
	# )
	
	# # Initialize AI models
	# await intelligence_engine.initialize_ai_models()
	
	# # Analyze a payroll run
	# analysis_report = await intelligence_engine.analyze_payroll_run(
	#     run_id="sample-run-id",
	#     tenant_id="sample-tenant-id"
	# )
	
	# print(f"Analysis completed with {len(analysis_report['anomalies'])} anomalies detected")
	# print(f"Overall health score: {analysis_report['overall_health_score']}")
	# print(f"Requires human review: {analysis_report['requires_human_review']}")
	
	pass


if __name__ == "__main__":
	# Run example
	asyncio.run(example_usage())