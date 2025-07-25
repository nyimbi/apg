"""
Budgeting & Forecasting Service

Business logic for budgeting and forecasting operations including
budget creation, variance analysis, forecasting algorithms, and reporting.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import json
import statistics
from dataclasses import dataclass

from .models import (
	CFBFBudget, CFBFBudgetLine, CFBFBudgetScenario, CFBFBudgetVersion,
	CFBFForecast, CFBFForecastLine, CFBFActualVsBudget, CFBFDrivers,
	CFBFTemplate, CFBFApproval, CFBFAllocation
)
from ..general_ledger.models import CFGLAccount, CFGLPosting, CFGLPeriod


@dataclass
class VarianceThreshold:
	"""Variance analysis thresholds"""
	amount_threshold: Decimal
	percentage_threshold: Decimal
	critical_percentage: Decimal


@dataclass
class ForecastParameters:
	"""Forecasting algorithm parameters"""
	algorithm: str
	periods: int
	confidence_level: float
	seasonal_adjustment: bool
	trend_dampening: float


class CFBFBudgetService:
	"""Service class for budget operations"""
	
	def __init__(self, db_session: Session, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
	
	def _log_operation(self, operation: str, details: str = ""):
		"""Log service operations"""
		print(f"[CFBFBudgetService] {operation}: {details}")
	
	def create_budget(
		self,
		budget_name: str,
		fiscal_year: int,
		start_date: date,
		end_date: date,
		scenario_id: Optional[str] = None,
		template_id: Optional[str] = None,
		user_id: str = None
	) -> CFBFBudget:
		"""Create a new budget"""
		
		# Generate budget number
		budget_number = self._generate_budget_number(fiscal_year)
		
		budget = CFBFBudget(
			tenant_id=self.tenant_id,
			budget_number=budget_number,
			budget_name=budget_name,
			fiscal_year=fiscal_year,
			start_date=start_date,
			end_date=end_date,
			scenario_id=scenario_id,
			template_id=template_id,
			created_by=user_id
		)
		
		self.db.add(budget)
		self.db.flush()  # Get the ID
		
		# If template provided, create budget lines from template
		if template_id:
			self._create_lines_from_template(budget, template_id)
		
		self._log_operation("create_budget", f"Created budget {budget_number}")
		return budget
	
	def _generate_budget_number(self, fiscal_year: int) -> str:
		"""Generate unique budget number"""
		sequence = self.db.query(func.count(CFBFBudget.budget_id)).filter(
			CFBFBudget.tenant_id == self.tenant_id,
			CFBFBudget.fiscal_year == fiscal_year
		).scalar() + 1
		
		return f"BUD-{fiscal_year}-{sequence:04d}"
	
	def _create_lines_from_template(self, budget: CFBFBudget, template_id: str):
		"""Create budget lines from template"""
		template = self.db.query(CFBFTemplate).filter(
			CFBFTemplate.template_id == template_id,
			CFBFTemplate.tenant_id == self.tenant_id
		).first()
		
		if not template or not template.template_data:
			return
		
		line_number = 1
		for template_line in template.template_data.get('lines', []):
			budget_line = CFBFBudgetLine(
				tenant_id=self.tenant_id,
				budget_id=budget.budget_id,
				line_number=line_number,
				description=template_line.get('description'),
				account_id=template_line.get('account_id'),
				calculation_method=template_line.get('calculation_method', 'Manual'),
				annual_amount=Decimal(str(template_line.get('amount', 0))),
				total_amount=Decimal(str(template_line.get('amount', 0)))
			)
			
			self.db.add(budget_line)
			line_number += 1
	
	def add_budget_line(
		self,
		budget_id: str,
		account_id: str,
		amount: Decimal,
		description: str = None,
		driver_id: str = None,
		cost_center: str = None,
		department: str = None
	) -> CFBFBudgetLine:
		"""Add a line to budget"""
		
		# Get next line number
		max_line = self.db.query(func.max(CFBFBudgetLine.line_number)).filter(
			CFBFBudgetLine.budget_id == budget_id
		).scalar() or 0
		
		budget_line = CFBFBudgetLine(
			tenant_id=self.tenant_id,
			budget_id=budget_id,
			line_number=max_line + 1,
			description=description,
			account_id=account_id,
			driver_id=driver_id,
			annual_amount=amount,
			total_amount=amount,
			cost_center=cost_center,
			department=department
		)
		
		self.db.add(budget_line)
		
		# Recalculate budget totals
		self._recalculate_budget_totals(budget_id)
		
		return budget_line
	
	def _recalculate_budget_totals(self, budget_id: str):
		"""Recalculate budget totals"""
		budget = self.db.query(CFBFBudget).filter(
			CFBFBudget.budget_id == budget_id
		).first()
		
		if budget:
			budget.calculate_totals()
	
	def submit_budget_for_approval(self, budget_id: str, user_id: str) -> bool:
		"""Submit budget for approval workflow"""
		budget = self.db.query(CFBFBudget).filter(
			CFBFBudget.budget_id == budget_id,
			CFBFBudget.tenant_id == self.tenant_id
		).first()
		
		if not budget or budget.status != 'Draft':
			return False
		
		budget.submit_for_approval(user_id)
		
		# Create approval workflow entries
		self._create_approval_workflow(budget)
		
		self._log_operation("submit_budget", f"Budget {budget.budget_number} submitted")
		return True
	
	def _create_approval_workflow(self, budget: CFBFBudget):
		"""Create approval workflow entries"""
		# This would be configured based on organizational approval hierarchy
		approval_levels = [
			{'level': 1, 'role': 'Manager', 'required': True},
			{'level': 2, 'role': 'Director', 'required': True},
			{'level': 3, 'role': 'CFO', 'required': budget.total_expenses > 1000000}
		]
		
		for level_config in approval_levels:
			if level_config['required']:
				approval = CFBFApproval(
					tenant_id=self.tenant_id,
					budget_id=budget.budget_id,
					approval_level=level_config['level'],
					approver_role=level_config['role'],
					required=level_config['required']
				)
				self.db.add(approval)
	
	def approve_budget(self, budget_id: str, approver_id: str, level: int, comments: str = None) -> bool:
		"""Approve budget at specific level"""
		approval = self.db.query(CFBFApproval).filter(
			CFBFApproval.budget_id == budget_id,
			CFBFApproval.approval_level == level,
			CFBFApproval.approver_id == approver_id
		).first()
		
		if not approval or not approval.can_approve():
			return False
		
		approval.approve(comments)
		
		# Check if all required approvals are complete
		if self._check_all_approvals_complete(budget_id):
			budget = self.db.query(CFBFBudget).filter(
				CFBFBudget.budget_id == budget_id
			).first()
			if budget:
				budget.approve_budget(approver_id)
		
		return True
	
	def _check_all_approvals_complete(self, budget_id: str) -> bool:
		"""Check if all required approvals are complete"""
		pending_count = self.db.query(func.count(CFBFApproval.approval_id)).filter(
			CFBFApproval.budget_id == budget_id,
			CFBFApproval.required == True,
			CFBFApproval.status == 'Pending'
		).scalar()
		
		return pending_count == 0
	
	def copy_budget(self, source_budget_id: str, new_name: str, new_fiscal_year: int) -> CFBFBudget:
		"""Copy existing budget to new fiscal year"""
		source_budget = self.db.query(CFBFBudget).filter(
			CFBFBudget.budget_id == source_budget_id,
			CFBFBudget.tenant_id == self.tenant_id
		).first()
		
		if not source_budget:
			raise ValueError("Source budget not found")
		
		# Calculate new dates
		year_diff = new_fiscal_year - source_budget.fiscal_year
		new_start_date = date(source_budget.start_date.year + year_diff, 
							 source_budget.start_date.month, source_budget.start_date.day)
		new_end_date = date(source_budget.end_date.year + year_diff,
						   source_budget.end_date.month, source_budget.end_date.day)
		
		# Create new budget
		new_budget = self.create_budget(
			budget_name=new_name,
			fiscal_year=new_fiscal_year,
			start_date=new_start_date,
			end_date=new_end_date,
			scenario_id=source_budget.scenario_id
		)
		
		# Copy budget lines
		for source_line in source_budget.lines:
			self.add_budget_line(
				budget_id=new_budget.budget_id,
				account_id=source_line.account_id,
				amount=source_line.annual_amount,
				description=source_line.description,
				driver_id=source_line.driver_id,
				cost_center=source_line.cost_center,
				department=source_line.department
			)
		
		return new_budget
	
	def get_budget_summary(self, budget_id: str) -> Dict[str, Any]:
		"""Get comprehensive budget summary"""
		budget = self.db.query(CFBFBudget).filter(
			CFBFBudget.budget_id == budget_id,
			CFBFBudget.tenant_id == self.tenant_id
		).first()
		
		if not budget:
			return {}
		
		return {
			'budget_info': {
				'budget_id': budget.budget_id,
				'budget_number': budget.budget_number,
				'budget_name': budget.budget_name,
				'fiscal_year': budget.fiscal_year,
				'status': budget.status,
				'total_revenue': float(budget.total_revenue),
				'total_expenses': float(budget.total_expenses),
				'net_income': float(budget.net_income)
			},
			'line_count': budget.line_count,
			'approval_status': budget.approval_status,
			'scenario': budget.scenario.scenario_name if budget.scenario else None
		}


class CFBFVarianceAnalysisService:
	"""Service class for variance analysis operations"""
	
	def __init__(self, db_session: Session, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.thresholds = VarianceThreshold(
			amount_threshold=Decimal('10000.00'),
			percentage_threshold=Decimal('10.00'),
			critical_percentage=Decimal('25.00')
		)
	
	def _log_analysis(self, operation: str, details: str = ""):
		"""Log analysis operations"""
		print(f"[CFBFVarianceAnalysisService] {operation}: {details}")
	
	def generate_variance_analysis(
		self,
		budget_id: str,
		analysis_date: date,
		user_id: str
	) -> List[CFBFActualVsBudget]:
		"""Generate comprehensive variance analysis"""
		
		budget = self.db.query(CFBFBudget).filter(
			CFBFBudget.budget_id == budget_id,
			CFBFBudget.tenant_id == self.tenant_id
		).first()
		
		if not budget:
			raise ValueError("Budget not found")
		
		variance_records = []
		
		for budget_line in budget.lines:
			# Get actual amounts from GL postings
			actual_amounts = self._get_actual_amounts(
				budget_line.account_id,
				budget.start_date,
				analysis_date
			)
			
			# Get budget amounts for the period
			budget_amounts = self._get_budget_amounts(budget_line, analysis_date)
			
			# Create variance record
			variance = self._create_variance_record(
				budget_id=budget_id,
				budget_line=budget_line,
				analysis_date=analysis_date,
				budget_amounts=budget_amounts,
				actual_amounts=actual_amounts,
				user_id=user_id
			)
			
			variance_records.append(variance)
		
		self._log_analysis("generate_variance", f"Generated {len(variance_records)} variance records")
		return variance_records
	
	def _get_actual_amounts(self, account_id: str, start_date: date, end_date: date) -> Dict[str, Decimal]:
		"""Get actual amounts from GL postings"""
		
		# Current period actual
		period_actual = self.db.query(
			func.sum(CFGLPosting.debit_amount - CFGLPosting.credit_amount)
		).filter(
			CFGLPosting.account_id == account_id,
			CFGLPosting.tenant_id == self.tenant_id,
			CFGLPosting.posting_date == end_date,
			CFGLPosting.is_posted == True
		).scalar() or Decimal('0.00')
		
		# Year-to-date actual
		ytd_actual = self.db.query(
			func.sum(CFGLPosting.debit_amount - CFGLPosting.credit_amount)
		).filter(
			CFGLPosting.account_id == account_id,
			CFGLPosting.tenant_id == self.tenant_id,
			CFGLPosting.posting_date >= start_date,
			CFGLPosting.posting_date <= end_date,
			CFGLPosting.is_posted == True
		).scalar() or Decimal('0.00')
		
		return {
			'period_actual': period_actual,
			'ytd_actual': ytd_actual
		}
	
	def _get_budget_amounts(self, budget_line: CFBFBudgetLine, analysis_date: date) -> Dict[str, Decimal]:
		"""Get budget amounts for the period"""
		
		# Simple calculation - could be enhanced with period distribution
		months_in_year = 12
		current_month = analysis_date.month
		
		period_budget = budget_line.annual_amount / months_in_year
		ytd_budget = period_budget * current_month
		
		return {
			'period_budget': period_budget,
			'ytd_budget': ytd_budget
		}
	
	def _create_variance_record(
		self,
		budget_id: str,
		budget_line: CFBFBudgetLine,
		analysis_date: date,
		budget_amounts: Dict[str, Decimal],
		actual_amounts: Dict[str, Decimal],
		user_id: str
	) -> CFBFActualVsBudget:
		"""Create variance analysis record"""
		
		variance = CFBFActualVsBudget(
			tenant_id=self.tenant_id,
			budget_id=budget_id,
			account_id=budget_line.account_id,
			analysis_date=analysis_date,
			fiscal_year=analysis_date.year,
			period_number=analysis_date.month,
			budget_amount=budget_amounts['period_budget'],
			actual_amount=actual_amounts['period_actual'],
			ytd_budget_amount=budget_amounts['ytd_budget'],
			ytd_actual_amount=actual_amounts['ytd_actual'],
			generated_by=user_id
		)
		
		# Calculate variance
		variance.calculate_variance()
		
		# Determine alert level
		variance.determine_alert_level()
		
		self.db.add(variance)
		return variance
	
	def get_significant_variances(
		self,
		budget_id: str,
		analysis_date: date,
		alert_level: str = None
	) -> List[CFBFActualVsBudget]:
		"""Get significant variances requiring attention"""
		
		query = self.db.query(CFBFActualVsBudget).filter(
			CFBFActualVsBudget.budget_id == budget_id,
			CFBFActualVsBudget.tenant_id == self.tenant_id,
			CFBFActualVsBudget.analysis_date == analysis_date,
			CFBFActualVsBudget.is_significant == True
		)
		
		if alert_level:
			query = query.filter(CFBFActualVsBudget.alert_level == alert_level)
		
		return query.order_by(desc(CFBFActualVsBudget.variance_percent)).all()
	
	def get_variance_trends(self, account_id: str, periods: int = 12) -> List[Dict[str, Any]]:
		"""Get variance trends for an account over multiple periods"""
		
		end_date = date.today()
		start_date = end_date - timedelta(days=periods * 30)  # Approximate monthly periods
		
		variances = self.db.query(CFBFActualVsBudget).filter(
			CFBFActualVsBudget.account_id == account_id,
			CFBFActualVsBudget.tenant_id == self.tenant_id,
			CFBFActualVsBudget.analysis_date >= start_date,
			CFBFActualVsBudget.analysis_date <= end_date
		).order_by(CFBFActualVsBudget.analysis_date).all()
		
		return [
			{
				'analysis_date': var.analysis_date.isoformat(),
				'variance_amount': float(var.variance_amount),
				'variance_percent': float(var.variance_percent),
				'is_favorable': var.is_favorable,
				'alert_level': var.alert_level
			}
			for var in variances
		]


class CFBFForecastService:
	"""Service class for forecasting operations"""
	
	def __init__(self, db_session: Session, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
	
	def _log_forecast(self, operation: str, details: str = ""):
		"""Log forecasting operations"""
		print(f"[CFBFForecastService] {operation}: {details}")
	
	def create_forecast(
		self,
		forecast_name: str,
		forecast_type: str = 'Rolling',
		periods_ahead: int = 12,
		base_budget_id: str = None,
		scenario_id: str = None,
		user_id: str = None
	) -> CFBFForecast:
		"""Create a new forecast"""
		
		forecast_number = self._generate_forecast_number()
		
		forecast = CFBFForecast(
			tenant_id=self.tenant_id,
			forecast_number=forecast_number,
			forecast_name=forecast_name,
			forecast_type=forecast_type,
			forecast_date=date.today(),
			start_date=date.today(),
			end_date=date.today() + timedelta(days=periods_ahead * 30),
			periods_ahead=periods_ahead,
			base_budget_id=base_budget_id,
			scenario_id=scenario_id,
			created_by=user_id
		)
		
		self.db.add(forecast)
		self.db.flush()
		
		self._log_forecast("create_forecast", f"Created forecast {forecast_number}")
		return forecast
	
	def _generate_forecast_number(self) -> str:
		"""Generate unique forecast number"""
		current_year = date.today().year
		sequence = self.db.query(func.count(CFBFForecast.forecast_id)).filter(
			CFBFForecast.tenant_id == self.tenant_id,
			func.extract('year', CFBFForecast.forecast_date) == current_year
		).scalar() + 1
		
		return f"FC-{current_year}-{sequence:04d}"
	
	def generate_trend_forecast(
		self,
		forecast_id: str,
		account_id: str,
		historical_periods: int = 12,
		algorithm: str = 'Linear'
	) -> CFBFForecastLine:
		"""Generate forecast using trend analysis"""
		
		forecast = self.db.query(CFBFForecast).filter(
			CFBFForecast.forecast_id == forecast_id
		).first()
		
		if not forecast:
			raise ValueError("Forecast not found")
		
		# Get historical data
		historical_data = self._get_historical_data(account_id, historical_periods)
		
		if len(historical_data) < 3:
			raise ValueError("Insufficient historical data for trend analysis")
		
		# Calculate trend
		trend_factor = self._calculate_trend_factor(historical_data, algorithm)
		
		# Generate forecast line
		forecast_line = CFBFForecastLine(
			tenant_id=self.tenant_id,
			forecast_id=forecast_id,
			line_number=self._get_next_line_number(forecast_id),
			account_id=account_id,
			calculation_method='Trend',
			historical_periods=historical_periods,
			base_amount=historical_data[-1],  # Last historical value
			trend_factor=trend_factor,
			forecast_amount=historical_data[-1] * trend_factor
		)
		
		# Calculate period forecasts
		period_forecasts = forecast_line.calculate_trend_forecast(forecast.periods_ahead)
		forecast_line.period_forecasts = {str(k): float(v) for k, v in period_forecasts.items()}
		
		self.db.add(forecast_line)
		return forecast_line
	
	def _get_historical_data(self, account_id: str, periods: int) -> List[Decimal]:
		"""Get historical data for an account"""
		
		end_date = date.today()
		historical_amounts = []
		
		for i in range(periods):
			period_start = end_date - timedelta(days=(i+1) * 30)
			period_end = end_date - timedelta(days=i * 30)
			
			amount = self.db.query(
				func.sum(CFGLPosting.debit_amount - CFGLPosting.credit_amount)
			).filter(
				CFGLPosting.account_id == account_id,
				CFGLPosting.tenant_id == self.tenant_id,
				CFGLPosting.posting_date >= period_start,
				CFGLPosting.posting_date < period_end,
				CFGLPosting.is_posted == True
			).scalar() or Decimal('0.00')
			
			historical_amounts.append(amount)
		
		return list(reversed(historical_amounts))  # Chronological order
	
	def _calculate_trend_factor(self, historical_data: List[Decimal], algorithm: str) -> Decimal:
		"""Calculate trend factor based on algorithm"""
		
		if algorithm == 'Linear':
			# Simple linear regression slope
			n = len(historical_data)
			x_values = list(range(1, n + 1))
			y_values = [float(val) for val in historical_data]
			
			# Calculate slope
			x_mean = statistics.mean(x_values)
			y_mean = statistics.mean(y_values)
			
			numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
			denominator = sum((x - x_mean) ** 2 for x in x_values)
			
			if denominator == 0:
				return Decimal('1.0')
			
			slope = numerator / denominator
			return Decimal(str(1.0 + (slope / y_mean if y_mean != 0 else 0)))
		
		elif algorithm == 'Exponential':
			# Exponential smoothing
			if len(historical_data) < 2:
				return Decimal('1.0')
			
			growth_rates = []
			for i in range(1, len(historical_data)):
				if historical_data[i-1] != 0:
					growth_rate = historical_data[i] / historical_data[i-1]
					growth_rates.append(float(growth_rate))
			
			if not growth_rates:
				return Decimal('1.0')
			
			avg_growth = statistics.mean(growth_rates)
			return Decimal(str(avg_growth))
		
		else:
			# Default to simple average growth
			return Decimal('1.0')
	
	def _get_next_line_number(self, forecast_id: str) -> int:
		"""Get next line number for forecast"""
		max_line = self.db.query(func.max(CFBFForecastLine.line_number)).filter(
			CFBFForecastLine.forecast_id == forecast_id
		).scalar() or 0
		
		return max_line + 1
	
	def generate_driver_based_forecast(
		self,
		forecast_id: str,
		account_id: str,
		driver_id: str
	) -> CFBFForecastLine:
		"""Generate forecast based on driver values"""
		
		driver = self.db.query(CFBFDrivers).filter(
			CFBFDrivers.driver_id == driver_id,
			CFBFDrivers.tenant_id == self.tenant_id
		).first()
		
		if not driver:
			raise ValueError("Driver not found")
		
		forecast = self.db.query(CFBFForecast).filter(
			CFBFForecast.forecast_id == forecast_id
		).first()
		
		if not forecast:
			raise ValueError("Forecast not found")
		
		# Calculate forecast based on driver projections
		base_amount = driver.base_value or Decimal('0.00')
		
		forecast_line = CFBFForecastLine(
			tenant_id=self.tenant_id,
			forecast_id=forecast_id,
			line_number=self._get_next_line_number(forecast_id),
			account_id=account_id,
			driver_id=driver_id,
			calculation_method='Driver',
			base_amount=base_amount,
			forecast_amount=base_amount  # Would be calculated based on driver formula
		)
		
		# Calculate period forecasts using driver values
		period_forecasts = {}
		for period in range(1, forecast.periods_ahead + 1):
			driver_value = driver.calculate_period_value(period)
			period_forecasts[period] = driver_value
		
		forecast_line.period_forecasts = {str(k): float(v) for k, v in period_forecasts.items()}
		
		self.db.add(forecast_line)
		return forecast_line
	
	def get_forecast_accuracy(self, forecast_id: str) -> Dict[str, Any]:
		"""Calculate forecast accuracy by comparing to actuals"""
		
		forecast = self.db.query(CFBFForecast).filter(
			CFBFForecast.forecast_id == forecast_id,
			CFBFForecast.tenant_id == self.tenant_id
		).first()
		
		if not forecast:
			return {}
		
		accuracy_metrics = {
			'forecast_id': forecast_id,
			'forecast_date': forecast.forecast_date.isoformat(),
			'line_accuracies': [],
			'overall_accuracy': 0.0,
			'mean_absolute_error': 0.0
		}
		
		total_error = 0.0
		line_count = 0
		
		for line in forecast.lines:
			# Get actual amounts for comparison (simplified)
			actual_amount = self._get_actual_amount_for_period(
				line.account_id,
				forecast.forecast_date,
				forecast.forecast_date + timedelta(days=30)
			)
			
			if line.forecast_amount != 0:
				error_percent = abs(actual_amount - line.forecast_amount) / line.forecast_amount * 100
				accuracy = max(0, 100 - error_percent)
			else:
				accuracy = 100 if actual_amount == 0 else 0
			
			accuracy_metrics['line_accuracies'].append({
				'account_id': line.account_id,
				'forecast_amount': float(line.forecast_amount),
				'actual_amount': float(actual_amount),
				'accuracy_percent': float(accuracy)
			})
			
			total_error += error_percent
			line_count += 1
		
		if line_count > 0:
			accuracy_metrics['mean_absolute_error'] = total_error / line_count
			accuracy_metrics['overall_accuracy'] = max(0, 100 - accuracy_metrics['mean_absolute_error'])
		
		return accuracy_metrics
	
	def _get_actual_amount_for_period(self, account_id: str, start_date: date, end_date: date) -> Decimal:
		"""Get actual amount for a specific period"""
		return self.db.query(
			func.sum(CFGLPosting.debit_amount - CFGLPosting.credit_amount)
		).filter(
			CFGLPosting.account_id == account_id,
			CFGLPosting.tenant_id == self.tenant_id,
			CFGLPosting.posting_date >= start_date,
			CFGLPosting.posting_date <= end_date,
			CFGLPosting.is_posted == True
		).scalar() or Decimal('0.00')


class CFBFDriverService:
	"""Service class for budget driver operations"""
	
	def __init__(self, db_session: Session, tenant_id: str):
		self.db = db_session
		self.tenant_id = tenant_id
	
	def _log_driver_operation(self, operation: str, details: str = ""):
		"""Log driver operations"""
		print(f"[CFBFDriverService] {operation}: {details}")
	
	def create_driver(
		self,
		driver_code: str,
		driver_name: str,
		data_type: str = 'Numeric',
		base_value: Decimal = None,
		growth_rate: Decimal = None,
		unit_of_measure: str = None
	) -> CFBFDrivers:
		"""Create a new budget driver"""
		
		driver = CFBFDrivers(
			tenant_id=self.tenant_id,
			driver_code=driver_code,
			driver_name=driver_name,
			data_type=data_type,
			base_value=base_value,
			growth_rate=growth_rate or Decimal('0.00'),
			unit_of_measure=unit_of_measure
		)
		
		self.db.add(driver)
		self._log_driver_operation("create_driver", f"Created driver {driver_code}")
		return driver
	
	def update_driver_values(
		self,
		driver_id: str,
		base_value: Decimal = None,
		growth_rate: Decimal = None,
		seasonal_factors: Dict[str, float] = None
	) -> bool:
		"""Update driver values"""
		
		driver = self.db.query(CFBFDrivers).filter(
			CFBFDrivers.driver_id == driver_id,
			CFBFDrivers.tenant_id == self.tenant_id
		).first()
		
		if not driver:
			return False
		
		if base_value is not None:
			driver.base_value = base_value
		
		if growth_rate is not None:
			driver.growth_rate = growth_rate
		
		if seasonal_factors is not None:
			driver.seasonal_factors = seasonal_factors
		
		self._log_driver_operation("update_driver", f"Updated driver {driver.driver_code}")
		return True
	
	def calculate_driver_projections(
		self,
		driver_id: str,
		periods: int = 12
	) -> Dict[int, Decimal]:
		"""Calculate driver projections for specified periods"""
		
		driver = self.db.query(CFBFDrivers).filter(
			CFBFDrivers.driver_id == driver_id,
			CFBFDrivers.tenant_id == self.tenant_id
		).first()
		
		if not driver:
			return {}
		
		projections = {}
		for period in range(1, periods + 1):
			projections[period] = driver.calculate_period_value(period)
		
		return projections
	
	def get_driver_impact_analysis(self, driver_id: str) -> Dict[str, Any]:
		"""Analyze the impact of a driver on budgets and forecasts"""
		
		driver = self.db.query(CFBFDrivers).filter(
			CFBFDrivers.driver_id == driver_id,
			CFBFDrivers.tenant_id == self.tenant_id
		).first()
		
		if not driver:
			return {}
		
		# Find budget lines using this driver
		budget_lines = self.db.query(CFBFBudgetLine).filter(
			CFBFBudgetLine.driver_id == driver_id,
			CFBFBudgetLine.tenant_id == self.tenant_id
		).all()
		
		# Find forecast lines using this driver  
		forecast_lines = self.db.query(CFBFForecastLine).filter(
			CFBFForecastLine.driver_id == driver_id,
			CFBFForecastLine.tenant_id == self.tenant_id
		).all()
		
		total_budget_impact = sum(line.total_amount for line in budget_lines)
		total_forecast_impact = sum(line.forecast_amount for line in forecast_lines)
		
		return {
			'driver_info': {
				'driver_id': driver.driver_id,
				'driver_code': driver.driver_code,
				'driver_name': driver.driver_name,
				'base_value': float(driver.base_value or 0),
				'growth_rate': float(driver.growth_rate)
			},
			'budget_impact': {
				'affected_lines': len(budget_lines),
				'total_amount': float(total_budget_impact)
			},
			'forecast_impact': {
				'affected_lines': len(forecast_lines),
				'total_amount': float(total_forecast_impact)
			}
		}