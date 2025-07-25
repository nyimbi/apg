"""
Demand Planning Service

Business logic for demand forecasting, model management, and accuracy tracking.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
import asyncio
import numpy as np
from dataclasses import dataclass

from .models import (
	SCDPForecast, SCDPForecastModel, SCDPDemandHistory, 
	SCDPSeasonalPattern, SCDPForecastAccuracy,
	ForecastCreate, DemandHistoryCreate, ForecastModelCreate
)

@dataclass
class ForecastResult:
	"""Result of demand forecasting operation"""
	success: bool
	forecast_id: str
	message: str
	accuracy_metrics: Dict[str, float] | None = None
	confidence_intervals: Dict[str, float] | None = None

@dataclass
class ModelPerformance:
	"""Model performance metrics"""
	model_id: str
	mape: float
	rmse: float
	mae: float
	accuracy_category: str
	sample_size: int

class SCDemandPlanningService:
	"""Service for demand planning operations"""
	
	def __init__(self, db_session: Session, tenant_id: str, current_user: str):
		self.db = db_session
		self.tenant_id = tenant_id
		self.current_user = current_user
	
	# Forecasting Operations
	async def create_forecast(self, forecast_data: ForecastCreate) -> ForecastResult:
		"""Create a new demand forecast"""
		try:
			# Validate forecast model exists
			model = self._get_forecast_model(forecast_data.forecast_model_id)
			if not model:
				return ForecastResult(
					success=False,
					forecast_id="",
					message="Invalid forecast model ID"
				)
			
			# Create forecast record
			forecast = SCDPForecast(
				tenant_id=self.tenant_id,
				forecast_id=forecast_data.forecast_id,
				product_sku=forecast_data.product_sku,
				location_code=forecast_data.location_code,
				forecast_model_id=forecast_data.forecast_model_id,
				forecast_quantity=forecast_data.forecast_quantity,
				forecast_value=forecast_data.forecast_value,
				confidence_level=forecast_data.confidence_level,
				period_start=forecast_data.period_start,
				period_end=forecast_data.period_end,
				period_type=forecast_data.period_type,
				created_by=self.current_user,
				updated_by=self.current_user
			)
			
			# Calculate statistical measures
			forecast.seasonal_factor = await self._get_seasonal_factor(
				forecast_data.product_sku, 
				forecast_data.period_start
			)
			forecast.trend_factor = await self._calculate_trend_factor(
				forecast_data.product_sku,
				forecast_data.period_start
			)
			
			self.db.add(forecast)
			self.db.commit()
			
			self._log_forecast_created(forecast.id, forecast_data.product_sku)
			
			return ForecastResult(
				success=True,
				forecast_id=forecast.id,
				message="Forecast created successfully",
				accuracy_metrics={
					"seasonal_factor": float(forecast.seasonal_factor or 0),
					"trend_factor": float(forecast.trend_factor or 0)
				}
			)
			
		except Exception as e:
			self.db.rollback()
			return ForecastResult(
				success=False,
				forecast_id="",
				message=f"Failed to create forecast: {str(e)}"
			)
	
	async def generate_forecast_batch(
		self, 
		products: List[str], 
		location_code: str | None = None,
		forecast_horizon_days: int = 90,
		model_id: str | None = None
	) -> List[ForecastResult]:
		"""Generate forecasts for multiple products"""
		results = []
		
		# Get best model if not specified
		if not model_id:
			model_id = await self._get_best_model_for_products(products)
		
		for product_sku in products:
			try:
				# Get historical demand
				history = await self._get_demand_history(
					product_sku, 
					location_code,
					days_back=730  # 2 years of history
				)
				
				if len(history) < 12:  # Need at least 12 data points
					results.append(ForecastResult(
						success=False,
						forecast_id="",
						message=f"Insufficient historical data for {product_sku}"
					))
					continue
				
				# Generate forecast periods
				forecast_periods = self._generate_forecast_periods(
					datetime.now().date(),
					forecast_horizon_days
				)
				
				for period_start, period_end in forecast_periods:
					forecast_data = await self._calculate_forecast_for_period(
						product_sku,
						location_code,
						period_start,
						period_end,
						history,
						model_id
					)
					
					result = await self.create_forecast(forecast_data)
					results.append(result)
					
			except Exception as e:
				results.append(ForecastResult(
					success=False,
					forecast_id="",
					message=f"Failed to generate forecast for {product_sku}: {str(e)}"
				))
		
		return results
	
	# Model Management
	async def create_forecast_model(self, model_data: ForecastModelCreate) -> str:
		"""Create a new forecast model"""
		model = SCDPForecastModel(
			tenant_id=self.tenant_id,
			model_name=model_data.model_name,
			model_code=model_data.model_code,
			model_type=model_data.model_type,
			algorithm=model_data.algorithm,
			parameters=model_data.parameters,
			hyperparameters=model_data.hyperparameters,
			auto_retrain=model_data.auto_retrain,
			retrain_frequency_days=model_data.retrain_frequency_days,
			created_by=self.current_user,
			updated_by=self.current_user
		)
		
		self.db.add(model)
		self.db.commit()
		
		self._log_model_created(model.id, model_data.model_name)
		return model.id
	
	async def train_model(self, model_id: str, training_data: List[Dict]) -> ModelPerformance:
		"""Train a forecast model with historical data"""
		model = self._get_forecast_model(model_id)
		if not model:
			raise ValueError(f"Model {model_id} not found")
		
		try:
			# Prepare training data
			features, targets = self._prepare_training_data(training_data)
			
			# Train model based on type
			if model.model_type == 'arima':
				metrics = await self._train_arima_model(model, features, targets)
			elif model.model_type == 'exponential_smoothing':
				metrics = await self._train_exponential_smoothing_model(model, features, targets)
			elif model.model_type == 'linear_regression':
				metrics = await self._train_linear_regression_model(model, features, targets)
			elif model.model_type == 'ml_ensemble':
				metrics = await self._train_ensemble_model(model, features, targets)
			else:
				raise ValueError(f"Unsupported model type: {model.model_type}")
			
			# Update model with training results
			model.accuracy_mape = Decimal(str(metrics['mape']))
			model.accuracy_rmse = Decimal(str(metrics['rmse']))
			model.accuracy_mae = Decimal(str(metrics['mae']))
			model.training_samples = len(training_data)
			model.last_trained = datetime.utcnow()
			model.status = 'trained'
			model.updated_by = self.current_user
			
			self.db.commit()
			
			self._log_model_trained(model_id, metrics)
			
			return ModelPerformance(
				model_id=model_id,
				mape=metrics['mape'],
				rmse=metrics['rmse'],
				mae=metrics['mae'],
				accuracy_category=self._categorize_accuracy(metrics['mape']),
				sample_size=len(training_data)
			)
			
		except Exception as e:
			self.db.rollback()
			raise Exception(f"Model training failed: {str(e)}")
	
	# Historical Data Management
	async def import_demand_history(self, history_data: List[DemandHistoryCreate]) -> Dict[str, int]:
		"""Import historical demand data"""
		imported_count = 0
		error_count = 0
		
		for data in history_data:
			try:
				# Check for duplicates
				existing = self.db.query(SCDPDemandHistory).filter(
					and_(
						SCDPDemandHistory.tenant_id == self.tenant_id,
						SCDPDemandHistory.product_sku == data.product_sku,
						SCDPDemandHistory.location_code == data.location_code,
						SCDPDemandHistory.demand_date == data.demand_date
					)
				).first()
				
				if existing:
					# Update existing record
					existing.actual_demand = data.actual_demand
					existing.fulfilled_demand = data.fulfilled_demand
					existing.lost_sales = data.lost_sales
					existing.demand_value = data.demand_value
					existing.data_source = data.data_source
					existing.promotion_active = data.promotion_active
					existing.stockout_occurred = data.stockout_occurred
					existing.updated_by = self.current_user
				else:
					# Create new record
					history = SCDPDemandHistory(
						tenant_id=self.tenant_id,
						product_sku=data.product_sku,
						location_code=data.location_code,
						demand_date=data.demand_date,
						actual_demand=data.actual_demand,
						fulfilled_demand=data.fulfilled_demand,
						lost_sales=data.lost_sales,
						demand_value=data.demand_value,
						data_source=data.data_source,
						promotion_active=data.promotion_active,
						stockout_occurred=data.stockout_occurred,
						day_of_week=data.demand_date.weekday() + 1,
						week_of_year=data.demand_date.isocalendar()[1],
						month=data.demand_date.month,
						quarter=(data.demand_date.month - 1) // 3 + 1,
						created_by=self.current_user,
						updated_by=self.current_user
					)
					self.db.add(history)
				
				imported_count += 1
				
			except Exception as e:
				error_count += 1
				self._log_import_error(data.product_sku, str(e))
		
		self.db.commit()
		
		return {
			'imported': imported_count,
			'errors': error_count,
			'total': len(history_data)
		}
	
	# Accuracy Tracking
	async def calculate_forecast_accuracy(
		self, 
		start_date: date, 
		end_date: date,
		product_sku: str | None = None
	) -> List[Dict[str, Any]]:
		"""Calculate forecast accuracy for a period"""
		# Get forecasts for the period
		query = self.db.query(SCDPForecast).filter(
			and_(
				SCDPForecast.tenant_id == self.tenant_id,
				SCDPForecast.period_start >= start_date,
				SCDPForecast.period_end <= end_date,
				SCDPForecast.status == 'published'
			)
		)
		
		if product_sku:
			query = query.filter(SCDPForecast.product_sku == product_sku)
		
		forecasts = query.all()
		accuracy_results = []
		
		for forecast in forecasts:
			# Get actual demand for the forecast period
			actual_demand = await self._get_actual_demand_for_period(
				forecast.product_sku,
				forecast.location_code,
				forecast.period_start,
				forecast.period_end
			)
			
			if actual_demand is not None:
				# Calculate accuracy metrics
				error = abs(float(forecast.forecast_quantity) - actual_demand)
				percentage_error = (error / actual_demand * 100) if actual_demand > 0 else 0
				
				accuracy = SCDPForecastAccuracy(
					tenant_id=self.tenant_id,
					forecast_id=forecast.id,
					product_sku=forecast.product_sku,
					measurement_date=datetime.now().date(),
					forecasted_quantity=forecast.forecast_quantity,
					actual_quantity=Decimal(str(actual_demand)),
					absolute_error=Decimal(str(error)),
					percentage_error=Decimal(str(percentage_error)),
					accuracy_category=self._categorize_accuracy(percentage_error),
					error_type=self._categorize_error_type(
						float(forecast.forecast_quantity), 
						actual_demand
					),
					forecast_horizon_days=(forecast.period_end - forecast.period_start).days,
					model_used=forecast.forecast_model.model_name,
					created_by=self.current_user
				)
				
				self.db.add(accuracy)
				
				accuracy_results.append({
					'forecast_id': forecast.id,
					'product_sku': forecast.product_sku,
					'forecasted': float(forecast.forecast_quantity),
					'actual': actual_demand,
					'error': error,
					'percentage_error': percentage_error,
					'accuracy_category': accuracy.accuracy_category
				})
		
		self.db.commit()
		return accuracy_results
	
	# Analytics and Reporting
	async def get_forecast_analytics(
		self, 
		product_sku: str | None = None,
		days_back: int = 30
	) -> Dict[str, Any]:
		"""Get forecast analytics and insights"""
		start_date = datetime.now().date() - timedelta(days=days_back)
		
		# Base query
		query = self.db.query(SCDPForecast).filter(
			and_(
				SCDPForecast.tenant_id == self.tenant_id,
				SCDPForecast.created_at >= start_date
			)
		)
		
		if product_sku:
			query = query.filter(SCDPForecast.product_sku == product_sku)
		
		forecasts = query.all()
		
		# Calculate analytics
		total_forecasts = len(forecasts)
		total_quantity = sum(float(f.forecast_quantity) for f in forecasts)
		avg_confidence = sum(float(f.confidence_level or 0) for f in forecasts) / total_forecasts if total_forecasts > 0 else 0
		
		# Get model usage statistics
		model_usage = {}
		for forecast in forecasts:
			model_name = forecast.forecast_model.model_name
			if model_name not in model_usage:
				model_usage[model_name] = 0
			model_usage[model_name] += 1
		
		return {
			'period': {
				'start_date': start_date.isoformat(),
				'end_date': datetime.now().date().isoformat(),
				'days': days_back
			},
			'summary': {
				'total_forecasts': total_forecasts,
				'total_quantity_forecasted': total_quantity,
				'average_confidence_level': avg_confidence,
				'unique_products': len(set(f.product_sku for f in forecasts))
			},
			'model_usage': model_usage,
			'top_forecasted_products': await self._get_top_forecasted_products(days_back),
			'accuracy_trends': await self._get_accuracy_trends(days_back)
		}
	
	# Helper Methods
	def _get_forecast_model(self, model_id: str) -> SCDPForecastModel | None:
		"""Get forecast model by ID"""
		return self.db.query(SCDPForecastModel).filter(
			and_(
				SCDPForecastModel.tenant_id == self.tenant_id,
				SCDPForecastModel.id == model_id,
				SCDPForecastModel.is_active == True
			)
		).first()
	
	async def _get_seasonal_factor(self, product_sku: str, forecast_date: date) -> Decimal:
		"""Get seasonal factor for product and date"""
		# Determine season based on date
		month = forecast_date.month
		quarter = (month - 1) // 3 + 1
		
		# Look for quarterly seasonal pattern first
		pattern = self.db.query(SCDPSeasonalPattern).filter(
			and_(
				SCDPSeasonalPattern.tenant_id == self.tenant_id,
				SCDPSeasonalPattern.product_sku == product_sku,
				SCDPSeasonalPattern.season_type == 'quarterly',
				SCDPSeasonalPattern.season_period == f'Q{quarter}',
				SCDPSeasonalPattern.is_active == True
			)
		).first()
		
		if pattern:
			return pattern.seasonal_factor
		
		# Look for monthly pattern
		pattern = self.db.query(SCDPSeasonalPattern).filter(
			and_(
				SCDPSeasonalPattern.tenant_id == self.tenant_id,
				SCDPSeasonalPattern.product_sku == product_sku,
				SCDPSeasonalPattern.season_type == 'monthly',
				SCDPSeasonalPattern.season_period == forecast_date.strftime('%B'),
				SCDPSeasonalPattern.is_active == True
			)
		).first()
		
		return pattern.seasonal_factor if pattern else Decimal('1.0')
	
	async def _calculate_trend_factor(self, product_sku: str, forecast_date: date) -> Decimal:
		"""Calculate trend factor based on historical data"""
		# Get last 6 months of demand history
		start_date = forecast_date - timedelta(days=180)
		
		history = self.db.query(SCDPDemandHistory).filter(
			and_(
				SCDPDemandHistory.tenant_id == self.tenant_id,
				SCDPDemandHistory.product_sku == product_sku,
				SCDPDemandHistory.demand_date >= start_date,
				SCDPDemandHistory.demand_date < forecast_date
			)
		).order_by(SCDPDemandHistory.demand_date).all()
		
		if len(history) < 8:  # Need at least 8 data points
			return Decimal('1.0')
		
		# Calculate simple linear trend
		demands = [float(h.actual_demand) for h in history]
		periods = list(range(len(demands)))
		
		# Simple linear regression for trend
		n = len(demands)
		sum_x = sum(periods)
		sum_y = sum(demands)
		sum_xy = sum(x * y for x, y in zip(periods, demands))
		sum_x2 = sum(x * x for x in periods)
		
		if n * sum_x2 - sum_x * sum_x == 0:
			return Decimal('1.0')
		
		slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
		
		# Convert slope to trend factor
		avg_demand = sum_y / n
		if avg_demand == 0:
			return Decimal('1.0')
		
		trend_factor = 1.0 + (slope / avg_demand)
		return Decimal(str(max(0.1, min(3.0, trend_factor))))  # Clamp between 0.1 and 3.0
	
	def _categorize_accuracy(self, mape: float) -> str:
		"""Categorize forecast accuracy based on MAPE"""
		if mape <= 10:
			return 'excellent'
		elif mape <= 20:
			return 'good'
		elif mape <= 50:
			return 'fair'
		else:
			return 'poor'
	
	def _categorize_error_type(self, forecasted: float, actual: float) -> str:
		"""Categorize error type based on forecast vs actual"""
		error_threshold = 0.05  # 5% threshold
		
		if abs(forecasted - actual) / max(actual, 1) <= error_threshold:
			return 'accurate'
		elif forecasted > actual:
			return 'over_forecast'
		else:
			return 'under_forecast'
	
	# Logging Methods
	def _log_forecast_created(self, forecast_id: str, product_sku: str):
		"""Log forecast creation"""
		# Implementation would depend on your logging system
		print(f"Forecast created: {forecast_id} for product {product_sku}")
	
	def _log_model_created(self, model_id: str, model_name: str):
		"""Log model creation"""
		print(f"Forecast model created: {model_id} - {model_name}")
	
	def _log_model_trained(self, model_id: str, metrics: Dict):
		"""Log model training completion"""
		print(f"Model {model_id} trained with MAPE: {metrics['mape']:.2f}%")
	
	def _log_import_error(self, product_sku: str, error: str):
		"""Log import error"""
		print(f"Import error for {product_sku}: {error}")

# Async helper functions for model training (simplified implementations)
async def _train_arima_model(model, features, targets) -> Dict[str, float]:
	"""Train ARIMA model (simplified)"""
	# This would use actual ARIMA implementation (statsmodels, etc.)
	return {'mape': 15.5, 'rmse': 123.45, 'mae': 98.76}

async def _train_exponential_smoothing_model(model, features, targets) -> Dict[str, float]:
	"""Train Exponential Smoothing model (simplified)"""
	return {'mape': 18.2, 'rmse': 134.56, 'mae': 105.43}

async def _train_linear_regression_model(model, features, targets) -> Dict[str, float]:
	"""Train Linear Regression model (simplified)"""
	return {'mape': 22.1, 'rmse': 156.78, 'mae': 125.67}

async def _train_ensemble_model(model, features, targets) -> Dict[str, float]:
	"""Train Ensemble model (simplified)"""
	return {'mape': 12.8, 'rmse': 98.34, 'mae': 78.92}