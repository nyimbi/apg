"""
Demand Planning Views

Flask-AppBuilder views for demand forecasting, model management, and analytics.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import GroupByChartView
from wtforms import Form, StringField, SelectField, DecimalField, DateField, BooleanField
from wtforms.validators import DataRequired, NumberRange

from .models import (
	SCDPForecast, SCDPForecastModel, SCDPDemandHistory, 
	SCDPSeasonalPattern, SCDPForecastAccuracy
)
from .service import SCDemandPlanningService

class SCDPForecastModelView(ModelView):
	"""View for managing demand forecasts"""
	datamodel = SQLAInterface(SCDPForecast)
	
	# List view configuration
	list_columns = [
		'forecast_id', 'product_sku', 'location_code', 'forecast_date',
		'forecast_quantity', 'forecast_value', 'confidence_level',
		'period_type', 'status', 'created_at'
	]
	
	search_columns = [
		'forecast_id', 'product_sku', 'location_code', 'status'
	]
	
	# Show/Edit view configuration
	show_columns = [
		'forecast_id', 'product_sku', 'location_code', 'forecast_date',
		'forecast_model', 'forecast_quantity', 'forecast_value', 'confidence_level',
		'period_start', 'period_end', 'period_type',
		'forecast_error', 'forecast_bias', 'seasonal_factor', 'trend_factor',
		'status', 'version', 'is_baseline',
		'created_at', 'created_by', 'updated_at', 'updated_by'
	]
	
	edit_columns = [
		'forecast_id', 'product_sku', 'location_code', 'forecast_date',
		'forecast_model', 'forecast_quantity', 'forecast_value', 'confidence_level',
		'period_start', 'period_end', 'period_type', 'status', 'is_baseline'
	]
	
	add_columns = edit_columns
	
	# Formatting
	formatters_columns = {
		'forecast_quantity': lambda x: f"{x:,.2f}" if x else "",
		'forecast_value': lambda x: f"${x:,.2f}" if x else "",
		'confidence_level': lambda x: f"{x:.1%}" if x else "",
		'seasonal_factor': lambda x: f"{x:.3f}" if x else "",
		'trend_factor': lambda x: f"{x:.3f}" if x else ""
	}
	
	# Labels
	label_columns = {
		'forecast_id': 'Forecast ID',
		'product_sku': 'Product SKU',
		'location_code': 'Location',
		'forecast_date': 'Forecast Date',
		'forecast_model': 'Forecast Model',
		'forecast_quantity': 'Forecasted Quantity',
		'forecast_value': 'Forecasted Value',
		'confidence_level': 'Confidence Level',
		'period_start': 'Period Start',
		'period_end': 'Period End',
		'period_type': 'Period Type',
		'seasonal_factor': 'Seasonal Factor',
		'trend_factor': 'Trend Factor',
		'is_baseline': 'Baseline Forecast'
	}
	
	# Default ordering
	base_order = ('forecast_date', 'desc')
	
	# Permissions
	base_permissions = ['can_list', 'can_show', 'can_add', 'can_edit', 'can_delete']

class SCDPForecastModelModelView(ModelView):
	"""View for managing forecast models"""
	datamodel = SQLAInterface(SCDPForecastModel)
	
	# List view configuration
	list_columns = [
		'model_name', 'model_code', 'model_type', 'algorithm',
		'accuracy_mape', 'accuracy_rmse', 'status', 'is_active',
		'last_trained', 'created_at'
	]
	
	search_columns = [
		'model_name', 'model_code', 'model_type', 'algorithm', 'status'
	]
	
	# Show/Edit view configuration
	show_columns = [
		'model_name', 'model_code', 'model_type', 'algorithm',
		'parameters', 'hyperparameters',
		'accuracy_mape', 'accuracy_rmse', 'accuracy_mae',
		'training_data_start', 'training_data_end', 'training_samples',
		'last_trained', 'status', 'is_active',
		'auto_retrain', 'retrain_frequency_days',
		'created_at', 'created_by', 'updated_at', 'updated_by'
	]
	
	edit_columns = [
		'model_name', 'model_code', 'model_type', 'algorithm',
		'parameters', 'hyperparameters', 'status', 'is_active',
		'auto_retrain', 'retrain_frequency_days'
	]
	
	add_columns = [
		'model_name', 'model_code', 'model_type', 'algorithm',
		'parameters', 'hyperparameters', 'auto_retrain', 'retrain_frequency_days'
	]
	
	# Formatting
	formatters_columns = {
		'accuracy_mape': lambda x: f"{x:.2%}" if x else "",
		'accuracy_rmse': lambda x: f"{x:,.2f}" if x else "",
		'accuracy_mae': lambda x: f"{x:,.2f}" if x else "",
		'last_trained': lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "Never"
	}
	
	# Labels
	label_columns = {
		'model_name': 'Model Name',
		'model_code': 'Model Code',
		'model_type': 'Model Type',
		'accuracy_mape': 'MAPE',
		'accuracy_rmse': 'RMSE',
		'accuracy_mae': 'MAE',
		'training_data_start': 'Training Start',
		'training_data_end': 'Training End',
		'training_samples': 'Training Samples',
		'last_trained': 'Last Trained',
		'is_active': 'Active',
		'auto_retrain': 'Auto Retrain',
		'retrain_frequency_days': 'Retrain Frequency (Days)'
	}
	
	# Default ordering
	base_order = ('last_trained', 'desc')

class SCDPDemandHistoryModelView(ModelView):
	"""View for managing historical demand data"""
	datamodel = SQLAInterface(SCDPDemandHistory)
	
	# List view configuration
	list_columns = [
		'product_sku', 'location_code', 'demand_date',
		'actual_demand', 'fulfilled_demand', 'lost_sales',
		'data_source', 'promotion_active', 'stockout_occurred'
	]
	
	search_columns = [
		'product_sku', 'location_code', 'data_source'
	]
	
	# Show/Edit view configuration  
	show_columns = [
		'product_sku', 'location_code', 'demand_date',
		'actual_demand', 'fulfilled_demand', 'lost_sales', 'demand_value',
		'day_of_week', 'week_of_year', 'month', 'quarter',
		'promotion_active', 'stockout_occurred', 'weather_impact', 'holiday_impact',
		'data_source', 'data_quality_score', 'is_outlier', 'outlier_reason',
		'created_at', 'created_by', 'updated_at', 'updated_by'
	]
	
	edit_columns = [
		'product_sku', 'location_code', 'demand_date',
		'actual_demand', 'fulfilled_demand', 'lost_sales', 'demand_value',
		'promotion_active', 'stockout_occurred', 'weather_impact', 'holiday_impact',
		'data_source', 'data_quality_score', 'is_outlier', 'outlier_reason'
	]
	
	add_columns = edit_columns
	
	# Formatting
	formatters_columns = {
		'actual_demand': lambda x: f"{x:,.2f}" if x else "",
		'fulfilled_demand': lambda x: f"{x:,.2f}" if x else "",
		'lost_sales': lambda x: f"{x:,.2f}" if x else "",
		'demand_value': lambda x: f"${x:,.2f}" if x else "",
		'data_quality_score': lambda x: f"{x:.2f}" if x else ""
	}
	
	# Labels
	label_columns = {
		'product_sku': 'Product SKU',
		'location_code': 'Location',
		'demand_date': 'Date',
		'actual_demand': 'Actual Demand',
		'fulfilled_demand': 'Fulfilled Demand',
		'lost_sales': 'Lost Sales',
		'demand_value': 'Demand Value',
		'day_of_week': 'Day of Week',
		'week_of_year': 'Week of Year',
		'promotion_active': 'Promotion Active',
		'stockout_occurred': 'Stockout Occurred',
		'weather_impact': 'Weather Impact',
		'holiday_impact': 'Holiday Impact',
		'data_source': 'Data Source',
		'data_quality_score': 'Data Quality Score',
		'is_outlier': 'Is Outlier',
		'outlier_reason': 'Outlier Reason'
	}
	
	# Default ordering
	base_order = ('demand_date', 'desc')

class SCDPSeasonalPatternModelView(ModelView):
	"""View for managing seasonal patterns"""
	datamodel = SQLAInterface(SCDPSeasonalPattern)
	
	# List view configuration
	list_columns = [
		'pattern_name', 'product_sku', 'location_code',
		'season_type', 'season_period', 'seasonal_factor',
		'confidence_interval', 'is_active'
	]
	
	search_columns = [
		'pattern_name', 'product_sku', 'location_code', 'season_type'
	]
	
	# Formatting
	formatters_columns = {
		'seasonal_factor': lambda x: f"{x:.3f}" if x else "",
		'confidence_interval': lambda x: f"{x:.3f}" if x else ""
	}
	
	# Labels
	label_columns = {
		'pattern_name': 'Pattern Name',
		'product_sku': 'Product SKU',
		'location_code': 'Location',
		'season_type': 'Season Type',
		'season_period': 'Season Period',
		'seasonal_factor': 'Seasonal Factor',
		'confidence_interval': 'Confidence Interval',
		'sample_size': 'Sample Size',
		'last_calculated': 'Last Calculated',
		'valid_from': 'Valid From',
		'valid_to': 'Valid To',
		'is_active': 'Active'
	}

class SCDPForecastAccuracyView(BaseView):
	"""View for forecast accuracy analysis"""
	
	@expose('/')
	@has_access
	def index(self):
		"""Forecast accuracy dashboard"""
		# Get query parameters
		product_sku = request.args.get('product_sku')
		days_back = int(request.args.get('days_back', 30))
		
		# Get accuracy data
		service = SCDemandPlanningService(
			self.appbuilder.get_session,
			self._get_tenant_id(),
			self._get_current_user()
		)
		
		accuracy_data = service.calculate_forecast_accuracy(
			datetime.now().date() - timedelta(days=days_back),
			datetime.now().date(),
			product_sku
		)
		
		# Calculate summary statistics
		if accuracy_data:
			avg_mape = sum(d['percentage_error'] for d in accuracy_data) / len(accuracy_data)
			accuracy_categories = {}
			for d in accuracy_data:
				cat = d['accuracy_category']
				accuracy_categories[cat] = accuracy_categories.get(cat, 0) + 1
		else:
			avg_mape = 0
			accuracy_categories = {}
		
		return self.render_template(
			'demand_planning/forecast_accuracy.html',
			accuracy_data=accuracy_data,
			avg_mape=avg_mape,
			accuracy_categories=accuracy_categories,
			product_sku=product_sku,
			days_back=days_back
		)

class SCDPDashboardView(BaseView):
	"""Main dashboard for demand planning"""
	
	@expose('/')
	@has_access
	def index(self):
		"""Main demand planning dashboard"""
		service = SCDemandPlanningService(
			self.appbuilder.get_session,
			self._get_tenant_id(),
			self._get_current_user()
		)
		
		# Get dashboard analytics
		analytics = service.get_forecast_analytics(days_back=30)
		
		# Get recent forecasts
		recent_forecasts = self.appbuilder.get_session.query(SCDPForecast)\
			.filter(SCDPForecast.tenant_id == self._get_tenant_id())\
			.order_by(SCDPForecast.created_at.desc())\
			.limit(10).all()
		
		# Get model performance
		models = self.appbuilder.get_session.query(SCDPForecastModel)\
			.filter(SCDPForecastModel.tenant_id == self._get_tenant_id())\
			.filter(SCDPForecastModel.is_active == True)\
			.order_by(SCDPForecastModel.accuracy_mape.asc())\
			.limit(5).all()
		
		return self.render_template(
			'demand_planning/dashboard.html',
			analytics=analytics,
			recent_forecasts=recent_forecasts,
			top_models=models
		)
	
	@expose('/forecast-wizard/')
	@has_access
	def forecast_wizard(self):
		"""Forecast generation wizard"""
		return self.render_template('demand_planning/forecast_wizard.html')
	
	@expose('/model-training/')
	@has_access
	def model_training(self):
		"""Model training interface"""
		models = self.appbuilder.get_session.query(SCDPForecastModel)\
			.filter(SCDPForecastModel.tenant_id == self._get_tenant_id())\
			.all()
		
		return self.render_template(
			'demand_planning/model_training.html',
			models=models
		)
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID from session"""
		# Implementation depends on your multi-tenancy setup
		return "default_tenant"
	
	def _get_current_user(self) -> str:
		"""Get current user"""
		return self.appbuilder.sm.user.username if self.appbuilder.sm.user else "system"

# Chart views for analytics
class SCDPForecastAccuracyChartView(GroupByChartView):
	"""Chart view for forecast accuracy trends"""
	datamodel = SQLAInterface(SCDPForecastAccuracy)
	chart_title = 'Forecast Accuracy Trends'
	
	definitions = [
		{
			'group': 'measurement_date',
			'series': ['percentage_error']
		},
		{
			'group': 'accuracy_category', 
			'series': ['percentage_error']
		}
	]

class SCDPDemandTrendsChartView(GroupByChartView):
	"""Chart view for demand trends"""
	datamodel = SQLAInterface(SCDPDemandHistory)
	chart_title = 'Demand Trends'
	
	definitions = [
		{
			'group': 'demand_date',
			'series': ['actual_demand', 'fulfilled_demand']
		},
		{
			'group': 'month',
			'series': ['actual_demand']
		}
	]