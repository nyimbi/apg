"""
Sales Forecasting Views

Flask-AppBuilder views for sales forecasting and analytics.
"""

from flask_appbuilder import ModelView
from flask_appbuilder.models.sqla.interface import SQLAInterface

from .models import (
	SOFForecast, SOFForecastModel, SOFHistoricalData, SOFSeasonalPattern
)


class SOFForecastView(ModelView):
	"""Sales forecast management view"""
	
	datamodel = SQLAInterface(SOFForecast)
	
	list_columns = [
		'forecast_name', 'period_start', 'period_end', 
		'predicted_revenue', 'actual_revenue', 'revenue_accuracy', 'status'
	]
	search_columns = ['forecast_name', 'scope_name']
	list_filters = ['status', 'period_type', 'scope_type']
	
	formatters_columns = {
		'predicted_revenue': lambda x: f"${x:,.2f}" if x else "$0.00",
		'actual_revenue': lambda x: f"${x:,.2f}" if x else "$0.00",
		'revenue_accuracy': lambda x: f"{x:.1f}%" if x else "N/A"
	}
	
	base_order = ('period_start', 'desc')


class SOFForecastModelView(ModelView):
	"""Forecast model management view"""
	
	datamodel = SQLAInterface(SOFForecastModel)
	
	list_columns = [
		'model_name', 'model_type', 'accuracy_percentage',
		'forecast_horizon_days', 'is_active', 'is_default'
	]
	search_columns = ['model_name', 'description']
	list_filters = ['model_type', 'is_active', 'is_default']
	
	formatters_columns = {
		'accuracy_percentage': lambda x: f"{x:.1f}%" if x else "0.0%"
	}
	
	base_order = ('model_name', 'asc')


class SOFHistoricalDataView(ModelView):
	"""Historical data management view"""
	
	datamodel = SQLAInterface(SOFHistoricalData)
	
	list_columns = [
		'data_date', 'period_type', 'scope_name', 
		'total_orders', 'total_revenue', 'total_customers'
	]
	search_columns = ['scope_name']
	list_filters = ['period_type', 'scope_type', 'data_date']
	
	formatters_columns = {
		'total_revenue': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	base_order = ('data_date', 'desc')


class SOFSeasonalPatternView(ModelView):
	"""Seasonal pattern management view"""
	
	datamodel = SQLAInterface(SOFSeasonalPattern)
	
	list_columns = [
		'pattern_name', 'pattern_type', 'scope_name',
		'confidence_level', 'years_of_data', 'is_active'
	]
	search_columns = ['pattern_name', 'scope_name']
	list_filters = ['pattern_type', 'scope_type', 'is_active']
	base_order = ('pattern_name', 'asc')


class ForecastingDashboardView(ModelView):
	"""Forecasting dashboard and analytics view"""
	
	datamodel = SQLAInterface(SOFForecast)
	
	# This would be a custom dashboard view with charts and analytics