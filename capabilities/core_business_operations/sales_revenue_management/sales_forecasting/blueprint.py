"""
Sales Forecasting Blueprint Registration

Flask blueprint registration for Sales Forecasting sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder

# Create blueprint
sales_forecasting_bp = Blueprint(
	'sales_forecasting',
	__name__,
	url_prefix='/sales_order_management/sales_forecasting'
)

def init_subcapability(appbuilder: AppBuilder):
	"""Initialize Sales Forecasting sub-capability with Flask-AppBuilder"""
	
	# Import views (would be implemented)
	from .views import (
		SOFForecastView, SOFForecastModelView, SOFHistoricalDataView,
		SOFSeasonalPatternView, ForecastingDashboardView
	)
	
	# Register views with AppBuilder
	appbuilder.add_view(
		SOFForecastView,
		"Forecasts",
		icon="fa-line-chart",
		category="Sales Forecasting",
		category_icon="fa-chart-line"
	)
	
	appbuilder.add_view(
		SOFForecastModelView,
		"Forecast Models",
		icon="fa-brain",
		category="Sales Forecasting"
	)
	
	appbuilder.add_view(
		SOFHistoricalDataView,
		"Historical Data",
		icon="fa-database",
		category="Sales Forecasting"
	)
	
	appbuilder.add_view(
		SOFSeasonalPatternView,
		"Seasonal Patterns",
		icon="fa-calendar",
		category="Sales Forecasting"
	)
	
	appbuilder.add_view(
		ForecastingDashboardView,
		"Forecasting Dashboard",
		icon="fa-dashboard",
		category="Sales Forecasting"
	)
	
	# Register blueprint
	appbuilder.get_app.register_blueprint(sales_forecasting_bp)