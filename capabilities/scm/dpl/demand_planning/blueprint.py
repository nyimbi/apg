"""
Demand Planning Blueprint

Flask blueprint registration for demand planning sub-capability.
"""

from flask import Blueprint
from flask_appbuilder import BaseView

from .views import (
	SCDPForecastModelView, SCDPForecastModelModelView, SCDPDemandHistoryModelView,
	SCDPSeasonalPatternModelView, SCDPForecastAccuracyView, SCDPDashboardView,
	SCDPForecastAccuracyChartView, SCDPDemandTrendsChartView
)
from .api import register_api

def create_blueprint():
	"""Create demand planning blueprint"""
	return Blueprint(
		'demand_planning',
		__name__,
		url_prefix='/supply_chain/demand_planning',
		template_folder='templates',
		static_folder='static'
	)

def init_views(appbuilder):
	"""Initialize views with Flask-AppBuilder"""
	
	# Model views
	appbuilder.add_view(
		SCDPForecastModelView,
		"Demand Forecasts",
		icon="fa-chart-line",
		category="Supply Chain",
		category_icon="fa-truck"
	)
	
	appbuilder.add_view(
		SCDPForecastModelModelView,
		"Forecast Models", 
		icon="fa-robot",
		category="Supply Chain"
	)
	
	appbuilder.add_view(
		SCDPDemandHistoryModelView,
		"Demand History",
		icon="fa-history", 
		category="Supply Chain"
	)
	
	appbuilder.add_view(
		SCDPSeasonalPatternModelView,
		"Seasonal Patterns",
		icon="fa-calendar-alt",
		category="Supply Chain"
	)
	
	# Dashboard and analysis views
	appbuilder.add_view(
		SCDPDashboardView,
		"Demand Planning Dashboard",
		icon="fa-dashboard",
		category="Supply Chain"
	)
	
	appbuilder.add_view(
		SCDPForecastAccuracyView,
		"Forecast Accuracy",
		icon="fa-bullseye",
		category="Supply Chain"
	)
	
	# Chart views
	appbuilder.add_view(
		SCDPForecastAccuracyChartView,
		"Accuracy Charts",
		icon="fa-chart-bar",
		category="Supply Chain"
	)
	
	appbuilder.add_view(
		SCDPDemandTrendsChartView,
		"Demand Trends Charts", 
		icon="fa-chart-area",
		category="Supply Chain"
	)

def register_blueprints(app):
	"""Register blueprints with Flask app"""
	# Register main blueprint
	blueprint = create_blueprint()
	app.register_blueprint(blueprint)
	
	# Register API blueprint
	register_api(app)