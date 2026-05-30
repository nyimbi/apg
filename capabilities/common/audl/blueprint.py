"""
APG Audit Logging Blueprint

Flask-AppBuilder blueprint registration for governed audit logging UI.
Integrates all views into APG platform with proper authentication and routing.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder, SQLA

# Import all audit logging views
from .views import (
	AuditDashboardView,
	AuditSearchView,
	ComplianceReportView,
	AuditInvestigationView,
	AuditSettingsView
)

# Import API endpoints
from .api import app as audit_api_app

def register_audit_views(appbuilder: AppBuilder) -> None:
	"""
	Register all audit logging views with Flask-AppBuilder
	
	This function integrates the governed audit logging capability
	into the APG platform with proper authentication, authorization,
	and UI patterns.
	
	Args:
		appbuilder: Flask-AppBuilder instance from APG platform
	"""
	
	# Register main audit views
	appbuilder.add_view_no_menu(AuditDashboardView)
	appbuilder.add_view_no_menu(AuditSearchView) 
	appbuilder.add_view_no_menu(ComplianceReportView)
	appbuilder.add_view_no_menu(AuditInvestigationView)
	appbuilder.add_view_no_menu(AuditSettingsView)
	
	# Create audit menu category
	appbuilder.add_link(
		"Audit Dashboard",
		href="/audit/",
		icon="fas fa-chart-line",
		category="APG Audit Intelligence",
		category_icon="fas fa-shield-alt"
	)
	
	appbuilder.add_link(
		"Search & Analytics", 
		href="/audit/search",
		icon="fas fa-search",
		category="APG Audit Intelligence"
	)
	
	appbuilder.add_link(
		"Compliance Reports",
		href="/audit/compliance", 
		icon="fas fa-balance-scale",
		category="APG Audit Intelligence"
	)
	
	appbuilder.add_link(
		"Investigations",
		href="/audit/investigate",
		icon="fas fa-search-plus", 
		category="APG Audit Intelligence"
	)
	
	appbuilder.add_link(
		"Settings",
		href="/audit/settings",
		icon="fas fa-cog",
		category="APG Audit Intelligence"
	)

def register_audit_api(app) -> None:
	"""
	Register audit logging API endpoints
	
	Mounts the FastAPI application for high-performance audit operations
	including real-time streaming, ML-powered analytics, and natural
	language queries.
	
	Args:
		app: Flask application instance
	"""
	
	# Mount FastAPI app for audit API endpoints
	from werkzeug.middleware.dispatcher import DispatcherMiddleware
	
	# Create dispatcher to handle both Flask and FastAPI
	app.wsgi_app = DispatcherMiddleware(
		app.wsgi_app,
		{'/audit/api': audit_api_app}
	)

# Blueprint for standalone usage
audit_blueprint = Blueprint(
	'audit',
	__name__,
	url_prefix='/audit',
	template_folder='templates',
	static_folder='static'
)

# Export for APG integration
__all__ = [
	"register_audit_views",
	"register_audit_api", 
	"audit_blueprint"
]