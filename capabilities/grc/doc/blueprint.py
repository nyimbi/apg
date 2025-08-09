"""
APG Document Service Flask-AppBuilder Blueprint

Comprehensive Flask-AppBuilder integration for document service with proper
view registration, URL routing, and permission configuration following APG patterns.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import logging
from flask import Blueprint
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.decorators import has_access

from .views import (
	DocumentServiceApi,
	DocumentServiceView, 
	DocumentModelView,
	DocumentTemplateModelView
)

logger = logging.getLogger(__name__)

# Create blueprint for additional routes if needed
document_service_bp = Blueprint(
	'document_service',
	__name__,
	url_prefix='/api/v1/document_service',
	template_folder='templates',
	static_folder='static'
)


def init_document_service(appbuilder: AppBuilder):
	"""
	Initialize APG Document Service with Flask-AppBuilder
	
	Registers all views, APIs, and configures proper permissions following
	APG integration patterns and Flask-AppBuilder best practices.
	"""
	logger.info("Initializing APG Document Service with Flask-AppBuilder")
	
	try:
		# Register REST API
		appbuilder.add_api(DocumentServiceApi)
		logger.info("Registered DocumentServiceApi REST endpoints")
		
		# Register main dashboard view
		appbuilder.add_view(
			DocumentServiceView,
			"Dashboard",
			icon="fa-dashboard",
			category="Document Service",
			category_icon="fa-files-o"
		)
		logger.info("Registered DocumentServiceView dashboard")
		
		# Register document management views
		appbuilder.add_view(
			DocumentModelView,
			"Documents",
			icon="fa-file-text",
			category="Document Service",
			category_icon="fa-files-o"
		)
		logger.info("Registered DocumentModelView for document management")
		
		# Register template management views
		appbuilder.add_view(
			DocumentTemplateModelView,
			"Templates",
			icon="fa-file-code-o",
			category="Document Service", 
			category_icon="fa-files-o"
		)
		logger.info("Registered DocumentTemplateModelView for template management")
		
		# Add specific view endpoints to dashboard
		appbuilder.add_link(
			"Analytics",
			href="/document_service/DocumentServiceView/analytics/",
			icon="fa-line-chart",
			category="Document Service"
		)
		
		appbuilder.add_link(
			"Metrics",
			href="/document_service/DocumentServiceView/metrics/",
			icon="fa-bar-chart",
			category="Document Service"
		)
		
		# Register blueprint for additional routes
		appbuilder.get_app.register_blueprint(document_service_bp)
		
		# Configure menu ordering
		appbuilder.add_separator("Document Service")
		
		logger.info("APG Document Service successfully initialized with Flask-AppBuilder")
		return document_service_bp
		
	except Exception as e:
		logger.error(f"Failed to initialize APG Document Service: {e}")
		raise


def create_admin_user_if_not_exists(appbuilder: AppBuilder):
	"""Create admin user for document service if it doesn't exist"""
	try:
		from flask_appbuilder.security.sqla.models import User, Role
		
		# Check if document admin role exists
		admin_role = appbuilder.sm.find_role("DocumentAdmin")
		if not admin_role:
			admin_role = appbuilder.sm.add_role("DocumentAdmin")
			logger.info("Created DocumentAdmin role")
		
		# Add permissions to admin role
		permissions = [
			("can_list", "DocumentModelView"),
			("can_show", "DocumentModelView"),
			("can_add", "DocumentModelView"),
			("can_edit", "DocumentModelView"),
			("can_delete", "DocumentModelView"),
			("can_list", "DocumentTemplateModelView"),
			("can_show", "DocumentTemplateModelView"),
			("can_add", "DocumentTemplateModelView"),
			("can_edit", "DocumentTemplateModelView"),
			("can_delete", "DocumentTemplateModelView"),
			("read", "DocumentServiceView"),
		]
		
		for permission_name, view_name in permissions:
			perm = appbuilder.sm.find_permission_view_menu(permission_name, view_name)
			if perm and perm not in admin_role.permissions:
				admin_role.permissions.append(perm)
				
		appbuilder.sm.get_session.commit()
		logger.info("Configured DocumentAdmin role permissions")
		
	except Exception as e:
		logger.error(f"Failed to create admin user configuration: {e}")


# API Blueprint routes (if additional custom routes are needed beyond Flask-AppBuilder)
@document_service_bp.route('/health')
def health_check():
	"""Health check endpoint for document service"""
	from flask import jsonify
	return jsonify({
		"status": "healthy",
		"service": "apg_document_service",
		"version": "1.0.0"
	})


@document_service_bp.route('/capabilities')
def capabilities():
	"""Document service capabilities endpoint"""
	from flask import jsonify
	return jsonify({
		"capabilities": [
			"document_creation",
			"document_management", 
			"template_management",
			"ai_processing",
			"search_and_analytics",
			"collaboration",
			"workflow_automation",
			"apg_composition_integration"
		],
		"api_version": "v1",
		"apg_integration": True
	})


# Export initialization function
__all__ = ['init_document_service', 'document_service_bp', 'create_admin_user_if_not_exists']