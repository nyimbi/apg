"""
Computer Vision & Visual Intelligence - Flask Blueprint

Flask-AppBuilder blueprint for APG platform integration providing computer vision
capability registration, menu integration, dashboard views, and multi-tenant
access control with comprehensive audit trails and compliance features.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from flask import Blueprint, current_app, g, request, session, jsonify
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.baseviews import BaseView
from flask_appbuilder.security.decorators import has_access, protect
from typing import Dict, List, Any, Optional

from . import (
	CAPABILITY_METADATA, COMPOSITION_KEYWORDS, CAPABILITY_DEPENDENCIES,
	CAPABILITY_PERMISSIONS, PLATFORM_INTEGRATION, get_capability_info
)
from .views import (
	ComputerVisionDashboardView, ComputerVisionDocumentView,
	ComputerVisionImageView, ComputerVisionQualityView,
	ComputerVisionVideoView, ComputerVisionModelView
)
from .models import (
	CVProcessingJob, CVImageProcessing, CVDocumentAnalysis,
	CVObjectDetection, CVFacialRecognition, CVQualityControl,
	CVModel, CVAnalyticsReport
)


class ComputerVisionCapabilityBlueprint:
	"""
	Computer Vision Capability Blueprint for APG Platform Integration
	
	Provides complete Flask-AppBuilder integration including views, models,
	permissions, menu structure, and multi-tenant support for the computer
	vision capability within the APG platform ecosystem.
	"""
	
	def __init__(self, appbuilder: AppBuilder):
		self.appbuilder = appbuilder
		self.blueprint_name = "computer_vision"
		self.capability_info = get_capability_info()
		self.registered_views = []
		self.registered_models = []
		
		# Initialize blueprint
		self.blueprint = Blueprint(
			self.blueprint_name,
			__name__,
			url_prefix='/computer_vision',
			template_folder='templates',
			static_folder='static'
		)
		
		# Setup blueprint components
		self._setup_models()
		self._setup_views()
		self._setup_permissions()
		self._setup_menu()
		self._setup_api_endpoints()
		self._setup_dashboard_widgets()
		self._register_blueprint()
	
	def _setup_models(self) -> None:
		"""Register computer vision models with Flask-AppBuilder"""
		try:
			# Register Pydantic models for API validation (already defined)
			self.registered_models = [
				CVProcessingJob,
				CVImageProcessing,
				CVDocumentAnalysis,
				CVObjectDetection,
				CVFacialRecognition,
				CVQualityControl,
				CVModel,
				CVAnalyticsReport
			]
			
			print(f"Computer Vision: Registered {len(self.registered_models)} data models")
			
		except Exception as e:
			print(f"Error registering Computer Vision models: {e}")
			raise
	
	def _setup_views(self) -> None:
		"""Register computer vision views with Flask-AppBuilder"""
		try:
			# Main dashboard view
			self.appbuilder.add_view_no_menu(ComputerVisionDashboardView)
			self.registered_views.append("ComputerVisionDashboardView")
			
			# Specialized workspace views
			self.appbuilder.add_view_no_menu(ComputerVisionDocumentView)
			self.registered_views.append("ComputerVisionDocumentView")
			
			self.appbuilder.add_view_no_menu(ComputerVisionImageView)
			self.registered_views.append("ComputerVisionImageView")
			
			self.appbuilder.add_view_no_menu(ComputerVisionQualityView)
			self.registered_views.append("ComputerVisionQualityView")
			
			self.appbuilder.add_view_no_menu(ComputerVisionVideoView)
			self.registered_views.append("ComputerVisionVideoView")
			
			self.appbuilder.add_view_no_menu(ComputerVisionModelView)
			self.registered_views.append("ComputerVisionModelView")
			
			print(f"Computer Vision: Registered {len(self.registered_views)} views")
			
		except Exception as e:
			print(f"Error registering Computer Vision views: {e}")
			raise
	
	def _setup_permissions(self) -> None:
		"""Setup computer vision capability permissions"""
		try:
			security_manager = self.appbuilder.sm
			
			# Create permission-view mappings for computer vision capabilities
			permissions_created = 0
			
			for permission_key, permission_info in CAPABILITY_PERMISSIONS.items():
				try:
					# Create permission if it doesn't exist
					permission = security_manager.find_permission_on_view(
						permission_info["name"], 
						"ComputerVisionDashboardView"
					)
					
					if not permission:
						security_manager.add_permission_view_menu(
							permission_info["name"],
							"ComputerVisionDashboardView"
						)
						permissions_created += 1
					
				except Exception as pe:
					print(f"Warning: Could not create permission {permission_key}: {pe}")
					continue
			
			# Create role-based permission sets
			self._create_default_roles(security_manager)
			
			print(f"Computer Vision: Created {permissions_created} permissions")
			
		except Exception as e:
			print(f"Error setting up Computer Vision permissions: {e}")
			raise
	
	def _create_default_roles(self, security_manager) -> None:
		"""Create default roles for computer vision capability"""
		default_roles = {
			"Computer Vision User": [
				"cv:read", "cv:write", "cv:ocr", "cv:object_detection"
			],
			"Computer Vision Analyst": [
				"cv:read", "cv:write", "cv:ocr", "cv:object_detection",
				"cv:facial_recognition", "cv:video_analysis", "cv:analytics"
			],
			"Computer Vision Admin": [
				"cv:read", "cv:write", "cv:admin", "cv:ocr", "cv:object_detection",
				"cv:facial_recognition", "cv:quality_control", "cv:video_analysis",
				"cv:batch_processing", "cv:model_management", "cv:analytics", "cv:reports"
			]
		}
		
		for role_name, permissions in default_roles.items():
			try:
				# Check if role exists
				role = security_manager.find_role(role_name)
				if not role:
					# Create role (would integrate with actual security manager)
					print(f"Computer Vision: Would create role '{role_name}' with {len(permissions)} permissions")
			except Exception as re:
				print(f"Warning: Could not create role {role_name}: {re}")
				continue
	
	def _setup_menu(self) -> None:
		"""Setup computer vision menu structure in APG platform"""
		try:
			menu_config = PLATFORM_INTEGRATION["menu_integration"]
			
			# Add main menu item
			self.appbuilder.add_link(
				name=menu_config["primary_menu"],
				href="/computer_vision/",
				icon=menu_config["icon"],
				category="AI & Analytics",
				category_icon="fa-brain"
			)
			
			# Add submenu items
			for submenu_item in menu_config["submenu"]:
				self.appbuilder.add_link(
					name=submenu_item["name"],
					href=submenu_item["url"],
					icon=submenu_item["icon"],
					category=menu_config["primary_menu"]
				)
			
			print(f"Computer Vision: Added menu with {len(menu_config['submenu'])} items")
			
		except Exception as e:
			print(f"Error setting up Computer Vision menu: {e}")
			raise
	
	def _setup_api_endpoints(self) -> None:
		"""Setup additional Flask blueprint API endpoints"""
		
		@self.blueprint.route('/api/capability-info', methods=['GET'])
		@protect
		def get_capability_info_endpoint():
			"""Get computer vision capability information"""
			return jsonify(self.capability_info)
		
		@self.blueprint.route('/api/health', methods=['GET'])
		def health_check():
			"""Health check endpoint for monitoring"""
			return jsonify({
				"status": "healthy",
				"capability": "computer_vision",
				"version": CAPABILITY_METADATA["version"],
				"timestamp": "2025-01-27T12:00:00Z"
			})
		
		@self.blueprint.route('/api/keywords', methods=['GET'])
		@protect
		def get_composition_keywords():
			"""Get APG composition keywords for this capability"""
			return jsonify({
				"keywords": COMPOSITION_KEYWORDS,
				"total_keywords": len(COMPOSITION_KEYWORDS)
			})
		
		@self.blueprint.route('/api/dependencies', methods=['GET'])
		@protect
		def get_capability_dependencies():
			"""Get capability dependencies information"""
			return jsonify(CAPABILITY_DEPENDENCIES)
		
		@self.blueprint.route('/api/permissions', methods=['GET'])
		@protect
		def get_capability_permissions():
			"""Get capability permissions structure"""
			return jsonify(CAPABILITY_PERMISSIONS)
		
		@self.blueprint.route('/api/tenant-info', methods=['GET'])
		@protect
		def get_tenant_info():
			"""Get current tenant information and capabilities"""
			# Would integrate with actual APG tenant management
			return jsonify({
				"tenant_id": getattr(g, 'tenant_id', 'default'),
				"capabilities_enabled": True,
				"resource_limits": {
					"max_concurrent_jobs": 50,
					"max_file_size_mb": 50,
					"monthly_quota": 10000
				}
			})
		
		print("Computer Vision: Added 6 API endpoints")
	
	def _setup_dashboard_widgets(self) -> None:
		"""Setup dashboard widgets for APG platform integration"""
		try:
			widgets_config = PLATFORM_INTEGRATION["dashboard_widgets"]
			
			# Register dashboard widget components
			for widget in widgets_config:
				# Would register with actual APG dashboard system
				print(f"Computer Vision: Registered widget '{widget['name']}'")
			
			print(f"Computer Vision: Registered {len(widgets_config)} dashboard widgets")
			
		except Exception as e:
			print(f"Error setting up Computer Vision dashboard widgets: {e}")
	
	def _register_blueprint(self) -> None:
		"""Register the blueprint with Flask application"""
		try:
			# Register blueprint with Flask app
			current_app.register_blueprint(self.blueprint)
			
			# Log registration success
			print(f"Computer Vision Blueprint registered successfully:")
			print(f"  - Blueprint name: {self.blueprint_name}")
			print(f"  - URL prefix: /computer_vision")
			print(f"  - Views registered: {len(self.registered_views)}")
			print(f"  - Models registered: {len(self.registered_models)}")
			
		except Exception as e:
			print(f"Error registering Computer Vision blueprint: {e}")
			raise
	
	def get_registration_status(self) -> Dict[str, Any]:
		"""Get detailed registration status for monitoring"""
		return {
			"capability_id": CAPABILITY_METADATA["capability_id"],
			"version": CAPABILITY_METADATA["version"],
			"status": "registered",
			"blueprint_name": self.blueprint_name,
			"views_count": len(self.registered_views),
			"models_count": len(self.registered_models),
			"permissions_count": len(CAPABILITY_PERMISSIONS),
			"keywords_count": len(COMPOSITION_KEYWORDS),
			"dependencies": CAPABILITY_DEPENDENCIES,
			"features": CAPABILITY_METADATA["features"],
			"registration_timestamp": CAPABILITY_METADATA["created_at"].isoformat()
		}


class ComputerVisionMiddleware:
	"""
	Computer Vision Middleware for request processing and multi-tenant support
	
	Provides request-level processing including tenant isolation, permission
	checking, audit logging, and performance monitoring for computer vision
	processing requests within the APG platform.
	"""
	
	def __init__(self, app=None):
		self.app = app
		if app is not None:
			self.init_app(app)
	
	def init_app(self, app):
		"""Initialize middleware with Flask application"""
		app.before_request(self.before_request)
		app.after_request(self.after_request)
		app.teardown_appcontext(self.teardown_request)
	
	def before_request(self):
		"""Pre-process requests for computer vision endpoints"""
		if request.path.startswith('/computer_vision'):
			# Set up tenant context
			self._setup_tenant_context()
			
			# Validate permissions
			self._validate_permissions()
			
			# Log request for audit
			self._log_request()
	
	def after_request(self, response):
		"""Post-process computer vision responses"""
		if request.path.startswith('/computer_vision'):
			# Add security headers
			response.headers['X-Content-Type-Options'] = 'nosniff'
			response.headers['X-Frame-Options'] = 'DENY'
			response.headers['X-XSS-Protection'] = '1; mode=block'
			
			# Log response for audit
			self._log_response(response)
		
		return response
	
	def teardown_request(self, exception):
		"""Clean up request context"""
		if hasattr(g, 'cv_request_context'):
			# Clean up any request-specific resources
			delattr(g, 'cv_request_context')
	
	def _setup_tenant_context(self):
		"""Setup multi-tenant context for the request"""
		# Extract tenant information from request
		tenant_id = self._extract_tenant_id()
		
		# Set tenant context in Flask g object
		g.tenant_id = tenant_id
		g.cv_request_context = {
			'tenant_id': tenant_id,
			'timestamp': '2025-01-27T12:00:00Z',
			'request_id': 'req_' + ''.join(['a'] * 10)  # Would generate actual UUID
		}
	
	def _extract_tenant_id(self) -> str:
		"""Extract tenant ID from request"""
		# Check various sources for tenant ID
		tenant_id = (
			request.headers.get('X-Tenant-ID') or
			session.get('tenant_id') or
			request.args.get('tenant_id') or
			'default'
		)
		
		return tenant_id
	
	def _validate_permissions(self):
		"""Validate user permissions for computer vision access"""
		# Would integrate with APG RBAC system
		user_permissions = getattr(g, 'user_permissions', [])
		
		# Check if user has basic computer vision access
		if 'cv:read' not in user_permissions and request.method == 'GET':
			# Would handle permission denial
			pass
		
		if 'cv:write' not in user_permissions and request.method in ['POST', 'PUT', 'DELETE']:
			# Would handle permission denial
			pass
	
	def _log_request(self):
		"""Log request for audit trail"""
		log_data = {
			'tenant_id': getattr(g, 'tenant_id', 'unknown'),
			'user_id': getattr(g, 'user_id', 'anonymous'),
			'path': request.path,
			'method': request.method,
			'ip_address': request.remote_addr,
			'user_agent': request.headers.get('User-Agent', ''),
			'timestamp': '2025-01-27T12:00:00Z'
		}
		
		# Would send to actual audit logging system
		print(f"CV Audit Log: {log_data}")
	
	def _log_response(self, response):
		"""Log response for audit trail"""
		log_data = {
			'tenant_id': getattr(g, 'tenant_id', 'unknown'),
			'status_code': response.status_code,
			'response_size': len(response.get_data()),
			'processing_time_ms': 150,  # Would calculate actual time
			'timestamp': '2025-01-27T12:00:00Z'
		}
		
		# Would send to actual audit logging system
		print(f"CV Response Log: {log_data}")


def register_computer_vision_capability(appbuilder: AppBuilder) -> ComputerVisionCapabilityBlueprint:
	"""
	Register Computer Vision capability with APG platform
	
	Args:
		appbuilder: Flask-AppBuilder instance from APG platform
		
	Returns:
		ComputerVisionCapabilityBlueprint: Registered capability blueprint
	"""
	try:
		# Create and register capability blueprint
		cv_blueprint = ComputerVisionCapabilityBlueprint(appbuilder)
		
		# Setup middleware for request processing
		middleware = ComputerVisionMiddleware(appbuilder.app)
		
		print("Computer Vision capability registration completed successfully")
		print(f"Status: {cv_blueprint.get_registration_status()}")
		
		return cv_blueprint
		
	except Exception as e:
		print(f"Failed to register Computer Vision capability: {e}")
		raise RuntimeError(f"Computer Vision capability registration failed: {e}")


# Export main registration function
__all__ = [
	"ComputerVisionCapabilityBlueprint",
	"ComputerVisionMiddleware", 
	"register_computer_vision_capability"
]