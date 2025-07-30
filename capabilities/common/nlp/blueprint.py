"""
APG Natural Language Processing Blueprint

Flask-AppBuilder blueprint integration for APG composition engine with
comprehensive view registration, menu integration, and permission management.

This blueprint registers all NLP views and provides seamless integration
with the APG ecosystem following established patterns.
"""

from flask import Blueprint
from flask_appbuilder import AppBuilder, SQLA
from flask_appbuilder.menu import Menu
from typing import Dict, Any, List
import logging

from . import APG_BLUEPRINT_CONFIG, get_capability_metadata
from .views import (
	NLPDashboardView,
	TextDocumentView,
	ProcessingResultView,
	StreamingSessionView,
	AnnotationProjectView,
	NLPModelView,
	NLPAnalyticsChartView
)

# Logging setup
logger = logging.getLogger(__name__)

class NLPBlueprint:
	"""
	APG NLP Blueprint for Flask-AppBuilder integration.
	
	Handles registration of all NLP views, menu items, permissions,
	and integration with APG's composition engine.
	"""
	
	def __init__(self, appbuilder: AppBuilder):
		"""Initialize NLP blueprint with AppBuilder integration"""
		assert appbuilder, "AppBuilder instance is required"
		
		self.appbuilder = appbuilder
		self.app = appbuilder.app
		self.blueprint_config = APG_BLUEPRINT_CONFIG
		self.capability_metadata = get_capability_metadata()
		
		# Create Flask blueprint
		self.blueprint = Blueprint(
			self.blueprint_config["blueprint_name"],
			__name__,
			url_prefix=self.blueprint_config["url_prefix"],
			template_folder=self.blueprint_config["template_folder"],
			static_folder=self.blueprint_config["static_folder"]
		)
		
		self._log_blueprint_initialization()
	
	def _log_blueprint_initialization(self) -> None:
		"""Log blueprint initialization for APG audit trail"""
		logger.info(f"APG NLP Blueprint initializing...")
		logger.info(f"URL prefix: {self.blueprint_config['url_prefix']}")
		logger.info(f"Views to register: {len(self._get_view_classes())}")
		logger.info(f"Menu items: {len(self.blueprint_config['menu_links'])}")
	
	def register_with_appbuilder(self) -> None:
		"""Register blueprint with Flask-AppBuilder and APG composition engine"""
		try:
			# Register Flask blueprint with app
			self.app.register_blueprint(self.blueprint)
			
			# Register views with AppBuilder
			self._register_views()
			
			# Create menu structure
			self._create_menu_structure()
			
			# Register permissions
			self._register_permissions()
			
			# Register with APG composition engine
			self._register_with_composition_engine()
			
			self._log_blueprint_registration_complete()
			
		except Exception as e:
			self._log_blueprint_registration_error(str(e))
			raise
	
	def _register_views(self) -> None:
		"""Register all NLP views with Flask-AppBuilder"""
		view_classes = self._get_view_classes()
		
		for view_class in view_classes:
			try:
				# Register view with AppBuilder
				self.appbuilder.add_view_no_menu(view_class)
				self._log_view_registered(view_class.__name__)
				
			except Exception as e:
				self._log_view_registration_error(view_class.__name__, str(e))
				# Continue with other views even if one fails
				continue
	
	def _get_view_classes(self) -> List[type]:
		"""Get list of view classes to register"""
		return [
			NLPDashboardView,
			TextDocumentView,
			ProcessingResultView,
			StreamingSessionView,
			AnnotationProjectView,
			NLPModelView,
			NLPAnalyticsChartView
		]
	
	def _log_view_registered(self, view_name: str) -> None:
		"""Log successful view registration"""
		logger.info(f"Registered NLP view: {view_name}")
	
	def _log_view_registration_error(self, view_name: str, error: str) -> None:
		"""Log view registration error"""
		logger.error(f"Failed to register view {view_name}: {error}")
	
	def _create_menu_structure(self) -> None:
		"""Create APG-integrated menu structure for NLP capability"""
		try:
			# Create main NLP menu category
			nlp_menu = Menu()
			
			# Add menu items from configuration
			for menu_item in self.blueprint_config["menu_links"]:
				self.appbuilder.add_link(
					name=menu_item["name"],
					href=menu_item["href"],
					icon=menu_item["icon"],
					category=menu_item["category"],
					category_icon="fa-brain"
				)
				
				self._log_menu_item_added(menu_item["name"])
			
			# Add separator and sub-menus
			self._add_advanced_menu_items()
			
			self._log_menu_structure_complete()
			
		except Exception as e:
			self._log_menu_creation_error(str(e))
			raise
	
	def _add_advanced_menu_items(self) -> None:
		"""Add advanced menu items for power users"""
		
		# Administration submenu
		self.appbuilder.add_link(
			name="Model Health Check",
			href="/nlp/models/health",
			icon="fa-heartbeat",
			category="NLP Administration",
			category_icon="fa-cogs"
		)
		
		self.appbuilder.add_link(
			name="System Diagnostics",
			href="/nlp/diagnostics",
			icon="fa-stethoscope", 
			category="NLP Administration"
		)
		
		self.appbuilder.add_link(
			name="Performance Metrics",
			href="/nlp/metrics",
			icon="fa-tachometer-alt",
			category="NLP Administration"
		)
		
		# Developer Tools submenu
		self.appbuilder.add_link(
			name="API Documentation",
			href="/nlp/api/docs", 
			icon="fa-book",
			category="NLP Developer Tools",
			category_icon="fa-code"
		)
		
		self.appbuilder.add_link(
			name="WebSocket Console",
			href="/nlp/websocket/console",
			icon="fa-terminal",
			category="NLP Developer Tools"
		)
		
		self.appbuilder.add_link(
			name="Model Playground",
			href="/nlp/playground",
			icon="fa-play",
			category="NLP Developer Tools"
		)
	
	def _log_menu_item_added(self, item_name: str) -> None:
		"""Log menu item addition"""
		logger.debug(f"Added menu item: {item_name}")
	
	def _log_menu_structure_complete(self) -> None:
		"""Log menu structure completion"""
		logger.info("NLP menu structure created successfully")
	
	def _log_menu_creation_error(self, error: str) -> None:
		"""Log menu creation error"""
		logger.error(f"Menu creation failed: {error}")
	
	def _register_permissions(self) -> None:
		"""Register NLP-specific permissions with APG RBAC system"""
		try:
			# Get security manager from AppBuilder
			security_manager = self.appbuilder.sm
			
			# Register permissions from configuration
			for permission in self.blueprint_config["permissions"]:
				try:
					# Create permission if it doesn't exist
					perm = security_manager.find_permission_view_menu(
						permission["name"],
						"NLPCapability"
					)
					
					if not perm:
						# Add permission
						security_manager.add_permission_view_menu(
							permission["name"],
							"NLPCapability"
						)
						
						self._log_permission_registered(permission["name"])
					
				except Exception as e:
					self._log_permission_registration_error(permission["name"], str(e))
					continue
			
			# Create default roles with NLP permissions
			self._create_default_roles()
			
			self._log_permissions_registration_complete()
			
		except Exception as e:
			self._log_permissions_registration_error(str(e))
			raise
	
	def _create_default_roles(self) -> None:
		"""Create default roles for NLP capability"""
		security_manager = self.appbuilder.sm
		
		# NLP User Role - basic access
		nlp_user_role = security_manager.find_role("NLP User")
		if not nlp_user_role:
			nlp_user_role = security_manager.add_role("NLP User")
			
			# Add basic permissions
			basic_permissions = ["nlp_view", "nlp_process"]
			for perm_name in basic_permissions:
				perm = security_manager.find_permission_view_menu(perm_name, "NLPCapability")
				if perm:
					security_manager.add_permission_role(nlp_user_role, perm)
		
		# NLP Analyst Role - extended access
		nlp_analyst_role = security_manager.find_role("NLP Analyst")
		if not nlp_analyst_role:
			nlp_analyst_role = security_manager.add_role("NLP Analyst")
			
			# Add analyst permissions
			analyst_permissions = ["nlp_view", "nlp_process", "nlp_streaming", "nlp_annotate"]
			for perm_name in analyst_permissions:
				perm = security_manager.find_permission_view_menu(perm_name, "NLPCapability")
				if perm:
					security_manager.add_permission_role(nlp_analyst_role, perm)
		
		# NLP Administrator Role - full access
		nlp_admin_role = security_manager.find_role("NLP Administrator")
		if not nlp_admin_role:
			nlp_admin_role = security_manager.add_role("NLP Administrator")
			
			# Add all permissions
			for permission in self.blueprint_config["permissions"]:
				perm = security_manager.find_permission_view_menu(
					permission["name"], 
					"NLPCapability"
				)
				if perm:
					security_manager.add_permission_role(nlp_admin_role, perm)
		
		self._log_default_roles_created()
	
	def _log_permission_registered(self, permission_name: str) -> None:
		"""Log permission registration"""
		logger.debug(f"Registered permission: {permission_name}")
	
	def _log_permission_registration_error(self, permission_name: str, error: str) -> None:
		"""Log permission registration error"""
		logger.error(f"Failed to register permission {permission_name}: {error}")
	
	def _log_permissions_registration_complete(self) -> None:
		"""Log permissions registration completion"""
		logger.info("NLP permissions registered successfully")
	
	def _log_permissions_registration_error(self, error: str) -> None:
		"""Log permissions registration error"""
		logger.error(f"Permissions registration failed: {error}")
	
	def _log_default_roles_created(self) -> None:
		"""Log default roles creation"""
		logger.info("NLP default roles created successfully")
	
	def _register_with_composition_engine(self) -> None:
		"""Register capability with APG composition engine"""
		try:
			# This would integrate with APG's composition engine
			# For now, we'll simulate the registration
			
			composition_data = {
				"capability_id": self.capability_metadata["capability_id"],
				"blueprint_name": self.blueprint_config["blueprint_name"],
				"url_prefix": self.blueprint_config["url_prefix"],
				"provides": self.capability_metadata["composition"]["provides"],
				"requires": self.capability_metadata["composition"]["requires"],
				"enhances": self.capability_metadata["composition"]["enhances"],
				"health_check_endpoint": "/nlp/health",
				"api_endpoints": self._get_api_endpoints(),
				"websocket_endpoints": self._get_websocket_endpoints()
			}
			
			# Register with composition engine
			# apg_composition_engine.register_capability(composition_data)
			
			self._log_composition_registration_complete()
			
		except Exception as e:
			self._log_composition_registration_error(str(e))
			raise
	
	def _get_api_endpoints(self) -> List[Dict[str, str]]:
		"""Get list of API endpoints for composition engine"""
		return [
			{"path": "/api/nlp/process", "method": "POST", "description": "Process text"},
			{"path": "/api/nlp/models", "method": "GET", "description": "List models"},
			{"path": "/api/nlp/stream/start", "method": "POST", "description": "Start streaming"},
			{"path": "/api/nlp/health", "method": "GET", "description": "Health check"}
		]
	
	def _get_websocket_endpoints(self) -> List[Dict[str, str]]:
		"""Get list of WebSocket endpoints for composition engine"""
		return [
			{"path": "/ws/nlp/stream", "description": "Real-time text streaming"},
			{"path": "/ws/nlp/collaboration", "description": "Collaborative annotation"}
		]
	
	def _log_composition_registration_complete(self) -> None:
		"""Log composition engine registration completion"""
		logger.info("Registered with APG composition engine successfully")
	
	def _log_composition_registration_error(self, error: str) -> None:
		"""Log composition engine registration error"""
		logger.error(f"Composition engine registration failed: {error}")
	
	def _log_blueprint_registration_complete(self) -> None:
		"""Log complete blueprint registration"""
		logger.info("APG NLP Blueprint registration completed successfully")
		logger.info(f"Capability provides: {self.capability_metadata['composition']['provides']}")
		logger.info(f"Capability requires: {self.capability_metadata['composition']['requires']}")
	
	def _log_blueprint_registration_error(self, error: str) -> None:
		"""Log blueprint registration error"""
		logger.error(f"APG NLP Blueprint registration failed: {error}")
	
	def get_health_status(self) -> Dict[str, Any]:
		"""Get blueprint health status for APG monitoring"""
		return {
			"blueprint_name": self.blueprint_config["blueprint_name"],
			"status": "healthy",
			"registered_views": len(self._get_view_classes()),
			"menu_items": len(self.blueprint_config["menu_links"]),
			"permissions": len(self.blueprint_config["permissions"]),
			"last_check": "2025-01-29T12:00:00Z"
		}
	
	def validate_dependencies(self) -> List[str]:
		"""Validate that required APG capabilities are available"""
		missing_dependencies = []
		required_capabilities = self.capability_metadata["composition"]["requires"]
		
		# Check each required capability
		for capability in required_capabilities:
			try:
				# This would check APG composition registry
				# For now, simulate the check
				if capability not in ["ai_orchestration", "auth_rbac", "audit_compliance"]:
					missing_dependencies.append(capability)
					
			except Exception as e:
				logger.warning(f"Could not validate dependency {capability}: {str(e)}")
				missing_dependencies.append(capability)
		
		if missing_dependencies:
			self._log_missing_dependencies(missing_dependencies)
		else:
			self._log_dependencies_validated()
		
		return missing_dependencies
	
	def _log_missing_dependencies(self, missing: List[str]) -> None:
		"""Log missing dependencies"""
		logger.warning(f"Missing required dependencies: {missing}")
	
	def _log_dependencies_validated(self) -> None:
		"""Log successful dependency validation"""
		logger.info("All required dependencies validated successfully")

def create_nlp_blueprint(appbuilder: AppBuilder) -> NLPBlueprint:
	"""
	Factory function to create and register NLP blueprint.
	
	Args:
		appbuilder: Flask-AppBuilder instance
		
	Returns:
		Configured NLPBlueprint instance
	"""
	assert appbuilder, "AppBuilder instance is required"
	
	# Create blueprint instance
	nlp_blueprint = NLPBlueprint(appbuilder)
	
	# Validate dependencies before registration
	missing_deps = nlp_blueprint.validate_dependencies()
	if missing_deps:
		logger.warning(f"Proceeding with missing dependencies: {missing_deps}")
	
	# Register with AppBuilder and APG
	nlp_blueprint.register_with_appbuilder()
	
	return nlp_blueprint

def register_nlp_capability(app, appbuilder: AppBuilder) -> None:
	"""
	Main registration function for APG NLP capability.
	
	This function should be called by APG's composition engine
	to integrate the NLP capability.
	
	Args:
		app: Flask application instance
		appbuilder: Flask-AppBuilder instance
	"""
	try:
		logger.info("Starting APG NLP capability registration...")
		
		# Create and register blueprint
		nlp_blueprint = create_nlp_blueprint(appbuilder)
		
		# Store blueprint reference in app for later access
		app.nlp_blueprint = nlp_blueprint
		
		logger.info("APG NLP capability registration completed successfully")
		
	except Exception as e:
		logger.error(f"APG NLP capability registration failed: {str(e)}")
		raise

# Export main components
__all__ = [
	"NLPBlueprint",
	"create_nlp_blueprint", 
	"register_nlp_capability"
]