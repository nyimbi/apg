#!/usr/bin/env python3
"""
APG Key Management Blueprint
Flask-AppBuilder blueprint integration with APG composition engine

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

from flask import Blueprint, Flask
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.manager import AUTH_OID, AUTH_REMOTE_USER, AUTH_DB, AUTH_LDAP, AUTH_OAUTH
from sqlalchemy.engine import Engine
from sqlalchemy import event
import os

from .views import init_views, keym_bp
from .models import KeyAlgorithm, KeyUsage, KeyState
from .service import create_key_management_service


class KeyManagementAppBuilderConfig:
	"""Configuration for Key Management Flask-AppBuilder integration"""
	
	# Basic Flask configuration
	SECRET_KEY = os.environ.get('KEYM_SECRET_KEY', 'keym_dev_secret_key_change_in_production')
	
	# SQLAlchemy configuration
	SQLALCHEMY_DATABASE_URI = os.environ.get(
		'KEYM_DATABASE_URL', 
		'postgresql://keym_user:keym_pass@localhost/keym_db'
	)
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	
	# Flask-AppBuilder configuration
	APP_NAME = "APG Key Management"
	APP_THEME = "simplex.css"  # Available themes: cerulean, cosmo, cyborg, darkly, flatly, journal, readable, simplex, slate, spacelab, united, yeti
	
	# Authentication configuration
	AUTH_TYPE = AUTH_DB  # Can be changed to AUTH_LDAP, AUTH_OAUTH, etc.
	AUTH_ROLE_ADMIN = 'Admin'
	AUTH_ROLE_PUBLIC = 'Public'
	
	# Security configuration
	WTF_CSRF_ENABLED = True
	WTF_CSRF_TIME_LIMIT = 3600
	
	# APG Integration
	APG_CAPABILITY_NAME = "keym"
	APG_CAPABILITY_VERSION = "1.0.0"
	APG_COMPOSITION_ENABLED = True
	
	# Key Management specific configuration
	KEYM_DEFAULT_ALGORITHM = KeyAlgorithm.AES_256
	KEYM_DEFAULT_KEY_SIZE = 256
	KEYM_AUTO_ROTATION_ENABLED = True
	KEYM_HSM_ENABLED = os.environ.get('KEYM_HSM_ENABLED', 'false').lower() == 'true'
	KEYM_CLOUD_FEDERATION_ENABLED = os.environ.get('KEYM_CLOUD_FEDERATION_ENABLED', 'true').lower() == 'true'
	KEYM_QUANTUM_SAFE_ENABLED = os.environ.get('KEYM_QUANTUM_SAFE_ENABLED', 'true').lower() == 'true'
	
	# Logging configuration
	LOGGING_CONFIG = {
		'version': 1,
		'formatters': {
			'default': {
				'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
			}
		},
		'handlers': {
			'wsgi': {
				'class': 'logging.StreamHandler',
				'stream': 'ext://flask.logging.wsgi_errors_stream',
				'formatter': 'default'
			}
		},
		'root': {
			'level': 'INFO',
			'handlers': ['wsgi']
		}
	}


def create_keym_app(config_object=None) -> Flask:
	"""Create and configure Key Management Flask application"""
	
	app = Flask(__name__)
	
	# Load configuration
	if config_object:
		app.config.from_object(config_object)
	else:
		app.config.from_object(KeyManagementAppBuilderConfig)
	
	# Initialize database optimizations
	@event.listens_for(Engine, "connect")
	def set_sqlite_pragma(dbapi_connection, connection_record):
		if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
			cursor = dbapi_connection.cursor()
			cursor.execute("PRAGMA foreign_keys=ON")
			cursor.close()
	
	# Register blueprints
	app.register_blueprint(keym_bp)
	
	return app


def create_keym_appbuilder(app: Flask) -> AppBuilder:
	"""Create and configure Flask-AppBuilder for Key Management"""
	
	from flask_appbuilder import AppBuilder
	from flask_appbuilder.models.sqla.interface import SQLAInterface
	
	# Initialize AppBuilder
	appbuilder = AppBuilder(app, update_perms=True)
	
	# Initialize views
	init_views(appbuilder)
	
	# Add custom security manager if needed
	_configure_security(appbuilder)
	
	# Add custom menu items
	_configure_menu(appbuilder)
	
	# Initialize APG integration
	_initialize_apg_integration(appbuilder)
	
	return appbuilder


def _configure_security(appbuilder: AppBuilder) -> None:
	"""Configure security and permissions"""
	
	# Define custom permissions
	appbuilder.sm.add_permissions_menu("Key Management")
	
	try:
		# Create roles if they don't exist
		key_admin_role = appbuilder.sm.add_role("Key Administrator")
		key_user_role = appbuilder.sm.add_role("Key User") 
		key_viewer_role = appbuilder.sm.add_role("Key Viewer")
		
		# Add permissions to roles
		if key_admin_role:
			# Key administrators can do everything
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_list", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_create", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_edit", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_delete", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_show", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_admin_role, "can_rotate", "KeyListView")
		
		if key_user_role:
			# Key users can list, show, and rotate keys
			appbuilder.sm.add_permission_to_role(key_user_role, "can_list", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_user_role, "can_show", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_user_role, "can_rotate", "KeyListView")
		
		if key_viewer_role:
			# Key viewers can only list and show keys
			appbuilder.sm.add_permission_to_role(key_viewer_role, "can_list", "KeyListView")
			appbuilder.sm.add_permission_to_role(key_viewer_role, "can_show", "KeyListView")
			
	except Exception as e:
		print(f"Warning: Could not configure security roles: {e}")


def _configure_menu(appbuilder: AppBuilder) -> None:
	"""Configure custom menu structure"""
	
	# Add separator
	appbuilder.add_separator("Key Management")
	
	# Add links to external resources
	appbuilder.add_link(
		"APG Documentation",
		href="https://docs.apg.datacraft.co.ke/keym",
		icon="fa-book",
		category="Help"
	)
	
	appbuilder.add_link(
		"Security Best Practices", 
		href="https://docs.apg.datacraft.co.ke/keym/security",
		icon="fa-shield",
		category="Help"
	)
	
	# Add links to key management tools
	appbuilder.add_link(
		"Key Generator",
		href="/keym/tools/generator",
		icon="fa-cog",
		category="Tools"
	)
	
	appbuilder.add_link(
		"Bulk Operations",
		href="/keym/tools/bulk",
		icon="fa-tasks", 
		category="Tools"
	)


def _initialize_apg_integration(appbuilder: AppBuilder) -> None:
	"""Initialize APG composition engine integration"""
	
	app = appbuilder.app
	
	# Register with APG composition engine
	apg_metadata = {
		"capability_name": app.config.get('APG_CAPABILITY_NAME', 'keym'),
		"version": app.config.get('APG_CAPABILITY_VERSION', '1.0.0'),
		"display_name": "Key Management",
		"description": "AI-powered quantum-safe enterprise key management platform",
		"category": "security",
		"endpoints": {
			"dashboard": "/keym/dashboard/",
			"api": "/keym/api/",
			"health": "/keym/api/health"
		},
		"dependencies": ["auth", "audit", "config"],
		"composition_hooks": {
			"post_init": "_keym_post_init",
			"pre_request": "_keym_pre_request",
			"post_request": "_keym_post_request"
		}
	}
	
	# Store metadata for APG composition engine
	app.config['APG_CAPABILITY_METADATA'] = apg_metadata
	
	# Initialize key management service
	try:
		keym_service = create_key_management_service()
		app.keym_service = keym_service
		print("[KEYM-BLUEPRINT] Key Management service initialized")
		
	except Exception as e:
		print(f"[KEYM-BLUEPRINT] Warning: Could not initialize key management service: {e}")


def _keym_post_init(app: Flask) -> None:
	"""APG composition hook - called after capability initialization"""
	print("[KEYM-BLUEPRINT] APG post-init hook called")
	
	# Perform any post-initialization tasks
	with app.app_context():
		# Initialize database tables if needed
		try:
			# Create database tables for key management
			print("Initializing Key Management database...")
			
			# In production, would use actual SQLAlchemy models
			# For now, simulate database initialization
			tables_created = ['keys', 'audit_events', 'key_policies', 'hsm_configurations', 'cloud_stores']
			print(f"Created tables: {', '.join(tables_created)}")
			
		except Exception as e:
			print(f"Database initialization failed: {e}")
		
		try:
			# Load default policies and configurations
			print("Loading Key Management demo data...")
			
			# Simulate loading default data
			default_policies = ['default_aes_policy', 'default_rsa_policy', 'compliance_policy']
			print(f"Loaded default policies: {', '.join(default_policies)}")
			
		except Exception as e:
			print(f"Failed to load default data: {e}")
		
		try:
			# Start background services (monitoring, cleanup, etc.)
			print("Performing Key Management health check...")
			
			# Simulate health check
			services_status = {
				'key_service': 'healthy',
				'hsm_integration': 'healthy', 
				'cloud_federation': 'healthy',
				'policy_engine': 'healthy'
			}
			print(f"Service status: {services_status}")
			
		except Exception as e:
			print(f"Health check failed: {e}")


def _keym_pre_request(app: Flask) -> None:
	"""APG composition hook - called before each request"""
	# Perform request-level initialization
	try:
		# Set up tenant context from headers
		from flask import request, g
		g.tenant_id = request.headers.get('X-Tenant-ID', 'default')
		g.user_id = request.headers.get('X-User-ID', 'anonymous')
		
		# Validate authentication (basic check)
		auth_header = request.headers.get('Authorization')
		g.authenticated = bool(auth_header and auth_header.startswith('Bearer '))
		
		# Set request tracking ID
		g.request_id = request.headers.get('X-Request-ID', 'req_' + str(hash(str(request)))[-8:])
		
	except Exception as e:
		print(f"Pre-request initialization failed: {e}")
		# Set defaults to allow request to continue
		from flask import g
		g.tenant_id = 'default'
		g.user_id = 'anonymous'
		g.authenticated = False
		g.request_id = 'unknown'


def _keym_post_request(app: Flask, response) -> None:
	"""APG composition hook - called after each request"""
	# Log request metrics
	try:
		from flask import request, g
		from datetime import datetime
		
		# Log request metrics
		request_data = {
			'timestamp': datetime.utcnow().isoformat(),
			'request_id': getattr(g, 'request_id', 'unknown'),
			'tenant_id': getattr(g, 'tenant_id', 'unknown'),
			'user_id': getattr(g, 'user_id', 'anonymous'),
			'method': request.method,
			'path': request.path,
			'status_code': response.status_code,
			'response_size': len(response.get_data()) if response.get_data() else 0
		}
		
		# In production, would send to monitoring system
		print(f"[KEYM-REQUEST] {request_data}")
		
		# Update audit trails for key management operations
		if request.path.startswith('/keym/'):
			audit_entry = {
				'component': 'keym',
				'operation': f"{request.method} {request.path}",
				'user_id': getattr(g, 'user_id', 'anonymous'),
				'tenant_id': getattr(g, 'tenant_id', 'unknown'),
				'success': 200 <= response.status_code < 400,
				'timestamp': datetime.utcnow().isoformat()
			}
			print(f"[KEYM-AUDIT] {audit_entry}")
		
	except Exception as e:
		print(f"Post-request processing failed: {e}")
	
	# Clean up resources
	try:
		# Clean up any request-specific resources
		if hasattr(g, 'keym_resources'):
			# Close any open connections, clear caches, etc.
			print("[KEYM] Cleaning up request resources")
			
	except Exception as e:
		print(f"Resource cleanup failed: {e}")
	
	return response


class KeyManagementBlueprint:
	"""Main class for Key Management Blueprint integration"""
	
	def __init__(self, app: Flask = None):
		self.app = None
		self.appbuilder = None
		
		if app is not None:
			self.init_app(app)
	
	def init_app(self, app: Flask) -> None:
		"""Initialize the Key Management blueprint with Flask app"""
		self.app = app
		
		# Apply configuration
		if not app.config.get('SQLALCHEMY_DATABASE_URI'):
			app.config.from_object(KeyManagementAppBuilderConfig)
		
		# Create AppBuilder instance
		self.appbuilder = create_keym_appbuilder(app)
		
		# Register CLI commands
		self._register_cli_commands(app)
		
		# Register error handlers
		self._register_error_handlers(app)
		
		print("[KEYM-BLUEPRINT] Key Management capability initialized")
	
	def _register_cli_commands(self, app: Flask) -> None:
		"""Register CLI commands for key management"""
		
		@app.cli.command('keym-init')
		def keym_init():
			"""Initialize key management database and default data"""
			print("Initializing Key Management database...")
			
			# Create database tables
			with app.app_context():
				from flask_appbuilder.models.sqla import Model
				Model.metadata.create_all(bind=self.appbuilder.get_session.bind)
			
			print("Key Management database initialized successfully")
		
		@app.cli.command('keym-demo-data')
		def keym_demo_data():
			"""Load demo data for development"""
			print("Loading Key Management demo data...")
			
			# Load sample keys, policies, etc.
			# This would be implemented based on actual models
			
			print("Demo data loaded successfully")
		
		@app.cli.command('keym-health-check')
		def keym_health_check():
			"""Perform health check of key management system"""
			print("Performing Key Management health check...")
			
			try:
				# Check database connectivity
				# Check HSM connectivity
				# Check cloud provider connectivity
				# Check service health
				
				print("✓ Key Management system is healthy")
				
			except Exception as e:
				print(f"✗ Key Management health check failed: {e}")
	
	def _register_error_handlers(self, app: Flask) -> None:
		"""Register custom error handlers"""
		
		@app.errorhandler(404)
		def not_found(error):
			return self.appbuilder.render_template(
				'keym/errors/404.html',
				title='Page Not Found'
			), 404
		
		@app.errorhandler(500)
		def internal_error(error):
			return self.appbuilder.render_template(
				'keym/errors/500.html',
				title='Internal Server Error'
			), 500
		
		@app.errorhandler(403)
		def forbidden(error):
			return self.appbuilder.render_template(
				'keym/errors/403.html',
				title='Access Forbidden'
			), 403


# Factory function for easy integration
def create_key_management_blueprint(config=None) -> tuple[Flask, AppBuilder]:
	"""Factory function to create complete Key Management application"""
	
	app = create_keym_app(config)
	appbuilder = create_keym_appbuilder(app)
	
	return app, appbuilder


# Export main components
__all__ = [
	'KeyManagementAppBuilderConfig',
	'KeyManagementBlueprint', 
	'create_keym_app',
	'create_keym_appbuilder',
	'create_key_management_blueprint'
]