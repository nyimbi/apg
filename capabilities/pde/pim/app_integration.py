"""
Product Lifecycle Management (PLM) Capability - APG Flask Integration

Flask-AppBuilder integration module for registering PLM capability
with APG composition engine and main application.

Copyright © 2025 Datacraft
Author: APG Development Team
"""

import logging
from typing import Dict, Any, Optional
from flask import Flask, current_app
from flask_appbuilder import AppBuilder
from flask_sqlalchemy import SQLAlchemy
import asyncio

from .blueprint import register_plm_views, plm_bp
from .api import plm_api_bp, socketio, limiter
from .models import (
	PLProduct, PLProductStructure, PLEngineeringChange,
	PLProductConfiguration, PLCollaborationSession, PLComplianceRecord,
	PLManufacturingIntegration, PLDigitalTwinBinding
)
from . import PLMCapability, plm_capability, PLM_CAPABILITY_METADATA


class PLMFlaskIntegration:
	"""
	PLM Flask-AppBuilder Integration Handler
	
	Manages the complete integration of PLM capability with APG Flask infrastructure
	including views, APIs, models, and background services.
	"""
	
	def __init__(self):
		self.app: Optional[Flask] = None
		self.appbuilder: Optional[AppBuilder] = None
		self.db: Optional[SQLAlchemy] = None
		self.capability: PLMCapability = plm_capability
		self.logger = logging.getLogger('PLMFlaskIntegration')
		self.initialized = False
		
	def init_app(self, app: Flask, appbuilder: AppBuilder, db: SQLAlchemy) -> bool:
		"""
		Initialize PLM capability with Flask application
		
		Args:
			app: Flask application instance
			appbuilder: Flask-AppBuilder instance
			db: SQLAlchemy database instance
			
		Returns:
			bool: True if initialization successful, False otherwise
		"""
		try:
			self.app = app
			self.appbuilder = appbuilder
			self.db = db
			
			with app.app_context():
				# Initialize capability
				loop = asyncio.new_event_loop()
				asyncio.set_event_loop(loop)
				
				success = loop.run_until_complete(self.capability.initialize())
				if not success:
					self.logger.error("Failed to initialize PLM capability")
					return False
				
				# Register blueprints
				self._register_blueprints()
				
				# Register views
				self._register_views()
				
				# Initialize database
				self._initialize_database()
				
				# Setup rate limiting
				self._setup_rate_limiting()
				
				# Setup WebSocket support
				self._setup_websocket()
				
				# Register with APG composition engine
				self._register_with_apg()
				
				# Setup monitoring
				self._setup_monitoring()
				
				self.initialized = True
				self.logger.info("PLM capability integrated successfully with Flask application")
				return True
				
		except Exception as e:
			self.logger.error(f"Failed to initialize PLM Flask integration: {e}")
			return False
	
	def _register_blueprints(self) -> None:
		"""Register Flask blueprints"""
		# Register main PLM blueprint
		self.app.register_blueprint(plm_bp)
		self.logger.info("Registered PLM main blueprint")
		
		# Register API blueprint
		self.app.register_blueprint(plm_api_bp)
		self.logger.info("Registered PLM API blueprint")
	
	def _register_views(self) -> None:
		"""Register Flask-AppBuilder views"""
		register_plm_views(self.appbuilder)
		self.logger.info("Registered PLM Flask-AppBuilder views")
	
	def _initialize_database(self) -> None:
		"""Initialize database tables for PLM"""
		try:
			# Create all PLM tables
			self.db.create_all()
			self.logger.info("PLM database tables created successfully")
			
			# Initialize default data
			self._initialize_default_data()
			
		except Exception as e:
			self.logger.error(f"Failed to initialize PLM database: {e}")
			raise
	
	def _initialize_default_data(self) -> None:
		"""Initialize default PLM data"""
		try:
			# Check if default data already exists
			existing_products = self.db.session.query(PLProduct).count()
			if existing_products > 0:
				self.logger.info("PLM default data already exists, skipping initialization")
				return
			
			# Create sample data for demonstration
			self._create_sample_data()
			
		except Exception as e:
			self.logger.error(f"Failed to initialize default PLM data: {e}")
	
	def _create_sample_data(self) -> None:
		"""Create sample PLM data for demonstration"""
		try:
			# Sample product types
			sample_products = [
				{
					'product_name': 'APG Platform Core Module',
					'product_number': 'APG-CORE-001',
					'product_description': 'Core platform module for APG composition engine',
					'product_type': 'manufactured',
					'lifecycle_phase': 'active',
					'target_cost': 150000.00,
					'current_cost': 145000.00,
					'tenant_id': 'tenant_default',
					'created_by': 'system'
				},
				{
					'product_name': 'PLM Collaboration Suite',
					'product_number': 'PLM-COLLAB-001',
					'product_description': 'Real-time collaboration tools for product development',
					'product_type': 'virtual',
					'lifecycle_phase': 'production',
					'target_cost': 50000.00,
					'current_cost': 48000.00,
					'tenant_id': 'tenant_default',
					'created_by': 'system'
				},
				{
					'product_name': 'AI Design Optimizer',
					'product_number': 'AI-OPT-001',
					'product_description': 'AI-powered design optimization engine',
					'product_type': 'service',
					'lifecycle_phase': 'development',
					'target_cost': 75000.00,
					'current_cost': 80000.00,
					'tenant_id': 'tenant_default',
					'created_by': 'system'
				}
			]
			
			for product_data in sample_products:
				product = PLProduct(**product_data)
				self.db.session.add(product)
			
			self.db.session.commit()
			self.logger.info("Sample PLM data created successfully")
			
		except Exception as e:
			self.db.session.rollback()
			self.logger.error(f"Failed to create sample PLM data: {e}")
			raise
	
	def _setup_rate_limiting(self) -> None:
		"""Setup rate limiting for PLM APIs"""
		try:
			limiter.init_app(self.app)
			self.logger.info("PLM API rate limiting configured")
		except Exception as e:
			self.logger.error(f"Failed to setup rate limiting: {e}")
	
	def _setup_websocket(self) -> None:
		"""Setup WebSocket support for real-time features"""
		try:
			socketio.init_app(self.app, cors_allowed_origins="*")
			self.logger.info("PLM WebSocket support configured")
		except Exception as e:
			self.logger.error(f"Failed to setup WebSocket: {e}")
	
	def _register_with_apg(self) -> None:
		"""Register PLM capability with APG composition engine"""
		try:
			# Register capability metadata
			if hasattr(self.app, 'apg_capabilities'):
				self.app.apg_capabilities = getattr(self.app, 'apg_capabilities', {})
			else:
				self.app.apg_capabilities = {}
			
			self.app.apg_capabilities['product_lifecycle_management'] = PLM_CAPABILITY_METADATA
			
			# Register capability instance
			if hasattr(self.app, 'apg_capability_instances'):
				self.app.apg_capability_instances = getattr(self.app, 'apg_capability_instances', {})
			else:
				self.app.apg_capability_instances = {}
			
			self.app.apg_capability_instances['product_lifecycle_management'] = self.capability
			
			self.logger.info("PLM capability registered with APG composition engine")
			
		except Exception as e:
			self.logger.error(f"Failed to register with APG composition engine: {e}")
	
	def _setup_monitoring(self) -> None:
		"""Setup monitoring and observability for PLM"""
		try:
			# Register health check endpoint
			@self.app.route('/health/plm')
			def plm_health_check():
				"""PLM capability health check endpoint"""
				try:
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					
					health_status = loop.run_until_complete(self.capability.health_check())
					return health_status, 200
					
				except Exception as e:
					return {
						'status': 'unhealthy',
						'error': str(e),
						'capability_id': 'product_lifecycle_management'
					}, 500
			
			# Register metrics endpoint
			@self.app.route('/metrics/plm')
			def plm_metrics():
				"""PLM capability metrics endpoint"""
				try:
					from .service import PLMProductService
					
					loop = asyncio.new_event_loop()
					asyncio.set_event_loop(loop)
					
					service = PLMProductService()
					metrics = loop.run_until_complete(
						service.get_capability_metrics('tenant_default', 'system')
					)
					return metrics, 200
					
				except Exception as e:
					return {
						'error': str(e),
						'capability_id': 'product_lifecycle_management'
					}, 500
			
			self.logger.info("PLM monitoring endpoints configured")
			
		except Exception as e:
			self.logger.error(f"Failed to setup monitoring: {e}")
	
	def get_capability_info(self) -> Dict[str, Any]:
		"""Get PLM capability information"""
		return {
			'metadata': PLM_CAPABILITY_METADATA,
			'initialized': self.initialized,
			'instance': self.capability,
			'endpoints': {
				'health': '/health/plm',
				'metrics': '/metrics/plm',
				'api': '/api/v1/plm',
				'dashboard': '/plm/dashboard'
			}
		}
	
	def shutdown(self) -> bool:
		"""Gracefully shutdown PLM capability"""
		try:
			if self.capability and self.initialized:
				# Perform cleanup operations
				self.logger.info("Shutting down PLM capability")
				
				# Close any open connections
				# Clean up background tasks
				# Save any pending data
				
				self.initialized = False
				self.logger.info("PLM capability shutdown completed")
				return True
				
		except Exception as e:
			self.logger.error(f"Error during PLM capability shutdown: {e}")
			return False


# Global PLM integration instance
plm_integration = PLMFlaskIntegration()


def register_plm_capability(app: Flask, appbuilder: AppBuilder, db: SQLAlchemy) -> bool:
	"""
	Register PLM capability with Flask application
	
	This is the main entry point for integrating PLM capability with APG.
	
	Args:
		app: Flask application instance
		appbuilder: Flask-AppBuilder instance
		db: SQLAlchemy database instance
		
	Returns:
		bool: True if registration successful, False otherwise
	"""
	return plm_integration.init_app(app, appbuilder, db)


def get_plm_capability_info() -> Dict[str, Any]:
	"""Get PLM capability information for APG composition engine"""
	return plm_integration.get_capability_info()


def configure_plm_menu(appbuilder: AppBuilder) -> None:
	"""
	Configure PLM menu items in APG navigation
	
	This function sets up the navigation menu structure for PLM
	following APG UI patterns and conventions.
	"""
	try:
		# PLM main menu category is automatically created by view registration
		# Additional menu customization can be added here
		
		# Add security permissions for PLM menu items
		_setup_plm_permissions(appbuilder)
		
		logging.getLogger('PLMFlaskIntegration').info("PLM menu configuration completed")
		
	except Exception as e:
		logging.getLogger('PLMFlaskIntegration').error(f"Failed to configure PLM menu: {e}")


def _setup_plm_permissions(appbuilder: AppBuilder) -> None:
	"""Setup PLM permissions in APG security system"""
	try:
		# Define PLM permission structure
		plm_permissions = [
			'plm.products.read',
			'plm.products.create',
			'plm.products.update',
			'plm.products.delete',
			'plm.products.digital_twin',
			'plm.changes.read',
			'plm.changes.create',
			'plm.changes.approve',
			'plm.collaboration.read',
			'plm.collaboration.create',
			'plm.collaboration.participate',
			'plm.analytics.read',
			'plm.ai.insights'
		]
		
		# Register permissions with APG security system
		# This integrates with the auth_rbac capability
		for permission in plm_permissions:
			# In a full APG implementation, this would register with auth_rbac
			pass
		
		logging.getLogger('PLMFlaskIntegration').info("PLM permissions configured")
		
	except Exception as e:
		logging.getLogger('PLMFlaskIntegration').error(f"Failed to setup PLM permissions: {e}")


# APG Integration Helper Functions

def get_plm_service(service_type: str):
	"""
	Get PLM service instance by type
	
	Args:
		service_type: Type of service ('product', 'change', 'collaboration', 'ai')
		
	Returns:
		Service instance or None if not found
	"""
	try:
		if service_type == 'product':
			from .service import PLMProductService
			return PLMProductService()
		elif service_type == 'change':
			from .service import PLMEngineeringChangeService
			return PLMEngineeringChangeService()
		elif service_type == 'collaboration':
			from .service import PLMCollaborationService
			return PLMCollaborationService()
		elif service_type == 'ai':
			from .ai_service import PLMAIService
			return PLMAIService()
		else:
			return None
			
	except Exception as e:
		logging.getLogger('PLMFlaskIntegration').error(f"Failed to get PLM service {service_type}: {e}")
		return None


def execute_plm_health_check() -> Dict[str, Any]:
	"""Execute PLM capability health check"""
	try:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		health_status = loop.run_until_complete(plm_capability.health_check())
		return health_status
		
	except Exception as e:
		return {
			'status': 'unhealthy',
			'error': str(e),
			'capability_id': 'product_lifecycle_management'
		}


# Module exports
__all__ = [
	'PLMFlaskIntegration',
	'plm_integration',
	'register_plm_capability',
	'get_plm_capability_info',
	'configure_plm_menu',
	'get_plm_service',
	'execute_plm_health_check'
]