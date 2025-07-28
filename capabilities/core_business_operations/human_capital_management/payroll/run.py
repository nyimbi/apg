"""
APG Payroll Management - Production Application Entry Point

Revolutionary payroll management system entry point with production
configuration, monitoring, and APG platform integration.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_appbuilder import AppBuilder, SQLA
from werkzeug.middleware.proxy_fix import ProxyFix

# APG Platform Imports
try:
	from ...config import APGConfig
	from ...auth_rbac.managers import APGSecurityManager
	from ...monitoring.middleware import APGMonitoringMiddleware
	from ...integration.manager import APGIntegrationManager
except ImportError:
	# Fallback for standalone deployment
	APGConfig = None
	APGSecurityManager = None
	APGMonitoringMiddleware = None
	APGIntegrationManager = None

# Payroll Module Imports
from .blueprint import create_payroll_blueprint
from .models import db
from .config import PayrollConfig

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(sys.stdout),
		logging.FileHandler('logs/payroll.log') if os.path.exists('logs') else logging.NullHandler()
	]
)

logger = logging.getLogger(__name__)


class PayrollApplication:
	"""Revolutionary APG Payroll Management Application."""
	
	def __init__(self, config_name: str = 'production'):
		"""Initialize the payroll application."""
		self.config_name = config_name
		self.app = None
		self.db = None
		self.appbuilder = None
		self.payroll_blueprint = None
		
		# Performance tracking
		self.start_time = datetime.utcnow()
		self.request_count = 0
		
	def create_app(self) -> Flask:
		"""Create and configure the Flask application."""
		try:
			logger.info(f"Creating APG Payroll Management application (config: {self.config_name})")
			
			# Create Flask app
			self.app = Flask(__name__)
			
			# Configure application
			self._configure_app()
			
			# Initialize extensions
			self._initialize_extensions()
			
			# Setup monitoring
			self._setup_monitoring()
			
			# Initialize APG platform integration
			self._initialize_apg_integration()
			
			# Initialize payroll blueprint
			self._initialize_payroll_blueprint()
			
			# Register error handlers
			self._register_error_handlers()
			
			# Register health checks
			self._register_health_checks()
			
			# Setup security
			self._setup_security()
			
			# Final configuration
			self._finalize_configuration()
			
			logger.info("APG Payroll Management application created successfully")
			return self.app
			
		except Exception as e:
			logger.error(f"Failed to create application: {e}")
			raise
	
	def _configure_app(self) -> None:
		"""Configure the Flask application."""
		try:
			# Load configuration
			if APGConfig:
				self.app.config.from_object(APGConfig(self.config_name))
			else:
				self.app.config.from_object(PayrollConfig.get_config(self.config_name))
			
			# Override with environment variables
			self.app.config.update({
				'SECRET_KEY': os.getenv('APG_SECRET_KEY', self.app.config.get('SECRET_KEY')),
				'SQLALCHEMY_DATABASE_URI': os.getenv('DATABASE_URL', self.app.config.get('SQLALCHEMY_DATABASE_URI')),
				'REDIS_URL': os.getenv('REDIS_URL', self.app.config.get('REDIS_URL')),
				'CELERY_BROKER_URL': os.getenv('CELERY_BROKER_URL', self.app.config.get('CELERY_BROKER_URL')),
			})
			
			# Production optimizations
			if self.config_name == 'production':
				self.app.config.update({
					'JSON_SORT_KEYS': False,
					'JSONIFY_PRETTYPRINT_REGULAR': False,
					'SEND_FILE_MAX_AGE_DEFAULT': 31536000,  # 1 year
				})
			
			logger.info("Application configuration loaded")
			
		except Exception as e:
			logger.error(f"Failed to configure application: {e}")
			raise
	
	def _initialize_extensions(self) -> None:
		"""Initialize Flask extensions."""
		try:
			# Initialize database
			self.db = SQLA(self.app)
			
			# Initialize AppBuilder with custom security manager
			if APGSecurityManager:
				self.appbuilder = AppBuilder(self.app, self.db.session, security_manager_class=APGSecurityManager)
			else:
				self.appbuilder = AppBuilder(self.app, self.db.session)
			
			# Configure proxy handling for production
			if self.config_name == 'production':
				self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
			
			logger.info("Extensions initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize extensions: {e}")
			raise
	
	def _setup_monitoring(self) -> None:
		"""Setup monitoring and observability."""
		try:
			# Initialize APG monitoring middleware
			if APGMonitoringMiddleware:
				APGMonitoringMiddleware(self.app)
			
			# Setup Prometheus metrics
			try:
				from prometheus_flask_exporter import PrometheusMetrics
				PrometheusMetrics(self.app)
			except ImportError:
				logger.warning("Prometheus metrics not available")
			
			# Setup Sentry error tracking
			if os.getenv('SENTRY_DSN'):
				try:
					import sentry_sdk
					from sentry_sdk.integrations.flask import FlaskIntegration
					from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
					
					sentry_sdk.init(
						dsn=os.getenv('SENTRY_DSN'),
						integrations=[
							FlaskIntegration(transaction_style='endpoint'),
							SqlalchemyIntegration()
						],
						traces_sample_rate=0.1,
						environment=self.config_name
					)
				except ImportError:
					logger.warning("Sentry SDK not available")
			
			logger.info("Monitoring setup completed")
			
		except Exception as e:
			logger.error(f"Failed to setup monitoring: {e}")
			# Monitoring is not critical, continue
			pass
	
	def _initialize_apg_integration(self) -> None:
		"""Initialize APG platform integration."""
		try:
			if APGIntegrationManager:
				self.apg_integration = APGIntegrationManager(
					app=self.app,
					capability_name='payroll_management'
				)
				logger.info("APG platform integration initialized")
			else:
				logger.info("Running in standalone mode without APG integration")
			
		except Exception as e:
			logger.error(f"Failed to initialize APG integration: {e}")
			# APG integration is optional, continue
			pass
	
	def _initialize_payroll_blueprint(self) -> None:
		"""Initialize the payroll management blueprint."""
		try:
			self.payroll_blueprint = create_payroll_blueprint(self.app, self.appbuilder)
			logger.info("Payroll blueprint initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize payroll blueprint: {e}")
			raise
	
	def _register_error_handlers(self) -> None:
		"""Register application error handlers."""
		
		@self.app.errorhandler(404)
		def not_found(error):
			return jsonify({
				'error': 'Resource not found',
				'status_code': 404,
				'timestamp': datetime.utcnow().isoformat()
			}), 404
		
		@self.app.errorhandler(500)
		def internal_error(error):
			logger.error(f"Internal server error: {error}")
			return jsonify({
				'error': 'Internal server error',
				'status_code': 500,
				'timestamp': datetime.utcnow().isoformat()
			}), 500
		
		@self.app.errorhandler(403)
		def forbidden(error):
			return jsonify({
				'error': 'Access forbidden',
				'status_code': 403,
				'timestamp': datetime.utcnow().isoformat()
			}), 403
		
		logger.info("Error handlers registered")
	
	def _register_health_checks(self) -> None:
		"""Register health check endpoints."""
		
		@self.app.route('/health')
		def health_check():
			"""Basic health check endpoint."""
			try:
				# Check database connectivity
				db_status = 'healthy'
				try:
					self.db.session.execute('SELECT 1')
					self.db.session.commit()
				except Exception as e:
					db_status = f'unhealthy: {str(e)}'
				
				# Check payroll services
				service_status = 'healthy'
				if self.payroll_blueprint:
					try:
						health = self.payroll_blueprint.health_check()
						if health.get('overall') != 'healthy':
							service_status = f"degraded: {health.get('overall')}"
					except Exception as e:
						service_status = f'error: {str(e)}'
				
				status = {
					'status': 'healthy' if db_status == 'healthy' and service_status == 'healthy' else 'degraded',
					'timestamp': datetime.utcnow().isoformat(),
					'version': '2.0.0-revolutionary',
					'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
					'request_count': self.request_count,
					'components': {
						'database': db_status,
						'payroll_services': service_status
					}
				}
				
				return jsonify(status), 200 if status['status'] == 'healthy' else 503
				
			except Exception as e:
				logger.error(f"Health check failed: {e}")
				return jsonify({
					'status': 'unhealthy',
					'error': str(e),
					'timestamp': datetime.utcnow().isoformat()
				}), 503
		
		@self.app.route('/health/detailed')
		def detailed_health_check():
			"""Detailed health check with component status."""
			try:
				if self.payroll_blueprint:
					return jsonify(self.payroll_blueprint.health_check())
				else:
					return jsonify({
						'status': 'degraded',
						'message': 'Payroll blueprint not initialized'
					}), 503
			except Exception as e:
				logger.error(f"Detailed health check failed: {e}")
				return jsonify({
					'status': 'error',
					'error': str(e),
					'timestamp': datetime.utcnow().isoformat()
				}), 500
		
		@self.app.route('/metrics')
		def metrics():
			"""Application metrics endpoint."""
			return jsonify({
				'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds(),
				'request_count': self.request_count,
				'memory_usage': self._get_memory_usage(),
				'timestamp': datetime.utcnow().isoformat()
			})
		
		# Track request count
		@self.app.before_request
		def track_requests():
			self.request_count += 1
		
		logger.info("Health check endpoints registered")
	
	def _setup_security(self) -> None:
		"""Setup security configurations."""
		try:
			# Security headers
			@self.app.after_request
			def set_security_headers(response):
				if self.config_name == 'production':
					response.headers['X-Content-Type-Options'] = 'nosniff'
					response.headers['X-Frame-Options'] = 'DENY'
					response.headers['X-XSS-Protection'] = '1; mode=block'
					response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
					response.headers['Content-Security-Policy'] = "default-src 'self'"
				return response
			
			logger.info("Security configuration completed")
			
		except Exception as e:
			logger.error(f"Failed to setup security: {e}")
			# Security setup failure is critical
			raise
	
	def _finalize_configuration(self) -> None:
		"""Finalize application configuration."""
		try:
			# Create database tables
			with self.app.app_context():
				self.db.create_all()
			
			# Log configuration summary
			logger.info(f"Application initialized with config: {self.config_name}")
			logger.info(f"Debug mode: {self.app.debug}")
			logger.info(f"Database URL: {self.app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')[:50]}...")
			
		except Exception as e:
			logger.error(f"Failed to finalize configuration: {e}")
			raise
	
	def _get_memory_usage(self) -> dict:
		"""Get current memory usage statistics."""
		try:
			import psutil
			process = psutil.Process()
			memory_info = process.memory_info()
			return {
				'rss': memory_info.rss,
				'vms': memory_info.vms,
				'percent': process.memory_percent()
			}
		except ImportError:
			return {'status': 'psutil not available'}
		except Exception as e:
			return {'error': str(e)}


# Configuration classes
class PayrollConfig:
	"""Payroll application configuration."""
	
	@staticmethod
	def get_config(config_name: str):
		"""Get configuration class by name."""
		configs = {
			'development': DevelopmentConfig,
			'testing': TestingConfig,
			'production': ProductionConfig
		}
		return configs.get(config_name, ProductionConfig)


class BaseConfig:
	"""Base configuration."""
	SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	SQLALCHEMY_RECORD_QUERIES = True
	
	# APG Platform Configuration
	APG_CAPABILITY_NAME = 'payroll_management'
	APG_CAPABILITY_VERSION = '2.0.0-revolutionary'
	
	# Redis Configuration
	REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
	
	# Celery Configuration
	CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
	CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')
	
	# Security Configuration
	WTF_CSRF_ENABLED = True
	WTF_CSRF_TIME_LIMIT = 3600


class DevelopmentConfig(BaseConfig):
	"""Development configuration."""
	DEBUG = True
	SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://localhost/apg_payroll_dev')
	LOG_LEVEL = 'DEBUG'


class TestingConfig(BaseConfig):
	"""Testing configuration."""
	TESTING = True
	SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
	WTF_CSRF_ENABLED = False
	LOG_LEVEL = 'DEBUG'


class ProductionConfig(BaseConfig):
	"""Production configuration."""
	DEBUG = False
	SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://localhost/apg_payroll')
	SQLALCHEMY_ENGINE_OPTIONS = {
		'pool_pre_ping': True,
		'pool_recycle': 300,
		'pool_size': 10,
		'max_overflow': 20
	}
	LOG_LEVEL = 'INFO'


# Application factory
def create_app(config_name: str = None) -> Flask:
	"""Application factory function."""
	if config_name is None:
		config_name = os.getenv('FLASK_ENV', 'production')
	
	app_instance = PayrollApplication(config_name)
	return app_instance.create_app()


# Create application instance
app = create_app()

# For Gunicorn
if __name__ == '__main__':
	port = int(os.getenv('PORT', 5000))
	debug = os.getenv('FLASK_ENV') == 'development'
	
	logger.info(f"Starting APG Payroll Management on port {port} (debug: {debug})")
	app.run(host='0.0.0.0', port=port, debug=debug)