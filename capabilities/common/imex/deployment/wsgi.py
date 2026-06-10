#!/usr/bin/env python3
"""
APG Import/Export (IMEX) WSGI Production Application

Purpose: Production WSGI application entry point for APG IMEX capability
         with comprehensive configuration, monitoring, and error handling.
Dependencies: flask, gunicorn, production_config
Usage Context: Production deployment entry point

This module provides:
- Production WSGI application factory
- Configuration management integration
- Health check endpoints
- Metrics and monitoring endpoints
- Error handling and logging setup
- Security middleware integration
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(current_dir))

from flask import Flask, jsonify, request
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app, *args, **kwargs):
        return app

# Import IMEX components
from models import *
from database import DatabaseManager, DatabaseConfig
from ai_intelligence import AIIntelligenceEngine
from service import ImportExportService
from views_simple import imex_views_bp, set_imex_service
from security import security_middleware, create_security_config, AuthenticationManager
from api_secure import secure_api_bp, initialize_secure_api
from performance import PerformanceMonitor

# Import deployment configuration
from deployment.production_config import create_production_config, ProductionConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
app: Flask = None
config: ProductionConfig = None
service: ImportExportService = None
performance_monitor: PerformanceMonitor = None

def create_app(config_override: ProductionConfig = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_override: Optional configuration override

    Returns:
        Flask: Configured Flask application
    """
    global config, service, performance_monitor

    # Create Flask app
    app = Flask(__name__)

    # Load configuration
    environment = os.getenv('APG_ENVIRONMENT', 'production')
    config = config_override or create_production_config(environment)

    # Configure Flask
    app.config.update({
        'SECRET_KEY': config.security.secret_key,
        'DEBUG': config.debug,
        'TESTING': config.testing,
        'MAX_CONTENT_LENGTH': config.max_content_length,
        'UPLOAD_FOLDER': config.upload_folder,
        'JSON_SORT_KEYS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': not config.environment.value == 'production'
    })

    # Setup CORS
    if config.security.allowed_origins:
        CORS(app, origins=config.security.allowed_origins)
    else:
        CORS(app)

    # Initialize components
    try:
        # Setup performance monitoring
        performance_monitor = PerformanceMonitor(
            collection_interval=config.monitoring.health_check_interval
        )
        performance_monitor.start_monitoring()
        logger.info("Performance monitoring started")

        # Setup database manager
        db_manager = DatabaseManager(config.database)

        # Setup AI engine
        ai_engine = AIIntelligenceEngine()

        # Setup main service
        service = ImportExportService(db_manager, ai_engine)

        # Initialize service asynchronously
        def initialize_service():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(ai_engine.initialize())
                loop.run_until_complete(service.initialize())
                logger.info("IMEX service initialized successfully")
            except Exception as e:
                logger.error(f"Service initialization failed: {e}")
            finally:
                loop.close()

        # Run initialization in thread to avoid blocking
        import threading
        init_thread = threading.Thread(target=initialize_service)
        init_thread.start()

        # Setup security
        auth_manager = AuthenticationManager(
            create_security_config(config.environment.value)
        )
        app.auth_manager = auth_manager

        # Apply security middleware
        security_middleware(app)

        # Register UI views
        set_imex_service(service)
        app.register_blueprint(imex_views_bp)

        # Register secure API
        initialize_secure_api(service, config.environment.value)
        app.register_blueprint(secure_api_bp)

        logger.info(f"APG IMEX application created for {config.environment.value}")

    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise

    return app

def setup_health_endpoints(app: Flask):
    """Setup health check and monitoring endpoints."""

    @app.route('/health')
    def health_check():
        """Basic health check endpoint."""
        try:
            # Check service health
            service_healthy = service is not None

            # Check performance monitoring
            monitoring_healthy = (
                performance_monitor is not None and
                hasattr(performance_monitor, '_monitoring_active')
            )

            # Check database connectivity (if available)
            db_healthy = True
            if service and hasattr(service, 'db_manager'):
                try:
                    # Simple connectivity check
                    db_healthy = service.db_manager is not None
                except Exception:
                    db_healthy = False

            health_status = {
                'status': 'healthy' if all([service_healthy, monitoring_healthy, db_healthy]) else 'degraded',
                'timestamp': performance_monitor.get_system_metrics_summary(hours=0.1) if performance_monitor else {},
                'service': 'running' if service_healthy else 'error',
                'monitoring': 'running' if monitoring_healthy else 'error',
                'database': 'connected' if db_healthy else 'disconnected',
                'version': config.app_version if config else '1.0.0'
            }

            status_code = 200 if health_status['status'] == 'healthy' else 503
            return jsonify(health_status), status_code

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': None
            }), 503

    @app.route('/ready')
    def readiness_check():
        """Kubernetes readiness probe endpoint."""
        try:
            # Check if service is fully initialized
            if service is None:
                return jsonify({'status': 'not_ready', 'reason': 'service_not_initialized'}), 503

            # Check if AI engine is ready
            if not hasattr(service, 'ai_engine') or service.ai_engine is None:
                return jsonify({'status': 'not_ready', 'reason': 'ai_engine_not_ready'}), 503

            return jsonify({
                'status': 'ready',
                'timestamp': performance_monitor.get_system_metrics_summary(hours=0.1) if performance_monitor else {}
            }), 200

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return jsonify({
                'status': 'not_ready',
                'error': str(e)
            }), 503

    @app.route('/metrics')
    def metrics_endpoint():
        """Prometheus metrics endpoint."""
        try:
            if not performance_monitor:
                return "# Metrics not available\n", 503

            # Get performance statistics
            stats = performance_monitor.get_performance_statistics()
            system_metrics = performance_monitor.get_system_metrics_summary(hours=1)

            # Generate Prometheus format metrics
            metrics_output = []

            # Add system metrics
            if 'metrics' in system_metrics:
                for metric_name, metric_data in system_metrics['metrics'].items():
                    if isinstance(metric_data, dict) and 'current' in metric_data:
                        metrics_output.append(f"imex_{metric_name}_current {metric_data['current']}")
                        metrics_output.append(f"imex_{metric_name}_average {metric_data.get('average', 0)}")

            # Add monitoring stats
            if 'metrics_summary' in stats:
                summary = stats['metrics_summary']
                metrics_output.append(f"imex_total_metrics_collected {summary.get('total_metrics_collected', 0)}")
                metrics_output.append(f"imex_active_alerts {summary.get('active_alerts', 0)}")
                metrics_output.append(f"imex_jobs_monitored {summary.get('jobs_monitored', 0)}")

            # Add service metrics
            if service:
                metrics_output.append(f"imex_active_jobs {len(getattr(service, 'active_jobs', {}))}")
                metrics_output.append(f"imex_job_executions {len(getattr(service, 'job_executions', {}))}")

            # Add application info
            metrics_output.append(f'imex_info{{version="{config.app_version if config else "1.0.0"}"}} 1')

            return '\n'.join(metrics_output) + '\n', 200, {'Content-Type': 'text/plain'}

        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return f"# Error generating metrics: {e}\n", 500, {'Content-Type': 'text/plain'}

    @app.route('/info')
    def info_endpoint():
        """Application information endpoint."""
        try:
            return jsonify({
                'name': config.app_name if config else 'APG-IMEX',
                'version': config.app_version if config else '1.0.0',
                'environment': config.environment.value if config else 'unknown',
                'build_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                },
                'features': {
                    'ai_enabled': service and hasattr(service, 'ai_engine') and service.ai_engine is not None,
                    'performance_monitoring': performance_monitor is not None,
                    'security_enabled': hasattr(app, 'auth_manager'),
                    'database_connected': service and hasattr(service, 'db_manager') and service.db_manager is not None
                }
            })
        except Exception as e:
            logger.error(f"Info endpoint failed: {e}")
            return jsonify({'error': str(e)}), 500

def create_application() -> Flask:
    """
    Application factory for production deployment.

    Returns:
        Flask: Configured production Flask application
    """
    try:
        # Create the main application
        flask_app = create_app()

        # Setup monitoring and health endpoints
        setup_health_endpoints(flask_app)

        # Add error handlers
        @flask_app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Resource not found'}), 404

        @flask_app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({'error': 'Internal server error'}), 500

        # Add request logging for production
        @flask_app.before_request
        def log_request_info():
            if not flask_app.config.get('TESTING'):
                logger.debug(f"Request: {request.method} {request.url}")

        logger.info("APG IMEX production application ready")
        return flask_app

    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise

# Create the WSGI application
try:
    application = create_application()
    logger.info("WSGI application created successfully")
except Exception as e:
    logger.error(f"WSGI application creation failed: {e}")
    # Create a minimal error application
    application = Flask(__name__)

    @application.route('/')
    def error_page():
        return jsonify({
            'error': 'Application initialization failed',
            'message': str(e),
            'status': 'error'
        }), 503

# For development server
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('APG_ENVIRONMENT', 'production') != 'production'

    logger.info(f"Starting development server on port {port}")
    application.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )
