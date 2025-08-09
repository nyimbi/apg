"""
Main Application Entry Point for APG Real-Time Collaboration

Provides both FastAPI and Flask-AppBuilder integration with proper initialization.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# APG RTC imports
from .config import get_config
from .database import initialize_database, close_database, test_database_connection
from .websocket_manager import websocket_manager
from .api import router as rtc_router

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan manager"""
	logger.info("Starting APG Real-Time Collaboration service...")
	
	try:
		# Initialize configuration
		config = get_config()
		logger.info(f"Loaded configuration for environment: {config.environment}")
		
		# Initialize database
		await initialize_database()
		logger.info("Database initialized successfully")
		
		# Test database connection
		if not await test_database_connection():
			raise RuntimeError("Database connection failed")
		
		# Initialize WebSocket manager
		await websocket_manager.start()
		logger.info("WebSocket manager started")
		
		# Set up signal handlers for graceful shutdown
		def signal_handler(signum, frame):
			logger.info(f"Received signal {signum}, initiating graceful shutdown...")
			asyncio.create_task(shutdown())
		
		signal.signal(signal.SIGINT, signal_handler)
		signal.signal(signal.SIGTERM, signal_handler)
		
		logger.info("APG Real-Time Collaboration service started successfully")
		
		yield
		
	except Exception as e:
		logger.error(f"Failed to start application: {e}")
		raise
	
	finally:
		# Cleanup on shutdown
		await shutdown()


async def shutdown():
	"""Graceful shutdown"""
	logger.info("Shutting down APG Real-Time Collaboration service...")
	
	try:
		# Stop WebSocket manager
		await websocket_manager.stop()
		logger.info("WebSocket manager stopped")
		
		# Close database connections
		await close_database()
		logger.info("Database connections closed")
		
	except Exception as e:
		logger.error(f"Error during shutdown: {e}")
	
	logger.info("Shutdown complete")


def create_app() -> FastAPI:
	"""Create and configure FastAPI application"""
	config = get_config()
	
	# Create FastAPI app with lifespan
	app = FastAPI(
		title="APG Real-Time Collaboration",
		description="Revolutionary real-time collaboration with Teams/Zoom/Meet features and Flask-AppBuilder integration",
		version="1.0.0",
		lifespan=lifespan,
		debug=config.debug
	)
	
	# Add middleware
	app.add_middleware(
		CORSMiddleware,
		allow_origins=config.security.cors_origins,
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	
	app.add_middleware(
		TrustedHostMiddleware,
		allowed_hosts=config.security.allowed_hosts
	)
	
	# Include API router
	app.include_router(rtc_router)
	
	# Add root endpoint
	@app.get("/")
	async def root():
		"""Root endpoint with service information"""
		config = get_config()
		return {
			"service": "APG Real-Time Collaboration",
			"version": "1.0.0",
			"environment": config.environment,
			"status": "running",
			"features": {
				"page_collaboration": True,
				"video_calls": True,
				"screen_sharing": True,
				"recording": True,
				"teams_integration": config.third_party.teams_enabled,
				"zoom_integration": config.third_party.zoom_enabled,
				"google_meet_integration": config.third_party.google_meet_enabled,
				"ai_features": True,
				"analytics": True
			},
			"endpoints": {
				"api": "/api/v1/rtc",
				"websocket": "/api/v1/rtc/ws",
				"health": "/api/v1/rtc/health",
				"docs": "/docs"
			}
		}
	
	# Add health check endpoint at root level
	@app.get("/health")
	async def health_check():
		"""Simple health check"""
		try:
			# Test database connection
			db_healthy = await test_database_connection()
			
			# Test WebSocket manager
			ws_stats = websocket_manager.get_connection_stats()
			
			status = "healthy" if db_healthy else "unhealthy"
			
			return {
				"status": status,
				"database": "healthy" if db_healthy else "unhealthy",
				"websocket": "healthy",
				"connections": ws_stats.get("total_connections", 0),
				"timestamp": "2024-01-30T12:00:00Z"
			}
			
		except Exception as e:
			logger.error(f"Health check failed: {e}")
			return JSONResponse(
				status_code=503,
				content={
					"status": "unhealthy",
					"error": str(e),
					"timestamp": "2024-01-30T12:00:00Z"
				}
			)
	
	# Add exception handlers
	@app.exception_handler(HTTPException)
	async def http_exception_handler(request, exc):
		"""Handle HTTP exceptions"""
		return JSONResponse(
			status_code=exc.status_code,
			content={
				"error": exc.detail,
				"status_code": exc.status_code,
				"timestamp": "2024-01-30T12:00:00Z"
			}
		)
	
	@app.exception_handler(Exception)
	async def general_exception_handler(request, exc):
		"""Handle general exceptions"""
		logger.error(f"Unhandled exception: {exc}")
		return JSONResponse(
			status_code=500,
			content={
				"error": "Internal server error",
				"status_code": 500,
				"timestamp": "2024-01-30T12:00:00Z"
			}
		)
	
	return app


# Create the application instance
app = create_app()


# Flask-AppBuilder integration class
class RTCFlaskAppBuilderIntegration:
	"""Integration with Flask-AppBuilder"""
	
	def __init__(self, appbuilder=None):
		self.appbuilder = appbuilder
		self.config = get_config()
	
	def register_with_appbuilder(self, appbuilder):
		"""Register RTC capability with Flask-AppBuilder"""
		from .blueprint import real_time_collaboration_blueprint
		
		# Register blueprint
		if hasattr(appbuilder.app, 'register_blueprint'):
			blueprint = real_time_collaboration_blueprint.create_blueprint()
			appbuilder.app.register_blueprint(blueprint)
		
		# Register views
		real_time_collaboration_blueprint.register_with_appbuilder(appbuilder)
		
		# Initialize capability
		real_time_collaboration_blueprint.initialize_capability(appbuilder.app)
		
		logger.info("APG Real-Time Collaboration registered with Flask-AppBuilder")
	
	def get_collaboration_widget_html(self, page_url: str) -> str:
		"""Get collaboration widget HTML for Flask-AppBuilder pages"""
		return f"""
		<div id="rtc-collaboration-widget" data-page-url="{page_url}">
			<!-- Collaboration widget will be loaded here -->
			<script src="/static/rtc/collaboration-widget.js"></script>
		</div>
		"""


# CLI commands
class RTCCLICommands:
	"""Command-line interface commands"""
	
	@staticmethod
	async def init_database():
		"""Initialize database"""
		try:
			await initialize_database()
			print("✅ Database initialized successfully")
		except Exception as e:
			print(f"❌ Database initialization failed: {e}")
			sys.exit(1)
	
	@staticmethod
	async def test_database():
		"""Test database connection"""
		try:
			if await test_database_connection():
				print("✅ Database connection successful")
			else:
				print("❌ Database connection failed")
				sys.exit(1)
		except Exception as e:
			print(f"❌ Database test failed: {e}")
			sys.exit(1)
	
	@staticmethod
	async def start_websocket_server():
		"""Start standalone WebSocket server"""
		config = get_config()
		
		try:
			await websocket_manager.start()
			print(f"✅ WebSocket server started on {config.websocket.host}:{config.websocket.port}")
			
			# Keep running
			while True:
				await asyncio.sleep(1)
				
		except KeyboardInterrupt:
			print("\n🛑 Shutting down WebSocket server...")
			await websocket_manager.stop()
		except Exception as e:
			print(f"❌ WebSocket server failed: {e}")
			sys.exit(1)
	
	@staticmethod
	def start_api_server():
		"""Start API server"""
		config = get_config()
		
		uvicorn.run(
			"app:app",
			host=config.api.host,
			port=config.api.port,
			workers=1 if config.api.reload else config.api.workers,
			reload=config.api.reload,
			log_level=config.logging.level.lower(),
			access_log=True
		)
	
	@staticmethod
	async def validate_config():
		"""Validate configuration"""
		try:
			config = get_config()
			print("✅ Configuration loaded successfully")
			print(f"   Environment: {config.environment}")
			print(f"   Database URL: {config.database.url}")
			print(f"   Redis URL: {config.redis.url}")
			print(f"   API Host: {config.api.host}:{config.api.port}")
			print(f"   WebSocket Host: {config.websocket.host}:{config.websocket.port}")
			
			# Test third-party integrations
			if config.third_party.teams_enabled:
				print("✅ Teams integration enabled")
			if config.third_party.zoom_enabled:
				print("✅ Zoom integration enabled")
			if config.third_party.google_meet_enabled:
				print("✅ Google Meet integration enabled")
			
		except Exception as e:
			print(f"❌ Configuration validation failed: {e}")
			sys.exit(1)


# Development utilities
async def run_development_setup():
	"""Set up development environment"""
	print("🚀 Setting up APG Real-Time Collaboration development environment...")
	
	try:
		# Validate configuration
		await RTCCLICommands.validate_config()
		
		# Initialize database
		await RTCCLICommands.init_database()
		
		# Test database connection
		await RTCCLICommands.test_database()
		
		print("✅ Development environment setup complete!")
		print("\nNext steps:")
		print("1. Start the API server: python -m uvicorn app:app --reload")
		print("2. Start the WebSocket server: python -c 'import asyncio; from app import RTCCLICommands; asyncio.run(RTCCLICommands.start_websocket_server())'")
		print("3. Open http://localhost:8000 to view the API")
		print("4. Open http://localhost:8000/docs for API documentation")
		
	except Exception as e:
		print(f"❌ Development setup failed: {e}")
		sys.exit(1)


if __name__ == "__main__":
	import sys
	
	if len(sys.argv) > 1:
		command = sys.argv[1]
		
		if command == "init-db":
			asyncio.run(RTCCLICommands.init_database())
		elif command == "test-db":
			asyncio.run(RTCCLICommands.test_database())
		elif command == "start-ws":
			asyncio.run(RTCCLICommands.start_websocket_server())
		elif command == "validate-config":
			asyncio.run(RTCCLICommands.validate_config())
		elif command == "dev-setup":
			asyncio.run(run_development_setup())
		elif command == "serve":
			RTCCLICommands.start_api_server()
		else:
			print(f"Unknown command: {command}")
			print("Available commands:")
			print("  init-db         - Initialize database")
			print("  test-db         - Test database connection")
			print("  start-ws        - Start WebSocket server")
			print("  validate-config - Validate configuration")
			print("  dev-setup       - Set up development environment")
			print("  serve           - Start API server")
			sys.exit(1)
	else:
		# Default: start API server
		RTCCLICommands.start_api_server()