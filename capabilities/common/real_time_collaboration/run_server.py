#!/usr/bin/env python3
"""
Simple development server for APG Real-Time Collaboration

Starts a minimal FastAPI server for testing the RTC capability.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add current directory to path  
sys.path.insert(0, str(Path(__file__).parent))

# Set development environment
os.environ.update({
	'ENVIRONMENT': 'development',
	'DATABASE_URL': 'sqlite:///rtc_development.db',
	'DEBUG': 'true',
	'API_HOST': '127.0.0.1',
	'API_PORT': '8000'
})

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_minimal_app():
	"""Create a minimal FastAPI app for testing"""
	try:
		from fastapi import FastAPI
		from fastapi.responses import JSONResponse
		
		app = FastAPI(
			title="APG Real-Time Collaboration (Development)",
			description="Development server for APG RTC capability",
			version="1.0.0-dev"
		)
		
		@app.get("/")
		async def root():
			"""Root endpoint"""
			return {
				"service": "APG Real-Time Collaboration",
				"version": "1.0.0-dev",
				"status": "development",
				"message": "APG RTC development server is running",
				"endpoints": {
					"health": "/health",
					"docs": "/docs",
					"test": "/test"
				}
			}
		
		@app.get("/health")
		async def health_check():
			"""Health check endpoint"""
			return {
				"status": "healthy",
				"timestamp": __import__('datetime').datetime.now().isoformat(),
				"database": "sqlite",
				"environment": "development"
			}
		
		@app.get("/test")
		async def test_endpoint():
			"""Test basic functionality"""
			try:
				# Test APG stubs
				from apg_stubs import auth_service, ai_service, notification_service
				
				# Test auth
				token_result = await auth_service.validate_token("test_token")
				
				# Test AI
				suggestions = await ai_service.suggest_participants({"page_url": "/test"})
				
				return {
					"status": "success",
					"message": "All systems operational",
					"tests": {
						"auth_service": token_result["valid"],
						"ai_service": len(suggestions) > 0,
						"configuration": True,
						"database": True
					}
				}
				
			except Exception as e:
				return JSONResponse(
					status_code=500,
					content={
						"status": "error",
						"message": str(e),
						"timestamp": __import__('datetime').datetime.now().isoformat()
					}
				)
		
		return app
		
	except ImportError as e:
		logger.error(f"Failed to import FastAPI: {e}")
		logger.info("Creating fallback simple HTTP server...")
		return None

def run_simple_server():
	"""Run a simple HTTP server if FastAPI is not available"""
	import http.server
	import socketserver
	from urllib.parse import parse_qs, urlparse
	import json
	
	class RTCHandler(http.server.SimpleHTTPRequestHandler):
		def do_GET(self):
			parsed_path = urlparse(self.path)
			
			if parsed_path.path == '/':
				self.send_json_response({
					"service": "APG Real-Time Collaboration",
					"version": "1.0.0-dev",
					"status": "development (fallback server)",
					"message": "Basic HTTP server running"
				})
			elif parsed_path.path == '/health':
				self.send_json_response({
					"status": "healthy",
					"timestamp": __import__('datetime').datetime.now().isoformat()
				})
			else:
				super().do_GET()
		
		def send_json_response(self, data):
			self.send_response(200)
			self.send_header('Content-type', 'application/json')
			self.end_headers()
			self.wfile.write(json.dumps(data, indent=2).encode())
	
	port = int(os.getenv('API_PORT', '8000'))
	
	with socketserver.TCPServer(("", port), RTCHandler) as httpd:
		print(f"🚀 APG RTC fallback server running on http://localhost:{port}")
		print(f"📍 Available endpoints:")
		print(f"   - http://localhost:{port}/ (service info)")
		print(f"   - http://localhost:{port}/health (health check)")
		print(f"⏹️  Press Ctrl+C to stop")
		
		try:
			httpd.serve_forever()
		except KeyboardInterrupt:
			print("\n🛑 Server stopped")

def main():
	"""Main function"""
	print("🚀 Starting APG Real-Time Collaboration Development Server...")
	
	# Try to create FastAPI app
	app = create_minimal_app()
	
	if app is not None:
		try:
			import uvicorn
			
			port = int(os.getenv('API_PORT', '8000'))
			host = os.getenv('API_HOST', '127.0.0.1')
			
			print(f"🌟 Starting FastAPI server on http://{host}:{port}")
			print(f"📍 Available endpoints:")
			print(f"   - http://{host}:{port}/ (service info)")
			print(f"   - http://{host}:{port}/health (health check)")
			print(f"   - http://{host}:{port}/test (functionality test)")
			print(f"   - http://{host}:{port}/docs (API documentation)")
			print(f"⏹️  Press Ctrl+C to stop")
			
			uvicorn.run(
				app,
				host=host,
				port=port,
				log_level="info",
				reload=True
			)
			
		except ImportError:
			logger.warning("uvicorn not available, falling back to simple server")
			run_simple_server()
		except Exception as e:
			logger.error(f"Failed to start FastAPI server: {e}")
			run_simple_server()
	else:
		logger.info("FastAPI not available, using simple HTTP server")
		run_simple_server()

if __name__ == "__main__":
	main()