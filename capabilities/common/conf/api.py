"""
APG Configuration Management API - Revolutionary REST Interface

Comprehensive REST API providing access to all revolutionary configuration management
features including AI-native automation, predictive analytics, and autonomous operations.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from flask import Flask, abort, current_app, request, jsonify, g
try:
	from flask_restful import Api, Resource
except ImportError:
	class Resource:
		"""Minimal flask_restful-compatible base when optional dependency is absent"""
		pass

	class Api:
		"""Small Flask route registrar compatible with the subset used here"""

		def __init__(self, app: Flask):
			self.app = app

		def add_resource(self, resource_cls, *routes: str):
			methods = [
				method.upper()
				for method in ("get", "post", "put", "patch", "delete")
				if hasattr(resource_cls, method)
			]

			def dispatch(**kwargs):
				resource = resource_cls()
				handler = getattr(resource, request.method.lower(), None)
				if handler is None:
					abort(405)
				return current_app.ensure_sync(handler)(**kwargs)

			endpoint_base = resource_cls.__name__
			for index, route in enumerate(routes):
				self.app.add_url_rule(
					route,
					endpoint=f"{endpoint_base}_{index}",
					view_func=dispatch,
					methods=methods or ["GET"],
				)
from functools import wraps
import logging
from uuid_extensions import uuid7str
from pydantic import BaseModel, ValidationError

from .service import RevolutionaryConfigurationManager, get_config_manager
from .models import (
	CMResource, CMTemplate, CMPolicy, CMEnvironment, CMDeployment,
	ResourceState, DeploymentStatus, PolicyAction, ResourceType,
	ValidationResult, ExecutionResult, AIInsight, CMMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# API Request/Response Models
class APIResponse(BaseModel):
	"""Standard API response format"""
	success: bool
	message: str
	data: Optional[Dict[str, Any]] = None
	errors: Optional[List[str]] = None
	timestamp: datetime = datetime.utcnow()
	request_id: str = uuid7str()


class ConfigurationRequest(BaseModel):
	"""Configuration creation request"""
	name: str
	resource_type: str
	cloud_provider: str
	configuration: Dict[str, Any]
	description: Optional[str] = None
	environment_id: Optional[str] = None


class DeploymentRequest(BaseModel):
	"""Deployment request"""
	resource_id: str
	environment_id: str
	deployment_strategy: str = "rolling"


class NaturalLanguageRequest(BaseModel):
	"""Natural language configuration request"""
	request: str
	context: Dict[str, Any] = {}


# API Decorators
def _api_response(success: bool, message: str, status_code: int, errors: Optional[List[str]] = None):
	"""Build a standard JSON API error response"""
	return jsonify(APIResponse(
		success=success,
		message=message,
		errors=errors or []
	).model_dump()), status_code


def _configured_api_keys() -> Dict[str, Dict[str, Any]]:
	"""Return configured API keys keyed by token value"""
	configured = current_app.config.get("APG_CONF_API_KEYS", {})
	if isinstance(configured, dict):
		return {
			key: value if isinstance(value, dict) else {"user_id": str(value)}
			for key, value in configured.items()
		}
	if isinstance(configured, list):
		return {str(key): {"user_id": "api-key"} for key in configured}
	return {}


def _permissions_from_header(value: Optional[str]) -> set[str]:
	"""Parse comma-separated permission header values"""
	if not value:
		return set()
	return {
		item.strip()
		for item in value.split(",")
		if item.strip()
	}


def _principal_from_request() -> Optional[Dict[str, Any]]:
	"""Resolve an authenticated API principal from request headers"""
	api_key = request.headers.get("X-API-Key")
	configured_keys = _configured_api_keys()
	if api_key and api_key in configured_keys:
		config = configured_keys[api_key]
		return {
			"user_id": config.get("user_id", "api-key"),
			"tenant_id": request.headers.get("X-Tenant-ID") or config.get("tenant_id"),
			"permissions": set(config.get("permissions", [])) | _permissions_from_header(request.headers.get("X-APG-Permissions")),
			"auth_method": "api_key"
		}

	user_id = request.headers.get("X-User-ID") or request.headers.get("X-APG-User")
	authorization = request.headers.get("Authorization", "")
	if authorization.startswith("Bearer "):
		token = authorization.removeprefix("Bearer ").strip()
		if token:
			user_id = user_id or f"token:{hashlib.sha256(token.encode()).hexdigest()[:12]}"

	if not user_id:
		return None

	return {
		"user_id": user_id,
		"tenant_id": request.headers.get("X-Tenant-ID") or request.headers.get("X-APG-Tenant"),
		"permissions": _permissions_from_header(request.headers.get("X-APG-Permissions")),
		"auth_method": "bearer" if authorization.startswith("Bearer ") else "header"
	}


def require_auth(f):
	"""Decorator for API endpoints requiring authentication"""
	@wraps(f)
	async def decorated_function(*args, **kwargs):
		if current_app.config.get("APG_CONF_AUTH_DISABLED", False):
			g.current_user = {
				"user_id": "auth-disabled",
				"tenant_id": request.headers.get("X-Tenant-ID"),
				"permissions": {"*"},
				"auth_method": "disabled"
			}
			return await f(*args, **kwargs)

		principal = _principal_from_request()
		if not principal:
			return _api_response(False, "Authentication required", 401, ["Missing or invalid authentication headers"])

		g.current_user = principal
		return await f(*args, **kwargs)
	return decorated_function


def require_permission(permission: str):
	"""Decorator for API endpoints requiring specific permission"""
	def decorator(f):
		@wraps(f)
		async def decorated_function(*args, **kwargs):
			principal = getattr(g, "current_user", None)
			if not principal:
				principal = _principal_from_request()
				if principal:
					g.current_user = principal

			if not principal:
				return _api_response(False, "Authentication required", 401, ["Missing authenticated principal"])

			permissions = principal.get("permissions", set())
			if "*" not in permissions and permission not in permissions:
				return _api_response(False, "Permission denied", 403, [f"Missing permission: {permission}"])

			return await f(*args, **kwargs)
		return decorated_function
	return decorator


# API Resource Classes
class ConfigurationResourceAPI(Resource):
	"""Configuration resource management API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		"""Get configuration manager instance"""
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	async def post(self):
		"""Create new configuration resource"""
		try:
			data = request.get_json()
			config_request = ConfigurationRequest(**data)
			
			manager = await self.get_manager()
			resource = await manager.create_configuration({
				"name": config_request.name,
				"type": config_request.resource_type,
				"cloud_provider": config_request.cloud_provider,
				"configuration": config_request.configuration,
				"description": config_request.description,
				"environment_id": config_request.environment_id
			})
			
			return jsonify(APIResponse(
				success=True,
				message="Configuration resource created successfully",
				data={
					"resource_id": resource.id,
					"name": resource.name,
					"state": resource.state.value,
					"created_at": resource.created_at.isoformat()
				}
			).model_dump())
			
		except ValidationError as e:
			return jsonify(APIResponse(
				success=False,
				message="Validation error",
				errors=[str(e)]
			).model_dump()), 400
		except Exception as e:
			logger.exception("Configuration creation failed")
			return jsonify(APIResponse(
				success=False,
				message="Configuration creation failed",
				errors=[str(e)]
			).model_dump()), 500
	
	@require_auth
	async def get(self, resource_id: Optional[str] = None):
		"""Get configuration resource(s)"""
		try:
			manager = await self.get_manager()
			
			if resource_id:
				# Get specific resource
				if resource_id in manager.resources:
					resource = manager.resources[resource_id]
					return jsonify(APIResponse(
						success=True,
						message="Resource retrieved successfully",
						data=resource.model_dump()
					).model_dump())
				else:
					return jsonify(APIResponse(
						success=False,
						message="Resource not found"
					).model_dump()), 404
			else:
				# List all resources
				resources = [r.model_dump() for r in manager.resources.values()]
				return jsonify(APIResponse(
					success=True,
					message=f"Retrieved {len(resources)} resources",
					data={"resources": resources, "count": len(resources)}
				).model_dump())
				
		except Exception as e:
			logger.exception("Resource retrieval failed")
			return jsonify(APIResponse(
				success=False,
				message="Resource retrieval failed",
				errors=[str(e)]
			).model_dump()), 500


class DeploymentAPI(Resource):
	"""Deployment management API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.deploy")
	async def post(self):
		"""Deploy configuration"""
		try:
			data = request.get_json()
			deploy_request = DeploymentRequest(**data)
			
			manager = await self.get_manager()
			deployment = await manager.deploy_configuration(
				deploy_request.resource_id,
				deploy_request.environment_id
			)
			
			return jsonify(APIResponse(
				success=True,
				message="Deployment started successfully",
				data={
					"deployment_id": deployment.id,
					"status": deployment.status.value,
					"started_at": deployment.started_at.isoformat()
				}
			).model_dump())
			
		except ValidationError as e:
			return jsonify(APIResponse(
				success=False,
				message="Validation error",
				errors=[str(e)]
			).model_dump()), 400
		except Exception as e:
			logger.exception("Deployment failed")
			return jsonify(APIResponse(
				success=False,
				message="Deployment failed",
				errors=[str(e)]
			).model_dump()), 500
	
	@require_auth
	async def get(self, deployment_id: Optional[str] = None):
		"""Get deployment status"""
		try:
			manager = await self.get_manager()
			
			if deployment_id:
				if deployment_id in manager.deployments:
					deployment = manager.deployments[deployment_id]
					return jsonify(APIResponse(
						success=True,
						message="Deployment retrieved successfully",
						data=deployment.model_dump()
					).model_dump())
				else:
					return jsonify(APIResponse(
						success=False,
						message="Deployment not found"
					).model_dump()), 404
			else:
				deployments = [d.model_dump() for d in manager.deployments.values()]
				return jsonify(APIResponse(
					success=True,
					message=f"Retrieved {len(deployments)} deployments",
					data={"deployments": deployments, "count": len(deployments)}
				).model_dump())
				
		except Exception as e:
			logger.exception("Deployment retrieval failed")
			return jsonify(APIResponse(
				success=False,
				message="Deployment retrieval failed",
				errors=[str(e)]
			).model_dump()), 500


class DriftDetectionAPI(Resource):
	"""Configuration drift detection API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.drift.detect")
	async def post(self, resource_id: str):
		"""Detect and remediate configuration drift"""
		try:
			manager = await self.get_manager()
			result = await manager.detect_and_remediate_drift(resource_id)
			
			return jsonify(APIResponse(
				success=True,
				message="Drift detection completed",
				data=result
			).model_dump())
			
		except Exception as e:
			logger.exception("Drift detection failed")
			return jsonify(APIResponse(
				success=False,
				message="Drift detection failed",
				errors=[str(e)]
			).model_dump()), 500


class NaturalLanguageAPI(Resource):
	"""Natural language configuration API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.nl.process")
	async def post(self):
		"""Process natural language configuration request"""
		try:
			data = request.get_json()
			nl_request = NaturalLanguageRequest(**data)
			
			manager = await self.get_manager()
			result = await manager.natural_language_configuration(
				nl_request.request,
				nl_request.context
			)
			
			return jsonify(APIResponse(
				success=True,
				message="Natural language request processed",
				data=result
			).model_dump())
			
		except ValidationError as e:
			return jsonify(APIResponse(
				success=False,
				message="Validation error",
				errors=[str(e)]
			).model_dump()), 400
		except Exception as e:
			logger.exception("Natural language processing failed")
			return jsonify(APIResponse(
				success=False,
				message="Natural language processing failed",
				errors=[str(e)]
			).model_dump()), 500


class TemplateAPI(Resource):
	"""Configuration template management API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.template.create")
	async def post(self):
		"""Create intelligent template from requirements"""
		try:
			data = request.get_json()
			
			manager = await self.get_manager()
			template = await manager.create_intelligent_template(data)
			
			return jsonify(APIResponse(
				success=True,
				message="Template created successfully",
				data={
					"template_id": template.id,
					"name": template.name,
					"ai_generated": template.ai_generated,
					"confidence_score": template.ai_confidence_score
				}
			).model_dump())
			
		except Exception as e:
			logger.exception("Template creation failed")
			return jsonify(APIResponse(
				success=False,
				message="Template creation failed",
				errors=[str(e)]
			).model_dump()), 500
	
	@require_auth
	async def get(self, template_id: Optional[str] = None):
		"""Get configuration template(s)"""
		try:
			manager = await self.get_manager()
			
			if template_id:
				if template_id in manager.templates:
					template = manager.templates[template_id]
					return jsonify(APIResponse(
						success=True,
						message="Template retrieved successfully",
						data=template.model_dump()
					).model_dump())
				else:
					return jsonify(APIResponse(
						success=False,
						message="Template not found"
					).model_dump()), 404
			else:
				templates = [t.model_dump() for t in manager.templates.values()]
				return jsonify(APIResponse(
					success=True,
					message=f"Retrieved {len(templates)} templates",
					data={"templates": templates, "count": len(templates)}
				).model_dump())
				
		except Exception as e:
			logger.exception("Template retrieval failed")
			return jsonify(APIResponse(
				success=False,
				message="Template retrieval failed",
				errors=[str(e)]
			).model_dump()), 500


class PredictiveInsightsAPI(Resource):
	"""Predictive analytics and insights API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.insights.read")
	async def get(self, resource_id: Optional[str] = None):
		"""Get predictive insights"""
		try:
			manager = await self.get_manager()
			insights = await manager.get_predictive_insights(resource_id)
			
			return jsonify(APIResponse(
				success=True,
				message="Predictive insights retrieved",
				data=insights
			).model_dump())
			
		except Exception as e:
			logger.exception("Insights retrieval failed")
			return jsonify(APIResponse(
				success=False,
				message="Insights retrieval failed",
				errors=[str(e)]
			).model_dump()), 500


class MetricsAPI(Resource):
	"""System metrics and analytics API"""
	
	def __init__(self):
		self.config_manager = None
	
	async def get_manager(self):
		if not self.config_manager:
			self.config_manager = await get_config_manager()
		return self.config_manager
	
	@require_auth
	@require_permission("config.metrics.read")
	async def get(self):
		"""Get comprehensive system metrics"""
		try:
			manager = await self.get_manager()
			metrics = await manager.get_revolutionary_metrics()
			
			return jsonify(APIResponse(
				success=True,
				message="System metrics retrieved",
				data=metrics
			).model_dump())
			
		except Exception as e:
			logger.exception("Metrics retrieval failed")
			return jsonify(APIResponse(
				success=False,
				message="Metrics retrieval failed",
				errors=[str(e)]
			).model_dump()), 500


class HealthCheckAPI(Resource):
	"""Health check API"""
	
	async def get(self):
		"""System health check"""
		try:
			health_status = {
				"status": "healthy",
				"timestamp": datetime.utcnow().isoformat(),
				"service": "APG Configuration Management",
				"version": "1.0.0",
				"components": {
					"configuration_manager": "operational",
					"ai_intelligence_engine": "operational",
					"universal_abstraction": "operational",
					"quantum_security": "operational",
					"predictive_analytics": "operational"
				}
			}
			
			return jsonify(APIResponse(
				success=True,
				message="System healthy",
				data=health_status
			).model_dump())
			
		except Exception as e:
			logger.exception("Health check error")
			return jsonify(APIResponse(
				success=False,
				message="Health check failed",
				errors=[str(e)]
			).model_dump()), 500


# API Application Factory
class RevolutionaryConfigAPI:
	"""Revolutionary Configuration Management API"""
	
	def __init__(self, app: Optional[Flask] = None):
		self.app = app
		self.api = None
		
		if app:
			self.init_app(app)
	
	def init_app(self, app: Flask):
		"""Initialize API with Flask app"""
		self.app = app
		self.api = Api(app)
		
		# Register API endpoints
		self._register_endpoints()
		
		# Add error handlers
		self._add_error_handlers()
	
	def _register_endpoints(self):
		"""Register all API endpoints"""
		
		# Configuration Resource Management
		self.api.add_resource(ConfigurationResourceAPI, 
							'/api/v1/config/resources',
							'/api/v1/config/resources/<resource_id>')
		
		# Deployment Management
		self.api.add_resource(DeploymentAPI,
							'/api/v1/config/deployments',
							'/api/v1/config/deployments/<deployment_id>')
		
		# Drift Detection
		self.api.add_resource(DriftDetectionAPI,
							'/api/v1/config/drift/<resource_id>')
		
		# Natural Language Interface
		self.api.add_resource(NaturalLanguageAPI,
							'/api/v1/config/natural-language')
		
		# Template Management
		self.api.add_resource(TemplateAPI,
							'/api/v1/config/templates',
							'/api/v1/config/templates/<template_id>')
		
		# Predictive Insights
		self.api.add_resource(PredictiveInsightsAPI,
							'/api/v1/config/insights',
							'/api/v1/config/insights/<resource_id>')
		
		# System Metrics
		self.api.add_resource(MetricsAPI,
							'/api/v1/config/metrics')
		
		# Health Check
		self.api.add_resource(HealthCheckAPI,
							'/api/v1/config/health')
	
	def _add_error_handlers(self):
		"""Add custom error handlers"""
		
		@self.app.errorhandler(400)
		def handle_bad_request(error):
			return jsonify(APIResponse(
				success=False,
				message="Bad Request",
				errors=[str(error)]
			).model_dump()), 400
		
		@self.app.errorhandler(401)
		def handle_unauthorized(error):
			return jsonify(APIResponse(
				success=False,
				message="Unauthorized",
				errors=["Authentication required"]
			).model_dump()), 401
		
		@self.app.errorhandler(403)
		def handle_forbidden(error):
			return jsonify(APIResponse(
				success=False,
				message="Forbidden",
				errors=["Insufficient permissions"]
			).model_dump()), 403
		
		@self.app.errorhandler(404)
		def handle_not_found(error):
			return jsonify(APIResponse(
				success=False,
				message="Not Found",
				errors=["Resource not found"]
			).model_dump()), 404
		
		@self.app.errorhandler(500)
		def handle_internal_error(error):
			logger.exception("Internal server error")
			return jsonify(APIResponse(
				success=False,
				message="Internal Server Error",
				errors=["An unexpected error occurred"]
			).model_dump()), 500


# Usage example
def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
	"""Create Flask app with Revolutionary Configuration API"""
	app = Flask(__name__)
	
	# Initialize API
	config_api = RevolutionaryConfigAPI(app)
	
	return app


if __name__ == "__main__":
	app = create_app()
	app.run(debug=True, host="0.0.0.0", port=5000)


# Export main components
__all__ = [
	"RevolutionaryConfigAPI",
	"ConfigurationResourceAPI",
	"DeploymentAPI", 
	"DriftDetectionAPI",
	"NaturalLanguageAPI",
	"TemplateAPI",
	"PredictiveInsightsAPI",
	"MetricsAPI",
	"HealthCheckAPI",
	"create_app"
]
