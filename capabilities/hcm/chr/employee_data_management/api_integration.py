"""
APG Employee Data Management - Flask API Integration

Flask Blueprint integration for the Employee API Gateway with
comprehensive REST API endpoints and OpenAPI documentation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, g
from flask_appbuilder import BaseView, expose, has_access
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized
import traceback

from .api_gateway import EmployeeAPIGateway, APIRequest, HTTPMethod
from .service import RevolutionaryEmployeeDataManagementService


# Create Flask Blueprint
employee_api_bp = Blueprint(
	'employee_api',
	__name__,
	url_prefix='/api/v1'
)


class EmployeeAPIView(BaseView):
	"""Flask-AppBuilder view for Employee API Gateway integration."""
	
	route_base = "/api/v1"
	
	def __init__(self):
		super().__init__()
		self.api_gateway = None
	
	def _get_api_gateway(self) -> EmployeeAPIGateway:
		"""Get or create API gateway instance."""
		if not self.api_gateway:
			tenant_id = self._get_tenant_id()
			self.api_gateway = EmployeeAPIGateway(tenant_id)
		return self.api_gateway
	
	def _get_tenant_id(self) -> str:
		"""Get tenant ID from request context."""
		# Would extract from user session or JWT token in production
		return "default_tenant"
	
	def _create_api_request(self, endpoint_path: str, method: HTTPMethod) -> APIRequest:
		"""Create API request object from Flask request."""
		return APIRequest(
			endpoint_path=endpoint_path,
			method=method,
			headers=dict(request.headers),
			query_params=dict(request.args),
			body=request.get_json(silent=True) if request.method in ['POST', 'PUT', 'PATCH'] else None,
			user_id=getattr(g, 'user_id', None),
			tenant_id=self._get_tenant_id(),
			client_ip=request.remote_addr
		)
	
	async def _handle_api_request(self, endpoint_path: str, method: HTTPMethod):
		"""Handle API request through gateway."""
		try:
			api_request = self._create_api_request(endpoint_path, method)
			gateway = self._get_api_gateway()
			
			response = await gateway.handle_request(api_request)
			
			# Convert APIResponse to Flask response
			flask_response = jsonify(response.data) if response.data else jsonify({})
			flask_response.status_code = response.status_code
			
			# Add custom headers
			for header_name, header_value in response.headers.items():
				flask_response.headers[header_name] = header_value
			
			# Add metadata headers
			flask_response.headers['X-Execution-Time'] = str(response.execution_time_ms)
			flask_response.headers['X-Cached'] = str(response.cached).lower()
			
			if response.error:
				return jsonify({'error': response.error}), response.status_code
			
			return flask_response
			
		except Exception as e:
			return jsonify({'error': f'Internal server error: {str(e)}'}), 500
	
	# Employee CRUD Endpoints
	@expose('/employees', methods=['GET'])
	@has_access
	def list_employees(self):
		"""List employees with filtering and pagination."""
		try:
			return asyncio.run(self._handle_api_request('/api/v1/employees', HTTPMethod.GET))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/employees', methods=['POST'])
	@has_access
	def create_employee(self):
		"""Create new employee."""
		try:
			return asyncio.run(self._handle_api_request('/api/v1/employees', HTTPMethod.POST))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/employees/<employee_id>', methods=['GET'])
	@has_access
	def get_employee(self, employee_id):
		"""Get employee by ID."""
		try:
			endpoint_path = f'/api/v1/employees/{employee_id}'
			return asyncio.run(self._handle_api_request(endpoint_path, HTTPMethod.GET))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/employees/<employee_id>', methods=['PUT'])
	@has_access
	def update_employee(self, employee_id):
		"""Update employee."""
		try:
			endpoint_path = f'/api/v1/employees/{employee_id}'
			return asyncio.run(self._handle_api_request(endpoint_path, HTTPMethod.PUT))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/employees/<employee_id>', methods=['DELETE'])
	@has_access
	def delete_employee(self, employee_id):
		"""Delete employee."""
		try:
			endpoint_path = f'/api/v1/employees/{employee_id}'
			return asyncio.run(self._handle_api_request(endpoint_path, HTTPMethod.DELETE))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	# AI and Analytics Endpoints
	@expose('/employees/<employee_id>/analyze', methods=['POST'])
	@has_access
	def analyze_employee(self, employee_id):
		"""Perform AI analysis on employee."""
		try:
			endpoint_path = f'/api/v1/employees/{employee_id}/analyze'
			return asyncio.run(self._handle_api_request(endpoint_path, HTTPMethod.POST))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	@expose('/analytics/dashboard', methods=['GET'])
	@has_access
	def analytics_dashboard(self):
		"""Get analytics dashboard data."""
		try:
			return asyncio.run(self._handle_api_request('/api/v1/analytics/dashboard', HTTPMethod.GET))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	# Integration Endpoints
	@expose('/integrations/sync', methods=['POST'])
	@has_access
	def integration_sync(self):
		"""Trigger integration synchronization."""
		try:
			return asyncio.run(self._handle_api_request('/api/v1/integrations/sync', HTTPMethod.POST))
		except Exception as e:
			return jsonify({'error': str(e)}), 500
	
	# Health and Status Endpoints
	@expose('/health', methods=['GET'])
	def health_check(self):
		"""API health check."""
		try:
			gateway = self._get_api_gateway()
			health_status = asyncio.run(gateway.health_check())
			return jsonify(health_status)
		except Exception as e:
			return jsonify({
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}), 500
	
	@expose('/stats', methods=['GET'])
	@has_access
	def api_statistics(self):
		"""Get API statistics."""
		try:
			gateway = self._get_api_gateway()
			stats = asyncio.run(gateway.get_api_statistics())
			return jsonify(stats)
		except Exception as e:
			return jsonify({'error': str(e)}), 500


# Blueprint route handlers (alternative approach)
@employee_api_bp.route('/employees', methods=['GET'])
def bp_list_employees():
	"""Blueprint route for listing employees."""
	try:
		gateway = EmployeeAPIGateway("default_tenant")
		api_request = APIRequest(
			endpoint_path='/api/v1/employees',
			method=HTTPMethod.GET,
			headers=dict(request.headers),
			query_params=dict(request.args),
			client_ip=request.remote_addr
		)
		
		response = asyncio.run(gateway.handle_request(api_request))
		
		if response.error:
			return jsonify({'error': response.error}), response.status_code
		
		flask_response = jsonify(response.data)
		flask_response.status_code = response.status_code
		flask_response.headers['X-Execution-Time'] = str(response.execution_time_ms)
		flask_response.headers['X-Cached'] = str(response.cached).lower()
		
		return flask_response
		
	except Exception as e:
		return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@employee_api_bp.route('/employees', methods=['POST'])
def bp_create_employee():
	"""Blueprint route for creating employee."""
	try:
		gateway = EmployeeAPIGateway("default_tenant")
		api_request = APIRequest(
			endpoint_path='/api/v1/employees',
			method=HTTPMethod.POST,
			headers=dict(request.headers),
			query_params=dict(request.args),
			body=request.get_json(),
			client_ip=request.remote_addr
		)
		
		response = asyncio.run(gateway.handle_request(api_request))
		
		if response.error:
			return jsonify({'error': response.error}), response.status_code
		
		flask_response = jsonify(response.data)
		flask_response.status_code = response.status_code
		flask_response.headers['X-Execution-Time'] = str(response.execution_time_ms)
		
		return flask_response
		
	except Exception as e:
		return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@employee_api_bp.route('/employees/<employee_id>', methods=['GET'])
def bp_get_employee(employee_id):
	"""Blueprint route for getting employee."""
	try:
		gateway = EmployeeAPIGateway("default_tenant")
		api_request = APIRequest(
			endpoint_path=f'/api/v1/employees/{employee_id}',
			method=HTTPMethod.GET,
			headers=dict(request.headers),
			query_params=dict(request.args),
			client_ip=request.remote_addr
		)
		
		response = asyncio.run(gateway.handle_request(api_request))
		
		if response.error:
			return jsonify({'error': response.error}), response.status_code
		
		flask_response = jsonify(response.data)
		flask_response.status_code = response.status_code
		flask_response.headers['X-Execution-Time'] = str(response.execution_time_ms)
		flask_response.headers['X-Cached'] = str(response.cached).lower()
		
		return flask_response
		
	except Exception as e:
		return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@employee_api_bp.route('/health', methods=['GET'])
def bp_health_check():
	"""Blueprint route for health check."""
	try:
		gateway = EmployeeAPIGateway("default_tenant")
		health_status = asyncio.run(gateway.health_check())
		return jsonify(health_status)
	except Exception as e:
		return jsonify({
			'status': 'unhealthy',
			'error': str(e),
			'timestamp': datetime.utcnow().isoformat()
		}), 500


# Error handlers
@employee_api_bp.errorhandler(400)
def handle_bad_request(error):
	"""Handle bad request errors."""
	return jsonify({
		'error': 'Bad Request',
		'message': str(error.description),
		'status_code': 400
	}), 400


@employee_api_bp.errorhandler(401)
def handle_unauthorized(error):
	"""Handle unauthorized errors."""
	return jsonify({
		'error': 'Unauthorized',
		'message': 'Authentication required',
		'status_code': 401
	}), 401


@employee_api_bp.errorhandler(404)
def handle_not_found(error):
	"""Handle not found errors."""
	return jsonify({
		'error': 'Not Found',
		'message': 'The requested resource was not found',
		'status_code': 404
	}), 404


@employee_api_bp.errorhandler(429)
def handle_rate_limit(error):
	"""Handle rate limit errors."""
	return jsonify({
		'error': 'Rate Limit Exceeded',
		'message': 'Too many requests, please try again later',
		'status_code': 429
	}), 429


@employee_api_bp.errorhandler(500)
def handle_internal_error(error):
	"""Handle internal server errors."""
	return jsonify({
		'error': 'Internal Server Error',
		'message': 'An unexpected error occurred',
		'status_code': 500,
		'trace': traceback.format_exc() if error else None
	}), 500


# OpenAPI/Swagger Documentation Generator
def generate_openapi_spec() -> Dict[str, Any]:
	"""Generate OpenAPI specification for the Employee API."""
	return {
		"openapi": "3.0.0",
		"info": {
			"title": "APG Employee Data Management API",
			"description": "Comprehensive API for employee data management with AI-powered insights",
			"version": "1.0.0",
			"contact": {
				"name": "Datacraft",
				"email": "nyimbi@gmail.com",
				"url": "https://www.datacraft.co.ke"
			}
		},
		"servers": [
			{
				"url": "/api/v1",
				"description": "Production API Server"
			}
		],
		"security": [
			{
				"bearerAuth": []
			}
		],
		"components": {
			"securitySchemes": {
				"bearerAuth": {
					"type": "http",
					"scheme": "bearer",
					"bearerFormat": "JWT"
				}
			},
			"schemas": {
				"Employee": {
					"type": "object",
					"required": ["first_name", "last_name", "work_email", "hire_date"],
					"properties": {
						"employee_id": {
							"type": "string",
							"format": "uuid",
							"description": "Unique employee identifier"
						},
						"employee_number": {
							"type": "string",
							"description": "Employee number (EMP######)"
						},
						"first_name": {
							"type": "string",
							"maxLength": 100,
							"description": "Employee first name"
						},
						"last_name": {
							"type": "string",
							"maxLength": 100,
							"description": "Employee last name"
						},
						"work_email": {
							"type": "string",
							"format": "email",
							"description": "Work email address"
						},
						"hire_date": {
							"type": "string",
							"format": "date",
							"description": "Employee hire date"
						},
						"department_id": {
							"type": "string",
							"description": "Department identifier"
						},
						"position_id": {
							"type": "string",
							"description": "Position identifier"
						},
						"employment_status": {
							"type": "string",
							"enum": ["Active", "Inactive", "Terminated", "On Leave", "Suspended"],
							"description": "Employment status"
						}
					}
				},
				"EmployeeList": {
					"type": "object",
					"properties": {
						"employees": {
							"type": "array",
							"items": {"$ref": "#/components/schemas/Employee"}
						},
						"total_count": {
							"type": "integer",
							"description": "Total number of employees"
						},
						"page": {
							"type": "integer",
							"description": "Current page number"
						},
						"limit": {
							"type": "integer",
							"description": "Number of items per page"
						}
					}
				},
				"AIAnalysisResult": {
					"type": "object",
					"properties": {
						"employee_id": {
							"type": "string",
							"description": "Employee identifier"
						},
						"retention_risk_score": {
							"type": "number",
							"minimum": 0,
							"maximum": 1,
							"description": "Retention risk score (0-1)"
						},
						"performance_prediction": {
							"type": "number",
							"minimum": 0,
							"maximum": 1,
							"description": "Performance prediction score"
						},
						"career_recommendations": {
							"type": "array",
							"items": {"type": "string"},
							"description": "AI-generated career recommendations"
						},
						"skills_analysis": {
							"type": "object",
							"description": "Skills gap analysis"
						}
					}
				},
				"Error": {
					"type": "object",
					"properties": {
						"error": {
							"type": "string",
							"description": "Error message"
						},
						"status_code": {
							"type": "integer",
							"description": "HTTP status code"
						},
						"timestamp": {
							"type": "string",
							"format": "date-time",
							"description": "Error timestamp"
						}
					}
				}
			}
		},
		"paths": {
			"/employees": {
				"get": {
					"summary": "List employees",
					"description": "Retrieve a list of employees with optional filtering and pagination",
					"tags": ["Employees"],
					"parameters": [
						{
							"name": "page",
							"in": "query",
							"description": "Page number",
							"schema": {"type": "integer", "default": 1}
						},
						{
							"name": "limit",
							"in": "query",
							"description": "Number of items per page",
							"schema": {"type": "integer", "default": 50, "maximum": 100}
						},
						{
							"name": "search",
							"in": "query",
							"description": "Search term",
							"schema": {"type": "string"}
						},
						{
							"name": "department",
							"in": "query",
							"description": "Filter by department ID",
							"schema": {"type": "string"}
						}
					],
					"responses": {
						"200": {
							"description": "List of employees",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/EmployeeList"}
								}
							}
						},
						"400": {
							"description": "Bad request",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Error"}
								}
							}
						}
					}
				},
				"post": {
					"summary": "Create employee",
					"description": "Create a new employee record",
					"tags": ["Employees"],
					"requestBody": {
						"required": True,
						"content": {
							"application/json": {
								"schema": {"$ref": "#/components/schemas/Employee"}
							}
						}
					},
					"responses": {
						"201": {
							"description": "Employee created successfully",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Employee"}
								}
							}
						},
						"400": {
							"description": "Invalid input",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Error"}
								}
							}
						}
					}
				}
			},
			"/employees/{employee_id}": {
				"get": {
					"summary": "Get employee",
					"description": "Retrieve a specific employee by ID",
					"tags": ["Employees"],
					"parameters": [
						{
							"name": "employee_id",
							"in": "path",
							"required": True,
							"description": "Employee ID",
							"schema": {"type": "string"}
						}
					],
					"responses": {
						"200": {
							"description": "Employee details",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Employee"}
								}
							}
						},
						"404": {
							"description": "Employee not found",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Error"}
								}
							}
						}
					}
				}
			},
			"/employees/{employee_id}/analyze": {
				"post": {
					"summary": "AI analyze employee",
					"description": "Perform AI analysis on employee data",
					"tags": ["AI Analytics"],
					"parameters": [
						{
							"name": "employee_id",
							"in": "path",
							"required": True,
							"description": "Employee ID",
							"schema": {"type": "string"}
						}
					],
					"responses": {
						"200": {
							"description": "AI analysis results",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/AIAnalysisResult"}
								}
							}
						},
						"404": {
							"description": "Employee not found",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/Error"}
								}
							}
						}
					}
				}
			},
			"/health": {
				"get": {
					"summary": "Health check",
					"description": "Check API health status",
					"tags": ["System"],
					"responses": {
						"200": {
							"description": "Health status",
							"content": {
								"application/json": {
									"schema": {
										"type": "object",
										"properties": {
											"status": {"type": "string"},
											"timestamp": {"type": "string"},
											"services": {"type": "object"},
											"integrations": {"type": "object"}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}


@employee_api_bp.route('/openapi.json', methods=['GET'])
def openapi_spec():
	"""Serve OpenAPI specification."""
	return jsonify(generate_openapi_spec())


# Documentation endpoint
@employee_api_bp.route('/docs', methods=['GET'])
def api_documentation():
	"""Serve API documentation."""
	return """
	<!DOCTYPE html>
	<html>
	<head>
		<title>APG Employee API Documentation</title>
		<link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css" />
		<style>
			html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
			*, *:before, *:after { box-sizing: inherit; }
			body { margin:0; background: #fafafa; }
		</style>
	</head>
	<body>
		<div id="swagger-ui"></div>
		<script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
		<script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-standalone-preset.js"></script>
		<script>
		window.onload = function() {
			SwaggerUIBundle({
				url: '/api/v1/openapi.json',
				dom_id: '#swagger-ui',
				deepLinking: true,
				presets: [
					SwaggerUIBundle.presets.apis,
					SwaggerUIStandalonePreset
				],
				plugins: [
					SwaggerUIBundle.plugins.DownloadUrl
				],
				layout: "StandaloneLayout"
			});
		};
		</script>
	</body>
	</html>
	"""