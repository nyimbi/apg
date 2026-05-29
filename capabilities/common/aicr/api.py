"""
REST API Endpoints for the AI Core Framework (AICR) Capability
==============================================================

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

Comprehensive REST API providing programmatic access to all AICR functionality
including model management, inference execution, pipeline orchestration,
monitoring, and administrative operations with full OpenAPI documentation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from flask import Blueprint, request, jsonify, g
from flask_restx import Api, Resource, fields, Namespace
from flask_restx.reqparse import RequestParser
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
from werkzeug.security import check_password_hash

from .service import AICoreService
from .models import AICRModel, AICRInferenceRequest, AICRPipeline
from .monitoring import ai_monitoring_system
from .ml_pipeline import ml_pipeline_framework
from .model_marketplace import model_marketplace
from .security import SecurityManager


# Create Flask-RESTX API
api_v1 = Api(
	version='1.0',
	title='AI Core Framework API',
	description='Comprehensive REST API for the AI Core Framework (AICR) capability',
	doc='/docs/',
	prefix='/api/v1'
)

# API Namespaces
models_ns = Namespace('models', description='AI Model operations')
inference_ns = Namespace('inference', description='Model inference operations')
pipelines_ns = Namespace('pipelines', description='ML Pipeline operations')
monitoring_ns = Namespace('monitoring', description='System monitoring operations')
marketplace_ns = Namespace('marketplace', description='Model marketplace operations')
admin_ns = Namespace('admin', description='Administrative operations')

api_v1.add_namespace(models_ns, path='/models')
api_v1.add_namespace(inference_ns, path='/inference')
api_v1.add_namespace(pipelines_ns, path='/pipelines')
api_v1.add_namespace(monitoring_ns, path='/monitoring')
api_v1.add_namespace(marketplace_ns, path='/marketplace')
api_v1.add_namespace(admin_ns, path='/admin')


# Request/Response Models
model_input = api_v1.model('ModelInput', {
	'name': fields.String(required=True, description='Model name'),
	'description': fields.String(description='Model description'),
	'model_type': fields.String(required=True, description='Type of model'),
	'framework': fields.String(required=True, description='ML framework'),
	'version': fields.String(description='Model version'),
	'file_path': fields.String(description='Path to model file'),
	'configuration': fields.Raw(description='Model configuration'),
	'metadata': fields.Raw(description='Additional metadata')
})

model_output = api_v1.model('ModelOutput', {
	'model_id': fields.String(description='Unique model identifier'),
	'name': fields.String(description='Model name'),
	'description': fields.String(description='Model description'),
	'model_type': fields.String(description='Type of model'),
	'framework': fields.String(description='ML framework'),
	'version': fields.String(description='Model version'),
	'status': fields.String(description='Model status'),
	'file_path': fields.String(description='Path to model file'),
	'configuration': fields.Raw(description='Model configuration'),
	'performance_metrics': fields.Raw(description='Performance metrics'),
	'created_at': fields.DateTime(description='Creation timestamp'),
	'updated_at': fields.DateTime(description='Last update timestamp')
})

inference_request = api_v1.model('InferenceRequest', {
	'model_id': fields.String(required=True, description='Model identifier'),
	'input_data': fields.Raw(required=True, description='Input data for inference'),
	'parameters': fields.Raw(description='Additional inference parameters'),
	'output_format': fields.String(description='Desired output format')
})

inference_response = api_v1.model('InferenceResponse', {
	'request_id': fields.String(description='Request identifier'),
	'model_id': fields.String(description='Model identifier'),
	'predictions': fields.Raw(description='Model predictions'),
	'confidence_scores': fields.Raw(description='Prediction confidence scores'),
	'processing_time_ms': fields.Float(description='Processing time in milliseconds'),
	'metadata': fields.Raw(description='Additional response metadata'),
	'timestamp': fields.DateTime(description='Response timestamp')
})

pipeline_input = api_v1.model('PipelineInput', {
	'name': fields.String(required=True, description='Pipeline name'),
	'description': fields.String(description='Pipeline description'),
	'pipeline_type': fields.String(required=True, description='Type of pipeline'),
	'stages': fields.List(fields.Raw, description='Pipeline stages configuration'),
	'training_config': fields.Raw(description='Training configuration'),
	'automl_config': fields.Raw(description='AutoML configuration'),
	'data_sources': fields.List(fields.String, description='Data source identifiers'),
	'schedule': fields.String(description='Execution schedule')
})

pipeline_output = api_v1.model('PipelineOutput', {
	'pipeline_id': fields.String(description='Pipeline identifier'),
	'name': fields.String(description='Pipeline name'),
	'description': fields.String(description='Pipeline description'),
	'pipeline_type': fields.String(description='Type of pipeline'),
	'status': fields.String(description='Pipeline status'),
	'stages_count': fields.Integer(description='Number of stages'),
	'execution_count': fields.Integer(description='Total executions'),
	'success_rate': fields.Float(description='Success rate percentage'),
	'last_execution': fields.DateTime(description='Last execution timestamp'),
	'created_at': fields.DateTime(description='Creation timestamp')
})

execution_output = api_v1.model('ExecutionOutput', {
	'execution_id': fields.String(description='Execution identifier'),
	'pipeline_id': fields.String(description='Pipeline identifier'),
	'status': fields.String(description='Execution status'),
	'started_at': fields.DateTime(description='Start timestamp'),
	'completed_at': fields.DateTime(description='Completion timestamp'),
	'duration_seconds': fields.Float(description='Execution duration'),
	'current_stage': fields.String(description='Current stage'),
	'stage_results': fields.Raw(description='Results by stage'),
	'metrics': fields.Raw(description='Execution metrics'),
	'errors': fields.List(fields.String, description='Error messages')
})


# Global AI service instance
ai_service = AICoreService()
security_manager = SecurityManager()


# Authentication decorator
def require_auth(f):
	"""Decorator for API authentication."""
	def decorated_function(*args, **kwargs):
		auth_header = request.headers.get('Authorization')
		if not auth_header:
			return {'error': 'Authorization header required'}, 401

		try:
			token = auth_header.split(' ')[1]  # Bearer token
			# Validate token with security manager
			user_info = asyncio.run(security_manager.validate_jwt_token(token))
			g.current_user = user_info
			return f(*args, **kwargs)
		except Exception as e:
			return {'error': f'Invalid token: {str(e)}'}, 401

	return decorated_function


# Error handlers
@api_v1.errorhandler(BadRequest)
def handle_bad_request(error):
	"""Handle bad request errors."""
	return {'error': 'Bad request', 'message': str(error)}, 400

@api_v1.errorhandler(NotFound)
def handle_not_found(error):
	"""Handle not found errors."""
	return {'error': 'Resource not found', 'message': str(error)}, 404

@api_v1.errorhandler(InternalServerError)
def handle_internal_error(error):
	"""Handle internal server errors."""
	return {'error': 'Internal server error', 'message': str(error)}, 500


# Models API
@models_ns.route('/')
class ModelList(Resource):
	"""Model collection endpoints."""

	@models_ns.doc('list_models')
	@models_ns.marshal_list_with(model_output)
	@require_auth
	def get(self):
		"""Get list of all models."""
		try:
			# Get query parameters
			parser = RequestParser()
			parser.add_argument('model_type', type=str, help='Filter by model type')
			parser.add_argument('framework', type=str, help='Filter by framework')
			parser.add_argument('status', type=str, help='Filter by status')
			parser.add_argument('limit', type=int, default=50, help='Limit results')
			parser.add_argument('offset', type=int, default=0, help='Offset for pagination')
			args = parser.parse_args()

			# Get models from AI service
			models = asyncio.run(ai_service.list_models(
				model_type=args.get('model_type'),
				framework=args.get('framework'),
				status=args.get('status'),
				limit=args.get('limit'),
				offset=args.get('offset')
			))

			return [model.model_dump() for model in models]

		except Exception as e:
			models_ns.abort(500, f'Error retrieving models: {str(e)}')

	@models_ns.doc('create_model')
	@models_ns.expect(model_input)
	@models_ns.marshal_with(model_output, code=201)
	@require_auth
	def post(self):
		"""Create a new model."""
		try:
			model_data = request.get_json()

			# Create model using AI service
			model = asyncio.run(ai_service.register_model(model_data))

			return model.model_dump(), 201

		except Exception as e:
			models_ns.abort(500, f'Error creating model: {str(e)}')


@models_ns.route('/<string:model_id>')
class Model(Resource):
	"""Individual model endpoints."""

	@models_ns.doc('get_model')
	@models_ns.marshal_with(model_output)
	@require_auth
	def get(self, model_id):
		"""Get model by ID."""
		try:
			model = asyncio.run(ai_service.get_model(model_id))
			if not model:
				models_ns.abort(404, f'Model {model_id} not found')

			return model.model_dump()

		except Exception as e:
			models_ns.abort(500, f'Error retrieving model: {str(e)}')

	@models_ns.doc('update_model')
	@models_ns.expect(model_input)
	@models_ns.marshal_with(model_output)
	@require_auth
	def put(self, model_id):
		"""Update model by ID."""
		try:
			model_data = request.get_json()
			model_data['model_id'] = model_id

			updated_model = asyncio.run(ai_service.update_model(model_id, model_data))
			if not updated_model:
				models_ns.abort(404, f'Model {model_id} not found')

			return updated_model.model_dump()

		except Exception as e:
			models_ns.abort(500, f'Error updating model: {str(e)}')

	@models_ns.doc('delete_model')
	@require_auth
	def delete(self, model_id):
		"""Delete model by ID."""
		try:
			success = asyncio.run(ai_service.delete_model(model_id))
			if not success:
				models_ns.abort(404, f'Model {model_id} not found')

			return {'message': f'Model {model_id} deleted successfully'}

		except Exception as e:
			models_ns.abort(500, f'Error deleting model: {str(e)}')


@models_ns.route('/<string:model_id>/deploy')
class ModelDeploy(Resource):
	"""Model deployment endpoints."""

	@models_ns.doc('deploy_model')
	@require_auth
	def post(self, model_id):
		"""Deploy model for inference."""
		try:
			deployment_config = request.get_json() or {}

			result = asyncio.run(ai_service.deploy_model(model_id, deployment_config))

			return {
				'message': f'Model {model_id} deployed successfully',
				'deployment_info': result
			}

		except Exception as e:
			models_ns.abort(500, f'Error deploying model: {str(e)}')

	@models_ns.doc('undeploy_model')
	@require_auth
	def delete(self, model_id):
		"""Undeploy model from inference."""
		try:
			result = asyncio.run(ai_service.undeploy_model(model_id))

			return {
				'message': f'Model {model_id} undeployed successfully',
				'result': result
			}

		except Exception as e:
			models_ns.abort(500, f'Error undeploying model: {str(e)}')


# Inference API
@inference_ns.route('/predict')
class Inference(Resource):
	"""Model inference endpoints."""

	@inference_ns.doc('run_inference')
	@inference_ns.expect(inference_request)
	@inference_ns.marshal_with(inference_response)
	@require_auth
	def post(self):
		"""Run model inference."""
		try:
			request_data = request.get_json()

			# Validate required fields
			if 'model_id' not in request_data or 'input_data' not in request_data:
				inference_ns.abort(400, 'model_id and input_data are required')

			# Create inference request
			inference_req = AICRInferenceRequest(
				model_id=request_data['model_id'],
				input_data=request_data['input_data'],
				parameters=request_data.get('parameters', {}),
				output_format=request_data.get('output_format', 'json')
			)

			# Execute inference
			result = asyncio.run(ai_service.run_inference(inference_req))

			return result.model_dump()

		except Exception as e:
			inference_ns.abort(500, f'Error running inference: {str(e)}')


@inference_ns.route('/batch')
class BatchInference(Resource):
	"""Batch inference endpoints."""

	@inference_ns.doc('run_batch_inference')
	@require_auth
	def post(self):
		"""Run batch model inference."""
		try:
			request_data = request.get_json()

			# Validate required fields
			if 'model_id' not in request_data or 'batch_data' not in request_data:
				inference_ns.abort(400, 'model_id and batch_data are required')

			model_id = request_data['model_id']
			batch_data = request_data['batch_data']

			# Execute batch inference
			results = []
			for i, input_data in enumerate(batch_data):
				inference_req = AICRInferenceRequest(
					model_id=model_id,
					input_data=input_data,
					parameters=request_data.get('parameters', {}),
					output_format=request_data.get('output_format', 'json')
				)

				result = asyncio.run(ai_service.run_inference(inference_req))
				results.append(result.model_dump())

			return {
				'batch_id': f'batch_{datetime.utcnow().isoformat()}',
				'model_id': model_id,
				'results': results,
				'total_predictions': len(results),
				'timestamp': datetime.utcnow().isoformat()
			}

		except Exception as e:
			inference_ns.abort(500, f'Error running batch inference: {str(e)}')


# Pipelines API
@pipelines_ns.route('/')
class PipelineList(Resource):
	"""Pipeline collection endpoints."""

	@pipelines_ns.doc('list_pipelines')
	@pipelines_ns.marshal_list_with(pipeline_output)
	@require_auth
	def get(self):
		"""Get list of all pipelines."""
		try:
			# Get query parameters
			parser = RequestParser()
			parser.add_argument('pipeline_type', type=str, help='Filter by pipeline type')
			parser.add_argument('status', type=str, help='Filter by status')
			parser.add_argument('limit', type=int, default=50, help='Limit results')
			parser.add_argument('offset', type=int, default=0, help='Offset for pagination')
			args = parser.parse_args()

			# Get pipelines from ML framework
			pipelines = list(ml_pipeline_framework.orchestrator.pipelines.values())

			# Apply filters
			if args.get('pipeline_type'):
				pipelines = [p for p in pipelines if p.training_config.model_type == args['pipeline_type']]

			# Apply pagination
			start = args.get('offset', 0)
			end = start + args.get('limit', 50)
			pipelines = pipelines[start:end]

			return [pipeline.model_dump() for pipeline in pipelines]

		except Exception as e:
			pipelines_ns.abort(500, f'Error retrieving pipelines: {str(e)}')

	@pipelines_ns.doc('create_pipeline')
	@pipelines_ns.expect(pipeline_input)
	@pipelines_ns.marshal_with(pipeline_output, code=201)
	@require_auth
	def post(self):
		"""Create a new pipeline."""
		try:
			pipeline_data = request.get_json()

			# Create pipeline from template or custom config
			if 'template' in pipeline_data:
				pipeline = asyncio.run(
					ml_pipeline_framework.create_pipeline_from_template(
						pipeline_data['template'],
						pipeline_data
					)
				)
			else:
				# Create custom pipeline (would need implementation)
				pipelines_ns.abort(400, 'Custom pipeline creation not yet implemented')

			return pipeline.model_dump(), 201

		except Exception as e:
			pipelines_ns.abort(500, f'Error creating pipeline: {str(e)}')


@pipelines_ns.route('/<string:pipeline_id>')
class Pipeline(Resource):
	"""Individual pipeline endpoints."""

	@pipelines_ns.doc('get_pipeline')
	@pipelines_ns.marshal_with(pipeline_output)
	@require_auth
	def get(self, pipeline_id):
		"""Get pipeline by ID."""
		try:
			pipeline = ml_pipeline_framework.orchestrator.pipelines.get(pipeline_id)
			if not pipeline:
				pipelines_ns.abort(404, f'Pipeline {pipeline_id} not found')

			return pipeline.model_dump()

		except Exception as e:
			pipelines_ns.abort(500, f'Error retrieving pipeline: {str(e)}')


@pipelines_ns.route('/<string:pipeline_id>/execute')
class PipelineExecute(Resource):
	"""Pipeline execution endpoints."""

	@pipelines_ns.doc('execute_pipeline')
	@pipelines_ns.marshal_with(execution_output, code=202)
	@require_auth
	def post(self, pipeline_id):
		"""Execute a pipeline."""
		try:
			execution_config = request.get_json() or {}

			execution_id = asyncio.run(
				ml_pipeline_framework.execute_pipeline(
					pipeline_id,
					input_data=execution_config.get('input_data'),
					execution_config=execution_config.get('execution_config')
				)
			)

			return {
				'execution_id': execution_id,
				'pipeline_id': pipeline_id,
				'status': 'started',
				'message': f'Pipeline {pipeline_id} execution started'
			}, 202

		except Exception as e:
			pipelines_ns.abort(500, f'Error executing pipeline: {str(e)}')


@pipelines_ns.route('/<string:pipeline_id>/executions')
class PipelineExecutions(Resource):
	"""Pipeline execution history endpoints."""

	@pipelines_ns.doc('get_pipeline_executions')
	@require_auth
	def get(self, pipeline_id):
		"""Get execution history for a pipeline."""
		try:
			# Get executions from orchestrator
			executions = [
				exec for exec in ml_pipeline_framework.orchestrator.executions.values()
				if exec.pipeline_id == pipeline_id
			]

			# Sort by start time (newest first)
			executions.sort(key=lambda x: x.started_at or datetime.min, reverse=True)

			return [execution.model_dump() for execution in executions]

		except Exception as e:
			pipelines_ns.abort(500, f'Error retrieving executions: {str(e)}')


@pipelines_ns.route('/executions/<string:execution_id>')
class ExecutionStatus(Resource):
	"""Individual execution status endpoints."""

	@pipelines_ns.doc('get_execution_status')
	@pipelines_ns.marshal_with(execution_output)
	@require_auth
	def get(self, execution_id):
		"""Get execution status by ID."""
		try:
			execution = asyncio.run(
				ml_pipeline_framework.get_execution_status(execution_id)
			)

			if not execution:
				pipelines_ns.abort(404, f'Execution {execution_id} not found')

			return execution.model_dump()

		except Exception as e:
			pipelines_ns.abort(500, f'Error retrieving execution status: {str(e)}')


# Monitoring API
@monitoring_ns.route('/health')
class SystemHealth(Resource):
	"""System health endpoints."""

	@monitoring_ns.doc('get_system_health')
	@require_auth
	def get(self):
		"""Get comprehensive system health status."""
		try:
			health_data = asyncio.run(ai_monitoring_system.get_system_health())
			return health_data

		except Exception as e:
			monitoring_ns.abort(500, f'Error retrieving system health: {str(e)}')


@monitoring_ns.route('/metrics')
class Metrics(Resource):
	"""Metrics endpoints."""

	@monitoring_ns.doc('get_metrics')
	@require_auth
	def get(self):
		"""Get system metrics."""
		try:
			# Get query parameters
			parser = RequestParser()
			parser.add_argument('metric_names', action='append', help='Metric names to filter')
			parser.add_argument('time_range_hours', type=int, default=1, help='Time range in hours')
			parser.add_argument('labels', type=str, help='JSON labels filter')
			args = parser.parse_args()

			# Calculate time range
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=args['time_range_hours'])

			# Parse labels filter if provided
			labels_filter = None
			if args.get('labels'):
				try:
					labels_filter = json.loads(args['labels'])
				except json.JSONDecodeError:
					monitoring_ns.abort(400, 'Invalid labels JSON format')

			# Get metrics
			metrics = asyncio.run(
				ai_monitoring_system.metrics_collector.get_metrics(
					metric_names=args.get('metric_names'),
					time_range=(start_time, end_time),
					labels_filter=labels_filter
				)
			)

			# Convert to JSON-serializable format
			metrics_data = [
				{
					'metric_name': m.metric_name,
					'metric_type': m.metric_type.value,
					'value': m.value,
					'timestamp': m.timestamp.isoformat(),
					'labels': m.labels,
					'source_component': m.source_component,
					'source_instance': m.source_instance
				}
				for m in metrics
			]

			return {
				'metrics': metrics_data,
				'time_range': {
					'start': start_time.isoformat(),
					'end': end_time.isoformat(),
					'hours': args['time_range_hours']
				},
				'count': len(metrics_data)
			}

		except Exception as e:
			monitoring_ns.abort(500, f'Error retrieving metrics: {str(e)}')


@monitoring_ns.route('/performance')
class Performance(Resource):
	"""Performance analysis endpoints."""

	@monitoring_ns.doc('get_performance_summary')
	@require_auth
	def get(self):
		"""Get performance summary and analysis."""
		try:
			# Get query parameters
			parser = RequestParser()
			parser.add_argument('time_range_hours', type=int, default=24, help='Analysis time range in hours')
			args = parser.parse_args()

			# Calculate time range
			end_time = datetime.utcnow()
			start_time = end_time - timedelta(hours=args['time_range_hours'])

			# Get performance summary
			performance_data = asyncio.run(
				ai_monitoring_system.get_performance_summary(
					time_range=(start_time, end_time)
				)
			)

			return performance_data

		except Exception as e:
			monitoring_ns.abort(500, f'Error retrieving performance data: {str(e)}')


# Marketplace API
@marketplace_ns.route('/models')
class MarketplaceModels(Resource):
	"""Model marketplace endpoints."""

	@marketplace_ns.doc('get_marketplace_models')
	@require_auth
	def get(self):
		"""Get models from marketplace."""
		try:
			# Get query parameters
			parser = RequestParser()
			parser.add_argument('category', type=str, help='Model category filter')
			parser.add_argument('framework', type=str, help='Framework filter')
			parser.add_argument('search', type=str, help='Search query')
			parser.add_argument('limit', type=int, default=20, help='Limit results')
			args = parser.parse_args()

			# Get models from marketplace
			models = asyncio.run(
				model_marketplace.search_models(
					query=args.get('search', ''),
					filters={
						'category': args.get('category'),
						'framework': args.get('framework')
					},
					limit=args.get('limit')
				)
			)

			return {
				'models': models,
				'total': len(models),
				'filters_applied': {k: v for k, v in args.items() if v is not None}
			}

		except Exception as e:
			marketplace_ns.abort(500, f'Error retrieving marketplace models: {str(e)}')


@marketplace_ns.route('/featured')
class FeaturedModels(Resource):
	"""Featured models endpoints."""

	@marketplace_ns.doc('get_featured_models')
	@require_auth
	def get(self):
		"""Get featured models from marketplace."""
		try:
			parser = RequestParser()
			parser.add_argument('limit', type=int, default=10, help='Limit results')
			args = parser.parse_args()

			featured_models = asyncio.run(
				model_marketplace.get_featured_models(limit=args.get('limit'))
			)

			return {
				'featured_models': featured_models,
				'count': len(featured_models)
			}

		except Exception as e:
			marketplace_ns.abort(500, f'Error retrieving featured models: {str(e)}')


# Admin API
@admin_ns.route('/status')
class AdminStatus(Resource):
	"""Administrative status endpoints."""

	@admin_ns.doc('get_admin_status')
	@require_auth
	def get(self):
		"""Get comprehensive administrative status."""
		try:
			# Check if current user has admin permissions
			if not g.current_user.get('is_admin', False):
				admin_ns.abort(403, 'Admin permissions required')

			status = {
				'ai_service': 'active' if ai_service._initialized else 'inactive',
				'monitoring_system': ai_monitoring_system.system_status.value,
				'ml_pipeline_framework': 'active' if ml_pipeline_framework._initialized else 'inactive',
				'model_marketplace': 'active' if model_marketplace._initialized else 'inactive',
				'security_manager': 'active' if security_manager._initialized else 'inactive',
				'timestamp': datetime.utcnow().isoformat()
			}

			return status

		except Exception as e:
			admin_ns.abort(500, f'Error retrieving admin status: {str(e)}')


# Create API Blueprint
def create_api_blueprint() -> Blueprint:
	"""Create and configure the AICR API blueprint.

	Returns:
		Blueprint: Configured API blueprint
	"""
	api_bp = Blueprint('aicr_api', __name__)
	api_v1.init_app(api_bp)

	# Initialize services on first request
	@api_bp.before_app_first_request
	def initialize_api_services():
		"""Initialize API services on first request."""
		try:
			# Initialize AI service
			asyncio.run(ai_service.initialize())

			# Initialize security manager
			asyncio.run(security_manager.initialize())

			logging.info("AICR API services initialized successfully")

		except Exception as e:
			logging.error(f"Failed to initialize AICR API services: {e}")

	return api_bp


# Export API blueprint creation function
__all__ = [
	'create_api_blueprint',
	'api_v1',
	'models_ns',
	'inference_ns',
	'pipelines_ns',
	'monitoring_ns',
	'marketplace_ns',
	'admin_ns'
]
