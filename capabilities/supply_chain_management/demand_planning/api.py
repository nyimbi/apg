"""
Demand Planning REST API

REST API endpoints for demand forecasting operations.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from marshmallow import Schema, fields as ma_fields, validate, ValidationError

from .service import SCDemandPlanningService
from .models import ForecastCreate, DemandHistoryCreate, ForecastModelCreate

# Create blueprint
demand_planning_api = Blueprint('demand_planning_api', __name__)
api = Api(demand_planning_api, version='1.0', title='Demand Planning API')

# Create namespace
ns = Namespace('demand_planning', description='Demand Planning Operations')
api.add_namespace(ns)

# API Models for documentation
forecast_model = api.model('Forecast', {
	'id': fields.String(required=True, description='Forecast ID'),
	'product_sku': fields.String(required=True, description='Product SKU'),
	'location_code': fields.String(description='Location code'),
	'forecast_quantity': fields.Float(required=True, description='Forecasted quantity'),
	'forecast_value': fields.Float(description='Forecasted value'),
	'confidence_level': fields.Float(description='Confidence level (0-1)'),
	'period_start': fields.Date(required=True, description='Forecast period start'),
	'period_end': fields.Date(required=True, description='Forecast period end'),
	'status': fields.String(description='Forecast status')
})

forecast_create_model = api.model('ForecastCreate', {
	'forecast_id': fields.String(required=True, description='Forecast ID'),
	'product_sku': fields.String(required=True, description='Product SKU'),
	'location_code': fields.String(description='Location code'),
	'forecast_model_id': fields.String(required=True, description='Forecast model ID'),
	'forecast_quantity': fields.Float(required=True, description='Forecasted quantity'),
	'forecast_value': fields.Float(description='Forecasted value'),
	'confidence_level': fields.Float(description='Confidence level (0-1)'),
	'period_start': fields.Date(required=True, description='Period start date'),
	'period_end': fields.Date(required=True, description='Period end date'),
	'period_type': fields.String(description='Period type (daily, weekly, monthly)')
})

model_performance_model = api.model('ModelPerformance', {
	'model_id': fields.String(required=True, description='Model ID'),
	'mape': fields.Float(required=True, description='Mean Absolute Percentage Error'),
	'rmse': fields.Float(required=True, description='Root Mean Square Error'),
	'mae': fields.Float(required=True, description='Mean Absolute Error'),
	'accuracy_category': fields.String(required=True, description='Accuracy category'),
	'sample_size': fields.Integer(required=True, description='Training sample size')
})

# Helper functions
def get_service() -> SCDemandPlanningService:
	"""Get demand planning service instance"""
	# This would typically get the database session and tenant info from the request context
	from flask_appbuilder import AppBuilder
	appbuilder = current_app.appbuilder
	
	return SCDemandPlanningService(
		db_session=appbuilder.get_session,
		tenant_id=request.headers.get('X-Tenant-ID', 'default'),
		current_user=request.headers.get('X-User-ID', 'api_user')
	)

def handle_async_result(async_func):
	"""Handle async function results"""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	
	return loop.run_until_complete(async_func)

@ns.route('/forecasts')
class ForecastListAPI(Resource):
	"""Forecast list operations"""
	
	@ns.doc('list_forecasts')
	@ns.marshal_list_with(forecast_model)
	def get(self):
		"""Get list of forecasts"""
		try:
			service = get_service()
			
			# Get query parameters
			product_sku = request.args.get('product_sku')
			location_code = request.args.get('location_code')
			start_date = request.args.get('start_date')
			end_date = request.args.get('end_date')
			status = request.args.get('status')
			limit = int(request.args.get('limit', 100))
			offset = int(request.args.get('offset', 0))
			
			# Build query filters
			filters = {}
			if product_sku:
				filters['product_sku'] = product_sku
			if location_code:
				filters['location_code'] = location_code
			if status:
				filters['status'] = status
			if start_date:
				filters['start_date'] = datetime.strptime(start_date, '%Y-%m-%d').date()
			if end_date:
				filters['end_date'] = datetime.strptime(end_date, '%Y-%m-%d').date()
			
			# Get forecasts (this would be implemented in the service)
			forecasts = []  # service.get_forecasts(filters, limit, offset)
			
			return [
				{
					'id': f.id,
					'product_sku': f.product_sku,
					'location_code': f.location_code,
					'forecast_quantity': float(f.forecast_quantity),
					'forecast_value': float(f.forecast_value) if f.forecast_value else None,
					'confidence_level': float(f.confidence_level) if f.confidence_level else None,
					'period_start': f.period_start.isoformat(),
					'period_end': f.period_end.isoformat(),
					'status': f.status
				}
				for f in forecasts
			]
			
		except Exception as e:
			api.abort(500, f"Failed to retrieve forecasts: {str(e)}")
	
	@ns.doc('create_forecast')
	@ns.expect(forecast_create_model)
	@ns.marshal_with(forecast_model)
	def post(self):
		"""Create a new forecast"""
		try:
			service = get_service()
			data = request.get_json()
			
			# Validate input data
			forecast_data = ForecastCreate(**data)
			
			# Create forecast
			result = handle_async_result(service.create_forecast(forecast_data))
			
			if result.success:
				return {
					'id': result.forecast_id,
					'message': result.message,
					'accuracy_metrics': result.accuracy_metrics
				}, 201
			else:
				api.abort(400, result.message)
				
		except ValidationError as e:
			api.abort(400, f"Validation error: {e.messages}")
		except Exception as e:
			api.abort(500, f"Failed to create forecast: {str(e)}")

@ns.route('/forecasts/batch')
class ForecastBatchAPI(Resource):
	"""Batch forecast operations"""
	
	@ns.doc('generate_batch_forecasts')
	def post(self):
		"""Generate forecasts for multiple products"""
		try:
			service = get_service()
			data = request.get_json()
			
			products = data.get('products', [])
			location_code = data.get('location_code')
			forecast_horizon_days = data.get('forecast_horizon_days', 90)
			model_id = data.get('model_id')
			
			if not products:
				api.abort(400, "Products list is required")
			
			# Generate batch forecasts
			results = handle_async_result(
				service.generate_forecast_batch(
					products=products,
					location_code=location_code,
					forecast_horizon_days=forecast_horizon_days,
					model_id=model_id
				)
			)
			
			# Summarize results
			successful = [r for r in results if r.success]
			failed = [r for r in results if not r.success]
			
			return {
				'total_processed': len(results),
				'successful': len(successful),
				'failed': len(failed),
				'results': [
					{
						'forecast_id': r.forecast_id,
						'success': r.success,
						'message': r.message
					}
					for r in results
				]
			}
			
		except Exception as e:
			api.abort(500, f"Failed to generate batch forecasts: {str(e)}")

@ns.route('/models')
class ForecastModelAPI(Resource):
	"""Forecast model operations"""
	
	@ns.doc('list_models')
	def get(self):
		"""Get list of forecast models"""
		try:
			service = get_service()
			
			# Get models (this would be implemented in the service)
			models = []  # service.get_models()
			
			return [
				{
					'id': m.id,
					'model_name': m.model_name,
					'model_code': m.model_code,
					'model_type': m.model_type,
					'algorithm': m.algorithm,
					'accuracy_mape': float(m.accuracy_mape) if m.accuracy_mape else None,
					'status': m.status,
					'is_active': m.is_active,
					'last_trained': m.last_trained.isoformat() if m.last_trained else None
				}
				for m in models
			]
			
		except Exception as e:
			api.abort(500, f"Failed to retrieve models: {str(e)}")
	
	@ns.doc('create_model')
	def post(self):
		"""Create a new forecast model"""
		try:
			service = get_service()
			data = request.get_json()
			
			# Validate input data
			model_data = ForecastModelCreate(**data)
			
			# Create model
			model_id = handle_async_result(service.create_forecast_model(model_data))
			
			return {
				'id': model_id,
				'message': 'Forecast model created successfully'
			}, 201
			
		except ValidationError as e:
			api.abort(400, f"Validation error: {e.messages}")
		except Exception as e:
			api.abort(500, f"Failed to create model: {str(e)}")

@ns.route('/models/<string:model_id>/train')
class ModelTrainingAPI(Resource):
	"""Model training operations"""
	
	@ns.doc('train_model')
	@ns.marshal_with(model_performance_model)
	def post(self, model_id):
		"""Train a forecast model"""
		try:
			service = get_service()
			data = request.get_json()
			
			training_data = data.get('training_data', [])
			if not training_data:
				api.abort(400, "Training data is required")
			
			# Train model
			performance = handle_async_result(
				service.train_model(model_id, training_data)
			)
			
			return {
				'model_id': performance.model_id,
				'mape': performance.mape,
				'rmse': performance.rmse,
				'mae': performance.mae,
				'accuracy_category': performance.accuracy_category,
				'sample_size': performance.sample_size
			}
			
		except ValueError as e:
			api.abort(404, str(e))
		except Exception as e:
			api.abort(500, f"Failed to train model: {str(e)}")

@ns.route('/history/import')
class DemandHistoryImportAPI(Resource):
	"""Demand history import operations"""
	
	@ns.doc('import_history')
	def post(self):
		"""Import historical demand data"""
		try:
			service = get_service()
			data = request.get_json()
			
			history_data_list = data.get('history_data', [])
			if not history_data_list:
				api.abort(400, "History data is required")
			
			# Validate and convert data
			validated_data = []
			for item in history_data_list:
				try:
					validated_data.append(DemandHistoryCreate(**item))
				except ValidationError as e:
					api.abort(400, f"Invalid history data: {e.messages}")
			
			# Import data
			result = handle_async_result(
				service.import_demand_history(validated_data)
			)
			
			return {
				'imported': result['imported'],
				'errors': result['errors'],
				'total': result['total'],
				'success_rate': result['imported'] / result['total'] * 100 if result['total'] > 0 else 0
			}
			
		except Exception as e:
			api.abort(500, f"Failed to import history data: {str(e)}")

@ns.route('/accuracy')
class ForecastAccuracyAPI(Resource):
	"""Forecast accuracy operations"""
	
	@ns.doc('calculate_accuracy')
	def get(self):
		"""Calculate forecast accuracy"""
		try:
			service = get_service()
			
			# Get query parameters
			start_date_str = request.args.get('start_date')
			end_date_str = request.args.get('end_date')
			product_sku = request.args.get('product_sku')
			
			if not start_date_str or not end_date_str:
				api.abort(400, "start_date and end_date are required")
			
			start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
			end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
			
			# Calculate accuracy
			accuracy_results = handle_async_result(
				service.calculate_forecast_accuracy(
					start_date=start_date,
					end_date=end_date,
					product_sku=product_sku
				)
			)
			
			return {
				'period': {
					'start_date': start_date.isoformat(),
					'end_date': end_date.isoformat()
				},
				'results': accuracy_results,
				'summary': {
					'total_forecasts': len(accuracy_results),
					'average_mape': sum(r['percentage_error'] for r in accuracy_results) / len(accuracy_results) if accuracy_results else 0,
					'accuracy_distribution': self._calculate_accuracy_distribution(accuracy_results)
				}
			}
			
		except ValueError as e:
			api.abort(400, str(e))
		except Exception as e:
			api.abort(500, f"Failed to calculate accuracy: {str(e)}")
	
	def _calculate_accuracy_distribution(self, results: List[Dict]) -> Dict[str, int]:
		"""Calculate accuracy category distribution"""
		distribution = {}
		for result in results:
			category = result.get('accuracy_category', 'unknown')
			distribution[category] = distribution.get(category, 0) + 1
		return distribution

@ns.route('/analytics')
class ForecastAnalyticsAPI(Resource):
	"""Forecast analytics operations"""
	
	@ns.doc('get_analytics')
	def get(self):
		"""Get forecast analytics and insights"""
		try:
			service = get_service()
			
			# Get query parameters
			product_sku = request.args.get('product_sku')
			days_back = int(request.args.get('days_back', 30))
			
			# Get analytics
			analytics = handle_async_result(
				service.get_forecast_analytics(
					product_sku=product_sku,
					days_back=days_back
				)
			)
			
			return analytics
			
		except Exception as e:
			api.abort(500, f"Failed to retrieve analytics: {str(e)}")

# Register blueprint
def register_api(app):
	"""Register the demand planning API blueprint"""
	app.register_blueprint(demand_planning_api, url_prefix='/api/supply_chain/demand_planning')