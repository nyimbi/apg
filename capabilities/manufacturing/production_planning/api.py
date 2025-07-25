"""
Production Planning API

REST API endpoints for production planning functionality including
master production schedules, production orders, demand forecasts, and capacity planning.
"""

from datetime import datetime, date
from typing import List, Optional
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .models import (
	MasterProductionScheduleCreate, ProductionOrderCreate, DemandForecastCreate, ResourceCapacityCreate,
	ProductionOrderStatus, SchedulingPriority, PlanningHorizon
)
from .service import ProductionPlanningService

# Create API namespace
api_bp = Blueprint('production_planning_api', __name__, url_prefix='/api/v1/manufacturing/production-planning')
api = Api(api_bp, doc='/docs/', title='Production Planning API', version='1.0')

# Define namespaces
ns_schedule = Namespace('schedules', description='Master Production Schedule operations')
ns_orders = Namespace('orders', description='Production Order operations')  
ns_forecasts = Namespace('forecasts', description='Demand Forecast operations')
ns_capacity = Namespace('capacity', description='Resource Capacity operations')

api.add_namespace(ns_schedule)
api.add_namespace(ns_orders)
api.add_namespace(ns_forecasts)
api.add_namespace(ns_capacity)

# Define API models for documentation
schedule_model = api.model('MasterProductionSchedule', {
	'schedule_name': fields.String(required=True, description='Schedule name'),
	'planning_period': fields.String(required=True, description='Planning period (e.g., 2024-Q1)'),
	'planning_horizon': fields.String(required=True, enum=['short_term', 'medium_term', 'long_term']),
	'product_id': fields.String(required=True, description='Product ID'),
	'facility_id': fields.String(required=True, description='Facility ID'),
	'production_line_id': fields.String(description='Production line ID'),
	'planned_quantity': fields.Float(required=True, description='Planned quantity'),
	'planned_start_date': fields.Date(required=True, description='Planned start date'),
	'planned_end_date': fields.Date(required=True, description='Planned end date'),
	'forecast_demand': fields.Float(description='Forecast demand'),
	'available_capacity': fields.Float(description='Available capacity'),
	'priority': fields.String(enum=['low', 'normal', 'high', 'urgent', 'critical'], default='normal'),
	'safety_stock_days': fields.Integer(default=0, description='Safety stock days'),
	'lead_time_days': fields.Integer(default=0, description='Lead time days')
})

order_model = api.model('ProductionOrder', {
	'order_number': fields.String(required=True, description='Order number'),
	'order_type': fields.String(required=True, description='Order type'),
	'master_schedule_id': fields.String(description='Master schedule ID'),
	'product_id': fields.String(required=True, description='Product ID'),
	'product_sku': fields.String(required=True, description='Product SKU'),
	'product_name': fields.String(required=True, description='Product name'),
	'facility_id': fields.String(required=True, description='Facility ID'),
	'production_line_id': fields.String(description='Production line ID'),
	'work_center_id': fields.String(description='Work center ID'),
	'ordered_quantity': fields.Float(required=True, description='Ordered quantity'),
	'scheduled_start_date': fields.DateTime(required=True, description='Scheduled start date'),
	'scheduled_end_date': fields.DateTime(required=True, description='Scheduled end date'),
	'priority': fields.String(enum=['low', 'normal', 'high', 'urgent', 'critical'], default='normal'),
	'estimated_labor_hours': fields.Float(description='Estimated labor hours'),
	'estimated_machine_hours': fields.Float(description='Estimated machine hours'),
	'bom_id': fields.String(description='Bill of materials ID'),
	'routing_id': fields.String(description='Routing ID'),
	'production_notes': fields.String(description='Production notes'),
	'special_instructions': fields.String(description='Special instructions'),
	'quality_requirements': fields.String(description='Quality requirements')
})

forecast_model = api.model('DemandForecast', {
	'forecast_name': fields.String(required=True, description='Forecast name'),
	'forecast_period': fields.String(required=True, description='Forecast period'),
	'forecast_type': fields.String(required=True, description='Forecast type'),
	'product_id': fields.String(required=True, description='Product ID'),
	'facility_id': fields.String(description='Facility ID'),
	'customer_id': fields.String(description='Customer ID'),
	'forecast_quantity': fields.Float(required=True, description='Forecast quantity'),
	'forecast_value': fields.Float(description='Forecast value'),
	'period_start_date': fields.Date(required=True, description='Period start date'),
	'period_end_date': fields.Date(required=True, description='Period end date'),
	'forecast_method': fields.String(description='Forecast method'),
	'confidence_level': fields.Float(description='Confidence level (0-100)'),
	'seasonality_factor': fields.Float(description='Seasonality factor'),
	'trend_factor': fields.Float(description='Trend factor')
})

capacity_model = api.model('ResourceCapacity', {
	'resource_type': fields.String(required=True, description='Resource type'),
	'resource_id': fields.String(required=True, description='Resource ID'),
	'resource_name': fields.String(required=True, description='Resource name'),
	'facility_id': fields.String(required=True, description='Facility ID'),
	'work_center_id': fields.String(description='Work center ID'),
	'planning_period': fields.String(required=True, description='Planning period'),
	'capacity_unit': fields.String(required=True, description='Capacity unit'),
	'available_capacity': fields.Float(required=True, description='Available capacity'),
	'period_start_date': fields.Date(required=True, description='Period start date'),
	'period_end_date': fields.Date(required=True, description='Period end date'),
	'shifts_per_day': fields.Integer(default=1, description='Shifts per day'),
	'hours_per_shift': fields.Float(default=8.0, description='Hours per shift'),
	'working_days_per_week': fields.Integer(default=5, description='Working days per week'),
	'max_capacity': fields.Float(description='Maximum capacity'),
	'min_capacity': fields.Float(description='Minimum capacity'),
	'setup_time_hours': fields.Float(description='Setup time hours'),
	'maintenance_time_hours': fields.Float(description='Maintenance time hours')
})

# Utility functions
async def get_service() -> ProductionPlanningService:
	"""Get production planning service instance"""
	# In real implementation, get from dependency injection or app context
	engine = create_async_engine(current_app.config['DATABASE_URL'])
	async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
	session = async_session()
	return ProductionPlanningService(session)

def get_current_tenant_id() -> str:
	"""Get current tenant ID from request context"""
	return request.headers.get('X-Tenant-ID', 'default-tenant')

def get_current_user_id() -> str:
	"""Get current user ID from request context"""
	return request.headers.get('X-User-ID', 'current-user')

# Master Production Schedule endpoints
@ns_schedule.route('/')
class MasterScheduleListResource(Resource):
	"""Master production schedule list operations"""
	
	@ns_schedule.marshal_list_with(schedule_model)
	async def get(self):
		"""Get all master production schedules"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		
		# Get query parameters
		facility_id = request.args.get('facility_id')
		planning_period = request.args.get('planning_period')
		
		# Mock response - implement actual service call
		return [
			{
				'id': 'sched-001',
				'schedule_name': 'Q1 2024 Widget Production',
				'planning_period': '2024-Q1',
				'planning_horizon': 'medium_term',
				'product_id': 'prod-001',
				'facility_id': 'facility-001',
				'planned_quantity': 10000.0,
				'planned_start_date': '2024-01-01',
				'planned_end_date': '2024-03-31',
				'priority': 'normal'
			}
		]
	
	@ns_schedule.expect(schedule_model)
	async def post(self):
		"""Create new master production schedule"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			schedule_data = MasterProductionScheduleCreate(**request.json)
			schedule = await service.create_master_schedule(tenant_id, schedule_data, user_id)
			
			return {
				'id': schedule.id,
				'message': 'Master production schedule created successfully'
			}, 201
		
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': 'Internal server error'}, 500

@ns_schedule.route('/<string:schedule_id>')
class MasterScheduleResource(Resource):
	"""Individual master production schedule operations"""
	
	async def get(self, schedule_id):
		"""Get master production schedule by ID"""
		# Mock response - implement actual service call
		return {
			'id': schedule_id,
			'schedule_name': 'Q1 2024 Widget Production',
			'planning_period': '2024-Q1',
			'status': 'active'
		}

# Production Order endpoints
@ns_orders.route('/')
class ProductionOrderListResource(Resource):
	"""Production order list operations"""
	
	@ns_orders.marshal_list_with(order_model)
	async def get(self):
		"""Get all production orders"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		
		# Get query parameters
		facility_id = request.args.get('facility_id')
		status = request.args.get('status')
		start_date = request.args.get('start_date')
		end_date = request.args.get('end_date')
		
		# Parse status filter
		status_filter = None
		if status:
			try:
				status_filter = [ProductionOrderStatus(status)]
			except ValueError:
				return {'error': f'Invalid status: {status}'}, 400
		
		# Parse dates
		parsed_start_date = None
		parsed_end_date = None
		try:
			if start_date:
				parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
			if end_date:
				parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
		except ValueError:
			return {'error': 'Invalid date format. Use YYYY-MM-DD'}, 400
		
		if facility_id:
			orders = await service.get_production_orders_by_facility(
				tenant_id, facility_id, status_filter, parsed_start_date, parsed_end_date
			)
			
			return [
				{
					'id': order.id,
					'order_number': order.order_number,
					'product_name': order.product_name,
					'ordered_quantity': float(order.ordered_quantity),
					'status': order.status,
					'scheduled_start_date': order.scheduled_start_date.isoformat(),
					'scheduled_end_date': order.scheduled_end_date.isoformat()
				}
				for order in orders
			]
		
		return []
	
	@ns_orders.expect(order_model)
	async def post(self):
		"""Create new production order"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			order_data = ProductionOrderCreate(**request.json)
			order = await service.create_production_order(tenant_id, order_data, user_id)
			
			return {
				'id': order.id,
				'order_number': order.order_number,
				'message': 'Production order created successfully'
			}, 201
		
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': 'Internal server error'}, 500

@ns_orders.route('/<string:order_id>')
class ProductionOrderResource(Resource):
	"""Individual production order operations"""
	
	async def get(self, order_id):
		"""Get production order by ID"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		
		order = await service.get_production_order(tenant_id, order_id)
		if not order:
			return {'error': 'Production order not found'}, 404
		
		return {
			'id': order.id,
			'order_number': order.order_number,
			'product_name': order.product_name,
			'ordered_quantity': float(order.ordered_quantity),
			'produced_quantity': float(order.produced_quantity or 0),
			'status': order.status,
			'priority': order.priority,
			'scheduled_start_date': order.scheduled_start_date.isoformat(),
			'scheduled_end_date': order.scheduled_end_date.isoformat(),
			'actual_start_date': order.actual_start_date.isoformat() if order.actual_start_date else None,
			'actual_end_date': order.actual_end_date.isoformat() if order.actual_end_date else None
		}

@ns_orders.route('/<string:order_id>/status')
class ProductionOrderStatusResource(Resource):
	"""Production order status update operations"""
	
	async def put(self, order_id):
		"""Update production order status"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		data = request.json
		if not data or 'status' not in data:
			return {'error': 'Status is required'}, 400
		
		try:
			new_status = ProductionOrderStatus(data['status'])
			actual_start_date = None
			actual_end_date = None
			
			if 'actual_start_date' in data:
				actual_start_date = datetime.fromisoformat(data['actual_start_date'])
			if 'actual_end_date' in data:
				actual_end_date = datetime.fromisoformat(data['actual_end_date'])
			
			order = await service.update_production_order_status(
				tenant_id, order_id, new_status, user_id, actual_start_date, actual_end_date
			)
			
			return {
				'id': order.id,
				'status': order.status,
				'message': 'Production order status updated successfully'
			}
		
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': 'Internal server error'}, 500

# Demand Forecast endpoints
@ns_forecasts.route('/')
class DemandForecastListResource(Resource):
	"""Demand forecast list operations"""
	
	@ns_forecasts.expect(forecast_model)
	async def post(self):
		"""Create new demand forecast"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			forecast_data = DemandForecastCreate(**request.json)
			forecast = await service.create_demand_forecast(tenant_id, forecast_data, user_id)
			
			return {
				'id': forecast.id,
				'forecast_name': forecast.forecast_name,
				'message': 'Demand forecast created successfully'
			}, 201
		
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': 'Internal server error'}, 500

# Resource Capacity endpoints
@ns_capacity.route('/')
class ResourceCapacityListResource(Resource):
	"""Resource capacity list operations"""
	
	@ns_capacity.expect(capacity_model)
	async def post(self):
		"""Create or update resource capacity"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		user_id = get_current_user_id()
		
		try:
			capacity_data = ResourceCapacityCreate(**request.json)
			capacity = await service.create_resource_capacity(tenant_id, capacity_data, user_id)
			
			return {
				'id': capacity.id,
				'resource_name': capacity.resource_name,
				'message': 'Resource capacity created successfully'
			}, 201
		
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			return {'error': 'Internal server error'}, 500

@ns_capacity.route('/utilization/<string:facility_id>/<string:planning_period>')
class CapacityUtilizationResource(Resource):
	"""Capacity utilization report operations"""
	
	async def get(self, facility_id, planning_period):
		"""Get capacity utilization report"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		
		try:
			report = await service.get_capacity_utilization_report(
				tenant_id, facility_id, planning_period
			)
			return report
		
		except Exception as e:
			return {'error': 'Internal server error'}, 500

@ns_schedule.route('/optimize/<string:facility_id>')
class ScheduleOptimizationResource(Resource):
	"""Schedule optimization operations"""
	
	async def post(self, facility_id):
		"""Optimize production schedule for facility"""
		service = await get_service()
		tenant_id = get_current_tenant_id()
		
		data = request.json or {}
		planning_horizon = PlanningHorizon(data.get('planning_horizon', 'short_term'))
		optimization_criteria = data.get('optimization_criteria', 'minimize_makespan')
		
		try:
			result = await service.optimize_production_schedule(
				tenant_id, facility_id, planning_horizon, optimization_criteria
			)
			return result
		
		except Exception as e:
			return {'error': 'Internal server error'}, 500

__all__ = ["api_bp"]