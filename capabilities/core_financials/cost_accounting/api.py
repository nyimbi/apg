"""
Cost Accounting REST API

REST API endpoints for Cost Accounting functionality.
Provides programmatic access to cost accounting operations.
"""

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema, fields, validate
from datetime import date, datetime
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFCACostCenter, CFCACostCategory, CFCACostDriver, CFCACostAllocation,
	CFCACostPool, CFCAActivity, CFCAActivityCost, CFCAProductCost,
	CFCAJobCost, CFCAStandardCost, CFCAVarianceAnalysis
)
from .service import CostAccountingService, CostAllocationRequest, JobCostSummary, VarianceReport
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

class CACostCenterSchema(Schema):
	"""Schema for Cost Center serialization"""
	cost_center_id = fields.String(dump_only=True)
	center_code = fields.String(required=True, validate=validate.Length(max=20))
	center_name = fields.String(required=True, validate=validate.Length(max=200))
	description = fields.String(allow_none=True)
	parent_center_id = fields.String(allow_none=True)
	center_type = fields.String(required=True, validate=validate.OneOf([
		'Production', 'Service', 'Administrative', 'Support', 'Revenue'
	]))
	responsibility_type = fields.String(required=True, validate=validate.OneOf([
		'Cost', 'Profit', 'Investment', 'Revenue'
	]))
	manager_name = fields.String(allow_none=True)
	manager_email = fields.Email(allow_none=True)
	department = fields.String(allow_none=True)
	location = fields.String(allow_none=True)
	annual_budget = fields.Decimal(places=2, allow_none=True)
	ytd_actual = fields.Decimal(places=2, dump_only=True)
	ytd_budget = fields.Decimal(places=2, dump_only=True)
	effective_date = fields.Date(required=True)
	end_date = fields.Date(allow_none=True)
	allow_cost_allocation = fields.Boolean(default=True)
	require_job_number = fields.Boolean(default=False)
	default_currency = fields.String(default='USD')
	is_active = fields.Boolean(default=True)
	level = fields.Integer(dump_only=True)
	path = fields.String(dump_only=True)


class CACostCategorySchema(Schema):
	"""Schema for Cost Category serialization"""
	category_id = fields.String(dump_only=True)
	category_code = fields.String(required=True, validate=validate.Length(max=20))
	category_name = fields.String(required=True, validate=validate.Length(max=200))
	description = fields.String(allow_none=True)
	parent_category_id = fields.String(allow_none=True)
	cost_type = fields.String(required=True, validate=validate.OneOf([
		'Direct', 'Indirect', 'Period'
	]))
	cost_behavior = fields.String(required=True, validate=validate.OneOf([
		'Fixed', 'Variable', 'Mixed'
	]))
	cost_nature = fields.String(required=True, validate=validate.OneOf([
		'Material', 'Labor', 'Overhead', 'Administrative', 'Selling'
	]))
	is_variable = fields.Boolean(default=True)
	is_traceable = fields.Boolean(default=True)
	is_controllable = fields.Boolean(default=True)
	gl_account_code = fields.String(allow_none=True)
	gl_account_id = fields.String(allow_none=True)
	effective_date = fields.Date(required=True)
	end_date = fields.Date(allow_none=True)
	is_active = fields.Boolean(default=True)
	level = fields.Integer(dump_only=True)
	path = fields.String(dump_only=True)


class CACostDriverSchema(Schema):
	"""Schema for Cost Driver serialization"""
	driver_id = fields.String(dump_only=True)
	driver_code = fields.String(required=True, validate=validate.Length(max=20))
	driver_name = fields.String(required=True, validate=validate.Length(max=200))
	description = fields.String(allow_none=True)
	unit_of_measure = fields.String(required=True, validate=validate.Length(max=50))
	driver_type = fields.String(required=True, validate=validate.OneOf([
		'Volume', 'Activity', 'Transaction', 'Facility'
	]))
	is_volume_based = fields.Boolean(default=True)
	is_activity_based = fields.Boolean(default=False)
	requires_measurement = fields.Boolean(default=True)
	calculation_method = fields.String(allow_none=True)
	calculation_frequency = fields.String(default='Monthly')
	default_rate = fields.Decimal(places=4, allow_none=True)
	current_capacity = fields.Decimal(places=2, allow_none=True)
	practical_capacity = fields.Decimal(places=2, allow_none=True)
	effective_date = fields.Date(required=True)
	end_date = fields.Date(allow_none=True)
	is_active = fields.Boolean(default=True)


class CAJobCostSchema(Schema):
	"""Schema for Job Cost serialization"""
	job_cost_id = fields.String(dump_only=True)
	job_number = fields.String(required=True, validate=validate.Length(max=50))
	job_name = fields.String(required=True, validate=validate.Length(max=200))
	job_description = fields.String(allow_none=True)
	cost_center_id = fields.String(required=True)
	cost_category_id = fields.String(required=True)
	customer_code = fields.String(allow_none=True)
	customer_name = fields.String(allow_none=True)
	project_code = fields.String(allow_none=True)
	contract_number = fields.String(allow_none=True)
	start_date = fields.Date(required=True)
	planned_completion_date = fields.Date(allow_none=True)
	actual_completion_date = fields.Date(allow_none=True)
	budgeted_cost = fields.Decimal(places=2, default=0)
	budgeted_hours = fields.Decimal(places=2, default=0)
	contract_value = fields.Decimal(places=2, allow_none=True)
	actual_material_cost = fields.Decimal(places=2, default=0)
	actual_labor_cost = fields.Decimal(places=2, default=0)
	actual_overhead_cost = fields.Decimal(places=2, default=0)
	actual_other_cost = fields.Decimal(places=2, default=0)
	actual_labor_hours = fields.Decimal(places=2, default=0)
	actual_machine_hours = fields.Decimal(places=2, default=0)
	committed_material_cost = fields.Decimal(places=2, default=0)
	committed_labor_cost = fields.Decimal(places=2, default=0)
	committed_other_cost = fields.Decimal(places=2, default=0)
	billed_to_date = fields.Decimal(places=2, default=0)
	percent_complete = fields.Decimal(places=2, default=0)
	billing_method = fields.String(allow_none=True)
	job_status = fields.String(default='Active')
	is_billable = fields.Boolean(default=True)
	is_closed = fields.Boolean(default=False)


class CAStandardCostSchema(Schema):
	"""Schema for Standard Cost serialization"""
	standard_cost_id = fields.String(dump_only=True)
	cost_object_type = fields.String(required=True, validate=validate.OneOf([
		'Product', 'Service', 'Activity', 'Job'
	]))
	cost_object_code = fields.String(required=True, validate=validate.Length(max=50))
	cost_object_name = fields.String(required=True, validate=validate.Length(max=200))
	cost_category_id = fields.String(required=True)
	cost_center_id = fields.String(allow_none=True)
	standard_cost_per_unit = fields.Decimal(places=4, required=True)
	standard_quantity_per_unit = fields.Decimal(places=4, default=1.0)
	standard_rate_per_quantity = fields.Decimal(places=4, required=True)
	unit_of_measure = fields.String(required=True, validate=validate.Length(max=50))
	quantity_unit_of_measure = fields.String(allow_none=True)
	effective_date = fields.Date(required=True)
	end_date = fields.Date(allow_none=True)
	fiscal_year = fields.Integer(required=True)
	version = fields.String(default='1.0')
	standard_type = fields.String(default='Attainable', validate=validate.OneOf([
		'Ideal', 'Attainable', 'Current', 'Historical'
	]))
	revision_reason = fields.String(allow_none=True)
	favorable_variance_threshold = fields.Decimal(places=2, default=5.0)
	unfavorable_variance_threshold = fields.Decimal(places=2, default=5.0)
	is_active = fields.Boolean(default=True)
	is_approved = fields.Boolean(default=False)
	approved_by = fields.String(allow_none=True)
	approved_date = fields.Date(allow_none=True)


# API Resource Classes

class CostCenterListApi(Resource):
	"""Cost Center List API"""
	
	def get(self):
		"""Get list of cost centers"""
		tenant_id = request.args.get('tenant_id', 'default_tenant')
		active_only = request.args.get('active_only', 'true').lower() == 'true'
		
		query = CFCACostCenter.query.filter_by(tenant_id=tenant_id)
		if active_only:
			query = query.filter_by(is_active=True)
		
		cost_centers = query.all()
		schema = CACostCenterSchema(many=True)
		
		return {
			'data': schema.dump(cost_centers),
			'count': len(cost_centers)
		}
	
	def post(self):
		"""Create new cost center"""
		schema = CACostCenterSchema()
		try:
			data = schema.load(request.json)
		except Exception as e:
			return {'error': str(e)}, 400
		
		tenant_id = request.json.get('tenant_id', 'default_tenant')
		service = CostAccountingService(tenant_id=tenant_id)
		
		try:
			cost_center = service.create_cost_center(data)
			return schema.dump(cost_center), 201
		except Exception as e:
			return {'error': str(e)}, 500


class CostCenterApi(Resource):
	"""Individual Cost Center API"""
	
	def get(self, cost_center_id):
		"""Get cost center by ID"""
		cost_center = CFCACostCenter.query.get(cost_center_id)
		if not cost_center:
			return {'error': 'Cost center not found'}, 404
		
		schema = CACostCenterSchema()
		result = schema.dump(cost_center)
		
		# Add budget variance information
		variance = cost_center.calculate_budget_variance()
		result['budget_variance'] = {
			'variance_amount': float(variance['variance_amount']),
			'variance_percent': float(variance['variance_percent']),
			'is_favorable': variance['is_favorable'],
			'is_significant': variance['is_significant']
		}
		
		return result
	
	def put(self, cost_center_id):
		"""Update cost center"""
		cost_center = CFCACostCenter.query.get(cost_center_id)
		if not cost_center:
			return {'error': 'Cost center not found'}, 404
		
		schema = CACostCenterSchema()
		try:
			data = schema.load(request.json, partial=True)
		except Exception as e:
			return {'error': str(e)}, 400
		
		for key, value in data.items():
			setattr(cost_center, key, value)
		
		db.session.commit()
		return schema.dump(cost_center)
	
	def delete(self, cost_center_id):
		"""Delete cost center (soft delete)"""
		cost_center = CFCACostCenter.query.get(cost_center_id)
		if not cost_center:
			return {'error': 'Cost center not found'}, 404
		
		cost_center.is_active = False
		db.session.commit()
		
		return {'message': 'Cost center deactivated'}, 200


class CostAllocationApi(Resource):
	"""Cost Allocation API"""
	
	def post(self):
		"""Execute cost allocation"""
		data = request.json
		
		required_fields = ['source_center_id', 'allocation_method', 'cost_amount', 'period']
		for field in required_fields:
			if field not in data:
				return {'error': f'Missing required field: {field}'}, 400
		
		try:
			tenant_id = data.get('tenant_id', 'default_tenant')
			service = CostAccountingService(tenant_id=tenant_id)
			
			allocation_request = CostAllocationRequest(
				source_center_id=data['source_center_id'],
				allocation_method=data['allocation_method'],
				cost_amount=Decimal(str(data['cost_amount'])),
				period=data['period'],
				allocation_basis=data.get('allocation_basis'),
				target_centers=data.get('target_centers'),
				cost_driver_id=data.get('cost_driver_id')
			)
			
			result = service.execute_cost_allocation(allocation_request)
			
			# Convert Decimal values to float for JSON serialization
			for allocation in result['allocations']:
				allocation['allocated_amount'] = float(allocation['allocated_amount'])
			
			result['total_cost'] = float(result['total_cost'])
			result['total_allocated'] = float(result['total_allocated'])
			result['unallocated_amount'] = float(result['unallocated_amount'])
			
			return result, 200
			
		except Exception as e:
			return {'error': str(e)}, 500


class JobCostApi(Resource):
	"""Job Costing API"""
	
	def get(self):
		"""Get job costs with optional filtering"""
		tenant_id = request.args.get('tenant_id', 'default_tenant')
		status = request.args.get('status')
		job_number = request.args.get('job_number')
		
		service = CostAccountingService(tenant_id=tenant_id)
		
		if job_number:
			try:
				job_summary = service.get_job_cost_summary(job_number)
				return {
					'job_number': job_summary.job_number,
					'job_name': job_summary.job_name,
					'total_cost': float(job_summary.total_cost),
					'budgeted_cost': float(job_summary.budgeted_cost),
					'percent_complete': float(job_summary.percent_complete),
					'profitability': {
						k: float(v) if isinstance(v, Decimal) else v
						for k, v in job_summary.profitability.items()
					}
				}
			except ValueError as e:
				return {'error': str(e)}, 404
		
		else:
			jobs = service.get_jobs_by_status(status)
			
			result = []
			for job in jobs:
				result.append({
					'job_number': job['job_number'],
					'job_name': job['job_name'],
					'status': job['status'],
					'total_cost': float(job['total_cost']),
					'budgeted_cost': float(job['budgeted_cost']),
					'percent_complete': float(job['percent_complete']),
					'is_profitable': job['is_profitable'],
					'is_over_budget': job['is_over_budget'],
					'profit_margin': float(job['profit_margin'])
				})
			
			return {
				'data': result,
				'count': len(result),
				'status_filter': status
			}
	
	def post(self):
		"""Create new job cost"""
		schema = CAJobCostSchema()
		try:
			data = schema.load(request.json)
		except Exception as e:
			return {'error': str(e)}, 400
		
		tenant_id = request.json.get('tenant_id', 'default_tenant')
		service = CostAccountingService(tenant_id=tenant_id)
		
		try:
			job_cost = service.create_job_cost(data)
			return schema.dump(job_cost), 201
		except Exception as e:
			return {'error': str(e)}, 500
	
	def put(self):
		"""Update job costs"""
		data = request.json
		
		if 'job_number' not in data or 'cost_updates' not in data:
			return {'error': 'job_number and cost_updates are required'}, 400
		
		tenant_id = data.get('tenant_id', 'default_tenant')
		service = CostAccountingService(tenant_id=tenant_id)
		
		try:
			# Convert string values to Decimal for cost updates
			cost_updates = {}
			for category, updates in data['cost_updates'].items():
				cost_updates[category] = {
					key: Decimal(str(value)) if isinstance(value, (int, float, str)) else value
					for key, value in updates.items()
				}
			
			updated_jobs = service.update_job_costs(data['job_number'], cost_updates)
			
			return {
				'message': f'Updated {len(updated_jobs)} job cost categories',
				'job_number': data['job_number'],
				'updated_categories': len(updated_jobs)
			}, 200
			
		except Exception as e:
			return {'error': str(e)}, 500


class VarianceAnalysisApi(Resource):
	"""Variance Analysis API"""
	
	def get(self):
		"""Get variance analysis reports"""
		tenant_id = request.args.get('tenant_id', 'default_tenant')
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		cost_object_type = request.args.get('cost_object_type')
		
		service = CostAccountingService(tenant_id=tenant_id)
		variance_reports = service.get_variance_report(period, cost_object_type)
		
		result = []
		for report in variance_reports:
			result.append({
				'cost_object': report.cost_object,
				'period': report.period,
				'total_variance': float(report.total_variance),
				'variance_percent': float(report.variance_percent),
				'is_significant': report.is_significant,
				'primary_variances': [
					{
						'variance_type': pv['variance_type'],
						'variance_amount': float(pv['variance_amount']),
						'variance_percent': float(pv['variance_percent']),
						'is_favorable': pv['is_favorable'],
						'primary_cause': pv['primary_cause'],
						'requires_action': pv['requires_action']
					}
					for pv in report.primary_variances
				]
			})
		
		return {
			'data': result,
			'count': len(result),
			'period': period,
			'cost_object_type_filter': cost_object_type
		}
	
	def post(self):
		"""Perform variance analysis"""
		data = request.json
		
		required_fields = ['standard_cost_id', 'actual_cost', 'actual_quantity', 'analysis_period']
		for field in required_fields:
			if field not in data:
				return {'error': f'Missing required field: {field}'}, 400
		
		tenant_id = data.get('tenant_id', 'default_tenant')
		service = CostAccountingService(tenant_id=tenant_id)
		
		try:
			# Convert numeric values to Decimal
			analysis_data = data.copy()
			analysis_data['actual_cost'] = Decimal(str(data['actual_cost']))
			analysis_data['actual_quantity'] = Decimal(str(data['actual_quantity']))
			
			variance_analysis = service.perform_variance_analysis(analysis_data)
			
			# Calculate variance components
			components = variance_analysis.calculate_variance_components()
			
			result = {
				'variance_id': variance_analysis.variance_id,
				'cost_object': variance_analysis.cost_object_name,
				'analysis_period': variance_analysis.analysis_period,
				'total_variance': float(variance_analysis.total_variance),
				'variance_percent': float(variance_analysis.variance_percent),
				'is_favorable': variance_analysis.is_favorable,
				'is_significant': variance_analysis.is_significant,
				'variance_components': {
					'price_rate_variance': float(components['price_rate_variance']),
					'quantity_efficiency_variance': float(components['quantity_efficiency_variance']),
					'volume_variance': float(components['volume_variance']),
					'favorable_price': components['favorable_price'],
					'favorable_quantity': components['favorable_quantity'],
					'favorable_total': components['favorable_total']
				},
				'potential_causes': variance_analysis.get_potential_causes(),
				'recommended_actions': variance_analysis.recommend_actions(),
				'significance_rating': variance_analysis.get_variance_significance()
			}
			
			return result, 201
			
		except Exception as e:
			return {'error': str(e)}, 500


class ABCAnalysisApi(Resource):
	"""Activity-Based Costing Analysis API"""
	
	def get(self):
		"""Get ABC profitability analysis"""
		tenant_id = request.args.get('tenant_id', 'default_tenant')
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id=tenant_id)
		abc_data = service.get_abc_profitability_analysis(period)
		
		# Convert Decimal values to float for JSON serialization
		result = {
			'period': abc_data['period'],
			'total_activity_costs': float(abc_data['total_activity_costs']),
			'products': []
		}
		
		for product in abc_data['products']:
			product_data = {
				'product_code': product['product_code'],
				'product_name': product['product_name'],
				'total_abc_cost': float(product['total_abc_cost']),
				'cost_per_unit': float(product['cost_per_unit']),
				'activity_breakdown': []
			}
			
			for activity in product['activity_breakdown']:
				product_data['activity_breakdown'].append({
					'activity_code': activity['activity_code'],
					'activity_name': activity['activity_name'],
					'consumption': float(activity['consumption']),
					'activity_rate': float(activity['activity_rate']),
					'allocated_cost': float(activity['allocated_cost'])
				})
			
			result['products'].append(product_data)
		
		return result


class DashboardApi(Resource):
	"""Cost Accounting Dashboard API"""
	
	def get(self):
		"""Get dashboard data"""
		tenant_id = request.args.get('tenant_id', 'default_tenant')
		period = request.args.get('period', datetime.now().strftime('%Y-%m'))
		
		service = CostAccountingService(tenant_id=tenant_id)
		dashboard_data = service.generate_cost_dashboard_data(period)
		
		# Convert Decimal values to float for JSON serialization
		result = {
			'period': dashboard_data['period'],
			'summary_metrics': dashboard_data['summary_metrics'],
			'budget_performance': []
		}
		
		for bp in dashboard_data['budget_performance']:
			result['budget_performance'].append({
				'center_code': bp['center_code'],
				'center_name': bp['center_name'],
				'budget_variance': float(bp['budget_variance']),
				'variance_percent': float(bp['variance_percent']),
				'is_favorable': bp['is_favorable']
			})
		
		result['top_variances'] = []
		for tv in dashboard_data['top_variances']:
			result['top_variances'].append({
				'cost_object': tv['cost_object'],
				'variance_type': tv['variance_type'],
				'variance_amount': float(tv['variance_amount']),
				'variance_percent': float(tv['variance_percent']),
				'is_favorable': tv['is_favorable'],
				'is_significant': tv['is_significant']
			})
		
		result['activity_utilization'] = dashboard_data['activity_utilization']
		result['job_status_summary'] = dashboard_data['job_status_summary']
		
		return result


# Flask Blueprint for API
def create_api_blueprint() -> Blueprint:
	"""Create API blueprint"""
	
	api_bp = Blueprint('cost_accounting_api', __name__, url_prefix='/api/cost_accounting')
	api = Api(api_bp)
	
	# Register API endpoints
	api.add_resource(CostCenterListApi, '/cost_centers')
	api.add_resource(CostCenterApi, '/cost_centers/<string:cost_center_id>')
	api.add_resource(CostAllocationApi, '/allocations')
	api.add_resource(JobCostApi, '/job_costs')
	api.add_resource(VarianceAnalysisApi, '/variance_analysis')
	api.add_resource(ABCAnalysisApi, '/abc_analysis')
	api.add_resource(DashboardApi, '/dashboard')
	
	return api_bp


# Additional utility functions for API

def serialize_decimal(obj):
	"""Convert Decimal objects to float for JSON serialization"""
	if isinstance(obj, Decimal):
		return float(obj)
	elif isinstance(obj, dict):
		return {key: serialize_decimal(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [serialize_decimal(item) for item in obj]
	else:
		return obj


def validate_period_format(period_str: str) -> bool:
	"""Validate period format (YYYY-MM)"""
	try:
		datetime.strptime(period_str, '%Y-%m')
		return True
	except ValueError:
		return False


def get_tenant_id_from_request() -> str:
	"""Extract tenant ID from request (placeholder implementation)"""
	# In a real implementation, this would extract tenant ID from JWT token,
	# session, or other authentication mechanism
	return request.args.get('tenant_id', 'default_tenant')