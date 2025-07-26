"""
Budgeting & Forecasting API

REST API endpoints for the Budgeting & Forecasting sub-capability
providing comprehensive API access to all budgeting and forecasting operations.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, List, Optional
import json
from sqlalchemy.exc import IntegrityError
from marshmallow import Schema, fields as ma_fields, validate

from .models import (
	CFBFBudget, CFBFBudgetLine, CFBFBudgetScenario, CFBFBudgetVersion,
	CFBFForecast, CFBFForecastLine, CFBFActualVsBudget, CFBFDrivers,
	CFBFTemplate, CFBFApproval, CFBFAllocation
)
from .service import CFBFBudgetService, CFBFVarianceAnalysisService, CFBFForecastService, CFBFDriverService
from ...auth_rbac.decorators import require_permission
from ...database import db


# Create API blueprint
budgeting_forecasting_api = Blueprint(
	'budgeting_forecasting_api',
	__name__,
	url_prefix='/api/core_financials/budgeting_forecasting'
)

# Create Flask-RESTX API
api = Api(
	budgeting_forecasting_api,
	version='1.0',
	title='Budgeting & Forecasting API',
	description='API endpoints for budgeting and forecasting operations',
	doc='/doc/'
)

# Create namespaces
budgets_ns = Namespace('budgets', description='Budget operations')
forecasts_ns = Namespace('forecasts', description='Forecast operations')
scenarios_ns = Namespace('scenarios', description='Scenario operations')
drivers_ns = Namespace('drivers', description='Driver operations')
variance_ns = Namespace('variance', description='Variance analysis operations')
templates_ns = Namespace('templates', description='Template operations')
approvals_ns = Namespace('approvals', description='Approval operations')

api.add_namespace(budgets_ns)
api.add_namespace(forecasts_ns)
api.add_namespace(scenarios_ns)
api.add_namespace(drivers_ns)
api.add_namespace(variance_ns)
api.add_namespace(templates_ns)
api.add_namespace(approvals_ns)


# Marshmallow Schemas for serialization/deserialization
class BudgetSchema(Schema):
	"""Budget serialization schema"""
	budget_id = ma_fields.Str(dump_only=True)
	budget_number = ma_fields.Str(dump_only=True)
	budget_name = ma_fields.Str(required=True, validate=validate.Length(min=1, max=200))
	description = ma_fields.Str(allow_none=True)
	fiscal_year = ma_fields.Int(required=True, validate=validate.Range(min=2000, max=2100))
	budget_period = ma_fields.Str(validate=validate.OneOf(['Annual', 'Quarterly', 'Monthly']))
	start_date = ma_fields.Date(required=True)
	end_date = ma_fields.Date(required=True)
	scenario_id = ma_fields.Str(allow_none=True)
	template_id = ma_fields.Str(allow_none=True)
	status = ma_fields.Str(dump_only=True)
	total_revenue = ma_fields.Decimal(dump_only=True)
	total_expenses = ma_fields.Decimal(dump_only=True)
	net_income = ma_fields.Decimal(dump_only=True)
	currency_code = ma_fields.Str(validate=validate.Length(equal=3))
	notes = ma_fields.Str(allow_none=True)
	created_on = ma_fields.DateTime(dump_only=True)


class BudgetLineSchema(Schema):
	"""Budget line serialization schema"""
	budget_line_id = ma_fields.Str(dump_only=True)
	budget_id = ma_fields.Str(required=True)
	line_number = ma_fields.Int(required=True)
	description = ma_fields.Str(allow_none=True)
	account_id = ma_fields.Str(required=True)
	driver_id = ma_fields.Str(allow_none=True)
	calculation_method = ma_fields.Str(validate=validate.OneOf(['Manual', 'Driver-Based', 'Formula']))
	calculation_formula = ma_fields.Str(allow_none=True)
	annual_amount = ma_fields.Decimal(required=True)
	total_amount = ma_fields.Decimal(dump_only=True)
	distribution_method = ma_fields.Str(validate=validate.OneOf(['Even', 'Seasonal', 'Custom']))
	cost_center = ma_fields.Str(allow_none=True)
	department = ma_fields.Str(allow_none=True)
	project = ma_fields.Str(allow_none=True)


class ForecastSchema(Schema):
	"""Forecast serialization schema"""
	forecast_id = ma_fields.Str(dump_only=True)
	forecast_number = ma_fields.Str(dump_only=True)
	forecast_name = ma_fields.Str(required=True, validate=validate.Length(min=1, max=200))
	description = ma_fields.Str(allow_none=True)
	forecast_type = ma_fields.Str(validate=validate.OneOf(['Rolling', 'Static', 'Budget-Based']))
	forecast_method = ma_fields.Str(validate=validate.OneOf(['Trend', 'Driver', 'Statistical']))
	periods_ahead = ma_fields.Int(validate=validate.Range(min=1, max=60))
	scenario_id = ma_fields.Str(allow_none=True)
	base_budget_id = ma_fields.Str(allow_none=True)
	confidence_level = ma_fields.Decimal(validate=validate.Range(min=0, max=100))
	algorithm_type = ma_fields.Str(validate=validate.OneOf(['Linear', 'Exponential', 'Seasonal', 'Regression']))


class VarianceAnalysisSchema(Schema):
	"""Variance analysis serialization schema"""
	variance_id = ma_fields.Str(dump_only=True)
	budget_id = ma_fields.Str(required=True)
	account_id = ma_fields.Str(required=True)
	analysis_date = ma_fields.Date(required=True)
	budget_amount = ma_fields.Decimal(dump_only=True)
	actual_amount = ma_fields.Decimal(dump_only=True)
	variance_amount = ma_fields.Decimal(dump_only=True)
	variance_percent = ma_fields.Decimal(dump_only=True)
	is_favorable = ma_fields.Bool(dump_only=True)
	is_significant = ma_fields.Bool(dump_only=True)
	alert_level = ma_fields.Str(dump_only=True)


# Initialize schemas
budget_schema = BudgetSchema()
budgets_schema = BudgetSchema(many=True)
budget_line_schema = BudgetLineSchema()
budget_lines_schema = BudgetLineSchema(many=True)
forecast_schema = ForecastSchema()
forecasts_schema = ForecastSchema(many=True)
variance_schema = VarianceAnalysisSchema()
variances_schema = VarianceAnalysisSchema(many=True)


def get_tenant_id() -> str:
	"""Get tenant ID from JWT token or request context"""
	# Implementation would extract tenant ID from JWT or session
	return request.headers.get('X-Tenant-ID', 'default_tenant')


def get_user_id() -> str:
	"""Get user ID from JWT token"""
	try:
		return get_jwt_identity()
	except:
		return 'api_user'


def handle_api_error(error: Exception) -> Dict[str, Any]:
	"""Standardized API error handling"""
	if isinstance(error, IntegrityError):
		return {
			'error': 'Data integrity error',
			'message': 'The operation violates data constraints',
			'details': str(error.orig) if hasattr(error, 'orig') else str(error)
		}, 400
	elif isinstance(error, ValueError):
		return {
			'error': 'Invalid input',
			'message': str(error)
		}, 400
	else:
		return {
			'error': 'Internal server error',
			'message': 'An unexpected error occurred'
		}, 500


# Budget API Endpoints
@budgets_ns.route('/')
class BudgetListAPI(Resource):
	"""Budget list operations"""
	
	@jwt_required()
	@require_permission('bf.read')
	def get(self):
		"""Get list of budgets"""
		try:
			tenant_id = get_tenant_id()
			
			# Parse query parameters
			fiscal_year = request.args.get('fiscal_year', type=int)
			status = request.args.get('status')
			scenario_id = request.args.get('scenario_id')
			page = request.args.get('page', 1, type=int)
			per_page = min(request.args.get('per_page', 20, type=int), 100)
			
			# Build query
			query = db.session.query(CFBFBudget).filter_by(tenant_id=tenant_id)
			
			if fiscal_year:
				query = query.filter_by(fiscal_year=fiscal_year)
			if status:
				query = query.filter_by(status=status)
			if scenario_id:
				query = query.filter_by(scenario_id=scenario_id)
			
			# Paginate
			budgets = query.order_by(CFBFBudget.created_on.desc()).paginate(
				page=page, per_page=per_page, error_out=False
			)
			
			return {
				'budgets': budgets_schema.dump(budgets.items),
				'pagination': {
					'page': page,
					'pages': budgets.pages,
					'per_page': per_page,
					'total': budgets.total
				}
			}
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	@require_permission('bf.create_budget')
	def post(self):
		"""Create new budget"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			data = request.get_json()
			errors = budget_schema.validate(data)
			if errors:
				return {'error': 'Validation failed', 'details': errors}, 400
			
			# Create budget using service
			service = CFBFBudgetService(db.session, tenant_id)
			budget = service.create_budget(
				budget_name=data['budget_name'],
				fiscal_year=data['fiscal_year'],
				start_date=datetime.strptime(data['start_date'], '%Y-%m-%d').date(),
				end_date=datetime.strptime(data['end_date'], '%Y-%m-%d').date(),
				scenario_id=data.get('scenario_id'),
				template_id=data.get('template_id'),
				user_id=user_id
			)
			
			db.session.commit()
			
			return {
				'message': 'Budget created successfully',
				'budget': budget_schema.dump(budget)
			}, 201
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


@budgets_ns.route('/<string:budget_id>')
class BudgetAPI(Resource):
	"""Individual budget operations"""
	
	@jwt_required()
	@require_permission('bf.read')
	def get(self, budget_id):
		"""Get budget details"""
		try:
			tenant_id = get_tenant_id()
			
			budget = db.session.query(CFBFBudget).filter_by(
				budget_id=budget_id,
				tenant_id=tenant_id
			).first()
			
			if not budget:
				return {'error': 'Budget not found'}, 404
			
			service = CFBFBudgetService(db.session, tenant_id)
			summary = service.get_budget_summary(budget_id)
			
			return {
				'budget': budget_schema.dump(budget),
				'summary': summary
			}
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	@require_permission('bf.write')
	def put(self, budget_id):
		"""Update budget"""
		try:
			tenant_id = get_tenant_id()
			
			budget = db.session.query(CFBFBudget).filter_by(
				budget_id=budget_id,
				tenant_id=tenant_id
			).first()
			
			if not budget:
				return {'error': 'Budget not found'}, 404
			
			if budget.status not in ['Draft', 'Submitted']:
				return {'error': 'Budget cannot be modified in current status'}, 400
			
			# Validate input
			data = request.get_json()
			errors = budget_schema.validate(data, partial=True)
			if errors:
				return {'error': 'Validation failed', 'details': errors}, 400
			
			# Update fields
			for field, value in data.items():
				if hasattr(budget, field) and field not in ['budget_id', 'budget_number']:
					setattr(budget, field, value)
			
			db.session.commit()
			
			return {
				'message': 'Budget updated successfully',
				'budget': budget_schema.dump(budget)
			}
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


@budgets_ns.route('/<string:budget_id>/submit')
class BudgetSubmitAPI(Resource):
	"""Budget submission for approval"""
	
	@jwt_required()
	@require_permission('bf.submit_budget')
	def post(self, budget_id):
		"""Submit budget for approval"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			service = CFBFBudgetService(db.session, tenant_id)
			success = service.submit_budget_for_approval(budget_id, user_id)
			
			if not success:
				return {'error': 'Budget cannot be submitted'}, 400
			
			db.session.commit()
			
			return {'message': 'Budget submitted for approval successfully'}
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


@budgets_ns.route('/<string:budget_id>/lines')
class BudgetLinesAPI(Resource):
	"""Budget lines operations"""
	
	@jwt_required()
	@require_permission('bf.read')
	def get(self, budget_id):
		"""Get budget lines"""
		try:
			tenant_id = get_tenant_id()
			
			lines = db.session.query(CFBFBudgetLine).filter_by(
				budget_id=budget_id,
				tenant_id=tenant_id
			).order_by(CFBFBudgetLine.line_number).all()
			
			return {'lines': budget_lines_schema.dump(lines)}
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	@require_permission('bf.write')
	def post(self, budget_id):
		"""Add budget line"""
		try:
			tenant_id = get_tenant_id()
			
			# Validate input
			data = request.get_json()
			errors = budget_line_schema.validate(data)
			if errors:
				return {'error': 'Validation failed', 'details': errors}, 400
			
			service = CFBFBudgetService(db.session, tenant_id)
			budget_line = service.add_budget_line(
				budget_id=budget_id,
				account_id=data['account_id'],
				amount=Decimal(str(data['annual_amount'])),
				description=data.get('description'),
				driver_id=data.get('driver_id'),
				cost_center=data.get('cost_center'),
				department=data.get('department')
			)
			
			db.session.commit()
			
			return {
				'message': 'Budget line added successfully',
				'line': budget_line_schema.dump(budget_line)
			}, 201
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


# Forecast API Endpoints
@forecasts_ns.route('/')
class ForecastListAPI(Resource):
	"""Forecast list operations"""
	
	@jwt_required()
	@require_permission('bf.read')
	def get(self):
		"""Get list of forecasts"""
		try:
			tenant_id = get_tenant_id()
			
			# Parse query parameters
			forecast_type = request.args.get('forecast_type')
			status = request.args.get('status')
			page = request.args.get('page', 1, type=int)
			per_page = min(request.args.get('per_page', 20, type=int), 100)
			
			# Build query
			query = db.session.query(CFBFForecast).filter_by(tenant_id=tenant_id)
			
			if forecast_type:
				query = query.filter_by(forecast_type=forecast_type)
			if status:
				query = query.filter_by(status=status)
			
			# Paginate
			forecasts = query.order_by(CFBFForecast.created_on.desc()).paginate(
				page=page, per_page=per_page, error_out=False
			)
			
			return {
				'forecasts': forecasts_schema.dump(forecasts.items),
				'pagination': {
					'page': page,
					'pages': forecasts.pages,
					'per_page': per_page,
					'total': forecasts.total
				}
			}
			
		except Exception as e:
			return handle_api_error(e)
	
	@jwt_required()
	@require_permission('bf.create_forecast')
	def post(self):
		"""Create new forecast"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			# Validate input
			data = request.get_json()
			errors = forecast_schema.validate(data)
			if errors:
				return {'error': 'Validation failed', 'details': errors}, 400
			
			# Create forecast using service
			service = CFBFForecastService(db.session, tenant_id)
			forecast = service.create_forecast(
				forecast_name=data['forecast_name'],
				forecast_type=data.get('forecast_type', 'Rolling'),
				periods_ahead=data.get('periods_ahead', 12),
				base_budget_id=data.get('base_budget_id'),
				scenario_id=data.get('scenario_id'),
				user_id=user_id
			)
			
			db.session.commit()
			
			return {
				'message': 'Forecast created successfully',
				'forecast': forecast_schema.dump(forecast)
			}, 201
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


@forecasts_ns.route('/<string:forecast_id>/generate')
class ForecastGenerateAPI(Resource):
	"""Forecast generation operations"""
	
	@jwt_required()
	@require_permission('bf.generate_forecast')
	def post(self, forecast_id):
		"""Generate forecast calculations"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			data = request.get_json() or {}
			algorithm = data.get('algorithm', 'Linear')
			account_ids = data.get('account_ids', [])
			
			service = CFBFForecastService(db.session, tenant_id)
			
			generated_lines = []
			for account_id in account_ids:
				forecast_line = service.generate_trend_forecast(
					forecast_id=forecast_id,
					account_id=account_id,
					algorithm=algorithm
				)
				generated_lines.append(forecast_line)
			
			db.session.commit()
			
			return {
				'message': f'Generated {len(generated_lines)} forecast lines',
				'lines_generated': len(generated_lines)
			}
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


# Variance Analysis API Endpoints
@variance_ns.route('/generate')
class VarianceGenerateAPI(Resource):
	"""Variance analysis generation"""
	
	@jwt_required()
	@require_permission('bf.variance_analysis')
	def post(self):
		"""Generate variance analysis"""
		try:
			tenant_id = get_tenant_id()
			user_id = get_user_id()
			
			data = request.get_json()
			if not data or 'budget_id' not in data:
				return {'error': 'budget_id is required'}, 400
			
			budget_id = data['budget_id']
			analysis_date = datetime.strptime(
				data.get('analysis_date', date.today().isoformat()),
				'%Y-%m-%d'
			).date()
			
			service = CFBFVarianceAnalysisService(db.session, tenant_id)
			variances = service.generate_variance_analysis(
				budget_id=budget_id,
				analysis_date=analysis_date,
				user_id=user_id
			)
			
			db.session.commit()
			
			return {
				'message': f'Generated variance analysis with {len(variances)} records',
				'variances': variances_schema.dump(variances)
			}
			
		except Exception as e:
			db.session.rollback()
			return handle_api_error(e)


@variance_ns.route('/significant')
class SignificantVariancesAPI(Resource):
	"""Significant variance retrieval"""
	
	@jwt_required()
	@require_permission('bf.variance_analysis')
	def get(self):
		"""Get significant variances"""
		try:
			tenant_id = get_tenant_id()
			
			budget_id = request.args.get('budget_id')
			analysis_date = request.args.get('analysis_date', date.today().isoformat())
			alert_level = request.args.get('alert_level')
			
			if not budget_id:
				return {'error': 'budget_id parameter is required'}, 400
			
			service = CFBFVarianceAnalysisService(db.session, tenant_id)
			variances = service.get_significant_variances(
				budget_id=budget_id,
				analysis_date=datetime.strptime(analysis_date, '%Y-%m-%d').date(),
				alert_level=alert_level
			)
			
			return {'variances': variances_schema.dump(variances)}
			
		except Exception as e:
			return handle_api_error(e)


@variance_ns.route('/trends/<string:account_id>')
class VarianceTrendsAPI(Resource):
	"""Variance trend analysis"""
	
	@jwt_required()
	@require_permission('bf.variance_analysis')
	def get(self, account_id):
		"""Get variance trends for account"""
		try:
			tenant_id = get_tenant_id()
			periods = request.args.get('periods', 12, type=int)
			
			service = CFBFVarianceAnalysisService(db.session, tenant_id)
			trends = service.get_variance_trends(account_id, periods)
			
			return {'trends': trends}
			
		except Exception as e:
			return handle_api_error(e)


# Driver API Endpoints
@drivers_ns.route('/')
class DriverListAPI(Resource):
	"""Driver list operations"""
	
	@jwt_required()
	@require_permission('bf.manage_drivers')
	def get(self):
		"""Get list of drivers"""
		try:
			tenant_id = get_tenant_id()
			
			drivers = db.session.query(CFBFDrivers).filter_by(
				tenant_id=tenant_id,
				is_active=True
			).order_by(CFBFDrivers.driver_code).all()
			
			driver_data = []
			for driver in drivers:
				driver_data.append({
					'driver_id': driver.driver_id,
					'driver_code': driver.driver_code,
					'driver_name': driver.driver_name,
					'category': driver.category,
					'base_value': float(driver.base_value or 0),
					'growth_rate': float(driver.growth_rate or 0),
					'unit_of_measure': driver.unit_of_measure
				})
			
			return {'drivers': driver_data}
			
		except Exception as e:
			return handle_api_error(e)


@drivers_ns.route('/<string:driver_id>/projections')
class DriverProjectionsAPI(Resource):
	"""Driver projection calculations"""
	
	@jwt_required()
	@require_permission('bf.manage_drivers')
	def get(self, driver_id):
		"""Get driver projections"""
		try:
			tenant_id = get_tenant_id()
			periods = request.args.get('periods', 12, type=int)
			
			service = CFBFDriverService(db.session, tenant_id)
			projections = service.calculate_driver_projections(driver_id, periods)
			
			# Convert Decimal values to float for JSON serialization
			projections_data = {
				str(period): float(value) for period, value in projections.items()
			}
			
			return {'projections': projections_data}
			
		except Exception as e:
			return handle_api_error(e)


@drivers_ns.route('/<string:driver_id>/impact')
class DriverImpactAPI(Resource):
	"""Driver impact analysis"""
	
	@jwt_required()
	@require_permission('bf.manage_drivers')
	def get(self, driver_id):
		"""Get driver impact analysis"""
		try:
			tenant_id = get_tenant_id()
			
			service = CFBFDriverService(db.session, tenant_id)
			impact = service.get_driver_impact_analysis(driver_id)
			
			return {'impact': impact}
			
		except Exception as e:
			return handle_api_error(e)


# Error handlers
@budgeting_forecasting_api.errorhandler(400)
def bad_request(error):
	return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@budgeting_forecasting_api.errorhandler(401)
def unauthorized(error):
	return jsonify({'error': 'Unauthorized', 'message': 'Authentication required'}), 401


@budgeting_forecasting_api.errorhandler(403)
def forbidden(error):
	return jsonify({'error': 'Forbidden', 'message': 'Insufficient permissions'}), 403


@budgeting_forecasting_api.errorhandler(404)
def not_found(error):
	return jsonify({'error': 'Not found', 'message': 'Resource not found'}), 404


@budgeting_forecasting_api.errorhandler(500)
def internal_error(error):
	return jsonify({'error': 'Internal server error', 'message': 'An unexpected error occurred'}), 500


# Register the blueprint
def register_api_blueprint(app):
	"""Register the API blueprint with the Flask app"""
	app.register_blueprint(budgeting_forecasting_api)