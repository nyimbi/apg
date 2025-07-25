"""
Fixed Asset Management REST API

REST API endpoints for Fixed Asset Management functionality.
Provides programmatic access to FAM operations including asset lifecycle
management, depreciation calculations, maintenance scheduling, and reporting.
"""

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from marshmallow import Schema, fields, ValidationError
from datetime import date, datetime
from typing import Dict, List, Any
from decimal import Decimal

from .models import (
	CFAMAsset, CFAMAssetCategory, CFAMDepreciationMethod, CFAMDepreciation,
	CFAMAssetAcquisition, CFAMAssetDisposal, CFAMAssetTransfer, CFAMAssetMaintenance,
	CFAMAssetInsurance, CFAMAssetValuation, CFAMAssetLease
)
from .service import FixedAssetManagementService
from ...auth_rbac.models import db


# Marshmallow Schemas for API serialization

class FAMAssetSchema(Schema):
	"""Schema for FAM Asset serialization"""
	asset_id = fields.String(dump_only=True)
	asset_number = fields.String(required=True)
	asset_tag = fields.String(allow_none=True)
	asset_name = fields.String(required=True)
	description = fields.String(allow_none=True)
	category_id = fields.String(required=True)
	acquisition_cost = fields.Decimal(required=True)
	salvage_value = fields.Decimal(default=0.00)
	current_book_value = fields.Decimal(dump_only=True)
	accumulated_depreciation = fields.Decimal(dump_only=True)
	currency_code = fields.String(default='USD')
	acquisition_date = fields.Date(required=True)
	placed_in_service_date = fields.Date(allow_none=True)
	disposal_date = fields.Date(dump_only=True)
	depreciation_method_id = fields.String(allow_none=True)
	useful_life_years = fields.Integer(allow_none=True)
	useful_life_months = fields.Integer(allow_none=True)
	location = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	cost_center = fields.String(allow_none=True)
	custodian = fields.String(allow_none=True)
	manufacturer = fields.String(allow_none=True)
	model = fields.String(allow_none=True)
	serial_number = fields.String(allow_none=True)
	year_manufactured = fields.Integer(allow_none=True)
	condition = fields.String(default='Good')
	status = fields.String(default='Active')
	is_depreciable = fields.Boolean(default=True)
	is_fully_depreciated = fields.Boolean(dump_only=True)
	is_leased = fields.Boolean(default=False)
	is_insured = fields.Boolean(default=False)
	notes = fields.String(allow_none=True)


class FAMAssetCategorySchema(Schema):
	"""Schema for FAM Asset Category serialization"""
	category_id = fields.String(dump_only=True)
	category_code = fields.String(required=True)
	category_name = fields.String(required=True)
	description = fields.String(allow_none=True)
	parent_category_id = fields.String(allow_none=True)
	default_useful_life_years = fields.Integer(default=5)
	default_useful_life_months = fields.Integer(default=0)
	default_depreciation_method_id = fields.String(allow_none=True)
	default_salvage_percent = fields.Decimal(default=0.00)
	is_active = fields.Boolean(default=True)
	allow_depreciation = fields.Boolean(default=True)
	require_location = fields.Boolean(default=True)
	require_custodian = fields.Boolean(default=False)
	minimum_cost = fields.Decimal(default=0.00)


class FAMDepreciationMethodSchema(Schema):
	"""Schema for FAM Depreciation Method serialization"""
	method_id = fields.String(dump_only=True)
	method_code = fields.String(required=True)
	method_name = fields.String(required=True)
	description = fields.String(allow_none=True)
	formula = fields.String(required=True)
	depreciation_rate = fields.Decimal(allow_none=True)
	convention = fields.String(default='half_year')
	is_active = fields.Boolean(default=True)
	is_system = fields.Boolean(dump_only=True)


class FAMAssetAcquisitionSchema(Schema):
	"""Schema for FAM Asset Acquisition serialization"""
	acquisition_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	acquisition_number = fields.String(dump_only=True)
	acquisition_type = fields.String(default='Purchase')
	acquisition_date = fields.Date(required=True)
	gross_cost = fields.Decimal(required=True)
	freight_cost = fields.Decimal(default=0.00)
	installation_cost = fields.Decimal(default=0.00)
	other_costs = fields.Decimal(default=0.00)
	total_cost = fields.Decimal(dump_only=True)
	vendor_name = fields.String(allow_none=True)
	vendor_id = fields.String(allow_none=True)
	purchase_order_number = fields.String(allow_none=True)
	invoice_number = fields.String(allow_none=True)
	invoice_date = fields.Date(allow_none=True)
	funding_source = fields.String(allow_none=True)
	project_id = fields.String(allow_none=True)
	department = fields.String(allow_none=True)
	cost_center = fields.String(allow_none=True)
	description = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)
	approved = fields.Boolean(dump_only=True)
	is_posted = fields.Boolean(dump_only=True)


class FAMAssetDisposalSchema(Schema):
	"""Schema for FAM Asset Disposal serialization"""
	disposal_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	disposal_number = fields.String(dump_only=True)
	disposal_date = fields.Date(required=True)
	disposal_method = fields.String(required=True)
	disposal_reason = fields.String(allow_none=True)
	book_value_at_disposal = fields.Decimal(dump_only=True)
	accumulated_depreciation_at_disposal = fields.Decimal(dump_only=True)
	disposal_proceeds = fields.Decimal(default=0.00)
	disposal_costs = fields.Decimal(default=0.00)
	net_proceeds = fields.Decimal(dump_only=True)
	gain_loss_amount = fields.Decimal(dump_only=True)
	is_gain = fields.Boolean(dump_only=True)
	purchaser_name = fields.String(allow_none=True)
	purchaser_contact = fields.String(allow_none=True)
	description = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)
	approved = fields.Boolean(dump_only=True)
	is_posted = fields.Boolean(dump_only=True)


class FAMAssetTransferSchema(Schema):
	"""Schema for FAM Asset Transfer serialization"""
	transfer_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	transfer_number = fields.String(dump_only=True)
	transfer_date = fields.Date(required=True)
	transfer_type = fields.String(default='Location')
	reason = fields.String(allow_none=True)
	from_location = fields.String(dump_only=True)
	from_department = fields.String(dump_only=True)
	from_cost_center = fields.String(dump_only=True)
	from_custodian = fields.String(dump_only=True)
	to_location = fields.String(allow_none=True)
	to_department = fields.String(allow_none=True)
	to_cost_center = fields.String(allow_none=True)
	to_custodian = fields.String(allow_none=True)
	effective_date = fields.Date(allow_none=True)
	transfer_cost = fields.Decimal(default=0.00)
	status = fields.String(default='Completed')
	description = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)


class FAMAssetMaintenanceSchema(Schema):
	"""Schema for FAM Asset Maintenance serialization"""
	maintenance_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	maintenance_number = fields.String(dump_only=True)
	maintenance_type = fields.String(required=True)
	maintenance_date = fields.Date(required=True)
	scheduled_date = fields.Date(allow_none=True)
	completed_date = fields.Date(allow_none=True)
	service_provider = fields.String(allow_none=True)
	technician_name = fields.String(allow_none=True)
	description = fields.String(required=True)
	work_performed = fields.String(allow_none=True)
	labor_hours = fields.Decimal(default=0.00)
	labor_rate = fields.Decimal(default=0.00)
	labor_cost = fields.Decimal(dump_only=True)
	parts_cost = fields.Decimal(default=0.00)
	other_costs = fields.Decimal(default=0.00)
	total_cost = fields.Decimal(dump_only=True)
	status = fields.String(default='Scheduled')
	priority = fields.String(default='Normal')
	recurring = fields.Boolean(default=False)
	maintenance_interval_days = fields.Integer(allow_none=True)
	notes = fields.String(allow_none=True)


class FAMAssetInsuranceSchema(Schema):
	"""Schema for FAM Asset Insurance serialization"""
	insurance_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	policy_number = fields.String(required=True)
	insurance_company = fields.String(required=True)
	policy_type = fields.String(required=True)
	coverage_type = fields.String(allow_none=True)
	coverage_amount = fields.Decimal(required=True)
	deductible_amount = fields.Decimal(default=0.00)
	premium_amount = fields.Decimal(required=True)
	policy_start_date = fields.Date(required=True)
	policy_end_date = fields.Date(required=True)
	agent_name = fields.String(allow_none=True)
	agent_phone = fields.String(allow_none=True)
	agent_email = fields.String(allow_none=True)
	auto_renew = fields.Boolean(default=False)
	is_active = fields.Boolean(default=True)
	notes = fields.String(allow_none=True)


class FAMAssetValuationSchema(Schema):
	"""Schema for FAM Asset Valuation serialization"""
	valuation_id = fields.String(dump_only=True)
	asset_id = fields.String(required=True)
	valuation_date = fields.Date(required=True)
	valuation_type = fields.String(required=True)
	valuation_method = fields.String(required=True)
	valuation_purpose = fields.String(allow_none=True)
	appraised_value = fields.Decimal(required=True)
	book_value_at_valuation = fields.Decimal(dump_only=True)
	revaluation_surplus_deficit = fields.Decimal(dump_only=True)
	impairment_loss = fields.Decimal(dump_only=True)
	valuation_basis = fields.String(allow_none=True)
	appraiser_name = fields.String(allow_none=True)
	appraiser_firm = fields.String(allow_none=True)
	effective_date = fields.Date(allow_none=True)
	next_valuation_date = fields.Date(allow_none=True)
	description = fields.String(allow_none=True)
	notes = fields.String(allow_none=True)
	approved = fields.Boolean(dump_only=True)
	is_posted = fields.Boolean(dump_only=True)


class FAMAssetLeaseSchema(Schema):
	"""Schema for FAM Asset Lease serialization"""
	lease_id = fields.String(dump_only=True)
	lease_number = fields.String(required=True)
	lease_name = fields.String(required=True)
	lease_type = fields.String(required=True)
	lessor_name = fields.String(required=True)
	lessor_contact = fields.String(allow_none=True)
	lease_start_date = fields.Date(required=True)
	lease_end_date = fields.Date(required=True)
	lease_term_months = fields.Integer(dump_only=True)
	base_monthly_payment = fields.Decimal(required=True)
	escalation_rate = fields.Decimal(default=0.00)
	incremental_borrowing_rate = fields.Decimal(allow_none=True)
	discount_rate_used = fields.Decimal(required=True)
	purchase_option = fields.Boolean(default=False)
	purchase_option_price = fields.Decimal(allow_none=True)
	leased_asset_description = fields.String(allow_none=True)
	leased_asset_location = fields.String(allow_none=True)
	is_active = fields.Boolean(default=True)
	notes = fields.String(allow_none=True)


# API Resource Classes

class FAMAssetApi(Resource):
	"""Asset API Resource"""
	
	def __init__(self):
		self.schema = FAMAssetSchema()
	
	def get(self, asset_id=None):
		"""Get asset(s)"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			if asset_id:
				asset = fam_service.get_asset(asset_id)
				if not asset:
					return {'error': 'Asset not found'}, 404
				return self.schema.dump(asset)
			else:
				# Get query parameters
				status = request.args.get('status')
				category_id = request.args.get('category_id')
				location = request.args.get('location')
				include_disposed = request.args.get('include_disposed', 'false').lower() == 'true'
				
				assets = fam_service.get_assets(status, category_id, location, include_disposed)
				return {'assets': self.schema.dump(assets, many=True)}
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def post(self):
		"""Create new asset"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			asset_data = self.schema.load(request.json)
			asset = fam_service.create_asset(asset_data)
			return self.schema.dump(asset), 201
		
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def put(self, asset_id):
		"""Update asset"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			update_data = self.schema.load(request.json, partial=True)
			asset = fam_service.update_asset(asset_id, update_data)
			if not asset:
				return {'error': 'Asset not found'}, 404
			return self.schema.dump(asset)
		
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def delete(self, asset_id):
		"""Delete asset (soft delete by setting status to Retired)"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			asset = fam_service.update_asset(asset_id, {'status': 'Retired'})
			if not asset:
				return {'error': 'Asset not found'}, 404
			return {'message': 'Asset retired successfully'}
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		"""Get current tenant ID - placeholder implementation"""
		# TODO: Implement proper tenant context
		return "default_tenant"


class FAMAssetTransferApi(Resource):
	"""Asset Transfer API Resource"""
	
	def __init__(self):
		self.schema = FAMAssetTransferSchema()
	
	def post(self):
		"""Create asset transfer"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			transfer_data = self.schema.load(request.json)
			transfer = fam_service.transfer_asset(transfer_data['asset_id'], transfer_data)
			return self.schema.dump(transfer), 201
		
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		return "default_tenant"


class FAMDepreciationApi(Resource):
	"""Depreciation API Resource"""
	
	def post(self):
		"""Calculate monthly depreciation"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			as_of_date_str = request.json.get('as_of_date')
			if as_of_date_str:
				as_of_date = datetime.strptime(as_of_date_str, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			depreciation_records = fam_service.calculate_monthly_depreciation(as_of_date)
			
			result = {
				'message': f'Calculated depreciation for {len(depreciation_records)} assets',
				'records_count': len(depreciation_records),
				'calculation_date': as_of_date.isoformat()
			}
			
			# Optionally post to GL
			if request.json.get('post_to_gl', False):
				journal_ids = fam_service.post_depreciation_to_gl(depreciation_records)
				result['journal_entries_posted'] = len(journal_ids)
				result['journal_entry_ids'] = journal_ids
			
			return result
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def get(self):
		"""Get depreciation schedule"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			asset_id = request.args.get('asset_id')
			months_ahead = int(request.args.get('months_ahead', 12))
			
			schedule = fam_service.get_depreciation_schedule(asset_id, months_ahead)
			return {'depreciation_schedule': schedule}
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		return "default_tenant"


class FAMAssetMaintenanceApi(Resource):
	"""Asset Maintenance API Resource"""
	
	def __init__(self):
		self.schema = FAMAssetMaintenanceSchema()
	
	def get(self, maintenance_id=None):
		"""Get maintenance records"""
		if maintenance_id:
			maintenance = CFAMAssetMaintenance.query.filter_by(
				tenant_id=self._get_tenant_id(),
				maintenance_id=maintenance_id
			).first()
			
			if not maintenance:
				return {'error': 'Maintenance record not found'}, 404
			
			return self.schema.dump(maintenance)
		else:
			# Get upcoming maintenance
			tenant_id = self._get_tenant_id()
			fam_service = FixedAssetManagementService(tenant_id)
			days_ahead = int(request.args.get('days_ahead', 90))
			
			maintenance_schedule = fam_service.get_maintenance_schedule(days_ahead)
			return {'maintenance_schedule': self.schema.dump(maintenance_schedule, many=True)}
	
	def post(self):
		"""Schedule maintenance"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			maintenance_data = self.schema.load(request.json)
			maintenance = fam_service.schedule_maintenance(maintenance_data)
			return self.schema.dump(maintenance), 201
		
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def put(self, maintenance_id):
		"""Complete maintenance"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			completion_data = request.json
			maintenance = fam_service.complete_maintenance(maintenance_id, completion_data)
			return self.schema.dump(maintenance)
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		return "default_tenant"


class FAMAssetInsuranceApi(Resource):
	"""Asset Insurance API Resource"""
	
	def __init__(self):
		self.schema = FAMAssetInsuranceSchema()
	
	def get(self):
		"""Get insurance renewals due"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		days_ahead = int(request.args.get('days_ahead', 30))
		
		try:
			renewals = fam_service.get_insurance_renewals_due(days_ahead)
			return {'insurance_renewals': self.schema.dump(renewals, many=True)}
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def post(self):
		"""Add insurance policy"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			insurance_data = self.schema.load(request.json)
			insurance = fam_service.add_insurance_policy(insurance_data)
			return self.schema.dump(insurance), 201
		
		except ValidationError as e:
			return {'error': 'Validation error', 'details': e.messages}, 400
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		return "default_tenant"


class FAMDashboardApi(Resource):
	"""Dashboard API Resource"""
	
	def get(self):
		"""Get dashboard summary"""
		tenant_id = self._get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			endpoint = request.args.get('endpoint', 'summary')
			
			if endpoint == 'summary':
				summary = fam_service.get_asset_summary()
				return summary
			
			elif endpoint == 'maintenance_alerts':
				days_ahead = int(request.args.get('days_ahead', 30))
				maintenance_schedule = fam_service.get_maintenance_schedule(days_ahead)
				
				alerts = []
				for maintenance in maintenance_schedule:
					days_until = (maintenance.scheduled_date - date.today()).days
					alerts.append({
						'asset_number': maintenance.asset.asset_number,
						'asset_name': maintenance.asset.asset_name,
						'maintenance_type': maintenance.maintenance_type,
						'scheduled_date': maintenance.scheduled_date.isoformat(),
						'days_until': days_until,
						'priority': maintenance.priority,
						'status': maintenance.status
					})
				
				return {'maintenance_alerts': alerts}
			
			elif endpoint == 'insurance_alerts':
				days_ahead = int(request.args.get('days_ahead', 60))
				renewals = fam_service.get_insurance_renewals_due(days_ahead)
				
				alerts = []
				for renewal in renewals:
					days_until = (renewal.policy_end_date - date.today()).days
					alerts.append({
						'asset_number': renewal.asset.asset_number,
						'asset_name': renewal.asset.asset_name,
						'policy_number': renewal.policy_number,
						'insurance_company': renewal.insurance_company,
						'policy_end_date': renewal.policy_end_date.isoformat(),
						'days_until': days_until,
						'coverage_amount': float(renewal.coverage_amount)
					})
				
				return {'insurance_alerts': alerts}
			
			elif endpoint == 'assets_by_location':
				assets_by_location = fam_service.get_assets_by_location()
				return {'assets_by_location': assets_by_location}
			
			else:
				return {'error': 'Invalid endpoint'}, 400
		
		except Exception as e:
			return {'error': str(e)}, 500
	
	def _get_tenant_id(self):
		return "default_tenant"


# Additional API Resource Classes (simplified for brevity)

class FAMAssetCategoryApi(Resource):
	def __init__(self):
		self.schema = FAMAssetCategorySchema()
	
	def get(self, category_id=None):
		tenant_id = self._get_tenant_id()
		if category_id:
			category = CFAMAssetCategory.query.filter_by(
				tenant_id=tenant_id, category_id=category_id
			).first()
			return self.schema.dump(category) if category else {'error': 'Not found'}, 404
		else:
			categories = CFAMAssetCategory.query.filter_by(tenant_id=tenant_id).all()
			return {'categories': self.schema.dump(categories, many=True)}
	
	def _get_tenant_id(self):
		return "default_tenant"


class FAMDepreciationMethodApi(Resource):
	def __init__(self):
		self.schema = FAMDepreciationMethodSchema()
	
	def get(self, method_id=None):
		tenant_id = self._get_tenant_id()
		if method_id:
			method = CFAMDepreciationMethod.query.filter_by(
				tenant_id=tenant_id, method_id=method_id
			).first()
			return self.schema.dump(method) if method else {'error': 'Not found'}, 404
		else:
			methods = CFAMDepreciationMethod.query.filter_by(tenant_id=tenant_id).all()
			return {'methods': self.schema.dump(methods, many=True)}
	
	def _get_tenant_id(self):
		return "default_tenant"


def create_api_blueprint() -> Blueprint:
	"""Create Fixed Asset Management API blueprint"""
	
	api_bp = Blueprint('fam_api', __name__, url_prefix='/api/core_financials/fam')
	api = Api(api_bp)
	
	# Asset endpoints
	api.add_resource(FAMAssetApi, '/assets', '/assets/<string:asset_id>')
	api.add_resource(FAMAssetTransferApi, '/assets/transfer')
	
	# Category endpoints
	api.add_resource(FAMAssetCategoryApi, '/categories', '/categories/<string:category_id>')
	
	# Depreciation Method endpoints
	api.add_resource(FAMDepreciationMethodApi, '/methods', '/methods/<string:method_id>')
	
	# Depreciation endpoints
	api.add_resource(FAMDepreciationApi, '/depreciation')
	
	# Maintenance endpoints
	api.add_resource(FAMAssetMaintenanceApi, '/maintenance', '/maintenance/<string:maintenance_id>')
	
	# Insurance endpoints
	api.add_resource(FAMAssetInsuranceApi, '/insurance')
	
	# Dashboard endpoints
	api.add_resource(FAMDashboardApi, '/dashboard')
	
	return api_bp


# Flask-AppBuilder API Classes (for integration with AppBuilder security)

class FAMAssetAppBuilderApi(BaseApi):
	"""Flask-AppBuilder API for Assets"""
	
	resource_name = 'asset'
	datamodel = SQLAInterface(CFAMAsset)
	
	@expose('/list')
	def list(self):
		"""List assets"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			assets = fam_service.get_assets()
			schema = FAMAssetSchema()
			return self.response(200, assets=schema.dump(assets, many=True))
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<string:asset_id>')
	def get(self, asset_id):
		"""Get specific asset"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			asset = fam_service.get_asset(asset_id)
			if not asset:
				return self.response_404()
			
			schema = FAMAssetSchema()
			return self.response(200, **schema.dump(asset))
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/', methods=['POST'])
	def post(self):
		"""Create new asset"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			schema = FAMAssetSchema()
			asset_data = schema.load(request.json)
			asset = fam_service.create_asset(asset_data)
			return self.response(201, **schema.dump(asset))
		except ValidationError as e:
			return self.response_400(message="Validation error", errors=e.messages)
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<string:asset_id>/transfer', methods=['POST'])
	def transfer(self, asset_id):
		"""Transfer asset"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			transfer_data = request.json
			transfer_data['asset_id'] = asset_id
			transfer = fam_service.transfer_asset(asset_id, transfer_data)
			
			schema = FAMAssetTransferSchema()
			return self.response(201, **schema.dump(transfer))
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/<string:asset_id>/depreciation_history')
	def depreciation_history(self, asset_id):
		"""Get asset depreciation history"""
		try:
			depreciation_records = CFAMDepreciation.query.filter_by(
				tenant_id=self.get_tenant_id(),
				asset_id=asset_id
			).order_by(CFAMDepreciation.depreciation_date.desc()).all()
			
			history = []
			for record in depreciation_records:
				history.append({
					'depreciation_date': record.depreciation_date.isoformat(),
					'period_name': record.period_name,
					'depreciation_amount': float(record.depreciation_amount),
					'accumulated_depreciation': float(record.accumulated_depreciation_after),
					'book_value': float(record.ending_book_value),
					'method': record.method.method_name if record.method else None,
					'is_posted': record.is_posted
				})
			
			return self.response(200, depreciation_history=history)
		except Exception as e:
			return self.response_400(message=str(e))
	
	def get_tenant_id(self):
		"""Get current tenant ID"""
		# TODO: Implement proper tenant context from security context
		return "default_tenant"


class FAMDepreciationAppBuilderApi(BaseApi):
	"""Flask-AppBuilder API for Depreciation"""
	
	resource_name = 'depreciation'
	
	@expose('/calculate', methods=['POST'])
	def calculate(self):
		"""Calculate monthly depreciation"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			data = request.json or {}
			as_of_date_str = data.get('as_of_date')
			
			if as_of_date_str:
				as_of_date = datetime.strptime(as_of_date_str, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			depreciation_records = fam_service.calculate_monthly_depreciation(as_of_date)
			
			result = {
				'message': f'Calculated depreciation for {len(depreciation_records)} assets',
				'records_count': len(depreciation_records),
				'calculation_date': as_of_date.isoformat()
			}
			
			if data.get('post_to_gl', False):
				journal_ids = fam_service.post_depreciation_to_gl(depreciation_records)
				result['journal_entries_posted'] = len(journal_ids)
				result['journal_entry_ids'] = journal_ids
			
			return self.response(200, **result)
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/schedule')
	def schedule(self):
		"""Get depreciation schedule"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			asset_id = request.args.get('asset_id')
			months_ahead = int(request.args.get('months_ahead', 12))
			
			schedule = fam_service.get_depreciation_schedule(asset_id, months_ahead)
			return self.response(200, depreciation_schedule=schedule)
		except Exception as e:
			return self.response_400(message=str(e))
	
	def get_tenant_id(self):
		return "default_tenant"


class FAMDashboardAppBuilderApi(BaseApi):
	"""Flask-AppBuilder API for Dashboard"""
	
	resource_name = 'dashboard'
	
	@expose('/summary')
	def summary(self):
		"""Get asset summary"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			summary = fam_service.get_asset_summary()
			return self.response(200, **summary)
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/maintenance_alerts')
	def maintenance_alerts(self):
		"""Get maintenance alerts"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			days_ahead = int(request.args.get('days_ahead', 30))
			maintenance_schedule = fam_service.get_maintenance_schedule(days_ahead)
			
			alerts = []
			for maintenance in maintenance_schedule:
				days_until = (maintenance.scheduled_date - date.today()).days
				alerts.append({
					'asset_number': maintenance.asset.asset_number,
					'asset_name': maintenance.asset.asset_name,
					'maintenance_type': maintenance.maintenance_type,
					'scheduled_date': maintenance.scheduled_date.isoformat(),
					'days_until': days_until,
					'priority': maintenance.priority,
					'status': maintenance.status,
					'service_provider': maintenance.service_provider
				})
			
			return self.response(200, maintenance_alerts=alerts)
		except Exception as e:
			return self.response_400(message=str(e))
	
	@expose('/insurance_alerts')
	def insurance_alerts(self):
		"""Get insurance renewal alerts"""
		tenant_id = self.get_tenant_id()
		fam_service = FixedAssetManagementService(tenant_id)
		
		try:
			days_ahead = int(request.args.get('days_ahead', 60))
			renewals = fam_service.get_insurance_renewals_due(days_ahead)
			
			alerts = []
			for renewal in renewals:
				days_until = (renewal.policy_end_date - date.today()).days
				alerts.append({
					'asset_number': renewal.asset.asset_number,
					'asset_name': renewal.asset.asset_name,
					'policy_number': renewal.policy_number,
					'insurance_company': renewal.insurance_company,
					'policy_end_date': renewal.policy_end_date.isoformat(),
					'days_until': days_until,
					'coverage_amount': float(renewal.coverage_amount),
					'premium_amount': float(renewal.premium_amount)
				})
			
			return self.response(200, insurance_alerts=alerts)
		except Exception as e:
			return self.response_400(message=str(e))
	
	def get_tenant_id(self):
		return "default_tenant"