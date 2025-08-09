"""
Fixed Asset Management Views

Flask-AppBuilder views for Fixed Asset Management functionality including
asset lifecycle management, depreciation, maintenance, and reporting.
"""

from flask import flash, redirect, request, url_for, jsonify, render_template
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.charts.views import DirectByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, SelectField, DecimalField, DateField, TextAreaField, BooleanField, IntegerField
from wtforms.validators import DataRequired, NumberRange, Email, Length
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
import json

from .models import (
	CFAMAsset, CFAMAssetCategory, CFAMDepreciationMethod, CFAMDepreciation,
	CFAMAssetAcquisition, CFAMAssetDisposal, CFAMAssetTransfer, CFAMAssetMaintenance,
	CFAMAssetInsurance, CFAMAssetValuation, CFAMAssetLease
)
from .service import FixedAssetManagementService
from ...auth_rbac.models import db


class FAMAssetModelView(ModelView):
	"""Fixed Asset Management - Asset Master View"""
	
	datamodel = SQLAInterface(CFAMAsset)
	
	list_title = "Fixed Assets"
	show_title = "Asset Details"
	add_title = "Add Asset"
	edit_title = "Edit Asset"
	
	list_columns = [
		'asset_number', 'asset_name', 'category.category_name', 'status',
		'acquisition_cost', 'current_book_value', 'accumulated_depreciation',
		'location', 'custodian', 'acquisition_date'
	]
	
	show_columns = [
		'asset_number', 'asset_tag', 'asset_name', 'description', 'category.category_name',
		'acquisition_cost', 'salvage_value', 'current_book_value', 'accumulated_depreciation',
		'currency_code', 'acquisition_date', 'placed_in_service_date', 'disposal_date',
		'depreciation_method.method_name', 'useful_life_years', 'useful_life_months',
		'location', 'department', 'cost_center', 'custodian',
		'manufacturer', 'model', 'serial_number', 'year_manufactured', 'condition',
		'status', 'is_depreciable', 'is_fully_depreciated', 'is_leased', 'is_insured',
		'last_maintenance_date', 'next_maintenance_date', 'maintenance_cost_ytd',
		'parent_asset.asset_name', 'is_component', 'component_percentage',
		'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset_number', 'asset_tag', 'asset_name', 'description', 'category',
		'acquisition_cost', 'salvage_value', 'currency_code',
		'acquisition_date', 'placed_in_service_date',
		'depreciation_method', 'useful_life_years', 'useful_life_months',
		'location', 'department', 'cost_center', 'custodian',
		'manufacturer', 'model', 'serial_number', 'year_manufactured', 'condition',
		'is_depreciable', 'parent_asset', 'is_component', 'component_percentage',
		'notes'
	]
	
	edit_columns = [
		'asset_tag', 'asset_name', 'description',
		'salvage_value', 'location', 'department', 'cost_center', 'custodian',
		'manufacturer', 'model', 'serial_number', 'year_manufactured', 'condition',
		'status', 'is_depreciable', 'useful_life_years', 'useful_life_months',
		'depreciation_method', 'notes'
	]
	
	search_columns = [
		'asset_number', 'asset_tag', 'asset_name', 'serial_number',
		'location', 'department', 'custodian', 'manufacturer', 'model'
	]
	
	order_columns = [
		'asset_number', 'asset_name', 'acquisition_cost', 'current_book_value',
		'acquisition_date', 'status'
	]
	
	base_order = ('asset_number', 'asc')
	
	formatters_columns = {
		'acquisition_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_book_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'accumulated_depreciation': lambda x: f"${x:,.2f}" if x else "$0.00",
		'salvage_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'maintenance_cost_ytd': lambda x: f"${x:,.2f}" if x else "$0.00",
		'component_percentage': lambda x: f"{x}%" if x else "",
		'status': lambda x: f'<span class="label label-{"success" if x == "Active" else "warning" if x == "Inactive" else "danger"}">{x}</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and generate asset number if needed"""
		item.tenant_id = self.get_tenant_id()
		
		if not item.asset_number:
			fam_service = FixedAssetManagementService(item.tenant_id)
			item.asset_number = fam_service._generate_asset_number()
		
		# Set current book value to acquisition cost initially
		item.current_book_value = item.acquisition_cost
	
	def pre_update(self, item):
		"""Update book value calculation"""
		if item.accumulated_depreciation:
			item.current_book_value = item.acquisition_cost - item.accumulated_depreciation
	
	@action("transfer_asset", "Transfer Asset", "Transfer selected assets", "fa-exchange-alt")
	def transfer_asset(self, assets):
		"""Transfer selected assets"""
		if len(assets) == 1:
			return redirect(url_for('FAMAssetTransferModelView.add', asset_id=assets[0].asset_id))
		flash(f"Please select only one asset to transfer", "warning")
		return redirect(self.get_redirect())
	
	@action("schedule_maintenance", "Schedule Maintenance", "Schedule maintenance for selected assets", "fa-wrench")
	def schedule_maintenance(self, assets):
		"""Schedule maintenance for selected assets"""
		if len(assets) == 1:
			return redirect(url_for('FAMAssetMaintenanceModelView.add', asset_id=assets[0].asset_id))
		flash(f"Please select only one asset for maintenance scheduling", "warning")
		return redirect(self.get_redirect())
	
	@action("dispose_asset", "Dispose Asset", "Dispose selected assets", "fa-trash")
	def dispose_asset(self, assets):
		"""Dispose selected assets"""
		if len(assets) == 1:
			if assets[0].is_disposal_eligible():
				return redirect(url_for('FAMAssetDisposalModelView.add', asset_id=assets[0].asset_id))
			else:
				flash(f"Asset {assets[0].asset_number} is not eligible for disposal", "warning")
		else:
			flash(f"Please select only one asset to dispose", "warning")
		return redirect(self.get_redirect())
	
	@expose('/depreciation_history/<asset_id>')
	def depreciation_history(self, asset_id):
		"""Show depreciation history for asset"""
		asset = CFAMAsset.query.filter_by(
			tenant_id=self.get_tenant_id(),
			asset_id=asset_id
		).first()
		
		if not asset:
			flash("Asset not found", "error")
			return redirect(self.get_redirect())
		
		depreciation_records = CFAMDepreciation.query.filter_by(
			tenant_id=self.get_tenant_id(),
			asset_id=asset_id
		).order_by(CFAMDepreciation.depreciation_date.desc()).all()
		
		return self.render_template(
			'fam/asset_depreciation_history.html',
			asset=asset,
			depreciation_records=depreciation_records,
			title=f"Depreciation History - {asset.asset_name}"
		)


class FAMAssetCategoryModelView(ModelView):
	"""Asset Category Management View"""
	
	datamodel = SQLAInterface(CFAMAssetCategory)
	
	list_title = "Asset Categories"
	show_title = "Category Details"
	add_title = "Add Category"
	edit_title = "Edit Category"
	
	list_columns = [
		'category_code', 'category_name', 'parent_category.category_name',
		'default_useful_life_years', 'default_depreciation_method.method_name',
		'is_active', 'allow_depreciation'
	]
	
	show_columns = [
		'category_code', 'category_name', 'description', 'parent_category.category_name',
		'default_useful_life_years', 'default_useful_life_months',
		'default_depreciation_method.method_name', 'default_salvage_percent',
		'is_active', 'allow_depreciation', 'require_location', 'require_custodian',
		'minimum_cost', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'category_code', 'category_name', 'description', 'parent_category',
		'default_useful_life_years', 'default_useful_life_months',
		'default_depreciation_method', 'default_salvage_percent',
		'is_active', 'allow_depreciation', 'require_location', 'require_custodian',
		'minimum_cost'
	]
	
	edit_columns = [
		'category_name', 'description', 'parent_category',
		'default_useful_life_years', 'default_useful_life_months',
		'default_depreciation_method', 'default_salvage_percent',
		'is_active', 'allow_depreciation', 'require_location', 'require_custodian',
		'minimum_cost'
	]
	
	search_columns = ['category_code', 'category_name', 'description']
	order_columns = ['category_code', 'category_name']
	base_order = ('category_code', 'asc')
	
	formatters_columns = {
		'default_salvage_percent': lambda x: f"{x}%" if x else "0%",
		'minimum_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
	}
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()


class FAMDepreciationMethodModelView(ModelView):
	"""Depreciation Method Management View"""
	
	datamodel = SQLAInterface(CFAMDepreciationMethod)
	
	list_title = "Depreciation Methods"
	show_title = "Method Details"
	add_title = "Add Method"
	edit_title = "Edit Method"
	
	list_columns = [
		'method_code', 'method_name', 'formula', 'depreciation_rate',
		'convention', 'is_active', 'is_system'
	]
	
	show_columns = [
		'method_code', 'method_name', 'description', 'formula',
		'depreciation_rate', 'convention', 'is_active', 'is_system',
		'created_on', 'updated_on'
	]
	
	add_columns = [
		'method_code', 'method_name', 'description', 'formula',
		'depreciation_rate', 'convention', 'is_active'
	]
	
	edit_columns = [
		'method_name', 'description', 'formula',
		'depreciation_rate', 'convention', 'is_active'
	]
	
	search_columns = ['method_code', 'method_name', 'description']
	order_columns = ['method_code', 'method_name']
	base_order = ('method_code', 'asc')
	
	formatters_columns = {
		'depreciation_rate': lambda x: f"{x}%" if x else "",
		'is_system': lambda x: '<span class="label label-info">System</span>' if x else '<span class="label label-default">User</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()
		item.is_system = False


class FAMAssetAcquisitionModelView(ModelView):
	"""Asset Acquisition Management View"""
	
	datamodel = SQLAInterface(CFAMAssetAcquisition)
	
	list_title = "Asset Acquisitions"
	show_title = "Acquisition Details"
	add_title = "Record Acquisition"
	edit_title = "Edit Acquisition"
	
	list_columns = [
		'acquisition_number', 'asset.asset_number', 'asset.asset_name',
		'acquisition_type', 'acquisition_date', 'total_cost',
		'vendor_name', 'approved', 'is_posted'
	]
	
	show_columns = [
		'acquisition_number', 'asset.asset_number', 'asset.asset_name',
		'acquisition_type', 'acquisition_date',
		'gross_cost', 'freight_cost', 'installation_cost', 'other_costs', 'total_cost',
		'vendor_name', 'vendor_id', 'purchase_order_number', 'invoice_number', 'invoice_date',
		'funding_source', 'project_id', 'department', 'cost_center',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'is_posted', 'posted_date', 'journal_entry_id',
		'description', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'acquisition_type', 'acquisition_date',
		'gross_cost', 'freight_cost', 'installation_cost', 'other_costs',
		'vendor_name', 'purchase_order_number', 'invoice_number', 'invoice_date',
		'funding_source', 'project_id', 'department', 'cost_center',
		'description', 'notes'
	]
	
	edit_columns = [
		'acquisition_type', 'acquisition_date',
		'gross_cost', 'freight_cost', 'installation_cost', 'other_costs',
		'vendor_name', 'purchase_order_number', 'invoice_number', 'invoice_date',
		'funding_source', 'department', 'cost_center',
		'description', 'notes'
	]
	
	search_columns = [
		'acquisition_number', 'asset.asset_number', 'asset.asset_name',
		'vendor_name', 'purchase_order_number', 'invoice_number'
	]
	
	order_columns = ['acquisition_number', 'acquisition_date', 'total_cost']
	base_order = ('acquisition_date', 'desc')
	
	formatters_columns = {
		'gross_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'approved': lambda x: '<span class="label label-success">Yes</span>' if x else '<span class="label label-warning">Pending</span>',
		'is_posted': lambda x: '<span class="label label-success">Posted</span>' if x else '<span class="label label-default">Draft</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and calculate total cost"""
		item.tenant_id = self.get_tenant_id()
		if not item.acquisition_number:
			fam_service = FixedAssetManagementService(item.tenant_id)
			item.acquisition_number = fam_service._generate_acquisition_number()
	
	def pre_update(self, item):
		"""Recalculate total cost"""
		item.calculate_total_cost()


class FAMAssetDisposalModelView(ModelView):
	"""Asset Disposal Management View"""
	
	datamodel = SQLAInterface(CFAMAssetDisposal)
	
	list_title = "Asset Disposals"
	show_title = "Disposal Details"
	add_title = "Record Disposal"
	edit_title = "Edit Disposal"
	
	list_columns = [
		'disposal_number', 'asset.asset_number', 'asset.asset_name',
		'disposal_date', 'disposal_method', 'book_value_at_disposal',
		'disposal_proceeds', 'gain_loss_amount', 'approved', 'is_posted'
	]
	
	show_columns = [
		'disposal_number', 'asset.asset_number', 'asset.asset_name',
		'disposal_date', 'disposal_method', 'disposal_reason',
		'book_value_at_disposal', 'accumulated_depreciation_at_disposal',
		'disposal_proceeds', 'disposal_costs', 'net_proceeds',
		'gain_loss_amount', 'is_gain',
		'purchaser_name', 'purchaser_contact', 'sales_agreement_number',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'is_posted', 'posted_date', 'journal_entry_id',
		'description', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'disposal_date', 'disposal_method', 'disposal_reason',
		'disposal_proceeds', 'disposal_costs',
		'purchaser_name', 'purchaser_contact',
		'description', 'notes'
	]
	
	edit_columns = [
		'disposal_date', 'disposal_method', 'disposal_reason',
		'disposal_proceeds', 'disposal_costs',
		'purchaser_name', 'purchaser_contact',
		'description', 'notes'
	]
	
	search_columns = [
		'disposal_number', 'asset.asset_number', 'asset.asset_name',
		'purchaser_name', 'disposal_method'
	]
	
	order_columns = ['disposal_number', 'disposal_date', 'gain_loss_amount']
	base_order = ('disposal_date', 'desc')
	
	formatters_columns = {
		'book_value_at_disposal': lambda x: f"${x:,.2f}" if x else "$0.00",
		'disposal_proceeds': lambda x: f"${x:,.2f}" if x else "$0.00",
		'net_proceeds': lambda x: f"${x:,.2f}" if x else "$0.00",
		'gain_loss_amount': lambda x: f'<span class="{"text-success" if x and x > 0 else "text-danger" if x and x < 0 else ""}">${x:,.2f}</span>' if x else "$0.00",
		'approved': lambda x: '<span class="label label-success">Yes</span>' if x else '<span class="label label-warning">Pending</span>',
		'is_posted': lambda x: '<span class="label label-success">Posted</span>' if x else '<span class="label label-default">Draft</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and calculate gain/loss"""
		item.tenant_id = self.get_tenant_id()
		if not item.disposal_number:
			fam_service = FixedAssetManagementService(item.tenant_id)
			item.disposal_number = fam_service._generate_disposal_number()
		
		if item.asset:
			item.book_value_at_disposal = item.asset.current_book_value
			item.accumulated_depreciation_at_disposal = item.asset.accumulated_depreciation
	
	def pre_update(self, item):
		"""Recalculate gain/loss"""
		item.calculate_gain_loss()


class FAMAssetTransferModelView(ModelView):
	"""Asset Transfer Management View"""
	
	datamodel = SQLAInterface(CFAMAssetTransfer)
	
	list_title = "Asset Transfers"
	show_title = "Transfer Details"
	add_title = "Transfer Asset"
	edit_title = "Edit Transfer"
	
	list_columns = [
		'transfer_number', 'asset.asset_number', 'asset.asset_name',
		'transfer_date', 'transfer_type', 'from_location', 'to_location',
		'from_custodian', 'to_custodian', 'status'
	]
	
	show_columns = [
		'transfer_number', 'asset.asset_number', 'asset.asset_name',
		'transfer_date', 'transfer_type', 'reason',
		'from_location', 'from_department', 'from_cost_center', 'from_custodian',
		'to_location', 'to_department', 'to_cost_center', 'to_custodian',
		'effective_date', 'transfer_cost', 'status', 'completed_date',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'description', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'transfer_date', 'transfer_type', 'reason',
		'to_location', 'to_department', 'to_cost_center', 'to_custodian',
		'effective_date', 'transfer_cost',
		'description', 'notes'
	]
	
	edit_columns = [
		'transfer_date', 'transfer_type', 'reason',
		'to_location', 'to_department', 'to_cost_center', 'to_custodian',
		'effective_date', 'transfer_cost', 'status',
		'description', 'notes'
	]
	
	search_columns = [
		'transfer_number', 'asset.asset_number', 'asset.asset_name',
		'from_location', 'to_location', 'from_custodian', 'to_custodian'
	]
	
	order_columns = ['transfer_number', 'transfer_date']
	base_order = ('transfer_date', 'desc')
	
	formatters_columns = {
		'transfer_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{"success" if x == "Completed" else "warning" if x == "Pending" else "default"}">{x}</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and from fields"""
		item.tenant_id = self.get_tenant_id()
		if not item.transfer_number:
			fam_service = FixedAssetManagementService(item.tenant_id)
			item.transfer_number = fam_service._generate_transfer_number()
		
		if item.asset:
			item.from_location = item.asset.location
			item.from_department = item.asset.department
			item.from_cost_center = item.asset.cost_center
			item.from_custodian = item.asset.custodian


class FAMAssetMaintenanceModelView(ModelView):
	"""Asset Maintenance Management View"""
	
	datamodel = SQLAInterface(CFAMAssetMaintenance)
	
	list_title = "Asset Maintenance"
	show_title = "Maintenance Details"
	add_title = "Schedule Maintenance"
	edit_title = "Edit Maintenance"
	
	list_columns = [
		'maintenance_number', 'asset.asset_number', 'asset.asset_name',
		'maintenance_type', 'scheduled_date', 'status', 'priority',
		'total_cost', 'service_provider'
	]
	
	show_columns = [
		'maintenance_number', 'asset.asset_number', 'asset.asset_name',
		'maintenance_type', 'maintenance_date', 'scheduled_date', 'completed_date',
		'service_provider', 'technician_name', 'work_order_number',
		'description', 'work_performed', 'parts_replaced', 'findings', 'recommendations',
		'labor_hours', 'labor_rate', 'labor_cost', 'parts_cost', 'other_costs', 'total_cost',
		'status', 'priority', 'urgency', 'recurring', 'maintenance_interval_days',
		'next_maintenance_date', 'quality_check_passed', 'safety_check_passed',
		'downtime_hours', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'maintenance_type', 'scheduled_date', 'description',
		'service_provider', 'technician_name', 'priority', 'urgency',
		'recurring', 'maintenance_interval_days', 'notes'
	]
	
	edit_columns = [
		'maintenance_type', 'scheduled_date', 'completed_date',
		'service_provider', 'technician_name', 'description', 'work_performed',
		'parts_replaced', 'findings', 'recommendations',
		'labor_hours', 'labor_rate', 'parts_cost', 'other_costs',
		'status', 'priority', 'urgency', 'recurring', 'maintenance_interval_days',
		'quality_check_passed', 'safety_check_passed', 'downtime_hours', 'notes'
	]
	
	search_columns = [
		'maintenance_number', 'asset.asset_number', 'asset.asset_name',
		'maintenance_type', 'service_provider', 'technician_name'
	]
	
	order_columns = ['maintenance_number', 'scheduled_date', 'total_cost']
	base_order = ('scheduled_date', 'desc')
	
	formatters_columns = {
		'total_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'labor_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'parts_cost': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{"success" if x == "Completed" else "info" if x == "In Progress" else "warning" if x == "Scheduled" else "default"}">{x}</span>',
		'priority': lambda x: f'<span class="label label-{"danger" if x == "Emergency" else "warning" if x == "High" else "default"}">{x}</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and maintenance number"""
		item.tenant_id = self.get_tenant_id()
		if not item.maintenance_number:
			fam_service = FixedAssetManagementService(item.tenant_id)
			item.maintenance_number = fam_service._generate_maintenance_number()
		item.maintenance_date = item.scheduled_date
	
	def pre_update(self, item):
		"""Calculate total cost and update asset maintenance date"""
		item.calculate_total_cost()
		
		if item.status == 'Completed' and item.completed_date:
			if item.asset:
				item.asset.last_maintenance_date = item.completed_date
				if item.recurring and item.maintenance_interval_days:
					item.asset.next_maintenance_date = (
						item.completed_date + timedelta(days=item.maintenance_interval_days)
					)


class FAMAssetInsuranceModelView(ModelView):
	"""Asset Insurance Management View"""
	
	datamodel = SQLAInterface(CFAMAssetInsurance)
	
	list_title = "Asset Insurance"
	show_title = "Insurance Details"
	add_title = "Add Insurance Policy"
	edit_title = "Edit Insurance"
	
	list_columns = [
		'policy_number', 'asset.asset_number', 'asset.asset_name',
		'insurance_company', 'policy_type', 'coverage_amount',
		'policy_start_date', 'policy_end_date', 'is_active'
	]
	
	show_columns = [
		'policy_number', 'asset.asset_number', 'asset.asset_name',
		'insurance_company', 'policy_type', 'coverage_type',
		'coverage_amount', 'deductible_amount', 'premium_amount',
		'policy_start_date', 'policy_end_date', 'auto_renew',
		'agent_name', 'agent_phone', 'agent_email',
		'claims_count', 'last_claim_date', 'total_claims_amount',
		'is_active', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'policy_number', 'insurance_company', 'policy_type', 'coverage_type',
		'coverage_amount', 'deductible_amount', 'premium_amount',
		'policy_start_date', 'policy_end_date', 'auto_renew',
		'agent_name', 'agent_phone', 'agent_email', 'notes'
	]
	
	edit_columns = [
		'insurance_company', 'policy_type', 'coverage_type',
		'coverage_amount', 'deductible_amount', 'premium_amount',
		'policy_start_date', 'policy_end_date', 'auto_renew',
		'agent_name', 'agent_phone', 'agent_email',
		'is_active', 'notes'
	]
	
	search_columns = [
		'policy_number', 'asset.asset_number', 'asset.asset_name',
		'insurance_company', 'agent_name'
	]
	
	order_columns = ['policy_number', 'policy_end_date', 'coverage_amount']
	base_order = ('policy_end_date', 'asc')
	
	formatters_columns = {
		'coverage_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'deductible_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'premium_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_claims_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'policy_end_date': lambda x: f'<span class="{"text-danger" if x and x < date.today() + timedelta(days=30) else ""}">{x}</span>' if x else "",
		'is_active': lambda x: '<span class="label label-success">Active</span>' if x else '<span class="label label-default">Inactive</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id"""
		item.tenant_id = self.get_tenant_id()


class FAMAssetValuationModelView(ModelView):
	"""Asset Valuation Management View"""
	
	datamodel = SQLAInterface(CFAMAssetValuation)
	
	list_title = "Asset Valuations"
	show_title = "Valuation Details"
	add_title = "Record Valuation"
	edit_title = "Edit Valuation"
	
	list_columns = [
		'asset.asset_number', 'asset.asset_name', 'valuation_date',
		'valuation_type', 'appraised_value', 'book_value_at_valuation',
		'revaluation_surplus_deficit', 'approved', 'is_posted'
	]
	
	show_columns = [
		'asset.asset_number', 'asset.asset_name', 'valuation_date',
		'valuation_type', 'valuation_method', 'valuation_purpose',
		'appraised_value', 'book_value_at_valuation', 'revaluation_surplus_deficit', 'impairment_loss',
		'valuation_basis', 'appraiser_name', 'appraiser_firm', 'appraiser_license',
		'effective_date', 'next_valuation_date',
		'requires_approval', 'approved', 'approved_by', 'approved_date',
		'is_posted', 'posted_date', 'journal_entry_id',
		'description', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'asset', 'valuation_date', 'valuation_type', 'valuation_method', 'valuation_purpose',
		'appraised_value', 'valuation_basis',
		'appraiser_name', 'appraiser_firm', 'appraiser_license',
		'effective_date', 'next_valuation_date',
		'description', 'notes'
	]
	
	edit_columns = [
		'valuation_date', 'valuation_type', 'valuation_method', 'valuation_purpose',
		'appraised_value', 'valuation_basis',
		'appraiser_name', 'appraiser_firm', 'appraiser_license',
		'effective_date', 'next_valuation_date',
		'description', 'notes'
	]
	
	search_columns = [
		'asset.asset_number', 'asset.asset_name', 'valuation_type',
		'appraiser_name', 'appraiser_firm'
	]
	
	order_columns = ['valuation_date', 'appraised_value']
	base_order = ('valuation_date', 'desc')
	
	formatters_columns = {
		'appraised_value': lambda x: f"${x:,.2f}" if x else "$0.00",
		'book_value_at_valuation': lambda x: f"${x:,.2f}" if x else "$0.00",
		'revaluation_surplus_deficit': lambda x: f'<span class="{"text-success" if x and x > 0 else "text-danger" if x and x < 0 else ""}">${x:,.2f}</span>' if x else "$0.00",
		'impairment_loss': lambda x: f'<span class="text-danger">${x:,.2f}</span>' if x and x > 0 else "$0.00",
		'approved': lambda x: '<span class="label label-success">Yes</span>' if x else '<span class="label label-warning">Pending</span>',
		'is_posted': lambda x: '<span class="label label-success">Posted</span>' if x else '<span class="label label-default">Draft</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and book value"""
		item.tenant_id = self.get_tenant_id()
		if item.asset:
			item.book_value_at_valuation = item.asset.current_book_value
	
	def pre_update(self, item):
		"""Calculate revaluation impact"""
		item.calculate_revaluation_impact()


class FAMAssetLeaseModelView(ModelView):
	"""Asset Lease Management View"""
	
	datamodel = SQLAInterface(CFAMAssetLease)
	
	list_title = "Asset Leases"
	show_title = "Lease Details"
	add_title = "Add Lease"
	edit_title = "Edit Lease"
	
	list_columns = [
		'lease_number', 'lease_name', 'lease_type', 'lessor_name',
		'lease_start_date', 'lease_end_date', 'base_monthly_payment',
		'current_lease_liability', 'current_rou_asset', 'is_active'
	]
	
	show_columns = [
		'lease_number', 'lease_name', 'lease_type', 'lease_classification',
		'lessor_name', 'lessor_contact', 'lease_start_date', 'lease_end_date', 'lease_term_months',
		'base_monthly_payment', 'escalation_rate', 'payment_frequency',
		'initial_lease_liability', 'initial_rou_asset', 'current_lease_liability', 'current_rou_asset',
		'incremental_borrowing_rate', 'discount_rate_used',
		'purchase_option', 'purchase_option_price', 'guaranteed_residual_value',
		'leased_asset_description', 'leased_asset_location',
		'is_active', 'notes', 'created_on', 'updated_on'
	]
	
	add_columns = [
		'lease_number', 'lease_name', 'lease_type', 'lessor_name', 'lessor_contact',
		'lease_start_date', 'lease_end_date', 'base_monthly_payment', 'escalation_rate',
		'incremental_borrowing_rate', 'purchase_option', 'purchase_option_price',
		'leased_asset_description', 'leased_asset_location', 'notes'
	]
	
	edit_columns = [
		'lease_name', 'lease_type', 'lessor_name', 'lessor_contact',
		'lease_start_date', 'lease_end_date', 'base_monthly_payment', 'escalation_rate',
		'incremental_borrowing_rate', 'purchase_option', 'purchase_option_price',
		'leased_asset_description', 'leased_asset_location', 'is_active', 'notes'
	]
	
	search_columns = [
		'lease_number', 'lease_name', 'lessor_name', 'leased_asset_description'
	]
	
	order_columns = ['lease_number', 'lease_start_date', 'lease_end_date']
	base_order = ('lease_start_date', 'desc')
	
	formatters_columns = {
		'base_monthly_payment': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_lease_liability': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_rou_asset': lambda x: f"${x:,.2f}" if x else "$0.00",
		'escalation_rate': lambda x: f"{x}%" if x else "0%",
		'incremental_borrowing_rate': lambda x: f"{x}%" if x else "",
		'is_active': lambda x: '<span class="label label-success">Active</span>' if x else '<span class="label label-default">Inactive</span>',
	}
	
	def pre_add(self, item):
		"""Set tenant_id and calculate lease term"""
		item.tenant_id = self.get_tenant_id()
		if item.lease_start_date and item.lease_end_date:
			item.calculate_lease_term_months()
	
	def pre_update(self, item):
		"""Recalculate lease term if dates changed"""
		if item.lease_start_date and item.lease_end_date:
			item.calculate_lease_term_months()


class FAMDepreciationReportView(BaseView):
	"""Depreciation Reporting View"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main depreciation report page"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		
		# Get summary data
		summary = fam_service.get_asset_summary()
		
		# Get upcoming depreciation
		upcoming_depreciation = fam_service.get_depreciation_schedule(months_ahead=12)
		
		return self.render_template(
			'fam/depreciation_report.html',
			summary=summary,
			upcoming_depreciation=upcoming_depreciation,
			title="Depreciation Report"
		)
	
	@expose('/api/depreciation_schedule')
	@has_access
	def api_depreciation_schedule(self):
		"""API endpoint for depreciation schedule data"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		months_ahead = int(request.args.get('months', 12))
		asset_id = request.args.get('asset_id')
		
		schedule = fam_service.get_depreciation_schedule(asset_id, months_ahead)
		return jsonify(schedule)
	
	@expose('/calculate_monthly')
	@has_access
	def calculate_monthly(self):
		"""Calculate monthly depreciation"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		
		try:
			as_of_date = request.args.get('as_of_date')
			if as_of_date:
				as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
			else:
				as_of_date = date.today()
			
			depreciation_records = fam_service.calculate_monthly_depreciation(as_of_date)
			
			flash(f"Calculated depreciation for {len(depreciation_records)} assets", "success")
			
			# Optionally post to GL
			if request.args.get('post_to_gl') == 'true':
				journal_ids = fam_service.post_depreciation_to_gl(depreciation_records)
				flash(f"Posted {len(journal_ids)} journal entries to GL", "info")
			
		except Exception as e:
			flash(f"Error calculating depreciation: {str(e)}", "error")
		
		return redirect(url_for('FAMDepreciationReportView.index'))


class FAMDashboardView(BaseView):
	"""Fixed Asset Management Dashboard"""
	
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Main dashboard page"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		
		# Get summary statistics
		summary = fam_service.get_asset_summary()
		
		# Get upcoming maintenance
		upcoming_maintenance = fam_service.get_maintenance_schedule(days_ahead=30)
		
		# Get insurance renewals due
		insurance_renewals = fam_service.get_insurance_renewals_due(days_ahead=60)
		
		# Get assets by location
		assets_by_location = fam_service.get_assets_by_location()
		
		return self.render_template(
			'fam/dashboard.html',
			summary=summary,
			upcoming_maintenance=upcoming_maintenance,
			insurance_renewals=insurance_renewals,
			assets_by_location=assets_by_location,
			title="Fixed Asset Management Dashboard"
		)
	
	@expose('/api/summary')
	@has_access
	def api_summary(self):
		"""API endpoint for dashboard summary data"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		summary = fam_service.get_asset_summary()
		return jsonify(summary)
	
	@expose('/api/maintenance_alerts')
	@has_access
	def api_maintenance_alerts(self):
		"""API endpoint for maintenance alerts"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		days_ahead = int(request.args.get('days', 30))
		
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
		
		return jsonify(alerts)
	
	@expose('/api/insurance_alerts')
	@has_access
	def api_insurance_alerts(self):
		"""API endpoint for insurance renewal alerts"""
		fam_service = FixedAssetManagementService(self.get_tenant_id())
		days_ahead = int(request.args.get('days', 60))
		
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
		
		return jsonify(alerts)
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID - placeholder implementation"""
		# TODO: Implement proper tenant context
		return "default_tenant"