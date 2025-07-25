"""
Expiry Date Management Views

Flask-AppBuilder views for expiry tracking, FEFO management,
waste reporting, and compliance monitoring.
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from datetime import datetime, timedelta

from .models import (
	IMEDMExpiryPolicy, IMEDMExpiryItem, IMEDMExpiryMovement,
	IMEDMDisposition, IMEDMExpiryAlert, IMEDMWasteReport
)
from .service import ExpiryDateService


class IMEDMExpiryItemView(ModelView):
	"""View for managing expiry-tracked items"""
	
	datamodel = SQLAInterface(IMEDMExpiryItem)
	
	list_columns = ['item_id', 'batch_number', 'expiry_date', 'current_quantity', 
					'expiry_status', 'alert_level', 'days_to_expiry']
	show_columns = ['item_id', 'batch_number', 'lot_number', 'serial_number',
					'warehouse_id', 'location_id', 'expiry_date', 'manufactured_date',
					'best_before_date', 'shelf_life_days', 'original_quantity',
					'current_quantity', 'allocated_quantity', 'available_quantity',
					'expiry_status', 'quality_status', 'disposition_status',
					'alert_level', 'shelf_life_extended', 'extended_expiry_date']
	edit_columns = ['item_id', 'batch_number', 'lot_number', 'warehouse_id', 'location_id',
					'expiry_date', 'manufactured_date', 'best_before_date', 'shelf_life_days',
					'expiry_status', 'quality_status', 'disposition_status', 'notes']
	add_columns = edit_columns + ['original_quantity', 'current_quantity', 'available_quantity']
	
	base_order = ('expiry_date', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'item_id': 'Item ID',
		'batch_number': 'Batch Number',
		'lot_number': 'Lot Number',
		'serial_number': 'Serial Number',
		'warehouse_id': 'Warehouse ID',
		'location_id': 'Location ID',
		'expiry_date': 'Expiry Date',
		'manufactured_date': 'Manufactured Date',
		'best_before_date': 'Best Before Date',
		'shelf_life_days': 'Shelf Life (Days)',
		'original_quantity': 'Original Quantity',
		'current_quantity': 'Current Quantity',
		'allocated_quantity': 'Allocated Quantity',
		'available_quantity': 'Available Quantity',
		'expiry_status': 'Expiry Status',
		'quality_status': 'Quality Status',
		'disposition_status': 'Disposition Status',
		'alert_level': 'Alert Level',
		'days_to_expiry': 'Days to Expiry',
		'shelf_life_extended': 'Shelf Life Extended',
		'extended_expiry_date': 'Extended Expiry Date'
	}
	
	@action("extend_shelf_life", "Extend Shelf Life", "Extend shelf life for selected items", "fa-calendar-plus", multiple=True)
	def extend_shelf_life_action(self, items):
		"""Extend shelf life for selected items"""
		count = 0
		for item in items:
			if not item.shelf_life_extended and item.expiry_status != 'Disposed':
				# Extend by 30 days (simplified)
				item.extended_expiry_date = item.expiry_date + timedelta(days=30)
				item.shelf_life_extended = True
				item.extension_reason = 'Management approval'
				item.extension_approved_by = self.get_current_user_id()
				item.extension_approval_date = datetime.utcnow()
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} items shelf life extended', 'success')
		else:
			flash('No items eligible for shelf life extension', 'warning')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"
	
	def get_current_user_id(self):
		return "current_user"


class IMEDMDispositionView(ModelView):
	"""View for managing dispositions"""
	
	datamodel = SQLAInterface(IMEDMDisposition)
	
	list_columns = ['disposition_number', 'disposition_date', 'expiry_item.batch_number',
					'disposition_type', 'quantity_disposed', 'status']
	show_columns = ['disposition_number', 'disposition_date', 'expiry_item', 'disposition_type',
					'disposition_reason', 'quantity_disposed', 'status', 'approved_by',
					'approval_date', 'executed_by', 'execution_date', 'disposal_cost',
					'recovery_value', 'net_loss', 'disposal_vendor']
	edit_columns = ['disposition_number', 'expiry_item_id', 'disposition_type', 'disposition_reason',
					'quantity_disposed', 'status', 'disposal_cost', 'recovery_value',
					'disposal_vendor', 'notes']
	add_columns = edit_columns
	
	base_order = ('disposition_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'disposition_number': 'Disposition Number',
		'disposition_date': 'Disposition Date',
		'expiry_item': 'Expiry Item',
		'expiry_item_id': 'Expiry Item ID',
		'disposition_type': 'Disposition Type',
		'disposition_reason': 'Disposition Reason',
		'quantity_disposed': 'Quantity Disposed',
		'approved_by': 'Approved By',
		'approval_date': 'Approval Date',
		'executed_by': 'Executed By',
		'execution_date': 'Execution Date',
		'disposal_cost': 'Disposal Cost',
		'recovery_value': 'Recovery Value',
		'net_loss': 'Net Loss',
		'disposal_vendor': 'Disposal Vendor'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMEDMExpiryAlertView(ModelView):
	"""View for managing expiry alerts"""
	
	datamodel = SQLAInterface(IMEDMExpiryAlert)
	
	list_columns = ['alert_date', 'alert_type', 'alert_level', 'alert_title',
					'expiry_date', 'days_to_expiry', 'status']
	show_columns = ['alert_date', 'alert_type', 'alert_level', 'alert_title', 'alert_message',
					'item_id', 'expiry_item', 'warehouse_id', 'expiry_date', 'days_to_expiry',
					'quantity_affected', 'value_at_risk', 'status', 'acknowledged_by',
					'acknowledgment_date', 'escalation_level']
	edit_columns = ['status', 'acknowledged_by', 'acknowledgment_date', 'resolution_date',
					'resolution_action', 'notes']
	
	# Alerts are typically system-generated
	add_columns = []
	can_create = False
	can_delete = False
	
	base_order = ('alert_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'alert_date': 'Alert Date',
		'alert_type': 'Alert Type',
		'alert_level': 'Alert Level',
		'alert_title': 'Alert Title',
		'alert_message': 'Alert Message',
		'item_id': 'Item ID',
		'expiry_item': 'Expiry Item',
		'warehouse_id': 'Warehouse ID',
		'expiry_date': 'Expiry Date',
		'days_to_expiry': 'Days to Expiry',
		'quantity_affected': 'Quantity Affected',
		'value_at_risk': 'Value at Risk',
		'acknowledged_by': 'Acknowledged By',
		'acknowledgment_date': 'Acknowledgment Date',
		'resolution_date': 'Resolution Date',
		'resolution_action': 'Resolution Action',
		'escalation_level': 'Escalation Level'
	}
	
	@action("acknowledge", "Acknowledge", "Acknowledge selected alerts", "fa-check", multiple=True)
	def acknowledge_action(self, alerts):
		"""Acknowledge selected alerts"""
		count = 0
		for alert in alerts:
			if alert.status == 'Active':
				alert.status = 'Acknowledged'
				alert.acknowledged_by = self.get_current_user_id()
				alert.acknowledgment_date = datetime.utcnow()
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} alerts acknowledged', 'success')
		else:
			flash('No active alerts to acknowledge', 'warning')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"
	
	def get_current_user_id(self):
		return "current_user"


class ExpiryDateDashboardView(BaseView):
	"""Expiry Date Management Dashboard"""
	
	route_base = "/expiry/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display Expiry Date Management dashboard"""
		
		try:
			with ExpiryDateService(self.get_tenant_id()) as service:
				dashboard_data = {
					'expiring_soon': service.get_items_expiring_soon_count(30),
					'expired_items': service.get_expired_items_count(),
					'waste_value_month': float(service.get_waste_value_current_month()),
					'expiring_items_7_days': service.get_items_expiring_soon_count(7),
					'recent_dispositions': self._get_recent_dispositions()
				}
				
				return self.render_template(
					'expiry_date_dashboard.html',
					dashboard_data=dashboard_data,
					title="Expiry Date Management Dashboard"
				)
		
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	@expose('/fefo_report')
	@has_access
	def fefo_report(self):
		"""Display FEFO compliance report"""
		
		try:
			item_id = request.args.get('item_id')
			warehouse_id = request.args.get('warehouse_id')
			
			with ExpiryDateService(self.get_tenant_id()) as service:
				if item_id:
					fefo_sequence = service.get_fefo_sequence(item_id, warehouse_id)
				else:
					fefo_sequence = []
				
				return self.render_template(
					'fefo_report.html',
					fefo_sequence=fefo_sequence,
					item_id=item_id,
					warehouse_id=warehouse_id,
					title="FEFO Compliance Report"
				)
		
		except Exception as e:
			flash(f'Error loading FEFO report: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	def _get_recent_dispositions(self) -> list:
		"""Get recent dispositions"""
		return []  # Simplified for now
	
	def get_tenant_id(self):
		return "default_tenant"