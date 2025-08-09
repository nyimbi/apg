"""
Stock Tracking & Control Views

Flask-AppBuilder views for inventory item management, stock levels,
movements tracking, and real-time inventory monitoring.
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from flask_appbuilder.charts.views import GroupByChartView
from flask_appbuilder.widgets import ListWidget, ShowWidget
from wtforms import SelectField, DecimalField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, NumberRange
from datetime import datetime, timedelta
from typing import Dict, List, Any

from .models import (
	IMSTCItem, IMSTCItemCategory, IMSTCUnitOfMeasure,
	IMSTCWarehouse, IMSTCLocation, IMSTCStockLevel,
	IMSTCStockMovement, IMSTCCycleCount, IMSTCCycleCountLine,
	IMSTCStockAlert
)
from .service import StockTrackingService


class IMSTCItemCategoryView(ModelView):
	"""View for managing item categories"""
	
	datamodel = SQLAInterface(IMSTCItemCategory)
	
	list_columns = ['category_code', 'category_name', 'parent_category.category_name', 'level', 'is_active']
	show_columns = ['category_code', 'category_name', 'description', 'parent_category', 'level', 'path',
					'is_active', 'sort_order', 'requires_serial_tracking', 'requires_lot_tracking',
					'requires_expiry_tracking', 'abc_classification']
	edit_columns = ['category_code', 'category_name', 'description', 'parent_category_id',
					'is_active', 'sort_order', 'requires_serial_tracking', 'requires_lot_tracking',
					'requires_expiry_tracking', 'abc_classification']
	add_columns = edit_columns
	
	base_order = ('category_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'category_code': 'Category Code',
		'category_name': 'Category Name',
		'parent_category': 'Parent Category',
		'requires_serial_tracking': 'Serial Tracking',
		'requires_lot_tracking': 'Lot Tracking',
		'requires_expiry_tracking': 'Expiry Tracking',
		'abc_classification': 'ABC Class'
	}
	
	def get_tenant_id(self):
		"""Get current tenant ID"""
		return "default_tenant"  # TODO: Implement tenant resolution


class IMSTCUnitOfMeasureView(ModelView):
	"""View for managing units of measure"""
	
	datamodel = SQLAInterface(IMSTCUnitOfMeasure)
	
	list_columns = ['uom_code', 'uom_name', 'uom_type', 'is_base_unit', 'conversion_factor', 'is_active']
	show_columns = ['uom_code', 'uom_name', 'description', 'uom_type', 'base_unit', 'conversion_factor',
					'is_active', 'is_base_unit', 'decimal_places']
	edit_columns = ['uom_code', 'uom_name', 'description', 'uom_type', 'base_unit_id', 'conversion_factor',
					'is_active', 'is_base_unit', 'decimal_places']
	add_columns = edit_columns
	
	base_order = ('uom_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'uom_code': 'UOM Code',
		'uom_name': 'UOM Name',
		'uom_type': 'UOM Type',
		'base_unit': 'Base Unit',
		'conversion_factor': 'Conversion Factor',
		'is_base_unit': 'Is Base Unit',
		'decimal_places': 'Decimal Places'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCWarehouseView(ModelView):
	"""View for managing warehouses"""
	
	datamodel = SQLAInterface(IMSTCWarehouse)
	
	list_columns = ['warehouse_code', 'warehouse_name', 'city', 'state_province', 'is_active', 'is_primary']
	show_columns = ['warehouse_code', 'warehouse_name', 'description', 'address_line1', 'address_line2',
					'city', 'state_province', 'postal_code', 'country_code', 'latitude', 'longitude',
					'is_active', 'is_primary', 'warehouse_type', 'allows_negative_stock', 'auto_allocate_stock',
					'temperature_controlled', 'min_temperature', 'max_temperature', 'manager_name',
					'phone_number', 'email_address']
	edit_columns = ['warehouse_code', 'warehouse_name', 'description', 'address_line1', 'address_line2',
					'city', 'state_province', 'postal_code', 'country_code', 'latitude', 'longitude',
					'is_active', 'is_primary', 'warehouse_type', 'allows_negative_stock', 'auto_allocate_stock',
					'temperature_controlled', 'min_temperature', 'max_temperature', 'manager_name',
					'phone_number', 'email_address']
	add_columns = edit_columns
	
	base_order = ('warehouse_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'warehouse_code': 'Warehouse Code',
		'warehouse_name': 'Warehouse Name',
		'address_line1': 'Address Line 1',
		'address_line2': 'Address Line 2',
		'state_province': 'State/Province',
		'postal_code': 'Postal Code',
		'country_code': 'Country',
		'is_primary': 'Primary Warehouse',
		'warehouse_type': 'Warehouse Type',
		'allows_negative_stock': 'Allow Negative Stock',
		'auto_allocate_stock': 'Auto Allocate Stock',
		'temperature_controlled': 'Temperature Controlled',
		'min_temperature': 'Min Temperature (°C)',
		'max_temperature': 'Max Temperature (°C)',
		'manager_name': 'Manager Name',
		'phone_number': 'Phone Number',
		'email_address': 'Email Address'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCLocationView(ModelView):
	"""View for managing storage locations"""
	
	datamodel = SQLAInterface(IMSTCLocation)
	
	list_columns = ['location_code', 'location_name', 'warehouse.warehouse_name', 'location_type', 'level', 'is_active']
	show_columns = ['location_code', 'location_name', 'description', 'warehouse', 'parent_location',
					'level', 'path', 'location_type', 'capacity_volume', 'capacity_weight',
					'length', 'width', 'height', is_active', 'is_pickable', 'is_receivable',
					'is_quarantine', 'is_damaged_goods', 'temperature_controlled', 'min_temperature',
					'max_temperature', 'humidity_controlled', 'max_humidity', 'pick_sequence',
					'put_sequence', 'abc_zone']
	edit_columns = ['location_code', 'location_name', 'description', 'warehouse_id', 'parent_location_id',
					'location_type', 'capacity_volume', 'capacity_weight', 'length', 'width', 'height',
					'is_active', 'is_pickable', 'is_receivable', 'is_quarantine', 'is_damaged_goods',
					'temperature_controlled', 'min_temperature', 'max_temperature', 'humidity_controlled',
					'max_humidity', 'pick_sequence', 'put_sequence', 'abc_zone']
	add_columns = edit_columns
	
	base_order = ('warehouse_id', 'asc', 'location_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'location_code': 'Location Code',
		'location_name': 'Location Name',
		'warehouse': 'Warehouse',
		'parent_location': 'Parent Location',
		'location_type': 'Location Type',
		'capacity_volume': 'Volume Capacity',
		'capacity_weight': 'Weight Capacity',
		'is_pickable': 'Can Pick From',
		'is_receivable': 'Can Receive To',
		'is_quarantine': 'Quarantine Location',
		'is_damaged_goods': 'Damaged Goods Area',
		'temperature_controlled': 'Temperature Controlled',
		'min_temperature': 'Min Temperature (°C)',
		'max_temperature': 'Max Temperature (°C)',
		'humidity_controlled': 'Humidity Controlled',
		'max_humidity': 'Max Humidity (%)',
		'pick_sequence': 'Pick Sequence',
		'put_sequence': 'Put Sequence',
		'abc_zone': 'ABC Zone'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCItemView(ModelView):
	"""View for managing inventory items"""
	
	datamodel = SQLAInterface(IMSTCItem)
	
	list_columns = ['item_code', 'item_name', 'category.category_name', 'item_type', 'abc_classification', 'is_active']
	show_columns = ['item_code', 'item_name', 'description', 'short_description', 'category', 'item_type',
					'abc_classification', 'primary_uom', 'weight', 'weight_uom', 'volume', 'volume_uom',
					'length', 'width', 'height', 'dimension_uom', 'requires_serial_tracking',
					'requires_lot_tracking', 'requires_expiry_tracking', 'shelf_life_days',
					'default_warehouse', 'default_location', 'min_stock_level', 'max_stock_level',
					'reorder_point', 'reorder_quantity', 'safety_stock', 'standard_cost', 'last_cost',
					'average_cost', 'cost_method', 'is_active', 'is_sellable', 'is_purchasable',
					'is_serialized', 'is_lot_controlled', 'is_perishable', 'is_hazardous',
					'requires_inspection', 'inspection_type', 'acceptable_quality_level',
					'regulatory_class', 'controlled_substance', 'requires_certification',
					'certification_type', 'storage_temperature_min', 'storage_temperature_max',
					'storage_humidity_max', 'special_handling_instructions']
	edit_columns = ['item_code', 'item_name', 'description', 'short_description', 'category_id', 'item_type',
					'abc_classification', 'primary_uom_id', 'weight', 'weight_uom', 'volume', 'volume_uom',
					'length', 'width', 'height', 'dimension_uom', 'requires_serial_tracking',
					'requires_lot_tracking', 'requires_expiry_tracking', 'shelf_life_days',
					'default_warehouse_id', 'default_location_id', 'min_stock_level', 'max_stock_level',
					'reorder_point', 'reorder_quantity', 'safety_stock', 'standard_cost', 'last_cost',
					'average_cost', 'cost_method', 'is_active', 'is_sellable', 'is_purchasable',
					'is_serialized', 'is_lot_controlled', 'is_perishable', 'is_hazardous',
					'requires_inspection', 'inspection_type', 'acceptable_quality_level',
					'regulatory_class', 'controlled_substance', 'requires_certification',
					'certification_type', 'storage_temperature_min', 'storage_temperature_max',
					'storage_humidity_max', 'special_handling_instructions']
	add_columns = edit_columns
	
	base_order = ('item_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'item_code': 'Item Code',
		'item_name': 'Item Name',
		'short_description': 'Short Description',
		'category': 'Category',
		'item_type': 'Item Type',
		'abc_classification': 'ABC Class',
		'primary_uom': 'Primary UOM',
		'weight_uom': 'Weight UOM',
		'volume_uom': 'Volume UOM',
		'dimension_uom': 'Dimension UOM',
		'requires_serial_tracking': 'Serial Tracking',
		'requires_lot_tracking': 'Lot Tracking',
		'requires_expiry_tracking': 'Expiry Tracking',
		'shelf_life_days': 'Shelf Life (Days)',
		'default_warehouse': 'Default Warehouse',
		'default_location': 'Default Location',
		'min_stock_level': 'Min Stock Level',
		'max_stock_level': 'Max Stock Level',
		'reorder_point': 'Reorder Point',
		'reorder_quantity': 'Reorder Quantity',
		'safety_stock': 'Safety Stock',
		'standard_cost': 'Standard Cost',
		'last_cost': 'Last Cost',
		'average_cost': 'Average Cost',
		'cost_method': 'Cost Method',
		'is_sellable': 'Sellable',
		'is_purchasable': 'Purchasable',
		'is_serialized': 'Serialized',
		'is_lot_controlled': 'Lot Controlled',
		'is_perishable': 'Perishable',
		'is_hazardous': 'Hazardous',
		'requires_inspection': 'Requires Inspection',
		'inspection_type': 'Inspection Type',
		'acceptable_quality_level': 'AQL (%)',
		'regulatory_class': 'Regulatory Class',
		'controlled_substance': 'Controlled Substance',
		'requires_certification': 'Requires Certification',
		'certification_type': 'Certification Type',
		'storage_temperature_min': 'Min Storage Temp (°C)',
		'storage_temperature_max': 'Max Storage Temp (°C)',
		'storage_humidity_max': 'Max Storage Humidity (%)',
		'special_handling_instructions': 'Special Handling'
	}
	
	@action("stock_inquiry", "Stock Inquiry", "", "fa-search", multiple=False)
	def stock_inquiry_action(self, items):
		"""View current stock levels for selected item"""
		if len(items) != 1:
			flash('Please select exactly one item', 'warning')
			return redirect(self.get_redirect())
		
		item = items[0]
		return redirect(url_for('IMSTCStockLevelView.list', 
								_flt_0_item_id=item.item_id))
	
	@action("movement_history", "Movement History", "", "fa-history", multiple=False)
	def movement_history_action(self, items):
		"""View movement history for selected item"""
		if len(items) != 1:
			flash('Please select exactly one item', 'warning')
			return redirect(self.get_redirect())
		
		item = items[0]
		return redirect(url_for('IMSTCStockMovementView.list', 
								_flt_0_item_id=item.item_id))
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCStockLevelView(ModelView):
	"""View for current stock levels"""
	
	datamodel = SQLAInterface(IMSTCStockLevel)
	
	list_columns = ['item.item_code', 'item.item_name', 'warehouse.warehouse_name', 'location.location_code',
					'lot_number', 'serial_number', 'on_hand_quantity', 'available_quantity', 'stock_status']
	show_columns = ['item', 'warehouse', 'location', 'uom', 'lot_number', 'serial_number', 'batch_number',
					'expiry_date', 'stock_status', 'quality_status', 'on_hand_quantity', 'allocated_quantity',
					'available_quantity', 'on_order_quantity', 'reserved_quantity', 'unit_cost', 'total_cost',
					'first_received_date', 'last_received_date', 'last_issued_date', 'last_counted_date',
					'container_number', 'pallet_number']
	
	# Stock levels are typically not directly editable - use movements instead
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	base_order = ('item.item_code', 'asc', 'warehouse.warehouse_code', 'asc', 'location.location_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id'], ['on_hand_quantity', '>', 0]]
	
	label_columns = {
		'item': 'Item',
		'warehouse': 'Warehouse',
		'location': 'Location',
		'uom': 'UOM',
		'lot_number': 'Lot Number',
		'serial_number': 'Serial Number',
		'batch_number': 'Batch Number',
		'expiry_date': 'Expiry Date',
		'stock_status': 'Stock Status',
		'quality_status': 'Quality Status',
		'on_hand_quantity': 'On Hand',
		'allocated_quantity': 'Allocated',
		'available_quantity': 'Available',
		'on_order_quantity': 'On Order',
		'reserved_quantity': 'Reserved',
		'unit_cost': 'Unit Cost',
		'total_cost': 'Total Cost',
		'first_received_date': 'First Received',
		'last_received_date': 'Last Received',
		'last_issued_date': 'Last Issued',
		'last_counted_date': 'Last Counted',
		'container_number': 'Container',
		'pallet_number': 'Pallet'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCStockMovementView(ModelView):
	"""View for stock movement history"""
	
	datamodel = SQLAInterface(IMSTCStockMovement)
	
	list_columns = ['movement_date', 'movement_type', 'item.item_code', 'warehouse.warehouse_name',
					'location.location_code', 'quantity', 'reference_number', 'status']
	show_columns = ['movement_type', 'movement_subtype', 'movement_date', 'item', 'warehouse', 'location',
					'uom', 'lot_number', 'serial_number', 'batch_number', 'expiry_date', 'quantity',
					'unit_cost', 'total_cost', 'reference_number', 'reference_type', 'from_warehouse',
					'from_location', 'to_warehouse', 'to_location', 'stock_status', 'quality_status',
					'running_balance', 'running_value', 'transaction_source', 'user_id', 'reason_code',
					'notes', 'status', 'posted_date', 'posted_by']
	
	# Movements are historical records - no editing after posting
	edit_columns = []
	add_columns = []
	can_create = False
	can_edit = False
	can_delete = False
	
	base_order = ('movement_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'movement_type': 'Movement Type',
		'movement_subtype': 'Subtype',
		'movement_date': 'Movement Date',
		'item': 'Item',
		'warehouse': 'Warehouse',
		'location': 'Location',
		'uom': 'UOM',
		'lot_number': 'Lot Number',
		'serial_number': 'Serial Number',
		'batch_number': 'Batch Number',
		'expiry_date': 'Expiry Date',
		'unit_cost': 'Unit Cost',
		'total_cost': 'Total Cost',
		'reference_number': 'Reference Number',
		'reference_type': 'Reference Type',
		'from_warehouse': 'From Warehouse',
		'from_location': 'From Location',
		'to_warehouse': 'To Warehouse',
		'to_location': 'To Location',
		'stock_status': 'Stock Status',
		'quality_status': 'Quality Status',
		'running_balance': 'Running Balance',
		'running_value': 'Running Value',
		'transaction_source': 'Transaction Source',
		'user_id': 'User ID',
		'reason_code': 'Reason Code',
		'posted_date': 'Posted Date',
		'posted_by': 'Posted By'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMSTCStockAlertView(ModelView):
	"""View for stock alerts and notifications"""
	
	datamodel = SQLAInterface(IMSTCStockAlert)
	
	list_columns = ['alert_type', 'alert_priority', 'alert_title', 'item.item_code', 'status', 'created_date']
	show_columns = ['alert_type', 'alert_priority', 'alert_title', 'alert_message', 'item', 'warehouse',
					'location', 'category', 'trigger_condition', 'threshold_value', 'current_value',
					'status', 'created_date', 'acknowledged_date', 'acknowledged_by', 'resolved_date',
					'resolved_by', 'auto_resolve', 'escalation_level', 'snooze_until', 'resolution_notes']
	edit_columns = ['status', 'acknowledged_date', 'acknowledged_by', 'resolved_date', 'resolved_by',
					'snooze_until', 'resolution_notes']
	
	# Alerts are typically system-generated
	add_columns = []
	can_create = False
	can_delete = False
	
	base_order = ('alert_priority', 'desc', 'created_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'alert_type': 'Alert Type',
		'alert_priority': 'Priority',
		'alert_title': 'Title',
		'alert_message': 'Message',
		'item': 'Item',
		'warehouse': 'Warehouse',
		'location': 'Location',
		'category': 'Category',
		'trigger_condition': 'Trigger Condition',
		'threshold_value': 'Threshold Value',
		'current_value': 'Current Value',
		'created_date': 'Created Date',
		'acknowledged_date': 'Acknowledged Date',
		'acknowledged_by': 'Acknowledged By',
		'resolved_date': 'Resolved Date',
		'resolved_by': 'Resolved By',
		'auto_resolve': 'Auto Resolve',
		'escalation_level': 'Escalation Level',
		'snooze_until': 'Snooze Until',
		'resolution_notes': 'Resolution Notes'
	}
	
	@action("acknowledge", "Acknowledge", "Acknowledge selected alerts", "fa-check", multiple=True)
	def acknowledge_action(self, alerts):
		"""Acknowledge selected alerts"""
		count = 0
		for alert in alerts:
			if alert.status == 'Active':
				alert.status = 'Acknowledged'
				alert.acknowledged_date = datetime.utcnow()
				alert.acknowledged_by = self.get_current_user_id()
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} alerts acknowledged', 'success')
		else:
			flash('No active alerts to acknowledge', 'warning')
		
		return redirect(self.get_redirect())
	
	@action("resolve", "Resolve", "Resolve selected alerts", "fa-times", multiple=True)
	def resolve_action(self, alerts):
		"""Resolve selected alerts"""
		count = 0
		for alert in alerts:
			if alert.status in ['Active', 'Acknowledged']:
				alert.status = 'Resolved'
				alert.resolved_date = datetime.utcnow()
				alert.resolved_by = self.get_current_user_id()
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} alerts resolved', 'success')
		else:
			flash('No alerts to resolve', 'warning')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"
	
	def get_current_user_id(self):
		"""Get current user ID"""
		return "current_user"  # TODO: Implement user resolution


class StockTrackingDashboardView(BaseView):
	"""Stock Tracking & Control Dashboard"""
	
	route_base = "/stock_tracking/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display Stock Tracking dashboard"""
		
		try:
			with StockTrackingService(self.get_tenant_id()) as service:
				# Get dashboard metrics
				dashboard_data = {
					'total_items': service.get_total_item_count(),
					'total_inventory_value': float(service.get_total_inventory_value()),
					'locations_count': service.get_locations_count(),
					'inventory_turnover': service.calculate_inventory_turnover(),
					'stockout_rate': service.calculate_stockout_rate(),
					'low_stock_items': service.get_low_stock_items(limit=10),
					'active_alerts': service.get_active_alerts(limit=10),
					'recent_movements': self._get_recent_movements(service)
				}
				
				return self.render_template(
					'stock_tracking_dashboard.html',
					dashboard_data=dashboard_data,
					title="Stock Tracking & Control Dashboard"
				)
		
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	@expose('/low_stock_report')
	@has_access
	def low_stock_report(self):
		"""Display low stock report"""
		
		try:
			with StockTrackingService(self.get_tenant_id()) as service:
				warehouse_id = request.args.get('warehouse_id')
				low_stock_items = service.get_low_stock_items(warehouse_id=warehouse_id, limit=100)
				
				return self.render_template(
					'low_stock_report.html',
					low_stock_items=low_stock_items,
					warehouse_id=warehouse_id,
					title="Low Stock Report"
				)
		
		except Exception as e:
			flash(f'Error loading low stock report: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	@expose('/inventory_valuation')
	@has_access
	def inventory_valuation(self):
		"""Display inventory valuation report"""
		
		try:
			with StockTrackingService(self.get_tenant_id()) as service:
				warehouse_id = request.args.get('warehouse_id')
				valuation_data = service.get_inventory_valuation(warehouse_id=warehouse_id)
				
				return self.render_template(
					'inventory_valuation.html',
					valuation_data=valuation_data,
					title="Inventory Valuation Report"
				)
		
		except Exception as e:
			flash(f'Error loading inventory valuation: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	@expose('/abc_analysis')
	@has_access
	def abc_analysis(self):
		"""Display ABC analysis report"""
		
		try:
			with StockTrackingService(self.get_tenant_id()) as service:
				warehouse_id = request.args.get('warehouse_id')
				abc_data = service.get_abc_analysis(warehouse_id=warehouse_id)
				
				return self.render_template(
					'abc_analysis.html',
					abc_data=abc_data,
					title="ABC Analysis Report"
				)
		
		except Exception as e:
			flash(f'Error loading ABC analysis: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	def _get_recent_movements(self, service: StockTrackingService) -> List[Dict[str, Any]]:
		"""Get recent stock movements"""
		
		# Get recent movements from the last 7 days
		end_date = datetime.now()
		start_date = end_date - timedelta(days=7)
		
		# This would typically be a method in the service, simplified here
		return []
	
	def get_tenant_id(self):
		"""Get current tenant ID"""
		return "default_tenant"


class StockMovementChartView(GroupByChartView):
	"""Chart view for stock movements"""
	
	datamodel = SQLAInterface(IMSTCStockMovement)
	chart_title = 'Stock Movements by Type'
	label_columns = {'movement_type': 'Movement Type', 'quantity': 'Quantity'}
	group_by_columns = ['movement_type']
	
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	def get_tenant_id(self):
		return "default_tenant"