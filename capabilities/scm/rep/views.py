"""
Replenishment & Reordering Views

Flask-AppBuilder views for supplier management, replenishment rules,
purchase orders, and automated reordering.
"""

from flask import request, jsonify, flash, redirect, url_for
from flask_appbuilder import ModelView, BaseView, expose, has_access
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.actions import action
from datetime import datetime, timedelta

from .models import (
	IMRRSupplier, IMRRSupplierItem, IMRRReplenishmentRule,
	IMRRReplenishmentSuggestion, IMRRPurchaseOrder, 
	IMRRPurchaseOrderLine, IMRRDemandForecast
)
from .service import ReplenishmentService


class IMRRSupplierView(ModelView):
	"""View for managing suppliers"""
	
	datamodel = SQLAInterface(IMRRSupplier)
	
	list_columns = ['supplier_code', 'supplier_name', 'supplier_type', 'lead_time_days', 
					'on_time_delivery_rate', 'is_active', 'is_approved']
	show_columns = ['supplier_code', 'supplier_name', 'description', 'supplier_type',
					'contact_person', 'email_address', 'phone_number', 'website_url',
					'address_line1', 'city', 'state_province', 'country_code',
					'payment_terms', 'currency_code', 'minimum_order_amount', 'lead_time_days',
					'on_time_delivery_rate', 'quality_rating', 'price_competitiveness',
					'is_active', 'is_approved']
	edit_columns = show_columns
	add_columns = edit_columns
	
	base_order = ('supplier_code', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'supplier_code': 'Supplier Code',
		'supplier_name': 'Supplier Name',
		'supplier_type': 'Supplier Type',
		'contact_person': 'Contact Person',
		'email_address': 'Email',
		'phone_number': 'Phone',
		'website_url': 'Website',
		'address_line1': 'Address',
		'state_province': 'State/Province',
		'country_code': 'Country',
		'payment_terms': 'Payment Terms',
		'currency_code': 'Currency',
		'minimum_order_amount': 'Min Order Amount',
		'lead_time_days': 'Lead Time (Days)',
		'on_time_delivery_rate': 'On-Time Delivery %',
		'quality_rating': 'Quality Rating',
		'price_competitiveness': 'Price Rating',
		'is_approved': 'Approved'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class IMRRReplenishmentRuleView(ModelView):
	"""View for managing replenishment rules"""
	
	datamodel = SQLAInterface(IMRRReplenishmentRule)
	
	list_columns = ['rule_name', 'rule_type', 'item_id', 'reorder_point', 'reorder_quantity', 
					'auto_generate_po', 'is_active']
	show_columns = ['rule_name', 'description', 'rule_type', 'item_id', 'category_id', 
					'warehouse_id', 'abc_classification', 'reorder_point', 'reorder_quantity',
					'max_stock_level', 'safety_stock', 'lead_time_days', 'auto_generate_po',
					'auto_approve_po', 'preferred_supplier', 'is_active', 'last_run_date',
					'next_run_date']
	edit_columns = ['rule_name', 'description', 'rule_type', 'item_id', 'category_id',
					'warehouse_id', 'abc_classification', 'reorder_point', 'reorder_quantity',
					'max_stock_level', 'safety_stock', 'lead_time_days', 'auto_generate_po',
					'auto_approve_po', 'preferred_supplier_id', 'is_active']
	add_columns = edit_columns
	
	base_order = ('rule_name', 'asc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'rule_name': 'Rule Name',
		'rule_type': 'Rule Type',
		'item_id': 'Item ID',
		'category_id': 'Category ID',
		'warehouse_id': 'Warehouse ID',
		'abc_classification': 'ABC Class',
		'reorder_point': 'Reorder Point',
		'reorder_quantity': 'Reorder Quantity',
		'max_stock_level': 'Max Stock Level',
		'safety_stock': 'Safety Stock',
		'lead_time_days': 'Lead Time (Days)',
		'auto_generate_po': 'Auto Generate PO',
		'auto_approve_po': 'Auto Approve PO',
		'preferred_supplier': 'Preferred Supplier',
		'last_run_date': 'Last Run Date',
		'next_run_date': 'Next Run Date'
	}
	
	@action("run_analysis", "Run Analysis", "Run replenishment analysis", "fa-play", multiple=True)
	def run_analysis_action(self, rules):
		"""Run replenishment analysis for selected rules"""
		try:
			with ReplenishmentService(self.get_tenant_id()) as service:
				total_suggestions = 0
				for rule in rules:
					suggestions = service.run_replenishment_analysis(rule.rule_id)
					total_suggestions += len(suggestions)
				
				flash(f'Generated {total_suggestions} replenishment suggestions', 'success')
		except Exception as e:
			flash(f'Error running analysis: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"


class IMRRReplenishmentSuggestionView(ModelView):
	"""View for managing replenishment suggestions"""
	
	datamodel = SQLAInterface(IMRRReplenishmentSuggestion)
	
	list_columns = ['suggestion_date', 'item_id', 'warehouse_id', 'current_stock', 'suggested_quantity',
					'priority_level', 'status']
	show_columns = ['suggestion_date', 'rule', 'item_id', 'warehouse_id', 'current_stock',
					'allocated_stock', 'available_stock', 'on_order_quantity', 'reorder_point',
					'suggested_quantity', 'recommended_supplier', 'unit_cost', 'total_cost',
					'priority_level', 'urgency_score', 'stockout_risk', 'status', 'reason_code']
	edit_columns = ['status', 'reviewed_by', 'review_date', 'review_notes']
	
	# Suggestions are typically system-generated
	add_columns = []
	can_create = False
	can_delete = False
	
	base_order = ('suggestion_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'suggestion_date': 'Suggestion Date',
		'item_id': 'Item ID',
		'warehouse_id': 'Warehouse ID',
		'current_stock': 'Current Stock',
		'allocated_stock': 'Allocated Stock',
		'available_stock': 'Available Stock',
		'on_order_quantity': 'On Order',
		'reorder_point': 'Reorder Point',
		'suggested_quantity': 'Suggested Quantity',
		'recommended_supplier': 'Recommended Supplier',
		'unit_cost': 'Unit Cost',
		'total_cost': 'Total Cost',
		'priority_level': 'Priority',
		'urgency_score': 'Urgency Score',
		'stockout_risk': 'Stockout Risk %',
		'reason_code': 'Reason Code',
		'reviewed_by': 'Reviewed By',
		'review_date': 'Review Date',
		'review_notes': 'Review Notes'
	}
	
	@action("approve", "Approve", "Approve selected suggestions", "fa-check", multiple=True)
	def approve_action(self, suggestions):
		"""Approve selected suggestions"""
		count = 0
		for suggestion in suggestions:
			if suggestion.status == 'Pending':
				suggestion.status = 'Approved'
				suggestion.review_date = datetime.utcnow()
				suggestion.reviewed_by = self.get_current_user_id()
				count += 1
		
		if count > 0:
			self.datamodel.session.commit()
			flash(f'{count} suggestions approved', 'success')
		else:
			flash('No pending suggestions to approve', 'warning')
		
		return redirect(self.get_redirect())
	
	@action("convert_to_po", "Convert to PO", "Convert to purchase order", "fa-shopping-cart", multiple=False)
	def convert_to_po_action(self, suggestions):
		"""Convert approved suggestion to purchase order"""
		if len(suggestions) != 1:
			flash('Please select exactly one suggestion', 'warning')
			return redirect(self.get_redirect())
		
		suggestion = suggestions[0]
		if suggestion.status != 'Approved':
			flash('Suggestion must be approved before conversion', 'warning')
			return redirect(self.get_redirect())
		
		try:
			with ReplenishmentService(self.get_tenant_id()) as service:
				po = service.convert_suggestion_to_po(suggestion.suggestion_id)
				flash(f'Purchase order {po.po_number} created successfully', 'success')
		except Exception as e:
			flash(f'Error creating purchase order: {str(e)}', 'error')
		
		return redirect(self.get_redirect())
	
	def get_tenant_id(self):
		return "default_tenant"
	
	def get_current_user_id(self):
		return "current_user"


class IMRRPurchaseOrderView(ModelView):
	"""View for managing purchase orders"""
	
	datamodel = SQLAInterface(IMRRPurchaseOrder)
	
	list_columns = ['po_number', 'po_date', 'supplier.supplier_name', 'total_amount', 
					'status', 'requested_delivery_date']
	show_columns = ['po_number', 'po_date', 'supplier', 'requested_delivery_date', 
					'promised_delivery_date', 'warehouse_id', 'currency_code', 'subtotal_amount',
					'tax_amount', 'shipping_amount', 'total_amount', 'status', 'approval_status',
					'created_from', 'terms_and_conditions', 'special_instructions', 'notes']
	edit_columns = ['po_number', 'supplier_id', 'requested_delivery_date', 'warehouse_id',
					'currency_code', 'subtotal_amount', 'tax_amount', 'shipping_amount',
					'total_amount', 'status', 'terms_and_conditions', 'special_instructions', 'notes']
	add_columns = edit_columns
	
	base_order = ('po_date', 'desc')
	base_filters = [['tenant_id', '==', 'get_tenant_id']]
	
	label_columns = {
		'po_number': 'PO Number',
		'po_date': 'PO Date',
		'supplier': 'Supplier',
		'requested_delivery_date': 'Requested Delivery',
		'promised_delivery_date': 'Promised Delivery',
		'warehouse_id': 'Warehouse ID',
		'currency_code': 'Currency',
		'subtotal_amount': 'Subtotal',
		'tax_amount': 'Tax Amount',
		'shipping_amount': 'Shipping',
		'total_amount': 'Total Amount',
		'approval_status': 'Approval Status',
		'created_from': 'Created From',
		'terms_and_conditions': 'Terms & Conditions',
		'special_instructions': 'Special Instructions'
	}
	
	def get_tenant_id(self):
		return "default_tenant"


class ReplenishmentDashboardView(BaseView):
	"""Replenishment & Reordering Dashboard"""
	
	route_base = "/replenishment/dashboard"
	default_view = 'index'
	
	@expose('/')
	@has_access
	def index(self):
		"""Display Replenishment dashboard"""
		
		try:
			with ReplenishmentService(self.get_tenant_id()) as service:
				dashboard_data = {
					'pending_orders': service.get_pending_purchase_orders_count(),
					'auto_reorder_items': service.get_auto_reorder_items_count(),
					'overdue_orders': service.get_overdue_orders_count(),
					'pending_suggestions': self._get_pending_suggestions_count(),
					'recent_pos': self._get_recent_pos()
				}
				
				return self.render_template(
					'replenishment_dashboard.html',
					dashboard_data=dashboard_data,
					title="Replenishment & Reordering Dashboard"
				)
		
		except Exception as e:
			flash(f'Error loading dashboard: {str(e)}', 'error')
			return self.render_template('error.html', error=str(e))
	
	def _get_pending_suggestions_count(self) -> int:
		"""Get count of pending suggestions"""
		from ....auth_rbac.models import get_session
		session = get_session()
		try:
			return session.query(IMRRReplenishmentSuggestion).filter(
				and_(
					IMRRReplenishmentSuggestion.tenant_id == self.get_tenant_id(),
					IMRRReplenishmentSuggestion.status == 'Pending'
				)
			).count()
		finally:
			session.close()
	
	def _get_recent_pos(self) -> list:
		"""Get recent purchase orders"""
		return []  # Simplified for now
	
	def get_tenant_id(self):
		return "default_tenant"