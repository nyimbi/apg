"""
Order Entry Views

Flask-AppBuilder views for order entry including customers, orders,
and order management interfaces.
"""

from datetime import datetime, date
from typing import Dict, List, Any
from flask import request, flash, redirect, url_for, jsonify
from flask_appbuilder import ModelView, action, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.widgets import ListWidget, ShowWidget
from flask_appbuilder.actions import action
from wtforms import Form, StringField, TextAreaField, DecimalField, DateField, SelectField, BooleanField
from wtforms.validators import DataRequired, Optional, NumberRange

from .models import (
	SOECustomer, SOEShipToAddress, SOESalesOrder, SOEOrderLine, 
	SOEOrderCharge, SOEPriceLevel, SOEOrderTemplate, SOEOrderSequence
)
from .service import OrderEntryService


class SOECustomerView(ModelView):
	"""Customer management view for order entry"""
	
	datamodel = SQLAInterface(SOECustomer)
	
	# List configuration
	list_columns = [
		'customer_number', 'customer_name', 'customer_type', 'email', 
		'phone', 'credit_limit', 'current_balance', 'is_active'
	]
	search_columns = ['customer_number', 'customer_name', 'email', 'phone']
	list_filters = ['customer_type', 'is_active', 'credit_hold']
	
	# Form configuration
	add_columns = [
		'customer_number', 'customer_name', 'customer_type', 'contact_name',
		'email', 'phone', 'mobile', 'billing_address_line1', 'billing_address_line2',
		'billing_city', 'billing_state_province', 'billing_postal_code', 'billing_country',
		'shipping_address_line1', 'shipping_address_line2', 'shipping_city',
		'shipping_state_province', 'shipping_postal_code', 'shipping_country',
		'payment_terms_code', 'preferred_payment_method', 'preferred_shipping_method',
		'credit_limit', 'tax_exempt', 'tax_exempt_number', 'currency_code',
		'sales_rep_id', 'is_active', 'allow_backorders', 'require_po_number',
		'notes'
	]
	edit_columns = add_columns + ['credit_hold', 'current_balance', 'ytd_orders']
	show_columns = edit_columns + ['customer_id', 'tenant_id', 'created_date', 'last_updated']
	
	# Formatting
	formatters_columns = {
		'credit_limit': lambda x: f"${x:,.2f}" if x else "$0.00",
		'current_balance': lambda x: f"${x:,.2f}" if x else "$0.00",
		'ytd_orders': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	# Order and pagination
	base_order = ('customer_name', 'asc')
	page_size = 50
	
	@action('create_order', 'Create Order', 'Create new order for this customer', 'fa-shopping-cart')
	def create_order_action(self, items):
		"""Action to create order for selected customer"""
		if len(items) != 1:
			flash('Please select exactly one customer', 'warning')
			return redirect(request.referrer)
		
		customer = items[0]
		return redirect(url_for('SOESalesOrderView.add', customer_id=customer.customer_id))
	
	@action('view_orders', 'View Orders', 'View orders for selected customers', 'fa-list')
	def view_orders_action(self, items):
		"""Action to view orders for selected customers"""
		if not items:
			flash('Please select at least one customer', 'warning')
			return redirect(request.referrer)
		
		customer_ids = [item.customer_id for item in items]
		# Redirect to orders view with customer filter
		return redirect(url_for('SOESalesOrderView.list', flt1_customer_id=','.join(customer_ids)))
	
	@expose('/api/search/')
	def api_search(self):
		"""API endpoint for customer search"""
		search_term = request.args.get('q', '')
		limit = int(request.args.get('limit', 10))
		
		if not search_term:
			return jsonify([])
		
		# Get service instance
		service = OrderEntryService(self.datamodel.session)
		customers = service.search_customers(
			tenant_id=request.current_user.tenant_id,
			search_term=search_term,
			limit=limit
		)
		
		results = []
		for customer in customers:
			results.append({
				'id': customer.customer_id,
				'text': f"{customer.customer_number} - {customer.customer_name}",
				'customer_number': customer.customer_number,
				'customer_name': customer.customer_name,
				'email': customer.email,
				'credit_limit': float(customer.credit_limit),
				'current_balance': float(customer.current_balance),
				'can_order': customer.is_active and not customer.credit_hold
			})
		
		return jsonify(results)


class SOEShipToAddressView(ModelView):
	"""Ship-to address management view"""
	
	datamodel = SQLAInterface(SOEShipToAddress)
	
	# List configuration
	list_columns = [
		'customer.customer_name', 'address_name', 'city', 'state_province',
		'postal_code', 'country', 'is_default', 'is_active'
	]
	search_columns = ['address_name', 'city', 'state_province', 'country']
	list_filters = ['country', 'state_province', 'is_default', 'is_active']
	
	# Form configuration
	add_columns = [
		'customer', 'address_name', 'contact_name', 'address_line1', 'address_line2',
		'city', 'state_province', 'postal_code', 'country', 'phone', 'email',
		'preferred_carrier', 'delivery_instructions', 'is_default', 'is_active',
		'requires_appointment', 'loading_dock_available'
	]
	edit_columns = add_columns + ['is_validated', 'validation_date']
	show_columns = edit_columns + ['ship_to_id', 'tenant_id', 'created_date']
	
	# Order
	base_order = ('customer.customer_name', 'asc')
	
	@action('validate_addresses', 'Validate Addresses', 'Validate selected addresses', 'fa-check-circle')
	def validate_addresses_action(self, items):
		"""Action to validate addresses"""
		validated_count = 0
		for address in items:
			# Mock address validation
			address.is_validated = True
			address.validation_date = datetime.utcnow()
			address.validation_service = 'Mock Service'
			validated_count += 1
		
		self.datamodel.session.commit()
		flash(f'Validated {validated_count} addresses', 'success')
		return redirect(request.referrer)


class SOESalesOrderView(ModelView):
	"""Sales order management view"""
	
	datamodel = SQLAInterface(SOESalesOrder)
	
	# List configuration
	list_columns = [
		'order_number', 'customer.customer_name', 'order_date', 'status',
		'total_amount', 'sales_rep_id', 'ship_to_address.address_name'
	]
	search_columns = ['order_number', 'customer_po_number', 'customer.customer_name']
	list_filters = ['status', 'order_type', 'order_date', 'sales_rep_id']
	
	# Form configuration
	add_columns = [
		'customer', 'ship_to_address', 'order_date', 'requested_date',
		'order_type', 'customer_po_number', 'payment_method', 'payment_terms_code',
		'shipping_method', 'sales_rep_id', 'description', 'notes'
	]
	edit_columns = add_columns + [
		'status', 'hold_status', 'hold_reason', 'approved', 'approval_notes'
	]
	show_columns = edit_columns + [
		'order_id', 'tenant_id', 'subtotal_amount', 'discount_amount',
		'tax_amount', 'shipping_amount', 'total_amount', 'created_date'
	]
	
	# Formatting
	formatters_columns = {
		'subtotal_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'discount_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'shipping_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'total_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'status': lambda x: f'<span class="label label-{self._get_status_class(x)}">{x}</span>'
	}
	
	# Related views
	related_views = [SOEOrderLine, SOEOrderCharge]
	
	# Order and pagination
	base_order = ('order_date', 'desc')
	page_size = 25
	
	def _get_status_class(self, status):
		"""Get CSS class for order status"""
		status_classes = {
			'DRAFT': 'default',
			'SUBMITTED': 'info',
			'APPROVED': 'success',
			'PROCESSING': 'warning',
			'SHIPPED': 'primary',
			'INVOICED': 'success',
			'CANCELLED': 'danger'
		}
		return status_classes.get(status, 'default')
	
	@action('submit_orders', 'Submit Orders', 'Submit selected orders for processing', 'fa-paper-plane')
	def submit_orders_action(self, items):
		"""Action to submit orders"""
		service = OrderEntryService(self.datamodel.session)
		submitted_count = 0
		errors = []
		
		for order in items:
			if order.status == 'DRAFT':
				try:
					result = service.submit_order(
						order.tenant_id,
						order.order_id,
						request.current_user.user_id
					)
					if result['success']:
						submitted_count += 1
				except Exception as e:
					errors.append(f"Order {order.order_number}: {str(e)}")
		
		if submitted_count > 0:
			flash(f'Successfully submitted {submitted_count} orders', 'success')
		
		if errors:
			for error in errors[:5]:  # Show max 5 errors
				flash(error, 'error')
		
		return redirect(request.referrer)
	
	@action('approve_orders', 'Approve Orders', 'Approve selected orders', 'fa-check')
	def approve_orders_action(self, items):
		"""Action to approve orders"""
		service = OrderEntryService(self.datamodel.session)
		approved_count = 0
		errors = []
		
		for order in items:
			if order.can_approve():
				try:
					result = service.approve_order(
						order.tenant_id,
						order.order_id,
						request.current_user.user_id,
						'Bulk approval action'
					)
					if result['success']:
						approved_count += 1
				except Exception as e:
					errors.append(f"Order {order.order_number}: {str(e)}")
		
		if approved_count > 0:
			flash(f'Successfully approved {approved_count} orders', 'success')
		
		if errors:
			for error in errors[:5]:
				flash(error, 'error')
		
		return redirect(request.referrer)
	
	@action('cancel_orders', 'Cancel Orders', 'Cancel selected orders', 'fa-times')
	def cancel_orders_action(self, items):
		"""Action to cancel orders"""
		# This would typically show a confirmation dialog
		service = OrderEntryService(self.datamodel.session)
		cancelled_count = 0
		errors = []
		
		for order in items:
			if order.can_cancel():
				try:
					result = service.cancel_order(
						order.tenant_id,
						order.order_id,
						request.current_user.user_id,
						'Bulk cancellation action'
					)
					if result['success']:
						cancelled_count += 1
				except Exception as e:
					errors.append(f"Order {order.order_number}: {str(e)}")
		
		if cancelled_count > 0:
			flash(f'Successfully cancelled {cancelled_count} orders', 'warning')
		
		if errors:
			for error in errors[:5]:
				flash(error, 'error')
		
		return redirect(request.referrer)
	
	@expose('/order_entry/')
	def order_entry_form(self):
		"""Custom order entry form"""
		# This would render a specialized order entry interface
		return self.render_template('order_entry/order_form.html')
	
	@expose('/api/calculate_totals/', methods=['POST'])
	def api_calculate_totals(self):
		"""API endpoint to calculate order totals"""
		order_data = request.get_json()
		
		# Mock calculation - in real implementation would use service
		subtotal = sum(
			float(line.get('quantity', 0)) * float(line.get('unit_price', 0))
			for line in order_data.get('lines', [])
		)
		
		tax_rate = 0.0825  # Mock tax rate
		tax_amount = subtotal * tax_rate
		shipping_amount = float(order_data.get('shipping_amount', 0))
		total_amount = subtotal + tax_amount + shipping_amount
		
		return jsonify({
			'subtotal_amount': round(subtotal, 2),
			'tax_amount': round(tax_amount, 2),
			'shipping_amount': shipping_amount,
			'total_amount': round(total_amount, 2)
		})


class SOEOrderLineView(ModelView):
	"""Order line management view"""
	
	datamodel = SQLAInterface(SOEOrderLine)
	
	# List configuration
	list_columns = [
		'sales_order.order_number', 'line_number', 'item_code', 'item_description',
		'quantity_ordered', 'unit_price', 'extended_amount', 'line_status'
	]
	search_columns = ['item_code', 'item_description']
	list_filters = ['line_status', 'item_type', 'line_type']
	
	# Form configuration
	add_columns = [
		'sales_order', 'line_number', 'line_type', 'item_code', 'item_description',
		'quantity_ordered', 'unit_of_measure', 'unit_price', 'discount_percentage',
		'tax_code', 'warehouse_id', 'special_instructions', 'notes'
	]
	edit_columns = add_columns + [
		'quantity_allocated', 'quantity_shipped', 'line_status'
	]
	show_columns = edit_columns + [
		'line_id', 'tenant_id', 'extended_amount', 'tax_amount', 'created_date'
	]
	
	# Formatting
	formatters_columns = {
		'unit_price': lambda x: f"${x:.4f}" if x else "$0.0000",
		'extended_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	# Order
	base_order = ('sales_order.order_number', 'asc')
	
	@action('allocate_inventory', 'Allocate Inventory', 'Allocate inventory for selected lines', 'fa-warehouse')
	def allocate_inventory_action(self, items):
		"""Action to allocate inventory"""
		allocated_count = 0
		for line in items:
			if line.can_allocate_inventory():
				# Mock inventory allocation
				line.quantity_allocated = line.quantity_ordered
				line.inventory_allocated = True
				line.allocation_id = f"ALLOC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
				line.line_status = 'ALLOCATED'
				allocated_count += 1
		
		self.datamodel.session.commit()
		flash(f'Allocated inventory for {allocated_count} lines', 'success')
		return redirect(request.referrer)


class SOEOrderChargeView(ModelView):
	"""Order charge management view"""
	
	datamodel = SQLAInterface(SOEOrderCharge)
	
	# List configuration
	list_columns = [
		'sales_order.order_number', 'charge_type', 'description',
		'charge_amount', 'is_taxable', 'tax_amount'
	]
	search_columns = ['description', 'charge_type']
	list_filters = ['charge_type', 'is_taxable', 'is_automatic']
	
	# Form configuration
	add_columns = [
		'sales_order', 'charge_type', 'description', 'charge_amount',
		'calculation_method', 'is_taxable', 'tax_code', 'notes'
	]
	edit_columns = add_columns + ['tax_amount']
	show_columns = edit_columns + ['charge_id', 'tenant_id', 'created_date']
	
	# Formatting
	formatters_columns = {
		'charge_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'tax_amount': lambda x: f"${x:,.2f}" if x else "$0.00"
	}


class SOEPriceLevelView(ModelView):
	"""Price level management view"""
	
	datamodel = SQLAInterface(SOEPriceLevel)
	
	# List configuration
	list_columns = [
		'level_code', 'level_name', 'discount_percentage', 'markup_percentage',
		'price_calculation_method', 'is_active', 'is_default'
	]
	search_columns = ['level_code', 'level_name']
	list_filters = ['price_calculation_method', 'is_active', 'is_default']
	
	# Form configuration
	add_columns = [
		'level_code', 'level_name', 'description', 'discount_percentage',
		'markup_percentage', 'price_calculation_method', 'minimum_order_amount',
		'minimum_annual_volume', 'customer_type', 'effective_date',
		'expiration_date', 'is_active', 'is_default'
	]
	edit_columns = add_columns
	show_columns = add_columns + ['price_level_id', 'tenant_id', 'created_date']
	
	# Formatting
	formatters_columns = {
		'discount_percentage': lambda x: f"{x}%" if x else "0%",
		'markup_percentage': lambda x: f"{x}%" if x else "0%",
		'minimum_order_amount': lambda x: f"${x:,.2f}" if x else "$0.00",
		'minimum_annual_volume': lambda x: f"${x:,.2f}" if x else "$0.00"
	}
	
	# Order
	base_order = ('level_code', 'asc')


class SOEOrderTemplateView(ModelView):
	"""Order template management view"""
	
	datamodel = SQLAInterface(SOEOrderTemplate)
	
	# List configuration
	list_columns = [
		'template_name', 'customer.customer_name', 'template_type',
		'usage_count', 'last_used_date', 'is_active'
	]
	search_columns = ['template_name', 'description']
	list_filters = ['template_type', 'is_active', 'is_public']
	
	# Form configuration
	add_columns = [
		'template_name', 'description', 'template_type', 'customer',
		'default_ship_to_id', 'default_requested_date_offset',
		'is_active', 'is_public', 'notes'
	]
	edit_columns = add_columns + ['usage_count', 'last_used_date']
	show_columns = edit_columns + ['template_id', 'tenant_id', 'created_date']
	
	# Related views
	related_views = ['SOEOrderTemplateLine']
	
	@action('create_orders', 'Create Orders', 'Create orders from selected templates', 'fa-plus')
	def create_orders_action(self, items):
		"""Action to create orders from templates"""
		if not items:
			flash('Please select at least one template', 'warning')
			return redirect(request.referrer)
		
		# This would typically show a form to select customer and other details
		template_ids = [item.template_id for item in items]
		return redirect(url_for('order_from_template', template_ids=','.join(template_ids)))


class SOEOrderSequenceView(ModelView):
	"""Order sequence management view"""
	
	datamodel = SQLAInterface(SOEOrderSequence)
	
	# List configuration
	list_columns = [
		'sequence_name', 'order_type', 'prefix', 'current_number',
		'suffix', 'reset_period', 'is_active'
	]
	search_columns = ['sequence_name', 'order_type']
	list_filters = ['order_type', 'reset_period', 'is_active']
	
	# Form configuration
	add_columns = [
		'sequence_name', 'order_type', 'prefix', 'suffix', 'number_length',
		'current_number', 'increment_by', 'reset_period', 'zero_pad', 'is_active'
	]
	edit_columns = add_columns + ['last_reset_date']
	show_columns = edit_columns + ['sequence_id', 'tenant_id', 'created_date']
	
	# Order
	base_order = ('order_type', 'asc')
	
	@action('reset_sequences', 'Reset Sequences', 'Reset selected sequences to 1', 'fa-redo')
	def reset_sequences_action(self, items):
		"""Action to reset sequences"""
		reset_count = 0
		for sequence in items:
			sequence.current_number = 1
			sequence.last_reset_date = date.today()
			reset_count += 1
		
		self.datamodel.session.commit()
		flash(f'Reset {reset_count} sequences', 'success')
		return redirect(request.referrer)


class OrderEntryDashboardView(ModelView):
	"""Dashboard view for order entry metrics and KPIs"""
	
	datamodel = SQLAInterface(SOESalesOrder)
	
	@expose('/dashboard/')
	def dashboard(self):
		"""Order entry dashboard"""
		service = OrderEntryService(self.datamodel.session)
		
		# Get metrics for current month
		today = date.today()
		start_of_month = today.replace(day=1)
		
		metrics = service.get_order_metrics(
			tenant_id=request.current_user.tenant_id,
			date_range=(start_of_month, today)
		)
		
		# Get pending orders count
		pending_orders = self.datamodel.session.query(SOESalesOrder).filter(
			SOESalesOrder.tenant_id == request.current_user.tenant_id,
			SOESalesOrder.status.in_(['DRAFT', 'SUBMITTED'])
		).count()
		
		# Get orders requiring approval
		approval_orders = self.datamodel.session.query(SOESalesOrder).filter(
			SOESalesOrder.tenant_id == request.current_user.tenant_id,
			SOESalesOrder.status == 'SUBMITTED',
			SOESalesOrder.requires_approval == True
		).count()
		
		dashboard_data = {
			'metrics': metrics,
			'pending_orders': pending_orders,
			'approval_orders': approval_orders,
			'today': today
		}
		
		return self.render_template(
			'order_entry/dashboard.html',
			dashboard_data=dashboard_data
		)