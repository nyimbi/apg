"""
Order Entry REST API

REST API endpoints for order entry operations including customer management,
order creation, and order processing.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from flask_appbuilder.security.decorators import protect
from marshmallow import Schema, fields as ma_fields, validate, post_load
from decimal import Decimal

from .models import (
	SOECustomer, SOEShipToAddress, SOESalesOrder, SOEOrderLine, 
	SOEOrderCharge, SOEPriceLevel, SOEOrderTemplate
)
from .service import OrderEntryService

# Create API blueprint
api_bp = Blueprint('order_entry_api', __name__, url_prefix='/api/v1/order_entry')
api = Api(api_bp, doc='/api/v1/order_entry/docs/', title='Order Entry API', version='1.0')

# Create namespaces
customers_ns = Namespace('customers', description='Customer operations')
orders_ns = Namespace('orders', description='Sales order operations')
templates_ns = Namespace('templates', description='Order template operations')

api.add_namespace(customers_ns)
api.add_namespace(orders_ns)
api.add_namespace(templates_ns)

# Schemas for serialization/deserialization

class CustomerSchema(Schema):
	"""Customer schema for API responses"""
	customer_id = ma_fields.Str(dump_only=True)
	customer_number = ma_fields.Str(required=True)
	customer_name = ma_fields.Str(required=True)
	customer_type = ma_fields.Str(validate=validate.OneOf(['RETAIL', 'WHOLESALE', 'CORPORATE']))
	contact_name = ma_fields.Str(allow_none=True)
	email = ma_fields.Email(allow_none=True)
	phone = ma_fields.Str(allow_none=True)
	credit_limit = ma_fields.Decimal(places=2)
	current_balance = ma_fields.Decimal(places=2, dump_only=True)
	is_active = ma_fields.Bool()
	credit_hold = ma_fields.Bool(dump_only=True)

class OrderLineSchema(Schema):
	"""Order line schema"""
	line_id = ma_fields.Str(dump_only=True)
	line_number = ma_fields.Int(required=True)
	item_code = ma_fields.Str(required=True)
	item_description = ma_fields.Str(allow_none=True)
	quantity_ordered = ma_fields.Decimal(places=4, required=True)
	unit_price = ma_fields.Decimal(places=4, required=True)
	extended_amount = ma_fields.Decimal(places=2, dump_only=True)
	discount_percentage = ma_fields.Decimal(places=2)
	tax_amount = ma_fields.Decimal(places=2, dump_only=True)
	line_status = ma_fields.Str(dump_only=True)

class OrderChargeSchema(Schema):
	"""Order charge schema"""
	charge_id = ma_fields.Str(dump_only=True)
	charge_type = ma_fields.Str(required=True)
	description = ma_fields.Str(required=True)
	charge_amount = ma_fields.Decimal(places=2, required=True)
	is_taxable = ma_fields.Bool()
	tax_amount = ma_fields.Decimal(places=2, dump_only=True)

class SalesOrderSchema(Schema):
	"""Sales order schema"""
	order_id = ma_fields.Str(dump_only=True)
	order_number = ma_fields.Str(dump_only=True)
	customer_id = ma_fields.Str(required=True)
	ship_to_id = ma_fields.Str(allow_none=True)
	order_date = ma_fields.Date(required=True)
	requested_date = ma_fields.Date(allow_none=True)
	customer_po_number = ma_fields.Str(allow_none=True)
	order_type = ma_fields.Str(validate=validate.OneOf(['STANDARD', 'RUSH', 'BACKORDER', 'DROP_SHIP']))
	status = ma_fields.Str(dump_only=True)
	subtotal_amount = ma_fields.Decimal(places=2, dump_only=True)
	tax_amount = ma_fields.Decimal(places=2, dump_only=True)
	shipping_amount = ma_fields.Decimal(places=2, dump_only=True)
	total_amount = ma_fields.Decimal(places=2, dump_only=True)
	lines = ma_fields.Nested(OrderLineSchema, many=True)
	charges = ma_fields.Nested(OrderChargeSchema, many=True)
	notes = ma_fields.Str(allow_none=True)

# Initialize schemas
customer_schema = CustomerSchema()
customers_schema = CustomerSchema(many=True)
order_schema = SalesOrderSchema()
orders_schema = SalesOrderSchema(many=True)

# Customer API endpoints

@customers_ns.route('/')
class CustomerListAPI(Resource):
	"""Customer list and creation endpoint"""
	
	@protect
	def get(self):
		"""Get list of customers with optional search"""
		tenant_id = request.headers.get('X-Tenant-ID')
		search_term = request.args.get('search', '')
		limit = int(request.args.get('limit', 50))
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		
		if search_term:
			customers = service.search_customers(tenant_id, search_term, limit)
		else:
			# Get all active customers (implement in service)
			customers = service.get_active_customers(tenant_id, limit)
		
		return {
			'customers': customers_schema.dump(customers),
			'total': len(customers)
		}
	
	@protect
	def post(self):
		"""Create new customer"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json()
		
		try:
			# Validate input
			customer_data = customer_schema.load(data)
			customer_data['tenant_id'] = tenant_id
			
			service = OrderEntryService(current_app.appbuilder.get_session)
			customer = service.create_customer(customer_data, request.current_user.user_id)
			
			return {
				'success': True,
				'customer': customer_schema.dump(customer)
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@customers_ns.route('/<string:customer_id>')
class CustomerAPI(Resource):
	"""Individual customer operations"""
	
	@protect
	def get(self, customer_id):
		"""Get customer by ID"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		customer = service.get_customer(tenant_id, customer_id)
		
		if not customer:
			return {'error': 'Customer not found'}, 404
		
		return {
			'customer': customer_schema.dump(customer)
		}
	
	@protect
	def put(self, customer_id):
		"""Update customer"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json()
		
		try:
			service = OrderEntryService(current_app.appbuilder.get_session)
			customer = service.get_customer(tenant_id, customer_id)
			
			if not customer:
				return {'error': 'Customer not found'}, 404
			
			# Update customer (implement in service)
			updated_customer = service.update_customer(customer, data, request.current_user.user_id)
			
			return {
				'success': True,
				'customer': customer_schema.dump(updated_customer)
			}
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@customers_ns.route('/<string:customer_id>/validation')
class CustomerValidationAPI(Resource):
	"""Customer validation for orders"""
	
	@protect
	def post(self, customer_id):
		"""Validate customer for order placement"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json()
		order_amount = Decimal(str(data.get('order_amount', 0)))
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		customer = service.get_customer(tenant_id, customer_id)
		
		if not customer:
			return {'error': 'Customer not found'}, 404
		
		validation_result = service.validate_customer_for_order(customer, order_amount)
		
		return {
			'validation': validation_result
		}

@customers_ns.route('/<string:customer_id>/ship-to-addresses')
class CustomerShipToAddressesAPI(Resource):
	"""Customer ship-to addresses"""
	
	@protect
	def get(self, customer_id):
		"""Get customer ship-to addresses"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		addresses = service.get_customer_ship_to_addresses(tenant_id, customer_id)
		
		# Convert to dict format (implement schema if needed)
		address_data = []
		for addr in addresses:
			address_data.append({
				'ship_to_id': addr.ship_to_id,
				'address_name': addr.address_name,
				'address_line1': addr.address_line1,
				'city': addr.city,
				'state_province': addr.state_province,
				'postal_code': addr.postal_code,
				'country': addr.country,
				'is_default': addr.is_default
			})
		
		return {
			'addresses': address_data
		}

# Sales Order API endpoints

@orders_ns.route('/')
class OrderListAPI(Resource):
	"""Order list and creation endpoint"""
	
	@protect
	def get(self):
		"""Get list of orders with filters"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		# Parse filters from query parameters
		filters = {
			'customer_id': request.args.get('customer_id'),
			'status': request.args.getlist('status') or request.args.get('status'),
			'order_date_from': request.args.get('order_date_from'),
			'order_date_to': request.args.get('order_date_to'),
			'sales_rep_id': request.args.get('sales_rep_id'),
			'order_number': request.args.get('order_number'),
			'limit': int(request.args.get('limit', 100))
		}
		
		# Convert date strings to date objects
		if filters['order_date_from']:
			filters['order_date_from'] = datetime.strptime(filters['order_date_from'], '%Y-%m-%d').date()
		if filters['order_date_to']:
			filters['order_date_to'] = datetime.strptime(filters['order_date_to'], '%Y-%m-%d').date()
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		orders = service.search_orders(tenant_id, filters)
		
		return {
			'orders': orders_schema.dump(orders),
			'total': len(orders)
		}
	
	@protect
	def post(self):
		"""Create new sales order"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json()
		
		try:
			# Validate input
			order_data = order_schema.load(data)
			order_data['tenant_id'] = tenant_id
			
			service = OrderEntryService(current_app.appbuilder.get_session)
			order = service.create_sales_order(tenant_id, order_data, request.current_user.user_id)
			
			return {
				'success': True,
				'order': order_schema.dump(order)
			}, 201
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@orders_ns.route('/<string:order_id>')
class OrderAPI(Resource):
	"""Individual order operations"""
	
	@protect
	def get(self, order_id):
		"""Get order by ID"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		order = service.get_sales_order(tenant_id, order_id)
		
		if not order:
			return {'error': 'Order not found'}, 404
		
		return {
			'order': order_schema.dump(order)
		}

@orders_ns.route('/<string:order_id>/submit')
class OrderSubmitAPI(Resource):
	"""Order submission endpoint"""
	
	@protect
	def post(self, order_id):
		"""Submit order for processing"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		try:
			service = OrderEntryService(current_app.appbuilder.get_session)
			result = service.submit_order(tenant_id, order_id, request.current_user.user_id)
			
			return result
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@orders_ns.route('/<string:order_id>/approve')
class OrderApprovalAPI(Resource):
	"""Order approval endpoint"""
	
	@protect
	def post(self, order_id):
		"""Approve order"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json() or {}
		notes = data.get('notes', '')
		
		try:
			service = OrderEntryService(current_app.appbuilder.get_session)
			result = service.approve_order(tenant_id, order_id, request.current_user.user_id, notes)
			
			return result
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@orders_ns.route('/<string:order_id>/cancel')
class OrderCancellationAPI(Resource):
	"""Order cancellation endpoint"""
	
	@protect
	def post(self, order_id):
		"""Cancel order"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json() or {}
		reason = data.get('reason', 'No reason provided')
		
		try:
			service = OrderEntryService(current_app.appbuilder.get_session)
			result = service.cancel_order(tenant_id, order_id, request.current_user.user_id, reason)
			
			return result
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

@orders_ns.route('/metrics')
class OrderMetricsAPI(Resource):
	"""Order metrics and analytics"""
	
	@protect
	def get(self):
		"""Get order metrics for date range"""
		tenant_id = request.headers.get('X-Tenant-ID')
		
		# Parse date range
		start_date_str = request.args.get('start_date', date.today().replace(day=1).isoformat())
		end_date_str = request.args.get('end_date', date.today().isoformat())
		
		start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
		end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		metrics = service.get_order_metrics(tenant_id, (start_date, end_date))
		
		return {
			'metrics': metrics
		}

# Order Template API endpoints

@templates_ns.route('/')
class OrderTemplateListAPI(Resource):
	"""Order template list endpoint"""
	
	@protect
	def get(self):
		"""Get list of order templates"""
		tenant_id = request.headers.get('X-Tenant-ID')
		customer_id = request.args.get('customer_id')
		
		service = OrderEntryService(current_app.appbuilder.get_session)
		templates = service.get_order_templates(tenant_id, customer_id)
		
		template_data = []
		for template in templates:
			template_data.append({
				'template_id': template.template_id,
				'template_name': template.template_name,
				'description': template.description,
				'template_type': template.template_type,
				'usage_count': template.usage_count,
				'last_used_date': template.last_used_date.isoformat() if template.last_used_date else None
			})
		
		return {
			'templates': template_data
		}

@templates_ns.route('/<string:template_id>/create-order')
class OrderFromTemplateAPI(Resource):
	"""Create order from template endpoint"""
	
	@protect
	def post(self, template_id):
		"""Create order from template"""
		tenant_id = request.headers.get('X-Tenant-ID')
		data = request.get_json()
		customer_id = data.get('customer_id')
		modifications = data.get('modifications', {})
		
		if not customer_id:
			return {'error': 'customer_id is required'}, 400
		
		try:
			service = OrderEntryService(current_app.appbuilder.get_session)
			order = service.create_order_from_template(
				tenant_id, template_id, customer_id, 
				request.current_user.user_id, modifications
			)
			
			return {
				'success': True,
				'order': order_schema.dump(order)
			}
			
		except Exception as e:
			return {
				'success': False,
				'error': str(e)
			}, 400

def init_api(app):
	"""Initialize API with Flask app"""
	app.register_blueprint(api_bp)