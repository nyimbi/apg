"""
Order Entry Service

Business logic for customer order entry including order creation, validation,
pricing calculation, and inventory checking.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from .models import (
	SOECustomer, SOEShipToAddress, SOESalesOrder, SOEOrderLine, 
	SOEOrderCharge, SOEPriceLevel, SOEOrderTemplate, SOEOrderTemplateLine,
	SOEOrderSequence
)


class OrderEntryService:
	"""Service class for order entry operations"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	# Customer Management Methods
	
	def get_customer(self, tenant_id: str, customer_id: str) -> Optional[SOECustomer]:
		"""Get customer by ID"""
		return self.db.query(SOECustomer).filter(
			and_(
				SOECustomer.tenant_id == tenant_id,
				SOECustomer.customer_id == customer_id,
				SOECustomer.is_active == True
			)
		).first()
	
	def search_customers(self, tenant_id: str, search_term: str, limit: int = 50) -> List[SOECustomer]:
		"""Search customers by name, number, or email"""
		search_filter = or_(
			SOECustomer.customer_name.ilike(f'%{search_term}%'),
			SOECustomer.customer_number.ilike(f'%{search_term}%'),
			SOECustomer.email.ilike(f'%{search_term}%')
		)
		
		return self.db.query(SOECustomer).filter(
			and_(
				SOECustomer.tenant_id == tenant_id,
				SOECustomer.is_active == True,
				search_filter
			)
		).limit(limit).all()
	
	def get_customer_ship_to_addresses(self, tenant_id: str, customer_id: str) -> List[SOEShipToAddress]:
		"""Get all active ship-to addresses for a customer"""
		return self.db.query(SOEShipToAddress).filter(
			and_(
				SOEShipToAddress.tenant_id == tenant_id,
				SOEShipToAddress.customer_id == customer_id,
				SOEShipToAddress.is_active == True
			)
		).order_by(SOEShipToAddress.is_default.desc(), SOEShipToAddress.address_name).all()
	
	def validate_customer_for_order(self, customer: SOECustomer, order_amount: Decimal) -> Dict[str, Any]:
		"""Validate customer can place an order"""
		return customer.can_place_order(order_amount)
	
	# Order Creation and Management
	
	def create_sales_order(self, tenant_id: str, order_data: Dict[str, Any], user_id: str) -> SOESalesOrder:
		"""Create a new sales order"""
		# Validate customer
		customer = self.get_customer(tenant_id, order_data['customer_id'])
		if not customer:
			raise ValueError("Customer not found or inactive")
		
		# Generate order number
		order_number = self._generate_order_number(tenant_id, order_data.get('order_type', 'STANDARD'))
		
		# Create order header
		order = SOESalesOrder(
			tenant_id=tenant_id,
			order_number=order_number,
			customer_id=order_data['customer_id'],
			ship_to_id=order_data.get('ship_to_id'),
			order_date=order_data.get('order_date', date.today()),
			requested_date=order_data.get('requested_date'),
			customer_po_number=order_data.get('customer_po_number'),
			quote_id=order_data.get('quote_id'),
			order_type=order_data.get('order_type', 'STANDARD'),
			description=order_data.get('description'),
			sales_rep_id=order_data.get('sales_rep_id'),
			payment_method=order_data.get('payment_method'),
			payment_terms_code=order_data.get('payment_terms_code', customer.payment_terms_code),
			shipping_method=order_data.get('shipping_method', customer.preferred_shipping_method),
			currency_code=order_data.get('currency_code', customer.currency_code),
			notes=order_data.get('notes'),
			created_by_user_id=user_id
		)
		
		# Set price level
		if order_data.get('price_level_id'):
			order.price_level_id = order_data['price_level_id']
		elif customer.price_level_id:
			order.price_level_id = customer.price_level_id
		
		self.db.add(order)
		self.db.flush()  # Get order ID
		
		# Add order lines
		if 'lines' in order_data:
			for line_num, line_data in enumerate(order_data['lines'], 1):
				self._create_order_line(order, line_num, line_data, user_id)
		
		# Add order charges
		if 'charges' in order_data:
			for charge_data in order_data['charges']:
				self._create_order_charge(order, charge_data, user_id)
		
		# Calculate totals
		order.calculate_totals()
		
		# Auto-calculate shipping if not provided
		if not order.shipping_amount and order_data.get('calculate_shipping', True):
			shipping_charge = self._calculate_shipping(order)
			if shipping_charge:
				self.db.add(shipping_charge)
		
		self.db.commit()
		return order
	
	def _create_order_line(self, order: SOESalesOrder, line_number: int, line_data: Dict[str, Any], user_id: str) -> SOEOrderLine:
		"""Create an order line"""
		# Get item information (would integrate with inventory management)
		item_info = self._get_item_info(order.tenant_id, line_data.get('item_id'), line_data.get('item_code'))
		
		# Calculate pricing
		pricing_info = self._calculate_line_pricing(
			order.tenant_id,
			order.customer_id,
			order.price_level_id,
			item_info,
			line_data.get('quantity_ordered', 1),
			line_data.get('unit_price')
		)
		
		line = SOEOrderLine(
			order_id=order.order_id,
			tenant_id=order.tenant_id,
			line_number=line_number,
			line_type=line_data.get('line_type', 'PRODUCT'),
			item_id=line_data.get('item_id'),
			item_code=line_data.get('item_code', item_info.get('item_code')),
			item_description=line_data.get('item_description', item_info.get('description')),
			item_type=line_data.get('item_type', item_info.get('item_type', 'PRODUCT')),
			quantity_ordered=line_data.get('quantity_ordered', 1),
			unit_of_measure=line_data.get('unit_of_measure', item_info.get('unit_of_measure', 'EA')),
			unit_price=pricing_info['unit_price'],
			list_price=pricing_info['list_price'],
			cost_price=pricing_info.get('cost_price', 0),
			discount_percentage=pricing_info.get('discount_percentage', 0),
			warehouse_id=line_data.get('warehouse_id', order.warehouse_id),
			requested_date=line_data.get('requested_date', order.requested_date),
			tax_code=line_data.get('tax_code', order.customer.default_tax_code),
			special_instructions=line_data.get('special_instructions'),
			notes=line_data.get('notes'),
			created_by_user_id=user_id
		)
		
		# Calculate extended amounts
		line.calculate_extended_amount()
		
		# Calculate tax
		if line.tax_code:
			tax_rate = self._get_tax_rate(order.tenant_id, line.tax_code, order.ship_to_address)
			line.tax_rate = tax_rate
			line.calculate_tax()
		
		# Calculate commission
		if order.commission_rate:
			line.commission_rate = order.commission_rate
			line.calculate_commission()
		
		self.db.add(line)
		return line
	
	def _create_order_charge(self, order: SOESalesOrder, charge_data: Dict[str, Any], user_id: str) -> SOEOrderCharge:
		"""Create an order charge"""
		charge = SOEOrderCharge(
			order_id=order.order_id,
			tenant_id=order.tenant_id,
			charge_type=charge_data['charge_type'],
			charge_code=charge_data.get('charge_code'),
			description=charge_data['description'],
			charge_amount=charge_data['charge_amount'],
			calculation_method=charge_data.get('calculation_method', 'FIXED'),
			calculation_base=charge_data.get('calculation_base'),
			is_taxable=charge_data.get('is_taxable', False),
			tax_code=charge_data.get('tax_code'),
			gl_account_id=charge_data.get('gl_account_id'),
			is_automatic=charge_data.get('is_automatic', False),
			notes=charge_data.get('notes'),
			created_by_user_id=user_id
		)
		
		# Calculate tax if applicable
		if charge.is_taxable and charge.tax_code:
			tax_rate = self._get_tax_rate(order.tenant_id, charge.tax_code, order.ship_to_address)
			charge.tax_rate = tax_rate
			charge.calculate_tax()
		
		self.db.add(charge)
		return charge
	
	# Order Processing Methods
	
	def submit_order(self, tenant_id: str, order_id: str, user_id: str) -> Dict[str, Any]:
		"""Submit order for processing"""
		order = self.get_sales_order(tenant_id, order_id)
		if not order:
			raise ValueError("Order not found")
		
		try:
			order.submit_order(user_id)
			
			# Check inventory availability
			inventory_check = self._check_inventory_availability(order)
			
			# Apply any automatic holds
			self._apply_automatic_holds(order)
			
			self.db.commit()
			
			return {
				'success': True,
				'order_number': order.order_number,
				'status': order.status,
				'requires_approval': order.requires_approval,
				'inventory_check': inventory_check,
				'holds': order.hold_status
			}
		except Exception as e:
			self.db.rollback()
			raise e
	
	def approve_order(self, tenant_id: str, order_id: str, user_id: str, notes: str = None) -> Dict[str, Any]:
		"""Approve an order"""
		order = self.get_sales_order(tenant_id, order_id)
		if not order:
			raise ValueError("Order not found")
		
		try:
			order.approve_order(user_id, notes)
			self.db.commit()
			
			return {
				'success': True,
				'order_number': order.order_number,
				'status': order.status,
				'approved_by': user_id,
				'approved_date': order.approved_date
			}
		except Exception as e:
			self.db.rollback()
			raise e
	
	def cancel_order(self, tenant_id: str, order_id: str, user_id: str, reason: str) -> Dict[str, Any]:
		"""Cancel an order"""
		order = self.get_sales_order(tenant_id, order_id)
		if not order:
			raise ValueError("Order not found")
		
		try:
			order.cancel_order(user_id, reason)
			
			# Release any inventory allocations
			self._release_inventory_allocations(order)
			
			self.db.commit()
			
			return {
				'success': True,
				'order_number': order.order_number,
				'status': order.status
			}
		except Exception as e:
			self.db.rollback()
			raise e
	
	# Order Retrieval Methods
	
	def get_sales_order(self, tenant_id: str, order_id: str) -> Optional[SOESalesOrder]:
		"""Get sales order by ID"""
		return self.db.query(SOESalesOrder).filter(
			and_(
				SOESalesOrder.tenant_id == tenant_id,
				SOESalesOrder.order_id == order_id
			)
		).first()
	
	def get_sales_order_by_number(self, tenant_id: str, order_number: str) -> Optional[SOESalesOrder]:
		"""Get sales order by order number"""
		return self.db.query(SOESalesOrder).filter(
			and_(
				SOESalesOrder.tenant_id == tenant_id,
				SOESalesOrder.order_number == order_number
			)
		).first()
	
	def search_orders(self, tenant_id: str, filters: Dict[str, Any]) -> List[SOESalesOrder]:
		"""Search orders with filters"""
		query = self.db.query(SOESalesOrder).filter(SOESalesOrder.tenant_id == tenant_id)
		
		if filters.get('customer_id'):
			query = query.filter(SOESalesOrder.customer_id == filters['customer_id'])
		
		if filters.get('status'):
			if isinstance(filters['status'], list):
				query = query.filter(SOESalesOrder.status.in_(filters['status']))
			else:
				query = query.filter(SOESalesOrder.status == filters['status'])
		
		if filters.get('order_date_from'):
			query = query.filter(SOESalesOrder.order_date >= filters['order_date_from'])
		
		if filters.get('order_date_to'):
			query = query.filter(SOESalesOrder.order_date <= filters['order_date_to'])
		
		if filters.get('sales_rep_id'):
			query = query.filter(SOESalesOrder.sales_rep_id == filters['sales_rep_id'])
		
		if filters.get('order_number'):
			query = query.filter(SOESalesOrder.order_number.ilike(f"%{filters['order_number']}%"))
		
		if filters.get('customer_po_number'):
			query = query.filter(SOESalesOrder.customer_po_number.ilike(f"%{filters['customer_po_number']}%"))
		
		# Sorting
		order_by = filters.get('order_by', 'order_date')
		if order_by == 'order_date':
			query = query.order_by(desc(SOESalesOrder.order_date))
		elif order_by == 'order_number':
			query = query.order_by(desc(SOESalesOrder.order_number))
		elif order_by == 'total_amount':
			query = query.order_by(desc(SOESalesOrder.total_amount))
		
		limit = filters.get('limit', 100)
		return query.limit(limit).all()
	
	# Order Template Methods
	
	def create_order_template(self, tenant_id: str, template_data: Dict[str, Any], user_id: str) -> SOEOrderTemplate:
		"""Create an order template"""
		template = SOEOrderTemplate(
			tenant_id=tenant_id,
			template_name=template_data['template_name'],
			description=template_data.get('description'),
			template_type=template_data.get('template_type', 'CUSTOMER'),
			customer_id=template_data.get('customer_id'),
			default_ship_to_id=template_data.get('default_ship_to_id'),
			default_requested_date_offset=template_data.get('default_requested_date_offset', 0),
			is_public=template_data.get('is_public', False),
			notes=template_data.get('notes'),
			created_by_user_id=user_id
		)
		
		self.db.add(template)
		self.db.flush()
		
		# Add template lines
		if 'lines' in template_data:
			for line_num, line_data in enumerate(template_data['lines'], 1):
				template_line = SOEOrderTemplateLine(
					template_id=template.template_id,
					tenant_id=tenant_id,
					line_number=line_num,
					item_id=line_data.get('item_id'),
					item_code=line_data['item_code'],
					item_description=line_data.get('item_description'),
					default_quantity=line_data.get('default_quantity', 1),
					minimum_quantity=line_data.get('minimum_quantity'),
					maximum_quantity=line_data.get('maximum_quantity'),
					is_required=line_data.get('is_required', False),
					allow_quantity_change=line_data.get('allow_quantity_change', True),
					notes=line_data.get('notes'),
					created_by_user_id=user_id
				)
				self.db.add(template_line)
		
		self.db.commit()
		return template
	
	def create_order_from_template(self, tenant_id: str, template_id: str, customer_id: str, user_id: str, 
									modifications: Dict[str, Any] = None) -> SOESalesOrder:
		"""Create an order from a template"""
		template = self.db.query(SOEOrderTemplate).filter(
			and_(
				SOEOrderTemplate.tenant_id == tenant_id,
				SOEOrderTemplate.template_id == template_id,
				SOEOrderTemplate.is_active == True
			)
		).first()
		
		if not template:
			raise ValueError("Template not found or inactive")
		
		# Generate order data from template
		order_data = template.create_order_from_template(customer_id)
		
		# Apply any modifications
		if modifications:
			order_data.update(modifications)
		
		# Create the order
		return self.create_sales_order(tenant_id, order_data, user_id)
	
	# Pricing and Calculation Methods
	
	def _calculate_line_pricing(self, tenant_id: str, customer_id: str, price_level_id: str, 
							   item_info: Dict[str, Any], quantity: Decimal, override_price: Decimal = None) -> Dict[str, Any]:
		"""Calculate pricing for an order line"""
		if override_price:
			return {
				'unit_price': override_price,
				'list_price': item_info.get('list_price', override_price),
				'cost_price': item_info.get('cost_price', 0),
				'discount_percentage': 0
			}
		
		list_price = item_info.get('list_price', 0)
		cost_price = item_info.get('cost_price', 0)
		
		# Apply price level discount
		if price_level_id:
			price_level = self.db.query(SOEPriceLevel).filter(
				and_(
					SOEPriceLevel.tenant_id == tenant_id,
					SOEPriceLevel.price_level_id == price_level_id,
					SOEPriceLevel.is_active == True
				)
			).first()
			
			if price_level and price_level.is_valid_for_date(date.today()):
				unit_price = price_level.calculate_price(list_price, cost_price)
				discount_percentage = ((list_price - unit_price) / list_price * 100) if list_price > 0 else 0
				
				return {
					'unit_price': unit_price,
					'list_price': list_price,
					'cost_price': cost_price,
					'discount_percentage': discount_percentage
				}
		
		# TODO: Add quantity break pricing, contract pricing, promotional pricing
		
		return {
			'unit_price': list_price,
			'list_price': list_price,
			'cost_price': cost_price,
			'discount_percentage': 0
		}
	
	def _calculate_shipping(self, order: SOESalesOrder) -> Optional[SOEOrderCharge]:
		"""Calculate shipping charges for an order"""
		# TODO: Integrate with shipping calculation service
		# This is a simplified implementation
		
		if not order.shipping_method or order.shipping_amount > 0:
			return None
		
		# Basic weight-based calculation
		total_weight = sum(
			(line.quantity_ordered * self._get_item_weight(line.item_id, line.item_code))
			for line in order.lines
		)
		
		if total_weight == 0:
			return None
		
		# Simple rate calculation
		shipping_rate = self._get_shipping_rate(order.shipping_method, total_weight, order.ship_to_address)
		
		if shipping_rate > 0:
			return SOEOrderCharge(
				order_id=order.order_id,
				tenant_id=order.tenant_id,
				charge_type='SHIPPING',
				description=f"Shipping - {order.shipping_method}",
				charge_amount=shipping_rate,
				is_automatic=True,
				is_taxable=False
			)
		
		return None
	
	# Inventory Integration Methods
	
	def _check_inventory_availability(self, order: SOESalesOrder) -> Dict[str, Any]:
		"""Check inventory availability for order lines"""
		# TODO: Integrate with inventory management system
		availability_results = []
		
		for line in order.lines:
			if line.item_type == 'PRODUCT' and not line.drop_ship:
				# Mock inventory check
				available_qty = self._get_available_inventory(
					order.tenant_id, 
					line.item_id, 
					line.item_code, 
					line.warehouse_id
				)
				
				line_result = {
					'line_id': line.line_id,
					'line_number': line.line_number,
					'item_code': line.item_code,
					'quantity_ordered': line.quantity_ordered,
					'quantity_available': available_qty,
					'sufficient_inventory': available_qty >= line.quantity_ordered,
					'shortage': max(0, line.quantity_ordered - available_qty)
				}
				
				if not line_result['sufficient_inventory']:
					if order.customer.allow_backorders:
						line.quantity_backordered = line_result['shortage']
						line.line_status = 'BACKORDERED'
					else:
						order.hold_status = 'INVENTORY_HOLD'
						order.hold_reason = 'Insufficient inventory and backorders not allowed'
				
				availability_results.append(line_result)
		
		return {
			'lines': availability_results,
			'all_available': all(result['sufficient_inventory'] for result in availability_results),
			'has_backorders': any(result['shortage'] > 0 for result in availability_results)
		}
	
	def _apply_automatic_holds(self, order: SOESalesOrder):
		"""Apply automatic holds based on business rules"""
		# Credit hold check
		if not order.hold_status:
			customer_check = order.customer.can_place_order(order.total_amount)
			if not customer_check['can_order']:
				order.hold_status = 'CREDIT_HOLD'
				order.hold_reason = '; '.join(customer_check['issues'])
	
	def _release_inventory_allocations(self, order: SOESalesOrder):
		"""Release inventory allocations for cancelled order"""
		# TODO: Integrate with inventory management system
		for line in order.lines:
			if line.inventory_allocated and line.allocation_id:
				# Mock inventory release
				line.inventory_allocated = False
				line.allocation_id = None
				line.quantity_allocated = 0
	
	# Helper Methods
	
	def _generate_order_number(self, tenant_id: str, order_type: str) -> str:
		"""Generate next order number"""
		sequence = self.db.query(SOEOrderSequence).filter(
			and_(
				SOEOrderSequence.tenant_id == tenant_id,
				SOEOrderSequence.order_type == order_type,
				SOEOrderSequence.is_active == True
			)
		).first()
		
		if sequence:
			return sequence.get_next_number()
		else:
			# Create default sequence if none exists
			sequence = SOEOrderSequence(
				tenant_id=tenant_id,
				sequence_name=f"{order_type} Orders",
				order_type=order_type,
				prefix="SO-",
				current_number=1
			)
			self.db.add(sequence)
			self.db.flush()
			return sequence.get_next_number()
	
	def _get_item_info(self, tenant_id: str, item_id: str = None, item_code: str = None) -> Dict[str, Any]:
		"""Get item information from inventory system"""
		# TODO: Integrate with inventory management system
		# Mock implementation
		return {
			'item_id': item_id,
			'item_code': item_code or 'UNKNOWN',
			'description': f'Item {item_code or item_id}',
			'item_type': 'PRODUCT',
			'unit_of_measure': 'EA',
			'list_price': Decimal('100.00'),
			'cost_price': Decimal('60.00'),
			'weight': Decimal('1.0')
		}
	
	def _get_item_weight(self, item_id: str, item_code: str) -> Decimal:
		"""Get item weight for shipping calculations"""
		# TODO: Get from inventory system
		return Decimal('1.0')  # Default 1 lb
	
	def _get_available_inventory(self, tenant_id: str, item_id: str, item_code: str, warehouse_id: str) -> Decimal:
		"""Get available inventory quantity"""
		# TODO: Integrate with inventory management system
		return Decimal('100.0')  # Mock availability
	
	def _get_tax_rate(self, tenant_id: str, tax_code: str, ship_to_address: SOEShipToAddress = None) -> Decimal:
		"""Get tax rate for tax code and location"""
		# TODO: Integrate with tax calculation system
		# Mock implementation
		tax_rates = {
			'STANDARD': Decimal('8.25'),
			'EXEMPT': Decimal('0.00'),
			'REDUCED': Decimal('4.25')
		}
		return tax_rates.get(tax_code, Decimal('8.25'))
	
	def _get_shipping_rate(self, shipping_method: str, weight: Decimal, ship_to_address: SOEShipToAddress = None) -> Decimal:
		"""Calculate shipping rate"""
		# TODO: Integrate with shipping calculation service
		# Mock implementation
		base_rates = {
			'STANDARD': Decimal('15.00'),
			'EXPRESS': Decimal('25.00'),
			'OVERNIGHT': Decimal('45.00')
		}
		base_rate = base_rates.get(shipping_method, Decimal('15.00'))
		
		# Add weight-based charges
		if weight > 10:
			base_rate += (weight - 10) * Decimal('2.00')
		
		return base_rate
	
	# Reporting and Analytics Methods
	
	def get_order_metrics(self, tenant_id: str, date_range: Tuple[date, date]) -> Dict[str, Any]:
		"""Get order metrics for a date range"""
		start_date, end_date = date_range
		
		base_query = self.db.query(SOESalesOrder).filter(
			and_(
				SOESalesOrder.tenant_id == tenant_id,
				SOESalesOrder.order_date >= start_date,
				SOESalesOrder.order_date <= end_date
			)
		)
		
		total_orders = base_query.count()
		total_amount = base_query.with_entities(func.sum(SOESalesOrder.total_amount)).scalar() or 0
		
		# Orders by status
		status_counts = {}
		status_query = base_query.with_entities(
			SOESalesOrder.status,
			func.count(SOESalesOrder.order_id),
			func.sum(SOESalesOrder.total_amount)
		).group_by(SOESalesOrder.status).all()
		
		for status, count, amount in status_query:
			status_counts[status] = {
				'count': count,
				'amount': amount or 0
			}
		
		# Average order value
		avg_order_value = total_amount / total_orders if total_orders > 0 else 0
		
		return {
			'total_orders': total_orders,
			'total_amount': float(total_amount),
			'average_order_value': float(avg_order_value),
			'orders_by_status': status_counts,
			'date_range': {
				'start_date': start_date.isoformat(),
				'end_date': end_date.isoformat()
			}
		}
	
	def get_customer_order_history(self, tenant_id: str, customer_id: str, limit: int = 50) -> List[SOESalesOrder]:
		"""Get customer order history"""
		return self.db.query(SOESalesOrder).filter(
			and_(
				SOESalesOrder.tenant_id == tenant_id,
				SOESalesOrder.customer_id == customer_id
			)
		).order_by(desc(SOESalesOrder.order_date)).limit(limit).all()
	
	def _log_order_created(self, order: SOESalesOrder, user_id: str):
		"""Log order creation for audit trail"""
		pass  # TODO: Implement audit logging
	
	def _log_order_status_change(self, order: SOESalesOrder, old_status: str, new_status: str, user_id: str):
		"""Log order status changes"""
		pass  # TODO: Implement audit logging