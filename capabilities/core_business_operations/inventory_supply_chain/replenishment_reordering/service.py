"""
Replenishment & Reordering Service

Business logic for automated replenishment, purchase order generation,
supplier management, and demand forecasting.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import sessionmaker, Session
import json

from ....auth_rbac.models import get_session
from .models import (
	IMRRSupplier, IMRRSupplierItem, IMRRReplenishmentRule,
	IMRRReplenishmentSuggestion, IMRRPurchaseOrder, 
	IMRRPurchaseOrderLine, IMRRDemandForecast
)


class ReplenishmentService:
	"""Service for replenishment and reordering operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.session = get_session()
	
	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type:
			self.session.rollback()
		else:
			self.session.commit()
		self.session.close()
	
	# Supplier Management
	
	def create_supplier(self, supplier_data: Dict[str, Any]) -> IMRRSupplier:
		"""Create a new supplier"""
		
		required_fields = ['supplier_code', 'supplier_name']
		for field in required_fields:
			if field not in supplier_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate supplier code
		existing_supplier = self.session.query(IMRRSupplier).filter(
			and_(
				IMRRSupplier.tenant_id == self.tenant_id,
				IMRRSupplier.supplier_code == supplier_data['supplier_code']
			)
		).first()
		
		if existing_supplier:
			raise ValueError(f"Supplier code '{supplier_data['supplier_code']}' already exists")
		
		supplier = IMRRSupplier(
			tenant_id=self.tenant_id,
			**supplier_data
		)
		
		self.session.add(supplier)
		self.session.flush()
		
		self._log_supplier_creation(supplier)
		return supplier
	
	def get_supplier_by_id(self, supplier_id: str) -> Optional[IMRRSupplier]:
		"""Get supplier by ID"""
		return self.session.query(IMRRSupplier).filter(
			and_(
				IMRRSupplier.tenant_id == self.tenant_id,
				IMRRSupplier.supplier_id == supplier_id
			)
		).first()
	
	# Replenishment Rules
	
	def create_replenishment_rule(self, rule_data: Dict[str, Any]) -> IMRRReplenishmentRule:
		"""Create a new replenishment rule"""
		
		required_fields = ['rule_name', 'rule_type']
		for field in required_fields:
			if field not in rule_data:
				raise ValueError(f"Missing required field: {field}")
		
		rule = IMRRReplenishmentRule(
			tenant_id=self.tenant_id,
			**rule_data
		)
		
		self.session.add(rule)
		self.session.flush()
		
		self._log_rule_creation(rule)
		return rule
	
	def run_replenishment_analysis(self, rule_id: str = None) -> List[IMRRReplenishmentSuggestion]:
		"""Run replenishment analysis and generate suggestions"""
		
		# Get active rules
		query = self.session.query(IMRRReplenishmentRule).filter(
			and_(
				IMRRReplenishmentRule.tenant_id == self.tenant_id,
				IMRRReplenishmentRule.is_active == True
			)
		)
		
		if rule_id:
			query = query.filter(IMRRReplenishmentRule.rule_id == rule_id)
		
		rules = query.all()
		suggestions = []
		
		for rule in rules:
			rule_suggestions = self._generate_suggestions_for_rule(rule)
			suggestions.extend(rule_suggestions)
			
			# Update last run date
			rule.last_run_date = datetime.utcnow()
			rule.next_run_date = datetime.utcnow() + timedelta(hours=rule.run_frequency_hours)
		
		return suggestions
	
	# Purchase Orders
	
	def create_purchase_order(self, po_data: Dict[str, Any]) -> IMRRPurchaseOrder:
		"""Create a new purchase order"""
		
		required_fields = ['po_number', 'supplier_id', 'warehouse_id']
		for field in required_fields:
			if field not in po_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate PO number
		existing_po = self.session.query(IMRRPurchaseOrder).filter(
			and_(
				IMRRPurchaseOrder.tenant_id == self.tenant_id,
				IMRRPurchaseOrder.po_number == po_data['po_number']
			)
		).first()
		
		if existing_po:
			raise ValueError(f"PO number '{po_data['po_number']}' already exists")
		
		po = IMRRPurchaseOrder(
			tenant_id=self.tenant_id,
			**po_data
		)
		
		self.session.add(po)
		self.session.flush()
		
		self._log_po_creation(po)
		return po
	
	def convert_suggestion_to_po(self, suggestion_id: str, po_data: Dict[str, Any] = None) -> IMRRPurchaseOrder:
		"""Convert replenishment suggestion to purchase order"""
		
		suggestion = self.session.query(IMRRReplenishmentSuggestion).filter(
			and_(
				IMRRReplenishmentSuggestion.tenant_id == self.tenant_id,
				IMRRReplenishmentSuggestion.suggestion_id == suggestion_id
			)
		).first()
		
		if not suggestion:
			raise ValueError(f"Suggestion {suggestion_id} not found")
		
		if suggestion.status != 'Approved':
			raise ValueError(f"Suggestion must be approved before conversion")
		
		# Generate PO number if not provided
		if not po_data:
			po_data = {}
		
		if 'po_number' not in po_data:
			po_data['po_number'] = self._generate_po_number()
		
		# Create PO
		po = self.create_purchase_order({
			**po_data,
			'supplier_id': suggestion.recommended_supplier_id,
			'warehouse_id': suggestion.warehouse_id,
			'created_from': 'Replenishment',
			'source_suggestion_id': suggestion_id
		})
		
		# Create PO line
		po_line = IMRRPurchaseOrderLine(
			tenant_id=self.tenant_id,
			po_id=po.po_id,
			line_number=1,
			item_id=suggestion.item_id,
			ordered_quantity=suggestion.suggested_quantity,
			unit_price=suggestion.unit_cost or Decimal('0'),
			line_total=suggestion.total_cost or Decimal('0'),
			requested_delivery_date=date.today() + timedelta(days=suggestion.lead_time_days),
			source_suggestion_id=suggestion_id
		)
		
		self.session.add(po_line)
		
		# Update suggestion status
		suggestion.status = 'Converted'
		suggestion.po_id = po.po_id
		suggestion.conversion_date = datetime.utcnow()
		
		return po
	
	# Dashboard Methods
	
	def get_pending_purchase_orders_count(self) -> int:
		"""Get count of pending purchase orders"""
		return self.session.query(IMRRPurchaseOrder).filter(
			and_(
				IMRRPurchaseOrder.tenant_id == self.tenant_id,
				IMRRPurchaseOrder.status.in_(['Draft', 'Sent', 'Acknowledged'])
			)
		).count()
	
	def get_auto_reorder_items_count(self) -> int:
		"""Get count of items with auto-reorder rules"""
		return self.session.query(IMRRReplenishmentRule).filter(
			and_(
				IMRRReplenishmentRule.tenant_id == self.tenant_id,
				IMRRReplenishmentRule.is_active == True,
				IMRRReplenishmentRule.auto_generate_po == True
			)
		).count()
	
	def get_overdue_orders_count(self) -> int:
		"""Get count of overdue purchase orders"""
		return self.session.query(IMRRPurchaseOrder).filter(
			and_(
				IMRRPurchaseOrder.tenant_id == self.tenant_id,
				IMRRPurchaseOrder.requested_delivery_date < date.today(),
				IMRRPurchaseOrder.status.in_(['Sent', 'Acknowledged'])
			)
		).count()
	
	# Private Helper Methods
	
	def _generate_suggestions_for_rule(self, rule: IMRRReplenishmentRule) -> List[IMRRReplenishmentSuggestion]:
		"""Generate replenishment suggestions for a specific rule"""
		
		suggestions = []
		
		# This is a simplified implementation
		# In practice, this would involve complex logic for:
		# - Calculating current stock levels
		# - Analyzing demand patterns
		# - Determining optimal order quantities
		# - Selecting best suppliers
		
		# For now, just create a placeholder suggestion
		if rule.item_id:  # Rule for specific item
			suggestion = IMRRReplenishmentSuggestion(
				tenant_id=self.tenant_id,
				rule_id=rule.rule_id,
				item_id=rule.item_id,
				warehouse_id=rule.warehouse_id or 'default_warehouse',
				current_stock=Decimal('10'),
				available_stock=Decimal('10'),
				reorder_point=rule.reorder_point or Decimal('20'),
				suggested_quantity=rule.reorder_quantity or Decimal('100'),
				recommended_supplier_id=rule.preferred_supplier_id,
				priority_level='Medium',
				reason_code='Low Stock'
			)
			
			self.session.add(suggestion)
			suggestions.append(suggestion)
		
		return suggestions
	
	def _generate_po_number(self) -> str:
		"""Generate unique PO number"""
		today = date.today()
		prefix = f"PO{today.year}{today.month:02d}"
		
		# Get next sequence number
		last_po = self.session.query(IMRRPurchaseOrder).filter(
			and_(
				IMRRPurchaseOrder.tenant_id == self.tenant_id,
				IMRRPurchaseOrder.po_number.like(f"{prefix}%")
			)
		).order_by(IMRRPurchaseOrder.po_number.desc()).first()
		
		if last_po:
			try:
				last_seq = int(last_po.po_number[-4:])
				next_seq = last_seq + 1
			except:
				next_seq = 1
		else:
			next_seq = 1
		
		return f"{prefix}{next_seq:04d}"
	
	# Logging Methods
	
	def _log_supplier_creation(self, supplier: IMRRSupplier):
		"""Log supplier creation"""
		print(f"Supplier created: {supplier.supplier_code} - {supplier.supplier_name}")
	
	def _log_rule_creation(self, rule: IMRRReplenishmentRule):
		"""Log replenishment rule creation"""
		print(f"Replenishment rule created: {rule.rule_name}")
	
	def _log_po_creation(self, po: IMRRPurchaseOrder):
		"""Log purchase order creation"""
		print(f"Purchase order created: {po.po_number}")