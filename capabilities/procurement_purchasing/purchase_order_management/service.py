"""
Purchase Order Management Service

Business logic for PO creation, receipt processing, and three-way matching.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from .models import PPOPurchaseOrder, PPOPurchaseOrderLine, PPOReceipt, PPOReceiptLine, PPOThreeWayMatch, PPOChangeOrder
from ...auth_rbac.models import get_db_session


class PurchaseOrderService:
	"""Service class for purchase order management operations"""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.db: Session = get_db_session()
	
	def create_purchase_order(self, po_data: Dict[str, Any], user_id: str) -> PPOPurchaseOrder:
		"""Create a new purchase order"""
		
		# Generate PO number
		po_number = self._generate_po_number()
		
		# Create PO
		po = PPOPurchaseOrder(
			tenant_id=self.tenant_id,
			po_number=po_number,
			title=po_data['title'],
			description=po_data.get('description'),
			vendor_id=po_data['vendor_id'],
			vendor_name=po_data['vendor_name'],
			buyer_id=user_id,
			buyer_name=po_data['buyer_name'],
			department=po_data.get('department'),
			required_date=po_data['required_date'],
			ship_to_location=po_data.get('ship_to_location'),
			payment_terms=po_data.get('payment_terms'),
			special_instructions=po_data.get('special_instructions'),
			notes=po_data.get('notes')
		)
		
		self.db.add(po)
		self.db.flush()
		
		# Add PO lines
		if 'lines' in po_data:
			for line_data in po_data['lines']:
				self._create_po_line(po, line_data)
		
		# Calculate totals
		po.calculate_totals()
		self.db.commit()
		
		return po
	
	def get_purchase_orders_by_status(self, status: str, limit: int = 100) -> List[PPOPurchaseOrder]:
		"""Get purchase orders by status"""
		return self.db.query(PPOPurchaseOrder).filter(
			and_(
				PPOPurchaseOrder.tenant_id == self.tenant_id,
				PPOPurchaseOrder.status == status
			)
		).order_by(desc(PPOPurchaseOrder.po_date)).limit(limit).all()
	
	def get_purchase_orders_needing_receipt(self) -> List[PPOPurchaseOrder]:
		"""Get POs that need receipt processing"""
		return self.db.query(PPOPurchaseOrder).filter(
			and_(
				PPOPurchaseOrder.tenant_id == self.tenant_id,
				PPOPurchaseOrder.status == 'Open',
				PPOPurchaseOrder.received_amount < PPOPurchaseOrder.total_amount
			)
		).all()
	
	def get_total_po_value_ytd(self) -> Decimal:
		"""Get total PO value year-to-date"""
		from sqlalchemy import func, extract
		
		current_year = datetime.now().year
		
		result = self.db.query(func.sum(PPOPurchaseOrder.total_amount)).filter(
			and_(
				PPOPurchaseOrder.tenant_id == self.tenant_id,
				extract('year', PPOPurchaseOrder.po_date) == current_year,
				PPOPurchaseOrder.status.in_(['Approved', 'Open', 'Closed'])
			)
		).scalar()
		
		return result or Decimal('0.00')
	
	def get_avg_processing_time(self) -> float:
		"""Get average PO processing time in hours"""
		# Placeholder implementation
		return 24.0
	
	def get_overdue_receipts(self) -> List[PPOPurchaseOrder]:
		"""Get POs with overdue receipts"""
		from datetime import timedelta
		
		cutoff_date = date.today() - timedelta(days=7)
		
		return self.db.query(PPOPurchaseOrder).filter(
			and_(
				PPOPurchaseOrder.tenant_id == self.tenant_id,
				PPOPurchaseOrder.status == 'Open',
				PPOPurchaseOrder.required_date < cutoff_date,
				PPOPurchaseOrder.received_amount < PPOPurchaseOrder.total_amount
			)
		).all()
	
	def _create_po_line(self, po: PPOPurchaseOrder, line_data: Dict[str, Any]) -> PPOPurchaseOrderLine:
		"""Create a PO line"""
		
		line = PPOPurchaseOrderLine(
			po_id=po.po_id,
			tenant_id=self.tenant_id,
			line_number=line_data['line_number'],
			description=line_data['description'],
			item_code=line_data.get('item_code'),
			item_description=line_data.get('item_description'),
			quantity_ordered=line_data.get('quantity_ordered', 1),
			unit_of_measure=line_data.get('unit_of_measure', 'EA'),
			unit_price=line_data.get('unit_price', 0),
			gl_account_id=line_data['gl_account_id'],
			cost_center=line_data.get('cost_center'),
			project_id=line_data.get('project_id'),
			required_date=line_data.get('required_date')
		)
		
		# Calculate line amount
		line.line_amount = line.quantity_ordered * line.unit_price
		
		po.lines.append(line)
		return line
	
	def _generate_po_number(self) -> str:
		"""Generate unique PO number"""
		now = datetime.now()
		year = now.year
		month = now.month
		
		prefix = f"PO-{year:04d}{month:02d}-"
		
		latest = self.db.query(PPOPurchaseOrder).filter(
			and_(
				PPOPurchaseOrder.tenant_id == self.tenant_id,
				PPOPurchaseOrder.po_number.like(f"{prefix}%")
			)
		).order_by(desc(PPOPurchaseOrder.po_number)).first()
		
		if latest:
			sequence = int(latest.po_number.split('-')[-1]) + 1
		else:
			sequence = 1
		
		return f"{prefix}{sequence:04d}"
	
	def __del__(self):
		"""Cleanup database session"""
		if hasattr(self, 'db'):
			self.db.close()
