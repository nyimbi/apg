"""
Expiry Date Management Service

Business logic for expiry tracking, FEFO management,
waste minimization, and compliance monitoring.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import and_, or_, func, text
import json

from ....auth_rbac.models import get_session
from .models import (
	IMEDMExpiryPolicy, IMEDMExpiryItem, IMEDMExpiryMovement,
	IMEDMDisposition, IMEDMExpiryAlert, IMEDMWasteReport
)


class ExpiryDateService:
	"""Service for expiry date management operations"""
	
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
	
	# Expiry Item Management
	
	def create_expiry_item(self, item_data: Dict[str, Any]) -> IMEDMExpiryItem:
		"""Create a new expiry-tracked item"""
		
		required_fields = ['item_id', 'warehouse_id', 'location_id', 'expiry_date', 'original_quantity']
		for field in required_fields:
			if field not in item_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Set current and available quantities if not specified
		if 'current_quantity' not in item_data:
			item_data['current_quantity'] = item_data['original_quantity']
		if 'available_quantity' not in item_data:
			item_data['available_quantity'] = item_data['original_quantity']
		
		expiry_item = IMEDMExpiryItem(
			tenant_id=self.tenant_id,
			**item_data
		)
		
		self.session.add(expiry_item)
		self.session.flush()
		
		# Set initial alert level
		self._update_alert_level(expiry_item)
		
		self._log_expiry_item_creation(expiry_item)
		return expiry_item
	
	def get_items_expiring_soon(self, days: int = 30, limit: int = 100) -> List[IMEDMExpiryItem]:
		"""Get items expiring within specified days"""
		
		cutoff_date = date.today() + timedelta(days=days)
		
		return self.session.query(IMEDMExpiryItem).filter(
			and_(
				IMEDMExpiryItem.tenant_id == self.tenant_id,
				IMEDMExpiryItem.expiry_date <= cutoff_date,
				IMEDMExpiryItem.expiry_status == 'Active',
				IMEDMExpiryItem.current_quantity > 0
			)
		).order_by(IMEDMExpiryItem.expiry_date).limit(limit).all()
	
	def get_expired_items(self, limit: int = 100) -> List[IMEDMExpiryItem]:
		"""Get expired items"""
		
		return self.session.query(IMEDMExpiryItem).filter(
			and_(
				IMEDMExpiryItem.tenant_id == self.tenant_id,
				IMEDMExpiryItem.expiry_date < date.today(),
				IMEDMExpiryItem.expiry_status.in_(['Active', 'Expiring']),
				IMEDMExpiryItem.current_quantity > 0
			)
		).order_by(IMEDMExpiryItem.expiry_date).limit(limit).all()
	
	# FEFO Management
	
	def get_fefo_sequence(self, item_id: str, warehouse_id: str = None) -> List[IMEDMExpiryItem]:
		"""Get items in FEFO (First Expired First Out) sequence"""
		
		query = self.session.query(IMEDMExpiryItem).filter(
			and_(
				IMEDMExpiryItem.tenant_id == self.tenant_id,
				IMEDMExpiryItem.item_id == item_id,
				IMEDMExpiryItem.available_quantity > 0,
				IMEDMExpiryItem.quality_status == 'Good'
			)
		)
		
		if warehouse_id:
			query = query.filter(IMEDMExpiryItem.warehouse_id == warehouse_id)
		
		return query.order_by(IMEDMExpiryItem.expiry_date).all()
	
	def validate_fefo_compliance(self, item_id: str, selected_batch: str, 
								 requested_quantity: Decimal) -> Dict[str, Any]:
		"""Validate if selection follows FEFO rules"""
		
		fefo_sequence = self.get_fefo_sequence(item_id)
		
		if not fefo_sequence:
			return {'compliant': True, 'reason': 'No items available'}
		
		# Find the selected batch in the sequence
		selected_item = None
		selected_index = -1
		
		for i, expiry_item in enumerate(fefo_sequence):
			if expiry_item.batch_number == selected_batch:
				selected_item = expiry_item
				selected_index = i
				break
		
		if not selected_item:
			return {'compliant': False, 'reason': 'Selected batch not found'}
		
		# Check if there are items with earlier expiry dates that have sufficient quantity
		for i, expiry_item in enumerate(fefo_sequence[:selected_index]):
			if expiry_item.available_quantity >= requested_quantity:
				return {
					'compliant': False,
					'reason': f'Batch {expiry_item.batch_number} expires earlier and has sufficient quantity',
					'recommended_batch': expiry_item.batch_number,
					'recommended_expiry': expiry_item.expiry_date.isoformat()
				}
		
		return {'compliant': True, 'reason': 'FEFO compliant'}
	
	# Disposition Management
	
	def create_disposition(self, disposition_data: Dict[str, Any]) -> IMEDMDisposition:
		"""Create a disposition record for expired items"""
		
		required_fields = ['expiry_item_id', 'disposition_type', 'disposition_reason', 'quantity_disposed']
		for field in required_fields:
			if field not in disposition_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Generate disposition number if not provided
		if 'disposition_number' not in disposition_data:
			disposition_data['disposition_number'] = self._generate_disposition_number()
		
		disposition = IMEDMDisposition(
			tenant_id=self.tenant_id,
			**disposition_data
		)
		
		self.session.add(disposition)
		self.session.flush()
		
		# Update expiry item status
		expiry_item = self.session.query(IMEDMExpiryItem).filter(
			IMEDMExpiryItem.expiry_item_id == disposition_data['expiry_item_id']
		).first()
		
		if expiry_item:
			expiry_item.disposition_status = disposition_data['disposition_type']
			if disposition.quantity_disposed >= expiry_item.current_quantity:
				expiry_item.expiry_status = 'Disposed'
		
		self._log_disposition_creation(disposition)
		return disposition
	
	# Alert Management
	
	def generate_expiry_alerts(self) -> List[IMEDMExpiryAlert]:
		"""Generate alerts for expiring and expired items"""
		
		alerts = []
		
		# Get items expiring in different timeframes
		notification_items = self.get_items_expiring_soon(days=60)
		warning_items = self.get_items_expiring_soon(days=30)
		critical_items = self.get_items_expiring_soon(days=7)
		expired_items = self.get_expired_items()
		
		# Generate alerts for each category
		for item in notification_items:
			if not self._alert_exists(item, 'Expiring', 'Notification'):
				alert = self._create_alert(item, 'Expiring', 'Notification')
				alerts.append(alert)
		
		for item in warning_items:
			if not self._alert_exists(item, 'Expiring', 'Warning'):
				alert = self._create_alert(item, 'Expiring', 'Warning')
				alerts.append(alert)
		
		for item in critical_items:
			if not self._alert_exists(item, 'Expiring', 'Critical'):
				alert = self._create_alert(item, 'Expiring', 'Critical')
				alerts.append(alert)
		
		for item in expired_items:
			if not self._alert_exists(item, 'Expired', 'Critical'):
				alert = self._create_alert(item, 'Expired', 'Critical')
				alerts.append(alert)
		
		return alerts
	
	# Dashboard Methods
	
	def get_items_expiring_soon_count(self, days: int = 30) -> int:
		"""Get count of items expiring soon"""
		cutoff_date = date.today() + timedelta(days=days)
		
		return self.session.query(IMEDMExpiryItem).filter(
			and_(
				IMEDMExpiryItem.tenant_id == self.tenant_id,
				IMEDMExpiryItem.expiry_date <= cutoff_date,
				IMEDMExpiryItem.expiry_date >= date.today(),
				IMEDMExpiryItem.expiry_status == 'Active',
				IMEDMExpiryItem.current_quantity > 0
			)
		).count()
	
	def get_expired_items_count(self) -> int:
		"""Get count of expired items"""
		return self.session.query(IMEDMExpiryItem).filter(
			and_(
				IMEDMExpiryItem.tenant_id == self.tenant_id,
				IMEDMExpiryItem.expiry_date < date.today(),
				IMEDMExpiryItem.expiry_status.in_(['Active', 'Expiring']),
				IMEDMExpiryItem.current_quantity > 0
			)
		).count()
	
	def get_waste_value_current_month(self) -> Decimal:
		"""Get waste value for current month"""
		first_day = date.today().replace(day=1)
		
		result = self.session.query(func.sum(IMEDMDisposition.net_loss)).filter(
			and_(
				IMEDMDisposition.tenant_id == self.tenant_id,
				IMEDMDisposition.disposition_date >= first_day,
				IMEDMDisposition.status == 'Completed'
			)
		).scalar()
		
		return result if result else Decimal('0')
	
	# Private Helper Methods
	
	def _update_alert_level(self, expiry_item: IMEDMExpiryItem):
		"""Update alert level based on days to expiry"""
		days_to_expiry = expiry_item.days_to_expiry
		
		if days_to_expiry < 0:
			expiry_item.alert_level = 'Critical'
			expiry_item.expiry_status = 'Expired'
		elif days_to_expiry <= 7:
			expiry_item.alert_level = 'Critical'
			expiry_item.expiry_status = 'Expiring'
		elif days_to_expiry <= 30:
			expiry_item.alert_level = 'Warning'
			expiry_item.expiry_status = 'Expiring'
		elif days_to_expiry <= 60:
			expiry_item.alert_level = 'Notification'
		else:
			expiry_item.alert_level = 'None'
	
	def _alert_exists(self, expiry_item: IMEDMExpiryItem, alert_type: str, alert_level: str) -> bool:
		"""Check if alert already exists for item"""
		return self.session.query(IMEDMExpiryAlert).filter(
			and_(
				IMEDMExpiryAlert.tenant_id == self.tenant_id,
				IMEDMExpiryAlert.expiry_item_id == expiry_item.expiry_item_id,
				IMEDMExpiryAlert.alert_type == alert_type,
				IMEDMExpiryAlert.alert_level == alert_level,
				IMEDMExpiryAlert.status == 'Active'
			)
		).first() is not None
	
	def _create_alert(self, expiry_item: IMEDMExpiryItem, alert_type: str, alert_level: str) -> IMEDMExpiryAlert:
		"""Create an expiry alert"""
		alert = IMEDMExpiryAlert(
			tenant_id=self.tenant_id,
			expiry_item_id=expiry_item.expiry_item_id,
			item_id=expiry_item.item_id,
			warehouse_id=expiry_item.warehouse_id,
			alert_type=alert_type,
			alert_level=alert_level,
			alert_title=f"{alert_type} Item: Batch {expiry_item.batch_number}",
			alert_message=f"Item {expiry_item.item_id} batch {expiry_item.batch_number} is {alert_type.lower()} on {expiry_item.expiry_date}",
			expiry_date=expiry_item.expiry_date,
			days_to_expiry=expiry_item.days_to_expiry,
			quantity_affected=expiry_item.current_quantity,
			value_at_risk=expiry_item.current_quantity * expiry_item.unit_cost
		)
		
		self.session.add(alert)
		return alert
	
	def _generate_disposition_number(self) -> str:
		"""Generate unique disposition number"""
		today = date.today()
		prefix = f"DISP{today.year}{today.month:02d}"
		
		# Get next sequence number
		last_disposition = self.session.query(IMEDMDisposition).filter(
			and_(
				IMEDMDisposition.tenant_id == self.tenant_id,
				IMEDMDisposition.disposition_number.like(f"{prefix}%")
			)
		).order_by(IMEDMDisposition.disposition_number.desc()).first()
		
		if last_disposition:
			try:
				last_seq = int(last_disposition.disposition_number[-4:])
				next_seq = last_seq + 1
			except:
				next_seq = 1
		else:
			next_seq = 1
		
		return f"{prefix}{next_seq:04d}"
	
	# Logging Methods
	
	def _log_expiry_item_creation(self, expiry_item: IMEDMExpiryItem):
		"""Log expiry item creation"""
		print(f"Expiry item created: {expiry_item.item_id}-{expiry_item.batch_number} expires {expiry_item.expiry_date}")
	
	def _log_disposition_creation(self, disposition: IMEDMDisposition):
		"""Log disposition creation"""
		print(f"Disposition created: {disposition.disposition_number} - {disposition.disposition_type}")