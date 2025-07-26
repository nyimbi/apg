"""
Stock Tracking & Control Service

Business logic for inventory level monitoring, location tracking,
movement control, and real-time stock management.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import sessionmaker, Session
import json

from ....auth_rbac.models import get_session
from .models import (
	IMSTCItem, IMSTCItemCategory, IMSTCUnitOfMeasure,
	IMSTCWarehouse, IMSTCLocation, IMSTCStockLevel,
	IMSTCStockMovement, IMSTCCycleCount, IMSTCCycleCountLine,
	IMSTCStockAlert
)


class StockTrackingService:
	"""Service for stock tracking and control operations"""
	
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
	
	# Item Management
	
	def create_item(self, item_data: Dict[str, Any]) -> IMSTCItem:
		"""Create a new inventory item"""
		
		# Validate required fields
		required_fields = ['item_code', 'item_name', 'primary_uom_id']
		for field in required_fields:
			if field not in item_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate item code
		existing_item = self.session.query(IMSTCItem).filter(
			and_(
				IMSTCItem.tenant_id == self.tenant_id,
				IMSTCItem.item_code == item_data['item_code']
			)
		).first()
		
		if existing_item:
			raise ValueError(f"Item code '{item_data['item_code']}' already exists")
		
		# Create the item
		item = IMSTCItem(
			tenant_id=self.tenant_id,
			**item_data
		)
		
		self.session.add(item)
		self.session.flush()  # Get the ID
		
		# Create initial stock level if default warehouse specified
		if item.default_warehouse_id and item.default_location_id:
			self._create_initial_stock_level(item)
		
		self._log_item_creation(item)
		return item
	
	def update_item(self, item_id: str, updates: Dict[str, Any]) -> IMSTCItem:
		"""Update an existing inventory item"""
		
		item = self.get_item_by_id(item_id)
		if not item:
			raise ValueError(f"Item with ID {item_id} not found")
		
		# Update fields
		for field, value in updates.items():
			if hasattr(item, field):
				setattr(item, field, value)
		
		item.updated_at = datetime.utcnow()
		self._log_item_update(item, updates)
		return item
	
	def get_item_by_id(self, item_id: str) -> Optional[IMSTCItem]:
		"""Get item by ID"""
		return self.session.query(IMSTCItem).filter(
			and_(
				IMSTCItem.tenant_id == self.tenant_id,
				IMSTCItem.item_id == item_id
			)
		).first()
	
	def get_item_by_code(self, item_code: str) -> Optional[IMSTCItem]:
		"""Get item by code"""
		return self.session.query(IMSTCItem).filter(
			and_(
				IMSTCItem.tenant_id == self.tenant_id,
				IMSTCItem.item_code == item_code
			)
		).first()
	
	def search_items(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[IMSTCItem]:
		"""Search items with filters"""
		
		query = self.session.query(IMSTCItem).filter(
			IMSTCItem.tenant_id == self.tenant_id
		)
		
		# Apply filters
		if 'search_text' in filters:
			search_text = f"%{filters['search_text']}%"
			query = query.filter(
				or_(
					IMSTCItem.item_code.ilike(search_text),
					IMSTCItem.item_name.ilike(search_text),
					IMSTCItem.description.ilike(search_text)
				)
			)
		
		if 'category_id' in filters:
			query = query.filter(IMSTCItem.category_id == filters['category_id'])
		
		if 'item_type' in filters:
			query = query.filter(IMSTCItem.item_type == filters['item_type'])
		
		if 'is_active' in filters:
			query = query.filter(IMSTCItem.is_active == filters['is_active'])
		
		if 'abc_classification' in filters:
			query = query.filter(IMSTCItem.abc_classification == filters['abc_classification'])
		
		return query.order_by(IMSTCItem.item_code).offset(offset).limit(limit).all()
	
	# Warehouse and Location Management
	
	def create_warehouse(self, warehouse_data: Dict[str, Any]) -> IMSTCWarehouse:
		"""Create a new warehouse"""
		
		required_fields = ['warehouse_code', 'warehouse_name']
		for field in required_fields:
			if field not in warehouse_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate warehouse code
		existing_warehouse = self.session.query(IMSTCWarehouse).filter(
			and_(
				IMSTCWarehouse.tenant_id == self.tenant_id,
				IMSTCWarehouse.warehouse_code == warehouse_data['warehouse_code']
			)
		).first()
		
		if existing_warehouse:
			raise ValueError(f"Warehouse code '{warehouse_data['warehouse_code']}' already exists")
		
		warehouse = IMSTCWarehouse(
			tenant_id=self.tenant_id,
			**warehouse_data
		)
		
		self.session.add(warehouse)
		self.session.flush()
		
		self._log_warehouse_creation(warehouse)
		return warehouse
	
	def create_location(self, location_data: Dict[str, Any]) -> IMSTCLocation:
		"""Create a new storage location"""
		
		required_fields = ['location_code', 'location_name', 'warehouse_id']
		for field in required_fields:
			if field not in location_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check for duplicate location code within warehouse
		existing_location = self.session.query(IMSTCLocation).filter(
			and_(
				IMSTCLocation.tenant_id == self.tenant_id,
				IMSTCLocation.warehouse_id == location_data['warehouse_id'],
				IMSTCLocation.location_code == location_data['location_code']
			)
		).first()
		
		if existing_location:
			raise ValueError(f"Location code '{location_data['location_code']}' already exists in this warehouse")
		
		# Set hierarchy level based on parent
		if location_data.get('parent_location_id'):
			parent = self.session.query(IMSTCLocation).filter(
				IMSTCLocation.location_id == location_data['parent_location_id']
			).first()
			if parent:
				location_data['level'] = parent.level + 1
				location_data['path'] = f"{parent.path}/{location_data['location_code']}" if parent.path else location_data['location_code']
		else:
			location_data['level'] = 0
			location_data['path'] = location_data['location_code']
		
		location = IMSTCLocation(
			tenant_id=self.tenant_id,
			**location_data
		)
		
		self.session.add(location)
		self.session.flush()
		
		self._log_location_creation(location)
		return location
	
	# Stock Level Management
	
	def get_current_stock(self, item_id: str, warehouse_id: str = None, location_id: str = None) -> List[IMSTCStockLevel]:
		"""Get current stock levels for an item"""
		
		query = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id,
				IMSTCStockLevel.on_hand_quantity > 0
			)
		)
		
		if warehouse_id:
			query = query.filter(IMSTCStockLevel.warehouse_id == warehouse_id)
		
		if location_id:
			query = query.filter(IMSTCStockLevel.location_id == location_id)
		
		return query.all()
	
	def get_total_on_hand(self, item_id: str, warehouse_id: str = None) -> Decimal:
		"""Get total on-hand quantity for an item"""
		
		query = self.session.query(func.sum(IMSTCStockLevel.on_hand_quantity)).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id
			)
		)
		
		if warehouse_id:
			query = query.filter(IMSTCStockLevel.warehouse_id == warehouse_id)
		
		result = query.scalar()
		return result if result else Decimal('0')
	
	def get_available_stock(self, item_id: str, warehouse_id: str = None) -> Decimal:
		"""Get available (on-hand minus allocated) quantity for an item"""
		
		query = self.session.query(func.sum(IMSTCStockLevel.available_quantity)).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id
			)
		)
		
		if warehouse_id:
			query = query.filter(IMSTCStockLevel.warehouse_id == warehouse_id)
		
		result = query.scalar()
		return result if result else Decimal('0')
	
	def get_low_stock_items(self, warehouse_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
		"""Get items that are below their reorder point"""
		
		# Complex query to get items where available stock is below reorder point
		query = text("""
			SELECT 
				i.item_id,
				i.item_code,
				i.item_name,
				i.reorder_point,
				i.min_stock_level,
				COALESCE(SUM(sl.available_quantity), 0) as current_stock,
				w.warehouse_name
			FROM im_stc_item i
			LEFT JOIN im_stc_stock_level sl ON i.item_id = sl.item_id
			LEFT JOIN im_stc_warehouse w ON sl.warehouse_id = w.warehouse_id
			WHERE i.tenant_id = :tenant_id
			AND i.is_active = true
			AND COALESCE(SUM(sl.available_quantity), 0) <= i.reorder_point
			{}
			GROUP BY i.item_id, i.item_code, i.item_name, i.reorder_point, i.min_stock_level, w.warehouse_name
			ORDER BY (COALESCE(SUM(sl.available_quantity), 0) / GREATEST(i.reorder_point, 1)) ASC
			LIMIT :limit
		""".format(
			"AND sl.warehouse_id = :warehouse_id" if warehouse_id else ""
		))
		
		params = {
			'tenant_id': self.tenant_id,
			'limit': limit
		}
		if warehouse_id:
			params['warehouse_id'] = warehouse_id
		
		result = self.session.execute(query, params)
		return [dict(row) for row in result]
	
	# Stock Movement Operations
	
	def receive_stock(self, receipt_data: Dict[str, Any]) -> IMSTCStockMovement:
		"""Receive stock into inventory"""
		
		required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity', 'unit_cost']
		for field in required_fields:
			if field not in receipt_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Create the movement record
		movement = IMSTCStockMovement(
			tenant_id=self.tenant_id,
			movement_type='Receipt',
			movement_subtype=receipt_data.get('movement_subtype', 'Purchase Receipt'),
			reference_number=receipt_data.get('reference_number'),
			reference_type=receipt_data.get('reference_type'),
			item_id=receipt_data['item_id'],
			warehouse_id=receipt_data['warehouse_id'],
			location_id=receipt_data['location_id'],
			uom_id=receipt_data.get('uom_id'),
			quantity=receipt_data['quantity'],
			unit_cost=receipt_data['unit_cost'],
			total_cost=receipt_data['quantity'] * receipt_data['unit_cost'],
			lot_number=receipt_data.get('lot_number'),
			serial_number=receipt_data.get('serial_number'),
			batch_number=receipt_data.get('batch_number'),
			expiry_date=receipt_data.get('expiry_date'),
			stock_status=receipt_data.get('stock_status', 'Available'),
			quality_status=receipt_data.get('quality_status', 'Approved'),
			user_id=receipt_data.get('user_id'),
			notes=receipt_data.get('notes')
		)
		
		self.session.add(movement)
		self.session.flush()
		
		# Update stock levels
		self._update_stock_level_for_receipt(movement)
		
		# Check for alerts
		self._check_stock_alerts(receipt_data['item_id'])
		
		self._log_stock_receipt(movement)
		return movement
	
	def issue_stock(self, issue_data: Dict[str, Any]) -> IMSTCStockMovement:
		"""Issue stock from inventory"""
		
		required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity']
		for field in required_fields:
			if field not in issue_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check available stock
		available = self._get_available_stock_for_location(
			issue_data['item_id'],
			issue_data['warehouse_id'],
			issue_data['location_id'],
			issue_data.get('lot_number'),
			issue_data.get('serial_number')
		)
		
		if available < issue_data['quantity']:
			raise ValueError(f"Insufficient stock available. Available: {available}, Requested: {issue_data['quantity']}")
		
		# Get cost for the issue
		unit_cost = self._get_fifo_cost(
			issue_data['item_id'],
			issue_data['warehouse_id'],
			issue_data['location_id'],
			issue_data['quantity']
		)
		
		# Create the movement record
		movement = IMSTCStockMovement(
			tenant_id=self.tenant_id,
			movement_type='Issue',
			movement_subtype=issue_data.get('movement_subtype', 'Sales Issue'),
			reference_number=issue_data.get('reference_number'),
			reference_type=issue_data.get('reference_type'),
			item_id=issue_data['item_id'],
			warehouse_id=issue_data['warehouse_id'],
			location_id=issue_data['location_id'],
			uom_id=issue_data.get('uom_id'),
			quantity=-issue_data['quantity'],  # Negative for issues
			unit_cost=unit_cost,
			total_cost=-issue_data['quantity'] * unit_cost,
			lot_number=issue_data.get('lot_number'),
			serial_number=issue_data.get('serial_number'),
			batch_number=issue_data.get('batch_number'),
			stock_status=issue_data.get('stock_status', 'Available'),
			user_id=issue_data.get('user_id'),
			notes=issue_data.get('notes')
		)
		
		self.session.add(movement)
		self.session.flush()
		
		# Update stock levels
		self._update_stock_level_for_issue(movement)
		
		# Check for alerts
		self._check_stock_alerts(issue_data['item_id'])
		
		self._log_stock_issue(movement)
		return movement
	
	def transfer_stock(self, transfer_data: Dict[str, Any]) -> IMSTCStockMovement:
		"""Transfer stock between locations"""
		
		required_fields = ['item_id', 'from_warehouse_id', 'from_location_id', 
						   'to_warehouse_id', 'to_location_id', 'quantity']
		for field in required_fields:
			if field not in transfer_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Check available stock at from location
		available = self._get_available_stock_for_location(
			transfer_data['item_id'],
			transfer_data['from_warehouse_id'],
			transfer_data['from_location_id'],
			transfer_data.get('lot_number'),
			transfer_data.get('serial_number')
		)
		
		if available < transfer_data['quantity']:
			raise ValueError(f"Insufficient stock available for transfer. Available: {available}, Requested: {transfer_data['quantity']}")
		
		# Get current cost
		unit_cost = self._get_current_unit_cost(
			transfer_data['item_id'],
			transfer_data['from_warehouse_id'],
			transfer_data['from_location_id']
		)
		
		# Create the movement record
		movement = IMSTCStockMovement(
			tenant_id=self.tenant_id,
			movement_type='Transfer',
			reference_number=transfer_data.get('reference_number'),
			reference_type=transfer_data.get('reference_type'),
			item_id=transfer_data['item_id'],
			warehouse_id=transfer_data['to_warehouse_id'],
			location_id=transfer_data['to_location_id'],
			from_warehouse_id=transfer_data['from_warehouse_id'],
			from_location_id=transfer_data['from_location_id'],
			to_warehouse_id=transfer_data['to_warehouse_id'],
			to_location_id=transfer_data['to_location_id'],
			uom_id=transfer_data.get('uom_id'),
			quantity=transfer_data['quantity'],
			unit_cost=unit_cost,
			total_cost=transfer_data['quantity'] * unit_cost,
			lot_number=transfer_data.get('lot_number'),
			serial_number=transfer_data.get('serial_number'),
			batch_number=transfer_data.get('batch_number'),
			stock_status=transfer_data.get('stock_status', 'Available'),
			user_id=transfer_data.get('user_id'),
			notes=transfer_data.get('notes')
		)
		
		self.session.add(movement)
		self.session.flush()
		
		# Update stock levels for both locations
		self._update_stock_level_for_transfer(movement)
		
		self._log_stock_transfer(movement)
		return movement
	
	def adjust_stock(self, adjustment_data: Dict[str, Any]) -> IMSTCStockMovement:
		"""Adjust stock levels (positive or negative)"""
		
		required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity', 'reason_code']
		for field in required_fields:
			if field not in adjustment_data:
				raise ValueError(f"Missing required field: {field}")
		
		# Get current cost for adjustment
		unit_cost = self._get_current_unit_cost(
			adjustment_data['item_id'],
			adjustment_data['warehouse_id'],
			adjustment_data['location_id']
		)
		
		# Create the movement record
		movement = IMSTCStockMovement(
			tenant_id=self.tenant_id,
			movement_type='Adjustment',
			reference_number=adjustment_data.get('reference_number'),
			item_id=adjustment_data['item_id'],
			warehouse_id=adjustment_data['warehouse_id'],
			location_id=adjustment_data['location_id'],
			uom_id=adjustment_data.get('uom_id'),
			quantity=adjustment_data['quantity'],
			unit_cost=unit_cost,
			total_cost=adjustment_data['quantity'] * unit_cost,
			lot_number=adjustment_data.get('lot_number'),
			serial_number=adjustment_data.get('serial_number'),
			batch_number=adjustment_data.get('batch_number'),
			stock_status=adjustment_data.get('stock_status', 'Available'),
			reason_code=adjustment_data['reason_code'],
			user_id=adjustment_data.get('user_id'),
			notes=adjustment_data.get('notes')
		)
		
		self.session.add(movement)
		self.session.flush()
		
		# Update stock levels
		self._update_stock_level_for_adjustment(movement)
		
		# Check for alerts
		self._check_stock_alerts(adjustment_data['item_id'])
		
		self._log_stock_adjustment(movement)
		return movement
	
	# Reporting and Analytics
	
	def get_inventory_valuation(self, warehouse_id: str = None, as_of_date: date = None) -> Dict[str, Any]:
		"""Get inventory valuation report"""
		
		if as_of_date is None:
			as_of_date = date.today()
		
		# Query for inventory valuation
		query = text("""
			SELECT 
				i.item_code,
				i.item_name,
				c.category_name,
				w.warehouse_name,
				SUM(sl.on_hand_quantity) as on_hand_qty,
				AVG(sl.unit_cost) as avg_unit_cost,
				SUM(sl.total_cost) as total_value
			FROM im_stc_item i
			JOIN im_stc_stock_level sl ON i.item_id = sl.item_id
			LEFT JOIN im_stc_item_category c ON i.category_id = c.category_id
			JOIN im_stc_warehouse w ON sl.warehouse_id = w.warehouse_id
			WHERE i.tenant_id = :tenant_id
			AND sl.on_hand_quantity > 0
			{}
			GROUP BY i.item_code, i.item_name, c.category_name, w.warehouse_name
			ORDER BY total_value DESC
		""".format(
			"AND sl.warehouse_id = :warehouse_id" if warehouse_id else ""
		))
		
		params = {'tenant_id': self.tenant_id}
		if warehouse_id:
			params['warehouse_id'] = warehouse_id
		
		result = self.session.execute(query, params)
		inventory_items = [dict(row) for row in result]
		
		# Calculate summary
		total_value = sum(item['total_value'] or 0 for item in inventory_items)
		total_items = len(inventory_items)
		
		return {
			'as_of_date': as_of_date.isoformat(),
			'warehouse_id': warehouse_id,
			'total_value': total_value,
			'total_items': total_items,
			'inventory_items': inventory_items
		}
	
	def get_inventory_turnover(self, item_id: str = None, days: int = 365) -> Dict[str, Any]:
		"""Calculate inventory turnover metrics"""
		
		end_date = date.today()
		start_date = end_date - timedelta(days=days)
		
		# Get cost of goods sold (issues)
		cogs_query = self.session.query(
			func.sum(IMSTCStockMovement.total_cost * -1)  # Issues are negative
		).filter(
			and_(
				IMSTCStockMovement.tenant_id == self.tenant_id,
				IMSTCStockMovement.movement_type == 'Issue',
				IMSTCStockMovement.movement_date >= start_date,
				IMSTCStockMovement.movement_date <= end_date
			)
		)
		
		if item_id:
			cogs_query = cogs_query.filter(IMSTCStockMovement.item_id == item_id)
		
		cogs = cogs_query.scalar() or Decimal('0')
		
		# Get average inventory value
		avg_inventory_query = self.session.query(
			func.avg(IMSTCStockLevel.total_cost)
		).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.on_hand_quantity > 0
			)
		)
		
		if item_id:
			avg_inventory_query = avg_inventory_query.filter(IMSTCStockLevel.item_id == item_id)
		
		avg_inventory = avg_inventory_query.scalar() or Decimal('0')
		
		# Calculate turnover
		turnover_ratio = cogs / avg_inventory if avg_inventory > 0 else Decimal('0')
		days_of_supply = Decimal(str(days)) / turnover_ratio if turnover_ratio > 0 else Decimal('0')
		
		return {
			'period_days': days,
			'cogs': float(cogs),
			'average_inventory_value': float(avg_inventory),
			'turnover_ratio': float(turnover_ratio),
			'days_of_supply': float(days_of_supply)
		}
	
	def get_abc_analysis(self, warehouse_id: str = None) -> List[Dict[str, Any]]:
		"""Perform ABC analysis on inventory items"""
		
		# Get item usage/value data
		query = text("""
			SELECT 
				i.item_id,
				i.item_code,
				i.item_name,
				SUM(ABS(sm.total_cost)) as annual_usage_value,
				SUM(sl.total_cost) as current_inventory_value
			FROM im_stc_item i
			LEFT JOIN im_stc_stock_movement sm ON i.item_id = sm.item_id 
				AND sm.movement_type = 'Issue'
				AND sm.movement_date >= CURRENT_DATE - INTERVAL '365 days'
			LEFT JOIN im_stc_stock_level sl ON i.item_id = sl.item_id
			WHERE i.tenant_id = :tenant_id
			AND i.is_active = true
			{}
			GROUP BY i.item_id, i.item_code, i.item_name
			HAVING SUM(ABS(sm.total_cost)) > 0
			ORDER BY annual_usage_value DESC
		""".format(
			"AND sl.warehouse_id = :warehouse_id" if warehouse_id else ""
		))
		
		params = {'tenant_id': self.tenant_id}
		if warehouse_id:
			params['warehouse_id'] = warehouse_id
		
		result = self.session.execute(query, params)
		items = [dict(row) for row in result]
		
		# Calculate cumulative percentages and assign ABC classifications
		total_value = sum(item['annual_usage_value'] or 0 for item in items)
		cumulative_value = 0
		
		for item in items:
			cumulative_value += item['annual_usage_value'] or 0
			cumulative_percentage = (cumulative_value / total_value * 100) if total_value > 0 else 0
			
			if cumulative_percentage <= 80:
				classification = 'A'
			elif cumulative_percentage <= 95:
				classification = 'B'
			else:
				classification = 'C'
			
			item['abc_classification'] = classification
			item['cumulative_percentage'] = cumulative_percentage
		
		return items
	
	# Alert Management
	
	def create_stock_alert(self, alert_data: Dict[str, Any]) -> IMSTCStockAlert:
		"""Create a new stock alert"""
		
		alert = IMSTCStockAlert(
			tenant_id=self.tenant_id,
			**alert_data
		)
		
		self.session.add(alert)
		self.session.flush()
		
		self._log_alert_creation(alert)
		return alert
	
	def get_active_alerts(self, alert_type: str = None, item_id: str = None) -> List[IMSTCStockAlert]:
		"""Get active stock alerts"""
		
		query = self.session.query(IMSTCStockAlert).filter(
			and_(
				IMSTCStockAlert.tenant_id == self.tenant_id,
				IMSTCStockAlert.status == 'Active'
			)
		)
		
		if alert_type:
			query = query.filter(IMSTCStockAlert.alert_type == alert_type)
		
		if item_id:
			query = query.filter(IMSTCStockAlert.item_id == item_id)
		
		return query.order_by(IMSTCStockAlert.alert_priority.desc(), IMSTCStockAlert.created_date).all()
	
	# Dashboard Methods
	
	def get_total_item_count(self) -> int:
		"""Get total active item count"""
		return self.session.query(IMSTCItem).filter(
			and_(
				IMSTCItem.tenant_id == self.tenant_id,
				IMSTCItem.is_active == True
			)
		).count()
	
	def get_total_inventory_value(self) -> Decimal:
		"""Get total inventory value"""
		result = self.session.query(func.sum(IMSTCStockLevel.total_cost)).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.on_hand_quantity > 0
			)
		).scalar()
		return result if result else Decimal('0')
	
	def get_locations_count(self) -> int:
		"""Get total active locations count"""
		return self.session.query(IMSTCLocation).filter(
			and_(
				IMSTCLocation.tenant_id == self.tenant_id,
				IMSTCLocation.is_active == True
			)
		).count()
	
	def calculate_inventory_turnover(self) -> float:
		"""Calculate overall inventory turnover ratio"""
		turnover_data = self.get_inventory_turnover()
		return turnover_data['turnover_ratio']
	
	def calculate_stockout_rate(self) -> float:
		"""Calculate stockout rate percentage"""
		total_items = self.get_total_item_count()
		if total_items == 0:
			return 0.0
		
		stockout_items = self.session.query(IMSTCItem).join(IMSTCStockLevel).filter(
			and_(
				IMSTCItem.tenant_id == self.tenant_id,
				IMSTCItem.is_active == True,
				IMSTCStockLevel.available_quantity <= 0
			)
		).distinct().count()
		
		return (stockout_items / total_items) * 100
	
	# Private Helper Methods
	
	def _create_initial_stock_level(self, item: IMSTCItem):
		"""Create initial stock level record for new item"""
		
		stock_level = IMSTCStockLevel(
			tenant_id=self.tenant_id,
			item_id=item.item_id,
			warehouse_id=item.default_warehouse_id,
			location_id=item.default_location_id,
			uom_id=item.primary_uom_id,
			on_hand_quantity=Decimal('0'),
			available_quantity=Decimal('0'),
			unit_cost=item.standard_cost or Decimal('0'),
			total_cost=Decimal('0')
		)
		
		self.session.add(stock_level)
	
	def _update_stock_level_for_receipt(self, movement: IMSTCStockMovement):
		"""Update stock levels for a receipt transaction"""
		
		# Find or create stock level record
		stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == movement.item_id,
				IMSTCStockLevel.warehouse_id == movement.warehouse_id,
				IMSTCStockLevel.location_id == movement.location_id,
				IMSTCStockLevel.lot_number == movement.lot_number,
				IMSTCStockLevel.serial_number == movement.serial_number,
				IMSTCStockLevel.stock_status == movement.stock_status
			)
		).first()
		
		if not stock_level:
			stock_level = IMSTCStockLevel(
				tenant_id=self.tenant_id,
				item_id=movement.item_id,
				warehouse_id=movement.warehouse_id,
				location_id=movement.location_id,
				uom_id=movement.uom_id,
				lot_number=movement.lot_number,
				serial_number=movement.serial_number,
				batch_number=movement.batch_number,
				expiry_date=movement.expiry_date,
				stock_status=movement.stock_status,
				quality_status=movement.quality_status
			)
			self.session.add(stock_level)
		
		# Update quantities and cost
		old_quantity = stock_level.on_hand_quantity
		new_quantity = old_quantity + movement.quantity
		
		# Weighted average cost calculation
		if new_quantity > 0:
			total_cost = (stock_level.total_cost + movement.total_cost)
			stock_level.unit_cost = total_cost / new_quantity
			stock_level.total_cost = total_cost
		
		stock_level.on_hand_quantity = new_quantity
		stock_level.available_quantity = new_quantity - stock_level.allocated_quantity
		stock_level.last_received_date = movement.movement_date
		
		if stock_level.first_received_date is None:
			stock_level.first_received_date = movement.movement_date
	
	def _update_stock_level_for_issue(self, movement: IMSTCStockMovement):
		"""Update stock levels for an issue transaction"""
		
		# Find stock level record (must exist for issues)
		stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == movement.item_id,
				IMSTCStockLevel.warehouse_id == movement.warehouse_id,
				IMSTCStockLevel.location_id == movement.location_id,
				IMSTCStockLevel.lot_number == movement.lot_number,
				IMSTCStockLevel.serial_number == movement.serial_number,
				IMSTCStockLevel.stock_status == movement.stock_status
			)
		).first()
		
		if not stock_level:
			raise ValueError("Stock level record not found for issue transaction")
		
		# Update quantities
		issue_quantity = abs(movement.quantity)  # Movement quantity is negative for issues
		stock_level.on_hand_quantity -= issue_quantity
		stock_level.available_quantity = stock_level.on_hand_quantity - stock_level.allocated_quantity
		stock_level.total_cost = stock_level.on_hand_quantity * stock_level.unit_cost
		stock_level.last_issued_date = movement.movement_date
	
	def _update_stock_level_for_transfer(self, movement: IMSTCStockMovement):
		"""Update stock levels for a transfer transaction"""
		
		# Decrease from location
		from_stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == movement.item_id,
				IMSTCStockLevel.warehouse_id == movement.from_warehouse_id,
				IMSTCStockLevel.location_id == movement.from_location_id,
				IMSTCStockLevel.lot_number == movement.lot_number,
				IMSTCStockLevel.serial_number == movement.serial_number,
				IMSTCStockLevel.stock_status == movement.stock_status
			)
		).first()
		
		if from_stock_level:
			from_stock_level.on_hand_quantity -= movement.quantity
			from_stock_level.available_quantity = from_stock_level.on_hand_quantity - from_stock_level.allocated_quantity
			from_stock_level.total_cost = from_stock_level.on_hand_quantity * from_stock_level.unit_cost
		
		# Increase to location
		to_stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == movement.item_id,
				IMSTCStockLevel.warehouse_id == movement.to_warehouse_id,
				IMSTCStockLevel.location_id == movement.to_location_id,
				IMSTCStockLevel.lot_number == movement.lot_number,
				IMSTCStockLevel.serial_number == movement.serial_number,
				IMSTCStockLevel.stock_status == movement.stock_status
			)
		).first()
		
		if not to_stock_level:
			to_stock_level = IMSTCStockLevel(
				tenant_id=self.tenant_id,
				item_id=movement.item_id,
				warehouse_id=movement.to_warehouse_id,
				location_id=movement.to_location_id,
				uom_id=movement.uom_id,
				lot_number=movement.lot_number,
				serial_number=movement.serial_number,
				batch_number=movement.batch_number,
				expiry_date=movement.expiry_date,
				stock_status=movement.stock_status,
				quality_status=movement.quality_status,
				unit_cost=movement.unit_cost
			)
			self.session.add(to_stock_level)
		
		to_stock_level.on_hand_quantity += movement.quantity
		to_stock_level.available_quantity = to_stock_level.on_hand_quantity - to_stock_level.allocated_quantity
		to_stock_level.total_cost = to_stock_level.on_hand_quantity * to_stock_level.unit_cost
	
	def _update_stock_level_for_adjustment(self, movement: IMSTCStockMovement):
		"""Update stock levels for an adjustment transaction"""
		
		# Find or create stock level record
		stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == movement.item_id,
				IMSTCStockLevel.warehouse_id == movement.warehouse_id,
				IMSTCStockLevel.location_id == movement.location_id,
				IMSTCStockLevel.lot_number == movement.lot_number,
				IMSTCStockLevel.serial_number == movement.serial_number,
				IMSTCStockLevel.stock_status == movement.stock_status
			)
		).first()
		
		if not stock_level:
			stock_level = IMSTCStockLevel(
				tenant_id=self.tenant_id,
				item_id=movement.item_id,
				warehouse_id=movement.warehouse_id,
				location_id=movement.location_id,
				uom_id=movement.uom_id,
				lot_number=movement.lot_number,
				serial_number=movement.serial_number,
				batch_number=movement.batch_number,
				expiry_date=movement.expiry_date,
				stock_status=movement.stock_status,
				quality_status=movement.quality_status,
				unit_cost=movement.unit_cost
			)
			self.session.add(stock_level)
		
		# Apply adjustment
		stock_level.on_hand_quantity += movement.quantity
		stock_level.available_quantity = stock_level.on_hand_quantity - stock_level.allocated_quantity
		stock_level.total_cost = stock_level.on_hand_quantity * stock_level.unit_cost
		stock_level.last_counted_date = movement.movement_date
	
	def _get_available_stock_for_location(self, item_id: str, warehouse_id: str, location_id: str, 
										  lot_number: str = None, serial_number: str = None) -> Decimal:
		"""Get available stock for specific location and optional lot/serial"""
		
		query = self.session.query(func.sum(IMSTCStockLevel.available_quantity)).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id,
				IMSTCStockLevel.warehouse_id == warehouse_id,
				IMSTCStockLevel.location_id == location_id
			)
		)
		
		if lot_number:
			query = query.filter(IMSTCStockLevel.lot_number == lot_number)
		
		if serial_number:
			query = query.filter(IMSTCStockLevel.serial_number == serial_number)
		
		result = query.scalar()
		return result if result else Decimal('0')
	
	def _get_fifo_cost(self, item_id: str, warehouse_id: str, location_id: str, quantity: Decimal) -> Decimal:
		"""Get FIFO cost for issuing stock"""
		
		# Get stock levels ordered by FIFO (first received first)
		stock_levels = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id,
				IMSTCStockLevel.warehouse_id == warehouse_id,
				IMSTCStockLevel.location_id == location_id,
				IMSTCStockLevel.available_quantity > 0
			)
		).order_by(IMSTCStockLevel.first_received_date).all()
		
		if not stock_levels:
			return Decimal('0')
		
		# Calculate weighted average cost for the quantity being issued
		remaining_qty = quantity
		total_cost = Decimal('0')
		
		for stock_level in stock_levels:
			if remaining_qty <= 0:
				break
			
			qty_from_level = min(remaining_qty, stock_level.available_quantity)
			total_cost += qty_from_level * stock_level.unit_cost
			remaining_qty -= qty_from_level
		
		return total_cost / quantity if quantity > 0 else Decimal('0')
	
	def _get_current_unit_cost(self, item_id: str, warehouse_id: str, location_id: str) -> Decimal:
		"""Get current unit cost for an item at a location"""
		
		stock_level = self.session.query(IMSTCStockLevel).filter(
			and_(
				IMSTCStockLevel.tenant_id == self.tenant_id,
				IMSTCStockLevel.item_id == item_id,
				IMSTCStockLevel.warehouse_id == warehouse_id,
				IMSTCStockLevel.location_id == location_id,
				IMSTCStockLevel.on_hand_quantity > 0
			)
		).first()
		
		if stock_level:
			return stock_level.unit_cost
		
		# Fallback to item standard cost
		item = self.get_item_by_id(item_id)
		return item.standard_cost if item else Decimal('0')
	
	def _check_stock_alerts(self, item_id: str):
		"""Check and generate stock alerts for an item"""
		
		item = self.get_item_by_id(item_id)
		if not item:
			return
		
		# Check for low stock
		available_stock = self.get_available_stock(item_id)
		
		if available_stock <= item.reorder_point:
			# Check if alert already exists
			existing_alert = self.session.query(IMSTCStockAlert).filter(
				and_(
					IMSTCStockAlert.tenant_id == self.tenant_id,
					IMSTCStockAlert.item_id == item_id,
					IMSTCStockAlert.alert_type == 'Low Stock',
					IMSTCStockAlert.status == 'Active'
				)
			).first()
			
			if not existing_alert:
				self.create_stock_alert({
					'alert_type': 'Low Stock',
					'alert_priority': 'High' if available_stock == 0 else 'Medium',
					'alert_title': f'Low Stock Alert: {item.item_name}',
					'alert_message': f'Item {item.item_code} is below reorder point. Available: {available_stock}, Reorder Point: {item.reorder_point}',
					'item_id': item_id,
					'threshold_value': item.reorder_point,
					'current_value': available_stock
				})
	
	# Logging Methods
	
	def _log_item_creation(self, item: IMSTCItem):
		"""Log item creation"""
		print(f"Item created: {item.item_code} - {item.item_name}")
	
	def _log_item_update(self, item: IMSTCItem, updates: Dict[str, Any]):
		"""Log item update"""
		print(f"Item updated: {item.item_code} - Changes: {list(updates.keys())}")
	
	def _log_warehouse_creation(self, warehouse: IMSTCWarehouse):
		"""Log warehouse creation"""
		print(f"Warehouse created: {warehouse.warehouse_code} - {warehouse.warehouse_name}")
	
	def _log_location_creation(self, location: IMSTCLocation):
		"""Log location creation"""
		print(f"Location created: {location.location_code} - {location.location_name}")
	
	def _log_stock_receipt(self, movement: IMSTCStockMovement):
		"""Log stock receipt"""
		print(f"Stock received: {movement.quantity} units of item {movement.item_id}")
	
	def _log_stock_issue(self, movement: IMSTCStockMovement):
		"""Log stock issue"""
		print(f"Stock issued: {abs(movement.quantity)} units of item {movement.item_id}")
	
	def _log_stock_transfer(self, movement: IMSTCStockMovement):
		"""Log stock transfer"""
		print(f"Stock transferred: {movement.quantity} units of item {movement.item_id}")
	
	def _log_stock_adjustment(self, movement: IMSTCStockMovement):
		"""Log stock adjustment"""
		print(f"Stock adjusted: {movement.quantity} units of item {movement.item_id}")
	
	def _log_alert_creation(self, alert: IMSTCStockAlert):
		"""Log alert creation"""
		print(f"Alert created: {alert.alert_type} - {alert.alert_title}")