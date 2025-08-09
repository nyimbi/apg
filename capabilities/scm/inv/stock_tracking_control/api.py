"""
Stock Tracking & Control API

REST API endpoints for inventory management, stock operations,
and real-time inventory tracking.
"""

from flask import request, jsonify, current_app
from flask_appbuilder.api import BaseApi, expose, rison
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.decorators import protect
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json

from .service import StockTrackingService
from .models import IMSTCItem, IMSTCStockLevel, IMSTCStockMovement, IMSTCStockAlert


class StockTrackingApi(BaseApi):
	"""Stock Tracking & Control API endpoints"""
	
	resource_name = 'stock_tracking'
	
	def __init__(self):
		super().__init__()
	
	@expose('/items', methods=['GET'])
	@protect()
	def get_items(self):
		"""Get inventory items with filtering and search"""
		
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			search_text = request.args.get('search', '')
			category_id = request.args.get('category_id')
			item_type = request.args.get('item_type')
			is_active = request.args.get('is_active', 'true').lower() == 'true'
			abc_classification = request.args.get('abc_classification')
			limit = min(int(request.args.get('limit', 100)), 1000)
			offset = int(request.args.get('offset', 0))
			
			# Build filters
			filters = {'is_active': is_active}
			if search_text:
				filters['search_text'] = search_text
			if category_id:
				filters['category_id'] = category_id
			if item_type:
				filters['item_type'] = item_type
			if abc_classification:
				filters['abc_classification'] = abc_classification
			
			with StockTrackingService(tenant_id) as service:
				items = service.search_items(filters, limit=limit, offset=offset)
				
				items_data = []
				for item in items:
					items_data.append({
						'item_id': item.item_id,
						'item_code': item.item_code,
						'item_name': item.item_name,
						'description': item.description,
						'category_name': item.category.category_name if item.category else None,
						'item_type': item.item_type,
						'abc_classification': item.abc_classification,
						'primary_uom': item.primary_uom.uom_code if item.primary_uom else None,
						'is_active': item.is_active,
						'reorder_point': float(item.reorder_point),
						'min_stock_level': float(item.min_stock_level),
						'max_stock_level': float(item.max_stock_level) if item.max_stock_level else None,
						'standard_cost': float(item.standard_cost),
						'requires_serial_tracking': item.requires_serial_tracking,
						'requires_lot_tracking': item.requires_lot_tracking,
						'requires_expiry_tracking': item.requires_expiry_tracking
					})
				
				return jsonify({
					'success': True,
					'data': items_data,
					'total': len(items_data),
					'limit': limit,
					'offset': offset
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/items/<string:item_id>', methods=['GET'])
	@protect()
	def get_item(self, item_id: str):
		"""Get specific inventory item details"""
		
		try:
			tenant_id = self._get_tenant_id()
			
			with StockTrackingService(tenant_id) as service:
				item = service.get_item_by_id(item_id)
				
				if not item:
					return jsonify({
						'success': False,
						'error': 'Item not found'
					}), 404
				
				# Get current stock levels
				stock_levels = service.get_current_stock(item_id)
				
				stock_data = []
				for stock in stock_levels:
					stock_data.append({
						'warehouse_id': stock.warehouse_id,
						'warehouse_name': stock.warehouse.warehouse_name if stock.warehouse else None,
						'location_id': stock.location_id,
						'location_code': stock.location.location_code if stock.location else None,
						'lot_number': stock.lot_number,
						'serial_number': stock.serial_number,
						'on_hand_quantity': float(stock.on_hand_quantity),
						'available_quantity': float(stock.available_quantity),
						'allocated_quantity': float(stock.allocated_quantity),
						'stock_status': stock.stock_status,
						'quality_status': stock.quality_status,
						'unit_cost': float(stock.unit_cost),
						'total_cost': float(stock.total_cost),
						'expiry_date': stock.expiry_date.isoformat() if stock.expiry_date else None
					})
				
				item_data = {
					'item_id': item.item_id,
					'item_code': item.item_code,
					'item_name': item.item_name,
					'description': item.description,
					'short_description': item.short_description,
					'category_id': item.category_id,
					'category_name': item.category.category_name if item.category else None,
					'item_type': item.item_type,
					'abc_classification': item.abc_classification,
					'primary_uom_id': item.primary_uom_id,
					'primary_uom_code': item.primary_uom.uom_code if item.primary_uom else None,
					'weight': float(item.weight) if item.weight else None,
					'weight_uom': item.weight_uom,
					'volume': float(item.volume) if item.volume else None,
					'volume_uom': item.volume_uom,
					'reorder_point': float(item.reorder_point),
					'min_stock_level': float(item.min_stock_level),
					'max_stock_level': float(item.max_stock_level) if item.max_stock_level else None,
					'safety_stock': float(item.safety_stock),
					'standard_cost': float(item.standard_cost),
					'last_cost': float(item.last_cost),
					'average_cost': float(item.average_cost),
					'cost_method': item.cost_method,
					'is_active': item.is_active,
					'requires_serial_tracking': item.requires_serial_tracking,
					'requires_lot_tracking': item.requires_lot_tracking,
					'requires_expiry_tracking': item.requires_expiry_tracking,
					'shelf_life_days': item.shelf_life_days,
					'is_perishable': item.is_perishable,
					'is_hazardous': item.is_hazardous,
					'stock_levels': stock_data,
					'total_on_hand': float(service.get_total_on_hand(item_id)),
					'total_available': float(service.get_available_stock(item_id))
				}
				
				return jsonify({
					'success': True,
					'data': item_data
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/stock_levels', methods=['GET'])
	@protect()
	def get_stock_levels(self):
		"""Get current stock levels with filtering"""
		
		try:
			tenant_id = self._get_tenant_id()
			
			# Get query parameters
			item_id = request.args.get('item_id')
			warehouse_id = request.args.get('warehouse_id')
			location_id = request.args.get('location_id')
			stock_status = request.args.get('stock_status')
			quality_status = request.args.get('quality_status')
			min_quantity = request.args.get('min_quantity', type=float)
			limit = min(int(request.args.get('limit', 100)), 1000)
			offset = int(request.args.get('offset', 0))
			
			with StockTrackingService(tenant_id) as service:
				# Build query filters
				from sqlalchemy import and_
				from .models import IMSTCStockLevel
				
				filters = [IMSTCStockLevel.tenant_id == tenant_id]
				
				if item_id:
					filters.append(IMSTCStockLevel.item_id == item_id)
				if warehouse_id:
					filters.append(IMSTCStockLevel.warehouse_id == warehouse_id)
				if location_id:
					filters.append(IMSTCStockLevel.location_id == location_id)
				if stock_status:
					filters.append(IMSTCStockLevel.stock_status == stock_status)
				if quality_status:
					filters.append(IMSTCStockLevel.quality_status == quality_status)
				if min_quantity is not None:
					filters.append(IMSTCStockLevel.on_hand_quantity >= min_quantity)
				
				# Query stock levels
				stock_levels = service.session.query(IMSTCStockLevel).filter(
					and_(*filters)
				).offset(offset).limit(limit).all()
				
				stock_data = []
				for stock in stock_levels:
					stock_data.append({
						'stock_level_id': stock.stock_level_id,
						'item_id': stock.item_id,
						'item_code': stock.item.item_code if stock.item else None,
						'item_name': stock.item.item_name if stock.item else None,
						'warehouse_id': stock.warehouse_id,
						'warehouse_name': stock.warehouse.warehouse_name if stock.warehouse else None,
						'location_id': stock.location_id,
						'location_code': stock.location.location_code if stock.location else None,
						'uom_code': stock.uom.uom_code if stock.uom else None,
						'lot_number': stock.lot_number,
						'serial_number': stock.serial_number,
						'batch_number': stock.batch_number,
						'expiry_date': stock.expiry_date.isoformat() if stock.expiry_date else None,
						'stock_status': stock.stock_status,
						'quality_status': stock.quality_status,
						'on_hand_quantity': float(stock.on_hand_quantity),
						'allocated_quantity': float(stock.allocated_quantity),
						'available_quantity': float(stock.available_quantity),
						'on_order_quantity': float(stock.on_order_quantity),
						'reserved_quantity': float(stock.reserved_quantity),
						'unit_cost': float(stock.unit_cost),
						'total_cost': float(stock.total_cost),
						'first_received_date': stock.first_received_date.isoformat() if stock.first_received_date else None,
						'last_received_date': stock.last_received_date.isoformat() if stock.last_received_date else None,
						'last_issued_date': stock.last_issued_date.isoformat() if stock.last_issued_date else None,
						'last_counted_date': stock.last_counted_date.isoformat() if stock.last_counted_date else None
					})
				
				return jsonify({
					'success': True,
					'data': stock_data,
					'total': len(stock_data),
					'limit': limit,
					'offset': offset
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/receive_stock', methods=['POST'])
	@protect()
	def receive_stock(self):
		"""Receive stock into inventory"""
		
		try:
			tenant_id = self._get_tenant_id()
			data = request.get_json()
			
			# Validate required fields
			required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity', 'unit_cost']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Missing required field: {field}'
					}), 400
			
			# Add user context
			data['user_id'] = self._get_current_user_id()
			
			with StockTrackingService(tenant_id) as service:
				movement = service.receive_stock(data)
				
				return jsonify({
					'success': True,
					'data': {
						'movement_id': movement.movement_id,
						'movement_type': movement.movement_type,
						'quantity': float(movement.quantity),
						'unit_cost': float(movement.unit_cost),
						'total_cost': float(movement.total_cost),
						'movement_date': movement.movement_date.isoformat(),
						'reference_number': movement.reference_number
					}
				})
		
		except ValueError as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/issue_stock', methods=['POST'])
	@protect()
	def issue_stock(self):
		"""Issue stock from inventory"""
		
		try:
			tenant_id = self._get_tenant_id()
			data = request.get_json()
			
			# Validate required fields
			required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Missing required field: {field}'
					}), 400
			
			# Add user context
			data['user_id'] = self._get_current_user_id()
			
			with StockTrackingService(tenant_id) as service:
				movement = service.issue_stock(data)
				
				return jsonify({
					'success': True,
					'data': {
						'movement_id': movement.movement_id,
						'movement_type': movement.movement_type,
						'quantity': float(abs(movement.quantity)),  # Show as positive
						'unit_cost': float(movement.unit_cost),
						'total_cost': float(abs(movement.total_cost)),  # Show as positive
						'movement_date': movement.movement_date.isoformat(),
						'reference_number': movement.reference_number
					}
				})
		
		except ValueError as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/transfer_stock', methods=['POST'])
	@protect()
	def transfer_stock(self):
		"""Transfer stock between locations"""
		
		try:
			tenant_id = self._get_tenant_id()
			data = request.get_json()
			
			# Validate required fields
			required_fields = ['item_id', 'from_warehouse_id', 'from_location_id', 
							   'to_warehouse_id', 'to_location_id', 'quantity']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Missing required field: {field}'
					}), 400
			
			# Add user context
			data['user_id'] = self._get_current_user_id()
			
			with StockTrackingService(tenant_id) as service:
				movement = service.transfer_stock(data)
				
				return jsonify({
					'success': True,
					'data': {
						'movement_id': movement.movement_id,
						'movement_type': movement.movement_type,
						'quantity': float(movement.quantity),
						'unit_cost': float(movement.unit_cost),
						'total_cost': float(movement.total_cost),
						'movement_date': movement.movement_date.isoformat(),
						'from_warehouse_id': movement.from_warehouse_id,
						'from_location_id': movement.from_location_id,
						'to_warehouse_id': movement.to_warehouse_id,
						'to_location_id': movement.to_location_id,
						'reference_number': movement.reference_number
					}
				})
		
		except ValueError as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/adjust_stock', methods=['POST'])
	@protect()
	def adjust_stock(self):
		"""Adjust stock levels"""
		
		try:
			tenant_id = self._get_tenant_id()
			data = request.get_json()
			
			# Validate required fields
			required_fields = ['item_id', 'warehouse_id', 'location_id', 'quantity', 'reason_code']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Missing required field: {field}'
					}), 400
			
			# Add user context
			data['user_id'] = self._get_current_user_id()
			
			with StockTrackingService(tenant_id) as service:
				movement = service.adjust_stock(data)
				
				return jsonify({
					'success': True,
					'data': {
						'movement_id': movement.movement_id,
						'movement_type': movement.movement_type,
						'quantity': float(movement.quantity),
						'unit_cost': float(movement.unit_cost),
						'total_cost': float(movement.total_cost),
						'movement_date': movement.movement_date.isoformat(),
						'reason_code': movement.reason_code,
						'reference_number': movement.reference_number
					}
				})
		
		except ValueError as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/low_stock_items', methods=['GET'])
	@protect()
	def get_low_stock_items(self):
		"""Get items that are below reorder point"""
		
		try:
			tenant_id = self._get_tenant_id()
			warehouse_id = request.args.get('warehouse_id')
			limit = min(int(request.args.get('limit', 100)), 1000)
			
			with StockTrackingService(tenant_id) as service:
				low_stock_items = service.get_low_stock_items(warehouse_id=warehouse_id, limit=limit)
				
				return jsonify({
					'success': True,
					'data': low_stock_items,
					'total': len(low_stock_items)
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/inventory_valuation', methods=['GET'])
	@protect()
	def get_inventory_valuation(self):
		"""Get inventory valuation report"""
		
		try:
			tenant_id = self._get_tenant_id()
			warehouse_id = request.args.get('warehouse_id')
			as_of_date_str = request.args.get('as_of_date')
			
			as_of_date = None
			if as_of_date_str:
				as_of_date = datetime.strptime(as_of_date_str, '%Y-%m-%d').date()
			
			with StockTrackingService(tenant_id) as service:
				valuation_data = service.get_inventory_valuation(
					warehouse_id=warehouse_id,
					as_of_date=as_of_date
				)
				
				return jsonify({
					'success': True,
					'data': valuation_data
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/alerts', methods=['GET'])
	@protect()
	def get_alerts(self):
		"""Get active stock alerts"""
		
		try:
			tenant_id = self._get_tenant_id()
			alert_type = request.args.get('alert_type')
			item_id = request.args.get('item_id')
			
			with StockTrackingService(tenant_id) as service:
				alerts = service.get_active_alerts(alert_type=alert_type, item_id=item_id)
				
				alerts_data = []
				for alert in alerts:
					alerts_data.append({
						'alert_id': alert.alert_id,
						'alert_type': alert.alert_type,
						'alert_priority': alert.alert_priority,
						'alert_title': alert.alert_title,
						'alert_message': alert.alert_message,
						'item_id': alert.item_id,
						'item_code': alert.item.item_code if alert.item else None,
						'item_name': alert.item.item_name if alert.item else None,
						'warehouse_id': alert.warehouse_id,
						'warehouse_name': alert.warehouse.warehouse_name if alert.warehouse else None,
						'threshold_value': float(alert.threshold_value) if alert.threshold_value else None,
						'current_value': float(alert.current_value) if alert.current_value else None,
						'status': alert.status,
						'created_date': alert.created_date.isoformat(),
						'escalation_level': alert.escalation_level
					})
				
				return jsonify({
					'success': True,
					'data': alerts_data,
					'total': len(alerts_data)
				})
		
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		# TODO: Implement proper tenant resolution from request context
		return "default_tenant"
	
	def _get_current_user_id(self) -> str:
		"""Get current user ID"""
		# TODO: Implement proper user resolution from request context  
		return "current_user"


def register_api_views(appbuilder: AppBuilder):
	"""Register Stock Tracking API views"""
	
	appbuilder.add_api(StockTrackingApi)
	print("Stock Tracking & Control API registered")