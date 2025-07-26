"""
Expiry Date Management API

REST API endpoints for expiry tracking, FEFO management,
and waste reporting.
"""

from flask import request, jsonify
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder import AppBuilder
from flask_appbuilder.security.decorators import protect
from datetime import datetime, date
from typing import Dict, List, Any

from .service import ExpiryDateService


class ExpiryDateApi(BaseApi):
	"""Expiry Date Management API endpoints"""
	
	resource_name = 'expiry_management'
	
	@expose('/expiring_items', methods=['GET'])
	@protect()
	def get_expiring_items(self):
		"""Get items expiring within specified days"""
		try:
			tenant_id = self._get_tenant_id()
			days = int(request.args.get('days', 30))
			limit = int(request.args.get('limit', 100))
			
			with ExpiryDateService(tenant_id) as service:
				items = service.get_items_expiring_soon(days=days, limit=limit)
				
				items_data = []
				for item in items:
					items_data.append({
						'expiry_item_id': item.expiry_item_id,
						'item_id': item.item_id,
						'batch_number': item.batch_number,
						'expiry_date': item.expiry_date.isoformat(),
						'days_to_expiry': item.days_to_expiry,
						'current_quantity': float(item.current_quantity),
						'expiry_status': item.expiry_status,
						'alert_level': item.alert_level
					})
				
				return jsonify({
					'success': True,
					'data': items_data,
					'total': len(items_data)
				})
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/fefo_sequence/<string:item_id>', methods=['GET'])
	@protect()
	def get_fefo_sequence(self, item_id: str):
		"""Get FEFO sequence for an item"""
		try:
			tenant_id = self._get_tenant_id()
			warehouse_id = request.args.get('warehouse_id')
			
			with ExpiryDateService(tenant_id) as service:
				sequence = service.get_fefo_sequence(item_id, warehouse_id)
				
				sequence_data = []
				for item in sequence:
					sequence_data.append({
						'expiry_item_id': item.expiry_item_id,
						'batch_number': item.batch_number,
						'expiry_date': item.expiry_date.isoformat(),
						'available_quantity': float(item.available_quantity),
						'days_to_expiry': item.days_to_expiry
					})
				
				return jsonify({
					'success': True,
					'data': sequence_data
				})
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	@expose('/validate_fefo', methods=['POST'])
	@protect()
	def validate_fefo(self):
		"""Validate FEFO compliance for a selection"""
		try:
			tenant_id = self._get_tenant_id()
			data = request.get_json()
			
			required_fields = ['item_id', 'selected_batch', 'requested_quantity']
			for field in required_fields:
				if field not in data:
					return jsonify({
						'success': False,
						'error': f'Missing required field: {field}'
					}), 400
			
			with ExpiryDateService(tenant_id) as service:
				validation = service.validate_fefo_compliance(
					data['item_id'],
					data['selected_batch'],
					data['requested_quantity']
				)
				
				return jsonify({
					'success': True,
					'data': validation
				})
				
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 500
	
	def _get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


def register_api_views(appbuilder: AppBuilder):
	"""Register Expiry Date Management API views"""
	
	appbuilder.add_api(ExpiryDateApi)
	print("Expiry Date Management API registered")