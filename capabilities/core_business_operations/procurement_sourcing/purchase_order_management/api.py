"""
Purchase Order Management API

REST API endpoints for purchase order management.
"""

from flask import Blueprint, request, jsonify
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access_api

from .models import PPOPurchaseOrder
from .service import PurchaseOrderService


class PurchaseOrderApi(BaseApi):
	"""REST API for purchase order management"""
	
	resource_name = 'purchase_order'
	datamodel = SQLAInterface(PPOPurchaseOrder)
	
	@expose('/list')
	@has_access_api
	def list(self):
		"""List purchase orders"""
		
		try:
			service = PurchaseOrderService(self.get_tenant_id())
			pos = service.get_purchase_orders_by_status('Open')
			
			result = []
			for po in pos:
				result.append({
					'po_id': po.po_id,
					'po_number': po.po_number,
					'title': po.title,
					'vendor_name': po.vendor_name,
					'status': po.status,
					'total_amount': float(po.total_amount)
				})
			
			return jsonify({
				'success': True,
				'data': result
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	def get_tenant_id(self) -> str:
		"""Get current tenant ID"""
		return "default_tenant"


def register_api_views(appbuilder: AppBuilder):
	"""Register API views with Flask-AppBuilder"""
	
	appbuilder.add_api(PurchaseOrderApi)
