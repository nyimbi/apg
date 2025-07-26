"""
Vendor Management API

REST API endpoints for vendor management.
"""

from flask import Blueprint, request, jsonify
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access_api

from .models import PPVVendor
from .service import VendorManagementService


class VendorApi(BaseApi):
	"""REST API for vendor management"""
	
	resource_name = 'vendor'
	datamodel = SQLAInterface(PPVVendor)
	
	@expose('/list')
	@has_access_api
	def list(self):
		"""List vendors"""
		
		try:
			service = VendorManagementService(self.get_tenant_id())
			vendors = service.get_top_vendors_by_spend(limit=50)
			
			return jsonify({
				'success': True,
				'data': vendors
			})
			
		except Exception as e:
			return jsonify({
				'success': False,
				'error': str(e)
			}), 400
	
	def get_tenant_id(self) -> str:
		return "default_tenant"


def register_api_views(appbuilder: AppBuilder):
	"""Register API views with Flask-AppBuilder"""
	
	appbuilder.add_api(VendorApi)