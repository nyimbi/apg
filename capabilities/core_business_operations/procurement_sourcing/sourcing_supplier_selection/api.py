"""
Sourcing & Supplier Selection API

REST API endpoints for RFQ/RFP management and supplier evaluation.
"""

from flask import Blueprint, request, jsonify
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access_api

from .models import PPSRFQHeader
from .service import SourcingSupplierSelectionService


class SourcingApi(BaseApi):
	"""REST API for sourcing and supplier selection"""
	
	resource_name = 'sourcing'
	datamodel = SQLAInterface(PPSRFQHeader)
	
	@expose('/rfqs')
	@has_access_api
	def list_rfqs(self):
		"""List active RFQs"""
		
		try:
			service = SourcingSupplierSelectionService(self.get_tenant_id())
			rfqs = service.get_active_rfqs()
			
			result = []
			for rfq in rfqs:
				result.append({
					'rfq_id': rfq.rfq_id,
					'rfq_number': rfq.rfq_number,
					'rfq_title': rfq.rfq_title,
					'status': rfq.status,
					'estimated_value': float(rfq.estimated_value)
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
		return "default_tenant"


def register_api_views(appbuilder: AppBuilder):
	"""Register API views with Flask-AppBuilder"""
	
	appbuilder.add_api(SourcingApi)