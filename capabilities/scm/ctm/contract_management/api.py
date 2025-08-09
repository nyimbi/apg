"""
Contract Management API

REST API endpoints for contract management.
"""

from flask import Blueprint, request, jsonify
from flask_appbuilder import AppBuilder
from flask_appbuilder.api import BaseApi, expose
from flask_appbuilder.models.sqla.interface import SQLAInterface
from flask_appbuilder.security.decorators import has_access_api

from .models import PPCContract
from .service import ContractManagementService


class ContractApi(BaseApi):
	"""REST API for contract management"""
	
	resource_name = 'contract'
	datamodel = SQLAInterface(PPCContract)
	
	@expose('/list')
	@has_access_api
	def list(self):
		"""List active contracts"""
		
		try:
			service = ContractManagementService(self.get_tenant_id())
			
			# Get contracts expiring soon for API response
			expiring_contracts = service.get_contracts_expiring_soon()
			
			result = []
			for contract in expiring_contracts:
				result.append({
					'contract_id': contract.contract_id,
					'contract_number': contract.contract_number,
					'contract_title': contract.contract_title,
					'vendor_name': contract.vendor_name,
					'expiration_date': contract.expiration_date.isoformat(),
					'contract_value': float(contract.contract_value),
					'status': contract.status
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
	
	@expose('/expiring_soon')
	@has_access_api
	def expiring_soon(self):
		"""Get contracts expiring soon"""
		
		try:
			days = int(request.args.get('days', 90))
			service = ContractManagementService(self.get_tenant_id())
			contracts = service.get_contracts_expiring_soon(days)
			
			result = []
			for contract in contracts:
				result.append({
					'contract_id': contract.contract_id,
					'contract_number': contract.contract_number,
					'contract_title': contract.contract_title,
					'vendor_name': contract.vendor_name,
					'expiration_date': contract.expiration_date.isoformat(),
					'days_until_expiration': (contract.expiration_date - contract.effective_date).days,
					'auto_renewal': contract.auto_renewal
				})
			
			return jsonify({
				'success': True,
				'data': result,
				'count': len(result)
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
	
	appbuilder.add_api(ContractApi)