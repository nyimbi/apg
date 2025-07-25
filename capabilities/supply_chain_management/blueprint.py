"""
Supply Chain Management Blueprint

Flask blueprint registration for supply chain management capability.
"""

from flask import Blueprint
from flask_appbuilder import BaseView

def create_blueprint():
	"""Create supply chain management blueprint"""
	return Blueprint(
		'supply_chain_management',
		__name__,
		url_prefix='/supply_chain',
		template_folder='templates',
		static_folder='static'
	)

def init_capability(appbuilder, subcapabilities: list[str]):
	"""Initialize Supply Chain Management capability with Flask-AppBuilder"""
	
	# Initialize each requested sub-capability
	for subcapability in subcapabilities:
		if subcapability == 'demand_planning':
			from .demand_planning.blueprint import init_views
			init_views(appbuilder)
		
		elif subcapability == 'logistics_transportation':
			# Logistics views would be initialized here
			pass
		
		elif subcapability == 'warehouse_management':
			# Warehouse management views would be initialized here  
			pass
		
		elif subcapability == 'supplier_relationship_management':
			# SRM views would be initialized here
			pass
	
	return True

def register_blueprints(app):
	"""Register blueprints with Flask app"""
	blueprint = create_blueprint()
	app.register_blueprint(blueprint)