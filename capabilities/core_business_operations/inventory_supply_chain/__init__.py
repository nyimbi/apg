"""
Supply Chain Management Capability

Comprehensive supply chain orchestration including demand planning, logistics management,
warehouse operations, and supplier relationship management.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Supply Chain Management',
	'code': 'SC',
	'version': '1.0.0',
	'description': 'End-to-end supply chain orchestration with demand planning, logistics, warehouse management, and supplier relations',
	'industry_focus': 'Manufacturing, Distribution, Retail',
	'subcapabilities': [
		'demand_planning',
		'logistics_transportation',
		'warehouse_management',
		'supplier_relationship_management'
	],
	'implemented_subcapabilities': [
		'demand_planning',
		'logistics_transportation', 
		'warehouse_management',
		'supplier_relationship_management'
	],
	'database_prefix': 'sc_',
	'menu_category': 'Supply Chain',
	'menu_icon': 'fa-truck'
}

# Import implemented sub-capabilities for discovery
from . import demand_planning
from . import logistics_transportation
from . import warehouse_management
from . import supplier_relationship_management

def get_capability_info() -> Dict[str, Any]:
	"""Get capability information"""
	return CAPABILITY_META

def get_subcapabilities() -> List[str]:
	"""Get list of available sub-capabilities"""
	return CAPABILITY_META['subcapabilities']

def get_implemented_subcapabilities() -> List[str]:
	"""Get list of currently implemented sub-capabilities"""
	return CAPABILITY_META['implemented_subcapabilities']

def validate_composition(subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate a composition of sub-capabilities"""
	errors = []
	warnings = []
	
	# Check if requested sub-capabilities are implemented
	implemented = get_implemented_subcapabilities()
	for subcap in subcapabilities:
		if subcap not in CAPABILITY_META['subcapabilities']:
			errors.append(f"Unknown sub-capability: {subcap}")
		elif subcap not in implemented:
			warnings.append(f"Sub-capability '{subcap}' is not yet implemented")
	
	# Check for recommended combinations
	if 'demand_planning' in subcapabilities and 'warehouse_management' not in subcapabilities:
		warnings.append("Warehouse Management is recommended when using Demand Planning for inventory optimization")
	
	if 'logistics_transportation' in subcapabilities and 'warehouse_management' not in subcapabilities:
		warnings.append("Warehouse Management is recommended when using Logistics & Transportation for complete fulfillment")
	
	if 'supplier_relationship_management' in subcapabilities and 'demand_planning' not in subcapabilities:
		warnings.append("Demand Planning is recommended when using SRM for better supplier coordination")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Supply Chain Management capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)