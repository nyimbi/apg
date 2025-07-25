"""
Sales & Order Management Capability

Comprehensive sales and order processing system including order entry,
processing workflows, pricing management, quotations, and sales forecasting.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Sales & Order Management',
	'code': 'SO',
	'version': '1.0.0',
	'description': 'Complete sales order lifecycle management with pricing, quotations, and forecasting',
	'industry_focus': 'All',
	'subcapabilities': [
		'order_entry',
		'order_processing',
		'pricing_discounts',
		'quotations',
		'sales_forecasting'
	],
	'implemented_subcapabilities': [
		'order_entry',
		'order_processing',
		'pricing_discounts',
		'quotations',
		'sales_forecasting'
	],
	'database_prefix': 'so_',
	'menu_category': 'Sales',
	'menu_icon': 'fa-shopping-cart'
}

# Import implemented sub-capabilities for discovery
from . import order_entry
from . import order_processing
from . import pricing_discounts
from . import quotations
from . import sales_forecasting

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
	
	# Check for logical dependencies
	if 'order_processing' in subcapabilities and 'order_entry' not in subcapabilities:
		errors.append("Order Entry is required when using Order Processing")
	
	if 'sales_forecasting' in subcapabilities:
		if 'order_entry' not in subcapabilities:
			warnings.append("Order Entry is recommended for Sales Forecasting accuracy")
	
	# Check for recommended combinations
	if 'order_entry' in subcapabilities and 'pricing_discounts' not in subcapabilities:
		warnings.append("Pricing & Discounts is recommended when using Order Entry")
	
	if 'quotations' in subcapabilities and 'order_entry' not in subcapabilities:
		warnings.append("Order Entry is recommended for quote-to-order conversion")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Sales & Order Management capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)