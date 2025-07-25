"""
Inventory Management Capability

Comprehensive inventory management system with real-time tracking, automated
replenishment, batch/lot traceability, and expiry date management for 
enterprise-grade operations across manufacturing, pharmaceutical, and retail industries.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Inventory Management', 
	'code': 'IM',
	'version': '1.0.0',
	'description': 'Enterprise inventory management with real-time tracking, automated replenishment, batch/lot traceability, and expiry management',
	'industry_focus': 'Manufacturing, Pharmaceutical, Food & Beverage, Retail',
	'subcapabilities': [
		'stock_tracking_control',
		'replenishment_reordering',
		'batch_lot_tracking', 
		'expiry_date_management'
	],
	'implemented_subcapabilities': [
		'stock_tracking_control',
		'replenishment_reordering',
		'batch_lot_tracking',
		'expiry_date_management'
	],
	'database_prefix': 'im_',
	'menu_category': 'Inventory',
	'menu_icon': 'fa-boxes'
}

# Import implemented sub-capabilities for discovery
from . import stock_tracking_control
from . import replenishment_reordering
from . import batch_lot_tracking
from . import expiry_date_management

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
	
	# Check if stock tracking is included (foundational for other modules)
	if 'stock_tracking_control' not in subcapabilities:
		if any(sc in subcapabilities for sc in ['replenishment_reordering', 'batch_lot_tracking', 'expiry_date_management']):
			errors.append("Stock Tracking & Control is required when using other inventory modules")
	
	# Check for recommended combinations for pharmaceutical/food industries
	if 'batch_lot_tracking' in subcapabilities and 'expiry_date_management' not in subcapabilities:
		warnings.append("Expiry Date Management is recommended when using Batch & Lot Tracking for compliance")
	
	if 'expiry_date_management' in subcapabilities and 'batch_lot_tracking' not in subcapabilities:
		warnings.append("Batch & Lot Tracking is recommended when using Expiry Date Management for full traceability")
	
	# Check for replenishment automation recommendations
	if 'stock_tracking_control' in subcapabilities and 'replenishment_reordering' not in subcapabilities:
		warnings.append("Replenishment & Reordering is recommended to automate inventory replenishment based on stock levels")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Inventory Management capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)