"""
Core Financials Capability

Central repository for all financial operations including general ledger,
accounts payable/receivable, cash management, and financial reporting.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Core Financials',
	'code': 'CF',
	'version': '1.0.0',
	'description': 'Comprehensive financial management system with GL, AP, AR, and reporting',
	'industry_focus': 'All',
	'subcapabilities': [
		'general_ledger',
		'accounts_payable', 
		'accounts_receivable',
		'cash_management',
		'fixed_asset_management',
		'budgeting_forecasting',
		'financial_reporting',
		'cost_accounting'
	],
	'implemented_subcapabilities': [
		'general_ledger',
		'accounts_payable',
		'accounts_receivable',
		'cash_management',
		'fixed_asset_management',
		'budgeting_forecasting',
		'financial_reporting',
		'cost_accounting'
	],
	'database_prefix': 'cf_',
	'menu_category': 'Financials',
	'menu_icon': 'fa-dollar-sign'
}

# Import implemented sub-capabilities for discovery
from . import general_ledger
from . import accounts_payable
from . import accounts_receivable
from . import cash_management
from . import fixed_asset_management
from . import budgeting_forecasting
from . import financial_reporting
from . import cost_accounting

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
	
	# Check if at least GL is included (required for other financial modules)
	if 'general_ledger' not in subcapabilities:
		if any(sc in subcapabilities for sc in ['accounts_payable', 'accounts_receivable', 'cost_accounting']):
			errors.append("General Ledger is required when using other financial modules")
	
	# Check for recommended combinations
	if 'accounts_payable' in subcapabilities and 'cash_management' not in subcapabilities:
		warnings.append("Cash Management is recommended when using Accounts Payable")
	
	if 'accounts_receivable' in subcapabilities and 'cash_management' not in subcapabilities:
		warnings.append("Cash Management is recommended when using Accounts Receivable")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Core Financials capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)