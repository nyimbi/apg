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
		'glr',  # General Ledger
		'apy',  # Accounts Payable
		'arc',  # Accounts Receivable
		'cbm',  # Cash & Bank Management
		'fam',  # Fixed Asset Management
		'bfc',  # Budgeting & Forecasting
		'rpt',  # Reporting
		'cos',  # Cost Accounting
		'auc',  # Audit and Compliance
		'bil',  # Billing and Invoicing
		'trm',  # Treasury Management
		'txm',  # Tax Management
		'exm',  # Expense Management
		'fco',  # Financial Consolidation
		'fed',  # Federated Learning
		'dep',  # Deposit Products Engine
		'acct', # Bank Account Management
		'eod',  # EOD/BOD Processing Engine
	],
	'implemented_subcapabilities': [
		'glr',  # General Ledger
		'apy',  # Accounts Payable
		'arc',  # Accounts Receivable
		'cbm',  # Cash & Bank Management
		'fam',  # Fixed Asset Management
		'bfc',  # Budgeting & Forecasting
		'rpt',  # Reporting
		'cos',  # Cost Accounting
		'auc',  # Audit and Compliance
		'bil',  # Billing and Invoicing
		'trm',  # Treasury Management
		'txm',  # Tax Management
		'exm',  # Expense Management
		'fco',  # Financial Consolidation
		'fed',  # Federated Learning
		'dep',  # Deposit Products Engine
		'acct', # Bank Account Management
		'eod',  # EOD/BOD Processing Engine
	],
	'database_prefix': 'cf_',
	'menu_category': 'Financials',
	'menu_icon': 'fa-dollar-sign'
}

# Import implemented sub-capabilities for discovery
from . import glr  # General Ledger
from . import apy  # Accounts Payable
from . import arc  # Accounts Receivable
from . import cbm  # Cash & Bank Management
from . import fam  # Fixed Asset Management
from . import bfc  # Budgeting & Forecasting
from . import rpt  # Reporting
from . import cos  # Cost Accounting
from . import auc  # Audit and Compliance
from . import bil  # Billing and Invoicing
from . import trm  # Treasury Management
from . import txm  # Tax Management
from . import exm  # Expense Management
from . import fco  # Financial Consolidation
from . import fed  # Federated Learning
try:
	from . import dep  # Deposit Products Engine
except Exception:
	pass
try:
	from . import acct  # Bank Account Management
except Exception:
	pass
try:
	from . import eod   # EOD/BOD Processing Engine
except Exception:
	pass

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
	if 'glr' not in subcapabilities:
		if any(sc in subcapabilities for sc in ['apy', 'arc', 'cos']):
			errors.append("General Ledger (glr) is required when using other financial modules")
	
	# Check for recommended combinations
	if 'apy' in subcapabilities and 'cbm' not in subcapabilities:
		warnings.append("Cash & Bank Management (cbm) is recommended when using Accounts Payable")
	
	if 'arc' in subcapabilities and 'cbm' not in subcapabilities:
		warnings.append("Cash & Bank Management (cbm) is recommended when using Accounts Receivable")
	
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