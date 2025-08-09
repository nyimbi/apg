"""
Human Resources Capability

Comprehensive HR management system covering employee lifecycle from recruitment
to retirement, including payroll, benefits, performance management, and compliance.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Human Resources', 
	'code': 'HR',
	'version': '1.0.0',
	'description': 'Comprehensive HR management system with employee lifecycle, payroll, benefits, and compliance',
	'industry_focus': 'All',
	'subcapabilities': [
		'pay',  # Payroll
		'tat',  # Time & Attendance
		'chr',  # Core HR/Employee Data Management
		'rec',  # Recruitment & Onboarding
		'prf',  # Performance Management
		'ben',  # Benefits Administration
		'lnd',  # Learning & Development
	],
	'implemented_subcapabilities': [
		'pay',  # Payroll
		'tat',  # Time & Attendance
		'chr',  # Core HR/Employee Data Management
		'rec',  # Recruitment & Onboarding
		'prf',  # Performance Management
		'ben',  # Benefits Administration
		'lnd',  # Learning & Development
	],
	'database_prefix': 'hr_',
	'menu_category': 'Human Resources',
	'menu_icon': 'fa-users'
}

# Import implemented sub-capabilities for discovery
from . import pay  # Payroll
from . import tat  # Time & Attendance
from . import chr  # Core HR/Employee Data Management
from . import rec  # Recruitment & Onboarding
from . import prf  # Performance Management
from . import ben  # Benefits Administration
from . import lnd  # Learning & Development

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
	
	# Check if Core HR is included (required for other HR modules)
	if 'chr' not in subcapabilities:
		if any(sc in subcapabilities for sc in ['pay', 'prf', 'ben']):
			errors.append("Core HR (chr) is required when using other HR modules")
	
	# Check for recommended combinations
	if 'pay' in subcapabilities and 'tat' not in subcapabilities:
		warnings.append("Time & Attendance (tat) is recommended when using Payroll for accurate hour tracking")
	
	if 'ben' in subcapabilities and 'pay' not in subcapabilities:
		warnings.append("Payroll (pay) integration is recommended when using Benefits Administration")
	
	if 'prf' in subcapabilities and 'lnd' not in subcapabilities:
		warnings.append("Learning & Development (lnd) integration enhances Performance Management effectiveness")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Human Resources capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)