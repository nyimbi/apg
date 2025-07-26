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
		'payroll',
		'time_attendance',
		'employee_data_management',
		'recruitment_onboarding',
		'performance_management',
		'benefits_administration',
		'learning_development'
	],
	'implemented_subcapabilities': [
		'payroll',
		'time_attendance', 
		'employee_data_management',
		'recruitment_onboarding',
		'performance_management',
		'benefits_administration',
		'learning_development'
	],
	'database_prefix': 'hr_',
	'menu_category': 'Human Resources',
	'menu_icon': 'fa-users'
}

# Import implemented sub-capabilities for discovery
from . import payroll
from . import time_attendance
from . import employee_data_management
from . import recruitment_onboarding
from . import performance_management
from . import benefits_administration
from . import learning_development

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
	
	# Check if Employee Data Management is included (required for other HR modules)
	if 'employee_data_management' not in subcapabilities:
		if any(sc in subcapabilities for sc in ['payroll', 'performance_management', 'benefits_administration']):
			errors.append("Employee Data Management is required when using other HR modules")
	
	# Check for recommended combinations
	if 'payroll' in subcapabilities and 'time_attendance' not in subcapabilities:
		warnings.append("Time & Attendance is recommended when using Payroll for accurate hour tracking")
	
	if 'benefits_administration' in subcapabilities and 'payroll' not in subcapabilities:
		warnings.append("Payroll integration is recommended when using Benefits Administration")
	
	if 'performance_management' in subcapabilities and 'learning_development' not in subcapabilities:
		warnings.append("Learning & Development integration enhances Performance Management effectiveness")
	
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