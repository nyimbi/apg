"""
Service Specific Capability

Specialized functionality for service-based businesses including project management,
resource scheduling, professional services automation, and field service management.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Service Specific',
	'code': 'SS',
	'version': '1.0.0',
	'description': 'Specialized functionality for service-based businesses including project management, resource scheduling, and field services',
	'industry_focus': 'Professional Services, Field Services, Consulting',
	'subcapabilities': [
		'project_management',
		'resource_scheduling',
		'time_expense_tracking',
		'service_contract_management',
		'field_service_management',
		'professional_services_automation'
	],
	'implemented_subcapabilities': [
		'project_management',
		'resource_scheduling',
		'time_expense_tracking',
		'service_contract_management',
		'field_service_management',
		'professional_services_automation'
	],
	'database_prefix': 'ss_',
	'menu_category': 'Services',
	'menu_icon': 'fa-briefcase'
}

# Import implemented sub-capabilities for discovery
from . import project_management
from . import resource_scheduling
from . import time_expense_tracking
from . import service_contract_management
from . import field_service_management
from . import professional_services_automation

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
	if 'project_management' in subcapabilities and 'time_expense_tracking' not in subcapabilities:
		warnings.append("Time & Expense Tracking is recommended when using Project Management for accurate project costing")
	
	if 'resource_scheduling' in subcapabilities and 'project_management' not in subcapabilities:
		warnings.append("Project Management is recommended when using Resource Scheduling for better resource allocation")
	
	if 'professional_services_automation' in subcapabilities:
		missing_deps = []
		for dep in ['project_management', 'time_expense_tracking', 'resource_scheduling']:
			if dep not in subcapabilities:
				missing_deps.append(dep)
		if missing_deps:
			warnings.append(f"PSA works best with: {', '.join(missing_deps)}")
	
	if 'field_service_management' in subcapabilities and 'service_contract_management' not in subcapabilities:
		warnings.append("Service Contract Management is recommended for field service warranty and SLA tracking")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize Service Specific capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)