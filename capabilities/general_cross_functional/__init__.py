"""
General Cross-Functional Capability

Cross-cutting capabilities that span multiple business functions including
CRM, analytics, document management, and governance systems.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'General Cross-Functional',
	'code': 'GC',
	'version': '1.0.0',
	'description': 'Cross-cutting capabilities spanning multiple business functions including CRM, analytics, and governance',
	'industry_focus': 'All Industries',
	'subcapabilities': [
		'customer_relationship_management',
		'business_intelligence_analytics',
		'enterprise_asset_management',
		'product_lifecycle_management',
		'ecommerce_b2b_b2c',
		'document_management',
		'workflow_business_process_mgmt',
		'governance_risk_compliance'
	],
	'implemented_subcapabilities': [
		'customer_relationship_management',
		'business_intelligence_analytics',
		'enterprise_asset_management',
		'product_lifecycle_management',
		'ecommerce_b2b_b2c',
		'document_management',
		'workflow_business_process_mgmt',
		'governance_risk_compliance'
	],
	'database_prefix': 'gc_',
	'menu_category': 'Cross-Functional',
	'menu_icon': 'fa-sitemap'
}

# Import implemented sub-capabilities for discovery
from . import customer_relationship_management
from . import business_intelligence_analytics
from . import enterprise_asset_management
from . import product_lifecycle_management
from . import ecommerce_b2b_b2c
from . import document_management
from . import workflow_business_process_mgmt
from . import governance_risk_compliance

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
	if 'business_intelligence_analytics' in subcapabilities and 'document_management' not in subcapabilities:
		warnings.append("Document Management is recommended for BI report storage and sharing")
	
	if 'workflow_business_process_mgmt' in subcapabilities and 'document_management' not in subcapabilities:
		warnings.append("Document Management is recommended for workflow document handling")
	
	if 'governance_risk_compliance' in subcapabilities:
		if 'document_management' not in subcapabilities:
			warnings.append("Document Management is strongly recommended for GRC policy and audit documentation")
		if 'workflow_business_process_mgmt' not in subcapabilities:
			warnings.append("Workflow/BPM is recommended for automated compliance processes")
	
	if 'enterprise_asset_management' in subcapabilities and 'workflow_business_process_mgmt' not in subcapabilities:
		warnings.append("Workflow/BPM is recommended for asset maintenance approval processes")
	
	if 'customer_relationship_management' in subcapabilities and 'business_intelligence_analytics' not in subcapabilities:
		warnings.append("Business Intelligence is recommended for CRM analytics and reporting")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def init_capability(appbuilder, subcapabilities: List[str] = None):
	"""Initialize General Cross-Functional capability with Flask-AppBuilder"""
	if subcapabilities is None:
		subcapabilities = get_implemented_subcapabilities()
	
	# Import and use blueprint initialization
	from .blueprint import init_capability
	return init_capability(appbuilder, subcapabilities)