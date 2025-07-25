"""
APG Enterprise Resource Planning (ERP) System
============================================

Complete hierarchical capability architecture providing enterprise-grade business 
management solutions across all major industries.

This package contains 12 major capabilities with 69 sub-capabilities, offering 
modular, composable ERP functionality that can be tailored to specific industry 
requirements and business needs.

Architecture Overview:
---------------------
- Core Financials: Complete financial management (8 sub-capabilities)
- Human Resources: Employee lifecycle management (7 sub-capabilities)  
- Procurement/Purchasing: Requisition-to-pay processes (5 sub-capabilities)
- Inventory Management: Real-time inventory control (4 sub-capabilities)
- Sales & Order Management: Order lifecycle management (5 sub-capabilities)
- Manufacturing: Production planning to execution (8 sub-capabilities)
- Supply Chain Management: End-to-end supply chain (4 sub-capabilities)
- Service Specific: Service industry operations (6 sub-capabilities)
- Pharmaceutical Specific: Regulatory compliance (5 sub-capabilities)
- Mining Specific: Mining operations (6 sub-capabilities)
- Platform Services: E-commerce/Marketplace (11 sub-capabilities)
- General Cross-Functional: Cross-cutting services (8 sub-capabilities)

Key Features:
------------
- Multi-tenant architecture with complete data isolation
- Industry-specific compliance (FDA, GMP, ISO, GDPR, etc.)
- Real-time business intelligence and analytics
- Workflow automation and approval processes
- REST APIs for external system integration
- Modern Python 3.12+ with type safety
- Flask-AppBuilder for automatic UI generation
- Comprehensive audit trails and security

Usage Example:
-------------
```python
from capabilities.composition import compose_application, get_industry_template

# Create a pharmaceutical ERP system
template = get_industry_template("pharmaceutical_erp")
result = compose_application(
    tenant_id="pharma_corp",
    capabilities=template.default_capabilities,
    industry_template="pharmaceutical_erp"
)

if result.success:
    app = result.flask_app
    app.run(debug=True)
```

Industry Templates:
------------------
- Manufacturing ERP: Production, quality, inventory, finance
- Pharmaceutical ERP: GMP compliance, serialization, clinical trials
- Service Company ERP: Project management, time tracking, billing
- Retail ERP: Multi-channel inventory, e-commerce, pricing
- Mining ERP: Operations, equipment, compliance, reporting
- And 8 more industry-specific templates

Integration:
-----------
Each capability provides:
- Database models with proper relationships
- Business logic services with validation
- Flask-AppBuilder views for web UI
- REST APIs for external integration
- Comprehensive permissions and security

Dependencies:
------------
- Python 3.12+
- Flask-AppBuilder 4.x
- SQLAlchemy 2.x
- Pydantic 2.x
- UUID Extensions
- Modern web stack

For complete documentation, see README.md in this directory.
"""

from typing import Dict, List, Any, Optional
import importlib
import pkgutil
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "APG Development Team"
__email__ = "apg-dev@company.com"
__license__ = "Proprietary"
__status__ = "Production"

# Package metadata
PACKAGE_INFO = {
	'name': 'APG ERP System',
	'version': __version__,
	'description': 'Enterprise Resource Planning system with hierarchical capabilities',
	'capabilities_count': 12,
	'subcapabilities_count': 69,
	'industry_templates': 13,
	'compliance_frameworks': ['FDA', 'GMP', 'GxP', 'ISO', 'GDPR', 'SOX', 'GAAP', 'IFRS'],
	'supported_industries': [
		'Manufacturing', 'Pharmaceutical', 'Service Companies', 'Retail', 
		'Mining', 'E-commerce', 'Healthcare', 'Financial Services',
		'Professional Services', 'Distribution', 'Food & Beverage'
	]
}

# Available capabilities registry
AVAILABLE_CAPABILITIES = {
	'CORE_FINANCIALS': {
		'module': 'core_financials',
		'name': 'Core Financials',
		'code': 'CF',
		'subcapabilities': 8,
		'industry_focus': 'All',
		'status': 'production'
	},
	'HUMAN_RESOURCES': {
		'module': 'human_resources',
		'name': 'Human Resources',
		'code': 'HR',
		'subcapabilities': 7,
		'industry_focus': 'All',
		'status': 'production'
	},
	'PROCUREMENT_PURCHASING': {
		'module': 'procurement_purchasing',
		'name': 'Procurement/Purchasing',
		'code': 'PP',
		'subcapabilities': 5,
		'industry_focus': 'All',
		'status': 'production'
	},
	'INVENTORY_MANAGEMENT': {
		'module': 'inventory_management',
		'name': 'Inventory Management',
		'code': 'IM',
		'subcapabilities': 4,
		'industry_focus': 'All (esp. Manufacturing, Pharma, Retail)',
		'status': 'production'
	},
	'SALES_ORDER_MANAGEMENT': {
		'module': 'sales_order_management',
		'name': 'Sales & Order Management',
		'code': 'SO',
		'subcapabilities': 5,
		'industry_focus': 'All',
		'status': 'production'
	},
	'MANUFACTURING': {
		'module': 'manufacturing',
		'name': 'Manufacturing',
		'code': 'MF',
		'subcapabilities': 8,
		'industry_focus': 'Manufacturing',
		'status': 'production'
	},
	'SUPPLY_CHAIN_MANAGEMENT': {
		'module': 'supply_chain_management',
		'name': 'Supply Chain Management',
		'code': 'SC',
		'subcapabilities': 4,
		'industry_focus': 'All',
		'status': 'production'
	},
	'SERVICE_SPECIFIC': {
		'module': 'service_specific',
		'name': 'Service Specific',
		'code': 'SS',
		'subcapabilities': 6,
		'industry_focus': 'Service, Consulting',
		'status': 'production'
	},
	'PHARMACEUTICAL_SPECIFIC': {
		'module': 'pharmaceutical_specific',
		'name': 'Pharmaceutical Specific',
		'code': 'PH',
		'subcapabilities': 5,
		'industry_focus': 'Pharmaceutical',
		'status': 'production'
	},
	'MINING_SPECIFIC': {
		'module': 'mining_specific',
		'name': 'Mining Specific',
		'code': 'MN',
		'subcapabilities': 6,
		'industry_focus': 'Mining',
		'status': 'production'
	},
	'PLATFORM_SERVICES': {
		'module': 'platform_services',
		'name': 'Platform Services (E-commerce & Marketplaces)',
		'code': 'PS',
		'subcapabilities': 11,
		'industry_focus': 'E-commerce, Marketplaces',
		'status': 'production'
	},
	'GENERAL_CROSS_FUNCTIONAL': {
		'module': 'general_cross_functional',
		'name': 'General Cross-Functional',
		'code': 'GC',
		'subcapabilities': 8,
		'industry_focus': 'All',
		'status': 'production'
	}
}

def get_package_info() -> Dict[str, Any]:
	"""Get package information and statistics"""
	return PACKAGE_INFO

def get_available_capabilities() -> Dict[str, Dict[str, Any]]:
	"""Get all available capabilities with metadata"""
	return AVAILABLE_CAPABILITIES

def get_capability_by_code(code: str) -> Optional[Dict[str, Any]]:
	"""Get capability information by code"""
	for cap_info in AVAILABLE_CAPABILITIES.values():
		if cap_info['code'] == code:
			return cap_info
	return None

def get_capabilities_by_industry(industry: str) -> List[Dict[str, Any]]:
	"""Get capabilities relevant to a specific industry"""
	relevant_caps = []
	industry_lower = industry.lower()
	
	for cap_info in AVAILABLE_CAPABILITIES.values():
		industry_focus = cap_info['industry_focus'].lower()
		if ('all' in industry_focus or 
			industry_lower in industry_focus or
			any(ind.strip().lower() in industry_focus for ind in industry.split())):
			relevant_caps.append(cap_info)
	
	return relevant_caps

def load_capability(capability_name: str):
	"""Dynamically load a capability module"""
	try:
		if capability_name.upper() not in AVAILABLE_CAPABILITIES:
			raise ValueError(f"Unknown capability: {capability_name}")
		
		module_name = AVAILABLE_CAPABILITIES[capability_name.upper()]['module']
		return importlib.import_module(f".{module_name}", package=__name__)
	except ImportError as e:
		raise ImportError(f"Failed to load capability {capability_name}: {e}")

def discover_subcapabilities(capability_name: str) -> List[str]:
	"""Discover sub-capabilities for a given capability"""
	try:
		capability_module = load_capability(capability_name)
		if hasattr(capability_module, 'get_subcapabilities'):
			return capability_module.get_subcapabilities()
		elif hasattr(capability_module, 'CAPABILITY_META'):
			return capability_module.CAPABILITY_META.get('subcapabilities', [])
		else:
			# Scan directory for sub-capabilities
			capability_path = Path(__file__).parent / capability_name.lower()
			if capability_path.exists():
				return [item.name for item in capability_path.iterdir() 
						if item.is_dir() and not item.name.startswith('__')]
	except Exception:
		pass
	return []

def validate_capability_combination(capabilities: List[str]) -> Dict[str, Any]:
	"""Validate a combination of capabilities for conflicts and dependencies"""
	validation_result = {
		'valid': True,
		'errors': [],
		'warnings': [],
		'conflicts': [],
		'missing_dependencies': []
	}
	
	# Check if all capabilities exist
	for cap in capabilities:
		if cap.upper() not in AVAILABLE_CAPABILITIES:
			validation_result['errors'].append(f"Unknown capability: {cap}")
			validation_result['valid'] = False
	
	# Industry-specific validation
	pharmaceutical_caps = [cap for cap in capabilities 
						  if 'PHARMACEUTICAL' in cap.upper()]
	manufacturing_caps = [cap for cap in capabilities 
						 if 'MANUFACTURING' in cap.upper()]
	
	if pharmaceutical_caps and not manufacturing_caps:
		validation_result['warnings'].append(
			"Pharmaceutical capabilities work best with Manufacturing capabilities"
		)
	
	# Financial integration check
	financial_dependent_caps = ['PROCUREMENT_PURCHASING', 'SALES_ORDER_MANAGEMENT', 
							   'HUMAN_RESOURCES', 'MANUFACTURING']
	has_financials = 'CORE_FINANCIALS' in [c.upper() for c in capabilities]
	
	if any(cap.upper() in financial_dependent_caps for cap in capabilities) and not has_financials:
		validation_result['warnings'].append(
			"Core Financials capability recommended for financial integration"
		)
	
	return validation_result

def get_system_statistics() -> Dict[str, Any]:
	"""Get comprehensive system statistics"""
	total_subcaps = sum(cap['subcapabilities'] for cap in AVAILABLE_CAPABILITIES.values())
	
	return {
		'capabilities': len(AVAILABLE_CAPABILITIES),
		'subcapabilities': total_subcaps,
		'industry_focus_areas': len(set(cap['industry_focus'] for cap in AVAILABLE_CAPABILITIES.values())),
		'production_ready_capabilities': len([cap for cap in AVAILABLE_CAPABILITIES.values() 
											 if cap['status'] == 'production']),
		'universal_capabilities': len([cap for cap in AVAILABLE_CAPABILITIES.values() 
									  if 'All' in cap['industry_focus']]),
		'industry_specific_capabilities': len([cap for cap in AVAILABLE_CAPABILITIES.values() 
											  if 'All' not in cap['industry_focus']]),
		'package_version': __version__,
		'compliance_frameworks': len(PACKAGE_INFO['compliance_frameworks']),
		'supported_industries': len(PACKAGE_INFO['supported_industries'])
	}

# Composition engine imports (lazy loading)
def get_composition_engine():
	"""Get the composition engine for creating custom ERP applications"""
	try:
		from .composition import (
			compose_application, discover_capabilities, validate_composition,
			get_industry_templates, get_industry_template
		)
		return {
			'compose_application': compose_application,
			'discover_capabilities': discover_capabilities,
			'validate_composition': validate_composition,
			'get_industry_templates': get_industry_templates,
			'get_industry_template': get_industry_template
		}
	except ImportError as e:
		raise ImportError(f"Composition engine not available: {e}")

# Convenience functions for common operations
def create_manufacturing_erp(tenant_id: str, **kwargs):
	"""Create a manufacturing ERP with standard capabilities"""
	engine = get_composition_engine()
	return engine['compose_application'](
		tenant_id=tenant_id,
		capabilities=['CORE_FINANCIALS', 'MANUFACTURING', 'INVENTORY_MANAGEMENT', 
					 'PROCUREMENT_PURCHASING', 'HUMAN_RESOURCES'],
		industry_template='manufacturing_erp',
		**kwargs
	)

def create_service_erp(tenant_id: str, **kwargs):
	"""Create a service company ERP with standard capabilities"""
	engine = get_composition_engine()
	return engine['compose_application'](
		tenant_id=tenant_id,
		capabilities=['CORE_FINANCIALS', 'SERVICE_SPECIFIC', 'HUMAN_RESOURCES', 
					 'GENERAL_CROSS_FUNCTIONAL'],
		industry_template='service_company_erp',
		**kwargs
	)

def create_pharmaceutical_erp(tenant_id: str, **kwargs):
	"""Create a pharmaceutical ERP with regulatory compliance"""
	engine = get_composition_engine()
	return engine['compose_application'](
		tenant_id=tenant_id,
		capabilities=['CORE_FINANCIALS', 'MANUFACTURING', 'PHARMACEUTICAL_SPECIFIC',
					 'INVENTORY_MANAGEMENT', 'PROCUREMENT_PURCHASING', 'HUMAN_RESOURCES'],
		industry_template='pharmaceutical_erp',
		**kwargs
	)

# Export main interfaces
__all__ = [
	# Package info
	'__version__', 'PACKAGE_INFO', 'AVAILABLE_CAPABILITIES',
	
	# Discovery functions
	'get_package_info', 'get_available_capabilities', 'get_capability_by_code',
	'get_capabilities_by_industry', 'discover_subcapabilities',
	
	# Validation functions
	'validate_capability_combination', 'get_system_statistics',
	
	# Composition functions
	'get_composition_engine', 'load_capability',
	
	# Convenience functions
	'create_manufacturing_erp', 'create_service_erp', 'create_pharmaceutical_erp'
]