"""
Supplier Relationship Management (SRM) Sub-Capability

Manages all aspects of interactions with suppliers, fostering collaboration
and performance optimization throughout the supply chain.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Supplier Relationship Management (SRM)',
	'code': 'SR',
	'version': '1.0.0',
	'capability': 'supply_chain_management',
	'description': 'Manages all aspects of interactions with suppliers, fostering collaboration and performance',
	'industry_focus': 'Manufacturing, Procurement, Distribution',
	'dependencies': [],
	'optional_dependencies': ['demand_planning', 'warehouse_management'],
	'database_tables': [
		'sc_sr_supplier',
		'sc_sr_supplier_contact',
		'sc_sr_performance_metric',
		'sc_sr_qualification',
		'sc_sr_contract',
		'sc_sr_risk_assessment',
		'sc_sr_collaboration_portal'
	],
	'api_endpoints': [
		'/api/supply_chain/srm/suppliers',
		'/api/supply_chain/srm/performance',
		'/api/supply_chain/srm/qualifications',
		'/api/supply_chain/srm/contracts',
		'/api/supply_chain/srm/risk_assessments'
	],
	'views': [
		'SCSRSupplierModelView',
		'SCSRPerformanceModelView',
		'SCSRQualificationModelView',
		'SCSRRiskAssessmentModelView',
		'SCSRDashboardView'
	],
	'permissions': [
		'srm.read',
		'srm.write',
		'srm.qualify',
		'srm.evaluate',
		'srm.admin'
	],
	'menu_items': [
		{
			'name': 'Suppliers',
			'endpoint': 'SCSRSupplierModelView.list',
			'icon': 'fa-industry',
			'permission': 'srm.read'
		},
		{
			'name': 'Performance',
			'endpoint': 'SCSRPerformanceModelView.list',
			'icon': 'fa-chart-line',
			'permission': 'srm.read'
		},
		{
			'name': 'Qualifications',
			'endpoint': 'SCSRQualificationModelView.list',
			'icon': 'fa-certificate',
			'permission': 'srm.read'
		},
		{
			'name': 'Risk Assessments',
			'endpoint': 'SCSRRiskAssessmentModelView.list',
			'icon': 'fa-exclamation-triangle',
			'permission': 'srm.read'
		},
		{
			'name': 'SRM Dashboard',
			'endpoint': 'SCSRDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'srm.read'
		}
	],
	'configuration': {
		'enable_supplier_portal': True,
		'auto_performance_tracking': True,
		'risk_assessment_frequency_days': 180,
		'qualification_renewal_days': 365,
		'enable_supplier_collaboration': True,
		'performance_scorecard_metrics': ['quality', 'delivery', 'cost', 'service']
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# No hard dependencies, but warn about useful integrations
	if 'demand_planning' not in available_subcapabilities:
		warnings.append("Demand Planning integration not available - supplier capacity planning may be limited")
	
	if 'warehouse_management' not in available_subcapabilities:
		warnings.append("Warehouse Management integration not available - receipt quality tracking may be limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}