"""
Mining Specific Capability

Industry-specific ERP functionality for mining and mineral extraction companies.
Manages mine planning, equipment management, grade control, tenement management, and production reporting.
"""

from typing import Dict, List, Any

# Capability metadata
CAPABILITY_META = {
	'name': 'Mining Specific',
	'code': 'MN',
	'version': '1.0.0',
	'description': 'Industry-specific ERP functionality for mining and mineral extraction companies',
	'industry_focus': 'Mining & Minerals',
	'regulatory_frameworks': ['MSHA', 'OSHA', 'EPA', 'SEC', 'JORC', 'NI 43-101'],
	'dependencies': [
		'core_financials',
		'inventory_management',
		'manufacturing',
		'audit_compliance',
		'auth_rbac'
	],
	'optional_dependencies': [
		'supply_chain_management',
		'human_resources',
		'procurement_purchasing',
		'enterprise_asset_management'
	],
	'subcapabilities': [
		'mine_planning_optimization',
		'equipment_fleet_management',
		'grade_control_blending',
		'tenement_management_system',
		'weighbridge_integration',
		'production_reporting'
	],
	'database_tables_prefix': 'mn_',
	'api_prefix': '/api/mining',
	'permissions_prefix': 'mn.',
	'configuration': {
		'geological_modeling': True,
		'resource_estimation': True,
		'mine_scheduling': True,
		'equipment_tracking': True,
		'grade_monitoring': True,
		'environmental_compliance': True,
		'safety_management': True,
		'production_optimization': True
	}
}

def get_capability_info() -> Dict[str, Any]:
	"""Get mining capability information"""
	return CAPABILITY_META

def validate_dependencies(available_capabilities: List[str]) -> Dict[str, Any]:
	"""Validate that required dependencies are available"""
	errors = []
	warnings = []
	
	# Check required dependencies
	required = CAPABILITY_META['dependencies']
	for dep in required:
		if dep not in available_capabilities:
			errors.append(f"Required dependency '{dep}' not available")
	
	# Check optional dependencies
	optional = CAPABILITY_META['optional_dependencies']
	for dep in optional:
		if dep not in available_capabilities:
			warnings.append(f"Optional dependency '{dep}' not available - some features may be limited")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_mining_operations() -> List[Dict[str, Any]]:
	"""Get supported mining operations"""
	return [
		{
			'operation_type': 'Open Pit',
			'description': 'Surface mining operations',
			'typical_equipment': ['Excavators', 'Haul Trucks', 'Dozers', 'Drills'],
			'planning_horizon': '10-30 years'
		},
		{
			'operation_type': 'Underground',
			'description': 'Subsurface mining operations',
			'typical_equipment': ['LHDs', 'Jumbo Drills', 'Trucks', 'Ventilation'],
			'planning_horizon': '5-20 years'
		},
		{
			'operation_type': 'Processing Plant',
			'description': 'Mineral processing and beneficiation',
			'typical_equipment': ['Crushers', 'Mills', 'Flotation', 'Filters'],
			'planning_horizon': '1-5 years'
		},
		{
			'operation_type': 'Tailings',
			'description': 'Tailings storage and management',
			'typical_equipment': ['Pumps', 'Pipelines', 'Thickeners', 'Dams'],
			'planning_horizon': '20-50 years'
		}
	]

def get_commodity_types() -> List[Dict[str, Any]]:
	"""Get supported commodity types"""
	return [
		{
			'commodity': 'Gold',
			'grade_units': 'g/t',
			'typical_cutoff': '0.5 g/t',
			'recovery_method': 'Cyanide Leaching'
		},
		{
			'commodity': 'Copper',
			'grade_units': '%',
			'typical_cutoff': '0.2%',
			'recovery_method': 'Flotation'
		},
		{
			'commodity': 'Iron Ore',
			'grade_units': '% Fe',
			'typical_cutoff': '55% Fe',
			'recovery_method': 'Dense Media Separation'
		},
		{
			'commodity': 'Coal',
			'grade_units': 'Ash %',
			'typical_cutoff': '35% Ash',
			'recovery_method': 'Washing'
		},
		{
			'commodity': 'Zinc',
			'grade_units': '%',
			'typical_cutoff': '1.0%',
			'recovery_method': 'Flotation'
		},
		{
			'commodity': 'Nickel',
			'grade_units': '%',
			'typical_cutoff': '0.8%',
			'recovery_method': 'Flotation/Smelting'
		}
	]

def get_equipment_categories() -> List[Dict[str, Any]]:
	"""Get mining equipment categories"""
	return [
		{
			'category': 'Loading',
			'equipment_types': ['Hydraulic Excavator', 'Wheel Loader', 'LHD'],
			'key_metrics': ['Bucket Capacity', 'Cycle Time', 'Availability']
		},
		{
			'category': 'Hauling',
			'equipment_types': ['Haul Truck', 'Articulated Truck', 'Conveyor'],
			'key_metrics': ['Payload', 'Speed', 'Fuel Consumption']
		},
		{
			'category': 'Drilling',
			'equipment_types': ['Blast Hole Drill', 'Development Drill', 'Production Drill'],
			'key_metrics': ['Penetration Rate', 'Hole Diameter', 'Accuracy']
		},
		{
			'category': 'Support',
			'equipment_types': ['Dozer', 'Grader', 'Water Cart', 'Service Truck'],
			'key_metrics': ['Operating Hours', 'Fuel Usage', 'Maintenance Cost']
		},
		{
			'category': 'Processing',
			'equipment_types': ['Crusher', 'Mill', 'Flotation Cell', 'Thickener'],
			'key_metrics': ['Throughput', 'Recovery', 'Power Consumption']
		}
	]

def get_safety_requirements() -> List[Dict[str, Any]]:
	"""Get mining safety requirements"""
	return [
		{
			'requirement_id': 'MN-SAF-001',
			'name': 'Personal Protective Equipment',
			'description': 'Mandatory PPE for all mining personnel',
			'category': 'Personnel Safety',
			'compliance_framework': 'MSHA'
		},
		{
			'requirement_id': 'MN-SAF-002',
			'name': 'Equipment Safety Inspections',
			'description': 'Pre-shift safety inspections for all equipment',
			'category': 'Equipment Safety',
			'compliance_framework': 'MSHA'
		},
		{
			'requirement_id': 'MN-SAF-003',
			'name': 'Gas Monitoring',
			'description': 'Continuous monitoring of hazardous gases',
			'category': 'Environmental Safety',
			'compliance_framework': 'MSHA'
		},
		{
			'requirement_id': 'MN-SAF-004',
			'name': 'Blast Safety Procedures',
			'description': 'Safe handling and detonation of explosives',
			'category': 'Explosives Safety',
			'compliance_framework': 'ATF/MSHA'
		},
		{
			'requirement_id': 'MN-SAF-005',
			'name': 'Emergency Response Planning',
			'description': 'Comprehensive emergency response procedures',
			'category': 'Emergency Management',
			'compliance_framework': 'MSHA/OSHA'
		}
	]

def get_environmental_controls() -> List[Dict[str, Any]]:
	"""Get mining environmental controls"""
	return [
		{
			'control_id': 'MN-ENV-001',
			'name': 'Water Quality Monitoring',
			'description': 'Continuous monitoring of water discharge quality',
			'category': 'Water Management',
			'regulatory_framework': 'EPA'
		},
		{
			'control_id': 'MN-ENV-002',
			'name': 'Air Quality Control',
			'description': 'Dust suppression and air quality monitoring',
			'category': 'Air Quality',
			'regulatory_framework': 'EPA'
		},
		{
			'control_id': 'MN-ENV-003',
			'name': 'Noise Level Management',
			'description': 'Control and monitoring of operational noise levels',
			'category': 'Noise Control',
			'regulatory_framework': 'EPA/Local'
		},
		{
			'control_id': 'MN-ENV-004',
			'name': 'Waste Rock Management',
			'description': 'Proper storage and management of waste rock',
			'category': 'Waste Management',
			'regulatory_framework': 'EPA'
		},
		{
			'control_id': 'MN-ENV-005',
			'name': 'Rehabilitation Planning',
			'description': 'Progressive and final rehabilitation of mined areas',
			'category': 'Land Rehabilitation',
			'regulatory_framework': 'State/Federal'
		}
	]