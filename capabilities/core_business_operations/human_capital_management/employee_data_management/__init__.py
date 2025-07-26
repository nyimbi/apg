"""
Employee Data Management Sub-Capability

Centralized database for employee information including personal data, employment history,
skills, certifications, and organizational structure management.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Employee Data Management',
	'code': 'EDM',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Centralized database for employee information (personal, employment history, skills, certifications)',
	'industry_focus': 'All',
	'dependencies': [],
	'optional_dependencies': ['payroll', 'performance_management', 'benefits_administration'],
	'database_tables': [
		'hr_edm_employee',
		'hr_edm_employment_history',
		'hr_edm_personal_info',
		'hr_edm_emergency_contact',
		'hr_edm_skill',
		'hr_edm_employee_skill',
		'hr_edm_certification',
		'hr_edm_employee_certification',
		'hr_edm_department',
		'hr_edm_position',
		'hr_edm_organization_structure'
	],
	'api_endpoints': [
		'/api/human_resources/edm/employees',
		'/api/human_resources/edm/departments',
		'/api/human_resources/edm/positions',
		'/api/human_resources/edm/skills',
		'/api/human_resources/edm/certifications',
		'/api/human_resources/edm/reports'
	],
	'views': [
		'HREmployeeModelView',
		'HRDepartmentModelView',
		'HRPositionModelView',
		'HRSkillModelView',
		'HRCertificationModelView',
		'HREmployeeDashboardView'
	],
	'permissions': [
		'edm.read',
		'edm.write',
		'edm.delete',
		'edm.view_sensitive',
		'edm.manage_structure',
		'edm.admin'
	],
	'menu_items': [
		{
			'name': 'Employees',
			'endpoint': 'HREmployeeModelView.list',
			'icon': 'fa-user',
			'permission': 'edm.read'
		},
		{
			'name': 'Departments',
			'endpoint': 'HRDepartmentModelView.list',
			'icon': 'fa-building',
			'permission': 'edm.read'
		},
		{
			'name': 'Positions',
			'endpoint': 'HRPositionModelView.list',
			'icon': 'fa-briefcase',
			'permission': 'edm.read'
		},
		{
			'name': 'Skills Management',
			'endpoint': 'HRSkillModelView.list',
			'icon': 'fa-cogs',
			'permission': 'edm.read'
		},
		{
			'name': 'Certifications',
			'endpoint': 'HRCertificationModelView.list',
			'icon': 'fa-certificate',
			'permission': 'edm.read'
		}
	],
	'configuration': {
		'employee_id_format': 'EMP{:06d}',
		'require_emergency_contact': True,
		'enable_skills_tracking': True,
		'enable_certifications': True,
		'enable_photo_upload': True,
		'default_probation_period': 90,  # days
		'enable_org_chart': True
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Employee Data Management has no hard dependencies
	# But warn about useful optional dependencies
	if 'payroll' not in available_subcapabilities:
		warnings.append("Payroll integration not available - employee payroll data will not be linked")
	
	if 'performance_management' not in available_subcapabilities:
		warnings.append("Performance Management integration not available - performance data will not be linked")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_departments() -> List[Dict[str, Any]]:
	"""Get default department structure"""
	return [
		{
			'code': 'EXEC',
			'name': 'Executive',
			'description': 'Executive leadership and management',
			'parent': None
		},
		{
			'code': 'HR',
			'name': 'Human Resources',
			'description': 'Human resources and talent management',
			'parent': None
		},
		{
			'code': 'FIN',
			'name': 'Finance',
			'description': 'Financial planning, accounting, and analysis',
			'parent': None
		},
		{
			'code': 'IT',
			'name': 'Information Technology',
			'description': 'Technology infrastructure and development',
			'parent': None
		},
		{
			'code': 'OPS',
			'name': 'Operations',
			'description': 'Core business operations',
			'parent': None
		},
		{
			'code': 'SALES',
			'name': 'Sales',
			'description': 'Sales and business development',
			'parent': None
		},
		{
			'code': 'MKT',
			'name': 'Marketing',
			'description': 'Marketing and communications',
			'parent': None
		}
	]

def get_default_positions() -> List[Dict[str, Any]]:
	"""Get default position templates"""
	return [
		{
			'code': 'CEO',
			'title': 'Chief Executive Officer',
			'department_code': 'EXEC',
			'level': 'Executive',
			'description': 'Chief executive and strategic leader'
		},
		{
			'code': 'CTO',
			'title': 'Chief Technology Officer', 
			'department_code': 'IT',
			'level': 'Executive',
			'description': 'Technology strategy and leadership'
		},
		{
			'code': 'CFO',
			'title': 'Chief Financial Officer',
			'department_code': 'FIN',
			'level': 'Executive',
			'description': 'Financial strategy and oversight'
		},
		{
			'code': 'HRBP',
			'title': 'HR Business Partner',
			'department_code': 'HR',
			'level': 'Manager',
			'description': 'Human resources business partnership'
		},
		{
			'code': 'DEVMGR',
			'title': 'Development Manager',
			'department_code': 'IT',
			'level': 'Manager',
			'description': 'Software development team management'
		},
		{
			'code': 'SDEV',
			'title': 'Software Developer',
			'department_code': 'IT',
			'level': 'Individual Contributor',
			'description': 'Software development and engineering'
		}
	]