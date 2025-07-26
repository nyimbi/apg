"""
Payroll Sub-Capability

Processes employee salaries, wages, deductions, and taxes accurately and compliantly.
Includes payroll calculation, tax withholding, benefits deductions, and reporting.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Payroll',
	'code': 'PR',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Processes employee salaries, wages, deductions, and taxes accurately and compliantly',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['time_attendance', 'benefits_administration'],
	'database_tables': [
		'hr_pr_payroll_period',
		'hr_pr_payroll_run',
		'hr_pr_employee_payroll',
		'hr_pr_pay_component',
		'hr_pr_payroll_line_item',
		'hr_pr_tax_table',
		'hr_pr_tax_calculation',
		'hr_pr_deduction_type',
		'hr_pr_employee_deduction',
		'hr_pr_pay_stub',
		'hr_pr_direct_deposit',
		'hr_pr_payroll_journal'
	],
	'api_endpoints': [
		'/api/human_resources/payroll/periods',
		'/api/human_resources/payroll/runs',
		'/api/human_resources/payroll/employees',
		'/api/human_resources/payroll/pay_components',
		'/api/human_resources/payroll/deductions',
		'/api/human_resources/payroll/tax_calculations',
		'/api/human_resources/payroll/reports'
	],
	'views': [
		'HRPayrollPeriodModelView',
		'HRPayrollRunModelView',
		'HREmployeePayrollModelView',
		'HRPayComponentModelView',
		'HRDeductionTypeModelView',
		'HRPayrollDashboardView'
	],
	'permissions': [
		'payroll.read',
		'payroll.write',
		'payroll.process',
		'payroll.approve',
		'payroll.finalize',
		'payroll.view_sensitive',
		'payroll.admin'
	],
	'menu_items': [
		{
			'name': 'Payroll Dashboard',
			'endpoint': 'HRPayrollDashboardView.index',
			'icon': 'fa-dashboard',
			'permission': 'payroll.read'
		},
		{
			'name': 'Payroll Periods',
			'endpoint': 'HRPayrollPeriodModelView.list',
			'icon': 'fa-calendar',
			'permission': 'payroll.read'
		},
		{
			'name': 'Payroll Runs',
			'endpoint': 'HRPayrollRunModelView.list',
			'icon': 'fa-play-circle',
			'permission': 'payroll.read'
		},
		{
			'name': 'Employee Payroll',
			'endpoint': 'HREmployeePayrollModelView.list',
			'icon': 'fa-money-bill',
			'permission': 'payroll.read'
		},
		{
			'name': 'Pay Components',
			'endpoint': 'HRPayComponentModelView.list',
			'icon': 'fa-plus-circle',
			'permission': 'payroll.read'
		},
		{
			'name': 'Deduction Types',
			'endpoint': 'HRDeductionTypeModelView.list',
			'icon': 'fa-minus-circle',
			'permission': 'payroll.read'
		}
	],
	'configuration': {
		'default_pay_frequency': 'Monthly',
		'tax_calculation_method': 'Progressive',
		'enable_direct_deposit': True,
		'enable_payroll_journal': True,
		'require_approval_workflow': True,
		'auto_calculate_taxes': True,
		'payroll_cutoff_days': 2,
		'default_currency': 'USD'
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	# Employee Data Management is required
	if 'employee_data_management' not in available_subcapabilities:
		errors.append("Employee Data Management is required for Payroll processing")
	
	# Warn about useful optional dependencies
	if 'time_attendance' not in available_subcapabilities:
		warnings.append("Time & Attendance integration not available - hours will need manual entry")
	
	if 'benefits_administration' not in available_subcapabilities:
		warnings.append("Benefits Administration integration not available - benefit deductions need manual setup")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}

def get_default_pay_components() -> List[Dict[str, Any]]:
	"""Get default pay component types"""
	return [
		{
			'code': 'BASE_SALARY',
			'name': 'Base Salary',
			'type': 'Earnings',
			'category': 'Regular',
			'is_taxable': True,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'HOURLY_WAGE',
			'name': 'Hourly Wage',
			'type': 'Earnings',
			'category': 'Regular',
			'is_taxable': True,
			'is_recurring': True,
			'calculation_method': 'Hours_x_Rate'
		},
		{
			'code': 'OVERTIME',
			'name': 'Overtime Pay',
			'type': 'Earnings',
			'category': 'Overtime',
			'is_taxable': True,
			'is_recurring': False,
			'calculation_method': 'Hours_x_Rate'
		},
		{
			'code': 'BONUS',
			'name': 'Bonus',
			'type': 'Earnings',
			'category': 'Bonus',
			'is_taxable': True,
			'is_recurring': False,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'COMMISSION',
			'name': 'Commission',
			'type': 'Earnings',
			'category': 'Commission',
			'is_taxable': True,
			'is_recurring': False,
			'calculation_method': 'Percentage'
		},
		{
			'code': 'FEDERAL_TAX',
			'name': 'Federal Income Tax',
			'type': 'Tax',
			'category': 'Federal',
			'is_taxable': False,
			'is_recurring': True,
			'calculation_method': 'Tax_Table'
		},
		{
			'code': 'STATE_TAX',
			'name': 'State Income Tax',
			'type': 'Tax',
			'category': 'State',
			'is_taxable': False,
			'is_recurring': True,
			'calculation_method': 'Tax_Table'
		},
		{
			'code': 'SOCIAL_SECURITY',
			'name': 'Social Security',
			'type': 'Tax',
			'category': 'FICA',
			'is_taxable': False,
			'is_recurring': True,
			'calculation_method': 'Percentage'
		},
		{
			'code': 'MEDICARE',
			'name': 'Medicare',
			'type': 'Tax',
			'category': 'FICA',
			'is_taxable': False,
			'is_recurring': True,
			'calculation_method': 'Percentage' 
		}
	]

def get_default_deduction_types() -> List[Dict[str, Any]]:
	"""Get default deduction types"""
	return [
		{
			'code': 'HEALTH_INS',
			'name': 'Health Insurance',
			'category': 'Benefits',
			'is_pre_tax': True,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'DENTAL_INS',
			'name': 'Dental Insurance',
			'category': 'Benefits',
			'is_pre_tax': True,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': '401K',
			'name': '401(k) Contribution',
			'category': 'Retirement',
			'is_pre_tax': True,
			'is_recurring': True,
			'calculation_method': 'Percentage'
		},
		{
			'code': 'LIFE_INS',
			'name': 'Life Insurance',
			'category': 'Benefits',
			'is_pre_tax': True,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'PARKING',
			'name': 'Parking Fee',
			'category': 'Other',
			'is_pre_tax': False,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'UNION_DUES',
			'name': 'Union Dues',
			'category': 'Other',
			'is_pre_tax': False,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		},
		{
			'code': 'GARNISHMENT',
			'name': 'Wage Garnishment',
			'category': 'Legal',
			'is_pre_tax': False,
			'is_recurring': True,
			'calculation_method': 'Fixed'
		}
	]