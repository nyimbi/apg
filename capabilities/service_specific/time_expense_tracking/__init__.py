"""
Time & Expense Tracking Sub-Capability

Records and tracks billable and non-billable time and expenses
for projects and client invoicing.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Time & Expense Tracking',
	'code': 'TE',
	'version': '1.0.0',
	'capability': 'service_specific',
	'description': 'Records and tracks billable and non-billable time and expenses for projects and client invoicing',
	'industry_focus': 'Professional Services, Consulting, Legal Services',
	'dependencies': [],
	'optional_dependencies': ['project_management', 'professional_services_automation'],
	'database_tables': [
		'ss_te_timesheet',
		'ss_te_time_entry',
		'ss_te_expense_report',
		'ss_te_expense_entry',
		'ss_te_billing_rate',
		'ss_te_approval_workflow',
		'ss_te_utilization_summary'
	],
	'configuration': {
		'enable_mobile_time_entry': True,
		'require_timesheet_approval': True,
		'default_timesheet_period': 'weekly',
		'auto_calculate_overtime': True,
		'enable_expense_receipt_capture': True,
		'mileage_rate_per_mile': 0.65
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META