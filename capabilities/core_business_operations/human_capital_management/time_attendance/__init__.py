"""
Time & Attendance Sub-Capability

Tracks employee work hours, overtime, and absenteeism for accurate payroll 
and resource planning. Includes time tracking, scheduling, and attendance reporting.
"""

from typing import Dict, List, Any

# Sub-capability metadata
SUBCAPABILITY_META = {
	'name': 'Time & Attendance',
	'code': 'TA',
	'version': '1.0.0',
	'capability': 'human_resources',
	'description': 'Tracks employee work hours, overtime, and absenteeism for accurate payroll and resource planning',
	'industry_focus': 'All',
	'dependencies': ['employee_data_management'],
	'optional_dependencies': ['payroll'],
	'database_tables': [
		'hr_ta_time_entry',
		'hr_ta_schedule',
		'hr_ta_shift_template',
		'hr_ta_attendance_record',
		'hr_ta_leave_request',
		'hr_ta_overtime_request',
		'hr_ta_time_off_type',
		'hr_ta_work_calendar',
		'hr_ta_time_clock',
		'hr_ta_timesheet'
	],
	'api_endpoints': [
		'/api/human_resources/time_attendance/time_entries',
		'/api/human_resources/time_attendance/schedules',
		'/api/human_resources/time_attendance/attendance',
		'/api/human_resources/time_attendance/leave_requests',
		'/api/human_resources/time_attendance/timesheets',
		'/api/human_resources/time_attendance/reports'
	],
	'views': [
		'HRTimeEntryModelView',
		'HRScheduleModelView',
		'HRAttendanceModelView',
		'HRLeaveRequestModelView',
		'HRTimesheetModelView',
		'HRTimeAttendanceDashboardView'
	],
	'permissions': [
		'time_attendance.read',
		'time_attendance.write',
		'time_attendance.approve',
		'time_attendance.manage_schedules',
		'time_attendance.view_reports',
		'time_attendance.admin'
	],
	'configuration': {
		'enable_time_clock': True,
		'enable_mobile_tracking': True,
		'require_manager_approval': True,
		'overtime_threshold_hours': 40,
		'grace_period_minutes': 15,
		'auto_break_deduction': True,
		'enable_geofencing': False,
		'default_work_hours_per_day': 8
	}
}

def get_subcapability_info() -> Dict[str, Any]:
	"""Get sub-capability information"""
	return SUBCAPABILITY_META

def validate_dependencies(available_subcapabilities: List[str]) -> Dict[str, Any]:
	"""Validate dependencies are met"""
	errors = []
	warnings = []
	
	if 'employee_data_management' not in available_subcapabilities:
		errors.append("Employee Data Management is required for Time & Attendance")
	
	if 'payroll' not in available_subcapabilities:
		warnings.append("Payroll integration not available - time data will not auto-sync to payroll")
	
	return {
		'valid': len(errors) == 0,
		'errors': errors,
		'warnings': warnings
	}