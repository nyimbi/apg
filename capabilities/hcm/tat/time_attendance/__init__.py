"""HCM Time and Attendance APG capability packet."""

from __future__ import annotations

from typing import Any

from .capability_contract import CAPABILITY_ID, get_capability_contract, evaluate_capability_rules
from .service import (
	AttendanceComplianceService,
	AttendanceScheduleService,
	TimeAttendanceLifecycleService,
	TimeAttendanceService,
	TimeEntryService,
)


SUBCAPABILITY_META: dict[str, Any] = {
	"name": "Time and Attendance",
	"code": "TA",
	"version": "2.1.0",
	"capability": "human_capital_management",
	"description": "Governs work policies, schedules, shifts, time entries, breaks, timesheets, leave, exceptions, payroll exports, and attendance agents.",
	"dependencies": ["employee_profile_lifecycle", "payroll_period_lifecycle", "workflow", "audl"],
	"optional_dependencies": ["device_registry", "location_policy", "privacy_policy"],
	"provides": [
		"time_policy_lifecycle",
		"work_schedule_lifecycle",
		"time_entry_lifecycle",
		"timesheet_lifecycle",
		"attendance_payroll_export",
		"attendance_agents",
	],
}


def get_subcapability_info() -> dict[str, Any]:
	"""Return package metadata for legacy HCM discovery."""
	return dict(SUBCAPABILITY_META)


def validate_dependencies(available_subcapabilities: list[str]) -> dict[str, Any]:
	"""Validate legacy HCM dependency names while the APG contract handles runtime composition."""
	errors: list[str] = []
	warnings: list[str] = []
	if "employee_data_management" not in available_subcapabilities and "employee_profile_lifecycle" not in available_subcapabilities:
		errors.append("Employee profile lifecycle is required for Time and Attendance")
	if "payroll" not in available_subcapabilities and "payroll_period_lifecycle" not in available_subcapabilities:
		warnings.append("Payroll period lifecycle is not available; payroll exports remain package-local")
	return {"valid": not errors, "errors": errors, "warnings": warnings}


__all__ = [
	"CAPABILITY_ID",
	"AttendanceComplianceService",
	"AttendanceScheduleService",
	"TimeAttendanceLifecycleService",
	"TimeAttendanceService",
	"TimeEntryService",
	"evaluate_capability_rules",
	"get_capability_contract",
	"get_subcapability_info",
	"validate_dependencies",
]
