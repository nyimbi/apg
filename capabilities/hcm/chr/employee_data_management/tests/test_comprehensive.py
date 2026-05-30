"""Compatibility coverage for the Employee Data Management package surface."""

from __future__ import annotations

from capabilities.hcm.chr.employee_data_management import EmployeeDataManagementService


def test_employee_data_management_public_service_surface_executes():
	service = EmployeeDataManagementService()

	department = service.create_department("department", "tenant-test", "HR", "Human Resources", "owner", "HR-000")
	position = service.create_position("position", "tenant-test", "HRBP", "HR Business Partner", department["id"], "professional")
	employee = service.create_employee(
		"employee",
		"tenant-test",
		"EMP-1",
		"Amina",
		"Otieno",
		"amina.otieno@example.com",
		department["id"],
		position["id"],
		"2026-01-01",
		"manager",
	)

	assert employee["department_id"] == department["id"]
	assert service.dashboard_summary("tenant-test")["employee_count"] == 1


def test_employee_data_management_legacy_aliases_point_to_public_service():
	from capabilities.hcm.chr.employee_data_management.service import (
		EmployeeDirectoryService,
		EmployeeLifecycleService,
		EmployeeProfileService,
		HCMEmployeeService,
	)

	assert EmployeeLifecycleService is EmployeeDataManagementService
	assert EmployeeProfileService is EmployeeDataManagementService
	assert EmployeeDirectoryService is EmployeeDataManagementService
	assert HCMEmployeeService is EmployeeDataManagementService
