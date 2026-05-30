"""Dependency-light API helpers for HCM Employee Data Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import EmployeeDataManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import EmployeeDataManagementService  # type: ignore


SERVICE = EmployeeDataManagementService()


def service() -> EmployeeDataManagementService:
	"""Return the process-local employee data service."""
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def create_department(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_department(
		payload.get("department_id", payload.get("id", "department")),
		payload["tenant_id"],
		payload["code"],
		payload["name"],
		payload["owner_id"],
		payload["cost_center"],
		payload.get("parent_department_id"),
	)


def create_position(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_position(
		payload.get("position_id", payload.get("id", "position")),
		payload["tenant_id"],
		payload["code"],
		payload["title"],
		payload["department_id"],
		payload["job_level"],
		int(payload.get("authorized_headcount", 1)),
		payload.get("compensation_band"),
		payload.get("reviewed_by"),
	)


def create_employee(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_employee(
		payload.get("employee_id", payload.get("id", "employee")),
		payload["tenant_id"],
		payload["employee_number"],
		payload["first_name"],
		payload["last_name"],
		payload["work_email"],
		payload["department_id"],
		payload["position_id"],
		payload["hire_date"],
		payload.get("manager_id"),
		payload.get("employment_type", "full_time"),
		payload.get("work_mode", "hybrid"),
		bool(payload.get("executive", False)),
		payload.get("metadata", {}),
	)


def change_employee_status(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_employee_status(
		payload["employee_id"],
		payload["tenant_id"],
		payload["status"],
		payload.get("reason", ""),
		payload.get("reviewed_by"),
	)


def record_personal_info(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_personal_info(
		payload.get("info_id", payload.get("id", "personal")),
		payload["tenant_id"],
		payload["employee_id"],
		payload["country"],
		payload["effective_date"],
		payload["privacy_basis"],
		payload.get("fields", {}),
	)


def record_emergency_contact(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_emergency_contact(
		payload.get("contact_id", payload.get("id", "contact")),
		payload["tenant_id"],
		payload["employee_id"],
		payload["name"],
		payload["relationship"],
		payload["phone"],
	)


def record_employment_history(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_employment_history(
		payload.get("history_id", payload.get("id", "history")),
		payload["tenant_id"],
		payload["employee_id"],
		payload["event_type"],
		payload["effective_date"],
		payload.get("reason"),
		payload.get("approved_by"),
	)


def assign_skill(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assign_skill(
		payload.get("skill_id", payload.get("id", "skill")),
		payload["tenant_id"],
		payload["employee_id"],
		payload["skill_name"],
		payload.get("level", "working"),
		payload.get("evidence"),
	)


def assign_certification(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.assign_certification(
		payload.get("certification_id", payload.get("id", "certification")),
		payload["tenant_id"],
		payload["employee_id"],
		payload["name"],
		payload["issuer"],
		payload["issued_on"],
		payload.get("expires_on"),
		payload.get("status", "active"),
		bool(payload.get("expiring", True)),
	)


def record_data_quality_issue(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_data_quality_issue(
		payload.get("issue_id", payload.get("id", "quality")),
		payload["tenant_id"],
		payload["domain"],
		payload["severity"],
		payload["description"],
		payload.get("owner_id"),
		payload.get("employee_id"),
	)


def register_employee_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_employee_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review employee data operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package probes."""
	return SERVICE.create_record(
		str(payload.get("id", "employee-record")),
		str(payload.get("tenant_id") or "default"),
		{
			"employee_number": payload.get("employee_number", "EMP-0001"),
			"first_name": payload.get("first_name", "Employee"),
			"last_name": payload.get("last_name", "Record"),
			"work_email": payload.get("work_email", "employee.record@example.com"),
			"hire_date": payload.get("hire_date", "2026-01-01"),
			"employment_type": payload.get("employment_type", "full_time"),
			"work_mode": payload.get("work_mode", "hybrid"),
		},
		str(payload.get("status") or "active"),
	)


def list_records(collection: str | None = None, tenant_id: str = "default") -> list[dict[str, Any]]:
	return SERVICE.list_records(collection, tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)


class HREmployeeRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class HRDepartmentRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class HRPositionRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class HRSkillRestApi:
	"""Compatibility shim for older REST endpoint registration."""


class HRCertificationRestApi:
	"""Compatibility shim for older REST endpoint registration."""


def register_api_endpoints(*_: Any, **__: Any) -> None:
	"""Compatibility hook for older Flask-AppBuilder setup code."""
	return None
