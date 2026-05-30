"""Screen-model helpers for HCM Employee Data Management."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import EmployeeDataManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import EmployeeDataManagementService  # type: ignore


NAVIGATION = [
	{"name": "Dashboard", "route": "/hcm/employees/dashboard", "icon": "layout-dashboard"},
	{"name": "Employees", "route": "/hcm/employees", "icon": "id-card"},
	{"name": "Departments", "route": "/hcm/employees/departments", "icon": "building-2"},
	{"name": "Positions", "route": "/hcm/employees/positions", "icon": "briefcase-business"},
	{"name": "Personal Info", "route": "/hcm/employees/personal-info", "icon": "shield-user"},
	{"name": "Contacts", "route": "/hcm/employees/contacts", "icon": "phone"},
	{"name": "History", "route": "/hcm/employees/history", "icon": "history"},
	{"name": "Skills", "route": "/hcm/employees/skills", "icon": "badge-check"},
	{"name": "Certifications", "route": "/hcm/employees/certifications", "icon": "certificate"},
	{"name": "Data Quality", "route": "/hcm/employees/data-quality", "icon": "clipboard-check"},
	{"name": "Agents", "route": "/hcm/employees/agents", "icon": "bot"},
	{"name": "Settings", "route": "/hcm/employees/settings", "icon": "settings"},
]


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {"screen": screen, "tenant_id": tenant_id, "navigation": NAVIGATION}


def dashboard_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("dashboard", tenant_id)
	model["summary"] = service.dashboard_summary(tenant_id)
	model["work_queue"] = {
		"draft_profiles": len([record for record in service.employees.values() if record["tenant_id"] == tenant_id and record["status"] == "draft"]),
		"active_profiles": len([record for record in service.employees.values() if record["tenant_id"] == tenant_id and record["status"] == "active"]),
		"open_quality_issues": len([record for record in service.data_quality_issues.values() if record["tenant_id"] == tenant_id and record["status"] == "open"]),
		"missing_emergency_contacts": max(0, len(service.list_records("employees", tenant_id)) - len(service.list_records("emergency_contacts", tenant_id))),
	}
	return model


def employee_registry_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("employees", tenant_id)
	model["records"] = service.list_records("employees", tenant_id)
	model["columns"] = ["employee_number", "full_name", "work_email", "department_id", "position_id", "manager_id", "status"]
	return model


def department_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("departments", tenant_id)
	model["records"] = service.list_records("departments", tenant_id)
	model["columns"] = ["code", "name", "owner_id", "cost_center", "parent_department_id", "status"]
	return model


def position_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("positions", tenant_id)
	model["records"] = service.list_records("positions", tenant_id)
	model["columns"] = ["code", "title", "department_id", "job_level", "authorized_headcount", "status"]
	return model


def personal_info_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("personal_info", tenant_id)
	model["records"] = service.list_records("personal_info", tenant_id)
	model["columns"] = ["employee_id", "country", "effective_date", "privacy_basis", "status"]
	return model


def emergency_contact_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("contacts", tenant_id)
	model["records"] = service.list_records("emergency_contacts", tenant_id)
	model["columns"] = ["employee_id", "name", "relationship", "phone", "status"]
	return model


def employment_history_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("history", tenant_id)
	model["records"] = service.list_records("employment_history", tenant_id)
	model["columns"] = ["employee_id", "event_type", "effective_date", "reason", "approved_by", "status"]
	return model


def skill_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("skills", tenant_id)
	model["records"] = service.list_records("skills", tenant_id)
	model["columns"] = ["employee_id", "skill_name", "level", "evidence", "status"]
	return model


def certification_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("certifications", tenant_id)
	model["records"] = service.list_records("certifications", tenant_id)
	model["columns"] = ["employee_id", "name", "issuer", "issued_on", "expires_on", "status"]
	return model


def data_quality_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("quality", tenant_id)
	model["records"] = service.list_records("data_quality_issues", tenant_id)
	model["columns"] = ["domain", "severity", "description", "owner_id", "employee_id", "status"]
	return model


def agent_workbench_model(service: EmployeeDataManagementService, tenant_id: str) -> dict[str, Any]:
	model = _base("agents", tenant_id)
	model["records"] = service.list_records("agents", tenant_id)
	model["actions"] = ["review_profile", "review_quality", "review_structure", "review_skills", "review_compliance"]
	return model


class HREmployeeModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class HRDepartmentModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class HRPositionModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class HRSkillModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class HRCertificationModelView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""


class HREmployeeDashboardView:
	"""Compatibility shim for older Flask-AppBuilder view imports."""
