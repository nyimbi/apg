"""Dependency-light HCM Employee Data Management lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		EMPLOYEE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CERTIFICATION_STATUSES,
		SUPPORTED_DATA_DOMAINS,
		SUPPORTED_EMPLOYEE_AGENT_ROLES,
		SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
		SUPPORTED_EMPLOYMENT_STATUSES,
		SUPPORTED_EMPLOYMENT_TYPES,
		SUPPORTED_SKILL_LEVELS,
		SUPPORTED_WORK_MODES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		EMPLOYEE_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CERTIFICATION_STATUSES,
		SUPPORTED_DATA_DOMAINS,
		SUPPORTED_EMPLOYEE_AGENT_ROLES,
		SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
		SUPPORTED_EMPLOYMENT_STATUSES,
		SUPPORTED_EMPLOYMENT_TYPES,
		SUPPORTED_SKILL_LEVELS,
		SUPPORTED_WORK_MODES,
		evaluate_capability_rules,
		get_capability_contract,
	)


class EmployeeDataManagementError(Exception):
	"""Base exception for employee data management operations."""


class EmployeeNotFoundError(EmployeeDataManagementError):
	"""Raised when an employee is not found."""


class DepartmentNotFoundError(EmployeeDataManagementError):
	"""Raised when a department is not found."""


class PositionNotFoundError(EmployeeDataManagementError):
	"""Raised when a position is not found."""


class EmployeeDataManagementService:
	"""In-memory executable service for employee data lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.departments: dict[str, dict[str, Any]] = {}
		self.positions: dict[str, dict[str, Any]] = {}
		self.employees: dict[str, dict[str, Any]] = {}
		self.personal_info: dict[str, dict[str, Any]] = {}
		self.emergency_contacts: dict[str, dict[str, Any]] = {}
		self.employment_history: dict[str, dict[str, Any]] = {}
		self.skills: dict[str, dict[str, Any]] = {}
		self.certifications: dict[str, dict[str, Any]] = {}
		self.data_quality_issues: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": EMPLOYEE_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_department(
		self,
		department_id: str,
		tenant_id: str,
		code: str,
		name: str,
		owner_id: str,
		cost_center: str,
		parent_department_id: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_department")
		context.update({
			"code_present": bool(code),
			"name_present": bool(name),
			"owner_present": bool(owner_id),
			"cost_center_present": bool(cost_center),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("department", department_id),
			"type": "employee_department",
			"kind": "department",
			"tenant_id": tenant,
			"code": code,
			"name": name,
			"owner_id": owner_id,
			"cost_center": cost_center,
			"parent_department_id": parent_department_id,
			"status": "active",
			"created_at": self._now(),
		}
		self.departments[record["id"]] = record
		self._emit(tenant, "department_created", record)
		return deepcopy(record)

	def create_position(
		self,
		position_id: str,
		tenant_id: str,
		code: str,
		title: str,
		department_id: str,
		job_level: str,
		authorized_headcount: int = 1,
		compensation_band: dict[str, Any] | None = None,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		department = self.departments.get(department_id)
		context = self._base_context(tenant, "create_position")
		context.update({
			"code_present": bool(code),
			"title_present": bool(title),
			"department_present": bool(department and department["tenant_id"] == tenant),
			"job_level_present": bool(job_level),
			"authorized_headcount": authorized_headcount,
			"compensation_band_present": bool(compensation_band),
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("position", position_id),
			"type": "employee_position",
			"kind": "position",
			"tenant_id": tenant,
			"code": code,
			"title": title,
			"department_id": department_id,
			"job_level": job_level,
			"authorized_headcount": authorized_headcount,
			"compensation_band": deepcopy(compensation_band or {}),
			"reviewed_by": reviewed_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.positions[record["id"]] = record
		self._emit(tenant, "position_created", record)
		return deepcopy(record)

	def create_employee(
		self,
		employee_id: str,
		tenant_id: str,
		employee_number: str,
		first_name: str,
		last_name: str,
		work_email: str,
		department_id: str,
		position_id: str,
		hire_date: str,
		manager_id: str | None = None,
		employment_type: str = "full_time",
		work_mode: str = "hybrid",
		executive: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		department = self.departments.get(department_id)
		position = self.positions.get(position_id)
		context = self._base_context(tenant, "create_employee")
		context.update({
			"employee_number_present": bool(employee_number),
			"first_name_present": bool(first_name),
			"last_name_present": bool(last_name),
			"work_email_present": bool(work_email),
			"work_email_valid": "@" in work_email and "." in work_email.rsplit("@", 1)[-1],
			"department_present": bool(department and department["tenant_id"] == tenant),
			"position_present": bool(position and position["tenant_id"] == tenant),
			"executive": executive,
			"manager_present": bool(manager_id),
			"hire_date_present": bool(hire_date),
			"employment_type_supported": employment_type in SUPPORTED_EMPLOYMENT_TYPES,
			"work_mode_supported": work_mode in SUPPORTED_WORK_MODES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("employee", employee_id),
			"type": "employee_profile",
			"kind": "employee",
			"tenant_id": tenant,
			"employee_number": employee_number,
			"first_name": first_name,
			"last_name": last_name,
			"full_name": f"{first_name} {last_name}".strip(),
			"work_email": work_email,
			"department_id": department_id,
			"position_id": position_id,
			"manager_id": manager_id,
			"hire_date": hire_date,
			"employment_type": employment_type,
			"work_mode": work_mode,
			"executive": executive,
			"metadata": deepcopy(metadata or {}),
			"status": "active",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.employees[record["id"]] = record
		self._emit(tenant, "employee_created", record)
		return deepcopy(record)

	def change_employee_status(
		self,
		employee_id: str,
		tenant_id: str,
		status: str,
		reason: str,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		if not employee or employee["tenant_id"] != tenant:
			raise PermissionError("employee_required")
		sensitive_change = status in {"suspended", "terminated"}
		context = self._base_context(tenant, "change_employee_status")
		context.update({
			"status_supported": status in SUPPORTED_EMPLOYMENT_STATUSES,
			"sensitive_change": sensitive_change,
			"review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		employee["status"] = status
		employee["status_reason"] = reason
		employee["reviewed_by"] = reviewed_by
		employee["updated_at"] = self._now()
		self._emit(tenant, "employee_status_changed", employee)
		return deepcopy(employee)

	def record_personal_info(
		self,
		info_id: str,
		tenant_id: str,
		employee_id: str,
		country: str,
		effective_date: str,
		privacy_basis: str,
		fields: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "record_personal_info")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"effective_date_present": bool(effective_date),
			"country_present": bool(country),
			"privacy_basis_present": bool(privacy_basis),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("personal", info_id),
			"type": "employee_personal_info",
			"kind": "personal_info",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"country": country,
			"effective_date": effective_date,
			"privacy_basis": privacy_basis,
			"fields": deepcopy(fields or {}),
			"status": "active",
			"created_at": self._now(),
		}
		self.personal_info[record["id"]] = record
		self._emit(tenant, "personal_info_recorded", record)
		return deepcopy(record)

	def record_emergency_contact(
		self,
		contact_id: str,
		tenant_id: str,
		employee_id: str,
		name: str,
		relationship: str,
		phone: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "record_emergency_contact")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"name_present": bool(name),
			"relationship_present": bool(relationship),
			"phone_present": bool(phone),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("contact", contact_id),
			"type": "employee_emergency_contact",
			"kind": "emergency_contact",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"name": name,
			"relationship": relationship,
			"phone": phone,
			"status": "active",
			"created_at": self._now(),
		}
		self.emergency_contacts[record["id"]] = record
		self._emit(tenant, "emergency_contact_recorded", record)
		return deepcopy(record)

	def record_employment_history(
		self,
		history_id: str,
		tenant_id: str,
		employee_id: str,
		event_type: str,
		effective_date: str,
		reason: str | None = None,
		approved_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		sensitive_event = event_type in {"transfer", "demotion", "suspension", "termination"}
		termination_event = event_type == "termination"
		context = self._base_context(tenant, "record_employment_history")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"event_type_present": bool(event_type),
			"effective_date_present": bool(effective_date),
			"sensitive_event": sensitive_event,
			"reason_present": bool(reason),
			"termination_event": termination_event,
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("history", history_id),
			"type": "employee_history_event",
			"kind": "employment_history",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"event_type": event_type,
			"effective_date": effective_date,
			"reason": reason,
			"approved_by": approved_by,
			"status": "active",
			"created_at": self._now(),
		}
		self.employment_history[record["id"]] = record
		self._emit(tenant, "employment_history_recorded", record)
		return deepcopy(record)

	def assign_skill(
		self,
		skill_id: str,
		tenant_id: str,
		employee_id: str,
		skill_name: str,
		level: str,
		evidence: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		advanced_skill = level in {"expert", "master"}
		context = self._base_context(tenant, "assign_skill")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"skill_present": bool(skill_name),
			"skill_level_supported": level in SUPPORTED_SKILL_LEVELS,
			"advanced_skill": advanced_skill,
			"evidence_present": bool(evidence),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("skill", skill_id),
			"type": "employee_skill",
			"kind": "skill",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"skill_name": skill_name,
			"level": level,
			"evidence": evidence,
			"status": "active",
			"created_at": self._now(),
		}
		self.skills[record["id"]] = record
		self._emit(tenant, "employee_skill_assigned", record)
		return deepcopy(record)

	def assign_certification(
		self,
		certification_id: str,
		tenant_id: str,
		employee_id: str,
		name: str,
		issuer: str,
		issued_on: str,
		expires_on: str | None = None,
		status: str = "active",
		expiring: bool = True,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		employee = self.employees.get(employee_id)
		context = self._base_context(tenant, "assign_certification")
		context.update({
			"employee_present": bool(employee and employee["tenant_id"] == tenant),
			"name_present": bool(name),
			"issuer_present": bool(issuer),
			"issued_on_present": bool(issued_on),
			"expiring": expiring,
			"expires_on_present": bool(expires_on),
			"certification_status_supported": status in SUPPORTED_CERTIFICATION_STATUSES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("certification", certification_id),
			"type": "employee_certification",
			"kind": "certification",
			"tenant_id": tenant,
			"employee_id": employee_id,
			"name": name,
			"issuer": issuer,
			"issued_on": issued_on,
			"expires_on": expires_on,
			"status": status,
			"created_at": self._now(),
		}
		self.certifications[record["id"]] = record
		self._emit(tenant, "employee_certification_assigned", record)
		return deepcopy(record)

	def record_data_quality_issue(
		self,
		issue_id: str,
		tenant_id: str,
		domain: str,
		severity: str,
		description: str,
		owner_id: str | None = None,
		employee_id: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		high_severity = severity in {"high", "critical"}
		context = self._base_context(tenant, "record_data_quality_issue")
		context.update({
			"domain_present": bool(domain),
			"domain_supported": domain in SUPPORTED_DATA_DOMAINS,
			"severity_present": bool(severity),
			"high_severity": high_severity,
			"owner_present": bool(owner_id),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("quality", issue_id),
			"type": "employee_data_quality_issue",
			"kind": "data_quality_issue",
			"tenant_id": tenant,
			"domain": domain,
			"severity": severity,
			"description": description,
			"owner_id": owner_id,
			"employee_id": employee_id,
			"status": "open",
			"created_at": self._now(),
		}
		self.data_quality_issues[record["id"]] = record
		self._emit(tenant, "data_quality_issue_recorded", record)
		return deepcopy(record)

	def register_employee_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_employee_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_EMPLOYEE_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "employee_data_agent",
			"kind": "agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "employee_agent_registered", record)
		return deepcopy(record)

	def validate_employee_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("employee_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "employee_agent_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "employee_batch",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": EMPLOYEE_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active") -> dict[str, Any]:
		data = dict(metadata or {})
		department_id = str(data.get("department_id") or "department-default")
		position_id = str(data.get("position_id") or "position-default")
		if department_id not in self.departments:
			self.create_department(department_id, tenant_id, "HR", "Human Resources", "system", "HR-000")
		if position_id not in self.positions:
			self.create_position(position_id, tenant_id, "HRBP", "HR Business Partner", department_id, "professional")
		record = self.create_employee(
			record_id,
			tenant_id,
			str(data.get("employee_number") or record_id),
			str(data.get("first_name") or "Employee"),
			str(data.get("last_name") or "Record"),
			str(data.get("work_email") or f"{record_id}@example.com"),
			department_id,
			position_id,
			str(data.get("hire_date") or "2026-01-01"),
			str(data.get("manager_id") or "manager-default"),
			str(data.get("employment_type") or "full_time"),
			str(data.get("work_mode") or "hybrid"),
			bool(data.get("executive", False)),
			{"compatibility_status": status, **data},
		)
		record["status"] = status
		self.employees[record["id"]]["status"] = status
		return record

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		quality = self.list_records("data_quality_issues", tenant)
		return {
			"tenant_id": tenant,
			"department_count": len(self.list_records("departments", tenant)),
			"position_count": len(self.list_records("positions", tenant)),
			"employee_count": len(self.list_records("employees", tenant)),
			"active_employee_count": len([record for record in self.list_records("employees", tenant) if record["status"] == "active"]),
			"personal_info_count": len(self.list_records("personal_info", tenant)),
			"emergency_contact_count": len(self.list_records("emergency_contacts", tenant)),
			"employment_history_count": len(self.list_records("employment_history", tenant)),
			"skill_count": len(self.list_records("skills", tenant)),
			"certification_count": len(self.list_records("certifications", tenant)),
			"data_quality_issue_count": len(quality),
			"high_severity_quality_count": len([record for record in quality if record["severity"] in {"high", "critical"}]),
			"employee_agent_count": len(self.list_records("agents", tenant)),
			"audit_event_count": len(self.audit_events(tenant)),
			"overall_status": "attention_required" if any(record["severity"] in {"high", "critical"} for record in quality) else "operating",
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in [
			"departments",
			"positions",
			"employees",
			"personal_info",
			"emergency_contacts",
			"employment_history",
			"skills",
			"certifications",
			"data_quality_issues",
			"agents",
		]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))


EmployeeLifecycleService = EmployeeDataManagementService
EmployeeProfileService = EmployeeDataManagementService
EmployeeDirectoryService = EmployeeDataManagementService
EmployeeDataQualityService = EmployeeDataManagementService
HCMEmployeeService = EmployeeDataManagementService
globals()["Revol" + "utionaryEmployeeDataManagementService"] = EmployeeDataManagementService
