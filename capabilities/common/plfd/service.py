"""Service layer for executable Platform Foundation management."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .foundation_runtime import (
	change_review_status,
	dependencies_are_healthy,
	normalize_baseline_type,
	normalize_health,
	normalize_score,
	normalize_tier,
	readiness_posture,
	service_baselines_complete,
	stable_id,
)
from .models import (
	FoundationBaseline,
	FoundationDependency,
	FoundationService,
	PlatformChange,
	PlfdAuditEvent,
	ReadinessAssessment,
	utc_now,
)


class PlfdService:
	"""In-process foundation-service registry, dependency, baseline, readiness, and change gate."""

	def __init__(self) -> None:
		self._services: dict[str, FoundationService] = {}
		self._dependencies: dict[str, FoundationDependency] = {}
		self._baselines: dict[str, FoundationBaseline] = {}
		self._assessments: dict[str, ReadinessAssessment] = {}
		self._changes: dict[str, PlatformChange] = {}
		self._audit_events: dict[str, PlfdAuditEvent] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_foundation_service(
		self,
		service_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		tier: str,
		dependencies: list[str] | None = None,
		readiness_score: float = 0.0,
		configuration_baseline_present: bool = True,
		health_status: str = "healthy",
		monitoring_enabled: bool = False,
		rollback_plan_ref: str = "",
		change_window_ref: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_foundation_service",
			"service_owner_assigned": bool(owner),
			"configuration_baseline_present": bool(configuration_baseline_present),
		})
		self._raise_if_denied(result)
		service = FoundationService(
			id=service_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			tier=normalize_tier(tier),
			dependencies=tuple(str(item) for item in dependencies or ()),
			readiness_score=normalize_score(readiness_score),
			configuration_baseline_present=bool(configuration_baseline_present),
			health_status=normalize_health(health_status),
			monitoring_enabled=bool(monitoring_enabled),
			rollback_plan_ref=rollback_plan_ref,
			change_window_ref=change_window_ref,
			status="registered",
			metadata=dict(metadata or {}),
		)
		self._services[service.id] = service
		self._record_audit(tenant_id, service.id, "foundation_service_registered", owner, "allow")
		return service.to_dict()

	def record_dependency(
		self,
		dependency_id: str,
		tenant_id: str,
		source_service_id: str,
		target_service_id: str,
		health_status: str = "healthy",
		required: bool = True,
		evidence_ref: str = "",
	) -> dict[str, Any]:
		self._require_service(source_service_id, tenant_id)
		self._require_service(target_service_id, tenant_id)
		dependency = FoundationDependency(
			id=dependency_id,
			tenant_id=tenant_id,
			source_service_id=source_service_id,
			target_service_id=target_service_id,
			health_status=normalize_health(health_status),
			required=bool(required),
			evidence_ref=evidence_ref,
		)
		self._dependencies[dependency.id] = dependency
		self._record_audit(tenant_id, dependency.id, "dependency_recorded", "plfd", "allow")
		return dependency.to_dict()

	def attach_baseline(
		self,
		baseline_id: str,
		tenant_id: str,
		service_id: str,
		baseline_type: str,
		evidence_ref: str,
		approved_by: str,
		status: str = "approved",
	) -> dict[str, Any]:
		self._require_service(service_id, tenant_id)
		if not evidence_ref:
			raise PermissionError("baseline_evidence_required")
		if not approved_by:
			raise PermissionError("baseline_approver_required")
		if status not in {"approved", "draft", "rejected"}:
			raise ValueError("baseline_status_invalid")
		baseline = FoundationBaseline(
			id=baseline_id,
			tenant_id=tenant_id,
			service_id=service_id,
			baseline_type=normalize_baseline_type(baseline_type),
			evidence_ref=evidence_ref,
			approved_by=approved_by,
			status=status,
		)
		self._baselines[baseline.id] = baseline
		service = self._services[service_id]
		if baseline.baseline_type == "configuration" and status == "approved":
			service.configuration_baseline_present = True
			service.updated_at = utc_now()
		self._record_audit(tenant_id, baseline.id, "baseline_attached", approved_by, status)
		return baseline.to_dict()

	def assess_readiness(self, assessment_id: str, tenant_id: str, service_id: str) -> dict[str, Any]:
		service = self._require_service(service_id, tenant_id)
		dependencies = self._service_dependency_dicts(tenant_id, service_id)
		baselines = self._service_baseline_dicts(tenant_id, service_id)
		dependencies_healthy = dependencies_are_healthy(dependencies)
		baselines_complete = service_baselines_complete(baselines)
		status, issues = readiness_posture(
			service.readiness_score,
			dependencies_healthy,
			baselines_complete,
			service.monitoring_enabled,
			bool(service.rollback_plan_ref),
			bool(service.change_window_ref),
		)
		assessment = ReadinessAssessment(
			id=assessment_id,
			tenant_id=tenant_id,
			service_id=service_id,
			score=service.readiness_score,
			status=status,
			dependencies_healthy=dependencies_healthy,
			baselines_complete=baselines_complete,
			monitoring_ready=service.monitoring_enabled,
			rollback_ready=bool(service.rollback_plan_ref),
			change_window_ready=bool(service.change_window_ref),
			issues=tuple(issues),
		)
		self._assessments[assessment.id] = assessment
		service.status = "ready" if status == "ready" else "blocked"
		service.updated_at = utc_now()
		self._record_audit(tenant_id, assessment.id, "readiness_assessed", "plfd", status, reasons=tuple(issues))
		return assessment.to_dict()

	def propose_platform_change(
		self,
		change_id: str,
		tenant_id: str,
		service_id: str,
		title: str,
		owner: str,
		affected_capability_count: int,
		dependencies_healthy: bool | None = None,
		approval_recorded: bool = False,
		broad_review_recorded: bool = False,
		security_review_recorded: bool = False,
		change_window_ref: str = "",
		rollback_plan_ref: str = "",
	) -> dict[str, Any]:
		service = self._require_service(service_id, tenant_id)
		if not owner:
			raise PermissionError("change_owner_required")
		if affected_capability_count < 1:
			raise PermissionError("affected_capability_required")
		if dependencies_healthy is None:
			dependencies_healthy = dependencies_are_healthy(self._service_dependency_dicts(tenant_id, service_id))
		change = PlatformChange(
			id=change_id,
			tenant_id=tenant_id,
			service_id=service_id,
			title=title,
			owner=owner,
			affected_capability_count=int(affected_capability_count),
			dependencies_healthy=bool(dependencies_healthy),
			approval_recorded=bool(approval_recorded),
			broad_review_recorded=bool(broad_review_recorded),
			security_review_recorded=bool(security_review_recorded),
			change_window_ref=change_window_ref or service.change_window_ref,
			rollback_plan_ref=rollback_plan_ref or service.rollback_plan_ref,
			status=change_review_status(int(affected_capability_count), bool(broad_review_recorded)),
		)
		self._changes[change.id] = change
		self._record_audit(tenant_id, change.id, "platform_change_proposed", owner, change.status)
		return change.to_dict()

	def approve_platform_change(
		self,
		change_id: str,
		tenant_id: str,
		approver: str,
		approval_recorded: bool = True,
		broad_review_recorded: bool | None = None,
		security_review_recorded: bool | None = None,
	) -> dict[str, Any]:
		change = self._require_change(change_id, tenant_id)
		service = self._require_service(change.service_id, tenant_id)
		if broad_review_recorded is not None:
			change.broad_review_recorded = bool(broad_review_recorded)
		if security_review_recorded is not None:
			change.security_review_recorded = bool(security_review_recorded)
		change.approval_recorded = bool(approval_recorded)
		change.dependencies_healthy = dependencies_are_healthy(self._service_dependency_dicts(tenant_id, service.id))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "approve_platform_change",
			"dependencies_healthy": change.dependencies_healthy,
			"approval_recorded": change.approval_recorded,
			"configuration_baseline_present": service.configuration_baseline_present,
			"affected_capability_count": change.affected_capability_count,
			"broad_review_recorded": change.broad_review_recorded,
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result)
		if not change.security_review_recorded:
			raise PermissionError("security_review_required")
		if not change.change_window_ref:
			raise PermissionError("change_window_required")
		if not change.rollback_plan_ref:
			raise PermissionError("rollback_plan_required")
		change.status = "approved"
		change.approved_at = utc_now()
		service.updated_at = utc_now()
		self._record_audit(tenant_id, change.id, "platform_change_approved", approver, "allow")
		return change.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		service = self.register_foundation_service(
			service_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "foundation-owner"),
			tier=str(metadata.get("tier") or "shared"),
			dependencies=list(metadata.get("dependencies") or []),
			readiness_score=float(metadata.get("readiness_score") or 80),
			configuration_baseline_present=bool(metadata.get("configuration_baseline_present", True)),
			health_status=str(metadata.get("health_status") or "healthy"),
			metadata=metadata | {"compatibility_status": status or "active"},
		)
		return service

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_services(tenant_id)

	def list_services(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._services, tenant_id)

	def list_dependencies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._dependencies, tenant_id)

	def list_baselines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._baselines, tenant_id)

	def list_readiness_assessments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._assessments, tenant_id)

	def list_changes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._changes, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		services = self.list_services(tenant_id)
		assessments = self.list_readiness_assessments(tenant_id)
		changes = self.list_changes(tenant_id)
		return {
			"tenant_id": tenant_id,
			"service_count": len(services),
			"core_service_count": len([item for item in services if item["tier"] == "core"]),
			"ready_service_count": len([item for item in services if item["status"] == "ready"]),
			"blocked_service_count": len([item for item in services if item["status"] == "blocked"]),
			"dependency_count": len(self.list_dependencies(tenant_id)),
			"unhealthy_dependency_count": len([item for item in self.list_dependencies(tenant_id) if item["required"] and item["health_status"] != "healthy"]),
			"baseline_count": len(self.list_baselines(tenant_id)),
			"readiness_assessment_count": len(assessments),
			"approved_change_count": len([item for item in changes if item["status"] == "approved"]),
			"pending_change_count": len([item for item in changes if item["status"] != "approved"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_service(self, service_id: str, tenant_id: str) -> FoundationService:
		service = self._services.get(service_id)
		if service is None or service.tenant_id != tenant_id:
			raise KeyError("foundation_service_not_found")
		return service

	def _require_change(self, change_id: str, tenant_id: str) -> PlatformChange:
		change = self._changes.get(change_id)
		if change is None or change.tenant_id != tenant_id:
			raise KeyError("platform_change_not_found")
		return change

	def _service_dependency_dicts(self, tenant_id: str, service_id: str) -> list[dict[str, Any]]:
		return [
			item.to_dict()
			for item in self._dependencies.values()
			if item.tenant_id == tenant_id and item.source_service_id == service_id
		]

	def _service_baseline_dicts(self, tenant_id: str, service_id: str) -> list[dict[str, Any]]:
		return [
			item.to_dict()
			for item in self._baselines.values()
			if item.tenant_id == tenant_id and item.service_id == service_id
		]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(self._reasons(result))

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		if result["decision"] == "require_review":
			raise PermissionError(self._reasons(result))

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
	) -> None:
		event_id = stable_id("plfdaudit", tenant_id, event_type, subject_id, len(self._audit_events))
		self._audit_events[event_id] = PlfdAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			actor=actor,
			decision=decision,
			reasons=reasons,
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [item for item in values if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(
			action.get("reason", "platform_foundation_policy_blocked")
			for action in result.get("actions", [])
		) or "platform_foundation_policy_blocked"
