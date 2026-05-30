"""Service layer for APG ESG/Carbon Tracking."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_ESGC_AGENT_ROLES,
	SUPPORTED_ESGC_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)
from .carbon_engine import CarbonEngine
from .models import (
	EmissionActivity,
	EmissionFactor,
	EmissionsInventory,
	EsgcAgent,
	EsgcAuditEvent,
	ReductionTarget,
	SustainabilityReport,
)


class EsgcService:
	"""Emissions inventory, factor library, reporting, target, and evidence service."""

	def __init__(self) -> None:
		self._inventories: dict[str, EmissionsInventory] = {}
		self._factors: dict[str, EmissionFactor] = {}
		self._activities: dict[str, EmissionActivity] = {}
		self._reports: dict[str, SustainabilityReport] = {}
		self._targets: dict[str, ReductionTarget] = {}
		self._agents: dict[str, EsgcAgent] = {}
		self._audit_events: dict[str, EsgcAuditEvent] = {}
		self._engine = CarbonEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_inventory(
		self,
		inventory_id: str,
		tenant_id: str,
		organization: str,
		owner: str,
		reporting_year: int,
		boundary_ref: str,
		geospatial_boundary: str,
		compliance_framework: str,
		status: str = "active",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_inventory",
			"organization_owner_assigned": bool(owner),
			"boundary_present": bool(boundary_ref and geospatial_boundary),
		})
		self._raise_if_denied(result)
		if not organization:
			raise PermissionError("organization_required")
		if not boundary_ref or not geospatial_boundary:
			raise PermissionError("boundary_required")
		if not compliance_framework:
			raise PermissionError("compliance_mapping_required")
		inventory = EmissionsInventory(
			id=inventory_id,
			tenant_id=tenant_id,
			organization=organization,
			owner=owner,
			reporting_year=int(reporting_year),
			boundary_ref=boundary_ref,
			geospatial_boundary=geospatial_boundary,
			compliance_framework=compliance_framework,
			status=status,
		)
		self._inventories[_state_key(tenant_id, inventory_id)] = inventory
		self._record_audit(tenant_id, inventory_id, "inventory_created", owner, result["decision"], metadata={"reporting_year": reporting_year})
		return inventory.to_dict()

	def register_factor(
		self,
		factor_id: str,
		tenant_id: str,
		name: str,
		scope: str,
		unit: str,
		co2e_per_unit: float,
		source: str,
		source_evidence: str,
		version: str,
		approved_source: bool,
		status: str = "active",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_factor",
			"factor_source_approved": bool(approved_source),
			"source_evidence_present": bool(source and source_evidence),
			"factor_version_present": bool(version),
		})
		self._raise_if_denied(result)
		self._require_scope(scope)
		factor = EmissionFactor(
			id=factor_id,
			tenant_id=tenant_id,
			name=name,
			scope=scope,
			unit=unit,
			co2e_per_unit=float(co2e_per_unit),
			source=source,
			source_evidence=source_evidence,
			version=version,
			approved_source=bool(approved_source),
			status=status,
		)
		self._factors[_state_key(tenant_id, factor_id)] = factor
		self._record_audit(tenant_id, factor_id, "factor_registered", "factor-library", result["decision"], metadata={"source": source, "version": version})
		return factor.to_dict()

	def record_activity(
		self,
		activity_id: str,
		tenant_id: str,
		inventory_id: str,
		factor_id: str,
		activity_type: str,
		quantity: float,
		unit: str,
		evidence_ref: str,
		expected_max_quantity: float | None = None,
		anomaly_review_recorded: bool = False,
	) -> dict[str, Any]:
		inventory = self._require_inventory(inventory_id, tenant_id)
		factor = self._require_factor(factor_id, tenant_id)
		if unit != factor.unit:
			raise PermissionError("activity_unit_factor_mismatch")
		anomaly_detected = self._engine.anomaly_detected(float(quantity), expected_max_quantity)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"factor_source_approved": factor.approved_source,
			"geospatial_boundary_present": bool(inventory.geospatial_boundary),
			"operation": "record_activity",
			"activity_evidence_present": bool(evidence_ref),
			"emission_anomaly_detected": anomaly_detected,
			"anomaly_review_recorded": bool(anomaly_review_recorded),
		})
		self._raise_if_denied(result)
		co2e_tonnes = self._engine.co2e_tonnes(float(quantity), factor.co2e_per_unit)
		status = "review_required" if result["decision"] == "require_review" else "recorded"
		activity = EmissionActivity(
			id=activity_id,
			tenant_id=tenant_id,
			inventory_id=inventory_id,
			factor_id=factor_id,
			activity_type=activity_type,
			scope=factor.scope,
			quantity=float(quantity),
			unit=unit,
			co2e_tonnes=co2e_tonnes,
			evidence_ref=evidence_ref,
			anomaly_detected=anomaly_detected,
			anomaly_review_recorded=bool(anomaly_review_recorded),
			status=status,
		)
		self._activities[_state_key(tenant_id, activity_id)] = activity
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=activity_id,
			event_type="emission_activity_recorded",
			actor=inventory.owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"co2e_tonnes": co2e_tonnes, "scope": factor.scope},
		)
		return activity.to_dict()

	def publish_report(
		self,
		report_id: str,
		tenant_id: str,
		inventory_id: str,
		report_type: str,
		period: str,
		compliance_mapping: str,
		audit_evidence_ref: str,
		approved_by: str,
		approval_recorded: bool,
	) -> dict[str, Any]:
		inventory = self._require_inventory(inventory_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_report",
			"approval_recorded": bool(approval_recorded),
			"compliance_mapping_present": bool(compliance_mapping),
			"audit_evidence_present": bool(audit_evidence_ref),
		})
		self._raise_if_denied(result)
		if not approved_by:
			raise PermissionError("report_approver_required")
		total = self._inventory_total(tenant_id, inventory_id)
		report = SustainabilityReport(
			id=report_id,
			tenant_id=tenant_id,
			inventory_id=inventory_id,
			report_type=report_type,
			period=period,
			total_co2e_tonnes=total,
			compliance_mapping=compliance_mapping,
			audit_evidence_ref=audit_evidence_ref,
			approved_by=approved_by,
			status="published",
		)
		self._reports[_state_key(tenant_id, report_id)] = report
		self._record_audit(tenant_id, report_id, "report_published", approved_by, result["decision"], metadata={"inventory_id": inventory.id, "total_co2e_tonnes": total})
		return report.to_dict()

	def create_target(
		self,
		target_id: str,
		tenant_id: str,
		inventory_id: str,
		name: str,
		baseline_year: int,
		target_year: int,
		baseline_co2e_tonnes: float,
		target_reduction_percent: float,
	) -> dict[str, Any]:
		self._require_inventory(inventory_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_target",
			"baseline_present": baseline_co2e_tonnes > 0 and baseline_year > 0,
		})
		self._raise_if_denied(result)
		current = self._inventory_total(tenant_id, inventory_id)
		progress = self._engine.reduction_progress_percent(float(baseline_co2e_tonnes), current, float(target_reduction_percent))
		target = ReductionTarget(
			id=target_id,
			tenant_id=tenant_id,
			inventory_id=inventory_id,
			name=name,
			baseline_year=int(baseline_year),
			target_year=int(target_year),
			baseline_co2e_tonnes=float(baseline_co2e_tonnes),
			target_reduction_percent=float(target_reduction_percent),
			current_co2e_tonnes=current,
			progress_percent=progress,
			status=self._engine.target_status(progress),
		)
		self._targets[_state_key(tenant_id, target_id)] = target
		self._record_audit(tenant_id, target_id, "target_created", "target-tracker", result["decision"], metadata={"progress_percent": progress})
		return target.to_dict()

	def list_inventories(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._inventories, tenant_id)

	def list_factors(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._factors, tenant_id)

	def list_activities(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._activities, tenant_id)

	def list_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reports, tenant_id)

	def list_targets(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._targets, tenant_id)

	def register_esgc_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"esgc_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_ESGC_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_ESGC_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		agent = EsgcAgent(
			id=agent_id or f"esgc-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._record_audit(tenant_id, agent.id, "esgc_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def list_esgc_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def validate_batch_esgc_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_esgc_mutation",
			"event_stream": event_stream,
		})

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing emissions activities as ESGC records."""
		return self.list_activities(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records an emissions activity from metadata."""
		metadata = dict(metadata or {})
		inventory_id = str(metadata.get("inventory_id") or "inventory-default")
		factor_id = str(metadata.get("factor_id") or "factor-default")
		if _state_key(tenant_id, inventory_id) not in self._inventories:
			self.create_inventory(inventory_id, tenant_id, "Default Organization", "sustainability", 2026, "boundary-default", "geo-default", "GHG Protocol", status=status)
		if _state_key(tenant_id, factor_id) not in self._factors:
			self.register_factor(factor_id, tenant_id, "Default factor", "scope_2", "kwh", 0.0004, "default-factor-library", "evidence-default", "v1", True)
		return self.record_activity(
			activity_id=record_id,
			tenant_id=tenant_id,
			inventory_id=inventory_id,
			factor_id=factor_id,
			activity_type=str(metadata.get("activity_type") or "electricity"),
			quantity=float(metadata.get("quantity") or 1.0),
			unit=str(metadata.get("unit") or "kwh"),
			evidence_ref=str(metadata.get("evidence_ref") or "evidence-default"),
		)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		activities = self.list_activities(tenant_id)
		return {
			"inventory_count": len(self.list_inventories(tenant_id)),
			"factor_count": len(self.list_factors(tenant_id)),
			"activity_count": len(activities),
			"total_co2e_tonnes": self._engine.inventory_total(activities),
			"review_required_activity_count": len([item for item in activities if item["status"] == "review_required"]),
			"report_count": len(self.list_reports(tenant_id)),
			"target_count": len(self.list_targets(tenant_id)),
			"esgc_agent_count": len(self.list_esgc_agents(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def _inventory_total(self, tenant_id: str, inventory_id: str) -> float:
		activities = [
			item for item in self.list_activities(tenant_id)
			if item["inventory_id"] == inventory_id
		]
		return self._engine.inventory_total(activities)

	def _list(self, values: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_scope(self, scope: str) -> None:
		if scope not in {"scope_1", "scope_2", "scope_3"}:
			raise PermissionError("scope_classification_required")

	def _require_inventory(self, inventory_id: str, tenant_id: str) -> EmissionsInventory:
		inventory = self._inventories.get(_state_key(tenant_id, inventory_id))
		if inventory is None or inventory.tenant_id != tenant_id:
			raise KeyError(f"unknown inventory: {inventory_id}")
		return inventory

	def _require_factor(self, factor_id: str, tenant_id: str) -> EmissionFactor:
		factor = self._factors.get(_state_key(tenant_id, factor_id))
		if factor is None or factor.tenant_id != tenant_id:
			raise KeyError(f"unknown emission factor: {factor_id}")
		return factor

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> EsgcAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = EsgcAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)
		self._audit_events[event_id] = event
		return event

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "esgc_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "esgc_policy_blocked")


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"
