"""Service layer for APG Environment Management."""

from __future__ import annotations

from typing import Any

from .capability_contract import DEFAULT_CONFIGURATION, evaluate_capability_rules, get_capability_contract
from .environment_engine import EnvironmentEngine
from .models import DriftReport, EnvmAuditEvent, EnvironmentDefinition, PromotionPath, PromotionRun, SecretScope


class EnvmService:
	"""Environment inventory, promotion, drift, secret-scope, and audit service."""

	def __init__(self) -> None:
		self._environments: dict[str, EnvironmentDefinition] = {}
		self._promotion_paths: dict[str, PromotionPath] = {}
		self._promotion_runs: dict[str, PromotionRun] = {}
		self._drift_reports: dict[str, DriftReport] = {}
		self._secret_scopes: dict[str, SecretScope] = {}
		self._audit_events: dict[str, EnvmAuditEvent] = {}
		self._engine = EnvironmentEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_environment(
		self,
		environment_id: str,
		tenant_id: str,
		name: str,
		stage: str,
		region: str,
		owner: str,
		configuration_source: str,
		rbac_policy: str,
		secret_scope_policy: str,
		approval_recorded: bool = True,
		status: str = "active",
	) -> dict[str, Any]:
		stage = stage.lower()
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_environment",
			"environment_owner_assigned": bool(owner),
			"environment": stage,
			"approval_recorded": bool(approval_recorded),
		})
		self._raise_if_denied(result)
		self._require_stage(stage)
		if not region:
			raise PermissionError("region_policy_required")
		if not configuration_source:
			raise PermissionError("configuration_source_required")
		if not rbac_policy:
			raise PermissionError("rbac_policy_required")
		if not secret_scope_policy:
			raise PermissionError("secret_scope_policy_required")
		fingerprint = self._engine.environment_fingerprint({
			"tenant_id": tenant_id,
			"environment_id": environment_id,
			"stage": stage,
			"region": region,
			"configuration_source": configuration_source,
			"rbac_policy": rbac_policy,
			"secret_scope_policy": secret_scope_policy,
		})
		environment = EnvironmentDefinition(
			id=environment_id,
			tenant_id=tenant_id,
			name=name,
			stage=stage,
			region=region,
			owner=owner,
			configuration_source=configuration_source,
			rbac_policy=rbac_policy,
			secret_scope_policy=secret_scope_policy,
			fingerprint=fingerprint,
			status=status,
			production_locked=stage == "production",
		)
		self._environments[environment_id] = environment
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=environment_id,
			event_type="environment_registered",
			actor=owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"stage": stage, "region": region},
		)
		return environment.to_dict()

	def create_promotion_path(
		self,
		path_id: str,
		tenant_id: str,
		source_environment_id: str,
		target_environment_id: str,
		deployment_link: str,
		rollback_environment_id: str,
		approval_recorded: bool,
		promotion_path_attached: bool = True,
	) -> dict[str, Any]:
		source = self._require_environment(source_environment_id, tenant_id)
		target = self._require_environment(target_environment_id, tenant_id)
		rollback = self._require_environment(rollback_environment_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "promote",
			"promotion_path_attached": bool(promotion_path_attached),
			"environment": target.stage,
			"approval_recorded": bool(approval_recorded),
		})
		self._raise_if_denied(result)
		if not deployment_link:
			raise PermissionError("deployment_link_required")
		if rollback.id == target.id:
			raise PermissionError("rollback_environment_must_differ")
		status = self._engine.promotion_status(approval_recorded, target.stage)
		path = PromotionPath(
			id=path_id,
			tenant_id=tenant_id,
			source_environment_id=source.id,
			target_environment_id=target.id,
			deployment_link=deployment_link,
			rollback_environment_id=rollback.id,
			approval_recorded=approval_recorded,
			status=status,
		)
		self._promotion_paths[path_id] = path
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=path_id,
			event_type="promotion_path_created",
			actor=target.owner,
			decision=result["decision"],
			metadata={"source": source.id, "target": target.id, "rollback": rollback.id},
		)
		return path.to_dict()

	def run_promotion(
		self,
		run_id: str,
		tenant_id: str,
		promotion_path_id: str,
		requested_by: str,
		artifact_ref: str,
		approval_recorded: bool,
	) -> dict[str, Any]:
		path = self._require_promotion_path(promotion_path_id, tenant_id)
		target = self._require_environment(path.target_environment_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "promote",
			"promotion_path_attached": True,
			"environment": target.stage,
			"approval_recorded": bool(approval_recorded),
		})
		self._raise_if_denied(result)
		if path.status == "blocked":
			raise PermissionError("promotion_path_blocked")
		if not artifact_ref:
			raise PermissionError("deployment_link_required")
		run = PromotionRun(
			id=run_id,
			tenant_id=tenant_id,
			promotion_path_id=promotion_path_id,
			requested_by=requested_by,
			artifact_ref=artifact_ref,
			status="promoted",
			approval_recorded=approval_recorded,
		)
		self._promotion_runs[run_id] = run
		self._record_audit(tenant_id, run_id, "environment_promoted", requested_by, result["decision"], metadata={"path_id": promotion_path_id})
		return run.to_dict()

	def record_drift(
		self,
		report_id: str,
		tenant_id: str,
		environment_id: str,
		declared_version: str,
		observed_version: str,
		changed_items: int,
		total_items: int,
		drift_review_recorded: bool = False,
		remediation_action: str = "",
	) -> dict[str, Any]:
		environment = self._require_environment(environment_id, tenant_id)
		drift_percent = self._engine.drift_percent(changed_items, total_items)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"drift_percent": drift_percent,
			"drift_review_recorded": bool(drift_review_recorded),
		})
		self._raise_if_denied(result)
		threshold = float(DEFAULT_CONFIGURATION["drift"]["drift_threshold_percent"])
		status = self._engine.drift_status(drift_percent, threshold, drift_review_recorded)
		report = DriftReport(
			id=report_id,
			tenant_id=tenant_id,
			environment_id=environment.id,
			declared_version=declared_version,
			observed_version=observed_version,
			drift_percent=drift_percent,
			changed_items=max(changed_items, 0),
			total_items=max(total_items, 0),
			status=status,
			drift_review_recorded=drift_review_recorded,
			remediation_action=remediation_action,
		)
		self._drift_reports[report_id] = report
		self._record_audit(
			tenant_id=tenant_id,
			subject_id=report_id,
			event_type="drift_recorded",
			actor=environment.owner,
			decision=result["decision"],
			reasons=tuple(action.get("reason", "") for action in result["actions"]),
			metadata={"environment_id": environment_id, "drift_percent": drift_percent},
		)
		return report.to_dict()

	def register_secret_scope(
		self,
		scope_id: str,
		tenant_id: str,
		environment_id: str,
		name: str,
		policy_ref: str,
		secret_refs: list[str] | tuple[str, ...],
		access_roles: list[str] | tuple[str, ...],
		secret_policy_attached: bool = True,
	) -> dict[str, Any]:
		environment = self._require_environment(environment_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"secret_scope_present": True,
			"secret_policy_attached": bool(secret_policy_attached and policy_ref),
		})
		self._raise_if_denied(result)
		if not secret_refs:
			raise PermissionError("secret_references_required")
		if not access_roles:
			raise PermissionError("access_roles_required")
		scope = SecretScope(
			id=scope_id,
			tenant_id=tenant_id,
			environment_id=environment.id,
			name=name,
			policy_ref=policy_ref,
			secret_refs=tuple(str(item) for item in secret_refs),
			access_roles=tuple(str(item) for item in access_roles),
		)
		self._secret_scopes[scope_id] = scope
		self._record_audit(tenant_id, scope_id, "secret_scope_registered", environment.owner, result["decision"], metadata={"environment_id": environment_id})
		return scope.to_dict()

	def list_environments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._environments, tenant_id)

	def list_promotion_paths(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._promotion_paths, tenant_id)

	def list_promotion_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._promotion_runs, tenant_id)

	def list_drift_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._drift_reports, tenant_id)

	def list_secret_scopes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._secret_scopes, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility surface exposing managed environments as records."""
		return self.list_environments(tenant_id)

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that registers an environment from metadata."""
		metadata = dict(metadata or {})
		return self.register_environment(
			environment_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			stage=str(metadata.get("stage") or "development"),
			region=str(metadata.get("region") or "local"),
			owner=str(metadata.get("owner") or "platform"),
			configuration_source=str(metadata.get("configuration_source") or "git://environment-config"),
			rbac_policy=str(metadata.get("rbac_policy") or "envm-default-rbac"),
			secret_scope_policy=str(metadata.get("secret_scope_policy") or "envm-default-secrets"),
			approval_recorded=bool(metadata.get("approval_recorded", True)),
			status=status,
		)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		environments = self.list_environments(tenant_id)
		drifts = self.list_drift_reports(tenant_id)
		return {
			"environment_count": len(environments),
			"production_environment_count": len([item for item in environments if item["stage"] == "production"]),
			"promotion_path_count": len(self.list_promotion_paths(tenant_id)),
			"promotion_run_count": len(self.list_promotion_runs(tenant_id)),
			"drift_report_count": len(drifts),
			"review_required_drift_count": len([item for item in drifts if item["status"] == "review_required"]),
			"secret_scope_count": len(self.list_secret_scopes(tenant_id)),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _list(self, values: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(values.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _require_stage(self, stage: str) -> None:
		if stage not in {"development", "test", "staging", "production"}:
			raise PermissionError("stage_policy_required")

	def _require_environment(self, environment_id: str, tenant_id: str) -> EnvironmentDefinition:
		environment = self._environments.get(environment_id)
		if environment is None or environment.tenant_id != tenant_id:
			raise KeyError(f"unknown environment: {environment_id}")
		return environment

	def _require_promotion_path(self, path_id: str, tenant_id: str) -> PromotionPath:
		path = self._promotion_paths.get(path_id)
		if path is None or path.tenant_id != tenant_id:
			raise KeyError(f"unknown promotion path: {path_id}")
		return path

	def _record_audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> EnvmAuditEvent:
		event_id = f"audit:{len(self._audit_events) + 1:06d}"
		event = EnvmAuditEvent(
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
			reasons = ", ".join(action.get("reason", "envm_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "envm_policy_blocked")
