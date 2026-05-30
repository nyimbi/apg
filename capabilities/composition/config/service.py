"""Service layer for the central configuration capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_CONFIG_AGENT_ROLES,
	SUPPORTED_CONFIG_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
	streaming_manifest,
)
from .models import (
	ConfigAgentRecord,
	ConfigAuditEventRecord,
	ConfigDeploymentRecord,
	ConfigDriftRecord,
	ConfigNamespaceRecord,
	ConfigTemplateRecord,
	ConfigurationRecord,
	stable_id,
	utc_now,
)


class CompositionConfigService:
	"""Dependency-light configuration runtime behind the capability contract."""

	def __init__(self) -> None:
		self._namespaces: dict[str, ConfigNamespaceRecord] = {}
		self._configurations: dict[str, ConfigurationRecord] = {}
		self._deployments: dict[str, ConfigDeploymentRecord] = {}
		self._templates: dict[str, ConfigTemplateRecord] = {}
		self._drift_records: dict[str, ConfigDriftRecord] = {}
		self._agents: dict[str, ConfigAgentRecord] = {}
		self._audit_events: list[ConfigAuditEventRecord] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_namespace(
		self,
		namespace_key: str,
		tenant_id: str,
		name: str,
		environment: str,
		owner_id: str,
		path_prefix: str,
		capability_id: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_namespace",
			"namespace_owner_assigned": bool(owner_id),
			"environment_present": bool(environment),
		})
		if not path_prefix.startswith("/"):
			raise ValueError("namespace_path_prefix_must_start_with_slash")
		namespace_id = stable_id("config_namespace", tenant_id, namespace_key)
		record = ConfigNamespaceRecord(
			id=namespace_id,
			tenant_id=tenant_id,
			name=name,
			environment=environment,
			owner_id=owner_id,
			path_prefix=path_prefix,
			capability_id=capability_id,
			metadata=dict(metadata or {}),
		)
		self._namespaces[namespace_id] = record
		self._audit(tenant_id, "namespace_registered", namespace_id, owner_id, {"environment": environment, "capability_id": capability_id})
		return record.to_dict()

	def create_configuration(
		self,
		config_key: str,
		tenant_id: str,
		namespace_id: str,
		key_path: str,
		value: dict[str, Any],
		owner_id: str,
		restricted: bool = False,
		secret: bool = False,
		schema: dict[str, Any] | None = None,
		secret_reference: str | None = None,
		policy_attached: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		namespace = self._get_namespace(namespace_id)
		if namespace.tenant_id != tenant_id:
			raise ValueError("namespace_tenant_mismatch")
		if not key_path.startswith(namespace.path_prefix):
			raise ValueError("configuration_key_outside_namespace")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_configuration",
			"restricted_config": bool(restricted),
			"schema_present": bool(schema),
			"secret_config": bool(secret),
			"secret_reference_present": bool(secret_reference),
		})
		config_id = stable_id("configuration", tenant_id, namespace_id, config_key)
		record = ConfigurationRecord(
			id=config_id,
			tenant_id=tenant_id,
			namespace_id=namespace_id,
			key_path=key_path,
			value=dict(value),
			version=1,
			owner_id=owner_id,
			restricted=restricted,
			secret=secret,
			schema=schema,
			secret_reference=secret_reference,
			metadata=dict(metadata or {}),
		)
		self._configurations[config_id] = record
		self._audit(tenant_id, "configuration_created", config_id, owner_id, {"key_path": key_path, "restricted": restricted, "secret": secret})
		return record.to_dict()

	def validate_configuration(self, configuration_id: str, actor_id: str, evidence: str) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		if not evidence:
			raise PermissionError("configuration_validation_evidence_required")
		config.validation_evidence = evidence
		config.status = "validated"
		config.updated_at = utc_now()
		self._audit(config.tenant_id, "configuration_validated", configuration_id, actor_id, {"evidence": evidence})
		return config.to_dict()

	def activate_configuration(self, configuration_id: str, actor_id: str, validation_evidence: str | None = None) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		next_evidence = validation_evidence or config.validation_evidence
		self._enforce_context({
			"tenant_context_present": bool(config.tenant_id),
			"operation": "activate_configuration",
			"validation_evidence_present": bool(next_evidence),
		})
		config.validation_evidence = next_evidence
		config.status = "active"
		config.updated_at = utc_now()
		self._audit(config.tenant_id, "configuration_activated", configuration_id, actor_id, {"version": config.version})
		return config.to_dict()

	def update_configuration(
		self,
		configuration_id: str,
		actor_id: str,
		value: dict[str, Any],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		self._enforce_context({"tenant_context_present": bool(config.tenant_id), "operation_type": "write", "policy_attached": policy_attached})
		config.value = dict(value)
		config.version += 1
		config.status = "draft"
		config.validation_evidence = None
		config.updated_at = utc_now()
		self._audit(config.tenant_id, "configuration_updated", configuration_id, actor_id, {"version": config.version})
		return config.to_dict()

	def deploy_configuration(
		self,
		deployment_key: str,
		tenant_id: str,
		configuration_id: str,
		environment: str,
		impact_level: str,
		actor_id: str,
		approved_by: str | None = None,
		canary_evidence: str | None = None,
		event_stream: str = "bytewax",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		if config.tenant_id != tenant_id:
			raise ValueError("configuration_tenant_mismatch")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deploy_configuration",
			"environment": environment,
			"approval_recorded": bool(approved_by),
			"impact_level": impact_level,
			"canary_evidence_present": bool(canary_evidence),
			"event_stream": event_stream,
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review" and not canary_evidence:
			raise PermissionError(",".join(result["matched_rules"]))
		if config.status != "active":
			raise PermissionError("configuration_must_be_active_before_deploy")
		deployment_id = stable_id("config_deployment", tenant_id, deployment_key)
		record = ConfigDeploymentRecord(
			id=deployment_id,
			tenant_id=tenant_id,
			configuration_id=configuration_id,
			environment=environment,
			impact_level=impact_level,
			status="deployed",
			approved_by=approved_by,
			canary_evidence=canary_evidence,
			event_stream=event_stream,
			metadata=dict(metadata or {}),
		)
		self._deployments[deployment_id] = record
		self._audit(tenant_id, "configuration_deployed", deployment_id, actor_id, {"configuration_id": configuration_id, "environment": environment})
		return record.to_dict()

	def rollback_configuration(
		self,
		deployment_id: str,
		actor_id: str,
		reason: str,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		deployment = self._get_deployment(deployment_id)
		self._enforce_context({
			"tenant_context_present": bool(deployment.tenant_id),
			"operation": "rollback_configuration",
			"rollback_reason_present": bool(reason),
			"event_stream": event_stream,
		})
		deployment.status = "rolled_back"
		self._audit(deployment.tenant_id, "configuration_rolled_back", deployment_id, actor_id, {"reason": reason, "event_stream": event_stream})
		return deployment.to_dict()

	def create_template(
		self,
		template_key: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		values: dict[str, Any],
		variable_schema: dict[str, Any],
		shared: bool = False,
		reviewed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_template",
			"shared_template": bool(shared),
			"review_recorded": bool(reviewed_by),
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review" and not reviewed_by:
			raise PermissionError(",".join(result["matched_rules"]))
		if not owner_id:
			raise PermissionError("template_owner_required")
		if not variable_schema:
			raise PermissionError("template_variable_schema_required")
		template_id = stable_id("config_template", tenant_id, template_key)
		record = ConfigTemplateRecord(
			id=template_id,
			tenant_id=tenant_id,
			name=name,
			owner_id=owner_id,
			values=dict(values),
			variable_schema=dict(variable_schema),
			shared=shared,
			reviewed_by=reviewed_by,
			metadata=dict(metadata or {}),
		)
		self._templates[template_id] = record
		self._audit(tenant_id, "template_created", template_id, owner_id, {"shared": shared})
		return record.to_dict()

	def record_drift(
		self,
		tenant_id: str,
		configuration_id: str,
		expected_version: int,
		observed_version: int,
		severity: str,
		actor_id: str = "system",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		if config.tenant_id != tenant_id:
			raise ValueError("configuration_tenant_mismatch")
		drift_id = stable_id("config_drift", tenant_id, configuration_id, str(observed_version))
		record = ConfigDriftRecord(
			id=drift_id,
			tenant_id=tenant_id,
			configuration_id=configuration_id,
			expected_version=expected_version,
			observed_version=observed_version,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self._drift_records[drift_id] = record
		self._audit(tenant_id, "drift_detected", drift_id, actor_id, {"configuration_id": configuration_id, "severity": severity})
		return record.to_dict()

	def register_config_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		instructions: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_config_agent",
			"agent_runtime_supported": runtime in SUPPORTED_CONFIG_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_CONFIG_AGENT_ROLES,
		})
		agent_id = stable_id("config_agent", tenant_id, name, runtime, role)
		record = ConfigAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime,
			role=role,
			instructions=instructions,
			metadata=dict(metadata or {}),
		)
		self._agents[agent_id] = record
		self._audit(tenant_id, "config_agent_registered", agent_id, name, {"runtime": runtime, "role": role})
		return record.to_dict()

	def validate_agent_config_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		agent = self._get_agent(agent_id)
		if agent.tenant_id != tenant_id:
			raise ValueError("agent_tenant_mismatch")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_config_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		return {"tenant_id": tenant_id, "agent_id": agent_id, "action": action, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def validate_batch_configuration_change(self, tenant_id: str, change_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation": "batch_configuration_change",
			"event_stream": event_stream,
		})
		return {"tenant_id": tenant_id, "change_count": change_count, "event_stream": event_stream, "stream": event_stream_name(), "processor": "bytewax"}

	def list_namespaces(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._namespaces, tenant_id)

	def list_configurations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._configurations, tenant_id)

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates, tenant_id)

	def list_drift_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._drift_records, tenant_id)

	def list_config_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [event.to_dict() for event in self._audit_events if tenant_id is None or event.tenant_id == tenant_id]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"namespace_count": len(self.list_namespaces(tenant_id)),
			"configuration_count": len(self.list_configurations(tenant_id)),
			"deployment_count": len(self.list_deployments(tenant_id)),
			"template_count": len(self.list_templates(tenant_id)),
			"drift_count": len(self.list_drift_records(tenant_id)),
			"config_agent_count": len(self.list_config_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		namespace = self.register_namespace(
			namespace_key=f"{record_id}-namespace",
			tenant_id=tenant_id,
			name=str((metadata or {}).get("namespace") or "Default Namespace"),
			environment=str((metadata or {}).get("environment") or "development"),
			owner_id=str((metadata or {}).get("owner_id") or "system"),
			path_prefix=str((metadata or {}).get("path_prefix") or "/default"),
			capability_id="composition_config",
		)
		config = self.create_configuration(
			config_key=record_id,
			tenant_id=tenant_id,
			namespace_id=namespace["id"],
			key_path=f"{namespace['path_prefix']}/{record_id}",
			value=dict(metadata or {}),
			owner_id=str((metadata or {}).get("owner_id") or "system"),
			policy_attached=policy_attached,
		)
		config["status"] = status
		return config

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_configurations(tenant_id)

	def _enforce_context(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, actor_id: str, metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append(
			ConfigAuditEventRecord(
				id=stable_id("config_audit", tenant_id, event_type, entity_id, str(len(self._audit_events))),
				tenant_id=tenant_id,
				event_type=event_type,
				entity_id=entity_id,
				actor_id=actor_id,
				metadata=dict(metadata or {}),
			)
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		return [record.to_dict() for record in records.values() if tenant_id is None or record.tenant_id == tenant_id]

	def _get_namespace(self, namespace_id: str) -> ConfigNamespaceRecord:
		try:
			return self._namespaces[namespace_id]
		except KeyError as exc:
			raise KeyError(f"unknown_namespace:{namespace_id}") from exc

	def _get_configuration(self, configuration_id: str) -> ConfigurationRecord:
		try:
			return self._configurations[configuration_id]
		except KeyError as exc:
			raise KeyError(f"unknown_configuration:{configuration_id}") from exc

	def _get_deployment(self, deployment_id: str) -> ConfigDeploymentRecord:
		try:
			return self._deployments[deployment_id]
		except KeyError as exc:
			raise KeyError(f"unknown_deployment:{deployment_id}") from exc

	def _get_agent(self, agent_id: str) -> ConfigAgentRecord:
		try:
			return self._agents[agent_id]
		except KeyError as exc:
			raise KeyError(f"unknown_config_agent:{agent_id}") from exc


CentralConfigurationService = CompositionConfigService

__all__ = [
	"CentralConfigurationService",
	"CompositionConfigService",
	"ConfigAgentRecord",
	"ConfigAuditEventRecord",
	"ConfigDeploymentRecord",
	"ConfigDriftRecord",
	"ConfigNamespaceRecord",
	"ConfigTemplateRecord",
	"ConfigurationRecord",
]
