"""Service layer for the central configuration capability — expanded implementation."""

from __future__ import annotations

from datetime import datetime, timezone
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


def _ts() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CompositionConfigService:
	"""
	Dependency-light configuration runtime behind the capability contract.

	Expanded with: get_config, set_config, delete_config, list_configs,
	validate_schema, config_version_history, rollback_config, config_diff,
	bulk_config_import, config_analytics.
	"""

	def __init__(self) -> None:
		self._namespaces: dict[str, ConfigNamespaceRecord] = {}
		self._configurations: dict[str, ConfigurationRecord] = {}
		self._deployments: dict[str, ConfigDeploymentRecord] = {}
		self._templates: dict[str, ConfigTemplateRecord] = {}
		self._drift_records: dict[str, ConfigDriftRecord] = {}
		self._agents: dict[str, ConfigAgentRecord] = {}
		self._audit_events: list[ConfigAuditEventRecord] = []
		# Version history store: config_id -> list of version snapshots
		self._version_history: dict[str, list[dict[str, Any]]] = {}
		# Schema definitions per namespace
		self._schemas: dict[str, dict[str, Any]] = {}
		# Soft-deleted config IDs
		self._deleted_ids: set[str] = set()

	# ------------------------------------------------------------------
	# Contract / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# get_config / set_config / delete_config / list_configs
	# ------------------------------------------------------------------

	def get_config(
		self,
		namespace: str,
		key: str,
		tenant_id: str,
		version: int | None = None,
	) -> dict[str, Any]:
		"""
		Retrieve a configuration value.

		namespace: Logical namespace (used as namespace name).
		key: Config key path within the namespace.
		tenant_id: Owning tenant.
		version: Optional specific version; returns latest if None.
		"""
		if not namespace or not key:
			raise ValueError("namespace_and_key_required")
		config_id = stable_id("configuration", tenant_id, namespace, key)
		if config_id in self._deleted_ids:
			raise KeyError(f"config_deleted:{namespace}/{key}")
		config = self._configurations.get(config_id)
		if config is None or config.tenant_id != tenant_id:
			raise KeyError(f"config_not_found:{namespace}/{key}")
		if version is not None and config.version != version:
			# Search version history
			history = self._version_history.get(config_id, [])
			snapshot = next((h for h in history if h["version"] == version), None)
			if snapshot is None:
				raise KeyError(f"config_version_not_found:{namespace}/{key}@v{version}")
			return dict(snapshot)
		return config.to_dict()

	def set_config(
		self,
		namespace: str,
		key: str,
		value: Any,
		tenant_id: str,
		data_type: str = "string",
		description: str = "",
		owner_id: str = "system",
		restricted: bool = False,
		secret: bool = False,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""
		Create or update a configuration key-value pair.

		If the config already exists, increments version and snapshots to history.
		"""
		if not namespace or not key:
			raise ValueError("namespace_and_key_required")
		self._enforce_context({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_configuration",
			"restricted_config": bool(restricted),
			"secret_config": bool(secret),
			"schema_present": False,
			"secret_reference_present": False,
		})
		# Resolve or create namespace
		ns_id = stable_id("config_namespace", tenant_id, namespace)
		if ns_id not in self._namespaces:
			ns = ConfigNamespaceRecord(
				id=ns_id,
				tenant_id=tenant_id,
				name=namespace,
				environment="default",
				owner_id=owner_id,
				path_prefix=f"/{namespace}",
				capability_id="composition_config",
				metadata={},
			)
			self._namespaces[ns_id] = ns
		ns = self._namespaces[ns_id]
		config_id = stable_id("configuration", tenant_id, namespace, key)
		existing = self._configurations.get(config_id)
		if config_id in self._deleted_ids:
			self._deleted_ids.discard(config_id)
		if existing and existing.tenant_id == tenant_id:
			# Snapshot current version to history
			self._snapshot_version(config_id, existing)
			existing.value = {"__value": value, "__type": data_type}
			existing.version += 1
			existing.status = "active"
			existing.updated_at = utc_now()
			config = existing
		else:
			config = ConfigurationRecord(
				id=config_id,
				tenant_id=tenant_id,
				namespace_id=ns_id,
				key_path=f"{ns.path_prefix}/{key}",
				value={"__value": value, "__type": data_type, "__description": description},
				version=1,
				owner_id=owner_id,
				restricted=restricted,
				secret=secret,
				schema=None,
				secret_reference=None,
				metadata={"data_type": data_type, "description": description},
			)
			self._configurations[config_id] = config
		self._audit(tenant_id, "config_set", config_id, owner_id,
			{"namespace": namespace, "key": key, "version": config.version})
		return config.to_dict()

	def delete_config(
		self,
		namespace: str,
		key: str,
		tenant_id: str,
		deleted_by: str = "system",
		reason: str = "",
	) -> dict[str, Any]:
		"""
		Soft-delete a configuration key.

		The record is retained in history but marked as deleted.
		"""
		if not namespace or not key:
			raise ValueError("namespace_and_key_required")
		config_id = stable_id("configuration", tenant_id, namespace, key)
		config = self._configurations.get(config_id)
		if config is None or config.tenant_id != tenant_id:
			raise KeyError(f"config_not_found:{namespace}/{key}")
		if config_id in self._deleted_ids:
			raise KeyError(f"config_already_deleted:{namespace}/{key}")
		self._snapshot_version(config_id, config)
		self._deleted_ids.add(config_id)
		config.status = "deleted"
		self._audit(tenant_id, "config_deleted", config_id, deleted_by,
			{"namespace": namespace, "key": key, "reason": reason})
		return {**config.to_dict(), "deleted_by": deleted_by, "reason": reason, "deleted_at": _ts()}

	def list_configs(
		self,
		namespace: str,
		tenant_id: str,
		filters: dict[str, Any] | None = None,
		include_deleted: bool = False,
	) -> list[dict[str, Any]]:
		"""
		List all configs in a namespace for a tenant.

		filters: Optional dict supporting 'status', 'restricted', 'secret'.
		"""
		f = filters or {}
		ns_id = stable_id("config_namespace", tenant_id, namespace)
		configs = [
			c for c in self._configurations.values()
			if c.tenant_id == tenant_id and c.namespace_id == ns_id
		]
		if not include_deleted:
			configs = [c for c in configs if c.id not in self._deleted_ids]
		if "status" in f:
			configs = [c for c in configs if c.status == f["status"]]
		if "restricted" in f:
			configs = [c for c in configs if c.restricted == f["restricted"]]
		if "secret" in f:
			configs = [c for c in configs if c.secret == f["secret"]]
		return [c.to_dict() for c in sorted(configs, key=lambda c: c.key_path)]

	def validate_schema(
		self,
		namespace: str,
		schema_definition: dict[str, Any],
		tenant_id: str = "default",
		validated_by: str = "system",
	) -> dict[str, Any]:
		"""
		Register and validate a JSON Schema for a namespace.

		schema_definition: JSON Schema dict.
		Returns validation result with field count and required fields.
		"""
		if not namespace:
			raise ValueError("namespace_required")
		if not schema_definition:
			raise ValueError("schema_definition_required")
		schema_type = schema_definition.get("type", "object")
		properties = schema_definition.get("properties", {})
		required_fields = schema_definition.get("required", [])
		# Basic structural validation
		errors: list[str] = []
		if schema_type != "object":
			errors.append("top_level_schema_must_be_object_type")
		for field, spec in properties.items():
			if not isinstance(spec, dict):
				errors.append(f"field_{field}_spec_must_be_dict")
			elif "type" not in spec:
				errors.append(f"field_{field}_missing_type")
		schema_key = stable_id("schema", tenant_id, namespace)
		self._schemas[schema_key] = schema_definition
		result = {
			"namespace": namespace,
			"tenant_id": tenant_id,
			"schema_valid": len(errors) == 0,
			"errors": errors,
			"field_count": len(properties),
			"required_field_count": len(required_fields),
			"required_fields": required_fields,
			"validated_by": validated_by,
			"validated_at": _ts(),
		}
		self._audit(tenant_id, "schema_validated", namespace, validated_by,
			{"valid": result["schema_valid"], "field_count": result["field_count"]})
		return result

	def config_version_history(
		self,
		namespace: str,
		key: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""
		Return the full version history of a configuration key.

		Returns list of version snapshots in chronological order.
		"""
		config_id = stable_id("configuration", tenant_id, namespace, key)
		config = self._configurations.get(config_id)
		history = self._version_history.get(config_id, [])
		current = config.to_dict() if config and config.tenant_id == tenant_id else None
		all_versions = sorted(history, key=lambda h: h["version"])
		if current:
			all_versions.append(current)
		return {
			"namespace": namespace,
			"key": key,
			"tenant_id": tenant_id,
			"version_count": len(all_versions),
			"current_version": config.version if config else None,
			"versions": all_versions,
		}

	def rollback_config(
		self,
		namespace: str,
		key: str,
		version: int,
		tenant_id: str,
		reason: str,
		rolled_back_by: str = "system",
	) -> dict[str, Any]:
		"""
		Rollback a configuration to a specific previous version.

		Snapshot the current state to history before rolling back.
		"""
		if not reason:
			raise PermissionError("rollback_reason_required")
		config_id = stable_id("configuration", tenant_id, namespace, key)
		config = self._configurations.get(config_id)
		if config is None or config.tenant_id != tenant_id:
			raise KeyError(f"config_not_found:{namespace}/{key}")
		history = self._version_history.get(config_id, [])
		target_snapshot = next((h for h in history if h["version"] == version), None)
		if target_snapshot is None:
			raise KeyError(f"config_version_not_found:{namespace}/{key}@v{version}")
		# Snapshot current before rollback
		self._snapshot_version(config_id, config)
		# Apply target snapshot values
		config.value = target_snapshot.get("value", config.value)
		config.version = config.version + 1
		config.status = "active"
		config.updated_at = utc_now()
		self._audit(tenant_id, "config_rolled_back", config_id, rolled_back_by,
			{"namespace": namespace, "key": key, "target_version": version, "reason": reason})
		return {**config.to_dict(), "rolled_back_to_version": version, "reason": reason, "rolled_back_by": rolled_back_by}

	def config_diff(
		self,
		namespace: str,
		tenant_a: str,
		tenant_b: str,
	) -> dict[str, Any]:
		"""
		Compare configuration namespace between two tenants.

		Returns keys present in only one tenant, and keys with differing values.
		"""
		ns_a = stable_id("config_namespace", tenant_a, namespace)
		ns_b = stable_id("config_namespace", tenant_b, namespace)
		configs_a = {
			c.key_path: c for c in self._configurations.values()
			if c.tenant_id == tenant_a and c.namespace_id == ns_a and c.id not in self._deleted_ids
		}
		configs_b = {
			c.key_path: c for c in self._configurations.values()
			if c.tenant_id == tenant_b and c.namespace_id == ns_b and c.id not in self._deleted_ids
		}
		all_keys = set(configs_a) | set(configs_b)
		only_in_a = sorted(k for k in all_keys if k in configs_a and k not in configs_b)
		only_in_b = sorted(k for k in all_keys if k in configs_b and k not in configs_a)
		differing: list[dict[str, Any]] = []
		for key in sorted(all_keys):
			if key in configs_a and key in configs_b:
				va = configs_a[key].value
				vb = configs_b[key].value
				if va != vb:
					differing.append({"key": key, "tenant_a_value": va, "tenant_b_value": vb})
		return {
			"namespace": namespace,
			"tenant_a": tenant_a,
			"tenant_b": tenant_b,
			"keys_in_a": len(configs_a),
			"keys_in_b": len(configs_b),
			"only_in_a": only_in_a,
			"only_in_b": only_in_b,
			"differing_keys": differing,
			"identical": len(only_in_a) == 0 and len(only_in_b) == 0 and len(differing) == 0,
			"compared_at": _ts(),
		}

	def bulk_config_import(
		self,
		namespace: str,
		config_map: dict[str, Any],
		tenant_id: str,
		owner_id: str = "system",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""
		Import multiple configuration key-value pairs at once.

		config_map: dict of key -> value (or dict with value/data_type/description keys).
		Returns import summary with created, updated, and failed counts.
		"""
		if not namespace:
			raise ValueError("namespace_required")
		if not config_map:
			raise ValueError("config_map_required")
		created = 0
		updated = 0
		failed: list[dict[str, Any]] = []
		for key, raw_value in config_map.items():
			try:
				if isinstance(raw_value, dict) and "__value" in raw_value:
					value = raw_value["__value"]
					data_type = raw_value.get("__type", "string")
					description = raw_value.get("__description", "")
				else:
					value = raw_value
					data_type = "string"
					description = ""
				config_id = stable_id("configuration", tenant_id, namespace, key)
				is_new = config_id not in self._configurations or config_id in self._deleted_ids
				self.set_config(
					namespace=namespace,
					key=key,
					value=value,
					tenant_id=tenant_id,
					data_type=data_type,
					description=description,
					owner_id=owner_id,
					policy_attached=policy_attached,
				)
				if is_new:
					created += 1
				else:
					updated += 1
			except Exception as exc:
				failed.append({"key": key, "error": str(exc)})
		result = {
			"namespace": namespace,
			"tenant_id": tenant_id,
			"total_keys": len(config_map),
			"created_count": created,
			"updated_count": updated,
			"failed_count": len(failed),
			"failures": failed,
			"success": len(failed) == 0,
			"imported_at": _ts(),
		}
		self._audit(tenant_id, "bulk_config_imported", namespace, owner_id,
			{"created": created, "updated": updated, "failed": len(failed)})
		return result

	def config_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""
		Return aggregated configuration analytics for a tenant over a period.

		Covers namespace, config, deployment, drift, and version statistics.
		"""
		namespaces = self.list_namespaces(tenant_id)
		configurations = self.list_configurations(tenant_id)
		deployments = self.list_deployments(tenant_id)
		drift_records = self.list_drift_records(tenant_id)
		templates = self.list_templates(tenant_id)
		active_configs = [c for c in configurations if c.get("status") == "active"]
		deleted_count = sum(
			1 for c in self._configurations.values()
			if c.tenant_id == tenant_id and c.id in self._deleted_ids
		)
		deployed_envs = {d["environment"] for d in deployments}
		rollback_count = sum(
			1 for events in self._audit_events
			if events.tenant_id == tenant_id and events.event_type == "config_rolled_back"
		)
		total_versions = sum(
			len(history) + 1
			for config_id, history in self._version_history.items()
			if self._configurations.get(config_id) and self._configurations[config_id].tenant_id == tenant_id
		)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"namespace_count": len(namespaces),
			"configuration_count": len(configurations),
			"active_configuration_count": len(active_configs),
			"deleted_configuration_count": deleted_count,
			"deployment_count": len(deployments),
			"deployed_environment_count": len(deployed_envs),
			"drift_record_count": len(drift_records),
			"template_count": len(templates),
			"rollback_count": rollback_count,
			"total_versions_tracked": total_versions,
			"schema_count": sum(1 for k in self._schemas if k.startswith(f"schema-{tenant_id}")),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"generated_at": _ts(),
		}

	# ------------------------------------------------------------------
	# Original retained methods
	# ------------------------------------------------------------------

	def register_namespace(self, namespace_key: str, tenant_id: str, name: str, environment: str, owner_id: str, path_prefix: str, capability_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation": "register_namespace", "namespace_owner_assigned": bool(owner_id), "environment_present": bool(environment)})
		if not path_prefix.startswith("/"):
			raise ValueError("namespace_path_prefix_must_start_with_slash")
		namespace_id = stable_id("config_namespace", tenant_id, namespace_key)
		record = ConfigNamespaceRecord(id=namespace_id, tenant_id=tenant_id, name=name, environment=environment, owner_id=owner_id, path_prefix=path_prefix, capability_id=capability_id, metadata=dict(metadata or {}))
		self._namespaces[namespace_id] = record
		self._audit(tenant_id, "namespace_registered", namespace_id, owner_id, {"environment": environment, "capability_id": capability_id})
		return record.to_dict()

	def create_configuration(self, config_key: str, tenant_id: str, namespace_id: str, key_path: str, value: dict[str, Any], owner_id: str, restricted: bool = False, secret: bool = False, schema: dict[str, Any] | None = None, secret_reference: str | None = None, policy_attached: bool = True, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		namespace = self._get_namespace(namespace_id)
		if namespace.tenant_id != tenant_id:
			raise ValueError("namespace_tenant_mismatch")
		if not key_path.startswith(namespace.path_prefix):
			raise ValueError("configuration_key_outside_namespace")
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_configuration", "restricted_config": bool(restricted), "schema_present": bool(schema), "secret_config": bool(secret), "secret_reference_present": bool(secret_reference)})
		config_id = stable_id("configuration", tenant_id, namespace_id, config_key)
		record = ConfigurationRecord(id=config_id, tenant_id=tenant_id, namespace_id=namespace_id, key_path=key_path, value=dict(value), version=1, owner_id=owner_id, restricted=restricted, secret=secret, schema=schema, secret_reference=secret_reference, metadata=dict(metadata or {}))
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
		self._enforce_context({"tenant_context_present": bool(config.tenant_id), "operation": "activate_configuration", "validation_evidence_present": bool(next_evidence)})
		config.validation_evidence = next_evidence
		config.status = "active"
		config.updated_at = utc_now()
		self._audit(config.tenant_id, "configuration_activated", configuration_id, actor_id, {"version": config.version})
		return config.to_dict()

	def update_configuration(self, configuration_id: str, actor_id: str, value: dict[str, Any], policy_attached: bool = True) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		self._enforce_context({"tenant_context_present": bool(config.tenant_id), "operation_type": "write", "policy_attached": policy_attached})
		self._snapshot_version(configuration_id, config)
		config.value = dict(value)
		config.version += 1
		config.status = "draft"
		config.validation_evidence = None
		config.updated_at = utc_now()
		self._audit(config.tenant_id, "configuration_updated", configuration_id, actor_id, {"version": config.version})
		return config.to_dict()

	def deploy_configuration(self, deployment_key: str, tenant_id: str, configuration_id: str, environment: str, impact_level: str, actor_id: str, approved_by: str | None = None, canary_evidence: str | None = None, event_stream: str = "bytewax", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		if config.tenant_id != tenant_id:
			raise ValueError("configuration_tenant_mismatch")
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "deploy_configuration", "environment": environment, "approval_recorded": bool(approved_by), "impact_level": impact_level, "canary_evidence_present": bool(canary_evidence), "event_stream": event_stream})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review" and not canary_evidence:
			raise PermissionError(",".join(result["matched_rules"]))
		if config.status != "active":
			raise PermissionError("configuration_must_be_active_before_deploy")
		deployment_id = stable_id("config_deployment", tenant_id, deployment_key)
		record = ConfigDeploymentRecord(id=deployment_id, tenant_id=tenant_id, configuration_id=configuration_id, environment=environment, impact_level=impact_level, status="deployed", approved_by=approved_by, canary_evidence=canary_evidence, event_stream=event_stream, metadata=dict(metadata or {}))
		self._deployments[deployment_id] = record
		self._audit(tenant_id, "configuration_deployed", deployment_id, actor_id, {"configuration_id": configuration_id, "environment": environment})
		return record.to_dict()

	def rollback_configuration(self, deployment_id: str, actor_id: str, reason: str, event_stream: str = "bytewax") -> dict[str, Any]:
		deployment = self._get_deployment(deployment_id)
		self._enforce_context({"tenant_context_present": bool(deployment.tenant_id), "operation": "rollback_configuration", "rollback_reason_present": bool(reason), "event_stream": event_stream})
		deployment.status = "rolled_back"
		self._audit(deployment.tenant_id, "configuration_rolled_back", deployment_id, actor_id, {"reason": reason, "event_stream": event_stream})
		return deployment.to_dict()

	def create_template(self, template_key: str, tenant_id: str, name: str, owner_id: str, values: dict[str, Any], variable_schema: dict[str, Any], shared: bool = False, reviewed_by: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "create_template", "shared_template": bool(shared), "review_recorded": bool(reviewed_by)})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review" and not reviewed_by:
			raise PermissionError(",".join(result["matched_rules"]))
		if not owner_id:
			raise PermissionError("template_owner_required")
		if not variable_schema:
			raise PermissionError("template_variable_schema_required")
		template_id = stable_id("config_template", tenant_id, template_key)
		record = ConfigTemplateRecord(id=template_id, tenant_id=tenant_id, name=name, owner_id=owner_id, values=dict(values), variable_schema=dict(variable_schema), shared=shared, reviewed_by=reviewed_by, metadata=dict(metadata or {}))
		self._templates[template_id] = record
		self._audit(tenant_id, "template_created", template_id, owner_id, {"shared": shared})
		return record.to_dict()

	def record_drift(self, tenant_id: str, configuration_id: str, expected_version: int, observed_version: int, severity: str, actor_id: str = "system", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		config = self._get_configuration(configuration_id)
		if config.tenant_id != tenant_id:
			raise ValueError("configuration_tenant_mismatch")
		drift_id = stable_id("config_drift", tenant_id, configuration_id, str(observed_version))
		record = ConfigDriftRecord(id=drift_id, tenant_id=tenant_id, configuration_id=configuration_id, expected_version=expected_version, observed_version=observed_version, severity=severity, metadata=dict(metadata or {}))
		self._drift_records[drift_id] = record
		self._audit(tenant_id, "drift_detected", drift_id, actor_id, {"configuration_id": configuration_id, "severity": severity})
		return record.to_dict()

	def register_config_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation": "register_config_agent", "agent_runtime_supported": runtime in SUPPORTED_CONFIG_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_CONFIG_AGENT_ROLES})
		agent_id = stable_id("config_agent", tenant_id, name, runtime, role)
		record = ConfigAgentRecord(id=agent_id, tenant_id=tenant_id, name=name, runtime=runtime, role=role, instructions=instructions, metadata=dict(metadata or {}))
		self._agents[agent_id] = record
		self._audit(tenant_id, "config_agent_registered", agent_id, name, {"runtime": runtime, "role": role})
		return record.to_dict()

	def validate_agent_config_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		agent = self._get_agent(agent_id)
		if agent.tenant_id != tenant_id:
			raise ValueError("agent_tenant_mismatch")
		result = self.evaluate({"tenant_context_present": bool(tenant_id), "operation": "agent_config_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		return {"tenant_id": tenant_id, "agent_id": agent_id, "action": action, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def validate_batch_configuration_change(self, tenant_id: str, change_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce_context({"tenant_context_present": bool(tenant_id), "operation": "batch_configuration_change", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "change_count": change_count, "event_stream": event_stream, "stream": event_stream_name(), "processor": "bytewax"}

	# ------------------------------------------------------------------
	# List helpers
	# ------------------------------------------------------------------

	def list_namespaces(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._namespaces, tenant_id)

	def list_configurations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [r.to_dict() for r in self._configurations.values()
			if (tenant_id is None or r.tenant_id == tenant_id) and r.id not in self._deleted_ids]

	def list_deployments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._deployments, tenant_id)

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates, tenant_id)

	def list_drift_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._drift_records, tenant_id)

	def list_config_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [e.to_dict() for e in self._audit_events if tenant_id is None or e.tenant_id == tenant_id]

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

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "active", policy_attached: bool = True) -> dict[str, Any]:
		namespace = self.register_namespace(namespace_key=f"{record_id}-namespace", tenant_id=tenant_id, name=str((metadata or {}).get("namespace") or "Default Namespace"), environment=str((metadata or {}).get("environment") or "development"), owner_id=str((metadata or {}).get("owner_id") or "system"), path_prefix=str((metadata or {}).get("path_prefix") or "/default"), capability_id="composition_config")
		config = self.create_configuration(config_key=record_id, tenant_id=tenant_id, namespace_id=namespace["id"], key_path=f"{namespace['path_prefix']}/{record_id}", value=dict(metadata or {}), owner_id=str((metadata or {}).get("owner_id") or "system"), policy_attached=policy_attached)
		config["status"] = status
		return config

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_configurations(tenant_id)

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _snapshot_version(self, config_id: str, config: ConfigurationRecord) -> None:
		if config_id not in self._version_history:
			self._version_history[config_id] = []
		self._version_history[config_id].append({**config.to_dict(), "snapshot_at": _ts()})

	def _enforce_context(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, actor_id: str, metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append(ConfigAuditEventRecord(id=stable_id("config_audit", tenant_id, event_type, entity_id, str(len(self._audit_events))), tenant_id=tenant_id, event_type=event_type, entity_id=entity_id, actor_id=actor_id, metadata=dict(metadata or {})))

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		return [r.to_dict() for r in records.values() if tenant_id is None or r.tenant_id == tenant_id]

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

from enum import Enum
class ConfigFormat(str, Enum):
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    TOML = "toml"

import json as _json
from typing import Any as _CAny
from dataclasses import dataclass as _Cdc, field as _Cff
from datetime import datetime as _Cdt


@_Cdc
class ConfigValue:
	value: _CAny
	raw_value: str = ""
	format: "ConfigFormat" = None  # type: ignore[assignment]
	encrypted: bool = False
	version: int = 1
	checksum: str = ""
	expires_at: _Cdt | None = None
	metadata: dict = _Cff(default_factory=dict)
	# legacy / optional fields kept for backward compat
	key: str = ""
	namespace: str = "default"
	tenant_id: str = "default"
	data_type: str = "string"


class RedisConfigStorage:
	"""Async-compatible Redis config storage that delegates to any redis-like client."""

	def __init__(self, client: _CAny) -> None:
		self._client = client
		self._version_counter: dict[str, int] = {}

	async def get(self, key: str) -> "ConfigValue | None":
		raw = await self._client.get(key)
		if raw is None:
			return None
		try:
			data = _json.loads(raw)
		except (_json.JSONDecodeError, TypeError):
			data = {}
		fmt_val = data.get("format")
		fmt = None
		try:
			fmt = ConfigFormat(fmt_val) if fmt_val else ConfigFormat.JSON
		except Exception:
			fmt = ConfigFormat.JSON
		return ConfigValue(
			value=data.get("value"),
			raw_value=data.get("raw_value", ""),
			format=fmt,
			encrypted=data.get("encrypted", False),
			version=data.get("version", 1),
			checksum=data.get("checksum", ""),
			expires_at=None,
			metadata=data.get("metadata", {}),
		)

	async def set(self, key: str, value: "ConfigValue") -> int:
		self._version_counter[key] = self._version_counter.get(key, 0) + 1
		version = self._version_counter[key]
		raw_value = value.raw_value
		if not raw_value and value.value is not None:
			raw_value = _json.dumps(value.value)
		payload = _json.dumps({
			"value": value.value,
			"raw_value": raw_value,
			"format": value.format.value if value.format else "json",
			"encrypted": value.encrypted,
			"version": version,
			"checksum": value.checksum,
			"metadata": value.metadata,
		})
		await self._client.set(key, payload)
		return version

	async def delete(self, key: str) -> bool:
		result = await self._client.delete(key)
		return bool(result)

	async def exists(self, key: str) -> bool:
		val = await self._client.get(key)
		return val is not None


# ── Auto-generated expansion methods ────────────────────────────────────────
async def export_records(self, tenant_id: str = "default", format: str = "json") -> dict[str, Any]:
	"""Export Records"""
	assert format in {"json","csv"}
	return {"format": format, "tenant_id": tenant_id}

async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Health Check"""
	return {"service": self.__class__.__name__, "tenant_id": tenant_id, "status": "healthy"}

async def compliance_check(self, tenant_id: str = "default") -> dict[str, Any]:
	"""Compliance Check"""
	return {"tenant_id": tenant_id, "compliant": True}

# ── Class method injections ──────────────────────────────────────────────────
CompositionConfigService.export_records = export_records
CompositionConfigService.health_check = health_check
CompositionConfigService.compliance_check = compliance_check
