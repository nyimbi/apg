"""Dependency-light REGY lifecycle runtime for generated APG applications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	PRIVILEGED_REGY_AGENT_ROLES,
	SUPPORTED_REGY_AGENT_ROLES,
	SUPPORTED_REGY_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


ALLOWED_REGIONS = {"local", "edge-africa", "edge-eu", "edge-east", "edge-west"}


@dataclass(frozen=True)
class RegistryServiceRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	service_type: str
	environment: str
	api_version: str
	contract_schema_ref: str
	health_endpoint: str
	status: str = "registered"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	labels: dict[str, Any] = field(default_factory=dict)
	routing_metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryInstanceRecord:
	id: str
	tenant_id: str
	service_id: str
	endpoint: str
	region: str
	health_probe: str
	weight: int = 100
	health: str = "healthy"
	status: str = "active"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryVersionRecord:
	id: str
	tenant_id: str
	service_id: str
	version: str
	contract_schema_ref: str
	breaking_change_detected: bool = False
	compatibility_review_recorded: bool = False
	status: str = "active"
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	migration_notes: str = ""
	eol_date: str = ""
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryGatewayPublication:
	id: str
	tenant_id: str
	service_id: str
	route_path: str
	strategy: str
	status: str
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryReviewRecord:
	id: str
	tenant_id: str
	subject_id: str
	review_type: str
	status: str = "pending"
	decision: str = "pending"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "require_review"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	requester: str = ""
	notes: str = ""
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	status: str
	contribution_disclosed: bool
	human_approval_required: bool
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class RegistryLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	operation: str
	mutation_count: int
	status: str
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	policy_decision: str = "allow"
	review_reasons: list[str] = field(default_factory=list)
	review_evidence: dict[str, Any] = field(default_factory=dict)
	required_processor: str = "bytewax"
	created_at: str = field(default_factory=lambda: _now())


class RegistryService:
	"""Tenant-scoped registry control-plane facade for generated APG apps."""

	def __init__(self) -> None:
		self._agent_runtimes = set(SUPPORTED_REGY_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_REGY_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_REGY_AGENT_ROLES)
		self._services: dict[tuple[str, str], RegistryServiceRecord] = {}
		self._instances: dict[tuple[str, str], RegistryInstanceRecord] = {}
		self._versions: dict[tuple[str, str], RegistryVersionRecord] = {}
		self._publications: dict[tuple[str, str], RegistryGatewayPublication] = {}
		self._reviews: dict[tuple[str, str], RegistryReviewRecord] = {}
		self._registry_agents: dict[tuple[str, str], RegistryAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], RegistryLifecycleBatchRecord] = {}
		self._events: list[RegistryAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_service(
		self,
		service_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		service_type: str,
		environment: str,
		api_version: str,
		contract_schema_ref: str,
		health_endpoint: str,
		routing_metadata: dict[str, Any] | None = None,
		labels: dict[str, Any] | None = None,
		production_review_recorded: bool = False,
		trace_propagation_configured: bool = True,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_service",
			"owner_assigned": bool(owner),
			"health_endpoint_present": bool(health_endpoint),
			"api_version_present": bool(api_version),
			"contract_schema_present": bool(contract_schema_ref),
			"duplicate_service_name": self._has_service_name(tenant_id, name),
			"environment": environment,
			"production_review_recorded": production_review_recorded,
			"trace_propagation_configured": trace_propagation_configured,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, service_id) in self._services:
			raise ValueError(f"service already exists for tenant: {service_id}")
		if not name:
			raise ValueError("service name is required")
		record = RegistryServiceRecord(
			id=service_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			service_type=service_type,
			environment=environment,
			api_version=api_version,
			contract_schema_ref=contract_schema_ref,
			health_endpoint=health_endpoint,
			status="pending_review" if result["decision"] == "require_review" else "registered",
			**_policy_kwargs(result, production_review_recorded),
			labels=dict(labels or {}),
			routing_metadata=dict(routing_metadata or {}),
		)
		self._services[self._tenant_key(tenant_id, service_id)] = record
		self._record_event(tenant_id, "service_registered", service_id, f"Registered service {name}.", {"matched_rules": result["matched_rules"], "status": record.status}, result, production_review_recorded)
		if result["decision"] == "require_review":
			self._reviews[self._tenant_key(tenant_id, f"production:{service_id}")] = RegistryReviewRecord(
				id=f"production:{service_id}",
				tenant_id=tenant_id,
				subject_id=service_id,
				review_type="production_registration",
				matched_rules=result["matched_rules"],
				policy_decision=result["decision"],
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result, production_review_recorded),
				requester=owner,
				notes="Production service registration requires review.",
			)
		return _dump(record)

	def register_instance(
		self,
		instance_id: str,
		tenant_id: str,
		service_id: str,
		endpoint: str,
		region: str,
		health_probe: str,
		weight: int = 100,
		health: str = "healthy",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._get_service(tenant_id, service_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_instance",
			"endpoint_present": bool(endpoint),
			"health_probe_present": bool(health_probe),
			"allowed_region": region in ALLOWED_REGIONS,
			"positive_weight": weight > 0,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, instance_id) in self._instances:
			raise ValueError(f"instance already exists for tenant: {instance_id}")
		record = RegistryInstanceRecord(
			id=instance_id,
			tenant_id=tenant_id,
			service_id=service_id,
			endpoint=endpoint,
			region=region,
			health_probe=health_probe,
			weight=weight,
			health=health,
			**_policy_kwargs(result),
			metadata=dict(metadata or {}),
		)
		self._instances[self._tenant_key(tenant_id, instance_id)] = record
		self._record_event(tenant_id, "instance_registered", instance_id, f"Registered instance {instance_id}.", {"service_id": service_id, "matched_rules": result["matched_rules"]}, result)
		return _dump(record)

	def discover_services(
		self,
		tenant_id: str,
		service_name: str | None = None,
		healthy_only: bool = True,
		requested_result_limit: int = 100,
		target_tenant_id: str | None = None,
		discovery_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		cross_tenant = bool(target_tenant_id and target_tenant_id != tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "discover_services",
			"cross_tenant_discovery": cross_tenant,
			"requested_result_limit": requested_result_limit,
			"discovery_review_recorded": discovery_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if result["decision"] == "require_review":
			review_id = f"discovery:{tenant_id}:{requested_result_limit}"
			self._reviews[self._tenant_key(tenant_id, review_id)] = RegistryReviewRecord(
				id=review_id,
				tenant_id=tenant_id,
				subject_id=tenant_id,
				review_type="discovery_limit",
				matched_rules=result["matched_rules"],
				policy_decision=result["decision"],
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result, discovery_review_recorded),
				requester=tenant_id,
				notes=f"Discovery requested limit {requested_result_limit}.",
			)
			self._record_event(tenant_id, "discovery_review_requested", review_id, "Requested discovery limit review.", {"requested_result_limit": requested_result_limit, "matched_rules": result["matched_rules"]}, result, discovery_review_recorded)
		target = target_tenant_id or tenant_id
		services = [
			_dump(service)
			for service in self._services.values()
			if service.tenant_id == target and (service_name is None or service.name == service_name)
		]
		instances_by_service = self._instances_by_service(target)
		if healthy_only:
			services = [
				service
				for service in services
				if any(instance["health"] == "healthy" for instance in instances_by_service.get(service["id"], []))
			]
		services = services[:requested_result_limit]
		return {
			"tenant_id": tenant_id,
			"target_tenant_id": target,
			"decision": result["decision"],
			"matched_rules": result["matched_rules"],
			"policy_decision": result["decision"],
			"review_reasons": _reasons(result),
			"review_evidence": _review_evidence(result, discovery_review_recorded),
			"total_count": len(services),
			"services": services,
			"instances": {service["id"]: instances_by_service.get(service["id"], []) for service in services},
		}

	def record_version(
		self,
		version_id: str,
		tenant_id: str,
		service_id: str,
		version: str,
		contract_schema_ref: str,
		breaking_change_detected: bool = False,
		compatibility_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		self._get_service(tenant_id, service_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_version",
			"breaking_change_detected": breaking_change_detected,
			"compatibility_review_recorded": compatibility_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = RegistryVersionRecord(
			id=version_id,
			tenant_id=tenant_id,
			service_id=service_id,
			version=version,
			contract_schema_ref=contract_schema_ref,
			breaking_change_detected=breaking_change_detected,
			compatibility_review_recorded=compatibility_review_recorded,
			status="pending_review" if result["decision"] == "require_review" else "active",
			**_policy_kwargs(result, compatibility_review_recorded),
		)
		self._versions[self._tenant_key(tenant_id, version_id)] = record
		if result["decision"] == "require_review":
			self._reviews[self._tenant_key(tenant_id, f"compatibility:{version_id}")] = RegistryReviewRecord(
				id=f"compatibility:{version_id}",
				tenant_id=tenant_id,
				subject_id=version_id,
				review_type="compatibility",
				matched_rules=result["matched_rules"],
				policy_decision=result["decision"],
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result, compatibility_review_recorded),
				requester=self._get_service(tenant_id, service_id).owner,
				notes="Breaking change requires compatibility review.",
			)
		self._record_event(tenant_id, "version_recorded", version_id, f"Recorded version {version}.", {"service_id": service_id, "matched_rules": result["matched_rules"]}, result, compatibility_review_recorded)
		return _dump(record)

	def publish_to_gateway(
		self,
		publication_id: str,
		tenant_id: str,
		service_id: str,
		route_path: str,
		strategy: str = "weighted",
	) -> dict[str, Any]:
		service = self._get_service(tenant_id, service_id)
		instances = self._service_instances(tenant_id, service_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_to_gateway",
			"service_registered": True,
			"service_review_complete": service.status == "registered",
			"healthy_instance_present": any(instance.health == "healthy" for instance in instances),
			"routing_metadata_present": bool(service.routing_metadata),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = RegistryGatewayPublication(
			id=publication_id,
			tenant_id=tenant_id,
			service_id=service_id,
			route_path=route_path,
			strategy=strategy,
			status="published",
			**_policy_kwargs(result),
		)
		self._publications[self._tenant_key(tenant_id, publication_id)] = record
		self._record_event(tenant_id, "gateway_publication_created", publication_id, f"Published {service.name} to gateway.", {"service_id": service_id, "matched_rules": result["matched_rules"]}, result)
		return _dump(record)

	def deprecate_version(
		self,
		version_id: str,
		tenant_id: str,
		migration_notes: str,
		eol_date: str,
		future_eol_date: bool = True,
	) -> dict[str, Any]:
		version = self._get_version(tenant_id, version_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deprecate_version",
			"migration_notes_present": bool(migration_notes),
			"future_eol_date": future_eol_date,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = RegistryVersionRecord(
			id=version.id,
			tenant_id=version.tenant_id,
			service_id=version.service_id,
			version=version.version,
			contract_schema_ref=version.contract_schema_ref,
			breaking_change_detected=version.breaking_change_detected,
			compatibility_review_recorded=version.compatibility_review_recorded,
			status="deprecated",
			**_policy_kwargs(result),
			migration_notes=migration_notes,
			eol_date=eol_date,
		)
		self._versions[self._tenant_key(tenant_id, version_id)] = record
		self._record_event(tenant_id, "version_deprecated", version_id, f"Deprecated version {version.version}.", {"eol_date": eol_date, "matched_rules": result["matched_rules"]}, result)
		return _dump(record)

	def override_health(self, tenant_id: str, instance_id: str, health: str, incident_reference: str) -> dict[str, Any]:
		instance = self._get_instance(tenant_id, instance_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "override_health",
			"incident_reference_present": bool(incident_reference),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = RegistryInstanceRecord(
			id=instance.id,
			tenant_id=instance.tenant_id,
			service_id=instance.service_id,
			endpoint=instance.endpoint,
			region=instance.region,
			health_probe=instance.health_probe,
			weight=instance.weight,
			health=health,
			status=instance.status,
			**_policy_kwargs(result),
			metadata=dict(instance.metadata),
			created_at=instance.created_at,
		)
		self._instances[self._tenant_key(tenant_id, instance_id)] = record
		self._record_event(tenant_id, "health_overridden", instance_id, f"Set instance health to {health}.", {"incident_reference": incident_reference, "matched_rules": result["matched_rules"]}, result)
		return _dump(record)

	def transfer_owner(
		self,
		tenant_id: str,
		service_id: str,
		new_owner: str,
		actor: str,
		owner_transfer_review_recorded: bool = False,
	) -> dict[str, Any]:
		service = self._get_service(tenant_id, service_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "transfer_owner",
			"owner_transfer_review_recorded": owner_transfer_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if not new_owner:
			raise ValueError("new_owner is required")
		if result["decision"] == "require_review":
			review_id = f"owner:{service_id}"
			self._reviews[self._tenant_key(tenant_id, review_id)] = RegistryReviewRecord(
				id=review_id,
				tenant_id=tenant_id,
				subject_id=service_id,
				review_type="owner_transfer",
				matched_rules=result["matched_rules"],
				policy_decision=result["decision"],
				review_reasons=_reasons(result),
				review_evidence=_review_evidence(result, owner_transfer_review_recorded),
				requester=actor,
				notes=f"Transfer owner from {service.owner} to {new_owner}.",
			)
			self._record_event(tenant_id, "owner_transfer_review_requested", review_id, "Requested owner transfer review.", {"service_id": service_id, "new_owner": new_owner, "matched_rules": result["matched_rules"]}, result, owner_transfer_review_recorded)
			return _dump(self._reviews[self._tenant_key(tenant_id, review_id)])
		transferred = RegistryServiceRecord(
			id=service.id,
			tenant_id=service.tenant_id,
			name=service.name,
			owner=new_owner,
			service_type=service.service_type,
			environment=service.environment,
			api_version=service.api_version,
			contract_schema_ref=service.contract_schema_ref,
			health_endpoint=service.health_endpoint,
			status=service.status,
			**_policy_kwargs(result, owner_transfer_review_recorded),
			labels=dict(service.labels),
			routing_metadata=dict(service.routing_metadata),
			created_at=service.created_at,
		)
		self._services[self._tenant_key(tenant_id, service_id)] = transferred
		self._record_event(tenant_id, "owner_transferred", service_id, f"Transferred service owner to {new_owner}.", {"actor": actor, "previous_owner": service.owner, "matched_rules": result["matched_rules"]}, result, owner_transfer_review_recorded)
		return _dump(transferred)

	def retire_service(
		self,
		tenant_id: str,
		service_id: str,
		actor: str,
		impact_review_recorded: bool,
		gateway_unpublish_recorded: bool,
	) -> dict[str, Any]:
		service = self._get_service(tenant_id, service_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_service",
			"impact_review_recorded": impact_review_recorded,
			"gateway_unpublish_recorded": gateway_unpublish_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		retired = RegistryServiceRecord(
			id=service.id,
			tenant_id=service.tenant_id,
			name=service.name,
			owner=service.owner,
			service_type=service.service_type,
			environment=service.environment,
			api_version=service.api_version,
			contract_schema_ref=service.contract_schema_ref,
			health_endpoint=service.health_endpoint,
			status="retired",
			**_policy_kwargs(result, impact_review_recorded and gateway_unpublish_recorded),
			labels=dict(service.labels),
			routing_metadata=dict(service.routing_metadata),
			created_at=service.created_at,
		)
		self._services[self._tenant_key(tenant_id, service_id)] = retired
		self._record_event(tenant_id, "service_retired", service_id, f"Retired service {service.name}.", {"actor": actor, "matched_rules": result["matched_rules"]}, result, impact_review_recorded and gateway_unpublish_recorded)
		return _dump(retired)

	def register_registry_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		"""Register a governed AI/automation registry participant."""
		self._enforce_tenant(tenant_id)
		normalized_runtime = _normalize_agent_token(runtime)
		normalized_role = _normalize_agent_token(role)
		privileged_role = normalized_role in self._privileged_agent_roles
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_registry_agent",
			"agent_runtime_supported": normalized_runtime in self._agent_runtimes,
			"agent_role_supported": normalized_role in self._agent_roles,
			"scope_present": bool(scope),
			"owner_present": bool(owner),
			"purpose_present": bool(purpose),
			"contribution_disclosed": contribution_disclosed,
			"privileged_role": privileged_role,
			"human_approval_required": human_approval_required,
		})
		if result["decision"] == "deny":
			self._record_event(tenant_id, "registry_agent_registration_denied", agent_id, f"Denied registry agent {name}.", {"matched_rules": result["matched_rules"]}, result)
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, agent_id) in self._registry_agents:
			raise ValueError(f"registry agent already exists for tenant: {agent_id}")
		record = RegistryAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			owner=owner,
			purpose=purpose,
			status=_status_for_decision(result),
			contribution_disclosed=contribution_disclosed,
			human_approval_required=human_approval_required,
			**_policy_kwargs(result, human_approval_required),
		)
		self._registry_agents[self._tenant_key(tenant_id, agent_id)] = record
		self._record_event(tenant_id, "registry_agent_registered", agent_id, f"Registered registry agent {name}.", {"matched_rules": result["matched_rules"], "status": record.status}, result, human_approval_required)
		return _dump(record)

	def validate_regy_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "registry_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate that registry lifecycle mutations are processed by Bytewax."""
		self._enforce_tenant(tenant_id)
		normalized_stream = _normalize_agent_token(event_stream)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_regy_lifecycle_batch",
			"event_stream": normalized_stream,
		})
		resolved_batch_id = batch_id or f"{operation}:{len(self._lifecycle_batches) + 1}"
		record = RegistryLifecycleBatchRecord(
			id=resolved_batch_id,
			tenant_id=tenant_id,
			event_stream=normalized_stream,
			operation=operation,
			mutation_count=mutation_count,
			status="denied" if result["decision"] == "deny" else "accepted",
			**_policy_kwargs(result),
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, resolved_batch_id)] = record
		self._record_event(tenant_id, "regy_lifecycle_batch_validated", resolved_batch_id, f"Validated REGY lifecycle batch through {normalized_stream}.", {"matched_rules": result["matched_rules"], "status": record.status}, result)
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		return _dump(record)

	def registry_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		services = [service for service in self._services.values() if service.tenant_id == tenant_id]
		instances = [instance for instance in self._instances.values() if instance.tenant_id == tenant_id]
		lifecycle_batches = [batch for batch in self._lifecycle_batches.values() if batch.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"service_count": len(services),
			"instance_count": len(instances),
			"healthy_instance_count": len([instance for instance in instances if instance.health == "healthy"]),
			"version_count": len([version for version in self._versions.values() if version.tenant_id == tenant_id]),
			"publication_count": len([publication for publication in self._publications.values() if publication.tenant_id == tenant_id]),
			"pending_review_count": len([review for review in self._reviews.values() if review.tenant_id == tenant_id and review.status == "pending"]),
			"pending_registry_agent_review_count": len([agent for agent in self._registry_agents.values() if agent.tenant_id == tenant_id and agent.status == "pending_review"]),
			"review_count": len(self.list_pending_reviews(tenant_id)),
			"registry_agent_count": len([agent for agent in self._registry_agents.values() if agent.tenant_id == tenant_id]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([batch for batch in lifecycle_batches if batch.status == "denied"]),
			"audit_event_count": len([event for event in self._events if event.tenant_id == tenant_id]),
		}

	def list_services(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._services, tenant_id)

	def list_instances(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._instances, tenant_id)

	def list_versions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._versions, tenant_id)

	def list_gateway_publications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._publications, tenant_id)

	def list_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reviews, tenant_id)

	def list_registry_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._registry_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [_dump(event) for event in self._events if tenant_id is None or event.tenant_id == tenant_id]

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return registry records awaiting governance review."""
		items = (
			self.list_services(tenant_id)
			+ self.list_instances(tenant_id)
			+ self.list_versions(tenant_id)
			+ self.list_gateway_publications(tenant_id)
			+ self.list_reviews(tenant_id)
			+ self.list_registry_agents(tenant_id)
			+ self.list_lifecycle_batches(tenant_id)
		)
		return [
			record
			for record in items
			if record.get("status") in {"pending", "pending_review", "review_required"}
			or record.get("decision") == "pending"
		]

	def _has_service_name(self, tenant_id: str, name: str) -> bool:
		return any(service.tenant_id == tenant_id and service.name == name for service in self._services.values())

	def _service_instances(self, tenant_id: str, service_id: str) -> list[RegistryInstanceRecord]:
		return [instance for instance in self._instances.values() if instance.tenant_id == tenant_id and instance.service_id == service_id]

	def _instances_by_service(self, tenant_id: str) -> dict[str, list[dict[str, Any]]]:
		grouped: dict[str, list[dict[str, Any]]] = {}
		for instance in self._instances.values():
			if instance.tenant_id == tenant_id:
				grouped.setdefault(instance.service_id, []).append(_dump(instance))
		return grouped

	def _get_service(self, tenant_id: str, service_id: str) -> RegistryServiceRecord:
		record = self._services.get(self._tenant_key(tenant_id, service_id))
		if record is None:
			raise KeyError(f"unknown service for tenant: {service_id}")
		return record

	def _get_instance(self, tenant_id: str, instance_id: str) -> RegistryInstanceRecord:
		record = self._instances.get(self._tenant_key(tenant_id, instance_id))
		if record is None:
			raise KeyError(f"unknown instance for tenant: {instance_id}")
		return record

	def _get_version(self, tenant_id: str, version_id: str) -> RegistryVersionRecord:
		record = self._versions.get(self._tenant_key(tenant_id, version_id))
		if record is None:
			raise KeyError(f"unknown version for tenant: {version_id}")
		return record

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
		policy_result: dict[str, Any] | None = None,
		review_recorded: bool = False,
	) -> None:
		result = policy_result or _allow_result()
		self._events.append(RegistryAuditEvent(
			id=f"event:{len(self._events) + 1}",
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			policy_decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=_reasons(result),
			review_evidence=_review_evidence(result, review_recorded),
			evidence=dict(evidence or {}),
		))

	def _list(self, store: dict[tuple[str, str], Any], tenant_id: str | None) -> list[dict[str, Any]]:
		return [_dump(record) for record in store.values() if tenant_id is None or record.tenant_id == tenant_id]

	def _enforce_tenant(self, tenant_id: str) -> None:
		if not tenant_id:
			_raise_if_blocked(self.evaluate({"tenant_context_present": False}))

	def _tenant_key(self, tenant_id: str, value: str) -> tuple[str, str]:
		return tenant_id, value


def _dump(record: Any) -> dict[str, Any]:
	return asdict(record)


def _allow_result() -> dict[str, Any]:
	return {"decision": "allow", "matched_rules": [], "actions": []}


def _reasons(result: dict[str, Any]) -> list[str]:
	return list(dict.fromkeys(
		str(action["reason"])
		for action in result.get("actions", [])
		if action.get("reason")
	))


def _review_evidence(result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
	return {
		"required_actions": list(dict.fromkeys(
			str(action.get("required_action"))
			for action in result.get("actions", [])
			if action.get("required_action")
		)),
		"reasons": _reasons(result),
		"review_recorded": bool(review_recorded),
	}


def _policy_kwargs(result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
	return {
		"decision": result["decision"],
		"matched_rules": list(result["matched_rules"]),
		"policy_decision": result["decision"],
		"review_reasons": _reasons(result),
		"review_evidence": _review_evidence(result, review_recorded),
	}


def _raise_if_blocked(result: dict[str, Any]) -> None:
	reasons = [action.get("reason", "registry_guardrail_failed") for action in result.get("actions", [])]
	raise PermissionError(",".join(reasons) or "registry_guardrail_failed")


def _normalize_agent_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _status_for_decision(result: dict[str, Any]) -> str:
	return "pending_review" if result["decision"] == "require_review" else "active"


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()
