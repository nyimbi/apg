"""Dependency-light CONN lifecycle runtime for generated APG applications."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .capability_contract import (
	PRIVILEGED_CONN_AGENT_ROLES,
	SUPPORTED_CONN_AGENT_ROLES,
	SUPPORTED_CONN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


@dataclass(frozen=True)
class ConnectorRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	source_ref: str
	checksum: str
	owner: str
	verified_source: bool = True
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ConnectionRecord:
	id: str
	tenant_id: str
	name: str
	connector_id: str
	owner: str
	environment: str
	contains_credentials: bool = True
	credential_vault_ref: str = ""
	credentials_encrypted: bool = True
	secret_rotation_recorded: bool = False
	last_test_passed: bool = False
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class FlowRecord:
	id: str
	tenant_id: str
	name: str
	source_connection_id: str
	target_connection_id: str
	owner: str
	mapping_ref: str
	lineage_enabled: bool = True
	quality_gate_ref: str = ""
	pii_detected: bool = False
	status: str = "created"
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class SyncRunRecord:
	id: str
	tenant_id: str
	flow_id: str
	mode: str
	batch_size: int
	status: str
	records_processed: int = 0
	quality_score: float | None = None
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ScheduleRecord:
	id: str
	tenant_id: str
	flow_id: str
	cron: str
	timezone: str
	status: str = "scheduled"
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ReviewRecord:
	id: str
	tenant_id: str
	subject_id: str
	review_type: str
	status: str = "pending"
	requester: str = ""
	notes: str = ""
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ConnectorAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ConnectorAgentRecord:
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
	created_at: str = field(default_factory=lambda: _now())


@dataclass(frozen=True)
class ConnectorLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	operation: str
	mutation_count: int
	status: str
	created_at: str = field(default_factory=lambda: _now())


class ConnService:
	"""Tenant-scoped connector lifecycle facade for generated APG apps."""

	def __init__(self) -> None:
		self._agent_runtimes = set(SUPPORTED_CONN_AGENT_RUNTIMES)
		self._agent_roles = set(SUPPORTED_CONN_AGENT_ROLES)
		self._privileged_agent_roles = set(PRIVILEGED_CONN_AGENT_ROLES)
		self._connectors: dict[tuple[str, str], ConnectorRecord] = {}
		self._connections: dict[tuple[str, str], ConnectionRecord] = {}
		self._flows: dict[tuple[str, str], FlowRecord] = {}
		self._runs: dict[tuple[str, str], SyncRunRecord] = {}
		self._schedules: dict[tuple[str, str], ScheduleRecord] = {}
		self._reviews: dict[tuple[str, str], ReviewRecord] = {}
		self._connector_agents: dict[tuple[str, str], ConnectorAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], ConnectorLifecycleBatchRecord] = {}
		self._events: list[ConnectorAuditEvent] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_connector(
		self,
		connector_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		source_ref: str,
		checksum: str,
		owner: str,
		verified_source: bool = True,
		marketplace_review_recorded: bool = False,
		auth_policy_attached: bool = True,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		supported_runtimes = get_capability_contract(tenant_id)["configuration"]["connectors"]["supported_runtimes"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_connector",
			"owner_assigned": bool(owner),
			"runtime_present": bool(runtime),
			"connector_runtime_supported": runtime in supported_runtimes,
			"source_present": bool(source_ref),
			"checksum_present": bool(checksum),
			"verified_source": verified_source,
			"marketplace_review_recorded": marketplace_review_recorded,
			"connector_runtime": runtime,
			"auth_policy_attached": auth_policy_attached,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, connector_id) in self._connectors:
			raise ValueError(f"connector already exists for tenant: {connector_id}")
		record = ConnectorRecord(
			id=connector_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime,
			source_ref=source_ref,
			checksum=checksum,
			owner=owner,
			verified_source=verified_source,
			status="pending_review" if result["decision"] == "require_review" else "registered",
			metadata=dict(metadata or {}),
		)
		self._connectors[self._tenant_key(tenant_id, connector_id)] = record
		self._record_event(tenant_id, "connector_registered", connector_id, f"Registered connector {name}.", {"matched_rules": result["matched_rules"], "status": record.status})
		if result["decision"] == "require_review":
			self._create_review(tenant_id, f"marketplace:{connector_id}", connector_id, "marketplace", owner, "Unverified connector source requires review.")
		return _dump(record)

	def register_connection(
		self,
		connection_id: str,
		tenant_id: str,
		name: str,
		connector_id: str,
		owner: str,
		environment: str,
		contains_credentials: bool = True,
		credential_vault_ref: str = "",
		credentials_encrypted: bool = True,
		cross_tenant_connection: bool = False,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		connector_registered = self._tenant_key(tenant_id, connector_id) in self._connectors
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_connection",
			"owner_assigned": bool(owner),
			"connector_registered": connector_registered,
			"contains_credentials": contains_credentials,
			"credential_vault_ref_present": bool(credential_vault_ref),
			"credentials_encrypted": credentials_encrypted,
			"cross_tenant_connection": cross_tenant_connection,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, connection_id) in self._connections:
			raise ValueError(f"connection already exists for tenant: {connection_id}")
		connector = self._get_connector(tenant_id, connector_id)
		if connector.status != "registered":
			raise PermissionError("connector_review_required")
		record = ConnectionRecord(
			id=connection_id,
			tenant_id=tenant_id,
			name=name,
			connector_id=connector_id,
			owner=owner,
			environment=environment,
			contains_credentials=contains_credentials,
			credential_vault_ref=credential_vault_ref,
			credentials_encrypted=credentials_encrypted,
			metadata=dict(metadata or {}),
		)
		self._connections[self._tenant_key(tenant_id, connection_id)] = record
		self._record_event(tenant_id, "connection_registered", connection_id, f"Registered connection {name}.", {"connector_id": connector_id, "matched_rules": result["matched_rules"]})
		return _dump(record)

	def record_connection_test(self, tenant_id: str, connection_id: str, passed: bool, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
		connection = self._get_connection(tenant_id, connection_id)
		record = self._replace_connection(connection, last_test_passed=passed)
		self._connections[self._tenant_key(tenant_id, connection_id)] = record
		self._record_event(tenant_id, "connection_test_recorded", connection_id, f"Connection test {'passed' if passed else 'failed'}.", dict(evidence or {}))
		return _dump(record)

	def activate_connection(
		self,
		tenant_id: str,
		connection_id: str,
		secret_rotation_recorded: bool,
		activation_review_recorded: bool = False,
	) -> dict[str, Any]:
		connection = self._get_connection(tenant_id, connection_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "activate_connection",
			"last_test_passed": connection.last_test_passed,
			"secret_rotation_recorded": secret_rotation_recorded,
			"environment": connection.environment,
			"activation_review_recorded": activation_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		if result["decision"] == "require_review":
			self._create_review(tenant_id, f"activation:{connection_id}", connection_id, "activation", connection.owner, "Production connection activation requires review.")
			record = self._replace_connection(connection, status="pending_review", secret_rotation_recorded=secret_rotation_recorded)
		else:
			record = self._replace_connection(connection, status="active", secret_rotation_recorded=secret_rotation_recorded)
		self._connections[self._tenant_key(tenant_id, connection_id)] = record
		self._record_event(tenant_id, "connection_activation_requested", connection_id, f"Connection activation decision: {record.status}.", {"matched_rules": result["matched_rules"]})
		return _dump(record)

	def create_flow(
		self,
		flow_id: str,
		tenant_id: str,
		name: str,
		source_connection_id: str,
		target_connection_id: str,
		owner: str,
		mapping_ref: str,
		lineage_enabled: bool = True,
		quality_gate_ref: str = "",
		pii_detected: bool = False,
		pii_policy_attached: bool = True,
	) -> dict[str, Any]:
		source = self._get_connection(tenant_id, source_connection_id)
		target = self._get_connection(tenant_id, target_connection_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_flow",
			"source_connection_active": source.status == "active",
			"target_connection_active": target.status == "active",
			"mapping_present": bool(mapping_ref),
			"lineage_enabled": lineage_enabled,
			"quality_gate_present": bool(quality_gate_ref),
			"pii_detected": pii_detected,
			"pii_policy_attached": pii_policy_attached,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = FlowRecord(
			id=flow_id,
			tenant_id=tenant_id,
			name=name,
			source_connection_id=source_connection_id,
			target_connection_id=target_connection_id,
			owner=owner,
			mapping_ref=mapping_ref,
			lineage_enabled=lineage_enabled,
			quality_gate_ref=quality_gate_ref,
			pii_detected=pii_detected,
		)
		self._flows[self._tenant_key(tenant_id, flow_id)] = record
		self._record_event(tenant_id, "flow_created", flow_id, f"Created flow {name}.", {"matched_rules": result["matched_rules"]})
		return _dump(record)

	def start_sync(
		self,
		run_id: str,
		tenant_id: str,
		flow_id: str,
		mode: str = "incremental",
		batch_size: int = 1000,
		monitoring_enabled: bool = True,
		schema_change_detected: bool = False,
		schema_review_recorded: bool = False,
	) -> dict[str, Any]:
		self._get_flow(tenant_id, flow_id)
		allowed_modes = get_capability_contract(tenant_id)["configuration"]["sync"]["allowed_modes"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_sync",
			"sync_mode_supported": mode in allowed_modes,
			"batch_size": batch_size,
			"monitoring_enabled": monitoring_enabled,
			"schema_change_detected": schema_change_detected,
			"schema_review_recorded": schema_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		status = "pending_review" if result["decision"] == "require_review" else "running"
		if result["decision"] == "require_review":
			self._create_review(tenant_id, f"schema:{run_id}", run_id, "schema_change", flow_id, "Schema change requires review before sync.")
		record = SyncRunRecord(id=run_id, tenant_id=tenant_id, flow_id=flow_id, mode=mode, batch_size=batch_size, status=status)
		self._runs[self._tenant_key(tenant_id, run_id)] = record
		self._record_event(tenant_id, "sync_started", run_id, f"Started sync run {run_id}.", {"matched_rules": result["matched_rules"], "status": status})
		return _dump(record)

	def complete_sync(self, tenant_id: str, run_id: str, records_processed: int, quality_score: float) -> dict[str, Any]:
		run = self._get_run(tenant_id, run_id)
		if run.status != "running":
			raise PermissionError("sync_run_not_running")
		record = SyncRunRecord(
			id=run.id,
			tenant_id=run.tenant_id,
			flow_id=run.flow_id,
			mode=run.mode,
			batch_size=run.batch_size,
			status="completed",
			records_processed=records_processed,
			quality_score=quality_score,
			created_at=run.created_at,
		)
		self._runs[self._tenant_key(tenant_id, run_id)] = record
		self._record_event(tenant_id, "sync_completed", run_id, f"Completed sync run {run_id}.", {"records_processed": records_processed, "quality_score": quality_score})
		return _dump(record)

	def schedule_flow(self, tenant_id: str, schedule_id: str, flow_id: str, cron: str, timezone: str) -> dict[str, Any]:
		self._get_flow(tenant_id, flow_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "schedule_flow",
			"timezone_present": bool(timezone),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = ScheduleRecord(id=schedule_id, tenant_id=tenant_id, flow_id=flow_id, cron=cron, timezone=timezone)
		self._schedules[self._tenant_key(tenant_id, schedule_id)] = record
		self._record_event(tenant_id, "flow_scheduled", schedule_id, f"Scheduled flow {flow_id}.", {"cron": cron, "timezone": timezone})
		return _dump(record)

	def replay_sync(self, tenant_id: str, run_id: str, replay_id: str, idempotency_key: str) -> dict[str, Any]:
		run = self._get_run(tenant_id, run_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "replay_sync",
			"idempotency_key_present": bool(idempotency_key),
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = SyncRunRecord(id=replay_id, tenant_id=tenant_id, flow_id=run.flow_id, mode=run.mode, batch_size=run.batch_size, status="queued")
		self._runs[self._tenant_key(tenant_id, replay_id)] = record
		self._record_event(tenant_id, "sync_replay_queued", replay_id, f"Queued replay for {run_id}.", {"idempotency_key": idempotency_key})
		return _dump(record)

	def transfer_owner(self, tenant_id: str, connection_id: str, new_owner: str, actor: str, owner_transfer_review_recorded: bool = False) -> dict[str, Any]:
		connection = self._get_connection(tenant_id, connection_id)
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
			review = self._create_review(tenant_id, f"owner:{connection_id}", connection_id, "owner_transfer", actor, f"Transfer owner to {new_owner}.")
			return _dump(review)
		record = self._replace_connection(connection, owner=new_owner)
		self._connections[self._tenant_key(tenant_id, connection_id)] = record
		self._record_event(tenant_id, "connection_owner_transferred", connection_id, f"Transferred owner to {new_owner}.", {"actor": actor})
		return _dump(record)

	def retire_connection(self, tenant_id: str, connection_id: str, actor: str, impact_review_recorded: bool) -> dict[str, Any]:
		connection = self._get_connection(tenant_id, connection_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "retire_connection",
			"impact_review_recorded": impact_review_recorded,
		})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		record = self._replace_connection(connection, status="retired")
		self._connections[self._tenant_key(tenant_id, connection_id)] = record
		self._record_event(tenant_id, "connection_retired", connection_id, f"Retired connection {connection.name}.", {"actor": actor})
		return _dump(record)

	def register_connector_agent(
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
		"""Register a governed AI/automation participant for connector work."""
		self._enforce_tenant(tenant_id)
		normalized_runtime = _normalize_agent_token(runtime)
		normalized_role = _normalize_agent_token(role)
		privileged_role = normalized_role in self._privileged_agent_roles
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_connector_agent",
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
			self._record_event(tenant_id, "connector_agent_registration_denied", agent_id, f"Denied connector agent {name}.", {"matched_rules": result["matched_rules"]})
			_raise_if_blocked(result)
		if self._tenant_key(tenant_id, agent_id) in self._connector_agents:
			raise ValueError(f"connector agent already exists for tenant: {agent_id}")
		record = ConnectorAgentRecord(
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
		)
		self._connector_agents[self._tenant_key(tenant_id, agent_id)] = record
		self._record_event(tenant_id, "connector_agent_registered", agent_id, f"Registered connector agent {name}.", {"matched_rules": result["matched_rules"], "status": record.status})
		return _dump(record)

	def validate_conn_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "connector_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		"""Validate that connector lifecycle mutations are processed by Bytewax."""
		self._enforce_tenant(tenant_id)
		normalized_stream = _normalize_agent_token(event_stream)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_conn_lifecycle_batch",
			"event_stream": normalized_stream,
		})
		resolved_batch_id = batch_id or f"{operation}:{len(self._lifecycle_batches) + 1}"
		record = ConnectorLifecycleBatchRecord(
			id=resolved_batch_id,
			tenant_id=tenant_id,
			event_stream=normalized_stream,
			operation=operation,
			mutation_count=mutation_count,
			status="denied" if result["decision"] == "deny" else "accepted",
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, resolved_batch_id)] = record
		self._record_event(tenant_id, "conn_lifecycle_batch_validated", resolved_batch_id, f"Validated CONN lifecycle batch through {normalized_stream}.", {"matched_rules": result["matched_rules"], "status": record.status})
		if result["decision"] == "deny":
			_raise_if_blocked(result)
		return _dump(record)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		connections = [connection for connection in self._connections.values() if connection.tenant_id == tenant_id]
		lifecycle_batches = [batch for batch in self._lifecycle_batches.values() if batch.tenant_id == tenant_id]
		return {
			"tenant_id": tenant_id,
			"connector_count": len([connector for connector in self._connectors.values() if connector.tenant_id == tenant_id]),
			"connection_count": len(connections),
			"active_connection_count": len([connection for connection in connections if connection.status == "active"]),
			"flow_count": len([flow for flow in self._flows.values() if flow.tenant_id == tenant_id]),
			"sync_run_count": len([run for run in self._runs.values() if run.tenant_id == tenant_id]),
			"pending_review_count": len([review for review in self._reviews.values() if review.tenant_id == tenant_id and review.status == "pending"]),
			"connector_agent_count": len([agent for agent in self._connector_agents.values() if agent.tenant_id == tenant_id]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([batch for batch in lifecycle_batches if batch.status == "denied"]),
			"audit_event_count": len([event for event in self._events if event.tenant_id == tenant_id]),
		}

	def list_connectors(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._connectors, tenant_id)

	def list_connections(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._connections, tenant_id)

	def list_flows(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._flows, tenant_id)

	def list_sync_runs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._runs, tenant_id)

	def list_schedules(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._schedules, tenant_id)

	def list_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reviews, tenant_id)

	def list_connector_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._connector_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [_dump(event) for event in self._events if tenant_id is None or event.tenant_id == tenant_id]

	def _create_review(self, tenant_id: str, review_id: str, subject_id: str, review_type: str, requester: str, notes: str) -> ReviewRecord:
		review = ReviewRecord(id=review_id, tenant_id=tenant_id, subject_id=subject_id, review_type=review_type, requester=requester, notes=notes)
		self._reviews[self._tenant_key(tenant_id, review_id)] = review
		self._record_event(tenant_id, f"{review_type}_review_requested", review_id, f"Requested {review_type} review.", {"subject_id": subject_id})
		return review

	def _replace_connection(self, connection: ConnectionRecord, **changes: Any) -> ConnectionRecord:
		values = _dump(connection)
		values.update(changes)
		return ConnectionRecord(**values)

	def _get_connector(self, tenant_id: str, connector_id: str) -> ConnectorRecord:
		record = self._connectors.get(self._tenant_key(tenant_id, connector_id))
		if record is None:
			raise KeyError(f"unknown connector for tenant: {connector_id}")
		return record

	def _get_connection(self, tenant_id: str, connection_id: str) -> ConnectionRecord:
		record = self._connections.get(self._tenant_key(tenant_id, connection_id))
		if record is None:
			raise KeyError(f"unknown connection for tenant: {connection_id}")
		return record

	def _get_flow(self, tenant_id: str, flow_id: str) -> FlowRecord:
		record = self._flows.get(self._tenant_key(tenant_id, flow_id))
		if record is None:
			raise KeyError(f"unknown flow for tenant: {flow_id}")
		return record

	def _get_run(self, tenant_id: str, run_id: str) -> SyncRunRecord:
		record = self._runs.get(self._tenant_key(tenant_id, run_id))
		if record is None:
			raise KeyError(f"unknown sync run for tenant: {run_id}")
		return record

	def _record_event(self, tenant_id: str, event_type: str, subject_id: str, message: str, evidence: dict[str, Any] | None = None) -> None:
		self._events.append(ConnectorAuditEvent(
			id=f"event:{len(self._events) + 1}",
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
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


def _raise_if_blocked(result: dict[str, Any]) -> None:
	reasons = [action.get("reason", "connector_guardrail_failed") for action in result.get("actions", [])]
	raise PermissionError(",".join(reasons) or "connector_guardrail_failed")


def _normalize_agent_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _status_for_decision(result: dict[str, Any]) -> str:
	return "pending_review" if result["decision"] == "require_review" else "active"


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()
