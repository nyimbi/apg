"""Dependency-light collaboration runtime for package-backed COLB composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from .capability_contract import (
	PRIVILEGED_COLB_AGENT_ROLES,
	SUPPORTED_COLB_AGENT_ROLES,
	SUPPORTED_COLB_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: object) -> str:
	key = "|".join(str(part) for part in parts)
	return f"{prefix}_{sha1(key.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class WorkspaceRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	participants: list[str]
	retention_policy: str
	external_participants: list[str] = field(default_factory=list)
	status: str = "active"
	review_status: str = "approved"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"participants": list(self.participants),
			"participant_count": len(self.participants) + len(self.external_participants),
			"retention_policy": self.retention_policy,
			"external_participants": list(self.external_participants),
			"status": self.status,
			"review_status": self.review_status,
			"created_at": self.created_at,
		}


@dataclass
class SessionRecord:
	id: str
	tenant_id: str
	workspace_id: str
	owner: str
	protocol: str
	status: str = "active"
	participants: list[str] = field(default_factory=list)
	recording_requested: bool = False
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"workspace_id": self.workspace_id,
			"owner": self.owner,
			"protocol": self.protocol,
			"status": self.status,
			"participants": list(self.participants),
			"recording_requested": self.recording_requested,
			"created_at": self.created_at,
		}


@dataclass
class ArtifactRecord:
	id: str
	tenant_id: str
	workspace_id: str
	name: str
	owner: str
	artifact_type: str
	version: str = "v1"
	external_share: bool = False
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"workspace_id": self.workspace_id,
			"name": self.name,
			"owner": self.owner,
			"artifact_type": self.artifact_type,
			"version": self.version,
			"external_share": self.external_share,
			"created_at": self.created_at,
		}


@dataclass
class AnnotationRecord:
	id: str
	tenant_id: str
	artifact_id: str
	author: str
	body: str
	status: str = "open"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"artifact_id": self.artifact_id,
			"author": self.author,
			"body": self.body,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class DecisionRecord:
	id: str
	tenant_id: str
	annotation_id: str
	owner: str
	decision: str
	evidence: list[str]
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"annotation_id": self.annotation_id,
			"owner": self.owner,
			"decision": self.decision,
			"evidence": list(self.evidence),
			"created_at": self.created_at,
		}


@dataclass
class PresenceRecord:
	id: str
	tenant_id: str
	session_id: str
	participant_id: str
	status: str
	cursor: dict[str, Any] = field(default_factory=dict)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"participant_id": self.participant_id,
			"status": self.status,
			"cursor": dict(self.cursor),
			"updated_at": self.updated_at,
		}


@dataclass
class CollaborationAuditEventRecord:
	id: str
	tenant_id: str
	action: str
	subject_id: str
	actor_id: str
	details: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"action": self.action,
			"subject_id": self.subject_id,
			"actor_id": self.actor_id,
			"details": dict(self.details),
			"created_at": self.created_at,
		}


@dataclass
class CollaborationAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool = True
	human_approval_required: bool = False
	status: str = "active"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"owner": self.owner,
			"purpose": self.purpose,
			"contribution_disclosed": self.contribution_disclosed,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class ColbLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"status": self.status,
			"created_at": self.created_at,
		}


class CollaborationRuntime:
	"""Deterministic tenant-scoped collaboration lifecycle used by generated apps."""

	def __init__(self) -> None:
		self._workspaces: dict[str, WorkspaceRecord] = {}
		self._sessions: dict[str, SessionRecord] = {}
		self._artifacts: dict[str, ArtifactRecord] = {}
		self._annotations: dict[str, AnnotationRecord] = {}
		self._decisions: dict[str, DecisionRecord] = {}
		self._presence: dict[str, PresenceRecord] = {}
		self._audit_events: list[CollaborationAuditEventRecord] = []
		self._collaboration_agents: dict[str, CollaborationAgentRecord] = {}
		self._lifecycle_batches: dict[str, ColbLifecycleBatchRecord] = {}
		self._agent_runtimes = {_normalize_token(item) for item in SUPPORTED_COLB_AGENT_RUNTIMES}
		self._agent_roles = {_normalize_token(item) for item in SUPPORTED_COLB_AGENT_ROLES}
		self._privileged_agent_roles = {_normalize_token(item) for item in PRIVILEGED_COLB_AGENT_ROLES}
		self._lifecycle_operations = {
			_normalize_token(item)
			for item in get_capability_contract()["streaming"]["required_operations"]
		}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_workspace(
		self,
		tenant_id: str,
		workspace_id: str,
		name: str,
		owner: str,
		participants: list[str],
		retention_policy: str,
		external_participants: list[str] | None = None,
		external_policy_attached: bool = True,
		external_access_expiry_present: bool = True,
		membership_review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		external = list(external_participants or [])
		participant_list = list(dict.fromkeys([item for item in participants if item]))
		if owner and owner not in participant_list:
			participant_list.insert(0, owner)
		participant_count = len(participant_list) + len(external)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_workspace",
			"workspace_owner_assigned": bool(owner),
			"workspace_name_present": bool(name),
			"participant_present": bool(participant_list),
			"retention_policy_attached": bool(retention_policy),
			"external_participant_present": bool(external),
			"external_policy_attached": bool(external_policy_attached),
			"external_access_expiry_present": bool(external_access_expiry_present),
			"participant_count": participant_count,
			"membership_review_recorded": bool(membership_review_recorded),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		required_actions = [action["required_action"] for action in result["actions"] if action["decision"] == "require_review"]
		record = WorkspaceRecord(
			id=workspace_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			participants=participant_list,
			retention_policy=retention_policy,
			external_participants=external,
			status="pending_review" if required_actions else "active",
			review_status="required" if required_actions else "approved",
		)
		self._workspaces[self._key(tenant_id, workspace_id)] = record
		self._audit(tenant_id, "workspace_created", workspace_id, owner, {"required_actions": required_actions})
		return record.to_dict()

	def approve_workspace(self, tenant_id: str, workspace_id: str, reviewer: str) -> dict[str, Any]:
		workspace = self._require_workspace(tenant_id, workspace_id)
		workspace.status = "active"
		workspace.review_status = "approved"
		self._audit(tenant_id, "workspace_approved", workspace_id, reviewer, {})
		return workspace.to_dict()

	def start_session(
		self,
		tenant_id: str,
		session_id: str,
		workspace_id: str,
		owner: str,
		protocol: str = "websocket",
		secure_transport: bool = True,
		protocol_healthy: bool = True,
		recording_requested: bool = False,
		recording_retention_policy_attached: bool = True,
		event_bus_present: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		workspace = self._require_workspace(tenant_id, workspace_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "start_session",
			"workspace_present": bool(workspace),
			"session_owner_assigned": bool(owner),
			"session_owner_is_member": owner in set(workspace.participants) | set(workspace.external_participants) | {workspace.owner},
			"workspace_active": workspace.status == "active",
			"realtime_session": True,
			"secure_transport": bool(secure_transport),
			"protocol_health": "healthy" if protocol_healthy else "unhealthy",
			"event_bus_present": bool(event_bus_present),
			"recording_requested": bool(recording_requested),
			"recording_retention_policy_attached": bool(recording_retention_policy_attached),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = SessionRecord(session_id, tenant_id, workspace_id, owner, protocol, participants=[owner], recording_requested=recording_requested)
		self._sessions[self._key(tenant_id, session_id)] = record
		self._audit(tenant_id, "session_started", session_id, owner, {"protocol": protocol})
		return record.to_dict()

	def join_session(self, tenant_id: str, session_id: str, participant_id: str) -> dict[str, Any]:
		session = self._require_session(tenant_id, session_id)
		workspace = self._require_workspace(tenant_id, session.workspace_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "join_session",
			"participant_is_member": participant_id in set(workspace.participants) | set(workspace.external_participants) | {workspace.owner},
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		if participant_id not in session.participants:
			session.participants.append(participant_id)
		self._audit(tenant_id, "session_joined", session_id, participant_id, {})
		return session.to_dict()

	def share_artifact(
		self,
		tenant_id: str,
		artifact_id: str,
		workspace_id: str,
		name: str,
		owner: str,
		artifact_type: str,
		artifact_policy_attached: bool = True,
		version_history_enabled: bool = True,
		external_share: bool = False,
		dlp_check_completed: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_workspace(tenant_id, workspace_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "share_artifact",
			"artifact_policy_attached": bool(artifact_policy_attached),
			"version_history_enabled": bool(version_history_enabled),
			"external_share_requested": bool(external_share),
			"dlp_check_completed": bool(dlp_check_completed),
			"duplicate_artifact_id": self._key(tenant_id, artifact_id) in self._artifacts,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = ArtifactRecord(artifact_id, tenant_id, workspace_id, name, owner, artifact_type, external_share=external_share)
		self._artifacts[self._key(tenant_id, artifact_id)] = record
		self._audit(tenant_id, "artifact_shared", artifact_id, owner, {"workspace_id": workspace_id})
		return record.to_dict()

	def add_annotation(self, tenant_id: str, annotation_id: str, artifact_id: str, author: str, body: str) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		artifact = self._artifacts.get(self._key(tenant_id, artifact_id))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_annotation",
			"artifact_present": bool(artifact),
			"annotation_author_present": bool(author),
			"annotation_body_present": bool(body),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = AnnotationRecord(annotation_id, tenant_id, artifact_id, author, body)
		self._annotations[self._key(tenant_id, annotation_id)] = record
		self._audit(tenant_id, "annotation_added", annotation_id, author, {"artifact_id": artifact_id})
		return record.to_dict()

	def record_decision(self, tenant_id: str, decision_id: str, annotation_id: str, owner: str, decision: str, evidence: list[str]) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		annotation = self._annotations.get(self._key(tenant_id, annotation_id))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_decision",
			"annotation_present": bool(annotation),
			"decision_owner_present": bool(owner),
			"decision_evidence_present": bool(evidence),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = DecisionRecord(decision_id, tenant_id, annotation_id, owner, decision, list(evidence))
		self._decisions[self._key(tenant_id, decision_id)] = record
		self._audit(tenant_id, "decision_recorded", decision_id, owner, {"annotation_id": annotation_id})
		return record.to_dict()

	def update_presence(self, tenant_id: str, session_id: str, participant_id: str, status: str, cursor: dict[str, Any] | None = None) -> dict[str, Any]:
		session = self._require_session(tenant_id, session_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "update_presence",
			"session_active": session.status == "active",
			"participant_is_member": participant_id in session.participants,
		})
		self._raise_if_denied(result)
		record = PresenceRecord(stable_id("presence", tenant_id, session_id, participant_id), tenant_id, session_id, participant_id, status, dict(cursor or {}))
		self._presence[self._key(tenant_id, record.id)] = record
		return record.to_dict()

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"workspace_count": len(self.list_workspaces(tenant_id)),
			"active_workspace_count": sum(1 for item in self.list_workspaces(tenant_id) if item["status"] == "active"),
			"session_count": len(self.list_sessions(tenant_id)),
			"artifact_count": len(self.list_artifacts(tenant_id)),
			"annotation_count": len(self.list_annotations(tenant_id)),
			"decision_count": len(self.list_decisions(tenant_id)),
			"presence_count": len(self.list_presence(tenant_id)),
			"collaboration_agent_count": len(self.list_collaboration_agents(tenant_id)),
			"pending_agent_review_count": sum(1 for item in self.list_collaboration_agents(tenant_id) if item["status"] == "pending_review"),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": sum(1 for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def register_collaboration_agent(
		self,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		record_key = self._key(tenant_id, agent_id)
		if record_key in self._collaboration_agents:
			raise ValueError(f"collaboration_agent_already_exists:{agent_id}")
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_collaboration_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		if not str(name or "").strip():
			raise ValueError("collaboration_agent_name_required")
		record = CollaborationAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status="pending_review" if result["decision"] == "require_review" else "active",
		)
		self._collaboration_agents[record_key] = record
		self._audit(tenant_id, "collaboration_agent_registered", agent_id, record.owner, {**record.to_dict(), "rule_decision": result["decision"]})
		return record.to_dict()

	def validate_colb_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "collaboration_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("colb_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_colb_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_colb_lifecycle_batch",
			"event_stream": stream_value,
			"mutation_count": mutation_count,
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		accepted = result["decision"] == "allow"
		record_id = batch_id or f"colb-batch-{len(self._lifecycle_batches) + 1:06d}"
		record = ColbLifecycleBatchRecord(
			id=record_id,
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._key(tenant_id, record_id)] = record
		self._audit(tenant_id, f"colb_lifecycle_batch_{record.status}", record_id, "bytewax", record.to_dict())
		self._raise_if_denied(result)
		return record.to_dict()

	def list_workspaces(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._workspaces.values(), tenant_id, "id")

	def list_sessions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._sessions.values(), tenant_id, "id")

	def list_artifacts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._artifacts.values(), tenant_id, "id")

	def list_annotations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._annotations.values(), tenant_id, "id")

	def list_decisions(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._decisions.values(), tenant_id, "id")

	def list_presence(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._presence.values(), tenant_id, "id")

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._audit_events, tenant_id, "created_at")

	def list_collaboration_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._collaboration_agents.values(), tenant_id, "id")

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._lifecycle_batches.values(), tenant_id, "id")

	def _require_tenant(self, tenant_id: str) -> None:
		self._raise_if_denied(self.evaluate({"tenant_context_present": bool(tenant_id)}))

	def _require_workspace(self, tenant_id: str, workspace_id: str) -> WorkspaceRecord:
		try:
			return self._workspaces[self._key(tenant_id, workspace_id)]
		except KeyError as exc:
			raise KeyError(f"workspace_not_found:{workspace_id}") from exc

	def _require_session(self, tenant_id: str, session_id: str) -> SessionRecord:
		try:
			return self._sessions[self._key(tenant_id, session_id)]
		except KeyError as exc:
			raise KeyError(f"session_not_found:{session_id}") from exc

	def _audit(self, tenant_id: str, action: str, subject_id: str, actor_id: str, details: dict[str, Any]) -> None:
		event = CollaborationAuditEventRecord(stable_id("audit", tenant_id, action, subject_id, len(self._audit_events) + 1), tenant_id, action, subject_id, actor_id, details)
		self._audit_events.append(event)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "collaboration_policy_blocked") for action in result["actions"]))

	@staticmethod
	def _key(tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	@staticmethod
	def _tenant_sorted(records: Any, tenant_id: str | None, sort_key: str) -> list[dict[str, Any]]:
		items = list(records)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: getattr(item, sort_key))]


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
