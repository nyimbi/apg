"""Dependency-light lifecycle surface for the CKM RTC capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .capability_contract import (
	SUPPORTED_RTC_AGENT_ROLES,
	SUPPORTED_RTC_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


@dataclass
class RtcSession:
	id: str
	tenant_id: str
	name: str
	owner_id: str
	context_ref: str
	participant_policy: list[str]
	status: str = "active"
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner_id": self.owner_id,
			"context_ref": self.context_ref,
			"participant_policy": list(self.participant_policy),
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class RtcParticipant:
	id: str
	tenant_id: str
	session_id: str
	user_id: str
	role: str
	joined_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"user_id": self.user_id,
			"role": self.role,
			"joined_at": self.joined_at,
		}


@dataclass
class RtcMessage:
	id: str
	tenant_id: str
	session_id: str
	author_id: str
	body: str
	status: str
	decision: str
	reasons: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"author_id": self.author_id,
			"body": self.body,
			"status": self.status,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"created_at": self.created_at,
		}


@dataclass
class RtcDecision:
	id: str
	tenant_id: str
	session_id: str
	decision_text: str
	trace_ref: str
	owner_id: str
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"session_id": self.session_id,
			"decision_text": self.decision_text,
			"trace_ref": self.trace_ref,
			"owner_id": self.owner_id,
			"created_at": self.created_at,
		}


@dataclass
class RtcAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"status": self.status,
		}


class RtcLifecycleService:
	"""In-package RTC lifecycle engine for generated APG applications."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._sessions: dict[str, RtcSession] = {}
		self._participants: dict[str, RtcParticipant] = {}
		self._messages: dict[str, RtcMessage] = {}
		self._decisions: dict[str, RtcDecision] = {}
		self._agents: dict[str, RtcAgent] = {}
		self._presence: dict[tuple[str, str], dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def describe(self) -> dict[str, Any]:
		return get_capability_contract(self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_session(
		self,
		session_id: str,
		name: str,
		owner_id: str,
		context_ref: str,
		participant_policy: list[str],
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "create_session",
			"owner_present": bool(owner_id),
			"participant_policy_attached": bool(participant_policy),
		})
		self._raise_on_deny(result)
		session = RtcSession(
			id=session_id,
			tenant_id=self.tenant_id,
			name=name,
			owner_id=owner_id,
			context_ref=context_ref,
			participant_policy=list(participant_policy),
		)
		self._sessions[session_id] = session
		self._record_audit("rtc_session_created", session.to_dict())
		return session.to_dict()

	def join_session(self, session_id: str, user_id: str, role: str = "participant") -> dict[str, Any]:
		session = self._sessions[session_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "join_session",
			"participant_allowed": user_id in session.participant_policy or user_id == session.owner_id,
		})
		self._raise_on_deny(result)
		participant = RtcParticipant(
			id=f"participant-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			session_id=session_id,
			user_id=user_id,
			role=role,
		)
		self._participants[participant.id] = participant
		self._record_audit("rtc_participant_joined", participant.to_dict())
		return participant.to_dict()

	def update_presence(
		self,
		session_id: str,
		user_id: str,
		status: str,
		heartbeat_id: str,
		context_ref: str | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "update_presence",
			"heartbeat_present": bool(heartbeat_id),
		})
		self._raise_on_deny(result)
		presence = {
			"tenant_id": self.tenant_id,
			"session_id": session_id,
			"user_id": user_id,
			"status": status,
			"heartbeat_id": heartbeat_id,
			"context_ref": context_ref,
			"updated_at": _utc_now(),
		}
		self._presence[(session_id, user_id)] = presence
		self._record_audit("rtc_presence_updated", presence)
		return dict(presence)

	def post_message(
		self,
		session_id: str,
		author_id: str,
		body: str,
		sensitive_content_detected: bool = False,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		session = self._sessions[session_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "post_message",
			"session_active": session.status == "active",
			"sensitive_content_detected": sensitive_content_detected,
			"review_recorded": review_recorded,
		})
		status = "posted"
		if result["decision"] == "deny":
			status = "blocked"
		elif result["decision"] == "require_review":
			status = "review_required"
		message = RtcMessage(
			id=f"message-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			session_id=session_id,
			author_id=author_id,
			body=body,
			status=status,
			decision=result["decision"],
			reasons=[action.get("reason", "rtc_policy") for action in result["actions"]],
		)
		self._messages[message.id] = message
		self._record_audit("rtc_message_posted", message.to_dict())
		return message.to_dict()

	def start_screen_share(self, session_id: str, user_id: str, permission_granted: bool) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "start_screen_share",
			"screen_share_permission": permission_granted,
		})
		self._raise_on_deny(result)
		event = {"tenant_id": self.tenant_id, "session_id": session_id, "user_id": user_id, "status": "started"}
		self._record_audit("rtc_screen_share_started", event)
		return dict(event)

	def start_recording(self, session_id: str, user_id: str, consent_ref: str) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "start_recording",
			"recording_consent_present": bool(consent_ref),
		})
		self._raise_on_deny(result)
		event = {"tenant_id": self.tenant_id, "session_id": session_id, "user_id": user_id, "consent_ref": consent_ref, "status": "started"}
		self._record_audit("rtc_recording_started", event)
		return dict(event)

	def capture_decision(self, session_id: str, owner_id: str, decision_text: str, trace_ref: str) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "capture_decision",
			"decision_trace_present": bool(trace_ref),
		})
		self._raise_on_deny(result)
		decision = RtcDecision(
			id=f"decision-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			session_id=session_id,
			decision_text=decision_text,
			trace_ref=trace_ref,
			owner_id=owner_id,
		)
		self._decisions[decision.id] = decision
		self._record_audit("rtc_decision_captured", decision.to_dict())
		return decision.to_dict()

	def register_rtc_agent(
		self,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		runtime_token = _normalize_token(runtime)
		role_token = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"rtc_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": runtime_token in SUPPORTED_RTC_AGENT_RUNTIMES,
			"agent_role_supported": role_token in SUPPORTED_RTC_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_on_deny(result)
		agent = RtcAgent(
			id=agent_id or f"rtc-agent-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			name=name,
			runtime=runtime_token,
			role=role_token,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[agent.id] = agent
		self._record_audit("rtc_agent_registered", agent.to_dict())
		return agent.to_dict()

	def validate_batch_rtc_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"requested_operation": "batch_rtc_mutation",
			"event_stream": _normalize_token(event_stream),
		})

	def dashboard_summary(self) -> dict[str, Any]:
		return {
			"tenant_id": self.tenant_id,
			"session_count": len(self._sessions),
			"active_session_count": sum(1 for item in self._sessions.values() if item.status == "active"),
			"participant_count": len(self._participants),
			"message_count": len(self._messages),
			"decision_count": len(self._decisions),
			"rtc_agent_count": len(self._agents),
			"presence_count": len(self._presence),
			"audit_event_count": len(self._audit_events),
			"streaming": self.describe()["streaming"],
		}

	def list_audit_events(self) -> list[dict[str, Any]]:
		return list(self._audit_events)

	def _raise_on_deny(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "rtc_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "rtc_policy_blocked")

	def _record_audit(self, event_type: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"id": f"audit-{uuid4().hex[:12]}",
			"tenant_id": self.tenant_id,
			"event_type": event_type,
			"payload": dict(payload),
			"recorded_at": _utc_now(),
		})


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()
