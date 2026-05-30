"""Dependency-light lifecycle surface for the CKM notification capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .capability_contract import (
	SUPPORTED_CHANNELS,
	SUPPORTED_NOTIFICATION_AGENT_ROLES,
	SUPPORTED_NOTIFICATION_AGENT_RUNTIMES,
	evaluate_capability_rules,
	get_capability_contract,
)


EXTERNAL_CHANNELS = {"email", "sms", "push", "voice", "webhook", "whatsapp", "slack", "teams", "web_push"}


@dataclass
class NotificationTemplate:
	id: str
	tenant_id: str
	name: str
	channels: list[str]
	content: dict[str, str]
	variable_schema: dict[str, Any]
	locale: str = "en"
	status: str = "draft"
	approved: bool = False
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"channels": list(self.channels),
			"content": dict(self.content),
			"variable_schema": dict(self.variable_schema),
			"locale": self.locale,
			"status": self.status,
			"approved": self.approved,
			"created_at": self.created_at,
		}


@dataclass
class NotificationPreference:
	id: str
	tenant_id: str
	recipient_id: str
	allowed_channels: list[str]
	suppressed_topics: list[str] = field(default_factory=list)
	quiet_hours: dict[str, Any] = field(default_factory=dict)
	consent_refs: dict[str, str] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"recipient_id": self.recipient_id,
			"allowed_channels": list(self.allowed_channels),
			"suppressed_topics": list(self.suppressed_topics),
			"quiet_hours": dict(self.quiet_hours),
			"consent_refs": dict(self.consent_refs),
		}


@dataclass
class NotificationAgent:
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


@dataclass
class NotificationProvider:
	id: str
	tenant_id: str
	name: str
	channel: str
	secret_ref: str
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"channel": self.channel,
			"secret_ref": self.secret_ref,
			"status": self.status,
		}


@dataclass
class NotificationDelivery:
	id: str
	tenant_id: str
	template_id: str
	recipient_id: str
	channels: list[str]
	topic: str
	status: str
	decision: str
	reasons: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=lambda: _utc_now())

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"template_id": self.template_id,
			"recipient_id": self.recipient_id,
			"channels": list(self.channels),
			"topic": self.topic,
			"status": self.status,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"created_at": self.created_at,
		}


class NotificationLifecycleService:
	"""In-package lifecycle engine for templates, preferences, deliveries, and agents."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self._templates: dict[str, NotificationTemplate] = {}
		self._preferences: dict[tuple[str, str], NotificationPreference] = {}
		self._agents: dict[str, NotificationAgent] = {}
		self._providers: dict[str, NotificationProvider] = {}
		self._deliveries: dict[str, NotificationDelivery] = {}
		self._audit_events: list[dict[str, Any]] = []

	def describe(self) -> dict[str, Any]:
		return get_capability_contract(self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_template(
		self,
		template_id: str,
		name: str,
		channels: list[str],
		content: dict[str, str],
		variable_schema: dict[str, Any],
		locale: str = "en",
	) -> dict[str, Any]:
		self._require_known_channels(channels)
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "create_template",
			"channel_content_complete": all(bool(content.get(channel)) for channel in channels),
		})
		self._raise_on_deny(result)
		template = NotificationTemplate(
			id=template_id,
			tenant_id=self.tenant_id,
			name=name,
			channels=list(channels),
			content=dict(content),
			variable_schema=dict(variable_schema),
			locale=locale,
		)
		self._templates[template_id] = template
		self._record_audit("notification_template_created", template.to_dict())
		return template.to_dict()

	def approve_template(self, template_id: str, reviewer_id: str) -> dict[str, Any]:
		template = self._templates[template_id]
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "activate_template",
			"variable_schema_attached": bool(template.variable_schema),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_on_deny(result)
		template.status = "active"
		template.approved = True
		payload = template.to_dict() | {"reviewer_id": reviewer_id}
		self._record_audit("notification_template_approved", payload)
		return template.to_dict()

	def set_preference(
		self,
		recipient_id: str,
		allowed_channels: list[str],
		suppressed_topics: list[str] | None = None,
		quiet_hours: dict[str, Any] | None = None,
		consent_refs: dict[str, str] | None = None,
	) -> dict[str, Any]:
		self._require_known_channels(allowed_channels)
		preference = NotificationPreference(
			id=f"pref-{recipient_id}",
			tenant_id=self.tenant_id,
			recipient_id=recipient_id,
			allowed_channels=list(allowed_channels),
			suppressed_topics=list(suppressed_topics or []),
			quiet_hours=dict(quiet_hours or {}),
			consent_refs=dict(consent_refs or {}),
		)
		self._preferences[(self.tenant_id, recipient_id)] = preference
		self._record_audit("notification_preference_updated", preference.to_dict())
		return preference.to_dict()

	def register_notification_agent(
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
			"notification_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": runtime_token in SUPPORTED_NOTIFICATION_AGENT_RUNTIMES,
			"agent_role_supported": role_token in SUPPORTED_NOTIFICATION_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_on_deny(result)
		agent = NotificationAgent(
			id=agent_id or f"not-agent-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			name=name,
			runtime=runtime_token,
			role=role_token,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[agent.id] = agent
		self._record_audit("notification_agent_registered", agent.to_dict())
		return agent.to_dict()

	def register_provider(
		self,
		provider_id: str,
		name: str,
		channel: str,
		secret_ref: str,
		status: str = "active",
	) -> dict[str, Any]:
		self._require_known_channels([channel])
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"operation": "register_provider",
			"provider_secret_ref_present": bool(secret_ref),
		})
		self._raise_on_deny(result)
		provider = NotificationProvider(
			id=provider_id,
			tenant_id=self.tenant_id,
			name=name,
			channel=channel,
			secret_ref=secret_ref,
			status=status,
		)
		self._providers[provider_id] = provider
		self._record_audit("notification_provider_registered", provider.to_dict())
		return provider.to_dict()

	def request_delivery(
		self,
		template_id: str,
		recipient_id: str,
		channels: list[str],
		topic: str,
		within_quiet_hours: bool = False,
		urgent_override_present: bool = False,
	) -> dict[str, Any]:
		self._require_known_channels(channels)
		template = self._templates[template_id]
		if not template.approved:
			raise PermissionError("template_approval_required")
		preference = self._preferences.get((self.tenant_id, recipient_id))
		recipient_suppressed = bool(preference and topic in preference.suppressed_topics)
		channel_allowed = not preference or all(channel in preference.allowed_channels for channel in channels)
		consent_present = bool(preference and all(channel in preference.consent_refs for channel in channels if channel in EXTERNAL_CHANNELS))
		deferral_scheduled = within_quiet_hours and not urgent_override_present
		result = self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"delivery_requested": True,
			"external_channel_requested": any(channel in EXTERNAL_CHANNELS for channel in channels),
			"recipient_consent_present": consent_present,
			"channel_allowed_by_preference": channel_allowed,
			"recipient_suppressed": recipient_suppressed,
			"within_quiet_hours": within_quiet_hours,
			"deferral_scheduled": deferral_scheduled,
			"urgent_override_present": urgent_override_present,
		})
		status = "queued"
		if result["decision"] == "deny":
			status = "blocked"
		elif result["decision"] == "require_review":
			status = "deferred"
		elif deferral_scheduled:
			status = "deferred"
		delivery = NotificationDelivery(
			id=f"delivery-{uuid4().hex[:12]}",
			tenant_id=self.tenant_id,
			template_id=template_id,
			recipient_id=recipient_id,
			channels=list(channels),
			topic=topic,
			status=status,
			decision=result["decision"],
			reasons=[action.get("reason", "notification_policy") for action in result["actions"]],
		)
		self._deliveries[delivery.id] = delivery
		self._record_audit("notification_delivery_requested", delivery.to_dict())
		return delivery.to_dict()

	def validate_batch_notification_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": bool(self.tenant_id),
			"requested_operation": "batch_notification_mutation",
			"event_stream": _normalize_token(event_stream),
		})

	def dashboard_summary(self) -> dict[str, Any]:
		return {
			"tenant_id": self.tenant_id,
			"template_count": len(self._templates),
			"approved_template_count": sum(1 for item in self._templates.values() if item.approved),
			"delivery_count": len(self._deliveries),
			"blocked_delivery_count": sum(1 for item in self._deliveries.values() if item.status == "blocked"),
			"notification_agent_count": len(self._agents),
			"provider_count": len(self._providers),
			"audit_event_count": len(self._audit_events),
			"streaming": self.describe()["streaming"],
		}

	def list_audit_events(self) -> list[dict[str, Any]]:
		return list(self._audit_events)

	def _require_known_channels(self, channels: list[str]) -> None:
		unknown = sorted(set(channels) - set(SUPPORTED_CHANNELS))
		if unknown:
			raise ValueError(f"unsupported_notification_channel: {', '.join(unknown)}")

	def _raise_on_deny(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "notification_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "notification_policy_blocked")

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
