"""Dependency-light notification runtime for package-backed NTFY composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: object) -> str:
	key = "|".join(str(part) for part in parts)
	return f"{prefix}_{sha1(key.encode('utf-8')).hexdigest()[:12]}"


@dataclass
class RecipientPreferenceRecord:
	id: str
	tenant_id: str
	recipient_id: str
	addresses: dict[str, str]
	preferred_channels: list[str]
	opted_in: bool = False
	unsubscribed: bool = False
	quiet_hours: dict[str, str] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"recipient_id": self.recipient_id,
			"addresses": dict(self.addresses),
			"preferred_channels": list(self.preferred_channels),
			"opted_in": self.opted_in,
			"unsubscribed": self.unsubscribed,
			"quiet_hours": dict(self.quiet_hours),
			"created_at": self.created_at,
		}


@dataclass
class ChannelProviderRecord:
	id: str
	tenant_id: str
	channel: str
	provider: str
	owner: str
	healthy: bool = True
	fallback_channel: str | None = None
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"channel": self.channel,
			"provider": self.provider,
			"owner": self.owner,
			"healthy": self.healthy,
			"fallback_channel": self.fallback_channel,
			"created_at": self.created_at,
		}


@dataclass
class NotificationTemplateRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	locale: str
	channels: list[str]
	content: dict[str, str]
	approved: bool = False
	version: str = "v1"
	created_at: str = field(default_factory=utc_now)
	approved_at: str | None = None
	approved_by: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"locale": self.locale,
			"channels": list(self.channels),
			"content": dict(self.content),
			"approved": self.approved,
			"version": self.version,
			"created_at": self.created_at,
			"approved_at": self.approved_at,
			"approved_by": self.approved_by,
		}


@dataclass
class DeliveryRecord:
	id: str
	tenant_id: str
	template_id: str
	recipient_id: str
	channel: str
	message_class: str
	priority: str
	status: str
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	idempotency_key: str | None = None
	created_at: str = field(default_factory=utc_now)
	delivered_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"template_id": self.template_id,
			"recipient_id": self.recipient_id,
			"channel": self.channel,
			"message_class": self.message_class,
			"priority": self.priority,
			"status": self.status,
			"required_actions": list(self.required_actions),
			"matched_rules": list(self.matched_rules),
			"idempotency_key": self.idempotency_key,
			"created_at": self.created_at,
			"delivered_at": self.delivered_at,
		}


@dataclass
class CampaignRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	template_id: str
	audience: list[str]
	channels: list[str]
	message_class: str = "marketing"
	approved: bool = False
	status: str = "draft"
	created_at: str = field(default_factory=utc_now)
	approved_at: str | None = None
	approved_by: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"template_id": self.template_id,
			"audience": list(self.audience),
			"channels": list(self.channels),
			"message_class": self.message_class,
			"approved": self.approved,
			"status": self.status,
			"created_at": self.created_at,
			"approved_at": self.approved_at,
			"approved_by": self.approved_by,
		}


@dataclass
class NotificationAuditEventRecord:
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


class NotificationRuntime:
	"""Deterministic tenant-scoped notification lifecycle used by generated apps."""

	def __init__(self) -> None:
		self._preferences: dict[str, RecipientPreferenceRecord] = {}
		self._channels: dict[str, ChannelProviderRecord] = {}
		self._templates: dict[str, NotificationTemplateRecord] = {}
		self._deliveries: dict[str, DeliveryRecord] = {}
		self._campaigns: dict[str, CampaignRecord] = {}
		self._audit_events: list[NotificationAuditEventRecord] = []
		self._idempotency: set[str] = set()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_channel(self, tenant_id: str, channel: str, provider: str, owner: str, healthy: bool = True, fallback_channel: str | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_channel",
			"provider_present": bool(provider),
			"channel_owner_assigned": bool(owner),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = ChannelProviderRecord(stable_id("channel", tenant_id, channel), tenant_id, channel, provider, owner, healthy, fallback_channel)
		self._channels[self._key(tenant_id, channel)] = record
		self._audit(tenant_id, "channel_registered", record.id, owner, record.to_dict())
		return record.to_dict()

	def register_preference(self, tenant_id: str, recipient_id: str, addresses: dict[str, str], preferred_channels: list[str], opted_in: bool = False, unsubscribed: bool = False, quiet_hours: dict[str, str] | None = None) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_preference",
			"recipient_address_present": bool(addresses),
			"channel_preferences_present": bool(preferred_channels),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = RecipientPreferenceRecord(stable_id("recipient", tenant_id, recipient_id), tenant_id, recipient_id, addresses, preferred_channels, opted_in, unsubscribed, dict(quiet_hours or {}))
		self._preferences[self._key(tenant_id, recipient_id)] = record
		self._audit(tenant_id, "preference_registered", record.id, recipient_id, {"opted_in": opted_in, "unsubscribed": unsubscribed})
		return record.to_dict()

	def register_template(self, tenant_id: str, template_id: str, name: str, owner: str, locale: str, channels: list[str], content: dict[str, str], approved: bool = False, version: str = "v1") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_template",
			"template_owner_assigned": bool(owner),
			"template_name_present": bool(name),
			"template_locale_present": bool(locale),
			"template_content_present": bool(content),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = NotificationTemplateRecord(template_id, tenant_id, name, owner, locale, channels, content, approved, version)
		if approved:
			record.approved_at = utc_now()
			record.approved_by = owner
		self._templates[self._key(tenant_id, template_id)] = record
		self._audit(tenant_id, "template_registered", template_id, owner, {"approved": approved, "channels": channels})
		return record.to_dict()

	def approve_template(self, tenant_id: str, template_id: str, approved_by: str) -> dict[str, Any]:
		template = self._require_template(tenant_id, template_id)
		template.approved = True
		template.approved_at = utc_now()
		template.approved_by = approved_by
		self._audit(tenant_id, "template_approved", template_id, approved_by, {})
		return template.to_dict()

	def send_message(
		self,
		tenant_id: str,
		template_id: str,
		recipient_id: str,
		channel: str,
		message_class: str = "transactional",
		priority: str = "normal",
		sensitive_payload: bool = False,
		payload_encrypted: bool = False,
		idempotency_key: str | None = None,
		webhook_signature_present: bool = True,
		event_bus_present: bool = True,
		quiet_hours_active: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		template = self._require_template(tenant_id, template_id)
		preference = self._preferences.get(self._key(tenant_id, recipient_id))
		provider = self._channels.get(self._key(tenant_id, channel))
		idempotency = idempotency_key or stable_id("idem", tenant_id, template_id, recipient_id, channel, len(self._deliveries) + 1)
		scoped_idempotency = self._key(tenant_id, idempotency)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "send_message",
			"template_present": bool(template),
			"template_approved": template.approved,
			"message_class": message_class,
			"recipient_opted_in": bool(preference and preference.opted_in),
			"recipient_unsubscribed": bool(preference and preference.unsubscribed),
			"sensitive_payload": sensitive_payload,
			"payload_encrypted": payload_encrypted,
			"provider_health": "healthy" if provider and provider.healthy else "unhealthy",
			"delivery_requested": True,
			"channel_enabled": bool(provider),
			"channel": channel,
			"webhook_signature_present": webhook_signature_present,
			"event_bus_present": event_bus_present,
			"audit_event_recorded": True,
			"delivery_ttl_present": True,
			"duplicate_idempotency_key": scoped_idempotency in self._idempotency,
			"quiet_hours_active": quiet_hours_active,
			"priority_override_allowed": priority in {"urgent", "critical"},
		})
		self._raise_if_denied(result)
		required_actions = [action["required_action"] for action in result["actions"] if action["decision"] == "require_review"]
		status = "review_required" if required_actions else "delivered"
		record = DeliveryRecord(
			id=stable_id("delivery", tenant_id, idempotency),
			tenant_id=tenant_id,
			template_id=template_id,
			recipient_id=recipient_id,
			channel=channel,
			message_class=message_class,
			priority=priority,
			status=status,
			required_actions=required_actions,
			matched_rules=list(result["matched_rules"]),
			idempotency_key=idempotency,
			delivered_at=utc_now() if status == "delivered" else None,
		)
		self._deliveries[record.id] = record
		self._idempotency.add(scoped_idempotency)
		self._audit(tenant_id, "message_sent", record.id, recipient_id, {"status": status, "channel": channel})
		return record.to_dict()

	def create_campaign(self, tenant_id: str, campaign_id: str, name: str, owner: str, template_id: str, audience: list[str], channels: list[str], message_class: str = "marketing") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_template(tenant_id, template_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_campaign",
			"audience_present": bool(audience),
			"campaign_owner_assigned": bool(owner),
			"state_change_requested": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		record = CampaignRecord(campaign_id, tenant_id, name, owner, template_id, audience, channels, message_class)
		self._campaigns[self._key(tenant_id, campaign_id)] = record
		self._audit(tenant_id, "campaign_created", campaign_id, owner, {"recipient_count": len(audience)})
		return record.to_dict()

	def approve_campaign(self, tenant_id: str, campaign_id: str, approved_by: str) -> dict[str, Any]:
		campaign = self._require_campaign(tenant_id, campaign_id)
		campaign.approved = True
		campaign.approved_at = utc_now()
		campaign.approved_by = approved_by
		campaign.status = "approved"
		self._audit(tenant_id, "campaign_approved", campaign_id, approved_by, {})
		return campaign.to_dict()

	def send_campaign(self, tenant_id: str, campaign_id: str, batch_review_recorded: bool = False) -> dict[str, Any]:
		campaign = self._require_campaign(tenant_id, campaign_id)
		template = self._require_template(tenant_id, campaign.template_id)
		preferences = [self._preferences.get(self._key(tenant_id, recipient_id)) for recipient_id in campaign.audience]
		all_recipients_opted_in = all(preference and preference.opted_in for preference in preferences)
		any_recipient_unsubscribed = any(preference and preference.unsubscribed for preference in preferences)
		providers = [self._channels.get(self._key(tenant_id, channel)) for channel in campaign.channels]
		all_channels_enabled = bool(providers) and all(provider for provider in providers)
		all_providers_healthy = all(provider and provider.healthy for provider in providers)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "send_campaign",
			"template_approved": template.approved,
			"campaign_approved": campaign.approved,
			"message_class": campaign.message_class,
			"recipient_opted_in": all_recipients_opted_in,
			"recipient_unsubscribed": any_recipient_unsubscribed,
			"recipient_count": len(campaign.audience),
			"batch_review_recorded": batch_review_recorded,
			"delivery_requested": True,
			"channel_enabled": all_channels_enabled,
			"provider_health": "healthy" if all_providers_healthy else "unhealthy",
			"event_bus_present": True,
			"audit_event_recorded": True,
		})
		self._raise_if_denied(result)
		required_actions = [action["required_action"] for action in result["actions"] if action["decision"] == "require_review"]
		campaign.status = "review_required" if required_actions else "sent"
		self._audit(tenant_id, "campaign_sent", campaign_id, campaign.owner, {"status": campaign.status, "recipient_count": len(campaign.audience)})
		return {"campaign": campaign.to_dict(), "required_actions": required_actions, "matched_rules": list(result["matched_rules"])}

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		deliveries = self.list_deliveries(tenant_id)
		return {
			"tenant_id": tenant_id,
			"recipient_count": len(self.list_preferences(tenant_id)),
			"channel_count": len(self.list_channels(tenant_id)),
			"template_count": len(self.list_templates(tenant_id)),
			"approved_template_count": sum(1 for template in self.list_templates(tenant_id) if template["approved"]),
			"campaign_count": len(self.list_campaigns(tenant_id)),
			"delivery_count": len(deliveries),
			"delivered_count": sum(1 for delivery in deliveries if delivery["status"] == "delivered"),
			"review_required_count": sum(1 for delivery in deliveries if delivery["status"] == "review_required"),
		}

	def list_preferences(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._preferences.values(), tenant_id, "recipient_id")

	def list_channels(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._channels.values(), tenant_id, "channel")

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._templates.values(), tenant_id, "id")

	def list_deliveries(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._deliveries.values(), tenant_id, "created_at")

	def list_campaigns(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._campaigns.values(), tenant_id, "id")

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._tenant_sorted(self._audit_events, tenant_id, "created_at")

	def _require_tenant(self, tenant_id: str) -> None:
		self._raise_if_denied(self.evaluate({"tenant_context_present": bool(tenant_id)}))

	def _require_template(self, tenant_id: str, template_id: str) -> NotificationTemplateRecord:
		try:
			return self._templates[self._key(tenant_id, template_id)]
		except KeyError as exc:
			raise KeyError(f"template_not_found:{template_id}") from exc

	def _require_campaign(self, tenant_id: str, campaign_id: str) -> CampaignRecord:
		try:
			return self._campaigns[self._key(tenant_id, campaign_id)]
		except KeyError as exc:
			raise KeyError(f"campaign_not_found:{campaign_id}") from exc

	def _audit(self, tenant_id: str, action: str, subject_id: str, actor_id: str, details: dict[str, Any]) -> None:
		event = NotificationAuditEventRecord(stable_id("audit", tenant_id, action, subject_id, len(self._audit_events) + 1), tenant_id, action, subject_id, actor_id, details)
		self._audit_events.append(event)

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(action.get("reason", "notification_policy_blocked") for action in result["actions"]))

	@staticmethod
	def _key(tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	@staticmethod
	def _tenant_sorted(records: Any, tenant_id: str | None, sort_key: str) -> list[dict[str, Any]]:
		items = list(records)
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: getattr(item, sort_key))]
