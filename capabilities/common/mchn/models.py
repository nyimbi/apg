"""Domain models for APG Multi-Channel Output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a stable UTC timestamp for dependency-light runtime records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class OutputChannel:
	"""Tenant-owned output channel such as email, SMS, push, PDF, web, API, or print."""

	id: str
	tenant_id: str
	name: str
	channel_type: str
	owner: str
	provider_ref: str
	health: str
	fallback_channel_id: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"channel_type": self.channel_type,
			"owner": self.owner,
			"provider_ref": self.provider_ref,
			"health": self.health,
			"fallback_channel_id": self.fallback_channel_id,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class OutputTemplate:
	"""Approved localized output template."""

	id: str
	tenant_id: str
	name: str
	channel_types: tuple[str, ...]
	subject_template: str
	body_template: str
	locale: str
	theme_ref: str
	approved: bool
	approved_by: str
	status: str = "published"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"channel_types": list(self.channel_types),
			"subject_template": self.subject_template,
			"body_template": self.body_template,
			"locale": self.locale,
			"theme_ref": self.theme_ref,
			"approved": self.approved,
			"approved_by": self.approved_by,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class DeliveryPolicy:
	"""Tenant delivery policy for recipient, throttle, encryption, and compliance controls."""

	id: str
	tenant_id: str
	name: str
	max_recipients: int
	throttle_per_minute: int
	requires_encryption_for_sensitive: bool
	compliance_ref: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"max_recipients": self.max_recipients,
			"throttle_per_minute": self.throttle_per_minute,
			"requires_encryption_for_sensitive": self.requires_encryption_for_sensitive,
			"compliance_ref": self.compliance_ref,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class DeliveryRoute:
	"""Route tying templates, policies, and primary/fallback channels together."""

	id: str
	tenant_id: str
	name: str
	template_id: str
	primary_channel_id: str
	fallback_channel_ids: tuple[str, ...]
	policy_id: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"template_id": self.template_id,
			"primary_channel_id": self.primary_channel_id,
			"fallback_channel_ids": list(self.fallback_channel_ids),
			"policy_id": self.policy_id,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class RenderedOutput:
	"""Rendered output payload for a route and channel."""

	id: str
	tenant_id: str
	route_id: str
	template_id: str
	channel_id: str
	channel_type: str
	recipient_ref: str
	subject: str
	body: str
	output_format: str
	sensitive_output: bool
	output_encrypted: bool
	status: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"route_id": self.route_id,
			"template_id": self.template_id,
			"channel_id": self.channel_id,
			"channel_type": self.channel_type,
			"recipient_ref": self.recipient_ref,
			"subject": self.subject,
			"body": self.body,
			"output_format": self.output_format,
			"sensitive_output": self.sensitive_output,
			"output_encrypted": self.output_encrypted,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class DeliveryBatch:
	"""Reviewed delivery batch for one route and rendered-output set."""

	id: str
	tenant_id: str
	route_id: str
	requested_by: str
	recipient_count: int
	rendered_output_ids: tuple[str, ...]
	delivery_review_recorded: bool
	status: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"route_id": self.route_id,
			"requested_by": self.requested_by,
			"recipient_count": self.recipient_count,
			"rendered_output_ids": list(self.rendered_output_ids),
			"delivery_review_recorded": self.delivery_review_recorded,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class DeliveryReceipt:
	"""Delivery receipt from an output channel."""

	id: str
	tenant_id: str
	batch_id: str
	rendered_output_id: str
	channel_id: str
	recipient_ref: str
	delivery_state: str
	provider_message_id: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"batch_id": self.batch_id,
			"rendered_output_id": self.rendered_output_id,
			"channel_id": self.channel_id,
			"recipient_ref": self.recipient_ref,
			"delivery_state": self.delivery_state,
			"provider_message_id": self.provider_message_id,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class MchnAuditEvent:
	"""Governance event emitted by multi-channel output operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


MchnRecord = RenderedOutput


@dataclass(frozen=True)
class MchnAgent:
	"""Registered AI output agent for multi-channel operations."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

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
			"created_at": self.created_at,
		}
