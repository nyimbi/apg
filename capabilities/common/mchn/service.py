"""Service layer for APG Multi-Channel Output."""

from __future__ import annotations

from typing import Any

from .capability_contract import (
	SUPPORTED_MCHN_AGENT_ROLES,
	SUPPORTED_MCHN_AGENT_RUNTIMES,
	evaluate_capability_rules,
	event_stream_name,
	get_capability_contract,
)
from .models import DeliveryBatch, DeliveryPolicy, DeliveryReceipt, DeliveryRoute, MchnAgent, MchnAuditEvent, OutputChannel, OutputTemplate, RenderedOutput
from .output_runtime import OutputRuntime


class MchnService:
	"""Tenant-aware channel, template, route, render, delivery, and receipt service."""

	def __init__(self) -> None:
		self._channels: dict[str, OutputChannel] = {}
		self._templates: dict[str, OutputTemplate] = {}
		self._policies: dict[str, DeliveryPolicy] = {}
		self._routes: dict[str, DeliveryRoute] = {}
		self._rendered_outputs: dict[str, RenderedOutput] = {}
		self._batches: dict[str, DeliveryBatch] = {}
		self._receipts: dict[str, DeliveryReceipt] = {}
		self._audit_events: dict[str, MchnAuditEvent] = {}
		self._agents: dict[str, MchnAgent] = {}
		self._runtime = OutputRuntime()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_channel(
		self,
		channel_id: str,
		tenant_id: str,
		name: str,
		channel_type: str,
		owner: str,
		provider_ref: str,
		health: str = "healthy",
		fallback_channel_id: str = "",
		status: str = "active",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_channel",
			"channel_owner_assigned": bool(owner),
			"provider_ref_present": bool(provider_ref),
		})
		self._raise_if_denied(result)
		health_value = self._runtime.normalize_health(health)
		channel = OutputChannel(
			id=channel_id,
			tenant_id=tenant_id,
			name=name,
			channel_type=self._runtime.normalize_channel_type(channel_type),
			owner=owner,
			provider_ref=provider_ref,
			health=health_value,
			fallback_channel_id=fallback_channel_id,
			status=status,
		)
		self._channels[_state_key(tenant_id, channel_id)] = channel
		self._audit(tenant_id, channel_id, "channel_created", owner, result["decision"], reasons=self._reasons(result), metadata={"channel_type": channel.channel_type, "health": health_value})
		return channel.to_dict()

	def publish_template(
		self,
		template_id: str,
		tenant_id: str,
		name: str,
		channel_types: list[str] | tuple[str, ...],
		subject_template: str,
		body_template: str,
		locale: str,
		theme_ref: str,
		approved: bool,
		approved_by: str,
		status: str = "published",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_template",
			"template_approved": bool(approved),
			"template_approver_present": bool(approved_by),
			"template_content_present": bool(subject_template or body_template),
			"template_channel_present": bool(channel_types),
		})
		self._raise_if_denied(result)
		types = tuple(self._runtime.normalize_channel_type(channel_type) for channel_type in channel_types)
		template = OutputTemplate(
			id=template_id,
			tenant_id=tenant_id,
			name=name,
			channel_types=types,
			subject_template=subject_template,
			body_template=body_template,
			locale=locale,
			theme_ref=theme_ref,
			approved=approved,
			approved_by=approved_by,
			status=status,
		)
		self._templates[_state_key(tenant_id, template_id)] = template
		self._audit(tenant_id, template_id, "template_published", approved_by, result["decision"], reasons=self._reasons(result), metadata={"locale": locale, "channel_types": list(types)})
		return template.to_dict()

	def create_delivery_policy(
		self,
		policy_id: str,
		tenant_id: str,
		name: str,
		max_recipients: int,
		throttle_per_minute: int,
		requires_encryption_for_sensitive: bool,
		compliance_ref: str,
		status: str = "active",
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_delivery_policy",
			"recipient_limit_valid": max_recipients > 0,
			"throttle_policy_valid": throttle_per_minute > 0,
			"compliance_ref_present": bool(compliance_ref),
		})
		self._raise_if_denied(result)
		policy = DeliveryPolicy(
			id=policy_id,
			tenant_id=tenant_id,
			name=name,
			max_recipients=max_recipients,
			throttle_per_minute=throttle_per_minute,
			requires_encryption_for_sensitive=requires_encryption_for_sensitive,
			compliance_ref=compliance_ref,
			status=status,
		)
		self._policies[_state_key(tenant_id, policy_id)] = policy
		self._audit(tenant_id, policy_id, "delivery_policy_created", "system", result["decision"], reasons=self._reasons(result), metadata={"max_recipients": max_recipients, "throttle_per_minute": throttle_per_minute})
		return policy.to_dict()

	def create_route(
		self,
		route_id: str,
		tenant_id: str,
		name: str,
		template_id: str,
		primary_channel_id: str,
		fallback_channel_ids: list[str] | tuple[str, ...],
		policy_id: str,
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		self._require_template(template_id, tenant_id)
		self._require_channel(primary_channel_id, tenant_id)
		for channel_id in fallback_channel_ids:
			self._require_channel(channel_id, tenant_id)
		self._require_policy(policy_id, tenant_id)
		route = DeliveryRoute(
			id=route_id,
			tenant_id=tenant_id,
			name=name,
			template_id=template_id,
			primary_channel_id=primary_channel_id,
			fallback_channel_ids=tuple(fallback_channel_ids),
			policy_id=policy_id,
			status=status,
		)
		self._routes[_state_key(tenant_id, route_id)] = route
		self._audit(tenant_id, route_id, "route_created", "system", "allow", metadata={"primary_channel_id": primary_channel_id, "fallback_count": len(fallback_channel_ids)})
		return route.to_dict()

	def render_output(
		self,
		output_id: str,
		tenant_id: str,
		route_id: str,
		recipient_ref: str,
		variables: dict[str, Any],
		output_format: str,
		sensitive_output: bool = False,
		output_encrypted: bool = True,
	) -> dict[str, Any]:
		route = self._require_route(route_id, tenant_id)
		template = self._require_template(route.template_id, tenant_id)
		primary_channel = self._require_channel(route.primary_channel_id, tenant_id)
		fallback_channels = [self._require_channel(channel_id, tenant_id) for channel_id in route.fallback_channel_ids]
		selected_channel_id = self._runtime.selected_channel_id(primary_channel.to_dict(), [channel.to_dict() for channel in fallback_channels])
		selected_channel = self._require_channel(selected_channel_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "render_output",
			"recipient_ref_present": bool(recipient_ref),
			"sensitive_output": bool(sensitive_output),
			"output_encrypted": bool(output_encrypted),
		})
		self._raise_if_denied(result)
		if selected_channel.channel_type not in template.channel_types:
			raise PermissionError("template_channel_mismatch")
		rendered = RenderedOutput(
			id=output_id,
			tenant_id=tenant_id,
			route_id=route_id,
			template_id=template.id,
			channel_id=selected_channel.id,
			channel_type=selected_channel.channel_type,
			recipient_ref=recipient_ref,
			subject=self._runtime.render_template(template.subject_template, variables),
			body=self._runtime.render_template(template.body_template, variables),
			output_format=self._runtime.normalize_format(output_format),
			sensitive_output=sensitive_output,
			output_encrypted=output_encrypted,
			status=self._runtime.rendered_status(sensitive_output, output_encrypted),
		)
		self._rendered_outputs[_state_key(tenant_id, output_id)] = rendered
		self._audit(tenant_id, output_id, "output_rendered", recipient_ref, result["decision"], reasons=self._reasons(result), metadata={"channel_id": selected_channel.id, "format": rendered.output_format})
		return rendered.to_dict()

	def deliver_batch(
		self,
		batch_id: str,
		tenant_id: str,
		route_id: str,
		requested_by: str,
		rendered_output_ids: list[str] | tuple[str, ...],
		recipient_count: int,
		delivery_review_recorded: bool = False,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		route = self._require_route(route_id, tenant_id)
		policy = self._require_policy(route.policy_id, tenant_id)
		primary_channel = self._require_channel(route.primary_channel_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deliver_batch",
			"channel_health": primary_channel.health,
			"delivery_requested": True,
			"delivery_actor_present": bool(requested_by),
			"rendered_output_present": bool(rendered_output_ids),
			"recipient_count": recipient_count,
			"delivery_review_recorded": bool(delivery_review_recorded),
			"event_stream": event_stream_name(event_stream),
		})
		self._raise_if_review_required(result, delivery_review_recorded)
		if recipient_count > policy.max_recipients and not delivery_review_recorded:
			raise PermissionError("delivery_policy_review_required")
		for output_id in rendered_output_ids:
			output = self._require_rendered_output(output_id, tenant_id)
			if output.route_id != route_id:
				raise KeyError("rendered_output_route_mismatch")
			if output.status != "ready":
				raise PermissionError("rendered_output_not_ready")
		batch = DeliveryBatch(
			id=batch_id,
			tenant_id=tenant_id,
			route_id=route_id,
			requested_by=requested_by,
			recipient_count=recipient_count,
			rendered_output_ids=tuple(rendered_output_ids),
			delivery_review_recorded=delivery_review_recorded,
			status=self._runtime.batch_status(recipient_count, delivery_review_recorded),
		)
		self._batches[_state_key(tenant_id, batch_id)] = batch
		self._audit(tenant_id, batch_id, "delivery_batch_queued", requested_by, result["decision"], reasons=self._reasons(result), metadata={"recipient_count": recipient_count, "output_count": len(rendered_output_ids)})
		return batch.to_dict()

	def record_receipt(
		self,
		receipt_id: str,
		tenant_id: str,
		batch_id: str,
		rendered_output_id: str,
		delivery_state: str,
		provider_message_id: str,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_receipt",
			"provider_message_present": bool(provider_message_id),
		})
		self._raise_if_denied(result)
		batch = self._require_batch(batch_id, tenant_id)
		output = self._require_rendered_output(rendered_output_id, tenant_id)
		if rendered_output_id not in batch.rendered_output_ids:
			raise KeyError("rendered_output_not_in_batch")
		receipt = DeliveryReceipt(
			id=receipt_id,
			tenant_id=tenant_id,
			batch_id=batch_id,
			rendered_output_id=rendered_output_id,
			channel_id=output.channel_id,
			recipient_ref=output.recipient_ref,
			delivery_state=self._runtime.normalize_delivery_state(delivery_state),
			provider_message_id=provider_message_id,
		)
		self._receipts[_state_key(tenant_id, receipt_id)] = receipt
		self._audit(tenant_id, receipt_id, "delivery_receipt_recorded", output.recipient_ref, result["decision"], reasons=self._reasons(result), metadata={"delivery_state": receipt.delivery_state})
		return receipt.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		channel_id = str(metadata.get("channel_id") or f"channel-{record_id}")
		if _state_key(tenant_id, channel_id) not in self._channels:
			self.create_channel(
				channel_id=channel_id,
				tenant_id=tenant_id,
				name=str(metadata.get("channel_name") or "Default output channel"),
				channel_type=str(metadata.get("channel_type") or "email"),
				owner=str(metadata.get("owner") or "system"),
				provider_ref=str(metadata.get("provider_ref") or "provider://local"),
			)
		template_id = str(metadata.get("template_id") or f"template-{record_id}")
		if _state_key(tenant_id, template_id) not in self._templates:
			self.publish_template(
				template_id=template_id,
				tenant_id=tenant_id,
				name=str(metadata.get("template_name") or "Default output template"),
				channel_types=(str(metadata.get("channel_type") or "email"),),
				subject_template=str(metadata.get("subject_template") or "APG output"),
				body_template=str(metadata.get("body_template") or "$message"),
				locale=str(metadata.get("locale") or "en"),
				theme_ref=str(metadata.get("theme_ref") or "mchn_omnichannel_output"),
				approved=bool(metadata.get("approved", True)),
				approved_by=str(metadata.get("approved_by") or "system"),
			)
		policy_id = str(metadata.get("policy_id") or f"policy-{record_id}")
		if _state_key(tenant_id, policy_id) not in self._policies:
			self.create_delivery_policy(
				policy_id=policy_id,
				tenant_id=tenant_id,
				name=str(metadata.get("policy_name") or "Default output policy"),
				max_recipients=int(metadata.get("max_recipients", 10000)),
				throttle_per_minute=int(metadata.get("throttle_per_minute", 1000)),
				requires_encryption_for_sensitive=True,
				compliance_ref=str(metadata.get("compliance_ref") or "compliance://default"),
			)
		route_id = str(metadata.get("route_id") or f"route-{record_id}")
		if _state_key(tenant_id, route_id) not in self._routes:
			self.create_route(
				route_id=route_id,
				tenant_id=tenant_id,
				name=str(metadata.get("route_name") or "Default output route"),
				template_id=template_id,
				primary_channel_id=channel_id,
				fallback_channel_ids=tuple(metadata.get("fallback_channel_ids") or ()),
				policy_id=policy_id,
			)
		return self.render_output(
			output_id=record_id,
			tenant_id=tenant_id,
			route_id=route_id,
			recipient_ref=str(metadata.get("recipient_ref") or "recipient"),
			variables={"message": str(metadata.get("message") or record_id), **dict(metadata.get("variables") or {})},
			output_format=str(metadata.get("output_format") or "text"),
			sensitive_output=bool(metadata.get("sensitive_output", False)),
			output_encrypted=bool(metadata.get("output_encrypted", True)),
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_rendered_outputs(tenant_id)

	def list_channels(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._channels, tenant_id)

	def list_templates(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._templates, tenant_id)

	def list_policies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._policies, tenant_id)

	def list_routes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._routes, tenant_id)

	def list_rendered_outputs(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._rendered_outputs, tenant_id)

	def list_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._batches, tenant_id)

	def list_receipts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._receipts, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def register_mchn_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		contribution_disclosed: bool = True,
		agent_id: str | None = None,
	) -> dict[str, Any]:
		normalized_runtime = _normalize_token(runtime)
		normalized_role = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"mchn_agent_present": True,
			"agent_registered": True,
			"agent_runtime_supported": normalized_runtime in SUPPORTED_MCHN_AGENT_RUNTIMES,
			"agent_role_supported": normalized_role in SUPPORTED_MCHN_AGENT_ROLES,
			"agent_scope_present": bool(scope),
			"agent_contribution_disclosed": contribution_disclosed,
		})
		self._raise_if_denied(result)
		agent = MchnAgent(
			id=agent_id or f"mchn-agent-{len(self._agents) + 1:06d}",
			tenant_id=tenant_id,
			name=name,
			runtime=normalized_runtime,
			role=normalized_role,
			scope=scope,
			contribution_disclosed=contribution_disclosed,
		)
		self._agents[_state_key(tenant_id, agent.id)] = agent
		self._audit(tenant_id, agent.id, "mchn_agent_registered", name, result["decision"], metadata=agent.to_dict())
		return agent.to_dict()

	def validate_batch_output_mutation(self, event_stream: str) -> dict[str, Any]:
		return self.evaluate({
			"tenant_context_present": True,
			"requested_operation": "batch_output_mutation",
			"event_stream": event_stream,
		})

	def list_mchn_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._agents, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		channels = self.list_channels(tenant_id)
		batches = self.list_batches(tenant_id)
		receipts = self.list_receipts(tenant_id)
		return {
			"tenant_id": tenant_id,
			"channel_count": len(channels),
			"template_count": len(self.list_templates(tenant_id)),
			"policy_count": len(self.list_policies(tenant_id)),
			"route_count": len(self.list_routes(tenant_id)),
			"rendered_output_count": len(self.list_rendered_outputs(tenant_id)),
			"delivery_batch_count": len(batches),
			"receipt_count": len(receipts),
			"mchn_agent_count": len(self.list_mchn_agents(tenant_id)),
			"unhealthy_channel_count": len([channel for channel in channels if channel["health"] == "unhealthy"]),
			"large_batch_count": len([batch for batch in batches if batch["recipient_count"] > 10000]),
			"failed_receipt_count": len([receipt for receipt in receipts if receipt["delivery_state"] in {"failed", "bounced"}]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
			"streaming": self.describe(tenant_id)["streaming"],
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_channel(self, channel_id: str, tenant_id: str) -> OutputChannel:
		channel = self._channels.get(_state_key(tenant_id, channel_id))
		if channel is None or channel.tenant_id != tenant_id:
			raise KeyError("output_channel_not_found")
		return channel

	def _require_template(self, template_id: str, tenant_id: str) -> OutputTemplate:
		template = self._templates.get(_state_key(tenant_id, template_id))
		if template is None or template.tenant_id != tenant_id:
			raise KeyError("output_template_not_found")
		return template

	def _require_policy(self, policy_id: str, tenant_id: str) -> DeliveryPolicy:
		policy = self._policies.get(_state_key(tenant_id, policy_id))
		if policy is None or policy.tenant_id != tenant_id:
			raise KeyError("delivery_policy_not_found")
		return policy

	def _require_route(self, route_id: str, tenant_id: str) -> DeliveryRoute:
		route = self._routes.get(_state_key(tenant_id, route_id))
		if route is None or route.tenant_id != tenant_id:
			raise KeyError("delivery_route_not_found")
		return route

	def _require_rendered_output(self, output_id: str, tenant_id: str) -> RenderedOutput:
		output = self._rendered_outputs.get(_state_key(tenant_id, output_id))
		if output is None or output.tenant_id != tenant_id:
			raise KeyError("rendered_output_not_found")
		return output

	def _require_batch(self, batch_id: str, tenant_id: str) -> DeliveryBatch:
		batch = self._batches.get(_state_key(tenant_id, batch_id))
		if batch is None or batch.tenant_id != tenant_id:
			raise KeyError("delivery_batch_not_found")
		return batch

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "output_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any], review_recorded: bool) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review" and not review_recorded:
			raise PermissionError(", ".join(self._reasons(result)) or "output_review_required")

	def _audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = self._runtime.stable_id("audit", {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"index": len(self._audit_events),
		})
		self._audit_events[event_id] = MchnAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "output_policy_blocked") for action in result.get("actions", ()))


def _normalize_token(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def _state_key(tenant_id: str, item_id: str) -> str:
	return f"{tenant_id}:{item_id}"
