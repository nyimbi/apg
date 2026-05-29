"""Service layer for APG Multi-Channel Output."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import DeliveryBatch, DeliveryPolicy, DeliveryReceipt, DeliveryRoute, MchnAuditEvent, OutputChannel, OutputTemplate, RenderedOutput
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
		})
		self._raise_if_denied(result)
		if not provider_ref:
			raise PermissionError("channel_provider_required")
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
		self._channels[channel_id] = channel
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
		})
		self._raise_if_denied(result)
		if not subject_template and not body_template:
			raise PermissionError("template_content_required")
		if approved and not approved_by:
			raise PermissionError("template_approver_required")
		types = tuple(self._runtime.normalize_channel_type(channel_type) for channel_type in channel_types)
		if not types:
			raise PermissionError("template_channel_required")
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
		self._templates[template_id] = template
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
		self._require_tenant(tenant_id)
		if max_recipients <= 0:
			raise PermissionError("recipient_policy_required")
		if throttle_per_minute <= 0:
			raise PermissionError("throttle_policy_required")
		if not compliance_ref:
			raise PermissionError("compliance_policy_required")
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
		self._policies[policy_id] = policy
		self._audit(tenant_id, policy_id, "delivery_policy_created", "system", "allow", metadata={"max_recipients": max_recipients, "throttle_per_minute": throttle_per_minute})
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
		self._routes[route_id] = route
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
			"sensitive_output": bool(sensitive_output),
			"output_encrypted": bool(output_encrypted),
		})
		self._raise_if_denied(result)
		if selected_channel.channel_type not in template.channel_types:
			raise PermissionError("template_channel_mismatch")
		if not recipient_ref:
			raise PermissionError("recipient_policy_required")
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
		self._rendered_outputs[output_id] = rendered
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
	) -> dict[str, Any]:
		route = self._require_route(route_id, tenant_id)
		policy = self._require_policy(route.policy_id, tenant_id)
		primary_channel = self._require_channel(route.primary_channel_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"channel_health": primary_channel.health,
			"delivery_requested": True,
			"recipient_count": recipient_count,
			"delivery_review_recorded": bool(delivery_review_recorded),
		})
		self._raise_if_review_required(result, delivery_review_recorded)
		if not requested_by:
			raise PermissionError("delivery_actor_required")
		if recipient_count <= 0:
			raise PermissionError("recipient_policy_required")
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
		self._batches[batch_id] = batch
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
		self._receipts[receipt_id] = receipt
		self._audit(tenant_id, receipt_id, "delivery_receipt_recorded", output.recipient_ref, "allow", metadata={"delivery_state": receipt.delivery_state})
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
		if channel_id not in self._channels:
			self.create_channel(
				channel_id=channel_id,
				tenant_id=tenant_id,
				name=str(metadata.get("channel_name") or "Default output channel"),
				channel_type=str(metadata.get("channel_type") or "email"),
				owner=str(metadata.get("owner") or "system"),
				provider_ref=str(metadata.get("provider_ref") or "provider://local"),
			)
		template_id = str(metadata.get("template_id") or f"template-{record_id}")
		if template_id not in self._templates:
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
		if policy_id not in self._policies:
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
		if route_id not in self._routes:
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
			"unhealthy_channel_count": len([channel for channel in channels if channel["health"] == "unhealthy"]),
			"large_batch_count": len([batch for batch in batches if batch["recipient_count"] > 10000]),
			"failed_receipt_count": len([receipt for receipt in receipts if receipt["delivery_state"] in {"failed", "bounced"}]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_channel(self, channel_id: str, tenant_id: str) -> OutputChannel:
		channel = self._channels.get(channel_id)
		if channel is None or channel.tenant_id != tenant_id:
			raise KeyError("output_channel_not_found")
		return channel

	def _require_template(self, template_id: str, tenant_id: str) -> OutputTemplate:
		template = self._templates.get(template_id)
		if template is None or template.tenant_id != tenant_id:
			raise KeyError("output_template_not_found")
		return template

	def _require_policy(self, policy_id: str, tenant_id: str) -> DeliveryPolicy:
		policy = self._policies.get(policy_id)
		if policy is None or policy.tenant_id != tenant_id:
			raise KeyError("delivery_policy_not_found")
		return policy

	def _require_route(self, route_id: str, tenant_id: str) -> DeliveryRoute:
		route = self._routes.get(route_id)
		if route is None or route.tenant_id != tenant_id:
			raise KeyError("delivery_route_not_found")
		return route

	def _require_rendered_output(self, output_id: str, tenant_id: str) -> RenderedOutput:
		output = self._rendered_outputs.get(output_id)
		if output is None or output.tenant_id != tenant_id:
			raise KeyError("rendered_output_not_found")
		return output

	def _require_batch(self, batch_id: str, tenant_id: str) -> DeliveryBatch:
		batch = self._batches.get(batch_id)
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
