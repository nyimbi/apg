"""Tenant access regressions for composition event integration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from capabilities.composition.events.apg_integration import (
	APGCapabilityInfo,
	APGEventStreamingIntegration,
	EventRoutingRule,
)


def _integration() -> APGEventStreamingIntegration:
	return APGEventStreamingIntegration(
		event_streaming_service=SimpleNamespace(),
		publishing_service=SimpleNamespace(),
		consumption_service=SimpleNamespace(),
	)


def _capability(**policy) -> APGCapabilityInfo:
	return APGCapabilityInfo(
		capability_id="billing",
		capability_name="Billing",
		capability_type="domain",
		version="1.0.0",
		endpoints={"api": "/billing"},
		event_patterns=["invoice.*"],
		dependencies=[],
		tenant_access_policy=policy,
	)


def test_capability_access_policy_restricts_tenant_streams():
	integration = _integration()
	integration.registered_capabilities["billing"] = _capability(
		visibility="restricted",
		allowed_tenants=["tenant-a"],
	)

	assert integration._capability_accessible_to_tenant(integration.registered_capabilities["billing"], "tenant-a")
	assert not integration._capability_accessible_to_tenant(integration.registered_capabilities["billing"], "tenant-b")

	integration.grant_capability_access("billing", ["tenant-b"])
	assert integration._capability_accessible_to_tenant(integration.registered_capabilities["billing"], "tenant-b")

	integration.revoke_capability_access("billing", ["tenant-b"])
	assert not integration._capability_accessible_to_tenant(integration.registered_capabilities["billing"], "tenant-b")


def test_route_event_skips_targets_blocked_for_event_tenant():
	async def scenario() -> list[str]:
		integration = _integration()
		integration.registered_capabilities["billing"] = _capability(
			visibility="restricted",
			allowed_tenants=["tenant-a"],
		)
		integration.routing_rules["invoice-route"] = EventRoutingRule(
			rule_id="invoice-route",
			source_pattern="orders",
			target_capabilities=["billing"],
			event_type_patterns=["invoice.*"],
		)
		routed_calls: list[str] = []

		async def record_route(event, target_capability, rule):
			routed_calls.append(target_capability)
			return True

		integration._route_event_to_capability = record_route
		event = SimpleNamespace(
			event_id="evt-1",
			event_type="invoice.created",
			source_capability="orders",
			tenant_id="tenant-b",
		)

		routed_to = await integration.route_event(event)
		assert routed_calls == []
		return routed_to

	assert asyncio.run(scenario()) == []


def test_composition_events_no_longer_grants_all_tenant_access_by_default():
	source = (
		__import__("pathlib").Path(__file__).resolve().parents[1]
		/ "capabilities"
		/ "composition"
		/ "events"
		/ "apg_integration.py"
	).read_text(encoding="utf-8")

	assert "return True  # For now, all capabilities are accessible to all tenants" not in source
	assert "tenant_access_policy: Dict[str, Any] = field(default_factory=dict)" in source
	assert "def grant_capability_access(self, capability_id: str, tenant_ids: List[str]) -> bool:" in source
	assert "if event_tenant_id and not self._capability_accessible_to_tenant(capability, event_tenant_id):" in source
