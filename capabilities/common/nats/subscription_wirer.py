"""NATS subscription auto-wirer for APG cross-capability integration.

Reads 'subscribes' declarations from capability_contract.py files at startup
and creates durable JetStream push consumers for each declared subscription.

Usage (from app startup or an ASGI lifespan handler)::

    from capabilities.common.nats.subscription_wirer import SubscriptionWirer

    wirer = SubscriptionWirer()
    await wirer.wire_all()          # reads all contracts, creates subscriptions
    await wirer.wire_capability("fintech_fraud")   # wire one capability only

Declaration format in capability_contract.py::

    subscribes = [
        {
            "source_capability": "fintech_gateway",
            "event_type": "payment_authorized",
            "handler": "on_payment_authorized",   # method name on the service class
            "filter": None,                         # optional dict to filter payload fields
        },
    ]

The wirer resolves the handler by importing the capability's service module and
calling handler(event: dict) for each received message.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from typing import Any, Callable

from .nats_adapter import _get_js
from .subject_registry import subject_for

_log = logging.getLogger(__name__)

# Hardcoded first 5 high-value cross-capability pairs.
# Each entry: (subscriber_capability, source_capability, event_type, handler_module, handler_fn)
# handler_module is the dotted import path; handler_fn is a top-level async callable.
PRIORITY_WIRING: list[tuple[str, str, str, str, str]] = [
    (
        "fintech_fraud",
        "fintech_gateway",
        "payment_authorized",
        "capabilities.fintech.fraud.service",
        "on_payment_authorized",
    ),
    (
        "fintech_aml",
        "fintech_kyc",
        "kyc_cleared",
        "capabilities.fintech.aml.service",
        "on_kyc_cleared",
    ),
    (
        "intel_correlation",
        "intel_alerts",
        "alert_created",
        "capabilities.intel.correlation.service",
        "on_alert_created",
    ),
    (
        "ntfy",
        "government_cas",
        "case_created",
        "capabilities.common.ntfy.service",
        "on_case_created",
    ),
    (
        "auth",
        "mob_mdm",
        "device_enrolled",
        "capabilities.common.auth.service",
        "on_device_enrolled",
    ),
]


class SubscriptionWirer:
    """Reads capability_contract.py 'subscribes' declarations and creates
    NATS JetStream durable push consumers at startup.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Any] = []

    async def wire_all(self) -> dict[str, int]:
        """Wire all priority subscriptions and any contract-declared ones.

        Returns a summary: {capability_id: subscription_count}
        """
        summary: dict[str, int] = {}
        for sub_cap, src_cap, event_type, module_path, handler_fn in PRIORITY_WIRING:
            handler = self._resolve_handler(module_path, handler_fn)
            if handler is None:
                _log.warning(
                    "Skipping wiring %s → %s.%s: handler not found",
                    src_cap, sub_cap, handler_fn,
                )
                continue
            sub = await self._create_subscription(
                subscriber_capability=sub_cap,
                source_capability=src_cap,
                event_type=event_type,
                handler=handler,
            )
            if sub is not None:
                summary[sub_cap] = summary.get(sub_cap, 0) + 1
                self._subscriptions.append(sub)

        _log.info(
            "SubscriptionWirer: wired %d subscriptions across %d capabilities",
            sum(summary.values()), len(summary),
        )
        return summary

    async def wire_capability(self, capability_id: str) -> int:
        """Wire only subscriptions for the named capability."""
        count = 0
        for sub_cap, src_cap, event_type, module_path, handler_fn in PRIORITY_WIRING:
            if sub_cap != capability_id:
                continue
            handler = self._resolve_handler(module_path, handler_fn)
            if handler is None:
                _log.warning("Handler %s.%s not found, skipping", module_path, handler_fn)
                continue
            sub = await self._create_subscription(
                subscriber_capability=sub_cap,
                source_capability=src_cap,
                event_type=event_type,
                handler=handler,
            )
            if sub is not None:
                count += 1
                self._subscriptions.append(sub)
        return count

    async def wire_from_contract(
        self,
        capability_id: str,
        subscribes: list[dict[str, Any]],
        service_instance: Any,
    ) -> int:
        """Wire subscriptions declared in a capability_contract.py subscribes list.

        subscribes format::

            [
                {
                    "source_capability": "fintech_gateway",
                    "event_type": "payment_authorized",
                    "handler": "on_payment_authorized",
                    "filter": {"entity_type": "payment_intent"},  # optional
                },
            ]

        service_instance: the service object that has the handler methods.
        """
        count = 0
        for decl in subscribes:
            src_cap = decl.get("source_capability", "")
            event_type = decl.get("event_type", "")
            handler_name = decl.get("handler", "")
            payload_filter: dict[str, Any] | None = decl.get("filter")

            if not (src_cap and event_type and handler_name):
                _log.warning("Incomplete subscription declaration in %s: %s", capability_id, decl)
                continue

            raw_handler = getattr(service_instance, handler_name, None)
            if raw_handler is None:
                _log.warning(
                    "Handler %s.%s not found on service instance",
                    capability_id, handler_name,
                )
                continue

            handler = self._make_filtered_handler(raw_handler, payload_filter)
            sub = await self._create_subscription(
                subscriber_capability=capability_id,
                source_capability=src_cap,
                event_type=event_type,
                handler=handler,
            )
            if sub is not None:
                count += 1
                self._subscriptions.append(sub)
        return count

    async def _create_subscription(
        self,
        *,
        subscriber_capability: str,
        source_capability: str,
        event_type: str,
        handler: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Create a durable JetStream push consumer."""
        subject = subject_for(source_capability, event_type)
        durable_name = f"{subscriber_capability}-{source_capability}-{event_type}".replace(".", "-")[:250]

        async def _msg_handler(msg: Any) -> None:
            try:
                payload = json.loads(msg.data.decode())
                if asyncio.iscoroutinefunction(handler):
                    await handler(payload)
                else:
                    handler(payload)
                await msg.ack()
                _log.debug(
                    "Handled %s → %s (%s)",
                    subject, subscriber_capability, durable_name,
                )
            except Exception as exc:
                _log.error(
                    "Handler error for %s → %s: %s",
                    subject, subscriber_capability, exc,
                )
                await msg.nak()

        try:
            js = await _get_js()
            sub = await js.subscribe(subject, durable=durable_name, cb=_msg_handler)
            _log.info(
                "Wired: %s.%s → %s (durable=%s)",
                source_capability, event_type, subscriber_capability, durable_name,
            )
            return sub
        except Exception as exc:
            _log.error(
                "Failed to subscribe %s → %s: %s",
                subject, subscriber_capability, exc,
            )
            return None

    @staticmethod
    def _resolve_handler(module_path: str, handler_fn: str) -> Callable | None:
        """Import a handler function from a dotted module path."""
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, handler_fn, None)
            if fn is None:
                _log.debug("Function %s not found in %s", handler_fn, module_path)
            return fn
        except ImportError as exc:
            _log.debug("Cannot import %s: %s", module_path, exc)
            return None

    @staticmethod
    def _make_filtered_handler(
        handler: Callable,
        payload_filter: dict[str, Any] | None,
    ) -> Callable:
        """Wrap a handler to only call it when payload matches all filter key/values."""
        if not payload_filter:
            return handler

        async def _filtered(payload: dict[str, Any]) -> None:
            for k, v in payload_filter.items():
                if payload.get(k) != v:
                    return
            if asyncio.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)

        return _filtered

    async def drain(self) -> None:
        """Unsubscribe all managed subscriptions gracefully."""
        for sub in self._subscriptions:
            try:
                await sub.unsubscribe()
            except Exception:
                pass
        self._subscriptions.clear()


# Module-level singleton for use in app startup
_wirer: SubscriptionWirer | None = None


def get_wirer() -> SubscriptionWirer:
    global _wirer
    if _wirer is None:
        _wirer = SubscriptionWirer()
    return _wirer
