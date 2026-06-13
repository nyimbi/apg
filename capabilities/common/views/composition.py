"""ComposedView — cross-capability parallel data assembly.

Assembles a screen model from multiple capabilities using asyncio.gather()
(not sequential calls which multiply latency).

The canonical_id is resolved via MDM before fan-out so every capability
receives the same stable UUID rather than a capability-local ID.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

from capabilities.common.reliability import safe_gather

_log = logging.getLogger(__name__)


@dataclass
class ComposedSource:
    """One data source in a ComposedView fan-out.

    capability_id: registered APG capability, e.g. "fintech_fraud"
    method:        method name on the capability's service class
    kwargs:        extra kwargs to pass (canonical_entity_id + tenant_id added automatically)
    label:         key in the composed result dict (defaults to capability_id)
    required:      if False, a failure returns None rather than propagating
    """
    capability_id: str
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.capability_id


class ComposedView:
    """Base class for cross-capability screen models.

    Subclass and implement ``build()`` to define which capabilities to fan out
    to. Each source is called in parallel via safe_gather(); failures in
    non-required sources return None instead of propagating.

    Example::

        class TransactionDetailView(ComposedView):
            async def build(self, tx_id: str, tenant_id: str) -> dict:
                return await self.compose(
                    tenant_id=tenant_id,
                    sources=[
                        ComposedSource("fintech_gateway", "get_payment",
                                       {"payment_id": tx_id}, label="payment"),
                        ComposedSource("fintech_fraud", "get_risk_assessment",
                                       {"transaction_id": tx_id}, label="fraud"),
                        ComposedSource("fintech_aml",   "get_flags",
                                       {"entity_id": tx_id},     label="aml"),
                    ],
                )
    """

    def __init__(self, tenant_id: str | None = None) -> None:
        self._tenant_id = tenant_id
        self._service_cache: dict[str, Any] = {}

    async def compose(
        self,
        tenant_id: str,
        sources: list[ComposedSource | tuple],
        canonical_entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Fan out to all sources in parallel; return merged result dict.

        Args:
            tenant_id: tenant scope
            sources: list of ComposedSource (or 3-tuple shorthand)
            canonical_entity_id: MDM canonical UUID injected into every call

        Returns:
            dict keyed by source.label with each source's result (or None on failure)
        """
        normalized = [self._normalize_source(s) for s in sources]

        async def _call_one(src: ComposedSource) -> tuple[str, Any]:
            try:
                svc = await self._get_service(src.capability_id, tenant_id)
                if svc is None:
                    _log.debug("No service for %s", src.capability_id)
                    return src.label, None
                method = getattr(svc, src.method, None)
                if method is None:
                    _log.debug("Method %s.%s not found", src.capability_id, src.method)
                    return src.label, None
                kwargs = dict(src.kwargs)
                kwargs["tenant_id"] = tenant_id
                if canonical_entity_id:
                    kwargs.setdefault("canonical_entity_id", canonical_entity_id)
                if asyncio.iscoroutinefunction(method):
                    result = await method(**kwargs)
                else:
                    result = method(**kwargs)
                return src.label, result
            except Exception as exc:
                if src.required:
                    raise
                _log.warning("ComposedView: %s.%s failed (non-required): %s", src.capability_id, src.method, exc)
                return src.label, None

        pairs = await safe_gather(
            *[_call_one(src) for src in normalized],
            label="composed_view",
            suppress_exceptions=False,
        )
        return {label: value for label, value in (pairs or []) if label is not None}

    @staticmethod
    def _normalize_source(src: ComposedSource | tuple) -> ComposedSource:
        """Accept either a ComposedSource or a (capability_id, method, kwargs) tuple."""
        if isinstance(src, ComposedSource):
            return src
        if isinstance(src, tuple):
            cap_id, method, *rest = src
            kwargs = rest[0] if rest else {}
            return ComposedSource(capability_id=cap_id, method=method, kwargs=kwargs)
        raise TypeError(f"Expected ComposedSource or tuple, got {type(src)}")

    async def _get_service(self, capability_id: str, tenant_id: str) -> Any:
        """Lazily import and instantiate a capability's service class."""
        if capability_id in self._service_cache:
            return self._service_cache[capability_id]

        # Attempt to import capabilities.{domain}.{cap}.service
        # capability_id format: "fintech_fraud" → capabilities/fintech/fraud/service.py
        parts = capability_id.split("_", 1)
        module_paths = []
        if len(parts) == 2:
            module_paths.append(f"capabilities.{parts[0]}.{parts[1]}.service")
        # Also try flattened: capabilities.fintech_fraud.service
        module_paths.append(f"capabilities.{capability_id}.service")
        # Common capabilities: capabilities.common.{cap}.service
        module_paths.append(f"capabilities.common.{capability_id}.service")

        for path in module_paths:
            try:
                mod = importlib.import_module(path)
                # Find a service class: look for *Service class or get_service() factory
                svc = None
                if hasattr(mod, "get_service"):
                    svc = mod.get_service(tenant_id=tenant_id)
                else:
                    for attr_name in dir(mod):
                        if attr_name.endswith("Service") and not attr_name.startswith("_"):
                            cls = getattr(mod, attr_name)
                            if isinstance(cls, type):
                                try:
                                    svc = cls(tenant_id=tenant_id)
                                except TypeError:
                                    svc = cls()
                                break
                if svc is not None:
                    self._service_cache[capability_id] = svc
                    return svc
            except ImportError:
                continue
            except Exception as exc:
                _log.debug("Cannot instantiate service for %s via %s: %s", capability_id, path, exc)
                continue

        _log.debug("No service found for capability_id=%s", capability_id)
        return None
