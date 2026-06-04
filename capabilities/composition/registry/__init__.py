"""APG Capability Registry capability.

Standalone package: ``pip install apg-composition-registry``

Quick start::

    from apg_composition_registry import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_registry
Provides      : capability_catalog_lifecycle, dependency_graph_management, composition_blueprint_validation, version_compatibility_governance, marketplace_publication_governance, registry_discovery
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-registry"
__capability_id__ = "composition_registry"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)
from .models import CRCapability, CRRegistry  # noqa: E402
from .service import CapabilityRegistryService, CRService  # noqa: E402


class MobileOfflineService:
    """Stub mobile-offline composition surface for registry consumers."""


__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
    "CRCapability",
    "CRRegistry",
    "CRService",
    "CapabilityRegistryService",
    "MobileOfflineService",
]
