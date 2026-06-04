"""APG Mobile App Platform capability.

Standalone package: ``pip install apg-mob-map``

Quick start::

    from apg_mob_map import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mob_map
Provides      : mobile_app_registry, cross_platform_build_workflow, offline_sync_workflow, push_notification_dispatch, biometric_auth_enrollment, app_version_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mob-map"
__capability_id__ = "mob_map"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]
