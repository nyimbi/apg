"""APG Mobile Device Management capability.

Standalone package: ``pip install apg-mob-mdm``

Quick start::

    from apg_mob_mdm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mob_mdm
Provides      : device_enrolment_workflow, mdm_policy_enforcement, compliance_monitoring, remote_wipe_workflow, app_distribution_workflow, mdm_profile_deployment
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mob-mdm"
__capability_id__ = "mob_mdm"

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
