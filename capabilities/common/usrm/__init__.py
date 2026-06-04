"""APG User Management capability.

Standalone package: ``pip install apg-common-usrm``

Quick start::

    from apg_common_usrm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : usrm
Provides      : user_directory, profile_management, consented_invitations, role_assignment_governance, access_review_workflows, deprovisioning_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-usrm"
__capability_id__ = "usrm"

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
