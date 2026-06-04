"""APG Mine Production Operations capability.

Standalone package: ``pip install apg-mining-pro``

Quick start::

    from apg_mining_pro import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mining_pro
Provides      : shift_report_workflow, production_ledger_management, blast_design_workflow, blast_firing_authorization, ore_tracking_management, grade_control_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mining-pro"
__capability_id__ = "mining_pro"

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
