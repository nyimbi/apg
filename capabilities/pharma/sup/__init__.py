"""APG Pharmaceutical Supply Chain capability.

Standalone package: ``pip install apg-pharma-sup``

Quick start::

    from apg_pharma_sup import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_sup
Provides      : active_ingredient_sourcing_workflow, cmo_management_workflow, demand_planning_workflow, import_licensing_workflow, supply_security_monitoring_workflow, supplier_qualification_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-sup"
__capability_id__ = "pharma_sup"

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
