"""APG Terminal Management System capability.

Standalone package: ``pip install apg-fintech-terminal``

Quick start::

    from apg_fintech_terminal import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_terminal
Provides      : terminal_lifecycle_management, terminal_key_injection_workflow, terminal_parameter_deployment, terminal_certificate_management, terminal_health_monitoring, pci_dss_compliance_tracking
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-terminal"
__capability_id__ = "fintech_terminal"

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
