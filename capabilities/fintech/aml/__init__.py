"""APG Anti Money Laundering capability.

Standalone package: ``pip install apg-fintech-aml``

Quick start::

    from apg_fintech_aml import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_aml
Provides      : transaction_monitoring, aml_alert_triage, sanctions_pep_escalation, suspicious_activity_case_management, sar_workflow, typology_rule_engine
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-aml"
__capability_id__ = "fintech_aml"

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
