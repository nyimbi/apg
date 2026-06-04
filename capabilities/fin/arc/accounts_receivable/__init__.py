"""APG arc_accounts_receivable capability.

Standalone package: ``pip install apg-fin-accounts_receivable``

Quick start::

    from apg_fin_accounts_receivable import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : arc_accounts_receivable
Provides      : customer_receivable_lifecycle, credit_assessment_workflow, invoice_lifecycle, invoice_line_management, payment_receipt_lifecycle, cash_application_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-fin-accounts_receivable"
__capability_id__ = "arc_accounts_receivable"

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
