"""APG Accounts Payable capability.

Standalone package: ``pip install apg-fin-accounts_payable``

Quick start::

    from apg_fin_accounts_payable import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : apy_accounts_payable
Provides      : vendor_payables_lifecycle, invoice_capture_and_matching, approval_workflow, payment_run_lifecycle, expense_reimbursement_lifecycle, ap_aging_and_close
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-fin-accounts_payable"
__capability_id__ = "apy_accounts_payable"

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
