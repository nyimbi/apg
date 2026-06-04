"""APG glr_general_ledger capability.

Standalone package: ``pip install apg-fin-general_ledger``

Quick start::

    from apg_fin_general_ledger import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : glr_general_ledger
Provides      : chart_of_accounts_lifecycle, ledger_dimension_management, accounting_period_lifecycle, journal_batch_lifecycle, journal_entry_lifecycle, journal_posting_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-fin-general_ledger"
__capability_id__ = "glr_general_ledger"

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
