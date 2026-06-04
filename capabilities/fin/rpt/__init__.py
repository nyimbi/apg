"""APG Financial Reporting capability.

Standalone package: ``pip install apg-fin-rpt``

Quick start::

    from apg_fin_rpt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fin_rpt
Provides      : financial_report_template_lifecycle, report_line_mapping, reporting_period_lifecycle, financial_statement_generation, statement_publication_workflow, financial_consolidation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-fin-rpt"
__capability_id__ = "fin_rpt"

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
