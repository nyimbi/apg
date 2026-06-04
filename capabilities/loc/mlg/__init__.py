"""APG Multi-Language & Localisation capability.

Standalone package: ``pip install apg-loc-mlg``

Quick start::

    from apg_loc_mlg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : loc_mlg
Provides      : locale_configuration, translation_management, rtl_support, date_number_formatting, content_localisation_workflow, locale_registry
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-loc-mlg"
__capability_id__ = "loc_mlg"

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
