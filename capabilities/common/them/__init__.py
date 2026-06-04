"""APG UI/UX Theming and Branding capability.

Standalone package: ``pip install apg-common-them``

Quick start::

    from apg_common_them import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : them
Provides      : theme_tokens, brand_governance, asset_libraries, preview_workflows, theme_publication_governance, visual_theming
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-them"
__capability_id__ = "them"

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
