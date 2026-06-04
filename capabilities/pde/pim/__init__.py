"""APG Product Information Management capability.

Standalone package: ``pip install apg-pde-pim``

Quick start::

    from apg_pde_pim import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pde_pim
Provides      : product_catalog_lifecycle, product_record_lifecycle, product_attribute_lifecycle, product_variant_lifecycle, product_content_lifecycle, product_asset_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-pde-pim"
__capability_id__ = "pde_pim"

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
