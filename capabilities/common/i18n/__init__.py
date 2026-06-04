"""APG Internationalization capability.

Standalone package: ``pip install apg-common-i18n``

Quick start::

    from apg_common_i18n import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : i18n
Provides      : locale_management, translation_memory, content_localization, language_fallbacks, regional_formatting, language_policy
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-i18n"
__capability_id__ = "i18n"

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
