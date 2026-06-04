"""APG Document Management capability.

Standalone package: ``pip install apg-grc-doc``

Quick start::

    from apg_grc_doc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : grc_doc
Provides      : document_repository_lifecycle, document_template_lifecycle, document_revision_workflow, document_approval_workflow, document_publication_workflow, document_retention_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-grc-doc"
__capability_id__ = "grc_doc"

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
