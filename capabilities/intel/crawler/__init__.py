"""APG Intelligence Crawler capability.

Standalone package: ``pip install apg-intel-crawler``

Quick start::

    from apg_intel_crawler import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_crawler
Provides      : source_intelligence_registry, crawl_job_lifecycle, extraction_pipeline, dataset_quality_control, validation_workflow, rag_graphrag_preparation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-intel-crawler"
__capability_id__ = "intel_crawler"

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
