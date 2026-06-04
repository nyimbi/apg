"""APG Time Series Analytics capability.

Standalone package: ``pip install apg-bia-tsa``

Quick start::

    from apg_bia_tsa import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_tsa
Provides      : high_frequency_time_series_ingestion, anomaly_detection, seasonality_decomposition, time_series_forecasting, stream_windowing, multi_stream_correlation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-tsa"
__capability_id__ = "bia_tsa"

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
