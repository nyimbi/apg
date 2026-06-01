"""APG Mobile Banking capability package."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import MobileBankingService

__all__ = ["MobileBankingService", "CAPABILITY_ID", "CAPABILITY_NAME", "CAPABILITY_VERSION", "get_capability_contract"]
