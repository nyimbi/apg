"""APG Embedded Finance capability package."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import EmbeddedFinanceService

__all__ = ["EmbeddedFinanceService", "CAPABILITY_ID", "CAPABILITY_NAME", "CAPABILITY_VERSION", "get_capability_contract"]
