"""APG FinTech Risk Management capability package."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import RiskManagementService

__all__ = ["RiskManagementService", "CAPABILITY_ID", "CAPABILITY_NAME", "CAPABILITY_VERSION", "get_capability_contract"]
