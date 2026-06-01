"""APG Cross-Border Remittance executable capability."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import RemittanceService

__all__ = ["CAPABILITY_ID", "CAPABILITY_NAME", "CAPABILITY_VERSION", "RemittanceService", "get_capability_contract"]
