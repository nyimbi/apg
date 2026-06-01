"""APG Fraud Detection executable capability."""

from .capability_contract import CAPABILITY_ID, CAPABILITY_NAME, CAPABILITY_VERSION, get_capability_contract
from .service import FraudDetectionService

__all__ = ["CAPABILITY_ID", "CAPABILITY_NAME", "CAPABILITY_VERSION", "FraudDetectionService", "get_capability_contract"]
