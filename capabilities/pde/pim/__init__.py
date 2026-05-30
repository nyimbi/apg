"""Product Information Management APG capability packet."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, evaluate_capability_rules, get_capability_contract
from .service import PIMService, PLMProductService, ProductInformationLifecycleService, ProductInformationService


__version__ = "2.1.0"
__capability_code__ = "PDE_PIM"
__capability_name__ = "Product Information Management"


__all__ = ["CAPABILITY_ID", "PIMService", "PLMProductService", "ProductInformationLifecycleService", "ProductInformationService", "evaluate_capability_rules", "get_capability_contract"]
