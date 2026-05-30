"""Sustainability and ESG APG capability packet."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, evaluate_capability_rules, get_capability_contract
from .service import ESGManagementLifecycleService, ESGManagementService, ESGReportingService, ESGRiskService, ESGService


__version__ = "2.1.0"
__capability_code__ = "ECD_ESG"
__capability_name__ = "Sustainability and ESG Management"


__all__ = ["CAPABILITY_ID", "ESGManagementLifecycleService", "ESGManagementService", "ESGReportingService", "ESGRiskService", "ESGService", "evaluate_capability_rules", "get_capability_contract"]
