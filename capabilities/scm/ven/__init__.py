"""SCM Vendor Management APG capability packet."""

from __future__ import annotations

from .capability_contract import CAPABILITY_ID, evaluate_capability_rules, get_capability_contract
from .service import (
	VendorLifecycleService,
	VendorManagementLifecycleService,
	VendorManagementService,
	VendorPerformanceService,
	VendorRiskService,
)


__version__ = "2.1.0"
__capability_code__ = "SCM_VEN"
__capability_name__ = "Vendor Management"


__all__ = [
	"CAPABILITY_ID",
	"VendorLifecycleService",
	"VendorManagementLifecycleService",
	"VendorManagementService",
	"VendorPerformanceService",
	"VendorRiskService",
	"evaluate_capability_rules",
	"get_capability_contract",
]
